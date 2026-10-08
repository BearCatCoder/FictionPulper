"""Rigorous post-selection evaluation for the completed Contrastive-v1 run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

import torch
from tokenizers import Tokenizer

from src.contrastive_pairs import load_pairs
from src.narrative_post_selection import (
    PROMPTS,
    evaluate_packed,
    generation_suite,
    load_json,
    load_model,
    repetition_aggregate,
    require,
    verify_file,
)
from src.train import GENERATION_SETTINGS, benchmark_excluded_ids
from src.train_contrastive import pair_ranking_loss
from src.train_tokenizer import sha256_file, write_json_atomic


PROTOCOL_PATH = Path("experiments/fictionpulper-15m-data30m-contrastive-v1/post-selection-protocol.json")
RUN_DIR = Path("runs/fictionpulper-15m-data30m-contrastive-v1/post-selection")
EXPECTED_BENCHMARK_PROTOCOL_SHA256 = "16b8f2df65e9be1df2923b9a93e9570258b8f2e16b156bbd3ebd27697659c60a"
EXPECTED_CHECKPOINT_SHA256 = "5dc3bd250432334c748d125d79eeb502fb4eb7cfe4363b69a0482526de3714b4"
EXPECTED_TRAINING_COMMIT = "dcf7c8e03702740ea0f839c1e3d17ecf92f0496b"
HISTORICAL_BENCHMARKS = (
    "data10m_test_data30m_train_disjoint",
    "legacy_seed_validation_leakage_clean",
    "legacy_seed_test",
)


def verify_protocol(protocol: dict[str, Any]) -> None:
    require(protocol["selection_metric"] == "Data30M validation loss only", "Selection metric drifted")
    require(protocol["candidate"]["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, "Candidate checkpoint drifted")
    require(protocol["training_git_commit"] == EXPECTED_TRAINING_COMMIT, "Training commit drifted")
    benchmark = protocol["benchmark_protocol"]
    require(benchmark["sha256"] == EXPECTED_BENCHMARK_PROTOCOL_SHA256, "Benchmark protocol hash is not fixed")
    require(tuple(benchmark["historical_in_order"]) == HISTORICAL_BENCHMARKS, "Historical benchmark set or order drifted")
    require(protocol["generation"]["settings"] == GENERATION_SETTINGS, "Generation settings drifted")
    require(protocol["generation"]["sample_seed"] == 11337, "Generation seed drifted")
    require(len(PROMPTS) == 10, "Historical suite is not exactly 10 prompts")
    require(protocol["narrative_protocol"]["fact_count"] == 13, "Narrative fact count drifted")
    require(protocol["narrative_protocol"]["prompt_count"] == 3, "Narrative prompt count drifted")
    # Protocol files and sealed records are metadata, not held-out pair/data loaders.
    for spec in (benchmark, protocol["narrative_protocol"], *protocol["sealed_sources"].values()):
        verify_file(Path(spec["path"]), spec["sha256"], "Locked protocol/sealed source")


def verify_selection_gate(
    protocol: dict[str, Any], *, load_checkpoint: Callable[[Path], dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Prove selection was final and holdouts untouched before opening any holdout."""
    candidate = protocol["candidate"]
    summary = load_json(Path(candidate["summary_path"]))
    manifest = load_json(Path(candidate["manifest_path"]))
    require(summary.get("status") == protocol["required_training_status"], "Training summary is incomplete")
    require(manifest.get("status") == protocol["required_training_status"], "Training manifest is incomplete")
    require(manifest.get("experiment_id") == protocol["experiment_id"], "Training experiment ID differs")
    require(summary.get("selection_metric") == protocol["selection_metric"], "Summary selection metric drifted")
    require(summary.get("curriculum_validation_used_for_selection") is False, "Curriculum validation affected selection")
    require(summary.get("held_out_access_during_training") is False, "Summary records held-out access during training")
    require(manifest.get("held_out_access_during_training") is False, "Manifest records held-out access during training")
    require(manifest.get("training_git_commit") == EXPECTED_TRAINING_COMMIT, "Training commit differs")
    require(manifest.get("completed_at") == summary.get("completed_at"), "Completion records differ")
    candidates = summary.get("selection_candidates", [])
    require(bool(candidates), "No Data30M selection candidates")
    selected = min(candidates, key=lambda item: item["data30m_validation_loss"])
    require(selected["step"] == summary.get("best_checkpoint_step") == candidate["expected_step"], "Checkpoint was not selected solely by Data30M validation")
    require(summary.get("best_data30m_validation_loss") == selected["data30m_validation_loss"], "Best validation loss differs")
    require(summary.get("best_checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256, "Summary checkpoint hash differs")
    require(manifest.get("best_checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256, "Manifest checkpoint hash differs")
    checkpoint_path = Path(candidate["checkpoint_path"])
    verify_file(checkpoint_path, EXPECTED_CHECKPOINT_SHA256, "Selected checkpoint")
    checkpoint = load_checkpoint(checkpoint_path)
    require(checkpoint.get("step") == candidate["expected_step"], "Checkpoint step differs")
    require(checkpoint.get("selection_metric") == protocol["selection_metric"], "Checkpoint selection metric differs")
    require(checkpoint.get("tokenizer_hash") == candidate["tokenizer_sha256"], "Checkpoint tokenizer differs")
    require(checkpoint.get("schedule_sha256") == summary.get("schedule_content_sha256"), "Checkpoint schedule differs")
    return summary, checkpoint, checkpoint_path


def run_after_gate(
    gate: Callable[[], tuple[dict[str, Any], dict[str, Any], Path]],
    heldout_loader: Callable[[dict[str, Any], dict[str, Any], Path], dict[str, Any]],
) -> dict[str, Any]:
    summary, checkpoint, checkpoint_path = gate()
    return heldout_loader(summary, checkpoint, checkpoint_path)


def _group_result(items: Iterable[dict[str, Any]]) -> dict[str, int | float]:
    values = list(items)
    return {
        "count": len(values),
        "pairwise_accuracy": mean(item["correct"] for item in values) if values else 0.0,
        "mean_margin": mean(item["margin"] for item in values) if values else 0.0,
    }


def aggregate_pair_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"aggregate": _group_result(results)}
    for field in ("state_type", "distance_bucket", "difficulty"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            groups[str(result[field])].append(result)
        output[f"by_{field}"] = {name: _group_result(groups[name]) for name in sorted(groups)}
    return output


def load_pair_split(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    verify_file(Path(spec["manifest_path"]), spec["manifest_sha256"], "Pair manifest")
    verify_file(Path(spec["path"]), spec["sha256"], "Pair file")
    verify_file(Path(spec["records_path"]), spec["records_sha256"], "Curriculum records")
    pairs = load_pairs(Path(spec["path"]), max_seq_len=1024)
    records = {item["id"]: item for item in _load_jsonl(Path(spec["records_path"]))}
    require(len(pairs) == spec["expected_pairs"], "Pair count changed")
    require(all(pair["document_id"] in records for pair in pairs), "Pair document_id is absent from curriculum records")
    return pairs, records


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


@torch.no_grad()
def evaluate_pairs(
    model: Any, pairs: list[dict[str, Any]], records: dict[str, dict[str, Any]], device: torch.device
) -> dict[str, Any]:
    results = []
    model.eval()
    for pair in pairs:
        record = records[pair["document_id"]]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, scored = pair_ranking_loss(model, pair, device)
        results.append({
            "pair_id": pair["pair_id"],
            "document_id": pair["document_id"],
            "state_type": pair["state_type"],
            "distance_bucket": record["distance_bucket"],
            "difficulty": record["difficulty"],
            "positive_score": scored["positive_score"],
            "negative_score": scored["negative_score"],
            "margin": scored["margin"],
            "correct": scored["ranking_accuracy"],
        })
    return {"score": "mean_log_probability_decision_span", "trials": results, **aggregate_pair_results(results)}


def _sealed_metrics(protocol: dict[str, Any]) -> dict[str, Any]:
    compute = load_json(Path(protocol["sealed_sources"]["compute5_metrics"]["path"]))
    narrative = load_json(Path(protocol["sealed_sources"]["narrative_metrics"]["path"]))
    names = ("data30m_test", *HISTORICAL_BENCHMARKS)
    compute_values = {"data30m_test": compute["primary_data30m_test"]}
    compute_values.update({name: compute["historical_post_selection_evaluations"][name] for name in HISTORICAL_BENCHMARKS})
    return {
        "compute5": {name: compute_values[name] for name in names},
        "narrative_v1": {name: narrative["metrics"][name] for name in names},
    }


def _historical_generation_models(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    historical = load_json(Path(protocol["sealed_sources"]["historical_generations"]["path"]))
    narrative = load_json(Path(protocol["sealed_sources"]["narrative_generations"]["path"]))
    require(historical["prompts"] == PROMPTS, "Sealed historical prompts drifted")
    require(historical["generation_settings"] == GENERATION_SETTINGS, "Sealed historical settings drifted")
    return historical["models"], narrative["models"]


def _delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        mode + "/" + key: candidate[mode][key] - baseline[mode][key]
        for mode in ("greedy", "sampled")
        for key in candidate[mode]
    }


def write_outputs(payloads: dict[str, Any], run_dir: Path, protocol_path: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, payload in payloads.items():
        path = run_dir / name
        write_json_atomic(path, payload)
        artifacts[name] = {"path": str(path), "sha256": sha256_file(path)}
    manifest = {
        "experiment_id": "fictionpulper-15m-data30m-contrastive-v1",
        "post_selection_protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "artifacts": artifacts,
    }
    write_json_atomic(run_dir / "artifact-manifest.json", manifest)
    return manifest


def evaluate_heldout(
    summary: dict[str, Any], checkpoint: dict[str, Any], checkpoint_path: Path,
    protocol: dict[str, Any], run_dir: Path, protocol_path: Path,
) -> dict[str, Any]:
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "Evaluation requires CUDA BF16")
    device = torch.device("cuda")
    tokenizer_path = Path(protocol["candidate"]["tokenizer_path"])
    verify_file(tokenizer_path, protocol["candidate"]["tokenizer_sha256"], "Tokenizer")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    pair_splits = {name: load_pair_split(spec) for name, spec in protocol["pairs"].items()}
    models = {"contrastive_v1": load_model(checkpoint, device)}
    for name, spec in protocol["models"].items():
        verify_file(Path(spec["checkpoint_path"]), spec["checkpoint_sha256"], f"{name} checkpoint")
        models[name] = load_model(torch.load(spec["checkpoint_path"], map_location=device, weights_only=False), device)
    pair_results = {
        split: {name: evaluate_pairs(model, pairs, records, device) for name, model in models.items()}
        for split, (pairs, records) in pair_splits.items()
    }

    contrastive_model = models["contrastive_v1"]
    lm_metrics = _sealed_metrics(protocol)
    lm_metrics["contrastive_v1"] = {
        "data30m_test": evaluate_packed(contrastive_model, tokenizer, protocol["data30m_test"], device)
    }
    benchmark_protocol = load_json(Path(protocol["benchmark_protocol"]["path"]))
    verify_file(
        Path(benchmark_protocol["data30m_split_path"]),
        benchmark_protocol["data30m_split_sha256"],
        "Data30M split manifest",
    )
    split_manifest = load_json(Path(benchmark_protocol["data30m_split_path"]))
    benchmark_by_name = {item["name"]: item for item in benchmark_protocol["benchmarks"]}
    for name in HISTORICAL_BENCHMARKS:
        spec = benchmark_by_name[name]
        verify_file(Path(spec["index_path"]), spec["index_sha256"], f"{name} benchmark index")
        exclusions = benchmark_excluded_ids(spec, split_manifest["assignments"])
        lm_metrics["contrastive_v1"][name] = evaluate_packed(
            contrastive_model, tokenizer, spec, device, excluded_ids=exclusions
        )

    historical_models, narrative_models = _historical_generation_models(protocol)
    seed = protocol["generation"]["sample_seed"]
    contrastive_historical = generation_suite(contrastive_model, tokenizer, device, PROMPTS, seed)
    historical_models = {**historical_models, "contrastive_v1": contrastive_historical}
    narrative_protocol = load_json(Path(protocol["narrative_protocol"]["path"]))
    prompts = [entry["prompt"] for entry in narrative_protocol["entries"]]
    contrastive_narrative = generation_suite(contrastive_model, tokenizer, device, prompts, seed)
    for output, definition in zip(contrastive_narrative, narrative_protocol["entries"], strict=True):
        output.update({key: definition[key] for key in ("id", "facts", "prompt_token_count", "fact_prefix_token_count", "tokens_after_fact_prefix")})
    narrative_models = {**narrative_models, "contrastive_v1_context1024": {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256, "entries": contrastive_narrative,
    }}
    generation_audit = {
        "post_checkpoint_selection": True,
        "used_for_checkpoint_selection": False,
        "semantic_scoring": "not_performed",
        "semantic_scoring_note": "Facts and generations are disclosed for human audit; no automatic semantic fact score is claimed.",
        "protocol_path": protocol["narrative_protocol"]["path"],
        "protocol_sha256": protocol["narrative_protocol"]["sha256"],
        "sample_seed": seed,
        "generation_settings": GENERATION_SETTINGS,
        "models": narrative_models,
    }
    historical_repetition = {name: repetition_aggregate(entries) for name, entries in historical_models.items()}
    narrative_repetition = {name: repetition_aggregate(value["entries"]) for name, value in narrative_models.items()}
    repetition = {
        "definition": "Tokenizer-v1 metrics over generated continuations only; prompts excluded",
        "historical_10_prompt": historical_repetition,
        "narrative_state_3_prompt": narrative_repetition,
        "controlled_deltas_vs_compute5": {
            "historical_10_prompt": _delta(historical_repetition["contrastive_v1"], historical_repetition["compute5"]),
            "narrative_state_3_prompt": _delta(narrative_repetition["contrastive_v1_context1024"], narrative_repetition["compute5_context1024"]),
        },
        "controlled_deltas_vs_narrative_v1": {
            "historical_10_prompt": _delta(historical_repetition["contrastive_v1"], historical_repetition["narrative"]),
            "narrative_state_3_prompt": _delta(narrative_repetition["contrastive_v1_context1024"], narrative_repetition["narrative_context1024"]),
        },
    }
    return write_outputs({
        "pair-evaluation.json": {"checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256, "splits": pair_results},
        "lm-evaluation.json": {"checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256, "models": lm_metrics},
        "generation-comparison.json": {"prompts": PROMPTS, "sample_seed": seed, "generation_settings": GENERATION_SETTINGS, "models": historical_models},
        "narrative-fact-audit.json": generation_audit,
        "repetition-comparison.json": repetition,
        "selection-record.json": {
            "status": "post_selection_evaluation_complete", "selection_metric": protocol["selection_metric"],
            "used_holdouts_for_checkpoint_selection": False, "training_git_commit": EXPECTED_TRAINING_COMMIT,
            "checkpoint_step": summary["best_checkpoint_step"], "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        },
    }, run_dir, protocol_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    protocol = load_json(args.protocol)
    verify_protocol(protocol)
    gate = lambda: verify_selection_gate(
        protocol, load_checkpoint=lambda path: torch.load(path, map_location="cpu", weights_only=False)
    )
    result = run_after_gate(
        gate, lambda summary, checkpoint, path: evaluate_heldout(
            summary, checkpoint, path, protocol, args.run_dir, args.protocol
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
