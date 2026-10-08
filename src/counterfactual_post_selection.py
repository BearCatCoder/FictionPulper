"""Strict post-selection evaluation for Counterfactual-v1."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import torch
import yaml
from tokenizers import Tokenizer

from src.counterfactual_narrative.evaluator import (
    aggregate_trials,
    evaluate_model_split,
    load_pairs,
)
from src.frozen_state_probe import run as run_frozen_probe
from src.mixed_schedule import verify_schedule
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
from src.narrative_state_localization import (
    encode_fixed_candidates,
    rank_scores,
    score_candidate_ids,
    validate_forced_choice_protocol,
)
from src.train import GENERATION_SETTINGS, benchmark_excluded_ids
from src.train_tokenizer import sha256_file, write_json_atomic


PROTOCOL_PATH = Path("experiments/fictionpulper-15m-data30m-counterfactual-v1/post-selection-protocol.json")
RUN_DIR = Path("runs/fictionpulper-15m-data30m-counterfactual-v1/post-selection")
EXPERIMENT_DIR = Path("experiments/fictionpulper-15m-data30m-counterfactual-v1")
EXPECTED_PROTOCOL_SHA256 = "b1ba01b7bc09d373a28433300540e2b6c9f9799c9a1a9121d0536307a9f21aab"
HISTORICAL_BENCHMARKS = (
    "data10m_test_data30m_train_disjoint",
    "legacy_seed_validation_leakage_clean",
    "legacy_seed_test",
)


def verify_protocol(protocol: dict[str, Any], protocol_path: Path = PROTOCOL_PATH) -> None:
    require(sha256_file(protocol_path) == EXPECTED_PROTOCOL_SHA256, "Post-selection protocol changed")
    require(protocol["version"] == 1, "Unsupported post-selection protocol")
    require(protocol["selection_metric"] == "Data30M validation loss only", "Selection metric changed")
    require(protocol["generation"]["settings"] == GENERATION_SETTINGS, "Generation settings changed")
    require(protocol["generation"]["sample_seed"] == 11337, "Generation seed changed")
    require(tuple(protocol["counterfactual"]["splits"]) == ("test", "generalization_holdout"), "Held-out split set changed")
    for section, path_key, hash_key in (
        ("counterfactual", "evaluator_path", "evaluator_sha256"),
        ("probe", "method_path", "method_sha256"),
        ("probe", "config_path", "config_sha256"),
    ):
        spec = protocol[section]
        verify_file(Path(spec[path_key]), spec[hash_key], f"Locked {section} input")


def _git_blob_hash(commit: str, path: str) -> str:
    content = subprocess.run(
        ["git", "show", f"{commit}:{path}"], check=True, capture_output=True
    ).stdout
    import hashlib
    return hashlib.sha256(content).hexdigest()


def verify_selection_gate(
    protocol: dict[str, Any], *, load_checkpoint: Callable[[Path], dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    """Prove final selection and training locks before any held-out loader runs."""
    candidate = protocol["candidate"]
    for key, label in (("summary", "Training summary"), ("manifest", "Training manifest")):
        verify_file(Path(candidate[f"{key}_path"]), candidate[f"{key}_sha256"], label)
    summary = load_json(Path(candidate["summary_path"]))
    manifest = load_json(Path(candidate["manifest_path"]))
    require(summary.get("status") == protocol["required_training_status"], "Training summary is incomplete")
    require(summary.get("optimizer_steps") == candidate["expected_step"], "Training step count is incomplete")
    require(summary.get("selection_metric") == protocol["selection_metric"], "Summary selection metric changed")
    require(summary.get("held_out_access_during_training") is False, "Summary records held-out access")
    require(manifest.get("held_out_access_during_training") is False, "Manifest records held-out access")
    require(manifest.get("training_git_commit") == protocol["training_commit"], "Training commit changed")
    require(manifest.get("fresh_random_initialization") is True and manifest.get("prior_checkpoint_loaded") is False, "Training initialization changed")

    eligible = [item for item in summary.get("metrics", []) if item.get("selection_eligible")]
    require(bool(eligible), "No Data30M validation candidates were recorded")
    selected = min(eligible, key=lambda item: item["data30m_validation"]["loss"])
    require(selected["step"] == summary["best_checkpoint_step"] == candidate["expected_step"], "Selected checkpoint is not the minimum validation candidate")
    require(selected["data30m_validation"]["loss"] == summary["best_data30m_validation_loss"], "Selected validation loss changed")

    locks = protocol["training_locks"]
    for path_key, hash_key in (
        ("config_path", "config_sha256"), ("tokenizer_path", "tokenizer_sha256"),
        ("counterfactual_manifest_path", "counterfactual_manifest_sha256"),
        ("packed_metadata_path", "packed_metadata_sha256"), ("schedule_path", "schedule_file_sha256"),
    ):
        verify_file(Path(locks[path_key]), locks[hash_key], f"Training lock {path_key}")
    require(manifest["config_sha256"] == locks["config_sha256"], "Manifest config hash changed")
    require(_git_blob_hash(protocol["training_commit"], locks["config_path"]) == locks["config_sha256"], "Training commit does not contain the locked config")
    require(manifest["tokenizer_sha256"] == locks["tokenizer_sha256"], "Manifest tokenizer hash changed")
    require(manifest["dataset_gates"]["manifest_sha256"] == locks["counterfactual_manifest_sha256"], "Manifest dataset lock changed")
    require(manifest["packed_metadata_sha256"] == locks["packed_metadata_sha256"], "Manifest packed-data lock changed")
    require(manifest["schedule_file_sha256"] == locks["schedule_file_sha256"], "Manifest schedule-file lock changed")
    schedule = load_json(Path(locks["schedule_path"]))
    require(verify_schedule(schedule) == locks["schedule_content_sha256"] == manifest["schedule_content_sha256"], "Schedule content lock changed")

    checkpoint_path = Path(candidate["checkpoint_path"])
    checkpoint_hash = verify_file(checkpoint_path, candidate["checkpoint_sha256"], "Selected checkpoint")
    require(summary["best_checkpoint_sha256"] == checkpoint_hash, "Summary checkpoint hash changed")
    checkpoint = load_checkpoint(checkpoint_path)
    require(checkpoint.get("step") == candidate["expected_step"], "Checkpoint step changed")
    require(checkpoint.get("selection_metric") == protocol["selection_metric"], "Checkpoint selection metric changed")
    require(checkpoint.get("tokenizer_hash") == locks["tokenizer_sha256"], "Checkpoint tokenizer changed")
    require(checkpoint.get("schedule_sha256") == locks["schedule_content_sha256"], "Checkpoint schedule changed")
    require(checkpoint.get("config") == yaml.safe_load(Path(locks["config_path"]).read_text(encoding="utf-8")), "Checkpoint config differs from locked config")
    require(checkpoint.get("data30m_validation", {}).get("loss") == summary["best_data30m_validation_loss"], "Checkpoint validation loss differs")
    require(checkpoint.get("source_valid_targets_observed") == summary["source_valid_targets_observed"], "Checkpoint exposure differs")
    require(checkpoint.get("pair_presentations") == summary["pair_presentations"], "Checkpoint pair exposure differs")

    proof = {
        "status": "PASS", "heldout_opened": False,
        "training_complete": True, "minimum_data30m_validation_candidate": True,
        "checkpoint_sha256": checkpoint_hash, "checkpoint_step": checkpoint["step"],
        "best_data30m_validation_loss": summary["best_data30m_validation_loss"],
        "training_commit": protocol["training_commit"], "config_sha256": locks["config_sha256"],
        "counterfactual_manifest_sha256": locks["counterfactual_manifest_sha256"],
        "schedule_file_sha256": locks["schedule_file_sha256"],
        "schedule_content_sha256": locks["schedule_content_sha256"],
        "held_out_access_during_training": False,
        "manifest_status": manifest.get("status"),
        "manifest_status_note": "The immutable trainer start manifest was not finalized; completion is independently established by the hash-locked final summary and selected final-step checkpoint.",
    }
    return summary, checkpoint, checkpoint_path, proof


def run_after_gate(gate: Callable[[], tuple[Any, ...]], heldout: Callable[..., Any]) -> Any:
    return heldout(*gate())


def _load_candidate(protocol: dict[str, Any], checkpoint: dict[str, Any], device: torch.device) -> tuple[Any, Tokenizer]:
    tokenizer = Tokenizer.from_file(protocol["training_locks"]["tokenizer_path"])
    return load_model(checkpoint, device), tokenizer


def run_counterfactual(protocol: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "Counterfactual evaluation requires CUDA BF16")
    spec = protocol["counterfactual"]
    verify_file(Path(spec["evaluator_path"]), spec["evaluator_sha256"], "Counterfactual evaluator")
    dataset_dir = Path(spec["dataset_directory"])
    manifest = load_json(dataset_dir / "manifest.json")
    require(sha256_file(dataset_dir / "manifest.json") == protocol["training_locks"]["counterfactual_manifest_sha256"], "Counterfactual manifest changed")
    tokenizer = Tokenizer.from_file(protocol["training_locks"]["tokenizer_path"])
    pairs = {split: load_pairs(dataset_dir, split) for split in spec["splits"]}
    device = torch.device("cuda")
    outputs = {}
    for name, model_spec in protocol["models"].items():
        if name == "capacity50m":
            continue
        verify_file(Path(model_spec["path"]), model_spec["sha256"], f"{name} checkpoint")
        payload = checkpoint if name == "counterfactual_v1" else torch.load(model_spec["path"], map_location="cpu", weights_only=False)
        model = load_model(payload, device)
        outputs[name] = {
            split: evaluate_model_split(model, values, tokenizer, device=device, batch_size=int(spec["batch_size"]), use_bf16=True, tie_tolerance=float(spec["tie_tolerance"]))
            for split, values in pairs.items()
        }
        del model
        torch.cuda.empty_cache()
    return {
        "status": "complete", "baseline_reuse": False,
        "baseline_reuse_note": "All models were freshly evaluated because the old baseline artifact's recorded commit did not contain the evaluator source.",
        "protocol": {"evaluator_sha256": spec["evaluator_sha256"], "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"), "splits": spec["splits"]},
        "models": outputs,
    }


def run_forced_choice(protocol: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    spec = protocol["forced_choice"]
    for path_key, hash_key in (("protocol_path", "protocol_sha256"), ("narrative_protocol_path", "narrative_protocol_sha256"), ("sealed_baseline_path", "sealed_baseline_sha256")):
        verify_file(Path(spec[path_key]), spec[hash_key], f"Forced-choice {path_key}")
    immutable = load_json(Path(spec["narrative_protocol_path"]))
    forced = load_json(Path(spec["protocol_path"]))
    validate_forced_choice_protocol(forced, immutable, spec["narrative_protocol_sha256"])
    baseline = load_json(Path(spec["sealed_baseline_path"]))
    require(baseline["protocol"]["sha256"] == spec["protocol_sha256"], "Sealed forced-choice protocol differs")
    expected_baselines = {"compute5", "narrative_v1", "contrastive_v1", "capacity50m"}
    require(set(baseline["models"]) == expected_baselines, "Sealed forced-choice model set differs")
    for name in expected_baselines:
        require(baseline["provenance"]["checkpoints"][name]["sha256"] == protocol["models"][name]["sha256"], f"Sealed {name} checkpoint differs")
    baseline_models = {}
    for name, value in baseline["models"].items():
        facts = value.get("facts", value.get("trials"))
        require(isinstance(facts, list) and len(facts) == 13, f"Sealed {name} fact rows differ")
        baseline_models[name] = {
            "top1_total": value["top1_total"], "denominator": value["denominator"],
            "mean_correct_vs_best_negative_margin": mean(item["correct_vs_best_negative_margin"] for item in facts),
            "facts": facts,
        }
    device = torch.device("cuda")
    model, tokenizer = _load_candidate(protocol, checkpoint, device)
    prompts = {entry["id"]: entry["prompt"] for entry in immutable["entries"]}
    trials = []
    for entry in forced["entries"]:
        context_ids, candidate_ids = encode_fixed_candidates(tokenizer, prompts[entry["sealed_entry_id"]] + entry["bridge"], entry["candidates"])
        scores = score_candidate_ids(model, context_ids, candidate_ids, device=device, use_bf16=True)
        trials.append({"fact_id": entry["fact_id"], **rank_scores(scores)})
    candidate_result = {
        "top1_total": sum(item["top1"] for item in trials), "denominator": 13,
        "mean_correct_vs_best_negative_margin": mean(item["correct_vs_best_negative_margin"] for item in trials),
        "facts": trials,
    }
    return {
        "status": "complete", "score": "mean_log_probability_decision_span",
        "protocol": baseline["protocol"], "immutable_narrative_protocol": baseline["immutable_narrative_protocol"],
        "sealed_baseline_reuse": {"path": spec["sealed_baseline_path"], "sha256": spec["sealed_baseline_sha256"]},
        "models": {**baseline_models, "counterfactual_v1": candidate_result},
    }


def run_generations(protocol: dict[str, Any], checkpoint: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec = protocol["generation"]
    for path_key, hash_key in (("historical_baseline_path", "historical_baseline_sha256"), ("narrative_baseline_path", "narrative_baseline_sha256")):
        verify_file(Path(spec[path_key]), spec[hash_key], f"Generation {path_key}")
    historical = load_json(Path(spec["historical_baseline_path"]))
    narrative = load_json(Path(spec["narrative_baseline_path"]))
    require(historical["prompts"] == PROMPTS and historical["generation_settings"] == GENERATION_SETTINGS, "Historical generation protocol differs")
    require(narrative["generation_settings"] == GENERATION_SETTINGS and narrative["sample_seed"] == spec["sample_seed"], "Narrative generation protocol differs")
    device = torch.device("cuda")
    model, tokenizer = _load_candidate(protocol, checkpoint, device)
    candidate_historical = generation_suite(model, tokenizer, device, PROMPTS, int(spec["sample_seed"]))
    historical_models = {**historical["models"], "counterfactual_v1": candidate_historical}
    historical_output = {**{key: historical[key] for key in ("prompts", "sample_seed", "generation_settings")}, "sealed_baseline_reuse": {"path": spec["historical_baseline_path"], "sha256": spec["historical_baseline_sha256"]}, "models": historical_models}

    narrative_protocol = load_json(Path(protocol["forced_choice"]["narrative_protocol_path"]))
    entries = generation_suite(model, tokenizer, device, [item["prompt"] for item in narrative_protocol["entries"]], int(spec["sample_seed"]))
    for output, definition in zip(entries, narrative_protocol["entries"], strict=True):
        output.update({key: definition[key] for key in ("id", "facts", "prompt_token_count", "fact_prefix_token_count", "tokens_after_fact_prefix")})
    narrative_models = {**narrative["models"], "counterfactual_v1_context1024": {"checkpoint_sha256": protocol["models"]["counterfactual_v1"]["sha256"], "entries": entries}}
    narrative_output = {
        "protocol_path": protocol["forced_choice"]["narrative_protocol_path"], "protocol_sha256": protocol["forced_choice"]["narrative_protocol_sha256"],
        "post_checkpoint_selection": True, "used_for_checkpoint_selection": False,
        "scoring_status": "manual_scoring_required", "semantic_scoring": "not_automated",
        "sample_seed": spec["sample_seed"], "generation_settings": GENERATION_SETTINGS,
        "sealed_baseline_reuse": {"path": spec["narrative_baseline_path"], "sha256": spec["narrative_baseline_sha256"]}, "models": narrative_models,
    }
    manual_rows = []
    for model_name, value in narrative_models.items():
        for entry in value["entries"]:
            for mode in ("greedy", "sampled"):
                for fact_type, fact_value in entry["facts"].items():
                    manual_rows.append({"model": model_name, "prompt_id": entry["id"], "mode": mode, "fact_type": fact_type, "expected_fact": fact_value, "generated_text": entry[mode], "retained": None, "evidence_quote": None, "reviewer_note": None})
    manual = {
        "status": "awaiting_manual_scoring", "score_definition": "A fact is retained only when the continuation semantically preserves the stated fact; record an exact evidence quote and reviewer note.",
        "fabricated_scores": False, "rows": manual_rows,
    }
    repetition = {
        "definition": "Tokenizer-v1 metrics over generated continuations only; prompts excluded",
        "historical_10_prompt": {name: repetition_aggregate(value) for name, value in historical_models.items()},
        "narrative_state_3_prompt": {name: repetition_aggregate(value["entries"]) for name, value in narrative_models.items()},
    }
    return historical_output, narrative_output, {"manual": manual, "repetition": repetition}


def _compact_probe_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_sha256": model.get("checkpoint_sha256"), "status": model["status"],
        "families": {name: {"best_layer": value["best_layer"], "best_layer_mean_validation_macro_f1": value["best_layer_mean_validation_macro_f1"], "comparison_summary": value["comparison_summary"]} for name, value in model.get("families", {}).items()},
    }


def run_probe(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = protocol["probe"]
    for path_key, hash_key in (("method_path", "method_sha256"), ("config_path", "config_sha256"), ("baseline_path", "baseline_sha256")):
        verify_file(Path(spec[path_key]), spec[hash_key], f"Probe {path_key}")
    config = yaml.safe_load(Path(spec["config_path"]).read_text(encoding="utf-8"))
    raw = run_frozen_probe(config, config_path=Path(spec["config_path"]))
    baseline = load_json(Path(spec["baseline_path"]))
    require(raw["protocol_version"] == baseline["protocol_version"] == 3, "Probe protocol version differs")
    require(raw["seeds"] == baseline["seeds"] and raw["shared_probe_hyperparameters"] == baseline["shared_probe_hyperparameters"], "Probe tuning differs")
    merged = {
        "status": "complete", "method_sha256": spec["method_sha256"],
        "baseline_reuse": {"path": spec["baseline_path"], "sha256": spec["baseline_sha256"]},
        "supported_requested_families": raw["supported_requested_families"],
        "unsupported_requested_families": raw["unsupported_requested_families"],
        "models": {name: _compact_probe_model(value) for name, value in baseline["models"].items()},
    }
    merged["models"]["counterfactual_v1"] = _compact_probe_model(raw["models"]["counterfactual_v1"])
    return raw, merged


def run_real_fiction(protocol: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    spec = protocol["real_fiction"]
    for path_key, hash_key in (("post_selection_protocol_path", "post_selection_protocol_sha256"), ("evaluation_protocol_path", "evaluation_protocol_sha256"), ("baseline_path", "baseline_sha256")):
        verify_file(Path(spec[path_key]), spec[hash_key], f"Real-fiction {path_key}")
    source = load_json(Path(spec["post_selection_protocol_path"]))
    baseline = load_json(Path(spec["baseline_path"]))
    require(set(baseline["models"]) == {"compute5", "narrative_v1", "contrastive_v1"}, "Real-fiction baseline model set differs")
    device = torch.device("cuda")
    model, tokenizer = _load_candidate(protocol, checkpoint, device)
    metrics = {"data30m_test": evaluate_packed(model, tokenizer, source["data30m_test"], device)}
    benchmark_protocol = load_json(Path(spec["evaluation_protocol_path"]))
    split_manifest = load_json(Path(benchmark_protocol["data30m_split_path"]))
    verify_file(Path(benchmark_protocol["data30m_split_path"]), benchmark_protocol["data30m_split_sha256"], "Data30M split manifest")
    by_name = {item["name"]: item for item in benchmark_protocol["benchmarks"]}
    for name in HISTORICAL_BENCHMARKS:
        benchmark = by_name[name]
        metrics[name] = evaluate_packed(model, tokenizer, benchmark, device, excluded_ids=benchmark_excluded_ids(benchmark, split_manifest["assignments"]))
    return {
        "status": "complete", "teacher_forced": True,
        "sealed_baseline_reuse": {"path": spec["baseline_path"], "sha256": spec["baseline_sha256"]},
        "models": {**baseline["models"], "counterfactual_v1": metrics},
    }


def _write(run_dir: Path, name: str, payload: Any) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / name
    write_json_atomic(path, payload)
    return {"path": str(path), "sha256": sha256_file(path)}


def _write_manifest(protocol_path: Path, run_dir: Path) -> dict[str, Any]:
    artifacts = {
        path.name: {"path": str(path), "sha256": sha256_file(path)}
        for path in sorted(run_dir.iterdir()) if path.is_file() and path.name != "artifact-manifest.json"
    }
    manifest = {"experiment_id": "fictionpulper-15m-data30m-counterfactual-v1", "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)}, "artifacts": artifacts}
    write_json_atomic(run_dir / "artifact-manifest.json", manifest)
    return manifest


def finalize(protocol: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    required = (
        "selection-gate.json", "counterfactual-evaluation.json", "13fact-forced-choice.json",
        "13fact-free-generation.json", "manual-fact-scoring.json", "generation-comparison.json",
        "repetition-comparison.json", "probe-candidate-raw.json", "probe-comparison.json",
        "real-fiction-evaluation.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    require(not missing, f"Cannot finalize; missing artifacts: {missing}")
    gate = load_json(run_dir / "selection-gate.json")
    require(gate["status"] == "PASS" and gate["heldout_opened"] is False, "Selection gate is not a PASS artifact")
    counterfactual = load_json(run_dir / "counterfactual-evaluation.json")
    forced = load_json(run_dir / "13fact-forced-choice.json")
    real = load_json(run_dir / "real-fiction-evaluation.json")
    repetition = load_json(run_dir / "repetition-comparison.json")
    probe = load_json(run_dir / "probe-comparison.json")
    manual = load_json(run_dir / "manual-fact-scoring.json")
    summary = {
        "status": "post_selection_evaluation_complete",
        "checkpoint_sha256": protocol["candidate"]["checkpoint_sha256"],
        "counterfactual": {
            model: {
                split: value["metrics"]["overall"]
                for split, value in splits.items()
            }
            for model, splits in counterfactual["models"].items()
        },
        "forced_choice": {
            model: {key: value[key] for key in ("top1_total", "denominator", "mean_correct_vs_best_negative_margin")}
            for model, value in forced["models"].items()
        },
        "real_fiction": {
            model: {
                benchmark: value["metrics"]
                for benchmark, value in benchmarks.items()
            }
            for model, benchmarks in real["models"].items()
        },
        "probe": probe,
        "repetition": repetition,
        "free_generation_manual_scoring_status": manual["status"],
    }
    _write(run_dir, "post-selection-summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--stage", choices=("gate", "counterfactual", "forced-choice", "free-generation", "probe", "real-fiction", "finalize", "all"), default="all")
    args = parser.parse_args()
    require(args.run_dir == RUN_DIR, f"Outputs must remain isolated under {RUN_DIR}")
    protocol = load_json(args.protocol)
    verify_protocol(protocol, args.protocol)
    summary, checkpoint, checkpoint_path, gate = verify_selection_gate(protocol, load_checkpoint=lambda path: torch.load(path, map_location="cpu", weights_only=False))
    _write(args.run_dir, "selection-gate.json", gate)
    stages = ("counterfactual", "forced-choice", "free-generation", "probe", "real-fiction") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "gate":
            continue
        if stage == "counterfactual":
            _write(args.run_dir, "counterfactual-evaluation.json", run_counterfactual(protocol, checkpoint))
        elif stage == "forced-choice":
            _write(args.run_dir, "13fact-forced-choice.json", run_forced_choice(protocol, checkpoint))
        elif stage == "free-generation":
            historical, narrative, other = run_generations(protocol, checkpoint)
            _write(args.run_dir, "generation-comparison.json", historical)
            _write(args.run_dir, "13fact-free-generation.json", narrative)
            _write(args.run_dir, "manual-fact-scoring.json", other["manual"])
            _write(args.run_dir, "repetition-comparison.json", other["repetition"])
        elif stage == "probe":
            raw, merged = run_probe(protocol)
            _write(args.run_dir, "probe-candidate-raw.json", raw)
            _write(args.run_dir, "probe-comparison.json", merged)
        elif stage == "real-fiction":
            _write(args.run_dir, "real-fiction-evaluation.json", run_real_fiction(protocol, checkpoint))
        elif stage == "finalize":
            finalize(protocol, args.run_dir)
    if args.stage == "all":
        finalize(protocol, args.run_dir)
    manifest = _write_manifest(args.protocol, args.run_dir)
    complete = args.stage in ("all", "finalize")
    compact = {
        "status": "post_selection_evaluation_complete" if complete else f"post_selection_stage_{args.stage}_complete",
        "selection_metric": protocol["selection_metric"], "used_holdouts_for_checkpoint_selection": False,
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": protocol["candidate"]["checkpoint_sha256"],
        "run_artifact_manifest": str(args.run_dir / "artifact-manifest.json"), "run_artifact_manifest_sha256": sha256_file(args.run_dir / "artifact-manifest.json"),
    }
    write_json_atomic(EXPERIMENT_DIR / "post-selection-record.json", compact)
    print(json.dumps({"stage": args.stage, "manifest": manifest, "record": compact}, indent=2))


if __name__ == "__main__":
    main()
