"""Strict staged post-selection evaluation for State-Transition-v1."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping

import torch
import yaml
from tokenizers import Tokenizer

from src.counterfactual_narrative.evaluator import evaluate_model_split, load_pairs
from src.counterfactual_post_selection import HISTORICAL_BENCHMARKS, _compact_probe_model
from src.evaluate_state_transition import evaluate_split, load_model_and_heads
from src.frozen_state_probe import run as run_frozen_probe
from src.mixed_schedule import verify_schedule
from src.narrative_post_selection import (
    PROMPTS, evaluate_packed, generation_suite, load_json, load_model,
    repetition_aggregate, require, verify_file,
)
from src.narrative_state_localization import (
    encode_fixed_candidates, rank_scores, score_candidate_ids,
    validate_forced_choice_protocol,
)
from src.train import GENERATION_SETTINGS, benchmark_excluded_ids
from src.train_state_transition import check_lm_only_equivalence, export_lm_only, head_config
from src.train_tokenizer import sha256_file, write_json_atomic


EXPERIMENT_ID = "fictionpulper-15m-data30m-state-transition-v1"
CANDIDATE_KEY = "state_transition_v1"
RUN_DIR = Path("runs/fictionpulper-15m-data30m-state-transition-v1/post-selection")
PROTOCOL_PATH = Path("experiments/fictionpulper-15m-data30m-state-transition-v1/post-selection-protocol.json")
STAGES = ("gate", "counterfactual", "auxiliary", "forced-choice", "free-generation", "real-fiction", "probe", "lm-export", "finalize", "all")


def verify_protocol(protocol: dict[str, Any], protocol_path: Path, expected_sha256: str) -> None:
    require(len(expected_sha256) == 64 and sha256_file(protocol_path) == expected_sha256, "Post-selection protocol hash changed")
    require(protocol["version"] == 1 and protocol["experiment_id"] == EXPERIMENT_ID, "Unsupported post-selection protocol")
    require(protocol["candidate_key"] == CANDIDATE_KEY, "Candidate key changed")
    require(protocol["selection_metric"] == "Data30M validation loss only", "Selection metric changed")
    require(protocol["generation"]["settings"] == GENERATION_SETTINGS, "Generation settings changed")
    require(protocol["generation"]["sample_seed"] == 11337, "Generation seed changed")
    require(tuple(protocol["counterfactual"]["splits"]) == ("test", "generalization_holdout"), "Held-out split set changed")
    require(tuple(protocol["auxiliary"]["splits"]) == ("test", "generalization"), "Auxiliary split set changed")
    # Do not hash or open evaluation inputs here. The hash-locked protocol is
    # sufficient to run the selection gate; all held-out artifacts are touched
    # only inside post-gate stage functions.


def _git_blob_hash(commit: str, path: str) -> str:
    content = subprocess.run(["git", "show", f"{commit}:{path}"], check=True, capture_output=True).stdout
    import hashlib
    return hashlib.sha256(content).hexdigest()


def verify_selection_gate(
    protocol: dict[str, Any], *, load_checkpoint: Callable[[Path], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    """Prove completion, selection, locks, exposure, provenance, and identity."""
    candidate = protocol["candidate"]
    for key in ("summary", "manifest"):
        verify_file(Path(candidate[f"{key}_path"]), candidate[f"{key}_sha256"], f"Training {key}")
    summary = load_json(Path(candidate["summary_path"]))
    manifest = load_json(Path(candidate["manifest_path"]))
    require(summary.get("status") == protocol["required_training_status"], "Training is incomplete")
    require(summary.get("optimizer_steps") == candidate["expected_step"], "Training step count is incomplete")
    require(summary.get("selection_metric") == protocol["selection_metric"], "Summary selection metric changed")
    require(manifest.get("selection_metric") == protocol["selection_metric"], "Manifest selection metric changed")
    require(manifest.get("held_out_access_during_training") is False, "Training accessed held-out data")
    require(manifest.get("fresh_random_initialization") is True and manifest.get("resume_checkpoint") is None, "Fresh initialization proof changed")
    require(manifest.get("training_git_commit") == protocol["training_commit"], "Training commit changed")
    eligible = [item for item in summary.get("metrics", []) if item.get("selection_eligible")]
    require(bool(eligible), "No selection-eligible Data30M validation candidates")
    selected = min(eligible, key=lambda item: float(item["data30m_validation"]["loss"]))
    require(selected["step"] == summary["best_checkpoint_step"] == candidate["expected_step"], "Checkpoint is not the minimum Data30M validation candidate")
    require(selected["data30m_validation"]["loss"] == summary["best_data30m_validation_loss"], "Selected validation loss changed")
    require(summary.get("exposure") == candidate["expected_exposure"], "Exact training exposure changed")

    locks = protocol["training_locks"]
    for path_key, hash_key in (
        ("config_path", "config_sha256"), ("tokenizer_path", "tokenizer_sha256"),
        ("counterfactual_manifest_path", "counterfactual_manifest_sha256"),
        ("state_manifest_path", "state_manifest_sha256"), ("state_train_path", "state_train_sha256"),
        ("packed_metadata_path", "packed_metadata_sha256"), ("schedule_path", "schedule_file_sha256"),
    ):
        verify_file(Path(locks[path_key]), locks[hash_key], f"Training lock {path_key}")
    require(_git_blob_hash(protocol["training_commit"], locks["config_path"]) == locks["config_sha256"], "Training commit lacks locked config")
    require(manifest["locks"]["tokenizer_sha256"] == locks["tokenizer_sha256"], "Manifest tokenizer lock changed")
    require(manifest["locks"]["state_manifest_sha256"] == locks["state_manifest_sha256"], "Manifest state lock changed")
    require(manifest["locks"]["state_train_sha256"] == locks["state_train_sha256"], "Manifest state train lock changed")
    require(manifest["dataset_gates"]["manifest_sha256"] == locks["counterfactual_manifest_sha256"], "Manifest source lock changed")
    schedule = load_json(Path(locks["schedule_path"]))
    require(verify_schedule(schedule) == locks["schedule_content_sha256"] == manifest["locks"]["schedule_content_sha256"], "Schedule content lock changed")

    checkpoint_path = Path(candidate["checkpoint_path"])
    checkpoint_hash = verify_file(checkpoint_path, candidate["checkpoint_sha256"], "Selected checkpoint")
    require(summary["best_checkpoint_sha256"] == checkpoint_hash, "Summary checkpoint identity changed")
    checkpoint = load_checkpoint(checkpoint_path)
    require(checkpoint.get("format") == "fictionpulper-state-transition-v1", "Checkpoint format changed")
    require(checkpoint.get("step") == candidate["expected_step"], "Checkpoint step changed")
    require(checkpoint.get("selection_metric") == protocol["selection_metric"], "Checkpoint selection metric changed")
    require(checkpoint.get("fresh_random_initialization") is True and checkpoint.get("resume_checkpoint") is None, "Checkpoint initialization provenance changed")
    require(checkpoint.get("model_initialization_seed") == protocol["seed"], "Checkpoint seed changed")
    require(checkpoint.get("head_config") == head_config(), "Checkpoint head contract changed")
    require(checkpoint.get("config") == yaml.safe_load(Path(locks["config_path"]).read_text(encoding="utf-8")), "Checkpoint config differs from locked config")
    require(checkpoint.get("locks") == protocol["checkpoint_locks"], "Checkpoint artifact locks changed")
    final_metric = next(item for item in checkpoint.get("metrics", []) if item.get("step") == candidate["expected_step"])
    require(final_metric.get("exposure") == {key: value for key, value in candidate["expected_exposure"].items() if key != "total_state_annotations"}, "Checkpoint exact exposure changed")
    proof = {
        "status": "PASS", "heldout_opened": False, "training_complete": True,
        "minimum_data30m_validation_candidate": True, "fresh_random_initialization": True,
        "exact_exposure": candidate["expected_exposure"], "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": checkpoint["step"], "best_data30m_validation_loss": summary["best_data30m_validation_loss"],
        "training_commit": protocol["training_commit"], "held_out_access_during_training": False,
    }
    return summary, checkpoint, checkpoint_path, proof


def run_after_gate(gate: Callable[[], tuple[Any, ...]], heldout: Callable[..., Any]) -> Any:
    return heldout(*gate())


def _candidate(protocol: Mapping[str, Any], checkpoint: Mapping[str, Any], device: torch.device) -> tuple[Any, Tokenizer]:
    tokenizer = Tokenizer.from_file(str(protocol["training_locks"]["tokenizer_path"]))
    return load_model(dict(checkpoint), device), tokenizer


def _sealed(protocol: Mapping[str, Any], section: str) -> dict[str, Any]:
    spec = protocol[section]
    verify_file(Path(spec["sealed_baseline_path"]), spec["sealed_baseline_sha256"], f"Sealed {section} baseline")
    baseline = load_json(Path(spec["sealed_baseline_path"]))
    require(
        {"compute5", "counterfactual_v1"} <= set(baseline.get("models", {})),
        f"Sealed {section} baseline must contain Compute5 and Counterfactual-v1",
    )
    return baseline


def run_counterfactual(protocol: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    spec = protocol["counterfactual"]
    verify_file(Path(spec["evaluator_path"]), spec["evaluator_sha256"], "Unchanged Counterfactual-v1 evaluator")
    baseline = _sealed(protocol, "counterfactual")
    require(baseline["protocol"]["evaluator_sha256"] == spec["evaluator_sha256"], "Baseline evaluator differs")
    tokenizer = Tokenizer.from_file(str(protocol["training_locks"]["tokenizer_path"]))
    device = torch.device("cuda")
    model = load_model(checkpoint, device)
    pairs = {split: load_pairs(Path(spec["dataset_directory"]), split) for split in spec["splits"]}
    candidate = {split: evaluate_model_split(model, values, tokenizer, device=device, batch_size=int(spec["batch_size"]), use_bf16=True, tie_tolerance=float(spec["tie_tolerance"])) for split, values in pairs.items()}
    return {**baseline, "sealed_baseline_reuse": {"path": spec["sealed_baseline_path"], "sha256": spec["sealed_baseline_sha256"]}, "models": {**baseline["models"], CANDIDATE_KEY: candidate}}


def run_auxiliary(protocol: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    spec = protocol["auxiliary"]
    tokenizer = Tokenizer.from_file(str(protocol["training_locks"]["tokenizer_path"]))
    device = torch.device("cuda")
    return {
        "status": "complete", "checkpoint_sha256": protocol["candidate"]["checkpoint_sha256"],
        "splits": {split: evaluate_split(
            checkpoint=checkpoint, tokenizer=tokenizer, state_directory=Path(spec["state_directory"]),
            source_directory=Path(spec["source_directory"]), split=split, locks=spec["locks"],
            device=device, batch_size=int(spec["batch_size"]), use_bf16=bool(spec["use_bf16"]),
        ) for split in spec["splits"]},
    }


def run_forced_choice(protocol: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    spec = protocol["forced_choice"]
    verify_file(Path(spec["protocol_path"]), spec["protocol_sha256"], "Forced-choice protocol")
    verify_file(Path(spec["narrative_protocol_path"]), spec["narrative_protocol_sha256"], "Narrative protocol")
    baseline = _sealed(protocol, "forced_choice")
    immutable = load_json(Path(spec["narrative_protocol_path"]))
    forced = load_json(Path(spec["protocol_path"]))
    validate_forced_choice_protocol(forced, immutable, spec["narrative_protocol_sha256"])
    require(baseline["protocol"]["sha256"] == spec["protocol_sha256"], "Sealed forced-choice protocol differs")
    model, tokenizer = _candidate(protocol, checkpoint, torch.device("cuda"))
    prompts = {entry["id"]: entry["prompt"] for entry in immutable["entries"]}
    facts = []
    for entry in forced["entries"]:
        context_ids, candidate_ids = encode_fixed_candidates(tokenizer, prompts[entry["sealed_entry_id"]] + entry["bridge"], entry["candidates"])
        facts.append({"fact_id": entry["fact_id"], **rank_scores(score_candidate_ids(model, context_ids, candidate_ids, device=torch.device("cuda"), use_bf16=True))})
    result = {"top1_total": sum(row["top1"] for row in facts), "denominator": 13, "mean_correct_vs_best_negative_margin": mean(row["correct_vs_best_negative_margin"] for row in facts), "facts": facts}
    return {**baseline, "sealed_baseline_reuse": {"path": spec["sealed_baseline_path"], "sha256": spec["sealed_baseline_sha256"]}, "models": {**baseline["models"], CANDIDATE_KEY: result}}


def run_generations(protocol: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, dict[str, Any]]:
    spec = protocol["generation"]
    baseline = _sealed(protocol, "generation")
    narrative_spec = protocol["forced_choice"]
    verify_file(Path(narrative_spec["narrative_protocol_path"]), narrative_spec["narrative_protocol_sha256"], "Narrative generation protocol")
    narrative_baseline = load_json(Path(spec["narrative_baseline_path"]))
    verify_file(Path(spec["narrative_baseline_path"]), spec["narrative_baseline_sha256"], "Sealed narrative generation baseline")
    require(baseline["prompts"] == PROMPTS and baseline["generation_settings"] == GENERATION_SETTINGS, "Historical generation protocol differs")
    require(narrative_baseline["generation_settings"] == GENERATION_SETTINGS, "13-fact generation settings differ")
    require({"compute5_context1024", "counterfactual_v1_context1024"} <= set(narrative_baseline["models"]), "13-fact baseline model set differs")
    model, tokenizer = _candidate(protocol, checkpoint, torch.device("cuda"))
    historical_entries = generation_suite(model, tokenizer, torch.device("cuda"), PROMPTS, int(spec["sample_seed"]))
    historical_models = {**baseline["models"], CANDIDATE_KEY: historical_entries}
    narrative_protocol = load_json(Path(narrative_spec["narrative_protocol_path"]))
    entries = generation_suite(model, tokenizer, torch.device("cuda"), [item["prompt"] for item in narrative_protocol["entries"]], int(spec["sample_seed"]))
    for output, definition in zip(entries, narrative_protocol["entries"], strict=True):
        output.update({key: definition[key] for key in ("id", "facts", "prompt_token_count", "fact_prefix_token_count", "tokens_after_fact_prefix")})
    narrative_key = f"{CANDIDATE_KEY}_context1024"
    narrative_models = {**narrative_baseline["models"], narrative_key: {"checkpoint_sha256": protocol["candidate"]["checkpoint_sha256"], "entries": entries}}
    manual_rows = [{"model": narrative_key, "prompt_id": entry["id"], "mode": mode, "fact_type": fact_type, "expected_fact": fact_value, "generated_text": entry[mode], "retained": None, "evidence_quote": None, "reviewer_note": None} for entry in entries for mode in ("greedy", "sampled") for fact_type, fact_value in entry["facts"].items()]
    return {
        "generation-comparison.json": {**{key: baseline[key] for key in ("prompts", "sample_seed", "generation_settings")}, "sealed_baseline_reuse": {"path": spec["sealed_baseline_path"], "sha256": spec["sealed_baseline_sha256"]}, "models": historical_models},
        "13fact-free-generation.json": {**{key: narrative_baseline[key] for key in ("protocol_path", "protocol_sha256", "post_checkpoint_selection", "used_for_checkpoint_selection", "sample_seed", "generation_settings")}, "scoring_status": "manual_scoring_required", "semantic_scoring": "not_automated", "models": narrative_models},
        "manual-fact-scoring.json": {"status": "awaiting_manual_scoring", "score_definition": "A fact is retained only when the continuation semantically preserves the stated fact; record an exact evidence quote and reviewer note.", "fabricated_scores": False, "rows": manual_rows},
        "repetition-comparison.json": {"definition": "Tokenizer-v1 metrics over generated continuations only; prompts excluded", "historical_10_prompt": {name: repetition_aggregate(value) for name, value in historical_models.items()}, "narrative_state_3_prompt": {name: repetition_aggregate(value["entries"]) for name, value in narrative_models.items()}},
    }


def run_real_fiction(protocol: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    spec = protocol["real_fiction"]
    baseline = _sealed(protocol, "real_fiction")
    evaluation_protocol = load_json(Path(spec["evaluation_protocol_path"]))
    verify_file(Path(spec["evaluation_protocol_path"]), spec["evaluation_protocol_sha256"], "Real-fiction evaluation protocol")
    source = load_json(Path(spec["data_protocol_path"]))
    verify_file(Path(spec["data_protocol_path"]), spec["data_protocol_sha256"], "Real-fiction data protocol")
    model, tokenizer = _candidate(protocol, checkpoint, torch.device("cuda"))
    metrics = {"data30m_test": evaluate_packed(model, tokenizer, source["data30m_test"], torch.device("cuda"))}
    split_manifest = load_json(Path(evaluation_protocol["data30m_split_path"]))
    verify_file(Path(evaluation_protocol["data30m_split_path"]), evaluation_protocol["data30m_split_sha256"], "Data30M split manifest")
    by_name = {item["name"]: item for item in evaluation_protocol["benchmarks"]}
    for name in HISTORICAL_BENCHMARKS:
        benchmark = by_name[name]
        metrics[name] = evaluate_packed(model, tokenizer, benchmark, torch.device("cuda"), excluded_ids=benchmark_excluded_ids(benchmark, split_manifest["assignments"]))
    return {**baseline, "sealed_baseline_reuse": {"path": spec["sealed_baseline_path"], "sha256": spec["sealed_baseline_sha256"]}, "models": {**baseline["models"], CANDIDATE_KEY: metrics}}


def run_probe(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = protocol["probe"]
    verify_file(Path(spec["method_path"]), spec["method_sha256"], "Frozen-probe method")
    verify_file(Path(spec["config_path"]), spec["config_sha256"], "Frozen-probe config")
    baseline = _sealed(protocol, "probe")
    sealed_raw = load_json(Path(spec["sealed_raw_path"]))
    verify_file(Path(spec["sealed_raw_path"]), spec["sealed_raw_sha256"], "Sealed raw probe baseline")
    config = yaml.safe_load(Path(spec["config_path"]).read_text(encoding="utf-8"))
    require(config["output_path"] == str(RUN_DIR / "probe-candidate-raw.json"), "Probe output path is not isolated")
    require([item["id"] for item in config["checkpoints"] if item.get("enabled", True)] == [CANDIDATE_KEY], "Probe config candidate changed")
    require(config["checkpoints"][0]["path"] == protocol["candidate"]["checkpoint_path"], "Probe checkpoint changed")
    raw = run_frozen_probe(config, config_path=Path(spec["config_path"]))
    require(raw["protocol_version"] == sealed_raw["protocol_version"] == 3, "Probe methodology version differs")
    require(raw["seeds"] == sealed_raw["seeds"] and raw["shared_probe_hyperparameters"] == sealed_raw["shared_probe_hyperparameters"], "Probe methodology differs")
    merged = {"status": "complete", "method_sha256": spec["method_sha256"], "sealed_baseline_reuse": {"path": spec["sealed_baseline_path"], "sha256": spec["sealed_baseline_sha256"]}, "supported_requested_families": raw["supported_requested_families"], "unsupported_requested_families": raw["unsupported_requested_families"], "models": {name: _compact_probe_model(value) for name, value in baseline["models"].items()}}
    merged["models"][CANDIDATE_KEY] = _compact_probe_model(raw["models"][CANDIDATE_KEY])
    return raw, merged


def run_lm_export(protocol: dict[str, Any], checkpoint: dict[str, Any], checkpoint_path: Path, run_dir: Path) -> dict[str, Any]:
    export_path = run_dir / "lm-only.pt"
    if export_path.exists():
        raise FileExistsError(f"Refusing to overwrite {export_path}")
    exported = export_lm_only(checkpoint_path, export_path)
    model, _ = load_model_and_heads(checkpoint, torch.device("cpu"))
    tokenizer = Tokenizer.from_file(str(protocol["training_locks"]["tokenizer_path"]))
    ids = tokenizer.encode("The rain began shortly after midnight.").ids
    equivalence = check_lm_only_equivalence(model, export_path, torch.tensor([ids], dtype=torch.long), generation_steps=4)
    return {"status": "PASS", "export_path": str(export_path), "export_sha256": sha256_file(export_path), "source_checkpoint_sha256": protocol["candidate"]["checkpoint_sha256"], "state_heads_absent": "state_heads" not in exported, **equivalence}


def _write_once(run_dir: Path, name: str, payload: Any) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / name
    if path.exists():
        existing = load_json(path)
        require(existing == payload, f"Refusing to overwrite non-identical artifact: {path}")
    else:
        write_json_atomic(path, payload)
    return {"path": str(path), "sha256": sha256_file(path)}


def finalize(protocol: dict[str, Any], protocol_path: Path, run_dir: Path) -> dict[str, Any]:
    required = ("selection-gate.json", "counterfactual-evaluation.json", "auxiliary-state-evaluation.json", "13fact-forced-choice.json", "13fact-free-generation.json", "manual-fact-scoring.json", "generation-comparison.json", "repetition-comparison.json", "real-fiction-evaluation.json", "probe-candidate-raw.json", "probe-comparison.json", "lm-export-equivalence.json", "lm-only.pt")
    missing = [name for name in required if not (run_dir / name).is_file()]
    require(not missing, f"Cannot finalize; missing artifacts: {missing}")
    require(load_json(run_dir / "selection-gate.json")["status"] == "PASS", "Selection gate is not PASS")
    require(load_json(run_dir / "manual-fact-scoring.json")["status"] == "awaiting_manual_scoring", "Manual semantic scoring status changed")
    summary = {"status": "post_selection_evaluation_complete", "candidate_key": CANDIDATE_KEY, "checkpoint_sha256": protocol["candidate"]["checkpoint_sha256"], "used_holdouts_for_checkpoint_selection": False, "manual_semantic_scoring": "awaiting_manual_scoring"}
    _write_once(run_dir, "post-selection-summary.json", summary)
    artifacts = {path.name: {"path": str(path), "sha256": sha256_file(path)} for path in sorted(run_dir.iterdir()) if path.is_file() and path.name != "artifact-manifest.json"}
    manifest = {"experiment_id": EXPERIMENT_ID, "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)}, "artifacts": artifacts}
    _write_once(run_dir, "artifact-manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--stage", choices=STAGES, default="all")
    args = parser.parse_args()
    require(args.run_dir == RUN_DIR, f"Outputs must remain isolated under {RUN_DIR}")
    protocol = load_json(args.protocol)
    verify_protocol(protocol, args.protocol, args.protocol_sha256)
    summary, checkpoint, checkpoint_path, gate = verify_selection_gate(protocol, load_checkpoint=lambda path: torch.load(path, map_location="cpu", weights_only=False))
    _write_once(args.run_dir, "selection-gate.json", gate)
    stages = STAGES[1:-2] if args.stage == "all" else (args.stage,)
    require(args.stage == "gate" or torch.cuda.is_available() and torch.cuda.is_bf16_supported() or args.stage in ("lm-export", "finalize"), "Evaluation stages require CUDA BF16")
    for stage in stages:
        if stage == "counterfactual": _write_once(args.run_dir, "counterfactual-evaluation.json", run_counterfactual(protocol, checkpoint))
        elif stage == "auxiliary": _write_once(args.run_dir, "auxiliary-state-evaluation.json", run_auxiliary(protocol, checkpoint))
        elif stage == "forced-choice": _write_once(args.run_dir, "13fact-forced-choice.json", run_forced_choice(protocol, checkpoint))
        elif stage == "free-generation":
            for name, payload in run_generations(protocol, checkpoint).items(): _write_once(args.run_dir, name, payload)
        elif stage == "real-fiction": _write_once(args.run_dir, "real-fiction-evaluation.json", run_real_fiction(protocol, checkpoint))
        elif stage == "probe":
            raw, merged = run_probe(protocol); _write_once(args.run_dir, "probe-candidate-raw.json", raw); _write_once(args.run_dir, "probe-comparison.json", merged)
        elif stage == "lm-export": _write_once(args.run_dir, "lm-export-equivalence.json", run_lm_export(protocol, checkpoint, checkpoint_path, args.run_dir))
        elif stage == "finalize": finalize(protocol, args.protocol, args.run_dir)
    if args.stage == "all": finalize(protocol, args.protocol, args.run_dir)
    print(json.dumps({"stage": args.stage, "run_dir": str(args.run_dir), "selection_step": summary["best_checkpoint_step"]}, indent=2))


if __name__ == "__main__":
    main()
