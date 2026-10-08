"""Strict staged post-selection evaluation for Dynamic-logit-v1."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
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


EXPERIMENT_ID = "fictionpulper-15m-data30m-dynamic-logit-v1"
CANDIDATE_KEY = "dynamic_logit_v1"
TRAINING_COMMIT = "ea8b4b11d862dcce02dbc5d0d4c0811231600784"
SELECTED_STEP = 2220
RUN_DIR = Path("runs/fictionpulper-15m-data30m-dynamic-logit-v1/post-selection")
PROTOCOL_PATH = Path("experiments/fictionpulper-15m-data30m-dynamic-logit-v1/post-selection-protocol.json")
STAGES = (
    "gate", "dynamic", "counterfactual", "forced-choice", "free-generation",
    "real-fiction", "probe", "finalize", "all",
)


def verify_protocol(protocol: dict[str, Any], protocol_path: Path, expected_sha256: str) -> None:
    """Verify the sealed orchestration contract without opening held-out inputs."""
    require(len(expected_sha256) == 64 and sha256_file(protocol_path) == expected_sha256,
            "Post-selection protocol hash changed")
    require(protocol.get("version") == 1 and protocol.get("experiment_id") == EXPERIMENT_ID,
            "Unsupported post-selection protocol")
    require(protocol.get("candidate_key") == CANDIDATE_KEY, "Candidate key changed")
    require(protocol.get("training_commit") == TRAINING_COMMIT, "Training commit changed")
    require(protocol.get("seed") == 1337, "Training seed changed")
    require(protocol.get("selection_metric") == "Data30M validation loss only", "Selection metric changed")
    require(protocol.get("candidate", {}).get("expected_step") == SELECTED_STEP, "Selected step changed")
    require(protocol["generation"]["settings"] == GENERATION_SETTINGS, "Generation settings changed")
    require(protocol["generation"]["sample_seed"] == 11337, "Generation seed changed")
    require(tuple(protocol["counterfactual"]["splits"]) == ("test", "generalization_holdout"),
            "Held-out counterfactual split set changed")
    require(tuple(protocol["dynamic"]["splits"]) == ("test", "generalization"),
            "Held-out dynamic split set changed")


def _git_blob_hash(commit: str, path: str) -> str:
    content = subprocess.run(
        ["git", "show", f"{commit}:{path}"], check=True, capture_output=True
    ).stdout
    return hashlib.sha256(content).hexdigest()


def _reject_auxiliary_heads(value: Mapping[str, Any], label: str) -> None:
    require(value.get("auxiliary_heads") is False, f"{label} auxiliary-head contract changed")
    require("state_heads" not in value and "heads" not in value, f"{label} contains auxiliary heads")


def verify_selection_gate(
    protocol: dict[str, Any], *, load_checkpoint: Callable[[Path], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    """Prove selection, exposure, provenance, and direct-LM identity before holdouts."""
    candidate = protocol["candidate"]
    for key in ("summary", "manifest"):
        verify_file(Path(candidate[f"{key}_path"]), candidate[f"{key}_sha256"], f"Training {key}")
    summary = load_json(Path(candidate["summary_path"]))
    manifest = load_json(Path(candidate["manifest_path"]))
    require(summary.get("status") == protocol["required_training_status"], "Training is incomplete")
    require(summary.get("optimizer_steps") == SELECTED_STEP, "Training step count is incomplete")
    require(summary.get("selection_metric") == protocol["selection_metric"], "Summary selection metric changed")
    require(manifest.get("selection_metric") == protocol["selection_metric"], "Manifest selection metric changed")
    require(manifest.get("held_out_dynamic_validation_selection_eligible") is False,
            "Dynamic diagnostic became selection eligible")
    require(manifest.get("fresh_random_initialization") is True and manifest.get("resume_checkpoint") is None,
            "Fresh initialization/no-resume proof changed")
    require(manifest.get("training_git_commit") == TRAINING_COMMIT, "Training commit changed")
    _reject_auxiliary_heads(summary, "Summary")
    _reject_auxiliary_heads(manifest, "Manifest")

    eligible = [item for item in summary.get("metrics", []) if item.get("selection_eligible")]
    require(bool(eligible), "No selection-eligible Data30M validation candidates")
    require(all(item.get("dynamic_validation_diagnostic", {}).get("selection_eligible") is False for item in eligible),
            "Dynamic held-out diagnostic influenced selection")
    selected = min(eligible, key=lambda item: float(item["data30m_validation"]["loss"]))
    require(selected.get("step") == summary.get("best_checkpoint_step") == SELECTED_STEP,
            "Checkpoint is not the minimum Data30M validation candidate")
    require(selected["data30m_validation"]["loss"] == summary.get("best_data30m_validation_loss"),
            "Selected validation loss changed")
    require(summary.get("exposure") == candidate["expected_exposure"], "Exact training exposure changed")

    locks = protocol["training_locks"]
    lock_pairs = (
        ("config_path", "config_sha256"), ("tokenizer_path", "tokenizer_sha256"),
        ("counterfactual_manifest_path", "counterfactual_manifest_sha256"),
        ("dynamic_manifest_path", "dynamic_manifest_sha256"),
        ("dynamic_train_path", "dynamic_train_sha256"),
        ("dynamic_validation_path", "dynamic_validation_sha256"),
        ("packed_metadata_path", "packed_metadata_sha256"),
        ("schedule_path", "schedule_file_sha256"),
    )
    for path_key, hash_key in lock_pairs:
        verify_file(Path(locks[path_key]), locks[hash_key], f"Training lock {path_key}")
    require(_git_blob_hash(TRAINING_COMMIT, locks["config_path"]) == locks["config_sha256"],
            "Training commit lacks locked config")
    require(manifest["locks"]["tokenizer_sha256"] == locks["tokenizer_sha256"],
            "Manifest tokenizer lock changed")
    require(manifest["dataset_gates"]["manifest_sha256"] == locks["counterfactual_manifest_sha256"],
            "Manifest counterfactual lock changed")
    for key in ("dynamic_manifest_sha256", "dynamic_train_sha256", "dynamic_validation_sha256"):
        require(manifest["locks"][key] == locks[key], f"Manifest {key} changed")
    schedule = load_json(Path(locks["schedule_path"]))
    require(verify_schedule(schedule) == locks["schedule_content_sha256"] == manifest["locks"]["schedule_content_sha256"],
            "Schedule content lock changed")

    checkpoint_path = Path(candidate["checkpoint_path"])
    checkpoint_hash = verify_file(checkpoint_path, candidate["checkpoint_sha256"], "Selected checkpoint")
    require(summary.get("best_checkpoint_sha256") == checkpoint_hash, "Summary checkpoint identity changed")
    checkpoint = load_checkpoint(checkpoint_path)
    require(checkpoint.get("format") == "fictionpulper-dynamic-logit-v1", "Checkpoint format changed")
    require(checkpoint.get("step") == SELECTED_STEP, "Checkpoint step changed")
    require(checkpoint.get("selection_metric") == protocol["selection_metric"], "Checkpoint selection metric changed")
    require(checkpoint.get("fresh_random_initialization") is True and checkpoint.get("resume_checkpoint") is None,
            "Checkpoint initialization provenance changed")
    require(checkpoint.get("model_initialization_seed") == 1337, "Checkpoint seed changed")
    _reject_auxiliary_heads(checkpoint, "Checkpoint")
    locked_config = yaml.safe_load(Path(locks["config_path"]).read_text(encoding="utf-8"))
    require(locked_config.get("selection") == {
        "metric": "data30m_validation_loss",
        "dynamic_validation_role": "held_out_diagnostic_only_not_selection",
        "test_access": "forbidden_during_training",
        "generalization_access": "forbidden_during_training",
    }, "Training held-out access policy changed")
    require(checkpoint.get("config") == locked_config, "Checkpoint config differs from locked config")
    require(checkpoint.get("locks") == protocol["checkpoint_locks"], "Checkpoint artifact locks changed")
    final_metric = next((item for item in checkpoint.get("metrics", []) if item.get("step") == SELECTED_STEP), None)
    require(final_metric is not None, "Checkpoint final metric is missing")
    assert final_metric is not None
    require(final_metric.get("exposure") == candidate["expected_exposure"], "Checkpoint exact exposure changed")
    require(final_metric.get("selection_eligible") is True
            and final_metric.get("data30m_validation", {}).get("loss") == summary["best_data30m_validation_loss"],
            "Checkpoint selected validation proof changed")
    proof = {
        "status": "PASS", "heldout_opened": False, "training_complete": True,
        "minimum_data30m_validation_candidate": True, "selection_metric": protocol["selection_metric"],
        "fresh_random_initialization": True, "resume_checkpoint": None, "auxiliary_heads": False,
        "exact_exposure": candidate["expected_exposure"], "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": SELECTED_STEP, "best_data30m_validation_loss": summary["best_data30m_validation_loss"],
        "training_commit": TRAINING_COMMIT, "held_out_access_during_training": False,
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
    require({"compute5", "counterfactual_v1"} <= set(baseline.get("models", {})),
            f"Sealed {section} baseline must contain Compute5 and Counterfactual-v1")
    return baseline


def _call_flexibly(function: Callable[..., Any], values: dict[str, Any]) -> Any:
    signature = inspect.signature(function)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return function(**values)
    accepted = {name: value for name, value in values.items() if name in signature.parameters}
    missing = [name for name, parameter in signature.parameters.items()
               if parameter.default is inspect.Parameter.empty
               and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
               and name not in accepted]
    require(not missing, f"Dynamic evaluator has unsupported required parameters: {missing}")
    return function(**accepted)


def run_dynamic(protocol: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Run the dedicated held-out evaluator, imported only after gate completion."""
    spec = protocol["dynamic"]
    if "evaluator_path" in spec:
        verify_file(Path(spec["evaluator_path"]), spec["evaluator_sha256"], "Dynamic-logit evaluator")
    module = importlib.import_module("src.evaluate_dynamic_logit")
    tokenizer = Tokenizer.from_file(str(protocol["training_locks"]["tokenizer_path"]))
    device = torch.device("cuda")
    model = load_model(checkpoint, device)
    outputs = {}
    provenance = {}
    for split in spec["splits"]:
        grouped, pairs, split_provenance = module.load_verified_split(
            split=split,
            dynamic_dir=Path(spec["dynamic_directory"]),
            transition_dir=Path(spec["transition_directory"]),
            source_dir=Path(spec["source_directory"]),
            tokenizer=tokenizer,
            locks=spec["locks"],
        )
        outputs[split] = module.evaluate_split(
            model, grouped, pairs, tokenizer, device=device, batch_size=int(spec["batch_size"])
        )
        provenance[split] = split_provenance
    return {
        "status": "complete", "checkpoint_sha256": protocol["candidate"]["checkpoint_sha256"],
        "score": "mean decision-span token log probability", "auxiliary_heads": False,
        "provenance": provenance, "splits": outputs,
    }


def run_counterfactual(protocol: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    spec = protocol["counterfactual"]
    verify_file(Path(spec["evaluator_path"]), spec["evaluator_sha256"], "Unchanged Counterfactual-v1 evaluator")
    baseline = _sealed(protocol, "counterfactual")
    require(baseline["protocol"]["evaluator_sha256"] == spec["evaluator_sha256"], "Baseline evaluator differs")
    tokenizer = Tokenizer.from_file(str(protocol["training_locks"]["tokenizer_path"]))
    device = torch.device("cuda")
    model = load_model(checkpoint, device)
    pairs = {split: load_pairs(Path(spec["dataset_directory"]), split) for split in spec["splits"]}
    candidate = {split: evaluate_model_split(model, values, tokenizer, device=device,
                 batch_size=int(spec["batch_size"]), use_bf16=True, tie_tolerance=float(spec["tie_tolerance"]))
                 for split, values in pairs.items()}
    return {**baseline, "sealed_baseline_reuse": {"path": spec["sealed_baseline_path"], "sha256": spec["sealed_baseline_sha256"]},
            "models": {**baseline["models"], CANDIDATE_KEY: candidate}}


def run_forced_choice(protocol: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    spec = protocol["forced_choice"]
    verify_file(Path(spec["protocol_path"]), spec["protocol_sha256"], "Forced-choice protocol")
    verify_file(Path(spec["narrative_protocol_path"]), spec["narrative_protocol_sha256"], "Narrative protocol")
    baseline = _sealed(protocol, "forced_choice")
    immutable, forced = load_json(Path(spec["narrative_protocol_path"])), load_json(Path(spec["protocol_path"]))
    validate_forced_choice_protocol(forced, immutable, spec["narrative_protocol_sha256"])
    require(baseline["protocol"]["sha256"] == spec["protocol_sha256"], "Sealed forced-choice protocol differs")
    model, tokenizer = _candidate(protocol, checkpoint, torch.device("cuda"))
    prompts = {entry["id"]: entry["prompt"] for entry in immutable["entries"]}
    facts = []
    for entry in forced["entries"]:
        context_ids, candidate_ids = encode_fixed_candidates(tokenizer, prompts[entry["sealed_entry_id"]] + entry["bridge"], entry["candidates"])
        facts.append({"fact_id": entry["fact_id"], **rank_scores(score_candidate_ids(model, context_ids, candidate_ids, device=torch.device("cuda"), use_bf16=True))})
    result = {"top1_total": sum(row["top1"] for row in facts), "denominator": 13,
              "mean_correct_vs_best_negative_margin": mean(row["correct_vs_best_negative_margin"] for row in facts), "facts": facts}
    return {**baseline, "sealed_baseline_reuse": {"path": spec["sealed_baseline_path"], "sha256": spec["sealed_baseline_sha256"]},
            "models": {**baseline["models"], CANDIDATE_KEY: result}}


def run_generations(protocol: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, dict[str, Any]]:
    spec, narrative_spec = protocol["generation"], protocol["forced_choice"]
    baseline = _sealed(protocol, "generation")
    verify_file(Path(narrative_spec["narrative_protocol_path"]), narrative_spec["narrative_protocol_sha256"], "Narrative generation protocol")
    verify_file(Path(spec["narrative_baseline_path"]), spec["narrative_baseline_sha256"], "Sealed narrative generation baseline")
    narrative_baseline = load_json(Path(spec["narrative_baseline_path"]))
    require(baseline["prompts"] == PROMPTS and baseline["generation_settings"] == GENERATION_SETTINGS,
            "Historical generation protocol differs")
    require(narrative_baseline["generation_settings"] == GENERATION_SETTINGS, "13-fact generation settings differ")
    require({"compute5_context1024", "counterfactual_v1_context1024"} <= set(narrative_baseline["models"]),
            "13-fact baseline model set differs")
    model, tokenizer = _candidate(protocol, checkpoint, torch.device("cuda"))
    historical_entries = generation_suite(model, tokenizer, torch.device("cuda"), PROMPTS, int(spec["sample_seed"]))
    historical_models = {**baseline["models"], CANDIDATE_KEY: historical_entries}
    narrative_protocol = load_json(Path(narrative_spec["narrative_protocol_path"]))
    entries = generation_suite(model, tokenizer, torch.device("cuda"), [item["prompt"] for item in narrative_protocol["entries"]], int(spec["sample_seed"]))
    for output, definition in zip(entries, narrative_protocol["entries"], strict=True):
        output.update({key: definition[key] for key in ("id", "facts", "prompt_token_count", "fact_prefix_token_count", "tokens_after_fact_prefix")})
    narrative_key = f"{CANDIDATE_KEY}_context1024"
    narrative_models = {**narrative_baseline["models"], narrative_key: {"checkpoint_sha256": protocol["candidate"]["checkpoint_sha256"], "entries": entries}}
    rows = [{"model": narrative_key, "prompt_id": entry["id"], "mode": mode, "fact_type": fact_type,
             "expected_fact": fact_value, "generated_text": entry[mode], "retained": None,
             "evidence_quote": None, "reviewer_note": None}
            for entry in entries for mode in ("greedy", "sampled") for fact_type, fact_value in entry["facts"].items()]
    return {
        "generation-comparison.json": {**{key: baseline[key] for key in ("prompts", "sample_seed", "generation_settings")},
            "sealed_baseline_reuse": {"path": spec["sealed_baseline_path"], "sha256": spec["sealed_baseline_sha256"]}, "models": historical_models},
        "13fact-free-generation.json": {**{key: narrative_baseline[key] for key in ("protocol_path", "protocol_sha256", "post_checkpoint_selection", "used_for_checkpoint_selection", "sample_seed", "generation_settings")},
            "scoring_status": "manual_scoring_required", "semantic_scoring": "not_automated", "models": narrative_models},
        "manual-fact-scoring.json": {"status": "awaiting_manual_scoring", "score_definition": "A fact is retained only when the continuation semantically preserves the stated fact; record an exact evidence quote and reviewer note.", "fabricated_scores": False, "rows": rows},
        "repetition-comparison.json": {"definition": "Tokenizer-v1 metrics over generated continuations only; prompts excluded",
            "historical_10_prompt": {name: repetition_aggregate(value) for name, value in historical_models.items()},
            "narrative_state_3_prompt": {name: repetition_aggregate(value["entries"]) for name, value in narrative_models.items()}},
    }


def run_real_fiction(protocol: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    spec = protocol["real_fiction"]
    baseline = _sealed(protocol, "real_fiction")
    verify_file(Path(spec["evaluation_protocol_path"]), spec["evaluation_protocol_sha256"], "Real-fiction evaluation protocol")
    verify_file(Path(spec["data_protocol_path"]), spec["data_protocol_sha256"], "Real-fiction data protocol")
    evaluation_protocol, source = load_json(Path(spec["evaluation_protocol_path"])), load_json(Path(spec["data_protocol_path"]))
    model, tokenizer = _candidate(protocol, checkpoint, torch.device("cuda"))
    metrics = {"data30m_test": evaluate_packed(model, tokenizer, source["data30m_test"], torch.device("cuda"))}
    split_manifest = load_json(Path(evaluation_protocol["data30m_split_path"]))
    verify_file(Path(evaluation_protocol["data30m_split_path"]), evaluation_protocol["data30m_split_sha256"], "Data30M split manifest")
    by_name = {item["name"]: item for item in evaluation_protocol["benchmarks"]}
    for name in HISTORICAL_BENCHMARKS:
        benchmark = by_name[name]
        metrics[name] = evaluate_packed(model, tokenizer, benchmark, torch.device("cuda"),
                                        excluded_ids=benchmark_excluded_ids(benchmark, split_manifest["assignments"]))
    return {**baseline, "sealed_baseline_reuse": {"path": spec["sealed_baseline_path"], "sha256": spec["sealed_baseline_sha256"]},
            "models": {**baseline["models"], CANDIDATE_KEY: metrics}}


def run_probe(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = protocol["probe"]
    verify_file(Path(spec["method_path"]), spec["method_sha256"], "Frozen-probe method")
    verify_file(Path(spec["config_path"]), spec["config_sha256"], "Frozen-probe config")
    baseline = _sealed(protocol, "probe")
    verify_file(Path(spec["sealed_raw_path"]), spec["sealed_raw_sha256"], "Sealed raw probe baseline")
    sealed_raw = load_json(Path(spec["sealed_raw_path"]))
    config = yaml.safe_load(Path(spec["config_path"]).read_text(encoding="utf-8"))
    require(config["output_path"] == str(RUN_DIR / "probe-candidate-raw.json"), "Probe output path is not isolated")
    require([item["id"] for item in config["checkpoints"] if item.get("enabled", True)] == [CANDIDATE_KEY],
            "Probe config candidate changed")
    require(config["checkpoints"][0]["path"] == protocol["candidate"]["checkpoint_path"], "Probe checkpoint changed")
    output_path = Path(config["output_path"])
    raw = load_json(output_path) if output_path.exists() else run_frozen_probe(config, config_path=Path(spec["config_path"]))
    require(raw["protocol_version"] == sealed_raw["protocol_version"] == 3, "Probe methodology version differs")
    require(raw["seeds"] == sealed_raw["seeds"] and raw["shared_probe_hyperparameters"] == sealed_raw["shared_probe_hyperparameters"],
            "Probe methodology differs")
    merged = {"status": "complete", "method_sha256": spec["method_sha256"],
              "sealed_baseline_reuse": {"path": spec["sealed_baseline_path"], "sha256": spec["sealed_baseline_sha256"]},
              "supported_requested_families": raw["supported_requested_families"],
              "unsupported_requested_families": raw["unsupported_requested_families"],
              "models": {name: _compact_probe_model(value) for name, value in baseline["models"].items()}}
    merged["models"][CANDIDATE_KEY] = _compact_probe_model(raw["models"][CANDIDATE_KEY])
    return raw, merged


def _write_once(run_dir: Path, name: str, payload: Any) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / name
    if path.exists():
        require(load_json(path) == payload, f"Refusing to overwrite non-identical artifact: {path}")
    else:
        write_json_atomic(path, payload)
    return {"path": str(path), "sha256": sha256_file(path)}


def finalize(protocol: dict[str, Any], protocol_path: Path, run_dir: Path) -> dict[str, Any]:
    required = (
        "selection-gate.json", "dynamic-logit-evaluation.json", "counterfactual-evaluation.json",
        "13fact-forced-choice.json", "13fact-free-generation.json", "manual-fact-scoring.json",
        "generation-comparison.json", "repetition-comparison.json", "real-fiction-evaluation.json",
        "probe-candidate-raw.json", "probe-comparison.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    require(not missing, f"Cannot finalize; missing artifacts: {missing}")
    gate = load_json(run_dir / "selection-gate.json")
    require(gate.get("status") == "PASS" and gate.get("heldout_opened") is False, "Selection gate is not PASS")
    require(load_json(run_dir / "manual-fact-scoring.json").get("status") == "awaiting_manual_scoring",
            "Manual semantic scoring status changed")
    summary = {"status": "post_selection_evaluation_complete", "candidate_key": CANDIDATE_KEY,
               "checkpoint_sha256": protocol["candidate"]["checkpoint_sha256"],
               "used_holdouts_for_checkpoint_selection": False, "manual_semantic_scoring": "awaiting_manual_scoring"}
    _write_once(run_dir, "post-selection-summary.json", summary)
    artifacts = {path.name: {"path": str(path), "sha256": sha256_file(path)}
                 for path in sorted(run_dir.iterdir()) if path.is_file() and path.name != "artifact-manifest.json"}
    manifest = {"experiment_id": EXPERIMENT_ID,
                "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
                "artifacts": artifacts}
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
    summary, checkpoint, _, gate = verify_selection_gate(
        protocol, load_checkpoint=lambda path: torch.load(path, map_location="cpu", weights_only=False)
    )
    _write_once(args.run_dir, "selection-gate.json", gate)
    stages = STAGES[1:-2] if args.stage == "all" else (args.stage,)
    require(args.stage in ("gate", "finalize") or torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            "Evaluation stages require CUDA BF16")
    for stage in stages:
        if stage == "dynamic":
            _write_once(args.run_dir, "dynamic-logit-evaluation.json", run_dynamic(protocol, checkpoint))
        elif stage == "counterfactual":
            _write_once(args.run_dir, "counterfactual-evaluation.json", run_counterfactual(protocol, checkpoint))
        elif stage == "forced-choice":
            _write_once(args.run_dir, "13fact-forced-choice.json", run_forced_choice(protocol, checkpoint))
        elif stage == "free-generation":
            for name, payload in run_generations(protocol, checkpoint).items():
                _write_once(args.run_dir, name, payload)
        elif stage == "real-fiction":
            _write_once(args.run_dir, "real-fiction-evaluation.json", run_real_fiction(protocol, checkpoint))
        elif stage == "probe":
            raw, merged = run_probe(protocol)
            _write_once(args.run_dir, "probe-candidate-raw.json", raw)
            _write_once(args.run_dir, "probe-comparison.json", merged)
        elif stage == "finalize":
            finalize(protocol, args.protocol, args.run_dir)
    if args.stage == "all":
        finalize(protocol, args.protocol, args.run_dir)
    print(json.dumps({"stage": args.stage, "run_dir": str(args.run_dir),
                      "selection_step": summary["best_checkpoint_step"]}, indent=2))


if __name__ == "__main__":
    main()
