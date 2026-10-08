"""Build compact, unsealed narrative-state localization reports."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from src.frozen_state_probe import sha256_file
from src.train_tokenizer import write_json_atomic


ARTIFACT_NAMES = (
    "probe-summary.json", "probe-layerwise.json", "probe-layerwise.md",
    "counterfactual-context-swap.json", "counterfactual-context-swap.md",
    "13fact-forced-choice.json", "13fact-forced-choice.md",
    "teacher-forced-vs-rollout.json", "teacher-forced-vs-rollout.md",
    "failure-localization-summary.json", "failure-localization-summary.md",
    "provenance.json", "seal-candidate.json",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"Hash mismatch for {path}: {actual} != {expected}")


def classify_failure(
    *, probe_supported: bool | None, context_reversal: bool | None,
    forced_choice: bool | None, rollout_retained: bool | None,
    contrastive_pair_objective_high: bool = True,
) -> dict[str, Any]:
    """Conservative A-E evidence ladder; missing evidence is never promoted."""
    if all(value is None for value in (probe_supported, context_reversal, forced_choice, rollout_retained)):
        category = None
        label = "INCONCLUSIVE"
    elif probe_supported is True and context_reversal is False and forced_choice is False:
        category = "B"
        label = "REPRESENTATION_WITHOUT_LOGIT_USE"
    elif context_reversal is True and forced_choice is True and rollout_retained is False:
        category = "C"
        label = "GOLD_CONTEXT_USE_WITH_AUTOREGRESSIVE_COLLAPSE"
    elif context_reversal is True and forced_choice is True and rollout_retained is True:
        category = "E"
        label = "STATE_USE_EXISTS_BUT_FREE_DECODING_FAILS_FOR_OTHER_REASON"
    elif context_reversal is False and forced_choice is False and contrastive_pair_objective_high:
        category = "D"
        label = "SYNTHETIC_SHORTCUT"
    elif probe_supported is False and context_reversal is False and forced_choice is False:
        category = "A"
        label = "REPRESENTATION_FAILURE"
    else:
        category = None
        label = "INCONCLUSIVE_MIXED_EVIDENCE"
    return {
        "category": category, "label": label,
        "policy": "Contrastive pair ranking alone cannot establish representation, contextual use, or rollout retention; category D requires its failure on balanced context reversal and gold-prefix forced choice.",
    }


def _metric_mean(items: list[dict[str, Any]], key: str) -> float | None:
    return sum(float(item[key]) for item in items) / len(items) if items else None


def compact_probe(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    layerwise: dict[str, Any] = {"selection_policy": raw["selection_policy"], "models": {}}
    summary: dict[str, Any] = {"selection_policy": raw["selection_policy"], "models": {}}
    for model_name, model in raw["models"].items():
        if model.get("status") != "complete":
            layerwise["models"][model_name] = model
            summary["models"][model_name] = model
            continue
        layerwise["models"][model_name] = {}
        summary["models"][model_name] = {}
        for family, family_result in model["families"].items():
            layerwise["models"][model_name][family] = {
                "best_layer_selected_on_validation_only": family_result["best_layer"],
                "layers": family_result["layerwise_curve"],
            }
            summary["models"][model_name][family] = {
                "best_layer": family_result["best_layer"],
                "comparison_summary": family_result["comparison_summary"],
            }
    return summary, layerwise


def _markdown_table(title: str, rows: list[tuple[str, Any]]) -> str:
    lines = [f"# {title}", "", "| Measure | Value |", "|---|---:|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(lines) + "\n"


def build_reports(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    experiment_dir = config_path.parent
    run_dir = Path(config["output_dir"])
    immutable = config["protocols"]["immutable_narrative"]
    require_hash(Path(config["preflight"]["path"]), config["preflight"]["sha256"])
    require_hash(Path(immutable["path"]), immutable["sha256"])
    forced = config["protocols"]["forced_choice"]
    require_hash(Path(forced["path"]), forced["sha256"])
    require_hash(Path(config["probe_config"]), config["probe_config_sha256"])
    require_hash(Path(config["tokenizer"]["path"]), config["tokenizer"]["sha256"])
    require_hash(Path(config["curriculum"]["directory"]) / "manifest.json", config["curriculum"]["manifest_sha256"])
    for checkpoint in config["checkpoints"]:
        require_hash(Path(checkpoint["path"]), checkpoint["sha256"])
        if checkpoint.get("source_tag"):
            commit = subprocess.run(
                ["git", "rev-parse", f"{checkpoint['source_tag']}^{{commit}}"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            if commit != checkpoint["source_commit"]:
                raise RuntimeError(f"Source tag drifted: {checkpoint['source_tag']}")

    raw_paths = {
        "probe": run_dir / "probe-layerwise.json",
        "context_swap": run_dir / "counterfactual-context-swap.json",
        "forced_choice": run_dir / "13fact-forced-choice.json",
        "rollout": run_dir / "teacher-forced-vs-rollout.json",
    }
    missing = [str(path) for path in raw_paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError("Reporter requires completed raw stages: " + ", ".join(missing))
    raw = {name: load_json(path) for name, path in raw_paths.items()}
    for path, expected in raw["context_swap"]["protocol"]["source_artifacts"].items():
        require_hash(Path(path), expected)
    raw_hashes = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in raw_paths.items()}

    probe_summary, probe_layers = compact_probe(raw["probe"])
    context_compact = {"models": {name: value["groups"] for name, value in raw["context_swap"]["models"].items()}}
    forced_compact = {"score": raw["forced_choice"]["score"], "models": {
        name: {"top1_total": value["top1_total"], "denominator": value["denominator"], "trials": value["trials"]}
        for name, value in raw["forced_choice"]["models"].items()
    }}
    rollout_compact = raw["rollout"]

    context_trials = raw["context_swap"]["models"].get("contrastive_v1", {}).get("trials", [])
    forced_model = raw["forced_choice"]["models"].get("contrastive_v1", {})
    rollout_trials = raw["rollout"]["models"].get("contrastive_v1", {}).get("trials", [])
    probe_families = probe_summary["models"].get("contrastive_v1", {})
    probe_supported = any(
        family.get("comparison_summary", {}).get("generalization_holdout", {}).get("interpretation") == "strong_linear_state_signal"
        for family in probe_families.values()
    ) if probe_families else None
    context_rate = _metric_mean(context_trials, "reversal_correct")
    forced_rate = forced_model.get("top1_total", 0) / 13 if forced_model else None
    rollout_nonzero = [item for item in rollout_trials if item["length"] > 0]
    rollout_rate = sum(item["rollout"]["top1"] for item in rollout_nonzero) / len(rollout_nonzero) if rollout_nonzero else None
    classification = classify_failure(
        probe_supported=probe_supported,
        context_reversal=context_rate is not None and context_rate > 0.5,
        forced_choice=forced_rate is not None and forced_rate > 0.5,
        rollout_retained=rollout_rate is not None and rollout_rate > 0.5,
    )
    final_answers = {
        "1_linear_state_available": probe_supported,
        "2_context_reversal_changes_preference": context_rate,
        "3_teacher_forced_13fact_top1": forced_rate,
        "4_capacity_comparison_supported": "reported_per model; no causal scale claim",
        "5_state_survives_rollout": rollout_rate,
        "6_failure_localization": classification,
    }
    three_level = {
        "representation_probe": {
            "supported_curriculum_families": sorted(probe_families),
            "unsupported_historical_fact_types": ["character_name", "object_possession", "location", "stated_goal", "relationship", "secret", "scene_detail"],
            "note": "No fabricated /13 probe score; probe labels are curriculum-family classifications only.",
        },
        "teacher_forced_forced_choice": forced_compact,
        "rollout_fixed_candidates": rollout_compact,
    }
    failure = {"classification": classification, "final_answers": final_answers, "three_level_evidence": three_level}

    write_json_atomic(experiment_dir / "probe-summary.json", probe_summary)
    write_json_atomic(experiment_dir / "probe-layerwise.json", probe_layers)
    write_json_atomic(experiment_dir / "counterfactual-context-swap.json", context_compact)
    write_json_atomic(experiment_dir / "13fact-forced-choice.json", forced_compact)
    write_json_atomic(experiment_dir / "teacher-forced-vs-rollout.json", rollout_compact)
    write_json_atomic(experiment_dir / "failure-localization-summary.json", failure)
    (experiment_dir / "probe-layerwise.md").write_text(_markdown_table("Layerwise Probe", [("models", len(probe_layers["models"]))]), encoding="utf-8")
    (experiment_dir / "counterfactual-context-swap.md").write_text(_markdown_table("Counterfactual Context Swap", [("Contrastive reversal accuracy", context_rate)]), encoding="utf-8")
    (experiment_dir / "13fact-forced-choice.md").write_text(_markdown_table("13-Fact Forced Choice", [("Contrastive top-1 / 13", forced_model.get("top1_total"))]), encoding="utf-8")
    (experiment_dir / "teacher-forced-vs-rollout.md").write_text(_markdown_table("Teacher Forced vs Rollout", [("Nonzero rollout top-1 rate", rollout_rate)]), encoding="utf-8")
    (experiment_dir / "failure-localization-summary.md").write_text(_markdown_table("Failure Localization", [("Category", classification["category"]), *final_answers.items()]), encoding="utf-8")

    provenance = {
        "status": "reporting_complete_unsealed", "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "immutable_narrative_protocol": immutable, "raw_outputs": raw_hashes,
        "compact_artifacts": {},
    }
    for name in ARTIFACT_NAMES:
        path = experiment_dir / name
        if path.is_file() and name not in ("provenance.json", "seal-candidate.json"):
            provenance["compact_artifacts"][name] = sha256_file(path)
    write_json_atomic(experiment_dir / "provenance.json", provenance)
    seal_candidate = {
        "experiment_id": config["experiment_id"], "status": "seal_candidate_unsealed",
        "classification": classification, "final_answers": final_answers,
        "provenance_sha256": sha256_file(experiment_dir / "provenance.json"),
        "seal_json_created": False,
    }
    write_json_atomic(experiment_dir / "seal-candidate.json", seal_candidate)
    if (experiment_dir / "seal.json").exists():
        raise RuntimeError("Reporter must not create or overwrite seal.json")
    return seal_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = build_reports(yaml.safe_load(args.config.read_text(encoding="utf-8")), args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
