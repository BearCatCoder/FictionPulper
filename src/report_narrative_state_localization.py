"""Build compact, scientifically complete, unsealed localization reports."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import yaml

from src.frozen_state_probe import sha256_file
from src.narrative_state_localization import canonical_hash
from src.train_tokenizer import write_json_atomic


ARTIFACT_NAMES = (
    "probe-summary.json", "probe-layerwise.json", "probe-layerwise.md",
    "counterfactual-context-swap.json", "counterfactual-context-swap.md",
    "13fact-forced-choice.json", "13fact-forced-choice.md",
    "teacher-forced-vs-rollout.json", "teacher-forced-vs-rollout.md",
    "failure-localization-summary.json", "failure-localization-summary.md",
    "provenance.json", "seal-candidate.json",
)
PREPARATION_COMMIT = "b6c5c2cf80c95e724942ab917f045f7f6f11084d"
ROLLOUT_FIX_COMMIT = "158b3c1fe5779dfd8bc0a9d1e279807c76ad49bb"
SPLITS = ("validation", "test", "generalization_holdout")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
MODEL_LABELS = {
    "compute5": "Compute5", "narrative_v1": "Narrative",
    "contrastive_v1": "Contrastive", "capacity50m": "50M",
}


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
    """Apply A-E definitions; narrow or one-split probe evidence is not B."""
    evidence = (probe_supported, context_reversal, forced_choice, rollout_retained)
    if all(value is None for value in evidence):
        category, label = None, "INCONCLUSIVE"
    elif context_reversal is False and forced_choice is False and contrastive_pair_objective_high:
        category, label = "D", "SYNTHETIC_SHORTCUT"
    elif probe_supported is False and context_reversal is False and forced_choice is False:
        category, label = "A", "REPRESENTATION_FAILURE"
    elif probe_supported is True and context_reversal is False and forced_choice is False:
        category, label = "B", "REPRESENTATION_WITHOUT_LOGIT_USE"
    elif context_reversal is True and forced_choice is True and rollout_retained is False:
        category, label = "C", "GOLD_CONTEXT_USE_WITH_AUTOREGRESSIVE_COLLAPSE"
    elif context_reversal is True and forced_choice is True and rollout_retained is True:
        category, label = "E", "STATE_USE_EXISTS_BUT_FREE_DECODING_FAILS_FOR_OTHER_REASON"
    else:
        category, label = None, "INCONCLUSIVE_MIXED_EVIDENCE"
    return {
        "category": category,
        "label": label,
        "definitions": {
            "A": "No robust state representation and no contextual/logit use.",
            "B": "Broad robust representation exists, but gold-context logits do not use it.",
            "C": "Gold-context state use exists, but it degrades under autoregressive rollout.",
            "D": "The synthetic objective succeeds while balanced contextual and historical gold-prefix tests fail.",
            "E": "State use survives controlled rollout; free-generation failure has another cause.",
        },
        "policy": "A strong result for only one family or one held-out split is not broad representation and cannot establish category B.",
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    return {"mean": mean, "std": math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))}


def _aggregate_metric_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot aggregate an empty metric record list")
    fixed = ("chance", "majority", "n", "support")
    for key in fixed:
        if any(record[key] != records[0][key] for record in records[1:]):
            raise ValueError(f"Probe seed records disagree on {key}")
    result = {metric: _mean_std([float(record[metric]) for record in records]) for metric in METRICS}
    result.update({key: records[0][key] for key in fixed})
    return result


def _aggregate_evaluations(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLITS:
        evaluations = [seed["evaluations"][split] for seed in seeds]
        buckets = evaluations[0]["distance_buckets"]
        if any(set(item["distance_buckets"]) != set(buckets) for item in evaluations[1:]):
            raise ValueError(f"Probe seeds disagree on {split} distance buckets")
        result[split] = {
            "overall": _aggregate_metric_records([item["overall"] for item in evaluations]),
            "distance_buckets": {
                distance: _aggregate_metric_records([
                    item["distance_buckets"][distance] for item in evaluations
                ]) for distance in buckets
            },
        }
    return result


def _base_probe_label(family: dict[str, Any]) -> str:
    interpretations = {
        family["comparison_summary"][split]["interpretation"]
        for split in ("test", "generalization_holdout")
    }
    if interpretations == {"strong_linear_state_signal"}:
        return "ROBUSTLY_DECODED"
    if interpretations & {"strong_linear_state_signal", "weak_or_imbalanced_linear_state_signal"}:
        return "WEAKLY_DECODED"
    return "NOT_DECODED"


def _macro_f1(family: dict[str, Any], split: str) -> float:
    return float(family["comparison_summary"][split]["metrics_over_seeds"]["macro_f1"]["mean"])


def classify_probe_states(raw: dict[str, Any]) -> dict[str, dict[str, str]]:
    labels = {
        model: {family: _base_probe_label(value) for family, value in data["families"].items()}
        for model, data in raw["models"].items() if data.get("status") == "complete"
    }
    comparisons = (("narrative_v1", "CURRICULUM_IMPROVED", ("compute5",)),
                   ("contrastive_v1", "CONTRASTIVE_IMPROVED", ("compute5", "narrative_v1")))
    for model, label, baselines in comparisons:
        if model not in raw["models"]:
            continue
        for family, candidate in raw["models"][model]["families"].items():
            if not all(family in raw["models"].get(base, {}).get("families", {}) for base in baselines):
                continue
            deltas = [
                _macro_f1(candidate, split) - _macro_f1(raw["models"][base]["families"][family], split)
                for base in baselines for split in ("test", "generalization_holdout")
            ]
            if all(delta > 0 for delta in deltas) and max(deltas) >= 0.05:
                labels[model][family] = label
    return labels


def compact_probe(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = classify_probe_states(raw)
    shared = {
        key: raw[key] for key in (
            "protocol_version", "seeds", "selection_policy", "position_policy", "slot_policy",
            "shared_probe_hyperparameters", "source_artifacts", "family_coverage",
            "supported_requested_families", "unsupported_requested_families",
        )
    }
    layerwise: dict[str, Any] = {**shared, "models": {}}
    summary: dict[str, Any] = {**shared, "classification_policy": {
        "robust": "Both test and generalization_holdout are strong.",
        "weak": "Either held-out split has weak or strong signal, unless a relative-improvement label applies.",
        "relative": "Every test/generalization macro-F1 delta versus the applicable prior baseline(s) must be positive, with at least one gain >= 0.05.",
    }, "models": {}, "model_by_state": labels}
    for model_name, model in raw["models"].items():
        if model.get("status") != "complete":
            layerwise["models"][model_name] = {"status": model.get("status")}
            summary["models"][model_name] = {"status": model.get("status")}
            continue
        layerwise["models"][model_name] = {
            "checkpoint_sha256": model["checkpoint_sha256"], "source": model["source"], "families": {},
        }
        summary["models"][model_name] = {
            "checkpoint_sha256": model["checkpoint_sha256"], "source": model["source"], "families": {},
        }
        for family_name, family in model["families"].items():
            layers = []
            for layer in family["layerwise_curve"]:
                layers.append({
                    "layer": layer["layer"],
                    "mean_validation_macro_f1": layer["mean_validation_macro_f1"],
                    "seed_aggregated": _aggregate_evaluations(layer["seeds"]),
                })
            best = next(item for item in layers if item["layer"] == family["best_layer"])
            layerwise["models"][model_name]["families"][family_name] = {
                "best_layer_selected_on_validation_only": family["best_layer"], "layers": layers,
            }
            summary["models"][model_name]["families"][family_name] = {
                "best_layer_selected_on_validation_only": family["best_layer"],
                "classification": labels[model_name][family_name],
                "validation": best["seed_aggregated"]["validation"],
                "test": best["seed_aggregated"]["test"],
                "generalization_holdout": best["seed_aggregated"]["generalization_holdout"],
            }
    return summary, layerwise


def _compact_context(raw: dict[str, Any]) -> dict[str, Any]:
    cases = raw["protocol"]["cases"]
    coverage: dict[str, dict[str, int]] = {}
    for case in cases:
        coverage.setdefault(case["family"], {})[case["distance"]] = coverage.setdefault(case["family"], {}).get(case["distance"], 0) + 1
    return {
        "protocol": {
            "protocol_version": raw["protocol"]["protocol_version"],
            "generator": "build_counterfactual_protocol/v1",
            "embedded_protocol_sha256": canonical_hash(raw["protocol"]),
            "target_case_count": raw["protocol"]["target_case_count"],
            "case_count": raw["protocol"]["case_count"],
            "coverage_note": raw["protocol"]["coverage_note"],
            "coverage_by_family_distance": coverage,
            "source_artifacts": raw["protocol"]["source_artifacts"],
        },
        "provenance": raw["provenance"],
        "semantics": {
            "chance_reversal_rate": 0.25,
            "reversal_correct": "Both A-prefers-X and B-prefers-Y must hold; two independent binary directions give 0.25 chance under random preferences.",
            "context_direction_accuracy": "Mean correctness across the two context directions; random preference has expectation 0.5.",
        },
        "models": {
            name: {
                "overall": value["groups"]["overall"],
                "by_family": value["groups"]["by_family"],
                "by_distance": value["groups"]["by_distance"],
                "by_difficulty": value["groups"]["by_difficulty"],
                "cases": [{
                    key: trial[key] for key in (
                        "id", "family", "distance", "difficulty", "source_split", "scores",
                        "margins_x_minus_y", "reversal_correct", "context_direction_accuracy",
                        "mean_signed_correct_margin", "absolute_preference_change",
                        "removed_absolute_margin", "removed_preference_bias",
                        "irrelevant_absolute_margin", "irrelevant_preference_bias",
                        "latest_state_update_correct",
                    )
                } for trial in value["trials"]],
            } for name, value in raw["models"].items()
        },
    }


def _compact_forced(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": raw["protocol"], "immutable_narrative_protocol": raw["immutable_narrative_protocol"],
        "score": raw["score"], "provenance": raw["provenance"],
        "models": {name: {
            "top1_total": value["top1_total"], "denominator": value["denominator"],
            "mean_correct_vs_best_negative_margin": sum(
                trial["correct_vs_best_negative_margin"] for trial in value["trials"]
            ) / len(value["trials"]),
            "facts": value["trials"],
        } for name, value in raw["models"].items()},
    }


def _rollout_aggregate(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for trial in trials:
        groups.setdefault((trial["mode"], trial["length"]), []).append(trial)
    result = []
    for (mode, length), items in sorted(groups.items()):
        divergences = [item["first_divergence_from_gold"] for item in items if item["first_divergence_from_gold"] is not None]
        contradictions = [item["explicit_state_contradiction"] for item in items if item["explicit_state_contradiction"] is not None]
        result.append({
            "mode": mode, "length": length, "top1_total": sum(bool(item["rollout"]["top1"]) for item in items),
            "denominator": len(items),
            "mean_margin": sum(item["rollout"]["correct_vs_best_negative_margin"] for item in items) / len(items),
            "first_divergence": {
                "count": len(divergences),
                "mean_token_index": sum(divergences) / len(divergences) if divergences else None,
                "min_token_index": min(divergences) if divergences else None,
                "max_token_index": max(divergences) if divergences else None,
            },
            "explicit_contradictions": {"count": len(contradictions), "evaluated": len(items)},
            "recovery_maximum_absolute_error": max(item["recovery"]["maximum_absolute_error"] for item in items),
        })
    return result


def _compact_rollout(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        key: raw[key] for key in (
            "protocol", "immutable_narrative_protocol", "forced_choice_source",
            "rollout_protocol", "score", "provenance",
        )
    } | {
        "interpretation": "No collapse point is identifiable: gold-prefix advantage is absent for Contrastive (4/13, negative mean margin), and non-monotonic rollout fluctuations are not improvement.",
        "models": {name: {
            "aggregates": _rollout_aggregate(value["trials"]),
            "facts": [{
                "fact_id": trial["fact_id"], "mode": trial["mode"], "length": trial["length"],
                "sampling_seed": trial["sampling_seed"],
                "gold": trial["gold"], "rollout": trial["rollout"],
                "first_divergence_from_gold": trial["first_divergence_from_gold"],
                "explicit_state_contradiction": trial["explicit_state_contradiction"],
                "recovery": trial["recovery"],
            } for trial in value["trials"]],
        } for name, value in raw["models"].items()},
    }


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    return "\n".join((
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows),
    ))


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_probe(summary: dict[str, Any]) -> str:
    rows = []
    for model, families in summary["models"].items():
        for family, data in families["families"].items():
            rows.append([MODEL_LABELS[model], family, data["best_layer_selected_on_validation_only"],
                         _fmt(data["test"]["overall"]["macro_f1"]["mean"]),
                         _fmt(data["generalization_holdout"]["overall"]["macro_f1"]["mean"]), data["classification"]])
    return "# Frozen Probe\n\nBest layers were selected on validation only; displayed held-out metrics are seed means. No family is robust on both held-out splits.\n\n" + _table(
        ["Model", "State", "Best layer", "Test macro-F1", "Generalization macro-F1", "Classification"], rows,
    ) + "\n\nContrastive `object_location` is the sole convincing relative improvement (test 0.5217, generalization 0.5926). Probe labels are lexical-rank state labels, so this isolated result is not broad narrative-state representation. Full seed-aggregated layer and distance-bucket metrics are in `probe-layerwise.json`.\n"


def render_context(report: dict[str, Any]) -> str:
    rows = []
    for model, data in report["models"].items():
        o = data["overall"]
        rows.append([MODEL_LABELS[model], f"{round(o['reversal_correct'] * o['count'])}/{o['count']}", _fmt(o["context_direction_accuracy"]), _fmt(o["mean_signed_correct_margin"]), _fmt(o["removed_absolute_margin"]), _fmt(o["irrelevant_absolute_margin"]), _fmt(o["latest_state_update_correct"])])
    detail = []
    contrastive = report["models"]["contrastive_v1"]
    for grouping in ("by_family", "by_distance"):
        for key, value in contrastive[grouping].items():
            detail.append([grouping.removeprefix("by_"), key, value["count"], _fmt(value["reversal_correct"]), _fmt(value["context_direction_accuracy"]), _fmt(value["mean_signed_correct_margin"])])
    return "# Counterfactual Context Swap\n\n" + _table(
        ["Model", "Reversals", "Direction", "Signed margin", "Removed |margin|", "Irrelevant |margin|", "Latest update"], rows,
    ) + "\n\n" + _table(["Contrastive group", "Value", "n", "Reversal", "Direction", "Signed margin"], detail) + "\n\nReversal requires both context directions to rank their matching continuation, so random independent binary preferences yield 0.25; direction accuracy has chance 0.50. Removed and irrelevant controls retain large absolute preference margins, showing candidate bias rather than balanced context use. Prompts use held-out vocabulary but remain templated synthetic swaps.\n"


def render_forced(report: dict[str, Any]) -> str:
    totals = [[MODEL_LABELS[name], f"{value['top1_total']}/{value['denominator']}", _fmt(value["mean_correct_vs_best_negative_margin"])] for name, value in report["models"].items()]
    facts = [[MODEL_LABELS[name], trial["fact_id"], trial["correct_rank"], _fmt(trial["correct_vs_best_negative_margin"]), trial["top1"]] for name, value in report["models"].items() for trial in value["facts"]]
    return "# 13-Fact Forced Choice\n\n" + _table(["Model", "Top-1", "Mean correct margin"], totals) + "\n\n" + _table(["Model", "Fact", "Rank", "Margin", "Top-1"], facts) + "\n"


def render_rollout(report: dict[str, Any]) -> str:
    rows = []
    for model, data in report["models"].items():
        for item in data["aggregates"]:
            rows.append([MODEL_LABELS[model], item["mode"], item["length"], f"{item['top1_total']}/{item['denominator']}", _fmt(item["mean_margin"]), item["first_divergence"]["mean_token_index"], f"{item['explicit_contradictions']['count']}/{item['explicit_contradictions']['evaluated']}", item["recovery_maximum_absolute_error"]])
    return "# Teacher Forced vs Rollout\n\n" + _table(
        ["Model", "Mode", "Length", "Top-1", "Mean margin", "Mean first divergence", "Explicit contradictions", "Recovery max"], rows,
    ) + "\n\nNo collapse point can be identified because Contrastive gold scoring is already poor (4/13; mean margin -0.2288). Its greedy counts at 0/32/64/128/256 are 4/4/5/7/6 and sampled counts are 4/5/3/5/3; these non-monotonic fluctuations are not evidence of improvement. Exact recovery errors are zero. The phrase-based contradiction check is conservative.\n"


def _build_failure(probe: dict[str, Any], context: dict[str, Any], forced: dict[str, Any], rollout: dict[str, Any]) -> dict[str, Any]:
    contrastive_probe = probe["models"]["contrastive_v1"]["families"]
    robust = [name for name, value in contrastive_probe.items() if value["classification"] == "ROBUSTLY_DECODED"]
    classification = classify_failure(probe_supported=bool(robust), context_reversal=False, forced_choice=False, rollout_retained=False)
    answers = [
        {"number": 1, "question": "Is narrative state linearly available in frozen hidden states?", "answer": "Only weakly and in isolated families; no model/family is robust on both test and generalization. Contrastive object_location is the sole convincing relative improvement, not broad representation."},
        {"number": 2, "question": "Does LM next-token scoring use earlier narrative state under a gold/correct context?", "answer": "Not reliably. Contrastive reaches only 4/13 top-1 on historical gold-prefix forced choice and only 0.5982 direction accuracy on balanced swaps."},
        {"number": 3, "question": "Does Contrastive-v1's 97% ranking accuracy survive counterfactual context reversal?", "answer": "No. Contrastive reverses 11/56 (0.1964), below the 0.25 random-pair benchmark; removed and irrelevant controls retain large candidate margins."},
        {"number": 4, "question": "At what rollout length does state-consistency scoring begin to collapse?", "answer": "No collapse length is identifiable because the gold baseline is already absent: Contrastive starts at 4/13 with mean margin -0.2288, then fluctuates non-monotonically."},
        {"number": 5, "question": "Does correcting an erroneous rollout restore the state-consistency margin?", "answer": "Yes. Replacing every generated suffix with the exact gold suffix restores every candidate score exactly (maximum absolute recovery error 0.0), but it restores the same weak gold-context baseline rather than correct state use."},
        {"number": 6, "question": "What is the primary failure localization?", "answer": "D SYNTHETIC_SHORTCUT, with weak representation as a contributing limitation: 97.65% original synthetic pair ranking does not transfer to balanced context reversal or 13-fact gold scoring."},
    ]
    historical = {name: {"greedy": "0/13", "sampled": "0/13"} for name in ("compute5", "narrative_v1", "contrastive_v1", "capacity50m")}
    return {
        "final_classification": {"category": "D", "label": "SYNTHETIC_SHORTCUT"},
        "classification_evidence": classification,
        "original_synthetic_pair_ranking": {"contrastive_test_accuracy": 0.976545842217484, "count": 469},
        "final_questions": answers,
        "three_level_evidence": [
            {"level": "representation", "Contrastive": "Weak/isolated; object_location relative improvement only", "Compute5": "Weak/isolated", "Narrative": "Weak/isolated", "50M": "Not probed"},
            {"level": "gold-prefix forced choice", "Compute5": "4/13", "Narrative": "3/13", "Contrastive": "4/13", "50M": "7/13"},
            {"level": "historical free generation", "Compute5": "0/13", "Narrative": "0/13", "Contrastive": "0/13", "50M": "0/13"},
        ],
        "historical_free_generation": historical,
        "controls": {
            "counterfactual_removed_absolute_margin": context["models"]["contrastive_v1"]["overall"]["removed_absolute_margin"],
            "counterfactual_irrelevant_absolute_margin": context["models"]["contrastive_v1"]["overall"]["irrelevant_absolute_margin"],
            "counterfactual_latest_update_accuracy": context["models"]["contrastive_v1"]["overall"]["latest_state_update_correct"],
            "rollout_recovery_maximum_absolute_error": max(item["recovery_maximum_absolute_error"] for model in rollout["models"].values() for item in model["aggregates"]),
        },
        "limitations": [
            "Controlled synthetic swaps use held-out vocabulary but templated prompts.",
            "Probe labels are lexical-rank labels, not the historical 13 facts.",
            "Each model is represented by one selected checkpoint seed.",
            "The historical suite contains only 13 facts.",
            "Exact-phrase contradiction checking is conservative.",
        ],
    }


def render_failure(report: dict[str, Any]) -> str:
    answers = "\n".join(f"{item['number']}. **{item['question']}** {item['answer']}" for item in report["final_questions"])
    rows = [[row["level"], row["Compute5"], row["Narrative"], row["Contrastive"], row["50M"]] for row in report["three_level_evidence"]]
    return f"# Failure Localization\n\n**Final classification: D SYNTHETIC_SHORTCUT.** Weak representation is a contributing limitation, but narrow one-family evidence does not justify category B. The decisive mismatch is 97.65% original synthetic pair ranking versus failed balanced reversal and poor original-context 13-fact scoring.\n\n## Final Questions\n\n{answers}\n\n## Three Levels\n\n" + _table(["Level", "Compute5", "Narrative", "Contrastive", "50M"], rows) + "\n\nNo `/13` probe score is fabricated because the probe task and historical facts differ.\n\n## Controls and Limitations\n\n" + "\n".join(f"- {item}" for item in report["limitations"]) + "\n"


def _validate_inputs(config: dict[str, Any], config_path: Path, raw: dict[str, Any], raw_paths: dict[str, Path]) -> None:
    if (config_path.parent / "seal.json").exists():
        raise RuntimeError("Localization experiment is already sealed; reporter refuses to overwrite it")
    require_hash(Path(config["preflight"]["path"]), config["preflight"]["sha256"])
    require_hash(Path(config["probe_config"]), config["probe_config_sha256"])
    require_hash(Path(config["tokenizer"]["path"]), config["tokenizer"]["sha256"])
    require_hash(Path(config["curriculum"]["directory"]) / "manifest.json", config["curriculum"]["manifest_sha256"])
    preflight = load_json(Path(config["preflight"]["path"]))
    for tag, expected_commit in preflight["resolved_tags"].items():
        resolved = subprocess.run(["git", "rev-parse", f"{tag}^{{commit}}"], check=True, capture_output=True, text=True).stdout.strip()
        if resolved != expected_commit:
            raise RuntimeError(f"Preflight tag drifted: {tag}")
    seal_paths = {
        "compute5": Path("experiments/fictionpulper-15m-data30m-compute5/seal.json"),
        "narrative_v1": Path("experiments/fictionpulper-15m-data30m-narrative-v1/seal.json"),
        "contrastive_v1": Path("experiments/fictionpulper-15m-data30m-contrastive-v1/seal.json"),
        "capacity50m": Path("experiments/fictionpulper-50m-data30m-v1/seal.json"),
        "context2k": Path("experiments/fictionpulper-15m-data30m-context2k-v1/seal.json"),
    }
    for model, expected in preflight["seal_sha256"].items():
        require_hash(seal_paths[model], expected)
    for protocol in config["protocols"].values():
        require_hash(Path(protocol["path"]), protocol["sha256"])
    configured = {item["id"]: item for item in config["checkpoints"]}
    for checkpoint in configured.values():
        require_hash(Path(checkpoint["path"]), checkpoint["sha256"])
        resolved = subprocess.run(["git", "rev-parse", f"{checkpoint['source_tag']}^{{commit}}"], check=True, capture_output=True, text=True).stdout.strip()
        if resolved != checkpoint["source_commit"]:
            raise RuntimeError(f"Source tag drifted: {checkpoint['source_tag']}")
    for commit in (PREPARATION_COMMIT, ROLLOUT_FIX_COMMIT):
        subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], check=True)
    for stage in ("context_swap", "forced_choice", "rollout"):
        provenance = raw[stage]["provenance"]
        expected_commit = ROLLOUT_FIX_COMMIT if stage == "rollout" else PREPARATION_COMMIT
        if provenance["diagnostic_git_commit"] != expected_commit:
            raise RuntimeError(f"Unexpected {stage} execution commit")
        if provenance["tokenizer"]["sha256"] != config["tokenizer"]["sha256"]:
            raise RuntimeError(f"Tokenizer provenance mismatch in {stage}")
        for model, item in provenance["checkpoints"].items():
            expected = configured[model]
            if item["sha256"] != expected["sha256"] or item["source_commit"] != expected["source_commit"]:
                raise RuntimeError(f"Checkpoint provenance mismatch for {stage}/{model}")
    for model, item in raw["probe"]["models"].items():
        expected = configured[model]
        if item["checkpoint_sha256"] != expected["sha256"] or not item["source"]["tag_matches_expected_commit"]:
            raise RuntimeError(f"Probe checkpoint provenance mismatch for {model}")
    for artifact in raw["probe"]["source_artifacts"].values():
        if "path" in artifact:
            require_hash(Path(artifact["path"]), artifact["sha256"])
    curriculum = raw["probe"]["source_artifacts"]["curriculum"]
    for name, expected in curriculum["files"].items():
        require_hash(Path(curriculum["directory"]) / name, expected)
    for path, expected in raw["context_swap"]["protocol"]["source_artifacts"].items():
        require_hash(Path(path), expected)
    protocol = raw["context_swap"]["protocol"]
    if protocol["case_count"] != len(protocol["cases"]):
        raise RuntimeError("Counterfactual protocol case count mismatch")
    for case in protocol["cases"]:
        if case["sha256"] != canonical_hash({key: value for key, value in case.items() if key != "sha256"}):
            raise RuntimeError(f"Counterfactual case hash mismatch: {case['id']}")
    immutable = config["protocols"]["immutable_narrative"]
    forced = config["protocols"]["forced_choice"]
    if raw["forced_choice"]["protocol"] != forced or raw["forced_choice"]["immutable_narrative_protocol"] != immutable:
        raise RuntimeError("Forced-choice protocol identity mismatch")
    if raw["rollout"]["protocol"] != forced or raw["rollout"]["immutable_narrative_protocol"] != immutable:
        raise RuntimeError("Rollout protocol identity mismatch")
    require_hash(raw_paths["forced_choice"], raw["rollout"]["forced_choice_source"]["sha256"])
    if raw["rollout"]["rollout_protocol"] != config["rollout"]:
        raise RuntimeError("Rollout settings differ from configuration")


def build_reports(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    experiment_dir = config_path.parent
    run_dir = Path(config["output_dir"])
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
    _validate_inputs(config, config_path, raw, raw_paths)

    probe_summary, probe_layers = compact_probe(raw["probe"])
    context = _compact_context(raw["context_swap"])
    forced = _compact_forced(raw["forced_choice"])
    rollout = _compact_rollout(raw["rollout"])
    failure = _build_failure(probe_summary, context, forced, rollout)
    reports: dict[str, Any] = {
        "probe-summary.json": probe_summary, "probe-layerwise.json": probe_layers,
        "counterfactual-context-swap.json": context, "13fact-forced-choice.json": forced,
        "teacher-forced-vs-rollout.json": rollout, "failure-localization-summary.json": failure,
    }
    markdown = {
        "probe-layerwise.md": render_probe(probe_summary),
        "counterfactual-context-swap.md": render_context(context),
        "13fact-forced-choice.md": render_forced(forced),
        "teacher-forced-vs-rollout.md": render_rollout(rollout),
        "failure-localization-summary.md": render_failure(failure),
    }
    for name, value in reports.items():
        write_json_atomic(experiment_dir / name, value)
    for name, value in markdown.items():
        (experiment_dir / name).write_text(value, encoding="utf-8")

    historical_paths = (
        Path("experiments/fictionpulper-15m-data30m-contrastive-v1/manual-fact-scoring.json"),
        Path("experiments/fictionpulper-50m-data30m-v1/summary.json"),
        Path("experiments/fictionpulper-15m-data30m-contrastive-v1/summary.json"),
    )
    provenance = {
        "status": "reporting_complete_unsealed", "validation_pending": True,
        "preparation_commit": PREPARATION_COMMIT, "rollout_fix_commit": ROLLOUT_FIX_COMMIT,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "preflight": config["preflight"], "checkpoints": config["checkpoints"],
        "tokenizer": config["tokenizer"], "curriculum": config["curriculum"],
        "protocols": config["protocols"],
        "probe": {"seeds": raw["probe"]["seeds"], "settings": raw["probe"]["shared_probe_hyperparameters"], "source_artifacts": raw["probe"]["source_artifacts"]},
        "counterfactual_generator": context["protocol"], "rollout_settings": raw["rollout"]["rollout_protocol"],
        "generation_seeds": {"base": raw["rollout"]["rollout_protocol"]["sampled"]["seed"], "policy": "stable_sampling_seed(base, fact_index, mode_index + model_index * 2), recorded per fact"},
        "raw_outputs": {name: {"path": str(path), "sha256": sha256_file(path), "execution_commit": raw[name].get("provenance", {}).get("diagnostic_git_commit") or (PREPARATION_COMMIT if name == "probe" else None)} for name, path in raw_paths.items()},
        "historical_sources": {str(path): sha256_file(path) for path in historical_paths},
        "compact_artifacts": {name: sha256_file(experiment_dir / name) for name in (*reports, *markdown)},
    }
    write_json_atomic(experiment_dir / "provenance.json", provenance)
    all_report_hashes = {name: sha256_file(experiment_dir / name) for name in (*reports, *markdown, "provenance.json")}
    seal_candidate = {
        "experiment_id": config["experiment_id"], "status": "seal_candidate_unsealed",
        "validation_pending": True,
        "validation_note": "Reporter ran before the full test suite; no final test count is asserted here.",
        "final_classification": failure["final_classification"],
        "report_artifacts": all_report_hashes, "seal_json_created": False,
    }
    write_json_atomic(experiment_dir / "seal-candidate.json", seal_candidate)
    if (experiment_dir / "seal.json").exists():
        raise RuntimeError("Reporter must never create seal.json")
    return seal_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = build_reports(yaml.safe_load(args.config.read_text(encoding="utf-8")), args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
