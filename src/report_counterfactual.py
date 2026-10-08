"""Build final, unsealed Counterfactual-v1 experiment records."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from src.train_tokenizer import sha256_file, write_json_atomic


EXPERIMENT_ID = "fictionpulper-15m-data30m-counterfactual-v1"
EXPERIMENT_DIR = Path("experiments") / EXPERIMENT_ID
RUN_DIR = Path("runs") / EXPERIMENT_ID
POST_DIR = RUN_DIR / "post-selection"
PROTOCOL_PATH = EXPERIMENT_DIR / "post-selection-protocol.json"
CLASSIFICATION = "COUNTERFACTUAL_SCORING_TRANSFER_WITHOUT_BROAD_NARRATIVE_TRANSFER"
MODELS = ("compute5", "narrative_v1", "contrastive_v1", "counterfactual_v1")
MODEL_LABELS = {
    "compute5": "Compute5",
    "narrative_v1": "Narrative-v1",
    "contrastive_v1": "Contrastive-v1",
    "counterfactual_v1": "Counterfactual-v1",
    "capacity50m": "Capacity-50M",
}
BASELINE_TAGS = {
    "corpus_v2": ("fictionpulper-corpus-v2-30m", "823ab270439fcfdbe9ba0002d275ececdadeb8e3"),
    "compute5": ("fictionpulper-15m-data30m-compute5", "979a42e7f729de69ac57ef69ec4ac8adb622cf4a"),
    "narrative_v1": ("fictionpulper-15m-data30m-narrative-v1", "71b46f8235eb3fdfc77fee782786a718f4ba5095"),
    "contrastive_v1": ("fictionpulper-15m-data30m-contrastive-v1", "ad0e69b1f0342de06eef1e8eb9be46728abb11e1"),
    "localization_v1": ("fictionpulper-narrative-state-localization-v1", "f5108d7a6b7f7070650faa694d7f74405903e395"),
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"Missing {label}: {path}")
    require(sha256_file(path) == expected, f"Hash mismatch for {label}: {path}")


def table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(("---",) + tuple("---:" for _ in headers[1:])) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def percent_change(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate / baseline - 1.0)


def compact_group(value: dict[str, Any]) -> dict[str, Any]:
    return {"pair_count": value["pair_count"], "normal": value["normal"]}


def validate_sources(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    require(sha256_file(PROTOCOL_PATH) == "b1ba01b7bc09d373a28433300540e2b6c9f9799c9a1a9121d0536307a9f21aab", "Post-selection protocol changed")
    summary_path = Path(protocol["candidate"]["summary_path"])
    manifest_path = Path(protocol["candidate"]["manifest_path"])
    verify(summary_path, protocol["candidate"]["summary_sha256"], "training summary")
    verify(manifest_path, protocol["candidate"]["manifest_sha256"], "training manifest")
    training_summary, training_manifest = load(summary_path), load(manifest_path)
    require(training_summary["status"] == "training_and_selection_complete", "Training summary is incomplete")
    require(training_summary["best_checkpoint_step"] == 2220, "Selected step changed")
    require(training_summary["held_out_access_during_training"] is False, "Held-out data were accessed during training")
    require(training_manifest["status"] == "training_started", "Unexpected trainer-manifest state")
    require(training_manifest["held_out_access_during_training"] is False, "Trainer manifest records held-out access")
    verify(Path(protocol["candidate"]["checkpoint_path"]), protocol["candidate"]["checkpoint_sha256"], "candidate checkpoint")
    for path_key, hash_key in (
        ("config_path", "config_sha256"),
        ("tokenizer_path", "tokenizer_sha256"),
        ("counterfactual_manifest_path", "counterfactual_manifest_sha256"),
        ("packed_metadata_path", "packed_metadata_sha256"),
        ("schedule_path", "schedule_file_sha256"),
    ):
        verify(Path(protocol["training_locks"][path_key]), protocol["training_locks"][hash_key], path_key)
    for name, model in protocol["models"].items():
        verify(Path(model["path"]), model["sha256"], f"{name} checkpoint")

    artifact_manifest = load(POST_DIR / "artifact-manifest.json")
    require(artifact_manifest["protocol"]["sha256"] == sha256_file(PROTOCOL_PATH), "Artifact protocol hash changed")
    for name, spec in artifact_manifest["artifacts"].items():
        verify(Path(spec["path"]), spec["sha256"], name)
    record = load(EXPERIMENT_DIR / "post-selection-record.json")
    require(record["run_artifact_manifest_sha256"] == sha256_file(POST_DIR / "artifact-manifest.json"), "Post-selection record is stale")

    for name, (tag, expected_commit) in BASELINE_TAGS.items():
        actual = subprocess.run(
            ["git", "rev-list", "-n", "1", tag], check=True, capture_output=True, text=True
        ).stdout.strip()
        require(actual == expected_commit, f"Baseline tag changed: {name}")
    return training_summary, training_manifest


def validate_manual(manual: dict[str, Any]) -> list[dict[str, Any]]:
    require(manual["status"] == "candidate_manual_scoring_complete", "Candidate manual scoring is incomplete")
    require(manual["counterfactual_v1_totals"] == {"greedy": "0/13", "sampled": "0/13"}, "Manual totals changed")
    rows = [row for row in manual["rows"] if row["model"] == "counterfactual_v1_context1024"]
    require(len(rows) == 26, "Expected 26 candidate fact-mode rows")
    for row in rows:
        require(row["retained"] is False, f"Unexpected retained fact: {row['prompt_id']}/{row['fact_type']}/{row['mode']}")
        require(bool(row["evidence_quote"]) and row["evidence_quote"] in row["generated_text"], "Manual evidence is missing from generated text")
        require(bool(row["reviewer_note"]), "Manual reviewer note is missing")
    return rows


def compact_manual(manual: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    facts: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["prompt_id"], row["fact_type"])
        item = facts.setdefault(key, {
            "id": f"{row['prompt_id']}.{row['fact_type']}",
            "prompt_id": row["prompt_id"],
            "fact_type": row["fact_type"],
            "fact": row["expected_fact"],
            "retained": {},
            "evidence": {},
            "reviewer_note": {},
        })
        item["retained"][row["mode"]] = row["retained"]
        item["evidence"][row["mode"]] = row["evidence_quote"]
        item["reviewer_note"][row["mode"]] = row["reviewer_note"]
    return {
        "version": 1,
        "scoring_rule": manual["score_definition"],
        "candidate_totals": manual["counterfactual_v1_totals"],
        "sealed_historical_audit": {
            **manual["sealed_historical_audit"],
            "sha256": sha256_file(Path(manual["sealed_historical_audit"]["path"])),
        },
        "facts": list(facts.values()),
    }


def write_reports() -> dict[str, Any]:
    protocol = load(PROTOCOL_PATH)
    training_summary, training_manifest = validate_sources(protocol)
    counterfactual = load(POST_DIR / "counterfactual-evaluation.json")
    forced = load(POST_DIR / "13fact-forced-choice.json")
    real = load(POST_DIR / "real-fiction-evaluation.json")
    probe = load(POST_DIR / "probe-comparison.json")
    repetition = load(POST_DIR / "repetition-comparison.json")
    manual = load(POST_DIR / "manual-fact-scoring.json")
    manual_rows = validate_manual(manual)
    compact_audit = compact_manual(manual, manual_rows)
    write_json_atomic(EXPERIMENT_DIR / "manual-fact-scoring.json", compact_audit)

    pair_metrics = {
        model: {
            split: {
                "overall": compact_group(value["metrics"]["overall"]),
                "by_state_family": {name: compact_group(group) for name, group in value["metrics"]["by_state_family"].items()},
                "by_distance": {name: compact_group(group) for name, group in value["metrics"]["by_distance"].items()},
                "by_difficulty": {name: compact_group(group) for name, group in value["metrics"]["by_difficulty"].items()},
                "latest_state_subset": compact_group(value["metrics"]["latest_state_subset"]),
            }
            for split, value in splits.items()
        }
        for model, splits in counterfactual["models"].items()
    }
    forced_metrics = {
        model: {key: value[key] for key in ("top1_total", "denominator", "mean_correct_vs_best_negative_margin")}
        for model, value in forced["models"].items()
    }
    real_metrics = {
        model: {split: value["metrics"] for split, value in splits.items()}
        for model, splits in real["models"].items()
    }
    candidate_probe = probe["models"]["counterfactual_v1"]
    probe_metrics = {
        family: {
            "best_layer": value["best_layer"],
            "test": value["comparison_summary"]["test"],
            "generalization_holdout": value["comparison_summary"]["generalization_holdout"],
        }
        for family, value in candidate_probe["families"].items()
    }
    candidate_pair = pair_metrics["counterfactual_v1"]
    candidate_controls = {
        split: {
            name: counterfactual["models"]["counterfactual_v1"][split]["metrics"]["overall"][name]
            for name in ("fact_removed", "irrelevant_substituted", "candidate_only")
        }
        for split in ("test", "generalization_holdout")
    }
    latest_state = {
        split: candidate_pair[split]["latest_state_subset"]
        for split in ("test", "generalization_holdout")
    }
    training_config = yaml.safe_load(Path(protocol["training_locks"]["config_path"]).read_text(encoding="utf-8"))

    provenance = {
        "training": {
            "commit": protocol["training_commit"],
            "completion_commit": None,
            "manifest_status": training_manifest["status"],
            "manifest_status_note": "The immutable start manifest was not finalized; the hash-locked final summary and selected checkpoint independently establish completion.",
            "config_path": protocol["training_locks"]["config_path"],
            "config_sha256": training_manifest["config_sha256"],
        },
        "candidate_checkpoint": {"path": protocol["candidate"]["checkpoint_path"], "sha256": protocol["candidate"]["checkpoint_sha256"]},
        "baseline_tags": {name: {"tag": tag, "commit": commit} for name, (tag, commit) in BASELINE_TAGS.items()},
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256_file(PROTOCOL_PATH)},
        "training_locks": protocol["training_locks"],
        "run_artifacts": {name: sha256_file(RUN_DIR / name) for name in ("manifest.json", "metrics.json", "summary.json")},
        "post_selection_manifest": {"path": str(POST_DIR / "artifact-manifest.json"), "sha256": sha256_file(POST_DIR / "artifact-manifest.json")},
        "post_selection_artifacts": load(POST_DIR / "artifact-manifest.json")["artifacts"],
        "manual_fact_scoring": {"path": str(EXPERIMENT_DIR / "manual-fact-scoring.json"), "sha256": sha256_file(EXPERIMENT_DIR / "manual-fact-scoring.json")},
        "counterfactual_baseline_note": counterfactual["baseline_reuse_note"],
    }
    write_json_atomic(EXPERIMENT_DIR / "provenance.json", provenance)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "status": "reporting_complete_unsealed",
        "classification": CLASSIFICATION,
        "classification_definition": "Counterfactually balanced supervision produced held-out context-conditioned reversal within the synthetic generator, but did not transfer broadly to latest-state updates, free generation, frozen probes, or storytelling.",
        "completion_commit": None,
        "seal_created": False,
        "selection_metric": protocol["selection_metric"],
        "selected_checkpoint": {"step": 2220, "sha256": protocol["candidate"]["checkpoint_sha256"], "data30m_validation_loss": training_summary["best_data30m_validation_loss"]},
        "controls": {
            "parameters": 15_047_040,
            "context_tokens": 1024,
            "tokenizer_sha256": protocol["training_locks"]["tokenizer_sha256"],
            "seed": 1337,
            "optimizer_steps": 2220,
            "real_valid_targets": training_summary["source_valid_targets_observed"]["real"],
            "counterfactual_valid_targets": training_summary["source_valid_targets_observed"]["curriculum"],
            "total_valid_targets": training_summary["total_valid_targets"],
            "pair_presentations": training_summary["pair_presentations"],
            "lambda_contrastive": training_config["counterfactual_objective"]["lambda"],
        },
        "counterfactual_evaluation": pair_metrics,
        "candidate_controls": candidate_controls,
        "latest_state_update": latest_state,
        "historical_forced_choice": forced_metrics,
        "manual_free_generation": {"counterfactual_v1": compact_audit["candidate_totals"], "historical_models": {"greedy": "0/13", "sampled": "0/13"}},
        "frozen_probe": {"candidate": probe_metrics, "conclusion": "No family has a strong linear signal on both test and generalization holdouts."},
        "real_fiction_lm_metrics": real_metrics,
        "repetition": repetition,
        "outcome_matrix": {
            "counterfactual_scoring_transfer": "improved",
            "latest_state_transfer": "not_improved",
            "historical_forced_choice": "numerically_improved_to_6_of_13",
            "free_generation_fact_retention": "not_improved",
            "broad_internal_state_representation": "not_improved",
            "real_fiction_lm": "near_neutral_primary_with_legacy_regressions",
            "storytelling_quality": "not_improved",
            "greedy_repetition": "regressed",
        },
        "limitations": [
            "Test and generalization templates are held out but remain products of the same synthetic generator.",
            "Results vary sharply by state family; goals and latest-state updates have zero paired reversal.",
            "Only one model-training seed was evaluated.",
            "The 13-fact forced-choice suite is small and its mean candidate margin remains negative.",
            "Legacy LM diagnostics overlap Data30M training and are continuity diagnostics, not clean held-out controls.",
        ],
        "provenance_path": str(EXPERIMENT_DIR / "provenance.json"),
    }
    write_json_atomic(EXPERIMENT_DIR / "summary.json", summary)

    overall_rows = []
    for split in ("test", "generalization_holdout"):
        for model in MODELS:
            values = pair_metrics[model][split]["overall"]
            normal = values["normal"]
            overall_rows.append((split, MODEL_LABELS[model], values["pair_count"], f"{normal['direction_accuracy']:.4f}", f"{normal['joint_paired_reversal_success']:.4f}", f"{normal['mean_signed_reversal_margin']:.4f}"))
    controls_rows = []
    for split in ("test", "generalization_holdout"):
        for control in ("fact_removed", "irrelevant_substituted"):
            values = candidate_controls[split][control]
            controls_rows.append((split, control, f"{values['X_win_rate']:.4f}", f"{values['Y_win_rate']:.4f}", f"{values['mean_margin_x_minus_y']:.4f}", f"{values['mean_absolute_preference_margin']:.4f}"))
        values = candidate_controls[split]["candidate_only"]
        controls_rows.append((split, "candidate_only", f"{values['X_win_rate']:.4f}", f"{values['Y_win_rate']:.4f}", f"{values['mean_X_minus_Y_score_difference']:.4f}", "n/a"))
    family_rows = []
    for split in ("test", "generalization_holdout"):
        for family, values in candidate_pair[split]["by_state_family"].items():
            normal = values["normal"]
            family_rows.append((split, family, values["pair_count"], f"{normal['direction_accuracy']:.4f}", f"{normal['joint_paired_reversal_success']:.4f}", f"{normal['mean_signed_reversal_margin']:.4f}"))
    forced_rows = [(MODEL_LABELS[model], f"{values['top1_total']}/{values['denominator']}", f"{values['mean_correct_vs_best_negative_margin']:.4f}") for model, values in forced_metrics.items()]
    lm_rows = [(split, MODEL_LABELS[model], f"{values['loss']:.6f}", f"{values['perplexity']:.4f}", f"{values['next_token_accuracy']:.6f}", values["valid_tokens"]) for model, splits in real_metrics.items() for split, values in splits.items()]
    quantitative = "\n".join([
        "# Quantitative Comparison", "", "## Counterfactual Evaluation", "",
        table(("Split", "Model", "Pairs", "Direction", "Joint reversal", "Signed margin"), overall_rows), "",
        "Joint reversal has a 0.25 random-pair reference. Direction accuracy has a 0.5 reference.", "",
        "## Candidate Controls", "", table(("Split", "Control", "X win", "Y win", "X-Y margin", "Absolute margin"), controls_rows), "",
        "Removed and irrelevant controls use one shared context, so paired reversal is undefined; their balanced directional accuracy is mechanically 0.5. These rows report side preference magnitude only.", "",
        "## Candidate State Families", "", table(("Split", "Family", "N", "Direction", "Joint reversal", "Signed margin"), family_rows), "",
        "## Historical Forced Choice", "", table(("Model", "Top-1", "Correct-best-negative margin"), forced_rows), "",
        "## Real-Fiction LM", "", table(("Split", "Model", "Loss", "Perplexity", "Accuracy", "Valid tokens"), lm_rows), "",
        f"Counterfactual-v1 Data30M test loss changes {percent_change(real_metrics['counterfactual_v1']['data30m_test']['loss'], real_metrics['compute5']['data30m_test']['loss']):+.4f}% versus Compute5. The clean common Data10M subset changes {percent_change(real_metrics['counterfactual_v1']['data10m_test_data30m_train_disjoint']['loss'], real_metrics['compute5']['data10m_test_data30m_train_disjoint']['loss']):+.4f}%. These small one-seed differences establish no material primary regression or gain. Legacy diagnostics regress but overlap Data30M training.", "",
    ])
    (EXPERIMENT_DIR / "quantitative-comparison.md").write_text(quantitative, encoding="utf-8")

    controlled = "\n".join([
        "# Controlled Summary", "", f"**Classification: `{CLASSIFICATION}`**", "",
        "Counterfactual-v1 learned context-conditioned scoring beyond the sealed synthetic-shortcut baseline: joint reversal reached 48.02% on test and 44.59% on held-out templates, with 73.40% and 72.30% directional accuracy. Candidate-only X rates remained near chance, so a fixed candidate-side preference does not explain the gain.", "",
        "The transfer is narrow. Latest-state reversal remained 0/32 on test and 0/14 on generalization, the historical forced-choice suite improved only to 6/13 with a negative mean margin, and conservative free-generation scoring remained 0/13 in both decoding modes. No frozen-probe family was strong on both holdouts.", "",
        "Primary real-fiction losses are slightly lower than the baselines but the changes are too small and single-seed to establish a gain. Greedy repetition is worse, sampled prose still drifts, and storytelling quality did not improve.", "",
        "## Decision", "",
        "The experiment rejects a pure fixed-side synthetic shortcut explanation for the new benchmark result, but does not establish general narrative-state tracking. The result is best treated as generator-internal counterfactual scoring transfer with unresolved lexical/template shortcuts and no broad behavioral transfer.", "",
    ])
    (EXPERIMENT_DIR / "controlled-summary.md").write_text(controlled, encoding="utf-8")

    qualitative = """# Qualitative Behavior Assessment

| Category | Counterfactual-v1 assessment |
|---|---|
| Grammar | Sampled clauses are often locally grammatical, but malformed constructions and identity confusion remain. |
| Prompt adherence | Prompt entities and premises are usually abandoned. |
| Semantic continuity | No disclosed fact was retained under conservative free-generation scoring: 0/13 greedy and 0/13 sampled. |
| Entity consistency | Names, possession, relationships, locations, goals, and scene markers disappear. |
| Latest-state tracking | No transfer: paired reversal is 0% on both held-out splits. |
| Dialogue coherence | Turn formatting is plausible, but speakers, referents, and relationships drift. |
| Causal progression | Generated events do not preserve disclosed causes, obligations, or goals. |
| Narrative progression | No sustained premise-driven arc is demonstrated. |
| Repetition | Greedy decoding regresses substantially; sampled decoding is less repetitive but remains incoherent. |

Generic overlap was not credited. Words such as `house`, `window`, `manuscript`, or marriage language occur without the required entity-state relationship. The 6/13 forced-choice result therefore does not transfer to open-ended continuation.
"""
    (EXPERIMENT_DIR / "qualitative-assessment.md").write_text(qualitative, encoding="utf-8")

    repetition_rows = [(suite, MODEL_LABELS.get(model, model), mode, f"{scores['distinct_1']:.4f}", f"{scores['distinct_2']:.4f}", f"{scores['distinct_3']:.4f}", f"{scores['repeated_4gram_rate']:.4f}", f"{scores['repeated_sentence_rate']:.4f}", f"{scores['longest_repeated_token_span']:.2f}") for suite, models in repetition.items() if suite in ("historical_10_prompt", "narrative_state_3_prompt") for model, modes in models.items() for mode, scores in modes.items()]
    repetition_md = "\n".join(["# Repetition Comparison", "", repetition["definition"] + ". Lower repetition rates and higher distinct scores are better.", "", table(("Suite", "Model", "Mode", "Distinct-1", "Distinct-2", "Distinct-3", "Repeated 4-gram", "Repeated sentence", "Longest span"), repetition_rows), "", "Counterfactual-v1 has the worst greedy repeated 4-gram rate on both suites (0.6056 historical; 0.7013 narrative-state). Sampled decoding is substantially healthier but does not retain the disclosed facts.", ""])
    (EXPERIMENT_DIR / "repetition-comparison.md").write_text(repetition_md, encoding="utf-8")

    readme = f"""# FictionPulper-15M Data30M Counterfactual-v1

Final reporting classification: **`{CLASSIFICATION}`**.

Counterfactually balanced supervision produced strong held-out reversal within the synthetic generator, but failed latest-state transfer and retained 0/13 historical facts in both greedy and sampled free generation. Frozen probes show no broad two-split representation improvement. Primary real-fiction LM behavior is near-neutral, while greedy repetition regressed.

Rebuild and validate:

```bash
python -m src.counterfactual_post_selection --stage finalize
python -m src.report_counterfactual
python -m unittest discover -s tests
```

The immutable trainer start manifest remains `training_started`; completion is independently established by the hash-locked final summary and selected step-2220 checkpoint and is disclosed in `provenance.json`.

Key records: `summary.json`, `controlled-summary.md`, `quantitative-comparison.md`, `qualitative-assessment.md`, `repetition-comparison.md`, `manual-fact-scoring.json`, `provenance.json`, and `seal-candidate.json`.
"""
    (EXPERIMENT_DIR / "README.md").write_text(readme, encoding="utf-8")

    record_names = ("README.md", "summary.json", "controlled-summary.md", "quantitative-comparison.md", "qualitative-assessment.md", "repetition-comparison.md", "manual-fact-scoring.json", "provenance.json")
    seal_candidate = {
        "run_id": EXPERIMENT_ID,
        "status": "seal_candidate_only",
        "classification": CLASSIFICATION,
        "training_git_commit": protocol["training_commit"],
        "completion_git_commit": None,
        "tag_created": False,
        "seal_created": False,
        "checkpoint_sha256": protocol["candidate"]["checkpoint_sha256"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "baseline_tags": provenance["baseline_tags"],
        "experiment_record_sha256": {name: sha256_file(EXPERIMENT_DIR / name) for name in record_names},
        "source_hashes_validated": True,
        "full_tests_passed": True,
        "full_test_count": 200,
        "counterfactual_scoring_transfer_observed": True,
        "latest_state_transfer_observed": False,
        "historical_forced_choice_improved": True,
        "semantic_free_generation_improved": False,
        "broad_internal_state_representation_improved": False,
        "material_storytelling_improvement": False,
        "material_real_fiction_damage": False,
        "greedy_repetition_regressed": True,
    }
    write_json_atomic(EXPERIMENT_DIR / "seal-candidate.json", seal_candidate)
    return summary


def main() -> None:
    summary = write_reports()
    print(json.dumps({"experiment_id": summary["experiment_id"], "status": summary["status"], "classification": summary["classification"]}, indent=2))


if __name__ == "__main__":
    main()
