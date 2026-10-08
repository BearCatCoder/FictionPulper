"""Build compact reports for the completed narrative-curriculum experiment.

This reporter reads completed artifacts only. It does not load model weights or
open held-out source JSONL, and it writes exclusively to the tracked experiment
directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.train_tokenizer import sha256_file, write_json_atomic


EXPERIMENT_ID = "fictionpulper-15m-data30m-narrative-v1"
EXPERIMENT_DIR = Path("experiments") / EXPERIMENT_ID
RUN_DIR = Path("runs") / EXPERIMENT_ID
POST_DIR = RUN_DIR / "post-selection"
BASELINE_RUN = Path("runs/fictionpulper-15m-data30m-compute5")
CURRICULUM_DIR = Path("data/narrative_curriculum_v1")
CHECKPOINT_DIR = Path("checkpoints") / EXPERIMENT_ID
STATE_GROUPS = ("by_state_type", "by_distance_bucket", "by_difficulty", "by_genre")
SOURCE_ARTIFACTS = (
    "evaluation-results.json",
    "curriculum-state-recall.json",
    "generation-comparison.json",
    "narrative-state-comparison.json",
    "repetition-comparison.json",
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def percent_change(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate / baseline - 1.0)


def metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in ("loss", "perplexity", "next_token_accuracy", "valid_tokens")
    }


def markdown_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    alignment = ("---",) + tuple("---:" for _ in headers[1:])
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(alignment) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def validate_sources(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    baseline: dict[str, Any],
    evaluation: dict[str, Any],
    recall: dict[str, Any],
    generation: dict[str, Any],
    narrative: dict[str, Any],
) -> None:
    require(summary["status"] == "training_and_selection_complete", "Training is incomplete")
    require(summary["selection_metric"] == "Data30M validation loss only", "Selection metric changed")
    require(summary["curriculum_validation_used_for_selection"] is False, "Curriculum validation affected selection")
    require(summary["best_checkpoint_step"] == baseline["best_checkpoint_epoch"] * 444 == 2220, "Selected steps differ")
    require(summary["total_valid_targets"] == baseline["total_valid_training_targets_processed"], "Target exposure differs")
    require(manifest["model_parameter_count"] == baseline["model_parameter_count"] == 15_047_040, "Model size differs")
    require(manifest["seed"] == 1337 and manifest["fresh_random_initialization"] is True, "Initialization control changed")
    require(evaluation["post_checkpoint_selection"] is True and evaluation["used_for_checkpoint_selection"] is False, "Evaluation was not post-selection")
    require(evaluation["checkpoint_sha256"] == summary["best_checkpoint_sha256"], "Evaluation checkpoint differs")
    require(generation["baseline_checkpoint_sha256"] == baseline["best_checkpoint_sha256"], "Baseline generation checkpoint differs")
    require(generation["candidate_checkpoint_sha256"] == summary["best_checkpoint_sha256"], "Candidate generation checkpoint differs")
    require(narrative["post_checkpoint_selection"] is True and narrative["used_for_checkpoint_selection"] is False, "Narrative diagnostic was not post-selection")
    require(len(generation["prompts"]) == 10, "Generation suite is not ten prompts")
    for split, expected_total in (("curriculum_test", 1405), ("curriculum_generalization_holdout", 618)):
        result = recall[split]
        require(result["aggregate"] == {"correct": 0, "total": expected_total, "accuracy": 0.0}, f"Unexpected recall result: {split}")
        for group_name in STATE_GROUPS:
            require(sum(item["total"] for item in result[group_name].values()) == expected_total, f"Incomplete recall grouping: {split}/{group_name}")
            require(all(item["correct"] == 0 and item["accuracy"] == 0.0 for item in result[group_name].values()), f"Nonzero grouped recall: {split}/{group_name}")


def build_provenance(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    baseline: dict[str, Any],
    baseline_manifest: dict[str, Any],
) -> dict[str, Any]:
    curriculum_manifest_path = CURRICULUM_DIR / "manifest.json"
    curriculum = load(curriculum_manifest_path)
    for name, details in curriculum["files"].items():
        require(sha256_file(CURRICULUM_DIR / name) == details["sha256"], f"Curriculum file hash changed: {name}")
    artifact_manifest = load(POST_DIR / "artifact-manifest.json")
    for name in SOURCE_ARTIFACTS:
        expected = artifact_manifest["artifacts"][name]["sha256"]
        require(sha256_file(POST_DIR / name) == expected, f"Post-selection artifact hash changed: {name}")

    checkpoint_hashes = {
        path.name: sha256_file(path)
        for path in sorted(CHECKPOINT_DIR.glob("*.pt"))
    }
    require(checkpoint_hashes["best-validation.pt"] == summary["best_checkpoint_sha256"], "Best checkpoint hash changed")
    return {
        "curriculum": {
            "config_sha256": curriculum["config_sha256"],
            "manifest_path": str(curriculum_manifest_path),
            "manifest_sha256": sha256_file(curriculum_manifest_path),
            "files": curriculum["files"],
        },
        "real_fiction_data": {
            "corpus_sha256": baseline_manifest["corpus_sha256"],
            "split_manifest_sha256": baseline_manifest["split_manifest_sha256"],
            "packed_metadata_sha256": manifest["real_artifacts"]["metadata_sha256"],
            "packed_artifacts": baseline_manifest["packed_artifacts"],
        },
        "tokenizer_sha256": baseline_manifest["tokenizer_hash"],
        "schedule": {
            "path": manifest["schedule_path"],
            "file_sha256": manifest["schedule_file_sha256"],
            "content_sha256": manifest["schedule_content_sha256"],
        },
        "training": {
            "commit": summary["training_git_commit"],
            "completion_commit": None,
            "config_sha256": manifest["config_sha256"],
        },
        "checkpoints": {
            "candidate": checkpoint_hashes,
            "compute5_best_validation": baseline["best_checkpoint_sha256"],
            "compute5_latest": baseline["final_checkpoint_sha256"],
        },
        "run_artifacts": {
            "manifest.json": sha256_file(RUN_DIR / "manifest.json"),
            "metrics.json": sha256_file(RUN_DIR / "metrics.json"),
            "summary.json": sha256_file(RUN_DIR / "summary.json"),
        },
        "post_selection_artifacts": {
            name: artifact_manifest["artifacts"][name] for name in SOURCE_ARTIFACTS
        },
        "protocols": {
            "post_selection": {
                "path": str(EXPERIMENT_DIR / "post-selection-protocol.json"),
                "sha256": sha256_file(EXPERIMENT_DIR / "post-selection-protocol.json"),
            },
            "evaluation": load(EXPERIMENT_DIR / "preflight.json")["sealed_evaluation_protocol"],
            "narrative_state": load(EXPERIMENT_DIR / "preflight.json")["narrative_state_protocol"],
        },
    }


def build_report() -> dict[str, Any]:
    summary = load(RUN_DIR / "summary.json")
    manifest = load(RUN_DIR / "manifest.json")
    baseline = load(BASELINE_RUN / "summary.json")
    baseline_manifest = load(BASELINE_RUN / "manifest.json")
    evaluation = load(POST_DIR / "evaluation-results.json")
    recall = load(POST_DIR / "curriculum-state-recall.json")
    generation = load(POST_DIR / "generation-comparison.json")
    narrative = load(POST_DIR / "narrative-state-comparison.json")
    repetition = load(POST_DIR / "repetition-comparison.json")
    validate_sources(summary, manifest, baseline, evaluation, recall, generation, narrative)
    provenance = build_provenance(summary, manifest, baseline, baseline_manifest)

    candidate_test = evaluation["metrics"]["data30m_test"]["metrics"]
    baseline_test = baseline["test_metrics"]
    real_metrics = {
        "compute5": metric_row(baseline_test),
        "narrative_curriculum": metric_row(candidate_test),
        "relative_change_percent": {
            key: percent_change(candidate_test[key], baseline_test[key])
            for key in ("loss", "perplexity", "next_token_accuracy")
        },
    }
    curriculum_metrics = {
        split: {
            "teacher_forced_lm": metric_row(evaluation["metrics"][split]["metrics"]),
            "exact_state_recall": {
                key: recall[split][key] for key in ("aggregate", *STATE_GROUPS)
            },
        }
        for split in ("curriculum_test", "curriculum_generalization_holdout")
    }
    semantic = {
        "method": "manual audit of the unchanged narrative-state outputs; 13 disclosed facts across three prompts",
        "candidate": {"greedy": {"correct": 0, "total": 13}, "sampled": {"correct": 0, "total": 13}},
        "compute5": {"greedy": {"correct": 0, "total": 13}, "sampled": {"correct": 0, "total": 13}},
    }
    resources = {
        "compute5": {
            "training_seconds": baseline["training_seconds"],
            "valid_targets_per_second": baseline["average_valid_tokens_per_second"],
            "peak_vram_gib": baseline["peak_gpu_memory_gb"],
        },
        "narrative_curriculum": {
            "training_seconds": summary["training_seconds"],
            "total_elapsed_seconds": summary["total_elapsed_seconds"],
            "valid_targets_per_second": summary["average_valid_targets_per_second"],
            "peak_vram_gib": summary["gpu_memory_peak_gb"],
        },
    }
    write_json_atomic(EXPERIMENT_DIR / "provenance.json", provenance)
    report = {
        "experiment_id": EXPERIMENT_ID,
        "status": "reporting_complete_unsealed",
        "completion_commit": None,
        "seal_created": False,
        "selection_metric": "Data30M validation loss only",
        "controlled_exposure": {
            "parameters": 15_047_040,
            "context": 1024,
            "seed": 1337,
            "total_valid_target_presentations": 135_886_355,
            "compute5_mix": {"real_fiction_percent": 100.0, "curriculum_percent": 0.0},
            "candidate_mix": summary["source_valid_target_percentages"],
        },
        "real_fiction_lm_metrics": real_metrics,
        "synthetic_curriculum_metrics": curriculum_metrics,
        "narrative_semantic_diagnostic": semantic,
        "repetition": {
            "report_path": str(EXPERIMENT_DIR / "repetition-comparison.md"),
            "source_sha256": provenance["post_selection_artifacts"]["repetition-comparison.json"]["sha256"],
        },
        "resources": resources,
        "curriculum_damage_assessment": {
            "data30m_test_loss_change_percent": real_metrics["relative_change_percent"]["loss"],
            "data30m_test_perplexity_change_percent": real_metrics["relative_change_percent"]["perplexity"],
            "data30m_test_accuracy_change_percent": real_metrics["relative_change_percent"]["next_token_accuracy"],
            "conclusion": "No material real-fiction LM damage: changes are below 0.3% in loss, perplexity, and accuracy, but no storytelling or state-retention benefit was demonstrated.",
        },
        "hypothesis_result": "failed: curriculum LM structure was learned, but exact state recall and manually audited semantic state retention were both zero",
        "provenance": {
            "path": str(EXPERIMENT_DIR / "provenance.json"),
            "sha256": sha256_file(EXPERIMENT_DIR / "provenance.json"),
        },
    }
    write_json_atomic(EXPERIMENT_DIR / "summary.json", report)
    write_json_atomic(
        EXPERIMENT_DIR / "post-selection-record.json",
        {
            "status": "post_selection_evaluation_complete",
            "used_for_checkpoint_selection": False,
            "selection_metric": report["selection_metric"],
            "checkpoint_sha256": summary["best_checkpoint_sha256"],
            "real_fiction_lm_metrics": real_metrics,
            "synthetic_curriculum_lm_metrics": {
                split: values["teacher_forced_lm"]
                for split, values in curriculum_metrics.items()
            },
            "exact_state_recall_aggregate": {
                split: values["exact_state_recall"]["aggregate"]
                for split, values in curriculum_metrics.items()
            },
            "grouped_state_recall_record": str(EXPERIMENT_DIR / "summary.json"),
            "narrative_semantic_diagnostic": semantic,
            "source_artifact_manifest_sha256": sha256_file(POST_DIR / "artifact-manifest.json"),
        },
    )

    comparison_rows = [
        ("Parameters", 15_047_040, 15_047_040, "0"),
        ("Context tokens", 1024, 1024, "0"),
        ("Seed", 1337, 1337, "0"),
        ("Valid target presentations", 135_886_355, summary["total_valid_targets"], "0"),
        ("Real / curriculum target mix", "100% / 0%", "85% / 15%", "controlled variable"),
        ("Selected step", 2220, summary["best_checkpoint_step"], "0"),
        ("Data30M validation loss", baseline["best_validation_loss"], summary["best_data30m_validation_loss"], f"{percent_change(summary['best_data30m_validation_loss'], baseline['best_validation_loss']):+.4f}%"),
        ("Data30M test loss", baseline_test["loss"], candidate_test["loss"], f"{real_metrics['relative_change_percent']['loss']:+.4f}%"),
        ("Data30M test perplexity", baseline_test["perplexity"], candidate_test["perplexity"], f"{real_metrics['relative_change_percent']['perplexity']:+.4f}%"),
        ("Data30M test next-token accuracy", baseline_test["next_token_accuracy"], candidate_test["next_token_accuracy"], f"{real_metrics['relative_change_percent']['next_token_accuracy']:+.4f}%"),
        ("Narrative facts retained, greedy", "0/13", "0/13", "0"),
        ("Narrative facts retained, sampled", "0/13", "0/13", "0"),
        ("Training seconds", baseline["training_seconds"], summary["training_seconds"], f"{percent_change(summary['training_seconds'], baseline['training_seconds']):+.2f}%"),
        ("Valid targets / second", baseline["average_valid_tokens_per_second"], summary["average_valid_targets_per_second"], f"{percent_change(summary['average_valid_targets_per_second'], baseline['average_valid_tokens_per_second']):+.2f}%"),
        ("Peak VRAM GiB", baseline["peak_gpu_memory_gb"], summary["gpu_memory_peak_gb"], f"{percent_change(summary['gpu_memory_peak_gb'], baseline['peak_gpu_memory_gb']):+.2f}%"),
    ]
    quantitative = "\n".join((
        "# Controlled Comparison",
        "",
        "Compute5 and the candidate use the same architecture, tokenizer, seed, context, optimizer schedule, selected step, and total valid-target exposure. The controlled variable is replacement of 15% of targets with Narrative Continuity Curriculum v1.",
        "",
        markdown_table(("Metric", "15M Compute5", "15M Narrative", "Candidate change"), comparison_rows),
        "",
        "Checkpoint selection used Data30M validation loss only. All test, generation, repetition, and state diagnostics were post-selection.",
        "",
    ))
    (EXPERIMENT_DIR / "quantitative-comparison.md").write_text(quantitative, encoding="utf-8")

    curriculum_lines = [
        "# Curriculum Metrics", "",
        "Teacher-forced next-token metrics and continuation state recall measure different behavior and are reported separately.", "",
    ]
    for split, title in (("curriculum_test", "Test"), ("curriculum_generalization_holdout", "Generalization Holdout")):
        item = curriculum_metrics[split]
        lm = item["teacher_forced_lm"]
        curriculum_lines.extend([
            f"## {title}", "",
            markdown_table(("LM loss", "Perplexity", "Next-token accuracy", "Valid tokens"), [(lm["loss"], lm["perplexity"], lm["next_token_accuracy"], lm["valid_tokens"])]), "",
            "Exact resolution recall requires complete token-ID equality from the resolution boundary. It is not teacher forced.", "",
            markdown_table(("Grouping", "Value", "Correct", "Total", "Accuracy"), [
                ("aggregate", "all", item["exact_state_recall"]["aggregate"]["correct"], item["exact_state_recall"]["aggregate"]["total"], item["exact_state_recall"]["aggregate"]["accuracy"]),
                *((group.replace("by_", ""), value, score["correct"], score["total"], score["accuracy"]) for group in STATE_GROUPS for value, score in item["exact_state_recall"][group].items()),
            ]), "",
        ])
    curriculum_lines.extend([
        "## Interpretation", "",
        "The low curriculum loss and roughly 48-51% teacher-forced accuracy show that the model learned synthetic curriculum token patterns. Exact recall remains zero in every state type, distance, difficulty, and genre group, including the shortest and easiest groups. The exact metric can reject valid paraphrases, but the separate human semantic audit also retained 0/13 facts under both greedy and sampled decoding. Therefore the zero is not explained only by strict wording, and the narrative-state hypothesis failed.", "",
    ])
    (EXPERIMENT_DIR / "curriculum-metrics.md").write_text("\n".join(curriculum_lines), encoding="utf-8")

    qualitative = """# Qualitative Assessment

## Verdict

**The curriculum learned as a language-modeling distribution but did not improve narrative state retention.** The candidate remains a locally plausible but weak story generator. Sampled outputs are less repetitive than greedy outputs, yet both models abandon premises, entities, goals, and disclosed facts. The candidate provides no material qualitative improvement over Compute5.

| Category | Candidate versus Compute5 |
|---|---|
| Grammar | Similar. Sampled passages often have locally grammatical clauses, but agreement and malformed constructions remain. |
| Complete sentences | Similar. Fixed-length continuations frequently stop mid-sentence. |
| Prompt adherence | No material improvement. Occasional prompt vocabulary survives, while core premises are usually abandoned. |
| Semantic continuity | No improvement. Local transitions can read smoothly, but events and attributes conflict or drift. |
| Dialogue coherence | No reliable improvement. Speaker turns are formatted plausibly but are circular or unrelated. |
| Repetition | Mixed. Candidate repetition improves modestly on aggregate, especially sampled output, but greedy loops remain severe. |
| Entity consistency | No demonstrated improvement; names, roles, possession, relationships, and locations are omitted or replaced. |
| Scene persistence | No reliable improvement. A scene may persist locally without preserving the requested setting or facts. |
| Causal progression | No improvement. Continuations juxtapose events rather than developing consequences from prior state. |
| Narrative progression | No material improvement. The ten prompts do not develop sustained premise-driven arcs. |

## Narrative Diagnostic

The unchanged three-prompt protocol contains 13 facts, all visible within the 1,024-token context. Manual audit found candidate greedy `0/13`, candidate sampled `0/13`, Compute5 greedy `0/13`, and Compute5 sampled `0/13`. Generic word overlap was not credited without the required entity-state relation. This semantic result independently confirms the token-exact held-out result rather than attributing failure solely to exact wording.

## Curriculum Damage

Replacing 15% of real-fiction targets caused no material real-fiction LM damage: Data30M test loss improved about 0.09%, perplexity improved about 0.29%, and next-token accuracy improved about 0.18%. Those tiny changes are effectively neutral at one run per condition and do not establish a benefit. Qualitative storytelling and semantic state retention also did not improve, so the curriculum consumed target exposure without achieving its intended behavior.
"""
    (EXPERIMENT_DIR / "qualitative-assessment.md").write_text(qualitative, encoding="utf-8")

    repetition_rows = []
    for suite, values in repetition.items():
        if suite == "definition":
            continue
        for model, modes in values.items():
            for mode, scores in modes.items():
                repetition_rows.append((suite, model, mode, scores["distinct_1"], scores["distinct_2"], scores["distinct_3"], scores["repeated_4gram_rate"], scores["repeated_sentence_rate"], scores["longest_repeated_token_span"]))
    repetition_md = "\n".join((
        "# Repetition Comparison", "",
        "Tokenizer-v1 metrics over generated continuations only; prompts are excluded. Lower repeated rates and higher distinct scores are better.", "",
        markdown_table(("Suite", "Model", "Mode", "Distinct-1", "Distinct-2", "Distinct-3", "Repeated 4-gram", "Repeated sentence", "Longest repeated span"), repetition_rows), "",
        "The candidate is modestly less repetitive on most aggregate rates, especially sampled output, but greedy repetition remains severe and this did not translate into state retention or narrative progression.", "",
    ))
    (EXPERIMENT_DIR / "repetition-comparison.md").write_text(repetition_md, encoding="utf-8")

    preflight = load(EXPERIMENT_DIR / "preflight.json")
    preflight["status"] = "reporting_complete_unsealed"
    preflight["reporting"] = {
        "source_artifacts_verified": True,
        "post_selection_complete": True,
        "manual_narrative_audit_recorded": True,
        "completion_commit": None,
        "seal_created": False,
    }
    write_json_atomic(EXPERIMENT_DIR / "preflight.json", preflight)
    return report


def main() -> None:
    report = build_report()
    print(json.dumps({
        "experiment_id": report["experiment_id"],
        "status": report["status"],
        "hypothesis_result": report["hypothesis_result"],
    }, indent=2))


if __name__ == "__main__":
    main()
