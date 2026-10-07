"""Build the controlled 15M Data10M versus Data30M experiment report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.train import GENERATION_SETTINGS
from src.train_tokenizer import write_json_atomic


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(evaluations: dict[str, Any], name: str) -> dict[str, Any]:
    return evaluations["historical_post_selection_evaluations"][name]["metrics"]


def percent_change(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate / baseline - 1.0)


def build_report(run_dir: Path, experiment_dir: Path) -> dict[str, Any]:
    baseline = load(Path("experiments/fictionpulper-15m-data10m-v1/summary.json"))
    baseline_run = load(Path("runs/fictionpulper-15m-data10m-v1/summary.json"))
    candidate = load(run_dir / "summary.json")
    evaluations = load(run_dir / "sealed-evaluations.json")
    baseline_common_validation = load(
        run_dir / "post-hoc/data10m-baseline-common-validation.json"
    )
    baseline_common_test = load(run_dir / "post-hoc/data10m-baseline-common-test.json")
    data10m_common_validation = metric(
        evaluations, "data10m_validation_data30m_train_disjoint"
    )
    data10m_common_test = metric(evaluations, "data10m_test_data30m_train_disjoint")
    data10m_full_test = metric(evaluations, "data10m_test_full")
    legacy_clean = metric(evaluations, "legacy_seed_validation_leakage_clean")
    legacy_test = metric(evaluations, "legacy_seed_test")

    quantitative = {
        "run_id": "fictionpulper-15m-data30m-v1",
        "experimental_variable": "amount and diversity of unique training fiction",
        "fixed_controls": {
            "parameters": 15047040,
            "context": 1024,
            "tokenizer": "v1",
            "tokenizer_sha256": "14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012",
            "seed": 1337,
            "optimizer": "AdamW",
            "peak_learning_rate": 0.001,
            "minimum_learning_rate": 0.0001,
            "weight_decay": 0.1,
            "gradient_clip": 1.0,
            "precision": "bf16",
        },
        "data10m": {
            "stories": 1565,
            "unique_training_targets": 8779154,
            "selected_epoch": 7,
            "target_presentations_to_selected_checkpoint": 61454078,
            "best_native_validation_loss": baseline["best_validation_loss"],
            "best_native_validation_perplexity": baseline["best_validation_perplexity"],
            "data10m_full_test": {
                "loss": baseline["corpus_v1_test_loss"],
                "perplexity": baseline["corpus_v1_test_perplexity"],
                "next_token_accuracy": baseline["corpus_v1_test_accuracy"],
                "classification": "original held-out test",
            },
            "common_95_record_test": baseline_common_test["metrics"],
            "common_91_record_validation": baseline_common_validation["metrics"],
            "legacy_clean_validation_loss": baseline[
                "legacy_seed_validation_leakage_clean_loss"
            ],
            "legacy_test_loss": baseline["legacy_seed_test_loss"],
            "legacy_test_perplexity": baseline["legacy_seed_test_perplexity"],
            "legacy_test_accuracy": baseline["legacy_seed_test_accuracy"],
            "training_seconds": baseline["training_seconds"],
            "total_elapsed_seconds": baseline_run["total_elapsed_seconds"],
            "valid_targets_per_second": baseline["average_valid_targets_per_second"],
            "peak_vram_gb": baseline["peak_gpu_memory_gb"],
        },
        "data30m": {
            "stories": 3707,
            "unique_training_targets": 27177271,
            "selected_epoch": candidate["best_validation_epoch"],
            "target_presentations_to_selected_checkpoint": candidate[
                "total_valid_training_targets_processed"
            ],
            "best_native_validation_loss": candidate["best_validation_loss"],
            "best_native_validation_perplexity": candidate[
                "best_validation_perplexity"
            ],
            "native_data30m_test": evaluations["primary_data30m_test"]["metrics"],
            "data10m_full_test": {
                **data10m_full_test,
                "classification": "identical records, but 45 were seen in Data30M train",
            },
            "common_95_record_test": data10m_common_test,
            "common_91_record_validation": data10m_common_validation,
            "legacy_clean_validation_loss": legacy_clean["loss"],
            "legacy_clean_validation_classification": (
                "historical definition; 48 records were seen in Data30M train"
            ),
            "legacy_test_loss": legacy_test["loss"],
            "legacy_test_perplexity": legacy_test["perplexity"],
            "legacy_test_accuracy": legacy_test["next_token_accuracy"],
            "legacy_test_classification": (
                "historical definition; 45 records were seen in Data30M train"
            ),
            "training_seconds": candidate["training_seconds"],
            "total_elapsed_seconds": candidate["total_elapsed_seconds"],
            "valid_targets_per_second": candidate[
                "average_valid_tokens_per_second"
            ],
            "peak_vram_gb": candidate["peak_gpu_memory_gb"],
        },
        "exposure": candidate["exposure"],
        "direct_common_held_out_changes_percent": {
            "test_loss": percent_change(
                data10m_common_test["loss"], baseline_common_test["metrics"]["loss"]
            ),
            "test_perplexity": percent_change(
                data10m_common_test["perplexity"],
                baseline_common_test["metrics"]["perplexity"],
            ),
            "test_accuracy": percent_change(
                data10m_common_test["next_token_accuracy"],
                baseline_common_test["metrics"]["next_token_accuracy"],
            ),
            "validation_loss": percent_change(
                data10m_common_validation["loss"],
                baseline_common_validation["metrics"]["loss"],
            ),
        },
        "comparability_note": (
            "Native validation sets differ and are not directly compared as identical benchmarks. "
            "The full Data10M test uses identical records but is exposure-contaminated for Data30M. "
            "The 95-record Data10M test subset is disjoint from both training sets and is the primary "
            "direct token-level comparison."
        ),
    }
    write_json_atomic(run_dir / "quantitative-comparison.json", quantitative)

    rows = [
        ("Parameters", 15047040, 15047040),
        ("Context", 1024, 1024),
        ("Tokenizer", "v1", "v1"),
        ("Unique training targets", 8779154, 27177271),
        ("Stories", 1565, 3707),
        ("Target presentations to selected checkpoint", 61454078, 81531813),
        ("Best native validation loss (different sets)", baseline["best_validation_loss"], candidate["best_validation_loss"]),
        ("Data10M full-test loss (45 Data30M exposures)", baseline["corpus_v1_test_loss"], data10m_full_test["loss"]),
        ("Data10M full-test PPL (45 Data30M exposures)", baseline["corpus_v1_test_perplexity"], data10m_full_test["perplexity"]),
        ("Data10M full-test accuracy (45 Data30M exposures)", baseline["corpus_v1_test_accuracy"], data10m_full_test["next_token_accuracy"]),
        ("Common 95-record test loss", baseline_common_test["metrics"]["loss"], data10m_common_test["loss"]),
        ("Common 95-record test PPL", baseline_common_test["metrics"]["perplexity"], data10m_common_test["perplexity"]),
        ("Common 95-record test accuracy", baseline_common_test["metrics"]["next_token_accuracy"], data10m_common_test["next_token_accuracy"]),
        ("Legacy clean validation loss (Data30M contaminated)", baseline["legacy_seed_validation_leakage_clean_loss"], legacy_clean["loss"]),
        ("Legacy test loss (Data30M contaminated)", baseline["legacy_seed_test_loss"], legacy_test["loss"]),
        ("Legacy test PPL (Data30M contaminated)", baseline["legacy_seed_test_perplexity"], legacy_test["perplexity"]),
        ("Legacy test accuracy (Data30M contaminated)", baseline["legacy_seed_test_accuracy"], legacy_test["next_token_accuracy"]),
        ("Training runtime seconds", baseline["training_seconds"], candidate["training_seconds"]),
        ("Valid targets/second", baseline["average_valid_targets_per_second"], candidate["average_valid_tokens_per_second"]),
        ("Peak VRAM GiB", baseline["peak_gpu_memory_gb"], candidate["peak_gpu_memory_gb"]),
    ]
    lines = [
        "# Quantitative Comparison",
        "",
        "| Metric | 15M Data10M | 15M Data30M |",
        "|---|---:|---:|",
        *[f"| {label} | {left} | {right} |" for label, left, right in rows],
        "",
        quantitative["comparability_note"],
    ]
    markdown = "\n".join(lines) + "\n"
    (run_dir / "quantitative-comparison.md").write_text(markdown, encoding="utf-8")
    (experiment_dir / "quantitative-comparison.md").write_text(markdown, encoding="utf-8")

    baseline_generations = load(
        Path("runs/fictionpulper-15m-data10m-v1/generations/best-validation-final.json")
    )
    candidate_generations = load(run_dir / "generations/best-validation-final.json")
    baseline_prompts = [entry["prompt"] for entry in baseline_generations["entries"]]
    candidate_prompts = [entry["prompt"] for entry in candidate_generations["entries"]]
    if baseline_prompts != candidate_prompts:
        raise RuntimeError("Generation prompt suites differ")
    if baseline_generations["sample_seed"] != candidate_generations["sample_seed"]:
        raise RuntimeError("Generation sampling seeds differ")
    if candidate_generations["generation_settings"] != GENERATION_SETTINGS:
        raise RuntimeError("Candidate generation settings differ from the locked protocol")
    generation_comparison = {
        "settings": {
            **GENERATION_SETTINGS,
            "sample_seed": candidate_generations["sample_seed"],
            "effective_sample_seed_policy": "sample_seed + prompt_index",
        },
        "data10m_checkpoint": {
            "epoch": baseline_generations["epoch"],
            "optimizer_step": baseline_generations["optimizer_step"],
        },
        "data30m_checkpoint": {
            "epoch": candidate_generations["epoch"],
            "optimizer_step": candidate_generations["optimizer_step"],
        },
        "entries": [
            {
                "prompt": baseline_entry["prompt"],
                "sample_seed": candidate_entry["sample_seed"],
                "data10m_greedy": baseline_entry["greedy"],
                "data30m_greedy": candidate_entry["greedy"],
                "data10m_sampled": baseline_entry["sampled"],
                "data30m_sampled": candidate_entry["sampled"],
            }
            for baseline_entry, candidate_entry in zip(
                baseline_generations["entries"],
                candidate_generations["entries"],
                strict=True,
            )
        ],
    }
    write_json_atomic(run_dir / "generation-comparison.json", generation_comparison)
    generation_lines = ["# Generation Comparison", ""]
    for index, entry in enumerate(generation_comparison["entries"], start=1):
        generation_lines.extend(
            [
                f"## Prompt {index}",
                "",
                f"> {entry['prompt']}",
                "",
                "### Data10M Greedy",
                "",
                entry["data10m_greedy"],
                "",
                "### Data30M Greedy",
                "",
                entry["data30m_greedy"],
                "",
                "### Data10M Sampled",
                "",
                entry["data10m_sampled"],
                "",
                "### Data30M Sampled",
                "",
                entry["data30m_sampled"],
                "",
            ]
        )
    generation_markdown = "\n".join(generation_lines)
    (run_dir / "generation-comparison.md").write_text(
        generation_markdown, encoding="utf-8"
    )
    (experiment_dir / "generation-comparison.md").write_text(
        generation_markdown, encoding="utf-8"
    )

    losses = [metric["validation_loss"] for metric in candidate["metrics"][1:]]
    learning_curve = {
        "classification": "validation improving but flattening",
        "epoch_validation_losses": {str(index + 1): loss for index, loss in enumerate(losses)},
        "loss_improvements": {
            "epoch_1_to_2": losses[0] - losses[1],
            "epoch_2_to_3": losses[1] - losses[2],
        },
        "interpretation": (
            "Validation improved at every epoch, but the absolute improvement shrank from epoch 2 "
            "to epoch 3. Additional optimization may help, but this experiment stops at epoch 3."
        ),
        "training_extended": False,
    }
    write_json_atomic(run_dir / "learning-curve-assessment.json", learning_curve)
    return quantitative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/fictionpulper-15m-data30m-v1")
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments/fictionpulper-15m-data30m-v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.run_dir, args.experiment_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
