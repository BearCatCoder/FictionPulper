"""Report the Data30M three-epoch versus five-epoch compute experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.train import GENERATION_SETTINGS
from src.train_tokenizer import write_json_atomic


BASELINE_RUN = Path("runs/fictionpulper-15m-data30m-v1")
CANDIDATE_RUN = Path("runs/fictionpulper-15m-data30m-compute5")
EXPERIMENT_DIR = Path("experiments/fictionpulper-15m-data30m-compute5")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def change(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate / baseline - 1.0)


def historical(evaluations: dict[str, Any], name: str) -> dict[str, Any]:
    return evaluations["historical_post_selection_evaluations"][name]["metrics"]


def build_report() -> dict[str, Any]:
    baseline = load(BASELINE_RUN / "summary.json")
    candidate = load(CANDIDATE_RUN / "summary.json")
    baseline_evaluations = load(BASELINE_RUN / "sealed-evaluations.json")
    candidate_evaluations = load(CANDIDATE_RUN / "sealed-evaluations.json")
    baseline_common = historical(
        baseline_evaluations, "data10m_test_data30m_train_disjoint"
    )
    candidate_common = historical(
        candidate_evaluations, "data10m_test_data30m_train_disjoint"
    )
    baseline_legacy_validation = historical(
        baseline_evaluations, "legacy_seed_validation_leakage_clean"
    )
    candidate_legacy_validation = historical(
        candidate_evaluations, "legacy_seed_validation_leakage_clean"
    )
    baseline_legacy_test = historical(baseline_evaluations, "legacy_seed_test")
    candidate_legacy_test = historical(candidate_evaluations, "legacy_seed_test")

    report = {
        "run_id": "fictionpulper-15m-data30m-compute5",
        "experimental_variable": "training duration from 3 to 5 document/chunk epochs",
        "fixed_controls": {
            "parameters": 15047040,
            "unique_train_targets": 27177271,
            "train_stories": 3707,
            "context": 1024,
            "tokenizer": "v1",
            "seed": 1337,
            "optimizer": "AdamW",
            "peak_learning_rate": 0.001,
            "minimum_learning_rate": 0.0001,
            "weight_decay": 0.1,
            "gradient_clip": 1.0,
            "precision": "bf16",
        },
        "data30m_3_epoch": {
            "epochs": 3,
            "target_presentations": baseline["total_valid_training_targets_processed"],
            "best_epoch": baseline["best_validation_epoch"],
            "best_validation_loss": baseline["best_validation_loss"],
            "data30m_test": baseline["test_metrics"],
            "common_held_out_test": baseline_common,
            "legacy_clean_validation": baseline_legacy_validation,
            "legacy_test": baseline_legacy_test,
            "training_seconds": baseline["training_seconds"],
            "throughput": baseline["average_valid_tokens_per_second"],
            "peak_vram_gb": baseline["peak_gpu_memory_gb"],
        },
        "data30m_compute5": {
            "epochs": 5,
            "target_presentations": candidate["total_valid_training_targets_processed"],
            "best_epoch": candidate["best_validation_epoch"],
            "best_validation_loss": candidate["best_validation_loss"],
            "data30m_test": candidate["test_metrics"],
            "common_held_out_test": candidate_common,
            "legacy_clean_validation": candidate_legacy_validation,
            "legacy_test": candidate_legacy_test,
            "training_seconds": candidate["training_seconds"],
            "throughput": candidate["average_valid_tokens_per_second"],
            "peak_vram_gb": candidate["peak_gpu_memory_gb"],
        },
        "relative_changes_percent": {
            "target_presentations": change(
                candidate["total_valid_training_targets_processed"],
                baseline["total_valid_training_targets_processed"],
            ),
            "best_validation_loss": change(
                candidate["best_validation_loss"], baseline["best_validation_loss"]
            ),
            "data30m_test_loss": change(
                candidate["test_loss"], baseline["test_loss"]
            ),
            "data30m_test_perplexity": change(
                candidate["test_perplexity"], baseline["test_perplexity"]
            ),
            "data30m_test_accuracy": change(
                candidate["test_next_token_accuracy"],
                baseline["test_next_token_accuracy"],
            ),
            "common_held_out_loss": change(
                candidate_common["loss"], baseline_common["loss"]
            ),
            "common_held_out_perplexity": change(
                candidate_common["perplexity"], baseline_common["perplexity"]
            ),
            "common_held_out_accuracy": change(
                candidate_common["next_token_accuracy"],
                baseline_common["next_token_accuracy"],
            ),
            "runtime": change(candidate["training_seconds"], baseline["training_seconds"]),
            "throughput": change(
                candidate["average_valid_tokens_per_second"],
                baseline["average_valid_tokens_per_second"],
            ),
            "peak_vram": change(
                candidate["peak_gpu_memory_gb"], baseline["peak_gpu_memory_gb"]
            ),
        },
    }
    write_json_atomic(CANDIDATE_RUN / "quantitative-comparison.json", report)

    rows = [
        ("Parameters", 15047040, 15047040),
        ("Unique train targets", 27177271, 27177271),
        ("Context", 1024, 1024),
        ("Tokenizer", "v1", "v1"),
        ("Epochs", 3, 5),
        ("Target presentations", 81531813, 135886355),
        ("Best epoch", baseline["best_validation_epoch"], candidate["best_validation_epoch"]),
        ("Best validation loss", baseline["best_validation_loss"], candidate["best_validation_loss"]),
        ("Data30M test loss", baseline["test_loss"], candidate["test_loss"]),
        ("Data30M test PPL", baseline["test_perplexity"], candidate["test_perplexity"]),
        ("Data30M test accuracy", baseline["test_next_token_accuracy"], candidate["test_next_token_accuracy"]),
        ("Common held-out loss", baseline_common["loss"], candidate_common["loss"]),
        ("Common held-out PPL", baseline_common["perplexity"], candidate_common["perplexity"]),
        ("Common held-out accuracy", baseline_common["next_token_accuracy"], candidate_common["next_token_accuracy"]),
        ("Legacy clean validation loss", baseline_legacy_validation["loss"], candidate_legacy_validation["loss"]),
        ("Legacy test loss", baseline_legacy_test["loss"], candidate_legacy_test["loss"]),
        ("Runtime seconds", baseline["training_seconds"], candidate["training_seconds"]),
        ("Valid targets/second", baseline["average_valid_tokens_per_second"], candidate["average_valid_tokens_per_second"]),
        ("Peak VRAM GiB", baseline["peak_gpu_memory_gb"], candidate["peak_gpu_memory_gb"]),
    ]
    lines = [
        "# Quantitative Comparison",
        "",
        "| Metric | Data30M 3-Epoch | Data30M Compute5 |",
        "|---|---:|---:|",
        *[f"| {label} | {left} | {right} |" for label, left, right in rows],
        "",
        "Both runs use identical data, tokenizer, architecture, initialization seed, and evaluation definitions. The five-epoch run uses a fresh cosine schedule and does not resume three-epoch weights.",
    ]
    markdown = "\n".join(lines) + "\n"
    (CANDIDATE_RUN / "quantitative-comparison.md").write_text(markdown, encoding="utf-8")
    (EXPERIMENT_DIR / "quantitative-comparison.md").write_text(
        markdown, encoding="utf-8"
    )

    baseline_generations = load(BASELINE_RUN / "generations/best-validation-final.json")
    candidate_generations = load(CANDIDATE_RUN / "generations/best-validation-final.json")
    if [entry["prompt"] for entry in baseline_generations["entries"]] != [
        entry["prompt"] for entry in candidate_generations["entries"]
    ]:
        raise RuntimeError("Generation prompt suites differ")
    if baseline_generations["generation_settings"] != GENERATION_SETTINGS:
        raise RuntimeError("Baseline generation settings changed")
    if candidate_generations["generation_settings"] != GENERATION_SETTINGS:
        raise RuntimeError("Candidate generation settings changed")
    if baseline_generations["sample_seed"] != candidate_generations["sample_seed"]:
        raise RuntimeError("Generation base seeds differ")
    generation = {
        "settings": {
            **GENERATION_SETTINGS,
            "sample_seed": candidate_generations["sample_seed"],
            "effective_seed_policy": "sample_seed + prompt_index",
        },
        "baseline_checkpoint": {
            "epoch": baseline_generations["epoch"],
            "step": baseline_generations["optimizer_step"],
        },
        "compute5_checkpoint": {
            "epoch": candidate_generations["epoch"],
            "step": candidate_generations["optimizer_step"],
        },
        "entries": [
            {
                "prompt": left["prompt"],
                "sample_seed": right["sample_seed"],
                "three_epoch_greedy": left["greedy"],
                "compute5_greedy": right["greedy"],
                "three_epoch_sampled": left["sampled"],
                "compute5_sampled": right["sampled"],
            }
            for left, right in zip(
                baseline_generations["entries"],
                candidate_generations["entries"],
                strict=True,
            )
        ],
    }
    write_json_atomic(CANDIDATE_RUN / "generation-comparison.json", generation)
    generation_lines = ["# Generation Comparison", ""]
    for index, entry in enumerate(generation["entries"], start=1):
        generation_lines.extend(
            [
                f"## Prompt {index}",
                "",
                f"> {entry['prompt']}",
                "",
                "### Three-Epoch Greedy",
                "",
                entry["three_epoch_greedy"],
                "",
                "### Compute5 Greedy",
                "",
                entry["compute5_greedy"],
                "",
                "### Three-Epoch Sampled",
                "",
                entry["three_epoch_sampled"],
                "",
                "### Compute5 Sampled",
                "",
                entry["compute5_sampled"],
                "",
            ]
        )
    generation_markdown = "\n".join(generation_lines)
    (CANDIDATE_RUN / "generation-comparison.md").write_text(
        generation_markdown, encoding="utf-8"
    )
    (EXPERIMENT_DIR / "generation-comparison.md").write_text(
        generation_markdown, encoding="utf-8"
    )

    epoch_metrics = candidate["metrics"][1:]
    validation_losses = [item["validation_loss"] for item in epoch_metrics]
    learning_curve = {
        "classification": "improving but flattening",
        "validation_minimum": min(validation_losses),
        "validation_minimum_epoch": 1 + validation_losses.index(min(validation_losses)),
        "epochs": [
            {
                "epoch": item["epoch"],
                "train_loss": item["training_loss"],
                "validation_loss": item["validation_loss"],
                "train_validation_gap": item["validation_loss"] - item["training_loss"],
            }
            for item in epoch_metrics
        ],
        "validation_loss_improvements": {
            f"epoch_{index}_to_{index + 1}": validation_losses[index - 1]
            - validation_losses[index]
            for index in range(1, len(validation_losses))
        },
        "interpretation": (
            "Validation improved through epoch 5, but each incremental loss reduction shrank. "
            "The run completed the locked schedule and was not extended."
        ),
    }
    write_json_atomic(CANDIDATE_RUN / "learning-curve-assessment.json", learning_curve)
    return report


def main() -> None:
    print(json.dumps(build_report(), indent=2))


if __name__ == "__main__":
    main()
