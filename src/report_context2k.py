"""Report the sealed 1024 versus 2048 context experiment."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from tokenizers import Tokenizer

from src.context_diagnostics import repetition_metrics
from src.train import GENERATION_SETTINGS
from src.train_tokenizer import write_json_atomic


BASELINE_RUN = Path("runs/fictionpulper-15m-data30m-compute5")
CANDIDATE_RUN = Path("runs/fictionpulper-15m-data30m-context2k-v1")
EXPERIMENT_DIR = Path("experiments/fictionpulper-15m-data30m-context2k-v1")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def change(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate / baseline - 1.0)


def historical(evaluations: dict[str, Any], name: str) -> dict[str, Any]:
    return evaluations["historical_post_selection_evaluations"][name]["metrics"]


def aggregate_repetition(entries: list[dict[str, Any]], mode: str) -> dict[str, float]:
    keys = (
        "distinct_1",
        "distinct_2",
        "distinct_3",
        "repeated_4gram_rate",
        "repeated_sentence_rate",
        "longest_repeated_token_span",
    )
    return {key: mean(entry["repetition"][mode][key] for entry in entries) for key in keys}


def build_report() -> dict[str, Any]:
    tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer.json")
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

    quantitative = {
        "run_id": "fictionpulper-15m-data30m-context2k-v1",
        "experimental_variable": "maximum sequence length from 1024 to 2048",
        "packing_note": (
            "Source token streams and valid targets are identical. Chunk counts, "
            "allocated slots, and padding are context-dependent."
        ),
        "compute5_context1024": {
            "parameters": 15047040,
            "context": 1024,
            "batch_size": 16,
            "gradient_accumulation_steps": 4,
            "allocated_slots_per_optimizer_step": 65536,
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
        "context2k": {
            "parameters": 15047040,
            "context": 2048,
            "batch_size": 16,
            "gradient_accumulation_steps": 2,
            "allocated_slots_per_optimizer_step": 65536,
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
            "data30m_test_loss": change(candidate["test_loss"], baseline["test_loss"]),
            "data30m_test_perplexity": change(
                candidate["test_perplexity"], baseline["test_perplexity"]
            ),
            "data30m_test_accuracy": change(
                candidate["test_next_token_accuracy"], baseline["test_next_token_accuracy"]
            ),
            "common_held_out_loss": change(
                candidate_common["loss"], baseline_common["loss"]
            ),
            "common_held_out_perplexity": change(
                candidate_common["perplexity"], baseline_common["perplexity"]
            ),
            "common_held_out_accuracy": change(
                candidate_common["next_token_accuracy"], baseline_common["next_token_accuracy"]
            ),
            "legacy_test_loss": change(
                candidate_legacy_test["loss"], baseline_legacy_test["loss"]
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
    write_json_atomic(CANDIDATE_RUN / "quantitative-comparison.json", quantitative)

    rows = [
        ("Parameters", 15047040, 15047040),
        ("Context", 1024, 2048),
        ("Tokenizer", "v1", "v1"),
        ("Corpus", "Data30M", "Data30M"),
        ("Unique train targets", 27177271, 27177271),
        ("Microbatch", 16, 16),
        ("Gradient accumulation", 4, 2),
        ("Allocated slots/step", 65536, 65536),
        ("Target presentations", baseline["total_valid_training_targets_processed"], candidate["total_valid_training_targets_processed"]),
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
    quantitative_markdown = "\n".join(
        [
            "# Quantitative Comparison",
            "",
            "| Metric | 15M / 1024 / Compute5 | 15M / 2048 |",
            "|---|---:|---:|",
            *[f"| {label} | {left} | {right} |" for label, left, right in rows],
            "",
            "Source token streams and valid targets are identical. Chunk counts, allocated slots, and document-tail padding are packing-dependent.",
        ]
    ) + "\n"
    (CANDIDATE_RUN / "quantitative-comparison.md").write_text(
        quantitative_markdown, encoding="utf-8"
    )
    (EXPERIMENT_DIR / "quantitative-comparison.md").write_text(
        quantitative_markdown, encoding="utf-8"
    )

    baseline_generations = load(BASELINE_RUN / "generations/best-validation-final.json")
    candidate_generations = load(CANDIDATE_RUN / "generations/best-validation-final.json")
    if baseline_generations["generation_settings"] != GENERATION_SETTINGS:
        raise RuntimeError("Compute5 generation settings changed")
    if candidate_generations["generation_settings"] != GENERATION_SETTINGS:
        raise RuntimeError("Context2k generation settings changed")
    if baseline_generations["sample_seed"] != candidate_generations["sample_seed"]:
        raise RuntimeError("Generation sample seeds differ")
    if [entry["prompt"] for entry in baseline_generations["entries"]] != [
        entry["prompt"] for entry in candidate_generations["entries"]
    ]:
        raise RuntimeError("Generation prompt suites differ")

    generation_entries = []
    for left, right in zip(
        baseline_generations["entries"], candidate_generations["entries"], strict=True
    ):
        generation_entries.append(
            {
                "prompt": left["prompt"],
                "sample_seed": right["sample_seed"],
                "compute5_greedy": left["greedy"],
                "context2k_greedy": right["greedy"],
                "compute5_sampled": left["sampled"],
                "context2k_sampled": right["sampled"],
                "repetition": {
                    "compute5": {
                        "greedy": repetition_metrics(tokenizer, left["greedy"]),
                        "sampled": repetition_metrics(tokenizer, left["sampled"]),
                    },
                    "context2k": {
                        "greedy": repetition_metrics(tokenizer, right["greedy"]),
                        "sampled": repetition_metrics(tokenizer, right["sampled"]),
                    },
                },
            }
        )
    generation = {
        "settings": {
            **GENERATION_SETTINGS,
            "sample_seed": candidate_generations["sample_seed"],
            "effective_seed_policy": "sample_seed + prompt_index",
        },
        "entries": generation_entries,
    }
    write_json_atomic(CANDIDATE_RUN / "generation-comparison.json", generation)
    generation_lines = ["# Generation Comparison", ""]
    for index, entry in enumerate(generation_entries, start=1):
        generation_lines.extend(
            [
                f"## Prompt {index}",
                "",
                f"> {entry['prompt']}",
                "",
                "### Compute5 Greedy",
                "",
                entry["compute5_greedy"],
                "",
                "### Context2k Greedy",
                "",
                entry["context2k_greedy"],
                "",
                "### Compute5 Sampled",
                "",
                entry["compute5_sampled"],
                "",
                "### Context2k Sampled",
                "",
                entry["context2k_sampled"],
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

    repetition_entries = []
    for entry in generation_entries:
        repetition_entries.append(
            {
                "prompt": entry["prompt"],
                "repetition": entry["repetition"],
            }
        )
    repetition = {
        "definition": "Tokenizer-v1 token n-grams over generated continuation only",
        "original_prompt_suite": {
            "entries": repetition_entries,
            "aggregate": {
                label: {
                    mode: {
                        key: mean(
                            entry["repetition"][label][mode][key]
                            for entry in repetition_entries
                        )
                        for key in (
                            "distinct_1",
                            "distinct_2",
                            "distinct_3",
                            "repeated_4gram_rate",
                            "repeated_sentence_rate",
                            "longest_repeated_token_span",
                        )
                    }
                    for mode in ("greedy", "sampled")
                }
                for label in ("compute5", "context2k")
            },
        },
    }
    retention = load(CANDIDATE_RUN / "context-retention-comparison.json")
    repetition["context_retention_suite"] = {
        label: {
            mode: aggregate_repetition(model["entries"], mode)
            for mode in ("greedy", "sampled")
        }
        for label, model in retention["models"].items()
    }
    write_json_atomic(CANDIDATE_RUN / "repetition-comparison.json", repetition)

    retention_lines = [
        "# Context-Retention Comparison",
        "",
        "All fact prefixes fall outside Compute5's 1,024-token window and inside context2k's 2,048-token window. The suite was run post hoc and did not affect checkpoint selection.",
        "",
    ]
    baseline_retention = retention["models"]["compute5_context1024"]["entries"]
    candidate_retention = retention["models"]["context2k"]["entries"]
    for left, right in zip(baseline_retention, candidate_retention, strict=True):
        retention_lines.extend(
            [
                f"## {left['id']}",
                "",
                f"Facts: `{json.dumps(left['facts'], sort_keys=True)}`",
                "",
                "### Compute5 Greedy",
                "",
                left["greedy"],
                "",
                "### Context2k Greedy",
                "",
                right["greedy"],
                "",
                "### Compute5 Sampled",
                "",
                left["sampled"],
                "",
                "### Context2k Sampled",
                "",
                right["sampled"],
                "",
            ]
        )
    retention_markdown = "\n".join(retention_lines)
    (CANDIDATE_RUN / "context-retention-comparison.md").write_text(
        retention_markdown, encoding="utf-8"
    )
    (EXPERIMENT_DIR / "context-retention-comparison.md").write_text(
        retention_markdown, encoding="utf-8"
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
    }
    write_json_atomic(CANDIDATE_RUN / "learning-curve-assessment.json", learning_curve)
    return quantitative


def main() -> None:
    print(json.dumps(build_report(), indent=2))


if __name__ == "__main__":
    main()
