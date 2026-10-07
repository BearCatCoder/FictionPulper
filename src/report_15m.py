"""Build direct 5M-versus-15M controlled experiment reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.train_tokenizer import write_json_atomic


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative_change(baseline: float, candidate: float) -> float:
    return 100.0 * (candidate / baseline - 1.0)


def build_quantitative() -> dict[str, Any]:
    baseline = load("runs/fictionpulper-5m-data10m-v1/summary.json")
    candidate = load("runs/fictionpulper-15m-data10m-v1/summary.json")
    packed = load("data/packed/corpus-v1/metadata.json")
    values = {
        "parameters": {
            "data10m_5m": baseline["model_parameter_count"],
            "data10m_15m": candidate["model_parameter_count"],
        },
        "training_stories": {
            "data10m_5m": packed["splits"]["train"]["document_count"],
            "data10m_15m": packed["splits"]["train"]["document_count"],
        },
        "train_targets": {
            "data10m_5m": packed["splits"]["train"]["valid_target_token_count"],
            "data10m_15m": packed["splits"]["train"]["valid_target_token_count"],
        },
        "context": {"data10m_5m": 1024, "data10m_15m": 1024},
        "tokenizer_sha256": {
            "data10m_5m": "14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012",
            "data10m_15m": "14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012",
        },
        "optimizer_steps": {"data10m_5m": 1470, "data10m_15m": 1470},
        "best_validation_loss": {
            "data10m_5m": baseline["best_validation_loss"],
            "data10m_15m": candidate["best_validation_loss"],
        },
        "corpus_v1_test_loss": {
            "data10m_5m": baseline["corpus_v1_test"]["loss"],
            "data10m_15m": candidate["corpus_v1_test"]["loss"],
        },
        "corpus_v1_test_perplexity": {
            "data10m_5m": baseline["corpus_v1_test"]["perplexity"],
            "data10m_15m": candidate["corpus_v1_test"]["perplexity"],
        },
        "corpus_v1_test_accuracy": {
            "data10m_5m": baseline["corpus_v1_test"]["next_token_accuracy"],
            "data10m_15m": candidate["corpus_v1_test"]["next_token_accuracy"],
        },
        "legacy_clean_validation_loss": {
            "data10m_5m": baseline["legacy_seed_validation_leakage_clean"]["loss"],
            "data10m_15m": candidate["legacy_seed_validation_leakage_clean"]["loss"],
        },
        "legacy_test_loss": {
            "data10m_5m": baseline["legacy_seed_test"]["loss"],
            "data10m_15m": candidate["legacy_seed_test"]["loss"],
        },
        "legacy_test_perplexity": {
            "data10m_5m": baseline["legacy_seed_test"]["perplexity"],
            "data10m_15m": candidate["legacy_seed_test"]["perplexity"],
        },
        "legacy_test_accuracy": {
            "data10m_5m": baseline["legacy_seed_test"]["next_token_accuracy"],
            "data10m_15m": candidate["legacy_seed_test"]["next_token_accuracy"],
        },
        "runtime_seconds": {
            "data10m_5m": baseline["training_seconds"],
            "data10m_15m": candidate["training_seconds"],
        },
        "valid_targets_per_second": {
            "data10m_5m": baseline["average_valid_tokens_per_second"],
            "data10m_15m": candidate["average_valid_tokens_per_second"],
        },
        "peak_vram_gb": {
            "data10m_5m": baseline["peak_gpu_memory_gb"],
            "data10m_15m": candidate["peak_gpu_memory_gb"],
        },
    }
    lower_is_better = {
        "best_validation_loss",
        "corpus_v1_test_loss",
        "corpus_v1_test_perplexity",
        "legacy_clean_validation_loss",
        "legacy_test_loss",
        "legacy_test_perplexity",
    }
    relative = {}
    for key, pair in values.items():
        if not all(isinstance(pair[name], (int, float)) for name in pair):
            continue
        change = relative_change(pair["data10m_5m"], pair["data10m_15m"])
        relative[key] = {
            "relative_change_percentage": change,
            "relative_improvement_percentage": -change if key in lower_is_better else change,
        }
    return {"values": values, "relative": relative}


def build_generations() -> dict[str, Any]:
    baseline = load(
        "runs/fictionpulper-5m-data10m-v1/generations/best-validation-final.json"
    )
    candidate = load(
        "runs/fictionpulper-15m-data10m-v1/generations/best-validation-final.json"
    )
    baseline_by_prompt = {entry["prompt"]: entry for entry in baseline["entries"]}
    candidate_by_prompt = {entry["prompt"]: entry for entry in candidate["entries"]}
    if list(baseline_by_prompt) != list(candidate_by_prompt):
        raise RuntimeError("Generation prompts changed")
    if baseline["sample_seed"] != 11337 or candidate["sample_seed"] != 11337:
        raise RuntimeError("Generation sampling seed changed")
    return {
        "prompts_unchanged": True,
        "greedy_max_new_tokens": 128,
        "sampled_max_new_tokens": 256,
        "sampling_temperature": 0.8,
        "sampling_top_k": 50,
        "sampling_top_p": 0.95,
        "sampling_seed": 11337,
        "entries": [
            {
                "prompt": prompt,
                "data10m_5m": {
                    "greedy": baseline_by_prompt[prompt]["greedy"],
                    "sampled": baseline_by_prompt[prompt]["sampled"],
                },
                "data10m_15m": {
                    "greedy": candidate_by_prompt[prompt]["greedy"],
                    "sampled": candidate_by_prompt[prompt]["sampled"],
                },
            }
            for prompt in baseline_by_prompt
        ],
    }


def write_quantitative_markdown(payload: dict[str, Any], path: Path) -> None:
    rows = [
        ("Parameters", "parameters"),
        ("Training stories", "training_stories"),
        ("Train targets", "train_targets"),
        ("Context", "context"),
        ("Tokenizer SHA-256", "tokenizer_sha256"),
        ("Optimizer steps", "optimizer_steps"),
        ("Best validation loss", "best_validation_loss"),
        ("Corpus-v1 test loss", "corpus_v1_test_loss"),
        ("Corpus-v1 test PPL", "corpus_v1_test_perplexity"),
        ("Corpus-v1 test accuracy", "corpus_v1_test_accuracy"),
        ("Legacy clean validation loss", "legacy_clean_validation_loss"),
        ("Legacy test loss", "legacy_test_loss"),
        ("Legacy test PPL", "legacy_test_perplexity"),
        ("Legacy test accuracy", "legacy_test_accuracy"),
        ("Runtime seconds", "runtime_seconds"),
        ("Valid targets/second", "valid_targets_per_second"),
        ("Peak VRAM GiB", "peak_vram_gb"),
    ]
    lines = [
        "# 5M versus 15M Quantitative Comparison",
        "",
        "| Metric | 5M Data10M | 15M Data10M | Relative change |",
        "|---|---:|---:|---:|",
    ]
    for label, key in rows:
        pair = payload["values"][key]
        relative = payload["relative"].get(key)
        change = (
            f"{relative['relative_change_percentage']:+.3f}%" if relative else "same"
        )
        lines.append(
            f"| {label} | {pair['data10m_5m']} | {pair['data10m_15m']} | {change} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_generation_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = ["# 5M versus 15M Generation Comparison", ""]
    for entry in payload["entries"]:
        lines.extend([f"## {entry['prompt']}", ""])
        for decoding in ("greedy", "sampled"):
            lines.extend(
                [
                    f"### 5M Data10M ({decoding})",
                    "",
                    entry["data10m_5m"][decoding],
                    "",
                    f"### 15M Data10M ({decoding})",
                    "",
                    entry["data10m_15m"][decoding],
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/fictionpulper-15m-data10m-v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quantitative = build_quantitative()
    generations = build_generations()
    write_json_atomic(args.output_dir / "quantitative-comparison.json", quantitative)
    write_quantitative_markdown(
        quantitative, args.output_dir / "quantitative-comparison.md"
    )
    write_json_atomic(args.output_dir / "generation-comparison.json", generations)
    write_generation_markdown(generations, args.output_dir / "generation-comparison.md")
    print(json.dumps(quantitative, indent=2))


if __name__ == "__main__":
    main()
