"""Build three-way reports for the controlled tokenizer-v2 experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from src.train_tokenizer import write_json_atomic
from src.train_tokenizer import load_corpus, load_split_assignments


EXPERIMENT_KEYS = ("smoke_v1", "data10m_v1", "tokenizer_v2")
EXPERIMENT_LABELS = {
    "smoke_v1": "FictionPulper-5M-smoke-v1",
    "data10m_v1": "FictionPulper-5M-data10M-v1",
    "tokenizer_v2": "FictionPulper-5M-data10M-tokenizerV2",
}


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_generation_comparison(generation_paths: dict[str, Path]) -> dict[str, Any]:
    generations = {name: load(path) for name, path in generation_paths.items()}
    prompts = [entry["prompt"] for entry in generations["smoke_v1"]["entries"]]
    for name, payload in generations.items():
        if [entry["prompt"] for entry in payload["entries"]] != prompts:
            raise RuntimeError(f"Generation prompts differ for {name}")
        if payload["sample_seed"] != 11337:
            raise RuntimeError(f"Generation seed changed for {name}")
    entries = []
    by_experiment = {
        name: {entry["prompt"]: entry for entry in payload["entries"]}
        for name, payload in generations.items()
    }
    for prompt in prompts:
        entries.append(
            {
                "prompt": prompt,
                "experiments": {
                    name: {
                        "greedy": by_experiment[name][prompt]["greedy"],
                        "sampled": by_experiment[name][prompt]["sampled"],
                    }
                    for name in EXPERIMENT_KEYS
                },
            }
        )
    return {
        "prompts_unchanged": True,
        "sampling_seed": 11337,
        "greedy_max_new_tokens": 128,
        "sampled_max_new_tokens": 256,
        "sampling_temperature": 0.8,
        "sampling_top_k": 50,
        "sampling_top_p": 0.95,
        "entries": entries,
    }


def build_quantitative_comparison() -> dict[str, Any]:
    smoke_summary = load("runs/smoke-5m-20261007-v1/summary.json")
    smoke_manifest = load("runs/smoke-5m-20261007-v1/manifest.json")
    data10m_summary = load("runs/fictionpulper-5m-data10m-v1/summary.json")
    data10m_metadata = load("data/packed/corpus-v1/metadata.json")
    v2_summary = load("runs/fictionpulper-5m-data10m-tokenizer-v2/summary.json")
    v2_metadata = load("data/packed/corpus-v1-tokenizer-v2/metadata.json")
    tokenizer_comparison = load(
        "experiments/fictionpulper-5m-data10m-tokenizer-v2/tokenizer-comparison.json"
    )
    smoke_corpus_test = load(
        "runs/fictionpulper-5m-data10m-tokenizer-v2/smoke-v1-post-hoc-corpus-v1-test.json"
    )["metrics"]
    smoke_clean_validation = load(
        "runs/fictionpulper-5m-data10m-tokenizer-v2/"
        "smoke-v1-post-hoc-legacy-clean-validation.json"
    )["metrics"]
    smoke_packed = smoke_manifest["packed_metadata"]
    smoke_words = sum(split["word_count"] for split in smoke_packed["splits"].values())
    smoke_tokens = sum(split["token_count"] for split in smoke_packed["splits"].values())
    data10m_words = sum(split["word_count"] for split in data10m_metadata["splits"].values())
    corpus_v1_test_bytes = tokenizer_comparison["tokenizers"]["original"]["splits"][
        "test"
    ]["text_bytes"]
    seed_records = load_corpus(Path("data/corpus/stories.jsonl"))
    seed_assignments = load_split_assignments(
        Path("data/corpus/stories.splits.json"), seed_records
    )
    legacy_test_bytes = sum(
        len(f"{record['title']}\n\n{record['text']}".encode("utf-8"))
        for record in seed_records
        if seed_assignments[record["id"]] == "test"
    )
    legacy_clean_validation_bytes = sum(
        len(f"{record['title']}\n\n{record['text']}".encode("utf-8"))
        for record in seed_records
        if seed_assignments[record["id"]] == "validation"
        and record["id"] != "162495d6f1c1"
    )

    def bits_per_byte(metrics: dict[str, Any], byte_count: int) -> float:
        return metrics["loss"] * metrics["valid_tokens"] / byte_count / math.log(2.0)

    return {
        "notes": {
            "smoke_corpus_v1_test": "post hoc; not used for checkpoint selection",
            "smoke_legacy_clean_validation": "post hoc; historical validation unchanged",
            "data10m_v1_legacy_clean_validation": "separately named audit metric",
            "tokenizer_v2_tests": "opened only after epoch-10 checkpoint selection",
        },
        "parameter_count": {
            "smoke_v1": smoke_summary["model_parameter_count"],
            "data10m_v1": data10m_summary["model_parameter_count"],
            "tokenizer_v2": v2_summary["model_parameter_count"],
        },
        "unique_corpus_stories": {
            "smoke_v1": sum(
                split["document_count"] for split in smoke_packed["splits"].values()
            ),
            "data10m_v1": sum(
                split["document_count"] for split in data10m_metadata["splits"].values()
            ),
            "tokenizer_v2": sum(
                split["document_count"] for split in v2_metadata["splits"].values()
            ),
        },
        "unique_corpus_words": {
            "smoke_v1": smoke_words,
            "data10m_v1": data10m_words,
            "tokenizer_v2": data10m_words,
        },
        "tokenizer_sha256": {
            "smoke_v1": smoke_packed["tokenizer_hash"],
            "data10m_v1": data10m_metadata["tokenizer_hash"],
            "tokenizer_v2": v2_metadata["tokenizer_hash"],
        },
        "tokens_per_word": {
            "smoke_v1": smoke_tokens / smoke_words,
            "data10m_v1": tokenizer_comparison["tokenizers"]["original"]["splits"][
                "train"
            ]["tokens_per_word"],
            "tokenizer_v2": tokenizer_comparison["tokenizers"]["tokenizer_v2"][
                "splits"
            ]["train"]["tokens_per_word"],
        },
        "packed_train_targets": {
            "smoke_v1": smoke_packed["splits"]["train"]["valid_target_token_count"],
            "data10m_v1": data10m_metadata["splits"]["train"][
                "valid_target_token_count"
            ],
            "tokenizer_v2": v2_metadata["splits"]["train"][
                "valid_target_token_count"
            ],
        },
        "best_validation_loss": {
            "smoke_v1": smoke_summary["best_validation_loss"],
            "data10m_v1": data10m_summary["best_validation_loss"],
            "tokenizer_v2": v2_summary["best_validation_loss"],
        },
        "corpus_v1_test_loss": {
            "smoke_v1": smoke_corpus_test["loss"],
            "data10m_v1": data10m_summary["corpus_v1_test"]["loss"],
            "tokenizer_v2": v2_summary["corpus_v1_test"]["loss"],
        },
        "corpus_v1_test_perplexity": {
            "smoke_v1": smoke_corpus_test["perplexity"],
            "data10m_v1": data10m_summary["corpus_v1_test"]["perplexity"],
            "tokenizer_v2": v2_summary["corpus_v1_test"]["perplexity"],
        },
        "corpus_v1_test_accuracy": {
            "smoke_v1": smoke_corpus_test["next_token_accuracy"],
            "data10m_v1": data10m_summary["corpus_v1_test"]["next_token_accuracy"],
            "tokenizer_v2": v2_summary["corpus_v1_test"]["next_token_accuracy"],
        },
        "corpus_v1_test_bits_per_byte": {
            "smoke_v1": bits_per_byte(smoke_corpus_test, corpus_v1_test_bytes),
            "data10m_v1": bits_per_byte(
                data10m_summary["corpus_v1_test"], corpus_v1_test_bytes
            ),
            "tokenizer_v2": bits_per_byte(
                v2_summary["corpus_v1_test"], corpus_v1_test_bytes
            ),
        },
        "legacy_clean_validation_loss": {
            "smoke_v1": smoke_clean_validation["loss"],
            "data10m_v1": data10m_summary["legacy_seed_validation_leakage_clean"]["loss"],
            "tokenizer_v2": v2_summary["legacy_seed_validation_leakage_clean"]["loss"],
        },
        "legacy_clean_validation_bits_per_byte": {
            "smoke_v1": bits_per_byte(
                smoke_clean_validation, legacy_clean_validation_bytes
            ),
            "data10m_v1": bits_per_byte(
                data10m_summary["legacy_seed_validation_leakage_clean"],
                legacy_clean_validation_bytes,
            ),
            "tokenizer_v2": bits_per_byte(
                v2_summary["legacy_seed_validation_leakage_clean"],
                legacy_clean_validation_bytes,
            ),
        },
        "legacy_test_loss": {
            "smoke_v1": smoke_summary["test_loss"],
            "data10m_v1": data10m_summary["legacy_seed_test"]["loss"],
            "tokenizer_v2": v2_summary["legacy_seed_test"]["loss"],
        },
        "legacy_test_perplexity": {
            "smoke_v1": smoke_summary["test_perplexity"],
            "data10m_v1": data10m_summary["legacy_seed_test"]["perplexity"],
            "tokenizer_v2": v2_summary["legacy_seed_test"]["perplexity"],
        },
        "legacy_test_accuracy": {
            "smoke_v1": smoke_summary["test_next_token_accuracy"],
            "data10m_v1": data10m_summary["legacy_seed_test"]["next_token_accuracy"],
            "tokenizer_v2": v2_summary["legacy_seed_test"]["next_token_accuracy"],
        },
        "legacy_test_bits_per_byte": {
            "smoke_v1": bits_per_byte(
                {
                    "loss": smoke_summary["test_loss"],
                    "valid_tokens": smoke_packed["splits"]["test"][
                        "valid_target_token_count"
                    ],
                },
                legacy_test_bytes,
            ),
            "data10m_v1": bits_per_byte(
                data10m_summary["legacy_seed_test"], legacy_test_bytes
            ),
            "tokenizer_v2": bits_per_byte(
                v2_summary["legacy_seed_test"], legacy_test_bytes
            ),
        },
        "training_runtime_seconds": {
            "smoke_v1": smoke_summary["training_seconds"],
            "data10m_v1": data10m_summary["training_seconds"],
            "tokenizer_v2": v2_summary["training_seconds"],
        },
        "valid_targets_per_second": {
            "smoke_v1": smoke_summary["average_valid_tokens_per_second"],
            "data10m_v1": data10m_summary["average_valid_tokens_per_second"],
            "tokenizer_v2": v2_summary["average_valid_tokens_per_second"],
        },
        "peak_vram_gb": {
            "smoke_v1": smoke_summary["peak_gpu_memory_gb"],
            "data10m_v1": data10m_summary["peak_gpu_memory_gb"],
            "tokenizer_v2": v2_summary["peak_gpu_memory_gb"],
        },
    }


def write_generation_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = ["# Three-Way Generation Comparison", ""]
    for entry in payload["entries"]:
        lines.extend([f"## {entry['prompt']}", ""])
        for decoding in ("greedy", "sampled"):
            for experiment in EXPERIMENT_KEYS:
                lines.extend(
                    [
                        f"### {EXPERIMENT_LABELS[experiment]} ({decoding})",
                        "",
                        entry["experiments"][experiment][decoding],
                        "",
                    ]
                )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_quantitative_markdown(payload: dict[str, Any], path: Path) -> None:
    rows = [
        ("Parameter count", "parameter_count"),
        ("Unique corpus stories", "unique_corpus_stories"),
        ("Unique corpus words", "unique_corpus_words"),
        ("Tokenizer SHA-256", "tokenizer_sha256"),
        ("Training tokens/word", "tokens_per_word"),
        ("Packed train targets", "packed_train_targets"),
        ("Best validation loss", "best_validation_loss"),
        ("Corpus-v1 test loss", "corpus_v1_test_loss"),
        ("Corpus-v1 test perplexity", "corpus_v1_test_perplexity"),
        ("Corpus-v1 test accuracy", "corpus_v1_test_accuracy"),
        ("Corpus-v1 test bits/byte", "corpus_v1_test_bits_per_byte"),
        ("Legacy clean validation loss", "legacy_clean_validation_loss"),
        ("Legacy clean validation bits/byte", "legacy_clean_validation_bits_per_byte"),
        ("Legacy test loss", "legacy_test_loss"),
        ("Legacy test perplexity", "legacy_test_perplexity"),
        ("Legacy test accuracy", "legacy_test_accuracy"),
        ("Legacy test bits/byte", "legacy_test_bits_per_byte"),
        ("Training runtime seconds", "training_runtime_seconds"),
        ("Valid targets/second", "valid_targets_per_second"),
        ("Peak VRAM GiB", "peak_vram_gb"),
    ]
    lines = [
        "# Three-Way Quantitative Comparison",
        "",
        "| Metric | Smoke v1 | Data10M v1 | Tokenizer v2 |",
        "|---|---:|---:|---:|",
    ]
    for label, key in rows:
        values = payload[key]
        lines.append(
            f"| {label} | {values['smoke_v1']} | {values['data10m_v1']} | "
            f"{values['tokenizer_v2']} |"
        )
    lines.extend(
        [
            "",
            "Smoke Corpus-v1 test and smoke leakage-clean validation are explicitly post hoc. "
            "They did not affect historical checkpoint selection or redefine historical metrics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/fictionpulper-5m-data10m-tokenizer-v2"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation = build_generation_comparison(
        {
            "smoke_v1": Path(
                "runs/smoke-5m-20261007-v1/generations/best-validation-final.json"
            ),
            "data10m_v1": Path(
                "runs/fictionpulper-5m-data10m-v1/generations/best-validation-final.json"
            ),
            "tokenizer_v2": args.output_dir / "generations/best-validation-final.json",
        }
    )
    quantitative = build_quantitative_comparison()
    write_json_atomic(args.output_dir / "generation-comparison.json", generation)
    write_generation_markdown(generation, args.output_dir / "generation-comparison.md")
    write_json_atomic(args.output_dir / "quantitative-comparison.json", quantitative)
    write_quantitative_markdown(
        quantitative, args.output_dir / "quantitative-comparison.md"
    )
    print(json.dumps(quantitative, indent=2))


if __name__ == "__main__":
    main()
