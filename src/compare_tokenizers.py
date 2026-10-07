"""Compare two tokenizers on identical FictionPulper documents and passages."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from src.tokenize_corpus import encode_document
from src.train_tokenizer import load_corpus, load_split_assignments, sha256_file, write_json_atomic


PASSAGE_PATTERNS = {
    "dialogue": re.compile(r"(?:^|\n)[ \t]*[\"“][A-Za-z]", re.MULTILINE),
    "names": re.compile(
        r"\b(?:Mallory|Clara|Pinecoffin|Bredenbutta|Peythroppe|Bronckhorst|Ivanitch)\b"
    ),
    "contractions": re.compile(r"\b[A-Za-z]+[’'](?:t|s|re|ve|d|ll|m)\b", re.IGNORECASE),
    "curly_quotation_marks": re.compile(r"[“”]"),
    "em_dashes": re.compile("—"),
    "historical_vocabulary": re.compile(
        r"\b(?:whilst|whereupon|thereupon|hither|thither|betwixt|peradventure)\b",
        re.IGNORECASE,
    ),
    "science_fiction_vocabulary": re.compile(
        r"\b(?:rocket|spaceship|spacecraft|Martian|planet|atomic|interplanetary)\b",
        re.IGNORECASE,
    ),
    "western_terminology": re.compile(
        r"\b(?:sheriff|cowboy|ranch|saloon|prairie|posse|stagecoach)\b",
        re.IGNORECASE,
    ),
}


def percentile_95(values: list[int]) -> int:
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]


def split_statistics(
    tokenizer: Tokenizer, records: list[dict[str, Any]]
) -> dict[str, Any]:
    token_counts = [len(encode_document(tokenizer, record)[0]) for record in records]
    total_tokens = sum(token_counts)
    total_words = sum(record["words"] for record in records)
    total_bytes = sum(
        len(f"{record['title']}\n\n{record['text']}".encode("utf-8")) for record in records
    )
    return {
        "documents": len(records),
        "words": total_words,
        "text_bytes": total_bytes,
        "total_tokens": total_tokens,
        "tokens_per_word": total_tokens / total_words,
        "bytes_per_token": total_bytes / total_tokens,
        "story_tokens": {
            "minimum": min(token_counts),
            "median": statistics.median(token_counts),
            "mean": statistics.mean(token_counts),
            "percentile_95": percentile_95(token_counts),
            "maximum": max(token_counts),
            "thresholds": {
                str(threshold): {
                    "count": sum(count <= threshold for count in token_counts),
                    "percentage": 100.0
                    * sum(count <= threshold for count in token_counts)
                    / len(token_counts),
                }
                for threshold in (1024, 2048, 4096)
            },
        },
    }


def select_passages(records: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    passages = {}
    for label, pattern in PASSAGE_PATTERNS.items():
        for record in records:
            match = pattern.search(record["text"])
            if match is None:
                continue
            start = max(0, match.start() - 100)
            end = min(len(record["text"]), match.end() + 140)
            passages[label] = {
                "story_id": record["id"],
                "title": record["title"],
                "text": record["text"][start:end].replace("\n", " ").strip(),
            }
            break
        if label not in passages:
            raise RuntimeError(f"No representative passage found for {label}")
    return passages


def compare_tokenizers(
    *,
    corpus_path: Path,
    split_path: Path,
    tokenizer_paths: dict[str, Path],
) -> dict[str, Any]:
    records = load_corpus(corpus_path)
    assignments = load_split_assignments(split_path, records)
    tokenizers = {
        name: Tokenizer.from_file(str(path)) for name, path in tokenizer_paths.items()
    }
    split_records = {
        split: [record for record in records if assignments[record["id"]] == split]
        for split in ("train", "validation", "test")
    }
    passages = select_passages(records)
    passage_comparison = {}
    for label, passage in passages.items():
        encoded = {}
        for tokenizer_name, tokenizer in tokenizers.items():
            encoding = tokenizer.encode(passage["text"])
            encoded[tokenizer_name] = {
                "token_count": len(encoding.ids),
                "bytes_per_token": len(passage["text"].encode("utf-8"))
                / len(encoding.ids),
                "tokens": encoding.tokens,
                "round_trip_exact": tokenizer.decode(encoding.ids) == passage["text"],
            }
        passage_comparison[label] = {**passage, "tokenizers": encoded}
    return {
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
        "definitions": {
            "total_tokens": "full packed document format including control, BOS, and EOS tokens",
            "bytes_per_token": "UTF-8 bytes in title, two newlines, and story text divided by full packed token count",
            "story_tokens": "full packed document token count",
        },
        "tokenizers": {
            name: {
                "path": str(tokenizer_paths[name]),
                "sha256": sha256_file(tokenizer_paths[name]),
                "vocab_size": tokenizer.get_vocab_size(),
                "normalizer": None if tokenizer.normalizer is None else str(tokenizer.normalizer),
                "splits": {
                    split: split_statistics(tokenizer, split_records[split])
                    for split in ("train", "validation", "test")
                },
            }
            for name, tokenizer in tokenizers.items()
        },
        "representative_passages": passage_comparison,
    }


def write_markdown(payload: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Tokenizer Comparison",
        "",
        f"Corpus SHA-256: `{payload['corpus_sha256']}`",
        f"Split SHA-256: `{payload['split_sha256']}`",
        "",
    ]
    for split in ("train", "validation", "test"):
        lines.extend(
            [
                f"## {split.title()}",
                "",
                "| Metric | Original | Tokenizer v2 |",
                "|---|---:|---:|",
            ]
        )
        original = payload["tokenizers"]["original"]["splits"][split]
        v2 = payload["tokenizers"]["tokenizer_v2"]["splits"][split]
        rows = [
            ("Total tokens", original["total_tokens"], v2["total_tokens"]),
            ("Tokens/word", original["tokens_per_word"], v2["tokens_per_word"]),
            ("Bytes/token", original["bytes_per_token"], v2["bytes_per_token"]),
        ]
        for key, label in (
            ("minimum", "Minimum story tokens"),
            ("median", "Median story tokens"),
            ("mean", "Mean story tokens"),
            ("percentile_95", "95th percentile"),
            ("maximum", "Maximum story tokens"),
        ):
            rows.append((label, original["story_tokens"][key], v2["story_tokens"][key]))
        for threshold in ("1024", "2048", "4096"):
            rows.append(
                (
                    f"% <= {threshold}",
                    original["story_tokens"]["thresholds"][threshold]["percentage"],
                    v2["story_tokens"]["thresholds"][threshold]["percentage"],
                )
            )
        lines.extend(f"| {label} | {old} | {new} |" for label, old, new in rows)
        lines.append("")
    lines.extend(["## Representative Passages", ""])
    for label, passage in payload["representative_passages"].items():
        original = passage["tokenizers"]["original"]
        v2 = passage["tokenizers"]["tokenizer_v2"]
        lines.extend(
            [
                f"### {label.replace('_', ' ').title()}",
                "",
                f"Story: `{passage['story_id']}` ({passage['title']})",
                "",
                f"> {passage['text']}",
                "",
                f"Original: {original['token_count']} tokens; v2: {v2['token_count']} tokens.",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--tokenizer-v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = compare_tokenizers(
        corpus_path=args.corpus,
        split_path=args.splits,
        tokenizer_paths={"original": args.original, "tokenizer_v2": args.tokenizer_v2},
    )
    write_json_atomic(args.output, payload)
    write_markdown(payload, args.output.with_suffix(".md"))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
