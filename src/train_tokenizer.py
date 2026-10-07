"""Train the FictionPulper byte-level BPE tokenizer from scratch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|story|>",
    "<|crime|>",
    "<|noir|>",
    "<|mystery|>",
    "<|horror|>",
    "<|science_fiction|>",
    "<|western|>",
    "<|adventure|>",
    "<|fantasy|>",
    "<|weird|>",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_corpus(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSON on {path}:{line_number}: {error}") from error
            for field in ("id", "title", "text"):
                if not isinstance(record.get(field), str) or not record[field]:
                    raise ValueError(f"Invalid {field!r} on {path}:{line_number}")
            if record["id"] in ids:
                raise ValueError(f"Duplicate corpus id {record['id']!r} on line {line_number}")
            ids.add(record["id"])
            records.append(record)
    if not records:
        raise ValueError(f"Corpus is empty: {path}")
    return records


def training_documents(records: list[dict[str, Any]]) -> Iterator[str]:
    for record in records:
        yield f"{record['title']}\n\n{record['text']}"


def load_split_assignments(path: Path, records: list[dict[str, Any]]) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assignments = payload["assignments"]
    record_ids = {record["id"] for record in records}
    if set(assignments) != record_ids:
        raise ValueError("Split assignments do not exactly match corpus document IDs")
    allowed = {"train", "validation", "test"}
    if set(assignments.values()) - allowed:
        raise ValueError("Split assignments contain an unknown split")
    return assignments


def build_tokenizer(
    records: list[dict[str, Any]],
    *,
    vocab_size: int,
    min_frequency: int,
    show_progress: bool = True,
) -> Tokenizer:
    if vocab_size <= len(SPECIAL_TOKENS) + 256:
        raise ValueError("vocab_size must leave room beyond special tokens and byte alphabet")
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
    )
    tokenizer.train_from_iterator(training_documents(records), trainer=trainer, length=len(records))
    if tokenizer.get_vocab_size() != vocab_size:
        raise RuntimeError(
            f"Tokenizer produced {tokenizer.get_vocab_size()} tokens, expected {vocab_size}"
        )
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tokenizer.save(str(temporary_path), pretty=True)
    temporary_path.replace(output_path)
    return sha256_file(output_path)


def tokenizer_statistics(
    tokenizer: Tokenizer,
    training_records: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
    assignments: dict[str, str],
    *,
    corpus_path: Path,
    output_path: Path,
    tokenizer_hash: str,
) -> dict[str, Any]:
    training_documents_text = [
        f"{record['title']}\n\n{record['text']}" for record in training_records
    ]
    training_token_count = sum(len(tokenizer.encode(text).ids) for text in training_documents_text)
    training_word_count = sum(record["words"] for record in training_records)
    training_byte_count = sum(len(text.encode("utf-8")) for text in training_documents_text)
    story_token_counts = [
        len(tokenizer.encode(f"{record['title']}\n\n{record['text']}").ids)
        for record in all_records
    ]
    sorted_counts = sorted(story_token_counts)

    def percentile_95(values: list[int]) -> int:
        return values[max(0, (95 * len(values) + 99) // 100 - 1)]

    thresholds = {}
    for threshold in (1024, 2048, 4096):
        count = sum(value <= threshold for value in story_token_counts)
        thresholds[str(threshold)] = {
            "count": count,
            "percentage": 100.0 * count / len(story_token_counts),
        }

    return {
        "tokenizer_type": "byte_level_bpe",
        "vocab_size": tokenizer.get_vocab_size(),
        "special_token_ids": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "training_document_count": len(training_records),
        "training_word_count": training_word_count,
        "training_token_count": training_token_count,
        "average_tokens_per_word": training_token_count / training_word_count,
        "average_bytes_per_token": training_byte_count / training_token_count,
        "all_document_count": len(all_records),
        "story_token_count_definition": "title, two newlines, and story text; excludes control tokens",
        "story_token_counts": {
            "minimum": min(story_token_counts),
            "median": statistics.median(story_token_counts),
            "mean": statistics.mean(story_token_counts),
            "percentile_95": percentile_95(sorted_counts),
            "maximum": max(story_token_counts),
            "thresholds": thresholds,
        },
        "split_document_counts": dict(
            sorted({split: list(assignments.values()).count(split) for split in set(assignments.values())}.items())
        ),
        "tokenization_examples": representative_examples(tokenizer, all_records),
        "inefficient_passages": inefficient_passages(tokenizer, all_records),
        "tokenizer_path": str(output_path),
        "tokenizer_sha256": tokenizer_hash,
    }


def representative_examples(
    tokenizer: Tokenizer, records: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    patterns = {
        "dialogue": re.compile(r"(?:^|\n)[“\"]"),
        "contractions": re.compile(r"\b\w+[’'](?:t|s|re|ve|d|ll|m)\b", re.I),
        "em_dash": re.compile("—"),
        "curly_quotation_marks": re.compile("[“”]"),
        "unusual_names": re.compile(
            r"\b(?:Ivanitch|Bronckhorst|Pinecoffin|Bredenbutta|Peythroppe)\b", re.I
        ),
        "historical_punctuation": re.compile(r"(?:,—|;—|,—|—;)"),
    }
    examples: dict[str, dict[str, Any]] = {}
    used_story_ids: set[str] = set()
    for label, pattern in patterns.items():
        for record in records:
            if record["id"] in used_story_ids:
                continue
            match = pattern.search(record["text"])
            if not match:
                continue
            start = max(0, match.start() - 70)
            end = min(len(record["text"]), match.end() + 110)
            snippet = record["text"][start:end].replace("\n", " ").strip()
            encoding = tokenizer.encode(snippet)
            examples[label] = {
                "story_id": record["id"],
                "title": record["title"],
                "text": snippet,
                "token_count": len(encoding.ids),
                "tokens": encoding.tokens,
                "round_trip_exact": tokenizer.decode(encoding.ids) == snippet,
            }
            used_story_ids.add(record["id"])
            break
    return examples


def inefficient_passages(
    tokenizer: Tokenizer, records: list[dict[str, Any]], limit: int = 10
) -> list[dict[str, Any]]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        for paragraph in record["text"].split("\n\n"):
            words = len(paragraph.split())
            if words < 20:
                continue
            token_count = len(tokenizer.encode(paragraph).ids)
            candidates.append(
                (
                    token_count / words,
                    {
                        "story_id": record["id"],
                        "title": record["title"],
                        "words": words,
                        "tokens": token_count,
                        "tokens_per_word": token_count / words,
                        "text": paragraph[:300],
                    },
                )
            )
    return [entry for _, entry in sorted(candidates, key=lambda item: item[0], reverse=True)[:limit]]


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    temporary_path.replace(path)


def update_corpus_statistics(corpus_path: Path, token_stats: dict[str, Any]) -> None:
    stats_path = corpus_path.with_suffix(".stats.json")
    if not stats_path.exists():
        return
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["tokenizer_diagnostics"] = token_stats
    write_json_atomic(stats_path, stats)


def train_from_config(config_path: Path, *, show_progress: bool = True) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tokenizer_config = config["tokenizer"]
    corpus_path = Path(config["data"]["corpus_path"])
    split_path = Path(config["data"]["split_path"])
    output_path = Path(tokenizer_config["path"])
    records = load_corpus(corpus_path)
    assignments = load_split_assignments(split_path, records)
    training_records = [record for record in records if assignments[record["id"]] == "train"]
    tokenizer = build_tokenizer(
        training_records,
        vocab_size=int(tokenizer_config["vocab_size"]),
        min_frequency=int(tokenizer_config.get("min_frequency", 2)),
        show_progress=show_progress,
    )
    tokenizer_hash = save_tokenizer(tokenizer, output_path)
    stats = tokenizer_statistics(
        tokenizer,
        training_records,
        records,
        assignments,
        corpus_path=corpus_path,
        output_path=output_path,
        tokenizer_hash=tokenizer_hash,
    )
    write_json_atomic(output_path.with_suffix(".meta.json"), stats)
    update_corpus_statistics(corpus_path, stats)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke-5m.yaml"))
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = train_from_config(args.config, show_progress=not args.no_progress)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
