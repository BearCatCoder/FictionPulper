"""Encode canonical FictionPulper splits into indexed uint16 token files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tokenizers import Tokenizer

from src.train_tokenizer import load_corpus, load_split_assignments, sha256_file, write_json_atomic


PREPROCESSING_VERSION = 6
CONTROL_GENRES = {
    "crime": ("crime", "criminal", "murder"),
    "noir": ("noir",),
    "mystery": ("mystery", "detective", "suspense", "thriller"),
    "horror": ("horror", "gothic", "ghost", "supernatural", "paranormal"),
    "science_fiction": ("science fiction", "sci-fi", "speculative", "dystopian"),
    "western": ("western", "frontier", "cowboy"),
    "adventure": ("adventure", "quest", "survival"),
    "fantasy": ("fantasy", "fairy tale", "magic", "magical", "folklore", "fable"),
    "weird": ("weird", "occult", "eldritch"),
}


def nearest_rank_percentile(values: list[int], percentile: float) -> int:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty list")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def chunk_statistics(token_counts: list[int], sequence_length: int) -> dict[str, Any]:
    chunks = [math.ceil((count - 1) / sequence_length) for count in token_counts]
    total_chunks = sum(chunks)
    total_targets = sum(count - 1 for count in token_counts)
    distribution = {
        "1": sum(count == 1 for count in chunks),
        "2": sum(count == 2 for count in chunks),
        "3-4": sum(3 <= count <= 4 for count in chunks),
        "5+": sum(count >= 5 for count in chunks),
    }
    return {
        "sequence_length": sequence_length,
        "minimum": min(chunks),
        "median": float(np.median(chunks)),
        "mean": float(np.mean(chunks)),
        "percentile_95": nearest_rank_percentile(chunks, 0.95),
        "maximum": max(chunks),
        "distribution": {
            label: {
                "count": count,
                "percentage": 100.0 * count / len(chunks),
            }
            for label, count in distribution.items()
        },
        "average_usable_targets_per_chunk": total_targets / total_chunks,
    }


def genre_control_token(genres: list[str]) -> str | None:
    normalized = " | ".join(genres).lower()
    for control, terms in CONTROL_GENRES.items():
        if any(term in normalized for term in terms):
            return f"<|{control}|>"
    return None


def encode_document(tokenizer: Tokenizer, record: dict[str, Any]) -> tuple[list[int], str | None]:
    control_token = genre_control_token(record["genres"])
    prefix = [tokenizer.token_to_id("<|story|>")]
    if control_token is not None:
        prefix.append(tokenizer.token_to_id(control_token))
    prefix.append(tokenizer.token_to_id("<|bos|>"))
    content = tokenizer.encode(f"{record['title']}\n\n{record['text']}").ids
    token_ids = prefix + content + [tokenizer.token_to_id("<|eos|>")]
    if any(token_id is None for token_id in token_ids):
        raise ValueError("Tokenizer is missing a required control token")
    if max(token_ids) >= np.iinfo(np.uint16).max:
        raise ValueError("Token ID does not fit in uint16")
    return token_ids, control_token


def write_split(
    split: str,
    records: list[dict[str, Any]],
    tokenizer: Tokenizer,
    output_dir: Path,
    sequence_length: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / f"{split}.bin"
    index_path = output_dir / f"{split}.index.json"
    temporary_bin_path = bin_path.with_suffix(".bin.tmp")
    index: list[dict[str, Any]] = []
    offset = 0

    with temporary_bin_path.open("wb") as output_file:
        for record in records:
            token_ids, control_token = encode_document(tokenizer, record)
            array = np.asarray(token_ids, dtype=np.uint16)
            array.tofile(output_file)
            index.append(
                {
                    "id": record["id"],
                    "offset": offset,
                    "length": len(token_ids),
                    "words": record["words"],
                    "genre_control_token": control_token,
                    "sequence_count": math.ceil((len(token_ids) - 1) / sequence_length),
                }
            )
            offset += len(token_ids)
        output_file.flush()
        os.fsync(output_file.fileno())
    temporary_bin_path.replace(bin_path)
    write_json_atomic(index_path, {"split": split, "documents": index})

    valid_targets = sum(document["length"] - 1 for document in index)
    sequence_count = sum(document["sequence_count"] for document in index)
    allocated_slots = sequence_count * sequence_length
    return {
        "document_count": len(records),
        "word_count": sum(record["words"] for record in records),
        "token_count": offset,
        "valid_target_token_count": valid_targets,
        "sequence_count": sequence_count,
        "allocated_chunk_slots": allocated_slots,
        "padding_slots": allocated_slots - valid_targets,
        "padding_percentage": 100.0 * (allocated_slots - valid_targets) / allocated_slots,
        "chunks_per_story": chunk_statistics(
            [document["length"] for document in index], sequence_length
        ),
        "bin_path": str(bin_path),
        "bin_sha256": sha256_file(bin_path),
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
    }


def derive_schedule(
    train_metadata: dict[str, Any],
    *,
    batch_size: int,
    sequence_length: int,
    gradient_accumulation_steps: int,
    target_epochs: int = 15,
) -> dict[str, Any]:
    tokens_per_microbatch = batch_size * sequence_length
    tokens_per_optimizer_step = tokens_per_microbatch * gradient_accumulation_steps
    microbatches_per_epoch = math.ceil(train_metadata["sequence_count"] / batch_size)
    optimizer_steps_per_epoch = math.ceil(microbatches_per_epoch / gradient_accumulation_steps)
    max_steps = optimizer_steps_per_epoch * target_epochs
    allocated_chunk_slots_per_epoch = train_metadata["sequence_count"] * sequence_length
    optimizer_slot_capacity_per_epoch = optimizer_steps_per_epoch * tokens_per_optimizer_step
    padding_slots_per_epoch = (
        allocated_chunk_slots_per_epoch - train_metadata["valid_target_token_count"]
    )
    return {
        "packed_train_tokens": train_metadata["token_count"],
        "real_target_tokens_per_epoch": train_metadata["valid_target_token_count"],
        "allocated_token_slots_per_microbatch": tokens_per_microbatch,
        "allocated_token_slots_per_optimizer_step": tokens_per_optimizer_step,
        "train_sequences": train_metadata["sequence_count"],
        "microbatches_per_epoch": microbatches_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "allocated_chunk_slots_per_epoch": allocated_chunk_slots_per_epoch,
        "padding_slots_per_epoch": padding_slots_per_epoch,
        "padding_percentage_of_chunk_slots": (
            100.0 * padding_slots_per_epoch / allocated_chunk_slots_per_epoch
        ),
        "optimizer_slot_capacity_per_epoch": optimizer_slot_capacity_per_epoch,
        "unused_optimizer_capacity_slots_per_epoch": (
            optimizer_slot_capacity_per_epoch - allocated_chunk_slots_per_epoch
        ),
        "document_chunk_epochs": target_epochs,
        "actual_corpus_passes": target_epochs,
        "recommended_max_steps": max_steps,
        "recommended_warmup_steps": max(1, round(max_steps * 0.05)),
    }


def pack_from_config(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    training_config = config["training"]
    model_config = config["model"]
    corpus_path = Path(data_config["corpus_path"])
    split_path = Path(data_config["split_path"])
    metadata_path = Path(data_config["packed_metadata_path"])
    output_dir = metadata_path.parent

    for label, path in (("corpus", corpus_path), ("split", split_path)):
        expected_hash = data_config.get(f"expected_{label}_sha256")
        actual_hash = sha256_file(path)
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(
                f"Locked {label} hash changed: {actual_hash}, expected {expected_hash}"
            )

    records = load_corpus(corpus_path)
    assignments = load_split_assignments(split_path, records)
    tokenizer_path = Path(config["tokenizer"]["path"])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_hash = sha256_file(tokenizer_path)
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))

    split_ids = {
        split: {record["id"] for record in records if assignments[record["id"]] == split}
        for split in ("train", "validation", "test")
    }
    if any(split_ids[left] & split_ids[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("Document leakage detected between splits")
    if set.union(*split_ids.values()) != {record["id"] for record in records}:
        raise RuntimeError("Split assignments do not cover the corpus")

    split_metadata = {}
    for split in ("train", "validation", "test"):
        split_records = [record for record in records if assignments[record["id"]] == split]
        split_metadata[split] = write_split(
            split,
            split_records,
            tokenizer,
            output_dir,
            int(model_config["max_seq_len"]),
        )

    schedule = derive_schedule(
        split_metadata["train"],
        batch_size=int(training_config["batch_size"]),
        sequence_length=int(model_config["max_seq_len"]),
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        target_epochs=int(training_config["epochs"]),
    )
    schedule_candidates = {
        str(epochs): derive_schedule(
            split_metadata["train"],
            batch_size=int(training_config["batch_size"]),
            sequence_length=int(model_config["max_seq_len"]),
            gradient_accumulation_steps=int(
                training_config["gradient_accumulation_steps"]
            ),
            target_epochs=int(epochs),
        )
        for epochs in training_config.get(
            "schedule_epoch_candidates", [training_config["epochs"]]
        )
    }
    all_story_token_counts = []
    total_words = 0
    for record in records:
        token_ids, _ = encode_document(tokenizer, record)
        all_story_token_counts.append(len(token_ids))
        total_words += record["words"]
    sorted_token_counts = sorted(all_story_token_counts)

    total_tokens = sum(item["token_count"] for item in split_metadata.values())
    tokenization_statistics = {
        "definition": "full packed documents including story/control/BOS/EOS tokens",
        "total_corpus_tokens": total_tokens,
        "split_tokens": {
            split: split_metadata[split]["token_count"]
            for split in ("train", "validation", "test")
        },
        "tokens_per_word": total_tokens / total_words,
        "story_tokens": {
            "minimum": min(all_story_token_counts),
            "median": float(np.median(all_story_token_counts)),
            "mean": float(np.mean(all_story_token_counts)),
            "percentile_95": nearest_rank_percentile(sorted_token_counts, 0.95),
            "maximum": max(all_story_token_counts),
            "thresholds": {
                str(threshold): {
                    "count": sum(value <= threshold for value in all_story_token_counts),
                    "percentage": 100.0
                    * sum(value <= threshold for value in all_story_token_counts)
                    / len(all_story_token_counts),
                }
                for threshold in (1024, 2048, 4096)
            },
        },
        "chunks_per_story": chunk_statistics(
            all_story_token_counts, int(model_config["max_seq_len"])
        ),
    }
    metadata = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "split_seed": split_payload["seed"],
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_hash": tokenizer_hash,
        "dtype": "uint16",
        "document_format": "<|story|> [genre] <|bos|> title blank-line text <|eos|>",
        "splits": split_metadata,
        "schedule": schedule,
        "schedule_candidates": schedule_candidates,
        "tokenization_statistics": tokenization_statistics,
    }
    legacy_stats_path = corpus_path.with_suffix(".stats.json")
    if legacy_stats_path.exists():
        legacy_stats = json.loads(legacy_stats_path.read_text(encoding="utf-8"))
        metadata["dataset"] = legacy_stats.get("dataset")
        metadata["dataset_source_revision"] = legacy_stats.get("dataset_revision")
    corpus_stats_path = data_config.get("corpus_stats_path")
    if corpus_stats_path:
        stats_path = Path(corpus_stats_path)
        metadata["corpus_stats_path"] = str(stats_path)
        metadata["corpus_stats_sha256"] = sha256_file(stats_path)
    smoke_metadata_path = data_config.get("comparison_smoke_packed_metadata_path")
    if smoke_metadata_path:
        smoke_path = Path(smoke_metadata_path)
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        smoke_tokens = sum(item["token_count"] for item in smoke["splits"].values())
        smoke_words = sum(item["word_count"] for item in smoke["splits"].values())
        metadata["smoke_tokenizer_efficiency_comparison"] = {
            "smoke_packed_metadata_path": str(smoke_path),
            "smoke_packed_metadata_sha256": sha256_file(smoke_path),
            "smoke_total_tokens": smoke_tokens,
            "smoke_tokens_per_word": smoke_tokens / smoke_words,
            "data10m_tokens_per_word": tokenization_statistics["tokens_per_word"],
            "relative_tokens_per_word_change_percentage": 100.0
            * (tokenization_statistics["tokens_per_word"] / (smoke_tokens / smoke_words) - 1.0),
        }
    write_json_atomic(metadata_path, metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke-5m.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = pack_from_config(args.config)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
