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


PREPROCESSING_VERSION = 5
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

    return {
        "document_count": len(records),
        "word_count": sum(record["words"] for record in records),
        "token_count": offset,
        "valid_target_token_count": sum(document["length"] - 1 for document in index),
        "sequence_count": sum(document["sequence_count"] for document in index),
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

    records = load_corpus(corpus_path)
    assignments = load_split_assignments(split_path, records)
    tokenizer_path = Path(config["tokenizer"]["path"])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_hash = sha256_file(tokenizer_path)
    corpus_stats = json.loads(corpus_path.with_suffix(".stats.json").read_text(encoding="utf-8"))
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
    )
    metadata = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "dataset": corpus_stats["dataset"],
        "dataset_source_revision": corpus_stats["dataset_revision"],
        "split_seed": split_payload["seed"],
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_hash": tokenizer_hash,
        "dtype": "uint16",
        "document_format": "<|story|> [genre] <|bos|> title blank-line text <|eos|>",
        "splits": split_metadata,
        "schedule": schedule,
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
