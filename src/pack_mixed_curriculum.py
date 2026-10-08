"""Pack curriculum train/validation data without opening its sealed test split."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tokenizers import Tokenizer

from src.train_tokenizer import sha256_file, write_json_atomic


PACKED_SPLITS = ("train", "validation")


def canonical_document_text(record: dict[str, Any]) -> str:
    """Curriculum JSONL stores the title once at the start of complete text."""
    title = str(record.get("title", ""))
    text = str(record.get("text", ""))
    if not title or not text.startswith(title + "\n\n"):
        raise ValueError("Curriculum text must use the canonical 'title\\n\\nbody' form")
    return text


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                record = json.loads(line)
                if not record.get("id") or not record.get("text"):
                    raise ValueError(f"Malformed curriculum record at {path}:{line_number}")
                records.append(record)
    if not records:
        raise ValueError(f"Curriculum split is empty: {path}")
    return records


def _pack_split(
    records: list[dict[str, Any]],
    *,
    split: str,
    output_dir: Path,
    tokenizer: Tokenizer,
    sequence_length: int,
    continuous_stream: bool = False,
) -> dict[str, Any]:
    story_id = tokenizer.token_to_id("<|story|>")
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    if story_id is None or bos_id is None or eos_id is None:
        raise RuntimeError("Tokenizer is missing curriculum document control tokens")
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / f"{split}.bin"
    temporary = bin_path.with_suffix(".bin.tmp")
    documents = []
    story_boundaries = []
    offset = 0
    with temporary.open("wb") as output:
        for record in records:
            content = tokenizer.encode(canonical_document_text(record)).ids
            control = record.get("genre_control_token")
            control_ids = []
            if control is not None:
                control_id = tokenizer.token_to_id(str(control))
                if control_id is None:
                    raise RuntimeError(f"Record {record['id']} uses an unknown genre token")
                control_ids.append(control_id)
            token_ids = [story_id, *control_ids, bos_id, *content, eos_id]
            np.asarray(token_ids, dtype=np.uint16).tofile(output)
            story_boundaries.append(
                {
                    "id": str(record["id"]),
                    "offset": offset,
                    "length": len(token_ids),
                    "difficulty": record.get("difficulty"),
                    "distance_bucket": record.get("distance_bucket"),
                    "template_family": record.get("template_family"),
                }
            )
            offset += len(token_ids)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(bin_path)
    if continuous_stream:
        documents = [
            {
                "id": "narrative-curriculum-v1-train-stream",
                "offset": 0,
                "length": offset,
                "sequence_count": math.ceil((offset - 1) / sequence_length),
            }
        ]
    else:
        documents = [
            {
                **story,
                "sequence_count": math.ceil((int(story["length"]) - 1) / sequence_length),
            }
            for story in story_boundaries
        ]
    index_path = bin_path.with_suffix(".index.json")
    write_json_atomic(
        index_path,
        {
            "split": split,
            "packing_policy": (
                "continuous EOS-delimited story stream; chunk boundaries may cross only "
                "the explicit <|eos|><|story|> boundary"
                if continuous_stream
                else "story-bounded"
            ),
            "story_count": len(story_boundaries),
            "story_boundaries": story_boundaries if continuous_stream else None,
            "documents": documents,
        },
    )
    valid_targets = sum(int(document["length"]) - 1 for document in documents)
    sequence_count = sum(int(document["sequence_count"]) for document in documents)
    return {
        "document_count": len(documents),
        "story_count": len(story_boundaries),
        "packing_policy": "continuous_eos_delimited" if continuous_stream else "story_bounded",
        "token_count": offset,
        "valid_target_token_count": valid_targets,
        "sequence_count": sequence_count,
        "allocated_chunk_slots": sequence_count * sequence_length,
        "bin_path": str(bin_path),
        "bin_sha256": sha256_file(bin_path),
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
    }


def pack_training_curriculum(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pack = config["curriculum_packing"]
    source_dir = Path(pack["source_dir"])
    output_dir = Path(pack["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite packed curriculum: {output_dir}")
    tokenizer_path = Path(config["tokenizer"]["path"])
    tokenizer_hash = sha256_file(tokenizer_path)
    if tokenizer_hash != config["tokenizer"]["expected_sha256"]:
        raise RuntimeError("Locked tokenizer hash changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    results = {}
    for split in PACKED_SPLITS:
        source_path = source_dir / f"{split}.jsonl"
        results[split] = _pack_split(
            _load_jsonl(source_path),
            split=split,
            output_dir=output_dir,
            tokenizer=tokenizer,
            sequence_length=int(config["model"]["max_seq_len"]),
            continuous_stream=split == "train",
        )
        results[split]["source_path"] = str(source_path)
        results[split]["source_sha256"] = sha256_file(source_path)
    metadata = {
        "policy": "training and diagnostic validation only; test is not opened or packed",
        "tokenizer_sha256": tokenizer_hash,
        "sequence_length": int(config["model"]["max_seq_len"]),
        "splits": results,
    }
    write_json_atomic(output_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(pack_training_curriculum(args.config), indent=2))


if __name__ == "__main__":
    main()
