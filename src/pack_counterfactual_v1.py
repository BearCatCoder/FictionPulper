"""Pack locked Counterfactual-v1 positive worlds without opening held-out splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml
from tokenizers import Tokenizer

from src.train_tokenizer import sha256_file, write_json_atomic


PACKED_SPLITS = ("train", "validation")


def _verify(path: Path, expected: str, label: str) -> str:
    if not path.is_file() or sha256_file(path) != expected:
        raise RuntimeError(f"Locked {label} hash changed: {path}")
    return expected


def verify_counterfactual_gates(
    config: Mapping[str, Any], *, verify_validation_source: bool = True
) -> dict[str, Any]:
    """Verify the dataset and model audit without reading held-out JSONL records."""
    locked = config["counterfactual_data"]
    root = Path(locked["directory"])
    manifest_path = root / "manifest.json"
    _verify(manifest_path, locked["expected_manifest_sha256"], "dataset manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or not manifest.get("safe_to_train"):
        raise RuntimeError("Counterfactual dataset manifest is not PASS/safe_to_train")
    if manifest.get("tokenizer_sha256") != config["tokenizer"]["expected_sha256"]:
        raise RuntimeError("Counterfactual dataset tokenizer lock changed")
    splits = PACKED_SPLITS if verify_validation_source else ("train",)
    for split in splits:
        specification = manifest["files"][f"{split}.jsonl"]
        _verify(root / f"{split}.jsonl", specification["sha256"], f"{split} source")

    audit_path = Path(locked["pretraining_audit_path"])
    _verify(audit_path, locked["expected_pretraining_audit_sha256"], "pretraining audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or not audit.get("gate", {}).get("passed"):
        raise RuntimeError("Counterfactual pretraining model audit is not PASS")
    inputs = audit.get("provenance", {})
    if inputs.get("dataset", {}).get("manifest_sha256") != locked["expected_manifest_sha256"]:
        raise RuntimeError("Pretraining audit names a different dataset manifest")
    if inputs.get("tokenizer", {}).get("sha256") != config["tokenizer"]["expected_sha256"]:
        raise RuntimeError("Pretraining audit names a different tokenizer")
    return {
        "manifest_sha256": locked["expected_manifest_sha256"],
        "pretraining_audit_sha256": locked["expected_pretraining_audit_sha256"],
        "pretraining_model_inputs": inputs.get("checkpoints", {}),
    }


def _load_pairs(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    _verify(path, expected_sha256, "counterfactual source")
    pairs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            pair = json.loads(line)
            if (
                not pair.get("pair_id")
                or set(pair.get("worlds", {})) != {"A", "B"}
                or set(pair.get("candidates", {})) != {"X", "Y"}
            ):
                raise ValueError(f"Malformed counterfactual pair at {path}:{line_number}")
            if pair["worlds"]["A"]["correct_candidate"] == pair["worlds"]["B"]["correct_candidate"]:
                raise ValueError(f"Pair does not reverse labels: {pair['pair_id']}")
            pairs.append(pair)
    if not pairs:
        raise ValueError(f"Counterfactual split is empty: {path}")
    return pairs


def _world_tokens(pair: Mapping[str, Any], world_name: str, tokenizer: Tokenizer) -> tuple[list[int], dict[str, Any]]:
    world = pair["worlds"][world_name]
    candidate_name = str(world["correct_candidate"])
    candidate = pair["candidates"][candidate_name]
    encoding = world["candidate_encodings"][candidate_name]
    token_ids = [int(value) for value in encoding["token_ids"]]
    start, end = int(encoding["decision_start"]), int(encoding["decision_end"])
    candidate_bytes = str(candidate["text"]).encode("utf-8")
    context_bytes = str(world["context"]).encode("utf-8")
    control = pair["metadata"].get("genre_control_token")
    prefix_length = 3 if control else 2
    if tokenizer.encode(str(world["context"]) + str(candidate["text"])).ids != token_ids[prefix_length:-1]:
        raise ValueError(f"Stored world encoding changed: {pair['pair_id']}:{world_name}")
    if token_ids[start:end] != [int(value) for value in candidate["token_ids"]]:
        raise ValueError(f"Candidate decision span changed: {pair['pair_id']}:{world_name}")
    return token_ids, {
        "pair_id": str(pair["pair_id"]),
        "world": world_name,
        "correct_candidate": candidate_name,
        "candidate_text": str(candidate["text"]),
        "candidate_utf8_hex": candidate_bytes.hex(),
        "candidate_utf8_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_byte_start": len(context_bytes),
        "candidate_byte_end": len(context_bytes) + len(candidate_bytes),
        "candidate_token_start": start,
        "candidate_token_end": end,
        "state_family": pair["abstract_counterfactual_variable"]["state_family"],
        "distance_bucket": pair["metadata"]["distance_bucket"],
        "difficulty": pair["metadata"]["difficulty"],
        "template_family": pair["metadata"]["template_family"],
    }


def _pack_split(
    pairs: Iterable[dict[str, Any]], *, split: str, output_dir: Path,
    tokenizer: Tokenizer, sequence_length: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / f"{split}.bin"
    temporary = bin_path.with_suffix(".bin.tmp")
    boundaries: list[dict[str, Any]] = []
    offset = 0
    pair_count = 0
    with temporary.open("wb") as output:
        for pair in pairs:
            pair_count += 1
            for world_name in ("A", "B"):
                token_ids, boundary = _world_tokens(pair, world_name, tokenizer)
                np.asarray(token_ids, dtype=np.uint16).tofile(output)
                boundary.update({
                    "id": f"{pair['pair_id']}:{world_name}",
                    "offset": offset,
                    "length": len(token_ids),
                    "candidate_absolute_token_start": offset + boundary["candidate_token_start"],
                    "candidate_absolute_token_end": offset + boundary["candidate_token_end"],
                })
                boundaries.append(boundary)
                offset += len(token_ids)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(bin_path)
    if split == "train":
        documents = [{
            "id": "counterfactual-v1-train-stream", "offset": 0, "length": offset,
            "sequence_count": math.ceil((offset - 1) / sequence_length),
        }]
        policy = "continuous_eos_delimited"
    else:
        documents = [{
            "id": boundary["id"], "offset": boundary["offset"], "length": boundary["length"],
            "sequence_count": math.ceil((boundary["length"] - 1) / sequence_length),
        } for boundary in boundaries]
        policy = "world_bounded"
    index_path = bin_path.with_suffix(".index.json")
    write_json_atomic(index_path, {
        "split": split,
        "packing_policy": policy,
        "documents": documents,
        "pair_count": pair_count,
        "world_count": len(boundaries),
        "world_boundaries": boundaries,
    })
    return {
        "pair_count": pair_count,
        "world_count": len(boundaries),
        "token_count": offset,
        "valid_target_token_count": sum(int(item["length"]) - 1 for item in documents),
        "sequence_count": sum(int(item["sequence_count"]) for item in documents),
        "packing_policy": policy,
        "bin_path": str(bin_path), "bin_sha256": sha256_file(bin_path),
        "index_path": str(index_path), "index_sha256": sha256_file(index_path),
    }


def pack_counterfactual(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gates = verify_counterfactual_gates(config)
    tokenizer_path = Path(config["tokenizer"]["path"])
    _verify(tokenizer_path, config["tokenizer"]["expected_sha256"], "tokenizer")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    root = Path(config["counterfactual_data"]["directory"])
    output_dir = Path(config["counterfactual_packing"]["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite counterfactual packed data: {output_dir}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    splits = {}
    try:
        for split in PACKED_SPLITS:
            source_path = root / f"{split}.jsonl"
            source_hash = manifest["files"][source_path.name]["sha256"]
            splits[split] = _pack_split(
                _load_pairs(source_path, source_hash), split=split, output_dir=output_dir,
                tokenizer=tokenizer, sequence_length=int(config["model"]["max_seq_len"]),
            )
            splits[split].update({"source_path": str(source_path), "source_sha256": source_hash})
        metadata = {
            "version": 1,
            "policy": "positive A-correct and B-correct worlds only; test/generalization unopened",
            "tokenizer_sha256": config["tokenizer"]["expected_sha256"],
            "dataset_gates": gates,
            "splits": splits,
        }
        write_json_atomic(output_dir / "metadata.json", metadata)
        return metadata
    except Exception:
        # Only the newly-created isolated directory may be cleaned on a failed build.
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def attach_pair_presentations(
    schedule: dict[str, Any], *, source_name: str, packed_index: Mapping[str, Any]
) -> dict[str, int]:
    """Attach candidate-span-overlapping pairs, deduplicated per microbatch."""
    boundaries = list(packed_index.get("world_boundaries", []))
    by_document: dict[str, list[dict[str, Any]]] = {}
    for boundary in boundaries:
        by_document.setdefault("counterfactual-v1-train-stream", []).append(boundary)
    for values in by_document.values():
        values.sort(key=lambda item: int(item["candidate_absolute_token_start"]))

    batch_size = int(schedule["batch_size"])
    pair_presentations = 0
    world_triggers = 0
    for batch_start in range(0, len(schedule["entries"]), batch_size):
        batch = schedule["entries"][batch_start:batch_start + batch_size]
        selected: dict[str, dict[str, Any]] = {}
        owners: dict[str, dict[str, Any]] = {}
        for entry in batch:
            entry["pair_presentations"] = []
            if entry["source"] != source_name:
                continue
            target_start = int(entry["relative_start"]) + 1
            target_end = target_start + int(entry["valid_targets"])
            for boundary in by_document.get(str(entry["document_id"]), []):
                start = int(boundary["candidate_absolute_token_start"])
                end = int(boundary["candidate_absolute_token_end"])
                if start >= target_end:
                    break
                if end <= target_start:
                    continue
                pair_id = str(boundary["pair_id"])
                owners.setdefault(pair_id, entry)
                attachment = selected.setdefault(pair_id, {
                    "pair_id": pair_id, "trigger_worlds": [],
                    "state_family": boundary["state_family"],
                    "distance_bucket": boundary["distance_bucket"],
                    "difficulty": boundary["difficulty"],
                })
                if boundary["world"] not in attachment["trigger_worlds"]:
                    attachment["trigger_worlds"].append(boundary["world"])
        for pair_id, attachment in selected.items():
            owners[pair_id]["pair_presentations"].append(attachment)
            pair_presentations += 1
            world_triggers += len(attachment["trigger_worlds"])
    exposure = {
        "pair_presentations": pair_presentations,
        "candidate_span_world_triggers": world_triggers,
        "unique_pairs": len({item["pair_id"] for entry in schedule["entries"] for item in entry["pair_presentations"]}),
    }
    schedule["pair_attachment"] = {
        "source": source_name,
        "policy": "correct-candidate decision-span overlap; each pair at most once per microbatch",
        **exposure,
    }
    return exposure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(pack_counterfactual(args.config), indent=2))


if __name__ == "__main__":
    main()
