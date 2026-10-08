"""Build and validate deterministic hard negatives for Narrative-v1 train stories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml
import numpy as np
from tokenizers import Tokenizer

from src.train_tokenizer import sha256_file, write_json_atomic


GENERATOR_VERSION = "contrastive-v1-primary-anchor-2"
SUPPORTED_STATE_TYPES = {
    "character_identity_role_relationship",
    "ownership_transfer_hiding_retrieval",
    "physical_scene_location_movement",
    "goal_persistence_completion_failure",
    "knowledge_ignorance_secret",
    "cause_effect",
    "promise_debt_obligation",
    "persistent_injury",
    "temporal_ordering",
}


class PairValidationError(ValueError):
    """Raised when a pair cannot be proved well-formed and state-contradicting."""


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not (value.get("id") or value.get("pair_id")):
                raise PairValidationError(f"Malformed record at {path}:{line_number}")
            records.append(value)
    if not records:
        raise PairValidationError(f"Empty input: {path}")
    return records


def _events(sidecar: dict[str, Any], state_type: str) -> list[dict[str, Any]]:
    return [event for event in sidecar["events"] if event["state_type"] == state_type]


def _wrong_resolution(sidecar: dict[str, Any], state_type: str) -> tuple[str, Any, Any]:
    events = _events(sidecar, state_type)
    names = sidecar["generation_values"]["names"]
    objects = sidecar["generation_values"]["objects"]
    locations = sidecar["generation_values"]["locations"]
    final = {(event["entity"], event["attribute"]): event["value"] for event in events}
    if state_type == "character_identity_role_relationship":
        role_event = next(event for event in events if event["attribute"] == "role")
        relationship = next(event for event in events if event["attribute"].startswith("relationship_to:"))
        expected = {"role_holder": role_event["entity"], "relationship_holder": relationship["entity"]}
        violating = {"role_holder": relationship["entity"], "relationship_holder": role_event["entity"]}
        text = f"Later, {names[2]} identified {relationship['entity']} as the {role_event['value']} and {role_event['entity']} as the trusted cousin."
    elif state_type == "ownership_transfer_hiding_retrieval":
        holder_events = [event for event in events if event["attribute"] == "holder"]
        hidden = next(event for event in events if event["attribute"] == "hidden_at")
        expected = {"holder": holder_events[-1]["value"], "retrieved": True}
        violating = {"holder": holder_events[0]["value"], "retrieved": False}
        text = f"At last, {holder_events[0]['value']} left the {holder_events[0]['entity']} hidden at the {hidden['value']} and did not return it."
    elif state_type == "physical_scene_location_movement":
        expected_location = next(value for (entity, attribute), value in final.items() if entity == names[0] and attribute == "location")
        wrong_location = next(value for (entity, attribute), value in final.items() if entity == names[0] and attribute == "previous_location")
        expected, violating = expected_location, wrong_location
        text = f"By nightfall, {names[0]}, {names[1]}, and the {objects[1]} had returned to the {wrong_location}."
    elif state_type == "goal_persistence_completion_failure":
        status = [event["value"] for event in events if event["attribute"] == "goal_status"][-1]
        wrong = "failed" if status == "completed" else "completed"
        goal = next(event["value"] for event in events if event["attribute"] == "goal")
        expected, violating = status, wrong
        text = f"The final report marked the goal to {goal} as {wrong}."
    elif state_type == "knowledge_ignorance_secret":
        learner = next(event for event in events if event["attribute"].startswith("knows:") and event["value"] is False)
        secret = learner["attribute"].removeprefix("knows:")
        expected, violating = True, False
        text = f"Even after the meeting, {learner['entity']} remained ignorant that {secret}."
    elif state_type == "cause_effect":
        cause = next(event["entity"] for event in events if event["attribute"] == "occurred")
        effect = next(event["entity"] for event in events if event["attribute"] == "caused_by")
        expected, violating = {"cause": cause, "effect": effect}, {"cause": effect, "effect": cause}
        text = f"The inquiry instead concluded that {effect} had caused the fact that {cause}."
    elif state_type == "promise_debt_obligation":
        obligation = next(event for event in events if event["attribute"].startswith("obligation_to:"))
        status = [event["value"] for event in events if event["attribute"] == obligation["attribute"]][-1]
        wrong = "fulfilled" if status == "outstanding" else "outstanding"
        expected, violating = status, wrong
        text = f"At the final accounting, {obligation['entity']}'s obligation was recorded as {wrong}."
    elif state_type == "persistent_injury":
        injury = next(event["value"] for event in events if event["attribute"] == "injury")
        person = next(event["entity"] for event in events if event["attribute"] == "injury")
        expected, violating = "unhealed", "healed"
        text = f"Hours later, {person}'s {injury} had healed completely and caused no further pain."
    elif state_type == "temporal_ordering":
        first = final[("timeline", "first")]
        second = final[("timeline", "second")]
        third = final[("timeline", "third")]
        expected, violating = [first, second, third], [second, first, third]
        text = f"The final timeline placed {second} first, {first} second, and {third} last."
    else:
        raise PairValidationError(f"Unsupported primary state type: {state_type}")
    return text, expected, violating


def build_pair(
    record: dict[str, Any],
    sidecar: dict[str, Any],
    tokenizer: Tokenizer,
    boundary: dict[str, Any],
    *,
    packed_document_id: str | None = None,
    packed_document_start: int = 0,
) -> dict[str, Any]:
    if record["id"] != sidecar.get("id") or record["id"] != boundary.get("id"):
        raise PairValidationError("Record, sidecar, and packed boundary IDs differ")
    primary_type = sidecar.get("primary_distance", {}).get("state_type")
    if primary_type not in SUPPORTED_STATE_TYPES:
        raise PairValidationError(f"{record['id']} has unsupported primary state")
    anchors = [item for item in sidecar["anchors"] if item["state_type"] == primary_type]
    if len(anchors) != 1:
        raise PairValidationError(f"{record['id']} has an ambiguous primary anchor")
    anchor = anchors[0]
    positive = anchor["resolved_quote"]
    if record["text"].count(positive) != 1:
        raise PairValidationError(f"{record['id']} positive resolution is not unique")
    character_start = record["text"].index(positive)
    if character_start != sidecar["primary_distance"]["resolved_character_start"]:
        raise PairValidationError(f"{record['id']} primary character offset differs")
    prefix = record["text"][:character_start]
    negative, expected_state, violating_state = _wrong_resolution(sidecar, primary_type)
    if not negative.endswith(".") or positive == negative or negative in record["text"]:
        raise PairValidationError(f"{record['id']} negative is not a distinct sentence")
    controls = [tokenizer.token_to_id("<|story|>")]
    if record.get("genre_control_token"):
        controls.append(tokenizer.token_to_id(record["genre_control_token"]))
    controls.append(tokenizer.token_to_id("<|bos|>"))
    if any(token is None for token in controls):
        raise PairValidationError("Tokenizer lacks a required control token")
    prefix_ids = [*controls, *tokenizer.encode(prefix).ids]
    positive_ids = [*controls, *tokenizer.encode(prefix + positive).ids]
    negative_ids = [*controls, *tokenizer.encode(prefix + negative).ids]
    if positive_ids[: len(prefix_ids)] != prefix_ids or negative_ids[: len(prefix_ids)] != prefix_ids:
        raise PairValidationError(f"{record['id']} has an ambiguous tokenization boundary")
    decision_start = len(prefix_ids)
    pair = {
        "pair_id": f"{record['id']}:{primary_type}",
        "document_id": record["id"],
        "generator_version": GENERATOR_VERSION,
        "state_type": primary_type,
        "template_family": record["template_family"],
        "positive_text": positive,
        "negative_texts": [negative],
        "prefix_text": prefix,
        "character_offsets": {"decision_start": character_start, "positive_decision_end": character_start + len(positive)},
        "token_offsets": {
            "decision_start": decision_start,
            "positive_decision_end": len(positive_ids),
            "negative_decision_ends": [len(negative_ids)],
        },
        "packed_offsets": {
            "packed_document_id": packed_document_id or record["id"],
            "packed_document_start": packed_document_start,
            "document_start": int(boundary["offset"]),
            "document_length": int(boundary["length"]),
            "decision_start": int(boundary["offset"]) + decision_start,
            "positive_decision_end": int(boundary["offset"]) + len(positive_ids),
        },
        "lengths": {"prefix_tokens": decision_start, "positive_tokens": len(positive_ids), "negative_tokens": [len(negative_ids)]},
        "expected_state_value": expected_state,
        "violating_state_values": [violating_state],
        "positive_token_ids": positive_ids,
        "negative_token_ids": [negative_ids],
    }
    pair["validation_hash"] = _canonical_hash(pair)
    validate_pair(pair, max_seq_len=1024)
    return pair


def validate_pair(pair: dict[str, Any], *, max_seq_len: int) -> None:
    recorded_hash = pair.get("validation_hash")
    unhashed = {key: value for key, value in pair.items() if key != "validation_hash"}
    if recorded_hash != _canonical_hash(unhashed):
        raise PairValidationError("Pair validation hash is missing or invalid")
    required = {"generator_version", "character_offsets", "token_offsets", "packed_offsets", "lengths", "expected_state_value", "violating_state_values"}
    if not required <= pair.keys() or pair["generator_version"] != GENERATOR_VERSION:
        raise PairValidationError("Pair metadata is incomplete or has the wrong version")
    start = int(pair["token_offsets"]["decision_start"])
    positive = pair["positive_token_ids"]
    negatives = pair["negative_token_ids"]
    if not negatives or any(tokens == positive for tokens in negatives):
        raise PairValidationError("A negative is missing or identical to the positive")
    if not 0 < start < len(positive) <= max_seq_len:
        raise PairValidationError("Positive decision span is empty or outside model context")
    if any(tokens[:start] != positive[:start] or not start < len(tokens) <= max_seq_len for tokens in negatives):
        raise PairValidationError("Negative context differs or its decision span is invalid")
    offsets = pair["token_offsets"]
    lengths = pair["lengths"]
    if int(offsets["positive_decision_end"]) != len(positive) or list(offsets["negative_decision_ends"]) != [len(tokens) for tokens in negatives]:
        raise PairValidationError("Decision token offsets differ from encoded lengths")
    if lengths != {"prefix_tokens": start, "positive_tokens": len(positive), "negative_tokens": [len(tokens) for tokens in negatives]}:
        raise PairValidationError("Recorded pair lengths are incorrect")
    packed = pair["packed_offsets"]
    if not packed.get("packed_document_id") or int(packed["document_start"]) < int(packed["packed_document_start"]):
        raise PairValidationError("Packed parent document metadata is invalid")
    if int(packed["decision_start"]) != int(packed["document_start"]) + start or int(packed["positive_decision_end"]) != int(packed["document_start"]) + len(positive):
        raise PairValidationError("Packed offsets are inconsistent")
    characters = pair["character_offsets"]
    if int(characters["positive_decision_end"]) - int(characters["decision_start"]) != len(pair["positive_text"]):
        raise PairValidationError("Positive character offsets are inconsistent")
    if len(pair["negative_texts"]) != len(negatives) or any(not text.endswith(".") for text in pair["negative_texts"]):
        raise PairValidationError("Negative prose is missing or not sentence-formed")
    if any(value == pair["expected_state_value"] for value in pair["violating_state_values"]):
        raise PairValidationError("Negative does not violate the expected state")


def load_pairs(path: Path, *, max_seq_len: int) -> list[dict[str, Any]]:
    pairs = _read_jsonl(path)
    for pair in pairs:
        validate_pair(pair, max_seq_len=max_seq_len)
    if len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise PairValidationError("Pair IDs are not unique")
    return pairs


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as output:
        for value in values:
            output.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _build_pair_values(
    records: list[dict[str, Any]],
    sidecars: list[dict[str, Any]],
    tokenizer: Tokenizer,
    *,
    split: str,
    boundaries: dict[str, dict[str, Any]] | None = None,
    packed_document_id: str | None = None,
    packed_document_start: int = 0,
) -> list[dict[str, Any]]:
    if any(record.get("split") != split for record in records) or any(sidecar.get("split") != split for sidecar in sidecars):
        raise PairValidationError(f"Pair sources contain records outside requested split {split}")
    by_sidecar = {value["id"]: value for value in sidecars}
    if set(by_sidecar) != {record["id"] for record in records}:
        raise PairValidationError("Record and sidecar IDs differ")
    if boundaries is None:
        offset = packed_document_start
        boundaries = {}
        for record in records:
            boundaries[record["id"]] = {
                "id": record["id"], "offset": offset, "length": int(record["token_count"])
            }
            offset += int(record["token_count"])
    if set(boundaries) != set(by_sidecar):
        raise PairValidationError("Records, sidecars, and packed boundaries differ")
    return [
        build_pair(
            record,
            by_sidecar[record["id"]],
            tokenizer,
            boundaries[record["id"]],
            packed_document_id=packed_document_id or record["id"],
            packed_document_start=packed_document_start,
        )
        for record in records
    ]


def _persist_pairs(
    pairs: list[dict[str, Any]],
    *,
    split: str,
    output_dir: Path,
    tokenizer_hash: str,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite contrastive pairs: {output_dir}")
    output_dir.mkdir(parents=True)
    pairs_path = output_dir / f"{split}-pairs.jsonl"
    _write_jsonl(pairs_path, pairs)
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "split": split,
        "pair_count": len(pairs),
        "negative_count": sum(len(pair["negative_token_ids"]) for pair in pairs),
        "tokenizer_sha256": tokenizer_hash,
        "source_hashes": source_hashes,
        "pairs_path": str(pairs_path),
        "pairs_sha256": sha256_file(pairs_path),
        "validation_hash": _canonical_hash([pair["validation_hash"] for pair in pairs]),
        "rejected_ambiguous": 0,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def build_pairs(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pair_config = config["contrastive"]["pairs"]
    locked = {
        "records": Path(pair_config["records_path"]),
        "sidecars": Path(pair_config["sidecars_path"]),
        "packed": Path(pair_config["packed_path"]),
        "index": Path(pair_config["packed_index_path"]),
    }
    for name, path in locked.items():
        if sha256_file(path) != pair_config[f"expected_{name}_sha256"]:
            raise RuntimeError(f"Locked pair source changed: {name}")
    tokenizer_path = Path(config["tokenizer"]["path"])
    if sha256_file(tokenizer_path) != config["tokenizer"]["expected_sha256"]:
        raise RuntimeError("Locked tokenizer hash changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    records = _read_jsonl(locked["records"])
    sidecars = _read_jsonl(locked["sidecars"])
    index = json.loads(locked["index"].read_text(encoding="utf-8"))
    boundaries = {value["id"]: value for value in index["story_boundaries"]}
    packed_documents = index.get("documents", [])
    if len(packed_documents) != 1:
        raise PairValidationError("Train curriculum must have exactly one scheduled packed document")
    packed_document = packed_documents[0]
    pairs = _build_pair_values(
        records,
        sidecars,
        tokenizer,
        split="train",
        boundaries=boundaries,
        packed_document_id=str(packed_document["id"]),
        packed_document_start=int(packed_document["offset"]),
    )
    packed_tokens = np.memmap(locked["packed"], dtype=np.uint16, mode="r")
    for pair in pairs:
        start = int(pair["packed_offsets"]["document_start"])
        positive = pair["positive_token_ids"]
        if packed_tokens[start : start + len(positive)].tolist() != positive:
            raise PairValidationError(f"{pair['pair_id']} positive differs from locked packed data")
    return _persist_pairs(
        pairs,
        split="train",
        output_dir=Path(pair_config["output_dir"]),
        tokenizer_hash=config["tokenizer"]["expected_sha256"],
        source_hashes={name: sha256_file(path) for name, path in locked.items()},
    )


def build_post_selection_pairs(
    *,
    split: str,
    records_path: Path,
    records_sha256: str,
    sidecars_path: Path,
    sidecars_sha256: str,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    if split not in {"test", "generalization_holdout"}:
        raise ValueError("Post-selection split must be test or generalization_holdout")
    sources = {"records": (records_path, records_sha256), "sidecars": (sidecars_path, sidecars_sha256)}
    for name, (path, expected) in sources.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"Supplied post-selection {name} hash changed")
    if sha256_file(tokenizer_path) != tokenizer_sha256:
        raise RuntimeError("Supplied post-selection tokenizer hash changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pairs = _build_pair_values(
        _read_jsonl(records_path), _read_jsonl(sidecars_path), tokenizer, split=split
    )
    return _persist_pairs(
        pairs,
        split=split,
        output_dir=output_dir,
        tokenizer_hash=tokenizer_sha256,
        source_hashes={name: expected for name, (_, expected) in sources.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--post-selection-split", choices=("test", "generalization_holdout"))
    parser.add_argument("--records", type=Path)
    parser.add_argument("--records-sha256")
    parser.add_argument("--sidecars", type=Path)
    parser.add_argument("--sidecars-sha256")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.validate:
        pairs = load_pairs(args.validate, max_seq_len=1024)
        print(json.dumps({"pair_count": len(pairs), "passed": True}, indent=2))
    elif args.post_selection_split:
        required = (args.records, args.records_sha256, args.sidecars, args.sidecars_sha256, args.output_dir)
        if any(value is None for value in required):
            parser.error("post-selection generation requires records, sidecars, hashes, and output-dir")
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        print(json.dumps(build_post_selection_pairs(
            split=args.post_selection_split,
            records_path=args.records,
            records_sha256=args.records_sha256,
            sidecars_path=args.sidecars,
            sidecars_sha256=args.sidecars_sha256,
            tokenizer_path=Path(config["tokenizer"]["path"]),
            tokenizer_sha256=config["tokenizer"]["expected_sha256"],
            output_dir=args.output_dir,
        ), indent=2))
    else:
        print(json.dumps(build_pairs(args.config), indent=2))


if __name__ == "__main__":
    main()
