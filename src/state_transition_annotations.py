"""Deterministic State-Transition-v1 sidecars for sealed Counterfactual-v1 data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

from tokenizers import Tokenizer

from src.continuity_curriculum.common import canonical_json, sha256_bytes, sha256_file


SCHEMA_VERSION = 1
GENERATOR_VERSION = "state-transition-v1"
SOURCE_SPLITS = ("train", "validation", "test", "generalization_holdout")
CARRY_OFFSETS = (32, 64, 128, 256)

FAMILY_MAP = {
    "object_ownership": "object_owner",
    "object_location": "object_location",
    "character_location": "character_location",
    "goal_intention": "goal_status",
    "knowledge_secret_holder": "knowledge_holder",
    "relationship": "relationship_state",
    "persistent_physical_state": "physical_condition",
    "object_door_state": "object_door_state",
    "cause_effect": "causal_state",
    "latest_state_update": "object_location",
}


class StateTransitionValidationError(ValueError):
    """Raised when a sidecar cannot be proved from its sealed source pair."""


def _hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _record_hash(record: dict[str, Any]) -> str:
    return _hash({key: value for key, value in record.items() if key != "stable_hash"})


def _canonical_jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return b"".join((canonical_json(record) + "\n").encode("utf-8") for record in records)


def _contains_subsequence(values: Sequence[int], needle: Sequence[int]) -> bool:
    return bool(needle) and any(
        list(values[index : index + len(needle)]) == list(needle)
        for index in range(len(values) - len(needle) + 1)
    )


def _indexed_slot(value: str, values: Sequence[str], prefix: str) -> str:
    try:
        return f"{prefix}_{list(values).index(value)}"
    except ValueError as error:
        raise StateTransitionValidationError(
            f"Value {value!r} is absent from lexical slot {prefix!r}"
        ) from error


def _entity_slot(pair: dict[str, Any]) -> str:
    variable = pair["abstract_counterfactual_variable"]
    family = variable["state_family"]
    entity = variable["entity"]
    slots = pair["metadata"]["lexical_slots"]
    if family in {"object_ownership", "object_location", "object_door_state", "latest_state_update"}:
        return _indexed_slot(entity, slots["objects"], "object")
    if family in {"character_location", "goal_intention", "relationship", "persistent_physical_state"}:
        return _indexed_slot(entity, slots["names"], "character")
    if family == "knowledge_secret_holder":
        if entity != slots["secret"]:
            raise StateTransitionValidationError("Knowledge entity is not the sealed secret slot")
        return "secret_0"
    if family == "cause_effect":
        if entity != slots["effect"]:
            raise StateTransitionValidationError("Causal entity is not the sealed effect slot")
        return "effect_0"
    raise StateTransitionValidationError(f"Unsupported source family: {family}")


def _state_class(pair: dict[str, Any], value: str) -> str:
    family = pair["abstract_counterfactual_variable"]["state_family"]
    slots = pair["metadata"]["lexical_slots"]
    if family in {"object_ownership", "knowledge_secret_holder"}:
        return _indexed_slot(value, slots["names"], "character")
    if family in {"object_location", "character_location", "latest_state_update"}:
        return _indexed_slot(value, slots["locations"], "location")
    if family == "goal_intention":
        return _indexed_slot(value, slots["goals"], "goal")
    if family == "cause_effect":
        return _indexed_slot(value, slots["causes"], "cause")
    if family == "relationship":
        classes = {"trusted ally": "relationship_trusted_ally", "declared rival": "relationship_declared_rival"}
    elif family == "persistent_physical_state":
        injury = slots["injury"]
        classes = {
            f"still affected by the {injury}": "physical_affected",
            f"fully recovered from the {injury}": "physical_recovered",
        }
    elif family == "object_door_state":
        classes = {"locked": "door_locked", "open": "door_open"}
    else:
        raise StateTransitionValidationError(f"Unsupported source family: {family}")
    try:
        return classes[value]
    except KeyError as error:
        raise StateTransitionValidationError(
            f"Unsupported value {value!r} for source family {family!r}"
        ) from error


def _event_span(context: str, context_ids: Sequence[int], evidence: str, tokenizer: Tokenizer,
                search_from: int) -> tuple[dict[str, Any], int]:
    char_start = context.find(evidence, search_from)
    if char_start < 0:
        raise StateTransitionValidationError(f"Event evidence is absent from context: {evidence!r}")
    char_end = char_start + len(evidence)
    token_start = len(tokenizer.encode(context[:char_start]).ids)
    token_end = len(tokenizer.encode(context[:char_end]).ids)
    span = {
        "byte_start": len(context[:char_start].encode("utf-8")),
        "byte_end": len(context[:char_end].encode("utf-8")),
        "token_start": token_start,
        "token_end": token_end,
        "token_ids": list(context_ids[token_start:token_end]),
    }
    return span, char_end


def _metadata(pair: dict[str, Any], transition_count: int) -> dict[str, Any]:
    source = pair["metadata"]
    return {
        "difficulty": source["difficulty"],
        "genre": source["genre"],
        "distance_bucket": source["distance_bucket"],
        "transition_count": transition_count,
    }


def annotate_pair(pair: dict[str, Any], tokenizer: Tokenizer) -> list[dict[str, Any]]:
    """Derive transition and carry annotations without modifying source text."""
    if pair.get("stable_hash") != _record_hash(pair):
        raise StateTransitionValidationError("Source stable hash mismatch")
    source_family = pair["abstract_counterfactual_variable"]["state_family"]
    try:
        family = FAMILY_MAP[source_family]
    except KeyError as error:
        raise StateTransitionValidationError(f"Unsupported source family: {source_family}") from error
    entity_slot = _entity_slot(pair)
    records: list[dict[str, Any]] = []
    for world_name in ("A", "B"):
        world = pair["worlds"][world_name]
        context = world["context"]
        context_ids = world["context_token_ids"]
        if tokenizer.encode(context).ids != context_ids:
            raise StateTransitionValidationError("Source context tokenization mismatch")
        events = world["state_events"]
        spans = []
        search_from = 0
        for event in events:
            span, search_from = _event_span(
                context, context_ids, event["evidence"], tokenizer, search_from
            )
            spans.append(span)
        previous: str | None = None
        transition_records = []
        for event, span in zip(events, spans, strict=True):
            post_state = _state_class(pair, event["value"])
            record = {
                "schema_version": SCHEMA_VERSION,
                "generator_version": GENERATOR_VERSION,
                "source_stable_hash": pair["stable_hash"],
                "split": pair["split"],
                "pair": pair["pair_id"],
                "world": world_name,
                "event": event["sequence"],
                "kind": "transition",
                "family": family,
                "entity_slot": entity_slot,
                "pre_state": previous,
                "transition": f"set_{family}",
                "post_state": post_state,
                "event_evidence": event["evidence"],
                "event_span": span,
                "prediction_token_index": span["token_end"],
                "metadata": _metadata(pair, len(events)),
            }
            record["stable_hash"] = _record_hash(record)
            transition_records.append(record)
            previous = post_state
        records.extend(transition_records)
        for index, transition in enumerate(transition_records):
            boundary = spans[index + 1]["token_start"] if index + 1 < len(spans) else len(context_ids)
            for distance in CARRY_OFFSETS:
                prediction_index = transition["prediction_token_index"] + distance
                if prediction_index >= boundary:
                    continue
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "generator_version": GENERATOR_VERSION,
                    "source_stable_hash": pair["stable_hash"],
                    "split": pair["split"],
                    "pair": pair["pair_id"],
                    "world": world_name,
                    "event": transition["event"],
                    "kind": "carry",
                    "family": family,
                    "entity_slot": entity_slot,
                    "post_state": transition["post_state"],
                    "distance": distance,
                    "prediction_token_index": prediction_index,
                    "metadata": _metadata(pair, len(events)),
                }
                record["stable_hash"] = _record_hash(record)
                records.append(record)
    return records


def _assert_abstract_label(label: str, pair: dict[str, Any], tokenizer: Tokenizer) -> None:
    if not label or " " in label:
        raise StateTransitionValidationError(f"Non-abstract class label: {label!r}")
    label_ids = tokenizer.encode(label).ids
    for world in pair["worlds"].values():
        streams = [world["context_token_ids"]]
        streams.extend(candidate["token_ids"] for candidate in pair["candidates"].values())
        if label in world["context"] or any(_contains_subsequence(stream, label_ids) for stream in streams):
            raise StateTransitionValidationError(f"Abstract label leaked into a source token stream: {label}")


def validate_annotations(pair: dict[str, Any], annotations: Sequence[dict[str, Any]],
                         tokenizer: Tokenizer) -> None:
    """Validate hashes, evidence, state chronology, carries, and abstract labels."""
    if pair.get("stable_hash") != _record_hash(pair):
        raise StateTransitionValidationError("Source stable hash mismatch")
    for record in annotations:
        if record.get("stable_hash") != _record_hash(record):
            raise StateTransitionValidationError("Annotation stable hash mismatch")
        if record.get("source_stable_hash") != pair["stable_hash"]:
            raise StateTransitionValidationError("Annotation source stable hash mismatch")
        if record.get("schema_version") != SCHEMA_VERSION or record.get("generator_version") != GENERATOR_VERSION:
            raise StateTransitionValidationError("Annotation version mismatch")
    expected = annotate_pair(pair, tokenizer)
    if list(annotations) != expected:
        raise StateTransitionValidationError(
            "Annotations differ from source evidence, latest-state chronology, or carry boundaries"
        )
    for record in annotations:
        _assert_abstract_label(record["post_state"], pair, tokenizer)
        _assert_abstract_label(record["family"], pair, tokenizer)
        _assert_abstract_label(record["entity_slot"], pair, tokenizer)
        if record["kind"] == "transition":
            _assert_abstract_label(record["transition"], pair, tokenizer)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def _verify_sources(source_dir: Path, tokenizer_path: Path,
                    expected_manifest_hash: str | None, expected_tokenizer_hash: str | None) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = source_dir / "manifest.json"
    if expected_manifest_hash and sha256_file(manifest_path) != expected_manifest_hash:
        raise StateTransitionValidationError("Locked source manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer_hash = sha256_file(tokenizer_path)
    locked_tokenizer_hash = expected_tokenizer_hash or manifest.get("tokenizer_sha256")
    if tokenizer_hash != locked_tokenizer_hash:
        raise StateTransitionValidationError("Locked tokenizer hash changed")
    source_hashes = {}
    for split in SOURCE_SPLITS:
        name = f"{split}.jsonl"
        path = source_dir / name
        actual = sha256_file(path)
        if actual != manifest.get("files", {}).get(name, {}).get("sha256"):
            raise StateTransitionValidationError(f"Locked source hash changed: {name}")
        source_hashes[name] = actual
    source_hashes["manifest.json"] = sha256_file(manifest_path)
    return manifest, source_hashes


def _subset_digest(records: Sequence[dict[str, Any]], kind: str) -> tuple[str, int]:
    selected = [record for record in records if record["kind"] == kind]
    return sha256_bytes(_canonical_jsonl_bytes(selected)), len(selected)


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def build_sidecars(source_dir: Path, tokenizer_path: Path, output_dir: Path, *,
                   expected_source_manifest_sha256: str | None = None,
                   expected_tokenizer_sha256: str | None = None) -> dict[str, Any]:
    """Verify sealed inputs and atomically create an immutable sidecar directory."""
    _, source_hashes = _verify_sources(
        source_dir, tokenizer_path, expected_source_manifest_sha256, expected_tokenizer_sha256
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    split_manifest: dict[str, Any] = {}
    payloads: dict[str, bytes] = {}
    all_records: list[dict[str, Any]] = []
    classes: dict[str, set[str]] = {}
    for source_split in SOURCE_SPLITS:
        output_split = "generalization" if source_split == "generalization_holdout" else source_split
        records = []
        for pair in _read_jsonl(source_dir / f"{source_split}.jsonl"):
            pair_records = annotate_pair(pair, tokenizer)
            validate_annotations(pair, pair_records, tokenizer)
            records.extend(pair_records)
        all_records.extend(records)
        for record in records:
            classes.setdefault(record["family"], set()).add(record["post_state"])
        payload = _canonical_jsonl_bytes(records)
        filename = f"{output_split}.jsonl"
        payloads[filename] = payload
        transition_hash, transition_count = _subset_digest(records, "transition")
        carry_hash, carry_count = _subset_digest(records, "carry")
        split_manifest[output_split] = {
            "source_split": source_split,
            "file": filename,
            "canonical_sha256": sha256_bytes(payload),
            "record_count": len(records),
            "transitions": {"canonical_sha256": transition_hash, "count": transition_count},
            "carries": {"canonical_sha256": carry_hash, "count": carry_count},
        }
    transition_hash, transition_count = _subset_digest(all_records, "transition")
    carry_hash, carry_count = _subset_digest(all_records, "carry")
    manifest = {
        "dataset": "FictionPulper State-Transition-v1",
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "carry_offsets": list(CARRY_OFFSETS),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "source_hashes": source_hashes,
        "state_classes": {family: sorted(values) for family, values in sorted(classes.items())},
        "transitions": {"canonical_sha256": transition_hash, "count": transition_count},
        "carries": {"canonical_sha256": carry_hash, "count": carry_count},
        "total_annotations": transition_count + carry_count,
        "splits": split_manifest,
    }
    payloads["manifest.json"] = (json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=True
    ) + "\n").encode("utf-8")

    if output_dir.exists():
        existing = {path.name for path in output_dir.iterdir() if path.is_file()}
        if existing != set(payloads) or any((output_dir / name).read_bytes() != payload for name, payload in payloads.items()):
            raise FileExistsError(f"Refusing non-identical overwrite: {output_dir}")
        return manifest
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for name, payload in payloads.items():
            _write_bytes(staging / name, payload)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/counterfactual_narrative_v1"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/tokenizer.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/state_transition_v1"))
    parser.add_argument("--expected-source-manifest-sha256")
    parser.add_argument("--expected-tokenizer-sha256")
    args = parser.parse_args()
    manifest = build_sidecars(
        args.source_dir,
        args.tokenizer,
        args.output_dir,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        expected_tokenizer_sha256=args.expected_tokenizer_sha256,
    )
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
