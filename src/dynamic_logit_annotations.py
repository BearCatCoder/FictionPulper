"""Deterministic Dynamic-logit-v1 decisions from sealed continuity artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from tokenizers import Tokenizer

from src.continuity_curriculum.common import canonical_json, sha256_bytes, sha256_file


SCHEMA_VERSION = 1
GENERATOR_VERSION = "dynamic-logit-v1"
SOURCE_SPLITS = ("train", "validation", "test", "generalization_holdout")
CATEGORY_PRECEDENCE = ("CURRENT", "PREVIOUS", "INITIAL", "OLDER", "NEVER_VALID")
MAX_CONTEXT_TOKENS = 1024


class DynamicLogitValidationError(ValueError):
    """Raised when an annotation cannot be proved from both sealed sources."""


def _hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _record_hash(record: dict[str, Any]) -> str:
    return _hash({key: value for key, value in record.items() if key != "stable_hash"})


def _canonical_jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return b"".join((canonical_json(record) + "\n").encode("utf-8") for record in records)


def categorize_candidate(value: str, history: Sequence[str]) -> tuple[str, list[str]]:
    """Classify a value against observed state history with fixed precedence."""
    if not history:
        raise DynamicLogitValidationError("A decision requires non-empty state history")
    applicable: list[str] = []
    if value == history[-1]:
        applicable.append("CURRENT")
    if len(history) >= 2 and value == history[-2]:
        applicable.append("PREVIOUS")
    if value == history[0]:
        applicable.append("INITIAL")
    if len(history) >= 3 and value in history[:-2]:
        applicable.append("OLDER")
    if value not in history:
        applicable.append("NEVER_VALID")
    ordered = [category for category in CATEGORY_PRECEDENCE if category in applicable]
    return ordered[0], ordered


def _special_prefix(pair: dict[str, Any], tokenizer: Tokenizer) -> list[int]:
    tokens = ["<|story|>"]
    control = pair["metadata"].get("genre_control_token")
    if control:
        tokens.append(control)
    tokens.append("<|bos|>")
    ids = [tokenizer.token_to_id(token) for token in tokens]
    if any(token_id is None for token_id in ids):
        raise DynamicLogitValidationError("Tokenizer lacks a required source control token")
    return [int(token_id) for token_id in ids]


def _eos_id(tokenizer: Tokenizer) -> int:
    token_id = tokenizer.token_to_id("<|eos|>")
    if token_id is None:
        raise DynamicLogitValidationError("Tokenizer lacks <|eos|>")
    return int(token_id)


def _candidate_values(pair: dict[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    for world in pair["worlds"].values():
        name = str(world["correct_candidate"])
        value = str(world["final_value"])
        if name in values and values[name] != value:
            raise DynamicLogitValidationError("Candidate has inconsistent sealed state values")
        values[name] = value
    if set(values) != {"X", "Y"}:
        raise DynamicLogitValidationError("Sealed worlds do not prove both candidate values")
    return values


def _validate_sources(pair: dict[str, Any], transitions: Sequence[dict[str, Any]],
                      tokenizer: Tokenizer) -> dict[str, list[dict[str, Any]]]:
    if pair.get("stable_hash") != _record_hash(pair):
        raise DynamicLogitValidationError("Counterfactual source stable hash mismatch")
    for candidate in pair.get("candidates", {}).values():
        if tokenizer.encode(candidate["text"]).ids != candidate["token_ids"]:
            raise DynamicLogitValidationError("Candidate tokenization mismatch")
    by_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for transition in transitions:
        if transition.get("kind") != "transition":
            continue
        if transition.get("stable_hash") != _record_hash(transition):
            raise DynamicLogitValidationError("Transition stable hash mismatch")
        if transition.get("source_stable_hash") != pair["stable_hash"]:
            raise DynamicLogitValidationError("Transition source hash mismatch")
        if transition.get("pair") != pair["pair_id"]:
            raise DynamicLogitValidationError("Transition pair mismatch")
        by_world[str(transition["world"])].append(transition)
    if set(by_world) != {"A", "B"}:
        raise DynamicLogitValidationError("Transition sidecars must cover both worlds")
    for world_name, world_transitions in by_world.items():
        world_transitions.sort(key=lambda item: int(item["event"]))
        world = pair["worlds"][world_name]
        context = world["context"]
        context_ids = world["context_token_ids"]
        events = world["state_events"]
        if tokenizer.encode(context).ids != context_ids:
            raise DynamicLogitValidationError("Source context tokenization mismatch")
        if len(context_ids) + len(_special_prefix(pair, tokenizer)) + 1 > MAX_CONTEXT_TOKENS:
            raise DynamicLogitValidationError("Source context exceeds 1024 tokens")
        if [item["event"] for item in world_transitions] != list(range(len(events))):
            raise DynamicLogitValidationError("Transition depth is not contiguous")
        for depth, (event, transition) in enumerate(zip(events, world_transitions, strict=True), 1):
            span = transition["event_span"]
            start, end = int(span["token_start"]), int(span["token_end"])
            byte_start, byte_end = int(span["byte_start"]), int(span["byte_end"])
            evidence = context.encode("utf-8")[byte_start:byte_end].decode("utf-8")
            if (
                int(event["sequence"]) != depth - 1
                or not event["evidence"]
                or transition["event_evidence"] != event["evidence"]
                or evidence != event["evidence"]
                or context_ids[start:end] != span["token_ids"]
                or int(transition["prediction_token_index"]) != end
            ):
                raise DynamicLogitValidationError("Transition span or depth mismatch")
            prefix = context.encode("utf-8")[:byte_end].decode("utf-8")
            if tokenizer.encode(prefix).ids != context_ids[:end]:
                raise DynamicLogitValidationError("Transition does not end on a tokenizer boundary")
    return by_world


def annotate_pair(pair: dict[str, Any], transitions: Sequence[dict[str, Any]],
                  tokenizer: Tokenizer) -> list[dict[str, Any]]:
    """Create candidate decisions without changing source prose or candidate surfaces."""
    by_world = _validate_sources(pair, transitions, tokenizer)
    candidate_values = _candidate_values(pair)
    prefix_specials = _special_prefix(pair, tokenizer)
    eos_id = _eos_id(tokenizer)
    records: list[dict[str, Any]] = []
    for world_name in ("A", "B"):
        world = pair["worlds"][world_name]
        context = world["context"]
        context_ids = list(world["context_token_ids"])
        world_transitions = by_world[world_name]
        boundaries = [
            ("transition", depth, int(item["event_span"]["token_end"]),
             int(item["event_span"]["byte_end"]))
            for depth, item in enumerate(world_transitions, 1)
        ]
        boundaries.append(("final", len(world_transitions), len(context_ids), len(context.encode("utf-8"))))
        history = [str(event["value"]) for event in world["state_events"]]
        for kind, depth, context_end, byte_end in boundaries:
            observed = history[:depth]
            latest_end = int(world_transitions[depth - 1]["event_span"]["token_end"])
            context_prefix = context.encode("utf-8")[:byte_end].decode("utf-8")
            for candidate_name in ("X", "Y"):
                candidate = pair["candidates"][candidate_name]
                candidate_ids = list(candidate["token_ids"])
                if tokenizer.encode(context_prefix + candidate["text"]).ids != context_ids[:context_end] + candidate_ids:
                    raise DynamicLogitValidationError("Ambiguous context/candidate tokenizer boundary")
                decision_start = len(prefix_specials) + context_end
                decision_end = decision_start + len(candidate_ids)
                input_ids = prefix_specials + context_ids[:context_end] + candidate_ids + [eos_id]
                if len(input_ids) > MAX_CONTEXT_TOKENS:
                    raise DynamicLogitValidationError("Dynamic decision exceeds 1024 tokens")
                value = candidate_values[candidate_name]
                category, applicable = categorize_candidate(value, observed)
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "generator_version": GENERATOR_VERSION,
                    "source_stable_hash": pair["stable_hash"],
                    "transition_stable_hash": world_transitions[depth - 1]["stable_hash"],
                    "split": pair["split"],
                    "pair": pair["pair_id"],
                    "world": world_name,
                    "boundary": kind,
                    "candidate": candidate_name,
                    "candidate_text": candidate["text"],
                    "token_ids": candidate_ids,
                    "category": category,
                    "all_applicable_categories": applicable,
                    "state_value": value,
                    "input_token_ids": input_ids,
                    "decision_start": decision_start,
                    "decision_end": decision_end,
                    "decision_offsets": {
                        "context_token_end": context_end,
                        "candidate_token_start": decision_start,
                        "candidate_token_end": decision_end,
                    },
                    "depth": depth,
                    "family": world_transitions[depth - 1]["family"],
                    "distance": context_end - latest_end,
                    "metadata": {
                        "difficulty": pair["metadata"]["difficulty"],
                        "genre": pair["metadata"]["genre"],
                        "distance_bucket": pair["metadata"]["distance_bucket"],
                        "transition_count": len(world_transitions),
                    },
                }
                record["stable_hash"] = _record_hash(record)
                records.append(record)
    return records


def validate_annotations(pair: dict[str, Any], transitions: Sequence[dict[str, Any]],
                         annotations: Sequence[dict[str, Any]], tokenizer: Tokenizer) -> None:
    """Validate all hashes, taxonomy, semantics, boundaries, spans, and lengths."""
    for record in annotations:
        if record.get("stable_hash") != _record_hash(record):
            raise DynamicLogitValidationError("Annotation stable hash mismatch")
        start, end = int(record["decision_start"]), int(record["decision_end"])
        if end - start != len(record["token_ids"]):
            raise DynamicLogitValidationError("Decision span length mismatch")
        if record["input_token_ids"][start:end] != record["token_ids"]:
            raise DynamicLogitValidationError("Decision token span mismatch")
        if len(record["input_token_ids"]) > MAX_CONTEXT_TOKENS:
            raise DynamicLogitValidationError("Dynamic decision exceeds 1024 tokens")
    if list(annotations) != annotate_pair(pair, transitions, tokenizer):
        raise DynamicLogitValidationError("Annotations differ from sealed source decisions")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _verify_manifests(source_dir: Path, transition_dir: Path, tokenizer_path: Path,
                      expected_source_manifest_sha256: str | None,
                      expected_transition_manifest_sha256: str | None,
                      expected_tokenizer_sha256: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = source_dir / "manifest.json"
    transition_path = transition_dir / "manifest.json"
    if expected_source_manifest_sha256 and sha256_file(source_path) != expected_source_manifest_sha256:
        raise DynamicLogitValidationError("Locked source manifest hash changed")
    if expected_transition_manifest_sha256 and sha256_file(transition_path) != expected_transition_manifest_sha256:
        raise DynamicLogitValidationError("Locked transition manifest hash changed")
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    transition_manifest = json.loads(transition_path.read_text(encoding="utf-8"))
    tokenizer_hash = sha256_file(tokenizer_path)
    if tokenizer_hash != (expected_tokenizer_sha256 or source_manifest.get("tokenizer_sha256")):
        raise DynamicLogitValidationError("Locked tokenizer hash changed")
    if transition_manifest.get("tokenizer_sha256") != tokenizer_hash:
        raise DynamicLogitValidationError("Transition tokenizer hash changed")
    if transition_manifest.get("source_hashes", {}).get("manifest.json") != sha256_file(source_path):
        raise DynamicLogitValidationError("Transition source manifest hash changed")
    return source_manifest, transition_manifest


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def build_annotations(source_dir: Path, transition_dir: Path, tokenizer_path: Path,
                      output_dir: Path, *, expected_source_manifest_sha256: str | None = None,
                      expected_transition_manifest_sha256: str | None = None,
                      expected_tokenizer_sha256: str | None = None) -> dict[str, Any]:
    """Verify sealed inputs and atomically create a new immutable annotation set."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {output_dir}")
    source_manifest, transition_manifest = _verify_manifests(
        source_dir, transition_dir, tokenizer_path, expected_source_manifest_sha256,
        expected_transition_manifest_sha256, expected_tokenizer_sha256,
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    payloads: dict[str, bytes] = {}
    splits: dict[str, Any] = {}
    total = 0
    for source_split in SOURCE_SPLITS:
        output_split = "generalization" if source_split == "generalization_holdout" else source_split
        source_name = f"{source_split}.jsonl"
        transition_name = f"{output_split}.jsonl"
        source_path = source_dir / source_name
        transition_path = transition_dir / transition_name
        source_hash = sha256_file(source_path)
        transition_bytes = transition_path.read_bytes()
        transition_hash = sha256_bytes(transition_bytes)
        if source_hash != source_manifest.get("files", {}).get(source_name, {}).get("sha256"):
            raise DynamicLogitValidationError(f"Locked source hash changed: {source_name}")
        if source_hash != transition_manifest.get("source_hashes", {}).get(source_name):
            raise DynamicLogitValidationError(f"Transition source hash changed: {source_name}")
        if transition_hash != transition_manifest.get("splits", {}).get(output_split, {}).get("canonical_sha256"):
            raise DynamicLogitValidationError(f"Locked transition hash changed: {transition_name}")
        indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in _read_jsonl(transition_path):
            if item.get("kind") == "transition":
                indexed[str(item.get("pair"))].append(item)
        records: list[dict[str, Any]] = []
        for pair in _read_jsonl(source_path):
            pair_records = annotate_pair(pair, indexed.get(str(pair.get("pair_id")), []), tokenizer)
            validate_annotations(pair, indexed[str(pair["pair_id"])], pair_records, tokenizer)
            records.extend(pair_records)
        payload = _canonical_jsonl_bytes(records)
        payloads[f"{output_split}.jsonl"] = payload
        splits[output_split] = {
            "source_split": source_split,
            "file": f"{output_split}.jsonl",
            "record_count": len(records),
            "canonical_sha256": sha256_bytes(payload),
        }
        total += len(records)
    manifest = {
        "dataset": "FictionPulper Dynamic-logit-v1",
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "category_precedence": list(CATEGORY_PRECEDENCE),
        "maximum_context_tokens": MAX_CONTEXT_TOKENS,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "source_manifest_sha256": sha256_file(source_dir / "manifest.json"),
        "transition_manifest_sha256": sha256_file(transition_dir / "manifest.json"),
        "total_annotations": total,
        "splits": splits,
    }
    payloads["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
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


build_dataset = build_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/counterfactual_narrative_v1"))
    parser.add_argument("--transition-dir", type=Path, default=Path("data/state_transition_v1"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/tokenizer.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/dynamic_logit_v1"))
    parser.add_argument("--expected-source-manifest-sha256")
    parser.add_argument("--expected-transition-manifest-sha256")
    parser.add_argument("--expected-tokenizer-sha256")
    args = parser.parse_args()
    manifest = build_annotations(
        args.source_dir, args.transition_dir, args.tokenizer, args.output_dir,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        expected_transition_manifest_sha256=args.expected_transition_manifest_sha256,
        expected_tokenizer_sha256=args.expected_tokenizer_sha256,
    )
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
