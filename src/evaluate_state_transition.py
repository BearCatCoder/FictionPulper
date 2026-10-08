"""Evaluate State-Transition-v1 auxiliary heads on sealed held-out sidecars."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.counterfactual_narrative.evaluator import load_pairs
from src.model import FictionPulperLM, model_config_from_dict
from src.state_transition_annotations import validate_annotations
from src.train_state_transition import (
    CANONICAL_CLASS_MAPS,
    build_state_heads,
    head_config,
    state_examples_for_pairs,
)
from src.train_tokenizer import sha256_file


SPLIT_SOURCE_NAMES = {"test": "test", "generalization": "generalization_holdout"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _subset_hash(records: Iterable[Mapping[str, Any]]) -> str:
    from src.continuity_curriculum.common import canonical_json, sha256_bytes

    payload = b"".join((canonical_json(dict(record)) + "\n").encode("utf-8") for record in records)
    return sha256_bytes(payload)


def load_verified_split(
    *, state_directory: Path, source_directory: Path, split: str,
    expected_state_manifest_sha256: str, expected_state_file_sha256: str,
    expected_source_manifest_sha256: str, expected_source_file_sha256: str,
    tokenizer: Tokenizer, expected_tokenizer_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Verify every manifest, file, record hash, and sidecar/source relationship."""
    _require(split in SPLIT_SOURCE_NAMES, f"Unsupported auxiliary split: {split}")
    state_manifest_path = state_directory / "manifest.json"
    source_manifest_path = source_directory / "manifest.json"
    _require(sha256_file(state_manifest_path) == expected_state_manifest_sha256, "State manifest hash changed")
    _require(sha256_file(source_manifest_path) == expected_source_manifest_sha256, "Source manifest hash changed")
    state_manifest = json.loads(state_manifest_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_split = SPLIT_SOURCE_NAMES[split]
    state_spec = state_manifest.get("splits", {}).get(split, {})
    state_path = state_directory / str(state_spec.get("file"))
    source_path = source_directory / f"{source_split}.jsonl"
    _require(sha256_file(state_path) == expected_state_file_sha256 == state_spec.get("canonical_sha256"), "State split hash changed")
    source_file_spec = source_manifest.get("files", {}).get(source_path.name, {})
    _require(sha256_file(source_path) == expected_source_file_sha256 == source_file_spec.get("sha256"), "Source split hash changed")
    _require(state_manifest.get("tokenizer_sha256") == expected_tokenizer_sha256, "State tokenizer lock changed")
    _require(state_manifest.get("source_hashes", {}).get("manifest.json") == expected_source_manifest_sha256, "State/source manifest relation changed")
    _require(state_manifest.get("source_hashes", {}).get(source_path.name) == expected_source_file_sha256, "State/source split relation changed")
    _require(state_spec.get("source_split") == source_split, "State source split identity changed")

    pairs = load_pairs(source_directory, source_split)
    by_id = {str(pair["pair_id"]): pair for pair in pairs}
    records = _read_jsonl(state_path)
    _require(len(records) == int(state_spec.get("record_count", -1)), "State record count changed")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        _require(record.get("split") == source_split, "State record split changed")
        pair_id = str(record.get("pair"))
        _require(pair_id in by_id, "State sidecar references an absent source pair")
        grouped[pair_id].append(record)
    _require(set(grouped) == set(by_id), "State sidecar/source pair coverage changed")
    for pair_id, pair_records in grouped.items():
        validate_annotations(by_id[pair_id], pair_records, tokenizer)
    for kind, key in (("transition", "transitions"), ("carry", "carries")):
        selected = [record for record in records if record["kind"] == kind]
        _require(len(selected) == int(state_spec[key]["count"]), f"State {kind} count changed")
        _require(_subset_hash(selected) == state_spec[key]["canonical_sha256"], f"State {kind} subset hash changed")
    return pairs, dict(grouped), {
        "state_manifest_sha256": expected_state_manifest_sha256,
        "state_file_sha256": expected_state_file_sha256,
        "source_manifest_sha256": expected_source_manifest_sha256,
        "source_file_sha256": expected_source_file_sha256,
        "source_split": source_split,
        "pair_count": len(pairs),
        "annotation_count": len(records),
    }


def load_model_and_heads(checkpoint: Mapping[str, Any], device: torch.device) -> tuple[FictionPulperLM, torch.nn.ModuleDict]:
    _require(checkpoint.get("format") == "fictionpulper-state-transition-v1", "Wrong checkpoint format")
    _require(checkpoint.get("head_config") == head_config(), "Auxiliary head contract changed")
    model = FictionPulperLM(model_config_from_dict(checkpoint["model_config"])).to(device)
    heads = build_state_heads(model.config.hidden_size).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    heads.load_state_dict(checkpoint["state_heads"], strict=True)
    model.eval()
    heads.eval()
    for parameter in [*model.parameters(), *heads.parameters()]:
        parameter.requires_grad_(False)
        parameter.grad = None
    return model, heads


@torch.inference_mode()
def evaluate_examples(
    model: FictionPulperLM, heads: torch.nn.ModuleDict,
    examples: Sequence[Mapping[str, Any]], *, device: torch.device,
    batch_size: int, use_bf16: bool,
) -> list[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows: list[dict[str, Any]] = []
    for offset in range(0, len(examples), batch_size):
        batch = examples[offset:offset + batch_size]
        maximum = max(len(item["token_ids"]) for item in batch)
        tokens = torch.zeros((len(batch), maximum), dtype=torch.long, device=device)
        for row_index, example in enumerate(batch):
            values = torch.tensor(example["token_ids"], dtype=torch.long, device=device)
            tokens[row_index, :len(values)] = values
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cpu", enabled=False)
        with autocast:
            _, loss, hidden_states = model(tokens, labels=None, return_hidden_states=True)
        _require(loss is None, "Auxiliary evaluation unexpectedly supplied LM labels")
        hidden = hidden_states[-1]
        for row_index, example in enumerate(batch):
            for record in example["annotations"]:
                family = str(record["family"])
                target_index = int(record["class_index"])
                logits = heads[family](hidden[row_index, int(record["hidden_position"])].float())
                ce = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_index], device=device))
                rows.append({
                    "pair": str(record["pair"]), "world": str(record["world"]),
                    "entity_slot": str(record["entity_slot"]), "event": int(record["event"]),
                    "kind": str(record["kind"]), "family": family,
                    "label": str(record["post_state"]), "class_index": target_index,
                    "predicted_class_index": int(logits.argmax().item()),
                    "ce": float(ce.item()), "correct": int(logits.argmax().item() == target_index),
                    "source_distance_bucket": str(record["metadata"]["distance_bucket"]),
                    "carry_distance": int(record["distance"]) if record["kind"] == "carry" else None,
                    "difficulty": str(record["metadata"]["difficulty"]),
                    "genre": str(record["metadata"]["genre"]),
                    "transition_count": int(record["metadata"]["transition_count"]),
                    "transition_depth": int(record["event"]) + 1,
                })
    _require(not any(parameter.requires_grad or parameter.grad is not None for parameter in [*model.parameters(), *heads.parameters()]), "Auxiliary evaluation changed frozen parameter state")
    return rows


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "ce": None, "accuracy": None}
    return {
        "count": len(rows),
        "ce": sum(float(row["ce"]) for row in rows) / len(rows),
        "accuracy": sum(int(row["correct"]) for row in rows) / len(rows),
    }


def _groups(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        values[str(row[field])].append(row)
    return {name: _summary(group) for name, group in sorted(values.items())}


def aggregate_auxiliary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate CE/accuracy, final-state, depth, and label-only baselines."""
    transitions = [row for row in rows if row["kind"] == "transition"]
    carries = [row for row in rows if row["kind"] == "carry"]
    output: dict[str, Any] = {
        "overall": _summary(rows), "transition": _summary(transitions), "carry": _summary(carries),
    }
    for field in ("family", "source_distance_bucket", "carry_distance", "difficulty", "genre"):
        selected = carries if field == "carry_distance" else rows
        output[f"by_{field}"] = _groups(selected, field)
        if field != "carry_distance":
            output[f"transition_by_{field}"] = _groups(transitions, field)
            output[f"carry_by_{field}"] = _groups(carries, field)

    by_entity: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_entity[(str(row["pair"]), str(row["world"]), str(row["entity_slot"]))].append(row)
    final_transitions: list[Mapping[str, Any]] = []
    final_carries: list[Mapping[str, Any]] = []
    for entity_rows in by_entity.values():
        entity_transitions = [row for row in entity_rows if row["kind"] == "transition"]
        _require(bool(entity_transitions), "Entity has no transition annotation")
        final_event = max(int(row["event"]) for row in entity_transitions)
        selected = [row for row in entity_transitions if int(row["event"]) == final_event]
        _require(len(selected) == 1, "Entity has a non-unique final transition")
        final_transitions.extend(selected)
        final_carries.extend(row for row in entity_rows if row["kind"] == "carry" and int(row["event"]) == final_event)
    output["latest_current_state"] = {
        "entity_count": len(by_entity),
        "final_transition": _summary(final_transitions),
        "valid_final_carries": _summary(final_carries),
        "combined": _summary([*final_transitions, *final_carries]),
        "final_transition_by_family": _groups(final_transitions, "family"),
        "valid_final_carries_by_family": _groups(final_carries, "family"),
        "valid_final_carries_by_distance": _groups(final_carries, "carry_distance"),
    }

    depths: dict[str, Any] = {}
    for label, predicate, availability in (
        ("1", lambda value: value == 1, lambda maximum: maximum >= 1),
        ("2", lambda value: value == 2, lambda maximum: maximum >= 2),
        ("3", lambda value: value == 3, lambda maximum: maximum >= 3),
        ("4+", lambda value: value >= 4, lambda maximum: maximum >= 4),
    ):
        selected = [row for row in transitions if predicate(int(row["transition_depth"]))]
        available = sum(availability(max(int(row["transition_depth"]) for row in entity_rows if row["kind"] == "transition")) for entity_rows in by_entity.values())
        depths[label] = {
            **_summary(selected), "available_entity_sequences": available,
            "unavailable_entity_sequences": len(by_entity) - available,
        }
    output["transition_depth"] = depths

    baselines = {}
    for family, family_rows in sorted(_partition(rows, "family").items()):
        counts = Counter(str(row["label"]) for row in family_rows)
        class_count = len(CANONICAL_CLASS_MAPS[family])
        baselines[family] = {
            "count": len(family_rows), "class_count": class_count,
            "chance_accuracy": 1.0 / class_count,
            "majority_accuracy": max(counts.values()) / len(family_rows),
            "majority_labels": sorted(label for label, count in counts.items() if count == max(counts.values())),
            "label_counts": dict(sorted(counts.items())),
        }
    output["baselines_by_family"] = baselines
    return output


def _partition(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row[field])].append(row)
    return dict(result)


def evaluate_split(
    *, checkpoint: Mapping[str, Any], tokenizer: Tokenizer, state_directory: Path,
    source_directory: Path, split: str, locks: Mapping[str, str], device: torch.device,
    batch_size: int, use_bf16: bool,
) -> dict[str, Any]:
    pairs, annotations, provenance = load_verified_split(
        state_directory=state_directory, source_directory=source_directory, split=split,
        expected_state_manifest_sha256=locks["state_manifest_sha256"],
        expected_state_file_sha256=locks[f"state_{split}_sha256"],
        expected_source_manifest_sha256=locks["source_manifest_sha256"],
        expected_source_file_sha256=locks[f"source_{SPLIT_SOURCE_NAMES[split]}_sha256"],
        tokenizer=tokenizer, expected_tokenizer_sha256=locks["tokenizer_sha256"],
    )
    model, heads = load_model_and_heads(checkpoint, device)
    examples = state_examples_for_pairs(pairs, annotations, tokenizer)
    rows = evaluate_examples(model, heads, examples, device=device, batch_size=batch_size, use_bf16=use_bf16)
    return {"provenance": provenance, "metrics": aggregate_auxiliary(rows), "rows": rows}

