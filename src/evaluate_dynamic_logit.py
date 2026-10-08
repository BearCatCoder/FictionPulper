"""Held-out Dynamic-logit-v1 evaluation using only tied LM logits.

Importing this module performs no artifact access.  The command line entry point
verifies all sealed inputs before parsing held-out records or loading a model.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.continuity_curriculum.common import canonical_json, sha256_bytes, sha256_file, write_json_atomic
from src.dynamic_logit_annotations import (
    CATEGORY_PRECEDENCE,
    MAX_CONTEXT_TOKENS,
    categorize_candidate,
    validate_annotations,
)
from src.model import FictionPulperLM, model_config_from_dict


EXPECTED_PARAMETER_COUNT = 15_047_040
SPLIT_SOURCE_NAMES = {"test": "test", "generalization": "generalization_holdout"}
DEPTH_BUCKETS = ("1", "2", "3", "4+")
SEALED = {
    "dynamic_manifest_sha256": "2ad62e6831b2e2714e3a06b8b1e2cd92f57766c3521c7a786cd047617f12f8c3",
    "transition_manifest_sha256": "36372208b3b4d35523668d4292c2884eab8ae68a239bddba4a131c3e32496593",
    "source_manifest_sha256": "41bfec4201e36a28fa5f4bea265b326ab7a0dbc67ba95c86533a1bdf191eede8",
    "tokenizer_sha256": "14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012",
    "checkpoint_sha256": "1d2ac1947afd10436729b007ae5d1fbeb9ef0815fc507ee15ab1a9e8e5c43575",
    "dynamic_test_sha256": "5b9a618e0a2ce92e628402777f45a41c1737cbed3b6e1ac899d2ea35c34d41e8",
    "dynamic_generalization_sha256": "d1a2fb49cb81a60bba92106a334a7e24a39058b23e1192b220d2b7a23069c0c7",
    "transition_test_sha256": "624098cbd821e57bc5b71ecac4204ebf3dca80bc807cfec88d33343f5fdfa04a",
    "transition_generalization_sha256": "7f324a4ee426f395d30db73266a6bf44e1e1e7cce38bc1f15c5d335218f8e0b5",
    "source_test_sha256": "2651ff8ddd8449ddf2a42cc56c8f4ee2cee09e2e632bd85788f2f2f1f1093c92",
    "source_generalization_sha256": "2d60efd1f07731f839b44d9e30a4f767497b7ec7d6612468bcdd9a4c9a485505",
}


class DynamicEvaluationError(ValueError):
    """Raised when a sealed artifact or evaluation record is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DynamicEvaluationError(message)


def record_hash(record: Mapping[str, Any]) -> str:
    """Return the canonical stable hash used by the sealed datasets."""
    value = {key: item for key, item in record.items() if key != "stable_hash"}
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def verify_file(path: Path, expected_sha256: str, description: str) -> str:
    """Require a real file with exactly the externally locked digest."""
    _require(path.is_file(), f"Missing {description}: {path}")
    _require(len(expected_sha256) == 64, f"Malformed expected hash for {description}")
    observed = sha256_file(path)
    _require(observed == expected_sha256, f"Hash mismatch for {description}: {path}")
    return observed


def read_jsonl(path: Path, *, expected_split: str | None = None) -> list[dict[str, Any]]:
    """Read non-empty JSON objects and reject malformed or hash-invalid records."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise DynamicEvaluationError(f"Invalid JSON in {path} line {line_number}") from error
            _require(isinstance(record, dict), f"Non-object record in {path} line {line_number}")
            _require(record.get("stable_hash") == record_hash(record),
                     f"Stable hash mismatch in {path} line {line_number}")
            if expected_split is not None:
                _require(record.get("split") == expected_split,
                         f"Split mismatch in {path} line {line_number}")
            records.append(record)
    _require(bool(records), f"Artifact is empty: {path}")
    return records


def group_candidate_sets(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str, int], list[dict[str, Any]]]:
    """Group complete candidate decisions and validate taxonomy/span contracts."""
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for source in records:
        record = dict(source)
        required = {"pair", "world", "boundary", "depth", "candidate", "category",
                    "all_applicable_categories", "input_token_ids", "token_ids",
                    "decision_start", "decision_end", "family", "distance", "metadata"}
        _require(required <= set(record), "Malformed dynamic decision record")
        start, end = int(record["decision_start"]), int(record["decision_end"])
        _require(0 < start < end < len(record["input_token_ids"]), "Invalid decision span")
        _require(record["input_token_ids"][start:end] == record["token_ids"],
                 "Decision span does not contain candidate tokens")
        category = str(record["category"])
        aliases = list(record["all_applicable_categories"])
        _require(bool(category in CATEGORY_PRECEDENCE and aliases
                 and aliases == [item for item in CATEGORY_PRECEDENCE if item in aliases]
                 and aliases[0] == category), "Invalid taxonomy precedence or aliases")
        key = (str(record["pair"]), str(record["world"]), str(record["boundary"]), int(record["depth"]))
        grouped[key].append(record)
    for key, candidates in grouped.items():
        candidates.sort(key=lambda item: str(item["candidate"]))
        names = [str(item["candidate"]) for item in candidates]
        _require(len(candidates) >= 2 and len(names) == len(set(names)), f"Incomplete candidate set: {key}")
        _require(sum(item["category"] == "CURRENT" for item in candidates) == 1,
                 f"Candidate set lacks one CURRENT candidate: {key}")
    return dict(sorted(grouped.items()))


@torch.inference_mode()
def score_token_spans(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    pad_id: int,
    use_bf16: bool,
) -> list[float]:
    """Score exact decision spans by arithmetic mean token log probability."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if use_bf16 and device.type != "cuda":
        raise ValueError("BF16 evaluation requires CUDA")
    output: list[float] = []
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        maximum = max(len(item["input_token_ids"]) - 1 for item in batch)
        inputs = torch.full((len(batch), maximum), pad_id, dtype=torch.long, device=device)
        for row, item in enumerate(batch):
            values = item["input_token_ids"][:-1]
            inputs[row, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        context = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if use_bf16 else torch.autocast(device.type, enabled=False))
        with context:
            logits, loss = model(inputs, labels=None)
        _require(loss is None, "Scoring unexpectedly produced a training loss")
        for row, item in enumerate(batch):
            start, end = int(item["decision_start"]), int(item["decision_end"])
            _require(0 < start < end <= len(item["input_token_ids"]), "Invalid decision span")
            targets = torch.tensor(item["input_token_ids"][start:end], dtype=torch.long, device=device)
            token_logits = logits[row, start - 1:end - 1].float()
            score = F.log_softmax(token_logits, -1).gather(1, targets[:, None]).mean()
            output.append(float(score.item()))
    _require(not any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()),
             "Evaluation changed frozen-model state")
    return output


def decision_result(records: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> dict[str, Any]:
    """Create one taxonomy result; margins only use non-CURRENT candidates."""
    _require(len(records) == len(scores) and len(records) >= 2, "Candidate records/scores disagree")
    current_indices = [index for index, item in enumerate(records) if item["category"] == "CURRENT"]
    _require(len(current_indices) == 1, "Decision must contain exactly one CURRENT candidate")
    current = current_indices[0]
    selected = max(range(len(scores)), key=lambda index: (float(scores[index]), -index))
    current_score = float(scores[current])
    margins: dict[str, float | None] = {}
    for category in CATEGORY_PRECEDENCE[1:]:
        alternatives = [
            float(scores[index]) for index, item in enumerate(records)
            if index != current and category in item.get("all_applicable_categories", [item["category"]])
        ]
        margins[category] = current_score - max(alternatives) if alternatives else None
    first = records[0]
    maximum_depth = int(first.get("metadata", {}).get("transition_count", first["depth"]))
    return {
        "pair": str(first["pair"]), "world": str(first["world"]),
        "boundary": str(first["boundary"]), "depth": int(first["depth"]),
        "depth_bucket": depth_bucket(int(first["depth"])), "maximum_depth": maximum_depth,
        "family": str(first["family"]),
        "source_state_family": str(first.get("source_state_family", first["family"])),
        "distance_bucket": str(first["metadata"]["distance_bucket"]),
        "token_distance": int(first["distance"]),
        "selected_candidate": str(records[selected]["candidate"]),
        "selected_category": str(records[selected]["category"]),
        "selected_aliases": list(records[selected].get("all_applicable_categories", [records[selected]["category"]])),
        "current_candidate": str(records[current]["candidate"]),
        "current_score": current_score, "best_noncurrent_score": max(
            float(score) for index, score in enumerate(scores) if index != current
        ),
        "current_correct": selected == current, "margins": margins,
    }


def depth_bucket(depth: int) -> str:
    _require(isinstance(depth, int) and not isinstance(depth, bool) and depth >= 1,
             "Depth must be a positive integer")
    return str(depth) if depth <= 3 else "4+"


def summarize_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report precedence counts/rates, alias counts/rates, and corrected margins."""
    count = len(rows)
    counts = {category: sum(row["selected_category"] == category for row in rows)
              for category in CATEGORY_PRECEDENCE}
    aliases = {category: sum(category in row["selected_aliases"] for row in rows)
               for category in CATEGORY_PRECEDENCE}
    margins = {}
    for category in CATEGORY_PRECEDENCE[1:]:
        values = [float(row["margins"][category]) for row in rows if row["margins"].get(category) is not None]
        margins[category] = {
            "count": len(values), "unavailable_count": count - len(values),
            "mean": mean(values) if values else None,
        }
    return {
        "count": count, "current_accuracy": sum(bool(row["current_correct"]) for row in rows) / count if count else None,
        "precedence_counts": counts,
        "precedence_rates": {key: value / count if count else None for key, value in counts.items()},
        "alias_counts": aliases,
        "alias_rates": {key: value / count if count else None for key, value in aliases.items()},
        "current_vs_noncurrent_margins": margins,
    }


def _group(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {name: summarize_results(values) for name, values in sorted(groups.items())}


def depth_availability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Distinguish unsupported depth strata from observed failures."""
    worlds: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["pair"]), str(row["world"]))
        worlds[key] = max(worlds.get(key, 0), int(row["maximum_depth"]))
    output = {}
    for bucket in DEPTH_BUCKETS:
        available = sum((depth >= int(bucket)) if bucket != "4+" else (depth >= 4)
                        for depth in worlds.values())
        selected = [row for row in rows if row["depth_bucket"] == bucket]
        output[bucket] = {
            "available_worlds": available, "unavailable_worlds": len(worlds) - available,
            "status": "available" if available else "unavailable", "metrics": summarize_results(selected),
        }
    return output


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "overall": summarize_results(rows),
        "by_state_family": _group(rows, "source_state_family"),
        "by_depth": depth_availability(rows),
        "by_distance_bucket": _group(rows, "distance_bucket"),
        "by_exact_token_distance": _group(rows, "token_distance"),
        "by_boundary": _group(rows, "boundary"),
    }


def _context_prefix(pair: Mapping[str, Any], record: Mapping[str, Any], tokenizer: Tokenizer) -> tuple[str, int, int]:
    """Return exact source prefix and latest evidence character span within it."""
    world = pair["worlds"][record["world"]]
    context = str(world["context"])
    events = world["state_events"]
    depth = int(record["depth"])
    _require(1 <= depth <= len(events), "Decision depth exceeds source history")
    positions: list[tuple[int, int]] = []
    cursor = 0
    for event in events:
        evidence = str(event["evidence"])
        start = context.find(evidence, cursor)
        _require(start >= 0, "State evidence is absent or out of sequence")
        positions.append((start, start + len(evidence)))
        cursor = start + len(evidence)
    if record["boundary"] == "transition":
        prefix_end = positions[depth - 1][1]
    else:
        _require(record["boundary"] == "final", "Unknown decision boundary")
        prefix_end = len(context)
    prefix = context[:prefix_end]
    context_end = int(record["decision_offsets"]["context_token_end"])
    _require(tokenizer.encode(prefix).ids == world["context_token_ids"][:context_end],
             "Source context prefix does not match the sealed decision")
    start, end = positions[depth - 1]
    _require(end <= prefix_end, "Latest evidence lies outside the decision context")
    return prefix, start, end


def rebuild_candidate_input(
    record: Mapping[str, Any], context: str, tokenizer: Tokenizer, *, control: str,
) -> dict[str, Any]:
    """Re-tokenize a changed context and preserve the exact sealed candidate span."""
    special_count = int(record["decision_start"]) - int(record["decision_offsets"]["context_token_end"])
    _require(special_count > 0, "Malformed special-token prefix")
    special_ids = list(record["input_token_ids"][:special_count])
    context_ids = tokenizer.encode(context).ids
    candidate_ids = list(record["token_ids"])
    combined = tokenizer.encode(context + str(record["candidate_text"])).ids
    _require(combined == context_ids + candidate_ids, "Changed context has an ambiguous candidate boundary")
    start = len(special_ids) + len(context_ids)
    rebuilt = dict(record)
    rebuilt.update({
        "input_token_ids": special_ids + combined + [record["input_token_ids"][-1]],
        "decision_start": start, "decision_end": start + len(candidate_ids), "control": control,
    })
    _require(len(rebuilt["input_token_ids"]) <= MAX_CONTEXT_TOKENS, "Control input exceeds context limit")
    return rebuilt


def build_fact_removal_set(
    records: Sequence[Mapping[str, Any]], pair: Mapping[str, Any], tokenizer: Tokenizer,
) -> list[dict[str, Any]]:
    """Remove exactly the latest observed transition evidence for each candidate."""
    output = []
    for record in records:
        prefix, start, end = _context_prefix(pair, record, tokenizer)
        evidence = pair["worlds"][record["world"]]["state_events"][int(record["depth"]) - 1]["evidence"]
        _require(prefix[start:end] == evidence, "Decisive evidence span changed")
        output.append(rebuild_candidate_input(record, prefix[:start] + prefix[end:], tokenizer,
                                              control="fact_removal"))
    return output


def build_context_swap_set(
    records: Sequence[Mapping[str, Any]], opposite: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Put each candidate after the matched sealed context from the opposite world."""
    by_candidate = {str(item["candidate"]): item for item in opposite}
    _require(set(by_candidate) == {str(item["candidate"]) for item in records},
             "Opposite worlds have different candidates")
    output = []
    for source in records:
        other = by_candidate[str(source["candidate"])]
        rebuilt = dict(other)
        rebuilt["control"] = "context_swap"
        rebuilt["source_world"] = str(source["world"])
        rebuilt["context_world"] = str(other["world"])
        output.append(rebuilt)
    return output


def margin_collapse_audit(full: Sequence[Mapping[str, Any]], removed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare signed current-vs-best-alternative margins before and after removal."""
    _require(len(full) == len(removed), "Full/removal result counts differ")
    values = []
    for normal, control in zip(full, removed, strict=True):
        before = float(normal["current_score"]) - float(normal["best_noncurrent_score"])
        after = float(control["current_score"]) - float(control["best_noncurrent_score"])
        values.append((before, after))
    return {
        "count": len(values),
        "mean_full_signed_margin": mean(item[0] for item in values) if values else None,
        "mean_removed_signed_margin": mean(item[1] for item in values) if values else None,
        "mean_absolute_margin_collapse": mean(abs(before) - abs(after) for before, after in values) if values else None,
        "collapse_rate": sum(abs(after) < abs(before) for before, after in values) / len(values) if values else None,
    }


def switch_audit(full: Sequence[Mapping[str, Any]], swapped: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit that opposite-world contexts switch candidate preference."""
    normal = {(row["pair"], row["world"], row["boundary"], row["depth"]): row for row in full}
    controls = {(row["pair"], row["world"], row["boundary"], row["depth"]): row for row in swapped}
    _require(set(normal) == set(controls), "Full/context-swap decisions do not align")
    observations = []
    for key, row in normal.items():
        opposite_world = "B" if key[1] == "A" else "A"
        opposite = normal.get((key[0], opposite_world, key[2], key[3]))
        _require(opposite is not None, "Context-swap decision lacks its opposite world")
        assert opposite is not None
        control = controls[key]
        observations.append({
            "preference_switched": control["selected_candidate"] != row["selected_candidate"],
            "matches_opposite_world": control["selected_candidate"] == opposite["selected_candidate"],
            "opposite_correct_candidates": row["current_candidate"] != opposite["current_candidate"],
        })
    count = len(observations)
    return {
        "count": count,
        "preference_switch_rate": sum(item["preference_switched"] for item in observations) / count if count else None,
        "matches_opposite_world_rate": sum(item["matches_opposite_world"] for item in observations) / count if count else None,
        "opposite_label_integrity": all(item["opposite_correct_candidates"] for item in observations),
    }


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DynamicEvaluationError(f"Malformed JSON artifact: {path}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def load_verified_split(
    *, split: str, dynamic_dir: Path, transition_dir: Path, source_dir: Path,
    tokenizer: Tokenizer, locks: Mapping[str, str],
) -> tuple[dict[tuple[str, str, str, int], list[dict[str, Any]]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Hash-verify and re-derive every held-out annotation from sealed sources."""
    _require(split in SPLIT_SOURCE_NAMES, f"Unsupported split: {split}")
    source_split = SPLIT_SOURCE_NAMES[split]
    dynamic_manifest_path = dynamic_dir / "manifest.json"
    transition_manifest_path = transition_dir / "manifest.json"
    source_manifest_path = source_dir / "manifest.json"
    verify_file(dynamic_manifest_path, locks["dynamic_manifest_sha256"], "dynamic manifest")
    verify_file(transition_manifest_path, locks["transition_manifest_sha256"], "transition manifest")
    verify_file(source_manifest_path, locks["source_manifest_sha256"], "counterfactual manifest")
    dynamic_manifest = _json_object(dynamic_manifest_path)
    transition_manifest = _json_object(transition_manifest_path)
    source_manifest = _json_object(source_manifest_path)
    _require(dynamic_manifest.get("dataset") == "FictionPulper Dynamic-logit-v1", "Wrong dynamic dataset")
    _require(dynamic_manifest.get("category_precedence") == list(CATEGORY_PRECEDENCE), "Taxonomy lock changed")
    _require(dynamic_manifest.get("tokenizer_sha256") == locks["tokenizer_sha256"], "Tokenizer lock changed")
    _require(dynamic_manifest.get("source_manifest_sha256") == locks["source_manifest_sha256"], "Source lock changed")
    _require(dynamic_manifest.get("transition_manifest_sha256") == locks["transition_manifest_sha256"],
             "Transition lock changed")

    dynamic_spec = dynamic_manifest.get("splits", {}).get(split, {})
    transition_spec = transition_manifest.get("splits", {}).get(split, {})
    dynamic_path = dynamic_dir / str(dynamic_spec.get("file"))
    transition_path = transition_dir / str(transition_spec.get("file"))
    source_path = source_dir / f"{source_split}.jsonl"
    dynamic_hash = locks[f"dynamic_{split}_sha256"]
    transition_hash = locks[f"transition_{split}_sha256"]
    source_hash = locks[f"source_{split}_sha256"]
    verify_file(dynamic_path, dynamic_hash, f"dynamic {split}")
    verify_file(transition_path, transition_hash, f"transition {split}")
    verify_file(source_path, source_hash, f"counterfactual {split}")
    _require(dynamic_spec.get("canonical_sha256") == dynamic_hash
             and dynamic_spec.get("source_split") == source_split, "Dynamic split manifest mismatch")
    _require(transition_spec.get("canonical_sha256") == transition_hash
             and transition_spec.get("source_split") == source_split, "Transition split manifest mismatch")
    _require(source_manifest.get("files", {}).get(source_path.name, {}).get("sha256") == source_hash,
             "Source split manifest mismatch")
    _require(transition_manifest.get("source_hashes", {}).get(source_path.name) == source_hash,
             "Transition/source relation changed")

    pairs = read_jsonl(source_path, expected_split=source_split)
    transitions = read_jsonl(transition_path, expected_split=source_split)
    annotations = read_jsonl(dynamic_path, expected_split=source_split)
    _require(len(annotations) == int(dynamic_spec.get("record_count", -1)), "Dynamic record count changed")
    _require(len(transitions) == int(transition_spec.get("record_count", -1)), "Transition record count changed")
    pair_index = {str(pair["pair_id"]): pair for pair in pairs}
    _require(len(pair_index) == len(pairs), "Duplicate source pair")
    transition_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    annotation_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in transitions:
        if item.get("kind") == "transition":
            transition_index[str(item.get("pair"))].append(item)
    for item in annotations:
        annotation_index[str(item.get("pair"))].append(item)
    _require(set(annotation_index) == set(pair_index), "Dynamic/source pair coverage changed")
    for pair_id, pair in pair_index.items():
        validate_annotations(pair, transition_index[pair_id], annotation_index[pair_id], tokenizer)
    grouped = group_candidate_sets(annotations)
    return grouped, pair_index, {
        "split": split, "source_split": source_split, "pair_count": len(pairs),
        "decision_count": len(grouped), "annotation_count": len(annotations),
        "hashes": {"dynamic": dynamic_hash, "transition": transition_hash, "source": source_hash},
    }


def load_model(checkpoint_path: Path, expected_sha256: str, device: torch.device) -> FictionPulperLM:
    """Load only the locked direct-LM checkpoint and reject auxiliary heads."""
    verify_file(checkpoint_path, expected_sha256, "Dynamic-logit-v1 checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, dict) and checkpoint.get("format") == "fictionpulper-dynamic-logit-v1",
             "Wrong checkpoint format")
    _require(checkpoint.get("auxiliary_heads") is False and "state_heads" not in checkpoint,
             "Dynamic checkpoint contains an auxiliary head")
    model = FictionPulperLM(model_config_from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    _require(model.trainable_parameter_count() == EXPECTED_PARAMETER_COUNT,
             f"Model has {model.trainable_parameter_count():,} parameters, expected {EXPECTED_PARAMETER_COUNT:,}")
    _require(model.lm_head.weight is model.embed_tokens.weight, "LM head is not tied to token embeddings")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _score_decisions(
    model: torch.nn.Module, decisions: Sequence[Sequence[Mapping[str, Any]]], *, device: torch.device,
    batch_size: int, pad_id: int,
) -> list[dict[str, Any]]:
    flat = [record for decision in decisions for record in decision]
    scores = score_token_spans(model, flat, device=device, batch_size=batch_size, pad_id=pad_id, use_bf16=True)
    rows, offset = [], 0
    for decision in decisions:
        rows.append(decision_result(decision, scores[offset:offset + len(decision)]))
        offset += len(decision)
    return rows


def evaluate_split(
    model: torch.nn.Module, grouped: Mapping[tuple[str, str, str, int], Sequence[Mapping[str, Any]]],
    pairs: Mapping[str, Mapping[str, Any]], tokenizer: Tokenizer, *, device: torch.device, batch_size: int,
) -> dict[str, Any]:
    keys = sorted(grouped)
    full_decisions = [list(grouped[key]) for key in keys]
    removed_decisions = [build_fact_removal_set(grouped[key], pairs[key[0]], tokenizer) for key in keys]
    swapped_decisions = []
    for key in keys:
        opposite = "B" if key[1] == "A" else "A"
        opposite_key = (key[0], opposite, key[2], key[3])
        _require(opposite_key in grouped, f"Missing opposite-world decision: {key}")
        swapped_decisions.append(build_context_swap_set(grouped[key], grouped[opposite_key]))
    pad_id = tokenizer.token_to_id("<|pad|>")
    _require(pad_id is not None, "Tokenizer lacks PAD")
    full = _score_decisions(model, full_decisions, device=device, batch_size=batch_size, pad_id=int(pad_id))
    removed = _score_decisions(model, removed_decisions, device=device, batch_size=batch_size, pad_id=int(pad_id))
    swapped_raw = _score_decisions(model, swapped_decisions, device=device, batch_size=batch_size, pad_id=int(pad_id))
    for rows in (full, removed, swapped_raw):
        for row in rows:
            row["source_state_family"] = str(
                pairs[str(row["pair"])]["abstract_counterfactual_variable"]["state_family"]
            )
    swapped = []
    for key, row in zip(keys, swapped_raw, strict=True):
        copied = dict(row)
        copied["world"] = key[1]
        swapped.append(copied)
    latest = [row for row in full if row["source_state_family"] == "latest_state_update"]
    return {
        "full_context": aggregate_results(full),
        "fact_removal": {"metrics": aggregate_results(removed), "margin_collapse": margin_collapse_audit(full, removed)},
        "context_swap": {"metrics": aggregate_results(swapped), "switch_audit": switch_audit(full, swapped)},
        "latest_state_update": {"present": bool(latest), **summarize_results(latest)},
        "rows": full,
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Held-out Dynamic-logit-v1 evaluation requires CUDA BF16")
    paths = [args.dynamic_dir / "manifest.json", args.transition_dir / "manifest.json",
             args.source_dir / "manifest.json", args.tokenizer, args.checkpoint]
    hashes = [SEALED["dynamic_manifest_sha256"], SEALED["transition_manifest_sha256"],
              SEALED["source_manifest_sha256"], SEALED["tokenizer_sha256"], SEALED["checkpoint_sha256"]]
    for path, digest, description in zip(paths, hashes,
                                         ("dynamic manifest", "transition manifest", "source manifest", "tokenizer", "checkpoint"),
                                         strict=True):
        verify_file(path, digest, description)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    _require(tokenizer.get_vocab_size() == 4096, "Tokenizer vocabulary changed")
    device = torch.device("cuda")
    model = load_model(args.checkpoint, SEALED["checkpoint_sha256"], device)
    results = {}
    provenance = {}
    for split in ("test", "generalization"):
        grouped, pairs, split_provenance = load_verified_split(
            split=split, dynamic_dir=args.dynamic_dir, transition_dir=args.transition_dir,
            source_dir=args.source_dir, tokenizer=tokenizer, locks=SEALED,
        )
        results[split] = evaluate_split(model, grouped, pairs, tokenizer, device=device,
                                        batch_size=args.batch_size)
        provenance[split] = split_provenance
    return {
        "evaluation": "FictionPulper Dynamic-logit-v1 held-out evaluation",
        "score": "mean decision-span token log probability", "full_context": True,
        "precision": "CUDA BF16", "model_parameter_count": EXPECTED_PARAMETER_COUNT,
        "tied_lm_head": True, "auxiliary_heads": False,
        "category_precedence": list(CATEGORY_PRECEDENCE),
        "provenance": {"sealed_hashes": dict(SEALED), "splits": provenance},
        "latest_state_test_generalization": {
            split: results[split]["latest_state_update"] for split in ("test", "generalization")
        },
        "splits": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-dir", type=Path, default=Path("data/dynamic_logit_v1"))
    parser.add_argument("--transition-dir", type=Path, default=Path("data/state_transition_v1"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/counterfactual_narrative_v1"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/tokenizer.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "checkpoints/fictionpulper-15m-data30m-dynamic-logit-v1/best-validation.pt"))
    parser.add_argument("--output", type=Path, default=Path(
        "runs/fictionpulper-15m-data30m-dynamic-logit-v1/held-out-evaluation.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    _require(args.batch_size > 0, "batch-size must be positive")
    result = run_evaluation(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, result)
    print(json.dumps({"output": str(args.output), "splits": list(result["splits"])}, indent=2))


if __name__ == "__main__":
    main()
