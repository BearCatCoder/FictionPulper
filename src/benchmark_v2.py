"""Validate the frozen Narrative Benchmark v2 specification and allocation plan."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.continuity_curriculum.common import canonical_json, sha256_bytes, sha256_file, write_jsonl


BENCHMARK_ID = "fictionpulper-narrative-benchmark-v2"
SCHEMA_VERSION = 2
EXPECTED_PROTOCOL_SHA256 = "84e34abf414d81ddef6e8baef71d07d2adf65b0dcfe152607b239f83b1095c79"
FAMILIES = (
    "static_fact",
    "ownership",
    "location",
    "knowledge",
    "goal",
    "causal_dependency",
    "temporal_ordering",
)
DEPTH_BUCKETS = ("1", "2", "3", "4_plus")
SPLIT_REPETITIONS = ("development", "test", "generalization", "generalization")
EVALUATION_MODES = ("forced_choice", "teacher_forced", "free_generation")


class BenchmarkValidationError(ValueError):
    """Raised when a v2 protocol artifact violates a frozen invariant."""


def stable_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def build_allocation_plan() -> list[dict[str, Any]]:
    """Build the deterministic 7 family x 4 depth x 4 repetition plan."""
    records: list[dict[str, Any]] = []
    ordinal = 0
    for family_index, family in enumerate(FAMILIES):
        for depth_index, depth_bucket in enumerate(DEPTH_BUCKETS):
            for repetition, split in enumerate(SPLIT_REPETITIONS):
                reasoning_depth = int(depth_bucket) if depth_bucket != "4_plus" else 4 + repetition % 2
                orientation = (family_index + depth_index + repetition) % 2
                pool = f"{split}-{family}-{depth_bucket}-pool-{repetition}"
                record: dict[str, Any] = {
                    "scenario_id": f"nbv2-{ordinal:03d}",
                    "schema_version": SCHEMA_VERSION,
                    "split": split,
                    "state_family": family,
                    "reasoning_depth": reasoning_depth,
                    "depth_bucket": depth_bucket,
                    "template_id": f"{split}-{family}-{depth_bucket}-template-{repetition}",
                    "template_family_id": f"{split}-{family}-family-{repetition}",
                    "lexical_pool_id": pool,
                    "name_pool_id": f"{pool}-names",
                    "verb_pool_id": f"{pool}-verbs",
                    "novelty": {
                        "names": "split_exclusive",
                        "verbs": "split_exclusive",
                        "template": "split_exclusive",
                        "jointly_unseen": split == "generalization",
                    },
                    "worlds": {
                        "A": {"correct_candidate": "X" if orientation == 0 else "Y"},
                        "B": {"correct_candidate": "Y" if orientation == 0 else "X"},
                    },
                    "candidate_presentation_order": (
                        ["X", "Y"]
                        if (family_index + depth_index + repetition + repetition // 2) % 2 == 0
                        else ["Y", "X"]
                    ),
                    "evaluation_modes": list(EVALUATION_MODES),
                }
                record["stable_hash"] = stable_hash(record)
                records.append(record)
                ordinal += 1
    return records


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkValidationError(message)


def validate_plan(records: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    _require(len(records) == protocol["scenario_count"] >= 100, "scenario count is not frozen or below 100")

    ids = [record["scenario_id"] for record in records]
    hashes = [record["stable_hash"] for record in records]
    _require(len(ids) == len(set(ids)), "duplicate scenario ID")
    _require(len(hashes) == len(set(hashes)), "duplicate scenario stable hash")
    for record in records:
        unhashed = {key: value for key, value in record.items() if key != "stable_hash"}
        _require(record["stable_hash"] == stable_hash(unhashed), f"stable hash mismatch: {record['scenario_id']}")
        _require(tuple(record["evaluation_modes"]) == EVALUATION_MODES, "evaluation modes changed or merged")
        _require(set(record["worlds"]) == {"A", "B"}, "scenario lacks paired worlds")
        _require(
            record["worlds"]["A"]["correct_candidate"] != record["worlds"]["B"]["correct_candidate"],
            "paired worlds do not reverse the answer",
        )
        expected_bucket = str(record["reasoning_depth"]) if record["reasoning_depth"] < 4 else "4_plus"
        _require(record["depth_bucket"] == expected_bucket, "reasoning depth and bucket disagree")

    cells = Counter((record["state_family"], record["depth_bucket"]) for record in records)
    expected_cells = {(family, depth): 4 for family in FAMILIES for depth in DEPTH_BUCKETS}
    _require(dict(cells) == expected_cells, "family/depth grid is not exactly balanced")
    split_counts = Counter(record["split"] for record in records)
    _require(dict(split_counts) == protocol["split_policy"]["scenario_counts"], "split counts changed")

    owners: dict[str, defaultdict[str, set[str]]] = {
        field: defaultdict(set) for field in (
            "template_id", "template_family_id", "name_pool_id", "verb_pool_id", "lexical_pool_id"
        )
    }
    for record in records:
        for field, values in owners.items():
            values[record[field]].add(record["split"])
        if record["split"] == "generalization":
            _require(record["novelty"]["jointly_unseen"] is True, "generalization novelty is not joint")
    leakage = {
        field: sorted(value for value, splits in values.items() if len(splits) > 1)
        for field, values in owners.items()
    }
    _require(not any(leakage.values()), "template or lexical pool crosses splits")

    orders = Counter(tuple(record["candidate_presentation_order"]) for record in records)
    _require(abs(orders[("X", "Y")] - orders[("Y", "X")]) <= 1, "candidate order is imbalanced")
    for field in ("split", "depth_bucket", "reasoning_depth", "state_family"):
        strata: defaultdict[Any, Counter[tuple[str, ...]]] = defaultdict(Counter)
        for record in records:
            strata[record[field]][tuple(record["candidate_presentation_order"])] += 1
        _require(
            all(abs(counts[("X", "Y")] - counts[("Y", "X")]) <= 1 for counts in strata.values()),
            f"candidate order is imbalanced by {field}",
        )
    world_a = Counter(record["worlds"]["A"]["correct_candidate"] for record in records)
    _require(world_a["X"] == world_a["Y"], "world-A labels are imbalanced")
    _require(records == build_allocation_plan(), "allocation plan differs from deterministic v2 plan")
    return {
        "passed": True,
        "scenario_count": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "family_depth_cells": len(cells),
        "scenarios_per_family_depth_cell": sorted(set(cells.values())),
        "candidate_orders": {"XY": orders[("X", "Y")], "YX": orders[("Y", "X")]},
        "world_a_labels": dict(sorted(world_a.items())),
        "allocation_id_leakage": leakage,
    }


def validate_benchmark(root: Path) -> dict[str, Any]:
    protocol_path = root / "protocol.json"
    _require(sha256_file(protocol_path) == EXPECTED_PROTOCOL_SHA256, "protocol artifact hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _require(protocol["benchmark_id"] == BENCHMARK_ID, "benchmark ID mismatch")
    _require(protocol["protocol_version"] == SCHEMA_VERSION, "protocol version mismatch")
    _require(protocol["status"] == "frozen-specification", "v2 specification is not frozen")
    _require(tuple(protocol["state_families"]) == FAMILIES, "state families changed")
    _require(tuple(protocol["depth_buckets"]) == DEPTH_BUCKETS, "depth buckets changed")

    artifacts = protocol["artifacts"]
    for name, artifact in artifacts.items():
        path = root / artifact["path"]
        _require(path.is_file(), f"missing {name} artifact: {path}")
        _require(sha256_file(path) == artifact["sha256"], f"{name} artifact hash mismatch")

    records = load_jsonl(root / artifacts["allocation_plan"]["path"])
    report = validate_plan(records, protocol)
    report["benchmark_id"] = BENCHMARK_ID
    report["artifact_hashes_verified"] = sorted(artifacts)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmarks/narrative-v2"))
    parser.add_argument("--initialize-plan", action="store_true", help="write the plan only when absent")
    args = parser.parse_args()
    plan_path = args.root / "scenario-allocation.jsonl"
    if args.initialize_plan:
        if plan_path.exists():
            raise FileExistsError(f"refusing to overwrite frozen allocation plan: {plan_path}")
        write_jsonl(plan_path, build_allocation_plan())
        print(f"Wrote {len(build_allocation_plan())} scenarios to {plan_path}")
        return
    print(json.dumps(validate_benchmark(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
