"""Create deterministic, leakage-audited splits for sealed Corpus-v2."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.train_tokenizer import load_corpus, sha256_file, write_json_atomic


SPLITS = ("train", "validation", "test")
SPLIT_FRACTIONS = {"train": 0.85, "validation": 0.075, "test": 0.075}


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def build_constraint_groups(
    records: list[dict[str, Any]],
    near_duplicate_edges: list[dict[str, Any]],
) -> tuple[list[list[str]], dict[str, Any]]:
    """Join source collections and transitive reviewed duplicate clusters."""
    disjoint = DisjointSet()
    record_ids = {record["id"] for record in records}
    collections: dict[str, list[str]] = defaultdict(list)
    for record in records:
        disjoint.find(record["id"])
        collection = record.get("source_collection")
        if not isinstance(collection, str) or not collection.strip():
            raise ValueError(f"Record {record['id']} has no source_collection")
        collections[collection].append(record["id"])

    for member_ids in collections.values():
        for record_id in member_ids[1:]:
            disjoint.union(member_ids[0], record_id)

    unknown_edge_members: set[str] = set()
    for edge in near_duplicate_edges:
        left_id = edge["left_id"]
        right_id = edge["right_id"]
        if left_id not in record_ids:
            unknown_edge_members.add(left_id)
        if right_id not in record_ids:
            unknown_edge_members.add(right_id)
        disjoint.union(left_id, right_id)

    groups: dict[str, list[str]] = defaultdict(list)
    for record_id in sorted(record_ids):
        groups[disjoint.find(record_id)].append(record_id)
    ordered_groups = sorted(groups.values(), key=lambda members: members[0])
    metadata = {
        "source_collection_count": len(collections),
        "near_duplicate_edge_count": len(near_duplicate_edges),
        "near_duplicate_external_member_count": len(unknown_edge_members),
        "constraint_group_count": len(ordered_groups),
        "largest_constraint_group_documents": max(map(len, ordered_groups)),
    }
    return ordered_groups, metadata


def grouped_anchored_assignments(
    records: list[dict[str, Any]],
    groups: list[list[str]],
    historical_assignments: dict[str, str],
    *,
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    record_ids = {record["id"] for record in records}
    retained_historical = record_ids & historical_assignments.keys()
    if set(historical_assignments.values()) - set(SPLITS):
        raise ValueError("Historical split manifest contains an unknown split")
    if set().union(*map(set, groups)) != record_ids:
        raise ValueError("Constraint groups do not cover the corpus exactly")

    total = len(records)
    targets = {
        "train": round(total * SPLIT_FRACTIONS["train"]),
        "validation": round(total * SPLIT_FRACTIONS["validation"]),
    }
    targets["test"] = total - targets["train"] - targets["validation"]

    assignments: dict[str, str] = {}
    assigned_counts = Counter()
    anchored_groups = []
    unanchored_groups = []
    for members in groups:
        historical_counts = Counter(
            historical_assignments[record_id]
            for record_id in members
            if record_id in historical_assignments
        )
        if not historical_counts:
            unanchored_groups.append(members)
            continue
        # A collection can contain old records from several splits. Majority assignment
        # preserves the greatest possible number under the indivisible-group rule.
        split = min(
            SPLITS,
            key=lambda candidate: (-historical_counts[candidate], SPLITS.index(candidate)),
        )
        for record_id in members:
            assignments[record_id] = split
        assigned_counts[split] += len(members)
        anchored_groups.append(
            {
                "group_id": members[0],
                "document_count": len(members),
                "historical_counts": dict(sorted(historical_counts.items())),
                "assigned_split": split,
                "historical_conflict": len(historical_counts) > 1,
            }
        )

    rng = random.Random(seed)
    rng.shuffle(unanchored_groups)
    unanchored_groups.sort(key=len, reverse=True)
    unanchored_group_assignments = []
    for members in unanchored_groups:
        group_size = len(members)
        split = min(
            SPLITS,
            key=lambda candidate: (
                sum(
                    (
                        (
                            assigned_counts[other]
                            + (group_size if other == candidate else 0)
                            - targets[other]
                        )
                        / targets[other]
                    )
                    ** 2
                    for other in SPLITS
                ),
                SPLITS.index(candidate),
            ),
        )
        for record_id in members:
            assignments[record_id] = split
        assigned_counts[split] += group_size
        unanchored_group_assignments.append(
            {
                "group_id": members[0],
                "document_count": group_size,
                "assigned_split": split,
            }
        )

    changed_historical = [
        {
            "id": record_id,
            "historical_split": historical_assignments[record_id],
            "data30m_split": assignments[record_id],
        }
        for record_id in sorted(retained_historical)
        if assignments[record_id] != historical_assignments[record_id]
    ]
    preserved_count = len(retained_historical) - len(changed_historical)
    metadata = {
        "seed": seed,
        "policy": (
            "Treat exact source collections and transitive reviewed near-duplicate clusters "
            "as indivisible. Assign historically mixed groups to their majority historical "
            "split, then balance unanchored groups toward 85/7.5/7.5 document counts."
        ),
        "target_fractions": SPLIT_FRACTIONS,
        "target_document_counts": targets,
        "actual_document_counts": dict(sorted(assigned_counts.items())),
        "actual_document_fractions": {
            split: assigned_counts[split] / total for split in SPLITS
        },
        "retained_historical_document_count": len(retained_historical),
        "preserved_historical_assignment_count": preserved_count,
        "changed_historical_assignment_count": len(changed_historical),
        "changed_historical_assignments": changed_historical,
        "historical_assignment_preservation_rate": preserved_count
        / len(retained_historical),
        "anchored_constraint_groups": anchored_groups,
        "unanchored_constraint_groups": unanchored_group_assignments,
    }
    return assignments, metadata


def audit_assignments(
    records: list[dict[str, Any]],
    assignments: dict[str, str],
    groups: list[list[str]],
    near_duplicate_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    record_ids = {record["id"] for record in records}
    if set(assignments) != record_ids:
        raise RuntimeError("Split assignments do not cover the corpus exactly")
    if set(assignments.values()) - set(SPLITS):
        raise RuntimeError("Split assignments contain an unknown split")

    group_violations = [
        {
            "member_ids": members,
            "splits": sorted({assignments[member] for member in members}),
        }
        for members in groups
        if len({assignments[member] for member in members}) > 1
    ]
    exact_hash_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        exact_hash_splits[record["text_hash"]].add(assignments[record["id"]])
    exact_violations = [
        text_hash for text_hash, splits in exact_hash_splits.items() if len(splits) > 1
    ]
    edge_violations = []
    edges_with_two_retained_members = 0
    for edge in near_duplicate_edges:
        left_id = edge["left_id"]
        right_id = edge["right_id"]
        if left_id not in assignments or right_id not in assignments:
            continue
        edges_with_two_retained_members += 1
        if assignments[left_id] != assignments[right_id]:
            edge_violations.append(
                {
                    "left_id": left_id,
                    "right_id": right_id,
                    "left_split": assignments[left_id],
                    "right_split": assignments[right_id],
                }
            )
    passed = not group_violations and not exact_violations and not edge_violations
    audit = {
        "passed": passed,
        "documents_checked": len(records),
        "all_documents_assigned_once": True,
        "pairwise_document_overlap": {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        },
        "constraint_groups_checked": len(groups),
        "constraint_group_cross_split_violations": group_violations,
        "unique_exact_text_hashes": len(exact_hash_splits),
        "exact_duplicate_cross_split_violations": exact_violations,
        "near_duplicate_edges_checked": len(near_duplicate_edges),
        "near_duplicate_edges_with_two_retained_members": edges_with_two_retained_members,
        "near_duplicate_cross_split_violations": edge_violations,
    }
    if not passed:
        raise RuntimeError(f"Data30M split leakage audit failed: {audit}")
    return audit


def prepare_data30m(
    *,
    corpus_path: Path,
    historical_split_path: Path,
    near_duplicate_edges_path: Path,
    exclusions_path: Path,
    output_dir: Path,
    split_seed: int,
) -> dict[str, Any]:
    records = load_corpus(corpus_path)
    historical_payload = json.loads(historical_split_path.read_text(encoding="utf-8"))
    near_duplicate_edges = load_jsonl(near_duplicate_edges_path)
    exclusions = json.loads(exclusions_path.read_text(encoding="utf-8"))
    groups, group_metadata = build_constraint_groups(records, near_duplicate_edges)
    assignments, split_metadata = grouped_anchored_assignments(
        records, groups, historical_payload["assignments"], seed=split_seed
    )
    leakage_audit = audit_assignments(records, assignments, groups, near_duplicate_edges)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "splits.json"
    audit_path = output_dir / "leakage-audit.json"
    split_payload = {
        **split_metadata,
        **group_metadata,
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "historical_split_path": str(historical_split_path),
        "historical_split_sha256": sha256_file(historical_split_path),
        "near_duplicate_edges_path": str(near_duplicate_edges_path),
        "near_duplicate_edges_sha256": sha256_file(near_duplicate_edges_path),
        "documented_seed_correctness_exclusions": [
            item["canonical_id"] for item in exclusions["exclusions"]
        ],
        "assignments": dict(sorted(assignments.items())),
    }
    write_json_atomic(split_path, split_payload)
    leakage_audit.update(
        {
            "split_path": str(split_path),
            "split_sha256": sha256_file(split_path),
            "historical_benchmark_effect": (
                "Corpus-v2 omits three documented historical train duplicates. Enforcing "
                "collection-level grouping also changes assignments in historically mixed "
                "collections, so Data30M validation/test define a new benchmark; sealed Data10M "
                "artifacts and reported metrics remain unchanged."
            ),
            "excluded_historical_train_ids": [
                item["canonical_id"] for item in exclusions["exclusions"]
            ],
        }
    )
    write_json_atomic(audit_path, leakage_audit)
    summary = {
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path),
        "record_count": len(records),
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
        "leakage_audit_path": str(audit_path),
        "leakage_audit_sha256": sha256_file(audit_path),
        "split_document_counts": split_metadata["actual_document_counts"],
        "preserved_historical_assignment_count": split_metadata[
            "preserved_historical_assignment_count"
        ],
        "changed_historical_assignment_count": split_metadata[
            "changed_historical_assignment_count"
        ],
        "leakage_audit_passed": leakage_audit["passed"],
        "historical_experiments_unchanged": True,
    }
    write_json_atomic(output_dir / "preparation-summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus_v2/corpus.jsonl"))
    parser.add_argument(
        "--historical-splits",
        type=Path,
        default=Path("data/corpus_v1/data10m-v1/splits.json"),
    )
    parser.add_argument(
        "--near-duplicate-edges",
        type=Path,
        default=Path("data/corpus_v2/near-duplicate-candidates.jsonl"),
    )
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=Path(
            "experiments/fictionpulper-corpus-v2-30m/seed-correctness-exclusions.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/corpus_v2/data30m-v1")
    )
    parser.add_argument("--split-seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_data30m(
        corpus_path=args.corpus,
        historical_split_path=args.historical_splits,
        near_duplicate_edges_path=args.near_duplicate_edges,
        exclusions_path=args.exclusions,
        output_dir=args.output_dir,
        split_seed=args.split_seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
