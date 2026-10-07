"""Resolve Corpus-v1 near duplicates and create anchored Data10M splits."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.corpus_v1 import validate_corpus_records, write_jsonl
from src.train_tokenizer import load_corpus, sha256_file, write_json_atomic


SPLITS = ("train", "validation", "test")
SPLIT_FRACTIONS = {"train": 0.85, "validation": 0.075, "test": 0.075}

# These decisions were made by full-text review of all reported pairs. Seed records
# win seed/new ties so their historical assignments remain anchored.
ALTERNATE_EDITION_PAIRS = {
    frozenset(("932510eaad15", "162495d6f1c1")),
    frozenset(("e89be87f3f58", "3650b1f0a3ad")),
    frozenset(("c5c114ed174c", "pg-839-655c632af99f")),
    frozenset(("2dc5bd4e48ac", "pg-2148-188c90b24baf")),
    frozenset(("0a2e1b20945d", "pg-2148-e57ce9341fd8")),
    frozenset(("pg-508-912f86bef5dc", "pg-1916-25fa5b227905")),
    frozenset(("pg-513-68cadc699c2a", "pg-1916-5386afcdf795")),
    frozenset(("pg-2138-389b1040deb2", "pg-2569-c9692329ed2f")),
    frozenset(("pg-2138-308634d20766", "pg-2569-1a079bb44573")),
}

PREFERRED_NON_SEED = {
    frozenset(("pg-508-912f86bef5dc", "pg-1916-25fa5b227905")): "pg-508-912f86bef5dc",
    frozenset(("pg-513-68cadc699c2a", "pg-1916-5386afcdf795")): "pg-513-68cadc699c2a",
    frozenset(("pg-2138-389b1040deb2", "pg-2569-c9692329ed2f")): "pg-2569-c9692329ed2f",
    frozenset(("pg-2138-308634d20766", "pg-2569-1a079bb44573")): "pg-2569-1a079bb44573",
}


def resolve_near_duplicates(
    records: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    seed_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {record["id"]: record for record in records}
    excluded_ids: set[str] = set()
    resolutions = []
    for pair in pairs:
        left_id = pair["left_id"]
        right_id = pair["right_id"]
        pair_key = frozenset((left_id, right_id))
        if left_id not in by_id or right_id not in by_id:
            raise ValueError(f"Near-duplicate pair references an unknown record: {pair_key}")
        seed_members = pair_key & seed_ids
        if len(seed_members) == 1:
            retained_id = next(iter(seed_members))
            reason = (
                "Full-text review found the same narrative. The original seed record is retained "
                "to preserve its anchored smoke assignment; the new Gutenberg duplicate is excluded."
            )
        elif pair_key in PREFERRED_NON_SEED:
            retained_id = PREFERRED_NON_SEED[pair_key]
            reason = (
                "Full-text review found the same narrative in two Gutenberg editions. The retained "
                "record has cleaner text, fewer transcription artifacts, or more complete structure."
            )
        elif len(seed_members) == 2:
            retained_id = left_id
            reason = (
                "Full-text review found alternate seed editions of the same narrative. The retained "
                "edition has fewer apparent corruptions; the untouched legacy benchmark still contains both."
            )
        else:
            raise ValueError(f"No reviewed retention decision for pair: {sorted(pair_key)}")
        excluded_id = right_id if retained_id == left_id else left_id
        if excluded_id in excluded_ids:
            raise ValueError(f"Record appears in multiple reviewed pairs: {excluded_id}")
        excluded_ids.add(excluded_id)
        classification = (
            "same story / alternate edition"
            if pair_key in ALTERNATE_EDITION_PAIRS
            else "same story / formatting variation"
        )
        left = by_id[left_id]
        right = by_id[right_id]
        resolutions.append(
            {
                "record_a_id": left_id,
                "record_b_id": right_id,
                "record_a_title": left["title"],
                "record_b_title": right["title"],
                "record_a_author": left.get("author"),
                "record_b_author": right.get("author"),
                "similarity": {
                    "simhash_hamming_distance": pair["simhash_hamming_distance"],
                    "word_5gram_jaccard": pair["word_5gram_jaccard"],
                    "length_ratio": pair["length_ratio"],
                },
                "classification": classification,
                "retained_id": retained_id,
                "excluded_ids": [excluded_id],
                "reason": reason,
            }
        )
    resolved = [record for record in records if record["id"] not in excluded_ids]
    report = {
        "review_method": "manual full-text comparison of every reported pair",
        "input_pair_count": len(pairs),
        "classification_counts": dict(
            sorted(Counter(item["classification"] for item in resolutions).items())
        ),
        "excluded_record_count": len(excluded_ids),
        "excluded_seed_ids": sorted(excluded_ids & seed_ids),
        "resolutions": resolutions,
    }
    return resolved, report


def grouped_anchored_assignments(
    records: list[dict[str, Any]],
    seed_assignments: dict[str, str],
    *,
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    record_ids = {record["id"] for record in records}
    assignments = {
        record_id: split
        for record_id, split in seed_assignments.items()
        if record_id in record_ids
    }
    if set(assignments.values()) - set(SPLITS):
        raise ValueError("Original seed manifest contains an unknown split")
    new_records = [record for record in records if record["id"] not in seed_assignments]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in new_records:
        groups[record["source_id"]].append(record)

    total = len(records)
    targets = {
        "train": round(total * SPLIT_FRACTIONS["train"]),
        "validation": round(total * SPLIT_FRACTIONS["validation"]),
    }
    targets["test"] = total - targets["train"] - targets["validation"]
    fixed_counts = Counter(assignments.values())
    additional_targets = {
        split: max(1, targets[split] - fixed_counts[split]) for split in SPLITS
    }
    additional_counts = Counter()

    ordered_groups = list(groups.items())
    random.Random(seed).shuffle(ordered_groups)
    ordered_groups.sort(key=lambda item: len(item[1]), reverse=True)
    group_assignments: dict[str, str] = {}
    for source_id, group_records in ordered_groups:
        group_size = len(group_records)
        best_split = min(
            SPLITS,
            key=lambda candidate: (
                sum(
                    (
                        (
                            additional_counts[split]
                            + (group_size if split == candidate else 0)
                            - additional_targets[split]
                        )
                        / additional_targets[split]
                    )
                    ** 2
                    for split in SPLITS
                ),
                SPLITS.index(candidate),
            ),
        )
        group_assignments[source_id] = best_split
        additional_counts[best_split] += group_size
        for record in group_records:
            assignments[record["id"]] = best_split

    if set(assignments) != record_ids:
        raise RuntimeError("Split assignments do not cover the resolved corpus")
    split_counts = Counter(assignments.values())
    group_leaks = {
        source_id: sorted({assignments[record["id"]] for record in group_records})
        for source_id, group_records in groups.items()
        if len({assignments[record["id"]] for record in group_records}) > 1
    }
    if group_leaks:
        raise RuntimeError(f"New source collections cross splits: {group_leaks}")
    metadata = {
        "seed": seed,
        "policy": "preserve retained seed assignments; group new records by source_id",
        "target_fractions": SPLIT_FRACTIONS,
        "target_document_counts": targets,
        "actual_document_counts": dict(sorted(split_counts.items())),
        "actual_document_fractions": {
            split: split_counts[split] / total for split in SPLITS
        },
        "retained_seed_document_counts": dict(sorted(fixed_counts.items())),
        "new_document_counts": dict(sorted(additional_counts.items())),
        "new_source_collection_count": len(groups),
        "new_collection_cross_split_violations": group_leaks,
        "group_assignments": dict(sorted(group_assignments.items())),
    }
    return assignments, metadata


def audit_seed_duplicate_splits(
    resolution_report: dict[str, Any],
    seed_ids: set[str],
    original_assignments: dict[str, str],
    data10m_assignments: dict[str, str],
) -> None:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for resolution in resolution_report["resolutions"]:
        left_id = resolution["record_a_id"]
        right_id = resolution["record_b_id"]
        adjacency[left_id].add(right_id)
        adjacency[right_id].add(left_id)
    cluster_audit = []
    visited: set[str] = set()
    for start_id in sorted(adjacency):
        if start_id in visited:
            continue
        pending = [start_id]
        cluster: set[str] = set()
        while pending:
            record_id = pending.pop()
            if record_id in cluster:
                continue
            cluster.add(record_id)
            pending.extend(adjacency[record_id] - cluster)
        visited.update(cluster)
        retained_members = sorted(cluster & data10m_assignments.keys())
        retained_splits = sorted(
            {data10m_assignments[record_id] for record_id in retained_members}
        )
        cluster_audit.append(
            {
                "member_ids": sorted(cluster),
                "retained_data10m_member_ids": retained_members,
                "retained_data10m_splits": retained_splits,
                "crosses_data10m_splits": len(retained_splits) > 1,
            }
        )
    leaking_clusters = [
        cluster for cluster in cluster_audit if cluster["crosses_data10m_splits"]
    ]
    resolution_report["duplicate_cluster_split_audit"] = cluster_audit
    resolution_report["data10m_near_duplicate_split_leakage"] = bool(leaking_clusters)
    if leaking_clusters:
        raise RuntimeError(f"Near-duplicate clusters cross Data10M splits: {leaking_clusters}")

    seed_pair_audit = []
    contamination = {"legacy_seed_validation": [], "legacy_seed_test": []}
    for resolution in resolution_report["resolutions"]:
        pair_ids = [resolution["record_a_id"], resolution["record_b_id"]]
        if not all(record_id in seed_ids for record_id in pair_ids):
            continue
        original_splits = [original_assignments[record_id] for record_id in pair_ids]
        evaluation_members = [
            (record_id, split)
            for record_id, split in zip(pair_ids, original_splits, strict=True)
            if split in {"validation", "test"}
        ]
        crosses_training_evaluation = "train" in original_splits and bool(
            evaluation_members
        )
        affected_metrics = []
        if crosses_training_evaluation:
            for record_id, split in evaluation_members:
                metric = f"legacy_seed_{split}"
                contamination[metric].append(record_id)
                affected_metrics.append(metric)
            resolution["reason"] += (
                " The alternate edition crossed the original smoke training/evaluation boundary, "
                "so the affected untouched legacy metric is flagged as contaminated; Data10M "
                "remains clean because the evaluation-side member is excluded."
            )
        retained_id = resolution["retained_id"]
        seed_pair_audit.append(
            {
                "record_a_id": pair_ids[0],
                "record_a_original_smoke_split": original_splits[0],
                "record_b_id": pair_ids[1],
                "record_b_original_smoke_split": original_splits[1],
                "classification": resolution["classification"],
                "retained_id": retained_id,
                "retained_data10m_split": data10m_assignments[retained_id],
                "crosses_training_evaluation_boundary": crosses_training_evaluation,
                "data10m_cluster_crosses_splits": False,
                "affected_legacy_metrics": affected_metrics,
            }
        )
    resolution_report["seed_to_seed_split_audit"] = seed_pair_audit
    resolution_report["legacy_metric_contamination"] = {
        metric: sorted(record_ids) for metric, record_ids in contamination.items()
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def prepare_data10m(
    *,
    corpus_path: Path,
    near_duplicates_path: Path,
    seed_corpus_path: Path,
    seed_split_path: Path,
    output_dir: Path,
    split_seed: int,
) -> dict[str, Any]:
    records = load_corpus(corpus_path)
    seed_records = load_corpus(seed_corpus_path)
    seed_ids = {record["id"] for record in seed_records}
    near_duplicate_pairs = load_jsonl(near_duplicates_path)
    resolved, resolution_report = resolve_near_duplicates(
        records, near_duplicate_pairs, seed_ids
    )
    validate_corpus_records(resolved)

    original_split_payload = json.loads(seed_split_path.read_text(encoding="utf-8"))
    assignments, split_metadata = grouped_anchored_assignments(
        resolved, original_split_payload["assignments"], seed=split_seed
    )
    for record_id, split in original_split_payload["assignments"].items():
        if record_id in assignments and assignments[record_id] != split:
            raise RuntimeError(f"Original seed assignment changed: {record_id}")

    audit_seed_duplicate_splits(
        resolution_report,
        seed_ids,
        original_split_payload["assignments"],
        assignments,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "corpus.jsonl"
    resolution_path = output_dir / "near-duplicate-resolution.json"
    split_path = output_dir / "splits.json"
    write_jsonl(resolved_path, resolved)
    write_json_atomic(resolution_path, resolution_report)
    split_payload = {
        **split_metadata,
        "original_seed_split_path": str(seed_split_path),
        "original_seed_split_sha256": sha256_file(seed_split_path),
        "excluded_seed_ids": resolution_report["excluded_seed_ids"],
        "assignments": dict(sorted(assignments.items())),
    }
    write_json_atomic(split_path, split_payload)
    summary = {
        "input_corpus_path": str(corpus_path),
        "input_corpus_sha256": sha256_file(corpus_path),
        "resolved_corpus_path": str(resolved_path),
        "resolved_corpus_sha256": sha256_file(resolved_path),
        "input_record_count": len(records),
        "resolved_record_count": len(resolved),
        "near_duplicate_resolution_path": str(resolution_path),
        "near_duplicate_resolution_sha256": sha256_file(resolution_path),
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
        "split_document_counts": split_metadata["actual_document_counts"],
        "all_retained_seed_assignments_preserved": True,
        "legacy_benchmark_remains_untouched": True,
    }
    write_json_atomic(output_dir / "preparation-summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus_v1/corpus.jsonl"))
    parser.add_argument(
        "--near-duplicates",
        type=Path,
        default=Path("data/corpus_v1/near-duplicates.jsonl"),
    )
    parser.add_argument("--seed-corpus", type=Path, default=Path("data/corpus/stories.jsonl"))
    parser.add_argument(
        "--seed-splits", type=Path, default=Path("data/corpus/stories.splits.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/corpus_v1/data10m-v1")
    )
    parser.add_argument("--split-seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_data10m(
        corpus_path=args.corpus,
        near_duplicates_path=args.near_duplicates,
        seed_corpus_path=args.seed_corpus,
        seed_split_path=args.seed_splits,
        output_dir=args.output_dir,
        split_seed=args.split_seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
