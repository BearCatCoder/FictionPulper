"""Bounded duplicate, template, state, distance, and quality audits."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from src.continuity_curriculum.common import normalized_text_hash, normalized_words, sha256_file, text_shingles


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def exact_duplicate_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    owners: dict[str, list[str]] = defaultdict(list)
    for record in records:
        owners[normalized_text_hash(record["text"])].append(record["id"])
    duplicates = [
        {"normalized_sha256": digest, "ids": ids}
        for digest, ids in sorted(owners.items())
        if len(ids) > 1
    ]
    return {
        "passed": not duplicates,
        "definition": "SHA-256 of normalized lowercase word sequence",
        "record_count": len(records),
        "unique_normalized_text_count": len(owners),
        "duplicate_groups": duplicates,
    }


def _fingerprint(shingles: set[str], size: int) -> tuple[int, ...]:
    return tuple(sorted(
        int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")
        for value in shingles
    )[:size])


def similarity_audit(
    records: list[dict[str, Any]],
    *,
    shingle_width: int,
    fingerprint_size: int,
    candidate_shared_hashes: int,
    maximum_posting_size: int,
    maximum_candidate_pairs: int,
    reject_jaccard: float,
) -> dict[str, Any]:
    """Bound work with common-posting and candidate-pair limits; never scan all pairs."""
    shingles_by_id: dict[str, set[str]] = {}
    split_by_id: dict[str, str] = {}
    postings: dict[int, list[str]] = defaultdict(list)
    for record in records:
        record_id = record["id"]
        shingles = text_shingles(record["text"], shingle_width)
        shingles_by_id[record_id] = shingles
        split_by_id[record_id] = record["split"]
        for value in _fingerprint(shingles, fingerprint_size):
            postings[value].append(record_id)

    shared: Counter[tuple[str, str]] = Counter()
    skipped_common = 0
    truncated = False
    for members in postings.values():
        if len(members) > maximum_posting_size:
            skipped_common += 1
            continue
        ordered = sorted(members)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                shared[(left, right)] += 1
                if len(shared) > maximum_candidate_pairs:
                    truncated = True
                    break
            if truncated:
                break
        if truncated:
            break

    candidates = sorted(pair for pair, count in shared.items() if count >= candidate_shared_hashes)
    flagged = []
    for left, right in candidates:
        left_shingles = shingles_by_id[left]
        right_shingles = shingles_by_id[right]
        union = left_shingles | right_shingles
        score = len(left_shingles & right_shingles) / len(union) if union else 1.0
        if score >= reject_jaccard:
            flagged.append({
                "left_id": left,
                "right_id": right,
                "left_split": split_by_id[left],
                "right_split": split_by_id[right],
                "jaccard": score,
            })
    return {
        "passed": not flagged and not truncated,
        "method": "bottom-k word-shingle candidates followed by exact Jaccard",
        "complexity_policy": "no all-pairs scan; common postings and total candidates are bounded",
        "shingle_width": shingle_width,
        "fingerprint_size": fingerprint_size,
        "candidate_shared_hashes": candidate_shared_hashes,
        "maximum_posting_size": maximum_posting_size,
        "maximum_candidate_pairs": maximum_candidate_pairs,
        "candidate_limit_exceeded": truncated,
        "reject_jaccard": reject_jaccard,
        "candidate_pair_count": len(candidates),
        "skipped_common_fingerprint_count": skipped_common,
        "flagged_pairs": flagged,
    }


def template_audit(records: list[dict[str, Any]], maximum_family_share: float) -> dict[str, Any]:
    counts = Counter(record["template_family"] for record in records)
    variants = Counter(str(record["paraphrase_variant"]) for record in records)
    total = len(records)
    shares = {family: count / total for family, count in sorted(counts.items())} if total else {}
    largest = max(shares.values(), default=0.0)
    return {
        "passed": bool(counts) and largest <= maximum_family_share,
        "maximum_family_share": maximum_family_share,
        "observed_largest_family_share": largest,
        "family_count": len(counts),
        "counts": dict(sorted(counts.items())),
        "shares": shares,
        "paraphrase_variant_counts": dict(sorted(variants.items())),
    }


def state_audit(records: list[dict[str, Any]], sidecars: list[dict[str, Any]], required: list[str]) -> dict:
    counts = Counter(state_type for record in records for state_type in record["state_types"])
    sidecar_ids = {sidecar["id"] for sidecar in sidecars}
    missing_types = sorted(set(required) - counts.keys())
    missing_sidecars = sorted(record["id"] for record in records if record["id"] not in sidecar_ids)
    return {
        "passed": not missing_types and not missing_sidecars,
        "required_state_types": required,
        "state_type_counts": dict(sorted(counts.items())),
        "missing_state_types": missing_types,
        "missing_sidecars": missing_sidecars,
        "replay_validated_records": len(records) - len(missing_sidecars),
    }


def distance_audit(records: list[dict[str, Any]], buckets: dict[str, dict[str, int]]) -> dict:
    counts = Counter(record["distance_bucket"] for record in records)
    violations = []
    errors: list[int] = []
    for record in records:
        definition = buckets[record["distance_bucket"]]
        distance = record["anchor_distance_tokens"]
        errors.append(abs(distance - definition["target"]))
        if not definition["minimum"] <= distance <= definition["maximum"]:
            violations.append({"id": record["id"], "distance": distance})
    return {
        "passed": not violations and not (set(buckets) - counts.keys()),
        "bucket_definitions": buckets,
        "counts": dict(sorted(counts.items())),
        "missing_buckets": sorted(set(buckets) - counts.keys()),
        "median_absolute_target_error": median(errors) if errors else None,
        "maximum_absolute_target_error": max(errors, default=None),
        "violations": violations,
    }


def quality_audit(
    records: list[dict[str, Any]],
    *,
    required_genres: list[str],
    maximum_document_tokens: int,
    minimum_words: int,
) -> dict:
    violations = []
    genres = Counter()
    for record in records:
        genres[record["genre"]] += 1
        words = len(normalized_words(record["text"]))
        reasons = []
        if words < minimum_words:
            reasons.append("too_short")
        if record["token_count"] > maximum_document_tokens:
            reasons.append("exceeds_model_context")
        if "\n\n" not in record["text"]:
            reasons.append("not_paragraph_fiction")
        if any(marker in record["text"] for marker in ("expected_final_state", "state_types", "<|")):
            reasons.append("metadata_visible_in_prose")
        if reasons:
            violations.append({"id": record["id"], "reasons": reasons})
    missing_genres = sorted(set(required_genres) - genres.keys())
    return {
        "passed": not violations and not missing_genres,
        "minimum_words": minimum_words,
        "maximum_document_tokens": maximum_document_tokens,
        "genre_counts": dict(sorted(genres.items())),
        "missing_genres": missing_genres,
        "violations": violations,
    }


def curriculum_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    token_counts = [record["token_count"] for record in records]
    word_counts = [record["word_count"] for record in records]
    dimensions = {
        "genre": lambda record: record["genre"],
        "difficulty": lambda record: str(record["difficulty"]),
        "distance": lambda record: record["distance_bucket"],
        "split": lambda record: record["split"],
    }
    tokens_by = {}
    for name, key in dimensions.items():
        counts = Counter()
        for record in records:
            counts[key(record)] += record["token_count"]
        tokens_by[name] = dict(sorted(counts.items()))
    state_tokens = Counter()
    for record in records:
        for state_type in record["state_types"]:
            state_tokens[state_type] += record["token_count"]
    tokens_by["state"] = dict(sorted(state_tokens.items()))
    template_counts = Counter(record["template_family"] for record in records)
    return {
        "story_count": len(records),
        "total_tokenizer_tokens": sum(token_counts),
        "word_count": sum(word_counts),
        "tokens_by": tokens_by,
        "story_tokens": {
            "median": median(token_counts) if token_counts else 0,
            "p95": percentile(token_counts, 0.95),
            "maximum": max(token_counts, default=0),
        },
        "story_words": {
            "median": median(word_counts) if word_counts else 0,
            "p95": percentile(word_counts, 0.95),
            "maximum": max(word_counts, default=0),
        },
        "template_concentration": {
            "counts": dict(sorted(template_counts.items())),
            "largest_share": max(template_counts.values(), default=0) / len(records) if records else 0,
        },
    }


def artifact_manifest(root: Path, relative_paths: list[Path], metadata: dict[str, Any]) -> dict:
    files = {}
    for relative in sorted(relative_paths, key=str):
        path = root / relative
        files[str(relative)] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {**metadata, "files": files}
