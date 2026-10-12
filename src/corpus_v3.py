"""Build and audit the separately versioned, rights-evidenced Corpus-v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from tokenizers import Tokenizer

from src.audit_corpus_v2 import dialogue_metrics, normalize_name, ocr_signals, parse_creator_string
from src.corpus_v1 import normalized_text_hash, write_jsonl
from src.corpus_v2 import GENRE_TERMS, near_duplicate_graph
from src.tokenize_corpus import encode_document
from src.train_tokenizer import sha256_file, write_json_atomic


CORPUS_VERSION = 3
PREPROCESSING_VERSION = 3
LEDGER_VERSION = 1
SPLITS = ("train", "validation", "test")
ALLOWED_RIGHTS = {"public_domain", "public_domain_us", "cc0", "cc_by"}
ALLOWED_GENRES = set(GENRE_TERMS)
DEFAULT_EVIDENCE_CUTOFF_DATE = "2026-10-12"
REQUIRED_GATES = {
    "acquisition_complete",
    "schema",
    "rights_evidence",
    "source_hashes",
    "metadata_provenance",
    "exact_duplicates",
    "near_duplicates",
    "story_boundaries",
    "quality",
    "author_concentration",
    "collection_concentration",
    "genre_distribution",
    "deterministic_review",
    "split_leakage",
    "token_accounting",
}
BOILERPLATE_RE = re.compile(r"project gutenberg|end of (?:the )?project gutenberg", re.I)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object JSONL row at {path}:{line_number}")
            records.append(value)
    return records


def verify_file(path: Path, expected_sha256: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"{label} hash changed: {actual}, expected {expected_sha256}")
    return actual


def load_ledger(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != LEDGER_VERSION or not isinstance(payload.get("artifacts"), dict):
        raise ValueError(f"Invalid Corpus-v3 acquisition ledger: {path}")
    return payload["artifacts"]


def save_ledger(path: Path, artifacts: dict[str, dict[str, Any]]) -> None:
    write_json_atomic(path, {"version": LEDGER_VERSION, "artifacts": artifacts})


def acquire_pinned_story(
    spec: dict[str, Any], *, cache_dir: Path, ledger: dict[str, dict[str, Any]], timeout: int
) -> tuple[str, dict[str, Any], bool]:
    """Acquire one immutable story, reusing bytes only after checksum verification."""
    artifact_id = str(spec["id"])
    expected = str(spec.get("source_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Source {artifact_id} requires a lowercase SHA-256")
    url = str(spec["source_url"])
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".txt"
    path = cache_dir / f"{hashlib.sha256(artifact_id.encode()).hexdigest()[:20]}{suffix}"
    entry = ledger.get(artifact_id, {})
    resumed = False
    if (
        entry.get("url") == url
        and entry.get("source_sha256") == expected
        and entry.get("status") == "complete"
        and path.is_file()
        and sha256_file(path) == expected
    ):
        data = path.read_bytes()
        resumed = True
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "FictionPulper/3.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
            etag = response.headers.get("ETag")
            modified = response.headers.get("Last-Modified")
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Source hash mismatch for {artifact_id}: {actual}, expected {expected}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        ledger[artifact_id] = {
            "id": artifact_id,
            "url": url,
            "source_sha256": expected,
            "byte_count": len(data),
            "cache_path": str(path),
            "etag": etag,
            "last_modified": modified,
            "status": "complete",
            "retrieved_at": utc_timestamp(),
        }
    return data.decode(spec.get("encoding", "utf-8-sig")), ledger.get(artifact_id, entry), resumed


def _rights_evidence(record: dict[str, Any]) -> dict[str, Any]:
    evidence = record.get("rights_evidence")
    if isinstance(evidence, dict):
        return evidence
    return {}


def canonicalize_record(record: dict[str, Any], tokenizer: Tokenizer) -> dict[str, Any]:
    text = str(record.get("text") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    source = str(record.get("source") or "")
    parsed_creator = parse_creator_string(record.get("author"))
    author = record.get("primary_author") or parsed_creator["primary_author"]
    creator_id = record.get("creator_id") or (
        f"{source}:{normalize_name(str(author))}" if author else "unknown"
    )
    collection_id = record.get("collection_id") or record.get("group_id") or record.get("source_id")
    source_hash = record.get("source_sha256") or record.get("provenance", {}).get("raw_sha256")
    canonical = {
        **record,
        "id": str(record.get("id") or ""),
        "source": source,
        "source_id": str(record.get("source_id") or source),
        "source_url": str(record.get("source_url") or ""),
        "title": str(record.get("title") or ""),
        "corpus_version": CORPUS_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "text": text,
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "words": len(text.split()),
        "rights_status": record.get("rights_status") or record.get("rights"),
        "rights_evidence": _rights_evidence(record),
        "source_sha256": source_hash,
        "collection_id": str(collection_id or ""),
        "creator_id": str(creator_id or "unknown"),
        "primary_author": author,
        "creator_evidence": record.get("creator_evidence") or {
            "method": "source_scoped_normalized_catalog_primary_creator",
            "primary_role": parsed_creator["primary_creator_role"],
            "identity_resolution": "alias_candidate_only",
        },
        "genres": sorted(set(record.get("broad_genres") or record.get("genres") or [])),
        "genre_evidence": record.get("genre_evidence") or (
            {
                "method": "inherited_source_catalog_mapping",
                "source_values": record["source_genres"],
                "explicitly_unclassified": False,
            }
            if record.get("broad_genres") and record.get("source_genres")
            else {}
        ),
        "publication_year": record.get("publication_year"),
        "publication_year_evidence": record.get("publication_year_evidence") or (
            {"status": "known", "source": record.get("publication_year_source")}
            if isinstance(record.get("publication_year"), int)
            else {"status": "unknown"}
        ),
        "boundary_evidence": record.get("boundary_evidence") or {},
    }
    canonical["tokenizer_v1_tokens"] = len(encode_document(tokenizer, canonical)[0])
    canonical["dialogue"] = dialogue_metrics(text)
    return canonical


def _valid_review_date(value: Any, cutoff_date: str = DEFAULT_EVIDENCE_CUTOFF_DATE) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        reviewed_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    cutoff = datetime.fromisoformat(cutoff_date).date()
    return reviewed_at.date() <= cutoff


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_schema(record: dict[str, Any]) -> bool:
    required_strings = ("id", "source", "source_url", "title", "collection_id", "source_sha256")
    return all(isinstance(record.get(field), str) and record[field].strip() for field in required_strings)


def _valid_rights_evidence(
    record: dict[str, Any], cutoff_date: str = DEFAULT_EVIDENCE_CUTOFF_DATE
) -> bool:
    evidence = record.get("rights_evidence")
    return (
        record.get("rights_status") in ALLOWED_RIGHTS
        and isinstance(evidence, dict)
        and all(_nonempty_string(evidence.get(key)) for key in ("basis", "evidence_url", "checked_by"))
        and _valid_review_date(evidence.get("checked_at"), cutoff_date)
    )


def _valid_creator_evidence(record: dict[str, Any]) -> bool:
    evidence = record.get("creator_evidence")
    return bool(
        record.get("primary_author")
        and record.get("creator_id") != "unknown"
        and isinstance(evidence, dict)
        and _nonempty_string(evidence.get("method"))
        and _nonempty_string(evidence.get("primary_role"))
    )


def _valid_genre_evidence(
    record: dict[str, Any], cutoff_date: str = DEFAULT_EVIDENCE_CUTOFF_DATE
) -> bool:
    genres = record.get("genres")
    evidence = record.get("genre_evidence")
    if not isinstance(genres, list) or any(genre not in ALLOWED_GENRES for genre in genres):
        return False
    if not isinstance(evidence, dict) or not _nonempty_string(evidence.get("method")):
        return False
    explicitly_unclassified = evidence.get("explicitly_unclassified")
    if explicitly_unclassified is not (not bool(genres)):
        return False
    source_values = evidence.get("source_values")
    if not genres:
        return True
    if not isinstance(source_values, list) or not source_values:
        return False
    normalized_values = [str(value).lower() for value in source_values]
    reviewed_mappings = evidence.get("reviewed_mappings", [])
    reviewed = (
        isinstance(reviewed_mappings, list)
        and _nonempty_string(evidence.get("reviewed_by"))
        and _valid_review_date(evidence.get("reviewed_at"), cutoff_date)
    )
    return all(
        any(term in value for term in GENRE_TERMS[genre] for value in normalized_values)
        or (reviewed and genre in reviewed_mappings)
        for genre in genres
    )


def _valid_year_evidence(record: dict[str, Any]) -> bool:
    evidence = record.get("publication_year_evidence")
    if not isinstance(evidence, dict) or evidence.get("status") not in {"known", "unknown"}:
        return False
    if evidence["status"] == "unknown":
        return record.get("publication_year") is None
    return isinstance(record.get("publication_year"), int) and _nonempty_string(
        evidence.get("source")
    )


def _valid_boundary(record: dict[str, Any]) -> bool:
    evidence = record.get("boundary_evidence")
    return isinstance(evidence, dict) and evidence.get("confidence") == "high" and _nonempty_string(
        evidence.get("method")
    )


def _quality_rejection(record: dict[str, Any], quality: dict[str, Any]) -> str | None:
    words = record.get("words")
    if not isinstance(words, int) or not int(quality["minimum_story_words"]) <= words <= int(
        quality["maximum_story_words"]
    ):
        return "story_length_out_of_range"
    signals = ocr_signals(record["text"])
    if signals["replacement_character"] or BOILERPLATE_RE.search(record["text"]):
        return "prohibited_text_signal"
    review_signals = set(quality.get("quarantine_signals", []))
    if any(signals.get(signal, 0) for signal in review_signals):
        return "quality_review_required"
    return None


def record_rejection(
    record: dict[str, Any], quality: dict[str, Any],
    evidence_cutoff_date: str = DEFAULT_EVIDENCE_CUTOFF_DATE,
) -> str | None:
    if not _valid_schema(record):
        return "invalid_schema"
    if record.get("rights_status") not in ALLOWED_RIGHTS:
        return "rights_status_not_allowed"
    if not _valid_rights_evidence(record, evidence_cutoff_date):
        return "incomplete_rights_evidence"
    if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("source_sha256", ""))):
        return "invalid_source_hash"
    if not _valid_creator_evidence(record):
        return "incomplete_creator_evidence"
    if not _valid_genre_evidence(record, evidence_cutoff_date):
        return "incomplete_genre_evidence"
    if not _valid_year_evidence(record):
        return "incomplete_publication_year_evidence"
    if not _valid_boundary(record):
        return "unverified_story_boundary"
    return _quality_rejection(record, quality)


def deterministic_review_sample(records: list[dict[str, Any]], seed: int, count: int) -> list[str]:
    if not records or count <= 0:
        return []
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for genre in record["genres"] or ["unclassified"]:
            strata[f"genre:{genre}"].append(record)
        dialogue = record["dialogue"]["dialogue_opening_paragraph_share"]
        strata["dialogue:high" if dialogue >= 0.2 else "dialogue:low"].append(record)
        strata[f"source:{record['source']}"].append(record)
    selected: set[str] = set()
    for label, candidates in sorted(strata.items()):
        selected.add(min(candidates, key=lambda item: hashlib.sha256(
            f"{seed}:{label}:{item['id']}".encode()
        ).hexdigest())["id"])
    if len(selected) > count:
        raise ValueError(f"review_sample_count={count} cannot cover {len(selected)} mandatory strata")
    for record in sorted(records, key=lambda item: hashlib.sha256(
        f"{seed}:fill:{item['id']}".encode()
    ).hexdigest()):
        if len(selected) >= count:
            break
        selected.add(record["id"])
    return sorted(selected)


def _union_groups(records: list[dict[str, Any]], clusters: list[dict[str, Any]]) -> dict[str, str]:
    parent = {record["id"]: record["id"] for record in records}

    def root(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(values: list[str]) -> None:
        roots = sorted({root(value) for value in values})
        for value in roots[1:]:
            parent[value] = roots[0]

    collections: defaultdict[str, list[str]] = defaultdict(list)
    for record in records:
        collections[record["collection_id"]].append(record["id"])
    for ids in collections.values():
        union(ids)
    known = set(parent)
    for cluster in clusters:
        union([record_id for record_id in cluster["record_ids"] if record_id in known])
    return {record_id: root(record_id) for record_id in parent}


def deterministic_splits(
    records: list[dict[str, Any]], clusters: list[dict[str, Any]], seed: int,
    ratios: dict[str, float],
) -> tuple[dict[str, str], dict[str, Any]]:
    if set(ratios) != set(SPLITS) or any(float(ratios[name]) <= 0 for name in SPLITS):
        raise ValueError("Split ratios must define positive train, validation, and test values")
    if abs(sum(float(ratios[name]) for name in SPLITS) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")
    group_for = _union_groups(records, clusters)
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[group_for[record["id"]]].append(record)
    total = sum(record["tokenizer_v1_tokens"] for record in records)
    targets = {split: total * float(ratios[split]) for split in SPLITS}
    totals = Counter()
    assignments: dict[str, str] = {}
    ordered = sorted(groups.items(), key=lambda item: hashlib.sha256(
        f"{seed}:{item[0]}".encode()
    ).hexdigest())
    for _, members in ordered:
        split = min(SPLITS, key=lambda name: (totals[name] / max(1.0, targets[name]), SPLITS.index(name)))
        for record in members:
            assignments[record["id"]] = split
            totals[split] += record["tokenizer_v1_tokens"]
    return assignments, audit_split_assignments(records, clusters, assignments)


def split_payload(
    assignments: dict[str, str], audit: dict[str, Any], *, seed: int
) -> dict[str, Any]:
    return {
        "version": CORPUS_VERSION,
        "seed": seed,
        "assignments": assignments,
        "audit": audit,
    }


def _sample_records_sha256(
    sample_ids: Iterable[str], records: list[dict[str, Any]]
) -> str:
    selected = set(sample_ids)
    return canonical_hash(sorted(
        (record for record in records if record["id"] in selected),
        key=lambda record: record["id"],
    ))


def _review_audit(
    sample_ids: list[str], records: list[dict[str, Any]], review: dict[str, Any],
    cutoff_date: str,
) -> dict[str, Any]:
    sample_review = review.get("sample_review")
    if not isinstance(sample_review, dict):
        sample_review = {}
    decisions = sample_review.get("decisions")
    if not isinstance(decisions, list):
        decisions = []
    malformed_entries = [index for index, item in enumerate(decisions) if not isinstance(item, dict)]
    decision_ids = [str(item.get("id") or "") for item in decisions if isinstance(item, dict)]
    duplicate_ids = sorted(record_id for record_id, count in Counter(decision_ids).items() if count > 1)
    valid_decisions = {
        str(item.get("id")): item
        for item in decisions
        if isinstance(item, dict)
        and _nonempty_string(item.get("id"))
        and item.get("decision") == "approve"
        and _nonempty_string(item.get("reviewer"))
        and _valid_review_date(item.get("reviewed_at"), cutoff_date)
        and _nonempty_string(item.get("notes"))
    }
    expected = set(sample_ids)
    approved = set(valid_decisions)
    actual_hash = _sample_records_sha256(sample_ids, records)
    expected_hash = sample_review.get("sample_records_sha256")
    missing = sorted(expected - approved)
    unexpected = sorted(approved - expected)
    invalid = sorted(set(decision_ids) - approved)
    passed = (
        expected_hash == actual_hash
        and not missing
        and not unexpected
        and not invalid
        and not duplicate_ids
        and not malformed_entries
    )
    return {
        "passed": passed,
        "sample_ids": sample_ids,
        "sample_records_sha256": actual_hash,
        "expected_sample_records_sha256": expected_hash,
        "approved_ids": sorted(approved),
        "missing_ids": missing,
        "unexpected_ids": unexpected,
        "invalid_decision_ids": invalid,
        "duplicate_decision_ids": duplicate_ids,
        "malformed_decision_entries": malformed_entries,
    }


def audit_split_assignments(
    records: list[dict[str, Any]], clusters: list[dict[str, Any]], assignments: dict[str, str]
) -> dict[str, Any]:
    record_ids = {record["id"] for record in records}
    assigned_ids = set(assignments)
    missing_ids = sorted(record_ids - assigned_ids)
    unexpected_ids = sorted(assigned_ids - record_ids)
    invalid_splits = sorted(
        record_id for record_id, split in assignments.items() if split not in SPLITS
    )
    collection_splits: defaultdict[str, set[str]] = defaultdict(set)
    totals: Counter[str] = Counter()
    for record in records:
        split = assignments.get(record["id"])
        if split in SPLITS:
            collection_splits[record["collection_id"]].add(split)
            tokens = record.get("tokenizer_v1_tokens")
            if isinstance(tokens, int) and not isinstance(tokens, bool):
                totals[split] += tokens
    collection_leaks = sorted(
        collection_id for collection_id, seen in collection_splits.items() if len(seen) > 1
    )
    cluster_leaks = sorted(
        str(cluster["cluster_id"])
        for cluster in clusters
        if len({assignments[item] for item in cluster["record_ids"] if item in assignments}) > 1
    )
    groups = _union_groups(records, clusters) if records else {}
    return {
        "passed": bool(records) and not (
            missing_ids or unexpected_ids or invalid_splits or collection_leaks or cluster_leaks
        ),
        "missing_assignment_ids": missing_ids,
        "unexpected_assignment_ids": unexpected_ids,
        "invalid_split_ids": invalid_splits,
        "collection_leaks": collection_leaks,
        "near_duplicate_cluster_leaks": cluster_leaks,
        "split_tokens": {split: totals[split] for split in SPLITS},
        "constraint_groups": len(set(groups.values())),
    }


def _review_exclusions(review: dict[str, Any], cutoff_date: str) -> list[str]:
    decisions = review.get("exclusion_decisions", [])
    if not isinstance(decisions, list):
        raise ValueError("Review exclusion_decisions must be a list")
    exclusions = []
    for item in decisions:
        if not isinstance(item, dict) or not all(
            (
                _nonempty_string(item.get("id")),
                item.get("decision") == "exclude",
                _nonempty_string(item.get("reviewer")),
                _valid_review_date(item.get("reviewed_at"), cutoff_date),
                _nonempty_string(item.get("rationale")),
            )
        ):
            raise ValueError("Every review exclusion requires an ID, attributable decision, date, and rationale")
        exclusions.append(str(item["id"]))
    if len(exclusions) != len(set(exclusions)):
        raise ValueError("Review exclusion IDs must be unique")
    return exclusions


def _concentration(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    tokens: Counter[str] = Counter()
    stories: Counter[str] = Counter()
    for record in records:
        key = str(record.get(field) or "unknown")
        stories[key] += 1
        tokens[key] += record["tokenizer_v1_tokens"]
    total_tokens = sum(tokens.values())
    total_stories = sum(stories.values())
    ordered = sorted(tokens, key=lambda key: (-tokens[key], key))
    if not ordered:
        return {
            "unique_count": 0, "largest_token_share": 1.0,
            "largest_story_share": 1.0, "top_10_token_share": 1.0,
            "top_20_token_share": 1.0,
        }
    return {
        "unique_count": len(ordered),
        "largest_token_share": tokens[ordered[0]] / total_tokens,
        "largest_story_share": max(stories.values()) / total_stories,
        "top_10_token_share": sum(tokens[key] for key in ordered[:10]) / total_tokens,
        "top_20_token_share": sum(tokens[key] for key in ordered[:20]) / total_tokens,
    }


def audit_records(
    records: list[dict[str, Any]], *, clusters: list[dict[str, Any]], assignments: dict[str, str],
    review: dict[str, Any], config: dict[str, Any], acquisition_failures: int = 0,
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    ids = [record["id"] for record in records]
    exact = [normalized_text_hash(record["text"]) for record in records]
    sample_ids = deterministic_review_sample(
        records, int(config["seed"]), int(config["audit"]["review_sample_count"])
    )
    unresolved_clusters = [
        cluster for cluster in clusters
        if sum(record_id in assignments for record_id in cluster["record_ids"]) > 1
    ]
    author = _concentration(records, "creator_id")
    collection = _concentration(records, "collection_id")
    limits = config["audit"]["concentration_limits"]
    evidence_cutoff_date = str(config["audit"]["evidence_cutoff_date"])
    genre_tokens: Counter[str] = Counter()
    valid_token_counts = all(
        isinstance(record.get("tokenizer_v1_tokens"), int)
        and not isinstance(record.get("tokenizer_v1_tokens"), bool)
        and record["tokenizer_v1_tokens"] > 0
        for record in records
    )
    total_tokens = sum(
        record["tokenizer_v1_tokens"] for record in records
        if isinstance(record.get("tokenizer_v1_tokens"), int)
        and not isinstance(record.get("tokenizer_v1_tokens"), bool)
    )
    for record in records:
        genres = record.get("genres", [])
        valid_genres = genres if isinstance(genres, list) and all(
            genre in ALLOWED_GENRES for genre in genres
        ) else []
        for genre in valid_genres or ["unclassified"]:
            genre_tokens[genre] += record["tokenizer_v1_tokens"]
    unclassified_share = genre_tokens["unclassified"] / max(1, total_tokens)
    split_audit = audit_split_assignments(records, clusters, assignments)
    recomputed_counts = {
        record["id"]: len(encode_document(tokenizer, record)[0]) for record in records
    }
    token_count_mismatches = sorted(
        record["id"] for record in records
        if record.get("tokenizer_v1_tokens") != recomputed_counts[record["id"]]
    )
    recomputed_tokens = sum(recomputed_counts.values())
    assigned_tokens = sum(split_audit["split_tokens"].values())
    gates = {
        "acquisition_complete": {
            "passed": acquisition_failures == 0,
            "failed_sources": acquisition_failures,
        },
        "schema": {"passed": len(ids) == len(set(ids)) and bool(records) and all(_valid_schema(record) for record in records)},
        "rights_evidence": {"passed": bool(records) and all(_valid_rights_evidence(record, evidence_cutoff_date) for record in records)},
        "source_hashes": {"passed": bool(records) and all(re.fullmatch(r"[0-9a-f]{64}", str(record.get("source_sha256") or "")) for record in records)},
        "metadata_provenance": {"passed": bool(records) and all(
            _valid_creator_evidence(record)
            and _valid_genre_evidence(record, evidence_cutoff_date)
            and _valid_year_evidence(record)
            for record in records
        )},
        "exact_duplicates": {"passed": len(exact) == len(set(exact))},
        "near_duplicates": {"passed": not unresolved_clusters, "unresolved_clusters": len(unresolved_clusters)},
        "story_boundaries": {"passed": bool(records) and all(_valid_boundary(record) for record in records)},
        "quality": {"passed": bool(records) and all(_quality_rejection(record, config["quality"]) is None for record in records)},
        "author_concentration": {"passed": author["largest_token_share"] <= float(limits["maximum_author_token_share"]), "metrics": author},
        "collection_concentration": {"passed": collection["largest_token_share"] <= float(limits["maximum_collection_token_share"]), "metrics": collection},
        "genre_distribution": {"passed": bool(records) and unclassified_share <= float(config["audit"]["maximum_unclassified_token_share"]), "unclassified_token_share": unclassified_share, "token_counts": dict(sorted(genre_tokens.items()))},
        "deterministic_review": _review_audit(
            sample_ids, records, review, evidence_cutoff_date
        ),
        "split_leakage": split_audit,
        "token_accounting": {
            "passed": bool(records)
            and valid_token_counts
            and split_audit["passed"]
            and not token_count_mismatches
            and total_tokens == assigned_tokens == recomputed_tokens,
            "exact_document_tokens": total_tokens,
            "assigned_document_tokens": assigned_tokens,
            "recomputed_document_tokens": recomputed_tokens,
            "mismatched_record_ids": token_count_mismatches,
        },
    }
    missing = sorted(REQUIRED_GATES - gates.keys())
    failed = sorted(name for name in REQUIRED_GATES if not gates[name]["passed"])
    return {"gates": gates, "missing_required_gates": missing, "failed_required_gates": failed}


def _select_diverse(records: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    """Deterministically interleave creators, favoring compact and dialogue-rich fiction."""
    by_author: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_author[record["creator_id"]].append(record)
    for values in by_author.values():
        values.sort(key=lambda record: (
            -float(record["dialogue"]["dialogue_opening_paragraph_share"]),
            abs(record["words"] - 3000), record["collection_id"], record["id"],
        ))
    selected: list[dict[str, Any]] = []
    total = 0
    authors = sorted(by_author)
    while authors and total < target:
        remaining = []
        for author in authors:
            if not by_author[author] or total >= target:
                continue
            record = by_author[author].pop(0)
            selected.append(record)
            total += record["tokenizer_v1_tokens"]
            if by_author[author]:
                remaining.append(author)
        authors = remaining
    return selected


def reset_stage_reports(stage_dir: Path) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    for path in stage_dir.glob("*.json"):
        path.unlink()


def stage_report_hashes(stage_dir: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(stage_dir.glob("*.json"))
    }


def build_corpus_v3(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("version") != CORPUS_VERSION:
        raise ValueError("Corpus-v3 config must declare version: 3")
    output_dir = Path(config["output_dir"])
    if output_dir.exists() and (output_dir / "seal.json").exists():
        raise RuntimeError(f"Refusing to overwrite sealed Corpus-v3 output: {output_dir}")
    tokenizer_spec = config["tokenizer"]
    tokenizer_path = Path(tokenizer_spec["path"])
    verify_file(tokenizer_path, tokenizer_spec["expected_sha256"], "tokenizer-v1")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    review_path = Path(config["review"]["path"])
    verify_file(review_path, config["review"]["expected_sha256"], "Corpus-v3 review decisions")
    review = yaml.safe_load(review_path.read_text(encoding="utf-8")) or {}
    if review.get("version") != 1:
        raise ValueError("Corpus-v3 review decisions must declare version: 1")
    evidence_cutoff_date = str(config["audit"]["evidence_cutoff_date"])
    exclusions = _review_exclusions(review, evidence_cutoff_date)

    candidates: list[dict[str, Any]] = []
    source_inputs: list[dict[str, Any]] = []
    base_spec = config.get("base")
    if base_spec:
        base_path = Path(base_spec["corpus_path"])
        digest = verify_file(base_path, base_spec["expected_sha256"], "sealed Corpus-v2 base")
        source_inputs.append({"kind": "base", "path": str(base_path), "sha256": digest})
        candidates.extend(load_jsonl(base_path))

    state_dir = output_dir / "state"
    ledger_path = state_dir / "acquisition-ledger.json"
    ledger = load_ledger(ledger_path)
    resumed = acquired = acquisition_failures = 0
    acquisition_failure_rows: list[dict[str, str]] = []
    for inventory_spec in config.get("source_inventories", []):
        inventory_path = Path(inventory_spec["path"])
        digest = verify_file(inventory_path, inventory_spec["expected_sha256"], "source inventory")
        source_inputs.append({"kind": "inventory", "path": str(inventory_path), "sha256": digest})
        for spec in load_jsonl(inventory_path):
            try:
                text, _, was_resumed = acquire_pinned_story(
                    spec, cache_dir=Path(config["acquisition"]["cache_dir"]), ledger=ledger,
                    timeout=int(config["acquisition"]["timeout_seconds"]),
                )
            except (OSError, UnicodeError, urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as error:
                acquisition_failures += 1
                acquisition_failure_rows.append({
                    "id": str(spec.get("id") or "unknown"),
                    "source_url": str(spec.get("source_url") or ""),
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                continue
            spec["text"] = text
            candidates.append(spec)
            resumed += int(was_resumed)
            acquired += int(not was_resumed)
            save_ledger(ledger_path, ledger)

    canonical = [canonicalize_record(record, tokenizer) for record in candidates]
    rejection_rows = []
    eligible = []
    seen_hashes: dict[str, str] = {}
    excluded = set(exclusions)
    encountered_ids = {record["id"] for record in canonical}
    unknown_exclusions = sorted(excluded - encountered_ids)
    if unknown_exclusions:
        raise ValueError(f"Review exclusions do not exist in candidates: {unknown_exclusions[:10]}")
    for record in sorted(canonical, key=lambda item: (item["source"], item["collection_id"], item["id"])):
        reason = "reviewed_exclusion" if record["id"] in excluded else record_rejection(
            record, config["quality"], evidence_cutoff_date
        )
        digest = normalized_text_hash(record["text"])
        if reason is None and digest in seen_hashes:
            reason = "exact_duplicate"
        if reason:
            rejection_rows.append({"id": record["id"], "reason": reason, "retained_id": seen_hashes.get(digest)})
        else:
            seen_hashes[digest] = record["id"]
            eligible.append(record)

    selected = _select_diverse(eligible, int(config["selection"]["target_tokens"]))
    edges, clusters = near_duplicate_graph(
        canonical,
        max_hamming=int(config["deduplication"]["simhash_max_hamming"]),
        minimum_jaccard=float(config["deduplication"]["minimum_jaccard"]),
        minimum_length_ratio=float(config["deduplication"]["minimum_length_ratio"]),
    )
    assignments, split_audit = deterministic_splits(
        selected, clusters, int(config["seed"]), config["splits"]
    )
    audit = audit_records(
        selected, clusters=clusters, assignments=assignments, review=review, config=config,
        acquisition_failures=acquisition_failures, tokenizer=tokenizer,
    )
    total_tokens = sum(record["tokenizer_v1_tokens"] for record in selected)
    target = int(config["selection"]["target_tokens"])
    target_reached = total_tokens >= target
    all_gates_pass = not audit["failed_required_gates"] and not audit["missing_required_gates"]
    status = "freeze_ready" if all_gates_pass else "stopped_audit_failures"
    stop_reason = "acquisition_incomplete" if acquisition_failures else (
        "target_reached" if target_reached and all_gates_pass else (
        "target_reached_with_audit_failures" if target_reached else (
        "compliant_supply_exhausted"
        if all_gates_pass
        else "candidate_supply_exhausted_with_audit_failures"
        )
        )
    )
    if not target_reached and all_gates_pass:
        status = "freeze_ready_below_feasibility_target"

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "corpus.jsonl", selected)
    write_jsonl(output_dir / "rejections.jsonl", rejection_rows)
    write_jsonl(output_dir / "acquisition-failures.jsonl", acquisition_failure_rows)
    write_jsonl(output_dir / "source-artifacts.jsonl", [ledger[key] for key in sorted(ledger)])
    write_jsonl(output_dir / "near-duplicate-candidates.jsonl", edges)
    write_jsonl(output_dir / "near-duplicate-clusters.jsonl", clusters)
    write_json_atomic(
        output_dir / "splits.json",
        split_payload(assignments, split_audit, seed=int(config["seed"])),
    )
    stage_dir = output_dir / "stages"
    reset_stage_reports(stage_dir)
    stages = []
    running = 0
    stage_values = sorted(int(value) for value in config["selection"]["stage_tokens"])
    prior = 0
    for record in selected:
        running += record["tokenizer_v1_tokens"]
        while stage_values and running >= stage_values[0]:
            threshold = stage_values.pop(0)
            report = {
                "threshold_tokens": threshold, "actual_tokens": running,
                "delta_tokens": running - prior, "last_record_id": record["id"],
                "prefix_record_count": selected.index(record) + 1,
                "prefix_canonical_sha256": canonical_hash(selected[: selected.index(record) + 1]),
                "required_gates": sorted(REQUIRED_GATES), "final_gate_status": audit["gates"],
            }
            write_json_atomic(stage_dir / f"{threshold}.json", report)
            stages.append(threshold)
            prior = running
    if not target_reached:
        write_json_atomic(stage_dir / "attainable.json", {
            "status": stop_reason,
            "target_tokens": target,
            "exact_actual_token_count": total_tokens,
            "story_count": len(selected),
            "required_gates": sorted(REQUIRED_GATES),
            "final_gate_status": audit["gates"],
            "failed_required_gates": audit["failed_required_gates"],
        })
    stats = {
        "corpus_version": CORPUS_VERSION,
        "status": status,
        "stop_reason": stop_reason,
        "target_is_feasibility_goal": True,
        "target_tokens": target,
        "target_reached": target_reached,
        "exact_actual_token_count": total_tokens,
        "stories": len(selected),
        "eligible_stories": len(eligible),
        "candidate_stories": len(canonical),
        "rejected_stories": len(rejection_rows),
        "source_artifacts_acquired": acquired,
        "source_artifacts_resumed": resumed,
        "acquisition_failures": acquisition_failures,
        "completed_stage_thresholds": stages,
        "missing_stage_thresholds": stage_values,
        "freeze_ready": all_gates_pass,
        "failed_required_gates": audit["failed_required_gates"],
        "corpus_sha256": sha256_file(output_dir / "corpus.jsonl"),
    }
    write_json_atomic(output_dir / "audits.json", audit)
    write_json_atomic(output_dir / "stats.json", stats)
    manifest = {
        "corpus_version": CORPUS_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "builder_sha256": sha256_file(Path(__file__)),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "review_sha256": sha256_file(review_path),
        "source_inputs": source_inputs,
        "source_artifacts_sha256": sha256_file(output_dir / "source-artifacts.jsonl"),
        "corpus_sha256": stats["corpus_sha256"],
        "split_sha256": sha256_file(output_dir / "splits.json"),
        "audits_sha256": sha256_file(output_dir / "audits.json"),
        "stats_sha256": sha256_file(output_dir / "stats.json"),
        "rejections_sha256": sha256_file(output_dir / "rejections.jsonl"),
        "acquisition_failures_sha256": sha256_file(output_dir / "acquisition-failures.jsonl"),
        "near_duplicate_candidates_sha256": sha256_file(output_dir / "near-duplicate-candidates.jsonl"),
        "near_duplicate_clusters_sha256": sha256_file(output_dir / "near-duplicate-clusters.jsonl"),
        "stage_reports_sha256": stage_report_hashes(stage_dir),
        "canonical_content_sha256": canonical_hash(selected),
        "determinism_note": "Canonical rows, selection, audits, splits, and stage reports are deterministic; retrieval timestamps and HTTP validators are not.",
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    if all_gates_pass:
        seal = {
            "corpus_id": "fictionpulper-corpus-v3",
            "status": "sealed",
            "corpus_version": CORPUS_VERSION,
            "story_count": len(selected),
            "exact_document_tokens": total_tokens,
            "target_tokens": target,
            "target_reached": target_reached,
            "stop_reason": stop_reason,
            "failed_required_gates": [],
            "unresolved_near_duplicate_clusters": 0,
            "cross_split_leakage": 0,
            "corpus_sha256": stats["corpus_sha256"],
            "split_sha256": sha256_file(output_dir / "splits.json"),
            "manifest_sha256": sha256_file(output_dir / "manifest.json"),
            "audits_sha256": sha256_file(output_dir / "audits.json"),
        }
        write_json_atomic(output_dir / "seal.json", seal)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/corpus-v3.yaml"))
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_corpus_v3(parse_args().config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
