"""Build Corpus-v2 from the immutable resolved Corpus-v1 and Gutenberg."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from tokenizers import Tokenizer

from src.corpus_v1 import (
    CATALOG_URL,
    TEXT_URL,
    Extraction,
    author_is_safely_public_domain,
    catalog_candidates,
    clean_gutenberg_text,
    extract_collection,
    normalize_heading,
    normalized_text_hash,
    review_sample,
    shingle_jaccard,
    simhash64,
    write_jsonl,
)
from src.tokenize_corpus import encode_document
from src.train_tokenizer import load_corpus, sha256_file, write_json_atomic


PREPROCESSING_VERSION = 2
MIN_STORY_WORDS = 500
MAX_STORY_WORDS = 20_000
ALLOWED_RIGHTS = {"public_domain", "public_domain_us"}
STRUCTURAL_TITLE = re.compile(
    r"^(?:(?:chapter|part|book|volume|section)\s+(?:[ivxlcdm]+|\d+)(?:\b.*)?|[ivxlcdm]+|\d+)$",
    re.IGNORECASE,
)
STRUCTURAL_LINE = re.compile(
    r"^\s*(?:chapter|part|book|volume)\s+(?:[ivxlcdm]+|\d+)\s*[.:~-]*\s*$",
    re.IGNORECASE,
)
GENRE_TERMS: dict[str, tuple[str, ...]] = {
    "crime": ("crime", "criminal", "murder", "thieves"),
    "mystery": ("mystery", "detective", "suspense"),
    "horror": ("horror", "ghost", "gothic", "supernatural"),
    "science_fiction": ("science fiction", "interplanetary", "space travel"),
    "western": ("western", "cowboy", "frontier"),
    "adventure": ("adventure", "sea stories", "pirates", "survival"),
    "fantasy": ("fantasy", "fairy tale", "folklore", "magic"),
    "weird": ("weird fiction", "occult", "eldritch"),
}


@dataclass(frozen=True)
class SourceArtifact:
    source_id: str
    url: str
    path: str
    sha256: str
    byte_count: int
    retrieval_method: str
    retrieved_at: str


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def broad_genres(explicit_metadata: Iterable[str]) -> list[str]:
    """Map only supplied catalog/record metadata, never story text, to broad genres."""
    metadata = " | ".join(str(value) for value in explicit_metadata).casefold()
    return [genre for genre, terms in GENRE_TERMS.items() if any(term in metadata for term in terms)]


def is_structural_title(title: str) -> bool:
    return STRUCTURAL_TITLE.fullmatch(normalize_heading(title)) is not None


def chapter_heading_count(text: str) -> int:
    return sum(STRUCTURAL_LINE.fullmatch(line) is not None for line in text.splitlines())


def classify_story_rejection(
    extraction: Extraction,
    *,
    minimum_words: int = MIN_STORY_WORDS,
    maximum_words: int = MAX_STORY_WORDS,
) -> str | None:
    if is_structural_title(extraction.title):
        return "chapter_or_structural_title"
    words = extraction.text.split()
    if len(words) < minimum_words:
        return "too_short_or_front_matter"
    if len(words) > maximum_words:
        return "too_long_possible_novel"
    if chapter_heading_count(extraction.text) >= 3:
        return "probable_novel_chapter_contamination"
    paragraphs = [part for part in re.split(r"\n\s*\n", extraction.text) if part.strip()]
    if len(paragraphs) < 3:
        return "insufficient_prose_paragraphs"
    nonspace = [character for character in extraction.text if not character.isspace()]
    if nonspace and sum(character.isalpha() for character in nonspace) / len(nonspace) < 0.65:
        return "damaged_ocr_or_nonprose"
    if extraction.text.count("\ufffd") / max(1, len(extraction.text)) > 0.001:
        return "damaged_ocr_replacement_characters"
    suspicious = re.findall(r"\b\w*[|{}<>]\w*\b|(?:\S*\d){3,}\S*", extraction.text)
    if len(suspicious) > max(8, len(words) // 100):
        return "damaged_ocr_suspicious_tokens"
    lines = [line.strip() for line in extraction.text.splitlines() if line.strip()]
    short_ratio = sum(len(line.split()) <= 8 for line in lines) / max(1, len(lines))
    if len(lines) >= 20 and short_ratio > 0.72:
        return "probable_poetry"
    return None


def extract_collection_v2(
    text: str,
    *,
    minimum_matches: int = 3,
    high_match_ratio: float = 0.8,
) -> tuple[list[Extraction], dict[str, Any]]:
    extractions, evidence = extract_collection(text)
    matched = int(evidence.get("matched_heading_count", 0))
    ratio = float(evidence.get("heading_match_ratio", 0.0))
    if evidence.get("confidence") == "high" and (
        matched < minimum_matches or ratio < high_match_ratio
    ):
        evidence = {**evidence, "confidence": "medium", "v2_reason": "strict_toc_threshold"}
        extractions = [
            Extraction(**{**asdict(extraction), "confidence": "medium"})
            for extraction in extractions
        ]
    return extractions, evidence


def canonical_base_record(
    record: dict[str, Any], existing_assignment: str | None
) -> dict[str, Any]:
    canonical = dict(record)
    source_genres = list(record.get("source_genres", record.get("genres", [])))
    canonical["source_genres"] = source_genres
    canonical["genres"] = broad_genres(source_genres)
    canonical["broad_genres"] = canonical["genres"]
    canonical["rights_status"] = record["rights"]
    canonical["preprocessing_version"] = PREPROCESSING_VERSION
    canonical["source_url"] = (
        TEXT_URL.format(book_id=record["source_id"].removeprefix("pg-"))
        if record.get("source") == "project_gutenberg"
        and str(record.get("source_id", "")).startswith("pg-")
        else record.get("source_url")
    )
    canonical["group_id"] = record.get("group_id") or record.get("source_id") or record["id"]
    canonical["existing_assignment"] = record.get("existing_assignment") or existing_assignment
    canonical["provenance"] = {
        **record.get("provenance", {}),
        "derived_from": "corpus-v1/data10m-v1",
        "source_record_id": record["id"],
        "base_record": True,
    }
    canonical["boundary_evidence"] = {
        "method": record["boundary_method"],
        "confidence": record["extraction_confidence"],
        "inherited_from_locked_base": True,
    }
    return canonical


def make_story_record(
    extraction: Extraction,
    collection: dict[str, str],
    artifact: SourceArtifact,
    collection_evidence: dict[str, Any],
) -> dict[str, Any]:
    book_id = collection["Text#"]
    text_hash = hashlib.sha256(extraction.text.encode("utf-8")).hexdigest()
    stable = hashlib.sha256(
        f"{book_id}\0{normalize_heading(extraction.title)}\0{text_hash}".encode()
    ).hexdigest()[:12]
    explicit = [
        item.strip()
        for field in ("Subjects", "Bookshelves")
        for item in collection.get(field, "").split(";")
        if item.strip()
    ]
    return {
        "id": f"pg-{book_id}-{stable}",
        "source": "project_gutenberg",
        "source_id": f"pg-{book_id}",
        "source_collection": " ".join(collection["Title"].split()),
        "source_url": artifact.url,
        "group_id": f"pg-{book_id}",
        "existing_assignment": None,
        "title": extraction.title,
        "author": collection.get("Authors") or None,
        "publication_year": None,
        "source_genres": explicit,
        "genres": broad_genres(explicit),
        "broad_genres": broad_genres(explicit),
        "rights": "public_domain_us",
        "rights_status": "public_domain_us",
        "preprocessing_version": PREPROCESSING_VERSION,
        "extraction_confidence": "high",
        "boundary_method": extraction.boundary_method,
        "boundary_evidence": {
            "method": extraction.boundary_method,
            "confidence": extraction.confidence,
            "story_start_line": extraction.start_line,
            "story_end_line": extraction.end_line,
            "toc_start_line": collection_evidence.get("toc_start_line"),
            "toc_title_count": collection_evidence.get("toc_title_count"),
            "matched_heading_count": collection_evidence.get("matched_heading_count"),
            "heading_match_ratio": collection_evidence.get("heading_match_ratio"),
        },
        "source_start_heading": extraction.title,
        "text_hash": text_hash,
        "words": len(extraction.text.split()),
        "provenance": {
            "catalog_record_id": book_id,
            "catalog_ebook_issued": collection.get("Issued") or None,
            "catalog_locc": collection.get("LoCC") or None,
            "source_url": artifact.url,
            "raw_sha256": artifact.sha256,
            "rights_basis": "Project Gutenberg public-domain-in-the-United-States distribution",
        },
        "text": extraction.text,
    }


def validate_canonical_records(
    records: list[dict[str, Any]], *, expected_base_ids: set[str] | None = None
) -> None:
    if not records:
        raise ValueError("Corpus must contain at least one record")
    ids: set[str] = set()
    hashes: set[str] = set()
    for index, record in enumerate(records, 1):
        for field in (
            "id", "source", "source_id", "group_id", "title", "rights",
            "rights_status", "extraction_confidence", "boundary_method", "text_hash", "text",
        ):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Record {index} has invalid {field!r}")
        if record["id"] in ids:
            raise ValueError(f"Duplicate corpus id: {record['id']}")
        ids.add(record["id"])
        if record["rights_status"] not in ALLOWED_RIGHTS or record["rights"] not in ALLOWED_RIGHTS:
            raise ValueError(f"Record {record['id']} is not confirmed public domain")
        if record["extraction_confidence"] != "high":
            raise ValueError(f"Non-high-confidence record admitted: {record['id']}")
        if not isinstance(record.get("genres"), list) or not isinstance(record.get("broad_genres"), list):
            raise ValueError(f"Record {record['id']} has invalid genres")
        if not isinstance(record.get("provenance"), dict) or not record["provenance"]:
            raise ValueError(f"Record {record['id']} has invalid provenance")
        if not isinstance(record.get("boundary_evidence"), dict) or not record["boundary_evidence"]:
            raise ValueError(f"Record {record['id']} has invalid boundary evidence")
        assignment = record.get("existing_assignment")
        if assignment is not None and assignment not in {"train", "validation", "test"}:
            raise ValueError(f"Record {record['id']} has invalid existing assignment")
        if not isinstance(record.get("words"), int) or record["words"] <= 0:
            raise ValueError(f"Record {record['id']} has invalid word count")
        if record.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError(f"Record {record['id']} has invalid preprocessing version")
        if not isinstance(record.get("tokenizer_v1_tokens"), int) or record["tokenizer_v1_tokens"] <= 0:
            raise ValueError(f"Record {record['id']} has invalid tokenizer-v1 token count")
        if record.get("source_url") is not None and not isinstance(record["source_url"], str):
            raise ValueError(f"Record {record['id']} has invalid source URL")
        actual_hash = hashlib.sha256(record["text"].encode()).hexdigest()
        if actual_hash != record["text_hash"]:
            raise ValueError(f"Record {record['id']} has an incorrect text hash")
        normalized_hash = normalized_text_hash(record["text"])
        if normalized_hash in hashes:
            raise ValueError(f"Normalized-text duplicate admitted: {record['id']}")
        hashes.add(normalized_hash)
    if expected_base_ids is not None and not expected_base_ids.issubset(ids):
        missing = sorted(expected_base_ids - ids)
        raise ValueError(f"Corpus does not preserve base IDs: {missing[:5]}")


def add_if_not_exact_duplicate(
    record: dict[str, Any],
    accepted: list[dict[str, Any]],
    exact_hashes: dict[str, dict[str, Any]],
    report: list[dict[str, Any]],
) -> bool:
    digest = normalized_text_hash(record["text"])
    canonical = exact_hashes.get(digest)
    if canonical is not None:
        report.append(
            {
                "normalized_text_hash": digest,
                "retained_id": canonical["id"],
                "rejected_id": record["id"],
                "retained_is_base": bool(canonical.get("provenance", {}).get("base_record")),
                "retained_title": canonical["title"],
                "rejected_title": record["title"],
                "retained_author": canonical.get("author"),
                "rejected_author": record.get("author"),
                "retained_source_id": canonical["source_id"],
                "rejected_source_id": record["source_id"],
                "reason": "exact_normalized_text_match_base_preferred"
                if canonical.get("provenance", {}).get("base_record")
                else "exact_normalized_text_match_first_seen_preferred",
            }
        )
        return False
    exact_hashes[digest] = record
    accepted.append(record)
    return True


def _opening_key(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.casefold())[:40]
    return " ".join(words)


def near_duplicate_graph(
    records: list[dict[str, Any]],
    *,
    max_hamming: int = 10,
    minimum_jaccard: float = 0.72,
    minimum_length_ratio: float = 0.75,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return uncertain review edges and transitive connected components."""
    hashes = [simhash64(record["text"]) for record in records]
    buckets: dict[tuple[str, Any], list[int]] = defaultdict(list)
    candidate_pairs: set[tuple[int, int]] = set()
    for index, (record, digest) in enumerate(zip(records, hashes, strict=True)):
        keys: list[tuple[str, Any]] = [
            ("simhash", (band, (digest >> (band * 16)) & 0xFFFF)) for band in range(4)
        ]
        title_author = (
            normalize_heading(record["title"]),
            normalize_heading(record.get("author") or ""),
        )
        keys.extend((("title_author", title_author), ("opening", _opening_key(record["text"]))))
        for key in keys:
            for previous in buckets[key]:
                candidate_pairs.add((previous, index))
            buckets[key].append(index)

    parent = list(range(len(records)))

    def root(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    edges = []
    for left_index, right_index in sorted(candidate_pairs):
        left, right = records[left_index], records[right_index]
        length_ratio = min(left["words"], right["words"]) / max(left["words"], right["words"])
        title_match = normalize_heading(left["title"]) == normalize_heading(right["title"])
        author_match = bool(left.get("author") and right.get("author")) and normalize_heading(
            left["author"]
        ) == normalize_heading(right["author"])
        opening_match = _opening_key(left["text"]) == _opening_key(right["text"])
        hamming = (hashes[left_index] ^ hashes[right_index]).bit_count()
        if length_ratio < minimum_length_ratio or hamming > max_hamming:
            if not (title_match and author_match and opening_match and length_ratio >= 0.6):
                continue
        jaccard = shingle_jaccard(left["text"], right["text"])
        if jaccard < minimum_jaccard and not (
            title_match and author_match and opening_match and jaccard >= 0.45
        ):
            continue
        union(left_index, right_index)
        edges.append(
            {
                "left_id": left["id"], "right_id": right["id"],
                "left_title": left["title"], "right_title": right["title"],
                "left_author": left.get("author"), "right_author": right.get("author"),
                "left_source_id": left["source_id"], "right_source_id": right["source_id"],
                "left_opening": left["text"][:500], "right_opening": right["text"][:500],
                "simhash_hamming_distance": hamming,
                "word_5gram_jaccard": jaccard,
                "length_ratio": length_ratio,
                "normalized_title_match": title_match,
                "normalized_author_match": author_match,
                "normalized_opening_match": opening_match,
                "action": "human_review_required",
            }
        )
    members: dict[int, list[str]] = defaultdict(list)
    for index, record in enumerate(records):
        members[root(index)].append(record["id"])
    clusters = [
        {"cluster_id": f"near-{number:04d}", "record_ids": sorted(ids), "size": len(ids)}
        for number, ids in enumerate(
            sorted((ids for ids in members.values() if len(ids) > 1), key=lambda ids: sorted(ids)), 1
        )
    ]
    id_to_cluster = {record_id: cluster["cluster_id"] for cluster in clusters for record_id in cluster["record_ids"]}
    for edge in edges:
        edge["cluster_id"] = id_to_cluster[edge["left_id"]]
    return edges, clusters


def exact_document_tokens(tokenizer: Tokenizer, record: dict[str, Any]) -> int:
    return len(encode_document(tokenizer, record)[0])


def target_reached(token_count: int, target_tokens: int) -> bool:
    return token_count >= target_tokens


def source_concentration_metrics(
    records: list[dict[str, Any]], token_counts: list[int] | None = None
) -> dict[str, Any]:
    if token_counts is not None and len(token_counts) != len(records):
        raise ValueError("Token counts must align with corpus records")

    def distribution(field: str) -> dict[str, Any]:
        counts = Counter(str(record.get(field) or "unknown") for record in records)
        total = len(records)
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        shares = [count / total for _, count in ordered]
        result = {
            "unique_count": len(counts),
            "largest_count": ordered[0][1],
            "largest_share": shares[0],
            "top_10_share": sum(shares[:10]),
            "herfindahl_index": sum(share * share for share in shares),
            "top_10": [{"value": value, "stories": count} for value, count in ordered[:10]],
        }
        if token_counts is not None:
            tokens_by_value: Counter[str] = Counter()
            for record, count in zip(records, token_counts, strict=True):
                tokens_by_value[str(record.get(field) or "unknown")] += count
            total_tokens = sum(token_counts)
            ordered_tokens = sorted(tokens_by_value.items(), key=lambda item: (-item[1], item[0]))
            result["largest_token_share"] = ordered_tokens[0][1] / total_tokens
            result["top_10_token_share"] = sum(
                count for _, count in ordered_tokens[:10]
            ) / total_tokens
            result["top_10_by_tokens"] = [
                {"value": value, "tokens": count, "token_share": count / total_tokens}
                for value, count in ordered_tokens[:10]
            ]
        return result

    genre_counts: Counter[str] = Counter()
    genre_tokens: Counter[str] = Counter()
    for index, record in enumerate(records):
        genres = record["broad_genres"] or ["unclassified"]
        for genre in genres:
            genre_counts[genre] += 1
            if token_counts is not None:
                genre_tokens[genre] += token_counts[index]
    return {
        "source": distribution("source"),
        "source_collection": distribution("source_collection"),
        "author": distribution("author"),
        "genre": {
            "story_counts": dict(sorted(genre_counts.items())),
            "token_counts": dict(sorted(genre_tokens.items())) if token_counts is not None else None,
        },
    }


def corpus_statistics(records: list[dict[str, Any]], token_counts: list[int]) -> dict[str, Any]:
    words = [record["words"] for record in records]
    rights = Counter(record["rights_status"] for record in records)
    genres = Counter(genre for record in records for genre in record["broad_genres"])
    confidence = Counter(record["extraction_confidence"] for record in records)
    publication_eras: Counter[str] = Counter()
    for record in records:
        year = record.get("publication_year")
        if year is None:
            publication_eras["unknown"] += 1
        elif year < 1850:
            publication_eras["before_1850"] += 1
        elif year < 1900:
            publication_eras["1850_1899"] += 1
        elif year < 1931:
            publication_eras["1900_1930"] += 1
        else:
            publication_eras["1931_or_later"] += 1
    ordered_words = sorted(words)
    ordered_tokens = sorted(token_counts)
    percentile = lambda values, p: values[max(0, math.ceil(len(values) * p) - 1)]
    return {
        "stories": len(records),
        "words": sum(words),
        "exact_document_tokens": sum(token_counts),
        "document_token_format": "<|story|> [broad genre control] <|bos|> title\\n\\ntext <|eos|>",
        "story_words": {
            "minimum": min(words), "median": statistics.median(words),
            "mean": statistics.mean(words), "p95": percentile(ordered_words, 0.95),
            "maximum": max(words),
        },
        "story_tokens": {
            "minimum": min(token_counts), "median": statistics.median(token_counts),
            "mean": statistics.mean(token_counts), "p95": percentile(ordered_tokens, 0.95),
            "maximum": max(token_counts),
        },
        "rights_status_counts": dict(sorted(rights.items())),
        "extraction_confidence_counts": dict(sorted(confidence.items())),
        "publication_era_counts": dict(sorted(publication_eras.items())),
        "broad_genre_counts": dict(sorted(genres.items())),
        "concentration": source_concentration_metrics(records, token_counts),
    }


def acquire(
    url: str,
    relative_path: Path,
    *,
    readable_dirs: list[Path],
    write_dir: Path,
    timeout: int,
) -> tuple[str, SourceArtifact]:
    candidates = [directory / relative_path for directory in readable_dirs]
    existing = next((path for path in candidates if path.exists()), None)
    if existing is not None:
        data = existing.read_bytes()
        method = "existing_cache"
        timestamp = file_timestamp(existing)
        path = existing
    else:
        path = write_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "FictionPulper/2.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        method = "network"
        timestamp = utc_timestamp()
    artifact = SourceArtifact(
        source_id=relative_path.stem,
        url=url,
        path=str(path),
        sha256=hashlib.sha256(data).hexdigest(),
        byte_count=len(data),
        retrieval_method=method,
        retrieved_at=timestamp,
    )
    return data.decode("utf-8-sig", errors="replace"), artifact


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_worktree_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def corpus_audits(
    records: list[dict[str, Any]],
    token_counts: list[int],
    *,
    expected_base_ids: set[str],
    selection: dict[str, Any],
) -> dict[str, Any]:
    validate_canonical_records(records, expected_base_ids=expected_base_ids)
    normalized_hashes = {normalized_text_hash(record["text"]) for record in records}
    new_chapter_findings = []
    new_source_title_findings = []
    new_quality_findings = []
    immutable_base_findings = []
    for record in records:
        extraction = Extraction(
            record["title"], 0, len(record["text"].splitlines()), "high",
            record["boundary_method"], record["text"],
        )
        quality_reason = classify_story_rejection(
            extraction,
            minimum_words=int(selection["minimum_story_words"]),
            maximum_words=int(selection["maximum_story_words"]),
        )
        is_base = bool(record["provenance"].get("base_record"))
        if quality_reason:
            (immutable_base_findings if is_base else new_quality_findings).append(record["id"])
        if (is_structural_title(record["title"]) or chapter_heading_count(record["text"]) >= 3):
            if not is_base:
                new_chapter_findings.append(record["id"])
        source_title = record.get("source_collection")
        if source_title and normalize_heading(record["title"]) == normalize_heading(source_title):
            if not is_base:
                new_source_title_findings.append(record["id"])

    concentration = source_concentration_metrics(records, token_counts)
    limits = selection["concentration_limits"]
    concentration_checks = {
        "source_collection_story_share": (
            concentration["source_collection"]["largest_share"],
            float(limits["maximum_source_collection_story_share"]),
        ),
        "source_collection_token_share": (
            concentration["source_collection"]["largest_token_share"],
            float(limits["maximum_source_collection_token_share"]),
        ),
        "author_story_share": (
            concentration["author"]["largest_share"],
            float(limits["maximum_author_story_share"]),
        ),
        "author_token_share": (
            concentration["author"]["largest_token_share"],
            float(limits["maximum_author_token_share"]),
        ),
    }
    return {
        "schema": {"passed": True, "records_checked": len(records)},
        "base_id_preservation": {"passed": True, "ids_checked": len(expected_base_ids)},
        "rights": {
            "passed": all(
                record["rights"] in ALLOWED_RIGHTS
                and record["rights_status"] in ALLOWED_RIGHTS
                for record in records
            ),
            "status_counts": dict(sorted(Counter(record["rights_status"] for record in records).items())),
        },
        "exact_deduplication": {
            "passed": len(normalized_hashes) == len(records),
            "unique_normalized_hashes": len(normalized_hashes),
        },
        "chapter_contamination": {
            "passed": not new_chapter_findings,
            "accepted_new_record_findings": new_chapter_findings,
        },
        "source_title_contamination": {
            "passed": not new_source_title_findings,
            "accepted_new_record_findings": new_source_title_findings,
        },
        "quality": {
            "passed": not new_quality_findings,
            "accepted_new_record_findings": new_quality_findings,
            "immutable_base_findings": immutable_base_findings,
        },
        "concentration": {
            "passed": all(actual <= limit for actual, limit in concentration_checks.values()),
            "checks": {
                name: {"actual": actual, "maximum": limit, "passed": actual <= limit}
                for name, (actual, limit) in concentration_checks.items()
            },
            "metrics": concentration,
        },
        "lengths": corpus_statistics(records, token_counts),
    }


def _write_stage(
    output_dir: Path,
    stage: int,
    records: list[dict[str, Any]],
    token_counts: list[int],
    cursor: int,
    *,
    expected_base_ids: set[str],
    selection: dict[str, Any],
) -> None:
    token_count = sum(token_counts)
    audits = corpus_audits(
        records, token_counts, expected_base_ids=expected_base_ids, selection=selection
    )
    audits["token_threshold"] = {
        "passed": token_count >= stage,
        "threshold": stage,
        "actual": token_count,
    }
    write_json_atomic(
        output_dir / "stages" / f"{stage // 1_000_000}m.json",
        {
            "threshold_tokens": stage,
            "actual_tokens": token_count,
            "story_count": len(records),
            "last_catalog_cursor": cursor,
            "last_record_id": records[-1]["id"],
            "audits": audits,
            "record_ids_sha256": hashlib.sha256(
                "\n".join(record["id"] for record in records).encode()
            ).hexdigest(),
            "record_ids": [record["id"] for record in records],
        },
    )


def build_corpus_v2(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_config, tokenizer_config = config["base"], config["tokenizer"]
    source_config, selection = config["source"], config["selection"]
    dedup = config["deduplication"]
    review_path = Path(dedup["reviewed_exclusions_path"])
    review_payload = yaml.safe_load(review_path.read_text(encoding="utf-8"))
    reviewed_exclusions = set(review_payload["exclude_ids"])
    if len(reviewed_exclusions) != len(review_payload["exclude_ids"]):
        raise ValueError("Near-duplicate review contains duplicate exclusion IDs")
    output_dir = Path(config["output_dir"])
    base_path, tokenizer_path = Path(base_config["corpus_path"]), Path(tokenizer_config["path"])
    splits_path = Path(base_config["existing_splits_path"])
    for label, path, expected in (
        ("base corpus", base_path, base_config["expected_sha256"]),
        ("tokenizer", tokenizer_path, tokenizer_config["expected_sha256"]),
        ("existing split metadata", splits_path, base_config["existing_splits_expected_sha256"]),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Locked {label} hash changed: {actual}, expected {expected}")
    base_raw = load_corpus(base_path)
    if len(base_raw) != int(base_config["expected_records"]):
        raise RuntimeError("Locked base corpus record count changed")
    split_payload = json.loads(splits_path.read_text())
    old_assignments = split_payload.get("assignments", {})
    base_all = [canonical_base_record(record, old_assignments.get(record["id"])) for record in base_raw]
    base = [record for record in base_all if record["id"] not in reviewed_exclusions]
    excluded_base = [record for record in base_all if record["id"] in reviewed_exclusions]
    expected_base_ids = {record["id"] for record in base}
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for record in base_all:
        record["tokenizer_v1_tokens"] = exact_document_tokens(tokenizer, record)
    validate_canonical_records(base, expected_base_ids=expected_base_ids)
    accepted = list(base)
    review_universe = list(base_all)
    reviewed_records = {record["id"]: record for record in excluded_base}
    additions: list[dict[str, Any]] = []
    exact_report: list[dict[str, Any]] = []
    review_samples: list[dict[str, Any]] = []
    medium_queue: list[dict[str, Any]] = []
    low_queue: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    chapter_audit: list[dict[str, Any]] = []
    source_title_audit: list[dict[str, Any]] = []
    quality_audit: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    exact_hashes = {normalized_text_hash(record["text"]): record for record in accepted}
    token_counts = [record["tokenizer_v1_tokens"] for record in accepted]
    total_tokens = sum(token_counts)
    for record in base:
        base_extraction = Extraction(
            record["title"], 0, len(record["text"].splitlines()), "high",
            record["boundary_method"], record["text"],
        )
        quality_reason = classify_story_rejection(
            base_extraction,
            minimum_words=int(selection["minimum_story_words"]),
            maximum_words=int(selection["maximum_story_words"]),
        )
        if quality_reason:
            quality_audit.append({
                "id": record["id"], "source_id": record["source_id"],
                "title": record["title"], "finding": quality_reason,
                "decision": "retained_immutable_base",
            })
        count = chapter_heading_count(record["text"])
        if count or is_structural_title(record["title"]):
            chapter_audit.append({
                "id": record["id"], "source_id": record["source_id"],
                "title": record["title"], "chapter_heading_count": count,
                "decision": "retained_immutable_base",
            })
        if record.get("source_collection") and normalize_heading(record["title"]) == normalize_heading(
            record["source_collection"]
        ):
            source_title_audit.append({
                "id": record["id"], "source_id": record["source_id"],
                "source_title": record["source_collection"], "story_title": record["title"],
                "decision": "retained_immutable_base",
            })
    stages = sorted(int(value) for value in selection["stage_tokens"])
    completed_stages = {stage for stage in stages if total_tokens >= stage}

    readable_dirs = [Path(value) for value in source_config["readable_cache_dirs"]]
    write_dir = Path(source_config["raw_write_dir"])
    catalog_url = source_config.get("catalog_url", CATALOG_URL)
    catalog_text, catalog_artifact = acquire(
        catalog_url, Path("pg_catalog.csv"), readable_dirs=readable_dirs,
        write_dir=write_dir, timeout=int(source_config["request_timeout_seconds"]),
    )
    source_manifest.append(asdict(catalog_artifact))
    candidates = catalog_candidates(csv.DictReader(io.StringIO(catalog_text)))
    processed = 0
    target = int(selection["target_tokens"])
    for cursor, collection in enumerate(candidates[: int(source_config["max_collections"])]):
        if target_reached(total_tokens, target):
            break
        book_id = collection["Text#"]
        text_url = source_config.get("text_url", TEXT_URL).format(book_id=book_id)
        try:
            raw_text, artifact = acquire(
                text_url, Path("texts") / f"pg-{book_id}.txt", readable_dirs=readable_dirs,
                write_dir=write_dir, timeout=int(source_config["request_timeout_seconds"]),
            )
        except (urllib.error.URLError, TimeoutError, UnicodeError, OSError) as error:
            rejections.append({"source_id": f"pg-{book_id}", "reason": "acquisition_failure", "error": str(error)})
            source_summaries.append({
                "source_id": f"pg-{book_id}", "source_title": collection["Title"],
                "source_url": text_url, "status": "acquisition_failure",
                "accepted_stories": 0, "accepted_tokens": 0,
                "rejection_counts": {"acquisition_failure": 1},
            })
            counters["acquisition_failure"] += 1
            continue
        source_manifest.append(asdict(artifact))
        processed += 1
        cleaned = clean_gutenberg_text(raw_text)
        extractions, evidence = extract_collection_v2(
            cleaned,
            minimum_matches=int(selection["minimum_toc_matches"]),
            high_match_ratio=float(selection["high_confidence_match_ratio"]),
        )
        diagnostic = {
            "source_id": f"pg-{book_id}", "source_title": collection["Title"],
            "source_url": artifact.url,
            "beginning_snippet": cleaned[:1200],
            "ending_snippet": cleaned[-1200:],
            "candidate_story_snippets": [
                {
                    "title": item.title,
                    "opening": item.text[:500],
                    "ending": item.text[-500:],
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                }
                for item in extractions[:10]
            ],
            **evidence,
        }
        source_summary: dict[str, Any] = {
            "source_id": f"pg-{book_id}", "source_title": collection["Title"],
            "source_url": artifact.url, "raw_sha256": artifact.sha256,
            "extraction_confidence": evidence.get("confidence"),
            "toc_title_count": evidence.get("toc_title_count", 0),
            "matched_heading_count": evidence.get("matched_heading_count", 0),
            "extracted_candidates": len(extractions), "accepted_stories": 0,
            "accepted_tokens": 0, "rejection_counts": Counter(),
        }
        if evidence.get("confidence") == "low":
            low_queue.append(diagnostic)
            source_summary["status"] = "low_confidence_rejected"
            source_summary["rejection_counts"]["low_confidence_collection"] += 1
            source_summary["rejection_counts"] = dict(source_summary["rejection_counts"])
            source_summaries.append(source_summary)
            counters["low_confidence_collection"] += 1
            continue
        if evidence.get("confidence") == "medium":
            medium_queue.append(diagnostic)
            source_summary["status"] = "medium_confidence_review"
            source_summary["rejection_counts"]["medium_confidence_collection"] += 1
            source_summary["rejection_counts"] = dict(source_summary["rejection_counts"])
            source_summaries.append(source_summary)
            counters["medium_confidence_collection"] += 1
            continue
        source_summary["status"] = "processed_high_confidence"
        lines = cleaned.splitlines()
        for extraction in extractions:
            sample = review_sample(extraction, collection, lines)
            if len(review_samples) < 500:
                review_samples.append(sample)
            reason = classify_story_rejection(
                extraction,
                minimum_words=int(selection["minimum_story_words"]),
                maximum_words=int(selection["maximum_story_words"]),
            )
            if (
                reason is None
                and normalize_heading(extraction.title) == normalize_heading(collection["Title"])
            ):
                reason = "source_title_match_possible_whole_book"
            chapter_count = chapter_heading_count(extraction.text)
            if chapter_count or is_structural_title(extraction.title):
                chapter_audit.append({
                    "source_id": f"pg-{book_id}", "title": extraction.title,
                    "chapter_heading_count": chapter_count, "decision": reason or "accepted",
                })
            if normalize_heading(extraction.title) == normalize_heading(collection["Title"]):
                source_title_audit.append({
                    "source_id": f"pg-{book_id}", "source_title": collection["Title"],
                    "story_title": extraction.title, "decision": reason or "accepted",
                })
            if reason:
                counters[reason] += 1
                source_summary["rejection_counts"][reason] += 1
                rejections.append({
                    "source_id": f"pg-{book_id}", "title": extraction.title,
                    "words": len(extraction.text.split()), "reason": reason,
                    "opening_snippet": extraction.text[:500],
                })
                continue
            record = make_story_record(extraction, collection, artifact, evidence)
            count = exact_document_tokens(tokenizer, record)
            record["tokenizer_v1_tokens"] = count
            if record["id"] in reviewed_exclusions:
                review_universe.append(record)
                reviewed_records[record["id"]] = record
                counters["reviewed_near_duplicate"] += 1
                source_summary["rejection_counts"]["reviewed_near_duplicate"] += 1
                continue
            if not add_if_not_exact_duplicate(record, accepted, exact_hashes, exact_report):
                counters["exact_duplicate"] += 1
                source_summary["rejection_counts"]["exact_duplicate"] += 1
                continue
            additions.append(record)
            review_universe.append(record)
            token_counts.append(count)
            total_tokens += count
            source_summary["accepted_stories"] += 1
            source_summary["accepted_tokens"] += count
            for stage in stages:
                if stage not in completed_stages and total_tokens >= stage:
                    _write_stage(
                        output_dir,
                        stage,
                        accepted,
                        token_counts,
                        cursor,
                        expected_base_ids=expected_base_ids,
                        selection=selection,
                    )
                    completed_stages.add(stage)
            if target_reached(total_tokens, target):
                break
        source_summary["rejection_counts"] = dict(sorted(source_summary["rejection_counts"].items()))
        source_summaries.append(source_summary)
        if processed % 25 == 0:
            print(f"collections={processed} stories={len(accepted)} exact_tokens={total_tokens:,}")
        delay = float(source_config["request_delay_seconds"])
        if delay:
            time.sleep(delay)

    missing_reviewed_ids = reviewed_exclusions - reviewed_records.keys()
    if missing_reviewed_ids:
        raise RuntimeError(
            "Reviewed near-duplicate exclusions were not encountered: "
            f"{sorted(missing_reviewed_ids)[:10]}"
        )
    validate_canonical_records(accepted, expected_base_ids=expected_base_ids)
    near_edges, near_clusters = near_duplicate_graph(
        review_universe, max_hamming=int(dedup["simhash_max_hamming"]),
        minimum_jaccard=float(dedup["minimum_jaccard"]),
        minimum_length_ratio=float(dedup["minimum_length_ratio"]),
    )
    accepted_ids = {record["id"] for record in accepted}
    unresolved_clusters = [
        cluster for cluster in near_clusters
        if sum(record_id in accepted_ids for record_id in cluster["record_ids"]) > 1
    ]
    unresolved_cluster_ids = {cluster["cluster_id"] for cluster in unresolved_clusters}
    unresolved_edges = [
        edge for edge in near_edges if edge["cluster_id"] in unresolved_cluster_ids
    ]
    resolution_report = [
        {
            "excluded_id": record_id,
            "title": reviewed_records[record_id]["title"],
            "source_id": reviewed_records[record_id]["source_id"],
            "was_locked_base_record": bool(
                reviewed_records[record_id]["provenance"].get("base_record")
            ),
            "decision": "exclude_reviewed_duplicate_edition_or_formatting_variant",
        }
        for record_id in review_payload["exclude_ids"]
    ]
    stats = corpus_statistics(accepted, token_counts)
    minimum_allowed, maximum_allowed = map(int, selection["allowed_token_range"])
    audits = corpus_audits(
        accepted,
        token_counts,
        expected_base_ids=expected_base_ids,
        selection=selection,
    )
    audits["base_id_preservation"]["original_locked_base_records"] = len(base_all)
    audits["base_id_preservation"]["documented_correctness_exclusions"] = [
        record["id"] for record in excluded_base
    ]
    audits["exact_deduplication"]["rejected_candidates"] = len(exact_report)
    audits["near_duplicate_review"] = {
        "passed": not unresolved_clusters,
        "status": "clear" if not unresolved_clusters else "human_review_required_before_freeze",
        "candidate_edges": len(near_edges),
        "candidate_clusters": len(near_clusters),
        "reviewed_exclusions_applied": len(reviewed_exclusions),
        "unresolved_edges": len(unresolved_edges),
        "unresolved_clusters": len(unresolved_clusters),
    }
    audits["token_target"] = {
        "passed": minimum_allowed <= total_tokens <= maximum_allowed,
        "target": target, "allowed_range": [minimum_allowed, maximum_allowed],
        "actual": total_tokens,
    }
    audits["freeze_ready"] = all(
        audit.get("passed", True) for audit in audits.values() if isinstance(audit, dict)
    )
    stats.update({
        "locked_base_stories": len(base_all), "retained_base_stories": len(base),
        "documented_base_correctness_exclusions": len(excluded_base),
        "new_stories": len(additions),
        "collections_processed": processed, "review_edges": len(near_edges),
        "review_clusters": len(near_clusters),
        "unresolved_review_clusters": len(unresolved_clusters),
        "reviewed_near_duplicate_exclusions": len(reviewed_exclusions),
        "rejection_counts": dict(sorted(counters.items())),
        "stop_reason": "target_reached" if total_tokens >= target else "sources_exhausted",
        "rights_basis": source_config["rights_basis"],
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "corpus.jsonl": accepted, "additions.jsonl": additions,
        "medium-review.jsonl": medium_queue, "low-confidence.jsonl": low_queue,
        "rejections.jsonl": rejections, "exact-duplicates.jsonl": exact_report,
        "near-duplicate-candidates.jsonl": near_edges,
        "near-duplicate-clusters.jsonl": near_clusters,
        "near-duplicate-unresolved.jsonl": unresolved_edges,
        "near-duplicate-resolution.jsonl": resolution_report,
        "review-samples.jsonl": review_samples,
        "chapter-contamination-audit.jsonl": chapter_audit,
        "source-title-audit.jsonl": source_title_audit,
        "quality-audit.jsonl": quality_audit,
        "source-manifest.jsonl": source_manifest,
        "source-summaries.jsonl": source_summaries,
    }
    for name, payload in outputs.items():
        write_jsonl(output_dir / name, payload)
    write_json_atomic(output_dir / "audits.json", audits)
    write_json_atomic(output_dir / "stats.json", stats)
    corpus_hash = sha256_file(output_dir / "corpus.jsonl")
    manifest = {
        "corpus_version": 2, "preprocessing_version": PREPROCESSING_VERSION,
        "created_at": utc_timestamp(), "config_path": str(config_path),
        "config_sha256": sha256_file(config_path), "git_commit": _git_commit(),
        "git_worktree_dirty": _git_worktree_dirty(),
        "builder_sha256": sha256_file(Path(__file__)),
        "base_corpus_path": str(base_path), "base_corpus_sha256": sha256_file(base_path),
        "existing_splits_path": str(splits_path), "existing_splits_sha256": sha256_file(splits_path),
        "tokenizer_path": str(tokenizer_path), "tokenizer_sha256": sha256_file(tokenizer_path),
        "near_duplicate_review_path": str(review_path),
        "near_duplicate_review_sha256": sha256_file(review_path),
        "source_manifest_sha256": sha256_file(output_dir / "source-manifest.jsonl"),
        "settings": config, "corpus_sha256": corpus_hash,
        "determinism_note": "Content and ordering are deterministic; timestamps and live retrieval metadata are not.",
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    failed_audits = [
        name for name, audit in audits.items()
        if isinstance(audit, dict) and audit.get("passed") is False
    ]
    if failed_audits:
        raise RuntimeError(
            f"Corpus-v2 is not freeze-ready; failed audits: {', '.join(failed_audits)}. "
            "Reports were retained for diagnosis and review."
        )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/corpus-v2.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build_corpus_v2(args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
