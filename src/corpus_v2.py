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
from typing import Any, Callable, Iterable

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
CHECKPOINT_VERSION = 1
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
    etag: str | None = None
    last_modified: str | None = None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_jsonl_sha256(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
    return digest.hexdigest()


def processing_fingerprint(
    *,
    selection: dict[str, Any],
    tokenizer_sha256: str,
    review_sha256: str,
    builder_sha256: str,
    collection: dict[str, str] | None = None,
) -> str:
    """Identify every deterministic input to a source processing shard."""
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "selection": selection,
        "tokenizer_sha256": tokenizer_sha256,
        "review_sha256": review_sha256,
        "builder_sha256": builder_sha256,
        "collection": collection,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_source_ledger(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != CHECKPOINT_VERSION or not isinstance(payload.get("sources"), dict):
        raise ValueError(f"Invalid Corpus-v2 source ledger: {path}")
    return payload["sources"]


def save_source_ledger(path: Path, sources: dict[str, dict[str, Any]]) -> None:
    write_json_atomic(path, {"version": CHECKPOINT_VERSION, "sources": sources})


def save_source_shard(path: Path, shard: dict[str, Any]) -> None:
    write_json_atomic(path, shard)


def load_valid_source_shard(
    entry: dict[str, Any],
    shard_path: Path,
    *,
    source_id: str,
    url: str,
    fingerprint: str,
) -> tuple[dict[str, Any], SourceArtifact] | None:
    """Load a completed source only after verifying its cached bytes and identity."""
    if (
        entry.get("source_id") != source_id
        or entry.get("url") != url
        or entry.get("processing_status") not in {"processed", "completed"}
        or entry.get("processing_fingerprint") != fingerprint
        or not shard_path.exists()
    ):
        return None
    artifact_payload = entry.get("artifact")
    if not isinstance(artifact_payload, dict):
        return None
    raw_path = Path(str(artifact_payload.get("path", "")))
    expected_sha = entry.get("raw_sha256")
    expected_shard_sha = entry.get("shard_sha256")
    if (
        not raw_path.is_file()
        or not isinstance(expected_sha, str)
        or artifact_payload.get("sha256") != expected_sha
        or not isinstance(expected_shard_sha, str)
        or sha256_file(shard_path) != expected_shard_sha
    ):
        return None
    raw_bytes = raw_path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != expected_sha:
        return None
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    if (
        shard.get("version") != CHECKPOINT_VERSION
        or shard.get("source_id") != source_id
        or shard.get("url") != url
        or shard.get("raw_sha256") != expected_sha
        or shard.get("processing_fingerprint") != fingerprint
    ):
        return None
    return shard, SourceArtifact(**artifact_payload)


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
            result["top_20_token_share"] = sum(
                count for _, count in ordered_tokens[:20]
            ) / total_tokens
            result["top_10_by_tokens"] = [
                {"value": value, "tokens": count, "token_share": count / total_tokens}
                for value, count in ordered_tokens[:10]
            ]
            result["top_20_by_tokens"] = [
                {"value": value, "tokens": count, "token_share": count / total_tokens}
                for value, count in ordered_tokens[:20]
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
        etag = None
        last_modified = None
    else:
        path = write_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "FictionPulper/2.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")
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
        etag=etag,
        last_modified=last_modified,
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


def process_source_to_shard(
    collection: dict[str, str],
    raw_text: str,
    artifact: SourceArtifact,
    tokenizer: Tokenizer,
    selection: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    """Extract and locally validate one source without corpus-global decisions."""
    book_id = collection["Text#"]
    source_id = f"pg-{book_id}"
    cleaned = clean_gutenberg_text(raw_text)
    extractions, evidence = extract_collection_v2(
        cleaned,
        minimum_matches=int(selection["minimum_toc_matches"]),
        high_match_ratio=float(selection["high_confidence_match_ratio"]),
    )
    diagnostic = {
        "source_id": source_id,
        "source_title": collection["Title"],
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
        "source_id": source_id,
        "source_title": collection["Title"],
        "source_url": artifact.url,
        "raw_sha256": artifact.sha256,
        "extraction_confidence": evidence.get("confidence"),
        "toc_title_count": evidence.get("toc_title_count", 0),
        "matched_heading_count": evidence.get("matched_heading_count", 0),
        "extracted_candidates": len(extractions),
        "accepted_stories": 0,
        "accepted_tokens": 0,
        "rejection_counts": {},
    }
    shard: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "source_id": source_id,
        "url": artifact.url,
        "raw_sha256": artifact.sha256,
        "processing_fingerprint": fingerprint,
        "collection": collection,
        "evidence": evidence,
        "diagnostic": diagnostic,
        "source_summary": source_summary,
        "events": [],
    }
    confidence = evidence.get("confidence")
    if confidence in {"low", "medium"}:
        reason = (
            "low_confidence_collection" if confidence == "low" else "medium_confidence_collection"
        )
        source_summary["status"] = (
            "low_confidence_rejected" if confidence == "low" else "medium_confidence_review"
        )
        source_summary["rejection_counts"] = {reason: 1}
        return shard

    source_summary["status"] = "processed_high_confidence"
    lines = cleaned.splitlines()
    for extraction in extractions:
        sample = review_sample(extraction, collection, lines)
        event: dict[str, Any] = {"review_sample": sample}
        reason = classify_story_rejection(
            extraction,
            minimum_words=int(selection["minimum_story_words"]),
            maximum_words=int(selection["maximum_story_words"]),
        )
        if reason is None and normalize_heading(extraction.title) == normalize_heading(
            collection["Title"]
        ):
            reason = "source_title_match_possible_whole_book"
        chapter_count = chapter_heading_count(extraction.text)
        if chapter_count or is_structural_title(extraction.title):
            finding = {
                "source_id": source_id,
                "title": extraction.title,
                "chapter_heading_count": chapter_count,
                "decision": reason or "accepted",
            }
            event["chapter_audit"] = finding
        if normalize_heading(extraction.title) == normalize_heading(collection["Title"]):
            finding = {
                "source_id": source_id,
                "source_title": collection["Title"],
                "story_title": extraction.title,
                "decision": reason or "accepted",
            }
            event["source_title_audit"] = finding
        if reason:
            rejected = {
                "source_id": source_id,
                "title": extraction.title,
                "words": len(extraction.text.split()),
                "reason": reason,
                "opening_snippet": extraction.text[:500],
            }
            event["rejection"] = rejected
            shard["events"].append(event)
            continue
        record = make_story_record(extraction, collection, artifact, evidence)
        record["tokenizer_v1_tokens"] = exact_document_tokens(tokenizer, record)
        event["candidate_record"] = record
        shard["events"].append(event)
    return shard


REQUIRED_CONTENT_AUDITS = {
    "schema",
    "rights",
    "exact_deduplication",
    "near_duplicate_review",
    "chapter_contamination",
    "source_title_contamination",
    "quality",
    "author_concentration",
    "source_concentration",
    "source_collection_concentration",
    "genre_distribution",
    "story_length",
    "token_length",
    "canonical_content",
}


def failed_required_audits(audits: dict[str, Any]) -> list[str]:
    failures = []
    for name in sorted(REQUIRED_CONTENT_AUDITS):
        audit = audits.get(name)
        if not isinstance(audit, dict) or audit.get("passed") is not True:
            failures.append(name)
    return failures


def replay_source_shard(
    shard: dict[str, Any],
    *,
    accepted: list[dict[str, Any]],
    additions: list[dict[str, Any]],
    token_counts: list[int],
    exact_hashes: dict[str, dict[str, Any]],
    exact_report: list[dict[str, Any]],
    review_universe: list[dict[str, Any]],
    reviewed_records: dict[str, dict[str, Any]],
    reviewed_exclusions: set[str],
    review_samples: list[dict[str, Any]],
    medium_queue: list[dict[str, Any]],
    low_queue: list[dict[str, Any]],
    rejections: list[dict[str, Any]],
    chapter_audit: list[dict[str, Any]],
    source_title_audit: list[dict[str, Any]],
    counters: Counter[str],
    target_tokens: int,
    on_accept: Callable[[int], None] | None = None,
) -> tuple[dict[str, Any], int]:
    """Replay deterministic local evidence and apply corpus-global decisions in order."""
    summary = dict(shard["source_summary"])
    summary_counts: Counter[str] = Counter(summary.get("rejection_counts", {}))
    confidence = shard["evidence"].get("confidence")
    total_tokens = sum(token_counts)
    if confidence == "low":
        low_queue.append(shard["diagnostic"])
        counters["low_confidence_collection"] += 1
    elif confidence == "medium":
        medium_queue.append(shard["diagnostic"])
        counters["medium_confidence_collection"] += 1
    else:
        events = shard.get("events")
        if not events:
            events = [{"candidate_record": record} for record in shard["candidate_records"]]
        for event in events:
            sample = event.get("review_sample")
            if sample is not None and len(review_samples) < 500:
                review_samples.append(sample)
            if event.get("chapter_audit") is not None:
                chapter_audit.append(event["chapter_audit"])
            if event.get("source_title_audit") is not None:
                source_title_audit.append(event["source_title_audit"])
            rejection = event.get("rejection")
            if rejection is not None:
                rejections.append(rejection)
                counters[rejection["reason"]] += 1
                summary_counts[rejection["reason"]] += 1
                continue
            record = event["candidate_record"]
            if record["id"] in reviewed_exclusions:
                # A locked base exclusion may reappear with the same stable ID when
                # its Gutenberg source is replayed. Preserve the original provenance
                # and keep each review identity unique.
                if record["id"] not in reviewed_records:
                    review_universe.append(record)
                    reviewed_records[record["id"]] = record
                counters["reviewed_near_duplicate"] += 1
                summary_counts["reviewed_near_duplicate"] += 1
                continue
            if not add_if_not_exact_duplicate(record, accepted, exact_hashes, exact_report):
                counters["exact_duplicate"] += 1
                summary_counts["exact_duplicate"] += 1
                continue
            additions.append(record)
            review_universe.append(record)
            count = record["tokenizer_v1_tokens"]
            token_counts.append(count)
            total_tokens += count
            summary["accepted_stories"] += 1
            summary["accepted_tokens"] += count
            if on_accept is not None:
                on_accept(total_tokens)
            if target_reached(total_tokens, target_tokens):
                break
    summary["rejection_counts"] = dict(sorted(summary_counts.items()))
    return summary, sum(token_counts)


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
    author_checks = {
        name: value for name, value in concentration_checks.items() if name.startswith("author_")
    }
    collection_checks = {
        name: value
        for name, value in concentration_checks.items()
        if name.startswith("source_collection_")
    }
    new_records = [record for record in records if not record["provenance"].get("base_record")]
    invalid_genres = sorted(
        {
            genre
            for record in records
            for genre in record["broad_genres"]
            if genre not in GENRE_TERMS
        }
    )
    new_story_length_findings = [
        record["id"]
        for record in new_records
        if not int(selection["minimum_story_words"])
        <= record["words"]
        <= int(selection["maximum_story_words"])
    ]
    immutable_story_length_findings = [
        record["id"]
        for record in records
        if record["provenance"].get("base_record")
        and not int(selection["minimum_story_words"])
        <= record["words"]
        <= int(selection["maximum_story_words"])
    ]
    stats = corpus_statistics(records, token_counts)
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
        "author_concentration": {
            "passed": all(actual <= limit for actual, limit in author_checks.values()),
            "checks": {
                name: {"actual": actual, "maximum": limit, "passed": actual <= limit}
                for name, (actual, limit) in author_checks.items()
            },
            "unique_authors": concentration["author"]["unique_count"],
            "largest_token_share": concentration["author"]["largest_token_share"],
            "top_10_token_share": concentration["author"]["top_10_token_share"],
            "top_20_token_share": concentration["author"]["top_20_token_share"],
        },
        "source_concentration": {
            "passed": (
                concentration["source"]["unique_count"] > 0
                and 0.0 < concentration["source"]["largest_token_share"] <= 1.0
            ),
            "unique_sources": concentration["source"]["unique_count"],
            "largest_token_share": concentration["source"]["largest_token_share"],
            "top_10_token_share": concentration["source"]["top_10_token_share"],
            "note": "Reported as an integrity/concentration audit; no source-platform cap is configured.",
        },
        "source_collection_concentration": {
            "passed": all(actual <= limit for actual, limit in collection_checks.values()),
            "checks": {
                name: {"actual": actual, "maximum": limit, "passed": actual <= limit}
                for name, (actual, limit) in collection_checks.items()
            },
            "unique_source_collections": concentration["source_collection"]["unique_count"],
            "largest_token_share": concentration["source_collection"]["largest_token_share"],
        },
        "genre_distribution": {
            "passed": not invalid_genres,
            "invalid_genres": invalid_genres,
            "story_counts": concentration["genre"]["story_counts"],
            "token_counts": concentration["genre"]["token_counts"],
        },
        "story_length": {
            "passed": not new_story_length_findings,
            "accepted_new_record_findings": new_story_length_findings,
            "immutable_base_findings": immutable_story_length_findings,
            "statistics": stats["story_words"],
        },
        "token_length": {
            "passed": len(token_counts) == len(records) and all(count > 0 for count in token_counts),
            "records_checked": len(token_counts),
            "statistics": stats["story_tokens"],
        },
        "canonical_content": {
            "passed": True,
            "sha256": canonical_jsonl_sha256(records),
            "record_ids_sha256": hashlib.sha256(
                "\n".join(record["id"] for record in records).encode("utf-8")
            ).hexdigest(),
        },
        "lengths": stats,
    }


def near_duplicate_audit(
    review_universe: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    dedup: dict[str, Any],
    reviewed_exclusions: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    edges, clusters = near_duplicate_graph(
        review_universe,
        max_hamming=int(dedup["simhash_max_hamming"]),
        minimum_jaccard=float(dedup["minimum_jaccard"]),
        minimum_length_ratio=float(dedup["minimum_length_ratio"]),
    )
    accepted_ids = {record["id"] for record in accepted}
    unresolved_clusters = [
        cluster
        for cluster in clusters
        if sum(record_id in accepted_ids for record_id in cluster["record_ids"]) > 1
    ]
    unresolved_cluster_ids = {cluster["cluster_id"] for cluster in unresolved_clusters}
    unresolved_edges = [
        edge for edge in edges if edge["cluster_id"] in unresolved_cluster_ids
    ]
    audit = {
        "passed": not unresolved_clusters,
        "status": "clear" if not unresolved_clusters else "human_review_required_before_freeze",
        "candidate_edges": len(edges),
        "candidate_clusters": len(clusters),
        "reviewed_exclusions_applied": len(reviewed_exclusions & set().union(
            *(set(cluster["record_ids"]) for cluster in clusters)
        )) if clusters else 0,
        "unresolved_edges": len(unresolved_edges),
        "unresolved_clusters": len(unresolved_clusters),
    }
    return edges, clusters, unresolved_edges, audit


def _rejection_total(counters: Counter[str], terms: tuple[str, ...]) -> int:
    return sum(count for reason, count in counters.items() if any(term in reason for term in terms))


def _write_stage(
    output_dir: Path,
    stage: int,
    records: list[dict[str, Any]],
    token_counts: list[int],
    cursor: int,
    *,
    expected_base_ids: set[str],
    selection: dict[str, Any],
    dedup: dict[str, Any],
    review_universe: list[dict[str, Any]],
    reviewed_exclusions: set[str],
    exact_report: list[dict[str, Any]],
    medium_queue: list[dict[str, Any]],
    low_queue: list[dict[str, Any]],
    counters: Counter[str],
    prior_story_count: int,
    prior_token_count: int,
) -> dict[str, int]:
    token_count = sum(token_counts)
    audits = corpus_audits(
        records, token_counts, expected_base_ids=expected_base_ids, selection=selection
    )
    near_edges, near_clusters, unresolved_edges, near_audit = near_duplicate_audit(
        review_universe, records, dedup, reviewed_exclusions
    )
    audits["near_duplicate_review"] = near_audit
    audits["token_threshold"] = {
        "passed": token_count >= stage,
        "threshold": stage,
        "actual": token_count,
    }
    concentration = audits["concentration"]["metrics"]
    stage_name = f"{stage // 1_000_000}m"
    stage_dir = output_dir / "stages"
    write_jsonl(stage_dir / f"{stage_name}-near-duplicate-candidates.jsonl", near_edges)
    write_jsonl(stage_dir / f"{stage_name}-near-duplicate-clusters.jsonl", near_clusters)
    write_jsonl(stage_dir / f"{stage_name}-near-duplicate-unresolved.jsonl", unresolved_edges)
    required_failures = failed_required_audits(audits)
    report = {
        "threshold_tokens": stage,
        "actual_tokens": token_count,
        "story_count": len(records),
        "delta_from_prior_stage": {
            "stories": len(records) - prior_story_count,
            "tokens": token_count - prior_token_count,
        },
        "unique_authors": concentration["author"]["unique_count"],
        "unique_source_collections": concentration["source_collection"]["unique_count"],
        "exact_duplicates_rejected": len(exact_report),
        "near_candidates": len(near_edges),
        "medium_review_items": len(medium_queue),
        "low_confidence_items": len(low_queue),
        "rights_rejections": _rejection_total(counters, ("rights", "copyright")),
        "quality_ocr_rejections": _rejection_total(
            counters, ("quality", "ocr", "damaged", "nonprose", "poetry", "paragraph", "too_short")
        ),
        "chapter_novel_rejections": _rejection_total(
            counters, ("chapter", "structural", "novel", "whole_book")
        ),
        "largest_author_token_share": concentration["author"]["largest_token_share"],
        "top_10_author_token_share": concentration["author"]["top_10_token_share"],
        "top_20_author_token_share": concentration["author"]["top_20_token_share"],
        "largest_source_collection_token_share": concentration["source_collection"][
            "largest_token_share"
        ],
        "largest_source_token_share": concentration["source"]["largest_token_share"],
        "last_catalog_cursor": cursor,
        "last_record_id": records[-1]["id"],
        "audits": audits,
        "required_audits_passed": not required_failures,
        "failed_required_audits": required_failures,
        "canonical_content_sha256": audits["canonical_content"]["sha256"],
        "canonical_index": [
            {"id": record["id"], "text_hash": record["text_hash"]} for record in records
        ],
        "record_ids_sha256": audits["canonical_content"]["record_ids_sha256"],
        "record_ids": [record["id"] for record in records],
        "near_duplicate_evidence": {
            "candidates_path": str(stage_dir / f"{stage_name}-near-duplicate-candidates.jsonl"),
            "clusters_path": str(stage_dir / f"{stage_name}-near-duplicate-clusters.jsonl"),
            "unresolved_path": str(stage_dir / f"{stage_name}-near-duplicate-unresolved.jsonl"),
        },
    }
    write_json_atomic(
        stage_dir / f"{stage_name}.json",
        report,
    )
    if required_failures or not audits["token_threshold"]["passed"]:
        failed = required_failures or ["token_threshold"]
        raise RuntimeError(f"Corpus-v2 {stage_name} stage failed audits: {', '.join(failed)}")
    return {"stories": len(records), "tokens": token_count}


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
    prior_stage = {"stories": len(base), "tokens": total_tokens}

    readable_dirs = [Path(value) for value in source_config["readable_cache_dirs"]]
    write_dir = Path(source_config["raw_write_dir"])
    state_dir = output_dir / "state"
    shard_dir = state_dir / "sources"
    ledger_path = state_dir / "processed-sources.json"
    ledger = load_source_ledger(ledger_path)
    builder_sha = sha256_file(Path(__file__))
    tokenizer_sha = sha256_file(tokenizer_path)
    review_sha = sha256_file(review_path)
    catalog_url = source_config.get("catalog_url", CATALOG_URL)
    catalog_text, catalog_artifact = acquire(
        catalog_url, Path("pg_catalog.csv"), readable_dirs=readable_dirs,
        write_dir=write_dir, timeout=int(source_config["request_timeout_seconds"]),
    )
    source_manifest.append(asdict(catalog_artifact))
    ledger[catalog_artifact.source_id] = {
        "source_id": catalog_artifact.source_id,
        "url": catalog_artifact.url,
        "retrieval_status": "retrieved",
        "raw_sha256": catalog_artifact.sha256,
        "etag": catalog_artifact.etag,
        "last_modified": catalog_artifact.last_modified,
        "processing_status": "catalog_loaded",
        "accepted_story_ids": [],
        "outcome": "catalog_loaded",
        "processed_at": utc_timestamp(),
        "processing_fingerprint": processing_fingerprint(
            selection=selection,
            tokenizer_sha256=tokenizer_sha,
            review_sha256=review_sha,
            builder_sha256=builder_sha,
        ),
        "artifact": asdict(catalog_artifact),
    }
    save_source_ledger(ledger_path, ledger)
    candidates = catalog_candidates(csv.DictReader(io.StringIO(catalog_text)))
    processed = 0
    sources_replayed = 0
    sources_processed = 0
    target = int(selection["target_tokens"])
    for cursor, collection in enumerate(candidates[: int(source_config["max_collections"])]):
        if target_reached(total_tokens, target):
            break
        book_id = collection["Text#"]
        source_id = f"pg-{book_id}"
        text_url = source_config.get("text_url", TEXT_URL).format(book_id=book_id)
        fingerprint = processing_fingerprint(
            selection=selection,
            tokenizer_sha256=tokenizer_sha,
            review_sha256=review_sha,
            builder_sha256=builder_sha,
            collection=collection,
        )
        shard_path = shard_dir / f"{source_id}.json"
        loaded = load_valid_source_shard(
            ledger.get(source_id, {}),
            shard_path,
            source_id=source_id,
            url=text_url,
            fingerprint=fingerprint,
        )
        if loaded is not None:
            shard, artifact = loaded
            sources_replayed += 1
        else:
            try:
                raw_text, artifact = acquire(
                    text_url,
                    Path("texts") / f"pg-{book_id}.txt",
                    readable_dirs=readable_dirs,
                    write_dir=write_dir,
                    timeout=int(source_config["request_timeout_seconds"]),
                )
            except (urllib.error.URLError, TimeoutError, UnicodeError, OSError) as error:
                rejection = {"source_id": source_id, "reason": "acquisition_failure", "error": str(error)}
                rejections.append(rejection)
                summary = {
                    "source_id": source_id, "source_title": collection["Title"],
                    "source_url": text_url, "status": "acquisition_failure",
                    "accepted_stories": 0, "accepted_tokens": 0,
                    "rejection_counts": {"acquisition_failure": 1},
                }
                source_summaries.append(summary)
                ledger[source_id] = {
                    "source_id": source_id,
                    "url": text_url,
                    "retrieval_status": "failed",
                    "raw_sha256": None,
                    "etag": None,
                    "last_modified": None,
                    "processing_status": "acquisition_failure",
                    "accepted_story_ids": [],
                    "outcome": summary,
                    "processed_at": utc_timestamp(),
                    "processing_fingerprint": fingerprint,
                    "artifact": None,
                }
                save_source_ledger(ledger_path, ledger)
                counters["acquisition_failure"] += 1
                continue
            shard = process_source_to_shard(
                collection, raw_text, artifact, tokenizer, selection, fingerprint
            )
            save_source_shard(shard_path, shard)
            sources_processed += 1
            ledger[source_id] = {
                "source_id": source_id,
                "url": text_url,
                "retrieval_status": "retrieved",
                "raw_sha256": artifact.sha256,
                "etag": artifact.etag,
                "last_modified": artifact.last_modified,
                "processing_status": "processed",
                "accepted_story_ids": [],
                "outcome": {
                    "status": shard["source_summary"]["status"],
                    "rejection_counts": shard["source_summary"]["rejection_counts"],
                },
                "processed_at": utc_timestamp(),
                "processing_fingerprint": fingerprint,
                "shard_sha256": sha256_file(shard_path),
                "artifact": asdict(artifact),
            }
            save_source_ledger(ledger_path, ledger)
        source_manifest.append(asdict(artifact))
        processed += 1
        accepted_before = len(additions)

        def audit_reached_stages(current_tokens: int) -> None:
            for stage in stages:
                if stage in completed_stages or current_tokens < stage:
                    continue
                snapshot = _write_stage(
                    output_dir,
                    stage,
                    accepted,
                    token_counts,
                    cursor,
                    expected_base_ids=expected_base_ids,
                    selection=selection,
                    dedup=dedup,
                    review_universe=review_universe,
                    reviewed_exclusions=reviewed_exclusions,
                    exact_report=exact_report,
                    medium_queue=medium_queue,
                    low_queue=low_queue,
                    counters=counters,
                    prior_story_count=prior_stage["stories"],
                    prior_token_count=prior_stage["tokens"],
                )
                prior_stage.update(snapshot)
                completed_stages.add(stage)

        source_summary, total_tokens = replay_source_shard(
            shard,
            accepted=accepted,
            additions=additions,
            token_counts=token_counts,
            exact_hashes=exact_hashes,
            exact_report=exact_report,
            review_universe=review_universe,
            reviewed_records=reviewed_records,
            reviewed_exclusions=reviewed_exclusions,
            review_samples=review_samples,
            medium_queue=medium_queue,
            low_queue=low_queue,
            rejections=rejections,
            chapter_audit=chapter_audit,
            source_title_audit=source_title_audit,
            counters=counters,
            target_tokens=target,
            on_accept=audit_reached_stages,
        )
        source_summaries.append(source_summary)
        accepted_story_ids = [record["id"] for record in additions[accepted_before:]]
        ledger[source_id] = {
            "source_id": source_id,
            "url": text_url,
            "retrieval_status": "retrieved",
            "raw_sha256": artifact.sha256,
            "etag": artifact.etag,
            "last_modified": artifact.last_modified,
            "processing_status": "completed",
            "accepted_story_ids": accepted_story_ids,
            "outcome": {
                "status": source_summary["status"],
                "rejection_counts": source_summary["rejection_counts"],
            },
            "processed_at": utc_timestamp(),
            "processing_fingerprint": fingerprint,
            "shard_sha256": sha256_file(shard_path),
            "artifact": asdict(artifact),
        }
        save_source_ledger(ledger_path, ledger)
        if processed % 25 == 0:
            print(f"collections={processed} stories={len(accepted)} exact_tokens={total_tokens:,}")
        if loaded is None:
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
    near_edges, near_clusters, unresolved_edges, near_audit = near_duplicate_audit(
        review_universe, accepted, dedup, reviewed_exclusions
    )
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
    near_audit["reviewed_exclusions_applied"] = len(reviewed_exclusions)
    audits["near_duplicate_review"] = near_audit
    audits["token_target"] = {
        "passed": minimum_allowed <= total_tokens <= maximum_allowed,
        "target": target, "allowed_range": [minimum_allowed, maximum_allowed],
        "actual": total_tokens,
    }
    missing_stages = sorted(set(stages) - completed_stages)
    audits["stage_completion"] = {
        "passed": not missing_stages,
        "required_thresholds": stages,
        "missing_thresholds": missing_stages,
    }
    required_failures = failed_required_audits(audits)
    audits["freeze_ready"] = {
        "passed": not required_failures
        and audits["token_target"]["passed"]
        and audits["base_id_preservation"]["passed"]
        and audits["stage_completion"]["passed"],
        "failed_required_audits": required_failures,
        "unresolved_near_duplicate_clusters": near_audit["unresolved_clusters"],
    }
    stats.update({
        "locked_base_stories": len(base_all), "retained_base_stories": len(base),
        "documented_base_correctness_exclusions": len(excluded_base),
        "new_stories": len(additions),
        "collections_processed": processed,
        "sources_replayed_from_ledger": sources_replayed,
        "sources_processed_from_raw": sources_processed,
        "review_edges": len(near_edges),
        "review_clusters": len(near_clusters),
        "unresolved_review_clusters": near_audit["unresolved_clusters"],
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
    corpus_hash = sha256_file(output_dir / "corpus.jsonl")
    expected_corpus_hash = audits["canonical_content"]["sha256"]
    audits["canonical_content"]["file_sha256"] = corpus_hash
    audits["canonical_content"]["passed"] = corpus_hash == expected_corpus_hash
    if not audits["canonical_content"]["passed"]:
        audits["freeze_ready"]["passed"] = False
        audits["freeze_ready"]["failed_required_audits"] = sorted(
            set(audits["freeze_ready"]["failed_required_audits"]) | {"canonical_content"}
        )
    write_json_atomic(output_dir / "audits.json", audits)
    write_json_atomic(output_dir / "stats.json", stats)
    manifest = {
        "corpus_version": 2, "preprocessing_version": PREPROCESSING_VERSION,
        "created_at": utc_timestamp(), "config_path": str(config_path),
        "config_sha256": sha256_file(config_path), "git_commit": _git_commit(),
        "git_worktree_dirty": _git_worktree_dirty(),
        "builder_sha256": builder_sha,
        "base_corpus_path": str(base_path), "base_corpus_sha256": sha256_file(base_path),
        "existing_splits_path": str(splits_path), "existing_splits_sha256": sha256_file(splits_path),
        "tokenizer_path": str(tokenizer_path), "tokenizer_sha256": sha256_file(tokenizer_path),
        "near_duplicate_review_path": str(review_path),
        "near_duplicate_review_sha256": sha256_file(review_path),
        "source_manifest_sha256": sha256_file(output_dir / "source-manifest.jsonl"),
        "source_ledger_path": str(ledger_path),
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
