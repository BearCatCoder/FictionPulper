"""Build FictionPulper Corpus v1 from high-confidence Gutenberg collections."""

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
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from src.train_tokenizer import load_corpus, sha256_file, write_json_atomic


CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
TEXT_URL = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
PREPROCESSING_VERSION = 1
MIN_STORY_WORDS = 500
MAX_STORY_WORDS = 10_000

START_MARKER = re.compile(
    r"^\s*\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$",
    re.IGNORECASE | re.MULTILINE,
)
END_MARKER = re.compile(
    r"^\s*\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$",
    re.IGNORECASE | re.MULTILINE,
)
TOC_HEADING = re.compile(
    r"^\s*(?:table of )?contents(?:\s+of\s+.+)?[.:]?\s*$", re.IGNORECASE
)
GENERIC_HEADING = re.compile(
    r"^(?:chapter|part|book|volume|section|story|stories|verse|bibliography|appendix)\b",
    re.IGNORECASE,
)
NON_STORY_TITLE = re.compile(
    r"^(?:preface|foreword|introduction\b.*|contents|acknowledgements?|bibliography|"
    r"index|notes?|glossary|appendix|(?:a )?(?:short )?history\b.*|about the author|"
    r"illustrations?|transcriber'?s notes?|.*\b(?:article|obituaries|bibliography)\b.*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Extraction:
    title: str
    start_line: int
    end_line: int
    confidence: str
    boundary_method: str
    text: str


def normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def normalized_text_hash(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def clean_gutenberg_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    start = START_MARKER.search(text)
    if start:
        text = text[start.end() :]
    end = END_MARKER.search(text)
    if end:
        text = text[: end.start()]
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def normalize_heading(value: str) -> str:
    value = value.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", value.casefold()))


def toc_title(line: str) -> str | None:
    is_indented = bool(line[:1].isspace())
    value = " ".join(line.strip().split())
    if not value or len(value) > 140:
        return None
    value = re.split(r"\.{2,}", value, maxsplit=1)[0].strip(" ._-")
    value = re.sub(r"\s+(?:page\s+)?[ivxlcdm\d]+$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^\s*(?:\d+|[IVXLCDM]+)[.)]\s+", "", value)
    normalized = normalize_heading(value)
    if not normalized or len(normalized.split()) > 18 or len(normalized) < 3:
        return None
    if re.fullmatch(r"(?:[ivxlcdm]+|\d+)", normalized, re.IGNORECASE):
        return None
    if GENERIC_HEADING.match(normalized):
        return None
    if not any(character.isalpha() for character in value):
        return None
    # Capitalization is useful inside an identified TOC, but never establishes a body boundary.
    has_structure = (
        bool(re.search(r"\.{2,}", line))
        or is_indented
        or value.isupper()
        or value.istitle()
    )
    return value if has_structure else None


def find_toc_titles(lines: list[str]) -> tuple[list[str], int, int] | None:
    search_limit = min(len(lines), 3000)
    for toc_start in range(search_limit):
        if not TOC_HEADING.match(lines[toc_start]):
            continue
        titles: list[str] = []
        normalized_seen: set[str] = set()
        last_candidate_line = toc_start
        for line_index in range(toc_start + 1, min(len(lines), toc_start + 500)):
            title = toc_title(lines[line_index])
            if title is None:
                if titles and len(lines[line_index].strip()) > 180:
                    break
                continue
            if titles and re.search(r"\b(?:and|or|of|the|in|to|from)\s*$", titles[-1], re.I):
                previous = titles.pop()
                normalized_seen.discard(normalize_heading(previous))
                title = f"{previous} {title}"
            normalized = normalize_heading(title)
            if normalized in normalized_seen:
                break
            normalized_seen.add(normalized)
            titles.append(title)
            last_candidate_line = line_index
        if len(titles) >= 3:
            return titles, toc_start, last_candidate_line + 1
    return None


def extract_collection(text: str) -> tuple[list[Extraction], dict[str, Any]]:
    lines = text.splitlines()
    toc = find_toc_titles(lines)
    if toc is None:
        return [], {"confidence": "low", "reason": "no_reliable_table_of_contents"}
    titles, toc_start, toc_end = toc
    positions: list[tuple[str, int]] = []
    cursor = toc_end
    for title in titles:
        normalized_title = normalize_heading(title)
        match = None
        for line_index in range(cursor, len(lines)):
            if normalize_heading(lines[line_index].strip()) == normalized_title:
                match = line_index
                break
        if match is not None:
            positions.append((title, match))
            cursor = match + 1

    match_ratio = len(positions) / len(titles)
    non_story_title_count = sum(
        NON_STORY_TITLE.match(normalize_heading(title)) is not None for title in titles
    )
    if len(positions) < 2:
        confidence = "low"
    elif len(positions) < 3 or match_ratio < 0.7 or non_story_title_count >= 2:
        confidence = "medium"
    else:
        confidence = "high"

    extractions = []
    for index, (title, start_line) in enumerate(positions):
        end_line = positions[index + 1][1] if index + 1 < len(positions) else len(lines)
        story_text = "\n".join(lines[start_line + 1 : end_line]).strip()
        extractions.append(
            Extraction(
                title=title,
                start_line=start_line,
                end_line=end_line,
                confidence=confidence,
                boundary_method="toc_heading_match",
                text=story_text,
            )
        )
    evidence = {
        "confidence": confidence,
        "boundary_method": "toc_heading_match",
        "toc_start_line": toc_start,
        "toc_title_count": len(titles),
        "matched_heading_count": len(positions),
        "heading_match_ratio": match_ratio,
        "non_story_title_indicators": non_story_title_count,
    }
    return extractions, evidence


def classify_story_rejection(extraction: Extraction) -> str | None:
    if NON_STORY_TITLE.match(normalize_heading(extraction.title)):
        return "front_or_back_matter_title"
    words = extraction.text.split()
    if len(words) < MIN_STORY_WORDS:
        return "too_short_or_front_matter"
    if len(words) > MAX_STORY_WORDS:
        return "too_long_possible_novel_or_section"
    paragraphs = [part for part in extraction.text.split("\n\n") if part.strip()]
    if len(paragraphs) < 3:
        return "insufficient_prose_paragraphs"
    alphabetic = sum(character.isalpha() for character in extraction.text)
    printable = sum(not character.isspace() for character in extraction.text)
    if printable and alphabetic / printable < 0.65:
        return "damaged_ocr_or_nonprose"
    lines = [line.strip() for line in extraction.text.splitlines() if line.strip()]
    short_line_ratio = sum(len(line.split()) <= 8 for line in lines) / max(1, len(lines))
    if len(lines) >= 20 and short_line_ratio > 0.72:
        return "probable_poetry"
    return None


def author_is_safely_public_domain(author: str) -> bool:
    if not author.strip():
        return True
    if re.search(r"(?<!\d)(?:1[0-9]{3}|20[0-9]{2})\??-\s*(?:;|$)", author):
        return False
    years = [int(value) for value in re.findall(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)", author)]
    if not years:
        return any(term in author.lower() for term in ("anonymous", "unknown"))
    return max(years) <= 1955


def catalog_candidates(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    candidates = []
    for row in rows:
        searchable = f"{row['Subjects']} {row['Bookshelves']}".lower()
        title = row["Title"].lower()
        if row["Type"] != "Text" or row["Language"] != "en":
            continue
        if "short stories" not in searchable and "short stories" not in title:
            continue
        if re.search(r"\b(?:essays?|poetry|poems?)\b", searchable):
            continue
        if not author_is_safely_public_domain(row["Authors"]):
            continue
        candidates.append(row)
    target_terms = re.compile(
        r"detective|mystery|ghost|horror|science fiction|western|adventure|fantasy|weird|sea stor|short stor|tales|fables",
        re.IGNORECASE,
    )
    return sorted(
        candidates,
        key=lambda row: (
            0 if target_terms.search(f"{row['Title']} {row['Subjects']}") else 1,
            int(row["Text#"]),
        ),
    )


def download(url: str, path: Path, *, timeout: int = 120) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8-sig", errors="replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "FictionPulper/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return data.decode("utf-8-sig", errors="replace")


def seed_record(record: dict[str, Any]) -> dict[str, Any]:
    text_hash = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
    return {
        "id": record["id"],
        "source": record["source"],
        "source_id": record["id"],
        "source_collection": record.get("source_collection"),
        "title": record["title"],
        "author": record.get("author"),
        "publication_year": None,
        "genres": record.get("genres", []),
        "rights": record["rights"],
        "extraction_confidence": "high",
        "boundary_method": "seed_corpus",
        "text_hash": text_hash,
        "words": record.get("words", len(record["text"].split())),
        "text": record["text"],
    }


def make_story_record(
    extraction: Extraction, collection: dict[str, str]
) -> dict[str, Any]:
    book_id = collection["Text#"]
    text_hash = hashlib.sha256(extraction.text.encode("utf-8")).hexdigest()
    stable = hashlib.sha256(
        f"{book_id}\0{normalize_heading(extraction.title)}\0{text_hash}".encode("utf-8")
    ).hexdigest()[:12]
    genres = [
        subject.strip()
        for subject in collection["Subjects"].split(";")
        if subject.strip() and len(subject.strip()) <= 100
    ]
    return {
        "id": f"pg-{book_id}-{stable}",
        "source": "project_gutenberg",
        "source_id": f"pg-{book_id}",
        "source_collection": " ".join(collection["Title"].split()),
        "title": extraction.title,
        "author": collection["Authors"] or None,
        "publication_year": None,
        "genres": genres,
        "rights": "public_domain",
        "extraction_confidence": extraction.confidence,
        "boundary_method": extraction.boundary_method,
        "source_start_heading": extraction.title,
        "text_hash": text_hash,
        "words": len(extraction.text.split()),
        "text": extraction.text,
    }


def review_sample(
    extraction: Extraction, collection: dict[str, str], lines: list[str]
) -> dict[str, Any]:
    preceding = "\n".join(lines[max(0, extraction.start_line - 8) : extraction.start_line])
    following = "\n".join(lines[extraction.end_line : extraction.end_line + 8])
    return {
        "source_collection_id": f"pg-{collection['Text#']}",
        "source_collection": " ".join(collection["Title"].split()),
        "detected_title": extraction.title,
        "preceding_text": preceding[-800:],
        "story_opening": extraction.text[:800],
        "story_ending": extraction.text[-800:],
        "following_text": following[:800],
        "boundary_method": extraction.boundary_method,
        "confidence": extraction.confidence,
    }


def simhash64(text: str) -> int:
    words = normalized_text(text).split()
    shingles = (" ".join(words[index : index + 5]) for index in range(max(1, len(words) - 4)))
    vector = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest())
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    return sum(1 << bit for bit, score in enumerate(vector) if score >= 0)


def shingle_jaccard(left: str, right: str) -> float:
    def shingles(text: str) -> set[str]:
        words = normalized_text(text).split()
        return {" ".join(words[index : index + 5]) for index in range(max(1, len(words) - 4))}

    left_set = shingles(left)
    right_set = shingles(right)
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def near_duplicate_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hashes = [simhash64(record["text"]) for record in records]
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    pairs: set[tuple[int, int]] = set()
    for index, value in enumerate(hashes):
        for band in range(4):
            key = (band, (value >> (band * 16)) & 0xFFFF)
            for previous in buckets[key]:
                pairs.add((previous, index))
            buckets[key].append(index)
    report = []
    for left_index, right_index in sorted(pairs):
        left = records[left_index]
        right = records[right_index]
        length_ratio = min(left["words"], right["words"]) / max(left["words"], right["words"])
        if length_ratio < 0.8:
            continue
        hamming = (hashes[left_index] ^ hashes[right_index]).bit_count()
        if hamming > 8:
            continue
        similarity = shingle_jaccard(left["text"], right["text"])
        if similarity < 0.82:
            continue
        report.append(
            {
                "left_id": left["id"],
                "right_id": right["id"],
                "left_title": left["title"],
                "right_title": right["title"],
                "left_source": left["source_id"],
                "right_source": right["source_id"],
                "simhash_hamming_distance": hamming,
                "word_5gram_jaccard": similarity,
                "length_ratio": length_ratio,
                "action": "review_only",
            }
        )
    return report


def percentile(values: list[int], percentile_value: int) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile_value / 100 * len(ordered)) - 1)]


def corpus_diagnostics(
    records: list[dict[str, Any]],
    tokenizer: Tokenizer,
    *,
    counters: Counter[str],
    near_duplicates: list[dict[str, Any]],
    collections_processed: int,
) -> dict[str, Any]:
    word_counts = [record["words"] for record in records]
    token_count = sum(
        len(tokenizer.encode(f"{record['title']}\n\n{record['text']}").ids) + 3
        for record in records
    )
    publication_years = Counter(
        record["publication_year"]
        for record in records
        if record.get("publication_year") is not None
    )
    confidence = Counter(record["extraction_confidence"] for record in records)
    return {
        "preprocessing_version": PREPROCESSING_VERSION,
        "unique_stories": len(records),
        "unique_authors": len({record["author"] for record in records if record.get("author")}),
        "source_collections": len(
            {record["source_collection"] for record in records if record.get("source_collection")}
        ),
        "collections_processed": collections_processed,
        "word_count": sum(word_counts),
        "token_estimate_current_smoke_tokenizer": token_count,
        "tokenizer_retrained": False,
        "genre_counts": dict(
            sorted(Counter(genre for record in records for genre in record["genres"]).items())
        ),
        "publication_year_distribution": dict(sorted(publication_years.items())),
        "unknown_publication_year_count": sum(
            record.get("publication_year") is None for record in records
        ),
        "minimum_story_words": min(word_counts),
        "median_story_words": statistics.median(word_counts),
        "mean_story_words": statistics.mean(word_counts),
        "percentile_95_story_words": percentile(word_counts, 95),
        "maximum_story_words": max(word_counts),
        "extraction_confidence_distribution": dict(sorted(confidence.items())),
        "source_counts": dict(sorted(Counter(record["source"] for record in records).items())),
        "boundary_method_counts": dict(
            sorted(Counter(record["boundary_method"] for record in records).items())
        ),
        "exact_duplicates_rejected": counters["exact_duplicates_rejected"],
        "near_duplicate_candidates": len(near_duplicates),
        "low_confidence_extractions_rejected": counters["low_confidence_collections"],
        "medium_confidence_review_items": counters["medium_review_items"],
        "rejection_counts": dict(sorted(counters.items())),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def validate_corpus_records(
    records: list[dict[str, Any]], *, expected_seed_ids: set[str] | None = None
) -> None:
    if not records:
        raise ValueError("Corpus must contain at least one record")
    required_strings = (
        "id",
        "source",
        "source_id",
        "title",
        "rights",
        "extraction_confidence",
        "boundary_method",
        "text_hash",
        "text",
    )
    ids: set[str] = set()
    normalized_hashes: set[str] = set()
    for index, record in enumerate(records, start=1):
        for field in required_strings:
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Record {index} has invalid {field!r}")
        if record["id"] in ids:
            raise ValueError(f"Duplicate corpus id: {record['id']}")
        ids.add(record["id"])
        if record["extraction_confidence"] != "high":
            raise ValueError(f"Non-high-confidence record admitted: {record['id']}")
        if not isinstance(record.get("genres"), list):
            raise ValueError(f"Record {record['id']} has invalid genres")
        if not isinstance(record.get("words"), int) or record["words"] <= 0:
            raise ValueError(f"Record {record['id']} has invalid word count")
        actual_text_hash = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
        if record["text_hash"] != actual_text_hash:
            raise ValueError(f"Record {record['id']} has an incorrect text hash")
        normalized_hash = normalized_text_hash(record["text"])
        if normalized_hash in normalized_hashes:
            raise ValueError(f"Normalized-text duplicate admitted: {record['id']}")
        normalized_hashes.add(normalized_hash)
    if expected_seed_ids is not None and not expected_seed_ids.issubset(ids):
        missing = sorted(expected_seed_ids - ids)
        raise ValueError(f"Corpus does not preserve seed IDs: {missing[:5]}")


def build_corpus_v1(
    *,
    seed_corpus_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
    raw_dir: Path,
    target_tokens: int,
    max_collections: int,
    request_delay: float,
) -> dict[str, Any]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    seed = [seed_record(record) for record in load_corpus(seed_corpus_path)]
    accepted = list(seed)
    additions: list[dict[str, Any]] = []
    medium_review: list[dict[str, Any]] = []
    low_diagnostics: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    exact_duplicates: list[dict[str, Any]] = []
    review_samples: list[dict[str, Any]] = []
    review_sample_collections: Counter[str] = Counter()
    counters: Counter[str] = Counter()
    exact_hashes = {normalized_text_hash(record["text"]): record for record in accepted}

    catalog_path = raw_dir / "pg_catalog.csv"
    catalog_text = download(CATALOG_URL, catalog_path)
    candidates = catalog_candidates(csv.DictReader(io.StringIO(catalog_text)))
    collections_processed = 0
    estimated_tokens = sum(
        len(tokenizer.encode(f"{record['title']}\n\n{record['text']}").ids) + 3
        for record in accepted
    )

    for collection in candidates[:max_collections]:
        if estimated_tokens >= target_tokens:
            break
        book_id = collection["Text#"]
        raw_path = raw_dir / "texts" / f"pg-{book_id}.txt"
        try:
            raw_text = download(TEXT_URL.format(book_id=book_id), raw_path)
        except (urllib.error.URLError, TimeoutError, UnicodeError) as error:
            counters["download_failures"] += 1
            rejections.append(
                {"source_collection_id": f"pg-{book_id}", "reason": "download_failure", "error": str(error)}
            )
            continue
        collections_processed += 1
        clean_text = clean_gutenberg_text(raw_text)
        extractions, evidence = extract_collection(clean_text)
        if evidence["confidence"] == "low":
            counters["low_confidence_collections"] += 1
            low_diagnostics.append(
                {
                    "source_collection_id": f"pg-{book_id}",
                    "source_collection": " ".join(collection["Title"].split()),
                    **evidence,
                }
            )
            continue
        lines = clean_text.splitlines()
        for extraction in extractions:
            sample = review_sample(extraction, collection, lines)
            collection_key = sample["source_collection_id"]
            if review_sample_collections[collection_key] < 2:
                review_samples.append(sample)
                review_sample_collections[collection_key] += 1
            if extraction.confidence == "medium":
                counters["medium_review_items"] += 1
                medium_review.append(sample)
                continue
            rejection_reason = classify_story_rejection(extraction)
            if rejection_reason:
                counters[rejection_reason] += 1
                rejections.append(
                    {
                        "source_collection_id": f"pg-{book_id}",
                        "title": extraction.title,
                        "reason": rejection_reason,
                        "words": len(extraction.text.split()),
                    }
                )
                continue
            record = make_story_record(extraction, collection)
            normalized_hash = normalized_text_hash(record["text"])
            if normalized_hash in exact_hashes:
                canonical = exact_hashes[normalized_hash]
                counters["exact_duplicates_rejected"] += 1
                exact_duplicates.append(
                    {
                        "canonical_id": canonical["id"],
                        "duplicate_id": record["id"],
                        "normalized_text_hash": normalized_hash,
                        "canonical_title": canonical["title"],
                        "duplicate_title": record["title"],
                        "canonical_author": canonical.get("author"),
                        "duplicate_author": record.get("author"),
                        "canonical_source": canonical["source_id"],
                        "duplicate_source": record["source_id"],
                    }
                )
                continue
            exact_hashes[normalized_hash] = record
            accepted.append(record)
            additions.append(record)
            estimated_tokens += (
                len(tokenizer.encode(f"{record['title']}\n\n{record['text']}").ids) + 3
            )
        if collections_processed % 25 == 0:
            print(
                f"collections={collections_processed} stories={len(accepted)} "
                f"estimated_tokens={estimated_tokens:,}"
            )
        if request_delay:
            time.sleep(request_delay)

    near_duplicates = near_duplicate_candidates(accepted)
    validate_corpus_records(accepted, expected_seed_ids={record["id"] for record in seed})
    stats = corpus_diagnostics(
        accepted,
        tokenizer,
        counters=counters,
        near_duplicates=near_duplicates,
        collections_processed=collections_processed,
    )
    stats.update(
        {
            "seed_corpus_path": str(seed_corpus_path),
            "seed_corpus_sha256": sha256_file(seed_corpus_path),
            "seed_story_count": len(seed),
            "new_story_count": len(additions),
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "catalog_url": CATALOG_URL,
            "catalog_sha256": sha256_file(catalog_path),
            "target_tokens": target_tokens,
            "stop_reason": (
                "target_reached"
                if stats["token_estimate_current_smoke_tokenizer"] >= target_tokens
                else "collection_limit_or_reliability_review_required"
            ),
        }
    )
    write_jsonl(output_dir / "corpus.jsonl", accepted)
    write_jsonl(output_dir / "additions.jsonl", additions)
    write_jsonl(output_dir / "medium-review.jsonl", medium_review)
    write_jsonl(output_dir / "low-confidence.jsonl", low_diagnostics)
    write_jsonl(output_dir / "rejections.jsonl", rejections)
    write_jsonl(output_dir / "exact-duplicates.jsonl", exact_duplicates)
    write_jsonl(output_dir / "near-duplicates.jsonl", near_duplicates)
    write_jsonl(output_dir / "review-samples.jsonl", review_samples)
    write_json_atomic(output_dir / "stats.json", stats)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-corpus", type=Path, default=Path("data/corpus/stories.jsonl"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/tokenizer.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/corpus_v1"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/gutenberg"))
    parser.add_argument("--target-tokens", type=int, default=10_000_000)
    parser.add_argument("--max-collections", type=int, default=1000)
    parser.add_argument("--request-delay", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_corpus_v1(
        seed_corpus_path=args.seed_corpus,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
        target_tokens=args.target_tokens,
        max_collections=args.max_collections,
        request_delay=args.request_delay,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
