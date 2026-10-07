"""Acquire, conservatively clean, and validate FictionPulper source fiction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi


SHORT_STORY_SOURCE = "short-stories"
GUTENBERG_BOOK_SOURCE = "gutenberg-books"
DEFAULT_SOURCE = SHORT_STORY_SOURCE

SHORT_STORY_DATASET = "Travis-ML/ShortStory-SFT-jsonl"
GUTENBERG_BOOK_DATASET = "ppirli/Gutenberg-Fiction"
DEFAULT_CONFIG = "default"
DEFAULT_SPLIT = "train"
DEFAULT_SEED = 1337
SPLIT_RATIOS = {"train": 0.85, "validation": 0.075, "test": 0.075}

SHORT_STORY_FIELDS = {
    "id",
    "author",
    "book",
    "title",
    "words",
    "kind",
    "genres",
    "text",
    "license",
}
GUTENBERG_BOOK_FIELDS = {"etextno", "book_title", "author", "issued", "text"}
SHORT_STORY_CANONICAL_FIELDS = {
    "id",
    "source",
    "source_collection",
    "title",
    "author",
    "genres",
    "words",
    "rights",
    "text",
}
GUTENBERG_BOOK_CANONICAL_FIELDS = {
    "id",
    "source",
    "title",
    "author",
    "genres",
    "publication_year",
    "rights",
    "text",
}

GENRE_TITLE_PATTERNS: dict[str, re.Pattern[str]] = {
    "crime": re.compile(r"\b(crime|criminal|murder|murderer|thief|robbery)\b", re.I),
    "noir": re.compile(r"\bnoir\b", re.I),
    "mystery": re.compile(r"\b(mystery|mysteries|detective|sleuth|sherlock|clue)\b", re.I),
    "horror": re.compile(r"\b(horror|ghost|haunted|vampire|terror|supernatural)\b", re.I),
    "science_fiction": re.compile(
        r"\b(science fiction|martian|mars|interplanetary|space voyage|time machine)\b",
        re.I,
    ),
    "western": re.compile(r"\b(western|cowboy|ranch|frontier|prairie)\b", re.I),
    "adventure": re.compile(r"\b(adventure|adventures|pirate|buccaneer|jungle|sea tale)\b", re.I),
    "fantasy": re.compile(r"\b(fantasy|fairy|wizard|magic|magical|enchanted)\b", re.I),
    "weird": re.compile(r"\b(weird|occult|eldritch)\b", re.I),
}

START_MARKER = re.compile(
    r"^\s*\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$",
    re.IGNORECASE | re.MULTILINE,
)
END_MARKER = re.compile(
    r"^\s*\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    required_fields: set[str]
    canonical_fields: set[str]
    canonicalize: Callable[[Mapping[str, Any]], dict[str, Any]]


def clean_story_text(text: str) -> str:
    """Normalize transport artifacts while leaving internal prose and paragraphs untouched."""
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()


def remove_obvious_story_boilerplate(text: str) -> str:
    """Drop only paragraphs that are unambiguously transcription or publisher artifacts."""
    retained = []
    for paragraph in text.split("\n\n"):
        normalized = " ".join(paragraph.split())
        is_boilerplate = (
            normalized.startswith("Note: Project Gutenberg also has an HTML version")
            or normalized.startswith("Produced by ")
            or normalized.startswith("Penguin Books Ltd, Harmondsworth")
            or (
                normalized.startswith("PAGE I.")
                and bool(re.search(r"\bII\.\s+.*\bIII\.", normalized))
            )
        )
        if not is_boilerplate:
            retained.append(paragraph)
    return "\n\n".join(retained).strip()


def clean_book_text(text: str) -> str:
    """Remove explicit Gutenberg wrappers and normalize whitespace conservatively."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    start = START_MARKER.search(text)
    if start:
        text = text[start.end() :]
    end = END_MARKER.search(text)
    if end:
        text = text[: end.start()]
    lines = [line.rstrip() for line in text.splitlines()]
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", "\n".join(lines)).strip()


def infer_book_genres(title: str) -> list[str]:
    """Infer only labels explicitly signaled by a whole-book title."""
    return [genre for genre, pattern in GENRE_TITLE_PATTERNS.items() if pattern.search(title)]


def canonicalize_short_story(source_record: Mapping[str, Any]) -> dict[str, Any]:
    require_source_fields(source_record, SHORT_STORY_FIELDS)
    if source_record["kind"] != "story":
        raise ValueError(f"source kind must be 'story', got {source_record['kind']!r}")
    license_value = source_record["license"]
    if not isinstance(license_value, str) or "public domain" not in license_value.lower():
        raise ValueError(f"source license is not marked public domain: {license_value!r}")

    text_value = source_record["text"]
    if not isinstance(text_value, str):
        raise ValueError("source text must be a string")
    title = source_record["title"]
    if not isinstance(title, str) or not title.strip():
        raise ValueError("source title must be a non-empty string")
    genres = source_record["genres"]
    if not isinstance(genres, list) or any(not isinstance(genre, str) for genre in genres):
        raise ValueError("source genres must be a list of strings")

    normalized_text = clean_story_text(text_value)
    source_words = source_record["words"]
    if not isinstance(source_words, int) or source_words != len(normalized_text.split()):
        raise ValueError("source story word count does not match source text")
    cleaned_text = remove_obvious_story_boilerplate(normalized_text)
    return {
        "id": str(source_record["id"]).strip(),
        "source": SHORT_STORY_DATASET,
        "source_collection": str(source_record["book"]).strip() or None,
        "title": title.strip(),
        "author": str(source_record["author"]).strip() or None,
        "genres": [genre.strip() for genre in genres if genre.strip()],
        "words": len(cleaned_text.split()),
        "rights": "public_domain",
        "text": cleaned_text,
    }


def canonicalize_gutenberg_book(source_record: Mapping[str, Any]) -> dict[str, Any]:
    require_source_fields(source_record, GUTENBERG_BOOK_FIELDS)
    text_value = source_record["text"]
    if not isinstance(text_value, str):
        raise ValueError("source text must be a string")
    title = str(source_record["book_title"] or "").strip()
    return {
        "id": f"pg-{source_record['etextno']}",
        "source": "project_gutenberg",
        "title": title or None,
        "author": str(source_record["author"] or "").strip() or None,
        "genres": infer_book_genres(title),
        "publication_year": None,
        "rights": "public_domain_us",
        "text": clean_book_text(text_value),
    }


SOURCE_SPECS = {
    SHORT_STORY_SOURCE: SourceSpec(
        dataset=SHORT_STORY_DATASET,
        required_fields=SHORT_STORY_FIELDS,
        canonical_fields=SHORT_STORY_CANONICAL_FIELDS,
        canonicalize=canonicalize_short_story,
    ),
    GUTENBERG_BOOK_SOURCE: SourceSpec(
        dataset=GUTENBERG_BOOK_DATASET,
        required_fields=GUTENBERG_BOOK_FIELDS,
        canonical_fields=GUTENBERG_BOOK_CANONICAL_FIELDS,
        canonicalize=canonicalize_gutenberg_book,
    ),
}


def require_source_fields(record: Mapping[str, Any], required_fields: set[str]) -> None:
    missing = required_fields - record.keys()
    if missing:
        raise ValueError(f"source record is missing fields: {sorted(missing)}")


def validate_schema(features: Mapping[str, Any], required_fields: set[str]) -> set[str]:
    names = set(features)
    missing = required_fields - names
    if missing:
        raise RuntimeError(
            f"Unsupported dataset schema. Expected at least {sorted(required_fields)}, "
            f"observed {sorted(names)}, missing {sorted(missing)}."
        )
    return names


def validate_record(record: Mapping[str, Any], source: str) -> None:
    spec = SOURCE_SPECS[source]
    if set(record) != spec.canonical_fields:
        missing = spec.canonical_fields - record.keys()
        extra = record.keys() - spec.canonical_fields
        raise ValueError(f"invalid canonical fields; missing={sorted(missing)}, extra={sorted(extra)}")
    if not isinstance(record["id"], str) or not record["id"].strip():
        raise ValueError("record id must be a non-empty string")
    if not isinstance(record["text"], str) or not record["text"].strip():
        raise ValueError("record text must be a non-empty string")
    if "\x00" in record["text"]:
        raise ValueError("record text contains a NUL character")
    if not isinstance(record["genres"], list) or any(
        not isinstance(genre, str) or not genre for genre in record["genres"]
    ):
        raise ValueError("record contains an invalid genre list")

    if source == SHORT_STORY_SOURCE:
        if not isinstance(record["title"], str) or not record["title"].strip():
            raise ValueError("story title must be a non-empty string")
        if not isinstance(record["words"], int) or record["words"] <= 0:
            raise ValueError("story words must be a positive integer")
        actual_words = len(record["text"].split())
        if record["words"] != actual_words:
            raise ValueError(
                f"story word count mismatch: source={record['words']}, actual={actual_words}"
            )


def resolve_revision(dataset_name: str, requested_revision: str | None) -> str:
    if requested_revision:
        return requested_revision
    return HfApi().dataset_info(dataset_name).sha


def load_source_dataset(
    dataset_name: str,
    config: str,
    split: str,
    revision: str,
) -> Dataset:
    dataset = load_dataset(
        dataset_name,
        name=config,
        split=split,
        revision=revision,
        streaming=False,
    )
    if not isinstance(dataset, Dataset):
        raise RuntimeError(f"Expected a Dataset for split {split!r}")
    return dataset


def prepare_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    source: str,
    include_unclassified_books: bool = False,
    max_documents: int | None = None,
) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    spec = SOURCE_SPECS[source]
    records: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    ids: set[str] = set()
    records_by_text_hash: dict[str, dict[str, Any]] = {}
    duplicate_report: list[dict[str, Any]] = []

    for row in rows:
        counters["source_records"] += 1
        try:
            record = spec.canonicalize(row)
            validate_record(record, source)
        except (TypeError, ValueError):
            counters["invalid_records"] += 1
            continue

        if record["id"] in ids:
            counters["duplicate_ids"] += 1
            continue
        text_hash = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
        if text_hash in records_by_text_hash:
            canonical = records_by_text_hash[text_hash]
            counters["duplicate_texts"] += 1
            duplicate_report.append(
                {
                    "canonical_id": canonical["id"],
                    "duplicate_id": record["id"],
                    "text_hash": text_hash,
                    "canonical_title": canonical.get("title"),
                    "duplicate_title": record.get("title"),
                    "canonical_author": canonical.get("author"),
                    "duplicate_author": record.get("author"),
                    "metadata_conflict": (
                        canonical.get("title") != record.get("title")
                        or canonical.get("author") != record.get("author")
                    ),
                }
            )
            continue
        if (
            source == GUTENBERG_BOOK_SOURCE
            and not record["genres"]
            and not include_unclassified_books
        ):
            counters["unclassified_books"] += 1
            continue

        ids.add(record["id"])
        records_by_text_hash[text_hash] = record
        records.append(record)
        if max_documents is not None and len(records) >= max_documents:
            break

    counters["accepted_documents"] = len(records)
    return records, counters, duplicate_report


def assign_document_splits(records: list[dict[str, Any]], seed: int) -> dict[str, str]:
    """Assign exact split sizes using a stable seeded hash of each document ID."""
    ordered = sorted(
        records,
        key=lambda record: hashlib.sha256(f"{seed}:{record['id']}".encode("utf-8")).digest(),
    )
    total = len(ordered)
    train_count = round(total * SPLIT_RATIOS["train"])
    validation_count = round(total * SPLIT_RATIOS["validation"])
    boundaries = (train_count, train_count + validation_count)
    assignments: dict[str, str] = {}
    for index, record in enumerate(ordered):
        if index < boundaries[0]:
            split = "train"
        elif index < boundaries[1]:
            split = "validation"
        else:
            split = "test"
        assignments[record["id"]] = split
    return assignments


def split_statistics(
    records: list[dict[str, Any]], assignments: Mapping[str, str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLIT_RATIOS:
        split_records = [record for record in records if assignments[record["id"]] == split]
        result[split] = {
            "documents": len(split_records),
            "words": sum(record.get("words", len(record["text"].split())) for record in split_records),
            "genre_counts": dict(
                sorted(Counter(genre for record in split_records for genre in record["genres"]).items())
            ),
        }
    return result


def corpus_statistics(
    records: list[dict[str, Any]],
    counters: Counter[str],
    *,
    source: str,
    dataset_name: str,
    config: str,
    split: str,
    revision: str,
    output_path: Path,
) -> dict[str, Any]:
    character_lengths = [len(record["text"]) for record in records]
    stats: dict[str, Any] = {
        "preprocessing_version": 4,
        "source_adapter": source,
        "dataset": dataset_name,
        "dataset_revision": revision,
        "config": config,
        "split": split,
        "output": str(output_path),
        "story_count": len(records),
        "character_count": sum(character_lengths),
        "author_counts": dict(sorted(Counter(record["author"] for record in records).items())),
        "genre_counts": dict(
            sorted(Counter(genre for record in records for genre in record["genres"]).items())
        ),
        "filter_counts": dict(sorted(counters.items())),
    }
    if source == SHORT_STORY_SOURCE:
        word_lengths = [record["words"] for record in records]
        stats.update(
            {
                "total_words": sum(word_lengths),
                "min_story_words": min(word_lengths),
                "median_story_words": int(statistics.median(word_lengths)),
                "max_story_words": max(word_lengths),
                "source_collection_count": len(
                    {record["source_collection"] for record in records}
                ),
            }
        )
    return stats


def write_corpus(output_path: Path, records: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    if not records:
        raise RuntimeError("No documents passed validation; no corpus was written.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    temporary_path.replace(output_path)

    stats_path = output_path.with_suffix(".stats.json")
    temporary_stats_path = stats_path.with_suffix(stats_path.suffix + ".tmp")
    with temporary_stats_path.open("w", encoding="utf-8") as stats_file:
        json.dump(stats, stats_file, indent=2, ensure_ascii=False)
        stats_file.write("\n")
        stats_file.flush()
        os.fsync(stats_file.fileno())
    temporary_stats_path.replace(stats_path)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False, sort_keys=True)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCE_SPECS), default=DEFAULT_SOURCE)
    parser.add_argument("--dataset", help="Override the selected source adapter's dataset")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--revision", help="Hub revision; defaults to the current commit SHA")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=Path("data/corpus/stories.jsonl"))
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--include-unclassified-books", action="store_true")
    args = parser.parse_args()
    if args.max_documents is not None and args.max_documents <= 0:
        parser.error("--max-documents must be positive")
    return args


def main() -> None:
    args = parse_args()
    spec = SOURCE_SPECS[args.source]
    dataset_name = args.dataset or spec.dataset
    revision = resolve_revision(dataset_name, args.revision)
    dataset = load_source_dataset(dataset_name, args.config, args.split, revision)
    observed_schema = validate_schema(dataset.features, spec.required_fields)
    print(f"dataset: {dataset_name}/{args.config}@{args.split}")
    print(f"revision: {revision}")
    print(f"observed schema: {sorted(observed_schema)}")

    records, counters, duplicate_report = prepare_records(
        dataset,
        source=args.source,
        include_unclassified_books=args.include_unclassified_books,
        max_documents=args.max_documents,
    )
    stats = corpus_statistics(
        records,
        counters,
        source=args.source,
        dataset_name=dataset_name,
        config=args.config,
        split=args.split,
        revision=revision,
        output_path=args.output,
    )
    assignments = assign_document_splits(records, args.seed)
    split_path = args.output.with_suffix(".splits.json")
    duplicate_path = args.output.with_suffix(".duplicates.json")
    stats["split_seed"] = args.seed
    stats["split_ratios"] = SPLIT_RATIOS
    stats["split_path"] = str(split_path)
    stats["split_statistics"] = split_statistics(records, assignments)
    stats["duplicate_report_path"] = str(duplicate_path)
    write_corpus(args.output, records, stats)
    write_json_atomic(
        split_path,
        {
            "preprocessing_version": 4,
            "seed": args.seed,
            "ratios": SPLIT_RATIOS,
            "assignments": dict(sorted(assignments.items())),
        },
    )
    write_json_atomic(
        duplicate_path,
        {
            "preprocessing_version": 4,
            "dataset": dataset_name,
            "dataset_revision": revision,
            "duplicate_count": len(duplicate_report),
            "duplicates": duplicate_report,
        },
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
