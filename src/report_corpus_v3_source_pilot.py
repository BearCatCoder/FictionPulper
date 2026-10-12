"""Run the hash-pinned Corpus-v3 bounded source-yield pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from tokenizers import Tokenizer

from src.audit_corpus_v2 import verify_locked_inputs
from src.corpus_v1 import clean_gutenberg_text, normalized_text_hash
from src.corpus_v2 import (
    SourceArtifact,
    broad_genres,
    classify_story_rejection,
    exact_document_tokens,
    extract_collection_v2,
    make_story_record,
    near_duplicate_graph,
)
from src.corpus_v3 import acquire_pinned_story, load_jsonl, load_ledger, save_ledger
from src.train_tokenizer import write_json_atomic


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _candidate_inventory_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _count(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "records": len(rows),
        "tokenizer_v1_tokens": sum(int(row["tokenizer_v1_tokens"]) for row in rows),
    }


def _concentration(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    tokens: Counter[str] = Counter()
    records: Counter[str] = Counter()
    for row in rows:
        value = str(row.get(field) or "unknown")
        records[value] += 1
        tokens[value] += int(row["tokenizer_v1_tokens"])
    total_tokens = sum(tokens.values())
    total_records = sum(records.values())
    ordered = sorted(tokens, key=lambda value: (-tokens[value], value))
    return {
        "unique_count": len(ordered),
        "largest_token_share": tokens[ordered[0]] / total_tokens if ordered else 1.0,
        "largest_record_share": max(records.values()) / total_records if records else 1.0,
        "top": [
            {
                "value": value,
                "records": records[value],
                "tokenizer_v1_tokens": tokens[value],
            }
            for value in ordered
        ],
    }


def _load_catalog(path: Path) -> dict[str, dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    return {f"pg-{row['Text#']}": row for row in csv.DictReader(io.StringIO(text))}


def _source_summary(
    source_id: str,
    rows: list[dict[str, Any]],
    artifact: dict[str, Any],
    catalog: dict[str, str],
) -> dict[str, Any]:
    gross = _count(rows)
    exact = _count([row for row in rows if row.get("exact_duplicate_of")])
    quality = Counter(str(row["quality_diagnostic"] or "otherwise_eligible") for row in rows)
    quality_tokens: Counter[str] = Counter()
    for row in rows:
        quality_tokens[str(row["quality_diagnostic"] or "otherwise_eligible")] += int(
            row["tokenizer_v1_tokens"]
        )
    genres: Counter[str] = Counter()
    genre_tokens: Counter[str] = Counter()
    for row in rows:
        for genre in row["genres"] or ["unclassified"]:
            genres[genre] += 1
            genre_tokens[genre] += int(row["tokenizer_v1_tokens"])
    return {
        "source_id": source_id,
        "title": " ".join(catalog["Title"].split()),
        "creator_catalog_value": catalog.get("Authors") or None,
        "source_url": artifact["url"],
        "source_sha256": artifact["source_sha256"],
        "source_bytes": artifact["byte_count"],
        "gross_extracted": gross,
        "quality_diagnostics": {
            reason: {"records": quality[reason], "tokenizer_v1_tokens": quality_tokens[reason]}
            for reason in sorted(quality)
        },
        "exact_duplicate_loss": exact,
        "duplicate_adjusted": {
            "records": gross["records"] - exact["records"],
            "tokenizer_v1_tokens": gross["tokenizer_v1_tokens"] - exact["tokenizer_v1_tokens"],
            "qualification": "Exact-text adjustment only; unresolved fuzzy matches are not silently excluded.",
        },
        "unresolved_near_duplicate_records": sum(bool(row["near_duplicate_ids"]) for row in rows),
        "genre_contribution_before_evidence_gates": {
            genre: {"records": genres[genre], "tokenizer_v1_tokens": genre_tokens[genre]}
            for genre in sorted(genres)
        },
        "rights_review": "not_completed",
        "boundary_review": "not_completed",
        "accepted_incremental_unique": {"records": 0, "tokenizer_v1_tokens": 0},
        "first_rejection_reason": "incomplete_rights_evidence",
    }


def generate_source_pilot_report(
    config_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise ValueError("Corpus-v3 source pilot config must declare version: 1")
    actual_hashes = verify_locked_inputs(config["inputs"])
    paths = {name: Path(spec["path"]) for name, spec in config["inputs"].items()}
    shortfall = _json(paths["shortfall_report"])
    ranked = shortfall["source_yield_hypotheses"][
        "existing_medium_confidence_gutenberg_collections"
    ]["top_catalogs_by_extracted_candidates"]
    selected_ids = [str(item["source_id"]) for item in config["sources"]]
    expected_ids = [str(item["source_id"]) for item in ranked[: len(selected_ids)]]
    if selected_ids != expected_ids:
        raise RuntimeError(f"Pilot sources no longer match issue #20 priority: {expected_ids}")

    source_manifest = {
        str(row["source_id"]): row for row in load_jsonl(paths["source_manifest"])
    }
    historical_summaries = {
        str(row["source_id"]): row for row in load_jsonl(paths["source_summaries"])
    }
    catalog = _load_catalog(paths["catalog"])
    tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
    base = load_jsonl(paths["corpus_v2"])
    base_exact = {normalized_text_hash(str(row["text"])): str(row["id"]) for row in base}

    cache_dir = Path(config["acquisition"]["cache_dir"])
    ledger_path = Path(config["acquisition"]["ledger_path"])
    ledger = load_ledger(ledger_path)
    candidates: list[dict[str, Any]] = []
    artifact_rows: dict[str, dict[str, Any]] = {}
    selection = config["selection"]
    for source in config["sources"]:
        source_id = str(source["source_id"])
        manifest = source_manifest.get(source_id)
        summary = historical_summaries.get(source_id)
        if manifest is None or summary is None or source_id not in catalog:
            raise RuntimeError(f"Missing locked evidence for {source_id}")
        if summary.get("extraction_confidence") != "medium":
            raise RuntimeError(f"{source_id} is no longer a medium-confidence review source")
        if manifest["sha256"] != summary["raw_sha256"] or manifest["url"] != summary["source_url"]:
            raise RuntimeError(f"Source manifest and summary disagree for {source_id}")
        raw_text, ledger_row, _ = acquire_pinned_story(
            {
                "id": source_id,
                "source_url": manifest["url"],
                "source_sha256": manifest["sha256"],
                "encoding": "utf-8-sig",
            },
            cache_dir=cache_dir,
            ledger=ledger,
            timeout=int(config["acquisition"]["timeout_seconds"]),
        )
        save_ledger(ledger_path, ledger)
        artifact_rows[source_id] = {
            "url": manifest["url"],
            "source_sha256": manifest["sha256"],
            "byte_count": int(ledger_row["byte_count"]),
        }
        extractions, evidence = extract_collection_v2(
            clean_gutenberg_text(raw_text),
            minimum_matches=int(selection["minimum_toc_matches"]),
            high_match_ratio=float(selection["high_confidence_match_ratio"]),
        )
        if evidence.get("confidence") != "medium":
            raise RuntimeError(f"Extraction confidence changed for {source_id}: {evidence}")
        if len(extractions) != int(summary["extracted_candidates"]):
            raise RuntimeError(f"Extracted candidate count changed for {source_id}")
        artifact = SourceArtifact(
            source_id=source_id,
            url=manifest["url"],
            path=str(ledger_row["cache_path"]),
            sha256=manifest["sha256"],
            byte_count=int(ledger_row["byte_count"]),
            retrieval_method="verified_pilot_cache",
            retrieved_at="omitted_from_deterministic_report",
        )
        for extraction in extractions:
            record = make_story_record(extraction, catalog[source_id], artifact, evidence)
            record["extraction_confidence"] = extraction.confidence
            record["tokenizer_v1_tokens"] = exact_document_tokens(tokenizer, record)
            record["quality_diagnostic"] = classify_story_rejection(
                extraction,
                minimum_words=int(selection["minimum_story_words"]),
                maximum_words=int(selection["maximum_story_words"]),
            )
            candidates.append(record)

    seen = dict(base_exact)
    for record in candidates:
        digest = normalized_text_hash(record["text"])
        record["exact_duplicate_of"] = seen.get(digest)
        if digest not in seen:
            seen[digest] = record["id"]

    unique_pilot = [record for record in candidates if not record["exact_duplicate_of"]]
    _, clusters = near_duplicate_graph(
        [*base, *unique_pilot],
        max_hamming=int(config["deduplication"]["simhash_max_hamming"]),
        minimum_jaccard=float(config["deduplication"]["minimum_jaccard"]),
        minimum_length_ratio=float(config["deduplication"]["minimum_length_ratio"]),
    )
    near_ids: defaultdict[str, set[str]] = defaultdict(set)
    pilot_ids = {record["id"] for record in candidates}
    for cluster in clusters:
        members = set(cluster["record_ids"])
        for record_id in members & pilot_ids:
            near_ids[record_id].update(members - {record_id})

    inventory = []
    for record in candidates:
        inventory.append({
            "id": record["id"],
            "source_id": record["source_id"],
            "source_url": record["source_url"],
            "source_sha256": record["provenance"]["raw_sha256"],
            "title": record["title"],
            "creator_catalog_value": record.get("author"),
            "creator_id": f"{record['source_id']}:{record.get('author') or 'unknown'}",
            "collection_id": record["group_id"],
            "genres": broad_genres(record["source_genres"]),
            "text_sha256": record["text_hash"],
            "story_start_line": record["boundary_evidence"]["story_start_line"],
            "story_end_line": record["boundary_evidence"]["story_end_line"],
            "boundary_method": record["boundary_evidence"]["method"],
            "boundary_confidence": record["boundary_evidence"]["confidence"],
            "words": record["words"],
            "tokenizer_v1_tokens": record["tokenizer_v1_tokens"],
            "quality_diagnostic": record["quality_diagnostic"],
            "exact_duplicate_of": record["exact_duplicate_of"],
            "near_duplicate_ids": sorted(near_ids[record["id"]]),
            "rights_review_status": "not_completed",
            "boundary_review_status": "not_completed",
            "disposition": "excluded_fail_closed",
            "first_rejection_reason": "incomplete_rights_evidence",
        })
    inventory.sort(key=lambda row: (selected_ids.index(row["source_id"]), row["story_start_line"], row["id"]))

    per_source = [
        _source_summary(
            source_id,
            [row for row in inventory if row["source_id"] == source_id],
            artifact_rows[source_id],
            catalog[source_id],
        )
        for source_id in selected_ids
    ]
    gross_records = sum(item["gross_extracted"]["records"] for item in per_source)
    gross_tokens = sum(item["gross_extracted"]["tokenizer_v1_tokens"] for item in per_source)
    duplicate_records = sum(item["exact_duplicate_loss"]["records"] for item in per_source)
    duplicate_tokens = sum(item["exact_duplicate_loss"]["tokenizer_v1_tokens"] for item in per_source)
    diagnostic_unique = [row for row in inventory if not row["exact_duplicate_of"]]
    author_concentration = _concentration(diagnostic_unique, "creator_id")
    collection_concentration = _concentration(diagnostic_unique, "collection_id")
    limits = config["concentration_limits"]
    observed = {
        "source_ids": selected_ids,
        "gross_records": gross_records,
        "gross_tokens": gross_tokens,
        "exact_duplicate_records": duplicate_records,
        "exact_duplicate_tokens": duplicate_tokens,
        "final_incremental_unique_records": 0,
        "final_incremental_unique_tokens": 0,
    }
    if observed != config["expected"]:
        raise RuntimeError(f"Source-pilot measurements changed: {observed}")
    report = {
        "report_id": config["report_id"],
        "version": 1,
        "status": "pilot_complete_no_admission",
        "inputs": {
            name: {"path": config["inputs"][name]["path"], "sha256": digest}
            for name, digest in sorted(actual_hashes.items())
        },
        "scope": {
            "source_ids": selected_ids,
            "selection_basis": "Top two medium-confidence catalogs by extracted candidates in issue #20.",
            "feasibility_target_is_not_quota": True,
        },
        "artifact_verification": {
            "passed": True,
            "deterministic_resume": "URL, pinned SHA-256, and cached bytes are verified before reuse.",
        },
        "candidate_inventory": {
            "path": config["candidate_inventory_path"],
            "records": len(inventory),
            "sha256": _candidate_inventory_sha256(inventory),
            "contains_story_text": False,
        },
        "per_source": per_source,
        "total_funnel": {
            "gross_acquired": {"records": gross_records, "tokenizer_v1_tokens": gross_tokens},
            "rejected_by_first_reason": {
                "incomplete_rights_evidence": {
                    "records": gross_records,
                    "tokenizer_v1_tokens": gross_tokens,
                }
            },
            "exact_duplicate_loss": {
                "records": duplicate_records,
                "tokenizer_v1_tokens": duplicate_tokens,
            },
            "duplicate_adjusted_diagnostic": {
                "records": gross_records - duplicate_records,
                "tokenizer_v1_tokens": gross_tokens - duplicate_tokens,
                "qualification": "Not admissible yield; fuzzy matches remain unresolved and evidence gates precede admission.",
            },
            "final_incremental_unique": {"records": 0, "tokenizer_v1_tokens": 0},
        },
        "review_effort": {
            "artifacts_verified": len(selected_ids),
            "candidate_boundaries_generated_for_review": gross_records,
            "attributable_rights_reviews_completed": 0,
            "attributable_boundary_reviews_completed": 0,
            "deterministic_membership_reviews_completed": 0,
        },
        "audit_diagnostics": {
            "rights_evidence": {"passed": False, "unreviewed_records": gross_records},
            "story_boundaries": {"passed": False, "medium_confidence_records": gross_records},
            "creator_evidence": {
                "passed": False,
                "reason": "Catalog creator strings contain editor/reteller roles and were not resolved to reviewed primary authors.",
            },
            "genre_evidence": {
                "passed": False,
                "unclassified_tokens": sum(
                    row["tokenizer_v1_tokens"] for row in inventory if not row["genres"]
                ),
                "gross_tokens": gross_tokens,
            },
            "quality": {
                "passed": False,
                "diagnostically_eligible_records": sum(
                    row["quality_diagnostic"] is None for row in inventory
                ),
                "diagnostically_rejected_records": sum(
                    row["quality_diagnostic"] is not None for row in inventory
                ),
            },
            "exact_duplicates": {"passed": True, "excluded_records": duplicate_records},
            "near_duplicates": {
                "passed": all(not row["near_duplicate_ids"] for row in diagnostic_unique),
                "unresolved_records": sum(bool(row["near_duplicate_ids"]) for row in diagnostic_unique),
            },
            "author_concentration": {
                "passed": author_concentration["largest_token_share"]
                <= float(limits["maximum_author_token_share"]),
                "maximum_token_share": float(limits["maximum_author_token_share"]),
                "metrics": author_concentration,
            },
            "collection_concentration": {
                "passed": collection_concentration["largest_token_share"]
                <= float(limits["maximum_collection_token_share"]),
                "maximum_token_share": float(limits["maximum_collection_token_share"]),
                "metrics": collection_concentration,
            },
            "publication_year_evidence": {
                "passed": True,
                "status": "unknown_not_inferred_from_ebook_issue_date",
            },
        },
        "decision": {
            "action": "stop_selected_sources",
            "reason": "Neither source has attributable edition-specific rights review or high-confidence reviewed story boundaries; pg-28 is dominated by sub-500-word fables and pg-25918 is not target-genre short fiction.",
            "corpus_v3_effect": "No records or tokens admitted; Corpus-v3 remains unsealed and issue #24 remains blocked.",
            "attainable_range_effect": "External-source yield remains unknown beyond this measured zero-admission pilot; Data50M and Data100M remain unsupported.",
            "recommended_next_action": "Stop expansion from these ranked catalogs. Any later recovery should prioritize attributable evidence for the 1,134 existing Gutenberg rows identified by issue #20 rather than weaken gates.",
        },
    }
    return report, inventory


def write_outputs(
    report: dict[str, Any], inventory: list[dict[str, Any]], *, report_path: Path,
    inventory_path: Path,
) -> None:
    write_json_atomic(report_path, report)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = inventory_path.with_suffix(inventory_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in inventory:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(inventory_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/corpus-v3-source-pilot-v1.yaml")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report, inventory = generate_source_pilot_report(args.config)
    write_outputs(
        report,
        inventory,
        report_path=Path(config["output_path"]),
        inventory_path=Path(config["candidate_inventory_path"]),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
