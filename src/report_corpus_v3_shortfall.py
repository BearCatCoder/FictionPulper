"""Generate the hash-grounded Corpus-v3 shortfall diagnosis."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from tokenizers import Tokenizer

from src.audit_corpus_v2 import ocr_signals, verify_locked_inputs
from src.corpus_v3 import canonicalize_record, load_jsonl, record_rejection
from src.train_tokenizer import write_json_atomic


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _count(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "records": len(records),
        "tokenizer_v1_tokens": sum(int(record["tokenizer_v1_tokens"]) for record in records),
    }


def summarize_funnel(
    candidates: list[dict[str, Any]],
    retained: list[dict[str, Any]],
    rejections: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reconcile exclusive first-reason rejection accounting."""
    by_id = {record["id"]: record for record in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("Candidate IDs are not unique")
    retained_id_list = [record["id"] for record in retained]
    rejected_id_list = [row["id"] for row in rejections]
    retained_ids = set(retained_id_list)
    rejected_ids = set(rejected_id_list)
    if len(retained_ids) != len(retained_id_list):
        raise ValueError("Retained IDs are not unique")
    if len(rejected_ids) != len(rejected_id_list):
        raise ValueError("Rejected IDs are not unique")
    if retained_ids & rejected_ids or retained_ids | rejected_ids != set(by_id):
        raise ValueError("Retained and rejected IDs do not partition candidates")
    reasons: dict[str, dict[str, int]] = {}
    for reason in sorted({str(row["reason"]) for row in rejections}):
        records = [by_id[row["id"]] for row in rejections if row["reason"] == reason]
        reasons[reason] = _count(records)
    candidate = _count(candidates)
    retained_count = _count([by_id[record_id] for record_id in retained_ids])
    rejected_count = _count([by_id[record_id] for record_id in rejected_ids])
    if candidate["records"] != retained_count["records"] + rejected_count["records"]:
        raise ValueError("Record accounting does not reconcile")
    if candidate["tokenizer_v1_tokens"] != (
        retained_count["tokenizer_v1_tokens"] + rejected_count["tokenizer_v1_tokens"]
    ):
        raise ValueError("Token accounting does not reconcile")
    if sum(value["records"] for value in reasons.values()) != rejected_count["records"]:
        raise ValueError("Exclusive rejection record categories do not reconcile")
    if sum(value["tokenizer_v1_tokens"] for value in reasons.values()) != rejected_count["tokenizer_v1_tokens"]:
        raise ValueError("Exclusive rejection token categories do not reconcile")
    return {
        "candidate": candidate,
        "retained": retained_count,
        "rejected": rejected_count,
        "exclusive_first_rejection_reason": reasons,
        "categories_are_exclusive": True,
    }


def _source_pool_summary(source_summaries: list[dict[str, Any]], confidence: str) -> dict[str, Any]:
    rows = [row for row in source_summaries if row.get("extraction_confidence") == confidence]
    extracted = sum(int(row.get("extracted_candidates", 0)) for row in rows)
    ranked = sorted(
        rows,
        key=lambda row: (-int(row.get("extracted_candidates", 0)), str(row.get("source_id"))),
    )
    return {
        "collection_records": len(rows),
        "gross_extracted_candidate_records": extracted,
        "duplicate_adjusted_candidate_record_range": None,
        "quality_adjusted_candidate_record_range": None,
        "token_yield_range": None,
        "token_yield_status": "unknown_unmeasured",
        "top_catalogs_by_extracted_candidates": [
            {
                "source_id": row["source_id"],
                "extracted_candidates": int(row.get("extracted_candidates", 0)),
            }
            for row in ranked[:10]
        ],
    }


def generate_shortfall_report(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise ValueError("Shortfall config must declare version: 1")
    inputs = config["inputs"]
    actual_hashes = verify_locked_inputs(inputs)
    paths = {name: Path(spec["path"]) for name, spec in inputs.items()}
    corpus_config = yaml.safe_load(paths["corpus_v3_config"].read_text(encoding="utf-8"))
    if corpus_config.get("source_inventories") != []:
        raise RuntimeError("This diagnosis is locked to the base-only Corpus-v3 feasibility build")

    tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
    candidates = [canonicalize_record(row, tokenizer) for row in load_jsonl(paths["corpus_v2"])]
    retained = load_jsonl(paths["corpus_v3"])
    rejections = load_jsonl(paths["corpus_v3_rejections"])
    quality = corpus_config["quality"]
    reported_reason = {row["id"]: row["reason"] for row in rejections}
    recomputed_reason = {}
    for record in candidates:
        reason = record_rejection(record, quality)
        if reason is not None:
            recomputed_reason[record["id"]] = reason
    if reported_reason != recomputed_reason:
        raise RuntimeError("Committed rejection rows do not match recomputed first reasons")
    funnel = summarize_funnel(candidates, retained, rejections)

    expected = config["expected"]
    observed = {
        "candidate_records": funnel["candidate"]["records"],
        "candidate_tokens": funnel["candidate"]["tokenizer_v1_tokens"],
        "retained_records": funnel["retained"]["records"],
        "retained_tokens": funnel["retained"]["tokenizer_v1_tokens"],
        "rejected_records": funnel["rejected"]["records"],
    }
    if observed != expected:
        raise RuntimeError(f"Shortfall totals changed: {observed}, expected {expected}")

    by_id = {record["id"]: record for record in candidates}
    quality_rows = [by_id[row["id"]] for row in rejections if row["reason"] == "quality_review_required"]
    quality_signals: dict[str, dict[str, int]] = {}
    for signal in quality["quarantine_signals"]:
        matching = [record for record in quality_rows if ocr_signals(record["text"])[signal]]
        quality_signals[signal] = _count(matching)

    source_manifest = {
        row["source_id"]: row
        for row in load_jsonl(paths["corpus_v2_source_manifest"])
        if row["source_id"] != "pg_catalog"
    }
    repair_candidates = []
    repair_outcomes: Counter[str] = Counter()
    repair_tokens: Counter[str] = Counter()
    for record in candidates:
        if reported_reason.get(record["id"]) != "invalid_schema" or record["source"] != "project_gutenberg":
            continue
        evidence = source_manifest.get(record["source_id"])
        if evidence is None:
            continue
        repaired = {
            **record,
            "source_sha256": evidence["sha256"],
            "source_url": evidence["url"],
            "rights_evidence": {
                "basis": "Project Gutenberg public-domain-in-the-United-States distribution",
                "evidence_url": evidence["url"],
                "checked_at": "conditional_provenance_migration_requires_review",
            },
        }
        outcome = record_rejection(repaired, quality) or "otherwise_eligible"
        repair_candidates.append(repaired)
        repair_outcomes[outcome] += 1
        repair_tokens[outcome] += repaired["tokenizer_v1_tokens"]
    repair_breakdown = {
        outcome: {"records": repair_outcomes[outcome], "tokenizer_v1_tokens": repair_tokens[outcome]}
        for outcome in sorted(repair_outcomes)
    }

    source_summaries = load_jsonl(paths["corpus_v2_source_summaries"])
    v2_stats = _json(paths["corpus_v2_stats"])
    stats = _json(paths["corpus_v3_stats"])
    audits = _json(paths["corpus_v3_audits"])
    manifest = _json(paths["corpus_v3_manifest"])
    attainable_stage = _json(paths["corpus_v3_attainable_stage"])
    acquisition_failures = load_jsonl(paths["corpus_v3_acquisition_failures"])
    source_artifacts = load_jsonl(paths["corpus_v3_source_artifacts"])
    near_edges = load_jsonl(paths["corpus_v3_near_duplicate_candidates"])
    near_clusters = load_jsonl(paths["corpus_v3_near_duplicate_clusters"])
    eligible_repair = repair_breakdown.get(
        "otherwise_eligible", {"records": 0, "tokenizer_v1_tokens": 0}
    )
    provenance_scenario = observed["retained_tokens"] + eligible_repair["tokenizer_v1_tokens"]
    upstream_rejections = {
        reason: {
            "records": count,
            "tokenizer_v1_tokens": None,
            "scope": "historical_corpus_v2_acquisition",
        }
        for reason, count in sorted(v2_stats["rejection_counts"].items())
    }
    schema_by_source = {
        source: _count([
            by_id[row["id"]] for row in rejections
            if row["reason"] == "invalid_schema" and by_id[row["id"]]["source"] == source
        ])
        for source in sorted({
            by_id[row["id"]]["source"] for row in rejections if row["reason"] == "invalid_schema"
        })
    }
    consistency_values = {
        "corpus_v2_stats_tokens": v2_stats["exact_document_tokens"],
        "corpus_v3_stats_tokens": stats["exact_actual_token_count"],
        "corpus_v3_audit_tokens": audits["gates"]["token_accounting"]["exact_document_tokens"],
        "attainable_stage_tokens": attainable_stage["exact_actual_token_count"],
    }
    if set(consistency_values.values()) != {observed["candidate_tokens"], observed["retained_tokens"]}:
        raise RuntimeError(f"Independent token reports do not reconcile: {consistency_values}")
    if v2_stats["exact_document_tokens"] != observed["candidate_tokens"]:
        raise RuntimeError("Corpus-v2 stats do not match recomputed candidate tokens")
    if any(
        value != observed["retained_tokens"]
        for name, value in consistency_values.items()
        if name != "corpus_v2_stats_tokens"
    ):
        raise RuntimeError("Corpus-v3 reports do not match recomputed retained tokens")
    if manifest["corpus_sha256"] != actual_hashes["corpus_v3"]:
        raise RuntimeError("Corpus-v3 manifest does not identify the retained corpus input")

    return {
        "report_id": config["report_id"],
        "version": 1,
        "status": "diagnosis_complete_not_training_ready",
        "inputs": {
            name: {"path": inputs[name]["path"], "sha256": actual_hashes[name]}
            for name in sorted(inputs)
        },
        "scope": {
            "source_inventories_configured": 0,
            "external_acquisition_attempts": len(source_artifacts) + len(acquisition_failures),
            "external_acquisition_failures": len(acquisition_failures),
            "sealed_artifacts_modified": False,
        },
        "funnel": funnel,
        "independent_token_accounting": consistency_values,
        "overlapping_quality_diagnostics": {
            "population": "exclusive quality_review_required rows",
            "categories_are_exclusive": False,
            "signals": quality_signals,
        },
        "major_shortfall_categories": {
            "external_candidate_availability": {"records": None, "tokenizer_v1_tokens": None, "status": "unknown_no_external_inventory"},
            "acquisition_or_resumability_failures": {"records": len(acquisition_failures), "tokenizer_v1_tokens": None},
            "exact_duplicate_losses": funnel["exclusive_first_rejection_reason"].get("exact_duplicate", {"records": 0, "tokenizer_v1_tokens": 0}),
            "near_duplicate_candidates": {"edges": len(near_edges), "clusters": len(near_clusters), "accepted_unresolved_clusters": audits["gates"]["near_duplicates"]["unresolved_clusters"]},
            "genre_admission_filter_losses": {"records": 0, "tokenizer_v1_tokens": 0, "note": "Genre is a corpus-level gate, not a current row rejection."},
            "length_filter_losses": funnel["exclusive_first_rejection_reason"].get("story_length_out_of_range", {"records": 0, "tokenizer_v1_tokens": 0}),
            "boundary_filter_losses": funnel["exclusive_first_rejection_reason"].get("unverified_story_boundary", {"records": 0, "tokenizer_v1_tokens": 0}),
            "creator_evidence_losses": funnel["exclusive_first_rejection_reason"].get("incomplete_creator_evidence", {"records": 0, "tokenizer_v1_tokens": 0}),
            "quality_review_losses": funnel["exclusive_first_rejection_reason"].get("quality_review_required", {"records": 0, "tokenizer_v1_tokens": 0}),
            "prohibited_text_losses": funnel["exclusive_first_rejection_reason"].get("prohibited_text_signal", {"records": 0, "tokenizer_v1_tokens": 0}),
            "schema_or_provenance_losses": funnel["exclusive_first_rejection_reason"].get("invalid_schema", {"records": 0, "tokenizer_v1_tokens": 0}),
            "selection_or_concentration_losses": {"records": stats["eligible_stories"] - stats["stories"], "tokenizer_v1_tokens": 0},
        },
        "historical_upstream_extraction_losses": {
            "categories": upstream_rejections,
            "record_unit_note": "Most rows are extracted story candidates; low- and medium-confidence reasons count whole collection records.",
            "token_accounting": "unknown because rejected candidate document-token counts were not retained",
        },
        "schema_loss_by_source": schema_by_source,
        "current_rejection_recovery_disposition": {
            "invalid_schema_project_gutenberg": {
                "status": "conditional_provenance_recovery",
                **schema_by_source.get("project_gutenberg", {"records": 0, "tokenizer_v1_tokens": 0}),
            },
            "invalid_schema_seed_corpus": {
                "status": "unavailable_without_original_source_bytes_rights_and_hashes",
                **schema_by_source.get("Travis-ML/ShortStory-SFT-jsonl", {"records": 0, "tokenizer_v1_tokens": 0}),
            },
            "incomplete_creator_evidence": {
                "status": "recoverability_unknown_pending_evidence",
                **funnel["exclusive_first_rejection_reason"].get("incomplete_creator_evidence", {"records": 0, "tokenizer_v1_tokens": 0}),
            },
            "quality_review_required": {
                "status": "recoverability_unknown_pending_human_review",
                **funnel["exclusive_first_rejection_reason"].get("quality_review_required", {"records": 0, "tokenizer_v1_tokens": 0}),
            },
            "prohibited_text_signal": {
                "status": "intentional_safeguard_exclusion",
                **funnel["exclusive_first_rejection_reason"].get("prohibited_text_signal", {"records": 0, "tokenizer_v1_tokens": 0}),
            },
        },
        "conditional_gutenberg_provenance_recovery": {
            "status": "measured_conditional_not_admitted",
            "source_records_with_hash_pinned_manifest_join": len(repair_candidates),
            "source_ids_with_hash_pinned_manifest_join": len({row["source_id"] for row in repair_candidates}),
            "assumption": "Join legacy Gutenberg rows to exact source bytes, verify deterministic text derivation, revalidate story boundaries, and independently validate rights evidence; no placeholder evidence is admissible.",
            "post_repair_first_reason": repair_breakdown,
        },
        "source_yield_hypotheses": {
            "existing_medium_confidence_gutenberg_collections": _source_pool_summary(source_summaries, "medium"),
            "existing_low_confidence_gutenberg_collections": _source_pool_summary(source_summaries, "low"),
            "new_external_catalogs": {
                "gross_token_yield_range": None,
                "duplicate_adjusted_token_yield_range": None,
                "quality_adjusted_token_yield_range": None,
                "status": "unknown_until_issue_21_bounded_source_pilot",
            },
        },
        "attainable_size": {
            "measured_individually_eligible_tokens": observed["retained_tokens"],
            "measured_provenance_only_scenario_tokens": provenance_scenario,
            "bounded_evidence_scenario_range_tokens": [observed["retained_tokens"], provenance_scenario],
            "conditional_additional_tokens": eligible_repair["tokenizer_v1_tokens"],
            "full_current_pool_upper_bound": "unknown_pending_creator_and_quality_review",
            "gross_current_candidate_ceiling_tokens": observed["candidate_tokens"],
            "data50m_current_pool": "unsupported",
            "data100m_current_pool": "unsupported",
            "external_expansion": "unknown",
            "gap_from_provenance_scenario_to_data50m": 50_000_000 - provenance_scenario,
            "uncertainty": "The scenario endpoint is not a full current-pool upper bound. It includes only the measured provenance-repair opportunity; creator-evidence and quality-review recovery remain unknown. Every recovered row must still pass duplicate, genre, concentration, quality, split-leakage, and human-review gates.",
        },
        "current_blockers": {
            "failed_required_gates": stats["failed_required_gates"],
            "unclassified_token_share": audits["gates"]["genre_distribution"]["unclassified_token_share"],
            "maximum_unclassified_token_share": corpus_config["audit"]["maximum_unclassified_token_share"],
            "pending_deterministic_review_records": len(audits["gates"]["deterministic_review"]["missing_ids"]),
        },
        "decision": {
            "recommended_next_action": "Run a bounded provenance-recovery pass for the 1,134 legacy Gutenberg rows, verifying source-byte/text linkage, boundaries, and rights before external expansion.",
            "safeguards_retained": [
                "rights evidence", "source hashing", "exact and near-duplicate review",
                "split leakage", "creator and collection concentration", "quality and OCR review",
                "high-confidence story boundaries", "genre coverage", "deterministic human review",
            ],
            "training_authorized": False,
            "corpus_v3_training_ready": False,
        },
    }


def write_shortfall_report(report: dict[str, Any], output_path: Path) -> None:
    write_json_atomic(output_path, report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/corpus-v3-shortfall-v1.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = generate_shortfall_report(args.config)
    output = Path(config["output_path"])
    write_shortfall_report(report, output)
    print(json.dumps({"output_path": str(output), "status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
