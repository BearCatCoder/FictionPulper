"""Generate the hash-locked Corpus-v3 integrity-blocker disposition."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.audit_corpus_v2 import verify_locked_inputs
from src.corpus_v3 import load_jsonl
from src.train_tokenizer import write_json_atomic


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def generate_blocker_report(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise ValueError("Corpus-v3 blocker report config must declare version: 1")
    actual_hashes = verify_locked_inputs(config["inputs"])
    paths = {name: Path(spec["path"]) for name, spec in config["inputs"].items()}
    shortfall = _json(paths["historical_shortfall_report"])
    stats = _json(paths["corrected_stats"])
    audits = _json(paths["corrected_audits"])
    manifest = _json(paths["corrected_manifest"])
    splits = _json(paths["corrected_splits"])
    attainable = _json(paths["corrected_attainable_stage"])
    rejections = load_jsonl(paths["corrected_rejections"])
    reason_counts = Counter(str(row["reason"]) for row in rejections)

    observed = {
        "candidate_records": stats["candidate_stories"],
        "retained_records": stats["stories"],
        "retained_tokens": stats["exact_actual_token_count"],
        "rejected_records": stats["rejected_stories"],
        "rejection_records_by_first_reason": dict(sorted(reason_counts.items())),
        "failed_required_gates": stats["failed_required_gates"],
    }
    if observed != config["expected_corrected"]:
        raise RuntimeError(f"Corrected blocker results changed: {observed}")
    if len(rejections) != stats["rejected_stories"]:
        raise RuntimeError("Corrected rejection rows do not match stats")
    if not stats["failed_required_gates"] or stats["freeze_ready"]:
        raise RuntimeError("Corrected Corpus-v3 unexpectedly reports freeze readiness")
    if stats["failed_required_gates"] != audits["failed_required_gates"]:
        raise RuntimeError("Corrected stats and audits disagree on failed gates")
    if (
        attainable.get("exact_actual_token_count") != stats["exact_actual_token_count"]
        or attainable.get("failed_required_gates") != stats["failed_required_gates"]
    ):
        raise RuntimeError("Corrected attainable stage and stats disagree")
    if splits.get("audit") != audits["gates"]["split_leakage"]:
        raise RuntimeError("Corrected split and audit leakage results disagree")
    manifest_locks = {
        "config_sha256": "corpus_v3_config",
        "builder_sha256": "corpus_v3_builder",
        "tokenizer_sha256": "tokenizer",
        "review_sha256": "corpus_v3_review",
        "source_artifacts_sha256": "corrected_source_artifacts",
        "corpus_sha256": "corrected_corpus",
        "split_sha256": "corrected_splits",
        "audits_sha256": "corrected_audits",
        "stats_sha256": "corrected_stats",
        "rejections_sha256": "corrected_rejections",
        "acquisition_failures_sha256": "corrected_acquisition_failures",
        "near_duplicate_candidates_sha256": "corrected_near_duplicate_candidates",
        "near_duplicate_clusters_sha256": "corrected_near_duplicate_clusters",
    }
    mismatches = {
        field: {"manifest": manifest.get(field), "actual": actual_hashes[input_name]}
        for field, input_name in manifest_locks.items()
        if manifest.get(field) != actual_hashes[input_name]
    }
    if mismatches:
        raise RuntimeError(f"Corrected manifest hash mismatch: {mismatches}")
    expected_stage_hashes = {
        "attainable.json": actual_hashes["corrected_attainable_stage"],
    }
    if manifest.get("stage_reports_sha256") != expected_stage_hashes:
        raise RuntimeError("Corrected manifest stage report hashes disagree")
    if manifest.get("source_inputs") != [{
        "kind": "base",
        "path": config["inputs"]["corpus_v2"]["path"],
        "sha256": actual_hashes["corpus_v2"],
    }]:
        raise RuntimeError("Corrected manifest source input does not match Corpus-v2")
    if stats["corpus_sha256"] != actual_hashes["corrected_corpus"]:
        raise RuntimeError("Corrected stats and corpus hash disagree")
    if Path(config["seal_path"]).exists():
        raise RuntimeError("Corrected Corpus-v3 unexpectedly has a seal")

    historical_gates = shortfall["current_blockers"]
    historical_funnel = shortfall["funnel"]
    return {
        "report_id": config["report_id"],
        "version": 1,
        "status": "disposition_complete_unsealed",
        "inputs": {
            name: {"path": config["inputs"][name]["path"], "sha256": digest}
            for name, digest in sorted(actual_hashes.items())
        },
        "before_correction": {
            "retained_records": historical_funnel["retained"]["records"],
            "retained_tokens": historical_funnel["retained"]["tokenizer_v1_tokens"],
            "failed_required_gates": historical_gates["failed_required_gates"],
            "unclassified_token_share": historical_gates["unclassified_token_share"],
            "pending_review_records": historical_gates["pending_deterministic_review_records"],
            "qualification": "Historical rights and token-accounting pass claims were invalidated by issue #22 defects.",
        },
        "after_correction": {
            **observed,
            "status": stats["status"],
            "stop_reason": stats["stop_reason"],
            "freeze_ready": False,
            "seal_written": False,
            "gate_summary": audits["gates"],
        },
        "failure_dispositions": {
            "synthetic_rights_check": {
                "classification": "software_defect_corrected",
                "root_cause": "Missing review dates were replaced with inherited_source_assertion.",
                "result": "2521 candidates now fail incomplete_rights_evidence instead of receiving synthetic evidence.",
                "follow_up": "Issue #21 must supply attributable rights evidence tied to immutable source bytes.",
            },
            "legacy_schema_and_provenance": {
                "classification": "data_evidence_limitation",
                "root_cause": "1841 legacy rows lack the complete v3 source, rights, and metadata schema.",
                "result": "Rows remain excluded; no source evidence was invented.",
                "follow_up": "Issue #21 may recover only hash-linked, rights-validated sources; unsupported seed rows remain unavailable.",
            },
            "genre_distribution": {
                "classification": "data_evidence_limitation_and_software_defect_corrected",
                "root_cause": "Historical retained tokens were 57.0408% unclassified, and arbitrary labels were not rejected.",
                "result": "Genres now use the controlled vocabulary; the corrected empty eligible set is not treated as passing.",
                "follow_up": "Issue #21 must provide evidence-backed genre metadata and meet the unchanged 35% cap.",
            },
            "deterministic_review": {
                "classification": "pending_human_evidence_and_software_defect_corrected",
                "root_cause": "The historical 32-record provisional sample had no human decisions; bare ID approvals lacked attribution.",
                "result": "Review now content-hash-locks complete sample records and requires a per-record reviewer, date, decision, and notes.",
                "follow_up": "Issue #21 must freeze final membership before genuine deterministic review; provisional IDs were not auto-approved.",
            },
            "split_and_token_accounting": {
                "classification": "software_defects_corrected",
                "root_cause": "The audit trusted a supplied split result and compared token totals to themselves.",
                "result": "Assignments, collection and fuzzy-cluster leakage, positive counts, split totals, and every tokenizer-v1 document count are independently reconciled.",
                "follow_up": "Re-run on every issue #21 candidate catalog; any mismatch remains a sealing blocker.",
            },
        },
        "safeguards_retained": [
            "immutable source hashes and acquisition failure blocking",
            "allowed rights status plus explicit attributable evidence",
            "high-confidence story boundaries",
            "exact and transitive fuzzy duplicate controls",
            "creator and collection concentration limits",
            "quality and prohibited-text quarantine",
            "controlled genre vocabulary and unchanged unclassified-token cap",
            "collection and duplicate-cluster split isolation",
            "tokenizer-v1 document-token reconciliation",
            "content-hash-locked attributable deterministic review",
        ],
        "seal_conditions": [
            "A nonempty approved candidate corpus passes every required gate.",
            "Every accepted record has immutable source-byte, rights, boundary, creator, genre, and year evidence.",
            "Final deterministic sample records are content-hash-locked and have genuine attributable approvals.",
            "Unclassified tokens are at or below 35% without inferred or fabricated labels.",
            "No exact duplicate or unresolved accepted fuzzy cluster remains.",
            "No collection or duplicate cluster crosses splits and tokenizer-v1 totals reconcile.",
            "Issue #21 reports actual unique-token yield; issue #24 remains blocked until all conditions pass.",
        ],
    }


def write_blocker_report(report: dict[str, Any], output_path: Path) -> None:
    write_json_atomic(output_path, report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/corpus-v3-blockers-v1.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_path = Path(config["output_path"])
    report = generate_blocker_report(args.config)
    write_blocker_report(report, output_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
