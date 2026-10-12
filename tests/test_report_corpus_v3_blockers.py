import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.report_corpus_v3_blockers import generate_blocker_report, write_blocker_report


class CorpusV3BlockerReportTests(unittest.TestCase):
    def test_generation_is_deterministic_and_fail_closed(self):
        config = {
            "version": 1,
            "report_id": "fixture",
            "seal_path": "seal.json",
            "inputs": {name: {"path": f"{name}.json", "sha256": name} for name in (
                "historical_shortfall_report", "corpus_v3_config", "corpus_v3_review",
                "corpus_v3_builder", "corpus_v2", "tokenizer", "corrected_stats",
                "corrected_audits", "corrected_rejections", "corrected_manifest",
                "corrected_corpus", "corrected_splits", "corrected_source_artifacts",
                "corrected_acquisition_failures", "corrected_near_duplicate_candidates",
                "corrected_near_duplicate_clusters", "corrected_attainable_stage",
            )},
            "expected_corrected": {
                "candidate_records": 2,
                "retained_records": 0,
                "retained_tokens": 0,
                "rejected_records": 2,
                "rejection_records_by_first_reason": {"incomplete_rights_evidence": 2},
                "failed_required_gates": ["rights_evidence"],
            },
        }
        objects = {
            "historical_shortfall_report.json": {
                "funnel": {"retained": {"records": 1, "tokenizer_v1_tokens": 10}},
                "current_blockers": {
                    "failed_required_gates": ["deterministic_review"],
                    "unclassified_token_share": 0.5,
                    "pending_deterministic_review_records": 1,
                },
            },
            "corrected_stats.json": {
                "candidate_stories": 2,
                "stories": 0,
                "exact_actual_token_count": 0,
                "rejected_stories": 2,
                "failed_required_gates": ["rights_evidence"],
                "freeze_ready": False,
                "status": "stopped_audit_failures",
                "stop_reason": "candidate_supply_exhausted_with_audit_failures",
                "corpus_sha256": "corrected_corpus",
            },
            "corrected_audits.json": {"gates": {"rights_evidence": {"passed": False}}},
            "corrected_splits.json": {"audit": {"passed": False}},
            "corrected_attainable_stage.json": {
                "exact_actual_token_count": 0,
                "failed_required_gates": ["rights_evidence"],
            },
            "corrected_manifest.json": {
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
                "stage_reports_sha256": {
                    "attainable.json": "corrected_attainable_stage",
                },
                "source_inputs": [{
                    "kind": "base", "path": "corpus_v2.json", "sha256": "corpus_v2",
                }],
            },
        }
        objects["corrected_audits.json"]["failed_required_gates"] = ["rights_evidence"]
        objects["corrected_audits.json"]["gates"]["split_leakage"] = {"passed": False}
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with (
                patch(
                    "src.report_corpus_v3_blockers.verify_locked_inputs",
                    return_value={name: name for name in config["inputs"]},
                ),
                patch(
                    "src.report_corpus_v3_blockers._json",
                    side_effect=lambda path: objects[path.name],
                ),
                patch(
                    "src.report_corpus_v3_blockers.load_jsonl",
                    return_value=[
                        {"id": "a", "reason": "incomplete_rights_evidence"},
                        {"id": "b", "reason": "incomplete_rights_evidence"},
                    ],
                ),
                patch("src.report_corpus_v3_blockers.Path.exists", return_value=False),
            ):
                first = generate_blocker_report(config_path)
                second = generate_blocker_report(config_path)
            self.assertEqual(first, second)
            self.assertFalse(first["after_correction"]["freeze_ready"])

    def test_serialization_is_deterministic(self):
        report = {"status": "disposition_complete_unsealed", "failed": ["rights"]}
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.json"
            second = Path(temporary) / "second.json"
            write_blocker_report(report, first)
            write_blocker_report(report, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(json.loads(first.read_text(encoding="utf-8")), report)


if __name__ == "__main__":
    unittest.main()
