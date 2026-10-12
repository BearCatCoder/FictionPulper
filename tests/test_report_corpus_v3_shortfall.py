import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.audit_corpus_v2 import verify_locked_inputs
from src.report_corpus_v3_shortfall import (
    _source_pool_summary,
    generate_shortfall_report,
    summarize_funnel,
    write_shortfall_report,
)
from src.train_tokenizer import sha256_file


class CorpusV3ShortfallReportTests(unittest.TestCase):
    def test_funnel_reconciles_records_and_tokens(self):
        candidates = [
            {"id": "kept", "tokenizer_v1_tokens": 11},
            {"id": "schema", "tokenizer_v1_tokens": 7},
            {"id": "quality", "tokenizer_v1_tokens": 5},
        ]
        result = summarize_funnel(
            candidates,
            [candidates[0]],
            [
                {"id": "schema", "reason": "invalid_schema"},
                {"id": "quality", "reason": "quality_review_required"},
            ],
        )
        self.assertEqual(result["candidate"], {"records": 3, "tokenizer_v1_tokens": 23})
        self.assertEqual(result["retained"], {"records": 1, "tokenizer_v1_tokens": 11})
        self.assertEqual(result["rejected"], {"records": 2, "tokenizer_v1_tokens": 12})
        self.assertTrue(result["categories_are_exclusive"])

    def test_funnel_rejects_incomplete_partition(self):
        candidates = [{"id": "a", "tokenizer_v1_tokens": 3}]
        with self.assertRaisesRegex(ValueError, "do not partition"):
            summarize_funnel(candidates, [], [])

    def test_funnel_rejects_duplicate_output_ids(self):
        candidates = [
            {"id": "a", "tokenizer_v1_tokens": 3},
            {"id": "b", "tokenizer_v1_tokens": 4},
        ]
        with self.assertRaisesRegex(ValueError, "Retained IDs are not unique"):
            summarize_funnel(candidates, [candidates[0], candidates[0]], [{"id": "b", "reason": "x"}])
        with self.assertRaisesRegex(ValueError, "Rejected IDs are not unique"):
            summarize_funnel(
                candidates,
                [candidates[0]],
                [{"id": "b", "reason": "x"}, {"id": "b", "reason": "x"}],
            )

    def test_source_pool_summary_is_deterministic_and_does_not_invent_tokens(self):
        rows = [
            {"source_id": "b", "extraction_confidence": "medium", "extracted_candidates": 2},
            {"source_id": "a", "extraction_confidence": "medium", "extracted_candidates": 2},
            {"source_id": "ignored", "extraction_confidence": "low", "extracted_candidates": 9},
        ]
        first = _source_pool_summary(rows, "medium")
        second = _source_pool_summary(list(reversed(rows)), "medium")
        self.assertEqual(first, second)
        self.assertEqual(first["gross_extracted_candidate_records"], 4)
        self.assertIsNone(first["token_yield_range"])
        self.assertIsNone(first["duplicate_adjusted_candidate_record_range"])
        self.assertEqual(first["top_catalogs_by_extracted_candidates"][0]["source_id"], "a")

    def test_report_generation_is_deterministic(self):
        candidates = [
            {"id": "kept", "source": "fixture", "source_id": "kept", "tokenizer_v1_tokens": 11},
            {"id": "legacy", "source": "project_gutenberg", "source_id": "pg-1", "tokenizer_v1_tokens": 7},
        ]
        retained = [candidates[0]]
        rejections = [{"id": "legacy", "reason": "invalid_schema"}]
        config = {
            "version": 1,
            "report_id": "fixture",
            "inputs": {name: {"path": f"{name}.json", "sha256": name} for name in (
                "corpus_v3_config", "tokenizer", "corpus_v2", "corpus_v2_stats",
                "corpus_v2_source_manifest", "corpus_v2_source_summaries", "corpus_v3",
                "corpus_v3_rejections", "corpus_v3_acquisition_failures",
                "corpus_v3_source_artifacts", "corpus_v3_near_duplicate_candidates",
                "corpus_v3_near_duplicate_clusters", "corpus_v3_stats", "corpus_v3_audits",
                "corpus_v3_manifest", "corpus_v3_attainable_stage",
            )},
            "expected": {
                "candidate_records": 2, "candidate_tokens": 18, "retained_records": 1,
                "retained_tokens": 11, "rejected_records": 1,
            },
        }

        def jsonl(path):
            return {
                "corpus_v2.json": candidates,
                "corpus_v3.json": retained,
                "corpus_v3_rejections.json": rejections,
                "corpus_v2_source_manifest.json": [{"source_id": "pg-1", "sha256": "a" * 64, "url": "https://example.test/1"}],
            }.get(path.name, [])

        def json_object(path):
            return {
                "corpus_v2_stats.json": {"exact_document_tokens": 18, "rejection_counts": {}},
                "corpus_v3_stats.json": {"exact_actual_token_count": 11, "eligible_stories": 1, "stories": 1, "failed_required_gates": []},
                "corpus_v3_audits.json": {"gates": {"token_accounting": {"exact_document_tokens": 11}, "near_duplicates": {"unresolved_clusters": 0}, "genre_distribution": {"unclassified_token_share": 0.0}, "deterministic_review": {"missing_ids": []}}},
                "corpus_v3_manifest.json": {"corpus_sha256": "corpus_v3"},
                "corpus_v3_attainable_stage.json": {"exact_actual_token_count": 11},
            }[path.name]

        def reject(record, _quality):
            return None if record.get("source_sha256") else (
                "invalid_schema" if record["id"] == "legacy" else None
            )

        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.yaml"
            corpus_config_path = Path(temporary) / "corpus-v3.yaml"
            corpus_config_path.write_text("source_inventories: []\n", encoding="utf-8")
            config["inputs"]["corpus_v3_config"]["path"] = str(corpus_config_path)
            config_path.write_text(__import__("yaml").safe_dump(config), encoding="utf-8")
            with (
                patch("src.report_corpus_v3_shortfall.verify_locked_inputs", return_value={name: name for name in config["inputs"]}),
                patch("src.report_corpus_v3_shortfall.Tokenizer.from_file", return_value=object()),
                patch("src.report_corpus_v3_shortfall.load_jsonl", side_effect=jsonl),
                patch("src.report_corpus_v3_shortfall._json", side_effect=json_object),
                patch("src.report_corpus_v3_shortfall.canonicalize_record", side_effect=lambda row, _tokenizer: row),
                patch("src.report_corpus_v3_shortfall.record_rejection", side_effect=reject),
                patch("src.report_corpus_v3_shortfall.yaml.safe_load", side_effect=[config, {"source_inventories": [], "quality": {"quarantine_signals": []}, "audit": {"maximum_unclassified_token_share": 0.35}}, config, {"source_inventories": [], "quality": {"quarantine_signals": []}, "audit": {"maximum_unclassified_token_share": 0.35}}]),
            ):
                first = generate_shortfall_report(config_path)
                second = generate_shortfall_report(config_path)
        self.assertEqual(first, second)
        self.assertEqual(first["funnel"]["candidate"]["tokenizer_v1_tokens"], 18)

    def test_report_serialization_is_deterministic(self):
        report = {"status": "complete", "accounting": {"records": 3, "tokens": 23}}
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.json"
            second = Path(temporary) / "second.json"
            write_shortfall_report(report, first)
            write_shortfall_report(report, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(json.loads(first.read_text(encoding="utf-8")), report)

    def test_locked_input_tampering_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            path.write_text("{}\n", encoding="utf-8")
            spec = {"fixture": {"path": os.path.relpath(path), "sha256": sha256_file(path)}}
            path.write_text('{"changed": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "input changed"):
                verify_locked_inputs(spec)


if __name__ == "__main__":
    unittest.main()
