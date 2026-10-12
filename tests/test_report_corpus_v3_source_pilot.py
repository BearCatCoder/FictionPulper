import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.report_corpus_v3_source_pilot import (
    _candidate_inventory_sha256,
    _concentration,
    _source_summary,
    write_outputs,
)


class CorpusV3SourcePilotReportTests(unittest.TestCase):
    def test_concentration_uses_token_and_record_shares(self):
        result = _concentration([
            {"creator_id": "a", "tokenizer_v1_tokens": 9},
            {"creator_id": "b", "tokenizer_v1_tokens": 1},
        ], "creator_id")
        self.assertEqual(result["unique_count"], 2)
        self.assertEqual(result["largest_token_share"], 0.9)
        self.assertEqual(result["largest_record_share"], 0.5)

    def test_source_summary_reconciles_gross_duplicates_quality_and_genres(self):
        rows = [
            {
                "tokenizer_v1_tokens": 11,
                "quality_diagnostic": None,
                "exact_duplicate_of": None,
                "near_duplicate_ids": [],
                "genres": ["mystery"],
            },
            {
                "tokenizer_v1_tokens": 7,
                "quality_diagnostic": "too_short_or_front_matter",
                "exact_duplicate_of": "base-a",
                "near_duplicate_ids": ["base-b"],
                "genres": [],
            },
        ]
        result = _source_summary(
            "pg-1",
            rows,
            {"url": "https://example.test/1", "source_sha256": "a" * 64, "byte_count": 9},
            {"Title": " Example  Collection ", "Authors": "Writer, Example"},
        )
        self.assertEqual(result["gross_extracted"], {"records": 2, "tokenizer_v1_tokens": 18})
        self.assertEqual(result["exact_duplicate_loss"], {"records": 1, "tokenizer_v1_tokens": 7})
        self.assertEqual(result["duplicate_adjusted"]["tokenizer_v1_tokens"], 11)
        self.assertEqual(result["accepted_incremental_unique"]["tokenizer_v1_tokens"], 0)

    def test_inventory_hash_and_serialization_are_deterministic(self):
        rows = [{"id": "a", "text_sha256": hashlib.sha256(b"story").hexdigest()}]
        self.assertEqual(_candidate_inventory_sha256(rows), _candidate_inventory_sha256(list(rows)))
        report = {"status": "pilot_complete_no_admission"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_report = root / "first.json"
            first_inventory = root / "first.jsonl"
            second_report = root / "second.json"
            second_inventory = root / "second.jsonl"
            write_outputs(report, rows, report_path=first_report, inventory_path=first_inventory)
            write_outputs(report, rows, report_path=second_report, inventory_path=second_inventory)
            self.assertEqual(first_report.read_bytes(), second_report.read_bytes())
            self.assertEqual(first_inventory.read_bytes(), second_inventory.read_bytes())
            self.assertNotIn("story", first_inventory.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(first_report.read_text(encoding="utf-8")), report)


if __name__ == "__main__":
    unittest.main()
