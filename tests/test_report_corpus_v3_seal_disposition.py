import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.report_corpus_v3_seal_disposition import (
    generate_seal_disposition,
    write_seal_disposition,
)


class CorpusV3SealDispositionTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        blocker = {
            "after_correction": {
                "candidate_records": 2,
                "retained_records": 0,
                "retained_tokens": 0,
                "rejection_records_by_first_reason": {"incomplete_rights_evidence": 2},
                "failed_required_gates": ["rights_evidence"],
                "freeze_ready": False,
                "seal_written": False,
            }
        }
        source = {
            "total_funnel": {
                "gross_acquired": {"records": 1, "tokenizer_v1_tokens": 10},
                "final_incremental_unique": {"records": 0, "tokenizer_v1_tokens": 0},
            },
            "decision": {"action": "stop_selected_sources"},
        }
        data100m = {
            "experiment": {"status": "prepared_blocked_on_corpus_v3"},
            "data": {
                "expected_corpus_sha256": None,
                "expected_split_sha256": None,
                "expected_packed_metadata_sha256": None,
            },
        }
        for name, value in (("blocker.json", blocker), ("source.json", source)):
            (root / name).write_text(json.dumps(value), encoding="utf-8")
        (root / "data100m.yaml").write_text(yaml.safe_dump(data100m), encoding="utf-8")
        config = {
            "version": 1,
            "report_id": "fixture",
            "inputs": {
                "blocker_report": {"path": str(root / "blocker.json"), "sha256": "a"},
                "source_pilot_report": {"path": str(root / "source.json"), "sha256": "b"},
                "data100m_config": {"path": str(root / "data100m.yaml"), "sha256": "c"},
            },
            "artifacts_that_must_be_absent": [str(root / "seal.json")],
            "expected": {
                "base_candidate_records": 2,
                "base_approved_records": 0,
                "base_approved_tokens": 0,
                "source_pilot_gross_records": 1,
                "source_pilot_gross_tokens": 10,
                "source_pilot_approved_records": 0,
                "source_pilot_approved_tokens": 0,
            },
        }
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def test_generation_records_no_go_without_creating_training_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = self._fixture(Path(temporary))
            with patch(
                "src.report_corpus_v3_seal_disposition.verify_locked_inputs",
                return_value={
                    "blocker_report": "a",
                    "source_pilot_report": "b",
                    "data100m_config": "c",
                },
            ):
                report = generate_seal_disposition(config_path)
            self.assertEqual(report["approved_supply"]["tokenizer_v1_document_tokens"], 0)
            self.assertEqual(report["experiment_decision"]["action"], "no_go_data_scaling")
            self.assertFalse(report["artifact_disposition"]["corpus_v3_seal_written"])

    def test_generation_fails_if_approved_supply_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._fixture(root)
            source_path = root / "source.json"
            source = json.loads(source_path.read_text(encoding="utf-8"))
            source["total_funnel"]["final_incremental_unique"]["records"] = 1
            source_path.write_text(json.dumps(source), encoding="utf-8")
            with patch(
                "src.report_corpus_v3_seal_disposition.verify_locked_inputs",
                return_value={
                    "blocker_report": "a",
                    "source_pilot_report": "b",
                    "data100m_config": "c",
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "approved supply changed"):
                    generate_seal_disposition(config_path)

    def test_generation_fails_if_data100m_is_unblocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._fixture(root)
            data100m_path = root / "data100m.yaml"
            data100m = yaml.safe_load(data100m_path.read_text(encoding="utf-8"))
            data100m["experiment"]["status"] = "ready"
            data100m_path.write_text(yaml.safe_dump(data100m), encoding="utf-8")
            with patch(
                "src.report_corpus_v3_seal_disposition.verify_locked_inputs",
                return_value={
                    "blocker_report": "a",
                    "source_pilot_report": "b",
                    "data100m_config": "c",
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "Data100M is no longer"):
                    generate_seal_disposition(config_path)

    def test_generation_fails_if_base_becomes_freeze_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._fixture(root)
            blocker_path = root / "blocker.json"
            blocker = json.loads(blocker_path.read_text(encoding="utf-8"))
            blocker["after_correction"]["freeze_ready"] = True
            blocker_path.write_text(json.dumps(blocker), encoding="utf-8")
            with patch(
                "src.report_corpus_v3_seal_disposition.verify_locked_inputs",
                return_value={
                    "blocker_report": "a",
                    "source_pilot_report": "b",
                    "data100m_config": "c",
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "no-seal disposition"):
                    generate_seal_disposition(config_path)

    def test_generation_fails_if_source_pilot_action_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._fixture(root)
            source_path = root / "source.json"
            source = json.loads(source_path.read_text(encoding="utf-8"))
            source["decision"]["action"] = "continue"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            with patch(
                "src.report_corpus_v3_seal_disposition.verify_locked_inputs",
                return_value={
                    "blocker_report": "a",
                    "source_pilot_report": "b",
                    "data100m_config": "c",
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "stop disposition"):
                    generate_seal_disposition(config_path)

    def test_generation_fails_if_a_forbidden_artifact_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._fixture(root)
            (root / "seal.json").write_text("{}", encoding="utf-8")
            with patch(
                "src.report_corpus_v3_seal_disposition.verify_locked_inputs",
                return_value={
                    "blocker_report": "a",
                    "source_pilot_report": "b",
                    "data100m_config": "c",
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "Unauthorized Corpus-v3"):
                    generate_seal_disposition(config_path)

    def test_serialization_is_deterministic(self):
        report = {"status": "stop_complete_no_approved_corpus"}
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.json"
            second = Path(temporary) / "second.json"
            write_seal_disposition(report, first)
            write_seal_disposition(report, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
