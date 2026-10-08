import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.narrative_post_selection import (
    PROMPTS,
    aggregate_recall,
    run_after_gate,
    verify_protocol,
    verify_selection_gate,
)
from src.train import GENERATION_SETTINGS


class PostSelectionGateTests(unittest.TestCase):
    def test_failed_gate_never_invokes_held_out_evaluator(self):
        opened = []

        def failed_gate():
            raise RuntimeError("Training is not complete")

        def evaluator(*_):
            opened.append("test")
            return {}

        with self.assertRaisesRegex(RuntimeError, "not complete"):
            run_after_gate({}, failed_gate, evaluator)
        self.assertEqual(opened, [])

    def test_gate_verifies_summary_before_loading_checkpoint(self):
        protocol = {
            "required_training_status": "training_and_selection_complete",
            "selection_metric": "Data30M validation loss only",
            "candidate": {
                "summary_path": "summary.json", "checkpoint_path": "best.pt",
                "expected_steps": 2, "tokenizer_sha256": "tokenizer",
            },
        }
        summary = {"status": "running"}
        with patch("src.narrative_post_selection.load_json", return_value=summary):
            with self.assertRaisesRegex(RuntimeError, "not complete"):
                verify_selection_gate(protocol, load_checkpoint=lambda _: self.fail("checkpoint opened"))


class ProtocolInvariantTests(unittest.TestCase):
    def setUp(self):
        self.protocol_path = Path("experiments/fictionpulper-15m-data30m-narrative-v1/post-selection-protocol.json")
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))

    def test_historical_prompts_settings_and_hashes_are_locked(self):
        self.assertEqual(len(PROMPTS), 10)
        self.assertEqual(self.protocol["generation"]["settings"], GENERATION_SETTINGS)
        self.assertEqual(self.protocol["generation"]["sample_seed"], 11337)
        verify_protocol(self.protocol)

    def test_protocol_hash_mutation_is_rejected_without_real_test_access(self):
        changed = copy.deepcopy(self.protocol)
        changed["narrative_state_protocol"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "hash changed"):
            verify_protocol(changed)

    def test_curriculum_hash_mutation_is_rejected_from_manifest_only(self):
        changed = copy.deepcopy(self.protocol)
        changed["held_out"]["curriculum_test"]["jsonl_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "manifest hash differs"):
            verify_protocol(changed)

    def test_required_holdout_order_and_compute5_checkpoint_are_explicit(self):
        self.assertEqual(
            self.protocol["held_out"]["historical_benchmarks_in_order"],
            ["data10m_test_data30m_train_disjoint", "legacy_seed_validation_leakage_clean", "legacy_seed_test"],
        )
        self.assertEqual(self.protocol["baseline"]["checkpoint_sha256"], "a9b53087a5c0dee6c2a8640999ac88d54cfb62a2d94833bc737f968f267c01fe")

    def test_recall_aggregation_is_exact_and_stratified(self):
        trials = [
            {"exact_match": True, "state_type": "goal", "distance_bucket": "d64", "difficulty": 1, "genre": "noir"},
            {"exact_match": False, "state_type": "goal", "distance_bucket": "d128", "difficulty": 2, "genre": "noir"},
        ]
        result = aggregate_recall(trials)
        self.assertEqual(result["aggregate"], {"correct": 1, "total": 2, "accuracy": 0.5})
        self.assertEqual(result["by_state_type"]["goal"]["total"], 2)


if __name__ == "__main__":
    unittest.main()
