import copy
import json
import unittest
from pathlib import Path

import torch

from src.counterfactual_post_selection import (
    EXPECTED_PROTOCOL_SHA256,
    PROTOCOL_PATH,
    run_after_gate,
    verify_protocol,
    verify_selection_gate,
)
from src.report_counterfactual import (
    CLASSIFICATION,
    EXPERIMENT_DIR,
    POST_DIR,
    compact_manual,
    validate_manual,
)
from src.train import GENERATION_SETTINGS
from src.train_tokenizer import sha256_file


class CounterfactualPostSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_protocol_is_immutable_and_generation_settings_are_exact(self):
        self.assertEqual(sha256_file(PROTOCOL_PATH), EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(self.protocol["generation"]["settings"], GENERATION_SETTINGS)
        self.assertEqual(self.protocol["counterfactual"]["splits"], ["test", "generalization_holdout"])
        verify_protocol(self.protocol)

    def test_gate_runs_before_any_heldout_callback(self):
        accesses = []

        def gate():
            accesses.append("gate")
            raise RuntimeError("blocked")

        def heldout(*_):
            accesses.append("heldout")

        with self.assertRaisesRegex(RuntimeError, "blocked"):
            run_after_gate(gate, heldout)
        self.assertEqual(accesses, ["gate"])

    def test_protocol_rejects_in_memory_semantic_drift(self):
        changed = copy.deepcopy(self.protocol)
        changed["selection_metric"] = "counterfactual test accuracy"
        with self.assertRaisesRegex(RuntimeError, "Selection metric changed"):
            verify_protocol(changed)

    def test_real_selection_gate_proves_final_minimum_and_locks(self):
        summary, checkpoint, path, proof = verify_selection_gate(
            self.protocol,
            load_checkpoint=lambda checkpoint_path: torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            ),
        )
        self.assertEqual(proof["status"], "PASS")
        self.assertFalse(proof["heldout_opened"])
        self.assertEqual(summary["best_checkpoint_step"], 2220)
        self.assertEqual(checkpoint["step"], 2220)
        self.assertEqual(path, Path(self.protocol["candidate"]["checkpoint_path"]))
        eligible = [item for item in summary["metrics"] if item.get("selection_eligible")]
        self.assertEqual(
            summary["best_data30m_validation_loss"],
            min(item["data30m_validation"]["loss"] for item in eligible),
        )

    def test_candidate_manual_scoring_is_complete_and_evidenced(self):
        manual = json.loads((POST_DIR / "manual-fact-scoring.json").read_text(encoding="utf-8"))
        rows = validate_manual(manual)
        compact = compact_manual(manual, rows)
        self.assertEqual(compact["candidate_totals"], {"greedy": "0/13", "sampled": "0/13"})
        self.assertEqual(len(compact["facts"]), 13)

    def test_final_report_records_conservative_classification(self):
        summary = json.loads((EXPERIMENT_DIR / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["classification"], CLASSIFICATION)
        self.assertEqual(summary["manual_free_generation"]["counterfactual_v1"], {"greedy": "0/13", "sampled": "0/13"})
        self.assertEqual(summary["latest_state_update"]["test"]["normal"]["joint_paired_reversal_success"], 0.0)


if __name__ == "__main__":
    unittest.main()
