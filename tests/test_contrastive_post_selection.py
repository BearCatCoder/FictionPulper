import copy
import json
import unittest
from pathlib import Path

from src.contrastive_post_selection import (
    EXPECTED_BENCHMARK_PROTOCOL_SHA256,
    aggregate_pair_results,
    run_after_gate,
    verify_protocol,
)
from src.train import GENERATION_SETTINGS


PROTOCOL_PATH = Path("experiments/fictionpulper-15m-data30m-contrastive-v1/post-selection-protocol.json")


class ContrastivePostSelectionTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_selection_gate_precedes_heldout_loader(self):
        accesses = []

        def gate():
            accesses.append("gate")
            raise RuntimeError("incomplete")

        def heldout(*_):
            accesses.append("heldout")
            return {}

        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            run_after_gate(gate, heldout)
        self.assertEqual(accesses, ["gate"])

    def test_benchmark_protocol_hash_is_fixed(self):
        self.assertEqual(self.protocol["benchmark_protocol"]["sha256"], EXPECTED_BENCHMARK_PROTOCOL_SHA256)
        changed = copy.deepcopy(self.protocol)
        changed["benchmark_protocol"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "not fixed"):
            verify_protocol(changed)

    def test_pair_grouping_uses_all_required_dimensions(self):
        trials = [
            {"correct": 1.0, "margin": 2.0, "state_type": "goal", "distance_bucket": "near", "difficulty": 1},
            {"correct": 0.0, "margin": -1.0, "state_type": "goal", "distance_bucket": "far", "difficulty": 2},
            {"correct": 1.0, "margin": 1.0, "state_type": "secret", "distance_bucket": "far", "difficulty": 2},
        ]
        result = aggregate_pair_results(trials)
        self.assertEqual(result["aggregate"], {"count": 3, "pairwise_accuracy": 2 / 3, "mean_margin": 2 / 3})
        self.assertEqual(result["by_state_type"]["goal"]["count"], 2)
        self.assertEqual(result["by_distance_bucket"]["far"]["pairwise_accuracy"], 0.5)
        self.assertEqual(result["by_difficulty"]["1"]["mean_margin"], 2.0)

    def test_generation_settings_and_counts_are_exact(self):
        self.assertEqual(self.protocol["generation"]["settings"], GENERATION_SETTINGS)
        self.assertEqual(self.protocol["generation"]["sample_seed"], 11337)
        self.assertEqual(self.protocol["narrative_protocol"]["fact_count"], 13)
        self.assertEqual(self.protocol["narrative_protocol"]["prompt_count"], 3)
        verify_protocol(self.protocol)


if __name__ == "__main__":
    unittest.main()
