import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluate_state_transition import aggregate_auxiliary
from src.state_transition_post_selection import run_after_gate, verify_protocol
from src.train import GENERATION_SETTINGS
from src.train_tokenizer import sha256_file


class StateTransitionPostSelectionTests(unittest.TestCase):
    def test_gate_precedes_heldout_access(self):
        calls = []
        def gate():
            calls.append("gate")
            raise RuntimeError("blocked")
        def heldout(*_):
            calls.append("heldout")
        with self.assertRaisesRegex(RuntimeError, "blocked"):
            run_after_gate(gate, heldout)
        self.assertEqual(calls, ["gate"])

    def test_auxiliary_aggregation_baselines_latest_and_depth(self):
        def row(event, kind, label, correct, ce, distance=None, pair="p"):
            return {"pair": pair, "world": "A", "entity_slot": "object_0", "event": event,
                    "kind": kind, "family": "object_location", "label": label,
                    "correct": correct, "ce": ce, "source_distance_bucket": "d064",
                    "carry_distance": distance, "difficulty": "1", "genre": "mystery",
                    "transition_depth": event + 1, "transition_count": 3}
        rows = [
            row(0, "transition", "location_0", 1, 0.1),
            row(1, "transition", "location_1", 0, 0.9),
            row(2, "transition", "location_1", 1, 0.2),
            row(2, "carry", "location_1", 1, 0.3, 32),
            row(0, "transition", "location_0", 0, 0.8, pair="q"),
        ]
        result = aggregate_auxiliary(rows)
        self.assertEqual(result["transition"]["count"], 4)
        self.assertEqual(result["carry"]["accuracy"], 1.0)
        self.assertEqual(result["latest_current_state"]["final_transition"]["count"], 2)
        self.assertEqual(result["latest_current_state"]["valid_final_carries"]["count"], 1)
        self.assertEqual(result["transition_depth"]["1"]["unavailable_entity_sequences"], 0)
        self.assertEqual(result["transition_depth"]["2"]["unavailable_entity_sequences"], 1)
        self.assertEqual(result["transition_depth"]["3"]["unavailable_entity_sequences"], 1)
        self.assertEqual(result["transition_depth"]["4+"]["unavailable_entity_sequences"], 2)
        baseline = result["baselines_by_family"]["object_location"]
        self.assertEqual(baseline["chance_accuracy"], 0.5)
        self.assertEqual(baseline["majority_accuracy"], 3 / 5)

    def test_protocol_rejects_hash_and_in_memory_drift(self):
        protocol = {
            "version": 1, "experiment_id": "fictionpulper-15m-data30m-state-transition-v1",
            "candidate_key": "state_transition_v1", "selection_metric": "Data30M validation loss only",
            "generation": {"settings": GENERATION_SETTINGS, "sample_seed": 11337},
            "counterfactual": {"splits": ["test", "generalization_holdout"]},
            "auxiliary": {"splits": ["test", "generalization"]},
            "forced_choice": {}, "probe": {}, "real_fiction": {},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            locked = root / "locked.txt"; locked.write_text("locked", encoding="utf-8")
            baseline = root / "baseline.json"; baseline.write_text("{}", encoding="utf-8")
            for section in ("counterfactual", "forced_choice", "probe"):
                protocol[section].update({"evaluator_path" if section == "counterfactual" else "protocol_path" if section == "forced_choice" else "method_path": str(locked)})
            protocol["counterfactual"]["evaluator_sha256"] = sha256_file(locked)
            protocol["forced_choice"].update({"protocol_sha256": sha256_file(locked), "narrative_protocol_path": str(locked), "narrative_protocol_sha256": sha256_file(locked)})
            protocol["probe"].update({"method_sha256": sha256_file(locked), "config_path": str(locked), "config_sha256": sha256_file(locked)})
            for section in ("counterfactual", "forced_choice", "generation", "real_fiction", "probe"):
                protocol[section].update({"sealed_baseline_path": str(baseline), "sealed_baseline_sha256": sha256_file(baseline)})
            protocol["probe"].update({"sealed_raw_path": str(baseline), "sealed_raw_sha256": sha256_file(baseline)})
            path = root / "protocol.json"; path.write_text(json.dumps(protocol), encoding="utf-8")
            digest = sha256_file(path)
            verify_protocol(protocol, path, digest)
            changed = copy.deepcopy(protocol); changed["selection_metric"] = "heldout accuracy"
            with self.assertRaisesRegex(RuntimeError, "Selection metric changed"):
                verify_protocol(changed, path, digest)
            with self.assertRaisesRegex(RuntimeError, "protocol hash changed"):
                verify_protocol(protocol, path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
