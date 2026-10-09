import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.benchmark_v2 import (
    BenchmarkValidationError,
    EXPECTED_PROTOCOL_SHA256,
    build_allocation_plan,
    stable_hash,
    validate_benchmark,
    validate_plan,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "benchmarks/narrative-v2"


class BenchmarkV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((BENCHMARK_ROOT / "protocol.json").read_text(encoding="utf-8"))

    def test_frozen_artifacts_validate_with_exact_balance_and_no_leakage(self):
        report = validate_benchmark(BENCHMARK_ROOT)
        self.assertTrue(report["passed"])
        self.assertEqual(report["scenario_count"], 112)
        self.assertEqual(report["split_counts"], {"development": 28, "generalization": 56, "test": 28})
        self.assertEqual(report["family_depth_cells"], 28)
        self.assertEqual(report["scenarios_per_family_depth_cell"], [4])
        self.assertEqual(report["candidate_orders"], {"XY": 56, "YX": 56})
        self.assertEqual(report["world_a_labels"], {"X": 56, "Y": 56})
        self.assertTrue(all(not values for values in report["allocation_id_leakage"].values()))
        self.assertEqual(len(EXPECTED_PROTOCOL_SHA256), 64)

    def test_plan_is_deterministic_and_has_all_three_distinct_modes(self):
        first = build_allocation_plan()
        self.assertEqual(first, build_allocation_plan())
        self.assertTrue(all(record["evaluation_modes"] == [
            "forced_choice", "teacher_forced", "free_generation"
        ] for record in first))
        self.assertTrue(all(record["reasoning_depth"] >= 4 for record in first if record["depth_bucket"] == "4_plus"))
        for field in ("split", "depth_bucket", "reasoning_depth", "state_family"):
            values = {}
            for record in first:
                counts = values.setdefault(record[field], {("X", "Y"): 0, ("Y", "X"): 0})
                counts[tuple(record["candidate_presentation_order"])] += 1
            self.assertTrue(all(abs(counts[("X", "Y")] - counts[("Y", "X")]) <= 1 for counts in values.values()))

    def test_schema_requires_array_order_and_modes(self):
        schema = json.loads((BENCHMARK_ROOT / "scenario.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["candidate_presentation_order"]["type"], "array")
        self.assertEqual(schema["properties"]["evaluation_modes"]["type"], "array")

    def test_cross_split_lexical_leakage_is_rejected_even_with_valid_hash(self):
        records = build_allocation_plan()
        records[1]["name_pool_id"] = records[0]["name_pool_id"]
        records[1]["stable_hash"] = stable_hash({key: value for key, value in records[1].items() if key != "stable_hash"})
        with self.assertRaisesRegex(BenchmarkValidationError, "crosses splits"):
            validate_plan(records, self.protocol)

    def test_family_depth_imbalance_is_rejected(self):
        records = build_allocation_plan()
        records[-1]["state_family"] = "static_fact"
        records[-1]["stable_hash"] = stable_hash({key: value for key, value in records[-1].items() if key != "stable_hash"})
        with self.assertRaises(BenchmarkValidationError):
            validate_plan(records, self.protocol)

    def test_artifact_hash_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("protocol.json", "scenario-allocation.jsonl", "scenario.schema.json", "human-review-rubric.json"):
                (root / name).write_bytes((BENCHMARK_ROOT / name).read_bytes())
            rubric = root / "human-review-rubric.json"
            rubric.write_text(rubric.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(BenchmarkValidationError, "hash mismatch"):
                validate_benchmark(root)

    def test_stable_hash_tampering_is_rejected(self):
        records = copy.deepcopy(build_allocation_plan())
        records[0]["stable_hash"] = "0" * 64
        with self.assertRaises(BenchmarkValidationError):
            validate_plan(records, self.protocol)


if __name__ == "__main__":
    unittest.main()
