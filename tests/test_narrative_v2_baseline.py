import copy
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
from tokenizers import Tokenizer

from src.continuity_curriculum.common import sha256_file
from src.narrative_v2_baseline import (
    MODEL_REGISTRY,
    aggregate_forced_choice,
    aggregate_teacher_forced,
    evaluate_forced_choice,
    evaluate_teacher_forced,
    run,
    score_span,
    verify_sealed_model,
)
from src.narrative_v2_loader import load_authored_split


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "benchmarks/narrative-v2"


class PredictNextModel:
    class Config:
        max_seq_len = 128

    config = Config()

    def __call__(self, input_ids):
        vocab_size = 32
        logits = torch.full((1, input_ids.shape[1], vocab_size), -20.0)
        for index in range(input_ids.shape[1] - 1):
            logits[0, index, input_ids[0, index + 1]] = 20.0
        return logits, None


def forced_record(scenario: str, world: str, correct: bool, *, family: str = "goal", depth: str = "2") -> dict:
    return {
        "scenario_id": scenario,
        "state_family": family,
        "depth_bucket": depth,
        "context_kind": "main",
        "world": world,
        "correct": correct,
        "tie": False,
        "signed_correct_margin": 1.0 if correct else -1.0,
    }


def teacher_record(scenario: str, world: str, preferred: bool) -> dict:
    return {
        "scenario_id": scenario,
        "state_family": "goal",
        "depth_bucket": "2",
        "world": world,
        "gold_preferred": preferred,
        "tie": False,
        "gold_preference_margin": 0.5 if preferred else -0.5,
        "gold": {"nll": 2.0},
        "counterfactual": {"nll": 2.5},
    }


class NarrativeV2BaselineTests(unittest.TestCase):
    def test_exact_span_scoring_excludes_prefix(self):
        result = score_span(
            cast(Any, PredictNextModel()), [8, 9, 10], [11, 12],
            device=torch.device("cpu"), use_bf16=False,
        )
        self.assertEqual(result["token_ids"], [11, 12])
        self.assertEqual(result["token_count"], 2)
        self.assertLess(result["nll"], 1e-6)

    def test_authored_candidate_spans_and_all_controls_are_scored(self):
        tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer/tokenizer.json"))
        scenario = load_authored_split(BENCHMARK_ROOT)[:1]

        def scorer(prefix, span):
            value = -float(sum(span)) / len(span)
            return {"token_ids": list(span), "token_count": len(span), "mean_log_probability": value, "nll": -value}

        forced = evaluate_forced_choice(scenario, tokenizer, scorer)
        self.assertEqual(len(forced), 5)
        self.assertEqual(
            {row["context_kind"] for row in forced},
            {"main", "candidate_only", "fact_removed", "irrelevant_substituted"},
        )
        self.assertEqual(sum(row["context_kind"] == "main" for row in forced), 2)

        malformed = copy.deepcopy(scenario)
        malformed[0]["candidates"]["X"]["token_ids"][0] += 1
        with self.assertRaisesRegex(RuntimeError, "token IDs changed"):
            evaluate_forced_choice(malformed, tokenizer, scorer)

    def test_aggregation_keeps_pair_success_world_accuracy_and_strata_distinct(self):
        records = [
            forced_record("one", "A", True), forced_record("one", "B", True),
            forced_record("two", "A", True), forced_record("two", "B", False),
        ]
        for kind in ("candidate_only", "fact_removed", "irrelevant_substituted"):
            for scenario, chosen in (("one", "X"), ("two", "Y")):
                records.append({
                    "scenario_id": scenario, "state_family": "goal", "depth_bucket": "2",
                    "context_kind": kind, "preferred_candidate": chosen, "tie": False,
                    "x_minus_y_margin": 1.0 if chosen == "X" else -1.0,
                })
        summary = aggregate_forced_choice(records)
        overall = next(row for row in summary["metrics"] if row["dimension"] == "overall")
        self.assertEqual(overall["world_accuracy"], 0.75)
        self.assertEqual(overall["joint_paired_reversal_success"], 0.5)
        self.assertEqual({row["context_kind"] for row in summary["controls"]}, {
            "candidate_only", "fact_removed", "irrelevant_substituted"
        })
        self.assertEqual({row["dimension"] for row in summary["metrics"]}, {"overall", "family", "depth"})

    def test_teacher_forced_scores_full_continuations_and_joint_preference(self):
        tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer/tokenizer.json"))
        scenario = load_authored_split(BENCHMARK_ROOT)[:1]

        def scorer(prefix, span):
            return {"token_ids": list(span), "token_count": len(span), "mean_log_probability": -1.0, "nll": 1.0}

        rows = evaluate_teacher_forced(scenario, tokenizer, scorer)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["gold"]["token_count"] > len(scenario[0]["candidates"]["X"]["token_ids"]) for row in rows))

        summary = aggregate_teacher_forced([
            teacher_record("one", "A", True), teacher_record("one", "B", True),
            teacher_record("two", "A", True), teacher_record("two", "B", False),
        ])
        overall = next(row for row in summary["metrics"] if row["dimension"] == "overall")
        self.assertEqual(overall["world_gold_preference_rate"], 0.75)
        self.assertEqual(overall["joint_gold_over_counterfactual_preference"], 0.5)

    def test_malformed_pairs_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "malformed A/B"):
            aggregate_forced_choice([forced_record("one", "A", True)])
        with self.assertRaisesRegex(RuntimeError, "malformed A/B"):
            aggregate_teacher_forced([teacher_record("one", "A", True)])

    def test_seal_checkpoint_and_parent_seal_hashes_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seal_path = root / "seal.json"
            checkpoint_path = root / "checkpoint.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            seal = {
                "status": "sealed", "run_id": "test-50m", "tokenizer_sha256": "tokenizer",
                "checkpoint_sha256": {"best_validation": sha256_file(checkpoint_path)},
                "baseline_seal_sha256": {"compute5": "parent"},
            }
            seal_path.write_text(json.dumps(seal), encoding="utf-8")
            registry = {
                "compute5": {"seal_sha256": "parent"},
                "50m": {
                    "model_id": "test-50m", "checkpoint": "checkpoint.pt",
                    "checkpoint_sha256": sha256_file(checkpoint_path), "seal": "seal.json",
                    "seal_sha256": sha256_file(seal_path),
                },
            }
            with patch.dict(MODEL_REGISTRY, registry, clear=True), patch(
                "src.narrative_v2_baseline.TOKENIZER_SHA256", "tokenizer"
            ):
                verify_sealed_model("50m", root)
                seal["baseline_seal_sha256"]["compute5"] = "wrong"
                seal_path.write_text(json.dumps(seal), encoding="utf-8")
                registry["50m"]["seal_sha256"] = sha256_file(seal_path)
                with self.assertRaisesRegex(RuntimeError, "Compute5 seal"):
                    verify_sealed_model("50m", root)

    def test_existing_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            args = Namespace(output_dir=output)
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                run(args)

    def test_registry_contains_only_exact_requested_baselines(self):
        self.assertEqual(set(MODEL_REGISTRY), {"compute5", "50m"})
        self.assertEqual(
            MODEL_REGISTRY["compute5"]["checkpoint_sha256"],
            "a9b53087a5c0dee6c2a8640999ac88d54cfb62a2d94833bc737f968f267c01fe",
        )
        self.assertEqual(
            MODEL_REGISTRY["50m"]["checkpoint_sha256"],
            "8da266fb064719d3f38e78d672b2e5cec0f14551c37658210322b9f46cbecd01",
        )


if __name__ == "__main__":
    unittest.main()
