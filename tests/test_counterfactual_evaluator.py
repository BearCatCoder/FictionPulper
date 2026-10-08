import copy
import tempfile
import unittest
from pathlib import Path

import torch

from src.continuity_curriculum.common import sha256_file
from src.counterfactual_narrative.evaluator import (
    aggregate_trials,
    audit_gate,
    pair_metrics,
    score_token_spans,
    verify_inputs,
)


class NextTokenModel(torch.nn.Module):
    def __init__(self, vocabulary_size=8):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.vocabulary_size = vocabulary_size

    def forward(self, tokens):
        batch, length = tokens.shape
        logits = torch.zeros(batch, length, self.vocabulary_size, device=tokens.device) + self.anchor * 0
        for row in range(batch):
            for position in range(length):
                logits[row, position, (int(tokens[row, position]) + 1) % self.vocabulary_size] = 4.0
        return logits, None


def synthetic_pair():
    return {
        "pair_id": "pair-1", "split": "validation",
        "abstract_counterfactual_variable": {"state_family": "object_location"},
        "metadata": {"distance_bucket": "d128", "difficulty": 2},
        "worlds": {"A": {"correct_candidate": "X"}, "B": {"correct_candidate": "Y"}},
    }


def latest_state_pair():
    pair = synthetic_pair()
    pair["pair_id"] = "pair-latest"
    pair["abstract_counterfactual_variable"]["state_family"] = "latest_state_update"
    return pair


class CounterfactualEvaluatorTests(unittest.TestCase):
    def test_exact_decision_span_mean_and_batched_padding(self):
        model = NextTokenModel()
        examples = [
            {"token_ids": [0, 1, 2], "decision_start": 1, "decision_end": 3},
            {"token_ids": [3, 4], "decision_start": 1, "decision_end": 2},
        ]
        scores = score_token_spans(model, examples, device=torch.device("cpu"), batch_size=2,
                                   pad_id=7, use_bf16=False)
        expected = torch.log_softmax(torch.tensor([0.0, 0, 0, 0, 4, 0, 0, 0]), -1)[4]
        self.assertAlmostEqual(scores[1], float(expected), places=6)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_pair_metrics_reversal_controls_and_candidate_only(self):
        scores = {
            ("normal", "A", "X"): -1.0, ("normal", "A", "Y"): -2.0,
            ("normal", "B", "X"): -3.0, ("normal", "B", "Y"): -1.0,
            ("fact_removed", None, "X"): -1.0, ("fact_removed", None, "Y"): -1.5,
            ("irrelevant_substituted", None, "X"): -2.0, ("irrelevant_substituted", None, "Y"): -1.0,
            ("candidate_only", None, "X"): -1.0, ("candidate_only", None, "Y"): -1.0,
        }
        trial = pair_metrics(synthetic_pair(), scores, 1e-12)
        self.assertTrue(trial["joint_paired_reversal_success"])
        self.assertEqual(trial["signed_reversal_margin"], 1.5)
        summary = aggregate_trials([trial])["overall"]
        self.assertEqual(summary["normal"]["joint_chance_reference"], 0.25)
        self.assertEqual(summary["candidate_only"]["tie_rate"], 1.0)
        for condition in ("fact_removed", "irrelevant_substituted"):
            self.assertFalse(summary[condition]["paired_success_defined"])
            self.assertIsNone(summary[condition]["joint_paired_reversal_success"])
            self.assertEqual(summary[condition]["balanced_directional_accuracy_ties_half_credit"], 0.5)
        latest_trial = pair_metrics(latest_state_pair(), scores, 1e-12)
        latest = aggregate_trials([trial, latest_trial])["latest_state_subset"]
        self.assertTrue(latest["present"])
        self.assertEqual(latest["normal"]["direction_accuracy"], 1.0)

    def test_gate_rejects_persistent_context_free_side_winner(self):
        good = {"metrics": {"overall": {
            "candidate_only": {"X_win_rate": 0.5, "Y_win_rate": 0.4},
            "fact_removed": {"shared_context_per_pair": True, "balanced_directional_accuracy_ties_half_credit": 0.5},
            "irrelevant_substituted": {"shared_context_per_pair": True, "balanced_directional_accuracy_ties_half_credit": 0.5},
        }}}
        models = {"toy": {"validation": good}}
        self.assertTrue(audit_gate(models, 0.55)["passed"])
        bad = copy.deepcopy(models)
        bad["toy"]["validation"]["metrics"]["overall"]["candidate_only"]["X_win_rate"] = 0.56
        self.assertFalse(audit_gate(bad, 0.55)["passed"])

    def test_verification_checks_manifest_files_seal_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text("tokenizer", encoding="utf-8")
            dataset = root / "dataset"
            dataset.mkdir()
            files = {}
            for split in ("validation", "test", "generalization_holdout"):
                path = dataset / f"{split}.jsonl"
                path.write_text("{}\n", encoding="utf-8")
                files[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            manifest = dataset / "manifest.json"
            manifest.write_text(__import__("json").dumps({
                "status": "PASS", "safe_to_train": True,
                "tokenizer_sha256": sha256_file(tokenizer), "files": files,
            }), encoding="utf-8")
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            seal = root / "seal.json"
            seal.write_text(__import__("json").dumps({
                "status": "sealed", "run_id": "sealed-toy",
                "tokenizer_sha256": sha256_file(tokenizer),
                "checkpoint_sha256": {"best_validation": sha256_file(checkpoint)},
            }), encoding="utf-8")
            config = {
                "tokenizer": {"path": str(tokenizer), "sha256": sha256_file(tokenizer)},
                "dataset": {"directory": str(dataset), "manifest_sha256": sha256_file(manifest)},
                "checkpoints": [{"id": "toy", "sealed_run_id": "sealed-toy",
                    "path": str(checkpoint), "sha256": sha256_file(checkpoint),
                    "seal_path": str(seal), "seal_sha256": sha256_file(seal)}],
            }
            self.assertEqual(verify_inputs(config)["checkpoints"]["toy"]["sha256"], sha256_file(checkpoint))
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "Hash mismatch"):
                verify_inputs(config)


if __name__ == "__main__":
    unittest.main()
