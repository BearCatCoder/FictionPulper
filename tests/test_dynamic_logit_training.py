import copy
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src.model import FictionPulperLM, ModelConfig, model_config_from_dict
from src.train_dynamic_logit import (
    LOCKED_DYNAMIC_LAMBDA,
    LOCKED_STALE_LAMBDA,
    aggregate_dynamic_metrics,
    batched_candidate_scores,
    candidate_set_cross_entropy,
    create_fresh_model,
    dynamic_objective,
    group_candidate_records,
    selected_pair_decisions,
    stable_token_unlikelihood,
    stale_unlikelihood_loss,
    validate_locked_config,
)
from src.train_mixed import LOCKED_PARAMETER_COUNT


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "data30m-15m-counterfactual-v1.yaml"


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=16, hidden_size=16, num_layers=1, num_attention_heads=2,
        num_key_value_heads=1, intermediate_size=24, max_seq_len=16,
    )


def record(candidate, category, tokens, *, pair="p", world="A", boundary="transition", depth=1,
           family="object_location", distance=0, aliases=None):
    return {
        "pair": pair, "world": world, "boundary": boundary, "depth": depth,
        "candidate": candidate, "category": category, "input_token_ids": tokens,
        "all_applicable_categories": aliases or [category],
        "decision_start": 1, "decision_end": len(tokens) - 1,
        "family": family, "distance": distance,
    }


class RecordingModel:
    def __init__(self, logits):
        self.logits = logits
        self.labels = []
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def parameters(self):
        return iter((self.anchor,))

    def __call__(self, inputs, labels=None):
        self.labels.append(labels)
        return self.logits[:inputs.shape[0], :inputs.shape[1]] + self.anchor * 0.0, None


class DynamicLossTests(unittest.TestCase):
    def test_batched_multi_token_scores_are_length_normalized(self):
        logits = torch.zeros(2, 3, 5)
        model = RecordingModel(logits)
        records = [
            record("X", "CURRENT", [0, 2, 0]),
            record("Y", "PREVIOUS", [0, 2, 2, 0]),
        ]
        scores, _ = batched_candidate_scores(model, records, torch.device("cpu"))
        self.assertTrue(torch.allclose(scores[0], scores[1]))
        self.assertEqual(model.labels, [None])

    def test_candidate_set_cross_entropy(self):
        scores = torch.tensor([2.0, 0.0, -1.0])
        expected = F.cross_entropy(scores.unsqueeze(0), torch.tensor([0]))
        self.assertTrue(torch.allclose(candidate_set_cross_entropy(scores, 0), expected))

    def test_stale_unlikelihood_is_stable_at_extreme_logits(self):
        high = stable_token_unlikelihood(torch.tensor([[1000.0, -1000.0]]), torch.tensor([0]))
        low = stable_token_unlikelihood(torch.tensor([[-1000.0, 1000.0]]), torch.tensor([0]))
        self.assertTrue(torch.isfinite(high))
        self.assertTrue(torch.isfinite(low))
        self.assertGreater(float(high), 1000.0)
        self.assertLess(float(low), 1e-6)

    def test_stale_mask_excludes_current_and_never_valid(self):
        records = [
            record("A", "CURRENT", [0, 1, 0]),
            record("B", "PREVIOUS", [0, 2, 0]),
            record("C", "INITIAL", [0, 3, 0]),
            record("D", "OLDER", [0, 4, 0]),
            record("E", "NEVER_VALID", [0, 5, 0]),
        ]
        logits = torch.zeros(5, 2, 8)
        loss, count = stale_unlikelihood_loss(logits, records)
        expected = -torch.log1p(-torch.tensor(1 / 8))
        self.assertEqual(count, 3)
        self.assertTrue(torch.allclose(loss, expected))

    def test_dynamic_forward_uses_labels_none_and_no_negative_ce(self):
        logits = torch.zeros(2, 2, 8, requires_grad=True)
        logits.data[0, 0, 1] = 2.0
        model = RecordingModel(logits)
        decision = [[
            record("X", "CURRENT", [0, 1, 0]),
            record("Y", "NEVER_VALID", [0, 2, 0]),
        ]]
        dynamic, stale, results = dynamic_objective(model, decision, torch.device("cpu"))
        self.assertEqual(model.labels, [None])
        self.assertEqual(float(stale.detach()), 0.0)
        self.assertGreater(float(dynamic.detach()), 0.0)
        self.assertEqual(results[0]["selected_category"], "CURRENT")

    def test_historical_aliases_share_the_candidate_margin(self):
        logits = torch.zeros(2, 2, 8, requires_grad=True)
        logits.data[0, 0, 1] = 2.0
        model = RecordingModel(logits)
        decision = [[
            record("X", "CURRENT", [0, 1, 0], aliases=["CURRENT"]),
            record("Y", "PREVIOUS", [0, 2, 0], aliases=["PREVIOUS", "INITIAL"]),
        ]]
        _, _, results = dynamic_objective(model, decision, torch.device("cpu"))
        self.assertEqual(
            results[0]["current_vs_previous_margin"],
            results[0]["current_vs_initial_margin"],
        )

    def test_category_margin_and_dimension_aggregation(self):
        results = [
            {"loss": 0.2, "current_correct": 1.0, "selected_category": "CURRENT",
             "family": "location", "depth": "2", "distance": "4", "boundary": "final",
             "current_vs_previous_margin": 2.0, "current_vs_initial_margin": None,
             "current_vs_older_margin": None, "current_vs_never_valid_margin": -1.0},
            {"loss": 0.4, "current_correct": 0.0, "selected_category": "PREVIOUS",
             "family": "owner", "depth": "1", "distance": "0", "boundary": "transition",
             "current_vs_previous_margin": -2.0, "current_vs_initial_margin": 1.0,
             "current_vs_older_margin": None, "current_vs_never_valid_margin": None},
        ]
        grouped = aggregate_dynamic_metrics(results)
        self.assertEqual(grouped["overall"]["current_accuracy"], 0.5)
        self.assertEqual(grouped["overall"]["selected_current_rate"], 0.5)
        self.assertEqual(grouped["overall"]["current_vs_previous_margin"], 0.0)
        for dimension in ("selected_category", "family", "depth", "distance", "boundary"):
            self.assertIn(f"by_{dimension}", grouped)


class GroupingAndGradientTests(unittest.TestCase):
    def test_grouping_and_schedule_selection_are_deterministic(self):
        records = [
            record("Y", "PREVIOUS", [0, 2, 0], pair="p2"),
            record("Y", "NEVER_VALID", [0, 2, 0], pair="p1"),
            record("X", "CURRENT", [0, 1, 0], pair="p2"),
            record("X", "CURRENT", [0, 1, 0], pair="p1"),
        ]
        first = group_candidate_records(records)
        second = group_candidate_records(reversed(records))
        pairs = [{"pair_id": "p2"}, {"pair_id": "p1"}]
        selected_first = selected_pair_decisions(pairs, first)
        selected_second = selected_pair_decisions(pairs, second)
        self.assertEqual(selected_first, selected_second)
        self.assertEqual([item[0]["pair"] for item in selected_first], ["p1", "p2"])
        self.assertEqual([item["candidate"] for item in selected_first[0]], ["X", "Y"])

    def test_direct_loss_reaches_lm_and_tied_embedding(self):
        torch.manual_seed(9)
        model = FictionPulperLM(tiny_config())
        self.assertIs(model.lm_head.weight, model.embed_tokens.weight)
        decisions = [[
            record("X", "CURRENT", [1, 2, 3, 0]),
            record("Y", "PREVIOUS", [1, 4, 5, 0]),
        ]]
        dynamic, stale, _ = dynamic_objective(model, decisions, torch.device("cpu"))
        (dynamic + 0.1 * stale).backward()
        self.assertIsNotNone(model.embed_tokens.weight.grad)
        self.assertGreater(float(model.embed_tokens.weight.grad.norm()), 0.0)
        self.assertTrue(any(parameter.grad is not None and float(parameter.grad.norm()) > 0
                            for parameter in model.layers.parameters()))


class LockAndInitializationTests(unittest.TestCase):
    def dynamic_config(self):
        config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
        name = "fictionpulper-15m-data30m-dynamic-logit-v1"
        config["experiment_id"] = name
        config["training"]["checkpoint_dir"] = f"checkpoints/{name}"
        config["logging"]["run_dir"] = f"runs/{name}"
        config["dynamic_logit_data"] = {
            "manifest_path": "data/dynamic_logit_v1/manifest.json",
            "train_path": "data/dynamic_logit_v1/train.jsonl",
            "validation_path": "data/dynamic_logit_v1/validation.jsonl",
            "expected_manifest_sha256": "PENDING_MANIFEST",
            "expected_train_sha256": "PENDING_TRAIN",
            "expected_validation_sha256": "PENDING_VALIDATION",
            "expected_transition_manifest_sha256": "PENDING_TRANSITION",
        }
        config["dynamic_logit_objective"] = {
            "version": "Dynamic-logit-v1", "pair_lambda": 0.25,
            "dynamic_lambda": 0.25, "stale_lambda": 0.10,
            "score": "mean_log_probability_exact_decision_span",
            "dynamic_loss": "candidate_set_cross_entropy",
            "stale_loss": "teacher_forced_token_unlikelihood",
            "stale_categories": ["PREVIOUS", "INITIAL", "OLDER"],
            "labels": None, "auxiliary_heads": False,
            "expected_decision_presentations": "PENDING_DECISIONS",
            "expected_stale_candidate_presentations": "PENDING_STALE",
        }
        config["selection"] = {
            "metric": "data30m_validation_loss",
            "dynamic_validation_role": "held_out_diagnostic_only_not_selection",
            "test_access": "forbidden_during_training",
            "generalization_access": "forbidden_during_training",
        }
        return config

    def test_config_locks_accept_protocol_and_reject_drift(self):
        config = self.dynamic_config()
        validate_locked_config(config, require_generated_locks=False)
        self.assertEqual(LOCKED_DYNAMIC_LAMBDA, 0.25)
        self.assertEqual(LOCKED_STALE_LAMBDA, 0.10)
        for section, key, value in (
            ("training", "max_steps", 2219),
            ("dynamic_logit_objective", "stale_lambda", 0.2),
            ("dynamic_logit_objective", "stale_categories", ["NEVER_VALID"]),
            ("dynamic_logit_data", "validation_path", "data/dynamic_logit_v1/test.jsonl"),
        ):
            changed = copy.deepcopy(config)
            changed[section][key] = value
            with self.subTest(section=section, key=key):
                with self.assertRaises(RuntimeError):
                    validate_locked_config(changed, require_generated_locks=False)

    def test_parameter_count_tying_and_fresh_seed_initialization(self):
        config = model_config_from_dict(yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))["model"])
        first = create_fresh_model(config)
        sample = first.embed_tokens.weight[0, :8].detach().clone()
        self.assertEqual(first.trainable_parameter_count(), 15_047_040)
        self.assertEqual(first.trainable_parameter_count(), LOCKED_PARAMETER_COUNT)
        self.assertIs(first.embed_tokens.weight, first.lm_head.weight)
        del first
        second = create_fresh_model(config)
        self.assertTrue(torch.equal(sample, second.embed_tokens.weight[0, :8]))


if __name__ == "__main__":
    unittest.main()
