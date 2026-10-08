import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src.model import FictionPulperLM, ModelConfig, model_config_from_dict
from src.train_mixed import LOCKED_PARAMETER_COUNT
from src.train_state_transition import (
    CANONICAL_CLASS_MAPS,
    REPRESENTATION,
    build_state_heads,
    check_lm_only_equivalence,
    create_fresh_model_and_heads,
    export_lm_only,
    head_config,
    load_checkpoint_for_validation,
    normalized_state_loss,
    parameter_counts,
    preflight_numerical_sanity,
    save_checkpoint,
    state_examples_for_pairs,
    state_loss_for_examples,
    validate_locked_config,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "data30m-15m-counterfactual-v1.yaml"


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32, hidden_size=16, num_layers=1, num_attention_heads=2,
        num_key_value_heads=1, intermediate_size=24, max_seq_len=16,
    )


def annotation(*, position=1, class_index=0, kind="transition", distance=None):
    value = {
        "family": "object_location", "class_index": class_index,
        "hidden_position": position, "kind": kind,
        "metadata": {"distance_bucket": "short", "difficulty": 1, "genre": "mystery"},
    }
    if distance is not None:
        value["distance"] = distance
    return value


def example(records, *, token_ids=None, name="example"):
    return {
        "example_id": name, "pair": "pair", "world": "A", "entity_slot": "object_0",
        "token_ids": token_ids or [1, 2, 3, 4], "annotations": records,
    }


class StateLossTests(unittest.TestCase):
    def test_exact_hidden_index_and_per_example_normalization(self):
        heads = build_state_heads(2)
        with torch.no_grad():
            for head in heads.values():
                head.weight.zero_()
                head.bias.zero_()
            heads["object_location"].weight.copy_(torch.tensor([[2.0, 0.0], [-2.0, 0.0]]))
        hidden = torch.zeros(2, 4, 2)
        hidden[0, 1, 0] = 1.0
        hidden[0, 2, 0] = -1.0
        hidden[1, 3, 0] = 0.5
        examples = [
            example([annotation(position=1, class_index=0), annotation(position=2, class_index=1)]),
            example([annotation(position=3, class_index=1)], name="second"),
        ]
        loss, results = normalized_state_loss(hidden, examples, heads)
        logits_a = torch.tensor([[2.0, -2.0]])
        logits_b = torch.tensor([[-2.0, 2.0]])
        logits_c = torch.tensor([[1.0, -1.0]])
        expected = 0.5 * (
            0.5 * (F.cross_entropy(logits_a, torch.tensor([0])) + F.cross_entropy(logits_b, torch.tensor([1])))
            + F.cross_entropy(logits_c, torch.tensor([1]))
        )
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(len(results), 3)
        # Three annotations in the first example would produce a different result;
        # annotation-rich examples receive no extra example-level weight.
        repeated = copy.deepcopy(examples)
        repeated[0]["annotations"].append(copy.deepcopy(repeated[0]["annotations"][0]))
        repeated_loss, _ = normalized_state_loss(hidden, repeated, heads)
        expected_repeated = 0.5 * (
            (2 * F.cross_entropy(logits_a, torch.tensor([0])) + F.cross_entropy(logits_b, torch.tensor([1]))) / 3
            + F.cross_entropy(logits_c, torch.tensor([1]))
        )
        self.assertTrue(torch.allclose(repeated_loss, expected_repeated))

    def test_state_loss_reaches_heads_and_base_lm(self):
        torch.manual_seed(7)
        model = FictionPulperLM(tiny_config())
        heads = build_state_heads(16)
        examples = [example([
            annotation(position=2, class_index=1),
            annotation(position=3, class_index=0, kind="carry", distance=32),
        ])]
        result = preflight_numerical_sanity(model, heads, examples, torch.device("cpu"))
        self.assertGreater(result["head_gradient_norm_sum"], 0)
        self.assertGreater(result["base_gradient_norm_sum"], 0)

    def test_state_forward_has_no_lm_labels(self):
        class RecordingLM(FictionPulperLM):
            def __init__(self):
                super().__init__(tiny_config())
                self.seen_labels = "unset"

            def forward(self, input_ids, labels=None, loss_mask=None, return_hidden_states=False):
                self.seen_labels = labels
                return super().forward(input_ids, labels, loss_mask, return_hidden_states)

        model = RecordingLM()
        loss, _ = state_loss_for_examples(
            model, build_state_heads(16), [example([annotation(position=2)])], torch.device("cpu")
        )
        self.assertIsNone(model.seen_labels)
        self.assertTrue(torch.isfinite(loss))


class InputAndParameterTests(unittest.TestCase):
    def test_prefix_position_consumes_event_and_labels_are_absent(self):
        class Tokenizer:
            ids = {"<|story|>": 1, "<|mystery|>": 2, "<|bos|>": 3}

            def token_to_id(self, value):
                return self.ids.get(value)

            def encode(self, value):
                return type("Encoding", (), {"ids": [30]})()

        pair = {
            "pair_id": "p", "metadata": {"genre_control_token": "<|mystery|>"},
            "worlds": {"A": {"context_token_ids": [7, 8, 9]}, "B": {"context_token_ids": [10]}},
        }
        records = {"p": [{
            "pair": "p", "world": "A", "entity_slot": "object_0", "family": "object_location",
            "post_state": "location_1", "prediction_token_index": 2, "kind": "transition",
            "metadata": {"distance_bucket": "short", "difficulty": 1, "genre": "mystery"},
        }]}
        examples = state_examples_for_pairs([pair], records, Tokenizer())
        self.assertEqual(examples[0]["token_ids"], [1, 2, 3, 7, 8, 9])
        self.assertEqual(examples[0]["annotations"][0]["hidden_position"], 4)
        self.assertNotIn(30, examples[0]["token_ids"])

    def test_base_and_auxiliary_parameters_are_separate(self):
        config = model_config_from_dict(yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))["model"])
        model = FictionPulperLM(config)
        heads = build_state_heads(config.hidden_size)
        counts = parameter_counts(model, heads)
        self.assertEqual(counts["base"], LOCKED_PARAMETER_COUNT)
        self.assertEqual(counts["auxiliary"], sum((config.hidden_size + 1) * len(classes) for classes in CANONICAL_CLASS_MAPS.values()))
        self.assertEqual(set(heads), set(CANONICAL_CLASS_MAPS))
        self.assertTrue(all(isinstance(head, torch.nn.Linear) for head in heads.values()))

    def test_fresh_initialization_is_deterministic(self):
        first_model, first_heads = create_fresh_model_and_heads(
            model_config_from_dict(yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))["model"])
        )
        first_model_value = first_model.embed_tokens.weight[0, :8].detach().clone()
        first_head_value = first_heads["object_location"].weight[0, :8].detach().clone()
        del first_model, first_heads
        second_model, second_heads = create_fresh_model_and_heads(
            model_config_from_dict(yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))["model"])
        )
        self.assertTrue(torch.equal(first_model_value, second_model.embed_tokens.weight[0, :8]))
        self.assertTrue(torch.equal(first_head_value, second_heads["object_location"].weight[0, :8]))


class CheckpointTests(unittest.TestCase):
    def test_auxiliary_reload_and_lm_only_export_equivalence(self):
        torch.manual_seed(4)
        model = FictionPulperLM(tiny_config())
        heads = build_state_heads(16)
        optimizer = torch.optim.AdamW([*model.parameters(), *heads.parameters()])
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            export = Path(temporary) / "lm.pt"
            save_checkpoint(
                checkpoint, model=model, state_heads=heads, optimizer=optimizer,
                config={"fixture": True}, step=3, locks={"tokenizer_sha256": "fixture"},
                metrics=[{"step": 3}], expected_base_parameter_count=model.trainable_parameter_count(),
            )
            reloaded_model = FictionPulperLM(tiny_config())
            reloaded_heads = build_state_heads(16)
            payload = load_checkpoint_for_validation(
                checkpoint, model=reloaded_model, state_heads=reloaded_heads,
                expected_locks={"tokenizer_sha256": "fixture"},
            )
            self.assertEqual(payload["head_config"], head_config())
            for name, value in heads.state_dict().items():
                self.assertTrue(torch.equal(value, reloaded_heads.state_dict()[name]))
            exported = export_lm_only(checkpoint, export)
            self.assertNotIn("state_heads", exported)
            result = check_lm_only_equivalence(model, export, torch.tensor([[1, 2, 3]]), generation_steps=3)
            self.assertTrue(result["logits_identical"])
            self.assertTrue(result["generation_identical"])


class LockTests(unittest.TestCase):
    def state_config(self):
        config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
        config["experiment_id"] = "fictionpulper-15m-data30m-state-transition-v1"
        config["training"]["checkpoint_dir"] = "checkpoints/fictionpulper-15m-data30m-state-transition-v1"
        config["logging"]["run_dir"] = "runs/fictionpulper-15m-data30m-state-transition-v1"
        config["state_transition_data"] = {
            "manifest_path": "data/state_transition_v1/manifest.json",
            "train_path": "data/state_transition_v1/train.jsonl",
            "expected_manifest_sha256": "PENDING_MANIFEST",
            "expected_train_sha256": "PENDING_TRAIN",
        }
        config["state_transition_objective"] = {
            "version": "State-Transition-v1", "pair_lambda": 0.25, "state_lambda": 0.25,
            "head_type": "linear", "representation": REPRESENTATION,
            "normalization": "mean_per_world_entity_then_mean_examples", "labels_in_input": False,
        }
        return config

    def test_locked_config_accepts_protocol_and_rejects_drift(self):
        config = self.state_config()
        validate_locked_config(config, require_generated_locks=False)
        mutations = [
            ("training", "max_steps", 3),
            ("training", "precision", "fp32"),
            ("state_transition_objective", "state_lambda", 1.0),
            ("state_transition_objective", "representation", "pre_norm"),
            ("state_transition_data", "train_path", "data/test.jsonl"),
        ]
        for section, key, value in mutations:
            changed = copy.deepcopy(config)
            changed[section][key] = value
            with self.subTest(section=section, key=key):
                with self.assertRaises(RuntimeError):
                    validate_locked_config(changed, require_generated_locks=False)
        changed = copy.deepcopy(config)
        changed["selection"]["metric"] = "state_accuracy"
        with self.assertRaisesRegex(RuntimeError, "selection"):
            validate_locked_config(changed, require_generated_locks=False)


if __name__ == "__main__":
    unittest.main()
