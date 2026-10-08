import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from src.frozen_state_probe import (
    SUPPORTED_LABEL_FAMILIES,
    aggregate_best_layer,
    classification_metrics,
    extract_states,
    freeze_model,
    labels_for_example,
    layer_names,
    resolve_source,
    train_probe,
    validate_family_coverage,
)
from src.model import FictionPulperLM, ModelConfig


def tiny_model() -> FictionPulperLM:
    return FictionPulperLM(ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=24,
        max_seq_len=16,
    ))


class FrozenStateProbeTests(unittest.TestCase):
    def test_hidden_state_api_preserves_default_and_captures_all_stages(self):
        torch.manual_seed(3)
        model = tiny_model().eval()
        tokens = torch.randint(0, 32, (2, 7))
        default = model(tokens)
        diagnostic = model(tokens, return_hidden_states=True)
        self.assertEqual(len(default), 2)
        self.assertEqual(len(diagnostic), 3)
        self.assertEqual(len(diagnostic[2]), 4)
        self.assertEqual(layer_names(2), ["embedding", "layer_01", "layer_02", "final_norm"])
        torch.testing.assert_close(default[0], diagnostic[0], rtol=0, atol=0)
        torch.testing.assert_close(diagnostic[2][-1], model.final_norm(diagnostic[2][-2]))

    def test_extraction_freezes_transformer_and_never_creates_gradients(self):
        model = tiny_model()
        freeze_model(model)
        examples = [
            {"input_ids": [1, 2, 3], "position": 2},
            {"input_ids": [4, 5, 6, 7, 8], "position": 4},
        ]
        states = extract_states(model, examples, batch_size=2, pad_id=0, device=torch.device("cpu"))
        self.assertEqual(set(states), set(layer_names(2)))
        self.assertTrue(all(value.shape == (2, 16) for value in states.values()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def _sidecar(self, state_type, events, *, names=None, locations=None):
        return {
            "id": "x", "split": "test",
            "primary_distance": {
                "state_type": state_type,
                "resolved_character_start": 100,
            },
            "generation_values": {
                "names": names or ["Zed", "Alice", "Mira"],
                "objects": ["rare key", "blue ledger"],
                "locations": locations or ["west pier", "north vault"],
            },
            "events": [
                {
                    "state_type": state_type,
                    "character_start": 10 + index,
                    "character_end": 30 + index,
                    **event,
                }
                for index, event in enumerate(events)
            ],
        }

    def test_state_labels_use_varying_lexical_rank_slots_not_generator_order(self):
        record = {"id": "x", "split": "test"}
        ownership_events = [
            {"entity": "rare key", "attribute": "holder", "value": "Zed"},
            {"entity": "rare key", "attribute": "hidden_at", "value": "west pier"},
            {"entity": "rare key", "attribute": "retrieved", "value": False},
        ]
        first = labels_for_example(record, self._sidecar(
            "ownership_transfer_hiding_retrieval", ownership_events,
            names=["Zed", "Alice", "Mira"], locations=["west pier", "north vault"],
        ))
        reordered = labels_for_example(record, self._sidecar(
            "ownership_transfer_hiding_retrieval", ownership_events,
            names=["Mira", "Zed", "Alice"], locations=["north vault", "west pier"],
        ))
        self.assertEqual(first, reordered)
        self.assertEqual(first, {
            "object_ownership": "character_2",
            "object_location": "location_1",
        })

        changed_holder = ownership_events.copy()
        changed_holder[0] = {"entity": "rare key", "attribute": "holder", "value": "Alice"}
        changed = labels_for_example(record, self._sidecar(
            "ownership_transfer_hiding_retrieval", changed_holder,
        ))
        self.assertEqual(changed["object_ownership"], "character_0")
        self.assertNotIn("Alice", repr(changed))

    def test_labels_cover_actual_semantic_state_families(self):
        record = {"id": "x", "split": "test"}
        cases = {
            "knowledge_holder": self._sidecar("knowledge_ignorance_secret", [
                {"entity": "Zed", "attribute": "knows:secret", "value": False},
                {"entity": "Mira", "attribute": "knows:secret", "value": True},
            ]),
            "relationship_state": self._sidecar("character_identity_role_relationship", [
                {"entity": "Zed", "attribute": "role", "value": "warden"},
                {"entity": "Mira", "attribute": "relationship_to:Zed", "value": "cousin"},
            ]),
            "injury_state": self._sidecar("persistent_injury", [
                {"entity": "Alice", "attribute": "injury", "value": "ankle"},
                {"entity": "Alice", "attribute": "injury_status", "value": "unhealed"},
            ]),
            "temporal_order": self._sidecar("temporal_ordering", [
                {"entity": "timeline", "attribute": "first", "value": "Mira rang bell"},
                {"entity": "timeline", "attribute": "second", "value": "Zed opened gate"},
            ]),
            "obligation_state": self._sidecar("promise_debt_obligation", [
                {"entity": "Alice", "attribute": "obligation_to:Zed", "value": "outstanding"},
            ]),
            "character_location": self._sidecar("physical_scene_location_movement", [
                {"entity": "Alice", "attribute": "location", "value": "west pier"},
                {"entity": "Mira", "attribute": "location", "value": "west pier"},
            ]),
        }
        for family, sidecar in cases.items():
            with self.subTest(family=family):
                labels = labels_for_example(record, sidecar)
                self.assertEqual(set(labels), {family})
                self.assertNotIn("Zed", repr(labels))
                self.assertNotIn("Mira", repr(labels))

    def test_coverage_requires_two_train_classes_and_known_held_out_labels(self):
        examples = {
            split: [{"labels": {"injury_state": "character_0"}}]
            for split in ("train", "validation", "test", "generalization_holdout")
        }
        with self.assertRaisesRegex(ValueError, "at least two train classes"):
            validate_family_coverage(examples, ["injury_state"])
        examples["train"].append({"labels": {"injury_state": "character_1"}})
        examples["test"][0]["labels"]["injury_state"] = "character_2"
        with self.assertRaisesRegex(ValueError, "labels absent from train"):
            validate_family_coverage(examples, ["injury_state"])

    def test_metrics_and_linear_probe_report_all_splits_and_distances(self):
        metrics = classification_metrics(
            np.array([0, 1, 1, 1]), np.array([0, 0, 1, 1]),
            num_classes=2, train_targets=np.array([0, 0, 0, 1]),
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.75)
        self.assertEqual(metrics["chance"], 0.5)
        self.assertEqual(metrics["support"], {"0": 2, "1": 2})

        rng = np.random.default_rng(5)
        examples = {}
        features = {}
        sizes = {"train": 20, "validation": 8, "test": 8, "generalization_holdout": 8}
        for split, size in sizes.items():
            labels = np.arange(size) % 2
            features[split] = np.column_stack((labels * 4 - 2, rng.normal(size=size))).astype(np.float32)
            examples[split] = [{
                "labels": {"object_ownership": f"character_{label}"},
                "distance_bucket": "d064" if index % 2 else "d128",
            } for index, label in enumerate(labels)]
        result = train_probe(
            features, examples, "object_ownership",
            {"epochs": 8, "batch_size": 8, "learning_rate": 0.05, "weight_decay": 0.0},
            1001,
        )
        self.assertEqual(result["selection_split"], "validation")
        self.assertEqual(set(result["evaluations"]), set(sizes))
        self.assertIn("d064", result["evaluations"]["test"]["distance_buckets"])
        self.assertGreaterEqual(result["evaluations"]["test"]["overall"]["accuracy"], 0.75)

        selection_only = train_probe(
            features, examples, "object_ownership",
            {"epochs": 2, "batch_size": 8, "learning_rate": 0.05, "weight_decay": 0.0},
            1001, report_held_out=False,
        )
        self.assertEqual(set(selection_only["evaluations"]), {"train", "validation"})

        summary = aggregate_best_layer([result, result, result])
        self.assertEqual(summary["test"]["metrics_over_seeds"]["accuracy"]["std"], 0.0)
        self.assertIn(summary["test"]["interpretation"], {
            "strong_linear_state_signal",
            "weak_or_imbalanced_linear_state_signal",
            "no_clear_linear_state_signal",
        })

    def test_protocol_has_shared_seeds_and_future_checkpoint_slot(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (root / "experiments/frozen-state-probe-v1/config.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["probe"]["seeds"], [1001, 1002, 1003])
        self.assertEqual({item["id"] for item in config["checkpoints"]}, {
            "compute5", "narrative_v1", "contrastive_future",
        })
        self.assertFalse(config["checkpoints"][-1]["enabled"])
        self.assertEqual(config["checkpoints"][0]["source_tag"], "fictionpulper-15m-data30m-compute5")
        source = resolve_source(config["checkpoints"][0])
        self.assertTrue(source["tag_matches_expected_commit"])
        self.assertEqual(source["resolved_commit"], config["checkpoints"][0]["source_commit"])
        self.assertEqual(
            set(SUPPORTED_LABEL_FAMILIES),
            set(config["labels"]["requested_families"][: len(SUPPORTED_LABEL_FAMILIES)]),
        )


if __name__ == "__main__":
    unittest.main()
