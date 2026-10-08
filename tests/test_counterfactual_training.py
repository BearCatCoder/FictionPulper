import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src.model import FictionPulperLM, model_config_from_dict
from src.mixed_schedule import verify_schedule
from src.pack_counterfactual_v1 import (
    PACKED_SPLITS,
    _world_tokens,
    attach_pair_presentations,
    verify_counterfactual_gates,
)
from src.train_contrastive import mean_span_log_probability
from src.train_counterfactual import (
    LOCKED_CURRICULUM_TARGETS,
    LOCKED_LAMBDA,
    LOCKED_REAL_TARGETS,
    attached_pairs,
    symmetric_pair_loss,
    validate_locked_config,
)
from src.train_mixed import LOCKED_PARAMETER_COUNT
from src.train_tokenizer import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "data30m-15m-counterfactual-v1.yaml"


def ranking_pair():
    return {
        "pair_id": "pair-1",
        "worlds": {
            "A": {
                "correct_candidate": "X",
                "candidate_encodings": {
                    "X": {"token_ids": [1, 3, 0], "decision_start": 1, "decision_end": 2},
                    "Y": {"token_ids": [1, 4, 0], "decision_start": 1, "decision_end": 2},
                },
            },
            "B": {
                "correct_candidate": "Y",
                "candidate_encodings": {
                    "X": {"token_ids": [2, 3, 0], "decision_start": 1, "decision_end": 2},
                    "Y": {"token_ids": [2, 4, 0], "decision_start": 1, "decision_end": 2},
                },
            },
        },
        "abstract_counterfactual_variable": {"state_family": "object_location"},
        "metadata": {"distance_bucket": "d128", "difficulty": 2},
    }


class ScoreModel:
    def __init__(self):
        self.labels = []

    def __call__(self, tokens, labels=None):
        self.labels.append(labels)
        logits = torch.zeros(1, tokens.shape[1], 6)
        context = int(tokens[0, 0])
        if context == 1:
            logits[0, 0, 3], logits[0, 0, 4] = 2.0, 0.0
        else:
            logits[0, 0, 3], logits[0, 0, 4] = 0.0, 3.0
        return logits, None


class CounterfactualObjectiveTests(unittest.TestCase):
    def test_symmetric_oriented_loss_and_ranking_labels_none(self):
        model = ScoreModel()
        loss, metrics = symmetric_pair_loss(model, ranking_pair(), torch.device("cpu"))
        expected = 0.5 * (F.softplus(torch.tensor(-2.0)) + F.softplus(torch.tensor(-3.0)))
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(model.labels, [None, None, None, None])
        self.assertEqual(metrics["world_A_accuracy"], 1.0)
        self.assertEqual(metrics["world_B_accuracy"], 1.0)
        self.assertEqual(metrics["paired_reversal_success"], 1.0)

    def test_candidate_score_is_length_normalized(self):
        logits = torch.zeros(1, 4, 5)
        tokens = torch.tensor([0, 2, 2, 2, 0])
        one = mean_span_log_probability(logits, tokens, 1, 2)
        three = mean_span_log_probability(logits, tokens, 1, 4)
        self.assertTrue(torch.allclose(one, three))

    def test_positive_packing_selects_only_each_world_correct_candidate(self):
        class Tokenizer:
            def encode(self, text):
                values = {
                    "context-a candidate-x": [7, 3], "context-b candidate-y": [8, 4],
                }
                return type("Encoding", (), {"ids": values[text]})()

        pair = {
            "pair_id": "pair", "metadata": {"genre_control_token": None},
            "candidates": {
                "X": {"text": " candidate-x", "token_ids": [3]},
                "Y": {"text": " candidate-y", "token_ids": [4]},
            },
            "worlds": {
                "A": {"context": "context-a", "correct_candidate": "X", "candidate_encodings": {
                    "X": {"token_ids": [1, 2, 7, 3, 9], "decision_start": 3, "decision_end": 4},
                    "Y": {"token_ids": [1, 2, 7, 4, 9], "decision_start": 3, "decision_end": 4},
                }},
                "B": {"context": "context-b", "correct_candidate": "Y", "candidate_encodings": {
                    "X": {"token_ids": [1, 2, 8, 3, 9], "decision_start": 3, "decision_end": 4},
                    "Y": {"token_ids": [1, 2, 8, 4, 9], "decision_start": 3, "decision_end": 4},
                }},
            },
            "abstract_counterfactual_variable": {"state_family": "object_location"},
        }
        pair["metadata"].update({"distance_bucket": "d64", "difficulty": 1, "template_family": "f"})
        a, a_meta = _world_tokens(pair, "A", Tokenizer())
        b, b_meta = _world_tokens(pair, "B", Tokenizer())
        self.assertEqual(a, pair["worlds"]["A"]["candidate_encodings"]["X"]["token_ids"])
        self.assertEqual(b, pair["worlds"]["B"]["candidate_encodings"]["Y"]["token_ids"])
        self.assertEqual((a_meta["correct_candidate"], b_meta["correct_candidate"]), ("X", "Y"))


class PairAttachmentTests(unittest.TestCase):
    def schedule(self):
        return {
            "batch_size": 2,
            "entries": [
                {"source": "curriculum", "document_id": "counterfactual-v1-train-stream", "relative_start": 0, "valid_targets": 8},
                {"source": "curriculum", "document_id": "counterfactual-v1-train-stream", "relative_start": 6, "valid_targets": 8},
                {"source": "real", "document_id": "real", "relative_start": 0, "valid_targets": 8},
            ],
        }

    def test_attachment_is_deterministic_and_deduplicated_within_microbatch(self):
        index = {"world_boundaries": [
            {"pair_id": "p", "world": "A", "candidate_absolute_token_start": 4, "candidate_absolute_token_end": 7,
             "state_family": "latest_state_update", "distance_bucket": "d128", "difficulty": 3},
            {"pair_id": "p", "world": "B", "candidate_absolute_token_start": 9, "candidate_absolute_token_end": 12,
             "state_family": "latest_state_update", "distance_bucket": "d128", "difficulty": 3},
        ]}
        first, second = self.schedule(), self.schedule()
        self.assertEqual(attach_pair_presentations(first, source_name="curriculum", packed_index=index),
                         attach_pair_presentations(second, source_name="curriculum", packed_index=index))
        self.assertEqual(first, second)
        presentations = [item for entry in first["entries"][:2] for item in entry["pair_presentations"]]
        self.assertEqual(len(presentations), 1)
        self.assertEqual(presentations[0]["trigger_worlds"], ["A", "B"])
        self.assertEqual(attached_pairs(first["entries"][:2], {"p": ranking_pair()})[0]["pair_id"], "pair-1")


class CounterfactualLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_exact_model_compute_mixer_and_selection_locks(self):
        validate_locked_config(self.config)
        model = FictionPulperLM(model_config_from_dict(self.config["model"]))
        self.assertEqual(model.trainable_parameter_count(), LOCKED_PARAMETER_COUNT)
        self.assertEqual(LOCKED_REAL_TARGETS + LOCKED_CURRICULUM_TARGETS, 135_886_355)
        self.assertEqual(self.config["mixed_data"]["schedule"]["total_chunks"], 142_080)
        self.assertEqual(self.config["training"]["batch_size"] * self.config["training"]["gradient_accumulation_steps"] * self.config["training"]["max_steps"], 142_080)
        self.assertEqual(self.config["counterfactual_objective"]["lambda"], LOCKED_LAMBDA)
        self.assertEqual(self.config["selection"]["metric"], "data30m_validation_loss")
        self.assertEqual(PACKED_SPLITS, ("train", "validation"))

    def test_manifest_and_pretraining_audit_hash_gates(self):
        verified = verify_counterfactual_gates(self.config)
        self.assertEqual(verified["manifest_sha256"], self.config["counterfactual_data"]["expected_manifest_sha256"])
        changed = copy.deepcopy(self.config)
        changed["counterfactual_data"]["expected_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "manifest"):
            verify_counterfactual_gates(changed)

    def test_generated_packed_and_schedule_hash_locks(self):
        packing = self.config["counterfactual_packing"]
        self.assertEqual(sha256_file(Path(packing["metadata_path"])), packing["expected_metadata_sha256"])
        for split in ("train", "validation"):
            path = Path(packing["output_dir"]) / f"{split}.bin"
            self.assertEqual(sha256_file(path), packing[f"expected_{split}_sha256"])
            self.assertEqual(sha256_file(path.with_suffix(".index.json")), packing[f"expected_{split}_index_sha256"])
        schedule_config = self.config["mixed_data"]["schedule"]
        schedule_path = Path(schedule_config["path"])
        self.assertEqual(sha256_file(schedule_path), schedule_config["expected_file_sha256"])
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        self.assertEqual(verify_schedule(schedule), schedule_config["expected_content_sha256"])
        attachment = schedule["pair_attachment"]
        objective = self.config["counterfactual_objective"]
        self.assertEqual(attachment["pair_presentations"], objective["expected_pair_presentations"])
        self.assertEqual(attachment["candidate_span_world_triggers"], objective["expected_candidate_span_world_triggers"])
        self.assertEqual(attachment["unique_pairs"], objective["expected_unique_pairs"])

    def test_selection_or_architecture_drift_is_rejected(self):
        changed = copy.deepcopy(self.config)
        changed["selection"]["metric"] = "counterfactual_validation"
        with self.assertRaisesRegex(RuntimeError, "selection"):
            validate_locked_config(changed, require_generated_locks=False)
        changed = copy.deepcopy(self.config)
        changed["model"]["hidden_size"] = 512
        with self.assertRaisesRegex(RuntimeError, "model"):
            validate_locked_config(changed, require_generated_locks=False)


if __name__ == "__main__":
    unittest.main()
