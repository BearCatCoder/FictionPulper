import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer

from src.continuity_curriculum.generator import generate_example
from src.contrastive_pairs import GENERATOR_VERSION, PairValidationError, _canonical_hash, build_pair, build_post_selection_pairs, load_pairs, validate_pair
from src.model import FictionPulperLM, model_config_from_dict
from src.train_contrastive import (
    LOCKED_LAMBDA,
    ScheduledPairIndex,
    combine_training_losses,
    mean_span_log_probability,
    pair_ranking_loss,
    validate_locked_config,
)
from src.train_mixed import LOCKED_PARAMETER_COUNT, LOCKED_TOTAL_VALID_TARGETS
from src.train_tokenizer import sha256_file


def pair(start=10, length=5, pair_id="pair", state="persistent_injury", packed_start=0, packed_id="stream"):
    value = {
        "pair_id": pair_id,
        "document_id": pair_id,
        "generator_version": GENERATOR_VERSION,
        "state_type": state,
        "template_family": "family",
        "positive_text": "He remained injured.",
        "negative_texts": ["He healed completely."],
        "prefix_text": "Before. ",
        "character_offsets": {"decision_start": 8, "positive_decision_end": 28},
        "token_offsets": {"decision_start": 2, "positive_decision_end": 4, "negative_decision_ends": [4]},
        "packed_offsets": {"packed_document_id": packed_id, "packed_document_start": packed_start, "document_start": start, "document_length": length, "decision_start": start + 2, "positive_decision_end": start + 4},
        "lengths": {"prefix_tokens": 2, "positive_tokens": 4, "negative_tokens": [4]},
        "expected_state_value": "unhealed",
        "violating_state_values": ["healed"],
        "positive_token_ids": [1, 2, 3, 4],
        "negative_token_ids": [[1, 2, 5, 6]],
    }
    value["validation_hash"] = _canonical_hash(value)
    return value


class ContrastivePairTests(unittest.TestCase):
    def test_pair_validation_and_hash_are_deterministic(self):
        first = pair()
        second = pair()
        self.assertEqual(first, second)
        validate_pair(first, max_seq_len=8)
        self.assertEqual(len(first["validation_hash"]), 64)

    def test_pair_validation_rejects_identical_or_nonviolating_negative(self):
        value = pair()
        value["negative_token_ids"] = [value["positive_token_ids"]]
        value["validation_hash"] = _canonical_hash({key: item for key, item in value.items() if key != "validation_hash"})
        with self.assertRaisesRegex(PairValidationError, "identical"):
            validate_pair(value, max_seq_len=8)

    def test_pair_validation_rejects_bad_offsets(self):
        value = pair()
        value["packed_offsets"]["decision_start"] += 1
        value["validation_hash"] = _canonical_hash({key: item for key, item in value.items() if key != "validation_hash"})
        with self.assertRaisesRegex(PairValidationError, "Packed offsets"):
            validate_pair(value, max_seq_len=8)

    def test_pair_artifact_loader_accepts_pair_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            path.write_text(json.dumps(pair()) + "\n", encoding="utf-8")
            self.assertEqual(load_pairs(path, max_seq_len=8)[0]["pair_id"], "pair")

    def test_exact_template_pair_construction_is_deterministic(self):
        config = yaml.safe_load(Path("configs/continuity-curriculum-v1.yaml").read_text())
        tokenizer = Tokenizer.from_file(config["tokenizer"]["path"])
        record, sidecar = generate_example(config, tokenizer, split="train", ordinal=0)
        boundary = {"id": record["id"], "offset": 123, "length": record["token_count"]}
        first = build_pair(record, sidecar, tokenizer, boundary)
        second = build_pair(record, sidecar, tokenizer, boundary)
        self.assertEqual(first, second)
        self.assertNotEqual(first["expected_state_value"], first["violating_state_values"][0])
        self.assertNotEqual(first["positive_text"], first["negative_texts"][0])

    def test_post_selection_builder_requires_and_records_supplied_hashes(self):
        config = yaml.safe_load(Path("configs/continuity-curriculum-v1.yaml").read_text())
        tokenizer = Tokenizer.from_file(config["tokenizer"]["path"])
        record, sidecar = generate_example(config, tokenizer, split="test", ordinal=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "test.jsonl"
            sidecars = root / "test.state.jsonl"
            records.write_text(json.dumps(record) + "\n", encoding="utf-8")
            sidecars.write_text(json.dumps(sidecar) + "\n", encoding="utf-8")
            manifest = build_post_selection_pairs(
                split="test",
                records_path=records,
                records_sha256=sha256_file(records),
                sidecars_path=sidecars,
                sidecars_sha256=sha256_file(sidecars),
                tokenizer_path=Path(config["tokenizer"]["path"]),
                tokenizer_sha256=config["tokenizer"]["expected_sha256"],
                output_dir=root / "pairs",
            )
            self.assertEqual(manifest["split"], "test")
            self.assertEqual(manifest["pair_count"], 1)
            with self.assertRaisesRegex(RuntimeError, "records hash"):
                build_post_selection_pairs(
                    split="test",
                    records_path=records,
                    records_sha256="0" * 64,
                    sidecars_path=sidecars,
                    sidecars_sha256=sha256_file(sidecars),
                    tokenizer_path=Path(config["tokenizer"]["path"]),
                    tokenizer_sha256=config["tokenizer"]["expected_sha256"],
                    output_dir=root / "rejected",
                )

    def test_schedule_resolution_preserves_entries_and_selects_only_curriculum(self):
        pairs = ScheduledPairIndex([
            pair(1_000_100, 5, packed_start=1_000_000),
            pair(1_000_300, 5, "other", packed_start=1_000_000),
        ])
        entries = [
            {"source": "real", "document_id": "stream", "relative_start": 100, "valid_targets": 8},
            {"source": "curriculum", "document_id": "wrong-stream", "relative_start": 98, "valid_targets": 8},
            {"source": "curriculum", "document_id": "stream", "relative_start": 98, "valid_targets": 8},
            {"source": "curriculum", "document_id": "stream", "relative_start": 100, "valid_targets": 4},
        ]
        original = copy.deepcopy(entries)
        self.assertEqual([item["pair_id"] for item in pairs.for_entries(entries)], ["pair"])
        self.assertEqual(entries, original)


class ContrastiveLossTests(unittest.TestCase):
    def test_mean_score_uses_only_decision_span(self):
        logits = torch.zeros(1, 4, 3)
        logits[0, 1, 1] = 2.0
        logits[0, 2, 2] = 4.0
        tokens = torch.tensor([0, 0, 1, 2, 0])
        observed = mean_span_log_probability(logits, tokens, 2, 4)
        expected = torch.stack((F.log_softmax(logits[0, 1], 0)[1], F.log_softmax(logits[0, 2], 0)[2])).mean()
        self.assertTrue(torch.allclose(observed, expected))

    def test_real_batch_is_ce_only_and_negative_never_enters_ce(self):
        ce = torch.tensor(2.0, requires_grad=True)
        self.assertIs(combine_training_losses(ce, [], LOCKED_LAMBDA), ce)
        ranking = torch.tensor(0.8, requires_grad=True)
        total = combine_training_losses(ce, [ranking], LOCKED_LAMBDA)
        self.assertAlmostEqual(float(total.detach()), 2.2)
        total.backward()
        self.assertEqual(float(ce.grad), 1.0)
        self.assertAlmostEqual(float(ranking.grad), LOCKED_LAMBDA)

    def test_ranking_forwards_never_receive_ce_labels(self):
        class RecordingModel:
            def __init__(self):
                self.labels = []

            def __call__(self, input_ids, labels=None):
                self.labels.append(labels)
                return torch.zeros(1, input_ids.shape[1], 7), None

        model = RecordingModel()
        _, result = pair_ranking_loss(model, pair(), torch.device("cpu"))
        self.assertEqual(model.labels, [None, None])
        self.assertEqual(result["negative_decision_span_tokens"], 2)
        self.assertEqual(result["ranking_forward_token_positions"], 6)


class ContrastiveLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(Path("configs/data30m-15m-contrastive-v1.yaml").read_text())

    def test_locked_config_architecture_exposure_and_selection(self):
        validate_locked_config(self.config)
        model = FictionPulperLM(model_config_from_dict(self.config["model"]))
        self.assertEqual(model.trainable_parameter_count(), LOCKED_PARAMETER_COUNT)
        self.assertEqual(sum(item["valid_targets"] for item in self.config["mixed_data"]["training_sources"].values()), LOCKED_TOTAL_VALID_TARGETS)
        self.assertEqual(self.config["training"]["max_steps"], 2220)
        self.assertEqual(self.config["contrastive"]["lambda"], 0.25)
        self.assertEqual(self.config["contrastive"]["expected_pair_presentations"], 65036)
        self.assertEqual(self.config["contrastive"]["expected_negative_decision_span_tokens"], 1639692)
        self.assertEqual(self.config["contrastive"]["expected_ranking_forward_token_positions"], 57543213)
        self.assertNotIn("test_path", self.config["data"])
        self.assertNotIn("curriculum_validation_path", self.config["mixed_data"])
        self.assertEqual(self.config["contrastive"]["pairs"]["output_dir"], "data/contrastive_narrative_v1")
        self.assertEqual(self.config["logging"]["run_dir"], "runs/fictionpulper-15m-data30m-contrastive-v1")

    def test_generated_train_pair_artifacts_match_locked_config(self):
        pair_config = self.config["contrastive"]["pairs"]
        self.assertEqual(sha256_file(Path(pair_config["path"])), pair_config["expected_sha256"])
        self.assertEqual(sha256_file(Path(pair_config["manifest_path"])), pair_config["expected_manifest_sha256"])
        with Path(pair_config["path"]).open("r", encoding="utf-8") as source:
            first = json.loads(next(source))
        self.assertEqual(first["packed_offsets"]["packed_document_id"], "narrative-curriculum-v1-train-stream")

    def test_locked_config_rejects_lambda_or_selection_drift(self):
        changed = copy.deepcopy(self.config)
        changed["contrastive"]["lambda"] = 0.5
        with self.assertRaisesRegex(RuntimeError, "Contrastive"):
            validate_locked_config(changed)
        changed = copy.deepcopy(self.config)
        changed["selection"]["metric"] = "curriculum_validation_loss"
        with self.assertRaisesRegex(RuntimeError, "Data30M"):
            validate_locked_config(changed)


if __name__ == "__main__":
    unittest.main()
