import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from src.dataset import PackedStoryDataset
from src.mixed_schedule import (
    ScheduledMixedDataset,
    build_mixed_schedule,
    persist_schedule,
    verify_schedule,
)
from src.model import FictionPulperLM, ModelConfig, model_config_from_dict
from src.pack_mixed_curriculum import PACKED_SPLITS, _pack_split, canonical_document_text
from src.train_mixed import (
    LOCKED_CHECKPOINT_STEPS,
    LOCKED_PARAMETER_COUNT,
    LOCKED_TOTAL_VALID_TARGETS,
    _save_checkpoint,
    _validate_schedule_contract,
    tracked_worktree_dirty,
    validate_locked_config,
    validate_runtime_environment,
)


def write_packed(root: Path, name: str, lengths: list[int]) -> PackedStoryDataset:
    bin_path = root / f"{name}.bin"
    tokens = np.arange(sum(lengths), dtype=np.uint16) % 100
    tokens.tofile(bin_path)
    documents = []
    offset = 0
    for index, length in enumerate(lengths):
        documents.append({"id": f"{name}-{index}", "offset": offset, "length": length})
        offset += length
    bin_path.with_suffix(".index.json").write_text(
        json.dumps({"documents": documents}), encoding="utf-8"
    )
    return PackedStoryDataset(
        bin_path, bin_path.with_suffix(".index.json"), sequence_length=8, pad_token_id=0
    )


class MixedScheduleTests(unittest.TestCase):
    def test_schedule_is_deterministic_exact_and_source_identifiable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {
                "real": write_packed(root, "real", [25, 25]),
                "curriculum": write_packed(root, "curriculum", [25, 25]),
            }
            kwargs = {
                "source_target_counts": {"real": 48, "curriculum": 16},
                "total_chunks": 8,
                "batch_size": 2,
                "gradient_accumulation_steps": 2,
                "max_steps": 2,
                "seed": 1337,
            }
            first = build_mixed_schedule(sources, **kwargs)
            second = build_mixed_schedule(sources, **kwargs)
            self.assertEqual(first, second)
            self.assertEqual(verify_schedule(first), first["schedule_sha256"])
            self.assertEqual(first["total_valid_targets"], 64)
            self.assertEqual(first["sources"]["real"]["valid_targets"], 48)
            self.assertEqual(first["sources"]["curriculum"]["valid_targets"], 16)
            self.assertTrue(
                all("document_id" in entry and "relative_start" in entry for entry in first["entries"])
            )

    def test_schedule_dataset_masks_to_persisted_presentation_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {
                "real": write_packed(root, "real", [25]),
                "curriculum": write_packed(root, "curriculum", [25]),
            }
            schedule = build_mixed_schedule(
                sources,
                source_target_counts={"real": 4, "curriculum": 4},
                total_chunks=2,
                batch_size=1,
                gradient_accumulation_steps=1,
                max_steps=2,
                seed=1337,
            )
            dataset = ScheduledMixedDataset(sources, schedule)
            observed = [int(dataset[index][1].ne(-100).sum()) for index in range(2)]
            expected = [entry["valid_targets"] for entry in schedule["entries"]]
            self.assertEqual(observed, expected)

    def test_schedule_hash_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {
                "real": write_packed(root, "real", [25]),
                "curriculum": write_packed(root, "curriculum", [25]),
            }
            schedule = build_mixed_schedule(
                sources,
                source_target_counts={"real": 4, "curriculum": 4},
                total_chunks=2,
                batch_size=1,
                gradient_accumulation_steps=1,
                max_steps=2,
                seed=1337,
            )
            schedule["entries"][0]["valid_targets"] += 1
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                verify_schedule(schedule)

    def test_schedule_refuses_to_overwrite_different_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.json"
            schedule = {"entries": [], "total_chunks": 0, "sources": {}}
            from src.mixed_schedule import _canonical_hash

            schedule["schedule_sha256"] = _canonical_hash(schedule)
            persist_schedule(path, schedule)
            self.assertEqual(persist_schedule(path, schedule), schedule)
            changed = copy.deepcopy(schedule)
            changed["total_chunks"] = 1
            changed["schedule_sha256"] = _canonical_hash(
                {key: value for key, value in changed.items() if key != "schedule_sha256"}
            )
            with self.assertRaisesRegex(FileExistsError, "different"):
                persist_schedule(path, changed)


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    def __init__(self):
        self.encoded = []

    def token_to_id(self, token):
        return {"<|story|>": 1, "<|bos|>": 2, "<|eos|>": 3}[token]

    def encode(self, text):
        self.encoded.append(text)
        return FakeEncoding([4, 5, 6])


class CurriculumPackingTests(unittest.TestCase):
    def test_complete_text_convention_does_not_duplicate_title(self):
        record = {"id": "story", "title": "The Brass Key", "text": "The Brass Key\n\nRain fell."}
        self.assertEqual(canonical_document_text(record), record["text"])
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = FakeTokenizer()
            _pack_split(
                [record],
                split="train",
                output_dir=Path(directory),
                tokenizer=tokenizer,
                sequence_length=8,
            )
            self.assertEqual(tokenizer.encoded, [record["text"]])
            self.assertEqual(tokenizer.encoded[0].count(record["title"]), 1)

    def test_noncanonical_title_body_record_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "canonical"):
            canonical_document_text({"title": "Title", "text": "Body only"})


class MixedTrainingConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(
            Path("configs/data30m-15m-narrative-v1.yaml").read_text(encoding="utf-8")
        )

    def test_locked_architecture_and_compute_policy(self):
        validate_locked_config(self.config)
        model = FictionPulperLM(model_config_from_dict(self.config["model"]))
        self.assertEqual(model.trainable_parameter_count(), LOCKED_PARAMETER_COUNT)
        self.assertEqual(
            sum(
                source["valid_targets"]
                for source in self.config["mixed_data"]["training_sources"].values()
            ),
            LOCKED_TOTAL_VALID_TARGETS,
        )
        self.assertNotIn("test_path", self.config["data"])
        self.assertEqual(self.config["training"]["checkpoint_steps"], LOCKED_CHECKPOINT_STEPS)
        self.assertTrue(self.config["training"]["require_clean_worktree"])
        self.assertEqual(self.config["curriculum_packing"]["source_dir"], "data/narrative_curriculum_v1")
        required_hashes = {
            "expected_corpus_sha256",
            "expected_split_sha256",
            "expected_packed_metadata_sha256",
            "expected_train_sha256",
            "expected_train_index_sha256",
            "expected_validation_sha256",
            "expected_validation_index_sha256",
        }
        self.assertTrue(required_hashes.issubset(self.config["data"]))

    def test_curriculum_is_exactly_fifteen_percent_by_valid_targets(self):
        curriculum = self.config["mixed_data"]["training_sources"]["curriculum"][
            "valid_targets"
        ]
        self.assertAlmostEqual(curriculum / LOCKED_TOTAL_VALID_TARGETS, 0.15, places=8)

    def test_pretraining_packer_cannot_open_test_or_holdout(self):
        self.assertEqual(PACKED_SPLITS, ("train", "validation"))
        self.assertNotIn("test", json.dumps(self.config["curriculum_packing"]))

    def test_locked_policy_rejects_drift(self):
        changed = copy.deepcopy(self.config)
        changed["training"]["warmup_steps"] = 110
        with self.assertRaisesRegex(RuntimeError, "Compute5"):
            validate_locked_config(changed)

    def test_schedule_contract_rejects_source_path_drift(self):
        schedule = {
            "seed": 1337,
            "batch_size": 16,
            "gradient_accumulation_steps": 4,
            "max_steps": 2220,
            "sequence_length": 1024,
            "total_chunks": 142080,
            "total_valid_targets": LOCKED_TOTAL_VALID_TARGETS,
            "sources": {
                name: {"valid_targets": details["valid_targets"]}
                for name, details in self.config["mixed_data"]["training_sources"].items()
            },
            "source_artifacts": {
                name: {"bin_path": details["path"]}
                for name, details in self.config["mixed_data"]["training_sources"].items()
            },
        }
        _validate_schedule_contract(self.config, schedule)
        schedule["source_artifacts"]["real"]["bin_path"] = "wrong.bin"
        with self.assertRaisesRegex(RuntimeError, "artifact path"):
            _validate_schedule_contract(self.config, schedule)

    def test_exact_architecture_field_drift_is_rejected(self):
        changed = copy.deepcopy(self.config)
        changed["model"]["normalization"] = "layernorm"
        with self.assertRaisesRegex(RuntimeError, "architecture"):
            validate_locked_config(changed)

    def test_tracked_worktree_policy_ignores_untracked_status(self):
        clean = type("Result", (), {"returncode": 0})()
        with patch("src.train_mixed.subprocess.run", return_value=clean) as run:
            self.assertFalse(tracked_worktree_dirty())
            self.assertEqual(run.call_count, 2)

    def test_runtime_rejects_cuda_without_bf16(self):
        with (
            patch("src.train_mixed.torch.cuda.is_available", return_value=True),
            patch("src.train_mixed.torch.cuda.is_bf16_supported", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "BF16"):
                validate_runtime_environment()

    def test_checkpoint_contains_scheduler_rng_and_fresh_init_evidence(self):
        model = FictionPulperLM(
            ModelConfig(
                vocab_size=16,
                hidden_size=8,
                num_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                intermediate_size=16,
                max_seq_len=8,
            )
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = {"training": {"max_steps": 2, "warmup_steps": 1, "learning_rate": 1e-3, "min_learning_rate": 1e-4}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            _save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                config=config,
                step=1,
                real_validation_loss=3.0,
                curriculum_validation_loss=2.0,
                schedule_sha256="abc",
                source_valid_targets_observed={"real": 10, "curriculum": 2},
                metrics=[],
            )
            payload = torch.load(path, weights_only=False)
            self.assertIn("scheduler", payload)
            self.assertIn("rng_state", payload)
            self.assertTrue(payload["fresh_random_initialization"])
            self.assertIsNone(payload["resume_checkpoint"])

    def test_checkpoint_failure_removes_temporary_file(self):
        model = FictionPulperLM(
            ModelConfig(16, 8, 1, 2, 1, 16, 8)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = {"training": {"max_steps": 2, "warmup_steps": 1, "learning_rate": 1e-3, "min_learning_rate": 1e-4}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"

            def fail_save(_payload, temporary):
                Path(temporary).write_bytes(b"partial")
                raise OSError("disk full")

            with patch("src.train_mixed.torch.save", side_effect=fail_save):
                with self.assertRaisesRegex(OSError, "disk full"):
                    _save_checkpoint(
                        path,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        step=1,
                        real_validation_loss=3.0,
                        curriculum_validation_loss=2.0,
                        schedule_sha256="abc",
                        source_valid_targets_observed={"real": 10, "curriculum": 2},
                        metrics=[],
                    )
            self.assertFalse(path.with_suffix(".pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
