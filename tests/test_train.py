import math
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.train import (
    GENERATION_SETTINGS,
    benchmark_excluded_ids,
    learning_rate_for_step,
    selected_packed_schedule,
    validate_packed_artifacts,
)
from src.train_tokenizer import sha256_file


class TrainingScheduleTests(unittest.TestCase):
    def test_linear_warmup_and_cosine_decay(self):
        values = [
            learning_rate_for_step(
                step,
                max_steps=420,
                warmup_steps=21,
                learning_rate=1e-3,
                min_learning_rate=1e-4,
            )
            for step in range(1, 421)
        ]
        self.assertAlmostEqual(values[0], 1e-3 / 21)
        self.assertAlmostEqual(values[20], 1e-3)
        self.assertAlmostEqual(values[-1], 1e-4)
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertTrue(all(left >= right for left, right in zip(values[20:], values[21:])))

    def test_schedule_rejects_invalid_step(self):
        with self.assertRaises(ValueError):
            learning_rate_for_step(
                0,
                max_steps=420,
                warmup_steps=21,
                learning_rate=1e-3,
                min_learning_rate=1e-4,
            )

    def test_data30m_primary_schedule_is_locked_to_three_epochs(self):
        config = yaml.safe_load(Path("configs/data30m-15m-v1.yaml").read_text())
        metadata = json.loads(
            Path("data/packed/corpus-v2-data30m/metadata.json").read_text()
        )
        schedule = selected_packed_schedule(metadata, config["training"]["epochs"])
        self.assertEqual(config["training"]["epochs"], 3)
        self.assertEqual(config["training"]["max_steps"], 1332)
        self.assertEqual(config["training"]["warmup_steps"], 67)
        self.assertEqual(config["training"]["checkpoint_epochs"], [1, 2, 3])
        self.assertEqual(schedule["optimizer_steps_per_epoch"], 444)
        self.assertEqual(schedule["recommended_max_steps"], 1332)
        self.assertEqual(schedule["recommended_warmup_steps"], 67)

    def test_data30m_compute5_changes_only_locked_duration_fields(self):
        baseline = yaml.safe_load(Path("configs/data30m-15m-v1.yaml").read_text())
        compute5 = yaml.safe_load(
            Path("configs/data30m-15m-compute5.yaml").read_text()
        )
        metadata = json.loads(
            Path("data/packed/corpus-v2-data30m/metadata.json").read_text()
        )
        self.assertEqual(compute5["model"], baseline["model"])
        self.assertEqual(compute5["tokenizer"], baseline["tokenizer"])
        self.assertEqual(compute5["data"], baseline["data"])
        for key in (
            "optimizer",
            "batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "min_learning_rate",
            "weight_decay",
            "grad_clip",
            "precision",
            "require_clean_worktree",
        ):
            self.assertEqual(compute5["training"][key], baseline["training"][key])
        schedule = selected_packed_schedule(metadata, compute5["training"]["epochs"])
        self.assertEqual(compute5["training"]["epochs"], 5)
        self.assertEqual(compute5["training"]["max_steps"], 2220)
        self.assertEqual(compute5["training"]["warmup_steps"], 111)
        self.assertEqual(compute5["training"]["checkpoint_epochs"], [1, 2, 3, 4, 5])
        self.assertEqual(schedule["optimizer_steps_per_epoch"], 444)
        self.assertEqual(schedule["recommended_max_steps"], 2220)
        self.assertEqual(schedule["recommended_warmup_steps"], 111)

    def test_context2k_changes_only_locked_context_and_batching_fields(self):
        baseline = yaml.safe_load(
            Path("configs/data30m-15m-compute5.yaml").read_text()
        )
        context2k = yaml.safe_load(
            Path("configs/data30m-15m-context2k-v1.yaml").read_text()
        )
        metadata = json.loads(
            Path("data/packed/corpus-v2-data30m-context2k/metadata.json").read_text()
        )
        for key, value in baseline["model"].items():
            if key != "max_seq_len":
                self.assertEqual(context2k["model"][key], value)
        self.assertEqual(context2k["model"]["max_seq_len"], 2048)
        self.assertEqual(context2k["tokenizer"], baseline["tokenizer"])
        for key in (
            "optimizer",
            "batch_size",
            "learning_rate",
            "min_learning_rate",
            "weight_decay",
            "grad_clip",
            "precision",
            "require_clean_worktree",
        ):
            self.assertEqual(context2k["training"][key], baseline["training"][key])
        self.assertEqual(context2k["training"]["gradient_accumulation_steps"], 2)
        schedule = selected_packed_schedule(metadata, context2k["training"]["epochs"])
        self.assertEqual(schedule["allocated_token_slots_per_optimizer_step"], 65_536)
        self.assertEqual(schedule["real_target_tokens_per_epoch"], 27_177_271)
        self.assertEqual(schedule["optimizer_steps_per_epoch"], 473)
        self.assertEqual(schedule["recommended_max_steps"], 2365)
        self.assertEqual(schedule["recommended_warmup_steps"], 118)
        self.assertEqual(context2k["evaluation"]["historical_sequence_length"], 1024)

    def test_packed_artifact_validation_rejects_changed_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = {}
            data = {}
            for split in ("train", "validation", "test"):
                bin_path = root / f"{split}.bin"
                index_path = root / f"{split}.index.json"
                bin_path.write_bytes(b"locked")
                index_path.write_text('{"documents": []}', encoding="utf-8")
                data[f"{split}_path"] = str(bin_path)
                splits[split] = {
                    "bin_path": str(bin_path),
                    "bin_sha256": sha256_file(bin_path),
                    "index_path": str(index_path),
                    "index_sha256": sha256_file(index_path),
                }
            validate_packed_artifacts({"data": data}, {"splits": splits})
            (root / "test.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "test packed binary hash changed"):
                validate_packed_artifacts({"data": data}, {"splits": splits})

    def test_benchmark_exclusions_are_derived_from_data30m_train(self):
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "test.index.json"
            index_path.write_text(
                json.dumps({"documents": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}),
                encoding="utf-8",
            )
            excluded = benchmark_excluded_ids(
                {
                    "name": "common",
                    "index_path": str(index_path),
                    "exclude_ids": ["c"],
                    "exclude_data30m_train_ids": True,
                    "expected_excluded_documents": 2,
                },
                {"a": "train", "b": "test", "c": "validation"},
            )
            self.assertEqual(excluded, {"a", "c"})

    def test_generation_protocol_matches_sealed_baseline(self):
        self.assertEqual(
            GENERATION_SETTINGS,
            {
                "greedy_max_new_tokens": 128,
                "sampled_max_new_tokens": 256,
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 0.95,
            },
        )


if __name__ == "__main__":
    unittest.main()
