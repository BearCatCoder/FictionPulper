import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.continuation_sft import (
    ContinuationDataset,
    FORMAT,
    build_schedule,
    load_config,
    load_export,
    run_post_selection,
    schedule_accounting,
    select_checkpoint,
    verify_selection,
)
from src.continuity_curriculum.common import sha256_file, write_json_atomic, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "continuation-sft-v1.yaml"


class ContinuationSFTTests(unittest.TestCase):
    def test_frozen_config_has_no_preselection_test_surface(self):
        config = load_config(CONFIG_PATH, sha256_file(CONFIG_PATH))
        encoded = json.dumps(config)
        self.assertNotIn("test", encoded)
        self.assertNotIn("post_selection", config)
        self.assertFalse(config["corpus"]["prepared_jsonl_authorized"])
        self.assertEqual(config["training"]["seed"], 9009)
        self.assertEqual(config["training"]["max_optimizer_steps"], 134)

    def test_padding_and_prompt_are_masked(self):
        dataset = ContinuationDataset([{
            "input_ids": [1, 2, 3, 4],
            "labels": [-100, -100, 4, 5],
        }], pad_id=0, length=8)
        inputs, labels = dataset[0]
        self.assertEqual(inputs.tolist(), [1, 2, 3, 4, 0, 0, 0, 0])
        self.assertEqual(labels.tolist(), [-100, -100, 4, 5, -100, -100, -100, -100])

    def test_schedule_is_deterministic_and_accounts_without_claiming_exact_15_percent(self):
        first = build_schedule()
        self.assertEqual(first, build_schedule())
        self.assertNotEqual(first, build_schedule(9010))
        accounting = schedule_accounting(first)
        self.assertEqual(accounting["sft_microbatches"], 40)
        self.assertEqual(accounting["replay_microbatches"], 228)
        self.assertEqual(accounting["sft_example_presentations"], 320)
        self.assertEqual(accounting["replay_example_presentations"], 1824)
        self.assertAlmostEqual(accounting["actual_sft_ratio"], 40 / 268)
        self.assertNotEqual(accounting["actual_sft_ratio"], accounting["target_sft_ratio"])

    def test_export_loader_rejects_tampering_and_test_entries(self):
        config = load_config(CONFIG_PATH, sha256_file(CONFIG_PATH))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {}
            for split, count in (("train", 160), ("validation", 20)):
                path = root / f"{split}.jsonl"
                write_jsonl(path, [{
                    "split": split, "input_ids": [1, 2, 3], "labels": [-100, 3, 4],
                    "continuation_target_count_including_eos": 2, "value": index,
                } for index in range(count)])
                artifacts[split] = {"path": str(path), "sha256": sha256_file(path), "examples": count}
            manifest = {
                "status": "sealed_train_validation_export", "included_splits": ["train", "validation"],
                "excluded_split": "test", "prepared_jsonl_used": False,
                "canonical_examples_sha256": config["corpus"]["examples_sha256"],
                "approval_sha256": config["corpus"]["approval_sha256"], "artifacts": artifacts,
            }
            manifest_path = root / "manifest.json"
            write_json_atomic(manifest_path, manifest)
            digest = sha256_file(manifest_path)
            manifest.update({
                "format": "fictionpulper-continuation-training-export-v1",
                "tokenizer_sha256": config["tokenizer"]["sha256"],
                "schema_sha256": "2057ea5c6f540ac68a54acd629551f9d3a44066bae81013be1bfac5e1dbb0634",
                "split_seed": config["corpus"]["split_seed"],
            })
            write_json_atomic(manifest_path, manifest)
            digest = sha256_file(manifest_path)
            with patch("src.continuation_sft.FROZEN_EXPORT_MANIFEST_SHA256", digest):
                self.assertEqual(set(load_export(config, manifest_path, digest)), {"train", "validation"})
            manifest["included_splits"].append("test")
            write_json_atomic(manifest_path, manifest)
            with self.assertRaisesRegex(RuntimeError, "manifest hash mismatch"):
                load_export(config, manifest_path, digest)

    def test_selection_uses_validation_nll_then_earliest_step(self):
        metrics = [
            {"step": 0, "selection_eligible": False, "continuation_validation": {"nll": 1.0}},
            {"step": 67, "selection_eligible": True, "continuation_validation": {"nll": 2.0}, "exposure": {"sft_examples": 160, "replay_examples": 912}},
            {"step": 134, "selection_eligible": True, "continuation_validation": {"nll": 2.0}},
        ]
        self.assertEqual(select_checkpoint(metrics)["step"], 67)

    def test_selected_checkpoint_gate_checks_base_and_fresh_optimizer_provenance(self):
        config = load_config(CONFIG_PATH, sha256_file(CONFIG_PATH))
        metrics = [
            {"epoch": 0, "step": 0, "selection_eligible": False, "continuation_validation": {"nll": 3.0}},
            {
                "epoch": 1, "step": 67, "selection_eligible": True,
                "continuation_validation": {"nll": 2.0},
                "exposure": {"sft_examples": 160, "replay_examples": 912},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "candidate.pt"
            payload = {
                "format": FORMAT, "resume_checkpoint": None, "optimizer_initialized_fresh": True,
                "base_checkpoint_sha256": config["base"]["checkpoint_sha256"], "step": 67,
                "seed": 9009, "selection_metric": "continuation_validation_nll",
                "exposure": {"sft_examples": 160, "replay_examples": 912}, "config": config,
                "config_sha256": "a" * 64, "export_manifest_sha256": "b" * 64,
                "preflight_sha256": "c" * 64,
                "epoch": 1, "metrics": metrics,
            }
            torch.save(payload, checkpoint_path)
            summary = {
                "status": "training_and_validation_selection_complete", "metrics": metrics,
                "selected_epoch": 1, "selected_step": 67,
                "selected_checkpoint_sha256": sha256_file(checkpoint_path),
                "heldout_access_during_training": False,
                "base_checkpoint_sha256": config["base"]["checkpoint_sha256"],
                "optimizer_initialized_fresh": True,
                "schedule_accounting": schedule_accounting(build_schedule()),
                "exposure": {"sft_examples": 320, "replay_examples": 1824},
                "selected_checkpoint_exposure": {"sft_examples": 160, "replay_examples": 912},
                "config_sha256": "a" * 64, "export_manifest_sha256": "b" * 64,
                "preflight_sha256": "c" * 64,
            }
            summary_path = root / "summary.json"
            write_json_atomic(summary_path, summary)
            verify_selection(config, summary_path, checkpoint_path)
            payload["base_checkpoint_sha256"] = "0" * 64
            torch.save(payload, checkpoint_path)
            summary["selected_checkpoint_sha256"] = sha256_file(checkpoint_path)
            write_json_atomic(summary_path, summary)
            with self.assertRaisesRegex(RuntimeError, "base provenance"):
                verify_selection(config, summary_path, checkpoint_path)

    def test_post_selection_does_not_open_holdouts_before_selection_gate(self):
        config = load_config(CONFIG_PATH, sha256_file(CONFIG_PATH))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("src.continuation_sft.load_jsonl") as opened:
                with self.assertRaisesRegex(RuntimeError, "missing validation selection summary"):
                    run_post_selection(
                        config, root / "summary.json", root / "candidate.pt", "1" * 64,
                        root / "output", root / "protocol.json", "0" * 64,
                    )
                opened.assert_not_called()
                self.assertFalse((root / "output").exists())

    def test_config_hash_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_bytes(CONFIG_PATH.read_bytes())
            digest = sha256_file(path)
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "config.*hash mismatch"):
                load_config(path, digest)


if __name__ == "__main__":
    unittest.main()
