import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.dynamic_logit_post_selection import (
    CANDIDATE_KEY,
    EXPERIMENT_ID,
    SELECTED_STEP,
    TRAINING_COMMIT,
    finalize,
    run_after_gate,
    verify_protocol,
    verify_selection_gate,
)
from src.train import GENERATION_SETTINGS
from src.train_tokenizer import sha256_file


def write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return sha256_file(path)


class DynamicLogitPostSelectionTests(unittest.TestCase):
    def protocol_fixture(self, root: Path):
        config = root / "config.yaml"
        config.write_text(yaml.safe_dump({
            "experiment_id": EXPERIMENT_ID,
            "selection": {
                "metric": "data30m_validation_loss",
                "dynamic_validation_role": "held_out_diagnostic_only_not_selection",
                "test_access": "forbidden_during_training",
                "generalization_access": "forbidden_during_training",
            },
        }), encoding="utf-8")
        files = {}
        for name in ("tokenizer", "counterfactual", "dynamic_manifest", "dynamic_train",
                     "dynamic_validation", "packed"):
            path = root / name
            path.write_text(name, encoding="utf-8")
            files[name] = path
        schedule = root / "schedule.json"
        write_json(schedule, {"schedule_content_sha256": "schedule-content"})
        exposure = {"real_targets": 1, "curriculum_targets": 2, "pair_evals": 3,
                    "dynamic_decisions": 4, "stale_candidates": 5}
        metrics = [
            {"step": 444, "selection_eligible": True, "data30m_validation": {"loss": 2.0},
             "dynamic_validation_diagnostic": {"selection_eligible": False}, "exposure": {}},
            {"step": SELECTED_STEP, "selection_eligible": True, "data30m_validation": {"loss": 1.0},
             "dynamic_validation_diagnostic": {"selection_eligible": False}, "exposure": exposure},
        ]
        checkpoint_locks = {
            "tokenizer_sha256": sha256_file(files["tokenizer"]), "schedule_file_sha256": sha256_file(schedule),
            "schedule_content_sha256": "schedule-content", "dynamic_manifest_sha256": sha256_file(files["dynamic_manifest"]),
            "dynamic_train_sha256": sha256_file(files["dynamic_train"]),
            "dynamic_validation_sha256": sha256_file(files["dynamic_validation"]), "counterfactual_train_sha256": "cf-train",
        }
        checkpoint = root / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        summary = {
            "status": "training_and_selection_complete", "optimizer_steps": SELECTED_STEP,
            "selection_metric": "Data30M validation loss only", "auxiliary_heads": False,
            "metrics": metrics, "best_checkpoint_step": SELECTED_STEP,
            "best_data30m_validation_loss": 1.0, "exposure": exposure,
            "best_checkpoint_sha256": sha256_file(checkpoint),
        }
        manifest = {
            "selection_metric": "Data30M validation loss only", "held_out_access_during_training": False,
            "held_out_dynamic_validation_selection_eligible": False, "fresh_random_initialization": True,
            "resume_checkpoint": None, "training_git_commit": TRAINING_COMMIT, "auxiliary_heads": False,
            "locks": checkpoint_locks, "dataset_gates": {"manifest_sha256": sha256_file(files["counterfactual"])},
        }
        summary_path, manifest_path = root / "summary.json", root / "manifest.json"
        summary_hash, manifest_hash = write_json(summary_path, summary), write_json(manifest_path, manifest)
        locks = {
            "config_path": str(config), "config_sha256": sha256_file(config),
            "tokenizer_path": str(files["tokenizer"]), "tokenizer_sha256": sha256_file(files["tokenizer"]),
            "counterfactual_manifest_path": str(files["counterfactual"]), "counterfactual_manifest_sha256": sha256_file(files["counterfactual"]),
            "dynamic_manifest_path": str(files["dynamic_manifest"]), "dynamic_manifest_sha256": sha256_file(files["dynamic_manifest"]),
            "dynamic_train_path": str(files["dynamic_train"]), "dynamic_train_sha256": sha256_file(files["dynamic_train"]),
            "dynamic_validation_path": str(files["dynamic_validation"]), "dynamic_validation_sha256": sha256_file(files["dynamic_validation"]),
            "packed_metadata_path": str(files["packed"]), "packed_metadata_sha256": sha256_file(files["packed"]),
            "schedule_path": str(schedule), "schedule_file_sha256": sha256_file(schedule),
            "schedule_content_sha256": "schedule-content",
        }
        payload = {
            "format": "fictionpulper-dynamic-logit-v1", "step": SELECTED_STEP,
            "selection_metric": "Data30M validation loss only", "fresh_random_initialization": True,
            "resume_checkpoint": None, "model_initialization_seed": 1337, "auxiliary_heads": False,
            "config": yaml.safe_load(config.read_text(encoding="utf-8")), "locks": checkpoint_locks,
            "metrics": metrics,
        }
        protocol = {
            "version": 1, "experiment_id": EXPERIMENT_ID, "candidate_key": CANDIDATE_KEY,
            "selection_metric": "Data30M validation loss only", "required_training_status": "training_and_selection_complete",
            "training_commit": TRAINING_COMMIT, "seed": 1337,
            "candidate": {"summary_path": str(summary_path), "summary_sha256": summary_hash,
                          "manifest_path": str(manifest_path), "manifest_sha256": manifest_hash,
                          "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                          "expected_step": SELECTED_STEP, "expected_exposure": exposure},
            "training_locks": locks, "checkpoint_locks": checkpoint_locks,
            "generation": {"settings": GENERATION_SETTINGS, "sample_seed": 11337},
            "counterfactual": {"splits": ["test", "generalization_holdout"]},
            "dynamic": {"splits": ["test", "generalization"]},
        }
        return protocol, payload

    def test_gate_precedes_heldout_access(self):
        calls = []

        def gate():
            calls.append("gate")
            raise RuntimeError("blocked")

        def heldout(*_):
            calls.append("heldout")

        with self.assertRaisesRegex(RuntimeError, "blocked"):
            run_after_gate(gate, heldout)
        self.assertEqual(calls, ["gate"])

    def test_protocol_hash_and_locked_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, _ = self.protocol_fixture(root)
            path = root / "protocol.json"
            digest = write_json(path, protocol)
            verify_protocol(protocol, path, digest)
            changed = copy.deepcopy(protocol)
            changed["training_commit"] = "0" * 40
            with self.assertRaisesRegex(RuntimeError, "Training commit changed"):
                verify_protocol(changed, path, digest)
            with self.assertRaisesRegex(RuntimeError, "protocol hash changed"):
                verify_protocol(protocol, path, "0" * 64)

    def test_gate_proves_selected_validation_and_all_locks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, payload = self.protocol_fixture(root)
            locks = protocol["training_locks"]
            with patch("src.dynamic_logit_post_selection._git_blob_hash", return_value=locks["config_sha256"]), \
                 patch("src.dynamic_logit_post_selection.verify_schedule", return_value=locks["schedule_content_sha256"]):
                _, _, _, proof = verify_selection_gate(protocol, load_checkpoint=lambda _: payload)
            self.assertTrue(proof["minimum_data30m_validation_candidate"])
            self.assertEqual(proof["checkpoint_step"], SELECTED_STEP)

            changed = copy.deepcopy(protocol)
            changed["training_locks"]["dynamic_manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "Training lock dynamic_manifest_path"):
                verify_selection_gate(changed, load_checkpoint=lambda _: payload)

    def test_gate_rejects_nonminimum_selected_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, payload = self.protocol_fixture(root)
            summary_path = Path(protocol["candidate"]["summary_path"])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["metrics"][0]["data30m_validation"]["loss"] = 0.5
            protocol["candidate"]["summary_sha256"] = write_json(summary_path, summary)
            with self.assertRaisesRegex(RuntimeError, "minimum Data30M validation"):
                verify_selection_gate(protocol, load_checkpoint=lambda _: payload)

    def test_gate_rejects_checkpoint_validation_proof_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, payload = self.protocol_fixture(root)
            payload["metrics"][-1]["data30m_validation"]["loss"] = 1.1
            locks = protocol["training_locks"]
            with patch("src.dynamic_logit_post_selection._git_blob_hash", return_value=locks["config_sha256"]), \
                 patch("src.dynamic_logit_post_selection.verify_schedule", return_value=locks["schedule_content_sha256"]), \
                 self.assertRaisesRegex(RuntimeError, "selected validation proof"):
                verify_selection_gate(protocol, load_checkpoint=lambda _: payload)

    def test_gate_rejects_auxiliary_heads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol, payload = self.protocol_fixture(root)
            payload["auxiliary_heads"] = True
            locks = protocol["training_locks"]
            with patch("src.dynamic_logit_post_selection._git_blob_hash", return_value=locks["config_sha256"]), \
                 patch("src.dynamic_logit_post_selection.verify_schedule", return_value=locks["schedule_content_sha256"]), \
                 self.assertRaisesRegex(RuntimeError, "auxiliary-head"):
                verify_selection_gate(protocol, load_checkpoint=lambda _: payload)

    def test_finalize_requires_every_stage_artifact(self):
        required = (
            "selection-gate.json", "dynamic-logit-evaluation.json", "counterfactual-evaluation.json",
            "13fact-forced-choice.json", "13fact-free-generation.json", "manual-fact-scoring.json",
            "generation-comparison.json", "repetition-comparison.json", "real-fiction-evaluation.json",
            "probe-candidate-raw.json", "probe-comparison.json",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "protocol.json"
            write_json(protocol_path, {})
            protocol = {"candidate": {"checkpoint_sha256": "a" * 64}}
            with self.assertRaisesRegex(RuntimeError, "missing artifacts"):
                finalize(protocol, protocol_path, root)
            for name in required:
                value = {"status": "complete"}
                if name == "selection-gate.json":
                    value = {"status": "PASS", "heldout_opened": False}
                elif name == "manual-fact-scoring.json":
                    value = {"status": "awaiting_manual_scoring"}
                write_json(root / name, value)
            manifest = finalize(protocol, protocol_path, root)
            self.assertEqual(manifest["experiment_id"], EXPERIMENT_ID)
            self.assertTrue((root / "post-selection-summary.json").is_file())
            self.assertTrue((root / "artifact-manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
