import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.data100m_preflight import (
    LOCKED_MODEL,
    LOCKED_TRAINING,
    assess_readiness,
    require_training_ready,
)
from src.model import FictionPulperLM, model_config_from_dict
from src.train import load_evaluation_protocol
from src.train import run_training


class Data100MPreflightTests(unittest.TestCase):
    def test_committed_config_locks_plain_50m_contract(self):
        config = yaml.safe_load(Path("configs/data100m-50m-v1.yaml").read_text())
        self.assertEqual(
            {key: config["model"][key] for key in LOCKED_MODEL}, LOCKED_MODEL
        )
        model = FictionPulperLM(model_config_from_dict(config["model"]))
        self.assertEqual(model.trainable_parameter_count(), 50_348_544)
        self.assertEqual(config["training"]["objective"], "ordinary_causal_language_modeling")
        self.assertEqual(config["training"]["initialization"], "fresh_random")
        self.assertIsNone(config["training"]["resume_checkpoint"])
        self.assertEqual(config["training"]["allocated_token_slots_per_optimizer_step"], 65_536)
        self.assertTrue(
            all(config["training"][key] == value for key, value in LOCKED_TRAINING.items())
        )

    def test_missing_corpus_v3_artifacts_fail_closed(self):
        source = yaml.safe_load(Path("configs/data100m-50m-v1.yaml").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source["data"].update({
                "corpus_seal_path": str(root / "seal.json"),
                "packed_metadata_path": str(root / "metadata.json"),
            })
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(source), encoding="utf-8")
            report = assess_readiness(config_path)
            self.assertEqual(report["status"], "blocked")
            self.assertFalse(report["training_started"])
            self.assertIn("corpus_v3_seal_missing", report["blockers"])
            self.assertIn("corpus_v3_packed_metadata_missing", report["blockers"])
            self.assertIn("corpus_v3_hash_locks_unset", report["blockers"])
            with self.assertRaisesRegex(RuntimeError, "training preflight is blocked"):
                require_training_ready(config_path)

    def test_complete_artifact_chain_passes_and_tampering_fails(self):
        source = yaml.safe_load(Path("configs/data100m-50m-v1.yaml").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            split_path = root / "splits.json"
            corpus.write_text('{"id":"a"}\n', encoding="utf-8")
            split_path.write_text('{"seed":1337,"assignments":{"a":"train"}}\n', encoding="utf-8")
            corpus_hash = hashlib.sha256(corpus.read_bytes()).hexdigest()
            split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
            source["data"].update({
                "corpus_path": str(corpus),
                "split_path": str(split_path),
                "expected_corpus_sha256": corpus_hash,
                "expected_split_sha256": split_hash,
            })
            split_metadata = {}
            for split in ("train", "validation", "test"):
                bin_path = root / f"{split}.bin"
                index_path = root / f"{split}.index.json"
                bin_path.write_bytes(split.encode())
                index_path.write_text('{"documents":[]}', encoding="utf-8")
                source["data"][f"{split}_path"] = str(bin_path)
                split_metadata[split] = {
                    "bin_path": str(bin_path),
                    "bin_sha256": hashlib.sha256(bin_path.read_bytes()).hexdigest(),
                    "index_path": str(index_path),
                    "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                }
            metadata = {
                "corpus_sha256": corpus_hash,
                "split_sha256": split_hash,
                "tokenizer_hash": source["tokenizer"]["expected_sha256"],
                "splits": split_metadata,
                "schedule_candidates": {
                    "5": {"recommended_max_steps": 10, "recommended_warmup_steps": 1}
                },
            }
            metadata_path = root / "metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            source["data"]["packed_metadata_path"] = str(metadata_path)
            source["data"]["expected_packed_metadata_sha256"] = hashlib.sha256(
                metadata_path.read_bytes()
            ).hexdigest()
            manifest_path = root / "manifest.json"
            audits_path = root / "audits.json"
            manifest_path.write_text('{"status":"freeze_ready"}', encoding="utf-8")
            audits_path.write_text('{"failed_required_gates":[]}', encoding="utf-8")
            seal_path = root / "seal.json"
            seal_path.write_text(json.dumps({
                "corpus_id": "fictionpulper-corpus-v3", "corpus_version": 3,
                "status": "sealed", "failed_required_gates": [],
                "unresolved_near_duplicate_clusters": 0, "cross_split_leakage": 0,
                "corpus_sha256": corpus_hash, "split_sha256": split_hash,
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "audits_sha256": hashlib.sha256(audits_path.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            source["data"]["corpus_seal_path"] = str(seal_path)
            source["training"]["max_steps"] = 10
            source["training"]["warmup_steps"] = 1
            gate_report = root / "preflight.json"
            gate_report.write_text(json.dumps({"hardware_gate": {
                "scope": "synthetic_model_only_no_corpus_access",
                "precision": "bf16", "batch_size": 16, "sequence_length": 1024,
                "optimizer": "fused_adamw", "optimizer_updates": 8,
                "model_parameter_count": 50_348_544,
                "tokenizer_sha256": source["tokenizer"]["expected_sha256"],
                "maximum_peak_reserved_gib": 16.0, "peak_reserved_gib": 9.7,
                "used_for_hyperparameter_tuning": False,
                "fits_vram_gate": True, "loss_decreased": True,
                "all_gradients_present": True, "all_gradients_finite": True,
                "checkpoint_reload_max_absolute_logit_difference": 0.0,
            }}), encoding="utf-8")
            source["hardware_gate"]["report_path"] = str(gate_report)
            source["hardware_gate"]["expected_report_sha256"] = hashlib.sha256(
                gate_report.read_bytes()
            ).hexdigest()
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(source), encoding="utf-8")
            self.assertEqual(assess_readiness(config_path)["status"], "ready_for_training")
            gate_payload = json.loads(gate_report.read_text())
            gate_payload["hardware_gate"]["batch_size"] = 8
            gate_report.write_text(json.dumps(gate_payload), encoding="utf-8")
            source["hardware_gate"]["expected_report_sha256"] = hashlib.sha256(
                gate_report.read_bytes()
            ).hexdigest()
            config_path.write_text(yaml.safe_dump(source), encoding="utf-8")
            self.assertIn("hardware_gate_failed", assess_readiness(config_path)["blockers"])
            gate_payload["hardware_gate"]["batch_size"] = 16
            gate_report.write_text(json.dumps(gate_payload), encoding="utf-8")
            source["hardware_gate"]["expected_report_sha256"] = hashlib.sha256(
                gate_report.read_bytes()
            ).hexdigest()
            config_path.write_text(yaml.safe_dump(source), encoding="utf-8")
            corpus.write_text('{"id":"changed"}\n', encoding="utf-8")
            report = assess_readiness(config_path)
            self.assertEqual(report["status"], "blocked")
            self.assertIn("corpus_v3_corpus_hash_mismatch", report["blockers"])

    def test_seal_backed_training_config_always_invokes_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump({"data": {"corpus_seal_path": "seal.json"}}),
                encoding="utf-8",
            )
            with patch(
                "src.data100m_preflight.require_training_ready",
                side_effect=RuntimeError("guard invoked"),
            ) as guard:
                with self.assertRaisesRegex(RuntimeError, "guard invoked"):
                    run_training(config_path)
            guard.assert_called_once_with(config_path)

    def test_evaluation_protocol_preregisters_clean_comparison(self):
        protocol = json.loads(
            Path("experiments/fictionpulper-50m-data100m-v1/evaluation-protocol.json").read_text()
        )
        self.assertTrue(
            protocol["common_clean_lm_comparison"]["required_before_candidate_comparison"]
        )
        self.assertEqual(
            protocol["primary_test"],
            "sealed Corpus-v3 test split, opened once after checkpoint selection",
        )
        self.assertEqual(
            protocol["narrative_benchmark_v2"]["test_access"],
            "after checkpoint selection",
        )
        config = yaml.safe_load(Path("configs/data100m-50m-v1.yaml").read_text())
        loaded = load_evaluation_protocol(config)
        assert loaded is not None
        self.assertEqual(loaded["experiment_id"], "fictionpulper-50m-data100m-v1")


if __name__ == "__main__":
    unittest.main()
