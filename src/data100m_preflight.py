"""Fail-closed preflight for the 50M Corpus-v3 plain-LM experiment."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from src.model import FictionPulperLM, model_config_from_dict
from src.train_tokenizer import sha256_file, write_json_atomic


LOCKED_MODEL = {
    "vocab_size": 4096,
    "hidden_size": 512,
    "num_layers": 16,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "intermediate_size": 1536,
    "max_seq_len": 1024,
    "activation": "swiglu",
    "normalization": "rmsnorm",
    "positional_encoding": "rope",
    "tie_word_embeddings": True,
    "attention_bias": False,
    "mlp_bias": False,
    "expected_parameter_count": 50_348_544,
}
LOCKED_TRAINING = {
    "initialization": "fresh_random",
    "resume_checkpoint": None,
    "objective": "ordinary_causal_language_modeling",
    "optimizer": "adamw",
    "batch_size": 16,
    "gradient_accumulation_steps": 4,
    "allocated_token_slots_per_optimizer_step": 65_536,
    "epochs": 5,
    "learning_rate": 1.0e-3,
    "min_learning_rate": 1.0e-4,
    "warmup_fraction": 0.05,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "precision": "bf16",
    "checkpoint_selection": "minimum_complete_validation_loss",
    "test_access": "after_checkpoint_selection",
    "checkpoint_epochs": [1, 2, 3, 4, 5],
}
LOCKED_HARDWARE = {
    "device": "cuda",
    "maximum_peak_reserved_gib": 16.0,
    "synthetic_fixed_subset_sequences": 16,
    "optimizer_updates": 8,
    "used_for_hyperparameter_tuning": False,
}


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assess_readiness(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    blockers: list[str] = []
    checks: dict[str, Any] = {}

    model_values = {key: config["model"].get(key) for key in LOCKED_MODEL}
    checks["locked_model_contract"] = model_values == LOCKED_MODEL
    if not checks["locked_model_contract"]:
        blockers.append("locked_model_contract_changed")
    model = FictionPulperLM(model_config_from_dict(config["model"]))
    parameter_count = model.trainable_parameter_count()
    checks["trainable_parameter_count"] = parameter_count
    if parameter_count != LOCKED_MODEL["expected_parameter_count"]:
        blockers.append("parameter_count_changed")

    training = config["training"]
    checks["locked_training_contract"] = all(
        training.get(key) == value for key, value in LOCKED_TRAINING.items()
    )
    if not checks["locked_training_contract"]:
        blockers.append("training_contract_changed")
    expected_slots = (
        int(training["batch_size"])
        * int(training["gradient_accumulation_steps"])
        * int(config["model"]["max_seq_len"])
    )
    checks["allocated_token_slots_per_optimizer_step"] = expected_slots
    if expected_slots != int(training["allocated_token_slots_per_optimizer_step"]):
        blockers.append("batch_geometry_changed")
    hardware = config["hardware_gate"]
    checks["locked_hardware_contract"] = all(
        hardware.get(key) == value for key, value in LOCKED_HARDWARE.items()
    )
    if not checks["locked_hardware_contract"]:
        blockers.append("hardware_contract_changed")

    tokenizer_path = Path(config["tokenizer"]["path"])
    expected_tokenizer_hash = config["tokenizer"]["expected_sha256"]
    if not tokenizer_path.exists():
        blockers.append("tokenizer_missing")
    elif sha256_file(tokenizer_path) != expected_tokenizer_hash:
        blockers.append("tokenizer_hash_changed")
    checks["tokenizer_v1_sha256"] = expected_tokenizer_hash

    evaluation = config["evaluation"]
    protocol_path = Path(evaluation["protocol_path"])
    if not protocol_path.exists():
        blockers.append("evaluation_protocol_missing")
    elif sha256_file(protocol_path) != evaluation["expected_protocol_sha256"]:
        blockers.append("evaluation_protocol_hash_changed")
    else:
        protocol = _load_json(protocol_path)
        narrative = protocol.get("narrative_benchmark_v2", {})
        baselines = protocol.get("historical_baselines", {})
        dependencies = [
            (narrative.get("protocol_path"), narrative.get("protocol_sha256")),
            (
                narrative.get("authored_manifest_path"),
                narrative.get("authored_manifest_sha256"),
            ),
            (baselines.get("compute5_seal_path"), baselines.get("compute5_seal_sha256")),
            (
                baselines.get("capacity50m_seal_path"),
                baselines.get("capacity50m_seal_sha256"),
            ),
        ]
        if any(
            not path
            or not _is_sha256(expected_hash)
            or not Path(path).exists()
            or sha256_file(Path(path)) != expected_hash
            for path, expected_hash in dependencies
        ):
            blockers.append("evaluation_protocol_dependency_changed")

    data = config["data"]
    unset_locks = [
        key
        for key in (
            "expected_corpus_sha256",
            "expected_split_sha256",
            "expected_packed_metadata_sha256",
        )
        if not _is_sha256(data.get(key))
    ]
    if unset_locks:
        blockers.append("corpus_v3_hash_locks_unset")
    checks["unset_hash_locks"] = unset_locks

    seal_path = Path(data["corpus_seal_path"])
    seal = None
    if not seal_path.exists():
        blockers.append("corpus_v3_seal_missing")
    else:
        seal = _load_json(seal_path)
        if (
            seal.get("corpus_id") != "fictionpulper-corpus-v3"
            or seal.get("corpus_version") != 3
            or seal.get("status") != "sealed"
            or seal.get("failed_required_gates")
            or seal.get("unresolved_near_duplicate_clusters") != 0
            or seal.get("cross_split_leakage") != 0
        ):
            blockers.append("corpus_v3_seal_invalid")
        if seal.get("corpus_sha256") != data.get("expected_corpus_sha256"):
            blockers.append("seal_corpus_hash_mismatch")
        if seal.get("split_sha256") != data.get("expected_split_sha256"):
            blockers.append("seal_split_hash_mismatch")
        for label, filename, hash_key in (
            ("manifest", "manifest.json", "manifest_sha256"),
            ("audits", "audits.json", "audits_sha256"),
        ):
            path = seal_path.parent / filename
            if not path.exists():
                blockers.append(f"corpus_v3_{label}_missing")
            elif sha256_file(path) != seal.get(hash_key):
                blockers.append(f"corpus_v3_{label}_hash_mismatch")

    for label, path_key, hash_key in (
        ("corpus", "corpus_path", "expected_corpus_sha256"),
        ("split", "split_path", "expected_split_sha256"),
    ):
        path = Path(data[path_key])
        if not path.exists():
            blockers.append(f"corpus_v3_{label}_missing")
        elif _is_sha256(data.get(hash_key)) and sha256_file(path) != data[hash_key]:
            blockers.append(f"corpus_v3_{label}_hash_mismatch")

    packed_path = Path(data["packed_metadata_path"])
    schedule = None
    if not packed_path.exists():
        blockers.append("corpus_v3_packed_metadata_missing")
    else:
        packed = _load_json(packed_path)
        expected_packed_hash = data.get("expected_packed_metadata_sha256")
        if not _is_sha256(expected_packed_hash) or sha256_file(packed_path) != expected_packed_hash:
            blockers.append("corpus_v3_packed_metadata_unlocked")
        if packed.get("corpus_sha256") != data.get("expected_corpus_sha256"):
            blockers.append("packed_corpus_hash_mismatch")
        if packed.get("split_sha256") != data.get("expected_split_sha256"):
            blockers.append("packed_split_hash_mismatch")
        if packed.get("tokenizer_hash") != expected_tokenizer_hash:
            blockers.append("packed_tokenizer_hash_mismatch")
        for split in ("train", "validation", "test"):
            split_metadata = packed.get("splits", {}).get(split, {})
            bin_path = Path(data[f"{split}_path"])
            index_path = bin_path.with_suffix(".index.json")
            if (
                split_metadata.get("bin_path") != str(bin_path)
                or split_metadata.get("index_path") != str(index_path)
            ):
                blockers.append(f"packed_{split}_path_mismatch")
                continue
            for artifact, path in (("bin", bin_path), ("index", index_path)):
                if not path.exists():
                    blockers.append(f"packed_{split}_{artifact}_missing")
                elif sha256_file(path) != split_metadata.get(f"{artifact}_sha256"):
                    blockers.append(f"packed_{split}_{artifact}_hash_mismatch")
        schedule = packed.get("schedule_candidates", {}).get(
            str(training["epochs"])
        )
        if schedule is None:
            blockers.append("packed_schedule_missing")
        else:
            if training.get("max_steps") != schedule["recommended_max_steps"]:
                blockers.append("max_steps_not_locked_to_packed_schedule")
            if training.get("warmup_steps") != schedule["recommended_warmup_steps"]:
                blockers.append("warmup_steps_not_locked_to_packed_schedule")

    report_path = Path(hardware["report_path"])
    expected_report_hash = hardware.get("expected_report_sha256")
    if not _is_sha256(expected_report_hash):
        blockers.append("hardware_gate_report_hash_unset")
    elif not report_path.exists():
        blockers.append("hardware_gate_report_missing")
    elif sha256_file(report_path) != expected_report_hash:
        blockers.append("hardware_gate_report_hash_mismatch")
    else:
        gate = _load_json(report_path).get("hardware_gate", {})
        gate_passed = (
            gate.get("scope") == "synthetic_model_only_no_corpus_access"
            and gate.get("precision") == training["precision"]
            and gate.get("batch_size") == training["batch_size"]
            and gate.get("sequence_length") == config["model"]["max_seq_len"]
            and gate.get("optimizer") == "fused_adamw"
            and gate.get("optimizer_updates") == hardware["optimizer_updates"]
            and gate.get("model_parameter_count") == LOCKED_MODEL["expected_parameter_count"]
            and gate.get("tokenizer_sha256") == expected_tokenizer_hash
            and gate.get("maximum_peak_reserved_gib") == hardware["maximum_peak_reserved_gib"]
            and gate.get("peak_reserved_gib", math.inf) <= hardware["maximum_peak_reserved_gib"]
            and gate.get("used_for_hyperparameter_tuning") is False
            and gate.get("fits_vram_gate") is True
            and gate.get("loss_decreased") is True
            and gate.get("all_gradients_present") is True
            and gate.get("all_gradients_finite") is True
            and gate.get("checkpoint_reload_max_absolute_logit_difference") == 0.0
        )
        checks["hardware_gate_passed"] = gate_passed
        if not gate_passed:
            blockers.append("hardware_gate_failed")

    unique_blockers = sorted(set(blockers))
    return {
        "experiment_id": config["experiment"]["id"],
        "status": "ready_for_training" if not unique_blockers else "blocked",
        "training_started": False,
        "full_training_claimed": False,
        "checks": checks,
        "schedule": schedule,
        "blockers": unique_blockers,
    }


def require_training_ready(config_path: Path) -> dict[str, Any]:
    report = assess_readiness(config_path)
    if report["status"] != "ready_for_training":
        raise RuntimeError(
            "Data100M training preflight is blocked: " + ", ".join(report["blockers"])
        )
    return report


def run_hardware_gate(config_path: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the BF16 hardware gate")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    model = FictionPulperLM(model_config_from_dict(config["model"])).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        fused=True,
    )
    batch_size = int(training["batch_size"])
    sequence_length = int(config["model"]["max_seq_len"])
    vocab_size = int(config["model"]["vocab_size"])
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    input_ids = torch.randint(
        0, vocab_size, (batch_size, sequence_length), generator=generator, device=device
    )
    labels = (input_ids + 1) % vocab_size
    updates = int(config["hardware_gate"]["optimizer_updates"])
    losses = []
    all_gradients_present = True
    all_gradients_finite = True
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(input_ids, labels)
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError("Hardware gate loss is not finite")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        all_gradients_present &= all(gradient is not None for gradient in gradients)
        all_gradients_finite &= all(
            gradient is not None and bool(torch.isfinite(gradient).all())
            for gradient in gradients
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip"]))
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "gate.pt"
        probe = input_ids[:1, :16]
        model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            expected_logits, _ = model(probe)
        torch.save(model.state_dict(), checkpoint)
        restored = FictionPulperLM(model_config_from_dict(config["model"])).to(device).eval()
        restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actual_logits, _ = restored(probe)
        reload_difference = float((expected_logits - actual_logits).abs().max())
    maximum_reserved = float(config["hardware_gate"]["maximum_peak_reserved_gib"])
    return {
        "scope": "synthetic_model_only_no_corpus_access",
        "device": torch.cuda.get_device_name(device),
        "precision": "bf16",
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "optimizer": "fused_adamw",
        "optimizer_updates": updates,
        "model_parameter_count": model.trainable_parameter_count(),
        "tokenizer_sha256": config["tokenizer"]["expected_sha256"],
        "losses": losses,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "all_gradients_present": all_gradients_present,
        "all_gradients_finite": all_gradients_finite,
        "checkpoint_reload_max_absolute_logit_difference": reload_difference,
        "elapsed_seconds": elapsed,
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "maximum_peak_reserved_gib": maximum_reserved,
        "fits_vram_gate": peak_reserved <= maximum_reserved,
        "used_for_hyperparameter_tuning": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data100m-50m-v1.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hardware-gate", action="store_true")
    parser.add_argument("--require-ready", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = assess_readiness(args.config)
    if args.hardware_gate:
        report["hardware_gate"] = run_hardware_gate(args.config)
        gate = report["hardware_gate"]
        if not (
            gate["fits_vram_gate"]
            and gate["loss_decreased"]
            and gate["all_gradients_present"]
            and gate["all_gradients_finite"]
            and gate["checkpoint_reload_max_absolute_logit_difference"] == 0.0
        ):
            raise RuntimeError("Synthetic BF16 hardware gate failed")
    if args.output:
        write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))
    if args.require_ready and report["status"] != "ready_for_training":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
