"""Train the locked FictionPulper narrative experiment from a mixed schedule."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import PackedStoryDataset
from src.mixed_schedule import ScheduledMixedDataset, verify_schedule
from src.model import FictionPulperLM, model_config_from_dict
from src.train import evaluate, learning_rate_for_step, set_seed
from src.train_tokenizer import sha256_file, write_json_atomic


LOCKED_PARAMETER_COUNT = 15_047_040
LOCKED_SEED = 1337
LOCKED_STEPS = 2220
LOCKED_TOTAL_VALID_TARGETS = 135_886_355
LOCKED_CHECKPOINT_STEPS = [444, 888, 1332, 1776, 2220]
LOCKED_MODEL = {
    "vocab_size": 4096,
    "hidden_size": 384,
    "num_layers": 8,
    "num_attention_heads": 6,
    "num_key_value_heads": 2,
    "intermediate_size": 1120,
    "max_seq_len": 1024,
    "activation": "swiglu",
    "normalization": "rmsnorm",
    "positional_encoding": "rope",
    "tie_word_embeddings": True,
    "attention_bias": False,
    "mlp_bias": False,
    "expected_parameter_count": LOCKED_PARAMETER_COUNT,
}


def tracked_worktree_dirty() -> bool | None:
    """Ignore untracked/ignored artifacts but reject staged or unstaged tracked edits."""
    for command in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        result = subprocess.run(command, check=False)
        if result.returncode == 1:
            return True
        if result.returncode != 0:
            return None
    return False


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def validate_runtime_environment() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for locked BF16 training")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The locked mixed run requires CUDA BF16 support")


def validate_locked_config(config: dict[str, Any]) -> None:
    training = config["training"]
    expected_training = {
        "optimizer": "adamw",
        "batch_size": 16,
        "gradient_accumulation_steps": 4,
        "epochs": 5,
        "max_steps": LOCKED_STEPS,
        "learning_rate": 1e-3,
        "min_learning_rate": 1e-4,
        "warmup_steps": 111,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "precision": "bf16",
        "validation_interval": 444,
        "checkpoint_steps": LOCKED_CHECKPOINT_STEPS,
        "require_clean_worktree": True,
    }
    observed = {key: training[key] for key in expected_training}
    if int(config["seed"]) != LOCKED_SEED or observed != expected_training:
        raise RuntimeError("Locked Compute5 policy changed")
    if config["model"] != LOCKED_MODEL:
        raise RuntimeError("Locked 15M architecture fields changed")
    if config["tokenizer"].get("type") != "BPE" or int(
        config["tokenizer"].get("vocab_size", -1)
    ) != LOCKED_MODEL["vocab_size"]:
        raise RuntimeError("Locked tokenizer configuration changed")
    source_targets = sum(
        int(details["valid_targets"])
        for details in config["mixed_data"]["training_sources"].values()
    )
    if source_targets != LOCKED_TOTAL_VALID_TARGETS:
        raise RuntimeError("Locked valid-target exposure changed")
    if set(config["mixed_data"]["training_sources"]) != {"real", "curriculum"}:
        raise RuntimeError("Mixed training sources must be exactly real and curriculum")
    forbidden = {"test_path", "holdout_path"}
    if forbidden.intersection(config.get("data", {})):
        raise RuntimeError("Training config must not contain test or holdout paths")


def validate_real_artifacts(config: dict[str, Any], tokenizer_hash: str) -> dict[str, Any]:
    data = config["data"]
    metadata_path = Path(data["packed_metadata_path"])
    if sha256_file(metadata_path) != data["expected_packed_metadata_sha256"]:
        raise RuntimeError("Locked packed metadata hash changed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for label, metadata_key, path_key in (
        ("corpus", "corpus_sha256", "corpus_path"),
        ("split", "split_sha256", "split_path"),
    ):
        expected = data[f"expected_{label}_sha256"]
        if metadata[metadata_key] != expected or sha256_file(Path(data[path_key])) != expected:
            raise RuntimeError(f"Locked real {label} artifact changed")
    if metadata["tokenizer_hash"] != tokenizer_hash:
        raise RuntimeError("Real packed data tokenizer hash changed")
    verified: dict[str, Any] = {}
    for split in ("train", "validation"):
        bin_path = Path(data[f"{split}_path"])
        index_path = bin_path.with_suffix(".index.json")
        expected_bin = data[f"expected_{split}_sha256"]
        expected_index = data[f"expected_{split}_index_sha256"]
        split_metadata = metadata["splits"][split]
        if split_metadata["bin_path"] != str(bin_path) or split_metadata["index_path"] != str(
            index_path
        ):
            raise RuntimeError(f"Locked real {split} paths differ from packed metadata")
        if split_metadata["bin_sha256"] != expected_bin or sha256_file(bin_path) != expected_bin:
            raise RuntimeError(f"Locked real {split} binary changed")
        if (
            split_metadata["index_sha256"] != expected_index
            or sha256_file(index_path) != expected_index
        ):
            raise RuntimeError(f"Locked real {split} index changed")
        verified[split] = {
            "bin_path": str(bin_path),
            "bin_sha256": expected_bin,
            "index_path": str(index_path),
            "index_sha256": expected_index,
        }
    return {"metadata_sha256": data["expected_packed_metadata_sha256"], "splits": verified}


def _packed_dataset(path: str, sequence_length: int, pad_token_id: int) -> PackedStoryDataset:
    bin_path = Path(path)
    return PackedStoryDataset(
        bin_path,
        bin_path.with_suffix(".index.json"),
        sequence_length=sequence_length,
        pad_token_id=pad_token_id,
    )


def _verify_schedule_sources(schedule: dict[str, Any]) -> None:
    for source, artifact in schedule["source_artifacts"].items():
        if sha256_file(Path(artifact["bin_path"])) != artifact["bin_sha256"]:
            raise RuntimeError(f"Scheduled packed binary changed for {source}")
        if sha256_file(Path(artifact["index_path"])) != artifact["index_sha256"]:
            raise RuntimeError(f"Scheduled packed index changed for {source}")


def _validate_schedule_contract(config: dict[str, Any], schedule: dict[str, Any]) -> None:
    training = config["training"]
    expected = {
        "seed": int(config["seed"]),
        "batch_size": int(training["batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "max_steps": int(training["max_steps"]),
        "sequence_length": int(config["model"]["max_seq_len"]),
        "total_chunks": int(config["mixed_data"]["schedule"]["total_chunks"]),
        "total_valid_targets": LOCKED_TOTAL_VALID_TARGETS,
    }
    for key, value in expected.items():
        if int(schedule[key]) != value:
            raise RuntimeError(f"Mixed schedule {key} differs from locked config")
    for source, details in config["mixed_data"]["training_sources"].items():
        if int(schedule["sources"][source]["valid_targets"]) != int(
            details["valid_targets"]
        ):
            raise RuntimeError(f"Mixed schedule exposure differs for {source}")
        if schedule["source_artifacts"][source]["bin_path"] != details["path"]:
            raise RuntimeError(f"Mixed schedule artifact path differs for {source}")


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _save_checkpoint(
    path: Path,
    *,
    model: FictionPulperLM,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    step: int,
    real_validation_loss: float,
    curriculum_validation_loss: float,
    schedule_sha256: str,
    source_valid_targets_observed: dict[str, int],
    metrics: list[dict[str, Any]],
    tokenizer_hash: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": {
            "step": step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "policy": "linear warmup then cosine decay",
            "max_steps": int(config["training"]["max_steps"]),
            "warmup_steps": int(config["training"]["warmup_steps"]),
            "maximum_learning_rate": float(config["training"]["learning_rate"]),
            "minimum_learning_rate": float(config["training"]["min_learning_rate"]),
        },
        "rng_state": capture_rng_state(),
        "model_config": asdict(model.config),
        "config": config,
        "step": step,
        "real_validation_loss": real_validation_loss,
        "curriculum_validation_loss": curriculum_validation_loss,
        "selection_metric": "Data30M validation loss only",
        "schedule_sha256": schedule_sha256,
        "source_valid_targets_observed": source_valid_targets_observed,
        "metrics": metrics,
        "tokenizer_hash": tokenizer_hash or config.get("tokenizer", {}).get("expected_sha256"),
        "model_initialization_seed": LOCKED_SEED,
        "fresh_random_initialization": True,
        "resume_checkpoint": None,
    }
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run_training(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_locked_config(config)
    if config["training"]["require_clean_worktree"] and tracked_worktree_dirty() is not False:
        raise RuntimeError("Locked training requires a clean tracked Git worktree")
    validate_runtime_environment()
    torch.set_float32_matmul_precision("high")
    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device("cuda")

    tokenizer_path = Path(config["tokenizer"]["path"])
    tokenizer_hash = sha256_file(tokenizer_path)
    if tokenizer_hash != config["tokenizer"]["expected_sha256"]:
        raise RuntimeError("Locked tokenizer hash changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != LOCKED_MODEL["vocab_size"]:
        raise RuntimeError("Tokenizer and model vocabulary sizes disagree")
    pad_token_id = tokenizer.token_to_id("<|pad|>")
    if pad_token_id is None:
        raise RuntimeError("Tokenizer has no pad token")
    real_artifacts = validate_real_artifacts(config, tokenizer_hash)

    schedule_path = Path(config["mixed_data"]["schedule"]["path"])
    schedule_file_sha256 = sha256_file(schedule_path)
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule_sha256 = verify_schedule(schedule)
    _validate_schedule_contract(config, schedule)
    _verify_schedule_sources(schedule)
    real_schedule_artifact = schedule["source_artifacts"]["real"]
    if real_schedule_artifact["bin_sha256"] != config["data"]["expected_train_sha256"]:
        raise RuntimeError("Schedule does not reference the locked real training binary")

    sequence_length = int(config["model"]["max_seq_len"])
    sources = {
        name: _packed_dataset(details["path"], sequence_length, pad_token_id)
        for name, details in config["mixed_data"]["training_sources"].items()
    }
    train_dataset = ScheduledMixedDataset(sources, schedule)
    real_validation = _packed_dataset(config["data"]["validation_path"], sequence_length, pad_token_id)
    curriculum_validation = _packed_dataset(
        config["mixed_data"]["curriculum_validation_path"], sequence_length, pad_token_id
    )

    model_config = model_config_from_dict(config["model"])
    model = FictionPulperLM(model_config).to(device)
    parameter_count = model.trainable_parameter_count()
    if parameter_count != LOCKED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Locked parameter count changed: {parameter_count:,}, expected {LOCKED_PARAMETER_COUNT:,}"
        )
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        fused=True,
    )
    run_name = run_id or config["experiment_id"]
    run_dir = Path(config["logging"]["run_dir"]) / run_name
    checkpoint_dir = Path(training["checkpoint_dir"])
    if run_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Run or checkpoint directory already exists; resume is forbidden")
    run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    copied_config = run_dir / "config.yaml"
    copied_schedule = run_dir / "mixed-source-schedule.json"
    shutil.copy2(config_path, copied_config)
    shutil.copy2(schedule_path, copied_schedule)
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    manifest = {
        "experiment_id": config["experiment_id"],
        "started_at": datetime.now().isoformat(),
        "training_git_commit": git_commit(),
        "seed": seed,
        "model_initialization_seed": seed,
        "fresh_random_initialization": True,
        "prior_checkpoint_loaded": False,
        "resume_checkpoint": None,
        "model_parameter_count": parameter_count,
        "model": LOCKED_MODEL,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "copied_config_path": str(copied_config),
        "copied_config_sha256": sha256_file(copied_config),
        "schedule_path": str(schedule_path),
        "schedule_content_sha256": schedule_sha256,
        "schedule_file_sha256": schedule_file_sha256,
        "copied_schedule_path": str(copied_schedule),
        "copied_schedule_sha256": sha256_file(copied_schedule),
        "real_artifacts": real_artifacts,
        "exposure": {
            "total_valid_targets": schedule["total_valid_targets"],
            "sources": schedule["sources"],
        },
        "checkpoint_selection": "Data30M validation loss only at steps 444, 888, 1332, 1776, 2220",
        "curriculum_validation_role": "diagnostic only",
    }
    write_json_atomic(run_dir / "manifest.json", manifest)

    loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    accumulation_steps = int(training["gradient_accumulation_steps"])
    checkpoint_steps = {int(step) for step in training["checkpoint_steps"]}
    global_step = 0
    total_valid_targets = 0
    total_source_targets = {name: 0 for name in sources}
    interval_valid_targets = 0
    interval_source_targets = {name: 0 for name in sources}
    interval_allocated_slots = 0
    interval_samples = 0
    interval_loss_sum = 0.0
    interval_gradient_norm_sum = 0.0
    interval_gradient_norm_max = 0.0
    interval_gradient_norm_count = 0
    total_training_seconds = 0.0
    metrics: list[dict[str, Any]] = []
    best_real_validation_loss = math.inf
    best_step = 0
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    wall_started = time.monotonic()

    try:
        initial_real = evaluate(
            model,
            real_validation,
            batch_size=int(training["batch_size"]),
            device=device,
            use_bf16=True,
        )
        initial_curriculum = evaluate(
            model,
            curriculum_validation,
            batch_size=int(training["batch_size"]),
            device=device,
            use_bf16=True,
        )
    except Exception:
        writer.close()
        raise
    metrics.append(
        {
            "step": 0,
            "selection_eligible": False,
            "data30m_validation": initial_real,
            "curriculum_validation_diagnostic": initial_curriculum,
            "elapsed_seconds": time.monotonic() - wall_started,
        }
    )
    write_json_atomic(run_dir / "metrics.json", {"checkpoints": metrics})
    interval_started = time.monotonic()
    try:
        for microbatch_index, (input_ids, labels) in enumerate(loader, start=1):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            valid_targets = int(labels.ne(-100).sum().item())
            schedule_start = (microbatch_index - 1) * int(training["batch_size"])
            scheduled_entries = schedule["entries"][
                schedule_start : schedule_start + int(input_ids.shape[0])
            ]
            scheduled_targets = sum(int(entry["valid_targets"]) for entry in scheduled_entries)
            if valid_targets != scheduled_targets:
                raise RuntimeError("Observed microbatch targets differ from the exact schedule")
            for entry in scheduled_entries:
                source = str(entry["source"])
                count = int(entry["valid_targets"])
                total_source_targets[source] += count
                interval_source_targets[source] += count
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(input_ids, labels)
            if loss is None or not math.isfinite(float(loss.item())):
                raise FloatingPointError("Training loss is NaN or Inf")
            interval_loss_sum += float(loss.item()) * valid_targets
            interval_valid_targets += valid_targets
            total_valid_targets += valid_targets
            interval_allocated_slots += labels.numel()
            interval_samples += int(input_ids.shape[0])
            (loss / accumulation_steps).backward()
            if microbatch_index % accumulation_steps:
                continue
            global_step += 1
            lr = learning_rate_for_step(
                global_step,
                max_steps=int(training["max_steps"]),
                warmup_steps=int(training["warmup_steps"]),
                learning_rate=float(training["learning_rate"]),
                min_learning_rate=float(training["min_learning_rate"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["grad_clip"])
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Gradient norm is NaN or Inf")
            gradient_norm_value = float(gradient_norm.item())
            interval_gradient_norm_sum += gradient_norm_value
            interval_gradient_norm_max = max(interval_gradient_norm_max, gradient_norm_value)
            interval_gradient_norm_count += 1
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if global_step not in checkpoint_steps:
                continue

            torch.cuda.synchronize(device)
            interval_training_seconds = time.monotonic() - interval_started
            total_training_seconds += interval_training_seconds
            real_metrics = evaluate(
                model,
                real_validation,
                batch_size=int(training["batch_size"]),
                device=device,
                use_bf16=True,
            )
            curriculum_metrics = evaluate(
                model,
                curriculum_validation,
                batch_size=int(training["batch_size"]),
                device=device,
                use_bf16=True,
            )
            metric = {
                "step": global_step,
                "epoch_equivalent": global_step // 444,
                "selection_eligible": True,
                "train_loss": interval_loss_sum / interval_valid_targets,
                "interval_valid_targets": interval_valid_targets,
                "interval_source_valid_targets": dict(interval_source_targets),
                "cumulative_valid_targets": total_valid_targets,
                "cumulative_source_valid_targets": dict(total_source_targets),
                "learning_rate": lr,
                "data30m_validation": real_metrics,
                "curriculum_validation_diagnostic": curriculum_metrics,
                "gradient_norm_mean": interval_gradient_norm_sum / interval_gradient_norm_count,
                "gradient_norm_max": interval_gradient_norm_max,
                "training_seconds": interval_training_seconds,
                "valid_targets_per_second": interval_valid_targets / interval_training_seconds,
                "allocated_slots_per_second": interval_allocated_slots / interval_training_seconds,
                "samples_per_second": interval_samples / interval_training_seconds,
                "gpu_memory_current_gb": torch.cuda.memory_allocated(device) / 1024**3,
                "gpu_memory_peak_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                "elapsed_seconds": time.monotonic() - wall_started,
            }
            metrics.append(metric)
            for name, value in (
                ("train/loss", metric["train_loss"]),
                ("train/learning_rate", lr),
                ("train/valid_targets_per_second", metric["valid_targets_per_second"]),
                ("train/gradient_norm_mean", metric["gradient_norm_mean"]),
                ("train/gpu_memory_gb", metric["gpu_memory_current_gb"]),
                ("validation/data30m_loss", real_metrics["loss"]),
                ("diagnostic/curriculum_validation_loss", curriculum_metrics["loss"]),
            ):
                writer.add_scalar(name, value, global_step)
            for source, count in total_source_targets.items():
                writer.add_scalar(f"train/{source}_valid_targets", count, global_step)
            write_json_atomic(run_dir / "metrics.json", {"checkpoints": metrics})
            checkpoint_args = {
                "model": model,
                "optimizer": optimizer,
                "config": config,
                "step": global_step,
                "real_validation_loss": float(real_metrics["loss"]),
                "curriculum_validation_loss": float(curriculum_metrics["loss"]),
                "schedule_sha256": schedule_sha256,
                "source_valid_targets_observed": dict(total_source_targets),
                "metrics": metrics,
                "tokenizer_hash": tokenizer_hash,
            }
            _save_checkpoint(checkpoint_dir / "latest.pt", **checkpoint_args)
            _save_checkpoint(checkpoint_dir / f"step-{global_step:04d}.pt", **checkpoint_args)
            if float(real_metrics["loss"]) < best_real_validation_loss:
                best_real_validation_loss = float(real_metrics["loss"])
                best_step = global_step
                _save_checkpoint(checkpoint_dir / "best-validation.pt", **checkpoint_args)
            interval_valid_targets = 0
            interval_source_targets = {name: 0 for name in sources}
            interval_allocated_slots = 0
            interval_samples = 0
            interval_loss_sum = 0.0
            interval_gradient_norm_sum = 0.0
            interval_gradient_norm_max = 0.0
            interval_gradient_norm_count = 0
            interval_started = time.monotonic()

        if global_step != LOCKED_STEPS:
            raise RuntimeError(f"Training ended at step {global_step}, expected {LOCKED_STEPS}")
        expected_source_targets = {
            name: int(details["valid_targets"]) for name, details in schedule["sources"].items()
        }
        if total_valid_targets != LOCKED_TOTAL_VALID_TARGETS or total_source_targets != expected_source_targets:
            raise RuntimeError("Observed source exposure differs from the exact schedule")
        best_checkpoint_path = checkpoint_dir / "best-validation.pt"
        completed_at = datetime.now().isoformat()
        summary = {
            "status": "training_and_selection_complete",
            "training_git_commit": manifest["training_git_commit"],
            "started_at": manifest["started_at"],
            "completed_at": completed_at,
            "optimizer_steps": global_step,
            "total_valid_targets": total_valid_targets,
            "source_valid_targets_observed": total_source_targets,
            "source_valid_target_percentages": {
                name: 100.0 * count / total_valid_targets
                for name, count in total_source_targets.items()
            },
            "best_data30m_validation_loss": best_real_validation_loss,
            "best_checkpoint_step": best_step,
            "best_checkpoint": str(best_checkpoint_path),
            "best_checkpoint_sha256": sha256_file(best_checkpoint_path),
            "selection_metric": "Data30M validation loss only",
            "curriculum_validation_used_for_selection": False,
            "schedule_content_sha256": schedule_sha256,
            "schedule_file_sha256": schedule_file_sha256,
            "training_seconds": total_training_seconds,
            "total_elapsed_seconds": time.monotonic() - wall_started,
            "average_valid_targets_per_second": total_valid_targets / total_training_seconds,
            "gpu_memory_current_gb": torch.cuda.memory_allocated(device) / 1024**3,
            "gpu_memory_peak_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
            "metrics": metrics,
        }
        write_json_atomic(run_dir / "summary.json", summary)
        manifest["completed_at"] = completed_at
        manifest["best_checkpoint_step"] = best_step
        manifest["best_checkpoint_sha256"] = summary["best_checkpoint_sha256"]
        manifest["source_valid_targets_observed"] = total_source_targets
        write_json_atomic(run_dir / "manifest.json", manifest)
        return summary
    except Exception as error:
        write_json_atomic(
            run_dir / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "optimizer_step": global_step,
                "source_valid_targets_observed": total_source_targets,
            },
        )
        raise
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(json.dumps(run_training(args.config, args.run_id), indent=2))


if __name__ == "__main__":
    main()
