"""Train and evaluate a locked FictionPulper-5M experiment configuration."""

from __future__ import annotations

import argparse
import json
import math
import os
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
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import PackedStoryDataset
from src.model import FictionPulperLM, ModelConfig, model_config_from_dict
from src.train_tokenizer import sha256_file, write_json_atomic


FIXED_PROMPTS = [
    "The man in the gray coat entered the station just before midnight.",
    "There was something moving beneath the floorboards.",
    "The rocket had been silent for three days when the signal arrived.",
    "Nobody in Dry Creek had seen the stranger before Tuesday.",
    "Detective Mallory looked at the locked door and knew somebody was lying.",
]

GENRE_PROMPTS = {
    "crime_noir": "The revolver on Mallory's desk had been fired twice, but the dead man carried three bullets.",
    "horror": "At midnight the portrait opened its eyes and whispered Clara's name.",
    "science_fiction": "The last transmission from Europa arrived nine years after the colony vanished.",
    "western_adventure": "By sundown, the stranger had crossed the desert and reached the abandoned mining camp.",
    "fantasy_weird": "Beyond the crooked forest stood a tower that cast no shadow.",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def learning_rate_for_step(
    step: int,
    *,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float,
    min_learning_rate: float,
) -> float:
    if not 1 <= step <= max_steps:
        raise ValueError("step must be in [1, max_steps]")
    if step <= warmup_steps:
        return learning_rate * step / warmup_steps
    decay_progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_learning_rate + coefficient * (learning_rate - min_learning_rate)


def make_dataset(
    config: dict[str, Any], split: str, pad_token_id: int
) -> PackedStoryDataset:
    path_key = {"train": "train_path", "validation": "validation_path", "test": "test_path"}[
        split
    ]
    bin_path = Path(config["data"][path_key])
    return PackedStoryDataset(
        bin_path,
        bin_path.with_suffix(".index.json"),
        sequence_length=int(config["model"]["max_seq_len"]),
        pad_token_id=pad_token_id,
    )


def make_dataset_from_path(
    bin_path: str | Path,
    *,
    sequence_length: int,
    pad_token_id: int,
    exclude_document_ids: set[str] | None = None,
) -> PackedStoryDataset:
    path = Path(bin_path)
    return PackedStoryDataset(
        path,
        path.with_suffix(".index.json"),
        sequence_length=sequence_length,
        pad_token_id=pad_token_id,
        exclude_document_ids=exclude_document_ids,
    )


@torch.no_grad()
def evaluate(
    model: FictionPulperLM,
    dataset: PackedStoryDataset,
    *,
    batch_size: int,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, float | int]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    loss_sum = 0.0
    valid_tokens = 0
    correct = 0
    allocated_slots = 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits, _ = model(input_ids)
        valid_mask = labels.ne(-100)
        loss_sum += F.cross_entropy(
            logits.float().reshape(-1, model.config.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ).item()
        valid_tokens += valid_mask.sum().item()
        correct += ((logits.argmax(dim=-1) == labels) & valid_mask).sum().item()
        allocated_slots += labels.numel()
    model.train()
    loss = loss_sum / valid_tokens
    if not math.isfinite(loss):
        raise FloatingPointError("Evaluation loss is NaN or Inf")
    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "next_token_accuracy": correct / valid_tokens,
        "valid_tokens": valid_tokens,
        "allocated_slots": allocated_slots,
    }


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator,
) -> int:
    logits = logits.float() / temperature
    if top_k > 0:
        threshold = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < threshold, -torch.inf)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    sampled_index = torch.multinomial(probabilities, 1, generator=generator)
    return int(sorted_indices[sampled_index].item())


@torch.no_grad()
def generate(
    model: FictionPulperLM,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    device: torch.device,
    use_bf16: bool,
    temperature: float | None = None,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 0,
) -> str:
    model.eval()
    prefix = [tokenizer.token_to_id("<|story|>"), tokenizer.token_to_id("<|bos|>")]
    context = torch.tensor(
        [prefix + tokenizer.encode(prompt).ids], dtype=torch.long, device=device
    )
    eos_token_id = tokenizer.token_to_id("<|eos|>")
    generated: list[int] = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(max_new_tokens):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits, _ = model(context[:, -model.config.max_seq_len :])
        next_logits = logits[0, -1]
        if temperature is None:
            next_token = int(next_logits.argmax().item())
        else:
            next_token = sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )
        generated.append(next_token)
        context = torch.cat(
            [context, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1
        )
        if next_token == eos_token_id:
            break
    model.train()
    return tokenizer.decode(generated, skip_special_tokens=True)


def generation_snapshot(
    model: FictionPulperLM,
    tokenizer: Tokenizer,
    *,
    epoch: int,
    step: int,
    prompts: list[str],
    output_path: Path,
    device: torch.device,
    use_bf16: bool,
    seed: int,
) -> dict[str, Any]:
    entries = []
    for prompt_index, prompt in enumerate(prompts):
        entries.append(
            {
                "prompt": prompt,
                "greedy": generate(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=128,
                    device=device,
                    use_bf16=use_bf16,
                ),
                "sampled": generate(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=256,
                    device=device,
                    use_bf16=use_bf16,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=50,
                    seed=seed + prompt_index,
                ),
            }
        )
    snapshot = {
        "epoch": epoch,
        "optimizer_step": step,
        "sample_seed": seed,
        "entries": entries,
    }
    write_json_atomic(output_path, snapshot)
    return snapshot


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def save_checkpoint(
    path: Path,
    *,
    model: FictionPulperLM,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    config: dict[str, Any],
    epoch: int,
    step: int,
    validation_loss: float,
    best_validation_loss: float,
    tokenizer_hash: str,
    metrics: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": {"step": step, "learning_rate": optimizer.param_groups[0]["lr"]},
        "model_config": asdict(model_config),
        "config": config,
        "epoch": epoch,
        "step": step,
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation_loss,
        "tokenizer_hash": tokenizer_hash,
        "rng_state": capture_rng_state(),
        "metrics": metrics,
    }
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_worktree_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def write_comparison_artifacts(
    *,
    run_dir: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    final_generations: dict[str, Any],
    packed_metadata: dict[str, Any],
) -> dict[str, Any]:
    comparison_config = config.get("comparison", {})
    smoke_summary = json.loads(
        Path(comparison_config["smoke_run_summary_path"]).read_text(encoding="utf-8")
    )
    smoke_generations = json.loads(
        Path(comparison_config["smoke_generations_path"]).read_text(encoding="utf-8")
    )
    smoke_by_prompt = {entry["prompt"]: entry for entry in smoke_generations["entries"]}
    generation_entries = []
    for entry in final_generations["entries"]:
        original = smoke_by_prompt[entry["prompt"]]
        generation_entries.append(
            {
                "prompt": entry["prompt"],
                "original_1_48m_model": {
                    "greedy": original["greedy"],
                    "sampled": original["sampled"],
                },
                "corpus_v1_10m_model": {
                    "greedy": entry["greedy"],
                    "sampled": entry["sampled"],
                },
            }
        )
    generation_comparison = {
        "prompts_unchanged": True,
        "greedy_max_new_tokens": 128,
        "sampled_max_new_tokens": 256,
        "sampling_temperature": 0.8,
        "sampling_top_k": 50,
        "sampling_top_p": 0.95,
        "sampling_seed": final_generations["sample_seed"],
        "entries": generation_entries,
    }
    write_json_atomic(run_dir / "generation-comparison.json", generation_comparison)
    generation_lines = ["# Generation Comparison", ""]
    for entry in generation_entries:
        generation_lines.extend(
            [
                f"## {entry['prompt']}",
                "",
                "### Original 1.48M-token model (greedy)",
                "",
                entry["original_1_48m_model"]["greedy"],
                "",
                "### Corpus-v1 ~10M-token model (greedy)",
                "",
                entry["corpus_v1_10m_model"]["greedy"],
                "",
                "### Original 1.48M-token model (sampled)",
                "",
                entry["original_1_48m_model"]["sampled"],
                "",
                "### Corpus-v1 ~10M-token model (sampled)",
                "",
                entry["corpus_v1_10m_model"]["sampled"],
                "",
            ]
        )
    (run_dir / "generation-comparison.md").write_text(
        "\n".join(generation_lines), encoding="utf-8"
    )

    smoke_best_metric = next(
        metric
        for metric in smoke_summary["metrics"]
        if metric["epoch"] == smoke_summary["best_validation_epoch"]
    )
    quantitative = {
        "parameters": {
            "smoke_v1": smoke_summary["model_parameter_count"],
            "data10m_v1": summary["model_parameter_count"],
        },
        "tokenizer_sha256": {
            "smoke_v1": packed_metadata["tokenizer_hash"],
            "data10m_v1": packed_metadata["tokenizer_hash"],
        },
        "unique_train_tokens": {
            "smoke_v1": json.loads(
                Path(comparison_config["smoke_run_manifest_path"]).read_text(encoding="utf-8")
            )["packed_metadata"]["splits"]["train"]["token_count"],
            "data10m_v1": packed_metadata["splits"]["train"]["token_count"],
        },
        "train_corpus_stories": {
            "smoke_v1": 604,
            "data10m_v1": packed_metadata["splits"]["train"]["document_count"],
        },
        "best_validation_loss": {
            "smoke_v1": smoke_summary["best_validation_loss"],
            "data10m_v1": summary["best_validation_loss"],
        },
        "validation_perplexity": {
            "smoke_v1": smoke_summary["best_validation_perplexity"],
            "data10m_v1": summary["best_validation_perplexity"],
        },
        "legacy_validation_loss": {
            "smoke_v1": smoke_summary["best_validation_loss"],
            "data10m_v1": summary["legacy_seed_validation"]["loss"],
        },
        "legacy_test_loss": {
            "smoke_v1": smoke_summary["test_loss"],
            "data10m_v1": summary["legacy_seed_test"]["loss"],
        },
        "legacy_test_perplexity": {
            "smoke_v1": smoke_summary["test_perplexity"],
            "data10m_v1": summary["legacy_seed_test"]["perplexity"],
        },
        "legacy_test_accuracy": {
            "smoke_v1": smoke_summary["test_next_token_accuracy"],
            "data10m_v1": summary["legacy_seed_test"]["next_token_accuracy"],
        },
        "peak_vram_gb": {
            "smoke_v1": smoke_summary["peak_gpu_memory_gb"],
            "data10m_v1": summary["peak_gpu_memory_gb"],
        },
        "valid_tokens_per_second": {
            "smoke_v1": smoke_summary["average_valid_tokens_per_second"],
            "data10m_v1": summary["average_valid_tokens_per_second"],
        },
        "training_runtime_seconds": {
            "smoke_v1": smoke_summary["training_seconds"],
            "data10m_v1": summary["training_seconds"],
        },
        "smoke_best_validation_accuracy": smoke_best_metric[
            "validation_next_token_accuracy"
        ],
    }
    write_json_atomic(run_dir / "quantitative-comparison.json", quantitative)
    rows = [
        ("Parameters", "parameters"),
        ("Tokenizer SHA-256", "tokenizer_sha256"),
        ("Unique train tokens", "unique_train_tokens"),
        ("Train corpus stories", "train_corpus_stories"),
        ("Best validation loss", "best_validation_loss"),
        ("Validation perplexity", "validation_perplexity"),
        ("Legacy validation loss", "legacy_validation_loss"),
        ("Legacy test loss", "legacy_test_loss"),
        ("Legacy test perplexity", "legacy_test_perplexity"),
        ("Legacy test accuracy", "legacy_test_accuracy"),
        ("Peak VRAM GiB", "peak_vram_gb"),
        ("Valid tokens/sec", "valid_tokens_per_second"),
        ("Training runtime sec", "training_runtime_seconds"),
    ]
    table_lines = [
        "# Quantitative Comparison",
        "",
        "| Metric | Smoke v1 | Data10M v1 |",
        "|---|---:|---:|",
    ]
    for label, key in rows:
        table_lines.append(
            f"| {label} | {quantitative[key]['smoke_v1']} | {quantitative[key]['data10m_v1']} |"
        )
    (run_dir / "quantitative-comparison.md").write_text(
        "\n".join(table_lines) + "\n", encoding="utf-8"
    )
    return {"generation": generation_comparison, "quantitative": quantitative}


def run_training(config_path: Path, run_id: str | None = None) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training = config["training"]
    seed = int(config["seed"])
    set_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("The locked smoke run requires CUDA")
    device = torch.device("cuda")
    use_bf16 = training["precision"] == "bf16" and torch.cuda.is_bf16_supported()
    if not use_bf16:
        raise RuntimeError("The locked smoke run requires BF16 support")
    torch.set_float32_matmul_precision("high")

    run_id = run_id or datetime.now().strftime("smoke-5m-%Y%m%d-%H%M%S")
    run_dir = Path("runs") / run_id
    generation_dir = run_dir / "generations"
    generation_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "config.yaml")

    tokenizer_path = Path(config["tokenizer"]["path"])
    tokenizer_hash = sha256_file(tokenizer_path)
    expected_tokenizer_hash = config["tokenizer"].get("expected_sha256")
    if expected_tokenizer_hash and tokenizer_hash != expected_tokenizer_hash:
        raise RuntimeError(
            f"Tokenizer hash changed: {tokenizer_hash}, expected {expected_tokenizer_hash}"
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model_config = model_config_from_dict(config["model"])
    if tokenizer.get_vocab_size() != model_config.vocab_size:
        raise RuntimeError("Tokenizer and model vocabulary sizes disagree")
    pad_token_id = tokenizer.token_to_id("<|pad|>")
    train_dataset = make_dataset(config, "train", pad_token_id)
    validation_dataset = make_dataset(config, "validation", pad_token_id)
    packed_metadata = json.loads(
        Path(config["data"]["packed_metadata_path"]).read_text(encoding="utf-8")
    )
    expected_packed_hash = config["data"].get("expected_packed_metadata_sha256")
    if expected_packed_hash and sha256_file(
        Path(config["data"]["packed_metadata_path"])
    ) != expected_packed_hash:
        raise RuntimeError("Locked packed metadata hash changed")
    for label, metadata_key, path_key in (
        ("corpus", "corpus_sha256", "corpus_path"),
        ("split", "split_sha256", "split_path"),
    ):
        expected_hash = config["data"].get(f"expected_{label}_sha256")
        if expected_hash and packed_metadata[metadata_key] != expected_hash:
            raise RuntimeError(f"Packed data {label} hash does not match the locked hash")
        if expected_hash and sha256_file(Path(config["data"][path_key])) != expected_hash:
            raise RuntimeError(f"Locked {label} artifact changed")
    if packed_metadata["tokenizer_hash"] != tokenizer_hash:
        raise RuntimeError("Packed data tokenizer hash does not match the configured tokenizer")
    expected_steps_per_epoch = packed_metadata["schedule"]["optimizer_steps_per_epoch"]
    if int(training["epochs"]) * expected_steps_per_epoch != int(training["max_steps"]):
        raise RuntimeError("Configured epochs and max_steps disagree with packed schedule")

    configured_checkpoint_dir = training.get("checkpoint_dir")
    checkpoint_dir = Path(configured_checkpoint_dir or "checkpoints")
    if configured_checkpoint_dir and checkpoint_dir.exists():
        raise FileExistsError(f"Checkpoint directory already exists: {checkpoint_dir}")
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir()
    for artifact_key in (
        "corpus_stats_path",
        "split_path",
    ):
        artifact_path = config["data"].get(artifact_key)
        if artifact_path:
            shutil.copy2(artifact_path, artifact_dir / Path(artifact_path).name)
    resolution_path = config.get("comparison", {}).get("near_duplicate_resolution_path")
    if resolution_path:
        shutil.copy2(resolution_path, artifact_dir / Path(resolution_path).name)

    manifest = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
        "git_commit": git_commit(),
        "git_worktree_dirty": git_worktree_dirty(),
        "seed": seed,
        "fresh_random_initialization": True,
        "tiny_overfit_checkpoint_used": False,
        "model_parameter_count": None,
        "tokenizer_hash": tokenizer_hash,
        "packed_metadata": packed_metadata,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "corpus_sha256": packed_metadata["corpus_sha256"],
        "split_manifest_sha256": packed_metadata["split_sha256"],
        "checkpoint_dir": str(checkpoint_dir),
    }
    write_json_atomic(run_dir / "manifest.json", manifest)

    model = FictionPulperLM(model_config).to(device)
    parameter_count = model.trainable_parameter_count()
    expected_parameter_count = int(config["model"].get("expected_parameter_count", 5_426_432))
    if parameter_count != expected_parameter_count:
        raise RuntimeError(
            f"Locked parameter count changed: {parameter_count}, "
            f"expected {expected_parameter_count}"
        )
    manifest["model_parameter_count"] = parameter_count
    write_json_atomic(run_dir / "manifest.json", manifest)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        fused=True,
    )
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    batch_size = int(training["batch_size"])
    accumulation_steps = int(training["gradient_accumulation_steps"])
    max_steps = int(training["max_steps"])
    warmup_steps = int(training["warmup_steps"])
    maximum_lr = float(training["learning_rate"])
    minimum_lr = float(training["min_learning_rate"])
    sample_seed = seed + 10_000
    metrics: list[dict[str, Any]] = []
    total_valid_tokens = 0
    total_allocated_slots = 0
    total_training_seconds = 0.0
    global_step = 0
    best_validation_loss = math.inf
    best_epoch = 0
    torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.monotonic()

    try:
        initial_validation = evaluate(
            model,
            validation_dataset,
            batch_size=batch_size,
            device=device,
            use_bf16=use_bf16,
        )
        initial_metric = {
            "epoch": 0,
            "optimizer_step": 0,
            "training_loss": None,
            "validation_loss": initial_validation["loss"],
            "validation_perplexity": initial_validation["perplexity"],
            "learning_rate": 0.0,
            "valid_tokens_per_second": None,
            "allocated_slots_per_second": None,
            "gpu_memory_current_gb": torch.cuda.memory_allocated(device) / 1024**3,
            "gpu_memory_peak_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
            "elapsed_seconds": time.monotonic() - wall_start,
        }
        metrics.append(initial_metric)
        generation_snapshot(
            model,
            tokenizer,
            epoch=0,
            step=0,
            prompts=FIXED_PROMPTS,
            output_path=generation_dir / "epoch-00.json",
            device=device,
            use_bf16=use_bf16,
            seed=sample_seed,
        )
        best_validation_loss = float(initial_validation["loss"])
        save_checkpoint(
            checkpoint_dir / "best-validation.pt",
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            config=config,
            epoch=0,
            step=0,
            validation_loss=best_validation_loss,
            best_validation_loss=best_validation_loss,
            tokenizer_hash=tokenizer_hash,
            metrics=metrics,
        )
        print(
            f"epoch=0 step=0 val_loss={initial_validation['loss']:.4f} "
            f"val_ppl={initial_validation['perplexity']:.2f} parameters={parameter_count:,}"
        )

        for epoch in range(1, int(training["epochs"]) + 1):
            generator = torch.Generator().manual_seed(seed + epoch)
            loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
                num_workers=0,
                pin_memory=True,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss_sum = 0.0
            epoch_valid_tokens = 0
            epoch_allocated_slots = 0
            epoch_gradient_norm_sum = 0.0
            epoch_gradient_norm_max = 0.0
            epoch_gradient_norm_count = 0
            accumulation_count = 0
            epoch_start = time.monotonic()
            for batch_index, (input_ids, labels) in enumerate(loader, start=1):
                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                valid_count = labels.ne(-100).sum().item()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(input_ids, labels)
                if loss is None or not math.isfinite(loss.item()):
                    raise FloatingPointError("Training loss is NaN or Inf")
                epoch_loss_sum += loss.item() * valid_count
                epoch_valid_tokens += valid_count
                epoch_allocated_slots += labels.numel()
                (loss / accumulation_steps).backward()
                accumulation_count += 1

                should_step = accumulation_count == accumulation_steps or batch_index == len(loader)
                if not should_step:
                    continue
                if accumulation_count != accumulation_steps:
                    correction = accumulation_steps / accumulation_count
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                next_step = global_step + 1
                lr = learning_rate_for_step(
                    next_step,
                    max_steps=max_steps,
                    warmup_steps=warmup_steps,
                    learning_rate=maximum_lr,
                    min_learning_rate=minimum_lr,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["grad_clip"])
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("Gradient norm is NaN or Inf")
                gradient_norm_value = float(gradient_norm.item())
                epoch_gradient_norm_sum += gradient_norm_value
                epoch_gradient_norm_max = max(epoch_gradient_norm_max, gradient_norm_value)
                epoch_gradient_norm_count += 1
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                global_step = next_step

            torch.cuda.synchronize(device)
            epoch_training_seconds = time.monotonic() - epoch_start
            total_training_seconds += epoch_training_seconds
            total_valid_tokens += epoch_valid_tokens
            total_allocated_slots += epoch_allocated_slots
            if global_step != epoch * expected_steps_per_epoch:
                raise RuntimeError("Observed optimizer steps differ from packed schedule")

            validation = evaluate(
                model,
                validation_dataset,
                batch_size=batch_size,
                device=device,
                use_bf16=use_bf16,
            )
            epoch_training_loss = epoch_loss_sum / epoch_valid_tokens
            metric = {
                "epoch": epoch,
                "optimizer_step": global_step,
                "training_loss": epoch_training_loss,
                "validation_loss": validation["loss"],
                "validation_perplexity": validation["perplexity"],
                "validation_next_token_accuracy": validation["next_token_accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "valid_tokens": epoch_valid_tokens,
                "allocated_slots": epoch_allocated_slots,
                "valid_tokens_per_second": epoch_valid_tokens / epoch_training_seconds,
                "allocated_slots_per_second": epoch_allocated_slots / epoch_training_seconds,
                "gradient_norm_mean": epoch_gradient_norm_sum / epoch_gradient_norm_count,
                "gradient_norm_max": epoch_gradient_norm_max,
                "gpu_memory_current_gb": torch.cuda.memory_allocated(device) / 1024**3,
                "gpu_memory_peak_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                "epoch_training_seconds": epoch_training_seconds,
                "elapsed_seconds": time.monotonic() - wall_start,
            }
            metrics.append(metric)
            writer.add_scalar("train/loss", epoch_training_loss, global_step)
            writer.add_scalar("validation/loss", validation["loss"], global_step)
            writer.add_scalar("validation/perplexity", validation["perplexity"], global_step)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("train/tokens_per_second", metric["valid_tokens_per_second"], global_step)
            writer.add_scalar("train/allocated_slots_per_second", metric["allocated_slots_per_second"], global_step)
            writer.add_scalar("train/gpu_memory_gb", metric["gpu_memory_current_gb"], global_step)
            writer.add_scalar("train/gradient_norm_mean", metric["gradient_norm_mean"], global_step)
            writer.add_scalar("train/gradient_norm_max", metric["gradient_norm_max"], global_step)
            write_json_atomic(run_dir / "metrics.json", {"epochs": metrics})
            generation_snapshot(
                model,
                tokenizer,
                epoch=epoch,
                step=global_step,
                prompts=FIXED_PROMPTS,
                output_path=generation_dir / f"epoch-{epoch:02d}.json",
                device=device,
                use_bf16=use_bf16,
                seed=sample_seed,
            )

            improved = float(validation["loss"]) < best_validation_loss
            if improved:
                best_validation_loss = float(validation["loss"])
                best_epoch = epoch
            checkpoint_arguments = {
                "model": model,
                "optimizer": optimizer,
                "model_config": model_config,
                "config": config,
                "epoch": epoch,
                "step": global_step,
                "validation_loss": float(validation["loss"]),
                "best_validation_loss": best_validation_loss,
                "tokenizer_hash": tokenizer_hash,
                "metrics": metrics,
            }
            save_checkpoint(checkpoint_dir / "latest.pt", **checkpoint_arguments)
            if improved:
                save_checkpoint(checkpoint_dir / "best-validation.pt", **checkpoint_arguments)
            if epoch in {1, 5, 10, 15, int(training["epochs"])}:
                save_checkpoint(
                    checkpoint_dir / f"epoch-{epoch:02d}.pt", **checkpoint_arguments
                )
            print(
                f"epoch={epoch} step={global_step} train_loss={epoch_training_loss:.4f} "
                f"val_loss={validation['loss']:.4f} val_ppl={validation['perplexity']:.2f} "
                f"lr={optimizer.param_groups[0]['lr']:.6g} "
                f"grad_norm={metric['gradient_norm_mean']:.3f}/{metric['gradient_norm_max']:.3f} "
                f"valid_tok/s={metric['valid_tokens_per_second']:.0f} "
                f"slots/s={metric['allocated_slots_per_second']:.0f}"
            )

        if global_step != max_steps:
            raise RuntimeError(f"Run ended at step {global_step}, expected {max_steps}")

        # The sealed test split is opened only after all epochs and best-checkpoint selection.
        best_checkpoint = torch.load(
            checkpoint_dir / "best-validation.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(best_checkpoint["model"])
        test_dataset = make_dataset(config, "test", pad_token_id)
        test_metrics = evaluate(
            model,
            test_dataset,
            batch_size=batch_size,
            device=device,
            use_bf16=use_bf16,
        )
        legacy_validation_metrics = None
        legacy_validation_clean_metrics = None
        legacy_test_metrics = None
        excluded_validation_ids: set[str] = set()
        if "legacy_validation_path" in config["data"] and "legacy_test_path" in config["data"]:
            legacy_validation_dataset = make_dataset_from_path(
                config["data"]["legacy_validation_path"],
                sequence_length=model_config.max_seq_len,
                pad_token_id=pad_token_id,
            )
            legacy_validation_metrics = evaluate(
                model,
                legacy_validation_dataset,
                batch_size=batch_size,
                device=device,
                use_bf16=use_bf16,
            )
            excluded_validation_ids = set(
                config["data"].get("legacy_validation_excluded_ids", [])
            )
            if excluded_validation_ids:
                legacy_validation_clean_dataset = make_dataset_from_path(
                    config["data"]["legacy_validation_path"],
                    sequence_length=model_config.max_seq_len,
                    pad_token_id=pad_token_id,
                    exclude_document_ids=excluded_validation_ids,
                )
                legacy_validation_clean_metrics = evaluate(
                    model,
                    legacy_validation_clean_dataset,
                    batch_size=batch_size,
                    device=device,
                    use_bf16=use_bf16,
                )
            legacy_test_dataset = make_dataset_from_path(
                config["data"]["legacy_test_path"],
                sequence_length=model_config.max_seq_len,
                pad_token_id=pad_token_id,
            )
            legacy_test_metrics = evaluate(
                model,
                legacy_test_dataset,
                batch_size=batch_size,
                device=device,
                use_bf16=use_bf16,
            )
            sealed_evaluations = {
                "checkpoint_selection_complete": True,
                "selection_metric": "Corpus-v1 validation loss only",
                "corpus_v1_test_evaluation_count": 1,
                "legacy_seed_validation_evaluation_count": 1,
                "legacy_seed_test_evaluation_count": 1,
                "corpus_v1_test": test_metrics,
                "legacy_seed_validation": legacy_validation_metrics,
                "legacy_seed_test": legacy_test_metrics,
            }
            if legacy_validation_clean_metrics is not None:
                sealed_evaluations["legacy_seed_validation_leakage_clean_evaluation_count"] = 1
                sealed_evaluations["legacy_seed_validation_leakage_clean"] = {
                    **legacy_validation_clean_metrics,
                    "excluded_ids": sorted(excluded_validation_ids),
                }
            write_json_atomic(run_dir / "sealed-evaluations.json", sealed_evaluations)
        final_generation_prompts = FIXED_PROMPTS + list(GENRE_PROMPTS.values())
        final_generations = generation_snapshot(
            model,
            tokenizer,
            epoch=int(best_checkpoint["epoch"]),
            step=int(best_checkpoint["step"]),
            prompts=final_generation_prompts,
            output_path=generation_dir / "best-validation-final.json",
            device=device,
            use_bf16=use_bf16,
            seed=sample_seed,
        )
        total_elapsed = time.monotonic() - wall_start
        summary = {
            "run_id": run_id,
            "status": "completed",
            "initial_validation_loss": metrics[0]["validation_loss"],
            "best_validation_loss": best_validation_loss,
            "best_validation_epoch": best_epoch,
            "final_validation_loss": metrics[-1]["validation_loss"],
            "best_validation_perplexity": math.exp(best_validation_loss),
            "test_loss": test_metrics["loss"],
            "test_perplexity": test_metrics["perplexity"],
            "test_next_token_accuracy": test_metrics["next_token_accuracy"],
            "total_valid_training_targets_processed": total_valid_tokens,
            "total_allocated_slots_processed": total_allocated_slots,
            "training_seconds": total_training_seconds,
            "total_elapsed_seconds": total_elapsed,
            "average_valid_tokens_per_second": total_valid_tokens / total_training_seconds,
            "average_allocated_slots_per_second": total_allocated_slots
            / total_training_seconds,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
            "model_parameter_count": parameter_count,
            "best_checkpoint": str(checkpoint_dir / "best-validation.pt"),
            "best_checkpoint_epoch": int(best_checkpoint["epoch"]),
            "best_checkpoint_sha256": sha256_file(checkpoint_dir / "best-validation.pt"),
            "final_checkpoint": str(checkpoint_dir / "latest.pt"),
            "final_checkpoint_sha256": sha256_file(checkpoint_dir / "latest.pt"),
            "metrics": metrics,
            "test_metrics": test_metrics,
            "final_generations": final_generations,
        }
        if legacy_validation_metrics is not None and legacy_test_metrics is not None:
            summary["corpus_v1_test"] = test_metrics
            summary["legacy_seed_validation"] = legacy_validation_metrics
            if legacy_validation_clean_metrics is not None:
                summary["legacy_seed_validation_leakage_clean"] = {
                    **legacy_validation_clean_metrics,
                    "excluded_ids": sorted(excluded_validation_ids),
                }
            summary["legacy_seed_test"] = legacy_test_metrics
        if config.get("comparison", {}).get("write_training_comparison_artifacts", True):
            write_comparison_artifacts(
                run_dir=run_dir,
                config=config,
                summary=summary,
                final_generations=final_generations,
                packed_metadata=packed_metadata,
            )
            summary["comparison_artifacts"] = {
                "generation": str(run_dir / "generation-comparison.json"),
                "quantitative": str(run_dir / "quantitative-comparison.json"),
            }
        write_json_atomic(run_dir / "summary.json", summary)
        manifest["completed_at"] = datetime.now().isoformat()
        manifest["best_checkpoint_epoch"] = int(best_checkpoint["epoch"])
        manifest["summary_path"] = str(run_dir / "summary.json")
        write_json_atomic(run_dir / "manifest.json", manifest)
        return summary
    except Exception as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "epoch_metrics": metrics,
            "optimizer_step": global_step,
            "timestamp": datetime.now().isoformat(),
        }
        write_json_atomic(run_dir / "failure.json", failure)
        raise
    finally:
        writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke-5m.yaml"))
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_training(args.config, args.run_id)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
