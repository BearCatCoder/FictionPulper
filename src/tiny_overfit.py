"""Deliberately memorize a tiny packed subset before any full smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from src.dataset import PackedStoryDataset
from src.model import FictionPulperLM, model_config_from_dict
from src.train_tokenizer import sha256_file, write_json_atomic


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_valid_predictions(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[int, int]:
    valid_mask = labels.ne(-100)
    correct = ((logits.argmax(dim=-1) == labels) & valid_mask).sum().item()
    return correct, valid_mask.sum().item()


@torch.no_grad()
def evaluate_metrics(
    model: FictionPulperLM,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    use_bf16: bool,
) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    valid_tokens = 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits, _ = model(input_ids)
        valid_mask = labels.ne(-100)
        loss_sum += F.cross_entropy(
            logits.float().reshape(-1, model.config.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ).item()
        batch_correct, batch_valid_tokens = count_valid_predictions(logits, labels)
        correct += batch_correct
        valid_tokens += batch_valid_tokens
    model.train()
    return {
        "loss": loss_sum / valid_tokens,
        "next_token_accuracy": correct / valid_tokens,
        "valid_tokens": valid_tokens,
    }


@torch.no_grad()
def compare_greedy_continuation(
    model: FictionPulperLM,
    tokenizer: Tokenizer,
    input_ids: torch.Tensor,
    *,
    prompt_tokens: int,
    generated_tokens: int,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, Any]:
    model.eval()
    non_padding = input_ids[input_ids != tokenizer.token_to_id("<|pad|>")]
    available_generation = min(generated_tokens, len(non_padding) - prompt_tokens)
    context = non_padding[:prompt_tokens].to(device).unsqueeze(0)
    expected = non_padding[prompt_tokens : prompt_tokens + available_generation].tolist()
    generated: list[int] = []
    for _ in range(available_generation):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits, _ = model(context[:, -model.config.max_seq_len :])
        next_token = int(logits[0, -1].argmax().item())
        generated.append(next_token)
        context = torch.cat(
            [context, torch.tensor([[next_token]], device=device, dtype=torch.long)], dim=1
        )
    model.train()
    matches = sum(actual == target for actual, target in zip(generated, expected, strict=True))
    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": available_generation,
        "exact_token_matches": matches,
        "exact_token_match_percentage": 100.0 * matches / max(1, available_generation),
        "prompt_text": tokenizer.decode(non_padding[:prompt_tokens].tolist()),
        "expected_text": tokenizer.decode(expected),
        "generated_text": tokenizer.decode(generated),
        "generated_token_ids": generated,
    }


def save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def run_tiny_overfit(
    config_path: Path,
    *,
    sequence_count: int,
    batch_size: int,
    max_steps: int,
    learning_rate: float,
    target_loss: float,
    eval_interval: int,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    set_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Tiny overfit requires CUDA in this project environment")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    model_config = model_config_from_dict(config["model"])
    tokenizer_path = Path(config["tokenizer"]["path"])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != model_config.vocab_size:
        raise RuntimeError("Tokenizer and model vocabulary sizes disagree")
    pad_token_id = tokenizer.token_to_id("<|pad|>")
    train_path = Path(config["data"]["train_path"])
    train_index_path = train_path.with_suffix(".index.json")
    dataset = PackedStoryDataset(
        train_path,
        train_index_path,
        sequence_length=model_config.max_seq_len,
        pad_token_id=pad_token_id,
        limit_sequences=sequence_count,
    )
    if not 100 <= len(dataset) <= 500:
        raise ValueError("Tiny overfit subset must contain 100-500 sequences")

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = FictionPulperLM(model_config).to(device)
    parameter_count = model.trainable_parameter_count()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0, fused=True
    )
    use_bf16 = torch.cuda.is_bf16_supported()
    packed_metadata = json.loads(
        Path(config["data"]["packed_metadata_path"]).read_text(encoding="utf-8")
    )

    print(f"trainable parameters: {parameter_count:,}")
    print(json.dumps(packed_metadata["schedule"], indent=2))
    print(
        f"tiny overfit: sequences={len(dataset)}, batch_size={batch_size}, "
        f"max_steps={max_steps}, precision={'bf16' if use_bf16 else 'fp32'}"
    )

    initial_metrics = evaluate_metrics(model, eval_loader, device, use_bf16)
    initial_loss = float(initial_metrics["loss"])
    expected_random_loss = math.log(model_config.vocab_size)
    print(
        f"initial loss: {initial_loss:.4f}; ln(vocab_size): {expected_random_loss:.4f}; "
        f"delta={initial_loss - expected_random_loss:+.4f}"
    )
    history = [{"step": 0, **initial_metrics}]
    subset_identity = hashlib.sha256(
        json.dumps(dataset.samples, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    valid_tokens_processed = 0
    allocated_token_slots_processed = 0
    missing_gradient_parameters: list[str] = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start_time = time.monotonic()
    iterator = iter(train_loader)
    final_step = 0
    for step in range(1, max_steps + 1):
        try:
            input_ids, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            input_ids, labels = next(iterator)
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            _, loss = model(input_ids, labels)
        assert loss is not None and math.isfinite(loss.item())
        loss.backward()
        if step == 1:
            missing_gradient_parameters = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            if missing_gradient_parameters:
                raise RuntimeError(
                    f"Missing gradients: {missing_gradient_parameters}"
                )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        valid_tokens_processed += labels.ne(-100).sum().item()
        allocated_token_slots_processed += labels.numel()
        final_step = step

        if step % eval_interval == 0 or step == max_steps:
            measured_metrics = evaluate_metrics(model, eval_loader, device, use_bf16)
            history.append({"step": step, **measured_metrics})
            print(
                f"step {step}: subset_loss={measured_metrics['loss']:.4f}, "
                f"accuracy={100.0 * measured_metrics['next_token_accuracy']:.2f}%"
            )
            if measured_metrics["loss"] <= target_loss:
                break

    torch.cuda.synchronize(device)
    elapsed_seconds = time.monotonic() - start_time
    peak_training_memory_bytes = torch.cuda.max_memory_allocated(device)
    final_metrics = history[-1]
    best_loss = min(float(entry["loss"]) for entry in history)
    generation_indices = [0, len(dataset) // 2, len(dataset) - 1]
    pre_save_generations = [
        compare_greedy_continuation(
            model,
            tokenizer,
            dataset[index][0],
            prompt_tokens=96,
            generated_tokens=96,
            device=device,
            use_bf16=use_bf16,
        )
        for index in generation_indices
    ]
    checkpoint_path = Path("checkpoints/tiny-overfit.pt")
    save_checkpoint_atomic(
        checkpoint_path,
        {
            "model": model.state_dict(),
            "model_config": asdict(model_config),
            "optimizer": optimizer.state_dict(),
            "step": final_step,
            "seed": seed,
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "subset_identity_sha256": subset_identity,
        },
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded_model = FictionPulperLM(model_config).to(device)
    reloaded_model.load_state_dict(checkpoint["model"])
    post_reload_generations = [
        compare_greedy_continuation(
            reloaded_model,
            tokenizer,
            dataset[index][0],
            prompt_tokens=96,
            generated_tokens=96,
            device=device,
            use_bf16=use_bf16,
        )
        for index in generation_indices
    ]
    checkpoint_reload_identical = all(
        before["generated_token_ids"] == after["generated_token_ids"]
        for before, after in zip(pre_save_generations, post_reload_generations, strict=True)
    )
    passed = (
        float(final_metrics["loss"]) <= target_loss
        and checkpoint_reload_identical
        and not missing_gradient_parameters
    )
    report = {
        "status": "passed" if passed else "failed",
        "seed": seed,
        "sequence_count": len(dataset),
        "subset_identity_sha256": subset_identity,
        "subset_non_padding_target_tokens": initial_metrics["valid_tokens"],
        "batch_size": batch_size,
        "max_steps": max_steps,
        "completed_steps": final_step,
        "learning_rate": learning_rate,
        "target_loss": target_loss,
        "initial_loss": initial_loss,
        "expected_random_initial_loss_ln_vocab": expected_random_loss,
        "initial_loss_delta_from_ln_vocab": initial_loss - expected_random_loss,
        "final_loss": final_metrics["loss"],
        "best_loss": best_loss,
        "final_next_token_accuracy": final_metrics["next_token_accuracy"],
        "elapsed_seconds": elapsed_seconds,
        "valid_tokens_processed": valid_tokens_processed,
        "allocated_token_slots_processed": allocated_token_slots_processed,
        "valid_tokens_per_second": valid_tokens_processed / elapsed_seconds,
        "allocated_token_slots_per_second": allocated_token_slots_processed / elapsed_seconds,
        "peak_gpu_memory_bytes": peak_training_memory_bytes,
        "peak_gpu_memory_gb": peak_training_memory_bytes / 1024**3,
        "trainable_parameter_count": parameter_count,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "missing_gradient_parameters": missing_gradient_parameters,
        "loss_history": history,
        "generation_indices": generation_indices,
        "pre_save_generations": pre_save_generations,
        "post_reload_generations": post_reload_generations,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_reload_identical": checkpoint_reload_identical,
    }
    report_path = Path("runs/tiny-overfit/report.json")
    write_json_atomic(report_path, report)
    if report["status"] != "passed":
        raise RuntimeError(
            f"Tiny overfit failed: final loss {report['final_loss']:.4f} > {target_loss:.4f}"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke-5m.yaml"))
    parser.add_argument("--sequences", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--target-loss", type=float, default=0.08)
    parser.add_argument("--eval-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_tiny_overfit(
        args.config,
        sequence_count=args.sequences,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        target_loss=args.target_loss,
        eval_interval=args.eval_interval,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
