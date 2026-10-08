"""Train locked Contrastive-v1 without changing Narrative-v1 CE presentations."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.contrastive_pairs import load_pairs
from src.mixed_schedule import ScheduledMixedDataset, verify_schedule
from src.model import FictionPulperLM, model_config_from_dict
from src.train import evaluate, learning_rate_for_step, set_seed
from src.train_mixed import (
    LOCKED_PARAMETER_COUNT,
    LOCKED_STEPS,
    LOCKED_TOTAL_VALID_TARGETS,
    _packed_dataset,
    _validate_schedule_contract,
    _verify_schedule_sources,
    capture_rng_state,
    git_commit,
    tracked_worktree_dirty,
    validate_locked_config as validate_narrative_lock,
    validate_real_artifacts,
    validate_runtime_environment,
)
from src.train_tokenizer import sha256_file, write_json_atomic


LOCKED_LAMBDA = 0.25


def validate_locked_config(config: dict[str, Any]) -> None:
    validate_narrative_lock(config)
    contrastive = config.get("contrastive", {})
    if contrastive.get("version") != "Contrastive-v1" or float(contrastive.get("lambda", -1)) != LOCKED_LAMBDA:
        raise RuntimeError("Locked Contrastive-v1 policy changed")
    if contrastive.get("score") != "mean_log_probability_decision_span" or contrastive.get("loss") != "softplus_negative_minus_positive":
        raise RuntimeError("Locked contrastive scoring policy changed")
    expected_exposure = {
        "expected_pair_presentations": 65_036,
        "expected_negative_decision_span_tokens": 1_639_692,
        "expected_ranking_forward_token_positions": 57_543_213,
    }
    if {key: int(contrastive.get(key, -1)) for key in expected_exposure} != expected_exposure:
        raise RuntimeError("Locked contrastive ranking exposure changed")
    if config["selection"] != {"metric": "data30m_validation_loss", "test_access": "forbidden_during_training"}:
        raise RuntimeError("Checkpoint selection must use only Data30M validation")
    if "curriculum_validation_path" in config["mixed_data"]:
        raise RuntimeError("Contrastive training must not access curriculum held-out data")
    if not str(config["training"]["checkpoint_dir"]).startswith("checkpoints/fictionpulper-15m-data30m-contrastive-v1"):
        raise RuntimeError("Contrastive checkpoints are not isolated")
    if config["logging"]["run_dir"] != "runs/fictionpulper-15m-data30m-contrastive-v1":
        raise RuntimeError("Contrastive run path is not isolated")


def mean_span_log_probability(logits: torch.Tensor, token_ids: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Mean next-token log probability over token indices [start, end)."""
    if token_ids.ndim != 1 or logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("Expected one unbatched token sequence and one batch of logits")
    if not 0 < start < end <= token_ids.numel() or logits.shape[1] < end - 1:
        raise ValueError("Decision span is empty or outside logits")
    log_probabilities = F.log_softmax(logits[0, start - 1 : end - 1].float(), dim=-1)
    return log_probabilities.gather(1, token_ids[start:end, None]).mean()


def pair_ranking_loss(model: Any, pair: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    start = int(pair["token_offsets"]["decision_start"])
    scores = []
    for ids in [pair["positive_token_ids"], *pair["negative_token_ids"]]:
        tokens = torch.tensor(ids, dtype=torch.long, device=device)
        logits, _ = model(tokens[:-1].unsqueeze(0))
        scores.append(mean_span_log_probability(logits, tokens, start, len(ids)))
    positive = scores[0]
    negative = torch.stack(scores[1:])
    losses = F.softplus(negative - positive)
    return losses.mean(), {
        "contrastive_loss": float(losses.mean().detach()),
        "positive_score": float(positive.detach()),
        "negative_score": float(negative.mean().detach()),
        "margin": float((positive - negative.mean()).detach()),
        "ranking_accuracy": float((positive > negative).float().mean().detach()),
        "state_type": pair["state_type"],
        "negative_decision_span_tokens": sum(len(ids) - start for ids in pair["negative_token_ids"]),
        "ranking_forward_token_positions": sum(
            len(ids) - 1 for ids in [pair["positive_token_ids"], *pair["negative_token_ids"]]
        ),
    }


class ScheduledPairIndex:
    """Resolve schedule entries within their identified packed document."""

    def __init__(self, pairs: Iterable[dict[str, Any]]) -> None:
        self.by_document: dict[str, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
        for pair in pairs:
            packed = pair["packed_offsets"]
            relative_start = int(packed["document_start"]) - int(packed["packed_document_start"])
            relative_end = relative_start + int(packed["document_length"])
            self.by_document[str(packed["packed_document_id"])].append(
                (relative_start, relative_end, pair)
            )
        for values in self.by_document.values():
            values.sort(key=lambda value: value[0])

    def for_entries(self, entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if entry["source"] != "curriculum":
                continue
            start = int(entry["relative_start"])
            end = start + int(entry["valid_targets"]) + 1
            for pair_start, pair_end, pair in self.by_document.get(str(entry["document_id"]), []):
                if pair_start >= end:
                    break
                if pair_end > start:
                    selected[pair["pair_id"]] = pair
        return list(selected.values())


def combine_training_losses(ce_loss: torch.Tensor, ranking_losses: list[torch.Tensor], weight: float) -> torch.Tensor:
    return ce_loss if not ranking_losses else ce_loss + weight * torch.stack(ranking_losses).mean()


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run_training(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_locked_config(config)
    if config["training"]["require_clean_worktree"] and tracked_worktree_dirty() is not False:
        raise RuntimeError("Locked training requires a clean tracked Git worktree")
    validate_runtime_environment()
    set_seed(int(config["seed"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    tokenizer_path = Path(config["tokenizer"]["path"])
    tokenizer_hash = sha256_file(tokenizer_path)
    if tokenizer_hash != config["tokenizer"]["expected_sha256"]:
        raise RuntimeError("Locked tokenizer hash changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise RuntimeError("Tokenizer has no pad token")
    validate_real_artifacts(config, tokenizer_hash)
    schedule_path = Path(config["mixed_data"]["schedule"]["path"])
    if sha256_file(schedule_path) != config["mixed_data"]["schedule"]["expected_file_sha256"]:
        raise RuntimeError("Exact Narrative-v1 schedule file changed")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule_hash = verify_schedule(schedule)
    if schedule_hash != config["mixed_data"]["schedule"]["expected_content_sha256"]:
        raise RuntimeError("Exact Narrative-v1 schedule content changed")
    _validate_schedule_contract(config, schedule)
    _verify_schedule_sources(schedule)
    sequence_length = int(config["model"]["max_seq_len"])
    sources = {name: _packed_dataset(details["path"], sequence_length, int(pad_id)) for name, details in config["mixed_data"]["training_sources"].items()}
    dataset = ScheduledMixedDataset(sources, schedule)
    validation = _packed_dataset(config["data"]["validation_path"], sequence_length, int(pad_id))
    pairs_path = Path(config["contrastive"]["pairs"]["path"])
    if sha256_file(pairs_path) != config["contrastive"]["pairs"]["expected_sha256"]:
        raise RuntimeError("Validated contrastive pair artifact changed")
    pair_manifest_path = Path(config["contrastive"]["pairs"]["manifest_path"])
    if sha256_file(pair_manifest_path) != config["contrastive"]["pairs"]["expected_manifest_sha256"]:
        raise RuntimeError("Validated contrastive pair manifest changed")
    pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
    if pair_manifest["pairs_sha256"] != config["contrastive"]["pairs"]["expected_sha256"]:
        raise RuntimeError("Pair manifest does not identify the locked pair artifact")
    pair_index = ScheduledPairIndex(load_pairs(pairs_path, max_seq_len=sequence_length))
    model_config = model_config_from_dict(config["model"])
    model = FictionPulperLM(model_config).to(device)
    if model.trainable_parameter_count() != LOCKED_PARAMETER_COUNT:
        raise RuntimeError("Locked parameter count changed")
    training = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]), fused=True)
    run_dir = Path(config["logging"]["run_dir"])
    checkpoint_dir = Path(training["checkpoint_dir"])
    if run_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Isolated run or checkpoint directory already exists")
    run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    started_at = datetime.now().isoformat()
    manifest: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "status": "training_started",
        "started_at": started_at,
        "training_git_commit": git_commit(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "seed": int(config["seed"]),
        "model_initialization_seed": int(config["seed"]),
        "fresh_random_initialization": True,
        "resume_checkpoint": None,
        "model_parameter_count": model.trainable_parameter_count(),
        "model_config": asdict(model_config),
        "tokenizer_sha256": tokenizer_hash,
        "schedule_path": str(schedule_path),
        "schedule_file_sha256": sha256_file(schedule_path),
        "schedule_content_sha256": schedule_hash,
        "pairs_path": str(pairs_path),
        "pairs_sha256": sha256_file(pairs_path),
        "pairs_manifest_path": str(pair_manifest_path),
        "pairs_manifest_sha256": sha256_file(pair_manifest_path),
        "checkpoint_selection": "Data30M validation loss only at steps 444, 888, 1332, 1776, 2220",
        "held_out_access_during_training": False,
    }
    write_json_atomic(run_dir / "manifest.json", manifest)
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=False, num_workers=0, pin_memory=True)
    accumulation = int(training["gradient_accumulation_steps"])
    checkpoint_steps = set(training["checkpoint_steps"])
    optimizer.zero_grad(set_to_none=True)
    step = 0
    total_targets = 0
    source_targets = defaultdict(int)
    interval = defaultdict(float)
    interval_by_type: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    wall_started = time.monotonic()
    initial_validation = evaluate(model, validation, batch_size=int(training["batch_size"]), device=device, use_bf16=True)
    metrics: list[dict[str, Any]] = [{
        "step": 0,
        "selection_eligible": False,
        "data30m_validation": initial_validation,
        "elapsed_seconds": time.monotonic() - wall_started,
    }]
    writer.add_scalar("validation/data30m_loss", initial_validation["loss"], 0)
    write_json_atomic(run_dir / "metrics.json", {"checkpoints": metrics})
    best_loss = math.inf
    best_step = 0
    started = time.monotonic()
    interval_started = started
    training_seconds = 0.0
    total_pairs = 0
    total_negative_decision_tokens = 0
    total_ranking_forward_positions = 0
    total_ce_allocated_positions = 0
    try:
        for microbatch, (input_ids, labels) in enumerate(loader, 1):
            input_ids, labels = input_ids.to(device), labels.to(device)
            offset = (microbatch - 1) * int(training["batch_size"])
            entries = schedule["entries"][offset : offset + input_ids.shape[0]]
            valid_targets = int(labels.ne(-100).sum())
            if valid_targets != sum(int(entry["valid_targets"]) for entry in entries):
                raise RuntimeError("CE targets differ from the exact Narrative-v1 schedule")
            for entry in entries:
                source_targets[str(entry["source"])] += int(entry["valid_targets"])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, ce_loss = model(input_ids, labels)
                ranking_results = [pair_ranking_loss(model, pair, device) for pair in pair_index.for_entries(entries)]
                ranking_losses = [result[0] for result in ranking_results]
                loss = combine_training_losses(ce_loss, ranking_losses, LOCKED_LAMBDA)  # type: ignore[arg-type]
            if ce_loss is None or not torch.isfinite(loss):
                raise FloatingPointError("Training loss is NaN or Inf")
            total_targets += valid_targets
            interval["ce_sum"] += float(ce_loss.detach()) * valid_targets
            interval["objective_sum"] += float(loss.detach())
            interval["microbatches"] += 1
            interval["targets"] += valid_targets
            allocated_positions = int(input_ids.numel())
            interval["ce_allocated_positions"] += allocated_positions
            total_ce_allocated_positions += allocated_positions
            for _, result in ranking_results:
                interval["contrastive_sum"] += result["contrastive_loss"]
                interval["positive_sum"] += result["positive_score"]
                interval["negative_sum"] += result["negative_score"]
                interval["margin_sum"] += result["margin"]
                interval["accuracy_sum"] += result["ranking_accuracy"]
                interval["pairs"] += 1
                interval["negative_decision_tokens"] += result["negative_decision_span_tokens"]
                interval["ranking_forward_positions"] += result["ranking_forward_token_positions"]
                total_pairs += 1
                total_negative_decision_tokens += int(result["negative_decision_span_tokens"])
                total_ranking_forward_positions += int(result["ranking_forward_token_positions"])
                typed = interval_by_type[result["state_type"]]
                typed["contrastive_loss"] += result["contrastive_loss"]
                for key in ("positive_score", "negative_score", "margin", "ranking_accuracy"):
                    typed[key] += result[key]
                typed["count"] += 1
            (loss / accumulation).backward()
            if microbatch % accumulation:
                continue
            step += 1
            lr = learning_rate_for_step(step, max_steps=LOCKED_STEPS, warmup_steps=int(training["warmup_steps"]), learning_rate=float(training["learning_rate"]), min_learning_rate=float(training["min_learning_rate"]))
            for group in optimizer.param_groups:
                group["lr"] = lr
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip"]))
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Gradient norm is NaN or Inf")
            interval["gradient_norm_sum"] += float(gradient_norm)
            interval["gradient_norm_max"] = max(interval["gradient_norm_max"], float(gradient_norm))
            interval["optimizer_steps"] += 1
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step not in checkpoint_steps:
                continue
            elapsed = time.monotonic() - interval_started
            training_seconds += elapsed
            validation_metrics = evaluate(model, validation, batch_size=int(training["batch_size"]), device=device, use_bf16=True)
            pair_count = int(interval["pairs"])
            metric: dict[str, Any] = {
                "step": step,
                "selection_eligible": True,
                "ce_loss": interval["ce_sum"] / interval["targets"],
                "mean_microbatch_objective": interval["objective_sum"] / interval["microbatches"],
                "objective_definition": "mean over microbatches of CE + 0.25 * mean pair contrastive loss",
                "contrastive_loss": interval["contrastive_sum"] / pair_count if pair_count else 0.0,
                "positive_score": interval["positive_sum"] / pair_count if pair_count else 0.0,
                "negative_score": interval["negative_sum"] / pair_count if pair_count else 0.0,
                "margin": interval["margin_sum"] / pair_count if pair_count else 0.0,
                "ranking_accuracy": interval["accuracy_sum"] / pair_count if pair_count else 0.0,
                "ranking_pair_presentations": pair_count,
                "negative_decision_span_tokens_scored": int(interval["negative_decision_tokens"]),
                "ranking_forward_token_positions": int(interval["ranking_forward_positions"]),
                "relative_compute_increase_proxy": interval["ranking_forward_positions"] / interval["ce_allocated_positions"],
                "ranking_by_state_type": {state: {key: value / values["count"] for key, value in values.items() if key != "count"} | {"count": int(values["count"])} for state, values in interval_by_type.items()},
                "data30m_validation": validation_metrics,
                "learning_rate": lr,
                "valid_targets_per_second": interval["targets"] / elapsed,
                "samples_per_second": interval["microbatches"] * int(training["batch_size"]) / elapsed,
                "gradient_norm_mean": interval["gradient_norm_sum"] / interval["optimizer_steps"],
                "gradient_norm_max": interval["gradient_norm_max"],
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
                "elapsed_seconds": time.monotonic() - started,
                "cumulative_valid_targets": total_targets,
                "cumulative_contrastive_pairs_scored": total_pairs,
                "cumulative_negative_decision_span_tokens_scored": total_negative_decision_tokens,
            }
            metrics.append(metric)
            for name in ("mean_microbatch_objective", "ce_loss", "contrastive_loss", "positive_score", "negative_score", "margin", "ranking_accuracy", "valid_targets_per_second", "samples_per_second", "gradient_norm_mean", "gradient_norm_max", "gpu_memory_allocated_gb", "gpu_memory_reserved_gb", "relative_compute_increase_proxy"):
                writer.add_scalar(f"train/{name}", metric[name], step)
            writer.add_scalar("validation/data30m_loss", validation_metrics["loss"], step)
            writer.add_scalar("train/learning_rate", lr, step)
            for state, values in metric["ranking_by_state_type"].items():
                for name, value in values.items():
                    writer.add_scalar(f"ranking/{state}/{name}", value, step)
            write_json_atomic(run_dir / "metrics.json", {"checkpoints": metrics})
            candidate_is_best = float(validation_metrics["loss"]) < best_loss
            if candidate_is_best:
                best_loss, best_step = float(validation_metrics["loss"]), step
            payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": {"step": step, "learning_rate": lr, "policy": "linear warmup then cosine decay"}, "rng_state": capture_rng_state(), "model_config": asdict(model_config), "config": config, "step": step, "data30m_validation": validation_metrics, "real_validation_loss": float(validation_metrics["loss"]), "validation_loss": float(validation_metrics["loss"]), "best_validation_loss": best_loss, "best_data30m_validation_loss": best_loss, "best_checkpoint_step": best_step, "metrics": metrics, "tokenizer_hash": tokenizer_hash, "schedule_sha256": schedule_hash, "pairs_sha256": sha256_file(pairs_path), "model_initialization_seed": 1337, "fresh_random_initialization": True, "resume_checkpoint": None, "selection_metric": "Data30M validation loss only", "selection_eligible": True, "positive_lm_target_presentations": total_targets, "contrastive_pairs_scored": total_pairs, "negative_decision_span_tokens_scored": total_negative_decision_tokens}
            _save(checkpoint_dir / "latest.pt", payload)
            _save(checkpoint_dir / f"step-{step:04d}.pt", payload)
            if candidate_is_best:
                _save(checkpoint_dir / "best-validation.pt", payload)
            interval = defaultdict(float)
            interval_by_type = defaultdict(lambda: defaultdict(float))
            interval_started = time.monotonic()
        expected_source_targets = {name: int(value["valid_targets"]) for name, value in schedule["sources"].items()}
        if step != LOCKED_STEPS or total_targets != LOCKED_TOTAL_VALID_TARGETS or dict(source_targets) != expected_source_targets:
            raise RuntimeError("Final step or positive valid-target exposure differs from Narrative-v1")
        if (
            total_pairs != int(config["contrastive"]["expected_pair_presentations"])
            or total_negative_decision_tokens != int(config["contrastive"]["expected_negative_decision_span_tokens"])
            or total_ranking_forward_positions != int(config["contrastive"]["expected_ranking_forward_token_positions"])
        ):
            raise RuntimeError("Observed contrastive ranking exposure differs from the locked schedule")
        completed_at = datetime.now().isoformat()
        best_checkpoint = checkpoint_dir / "best-validation.pt"
        summary = {"status": "training_and_selection_complete", "started_at": started_at, "completed_at": completed_at, "optimizer_steps": step, "positive_lm_target_presentations": total_targets, "total_valid_targets": total_targets, "source_valid_targets_observed": dict(source_targets), "contrastive_pairs_scored": total_pairs, "negative_decision_span_tokens_scored": total_negative_decision_tokens, "ranking_forward_token_positions": total_ranking_forward_positions, "ce_allocated_token_positions": total_ce_allocated_positions, "relative_compute_increase_proxy": total_ranking_forward_positions / total_ce_allocated_positions, "relative_compute_increase_proxy_definition": "ranking forward token positions divided by scheduled CE allocated token positions", "training_seconds": training_seconds, "total_runtime_seconds": time.monotonic() - wall_started, "best_data30m_validation_loss": best_loss, "best_checkpoint_step": best_step, "best_checkpoint": str(best_checkpoint), "best_checkpoint_sha256": sha256_file(best_checkpoint), "selection_metric": "Data30M validation loss only", "curriculum_validation_used_for_selection": False, "held_out_access_during_training": False, "schedule_content_sha256": schedule_hash, "pairs_sha256": sha256_file(pairs_path), "lambda": LOCKED_LAMBDA, "pre_training_data30m_validation": initial_validation, "selection_candidates": [{"step": item["step"], "data30m_validation_loss": item["data30m_validation"]["loss"]} for item in metrics if item.get("selection_eligible")], "metrics": metrics}
        write_json_atomic(run_dir / "summary.json", summary)
        manifest.update({"status": summary["status"], "completed_at": completed_at, "summary_path": str(run_dir / "summary.json"), "summary_sha256": sha256_file(run_dir / "summary.json"), "best_checkpoint_step": best_step, "best_checkpoint": str(best_checkpoint), "best_checkpoint_sha256": summary["best_checkpoint_sha256"], "positive_lm_target_presentations": total_targets, "contrastive_pairs_scored": total_pairs, "negative_decision_span_tokens_scored": total_negative_decision_tokens})
        write_json_atomic(run_dir / "manifest.json", manifest)
        return summary
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_training(args.config), indent=2))


if __name__ == "__main__":
    main()
