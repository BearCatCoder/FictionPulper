"""Train locked Counterfactual-v1 with positive CE and symmetric paired ranking."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.continuity_curriculum.common import canonical_json, sha256_bytes
from src.mixed_schedule import ScheduledMixedDataset, verify_schedule
from src.model import FictionPulperLM, model_config_from_dict
from src.pack_counterfactual_v1 import verify_counterfactual_gates
from src.train import evaluate, learning_rate_for_step, set_seed
from src.train_contrastive import mean_span_log_probability
from src.train_mixed import (
    LOCKED_CHECKPOINT_STEPS, LOCKED_MODEL, LOCKED_PARAMETER_COUNT, LOCKED_SEED,
    LOCKED_STEPS, LOCKED_TOTAL_VALID_TARGETS, _packed_dataset,
    _validate_schedule_contract, _verify_schedule_sources, capture_rng_state,
    git_commit, tracked_worktree_dirty, validate_real_artifacts,
    validate_runtime_environment,
)
from src.train_tokenizer import sha256_file, write_json_atomic


LOCKED_LAMBDA = 0.25
LOCKED_REAL_TARGETS = 115_503_402
LOCKED_CURRICULUM_TARGETS = 20_382_953


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            "test" in str(key).lower() or "generalization" in str(key).lower()
            or _contains_forbidden_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def validate_locked_config(config: dict[str, Any], *, require_generated_locks: bool = True) -> None:
    expected_training = {
        "optimizer": "adamw", "batch_size": 16, "gradient_accumulation_steps": 4,
        "max_steps": 2220, "learning_rate": 1e-3, "min_learning_rate": 1e-4,
        "warmup_steps": 111, "weight_decay": 0.1, "grad_clip": 1.0,
        "precision": "bf16", "epochs": 5, "validation_interval": 444,
        "checkpoint_steps": LOCKED_CHECKPOINT_STEPS, "require_clean_worktree": True,
        "checkpoint_dir": "checkpoints/fictionpulper-15m-data30m-counterfactual-v1",
    }
    if int(config["seed"]) != LOCKED_SEED or config["model"] != LOCKED_MODEL:
        raise RuntimeError("Locked Counterfactual-v1 model/seed changed")
    if {key: config["training"][key] for key in expected_training} != expected_training:
        raise RuntimeError("Locked Counterfactual-v1 optimizer/compute policy changed")
    if config["logging"]["run_dir"] != "runs/fictionpulper-15m-data30m-counterfactual-v1":
        raise RuntimeError("Counterfactual-v1 run directory is not isolated")
    sources = config["mixed_data"]["training_sources"]
    if set(sources) != {"real", "curriculum"} or int(sources["real"]["valid_targets"]) != LOCKED_REAL_TARGETS or int(sources["curriculum"]["valid_targets"]) != LOCKED_CURRICULUM_TARGETS:
        raise RuntimeError("Locked 85/15 exposure changed")
    if int(config["mixed_data"]["schedule"]["total_chunks"]) != 142_080:
        raise RuntimeError("Locked mixer chunk count changed")
    objective = config["counterfactual_objective"]
    expected_objective = {
        "version": "Counterfactual-v1", "lambda": LOCKED_LAMBDA,
        "score": "mean_log_probability_exact_candidate_decision_span",
        "loss": "symmetric_softplus_incorrect_minus_correct",
        "ranking_labels": None, "positive_ce_policy": "correct_worlds_only",
    }
    if any(objective.get(key) != value for key, value in expected_objective.items()):
        raise RuntimeError("Locked symmetric Counterfactual-v1 objective changed")
    if int(objective.get("diagnostic_train_pairs", -1)) != 64:
        raise RuntimeError("Locked train-only diagnostic subset changed")
    if config["selection"] != {
        "metric": "data30m_validation_loss",
        "counterfactual_validation_role": "packing_validation_only_not_loaded_by_trainer",
        "test_access": "forbidden_during_training",
        "generalization_access": "forbidden_during_training",
    }:
        raise RuntimeError("Checkpoint selection/access lock changed")
    trainer_visible = {"data": config["data"], "mixed_data": config["mixed_data"]}
    if _contains_forbidden_key(trainer_visible):
        raise RuntimeError("Trainer-visible configuration contains held-out access")
    if require_generated_locks:
        generated = [
            config["counterfactual_packing"][key] for key in (
                "expected_metadata_sha256", "expected_train_sha256", "expected_train_index_sha256",
                "expected_validation_sha256", "expected_validation_index_sha256",
            )
        ] + [config["mixed_data"]["schedule"][key] for key in ("expected_file_sha256", "expected_content_sha256")]
        generated += [objective[key] for key in ("expected_pair_presentations", "expected_candidate_span_world_triggers", "expected_unique_pairs")]
        if any(str(value).startswith("PENDING_") for value in generated):
            raise RuntimeError("Generated packing/schedule exposure locks are still pending")


def load_train_pairs(path: Path, expected_sha256: str) -> dict[str, dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("Locked counterfactual train source changed")
    pairs = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            pair = json.loads(line)
            unhashed = {key: value for key, value in pair.items() if key != "stable_hash"}
            if pair.get("stable_hash") != sha256_bytes(canonical_json(unhashed).encode("utf-8")):
                raise RuntimeError(f"Counterfactual pair hash changed at line {line_number}")
            if pair.get("split") != "train":
                raise RuntimeError("Ranking source contains a non-training pair")
            pairs[str(pair["pair_id"])] = pair
    return pairs


def score_candidate(model: Any, encoding: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    tokens = torch.tensor(encoding["token_ids"], dtype=torch.long, device=device)
    logits, loss = model(tokens[:-1].unsqueeze(0), labels=None)
    if loss is not None:
        raise RuntimeError("Ranking forward unexpectedly produced CE loss")
    return mean_span_log_probability(
        logits, tokens, int(encoding["decision_start"]), int(encoding["decision_end"])
    )


def symmetric_pair_loss(model: Any, pair: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    scores: dict[str, dict[str, torch.Tensor]] = {}
    losses = []
    accuracies = {}
    margins = {}
    for world_name in ("A", "B"):
        world = pair["worlds"][world_name]
        scores[world_name] = {
            candidate: score_candidate(model, world["candidate_encodings"][candidate], device)
            for candidate in ("X", "Y")
        }
        correct = str(world["correct_candidate"])
        incorrect = "Y" if correct == "X" else "X"
        margin = scores[world_name][correct] - scores[world_name][incorrect]
        losses.append(F.softplus(-margin))
        margins[world_name] = margin
        accuracies[world_name] = margin > 0
    loss = 0.5 * (losses[0] + losses[1])
    return loss, {
        "ranking_loss": float(loss.detach()),
        "world_A_accuracy": float(accuracies["A"].detach()),
        "world_B_accuracy": float(accuracies["B"].detach()),
        "paired_reversal_success": float((accuracies["A"] & accuracies["B"]).detach()),
        "signed_margin": float((0.5 * (margins["A"] + margins["B"])).detach()),
        "state_family": pair["abstract_counterfactual_variable"]["state_family"],
        "distance_bucket": pair["metadata"]["distance_bucket"],
        "difficulty": str(pair["metadata"]["difficulty"]),
    }


def source_token_losses(logits: torch.Tensor, labels: torch.Tensor, entries: Iterable[Mapping[str, Any]]) -> dict[str, tuple[torch.Tensor, int]]:
    per_token = F.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
    results = {}
    entries_list = list(entries)
    for source in sorted({str(entry["source"]) for entry in entries_list}):
        rows = torch.tensor([str(entry["source"]) == source for entry in entries_list], device=labels.device)
        mask = rows[:, None] & labels.ne(-100)
        count = int(mask.sum().item())
        results[source] = (per_token.masked_select(mask).sum(), count)
    return results


def attached_pairs(entries: Iterable[Mapping[str, Any]], pairs: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ids = [str(item["pair_id"]) for entry in entries for item in entry.get("pair_presentations", [])]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Schedule supplied duplicate pair presentations in a microbatch")
    try:
        return [pairs[pair_id] for pair_id in ids]
    except KeyError as error:
        raise RuntimeError(f"Scheduled pair is absent from locked train source: {error}") from error


def _group_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    def summary(values: list[dict[str, Any]]) -> dict[str, float | int]:
        keys = ("ranking_loss", "world_A_accuracy", "world_B_accuracy", "paired_reversal_success", "signed_margin")
        return {**{key: sum(item[key] for item in values) / len(values) for key in keys}, "count": len(values)}
    output: dict[str, Any] = {"overall": summary(results)} if results else {"overall": {"count": 0}}
    for field in ("state_family", "distance_bucket", "difficulty"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            groups[str(result[field])].append(result)
        output[f"by_{field}"] = {key: summary(values) for key, values in sorted(groups.items())}
    latest = [item for item in results if item["state_family"] == "latest_state_update"]
    output["latest_state_subset"] = summary(latest) if latest else {"count": 0}
    return output


@torch.no_grad()
def training_control_diagnostics(
    model: Any, pairs: Iterable[dict[str, Any]], tokenizer: Tokenizer, device: torch.device
) -> dict[str, Any]:
    """Score a fixed train-only subset; these diagnostics never drive selection."""
    eos = tokenizer.token_to_id("<|eos|>")
    story = tokenizer.token_to_id("<|story|>")
    bos = tokenizer.token_to_id("<|bos|>")
    if eos is None or story is None or bos is None:
        raise RuntimeError("Tokenizer lacks required control tokens")
    results: dict[str, list[float]] = {
        "fact_removed_margin_x_minus_y": [],
        "irrelevant_substituted_margin_x_minus_y": [],
    }
    model_was_training = model.training
    model.eval()
    for pair in pairs:
        control = pair["metadata"].get("genre_control_token")
        prefix = [story]
        if control:
            control_id = tokenizer.token_to_id(control)
            if control_id is None:
                raise RuntimeError("Train diagnostic uses an unknown genre token")
            prefix.append(control_id)
        prefix.append(bos)
        for condition, ids_key in (
            ("fact_removed", "fact_removed_token_ids"),
            ("irrelevant_substituted", "irrelevant_substituted_token_ids"),
        ):
            context_ids = pair["controls"][ids_key]
            scores = {}
            for candidate_name in ("X", "Y"):
                candidate_ids = pair["candidates"][candidate_name]["token_ids"]
                encoding = {
                    "token_ids": [*prefix, *context_ids, *candidate_ids, eos],
                    "decision_start": len(prefix) + len(context_ids),
                    "decision_end": len(prefix) + len(context_ids) + len(candidate_ids),
                }
                scores[candidate_name] = float(score_candidate(model, encoding, device))
            results[f"{condition}_margin_x_minus_y"].append(scores["X"] - scores["Y"])
    if model_was_training:
        model.train()
    return {
        key: {
            "count": len(values),
            "mean_margin_x_minus_y": sum(values) / len(values),
            "mean_absolute_margin": sum(abs(value) for value in values) / len(values),
            "X_win_rate": sum(value > 0 for value in values) / len(values),
            "Y_win_rate": sum(value < 0 for value in values) / len(values),
        }
        for key, values in results.items()
    }


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
    set_seed(LOCKED_SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    tokenizer_path = Path(config["tokenizer"]["path"])
    tokenizer_hash = sha256_file(tokenizer_path)
    if tokenizer_hash != config["tokenizer"]["expected_sha256"]:
        raise RuntimeError("Locked tokenizer hash changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None or tokenizer.get_vocab_size() != 4096:
        raise RuntimeError("Locked tokenizer contract changed")
    gates = verify_counterfactual_gates(config, verify_validation_source=False)
    real_artifacts = validate_real_artifacts(config, tokenizer_hash)

    packing = config["counterfactual_packing"]
    metadata_path = Path(packing["metadata_path"])
    if sha256_file(metadata_path) != packing["expected_metadata_sha256"]:
        raise RuntimeError("Counterfactual packed metadata changed")
    packed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    train_details = packed_metadata["splits"]["train"]
    if sha256_file(Path(train_details["bin_path"])) != packing["expected_train_sha256"] or sha256_file(Path(train_details["index_path"])) != packing["expected_train_index_sha256"]:
        raise RuntimeError("Counterfactual packed train artifact changed")
    schedule_path = Path(config["mixed_data"]["schedule"]["path"])
    if sha256_file(schedule_path) != config["mixed_data"]["schedule"]["expected_file_sha256"]:
        raise RuntimeError("Counterfactual mixed schedule file changed")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule_hash = verify_schedule(schedule)
    if schedule_hash != config["mixed_data"]["schedule"]["expected_content_sha256"]:
        raise RuntimeError("Counterfactual mixed schedule content changed")
    _validate_schedule_contract(config, schedule)
    _verify_schedule_sources(schedule)
    expected_attachment = {
        "pair_presentations": int(config["counterfactual_objective"]["expected_pair_presentations"]),
        "candidate_span_world_triggers": int(config["counterfactual_objective"]["expected_candidate_span_world_triggers"]),
        "unique_pairs": int(config["counterfactual_objective"]["expected_unique_pairs"]),
    }
    if any(int(schedule["pair_attachment"][key]) != value for key, value in expected_attachment.items()):
        raise RuntimeError("Locked pair-presentation exposure changed")

    pairs = load_train_pairs(Path(config["counterfactual_data"]["train_path"]), config["counterfactual_data"]["expected_train_sha256"])
    sequence_length = int(config["model"]["max_seq_len"])
    sources = {name: _packed_dataset(details["path"], sequence_length, int(pad_id)) for name, details in config["mixed_data"]["training_sources"].items()}
    dataset = ScheduledMixedDataset(sources, schedule)
    validation = _packed_dataset(config["data"]["validation_path"], sequence_length, int(pad_id))
    model_config = model_config_from_dict(config["model"])
    model = FictionPulperLM(model_config).to(device)
    if model.trainable_parameter_count() != LOCKED_PARAMETER_COUNT:
        raise RuntimeError("Locked model parameter count changed")
    training = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]), fused=True)
    run_dir = Path(config["logging"]["run_dir"])
    checkpoint_dir = Path(training["checkpoint_dir"])
    if checkpoint_dir.exists():
        raise FileExistsError("Counterfactual checkpoint path already exists; resume is forbidden")
    if run_dir.exists():
        unexpected = {path.name for path in run_dir.iterdir()} - {"pretraining-audit"}
        if unexpected:
            raise FileExistsError("Counterfactual run path contains training output; resume is forbidden")
    else:
        run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    manifest = {
        "experiment_id": config["experiment_id"], "status": "training_started",
        "started_at": datetime.now().isoformat(), "training_git_commit": git_commit(),
        "config_sha256": sha256_file(config_path), "fresh_random_initialization": True,
        "resume_checkpoint": None, "prior_checkpoint_loaded": False,
        "model_parameter_count": model.trainable_parameter_count(), "model_config": asdict(model_config),
        "tokenizer_sha256": tokenizer_hash, "dataset_gates": gates, "real_artifacts": real_artifacts,
        "packed_metadata_sha256": sha256_file(metadata_path), "schedule_file_sha256": sha256_file(schedule_path),
        "schedule_content_sha256": schedule_hash, "expected_exposure": expected_attachment,
        "selection_metric": "Data30M validation loss only", "held_out_access_during_training": False,
    }
    write_json_atomic(run_dir / "manifest.json", manifest)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    checkpoint_steps = set(LOCKED_CHECKPOINT_STEPS)
    accumulation = 4
    optimizer.zero_grad(set_to_none=True)
    step = total_targets = total_pairs = 0
    source_targets: dict[str, int] = defaultdict(int)
    interval: dict[str, float] = defaultdict(float)
    ranking_results: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    best_loss, best_step = math.inf, 0
    wall_started = time.monotonic()
    initial_validation = evaluate(model, validation, batch_size=16, device=device, use_bf16=True)
    metrics.append({"step": 0, "selection_eligible": False, "data30m_validation": initial_validation})
    interval_started = time.monotonic()
    diagnostic_subset = list(pairs.values())[:int(config["counterfactual_objective"]["diagnostic_train_pairs"])]
    try:
        for microbatch, (input_ids, labels) in enumerate(loader, 1):
            input_ids, labels = input_ids.to(device), labels.to(device)
            offset = (microbatch - 1) * 16
            entries = schedule["entries"][offset:offset + input_ids.shape[0]]
            valid_targets = int(labels.ne(-100).sum())
            if valid_targets != sum(int(entry["valid_targets"]) for entry in entries):
                raise RuntimeError("Observed CE targets differ from schedule")
            selected = attached_pairs(entries, pairs)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, ce_loss = model(input_ids, labels)
                pair_outputs = [symmetric_pair_loss(model, pair, device) for pair in selected]
                rank_losses = [item[0] for item in pair_outputs]
                ranking_loss = torch.stack(rank_losses).mean() if rank_losses else logits.sum() * 0.0
                objective = ce_loss + LOCKED_LAMBDA * ranking_loss  # type: ignore[operator]
            if ce_loss is None or not torch.isfinite(objective):
                raise FloatingPointError("Counterfactual objective is NaN or Inf")
            source_losses = source_token_losses(logits, labels, entries)
            for source, (loss_sum, count) in source_losses.items():
                interval[f"{source}_ce_sum"] += float(loss_sum.detach())
                interval[f"{source}_targets"] += count
                source_targets[source] += count
            interval["objective_sum"] += float(objective.detach())
            interval["ranking_sum"] += float(ranking_loss.detach()) * len(rank_losses)
            interval["microbatches"] += 1
            interval["targets"] += valid_targets
            interval["samples"] += input_ids.shape[0]
            total_targets += valid_targets
            total_pairs += len(rank_losses)
            ranking_results.extend(item[1] for item in pair_outputs)
            (objective / accumulation).backward()
            if microbatch % accumulation:
                continue
            step += 1
            lr = learning_rate_for_step(step, max_steps=LOCKED_STEPS, warmup_steps=111, learning_rate=1e-3, min_learning_rate=1e-4)
            for group in optimizer.param_groups:
                group["lr"] = lr
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Gradient norm is NaN or Inf")
            interval["grad_norm_sum"] += float(grad_norm)
            interval["grad_norm_max"] = max(interval["grad_norm_max"], float(grad_norm))
            interval["steps"] += 1
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step not in checkpoint_steps:
                continue
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - interval_started
            validation_metrics = evaluate(model, validation, batch_size=16, device=device, use_bf16=True)
            grouped = _group_metrics(ranking_results)
            controls = training_control_diagnostics(model, diagnostic_subset, tokenizer, device)
            metric = {
                "step": step, "selection_eligible": True,
                "total_objective": interval["objective_sum"] / interval["microbatches"],
                "real_ce_loss": interval["real_ce_sum"] / interval["real_targets"],
                "curriculum_ce_loss": interval["curriculum_ce_sum"] / interval["curriculum_targets"],
                "ranking": grouped, "ranking_pair_presentations": len(ranking_results),
                "training_only_control_diagnostics": controls,
                "data30m_validation": validation_metrics, "learning_rate": lr,
                "valid_targets_per_second": interval["targets"] / elapsed,
                "samples_per_second": interval["samples"] / elapsed,
                "gradient_norm_mean": interval["grad_norm_sum"] / interval["steps"],
                "gradient_norm_max": interval["grad_norm_max"],
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
                "cumulative_valid_targets": total_targets,
                "cumulative_source_valid_targets": dict(source_targets),
                "cumulative_pair_presentations": total_pairs,
                "elapsed_seconds": time.monotonic() - wall_started,
            }
            metrics.append(metric)
            for name in ("total_objective", "real_ce_loss", "curriculum_ce_loss", "valid_targets_per_second", "samples_per_second", "gradient_norm_mean", "gradient_norm_max", "gpu_memory_allocated_gb", "gpu_memory_reserved_gb"):
                writer.add_scalar(f"train/{name}", metric[name], step)
            for name, value in grouped["overall"].items():
                if name != "count": writer.add_scalar(f"ranking/{name}", value, step)
            for dimension in ("by_state_family", "by_distance_bucket", "by_difficulty"):
                for group, values in grouped[dimension].items():
                    for name, value in values.items():
                        writer.add_scalar(f"ranking/{dimension}/{group}/{name}", value, step)
            for name, value in grouped["latest_state_subset"].items():
                writer.add_scalar(f"ranking/latest_state_subset/{name}", value, step)
            for condition, values in controls.items():
                for name, value in values.items():
                    writer.add_scalar(f"diagnostic_train_only/{condition}/{name}", value, step)
            writer.add_scalar("validation/data30m_loss", validation_metrics["loss"], step)
            writer.add_scalar("train/learning_rate", lr, step)
            write_json_atomic(run_dir / "metrics.json", {"checkpoints": metrics})
            candidate_best = float(validation_metrics["loss"]) < best_loss
            if candidate_best:
                best_loss, best_step = float(validation_metrics["loss"]), step
            payload = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": {"step": step, "learning_rate": lr, "policy": "linear warmup then cosine decay"},
                "rng_state": capture_rng_state(), "model_config": asdict(model_config), "config": config,
                "step": step, "data30m_validation": validation_metrics, "metrics": metrics,
                "tokenizer_hash": tokenizer_hash, "schedule_sha256": schedule_hash,
                "fresh_random_initialization": True, "resume_checkpoint": None,
                "selection_metric": "Data30M validation loss only",
                "source_valid_targets_observed": dict(source_targets), "pair_presentations": total_pairs,
            }
            _save(checkpoint_dir / "latest.pt", payload)
            _save(checkpoint_dir / f"step-{step:04d}.pt", payload)
            if candidate_best: _save(checkpoint_dir / "best-validation.pt", payload)
            interval = defaultdict(float)
            ranking_results = []
            interval_started = time.monotonic()
        expected_sources = {name: int(item["valid_targets"]) for name, item in schedule["sources"].items()}
        if step != LOCKED_STEPS or total_targets != LOCKED_TOTAL_VALID_TARGETS or dict(source_targets) != expected_sources or total_pairs != expected_attachment["pair_presentations"]:
            raise RuntimeError("Final locked exposure does not match schedule")
        best = checkpoint_dir / "best-validation.pt"
        summary = {
            "status": "training_and_selection_complete", "optimizer_steps": step,
            "total_valid_targets": total_targets, "source_valid_targets_observed": dict(source_targets),
            "pair_presentations": total_pairs, "best_data30m_validation_loss": best_loss,
            "best_checkpoint_step": best_step, "best_checkpoint": str(best),
            "best_checkpoint_sha256": sha256_file(best), "selection_metric": "Data30M validation loss only",
            "held_out_access_during_training": False, "metrics": metrics,
        }
        write_json_atomic(run_dir / "summary.json", summary)
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
