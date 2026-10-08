"""Train locked Dynamic-logit-v1 using only the tied language-model logits."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.continuity_curriculum.common import canonical_json, sha256_bytes
from src.dynamic_logit_annotations import CATEGORY_PRECEDENCE
from src.mixed_schedule import ScheduledMixedDataset, verify_schedule
from src.model import FictionPulperLM, ModelConfig, model_config_from_dict
from src.pack_counterfactual_v1 import verify_counterfactual_gates
from src.train import evaluate, learning_rate_for_step, set_seed
from src.train_counterfactual import (
    LOCKED_CURRICULUM_TARGETS,
    LOCKED_LAMBDA,
    LOCKED_REAL_TARGETS,
    _contains_forbidden_key,
    _group_metrics,
    attached_pairs,
    load_train_pairs,
    source_token_losses,
    symmetric_pair_loss,
)
from src.train_mixed import (
    LOCKED_CHECKPOINT_STEPS,
    LOCKED_MODEL,
    LOCKED_PARAMETER_COUNT,
    LOCKED_SEED,
    LOCKED_STEPS,
    LOCKED_TOTAL_VALID_TARGETS,
    _packed_dataset,
    _validate_schedule_contract,
    _verify_schedule_sources,
    capture_rng_state,
    git_commit,
    tracked_worktree_dirty,
    validate_real_artifacts,
    validate_runtime_environment,
)
from src.train_tokenizer import sha256_file, write_json_atomic


LOCKED_DYNAMIC_LAMBDA = 0.25
LOCKED_STALE_LAMBDA = 0.10
EXPECTED_PARAMETER_COUNT = 15_047_040
DYNAMIC_MANIFEST_PATH = "data/dynamic_logit_v1/manifest.json"
DYNAMIC_TRAIN_PATH = "data/dynamic_logit_v1/train.jsonl"
DYNAMIC_VALIDATION_PATH = "data/dynamic_logit_v1/validation.jsonl"
STALE_CATEGORIES = frozenset({"PREVIOUS", "INITIAL", "OLDER"})
DECISION_KEY_FIELDS = ("pair", "world", "boundary", "depth")


def _record_hash(record: Mapping[str, Any]) -> str:
    unhashed = {key: value for key, value in record.items() if key != "stable_hash"}
    return sha256_bytes(canonical_json(unhashed).encode("utf-8"))


def group_candidate_records(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str, int], list[dict[str, Any]]]:
    """Form exact candidate sets and reject incomplete or ambiguous decisions."""
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for source in records:
        record = dict(source)
        key = (str(record["pair"]), str(record["world"]), str(record["boundary"]), int(record["depth"]))
        grouped[key].append(record)
    output = {}
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=lambda item: str(item["candidate"]))
        categories = [str(item["category"]) for item in candidates]
        names = [str(item["candidate"]) for item in candidates]
        if len(candidates) < 2 or len(set(names)) != len(names) or categories.count("CURRENT") != 1:
            raise RuntimeError(f"Invalid dynamic candidate set: {key}")
        if any(category not in CATEGORY_PRECEDENCE for category in categories):
            raise RuntimeError(f"Unknown dynamic taxonomy category: {key}")
        output[key] = candidates
    return output


def _read_and_verify_sidecar(path: Path, expected_sha256: str, split: str) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"Locked dynamic {split} sidecar changed")
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != split or record.get("stable_hash") != _record_hash(record):
                raise RuntimeError(f"Invalid dynamic {split} record at line {line_number}")
            records.append(record)
    return records


def load_dynamic_data(
    *, manifest_path: Path, train_path: Path, validation_path: Path,
    expected_manifest_sha256: str, expected_train_sha256: str,
    expected_validation_sha256: str, expected_tokenizer_sha256: str | None = None,
    expected_source_manifest_sha256: str | None = None,
    expected_transition_manifest_sha256: str | None = None,
    enforce_locked_paths: bool = True,
) -> tuple[dict[tuple[str, str, str, int], list[dict[str, Any]]], dict[tuple[str, str, str, int], list[dict[str, Any]]]]:
    """Hash-verify both trainer-visible splits and their manifest before parsing."""
    if enforce_locked_paths and (
        str(manifest_path), str(train_path), str(validation_path)
    ) != (DYNAMIC_MANIFEST_PATH, DYNAMIC_TRAIN_PATH, DYNAMIC_VALIDATION_PATH):
        raise RuntimeError("Locked Dynamic-logit-v1 paths changed")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("Locked dynamic manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "FictionPulper Dynamic-logit-v1":
        raise RuntimeError("Dynamic manifest dataset changed")
    locks = {
        "tokenizer_sha256": expected_tokenizer_sha256,
        "source_manifest_sha256": expected_source_manifest_sha256,
        "transition_manifest_sha256": expected_transition_manifest_sha256,
    }
    if any(value is not None and manifest.get(key) != value for key, value in locks.items()):
        raise RuntimeError("Dynamic annotation provenance lock changed")
    paths = {"train": (train_path, expected_train_sha256), "validation": (validation_path, expected_validation_sha256)}
    results = {}
    for split, (path, expected_hash) in paths.items():
        details = manifest.get("splits", {}).get(split, {})
        if details.get("file") != path.name or details.get("canonical_sha256") != expected_hash:
            raise RuntimeError(f"Dynamic {split} manifest lock is inconsistent")
        records = _read_and_verify_sidecar(path, expected_hash, split)
        if len(records) != int(details.get("record_count", -1)):
            raise RuntimeError(f"Dynamic {split} record count changed")
        results[split] = group_candidate_records(records)
    return results["train"], results["validation"]


def selected_pair_decisions(
    selected_pairs: Iterable[Mapping[str, Any]],
    grouped: Mapping[tuple[str, str, str, int], Sequence[Mapping[str, Any]]],
) -> list[list[dict[str, Any]]]:
    """Select all decisions for scheduled pairs in stable pair and key order."""
    pair_ids = [str(pair["pair_id"]) for pair in selected_pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise RuntimeError("A microbatch selected the same dynamic pair twice")
    available = {key[0] for key in grouped}
    missing = set(pair_ids) - available
    if missing:
        raise RuntimeError(f"Selected pair lacks dynamic decisions: {sorted(missing)}")
    wanted = set(pair_ids)
    return [[dict(record) for record in grouped[key]] for key in sorted(grouped) if key[0] in wanted]


def batched_candidate_scores(
    model: Any, records: Sequence[Mapping[str, Any]], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean exact-span log probabilities and teacher-forced token logits."""
    if not records:
        parameter = next(model.parameters())
        return parameter.sum().reshape(1)[:0], parameter.sum().reshape(1, 1, 1)[:0]
    lengths = [len(item["input_token_ids"]) - 1 for item in records]
    if min(lengths) < 1:
        raise ValueError("Dynamic candidate input is too short")
    inputs = torch.zeros(len(records), max(lengths), dtype=torch.long, device=device)
    for row, (record, length) in enumerate(zip(records, lengths, strict=True)):
        inputs[row, :length] = torch.tensor(record["input_token_ids"][:-1], device=device)
    logits, loss = model(inputs, labels=None)
    if loss is not None:
        raise RuntimeError("Dynamic scoring forward unexpectedly produced CE loss")
    scores = []
    for row, record in enumerate(records):
        start, end = int(record["decision_start"]), int(record["decision_end"])
        targets = torch.tensor(record["input_token_ids"][start:end], dtype=torch.long, device=device)
        if start < 1 or end <= start or end > len(record["input_token_ids"]):
            raise ValueError("Invalid dynamic decision span")
        token_logits = logits[row, start - 1:end - 1].float()
        scores.append(F.log_softmax(token_logits, -1).gather(1, targets[:, None]).mean())
    return torch.stack(scores), logits


def candidate_set_cross_entropy(scores: torch.Tensor, current_index: int) -> torch.Tensor:
    if scores.ndim != 1 or not 0 <= current_index < scores.numel():
        raise ValueError("Candidate scores/current index disagree")
    return -F.log_softmax(scores.float(), dim=0)[current_index]


def stable_token_unlikelihood(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Compute -log(1-p(token)) without subtracting a probability from one."""
    flat = logits.float().reshape(-1, logits.shape[-1])
    targets = token_ids.reshape(-1, 1)
    target_logits = flat.gather(1, targets).squeeze(1)
    masked = flat.scatter(1, targets, -torch.inf)
    return F.softplus(target_logits - torch.logsumexp(masked, dim=1)).mean()


def stale_unlikelihood_loss(
    logits: torch.Tensor, records: Sequence[Mapping[str, Any]], *, zero: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Penalize only historical stale candidates, never NEVER_VALID negatives."""
    losses = []
    for row, record in enumerate(records):
        if str(record["category"]) not in STALE_CATEGORIES:
            continue
        start, end = int(record["decision_start"]), int(record["decision_end"])
        targets = torch.tensor(record["input_token_ids"][start:end], dtype=torch.long, device=logits.device)
        losses.append(stable_token_unlikelihood(logits[row, start - 1:end - 1], targets))
    if losses:
        return torch.stack(losses).mean(), len(losses)
    anchor = zero if zero is not None else logits.sum()
    return anchor * 0.0, 0


def dynamic_objective(
    model: Any, decisions: Sequence[Sequence[Mapping[str, Any]]], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Score candidate sets directly through the LM, with no auxiliary CE labels."""
    if not decisions:
        zero = next(model.parameters()).sum() * 0.0
        return zero, zero, []
    flat = [record for decision in decisions for record in decision]
    scores, logits = batched_candidate_scores(model, flat, device)
    dynamic_losses, results = [], []
    offset = 0
    for decision in decisions:
        decision_scores = scores[offset:offset + len(decision)]
        current = next(index for index, item in enumerate(decision) if item["category"] == "CURRENT")
        dynamic_losses.append(candidate_set_cross_entropy(decision_scores, current))
        current_score = decision_scores[current]
        category_scores: dict[str, torch.Tensor] = {}
        for index, item in enumerate(decision):
            aliases = item.get("all_applicable_categories", [item["category"]])
            for category in aliases:
                if category != "CURRENT":
                    category_scores[str(category)] = decision_scores[index]
        result: dict[str, Any] = {
            "loss": float(dynamic_losses[-1].detach()),
            "current_correct": float(decision_scores.argmax().eq(torch.tensor(current, device=device)).detach()),
            "selected_category": str(decision[int(decision_scores.argmax())]["category"]),
            "family": str(decision[0]["family"]), "depth": str(decision[0]["depth"]),
            "distance": str(decision[0]["distance"]), "boundary": str(decision[0]["boundary"]),
        }
        for category in CATEGORY_PRECEDENCE[1:]:
            key = f"current_vs_{category.lower()}_margin"
            result[key] = float((current_score - category_scores[category]).detach()) if category in category_scores else None
        results.append(result)
        offset += len(decision)
    stale_loss, stale_count = stale_unlikelihood_loss(logits, flat, zero=scores.sum())
    for result in results:
        result["stale_candidate_count"] = stale_count
    return torch.stack(dynamic_losses).mean(), stale_loss, results


def aggregate_dynamic_metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    margin_keys = tuple(f"current_vs_{name.lower()}_margin" for name in CATEGORY_PRECEDENCE[1:])

    def summarize(values: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        summary: dict[str, float | int] = {
            "loss": sum(float(item["loss"]) for item in values) / len(values),
            "current_accuracy": sum(float(item["current_correct"]) for item in values) / len(values),
            "count": len(values),
        }
        for category in CATEGORY_PRECEDENCE:
            summary[f"selected_{category.lower()}_rate"] = sum(item["selected_category"] == category for item in values) / len(values)
        for key in margin_keys:
            present = [float(item[key]) for item in values if item.get(key) is not None]
            if present:
                summary[key] = sum(present) / len(present)
                summary[f"{key}_count"] = len(present)
        return summary

    output: dict[str, Any] = {"overall": summarize(results) if results else {"count": 0}}
    for field in ("selected_category", "family", "depth", "distance", "boundary"):
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in results:
            groups[str(item[field])].append(item)
        output[f"by_{field}"] = {key: summarize(value) for key, value in sorted(groups.items())}
    return output


@torch.no_grad()
def evaluate_dynamic_validation(
    model: Any, grouped: Mapping[tuple[str, str, str, int], Sequence[Mapping[str, Any]]],
    device: torch.device, *, batch_decisions: int = 32,
) -> dict[str, Any]:
    """Held-out diagnostic only; callers must not use this result for selection."""
    was_training = model.training
    model.eval()
    decisions = [grouped[key] for key in sorted(grouped)]
    results = []
    dynamic_sum = stale_sum = 0.0
    batches = 0
    for start in range(0, len(decisions), batch_decisions):
        dynamic, stale, batch_results = dynamic_objective(model, decisions[start:start + batch_decisions], device)
        dynamic_sum += float(dynamic) * len(batch_results)
        stale_sum += float(stale) * len(batch_results)
        batches += len(batch_results)
        results.extend(batch_results)
    if was_training:
        model.train()
    return {
        "dynamic_loss": dynamic_sum / batches if batches else 0.0,
        "stale_loss": stale_sum / batches if batches else 0.0,
        "metrics": aggregate_dynamic_metrics(results),
        "selection_eligible": False,
    }


def create_fresh_model(model_config: ModelConfig, *, device: torch.device | str = "cpu") -> FictionPulperLM:
    set_seed(LOCKED_SEED)
    model = FictionPulperLM(model_config).to(device)
    if LOCKED_PARAMETER_COUNT != EXPECTED_PARAMETER_COUNT or model.trainable_parameter_count() != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("Locked base model parameter count changed")
    if model.lm_head.weight is not model.embed_tokens.weight:
        raise RuntimeError("Dynamic-logit-v1 requires tied LM embeddings")
    return model


def validate_locked_config(config: dict[str, Any], *, require_generated_locks: bool = True) -> None:
    expected_id = "fictionpulper-15m-data30m-dynamic-logit-v1"
    expected_training = {
        "optimizer": "adamw", "batch_size": 16, "gradient_accumulation_steps": 4,
        "max_steps": LOCKED_STEPS, "learning_rate": 1e-3, "min_learning_rate": 1e-4,
        "warmup_steps": 111, "weight_decay": 0.1, "grad_clip": 1.0, "precision": "bf16",
        "epochs": 5, "validation_interval": 444, "checkpoint_steps": LOCKED_CHECKPOINT_STEPS,
        "require_clean_worktree": True, "checkpoint_dir": f"checkpoints/{expected_id}",
    }
    if config.get("experiment_id") != expected_id or config.get("logging", {}).get("run_dir") != f"runs/{expected_id}":
        raise RuntimeError("Dynamic-logit-v1 experiment/output paths changed")
    if int(config.get("seed", -1)) != LOCKED_SEED or config.get("model") != LOCKED_MODEL:
        raise RuntimeError("Locked Dynamic-logit-v1 model/seed changed")
    if {key: config["training"].get(key) for key in expected_training} != expected_training:
        raise RuntimeError("Locked Dynamic-logit-v1 optimizer/compute policy changed")
    sources = config.get("mixed_data", {}).get("training_sources", {})
    if set(sources) != {"real", "curriculum"} or int(sources["real"]["valid_targets"]) != LOCKED_REAL_TARGETS or int(sources["curriculum"]["valid_targets"]) != LOCKED_CURRICULUM_TARGETS:
        raise RuntimeError("Locked 85/15 exposure changed")
    if int(config["mixed_data"]["schedule"].get("total_chunks", -1)) != 142_080:
        raise RuntimeError("Locked schedule changed")
    counterfactual = config.get("counterfactual_objective", {})
    expected_counterfactual = {
        "version": "Counterfactual-v1", "lambda": LOCKED_LAMBDA,
        "score": "mean_log_probability_exact_candidate_decision_span",
        "loss": "symmetric_softplus_incorrect_minus_correct", "ranking_labels": None,
        "positive_ce_policy": "correct_worlds_only", "expected_pair_presentations": 41_238,
        "expected_candidate_span_world_triggers": 60_616, "expected_unique_pairs": 5_942,
        "diagnostic_train_pairs": 64,
    }
    if counterfactual != expected_counterfactual:
        raise RuntimeError("Locked Counterfactual-v1 objective changed")
    objective = config.get("dynamic_logit_objective", {})
    expected_objective = {
        "version": "Dynamic-logit-v1", "pair_lambda": LOCKED_LAMBDA,
        "dynamic_lambda": LOCKED_DYNAMIC_LAMBDA, "stale_lambda": LOCKED_STALE_LAMBDA,
        "score": "mean_log_probability_exact_decision_span",
        "dynamic_loss": "candidate_set_cross_entropy", "stale_loss": "teacher_forced_token_unlikelihood",
        "stale_categories": ["PREVIOUS", "INITIAL", "OLDER"], "labels": None,
        "auxiliary_heads": False,
    }
    if any(objective.get(key) != value for key, value in expected_objective.items()):
        raise RuntimeError("Locked Dynamic-logit-v1 objective changed")
    data = config.get("dynamic_logit_data", {})
    expected_paths = {
        "manifest_path": DYNAMIC_MANIFEST_PATH, "train_path": DYNAMIC_TRAIN_PATH,
        "validation_path": DYNAMIC_VALIDATION_PATH,
    }
    if any(data.get(key) != value for key, value in expected_paths.items()):
        raise RuntimeError("Locked dynamic annotation paths changed")
    artifact_paths = {
        ("tokenizer", "path"): "data/tokenizer/tokenizer.json",
        ("data", "train_path"): "data/packed/corpus-v2-data30m/train.bin",
        ("data", "validation_path"): "data/packed/corpus-v2-data30m/validation.bin",
        ("counterfactual_data", "directory"): "data/counterfactual_narrative_v1",
        ("counterfactual_data", "train_path"): "data/counterfactual_narrative_v1/train.jsonl",
        ("counterfactual_packing", "metadata_path"): "data/counterfactual_narrative_v1/packed/metadata.json",
    }
    if any(config[section].get(key) != expected for (section, key), expected in artifact_paths.items()):
        raise RuntimeError("Locked training artifact paths changed")
    if (
        sources["real"].get("path") != "data/packed/corpus-v2-data30m/train.bin"
        or sources["curriculum"].get("path") != "data/counterfactual_narrative_v1/packed/train.bin"
        or config["mixed_data"]["schedule"].get("path") != "data/counterfactual_narrative_v1/packed/mixed-source-schedule.json"
    ):
        raise RuntimeError("Locked mixed-source paths changed")
    if config.get("selection") != {
        "metric": "data30m_validation_loss",
        "dynamic_validation_role": "held_out_diagnostic_only_not_selection",
        "test_access": "forbidden_during_training", "generalization_access": "forbidden_during_training",
    }:
        raise RuntimeError("Checkpoint selection/access lock changed")
    if _contains_forbidden_key({"data": config["data"], "mixed_data": config["mixed_data"], "dynamic_logit_data": data}):
        raise RuntimeError("Trainer-visible configuration contains test/generalization access")
    if require_generated_locks:
        generated = [data.get(key) for key in (
            "expected_manifest_sha256", "expected_train_sha256", "expected_validation_sha256",
            "expected_transition_manifest_sha256",
        )]
        generated += [config["mixed_data"]["schedule"].get(key) for key in ("expected_file_sha256", "expected_content_sha256")]
        generated += [objective.get(key) for key in ("expected_decision_presentations", "expected_stale_candidate_presentations")]
        if any(value is None or str(value).startswith("PENDING_") for value in generated):
            raise RuntimeError("Generated Dynamic-logit-v1 locks are pending")


def _save_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_checkpoint(
    path: Path, *, model: FictionPulperLM, optimizer: torch.optim.Optimizer,
    config: dict[str, Any], step: int, locks: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]],
) -> None:
    if model.trainable_parameter_count() != EXPECTED_PARAMETER_COUNT or model.lm_head.weight is not model.embed_tokens.weight:
        raise RuntimeError("Refusing checkpoint with changed direct-LM architecture")
    _save_atomic(path, {
        "format": "fictionpulper-dynamic-logit-v1", "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": {
            "step": step, "learning_rate": optimizer.param_groups[0]["lr"],
            "policy": "linear warmup then cosine decay", "max_steps": LOCKED_STEPS,
            "warmup_steps": 111, "maximum_learning_rate": 1e-3, "minimum_learning_rate": 1e-4,
        },
        "rng_state": capture_rng_state(), "model_config": asdict(model.config), "config": config,
        "locks": dict(locks), "metrics": list(metrics), "step": step,
        "model_initialization_seed": LOCKED_SEED, "fresh_random_initialization": True,
        "resume_checkpoint": None, "selection_metric": "Data30M validation loss only",
        "auxiliary_heads": False,
    })


def run_training(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_locked_config(config)
    if config["training"]["require_clean_worktree"] and tracked_worktree_dirty() is not False:
        raise RuntimeError("Locked training requires a clean tracked Git worktree")
    validate_runtime_environment()
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
    schedule_path = Path(config["mixed_data"]["schedule"]["path"])
    if sha256_file(schedule_path) != config["mixed_data"]["schedule"]["expected_file_sha256"]:
        raise RuntimeError("Locked schedule file changed")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule_hash = verify_schedule(schedule)
    if schedule_hash != config["mixed_data"]["schedule"]["expected_content_sha256"]:
        raise RuntimeError("Locked schedule content changed")
    _validate_schedule_contract(config, schedule)
    _verify_schedule_sources(schedule)
    pairs = load_train_pairs(Path(config["counterfactual_data"]["train_path"]), config["counterfactual_data"]["expected_train_sha256"])
    dynamic_config = config["dynamic_logit_data"]
    train_dynamic, validation_dynamic = load_dynamic_data(
        manifest_path=Path(dynamic_config["manifest_path"]), train_path=Path(dynamic_config["train_path"]),
        validation_path=Path(dynamic_config["validation_path"]), expected_manifest_sha256=dynamic_config["expected_manifest_sha256"],
        expected_train_sha256=dynamic_config["expected_train_sha256"], expected_validation_sha256=dynamic_config["expected_validation_sha256"],
        expected_tokenizer_sha256=tokenizer_hash,
        expected_source_manifest_sha256=config["counterfactual_data"]["expected_manifest_sha256"],
        expected_transition_manifest_sha256=dynamic_config["expected_transition_manifest_sha256"],
    )
    sequence_length = int(config["model"]["max_seq_len"])
    sources = {name: _packed_dataset(item["path"], sequence_length, int(pad_id)) for name, item in config["mixed_data"]["training_sources"].items()}
    dataset = ScheduledMixedDataset(sources, schedule)
    validation = _packed_dataset(config["data"]["validation_path"], sequence_length, int(pad_id))
    model = create_fresh_model(model_config_from_dict(config["model"]), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1, fused=True)
    run_dir, checkpoint_dir = Path(config["logging"]["run_dir"]), Path(config["training"]["checkpoint_dir"])
    if run_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Dynamic-logit-v1 output exists; resume is forbidden")
    run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    locks = {
        "tokenizer_sha256": tokenizer_hash, "schedule_file_sha256": sha256_file(schedule_path),
        "schedule_content_sha256": schedule_hash, "dynamic_manifest_sha256": dynamic_config["expected_manifest_sha256"],
        "dynamic_train_sha256": dynamic_config["expected_train_sha256"],
        "dynamic_validation_sha256": dynamic_config["expected_validation_sha256"],
        "counterfactual_train_sha256": config["counterfactual_data"]["expected_train_sha256"],
    }
    write_json_atomic(run_dir / "manifest.json", {
        "experiment_id": config["experiment_id"], "status": "training_started", "started_at": datetime.now().isoformat(),
        "training_git_commit": git_commit(), "fresh_random_initialization": True, "resume_checkpoint": None,
        "model_parameter_count": model.trainable_parameter_count(), "auxiliary_heads": False, "locks": locks,
        "dataset_gates": gates, "real_artifacts": real_artifacts, "selection_metric": "Data30M validation loss only",
        "held_out_dynamic_validation_selection_eligible": False,
    })
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    optimizer.zero_grad(set_to_none=True)
    step = total_targets = total_pairs = total_decisions = total_stale = 0
    source_targets: dict[str, int] = defaultdict(int)
    interval: dict[str, float] = defaultdict(float)
    pair_results: list[dict[str, Any]] = []
    dynamic_results: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = [{"step": 0, "selection_eligible": False, "data30m_validation": evaluate(model, validation, batch_size=16, device=device, use_bf16=True)}]
    best_loss, best_step = math.inf, 0
    started = interval_started = time.monotonic()
    try:
        for microbatch, (input_ids, labels) in enumerate(loader, 1):
            input_ids, labels = input_ids.to(device), labels.to(device)
            entries = schedule["entries"][(microbatch - 1) * 16:(microbatch - 1) * 16 + input_ids.shape[0]]
            valid_targets = int(labels.ne(-100).sum())
            if valid_targets != sum(int(entry["valid_targets"]) for entry in entries):
                raise RuntimeError("Observed CE targets differ from schedule")
            selected = attached_pairs(entries, pairs)
            decisions = selected_pair_decisions(selected, train_dynamic)
            if all(entry["source"] == "real" for entry in entries) and (selected or decisions):
                raise RuntimeError("Real-only microbatch acquired curriculum attachments")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, ce_loss = model(input_ids, labels)
                pair_outputs = [symmetric_pair_loss(model, pair, device) for pair in selected]
                pair_loss = torch.stack([item[0] for item in pair_outputs]).mean() if pair_outputs else logits.sum() * 0.0
                dynamic_loss, stale_loss, current_results = dynamic_objective(model, decisions, device)
                objective = ce_loss + LOCKED_LAMBDA * pair_loss + LOCKED_DYNAMIC_LAMBDA * dynamic_loss + LOCKED_STALE_LAMBDA * stale_loss  # type: ignore[operator]
            if ce_loss is None or not torch.isfinite(objective):
                raise FloatingPointError("Dynamic-logit-v1 objective is NaN or Inf")
            for source, (loss_sum, count) in source_token_losses(logits, labels, entries).items():
                interval[f"{source}_ce_sum"] += float(loss_sum.detach())
                interval[f"{source}_targets"] += count
                source_targets[source] += count
            for name, value in (("objective_raw", objective), ("ce_raw", ce_loss), ("pair_raw", pair_loss), ("dynamic_raw", dynamic_loss), ("stale_raw", stale_loss)):
                interval[f"{name}_sum"] += float(value.detach())
            interval["pair_weighted_sum"] += LOCKED_LAMBDA * float(pair_loss.detach())
            interval["dynamic_weighted_sum"] += LOCKED_DYNAMIC_LAMBDA * float(dynamic_loss.detach())
            interval["stale_weighted_sum"] += LOCKED_STALE_LAMBDA * float(stale_loss.detach())
            interval["ce_weighted_sum"] += float(ce_loss.detach())
            interval["microbatches"] += 1
            interval["targets"] += valid_targets
            interval["samples"] += input_ids.shape[0]
            pair_results.extend(item[1] for item in pair_outputs)
            dynamic_results.extend(current_results)
            total_targets += valid_targets
            total_pairs += len(pair_outputs)
            total_decisions += len(current_results)
            total_stale += sum(sum(item["category"] in STALE_CATEGORIES for item in decision) for decision in decisions)
            (objective / 4).backward()
            if microbatch % 4:
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
            if step not in set(LOCKED_CHECKPOINT_STEPS):
                continue
            validation_metrics = evaluate(model, validation, batch_size=16, device=device, use_bf16=True)
            dynamic_validation = evaluate_dynamic_validation(model, validation_dynamic, device)
            elapsed = time.monotonic() - interval_started
            loss_names = ("objective_raw", "ce_raw", "ce_weighted", "pair_raw", "pair_weighted", "dynamic_raw", "dynamic_weighted", "stale_raw", "stale_weighted")
            metric = {
                "step": step, "selection_eligible": True, "data30m_validation": validation_metrics,
                "dynamic_validation_diagnostic": dynamic_validation,
                "loss_scales": {name: interval[f"{name}_sum"] / interval["microbatches"] for name in loss_names},
                "real_ce_loss": interval["real_ce_sum"] / interval["real_targets"],
                "curriculum_ce_loss": interval["curriculum_ce_sum"] / interval["curriculum_targets"],
                "ranking": _group_metrics(pair_results), "dynamic": aggregate_dynamic_metrics(dynamic_results),
                "learning_rate": lr, "valid_targets_per_second": interval["targets"] / elapsed,
                "samples_per_second": interval["samples"] / elapsed,
                "gradient_norm_mean": interval["grad_norm_sum"] / interval["steps"], "gradient_norm_max": interval["grad_norm_max"],
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
                "exposure": {"real_targets": source_targets["real"], "curriculum_targets": source_targets["curriculum"], "pair_evals": total_pairs, "dynamic_decisions": total_decisions, "stale_candidates": total_stale},
                "elapsed_seconds": time.monotonic() - started,
            }
            metrics.append(metric)
            for name, value in metric["loss_scales"].items():
                writer.add_scalar(f"loss_scale/{name}", value, step)
            writer.add_scalar("validation/data30m_loss", validation_metrics["loss"], step)
            writer.add_scalar("diagnostic/dynamic_validation_loss", dynamic_validation["dynamic_loss"], step)
            write_json_atomic(run_dir / "metrics.json", {"checkpoints": metrics})
            candidate_best = float(validation_metrics["loss"]) < best_loss
            if candidate_best:
                best_loss, best_step = float(validation_metrics["loss"]), step
            names = ("latest.pt", f"step-{step:04d}.pt") + (("best-validation.pt",) if candidate_best else ())
            for name in names:
                save_checkpoint(checkpoint_dir / name, model=model, optimizer=optimizer, config=config, step=step, locks=locks, metrics=metrics)
            interval, pair_results, dynamic_results = defaultdict(float), [], []
            interval_started = time.monotonic()
        expected_sources = {name: int(item["valid_targets"]) for name, item in schedule["sources"].items()}
        objective_config = config["dynamic_logit_objective"]
        if (
            step != LOCKED_STEPS or total_targets != LOCKED_TOTAL_VALID_TARGETS or dict(source_targets) != expected_sources
            or total_pairs != int(config["counterfactual_objective"]["expected_pair_presentations"])
            or total_decisions != int(objective_config["expected_decision_presentations"])
            or total_stale != int(objective_config["expected_stale_candidate_presentations"])
        ):
            raise RuntimeError("Final locked exposure does not match schedule")
        best = checkpoint_dir / "best-validation.pt"
        summary = {
            "status": "training_and_selection_complete", "optimizer_steps": step,
            "model_parameters": model.trainable_parameter_count(), "auxiliary_heads": False,
            "exposure": {"real_targets": source_targets["real"], "curriculum_targets": source_targets["curriculum"], "pair_evals": total_pairs, "dynamic_decisions": total_decisions, "stale_candidates": total_stale},
            "best_data30m_validation_loss": best_loss, "best_checkpoint_step": best_step,
            "best_checkpoint": str(best), "best_checkpoint_sha256": sha256_file(best),
            "selection_metric": "Data30M validation loss only", "metrics": metrics,
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
