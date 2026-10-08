"""Train locked State-Transition-v1 with LM, pair-ranking, and state losses."""

from __future__ import annotations

import argparse
import copy
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
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.mixed_schedule import ScheduledMixedDataset, verify_schedule
from src.model import FictionPulperLM, ModelConfig, model_config_from_dict
from src.pack_counterfactual_v1 import verify_counterfactual_gates
from src.state_transition_annotations import validate_annotations
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


LOCKED_STATE_LAMBDA = 0.25
REPRESENTATION = "final_hidden_after_final_rmsnorm"
STATE_MANIFEST_PATH = "data/state_transition_v1/manifest.json"
STATE_TRAIN_PATH = "data/state_transition_v1/train.jsonl"

# These maps are deliberately fixed rather than inferred from the sidecar. A changed
# lexical inventory must therefore become a new training protocol.
CANONICAL_CLASS_MAPS: dict[str, dict[str, int]] = {
    "object_owner": {f"character_{index}": index for index in range(2)},
    "object_location": {f"location_{index}": index for index in range(2)},
    "character_location": {f"location_{index}": index for index in range(2)},
    "goal_status": {f"goal_{index}": index for index in range(2)},
    "knowledge_holder": {f"character_{index}": index for index in range(2)},
    "relationship_state": {"relationship_trusted_ally": 0, "relationship_declared_rival": 1},
    "physical_condition": {"physical_affected": 0, "physical_recovered": 1},
    "object_door_state": {"door_locked": 0, "door_open": 1},
    "causal_state": {f"cause_{index}": index for index in range(2)},
}


def head_config() -> dict[str, Any]:
    return {
        "type": "linear",
        "representation": REPRESENTATION,
        "class_maps": copy.deepcopy(CANONICAL_CLASS_MAPS),
    }


def build_state_heads(hidden_size: int) -> nn.ModuleDict:
    """Build training-only linear classifiers in canonical family order."""
    return nn.ModuleDict({
        family: nn.Linear(hidden_size, len(classes))
        for family, classes in CANONICAL_CLASS_MAPS.items()
    })


def parameter_counts(model: FictionPulperLM, heads: nn.ModuleDict) -> dict[str, int]:
    base = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    auxiliary = sum(parameter.numel() for parameter in heads.parameters() if parameter.requires_grad)
    return {"base": base, "auxiliary": auxiliary, "total_training": base + auxiliary}


def create_fresh_model_and_heads(
    model_config: ModelConfig, *, device: torch.device | str = "cpu"
) -> tuple[FictionPulperLM, nn.ModuleDict]:
    """Create the only permitted initialization, deterministically from seed 1337."""
    set_seed(LOCKED_SEED)
    model = FictionPulperLM(model_config).to(device)
    if model.trainable_parameter_count() != LOCKED_PARAMETER_COUNT:
        raise RuntimeError("Locked base model parameter count changed")
    heads = build_state_heads(model_config.hidden_size).to(device)
    return model, heads


def _contains_subsequence(values: Sequence[int], needle: Sequence[int]) -> bool:
    return bool(needle) and any(
        list(values[index:index + len(needle)]) == list(needle)
        for index in range(len(values) - len(needle) + 1)
    )


def _record_hash(record: Mapping[str, Any]) -> str:
    from src.continuity_curriculum.common import canonical_json, sha256_bytes

    unhashed = {key: value for key, value in record.items() if key != "stable_hash"}
    return sha256_bytes(canonical_json(unhashed).encode("utf-8"))


def load_state_training_data(
    *, manifest_path: Path, train_path: Path, expected_manifest_sha256: str,
    expected_train_sha256: str, pairs: Mapping[str, dict[str, Any]], tokenizer: Tokenizer,
    expected_tokenizer_sha256: str | None = None, enforce_locked_paths: bool = True,
    expected_source_manifest_sha256: str | None = None,
    expected_source_train_sha256: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load and prove the train-only sidecar against its locked source pairs."""
    if enforce_locked_paths and (str(manifest_path) != STATE_MANIFEST_PATH or str(train_path) != STATE_TRAIN_PATH):
        raise RuntimeError("Locked State-Transition-v1 paths changed")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("Locked state-transition manifest changed")
    if sha256_file(train_path) != expected_train_sha256:
        raise RuntimeError("Locked state-transition train sidecar changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = manifest.get("splits", {}).get("train", {})
    if (
        manifest.get("dataset") != "FictionPulper State-Transition-v1"
        or (expected_tokenizer_sha256 is not None and manifest.get("tokenizer_sha256") != expected_tokenizer_sha256)
        or split.get("file") != "train.jsonl"
        or split.get("canonical_sha256") != expected_train_sha256
    ):
        raise RuntimeError("State-transition manifest lock is inconsistent")
    expected_classes = {
        family: sorted(classes)
        for family, classes in sorted(CANONICAL_CLASS_MAPS.items())
    }
    if manifest.get("state_classes") != expected_classes:
        raise RuntimeError("State-transition class inventory changed")
    source_hashes = manifest.get("source_hashes", {})
    if expected_source_manifest_sha256 is not None and source_hashes.get("manifest.json") != expected_source_manifest_sha256:
        raise RuntimeError("State-transition source manifest lock changed")
    if expected_source_train_sha256 is not None and source_hashes.get("train.jsonl") != expected_source_train_sha256:
        raise RuntimeError("State-transition source train lock changed")
    records: list[dict[str, Any]] = []
    with train_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != "train" or record.get("stable_hash") != _record_hash(record):
                raise RuntimeError(f"Invalid train sidecar record at line {line_number}")
            family = str(record.get("family"))
            if family not in CANONICAL_CLASS_MAPS or record.get("post_state") not in CANONICAL_CLASS_MAPS[family]:
                raise RuntimeError(f"Non-canonical state class at line {line_number}")
            records.append(record)
    if len(records) != int(split.get("record_count", -1)):
        raise RuntimeError("State-transition train record count changed")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["pair"])].append(record)
    if set(grouped) - set(pairs):
        raise RuntimeError("State sidecar references an absent train pair")
    for pair_id, annotations in grouped.items():
        validate_annotations(pairs[pair_id], annotations, tokenizer)
        for record in annotations:
            label_ids = tokenizer.encode(str(record["post_state"])).ids
            world_ids = pairs[pair_id]["worlds"][record["world"]]["context_token_ids"]
            if _contains_subsequence(world_ids, label_ids):
                raise RuntimeError("State label entered the model token stream")
    transitions = sum(record["kind"] == "transition" for record in records)
    carries = sum(record["kind"] == "carry" for record in records)
    if transitions != int(split["transitions"]["count"]) or carries != int(split["carries"]["count"]):
        raise RuntimeError("State-transition kind counts changed")
    return dict(grouped)


def state_examples_for_pairs(
    selected_pairs: Iterable[Mapping[str, Any]], annotations: Mapping[str, list[dict[str, Any]]],
    tokenizer: Tokenizer,
) -> list[dict[str, Any]]:
    """Create label-free model inputs and exact post-token representation positions."""
    examples = []
    for pair in selected_pairs:
        pair_id = str(pair["pair_id"])
        by_world_entity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in annotations.get(pair_id, []):
            by_world_entity[(str(record["world"]), str(record["entity_slot"]))].append(record)
        for (world, entity), records in sorted(by_world_entity.items()):
            control = pair["metadata"].get("genre_control_token")
            prefix_tokens = [tokenizer.token_to_id("<|story|>")]
            if control:
                prefix_tokens.append(tokenizer.token_to_id(control))
            prefix_tokens.append(tokenizer.token_to_id("<|bos|>"))
            if any(token is None for token in prefix_tokens):
                raise RuntimeError("State example uses an unknown control token")
            context_ids = [int(value) for value in pair["worlds"][world]["context_token_ids"]]
            token_ids = [int(value) for value in prefix_tokens] + context_ids
            if len(token_ids) > 1024:
                raise RuntimeError("State example exceeds the locked context length")
            prepared = []
            for record in records:
                boundary = int(record["prediction_token_index"])
                if not 1 <= boundary <= len(context_ids):
                    raise RuntimeError("State prediction boundary is outside its context")
                prepared.append({
                    **record,
                    "class_index": CANONICAL_CLASS_MAPS[str(record["family"])][str(record["post_state"])],
                    "hidden_position": len(prefix_tokens) + boundary - 1,
                })
            examples.append({
                "example_id": f"{pair_id}:{world}:{entity}", "pair": pair_id, "world": world,
                "entity_slot": entity, "token_ids": token_ids, "annotations": prepared,
            })
    return examples


def normalized_state_loss(
    final_hidden: torch.Tensor, examples: Sequence[Mapping[str, Any]], heads: nn.ModuleDict,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Mean annotations per world/entity sequence, then weight examples equally."""
    if final_hidden.ndim != 3 or final_hidden.shape[0] != len(examples):
        raise ValueError("Hidden-state batch and state examples disagree")
    example_losses = []
    results: list[dict[str, Any]] = []
    for row, example in enumerate(examples):
        annotation_losses = []
        for record in example["annotations"]:
            position = int(record["hidden_position"])
            if not 0 <= position < final_hidden.shape[1]:
                raise ValueError("State hidden position is outside the input")
            family = str(record["family"])
            target = torch.tensor([int(record["class_index"])], device=final_hidden.device)
            logits = heads[family](final_hidden[row, position].float()).unsqueeze(0)
            loss = F.cross_entropy(logits, target)
            annotation_losses.append(loss)
            results.append({
                "loss": float(loss.detach()), "correct": float(logits.argmax(-1).eq(target).item()),
                "kind": str(record["kind"]), "family": family,
                "distance_bucket": str(record["metadata"]["distance_bucket"]),
                "difficulty": str(record["metadata"]["difficulty"]),
                "genre": str(record["metadata"]["genre"]),
                "carry_distance": str(record.get("distance", "transition")),
            })
        if not annotation_losses:
            raise ValueError("A state example cannot have zero annotations")
        example_losses.append(torch.stack(annotation_losses).mean())
    if not example_losses:
        return final_hidden.sum() * 0.0, []
    return torch.stack(example_losses).mean(), results


def state_loss_for_examples(
    model: FictionPulperLM, heads: nn.ModuleDict, examples: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if not examples:
        return next(model.parameters()).sum() * 0.0, []
    max_length = max(len(example["token_ids"]) for example in examples)
    pad = torch.zeros(len(examples), max_length, dtype=torch.long, device=device)
    for row, example in enumerate(examples):
        ids = torch.tensor(example["token_ids"], dtype=torch.long, device=device)
        pad[row, :len(ids)] = ids
    output = model(pad, labels=None, return_hidden_states=True)
    _, loss, hidden_states = output
    if loss is not None:
        raise RuntimeError("State forward unexpectedly received labels")
    return normalized_state_loss(hidden_states[-1], examples, heads)


def group_state_metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def summarize(values: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        return {
            "ce": sum(float(item["loss"]) for item in values) / len(values),
            "accuracy": sum(float(item["correct"]) for item in values) / len(values),
            "count": len(values),
        }
    output: dict[str, Any] = {"overall": summarize(results) if results else {"count": 0}}
    for field in ("kind", "family", "distance_bucket", "difficulty", "genre", "carry_distance"):
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for result in results:
            groups[str(result[field])].append(result)
        output[f"by_{field}"] = {key: summarize(value) for key, value in sorted(groups.items())}
    for kind in ("transition", "carry"):
        selected = [item for item in results if item["kind"] == kind]
        output[kind] = summarize(selected) if selected else {"count": 0}
        for field in ("family", "distance_bucket", "difficulty", "genre"):
            groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for result in selected:
                groups[str(result[field])].append(result)
            output[f"{kind}_by_{field}"] = {
                key: summarize(value) for key, value in sorted(groups.items())
            }
    carry_distances: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        if result["kind"] == "carry":
            carry_distances[str(result["carry_distance"])].append(result)
    output["carry_by_distance"] = {
        key: summarize(value) for key, value in sorted(carry_distances.items())
    }
    return output


def validate_locked_config(config: dict[str, Any], *, require_generated_locks: bool = True) -> None:
    """Reject any drift from Counterfactual-v1 except isolated output paths."""
    expected_training = {
        "optimizer": "adamw", "batch_size": 16, "gradient_accumulation_steps": 4,
        "max_steps": LOCKED_STEPS, "learning_rate": 1e-3, "min_learning_rate": 1e-4,
        "warmup_steps": 111, "weight_decay": 0.1, "grad_clip": 1.0, "precision": "bf16",
        "epochs": 5, "validation_interval": 444, "checkpoint_steps": LOCKED_CHECKPOINT_STEPS,
        "require_clean_worktree": True,
        "checkpoint_dir": "checkpoints/fictionpulper-15m-data30m-state-transition-v1",
    }
    if int(config.get("seed", -1)) != LOCKED_SEED or config.get("model") != LOCKED_MODEL:
        raise RuntimeError("Locked State-Transition-v1 model/seed changed")
    if {key: config["training"].get(key) for key in expected_training} != expected_training:
        raise RuntimeError("Locked State-Transition-v1 optimizer/compute policy changed")
    if config.get("experiment_id") != "fictionpulper-15m-data30m-state-transition-v1" or config.get("logging", {}).get("run_dir") != "runs/fictionpulper-15m-data30m-state-transition-v1":
        raise RuntimeError("State-Transition-v1 output paths changed")
    sources = config.get("mixed_data", {}).get("training_sources", {})
    if set(sources) != {"real", "curriculum"} or int(sources["real"]["valid_targets"]) != LOCKED_REAL_TARGETS or int(sources["curriculum"]["valid_targets"]) != LOCKED_CURRICULUM_TARGETS:
        raise RuntimeError("Locked 85/15 exposure changed")
    schedule = config["mixed_data"]["schedule"]
    if int(schedule.get("total_chunks", -1)) != 142_080:
        raise RuntimeError("Locked schedule changed")
    objective = config.get("state_transition_objective", {})
    expected_objective = {
        "version": "State-Transition-v1", "pair_lambda": LOCKED_LAMBDA,
        "state_lambda": LOCKED_STATE_LAMBDA, "head_type": "linear",
        "representation": REPRESENTATION, "normalization": "mean_per_world_entity_then_mean_examples",
        "labels_in_input": False,
    }
    if any(objective.get(key) != value for key, value in expected_objective.items()):
        raise RuntimeError("Locked State-Transition-v1 objective changed")
    counterfactual_objective = config.get("counterfactual_objective", {})
    expected_counterfactual = {
        "version": "Counterfactual-v1", "lambda": LOCKED_LAMBDA,
        "score": "mean_log_probability_exact_candidate_decision_span",
        "loss": "symmetric_softplus_incorrect_minus_correct",
        "ranking_labels": None, "positive_ce_policy": "correct_worlds_only",
        "expected_pair_presentations": 41_238,
        "expected_candidate_span_world_triggers": 60_616,
        "expected_unique_pairs": 5_942, "diagnostic_train_pairs": 64,
    }
    if counterfactual_objective != expected_counterfactual:
        raise RuntimeError("Locked Counterfactual-v1 objective changed")
    state_data = config.get("state_transition_data", {})
    if state_data.get("manifest_path") != STATE_MANIFEST_PATH or state_data.get("train_path") != STATE_TRAIN_PATH:
        raise RuntimeError("Locked state-transition paths changed")
    expected_paths = {
        ("tokenizer", "path"): "data/tokenizer/tokenizer.json",
        ("data", "train_path"): "data/packed/corpus-v2-data30m/train.bin",
        ("data", "validation_path"): "data/packed/corpus-v2-data30m/validation.bin",
        ("counterfactual_data", "directory"): "data/counterfactual_narrative_v1",
        ("counterfactual_data", "train_path"): "data/counterfactual_narrative_v1/train.jsonl",
        ("counterfactual_packing", "metadata_path"): "data/counterfactual_narrative_v1/packed/metadata.json",
        ("mixed_data", "schedule"): None,
    }
    for (section, key), expected in expected_paths.items():
        observed = config[section][key]
        if section == "mixed_data":
            observed = observed["path"]
            expected = "data/counterfactual_narrative_v1/packed/mixed-source-schedule.json"
        if observed != expected:
            raise RuntimeError("Locked training artifact paths changed")
    if sources["real"].get("path") != "data/packed/corpus-v2-data30m/train.bin" or sources["curriculum"].get("path") != "data/counterfactual_narrative_v1/packed/train.bin":
        raise RuntimeError("Locked mixed-source paths changed")
    selection = config.get("selection")
    if selection != {
        "metric": "data30m_validation_loss",
        "counterfactual_validation_role": "packing_validation_only_not_loaded_by_trainer",
        "test_access": "forbidden_during_training",
        "generalization_access": "forbidden_during_training",
    }:
        raise RuntimeError("Checkpoint selection/access lock changed")
    if _contains_forbidden_key({"data": config["data"], "mixed_data": config["mixed_data"], "state_transition_data": state_data}):
        raise RuntimeError("Trainer-visible configuration contains held-out access")
    if require_generated_locks:
        hashes = [
            schedule.get("expected_file_sha256"), schedule.get("expected_content_sha256"),
            state_data.get("expected_manifest_sha256"), state_data.get("expected_train_sha256"),
        ]
        if any(not value or str(value).startswith("PENDING_") for value in hashes):
            raise RuntimeError("Generated State-Transition-v1 locks are pending")


def preflight_numerical_sanity(
    model: FictionPulperLM, heads: nn.ModuleDict, examples: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    """Prove state loss reaches both its linear heads and the base transformer."""
    model.zero_grad(set_to_none=True)
    heads.zero_grad(set_to_none=True)
    loss, _ = state_loss_for_examples(model, heads, examples, device)
    loss.backward()
    head_norm = sum(float(parameter.grad.float().norm()) for parameter in heads.parameters() if parameter.grad is not None)
    base_norm = sum(float(parameter.grad.float().norm()) for parameter in model.parameters() if parameter.grad is not None)
    if not math.isfinite(float(loss.detach())) or head_norm <= 0.0 or base_norm <= 0.0:
        raise RuntimeError("State-loss gradient preflight failed")
    model.zero_grad(set_to_none=True)
    heads.zero_grad(set_to_none=True)
    return {"state_loss": float(loss.detach()), "head_gradient_norm_sum": head_norm, "base_gradient_norm_sum": base_norm}


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
    path: Path, *, model: FictionPulperLM, state_heads: nn.ModuleDict,
    optimizer: torch.optim.Optimizer, config: dict[str, Any], step: int,
    locks: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]],
    expected_base_parameter_count: int = LOCKED_PARAMETER_COUNT,
) -> None:
    counts = parameter_counts(model, state_heads)
    if counts["base"] != expected_base_parameter_count:
        raise RuntimeError("Refusing checkpoint with changed base parameter count")
    _save_atomic(path, {
        "format": "fictionpulper-state-transition-v1", "model": model.state_dict(),
        "state_heads": state_heads.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": {
            "step": int(step), "learning_rate": optimizer.param_groups[0]["lr"],
            "policy": "linear warmup then cosine decay", "max_steps": LOCKED_STEPS,
            "warmup_steps": 111, "maximum_learning_rate": 1e-3,
            "minimum_learning_rate": 1e-4,
        },
        "rng_state": capture_rng_state(), "model_config": asdict(model.config),
        "head_config": head_config(), "config": config, "locks": dict(locks),
        "metrics": list(metrics), "step": int(step), "parameter_counts": counts,
        "model_initialization_seed": LOCKED_SEED, "fresh_random_initialization": True,
        "resume_checkpoint": None, "selection_metric": "Data30M validation loss only",
    })


def load_checkpoint_for_validation(
    path: Path, *, model: FictionPulperLM, state_heads: nn.ModuleDict,
    optimizer: torch.optim.Optimizer | None = None, expected_locks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly validate and load a checkpoint for inspection, never training resume."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "fictionpulper-state-transition-v1" or payload.get("head_config") != head_config():
        raise RuntimeError("State-transition checkpoint head contract changed")
    if payload.get("model_initialization_seed") != LOCKED_SEED or payload.get("resume_checkpoint") is not None:
        raise RuntimeError("State-transition checkpoint provenance changed")
    if payload.get("parameter_counts") != parameter_counts(model, state_heads):
        raise RuntimeError("State-transition checkpoint parameter accounting changed")
    if payload.get("model_config") != asdict(model.config) or payload.get("fresh_random_initialization") is not True:
        raise RuntimeError("State-transition checkpoint model contract changed")
    if expected_locks is not None and payload.get("locks") != dict(expected_locks):
        raise RuntimeError("State-transition checkpoint artifact locks changed")
    model.load_state_dict(payload["model"], strict=True)
    state_heads.load_state_dict(payload["state_heads"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def export_lm_only(checkpoint_path: Path, output_path: Path) -> dict[str, Any]:
    """Export the standard head-free model payload consumed by LM tooling."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "fictionpulper-state-transition-v1" or "state_heads" not in payload:
        raise RuntimeError("Not a State-Transition-v1 training checkpoint")
    exported = {
        "model": payload["model"], "model_config": payload["model_config"],
        "config": payload["config"], "step": payload["step"],
        "tokenizer_hash": payload.get("locks", {}).get("tokenizer_sha256"),
        "selection_metric": payload["selection_metric"],
        "state_transition_training_source": str(checkpoint_path),
    }
    _save_atomic(output_path, exported)
    return exported


@torch.no_grad()
def check_lm_only_equivalence(
    model: FictionPulperLM, export_path: Path, input_ids: torch.Tensor,
    *, generation_steps: int = 4,
) -> dict[str, Any]:
    """Prove export logits and greedy tokens equal the in-memory LM with inactive heads."""
    payload = torch.load(export_path, map_location="cpu", weights_only=False)
    exported = FictionPulperLM(model_config_from_dict(payload["model_config"]))
    exported.load_state_dict(payload["model"], strict=True)
    source = copy.deepcopy(model).cpu().eval()
    exported.eval()
    tokens_a, tokens_b = input_ids.cpu().clone(), input_ids.cpu().clone()
    logits_a, _ = source(tokens_a)
    logits_b, _ = exported(tokens_b)
    logits_equal = torch.equal(logits_a, logits_b)
    generated_a: list[int] = []
    generated_b: list[int] = []
    for _ in range(generation_steps):
        out_a, _ = source(tokens_a)
        out_b, _ = exported(tokens_b)
        next_a = out_a[:, -1].argmax(-1, keepdim=True)
        next_b = out_b[:, -1].argmax(-1, keepdim=True)
        generated_a.append(int(next_a[0, 0]))
        generated_b.append(int(next_b[0, 0]))
        tokens_a, tokens_b = torch.cat((tokens_a, next_a), 1), torch.cat((tokens_b, next_b), 1)
    if not logits_equal or generated_a != generated_b:
        raise RuntimeError("LM-only export is not numerically equivalent")
    return {"logits_identical": True, "generation_identical": True, "generated_ids": generated_a}


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
        raise RuntimeError("Locked schedule file changed")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule_hash = verify_schedule(schedule)
    if schedule_hash != config["mixed_data"]["schedule"]["expected_content_sha256"]:
        raise RuntimeError("Locked schedule content changed")
    _validate_schedule_contract(config, schedule)
    _verify_schedule_sources(schedule)
    expected_attachment = {
        "pair_presentations": 41_238,
        "candidate_span_world_triggers": 60_616,
        "unique_pairs": 5_942,
    }
    if any(int(schedule["pair_attachment"][key]) != value for key, value in expected_attachment.items()):
        raise RuntimeError("Locked pair-presentation exposure changed")
    pairs = load_train_pairs(Path(config["counterfactual_data"]["train_path"]), config["counterfactual_data"]["expected_train_sha256"])
    state_config = config["state_transition_data"]
    annotations = load_state_training_data(
        manifest_path=Path(state_config["manifest_path"]), train_path=Path(state_config["train_path"]),
        expected_manifest_sha256=state_config["expected_manifest_sha256"],
        expected_train_sha256=state_config["expected_train_sha256"], pairs=pairs, tokenizer=tokenizer,
        expected_tokenizer_sha256=tokenizer_hash,
        expected_source_manifest_sha256=config["counterfactual_data"]["expected_manifest_sha256"],
        expected_source_train_sha256=config["counterfactual_data"]["expected_train_sha256"],
    )
    sequence_length = int(config["model"]["max_seq_len"])
    sources = {name: _packed_dataset(details["path"], sequence_length, int(pad_id)) for name, details in config["mixed_data"]["training_sources"].items()}
    dataset = ScheduledMixedDataset(sources, schedule)
    validation = _packed_dataset(config["data"]["validation_path"], sequence_length, int(pad_id))
    model, heads = create_fresh_model_and_heads(model_config_from_dict(config["model"]), device=device)
    counts = parameter_counts(model, heads)
    print(f"base parameters: {counts['base']:,}; auxiliary parameters: {counts['auxiliary']:,}")
    optimizer = torch.optim.AdamW([*model.parameters(), *heads.parameters()], lr=1e-3, weight_decay=0.1, fused=True)
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    run_dir = Path(config["logging"]["run_dir"])
    if checkpoint_dir.exists() or run_dir.exists():
        raise FileExistsError("State-Transition-v1 output exists; resume is forbidden")
    checkpoint_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    locks = {
        "tokenizer_sha256": tokenizer_hash, "schedule_file_sha256": sha256_file(schedule_path),
        "schedule_content_sha256": schedule_hash, "state_manifest_sha256": state_config["expected_manifest_sha256"],
        "state_train_sha256": state_config["expected_train_sha256"],
        "counterfactual_train_sha256": config["counterfactual_data"]["expected_train_sha256"],
    }
    write_json_atomic(run_dir / "manifest.json", {
        "experiment_id": config["experiment_id"], "status": "training_started",
        "started_at": datetime.now().isoformat(), "training_git_commit": git_commit(),
        "fresh_random_initialization": True, "resume_checkpoint": None,
        "parameter_counts": counts, "model_config": asdict(model.config), "head_config": head_config(),
        "locks": locks, "dataset_gates": gates, "real_artifacts": real_artifacts,
        "selection_metric": "Data30M validation loss only", "held_out_access_during_training": False,
    })
    first_examples = state_examples_for_pairs([next(iter(pairs.values()))], annotations, tokenizer)
    preflight = preflight_numerical_sanity(model, heads, first_examples, device)
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    optimizer.zero_grad(set_to_none=True)
    step = total_targets = total_pairs = total_transitions = total_carries = 0
    source_targets: dict[str, int] = defaultdict(int)
    metrics: list[dict[str, Any]] = [{"step": 0, "selection_eligible": False, "data30m_validation": evaluate(model, validation, batch_size=16, device=device, use_bf16=True), "preflight": preflight}]
    interval: dict[str, float] = defaultdict(float)
    pair_results: list[dict[str, Any]] = []
    state_results: list[dict[str, Any]] = []
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
            examples = state_examples_for_pairs(selected, annotations, tokenizer)
            if all(entry["source"] == "real" for entry in entries) and (selected or examples):
                raise RuntimeError("Real-only microbatch acquired pair/state attachments")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, ce_loss = model(input_ids, labels)
                pair_outputs = [symmetric_pair_loss(model, pair, device) for pair in selected]
                rank_loss = torch.stack([item[0] for item in pair_outputs]).mean() if pair_outputs else logits.sum() * 0.0
                state_loss, current_state_results = state_loss_for_examples(model, heads, examples, device)
                objective = ce_loss + LOCKED_LAMBDA * rank_loss + LOCKED_STATE_LAMBDA * state_loss  # type: ignore[operator]
            if ce_loss is None or not torch.isfinite(objective):
                raise FloatingPointError("State-Transition-v1 objective is NaN or Inf")
            for source, (loss_sum, count) in source_token_losses(logits, labels, entries).items():
                interval[f"{source}_ce_sum"] += float(loss_sum.detach())
                interval[f"{source}_targets"] += count
                source_targets[source] += count
            interval["objective_raw_sum"] += float(objective.detach())
            interval["ce_raw_sum"] += float(ce_loss.detach())
            interval["ce_weighted_sum"] += float(ce_loss.detach())
            interval["pair_raw_sum"] += float(rank_loss.detach())
            interval["pair_weighted_sum"] += LOCKED_LAMBDA * float(rank_loss.detach())
            interval["state_raw_sum"] += float(state_loss.detach())
            interval["state_weighted_sum"] += LOCKED_STATE_LAMBDA * float(state_loss.detach())
            interval["microbatches"] += 1
            interval["targets"] += valid_targets
            interval["samples"] += input_ids.shape[0]
            pair_results.extend(item[1] for item in pair_outputs)
            state_results.extend(current_state_results)
            total_targets += valid_targets
            total_pairs += len(pair_outputs)
            total_transitions += sum(item["kind"] == "transition" for item in current_state_results)
            total_carries += sum(item["kind"] == "carry" for item in current_state_results)
            (objective / 4).backward()
            if microbatch % 4:
                continue
            step += 1
            lr = learning_rate_for_step(step, max_steps=LOCKED_STEPS, warmup_steps=111, learning_rate=1e-3, min_learning_rate=1e-4)
            for group in optimizer.param_groups:
                group["lr"] = lr
            grad_norm = torch.nn.utils.clip_grad_norm_([*model.parameters(), *heads.parameters()], 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Gradient norm is NaN or Inf")
            interval["grad_norm_sum"] += float(grad_norm)
            interval["grad_norm_max"] = max(interval["grad_norm_max"], float(grad_norm))
            interval["steps"] += 1
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step not in set(LOCKED_CHECKPOINT_STEPS):
                continue
            elapsed = time.monotonic() - interval_started
            validation_metrics = evaluate(model, validation, batch_size=16, device=device, use_bf16=True)
            state_grouped = group_state_metrics(state_results)
            metric = {
                "step": step, "selection_eligible": True, "data30m_validation": validation_metrics,
                "loss_scales": {name: interval[f"{name}_sum"] / interval["microbatches"] for name in ("objective_raw", "ce_raw", "ce_weighted", "pair_raw", "pair_weighted", "state_raw", "state_weighted")},
                "real_ce_loss": interval["real_ce_sum"] / interval["real_targets"],
                "curriculum_ce_loss": interval["curriculum_ce_sum"] / interval["curriculum_targets"],
                "ranking": _group_metrics(pair_results), "state": state_grouped,
                "learning_rate": lr, "valid_targets_per_second": interval["targets"] / elapsed,
                "samples_per_second": interval["samples"] / elapsed,
                "gradient_norm_mean": interval["grad_norm_sum"] / interval["steps"],
                "gradient_norm_max": interval["grad_norm_max"],
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
                "peak_gpu_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                "peak_gpu_memory_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
                "exposure": {"real_targets": source_targets["real"], "curriculum_targets": source_targets["curriculum"], "pair_evals": total_pairs, "transition_annotations": total_transitions, "carry_annotations": total_carries},
                "elapsed_seconds": time.monotonic() - started,
            }
            metrics.append(metric)
            for name, value in metric["loss_scales"].items():
                writer.add_scalar(f"loss_scale/{name}", value, step)
            for name in ("learning_rate", "valid_targets_per_second", "samples_per_second", "gradient_norm_mean", "gradient_norm_max", "gpu_memory_allocated_gb", "gpu_memory_reserved_gb", "peak_gpu_memory_allocated_gb", "peak_gpu_memory_reserved_gb"):
                writer.add_scalar(f"train/{name}", metric[name], step)
            for dimension, groups in state_grouped.items():
                if not isinstance(groups, dict) or "count" in groups:
                    groups = {"all": groups}
                for group, values in groups.items():
                    for name, value in values.items():
                        writer.add_scalar(f"state/{dimension}/{group}/{name}", value, step)
            writer.add_scalar("validation/data30m_loss", validation_metrics["loss"], step)
            write_json_atomic(run_dir / "metrics.json", {"checkpoints": metrics})
            candidate_best = float(validation_metrics["loss"]) < best_loss
            if candidate_best:
                best_loss, best_step = float(validation_metrics["loss"]), step
            for name in ("latest.pt", f"step-{step:04d}.pt") + (("best-validation.pt",) if candidate_best else ()):
                save_checkpoint(checkpoint_dir / name, model=model, state_heads=heads, optimizer=optimizer, config=config, step=step, locks=locks, metrics=metrics)
            interval, pair_results, state_results = defaultdict(float), [], []
            interval_started = time.monotonic()
        expected_sources = {name: int(item["valid_targets"]) for name, item in schedule["sources"].items()}
        if step != LOCKED_STEPS or total_targets != LOCKED_TOTAL_VALID_TARGETS or dict(source_targets) != expected_sources or total_pairs != expected_attachment["pair_presentations"]:
            raise RuntimeError("Final locked exposure does not match schedule")
        best = checkpoint_dir / "best-validation.pt"
        summary = {
            "status": "training_and_selection_complete", "optimizer_steps": step,
            "base_parameters": counts["base"], "auxiliary_parameters": counts["auxiliary"],
            "total_training_parameters": counts["total_training"],
            "exposure": {"real_targets": source_targets["real"], "curriculum_targets": source_targets["curriculum"], "pair_evals": total_pairs, "transition_annotations": total_transitions, "carry_annotations": total_carries, "total_state_annotations": total_transitions + total_carries},
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
