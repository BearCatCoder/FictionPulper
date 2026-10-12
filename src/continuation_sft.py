"""Hash-locked export, preflight, training, and post-selection for issue #9."""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from src.continuation_pilot import (
    PRODUCTION_SCHEMA_SHA256,
    deterministic_splits,
    serialize_example,
)
from src.continuity_curriculum.common import canonical_json, load_jsonl, sha256_bytes, sha256_file, write_json_atomic, write_jsonl
from src.dataset import PackedStoryDataset
from src.model import FictionPulperLM, ModelConfig
from src.narrative_v2_baseline import (
    aggregate_forced_choice,
    aggregate_teacher_forced,
    evaluate_forced_choice,
    evaluate_teacher_forced,
    score_span,
)
from src.narrative_v2_free_generation import (
    generate_records as generate_benchmark_records,
    load_protocol as load_benchmark_protocol,
)
from src.narrative_v2_loader import load_authored_split
from src.narrative_v2_scoring import (
    aggregate_diagnostics as aggregate_benchmark_diagnostics,
    build_review_packet as build_benchmark_review_packet,
    review_id as benchmark_review_id,
)
from src.scene_scorecard import build_review_packet, generate_matrix, load_protocol as load_scene_protocol, validate_matrix
from src.train import evaluate, set_seed


FORMAT = "fictionpulper-continuation-sft-v1"
EXPERIMENT_ID = "fictionpulper-50m-data30m-continuation-sft-v1"
FROZEN_CONFIG_SHA256 = "07a4c2f93a285f32a865e04db0074d8aa7582c396662b2908f57ccd0baa92523"
FROZEN_POST_SELECTION_PROTOCOL_SHA256 = "f45041dcfb87e4f3dea8c0abce988beb346a8d77991305650cd8e46bebed0b87"
SPLIT_COUNTS = {"train": 160, "validation": 20, "test": 20}
SPLIT_GENRE_COUNTS = {
    split: {genre: count for genre in ("crime_noir", "horror_occult", "science_fiction", "western_adventure", "weird_fantasy")}
    for split, count in (("train", 32), ("validation", 4), ("test", 4))
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_file(path: Path, expected: str, label: str) -> str:
    require(path.is_file(), f"missing {label}: {path}")
    actual = sha256_file(path)
    require(actual == expected, f"{label} hash mismatch")
    return actual


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    actual = verify_file(path, expected_sha256, "frozen SFT config")
    require(actual == FROZEN_CONFIG_SHA256, "SFT config differs from the reviewed preregistration")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(config.get("experiment_id") == EXPERIMENT_ID, "experiment identity changed")
    training = config["training"]
    expected = {
        "seed": 9009, "optimizer": "adamw", "learning_rate": 1e-5,
        "weight_decay": 0.1, "batch_size": 8, "gradient_accumulation_steps": 2,
        "precision": "bf16", "grad_clip": 1.0, "sft_epochs": 2,
        "sft_batches_per_epoch": 20, "replay_batches_per_epoch": 114,
        "max_optimizer_steps": 134,
        "resume": False,
    }
    require({key: training.get(key) for key in expected} == expected, "frozen training policy changed")
    require(config["selection"] == {
        "metric": "continuation_validation_nll", "validation_points": [0, 1, 2],
        "tie_break": "earliest_optimizer_step", "heldout_access": "post_selection_only",
    }, "selection policy changed")
    require("post_selection" not in config, "training config must not contain post-selection paths")
    require("test" not in canonical_json(config), "training config exposes a test surface")
    return config


def _approval(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config["corpus"]
    path = Path(spec["approval_path"])
    verify_file(path, spec["approval_sha256"], "continuation approval")
    approval = json.loads(path.read_text(encoding="utf-8"))
    require(approval.get("status") == "APPROVED" and approval.get("approved") is True, "corpus is not approved")
    require(approval.get("approval_mode") == "single_owner_research_pilot", "approval scope changed")
    require(approval.get("examples_sha256") == spec["examples_sha256"], "approval examples lock changed")
    require(approval.get("tokenizer_sha256") == config["tokenizer"]["sha256"], "approval tokenizer lock changed")
    return approval


def _derive_export_rows(config: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Rebuild the authorized train/validation rows from canonical approved data."""
    _approval(config)
    corpus = config["corpus"]
    examples_path = Path(corpus["examples_path"])
    verify_file(examples_path, corpus["examples_sha256"], "canonical continuation examples")
    tokenizer_path = Path(config["tokenizer"]["path"])
    verify_file(tokenizer_path, config["tokenizer"]["sha256"], "tokenizer")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    records = load_jsonl(examples_path)
    require(len(records) == 200, "canonical continuation corpus must contain 200 examples")
    assignments = deterministic_splits(records, SPLIT_COUNTS, int(corpus["split_seed"]), SPLIT_GENRE_COUNTS)
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for record in sorted(records, key=lambda row: row["example_id"]):
        split = assignments[record["example_id"]]
        if split == "test":
            continue
        encoded = serialize_example(record, tokenizer, boundary="", maximum_tokens=1024)
        rows[split].append({
            "example_id": record["example_id"], "split": split,
            "content_sha256": sha256_bytes(canonical_json(record).encode("utf-8")),
            "input_ids": encoded["input_ids"], "labels": encoded["labels"],
            "continuation_target_count_including_eos": encoded["continuation_target_count_including_eos"],
        })
    require({key: len(value) for key, value in rows.items()} == {"train": 160, "validation": 20}, "export split counts changed")
    return rows


def export_training_corpus(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Create a write-once token-only train/validation export from canonical examples."""
    require(not output_dir.exists(), f"refusing to overwrite export: {output_dir}")
    rows = _derive_export_rows(config)
    corpus = config["corpus"]
    output_dir.mkdir(parents=True)
    artifacts = {}
    for split in ("train", "validation"):
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, rows[split])
        artifacts[split] = {"path": str(path), "sha256": sha256_file(path), "examples": len(rows[split])}
    manifest = {
        "format": "fictionpulper-continuation-training-export-v1", "status": "sealed_train_validation_export",
        "canonical_examples_sha256": corpus["examples_sha256"], "approval_sha256": corpus["approval_sha256"],
        "tokenizer_sha256": config["tokenizer"]["sha256"], "schema_sha256": PRODUCTION_SCHEMA_SHA256,
        "split_seed": corpus["split_seed"], "included_splits": ["train", "validation"],
        "excluded_split": "test", "prepared_jsonl_used": False, "artifacts": artifacts,
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    manifest["manifest_sha256"] = sha256_file(output_dir / "manifest.json")
    return manifest


def load_export(config: Mapping[str, Any], manifest_path: Path, expected_sha256: str) -> dict[str, list[dict[str, Any]]]:
    verify_file(manifest_path, expected_sha256, "training export manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("status") == "sealed_train_validation_export", "training export is not sealed")
    require(manifest.get("included_splits") == ["train", "validation"] and manifest.get("excluded_split") == "test", "training export split policy changed")
    require(manifest.get("prepared_jsonl_used") is False, "audit prepared.jsonl is forbidden")
    require(manifest.get("canonical_examples_sha256") == config["corpus"]["examples_sha256"], "export corpus provenance changed")
    require(manifest.get("approval_sha256") == config["corpus"]["approval_sha256"], "export approval provenance changed")
    result = {}
    for split, expected_count in (("train", 160), ("validation", 20)):
        spec = manifest["artifacts"][split]
        path = Path(spec["path"])
        verify_file(path, spec["sha256"], f"SFT {split} export")
        rows = load_jsonl(path)
        require(len(rows) == expected_count and all(row["split"] == split for row in rows), f"SFT {split} export changed")
        for row in rows:
            labels = row.get("labels", [])
            inputs = row.get("input_ids", [])
            target_count = int(row.get("continuation_target_count_including_eos", -1))
            require(len(inputs) == len(labels) and 0 < target_count <= len(labels), f"{split} export has invalid serialization")
            require(sum(value != -100 for value in labels) == target_count, f"{split} export target count changed")
            require(labels == [-100] * (len(labels) - target_count) + labels[-target_count:], f"{split} export mask is not continuation-only")
            require(all(value != -100 for value in labels[-target_count:]), f"{split} export target mask has a gap")
        result[split] = rows
    expected_rows = _derive_export_rows(config)
    require(result == expected_rows, "training export does not exactly reproduce the approved canonical corpus")
    return result


class ContinuationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: Sequence[Mapping[str, Any]], pad_id: int, length: int = 1024) -> None:
        self.rows, self.pad_id, self.length = list(rows), pad_id, length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        inputs, labels = list(row["input_ids"]), list(row["labels"])
        require(len(inputs) == len(labels) <= self.length, "invalid exported sequence length")
        require(any(value != -100 for value in labels), "continuation example has no targets")
        padding = self.length - len(inputs)
        return torch.tensor(inputs + [self.pad_id] * padding), torch.tensor(labels + [-100] * padding)


def build_schedule(seed: int = 9009) -> list[dict[str, int | str]]:
    """Freeze two epochs of 20 SFT and 114 replay batches each."""
    schedule = []
    for epoch in (1, 2):
        sources = ["sft"] * 20 + ["replay"] * 114
        random.Random(seed + epoch).shuffle(sources)
        schedule.extend({"epoch": epoch, "source": source} for source in sources)
    return schedule


def schedule_accounting(schedule: Sequence[Mapping[str, Any]], batch_size: int = 8) -> dict[str, Any]:
    sft = sum(row["source"] == "sft" for row in schedule)
    replay = len(schedule) - sft
    return {
        "target_sft_ratio": 0.15, "ratio_unit": "microbatches_and_examples_at_equal_batch_size",
        "sft_microbatches": sft, "replay_microbatches": replay,
        "sft_example_presentations": sft * batch_size, "replay_example_presentations": replay * batch_size,
        "actual_sft_ratio": sft / len(schedule), "exact_target_infeasible_reason": "320 required SFT presentations / 8 = 40 batches; no integer total batch count makes 40 exactly 15%",
    }


def load_base_model(config: Mapping[str, Any], device: torch.device) -> tuple[FictionPulperLM, dict[str, Any]]:
    base = config["base"]
    seal_path, checkpoint_path = Path(base["seal_path"]), Path(base["checkpoint_path"])
    verify_file(seal_path, base["seal_sha256"], "base seal")
    verify_file(checkpoint_path, base["checkpoint_sha256"], "base checkpoint")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    require(seal.get("status") == "sealed", "base experiment is not sealed")
    require(seal.get("checkpoint_sha256", {}).get("best_validation") == base["checkpoint_sha256"], "base seal selected checkpoint changed")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    require(payload.get("tokenizer_hash", payload.get("tokenizer_sha256")) == config["tokenizer"]["sha256"], "base tokenizer provenance changed")
    model = FictionPulperLM(ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model, payload


@torch.no_grad()
def continuation_metrics(model: FictionPulperLM, dataset: Dataset, device: torch.device, use_bf16: bool) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model.eval()
    loss_sum = targets = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits, _ = model(inputs)
        loss_sum += float(F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum"))
        targets += int(labels.ne(-100).sum())
    model.train()
    nll = loss_sum / targets
    require(math.isfinite(nll), "continuation validation NLL is not finite")
    return {"nll": nll, "perplexity": math.exp(nll), "targets": targets}


def _batch(dataset: Dataset, indices: Sequence[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [dataset[index] for index in indices]
    return torch.stack([row[0] for row in rows]).to(device), torch.stack([row[1] for row in rows]).to(device)


def _epoch_indices(size: int, presentations: int, seed: int) -> list[int]:
    require(presentations % size == 0, "presentations must be whole deterministic dataset passes")
    output = []
    for repetition in range(presentations // size):
        values = list(range(size))
        random.Random(seed + repetition).shuffle(values)
        output.extend(values)
    return output


def save_checkpoint(path: Path, *, model: FictionPulperLM, optimizer: torch.optim.Optimizer, config: Mapping[str, Any], config_sha256: str, export_sha256: str, preflight_sha256: str, base_sha256: str, epoch: int, step: int, metrics: Sequence[Mapping[str, Any]], exposure: Mapping[str, int]) -> None:
    require(not path.exists(), f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT, "model": model.state_dict(), "model_config": asdict(model.config),
        "optimizer": optimizer.state_dict(), "optimizer_initialized_fresh": True,
        "resume_checkpoint": None, "config": dict(config), "config_sha256": config_sha256,
        "export_manifest_sha256": export_sha256, "base_checkpoint_sha256": base_sha256,
        "preflight_sha256": preflight_sha256,
        "epoch": epoch, "step": step, "metrics": list(metrics), "exposure": dict(exposure),
        "selection_metric": "continuation_validation_nll", "seed": 9009,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def select_checkpoint(metrics: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [row for row in metrics if row.get("selection_eligible")]
    require(bool(eligible), "no selection-eligible checkpoints")
    return min(eligible, key=lambda row: (float(row["continuation_validation"]["nll"]), int(row["step"])))


def preflight(
    config: Mapping[str, Any], export_path: Path, export_sha256: str, *,
    config_sha256: str, device: torch.device,
) -> dict[str, Any]:
    require(device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "real-model preflight requires CUDA BF16")
    rows = load_export(config, export_path, export_sha256)
    tokenizer = Tokenizer.from_file(config["tokenizer"]["path"])
    pad_id = tokenizer.token_to_id("<|pad|>")
    require(pad_id is not None, "tokenizer lacks pad token")
    dataset = ContinuationDataset(rows["train"], int(pad_id))
    inputs, labels = _batch(dataset, list(range(8)), device)
    require(torch.all(labels[inputs.eq(int(pad_id))].eq(-100)), "padding entered the loss mask")
    model, _ = load_base_model(config, device)
    original = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1, fused=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(inputs, labels)
    require(loss is not None and torch.isfinite(loss), "BF16 fit loss is not finite")
    assert loss is not None
    loss.backward()
    missing_gradients = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    nonfinite_gradients = [
        name for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    require(not missing_gradients, f"trainable parameters lack gradients: {missing_gradients}")
    require(not nonfinite_gradients, f"trainable parameters have non-finite gradients: {nonfinite_gradients}")
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    require(torch.isfinite(grad_norm) and float(grad_norm) > 0, "gradient preflight failed")
    optimizer.step()
    with torch.no_grad():
        probe_after = model(inputs[:1])[0].cpu()
    serialized = io.BytesIO()
    torch.save(original, serialized)
    serialized.seek(0)
    model.load_state_dict(torch.load(serialized, map_location=device, weights_only=True), strict=True)
    require(all(torch.equal(value, model.state_dict()[key]) for key, value in original.items()), "exact base reload failed")

    tiny = ContinuationDataset(rows["train"][:1], int(pad_id))
    tiny_inputs, tiny_labels = _batch(tiny, [0], device)
    tiny_model, _ = load_base_model(config, device)
    tiny_optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3, weight_decay=0.0, fused=True)
    losses = []
    for _ in range(24):
        tiny_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, tiny_loss = tiny_model(tiny_inputs, tiny_labels)
        assert tiny_loss is not None
        tiny_loss.backward()
        tiny_optimizer.step()
        losses.append(float(tiny_loss.detach()))
    require(losses[-1] < losses[0] * 0.5, "tiny-example memorization plumbing gate failed")
    return {
        "status": "PASS", "serialization_and_masks": "PASS", "finite_gradient": float(grad_norm),
        "all_trainable_parameters_have_finite_gradients": True,
        "exact_base_reload": True, "cuda_bf16_real_model_fit": True,
        "tiny_example_memorization": {"initial_loss": losses[0], "final_loss": losses[-1], "evidence_role": "plumbing_only"},
        "optimizer_state_loaded_from_base": False, "post_update_probe_finite": bool(torch.isfinite(probe_after).all()),
        "config_sha256": config_sha256, "export_manifest_sha256": export_sha256,
        "base_checkpoint_sha256": config["base"]["checkpoint_sha256"],
        "base_seal_sha256": config["base"]["seal_sha256"],
        "tokenizer_sha256": config["tokenizer"]["sha256"],
    }


def train(
    config: Mapping[str, Any], config_sha256: str, export_path: Path,
    export_sha256: str, preflight_path: Path, preflight_sha256: str,
) -> dict[str, Any]:
    device = torch.device("cuda")
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "training requires CUDA BF16")
    verify_file(preflight_path, preflight_sha256, "passed SFT preflight")
    preflight_report = json.loads(preflight_path.read_text(encoding="utf-8"))
    require(preflight_report.get("status") == "PASS", "SFT preflight did not pass")
    require(preflight_report.get("config_sha256") == config_sha256, "preflight config lock changed")
    require(preflight_report.get("export_manifest_sha256") == export_sha256, "preflight export lock changed")
    require(preflight_report.get("all_trainable_parameters_have_finite_gradients") is True, "preflight gradient coverage is incomplete")
    output = Path(config["training"]["output_dir"])
    checkpoints = Path(config["training"]["checkpoint_dir"])
    require(not output.exists() and not checkpoints.exists(), "outputs exist; resume and overwrite are forbidden")
    rows = load_export(config, export_path, export_sha256)
    _approval(config)
    tokenizer = Tokenizer.from_file(config["tokenizer"]["path"])
    pad_id = tokenizer.token_to_id("<|pad|>")
    require(pad_id is not None, "tokenizer lacks pad token")
    sft = ContinuationDataset(rows["train"], int(pad_id))
    validation = ContinuationDataset(rows["validation"], int(pad_id))
    replay_spec = config["pre_selection"]["replay"]
    verify_file(Path(replay_spec["bin_path"]), replay_spec["bin_sha256"], "Data30M replay binary")
    verify_file(Path(replay_spec["index_path"]), replay_spec["index_sha256"], "Data30M replay index")
    replay = PackedStoryDataset(replay_spec["bin_path"], replay_spec["index_path"], sequence_length=1024, pad_token_id=int(pad_id))
    model, _ = load_base_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1, fused=True)
    set_seed(9009)
    schedule = build_schedule()
    accounting = schedule_accounting(schedule)
    require(accounting["sft_example_presentations"] == 320, "SFT exposure changed")
    output.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    metrics: list[dict[str, Any]] = [{"epoch": 0, "step": 0, "selection_eligible": False, "continuation_validation": continuation_metrics(model, validation, device, True)}]
    exposure = {"sft_examples": 0, "replay_examples": 0, "sft_targets": 0, "replay_targets": 0}
    step = microbatch = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in (1, 2):
        epoch_schedule = [row for row in schedule if row["epoch"] == epoch]
        sft_indices = _epoch_indices(len(sft), 160, 9009 + epoch * 100)
        replay_indices = list(range(len(replay)))
        random.Random(9009 + epoch * 1000).shuffle(replay_indices)
        replay_indices = replay_indices[:912]
        positions = {"sft": 0, "replay": 0}
        for entry in epoch_schedule:
            source = str(entry["source"])
            dataset = sft if source == "sft" else replay
            indices = sft_indices if source == "sft" else replay_indices
            start = positions[source]
            inputs, labels = _batch(dataset, indices[start:start + 8], device)
            positions[source] += 8
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(inputs, labels)
            require(loss is not None and torch.isfinite(loss), "training loss is not finite")
            assert loss is not None
            targets = int(labels.ne(-100).sum())
            exposure[f"{source}_examples"] += inputs.shape[0]
            exposure[f"{source}_targets"] += targets
            (loss / 2).backward()
            microbatch += 1
            if microbatch % 2 == 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                require(torch.isfinite(norm), "training gradient is not finite")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
        metric = {
            "epoch": epoch, "step": step, "selection_eligible": True,
            "continuation_validation": continuation_metrics(model, validation, device, True),
            "exposure": dict(exposure),
        }
        metrics.append(metric)
        save_checkpoint(checkpoints / f"epoch-{epoch:02d}.pt", model=model, optimizer=optimizer, config=config, config_sha256=config_sha256, export_sha256=export_sha256, preflight_sha256=preflight_sha256, base_sha256=config["base"]["checkpoint_sha256"], epoch=epoch, step=step, metrics=metrics, exposure=exposure)
    require(
        exposure["sft_examples"] == 320
        and exposure["replay_examples"] == 1824
        and step == int(config["training"]["max_optimizer_steps"]),
        "final exposure or compute cap changed",
    )
    selected = select_checkpoint(metrics)
    selected_path = checkpoints / f"epoch-{selected['epoch']:02d}.pt"
    summary = {
        "status": "training_and_validation_selection_complete", "selection_metric": "continuation_validation_nll",
        "selected_epoch": selected["epoch"], "selected_step": selected["step"], "selected_checkpoint": str(selected_path),
        "selected_checkpoint_sha256": sha256_file(selected_path), "metrics": metrics, "exposure": exposure,
        "selected_checkpoint_exposure": selected["exposure"],
        "schedule_accounting": accounting, "seed": 9009, "heldout_access_during_training": False,
        "base_checkpoint_sha256": config["base"]["checkpoint_sha256"], "optimizer_initialized_fresh": True,
        "config_sha256": config_sha256, "export_manifest_sha256": export_sha256,
        "preflight_sha256": preflight_sha256,
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def verify_selection(config: Mapping[str, Any], summary_path: Path, checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary.get("status") == "training_and_validation_selection_complete", "training selection is incomplete")
    require(summary.get("heldout_access_during_training") is False, "training holdout isolation changed")
    require(summary.get("base_checkpoint_sha256") == config["base"]["checkpoint_sha256"], "summary base provenance changed")
    require(summary.get("optimizer_initialized_fresh") is True, "summary optimizer provenance changed")
    accounting = schedule_accounting(build_schedule())
    require(summary.get("schedule_accounting") == accounting, "summary schedule accounting changed")
    exposure = summary.get("exposure", {})
    require(exposure.get("sft_examples") == 320 and exposure.get("replay_examples") == 1824, "summary exact exposure changed")
    selected = select_checkpoint(summary["metrics"])
    require(selected["epoch"] == summary["selected_epoch"] and selected["step"] == summary["selected_step"], "selected checkpoint is not validation minimum with earliest-step tie break")
    verify_file(checkpoint_path, summary["selected_checkpoint_sha256"], "selected SFT checkpoint")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(payload.get("format") == FORMAT and payload.get("resume_checkpoint") is None, "candidate checkpoint provenance changed")
    require(payload.get("base_checkpoint_sha256") == config["base"]["checkpoint_sha256"], "candidate base provenance changed")
    require(payload.get("optimizer_initialized_fresh") is True, "candidate optimizer provenance changed")
    require(payload.get("step") == summary["selected_step"], "candidate checkpoint step changed")
    require(payload.get("epoch") == summary["selected_epoch"], "candidate checkpoint epoch changed")
    expected_epoch = int(summary["selected_epoch"])
    require(int(summary["selected_step"]) == expected_epoch * 67, "selected step does not match the frozen schedule")
    expected_selected_examples = {
        "sft_examples": expected_epoch * 160,
        "replay_examples": expected_epoch * 912,
    }
    selected_exposure = summary.get("selected_checkpoint_exposure", {})
    require(
        all(selected_exposure.get(key) == value for key, value in expected_selected_examples.items()),
        "selected exposure does not match the selected epoch",
    )
    require(selected.get("exposure") == selected_exposure, "selected metric exposure changed")
    require(payload.get("seed") == 9009 and payload.get("selection_metric") == "continuation_validation_nll", "candidate protocol changed")
    require(payload.get("exposure") == summary.get("selected_checkpoint_exposure"), "candidate selected exposure changed")
    require(
        payload.get("metrics") == summary["metrics"][:expected_epoch + 1],
        "candidate checkpoint metrics do not match the selection summary",
    )
    require(payload.get("config") == dict(config), "candidate frozen config changed")
    require(payload.get("config_sha256") == summary.get("config_sha256"), "candidate config hash changed")
    require(payload.get("export_manifest_sha256") == summary.get("export_manifest_sha256"), "candidate export hash changed")
    require(payload.get("preflight_sha256") == summary.get("preflight_sha256"), "candidate preflight hash changed")
    return summary, payload


def _write_once(path: Path, value: Any, *, jsonl: bool = False) -> None:
    require(not path.exists(), f"refusing to overwrite post-selection artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        write_jsonl(path, value)
    else:
        write_json_atomic(path, value)


def run_post_selection(
    config: Mapping[str, Any], summary_path: Path, checkpoint_path: Path,
    summary_sha256: str, output_dir: Path, protocol_path: Path, protocol_sha256: str,
) -> dict[str, Any]:
    """Open all holdouts only after proving validation-only selection."""
    final_output_dir = output_dir
    output_dir = output_dir.with_name(output_dir.name + "-attempt")
    require(
        not final_output_dir.exists() and not output_dir.exists(),
        f"refusing to overwrite post-selection output or preserved attempt: {final_output_dir}",
    )
    verify_file(summary_path, summary_sha256, "validation selection summary")
    summary, checkpoint = verify_selection(config, summary_path, checkpoint_path)
    verify_file(protocol_path, protocol_sha256, "frozen post-selection protocol")
    require(protocol_sha256 == FROZEN_POST_SELECTION_PROTOCOL_SHA256, "post-selection protocol differs from the reviewed preregistration")
    post = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(post.get("experiment_id") == EXPERIMENT_ID and post.get("version") == 1, "post-selection protocol identity changed")
    device = torch.device("cuda")
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "post-selection requires CUDA BF16")
    verify_file(Path(config["tokenizer"]["path"]), config["tokenizer"]["sha256"], "post-selection tokenizer")
    tokenizer = Tokenizer.from_file(config["tokenizer"]["path"])
    model = FictionPulperLM(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    base_model, _ = load_base_model(config, device)
    base_model.eval()

    # Continuation test is first opened here, after the gate above.
    continuation_path = Path(post["continuation_examples_path"])
    verify_file(continuation_path, config["corpus"]["examples_sha256"], "canonical continuation test source")
    canonical = load_jsonl(continuation_path)
    assignments = deterministic_splits(canonical, SPLIT_COUNTS, int(config["corpus"]["split_seed"]), SPLIT_GENRE_COUNTS)
    test_rows = []
    for row in canonical:
        if assignments[row["example_id"]] == "test":
            encoded = serialize_example(row, tokenizer, boundary="", maximum_tokens=1024)
            test_rows.append({"input_ids": encoded["input_ids"], "labels": encoded["labels"]})
    pad_id = tokenizer.token_to_id("<|pad|>")
    assert pad_id is not None
    continuation_dataset = ContinuationDataset(test_rows, int(pad_id))
    continuation_test: dict[str, Any] = {
        "base": continuation_metrics(base_model, continuation_dataset, device, True),
        "candidate": continuation_metrics(model, continuation_dataset, device, True),
    }
    continuation_test["candidate_minus_base_nll"] = continuation_test["candidate"]["nll"] - continuation_test["base"]["nll"]
    clean = post["clean_fiction"]
    verify_file(Path(clean["bin_path"]), clean["bin_sha256"], "clean-fiction test binary")
    verify_file(Path(clean["index_path"]), clean["index_sha256"], "clean-fiction test index")
    clean_data = PackedStoryDataset(clean["bin_path"], clean["index_path"], sequence_length=1024, pad_token_id=int(pad_id))
    clean_likelihood: dict[str, Any] = {
        "base": evaluate(base_model, clean_data, batch_size=8, device=device, use_bf16=True),
        "candidate": evaluate(model, clean_data, batch_size=8, device=device, use_bf16=True),
    }
    clean_likelihood["candidate_minus_base_loss"] = clean_likelihood["candidate"]["loss"] - clean_likelihood["base"]["loss"]

    scene_protocol, prompts, scene_hashes = load_scene_protocol(Path(post["scene_protocol_root"]))
    require(scene_hashes == {
        "protocol_sha256": post["scene_protocol_sha256"],
        "prompts_sha256": post["scene_prompts_sha256"],
        "human_review_rubric_sha256": post["scene_rubric_sha256"],
    }, "Scene Scorecard config locks changed")
    scene_records = generate_matrix(model, tokenizer, prompts, scene_protocol, device=device, use_bf16=True)
    base = post["base_scene_raw"]
    verify_file(Path(base["path"]), base["sha256"], "sealed base raw scene outputs")
    base_records = load_jsonl(Path(base["path"]))
    validate_matrix(base_records, prompts, scene_protocol)
    combined = [
        {**row, "generation_id": f"base:{row['generation_id']}"}
        for row in base_records
    ] + [
        {**row, "generation_id": f"candidate:{row['generation_id']}"}
        for row in scene_records
    ]
    scene_packet, scene_key = build_review_packet(
        combined, seed=int(scene_protocol["review"]["blinded_packet_seed"]),
    )
    model_by_review = {
        review_id: str(metadata["generation_id"]).split(":", 1)[0]
        for review_id, metadata in scene_key.items()
    }
    scene_dir = output_dir / "scene-scorecard"

    benchmark_root = Path(post["benchmark_v2_root"])
    verify_file(
        benchmark_root / "authored-manifest.json",
        post["benchmark_v2_authored_manifest_sha256"],
        "Benchmark v2 authored manifest",
    )
    _, benchmark_hashes = load_benchmark_protocol(benchmark_root)
    require(benchmark_hashes["protocol_sha256"] == post["benchmark_v2_protocol_sha256"], "Benchmark v2 protocol config lock changed")
    for split, base_spec in post["benchmark_v2_base_free_generation"].items():
        verify_file(Path(base_spec["path"]), base_spec["sha256"], f"sealed base Benchmark v2 {split} generations")
    benchmark_results = {"base": {}, "candidate": {}}
    benchmark_scenarios: dict[str, list[dict[str, Any]]] = {}
    for model_name, evaluated_model in (("base", base_model), ("candidate", model)):
        scorer: Callable[[Sequence[int], Sequence[int]], dict[str, Any]] = lambda prefix, span, current=evaluated_model: score_span(current, prefix, span, device=device, use_bf16=True)
        for split, test_complete in (("development", False), ("test", False), ("generalization", True)):
            scenarios = load_authored_split(
                benchmark_root, split,
                checkpoint_selection_complete=split != "development",
                test_evaluation_complete=test_complete,
            )
            benchmark_scenarios[split] = scenarios
            forced = evaluate_forced_choice(scenarios, tokenizer, scorer)
            teacher = evaluate_teacher_forced(scenarios, tokenizer, scorer)
            benchmark_results[model_name][split] = {"forced_choice": aggregate_forced_choice(forced), "teacher_forced": aggregate_teacher_forced(teacher)}

    benchmark_generations: dict[str, Any] = {}
    benchmark_generation_dir = output_dir / "benchmark-v2"
    rubric = json.loads((benchmark_root / "human-review-rubric.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True)
    for split in ("development", "test", "generalization"):
        records = generate_benchmark_records(
            scenarios=benchmark_scenarios[split], protocol=load_benchmark_protocol(benchmark_root)[0],
            model=model, tokenizer=tokenizer, split=split, blinded_model_id="model-c",
            device=device, use_bf16=True,
        )
        split_dir = benchmark_generation_dir / split
        raw_path = split_dir / "raw-generations.jsonl"
        packet_path = split_dir / "blinded-review-packet.jsonl"
        _write_once(raw_path, records, jsonl=True)
        benchmark_packet = build_benchmark_review_packet(
            records, randomization_seed=int(rubric["randomization_seed"]),
        )
        _write_once(packet_path, benchmark_packet, jsonl=True)
        _write_once(split_dir / "private-review-key.json", {
            "blinded_model_id": "model-c",
            "model_identity": summary["selected_checkpoint_sha256"],
            "review_metadata": {
                benchmark_review_id(row): {
                    field: row.get(field) for field in (
                        "scenario_id", "world", "generation_mode", "sampling_seed",
                        "blinded_model_id", "generated_text_sha256",
                    )
                }
                for row in records
            },
            "human_review_status": "pending",
        })
        benchmark_generations[split] = {
            "generation_count": len(records),
            "raw_sha256": sha256_file(raw_path),
            "review_packet_sha256": sha256_file(packet_path),
            "diagnostics": aggregate_benchmark_diagnostics(records),
        }
    def scene_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        return {
            "diagnostic_only": True,
            "outputs": count,
            "mean_repeated_4gram_rate": sum(float(row["diagnostics"]["repetition"]["repeated_4gram_rate"]) for row in rows) / count,
            "mean_repeated_sentence_rate": sum(float(row["diagnostics"]["repetition"]["repeated_sentence_rate"]) for row in rows) / count,
            "exact_atomic_fact_phrase_mentions": sum(sum(row["diagnostics"]["exact_atomic_fact_phrase_mentions"].values()) for row in rows),
            "length_compliant_outputs": sum(row["diagnostics"]["length_classification"] == "compliant" for row in rows),
        }

    report = {
        "status": "automated_post_selection_complete_human_review_pending", "selection_verified_before_holdout_open": True,
        "selected_checkpoint_sha256": summary["selected_checkpoint_sha256"],
        "selection_summary_sha256": summary_sha256,
        "post_selection_protocol_sha256": protocol_sha256,
        "continuation_test": continuation_test, "clean_fiction_likelihood": clean_likelihood,
        "scene_scorecard": {
            **scene_hashes, "base_outputs": 40, "candidate_outputs": 40,
            "blinded_packet_outputs": 80, "sealed_base_raw_reused": base,
            "automated_diagnostics": {
                "base": scene_diagnostics(base_records),
                "candidate": scene_diagnostics(scene_records),
            },
            "human_review": "pending",
        },
        "benchmark_v2": {
            **benchmark_hashes, "automated_modes": benchmark_results,
            "candidate_free_generation": benchmark_generations,
            "sealed_base_free_generation_reused": post["benchmark_v2_base_free_generation"],
            "human_review": "pending",
        },
        "decision": {
            "status": "STOP_NO_PROMOTION",
            "reason": "Independent blinded Scene Scorecard review is pending; automated diagnostics and likelihood cannot establish narrative improvement.",
            "next_action": "Complete genuine independent review of the matched base/candidate Scene Scorecard packet before any promotion or additional SFT setting.",
            "extra_compute_confound": "The base received no matched LM-only updates, so this pilot does not isolate an SFT-specific causal effect.",
        },
    }
    scene_dir.mkdir()
    _write_once(scene_dir / "candidate-raw.jsonl", scene_records, jsonl=True)
    _write_once(scene_dir / "blinded-review-packet.jsonl", scene_packet, jsonl=True)
    _write_once(scene_dir / "private-review-key.json", {
        "model_identities": {"base": config["base"]["checkpoint_sha256"], "candidate": summary["selected_checkpoint_sha256"]},
        "model_by_review_id": model_by_review, "review_metadata": scene_key, "human_review_status": "pending",
    })
    _write_once(output_dir / "report.json", report)
    output_dir.rename(final_output_dir)
    return report


def correct_scene_review_packet(
    config: Mapping[str, Any], source_dir: Path, source_report_sha256: str,
    candidate_raw_sha256: str, output_dir: Path, protocol_path: Path,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Derive a corrected scene packet without changing the completed run."""
    require(not output_dir.exists(), f"refusing to overwrite correction output: {output_dir}")
    report_path = source_dir / "report.json"
    verify_file(report_path, source_report_sha256, "source post-selection report")
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    verify_file(protocol_path, protocol_sha256, "frozen post-selection protocol")
    require(protocol_sha256 == FROZEN_POST_SELECTION_PROTOCOL_SHA256, "post-selection protocol differs from the reviewed preregistration")
    post = json.loads(protocol_path.read_text(encoding="utf-8"))
    scene_protocol, prompts, scene_hashes = load_scene_protocol(Path(post["scene_protocol_root"]))
    require(scene_hashes["protocol_sha256"] == post["scene_protocol_sha256"], "scene protocol lock changed")
    base_spec = post["base_scene_raw"]
    base_path = Path(base_spec["path"])
    verify_file(base_path, base_spec["sha256"], "sealed base raw scene outputs")
    base_records = load_jsonl(base_path)
    candidate_path = source_dir / "scene-scorecard" / "candidate-raw.jsonl"
    verify_file(candidate_path, candidate_raw_sha256, "candidate raw scene matrix")
    candidate_records = load_jsonl(candidate_path)
    validate_matrix(base_records, prompts, scene_protocol)
    validate_matrix(candidate_records, prompts, scene_protocol)
    combined = [
        {**row, "generation_id": f"base:{row['generation_id']}"}
        for row in base_records
    ] + [
        {**row, "generation_id": f"candidate:{row['generation_id']}"}
        for row in candidate_records
    ]
    packet, key = build_review_packet(
        combined, seed=int(scene_protocol["review"]["blinded_packet_seed"]),
    )
    require(len(packet) == 80 and len(key) == 80, "corrected scene packet is incomplete")
    packet_path = output_dir / "blinded-review-packet.jsonl"
    key_path = output_dir / "private-review-key.json"
    _write_once(packet_path, packet, jsonl=True)
    _write_once(key_path, {
        "model_identities": {
            "base": config["base"]["checkpoint_sha256"],
            "candidate": source_report["selected_checkpoint_sha256"],
        },
        "model_by_review_id": {
            review_id: str(metadata["generation_id"]).split(":", 1)[0]
            for review_id, metadata in key.items()
        },
        "review_metadata": key,
        "human_review_status": "pending",
    })
    correction = {
        "status": "corrected_scene_review_packet_human_review_pending",
        "source_report_sha256": source_report_sha256,
        "source_candidate_raw_sha256": sha256_file(candidate_path),
        "source_flawed_packet_sha256": sha256_file(source_dir / "scene-scorecard" / "blinded-review-packet.jsonl"),
        "defect": "The original scene packet path contained the Benchmark v2 generalization packet because a local variable was reused after scene packet construction.",
        "raw_generations_changed": False,
        "model_or_decoding_changed": False,
        "corrected_output_count": len(packet),
        "corrected_packet_sha256": sha256_file(packet_path),
        "corrected_key_sha256": sha256_file(key_path),
        "human_review": "pending",
    }
    _write_once(output_dir / "correction-report.json", correction)
    return correction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/continuation-sft-v1.yaml"))
    parser.add_argument("--config-sha256", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--output-dir", type=Path, default=Path("data/continuation_sft_v1"))
    for name in ("preflight", "train"):
        child = sub.add_parser(name)
        child.add_argument("--export-manifest", type=Path, required=True)
        child.add_argument("--export-sha256", required=True)
        if name == "preflight":
            child.add_argument("--output", type=Path, default=Path("runs/fictionpulper-50m-data30m-continuation-sft-v1-preflight.json"))
        else:
            child.add_argument("--preflight", type=Path, required=True)
            child.add_argument("--preflight-sha256", required=True)
    post = sub.add_parser("post-selection")
    post.add_argument("--summary", type=Path, required=True)
    post.add_argument("--summary-sha256", required=True)
    post.add_argument("--checkpoint", type=Path, required=True)
    post.add_argument("--protocol", type=Path, default=Path("experiments/fictionpulper-50m-data30m-continuation-sft-v1/post-selection-protocol.json"))
    post.add_argument("--protocol-sha256", required=True)
    post.add_argument("--output-dir", type=Path, default=Path("runs/fictionpulper-50m-data30m-continuation-sft-v1/post-selection"))
    correction = sub.add_parser("correct-scene-packet")
    correction.add_argument("--source-dir", type=Path, required=True)
    correction.add_argument("--source-report-sha256", required=True)
    correction.add_argument("--candidate-raw-sha256", required=True)
    correction.add_argument("--protocol", type=Path, default=Path("experiments/fictionpulper-50m-data30m-continuation-sft-v1/post-selection-protocol.json"))
    correction.add_argument("--protocol-sha256", required=True)
    correction.add_argument("--output-dir", type=Path, default=Path("runs/fictionpulper-50m-data30m-continuation-sft-v1/post-selection-correction"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.config_sha256)
    if args.command == "export":
        result = export_training_corpus(config, args.output_dir)
    elif args.command == "preflight":
        result = preflight(
            config, args.export_manifest, args.export_sha256,
            config_sha256=args.config_sha256, device=torch.device("cuda"),
        )
        _write_once(args.output, result)
    elif args.command == "train":
        result = train(
            config, args.config_sha256, args.export_manifest, args.export_sha256,
            args.preflight, args.preflight_sha256,
        )
    elif args.command == "post-selection":
        result = run_post_selection(
            config, args.summary, args.checkpoint, args.summary_sha256, args.output_dir,
            args.protocol, args.protocol_sha256,
        )
    else:
        result = correct_scene_review_packet(
            config, args.source_dir, args.source_report_sha256,
            args.candidate_raw_sha256, args.output_dir, args.protocol,
            args.protocol_sha256,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
