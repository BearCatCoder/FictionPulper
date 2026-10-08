"""Frozen-model shortcut audit for Counterfactual Narrative v1."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch
import yaml
from tokenizers import Tokenizer

from src.continuity_curriculum.common import canonical_json, sha256_bytes, sha256_file, write_json_atomic
from src.frozen_state_probe import freeze_model, load_frozen_checkpoint


HELD_OUT_SPLITS = ("validation", "test", "generalization_holdout")


def _verify(path: Path, expected: str, description: str) -> str:
    if not path.is_file():
        raise RuntimeError(f"Missing {description}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"Hash mismatch for {description}: {path}")
    return observed


def verify_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """Verify every sealed input before loading a tokenizer, dataset, or model."""
    tokenizer = config["tokenizer"]
    tokenizer_hash = _verify(Path(tokenizer["path"]), tokenizer["sha256"], "tokenizer")
    dataset = config["dataset"]
    dataset_dir = Path(dataset["directory"])
    manifest_path = dataset_dir / "manifest.json"
    manifest_hash = _verify(manifest_path, dataset["manifest_sha256"], "dataset manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or not manifest.get("safe_to_train"):
        raise RuntimeError("Counterfactual dataset manifest is not a full PASS artifact")
    if manifest.get("tokenizer_sha256") != tokenizer_hash:
        raise RuntimeError("Dataset and configured tokenizer hashes differ")
    files = {}
    for relative, specification in manifest.get("files", {}).items():
        path = dataset_dir / relative
        observed = _verify(path, specification["sha256"], f"dataset file {relative}")
        if path.stat().st_size != int(specification["bytes"]):
            raise RuntimeError(f"Byte-size mismatch for dataset file: {path}")
        files[relative] = observed
    required = {f"{split}.jsonl" for split in HELD_OUT_SPLITS}
    if not required <= set(files):
        raise RuntimeError("Dataset manifest omits a required held-out split")

    checkpoints = {}
    for specification in config["checkpoints"]:
        model_id = specification["id"]
        seal_path = Path(specification["seal_path"])
        seal_hash = _verify(seal_path, specification["seal_sha256"], f"{model_id} seal")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if seal.get("status") != "sealed" or seal.get("run_id") != specification["sealed_run_id"]:
            raise RuntimeError(f"Invalid sealed identity for {model_id}")
        sealed_checkpoint = seal["checkpoint_sha256"]
        if isinstance(sealed_checkpoint, dict):
            sealed_checkpoint = sealed_checkpoint[specification.get("checkpoint_seal_key", "best_validation")]
        if sealed_checkpoint != specification["sha256"]:
            raise RuntimeError(f"Configured checkpoint is not the one named by the {model_id} seal")
        checkpoint_hash = _verify(Path(specification["path"]), specification["sha256"], f"{model_id} checkpoint")
        seal_tokenizer = seal.get("tokenizer_sha256", seal.get("input_sha256", {}).get("tokenizer"))
        if seal_tokenizer != tokenizer_hash:
            raise RuntimeError(f"{model_id} seal uses a different tokenizer")
        checkpoints[model_id] = {
            "path": specification["path"], "sha256": checkpoint_hash,
            "seal_path": str(seal_path), "seal_sha256": seal_hash,
        }
    return {
        "tokenizer": {"path": tokenizer["path"], "sha256": tokenizer_hash},
        "dataset": {"directory": str(dataset_dir), "manifest_sha256": manifest_hash, "files": files},
        "checkpoints": checkpoints,
    }


def load_pairs(dataset_dir: Path, split: str) -> list[dict[str, Any]]:
    if split not in HELD_OUT_SPLITS:
        raise ValueError(f"Not a held-out split: {split}")
    pairs = []
    with (dataset_dir / f"{split}.jsonl").open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            pair = json.loads(line)
            supplied = pair.get("stable_hash")
            unhashed = {key: value for key, value in pair.items() if key != "stable_hash"}
            if supplied != sha256_bytes(canonical_json(unhashed).encode("utf-8")):
                raise RuntimeError(f"Stable hash mismatch in {split}.jsonl line {line_number}")
            if pair.get("schema_version") != 1 or pair.get("generator_version") != "counterfactual-narrative-v1":
                raise RuntimeError(f"Unsupported schema in {split}.jsonl line {line_number}")
            if (pair.get("split") != split or set(pair.get("worlds", {})) != {"A", "B"}
                    or set(pair.get("candidates", {})) != {"X", "Y"}):
                raise RuntimeError(f"Malformed pair identity in {split}.jsonl line {line_number}")
            if pair["worlds"]["A"]["correct_candidate"] == pair["worlds"]["B"]["correct_candidate"]:
                raise RuntimeError(f"Pair does not reverse labels: {pair['pair_id']}")
            pairs.append(pair)
    if not pairs:
        raise RuntimeError(f"Held-out split is empty: {split}")
    return pairs


@torch.inference_mode()
def score_token_spans(
    model: torch.nn.Module,
    examples: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    pad_id: int,
    use_bf16: bool,
) -> list[float]:
    """Score each exact [decision_start, decision_end) span by mean token log P."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    freeze_model(model)  # type: ignore[arg-type]
    output: list[float] = []
    for offset in range(0, len(examples), batch_size):
        batch = examples[offset:offset + batch_size]
        maximum = max(len(item["token_ids"]) for item in batch)
        tokens = torch.full((len(batch), maximum), pad_id, dtype=torch.long, device=device)
        for row, item in enumerate(batch):
            ids = item["token_ids"]
            start, end = int(item["decision_start"]), int(item["decision_end"])
            if not 0 < start < end <= len(ids):
                raise ValueError("Decision span must be non-empty and have a preceding token")
            tokens[row, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cpu", enabled=False)
        with autocast:
            logits = model(tokens)[0]
        for row, item in enumerate(batch):
            start, end = int(item["decision_start"]), int(item["decision_end"])
            targets = tokens[row, start:end]
            log_probs = torch.log_softmax(logits[row, start - 1:end - 1].float(), dim=-1)
            output.append(float(log_probs.gather(-1, targets.unsqueeze(-1)).mean().item()))
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Scoring changed frozen-model state")
    return output


def _prefix(tokenizer: Tokenizer, genre_control: str | None) -> list[int]:
    names = ["<|story|>", *([genre_control] if genre_control else []), "<|bos|>"]
    values = [tokenizer.token_to_id(name) for name in names]
    if any(value is None for value in values):
        raise RuntimeError(f"Tokenizer lacks a required control token: {names}")
    return [int(value) for value in values]


def build_scoring_examples(pairs: Sequence[dict[str, Any]], tokenizer: Tokenizer) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build model inputs and descriptors in matching deterministic order."""
    examples: list[dict[str, Any]] = []
    descriptors: list[dict[str, Any]] = []
    eos = tokenizer.token_to_id("<|eos|>")
    if eos is None:
        raise RuntimeError("Tokenizer lacks EOS")
    for pair in pairs:
        candidates = pair["candidates"]
        for world_name in ("A", "B"):
            for candidate_name in ("X", "Y"):
                encoding = pair["worlds"][world_name]["candidate_encodings"][candidate_name]
                examples.append(dict(encoding))
                descriptors.append({"pair_id": pair["pair_id"], "condition": "normal", "world": world_name, "candidate": candidate_name})
        prefix = _prefix(tokenizer, pair["metadata"].get("genre_control_token"))
        for condition, context_key, ids_key in (
            ("fact_removed", "fact_removed_context", "fact_removed_token_ids"),
            ("irrelevant_substituted", "irrelevant_substituted_context", "irrelevant_substituted_token_ids"),
        ):
            context = pair["controls"][context_key]
            context_ids = pair["controls"][ids_key]
            if tokenizer.encode(context).ids != context_ids:
                raise RuntimeError(f"Control tokenization mismatch: {pair['pair_id']}:{condition}")
            for candidate_name in ("X", "Y"):
                candidate_ids = candidates[candidate_name]["token_ids"]
                if tokenizer.encode(context + candidates[candidate_name]["text"]).ids != [*context_ids, *candidate_ids]:
                    raise RuntimeError(f"Ambiguous control/candidate boundary: {pair['pair_id']}:{condition}")
                start = len(prefix) + len(context_ids)
                examples.append({"token_ids": [*prefix, *context_ids, *candidate_ids, eos],
                                 "decision_start": start, "decision_end": start + len(candidate_ids)})
                descriptors.append({"pair_id": pair["pair_id"], "condition": condition, "candidate": candidate_name})
        # Candidate-only scoring intentionally omits both narrative and genre context.
        bare_prefix = _prefix(tokenizer, None)
        for candidate_name in ("X", "Y"):
            candidate_ids = candidates[candidate_name]["token_ids"]
            examples.append({"token_ids": [*bare_prefix, *candidate_ids, eos],
                             "decision_start": len(bare_prefix), "decision_end": len(bare_prefix) + len(candidate_ids)})
            descriptors.append({"pair_id": pair["pair_id"], "condition": "candidate_only", "candidate": candidate_name})
    return examples, descriptors


def _preference(margin: float, tolerance: float) -> str:
    if margin > tolerance:
        return "X"
    if margin < -tolerance:
        return "Y"
    return "tie"


def pair_metrics(pair: dict[str, Any], scores: dict[tuple[str, str | None, str], float], tie_tolerance: float) -> dict[str, Any]:
    normal = {}
    signed = {}
    for world in ("A", "B"):
        x = scores[("normal", world, "X")]
        y = scores[("normal", world, "Y")]
        margin = x - y
        expected = pair["worlds"][world]["correct_candidate"]
        preference = _preference(margin, tie_tolerance)
        normal[world] = {"X": x, "Y": y, "margin_x_minus_y": margin,
                         "preference": preference, "correct": preference == expected}
        signed[world] = margin if expected == "X" else -margin
    controls = {}
    for condition in ("fact_removed", "irrelevant_substituted", "candidate_only"):
        x = scores[(condition, None, "X")]
        y = scores[(condition, None, "Y")]
        margin = x - y
        preference = _preference(margin, tie_tolerance)
        direction_scores = {
            world: 0.5 if preference == "tie" else float(preference == pair["worlds"][world]["correct_candidate"])
            for world in ("A", "B")
        }
        controls[condition] = {"X": x, "Y": y, "margin_x_minus_y": margin,
                               "absolute_preference_margin": abs(margin), "preference": preference,
                               "shared_context_direction_scores": direction_scores}
    return {
        "pair_id": pair["pair_id"], "split": pair["split"],
        "state_family": pair["abstract_counterfactual_variable"]["state_family"],
        "distance": pair["metadata"]["distance_bucket"], "difficulty": pair["metadata"]["difficulty"],
        "expected": {world: pair["worlds"][world]["correct_candidate"] for world in ("A", "B")},
        "normal": normal,
        "joint_paired_reversal_success": normal["A"]["correct"] and normal["B"]["correct"],
        "signed_reversal_margin": mean(signed.values()),
        "controls": controls,
    }


def _rate(trials: Sequence[dict[str, Any]], condition: str, preference: str) -> float:
    return sum(item["controls"][condition]["preference"] == preference for item in trials) / len(trials)


def summarize_trials(trials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        raise ValueError("Cannot summarize no trials")
    normal_directions = [(trial, world) for trial in trials for world in ("A", "B")]
    x_directions = [(trial, world) for trial, world in normal_directions if trial["expected"][world] == "X"]
    y_directions = [(trial, world) for trial, world in normal_directions if trial["expected"][world] == "Y"]
    result: dict[str, Any] = {
        "pair_count": len(trials),
        "normal": {
            "direction_accuracy": sum(trial["normal"][world]["correct"] for trial, world in normal_directions) / len(normal_directions),
            "world_A_accuracy": mean(float(trial["normal"]["A"]["correct"]) for trial in trials),
            "world_B_accuracy": mean(float(trial["normal"]["B"]["correct"]) for trial in trials),
            "X_direction_win_rate": mean(float(trial["normal"][world]["correct"]) for trial, world in x_directions),
            "Y_direction_win_rate": mean(float(trial["normal"][world]["correct"]) for trial, world in y_directions),
            "joint_paired_reversal_success": mean(float(trial["joint_paired_reversal_success"]) for trial in trials),
            "joint_chance_reference": 0.25,
            "mean_signed_reversal_margin": mean(trial["signed_reversal_margin"] for trial in trials),
        },
    }
    for condition in ("fact_removed", "irrelevant_substituted"):
        # One shared context is evaluated against two opposite labels. It cannot
        # independently represent A and B; half-credit ties make balanced
        # directional accuracy exactly chance (0.5), by construction.
        result[condition] = {
            "shared_context_per_pair": True,
            "paired_success_defined": False,
            "joint_paired_reversal_success": None,
            "paired_success_note": "A fixed shared context cannot simultaneously prefer opposite A/B labels.",
            "world_A_directional_accuracy_ties_half_credit": mean(trial["controls"][condition]["shared_context_direction_scores"]["A"] for trial in trials),
            "world_B_directional_accuracy_ties_half_credit": mean(trial["controls"][condition]["shared_context_direction_scores"]["B"] for trial in trials),
            "balanced_directional_accuracy_ties_half_credit": mean(
                trial["controls"][condition]["shared_context_direction_scores"][world]
                for trial in trials for world in ("A", "B")
            ),
            "chance_reference": 0.5,
            "X_win_rate": _rate(trials, condition, "X"),
            "Y_win_rate": _rate(trials, condition, "Y"),
            "tie_rate": _rate(trials, condition, "tie"),
            "mean_margin_x_minus_y": mean(trial["controls"][condition]["margin_x_minus_y"] for trial in trials),
            "mean_absolute_preference_margin": mean(trial["controls"][condition]["absolute_preference_margin"] for trial in trials),
        }
    result["candidate_only"] = {
        "context": "<|story|><|bos|> followed by the candidate; no genre or narrative context",
        "X_win_rate": _rate(trials, "candidate_only", "X"),
        "Y_win_rate": _rate(trials, "candidate_only", "Y"),
        "tie_rate": _rate(trials, "candidate_only", "tie"),
        "mean_X_minus_Y_score_difference": mean(trial["controls"]["candidate_only"]["margin_x_minus_y"] for trial in trials),
    }
    return result


def aggregate_trials(trials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output = {"overall": summarize_trials(trials)}
    for field in ("state_family", "distance", "difficulty"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trial in trials:
            groups[str(trial[field])].append(trial)
        output[f"by_{field}"] = {key: summarize_trials(values) for key, values in sorted(groups.items())}
    latest = [trial for trial in trials if trial["state_family"] == "latest_state_update"]
    output["latest_state_subset"] = (
        {"present": True, **summarize_trials(latest)}
        if latest else {"present": False, "pair_count": 0, "normal": {"direction_accuracy": None}}
    )
    return output


def evaluate_model_split(
    model: torch.nn.Module, pairs: Sequence[dict[str, Any]], tokenizer: Tokenizer, *,
    device: torch.device, batch_size: int, use_bf16: bool, tie_tolerance: float,
) -> dict[str, Any]:
    examples, descriptors = build_scoring_examples(pairs, tokenizer)
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise RuntimeError("Tokenizer lacks PAD")
    values = score_token_spans(model, examples, device=device, batch_size=batch_size,
                               pad_id=pad_id, use_bf16=use_bf16)
    by_pair: dict[str, dict[tuple[str, str | None, str], float]] = defaultdict(dict)
    for descriptor, value in zip(descriptors, values, strict=True):
        key = (descriptor["condition"], descriptor.get("world"), descriptor["candidate"])
        by_pair[descriptor["pair_id"]][key] = value
    trials = [pair_metrics(pair, by_pair[pair["pair_id"]], tie_tolerance) for pair in pairs]
    return {"metrics": aggregate_trials(trials), "trials": trials}


def audit_gate(models: dict[str, Any], maximum_side_win_rate: float) -> dict[str, Any]:
    checks = []
    for model_id, splits in models.items():
        for split, result in splits.items():
            candidate = result["metrics"]["overall"]["candidate_only"]
            checks.append({
                "model": model_id, "split": split,
                "X_win_rate": candidate["X_win_rate"], "Y_win_rate": candidate["Y_win_rate"],
                "maximum_allowed": maximum_side_win_rate,
                "passed": candidate["X_win_rate"] <= maximum_side_win_rate and candidate["Y_win_rate"] <= maximum_side_win_rate,
            })
    controls = {
        "passed": all(
            result["metrics"]["overall"][condition]["shared_context_per_pair"]
            and result["metrics"]["overall"][condition]["balanced_directional_accuracy_ties_half_credit"] == 0.5
            for splits in models.values() for result in splits.values()
            for condition in ("fact_removed", "irrelevant_substituted")
        ),
        "note": "Removed and irrelevant controls have one shared context per pair; opposite A/B labels therefore cancel exactly. Preference magnitude is diagnostic, not paired success.",
    }
    return {"passed": all(item["passed"] for item in checks) and controls["passed"],
            "candidate_only_no_persistent_side_winner": checks, "shared_context_controls": controls}


def run_audit(config: dict[str, Any]) -> dict[str, Any]:
    provenance = verify_inputs(config)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Real sealed-checkpoint audit requires CUDA BF16")
    tokenizer = Tokenizer.from_file(config["tokenizer"]["path"])
    dataset_dir = Path(config["dataset"]["directory"])
    pairs = {split: load_pairs(dataset_dir, split) for split in HELD_OUT_SPLITS}
    device = torch.device("cuda")
    models = {}
    for specification in config["checkpoints"]:
        model, _ = load_frozen_checkpoint(Path(specification["path"]), device)
        models[specification["id"]] = {
            split: evaluate_model_split(
                model, split_pairs, tokenizer, device=device,
                batch_size=int(config["evaluation"]["batch_size"]), use_bf16=True,
                tie_tolerance=float(config["evaluation"]["tie_tolerance"]),
            )
            for split, split_pairs in pairs.items()
        }
        del model
        torch.cuda.empty_cache()
    gate = audit_gate(models, float(config["gate"]["maximum_candidate_side_win_rate"]))
    return {
        "audit": "counterfactual_narrative_v1_pretraining_model_shortcut_audit",
        "status": "PASS" if gate["passed"] else "STOP",
        "score_definition": "arithmetic mean of exact decision-span token log probabilities",
        "control_metric_definition": "Each control has one shared context, not independent A/B worlds; paired reversal success is undefined and preference magnitude is reported.",
        "provenance": {**provenance, "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()},
        "gate": gate, "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/counterfactual-pretraining-audit-v1.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["output_path"])
    allowed_root = Path("runs/fictionpulper-15m-data30m-counterfactual-v1/pretraining-audit")
    if output != allowed_root / "results.json":
        raise RuntimeError(f"Audit output must remain isolated at {allowed_root / 'results.json'}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run_audit(config)
    write_json_atomic(output, result)
    print(json.dumps({"status": result["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
