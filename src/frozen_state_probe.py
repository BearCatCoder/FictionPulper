"""Extract frozen narrative states and train deterministic linear probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from tokenizers import Tokenizer
from torch import nn

from src.model import FictionPulperLM, ModelConfig


SUPPORTED_LABEL_FAMILIES = (
    "object_ownership",
    "object_location",
    "character_location",
    "knowledge_holder",
    "relationship_state",
    "injury_state",
    "temporal_order",
    "obligation_state",
)
UNSUPPORTED_REASONS = {
    "goal_status": "goal status is active for every valid pre-resolution goal example",
    "object_retrieval_state": "retrieved is false for every valid pre-resolution ownership example",
    "cause_effect": "cause/effect events provide no varying structural class label",
    "primary_state_type": "predicts task/template identity rather than narrative state",
    "difficulty": "predicts curriculum construction metadata rather than narrative state",
    "genre": "predicts genre metadata rather than narrative state",
    "primary_attribute": "predicts event schema/template rather than narrative state",
    "primary_entity_slot": "is underspecified without a state-specific semantic role",
}
SPLITS = ("train", "validation", "test", "generalization_holdout")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def layer_names(num_layers: int) -> list[str]:
    return ["embedding", *(f"layer_{index + 1:02d}" for index in range(num_layers)), "final_norm"]


def freeze_model(model: FictionPulperLM) -> None:
    model.eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Transformer freezing failed")


def _primary_events(sidecar: dict[str, Any]) -> list[dict[str, Any]]:
    primary = sidecar["primary_distance"]
    boundary = int(primary["resolved_character_start"])
    events = [
        event
        for event in sidecar["events"]
        if event["state_type"] == primary["state_type"]
        and int(event["character_end"]) <= boundary
    ]
    if not events:
        raise ValueError(f"{sidecar['id']}: primary state has no pre-resolution event")
    return sorted(events, key=lambda event: (event["character_start"], event["character_end"], event["attribute"]))


def _slot_maps(sidecar: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    values = sidecar["generation_values"]
    # Lexical rank is independent of generator argument order and yields the
    # same finite class vocabulary for split-exclusive surface forms.
    characters = {
        str(value): f"character_{index}"
        for index, value in enumerate(sorted(values["names"], key=lambda item: (item.casefold(), item)))
    }
    locations = {
        str(value): f"location_{index}"
        for index, value in enumerate(sorted(values["locations"], key=lambda item: (item.casefold(), item)))
    }
    return characters, locations


def labels_for_example(record: dict[str, Any], sidecar: dict[str, Any]) -> dict[str, str]:
    if record["id"] != sidecar["id"] or record["split"] != sidecar["split"]:
        raise ValueError("Record and sidecar identity mismatch")
    events = _primary_events(sidecar)
    state_type = str(sidecar["primary_distance"]["state_type"])
    characters, locations = _slot_maps(sidecar)

    def one(attribute: str, *, value: Any | None = None) -> dict[str, Any]:
        matches = [
            event for event in events
            if event["attribute"] == attribute and (value is None or event["value"] == value)
        ]
        if len(matches) != 1:
            raise ValueError(f"{sidecar['id']}: expected one visible {attribute!r} event, got {len(matches)}")
        return matches[0]

    if state_type == "ownership_transfer_hiding_retrieval":
        holder = one("holder")
        hidden_at = one("hidden_at")
        return {
            "object_ownership": characters[str(holder["value"])],
            "object_location": locations[str(hidden_at["value"])],
        }
    if state_type == "physical_scene_location_movement":
        destinations = {str(event["value"]) for event in events if event["attribute"] == "location"}
        if len(destinations) != 1:
            raise ValueError(f"{sidecar['id']}: character locations disagree before resolution")
        return {"character_location": locations[destinations.pop()]}
    if state_type == "knowledge_ignorance_secret":
        known = [event for event in events if str(event["attribute"]).startswith("knows:") and event["value"] is True]
        if len(known) != 1:
            raise ValueError(f"{sidecar['id']}: expected exactly one pre-resolution knowledge holder")
        return {"knowledge_holder": characters[str(known[0]["entity"])]}
    if state_type == "character_identity_role_relationship":
        role = one("role")
        relations = [event for event in events if str(event["attribute"]).startswith("relationship_to:")]
        if len(relations) != 1:
            raise ValueError(f"{sidecar['id']}: expected exactly one visible relationship")
        relation = relations[0]
        target = str(relation["attribute"]).split(":", 1)[1]
        return {"relationship_state": (
            f"role={characters[str(role['entity'])]}|"
            f"relation={characters[str(relation['entity'])]}->{characters[target]}"
        )}
    if state_type == "persistent_injury":
        injury = one("injury")
        return {"injury_state": characters[str(injury["entity"])]}
    if state_type == "temporal_ordering":
        first = one("first")
        first_value = str(first["value"])
        actors = [name for name in characters if first_value.startswith(name + " ")]
        if len(actors) != 1:
            raise ValueError(f"{sidecar['id']}: cannot identify first temporal actor")
        return {"temporal_order": characters[actors[0]]}
    if state_type == "promise_debt_obligation":
        obligations = [event for event in events if str(event["attribute"]).startswith("obligation_to:")]
        if len(obligations) != 1:
            raise ValueError(f"{sidecar['id']}: expected exactly one visible obligation")
        obligation = obligations[0]
        creditor = str(obligation["attribute"]).split(":", 1)[1]
        return {"obligation_state": (
            f"debtor={characters[str(obligation['entity'])]}|creditor={characters[creditor]}"
        )}
    return {}


def validate_family_coverage(
    examples: dict[str, list[dict[str, Any]]], families: Iterable[str]
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for family in families:
        labels = {
            split: [example["labels"][family] for example in examples[split] if family in example["labels"]]
            for split in SPLITS
        }
        train_classes = set(labels["train"])
        if len(train_classes) < 2:
            raise ValueError(f"{family}: requires at least two train classes, got {sorted(train_classes)}")
        for split in SPLITS[1:]:
            if not labels[split]:
                raise ValueError(f"{family}: no eligible {split} examples")
            unknown = set(labels[split]) - train_classes
            if unknown:
                raise ValueError(f"{family}: {split} labels absent from train: {sorted(unknown)}")
        report[family] = {
            "classes": sorted(train_classes),
            "support": {split: dict(sorted(Counter(labels[split]).items())) for split in SPLITS},
        }
    return report


def build_probe_examples(
    curriculum_dir: Path,
    tokenizer: Tokenizer,
    *,
    maximum_examples_per_split: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    story_id = tokenizer.token_to_id("<|story|>")
    bos_id = tokenizer.token_to_id("<|bos|>")
    if story_id is None or bos_id is None:
        raise ValueError("Tokenizer lacks required story/BOS tokens")
    result: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        records = {record["id"]: record for record in load_jsonl(curriculum_dir / f"{split}.jsonl")}
        sidecars = {sidecar["id"]: sidecar for sidecar in load_jsonl(curriculum_dir / f"{split}.state.jsonl")}
        if records.keys() != sidecars.keys():
            raise ValueError(f"{split}: story and sidecar IDs differ")
        selected_ids = sorted(records)
        if maximum_examples_per_split is not None:
            selected_ids = selected_ids[:maximum_examples_per_split]
        examples = []
        for identifier in selected_ids:
            record, sidecar = records[identifier], sidecars[identifier]
            primary = sidecar["primary_distance"]
            boundary = int(primary["resolved_character_start"])
            text_prefix = record["text"][:boundary]
            text_ids = tokenizer.encode(text_prefix).ids
            anchors = [anchor for anchor in sidecar["anchors"] if anchor["state_type"] == primary["state_type"]]
            if len(anchors) != 1 or int(anchors[0]["resolved_token"]) != len(text_ids):
                raise ValueError(f"{identifier}: primary pre-resolution anchor is not canonical")
            prefix = [story_id]
            control = record.get("genre_control_token")
            if control is not None:
                control_id = tokenizer.token_to_id(control)
                if control_id is None:
                    raise ValueError(f"{identifier}: unknown genre control token")
                prefix.append(control_id)
            input_ids = [*prefix, bos_id, *text_ids]
            if not input_ids:
                raise ValueError(f"{identifier}: empty probe prefix")
            examples.append({
                "id": identifier,
                "split": split,
                "input_ids": input_ids,
                "position": len(input_ids) - 1,
                "distance_bucket": record["distance_bucket"],
                "labels": labels_for_example(record, sidecar),
            })
        result[split] = examples
    return result


def load_frozen_checkpoint(path: Path, device: torch.device) -> tuple[FictionPulperLM, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if "model" not in payload or "model_config" not in payload:
        raise ValueError(f"Unsupported checkpoint structure: {path}")
    model = FictionPulperLM(ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model"])
    freeze_model(model)
    return model, payload


def extract_states(
    model: FictionPulperLM,
    examples: list[dict[str, Any]],
    *,
    batch_size: int,
    pad_id: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    names = layer_names(model.config.num_layers)
    collected: list[list[np.ndarray]] = [[] for _ in names]
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        maximum = max(len(example["input_ids"]) for example in batch)
        input_ids = torch.full((len(batch), maximum), pad_id, dtype=torch.long, device=device)
        positions = torch.tensor([example["position"] for example in batch], device=device)
        for row, example in enumerate(batch):
            values = torch.tensor(example["input_ids"], dtype=torch.long, device=device)
            input_ids[row, : values.numel()] = values
        with torch.inference_mode():
            output = model(input_ids, return_hidden_states=True)
            hidden_states = output[2]
            rows = torch.arange(len(batch), device=device)
            for index, states in enumerate(hidden_states):
                selected = states[rows, positions].float().cpu().numpy()
                collected[index].append(selected)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Frozen transformer accumulated gradients")
    return {
        name: np.concatenate(chunks, axis=0) if chunks else np.empty((0, model.config.hidden_size), np.float32)
        for name, chunks in zip(names, collected, strict=True)
    }


def classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    num_classes: int,
    train_targets: np.ndarray,
) -> dict[str, Any]:
    support = np.bincount(targets, minlength=num_classes)
    correct = predictions == targets
    recalls, f1s = [], []
    for label in range(num_classes):
        true_positive = int(np.sum((predictions == label) & (targets == label)))
        false_positive = int(np.sum((predictions == label) & (targets != label)))
        false_negative = int(np.sum((predictions != label) & (targets == label)))
        if support[label]:
            recalls.append(true_positive / int(support[label]))
            denominator = 2 * true_positive + false_positive + false_negative
            f1s.append(2 * true_positive / denominator if denominator else 0.0)
    majority = int(np.bincount(train_targets, minlength=num_classes).argmax())
    return {
        "accuracy": float(correct.mean()) if len(targets) else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "chance": 1.0 / num_classes,
        "majority": float(np.mean(targets == majority)) if len(targets) else 0.0,
        "support": {str(index): int(value) for index, value in enumerate(support)},
        "n": int(len(targets)),
    }


def _predict(probe: nn.Linear, features: torch.Tensor, batch_size: int) -> np.ndarray:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            outputs.append(probe(features[start : start + batch_size]).argmax(dim=-1).numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.int64)


def train_probe(
    features: dict[str, np.ndarray],
    examples: dict[str, list[dict[str, Any]]],
    family: str,
    hyperparameters: dict[str, Any],
    seed: int,
    *,
    report_held_out: bool = True,
) -> dict[str, Any]:
    eligible_examples: dict[str, list[dict[str, Any]]] = {}
    eligible_features: dict[str, np.ndarray] = {}
    for split in SPLITS:
        indices = [index for index, example in enumerate(examples[split]) if family in example["labels"]]
        eligible_examples[split] = [examples[split][index] for index in indices]
        eligible_features[split] = features[split][indices]
    examples = eligible_examples
    features = eligible_features
    train_labels = [example["labels"][family] for example in examples["train"]]
    classes = sorted(set(train_labels))
    if len(classes) < 2:
        raise ValueError(f"{family}: requires at least two train classes")
    class_to_id = {label: index for index, label in enumerate(classes)}
    labels: dict[str, np.ndarray] = {}
    evaluation_splits = SPLITS if report_held_out else ("train", "validation")
    for split in evaluation_splits:
        unknown = sorted({example["labels"][family] for example in examples[split]} - class_to_id.keys())
        if unknown:
            raise ValueError(f"{family}: {split} has labels absent from train: {unknown}")
        labels[split] = np.asarray([class_to_id[example["labels"][family]] for example in examples[split]], dtype=np.int64)

    mean = features["train"].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features["train"].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    tensors = {
        split: torch.from_numpy((features[split] - mean) / std)
        for split in evaluation_splits
    }
    targets = {split: torch.from_numpy(labels[split]) for split in evaluation_splits}
    seed_everything(seed)
    probe = nn.Linear(features["train"].shape[1], len(classes), bias=True)
    if not isinstance(probe, nn.Linear) or probe.bias is None:
        raise RuntimeError("Probe must be a biased nn.Linear")
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(hyperparameters["learning_rate"]),
        weight_decay=float(hyperparameters["weight_decay"]),
    )
    batch_size = int(hyperparameters["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    best_key: tuple[float, float, int] | None = None
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(hyperparameters["epochs"]) + 1):
        probe.train()
        permutation = torch.randperm(len(tensors["train"]), generator=generator)
        for start in range(0, len(permutation), batch_size):
            rows = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(probe(tensors["train"][rows]), targets["train"][rows])
            loss.backward()
            optimizer.step()
        probe.eval()
        validation_predictions = _predict(probe, tensors["validation"], batch_size)
        metrics = classification_metrics(
            validation_predictions, labels["validation"], num_classes=len(classes), train_targets=labels["train"]
        )
        validation_loss = float(nn.functional.cross_entropy(
            probe(tensors["validation"]), targets["validation"]
        ).item())
        candidate = (metrics["macro_f1"], -validation_loss, -epoch)
        if best_key is None or candidate > best_key:
            best_key = candidate
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in probe.state_dict().items()
            }
    if best_state is None or best_epoch is None:
        raise RuntimeError("Probe training completed without selecting an epoch")
    probe.load_state_dict(best_state)
    probe.eval()
    evaluations: dict[str, Any] = {}
    for split in evaluation_splits:
        predictions = _predict(probe, tensors[split], batch_size)
        overall = classification_metrics(predictions, labels[split], num_classes=len(classes), train_targets=labels["train"])
        buckets = {}
        distances = np.asarray([example["distance_bucket"] for example in examples[split]])
        for bucket in sorted(set(distances.tolist())):
            mask = distances == bucket
            buckets[bucket] = classification_metrics(
                predictions[mask], labels[split][mask], num_classes=len(classes), train_targets=labels["train"]
            )
        evaluations[split] = {"overall": overall, "distance_buckets": buckets}
    return {
        "classes": classes,
        "selected_epoch": best_epoch,
        "selection_split": "validation",
        "selection_metric": "macro_f1_then_loss_then_earliest_epoch",
        "evaluations": evaluations,
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values))


def aggregate_best_layer(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for split in ("test", "generalization_holdout"):
        overall = [result["evaluations"][split]["overall"] for result in seed_results]
        metrics = {}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = np.asarray([item[metric] for item in overall], dtype=np.float64)
            metrics[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
        chance = _mean(item["chance"] for item in overall)
        majority = _mean(item["majority"] for item in overall)
        if (
            metrics["macro_f1"]["mean"] > chance + 0.05
            and metrics["balanced_accuracy"]["mean"] > chance + 0.05
            and metrics["accuracy"]["mean"] > majority + 0.05
        ):
            interpretation = "strong_linear_state_signal"
        elif (
            metrics["macro_f1"]["mean"] > chance
            or metrics["balanced_accuracy"]["mean"] > chance
        ):
            interpretation = "weak_or_imbalanced_linear_state_signal"
        else:
            interpretation = "no_clear_linear_state_signal"
        summaries[split] = {
            "metrics_over_seeds": metrics,
            "chance": chance,
            "majority_accuracy": majority,
            "interpretation": interpretation,
            "interpretation_policy": (
                "strong requires macro-F1 and balanced accuracy > chance+0.05 and "
                "accuracy > majority+0.05; weak exceeds chance on either balanced metric"
            ),
        }
    return summaries


def resolve_source(spec: dict[str, Any]) -> dict[str, Any]:
    tag = spec.get("source_tag")
    expected_commit = spec.get("source_commit")
    commit = None
    if tag:
        commit = subprocess.run(
            ["git", "rev-parse", f"{tag}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if expected_commit is not None and commit != expected_commit:
            raise RuntimeError(
                f"Source tag {tag!r} resolved to {commit}, expected {expected_commit}"
            )
    return {
        "tag": tag,
        "expected_commit": expected_commit,
        "resolved_commit": commit,
        "tag_matches_expected_commit": commit == expected_commit if tag else None,
    }


def run(config: dict[str, Any], *, config_path: Path | None = None) -> dict[str, Any]:
    requested = list(config["labels"]["requested_families"])
    supported = [family for family in requested if family in SUPPORTED_LABEL_FAMILIES]
    unsupported = [
        {
            "family": family,
            "reason": UNSUPPORTED_REASONS.get(
                family, "no reliable varying pre-resolution structural label implementation"
            ),
        }
        for family in requested if family not in SUPPORTED_LABEL_FAMILIES
    ]
    tokenizer_path = Path(config["data"]["tokenizer_path"])
    curriculum_dir = Path(config["data"]["curriculum_dir"])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    examples = build_probe_examples(
        curriculum_dir, tokenizer,
        maximum_examples_per_split=config["data"].get("maximum_examples_per_split"),
    )
    coverage = validate_family_coverage(examples, supported)
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("Tokenizer lacks pad token")
    device = torch.device(config["extraction"]["device"] if torch.cuda.is_available() else "cpu")
    results: dict[str, Any] = {
        "protocol_version": int(config["protocol_version"]),
        "seeds": list(config["probe"]["seeds"]),
        "shared_probe_hyperparameters": {
            key: value for key, value in config["probe"].items() if key != "seeds"
        },
        "supported_requested_families": supported,
        "unsupported_requested_families": unsupported,
        "family_coverage": coverage,
        "position_policy": "last token of canonical text prefix ending at primary resolved_character_start",
        "slot_policy": (
            "case-insensitive lexical rank within each story's configured character and location values; "
            "generator list order and split-global vocabulary are not used as labels"
        ),
        "selection_policy": "train probes on train; select epoch and layer on validation; report test and generalization afterward",
        "source_artifacts": {
            "config": {
                "path": str(config_path) if config_path is not None else None,
                "sha256": sha256_file(config_path) if config_path is not None else None,
            },
            "tokenizer": {"path": str(tokenizer_path), "sha256": sha256_file(tokenizer_path)},
            "curriculum": {
                "directory": str(curriculum_dir),
                "files": {
                    str(path.relative_to(curriculum_dir)): sha256_file(path)
                    for path in [
                        curriculum_dir / "manifest.json",
                        *(curriculum_dir / f"{split}{suffix}" for split in SPLITS for suffix in (".jsonl", ".state.jsonl")),
                    ]
                },
            },
        },
        "models": {},
    }
    for spec in config["checkpoints"]:
        if not spec.get("enabled", True):
            results["models"][spec["id"]] = {
                "status": "configured_not_available",
                "architecture": spec["architecture"],
                "source": resolve_source(spec),
            }
            continue
        checkpoint_path = Path(spec["path"])
        model, _ = load_frozen_checkpoint(checkpoint_path, device)
        states_by_split = {
            split: extract_states(
                model, examples[split], batch_size=int(config["extraction"]["batch_size"]),
                pad_id=pad_id, device=device,
            )
            for split in SPLITS
        }
        model_result: dict[str, Any] = {
            "status": "complete",
            "architecture": spec["architecture"],
            "source": resolve_source(spec),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "transformer_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "transformer_parameters_with_gradients": sum(p.grad is not None for p in model.parameters()),
            "families": {},
        }
        names = layer_names(model.config.num_layers)
        for family in supported:
            curves = []
            for name in names:
                layer_features = {split: states_by_split[split][name] for split in SPLITS}
                seed_results = [
                    train_probe(
                        layer_features, examples, family, config["probe"], int(seed),
                        report_held_out=False,
                    )
                    for seed in config["probe"]["seeds"]
                ]
                curves.append({
                    "layer": name,
                    "seeds": seed_results,
                    "mean_validation_macro_f1": _mean(
                        item["evaluations"]["validation"]["overall"]["macro_f1"] for item in seed_results
                    ),
                })
            best_curve = max(curves, key=lambda item: (item["mean_validation_macro_f1"], -names.index(item["layer"])))
            best_features = {
                split: states_by_split[split][best_curve["layer"]] for split in SPLITS
            }
            best_seed_results = [
                train_probe(best_features, examples, family, config["probe"], int(seed))
                for seed in config["probe"]["seeds"]
            ]
            model_result["families"][family] = {
                "best_layer": best_curve["layer"],
                "best_layer_mean_validation_macro_f1": best_curve["mean_validation_macro_f1"],
                "best_layer_seed_results": best_seed_results,
                "comparison_summary": aggregate_best_layer(best_seed_results),
                "layerwise_curve": curves,
            }
        results["models"][spec["id"]] = model_result
        del model, states_by_split
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = run(config, config_path=args.config)
    output = Path(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({
        "output": str(output),
        "models": list(result["models"]),
        "supported": result["supported_requested_families"],
        "unsupported": result["unsupported_requested_families"],
    }, indent=2))


if __name__ == "__main__":
    main()
