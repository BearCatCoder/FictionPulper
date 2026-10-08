"""Frozen-model narrative-state localization diagnostics.

The module keeps protocol construction, token-span scoring, and rollout
replacement as pure functions so their boundary conditions can be tested on
CPU without opening checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Sequence

import torch
import yaml
from tokenizers import Tokenizer

from src.context_diagnostics import INTERVENING_PARAGRAPHS
from src.frozen_state_probe import freeze_model, load_frozen_checkpoint, sha256_file
from src.train import sample_next_token
from src.train_tokenizer import write_json_atomic


FAMILIES = (
    "ownership", "object_location", "character_location", "goal", "knowledge",
    "relationship", "persistent_physical_state", "object_state", "cause_effect",
)
DISTANCES = ("d064", "d128", "d256", "d384", "d512", "d768", "d900")
STATE_TYPES = {
    "ownership": "ownership_transfer_hiding_retrieval",
    "object_location": "ownership_transfer_hiding_retrieval",
    "character_location": "physical_scene_location_movement",
    "goal": "goal_persistence_completion_failure",
    "knowledge": "knowledge_ignorance_secret",
    "relationship": "character_identity_role_relationship",
    "persistent_physical_state": "persistent_injury",
    "object_state": "ownership_transfer_hiding_retrieval",
    "cause_effect": "cause_effect",
}
EXPECTED_NARRATIVE_PROTOCOL_SHA256 = "6c83108450efc8973c360278c1f6b6e4262143a40ca58c8d69a4eb165ff78f45"


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def stable_sampling_seed(base_seed: int, case_index: int, mode_index: int) -> int:
    if min(base_seed, case_index, mode_index) < 0:
        raise ValueError("Sampling seed components must be non-negative")
    return base_seed + case_index * 1009 + mode_index * 9176


def mean_decision_span_log_probability(
    logits: torch.Tensor, token_ids: torch.Tensor, decision_start: int
) -> torch.Tensor:
    """Return length-normalized log P(tokens[decision_start:])."""
    if logits.ndim != 3 or token_ids.ndim != 2 or logits.shape[:2] != token_ids.shape:
        raise ValueError("logits and token_ids must have matching [batch, sequence] dimensions")
    if not 0 < decision_start < token_ids.shape[1]:
        raise ValueError("Decision span must be non-empty and have a preceding token")
    log_probs = torch.log_softmax(logits[:, decision_start - 1 : -1].float(), dim=-1)
    targets = token_ids[:, decision_start:]
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean(dim=-1)


@torch.no_grad()
def score_candidate_ids(
    model: torch.nn.Module,
    context_ids: Sequence[int],
    candidate_ids: Sequence[Sequence[int]],
    *,
    device: torch.device,
    use_bf16: bool,
    maximum_context: int = 1024,
) -> list[float]:
    freeze_model(model)  # type: ignore[arg-type]
    if not candidate_ids or any(not candidate for candidate in candidate_ids):
        raise ValueError("Each candidate decision span must be non-empty")
    sequences = [[*context_ids, *candidate] for candidate in candidate_ids]
    if any(len(sequence) > maximum_context for sequence in sequences):
        raise ValueError("Context plus decision span exceeds maximum context")
    scores = []
    for sequence in sequences:
        tokens = torch.tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast(device_type="cpu", enabled=False)
        with autocast:
            logits = model(tokens)[0]
        scores.append(float(mean_decision_span_log_probability(logits, tokens, len(context_ids)).item()))
    if any(parameter.grad is not None or parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Localization scoring modified transformer freeze state")
    return scores


def _filler(distance: str) -> str:
    target = int(distance[1:])
    paragraphs: list[str] = []
    words = 0
    index = 0
    # Token count is validated later with the locked tokenizer; this deterministic
    # word proxy only chooses material from the sealed neutral paragraph source.
    while words < target:
        paragraph = INTERVENING_PARAGRAPHS[index % len(INTERVENING_PARAGRAPHS)]
        paragraphs.append(paragraph)
        words += len(paragraph.split())
        index += 1
    return "\n\n".join(paragraphs)


def _token_distance_filler(distance: str, tokenizer: Tokenizer) -> str:
    target = int(distance[1:])
    paragraphs: list[str] = []
    index = 0
    while len(tokenizer.encode("\n\n".join(paragraphs)).ids) < target:
        paragraphs.append(INTERVENING_PARAGRAPHS[index % len(INTERVENING_PARAGRAPHS)])
        index += 1
    if len(tokenizer.encode("\n\n".join(paragraphs)).ids) > target:
        paragraphs.pop()
        words = INTERVENING_PARAGRAPHS[index % len(INTERVENING_PARAGRAPHS)].split()
        partial = "\n\n".join(paragraphs)
        for word_index, _ in enumerate(words, start=1):
            candidate = "\n\n".join([*paragraphs, " ".join(words[:word_index])])
            if len(tokenizer.encode(candidate).ids) > target:
                break
            partial = candidate
        return partial
    return "\n\n".join(paragraphs)


def _family_text(family: str, names: Sequence[str], objects: Sequence[str], locations: Sequence[str]) -> tuple[str, str, str, str]:
    a, b = names[:2]
    obj = objects[0]
    first, second = locations[:2]
    templates = {
        "ownership": (f"{a} gave the {obj} to {b}.", f"{b} gave the {obj} to {a}.", f" {b} now owned the {obj}.", f" {a} now owned the {obj}."),
        "object_location": (f"{a} left the {obj} at the {first}.", f"{a} left the {obj} at the {second}.", f" The {obj} was at the {first}.", f" The {obj} was at the {second}."),
        "character_location": (f"{a} remained at the {first}.", f"{a} remained at the {second}.", f" {a} was at the {first}.", f" {a} was at the {second}."),
        "goal": (f"{a} intended to reach the {first}.", f"{a} intended to reach the {second}.", f" {a}'s goal was to reach the {first}.", f" {a}'s goal was to reach the {second}."),
        "knowledge": (f"Only {a} learned where the {obj} was hidden.", f"Only {b} learned where the {obj} was hidden.", f" {a} knew where the {obj} was.", f" {b} knew where the {obj} was."),
        "relationship": (f"{a} was {b}'s sibling.", f"{a} was {b}'s employer.", f" {a} was {b}'s sibling.", f" {a} was {b}'s employer."),
        "persistent_physical_state": (f"{a} injured an ankle and had not recovered.", f"{b} injured an ankle and had not recovered.", f" {a} still had the injury.", f" {b} still had the injury."),
        "object_state": (f"The {obj} remained hidden and unretrieved.", f"{a} retrieved the hidden {obj}.", f" The {obj} was still hidden.", f" The {obj} had been retrieved."),
        "cause_effect": (f"A broken cable blocked the {first}.", f"Heavy rain blocked the {first}.", f" The cable caused the blockage.", " The rain caused the blockage."),
    }
    if family not in templates:
        raise ValueError(f"Unsupported counterfactual family: {family}")
    return templates[family]


def build_counterfactual_case(
    *, family: str, distance: str, split: str, source_id: str,
    names: Sequence[str], objects: Sequence[str], locations: Sequence[str], difficulty: int,
    filler: str | None = None, lexical_combination: str | None = None,
) -> dict[str, Any]:
    if distance not in DISTANCES or split not in ("test", "generalization_holdout"):
        raise ValueError("Counterfactual source must use a supported held-out split/distance")
    if len(names) < 2 or len(objects) < 1 or len(locations) < 2:
        raise ValueError("Counterfactual source lacks names, objects, or locations")
    fact_a, fact_b, candidate_x, candidate_y = _family_text(family, names, objects, locations)
    filler = filler if filler is not None else _filler(distance)
    lexical_note = ""
    if lexical_combination:
        adjective, atmosphere = lexical_combination.split("|", 1)
        lexical_note = f" The scene appeared {adjective}, and the mood was {atmosphere}."
    stem = "Read the account and complete the final statement consistently." + lexical_note
    suffix = "\n\nAfter the intervening events, the account confirms:"
    base_a = f"{stem}\n\n{fact_a}\n\n{filler}"
    context_a = base_a + suffix
    context_b = f"{stem}\n\n{fact_b}\n\n{filler}{suffix}"
    removed = f"{stem}\n\n{filler}{suffix}"
    irrelevant = f"{stem}\n\nA clock in the hall stopped at noon.\n\n{filler}{suffix}"
    update = f"{base_a}\n\nCorrection recorded later: {fact_b}{suffix}"
    case = {
        "id": f"{split}-{family}-{distance}", "family": family, "distance": distance,
        "difficulty": int(difficulty), "source_split": split, "source_id": source_id,
        "source_lexical_values": sorted({*names, *objects, *locations, *([lexical_combination] if lexical_combination else [])}),
        "contexts": {"a": context_a, "b": context_b, "removed": removed, "irrelevant": irrelevant, "late_update": update},
        "candidates": {"x": candidate_x, "y": candidate_y},
        "expected": {"a": "x", "b": "y", "late_update": "y"},
    }
    case["sha256"] = canonical_hash(case)
    return case


def validate_counterfactual_cases(cases: Sequence[dict[str, Any]], tokenizer: Tokenizer, maximum_context: int = 1024) -> None:
    seen: set[tuple[str, str]] = set()
    source_values: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        key = (case["family"], case["distance"])
        if key in seen:
            raise ValueError(f"Duplicate family/distance case: {key}")
        seen.add(key)
        if case["expected"] != {"a": "x", "b": "y", "late_update": "y"}:
            raise ValueError("Counterfactual expected directions drifted")
        if case["sha256"] != canonical_hash({key: value for key, value in case.items() if key != "sha256"}):
            raise ValueError("Counterfactual deterministic hash mismatch")
        if case["candidates"]["x"] == case["candidates"]["y"]:
            raise ValueError("Candidate alternatives must differ")
        source_values[case["source_split"]].update(case["source_lexical_values"])
        for context in case["contexts"].values():
            for candidate in case["candidates"].values():
                context_ids = tokenizer.encode(context).ids
                candidate_ids = tokenizer.encode(candidate).ids
                combined_ids = tokenizer.encode(context + candidate).ids
                if combined_ids != [*context_ids, *candidate_ids]:
                    raise ValueError(f"{case['id']} has an ambiguous context/candidate token boundary")
                if 2 + len(combined_ids) > maximum_context:
                    raise ValueError(f"{case['id']} exceeds maximum context")
    if source_values["test"] & source_values["generalization_holdout"]:
        raise ValueError("Held-out lexical pools overlap across source splits")


def build_counterfactual_protocol(curriculum_dir: Path, tokenizer: Tokenizer) -> dict[str, Any]:
    cases = []
    split_data = {}
    for split in ("test", "generalization_holdout"):
        records = {item["id"]: item for item in _load_jsonl(curriculum_dir / f"{split}.jsonl")}
        sidecars = _load_jsonl(curriculum_dir / f"{split}.state.jsonl")
        split_data[split] = (records, sidecars)
    for family_index, family in enumerate(FAMILIES):
        for distance_index, distance in enumerate(DISTANCES):
            split = ("test", "generalization_holdout")[(family_index + distance_index) % 2]
            records, sidecars = split_data[split]
            state_type = STATE_TYPES[family]
            eligible = sorted(
                (item for item in sidecars if state_type in item["state_types"] and records[item["id"]]["distance_bucket"] == distance),
                key=lambda item: item["id"],
            )
            if not eligible:
                split = "generalization_holdout" if split == "test" else "test"
                records, sidecars = split_data[split]
                eligible = sorted(
                    (item for item in sidecars if state_type in item["state_types"] and records[item["id"]]["distance_bucket"] == distance),
                    key=lambda item: item["id"],
                )
            if not eligible:
                continue
            source = eligible[0]
            values = source["generation_values"]
            cases.append(build_counterfactual_case(
                family=family, distance=distance, split=split, source_id=source["id"],
                names=values["names"], objects=values["objects"], locations=values["locations"],
                difficulty=records[source["id"]]["difficulty"], filler=_token_distance_filler(distance, tokenizer),
                lexical_combination=values["lexical_combination"],
            ))
    cases.sort(key=lambda item: (FAMILIES.index(item["family"]), DISTANCES.index(item["distance"])))
    validate_counterfactual_cases(cases, tokenizer)
    source_paths = [
        curriculum_dir / "manifest.json",
        *(curriculum_dir / f"{split}{suffix}" for split in ("test", "generalization_holdout") for suffix in (".jsonl", ".state.jsonl")),
    ]
    return {
        "protocol_version": 1, "target_case_count": 63, "case_count": len(cases),
        "coverage_note": "One case per family/distance where the sealed sidecars contain that state/distance cell.",
        "source_artifacts": {str(path): sha256_file(path) for path in source_paths},
        "cases": cases,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def validate_forced_choice_protocol(protocol: dict[str, Any], immutable: dict[str, Any], immutable_hash: str) -> None:
    if immutable_hash != EXPECTED_NARRATIVE_PROTOCOL_SHA256 or protocol["immutable_protocol_sha256"] != immutable_hash:
        raise ValueError("Immutable narrative protocol hash mismatch")
    facts = [(entry["id"], key, value) for entry in immutable["entries"] for key, value in entry["facts"].items()]
    entries = protocol.get("entries", [])
    if len(facts) != 13 or len(entries) != 13:
        raise ValueError("Forced-choice protocol must contain exactly 13 immutable facts")
    for index, (entry, fact) in enumerate(zip(entries, facts, strict=True)):
        prompt_id, fact_type, fact_value = fact
        if (entry["sealed_entry_id"], entry["fact_type"], entry["fact_value"]) != fact:
            raise ValueError(f"Forced-choice fact {index} differs from immutable protocol")
        if entry["fact_id"] != f"{prompt_id}:{fact_type}" or len(entry["candidates"]) < 3:
            raise ValueError(f"Forced-choice fact {index} has invalid identity/candidates")
        if entry["correct_index"] != 0 or entry["decision_span_policy"] != "all candidate token IDs; arithmetic mean log probability":
            raise ValueError(f"Forced-choice fact {index} has invalid scoring policy")


def rank_scores(scores: Sequence[float], correct_index: int = 0) -> dict[str, Any]:
    if len(scores) < 2 or not 0 <= correct_index < len(scores):
        raise ValueError("Candidate score set is invalid")
    correct = scores[correct_index]
    negatives = [score for index, score in enumerate(scores) if index != correct_index]
    rank = 1 + sum(score > correct for score in negatives)
    return {"candidate_mean_log_probabilities": list(scores), "correct_rank": rank,
            "correct_vs_best_negative_margin": correct - max(negatives), "top1": rank == 1}


def rollout_replacement(prefix_ids: Sequence[int], generated_ids: Sequence[int], length: int) -> tuple[list[int], dict[str, int]]:
    if length < 0 or length > len(prefix_ids) or len(generated_ids) != length:
        raise ValueError("Rollout replacement requires exactly N generated IDs and N available suffix IDs")
    start = len(prefix_ids) - length
    replaced = [*prefix_ids[:start], *generated_ids]
    if len(replaced) != len(prefix_ids):
        raise RuntimeError("Rollout replacement moved the decision boundary")
    return replaced, {"retained_end": start, "replaced_start": start, "replaced_end": len(prefix_ids), "decision_start": len(prefix_ids)}


def validate_rollout_context(prefix_ids: Sequence[int], candidate_ids: Sequence[Sequence[int]], length: int, maximum_context: int = 1024) -> None:
    if length < 0 or length > len(prefix_ids):
        raise ValueError("Rollout length exceeds the gold prefix")
    if any(not candidate or len(prefix_ids) + len(candidate) > maximum_context for candidate in candidate_ids):
        raise ValueError("Predetermined candidate decision boundary exceeds maximum context")


def first_divergence(gold_ids: Sequence[int], generated_ids: Sequence[int]) -> int | None:
    for index, (gold, generated) in enumerate(zip(gold_ids, generated_ids, strict=True)):
        if gold != generated:
            return index
    return None


def recovery_error(gold_scores: Sequence[float], recovered_scores: Sequence[float], tolerance: float = 1e-6) -> dict[str, Any]:
    if len(gold_scores) != len(recovered_scores):
        raise ValueError("Recovery score vectors differ in length")
    deltas = [recovered - gold for gold, recovered in zip(gold_scores, recovered_scores, strict=True)]
    maximum = max((abs(value) for value in deltas), default=0.0)
    return {"deltas": deltas, "maximum_absolute_error": maximum, "within_tolerance": maximum <= tolerance}


@torch.no_grad()
def generate_exact_ids(
    model: torch.nn.Module, retained_ids: Sequence[int], count: int, *, device: torch.device,
    mode: str, settings: dict[str, Any], seed: int,
) -> list[int]:
    freeze_model(model)  # type: ignore[arg-type]
    output = list(retained_ids)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(count):
        tokens = torch.tensor(output, dtype=torch.long, device=device).unsqueeze(0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(tokens)[0][:, -1, :]
        if mode == "greedy":
            next_token = int(logits.argmax().item())
        else:
            next_token = sample_next_token(
                logits, temperature=float(settings["temperature"]), top_k=int(settings["top_k"]),
                top_p=float(settings["top_p"]), generator=generator,
            )
        output.append(next_token)
    return output[-count:] if count else []


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prefix_ids(tokenizer: Tokenizer, text: str) -> list[int]:
    story = tokenizer.token_to_id("<|story|>")
    bos = tokenizer.token_to_id("<|bos|>")
    if story is None or bos is None:
        raise ValueError("Tokenizer lacks story/BOS tokens")
    return [story, bos, *tokenizer.encode(text).ids]


def _verify(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise RuntimeError(f"Hash mismatch: {path}")


def _verify_common_sources(config: dict[str, Any]) -> None:
    _verify(Path(config["tokenizer"]["path"]), config["tokenizer"]["sha256"])
    _verify(Path(config["curriculum"]["directory"]) / "manifest.json", config["curriculum"]["manifest_sha256"])
    for spec in config["checkpoints"]:
        _verify(Path(spec["path"]), spec["sha256"])


def _evaluation_provenance(config: dict[str, Any], model_names: Iterable[str]) -> dict[str, Any]:
    specs = {item["id"]: item for item in config["checkpoints"]}
    return {
        "diagnostic_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "tracked_worktree_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()),
        "tokenizer": dict(config["tokenizer"]),
        "checkpoints": {
            name: {
                "path": specs[name]["path"],
                "sha256": specs[name]["sha256"],
                "source_tag": specs[name]["source_tag"],
                "source_commit": specs[name]["source_commit"],
            }
            for name in model_names
        },
    }


def encode_fixed_candidates(
    tokenizer: Tokenizer, prefix_text: str, candidates: Sequence[str]
) -> tuple[list[int], list[list[int]]]:
    context_ids = _prefix_ids(tokenizer, prefix_text)
    candidate_ids = [tokenizer.encode(candidate).ids for candidate in candidates]
    controls = context_ids[:2]
    text_ids = context_ids[2:]
    for candidate, ids in zip(candidates, candidate_ids, strict=True):
        if tokenizer.encode(prefix_text + candidate).ids != [*text_ids, *ids]:
            raise ValueError("Ambiguous forced-choice token boundary")
    return controls + text_ids, candidate_ids


def _load_models(config: dict[str, Any], names: Iterable[str]) -> tuple[dict[str, torch.nn.Module], torch.device]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Actual localization evaluation requires CUDA BF16")
    device = torch.device("cuda")
    models = {}
    specs = {item["id"]: item for item in config["checkpoints"]}
    for name in names:
        spec = specs[name]
        path = Path(spec["path"])
        _verify(path, spec["sha256"])
        models[name] = load_frozen_checkpoint(path, device)[0]
    return models, device


def run_forced_choice(config: dict[str, Any]) -> dict[str, Any]:
    _verify_common_sources(config)
    source_path = Path(config["protocols"]["immutable_narrative"]["path"])
    _verify(source_path, config["protocols"]["immutable_narrative"]["sha256"])
    immutable = _load_json(source_path)
    forced_spec = config["protocols"]["forced_choice"]
    protocol_path = Path(forced_spec["path"])
    _verify(protocol_path, forced_spec["sha256"])
    protocol = _load_json(protocol_path)
    validate_forced_choice_protocol(protocol, immutable, sha256_file(source_path))
    tokenizer = Tokenizer.from_file(config["tokenizer"]["path"])
    model_names = ("compute5", "narrative_v1", "contrastive_v1", "capacity50m")
    models, device = _load_models(config, model_names)
    prompts = {entry["id"]: entry["prompt"] for entry in immutable["entries"]}
    result_models = {}
    for model_name, model in models.items():
        trials = []
        for entry in protocol["entries"]:
            context_ids, candidate_ids = encode_fixed_candidates(
                tokenizer, prompts[entry["sealed_entry_id"]] + entry["bridge"], entry["candidates"]
            )
            scores = score_candidate_ids(model, context_ids, candidate_ids, device=device, use_bf16=True)
            trials.append({"fact_id": entry["fact_id"], **rank_scores(scores)})
        result_models[model_name] = {"top1_total": sum(item["top1"] for item in trials), "denominator": 13, "trials": trials}
    return {
        "score": "mean_log_probability_decision_span",
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "immutable_narrative_protocol": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "provenance": _evaluation_provenance(config, model_names),
        "models": result_models,
    }


def run_context_swap(config: dict[str, Any]) -> dict[str, Any]:
    _verify_common_sources(config)
    tokenizer = Tokenizer.from_file(config["tokenizer"]["path"])
    protocol = build_counterfactual_protocol(Path(config["curriculum"]["directory"]), tokenizer)
    model_names = ("compute5", "narrative_v1", "contrastive_v1")
    models, device = _load_models(config, model_names)
    output: dict[str, Any] = {
        "protocol": protocol,
        "provenance": _evaluation_provenance(config, model_names),
        "models": {},
    }
    for model_name, model in models.items():
        trials = []
        for case in protocol["cases"]:
            scores = {}
            candidate_texts = [case["candidates"][key] for key in ("x", "y")]
            for context_name, context in case["contexts"].items():
                context_ids, candidates = encode_fixed_candidates(tokenizer, context, candidate_texts)
                scores[context_name] = score_candidate_ids(model, context_ids, candidates, device=device, use_bf16=True)
            margin = {key: values[0] - values[1] for key, values in scores.items()}
            trials.append({**{key: case[key] for key in ("id", "family", "distance", "difficulty", "source_split")},
                "scores": scores, "margins_x_minus_y": margin,
                "reversal_correct": margin["a"] > 0 and margin["b"] < 0,
                "context_direction_accuracy": ((margin["a"] > 0) + (margin["b"] < 0)) / 2,
                "mean_signed_correct_margin": (margin["a"] - margin["b"]) / 2,
                "absolute_preference_change": abs(margin["a"] - margin["b"]),
                "removed_absolute_margin": abs(margin["removed"]), "removed_preference_bias": margin["removed"],
                "irrelevant_absolute_margin": abs(margin["irrelevant"]), "irrelevant_preference_bias": margin["irrelevant"],
                "latest_state_update_correct": margin["late_update"] < 0})
        output["models"][model_name] = {"trials": trials, "groups": aggregate_context_swap(trials)}
    return output


def aggregate_context_swap(trials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fields = ("family", "distance", "difficulty")
    metrics = ("reversal_correct", "context_direction_accuracy", "mean_signed_correct_margin", "absolute_preference_change",
               "removed_absolute_margin", "removed_preference_bias", "irrelevant_absolute_margin", "irrelevant_preference_bias", "latest_state_update_correct")
    summarize = lambda items: {
        "count": len(items),
        **{metric: mean(float(item[metric]) for item in items) for metric in metrics},
    }
    output = {"overall": summarize(list(trials))}
    for field in fields:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trial in trials:
            grouped[str(trial[field])].append(trial)
        output[f"by_{field}"] = {key: summarize(items) for key, items in sorted(grouped.items())}
    return output


def run_rollout(config: dict[str, Any]) -> dict[str, Any]:
    _verify_common_sources(config)
    forced_path = Path(config["output_dir"]) / "13fact-forced-choice.json"
    if not forced_path.is_file():
        raise RuntimeError("Rollout requires completed forced-choice output")
    source_spec = config["protocols"]["immutable_narrative"]
    source_path = Path(source_spec["path"])
    _verify(source_path, source_spec["sha256"])
    immutable = _load_json(source_path)
    forced_spec = config["protocols"]["forced_choice"]
    protocol_path = Path(forced_spec["path"])
    _verify(protocol_path, forced_spec["sha256"])
    protocol = _load_json(protocol_path)
    validate_forced_choice_protocol(protocol, immutable, sha256_file(source_path))
    tokenizer = Tokenizer.from_file(config["tokenizer"]["path"])
    prompts = {entry["id"]: entry["prompt"] for entry in immutable["entries"]}
    model_names = ("compute5", "narrative_v1", "contrastive_v1", "capacity50m")
    models, device = _load_models(config, model_names)
    settings = config["rollout"]
    output: dict[str, Any] = {
        "score": "mean_log_probability_decision_span",
        "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "forced_choice_source": {"path": str(forced_path), "sha256": sha256_file(forced_path)},
        "immutable_narrative_protocol": {"path": str(source_path), "sha256": sha256_file(source_path)},
        "rollout_protocol": settings,
        "provenance": _evaluation_provenance(config, model_names),
        "models": {},
    }
    for model_index, (model_name, model) in enumerate(models.items()):
        trials = []
        for case_index, entry in enumerate(protocol["entries"]):
            prefix, candidates = encode_fixed_candidates(
                tokenizer, prompts[entry["sealed_entry_id"]] + entry["bridge"], entry["candidates"]
            )
            gold_scores = score_candidate_ids(model, prefix, candidates, device=device, use_bf16=True)
            gold_rank = rank_scores(gold_scores)
            negatives = [value.strip() for value in entry["candidates"][1:]]
            for mode_index, mode in enumerate(settings["modes"]):
                for length in settings["lengths"]:
                    retained = prefix[: len(prefix) - int(length)]
                    seed = stable_sampling_seed(int(settings["sampled"]["seed"]), case_index, mode_index + model_index * 2)
                    generated = generate_exact_ids(
                        model, retained, int(length), device=device, mode=mode,
                        settings=settings["sampled"], seed=seed,
                    )
                    rolled, boundaries = rollout_replacement(prefix, generated, int(length))
                    rollout_scores = score_candidate_ids(model, rolled, candidates, device=device, use_bf16=True)
                    gold_suffix = prefix[len(prefix) - int(length):] if length else []
                    recovered, _ = rollout_replacement(rolled, gold_suffix, int(length))
                    recovered_scores = score_candidate_ids(model, recovered, candidates, device=device, use_bf16=True)
                    recovery = recovery_error(gold_scores, recovered_scores, float(settings["recovery_tolerance"]))
                    if not recovery["within_tolerance"]:
                        raise RuntimeError("Recovery control failed")
                    generated_text = tokenizer.decode(generated)
                    contradiction = next((negative for negative in negatives if negative in generated_text), None)
                    trials.append({
                        "fact_id": entry["fact_id"], "mode": mode, "length": int(length), "sampling_seed": seed,
                        "gold": gold_rank, "rollout": rank_scores(rollout_scores), "boundaries": boundaries,
                        "first_divergence_from_gold": first_divergence(gold_suffix, generated),
                        "explicit_state_contradiction": contradiction, "recovery": recovery,
                    })
        output["models"][model_name] = {"trials": trials}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("probe", "context-swap", "forced-choice", "rollout"), required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "probe":
        raise SystemExit(f"Run python -m src.frozen_state_probe --config {config['probe_config']}")
    result = run_context_swap(config) if args.stage == "context-swap" else run_forced_choice(config) if args.stage == "forced-choice" else run_rollout(config)
    name = {"context-swap": "counterfactual-context-swap.json", "forced-choice": "13fact-forced-choice.json", "rollout": "teacher-forced-vs-rollout.json"}[args.stage]
    write_json_atomic(output_dir / name, result)


if __name__ == "__main__":
    main()
