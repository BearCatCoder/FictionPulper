"""Deterministic symmetric counterfactual construction, validation, and auditing."""

from __future__ import annotations

import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import yaml
from tokenizers import Tokenizer

from src.continuity_curriculum.common import (
    canonical_json,
    normalized_words,
    sha256_bytes,
    sha256_file,
    stable_seed,
    write_json_atomic,
    write_jsonl,
)
from src.continuity_curriculum.generator import (
    FAMILY_BRIDGES,
    FILLER_DETAILS,
    GENRES,
    GENRE_CONTROL_CANDIDATES,
    GENRE_SCENES,
    SPLITS,
)


SCHEMA_VERSION = 1
GENERATOR_VERSION = "counterfactual-narrative-v1"
MAIN_SPLITS = ("train", "validation", "test")


class CounterfactualValidationError(ValueError):
    """Raised when a pair is not an exact, provable counterfactual."""


def _hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _control(tokenizer: Tokenizer, genre: str) -> str | None:
    return next((value for value in GENRE_CONTROL_CANDIDATES[genre]
                 if tokenizer.token_to_id(value) is not None), None)


def _render_state(family: str, slots: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    """Return entity, attribute, value0, value1, sentence0, sentence1 templates."""
    a, b, _ = slots["names"]
    obj = slots["objects"][0]
    origin, destination = slots["locations"]
    if family == "object_ownership":
        return obj, "owner", a, b, f"{a} took ownership of the {obj}.", f"{b} took ownership of the {obj}."
    if family == "object_location":
        return obj, "location", origin, destination, f"The {obj} was placed at the {origin}.", f"The {obj} was placed at the {destination}."
    if family == "character_location":
        return a, "location", origin, destination, f"{a} arrived at the {origin}.", f"{a} arrived at the {destination}."
    if family == "goal_intention":
        first, second = slots["goals"]
        return a, "goal", first, second, f"{a} resolved to {first}.", f"{a} resolved to {second}."
    if family == "knowledge_secret_holder":
        secret = slots["secret"]
        return secret, "known_by", a, b, f"Only {a} learned that {secret}.", f"Only {b} learned that {secret}."
    if family == "relationship":
        return a, f"relationship_to:{b}", "trusted ally", "declared rival", f"{a} became {b}'s trusted ally.", f"{a} became {b}'s declared rival."
    if family == "persistent_physical_state":
        injury = slots["injury"]
        return a, "physical_state", f"still affected by the {injury}", f"fully recovered from the {injury}", f"{a} remained affected by the {injury}.", f"{a} fully recovered from the {injury}."
    if family == "object_door_state":
        return obj, "closure_state", "locked", "open", f"The case holding the {obj} was locked.", f"The case holding the {obj} was open."
    if family == "cause_effect":
        effect = slots["effect"]
        first, second = slots["causes"]
        return effect, "caused_by", first, second, f"The inquiry found that {first} caused the fact that {effect}.", f"The inquiry found that {second} caused the fact that {effect}."
    if family == "latest_state_update":
        return obj, "latest_location", origin, destination, f"The latest report moved the {obj} to the {origin}.", f"The latest report moved the {obj} to the {destination}."
    raise CounterfactualValidationError(f"Unsupported state family: {family}")


def _candidate(entity: str, attribute: str, value: str) -> str:
    if attribute in {"location", "latest_location"}:
        return f"At that moment, the {entity} was at the {value}."
    if attribute == "owner":
        return f"At that moment, the {entity} belonged to {value}."
    if attribute == "goal":
        return f"At that moment, {entity} intended to {value}."
    if attribute == "known_by":
        return f"At that moment, the secret was known only by {value}."
    if attribute.startswith("relationship_to:"):
        other = attribute.split(":", 1)[1]
        return f"At that moment, {entity} was the {value} of {other}."
    if attribute == "physical_state":
        return f"At that moment, {entity} was {value}."
    if attribute == "closure_state":
        return f"At that moment, the case holding the {entity} was {value}."
    if attribute == "caused_by":
        return f"The recorded effect occurred because {value}."
    raise CounterfactualValidationError(f"No continuation for attribute {attribute}")


def _encoded_world(tokenizer: Tokenizer, context: str, candidate: str, control: str | None) -> list[int]:
    prefix = [tokenizer.token_to_id("<|story|>")]
    if control:
        prefix.append(tokenizer.token_to_id(control))
    prefix.append(tokenizer.token_to_id("<|bos|>"))
    eos = tokenizer.token_to_id("<|eos|>")
    if any(value is None for value in [*prefix, eos]):
        raise CounterfactualValidationError("tokenizer-v1 lacks required special tokens")
    return [*prefix, *tokenizer.encode(context + candidate).ids, eos]  # type: ignore[list-item]


def generate_pair(config: dict[str, Any], source: dict[str, Any], tokenizer: Tokenizer,
                  *, split: str, ordinal: int) -> dict[str, Any]:
    rng = random.Random(stable_seed(int(config["seed"]), split, ordinal))
    pools = source["holdouts"][split]
    names = rng.sample(pools["names"], 3)
    objects = rng.sample(pools["objects"], 2)
    locations = rng.sample(pools["locations"], 2)
    family = config["state_families"][ordinal % len(config["state_families"])]
    template = config["pair_templates"][split][(ordinal // len(config["state_families"])) % len(config["pair_templates"][split])]
    genre = GENRES[(ordinal + SPLITS.index(split)) % len(GENRES)]
    difficulty = ordinal % 5 + 1
    compatible = source["difficulty_distance_buckets"][str(difficulty)]
    distance_bucket = compatible[(ordinal // 5) % len(compatible)]
    distance = source["distance_buckets"][distance_bucket]
    lexical_combination = pools["lexical_combinations"][ordinal % len(pools["lexical_combinations"])]
    slots = {
        "names": names,
        "objects": objects,
        "locations": locations,
        "goals": rng.sample(pools["goals"], 2),
        "secret": pools["secrets"][ordinal % len(pools["secrets"])],
        "injury": pools["injuries"][ordinal % len(pools["injuries"])],
        "causes": rng.sample(pools["causes"], 2),
        "effect": pools["effects"][ordinal % len(pools["effects"])],
        "lexical_combination": lexical_combination,
    }
    entity, attribute, value0, value1, sentence0, sentence1 = _render_state(family, slots)
    semantic = [(value0, sentence0), (value1, sentence1)]
    orientation = (
        ordinal + ordinal // len(config["state_families"]) + SPLITS.index(split)
    ) % 2
    if orientation:
        semantic.reverse()
    candidates = {
        "X": {"text": " " + _candidate(entity, attribute, semantic[0][0])},
        "Y": {"text": " " + _candidate(entity, attribute, semantic[1][0])},
    }
    for candidate in candidates.values():
        candidate["token_ids"] = tokenizer.encode(candidate["text"]).ids

    opening = GENRE_SCENES[genre][ordinal % len(GENRE_SCENES[genre])]
    bridge = FAMILY_BRIDGES[template]
    irrelevant = f"{names[2]} checked the weather and entered scene number {ordinal + 2} in the road ledger."
    worlds: dict[str, dict[str, Any]] = {}
    world_distances: dict[str, int] = {}
    for world_index, world_name in enumerate(("A", "B")):
        final_index = world_index if orientation == 0 else 1 - world_index
        initial_index = 1 - final_index
        initial_sentence = semantic[initial_index][1]
        decisive = semantic[final_index][1]
        state_sentences = (
            (initial_sentence, decisive)
            if family == "latest_state_update"
            else (decisive,)
        )
        base = " ".join((opening, bridge, *state_sentences, irrelevant))
        filler: list[str] = []
        observed = 0
        while True:
            context = " ".join((base, *filler))
            decisive_end = context.index(decisive) + len(decisive)
            observed = len(tokenizer.encode(context[decisive_end:]).ids)
            if distance["minimum"] <= observed <= distance["maximum"]:
                break
            if observed > distance["maximum"] or len(filler) > 150:
                raise RuntimeError(f"Could not realize {distance_bucket} for {split}:{ordinal}")
            filler.append(FILLER_DETAILS[(ordinal * 3 + len(filler)) % len(FILLER_DETAILS)])
        start = context.index(decisive)
        correct = "X" if final_index == 0 else "Y"
        token_start = len(tokenizer.encode(context[:start]).ids)
        token_end = len(tokenizer.encode(context[: start + len(decisive)]).ids)
        control = _control(tokenizer, genre)
        encoded = _encoded_world(tokenizer, context, candidates[correct]["text"], control)
        if len(encoded) > int(config["maximum_world_tokens"]):
            raise RuntimeError(f"World exceeds context limit: {split}:{ordinal}:{world_name}")
        context_ids = tokenizer.encode(context).ids
        context_prefix_tokens = 2 + int(control is not None) + len(context_ids)
        candidate_encodings = {}
        for candidate_name, candidate in candidates.items():
            candidate_world = _encoded_world(tokenizer, context, candidate["text"], control)
            if candidate_world[2 + int(control is not None):context_prefix_tokens] != context_ids:
                raise CounterfactualValidationError("ambiguous context/candidate token boundary")
            candidate_encodings[candidate_name] = {
                "token_ids": candidate_world,
                "decision_start": context_prefix_tokens,
                "decision_end": len(candidate_world) - 1,
            }
        world_distances[world_name] = observed
        worlds[world_name] = {
            "context": context,
            "context_token_ids": tokenizer.encode(context).ids,
            "correct_candidate": correct,
            "decisive_sentence": decisive,
            "decision_span": {"byte_start": start, "byte_end": start + len(decisive),
                              "token_start": token_start, "token_end": token_end,
                              "token_ids": tokenizer.encode(decisive).ids},
            "positive_world_token_count": len(encoded),
            "anchor_distance_tokens": observed,
            "candidate_encodings": candidate_encodings,
            "state_events": (
                [
                    {"sequence": 0, "value": semantic[initial_index][0], "evidence": initial_sentence},
                    {"sequence": 1, "value": semantic[final_index][0], "evidence": decisive},
                ]
                if family == "latest_state_update"
                else [{"sequence": 0, "value": semantic[final_index][0], "evidence": decisive}]
            ),
            "final_value": semantic[final_index][0],
        }

    removed_context = " ".join((opening, bridge, irrelevant))
    replacement = f"{names[2]} noted that the lamps along the road had been cleaned that morning."
    irrelevant_context = " ".join((opening, bridge, replacement, irrelevant))
    pair: dict[str, Any] = {
        "pair_id": "cfn-v1-" + _hash([config["seed"], split, ordinal, family, template])[:20],
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "split": split,
        "worlds": worlds,
        "candidates": candidates,
        "abstract_counterfactual_variable": {"state_family": family, "entity": entity,
                                              "attribute": attribute, "values": [value0, value1]},
        "state_proof": {"rule": "last event for entity/attribute wins",
                        "opposite_final_values": [worlds["A"]["final_value"], worlds["B"]["final_value"]]},
        "controls": {
            "fact_removed_context": removed_context,
            "fact_removed_token_ids": tokenizer.encode(removed_context).ids,
            "irrelevant_sentence": replacement,
            "irrelevant_substituted_context": irrelevant_context,
            "irrelevant_substituted_token_ids": tokenizer.encode(irrelevant_context).ids,
        },
        "metadata": {
            "genre": genre, "genre_control_token": _control(tokenizer, genre),
            "template_family": template, "distance_bucket": distance_bucket,
            "difficulty": difficulty, "anchor_distance_tokens": world_distances,
            "lexical_slots": slots,
        },
    }
    pair["stable_hash"] = _hash(pair)
    validate_pair(pair, config=config, source=source, tokenizer=tokenizer)
    return pair


def validate_pair(pair: dict[str, Any], *, config: dict[str, Any], source: dict[str, Any],
                  tokenizer: Tokenizer) -> None:
    supplied_hash = pair.get("stable_hash")
    unhashed = {key: value for key, value in pair.items() if key != "stable_hash"}
    if supplied_hash != _hash(unhashed):
        raise CounterfactualValidationError("stable hash mismatch")
    if pair.get("schema_version") != SCHEMA_VERSION or pair.get("generator_version") != GENERATOR_VERSION:
        raise CounterfactualValidationError("schema or generator version mismatch")
    worlds = pair.get("worlds", {})
    candidates = pair.get("candidates", {})
    if set(worlds) != {"A", "B"} or set(candidates) != {"X", "Y"}:
        raise CounterfactualValidationError("pair must contain A/B worlds and shared X/Y candidates")
    if worlds["A"]["correct_candidate"] == worlds["B"]["correct_candidate"]:
        raise CounterfactualValidationError("world labels do not reverse")
    variable = pair["abstract_counterfactual_variable"]
    if variable["state_family"] not in config["state_families"]:
        raise CounterfactualValidationError("unknown state family")
    if set(pair["state_proof"]["opposite_final_values"]) != set(variable["values"]):
        raise CounterfactualValidationError("state proof does not contain opposite values")
    for world_name, world in worlds.items():
        context = world["context"]
        if tokenizer.encode(context).ids != world["context_token_ids"]:
            raise CounterfactualValidationError(f"{world_name} context tokenization mismatch")
        span = world["decision_span"]
        decisive = world["decisive_sentence"]
        if context[span["byte_start"]:span["byte_end"]] != decisive:
            raise CounterfactualValidationError(f"{world_name} decisive byte span mismatch")
        if tokenizer.encode(decisive).ids != span["token_ids"]:
            raise CounterfactualValidationError(f"{world_name} decisive token encoding mismatch")
        expected_token_span = (
            len(tokenizer.encode(context[:span["byte_start"]]).ids),
            len(tokenizer.encode(context[:span["byte_end"]]).ids),
        )
        if (span["token_start"], span["token_end"]) != expected_token_span:
            raise CounterfactualValidationError(f"{world_name} decisive token span mismatch")
        expected_sequences = [0, 1] if variable["state_family"] == "latest_state_update" else [0]
        if [event["sequence"] for event in world["state_events"]] != expected_sequences:
            raise CounterfactualValidationError(f"{world_name} state history is malformed")
        if world["state_events"][-1]["value"] != world["final_value"]:
            raise CounterfactualValidationError(f"{world_name} violates latest-state semantics")
        correct = world["correct_candidate"]
        expected_text = " " + _candidate(variable["entity"], variable["attribute"], world["final_value"])
        if candidates[correct]["text"] != expected_text:
            raise CounterfactualValidationError(f"{world_name} label is not proved by final state")
        count = len(_encoded_world(tokenizer, context, candidates[correct]["text"],
                                   pair["metadata"]["genre_control_token"]))
        if count != world["positive_world_token_count"] or count > config["maximum_world_tokens"]:
            raise CounterfactualValidationError(f"{world_name} world token count mismatch")
        bucket = source["distance_buckets"][pair["metadata"]["distance_bucket"]]
        if not bucket["minimum"] <= world["anchor_distance_tokens"] <= bucket["maximum"]:
            raise CounterfactualValidationError(f"{world_name} distance outside configured bucket")
        if pair["metadata"]["anchor_distance_tokens"].get(world_name) != world["anchor_distance_tokens"]:
            raise CounterfactualValidationError(f"{world_name} distance metadata mismatch")
        for candidate_name, candidate in candidates.items():
            encoding = world["candidate_encodings"][candidate_name]
            expected = _encoded_world(
                tokenizer, context, candidate["text"], pair["metadata"]["genre_control_token"]
            )
            if encoding["token_ids"] != expected:
                raise CounterfactualValidationError(f"{world_name}/{candidate_name} encoding mismatch")
            start, end = int(encoding["decision_start"]), int(encoding["decision_end"])
            if not 0 < start < end == len(expected) - 1:
                raise CounterfactualValidationError(f"{world_name}/{candidate_name} decision span mismatch")
            if expected[start:end] != candidate["token_ids"]:
                raise CounterfactualValidationError(f"{world_name}/{candidate_name} candidate span mismatch")
    for candidate in candidates.values():
        if tokenizer.encode(candidate["text"]).ids != candidate["token_ids"]:
            raise CounterfactualValidationError("candidate tokenization mismatch")
    controls = pair["controls"]
    if any(event["evidence"] in controls["fact_removed_context"]
           for world in worlds.values() for event in world["state_events"]):
        raise CounterfactualValidationError("fact-removal control retains state evidence")
    if controls["irrelevant_sentence"] in worlds["A"]["context"] or controls["irrelevant_sentence"] in worlds["B"]["context"]:
        raise CounterfactualValidationError("irrelevant replacement leaked into normal worlds")
    if controls["irrelevant_sentence"] not in controls["irrelevant_substituted_context"]:
        raise CounterfactualValidationError("irrelevant replacement control is missing")
    if any(event["evidence"] in controls["irrelevant_substituted_context"]
           for world in worlds.values() for event in world["state_events"]):
        raise CounterfactualValidationError("irrelevant replacement retains state evidence")


def classifier_examples(pairs: list[dict[str, Any]], feature: str) -> list[tuple[tuple[Any, ...], int, str]]:
    examples = []
    for pair in pairs:
        for candidate_name in ("X", "Y"):
            candidate = pair["candidates"][candidate_name]
            values = normalized_words(candidate["text"]) if feature == "word" else candidate["token_ids"]
            bag = tuple(sorted(Counter(values).items(), key=lambda item: str(item[0])))
            for world_name in ("A", "B"):
                label = int(pair["worlds"][world_name]["correct_candidate"] == candidate_name)
                examples.append((bag, label, f"{pair['pair_id']}:{world_name}:{candidate_name}"))
    return examples


def _logistic(train: list[tuple[tuple[Any, ...], int, str]], test: list[tuple[tuple[Any, ...], int, str]],
              *, epochs: int, rate: float) -> dict[str, Any]:
    weights: defaultdict[Any, float] = defaultdict(float)
    bias = 0.0
    for _ in range(epochs):
        gradients: defaultdict[Any, float] = defaultdict(float)
        bias_gradient = 0.0
        for bag, label, _ in train:
            score = bias + sum(weights[key] * count for key, count in bag)
            probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
            error = probability - label
            bias_gradient += error
            for key, count in bag:
                gradients[key] += error * count
        scale = rate / max(1, len(train))
        bias -= scale * bias_gradient
        for key, gradient in gradients.items():
            weights[key] -= scale * gradient
    correct = 0
    for bag, label, _ in test:
        score = bias + sum(weights[key] * count for key, count in bag)
        correct += int((score >= 0.0) == bool(label))
    return {"train_examples": len(train), "evaluation_examples": len(test),
            "accuracy": correct / len(test) if test else None,
            "nonzero_weights": sum(abs(value) > 1e-12 for value in weights.values()),
            "bias": bias}


def _balance(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions: dict[str, Callable[[dict], str]] = {
        "overall": lambda pair: "all",
        "state_family": lambda pair: pair["abstract_counterfactual_variable"]["state_family"],
        "template_family": lambda pair: pair["metadata"]["template_family"],
    }
    reports = {}
    passed = True
    for dimension, key in dimensions.items():
        groups: defaultdict[str, Counter] = defaultdict(Counter)
        for pair in pairs:
            for world in pair["worlds"].values():
                groups[key(pair)][world["correct_candidate"]] += 1
        values = {}
        for group, counts in sorted(groups.items()):
            total = sum(counts.values())
            share = counts["X"] / total
            a_pairs = [pair for pair in pairs if key(pair) == group]
            a_x = sum(pair["worlds"]["A"]["correct_candidate"] == "X" for pair in a_pairs)
            values[group] = {"X": counts["X"], "Y": counts["Y"], "X_share": share,
                             "world_A_X": a_x, "world_A_Y": len(a_pairs) - a_x}
            passed &= counts["X"] == counts["Y"]
            passed &= abs(a_x - (len(a_pairs) - a_x)) <= 1
        reports[dimension] = values
    return {"passed": passed, "dimensions": reports}


def audit_dataset(pairs: list[dict[str, Any]], config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    pair_ids = [pair["pair_id"] for pair in pairs]
    hashes = [pair["stable_hash"] for pair in pairs]
    split_by_id = defaultdict(set)
    lexical_owners: dict[str, defaultdict[str, set[str]]] = {
        key: defaultdict(set) for key in (
            "names", "objects", "locations", "goals", "secret", "injury", "causes",
            "effect", "lexical_combination",
        )
    }
    for pair in pairs:
        split_by_id[pair["pair_id"]].add(pair["split"])
        slots = pair["metadata"]["lexical_slots"]
        for key, owners in lexical_owners.items():
            value = slots[key]
            for item in value if isinstance(value, list) else [value]:
                owners[item].add(pair["split"])
    leakage = {key: {value: sorted(splits) for value, splits in owners.items() if len(splits) > 1}
               for key, owners in lexical_owners.items()}
    split_report = {"passed": not any(leakage.values()) and all(len(value) == 1 for value in split_by_id.values()),
                    "pair_split_violations": sorted(key for key, value in split_by_id.items() if len(value) != 1),
                    "lexical_leakage": leakage,
                    "templates_by_split": {split: sorted({pair["metadata"]["template_family"] for pair in pairs if pair["split"] == split}) for split in SPLITS}}
    candidate_owners: defaultdict[str, set[str]] = defaultdict(set)
    context_owners: defaultdict[str, list[str]] = defaultdict(list)
    for pair in pairs:
        for candidate in pair["candidates"].values():
            candidate_owners[_hash(candidate["text"])].add(pair["pair_id"])
        for world_name, world in pair["worlds"].items():
            context_owners[_hash(" ".join(normalized_words(world["context"])))].append(
                f"{pair['pair_id']}:{world_name}"
            )
    duplicate_contexts = {digest: owners for digest, owners in context_owners.items() if len(owners) > 1}
    duplicates = {"passed": len(pair_ids) == len(set(pair_ids)) and len(hashes) == len(set(hashes)) and not duplicate_contexts,
                  "duplicate_pair_ids": [key for key, count in Counter(pair_ids).items() if count > 1],
                  "duplicate_stable_hashes": [key for key, count in Counter(hashes).items() if count > 1],
                  "duplicate_context_groups": duplicate_contexts,
                  "cross_pair_candidate_duplicates": sum(len(owners) > 1 for owners in candidate_owners.values())}
    tokens_by: dict[str, Counter] = {key: Counter() for key in ("split", "state_family", "template_family", "genre", "distance_bucket", "difficulty")}
    for pair in pairs:
        amount = sum(world["positive_world_token_count"] for world in pair["worlds"].values())
        metadata = pair["metadata"]
        values = {"split": pair["split"], "state_family": pair["abstract_counterfactual_variable"]["state_family"],
                  "template_family": metadata["template_family"], "genre": metadata["genre"],
                  "distance_bucket": metadata["distance_bucket"], "difficulty": str(metadata["difficulty"])}
        for key, value in values.items():
            tokens_by[key][value] += amount
    train = [pair for pair in pairs if pair["split"] == "train"]
    classifier_reports = {}
    threshold = float(config["audit"]["classifier_acceptance_accuracy"])
    for feature in ("word", "token_id"):
        train_examples = classifier_examples(train, "word" if feature == "word" else "token")
        heldouts = {}
        for split in ("test", "generalization_holdout"):
            evaluation = classifier_examples([pair for pair in pairs if pair["split"] == split], "word" if feature == "word" else "token")
            result = _logistic(train_examples, evaluation, epochs=int(config["audit"]["classifier_epochs"]), rate=float(config["audit"]["classifier_learning_rate"]))
            result["passed"] = result["accuracy"] is not None and result["accuracy"] <= threshold
            heldouts[split] = result
        cancellation = all(
            (counts := Counter(label for observed_bag, label, _ in train_examples if observed_bag == bag))[0]
            == counts[1] > 0
            for bag in {example[0] for example in train_examples}
        )
        classifier_reports[feature] = {"passed": cancellation and all(item["passed"] for item in heldouts.values()),
                                       "exact_feature_label_cancellation": cancellation, "heldouts": heldouts}
    controls = {"passed": all(
        all(event["evidence"] not in pair["controls"]["fact_removed_context"]
            for world in pair["worlds"].values() for event in world["state_events"])
        and pair["controls"]["irrelevant_sentence"] in pair["controls"]["irrelevant_substituted_context"]
        and pair["controls"]["irrelevant_sentence"] not in pair["worlds"]["A"]["context"]
        and pair["controls"]["irrelevant_sentence"] not in pair["worlds"]["B"]["context"]
        for pair in pairs), "records_checked": len(pairs)}
    update_pairs = [pair for pair in pairs
                    if pair["abstract_counterfactual_variable"]["state_family"] == "latest_state_update"]
    latest_state = {"passed": bool(update_pairs) and all(
        len(world["state_events"]) == 2
        and world["state_events"][-1]["value"] == world["final_value"]
        and world["state_events"][0]["value"] != world["state_events"][-1]["value"]
        for pair in update_pairs for world in pair["worlds"].values()),
        "worlds_checked": len(update_pairs) * 2}
    surface_labels: defaultdict[str, Counter] = defaultdict(Counter)
    for pair in pairs:
        for candidate_name, candidate in pair["candidates"].items():
            for world in pair["worlds"].values():
                surface_labels[_hash(candidate["text"])][
                    "correct" if world["correct_candidate"] == candidate_name else "incorrect"
                ] += 1
    surface_balance = {"passed": all(counts["correct"] == counts["incorrect"] for counts in surface_labels.values()),
                       "unique_continuation_surfaces": len(surface_labels),
                       "imbalanced_surface_hashes": sorted(digest for digest, counts in surface_labels.items()
                                                            if counts["correct"] != counts["incorrect"])}
    reports = {"construction": {"passed": True, "validated_pairs": len(pairs)},
               "latest_state_semantics": latest_state,
               "continuation_surface_balance": surface_balance,
               "balance": _balance(pairs), "split_isolation": split_report,
               "duplicates": duplicates, "controls": controls,
               "shortcut_classifiers": {"passed": all(value["passed"] for value in classifier_reports.values()),
                                          "acceptance_accuracy": threshold, "classifiers": classifier_reports},
               "token_totals": {"passed": bool(pairs), "positive_tokens_by": {key: dict(sorted(value.items())) for key, value in tokens_by.items()}},
               "lexical_leakage_tabulation": {"passed": not any(leakage.values()), "dimensions": leakage}}
    reports["passed"] = all(report["passed"] for key, report in reports.items() if key != "passed")
    return reports


def _validate_config(config: dict[str, Any], source: dict[str, Any]) -> None:
    if config["output_path"] != "data/counterfactual_narrative_v1":
        raise ValueError("Counterfactual-v1 output path is locked")
    if tuple(config["splits"]["main"]) != MAIN_SPLITS or config["splits"]["main"] != {"train": 0.9, "validation": 0.05, "test": 0.05}:
        raise ValueError("Main splits must be deterministic 90/5/5")
    if set(config["pair_templates"]) != set(SPLITS):
        raise ValueError("Every split requires an exclusive template pool")
    seen = set()
    for split in SPLITS:
        templates = set(config["pair_templates"][split])
        if seen & templates or not templates <= set(source["holdouts"][split]["template_families"]):
            raise ValueError("Pair templates must be split-exclusive Narrative-v1 templates")
        seen |= templates
        pools = source["holdouts"][split]
        if len(pools["goals"]) < 2 or len(pools["causes"]) < 2:
            raise ValueError(f"{split} needs at least two goals and causes")


def build_dataset(config_path: Path, *, max_pairs: int | None = None) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = Path(config["source_narrative_config"])
    if sha256_file(source_path) != config["source_narrative_config_sha256"]:
        raise RuntimeError("sealed Narrative-v1 source config hash changed")
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    _validate_config(config, source)
    tokenizer_path = Path(config["tokenizer"]["path"])
    if sha256_file(tokenizer_path) != config["tokenizer"]["expected_sha256"]:
        raise RuntimeError("tokenizer-v1 hash changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    targets = {split: round(config["target"]["main_positive_tokens"] * config["splits"]["main"][split]) for split in MAIN_SPLITS}
    targets["generalization_holdout"] = config["target"]["generalization_positive_tokens"]
    counts = Counter()
    ordinals = Counter()
    pairs = []
    cycle = 0
    while True:
        if max_pairs is not None:
            if len(pairs) >= max_pairs:
                break
            split = SPLITS[cycle % len(SPLITS)]
            cycle += 1
        else:
            remaining = [split for split in SPLITS if counts[split] < targets[split]]
            if not remaining:
                break
            split = max(remaining, key=lambda value: targets[value] - counts[value])
        pair = generate_pair(config, source, tokenizer, split=split, ordinal=ordinals[split])
        ordinals[split] += 1
        pairs.append(pair)
        counts[split] += sum(world["positive_world_token_count"] for world in pair["worlds"].values())
    report = audit_dataset(pairs, config, source)
    main_tokens = sum(counts[split] for split in MAIN_SPLITS)
    target_ok = config["target"]["allowed_main_positive_token_range"][0] <= main_tokens <= config["target"]["allowed_main_positive_token_range"][1]
    safe_to_train = bool(report["passed"] and target_ok and max_pairs is None)
    status = "PASS" if safe_to_train else "STOP"
    output = Path(config["output_path"])
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(f"Refusing to mutate existing artifact: {output}")
        output.rmdir()
    staging = output.with_name(output.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    files = []
    for split in SPLITS:
        path = Path(f"{split}.jsonl")
        write_jsonl(staging / path, (pair for pair in pairs if pair["split"] == split))
        files.append(path)
    write_json_atomic(staging / "audit-report.json", report)
    files.append(Path("audit-report.json"))
    manifest = {
        "dataset": "FictionPulper Counterfactual-v1", "schema_version": SCHEMA_VERSION,
        "status": status, "safe_to_train": safe_to_train,
        "development_build": max_pairs is not None, "development_max_pairs": max_pairs,
        "pair_count": len(pairs), "positive_token_counts": dict(counts),
        "main_positive_tokens": main_tokens, "audits_passed": report["passed"],
        "target_range_passed": target_ok, "config_sha256": sha256_file(config_path),
        "source_narrative_config_sha256": sha256_file(source_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "files": {str(path): {"sha256": sha256_file(staging / path), "bytes": (staging / path).stat().st_size} for path in files},
    }
    write_json_atomic(staging / "manifest.json", manifest)
    staging.replace(output)
    return manifest
