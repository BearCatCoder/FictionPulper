"""Deterministically author and audit Narrative Benchmark v2 scenarios."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml
from tokenizers import Tokenizer

from src.benchmark_v2 import load_jsonl, validate_benchmark
from src.continuity_curriculum.common import (
    canonical_json,
    normalized_text_hash,
    normalized_words,
    sha256_bytes,
    sha256_file,
    text_shingles,
)


SPLITS = ("development", "test", "generalization")
SCENARIO_FILES = {split: f"{split}.jsonl" for split in SPLITS}
CONTROL_FILES = {split: f"{split}-candidate-controls.jsonl" for split in SPLITS}
OUTPUT_FILES = (
    *SCENARIO_FILES.values(),
    *CONTROL_FILES.values(),
    "authoring-report.json",
    "authoring-validation-results.md",
    "authored-manifest.json",
)
JOIN_FIELDS = (
    "scenario_id", "schema_version", "split", "state_family", "reasoning_depth", "depth_bucket",
    "template_id", "template_family_id", "lexical_pool_id", "name_pool_id", "verb_pool_id",
    "candidate_presentation_order", "evaluation_modes", "novelty",
)
FAMILY_OPERATIONS = {
    "static_fact": "infer_annotation",
    "ownership": "transfer_title",
    "location": "follow_route",
    "knowledge": "relay_recipient",
    "goal": "advance_objective",
    "causal_dependency": "propagate_cause",
    "temporal_ordering": "advance_precedence",
}
_TRAINING_AUDIT_CACHE: dict[str, dict[str, Any]] = {}
SURFACE_FIELDS = (
    "context", "gold_continuation", "counterfactual_continuation", "free_generation_prompt",
    "fact_removed_context", "irrelevant_substituted_context", "candidate",
)


class AuthoringValidationError(ValueError):
    """Raised when authored data violates the v2 authoring contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthoringValidationError(message)


def authored_stable_hash(record: dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "stable_hash"}
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def _choice(values: list[str], ordinal: int, offset: int = 0) -> str:
    return values[(ordinal + offset) % len(values)]


def _mapping_text(mapping: dict[str, str], family: dict[str, str], step: int) -> str:
    left, right = mapping.items()
    return (
        f"{family['transition_noun'].title()} {step} maps {left[0]} to {left[1]} "
        f"and maps {right[0]} to {right[1]}."
    )


def _render_development(
    *, style: str, actor: str, verb: str, witness: str, family: dict[str, str], initial: str,
    rule_texts: list[str], fact_present: bool = True, irrelevant: str | None = None,
) -> str:
    premise = (
        f"The {style} records that {actor} {verb} {initial} into the {family['subject']} chain."
        if fact_present else f"The decisive {family['subject']} line in the {style} has been erased."
    )
    if irrelevant is not None:
        premise = f"The {style} mentions an unrelated marker, {irrelevant}, beside the {family['subject']}."
    rules = " ".join(rule_texts)
    return " ".join(part for part in (premise, rules, f"{witness} follows every entry before acting.") if part)


def _render_test(
    *, style: str, actor: str, verb: str, witness: str, family: dict[str, str], initial: str,
    rule_texts: list[str], fact_present: bool = True, irrelevant: str | None = None,
) -> str:
    if fact_present:
        premise = f"WITNESS {actor}: I {verb} {initial} as the opening {family['state_noun']}."
    else:
        premise = f"WITNESS [REDACTED]: the opening {family['state_noun']} is unavailable."
    if irrelevant is not None:
        premise = f"WITNESS {actor}: the weather card showed {irrelevant}; it was not an opening {family['state_noun']}."
    rules = " ".join(f"RULE-{index + 1} :: {text}" for index, text in enumerate(rule_texts))
    return f"TESTIMONY/{style.upper()} | {premise} {rules} DECISION HELD FOR {witness}."


def _render_generalization(
    *, style: str, actor: str, verb: str, witness: str, family: dict[str, str], initial: str,
    rule_texts: list[str], fact_present: bool = True, irrelevant: str | None = None,
) -> str:
    if fact_present:
        premise = f"<{style}:{actor}> seed({family['state_noun']}) := {initial} after {verb}."
    else:
        premise = f"<{style}:unknown> seed({family['state_noun']}) := MISSING."
    if irrelevant is not None:
        premise = f"<{style}:{actor}> weather_note := {irrelevant}; seed({family['state_noun']}) := MISSING."
    rules = " ".join(f"TRANSFORM[{index + 1}]({text})" for index, text in enumerate(rule_texts))
    return f"{premise} {rules} RESOLVE[{witness}] only after terminal transform."


RENDERERS = {
    "development_prose": _render_development,
    "test_transcript": _render_test,
    "generalization_symbolic": _render_generalization,
}


def _world_values(lexicon: list[str], ordinal: int, depth: int) -> tuple[list[str], list[str], str]:
    start = (ordinal * 3) % len(lexicon)
    rotated = lexicon[start:] + lexicon[:start]
    needed = depth * 2 + 1
    _require(len(rotated) >= needed, "lexicon cannot support requested proof depth")
    return rotated[0 : depth * 2 : 2], rotated[1 : depth * 2 : 2], rotated[depth * 2]


def _candidate_text(split: str, witness: str, state_noun: str, value: str) -> str:
    if split == "development":
        return f" {witness} acts on the resolved {state_noun} {value}."
    if split == "test":
        return f" DECISION::{state_noun}::{value} is assigned to {witness}."
    return f" OUTPUT<{witness}|{state_noun}>={value}."


def _world_surface_text(
    split: str, *, candidate: str, style: str, actor: str, subject: str,
    state_noun: str, transition_noun: str, answer: str, other: str, context: str, marker: str,
) -> tuple[str, str, str, str, str, str]:
    if split == "development":
        gold = candidate + f" The {style} resolution of {actor}'s {subject} {transition_noun} chain terminates at {answer}."
        counter = candidate + f" The {style} resolution for {actor}'s {subject} {transition_noun} would instead terminate at {other}."
        prompt = context + " Continue from the unresolved decision without restating the rules."
        required = f"For {actor}'s {style} case about the {subject}, the {transition_noun} yields terminal {state_noun} {answer}."
        forbidden = f"For {actor}'s {style} case about the {subject}, the {transition_noun} yields terminal {state_noun} {other}."
        review = f"Does {actor}'s {marker} continuation about the {subject} preserve {answer} after the {transition_noun} chain?"
    elif split == "test":
        gold = candidate + f" TESTIMONY {style}/{actor}/{subject} CONFIRMS {answer} AFTER {transition_noun}."
        counter = candidate + f" TESTIMONY {style}/{actor}/{subject} WOULD REQUIRE {other}, CONTRARY TO {transition_noun}."
        prompt = context + " NARRATE THE NEXT TESTIMONY ENTRY; DO NOT QUOTE A RULE."
        required = f"Adjudication {marker} accepts {answer} after {actor}'s {subject} {transition_noun}."
        forbidden = f"Adjudication {marker} accepts {other} after {actor}'s {subject} {transition_noun}."
        review = f"Did {actor}'s {marker} testimony resolve the {subject} {state_noun} through {transition_noun} to {answer}?"
    else:
        gold = candidate + f" TERMINAL_ASSERT({actor},{subject},{transition_noun},{state_noun}=={answer})."
        counter = candidate + f" INVALID_BRANCH({actor},{subject},{transition_noun},{state_noun}=={other})."
        prompt = context + " EMIT_CONTINUATION[without_transform_repetition]."
        required = f"terminal({actor},{marker},{subject},{transition_noun},{state_noun},{answer})"
        forbidden = f"terminal({actor},{marker},{subject},{transition_noun},{state_noun},{other})"
        review = f"VERIFY terminal({actor},{marker},{subject},{transition_noun},{state_noun}) equals {answer}."
    return gold, counter, prompt, required, forbidden, review


def replay_events(record: dict[str, Any], initial_value: str | None) -> tuple[str, list[str]]:
    """Replay the declared dependency chain; no context prose is trusted."""
    state: str | None = None
    completed: list[str] = []
    for index, event in enumerate(record["events"]):
        expected_dependencies = [] if index == 0 else [record["events"][index - 1]["event_id"]]
        _require(event["dependencies"] == expected_dependencies, f"non-minimal or broken dependency at {event['event_id']}")
        if index == 0:
            _require(event["predicate"] == "initialize_from_intervention", "first event is not the intervention")
            _require(event["object"] == "WORLD_INTERVENTION_VALUE", "first event has invalid operand")
            _require(initial_value is not None, "proof replay lacks decisive initial fact")
            state = initial_value
        else:
            _require(event["predicate"] == FAMILY_OPERATIONS[record["state_family"]], "family operation mismatch")
            try:
                mapping = json.loads(event["object"])
            except (json.JSONDecodeError, TypeError) as error:
                raise AuthoringValidationError(f"invalid operation mapping at {event['event_id']}") from error
            _require(isinstance(mapping, dict) and len(mapping) == 2, "operation mapping must contain both worlds")
            _require(state in mapping, f"operation cannot consume replay state at {event['event_id']}")
            state = mapping[state]
        completed.append(event["event_id"])
    _require(state is not None, "proof replay produced no state")
    assert state is not None
    return state, completed


def build_scenario(
    allocation: dict[str, Any], config: dict[str, Any], tokenizer: Tokenizer
) -> dict[str, Any]:
    ordinal = int(allocation["scenario_id"].rsplit("-", 1)[1])
    split = allocation["split"]
    split_config = config["splits"][split]
    family = config["families"][allocation["state_family"]]
    renderer_name = config["renderers"][split]
    renderer = RENDERERS[renderer_name]
    names = split_config["names"]
    actor = _choice(names, ordinal)
    witness = _choice(names, ordinal, 3)
    verb = _choice(split_config["verbs"], ordinal)
    style = _choice(split_config["paraphrases"], ordinal // 4)
    depth = allocation["reasoning_depth"]
    track_a, track_b, neutral = _world_values(split_config["lexicon"], ordinal, depth)

    events = [{
        "event_id": "e0", "sequence": 0, "subject": actor,
        "predicate": "initialize_from_intervention", "object": "WORLD_INTERVENTION_VALUE",
        "dependencies": [], "text": f"Initialize the {family['state_noun']} from the world's decisive premise.",
    }]
    mappings: list[dict[str, str]] = []
    for step in range(1, depth):
        mapping = {track_a[step - 1]: track_a[step], track_b[step - 1]: track_b[step]}
        mappings.append(mapping)
        events.append({
            "event_id": f"e{step}", "sequence": step, "subject": family["subject"],
            "predicate": FAMILY_OPERATIONS[allocation["state_family"]],
            "object": canonical_json(mapping), "dependencies": [f"e{step - 1}"],
            "text": f"Apply {family['transition_noun']} {step} to the preceding state.",
        })
    rule_texts = [_mapping_text(mapping, family, index + 1) for index, mapping in enumerate(mappings)]
    render_args = {
        "style": style, "actor": actor, "verb": verb, "witness": witness, "family": family,
        "rule_texts": rule_texts,
    }
    context_a = renderer(**render_args, initial=track_a[0])
    context_b = renderer(**render_args, initial=track_b[0])
    fact_removed = renderer(**render_args, initial="", fact_present=False)
    irrelevant = renderer(**render_args, initial="", fact_present=False, irrelevant=neutral)
    answer_a, answer_b = track_a[-1], track_b[-1]
    a_label = allocation["worlds"]["A"]["correct_candidate"]
    candidate_values = {a_label: answer_a, "Y" if a_label == "X" else "X": answer_b}
    candidates: dict[str, dict[str, Any]] = {}
    for label, value in candidate_values.items():
        text = _candidate_text(split, witness, family["state_noun"], value)
        candidates[label] = {"text": text, "token_ids": tokenizer.encode(text).ids}

    def world(label: str, context: str, initial: str, answer: str, other: str) -> dict[str, Any]:
        correct = allocation["worlds"][label]["correct_candidate"]
        proof_steps = [f"e0 initializes {initial}."]
        track = track_a if label == "A" else track_b
        proof_marker = " ".join([*track, neutral])
        proof_steps.extend(
            f"e{index} applies {FAMILY_OPERATIONS[allocation['state_family']]} and yields {track[index]}."
            for index in range(1, depth)
        )
        gold, counter, prompt, proposition, forbidden, review = _world_surface_text(
            split, candidate=candidates[correct]["text"], style=style, actor=actor,
            subject=family["subject"], state_noun=family["state_noun"], transition_noun=family["transition_noun"], answer=answer,
            other=other, context=context, marker=proof_marker,
        )
        _, counter_text, _, _, _, _ = _world_surface_text(
            split, candidate=candidates["Y" if correct == "X" else "X"]["text"], style=style,
            actor=actor, subject=family["subject"], state_noun=family["state_noun"], transition_noun=family["transition_noun"],
            answer=answer, other=other, context=context, marker=proof_marker,
        )
        return {
            "context": context,
            "correct_candidate": correct,
            "proof": {
                "supporting_event_ids": [event["event_id"] for event in events],
                "inference_chain": proof_steps,
                "answer_value": answer,
            },
            "gold_continuation": gold,
            "counterfactual_continuation": counter_text,
            "free_generation": {
                "prompt": prompt,
                "required_propositions": [proposition],
                "forbidden_propositions": [forbidden],
                "review_question": review,
            },
        }

    record: dict[str, Any] = {key: copy.deepcopy(allocation[key]) for key in JOIN_FIELDS}
    record.update({
        "entities": [
            {"entity_id": "actor", "name": actor, "source_pool_id": split_config["source_pool"], "paraphrase_id": style, "renderer": renderer_name},
            {"entity_id": "witness", "name": witness, "source_pool_id": split_config["source_pool"], "paraphrase_id": style, "renderer": renderer_name},
        ],
        "events": events,
        "intervention": {
            "changed_event_id": "e0", "world_a_value": track_a[0], "world_b_value": track_b[0],
            "unchanged_event_ids": [event["event_id"] for event in events[1:]],
        },
        "candidates": candidates,
        "worlds": {
            "A": world("A", context_a, track_a[0], answer_a, answer_b),
            "B": world("B", context_b, track_b[0], answer_b, answer_a),
        },
        "controls": {"fact_removed_context": fact_removed, "irrelevant_substituted_context": irrelevant},
        "metadata": {
            "genre": family["genre"], "source": "benchmark-v2-authored",
            "authoring_revision": config["authoring_revision"],
        },
    })
    record["stable_hash"] = authored_stable_hash(record)
    return record


def build_scenarios(
    allocations: list[dict[str, Any]], config: dict[str, Any], tokenizer: Tokenizer
) -> list[dict[str, Any]]:
    return [build_scenario(allocation, config, tokenizer) for allocation in allocations]


def validate_schema(instance: Any, schema: dict[str, Any], root: dict[str, Any], path: str = "$") -> None:
    """Validate every JSON-Schema feature used by the frozen scenario schema."""
    if "$ref" in schema:
        target: Any = root
        for component in schema["$ref"].removeprefix("#/").split("/"):
            target = target[component]
        validate_schema(instance, target, root, path)
        return
    if "oneOf" in schema:
        successes = 0
        for option in schema["oneOf"]:
            try:
                validate_schema(instance, option, root, path)
                successes += 1
            except AuthoringValidationError:
                pass
        _require(successes == 1, f"{path}: expected exactly one schema alternative")
    if "const" in schema:
        _require(instance == schema["const"], f"{path}: const mismatch")
    if "enum" in schema:
        _require(instance in schema["enum"], f"{path}: value outside enum")
    expected_type = schema.get("type")
    type_checks = {
        "object": lambda value: isinstance(value, dict),
        "array": lambda value: isinstance(value, list),
        "string": lambda value: isinstance(value, str),
        "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
        "boolean": lambda value: isinstance(value, bool),
    }
    if expected_type:
        _require(type_checks[expected_type](instance), f"{path}: expected {expected_type}")
    if isinstance(instance, str):
        _require(len(instance) >= schema.get("minLength", 0), f"{path}: string too short")
        if "pattern" in schema:
            _require(re.search(schema["pattern"], instance) is not None, f"{path}: pattern mismatch")
    if isinstance(instance, int) and not isinstance(instance, bool) and "minimum" in schema:
        _require(instance >= schema["minimum"], f"{path}: below minimum")
    if isinstance(instance, dict):
        required = set(schema.get("required", []))
        _require(required <= set(instance), f"{path}: missing properties {sorted(required - set(instance))}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            _require(set(instance) <= set(properties), f"{path}: unexpected properties {sorted(set(instance) - set(properties))}")
        for key, value in instance.items():
            if key in properties:
                validate_schema(value, properties[key], root, f"{path}.{key}")
    if isinstance(instance, list):
        _require(len(instance) >= schema.get("minItems", 0), f"{path}: too few items")
        prefix = schema.get("prefixItems", [])
        for index, item_schema in enumerate(prefix):
            _require(index < len(instance), f"{path}: missing prefix item")
            validate_schema(instance[index], item_schema, root, f"{path}[{index}]")
        items = schema.get("items")
        if items is False:
            _require(len(instance) == len(prefix), f"{path}: extra array items")
        elif isinstance(items, dict):
            for index, item in enumerate(instance[len(prefix):], start=len(prefix)):
                validate_schema(item, items, root, f"{path}[{index}]")


def template_signature(text: str, config: dict[str, Any]) -> str:
    replacements: list[tuple[str, str]] = []
    for values in config["splits"].values():
        for field, marker in (("names", "NAME"), ("verbs", "VERB"), ("lexicon", "VALUE"), ("paraphrases", "STYLE")):
            replacements.extend((value.casefold(), marker) for value in values[field])
    for family in config["families"].values():
        replacements.extend((value.casefold(), "SEMANTIC") for value in family.values())
    normalized = " ".join(normalized_words(text))
    for value, marker in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        normalized = re.sub(rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])", marker, normalized)
    normalized = re.sub(r"\b[0-9]+\b", "NUMBER", normalized)
    return normalized


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def _surface_rows(record: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for world_name, world in record["worlds"].items():
        rows.extend((
            (f"context:{world_name}", world["context"]),
            (f"gold_continuation:{world_name}", world["gold_continuation"]),
            (f"counterfactual_continuation:{world_name}", world["counterfactual_continuation"]),
            (f"free_generation_prompt:{world_name}", world["free_generation"]["prompt"]),
            (f"required_proposition:{world_name}", " ".join(world["free_generation"]["required_propositions"])),
            (f"forbidden_proposition:{world_name}", " ".join(world["free_generation"]["forbidden_propositions"])),
            (f"review_question:{world_name}", world["free_generation"]["review_question"]),
        ))
    rows.extend((
        ("fact_removed_context", record["controls"]["fact_removed_context"]),
        ("irrelevant_substituted_context", record["controls"]["irrelevant_substituted_context"]),
        ("candidate:X", record["candidates"]["X"]["text"]),
        ("candidate:Y", record["candidates"]["Y"]["text"]),
    ))
    return rows


def audit_template_signatures(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    signatures: defaultdict[str, set[str]] = defaultdict(set)
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        split = record["split"]
        for kind, text in _surface_rows(record):
            signature = template_signature(text, config)
            signatures[split].add(signature)
            counts[split][kind.split(":", 1)[0]] += 1
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            overlap = signatures[left] & signatures[right]
            _require(not overlap, f"normalized authored-surface template signature crosses {left}/{right}")
    return {
        "status": "PASS",
        "unique_signature_counts": {split: len(signatures[split]) for split in SPLITS},
        "surface_counts": {split: dict(sorted(counts[split].items())) for split in SPLITS},
        "surfaces": sorted({kind for split_counts in counts.values() for kind in split_counts}),
    }


def _audit_duplicates(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    raw_rows = [
        (record["scenario_id"], record["split"], kind, text)
        for record in records for kind, text in _surface_rows(record)
    ]
    rows = [
        (scenario_id, split, kind, normalized_text_hash(text), text_shingles(text))
        for scenario_id, split, kind, text in raw_rows
    ]
    exact_seen: dict[str, tuple[str, str]] = {}
    exact_duplicates: list[dict[str, str]] = []
    near_duplicates: list[dict[str, Any]] = []
    for index, (scenario_id, split, kind, digest, shingles) in enumerate(rows):
        if digest in exact_seen and exact_seen[digest][0] != scenario_id:
            exact_duplicates.append({"left": exact_seen[digest][0], "right": scenario_id, "surface": kind})
        exact_seen[digest] = (scenario_id, kind)
        for other_id, other_split, other_kind, _, other_shingles in rows[:index]:
            if other_id == scenario_id:
                continue
            similarity = _jaccard(shingles, other_shingles)
            if similarity > threshold:
                near_duplicates.append({
                    "left": other_id, "right": scenario_id, "left_split": other_split,
                    "right_split": split, "left_surface": other_kind, "right_surface": kind,
                    "similarity": round(similarity, 6),
                })
    if exact_duplicates:
        raise AuthoringValidationError(f"exact duplicate authored surfaces: {exact_duplicates[0]}")
    if near_duplicates:
        raise AuthoringValidationError(f"near-duplicate authored surfaces: {near_duplicates[0]}")
    return {
        "surface_count": len(rows), "normalized_exact_duplicates": 0,
        "near_duplicates_above_threshold": 0, "threshold": threshold,
        "same_scenario_cross_surface_exemptions": "paired worlds and prompt/context coupling only",
    }


def _tracked_source_audit(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    paths = [Path(path) for path in config["training_contamination"]["source_provenance"]]
    checked: list[dict[str, str]] = []
    source_words: set[str] = set()
    normalized_sources: list[str] = []
    source_shingles: set[str] = set()
    source_signatures: set[str] = set()
    for path in paths:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)], capture_output=True, text=True, check=False
        )
        _require(result.returncode == 0, f"contamination source is not tracked: {path}")
        text = path.read_text(encoding="utf-8")
        words = normalized_words(text)
        source_words.update(words)
        normalized_sources.append(" " + " ".join(words) + " ")
        source_shingles.update(text_shingles(text, width=8))
        for line in text.splitlines():
            if len(normalized_words(line)) >= 5:
                source_signatures.add(template_signature(line, config))
        checked.append({"path": str(path), "sha256": sha256_file(path)})

    authored_names = {entity["name"].casefold() for record in records for entity in record["entities"]}
    authored_verbs = {record["events"][0]["predicate"].casefold() for record in records}
    # Predicate is structural; configured surface verbs are the contamination-sensitive words.
    authored_verbs.update(value.casefold() for split in config["splits"].values() for value in split["verbs"])
    phrase_hits: set[str] = set()
    signature_hits: set[str] = set()
    for record in records:
        for kind, text in _surface_rows(record):
            phrase_hits.update(text_shingles(text, width=8) & source_shingles)
            signature = template_signature(text, config)
            if signature in source_signatures:
                signature_hits.add(f"{record['scenario_id']}:{kind}")
    name_hits = sorted(authored_names & source_words)
    verb_hits = sorted(
        verb for verb in authored_verbs
        if any(f" {' '.join(normalized_words(verb))} " in source for source in normalized_sources)
    )
    _require(not name_hits, f"training contamination in authored names: {name_hits}")
    _require(not verb_hits, f"training contamination in authored verbs: {verb_hits}")
    _require(not phrase_hits, f"training contamination in authored phrases: {sorted(phrase_hits)[:1]}")
    _require(not signature_hits, f"training contamination in template signatures: {sorted(signature_hits)[:1]}")
    return {
        "status": "PASS", "checked_sources": checked, "name_hits": [], "verb_hits": [],
        "normalized_eight_word_phrase_hits": [], "template_signature_hits": [],
    }


def _iter_text_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            if key.endswith(("token_ids", "stable_hash", "sha256")):
                continue
            if isinstance(child, (str, dict, list)):
                yield from _iter_text_values(child)
    elif isinstance(value, list):
        if value and isinstance(value[0], (int, float, bool)):
            return
        for child in value:
            if isinstance(child, (str, dict, list)):
                yield from _iter_text_values(child)


def audit_training_artifacts(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Stream ignored training JSONL and compare its actual rendered strings."""
    cache_key = sha256_bytes(canonical_json({
        "artifacts": config["training_contamination"]["training_artifacts"],
        "record_hashes": [record["stable_hash"] for record in records],
    }).encode("utf-8"))
    if cache_key in _TRAINING_AUDIT_CACHE:
        return copy.deepcopy(_TRAINING_AUDIT_CACHE[cache_key])
    names = {entity["name"].casefold() for record in records for entity in record["entities"]}
    verbs = {value.casefold() for split in config["splits"].values() for value in split["verbs"]}
    authored_phrases: set[str] = set()
    authored_signature_shingles: set[str] = set()
    for record in records:
        for _, text in _surface_rows(record):
            authored_phrases.update(text_shingles(text, width=8))
            authored_signature_shingles.update(text_shingles(template_signature(text, config), width=8))

    name_hits: set[str] = set()
    verb_hits: set[str] = set()
    phrase_hits: set[str] = set()
    signature_hits: set[str] = set()
    checked: list[dict[str, Any]] = []
    for specification in config["training_contamination"]["training_artifacts"]:
        path = Path(specification["path"])
        _require(path.is_file(), f"required training contamination artifact is absent: {path}")
        actual_hash = sha256_file(path)
        _require(actual_hash == specification["sha256"], f"training contamination artifact hash mismatch: {path}")
        record_count = 0
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                record_count += 1
                try:
                    artifact_record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise AuthoringValidationError(f"invalid training JSONL at {path}:{line_number}") from error
                for text in _iter_text_values(artifact_record):
                    words = normalized_words(text)
                    if not words:
                        continue
                    padded = " " + " ".join(words) + " "
                    name_hits.update(name for name in names if f" {name} " in padded)
                    verb_hits.update(
                        verb for verb in verbs
                        if f" {' '.join(normalized_words(verb))} " in padded
                    )
                    if len(words) >= 8:
                        phrase_hits.update(text_shingles(text, width=8) & authored_phrases)
                        signature_hits.update(
                            text_shingles(template_signature(text, config), width=8)
                            & authored_signature_shingles
                        )
        checked.append({"path": str(path), "sha256": actual_hash, "records": record_count})
    _require(not name_hits, f"rendered training artifact name contamination: {sorted(name_hits)}")
    _require(not verb_hits, f"rendered training artifact verb contamination: {sorted(verb_hits)}")
    _require(not phrase_hits, f"rendered training artifact phrase contamination: {sorted(phrase_hits)[:1]}")
    _require(not signature_hits, f"rendered training artifact template contamination: {sorted(signature_hits)[:1]}")
    report = {
        "status": "PASS", "checked_artifacts": checked,
        "records_checked": sum(item["records"] for item in checked),
        "name_hits": [], "verb_hits": [], "normalized_eight_word_phrase_hits": [],
        "normalized_template_shingle_hits": [],
    }
    _TRAINING_AUDIT_CACHE[cache_key] = copy.deepcopy(report)
    return report


def _candidate_control(record: dict[str, Any], tokenizer: Tokenizer) -> dict[str, Any]:
    answer_values = {world["proof"]["answer_value"] for world in record["worlds"].values()}
    frames = []
    for label in ("X", "Y"):
        candidate = record["candidates"][label]
        encoded = tokenizer.encode(candidate["text"]).ids
        _require(candidate["token_ids"] == encoded and bool(encoded), f"candidate token boundary mismatch: {record['scenario_id']}:{label}")
        _require(tokenizer.decode(encoded) == candidate["text"], f"candidate token round trip mismatch: {record['scenario_id']}:{label}")
        value = next((value for value in answer_values if value in normalized_words(candidate["text"])), None)
        _require(value is not None, f"candidate lacks exactly one answer value: {record['scenario_id']}:{label}")
        _require(sum(candidate["text"].count(item) for item in answer_values) == 1, "candidate contains multiple answer values")
        frames.append(candidate["text"].replace(value, "<ANSWER>"))
    _require(frames[0] == frames[1], f"candidate frame cancellation failed: {record['scenario_id']}")
    return {
        "scenario_id": record["scenario_id"], "split": record["split"],
        "candidate_presentation_order": record["candidate_presentation_order"],
        "world_labels": {name: world["correct_candidate"] for name, world in record["worlds"].items()},
        "candidates": copy.deepcopy(record["candidates"]),
        "X": {"token_count": len(record["candidates"]["X"]["token_ids"]), "character_count": len(record["candidates"]["X"]["text"])},
        "Y": {"token_count": len(record["candidates"]["Y"]["token_ids"]), "character_count": len(record["candidates"]["Y"]["text"])},
        "shared_frame": frames[0], "frame_cancellation": True,
    }


def audit_scenarios(
    records: list[dict[str, Any]], allocations: list[dict[str, Any]], config: dict[str, Any],
    schema: dict[str, Any], tokenizer: Tokenizer,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require(len(records) == len(allocations) == 112, "authored/allocation count mismatch")
    allocation_by_id = {record["scenario_id"]: record for record in allocations}
    ids: set[str] = set()
    hashes: set[str] = set()
    split_counts: Counter[str] = Counter()
    family_depth: Counter[str] = Counter()
    depth_counts: Counter[int] = Counter()
    world_labels: Counter[str] = Counter()
    orders: Counter[str] = Counter()
    paraphrases: Counter[str] = Counter()
    candidate_controls: list[dict[str, Any]] = []

    isolation_fields = {
        "source_pool": lambda values: [values["source_pool"]], "names": lambda values: values["names"],
        "verbs": lambda values: values["verbs"], "lexicon": lambda values: values["lexicon"],
        "paraphrases": lambda values: values["paraphrases"],
    }
    for field, extract in isolation_fields.items():
        owners: defaultdict[str, set[str]] = defaultdict(set)
        for split, values in config["splits"].items():
            for value in extract(values):
                owners[value.casefold()].add(split)
        _require(not any(len(splits) > 1 for splits in owners.values()), f"configured {field} pools cross splits")

    for record in records:
        validate_schema(record, schema, schema)
        scenario_id = record["scenario_id"]
        _require(scenario_id not in ids, f"duplicate scenario ID: {scenario_id}")
        ids.add(scenario_id)
        _require(record["stable_hash"] == authored_stable_hash(record), f"stable hash mismatch: {scenario_id}")
        _require(record["stable_hash"] not in hashes, f"duplicate stable hash: {scenario_id}")
        hashes.add(record["stable_hash"])
        allocation = allocation_by_id.get(scenario_id)
        _require(allocation is not None, f"scenario absent from allocation: {scenario_id}")
        assert allocation is not None
        for field in JOIN_FIELDS:
            _require(record[field] == allocation[field], f"allocation join mismatch for {scenario_id}: {field}")

        event_ids = [event["event_id"] for event in record["events"]]
        _require(len(event_ids) == record["reasoning_depth"], f"event depth mismatch: {scenario_id}")
        _require(len(event_ids) == len(set(event_ids)), f"duplicate event ID: {scenario_id}")
        _require(event_ids == [f"e{index}" for index in range(len(event_ids))], f"event sequence mismatch: {scenario_id}")
        _require(record["intervention"]["changed_event_id"] == "e0", "changed event is not proof root")
        _require(record["intervention"]["unchanged_event_ids"] == event_ids[1:], "unchanged event list mismatch")
        for world_name, world in record["worlds"].items():
            initial = record["intervention"][f"world_{world_name.lower()}_value"]
            answer, replayed = replay_events(record, initial)
            _require(answer == world["proof"]["answer_value"], f"proof answer does not replay: {scenario_id}:{world_name}")
            _require(replayed == world["proof"]["supporting_event_ids"] == event_ids, f"proof references do not replay: {scenario_id}:{world_name}")
            _require(len(world["proof"]["inference_chain"]) == len(event_ids), f"inference depth mismatch: {scenario_id}")
            _require(initial in normalized_words(world["context"]), f"world context lacks intervention: {scenario_id}:{world_name}")
            correct = world["correct_candidate"]
            other = "Y" if correct == "X" else "X"
            _require(world["gold_continuation"].startswith(record["candidates"][correct]["text"]), f"gold continuation/candidate mismatch: {scenario_id}:{world_name}")
            _require(world["counterfactual_continuation"].startswith(record["candidates"][other]["text"]), f"counterfactual continuation/candidate mismatch: {scenario_id}:{world_name}")
            if record["reasoning_depth"] > 1:
                _require(initial != answer, f"depth>1 premise directly states answer: {scenario_id}:{world_name}")
        try:
            replay_events(record, None)
        except AuthoringValidationError:
            pass
        else:
            raise AuthoringValidationError(f"fact-removed proof unexpectedly replays: {scenario_id}")
        split = record["split"]
        own = config["splits"][split]
        ordinal = int(scenario_id.rsplit("-", 1)[1])
        family = config["families"][record["state_family"]]
        renderer = RENDERERS[config["renderers"][split]]
        actor, witness = (entity["name"] for entity in record["entities"])
        style = next(iter({entity["paraphrase_id"] for entity in record["entities"]}))
        verb = _choice(own["verbs"], ordinal)
        rule_texts = [
            _mapping_text(json.loads(event["object"]), family, index)
            for index, event in enumerate(record["events"][1:], start=1)
        ]
        _, _, neutral = _world_values(own["lexicon"], ordinal, record["reasoning_depth"])
        render_args = {
            "style": style, "actor": actor, "verb": verb, "witness": witness,
            "family": family, "rule_texts": rule_texts, "initial": "", "fact_present": False,
        }
        _require(record["controls"]["fact_removed_context"] == renderer(**render_args), f"fact-removed control render mismatch: {scenario_id}")
        _require(record["controls"]["irrelevant_substituted_context"] == renderer(**render_args, irrelevant=neutral), f"irrelevant control render mismatch: {scenario_id}")

        full_text = canonical_json(record).casefold()
        _require(all(entity["source_pool_id"] == own["source_pool"] for entity in record["entities"]), "source pool mismatch")
        _require(all(entity["renderer"] == config["renderers"][split] for entity in record["entities"]), "renderer mismatch")
        styles = {entity["paraphrase_id"] for entity in record["entities"]}
        _require(len(styles) == 1 and styles <= set(own["paraphrases"]), "paraphrase mismatch")
        paraphrases.update(styles)
        for other_split, other in config["splits"].items():
            if other_split == split:
                continue
            forbidden = [other["source_pool"], *other["names"], *other["verbs"], *other["lexicon"], *other["paraphrases"]]
            _require(not any(re.search(rf"(?<![a-z]){re.escape(term.casefold())}(?![a-z])", full_text) for term in forbidden), f"cross-split surface leakage: {scenario_id}")
        candidate_controls.append(_candidate_control(record, tokenizer))
        split_counts[split] += 1
        family_depth[f"{record['state_family']}/{record['depth_bucket']}"] += 1
        depth_counts[record["reasoning_depth"]] += 1
        world_labels.update(world["correct_candidate"] for world in record["worlds"].values())
        orders["".join(record["candidate_presentation_order"])] += 1

    _require(ids == set(allocation_by_id), "allocation has unauthored scenarios")
    _require(world_labels == {"X": 112, "Y": 112}, "candidate labels are not balanced")
    _require(orders == {"XY": 56, "YX": 56}, "candidate presentation is not balanced")
    _require(all(count == 4 for count in family_depth.values()) and len(family_depth) == 28, "family/depth cells incomplete")
    template_report = audit_template_signatures(records, config)
    for field in ("template_id", "template_family_id", "lexical_pool_id", "name_pool_id", "verb_pool_id"):
        owners: defaultdict[str, set[str]] = defaultdict(set)
        for record in records:
            owners[record[field]].add(record["split"])
        _require(not any(len(splits) > 1 for splits in owners.values()), f"{field} crosses splits")

    duplicate_report = _audit_duplicates(records, float(config["near_duplicate_threshold"]))
    source_provenance = _tracked_source_audit(records, config)
    training_artifacts = audit_training_artifacts(records, config)
    token_lengths = [abs(control["X"]["token_count"] - control["Y"]["token_count"]) for control in candidate_controls]
    report = {
        "status": "PASS", "release_ready": True, "git_commit": "PENDING_UNTIL_COMMIT",
        "schema_validation": {"status": "PASS", "records": len(records), "implementation": "direct-frozen-schema-recursion"},
        "allocation_join": {"status": "PASS", "records": len(records)},
        "scenario_count": len(records), "split_counts": dict(sorted(split_counts.items())),
        "counts_by_family_depth": dict(sorted(family_depth.items())),
        "reasoning_depth_counts": dict(sorted(depth_counts.items())),
        "proof_replay": {"status": "PASS", "intact_worlds_replayed": len(records) * 2, "fact_removed_replay_failures": len(records)},
        "candidate_orders": dict(sorted(orders.items())), "world_candidate_labels": dict(sorted(world_labels.items())),
        "paraphrase_counts": dict(sorted(paraphrases.items())),
        "template_signature_isolation": template_report,
        "isolation": {"source_pool": "PASS", "template_family": "PASS", "names": "PASS", "verbs": "PASS", "lexicon": "PASS"},
        "controls": {
            "status": "PASS", "candidate_only_records": len(candidate_controls), "candidate_frame_cancellation": len(candidate_controls),
            "candidate_token_length_delta_max": max(token_lengths), "candidate_order_balanced": True,
            "fact_removed_contexts": len(records), "fact_removed_chain_replayable": 0,
            "irrelevant_substituted_contexts": len(records), "decisive_world_selection_leaks": 0,
        },
        "token_boundary": {
            "status": "PASS", "tokenizer_path": config["tokenizer"]["path"],
            "tokenizer_sha256": config["tokenizer"]["sha256"], "candidates_checked": len(records) * 2,
        },
        "duplicate_audit": duplicate_report, "cross_split_leakage": 0,
        "training_contamination": {
            "status": "PASS", "actual_training_artifacts": training_artifacts,
            "source_config_provenance": source_provenance,
        },
    }
    return report, candidate_controls


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return b"".join((canonical_json(record) + "\n").encode("utf-8") for record in records)


def _validation_markdown(report: dict[str, Any]) -> str:
    return f"""# Authored Scenario Audit

This file contains deterministic generator and audit facts only. It does not claim that tests, model training, model evaluation, inference, checkpoint selection, or human review ran.

```text
status: {report['status']}
release_ready: {str(report['release_ready']).lower()}
git_commit: {report['git_commit']}
scenario_count: {report['scenario_count']}
split_counts: development={report['split_counts']['development']}, test={report['split_counts']['test']}, generalization={report['split_counts']['generalization']}
family_depth_cells: {len(report['counts_by_family_depth'])} (4 scenarios each)
reasoning_depth_counts: {json.dumps(report['reasoning_depth_counts'], sort_keys=True)}
proof_replay: PASS ({report['proof_replay']['intact_worlds_replayed']} worlds)
template_signature_isolation: PASS
cross_split_leakage: 0
all_surface_exact_duplicates: 0
all_surface_near_duplicates_above_{report['duplicate_audit']['threshold']:.2f}: 0
candidate_only_controls: {report['controls']['candidate_only_records']}
fact_removed_replayable_chains: 0
token_boundary: PASS ({report['token_boundary']['tokenizer_sha256']})
training_contamination: PASS ({report['training_contamination']['actual_training_artifacts']['records_checked']} rendered training records; {len(report['training_contamination']['source_config_provenance']['checked_sources'])} provenance files)
```

The manifest hashes this computed report and all other generated artifacts except the manifest itself. Test and generalization remain post-selection data; build-time authorship does not authorize evaluation-time access.
"""


def build_outputs(config_path: Path) -> tuple[Path, dict[str, bytes]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(config["benchmark_root"])
    validate_benchmark(root)
    for name, expected in config["locked_inputs"].items():
        _require(sha256_file(root / name) == expected, f"locked input hash mismatch: {name}")
    tokenizer_path = Path(config["tokenizer"]["path"])
    tracked_tokenizer = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(tokenizer_path)], capture_output=True, text=True, check=False
    )
    _require(tracked_tokenizer.returncode == 0, f"tokenizer is not tracked: {tokenizer_path}")
    _require(sha256_file(tokenizer_path) == config["tokenizer"]["sha256"], "tokenizer hash mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    allocations = load_jsonl(Path(config["allocation_path"]))
    records = build_scenarios(allocations, config, tokenizer)
    schema = json.loads((root / "scenario.schema.json").read_text(encoding="utf-8"))
    report, candidate_controls = audit_scenarios(records, allocations, config, schema, tokenizer)
    outputs = {
        SCENARIO_FILES[split]: _jsonl_bytes(record for record in records if record["split"] == split)
        for split in SPLITS
    }
    for split in SPLITS:
        outputs[CONTROL_FILES[split]] = _jsonl_bytes(
            control for control in candidate_controls if control["split"] == split
        )
    outputs["authoring-report.json"] = _json_bytes(report)
    outputs["authoring-validation-results.md"] = _validation_markdown(report).encode("utf-8")
    manifest = {
        "benchmark_id": "fictionpulper-narrative-benchmark-v2", "authoring_revision": config["authoring_revision"],
        "scenario_count": len(records), "git_commit": "PENDING_UNTIL_COMMIT",
        "config": {"path": config["manifest_config_path"], "sha256": sha256_file(config_path)},
        "generator": {"path": "src/narrative_v2_authoring.py", "sha256": sha256_file(Path(__file__))},
        "locked_inputs": config["locked_inputs"], "tokenizer": config["tokenizer"],
        "generated_artifacts": {name: {"sha256": sha256_bytes(content)} for name, content in sorted(outputs.items())},
        "access_policy": {
            "development": "default readable split; scorer development only; never weight training",
            "test": "requires checkpoint_selection_complete attestation",
            "generalization": "requires checkpoint_selection_complete and test_evaluation_complete attestations",
        },
        "self_hash": "OMITTED_TO_AVOID_CIRCULARITY",
    }
    outputs["authored-manifest.json"] = _json_bytes(manifest)
    return root, outputs


def generate_or_verify(config_path: Path) -> str:
    root, outputs = build_outputs(config_path)
    existing = [name for name in OUTPUT_FILES if (root / name).exists()]
    if existing:
        _require(len(existing) == len(OUTPUT_FILES), f"partial output set; refusing overwrite: {existing}")
        mismatches = [name for name, content in outputs.items() if (root / name).read_bytes() != content]
        _require(not mismatches, f"existing outputs differ; refusing overwrite: {mismatches}")
        return "verified"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        staging = Path(temporary)
        for name, content in outputs.items():
            (staging / name).write_bytes(content)
        for name in OUTPUT_FILES:
            (staging / name).replace(root / name)
    return "generated"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/narrative-v2-authoring.yaml"))
    args = parser.parse_args()
    status = generate_or_verify(args.config)
    root = yaml.safe_load(args.config.read_text(encoding="utf-8"))["benchmark_root"]
    print(json.dumps({"passed": True, "status": status, "root": root}, sort_keys=True))


if __name__ == "__main__":
    main()
