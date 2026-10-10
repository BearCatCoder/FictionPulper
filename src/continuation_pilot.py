"""Prepare, audit, review, and finalize the rights-cleared continuation pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml
from tokenizers import Tokenizer

from src.continuity_curriculum.common import (
    canonical_json,
    load_jsonl,
    normalized_text_hash,
    normalized_words,
    sha256_bytes,
    sha256_file,
    text_shingles,
    write_json_atomic,
    write_jsonl,
)
from src.narrative_v2_authoring import validate_schema


VERSION = 1
SPLITS = ("train", "validation", "test")
GENRES = ("crime_noir", "horror_occult", "science_fiction", "western_adventure", "weird_fantasy")
STATE_FAMILIES = ("possession", "location", "knowledge", "goal", "causal_update")
REVIEW_FIELDS = (
    "premise_preserved", "motive_preserved", "opening_facts_consistent",
    "no_narrative_loop", "plot_advanced", "genre_voice", "meaningful_action",
    "meaningful_consequence", "natural_readable_continuation",
)
REVIEW_LABELS = {"yes", "no", "uncertain"}
FACT_LABELS = {"retained", "contradicted", "uncertain"}
APPROVAL_FILE = "approval.json"
DEFAULT_OUTPUT_DIR = Path("runs/continuation-pilot-v1")
PRODUCTION_SCHEMA_SHA256 = "2de976d756ad247f1907b5a407496231d85b38b350ad61159d34bbc5bfa7373d"
PRODUCTION_CONTRACT: dict[str, Any] = {
    "version": 1,
    "seed": 8008,
    "review_seed": 8018,
    "schema_path": "schemas/continuation-pilot-v1.schema.json",
    "tokenizer": {
        "path": "data/tokenizer/tokenizer.json",
        "sha256": "14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012",
    },
    "expected_examples": 200,
    "expected_split_counts": {"train": 160, "validation": 20, "test": 20},
    "expected_genre_counts": {genre: 40 for genre in GENRES},
    "expected_state_family_counts": {family: 40 for family in STATE_FAMILIES},
    "expected_stale_state_distractor_counts": {"present": 100, "absent": 100},
    "expected_genre_state_cell_count": 8,
    "expected_split_genre_counts": {
        "train": {genre: 32 for genre in GENRES},
        "validation": {genre: 4 for genre in GENRES},
        "test": {genre: 4 for genre in GENRES},
    },
    "word_counts": {
        "opening_minimum": 60, "opening_maximum": 120,
        "continuation_minimum": 150, "continuation_maximum": 250,
    },
    "serialization": {
        "format": "<|story|><|bos|>{opening}{continuation}<|eos|>",
        "boundary": "", "maximum_tokens": 1024,
    },
    "contamination": {
        "benchmark_v2": [
            {"path": "benchmarks/narrative-v2/development.jsonl", "sha256": "c38842d3bf98c52ebe8d46df30b89122177a099d91a3166fa0b881fb1707f4e4"},
            {"path": "benchmarks/narrative-v2/test.jsonl", "sha256": "b555c5dc145b4c83e36ce5d354767c2c73e9290187a166fa339de9366869f10c"},
            {"path": "benchmarks/narrative-v2/generalization.jsonl", "sha256": "9d40c81499cced1b786866846493258e331ad870be9f5fb6ba6133643ff3672d"},
            {"path": "benchmarks/narrative-v2/development-candidate-controls.jsonl", "sha256": "306235b9963667c5b80de42433540eeff6d2791aaf7280c8a6014d32258e863b"},
            {"path": "benchmarks/narrative-v2/test-candidate-controls.jsonl", "sha256": "f2f96923b6f90f952392e86a7269f31547da6209f2213b270a365ca1df06fb24"},
            {"path": "benchmarks/narrative-v2/generalization-candidate-controls.jsonl", "sha256": "9b76ecb9fe3febcebbec138074d0ea70b04f36f3f995a7a1bccd767c347a0606"},
        ],
        "scene_scorecard": {
            "path": "benchmarks/scene-scorecard-v1/prompts.jsonl",
            "sha256": "fe828244fec8fda482452792451ecb86d34518ee5c5695646f76b3a43f04ff16",
        },
    },
}
RELOCATABLE_CONFIG_FIELDS = {
    "examples_path", "reviews_path", "custodian_verification_path", "output_dir",
}


class PilotValidationError(ValueError):
    """Raised when pilot inputs do not satisfy the frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotValidationError(message)


def _load_config(path: Path, *, enforce_production: bool = True) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(config, dict) and config.get("version") == VERSION, "pilot config must declare version: 1")
    if enforce_production:
        require(set(config) == set(PRODUCTION_CONTRACT) | RELOCATABLE_CONFIG_FIELDS, "production config fields differ from frozen contract")
        for field, expected in PRODUCTION_CONTRACT.items():
            require(config.get(field) == expected, f"production contract changed: {field}")
        schema_path = Path(config["schema_path"])
        _verify(schema_path, PRODUCTION_SCHEMA_SHA256, "production continuation-pilot schema")
    return config


def _raw_output_dir(config_path: Path) -> Path:
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, UnicodeError):
        return DEFAULT_OUTPUT_DIR
    if isinstance(value, dict) and isinstance(value.get("output_dir"), str) and value["output_dir"]:
        return Path(value["output_dir"])
    return DEFAULT_OUTPUT_DIR


def _remove_approval(output_dir: Path) -> None:
    approval = output_dir / APPROVAL_FILE
    if approval.exists():
        approval.unlink()


def _verify(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"missing {label}: {path}")
    require(sha256_file(path) == expected, f"{label} hash mismatch")


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            if not key.endswith(("sha256", "token_ids")):
                yield from _iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_strings(child)


def _word_count(text: str) -> int:
    return len(normalized_words(text))


def content_sha256(record: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(record).encode("utf-8"))


def validate_example(record: dict[str, Any], schema: dict[str, Any], config: dict[str, Any]) -> None:
    try:
        validate_schema(record, schema, schema)
    except ValueError as error:
        raise PilotValidationError(str(error)) from error
    require(record["schema_version"] == VERSION, f"{record.get('example_id')}: schema version changed")
    require(record["opening"].endswith("\n\n") and not record["opening"].endswith("\n\n\n"), f"{record['example_id']}: opening must end in exactly two LF bytes")
    require(bool(record["continuation"]) and not record["continuation"][0].isspace(), f"{record['example_id']}: continuation must not start with whitespace")
    opening_words = _word_count(record["opening"])
    continuation_words = _word_count(record["continuation"])
    limits = config["word_counts"]
    require(limits["opening_minimum"] <= opening_words <= limits["opening_maximum"], f"{record['example_id']}: opening word count out of range")
    require(limits["continuation_minimum"] <= continuation_words <= limits["continuation_maximum"], f"{record['example_id']}: continuation word count out of range")
    require(3 <= len(record["atomic_facts"]) <= 5, f"{record['example_id']}: exactly 3-5 atomic facts required")
    require(len(set(record["atomic_facts"])) == len(record["atomic_facts"]), f"{record['example_id']}: duplicate atomic facts")
    prose = record["opening"] + record["continuation"]
    transitions = record["event_transitions"]
    require(any(item["state_family"] == record["primary_state_family"] for item in transitions), f"{record['example_id']}: no transition audits the primary state family")
    require(len({item["transition_id"] for item in transitions}) == len(transitions), f"{record['example_id']}: duplicate transition IDs")
    for transition in transitions:
        require(transition["evidence"]["before_quote"] in record["opening"], f"{record['example_id']}: transition before evidence is not in opening")
        require(transition["evidence"]["event_quote"] in record["continuation"], f"{record['example_id']}: transition event evidence is not in continuation")
        require(transition["evidence"]["after_quote"] in record["continuation"], f"{record['example_id']}: transition after evidence is not in continuation")
    metadata_values = [record["genre"], record["primary_state_family"], record["premise"], record["protagonist_motive"], *record["atomic_facts"]]
    require(not any(f"<|{token}|>" in prose for token in ("story", "bos", "eos", "pad")), f"{record['example_id']}: control token in prose")
    require(not any(value in prose for value in metadata_values if value in {record["genre"], record["primary_state_family"]}), f"{record['example_id']}: metadata label visible in prose")
    require(not any(str(value).startswith("scene-post-") for value in record["authoring_inputs"]), f"{record['example_id']}: post-selection prompt used for editing")
    rights = record["rights"]
    provenance = record["provenance"]
    require(re.fullmatch(r"[0-9a-f]{64}", rights["evidence_sha256"]) is not None, f"{record['example_id']}: invalid rights evidence hash")
    require(re.fullmatch(r"[0-9a-f]{64}", provenance["source_sha256"]) is not None, f"{record['example_id']}: invalid source hash")


def _group_components(records: Sequence[dict[str, Any]]) -> list[list[str]]:
    parent = {row["example_id"]: row["example_id"] for row in records}

    def root(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(ids: list[str]) -> None:
        roots = sorted({root(value) for value in ids})
        for value in roots[1:]:
            parent[value] = roots[0]

    for field in ("source_id", "original_story_id"):
        groups: defaultdict[str, list[str]] = defaultdict(list)
        for row in records:
            groups[row["provenance"][field]].append(row["example_id"])
        for ids in groups.values():
            union(ids)
    duplicate_groups: defaultdict[str, list[str]] = defaultdict(list)
    for row in records:
        duplicate_groups[row["duplicate_cluster_id"]].append(row["example_id"])
    for ids in duplicate_groups.values():
        union(ids)
    components: defaultdict[str, list[str]] = defaultdict(list)
    for identifier in parent:
        components[root(identifier)].append(identifier)
    return [sorted(ids) for ids in components.values()]


def _subset(groups: list[list[str]], target: int, seed: int, label: str) -> tuple[list[list[str]], list[list[str]]]:
    ordered = sorted(groups, key=lambda ids: hashlib.sha256(f"{seed}:{label}:{','.join(ids)}".encode()).hexdigest())
    possible: dict[int, tuple[int, ...]] = {0: ()}
    for index, ids in enumerate(ordered):
        for total, selected in sorted(list(possible.items()), reverse=True):
            candidate = total + len(ids)
            if candidate <= target and candidate not in possible:
                possible[candidate] = (*selected, index)
    require(target in possible, f"group constraints cannot produce exact {label} count {target}")
    chosen = set(possible[target])
    return [group for index, group in enumerate(ordered) if index in chosen], [group for index, group in enumerate(ordered) if index not in chosen]


def deterministic_splits(
    records: Sequence[dict[str, Any]], counts: dict[str, int], seed: int,
    split_genre_counts: dict[str, dict[str, int]],
) -> dict[str, str]:
    require(set(counts) == set(SPLITS), "split counts must define train, validation, and test")
    require(sum(counts.values()) == len(records), "split counts do not match example count")
    groups = _group_components(records)
    by_id = {row["example_id"]: row for row in records}
    groups_by_genre: defaultdict[str, list[list[str]]] = defaultdict(list)
    for group in groups:
        genres = {by_id[identifier]["genre"] for identifier in group}
        require(len(genres) == 1, f"mixed-genre transitive group cannot be split: {group}")
        groups_by_genre[next(iter(genres))].append(group)
    require(set(split_genre_counts) == set(SPLITS), "split-by-genre counts must define every split")
    require(all(set(split_genre_counts[split]) == set(GENRES) for split in SPLITS), "split-by-genre counts must define every frozen genre")
    selected: dict[str, list[list[str]]] = {split: [] for split in SPLITS}
    for genre in GENRES:
        genre_groups = groups_by_genre.get(genre, [])
        expected_total = sum(int(split_genre_counts[split][genre]) for split in SPLITS)
        require(sum(len(group) for group in genre_groups) == expected_total, f"{genre}: split-by-genre counts do not match examples")
        test, remaining = _subset(genre_groups, int(split_genre_counts["test"][genre]), seed, f"{genre}:test")
        validation, train = _subset(remaining, int(split_genre_counts["validation"][genre]), seed, f"{genre}:validation")
        selected["train"].extend(train)
        selected["validation"].extend(validation)
        selected["test"].extend(test)
    assignments = {identifier: split for split, values in selected.items() for group in values for identifier in group}
    require(Counter(assignments.values()) == Counter(counts), "exact split counts were not achieved")
    actual_genres = Counter((assignments[row["example_id"]], row["genre"]) for row in records)
    expected_genres = Counter({(split, genre): int(split_genre_counts[split][genre]) for split in SPLITS for genre in GENRES})
    require(actual_genres == expected_genres, "exact split-by-genre counts were not achieved")
    return assignments


def serialize_example(record: dict[str, Any], tokenizer: Tokenizer, *, boundary: str, maximum_tokens: int, pad_to: int | None = None) -> dict[str, Any]:
    story = tokenizer.token_to_id("<|story|>")
    bos = tokenizer.token_to_id("<|bos|>")
    eos = tokenizer.token_to_id("<|eos|>")
    pad = tokenizer.token_to_id("<|pad|>")
    require(None not in (story, bos, eos, pad), "tokenizer lacks required control tokens")
    require(boundary == "", "serialization must not insert bytes between opening and continuation")
    require(record["opening"].endswith("\n\n") and not record["opening"].endswith("\n\n\n"), f"{record['example_id']}: opening must end in exactly two LF bytes")
    require(bool(record["continuation"]) and not record["continuation"][0].isspace(), f"{record['example_id']}: continuation must not start with whitespace")
    opening_ids = tokenizer.encode(record["opening"]).ids
    continuation_ids = tokenizer.encode(record["continuation"]).ids
    token_ids = [int(story), int(bos), *opening_ids, *continuation_ids, int(eos)]
    boundary_index = 2 + len(opening_ids)
    require(token_ids == [int(story), int(bos), *tokenizer.encode(record["opening"] + record["continuation"]).ids, int(eos)], f"{record['example_id']}: separate tokenization changes the frozen opening/continuation boundary")
    require(len(token_ids) <= maximum_tokens, f"{record['example_id']}: {len(token_ids)} tokens exceeds {maximum_tokens}; truncation forbidden")
    input_ids = token_ids[:-1]
    labels = [-100] * (boundary_index - 1) + token_ids[boundary_index:]
    require(len(input_ids) == len(labels), "input/label alignment failed")
    require(labels[-1] == eos and sum(value != -100 for value in labels) == len(continuation_ids) + 1, "continuation/EOS target mask failed")
    if pad_to is not None:
        require(pad_to >= len(input_ids), "padding target is shorter than serialized example")
        padding = pad_to - len(input_ids)
        input_ids += [int(pad)] * padding
        labels += [-100] * padding
    return {
        "token_ids": token_ids, "input_ids": input_ids, "labels": labels,
        "token_count": len(token_ids), "opening_continuation_boundary_token_index": boundary_index,
        "continuation_target_count_including_eos": len(continuation_ids) + 1,
        "truncated": False,
    }


def _reference_surfaces(config: dict[str, Any]) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    surfaces: list[tuple[str, str]] = []
    for specification in config["contamination"]["benchmark_v2"]:
        path = Path(specification["path"])
        _verify(path, specification["sha256"], "Benchmark v2 authored surface file")
        for row_number, row in enumerate(load_jsonl(path), 1):
            for index, text in enumerate(_iter_strings(row)):
                if len(normalized_words(text)) >= 5:
                    surfaces.append((f"{path}:{row_number}:{index}", text))
    scene_spec = config["contamination"]["scene_scorecard"]
    scene_path = Path(scene_spec["path"])
    _verify(scene_path, scene_spec["sha256"], "Scene Scorecard prompts")
    prompts = load_jsonl(scene_path)
    require(len(prompts) == 10 and sum(row["prompt_id"].startswith("scene-post-") for row in prompts) == 5, "Scene Scorecard prompt set changed")
    for row in prompts:
        for field in ("prompt", "premise", "protagonist_motive"):
            surfaces.append((f"{row['prompt_id']}:{field}", row[field]))
        for fact in row["atomic_facts"]:
            surfaces.append((f"{row['prompt_id']}:{fact['id']}", fact["text"]))
    return surfaces, prompts


def contamination_audit(records: Sequence[dict[str, Any]], assignments: dict[str, str], config: dict[str, Any]) -> dict[str, Any]:
    references, prompts = _reference_surfaces(config)
    pilot_rows: list[tuple[str, str, str]] = []
    for row in records:
        pilot_rows.extend((
            (row["example_id"], "opening", row["opening"]),
            (row["example_id"], "continuation", row["continuation"]),
            (row["example_id"], "complete", row["opening"] + row["continuation"]),
            (row["example_id"], "premise", row["premise"]),
            (row["example_id"], "protagonist_motive", row["protagonist_motive"]),
            (row["example_id"], "action_summary", row["action_summary"]),
            (row["example_id"], "consequence_summary", row["consequence_summary"]),
        ))
        pilot_rows.extend((row["example_id"], "atomic_fact", fact) for fact in row["atomic_facts"])
        for transition in row["event_transitions"]:
            pilot_rows.extend(
                (row["example_id"], f"transition_{field}", transition[field])
                for field in ("before", "event", "after")
            )
    exact_seen: dict[str, tuple[str, str]] = {}
    exact_collisions: list[dict[str, str]] = []
    shingle_owners: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for identifier, kind, text in pilot_rows:
        digest = normalized_text_hash(text)
        if digest in exact_seen and exact_seen[digest][0] != identifier:
            exact_collisions.append({"left": exact_seen[digest][0], "right": identifier, "kind": kind})
        exact_seen[digest] = (identifier, kind)
        for shingle in text_shingles(text, width=5):
            shingle_owners[shingle].append((identifier, kind))
    pilot_shingle_collisions = []
    for shingle, owners in sorted(shingle_owners.items()):
        ids = sorted({owner[0] for owner in owners})
        if len(ids) > 1:
            pilot_shingle_collisions.append({"shingle": shingle, "example_ids": ids, "splits": sorted({assignments[value] for value in ids})})
    reference_exact = {normalized_text_hash(text): label for label, text in references}
    reference_shingles: dict[str, str] = {}
    for label, text in references:
        for shingle in text_shingles(text, width=5):
            reference_shingles.setdefault(shingle, label)
    reference_collisions = []
    for identifier, kind, text in pilot_rows:
        digest = normalized_text_hash(text)
        if digest in reference_exact:
            reference_collisions.append({"example_id": identifier, "kind": kind, "reference": reference_exact[digest], "match": "normalized_exact"})
        for shingle in sorted(text_shingles(text, width=5) & reference_shingles.keys()):
            reference_collisions.append({"example_id": identifier, "kind": kind, "reference": reference_shingles[shingle], "match": "normalized_5_word_shingle", "shingle": shingle})
    scorecard_names: set[str] = set()
    for prompt in prompts:
        candidate_text = " ".join([prompt["protagonist"], *[fact["text"] for fact in prompt["atomic_facts"]]])
        scorecard_names.update(match.casefold() for match in re.findall(r"\b[A-Z][a-z]{2,}\b", candidate_text))
    name_collisions = [
        {"example_id": row["example_id"], "name": name}
        for row in records for name in row["character_names"] if name.casefold() in scorecard_names
    ]
    post_inputs = [row["example_id"] for row in records if any(value.startswith("scene-post-") for value in row["authoring_inputs"])]
    passed = not (exact_collisions or pilot_shingle_collisions or reference_collisions or name_collisions or post_inputs)
    return {
        "passed": passed,
        "normalized_exact_pilot_collisions": exact_collisions,
        "normalized_5_word_shingle_pilot_collisions": pilot_shingle_collisions,
        "benchmark_and_scorecard_collisions": reference_collisions,
        "scorecard_name_collisions": name_collisions,
        "post_selection_authoring_input_violations": post_inputs,
        "semantic_limit": "Automated lexical checks cannot establish absence of paraphrased premise or distinctive-fact-combination collisions; custodian verification is required.",
        "benchmark_surface_count": len(references),
        "scene_scorecard_prompt_count": len(prompts),
    }


def audit_examples(records: list[dict[str, Any]], config: dict[str, Any], schema: dict[str, Any], tokenizer: Tokenizer) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ids = [row.get("example_id") for row in records]
    schema_errors = []
    for index, row in enumerate(records):
        try:
            validate_example(row, schema, config)
        except (PilotValidationError, ValueError) as error:
            schema_errors.append({"row": index + 1, "example_id": row.get("example_id"), "error": str(error)})
    assignments: dict[str, str] = {}
    split_error = None
    if not schema_errors and len(ids) == len(set(ids)):
        try:
            assignments = deterministic_splits(
                records, config["expected_split_counts"], int(config["seed"]),
                config["expected_split_genre_counts"],
            )
        except PilotValidationError as error:
            split_error = str(error)
    prepared = []
    serialization_errors = []
    if assignments:
        for row in sorted(records, key=lambda item: item["example_id"]):
            try:
                serialized = serialize_example(row, tokenizer, boundary=config["serialization"]["boundary"], maximum_tokens=int(config["serialization"]["maximum_tokens"]))
                prepared.append({**row, "split": assignments[row["example_id"]], "content_sha256": content_sha256(row), "serialization": serialized})
            except PilotValidationError as error:
                serialization_errors.append({"example_id": row["example_id"], "error": str(error)})
    count_gate = len(records) == int(config["expected_examples"])
    genre_counts = Counter(row.get("genre") for row in records)
    state_counts = Counter(row.get("primary_state_family") for row in records)
    stale_counts = Counter("present" if row.get("stale_state_distractor_present") is True else "absent" for row in records)
    genre_state_counts = Counter((row.get("genre"), row.get("primary_state_family")) for row in records)
    genre_gate = genre_counts == Counter(config["expected_genre_counts"])
    state_gate = state_counts == Counter(config["expected_state_family_counts"])
    stale_gate = stale_counts == Counter(config["expected_stale_state_distractor_counts"])
    expected_cell_count = int(config["expected_genre_state_cell_count"])
    genre_state_gate = genre_state_counts == Counter({(genre, family): expected_cell_count for genre in GENRES for family in STATE_FAMILIES})
    group_leaks = []
    if assignments:
        group_leaks = [
            {"example_ids": group, "splits": sorted({assignments[identifier] for identifier in group})}
            for group in _group_components(records)
            if len({assignments[identifier] for identifier in group}) > 1
        ]
    split_gate = bool(assignments) and Counter(assignments.values()) == Counter(config["expected_split_counts"]) and not group_leaks
    split_genre_actual = {
        split: {genre: sum(assignments.get(str(row.get("example_id"))) == split and row.get("genre") == genre for row in records) for genre in GENRES}
        for split in SPLITS
    }
    split_genre_gate = bool(assignments) and split_genre_actual == config["expected_split_genre_counts"]
    contamination = contamination_audit(records, assignments, config) if assignments else {"passed": False, "not_run": "schema or split gate failed"}
    token_counts_by_split = {
        split: sum(row["serialization"]["token_count"] for row in prepared if row["split"] == split)
        for split in SPLITS
    }
    gates = {
        "exact_example_count": {"passed": count_gate, "actual": len(records), "expected": config["expected_examples"]},
        "strict_schema_and_lengths": {"passed": not schema_errors and len(ids) == len(set(ids)), "errors": schema_errors, "unique_ids": len(ids) == len(set(ids))},
        "genre_balance": {"passed": genre_gate, "actual": dict(sorted(genre_counts.items())), "expected": config["expected_genre_counts"]},
        "primary_state_family_balance": {"passed": state_gate, "actual": dict(sorted(state_counts.items())), "expected": config["expected_state_family_counts"]},
        "stale_state_distractor_balance": {"passed": stale_gate, "actual": dict(sorted(stale_counts.items())), "expected": config["expected_stale_state_distractor_counts"], "purpose": "lexical/task-shortcut control"},
        "genre_primary_state_cross_balance": {"passed": genre_state_gate, "actual": {f"{genre}/{family}": genre_state_counts[(genre, family)] for genre in GENRES for family in STATE_FAMILIES}, "expected_per_cell": expected_cell_count, "purpose": "prevents one-to-one genre/state label shortcuts"},
        "grouped_exact_splits": {"passed": split_gate and split_genre_gate, "error": split_error, "actual": dict(sorted(Counter(assignments.values()).items())), "split_genre_counts": split_genre_actual, "expected_split_genre_counts": config["expected_split_genre_counts"], "source_story_or_duplicate_cluster_leaks": group_leaks},
        "contamination": contamination,
        "serialization_and_masking": {
            "passed": not serialization_errors and len(prepared) == len(records) and bool(records),
            "errors": serialization_errors,
            "maximum_tokens": config["serialization"]["maximum_tokens"],
            "truncation_allowed": False,
            "tokenizer_v1_tokens_total": sum(token_counts_by_split.values()),
            "tokenizer_v1_tokens_by_split": token_counts_by_split,
        },
        "rights_and_provenance_documented": {"passed": not schema_errors and bool(records), "note": "Document presence is not proof of legal clearance; custodian verification is required at finalize."},
    }
    failed = [name for name, gate in gates.items() if not gate["passed"]]
    return {"status": "PASS_AUTOMATED_AUDIT" if not failed else "STOP", "approved": False, "failed_gates": failed, "gates": gates}, prepared


def prepare(config_path: Path, *, enforce_production: bool = True) -> dict[str, Any]:
    output_dir = _raw_output_dir(config_path)
    _remove_approval(output_dir)
    if enforce_production and output_dir != DEFAULT_OUTPUT_DIR:
        _remove_approval(DEFAULT_OUTPUT_DIR)
    report_path = output_dir / "audit-report.json"
    try:
        config = _load_config(config_path, enforce_production=enforce_production)
        source_path = Path(config["examples_path"])
        if not source_path.is_file():
            report = {
                "version": VERSION, "status": "STOP", "approved": False,
                "failed_gates": ["canonical_examples_present", "rights_clearance", "human_review"],
                "pending_external_inputs": [
                    f"exactly {config['expected_examples']} canonical examples at {source_path}",
                    "per-example rights/provenance evidence and custodian verification",
                    "two genuine independent human reviews per example and adjudication where required",
                ],
            }
            write_json_atomic(report_path, report)
            return report
        schema_path = Path(config["schema_path"])
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        tokenizer_spec = config["tokenizer"]
        tokenizer_path = Path(tokenizer_spec["path"])
        _verify(tokenizer_path, tokenizer_spec["sha256"], "tokenizer-v1")
        records = load_jsonl(source_path)
        report, prepared = audit_examples(records, config, schema, Tokenizer.from_file(str(tokenizer_path)))
        write_jsonl(output_dir / "prepared.jsonl", prepared)
        report.update({
            "version": VERSION, "approved": False, "examples_sha256": sha256_file(source_path),
            "prepared_sha256": sha256_file(output_dir / "prepared.jsonl"), "config_sha256": sha256_file(config_path),
            "schema_sha256": sha256_file(schema_path), "tokenizer_sha256": tokenizer_spec["sha256"],
            "production_contract_enforced": enforce_production,
            "pending_external_inputs": ["custodian verification", "two independent reviews and any required adjudication"],
        })
        write_json_atomic(report_path, report)
        return report
    except Exception as error:
        _remove_approval(output_dir)
        report = {
            "version": VERSION, "status": "STOP", "approved": False,
            "error": f"{type(error).__name__}: {error}", "pending_external_inputs": True,
        }
        write_json_atomic(report_path, report)
        return report


def audit(config_path: Path, *, enforce_production: bool = True) -> dict[str, Any]:
    return prepare(config_path, enforce_production=enforce_production)


def export_review(config_path: Path, *, enforce_production: bool = True) -> dict[str, Any]:
    report = prepare(config_path, enforce_production=enforce_production)
    if report["status"] != "PASS_AUTOMATED_AUDIT":
        return report
    config = _load_config(config_path, enforce_production=enforce_production)
    output_dir = Path(config["output_dir"])
    prepared_path = output_dir / "prepared.jsonl"
    packet = []
    key: dict[str, Any] = {}
    for row in load_jsonl(prepared_path):
        review_id = "pilot-review-" + sha256_bytes(f"{row['example_id']}:{row['content_sha256']}".encode())[:20]
        packet_row = {
            "review_id": review_id, "example_id": row["example_id"], "genre": row["genre"],
            "opening": row["opening"], "continuation": row["continuation"], "premise": row["premise"],
            "protagonist": row["protagonist"], "protagonist_motive": row["protagonist_motive"],
            "atomic_facts": row["atomic_facts"], "content_sha256": row["content_sha256"],
            "reviewer_id": "", "reviewer_role": "original", "independent_review_confirmed": None,
            "genuine_human_attestation": None, "fact_labels": {},
            **{field: "" for field in REVIEW_FIELDS}, "evidence": {}, "reviewer_note": "",
        }
        packet.append(packet_row)
        key[review_id] = {"example_id": row["example_id"], "split": row["split"], "content_sha256": row["content_sha256"]}
    packet.sort(key=lambda row: hashlib.sha256(f"{config['review_seed']}:{row['review_id']}".encode()).hexdigest())
    packet_path = output_dir / "review-packet.jsonl"
    write_jsonl(packet_path, packet)
    private_key = {
        "version": VERSION, "prepared_sha256": sha256_file(prepared_path), "packet_sha256": sha256_file(packet_path),
        "config_sha256": report["config_sha256"], "schema_sha256": report["schema_sha256"],
        "tokenizer_sha256": report["tokenizer_sha256"],
        "reviews_must_bind_packet_sha256": True, "mapping": key,
        "warning": "IDs and attestations do not prove human identity, independence, or rights. Finalization requires separate custodian verification.",
    }
    write_json_atomic(output_dir / "review-key.json", private_key)
    return {"status": "REVIEW_PENDING", "approved": False, "review_count": len(packet), "packet_sha256": private_key["packet_sha256"]}


def load_prepared_records(
    prepared_path: Path, *, splits: Sequence[str] = ("train", "validation"),
    post_selection_complete: bool = False,
) -> list[dict[str, Any]]:
    """Load audit artifacts with test hidden until explicit post-selection completion."""
    requested = set(splits)
    require(bool(requested) and requested <= set(SPLITS), "prepared loader requested an unknown split")
    require("test" not in requested or post_selection_complete, "test split requires post-selection completion attestation")
    rows = load_jsonl(prepared_path)
    require(all(row.get("split") in SPLITS for row in rows), "prepared artifact contains an invalid split")
    return [row for row in rows if row["split"] in requested]


def _validate_review(row: dict[str, Any], packet: dict[str, Any]) -> None:
    immutable = {key: packet[key] for key in ("review_id", "example_id", "genre", "opening", "continuation", "premise", "protagonist", "protagonist_motive", "atomic_facts", "content_sha256")}
    expected = {*packet.keys(), "packet_sha256"}
    require(set(row) == expected, f"{packet['review_id']}: review fields differ from frozen packet")
    require(all(row[key] == value for key, value in immutable.items()), f"{packet['review_id']}: review content changed")
    require(row["packet_sha256"] and re.fullmatch(r"[0-9a-f]{64}", row["packet_sha256"]) is not None, "review packet hash missing")
    require(row["reviewer_role"] in {"original", "adjudicator"}, "invalid reviewer role")
    require(bool(str(row["reviewer_id"]).strip()) and row["reviewer_id"] == row["reviewer_id"].strip(), "reviewer_id required")
    require(row["independent_review_confirmed"] is True and row["genuine_human_attestation"] is True, "independent genuine-human attestations required")
    require(set(row["fact_labels"]) == set(row["atomic_facts"]), "fact labels incomplete")
    require(set(row["fact_labels"].values()) <= FACT_LABELS, "invalid fact label")
    require(all(row[field] in REVIEW_LABELS for field in REVIEW_FIELDS), "invalid review label")
    evidence_fields = {*REVIEW_FIELDS, *row["atomic_facts"]}
    require(set(row["evidence"]) == evidence_fields and all(str(value).strip() for value in row["evidence"].values()), "review evidence incomplete")
    prose = row["opening"] + "\n" + row["continuation"]
    require(all(value == "<ABSENT>" or value in prose for value in row["evidence"].values()), "review evidence must quote prose verbatim or use <ABSENT>")
    require(bool(row["reviewer_note"].strip()), "reviewer note required")


def resolve_reviews(packet: list[dict[str, Any]], reviews: list[dict[str, Any]], packet_sha256: str) -> list[dict[str, Any]]:
    by_id = {row["review_id"]: row for row in packet}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reviews:
        require(row.get("review_id") in by_id, f"unknown review_id: {row.get('review_id')}")
        _validate_review(row, by_id[row["review_id"]])
        require(row["packet_sha256"] == packet_sha256, "review packet hash mismatch; reviews are invalidated")
        grouped[row["review_id"]].append(row)
    resolved = []
    for identifier in by_id:
        originals = [row for row in grouped[identifier] if row["reviewer_role"] == "original"]
        adjudicators = [row for row in grouped[identifier] if row["reviewer_role"] == "adjudicator"]
        require(len(originals) == 2 and len({row["reviewer_id"] for row in originals}) == 2, f"{identifier}: exactly two distinct original reviewers required")
        disputed_fields = {
            field for field in REVIEW_FIELDS
            if originals[0][field] != originals[1][field] or "uncertain" in {originals[0][field], originals[1][field]}
        }
        disputed_facts = {
            fact for fact in originals[0]["fact_labels"]
            if originals[0]["fact_labels"][fact] != originals[1]["fact_labels"][fact]
            or "uncertain" in {originals[0]["fact_labels"][fact], originals[1]["fact_labels"][fact]}
        }
        if disputed_fields or disputed_facts:
            require(len(adjudicators) == 1, f"{identifier}: independent adjudication required")
            adjudicator = adjudicators[0]
            require(adjudicator["reviewer_id"] not in {row["reviewer_id"] for row in originals}, f"{identifier}: adjudicator must be distinct")
            require(all(adjudicator[field] != "uncertain" for field in disputed_fields), f"{identifier}: adjudication did not resolve field uncertainty")
            require(all(adjudicator["fact_labels"][fact] != "uncertain" for fact in disputed_facts), f"{identifier}: adjudication did not resolve fact uncertainty")
            merged = dict(originals[0])
            merged["fact_labels"] = dict(originals[0]["fact_labels"])
            merged["evidence"] = dict(originals[0]["evidence"])
            provenance: dict[str, Any] = {
                "original_reviewer_ids": [originals[0]["reviewer_id"], originals[1]["reviewer_id"]],
                "adjudicator_id": adjudicator["reviewer_id"], "fields": {}, "facts": {},
            }
            for field in REVIEW_FIELDS:
                if field in disputed_fields:
                    merged[field] = adjudicator[field]
                    merged["evidence"][field] = adjudicator["evidence"][field]
                    provenance["fields"][field] = "adjudicated"
                else:
                    provenance["fields"][field] = "original_consensus"
            for fact in merged["fact_labels"]:
                if fact in disputed_facts:
                    merged["fact_labels"][fact] = adjudicator["fact_labels"][fact]
                    merged["evidence"][fact] = adjudicator["evidence"][fact]
                    provenance["facts"][fact] = "adjudicated"
                else:
                    provenance["facts"][fact] = "original_consensus"
            merged["reviewer_id"] = f"merged:{originals[0]['reviewer_id']}:{originals[1]['reviewer_id']}:{adjudicator['reviewer_id']}"
            merged["reviewer_role"] = "resolved"
            merged["reviewer_note"] = canonical_json({
                "original_notes": [originals[0]["reviewer_note"], originals[1]["reviewer_note"]],
                "adjudicator_note": adjudicator["reviewer_note"],
            })
            merged["resolution_provenance"] = provenance
            resolved.append(merged)
        else:
            require(not adjudicators, f"{identifier}: unnecessary adjudication")
            resolved.append(originals[0])
    return resolved


def resolved_reviews_pass_quality(resolved: Sequence[dict[str, Any]]) -> bool:
    return bool(resolved) and all(
        all(value == "retained" for value in row["fact_labels"].values())
        and all(row[field] == "yes" for field in REVIEW_FIELDS)
        for row in resolved
    )


def _validate_custodian(value: dict[str, Any], *, prepared_sha256: str, packet_sha256: str, reviewer_ids: set[str], adjudicator_ids: set[str]) -> None:
    expected = {
        "version", "prepared_sha256", "review_packet_sha256", "custodian_id", "verified_at",
        "verification_evidence_reference", "rights_evidence_examined", "provenance_evidence_examined",
        "all_examples_rights_cleared", "reviewer_identities_verified", "reviewers_genuine_humans",
        "independent_review_process_verified", "adjudicator_identities_verified",
        "distinctive_scorecard_fact_combinations_checked",
        "no_distinctive_scorecard_fact_combination_collision",
        "scene_post_prompts_excluded_from_authoring_editing_and_candidate_selection",
        "verified_reviewer_ids", "verified_adjudicator_ids",
    }
    require(set(value) == expected, "custodian verification fields differ from frozen contract")
    require(value["version"] == VERSION and value["prepared_sha256"] == prepared_sha256 and value["review_packet_sha256"] == packet_sha256, "custodian verification hash mismatch")
    require(all(bool(str(value[field]).strip()) for field in ("custodian_id", "verified_at", "verification_evidence_reference")), "custodian identity, date, and evidence reference required")
    booleans = expected - {"version", "prepared_sha256", "review_packet_sha256", "custodian_id", "verified_at", "verification_evidence_reference", "verified_reviewer_ids", "verified_adjudicator_ids"}
    require(all(value[field] is True for field in booleans), "custodian verification gate is incomplete")
    require(set(value["verified_reviewer_ids"]) == reviewer_ids, "custodian did not verify every original reviewer")
    require(set(value["verified_adjudicator_ids"]) == adjudicator_ids, "custodian adjudicator verification mismatch")


def finalize(config_path: Path, *, enforce_production: bool = True) -> dict[str, Any]:
    output_dir = _raw_output_dir(config_path)
    _remove_approval(output_dir)
    if enforce_production and output_dir != DEFAULT_OUTPUT_DIR:
        _remove_approval(DEFAULT_OUTPUT_DIR)
    report_path = output_dir / "finalization-report.json"
    try:
        require(enforce_production, "test/helper configurations can never produce approval")
        config = _load_config(config_path, enforce_production=True)
        audit_report = prepare(config_path, enforce_production=True)
        audit_failure = audit_report.get("failed_gates", [audit_report.get("error", "unknown audit failure")])
        require(audit_report["status"] == "PASS_AUTOMATED_AUDIT", f"automated audit has not passed: {audit_failure}")
        packet_path = output_dir / "review-packet.jsonl"
        key_path = output_dir / "review-key.json"
        require(packet_path.is_file() and key_path.is_file(), "review export is pending")
        key = json.loads(key_path.read_text(encoding="utf-8"))
        prepared_sha256 = sha256_file(output_dir / "prepared.jsonl")
        packet_sha256 = sha256_file(packet_path)
        require(key["prepared_sha256"] == prepared_sha256 and key["packet_sha256"] == packet_sha256, "prepared data changed; reviews are invalidated")
        require(
            key.get("config_sha256") == audit_report["config_sha256"]
            and key.get("schema_sha256") == audit_report["schema_sha256"]
            and key.get("tokenizer_sha256") == audit_report["tokenizer_sha256"],
            "pilot contract changed; reviews are invalidated",
        )
        reviews_path = Path(config["reviews_path"])
        custodian_path = Path(config["custodian_verification_path"])
        require(reviews_path.is_file(), f"completed reviews pending: {reviews_path}")
        require(custodian_path.is_file(), f"custodian verification pending: {custodian_path}")
        reviews = load_jsonl(reviews_path)
        resolved = resolve_reviews(load_jsonl(packet_path), reviews, packet_sha256)
        require(resolved_reviews_pass_quality(resolved), "one or more examples failed final human quality approval")
        reviewer_ids = {row["reviewer_id"] for row in reviews if row["reviewer_role"] == "original"}
        adjudicator_ids = {row["reviewer_id"] for row in reviews if row["reviewer_role"] == "adjudicator"}
        custodian = json.loads(custodian_path.read_text(encoding="utf-8"))
        _validate_custodian(custodian, prepared_sha256=prepared_sha256, packet_sha256=packet_sha256, reviewer_ids=reviewer_ids, adjudicator_ids=adjudicator_ids)
        report = {
            "version": VERSION, "status": "APPROVED", "approved": True,
            "example_count": len(resolved), "prepared_sha256": prepared_sha256,
            "review_packet_sha256": packet_sha256, "reviews_sha256": sha256_file(reviews_path),
            "custodian_verification_sha256": sha256_file(custodian_path),
            "tokenizer_v1_tokens_total": audit_report["gates"]["serialization_and_masking"]["tokenizer_v1_tokens_total"],
            "tokenizer_v1_tokens_by_split": audit_report["gates"]["serialization_and_masking"]["tokenizer_v1_tokens_by_split"],
            "warning": "Approval records custodian verification; software cannot independently prove human identity, independence, or legal rights.",
        }
        write_json_atomic(report_path, report)
        write_json_atomic(output_dir / APPROVAL_FILE, report)
        return report
    except Exception as error:
        report = {"version": VERSION, "status": "STOP", "approved": False, "error": str(error), "pending_external_inputs": True}
        write_json_atomic(report_path, report)
        _remove_approval(output_dir)
        if enforce_production and output_dir != DEFAULT_OUTPUT_DIR:
            _remove_approval(DEFAULT_OUTPUT_DIR)
        return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "audit", "export-review", "finalize"))
    parser.add_argument("--config", type=Path, default=Path("configs/continuation-pilot-v1.yaml"))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    handlers = {"prepare": prepare, "audit": audit, "export-review": export_review, "finalize": finalize}
    report = handlers[args.command](args.config)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("status") == "STOP":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
