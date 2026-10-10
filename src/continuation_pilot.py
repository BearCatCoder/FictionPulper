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
    "rights_and_provenance_acceptable", "premise_preserved", "motive_preserved",
    "opening_facts_consistent", "contradiction_free", "no_narrative_loop",
    "plot_advanced", "genre_voice", "meaningful_action", "meaningful_consequence",
    "natural_readable_continuation", "exploratory_sft_suitable",
)
REVIEW_LABELS = {"yes", "no", "uncertain"}
FACT_LABELS = {"retained", "contradicted", "uncertain"}
EVIDENCE_MINIMUM_WORDS = 3
APPROVAL_MODES = {"single_owner_research_pilot", "strict_independent"}
RISK_FLAGS = (
    "ambiguous", "low_quality", "possible_contamination", "possible_duplicate",
    "rights_sensitive", "provenance_incomplete", "semantic_scorecard_collision",
    "heldout_prompt_risk", "other_problematic",
)
CORPUS_V3_CANDIDATE_REFERENCE = re.compile(
    r"(?:fictionpulper[-_/ ]?corpus[-_/ ]?v3|corpus[-_/ ]?v3|data[/\\]corpus_v3)",
    re.IGNORECASE,
)
APPROVAL_FILE = "approval.json"
DEFAULT_OUTPUT_DIR = Path("runs/continuation-pilot-v1")
PRODUCTION_SCHEMA_SHA256 = "2057ea5c6f540ac68a54acd629551f9d3a44066bae81013be1bfac5e1dbb0634"
PRODUCTION_CONTRACT: dict[str, Any] = {
    "version": 1,
    "seed": 8008,
    "review_seed": 8018,
    "approval_mode": "single_owner_research_pilot",
    "owner_review": {
        "target_sample_size": 50,
        "selection_algorithm": "deterministic_stratified_all_flags_v1",
    },
    "risk_flag_vocabulary": list(RISK_FLAGS),
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
    "examples_path", "reviews_path", "owner_verification_path", "strict_verification_path",
    "output_dir", "retained_artifact_roots",
}
RETAINED_ARTIFACT_ROOTS = {"sources", "rights_evidence", "prompts", "raw_outputs"}
RETAINED_ARTIFACT_DIRECTORIES = {
    "sources": "sources", "rights_evidence": "rights_evidence",
    "prompts": "prompts", "raw_outputs": "raw_outputs",
}


class PilotValidationError(ValueError):
    """Raised when pilot inputs do not satisfy the frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotValidationError(message)


def _load_config(
    path: Path, *, enforce_production: bool = True, approval_mode: str | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(config, dict) and config.get("version") == VERSION, "pilot config must declare version: 1")
    if enforce_production:
        require(set(config) == set(PRODUCTION_CONTRACT) | RELOCATABLE_CONFIG_FIELDS, "production config fields differ from frozen contract")
        for field, expected in PRODUCTION_CONTRACT.items():
            require(config.get(field) == expected, f"production contract changed: {field}")
        schema_path = Path(config["schema_path"])
        _verify(schema_path, PRODUCTION_SCHEMA_SHA256, "production continuation-pilot schema")
    selected_mode = approval_mode or config.get("approval_mode")
    require(selected_mode in APPROVAL_MODES, f"unknown approval mode: {selected_mode}")
    roots = config.get("retained_artifact_roots")
    require(isinstance(roots, dict), "retained artifact roots must be an object")
    assert isinstance(roots, dict)
    require(set(roots) == RETAINED_ARTIFACT_ROOTS, "retained artifact roots differ from frozen contract")
    require(all(isinstance(value, str) and value for value in roots.values()), "retained artifact roots must be non-empty paths")
    if enforce_production:
        data_root = Path(config["examples_path"]).parent
        expected_roots = {
            name: data_root / directory for name, directory in RETAINED_ARTIFACT_DIRECTORIES.items()
        }
        require(
            all(Path(roots[name]) == expected for name, expected in expected_roots.items()),
            "production retained artifact roots must be fixed subdirectories beside canonical examples",
        )
    config["approval_mode"] = selected_mode
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


def _review_artifact_paths(output_dir: Path, approval_mode: str) -> tuple[Path, Path]:
    suffix = "" if approval_mode == "single_owner_research_pilot" else "-strict-independent"
    return output_dir / f"review-packet{suffix}.jsonl", output_dir / f"review-key{suffix}.json"


def _write_jsonl_once(path: Path, rows: list[dict[str, Any]]) -> None:
    expected = _jsonl_bytes(rows)
    if path.exists():
        require(path.read_bytes() == expected, f"write-once review artifact already exists with different content: {path}")
        return
    write_jsonl(path, rows)


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    expected = _json_bytes(value)
    if path.exists():
        require(path.read_bytes() == expected, f"write-once review artifact already exists with different content: {path}")
        return
    write_json_atomic(path, value)


def _jsonl_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


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


def _references_corpus_v3_candidate(value: Any) -> bool:
    return any(CORPUS_V3_CANDIDATE_REFERENCE.search(text) is not None for text in _iter_strings(value))


def content_sha256(record: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(record).encode("utf-8"))


def _retained_bytes(
    config: dict[str, Any], root_name: str, relative_path: str, expected_sha256: str, label: str,
) -> bytes:
    relative = Path(relative_path)
    require(
        not relative.is_absolute()
        and bool(relative.parts)
        and all(part not in {"", ".", ".."} for part in relative.parts)
        and relative.as_posix() == relative_path,
        f"{label}: retained path must be a safe root-relative POSIX path",
    )
    root = Path(config["retained_artifact_roots"][root_name]).resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise PilotValidationError(f"{label}: retained path escapes approved root") from error
    require(candidate.is_file(), f"{label}: retained bytes are unavailable: {relative_path}")
    data = candidate.read_bytes()
    require(sha256_bytes(data) == expected_sha256, f"{label}: retained byte hash mismatch")
    return data


def _prompt_text(prompt: dict[str, Any], config: dict[str, Any], label: str) -> str:
    if "text" in prompt:
        data = prompt["text"].encode("utf-8")
        require(sha256_bytes(data) == prompt["sha256"], f"{label}: prompt hash mismatch")
    else:
        data = _retained_bytes(config, "prompts", prompt["reference"], prompt["sha256"], label)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PilotValidationError(f"{label}: retained prompt must be UTF-8 text") from error


def _retained_text(
    config: dict[str, Any], root_name: str, relative_path: str, expected_sha256: str, label: str,
) -> str:
    data = _retained_bytes(config, root_name, relative_path, expected_sha256, label)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PilotValidationError(f"{label}: retained text must be UTF-8") from error


def _expected_assessment_output(record: dict[str, Any]) -> dict[str, Any]:
    assessment = record["preliminary_assessment"]
    output = {
        "version": VERSION,
        "example_id": record["example_id"],
        "assessed_content_sha256": sha256_bytes(
            (record["opening"] + record["continuation"]).encode("utf-8")
        ),
        "method": assessment["method"],
        "assessor_identity": assessment["assessor_identity"],
        "prompt_sha256": assessment["prompt"]["sha256"],
        "generation_configuration": assessment["generation_configuration"],
        "assessment_date": assessment["assessment_date"],
        "risk_flags": assessment["risk_flags"],
        "summary": assessment["summary"],
    }
    if assessment["method"] == "ai_assisted":
        output["model_identity"] = assessment["model_identity"]
    return output


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
    require(
        not _references_corpus_v3_candidate(record["authoring_inputs"]),
        f"{record['example_id']}: unsealed Corpus-v3 candidate ID/path used as an authoring input",
    )
    rights = record["rights"]
    provenance = record["provenance"]
    require(re.fullmatch(r"[0-9a-f]{64}", rights["evidence_sha256"]) is not None, f"{record['example_id']}: invalid rights evidence hash")
    _retained_bytes(
        config, "rights_evidence", rights["evidence_path"], rights["evidence_sha256"],
        f"{record['example_id']}: rights evidence",
    )
    source_class = provenance["source_class"]
    expected_provenance = {"source_class", "source_id", "original_story_id", "pretraining_exposure"}
    if source_class in {"public_domain_adaptation", "ai_assisted_public_domain_adaptation"}:
        expected_provenance.add("public_domain_source")
        public_domain = provenance.get("public_domain_source")
        require(isinstance(public_domain, dict), f"{record['example_id']}: public-domain source provenance required")
        require(
            public_domain["independently_acquired_from_documented_source"] is True
            and public_domain["copied_from_unsealed_corpus_v3_candidate"] is False
            and public_domain["acquisition_method"] == "independent_download_from_documented_source_url",
            f"{record['example_id']}: public-domain source must be independently acquired and not copied from unsealed Corpus-v3",
        )
        require(
            not _references_corpus_v3_candidate({
                "source_path": public_domain["source_path"],
                "acquisition_evidence_reference": public_domain["acquisition_evidence_reference"],
                "acquisition_evidence_path": public_domain["acquisition_evidence_path"],
            }),
            f"{record['example_id']}: acquisition evidence references an unsealed Corpus-v3 candidate artifact",
        )
        _retained_bytes(
            config, "rights_evidence", public_domain["acquisition_evidence_path"],
            public_domain["acquisition_evidence_sha256"],
            f"{record['example_id']}: independent acquisition evidence",
        )
        _retained_bytes(
            config, "sources", public_domain["source_path"], public_domain["source_sha256"],
            f"{record['example_id']}: public-domain source",
        )
        require(rights["basis"] == "public_domain", f"{record['example_id']}: adaptation rights basis must be public_domain")
        require(
            public_domain["rights_evidence_reference"] == rights["evidence_reference"]
            and public_domain["rights_evidence_sha256"] == rights["evidence_sha256"]
            and public_domain["rights_check_date"] == rights["checked_at"],
            f"{record['example_id']}: public-domain rights linkage mismatch",
        )
        if public_domain["corpus_v3_overlap"] in {"identified", "uncertain"}:
            require(
                provenance["pretraining_exposure"]["classification"] in {"identified", "uncertain"},
                f"{record['example_id']}: Corpus-v3 source overlap requires identified/uncertain pretraining exposure disclosure",
            )
    if source_class in {"ai_assisted_original", "ai_assisted_public_domain_adaptation"}:
        expected_provenance.add("ai_assistance")
        ai = provenance.get("ai_assistance")
        require(isinstance(ai, dict), f"{record['example_id']}: AI-assistance provenance required")
        _prompt_text(ai["prompt"], config, f"{record['example_id']}: AI authoring")
        _retained_bytes(
            config, "raw_outputs", ai["raw_output_path"], ai["raw_output_sha256"],
            f"{record['example_id']}: AI raw output",
        )
        final_hash = sha256_bytes((record["opening"] + record["continuation"]).encode("utf-8"))
        require(ai["final_content_sha256"] == final_hash, f"{record['example_id']}: final content hash linkage mismatch")
        if source_class == "ai_assisted_original":
            require(rights["basis"] == "original_author_owned", f"{record['example_id']}: AI original rights basis must be original_author_owned")
    require(set(provenance) == expected_provenance, f"{record['example_id']}: provenance fields do not match source class")
    assessment = record["preliminary_assessment"]
    _prompt_text(assessment["prompt"], config, f"{record['example_id']}: preliminary assessment")
    assessment_fields = {
        "method", "assessor_identity", "prompt", "generation_configuration", "assessment_date",
        "raw_output_path", "raw_output_sha256", "risk_flags", "summary",
    }
    if assessment["method"] == "ai_assisted":
        assessment_fields.add("model_identity")
    require(set(assessment) == assessment_fields, f"{record['example_id']}: preliminary assessment fields do not match method")
    assessment_bytes = _retained_bytes(
        config, "raw_outputs", assessment["raw_output_path"], assessment["raw_output_sha256"],
        f"{record['example_id']}: preliminary assessment raw output",
    )
    flags = assessment["risk_flags"]
    require(len(flags) == len(set(flags)), f"{record['example_id']}: duplicate preliminary risk flags")
    require(set(flags) <= set(config["risk_flag_vocabulary"]) == set(RISK_FLAGS), f"{record['example_id']}: invalid or weakened risk vocabulary")
    expected_assessment = _expected_assessment_output(record)
    try:
        parsed_assessment = json.loads(assessment_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PilotValidationError(f"{record['example_id']}: preliminary assessment output must be canonical JSON") from error
    require(isinstance(parsed_assessment, dict), f"{record['example_id']}: preliminary assessment output must be a JSON object")
    expected_bytes = (canonical_json(expected_assessment) + "\n").encode("utf-8")
    require(parsed_assessment == expected_assessment, f"{record['example_id']}: preliminary assessment output does not exactly match metadata")
    require(assessment_bytes == expected_bytes, f"{record['example_id']}: preliminary assessment output is not canonical JSON")


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


def _source_input_exclusion_audit(
    records: Sequence[dict[str, Any]], references: Sequence[tuple[str, str]], config: dict[str, Any],
) -> dict[str, Any]:
    distinctive_width = 12
    sources: dict[tuple[str, str], str] = {}
    for row in records:
        source = row["provenance"].get("public_domain_source")
        if source is not None:
            key = (source["source_path"], source["source_sha256"])
            sources.setdefault(key, row["example_id"])
    reference_rules = []
    for reference, reference_text in references:
        words = normalized_words(reference_text)
        reference_rules.append((
            reference, words, " ".join(words),
            text_shingles(reference_text, width=distinctive_width) if len(words) >= distinctive_width else set(),
        ))
    collisions = []
    for (path, digest), example_id in sorted(sources.items()):
        source_text = _retained_text(
            config, "sources", path, digest, f"{example_id}: public-domain source",
        )
        source_words = normalized_words(source_text)
        source_normalized = " ".join(source_words)
        source_long_shingles = text_shingles(source_text, width=distinctive_width)
        for reference, reference_words, reference_normalized, reference_long_shingles in reference_rules:
            match = None
            detail: dict[str, Any] = {}
            if source_normalized == reference_normalized:
                match = "normalized_exact"
            elif reference_words and len(reference_words) <= len(source_words) and any(
                source_words[index:index + len(reference_words)] == reference_words
                for index in range(len(source_words) - len(reference_words) + 1)
            ):
                match = "full_reference_containment"
            elif len(reference_words) >= distinctive_width:
                overlap = source_long_shingles & reference_long_shingles
                if overlap:
                    match = "normalized_12_word_shingle"
                    detail["shingle"] = min(overlap)
            if match is not None:
                collisions.append({
                    "example_id": example_id, "source_path": path, "reference": reference,
                    "match": match, **detail,
                })
    return {
        "passed": not collisions,
        "retained_source_count": len(sources),
        "collisions": collisions,
        "rule": "Normalized exact and full-reference containment are prohibited. Distinctive overlap uses normalized 12-word shingles; the ordinary five-word pilot rule is intentionally not applied to complete source works.",
    }


def contamination_audit(records: Sequence[dict[str, Any]], assignments: dict[str, str], config: dict[str, Any]) -> dict[str, Any]:
    references, prompts = _reference_surfaces(config)
    source_input_exclusion = _source_input_exclusion_audit(records, references, config)
    pilot_rows: list[tuple[str, str, str]] = []
    candidate_metadata_rows: list[tuple[str, str, str]] = []
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
        candidate_metadata_rows.extend(
            (row["example_id"], f"candidate_metadata_{index}", text)
            for index, text in enumerate(_iter_strings({
                "authoring_inputs": row["authoring_inputs"],
                "rights": row["rights"],
                "provenance": row["provenance"],
                "preliminary_assessment": row["preliminary_assessment"],
            }))
        )
        ai = row["provenance"].get("ai_assistance")
        if ai is not None:
            candidate_metadata_rows.append((
                row["example_id"], "ai_authoring_prompt_resolved",
                _prompt_text(ai["prompt"], config, f"{row['example_id']}: AI authoring"),
            ))
            candidate_metadata_rows.append((
                row["example_id"], "ai_raw_output_resolved",
                _retained_text(
                    config, "raw_outputs", ai["raw_output_path"], ai["raw_output_sha256"],
                    f"{row['example_id']}: AI raw output",
                ),
            ))
        assessment = row["preliminary_assessment"]
        candidate_metadata_rows.append((
            row["example_id"], "preliminary_assessment_prompt_resolved",
            _prompt_text(assessment["prompt"], config, f"{row['example_id']}: preliminary assessment"),
        ))
        candidate_metadata_rows.append((
            row["example_id"], "preliminary_assessment_raw_output_resolved",
            _retained_text(
                config, "raw_outputs", assessment["raw_output_path"], assessment["raw_output_sha256"],
                f"{row['example_id']}: preliminary assessment raw output",
            ),
        ))
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
    for identifier, kind, text in [*pilot_rows, *candidate_metadata_rows]:
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
    passed = not (
        exact_collisions or pilot_shingle_collisions or reference_collisions
        or name_collisions or post_inputs or not source_input_exclusion["passed"]
    )
    return {
        "passed": passed,
        "normalized_exact_pilot_collisions": exact_collisions,
        "normalized_5_word_shingle_pilot_collisions": pilot_shingle_collisions,
        "benchmark_and_scorecard_collisions": reference_collisions,
        "scorecard_name_collisions": name_collisions,
        "post_selection_authoring_input_violations": post_inputs,
        "source_input_exclusion": source_input_exclusion,
        "semantic_limit": "Automated lexical checks cannot establish absence of paraphrased premise or distinctive-fact-combination collisions; hash-bound final human verification is required.",
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
    risk_counts = Counter(
        flag for row in records for flag in row.get("preliminary_assessment", {}).get("risk_flags", [])
    )
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
        "preliminary_assessment_provenance": {
            "passed": not schema_errors and bool(records),
            "risk_flag_counts": {flag: risk_counts[flag] for flag in RISK_FLAGS},
            "limitation": "Automated and AI-assisted assessments are diagnostics only, never human review. Risk flags cannot convert any automated hard-gate failure into a pass.",
        },
        "serialization_and_masking": {
            "passed": not serialization_errors and len(prepared) == len(records) and bool(records),
            "errors": serialization_errors,
            "maximum_tokens": config["serialization"]["maximum_tokens"],
            "truncation_allowed": False,
            "tokenizer_v1_tokens_total": sum(token_counts_by_split.values()),
            "tokenizer_v1_tokens_by_split": token_counts_by_split,
        },
        "rights_and_provenance_documented": {"passed": not schema_errors and bool(records), "note": "Document presence is not proof of legal clearance; mode-specific final human verification is required."},
    }
    failed = [name for name, gate in gates.items() if not gate["passed"]]
    return {"status": "PASS_AUTOMATED_AUDIT" if not failed else "STOP", "approved": False, "failed_gates": failed, "gates": gates}, prepared


def prepare(
    config_path: Path, *, enforce_production: bool = True, approval_mode: str | None = None,
) -> dict[str, Any]:
    output_dir = _raw_output_dir(config_path)
    _remove_approval(output_dir)
    if enforce_production and output_dir != DEFAULT_OUTPUT_DIR:
        _remove_approval(DEFAULT_OUTPUT_DIR)
    report_path = output_dir / "audit-report.json"
    try:
        config = _load_config(
            config_path, enforce_production=enforce_production, approval_mode=approval_mode,
        )
        source_path = Path(config["examples_path"])
        if not source_path.is_file():
            report = {
                "version": VERSION, "approval_mode": config["approval_mode"],
                "status": "STOP", "approved": False,
                "failed_gates": ["canonical_examples_present", "rights_clearance", "human_review"],
                "pending_external_inputs": [
                    f"exactly {config['expected_examples']} canonical examples at {source_path}",
                    "per-example retained source/rights/prompt/raw-output bytes plus authoring, exposure, and preliminary-assessment provenance",
                    (
                        "real owner review of the deterministic packet and hash-bound owner verification"
                        if config["approval_mode"] == "single_owner_research_pilot"
                        else "two genuine independent reviews per example, required adjudication, and strict verification"
                    ),
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
            "version": VERSION, "approval_mode": config["approval_mode"], "approved": False,
            "examples_sha256": sha256_file(source_path),
            "prepared_sha256": sha256_file(output_dir / "prepared.jsonl"), "config_sha256": sha256_file(config_path),
            "schema_sha256": sha256_file(schema_path), "tokenizer_sha256": tokenizer_spec["sha256"],
            "production_contract_enforced": enforce_production,
            "pending_external_inputs": (
                ["real owner review of the selected/flagged packet", "hash-bound owner verification"]
                if config["approval_mode"] == "single_owner_research_pilot"
                else ["two independent reviews per example", "any required adjudication", "strict verification"]
            ),
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


def audit(
    config_path: Path, *, enforce_production: bool = True, approval_mode: str | None = None,
) -> dict[str, Any]:
    return prepare(config_path, enforce_production=enforce_production, approval_mode=approval_mode)


def select_owner_review_rows(
    prepared: Sequence[dict[str, Any]], *, target: int, seed: int,
) -> list[tuple[dict[str, Any], list[str]]]:
    """Select all flagged rows plus a deterministic sample covering every observed stratum."""
    require(0 < target <= len(prepared), "owner review target must fit the prepared corpus")
    by_id = {row["example_id"]: row for row in prepared}
    require(len(by_id) == len(prepared), "owner selection requires unique example IDs")
    reasons: defaultdict[str, set[str]] = defaultdict(set)

    for row in prepared:
        flags = row["preliminary_assessment"]["risk_flags"]
        for flag in flags:
            reasons[row["example_id"]].add(f"risk_flag:{flag}")
        if flags:
            reasons[row["example_id"]].add("risk_stratum:flagged")

    requirements: dict[str, list[dict[str, Any]]] = {}
    for field, values in (("genre", GENRES), ("primary_state_family", STATE_FAMILIES), ("split", SPLITS)):
        for value in values:
            requirements[f"{field}:{value}"] = [row for row in prepared if row[field] == value]
    for value in sorted({row["provenance"]["source_class"] for row in prepared}):
        requirements[f"source_class:{value}"] = [row for row in prepared if row["provenance"]["source_class"] == value]
    for value in sorted({row["provenance"]["pretraining_exposure"]["classification"] for row in prepared}):
        requirements[f"pretraining_exposure:{value}"] = [
            row for row in prepared if row["provenance"]["pretraining_exposure"]["classification"] == value
        ]
    clear = [row for row in prepared if not row["preliminary_assessment"]["risk_flags"]]
    if clear:
        requirements["risk_stratum:clear"] = clear
    lengths = {row["example_id"]: _word_count(row["opening"] + row["continuation"]) for row in prepared}
    shortest = min(lengths.values())
    longest = max(lengths.values())
    requirements["length:shortest"] = [row for row in prepared if lengths[row["example_id"]] == shortest]
    requirements["length:longest"] = [row for row in prepared if lengths[row["example_id"]] == longest]

    def rank(row: dict[str, Any], label: str) -> str:
        return hashlib.sha256(f"{seed}:{label}:{row['example_id']}:{row['content_sha256']}".encode()).hexdigest()

    for label, candidates in sorted(requirements.items()):
        require(bool(candidates), f"owner review stratum cannot be represented: {label}")
        already_selected = [row for row in candidates if row["example_id"] in reasons]
        chosen = min(already_selected or candidates, key=lambda row: rank(row, label))
        reasons[chosen["example_id"]].add(label)
    for row in sorted(prepared, key=lambda item: rank(item, "sample-fill")):
        if len(reasons) >= target:
            break
        reasons[row["example_id"]].add("deterministic_sample_fill")
    require(len(reasons) >= target, "owner packet could not reach its frozen sample target")
    return [
        (by_id[identifier], sorted(selection_reasons))
        for identifier, selection_reasons in sorted(reasons.items(), key=lambda item: rank(by_id[item[0]], "packet-order"))
    ]


def build_review_packet(
    prepared: Sequence[dict[str, Any]], config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Construct the complete deterministic packet and private split mapping."""
    if config["approval_mode"] == "single_owner_research_pilot":
        selected = select_owner_review_rows(
            prepared, target=int(config["owner_review"]["target_sample_size"]),
            seed=int(config["review_seed"]),
        )
    else:
        selected = [(row, ["strict_all_examples"]) for row in prepared]
    packet = []
    mapping: dict[str, dict[str, str]] = {}
    for row, selection_reasons in selected:
        review_id = "pilot-review-" + sha256_bytes(f"{row['example_id']}:{row['content_sha256']}".encode())[:20]
        packet.append({
            "review_id": review_id, "example_id": row["example_id"], "genre": row["genre"],
            "opening": row["opening"], "continuation": row["continuation"], "premise": row["premise"],
            "protagonist": row["protagonist"], "protagonist_motive": row["protagonist_motive"],
            "atomic_facts": row["atomic_facts"], "content_sha256": row["content_sha256"],
            "provenance": row["provenance"], "rights": row["rights"],
            "preliminary_assessment": row["preliminary_assessment"],
            "selection_reasons": selection_reasons,
            "reviewer_id": "",
            "reviewer_role": "owner" if config["approval_mode"] == "single_owner_research_pilot" else "original",
            "independent_review_confirmed": None, "genuine_human_attestation": None, "fact_labels": {},
            **{field: "" for field in REVIEW_FIELDS}, "evidence": {}, "reviewer_note": "",
        })
        mapping[review_id] = {
            "example_id": row["example_id"], "split": row["split"],
            "content_sha256": row["content_sha256"],
        }
    packet.sort(key=lambda row: hashlib.sha256(f"{config['review_seed']}:{row['review_id']}".encode()).hexdigest())
    return packet, mapping


def _review_key(
    *, config: dict[str, Any], audit_report: dict[str, Any], prepared_sha256: str,
    packet_sha256: str, mapping: dict[str, dict[str, str]],
) -> dict[str, Any]:
    return {
        "version": VERSION, "approval_mode": config["approval_mode"],
        "prepared_sha256": prepared_sha256, "packet_sha256": packet_sha256,
        "config_sha256": audit_report["config_sha256"],
        "schema_sha256": audit_report["schema_sha256"],
        "tokenizer_sha256": audit_report["tokenizer_sha256"],
        "reviews_must_bind_packet_sha256": True, "mapping": mapping,
        "warning": "Preliminary automated/AI-assisted diagnostics are not human review. IDs and attestations do not prove human identity or rights.",
    }


def export_review(
    config_path: Path, *, enforce_production: bool = True, approval_mode: str | None = None,
) -> dict[str, Any]:
    report = prepare(
        config_path, enforce_production=enforce_production, approval_mode=approval_mode,
    )
    if report["status"] != "PASS_AUTOMATED_AUDIT":
        return report
    config = _load_config(
        config_path, enforce_production=enforce_production, approval_mode=approval_mode,
    )
    output_dir = Path(config["output_dir"])
    prepared_path = output_dir / "prepared.jsonl"
    prepared = load_jsonl(prepared_path)
    packet, mapping = build_review_packet(prepared, config)
    packet_path, key_path = _review_artifact_paths(output_dir, config["approval_mode"])
    _write_jsonl_once(packet_path, packet)
    private_key = _review_key(
        config=config, audit_report=report, prepared_sha256=sha256_file(prepared_path),
        packet_sha256=sha256_file(packet_path), mapping=mapping,
    )
    _write_json_once(key_path, private_key)
    return {
        "status": "REVIEW_PENDING", "approved": False,
        "approval_mode": config["approval_mode"], "review_count": len(packet),
        "packet_sha256": private_key["packet_sha256"],
    }


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


def _validate_review(row: dict[str, Any], packet: dict[str, Any], approval_mode: str) -> None:
    mutable = {
        "reviewer_id", "reviewer_role", "independent_review_confirmed",
        "genuine_human_attestation", "fact_labels", "evidence", "reviewer_note", *REVIEW_FIELDS,
    }
    immutable = {key: value for key, value in packet.items() if key not in mutable}
    expected = {*packet.keys(), "packet_sha256"}
    require(set(row) == expected, f"{packet['review_id']}: review fields differ from frozen packet")
    require(all(row[key] == value for key, value in immutable.items()), f"{packet['review_id']}: review content changed")
    require(row["packet_sha256"] and re.fullmatch(r"[0-9a-f]{64}", row["packet_sha256"]) is not None, "review packet hash missing")
    require(bool(str(row["reviewer_id"]).strip()) and row["reviewer_id"] == row["reviewer_id"].strip(), "reviewer_id required")
    if approval_mode == "single_owner_research_pilot":
        require(row["reviewer_role"] == "owner", "owner mode requires reviewer_role owner")
        require(row["independent_review_confirmed"] is False, "owner review must not claim independence")
        require(row["genuine_human_attestation"] is True, "genuine-human owner attestation required")
    else:
        require(row["reviewer_role"] in {"original", "adjudicator"}, "invalid reviewer role")
        require(row["independent_review_confirmed"] is True and row["genuine_human_attestation"] is True, "independent genuine-human attestations required")
    require(set(row["fact_labels"]) == set(row["atomic_facts"]), "fact labels incomplete")
    require(set(row["fact_labels"].values()) <= FACT_LABELS, "invalid fact label")
    require(all(row[field] in REVIEW_LABELS for field in REVIEW_FIELDS), "invalid review label")
    evidence_fields = {*REVIEW_FIELDS, *row["atomic_facts"]}
    require(set(row["evidence"]) == evidence_fields and all(str(value).strip() for value in row["evidence"].values()), "review evidence incomplete")
    prose = row["opening"] + "\n" + row["continuation"]
    for field, evidence in row["evidence"].items():
        if field == "rights_and_provenance_acceptable":
            require(evidence == "<PROVENANCE_RECORD>", "rights/provenance evidence must be exactly <PROVENANCE_RECORD>")
        elif evidence == "<PROVENANCE_RECORD>":
            raise PilotValidationError("provenance sentinel is allowed only for rights/provenance review")
        elif evidence == "<ABSENT>":
            absence_is_appropriate = (
                (field in row["fact_labels"] and row["fact_labels"][field] != "retained")
                or (field in REVIEW_FIELDS and row[field] != "yes")
            )
            require(absence_is_appropriate, f"{field}: absence sentinel is inconsistent with a positive/retained judgment")
        else:
            require(evidence in prose, f"{field}: review evidence must quote prose verbatim")
            require(
                _word_count(evidence) >= EVIDENCE_MINIMUM_WORDS,
                f"{field}: review evidence quote must contain at least {EVIDENCE_MINIMUM_WORDS} normalized words",
            )
    require(bool(row["reviewer_note"].strip()), "reviewer note required")


def resolve_reviews(
    packet: list[dict[str, Any]], reviews: list[dict[str, Any]], packet_sha256: str,
    approval_mode: str = "strict_independent",
) -> list[dict[str, Any]]:
    by_id = {row["review_id"]: row for row in packet}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reviews:
        require(row.get("review_id") in by_id, f"unknown review_id: {row.get('review_id')}")
        _validate_review(row, by_id[row["review_id"]], approval_mode)
        require(row["packet_sha256"] == packet_sha256, "review packet hash mismatch; reviews are invalidated")
        grouped[row["review_id"]].append(row)
    resolved = []
    for identifier in by_id:
        if approval_mode == "single_owner_research_pilot":
            owner_reviews = grouped[identifier]
            require(len(owner_reviews) == 1, f"{identifier}: exactly one owner review required")
            resolved.append(owner_reviews[0])
            continue
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


def _validate_owner_verification(
    value: dict[str, Any], *, prepared_sha256: str, packet_sha256: str,
    owner_id: str, reviewed_ids: set[str], unreviewed_count: int,
) -> None:
    expected = {
        "version", "approval_mode", "prepared_sha256", "review_packet_sha256",
        "owner_id", "verified_at", "verification_evidence_reference",
        "genuine_human_attestation", "all_source_rights_and_provenance_examined",
        "all_example_identities_verified", "all_final_content_hash_linkages_verified",
        "distinctive_scorecard_fact_combinations_checked",
        "no_distinctive_scorecard_fact_combination_collision",
        "held_out_prompts_excluded_from_authoring_editing_and_candidate_selection",
        "no_unsealed_corpus_v3_candidate_bytes_or_artifacts_used_for_authoring_editing_or_candidate_selection",
        "corpus_v3_and_pretraining_exposure_overlap_audited_and_disclosed",
        "all_selected_and_flagged_examples_reviewed", "reviewed_example_ids",
        "unreviewed_unflagged_example_count", "unreviewed_unflagged_limitations_acknowledged",
        "automated_ai_diagnostics_are_not_human_review_acknowledged", "final_approval",
    }
    require(set(value) == expected, "owner verification fields differ from frozen contract")
    require(
        value["version"] == VERSION
        and value["approval_mode"] == "single_owner_research_pilot"
        and value["prepared_sha256"] == prepared_sha256
        and value["review_packet_sha256"] == packet_sha256,
        "owner verification hash or mode mismatch",
    )
    require(value["owner_id"] == owner_id and bool(value["verified_at"]) and bool(value["verification_evidence_reference"]), "owner identity, date, and evidence reference required")
    booleans = expected - {
        "version", "approval_mode", "prepared_sha256", "review_packet_sha256", "owner_id",
        "verified_at", "verification_evidence_reference", "reviewed_example_ids",
        "unreviewed_unflagged_example_count",
    }
    require(all(value[field] is True for field in booleans), "owner verification gate is incomplete")
    require(set(value["reviewed_example_ids"]) == reviewed_ids, "owner verification does not cover the review packet")
    require(value["unreviewed_unflagged_example_count"] == unreviewed_count, "owner verification has the wrong unreviewed count")


def _validate_strict_verification(value: dict[str, Any], *, prepared_sha256: str, packet_sha256: str, reviewer_ids: set[str], adjudicator_ids: set[str]) -> None:
    expected = {
        "version", "approval_mode", "prepared_sha256", "review_packet_sha256", "custodian_id", "verified_at",
        "verification_evidence_reference", "rights_evidence_examined", "provenance_evidence_examined",
        "all_examples_rights_cleared", "reviewer_identities_verified", "reviewers_genuine_humans",
        "independent_review_process_verified", "adjudicator_identities_verified",
        "distinctive_scorecard_fact_combinations_checked",
        "no_distinctive_scorecard_fact_combination_collision",
        "scene_post_prompts_excluded_from_authoring_editing_and_candidate_selection",
        "no_unsealed_corpus_v3_candidate_bytes_or_artifacts_used_for_authoring_editing_or_candidate_selection",
        "corpus_v3_and_pretraining_exposure_overlap_audited_and_disclosed",
        "verified_reviewer_ids", "verified_adjudicator_ids",
    }
    require(set(value) == expected, "strict verification fields differ from frozen contract")
    require(value["version"] == VERSION and value["approval_mode"] == "strict_independent" and value["prepared_sha256"] == prepared_sha256 and value["review_packet_sha256"] == packet_sha256, "strict verification hash or mode mismatch")
    require(all(bool(str(value[field]).strip()) for field in ("custodian_id", "verified_at", "verification_evidence_reference")), "strict verifier identity, date, and evidence reference required")
    booleans = expected - {"version", "approval_mode", "prepared_sha256", "review_packet_sha256", "custodian_id", "verified_at", "verification_evidence_reference", "verified_reviewer_ids", "verified_adjudicator_ids"}
    require(all(value[field] is True for field in booleans), "strict verification gate is incomplete")
    require(set(value["verified_reviewer_ids"]) == reviewer_ids, "strict verifier did not verify every original reviewer")
    require(set(value["verified_adjudicator_ids"]) == adjudicator_ids, "strict adjudicator verification mismatch")


def finalize(
    config_path: Path, *, enforce_production: bool = True, approval_mode: str | None = None,
) -> dict[str, Any]:
    selected_mode = approval_mode or "single_owner_research_pilot"
    output_dir = _raw_output_dir(config_path)
    _remove_approval(output_dir)
    if enforce_production and output_dir != DEFAULT_OUTPUT_DIR:
        _remove_approval(DEFAULT_OUTPUT_DIR)
    report_path = output_dir / "finalization-report.json"
    try:
        require(enforce_production, "test/helper configurations can never produce approval")
        config = _load_config(config_path, enforce_production=True, approval_mode=approval_mode)
        selected_mode = config["approval_mode"]
        audit_report = prepare(config_path, enforce_production=True, approval_mode=approval_mode)
        audit_failure = audit_report.get("failed_gates", [audit_report.get("error", "unknown audit failure")])
        require(audit_report["status"] == "PASS_AUTOMATED_AUDIT", f"automated audit has not passed: {audit_failure}")
        packet_path, key_path = _review_artifact_paths(output_dir, config["approval_mode"])
        require(packet_path.is_file() and key_path.is_file(), "review export is pending")
        prepared_path = output_dir / "prepared.jsonl"
        prepared = load_jsonl(prepared_path)
        prepared_sha256 = sha256_file(prepared_path)
        expected_packet, expected_mapping = build_review_packet(prepared, config)
        expected_packet_bytes = _jsonl_bytes(expected_packet)
        require(
            packet_path.read_bytes() == expected_packet_bytes,
            "review packet bytes, content, order, or deterministic selection differ from fresh reconstruction",
        )
        packet = load_jsonl(packet_path)
        packet_sha256 = sha256_bytes(expected_packet_bytes)
        expected_key = _review_key(
            config=config, audit_report=audit_report, prepared_sha256=prepared_sha256,
            packet_sha256=packet_sha256, mapping=expected_mapping,
        )
        require(
            key_path.read_bytes() == _json_bytes(expected_key),
            "review key bytes, mapping, or contract metadata differ from fresh reconstruction",
        )
        reviews_path = Path(config["reviews_path"])
        require(reviews_path.is_file(), f"completed reviews pending: {reviews_path}")
        reviews = load_jsonl(reviews_path)
        resolved = resolve_reviews(packet, reviews, packet_sha256, config["approval_mode"])
        require(resolved_reviews_pass_quality(resolved), "one or more examples failed final human quality approval")
        reviewed_ids = {row["example_id"] for row in resolved}
        all_ids = {row["example_id"] for row in prepared}
        common = {
            "version": VERSION, "status": "APPROVED", "approved": True,
            "approval_mode": config["approval_mode"], "example_count": len(prepared),
            "prepared_sha256": prepared_sha256,
            "review_packet_sha256": packet_sha256, "reviews_sha256": sha256_file(reviews_path),
            "tokenizer_v1_tokens_total": audit_report["gates"]["serialization_and_masking"]["tokenizer_v1_tokens_total"],
            "tokenizer_v1_tokens_by_split": audit_report["gates"]["serialization_and_masking"]["tokenizer_v1_tokens_by_split"],
            "examples_sha256": audit_report["examples_sha256"],
            "schema_sha256": audit_report["schema_sha256"],
            "config_sha256": audit_report["config_sha256"],
            "tokenizer_sha256": audit_report["tokenizer_sha256"],
            "unsealed_corpus_v3_candidate_bytes_or_artifacts_used": False,
            "corpus_v3_and_pretraining_exposure_overlap_audited_and_disclosed": True,
        }
        if config["approval_mode"] == "single_owner_research_pilot":
            owner_ids = {row["reviewer_id"] for row in resolved}
            require(len(owner_ids) == 1, "all owner reviews must be completed by one identified owner")
            verification_path = Path(config["owner_verification_path"])
            require(verification_path.is_file(), f"owner verification pending: {verification_path}")
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
            _validate_owner_verification(
                verification, prepared_sha256=prepared_sha256, packet_sha256=packet_sha256,
                owner_id=next(iter(owner_ids)), reviewed_ids=reviewed_ids,
                unreviewed_count=len(all_ids - reviewed_ids),
            )
            report = {
                **common,
                "independent_human_validation": False,
                "publication_grade_certification": False,
                "adjudication_performed": False,
                "directly_reviewed_count": len(reviewed_ids),
                "directly_reviewed_example_ids": sorted(reviewed_ids),
                "unreviewed_unflagged_count": len(all_ids - reviewed_ids),
                "owner_verification_sha256": sha256_file(verification_path),
                "automated_ai_diagnostic_limitation": "Preliminary automated or AI-assisted assessments are diagnostics, not human review. Unreviewed unflagged examples have no direct human quality judgment.",
                "scope_limitation": "Approval is limited to an exploratory single-owner research pilot and does not establish general corpus fitness.",
            }
        else:
            require(reviewed_ids == all_ids, "strict independent mode must review every example")
            reviewer_ids = {row["reviewer_id"] for row in reviews if row["reviewer_role"] == "original"}
            adjudicator_ids = {row["reviewer_id"] for row in reviews if row["reviewer_role"] == "adjudicator"}
            verification_path = Path(config["strict_verification_path"])
            require(verification_path.is_file(), f"strict verification pending: {verification_path}")
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
            _validate_strict_verification(
                verification, prepared_sha256=prepared_sha256, packet_sha256=packet_sha256,
                reviewer_ids=reviewer_ids, adjudicator_ids=adjudicator_ids,
            )
            report = {
                **common,
                "independent_human_validation": True,
                "publication_grade_certification": False,
                "directly_reviewed_count": len(reviewed_ids),
                "directly_reviewed_example_ids": sorted(reviewed_ids),
                "unreviewed_count": 0,
                "adjudicated_example_count": sum("resolution_provenance" in row for row in resolved),
                "strict_verification_sha256": sha256_file(verification_path),
                "scope_limitation": "Strict independent validation does not itself claim publication-grade certification.",
            }
        write_json_atomic(report_path, report)
        write_json_atomic(output_dir / APPROVAL_FILE, report)
        return report
    except Exception as error:
        report = {
            "version": VERSION, "approval_mode": selected_mode,
            "status": "STOP", "approved": False, "error": str(error),
            "pending_external_inputs": True,
        }
        write_json_atomic(report_path, report)
        _remove_approval(output_dir)
        if enforce_production and output_dir != DEFAULT_OUTPUT_DIR:
            _remove_approval(DEFAULT_OUTPUT_DIR)
        return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "audit", "export-review", "finalize"))
    parser.add_argument("--config", type=Path, default=Path("configs/continuation-pilot-v1.yaml"))
    parser.add_argument("--approval-mode", choices=sorted(APPROVAL_MODES))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    handlers = {"prepare": prepare, "audit": audit, "export-review": export_review, "finalize": finalize}
    report = handlers[args.command](args.config, approval_mode=args.approval_mode)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("status") == "STOP":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
