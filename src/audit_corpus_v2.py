"""Read-only, hash-grounded audit of sealed Corpus-v2 and its Data30M packing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import yaml
from tokenizers import Tokenizer

from src.tokenize_corpus import encode_document
from src.train_tokenizer import sha256_file, write_json_atomic


SPLITS = ("train", "validation", "test")
SPECIAL_CONTEXT_TOKENS = {"<|story|>", "<|bos|>"}
QUOTE_CHARS = {'"', "“", "”", "‘", "’"}
WORD_RE = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
CREATOR_ROLE_RE = re.compile(r"\s*\[([^\]]+)\]\s*$")
OCR_PATTERNS: dict[str, re.Pattern[str]] = {
    "replacement_character": re.compile("\ufffd"),
    "transcriber_or_editor_note": re.compile(r"\[(?:transcriber|editor)'?s? note|transcriber's note", re.I),
    "gutenberg_boilerplate": re.compile(r"project gutenberg|end of (?:the )?project gutenberg", re.I),
    "broken_line_hyphenation": re.compile(r"[A-Za-z]{2,}-\n[A-Za-z]{2,}"),
    "suspicious_markup_glyph": re.compile(r"[{}<>|]"),
    "digit_intrusion": re.compile(r"\b[A-Za-z]+\d+[A-Za-z\d]*\b|\b\d+[A-Za-z]+[A-Za-z\d]*\b"),
}


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object JSONL row at {path}:{line_number}")
            records.append(value)
    return records


def verify_locked_inputs(inputs: dict[str, dict[str, str]]) -> dict[str, str]:
    actual: dict[str, str] = {}
    for name, spec in inputs.items():
        path = Path(spec["path"])
        if path.is_absolute():
            raise ValueError(f"Tracked audit paths must be relative: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing locked audit input {name}: {path}")
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise RuntimeError(
                f"Locked audit input changed ({name}): {digest}, expected {spec['sha256']}"
            )
        actual[name] = digest
    return actual


def execution_provenance(config: dict[str, Any]) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return {
        "analyzed_base_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "generation_worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "dirty_state_caveat": (
            "The audit was generated from the named base commit plus the uncommitted issue-5 "
            "implementation. File hashes, not a future commit ID, ground generated content."
        ),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(package)
            for name, package in (
                ("numpy", "numpy"), ("tokenizers", "tokenizers"), ("pyyaml", "PyYAML")
            )
        },
        "reproduction_command": (
            f"python -m src.audit_corpus_v2 --config "
            f"{config.get('_config_path', 'configs/corpus-v2-audit.yaml')}"
        ),
        "recorded_validation": config.get("validation", {}),
    }


def nearest_rank(values: Sequence[int | float], percentile: float) -> int | float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * percentile) - 1)]


def distribution(values: Sequence[int | float]) -> dict[str, float | int]:
    return {
        "minimum": min(values),
        "p10": nearest_rank(values, 0.10),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p90": nearest_rank(values, 0.90),
        "p95": nearest_rank(values, 0.95),
        "p99": nearest_rank(values, 0.99),
        "maximum": max(values),
    }


def normalize_name(value: str | None) -> str:
    if not value:
        return "unknown"
    name = unicodedata.normalize("NFKC", value).casefold()
    name = re.sub(r",?\s*(?:ca?\.\s*)?\d{4}\s*[-–]\s*(?:\d{4})?\s*$", "", name)
    name = re.sub(r"[^\w]+", " ", name, flags=re.UNICODE)
    return " ".join(name.split()) or "unknown"


def parse_creator_string(value: str | None) -> dict[str, Any]:
    parts = [part.strip() for part in (value or "").split(";") if part.strip()]
    creators = []
    for part in parts:
        role_match = CREATOR_ROLE_RE.search(part)
        role = role_match.group(1).strip().casefold() if role_match else "unspecified"
        name = CREATOR_ROLE_RE.sub("", part).strip()
        creators.append({"name": name, "role": role})
    primary = creators[0] if creators else {"name": "unknown", "role": "unknown"}
    primary_author = (
        primary["name"] if primary["role"] in {"unspecified", "author"} else None
    )
    return {
        "creators": creators,
        "primary_creator": primary["name"],
        "primary_creator_role": primary["role"],
        "primary_author": primary_author,
        "contributors": creators[1:],
    }


def collection_key(record: dict[str, Any]) -> str:
    """Use stable Gutenberg IDs; seed records lack collection IDs, so scope normalized titles."""
    if record.get("source") == "project_gutenberg" and record.get("source_id"):
        return f"project_gutenberg:{record['source_id']}"
    title = unicodedata.normalize("NFKC", str(record.get("source_collection") or "unknown"))
    normalized = " ".join(re.sub(r"[^\w]+", " ", title.casefold()).split())
    return f"{record.get('source', 'unknown')}:{normalized}"


def ranked_concentration(
    records: list[dict[str, Any]], token_counts: dict[str, int], key, label
) -> dict[str, Any]:
    stories: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    labels: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        value = key(record)
        stories[value] += 1
        tokens[value] += token_counts[record["id"]]
        labels[value][str(label(record) or value)] += 1
    total_stories, total_tokens = len(records), sum(token_counts.values())
    ordered = sorted(tokens, key=lambda value: (-tokens[value], value))
    shares = [tokens[value] / total_tokens for value in ordered]
    return {
        "unique_count": len(ordered),
        "largest_story_share": max(stories.values()) / total_stories,
        "largest_token_share": shares[0],
        "top_10_token_share": sum(shares[:10]),
        "top_20_token_share": sum(shares[:20]),
        "token_herfindahl_index": sum(share * share for share in shares),
        "top_10": [
            {
                "key": value,
                "label": labels[value].most_common(1)[0][0],
                "stories": stories[value],
                "tokens": tokens[value],
                "token_share": tokens[value] / total_tokens,
            }
            for value in ordered[:10]
        ],
    }


def dialogue_metrics(text: str) -> dict[str, int | float]:
    words = WORD_RE.findall(text)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    dialogue_paragraphs = sum(part.startswith(('"', "“", "‘", "'")) for part in paragraphs)
    stack: list[str] = []
    outer_span: list[str] = []
    closed_span_words = 0
    paragraph_reopens = 0
    unmatched_closings = 0
    pairs = {"“": ("curly_double", True), "”": ("curly_double", False),
             "‘": ("curly_single", True), "’": ("curly_single", False)}
    for paragraph in paragraphs:
        first_nonspace = len(paragraph) - len(paragraph.lstrip())
        for index, character in enumerate(paragraph):
            was_inside = bool(stack)
            if character == '"':
                if stack and stack[-1] == "straight":
                    if index == first_nonspace:
                        paragraph_reopens += 1
                    else:
                        stack.pop()
                else:
                    stack.append("straight")
            elif character in pairs:
                quote_type, opening = pairs[character]
                if opening:
                    if stack and stack[-1] == quote_type and index == first_nonspace:
                        paragraph_reopens += 1
                    else:
                        stack.append(quote_type)
                elif stack and stack[-1] == quote_type:
                    stack.pop()
                elif not (character == "’" and index > 0 and paragraph[index - 1].isalnum()):
                    unmatched_closings += 1
            if was_inside and character not in QUOTE_CHARS:
                outer_span.append(character)
            if was_inside and not stack:
                closed_span_words += len(WORD_RE.findall("".join(outer_span)))
                outer_span.clear()
        if stack:
            outer_span.append("\n")
    quote_marks = sum(text.count(character) for character in QUOTE_CHARS)
    return {
        "words": len(words),
        "paragraphs": len(paragraphs),
        "dialogue_opening_paragraphs": dialogue_paragraphs,
        "dialogue_opening_paragraph_share": dialogue_paragraphs / max(1, len(paragraphs)),
        "quote_mark_count": quote_marks,
        "quote_marks_per_1000_words": 1000.0 * quote_marks / max(1, len(words)),
        "closed_quote_span_words": closed_span_words,
        "closed_quote_span_word_share": closed_span_words / max(1, len(words)),
        "paragraph_reopen_count": paragraph_reopens,
        "unclosed_quote_span_count": int(bool(stack)),
        "unmatched_closing_quote_count": unmatched_closings,
    }


def ocr_signals(text: str) -> dict[str, int]:
    return {name: len(pattern.findall(text)) for name, pattern in OCR_PATTERNS.items()}


def _inside_dialogue(prefix: str) -> bool:
    straight = prefix.count('"') % 2 == 1
    curly = prefix.rfind("“") > prefix.rfind("”")
    curly_single = prefix.rfind("‘") > prefix.rfind("’")
    return straight or curly or curly_single


def _bytelevel_decoder() -> dict[str, int]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = list(byte_values)
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {chr(codepoint): byte for byte, codepoint in zip(byte_values, codepoints, strict=True)}


BYTELEVEL_DECODER = _bytelevel_decoder()


def bytelevel_token_bytes(token: str) -> bytes | None:
    if token.startswith("<|") and token.endswith("|>"):
        return None
    try:
        return bytes(BYTELEVEL_DECODER[character] for character in token)
    except KeyError as error:
        raise ValueError(f"Token is not ByteLevel encoded: {token!r}") from error


def utf8_token_boundary(left_token: str, right_token: str) -> dict[str, bool]:
    return utf8_sequence_boundary([left_token], [right_token])


def utf8_sequence_boundary(
    left_tokens: Sequence[str], right_tokens: Sequence[str]
) -> dict[str, bool]:
    left_parts = [bytelevel_token_bytes(token) for token in left_tokens]
    right_parts = [bytelevel_token_bytes(token) for token in right_tokens]
    if any(part is None for part in left_parts + right_parts):
        return {
            "adjacent_byte_fragment": False,
            "byte_sequence_cut": False,
            "utf8_codepoint_cut": False,
        }
    left_bytes = b"".join(part for part in left_parts if part is not None)
    right_bytes = b"".join(part for part in right_parts if part is not None)
    adjacent_left = left_parts[-1] or b""
    adjacent_right = right_parts[0] or b""
    left_fragment = False
    right_fragment = False
    try:
        adjacent_left.decode("utf-8")
    except UnicodeDecodeError:
        left_fragment = True
    try:
        adjacent_right.decode("utf-8")
    except UnicodeDecodeError:
        right_fragment = True
    adjacent_byte_fragment = left_fragment or right_fragment
    utf8_codepoint_cut = False
    combined = left_bytes + right_bytes
    boundary = len(left_bytes)
    if adjacent_byte_fragment:
        for split_start in range(max(0, boundary - 3), boundary):
            for split_end in range(boundary + 1, min(len(combined), boundary + 4) + 1):
                candidate = combined[split_start:split_end]
                try:
                    decoded = candidate.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if len(decoded) == 1 and ord(decoded) > 127:
                    utf8_codepoint_cut = True
                    break
            if utf8_codepoint_cut:
                break
    return {
        "adjacent_byte_fragment": adjacent_byte_fragment,
        "byte_sequence_cut": utf8_codepoint_cut,
        "utf8_codepoint_cut": utf8_codepoint_cut,
    }


def classify_chunk_boundary(
    left: str,
    right: str,
    *,
    prefix: str = "",
    next_token: str = "",
    utf8_codepoint_cut: bool = False,
) -> dict[str, bool | str]:
    left_trimmed = left.rstrip(" \t")
    paragraph = re.search(r"\n\s*\n\s*$", left) is not None
    line = not paragraph and left_trimmed.endswith("\n")
    sentence = re.search(r"[.!?][\"'”’)]?\s*$", left) is not None
    dialogue = _inside_dialogue(prefix) or right.lstrip().startswith(tuple(QUOTE_CHARS))
    subword = bool(
        left and right and left[-1].isalnum() and right[0].isalnum()
        and not next_token.startswith(("Ġ", "Ċ"))
    )
    if utf8_codepoint_cut:
        primary = "utf8_codepoint"
    elif paragraph:
        primary = "paragraph"
    elif line:
        primary = "line"
    elif sentence:
        primary = "sentence"
    elif dialogue:
        primary = "dialogue"
    elif subword:
        primary = "subword"
    else:
        primary = "token"
    return {
        "primary": primary,
        "paragraph": paragraph,
        "line": line,
        "sentence": sentence,
        "dialogue": dialogue,
        "subword": subword,
        "utf8_codepoint": utf8_codepoint_cut,
    }


def mandatory_sample_strata(
    records: list[dict[str, Any]], metrics: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        genres = record.get("broad_genres") or record.get("genres") or ["unclassified"]
        for genre in genres:
            strata[f"genre:{genre}"].append(record)
        dialogue = metrics[record["id"]]["dialogue"]["dialogue_opening_paragraph_share"]
        strata["dialogue:high" if dialogue >= 0.20 else "dialogue:low"].append(record)
        has_ocr = any(metrics[record["id"]]["ocr"].values())
        strata["ocr:signal" if has_ocr else "ocr:clean"].append(record)
        strata[f"source:{record['source']}"].append(record)
        boundary_label = f"boundary_method:{record.get('boundary_method', 'unknown')}"
        strata[boundary_label].append(record)
    ordered_lengths = sorted(int(record["words"]) for record in records)
    short_cutoff = nearest_rank(ordered_lengths, 0.10)
    long_cutoff = nearest_rank(ordered_lengths, 0.90)
    for record in records:
        if record["words"] <= short_cutoff:
            strata["length:shortest_decile"].append(record)
        if record["words"] >= long_cutoff:
            strata["length:longest_decile"].append(record)
    empty = sorted(label for label, candidates in strata.items() if not candidates)
    if empty:
        raise ValueError(f"Mandatory sample strata have no candidates: {empty}")
    return dict(strata)


def deterministic_sample(
    records: list[dict[str, Any]], metrics: dict[str, dict[str, Any]], *, seed: int, count: int
) -> list[dict[str, Any]]:
    strata = mandatory_sample_strata(records, metrics)
    ordered_records = sorted(
        records,
        key=lambda record: hashlib.sha256(f"{seed}:{record['id']}".encode()).hexdigest(),
    )
    selected: dict[str, set[str]] = defaultdict(set)
    selected_ids: set[str] = set()
    for label in sorted(strata):
        candidate = min(
            strata[label],
            key=lambda record: hashlib.sha256(f"{seed}:{label}:{record['id']}".encode()).hexdigest(),
        )
        selected[candidate["id"]].add(label)
        selected_ids.add(candidate["id"])
    if len(selected_ids) > count:
        raise ValueError(
            f"sample_count={count} cannot cover {len(selected_ids)} unique mandatory-stratum selections"
        )
    for record in ordered_records:
        if len(selected_ids) >= count:
            break
        selected_ids.add(record["id"])
        selected[record["id"]].add("deterministic_fill")
    by_id = {record["id"]: record for record in records}
    return [
        {
            "id": record_id,
            "strata": sorted(selected[record_id]),
            "title": by_id[record_id]["title"],
            "author": by_id[record_id].get("author"),
            "source": by_id[record_id]["source"],
            "collection_key": collection_key(by_id[record_id]),
            "words": by_id[record_id]["words"],
            "genres": by_id[record_id].get("broad_genres") or by_id[record_id].get("genres") or [],
            "opening_excerpt": by_id[record_id]["text"][:240],
            "ending_excerpt": by_id[record_id]["text"][-240:],
            "automated_dialogue": metrics[record_id]["dialogue"],
            "automated_ocr_signals": metrics[record_id]["ocr"],
        }
        for record_id in sorted(selected_ids)
    ]


def validate_manual_review(path: Path, sample_ids: set[str]) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "pending", "reviewed": 0, "required": len(sample_ids)}
    payload = _json(path)
    reviews = payload.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("Manual review must contain a reviews list")
    ids = [item.get("id") for item in reviews]
    if len(ids) != len(set(ids)) or set(ids) != sample_ids:
        raise ValueError("Manual review IDs must match the deterministic sample exactly")
    allowed = {"pass", "concern", "fail"}
    for item in reviews:
        for field in ("story_boundary", "ocr_quality", "prose_quality"):
            if item.get(field) not in allowed:
                raise ValueError(f"Invalid {field} review for {item.get('id')}")
        if not isinstance(item.get("notes"), str):
            raise ValueError(f"Missing review notes for {item.get('id')}")
    return {
        "status": "complete",
        "reviewed": len(reviews),
        "required": len(sample_ids),
        "counts": {
            field: dict(sorted(Counter(item[field] for item in reviews).items()))
            for field in ("story_boundary", "ocr_quality", "prose_quality")
        },
        "review_sha256": sha256_file(path),
        "review_path": str(path),
    }


def validate_metadata_linkage(
    config: dict[str, Any], metadata: dict[str, Any], split_payload: dict[str, Any]
) -> None:
    inputs = config["inputs"]
    expected_top_level = {
        "corpus_path": inputs["corpus"]["path"],
        "corpus_sha256": inputs["corpus"]["sha256"],
        "split_path": inputs["split"]["path"],
        "split_sha256": inputs["split"]["sha256"],
        "tokenizer_path": inputs["tokenizer"]["path"],
        "tokenizer_hash": inputs["tokenizer"]["sha256"],
        "dtype": "uint16",
    }
    for field, expected in expected_top_level.items():
        if metadata.get(field) != expected:
            raise RuntimeError(f"Packed metadata {field} does not match the locked input")
    if split_payload.get("corpus_sha256") != inputs["corpus"]["sha256"]:
        raise RuntimeError("Split payload does not identify the locked corpus")
    if split_payload.get("corpus_path") != inputs["corpus"]["path"]:
        raise RuntimeError("Split payload corpus_path does not match the locked corpus")
    for split in SPLITS:
        packed = metadata.get("splits", {}).get(split)
        if not isinstance(packed, dict):
            raise RuntimeError(f"Packed metadata is missing split {split}")
        for kind in ("bin", "index"):
            input_name = f"{split}_{kind}"
            expected_path = inputs[input_name]["path"]
            expected_hash = inputs[input_name]["sha256"]
            if packed.get(f"{kind}_path") != expected_path:
                raise RuntimeError(f"Packed metadata {input_name} path mismatch")
            if packed.get(f"{kind}_sha256") != expected_hash:
                raise RuntimeError(f"Packed metadata {input_name} hash mismatch")
            if sha256_file(Path(expected_path)) != expected_hash:
                raise RuntimeError(f"Packed {input_name} bytes do not match configured hash")


def validate_packed(
    config: dict[str, Any], records: list[dict[str, Any]], tokenizer: Tokenizer
) -> tuple[dict[str, int], dict[str, Any]]:
    inputs = config["inputs"]
    metadata = _json(Path(inputs["packed_metadata"]["path"]))
    split_payload = _json(Path(inputs["split"]["path"]))
    validate_metadata_linkage(config, metadata, split_payload)
    assignments = split_payload["assignments"]
    record_ids = {record["id"] for record in records}
    records_by_id = {record["id"]: record for record in records}
    if set(assignments) != record_ids:
        raise RuntimeError("Split assignments do not cover the canonical corpus exactly")
    sequence_length = int(config["sequence_length"])
    token_counts: dict[str, int] = {}
    boundary_counts: Counter[str] = Counter()
    feature_counts: Counter[str] = Counter()
    total_boundaries = 0
    continuation_without_controls = 0
    exact_reencoded_documents = 0
    utf8_codepoint_cuts = 0
    byte_sequence_cuts = 0
    adjacent_byte_fragment_boundaries = 0
    per_split: dict[str, Any] = {}
    story_id = tokenizer.token_to_id("<|story|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    context_ids = {tokenizer.token_to_id(token) for token in SPECIAL_CONTEXT_TOKENS}
    context_ids.update(
        token_id for token_id in range(tokenizer.get_vocab_size())
        if (tokenizer.id_to_token(token_id) or "").startswith("<|")
    )
    for split in SPLITS:
        index = _json(Path(inputs[f"{split}_index"]["path"]))
        documents = index.get("documents")
        if index.get("split") != split or not isinstance(documents, list):
            raise RuntimeError(f"Invalid {split} index schema")
        bin_path = Path(inputs[f"{split}_bin"]["path"])
        tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        expected_offset = 0
        sequence_count = 0
        split_boundaries = 0
        for document in documents:
            record_id = document["id"]
            offset, length = int(document["offset"]), int(document["length"])
            if record_id not in record_ids or assignments[record_id] != split:
                raise RuntimeError(f"Packed document {record_id} has an invalid split")
            if offset != expected_offset or length < 2 or offset + length > len(tokens):
                raise RuntimeError(f"Invalid packed extent for {record_id}")
            ids = [int(value) for value in tokens[offset : offset + length]]
            record = records_by_id[record_id]
            expected_ids, expected_control = encode_document(tokenizer, record)
            if ids != expected_ids:
                raise RuntimeError(f"Packed tokens differ from canonical encode_document for {record_id}")
            exact_reencoded_documents += 1
            if (
                int(document.get("words", -1)) != int(record["words"])
                or document.get("genre_control_token") != expected_control
                or length != len(expected_ids)
            ):
                raise RuntimeError(f"Packed index metadata is misattributed for {record_id}")
            if ids[0] != story_id or ids[-1] != eos_id:
                raise RuntimeError(f"Packed controls are invalid for {record_id}")
            expected_sequences = math.ceil((length - 1) / sequence_length)
            if int(document["sequence_count"]) != expected_sequences:
                raise RuntimeError(f"Incorrect sequence_count for {record_id}")
            token_counts[record_id] = length
            sequence_count += expected_sequences
            canonical_content = f"{record['title']}\n\n{record['text']}"
            content_encoding = tokenizer.encode(canonical_content)
            prefix_length = len(expected_ids) - len(content_encoding.ids) - 1
            for boundary in range(sequence_length, length - 1, sequence_length):
                content_index = boundary - prefix_length
                if not 0 <= content_index < len(content_encoding.offsets):
                    raise RuntimeError(f"Boundary offset is outside canonical content for {record_id}")
                character_offset = content_encoding.offsets[content_index][0]
                left = canonical_content[max(0, character_offset - 256) : character_offset]
                right = canonical_content[character_offset : character_offset + 256]
                prefix = canonical_content[:character_offset]
                next_token = tokenizer.id_to_token(ids[boundary]) or ""
                left_tokens = [
                    tokenizer.id_to_token(token_id) or ""
                    for token_id in ids[max(0, boundary - 4) : boundary]
                ]
                right_tokens = [
                    tokenizer.id_to_token(token_id) or ""
                    for token_id in ids[boundary : min(length, boundary + 4)]
                ]
                byte_finding = utf8_sequence_boundary(left_tokens, right_tokens)
                finding = classify_chunk_boundary(
                    left,
                    right,
                    prefix=prefix,
                    next_token=next_token,
                    utf8_codepoint_cut=byte_finding["utf8_codepoint_cut"],
                )
                boundary_counts[str(finding["primary"])] += 1
                for feature in (
                    "paragraph", "line", "sentence", "dialogue", "subword", "utf8_codepoint"
                ):
                    feature_counts[feature] += int(bool(finding[feature]))
                byte_sequence_cuts += int(byte_finding["byte_sequence_cut"])
                adjacent_byte_fragment_boundaries += int(byte_finding["adjacent_byte_fragment"])
                utf8_codepoint_cuts += int(byte_finding["utf8_codepoint_cut"])
                continuation_without_controls += int(ids[boundary] not in context_ids)
                total_boundaries += 1
                split_boundaries += 1
            expected_offset += length
        if expected_offset != len(tokens):
            raise RuntimeError(f"{split} index does not cover its binary exactly")
        packed = metadata["splits"][split]
        if (
            packed["document_count"] != len(documents)
            or packed["token_count"] != len(tokens)
            or packed["sequence_count"] != sequence_count
        ):
            raise RuntimeError(f"{split} packed metadata disagrees with bin/index")
        per_split[split] = {
            "documents": len(documents), "tokens": len(tokens),
            "sequences": sequence_count, "internal_boundaries": split_boundaries,
        }
    if set(token_counts) != record_ids:
        raise RuntimeError("Packed indexes do not cover canonical record IDs exactly")
    boundary_report = {
        "method": (
            "Every internal loader cut at relative token offsets 1024*n was read from the "
            "existing uint16 bins after exact canonical encode_document comparison. Structural "
            "features use canonical text and tokenizer offsets; no partial token side is decoded. "
            "Exact ByteLevel vocabulary bytes identify byte fragments and UTF-8 codepoint cuts. "
            "Primary precedence is utf8_codepoint>paragraph>line>sentence>dialogue>subword>token."
        ),
        "sequence_length": sequence_length,
        "internal_boundary_count": total_boundaries,
        "primary_counts": dict(sorted(boundary_counts.items())),
        "primary_percentages": {
            key: 100.0 * value / max(1, total_boundaries)
            for key, value in sorted(boundary_counts.items())
        },
        "overlapping_feature_counts": dict(sorted(feature_counts.items())),
        "byte_sequence_cut_count": byte_sequence_cuts,
        "utf8_codepoint_cut_count": utf8_codepoint_cuts,
        "boundaries_adjacent_to_individually_invalid_utf8_token_fragments": (
            adjacent_byte_fragment_boundaries
        ),
        "exact_canonical_reencoded_document_count": exact_reencoded_documents,
        "continuation_chunks_without_repeated_control_context": continuation_without_controls,
        "continuation_chunks_without_repeated_control_context_percentage": (
            100.0 * continuation_without_controls / max(1, total_boundaries)
        ),
        "confidence": {
            "utf8_codepoint_and_byte_sequence": "exact from reversible ByteLevel token bytes",
            "paragraph_line_sentence": "high from canonical text at tokenizer-provided offsets",
            "subword": "medium; inferred from ByteLevel word-start markers and canonical adjacent text",
            "dialogue": "medium; quote-state heuristic, not semantic speech attribution",
        },
        "splits": per_split,
    }
    return token_counts, boundary_report


def audit(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Audit config must be a mapping")
    config["_config_path"] = str(config_path)
    provenance = execution_provenance(config)
    locked_hashes = verify_locked_inputs(config["inputs"])
    before = dict(locked_hashes)
    seal = _json(Path(config["inputs"]["acquisition_seal"]["path"]))
    if seal.get("status") != "sealed" or seal.get("corpus_sha256") != locked_hashes["corpus"]:
        raise RuntimeError("Acquisition seal does not identify the configured corpus")
    records = load_jsonl(Path(config["inputs"]["corpus"]["path"]))
    if len(records) != seal.get("story_count"):
        raise RuntimeError("Corpus record count disagrees with acquisition seal")
    tokenizer = Tokenizer.from_file(config["inputs"]["tokenizer"]["path"])
    token_counts, boundaries = validate_packed(config, records, tokenizer)

    per_record: dict[str, dict[str, Any]] = {}
    ocr_totals: Counter[str] = Counter()
    ocr_documents: Counter[str] = Counter()
    closed_quote_span_word_total = 0
    quote_mark_total = 0
    prose_word_total = 0
    paragraph_total = 0
    dialogue_opening_paragraph_total = 0
    for record in records:
        dialogue = dialogue_metrics(record["text"])
        ocr = ocr_signals(record["text"])
        per_record[record["id"]] = {"dialogue": dialogue, "ocr": ocr}
        closed_quote_span_word_total += int(dialogue["closed_quote_span_words"])
        quote_mark_total += int(dialogue["quote_mark_count"])
        prose_word_total += int(dialogue["words"])
        paragraph_total += int(dialogue["paragraphs"])
        dialogue_opening_paragraph_total += int(dialogue["dialogue_opening_paragraphs"])
        for signal, count in ocr.items():
            ocr_totals[signal] += count
            ocr_documents[signal] += int(count > 0)

    creator_metadata = {record["id"]: parse_creator_string(record.get("author")) for record in records}
    primary_author_records = [
        record for record in records if creator_metadata[record["id"]]["primary_author"]
    ]
    primary_author_forms: defaultdict[str, set[str]] = defaultdict(set)
    role_counts: Counter[str] = Counter()
    records_with_contributors = 0
    for record in records:
        parsed = creator_metadata[record["id"]]
        primary_author = parsed["primary_author"]
        if primary_author:
            primary_author_forms[normalize_name(primary_author)].add(primary_author)
        records_with_contributors += int(bool(parsed["contributors"]))
        for creator in parsed["creators"]:
            role_counts[creator["role"]] += 1
    alias_groups = [
        {"canonical_key": key, "forms": sorted(forms)}
        for key, forms in sorted(primary_author_forms.items()) if len(forms) > 1
    ]
    primary_author = ranked_concentration(
        primary_author_records,
        {record["id"]: token_counts[record["id"]] for record in primary_author_records},
        lambda record: normalize_name(creator_metadata[record["id"]]["primary_author"]),
        lambda record: creator_metadata[record["id"]]["primary_author"],
    )
    primary_author.update(
        {
            "definition": (
                "First semicolon-delimited creator only when its bracketed role is absent or Author; "
                "normalized keys are alias candidates, not verified identities."
            ),
            "covered_story_count": len(primary_author_records),
            "covered_token_count": sum(token_counts[record["id"]] for record in primary_author_records),
            "unknown_or_non_author_primary_story_count": len(records) - len(primary_author_records),
            "raw_primary_author_form_count": len(
                {creator_metadata[record["id"]]["primary_author"] for record in primary_author_records}
            ),
            "normalized_primary_author_key_count": len(primary_author_forms),
            "alias_group_count": len(alias_groups),
            "alias_candidate_groups": alias_groups,
            "normalization_limit": "Case/punctuation normalization plus terminal lifespan stripping only; no identity resolution.",
        }
    )
    raw_creator_strings = {str(record.get("author") or "unknown") for record in records}
    normalized_creator_strings = {normalize_name(record.get("author")) for record in records}
    creator_fields = {
        "schema_note": (
            "The corpus author field is a Project Gutenberg-style creator string, often "
            "semicolon-delimited and role-annotated; it is not an individual-author ID."
        ),
        "raw_creator_string_count": len(raw_creator_strings),
        "normalized_whole_creator_string_count": len(normalized_creator_strings),
        "records_with_additional_contributors": records_with_contributors,
        "creator_role_occurrences": dict(sorted(role_counts.items())),
    }
    collection = ranked_concentration(
        records,
        token_counts,
        collection_key,
        lambda record: record.get("source_collection") or "unknown",
    )
    collection["stable_key_method"] = (
        "source-scoped Gutenberg source_id; source-scoped normalized source_collection title "
        "for seed records whose source_id is story-level"
    )

    genre_stories: Counter[str] = Counter()
    genre_tokens: Counter[str] = Counter()
    genre_memberships = 0
    for record in records:
        genres = record.get("broad_genres") or record.get("genres") or ["unclassified"]
        genre_memberships += len(genres)
        for genre in genres:
            genre_stories[genre] += 1
            genre_tokens[genre] += token_counts[record["id"]]
    total_tokens = sum(token_counts.values())
    genres = {
        "accounting": "Multilabel: each label receives the full story/token weight; shares may sum above 100%.",
        "membership_count": genre_memberships,
        "labels": {
            genre: {
                "stories": genre_stories[genre],
                "story_share": genre_stories[genre] / len(records),
                "tokens": genre_tokens[genre],
                "token_share": genre_tokens[genre] / total_tokens,
            }
            for genre in sorted(genre_stories)
        },
    }

    years = [record.get("publication_year") for record in records]
    known_years = [year for year in years if isinstance(year, int)]
    ebook_dates = [
        record.get("provenance", {}).get("catalog_ebook_issued")
        for record in records
        if record.get("provenance", {}).get("catalog_ebook_issued")
    ]
    period = {
        "reliably_known_publication_year_count": len(known_years),
        "unknown_publication_year_count": len(records) - len(known_years),
        "unknown_publication_year_percentage": 100.0 * (len(records) - len(known_years)) / len(records),
        "publication_year_range": [min(known_years), max(known_years)] if known_years else None,
        "catalog_ebook_issue_date_count_reported_separately": len(ebook_dates),
        "catalog_ebook_issue_date_note": "Project Gutenberg ebook issue dates are not original publication dates and are excluded from period estimates.",
    }

    word_lengths = [int(record["words"]) for record in records]
    packed_lengths = [token_counts[record["id"]] for record in records]
    length_report = {
        "words": distribution(word_lengths),
        "packed_tokens": distribution(packed_lengths),
        "word_bands": {
            "under_500": sum(value < 500 for value in word_lengths),
            "500_1999": sum(500 <= value < 2000 for value in word_lengths),
            "2000_4999": sum(2000 <= value < 5000 for value in word_lengths),
            "5000_9999": sum(5000 <= value < 10000 for value in word_lengths),
            "10000_plus": sum(value >= 10000 for value in word_lengths),
        },
        "documents_requiring_continuation_chunks": sum(value > config["sequence_length"] + 1 for value in packed_lengths),
    }
    opening_shares = [
        float(per_record[record["id"]]["dialogue"]["dialogue_opening_paragraph_share"])
        for record in records
    ]
    closed_span_shares = [
        float(per_record[record["id"]]["dialogue"]["closed_quote_span_word_share"])
        for record in records
    ]
    quote_densities = [
        float(per_record[record["id"]]["dialogue"]["quote_marks_per_1000_words"])
        for record in records
    ]
    dialogue_report = {
        "definition": (
            "Independent style proxies: paragraph openings with quote marks; quote-mark density; "
            "and words in fully closed straight/curly spans. Reopened multi-paragraph quotations "
            "remain open, nested curly quotes are stacked, and unclosed outer spans are excluded. "
            "None is semantic speaker attribution."
        ),
        "corpus_dialogue_opening_paragraph_share": (
            dialogue_opening_paragraph_total / max(1, paragraph_total)
        ),
        "story_dialogue_opening_paragraph_share_distribution": distribution(opening_shares),
        "corpus_quote_marks_per_1000_words": 1000.0 * quote_mark_total / max(1, prose_word_total),
        "story_quote_marks_per_1000_words_distribution": distribution(quote_densities),
        "corpus_closed_quote_span_word_share": (
            closed_quote_span_word_total / max(1, prose_word_total)
        ),
        "story_closed_quote_span_word_share_distribution": distribution(closed_span_shares),
        "stories_with_no_dialogue_opening_paragraph": sum(value == 0 for value in opening_shares),
        "stories_with_unclosed_quote_span": sum(
            int(per_record[record["id"]]["dialogue"]["unclosed_quote_span_count"] > 0)
            for record in records
        ),
        "paragraph_reopen_count": sum(
            int(per_record[record["id"]]["dialogue"]["paragraph_reopen_count"])
            for record in records
        ),
    }
    records_with_any_ocr = sum(any(per_record[record["id"]]["ocr"].values()) for record in records)
    ocr_report = {
        "method": "Conservative accepted-text pattern signals; counts are review leads, not confirmed OCR errors.",
        "records_with_any_signal": records_with_any_ocr,
        "records_with_any_signal_percentage": 100.0 * records_with_any_ocr / len(records),
        "signal_occurrences": dict(sorted(ocr_totals.items())),
        "documents_by_signal": dict(sorted(ocr_documents.items())),
    }

    rights_status = Counter(str(record.get("rights_status") or record.get("rights") or "missing") for record in records)
    rights_basis_count = sum(bool(record.get("provenance", {}).get("rights_basis")) for record in records)
    raw_hash_count = sum(bool(record.get("provenance", {}).get("raw_sha256")) for record in records)
    source_url_count = sum(bool(record.get("source_url") or record.get("provenance", {}).get("source_url")) for record in records)
    rights = {
        "status_counts": dict(sorted(rights_status.items())),
        "allowed_status_count": sum(rights_status[value] for value in ("public_domain", "public_domain_us")),
        "explicit_rights_basis_count": rights_basis_count,
        "explicit_rights_basis_percentage": 100.0 * rights_basis_count / len(records),
        "raw_source_hash_count": raw_hash_count,
        "source_url_count": source_url_count,
        "evidence_complete_count": sum(
            bool(record.get("provenance", {}).get("rights_basis"))
            and bool(record.get("provenance", {}).get("raw_sha256"))
            and bool(record.get("source_url") or record.get("provenance", {}).get("source_url"))
            for record in records
        ),
    }

    exact_hashes = Counter(record["text_hash"] for record in records)
    normalized_texts: Counter[str] = Counter()
    title_author: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    openings: defaultdict[str, list[str]] = defaultdict(list)
    for record in records:
        normalized = " ".join(WORD_RE.findall(record["text"].casefold()))
        normalized_texts[hashlib.sha256(normalized.encode()).hexdigest()] += 1
        title_key = " ".join(WORD_RE.findall(record["title"].casefold()))
        parsed = creator_metadata[record["id"]]
        attribution = parsed["primary_author"] or record.get("author")
        title_author[(title_key, normalize_name(attribution))].append(record["id"])
        opening = " ".join(WORD_RE.findall(record["text"].casefold())[:40])
        openings[hashlib.sha256(opening.encode()).hexdigest()].append(record["id"])
    duplicates = {
        "exact_text_hash_duplicate_groups": sum(count > 1 for count in exact_hashes.values()),
        "normalized_alphanumeric_text_duplicate_groups": sum(count > 1 for count in normalized_texts.values()),
        "normalized_title_primary_author_or_creator_groups_with_multiple_records": sum(
            len(ids) > 1 for ids in title_author.values()
        ),
        "forty_word_opening_groups_with_multiple_records": sum(len(ids) > 1 for ids in openings.values()),
        "forty_word_opening_records_in_groups": sum(len(ids) for ids in openings.values() if len(ids) > 1),
        "limitation": "Title/opening collisions are candidate patterns only. This audit does not rerun all-pairs fuzzy duplicate review; the acquisition seal records the completed 155-cluster review and 162 exclusions.",
        "sealed_review": {
            "unresolved_near_duplicate_clusters": seal.get("unresolved_near_duplicate_clusters"),
            "documented_seed_correctness_exclusions": seal.get("documented_seed_correctness_exclusions"),
        },
    }

    samples = deterministic_sample(
        records, per_record, seed=int(config["seed"]), count=int(config["sample_count"])
    )
    mandatory_labels = sorted(mandatory_sample_strata(records, per_record))
    covered_labels = sorted(
        {label for sample in samples for label in sample["strata"] if label != "deterministic_fill"}
    )
    uncovered_labels = sorted(set(mandatory_labels) - set(covered_labels))
    if uncovered_labels:
        raise RuntimeError(f"Deterministic sample missed mandatory strata: {uncovered_labels}")
    output_dir = Path(config["output_dir"])
    if output_dir.is_absolute():
        raise ValueError("Tracked audit output path must be relative")
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "representative-samples.json"
    sample_coverage = {
        "mandatory_strata": mandatory_labels,
        "covered_strata": covered_labels,
        "uncovered_strata": uncovered_labels,
    }
    write_json_atomic(
        samples_path, {"seed": config["seed"], "coverage": sample_coverage, "samples": samples}
    )
    manual_review = validate_manual_review(Path(config["manual_review_path"]), {item["id"] for item in samples})

    priorities = [
        {
            "priority": "P0", "issue": "Historical-period metadata is absent",
            "evidence": f"{period['unknown_publication_year_percentage']:.1f}% of publication years are unknown.",
        },
        {
            "priority": "P0", "issue": "Genre metadata leaves most token mass unclassified",
            "evidence": f"Unclassified token share is {genres['labels']['unclassified']['token_share']:.1%}.",
        },
        {
            "priority": "P1", "issue": "Continuation chunks lose document control context",
            "evidence": f"{boundaries['continuation_chunks_without_repeated_control_context_percentage']:.1f}% of internal 1024-token chunks start without repeated controls.",
        },
        {
            "priority": "P1", "issue": "Rights assertions are stronger than row-level evidence for inherited seed records",
            "evidence": f"Explicit rights basis exists for {rights['explicit_rights_basis_percentage']:.1f}% of records.",
        },
        {
            "priority": "P1", "issue": "Accepted text retains OCR/editorial review signals",
            "evidence": f"{ocr_report['records_with_any_signal_percentage']:.1f}% of stories trigger at least one conservative signal, including {ocr_documents['gutenberg_boilerplate']} with Gutenberg boilerplate.",
        },
        {
            "priority": "P2", "issue": "Residual duplicate-pattern candidates need review on admission",
            "evidence": f"{duplicates['forty_word_opening_records_in_groups']} records share a normalized 40-word opening with another record.",
        },
    ]
    admission_rules = [
        "Require source-scoped stable creator IDs, explicit creator roles, a defensible primary-author field, and stable collection IDs; calculate story- and token-weighted caps before admission.",
        "Require evidence-backed original publication year or explicit unknown; never substitute ebook issue date. Report unknown token share.",
        "Require provenance-backed multilabel genre or explicit unclassified; cap unclassified tokens and audit each genre by stories and tokens.",
        "Require row-level rights status, rights basis, source URL, and immutable raw-source hash; quarantine incomplete inherited evidence.",
        "Reject exact normalized text duplicates and human-review title/author, opening, and fuzzy near-duplicate clusters before splitting.",
        "Reject replacement characters and Gutenberg boilerplate; review transcriber notes, digit intrusions, markup glyphs, and line-hyphen signals in accepted text.",
        "Require high-confidence story boundaries and review shortest/longest, OCR-signal, dialogue-extreme, genre, source, and collection strata deterministically.",
        "Audit actual packed cuts at the configured context. Prefer paragraph/sentence-aware packing or repeat explicit continuation/document controls, then lock packed hashes.",
        "Split only after collection and duplicate clustering; keep every cluster in one split and re-run leakage checks.",
    ]
    report = {
        "audit_id": "fictionpulper-corpus-v2-audit-v1",
        "output_path": str(output_dir / "audit-report.json"),
        "status": "complete" if manual_review["status"] == "complete" else "manual_review_pending",
        "provenance": provenance,
        "scope": {
            "acquisition_artifact": "sealed Corpus-v2; never modified",
            "post_seal_artifacts": "Data30M split and packed bins/indexes are not part of the acquisition seal and are independently hash-locked by this audit config.",
        },
        "inputs": {name: {"path": spec["path"], "sha256": locked_hashes[name]} for name, spec in config["inputs"].items()},
        "counts": {"stories": len(records), "packed_tokens": total_tokens},
        "primary_author_concentration": primary_author,
        "creator_string_metadata": creator_fields,
        "collection_concentration": collection,
        "genre": genres,
        "historical_period": period,
        "story_lengths": length_report,
        "dialogue": dialogue_report,
        "ocr": ocr_report,
        "rights": rights,
        "duplicates": duplicates,
        "chunk_boundaries_1024": boundaries,
        "representative_sample": {
            "selection": "Deterministic hash-ranked mandatory coverage of every observed genre, source, boundary method, dialogue-opening extreme, OCR-signal/clean stratum, and shortest/longest decile, then deterministic fill. The audit fails if sample_count cannot cover all mandatory selections.",
            "path": str(samples_path),
            "sha256": sha256_file(samples_path),
            "count": len(samples),
            "coverage": sample_coverage,
            "manual_review": manual_review,
        },
        "priority_ranked_issues": priorities,
        "corpus_v3_admission_rules": admission_rules,
    }
    report_path = output_dir / "audit-report.json"
    write_json_atomic(report_path, report)
    after = verify_locked_inputs(config["inputs"])
    if after != before:
        raise RuntimeError("An immutable audit input changed during execution")
    manifest = {
        "audit_id": report["audit_id"],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "auditor_path": "src/audit_corpus_v2.py",
        "auditor_sha256": sha256_file(Path(__file__)),
        "generation_provenance": provenance,
        "immutable_inputs_unchanged": True,
        "inputs": report["inputs"],
        "outputs": {
            "audit_report": {"path": str(report_path), "sha256": sha256_file(report_path)},
            "representative_samples": {"path": str(samples_path), "sha256": sha256_file(samples_path)},
        },
    }
    if manual_review["status"] == "complete":
        manifest["outputs"]["sample_review"] = {
            "path": config["manual_review_path"],
            "sha256": manual_review["review_sha256"],
        }
    write_json_atomic(output_dir / "manifest.json", manifest)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/corpus-v2-audit.yaml"))
    return parser.parse_args()


def main() -> None:
    report = audit(parse_args().config)
    print(json.dumps({
        "status": report["status"],
        "stories": report["counts"]["stories"],
        "packed_tokens": report["counts"]["packed_tokens"],
        "report": report["output_path"],
    }, indent=2))


if __name__ == "__main__":
    main()
