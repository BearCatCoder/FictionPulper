"""Shared deterministic I/O and text helpers for the continuity curriculum."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def stable_seed(seed: int, *parts: object) -> int:
    payload = canonical_json([seed, *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def normalized_words(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def normalized_text_hash(text: str) -> str:
    return sha256_bytes(" ".join(normalized_words(text)).encode("utf-8"))


def text_shingles(text: str, width: int = 5) -> set[str]:
    words = normalized_words(text)
    if len(words) < width:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + width]) for index in range(len(words) - width + 1)}


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, indent=2, sort_keys=True, ensure_ascii=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(canonical_json(record) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]
