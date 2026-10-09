"""Access-controlled loader for authored Narrative Benchmark v2 splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.continuity_curriculum.common import sha256_file


class BenchmarkAccessError(PermissionError):
    """Raised before a held-out split is opened without required attestations."""


def load_authored_split(
    root: Path,
    split: str = "development",
    *,
    checkpoint_selection_complete: bool = False,
    test_evaluation_complete: bool = False,
    controls: bool = False,
) -> list[dict[str, Any]]:
    if split not in {"development", "test", "generalization"}:
        raise ValueError(f"unknown benchmark split: {split}")
    if split in {"test", "generalization"} and not checkpoint_selection_complete:
        raise BenchmarkAccessError("test/generalization require --checkpoint-selection-complete")
    if split == "generalization" and not test_evaluation_complete:
        raise BenchmarkAccessError("generalization requires --test-evaluation-complete")
    manifest = json.loads((root / "authored-manifest.json").read_text(encoding="utf-8"))
    filename = f"{split}-candidate-controls.jsonl" if controls else f"{split}.jsonl"
    expected = manifest["generated_artifacts"][filename]["sha256"]
    path = root / filename
    if sha256_file(path) != expected:
        raise ValueError(f"authored split hash mismatch: {filename}")
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmarks/narrative-v2"))
    parser.add_argument("--split", choices=("development", "test", "generalization"), default="development")
    parser.add_argument("--checkpoint-selection-complete", action="store_true")
    parser.add_argument("--test-evaluation-complete", action="store_true")
    parser.add_argument("--controls", action="store_true", help="load candidate-only controls for the selected split")
    args = parser.parse_args()
    records = load_authored_split(
        args.root, args.split,
        checkpoint_selection_complete=args.checkpoint_selection_complete,
        test_evaluation_complete=args.test_evaluation_complete,
        controls=args.controls,
    )
    print(json.dumps({"split": args.split, "controls": args.controls, "records": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
