"""Build or independently validate Narrative Continuity Curriculum v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from tokenizers import Tokenizer

from src.continuity_curriculum.audits import (
    distance_audit,
    exact_duplicate_audit,
    quality_audit,
    similarity_audit,
    state_audit,
    template_audit,
)
from src.continuity_curriculum.common import load_jsonl, sha256_file
from src.continuity_curriculum.generator import GENRES, SPLITS, STATE_TYPES, build_curriculum
from src.continuity_curriculum.validation import validate_example, validate_split_isolation


def validate_existing(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tokenizer_path = Path(config["tokenizer"]["path"])
    if sha256_file(tokenizer_path) != config["tokenizer"]["expected_sha256"]:
        raise RuntimeError("tokenizer-v1 hash does not match the locked configuration")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    root = Path(config["output_path"])
    records = [record for split in SPLITS for record in load_jsonl(root / f"{split}.jsonl")]
    sidecars = [sidecar for split in SPLITS for sidecar in load_jsonl(root / f"{split}.state.jsonl")]
    by_id = {sidecar["id"]: sidecar for sidecar in sidecars}
    for record in records:
        validate_example(record, by_id[record["id"]], config=config, tokenizer=tokenizer)
    isolation = validate_split_isolation(records, sidecars)
    exact = exact_duplicate_audit(records)
    similarity = similarity_audit(records, **config["audits"]["similarity"])
    reports = {
        "split_isolation": isolation,
        "templates": template_audit(records, config["audits"]["maximum_template_family_share"]),
        "states": state_audit(records, sidecars, list(STATE_TYPES)),
        "distances": distance_audit(records, config["distance_buckets"]),
        "duplicates": {"passed": exact["passed"] and similarity["passed"], "exact": exact, "similarity": similarity},
        "quality": quality_audit(
            records,
            required_genres=list(GENRES),
            maximum_document_tokens=config["maximum_document_tokens"],
            minimum_words=config["audits"]["minimum_story_words"],
        ),
    }
    return {
        "passed": all(report["passed"] for report in reports.values()),
        "record_count": len(records),
        "reports": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/continuity-curriculum-v1.yaml"))
    parser.add_argument("--max-examples", type=int, help="Development-only deterministic size cap")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_existing(args.config) if args.validate_only else build_curriculum(
        args.config, max_examples=args.max_examples
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
