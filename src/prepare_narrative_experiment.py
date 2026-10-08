"""Generate the compact preflight record without opening held-out curriculum JSONL."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from src.mixed_schedule import verify_schedule
from src.narrative_post_selection import PROTOCOL_PATH, load_json, verify_protocol
from src.train_tokenizer import sha256_file, write_json_atomic


BASELINES = {
    "corpus_v2": ("fictionpulper-corpus-v2-30m", Path("experiments/fictionpulper-corpus-v2-30m/seal.json")),
    "data30m_v1": ("fictionpulper-15m-data30m-v1", Path("experiments/fictionpulper-15m-data30m-v1/seal.json")),
    "compute5": ("fictionpulper-15m-data30m-compute5", Path("experiments/fictionpulper-15m-data30m-compute5/seal.json")),
    "context2k": ("fictionpulper-15m-data30m-context2k-v1", Path("experiments/fictionpulper-15m-data30m-context2k-v1/seal.json")),
    "capacity50m": ("fictionpulper-50m-data30m-v1", Path("experiments/fictionpulper-50m-data30m-v1/seal.json")),
}


def tag_commit(tag: str) -> str:
    return subprocess.run(["git", "rev-parse", f"{tag}^{{commit}}"], check=True, capture_output=True, text=True).stdout.strip()


def build_preflight(protocol_path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    verify_protocol(protocol)
    curriculum_manifest_path = Path("data/narrative_curriculum_v1/manifest.json")
    curriculum = load_json(curriculum_manifest_path)
    schedule_path = Path("data/narrative_curriculum_v1/packed/mixed-source-schedule.json")
    schedule = load_json(schedule_path)
    schedule_content_hash = verify_schedule(schedule)
    config_path = Path("configs/data30m-15m-narrative-v1.yaml")
    run_dir = Path("runs/fictionpulper-15m-data30m-narrative-v1")
    run_manifest = load_json(run_dir / "manifest.json") if (run_dir / "manifest.json").is_file() else {}
    summary = load_json(run_dir / "summary.json") if (run_dir / "summary.json").is_file() else {}
    return {
        "experiment_id": protocol["experiment_id"],
        "status": summary.get("status", "prepared_not_trained"),
        "selection_metric": protocol["selection_metric"],
        "baseline_tags": {name: {"tag": tag, "commit": tag_commit(tag), "seal_sha256": sha256_file(seal)} for name, (tag, seal) in BASELINES.items()},
        "configuration": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "post_selection_protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "sealed_evaluation_protocol": protocol["sealed_evaluation_protocol"],
        "narrative_state_protocol": protocol["narrative_state_protocol"],
        "curriculum": {
            "manifest_path": str(curriculum_manifest_path),
            "manifest_sha256": sha256_file(curriculum_manifest_path),
            "config_sha256": curriculum["config_sha256"],
            "files": curriculum["files"],
        },
        "mixed_schedule": {"path": str(schedule_path), "file_sha256": sha256_file(schedule_path), "content_sha256": schedule_content_hash},
        "training": {
            "commit": summary.get("training_git_commit", run_manifest.get("training_git_commit")),
            "started_at": summary.get("started_at", run_manifest.get("started_at")),
            "completed_at": summary.get("completed_at", run_manifest.get("completed_at")),
            "status": summary.get("status", "not_started"),
            "best_checkpoint_step": summary.get("best_checkpoint_step"),
            "best_checkpoint_sha256": summary.get("best_checkpoint_sha256"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--output", type=Path, default=Path("experiments/fictionpulper-15m-data30m-narrative-v1/preflight.json"))
    args = parser.parse_args()
    payload = build_preflight(args.protocol)
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
