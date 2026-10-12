"""Generate the hash-locked Corpus-v3 seal-and-pack stop disposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.audit_corpus_v2 import verify_locked_inputs
from src.train_tokenizer import write_json_atomic


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def generate_seal_disposition(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise ValueError("Corpus-v3 seal disposition config must declare version: 1")

    actual_hashes = verify_locked_inputs(config["inputs"])
    paths = {name: Path(spec["path"]) for name, spec in config["inputs"].items()}
    blocker_report = _json(paths["blocker_report"])
    source_report = _json(paths["source_pilot_report"])
    data100m_config = yaml.safe_load(paths["data100m_config"].read_text(encoding="utf-8"))

    base = blocker_report["after_correction"]
    expansion = source_report["total_funnel"]
    observed = {
        "base_candidate_records": int(base["candidate_records"]),
        "base_approved_records": int(base["retained_records"]),
        "base_approved_tokens": int(base["retained_tokens"]),
        "source_pilot_gross_records": int(expansion["gross_acquired"]["records"]),
        "source_pilot_gross_tokens": int(
            expansion["gross_acquired"]["tokenizer_v1_tokens"]
        ),
        "source_pilot_approved_records": int(expansion["final_incremental_unique"]["records"]),
        "source_pilot_approved_tokens": int(
            expansion["final_incremental_unique"]["tokenizer_v1_tokens"]
        ),
    }
    if observed != config["expected"]:
        raise RuntimeError(f"Corpus-v3 approved supply changed: {observed}")
    if base.get("freeze_ready") or base.get("seal_written") or not base["failed_required_gates"]:
        raise RuntimeError("Issue #22 no longer supports the no-seal disposition")
    if source_report["decision"].get("action") != "stop_selected_sources":
        raise RuntimeError("Issue #21 no longer supports the stop disposition")

    forbidden = [Path(path) for path in config["artifacts_that_must_be_absent"]]
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise RuntimeError(f"Unauthorized Corpus-v3 seal or pack exists: {present}")

    data = data100m_config["data"]
    unset_locks = [
        key
        for key in (
            "expected_corpus_sha256",
            "expected_split_sha256",
            "expected_packed_metadata_sha256",
        )
        if data.get(key) is None
    ]
    if (
        data100m_config["experiment"].get("status") != "prepared_blocked_on_corpus_v3"
        or len(unset_locks) != 3
    ):
        raise RuntimeError("Data100M is no longer explicitly blocked on an approved Corpus-v3")

    failed_gates = list(base["failed_required_gates"])
    return {
        "report_id": config["report_id"],
        "version": 1,
        "status": "stop_complete_no_approved_corpus",
        "inputs": {
            name: {"path": config["inputs"][name]["path"], "sha256": digest}
            for name, digest in sorted(actual_hashes.items())
        },
        "approved_supply": {
            "records": 0,
            "tokenizer_v1_document_tokens": 0,
            "incremental_unique_records_vs_data30m": 0,
            "incremental_unique_tokens_vs_data30m": 0,
            "qualification": (
                "This is the exact approved supply from the evaluated base and bounded two-source "
                "pilot, not an upper bound on all possible external supply."
            ),
        },
        "evidence": {
            "corrected_base": {
                "candidate_records": observed["base_candidate_records"],
                "approved_records": observed["base_approved_records"],
                "approved_tokens": observed["base_approved_tokens"],
                "rejected_by_first_reason": base["rejection_records_by_first_reason"],
                "failed_required_gates": failed_gates,
            },
            "bounded_source_pilot": {
                "gross_records": observed["source_pilot_gross_records"],
                "gross_tokens": observed["source_pilot_gross_tokens"],
                "approved_records": observed["source_pilot_approved_records"],
                "approved_tokens": observed["source_pilot_approved_tokens"],
                "action": source_report["decision"]["action"],
            },
        },
        "artifact_disposition": {
            "canonical_membership_frozen": False,
            "approved_training_split_frozen": False,
            "corpus_v3_seal_written": False,
            "data100m_pack_written": False,
            "forbidden_artifacts_verified_absent": [str(path) for path in forbidden],
            "reason": (
                "A zero-record corpus cannot pass the required nonempty gates or support "
                "training."
            ),
        },
        "preflight_disposition": {
            "data100m_status": data100m_config["experiment"]["status"],
            "unset_data_hash_locks": unset_locks,
            "real_data_preflight_run": False,
            "reason": (
                "The fail-closed preflight requires a valid seal, corpus and split hashes, packed "
                "metadata, binaries, indexes, and a locked schedule; none may be created from zero "
                "approved records."
            ),
        },
        "experiment_decision": {
            "action": "no_go_data_scaling",
            "data50m_supported": False,
            "data100m_supported": False,
            "training_issue_created": False,
            "common_test_membership_audit_run": False,
            "reason": (
                "There is no candidate training membership to compare against both historical "
                "training sets, and no approved data increase to justify a scaling experiment."
            ),
        },
        "reopen_conditions": [
            "A future bounded acquisition decision supplies a nonempty approved canonical corpus.",
            "Every corrected rights, provenance, boundary, quality, diversity, duplicate, "
            "review, split, tokenizer, and hash-chain gate passes.",
            "The actual attained size is accepted under truthful corpus and experiment naming.",
            "A proposed common test split is audited against both historical baseline training "
            "memberships and the new candidate membership.",
        ],
    }


def write_seal_disposition(report: dict[str, Any], output_path: Path) -> None:
    write_json_atomic(output_path, report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/corpus-v3-seal-disposition-v1.yaml")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = generate_seal_disposition(args.config)
    write_seal_disposition(report, Path(config["output_path"]))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
