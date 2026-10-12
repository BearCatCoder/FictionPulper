"""Deterministic Level A single-owner review for frozen Narrative Benchmark v2 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import subprocess
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from src.continuity_curriculum.common import (
    canonical_json,
    load_jsonl,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
    write_jsonl,
)
from src.narrative_v2_free_generation import (
    _validated_run,
    load_protocol,
)
from src.narrative_v2_scoring import (
    LABELS,
    aggregate_diagnostics,
    build_review_packet,
    paired_bootstrap_interval,
)


VERSION = "narrative-v2-owner-review-v1"
SPLITS = ("development", "test", "generalization")
DEPTHS = ("1", "2", "3", "4_plus")
TRIALS: tuple[tuple[str, int | None], ...] = (
    ("greedy", None),
    ("sampled", 11337),
    ("sampled", 21337),
    ("sampled", 31337),
)
PACKET_CONTENT_FIELDS = (
    "owner_review_id", "generated_text_sha256", "prompt", "generated_text",
    "review_question", "required_propositions", "forbidden_propositions",
)
PACKET_FIELDS = (*PACKET_CONTENT_FIELDS, "reviewer_id", "reviewer_role",
                 "independent_review_confirmed", "genuine_human_attestation",
                 "label", "confidence", "evidence_quote", "reviewer_note")
MUTABLE_REVIEW_FIELDS = {
    "reviewer_id", "genuine_human_attestation", "label", "confidence",
    "evidence_quote", "reviewer_note",
}


class OwnerReviewError(ValueError):
    """Raised when owner-review evidence violates the frozen Level A contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OwnerReviewError(message)


def _rank(seed: int, *parts: object) -> str:
    return hashlib.sha256(canonical_json([seed, *parts]).encode("utf-8")).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    require(config.get("version") == VERSION, "owner-review config version mismatch")
    require(config.get("expected_population") == 1792, "expected population must be 1792")
    require(config.get("expected_sample") == 336, "expected sample must be 336")
    require(config.get("bootstrap_replicates") == 10_000, "bootstrap replicates changed")
    require(config.get("bootstrap_seed") == 260_209, "bootstrap seed changed")
    runs = config.get("runs")
    require(isinstance(runs, list) and len(runs) == 6, "exactly six source runs are required")
    cells = {(run.get("model"), run.get("split")) for run in runs}
    require(cells == {(model, split) for model in ("compute5", "50m") for split in SPLITS},
            "source runs must be exactly 2 models x 3 splits")
    hash_fields = ("raw_generations_sha256", "generation_manifest_sha256",
                   "review_packet_sha256", "review_key_sha256")
    for run in runs:
        path_value = Path(str(run.get("path", "")))
        require(bool(str(path_value)) and not path_value.is_absolute(), "run paths must be repo-relative")
        require(all(re.fullmatch(r"[0-9a-f]{64}", str(run.get(field, ""))) for field in hash_fields),
                "every source artifact requires a complete SHA-256")
    return config


def _historical_artifacts(root: Path, run: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    base = root / run["path"]
    return tuple(base / name for name in (
        "raw-generations.jsonl", "generation-manifest.json",
        "review-packet.jsonl", "review-key.json",
    ))  # type: ignore[return-value]


def validate_source_runs(
    root: Path, config: dict[str, Any], *, checkpoint_selection_complete: bool,
    test_evaluation_complete: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate all historical locks with the existing v2 validators."""
    require(checkpoint_selection_complete, "held-out access requires checkpoint-selection attestation")
    require(test_evaluation_complete, "generalization access requires test-evaluation attestation")
    benchmark_root = root / config["benchmark_root"]
    protocol, hashes = load_protocol(benchmark_root)
    require(sha256_file(benchmark_root / "human-review-rubric.json") == config["rubric_sha256"],
            "rubric identity mismatch")
    rubric = json.loads((benchmark_root / "human-review-rubric.json").read_text(encoding="utf-8"))
    all_records: list[dict[str, Any]] = []
    identities = []
    for run in config["runs"]:
        raw_path, manifest_path, packet_path, key_path = _historical_artifacts(root, run)
        for path, field in zip(
            (raw_path, manifest_path, packet_path, key_path),
            ("raw_generations_sha256", "generation_manifest_sha256", "review_packet_sha256", "review_key_sha256"),
            strict=True,
        ):
            require(path.is_file(), f"missing historical artifact: {path.relative_to(root)}")
            require(sha256_file(path) == run[field], f"historical artifact hash mismatch: {path.relative_to(root)}")
        args = argparse.Namespace(
            generations=raw_path, generation_manifest=manifest_path,
            benchmark_root=benchmark_root,
            checkpoint_selection_complete=checkpoint_selection_complete,
            test_evaluation_complete=test_evaluation_complete,
        )
        records, manifest = _validated_run(args, protocol, hashes)
        require(manifest["model_id"] == run["model_id"], "configured model identity mismatch")
        require(manifest["blinded_model_id"] == run["blinded_model_id"], "configured blinded model mismatch")
        require(manifest["split"] == run["split"], "configured split mismatch")
        historical_packet = load_jsonl(packet_path)
        require(historical_packet == build_review_packet(
            records, randomization_seed=int(rubric["randomization_seed"])
        ), "historical review packet differs from frozen generation records")
        historical_key = json.loads(key_path.read_text(encoding="utf-8"))
        require(historical_key.get("review_packet_sha256") == run["review_packet_sha256"],
                "historical review key packet hash mismatch")
        require(historical_key.get("generation_manifest_sha256") == run["generation_manifest_sha256"],
                "historical review key manifest hash mismatch")
        require(historical_key.get("rubric_sha256") == config["rubric_sha256"],
                "historical review key rubric hash mismatch")
        require(historical_key.get("model_mapping") == {run["blinded_model_id"]: run["model_id"]},
                "historical review key model mapping mismatch")
        require(set(historical_key.get("review_metadata", {})) == {row["review_id"] for row in historical_packet},
                "historical review key metadata is incomplete")
        all_records.extend(records)
        identities.append({
            "model": run["model"], "model_id": run["model_id"],
            "blinded_model_id": run["blinded_model_id"], "split": run["split"],
            "path": run["path"],
            **{field: run[field] for field in (
                "raw_generations_sha256", "generation_manifest_sha256",
                "review_packet_sha256", "review_key_sha256",
            )},
        })
    require(len(all_records) == config["expected_population"], "source population is incomplete")
    require(len({row["generation_id"] for row in all_records}) == len(all_records),
            "source generation IDs are not globally unique")
    return all_records, identities


def select_balanced_records(
    records: Sequence[dict[str, Any]], *, selection_seed: int, expected_sample: int = 336,
) -> list[dict[str, Any]]:
    """Select one text-blind matched quartet for every split/family/depth cell."""
    structural: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    models_by_cell: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for row in records:
        cell = (str(row["split"]), str(row["state_family"]), str(row["depth_bucket"]))
        structural[cell].add(str(row["scenario_id"]))
        models_by_cell[(*cell, str(row["scenario_id"]))].add(str(row["blinded_model_id"]))
    families = sorted({cell[1] for cell in structural})
    require(len(families) == 7, "population must contain seven state families")
    require(set(structural) == {(split, family, depth) for split in SPLITS for family in families for depth in DEPTHS},
            "population family/depth cells are incomplete")
    selected: set[tuple[str, str, str, int | None]] = set()
    for split in SPLITS:
        for family in families:
            trials = sorted(TRIALS, key=lambda trial: _rank(selection_seed, "trial", split, family, *trial))
            for depth, trial in zip(DEPTHS, trials, strict=True):
                candidates = structural[(split, family, depth)]
                expected_candidates = 2 if split == "generalization" else 1
                require(len(candidates) == expected_candidates,
                        f"{split}/{family}/{depth}: unexpected scenario count")
                scenario = min(candidates, key=lambda value: _rank(
                    selection_seed, "scenario", split, family, depth, value
                ))
                require(len(models_by_cell[(split, family, depth, scenario)]) == 2,
                        f"{scenario}: both models are required")
                selected.add((split, scenario, trial[0], trial[1]))
    output = [row for row in records if (
        row["split"], row["scenario_id"], row["generation_mode"], row.get("sampling_seed")
    ) in selected]
    require(len(selected) == 84 and len(output) == expected_sample, "balanced sample size mismatch")
    require(len({row["generation_id"] for row in output}) == len(output), "selected generation IDs are not unique")
    return sorted(output, key=lambda row: str(row["generation_id"]))


def _owner_review_id(record: dict[str, Any], seed: int) -> str:
    return "owner-review-" + sha256_bytes(canonical_json([
        seed, record["generation_id"], record["generated_text_sha256"]
    ]).encode("utf-8"))[:24]


def build_owner_packet(
    selected: Sequence[dict[str, Any]], *, packet_order_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    packet = []
    metadata = {}
    for record in selected:
        identifier = _owner_review_id(record, packet_order_seed)
        packet.append({
            "owner_review_id": identifier,
            "generated_text_sha256": record["generated_text_sha256"],
            "prompt": record["prompt"], "generated_text": record["generated_text"],
            "review_question": record["review_question"],
            "required_propositions": record["required_propositions"],
            "forbidden_propositions": record["forbidden_propositions"],
            "reviewer_id": "", "reviewer_role": "owner",
            "independent_review_confirmed": False,
            "genuine_human_attestation": None, "label": "", "confidence": None,
            "evidence_quote": "", "reviewer_note": "",
        })
        metadata[identifier] = {
            field: record.get(field) for field in (
                "generation_id", "scenario_id", "world", "generation_mode", "sampling_seed",
                "effective_seed", "blinded_model_id", "split", "state_family",
                "reasoning_depth", "depth_bucket", "generated_text_sha256",
            )
        }
    packet.sort(key=lambda row: _rank(packet_order_seed, "packet", row["owner_review_id"]))
    require(len(metadata) == len(packet), "owner review IDs are not unique")
    return packet, metadata


def _balance(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def counts(field: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row[field]) for row in selected).items()))
    trial_counts = Counter(
        "greedy" if row["generation_mode"] == "greedy" else f"sampled:{row['sampling_seed']}"
        for row in selected
    )
    return {
        "rows": len(selected), "quartet_blocks": len(selected) // 4,
        "models": counts("blinded_model_id"), "splits": counts("split"),
        "families": counts("state_family"), "depths": counts("depth_bucket"),
        "worlds": counts("world"), "trials": dict(sorted(trial_counts.items())),
    }


def _write_once_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite owner-review artifact: {path}")
    write_json_atomic(path, value)


def _write_once_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite owner-review artifact: {path}")
    write_jsonl(path, rows)


def export_owner_review(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    require(output_dir.is_relative_to(root), "owner-review output directory must be inside the repository")
    config = load_config(config_path)
    records, sources = validate_source_runs(
        root, config,
        checkpoint_selection_complete=args.checkpoint_selection_complete,
        test_evaluation_complete=args.test_evaluation_complete,
    )
    selected = select_balanced_records(
        records, selection_seed=int(config["selection_seed"]),
        expected_sample=int(config["expected_sample"]),
    )
    packet, metadata = build_owner_packet(selected, packet_order_seed=int(config["packet_order_seed"]))
    packet_path = output_dir / "owner-review-packet.jsonl"
    key_path = output_dir / "owner-review-private-key.json"
    if packet_path.exists() or key_path.exists():
        raise FileExistsError("refusing to overwrite owner packet or private key")
    _write_once_jsonl(packet_path, packet)
    key = {
        "version": VERSION, "status": "OWNER_REVIEW_PENDING",
        "reviewer_role": "owner", "independent_review_confirmed": False,
        "review_packet_path": str(packet_path.relative_to(root)),
        "review_packet_sha256": sha256_file(packet_path),
        "config_path": str(config_path.relative_to(root)), "config_sha256": sha256_file(config_path),
        "rubric_path": f"{config['benchmark_root']}/human-review-rubric.json",
        "rubric_sha256": config["rubric_sha256"], "source_runs": sources,
        "model_mapping": {
            run["blinded_model_id"]: run["model_id"] for run in config["runs"]
        },
        "sample_balance": _balance(selected), "selected_ids": [row["owner_review_id"] for row in packet],
        "review_metadata": metadata,
        "limitations": [
            "single-owner evidence; not independent validation",
            "deterministic sample only; not a full-population human review",
            "no inter-reviewer agreement or adjudication is available",
            "automated diagnostics are diagnostic only and are not owner decisions",
        ],
    }
    _write_once_json(key_path, key)
    result = {"status": "OWNER_REVIEW_PENDING", "review_rows": len(packet),
              "packet": str(packet_path), "packet_sha256": key["review_packet_sha256"],
              "private_key": str(key_path)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _normalized_owner_id(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip()


def validate_owner_reviews(
    packet: Sequence[dict[str, Any]], reviews: Sequence[dict[str, Any]],
    verification: dict[str, Any], *, packet_sha256: str,
) -> tuple[dict[str, str], str]:
    packet_by_id = {str(row["owner_review_id"]): row for row in packet}
    require(len(packet_by_id) == len(packet), "packet owner review IDs are not unique")
    expected_review_fields = {*PACKET_FIELDS, "packet_sha256"}
    labels: dict[str, str] = {}
    owner_ids = set()
    for row in reviews:
        identifier = str(row.get("owner_review_id", ""))
        require(identifier in packet_by_id, f"unknown owner_review_id: {identifier}")
        require(identifier not in labels, f"{identifier}: exactly one owner review required")
        require(set(row) == expected_review_fields, f"{identifier}: review fields differ from packet contract")
        original = packet_by_id[identifier]
        for field in PACKET_FIELDS:
            if field not in MUTABLE_REVIEW_FIELDS:
                require(row[field] == original[field], f"{identifier}: immutable packet content changed")
        require(row["packet_sha256"] == packet_sha256, f"{identifier}: packet hash mismatch")
        normalized = _normalized_owner_id(row["reviewer_id"])
        require(bool(normalized) and row["reviewer_id"] == normalized, "normalized nonblank owner ID required")
        owner_ids.add(normalized)
        require(row["reviewer_role"] == "owner", "reviewer role must be owner")
        require(row["independent_review_confirmed"] is False, "owner review must not claim independence")
        require(row["genuine_human_attestation"] is True, "genuine-human attestation required")
        require(row["label"] in LABELS, f"{identifier}: invalid label")
        confidence = row["confidence"]
        require(isinstance(confidence, int) and not isinstance(confidence, bool) and 1 <= confidence <= 5,
                f"{identifier}: confidence must be an integer from 1 to 5")
        evidence = str(row["evidence_quote"])
        generated = str(original["generated_text"])
        require((not generated and evidence == "<EMPTY_OUTPUT>") or (bool(evidence) and evidence in generated),
                f"{identifier}: evidence must be verbatim or <EMPTY_OUTPUT>")
        require(bool(str(row["reviewer_note"]).strip()), f"{identifier}: reviewer note required")
        labels[identifier] = str(row["label"])
    require(set(labels) == set(packet_by_id), "reviews must contain exactly one row per packet row")
    require(len(owner_ids) == 1, "the same one owner ID must review every row")
    owner_id = next(iter(owner_ids))
    expected_verification_fields = {
        "version", "review_packet_sha256", "owner_id", "reviewed_at",
        "selected_owner_review_ids", "genuine_human_attestation",
        "single_owner_evidence_acknowledged", "non_independent_evidence_acknowledged",
        "sample_only_evidence_acknowledged", "no_agreement_available_acknowledged",
        "no_adjudication_performed_acknowledged",
    }
    require(set(verification) == expected_verification_fields,
            "owner verification fields differ from frozen contract")
    require(verification["version"] == VERSION and verification["review_packet_sha256"] == packet_sha256,
            "owner verification version or packet hash mismatch")
    require(verification["owner_id"] == owner_id and bool(str(verification["reviewed_at"]).strip()),
            "owner verification identity/date mismatch")
    try:
        date.fromisoformat(str(verification["reviewed_at"]))
    except ValueError as error:
        raise OwnerReviewError("owner verification date must be ISO YYYY-MM-DD") from error
    require(set(verification["selected_owner_review_ids"]) == set(packet_by_id)
            and len(verification["selected_owner_review_ids"]) == len(packet_by_id),
            "owner verification selected IDs are incomplete")
    booleans = expected_verification_fields - {
        "version", "review_packet_sha256", "owner_id", "reviewed_at", "selected_owner_review_ids",
    }
    require(all(verification[field] is True for field in booleans),
            "owner verification attestations and acknowledgements must all be true")
    return labels, owner_id


def _pair_selected(
    selected: Sequence[dict[str, Any]], metadata: dict[str, dict[str, Any]], labels: dict[str, str],
    packet_order_seed: int,
) -> list[dict[str, Any]]:
    by_generation = {row["generation_id"]: row for row in selected}
    grouped: dict[tuple[str, str, str, int | None], dict[str, tuple[dict[str, Any], str]]] = defaultdict(dict)
    for review_identifier, hidden in metadata.items():
        generation = by_generation[str(hidden["generation_id"])]
        require(_owner_review_id(generation, packet_order_seed) == review_identifier,
                "private key owner review ID mismatch")
        key = (str(hidden["blinded_model_id"]), str(hidden["scenario_id"]),
               str(hidden["generation_mode"]), hidden.get("sampling_seed"))
        grouped[key][str(hidden["world"])] = (generation, labels[review_identifier])
    pairs = []
    for (model, scenario, mode, seed), worlds in sorted(grouped.items(), key=lambda item: str(item[0])):
        require(set(worlds) == {"A", "B"}, f"{scenario}: selected A/B pair is incomplete")
        first = worlds["A"][0]
        world_labels = {world: value[1] for world, value in worlds.items()}
        pairs.append({
            "blinded_model_id": model, "scenario_id": scenario,
            "generation_mode": mode, "sampling_seed": seed, "split": first["split"],
            "state_family": first["state_family"], "reasoning_depth": first["reasoning_depth"],
            "depth_bucket": first["depth_bucket"], "world_labels": world_labels,
            "strict_success": all(value == "consistent" for value in world_labels.values()),
            "unresolved_uncertain": any(value == "uncertain" for value in world_labels.values()),
        })
    return pairs


def _metric(pairs: Sequence[dict[str, Any]], *, replicates: int, seed: int) -> dict[str, Any]:
    require(bool(pairs), "cannot score an empty pair stratum")
    resolved = [pair for pair in pairs if not pair["unresolved_uncertain"]]
    return {
        "pair_count": len(pairs), "successful_pair_count": sum(pair["strict_success"] for pair in pairs),
        "strict_pair_consistency_rate": sum(pair["strict_success"] for pair in pairs) / len(pairs),
        "unresolved_uncertain_pair_count": len(pairs) - len(resolved),
        "resolved_pair_count": len(resolved),
        "resolved_only_pair_consistency_rate": (
            sum(pair["strict_success"] for pair in resolved) / len(resolved) if resolved else None
        ),
        "strict_rate_interval": paired_bootstrap_interval(pairs, replicates=replicates, seed=seed),
    }


def score_owner_pairs(
    pairs: Sequence[dict[str, Any]], *, bootstrap_replicates: int, bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    strata: dict[tuple[str, str, int | None, str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        mode = str(pair["generation_mode"])
        seed = pair.get("sampling_seed")
        for dimension, value in (
            ("overall", "all"), ("split", pair["split"]), ("family", pair["state_family"]),
            ("depth", pair["depth_bucket"]), ("reasoning_depth", pair["reasoning_depth"]),
            ("family_depth", f"{pair['state_family']}:{pair['depth_bucket']}"),
        ):
            strata[(pair["blinded_model_id"], mode, seed, dimension, str(value))].append(pair)
    metrics = [{
        "blinded_model_id": key[0], "generation_mode": key[1], "sampling_seed": key[2],
        "dimension": key[3], "stratum": key[4],
        **_metric(items, replicates=bootstrap_replicates, seed=bootstrap_seed),
    } for key, items in sorted(strata.items(), key=lambda item: tuple(str(value) for value in item[0]))]
    summaries = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        if metric["generation_mode"] == "sampled":
            groups[(metric["blinded_model_id"], metric["dimension"], metric["stratum"])].append(metric)
    for (model, dimension, stratum), items in sorted(groups.items()):
        items.sort(key=lambda item: item["sampling_seed"])
        values = [item["strict_pair_consistency_rate"] for item in items]
        summaries.append({
            "blinded_model_id": model, "dimension": dimension, "stratum": stratum,
            "seed_count": len(values), "sampling_seeds": [item["sampling_seed"] for item in items],
            "strict_rate_mean": statistics.fmean(values),
            "strict_rate_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    return metrics, summaries


def _scorer_provenance(root: Path) -> dict[str, Any]:
    path = Path(__file__).resolve()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
        check=True, capture_output=True, text=True,
    ).stdout
    return {"scorer_version": VERSION, "scorer_path": str(path.relative_to(root)),
            "scorer_sha256": sha256_file(path), "git_commit": commit,
            "tracked_worktree_clean": not bool(status.strip())}


def score_owner_review(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = load_config(config_path)
    records, sources = validate_source_runs(
        root, config, checkpoint_selection_complete=args.checkpoint_selection_complete,
        test_evaluation_complete=args.test_evaluation_complete,
    )
    selected = select_balanced_records(records, selection_seed=int(config["selection_seed"]),
                                       expected_sample=int(config["expected_sample"]))
    expected_packet, expected_metadata = build_owner_packet(
        selected, packet_order_seed=int(config["packet_order_seed"])
    )
    packet = load_jsonl(args.packet)
    require(packet == expected_packet, "owner packet differs from deterministic selected population")
    packet_hash = sha256_file(args.packet)
    key = json.loads(args.private_key.read_text(encoding="utf-8"))
    require(key.get("version") == VERSION and key.get("review_packet_sha256") == packet_hash,
            "private key packet binding mismatch")
    require(key.get("config_sha256") == sha256_file(config_path), "private key config mismatch")
    require(key.get("source_runs") == sources, "private key source identities mismatch")
    require(key.get("review_metadata") == expected_metadata, "private key hidden metadata mismatch")
    reviews = load_jsonl(args.reviews)
    verification = json.loads(args.owner_verification.read_text(encoding="utf-8"))
    labels, owner_id = validate_owner_reviews(packet, reviews, verification, packet_sha256=packet_hash)
    pairs = _pair_selected(selected, expected_metadata, labels, int(config["packet_order_seed"]))
    metrics, summaries = score_owner_pairs(
        pairs, bootstrap_replicates=int(config["bootstrap_replicates"]),
        bootstrap_seed=int(config["bootstrap_seed"]),
    )
    report = {
        "version": VERSION, "status": "COMPLETE_SINGLE_OWNER_LEVEL_A",
        "evidence_level": "Level A single-owner research evaluation",
        "primary_metric": "single_owner_blinded_pair_consistency",
        "owner_id": owner_id, "reviewer_count": 1, "independent_review_confirmed": False,
        "generation_count": len(selected), "scenario_pair_trial_count": len(pairs),
        "coverage": _balance(selected), "metrics": metrics, "sampled_seed_summary": summaries,
        "agreement": {"status": "not_available_single_owner", "raw_agreement": None, "cohen_kappa": None},
        "adjudication": {"status": "not_performed", "count": 0},
        "automated_diagnostics": {
            "status": "diagnostic_only_not_owner_decisions",
            "summary": aggregate_diagnostics(selected),
        },
        "hashes": {
            "config_sha256": sha256_file(config_path), "packet_sha256": packet_hash,
            "private_key_sha256": sha256_file(args.private_key), "reviews_sha256": sha256_file(args.reviews),
            "owner_verification_sha256": sha256_file(args.owner_verification),
            "rubric_sha256": config["rubric_sha256"],
        },
        "source_runs": sources, **_scorer_provenance(root),
        "limitations": [
            "One owner reviewed a deterministic 336-output sample; results are conditional on this sample and reviewer.",
            "This is not independent validation, full-population review, agreement evidence, adjudication, or publication-grade certification.",
            "Bootstrap intervals quantify resampling uncertainty within the deterministic sample, not reviewer uncertainty or population sampling design.",
            "Automated fact, premise, contradiction, entity, causal, and repetition measures are lexical diagnostics only.",
        ],
        "model_mapping_revealed_after_completed_scoring": key["model_mapping"],
    }
    _write_once_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=Path("configs/narrative-v2-owner-review-v1.json"))
    common.add_argument("--checkpoint-selection-complete", action="store_true", required=True)
    common.add_argument("--test-evaluation-complete", action="store_true", required=True)
    export_parser = subparsers.add_parser("export", parents=[common], help="export write-once owner packet and private key")
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.set_defaults(handler=export_owner_review)
    score_parser = subparsers.add_parser("score", parents=[common], help="validate and score completed owner reviews")
    score_parser.add_argument("--packet", type=Path, required=True)
    score_parser.add_argument("--private-key", type=Path, required=True)
    score_parser.add_argument("--reviews", type=Path, required=True)
    score_parser.add_argument("--owner-verification", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.set_defaults(handler=score_owner_review)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
