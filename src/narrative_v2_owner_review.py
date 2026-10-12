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
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

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
STAGED_VERSION = "narrative-v2-owner-review-staged-v1"
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
STAGE_REVIEW_FIELDS = {*PACKET_FIELDS, "packet_sha256", "stage_manifest_sha256", "reviewed_at"}


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


def load_staged_config(root: Path, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    require(config.get("version") == STAGED_VERSION, "staged owner-review config version mismatch")
    base_path = root / str(config.get("base_config_path", ""))
    require(base_path.is_file() and sha256_file(base_path) == config.get("base_config_sha256"),
            "staged config base-config identity mismatch")
    base = load_config(base_path)
    require(config.get("bootstrap_replicates") == 10_000, "staged bootstrap replicates changed")
    require(config.get("bootstrap_seed") == 260_209, "staged bootstrap seed changed")
    require(config.get("stage_1", {}).get("blocks") == 20, "Stage 1 must contain 20 blocks")
    require(config.get("stage_1", {}).get("rows") == 80, "Stage 1 must contain 80 rows")
    require(config.get("stage_2", {}).get("blocks") == 20, "Stage 2 must contain 20 blocks")
    require(config.get("stage_2", {}).get("rows") == 80, "Stage 2 must contain 80 rows")
    trigger = config.get("stage_2_trigger", {})
    require(trigger == {
        "paired_difference_ci_width_greater_than": 0.30,
        "ci_includes_zero_and_absolute_difference_at_least": 0.10,
        "either_model_unresolved_uncertain_pair_rate_greater_than": 0.10,
    }, "Stage 2 trigger rule changed")
    return config, base


def _trial_name(mode: str, seed: int | None) -> str:
    return "greedy" if mode == "greedy" else f"sampled:{seed}"


def structural_blocks(selected: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        key = (
            str(row["split"]), str(row["state_family"]), str(row["depth_bucket"]),
            str(row["scenario_id"]), str(row["generation_mode"]), row.get("sampling_seed"),
        )
        grouped[key].append(row)
    blocks = []
    for (split, family, depth, scenario, mode, seed), rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        require(len(rows) == 4, f"{scenario}/{mode}/{seed}: quartet block must contain four rows")
        require({row["blinded_model_id"] for row in rows} == {"baseline-a", "baseline-b"},
                f"{scenario}/{mode}/{seed}: quartet must contain both models")
        require(Counter(row["world"] for row in rows) == {"A": 2, "B": 2},
                f"{scenario}/{mode}/{seed}: quartet must contain both worlds for both models")
        block_id = "owner-block-" + sha256_bytes(canonical_json([
            split, family, depth, scenario, mode, seed
        ]).encode("utf-8"))[:24]
        blocks.append({
            "block_id": block_id, "split": split, "state_family": family,
            "depth_bucket": depth, "scenario_id": scenario, "generation_mode": mode,
            "sampling_seed": seed, "trial": _trial_name(mode, seed),
            "records": sorted(rows, key=lambda row: str(row["generation_id"])),
        })
    require(len(blocks) == 84, "full owner selection must contain 84 structural blocks")
    return blocks


def _quota_select(
    candidates: Sequence[dict[str, Any]], *, seed: int, attempt_limit: int,
    trial_targets: dict[str, int], depth_targets: dict[str, int],
    split_targets: dict[str, int], family_targets: dict[str, int],
) -> list[dict[str, Any]]:
    dimensions = {
        "trial": trial_targets, "depth_bucket": depth_targets,
        "split": split_targets, "state_family": family_targets,
    }
    target_count = sum(trial_targets.values())
    require(all(sum(target.values()) == target_count for target in dimensions.values()),
            "selection quota dimensions have different totals")
    for attempt in range(attempt_limit):
        counts = {dimension: Counter() for dimension in dimensions}
        chosen = []
        for block in sorted(candidates, key=lambda item: _rank(seed, attempt, item["block_id"])):
            if all(counts[dimension][block[dimension]] < target[block[dimension]]
                   for dimension, target in dimensions.items()):
                chosen.append(block)
                for dimension in dimensions:
                    counts[dimension][block[dimension]] += 1
                if len(chosen) == target_count:
                    break
        if len(chosen) == target_count and all(
            dict(counts[dimension]) == target for dimension, target in dimensions.items()
        ):
            return sorted(chosen, key=lambda item: item["block_id"])
    raise OwnerReviewError(f"no quota-balanced selection found in {attempt_limit} deterministic attempts")


def select_stage1_blocks(blocks: Sequence[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    stage = config["stage_1"]
    families = sorted({block["state_family"] for block in blocks})
    require(len(families) == 7, "Stage 1 requires seven families")
    two_block_family = min(families, key=lambda family: _rank(int(stage["selection_seed"]), "family", family))
    family_targets = {family: 2 if family == two_block_family else 3 for family in families}
    selected = _quota_select(
        blocks, seed=int(stage["selection_seed"]),
        attempt_limit=int(config["selection_attempt_limit"]),
        trial_targets={str(key): int(value) for key, value in stage["trial_block_counts"].items()},
        depth_targets={str(key): int(value) for key, value in stage["depth_block_counts"].items()},
        split_targets={str(key): int(value) for key, value in stage["split_block_counts"].items()},
        family_targets=family_targets,
    )
    validate_stage_blocks(selected, stage=1)
    return selected


def select_stage2_blocks(
    blocks: Sequence[dict[str, Any]], stage1: Sequence[dict[str, Any]], config: dict[str, Any],
) -> list[dict[str, Any]]:
    stage = config["stage_2"]
    stage1_ids = {block["block_id"] for block in stage1}
    families = sorted({block["state_family"] for block in blocks})
    five_block_families = set(sorted(
        families, key=lambda family: _rank(int(stage["selection_seed"]), "cumulative-family", family)
    )[:2])
    cumulative_targets = {family: 5 if family in five_block_families else 6 for family in families}
    stage1_counts = Counter(block["state_family"] for block in stage1)
    family_targets = {family: cumulative_targets[family] - stage1_counts[family] for family in families}
    require(all(value >= 0 for value in family_targets.values()), "Stage 2 family target is infeasible")
    selected = _quota_select(
        [block for block in blocks if block["block_id"] not in stage1_ids],
        seed=int(stage["selection_seed"]), attempt_limit=int(config["selection_attempt_limit"]),
        trial_targets={str(key): int(value) for key, value in stage["additional_trial_block_counts"].items()},
        depth_targets={str(key): int(value) for key, value in stage["additional_depth_block_counts"].items()},
        split_targets={str(key): int(value) for key, value in stage["additional_split_block_counts"].items()},
        family_targets=family_targets,
    )
    validate_stage_blocks(selected, stage=2)
    require(not stage1_ids.intersection(block["block_id"] for block in selected),
            "Stage 2 selection overlaps Stage 1")
    cumulative = [*stage1, *selected]
    require(sorted(Counter(block["state_family"] for block in cumulative).values()) == [5, 5, 6, 6, 6, 6, 6],
            "cumulative Stage 2 family margin is not balanced")
    return selected


def validate_stage_blocks(blocks: Sequence[dict[str, Any]], *, stage: int) -> None:
    require(len(blocks) == 20 and sum(len(block["records"]) for block in blocks) == 80,
            f"Stage {stage} must contain 20 quartet blocks and 80 rows")
    require(Counter(block["trial"] for block in blocks) == {
        "greedy": 5, "sampled:11337": 5, "sampled:21337": 5, "sampled:31337": 5,
    }, f"Stage {stage} trial margin mismatch")
    require(Counter(block["depth_bucket"] for block in blocks) == {depth: 5 for depth in DEPTHS},
            f"Stage {stage} depth margin mismatch")
    expected_splits = ({"development": 6, "test": 7, "generalization": 7}
                       if stage == 1 else {"development": 7, "test": 6, "generalization": 7})
    require(Counter(block["split"] for block in blocks) == expected_splits,
            f"Stage {stage} split margin mismatch")
    family_counts = Counter(block["state_family"] for block in blocks)
    if stage == 1:
        require(sorted(family_counts.values()) == [2, 3, 3, 3, 3, 3, 3],
                "Stage 1 family coverage must be one 2-block and six 3-block cells")


def _block_coverage(blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "block_count": len(blocks), "row_count": sum(len(block["records"]) for block in blocks),
        "trial_block_counts": dict(sorted(Counter(block["trial"] for block in blocks).items())),
        "depth_block_counts": dict(sorted(Counter(block["depth_bucket"] for block in blocks).items())),
        "split_block_counts": dict(sorted(Counter(block["split"] for block in blocks).items())),
        "family_block_counts": dict(sorted(Counter(block["state_family"] for block in blocks).items())),
    }


def _public_coverage(blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    coverage = _block_coverage(blocks)
    return {
        "matched_quartet_blocks": coverage["block_count"], "rows": coverage["row_count"],
        "models_per_block": 2, "worlds_per_model_per_block": 2,
        "trial_cell_block_counts": sorted(coverage["trial_block_counts"].values()),
        "depth_cell_block_counts": sorted(coverage["depth_block_counts"].values()),
        "split_cell_block_counts": sorted(coverage["split_block_counts"].values()),
        "family_cell_block_counts": sorted(coverage["family_block_counts"].values()),
        "complete_family_by_depth_by_split_coverage": False,
    }


def _json_sha(value: Any) -> str:
    encoded = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    return sha256_bytes(encoded)


def _jsonl_sha(rows: Sequence[dict[str, Any]]) -> str:
    return sha256_bytes("".join(canonical_json(row) + "\n" for row in rows).encode("utf-8"))


def _full_key(
    root: Path, base_path: Path, base: dict[str, Any], sources: list[dict[str, Any]],
    selected: Sequence[dict[str, Any]], packet: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]], packet_path: Path,
) -> dict[str, Any]:
    return {
        "version": VERSION, "status": "OWNER_REVIEW_PENDING", "reviewer_role": "owner",
        "independent_review_confirmed": False,
        "review_packet_path": str(packet_path.relative_to(root)),
        "review_packet_sha256": _jsonl_sha(packet),
        "config_path": str(base_path.relative_to(root)), "config_sha256": sha256_file(base_path),
        "rubric_path": f"{base['benchmark_root']}/human-review-rubric.json",
        "rubric_sha256": base["rubric_sha256"], "source_runs": sources,
        "model_mapping": {run["blinded_model_id"]: run["model_id"] for run in base["runs"]},
        "sample_balance": _balance(selected),
        "selected_ids": [row["owner_review_id"] for row in packet],
        "review_metadata": metadata,
        "limitations": [
            "single-owner evidence; not independent validation",
            "deterministic sample only; not a full-population human review",
            "no inter-reviewer agreement or adjudication is available",
            "automated diagnostics are diagnostic only and are not owner decisions",
        ],
    }


def _write_or_verify_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if path.exists():
        require(sha256_file(path) == _jsonl_sha(rows) and load_jsonl(path) == list(rows),
                f"existing artifact differs from deterministic bytes: {path}")
    else:
        _write_once_jsonl(path, rows)


def _write_or_verify_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        require(sha256_file(path) == _json_sha(value)
                and json.loads(path.read_text(encoding="utf-8")) == value,
                f"existing artifact differs from deterministic bytes: {path}")
    else:
        _write_once_json(path, value)


def _staged_materials(
    root: Path, staged_path: Path, *, checkpoint_selection_complete: bool,
    test_evaluation_complete: bool,
) -> dict[str, Any]:
    staged, base = load_staged_config(root, staged_path)
    base_path = root / staged["base_config_path"]
    records, sources = validate_source_runs(
        root, base, checkpoint_selection_complete=checkpoint_selection_complete,
        test_evaluation_complete=test_evaluation_complete,
    )
    full_selected = select_balanced_records(
        records, selection_seed=int(base["selection_seed"]), expected_sample=int(base["expected_sample"])
    )
    full_packet, full_metadata = build_owner_packet(
        full_selected, packet_order_seed=int(base["packet_order_seed"])
    )
    full_packet_path = root / staged["full_packet_path"]
    full_key_path = root / staged["full_private_key_path"]
    expected_full_key = _full_key(
        root, base_path, base, sources, full_selected, full_packet, full_metadata, full_packet_path
    )
    _write_or_verify_jsonl(full_packet_path, full_packet)
    _write_or_verify_json(full_key_path, expected_full_key)
    require(sha256_file(full_packet_path) == staged["full_packet_sha256"], "full packet historical hash changed")
    require(sha256_file(full_key_path) == staged["full_private_key_sha256"], "full private key historical hash changed")
    blocks = structural_blocks(full_selected)
    stage1_blocks = select_stage1_blocks(blocks, staged)
    stage2_blocks = select_stage2_blocks(blocks, stage1_blocks, staged)
    stage_generation_ids = {
        row["generation_id"] for block in stage1_blocks for row in block["records"]
    }
    stage_ids = {
        identifier for identifier, hidden in full_metadata.items()
        if hidden["generation_id"] in stage_generation_ids
    }
    stage_packet = [row for row in full_packet if row["owner_review_id"] in stage_ids]
    require(len(stage_packet) == 80, "Stage 1 packet row count mismatch")
    stage_metadata = {identifier: full_metadata[identifier] for identifier in stage_ids}
    return {
        "staged": staged, "base": base, "sources": sources,
        "full_selected": full_selected, "full_packet": full_packet,
        "full_metadata": full_metadata, "full_packet_path": full_packet_path,
        "full_key_path": full_key_path, "blocks": blocks,
        "stage1_blocks": stage1_blocks, "stage2_blocks": stage2_blocks,
        "stage_packet": stage_packet, "stage_metadata": stage_metadata,
    }


def _stage1_public_manifest(
    staged_path: Path, staged: dict[str, Any], base: dict[str, Any],
    blocks: Sequence[dict[str, Any]], packet_sha256: str,
) -> dict[str, Any]:
    return {
        "version": STAGED_VERSION, "stage": 1, "status": "OWNER_REVIEW_PENDING",
        "review_packet_sha256": packet_sha256,
        "rubric_path": f"{base['benchmark_root']}/human-review-rubric.json",
        "rubric_sha256": base["rubric_sha256"],
        "staged_config_sha256": sha256_file(staged_path),
        "coverage": _public_coverage(blocks),
        "supported_comparisons": staged["supported_comparisons"],
        "unsupported_comparisons": staged["unsupported_comparisons"],
        "rubric_guidance": {
            "labels": list(LABELS), "confidence": "integer 1 through 5",
            "decision_rule": "Consistent requires every required proposition and no forbidden proposition; lexical overlap alone is insufficient.",
            "evidence": "Quote generated text verbatim, or use <EMPTY_OUTPUT> only for empty output.",
            "note": "A nonempty owner rationale is required.",
        },
        "limitations": [
            "Stage 1 is a deterministic single-owner exploratory sample, not independent validation.",
            "Overall arm rates are descriptive; the matched arm difference is exploratory.",
            "Trial cells contain five blocks per arm; split, family, depth, and reasoning-depth cells are descriptive only.",
            "The sample is not powered, representative, confirmatory, or complete family-by-depth-by-split coverage.",
            "No model identity, condition assignment, source path, or model mapping is present in this public manifest.",
        ],
    }


def export_staged_owner_review(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    staged_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    require(output_dir.is_relative_to(root), "owner-review output directory must be inside the repository")
    materials = _staged_materials(
        root, staged_path, checkpoint_selection_complete=args.checkpoint_selection_complete,
        test_evaluation_complete=args.test_evaluation_complete,
    )
    require(output_dir == materials["full_packet_path"].parent,
            "staged output directory must match the frozen full-artifact directory")
    packet = materials["stage_packet"]
    packet_hash = _jsonl_sha(packet)
    public_manifest = _stage1_public_manifest(
        staged_path, materials["staged"], materials["base"], materials["stage1_blocks"], packet_hash
    )
    packet_path = output_dir / "stage-1-owner-review-packet.jsonl"
    manifest_path = output_dir / "stage-1-public-manifest.json"
    key_path = output_dir / "stage-1-private-key.json"
    private_key = {
        "version": STAGED_VERSION, "stage": 1, "status": "OWNER_REVIEW_PENDING",
        "review_packet_path": str(packet_path.relative_to(root)),
        "review_packet_sha256": packet_hash,
        "public_manifest_path": str(manifest_path.relative_to(root)),
        "public_manifest_sha256": _json_sha(public_manifest),
        "staged_config_path": str(staged_path.relative_to(root)),
        "staged_config_sha256": sha256_file(staged_path),
        "full_packet_sha256": materials["staged"]["full_packet_sha256"],
        "full_private_key_sha256": materials["staged"]["full_private_key_sha256"],
        "source_runs": materials["sources"],
        "model_mapping": {
            run["blinded_model_id"]: run["model_id"] for run in materials["base"]["runs"]
        },
        "anonymous_arm_mapping": {"arm-1": "baseline-a", "arm-2": "baseline-b"},
        "selected_ids": [row["owner_review_id"] for row in packet],
        "review_metadata": materials["stage_metadata"],
        "sample_coverage": _block_coverage(materials["stage1_blocks"]),
        "selected_blocks": [{
            key: block[key] for key in (
                "block_id", "split", "state_family", "depth_bucket", "scenario_id",
                "generation_mode", "sampling_seed", "trial",
            )
        } for block in materials["stage1_blocks"]],
        "stage_2": {
            "status": "NOT_AUTHORIZED_PENDING_STAGE_1_SCORE",
            "candidate_packet_generated": False,
            "trigger_rule": materials["staged"]["stage_2_trigger"],
        },
        "limitations": public_manifest["limitations"],
    }
    _write_or_verify_jsonl(packet_path, packet)
    _write_or_verify_json(manifest_path, public_manifest)
    _write_or_verify_json(key_path, private_key)
    result = {
        "status": "STAGE_1_OWNER_REVIEW_PENDING", "review_rows": len(packet),
        "matched_quartet_blocks": len(materials["stage1_blocks"]),
        "packet_sha256": sha256_file(packet_path),
        "public_manifest_sha256": sha256_file(manifest_path),
        "private_key_sha256": sha256_file(key_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _load_public_stage(packet_path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packet = load_jsonl(packet_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_fields = {
        "version", "stage", "status", "review_packet_sha256", "rubric_path",
        "rubric_sha256", "staged_config_sha256", "coverage", "supported_comparisons",
        "unsupported_comparisons", "rubric_guidance", "limitations",
    }
    require(set(manifest) == expected_manifest_fields, "Stage 1 public manifest fields differ from contract")
    require(manifest.get("version") == STAGED_VERSION and manifest.get("stage") == 1,
            "Stage 1 public manifest identity mismatch")
    require(manifest.get("review_packet_sha256") == sha256_file(packet_path),
            "Stage 1 public manifest packet binding mismatch")
    rubric_path = Path(__file__).resolve().parents[1] / str(manifest.get("rubric_path", ""))
    require(rubric_path.is_file() and sha256_file(rubric_path) == manifest.get("rubric_sha256"),
            "Stage 1 public manifest rubric identity mismatch")
    require(len(packet) == 80, "Stage 1 public packet must contain 80 rows")
    require(all(set(row) == set(PACKET_FIELDS) for row in packet),
            "Stage 1 public packet fields differ from blind contract")
    return packet, manifest


def _validate_stage_review_row(
    row: dict[str, Any], original: dict[str, Any], *, packet_sha256: str,
    manifest_sha256: str,
) -> str:
    identifier = str(original["owner_review_id"])
    require(set(row) == STAGE_REVIEW_FIELDS, f"{identifier}: Stage 1 review fields differ from contract")
    for field in PACKET_FIELDS:
        if field not in MUTABLE_REVIEW_FIELDS:
            require(row[field] == original[field], f"{identifier}: immutable packet content changed")
    require(row["packet_sha256"] == packet_sha256, f"{identifier}: packet hash mismatch")
    require(row["stage_manifest_sha256"] == manifest_sha256, f"{identifier}: manifest hash mismatch")
    owner_id = _normalized_owner_id(row["reviewer_id"])
    require(bool(owner_id) and row["reviewer_id"] == owner_id, "normalized nonblank owner ID required")
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
    timestamp = str(row["reviewed_at"])
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise OwnerReviewError(f"{identifier}: reviewed_at must be an ISO timestamp") from error
    offset = parsed.utcoffset()
    require(parsed.tzinfo is not None and offset is not None,
            f"{identifier}: reviewed_at must include UTC timezone")
    require(offset is not None and offset.total_seconds() == 0, f"{identifier}: reviewed_at must be UTC")
    return owner_id


def validate_stage_reviews(
    packet: Sequence[dict[str, Any]], reviews: Sequence[dict[str, Any]], *,
    packet_sha256: str, manifest_sha256: str, require_complete: bool,
) -> tuple[dict[str, dict[str, Any]], str | None]:
    packet_by_id = {str(row["owner_review_id"]): row for row in packet}
    require(len(packet_by_id) == len(packet), "Stage 1 packet IDs are not unique")
    completed: dict[str, dict[str, Any]] = {}
    owner_ids = set()
    for row in reviews:
        identifier = str(row.get("owner_review_id", ""))
        require(identifier in packet_by_id, f"unknown Stage 1 owner_review_id: {identifier}")
        require(identifier not in completed, f"{identifier}: original judgment is duplicated")
        owner_ids.add(_validate_stage_review_row(
            row, packet_by_id[identifier], packet_sha256=packet_sha256,
            manifest_sha256=manifest_sha256,
        ))
        completed[identifier] = row
    require(len(owner_ids) <= 1, "the same one owner ID must review every Stage 1 row")
    if require_complete:
        require(set(completed) == set(packet_by_id), "Stage 1 reviews are incomplete")
    return completed, next(iter(owner_ids)) if owner_ids else None


def _utc_timestamp(now: datetime) -> str:
    require(now.tzinfo is not None and now.utcoffset() is not None, "review clock must be timezone-aware")
    return now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def run_terminal_review(
    args: argparse.Namespace, *, input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    packet, manifest = _load_public_stage(args.packet, args.manifest)
    packet_hash = sha256_file(args.packet)
    manifest_hash = sha256_file(args.manifest)
    existing = load_jsonl(args.reviews) if args.reviews.exists() else []
    completed, existing_owner = validate_stage_reviews(
        packet, existing, packet_sha256=packet_hash, manifest_sha256=manifest_hash,
        require_complete=False,
    )
    owner_id = _normalized_owner_id(args.reviewer_id)
    require(bool(owner_id) and owner_id == args.reviewer_id, "normalized nonblank reviewer ID required")
    require(existing_owner in {None, owner_id}, "review progress belongs to a different owner")
    pending = [row for row in packet if row["owner_review_id"] not in completed]
    if args.dry_run:
        if pending:
            row = pending[0]
            output_fn(f"DRY RUN: next Stage 1 row {len(completed) + 1}/{len(packet)}")
            output_fn(canonical_json({field: row[field] for field in PACKET_CONTENT_FIELDS}))
            output_fn(canonical_json(manifest["rubric_guidance"]))
        return {"status": "DRY_RUN", "completed": len(completed), "remaining": len(pending)}
    for index, original in enumerate(pending, start=len(completed) + 1):
        output_fn(f"\nStage 1 row {index}/{len(packet)}")
        output_fn(f"Prompt:\n{original['prompt']}")
        output_fn(f"Continuation:\n{original['generated_text'] or '<EMPTY_OUTPUT>'}")
        output_fn(f"Question: {original['review_question']}")
        output_fn(f"Required: {canonical_json(original['required_propositions'])}")
        output_fn(f"Forbidden: {canonical_json(original['forbidden_propositions'])}")
        output_fn(f"Rubric: {manifest['rubric_guidance']['decision_rule']}")
        label = input_fn("Label [consistent/inconsistent/uncertain]: ").strip()
        require(label in LABELS, "invalid label; no decision was saved")
        confidence_text = input_fn("Confidence [1-5]: ").strip()
        require(confidence_text.isdigit() and 1 <= int(confidence_text) <= 5,
                "invalid confidence; no decision was saved")
        evidence = input_fn("Verbatim evidence (or <EMPTY_OUTPUT>): ")
        generated = str(original["generated_text"])
        require((not generated and evidence == "<EMPTY_OUTPUT>") or (bool(evidence) and evidence in generated),
                "evidence is not verbatim; no decision was saved")
        note = input_fn("Reviewer note: ").strip()
        require(bool(note), "reviewer note is required; no decision was saved")
        reviewed = dict(original)
        reviewed.update({
            "reviewer_id": owner_id, "reviewer_role": "owner",
            "independent_review_confirmed": False, "genuine_human_attestation": True,
            "label": label, "confidence": int(confidence_text), "evidence_quote": evidence,
            "reviewer_note": note, "packet_sha256": packet_hash,
            "stage_manifest_sha256": manifest_hash, "reviewed_at": _utc_timestamp(now_fn()),
        })
        completed[str(original["owner_review_id"])] = reviewed
        ordered = [completed[row["owner_review_id"]] for row in packet if row["owner_review_id"] in completed]
        write_jsonl(args.reviews, ordered)
    verification_written = False
    if args.final_attestation:
        require(len(completed) == len(packet), "final attestation requires all Stage 1 rows")
        require(args.owner_verification is not None, "--owner-verification is required with final attestation")
        verification = {
            "version": STAGED_VERSION, "stage": 1, "review_packet_sha256": packet_hash,
            "public_manifest_sha256": manifest_hash, "owner_id": owner_id,
            "completed_at": _utc_timestamp(now_fn()),
            "selected_owner_review_ids": [row["owner_review_id"] for row in packet],
            "genuine_human_attestation": True, "single_owner_evidence_acknowledged": True,
            "non_independent_evidence_acknowledged": True, "sample_only_evidence_acknowledged": True,
            "no_agreement_available_acknowledged": True,
            "no_adjudication_performed_acknowledged": True,
            "automated_diagnostics_are_not_owner_decisions_acknowledged": True,
        }
        _write_once_json(args.owner_verification, verification)
        verification_written = True
    return {
        "status": "STAGE_1_REVIEW_COMPLETE" if len(completed) == len(packet) else "STAGE_1_REVIEW_IN_PROGRESS",
        "completed": len(completed), "remaining": len(packet) - len(completed),
        "verification_written": verification_written,
    }


def _validate_stage_verification(
    value: dict[str, Any], *, packet: Sequence[dict[str, Any]], packet_sha256: str,
    manifest_sha256: str, owner_id: str,
) -> None:
    expected = {
        "version", "stage", "review_packet_sha256", "public_manifest_sha256", "owner_id",
        "completed_at", "selected_owner_review_ids", "genuine_human_attestation",
        "single_owner_evidence_acknowledged", "non_independent_evidence_acknowledged",
        "sample_only_evidence_acknowledged", "no_agreement_available_acknowledged",
        "no_adjudication_performed_acknowledged",
        "automated_diagnostics_are_not_owner_decisions_acknowledged",
    }
    require(set(value) == expected, "Stage 1 owner verification fields differ from contract")
    require(value["version"] == STAGED_VERSION and value["stage"] == 1,
            "Stage 1 owner verification identity mismatch")
    require(value["review_packet_sha256"] == packet_sha256
            and value["public_manifest_sha256"] == manifest_sha256,
            "Stage 1 owner verification artifact binding mismatch")
    require(value["owner_id"] == owner_id, "Stage 1 owner verification identity mismatch")
    try:
        completed_at = datetime.fromisoformat(str(value["completed_at"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise OwnerReviewError("Stage 1 completed_at must be an ISO UTC timestamp") from error
    completed_offset = completed_at.utcoffset()
    require(completed_at.tzinfo is not None and completed_offset is not None
            and completed_offset.total_seconds() == 0,
            "Stage 1 completed_at must be UTC")
    expected_ids = [row["owner_review_id"] for row in packet]
    require(value["selected_owner_review_ids"] == expected_ids,
            "Stage 1 owner verification IDs/order are incomplete")
    booleans = expected - {
        "version", "stage", "review_packet_sha256", "public_manifest_sha256", "owner_id",
        "completed_at", "selected_owner_review_ids",
    }
    require(all(value[field] is True for field in booleans),
            "Stage 1 owner verification acknowledgements must all be true")


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_model_difference(
    pairs: Sequence[dict[str, Any]], *, arm_mapping: dict[str, str], replicates: int, seed: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int | None], dict[str, dict[str, Any]]] = defaultdict(dict)
    reverse = {model: arm for arm, model in arm_mapping.items()}
    require(set(reverse) == {"baseline-a", "baseline-b"}, "anonymous arm mapping is invalid")
    for pair in pairs:
        key = (str(pair["scenario_id"]), str(pair["generation_mode"]), pair.get("sampling_seed"))
        grouped[key][reverse[str(pair["blinded_model_id"])]] = pair
    require(len(grouped) == 20 and all(set(value) == {"arm-1", "arm-2"} for value in grouped.values()),
            "paired model difference requires 20 blocks with both anonymous arms")
    differences = [
        int(arms["arm-2"]["strict_success"]) - int(arms["arm-1"]["strict_success"])
        for _, arms in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]
    point = statistics.fmean(differences)
    rng = random.Random(seed)
    bootstraps = [statistics.fmean(differences[rng.randrange(len(differences))] for _ in differences)
                  for _ in range(replicates)]
    lower = _percentile(bootstraps, 0.025)
    upper = _percentile(bootstraps, 0.975)
    return {
        "comparison": "arm-2 minus arm-1 strict pair-consistency rate",
        "matched_block_count": len(differences), "point_difference": point,
        "bootstrap": {
            "method": "percentile paired bootstrap over matched quartet blocks",
            "replicates": replicates, "seed": seed, "confidence_level": 0.95,
            "lower": lower, "upper": upper, "width": upper - lower,
        },
    }


def stage2_trigger_decision(
    difference: dict[str, Any], unresolved_rates: dict[str, float], rule: dict[str, float],
) -> dict[str, Any]:
    interval = difference["bootstrap"]
    conditions = {
        "ci_width_exceeds_0_30": interval["width"] > rule["paired_difference_ci_width_greater_than"],
        "ci_includes_zero_and_absolute_difference_at_least_0_10": (
            interval["lower"] <= 0 <= interval["upper"]
            and abs(difference["point_difference"]) >= rule["ci_includes_zero_and_absolute_difference_at_least"]
        ),
        "either_arm_unresolved_uncertain_rate_exceeds_0_10": any(
            value > rule["either_model_unresolved_uncertain_pair_rate_greater_than"]
            for value in unresolved_rates.values()
        ),
    }
    return {
        "rule_type": "preregistered decision rule; not a power calculation",
        "conditions": conditions, "triggered": any(conditions.values()),
        "decision": "AUTHORIZE_STAGE_2" if any(conditions.values()) else "STOP_AFTER_STAGE_1",
        "authorized_additional_blocks": 20 if any(conditions.values()) else 0,
        "authorized_additional_rows": 80 if any(conditions.values()) else 0,
        "stage_2_packet_generated": False,
    }


def _descriptive_stage_metrics(
    pairs: Sequence[dict[str, Any]], arm_mapping: dict[str, str],
) -> list[dict[str, Any]]:
    reverse = {model: arm for arm, model in arm_mapping.items()}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        arm = reverse[str(pair["blinded_model_id"])]
        for dimension, value in (
            ("overall", "all"), ("trial", _trial_name(pair["generation_mode"], pair.get("sampling_seed"))),
            ("split", pair["split"]), ("family", pair["state_family"]),
            ("depth", pair["depth_bucket"]), ("reasoning_depth", str(pair["reasoning_depth"])),
        ):
            groups[(arm, dimension, str(value))].append(pair)
    output = []
    for (arm, dimension, value), items in sorted(groups.items()):
        resolved = [item for item in items if not item["unresolved_uncertain"]]
        output.append({
            "anonymous_arm": arm, "dimension": dimension, "stratum": value,
            "cell_size_blocks": len(items),
            "strict_pair_consistency_rate": sum(item["strict_success"] for item in items) / len(items),
            "unresolved_uncertain_pair_count": len(items) - len(resolved),
            "unresolved_uncertain_pair_rate": (len(items) - len(resolved)) / len(items),
            "resolved_only_pair_consistency_rate": (
                sum(item["strict_success"] for item in resolved) / len(resolved) if resolved else None
            ),
            "interpretation": "descriptive_only",
        })
    return output


def _scorer_provenance(root: Path) -> dict[str, Any]:
    path = Path(__file__).resolve()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
        check=True, capture_output=True, text=True,
    ).stdout
    return {"scorer_version": STAGED_VERSION, "scorer_path": str(path.relative_to(root)),
            "scorer_sha256": sha256_file(path), "git_commit": commit,
            "tracked_worktree_clean": not bool(status.strip())}


def score_owner_review(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    staged_path = args.config.resolve()
    materials = _staged_materials(
        root, staged_path, checkpoint_selection_complete=args.checkpoint_selection_complete,
        test_evaluation_complete=args.test_evaluation_complete,
    )
    packet, manifest = _load_public_stage(args.packet, args.manifest)
    expected_packet = materials["stage_packet"]
    expected_manifest = _stage1_public_manifest(
        staged_path, materials["staged"], materials["base"], materials["stage1_blocks"],
        _jsonl_sha(expected_packet),
    )
    require(packet == expected_packet, "Stage 1 packet differs from deterministic selection")
    require(manifest == expected_manifest, "Stage 1 public manifest differs from frozen contract")
    packet_hash = sha256_file(args.packet)
    manifest_hash = sha256_file(args.manifest)
    key = json.loads(args.private_key.read_text(encoding="utf-8"))
    require(key.get("version") == STAGED_VERSION and key.get("stage") == 1,
            "Stage 1 private key identity mismatch")
    require(key.get("review_packet_sha256") == packet_hash
            and key.get("public_manifest_sha256") == manifest_hash,
            "Stage 1 private key artifact binding mismatch")
    require(key.get("staged_config_sha256") == sha256_file(staged_path),
            "Stage 1 private key config binding mismatch")
    require(key.get("review_metadata") == materials["stage_metadata"],
            "Stage 1 private metadata differs from deterministic selection")
    require(key.get("anonymous_arm_mapping") == {"arm-1": "baseline-a", "arm-2": "baseline-b"},
            "Stage 1 anonymous arm mapping mismatch")
    require(key.get("sample_coverage") == _block_coverage(materials["stage1_blocks"]),
            "Stage 1 private sample coverage mismatch")
    require(key.get("source_runs") == materials["sources"],
            "Stage 1 private source identities mismatch")
    require(key.get("model_mapping") == {
        run["blinded_model_id"]: run["model_id"] for run in materials["base"]["runs"]
    }, "Stage 1 private model mapping mismatch")
    require(key.get("full_packet_sha256") == materials["staged"]["full_packet_sha256"]
            and key.get("full_private_key_sha256") == materials["staged"]["full_private_key_sha256"],
            "Stage 1 private full-artifact identity mismatch")
    require(key.get("selected_ids") == [row["owner_review_id"] for row in packet],
            "Stage 1 private selected IDs mismatch")
    expected_blocks = [{
        field: block[field] for field in (
            "block_id", "split", "state_family", "depth_bucket", "scenario_id",
            "generation_mode", "sampling_seed", "trial",
        )
    } for block in materials["stage1_blocks"]]
    require(key.get("selected_blocks") == expected_blocks,
            "Stage 1 private selected blocks mismatch")
    reviews = load_jsonl(args.reviews)
    completed, owner_id = validate_stage_reviews(
        packet, reviews, packet_sha256=packet_hash, manifest_sha256=manifest_hash,
        require_complete=True,
    )
    require(owner_id is not None, "Stage 1 owner identity is missing")
    validated_owner_id = str(owner_id)
    verification = json.loads(args.owner_verification.read_text(encoding="utf-8"))
    _validate_stage_verification(
        verification, packet=packet, packet_sha256=packet_hash,
        manifest_sha256=manifest_hash, owner_id=validated_owner_id,
    )
    labels = {identifier: str(row["label"]) for identifier, row in completed.items()}
    pairs = _pair_selected(
        [row for block in materials["stage1_blocks"] for row in block["records"]],
        materials["stage_metadata"], labels, int(materials["base"]["packet_order_seed"]),
    )
    arm_mapping = key["anonymous_arm_mapping"]
    metrics = _descriptive_stage_metrics(pairs, arm_mapping)
    overall = {
        row["anonymous_arm"]: row for row in metrics
        if row["dimension"] == "overall" and row["stratum"] == "all"
    }
    require(set(overall) == {"arm-1", "arm-2"}
            and all(row["cell_size_blocks"] == 20 for row in overall.values()),
            "Stage 1 overall arm cells must contain 20 pairs")
    difference = paired_model_difference(
        pairs, arm_mapping=arm_mapping,
        replicates=int(materials["staged"]["bootstrap_replicates"]),
        seed=int(materials["staged"]["bootstrap_seed"]),
    )
    unresolved_rates = {
        arm: float(row["unresolved_uncertain_pair_rate"]) for arm, row in overall.items()
    }
    trigger = stage2_trigger_decision(
        difference, unresolved_rates, materials["staged"]["stage_2_trigger"]
    )
    diagnostics = aggregate_diagnostics(
        [row for block in materials["stage1_blocks"] for row in block["records"]]
    )
    reverse_arms = {model: arm for arm, model in arm_mapping.items()}
    for row in diagnostics["strata"]:
        row["anonymous_arm"] = reverse_arms[row.pop("blinded_model_id")]
    source_identity_sha256 = sha256_bytes(canonical_json(materials["sources"]).encode("utf-8"))
    report = {
        "version": STAGED_VERSION, "stage": 1,
        "status": "COMPLETE_PARTIAL_SINGLE_OWNER_EXPLORATORY_EVIDENCE",
        "owner_id": validated_owner_id, "reviewer_count": 1, "independent_review_confirmed": False,
        "evidence_scope": "partial single-owner exploratory evidence",
        "coverage": _public_coverage(materials["stage1_blocks"]),
        "supported_comparisons": materials["staged"]["supported_comparisons"],
        "unsupported_comparisons": materials["staged"]["unsupported_comparisons"],
        "descriptive_metrics": metrics, "paired_model_difference": difference,
        "agreement": {"status": "not_available_single_owner", "raw_agreement": None, "cohen_kappa": None},
        "adjudication": {"status": "not_performed", "count": 0},
        "automated_diagnostics": {
            "status": "diagnostic_only_not_owner_decisions", "summary": diagnostics,
        },
        "stage_2_decision": trigger,
        "model_mapping": {"status": "withheld_until_all_triggered_stages_are_complete"},
        "hashes": {
            "staged_config_sha256": sha256_file(staged_path), "packet_sha256": packet_hash,
            "public_manifest_sha256": manifest_hash, "private_key_sha256": sha256_file(args.private_key),
            "reviews_sha256": sha256_file(args.reviews),
            "owner_verification_sha256": sha256_file(args.owner_verification),
            "rubric_sha256": materials["base"]["rubric_sha256"],
            "historical_source_identity_bundle_sha256": source_identity_sha256,
        },
        **_scorer_provenance(root),
        "limitations": expected_manifest["limitations"] + [
            "The paired bootstrap is conditional on these 20 deterministically selected blocks and one owner.",
            "The Stage 2 rule is a preregistered decision rule, not a statistical power calculation.",
        ],
    }
    _write_once_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=Path("configs/narrative-v2-owner-review-staged-v1.json"))
    common.add_argument("--checkpoint-selection-complete", action="store_true", required=True)
    common.add_argument("--test-evaluation-complete", action="store_true", required=True)
    export_parser = subparsers.add_parser("export", parents=[common], help="verify full artifacts and export Stage 1")
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.set_defaults(handler=export_staged_owner_review)
    review_parser = subparsers.add_parser("review", help="resume terminal Stage 1 owner review")
    review_parser.add_argument("--packet", type=Path, required=True)
    review_parser.add_argument("--manifest", type=Path, required=True)
    review_parser.add_argument("--reviews", type=Path, required=True)
    review_parser.add_argument("--reviewer-id", required=True)
    review_parser.add_argument("--dry-run", action="store_true")
    review_parser.add_argument("--final-attestation", action="store_true")
    review_parser.add_argument("--owner-verification", type=Path)
    review_parser.set_defaults(handler=run_terminal_review)
    score_parser = subparsers.add_parser("score", parents=[common], help="validate and score completed Stage 1")
    score_parser.add_argument("--packet", type=Path, required=True)
    score_parser.add_argument("--manifest", type=Path, required=True)
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
