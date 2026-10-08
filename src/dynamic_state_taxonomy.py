"""Read-only stale-state taxonomy for sealed counterfactual evaluations."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence, cast

from src.continuity_curriculum.common import canonical_json, sha256_bytes, sha256_file, write_json_atomic


CATEGORIES = ("CURRENT", "PREVIOUS", "INITIAL", "OLDER", "NEVER_VALID", "OTHER_UNRESOLVED")
DEPTH_BUCKETS = ("1", "2", "3", "4+")


class TaxonomyValidationError(ValueError):
    """Raised when sealed source records and score trials cannot be joined safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TaxonomyValidationError(message)


def _stable_hash(record: Mapping[str, Any]) -> str:
    unhashed = {key: value for key, value in record.items() if key != "stable_hash"}
    return sha256_bytes(canonical_json(unhashed).encode("utf-8"))


def depth_bucket(depth: int) -> str:
    """Return the fixed transition-depth reporting bucket."""
    _require(isinstance(depth, int) and not isinstance(depth, bool) and depth >= 1,
             "Transition depth must be a positive integer")
    return str(depth) if depth <= 3 else "4+"


def _candidate_values(pair: Mapping[str, Any]) -> dict[str, str]:
    candidates_value = pair.get("candidates")
    _require(isinstance(candidates_value, Mapping) and len(candidates_value) >= 2,
             f"Pair {pair.get('pair_id')!r} must have at least two candidates")
    candidates = cast(Mapping[Any, Any], candidates_value)
    values: dict[str, str] = {}
    for name, candidate in candidates.items():
        _require(isinstance(name, str) and isinstance(candidate, Mapping),
                 f"Malformed candidate in pair {pair.get('pair_id')!r}")
        value = candidate.get("value")
        if isinstance(value, str):
            values[name] = value

    # Counterfactual-v1 candidates predate an explicit value field. Its reversed
    # labels prove this mapping without parsing candidate prose.
    for world in pair.get("worlds", {}).values():
        if not isinstance(world, Mapping):
            continue
        candidate = world.get("correct_candidate")
        value = world.get("final_value")
        if isinstance(candidate, str) and isinstance(value, str):
            previous = values.setdefault(candidate, value)
            _require(previous == value,
                     f"Candidate {candidate!r} maps to conflicting values in pair {pair.get('pair_id')!r}")
    return values


def _history(pair: Mapping[str, Any], world_name: str) -> list[str]:
    worlds_value = pair.get("worlds")
    _require(isinstance(worlds_value, Mapping) and world_name in worlds_value,
             f"Pair {pair.get('pair_id')!r} lacks world {world_name!r}")
    worlds = cast(Mapping[str, Any], worlds_value)
    world = worlds[world_name]
    _require(isinstance(world, Mapping), f"Malformed world {world_name!r}")
    events_value = world.get("state_events")
    _require(bool(isinstance(events_value, list) and events_value),
             f"World {world_name!r} has no state history")
    events = cast(list[Any], events_value)
    sequence = [event.get("sequence") for event in events if isinstance(event, Mapping)]
    values = [event.get("value") for event in events if isinstance(event, Mapping)]
    _require(len(sequence) == len(events) and sequence == list(range(len(events))),
             f"World {world_name!r} has malformed state-event sequence")
    _require(all(isinstance(value, str) and value for value in values),
             f"World {world_name!r} has malformed state values")
    _require(world.get("final_value") == values[-1],
             f"World {world_name!r} final value contradicts its state history")
    return values  # type: ignore[return-value]


def classify_world_choice(
    pair: Mapping[str, Any], world_name: str, selected_candidate: str | None,
) -> tuple[str, list[str]]:
    """Classify a choice and return its precedence winner plus all aliases."""
    history = _history(pair, world_name)
    values = _candidate_values(pair)
    if selected_candidate is None or selected_candidate not in values:
        return "OTHER_UNRESOLVED", ["OTHER_UNRESOLVED"]
    selected_value = values[selected_candidate]
    aliases: list[str] = []
    if selected_value == history[-1]:
        aliases.append("CURRENT")
    if len(history) >= 2 and selected_value == history[-2]:
        aliases.append("PREVIOUS")
    if selected_value == history[0]:
        aliases.append("INITIAL")
    if selected_value in history[1:-2]:
        aliases.append("OLDER")
    if selected_value not in history:
        aliases.append("NEVER_VALID")
    if not aliases:
        aliases.append("OTHER_UNRESOLVED")
    ordered = [category for category in CATEGORIES if category in aliases]
    return ordered[0], ordered


def _normal_scores(trial: Mapping[str, Any], world_name: str,
                   candidates: Sequence[str]) -> tuple[dict[str, float], str | None]:
    normal_value = trial.get("normal")
    _require(isinstance(normal_value, Mapping) and isinstance(normal_value.get(world_name), Mapping),
             f"Trial {trial.get('pair_id')!r} lacks normal scores for world {world_name}")
    normal = cast(Mapping[str, Any], normal_value)
    world = normal[world_name]
    world = cast(Mapping[str, Any], world)
    scores: dict[str, float] = {}
    for candidate in candidates:
        score = world.get(candidate)
        _require(isinstance(score, (int, float)) and not isinstance(score, bool)
                 and math.isfinite(float(score)),
                 f"Trial {trial.get('pair_id')!r}/{world_name} has a malformed {candidate!r} score")
        scores[candidate] = float(cast(int | float, score))
    preference = world.get("preference")
    _require(preference in {*candidates, "tie"},
             f"Trial {trial.get('pair_id')!r}/{world_name} has an invalid preference")
    selected = None if preference == "tie" else str(preference)
    if selected is not None:
        maximum = max(scores.values())
        _require(math.isclose(scores[selected], maximum, rel_tol=0.0, abs_tol=1e-12),
                 f"Trial {trial.get('pair_id')!r}/{world_name} preference contradicts scores")
    return scores, selected


def _validate_pair(pair: Mapping[str, Any]) -> None:
    pair_id = pair.get("pair_id")
    _require(bool(isinstance(pair_id, str) and pair_id), "Pair lacks a valid pair_id")
    _require(pair.get("stable_hash") == _stable_hash(pair), f"Stable hash mismatch for pair {pair_id!r}")
    _require(isinstance(pair.get("split"), str), f"Pair {pair_id!r} lacks a split")
    _require(bool(isinstance(pair.get("abstract_counterfactual_variable"), Mapping)
             and isinstance(pair["abstract_counterfactual_variable"].get("state_family"), str)),
             f"Pair {pair_id!r} lacks a state family")
    _candidate_values(pair)
    worlds_value = pair.get("worlds")
    _require(bool(isinstance(worlds_value, Mapping) and worlds_value), f"Pair {pair_id!r} lacks worlds")
    worlds = cast(Mapping[Any, Any], worlds_value)
    for world_name in worlds:
        _history(pair, str(world_name))


def _iter_model_splits(evaluation: Mapping[str, Any]) -> Iterable[tuple[str, str, Sequence[Mapping[str, Any]]]]:
    models_value = evaluation.get("models")
    _require(bool(isinstance(models_value, Mapping) and models_value),
             "Evaluation lacks a non-empty models mapping")
    models = cast(Mapping[Any, Any], models_value)
    for model, splits in models.items():
        _require(isinstance(model, str) and isinstance(splits, Mapping), "Malformed evaluation model entry")
        for split, result in splits.items():
            _require(isinstance(split, str) and isinstance(result, Mapping),
                     f"Malformed evaluation split for model {model!r}")
            trials_value = result.get("trials")
            _require(isinstance(trials_value, list), f"Evaluation {model!r}/{split!r} lacks trials")
            yield model, split, cast(Sequence[Mapping[str, Any]], trials_value)


def classify_trials(pairs: Sequence[Mapping[str, Any]], evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Join sealed pairs to stored evaluation scores; no model inference is performed."""
    by_id: dict[str, Mapping[str, Any]] = {}
    for pair in pairs:
        _validate_pair(pair)
        pair_id = str(pair["pair_id"])
        _require(pair_id not in by_id, f"Duplicate pair_id {pair_id!r}")
        by_id[pair_id] = pair

    rows: list[dict[str, Any]] = []
    for model, split, trials in _iter_model_splits(evaluation):
        seen: set[str] = set()
        expected = {pair_id for pair_id, pair in by_id.items() if pair["split"] == split
                    and any(len(_history(pair, str(world))) >= 2 for world in pair["worlds"])}
        for trial in trials:
            _require(isinstance(trial, Mapping), f"Malformed trial in {model!r}/{split!r}")
            pair_id_value = trial.get("pair_id")
            _require(isinstance(pair_id_value, str) and pair_id_value in by_id,
                     f"Trial in {model!r}/{split!r} references an unknown pair")
            pair_id = cast(str, pair_id_value)
            _require(pair_id not in seen, f"Duplicate trial for {model!r}/{split!r}/{pair_id!r}")
            seen.add(pair_id)
            pair = by_id[pair_id]
            _require(pair["split"] == split and trial.get("split") == split,
                     f"Trial split mismatch for {pair_id!r}")
            family = pair["abstract_counterfactual_variable"]["state_family"]
            _require(trial.get("state_family") == family, f"Trial state-family mismatch for {pair_id!r}")
            distance = pair.get("metadata", {}).get("distance_bucket")
            _require(isinstance(distance, str) and trial.get("distance") == distance,
                     f"Trial distance mismatch for {pair_id!r}")
            candidates = list(pair["candidates"])
            for world_name in pair["worlds"]:
                history = _history(pair, str(world_name))
                if len(history) < 2:
                    continue
                scores, selected = _normal_scores(trial, str(world_name), candidates)
                current = pair["worlds"][world_name].get("correct_candidate")
                _require(current in candidates, f"World {world_name!r} has an invalid current candidate")
                alternatives = [candidate for candidate in candidates if candidate != current]
                _require(bool(alternatives), f"World {world_name!r} has no alternative candidate")
                selected_alternative = max(alternatives, key=lambda candidate: scores[candidate])
                category, aliases = classify_world_choice(pair, str(world_name), selected)
                rows.append({
                    "model": model, "split": split, "pair_id": pair_id, "world": str(world_name),
                    "state_family": family, "transition_depth": len(history),
                    "transition_depth_bucket": depth_bucket(len(history)), "distance": distance,
                    "selected_candidate": selected, "current_candidate": current,
                    "selected_alternative_candidate": selected_alternative,
                    "category": category, "category_aliases": aliases,
                    "signed_lm_margin_current_minus_selected_alternative":
                        scores[str(current)] - scores[selected_alternative],
                })
        _require(seen >= expected,
                 f"Evaluation {model!r}/{split!r} omits multi-transition pairs: {sorted(expected - seen)}")
    _require(bool(rows), "Evaluation contains no multi-transition world choices")
    return rows


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate taxonomy results over the complete requested grouping key."""
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    fields = ("model", "split", "state_family", "transition_depth_bucket", "distance")
    for row in rows:
        groups[tuple(str(row[field]) for field in fields)].append(row)
    output = []
    for key, selected in sorted(groups.items()):
        total = len(selected)
        counts = {category: sum(row["category"] == category for row in selected) for category in CATEGORIES}
        alias_counts = {
            category: sum(category in row["category_aliases"] for row in selected) for category in CATEGORIES
        }
        margins = [float(row["signed_lm_margin_current_minus_selected_alternative"]) for row in selected]
        output.append({
            **dict(zip(fields, key, strict=True)), "choice_count": total,
            "counts": counts,
            "percentages": {category: counts[category] * 100.0 / total for category in CATEGORIES},
            "alias_counts": alias_counts,
            "alias_percentages": {category: alias_counts[category] * 100.0 / total for category in CATEGORIES},
            "mean_signed_lm_margin_current_minus_selected_alternative": mean(margins),
        })
    return output


def build_taxonomy(pairs: Sequence[Mapping[str, Any]], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    rows = classify_trials(pairs, evaluation)
    return {
        "taxonomy": "dynamic-stale-state-baseline-v1", "read_only": True,
        "category_precedence": list(CATEGORIES[:-1]), "choices": rows,
        "groups": aggregate_rows(rows),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TaxonomyValidationError(f"Invalid JSON in {path} line {line_number}") from error
            _require(isinstance(record, dict), f"Non-object record in {path} line {line_number}")
            records.append(record)
    _require(bool(records), f"Source file is empty: {path}")
    return records


def verify_hash(path: Path, expected: str, description: str) -> str:
    _require(path.is_file(), f"Missing {description}: {path}")
    observed = sha256_file(path)
    _require(observed == expected, f"Hash mismatch for {description}: {path}")
    return observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True,
                        help="Sealed counterfactual JSONL; repeat for multiple splits")
    parser.add_argument("--source-sha256", action="append", required=True,
                        help="Expected hash corresponding to each --source")
    parser.add_argument("--evaluation", type=Path, required=True,
                        help="Existing counterfactual evaluator results JSON")
    parser.add_argument("--evaluation-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Explicit destination for the derived taxonomy JSON")
    args = parser.parse_args()
    _require(len(args.source) == len(args.source_sha256),
             "Each --source requires one --source-sha256")
    inputs = [*args.source, args.evaluation]
    _require(args.output not in inputs, "Output must not overwrite an input")
    pairs: list[dict[str, Any]] = []
    source_hashes = {}
    for path, expected in zip(args.source, args.source_sha256, strict=True):
        source_hashes[str(path)] = verify_hash(path, expected, "sealed source")
        pairs.extend(load_jsonl(path))
    evaluation_hash = verify_hash(args.evaluation, args.evaluation_sha256, "sealed evaluation")
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    _require(isinstance(evaluation, dict), "Evaluation root must be an object")
    result = build_taxonomy(pairs, evaluation)
    result["provenance"] = {
        "source_sha256": source_hashes,
        "evaluation": {"path": str(args.evaluation), "sha256": evaluation_hash},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, result)
    print(json.dumps({"output": str(args.output), "choices": len(result["choices"])}, indent=2))


if __name__ == "__main__":
    main()
