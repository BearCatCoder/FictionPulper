"""Diagnostic and blinded-human scoring for Narrative Benchmark v2 generations."""

from __future__ import annotations

import math
import random
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from src.continuity_curriculum.common import canonical_json, sha256_bytes


LABELS = ("consistent", "inconsistent", "uncertain")
REVIEW_CONTENT_FIELDS = (
    "review_id", "blinded_model_id", "generated_text_sha256", "prompt",
    "generated_text", "review_question", "required_propositions", "forbidden_propositions",
)
CAUSAL_WORDS = {
    "because", "chain", "consequently", "follows", "hence", "leads", "resolved",
    "result", "therefore", "thus", "yields",
}
STATE_RELATION_WORDS = {
    "acts", "as", "became", "becomes", "is", "means", "remains", "resolved",
    "result", "terminal", "terminates", "was", "yields",
}
NEGATIONS = {"isn't", "never", "no", "not", "wasn't", "without", "wrong"}
STATE_TERMS = {
    "static_fact": "annotation",
    "ownership": "titleholder",
    "location": "destination",
    "knowledge": "recipient",
    "goal": "objective",
    "causal_dependency": "trigger",
    "temporal_ordering": "precedence",
}


class ReviewValidationError(ValueError):
    """Raised when review data cannot support the frozen scoring protocol."""


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.casefold())


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase_words = _words(phrase)
    if not phrase_words:
        return False
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(word) for word in phrase_words) + r"(?!\w)"
    return bool(re.search(pattern, text.casefold()))


def _mention_state(text: str, value: str) -> dict[str, bool]:
    words = _words(text)
    value_words = _words(value)
    if not value_words:
        return {"mentioned": False, "negated": False}
    mentioned = False
    negated = False
    width = len(value_words)
    for index in range(len(words) - width + 1):
        if words[index : index + width] != value_words:
            continue
        mentioned = True
        preceding = words[max(0, index - 2) : index]
        following = words[index + width : index + width + 3]
        negated = negated or (
            bool(preceding) and preceding[-1] in NEGATIONS
        ) or (
            len(following) >= 2
            and following[0] in {"is", "means", "was"}
            and following[1] in NEGATIONS
        )
    return {"mentioned": mentioned, "negated": negated}


def _asserted_state(
    text: str,
    value: str,
    *,
    entity_names: Sequence[str],
    state_term: str,
    relation_words: set[str] = STATE_RELATION_WORDS,
) -> bool:
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence_words = set(_words(sentence))
        mention = _mention_state(sentence, value)
        if (
            mention["mentioned"]
            and not mention["negated"]
            and state_term in sentence_words
            and any(_contains_phrase(sentence, name) for name in entity_names)
            and bool(sentence_words & relation_words)
        ):
            return True
    return False


def repetition_diagnostics(text: str, token_ids: Sequence[int]) -> dict[str, float | int]:
    ids = list(token_ids)
    result: dict[str, float | int] = {"token_count": len(ids)}
    for width in (1, 2, 3):
        spans = [tuple(ids[index : index + width]) for index in range(max(0, len(ids) - width + 1))]
        result[f"distinct_{width}"] = len(set(spans)) / len(spans) if spans else 0.0
    four_grams = [tuple(ids[index : index + 4]) for index in range(max(0, len(ids) - 3))]
    result["repeated_4gram_rate"] = (
        (len(four_grams) - len(set(four_grams))) / len(four_grams) if four_grams else 0.0
    )
    sentences = [
        re.sub(r"\s+", " ", sentence.strip().casefold())
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    sentence_counts = Counter(sentences)
    result["sentence_count"] = len(sentences)
    result["repeated_sentence_rate"] = (
        sum(count - 1 for count in sentence_counts.values()) / len(sentences)
        if sentences else 0.0
    )
    return result


def diagnostic_score(record: dict[str, Any]) -> dict[str, Any]:
    """Return conservative proposition diagnostics, never a semantic primary score."""
    text = str(record["generated_text"])
    current = _mention_state(text, str(record["answer_value"]))
    opposing = _mention_state(text, str(record["opposing_answer_value"]))
    entity_names = [str(name) for name in record.get("entity_names", [])]
    state_term = STATE_TERMS[str(record["state_family"])]
    stale = {
        str(value): {
            **_mention_state(text, str(value)),
            "asserted": _asserted_state(
                text, str(value), entity_names=entity_names, state_term=state_term
            ),
        }
        for value in record.get("stale_values", [])
    }
    initial = _mention_state(text, str(record["initial_value"]))
    opposing_initial = _mention_state(text, str(record["opposing_initial_value"]))
    entity_mentions = {
        name: _contains_phrase(text, name) for name in entity_names
    }
    words = set(_words(text))
    relation_anchored = state_term in words and any(entity_mentions.values())
    current_asserted = _asserted_state(
        text, str(record["answer_value"]), entity_names=entity_names, state_term=state_term
    )
    opposing_asserted = _asserted_state(
        text, str(record["opposing_answer_value"]), entity_names=entity_names,
        state_term=state_term,
    )
    current_only = current_asserted and not opposing_asserted
    contradiction = opposing_asserted or (current["mentioned"] and current["negated"] and relation_anchored)
    depth = int(record["reasoning_depth"])
    causal_marker = bool(words & CAUSAL_WORDS)
    causal_assertion = _asserted_state(
        text, str(record["answer_value"]), entity_names=entity_names,
        state_term=state_term, relation_words=CAUSAL_WORDS,
    )
    stale_mentioned = any(item["asserted"] for item in stale.values())
    return {
        "diagnostic_only": True,
        "named_entity_retention": entity_mentions,
        "named_entity_retention_rate": (
            sum(entity_mentions.values()) / len(entity_mentions) if entity_mentions else None
        ),
        "current_fact": current,
        "stale_or_counterfactual_fact": opposing,
        "state_relation_term": state_term,
        "state_relation_anchored": relation_anchored,
        "current_fact_asserted": current_asserted,
        "counterfactual_fact_asserted": opposing_asserted,
        "stale_facts": stale,
        "stale_fact_mentioned": stale_mentioned,
        "current_vs_stale": (
            "current_and_stale" if current["mentioned"] and stale_mentioned
            else "current_only" if current["mentioned"]
            else "stale_only" if stale_mentioned
            else "neither"
        ),
        "current_fact_only": current_only,
        "explicit_contradiction": contradiction,
        "premise": {
            "initial_value": initial,
            "opposing_initial_value": opposing_initial,
            "retained_without_counterfactual": (
                _asserted_state(
                    text, str(record["initial_value"]), entity_names=entity_names,
                    state_term=state_term,
                )
                and not _asserted_state(
                    text, str(record["opposing_initial_value"]), entity_names=entity_names,
                    state_term=state_term,
                )
            ),
        },
        "causal_adherence": {
            "causal_marker_present": causal_marker,
            "causal_relation_with_terminal_fact": causal_assertion,
            "terminal_fact_present": current_only,
            "diagnostic_pass": current_only and (depth == 1 or causal_assertion),
        },
        "repetition": repetition_diagnostics(text, record.get("generated_token_ids", [])),
    }


def aggregate_diagnostics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        diagnostic = record.get("diagnostics") or diagnostic_score(record)
        for dimension, value in (
            ("overall", "all"),
            ("split", str(record["split"])),
            ("family", str(record["state_family"])),
            ("depth", str(record["depth_bucket"])),
            ("reasoning_depth", str(record["reasoning_depth"])),
            ("family_depth", f"{record['state_family']}:{record['depth_bucket']}"),
        ):
            groups[(str(record["blinded_model_id"]), dimension, value, _trial_key(record))].append(diagnostic)
    strata = []
    for (model_id, dimension, value, trial), items in sorted(groups.items()):
        rates = {}
        for field in ("current_fact_only", "explicit_contradiction"):
            rates[field + "_rate"] = sum(bool(item[field]) for item in items) / len(items)
        rates["causal_adherence_rate"] = sum(
            bool(item["causal_adherence"]["diagnostic_pass"]) for item in items
        ) / len(items)
        entity_rates = [
            float(item["named_entity_retention_rate"])
            for item in items if item["named_entity_retention_rate"] is not None
        ]
        repetition = [float(item["repetition"]["repeated_4gram_rate"]) for item in items]
        repeated_sentences = [float(item["repetition"]["repeated_sentence_rate"]) for item in items]
        strata.append({
            "blinded_model_id": model_id,
            "dimension": dimension,
            "stratum": value,
            "trial": trial,
            "generation_count": len(items),
            **rates,
            "mean_named_entity_retention_rate": statistics.fmean(entity_rates) if entity_rates else None,
            "stale_fact_incidence_rate": sum(bool(item["stale_fact_mentioned"]) for item in items) / len(items),
            "premise_retention_rate": sum(
                bool(item["premise"]["retained_without_counterfactual"]) for item in items
            ) / len(items),
            "mean_repeated_4gram_rate": statistics.fmean(repetition),
            "mean_repeated_sentence_rate": statistics.fmean(repeated_sentences),
        })
    return {
        "status": "diagnostic_only",
        "primary_metric": "none; blinded human pair consistency is primary",
        "strata": strata,
    }


def _trial_key(record: dict[str, Any]) -> str:
    seed = record.get("sampling_seed")
    return "greedy" if record["generation_mode"] == "greedy" else f"sampled:{seed}"


def review_id(record: dict[str, Any]) -> str:
    payload = [record["generation_id"], record["generated_text_sha256"], record["blinded_model_id"]]
    return "review-" + sha256_bytes(canonical_json(payload).encode("utf-8"))[:20]


def build_review_packet(records: Sequence[dict[str, Any]], *, randomization_seed: int) -> list[dict[str, Any]]:
    packet = []
    for record in records:
        packet.append({
            "review_id": review_id(record),
            "blinded_model_id": record["blinded_model_id"],
            "generated_text_sha256": record["generated_text_sha256"],
            "prompt": record["prompt"],
            "generated_text": record["generated_text"],
            "review_question": record["review_question"],
            "required_propositions": record["required_propositions"],
            "forbidden_propositions": record["forbidden_propositions"],
            "reviewer_id": "",
            "label": "",
            "confidence": None,
            "evidence_quote": "",
            "reviewer_note": "",
            "adjudicated_label": None,
        })
    rng = random.Random(randomization_seed)
    rng.shuffle(packet)
    return packet


def validate_reviews(
    generations: Sequence[dict[str, Any]], reviews: Sequence[dict[str, Any]]
) -> dict[str, str]:
    generations_by_review = {review_id(record): record for record in generations}
    if len(generations_by_review) != len(generations):
        raise ReviewValidationError("generation review IDs are not unique")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reviews:
        identifier = str(row.get("review_id", ""))
        if identifier not in generations_by_review:
            raise ReviewValidationError(f"unknown review_id: {identifier}")
        generation = generations_by_review[identifier]
        for field in (
            "scenario_id", "world", "generation_mode", "sampling_seed",
            "blinded_model_id", "generated_text_sha256",
        ):
            if row.get(field) != generation.get(field):
                raise ReviewValidationError(f"{identifier}: {field} differs from generation")
        if row.get("label") not in LABELS:
            raise ReviewValidationError(f"{identifier}: invalid label")
        if row.get("adjudicated_label") not in (*LABELS, None):
            raise ReviewValidationError(f"{identifier}: invalid adjudicated_label")
        reviewer = str(row.get("reviewer_id", "")).strip()
        if not reviewer:
            raise ReviewValidationError(f"{identifier}: reviewer_id is required")
        if reviewer != row.get("reviewer_id"):
            raise ReviewValidationError(f"{identifier}: reviewer_id must not contain surrounding whitespace")
        confidence = row.get("confidence")
        if not isinstance(confidence, int) or isinstance(confidence, bool) or not 1 <= confidence <= 5:
            raise ReviewValidationError(f"{identifier}: confidence must be an integer from 1 to 5")
        evidence = str(row.get("evidence_quote", "")).strip()
        empty_output_evidence = not generation["generated_text"] and evidence == "<EMPTY_OUTPUT>"
        if not empty_output_evidence and (not evidence or evidence not in generation["generated_text"]):
            raise ReviewValidationError(f"{identifier}: evidence_quote must occur verbatim in generated text")
        if not str(row.get("reviewer_note", "")).strip():
            raise ReviewValidationError(f"{identifier}: reviewer_note is required")
        grouped[identifier].append(row)

    missing = sorted(set(generations_by_review) - set(grouped))
    if missing:
        raise ReviewValidationError(f"missing reviews for {len(missing)} generations")

    resolved: dict[str, str] = {}
    original_reviewer_pairs: set[tuple[str, str]] = set()
    for identifier, rows in grouped.items():
        originals = [row for row in rows if row["adjudicated_label"] is None]
        adjudications = [row for row in rows if row["adjudicated_label"] is not None]
        if len(originals) != 2 or len({row["reviewer_id"] for row in originals}) != 2:
            raise ReviewValidationError(f"{identifier}: exactly two independent original reviews required")
        reviewer_ids = sorted(str(row["reviewer_id"]) for row in originals)
        original_reviewer_pairs.add((reviewer_ids[0], reviewer_ids[1]))
        needs_adjudication = (
            originals[0]["label"] != originals[1]["label"]
            or originals[0]["label"] == "uncertain"
        )
        if needs_adjudication:
            if len(adjudications) != 1:
                raise ReviewValidationError(f"{identifier}: exactly one adjudication required")
            if adjudications[0]["reviewer_id"] in {row["reviewer_id"] for row in originals}:
                raise ReviewValidationError(f"{identifier}: adjudicator must be independent")
            if adjudications[0]["label"] != adjudications[0]["adjudicated_label"]:
                raise ReviewValidationError(f"{identifier}: adjudicator label fields disagree")
            resolved[identifier] = str(adjudications[0]["adjudicated_label"])
        else:
            if adjudications:
                raise ReviewValidationError(f"{identifier}: unnecessary adjudication supplied")
            resolved[identifier] = str(originals[0]["label"])
    if len(original_reviewer_pairs) != 1:
        raise ReviewValidationError("the same two original reviewers must score every generation")
    return resolved


def _cohen_kappa(label_pairs: Sequence[tuple[str, str]]) -> float | None:
    if not label_pairs:
        return None
    observed = sum(left == right for left, right in label_pairs) / len(label_pairs)
    left_counts = Counter(left for left, _ in label_pairs)
    right_counts = Counter(right for _, right in label_pairs)
    expected = sum(
        left_counts[label] * right_counts[label] for label in LABELS
    ) / (len(label_pairs) ** 2)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1.0 - expected)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take percentile of empty values")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_bootstrap_interval(
    pairs: Sequence[dict[str, Any]], *, replicates: int, seed: int
) -> dict[str, Any]:
    if not pairs or replicates <= 0:
        raise ValueError("paired bootstrap requires pairs and positive replicates")
    rng = random.Random(seed)
    rates = []
    for _ in range(replicates):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        rates.append(sum(item["strict_success"] for item in sample) / len(sample))
    rates.sort()
    return {
        "method": "percentile paired bootstrap",
        "replicates": replicates,
        "seed": seed,
        "confidence_level": 0.95,
        "lower": _percentile(rates, 0.025),
        "upper": _percentile(rates, 0.975),
    }


def _pair_rows(
    generations: Sequence[dict[str, Any]], resolved: dict[str, str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int | None], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in generations:
        key = (
            str(record["blinded_model_id"]), str(record["scenario_id"]),
            str(record["generation_mode"]), record.get("sampling_seed"),
        )
        world = str(record["world"])
        if world in grouped[key]:
            raise ReviewValidationError(
                f"{record['scenario_id']} {record['generation_mode']}/{record.get('sampling_seed')}: "
                f"duplicate world {world}"
            )
        grouped[key][world] = record
    pairs = []
    for (model_id, scenario_id, mode, seed), worlds in sorted(grouped.items()):
        if set(worlds) != {"A", "B"}:
            raise ReviewValidationError(f"{scenario_id} {mode}/{seed}: A/B generation pair is incomplete")
        labels = {world: resolved[review_id(record)] for world, record in worlds.items()}
        first = worlds["A"]
        unresolved = any(label == "uncertain" for label in labels.values())
        pairs.append({
            "blinded_model_id": model_id,
            "scenario_id": scenario_id,
            "generation_mode": mode,
            "sampling_seed": seed,
            "split": first["split"],
            "state_family": first["state_family"],
            "reasoning_depth": first["reasoning_depth"],
            "depth_bucket": first["depth_bucket"],
            "world_labels": labels,
            "strict_success": all(label == "consistent" for label in labels.values()),
            "unresolved_uncertain": unresolved,
        })
    return pairs


def _metric(pairs: Sequence[dict[str, Any]], *, replicates: int, seed: int) -> dict[str, Any]:
    resolved_pairs = [pair for pair in pairs if not pair["unresolved_uncertain"]]
    return {
        "pair_count": len(pairs),
        "successful_pair_count": sum(pair["strict_success"] for pair in pairs),
        "strict_pair_consistency_rate": sum(pair["strict_success"] for pair in pairs) / len(pairs),
        "unresolved_uncertain_pair_count": len(pairs) - len(resolved_pairs),
        "resolved_pair_count": len(resolved_pairs),
        "resolved_only_pair_consistency_rate": (
            sum(pair["strict_success"] for pair in resolved_pairs) / len(resolved_pairs)
            if resolved_pairs else None
        ),
        "strict_rate_interval": paired_bootstrap_interval(pairs, replicates=replicates, seed=seed),
    }


def score_human_reviews(
    generations: Sequence[dict[str, Any]],
    reviews: Sequence[dict[str, Any]],
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 260_209,
) -> dict[str, Any]:
    resolved = validate_reviews(generations, reviews)
    pairs = _pair_rows(generations, resolved)
    original_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reviews:
        if row["adjudicated_label"] is None:
            original_groups[row["review_id"]].append(row)
    label_pairs = []
    for _, rows in sorted(original_groups.items()):
        ordered = sorted(rows, key=lambda row: str(row["reviewer_id"]))
        label_pairs.append((ordered[0]["label"], ordered[1]["label"]))
    strata: dict[tuple[str, str, int | None, str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        trial = "greedy" if pair["generation_mode"] == "greedy" else "sampled"
        seed = pair["sampling_seed"]
        for dimension, value in (
            ("overall", "all"),
            ("split", pair.get("split", "unknown")),
            ("family", pair["state_family"]),
            ("depth", pair["depth_bucket"]),
            ("reasoning_depth", pair.get("reasoning_depth", "unknown")),
            ("family_depth", f"{pair['state_family']}:{pair['depth_bucket']}"),
        ):
            strata[(pair["blinded_model_id"], trial, seed, dimension, str(value))].append(pair)
    metrics = []
    for (model_id, mode, sample_seed, dimension, value), stratum_pairs in sorted(
        strata.items(), key=lambda item: tuple(str(part) for part in item[0])
    ):
        metrics.append({
            "blinded_model_id": model_id,
            "generation_mode": mode,
            "sampling_seed": sample_seed,
            "dimension": dimension,
            "stratum": value,
            **_metric(stratum_pairs, replicates=bootstrap_replicates, seed=bootstrap_seed),
        })

    sampled_summary = []
    sampled_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        if metric["generation_mode"] == "sampled":
            sampled_groups[(metric["blinded_model_id"], metric["dimension"], metric["stratum"])].append(metric)
    for (model_id, dimension, value), seed_metrics in sorted(sampled_groups.items()):
        seed_metrics.sort(key=lambda item: item["sampling_seed"])
        values = [item["strict_pair_consistency_rate"] for item in seed_metrics]
        sampled_summary.append({
            "blinded_model_id": model_id,
            "dimension": dimension,
            "stratum": value,
            "seed_count": len(values),
            "sampling_seeds": [item["sampling_seed"] for item in seed_metrics],
            "strict_rate_mean": statistics.fmean(values),
            "strict_rate_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    return {
        "status": "complete",
        "primary_metric": "blinded_human_pair_consistency",
        "generation_count": len(generations),
        "scenario_pair_trial_count": len(pairs),
        "review_count": len(reviews),
        "agreement": {
            "original_review_pair_count": len(label_pairs),
            "raw_agreement": sum(left == right for left, right in label_pairs) / len(label_pairs),
            "cohen_kappa": _cohen_kappa(label_pairs),
        },
        "metrics": metrics,
        "sampled_seed_summary": sampled_summary,
    }
