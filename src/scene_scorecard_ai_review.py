"""Validate and aggregate issue #41's retained AI judgments; never generate labels."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from src.continuation_sft import verify_file
from src.continuity_curriculum.common import (
    load_jsonl, normalized_words, sha256_bytes, sha256_file, write_json_atomic,
)
from src.scene_scorecard import (
    FACT_LABELS, HUMAN_FIELDS, YES_NO_UNCERTAIN, _git_metadata,
    build_review_packet, load_protocol, require, validate_matrix, wilson_interval,
)


RUN_ROOT = Path("runs/fictionpulper-50m-data30m-continuation-sft-v1")
REVIEW_ROOT = Path("runs/scene-sol-ai-review-v1")
PROCEDURE = Path("docs/scene-scorecard-sol-ai-evaluation-v1.md")
PROCEDURE_HASH = "58d216bafb230f2dd870c5e6da3a4aa96dc9aa8e2d7ae2a3f68c824b793cbe4a"
PACKET_HASH = "69ce3baf7efd378fcc3f7cfa94e6838287bfc89ab1c11918f4575d11c3b78a5b"
DECISIONS_HASH = "83fcde9528a77f912e83d2ab0fd6e71fd5cb60fa4b11196503836c6d5c0035fc"
KEY_HASH = "3ef261147a5ed9cdc27b0b4c3357e3a45b9bdd6482b1afd2b3473ade193ce86c"
CORRECTION_HASH = "3521d0af111b8ebfcd151bf9b21ac0617913830e7a236a84a7b1c2ec483d163f"
POST_PROTOCOL = Path("experiments/fictionpulper-50m-data30m-continuation-sft-v1/post-selection-protocol.json")
POST_HASH = "f45041dcfb87e4f3dea8c0abce988beb346a8d77991305650cd8e46bebed0b87"
SOURCE_REPORT_HASH = "8f84a415bfa200aba250db660638bf471222190bfa9c3e4d70cfa9f40fcb7cdd"
CANDIDATE_HASH = "ee91eb1c1ddffa0795e672797731f72c4959062e76904cd11bebdc79386aa767"
DESIRED = {field: "no" if field in ("contradiction_present", "loop_or_repetition_present")
           else "yes" for field in HUMAN_FIELDS}


def validate_decisions(
    packet: Sequence[dict[str, Any]], artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate complete AI-only provenance, coverage, bindings and evidence."""
    require(set(artifact) == {
        "schema", "evaluation_kind", "evaluator", "packet_sha256", "procedure_sha256",
        "procedure_freeze_commit", "private_mapping_used_for_labels", "decisions",
    }, "AI artifact fields differ from contract")
    require(artifact["schema"] == "scene-sol-ai-decisions-v1"
            and artifact["evaluation_kind"] == "ai_only_exploratory", "AI schema mismatch")
    require(artifact["evaluator"] == {
        "name": "GPT-6.1 Sol", "model_id": "openai/gpt-6.1-sol",
        "execution": "OpenCode conversation; no external model API",
        "independent": False, "prior_aggregate_exposure": True,
    }, "AI evaluator must not impersonate human review or independence")
    require(artifact["private_mapping_used_for_labels"] is False, "labels used private mapping")
    require(artifact["packet_sha256"] == PACKET_HASH
            and artifact["procedure_sha256"] == PROCEDURE_HASH
            and artifact["procedure_freeze_commit"] == "cc5fb4c", "AI artifact binding mismatch")
    by_id = {row["review_id"]: row for row in packet}
    require(len(packet) == len(by_id) == 80, "expected 80 unique packet rows")
    rows = artifact["decisions"]
    require(isinstance(rows, list) and len(rows) == 80, "AI decisions incomplete")
    seen: set[str] = set()
    validated = []
    for row in rows:
        require(set(row) == {
            "review_id", "generated_text_sha256", "atomic_fact_labels",
            *HUMAN_FIELDS, "evidence", "reviewer_note",
        }, "AI row fields differ from contract")
        identifier = row["review_id"]
        require(identifier in by_id and identifier not in seen, "unknown or duplicate AI row")
        seen.add(identifier)
        original = by_id[identifier]
        text = original["generated_text"]
        require(row["generated_text_sha256"] == original["generated_text_sha256"]
                == sha256_bytes(text.encode("utf-8")), "AI text binding mismatch")
        fact_ids = {fact["id"] for fact in original["atomic_facts"]}
        labels = row["atomic_fact_labels"]
        require(isinstance(labels, dict) and set(labels) == fact_ids
                and set(labels.values()) <= FACT_LABELS, "invalid AI fact labels")
        require(all(row[field] in YES_NO_UNCERTAIN for field in HUMAN_FIELDS), "invalid AI dimension")
        evidence = row["evidence"]
        require(isinstance(evidence, dict) and set(evidence) == fact_ids | set(HUMAN_FIELDS),
                "AI evidence incomplete")
        for field, item in evidence.items():
            require(isinstance(item, dict) and set(item) == {"quote", "note"}, "invalid evidence fields")
            quote, note = item["quote"], item["note"]
            require(isinstance(note, str) and bool(note.strip()), "evidence note required")
            require(isinstance(quote, str) and bool(quote.strip()), "evidence quote required")
            require(quote in {"<ABSENT>", "<EMPTY_OUTPUT>"} or quote in text,
                    "invented AI evidence quotation")
            require(quote != "<EMPTY_OUTPUT>" or text == "", "empty marker on nonempty output")
            require(text != "" or quote == "<EMPTY_OUTPUT>", "empty output needs empty marker")
            positive = labels.get(field) in {"retained", "contradicted"} or row.get(field) == "yes"
            require(not positive or quote not in {"<ABSENT>", "<EMPTY_OUTPUT>"},
                    "affirmative judgment needs verbatim evidence")
        require(isinstance(row["reviewer_note"], str) and bool(row["reviewer_note"].strip()),
                "AI row rationale required")
        count = len(normalized_words(text))
        require(count == original["word_count"]
                and original["length_compliant"] is (150 <= count <= 250), "packet length mismatch")
        validated.append({**row, "word_count": count, "length_compliant": 150 <= count <= 250})
    require(seen == set(by_id), "AI review coverage mismatch")
    return validated


def proportion(successes: int, total: int) -> dict[str, Any]:
    return {"numerator": successes, "denominator": total, "rate": successes / total,
            "interval": wilson_interval(successes, total)}


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    require(bool(rows), "cannot summarize empty stratum")
    count = len(rows)
    fact_counts = Counter(label for row in rows for label in row["atomic_fact_labels"].values())
    dimensions = {}
    for field, desired in DESIRED.items():
        counts = Counter(row[field] for row in rows)
        dimensions[field] = {
            "desired_label": desired, **proportion(counts[desired], count),
            "label_counts": {value: counts[value] for value in sorted(YES_NO_UNCERTAIN)},
        }
    return {
        "sample_size": {"outputs": count, "atomic_facts": count * 4},
        "fact_retention": proportion(fact_counts["retained"], count * 4),
        "fact_label_counts": {value: fact_counts[value] for value in sorted(FACT_LABELS)},
        "coherent_outputs": proportion(sum(row["coherent"] for row in rows), count),
        "length_compliant": proportion(sum(row["length_compliant"] for row in rows), count),
        "ai_dimensions": dimensions,
        "unresolved_outputs": sum(any(value == "uncertain" for value in
                                      [*row["atomic_fact_labels"].values(), *(row[f] for f in HUMAN_FIELDS)])
                                  for row in rows),
        "unresolved_judgments": fact_counts["uncertain"] + sum(
            dimensions[field]["label_counts"]["uncertain"] for field in HUMAN_FIELDS),
    }


def add_scores(rows: Sequence[dict[str, Any]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        retained = sum(value == "retained" for value in row["atomic_fact_labels"].values())
        coherent = (retained >= 3 and row["length_compliant"]
                    and all(row[field] == desired for field, desired in DESIRED.items()))
        scored.append({**row, **metadata[row["review_id"]],
                       "retained_facts": retained, "coherent": bool(coherent)})
    return scored


def paired_difference(
    base: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]], *,
    replicates: int, seed: int,
) -> dict[str, Any]:
    """Resample matched prompt clusters, retaining all four trials per prompt."""
    def pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return row["prompt_id"], row["generation_mode"], row["sampling_seed"]

    base_by_key = {pair_key(row): row for row in base}
    candidate_by_key = {pair_key(row): row for row in candidate}
    require(set(base_by_key) == set(candidate_by_key) and len(base_by_key) == len(base) == len(candidate),
            "paired coverage mismatch")
    prompts = sorted({row["prompt_id"] for row in base})
    measures = {"fact_retention": lambda r: r["retained_facts"] / 4,
                "coherent_outputs": lambda r: float(r["coherent"]),
                "length_compliant": lambda r: float(r["length_compliant"])}
    measures.update({field: (lambda r, f=field, d=desired: float(r[f] == d))
                     for field, desired in DESIRED.items()})
    clusters = {name: [sum(measure(candidate_by_key[k]) - measure(base_by_key[k])
                           for k in base_by_key if k[0] == prompt) / 4
                       for prompt in prompts] for name, measure in measures.items()}
    draws = {name: [] for name in measures}
    rng = random.Random(seed)
    for _ in range(replicates):
        indices = [rng.randrange(len(prompts)) for _ in prompts]
        for name, values in clusters.items():
            draws[name].append(sum(values[index] for index in indices) / len(prompts))

    def percentile(values: list[float], fraction: float) -> float:
        values = sorted(values)
        position = (len(values) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (position - low) * (values[high] - values[low])

    return {
        "method": "paired prompt-cluster percentile bootstrap; desired-label rates",
        "replicates": replicates, "seed": seed, "matched_outputs": len(base),
        "prompt_clusters": len(prompts), "confidence_level": 0.95,
        "metrics": {name: {"candidate_minus_base": sum(values) / len(prompts),
                           "lower": percentile(draws[name], 0.025),
                           "upper": percentile(draws[name], 0.975)}
                    for name, values in clusters.items()},
        "limitation": "Conditional on ten fixed prompt clusters and one exposed AI evaluator; degenerate intervals do not prove equivalence or certainty.",
    }


def score(reviews: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), "refusing to overwrite AI Scene report")
    hashes = {"procedure": verify_file(PROCEDURE, PROCEDURE_HASH, "AI procedure"),
              "ai_decisions": verify_file(reviews, DECISIONS_HASH, "locked AI decisions")}
    correction_dir = RUN_ROOT / "post-selection-correction"
    packet_path = correction_dir / "blinded-review-packet.jsonl"
    hashes["packet"] = verify_file(packet_path, PACKET_HASH, "corrected Scene packet")
    packet = load_jsonl(packet_path)
    artifact = json.loads(reviews.read_text(encoding="utf-8"))
    validated = validate_decisions(packet, artifact)
    # Only validated, hash-locked decisions reach private source/mapping access.
    hashes["post_selection_protocol"] = verify_file(POST_PROTOCOL, POST_HASH, "post-selection protocol")
    post = json.loads(POST_PROTOCOL.read_text(encoding="utf-8"))
    protocol, prompts, protocol_hashes = load_protocol(Path(post["scene_protocol_root"]))
    require(protocol_hashes["protocol_sha256"] == post["scene_protocol_sha256"], "Scene lock mismatch")
    hashes.update(protocol_hashes)
    hashes["source_report"] = verify_file(RUN_ROOT / "post-selection/report.json", SOURCE_REPORT_HASH, "source report")
    source_report = json.loads((RUN_ROOT / "post-selection/report.json").read_text(encoding="utf-8"))
    hashes["correction_report"] = verify_file(correction_dir / "correction-report.json", CORRECTION_HASH, "correction report")
    correction = json.loads((correction_dir / "correction-report.json").read_text(encoding="utf-8"))
    hashes["private_key"] = verify_file(correction_dir / "private-review-key.json", KEY_HASH, "corrected private key")
    key = json.loads((correction_dir / "private-review-key.json").read_text(encoding="utf-8"))
    matrices = {}
    for model, path, expected in (
        ("base", Path(post["base_scene_raw"]["path"]), post["base_scene_raw"]["sha256"]),
        ("candidate", RUN_ROOT / "post-selection/scene-scorecard/candidate-raw.jsonl", CANDIDATE_HASH),
    ):
        hashes[f"{model}_raw"] = verify_file(path, expected, f"{model} raw matrix")
        matrices[model] = load_jsonl(path)
        validate_matrix(matrices[model], prompts, protocol)
    combined = [{**row, "generation_id": f"{model}:{row['generation_id']}"}
                for model in ("base", "candidate") for row in matrices[model]]
    rebuilt, metadata = build_review_packet(combined, seed=protocol["review"]["blinded_packet_seed"])
    require(rebuilt == packet and metadata == key["review_metadata"], "corrected packet/source/key mismatch")
    mapping = {identifier: row["generation_id"].split(":", 1)[0] for identifier, row in metadata.items()}
    require(mapping == key["model_by_review_id"], "model mapping mismatch")
    require(key["model_identities"] == {"base": protocol["baseline"]["checkpoint_sha256"],
                                        "candidate": source_report["selected_checkpoint_sha256"]}, "checkpoint identity mismatch")
    require(correction["corrected_packet_sha256"] == PACKET_HASH
            and correction["corrected_key_sha256"] == KEY_HASH
            and correction["source_candidate_raw_sha256"] == CANDIDATE_HASH
            and correction["source_report_sha256"] == SOURCE_REPORT_HASH, "correction bindings mismatch")
    scored = add_scores(validated, metadata)
    models = {}
    for model in ("base", "candidate"):
        rows = [row for row in scored if mapping[row["review_id"]] == model]
        require(len(rows) == 40, "model coverage incomplete")
        summary = summarize(rows)
        strata = []
        for fields in (("selection_role",), ("generation_mode", "sampling_seed"),
                       ("selection_role", "generation_mode", "sampling_seed"), ("prompt_id",)):
            groups = {tuple(row[field] for field in fields) for row in rows}
            for group in sorted(groups, key=str):
                selected = [row for row in rows if tuple(row[field] for field in fields) == group]
                strata.append({"group_by": list(fields), "values": dict(zip(fields, group)), **summarize(selected)})
        genres = {row["prompt_id"]: row["genre"] for row in prompts}
        for genre in sorted(set(genres.values())):
            strata.append({"group_by": ["genre"], "values": {"genre": genre},
                           **summarize([row for row in rows if genres[row["prompt_id"]] == genre])})
        gates = protocol["success_gates"]["suite_promotion"]
        gate_results = {
            "fact_retention": summary["fact_retention"]["rate"] >= gates["fact_retention_rate_minimum"],
            "premise_adherence": summary["ai_dimensions"]["premise_adherence"]["rate"] >= gates["premise_adherence_rate_minimum"],
            "coherent_output_rate": summary["coherent_outputs"]["rate"] >= gates["coherent_output_rate_minimum"],
            "per_prompt_coherent_output_rate": all(
                sum(r["coherent"] for r in rows if r["prompt_id"] == p["prompt_id"]) / 4
                >= gates["per_prompt_coherent_output_rate_minimum"] for p in prompts),
        }
        raw = matrices[model]
        models[model] = {**summary, "strata": strata, "ai_adapted_gate_results": gate_results,
                         "ai_adapted_suite_pass": all(gate_results.values()),
                         "mechanical_length_coherence_upper_bound": summary["length_compliant"]["rate"],
                         "mechanical_length_by_stop": dict(Counter(
                             f"{row['diagnostics']['length_classification']}:{row['stop_reason']}" for row in raw)),
                         "automated_diagnostics": {
                             "exact_fact_phrase_mentions": sum(sum(r["diagnostics"]["exact_atomic_fact_phrase_mentions"].values()) for r in raw),
                             "fact_denominator": 160,
                             "mean_repeated_4gram_rate": sum(r["diagnostics"]["repetition"]["repeated_4gram_rate"] for r in raw) / 40,
                             "mean_repeated_sentence_rate": sum(r["diagnostics"]["repetition"]["repeated_sentence_rate"] for r in raw) / 40,
                         }}
    by_model = {model: [row for row in scored if mapping[row["review_id"]] == model]
                for model in ("base", "candidate")}
    require(sha256_file(reviews) == DECISIONS_HASH, "decisions changed during scoring")
    hashes["ai_scorer"] = sha256_file(Path(__file__))
    report = {
        "status": "COMPLETE_AI_ONLY_EXPLORATORY_EVIDENCE", "evaluator": artifact["evaluator"],
        "independent_human_review_complete": False, "human_reviewer_assignments": [],
        "agreement": {"status": "not_available"}, "adjudication": {"status": "not_performed", "count": 0},
        "decision": "STOP_NO_PROMOTION", "showcase": None,
        "models": models, "paired_difference": paired_difference(
            by_model["base"], by_model["candidate"], replicates=protocol["review"]["bootstrap_replicates"],
            seed=protocol["review"]["bootstrap_seed"]),
        "checkpoint_identities": key["model_identities"], "hashes": hashes,
        "provenance": _git_metadata(Path(__file__).resolve().parents[1]),
        "outputs": [{**row, "model": mapping[row["review_id"]]} for row in scored],
        "limitations": [
            "One owner-authorized AI evaluator; prior aggregates were visible, so not independent or uncontaminated.",
            "Semantic AI scores, lexical diagnostics, mechanical length and historical likelihood are distinct evidence.",
            "Wilson intervals are descriptive; atomic facts within outputs are dependent, and small genre/trial cells are underpowered.",
            "Unknown labels fail desired-label gates; unresolved judgments are preserved without simulated adjudication.",
            "No matched LM-only update control: base/candidate differences do not isolate SFT causality.",
            "Inspected post-selection outputs remain retrospective evidence, never new training or selection data.",
            "No promotion of #12 follows from an AI-adapted score; historical independent human review remains pending.",
        ],
    }
    write_json_atomic(output, report)
    print(json.dumps({"status": report["status"], "decision": report["decision"],
                      "report_sha256": sha256_file(output),
                      "models": {model: {k: v for k, v in summary.items() if k != "strata"}
                                 for model, summary in models.items()},
                      "paired_difference": report["paired_difference"]}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, default=REVIEW_ROOT / "decisions.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score(args.reviews, args.output)


if __name__ == "__main__":
    main()
