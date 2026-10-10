"""Generate and score the frozen FictionPulper scene scorecard v1."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence, cast

import torch
from tokenizers import Tokenizer

from src.continuity_curriculum.common import (
    canonical_json,
    load_jsonl,
    normalized_words,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
    write_jsonl,
)
from src.model import FictionPulperLM, ModelConfig
from src.narrative_v2_baseline import TOKENIZER_SHA256, verify_sealed_model
from src.narrative_v2_free_generation import generate_continuation
from src.narrative_v2_scoring import repetition_diagnostics


PROTOCOL_SHA256 = "7c438ad612d7c344f17ab32843f230e49ef3194565fc0aff45a4a06cb41f2718"
RUNNER_VERSION = "scene-scorecard-v1-1"
DEFAULT_PROTOCOL_ROOT = Path("benchmarks/scene-scorecard-v1")
DEFAULT_OUTPUT_DIR = Path("runs/scene-scorecard-v1/fictionpulper-50m-data30m-v1")
YES_NO_UNCERTAIN = {"yes", "no", "uncertain"}
FACT_LABELS = {"retained", "not_retained", "contradicted", "uncertain"}
HUMAN_FIELDS = (
    "premise_adherence",
    "motive_preservation",
    "contradiction_present",
    "loop_or_repetition_present",
    "plot_advancement",
    "genre_voice",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_protocol(protocol_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    protocol_path = protocol_root / "protocol.json"
    require(protocol_path.is_file(), f"missing protocol: {protocol_path}")
    protocol_hash = sha256_file(protocol_path)
    require(protocol_hash == PROTOCOL_SHA256, "frozen scene protocol hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol.get("protocol_id") == "fictionpulper-scene-scorecard-v1", "protocol ID changed")
    require(protocol.get("protocol_version") == 1, "protocol version changed")
    hashes = {"protocol_sha256": protocol_hash}
    for name in ("prompts", "human_review_rubric"):
        specification = protocol["artifacts"][name]
        artifact_path = protocol_root / specification["path"]
        actual = sha256_file(artifact_path)
        require(actual == specification["sha256"], f"{name} hash mismatch")
        hashes[f"{name}_sha256"] = actual
    prompts = load_jsonl(protocol_root / protocol["artifacts"]["prompts"]["path"])
    validate_protocol(protocol, prompts)
    return protocol, prompts, hashes


def validate_protocol(protocol: dict[str, Any], prompts: Sequence[dict[str, Any]]) -> None:
    require(len(prompts) == 10, "scene scorecard must contain exactly ten prompts")
    require(len({row["prompt_id"] for row in prompts}) == 10, "prompt IDs must be unique")
    development = [row for row in prompts if row["selection_role"] == "development_regression"]
    post = [row for row in prompts if row["selection_role"] == "post_selection_reserved"]
    require(len(development) == len(post) == 5, "scorecard must contain five development and five post-selection prompts")
    require(all(row["prior_exposure"]["exposed"] is True for row in development), "development exposure is incomplete")
    require(all(row["prior_exposure"]["exposed"] is False for row in post), "post-selection prompts must be newly reserved")
    require(all(len(row["atomic_facts"]) == 4 for row in prompts), "every prompt must define four atomic facts")
    require(all(row.get("premise") and row.get("protagonist_motive") for row in prompts), "premise and motive are required")
    decoding = protocol["decoding"]
    require(decoding["greedy"] == {"max_new_tokens": 300}, "greedy decoding changed")
    require(decoding["sampled"] == {
        "base_seeds": [11337, 21337, 31337],
        "max_new_tokens": 300,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.95,
    }, "sampled decoding changed")
    require(protocol["matrix"] == {"outputs_per_prompt": 4, "prompt_count": 10, "total_outputs": 40}, "generation matrix changed")
    gates = protocol["success_gates"]["suite_promotion"]
    require(gates["fact_retention_rate_minimum"] == 0.7, "fact retention gate changed")
    require(gates["premise_adherence_rate_minimum"] == 0.8, "premise adherence gate changed")


def generation_id(prompt_id: str, mode: str, seed: int | None) -> str:
    digest = sha256_bytes(canonical_json([prompt_id, mode, seed]).encode("utf-8"))[:20]
    return f"scene-{digest}"


def review_id(record: dict[str, Any]) -> str:
    digest = sha256_bytes(canonical_json([
        record["generation_id"], record["generated_text_sha256"]
    ]).encode("utf-8"))[:20]
    return f"review-{digest}"


def text_diagnostics(text: str, token_ids: Sequence[int], prompt: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    words = normalized_words(text)
    minimum = int(protocol["word_count"]["compliant_minimum"])
    maximum = int(protocol["word_count"]["compliant_maximum"])
    lowered = " ".join(words)
    fact_mentions = {
        fact["id"]: " ".join(normalized_words(fact["text"])) in lowered
        for fact in prompt["atomic_facts"]
    }
    return {
        "diagnostic_only": True,
        "word_count": len(words),
        "length_classification": "short" if len(words) < minimum else "overlength" if len(words) > maximum else "compliant",
        "exact_atomic_fact_phrase_mentions": fact_mentions,
        "exact_atomic_fact_phrase_mention_rate": sum(fact_mentions.values()) / len(fact_mentions),
        "repetition": repetition_diagnostics(text, token_ids),
    }


@torch.no_grad()
def generate_matrix(
    model: FictionPulperLM,
    tokenizer: Tokenizer,
    prompts: Sequence[dict[str, Any]],
    protocol: dict[str, Any],
    *,
    device: torch.device,
    use_bf16: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sampled = protocol["decoding"]["sampled"]
    for prompt in prompts:
        trials = [("greedy", None), *(("sampled", seed) for seed in sampled["base_seeds"])]
        for mode, seed in trials:
            settings = protocol["decoding"][mode]
            output = generate_continuation(
                model,
                tokenizer,
                prompt["prompt"],
                max_new_tokens=int(settings["max_new_tokens"]),
                device=device,
                use_bf16=use_bf16,
                temperature=None if mode == "greedy" else float(settings["temperature"]),
                top_k=0 if mode == "greedy" else int(settings["top_k"]),
                top_p=1.0 if mode == "greedy" else float(settings["top_p"]),
                seed=0 if seed is None else int(seed),
            )
            record = {
                "generation_id": generation_id(prompt["prompt_id"], mode, seed),
                "prompt_id": prompt["prompt_id"],
                "selection_role": prompt["selection_role"],
                "genre": prompt["genre"],
                "prompt": prompt["prompt"],
                "atomic_facts": prompt["atomic_facts"],
                "premise": prompt["premise"],
                "protagonist": prompt["protagonist"],
                "protagonist_motive": prompt["protagonist_motive"],
                "generation_mode": mode,
                "sampling_seed": seed,
                **output,
            }
            record["generated_text_sha256"] = sha256_bytes(record["generated_text"].encode("utf-8"))
            record["diagnostics"] = text_diagnostics(
                record["generated_text"], record["generated_token_ids"], prompt, protocol
            )
            records.append(record)
    validate_matrix(records, prompts, protocol)
    return records


def validate_matrix(records: Sequence[dict[str, Any]], prompts: Sequence[dict[str, Any]], protocol: dict[str, Any]) -> None:
    require(len(records) == 40, "generation matrix must contain exactly 40 outputs")
    expected = {
        (prompt["prompt_id"], mode, seed)
        for prompt in prompts
        for mode, seed in [("greedy", None), *(("sampled", value) for value in (11337, 21337, 31337))]
    }
    actual = {(row["prompt_id"], row["generation_mode"], row["sampling_seed"]) for row in records}
    require(actual == expected and len(actual) == len(records), "generation matrix is incomplete or duplicated")
    for row in records:
        require(row["generated_text_sha256"] == sha256_bytes(row["generated_text"].encode("utf-8")), "generated text hash mismatch")
        require(row["actual_generated_token_count"] == len(row["generated_token_ids"]), "generated token count mismatch")
        maximum = int(protocol["decoding"][row["generation_mode"]]["max_new_tokens"])
        require(row["stop_reason"] in {"eos", "max_new_tokens"}, "invalid stop reason")
        if row["stop_reason"] == "max_new_tokens":
            require(len(row["generated_token_ids"]) == maximum, "max-token output has wrong token count")


def build_review_packet(records: Sequence[dict[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packet = []
    metadata = {}
    for record in records:
        identifier = review_id(record)
        packet.append({
            "review_id": identifier,
            "genre": record["genre"],
            "prompt": record["prompt"],
            "atomic_facts": record["atomic_facts"],
            "premise": record["premise"],
            "protagonist_motive": record["protagonist_motive"],
            "generated_text": record["generated_text"],
            "generated_text_sha256": record["generated_text_sha256"],
            "word_count": record["diagnostics"]["word_count"],
            "length_compliant": record["diagnostics"]["length_classification"] == "compliant",
            "reviewer_id": "",
            "reviewer_role": "original",
            "independent_review_confirmed": None,
            "atomic_fact_labels": {},
            **{field: "" for field in HUMAN_FIELDS},
            "evidence": {},
            "reviewer_note": "",
        })
        metadata[identifier] = {
            key: record[key]
            for key in ("generation_id", "prompt_id", "selection_role", "generation_mode", "sampling_seed")
        }
    random.Random(seed).shuffle(packet)
    return packet, metadata


def _git_metadata(root: Path) -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, check=True, capture_output=True, text=True).stdout
    return {"git_commit": commit, "tracked_worktree_clean": not bool(status.strip())}


def run_baseline(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    require(not args.output_dir.exists(), f"refusing to overwrite existing output directory: {args.output_dir}")
    protocol, prompts, protocol_hashes = load_protocol(args.protocol_root)
    baseline = protocol["baseline"]
    require(baseline["tokenizer_sha256"] == TOKENIZER_SHA256, "registered tokenizer identity changed")
    model_identity = verify_sealed_model("50m", root)
    for key in ("model_id", "checkpoint_sha256", "seal_sha256"):
        require(model_identity[key] == baseline[key], f"protocol baseline {key} mismatch")
    tokenizer_path = root / baseline["tokenizer_path"]
    require(tokenizer_path.is_file(), f"missing tokenizer: {tokenizer_path}")
    require(sha256_file(tokenizer_path) == baseline["tokenizer_sha256"], "tokenizer hash mismatch")

    device = torch.device(args.device)
    use_bf16 = args.precision == "bf16"
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA is unavailable")
        require(not use_bf16 or torch.cuda.is_bf16_supported(), "CUDA BF16 is unavailable")
    else:
        require(not use_bf16, "BF16 scene generation requires CUDA")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    checkpoint = torch.load(model_identity["checkpoint_path"], map_location=device, weights_only=False)
    checkpoint_tokenizer = checkpoint.get("tokenizer_hash", checkpoint.get("tokenizer_sha256"))
    require(checkpoint_tokenizer == baseline["tokenizer_sha256"], "checkpoint tokenizer hash mismatch")
    model = FictionPulperLM(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    records = generate_matrix(model, tokenizer, prompts, protocol, device=device, use_bf16=use_bf16)
    packet, review_metadata = build_review_packet(records, seed=int(protocol["review"]["blinded_packet_seed"]))

    raw_path = args.output_dir / "raw-generations.jsonl"
    diagnostics_path = args.output_dir / "diagnostic-summary.json"
    packet_path = args.output_dir / "blinded-review-packet.jsonl"
    key_path = args.output_dir / "private-review-key.json"
    manifest_path = args.output_dir / "manifest.json"
    write_jsonl(raw_path, records)
    length_counts = {
        label: sum(row["diagnostics"]["length_classification"] == label for row in records)
        for label in ("short", "compliant", "overlength")
    }
    stop_counts = {
        reason: sum(row["stop_reason"] == reason for row in records)
        for reason in ("eos", "max_new_tokens")
    }
    diagnostic_summary = {
        "status": "diagnostic_only_human_review_pending",
        "output_count": len(records),
        "length_classification_counts": length_counts,
        "stop_reason_counts": stop_counts,
        "mean_word_count": sum(row["diagnostics"]["word_count"] for row in records) / len(records),
        "warning": "Lexical fact mentions, word counts, and repetition are automated diagnostics, not human scorecard results.",
    }
    write_json_atomic(diagnostics_path, diagnostic_summary)
    write_jsonl(packet_path, packet)
    review_key = {
        "packet_sha256": sha256_file(packet_path),
        "raw_generations_sha256": sha256_file(raw_path),
        "protocol_sha256": protocol_hashes["protocol_sha256"],
        "model_mapping": {"model-a": baseline["model_id"]},
        "review_metadata": review_metadata,
        "instructions": "Two genuine independent reviewers complete separate copies. Any disagreement or uncertain field requires a complete row from an independent adjudicator. Preserve all original rows.",
    }
    write_json_atomic(key_path, review_key)
    manifest = {
        "status": "generation_complete_human_review_pending",
        "runner_version": RUNNER_VERSION,
        **protocol_hashes,
        "model_id": baseline["model_id"],
        "checkpoint_path": baseline["checkpoint_path"],
        "checkpoint_sha256": baseline["checkpoint_sha256"],
        "seal_path": baseline["seal_path"],
        "seal_sha256": baseline["seal_sha256"],
        "tokenizer_path": baseline["tokenizer_path"],
        "tokenizer_sha256": baseline["tokenizer_sha256"],
        "model_config_sha256": sha256_bytes(canonical_json(asdict(model.config)).encode("utf-8")),
        "prompt_count": len(prompts),
        "generation_count": len(records),
        "decoding": protocol["decoding"],
        "precision": args.precision,
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        **_git_metadata(root),
        "artifact_sha256": {
            "raw_generations": sha256_file(raw_path),
            "diagnostic_summary": sha256_file(diagnostics_path),
            "blinded_review_packet": sha256_file(packet_path),
            "private_review_key": sha256_file(key_path),
        },
        "reproduction_command": "python -m src.scene_scorecard baseline",
    }
    write_json_atomic(manifest_path, manifest)
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, indent=2, sort_keys=True))


def _validate_review_row(row: dict[str, Any], packet: dict[str, Any]) -> None:
    require(row.get("independent_review_confirmed") is True, "reviewer must confirm independent review")
    require(str(row.get("reviewer_id", "")).strip() == row.get("reviewer_id") and bool(row.get("reviewer_id")), "reviewer_id is required without surrounding whitespace")
    require(row.get("reviewer_role") in {"original", "adjudicator"}, "invalid reviewer role")
    expected_facts = {fact["id"] for fact in packet["atomic_facts"]}
    labels = row.get("atomic_fact_labels")
    require(isinstance(labels, dict) and set(labels) == expected_facts, "atomic fact labels are incomplete")
    labels = cast(dict[str, str], labels)
    require(set(labels.values()) <= FACT_LABELS, "invalid atomic fact label")
    for field in HUMAN_FIELDS:
        require(row.get(field) in YES_NO_UNCERTAIN, f"invalid {field}")
    evidence = row.get("evidence")
    evidence_fields = {*expected_facts, *HUMAN_FIELDS}
    require(isinstance(evidence, dict) and set(evidence) == evidence_fields, "review evidence is incomplete")
    evidence = cast(dict[str, str], evidence)
    generated_text = str(packet["generated_text"])
    for field, quote in evidence.items():
        require(isinstance(quote, str) and bool(quote.strip()), f"empty evidence for {field}")
        special = quote in {"<ABSENT>", "<EMPTY_OUTPUT>"}
        require(special or quote in generated_text, f"evidence for {field} is not a verbatim output quote")
        require(quote != "<EMPTY_OUTPUT>" or not generated_text, "<EMPTY_OUTPUT> is valid only for empty output")
    for field in ("genre", "prompt", "atomic_facts", "premise", "protagonist_motive", "generated_text", "generated_text_sha256", "word_count", "length_compliant"):
        require(row.get(field) == packet.get(field), f"review content differs from packet: {field}")
    require(bool(str(row.get("reviewer_note", "")).strip()), "reviewer_note is required")


def resolve_reviews(packet: Sequence[dict[str, Any]], reviews: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    packet_by_id = {row["review_id"]: row for row in packet}
    grouped: dict[str, list[dict[str, Any]]] = {identifier: [] for identifier in packet_by_id}
    for row in reviews:
        identifier = str(row.get("review_id", ""))
        require(identifier in packet_by_id, f"unknown review_id: {identifier}")
        _validate_review_row(row, packet_by_id[identifier])
        grouped[identifier].append(row)
    resolved = []
    reviewer_pairs = []
    score_fields = ("atomic_fact_labels", *HUMAN_FIELDS)
    for identifier, rows in grouped.items():
        originals = [row for row in rows if row["reviewer_role"] == "original"]
        adjudicators = [row for row in rows if row["reviewer_role"] == "adjudicator"]
        require(len(originals) == 2 and len({row["reviewer_id"] for row in originals}) == 2, f"{identifier}: exactly two independent original reviews required")
        pair = tuple(sorted(row["reviewer_id"] for row in originals))
        reviewer_pairs.append((pair[0], pair[1]))
        disagreement = any(originals[0][field] != originals[1][field] for field in score_fields)
        uncertain = any(
            value == "uncertain"
            for row in originals
            for field in score_fields
            for value in (row[field].values() if field == "atomic_fact_labels" else [row[field]])
        )
        if disagreement or uncertain:
            require(len(adjudicators) == 1, f"{identifier}: one independent adjudication required")
            require(adjudicators[0]["reviewer_id"] not in pair, f"{identifier}: adjudicator is not independent")
            require(not any(value == "uncertain" for field in score_fields for value in (adjudicators[0][field].values() if field == "atomic_fact_labels" else [adjudicators[0][field]])), f"{identifier}: adjudication must resolve uncertainty")
            final = adjudicators[0]
        else:
            require(not adjudicators, f"{identifier}: unnecessary adjudication")
            final = originals[0]
        resolved.append(final)
    require(len(set(reviewer_pairs)) == 1, "the same two original reviewers must score all outputs")
    return resolved, reviewer_pairs


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    require(total > 0, "Wilson interval requires a positive denominator")
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return {"confidence_level": 0.95, "method": "Wilson score", "lower": center - margin, "upper": center + margin}


def _score_summary(scored: Sequence[dict[str, Any]]) -> dict[str, Any]:
    require(bool(scored), "human score summary requires at least one output")
    fact_successes = sum(row["retained_facts"] for row in scored)
    fact_total = sum(row["fact_total"] for row in scored)
    premise_successes = sum(row["premise_adherence"] == "yes" for row in scored)
    coherent_successes = sum(row["coherent"] for row in scored)
    desired_labels = {
        "premise_adherence": "yes",
        "motive_preservation": "yes",
        "contradiction_present": "no",
        "loop_or_repetition_present": "no",
        "plot_advancement": "yes",
        "genre_voice": "yes",
    }
    dimension_metrics = {}
    for field, desired in desired_labels.items():
        successes = sum(row[field] == desired for row in scored)
        dimension_metrics[field] = {
            "desired_label": desired,
            "numerator": successes,
            "denominator": len(scored),
            "rate": successes / len(scored),
            "interval": wilson_interval(successes, len(scored)),
        }
    return {
        "sample_size": {"outputs": len(scored), "atomic_facts": fact_total},
        "fact_retention": {"numerator": fact_successes, "denominator": fact_total, "rate": fact_successes / fact_total, "interval": wilson_interval(fact_successes, fact_total)},
        "premise_adherence": {"numerator": premise_successes, "denominator": len(scored), "rate": premise_successes / len(scored), "interval": wilson_interval(premise_successes, len(scored))},
        "coherent_outputs": {"numerator": coherent_successes, "denominator": len(scored), "rate": coherent_successes / len(scored), "interval": wilson_interval(coherent_successes, len(scored))},
        "human_dimensions": dimension_metrics,
    }


def score_resolved_reviews(resolved: Sequence[dict[str, Any]], metadata: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    scored = []
    for row in resolved:
        retained = sum(label == "retained" for label in row["atomic_fact_labels"].values())
        fact_total = len(row["atomic_fact_labels"])
        coherent = (
            retained / fact_total >= 0.7
            and row["premise_adherence"] == "yes"
            and row["motive_preservation"] == "yes"
            and row["contradiction_present"] == "no"
            and row["loop_or_repetition_present"] == "no"
            and row["plot_advancement"] == "yes"
            and row["genre_voice"] == "yes"
            and row["length_compliant"] is True
        )
        scored.append({**metadata[row["review_id"]], "review_id": row["review_id"], "retained_facts": retained, "fact_total": fact_total, "coherent": coherent, **{field: row[field] for field in HUMAN_FIELDS}, "word_count": row["word_count"]})
    pooled = _score_summary(scored)
    fact_successes = pooled["fact_retention"]["numerator"]
    fact_total = pooled["fact_retention"]["denominator"]
    premise_successes = pooled["premise_adherence"]["numerator"]
    coherent_successes = pooled["coherent_outputs"]["numerator"]
    per_prompt = {
        prompt_id: sum(row["coherent"] for row in scored if row["prompt_id"] == prompt_id) / 4
        for prompt_id in sorted({row["prompt_id"] for row in scored})
    }
    gates = protocol["success_gates"]["suite_promotion"]
    gate_results = {
        "fact_retention": fact_successes / fact_total >= gates["fact_retention_rate_minimum"],
        "premise_adherence": premise_successes / len(scored) >= gates["premise_adherence_rate_minimum"],
        "coherent_output_rate": coherent_successes / len(scored) >= gates["coherent_output_rate_minimum"],
        "per_prompt_coherent_output_rate": min(per_prompt.values()) >= gates["per_prompt_coherent_output_rate_minimum"],
    }
    suite_promoted = all(gate_results.values())
    role_summaries = [
        {"selection_role": role, **_score_summary([row for row in scored if row["selection_role"] == role])}
        for role in ("development_regression", "post_selection_reserved")
    ]
    trial_summaries = []
    for mode, seed in (("greedy", None), ("sampled", 11337), ("sampled", 21337), ("sampled", 31337)):
        trial_summaries.append({
            "trial_cell": "greedy" if mode == "greedy" else f"sampled:{seed}",
            "generation_mode": mode,
            "sampling_seed": seed,
            **_score_summary([
                row for row in scored
                if row["generation_mode"] == mode and row["sampling_seed"] == seed
            ]),
        })
    eligible = [row for row in scored if row["selection_role"] == "post_selection_reserved" and row["coherent"]]
    mode_order = {"greedy": 0, "sampled": 1}
    eligible.sort(key=lambda row: (-row["retained_facts"], abs(row["word_count"] - 200), row["prompt_id"], mode_order[row["generation_mode"]], row["sampling_seed"] or 0))
    return {
        "status": "complete_independent_human_review",
        **pooled,
        "pooled_scores": pooled,
        "stratified_scores": {
            "selection_role": role_summaries,
            "trial_cell": trial_summaries,
        },
        "promotion_population": "pooled complete 40-output suite; strata are descriptive and do not redefine promotion",
        "per_prompt_coherent_output_rate": per_prompt,
        "gate_results": gate_results,
        "suite_promoted": suite_promoted,
        "showcase_generation_id": eligible[0]["generation_id"] if suite_promoted and eligible else None,
        "showcase_rule": "No showcase unless every suite gate passes; then use the frozen deterministic ordering over coherent post-selection outputs.",
        "outputs": scored,
    }


def score_reviews(args: argparse.Namespace) -> None:
    require(not args.output.exists(), f"refusing to overwrite existing artifact: {args.output}")
    protocol, _, hashes = load_protocol(args.protocol_root)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    require(manifest["protocol_sha256"] == hashes["protocol_sha256"], "manifest protocol mismatch")
    require(sha256_file(args.packet) == manifest["artifact_sha256"]["blinded_review_packet"], "review packet hash mismatch")
    require(sha256_file(args.key) == manifest["artifact_sha256"]["private_review_key"], "review key hash mismatch")
    packet = load_jsonl(args.packet)
    key = json.loads(args.key.read_text(encoding="utf-8"))
    require(key["packet_sha256"] == sha256_file(args.packet), "private key packet mismatch")
    resolved, reviewer_pairs = resolve_reviews(packet, load_jsonl(args.reviews))
    report = score_resolved_reviews(resolved, key["review_metadata"], protocol)
    report.update({"protocol_sha256": hashes["protocol_sha256"], "reviews_sha256": sha256_file(args.reviews), "reviewer_pair_count": len(set(reviewer_pairs)), "warning": "Human results require genuine independent reviewers; software validation cannot establish real-world independence."})
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser("baseline", help="generate the sealed 50M baseline and blinded packet")
    baseline.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    baseline.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    baseline.add_argument("--device", default="cuda")
    baseline.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    baseline.set_defaults(handler=run_baseline)
    score = subparsers.add_parser("score-reviews", help="validate and score completed independent reviews")
    score.add_argument("--protocol-root", type=Path, default=DEFAULT_PROTOCOL_ROOT)
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--packet", type=Path, required=True)
    score.add_argument("--key", type=Path, required=True)
    score.add_argument("--reviews", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(handler=score_reviews)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
