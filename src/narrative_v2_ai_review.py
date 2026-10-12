"""Score authorized AI decisions separately from the frozen human review workflow."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.continuity_curriculum.common import sha256_file
from src import narrative_v2_owner_review as owner


VERSION = "narrative-v2-sol-ai-evaluation-v1"
EVALUATOR = "openai/gpt-6.1-sol"
PACKET_HASH = "ce11fad2c07a412c8aa7b3cbc1eccc59d6d5d4c2b982cdea4fb5696397e92a0b"
MANIFEST_HASH = "ac62a19faec0e47c51eb3823d21039f1c03900de188199076d7659b5452f7c15"


def validate_ai_decisions(
    packet: list[dict[str, Any]], decisions: dict[str, Any],
    *, packet_hash: str, manifest_hash: str,
) -> dict[str, str]:
    """Reject human impersonation, partial reviews, and unbound evidence."""
    owner.require(set(decisions) == {
        "version", "evaluator", "reviewer_role", "genuine_human_attestation",
        "independent_review_confirmed", "packet_sha256", "manifest_sha256", "rows",
    }, "AI decision artifact fields differ from contract")
    owner.require(decisions["version"] == VERSION and decisions["evaluator"] == EVALUATOR,
                  "AI evaluator identity mismatch")
    owner.require(decisions["reviewer_role"] == "ai"
                  and decisions["genuine_human_attestation"] is False
                  and decisions["independent_review_confirmed"] is False,
                  "AI decisions must not claim human review or independence")
    owner.require(decisions["packet_sha256"] == packet_hash
                  and decisions["manifest_sha256"] == manifest_hash,
                  "AI decision packet/manifest binding mismatch")
    labels: dict[str, str] = {}
    numbers: set[int] = set()
    for row in decisions["rows"]:
        owner.require(set(row) == {
            "row_number", "label", "confidence", "evidence_type", "evidence_quote", "note",
        }, "AI row fields differ from contract")
        number = row["row_number"]
        owner.require(type(number) is int and 1 <= number <= len(packet)
                      and number not in numbers, "unknown or duplicate AI row number")
        numbers.add(number)
        original = packet[number - 1]
        owner.require(row["label"] in owner.LABELS, "invalid AI label")
        owner.require(type(row["confidence"]) is int and 1 <= row["confidence"] <= 5,
                      "invalid AI confidence")
        owner.require(isinstance(row["note"], str) and bool(row["note"].strip()),
                      "AI rationale is required")
        text = original["generated_text"]
        quote = row["evidence_quote"]
        evidence_type = row["evidence_type"]
        owner.require(isinstance(quote, str), "AI quote must be text")
        if evidence_type == "required_proposition_absent":
            owner.require(bool(text) and bool(original["required_propositions"])
                          and row["label"] == "inconsistent" and quote == "",
                          "invalid AI omission evidence")
        elif evidence_type == "empty_output":
            owner.require(text == "" and quote == "<EMPTY_OUTPUT>"
                          and row["label"] == "inconsistent", "invalid AI empty evidence")
        else:
            owner.require(evidence_type == "verbatim_quote" and bool(quote) and quote in text,
                          "AI evidence must occur verbatim")
        labels[original["owner_review_id"]] = row["label"]
    owner.require(numbers == set(range(1, len(packet) + 1)), "AI decisions are incomplete")
    return labels


def score(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    owner.require(not args.output.exists(), "refusing to overwrite AI report")
    packet, manifest = owner._load_public_stage(args.packet, args.manifest)
    packet_hash, manifest_hash = sha256_file(args.packet), sha256_file(args.manifest)
    owner.require(packet_hash == PACKET_HASH and manifest_hash == MANIFEST_HASH,
                  "AI evaluation must use the frozen Stage 1 identities")
    decisions = json.loads(args.reviews.read_text(encoding="utf-8"))
    labels = validate_ai_decisions(packet, decisions, packet_hash=packet_hash,
                                   manifest_hash=manifest_hash)
    # Freeze decisions before opening private aggregation metadata.
    review_hash = sha256_file(args.reviews)
    materials = owner._staged_materials(
        root, args.config.resolve(),
        checkpoint_selection_complete=args.checkpoint_selection_complete,
        test_evaluation_complete=args.test_evaluation_complete,
    )
    owner.require(packet == materials["stage_packet"], "AI packet selection mismatch")
    owner.require(manifest == owner._stage1_public_manifest(
        args.config.resolve(), materials["staged"], materials["base"],
        materials["stage1_blocks"], packet_hash,
    ), "AI public manifest mismatch")
    # The unchanged exporter verifies every existing full/stage artifact, including the key.
    owner.export_staged_owner_review(argparse.Namespace(
        config=args.config, output_dir=args.packet.parent,
        checkpoint_selection_complete=args.checkpoint_selection_complete,
        test_evaluation_complete=args.test_evaluation_complete,
    ))
    selected = [row for block in materials["stage1_blocks"] for row in block["records"]]
    pairs = owner._pair_selected(selected, materials["stage_metadata"], labels,
                                 int(materials["base"]["packet_order_seed"]))
    arms = {"arm-1": "baseline-a", "arm-2": "baseline-b"}
    metrics = owner._descriptive_stage_metrics(pairs, arms)
    difference = owner.paired_model_difference(
        pairs, arm_mapping=arms, replicates=materials["staged"]["bootstrap_replicates"],
        seed=materials["staged"]["bootstrap_seed"],
    )
    unresolved = {row["anonymous_arm"]: row["unresolved_uncertain_pair_rate"]
                  for row in metrics if row["dimension"] == "overall"}
    decision = owner.stage2_trigger_decision(
        difference, unresolved, materials["staged"]["stage_2_trigger"],
    )
    decision["rule_type"] = "AI adaptation of historical human rule; not an owner-stage decision"
    diagnostics = owner.aggregate_diagnostics(selected)
    reverse = {model: arm for arm, model in arms.items()}
    for row in diagnostics["strata"]:
        row["anonymous_arm"] = reverse[row.pop("blinded_model_id")]
    owner.require(sha256_file(args.reviews) == review_hash, "AI decisions changed during scoring")
    report = {
        "version": VERSION, "status": "COMPLETE_AI_ONLY_EXPLORATORY_EVIDENCE",
        "evaluator": EVALUATOR, "reviewer_role": "ai", "genuine_human_attestation": False,
        "independent_review_confirmed": False, "human_owner_stage_complete": False,
        "row_label_counts": dict(Counter(labels.values())),
        "coverage": owner._block_coverage(materials["stage1_blocks"]),
        "descriptive_metrics": metrics, "paired_model_difference": difference,
        "stage_2_decision": decision, "automated_diagnostics": diagnostics,
        "agreement": {"status": "not_available", "cohen_kappa": None},
        "adjudication": {"status": "not_performed", "count": 0},
        "model_mapping": {"status": "withheld"},
        "hashes": {
            "packet_sha256": packet_hash, "manifest_sha256": manifest_hash,
            "ai_reviews_sha256": review_hash, "config_sha256": sha256_file(args.config),
            "ai_scorer_sha256": sha256_file(Path(__file__)),
            "rubric_sha256": materials["base"]["rubric_sha256"],
            "private_key_sha256": sha256_file(args.packet.parent / "stage-1-private-key.json"),
        },
        "source_identity_bundle": [{k: v for k, v in source.items()
                                    if k.endswith("sha256")} for source in materials["sources"]],
        "aggregation_provenance": owner._scorer_provenance(root),
        "limitations": [
            "AI-only evidence; no completed owner or independent human review.",
            "Prior AI aggregate findings were visible; this is not independent replication.",
            "Labels were authored in the OpenCode conversation; no separate API call was made.",
            "Bootstrap is conditional on 20 deterministic blocks and one AI evaluator.",
            "A degenerate all-failure bootstrap does not establish equivalence or population certainty.",
            "Small strata are descriptive, not powered, representative, or confirmatory.",
            "Historical human protocols and evidence are unchanged; diagnostics remain separate.",
        ],
    }
    owner._write_once_json(args.output, report)
    print(json.dumps({"status": report["status"], "labels": report["row_label_counts"],
                      "stage_2": decision}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/narrative-v2-owner-review-staged-v1.json"))
    for name in ("packet", "manifest", "reviews", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--checkpoint-selection-complete", action="store_true", required=True)
    parser.add_argument("--test-evaluation-complete", action="store_true", required=True)
    score(parser.parse_args())


if __name__ == "__main__":
    main()
