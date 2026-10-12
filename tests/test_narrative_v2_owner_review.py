import copy
import json
import random
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from src.continuity_curriculum.common import sha256_bytes, sha256_file, write_json_atomic, write_jsonl
from src.narrative_v2_owner_review import (
    DEPTHS,
    SPLITS,
    TRIALS,
    OwnerReviewError,
    build_owner_packet,
    parse_args,
    score_owner_pairs,
    score_owner_review,
    select_balanced_records,
    validate_owner_reviews,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/narrative-v2-owner-review-v1.json"


def generation(
    model: str, split: str, family: str, depth: str, scenario: str,
    mode: str, seed: int | None, world: str,
) -> dict:
    text = f"Alden says the annotation is {'amber' if world == 'A' else 'badger'}."
    identifier = f"{model}:{split}:{scenario}:{mode}:{seed}:{world}"
    return {
        "generation_id": identifier, "scenario_id": scenario, "world": world,
        "generation_mode": mode, "sampling_seed": seed, "effective_seed": seed,
        "blinded_model_id": model, "split": split, "state_family": family,
        "reasoning_depth": 5 if depth == "4_plus" else int(depth), "depth_bucket": depth,
        "generated_text_sha256": sha256_bytes(text.encode()), "prompt": "Alden reads the ledger.",
        "generated_text": text, "review_question": "Does the state remain correct?",
        "required_propositions": ["The current state is preserved."],
        "forbidden_propositions": ["The counterfactual state is asserted."],
        "diagnostics": {
            "current_fact_only": True, "explicit_contradiction": False,
            "causal_adherence": {"diagnostic_pass": True},
            "named_entity_retention_rate": 1.0, "stale_fact_mentioned": False,
            "premise": {"retained_without_counterfactual": True},
            "repetition": {"repeated_4gram_rate": 0.0, "repeated_sentence_rate": 0.0},
        },
    }


def population() -> list[dict]:
    rows = []
    for model in ("model-a", "model-b"):
        for split in SPLITS:
            for family_index in range(7):
                family = f"family-{family_index}"
                for depth in DEPTHS:
                    count = 2 if split == "generalization" else 1
                    for scenario_index in range(count):
                        scenario = f"{split}:{family}:{depth}:{scenario_index}"
                        for mode, seed in TRIALS:
                            for world in ("A", "B"):
                                rows.append(generation(
                                    model, split, family, depth, scenario, mode, seed, world
                                ))
    return rows


def completed_reviews(packet: list[dict], packet_hash: str, owner: str = "BearCatCoder") -> list[dict]:
    rows = []
    for original in packet:
        row = copy.deepcopy(original)
        row.update({
            "reviewer_id": owner, "genuine_human_attestation": True,
            "label": "consistent", "confidence": 5,
            "evidence_quote": row["generated_text"], "reviewer_note": "The state is explicit.",
            "packet_sha256": packet_hash,
        })
        rows.append(row)
    return rows


def verification(packet: list[dict], packet_hash: str, owner: str = "BearCatCoder") -> dict:
    return {
        "version": "narrative-v2-owner-review-v1", "review_packet_sha256": packet_hash,
        "owner_id": owner, "reviewed_at": "2026-10-12",
        "selected_owner_review_ids": [row["owner_review_id"] for row in packet],
        "genuine_human_attestation": True, "single_owner_evidence_acknowledged": True,
        "non_independent_evidence_acknowledged": True, "sample_only_evidence_acknowledged": True,
        "no_agreement_available_acknowledged": True, "no_adjudication_performed_acknowledged": True,
    }


class NarrativeV2OwnerReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = population()

    def test_selection_is_order_invariant_balanced_and_text_blind(self):
        selected = select_balanced_records(self.records, selection_seed=280001)
        shuffled = list(self.records)
        random.Random(19).shuffle(shuffled)
        for row in shuffled:
            row["generated_text"] = "changed text must not affect selection"
            row["diagnostics"] = {"ignored": True}
        repeated = select_balanced_records(shuffled, selection_seed=280001)
        self.assertEqual(
            [row["generation_id"] for row in selected],
            [row["generation_id"] for row in repeated],
        )
        self.assertEqual(len(selected), 336)
        blocks = {(row["split"], row["state_family"], row["depth_bucket"],
                   row["scenario_id"], row["generation_mode"], row["sampling_seed"])
                  for row in selected}
        self.assertEqual(len(blocks), 84)
        for split in SPLITS:
            for family_index in range(7):
                group = [row for row in selected if row["split"] == split
                         and row["state_family"] == f"family-{family_index}"]
                trials = {(row["generation_mode"], row["sampling_seed"]) for row in group}
                self.assertEqual(trials, set(TRIALS))
                self.assertEqual({row["depth_bucket"] for row in group}, set(DEPTHS))
        self.assertTrue(all(len({row["world"] for row in selected if
            row["blinded_model_id"] == model and row["scenario_id"] == scenario
            and row["generation_mode"] == mode and row["sampling_seed"] == seed}) == 2
            for model, scenario, mode, seed in {
                (row["blinded_model_id"], row["scenario_id"], row["generation_mode"], row["sampling_seed"])
                for row in selected
            }))

    def test_packet_is_fully_condition_blind(self):
        selected = select_balanced_records(self.records, selection_seed=280001)
        packet, key = build_owner_packet(selected, packet_order_seed=280002)
        forbidden = {
            "actual_model_id", "model_id", "blinded_model_id", "split", "scenario_id",
            "world", "generation_mode", "sampling_seed", "state_family", "depth_bucket",
            "reasoning_depth", "source_path",
        }
        self.assertEqual(len(packet), 336)
        self.assertTrue(all(not forbidden.intersection(row) for row in packet))
        self.assertTrue(all(row["reviewer_role"] == "owner" for row in packet))
        self.assertTrue(all(row["independent_review_confirmed"] is False for row in packet))
        self.assertTrue(all(row["genuine_human_attestation"] is None for row in packet))
        self.assertEqual(set(key), {row["owner_review_id"] for row in packet})

    def test_owner_validation_rejects_tampering_identity_and_fake_attestation(self):
        selected = select_balanced_records(self.records, selection_seed=280001)[:2]
        packet, _ = build_owner_packet(selected, packet_order_seed=280002)
        packet_hash = "a" * 64
        rows = completed_reviews(packet, packet_hash)
        labels, owner = validate_owner_reviews(
            packet, rows, verification(packet, packet_hash), packet_sha256=packet_hash
        )
        self.assertEqual(owner, "BearCatCoder")
        self.assertEqual(set(labels.values()), {"consistent"})

        changed = copy.deepcopy(rows)
        changed[0]["prompt"] = "tampered"
        with self.assertRaisesRegex(OwnerReviewError, "immutable packet content"):
            validate_owner_reviews(packet, changed, verification(packet, packet_hash),
                                   packet_sha256=packet_hash)
        changed = copy.deepcopy(rows)
        changed[0]["reviewer_id"] = "OtherOwner"
        with self.assertRaisesRegex(OwnerReviewError, "same one owner"):
            validate_owner_reviews(packet, changed, verification(packet, packet_hash),
                                   packet_sha256=packet_hash)
        changed = copy.deepcopy(rows)
        changed[0]["genuine_human_attestation"] = False
        with self.assertRaisesRegex(OwnerReviewError, "genuine-human"):
            validate_owner_reviews(packet, changed, verification(packet, packet_hash),
                                   packet_sha256=packet_hash)
        changed = copy.deepcopy(rows)
        changed[0]["adjudicated_label"] = "consistent"
        with self.assertRaisesRegex(OwnerReviewError, "fields differ"):
            validate_owner_reviews(packet, changed, verification(packet, packet_hash),
                                   packet_sha256=packet_hash)

    def test_scoring_strata_bootstrap_and_sampled_summaries_are_deterministic(self):
        pairs = []
        for model in ("model-a", "model-b"):
            for mode, seed in TRIALS:
                pairs.append({
                    "blinded_model_id": model, "generation_mode": mode, "sampling_seed": seed,
                    "split": "development", "state_family": "ownership", "reasoning_depth": 2,
                    "depth_bucket": "2", "strict_success": seed != 21337,
                    "unresolved_uncertain": seed == 31337,
                })
        first = score_owner_pairs(pairs, bootstrap_replicates=100, bootstrap_seed=260209)
        second = score_owner_pairs(pairs, bootstrap_replicates=100, bootstrap_seed=260209)
        self.assertEqual(first, second)
        metrics, summaries = first
        self.assertEqual({row["dimension"] for row in metrics},
                         {"overall", "split", "family", "depth", "reasoning_depth", "family_depth"})
        self.assertTrue(all(row["seed_count"] == 3 for row in summaries))
        uncertain = [row for row in metrics if row["sampling_seed"] == 31337]
        self.assertTrue(all(row["unresolved_uncertain_pair_count"] == 1 for row in uncertain))
        self.assertTrue(all(row["resolved_only_pair_consistency_rate"] is None for row in uncertain))

    def test_score_report_marks_single_owner_limitations_and_null_agreement(self):
        selected = [
            generation("model-a", "development", "static_fact", "1", "scenario", "greedy", None, world)
            for world in ("A", "B")
        ]
        packet, metadata = build_owner_packet(selected, packet_order_seed=280002)
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            packet_path = temp / "packet.jsonl"
            key_path = temp / "key.json"
            reviews_path = temp / "reviews.jsonl"
            verification_path = temp / "verification.json"
            output_path = temp / "report.json"
            write_jsonl(packet_path, packet)
            packet_hash = sha256_file(packet_path)
            key = {
                "version": "narrative-v2-owner-review-v1", "review_packet_sha256": packet_hash,
                "config_sha256": sha256_file(CONFIG), "source_runs": [],
                "review_metadata": metadata, "model_mapping": {"model-a": "actual-a"},
            }
            write_json_atomic(key_path, key)
            write_jsonl(reviews_path, completed_reviews(packet, packet_hash))
            write_json_atomic(verification_path, verification(packet, packet_hash))
            args = Namespace(
                config=CONFIG, checkpoint_selection_complete=True, test_evaluation_complete=True,
                packet=packet_path, private_key=key_path, reviews=reviews_path,
                owner_verification=verification_path, output=output_path,
            )
            with patch("src.narrative_v2_owner_review.validate_source_runs", return_value=([], [])), \
                 patch("src.narrative_v2_owner_review.select_balanced_records", return_value=selected), \
                 patch("builtins.print"):
                report = score_owner_review(args)
            self.assertEqual(report["agreement"]["cohen_kappa"], None)
            self.assertEqual(report["agreement"]["status"], "not_available_single_owner")
            self.assertEqual(report["adjudication"]["count"], 0)
            self.assertIn("model_mapping_revealed_after_completed_scoring", report)
            self.assertTrue(any("not independent" in item for item in report["limitations"]))
            self.assertEqual(report["automated_diagnostics"]["status"],
                             "diagnostic_only_not_owner_decisions")

    def test_cli_requires_both_held_out_access_attestations(self):
        with self.assertRaises(SystemExit):
            parse_args(["export", "--output-dir", "runs/owner"])
        args = parse_args([
            "export", "--output-dir", "runs/owner",
            "--checkpoint-selection-complete", "--test-evaluation-complete",
        ])
        self.assertTrue(args.checkpoint_selection_complete)
        self.assertTrue(args.test_evaluation_complete)
        with self.assertRaises(SystemExit):
            parse_args([
                "score", "--packet", "p", "--private-key", "k", "--reviews", "r",
                "--owner-verification", "v", "--output", "o",
                "--checkpoint-selection-complete",
            ])

    def test_export_rejects_external_output_before_source_access(self):
        args = Namespace(
            config=CONFIG, output_dir=Path("/tmp/outside-fictionpulper"),
            checkpoint_selection_complete=True, test_evaluation_complete=True,
        )
        from src.narrative_v2_owner_review import export_owner_review
        with patch("src.narrative_v2_owner_review.validate_source_runs") as validator:
            with self.assertRaisesRegex(OwnerReviewError, "inside the repository"):
                export_owner_review(args)
            validator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
