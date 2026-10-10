import copy
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
from tokenizers import Tokenizer

from src.scene_scorecard import (
    DEFAULT_PROTOCOL_ROOT,
    build_review_packet,
    generate_matrix,
    load_protocol,
    resolve_reviews,
    score_resolved_reviews,
    text_diagnostics,
    wilson_interval,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = ROOT / DEFAULT_PROTOCOL_ROOT


def completed_row(packet_row: dict[str, Any], reviewer: str, *, role: str = "original") -> dict[str, Any]:
    row = copy.deepcopy(packet_row)
    row.update({
        "reviewer_id": reviewer,
        "reviewer_role": role,
        "independent_review_confirmed": True,
        "atomic_fact_labels": {fact["id"]: "retained" for fact in row["atomic_facts"]},
        "premise_adherence": "yes",
        "motive_preservation": "yes",
        "contradiction_present": "no",
        "loop_or_repetition_present": "no",
        "plot_advancement": "yes",
        "genre_voice": "yes",
        "evidence": {
            **{fact["id"]: row["generated_text"] or "<EMPTY_OUTPUT>" for fact in row["atomic_facts"]},
            **{
                field: row["generated_text"] or "<EMPTY_OUTPUT>"
                for field in (
                    "premise_adherence", "motive_preservation", "contradiction_present",
                    "loop_or_repetition_present", "plot_advancement", "genre_voice",
                )
            },
        },
        "reviewer_note": "The continuation explicitly preserves the setup and advances it.",
    })
    return row


class SceneScorecardTests(unittest.TestCase):
    def test_frozen_protocol_has_exact_prompt_and_trial_design(self):
        protocol, prompts, hashes = load_protocol(PROTOCOL_ROOT)
        self.assertEqual(len(prompts), 10)
        self.assertEqual(sum(row["prior_exposure"]["exposed"] for row in prompts), 5)
        self.assertEqual({len(row["atomic_facts"]) for row in prompts}, {4})
        self.assertEqual(protocol["decoding"]["sampled"]["base_seeds"], [11337, 21337, 31337])
        self.assertEqual(protocol["matrix"]["total_outputs"], 40)
        self.assertEqual(protocol["success_gates"]["suite_promotion"]["fact_retention_rate_minimum"], 0.7)
        self.assertEqual(protocol["success_gates"]["suite_promotion"]["premise_adherence_rate_minimum"], 0.8)
        self.assertEqual(len(hashes["protocol_sha256"]), 64)

    def test_word_count_and_length_classes_are_continuation_only(self):
        protocol, prompts, _ = load_protocol(PROTOCOL_ROOT)
        short = text_diagnostics("one two three", [1, 2, 3], prompts[0], protocol)
        compliant = text_diagnostics("word " * 200, list(range(200)), prompts[0], protocol)
        long = text_diagnostics("word " * 251, list(range(251)), prompts[0], protocol)
        self.assertEqual(short["length_classification"], "short")
        self.assertEqual(compliant["length_classification"], "compliant")
        self.assertEqual(long["length_classification"], "overlength")

    def test_generation_matrix_uses_exact_base_seeds_and_preserves_stops(self):
        protocol, prompts, _ = load_protocol(PROTOCOL_ROOT)

        def fake_generate(*args, max_new_tokens, seed, **kwargs):
            return {
                "generated_text": f"output seed {seed}",
                "generated_token_ids": [1] * max_new_tokens,
                "actual_generated_token_count": max_new_tokens,
                "stop_reason": "max_new_tokens",
                "prompt_token_count": 10,
            }

        with patch("src.scene_scorecard.generate_continuation", side_effect=fake_generate):
            records = generate_matrix(
                cast(Any, None), cast(Any, None), prompts, protocol,
                device=torch.device("cpu"), use_bf16=False,
            )
        self.assertEqual(len(records), 40)
        self.assertEqual({row["sampling_seed"] for row in records}, {None, 11337, 21337, 31337})
        self.assertTrue(all(row["actual_generated_token_count"] == 300 for row in records))

    def test_review_packet_is_deterministic_and_hides_trial_identity(self):
        protocol, prompts, _ = load_protocol(PROTOCOL_ROOT)
        record = {
            "generation_id": "generation-1", "prompt_id": prompts[0]["prompt_id"],
            "selection_role": prompts[0]["selection_role"], "genre": prompts[0]["genre"],
            "prompt": prompts[0]["prompt"], "atomic_facts": prompts[0]["atomic_facts"],
            "premise": prompts[0]["premise"], "protagonist_motive": prompts[0]["protagonist_motive"],
            "generation_mode": "sampled", "sampling_seed": 11337,
            "generated_text": "Mallory entered the club.", "generated_text_sha256": "a" * 64,
            "diagnostics": {"word_count": 4, "length_classification": "short"},
        }
        first, key = build_review_packet([record], seed=protocol["review"]["blinded_packet_seed"])
        second, _ = build_review_packet([record], seed=protocol["review"]["blinded_packet_seed"])
        self.assertEqual(first, second)
        self.assertNotIn("generation_mode", first[0])
        self.assertNotIn("sampling_seed", first[0])
        self.assertEqual(key[first[0]["review_id"]]["sampling_seed"], 11337)

    def test_review_resolution_requires_two_independent_people_and_adjudication(self):
        protocol, prompts, _ = load_protocol(PROTOCOL_ROOT)
        record = {
            "generation_id": "generation-1", "prompt_id": prompts[0]["prompt_id"],
            "selection_role": prompts[0]["selection_role"], "genre": prompts[0]["genre"],
            "prompt": prompts[0]["prompt"], "atomic_facts": prompts[0]["atomic_facts"],
            "premise": prompts[0]["premise"], "protagonist_motive": prompts[0]["protagonist_motive"],
            "generation_mode": "greedy", "sampling_seed": None,
            "generated_text": "A complete continuation.", "generated_text_sha256": "b" * 64,
            "diagnostics": {"word_count": 180, "length_classification": "compliant"},
        }
        packet, _ = build_review_packet([record], seed=protocol["review"]["blinded_packet_seed"])
        reviews = [completed_row(packet[0], "reviewer-a"), completed_row(packet[0], "reviewer-b")]
        resolved, _ = resolve_reviews(packet, reviews)
        self.assertEqual(len(resolved), 1)
        invalid_evidence = copy.deepcopy(reviews)
        invalid_evidence[0]["evidence"]["fact-1"] = "invented quotation"
        with self.assertRaisesRegex(RuntimeError, "not a verbatim"):
            resolve_reviews(packet, invalid_evidence)
        reviews[1]["premise_adherence"] = "no"
        with self.assertRaisesRegex(RuntimeError, "adjudication required"):
            resolve_reviews(packet, reviews)
        reviews.append(completed_row(packet[0], "reviewer-c", role="adjudicator"))
        resolved, _ = resolve_reviews(packet, reviews)
        self.assertEqual(resolved[0]["reviewer_id"], "reviewer-c")

    def test_frozen_gates_and_showcase_order_are_applied_only_after_suite_passes(self):
        protocol, prompts, _ = load_protocol(PROTOCOL_ROOT)
        packet = []
        metadata = {}
        for prompt in prompts:
            for index, (mode, seed) in enumerate((("greedy", None), ("sampled", 11337), ("sampled", 21337), ("sampled", 31337))):
                identifier = f"review-{prompt['prompt_id']}-{index}"
                packet_row = {
                    "review_id": identifier, "genre": prompt["genre"], "prompt": prompt["prompt"],
                    "atomic_facts": prompt["atomic_facts"], "premise": prompt["premise"],
                    "protagonist_motive": prompt["protagonist_motive"], "generated_text": "valid story",
                    "generated_text_sha256": "c" * 64, "word_count": 200 + index,
                    "length_compliant": True, "reviewer_id": "", "reviewer_role": "original",
                    "independent_review_confirmed": None, "atomic_fact_labels": {},
                    **{field: "" for field in ("premise_adherence", "motive_preservation", "contradiction_present", "loop_or_repetition_present", "plot_advancement", "genre_voice")},
                    "evidence": {}, "reviewer_note": "",
                }
                packet.append(packet_row)
                metadata[identifier] = {"generation_id": identifier, "prompt_id": prompt["prompt_id"], "selection_role": prompt["selection_role"], "generation_mode": mode, "sampling_seed": seed}
        resolved = [completed_row(row, "reviewer-a") for row in packet]
        report = score_resolved_reviews(resolved, metadata, protocol)
        self.assertTrue(report["suite_promoted"])
        self.assertEqual(report["showcase_generation_id"], "review-scene-post-01-0")
        self.assertEqual(report["fact_retention"]["denominator"], 160)
        self.assertEqual(report["pooled_scores"]["sample_size"], {"outputs": 40, "atomic_facts": 160})
        role_scores = {
            row["selection_role"]: row
            for row in report["stratified_scores"]["selection_role"]
        }
        self.assertEqual(set(role_scores), {"development_regression", "post_selection_reserved"})
        self.assertTrue(all(row["sample_size"] == {"outputs": 20, "atomic_facts": 80} for row in role_scores.values()))
        trial_scores = {
            row["trial_cell"]: row
            for row in report["stratified_scores"]["trial_cell"]
        }
        self.assertEqual(set(trial_scores), {"greedy", "sampled:11337", "sampled:21337", "sampled:31337"})
        self.assertTrue(all(row["sample_size"] == {"outputs": 10, "atomic_facts": 40} for row in trial_scores.values()))
        self.assertTrue(all(row["fact_retention"]["interval"]["method"] == "Wilson score" for row in trial_scores.values()))
        resolved[0]["premise_adherence"] = "no"
        resolved[0]["atomic_fact_labels"] = {
            fact_id: "not_retained" for fact_id in resolved[0]["atomic_fact_labels"]
        }
        resolved[1]["motive_preservation"] = "no"
        resolved[2]["loop_or_repetition_present"] = "yes"
        resolved[21]["contradiction_present"] = "yes"
        report = score_resolved_reviews(resolved, metadata, protocol)
        self.assertFalse(report["suite_promoted"])
        self.assertIsNone(report["showcase_generation_id"])
        role_scores = {
            row["selection_role"]: row
            for row in report["stratified_scores"]["selection_role"]
        }
        self.assertEqual(role_scores["development_regression"]["fact_retention"]["numerator"], 76)
        self.assertEqual(role_scores["post_selection_reserved"]["fact_retention"]["numerator"], 80)
        trial_scores = {
            row["trial_cell"]: row
            for row in report["stratified_scores"]["trial_cell"]
        }
        self.assertEqual(trial_scores["greedy"]["fact_retention"]["numerator"], 36)
        self.assertEqual(trial_scores["sampled:11337"]["coherent_outputs"]["numerator"], 8)
        self.assertEqual(trial_scores["sampled:21337"]["coherent_outputs"]["numerator"], 9)
        self.assertEqual(trial_scores["sampled:31337"]["coherent_outputs"]["numerator"], 10)
        self.assertIn("pooled complete 40-output suite", report["promotion_population"])

    def test_wilson_interval_discloses_small_sample_uncertainty(self):
        interval = wilson_interval(32, 40)
        self.assertLess(interval["lower"], 0.8)
        self.assertGreater(interval["upper"], 0.8)

    def test_existing_output_directory_fails_before_model_verification(self):
        from argparse import Namespace
        from src.scene_scorecard import run_baseline

        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(output_dir=Path(directory), protocol_root=PROTOCOL_ROOT)
            with patch("src.scene_scorecard.verify_sealed_model") as verifier:
                with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                    run_baseline(args)
                verifier.assert_not_called()


if __name__ == "__main__":
    unittest.main()
