import copy
import hashlib
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
from tokenizers import Tokenizer

from src.continuity_curriculum.common import (
    canonical_json,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
    write_jsonl,
)
from src.narrative_v2_free_generation import (
    SCORER_VERSION,
    _scorer_provenance,
    _validated_completed_reviews,
    effective_seed,
    generate_continuation,
    generate_records,
    load_protocol,
    parse_args,
    validate_generation_artifacts,
)
from src.narrative_v2_loader import load_authored_split
from src.narrative_v2_scoring import (
    ReviewValidationError,
    build_review_packet,
    diagnostic_score,
    paired_bootstrap_interval,
    review_id,
    score_human_reviews,
    validate_reviews,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "benchmarks/narrative-v2"


def generation(
    scenario: str,
    world: str,
    mode: str = "greedy",
    seed: int | None = None,
    *,
    text: str | None = None,
) -> dict:
    answer = "amber" if world == "A" else "badger"
    opposing = "badger" if world == "A" else "amber"
    generated_text = text if text is not None else f"Alden therefore follows {answer}."
    identifier = f"{scenario}-{world}-{mode}-{seed}"
    return {
        "generation_id": identifier,
        "scenario_id": scenario,
        "scenario_stable_hash": "0" * 64,
        "split": "development",
        "state_family": "static_fact",
        "reasoning_depth": 2,
        "depth_bucket": "2",
        "world": world,
        "generation_mode": mode,
        "sampling_seed": seed,
        "effective_seed": None if seed is None else seed + 100,
        "blinded_model_id": "model-a",
        "prompt": "Alden reads the ledger.",
        "review_question": "Does the continuation preserve the current state?",
        "required_propositions": [f"The result is {answer}."],
        "forbidden_propositions": [f"The result is {opposing}."],
        "entity_names": ["Alden", "Delia"],
        "answer_value": answer,
        "opposing_answer_value": opposing,
        "stale_values": ["copper" if world == "A" else "dogwood"],
        "initial_value": "copper" if world == "A" else "dogwood",
        "opposing_initial_value": "dogwood" if world == "A" else "copper",
        "supporting_event_ids": ["e0", "e1"],
        "generated_text": generated_text,
        "generated_text_sha256": sha256_bytes(generated_text.encode("utf-8")),
        "generated_token_ids": [1, 2, 3, 4],
        "actual_generated_token_count": 4,
        "stop_reason": "max_new_tokens",
        "prompt_token_count": 5,
    }


def review(record: dict, reviewer: str, label: str, adjudicated: str | None = None) -> dict:
    evidence = record["generated_text"] or "<EMPTY_OUTPUT>"
    return {
        "review_id": review_id(record),
        "scenario_id": record["scenario_id"],
        "world": record["world"],
        "generation_mode": record["generation_mode"],
        "sampling_seed": record["sampling_seed"],
        "blinded_model_id": record["blinded_model_id"],
        "generated_text_sha256": record["generated_text_sha256"],
        "reviewer_id": reviewer,
        "label": label,
        "confidence": 4,
        "evidence_quote": evidence,
        "reviewer_note": "The explicit state determines the label.",
        "adjudicated_label": adjudicated,
    }


class EosModel:
    class Config:
        max_seq_len = 1024

    def __init__(self, vocab_size: int, eos_token_id: int):
        self.config = self.Config()
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

    def eval(self):
        return self

    def __call__(self, input_ids):
        logits = torch.full(
            (1, input_ids.shape[1], self.vocab_size), -100.0, device=input_ids.device
        )
        logits[0, -1, self.eos_token_id] = 100.0
        return logits, None


class SampleModel(EosModel):
    def __call__(self, input_ids):
        logits = torch.arange(
            self.vocab_size, dtype=torch.float32, device=input_ids.device
        ).repeat(1, input_ids.shape[1], 1)
        logits[0, -1, self.eos_token_id] = -100.0
        return logits, None


class NarrativeV2FreeGenerationTests(unittest.TestCase):
    def test_protocol_locks_and_effective_seed_match_frozen_definition(self):
        protocol, hashes = load_protocol(BENCHMARK_ROOT)
        self.assertEqual(protocol["decoding"]["sampled"]["base_seeds"], [11337, 21337, 31337])
        self.assertEqual(len(hashes["protocol_sha256"]), 64)
        payload = canonical_json([11337, "nbv2-000", "A", "model-a"]).encode("utf-8")
        expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        self.assertEqual(effective_seed(11337, "nbv2-000", "A", "model-a"), expected)

    def test_generation_preserves_empty_eos_output_and_stop_reason(self):
        tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer/tokenizer.json"))
        eos_token_id = tokenizer.token_to_id("<|eos|>")
        model = EosModel(tokenizer.get_vocab_size(), eos_token_id)
        result = generate_continuation(
            cast(Any, model), tokenizer, "Alden opened the ledger.", max_new_tokens=8,
            device=torch.device("cpu"), use_bf16=False,
        )
        self.assertEqual(result["generated_token_ids"], [eos_token_id])
        self.assertEqual(result["generated_text"], "")
        self.assertEqual(result["actual_generated_token_count"], 1)
        self.assertEqual(result["stop_reason"], "eos")

    def test_sampled_generation_repeats_exact_ids_for_effective_seed(self):
        tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer/tokenizer.json"))
        model = SampleModel(tokenizer.get_vocab_size(), tokenizer.token_to_id("<|eos|>"))
        settings = {
            "max_new_tokens": 12, "device": torch.device("cpu"), "use_bf16": False,
            "temperature": 0.8, "top_k": 50, "top_p": 0.95, "seed": 123456,
        }
        first = generate_continuation(cast(Any, model), tokenizer, "Alden read.", **settings)
        second = generate_continuation(cast(Any, model), tokenizer, "Alden read.", **settings)
        self.assertEqual(first["generated_token_ids"], second["generated_token_ids"])
        self.assertEqual(first["actual_generated_token_count"], 12)

    def test_adversarial_diagnostics_do_not_use_substrings_or_lexical_overlap(self):
        base = generation("scenario-1", "A")

        current = diagnostic_score({**base, "generated_text": "Alden therefore resolved the annotation as amber."})
        self.assertTrue(current["current_fact_only"])
        self.assertTrue(current["causal_adherence"]["diagnostic_pass"])
        self.assertTrue(current["named_entity_retention"]["Alden"])

        both = diagnostic_score({**base, "generated_text": "Alden says the annotation is amber, not badger."})
        self.assertTrue(both["current_fact_only"])
        self.assertFalse(both["explicit_contradiction"])

        negated = diagnostic_score({**base, "generated_text": "Alden says amber is not the annotation."})
        self.assertTrue(negated["explicit_contradiction"])

        stale = diagnostic_score({**base, "generated_text": "Alden says copper yields annotation amber."})
        self.assertEqual(stale["current_vs_stale"], "current_and_stale")

        incidental = diagnostic_score({
            **base,
            "generated_text": "Alden discussed the annotation. After midnight, an amber lamp glowed.",
        })
        self.assertFalse(incidental["current_fact_only"])
        self.assertFalse(incidental["causal_adherence"]["diagnostic_pass"])

        same_sentence = diagnostic_score({
            **base,
            "generated_text": "After midnight, Alden filed the annotation beside an amber lamp.",
        })
        self.assertFalse(same_sentence["current_fact_only"])
        self.assertFalse(same_sentence["causal_adherence"]["diagnostic_pass"])

        substring = diagnostic_score({**base, "generated_text": "Aldenberry therefore acts."})
        self.assertFalse(substring["named_entity_retention"]["Alden"])
        self.assertFalse(substring["causal_adherence"]["diagnostic_pass"])

        repeated = diagnostic_score({
            **base,
            "generated_text": "Again. Again.",
            "generated_token_ids": [1, 2, 3, 4, 1, 2, 3, 4],
        })
        self.assertGreater(repeated["repetition"]["repeated_4gram_rate"], 0)
        self.assertEqual(repeated["repetition"]["repeated_sentence_rate"], 0.5)

    def test_review_export_is_blinded_and_byte_repeatable(self):
        records = [generation("scenario-1", world) for world in ("A", "B")]
        first = build_review_packet(records, randomization_seed=260209)
        second = build_review_packet(records, randomization_seed=260209)
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertEqual({row["blinded_model_id"] for row in first}, {"model-a"})
        self.assertTrue(all("checkpoint" not in canonical_json(row) for row in first))
        self.assertTrue(all("scenario_id" not in row for row in first))
        self.assertTrue(all("world" not in row for row in first))
        self.assertTrue(all("generation_mode" not in row for row in first))
        self.assertTrue(all("sampling_seed" not in row for row in first))

    def test_completed_reviews_are_bound_to_packet_and_private_key(self):
        records = [generation("scenario-1", world) for world in ("A", "B")]
        packet = build_review_packet(records, randomization_seed=260209)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet_path = root / "packet.jsonl"
            manifest_path = root / "manifest.json"
            key_path = root / "key.json"
            reviews_path = root / "reviews.jsonl"
            write_jsonl(packet_path, packet)
            write_json_atomic(
                manifest_path, {"blinded_model_id": "model-a", "model_id": "test"}
            )
            key = {
                "review_packet_sha256": sha256_file(packet_path),
                "generation_manifest_sha256": sha256_file(manifest_path),
                "rubric_sha256": sha256_file(BENCHMARK_ROOT / "human-review-rubric.json"),
                "model_mapping": {"model-a": "test"},
                "review_metadata": {
                    review_id(record): {
                        field: record.get(field)
                        for field in (
                            "scenario_id", "world", "generation_mode", "sampling_seed",
                            "blinded_model_id", "generated_text_sha256",
                        )
                    }
                    for record in records
                },
            }
            write_json_atomic(key_path, key)
            completed = []
            for packet_row in packet:
                for reviewer in ("r1", "r2"):
                    row = copy.deepcopy(packet_row)
                    row.update({
                        "reviewer_id": reviewer, "label": "consistent", "confidence": 4,
                        "evidence_quote": row["generated_text"], "reviewer_note": "Explicit state.",
                        "adjudicated_label": None,
                    })
                    completed.append(row)
            write_jsonl(reviews_path, completed)
            args = Namespace(
                benchmark_root=BENCHMARK_ROOT, review_packet=packet_path,
                review_key=key_path, generation_manifest=manifest_path, reviews=reviews_path,
            )
            enriched, _ = _validated_completed_reviews(args, records)
            self.assertEqual({row["world"] for row in enriched}, {"A", "B"})

            completed[0]["prompt"] = "tampered prompt"
            write_jsonl(reviews_path, completed)
            with self.assertRaisesRegex(RuntimeError, "reviewed prompt differs"):
                _validated_completed_reviews(args, records)

    def test_review_validation_requires_independent_adjudication(self):
        records = [generation("scenario-1", world) for world in ("A", "B")]
        reviews = []
        reviews.extend([review(records[0], "r1", "consistent"), review(records[0], "r2", "inconsistent")])
        reviews.extend([review(records[1], "r1", "consistent"), review(records[1], "r2", "consistent")])
        with self.assertRaisesRegex(ReviewValidationError, "adjudication required"):
            validate_reviews(records, reviews)
        reviews.append(review(records[0], "r3", "consistent", "consistent"))
        resolved = validate_reviews(records, reviews)
        self.assertEqual(resolved[review_id(records[0])], "consistent")

        invalid = copy.deepcopy(reviews)
        invalid[0]["evidence_quote"] = "invented evidence"
        with self.assertRaisesRegex(ReviewValidationError, "evidence_quote"):
            validate_reviews(records, invalid)

        different_pair = copy.deepcopy(reviews)
        for row in different_pair:
            if row["review_id"] == review_id(records[1]):
                row["reviewer_id"] = {"r1": "r4", "r2": "r5"}.get(
                    row["reviewer_id"], row["reviewer_id"]
                )
        with self.assertRaisesRegex(ReviewValidationError, "same two original reviewers"):
            validate_reviews(records, different_pair)

        whitespace_identity = copy.deepcopy(reviews)
        whitespace_identity[1]["reviewer_id"] = " r2 "
        with self.assertRaisesRegex(ReviewValidationError, "surrounding whitespace"):
            validate_reviews(records, whitespace_identity)

    def test_empty_output_can_be_reviewed_without_discarding_generation(self):
        records = [generation("scenario-1", world, text="") for world in ("A", "B")]
        reviews = [
            review(record, reviewer, "inconsistent")
            for record in records
            for reviewer in ("r1", "r2")
        ]
        resolved = validate_reviews(records, reviews)
        self.assertEqual(set(resolved.values()), {"inconsistent"})

    def test_family_depth_uncertainty_and_sampled_seed_summary_are_repeatable(self):
        records = []
        for mode, seed in [("greedy", None), ("sampled", 11337), ("sampled", 21337), ("sampled", 31337)]:
            records.extend(generation("scenario-1", world, mode, seed) for world in ("A", "B"))
        reviews = [
            review(record, reviewer, "consistent")
            for record in records
            for reviewer in ("r1", "r2")
        ]
        first = score_human_reviews(records, reviews, bootstrap_replicates=100, bootstrap_seed=260209)
        second = score_human_reviews(records, reviews, bootstrap_replicates=100, bootstrap_seed=260209)
        self.assertEqual(first, second)
        dimensions = {item["dimension"] for item in first["metrics"]}
        self.assertEqual(
            dimensions,
            {"overall", "split", "family", "depth", "reasoning_depth", "family_depth"},
        )
        self.assertTrue(all(item["strict_pair_consistency_rate"] == 1.0 for item in first["metrics"]))
        self.assertTrue(all(item["seed_count"] == 3 for item in first["sampled_seed_summary"]))
        self.assertEqual(first["agreement"]["raw_agreement"], 1.0)

    def test_bootstrap_repeatability_and_cli_access_flags(self):
        pairs = [{"strict_success": value} for value in (True, False, True, True)]
        first = paired_bootstrap_interval(pairs, replicates=100, seed=260209)
        self.assertEqual(first, paired_bootstrap_interval(pairs, replicates=100, seed=260209))
        args = parse_args([
            "generate", "--checkpoint", "model.pt", "--checkpoint-sha256", "a" * 64,
            "--tokenizer", "tokenizer.json", "--tokenizer-sha256", "b" * 64,
            "--model-id", "model", "--blinded-model-id", "model-a",
            "--output-dir", "runs/test", "--split", "generalization",
            "--checkpoint-selection-complete", "--test-evaluation-complete",
        ])
        self.assertTrue(args.checkpoint_selection_complete)
        self.assertTrue(args.test_evaluation_complete)

    def test_generation_matrix_validation_rejects_partial_and_tampered_runs(self):
        protocol, _ = load_protocol(BENCHMARK_ROOT)
        scenarios = load_authored_split(BENCHMARK_ROOT)[:1]

        def fake_generation(*args, max_new_tokens, **kwargs):
            return {
                "generated_text": "Alden resolves the annotation as amber.",
                "generated_token_ids": [1] * max_new_tokens,
                "actual_generated_token_count": max_new_tokens,
                "stop_reason": "max_new_tokens",
                "prompt_token_count": 10,
            }

        with patch("src.narrative_v2_free_generation.generate_continuation", side_effect=fake_generation):
            records = generate_records(
                scenarios=scenarios, protocol=protocol, model=cast(Any, None),
                tokenizer=cast(Any, None), split="development", blinded_model_id="model-a",
                device=torch.device("cpu"), use_bf16=False,
            )
        raw_hash = sha256_bytes(
            "".join(canonical_json(record) + "\n" for record in records).encode("utf-8")
        )
        manifest = {
            "scorer_version": SCORER_VERSION,
            "scorer_sha256": _scorer_provenance()["scorer_sha256"],
            "raw_generations_sha256": raw_hash,
            "decoding": protocol["decoding"],
            "scenario_count": 1,
            "generation_count": 8,
            "split": "development",
            "blinded_model_id": "model-a",
        }
        validate_generation_artifacts(
            records, manifest, scenarios, protocol, generations_sha256=raw_hash
        )
        with self.assertRaisesRegex(RuntimeError, "generation count mismatch"):
            validate_generation_artifacts(
                records[:-1], manifest, scenarios, protocol, generations_sha256=raw_hash
            )
        tampered = copy.deepcopy(records)
        tampered[0]["generated_text"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "generated text hash mismatch"):
            validate_generation_artifacts(
                tampered, manifest, scenarios, protocol, generations_sha256=raw_hash
            )


if __name__ == "__main__":
    unittest.main()
