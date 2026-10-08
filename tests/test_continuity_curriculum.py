import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import yaml
from tokenizers import Tokenizer

from src.continuity_curriculum.audits import curriculum_stats, exact_duplicate_audit, similarity_audit
from src.continuity_curriculum.generator import (
    GENRES,
    SPLITS,
    STATE_TYPES,
    _validate_config,
    build_curriculum,
    generate_example,
)
from src.continuity_curriculum.validation import ValidationError, validate_example


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "continuity-curriculum-v1.yaml"
TOKENIZER_PATH = ROOT / "data" / "tokenizer" / "tokenizer.json"


class ContinuityCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    def test_config_has_exact_root_four_splits_and_disjoint_holdouts(self):
        _validate_config(self.config)
        self.assertEqual(self.config["output_path"], "data/narrative_curriculum_v1")
        self.assertEqual(tuple(self.config["holdouts"]), SPLITS)
        self.assertEqual(self.config["splits"]["main"], {
            "train": 0.90, "validation": 0.05, "test": 0.05,
        })
        leaked = copy.deepcopy(self.config)
        leaked["holdouts"]["generalization_holdout"]["objects"][0] = (
            leaked["holdouts"]["train"]["objects"][0]
        )
        with self.assertRaisesRegex(ValueError, "belongs to both"):
            _validate_config(leaked)

    def test_train_generation_is_deterministic_and_covers_required_matrix(self):
        records = []
        observed_states = set()
        observed_distances = set()
        observed_genres = set()
        observed_difficulties = set()
        for ordinal in range(35):
            record, sidecar = generate_example(
                self.config, self.tokenizer, split="train", ordinal=ordinal
            )
            records.append(record)
            observed_states.update(record["state_types"])
            observed_distances.add(record["distance_bucket"])
            observed_genres.add(record["genre"])
            observed_difficulties.add(record["difficulty"])
            self.assertTrue(validate_example(
                record, sidecar, config=self.config, tokenizer=self.tokenizer
            )["passed"])
            self.assertLessEqual(record["token_count"], self.config["maximum_document_tokens"])
            if record["genre_control_token"] is not None:
                self.assertIsNotNone(self.tokenizer.token_to_id(record["genre_control_token"]))
            self.assertTrue(record["text"].startswith(record["title"] + "\n\n"))
            self.assertNotIn("<|", record["text"])

        first = generate_example(self.config, self.tokenizer, split="train", ordinal=17)
        second = generate_example(self.config, self.tokenizer, split="train", ordinal=17)
        self.assertEqual(first, second)
        self.assertEqual(observed_states, set(STATE_TYPES))
        self.assertEqual(observed_distances, set(self.config["distance_buckets"]))
        self.assertEqual(observed_genres, set(GENRES))
        self.assertEqual(observed_difficulties, set(range(1, 6)))
        stats = curriculum_stats(records)
        self.assertEqual(stats["story_count"], 35)
        self.assertIn("state", stats["tokens_by"])
        self.assertIn("p95", stats["story_tokens"])

    def test_level_five_has_compound_entities_goals_and_updates(self):
        record, sidecar = generate_example(
            self.config, self.tokenizer, split="train", ordinal=4
        )
        self.assertEqual(record["difficulty"], 5)
        self.assertTrue({
            "ownership_transfer_hiding_retrieval",
            "physical_scene_location_movement",
            "goal_persistence_completion_failure",
        } <= set(record["state_types"]))
        self.assertGreaterEqual(len(sidecar["generation_values"]["names"]), 3)
        self.assertGreaterEqual(len(sidecar["generation_values"]["objects"]), 2)
        self.assertGreaterEqual(len(sidecar["generation_values"]["locations"]), 2)

    def test_validator_rejects_tampered_state_evidence_and_anchor(self):
        record, sidecar = generate_example(
            self.config, self.tokenizer, split="train", ordinal=9
        )
        bad_state = copy.deepcopy(sidecar)
        entity = next(iter(bad_state["expected_final_state"]))
        attribute = next(iter(bad_state["expected_final_state"][entity]))
        bad_state["expected_final_state"][entity][attribute] = "tampered"
        with self.assertRaisesRegex(ValidationError, "final state"):
            validate_example(record, bad_state, config=self.config, tokenizer=self.tokenizer)

        bad_anchor = copy.deepcopy(sidecar)
        bad_anchor["anchors"][0]["resolved_token"] += 1
        with self.assertRaisesRegex(ValidationError, "anchor token"):
            validate_example(record, bad_anchor, config=self.config, tokenizer=self.tokenizer)

    def test_duplicate_similarity_audit_is_bounded_and_detects_matches(self):
        records = [
            {"id": "a", "split": "train", "text": "One quiet road led home. " * 20},
            {"id": "b", "split": "validation", "text": "ONE, quiet road led home! " * 20},
        ]
        self.assertFalse(exact_duplicate_audit(records)["passed"])
        records[1]["text"] += "A final different sentence."
        report = similarity_audit(
            records,
            shingle_width=3,
            fingerprint_size=32,
            candidate_shared_hashes=1,
            maximum_posting_size=20,
            maximum_candidate_pairs=100,
            reject_jaccard=0.4,
        )
        self.assertFalse(report["passed"])
        self.assertIn("no all-pairs", report["complexity_policy"])
        self.assertFalse(report["candidate_limit_exceeded"])

    def test_one_example_build_writes_exact_requested_artifacts(self):
        # A one-record train-only development cap exercises output without opening
        # generated test or generalization-holdout story content.
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            config = copy.deepcopy(self.config)
            config["tokenizer"]["path"] = str(TOKENIZER_PATH)
            local_config = temporary_path / "config.yaml"
            local_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(temporary_path)
                manifest = build_curriculum(local_config, max_examples=1)
            finally:
                os.chdir(previous)
            output = temporary_path / "data" / "narrative_curriculum_v1"
            self.assertEqual(manifest["development_limit"], 1)
            expected = {
                "stats.json", "manifest.json", "template-audit.json", "state-audit.json",
                "distance-audit.json", "duplicate-audit.json", "quality-audit.json",
                *(f"{split}.jsonl" for split in SPLITS),
                *(f"{split}.state.jsonl" for split in SPLITS),
            }
            self.assertEqual(expected, {path.name for path in output.iterdir()})
            stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["story_count"], 1)
            self.assertGreater(stats["total_tokenizer_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
