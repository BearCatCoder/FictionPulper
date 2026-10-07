import unittest
from typing import Any

from src.prepare_data10m import (
    audit_seed_duplicate_splits,
    grouped_anchored_assignments,
    resolve_near_duplicates,
)


def record(record_id: str, source_id: str = "seed") -> dict:
    return {
        "id": record_id,
        "source_id": source_id,
        "title": record_id,
        "author": "Author",
    }


class Data10MPreparationTests(unittest.TestCase):
    def test_seed_duplicate_audit_detects_evaluation_member_in_either_position(self):
        report: dict[str, Any] = {
            "resolutions": [
                {
                    "record_a_id": "validation-copy",
                    "record_b_id": "train-copy",
                    "classification": "same story / alternate edition",
                    "retained_id": "train-copy",
                    "reason": "Reviewed.",
                }
            ]
        }
        audit_seed_duplicate_splits(
            report,
            {"train-copy", "validation-copy"},
            {"train-copy": "train", "validation-copy": "validation"},
            {"train-copy": "train"},
        )
        audit = report["seed_to_seed_split_audit"][0]
        self.assertTrue(audit["crosses_training_evaluation_boundary"])
        self.assertEqual(audit["affected_legacy_metrics"], ["legacy_seed_validation"])
        self.assertEqual(
            report["legacy_metric_contamination"]["legacy_seed_validation"],
            ["validation-copy"],
        )

    def test_seed_duplicate_audit_does_not_flag_train_train_pair(self):
        report: dict[str, Any] = {
            "resolutions": [
                {
                    "record_a_id": "train-a",
                    "record_b_id": "train-b",
                    "classification": "same story / alternate edition",
                    "retained_id": "train-a",
                    "reason": "Reviewed.",
                }
            ]
        }
        audit_seed_duplicate_splits(
            report,
            {"train-a", "train-b"},
            {"train-a": "train", "train-b": "train"},
            {"train-a": "train"},
        )
        audit = report["seed_to_seed_split_audit"][0]
        self.assertFalse(audit["crosses_training_evaluation_boundary"])
        self.assertEqual(audit["affected_legacy_metrics"], [])
        self.assertEqual(report["legacy_metric_contamination"]["legacy_seed_test"], [])

    def test_duplicate_cluster_audit_rejects_transitive_split_leakage(self):
        report: dict[str, Any] = {
            "resolutions": [
                {
                    "record_a_id": "a",
                    "record_b_id": "b",
                    "classification": "same story / formatting variation",
                    "retained_id": "a",
                    "reason": "Reviewed.",
                },
                {
                    "record_a_id": "b",
                    "record_b_id": "c",
                    "classification": "same story / formatting variation",
                    "retained_id": "c",
                    "reason": "Reviewed.",
                },
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "clusters cross Data10M splits"):
            audit_seed_duplicate_splits(
                report,
                set(),
                {},
                {"a": "train", "c": "validation"},
            )

    def test_seed_assignments_are_preserved_and_new_collections_are_grouped(self):
        records = [record("seed-a"), record("seed-b")]
        records += [record("new-a", "pg-1"), record("new-b", "pg-1")]
        records += [record("new-c", "pg-2")]
        assignments, metadata = grouped_anchored_assignments(
            records, {"seed-a": "validation", "seed-b": "test"}, seed=1337
        )
        self.assertEqual(assignments["seed-a"], "validation")
        self.assertEqual(assignments["seed-b"], "test")
        self.assertEqual(assignments["new-a"], assignments["new-b"])
        self.assertEqual(metadata["new_collection_cross_split_violations"], {})

    def test_seed_record_wins_seed_new_duplicate_pair(self):
        records = [record("seed-a"), record("pg-1-story", "pg-1")]
        pairs = [
            {
                "left_id": "seed-a",
                "right_id": "pg-1-story",
                "simhash_hamming_distance": 1,
                "word_5gram_jaccard": 0.99,
                "length_ratio": 1.0,
            }
        ]
        resolved, report = resolve_near_duplicates(records, pairs, {"seed-a"})
        self.assertEqual([item["id"] for item in resolved], ["seed-a"])
        self.assertEqual(report["resolutions"][0]["retained_id"], "seed-a")


if __name__ == "__main__":
    unittest.main()
