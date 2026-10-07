import unittest

from src.prepare_data30m import (
    audit_assignments,
    build_constraint_groups,
    grouped_anchored_assignments,
)


def record(record_id: str, collection: str, text_hash: str | None = None) -> dict:
    return {
        "id": record_id,
        "source_collection": collection,
        "text_hash": text_hash or record_id,
    }


class Data30MPreparationTests(unittest.TestCase):
    def test_collection_and_transitive_duplicate_constraints_are_combined(self):
        records = [
            record("a", "one"),
            record("b", "one"),
            record("c", "two"),
            record("d", "three"),
        ]
        edges = [
            {"left_id": "b", "right_id": "excluded"},
            {"left_id": "excluded", "right_id": "c"},
        ]
        groups, metadata = build_constraint_groups(records, edges)
        self.assertIn(["a", "b", "c"], groups)
        self.assertIn(["d"], groups)
        self.assertEqual(metadata["near_duplicate_external_member_count"], 1)

    def test_historical_majority_maximizes_preserved_assignments(self):
        records = [record("a", "one"), record("b", "one"), record("c", "one")]
        groups, _ = build_constraint_groups(records, [])
        assignments, metadata = grouped_anchored_assignments(
            records,
            groups,
            {"a": "train", "b": "train", "c": "validation"},
            seed=1337,
        )
        self.assertEqual(set(assignments.values()), {"train"})
        self.assertEqual(metadata["preserved_historical_assignment_count"], 2)
        self.assertEqual(metadata["changed_historical_assignment_count"], 1)

    def test_audit_rejects_cross_split_constraint_group(self):
        records = [record("a", "one"), record("b", "one")]
        groups, _ = build_constraint_groups(records, [])
        with self.assertRaisesRegex(RuntimeError, "leakage audit failed"):
            audit_assignments(
                records,
                {"a": "train", "b": "test"},
                groups,
                [],
            )

    def test_exact_duplicate_hash_cannot_cross_splits(self):
        records = [record("a", "one", "same"), record("b", "two", "same")]
        groups, _ = build_constraint_groups(records, [])
        with self.assertRaisesRegex(RuntimeError, "leakage audit failed"):
            audit_assignments(
                records,
                {"a": "train", "b": "validation"},
                groups,
                [],
            )


if __name__ == "__main__":
    unittest.main()
