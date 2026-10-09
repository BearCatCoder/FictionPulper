import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.corpus_v3 import (
    REQUIRED_GATES,
    acquire_pinned_story,
    audit_records,
    deterministic_review_sample,
    deterministic_splits,
    load_ledger,
    record_rejection,
    save_ledger,
    split_payload,
)


def story(record_id: str, *, collection: str = "collection-a", author: str = "author-a") -> dict:
    text = "\n\n".join([
        '"Come in," she said. ' + "Ordinary narrative action followed. " * 40,
        "The compact scene continued with clear prose. " * 40,
        "At last the characters reached an ending. " * 40,
    ])
    return {
        "id": record_id,
        "source": "fixture",
        "source_url": f"https://example.test/{record_id}.txt",
        "source_sha256": hashlib.sha256(collection.encode()).hexdigest(),
        "title": record_id,
        "collection_id": collection,
        "creator_id": author,
        "primary_author": author,
        "creator_evidence": {"method": "fixture_registry", "primary_role": "author"},
        "rights_status": "public_domain",
        "rights_evidence": {
            "basis": "fixture public-domain register",
            "evidence_url": "https://example.test/rights",
            "checked_at": "2026-10-09",
        },
        "boundary_evidence": {"method": "curated_story_file", "confidence": "high"},
        "genres": ["mystery"],
        "genre_evidence": {"method": "fixture_catalog"},
        "publication_year": None,
        "publication_year_evidence": {"status": "unknown"},
        "words": len(text.split()),
        "tokenizer_v1_tokens": 100,
        "dialogue": {"dialogue_opening_paragraph_share": 1 / 3},
        "text": text,
    }


QUALITY = {
    "minimum_story_words": 100,
    "maximum_story_words": 20_000,
    "quarantine_signals": [],
}


class CorpusV3Tests(unittest.TestCase):
    def test_pinned_acquisition_resumes_only_verified_bytes(self):
        payload = b"A complete story."
        digest = hashlib.sha256(payload).hexdigest()

        class Response:
            headers = {"ETag": '"fixture"', "Last-Modified": "today"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return payload

        spec = {
            "id": "story-1", "source_url": "https://example.test/story.txt",
            "source_sha256": digest,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = {}
            with patch("src.corpus_v3.urllib.request.urlopen", return_value=Response()) as fetch:
                text, _, resumed = acquire_pinned_story(
                    spec, cache_dir=root / "raw", ledger=ledger, timeout=1
                )
            self.assertEqual(text, payload.decode())
            self.assertFalse(resumed)
            self.assertEqual(fetch.call_count, 1)
            ledger_path = root / "state" / "ledger.json"
            save_ledger(ledger_path, ledger)
            loaded = load_ledger(ledger_path)
            with patch("src.corpus_v3.urllib.request.urlopen") as fetch:
                _, _, resumed = acquire_pinned_story(
                    spec, cache_dir=root / "raw", ledger=loaded, timeout=1
                )
            self.assertTrue(resumed)
            fetch.assert_not_called()

    def test_acquisition_rejects_unpinned_and_changed_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "requires a lowercase SHA-256"):
                acquire_pinned_story(
                    {"id": "x", "source_url": "https://example.test/x", "source_sha256": ""},
                    cache_dir=Path(temporary), ledger={}, timeout=1,
                )

    def test_rights_boundary_and_quality_evidence_are_required(self):
        valid = story("valid")
        self.assertIsNone(record_rejection(valid, QUALITY))
        missing_rights = {**valid, "rights_evidence": {}}
        self.assertEqual(record_rejection(missing_rights, QUALITY), "incomplete_rights_evidence")
        weak_boundary = {**valid, "boundary_evidence": {"method": "guess", "confidence": "medium"}}
        self.assertEqual(record_rejection(weak_boundary, QUALITY), "unverified_story_boundary")
        missing_creator = {**valid, "creator_evidence": {}}
        self.assertEqual(record_rejection(missing_creator, QUALITY), "incomplete_creator_evidence")
        boilerplate = {**valid, "text": valid["text"] + "\nProject Gutenberg"}
        self.assertEqual(record_rejection(boilerplate, QUALITY), "prohibited_text_signal")

    def test_collection_and_duplicate_clusters_cannot_cross_splits(self):
        records = [
            story("a", collection="one"), story("b", collection="one"),
            story("c", collection="two"), story("d", collection="three"),
        ]
        clusters = [{"cluster_id": "near-1", "record_ids": ["c", "d"], "size": 2}]
        first, audit = deterministic_splits(
            records, clusters, 1337, {"train": 0.85, "validation": 0.075, "test": 0.075}
        )
        second, _ = deterministic_splits(
            records, clusters, 1337, {"train": 0.85, "validation": 0.075, "test": 0.075}
        )
        self.assertEqual(first, second)
        self.assertEqual(first["a"], first["b"])
        self.assertEqual(first["c"], first["d"])
        self.assertTrue(audit["passed"])
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            deterministic_splits(
                records, [], 1337, {"train": 0.8, "validation": 0.1, "test": 0.05}
            )

    def test_split_payload_contract_includes_seed_for_packing(self):
        payload = split_payload(
            {"a": "train"}, {"passed": True}, seed=1337
        )
        self.assertEqual(payload["version"], 3)
        self.assertEqual(payload["seed"], 1337)
        self.assertEqual(payload["assignments"], {"a": "train"})

    def test_review_sample_is_deterministic_and_stratified(self):
        records = [story("a"), story("b", collection="b", author="author-b")]
        self.assertEqual(
            deterministic_review_sample(records, 1337, 2),
            deterministic_review_sample(list(reversed(records)), 1337, 2),
        )

    def test_unresolved_near_duplicate_and_missing_review_block_freeze(self):
        records = [story("a", collection="one"), story("b", collection="two", author="author-b")]
        cluster = {"cluster_id": "near-1", "record_ids": ["a", "b"], "size": 2}
        assignments, split_audit = deterministic_splits(
            records, [cluster], 1337, {"train": 0.85, "validation": 0.075, "test": 0.075}
        )
        config = {
            "seed": 1337,
            "quality": QUALITY,
            "audit": {
                "review_sample_count": 2,
                "maximum_unclassified_token_share": 1.0,
                "concentration_limits": {
                    "maximum_author_token_share": 1.0,
                    "maximum_collection_token_share": 1.0,
                },
            },
        }
        result = audit_records(
            records, clusters=[cluster], assignments=assignments, split_audit=split_audit,
            review={"approved_sample_ids": []}, config=config,
        )
        self.assertEqual(set(result["gates"]), REQUIRED_GATES)
        self.assertIn("near_duplicates", result["failed_required_gates"])
        self.assertIn("deterministic_review", result["failed_required_gates"])
        self.assertTrue(result["gates"]["split_leakage"]["passed"])

    def test_acquisition_failure_blocks_freeze(self):
        records = [story("a")]
        assignments, split_audit = deterministic_splits(
            records, [], 1337, {"train": 0.85, "validation": 0.075, "test": 0.075}
        )
        config = {
            "seed": 1337,
            "quality": QUALITY,
            "audit": {
                "review_sample_count": 1,
                "maximum_unclassified_token_share": 1.0,
                "concentration_limits": {
                    "maximum_author_token_share": 1.0,
                    "maximum_collection_token_share": 1.0,
                },
            },
        }
        result = audit_records(
            records,
            clusters=[],
            assignments=assignments,
            split_audit=split_audit,
            review={"approved_sample_ids": ["a"]},
            config=config,
            acquisition_failures=1,
        )
        self.assertIn("acquisition_complete", result["failed_required_gates"])
        self.assertEqual(result["gates"]["acquisition_complete"]["failed_sources"], 1)


if __name__ == "__main__":
    unittest.main()
