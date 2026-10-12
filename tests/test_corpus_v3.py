import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.corpus_v3 import (
    REQUIRED_GATES,
    acquire_pinned_story,
    audit_split_assignments,
    audit_records,
    deterministic_review_sample,
    deterministic_splits,
    load_ledger,
    record_rejection,
    reset_stage_reports,
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
            "checked_by": "fixture-rights-reviewer",
        },
        "boundary_evidence": {"method": "curated_story_file", "confidence": "high"},
        "genres": ["mystery"],
        "genre_evidence": {
            "method": "fixture_catalog",
            "source_values": ["mystery"],
            "explicitly_unclassified": False,
        },
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


def audit_config(*, sample_count: int = 1, maximum_unclassified: float = 1.0) -> dict:
    return {
        "seed": 1337,
        "quality": QUALITY,
        "audit": {
            "evidence_cutoff_date": "2026-10-12",
            "review_sample_count": sample_count,
            "maximum_unclassified_token_share": maximum_unclassified,
            "concentration_limits": {
                "maximum_author_token_share": 1.0,
                "maximum_collection_token_share": 1.0,
            },
        },
    }


def approved_review(*records: dict) -> dict:
    ordered = sorted(records, key=lambda record: record["id"])
    digest = hashlib.sha256()
    for record in ordered:
        digest.update((__import__("json").dumps(record, ensure_ascii=False) + "\n").encode())
    return {
        "version": 1,
        "exclusion_decisions": [],
        "sample_review": {
            "sample_records_sha256": digest.hexdigest(),
            "decisions": [
                {
                    "id": record_id,
                    "decision": "approve",
                    "reviewer": "fixture-reviewer",
                    "reviewed_at": "2026-10-12",
                    "notes": "Fixture evidence reviewed.",
                }
                for record_id in [record["id"] for record in records]
            ],
        },
    }


def run_audit(records: list[dict], **kwargs) -> dict:
    with patch(
        "src.corpus_v3.encode_document",
        side_effect=lambda _tokenizer, record: ([0] * max(0, record["tokenizer_v1_tokens"]), None),
    ):
        return audit_records(records, tokenizer=object(), **kwargs)


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
            class ChangedResponse:
                headers = {}

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return None

                def read(self):
                    return b"changed"

            spec = {
                "id": "x",
                "source_url": "https://example.test/x",
                "source_sha256": hashlib.sha256(b"expected").hexdigest(),
            }
            with patch("src.corpus_v3.urllib.request.urlopen", return_value=ChangedResponse()):
                with self.assertRaisesRegex(RuntimeError, "Source hash mismatch"):
                    acquire_pinned_story(
                        spec, cache_dir=Path(temporary), ledger={}, timeout=1,
                    )

    def test_rights_boundary_and_quality_evidence_are_required(self):
        valid = story("valid")
        self.assertIsNone(record_rejection(valid, QUALITY))
        missing_rights = {**valid, "rights_evidence": {}}
        self.assertEqual(record_rejection(missing_rights, QUALITY), "incomplete_rights_evidence")
        placeholder_rights = {
            **valid,
            "rights_evidence": {**valid["rights_evidence"], "checked_at": "inherited_source_assertion"},
        }
        self.assertEqual(record_rejection(placeholder_rights, QUALITY), "incomplete_rights_evidence")
        unattributed_rights = {
            **valid,
            "rights_evidence": {key: value for key, value in valid["rights_evidence"].items() if key != "checked_by"},
        }
        self.assertEqual(record_rejection(unattributed_rights, QUALITY), "incomplete_rights_evidence")
        future_rights = {
            **valid,
            "rights_evidence": {**valid["rights_evidence"], "checked_at": "9999-12-31"},
        }
        self.assertEqual(record_rejection(future_rights, QUALITY), "incomplete_rights_evidence")
        boolean_creator = {
            **valid,
            "creator_evidence": {"method": True, "primary_role": True},
        }
        self.assertEqual(record_rejection(boolean_creator, QUALITY), "incomplete_creator_evidence")
        boolean_boundary = {
            **valid,
            "boundary_evidence": {"method": True, "confidence": "high"},
        }
        self.assertEqual(record_rejection(boolean_boundary, QUALITY), "unverified_story_boundary")
        weak_boundary = {**valid, "boundary_evidence": {"method": "guess", "confidence": "medium"}}
        self.assertEqual(record_rejection(weak_boundary, QUALITY), "unverified_story_boundary")
        missing_creator = {**valid, "creator_evidence": {}}
        self.assertEqual(record_rejection(missing_creator, QUALITY), "incomplete_creator_evidence")
        boilerplate = {**valid, "text": valid["text"] + "\nProject Gutenberg"}
        self.assertEqual(record_rejection(boilerplate, QUALITY), "prohibited_text_signal")

    def test_unknown_genres_and_inconsistent_unknown_evidence_are_rejected(self):
        valid = story("valid")
        invented = {**valid, "genres": ["invented_label"]}
        self.assertEqual(record_rejection(invented, QUALITY), "incomplete_genre_evidence")
        inconsistent = {
            **valid,
            "genres": [],
            "genre_evidence": {"method": "fixture_catalog", "explicitly_unclassified": False},
        }
        self.assertEqual(record_rejection(inconsistent, QUALITY), "incomplete_genre_evidence")
        unsupported = {
            **valid,
            "genre_evidence": {"method": "fixture_catalog", "explicitly_unclassified": False},
        }
        self.assertEqual(record_rejection(unsupported, QUALITY), "incomplete_genre_evidence")
        unsupported_mapping = {
            **valid,
            "genre_evidence": {
                "method": "fixture_catalog",
                "source_values": ["romance"],
                "explicitly_unclassified": False,
            },
        }
        self.assertEqual(record_rejection(unsupported_mapping, QUALITY), "incomplete_genre_evidence")
        reviewed_mapping = {
            **unsupported_mapping,
            "genre_evidence": {
                **unsupported_mapping["genre_evidence"],
                "reviewed_mappings": ["mystery"],
                "reviewed_by": "fixture-genre-reviewer",
                "reviewed_at": "2026-10-12",
            },
        }
        self.assertIsNone(record_rejection(reviewed_mapping, QUALITY))

    def test_unknown_year_evidence_cannot_support_an_asserted_year(self):
        record = story("a")
        record["publication_year"] = 1920
        self.assertEqual(
            record_rejection(record, QUALITY), "incomplete_publication_year_evidence"
        )
        record["publication_year_evidence"] = {"status": "known", "source": True}
        self.assertEqual(
            record_rejection(record, QUALITY), "incomplete_publication_year_evidence"
        )

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

    def test_split_audit_recomputes_leakage_and_assignment_coverage(self):
        records = [story("a", collection="one"), story("b", collection="one")]
        audit = audit_split_assignments(records, [], {"a": "train", "b": "test"})
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["collection_leaks"], ["one"])
        missing = audit_split_assignments(records, [], {"a": "train", "extra": "invalid"})
        self.assertFalse(missing["passed"])
        self.assertEqual(missing["missing_assignment_ids"], ["b"])
        self.assertEqual(missing["unexpected_assignment_ids"], ["extra"])
        self.assertEqual(missing["invalid_split_ids"], ["extra"])

    def test_split_payload_contract_includes_seed_for_packing(self):
        payload = split_payload(
            {"a": "train"}, {"passed": True}, seed=1337
        )
        self.assertEqual(payload["version"], 3)
        self.assertEqual(payload["seed"], 1337)
        self.assertEqual(payload["assignments"], {"a": "train"})

    def test_stage_reset_removes_stale_generated_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage_dir = Path(temporary) / "stages"
            stage_dir.mkdir()
            (stage_dir / "40000000.json").write_text("{}\n", encoding="utf-8")
            (stage_dir / "attainable.json").write_text("{}\n", encoding="utf-8")
            reset_stage_reports(stage_dir)
            self.assertEqual(list(stage_dir.iterdir()), [])

    def test_review_sample_is_deterministic_and_stratified(self):
        records = [story("a"), story("b", collection="b", author="author-b")]
        self.assertEqual(
            deterministic_review_sample(records, 1337, 2),
            deterministic_review_sample(list(reversed(records)), 1337, 2),
        )

    def test_unresolved_near_duplicate_and_missing_review_block_freeze(self):
        records = [story("a", collection="one"), story("b", collection="two", author="author-b")]
        cluster = {"cluster_id": "near-1", "record_ids": ["a", "b"], "size": 2}
        assignments, _ = deterministic_splits(
            records, [cluster], 1337, {"train": 0.85, "validation": 0.075, "test": 0.075}
        )
        result = run_audit(
            records, clusters=[cluster], assignments=assignments,
            review={"version": 1, "sample_review": {"decisions": []}},
            config=audit_config(sample_count=2),
        )
        self.assertEqual(set(result["gates"]), REQUIRED_GATES)
        self.assertIn("near_duplicates", result["failed_required_gates"])
        self.assertIn("deterministic_review", result["failed_required_gates"])
        self.assertTrue(result["gates"]["split_leakage"]["passed"])

    def test_acquisition_failure_blocks_freeze(self):
        records = [story("a")]
        assignments, _ = deterministic_splits(
            records, [], 1337, {"train": 0.85, "validation": 0.075, "test": 0.075}
        )
        result = run_audit(
            records,
            clusters=[],
            assignments=assignments,
            review=approved_review(records[0]),
            config=audit_config(),
            acquisition_failures=1,
        )
        self.assertIn("acquisition_complete", result["failed_required_gates"])
        self.assertEqual(result["gates"]["acquisition_complete"]["failed_sources"], 1)

    def test_unclassified_share_and_unattributed_review_block_freeze(self):
        record = story("a")
        record["genres"] = []
        record["genre_evidence"] = {
            "method": "fixture_catalog",
            "source_values": [],
            "explicitly_unclassified": True,
        }
        result = run_audit(
            [record], clusters=[], assignments={"a": "train"},
            review={
                "version": 1,
                "sample_review": {
                    "sample_records_sha256": approved_review(record)["sample_review"]["sample_records_sha256"],
                    "decisions": [{"id": "a", "decision": "approve"}],
                },
            },
            config=audit_config(maximum_unclassified=0.35),
        )
        self.assertIn("genre_distribution", result["failed_required_gates"])
        self.assertIn("deterministic_review", result["failed_required_gates"])
        self.assertEqual(result["gates"]["deterministic_review"]["invalid_decision_ids"], ["a"])

    def test_review_hash_binds_the_complete_sample_records(self):
        original = story("a")
        review = approved_review(original)
        changed = {**original, "title": "Changed after review"}
        result = run_audit(
            [changed], clusters=[], assignments={"a": "train"}, review=review,
            config=audit_config(),
        )
        self.assertIn("deterministic_review", result["failed_required_gates"])
        self.assertNotEqual(
            result["gates"]["deterministic_review"]["sample_records_sha256"],
            result["gates"]["deterministic_review"]["expected_sample_records_sha256"],
        )

    def test_malformed_review_entries_block_freeze(self):
        record = story("a")
        review = approved_review(record)
        review["sample_review"]["decisions"].append("malformed")
        result = run_audit(
            [record], clusters=[], assignments={"a": "train"}, review=review,
            config=audit_config(),
        )
        self.assertIn("deterministic_review", result["failed_required_gates"])
        self.assertEqual(
            result["gates"]["deterministic_review"]["malformed_decision_entries"], [1]
        )
        review = approved_review(record)
        review["sample_review"]["decisions"][0]["reviewer"] = True
        result = run_audit(
            [record], clusters=[], assignments={"a": "train"}, review=review,
            config=audit_config(),
        )
        self.assertIn("deterministic_review", result["failed_required_gates"])

    def test_token_accounting_rejects_invalid_counts(self):
        record = story("a")
        record["tokenizer_v1_tokens"] = -1
        result = run_audit(
            [record], clusters=[], assignments={"a": "train"}, review=approved_review(record),
            config=audit_config(),
        )
        self.assertIn("token_accounting", result["failed_required_gates"])
        self.assertFalse(result["gates"]["token_accounting"]["passed"])

    def test_token_accounting_reconciles_each_document(self):
        records = [story("a", collection="a"), story("b", collection="b", author="b")]
        with patch(
            "src.corpus_v3.encode_document",
            side_effect=lambda _tokenizer, record: (
                [0] * (99 if record["id"] == "a" else 101), None
            ),
        ):
            result = audit_records(
                records, clusters=[], assignments={"a": "train", "b": "test"},
                review=approved_review(*records), config=audit_config(sample_count=2),
                tokenizer=object(),
            )
        self.assertIn("token_accounting", result["failed_required_gates"])
        self.assertEqual(
            result["gates"]["token_accounting"]["mismatched_record_ids"], ["a", "b"]
        )

    def test_exact_duplicates_fail_the_independent_gate(self):
        records = [story("a", collection="a"), story("b", collection="b", author="b")]
        result = run_audit(
            records, clusters=[], assignments={"a": "train", "b": "test"},
            review=approved_review(*records), config=audit_config(sample_count=2),
        )
        self.assertIn("exact_duplicates", result["failed_required_gates"])


if __name__ == "__main__":
    unittest.main()
