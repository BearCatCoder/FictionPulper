import hashlib
import json
import tempfile
import unittest
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from src.corpus_v1 import Extraction
from src.corpus_v2 import (
    CHECKPOINT_VERSION,
    REQUIRED_CONTENT_AUDITS,
    SourceArtifact,
    _write_stage,
    acquire,
    add_if_not_exact_duplicate,
    canonical_jsonl_sha256,
    canonical_base_record,
    classify_story_rejection,
    extract_collection_v2,
    failed_required_audits,
    load_source_ledger,
    load_valid_source_shard,
    near_duplicate_graph,
    replay_source_shard,
    save_source_ledger,
    save_source_shard,
    source_concentration_metrics,
    target_reached,
    validate_canonical_records,
)


def extraction(title: str, words: int = 600, text: str | None = None) -> Extraction:
    if text is None:
        paragraph = "This is ordinary narrative prose with dialogue and action. "
        text = "\n\n".join((paragraph * ((words // 9) + 1),) * 3)
        text = " ".join(text.split()[:words]) + "\n\nA second paragraph continues.\n\nA third paragraph ends."
    return Extraction(title, 0, 10, "high", "toc_heading_match", text)


def canonical(record_id: str, text: str, **updates: object) -> dict:
    record = {
        "id": record_id, "source": "fixture", "source_id": "source-1",
        "source_collection": "Collection", "group_id": "source-1",
        "existing_assignment": None, "title": record_id, "author": "Writer",
        "publication_year": None, "genres": ["Mystery fiction"],
        "broad_genres": ["mystery"], "rights": "public_domain_us",
        "rights_status": "public_domain_us", "extraction_confidence": "high",
        "preprocessing_version": 2, "tokenizer_v1_tokens": 10,
        "source_url": None,
        "boundary_method": "fixture", "text_hash": hashlib.sha256(text.encode()).hexdigest(),
        "boundary_evidence": {"method": "fixture", "confidence": "high"},
        "words": len(text.split()), "provenance": {"fixture": True}, "text": text,
    }
    record.update(updates)
    return record


def source_shard(source_id: str, records: list[dict], fingerprint: str = "fingerprint") -> dict:
    return {
        "version": CHECKPOINT_VERSION,
        "source_id": source_id,
        "url": f"https://example.test/{source_id}.txt",
        "raw_sha256": hashlib.sha256(source_id.encode()).hexdigest(),
        "processing_fingerprint": fingerprint,
        "collection": {"Text#": source_id.removeprefix("pg-")},
        "evidence": {"confidence": "high"},
        "diagnostic": {},
        "source_summary": {
            "source_id": source_id,
            "status": "processed_high_confidence",
            "accepted_stories": 0,
            "accepted_tokens": 0,
            "rejection_counts": {},
        },
        "candidate_records": records,
        "rejections": [],
        "review_samples": [],
        "chapter_audit": [],
        "source_title_audit": [],
        "quality_audit": [],
    }


def replay(shards: list[dict]) -> list[dict]:
    accepted: list[dict] = []
    additions: list[dict] = []
    token_counts: list[int] = []
    exact_hashes: dict[str, dict] = {}
    exact_report: list[dict] = []
    review_universe: list[dict] = []
    reviewed_records: dict[str, dict] = {}
    review_samples: list[dict] = []
    medium_queue: list[dict] = []
    low_queue: list[dict] = []
    rejections: list[dict] = []
    chapter_audit: list[dict] = []
    source_title_audit: list[dict] = []
    counters: Counter[str] = Counter()
    for shard in shards:
        replay_source_shard(
            shard,
            accepted=accepted,
            additions=additions,
            token_counts=token_counts,
            exact_hashes=exact_hashes,
            exact_report=exact_report,
            review_universe=review_universe,
            reviewed_records=reviewed_records,
            reviewed_exclusions=set(),
            review_samples=review_samples,
            medium_queue=medium_queue,
            low_queue=low_queue,
            rejections=rejections,
            chapter_audit=chapter_audit,
            source_title_audit=source_title_audit,
            counters=counters,
            target_tokens=1_000_000,
        )
    return accepted


class CorpusV2Tests(unittest.TestCase):
    def test_novel_chapter_rejection(self):
        body = "\n\n".join([
            "CHAPTER I", "Prose " * 200, "CHAPTER II", "More prose " * 200,
            "CHAPTER III", "Ending prose " * 200,
        ])
        self.assertEqual(
            classify_story_rejection(extraction("A Novel", text=body)),
            "probable_novel_chapter_contamination",
        )

    def test_roman_and_numeric_titles_are_rejected(self):
        self.assertEqual(classify_story_rejection(extraction("XIV")), "chapter_or_structural_title")
        self.assertEqual(classify_story_rejection(extraction("12")), "chapter_or_structural_title")
        self.assertEqual(classify_story_rejection(extraction("Chapter 12")), "chapter_or_structural_title")

    def test_toc_matching_requires_high_coverage(self):
        text = """CONTENTS
 THE RED ROOM....1
 THE OPEN WINDOW....8
 THE LAST STAR....14

THE RED ROOM

First story.

THE OPEN WINDOW

Second story.

THE LAST STAR

Third story.
"""
        stories, evidence = extract_collection_v2(text)
        self.assertEqual(evidence["confidence"], "high")
        self.assertEqual([story.title for story in stories], ["THE RED ROOM", "THE OPEN WINDOW", "THE LAST STAR"])

    def test_twenty_thousand_words_is_allowed_but_more_is_not(self):
        allowed = extraction("Story", text="word " * 20_000 + "\n\nTwo.\n\nThree.")
        # The added paragraph words make this exceed the bound; use exact paragraphs instead.
        allowed_text = " ".join(["word"] * 19_996) + "\n\nword word\n\nword word"
        self.assertIsNone(classify_story_rejection(extraction("Story", text=allowed_text)))
        self.assertEqual(
            classify_story_rejection(extraction("Story", text=allowed.text)),
            "too_long_possible_novel",
        )

    def test_exact_cross_corpus_duplicate_prefers_base(self):
        base = canonical_base_record(canonical("base", "Same story text."), "validation")
        accepted = [base]
        hashes = {hashlib.sha256(b"same story text").hexdigest(): base}
        report: list[dict] = []
        duplicate = canonical("new", "SAME, STORY TEXT!", source_id="source-2")
        self.assertFalse(add_if_not_exact_duplicate(duplicate, accepted, hashes, report))
        self.assertEqual(report[0]["retained_id"], "base")
        self.assertTrue(report[0]["retained_is_base"])

    def test_near_duplicate_clusters_are_transitive(self):
        common = " ".join(f"word{i % 30}" for i in range(700))
        first = common
        second = common.replace("word5", "changed", 3)
        third = second.replace("word8", "different", 3)
        records = [canonical("a", first), canonical("b", second), canonical("c", third)]
        edges, clusters = near_duplicate_graph(records, minimum_jaccard=0.6)
        self.assertGreaterEqual(len(edges), 2)
        self.assertEqual(clusters[0]["record_ids"], ["a", "b", "c"])

    def test_existing_id_and_assignment_are_preserved(self):
        original = canonical("old-id", "A valid story.")
        converted = canonical_base_record(original, "test")
        self.assertEqual(converted["id"], "old-id")
        self.assertEqual(converted["existing_assignment"], "test")
        self.assertEqual(converted["rights_status"], original["rights"])

    def test_rights_filtering(self):
        record = canonical("bad", "A valid story.", rights="copyright", rights_status="copyright")
        with self.assertRaisesRegex(ValueError, "public domain"):
            validate_canonical_records([record])

    def test_ocr_rejection(self):
        text = ("word " * 550) + ("\ufffd" * 30) + "\n\nParagraph two.\n\nParagraph three."
        self.assertEqual(
            classify_story_rejection(extraction("Story", text=text)),
            "damaged_ocr_replacement_characters",
        )

    def test_source_concentration_metrics(self):
        records = [canonical("a", "one", source="base"), canonical("b", "two", source="base")]
        records.append(canonical("c", "three", source="new", source_collection="Other"))
        metrics = source_concentration_metrics(records)
        self.assertAlmostEqual(metrics["source"]["largest_share"], 2 / 3)
        self.assertEqual(metrics["source"]["unique_count"], 2)

        weighted = source_concentration_metrics(records, [30, 20, 10])
        self.assertAlmostEqual(weighted["author"]["top_20_token_share"], 1.0)
        self.assertEqual(len(weighted["author"]["top_20_by_tokens"]), 1)

    def test_target_stopping(self):
        self.assertFalse(target_reached(29_999_999, 30_000_000))
        self.assertTrue(target_reached(30_000_000, 30_000_000))

    def test_canonical_schema_validation_and_base_ids(self):
        valid = canonical("base", "A valid story.")
        validate_canonical_records([valid], expected_base_ids={"base"})
        with self.assertRaisesRegex(ValueError, "preserve base IDs"):
            validate_canonical_records([valid], expected_base_ids={"missing"})
        malformed = dict(valid, title="")
        with self.assertRaisesRegex(ValueError, "title"):
            validate_canonical_records([malformed])
        malformed_provenance = dict(valid, provenance={})
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_canonical_records([malformed_provenance])
        malformed_boundary = dict(valid, boundary_evidence={})
        with self.assertRaisesRegex(ValueError, "boundary evidence"):
            validate_canonical_records([malformed_boundary])
        malformed_version = dict(valid, preprocessing_version=1)
        with self.assertRaisesRegex(ValueError, "preprocessing version"):
            validate_canonical_records([malformed_version])

    def test_required_audits_cannot_be_skipped(self):
        passing = {name: {"passed": True} for name in REQUIRED_CONTENT_AUDITS}
        self.assertEqual(failed_required_audits(passing), [])
        passing.pop("near_duplicate_review")
        self.assertEqual(failed_required_audits(passing), ["near_duplicate_review"])
        passing["near_duplicate_review"] = {"passed": False}
        self.assertEqual(failed_required_audits(passing), ["near_duplicate_review"])

    def test_source_ledger_and_shard_verify_cached_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = root / "source.txt"
            raw_path.write_bytes(b"source bytes")
            digest = hashlib.sha256(b"source bytes").hexdigest()
            artifact = SourceArtifact(
                source_id="pg-1",
                url="https://example.test/pg-1.txt",
                path=str(raw_path),
                sha256=digest,
                byte_count=12,
                retrieval_method="fixture",
                retrieved_at="2026-01-01T00:00:00+00:00",
                etag='"fixture"',
                last_modified="Wed, 01 Jan 2026 00:00:00 GMT",
            )
            shard = source_shard("pg-1", [canonical("story", "story text")])
            shard["raw_sha256"] = digest
            shard_path = root / "pg-1.json"
            save_source_shard(shard_path, shard)
            entry = {
                "source_id": "pg-1",
                "url": artifact.url,
                "raw_sha256": digest,
                "processing_status": "completed",
                "processing_fingerprint": "fingerprint",
                "shard_sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
                "artifact": asdict(artifact),
            }
            ledger_path = root / "ledger.json"
            save_source_ledger(ledger_path, {"pg-1": entry})
            loaded_entry = load_source_ledger(ledger_path)["pg-1"]
            loaded = load_valid_source_shard(
                loaded_entry,
                shard_path,
                source_id="pg-1",
                url=artifact.url,
                fingerprint="fingerprint",
            )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded[1].etag, '"fixture"')
            self.assertIsNone(load_valid_source_shard(
                loaded_entry, shard_path, source_id="pg-1", url=artifact.url,
                fingerprint="changed",
            ))
            shard_path.write_text("{}\n", encoding="utf-8")
            self.assertIsNone(load_valid_source_shard(
                loaded_entry, shard_path, source_id="pg-1", url=artifact.url,
                fingerprint="fingerprint",
            ))
            save_source_shard(shard_path, shard)
            raw_path.write_bytes(b"tampered")
            self.assertIsNone(load_valid_source_shard(
                loaded_entry, shard_path, source_id="pg-1", url=artifact.url,
                fingerprint="fingerprint",
            ))

    def test_resumed_replay_matches_uninterrupted_canonical_hash(self):
        first = source_shard(
            "pg-1",
            [canonical("first", "First unique story.", source_id="pg-1", tokenizer_v1_tokens=7)],
        )
        second = source_shard(
            "pg-2",
            [canonical("second", "Second unique story.", source_id="pg-2", tokenizer_v1_tokens=9)],
        )
        uninterrupted = replay([first, second])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = root / "pg-1.txt"
            raw_path.write_bytes(b"pg-1")
            checkpoint = root / "pg-1.json"
            save_source_shard(checkpoint, first)
            artifact = SourceArtifact(
                source_id="pg-1",
                url=first["url"],
                path=str(raw_path),
                sha256=first["raw_sha256"],
                byte_count=4,
                retrieval_method="fixture",
                retrieved_at="2026-01-01T00:00:00+00:00",
            )
            ledger_path = root / "ledger.json"
            save_source_ledger(ledger_path, {"pg-1": {
                "source_id": "pg-1",
                "url": first["url"],
                "raw_sha256": first["raw_sha256"],
                "processing_status": "completed",
                "processing_fingerprint": "fingerprint",
                "shard_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "artifact": asdict(artifact),
            }})
            resumed_checkpoint = load_valid_source_shard(
                load_source_ledger(ledger_path)["pg-1"],
                checkpoint,
                source_id="pg-1",
                url=first["url"],
                fingerprint="fingerprint",
            )
            assert resumed_checkpoint is not None
            resumed_first = resumed_checkpoint[0]
            resumed = replay([resumed_first, second])
        self.assertEqual(resumed, uninterrupted)
        self.assertEqual(canonical_jsonl_sha256(resumed), canonical_jsonl_sha256(uninterrupted))

    def test_low_confidence_shard_replay_preserves_token_total(self):
        shard = source_shard("pg-1", [])
        shard["evidence"] = {"confidence": "low"}
        shard["diagnostic"] = {"source_id": "pg-1"}
        shard["source_summary"]["status"] = "low_confidence_rejected"
        shard["source_summary"]["rejection_counts"] = {"low_confidence_collection": 1}
        accepted = [canonical("base", "Base story.", tokenizer_v1_tokens=13)]
        token_counts = [13]
        summary, total = replay_source_shard(
            shard,
            accepted=accepted,
            additions=[],
            token_counts=token_counts,
            exact_hashes={},
            exact_report=[],
            review_universe=[],
            reviewed_records={},
            reviewed_exclusions=set(),
            review_samples=[],
            medium_queue=[],
            low_queue=[],
            rejections=[],
            chapter_audit=[],
            source_title_audit=[],
            counters=Counter(),
            target_tokens=100,
        )
        self.assertEqual(total, 13)
        self.assertEqual(summary["status"], "low_confidence_rejected")

    def test_replayed_exclusion_preserves_locked_base_provenance(self):
        base = canonical_base_record(canonical("duplicate", "Base edition."), "train")
        candidate = canonical(
            "duplicate", "Candidate edition.", source_id="pg-1", tokenizer_v1_tokens=9
        )
        shard = source_shard("pg-1", [candidate])
        review_universe = [base]
        reviewed_records = {base["id"]: base}
        replay_source_shard(
            shard,
            accepted=[],
            additions=[],
            token_counts=[],
            exact_hashes={},
            exact_report=[],
            review_universe=review_universe,
            reviewed_records=reviewed_records,
            reviewed_exclusions={"duplicate"},
            review_samples=[],
            medium_queue=[],
            low_queue=[],
            rejections=[],
            chapter_audit=[],
            source_title_audit=[],
            counters=Counter(),
            target_tokens=100,
        )
        self.assertEqual([record["id"] for record in review_universe], ["duplicate"])
        self.assertTrue(reviewed_records["duplicate"]["provenance"]["base_record"])

    def test_stage_report_runs_every_required_audit(self):
        text = extraction("Story").text
        record = canonical("story", text, tokenizer_v1_tokens=20)
        selection = {
            "minimum_story_words": 500,
            "maximum_story_words": 20_000,
            "concentration_limits": {
                "maximum_source_collection_story_share": 1.0,
                "maximum_source_collection_token_share": 1.0,
                "maximum_author_story_share": 1.0,
                "maximum_author_token_share": 1.0,
            },
        }
        dedup = {
            "simhash_max_hamming": 10,
            "minimum_jaccard": 0.72,
            "minimum_length_ratio": 0.75,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            snapshot = _write_stage(
                output,
                20,
                [record],
                [20],
                0,
                expected_base_ids=set(),
                selection=selection,
                dedup=dedup,
                review_universe=[record],
                reviewed_exclusions=set(),
                exact_report=[],
                medium_queue=[],
                low_queue=[],
                counters=Counter(),
                prior_story_count=0,
                prior_token_count=0,
            )
            report = json.loads((output / "stages" / "0m.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshot, {"stories": 1, "tokens": 20})
        self.assertTrue(report["required_audits_passed"])
        self.assertEqual(failed_required_audits(report["audits"]), [])
        self.assertIn("top_20_author_token_share", report)
        self.assertEqual(report["largest_source_token_share"], 1.0)
        self.assertEqual(report["largest_source_collection_token_share"], 1.0)
        self.assertEqual(report["canonical_content_sha256"], canonical_jsonl_sha256([record]))

    def test_network_acquisition_captures_http_validators(self):
        class Response:
            headers = {"ETag": '"abc"', "Last-Modified": "today"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return b"fixture"

        with tempfile.TemporaryDirectory() as temporary, patch(
            "src.corpus_v2.urllib.request.urlopen", return_value=Response()
        ):
            _, artifact = acquire(
                "https://example.test/source.txt",
                Path("source.txt"),
                readable_dirs=[],
                write_dir=Path(temporary),
                timeout=1,
            )
        self.assertEqual(artifact.etag, '"abc"')
        self.assertEqual(artifact.last_modified, "today")


if __name__ == "__main__":
    unittest.main()
