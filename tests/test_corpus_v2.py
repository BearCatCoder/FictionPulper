import hashlib
import unittest

from src.corpus_v1 import Extraction
from src.corpus_v2 import (
    add_if_not_exact_duplicate,
    canonical_base_record,
    classify_story_rejection,
    extract_collection_v2,
    near_duplicate_graph,
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


if __name__ == "__main__":
    unittest.main()
