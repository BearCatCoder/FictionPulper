import unittest
import hashlib

from src.corpus_v1 import (
    author_is_safely_public_domain,
    catalog_candidates,
    classify_story_rejection,
    extract_collection,
    near_duplicate_candidates,
    normalized_text_hash,
    validate_corpus_records,
)


COLLECTION = """TABLE OF CONTENTS

THE RED ROOM........1
 The Open Window
THE LOST STAR.......24

THE RED ROOM

This is the first story. "Come inside," she said. "The room is cold."

"I will not," he answered. The old house watched them from the hill.

The first story continues with enough prose to represent a complete narrative ending.

THE OPEN WINDOW

This is the second story. It has dialogue, paragraphs, and a definite opening.

The visitor crossed the room and looked outside. Nothing moved in the garden.

At last the window closed, and the visitor understood why it had been left open.

THE LOST STAR

This is the third story, beginning beneath a dark and unfamiliar sky.

The astronomer checked her instruments and called to the waiting captain.

Together they watched the lost star return to its appointed place.
"""


class CollectionExtractionTests(unittest.TestCase):
    def test_open_ended_author_lifespan_is_not_treated_as_a_death_year(self):
        self.assertFalse(author_is_safely_public_domain("Writer, Robert, 1950-"))
        self.assertTrue(author_is_safely_public_domain("Writer, Robert, 1850-1916"))

    def test_toc_and_matching_body_headings_produce_high_confidence_boundaries(self):
        extractions, evidence = extract_collection(COLLECTION)
        self.assertEqual(evidence["confidence"], "high")
        self.assertEqual([item.title for item in extractions], [
            "THE RED ROOM",
            "The Open Window",
            "THE LOST STAR",
        ])
        self.assertTrue(all(item.boundary_method == "toc_heading_match" for item in extractions))
        self.assertNotIn("THE OPEN WINDOW", extractions[0].text)

    def test_capitalized_headings_without_toc_are_not_automatically_extracted(self):
        extractions, evidence = extract_collection(
            "THE FIRST STORY\n\nProse.\n\nTHE SECOND STORY\n\nMore prose."
        )
        self.assertEqual(extractions, [])
        self.assertEqual(evidence["confidence"], "low")

    def test_mixed_story_and_essay_omnibus_is_queued_for_review(self):
        collection = COLLECTION.replace(
            "THE RED ROOM........1\n The Open Window\n",
            "AN ENTERTAINING ARTICLE........1\nA SHORT HISTORY OF FABLES\n",
        ).replace("THE RED ROOM", "AN ENTERTAINING ARTICLE").replace(
            "THE OPEN WINDOW", "A SHORT HISTORY OF FABLES"
        )
        extractions, evidence = extract_collection(collection)
        self.assertEqual(len(extractions), 3)
        self.assertEqual(evidence["confidence"], "medium")
        self.assertEqual(evidence["non_story_title_indicators"], 2)

    def test_novel_chapter_numbers_do_not_create_story_boundaries(self):
        text = """CONTENTS
 THE NOVEL
 I
 II
 III
 A REAL STORY
 ANOTHER STORY

THE NOVEL

Novel opening.

I

First chapter.

II

Second chapter.

III

Third chapter.

A REAL STORY

Story prose.

ANOTHER STORY

More story prose.
"""
        extractions, _ = extract_collection(text)
        self.assertEqual(
            [item.title for item in extractions],
            ["THE NOVEL", "A REAL STORY", "ANOTHER STORY"],
        )
        self.assertIn("First chapter", extractions[0].text)
        self.assertIn("Third chapter", extractions[0].text)

    def test_word_length_filter_records_reason(self):
        extraction = extract_collection(COLLECTION)[0][0]
        self.assertEqual(classify_story_rejection(extraction), "too_short_or_front_matter")

    def test_normalized_hash_ignores_case_punctuation_and_spacing(self):
        self.assertEqual(
            normalized_text_hash("The  Red-Room!"),
            normalized_text_hash("the red room"),
        )

    def test_near_duplicates_are_reported_not_removed(self):
        base = " ".join(f"word{index % 30}" for index in range(800))
        variant = base.replace("word5", "wordfive", 2)
        records = [
            {
                "id": "one",
                "title": "One",
                "source_id": "a",
                "words": 800,
                "text": base,
            },
            {
                "id": "two",
                "title": "Two",
                "source_id": "b",
                "words": 800,
                "text": variant,
            },
        ]
        report = near_duplicate_candidates(records)
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["action"], "review_only")

    def test_catalog_rejects_mixed_poetry_or_essay_volumes(self):
        base = {
            "Text#": "1",
            "Type": "Text",
            "Issued": "2000-01-01",
            "Title": "Collected Stories",
            "Language": "en",
            "Authors": "Writer, 1800-1870",
            "Subjects": "Short stories, English; English essays",
            "LoCC": "PR",
            "Bookshelves": "",
        }
        self.assertEqual(catalog_candidates([base]), [])
        fiction = {**base, "Subjects": "Short stories, English; Fantasy fiction"}
        self.assertEqual(catalog_candidates([fiction]), [fiction])

    def test_corpus_validation_rejects_empty_or_malformed_records(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            validate_corpus_records([])
        malformed = {
            "id": "story-1",
            "source": "fixture",
            "source_id": "fixture-1",
            "title": "",
            "rights": "public_domain",
            "extraction_confidence": "high",
            "boundary_method": "fixture",
            "text_hash": hashlib.sha256(b"Some prose.").hexdigest(),
            "genres": [],
            "words": 2,
            "text": "Some prose.",
        }
        with self.assertRaisesRegex(ValueError, "title"):
            validate_corpus_records([malformed])


if __name__ == "__main__":
    unittest.main()
