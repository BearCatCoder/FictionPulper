import unittest
from collections import Counter

from src.prepare_corpus import (
    GUTENBERG_BOOK_CANONICAL_FIELDS,
    GUTENBERG_BOOK_SOURCE,
    SHORT_STORY_CANONICAL_FIELDS,
    SHORT_STORY_FIELDS,
    SHORT_STORY_SOURCE,
    assign_document_splits,
    canonicalize_gutenberg_book,
    canonicalize_short_story,
    clean_book_text,
    clean_story_text,
    remove_obvious_story_boilerplate,
    prepare_records,
    validate_record,
    validate_schema,
)


def story_record(**overrides):
    record = {
        "id": "story-123",
        "author": "Example, Alice",
        "book": "Example Collection",
        "title": "A Complete Story",
        "words": 4,
        "kind": "story",
        "prompt_brief": "Generated prompt that must not be copied.",
        "prompt_detailed": "Generated detailed prompt that must not be copied.",
        "genres": ["mystery", "adventure"],
        "text": "First paragraph.\n\nSecond paragraph.",
        "license": "Public domain (Project Gutenberg)",
    }
    record.update(overrides)
    return record


def book_record(**overrides):
    record = {
        "etextno": 123,
        "book_title": "The Haunted Detective Adventure",
        "author": "Example, Alice",
        "issued": "2020-01-01",
        "text": "A story paragraph.\n\n" * 20,
    }
    record.update(overrides)
    return record


class CleanTextTests(unittest.TestCase):
    def test_story_cleaning_preserves_internal_paragraphs_and_prose(self):
        text = '  “Run!” she said.  \r\n\r\nHe ran.  \r\n\r\n\r\nThe end.\x00  '
        self.assertEqual(
            clean_story_text(text),
            '“Run!” she said.  \n\nHe ran.  \n\n\nThe end.',
        )

    def test_book_cleaning_strips_explicit_gutenberg_wrapper(self):
        text = (
            "license\n*** START OF THE PROJECT GUTENBERG EBOOK A BOOK ***\n"
            "Story.\n\n\n\nNext.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK A BOOK ***\nlicense"
        )
        self.assertEqual(clean_book_text(text), "Story.\n\nNext.")

    def test_only_obvious_story_boilerplate_is_removed(self):
        text = (
            "Opening prose.\n\n"
            "Note: Project Gutenberg also has an HTML version of this file at http://example.test.\n\n"
            "Closing prose."
        )
        self.assertEqual(remove_obvious_story_boilerplate(text), "Opening prose.\n\nClosing prose.")


class ShortStoryRecordTests(unittest.TestCase):
    def test_canonical_story_has_exact_fields_and_no_prompts(self):
        record = canonicalize_short_story(story_record())
        self.assertEqual(set(record), SHORT_STORY_CANONICAL_FIELDS)
        self.assertNotIn("prompt_brief", record)
        self.assertNotIn("prompt_detailed", record)
        self.assertEqual(record["source_collection"], "Example Collection")
        self.assertEqual(record["rights"], "public_domain")

    def test_non_story_kind_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "kind"):
            canonicalize_short_story(story_record(kind="essay"))

    def test_non_public_domain_license_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "license"):
            canonicalize_short_story(story_record(license="unknown"))

    def test_empty_title_and_text_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "title"):
            canonicalize_short_story(story_record(title=""))
        record = canonicalize_short_story(story_record(text="", words=0))
        with self.assertRaisesRegex(ValueError, "text"):
            validate_record(record, SHORT_STORY_SOURCE)

    def test_word_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "word count"):
            canonicalize_short_story(story_record(words=99))

    def test_schema_requires_original_story_fields_but_allows_prompts(self):
        observed = validate_schema(
            {**{field: object() for field in SHORT_STORY_FIELDS}, "prompt_brief": object()},
            SHORT_STORY_FIELDS,
        )
        self.assertIn("prompt_brief", observed)
        with self.assertRaisesRegex(RuntimeError, "Unsupported dataset schema"):
            validate_schema({"id": object(), "text": object()}, SHORT_STORY_FIELDS)

    def test_duplicate_ids_and_cleaned_texts_are_rejected(self):
        rows = [
            story_record(id="one"),
            story_record(id="one", text="Different story text.", words=3),
            story_record(id="two"),
            story_record(id="three", text="Unique final story.", words=3),
        ]
        records, counters, duplicates = prepare_records(rows, source=SHORT_STORY_SOURCE)
        self.assertEqual([record["id"] for record in records], ["one", "three"])
        self.assertEqual(counters["duplicate_ids"], 1)
        self.assertEqual(counters["duplicate_texts"], 1)
        self.assertEqual(duplicates[0]["canonical_id"], "one")
        self.assertEqual(duplicates[0]["duplicate_id"], "two")
        self.assertFalse(duplicates[0]["metadata_conflict"])

    def test_split_assignments_are_deterministic_and_disjoint(self):
        records = [
            canonicalize_short_story(story_record(id=f"story-{index}"))
            for index in range(40)
        ]
        first = assign_document_splits(records, seed=1337)
        second = assign_document_splits(list(reversed(records)), seed=1337)
        self.assertEqual(first, second)
        self.assertEqual(Counter(first.values()), {"train": 34, "validation": 3, "test": 3})
        self.assertEqual(set(first), {record["id"] for record in records})


class GutenbergBookTests(unittest.TestCase):
    def test_optional_book_adapter_remains_available(self):
        record = canonicalize_gutenberg_book(book_record())
        self.assertEqual(set(record), GUTENBERG_BOOK_CANONICAL_FIELDS)
        self.assertEqual(record["id"], "pg-123")
        self.assertEqual(record["genres"], ["mystery", "horror", "adventure"])
        validate_record(record, GUTENBERG_BOOK_SOURCE)


if __name__ == "__main__":
    unittest.main()
