import json
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer

from src.train_tokenizer import (
    SPECIAL_TOKENS,
    build_tokenizer,
    load_corpus,
    load_split_assignments,
    save_tokenizer,
)


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "id": "one",
                "title": "The Locked Room",
                "text": "The rain began at midnight.\n\n“Who is there?” Mallory asked.",
            },
            {
                "id": "two",
                "title": "The Rocket",
                "text": "The rocket was silent.\n\nThen the signal came from Mars.",
            },
        ]

    def test_special_tokens_vocab_size_round_trip_and_reload(self):
        tokenizer = build_tokenizer(
            self.records,
            vocab_size=300,
            min_frequency=1,
            show_progress=False,
        )
        self.assertEqual(tokenizer.get_vocab_size(), 300)
        self.assertEqual(
            [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS],
            list(range(len(SPECIAL_TOKENS))),
        )
        sample = 'Old prose—unchanged.\n\n“Who’s there?”'
        self.assertEqual(tokenizer.decode(tokenizer.encode(sample).ids), sample)
        self.assertEqual(tokenizer.encode("<|story|>").tokens, ["<|story|>"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            save_tokenizer(tokenizer, path)
            reloaded = Tokenizer.from_file(str(path))
            self.assertEqual(reloaded.decode(reloaded.encode(sample).ids), sample)
            self.assertEqual(reloaded.get_vocab_size(), 300)

    def test_corpus_loader_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in [self.records[0], self.records[0]]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate corpus id"):
                load_corpus(path)

    def test_split_loader_requires_every_document_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            path.write_text(json.dumps({"assignments": {"one": "train"}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly match"):
                load_split_assignments(path, self.records)


if __name__ == "__main__":
    unittest.main()
