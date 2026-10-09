import hashlib
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from tokenizers import Tokenizer

from src.audit_corpus_v2 import (
    bytelevel_token_bytes,
    classify_chunk_boundary,
    collection_key,
    deterministic_sample,
    dialogue_metrics,
    mandatory_sample_strata,
    main,
    normalize_name,
    ocr_signals,
    parse_creator_string,
    utf8_sequence_boundary,
    utf8_token_boundary,
    validate_metadata_linkage,
    validate_packed,
    verify_locked_inputs,
)
from src.tokenize_corpus import encode_document
from src.train_tokenizer import sha256_file


class CorpusV2AuditTests(unittest.TestCase):
    def test_hash_verification_fails_closed_on_tampering(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = Path(directory) / "locked.txt"
            path.write_text("original", encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            relative = path.relative_to(Path.cwd())
            spec = {"fixture": {"path": str(relative), "sha256": digest}}
            self.assertEqual(verify_locked_inputs(spec), {"fixture": digest})
            path.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Locked audit input changed"):
                verify_locked_inputs(spec)

    def test_hash_verification_is_read_only(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = Path(directory) / "locked.txt"
            path.write_bytes(b"immutable")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            before = path.stat()
            relative = path.relative_to(Path.cwd())
            verify_locked_inputs({"fixture": {"path": str(relative), "sha256": digest}})
            after = path.stat()
            self.assertEqual(path.read_bytes(), b"immutable")
            self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)

    def test_author_alias_normalization_only_strips_terminal_lifespan(self):
        self.assertEqual(
            normalize_name("Andersen, H. C. (Hans Christian), 1805-1875"),
            normalize_name("Andersen, H. C. (Hans Christian)"),
        )
        self.assertNotEqual(normalize_name("Henry, O."), normalize_name("O. Henry"))

    def test_creator_string_separates_primary_author_and_roles(self):
        parsed = parse_creator_string(
            "Coppée, François, 1842-1908; Matthews, Brander, 1852-1929 [Editor]; "
            "Learned, Walter, 1847-1915 [Translator]"
        )
        self.assertEqual(parsed["primary_author"], "Coppée, François, 1842-1908")
        self.assertEqual([item["role"] for item in parsed["contributors"]], ["editor", "translator"])
        editor = parse_creator_string("Jacobs, Joseph, 1854-1916 [Editor]")
        self.assertIsNone(editor["primary_author"])

    def test_collection_key_uses_stable_gutenberg_id_and_scoped_seed_title(self):
        self.assertEqual(
            collection_key({"source": "project_gutenberg", "source_id": "pg-12"}),
            "project_gutenberg:pg-12",
        )
        first = collection_key({"source": "seed", "source_collection": "A Collection!"})
        second = collection_key({"source": "other", "source_collection": "A Collection"})
        self.assertNotEqual(first, second)

    def test_dialogue_metric_counts_balanced_quotes(self):
        result = dialogue_metrics(
            'Narration here.\n\n“Two spoken words,” she said.\n\n"Three more words."\n\n‘Old style speech.’'
        )
        self.assertEqual(result["closed_quote_span_words"], 9)
        self.assertEqual(result["dialogue_opening_paragraphs"], 3)
        self.assertGreater(result["closed_quote_span_word_share"], 0.4)

    def test_dialogue_metric_handles_multiparagraph_reopen(self):
        result = dialogue_metrics("“First paragraph words.\n\n“Second paragraph closes.”")
        self.assertEqual(result["closed_quote_span_words"], 6)
        self.assertEqual(result["paragraph_reopen_count"], 1)
        self.assertEqual(result["unclosed_quote_span_count"], 0)
        self.assertEqual(result["dialogue_opening_paragraph_share"], 1.0)

    def test_dialogue_metric_handles_nested_and_unmatched_quotes(self):
        nested = dialogue_metrics("“Outer ‘inner words’ ending.”")
        self.assertEqual(nested["closed_quote_span_words"], 4)
        self.assertEqual(nested["unclosed_quote_span_count"], 0)
        unmatched = dialogue_metrics("Narration. “This never closes")
        self.assertEqual(unmatched["closed_quote_span_words"], 0)
        self.assertEqual(unmatched["unclosed_quote_span_count"], 1)

    def test_ocr_signals_are_independent_review_leads(self):
        result = ocr_signals("[Transcriber's Note: bad|scan] inter-\nrupted A1pha \ufffd")
        self.assertEqual(result["replacement_character"], 1)
        self.assertEqual(result["transcriber_or_editor_note"], 1)
        self.assertEqual(result["broken_line_hyphenation"], 1)
        self.assertEqual(result["suspicious_markup_glyph"], 1)
        self.assertEqual(result["digit_intrusion"], 1)

    def test_chunk_boundary_primary_precedence_and_features(self):
        paragraph = classify_chunk_boundary("End.\n\n", "Next", prefix="End.\n\n")
        self.assertEqual(paragraph["primary"], "paragraph")
        self.assertTrue(paragraph["sentence"])
        sentence = classify_chunk_boundary("The end. ", "Next", prefix="The end. ")
        self.assertEqual(sentence["primary"], "sentence")
        dialogue = classify_chunk_boundary("He said “half ", "spoken”, then", prefix="He said “half ")
        self.assertEqual(dialogue["primary"], "dialogue")
        subword = classify_chunk_boundary("contin", "uation", prefix="contin", next_token="uation")
        self.assertEqual(subword["primary"], "subword")

    def test_bytelevel_utf8_boundaries_cover_quotes_dash_and_currency(self):
        tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer.json")
        for text in ("“", "—", "£"):
            tokens = tokenizer.encode(text).tokens
            self.assertEqual(b"".join(bytelevel_token_bytes(token) or b"" for token in tokens), text.encode())
        self.assertTrue(utf8_token_boundary("â", "Ģľ")["utf8_codepoint_cut"])
        self.assertTrue(utf8_token_boundary("â", "ĢĶ")["utf8_codepoint_cut"])
        self.assertTrue(utf8_sequence_boundary(["â"], ["Ģ", "ľ"])["utf8_codepoint_cut"])
        self.assertTrue(utf8_sequence_boundary(["â", "Ģ"], ["ľ"])["utf8_codepoint_cut"])
        currency = tokenizer.encode("£").tokens
        self.assertEqual(currency, ["Â", "£"])
        self.assertTrue(utf8_token_boundary(*currency)["byte_sequence_cut"])
        self.assertTrue(utf8_token_boundary(*currency)["utf8_codepoint_cut"])
        self.assertFalse(utf8_token_boundary("âĢľ", "âĢĶ")["byte_sequence_cut"])
        before_character = utf8_sequence_boundary(["?"], ["Â", "Ķ"])
        self.assertTrue(before_character["adjacent_byte_fragment"])
        self.assertFalse(before_character["byte_sequence_cut"])

    def test_deterministic_sampling_is_order_independent(self):
        records = [
            {
                "id": value,
                "title": value,
                "author": "Author",
                "source": "seed",
                "source_collection": "Book",
                "words": 10,
                "genres": [genre],
                "text": f"Opening {value}. Closing {value}.",
            }
            for value, genre in (("a", "mystery"), ("b", "horror"), ("c", "fantasy"))
        ]
        metrics = {
            value: {
                "dialogue": {
                    "dialogue_opening_paragraph_share": 0.3 if value == "a" else 0.0
                },
                "ocr": {"replacement_character": int(value == "b")},
            }
            for value in ("a", "b", "c")
        }
        first = deterministic_sample(records, metrics, seed=7, count=3)
        second = deterministic_sample(list(reversed(records)), metrics, seed=7, count=3)
        self.assertEqual(first, second)
        required = set(mandatory_sample_strata(records, metrics))
        covered = {label for sample in first for label in sample["strata"]}
        self.assertTrue(required <= covered)

    def test_sampling_fails_if_count_cannot_cover_mandatory_selections(self):
        records = [
            {
                "id": value,
                "title": value,
                "author": "Author",
                "source": value,
                "source_collection": value,
                "boundary_method": value,
                "words": words,
                "genres": [genre],
                "text": value,
            }
            for value, words, genre in (("a", 10, "mystery"), ("b", 100, "horror"))
        ]
        metrics = {
            "a": {"dialogue": {"dialogue_opening_paragraph_share": 0.0}, "ocr": {"x": 0}},
            "b": {"dialogue": {"dialogue_opening_paragraph_share": 0.5}, "ocr": {"x": 1}},
        }
        with self.assertRaisesRegex(ValueError, "cannot cover"):
            deterministic_sample(records, metrics, seed=1, count=1)

    def _packed_fixture(self, root: Path):
        tokenizer_path = Path("data/tokenizer/tokenizer.json")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        records = [
            {
                "id": split,
                "title": f"Title {split}",
                "text": f"Text for {split} with £ and — punctuation.",
                "genres": [],
                "words": 7,
            }
            for split in ("train", "validation", "test")
        ]
        assignments = {record["id"]: record["id"] for record in records}
        corpus_hash = "c" * 64
        corpus_path = "synthetic/corpus.jsonl"
        split_path = root / "splits.json"
        split_payload = {
            "corpus_path": corpus_path,
            "corpus_sha256": corpus_hash,
            "assignments": assignments,
        }
        split_path.write_text(json.dumps(split_payload), encoding="utf-8")
        inputs = {
            "corpus": {"path": corpus_path, "sha256": corpus_hash},
            "tokenizer": {"path": str(tokenizer_path), "sha256": sha256_file(tokenizer_path)},
            "split": {"path": str(split_path), "sha256": sha256_file(split_path)},
        }
        split_metadata = {}
        for record in records:
            split = record["id"]
            ids, control = encode_document(tokenizer, record)
            bin_path = root / f"{split}.bin"
            np.asarray(ids, dtype=np.uint16).tofile(bin_path)
            index_path = root / f"{split}.index.json"
            index = {
                "split": split,
                "documents": [{
                    "id": split,
                    "offset": 0,
                    "length": len(ids),
                    "words": record["words"],
                    "genre_control_token": control,
                    "sequence_count": math.ceil((len(ids) - 1) / 8),
                }],
            }
            index_path.write_text(json.dumps(index), encoding="utf-8")
            inputs[f"{split}_bin"] = {"path": str(bin_path), "sha256": sha256_file(bin_path)}
            inputs[f"{split}_index"] = {
                "path": str(index_path), "sha256": sha256_file(index_path)
            }
            split_metadata[split] = {
                "document_count": 1,
                "token_count": len(ids),
                "sequence_count": math.ceil((len(ids) - 1) / 8),
                "bin_path": str(bin_path),
                "bin_sha256": sha256_file(bin_path),
                "index_path": str(index_path),
                "index_sha256": sha256_file(index_path),
            }
        metadata_path = root / "metadata.json"
        metadata = {
            "corpus_path": corpus_path,
            "corpus_sha256": corpus_hash,
            "split_path": str(split_path),
            "split_sha256": sha256_file(split_path),
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_hash": sha256_file(tokenizer_path),
            "dtype": "uint16",
            "splits": split_metadata,
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        inputs["packed_metadata"] = {
            "path": str(metadata_path), "sha256": sha256_file(metadata_path)
        }
        return {"inputs": inputs, "sequence_length": 8}, records, tokenizer, metadata, split_payload

    def test_validate_packed_rejects_canonical_token_tampering(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory).relative_to(Path.cwd())
            config, records, tokenizer, metadata, _ = self._packed_fixture(root)
            token_counts, report = validate_packed(config, records, tokenizer)
            self.assertEqual(set(token_counts), {"train", "validation", "test"})
            self.assertEqual(report["exact_canonical_reencoded_document_count"], 3)
            bin_path = Path(config["inputs"]["train_bin"]["path"])
            values = np.fromfile(bin_path, dtype=np.uint16)
            values[3] = (int(values[3]) + 1) % tokenizer.get_vocab_size()
            values.tofile(bin_path)
            digest = sha256_file(bin_path)
            config["inputs"]["train_bin"]["sha256"] = digest
            metadata["splits"]["train"]["bin_sha256"] = digest
            Path(config["inputs"]["packed_metadata"]["path"]).write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "canonical encode_document"):
                validate_packed(config, records, tokenizer)

    def test_validate_packed_rejects_index_misattribution(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory).relative_to(Path.cwd())
            config, records, tokenizer, metadata, _ = self._packed_fixture(root)
            index_path = Path(config["inputs"]["test_index"]["path"])
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["documents"][0]["words"] = 999
            index_path.write_text(json.dumps(index), encoding="utf-8")
            digest = sha256_file(index_path)
            config["inputs"]["test_index"]["sha256"] = digest
            metadata["splits"]["test"]["index_sha256"] = digest
            Path(config["inputs"]["packed_metadata"]["path"]).write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "misattributed"):
                validate_packed(config, records, tokenizer)

    def test_metadata_linkage_rejects_wrong_corpus_hash(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory).relative_to(Path.cwd())
            config, _, _, metadata, split_payload = self._packed_fixture(root)
            metadata["corpus_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "corpus_sha256"):
                validate_metadata_linkage(config, metadata, split_payload)

    def test_main_reports_configured_output_path(self):
        report = {
            "status": "complete",
            "counts": {"stories": 3, "packed_tokens": 30},
            "output_path": "custom/output/audit-report.json",
        }
        output = io.StringIO()
        with patch("src.audit_corpus_v2.parse_args"), patch(
            "src.audit_corpus_v2.audit", return_value=report
        ), redirect_stdout(output):
            main()
        self.assertEqual(json.loads(output.getvalue())["report"], report["output_path"])


if __name__ == "__main__":
    unittest.main()
