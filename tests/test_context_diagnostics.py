import unittest

from tokenizers import Tokenizer

from src.context_diagnostics import (
    build_context_retention_protocol,
    build_narrative_state_protocol,
    longest_repeated_token_span,
    repetition_metrics,
)


class ContextDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer.json")

    def test_context_prompts_cross_1024_but_fit_2048(self):
        protocol = build_context_retention_protocol(self.tokenizer)
        self.assertEqual(len(protocol["entries"]), 3)
        for entry in protocol["entries"]:
            self.assertGreater(entry["prompt_token_count"], 1024)
            self.assertLessEqual(entry["prompt_token_count"], 2046)
            self.assertGreater(entry["tokens_after_fact_prefix"], 1024)

    def test_narrative_state_prompts_and_facts_fit_1024(self):
        protocol = build_narrative_state_protocol(self.tokenizer)
        self.assertEqual(protocol["source_suite"], "fictionpulper-context-retention-v1")
        self.assertEqual(len(protocol["entries"]), 3)
        for entry in protocol["entries"]:
            self.assertLessEqual(entry["prompt_token_count"] + 2, 1024)
            self.assertGreater(entry["tokens_after_fact_prefix"], 400)
            self.assertTrue(entry["fact_prefix_visible_at_1024"])

    def test_longest_repeated_token_span_requires_non_overlapping_spans(self):
        self.assertEqual(longest_repeated_token_span([1, 2, 3, 1, 2, 3]), 3)
        self.assertEqual(longest_repeated_token_span([1, 1, 1]), 1)
        self.assertEqual(longest_repeated_token_span([1, 2, 3]), 0)

    def test_repetition_metrics_detect_repeated_text(self):
        repeated = repetition_metrics(self.tokenizer, "The bell rang. The bell rang.")
        varied = repetition_metrics(self.tokenizer, "The bell rang. A door opened.")
        self.assertGreater(repeated["repeated_4gram_rate"], varied["repeated_4gram_rate"])
        self.assertGreater(repeated["repeated_sentence_rate"], 0.0)
        self.assertGreater(repeated["longest_repeated_token_span"], 0)


if __name__ == "__main__":
    unittest.main()
