import unittest
from pathlib import Path

from src.evaluate_context_retention import fact_visibility, parse_args
from src.report_capacity50m import validate_generation_pair
from src.train import GENERATION_SETTINGS


class CapacityEvaluationTests(unittest.TestCase):
    def test_capacity_visibility_includes_controls_and_full_generation(self):
        entry = {"id": "longest", "prompt_token_count": 647}
        visibility = fact_visibility(entry, context=1024, max_new_tokens=256)
        self.assertEqual(visibility["control_token_count"], 2)
        self.assertEqual(visibility["maximum_sequence_tokens"], 905)
        self.assertTrue(visibility["fact_prefix_visible_for_full_generation"])

    def test_visibility_rejects_prefix_eviction_during_generation(self):
        entry = {"id": "too-long", "prompt_token_count": 900}
        with self.assertRaisesRegex(RuntimeError, "would leave the 1024-token context"):
            fact_visibility(entry, context=1024, max_new_tokens=128)

    def test_context2k_out_of_window_baseline_remains_supported(self):
        entry = {"id": "context2k", "prompt_token_count": 1144}
        visibility = fact_visibility(entry, context=1024, max_new_tokens=128)
        self.assertFalse(visibility["fact_prefix_visible_at_generation_start"])
        self.assertFalse(visibility["fact_prefix_visible_for_full_generation"])

    def test_context2k_defaults_remain_available(self):
        args = parse_args(
            [
                "--baseline-checkpoint",
                "baseline.pt",
                "--candidate-checkpoint",
                "candidate.pt",
                "--protocol",
                "protocol.json",
                "--tokenizer",
                "tokenizer.json",
                "--output",
                "output.json",
            ]
        )
        self.assertEqual(args.baseline_label, "compute5_context1024")
        self.assertEqual(args.candidate_label, "context2k")
        self.assertEqual(args.baseline_context, 1024)
        self.assertEqual(args.candidate_context, 2048)
        self.assertEqual(args.output, Path("output.json"))

    def test_arbitrary_labels_and_contexts_parse(self):
        args = parse_args(
            [
                "--baseline-checkpoint",
                "baseline.pt",
                "--candidate-checkpoint",
                "candidate.pt",
                "--protocol",
                "protocol.json",
                "--tokenizer",
                "tokenizer.json",
                "--output",
                "output.json",
                "--baseline-label",
                "small",
                "--candidate-label",
                "large",
                "--baseline-context",
                "1024",
                "--candidate-context",
                "1024",
            ]
        )
        self.assertEqual((args.baseline_label, args.candidate_label), ("small", "large"))
        self.assertEqual((args.baseline_context, args.candidate_context), (1024, 1024))

    def test_generation_validation_checks_per_prompt_seeds(self):
        def suite() -> dict:
            return {
                "sample_seed": 100,
                "generation_settings": dict(GENERATION_SETTINGS),
                "entries": [
                    {"prompt": f"prompt-{index}", "sample_seed": 100 + index}
                    for index in range(10)
                ],
            }

        baseline = suite()
        candidate = suite()
        validate_generation_pair(baseline, candidate)
        candidate["entries"][4]["sample_seed"] = 999
        with self.assertRaisesRegex(RuntimeError, "per-prompt seed"):
            validate_generation_pair(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
