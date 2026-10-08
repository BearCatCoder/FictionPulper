import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.continuity_curriculum.common import canonical_json, sha256_bytes, sha256_file
from src.evaluate_dynamic_logit import (
    DynamicEvaluationError,
    aggregate_results,
    build_fact_removal_set,
    categorize_candidate,
    decision_result,
    depth_availability,
    group_candidate_sets,
    margin_collapse_audit,
    read_jsonl,
    score_token_spans,
    switch_audit,
    verify_file,
)


def candidate(name, category, aliases, tokens, *, depth=1, maximum_depth=2, world="A"):
    return {
        "pair": "pair", "world": world, "boundary": "final", "depth": depth,
        "candidate": name, "candidate_text": f" {name}", "category": category,
        "all_applicable_categories": aliases, "input_token_ids": tokens,
        "token_ids": tokens[1:-1], "decision_start": 1, "decision_end": len(tokens) - 1,
        "decision_offsets": {"context_token_end": 0}, "family": "latest_state_update",
        "distance": 17, "metadata": {"distance_bucket": "short", "transition_count": maximum_depth},
    }


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.logits = logits

    def forward(self, values, labels=None):
        return self.logits[:values.shape[0], :values.shape[1]] + self.anchor * 0.0, None


class CharacterTokenizer:
    class Encoding:
        def __init__(self, values):
            self.ids = values

    def encode(self, text):
        return self.Encoding([ord(character) for character in text])


class DynamicEvaluationTests(unittest.TestCase):
    def test_taxonomy_precedence_alias_counts_and_corrected_margin(self):
        self.assertEqual(categorize_candidate("a", ["a"]), ("CURRENT", ["CURRENT", "INITIAL"]))
        self.assertEqual(categorize_candidate("a", ["a", "b"]),
                         ("PREVIOUS", ["PREVIOUS", "INITIAL"]))
        self.assertEqual(categorize_candidate("a", ["a", "b", "c"]),
                         ("INITIAL", ["INITIAL", "OLDER"]))
        self.assertEqual(categorize_candidate("b", ["a", "b", "c", "d"]), ("OLDER", ["OLDER"]))
        self.assertEqual(categorize_candidate("z", ["a"]), ("NEVER_VALID", ["NEVER_VALID"]))
        current = candidate("X", "CURRENT", ["CURRENT", "INITIAL"], [0, 1, 9])
        stale = candidate("Y", "NEVER_VALID", ["NEVER_VALID"], [0, 2, 9])
        grouped = group_candidate_sets([stale, current])
        row = decision_result(next(iter(grouped.values())), [2.0, 1.0])
        # Sorted X/Y means the CURRENT score is 2.0. INITIAL on CURRENT is not a competitor.
        self.assertIsNone(row["margins"]["INITIAL"])
        self.assertEqual(row["margins"]["NEVER_VALID"], 1.0)
        summary = aggregate_results([row])["overall"]
        self.assertEqual(summary["precedence_counts"]["CURRENT"], 1)
        self.assertEqual(summary["alias_counts"]["INITIAL"], 1)
        self.assertEqual(summary["current_vs_noncurrent_margins"]["INITIAL"]["unavailable_count"], 1)

    def test_multi_token_scores_use_mean_not_sum(self):
        logits = torch.zeros(2, 3, 5)
        model = FixedLogitModel(logits)
        records = [
            {"input_token_ids": [0, 2, 4], "decision_start": 1, "decision_end": 2},
            {"input_token_ids": [0, 2, 3, 4], "decision_start": 1, "decision_end": 3},
        ]
        scores = score_token_spans(model, records, device=torch.device("cpu"), batch_size=2,
                                   pad_id=0, use_bf16=False)
        self.assertAlmostEqual(scores[0], -torch.log(torch.tensor(5.0)).item())
        self.assertAlmostEqual(scores[0], scores[1])

    def test_full_vs_removal_margin_collapse_construction(self):
        base = {"pair": "p", "world": "A", "boundary": "final", "depth": 1,
                "current_score": 3.0, "best_noncurrent_score": 1.0}
        removed = {**base, "current_score": 1.2, "best_noncurrent_score": 1.0}
        audit = margin_collapse_audit([base], [removed])
        self.assertEqual(audit["collapse_rate"], 1.0)
        self.assertAlmostEqual(audit["mean_absolute_margin_collapse"], 1.8)

        tokenizer = CharacterTokenizer()
        context = "Old evidence. Latest evidence. Tail remains."
        candidate_text = " X"
        context_ids = tokenizer.encode(context).ids
        candidate_ids = tokenizer.encode(candidate_text).ids
        record = candidate("X", "CURRENT", ["CURRENT"], [7, *context_ids, *candidate_ids, 8], depth=2)
        record.update({
            "candidate_text": candidate_text, "token_ids": candidate_ids,
            "decision_start": 1 + len(context_ids),
            "decision_end": 1 + len(context_ids) + len(candidate_ids),
            "decision_offsets": {"context_token_end": len(context_ids)},
        })
        pair = {"worlds": {"A": {
            "context": context, "context_token_ids": context_ids,
            "state_events": [
                {"evidence": "Old evidence."}, {"evidence": "Latest evidence."},
            ],
        }}}
        rebuilt = build_fact_removal_set([record], pair, tokenizer)[0]
        expected_context = "Old evidence.  Tail remains."
        expected_ids = [7, *tokenizer.encode(expected_context + candidate_text).ids, 8]
        self.assertEqual(rebuilt["input_token_ids"], expected_ids)
        self.assertEqual(rebuilt["input_token_ids"][rebuilt["decision_start"]:rebuilt["decision_end"]],
                         candidate_ids)

    def test_context_swap_switch_audit(self):
        def row(world, selected, current):
            return {"pair": "p", "world": world, "boundary": "final", "depth": 2,
                    "selected_candidate": selected, "current_candidate": current}
        full = [row("A", "X", "X"), row("B", "Y", "Y")]
        swapped = [row("A", "Y", "Y"), row("B", "X", "X")]
        audit = switch_audit(full, swapped)
        self.assertEqual(audit["preference_switch_rate"], 1.0)
        self.assertEqual(audit["matches_opposite_world_rate"], 1.0)
        self.assertTrue(audit["opposite_label_integrity"])

    def test_unavailable_depths_are_explicit(self):
        first = decision_result([
            candidate("X", "CURRENT", ["CURRENT", "INITIAL"], [0, 1, 9]),
            candidate("Y", "NEVER_VALID", ["NEVER_VALID"], [0, 2, 9]),
        ], [1.0, 0.0])
        availability = depth_availability([first])
        self.assertEqual(availability["1"]["status"], "available")
        self.assertEqual(availability["2"]["available_worlds"], 1)
        self.assertEqual(availability["3"]["status"], "unavailable")
        self.assertEqual(availability["4+"]["unavailable_worlds"], 1)

    def test_malformed_and_hash_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            malformed = root / "bad.jsonl"
            malformed.write_text("{not json}\n", encoding="utf-8")
            with self.assertRaisesRegex(DynamicEvaluationError, "Invalid JSON"):
                read_jsonl(malformed)

            valid = root / "valid.jsonl"
            record = {"split": "test", "value": 1}
            record["stable_hash"] = sha256_bytes(canonical_json(record).encode("utf-8"))
            # Hash was computed with the absent stable_hash field, matching the sealed convention.
            valid.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DynamicEvaluationError, "Hash mismatch"):
                verify_file(valid, "0" * 64, "fixture")
            loaded = read_jsonl(valid, expected_split="test")
            tampered = copy.deepcopy(loaded[0])
            tampered["value"] = 2
            valid.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DynamicEvaluationError, "Stable hash mismatch"):
                read_jsonl(valid, expected_split="test")


if __name__ == "__main__":
    unittest.main()
