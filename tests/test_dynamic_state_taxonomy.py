import copy
import unittest

from src.continuity_curriculum.common import canonical_json, sha256_bytes
from src.dynamic_state_taxonomy import (
    CATEGORIES,
    TaxonomyValidationError,
    aggregate_rows,
    build_taxonomy,
    classify_world_choice,
)


def seal(record):
    record["stable_hash"] = sha256_bytes(canonical_json(record).encode("utf-8"))
    return record


def pair(pair_id, split, history, candidates, current, family="object_location", distance="d128"):
    return seal({
        "pair_id": pair_id,
        "split": split,
        "abstract_counterfactual_variable": {"state_family": family},
        "metadata": {"distance_bucket": distance},
        "candidates": {name: {"value": value, "text": name} for name, value in candidates.items()},
        "worlds": {
            "A": {
                "state_events": [
                    {"sequence": index, "value": value, "evidence": str(index)}
                    for index, value in enumerate(history)
                ],
                "final_value": history[-1],
                "correct_candidate": current,
            }
        },
    })


def trial(source, scores, preference):
    return {
        "pair_id": source["pair_id"],
        "split": source["split"],
        "state_family": source["abstract_counterfactual_variable"]["state_family"],
        "distance": source["metadata"]["distance_bucket"],
        "normal": {"A": {**scores, "preference": preference}},
    }


class DynamicStateTaxonomyTests(unittest.TestCase):
    def test_all_categories_precedence_and_aliases(self):
        source = pair(
            "p", "test", ["initial", "older", "previous", "current"],
            {"C": "current", "P": "previous", "I": "initial", "O": "older", "N": "never"}, "C",
        )
        expected = {"C": "CURRENT", "P": "PREVIOUS", "I": "INITIAL", "O": "OLDER", "N": "NEVER_VALID"}
        for candidate, category in expected.items():
            self.assertEqual(classify_world_choice(source, "A", candidate)[0], category)
        self.assertEqual(classify_world_choice(source, "A", None),
                         ("OTHER_UNRESOLVED", ["OTHER_UNRESOLVED"]))

        depth_two = pair("p2", "test", ["first", "now"], {"C": "now", "P": "first"}, "C")
        self.assertEqual(classify_world_choice(depth_two, "A", "P"),
                         ("PREVIOUS", ["PREVIOUS", "INITIAL"]))
        repeated = pair("p3", "test", ["same", "middle", "same"],
                        {"C": "same", "M": "middle"}, "C")
        self.assertEqual(classify_world_choice(repeated, "A", "C"),
                         ("CURRENT", ["CURRENT", "INITIAL"]))

    def test_grouping_percentages_alias_counts_and_margins(self):
        first = pair("p1", "test", ["old", "new"], {"C": "new", "P": "old"}, "C")
        second = pair("p2", "test", ["old", "new"], {"C": "new", "P": "old"}, "C")
        evaluation = {"models": {"toy": {"test": {"trials": [
            trial(first, {"C": -1.0, "P": -3.0}, "C"),
            trial(second, {"C": -4.0, "P": -2.0}, "P"),
        ]}}}}
        result = build_taxonomy([first, second], evaluation)
        self.assertEqual([row["category"] for row in result["choices"]], ["CURRENT", "PREVIOUS"])
        group = result["groups"][0]
        self.assertEqual(
            (group["model"], group["split"], group["state_family"],
             group["transition_depth_bucket"], group["distance"]),
            ("toy", "test", "object_location", "2", "d128"),
        )
        self.assertEqual(group["counts"]["CURRENT"], 1)
        self.assertEqual(group["percentages"]["CURRENT"], 50.0)
        self.assertEqual(group["alias_counts"]["INITIAL"], 1)
        self.assertEqual(group["alias_percentages"]["INITIAL"], 50.0)
        self.assertEqual(group["mean_signed_lm_margin_current_minus_selected_alternative"], 0.0)
        self.assertEqual(sum(group["counts"].values()), group["choice_count"])
        self.assertEqual(set(group["counts"]), set(CATEGORIES))

        rows = copy.deepcopy(result["choices"])
        rows[1]["model"] = "other"
        rows[1]["split"] = "validation"
        rows[1]["state_family"] = "goal_intention"
        rows[1]["transition_depth_bucket"] = "4+"
        rows[1]["distance"] = "d512"
        self.assertEqual(len(aggregate_rows(rows)), 2)

    def test_tie_is_unresolved_but_retains_signed_margin(self):
        source = pair("tie", "test", ["old", "new"], {"C": "new", "P": "old"}, "C")
        evaluation = {"models": {"toy": {"test": {"trials": [
            trial(source, {"C": -2.0, "P": -2.0}, "tie")
        ]}}}}
        row = build_taxonomy([source], evaluation)["choices"][0]
        self.assertEqual(row["category"], "OTHER_UNRESOLVED")
        self.assertEqual(row["signed_lm_margin_current_minus_selected_alternative"], 0.0)

    def test_malformed_trial_is_rejected(self):
        source = pair("bad", "test", ["old", "new"], {"C": "new", "P": "old"}, "C")
        malformed = trial(source, {"C": -1.0, "P": -2.0}, "P")
        evaluation = {"models": {"toy": {"test": {"trials": [malformed]}}}}
        with self.assertRaisesRegex(TaxonomyValidationError, "contradicts scores"):
            build_taxonomy([source], evaluation)
        del malformed["normal"]["A"]["P"]
        with self.assertRaisesRegex(TaxonomyValidationError, "malformed.*score"):
            build_taxonomy([source], evaluation)


if __name__ == "__main__":
    unittest.main()
