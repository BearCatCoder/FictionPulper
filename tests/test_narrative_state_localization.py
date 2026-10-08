import json
import unittest
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer

from src.frozen_state_probe import retain_layerwise_held_out, sha256_file
from src.narrative_state_localization import (
    EXPECTED_NARRATIVE_PROTOCOL_SHA256,
    build_counterfactual_case,
    build_counterfactual_protocol,
    canonical_hash,
    encode_fixed_candidates,
    first_divergence,
    mean_decision_span_log_probability,
    rank_scores,
    recovery_error,
    rollout_replacement,
    score_candidate_ids,
    select_next_token,
    stable_sampling_seed,
    validate_counterfactual_cases,
    validate_forced_choice_protocol,
    validate_rollout_context,
    aggregate_context_swap,
)
from src.report_narrative_state_localization import (
    ARTIFACT_NAMES,
    _aggregate_metric_records,
    classify_failure,
    classify_probe_states,
    compact_probe,
)


ROOT = Path(__file__).resolve().parents[1]


class TinyScorer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, tokens):
        batch, length = tokens.shape
        logits = torch.zeros(batch, length, 16, device=tokens.device) + self.weight * 0
        return logits, None


class NarrativeStateLocalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(ROOT / "data/tokenizer/tokenizer.json"))

    def test_probe_layerwise_heldout_is_opt_in_and_localization_enables_it(self):
        self.assertFalse(retain_layerwise_held_out({"probe": {}}))
        config = yaml.safe_load((ROOT / "experiments/narrative-state-localization-v1/probe-config.yaml").read_text())
        self.assertTrue(retain_layerwise_held_out(config))
        self.assertEqual(config["probe"]["seeds"], [1001, 1002, 1003])
        self.assertEqual(config["probe"]["epochs"], 30)

    def test_preflight_records_context2k_and_all_required_tags(self):
        preflight = json.loads((ROOT / "experiments/narrative-state-localization-v1/preflight.json").read_text())
        self.assertTrue(preflight["tracked_worktree_clean"])
        self.assertTrue(preflight["master_synchronized_with_origin_master"])
        self.assertEqual(len(preflight["resolved_tags"]), 5)
        self.assertIn("fictionpulper-15m-data30m-context2k-v1", preflight["resolved_tags"])

    def test_decision_span_is_exact_and_length_normalized(self):
        logits = torch.zeros(1, 5, 4)
        tokens = torch.tensor([[0, 1, 2, 3, 1]])
        logits[0, 2, 3] = 2.0
        logits[0, 3, 1] = 4.0
        score = mean_decision_span_log_probability(logits, tokens, 3)
        expected = torch.stack((
            torch.log_softmax(logits[0, 2], -1)[3],
            torch.log_softmax(logits[0, 3], -1)[1],
        )).mean()
        torch.testing.assert_close(score[0], expected)
        one_token = mean_decision_span_log_probability(logits[:, :4], tokens[:, :4], 3)
        self.assertNotAlmostEqual(float(score[0]), float(one_token[0]))

    def test_scoring_freezes_model_and_never_creates_gradients(self):
        model = TinyScorer()
        scores = score_candidate_ids(model, [1, 2], [[3], [4, 5]], device=torch.device("cpu"), use_bf16=False)
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_counterfactual_reversal_removal_and_latest_update(self):
        case = build_counterfactual_case(
            family="object_location", distance="d064", split="test", source_id="sealed-test-1",
            names=["Beatrice", "Anton"], objects=["velvet case"],
            locations=["canal bridge", "parish library"], difficulty=2,
        )
        self.assertEqual(case["expected"], {"a": "x", "b": "y", "late_update": "y"})
        self.assertNotIn("left the velvet case", case["contexts"]["removed"])
        self.assertIn("Correction recorded later", case["contexts"]["late_update"])
        self.assertEqual(case["candidates"], dict(case["candidates"]))
        validate_counterfactual_cases([case], self.tokenizer)

    def test_ownership_transfer_candidates_follow_the_recipient(self):
        case = build_counterfactual_case(
            family="ownership", distance="d064", split="test", source_id="sealed-test-1",
            names=["Beatrice", "Anton"], objects=["velvet case"],
            locations=["canal bridge", "parish library"], difficulty=2,
        )
        self.assertIn("to Anton", case["contexts"]["a"])
        self.assertIn("Anton now owned", case["candidates"]["x"])
        self.assertIn("to Beatrice", case["contexts"]["b"])
        self.assertIn("Beatrice now owned", case["candidates"]["y"])

    def test_counterfactual_hash_and_source_isolation_are_deterministic(self):
        kwargs = dict(family="ownership", distance="d064", split="test", source_id="x",
                      names=["Beatrice", "Anton"], objects=["velvet case"],
                      locations=["canal bridge", "parish library"], difficulty=1)
        first = build_counterfactual_case(**kwargs)
        second = build_counterfactual_case(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["sha256"], canonical_hash({key: value for key, value in first.items() if key != "sha256"}))
        other = build_counterfactual_case(**{**kwargs, "distance": "d128", "split": "generalization_holdout", "source_id": "y"})
        with self.assertRaisesRegex(ValueError, "lexical pools overlap"):
            validate_counterfactual_cases([first, other], self.tokenizer)

    def test_real_counterfactual_protocol_is_deterministic_and_reports_supported_coverage(self):
        curriculum = ROOT / "data/narrative_curriculum_v1"
        first = build_counterfactual_protocol(curriculum, self.tokenizer)
        second = build_counterfactual_protocol(curriculum, self.tokenizer)
        self.assertEqual(canonical_hash(first), canonical_hash(second))
        self.assertEqual(first["target_case_count"], 63)
        self.assertEqual(first["case_count"], 56)
        self.assertEqual(len(first["source_artifacts"]), 5)

    def test_immutable_13fact_protocol_exact_order_and_hash(self):
        immutable_path = ROOT / "experiments/fictionpulper-50m-data30m-v1/narrative-state-protocol.json"
        protocol_path = ROOT / "experiments/narrative-state-localization-v1/13fact-forced-choice-protocol.json"
        self.assertEqual(sha256_file(immutable_path), EXPECTED_NARRATIVE_PROTOCOL_SHA256)
        immutable = json.loads(immutable_path.read_text())
        protocol = json.loads(protocol_path.read_text())
        validate_forced_choice_protocol(protocol, immutable, sha256_file(immutable_path))
        self.assertEqual(len(protocol["entries"]), 13)
        self.assertTrue(all(len(entry["candidates"]) >= 3 for entry in protocol["entries"]))
        prompts = {entry["id"]: entry["prompt"] for entry in immutable["entries"]}
        for entry in protocol["entries"]:
            context, candidates = encode_fixed_candidates(
                self.tokenizer,
                prompts[entry["sealed_entry_id"]] + entry["bridge"],
                entry["candidates"],
            )
            self.assertTrue(context)
            self.assertTrue(all(candidates))

    def test_ranking_and_rollout_replacement_boundaries(self):
        self.assertEqual(rank_scores([-2.0, -3.0, -4.0])["correct_rank"], 1)
        replaced, boundaries = rollout_replacement([1, 2, 3, 4, 5], [8, 9], 2)
        self.assertEqual(replaced, [1, 2, 3, 8, 9])
        self.assertEqual(boundaries["decision_start"], 5)
        self.assertEqual(first_divergence([4, 5], [4, 9]), 1)
        validate_rollout_context([1] * 1000, [[2] * 24], 256)
        with self.assertRaisesRegex(ValueError, "maximum context"):
            validate_rollout_context([1] * 1000, [[2] * 25], 256)
        with self.assertRaises(ValueError):
            rollout_replacement([1, 2], [3], 2)

    def test_sampling_seed_and_recovery_controls(self):
        self.assertEqual(stable_sampling_seed(11337, 4, 1), stable_sampling_seed(11337, 4, 1))
        self.assertNotEqual(stable_sampling_seed(11337, 4, 0), stable_sampling_seed(11337, 4, 1))
        recovered = recovery_error([-1.0, -2.0], [-1.0, -2.0 + 1e-8])
        self.assertTrue(recovered["within_tolerance"])
        self.assertFalse(recovery_error([-1.0], [-1.1])["within_tolerance"])
        settings = {"temperature": 0.8, "top_k": 2, "top_p": 0.95}
        token = select_next_token(
            torch.tensor([0.0, 1.0, 2.0]), mode="sampled", settings=settings,
            generator=torch.Generator().manual_seed(11337),
        )
        self.assertIn(token, (1, 2))
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            select_next_token(
                torch.zeros(1, 3), mode="sampled", settings=settings,
                generator=torch.Generator().manual_seed(11337),
            )

    def test_context_swap_aggregation_has_primary_overall_statistic(self):
        trial = {
            "family": "ownership", "distance": "d064", "difficulty": 1,
            "reversal_correct": True, "context_direction_accuracy": 1.0,
            "mean_signed_correct_margin": 0.5, "absolute_preference_change": 1.0,
            "removed_absolute_margin": 0.1, "removed_preference_bias": 0.1,
            "irrelevant_absolute_margin": 0.2, "irrelevant_preference_bias": -0.2,
            "latest_state_update_correct": True,
        }
        result = aggregate_context_swap([trial])
        self.assertEqual(result["overall"]["reversal_correct"], 1.0)
        self.assertEqual(result["by_family"]["ownership"]["count"], 1)

    def test_report_classification_is_conservative_and_unsealed(self):
        self.assertEqual(classify_failure(probe_supported=False, context_reversal=False, forced_choice=False, rollout_retained=False, contrastive_pair_objective_high=False)["category"], "A")
        self.assertEqual(classify_failure(probe_supported=True, context_reversal=False, forced_choice=False, rollout_retained=True, contrastive_pair_objective_high=False)["category"], "B")
        self.assertEqual(classify_failure(probe_supported=True, context_reversal=True, forced_choice=True, rollout_retained=False)["category"], "C")
        self.assertEqual(classify_failure(probe_supported=False, context_reversal=False, forced_choice=False, rollout_retained=False)["category"], "D")
        self.assertEqual(classify_failure(probe_supported=True, context_reversal=False, forced_choice=False, rollout_retained=True)["category"], "D")
        self.assertEqual(classify_failure(probe_supported=True, context_reversal=True, forced_choice=True, rollout_retained=True)["category"], "E")
        self.assertIsNone(classify_failure(probe_supported=None, context_reversal=None, forced_choice=None, rollout_retained=None)["category"])
        self.assertNotIn("seal.json", ARTIFACT_NAMES)

    def test_probe_seed_aggregation_retains_support_and_is_deterministic(self):
        records = [
            {"accuracy": 0.4, "balanced_accuracy": 0.5, "macro_f1": 0.3,
             "chance": 0.5, "majority": 0.6, "n": 5, "support": {"0": 3, "1": 2}},
            {"accuracy": 0.6, "balanced_accuracy": 0.7, "macro_f1": 0.5,
             "chance": 0.5, "majority": 0.6, "n": 5, "support": {"0": 3, "1": 2}},
        ]
        first = _aggregate_metric_records(records)
        self.assertEqual(first, _aggregate_metric_records(records))
        self.assertAlmostEqual(first["macro_f1"]["mean"], 0.4)
        self.assertAlmostEqual(first["macro_f1"]["std"], 0.1)
        self.assertEqual(first["support"], {"0": 3, "1": 2})

    def test_real_probe_relative_classification_is_only_contrastive_object_location(self):
        raw = json.loads((ROOT / "runs/narrative-state-localization-v1/probe-layerwise.json").read_text())
        labels = classify_probe_states(raw)
        relative = [(model, family, label) for model, families in labels.items()
                    for family, label in families.items() if label.endswith("_IMPROVED")]
        self.assertEqual(relative, [("contrastive_v1", "object_location", "CONTRASTIVE_IMPROVED")])
        self.assertFalse(any(label == "ROBUSTLY_DECODED" for families in labels.values() for label in families.values()))

    def test_real_probe_compaction_drops_seed_epoch_details_and_is_reproducible(self):
        raw = json.loads((ROOT / "runs/narrative-state-localization-v1/probe-layerwise.json").read_text())
        first = compact_probe(raw)
        second = compact_probe(raw)
        self.assertEqual(first, second)
        encoded = json.dumps(first, sort_keys=True)
        self.assertNotIn('"selected_epoch"', encoded)
        layer = first[1]["models"]["contrastive_v1"]["families"]["object_location"]["layers"][1]
        self.assertIn("distance_buckets", layer["seed_aggregated"]["test"])

    def test_generated_report_answers_all_questions_and_artifact_hashes_reproduce(self):
        experiment = ROOT / "experiments/narrative-state-localization-v1"
        failure = json.loads((experiment / "failure-localization-summary.json").read_text())
        self.assertEqual(failure["final_classification"], {"category": "D", "label": "SYNTHETIC_SHORTCUT"})
        self.assertEqual([item["number"] for item in failure["final_questions"]], [1, 2, 3, 4, 5, 6])
        expected_phrases = (
            "linearly available", "next-token scoring", "97% ranking accuracy",
            "rollout length", "correcting an erroneous rollout", "primary failure localization",
        )
        self.assertTrue(all(
            phrase in item["question"]
            for phrase, item in zip(expected_phrases, failure["final_questions"], strict=True)
        ))
        self.assertIn("97.65%", failure["final_questions"][5]["answer"])
        candidate = json.loads((experiment / "seal-candidate.json").read_text())
        self.assertTrue(candidate["validation_pending"])
        self.assertFalse(candidate["seal_json_created"])
        for name, expected in candidate["report_artifacts"].items():
            self.assertEqual(sha256_file(experiment / name), expected)
        self.assertFalse((experiment / "seal.json").exists())


if __name__ == "__main__":
    unittest.main()
