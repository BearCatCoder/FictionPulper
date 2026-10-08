import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import yaml
from tokenizers import Tokenizer

from src.continuity_curriculum.common import sha256_file
from src.counterfactual_narrative.core import (
    CounterfactualValidationError,
    _validate_config,
    audit_dataset,
    build_dataset,
    classifier_examples,
    generate_pair,
    validate_pair,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "counterfactual-narrative-v1.yaml"
SOURCE_PATH = ROOT / "configs" / "continuity-curriculum-v1.yaml"
TOKENIZER_PATH = ROOT / "data" / "tokenizer" / "tokenizer.json"


class CounterfactualNarrativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.source = yaml.safe_load(SOURCE_PATH.read_text(encoding="utf-8"))
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    def pair(self, ordinal=0, split="train"):
        return generate_pair(self.config, self.source, self.tokenizer,
                             split=split, ordinal=ordinal)

    def test_config_reuses_locked_resources_and_exclusive_templates(self):
        _validate_config(self.config, self.source)
        self.assertEqual(self.config["target"]["main_positive_tokens"], 4_500_000)
        self.assertEqual(sha256_file(TOKENIZER_PATH), self.config["tokenizer"]["expected_sha256"])
        templates = [set(values) for values in self.config["pair_templates"].values()]
        self.assertTrue(all(not left & right for index, left in enumerate(templates)
                            for right in templates[index + 1:]))

    def test_all_state_families_are_deterministic_reversed_and_tokenized(self):
        observed = set()
        for ordinal in range(10):
            pair = self.pair(ordinal)
            self.assertEqual(pair, self.pair(ordinal))
            observed.add(pair["abstract_counterfactual_variable"]["state_family"])
            self.assertNotEqual(pair["worlds"]["A"]["correct_candidate"],
                                pair["worlds"]["B"]["correct_candidate"])
            for candidate in pair["candidates"].values():
                self.assertEqual(candidate["token_ids"], self.tokenizer.encode(candidate["text"]).ids)
            validate_pair(pair, config=self.config, source=self.source, tokenizer=self.tokenizer)
        self.assertEqual(observed, set(self.config["state_families"]))

    def test_candidate_objects_are_shared_and_each_surface_cancels(self):
        pair = self.pair(3)
        examples = classifier_examples([pair], "word")
        labels = {}
        for bag, label, row_id in examples:
            candidate = row_id.rsplit(":", 1)[-1]
            labels.setdefault(candidate, []).append(label)
        self.assertEqual({key: sorted(values) for key, values in labels.items()},
                         {"X": [0, 1], "Y": [0, 1]})
        self.assertEqual(set(pair["candidates"]), {"X", "Y"})
        self.assertNotIn("candidates", pair["worlds"]["A"])
        self.assertNotIn("candidates", pair["worlds"]["B"])

    def test_latest_state_and_state_proof_reject_tampering(self):
        pair = self.pair(9)
        for world in pair["worlds"].values():
            self.assertEqual(world["final_value"], world["state_events"][-1]["value"])
            self.assertNotEqual(world["state_events"][0]["value"], world["state_events"][1]["value"])
        ordinary = self.pair(0)
        self.assertTrue(all(len(world["state_events"]) == 1 for world in ordinary["worlds"].values()))
        bad = copy.deepcopy(pair)
        bad["worlds"]["A"]["state_events"][-1]["value"] = "tampered"
        unhashed = {key: value for key, value in bad.items() if key != "stable_hash"}
        from src.continuity_curriculum.common import canonical_json, sha256_bytes
        bad["stable_hash"] = sha256_bytes(canonical_json(unhashed).encode("utf-8"))
        with self.assertRaisesRegex(CounterfactualValidationError, "latest-state"):
            validate_pair(bad, config=self.config, source=self.source, tokenizer=self.tokenizer)

    def test_decision_spans_fact_removal_and_irrelevant_controls(self):
        pair = self.pair(6)
        removed = pair["controls"]["fact_removed_context"]
        substituted = pair["controls"]["irrelevant_substituted_context"]
        for world in pair["worlds"].values():
            span = world["decision_span"]
            self.assertEqual(world["context"][span["byte_start"]:span["byte_end"]],
                             world["decisive_sentence"])
            self.assertNotIn(world["decisive_sentence"], removed)
            self.assertNotIn(world["decisive_sentence"], substituted)
            self.assertNotIn(pair["controls"]["irrelevant_sentence"], world["context"])
            for candidate_name, candidate in pair["candidates"].items():
                encoding = world["candidate_encodings"][candidate_name]
                self.assertEqual(
                    encoding["token_ids"][encoding["decision_start"]:encoding["decision_end"]],
                    candidate["token_ids"],
                )
        self.assertIn(pair["controls"]["irrelevant_sentence"], substituted)
        self.assertEqual(pair["controls"]["fact_removed_token_ids"],
                          self.tokenizer.encode(removed).ids)

    def test_balance_split_isolation_and_shortcut_gate(self):
        pairs = [self.pair(ordinal, split) for split in ("train", "validation", "test", "generalization_holdout")
                 for ordinal in range(20)]
        report = audit_dataset(pairs, self.config, self.source)
        self.assertTrue(report["passed"], json.dumps(report, indent=2))
        self.assertEqual(report["balance"]["dimensions"]["overall"]["all"]["X_share"], 0.5)
        self.assertTrue(report["split_isolation"]["passed"])
        for classifier in report["shortcut_classifiers"]["classifiers"].values():
            self.assertTrue(classifier["exact_feature_label_cancellation"])
            self.assertTrue(all(result["accuracy"] <= 0.55 for result in classifier["heldouts"].values()))

    def test_stable_hash_detects_mutation(self):
        pair = self.pair(1)
        pair["candidates"]["X"]["text"] += " altered"
        with self.assertRaisesRegex(CounterfactualValidationError, "stable hash"):
            validate_pair(pair, config=self.config, source=self.source, tokenizer=self.tokenizer)

    def test_development_build_is_stop_and_does_not_mutate_inputs(self):
        before = (sha256_file(SOURCE_PATH), sha256_file(TOKENIZER_PATH))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = copy.deepcopy(self.config)
            config["source_narrative_config"] = str(SOURCE_PATH)
            config["tokenizer"]["path"] = str(TOKENIZER_PATH)
            local_config = root / "config.yaml"
            local_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                manifest = build_dataset(local_config, max_pairs=40)
            finally:
                os.chdir(previous)
            output = root / "data" / "counterfactual_narrative_v1"
            self.assertEqual(manifest["status"], "STOP")
            self.assertFalse(manifest["safe_to_train"])
            self.assertTrue(manifest["audits_passed"])
            self.assertEqual(len(list(output.glob("*.jsonl"))), 4)
            self.assertTrue((output / "audit-report.json").exists())
        self.assertEqual(before, (sha256_file(SOURCE_PATH), sha256_file(TOKENIZER_PATH)))


if __name__ == "__main__":
    unittest.main()
