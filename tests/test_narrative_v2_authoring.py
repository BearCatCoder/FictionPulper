import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml
from tokenizers import Tokenizer

from src.benchmark_v2 import load_jsonl
from src.narrative_v2_authoring import (
    AuthoringValidationError,
    RENDERERS,
    _candidate_text,
    _mapping_text,
    audit_scenarios,
    audit_template_signatures,
    audit_training_artifacts,
    authored_stable_hash,
    build_outputs,
    build_scenarios,
    generate_or_verify,
    replay_events,
    validate_schema,
)
from src.narrative_v2_loader import BenchmarkAccessError, load_authored_split
from src.continuity_curriculum.common import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/narrative-v2-authoring.yaml"
BENCHMARK_ROOT = ROOT / "benchmarks/narrative-v2"


class NarrativeV2AuthoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.allocations = load_jsonl(BENCHMARK_ROOT / "scenario-allocation.jsonl")
        cls.schema = json.loads((BENCHMARK_ROOT / "scenario.schema.json").read_text(encoding="utf-8"))
        cls.tokenizer = Tokenizer.from_file(str(ROOT / cls.config["tokenizer"]["path"]))
        cls.records = build_scenarios(cls.allocations, cls.config, cls.tokenizer)

    def audit(self, records):
        return audit_scenarios(records, self.allocations, self.config, self.schema, self.tokenizer)

    def test_schema_allocation_depth_replay_and_balanced_worlds(self):
        report, controls = self.audit(self.records)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["release_ready"])
        self.assertEqual(report["split_counts"], {"development": 28, "generalization": 56, "test": 28})
        self.assertEqual(set(report["reasoning_depth_counts"]), {1, 2, 3, 4, 5})
        self.assertEqual(report["proof_replay"]["intact_worlds_replayed"], 224)
        self.assertEqual(report["proof_replay"]["fact_removed_replay_failures"], 112)
        self.assertEqual(report["world_candidate_labels"], {"X": 112, "Y": 112})
        self.assertEqual(len(report["counts_by_family_depth"]), 28)
        self.assertEqual(len(controls), 112)

    def test_depth_greater_than_one_requires_every_transition(self):
        record = next(record for record in self.records if record["reasoning_depth"] == 5)
        for world_name in ("A", "B"):
            initial = record["intervention"][f"world_{world_name.lower()}_value"]
            answer, event_ids = replay_events(record, initial)
            self.assertEqual(answer, record["worlds"][world_name]["proof"]["answer_value"])
            self.assertEqual(len(event_ids), 5)
            self.assertNotEqual(initial, answer)
        with self.assertRaisesRegex(AuthoringValidationError, "lacks decisive initial fact"):
            replay_events(record, None)
        broken = copy.deepcopy(record)
        mapping = json.loads(broken["events"][2]["object"])
        mapping.pop(next(iter(mapping)))
        broken["events"][2]["object"] = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
        with self.assertRaisesRegex(AuthoringValidationError, "mapping|consume"):
            replay_events(broken, broken["intervention"]["world_a_value"])

    def test_split_renderers_and_normalized_template_signatures_are_isolated(self):
        report, _ = self.audit(self.records)
        self.assertEqual(report["template_signature_isolation"]["status"], "PASS")
        self.assertEqual(set(self.config["renderers"].values()), set(RENDERERS))

        records = copy.deepcopy(self.records)
        target = next(record for record in records if record["scenario_id"] == "nbv2-001")
        split = self.config["splits"]["test"]
        family = self.config["families"][target["state_family"]]
        initial = target["intervention"]["world_a_value"]
        target["worlds"]["A"]["context"] = RENDERERS["development_prose"](
            style=split["paraphrases"][0], actor=target["entities"][0]["name"],
            verb=split["verbs"][1], witness=target["entities"][1]["name"], family=family,
            initial=initial, rule_texts=[],
        )
        target["stable_hash"] = authored_stable_hash(target)
        with self.assertRaisesRegex(AuthoringValidationError, "authored-surface template signature crosses"):
            self.audit(records)

    def test_shared_candidate_and_continuation_template_is_rejected(self):
        records = copy.deepcopy(self.records)
        target = next(record for record in records if record["scenario_id"] == "nbv2-001")
        witness = target["entities"][1]["name"]
        state_noun = self.config["families"][target["state_family"]]["state_noun"]
        answers = {
            world["correct_candidate"]: world["proof"]["answer_value"]
            for world in target["worlds"].values()
        }
        for label in ("X", "Y"):
            target["candidates"][label]["text"] = _candidate_text("development", witness, state_noun, answers[label])
        target["worlds"]["A"]["gold_continuation"] = (
            target["candidates"][target["worlds"]["A"]["correct_candidate"]]["text"]
            + " The chronicle resolution of Norwyn's seal inference chain terminates at flint."
        )
        with self.assertRaisesRegex(AuthoringValidationError, "authored-surface template signature crosses"):
            audit_template_signatures(records, self.config)

    def test_explicit_candidate_controls_and_token_boundaries(self):
        report, controls = self.audit(self.records)
        self.assertEqual(report["controls"]["candidate_only_records"], 112)
        self.assertEqual(report["controls"]["candidate_frame_cancellation"], 112)
        self.assertEqual(report["token_boundary"]["status"], "PASS")
        self.assertTrue(all(control["frame_cancellation"] for control in controls))

        records = copy.deepcopy(self.records)
        target = records[0]
        old_candidate = target["candidates"]["X"]["text"]
        target["candidates"]["X"]["text"] = " Unequal framing " + target["candidates"]["X"]["text"]
        target["candidates"]["X"]["token_ids"] = self.tokenizer.encode(target["candidates"]["X"]["text"]).ids
        for world in target["worlds"].values():
            world["gold_continuation"] = world["gold_continuation"].replace(old_candidate, target["candidates"]["X"]["text"])
            world["counterfactual_continuation"] = world["counterfactual_continuation"].replace(old_candidate, target["candidates"]["X"]["text"])
        target["stable_hash"] = authored_stable_hash(target)
        with self.assertRaisesRegex(AuthoringValidationError, "frame cancellation"):
            self.audit(records)

    def test_all_surface_exact_and_near_duplicates_are_rejected(self):
        exact = copy.deepcopy(self.records)
        exact[4]["worlds"]["A"]["context"] = exact[0]["worlds"]["A"]["context"]
        exact[4]["stable_hash"] = authored_stable_hash(exact[4])
        with self.assertRaisesRegex(AuthoringValidationError, "exact duplicate authored surfaces"):
            self.audit(exact)

        near = copy.deepcopy(self.records)
        near[4]["worlds"]["A"]["context"] = near[0]["worlds"]["A"]["context"] + " Extra."
        near[4]["stable_hash"] = authored_stable_hash(near[4])
        with self.assertRaisesRegex(AuthoringValidationError, "near-duplicate authored surfaces"):
            self.audit(near)

    def test_cross_split_surface_leakage_is_rejected(self):
        records = copy.deepcopy(self.records)
        target = next(record for record in records if record["split"] == "test")
        target["worlds"]["A"]["context"] += " Alden appeared."
        target["stable_hash"] = authored_stable_hash(target)
        with self.assertRaisesRegex(AuthoringValidationError, "cross-split surface leakage"):
            self.audit(records)

    def test_nested_schema_rejects_invalid_token_ids_and_empty_propositions(self):
        invalid_tokens = copy.deepcopy(self.records[0])
        invalid_tokens["candidates"]["X"]["token_ids"] = ["not-an-integer"]
        with self.assertRaisesRegex(AuthoringValidationError, "expected integer"):
            validate_schema(invalid_tokens, self.schema, self.schema)

        empty_propositions = copy.deepcopy(self.records[0])
        empty_propositions["worlds"]["A"]["free_generation"]["required_propositions"] = []
        with self.assertRaisesRegex(AuthoringValidationError, "too few items"):
            validate_schema(empty_propositions, self.schema, self.schema)

    def test_dangling_proof_reference_is_rejected(self):
        records = copy.deepcopy(self.records)
        records[0]["worlds"]["A"]["proof"]["supporting_event_ids"] = ["e404"]
        records[0]["stable_hash"] = authored_stable_hash(records[0])
        with self.assertRaisesRegex(AuthoringValidationError, "proof references"):
            self.audit(records)

    def test_rendered_training_artifact_contamination_and_hash_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "train.jsonl"
            artifact.write_text(json.dumps({"text": self.records[0]["worlds"]["A"]["context"]}) + "\n", encoding="utf-8")
            config = copy.deepcopy(self.config)
            config["training_contamination"]["training_artifacts"] = [{
                "path": str(artifact), "sha256": sha256_file(artifact),
            }]
            with self.assertRaisesRegex(AuthoringValidationError, "rendered training artifact .* contamination"):
                audit_training_artifacts(self.records, config)
            config["training_contamination"]["training_artifacts"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(AuthoringValidationError, "artifact hash mismatch"):
                audit_training_artifacts(self.records, config)

    def test_loader_defaults_to_development_and_gates_hidden_splits(self):
        development = load_authored_split(BENCHMARK_ROOT)
        development_controls = load_authored_split(BENCHMARK_ROOT, controls=True)
        self.assertEqual(len(development), 28)
        self.assertEqual(len(development_controls), 28)
        with tempfile.TemporaryDirectory() as directory:
            absent_root = Path(directory)
            with self.assertRaises(BenchmarkAccessError):
                load_authored_split(absent_root, "test")
            with self.assertRaises(BenchmarkAccessError):
                load_authored_split(absent_root, "test", controls=True)
            with self.assertRaises(BenchmarkAccessError):
                load_authored_split(absent_root, "generalization", checkpoint_selection_complete=True)
            with self.assertRaises(BenchmarkAccessError):
                load_authored_split(
                    absent_root, "generalization", checkpoint_selection_complete=True, controls=True
                )
        test = load_authored_split(BENCHMARK_ROOT, "test", checkpoint_selection_complete=True)
        test_controls = load_authored_split(
            BENCHMARK_ROOT, "test", checkpoint_selection_complete=True, controls=True
        )
        generalization = load_authored_split(
            BENCHMARK_ROOT, "generalization",
            checkpoint_selection_complete=True, test_evaluation_complete=True,
        )
        generalization_controls = load_authored_split(
            BENCHMARK_ROOT, "generalization", checkpoint_selection_complete=True,
            test_evaluation_complete=True, controls=True,
        )
        self.assertEqual((len(test), len(generalization)), (28, 56))
        self.assertEqual((len(test_controls), len(generalization_controls)), (28, 56))

    def test_reproduction_is_byte_deterministic_and_safe(self):
        root, first = build_outputs(CONFIG_PATH)
        _, second = build_outputs(CONFIG_PATH)
        self.assertEqual(first, second)
        for name, expected in first.items():
            self.assertEqual((root / name).read_bytes(), expected)
        manifest = json.loads((root / "authored-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["generated_artifacts"]), set(first) - {"authored-manifest.json"})
        self.assertEqual(manifest["self_hash"], "OMITTED_TO_AVOID_CIRCULARITY")

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            config = copy.deepcopy(self.config)
            config["benchmark_root"] = str(directory / "benchmark")
            config["allocation_path"] = str(BENCHMARK_ROOT / "scenario-allocation.jsonl")
            (directory / "benchmark").mkdir()
            for name in config["locked_inputs"]:
                (directory / "benchmark" / name).write_bytes((BENCHMARK_ROOT / name).read_bytes())
            config_path = directory / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            self.assertEqual(generate_or_verify(config_path), "generated")
            self.assertEqual(generate_or_verify(config_path), "verified")
            output = directory / "benchmark/development.jsonl"
            output.write_bytes(output.read_bytes() + b"\n")
            with self.assertRaisesRegex(AuthoringValidationError, "refusing overwrite"):
                generate_or_verify(config_path)


if __name__ == "__main__":
    unittest.main()
