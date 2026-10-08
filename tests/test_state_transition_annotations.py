import copy
import json
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from src.state_transition_annotations import (
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    StateTransitionValidationError,
    annotate_pair,
    build_sidecars,
    validate_annotations,
)


def fixture_tokenizer(texts):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=180, special_tokens=["<unk>"]))
    return tokenizer


def hashed(value):
    from src.continuity_curriculum.common import canonical_json, sha256_bytes

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def fixture_pair(tokenizer, *, family="latest_state_update"):
    filler = " ".join(f"detail{i}" for i in range(180))
    first = "The latest report moved the brass key to the cellar."
    second = "The latest report moved the brass key to the harbor."
    context = f"Rain fell. {first} {filler} {second} {filler}"
    slots = {
        "names": ["Ada", "Bram", "Cyra"],
        "objects": ["brass key", "ledger"],
        "locations": ["cellar", "harbor"],
        "goals": ["warn the guard", "hide the map"],
        "secret": "the bell is false",
        "injury": "burned hand",
        "causes": ["the storm", "the loose wire"],
        "effect": "the beacon failed",
        "lexical_combination": "fixture",
    }
    events = [
        {"sequence": 0, "value": "cellar", "evidence": first},
        {"sequence": 1, "value": "harbor", "evidence": second},
    ]
    world = {"context": context, "context_token_ids": tokenizer.encode(context).ids,
             "state_events": events, "final_value": "harbor"}
    pair = {
        "pair_id": "fixture-pair",
        "schema_version": 1,
        "generator_version": "fixture",
        "split": "train",
        "worlds": {"A": copy.deepcopy(world), "B": copy.deepcopy(world)},
        "candidates": {
            "X": {"text": " At that moment it was elsewhere.", "token_ids": tokenizer.encode(" At that moment it was elsewhere.").ids},
            "Y": {"text": " Nothing changed afterward.", "token_ids": tokenizer.encode(" Nothing changed afterward.").ids},
        },
        "abstract_counterfactual_variable": {
            "state_family": family, "entity": "brass key", "attribute": "latest_location",
            "values": ["cellar", "harbor"],
        },
        "metadata": {"genre": "mystery", "distance_bucket": "long", "difficulty": 4,
                     "lexical_slots": slots},
    }
    pair["stable_hash"] = hashed(pair)
    return pair


class StateTransitionAnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        corpus = [
            "Rain fell. The latest report moved the brass key to the cellar. "
            "The latest report moved the brass key to the harbor. "
            + " ".join(f"detail{i}" for i in range(180)),
            " At that moment it was elsewhere. Nothing changed afterward.",
        ]
        cls.tokenizer = fixture_tokenizer(corpus)

    def test_versions_determinism_and_multi_update_pre_post(self):
        pair = fixture_pair(self.tokenizer)
        first = annotate_pair(pair, self.tokenizer)
        self.assertEqual(first, annotate_pair(pair, self.tokenizer))
        transitions = [record for record in first if record["kind"] == "transition" and record["world"] == "A"]
        self.assertEqual((transitions[0]["pre_state"], transitions[0]["post_state"]), (None, "location_0"))
        self.assertEqual((transitions[1]["pre_state"], transitions[1]["post_state"]), ("location_0", "location_1"))
        self.assertTrue(all(record["schema_version"] == SCHEMA_VERSION for record in first))
        self.assertTrue(all(record["generator_version"] == GENERATOR_VERSION for record in first))
        validate_annotations(pair, first, self.tokenizer)

    def test_carries_stop_before_update_and_use_latest_state(self):
        pair = fixture_pair(self.tokenizer)
        records = annotate_pair(pair, self.tokenizer)
        transitions = [record for record in records if record["kind"] == "transition" and record["world"] == "A"]
        carries = [record for record in records if record["kind"] == "carry" and record["world"] == "A"]
        second_start = transitions[1]["event_span"]["token_start"]
        first_carries = [record for record in carries if record["event"] == 0]
        self.assertTrue(first_carries)
        self.assertTrue(all(record["prediction_token_index"] < second_start for record in first_carries))
        self.assertTrue(all(record["post_state"] == "location_0" for record in first_carries))
        self.assertTrue(all(record["post_state"] == "location_1" for record in carries if record["event"] == 1))

    def test_slots_are_abstract_and_labels_are_absent(self):
        pair = fixture_pair(self.tokenizer)
        records = annotate_pair(pair, self.tokenizer)
        self.assertEqual({record["entity_slot"] for record in records}, {"object_0"})
        labels = {record["post_state"] for record in records}
        self.assertEqual(labels, {"location_0", "location_1"})
        source_text = " ".join(world["context"] for world in pair["worlds"].values())
        self.assertTrue(all(label not in source_text for label in labels))
        validate_annotations(pair, records, self.tokenizer)

    def test_hash_and_carry_tampering_are_rejected(self):
        pair = fixture_pair(self.tokenizer)
        records = annotate_pair(pair, self.tokenizer)
        bad_hash = copy.deepcopy(records)
        bad_hash[0]["post_state"] = "location_9"
        with self.assertRaisesRegex(StateTransitionValidationError, "stable hash"):
            validate_annotations(pair, bad_hash, self.tokenizer)
        bad_carry = copy.deepcopy(records)
        carry = next(record for record in bad_carry if record["kind"] == "carry")
        carry["prediction_token_index"] += 1
        unhashed = {key: value for key, value in carry.items() if key != "stable_hash"}
        carry["stable_hash"] = hashed(unhashed)
        with self.assertRaisesRegex(StateTransitionValidationError, "differ"):
            validate_annotations(pair, bad_carry, self.tokenizer)

    def test_exact_byte_and_token_spans(self):
        pair = fixture_pair(self.tokenizer)
        transition = next(record for record in annotate_pair(pair, self.tokenizer)
                          if record["kind"] == "transition" and record["world"] == "A")
        span = transition["event_span"]
        encoded = pair["worlds"]["A"]["context"].encode("utf-8")
        self.assertEqual(encoded[span["byte_start"]:span["byte_end"]].decode(), transition["event_evidence"])
        self.assertEqual(pair["worlds"]["A"]["context_token_ids"][span["token_start"]:span["token_end"]], span["token_ids"])
        self.assertEqual(transition["prediction_token_index"], span["token_end"])

    def test_atomic_build_locks_inputs_and_refuses_nonidentical_overwrite(self):
        from src.continuity_curriculum.common import canonical_json, sha256_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            tokenizer_path = root / "tokenizer.json"
            self.tokenizer.save(str(tokenizer_path))
            files = {}
            for split in ("train", "validation", "test", "generalization_holdout"):
                pair = fixture_pair(self.tokenizer)
                pair["split"] = split
                pair["stable_hash"] = hashed({key: value for key, value in pair.items() if key != "stable_hash"})
                path = source / f"{split}.jsonl"
                path.write_text(canonical_json(pair) + "\n", encoding="utf-8")
                files[path.name] = {"sha256": sha256_file(path)}
            source_manifest = source / "manifest.json"
            source_manifest.write_text(json.dumps({
                "tokenizer_sha256": sha256_file(tokenizer_path), "files": files,
            }, sort_keys=True), encoding="utf-8")
            output = root / "state_transition_v1"
            manifest = build_sidecars(
                source, tokenizer_path, output,
                expected_source_manifest_sha256=sha256_file(source_manifest),
                expected_tokenizer_sha256=sha256_file(tokenizer_path),
            )
            self.assertEqual(set(manifest["splits"]), {"train", "validation", "test", "generalization"})
            self.assertEqual(manifest, build_sidecars(source, tokenizer_path, output))
            (output / "train.jsonl").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "non-identical"):
                build_sidecars(source, tokenizer_path, output)
            with self.assertRaisesRegex(StateTransitionValidationError, "manifest hash"):
                build_sidecars(
                    source, tokenizer_path, root / "other",
                    expected_source_manifest_sha256="0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
