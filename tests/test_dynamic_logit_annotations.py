import copy
import json
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from src.continuity_curriculum.common import canonical_json, sha256_bytes, sha256_file
from src.dynamic_logit_annotations import (
    CATEGORY_PRECEDENCE,
    DynamicLogitValidationError,
    annotate_pair,
    build_annotations,
    categorize_candidate,
    validate_annotations,
)


def hashed(value):
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def fixture_tokenizer(texts):
    special = ["<|story|>", "<|bos|>", "<|eos|>", "<|mystery|>", "<unk>"]
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=220, special_tokens=special))
    return tokenizer


def fixture_pair(tokenizer, *, split="train"):
    sentences = [
        "The brass key was moved to the cellar.",
        "The brass key was moved to the harbor.",
        "The brass key was moved to the tower.",
    ]
    context = "Rain fell. " + " ".join(sentences) + " Dawn came quietly."
    candidates = {
        "X": {"text": " At that moment, the brass key was at the tower."},
        "Y": {"text": " At that moment, the brass key was at the cellar."},
    }
    for candidate in candidates.values():
        candidate["token_ids"] = tokenizer.encode(candidate["text"]).ids
    events = [
        {"sequence": index, "value": value, "evidence": sentence}
        for index, (value, sentence) in enumerate(zip(("cellar", "harbor", "tower"), sentences, strict=True))
    ]
    world_a = {
        "context": context,
        "context_token_ids": tokenizer.encode(context).ids,
        "correct_candidate": "X",
        "state_events": events,
        "final_value": "tower",
    }
    world_b = copy.deepcopy(world_a)
    world_b["correct_candidate"] = "Y"
    world_b["state_events"] = [{"sequence": 0, "value": "cellar", "evidence": sentences[0]}]
    world_b["final_value"] = "cellar"
    pair = {
        "pair_id": "dynamic-fixture",
        "schema_version": 1,
        "generator_version": "fixture",
        "split": split,
        "worlds": {"A": world_a, "B": world_b},
        "candidates": candidates,
        "abstract_counterfactual_variable": {
            "state_family": "latest_state_update", "entity": "brass key",
            "attribute": "latest_location", "values": ["tower", "cellar"],
        },
        "metadata": {
            "genre": "mystery", "genre_control_token": "<|mystery|>",
            "distance_bucket": "short", "difficulty": 5,
        },
    }
    pair["stable_hash"] = hashed(pair)
    return pair


def fixture_transitions(pair, tokenizer):
    records = []
    for world_name, world in pair["worlds"].items():
        search_from = 0
        for event in world["state_events"]:
            char_start = world["context"].index(event["evidence"], search_from)
            char_end = char_start + len(event["evidence"])
            search_from = char_end
            token_start = len(tokenizer.encode(world["context"][:char_start]).ids)
            token_end = len(tokenizer.encode(world["context"][:char_end]).ids)
            record = {
                "schema_version": 1, "generator_version": "state-transition-v1",
                "source_stable_hash": pair["stable_hash"], "split": pair["split"],
                "pair": pair["pair_id"], "world": world_name, "event": event["sequence"],
                "kind": "transition", "family": "object_location", "entity_slot": "object_0",
                "pre_state": None, "transition": "set_object_location", "post_state": "fixture",
                "event_evidence": event["evidence"],
                "event_span": {
                    "byte_start": len(world["context"][:char_start].encode()),
                    "byte_end": len(world["context"][:char_end].encode()),
                    "token_start": token_start, "token_end": token_end,
                    "token_ids": world["context_token_ids"][token_start:token_end],
                },
                "prediction_token_index": token_end, "metadata": {"transition_count": len(world["state_events"])},
            }
            record["stable_hash"] = hashed(record)
            records.append(record)
    return records


class DynamicLogitAnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = fixture_tokenizer([
            "Rain fell. The brass key was moved to the cellar. The brass key was moved to the harbor. "
            "The brass key was moved to the tower. Dawn came quietly.",
            " At that moment, the brass key was at the tower.",
            " At that moment, the brass key was at the cellar.",
        ])

    def test_taxonomy_precedence_preserves_all_applicable_categories(self):
        self.assertEqual(CATEGORY_PRECEDENCE, ("CURRENT", "PREVIOUS", "INITIAL", "OLDER", "NEVER_VALID"))
        self.assertEqual(categorize_candidate("cellar", ["cellar"]), ("CURRENT", ["CURRENT", "INITIAL"]))
        self.assertEqual(categorize_candidate("cellar", ["cellar", "harbor"]),
                         ("PREVIOUS", ["PREVIOUS", "INITIAL"]))
        self.assertEqual(categorize_candidate("cellar", ["cellar", "harbor", "tower"]),
                         ("INITIAL", ["INITIAL", "OLDER"]))
        self.assertEqual(categorize_candidate("never", ["cellar"]), ("NEVER_VALID", ["NEVER_VALID"]))

    def test_candidate_correctness_depth_and_deterministic_state_switch(self):
        pair = fixture_pair(self.tokenizer)
        transitions = fixture_transitions(pair, self.tokenizer)
        records = annotate_pair(pair, transitions, self.tokenizer)
        self.assertEqual(records, annotate_pair(pair, transitions, self.tokenizer))
        world_a_x = [item for item in records if item["world"] == "A" and item["candidate"] == "X"]
        world_a_y = [item for item in records if item["world"] == "A" and item["candidate"] == "Y"]
        self.assertEqual([item["depth"] for item in world_a_x], [1, 2, 3, 3])
        self.assertEqual([item["category"] for item in world_a_x],
                         ["NEVER_VALID", "NEVER_VALID", "CURRENT", "CURRENT"])
        self.assertEqual([item["category"] for item in world_a_y],
                         ["CURRENT", "PREVIOUS", "INITIAL", "INITIAL"])
        self.assertEqual(world_a_y[1]["all_applicable_categories"], ["PREVIOUS", "INITIAL"])
        self.assertTrue(all(item["token_ids"] == pair["candidates"][item["candidate"]]["token_ids"] for item in records))
        self.assertTrue(all(item["input_token_ids"][item["decision_start"]:item["decision_end"]] == item["token_ids"] for item in records))
        self.assertTrue(all(item["distance"] == 0 for item in records if item["boundary"] == "transition"))
        self.assertTrue(all(item["distance"] > 0 for item in records if item["boundary"] == "final"))
        validate_annotations(pair, transitions, records, self.tokenizer)

    def test_tampering_in_sources_annotations_and_depth_is_rejected(self):
        pair = fixture_pair(self.tokenizer)
        transitions = fixture_transitions(pair, self.tokenizer)
        records = annotate_pair(pair, transitions, self.tokenizer)
        bad = copy.deepcopy(records)
        bad[0]["category"] = "CURRENT"
        with self.assertRaisesRegex(DynamicLogitValidationError, "stable hash"):
            validate_annotations(pair, transitions, bad, self.tokenizer)
        bad_transition = copy.deepcopy(transitions)
        bad_transition[1]["event"] = 9
        bad_transition[1]["stable_hash"] = hashed({key: value for key, value in bad_transition[1].items() if key != "stable_hash"})
        with self.assertRaisesRegex(DynamicLogitValidationError, "depth"):
            annotate_pair(pair, bad_transition, self.tokenizer)
        bad_pair = copy.deepcopy(pair)
        bad_pair["worlds"]["A"]["context"] += " altered"
        with self.assertRaisesRegex(DynamicLogitValidationError, "source stable hash"):
            annotate_pair(bad_pair, transitions, self.tokenizer)

    def test_atomic_build_verifies_hashes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir, transition_dir = root / "source", root / "transitions"
            source_dir.mkdir()
            transition_dir.mkdir()
            tokenizer_path = root / "tokenizer.json"
            self.tokenizer.save(str(tokenizer_path))
            source_files = {}
            transition_splits = {}
            for source_split in ("train", "validation", "test", "generalization_holdout"):
                output_split = "generalization" if source_split == "generalization_holdout" else source_split
                pair = fixture_pair(self.tokenizer, split=source_split)
                transitions = fixture_transitions(pair, self.tokenizer)
                source_path = source_dir / f"{source_split}.jsonl"
                transition_path = transition_dir / f"{output_split}.jsonl"
                source_path.write_text(canonical_json(pair) + "\n", encoding="utf-8")
                transition_payload = b"".join((canonical_json(item) + "\n").encode() for item in transitions)
                transition_path.write_bytes(transition_payload)
                source_files[source_path.name] = {"sha256": sha256_file(source_path)}
                transition_splits[output_split] = {"canonical_sha256": sha256_bytes(transition_payload)}
            source_manifest = {
                "tokenizer_sha256": sha256_file(tokenizer_path), "files": source_files,
            }
            source_manifest_path = source_dir / "manifest.json"
            source_manifest_path.write_text(json.dumps(source_manifest, sort_keys=True), encoding="utf-8")
            transition_manifest = {
                "tokenizer_sha256": sha256_file(tokenizer_path),
                "source_hashes": {
                    **{name: value["sha256"] for name, value in source_files.items()},
                    "manifest.json": sha256_file(source_manifest_path),
                },
                "splits": transition_splits,
            }
            transition_manifest_path = transition_dir / "manifest.json"
            transition_manifest_path.write_text(json.dumps(transition_manifest, sort_keys=True), encoding="utf-8")
            output = root / "dynamic_logit_v1"
            manifest = build_annotations(
                source_dir, transition_dir, tokenizer_path, output,
                expected_source_manifest_sha256=sha256_file(source_manifest_path),
                expected_transition_manifest_sha256=sha256_file(transition_manifest_path),
                expected_tokenizer_sha256=sha256_file(tokenizer_path),
            )
            self.assertEqual(set(manifest["splits"]), {"train", "validation", "test", "generalization"})
            self.assertTrue(all((output / f"{split}.jsonl").is_file() for split in manifest["splits"]))
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                build_annotations(source_dir, transition_dir, tokenizer_path, output)
            (source_dir / "train.jsonl").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(DynamicLogitValidationError, "source hash"):
                build_annotations(source_dir, transition_dir, tokenizer_path, root / "other")


if __name__ == "__main__":
    unittest.main()
