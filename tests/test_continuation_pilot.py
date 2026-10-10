import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml
from tokenizers import Tokenizer

from src.continuation_pilot import (
    PilotValidationError,
    audit_examples,
    content_sha256,
    deterministic_splits,
    export_review,
    finalize,
    load_prepared_records,
    prepare,
    resolved_reviews_pass_quality,
    resolve_reviews,
    serialize_example,
    validate_example,
)
from src.continuity_curriculum.common import load_jsonl, sha256_file, write_json_atomic, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas/continuation-pilot-v1.schema.json"
TOKENIZER_PATH = ROOT / "data/tokenizer/tokenizer.json"


def words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count)) + "."


def compact_unique_words(marker: str, index: int, count: int) -> str:
    unit = [marker, str(index), "moves", "now"]
    return " ".join((unit * ((count + 3) // 4))[:count]) + "."


def example(index: int) -> dict:
    genres = ("crime_noir", "horror_occult", "science_fiction", "western_adventure", "weird_fantasy")
    states = ("possession", "location", "knowledge", "goal", "causal_update")
    identifier = f"cpv1-example-{index}"
    return {
        "schema_version": 1,
        "example_id": identifier,
        "opening": words(f"opening{index}x", 8) + "\n\n",
        "continuation": words(f"continuation{index}x", 12),
        "genre": genres[(index // 5) % 5],
        "premise": f"Unique premise {index}",
        "protagonist": f"Hero{index}",
        "protagonist_motive": f"Hero{index} wants a unique resolution {index}",
        "character_names": [f"Hero{index}"],
        "atomic_facts": [f"fact {index} alpha", f"fact {index} beta", f"fact {index} gamma"],
        "primary_state_family": states[index % 5],
        "event_transitions": [{
            "transition_id": "transition-1", "state_family": states[index % 5],
            "before": f"state before {index}", "event": f"event changes state {index}",
            "after": f"state after {index}",
            "evidence": {
                "before_quote": f"opening{index}x0", "event_quote": f"continuation{index}x0",
                "after_quote": f"continuation{index}x1",
            },
        }],
        "stale_state_distractor_present": index % 2 == 0,
        "action_summary": f"Hero{index} acts",
        "consequence_summary": f"A consequence {index} follows",
        "duplicate_cluster_id": f"duplicate-{index}",
        "authoring_inputs": [f"independent-brief-{index}"],
        "rights": {
            "basis": "original_commission", "license_or_status": "exclusive permission",
            "evidence_reference": f"private-rights-{index}", "evidence_sha256": f"{index + 1:064x}",
            "rights_holder_or_author": f"Author {index}",
        },
        "provenance": {
            "source_id": f"source-{index}", "original_story_id": f"story-{index}",
            "source_reference": f"private-source-{index}", "source_sha256": f"{index + 101:064x}",
            "authoring_method": "independently_authored", "baseline_training_exposure": "none",
        },
    }


def production_example(index: int) -> dict:
    row = example(index)
    genres = ("crime_noir", "horror_occult", "science_fiction", "western_adventure", "weird_fantasy")
    states = ("possession", "location", "knowledge", "goal", "causal_update")
    row["genre"] = genres[index // 40]
    row["primary_state_family"] = states[(index % 40) // 8]
    row["event_transitions"][0]["state_family"] = row["primary_state_family"]
    row["opening"] = compact_unique_words("openmark", index, 60) + "\n\n"
    row["continuation"] = compact_unique_words("continuemark", index, 152)
    row["event_transitions"][0]["evidence"] = {
        "before_quote": f"openmark {index} moves now",
        "event_quote": f"continuemark {index} moves now",
        "after_quote": f"continuemark {index} moves now",
    }
    row["stale_state_distractor_present"] = index < 100
    return row


def completed(packet: dict, reviewer: str, packet_hash: str, role: str = "original") -> dict:
    row = copy.deepcopy(packet)
    row.update({
        "packet_sha256": packet_hash,
        "reviewer_id": reviewer,
        "reviewer_role": role,
        "independent_review_confirmed": True,
        "genuine_human_attestation": True,
        "fact_labels": {fact: "retained" for fact in row["atomic_facts"]},
        "premise_preserved": "yes", "motive_preserved": "yes",
        "opening_facts_consistent": "yes", "no_narrative_loop": "yes",
        "plot_advanced": "yes", "genre_voice": "yes", "meaningful_action": "yes",
        "meaningful_consequence": "yes", "natural_readable_continuation": "yes",
        "evidence": {
            **{fact: row["opening"] for fact in row["atomic_facts"]},
            **{field: row["continuation"] for field in (
                "premise_preserved", "motive_preserved", "opening_facts_consistent",
                "no_narrative_loop", "plot_advanced", "genre_voice", "meaningful_action",
                "meaningful_consequence", "natural_readable_continuation",
            )},
        },
        "reviewer_note": "Read independently and checked every required dimension.",
    })
    return row


class ContinuationPilotTests(unittest.TestCase):
    def setUp(self):
        self.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    def config(self, root: Path, records: list[dict] | None = None) -> tuple[Path, dict]:
        data = root / "private"
        output = root / "runs"
        examples = data / "examples.jsonl"
        if records is not None:
            write_jsonl(examples, records)
        production = yaml.safe_load((ROOT / "configs/continuation-pilot-v1.yaml").read_text(encoding="utf-8"))
        production.update({
            "examples_path": str(examples), "reviews_path": str(data / "reviews.jsonl"),
            "custodian_verification_path": str(data / "custodian.json"), "output_dir": str(output),
            "schema_path": str(SCHEMA_PATH), "expected_examples": 25,
            "expected_split_counts": {"train": 15, "validation": 5, "test": 5},
            "expected_genre_counts": {genre: 5 for genre in production["expected_genre_counts"]},
            "expected_state_family_counts": {family: 5 for family in production["expected_state_family_counts"]},
            "expected_stale_state_distractor_counts": {"present": 13, "absent": 12},
            "expected_genre_state_cell_count": 1,
            "expected_split_genre_counts": {
                "train": {genre: 3 for genre in production["expected_genre_counts"]},
                "validation": {genre: 1 for genre in production["expected_genre_counts"]},
                "test": {genre: 1 for genre in production["expected_genre_counts"]},
            },
            "word_counts": {"opening_minimum": 5, "opening_maximum": 20, "continuation_minimum": 10, "continuation_maximum": 30},
        })
        production["tokenizer"]["path"] = str(TOKENIZER_PATH)
        production["contamination"]["benchmark_v2"] = [
            {"path": str(ROOT / item["path"]), "sha256": item["sha256"]}
            for item in production["contamination"]["benchmark_v2"]
        ]
        scene = production["contamination"]["scene_scorecard"]
        scene["path"] = str(ROOT / scene["path"])
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(production, sort_keys=False), encoding="utf-8")
        return path, production

    def production_config(self, root: Path, records: list[dict] | None = None) -> tuple[Path, dict]:
        config = yaml.safe_load((ROOT / "configs/continuation-pilot-v1.yaml").read_text(encoding="utf-8"))
        data = root / "private"
        config.update({
            "examples_path": str(data / "examples.jsonl"),
            "reviews_path": str(data / "reviews.jsonl"),
            "custodian_verification_path": str(data / "custodian.json"),
            "output_dir": str(root / "runs"),
        })
        if records is not None:
            write_jsonl(Path(config["examples_path"]), records)
        path = root / "production-config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path, config

    def test_production_config_freezes_contract(self):
        config = yaml.safe_load((ROOT / "configs/continuation-pilot-v1.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["expected_examples"], 200)
        self.assertEqual(config["expected_split_counts"], {"train": 160, "validation": 20, "test": 20})
        self.assertEqual(set(config["expected_genre_counts"].values()), {40})
        self.assertEqual(set(config["expected_state_family_counts"].values()), {40})
        self.assertEqual(config["expected_stale_state_distractor_counts"], {"present": 100, "absent": 100})
        self.assertEqual(config["expected_genre_state_cell_count"], 8)
        self.assertTrue(all(set(counts.values()) == ({32} if split == "train" else {4}) for split, counts in config["expected_split_genre_counts"].items()))
        self.assertEqual(config["serialization"]["maximum_tokens"], 1024)
        self.assertEqual(config["tokenizer"]["sha256"], sha256_file(TOKENIZER_PATH))

    def test_strict_schema_lengths_facts_rights_and_post_prompt_use(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config = self.config(Path(directory))
            row = example(0)
            validate_example(row, self.schema, config)
            bad_boundary = {**row, "opening": row["opening"].removesuffix("\n\n")}
            with self.assertRaisesRegex(PilotValidationError, "pattern mismatch|two LF"):
                validate_example(bad_boundary, self.schema, config)
            leading = {**row, "continuation": " " + row["continuation"]}
            with self.assertRaisesRegex(PilotValidationError, "pattern mismatch|must not start"):
                validate_example(leading, self.schema, config)
            unknown = {**row, "unexpected": True}
            with self.assertRaisesRegex(PilotValidationError, "unexpected properties"):
                validate_example(unknown, self.schema, config)
            too_few = {**row, "atomic_facts": row["atomic_facts"][:2]}
            with self.assertRaisesRegex(PilotValidationError, "too few|3-5"):
                validate_example(too_few, self.schema, config)
            post = {**row, "authoring_inputs": ["scene-post-01"]}
            with self.assertRaisesRegex(PilotValidationError, "post-selection"):
                validate_example(post, self.schema, config)
            rights = copy.deepcopy(row)
            rights["rights"]["evidence_sha256"] = "pending"
            with self.assertRaises(PilotValidationError):
                validate_example(rights, self.schema, config)

            transition = copy.deepcopy(row)
            transition["event_transitions"][0]["unexpected"] = True
            with self.assertRaisesRegex(PilotValidationError, "unexpected properties"):
                validate_example(transition, self.schema, config)
            transition = copy.deepcopy(row)
            transition["event_transitions"][0]["state_family"] = "temporal_ordering"
            with self.assertRaises(PilotValidationError):
                validate_example(transition, self.schema, config)
            transition = copy.deepcopy(row)
            transition["event_transitions"][0]["evidence"]["after_quote"] = "not in prose"
            with self.assertRaisesRegex(PilotValidationError, "after evidence"):
                validate_example(transition, self.schema, config)

    def test_grouped_split_is_deterministic_exact_and_leak_free(self):
        records = [example(index) for index in range(25)]
        records[1]["provenance"]["source_id"] = records[0]["provenance"]["source_id"]
        records[1]["duplicate_cluster_id"] = records[0]["duplicate_cluster_id"]
        split_genres = {split: {genre: count for genre in ("crime_noir", "horror_occult", "science_fiction", "western_adventure", "weird_fantasy")} for split, count in (("train", 3), ("validation", 1), ("test", 1))}
        first = deterministic_splits(records, {"train": 15, "validation": 5, "test": 5}, 8008, split_genres)
        second = deterministic_splits(list(reversed(records)), {"train": 15, "validation": 5, "test": 5}, 8008, split_genres)
        self.assertEqual(first, second)
        self.assertEqual(first["cpv1-example-0"], first["cpv1-example-1"])
        self.assertEqual(dict(sorted(__import__("collections").Counter(first.values()).items())), {"test": 5, "train": 15, "validation": 5})
        for genre in split_genres["train"]:
            genre_rows = [row for row in records if row["genre"] == genre]
            self.assertEqual(__import__("collections").Counter(first[row["example_id"]] for row in genre_rows), {"train": 3, "validation": 1, "test": 1})
        mixed = copy.deepcopy(records)
        mixed[5]["provenance"]["source_id"] = mixed[0]["provenance"]["source_id"]
        with self.assertRaisesRegex(PilotValidationError, "mixed-genre"):
            deterministic_splits(mixed, {"train": 15, "validation": 5, "test": 5}, 8008, split_genres)

    def test_serialization_masks_opening_and_padding_but_targets_continuation_and_eos(self):
        row = example(0)
        serialized = serialize_example(row, self.tokenizer, boundary="", maximum_tokens=1024, pad_to=1024)
        boundary = serialized["opening_continuation_boundary_token_index"]
        self.assertEqual(serialized["token_ids"][:2], [4, 1])
        self.assertEqual(serialized["token_ids"][-1], 2)
        self.assertTrue(all(value == -100 for value in serialized["labels"][:boundary - 1]))
        self.assertEqual(serialized["labels"][boundary - 1], serialized["token_ids"][boundary])
        self.assertEqual(serialized["labels"][len(serialized["token_ids"]) - 2], 2)
        self.assertTrue(all(value == -100 for value in serialized["labels"][len(serialized["token_ids"]) - 1:]))
        self.assertFalse(serialized["truncated"])
        serialized_prose = self.tokenizer.decode(serialized["token_ids"][2:-1])
        self.assertNotIn(row["event_transitions"][0]["before"], serialized_prose)
        self.assertNotIn("stale_state_distractor_present", serialized_prose)
        with self.assertRaisesRegex(PilotValidationError, "truncation forbidden"):
            serialize_example(row, self.tokenizer, boundary="", maximum_tokens=5)
        with self.assertRaisesRegex(PilotValidationError, "must not insert"):
            serialize_example(row, self.tokenizer, boundary="\n\n", maximum_tokens=1024)

    def test_audit_checks_counts_balances_contamination_and_serialization(self):
        records = [example(index) for index in range(25)]
        with tempfile.TemporaryDirectory() as directory:
            _, config = self.config(Path(directory))
            report, prepared = audit_examples(records, config, self.schema, self.tokenizer)
            self.assertEqual(report["status"], "PASS_AUTOMATED_AUDIT")
            self.assertEqual(len(prepared), 25)
            token_gate = report["gates"]["serialization_and_masking"]
            self.assertEqual(
                token_gate["tokenizer_v1_tokens_total"],
                sum(row["serialization"]["token_count"] for row in prepared),
            )
            self.assertEqual(set(token_gate["tokenizer_v1_tokens_by_split"]), {"train", "validation", "test"})
            self.assertTrue(report["gates"]["stale_state_distractor_balance"]["passed"])
            self.assertTrue(report["gates"]["genre_primary_state_cross_balance"]["passed"])
            stale_unbalanced = copy.deepcopy(records)
            stale_unbalanced[1]["stale_state_distractor_present"] = True
            stale_report, _ = audit_examples(stale_unbalanced, config, self.schema, self.tokenizer)
            self.assertIn("stale_state_distractor_balance", stale_report["failed_gates"])
            cross_unbalanced = copy.deepcopy(records)
            cross_unbalanced[0]["primary_state_family"] = "location"
            cross_unbalanced[0]["event_transitions"][0]["state_family"] = "location"
            cross_report, _ = audit_examples(cross_unbalanced, config, self.schema, self.tokenizer)
            self.assertIn("genre_primary_state_cross_balance", cross_report["failed_gates"])
            contaminated = copy.deepcopy(records)
            contaminated[1]["continuation"] = contaminated[0]["continuation"]
            contaminated[1]["event_transitions"][0]["evidence"]["event_quote"] = "continuation0x0"
            contaminated[1]["event_transitions"][0]["evidence"]["after_quote"] = "continuation0x1"
            report, _ = audit_examples(contaminated, config, self.schema, self.tokenizer)
            self.assertIn("contamination", report["failed_gates"])
            self.assertTrue(report["gates"]["contamination"]["normalized_exact_pilot_collisions"])
            scorecard = copy.deepcopy(records)
            scorecard[0]["character_names"] = ["Mallory"]
            report, _ = audit_examples(scorecard, config, self.schema, self.tokenizer)
            self.assertTrue(report["gates"]["contamination"]["scorecard_name_collisions"])
            prompts = load_jsonl(ROOT / "benchmarks/scene-scorecard-v1/prompts.jsonl")
            metadata_copy = copy.deepcopy(records)
            metadata_copy[0]["premise"] = prompts[0]["premise"]
            metadata_copy[0]["atomic_facts"][0] = prompts[0]["atomic_facts"][0]["text"]
            report, _ = audit_examples(metadata_copy, config, self.schema, self.tokenizer)
            collisions = report["gates"]["contamination"]["benchmark_and_scorecard_collisions"]
            self.assertTrue(any(item["kind"] in {"premise", "atomic_fact"} for item in collisions))
            self.assertIn("cannot establish", report["gates"]["contamination"]["semantic_limit"])

    def test_missing_external_inputs_report_stop_without_fabrication(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path, config = self.config(Path(directory))
            report = prepare(config_path, enforce_production=False)
            self.assertEqual(report["status"], "STOP")
            self.assertFalse(report["approved"])
            self.assertIn("exactly 25 canonical examples", report["pending_external_inputs"][0])
            self.assertFalse((Path(config["output_dir"]) / "approval.json").exists())

    def test_prepare_and_review_export_are_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path, config = self.config(Path(directory), [example(index) for index in range(25)])
            first = prepare(config_path, enforce_production=False)
            self.assertEqual(first["status"], "PASS_AUTOMATED_AUDIT")
            prepared_path = Path(config["output_dir"]) / "prepared.jsonl"
            first_bytes = prepared_path.read_bytes()
            second = prepare(config_path, enforce_production=False)
            self.assertEqual(first, second)
            self.assertEqual(first_bytes, prepared_path.read_bytes())
            export_review(config_path, enforce_production=False)
            packet_path = Path(config["output_dir"]) / "review-packet.jsonl"
            packet_bytes = packet_path.read_bytes()
            export_review(config_path, enforce_production=False)
            self.assertEqual(packet_bytes, packet_path.read_bytes())
            loaded = load_prepared_records(prepared_path)
            self.assertEqual({row["split"] for row in loaded}, {"train", "validation"})
            with self.assertRaisesRegex(PilotValidationError, "post-selection"):
                load_prepared_records(prepared_path, splits=("test",))
            test_rows = load_prepared_records(prepared_path, splits=("test",), post_selection_complete=True)
            self.assertEqual({row["split"] for row in test_rows}, {"test"})

    def test_review_hash_distinct_people_and_adjudication_are_required(self):
        packet = {
            "review_id": "r1", "example_id": "cpv1-x", "genre": "crime_noir",
            "opening": "opening evidence", "continuation": "continuation evidence", "premise": "p",
            "protagonist": "h", "protagonist_motive": "m", "atomic_facts": ["f1", "f2", "f3"],
            "content_sha256": "a" * 64, "reviewer_id": "", "reviewer_role": "original",
            "independent_review_confirmed": None, "genuine_human_attestation": None, "fact_labels": {},
            "premise_preserved": "", "motive_preserved": "", "opening_facts_consistent": "",
            "no_narrative_loop": "", "plot_advanced": "", "genre_voice": "", "meaningful_action": "",
            "meaningful_consequence": "", "natural_readable_continuation": "", "evidence": {}, "reviewer_note": "",
        }
        digest = "b" * 64
        rows = [completed(packet, "person-a", digest), completed(packet, "person-b", digest)]
        self.assertEqual(len(resolve_reviews([packet], rows, digest)), 1)
        with self.assertRaisesRegex(PilotValidationError, "distinct"):
            resolve_reviews([packet], [rows[0], completed(packet, "person-a", digest)], digest)
        rows[1]["plot_advanced"] = "no"
        rows[0]["genre_voice"] = "no"
        rows[1]["genre_voice"] = "no"
        with self.assertRaisesRegex(PilotValidationError, "adjudication"):
            resolve_reviews([packet], rows, digest)
        rows.append(completed(packet, "person-c", digest, "adjudicator"))
        merged = resolve_reviews([packet], rows, digest)[0]
        self.assertEqual(merged["plot_advanced"], "yes")
        self.assertEqual(merged["genre_voice"], "no")
        self.assertEqual(merged["resolution_provenance"]["fields"]["plot_advanced"], "adjudicated")
        self.assertEqual(merged["resolution_provenance"]["fields"]["genre_voice"], "original_consensus")
        self.assertFalse(all(merged[field] == "yes" for field in (
            "premise_preserved", "motive_preserved", "opening_facts_consistent",
            "no_narrative_loop", "plot_advanced", "genre_voice", "meaningful_action",
            "meaningful_consequence", "natural_readable_continuation",
        )))
        self.assertFalse(resolved_reviews_pass_quality([merged]))
        rows[0]["packet_sha256"] = "c" * 64
        with self.assertRaisesRegex(PilotValidationError, "invalidated"):
            resolve_reviews([packet], rows, digest)

    def test_scaled_helper_config_cannot_approve_or_bypass_production_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, config = self.config(root, [example(index) for index in range(25)])
            export_review(config_path, enforce_production=False)
            output = Path(config["output_dir"])
            write_json_atomic(output / "approval.json", {"approved": True})
            stopped = finalize(config_path, enforce_production=False)
            self.assertEqual(stopped["status"], "STOP")
            self.assertIn("never produce approval", stopped["error"])
            self.assertFalse((output / "approval.json").exists())
            write_json_atomic(output / "approval.json", {"approved": True})
            bypass = prepare(config_path)
            self.assertEqual(bypass["status"], "STOP")
            self.assertIn("production contract changed", bypass["error"])
            self.assertFalse((output / "approval.json").exists())
            write_json_atomic(output / "approval.json", {"approved": True})
            export_bypass = export_review(config_path)
            self.assertEqual(export_bypass["status"], "STOP")
            self.assertFalse((output / "approval.json").exists())

    def test_production_finalize_approves_only_complete_inputs_and_invalidates_stale_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, config = self.production_config(root, [production_example(index) for index in range(200)])
            exported = export_review(config_path)
            self.assertEqual(exported["status"], "REVIEW_PENDING")
            output = Path(config["output_dir"])
            packet_path = output / "review-packet.jsonl"
            packet_hash = sha256_file(packet_path)
            reviews = [
                completed(packet, reviewer, packet_hash)
                for packet in load_jsonl(packet_path) for reviewer in ("reviewer-a", "reviewer-b")
            ]
            write_jsonl(Path(config["reviews_path"]), reviews)
            prepared_hash = sha256_file(output / "prepared.jsonl")
            custodian = {
                "version": 1, "prepared_sha256": prepared_hash, "review_packet_sha256": packet_hash,
                "custodian_id": "custodian-a", "verified_at": "2026-10-10",
                "verification_evidence_reference": "private verification ledger",
                "rights_evidence_examined": True, "provenance_evidence_examined": True,
                "all_examples_rights_cleared": True, "reviewer_identities_verified": True,
                "reviewers_genuine_humans": True, "independent_review_process_verified": True,
                "adjudicator_identities_verified": True,
                "distinctive_scorecard_fact_combinations_checked": True,
                "no_distinctive_scorecard_fact_combination_collision": True,
                "scene_post_prompts_excluded_from_authoring_editing_and_candidate_selection": True,
                "verified_reviewer_ids": ["reviewer-a", "reviewer-b"], "verified_adjudicator_ids": [],
            }
            write_json_atomic(Path(config["custodian_verification_path"]), custodian)
            approved = finalize(config_path)
            self.assertEqual(approved["status"], "APPROVED")
            self.assertGreater(approved["tokenizer_v1_tokens_total"], 0)
            self.assertEqual(set(approved["tokenizer_v1_tokens_by_split"]), {"train", "validation", "test"})
            self.assertTrue((output / "approval.json").is_file())

            Path(config["reviews_path"]).write_text("{malformed\n", encoding="utf-8")
            malformed = finalize(config_path)
            self.assertEqual(malformed["status"], "STOP")
            self.assertFalse((output / "approval.json").exists())
            write_jsonl(Path(config["reviews_path"]), reviews)
            self.assertEqual(finalize(config_path)["status"], "APPROVED")

            original_config = config_path.read_bytes()
            changed_config = yaml.safe_load(original_config)
            changed_config["word_counts"]["opening_maximum"] += 1
            config_path.write_text(yaml.safe_dump(changed_config, sort_keys=False), encoding="utf-8")
            contract_invalidated = finalize(config_path)
            self.assertIn("contract changed", contract_invalidated["error"])
            self.assertFalse((output / "approval.json").exists())
            config_path.write_bytes(original_config)
            self.assertEqual(finalize(config_path)["status"], "APPROVED")

            Path(config["examples_path"]).unlink()
            stopped_prepare = prepare(config_path)
            self.assertEqual(stopped_prepare["status"], "STOP")
            self.assertFalse((output / "approval.json").exists())


if __name__ == "__main__":
    unittest.main()
