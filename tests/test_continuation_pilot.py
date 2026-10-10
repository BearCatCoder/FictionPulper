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
    select_owner_review_rows,
    serialize_example,
    validate_example,
)
from src.continuity_curriculum.common import canonical_json, load_jsonl, sha256_bytes, sha256_file, write_json_atomic, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas/continuation-pilot-v1.schema.json"
TOKENIZER_PATH = ROOT / "data/tokenizer/tokenizer.json"


def words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count)) + "."


def compact_unique_words(marker: str, index: int, count: int) -> str:
    unit = [marker, str(index), "moves", "now"]
    return " ".join((unit * ((count + 3) // 4))[:count]) + "."


def artifact_bytes(kind: str, index: int) -> bytes:
    return f"retained {kind} bytes for fixture {index}\n".encode()


def assessment_output_bytes(row: dict) -> bytes:
    assessment = row["preliminary_assessment"]
    output = {
        "version": 1,
        "example_id": row["example_id"],
        "assessed_content_sha256": sha256_bytes((row["opening"] + row["continuation"]).encode()),
        "method": assessment["method"],
        "assessor_identity": assessment["assessor_identity"],
        "prompt_sha256": assessment["prompt"]["sha256"],
        "generation_configuration": assessment["generation_configuration"],
        "assessment_date": assessment["assessment_date"],
        "risk_flags": assessment["risk_flags"],
        "summary": assessment["summary"],
    }
    if assessment["method"] == "ai_assisted":
        output["model_identity"] = assessment["model_identity"]
    return (canonical_json(output) + "\n").encode()


def example(index: int) -> dict:
    genres = ("crime_noir", "horror_occult", "science_fiction", "western_adventure", "weird_fantasy")
    states = ("possession", "location", "knowledge", "goal", "causal_update")
    identifier = f"cpv1-example-{index}"
    opening = words(f"opening{index}x", 8) + "\n\n"
    continuation = words(f"continuation{index}x", 12)
    prompt_text = f"Write and assess continuation example {index}."
    prompt = {"text": prompt_text, "sha256": sha256_bytes(prompt_text.encode())}
    source_class = (
        "public_domain_adaptation", "ai_assisted_original",
        "ai_assisted_public_domain_adaptation",
    )[index % 3]
    rights_basis = "original_author_owned" if source_class == "ai_assisted_original" else "public_domain"
    provenance = {
        "source_class": source_class,
        "source_id": f"source-{index}",
        "original_story_id": f"story-{index}",
        "pretraining_exposure": {
            "classification": ("not_identified", "identified", "uncertain")[index % 3],
            "audit_method": "Compared source identity against the sealed pretraining manifest.",
            "disclosure": f"Exposure classification recorded for source {index}.",
        },
    }
    if source_class != "ai_assisted_original":
        provenance["public_domain_source"] = {
            "author": f"Public Domain Author {index}", "title": f"Source Story {index}",
            "source_url": f"https://example.test/source-{index}", "source_edition": "Verified scan edition",
            "source_path": f"source-{index}.txt",
            "source_sha256": sha256_bytes(artifact_bytes("source", index)),
            "rights_basis": "Public domain in the United States",
            "rights_evidence_reference": f"private-rights-{index}",
            "rights_evidence_sha256": sha256_bytes(artifact_bytes("rights", index)),
            "rights_check_date": "2026-10-10",
            "acquisition_method": "independent_download_from_documented_source_url",
            "acquisition_evidence_reference": f"download-ledger-{index}",
            "acquisition_evidence_path": f"rights-{index}.txt",
            "acquisition_evidence_sha256": sha256_bytes(artifact_bytes("rights", index)),
            "independently_acquired_from_documented_source": True,
            "copied_from_unsealed_corpus_v3_candidate": False,
            "corpus_v3_overlap": ("not_identified", "identified", "uncertain")[index % 3],
            "corpus_v3_overlap_disclosure": f"Corpus-v3 overlap classification recorded for source {index}.",
        }
    if source_class != "public_domain_adaptation":
        provenance["ai_assistance"] = {
            "model_identity": "test-model/exact-revision", "prompt": prompt,
            "generation_configuration": '{"temperature":0.0,"seed":1}',
            "generation_date": "2026-10-10", "raw_output_path": f"ai-{index}.txt",
            "raw_output_sha256": sha256_bytes(artifact_bytes("ai-output", index)),
            "editing_history": ["Human owner corrected and documented the draft."],
            "final_content_sha256": sha256_bytes((opening + continuation).encode()),
        }
    record = {
        "schema_version": 1,
        "example_id": identifier,
        "opening": opening,
        "continuation": continuation,
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
            "basis": rights_basis, "license_or_status": "verified test status",
            "evidence_reference": f"private-rights-{index}", "evidence_path": f"rights-{index}.txt",
            "evidence_sha256": sha256_bytes(artifact_bytes("rights", index)),
            "rights_holder_or_author": f"Author {index}", "checked_at": "2026-10-10",
        },
        "provenance": provenance,
        "preliminary_assessment": {
            "method": "automated", "assessor_identity": "fixture-auditor-v1", "prompt": prompt,
            "generation_configuration": '{"deterministic":true}', "assessment_date": "2026-10-10",
            "raw_output_path": f"assessment-{index}.txt",
            "raw_output_sha256": "0" * 64, "risk_flags": [],
            "summary": "No preliminary risk was identified by the fixture diagnostic.",
        },
    }
    record["preliminary_assessment"]["raw_output_sha256"] = sha256_bytes(assessment_output_bytes(record))
    return record


def production_example(index: int) -> dict:
    row = example(index)
    genres = ("crime_noir", "horror_occult", "science_fiction", "western_adventure", "weird_fantasy")
    states = ("possession", "location", "knowledge", "goal", "causal_update")
    row["genre"] = genres[index // 40]
    row["primary_state_family"] = states[(index % 40) // 8]
    row["event_transitions"][0]["state_family"] = row["primary_state_family"]
    row["opening"] = compact_unique_words("openmark", index, 60) + "\n\n"
    row["continuation"] = compact_unique_words("continuemark", index, 152)
    if "ai_assistance" in row["provenance"]:
        row["provenance"]["ai_assistance"]["final_content_sha256"] = sha256_bytes(
            (row["opening"] + row["continuation"]).encode()
        )
    row["event_transitions"][0]["evidence"] = {
        "before_quote": f"openmark {index} moves now",
        "event_quote": f"continuemark {index} moves now",
        "after_quote": f"continuemark {index} moves now",
    }
    row["stale_state_distractor_present"] = index < 100
    row["preliminary_assessment"]["raw_output_sha256"] = sha256_bytes(assessment_output_bytes(row))
    return row


def completed(packet: dict, reviewer: str, packet_hash: str, role: str = "original") -> dict:
    row = copy.deepcopy(packet)
    row.update({
        "packet_sha256": packet_hash,
        "reviewer_id": reviewer,
        "reviewer_role": role,
        "independent_review_confirmed": role != "owner",
        "genuine_human_attestation": True,
        "fact_labels": {fact: "retained" for fact in row["atomic_facts"]},
        "rights_and_provenance_acceptable": "yes", "premise_preserved": "yes", "motive_preserved": "yes",
        "opening_facts_consistent": "yes", "contradiction_free": "yes", "no_narrative_loop": "yes",
        "plot_advanced": "yes", "genre_voice": "yes", "meaningful_action": "yes",
        "meaningful_consequence": "yes", "natural_readable_continuation": "yes",
        "exploratory_sft_suitable": "yes",
        "evidence": {
            **{fact: row["opening"] for fact in row["atomic_facts"]},
            **{field: row["continuation"] for field in (
                "premise_preserved", "motive_preserved", "opening_facts_consistent",
                "contradiction_free", "no_narrative_loop", "plot_advanced", "genre_voice",
                "meaningful_action", "meaningful_consequence", "natural_readable_continuation",
                "exploratory_sft_suitable",
            )},
            "rights_and_provenance_acceptable": "<PROVENANCE_RECORD>",
        },
        "reviewer_note": "Read independently and checked every required dimension.",
    })
    return row


class ContinuationPilotTests(unittest.TestCase):
    def setUp(self):
        self.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    def retained_roots(self, root: Path, records: list[dict] | None = None) -> dict[str, str]:
        roots = {
            "sources": root / "sources", "rights_evidence": root / "rights_evidence",
            "prompts": root / "prompts", "raw_outputs": root / "raw_outputs",
        }
        for path in roots.values():
            path.mkdir(parents=True, exist_ok=True)
        retained_records = records if records is not None else [example(index) for index in range(200)]
        for row in retained_records:
            index = int(row["example_id"].rsplit("-", 1)[1])
            (roots["sources"] / f"source-{index}.txt").write_bytes(artifact_bytes("source", index))
            (roots["rights_evidence"] / f"rights-{index}.txt").write_bytes(artifact_bytes("rights", index))
            (roots["raw_outputs"] / f"ai-{index}.txt").write_bytes(artifact_bytes("ai-output", index))
            (roots["raw_outputs"] / f"assessment-{index}.txt").write_bytes(assessment_output_bytes(row))
        return {name: str(path) for name, path in roots.items()}

    def bind_assessment_output(self, config: dict, row: dict, suffix: str) -> None:
        path_name = f"{row['example_id']}-{suffix}.json"
        data = assessment_output_bytes(row)
        (Path(config["retained_artifact_roots"]["raw_outputs"]) / path_name).write_bytes(data)
        row["preliminary_assessment"]["raw_output_path"] = path_name
        row["preliminary_assessment"]["raw_output_sha256"] = sha256_bytes(data)

    def config(self, root: Path, records: list[dict] | None = None) -> tuple[Path, dict]:
        data = root / "private"
        output = root / "runs"
        examples = data / "examples.jsonl"
        if records is not None:
            write_jsonl(examples, records)
        production = yaml.safe_load((ROOT / "configs/continuation-pilot-v1.yaml").read_text(encoding="utf-8"))
        production.update({
            "examples_path": str(examples), "reviews_path": str(data / "reviews.jsonl"),
            "owner_verification_path": str(data / "owner.json"),
            "strict_verification_path": str(data / "strict.json"), "output_dir": str(output),
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
            "retained_artifact_roots": self.retained_roots(data, records),
        })
        production["owner_review"]["target_sample_size"] = 10
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
            "owner_verification_path": str(data / "owner.json"),
            "strict_verification_path": str(data / "strict.json"),
            "output_dir": str(root / "runs"),
            "retained_artifact_roots": self.retained_roots(data, records),
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
            corpus_v3_input = {**row, "authoring_inputs": ["data/corpus_v3/stages/candidate-17.json"]}
            with self.assertRaisesRegex(PilotValidationError, "Corpus-v3 candidate"):
                validate_example(corpus_v3_input, self.schema, config)
            rights = copy.deepcopy(row)
            rights["rights"]["evidence_sha256"] = "pending"
            with self.assertRaises(PilotValidationError):
                validate_example(rights, self.schema, config)
            missing_public_domain = copy.deepcopy(row)
            del missing_public_domain["provenance"]["public_domain_source"]
            with self.assertRaisesRegex(PilotValidationError, "public_domain_source|public-domain"):
                validate_example(missing_public_domain, self.schema, config)
            ai_original = example(1)
            ai_original["provenance"]["ai_assistance"]["final_content_sha256"] = "0" * 64
            with self.assertRaisesRegex(PilotValidationError, "final content hash"):
                validate_example(ai_original, self.schema, config)
            bad_prompt = example(2)
            bad_prompt["preliminary_assessment"]["prompt"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(PilotValidationError, "prompt hash"):
                validate_example(bad_prompt, self.schema, config)
            extra_ai = copy.deepcopy(row)
            extra_ai["provenance"]["ai_assistance"] = example(1)["provenance"]["ai_assistance"]
            with self.assertRaisesRegex(PilotValidationError, "source class"):
                validate_example(extra_ai, self.schema, config)

            false_acquisition = copy.deepcopy(row)
            false_acquisition["provenance"]["public_domain_source"]["independently_acquired_from_documented_source"] = False
            with self.assertRaises(PilotValidationError):
                validate_example(false_acquisition, self.schema, config)
            absent_acquisition = copy.deepcopy(row)
            del absent_acquisition["provenance"]["public_domain_source"]["copied_from_unsealed_corpus_v3_candidate"]
            with self.assertRaises(PilotValidationError):
                validate_example(absent_acquisition, self.schema, config)
            disclosed_overlap = example(2)
            validate_example(disclosed_overlap, self.schema, config)
            undisclosed_overlap = copy.deepcopy(row)
            undisclosed_overlap["provenance"]["public_domain_source"]["corpus_v3_overlap"] = "identified"
            with self.assertRaisesRegex(PilotValidationError, "pretraining exposure disclosure"):
                validate_example(undisclosed_overlap, self.schema, config)

            unsafe_source = copy.deepcopy(row)
            unsafe_source["provenance"]["public_domain_source"]["source_path"] = "../source-0.txt"
            with self.assertRaisesRegex(PilotValidationError, "safe root-relative"):
                validate_example(unsafe_source, self.schema, config)
            unsafe_rights = copy.deepcopy(row)
            unsafe_rights["rights"]["evidence_path"] = str(Path(directory) / "rights-0.txt")
            with self.assertRaisesRegex(PilotValidationError, "safe root-relative"):
                validate_example(unsafe_rights, self.schema, config)
            bad_source_hash = copy.deepcopy(row)
            bad_source_hash["provenance"]["public_domain_source"]["source_sha256"] = "0" * 64
            with self.assertRaisesRegex(PilotValidationError, "retained byte hash"):
                validate_example(bad_source_hash, self.schema, config)
            bad_raw_output = example(1)
            bad_raw_output["provenance"]["ai_assistance"]["raw_output_sha256"] = "0" * 64
            with self.assertRaisesRegex(PilotValidationError, "retained byte hash"):
                validate_example(bad_raw_output, self.schema, config)

            mismatched_assessment = example(0)
            mismatched_output = json.loads(assessment_output_bytes(mismatched_assessment))
            mismatched_output["risk_flags"] = ["ambiguous"]
            mismatched_bytes = (canonical_json(mismatched_output) + "\n").encode()
            mismatch_path = Path(config["retained_artifact_roots"]["raw_outputs"]) / "mismatch.json"
            mismatch_path.write_bytes(mismatched_bytes)
            mismatched_assessment["preliminary_assessment"].update({
                "raw_output_path": "mismatch.json",
                "raw_output_sha256": sha256_bytes(mismatched_bytes),
            })
            with self.assertRaisesRegex(PilotValidationError, "does not exactly match metadata"):
                validate_example(mismatched_assessment, self.schema, config)

            ai_assessment = example(0)
            ai_assessment["preliminary_assessment"].update({
                "method": "ai_assisted", "model_identity": "assessment-model/exact-revision",
            })
            self.bind_assessment_output(config, ai_assessment, "ai-assessment")
            validate_example(ai_assessment, self.schema, config)
            missing_model = copy.deepcopy(ai_assessment)
            del missing_model["preliminary_assessment"]["model_identity"]
            with self.assertRaisesRegex(PilotValidationError, "fields do not match method"):
                validate_example(missing_model, self.schema, config)

            prompt_text = "A retained exact authoring prompt for the fixture."
            prompt_path = Path(config["retained_artifact_roots"]["prompts"]) / "authoring-1.txt"
            prompt_path.write_text(prompt_text, encoding="utf-8", newline="\n")
            referenced_prompt = example(1)
            referenced_prompt["provenance"]["ai_assistance"]["prompt"] = {
                "reference": "authoring-1.txt", "sha256": sha256_file(prompt_path),
            }
            validate_example(referenced_prompt, self.schema, config)
            referenced_prompt["provenance"]["ai_assistance"]["prompt"]["reference"] = "../authoring-1.txt"
            with self.assertRaisesRegex(PilotValidationError, "safe root-relative"):
                validate_example(referenced_prompt, self.schema, config)

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
            contaminated[1]["provenance"]["ai_assistance"]["final_content_sha256"] = sha256_bytes(
                (contaminated[1]["opening"] + contaminated[1]["continuation"]).encode()
            )
            self.bind_assessment_output(config, contaminated[1], "contaminated-content")
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

            reserved_prompt = prompts[0]["prompt"]
            hidden_input = copy.deepcopy(records)
            hidden_input[0]["authoring_inputs"] = [reserved_prompt]
            report, _ = audit_examples(hidden_input, config, self.schema, self.tokenizer)
            self.assertTrue(any(
                item["kind"].startswith("candidate_metadata_")
                for item in report["gates"]["contamination"]["benchmark_and_scorecard_collisions"]
            ))
            hidden_ai_config = copy.deepcopy(records)
            hidden_ai_config[1]["provenance"]["ai_assistance"]["generation_configuration"] = reserved_prompt
            report, _ = audit_examples(hidden_ai_config, config, self.schema, self.tokenizer)
            self.assertTrue(report["gates"]["contamination"]["benchmark_and_scorecard_collisions"])
            hidden_assessment_config = copy.deepcopy(records)
            hidden_assessment_config[2]["preliminary_assessment"]["generation_configuration"] = reserved_prompt
            self.bind_assessment_output(config, hidden_assessment_config[2], "reserved-config")
            report, _ = audit_examples(hidden_assessment_config, config, self.schema, self.tokenizer)
            self.assertTrue(report["gates"]["contamination"]["benchmark_and_scorecard_collisions"])
            hidden_assessment_prompt = copy.deepcopy(records)
            prompt_path = Path(config["retained_artifact_roots"]["prompts"]) / "reserved.txt"
            prompt_path.write_text(reserved_prompt, encoding="utf-8", newline="\n")
            hidden_assessment_prompt[2]["preliminary_assessment"]["prompt"] = {
                "reference": "reserved.txt", "sha256": sha256_file(prompt_path),
            }
            self.bind_assessment_output(config, hidden_assessment_prompt[2], "reserved-prompt")
            report, _ = audit_examples(hidden_assessment_prompt, config, self.schema, self.tokenizer)
            self.assertTrue(any(
                item["kind"] == "preliminary_assessment_prompt_resolved"
                for item in report["gates"]["contamination"]["benchmark_and_scorecard_collisions"]
            ))

            source_contaminated = copy.deepcopy(records)
            source_bytes = artifact_bytes("source", 0) + reserved_prompt.encode() + b"\n"
            source_path = Path(config["retained_artifact_roots"]["sources"]) / "source-0-reserved.txt"
            source_path.write_bytes(source_bytes)
            source_contaminated[0]["provenance"]["public_domain_source"].update({
                "source_path": source_path.name, "source_sha256": sha256_bytes(source_bytes),
            })
            report, _ = audit_examples(source_contaminated, config, self.schema, self.tokenizer)
            source_gate = report["gates"]["contamination"]["source_input_exclusion"]
            self.assertIn("contamination", report["failed_gates"])
            self.assertFalse(source_gate["passed"])
            self.assertTrue(any(
                item["match"] == "full_reference_containment" for item in source_gate["collisions"]
            ))
            self.assertIn("12-word", source_gate["rule"])

    def test_owner_selection_is_deterministic_covers_strata_and_includes_all_flags(self):
        records = [example(index) for index in range(25)]
        records[3]["preliminary_assessment"]["risk_flags"] = ["ambiguous"]
        records[17]["preliminary_assessment"]["risk_flags"] = ["rights_sensitive", "low_quality"]
        for row in (records[3], records[17]):
            row["preliminary_assessment"]["raw_output_sha256"] = sha256_bytes(assessment_output_bytes(row))
        with tempfile.TemporaryDirectory() as directory:
            _, config = self.config(Path(directory), records)
            report, prepared = audit_examples(records, config, self.schema, self.tokenizer)
            self.assertEqual(report["status"], "PASS_AUTOMATED_AUDIT")
            first = select_owner_review_rows(prepared, target=10, seed=8018)
            second = select_owner_review_rows(list(reversed(prepared)), target=10, seed=8018)
            self.assertEqual(first, second)
            selected = {row["example_id"]: reasons for row, reasons in first}
            self.assertIn("cpv1-example-3", selected)
            self.assertIn("cpv1-example-17", selected)
            selected_rows = [row for row, _ in first]
            self.assertEqual({row["genre"] for row in selected_rows}, set(config["expected_genre_counts"]))
            self.assertEqual({row["primary_state_family"] for row in selected_rows}, set(config["expected_state_family_counts"]))
            self.assertEqual({row["split"] for row in selected_rows}, {"train", "validation", "test"})
            self.assertEqual({row["provenance"]["source_class"] for row in selected_rows}, {
                "public_domain_adaptation", "ai_assisted_original", "ai_assisted_public_domain_adaptation",
            })
            all_reasons = {reason for reasons in selected.values() for reason in reasons}
            self.assertIn("length:shortest", all_reasons)
            self.assertIn("length:longest", all_reasons)
            self.assertIn("risk_stratum:clear", all_reasons)
            self.assertIn("risk_stratum:flagged", all_reasons)

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
            "opening": "opening evidence appears clearly", "continuation": "continuation evidence appears clearly", "premise": "p",
            "protagonist": "h", "protagonist_motive": "m", "atomic_facts": ["f1", "f2", "f3"],
            "content_sha256": "a" * 64, "reviewer_id": "", "reviewer_role": "original",
            "independent_review_confirmed": None, "genuine_human_attestation": None, "fact_labels": {},
            "rights_and_provenance_acceptable": "", "premise_preserved": "", "motive_preserved": "",
            "opening_facts_consistent": "", "contradiction_free": "", "no_narrative_loop": "",
            "plot_advanced": "", "genre_voice": "", "meaningful_action": "",
            "meaningful_consequence": "", "natural_readable_continuation": "",
            "exploratory_sft_suitable": "", "evidence": {}, "reviewer_note": "",
        }
        digest = "b" * 64
        rows = [completed(packet, "person-a", digest), completed(packet, "person-b", digest)]
        self.assertEqual(len(resolve_reviews([packet], rows, digest)), 1)
        bad_sentinel = copy.deepcopy(rows)
        bad_sentinel[0]["evidence"]["premise_preserved"] = "<PROVENANCE_RECORD>"
        with self.assertRaisesRegex(PilotValidationError, "only for rights/provenance"):
            resolve_reviews([packet], bad_sentinel, digest)
        bad_absence = copy.deepcopy(rows)
        bad_absence[0]["evidence"]["premise_preserved"] = "<ABSENT>"
        with self.assertRaisesRegex(PilotValidationError, "positive/retained"):
            resolve_reviews([packet], bad_absence, digest)
        bad_rights_evidence = copy.deepcopy(rows)
        bad_rights_evidence[0]["evidence"]["rights_and_provenance_acceptable"] = packet["opening"]
        with self.assertRaisesRegex(PilotValidationError, "must be exactly"):
            resolve_reviews([packet], bad_rights_evidence, digest)
        trivial_quote = copy.deepcopy(rows)
        trivial_quote[0]["evidence"]["premise_preserved"] = "opening"
        with self.assertRaisesRegex(PilotValidationError, "at least 3 normalized words"):
            resolve_reviews([packet], trivial_quote, digest)
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
            "contradiction_free", "no_narrative_loop", "plot_advanced", "genre_voice",
            "meaningful_action", "meaningful_consequence", "natural_readable_continuation",
            "exploratory_sft_suitable", "rights_and_provenance_acceptable",
        )))
        self.assertFalse(resolved_reviews_pass_quality([merged]))
        rows[0]["packet_sha256"] = "c" * 64
        with self.assertRaisesRegex(PilotValidationError, "invalidated"):
            resolve_reviews([packet], rows, digest)

        owner_packet = copy.deepcopy(packet)
        owner_packet["reviewer_role"] = "owner"
        owner = completed(owner_packet, "owner-a", digest, "owner")
        owner["exploratory_sft_suitable"] = "uncertain"
        resolved_owner = resolve_reviews(
            [owner_packet], [owner], digest, "single_owner_research_pilot",
        )
        self.assertFalse(resolved_reviews_pass_quality(resolved_owner))
        owner["exploratory_sft_suitable"] = "no"
        self.assertFalse(resolved_reviews_pass_quality(resolve_reviews(
            [owner_packet], [owner], digest, "single_owner_research_pilot",
        )))

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
            packet = load_jsonl(packet_path)
            self.assertEqual(len(packet), 50)
            reviews = [completed(row, "owner-a", packet_hash, "owner") for row in packet]
            write_jsonl(Path(config["reviews_path"]), reviews)
            prepared_hash = sha256_file(output / "prepared.jsonl")
            owner_verification = {
                "version": 1, "approval_mode": "single_owner_research_pilot",
                "prepared_sha256": prepared_hash, "review_packet_sha256": packet_hash,
                "owner_id": "owner-a", "verified_at": "2026-10-10",
                "verification_evidence_reference": "private verification ledger",
                "genuine_human_attestation": True,
                "all_source_rights_and_provenance_examined": True,
                "all_example_identities_verified": True,
                "all_final_content_hash_linkages_verified": True,
                "distinctive_scorecard_fact_combinations_checked": True,
                "no_distinctive_scorecard_fact_combination_collision": True,
                "held_out_prompts_excluded_from_authoring_editing_and_candidate_selection": True,
                "no_unsealed_corpus_v3_candidate_bytes_or_artifacts_used_for_authoring_editing_or_candidate_selection": True,
                "corpus_v3_and_pretraining_exposure_overlap_audited_and_disclosed": True,
                "all_selected_and_flagged_examples_reviewed": True,
                "reviewed_example_ids": sorted(row["example_id"] for row in packet),
                "unreviewed_unflagged_example_count": 150,
                "unreviewed_unflagged_limitations_acknowledged": True,
                "automated_ai_diagnostics_are_not_human_review_acknowledged": True,
                "final_approval": True,
            }
            write_json_atomic(Path(config["owner_verification_path"]), owner_verification)
            rejected_verification = copy.deepcopy(owner_verification)
            rejected_verification["no_unsealed_corpus_v3_candidate_bytes_or_artifacts_used_for_authoring_editing_or_candidate_selection"] = False
            write_json_atomic(Path(config["owner_verification_path"]), rejected_verification)
            self.assertIn("verification gate is incomplete", finalize(config_path)["error"])
            write_json_atomic(Path(config["owner_verification_path"]), owner_verification)
            approved = finalize(config_path)
            self.assertEqual(approved["status"], "APPROVED")
            self.assertEqual(approved["approval_mode"], "single_owner_research_pilot")
            self.assertFalse(approved["independent_human_validation"])
            self.assertFalse(approved["publication_grade_certification"])
            self.assertFalse(approved["adjudication_performed"])
            self.assertEqual(approved["directly_reviewed_count"], 50)
            self.assertEqual(approved["unreviewed_unflagged_count"], 150)
            self.assertFalse(approved["unsealed_corpus_v3_candidate_bytes_or_artifacts_used"])
            self.assertTrue(approved["corpus_v3_and_pretraining_exposure_overlap_audited_and_disclosed"])
            self.assertIn("not human review", approved["automated_ai_diagnostic_limitation"])
            self.assertGreater(approved["tokenizer_v1_tokens_total"], 0)
            self.assertEqual(set(approved["tokenizer_v1_tokens_by_split"]), {"train", "validation", "test"})
            self.assertTrue((output / "approval.json").is_file())

            key_path = output / "review-key.json"
            original_packet = packet_path.read_bytes()
            original_key = key_path.read_bytes()
            tampered_packet = load_jsonl(packet_path)
            tampered_packet[0]["selection_reasons"].append("tampered")
            write_jsonl(packet_path, tampered_packet)
            tampered_key = json.loads(key_path.read_text(encoding="utf-8"))
            tampered_key["packet_sha256"] = sha256_file(packet_path)
            write_json_atomic(key_path, tampered_key)
            tampered = finalize(config_path)
            self.assertIn("fresh reconstruction", tampered["error"])
            self.assertFalse((output / "approval.json").exists())
            packet_path.write_bytes(original_packet)
            key_path.write_bytes(original_key)
            tampered_key = json.loads(key_path.read_text(encoding="utf-8"))
            first_mapping = next(iter(tampered_key["mapping"].values()))
            first_mapping["split"] = "test" if first_mapping["split"] != "test" else "train"
            write_json_atomic(key_path, tampered_key)
            key_tampered = finalize(config_path)
            self.assertIn("review key", key_tampered["error"])
            self.assertFalse((output / "approval.json").exists())
            key_path.write_bytes(original_key)
            self.assertEqual(finalize(config_path)["status"], "APPROVED")

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

    def test_strict_mode_production_contract_requires_two_reviewers_and_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, config = self.production_config(
                root, [production_example(index) for index in range(200)],
            )
            exported = export_review(config_path, approval_mode="strict_independent")
            self.assertEqual(exported["approval_mode"], "strict_independent")
            self.assertEqual(exported["review_count"], 200)
            output = Path(config["output_dir"])
            packet_path = output / "review-packet-strict-independent.jsonl"
            packet_hash = sha256_file(packet_path)
            packet = load_jsonl(packet_path)
            incomplete = [completed(row, "reviewer-a", packet_hash) for row in packet]
            write_jsonl(Path(config["reviews_path"]), incomplete)
            stopped = finalize(config_path, approval_mode="strict_independent")
            self.assertIn("exactly two distinct", stopped["error"])
            self.assertFalse((output / "approval.json").exists())

            reviews = [
                completed(row, reviewer, packet_hash)
                for row in packet for reviewer in ("reviewer-a", "reviewer-b")
            ]
            write_jsonl(Path(config["reviews_path"]), reviews)
            strict_verification = {
                "version": 1, "approval_mode": "strict_independent",
                "prepared_sha256": sha256_file(output / "prepared.jsonl"),
                "review_packet_sha256": packet_hash,
                "custodian_id": "strict-verifier", "verified_at": "2026-10-10",
                "verification_evidence_reference": "private strict verification ledger",
                "rights_evidence_examined": True, "provenance_evidence_examined": True,
                "all_examples_rights_cleared": True, "reviewer_identities_verified": True,
                "reviewers_genuine_humans": True, "independent_review_process_verified": True,
                "adjudicator_identities_verified": True,
                "distinctive_scorecard_fact_combinations_checked": True,
                "no_distinctive_scorecard_fact_combination_collision": True,
                "scene_post_prompts_excluded_from_authoring_editing_and_candidate_selection": True,
                "no_unsealed_corpus_v3_candidate_bytes_or_artifacts_used_for_authoring_editing_or_candidate_selection": True,
                "corpus_v3_and_pretraining_exposure_overlap_audited_and_disclosed": True,
                "verified_reviewer_ids": ["reviewer-a", "reviewer-b"],
                "verified_adjudicator_ids": [],
            }
            write_json_atomic(Path(config["strict_verification_path"]), strict_verification)
            rejected_verification = copy.deepcopy(strict_verification)
            rejected_verification["corpus_v3_and_pretraining_exposure_overlap_audited_and_disclosed"] = False
            write_json_atomic(Path(config["strict_verification_path"]), rejected_verification)
            self.assertIn(
                "verification gate is incomplete",
                finalize(config_path, approval_mode="strict_independent")["error"],
            )
            write_json_atomic(Path(config["strict_verification_path"]), strict_verification)
            approved = finalize(config_path, approval_mode="strict_independent")
            self.assertEqual(approved["status"], "APPROVED")
            self.assertTrue(approved["independent_human_validation"])
            self.assertFalse(approved["publication_grade_certification"])
            self.assertEqual(approved["directly_reviewed_count"], 200)
            self.assertEqual(approved["unreviewed_count"], 0)
            self.assertFalse(approved["unsealed_corpus_v3_candidate_bytes_or_artifacts_used"])
            self.assertTrue(approved["corpus_v3_and_pretraining_exposure_overlap_audited_and_disclosed"])

            key_path = output / "review-key-strict-independent.json"
            original_packet = packet_path.read_bytes()
            original_key = key_path.read_bytes()
            tampered_packet = load_jsonl(packet_path)
            tampered_packet[0]["opening"] += " tampered"
            write_jsonl(packet_path, tampered_packet)
            tampered_key = json.loads(key_path.read_text(encoding="utf-8"))
            tampered_key["packet_sha256"] = sha256_file(packet_path)
            write_json_atomic(key_path, tampered_key)
            tampered = finalize(config_path, approval_mode="strict_independent")
            self.assertIn("fresh reconstruction", tampered["error"])
            self.assertFalse((output / "approval.json").exists())
            packet_path.write_bytes(original_packet)
            key_path.write_bytes(original_key)
            self.assertEqual(
                finalize(config_path, approval_mode="strict_independent")["status"], "APPROVED",
            )


if __name__ == "__main__":
    unittest.main()
