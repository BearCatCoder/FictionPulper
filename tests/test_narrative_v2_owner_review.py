import copy
import json
import random
import tempfile
import unittest
from argparse import Namespace
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.continuity_curriculum.common import (
    load_jsonl, sha256_bytes, sha256_file, write_json_atomic, write_jsonl,
)
from src.narrative_v2_owner_review import (
    DEPTHS,
    SPLITS,
    TRIALS,
    OwnerReviewError,
    _jsonl_sha,
    _stage1_public_manifest,
    build_owner_packet,
    paired_model_difference,
    parse_args,
    run_terminal_review,
    score_owner_review,
    select_balanced_records,
    select_stage1_blocks,
    select_stage2_blocks,
    stage2_trigger_decision,
    structural_blocks,
    validate_stage_reviews,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/narrative-v2-owner-review-v1.json"
STAGED_CONFIG = ROOT / "configs/narrative-v2-owner-review-staged-v1.json"
RUBRIC = ROOT / "benchmarks/narrative-v2/human-review-rubric.json"


def generation(
    model: str, split: str, family: str, depth: str, scenario: str,
    mode: str, seed: int | None, world: str,
) -> dict:
    text = f"Alden says the annotation is {'amber' if world == 'A' else 'badger'}."
    identifier = f"{model}:{split}:{scenario}:{mode}:{seed}:{world}"
    return {
        "generation_id": identifier, "scenario_id": scenario, "world": world,
        "generation_mode": mode, "sampling_seed": seed, "effective_seed": seed,
        "blinded_model_id": model, "split": split, "state_family": family,
        "reasoning_depth": 5 if depth == "4_plus" else int(depth), "depth_bucket": depth,
        "generated_text_sha256": sha256_bytes(text.encode()), "prompt": "Alden reads the ledger.",
        "generated_text": text, "review_question": "Does the state remain correct?",
        "required_propositions": ["The current state is preserved."],
        "forbidden_propositions": ["The counterfactual state is asserted."],
        "diagnostics": {
            "current_fact_only": True, "explicit_contradiction": False,
            "causal_adherence": {"diagnostic_pass": True},
            "named_entity_retention_rate": 1.0, "stale_fact_mentioned": False,
            "premise": {"retained_without_counterfactual": True},
            "repetition": {"repeated_4gram_rate": 0.0, "repeated_sentence_rate": 0.0},
        },
    }


def population() -> list[dict]:
    rows = []
    for model in ("baseline-a", "baseline-b"):
        for split in SPLITS:
            for family_index in range(7):
                family = f"family-{family_index}"
                for depth in DEPTHS:
                    count = 2 if split == "generalization" else 1
                    for scenario_index in range(count):
                        scenario = f"{split}:{family}:{depth}:{scenario_index}"
                        for mode, seed in TRIALS:
                            for world in ("A", "B"):
                                rows.append(generation(
                                    model, split, family, depth, scenario, mode, seed, world
                                ))
    return rows


def stage_fixture() -> dict:
    staged = json.loads(STAGED_CONFIG.read_text())
    base = json.loads(BASE_CONFIG.read_text())
    full_selected = select_balanced_records(population(), selection_seed=base["selection_seed"])
    full_packet, full_metadata = build_owner_packet(
        full_selected, packet_order_seed=base["packet_order_seed"]
    )
    blocks = structural_blocks(full_selected)
    stage1 = select_stage1_blocks(blocks, staged)
    stage2 = select_stage2_blocks(blocks, stage1, staged)
    generation_ids = {row["generation_id"] for block in stage1 for row in block["records"]}
    identifiers = {
        identifier for identifier, hidden in full_metadata.items()
        if hidden["generation_id"] in generation_ids
    }
    packet = [row for row in full_packet if row["owner_review_id"] in identifiers]
    metadata = {identifier: full_metadata[identifier] for identifier in identifiers}
    return {
        "staged": staged, "base": base, "full_selected": full_selected,
        "full_packet": full_packet, "full_metadata": full_metadata, "blocks": blocks,
        "stage1_blocks": stage1, "stage2_blocks": stage2,
        "stage_packet": packet, "stage_metadata": metadata, "sources": [],
    }


def completed_review(
    original: dict, packet_hash: str, manifest_hash: str, *, owner: str = "BearCatCoder",
    label: str = "consistent", reviewed_at: str = "2026-10-12T12:00:00Z",
    evidence_type: str = "verbatim_quote", evidence_quote: str | None = None,
    legacy_schema: bool = False,
) -> dict:
    row = copy.deepcopy(original)
    row.update({
        "reviewer_id": owner, "reviewer_role": "owner",
        "independent_review_confirmed": False, "genuine_human_attestation": True,
        "label": label, "confidence": 5,
        "evidence_quote": row["generated_text"] if evidence_quote is None else evidence_quote,
        "reviewer_note": "The state is explicit.", "packet_sha256": packet_hash,
        "stage_manifest_sha256": manifest_hash, "reviewed_at": reviewed_at,
    })
    if not legacy_schema:
        row.update({"review_record_schema_version": 2, "evidence_type": evidence_type})
    return row


def verification(packet: list[dict], packet_hash: str, manifest_hash: str) -> dict:
    return {
        "version": "narrative-v2-owner-review-staged-v1", "stage": 1,
        "review_packet_sha256": packet_hash, "public_manifest_sha256": manifest_hash,
        "owner_id": "BearCatCoder", "completed_at": "2026-10-12T13:00:00Z",
        "selected_owner_review_ids": [row["owner_review_id"] for row in packet],
        "genuine_human_attestation": True, "single_owner_evidence_acknowledged": True,
        "non_independent_evidence_acknowledged": True, "sample_only_evidence_acknowledged": True,
        "no_agreement_available_acknowledged": True,
        "no_adjudication_performed_acknowledged": True,
        "automated_diagnostics_are_not_owner_decisions_acknowledged": True,
    }


class NarrativeV2OwnerReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.materials = stage_fixture()

    def test_stage1_exact_balance_is_order_and_text_diagnostic_blind(self):
        selected = self.materials["stage1_blocks"]
        self.assertEqual(len(selected), 20)
        self.assertEqual(Counter(block["trial"] for block in selected), {
            "greedy": 5, "sampled:11337": 5, "sampled:21337": 5, "sampled:31337": 5,
        })
        self.assertEqual(Counter(block["depth_bucket"] for block in selected), {depth: 5 for depth in DEPTHS})
        self.assertEqual(Counter(block["split"] for block in selected), {
            "development": 6, "test": 7, "generalization": 7,
        })
        self.assertEqual(sorted(Counter(block["state_family"] for block in selected).values()),
                         [2, 3, 3, 3, 3, 3, 3])
        changed = copy.deepcopy(self.materials["full_selected"])
        random.Random(99).shuffle(changed)
        for row in changed:
            row["generated_text"] = "quality must not affect selection"
            row["diagnostics"] = {"ignored": True}
        repeated = select_stage1_blocks(structural_blocks(changed), self.materials["staged"])
        self.assertEqual([block["block_id"] for block in selected],
                         [block["block_id"] for block in repeated])

    def test_stage2_is_deterministic_disjoint_and_cumulatively_balanced(self):
        first = self.materials["stage2_blocks"]
        repeated = select_stage2_blocks(
            list(reversed(self.materials["blocks"])), self.materials["stage1_blocks"],
            self.materials["staged"],
        )
        self.assertEqual([block["block_id"] for block in first],
                         [block["block_id"] for block in repeated])
        self.assertFalse({block["block_id"] for block in first}.intersection(
            block["block_id"] for block in self.materials["stage1_blocks"]
        ))
        cumulative = [*self.materials["stage1_blocks"], *first]
        self.assertEqual(Counter(block["trial"] for block in cumulative), {
            "greedy": 10, "sampled:11337": 10, "sampled:21337": 10, "sampled:31337": 10,
        })
        self.assertEqual(Counter(block["depth_bucket"] for block in cumulative),
                         {depth: 10 for depth in DEPTHS})
        self.assertEqual(Counter(block["split"] for block in cumulative),
                         {"development": 13, "test": 13, "generalization": 14})
        self.assertEqual(sorted(Counter(block["state_family"] for block in cumulative).values()),
                         [5, 5, 6, 6, 6, 6, 6])

    def test_stage_packet_and_public_manifest_do_not_expose_key_or_conditions(self):
        packet = self.materials["stage_packet"]
        forbidden = {
            "actual_model_id", "model_id", "blinded_model_id", "split", "scenario_id", "world",
            "generation_mode", "sampling_seed", "state_family", "depth_bucket", "reasoning_depth",
            "source_path", "model_mapping",
        }
        self.assertEqual(len(packet), 80)
        self.assertTrue(all(not forbidden.intersection(row) for row in packet))
        manifest = _stage1_public_manifest(
            STAGED_CONFIG, self.materials["staged"], self.materials["base"],
            self.materials["stage1_blocks"], _jsonl_sha(packet),
        )
        serialized = json.dumps(manifest)
        self.assertNotIn("baseline-a", serialized)
        self.assertNotIn("baseline-b", serialized)
        self.assertNotIn("fictionpulper-15m", serialized)
        self.assertEqual(manifest["coverage"]["split_cell_block_counts"], [6, 7, 7])
        self.assertFalse(manifest["coverage"]["complete_family_by_depth_by_split_coverage"])

    def test_terminal_review_saves_each_row_resumes_and_never_replaces(self):
        packet = self.materials["stage_packet"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet_path = root / "packet.jsonl"
            manifest_path = root / "manifest.json"
            reviews_path = root / "reviews.jsonl"
            verification_path = root / "verification.json"
            write_jsonl(packet_path, packet)
            manifest = _stage1_public_manifest(
                STAGED_CONFIG, self.materials["staged"], self.materials["base"],
                self.materials["stage1_blocks"], sha256_file(packet_path),
            )
            write_json_atomic(manifest_path, manifest)
            packet_hash = sha256_file(packet_path)
            manifest_hash = sha256_file(manifest_path)
            progress = [
                completed_review(row, packet_hash, manifest_hash, legacy_schema=True)
                for row in packet[:78]
            ]
            write_jsonl(reviews_path, progress)
            answers = iter([
                "consistent", "5", packet[78]["generated_text"], "Explicit state.",
            ])

            def interrupted(prompt: str) -> str:
                try:
                    return next(answers)
                except StopIteration as error:
                    raise RuntimeError("simulated interruption") from error

            args = Namespace(
                packet=packet_path, manifest=manifest_path, reviews=reviews_path,
                reviewer_id="BearCatCoder", dry_run=False, final_attestation=False,
                owner_verification=None,
            )
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                run_terminal_review(
                    args, input_fn=interrupted, output_fn=lambda _: None,
                    now_fn=lambda: datetime(2026, 10, 12, 14, 0, tzinfo=timezone.utc),
                )
            saved_rows = load_jsonl(reviews_path)
            self.assertEqual(len(saved_rows), 79)
            self.assertEqual(saved_rows[-1]["reviewed_at"], "2026-10-12T14:00:00Z")
            immutable_first = copy.deepcopy(saved_rows[0])
            answers = iter(["consistent", "4", packet[79]["generated_text"], "Still explicit."])
            args.final_attestation = True
            args.owner_verification = verification_path
            result = run_terminal_review(
                args, input_fn=lambda _: next(answers), output_fn=lambda _: None,
                now_fn=lambda: datetime(2026, 10, 12, 15, 0, tzinfo=timezone.utc),
            )
            self.assertEqual(result["status"], "STAGE_1_REVIEW_COMPLETE")
            completed = load_jsonl(reviews_path)
            self.assertEqual(completed[0], immutable_first)
            self.assertTrue(verification_path.is_file())
            with self.assertRaises(FileExistsError):
                run_terminal_review(args, input_fn=lambda _: "", output_fn=lambda _: None)

    def test_terminal_review_persists_required_fact_omission_without_quote(self):
        packet = self.materials["stage_packet"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet_path = root / "packet.jsonl"
            manifest_path = root / "manifest.json"
            reviews_path = root / "reviews.jsonl"
            write_jsonl(packet_path, packet)
            manifest = _stage1_public_manifest(
                STAGED_CONFIG, self.materials["staged"], self.materials["base"],
                self.materials["stage1_blocks"], sha256_file(packet_path),
            )
            write_json_atomic(manifest_path, manifest)
            packet_hash = sha256_file(packet_path)
            manifest_hash = sha256_file(manifest_path)
            write_jsonl(reviews_path, [
                completed_review(row, packet_hash, manifest_hash) for row in packet[:79]
            ])
            answers = iter([
                "inconsistent", "4", "required_proposition_absent",
                "The required terminal fact is never stated.",
            ])
            result = run_terminal_review(
                Namespace(
                    packet=packet_path, manifest=manifest_path, reviews=reviews_path,
                    reviewer_id="BearCatCoder", dry_run=False, final_attestation=False,
                    owner_verification=None,
                ),
                input_fn=lambda _: next(answers), output_fn=lambda _: None,
                now_fn=lambda: datetime(2026, 10, 12, 16, 0, tzinfo=timezone.utc),
            )
            self.assertEqual(result["status"], "STAGE_1_REVIEW_COMPLETE")
            saved = load_jsonl(reviews_path)[-1]
            self.assertEqual(saved["review_record_schema_version"], 2)
            self.assertEqual(saved["evidence_type"], "required_proposition_absent")
            self.assertEqual(saved["evidence_quote"], "")
            self.assertEqual(saved["reviewer_note"], "The required terminal fact is never stated.")

    def test_review_validation_rejects_timestamps_tampering_and_wrong_bindings(self):
        packet = self.materials["stage_packet"]
        packet_hash = "a" * 64
        manifest_hash = "b" * 64
        rows = [completed_review(row, packet_hash, manifest_hash) for row in packet]
        completed, owner = validate_stage_reviews(
            packet, rows, packet_sha256=packet_hash, manifest_sha256=manifest_hash,
            require_complete=True,
        )
        self.assertEqual(len(completed), 80)
        self.assertEqual(owner, "BearCatCoder")
        changed = copy.deepcopy(rows)
        changed[0]["prompt"] = "tampered"
        with self.assertRaisesRegex(OwnerReviewError, "immutable"):
            validate_stage_reviews(packet, changed, packet_sha256=packet_hash,
                                   manifest_sha256=manifest_hash, require_complete=True)
        changed = copy.deepcopy(rows)
        changed[0]["reviewed_at"] = "2026-10-12T12:00:00"
        with self.assertRaisesRegex(OwnerReviewError, "UTC timezone"):
            validate_stage_reviews(packet, changed, packet_sha256=packet_hash,
                                   manifest_sha256=manifest_hash, require_complete=True)
        changed = copy.deepcopy(rows)
        changed[0]["stage_manifest_sha256"] = "c" * 64
        with self.assertRaisesRegex(OwnerReviewError, "manifest hash"):
            validate_stage_reviews(packet, changed, packet_sha256=packet_hash,
                                   manifest_sha256=manifest_hash, require_complete=True)

    def test_evidence_types_distinguish_omission_contradiction_and_empty_output(self):
        packet_hash = "a" * 64
        manifest_hash = "b" * 64
        nonempty = copy.deepcopy(self.materials["stage_packet"][0])

        omission = completed_review(
            nonempty, packet_hash, manifest_hash, label="inconsistent",
            evidence_type="required_proposition_absent", evidence_quote="",
        )
        omission["reviewer_note"] = "The continuation never states the required terminal fact."
        completed, _ = validate_stage_reviews(
            [nonempty], [omission], packet_sha256=packet_hash,
            manifest_sha256=manifest_hash, require_complete=True,
        )
        self.assertEqual(completed[nonempty["owner_review_id"]]["evidence_quote"], "")

        contradictory_output = copy.deepcopy(nonempty)
        contradictory_output["owner_review_id"] = "owner-review-contradiction"
        contradictory_output["generated_text"] = "The counterfactual state is asserted."
        contradictory_output["generated_text_sha256"] = sha256_bytes(
            contradictory_output["generated_text"].encode()
        )
        contradiction = completed_review(
            contradictory_output, packet_hash, manifest_hash, label="inconsistent",
            evidence_type="verbatim_quote",
            evidence_quote="The counterfactual state is asserted.",
        )
        validate_stage_reviews(
            [contradictory_output], [contradiction], packet_sha256=packet_hash,
            manifest_sha256=manifest_hash, require_complete=True,
        )

        empty = copy.deepcopy(nonempty)
        empty["owner_review_id"] = "owner-review-empty"
        empty["generated_text"] = ""
        empty["generated_text_sha256"] = sha256_bytes(b"")
        empty_review = completed_review(
            empty, packet_hash, manifest_hash, label="inconsistent",
            evidence_type="empty_output", evidence_quote="<EMPTY_OUTPUT>",
        )
        validate_stage_reviews(
            [empty], [empty_review], packet_sha256=packet_hash,
            manifest_sha256=manifest_hash, require_complete=True,
        )

        fabricated = copy.deepcopy(contradiction)
        fabricated["evidence_quote"] = "This quotation was never generated."
        with self.assertRaisesRegex(OwnerReviewError, "occur verbatim"):
            validate_stage_reviews(
                [contradictory_output], [fabricated], packet_sha256=packet_hash,
                manifest_sha256=manifest_hash, require_complete=True,
            )

        false_empty = completed_review(
            nonempty, packet_hash, manifest_hash, label="inconsistent",
            evidence_type="empty_output", evidence_quote="<EMPTY_OUTPUT>",
        )
        with self.assertRaisesRegex(OwnerReviewError, "only for empty"):
            validate_stage_reviews(
                [nonempty], [false_empty], packet_sha256=packet_hash,
                manifest_sha256=manifest_hash, require_complete=True,
            )

    def test_legacy_stage_review_records_remain_readable(self):
        packet = self.materials["stage_packet"][:1]
        packet_hash = "a" * 64
        manifest_hash = "b" * 64
        legacy = completed_review(
            packet[0], packet_hash, manifest_hash, legacy_schema=True,
        )
        completed, owner = validate_stage_reviews(
            packet, [legacy], packet_sha256=packet_hash,
            manifest_sha256=manifest_hash, require_complete=True,
        )
        self.assertEqual(owner, "BearCatCoder")
        self.assertNotIn("evidence_type", completed[packet[0]["owner_review_id"]])

    def test_paired_difference_and_trigger_rule_are_deterministic(self):
        pairs = []
        for index in range(20):
            for model in ("baseline-a", "baseline-b"):
                pairs.append({
                    "scenario_id": f"scenario-{index}", "generation_mode": "greedy",
                    "sampling_seed": None, "blinded_model_id": model,
                    "strict_success": model == "baseline-b" or index % 2 == 0,
                    "unresolved_uncertain": False,
                })
        first = paired_model_difference(
            pairs, arm_mapping={"arm-1": "baseline-a", "arm-2": "baseline-b"},
            replicates=500, seed=260209,
        )
        second = paired_model_difference(
            list(reversed(pairs)), arm_mapping={"arm-1": "baseline-a", "arm-2": "baseline-b"},
            replicates=500, seed=260209,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["matched_block_count"], 20)
        rule = self.materials["staged"]["stage_2_trigger"]
        decision = stage2_trigger_decision(first, {"arm-1": 0.0, "arm-2": 0.0}, rule)
        self.assertTrue(decision["triggered"])
        self.assertEqual(decision["authorized_additional_rows"], 80)
        narrow = {"point_difference": 0.0, "bootstrap": {"lower": -0.05, "upper": 0.05, "width": 0.10}}
        self.assertFalse(stage2_trigger_decision(narrow, {"arm-1": 0.1, "arm-2": 0.0}, rule)["triggered"])
        self.assertTrue(stage2_trigger_decision(narrow, {"arm-1": 0.1001, "arm-2": 0.0}, rule)["triggered"])

    def test_stage1_score_is_exploratory_and_never_reveals_mapping(self):
        materials = self.materials
        packet = materials["stage_packet"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet_path = root / "packet.jsonl"
            manifest_path = root / "manifest.json"
            key_path = root / "key.json"
            reviews_path = root / "reviews.jsonl"
            verification_path = root / "verification.json"
            output_path = root / "report.json"
            write_jsonl(packet_path, packet)
            packet_hash = sha256_file(packet_path)
            manifest = _stage1_public_manifest(
                STAGED_CONFIG, materials["staged"], materials["base"],
                materials["stage1_blocks"], packet_hash,
            )
            write_json_atomic(manifest_path, manifest)
            manifest_hash = sha256_file(manifest_path)
            key = {
                "version": "narrative-v2-owner-review-staged-v1", "stage": 1,
                "review_packet_sha256": packet_hash, "public_manifest_sha256": manifest_hash,
                "staged_config_sha256": sha256_file(STAGED_CONFIG),
                "review_metadata": materials["stage_metadata"],
                "anonymous_arm_mapping": {"arm-1": "baseline-a", "arm-2": "baseline-b"},
                "source_runs": [],
                "model_mapping": {
                    run["blinded_model_id"]: run["model_id"] for run in materials["base"]["runs"]
                },
                "full_packet_sha256": materials["staged"]["full_packet_sha256"],
                "full_private_key_sha256": materials["staged"]["full_private_key_sha256"],
                "selected_ids": [row["owner_review_id"] for row in packet],
                "selected_blocks": [{
                    field: block[field] for field in (
                        "block_id", "split", "state_family", "depth_bucket", "scenario_id",
                        "generation_mode", "sampling_seed", "trial",
                    )
                } for block in materials["stage1_blocks"]],
                "sample_coverage": {
                    "block_count": 20, "row_count": 80,
                    "trial_block_counts": dict(sorted(Counter(block["trial"] for block in materials["stage1_blocks"]).items())),
                    "depth_block_counts": dict(sorted(Counter(block["depth_bucket"] for block in materials["stage1_blocks"]).items())),
                    "split_block_counts": dict(sorted(Counter(block["split"] for block in materials["stage1_blocks"]).items())),
                    "family_block_counts": dict(sorted(Counter(block["state_family"] for block in materials["stage1_blocks"]).items())),
                },
            }
            write_json_atomic(key_path, key)
            reviews = [completed_review(row, packet_hash, manifest_hash) for row in packet]
            write_jsonl(reviews_path, reviews)
            write_json_atomic(verification_path, verification(packet, packet_hash, manifest_hash))
            args = Namespace(
                config=STAGED_CONFIG, checkpoint_selection_complete=True,
                test_evaluation_complete=True, packet=packet_path, manifest=manifest_path,
                private_key=key_path, reviews=reviews_path,
                owner_verification=verification_path, output=output_path,
            )
            with patch("src.narrative_v2_owner_review._staged_materials", return_value=materials), \
                 patch("builtins.print"):
                report = score_owner_review(args)
            serialized = json.dumps(report)
            self.assertEqual(report["evidence_scope"], "partial single-owner exploratory evidence")
            self.assertEqual(report["model_mapping"]["status"],
                             "withheld_until_all_triggered_stages_are_complete")
            self.assertNotIn("fictionpulper-15m", serialized)
            self.assertNotIn("fictionpulper-50m", serialized)
            self.assertNotIn("baseline-a", serialized)
            self.assertNotIn("baseline-b", serialized)
            trial_cells = [row for row in report["descriptive_metrics"] if row["dimension"] == "trial"]
            self.assertTrue(all(row["cell_size_blocks"] == 5 for row in trial_cells))
            self.assertFalse(report["stage_2_decision"]["stage_2_packet_generated"])

    def test_cli_access_flags_and_review_does_not_require_private_key(self):
        with self.assertRaises(SystemExit):
            parse_args(["export", "--output-dir", "runs/owner"])
        export = parse_args([
            "export", "--output-dir", "runs/owner",
            "--checkpoint-selection-complete", "--test-evaluation-complete",
        ])
        self.assertTrue(export.checkpoint_selection_complete)
        review = parse_args([
            "review", "--packet", "packet", "--manifest", "manifest",
            "--reviews", "reviews", "--reviewer-id", "owner", "--dry-run",
        ])
        self.assertFalse(hasattr(review, "private_key"))
        self.assertTrue(review.dry_run)


if __name__ == "__main__":
    unittest.main()
