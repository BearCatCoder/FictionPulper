import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.continuity_curriculum.common import sha256_bytes
from src.scene_scorecard import HUMAN_FIELDS, resolve_reviews
from src.scene_scorecard_ai_review import (
    PACKET_HASH, PROCEDURE_HASH, add_scores, paired_difference, score,
    summarize, validate_decisions,
)


def fixture():
    packet, decisions = [], []
    for i in range(80):
        text = "word " * 200
        digest = sha256_bytes(text.encode())
        packet.append({
            "review_id": f"row-{i}", "generated_text": text,
            "generated_text_sha256": digest, "word_count": 200,
            "length_compliant": True,
            "atomic_facts": [{"id": f"fact-{j}", "text": f"Fact {j}"} for j in range(4)],
        })
        evidence = {field: {"quote": "<ABSENT>", "note": "No semantic support."}
                    for field in [*(f"fact-{j}" for j in range(4)), *HUMAN_FIELDS]}
        evidence["contradiction_present"]["quote"] = "word"
        evidence["loop_or_repetition_present"]["quote"] = "word"
        decisions.append({
            "review_id": f"row-{i}", "generated_text_sha256": digest,
            "atomic_fact_labels": {f"fact-{j}": "not_retained" for j in range(4)},
            **{field: "no" for field in HUMAN_FIELDS},
            "evidence": evidence, "reviewer_note": "No supplied relation is preserved.",
        })
    artifact = {
        "schema": "scene-sol-ai-decisions-v1", "evaluation_kind": "ai_only_exploratory",
        "evaluator": {"name": "GPT-6.1 Sol", "model_id": "openai/gpt-6.1-sol",
                      "execution": "OpenCode conversation; no external model API",
                      "independent": False, "prior_aggregate_exposure": True},
        "packet_sha256": PACKET_HASH, "procedure_sha256": PROCEDURE_HASH,
        "procedure_freeze_commit": "cc5fb4c", "private_mapping_used_for_labels": False,
        "decisions": decisions,
    }
    return packet, artifact


class SceneAIReviewTests(unittest.TestCase):
    def test_complete_coverage_and_ai_human_boundary(self):
        packet, artifact = fixture()
        self.assertEqual(len(validate_decisions(packet, artifact)), 80)
        for change in (
            lambda a: a.update(genuine_human_attestation=True),
            lambda a: a["evaluator"].update(independent=True),
            lambda a: a["evaluator"].update(name="BearCatCoder"),
            lambda a: a.update(private_mapping_used_for_labels=True),
            lambda a: a["decisions"][0].update(reviewer_role="original"),
        ):
            bad = copy.deepcopy(artifact)
            change(bad)
            with self.assertRaises(RuntimeError):
                validate_decisions(packet, bad)
        with self.assertRaisesRegex(RuntimeError, "independent review"):
            resolve_reviews(packet, artifact["decisions"])

    def test_missing_duplicate_and_unknown_rows_fail(self):
        packet, artifact = fixture()
        for mutation in (
            lambda a: a["decisions"].pop(),
            lambda a: a["decisions"].__setitem__(1, a["decisions"][0]),
            lambda a: a["decisions"][0].update(review_id="unknown"),
        ):
            bad = copy.deepcopy(artifact)
            mutation(bad)
            with self.assertRaises(RuntimeError):
                validate_decisions(packet, bad)

    def test_wrong_bindings_and_length_fail(self):
        packet, artifact = fixture()
        for field in ("packet_sha256", "procedure_sha256", "procedure_freeze_commit"):
            bad = copy.deepcopy(artifact)
            bad[field] = "bad"
            with self.assertRaises(RuntimeError):
                validate_decisions(packet, bad)
        artifact["decisions"][0]["generated_text_sha256"] = "bad"
        with self.assertRaisesRegex(RuntimeError, "text binding"):
            validate_decisions(packet, artifact)
        packet, artifact = fixture()
        packet[0]["word_count"] = 201
        with self.assertRaisesRegex(RuntimeError, "length mismatch"):
            validate_decisions(packet, artifact)

    def test_evidence_must_be_complete_verbatim_and_support_affirmation(self):
        packet, artifact = fixture()
        for mutate in (
            lambda r: r["evidence"].pop("fact-0"),
            lambda r: r["evidence"]["fact-0"].update(quote="invented"),
            lambda r: r["evidence"]["fact-0"].update(note=""),
            lambda r: r["evidence"]["fact-0"].update(quote="<EMPTY_OUTPUT>"),
            lambda r: r["atomic_fact_labels"].update({"fact-0": "retained"}),
            lambda r: r.update(premise_adherence="yes"),
            lambda r: r.update(genre_voice="maybe"),
        ):
            bad = copy.deepcopy(artifact)
            mutate(bad["decisions"][0])
            with self.assertRaises(RuntimeError):
                validate_decisions(packet, bad)

    def test_empty_and_whitespace_output_evidence_distinct(self):
        packet, artifact = fixture()
        for text in ("", " \n "):
            packet[0].update(generated_text=text, generated_text_sha256=sha256_bytes(text.encode()),
                             word_count=0, length_compliant=False)
            row = artifact["decisions"][0]
            row["generated_text_sha256"] = packet[0]["generated_text_sha256"]
            for item in row["evidence"].values():
                item["quote"] = "<EMPTY_OUTPUT>" if not text else "<ABSENT>"
            self.assertEqual(len(validate_decisions(packet, artifact)), 80)
            if text:
                row["evidence"]["fact-0"]["quote"] = "<EMPTY_OUTPUT>"
                with self.assertRaisesRegex(RuntimeError, "nonempty"):
                    validate_decisions(packet, artifact)

    def test_uncertainty_and_length_fail_coherence_without_adjudication(self):
        packet, artifact = fixture()
        rows = validate_decisions(packet, artifact)
        for row in rows:
            row["atomic_fact_labels"] = {f"fact-{j}": "retained" for j in range(4)}
            row.update({f: "no" if f in ("contradiction_present", "loop_or_repetition_present")
                        else "yes" for f in HUMAN_FIELDS})
        rows[0]["genre_voice"] = "uncertain"
        rows[1]["length_compliant"] = False
        metadata = {r["review_id"]: {} for r in rows}
        summary = summarize(add_scores(rows, metadata))
        self.assertEqual(summary["coherent_outputs"]["numerator"], 78)
        self.assertEqual(summary["unresolved_outputs"], 1)
        self.assertEqual(summary["unresolved_judgments"], 1)
        self.assertEqual(summary["fact_retention"]["denominator"], 320)

    def test_bootstrap_preserves_matched_prompt_clusters_and_direction(self):
        packet, artifact = fixture()
        rows = validate_decisions(packet, artifact)[:40]
        metadata = {row["review_id"]: {"prompt_id": f"prompt-{i // 4}",
                    "generation_mode": "greedy" if i % 4 == 0 else "sampled",
                    "sampling_seed": None if i % 4 == 0 else i % 4} for i, row in enumerate(rows)}
        base = add_scores(rows, metadata)
        candidate = copy.deepcopy(base)
        for row in candidate:
            row["length_compliant"] = False
        result = paired_difference(base, candidate, replicates=100, seed=270028)
        self.assertEqual(result["prompt_clusters"], 10)
        self.assertEqual(result["matched_outputs"], 40)
        self.assertEqual(result["metrics"]["length_compliant"],
                         {"candidate_minus_base": -1, "lower": -1, "upper": -1})
        self.assertEqual(result, paired_difference(base, candidate, replicates=100, seed=270028))
        with self.assertRaisesRegex(RuntimeError, "coverage"):
            paired_difference(base, candidate[:-1], replicates=100, seed=270028)

    def test_scorer_refuses_overwrite_before_loading_inputs(self):
        with TemporaryDirectory() as directory:
            existing = Path(directory) / "report.json"
            existing.touch()
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                score(Path(directory) / "missing.json", existing)


if __name__ == "__main__":
    unittest.main()
