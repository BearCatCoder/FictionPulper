"""Evidence-boundary tests for the authorized AI-only evaluation."""

import copy
import unittest

from src.narrative_v2_ai_review import EVALUATOR, VERSION, validate_ai_decisions
from src.narrative_v2_owner_review import OwnerReviewError


class AIReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.packet = [{"owner_review_id": "row-1", "generated_text": " unrelated",
                        "required_propositions": ["required fact"]}]
        self.decisions = {
            "version": VERSION, "evaluator": EVALUATOR, "reviewer_role": "ai",
            "genuine_human_attestation": False, "independent_review_confirmed": False,
            "packet_sha256": "packet", "manifest_sha256": "manifest",
            "rows": [{"row_number": 1, "label": "inconsistent", "confidence": 5,
                      "evidence_type": "required_proposition_absent", "evidence_quote": "",
                      "note": "The required fact is not expressed."}],
        }

    def validate(self, decisions=None):
        return validate_ai_decisions(self.packet, decisions or self.decisions,
                                     packet_hash="packet", manifest_hash="manifest")

    def test_ai_omission_and_whitespace_are_valid(self) -> None:
        self.assertEqual(self.validate(), {"row-1": "inconsistent"})
        self.packet[0]["generated_text"] = "\n  "
        self.assertEqual(self.validate(), {"row-1": "inconsistent"})

    def test_cannot_claim_human_identity_or_independence(self) -> None:
        for field, value in (("evaluator", "BearCatCoder"), ("reviewer_role", "owner"),
                             ("genuine_human_attestation", True),
                             ("independent_review_confirmed", True)):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.decisions)
                changed[field] = value
                with self.assertRaises(OwnerReviewError):
                    self.validate(changed)

    def test_rejects_incomplete_duplicate_and_unbound_decisions(self) -> None:
        for change in ("missing", "duplicate", "packet", "manifest"):
            with self.subTest(change=change):
                changed = copy.deepcopy(self.decisions)
                if change == "missing":
                    changed["rows"] = []
                elif change == "duplicate":
                    changed["rows"] *= 2
                else:
                    changed[f"{change}_sha256"] = "tampered"
                with self.assertRaises(OwnerReviewError):
                    self.validate(changed)

    def test_omission_cannot_support_consistent_or_invent_quote(self) -> None:
        for field, value in (("label", "consistent"), ("evidence_quote", "invented"),
                             ("confidence", True), ("row_number", True)):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.decisions)
                changed["rows"][0][field] = value
                with self.assertRaises(OwnerReviewError):
                    self.validate(changed)

    def test_zero_byte_output_is_not_whitespace(self) -> None:
        self.decisions["rows"][0].update(evidence_type="empty_output", evidence_quote="<EMPTY_OUTPUT>")
        with self.assertRaises(OwnerReviewError):
            self.validate()
        self.packet[0]["generated_text"] = ""
        self.assertEqual(self.validate(), {"row-1": "inconsistent"})

    def test_supporting_quote_must_be_verbatim(self) -> None:
        self.decisions["rows"][0].update(label="uncertain", evidence_type="verbatim_quote",
                                         evidence_quote="invented")
        with self.assertRaises(OwnerReviewError):
            self.validate()
        self.decisions["rows"][0]["evidence_quote"] = "unrelated"
        self.assertEqual(self.validate(), {"row-1": "uncertain"})


if __name__ == "__main__":
    unittest.main()
