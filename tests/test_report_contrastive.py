import json
import unittest
from pathlib import Path

from src.report_contrastive import MANUAL_AUDIT, percent_change, validate_manual_audit


class ContrastiveReportTests(unittest.TestCase):
    def test_percent_change_direction(self):
        self.assertAlmostEqual(percent_change(9.0, 10.0), -10.0)

    def test_manual_record_exactly_matches_protocol_and_scores_all_false(self):
        protocol = json.loads(Path("experiments/fictionpulper-15m-data30m-contrastive-v1/post-selection-protocol.json").read_text(encoding="utf-8"))
        audit = json.loads(MANUAL_AUDIT.read_text(encoding="utf-8"))
        validate_manual_audit(audit, protocol)
        self.assertEqual(len(audit["facts"]), 13)

    def test_generated_summary_is_unsealed_and_classified(self):
        path = Path("experiments/fictionpulper-15m-data30m-contrastive-v1/summary.json")
        if not path.exists():
            self.skipTest("Run reporter before generated-record assertions")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["classification"], "OBJECTIVE_LEARNED_WITHOUT_NARRATIVE_TRANSFER")
        self.assertFalse(summary["seal_created"])
        for model in ("compute5", "narrative_v1", "contrastive_v1"):
            for mode in ("greedy", "sampled"):
                self.assertEqual(summary["manual_narrative_fact_retention"]["totals"][model][mode], {"correct": 0, "total": 13})


if __name__ == "__main__":
    unittest.main()
