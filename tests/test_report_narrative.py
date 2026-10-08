import json
import unittest
from pathlib import Path

from src.report_narrative import markdown_table, percent_change, validate_sources


class NarrativeReportTests(unittest.TestCase):
    def test_markdown_table_preserves_requested_columns(self):
        table = markdown_table(("Metric", "Baseline", "Candidate", "Change"), [("loss", 3.2, 3.1, "-3%")])
        self.assertEqual(table.splitlines()[0], "| Metric | Baseline | Candidate | Change |")

    def test_percent_change_direction(self):
        self.assertAlmostEqual(percent_change(9, 10), -10.0)

    def test_validation_rejects_nonzero_exact_recall(self):
        summary = {
            "status": "training_and_selection_complete", "selection_metric": "Data30M validation loss only",
            "curriculum_validation_used_for_selection": False, "best_checkpoint_step": 2220,
            "total_valid_targets": 10, "best_checkpoint_sha256": "candidate",
        }
        manifest = {"model_parameter_count": 15_047_040, "seed": 1337, "fresh_random_initialization": True}
        baseline = {"best_checkpoint_epoch": 5, "total_valid_training_targets_processed": 10, "model_parameter_count": 15_047_040, "best_checkpoint_sha256": "baseline"}
        evaluation = {"post_checkpoint_selection": True, "used_for_checkpoint_selection": False, "checkpoint_sha256": "candidate"}
        recall = {}
        for split, total in (("curriculum_test", 1405), ("curriculum_generalization_holdout", 618)):
            recall[split] = {"aggregate": {"correct": 0, "total": total, "accuracy": 0.0}}
            recall[split].update({name: {"x": {"correct": 0, "total": total, "accuracy": 0.0}} for name in ("by_state_type", "by_distance_bucket", "by_difficulty", "by_genre")})
        recall["curriculum_test"]["aggregate"]["correct"] = 1
        generation = {"baseline_checkpoint_sha256": "baseline", "candidate_checkpoint_sha256": "candidate", "prompts": list(range(10))}
        narrative = {"post_checkpoint_selection": True, "used_for_checkpoint_selection": False}
        with self.assertRaisesRegex(RuntimeError, "Unexpected recall"):
            validate_sources(summary, manifest, baseline, evaluation, recall, generation, narrative)

    def test_generated_summary_records_failed_hypothesis_as_unsealed(self):
        experiment_dir = Path("experiments/fictionpulper-15m-data30m-narrative-v1")
        summary = json.loads((experiment_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["seal_created"])
        self.assertIsNone(summary["completion_commit"])
        self.assertIn("failed", summary["hypothesis_result"])
        for model in summary["narrative_semantic_diagnostic"].values():
            if isinstance(model, dict):
                self.assertTrue(all(score["correct"] == 0 for score in model.values()))


if __name__ == "__main__":
    unittest.main()
