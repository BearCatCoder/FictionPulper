# FictionPulper-15M Data30M Counterfactual-v1

Final reporting classification: **`COUNTERFACTUAL_SCORING_TRANSFER_WITHOUT_BROAD_NARRATIVE_TRANSFER`**.

Counterfactually balanced supervision produced strong held-out reversal within the synthetic generator, but failed latest-state transfer and retained 0/13 historical facts in both greedy and sampled free generation. Frozen probes show no broad two-split representation improvement. Primary real-fiction LM behavior is near-neutral, while greedy repetition regressed.

Rebuild and validate:

```bash
python -m src.counterfactual_post_selection --stage finalize
python -m src.report_counterfactual
python -m unittest discover -s tests
```

The immutable trainer start manifest remains `training_started`; completion is independently established by the hash-locked final summary and selected step-2220 checkpoint and is disclosed in `provenance.json`.

Key records: `summary.json`, `controlled-summary.md`, `quantitative-comparison.md`, `qualitative-assessment.md`, `repetition-comparison.md`, `manual-fact-scoring.json`, `provenance.json`, and `seal-candidate.json`.
