# FictionPulper-15M Data30M Contrastive-v1

Final reporting classification: **`OBJECTIVE_LEARNED_WITHOUT_NARRATIVE_TRANSFER`**.

The run learned the synthetic contrastive ranking task (97.65% test, 95.17% generalization) but retained 0/13 disclosed narrative facts under both greedy and sampled decoding. Frozen probes show a narrow object-location improvement, not broad representation change. Primary real-fiction LM metrics are near-neutral; historical diagnostics include regressions. No final seal, tag, or commit is created by this report.

Rebuild and validate:

```bash
python -m src.report_contrastive
python -m unittest discover -s tests
```

Key records: `summary.json`, `controlled-summary.md`, `quantitative-comparison.md`, `qualitative-assessment.md`, `repetition-comparison.md`, `manual-fact-scoring.json`, `provenance.json`, and `seal-candidate.json`.
