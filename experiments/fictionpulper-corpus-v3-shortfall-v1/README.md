# Corpus-v3 Shortfall Diagnosis v1

This versioned report diagnoses the base-only Corpus-v3 feasibility build. It
does not modify Corpus-v2 or Corpus-v3 data, admit new text, complete pending
human review, authorize training, or seal a corpus.

## Reproduce

From the repository root with the existing environment:

```bash
.venv/bin/python -m src.report_corpus_v3_shortfall \
  --config configs/corpus-v3-shortfall-v1.yaml
```

The config hash-locks every ignored local evidence artifact consumed by the
report. The command fails if an input changes, if the retained and rejected
records do not partition all candidates, if recomputed tokenizer-v1 totals do
not reconcile, or if recomputed first-rejection reasons differ from the
committed feasibility artifacts.

The generated `shortfall-report.json` contains no story text. Measured current
supply, conditional evidence-repair opportunity, and unmeasured external-source
hypotheses are reported separately.
