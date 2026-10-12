# Corpus-v3 Shortfall Diagnosis v1

This historical issue #20 artifact is preserved unchanged. Issue #22 later
invalidated its rights and token-accounting pass claims; see
[`fictionpulper-corpus-v3-blockers-v1`](../fictionpulper-corpus-v3-blockers-v1/README.md).

This versioned report diagnoses the base-only Corpus-v3 feasibility build. It
does not modify Corpus-v2 or Corpus-v3 data, admit new text, complete pending
human review, authorize training, or seal a corpus.

## Historical Reproduction

At the issue #20 result commit (`d54cfb4`), run:

```bash
.venv/bin/python -m src.report_corpus_v3_shortfall \
  --config configs/corpus-v3-shortfall-v1.yaml
```

The config hash-locks every ignored local evidence artifact consumed by the
report. On the corrected issue #22 tree, the historical command intentionally
fails its input hashes rather than reinterpret the old report under new gates.
At its frozen commit, it fails if an input changes, if retained and rejected
records do not partition all candidates, if recomputed tokenizer-v1 totals do
not reconcile, or if recomputed first-rejection reasons differ from the
committed feasibility artifacts.

The generated `shortfall-report.json` contains no story text. Measured current
supply, conditional evidence-repair opportunity, and unmeasured external-source
hypotheses are reported separately.
