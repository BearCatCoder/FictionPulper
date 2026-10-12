# Corpus-v3 Source Pilot v1

Issue #21 runs a bounded, fail-closed measurement of the two highest-ranked
medium-confidence catalogs from issue #20: `pg-25918` and `pg-28`. This is a
source-yield pilot, not an approved Corpus-v3 build.

The pilot verifies immutable source bytes, reproduces the existing collection
extraction, measures tokenizer-v1 gross and exact-duplicate-adjusted candidate
tokens against sealed Corpus-v2, and records quality and fuzzy-duplicate
diagnostics. It does not infer that historical medium-confidence boundaries are
high confidence and does not invent attributable rights or boundary reviews.

## Reproduce

From the repository root with the existing environment and ignored local data:

```bash
source .venv/bin/activate
python -m src.report_corpus_v3_source_pilot \
  --config configs/corpus-v3-source-pilot-v1.yaml
python -m unittest tests.test_report_corpus_v3_source_pilot tests.test_corpus_v3
```

The generated `candidate-inventory.jsonl` contains source and derived-text
hashes, boundary locators, token counts, and dispositions, but no story text.
The machine-readable result is `source-yield-report.json`.

## Validation

- Implementation commit: `4d69713`.
- Focused suite: 21 tests passed.
- Full repository suite: 368 tests passed in 106.668 seconds.
- A second cache-resumed execution reproduced the report and inventory
  byte-for-byte.
- Report SHA-256: `616033fe00a5dcc5cc2c1eeb8b2a3d72bae21bfacfb0d29bd42a9f3c7680e6af`.
- Candidate inventory SHA-256:
  `d36d727d229c9e874219379f016813bc0fe193870a0a31ebc0630d112d64ab86`.

## Decision

The two pinned artifacts contain 174 extracted candidates and 204,805 gross
tokenizer-v1 document tokens. Exact-text comparison against sealed Corpus-v2
and across the pilot removes none, but that 204,805-token diagnostic is not
admissible yield. Only 35 candidates pass the downstream automated quality
diagnostic, 180,645 tokens are unclassified, creator and collection
concentration exceed the unchanged limits, and no attributable rights or
boundary reviews were completed.

Both selected catalogs therefore remain excluded. Final incremental unique
yield is zero records and zero tokens, no Corpus-v3 seal is written, and
Data50M/Data100M remain unsupported. The decision is to stop these catalogs and
prioritize the evidence recovery opportunity for the existing 1,134 Gutenberg
rows if future work is commissioned. These are measured results, not projected
supply.
