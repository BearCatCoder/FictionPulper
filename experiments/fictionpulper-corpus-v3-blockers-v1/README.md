# Corpus-v3 Integrity Blockers v1

Issue #22 corrected verified fail-open audit defects and reran the base-only
candidate set without changing sealed Corpus-v2 or the historical issue #20
report. The result is intentionally unsealed.

## Result

- Candidates: 4,362.
- Accepted records and tokens: 0.
- First-reason exclusions: 1,841 `invalid_schema` and 2,521
  `incomplete_rights_evidence`.
- Review: no approvals synthesized; final membership and attributable human
  decisions remain pending.
- Genre: the historical retained set was 57.0408% unclassified; the corrected
  empty set is not allowed to pass the gate.
- Seal: absent; training remains unauthorized.

The corrected builder validates real review dates, controlled genre labels,
content-hash-locked attributable sample decisions, assignment coverage, collection and
fuzzy-cluster isolation, positive token counts, split totals, and tokenizer-v1
recomputed totals. The complete before/after gates, dispositions, safeguards,
and seal conditions are in [`blocker-report.json`](blocker-report.json).

## Reproduce

From the repository root with the existing environment and ignored local data:

```bash
source .venv/bin/activate
python -m src.corpus_v3 --config configs/corpus-v3.yaml
python -m src.report_corpus_v3_blockers \
  --config configs/corpus-v3-blockers-v1.yaml
python -m unittest tests.test_corpus_v3 tests.test_report_corpus_v3_blockers
```

The report config hash-locks the corrected builder, configs, tokenizer, sealed
Corpus-v2 input, historical shortfall report, and corrected local outputs. Any
change fails report generation instead of silently changing the disposition.

## Handoff

Issue #21 must supply immutable source bytes and attributable rights, boundary,
creator, genre, and publication-year evidence, then freeze final membership and
complete genuine deterministic review. Issue #24 remains blocked until a
nonempty corpus passes every required gate; closing issue #22 does not authorize
sealing, packing, or training.
