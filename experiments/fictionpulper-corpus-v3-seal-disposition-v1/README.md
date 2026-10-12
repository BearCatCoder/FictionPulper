# Corpus-v3 Seal Disposition v1

Issue #24 evaluated the completed issue #22 base-corpus disposition and the
completed issue #21 bounded source pilot. It stopped without sealing or packing
because the evaluated supply contains zero approved records and zero approved
tokenizer-v1 document tokens.

## Decision

- Corrected base candidates: 4,362 records; zero approved records and tokens.
- Bounded source pilot: 174 extracted records and 204,805 gross diagnostic
  tokens; zero approved incremental records and tokens.
- Exact approved incremental content versus Data30M: zero records and tokens.
- Data50M/Data100M naming and execution: unsupported.
- Final decision: `no_go_data_scaling`.

The zero total is exact for the evaluated base and two-source pilot. It is not
an upper bound on every possible external source. A future acquisition decision
may reopen corpus recovery only with a nonempty approved inventory that passes
every corrected gate.

## Stop Enforcement

No canonical membership or approved training split was frozen, no canonical
Corpus-v3 seal was created, and no binary or index was written at the configured
Data100M pack paths. The corrected audit's zero-member diagnostic split is not a
training split. Real-data Data100M preflight was not run: its required corpus,
split, packed-metadata, binary, index, and schedule locks cannot exist for a
zero-record corpus. The prepared Data100M config remains blocked and its three
generated data hashes remain unset.

There is also no new candidate training membership against which to audit a
common test split. Any future training proposal must perform that audit against
both historical baseline training memberships and the new candidate membership
before claiming a clean common test.

## Reproduce

From the repository root with the existing environment and ignored local data:

```bash
source .venv/bin/activate
python -m src.report_corpus_v3_seal_disposition \
  --config configs/corpus-v3-seal-disposition-v1.yaml
python -m unittest tests.test_report_corpus_v3_seal_disposition
```

The command verifies every input hash, reconciles the issue #21 and #22 counts,
checks that seal and Data100M pack paths are absent, verifies the blocked
Data100M config, and deterministically rewrites `disposition.json`. Any changed
evidence, admitted supply, authorized seal, or packed artifact fails closed.

## Validation

- Focused corpus, tokenizer, dataset, report, and preflight suite: 44 tests
  passed in 1.297 seconds.
- Full repository suite: 375 tests passed in 107.226 seconds.
- Disposition SHA-256:
  `8c142ae784eb028b8d6236940378ee9eed7819515d76d7fd5cbd0fd0c8455e93`.
