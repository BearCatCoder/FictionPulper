# FictionPulper Corpus v2 30M

This record freezes the acquisition-only Corpus-v2 artifact used as the input to the controlled Data30M experiment.

- 4,362 stories
- 17,471,134 words
- 30,004,706 exact tokenizer-v1 document tokens
- Corpus SHA-256: `0fb5e5080b3c4ee26ea178201d349d6947052f664f2bc415b5a1b9219d496087`
- Tokenizer-v1 SHA-256: `14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012`
- Build commit: `54153002897e57aa1a4e0f9e95697356b3fc8860`

`freeze-report.json` contains compact corpus, audit, concentration, hash, and resume-replay evidence. `seed-correctness-exclusions.json` explicitly records the three duplicate train records removed from the 1,844-record historical Data10M seed. Historical experiment artifacts and metrics were not modified.

The large canonical corpus, raw caches, processed-source shards, and stage reports remain under ignored data directories. No Corpus-v2 split or packed dataset is part of this corpus seal.
