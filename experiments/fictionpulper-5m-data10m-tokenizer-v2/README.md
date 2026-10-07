# FictionPulper-5M-data10M-tokenizerV2

This controlled experiment changes only the 4,096-token byte-level BPE tokenizer. It uses the exact sealed Data10M-v1 corpus and split manifest, the same 5,426,432-parameter architecture, seed 1337, optimization policy, context length, and ten document/chunk epochs.

Tokenizer v2 was learned only from the 1,565 Data10M training documents. Validation and test documents did not participate in vocabulary or merge learning. It has no Unicode normalizer and preserves the original IDs and semantics of all 14 special/control tokens.

- Original tokenizer SHA-256: `14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012`
- Tokenizer v2 SHA-256: `108677dabadfb2888d4f8247b9377f5c074559414421b0245bd5896a9527b3c9`
- Locked corpus SHA-256: `7cd6323095a33336e4a002d96554aebd9deb26b5a6971f3d2044e67dfb035f5b`
- Locked split SHA-256: `01e708c7454a044539921bc179ff1b9b3f31aba616d90a1b22b6ea3e628a1229`
- Derived schedule: 143 steps/epoch, 1,430 total steps, 72 warmup steps

See `tokenizer-comparison.md` for split-level efficiency and representative passage comparisons. `preflight.json` records the locked inputs and packed artifact hashes.
