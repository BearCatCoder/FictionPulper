# FictionPulper-5M-data10M-tokenizerV2

This controlled experiment changes only the 4,096-token byte-level BPE tokenizer. It uses the exact sealed Data10M-v1 corpus and split manifest, the same 5,426,432-parameter architecture, seed 1337, optimization policy, context length, and ten document/chunk epochs.

Tokenizer v2 was learned only from the 1,565 Data10M training documents. Validation and test documents did not participate in vocabulary or merge learning. It has no Unicode normalizer and preserves the original IDs and semantics of all 14 special/control tokens.

- Original tokenizer SHA-256: `14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012`
- Tokenizer v2 SHA-256: `108677dabadfb2888d4f8247b9377f5c074559414421b0245bd5896a9527b3c9`
- Locked corpus SHA-256: `7cd6323095a33336e4a002d96554aebd9deb26b5a6971f3d2044e67dfb035f5b`
- Locked split SHA-256: `01e708c7454a044539921bc179ff1b9b3f31aba616d90a1b22b6ea3e628a1229`
- Derived schedule: 143 steps/epoch, 1,430 total steps, 72 warmup steps

See `tokenizer-comparison.md` for split-level efficiency and representative passage comparisons. `preflight.json` records the locked inputs and packed artifact hashes.

## Results

The run completed 1,430 optimizer steps from fresh random weights and selected epoch 10. Tokenizer v2 reduced Data10M train tokenization from 8,780,719 to 8,543,829 tokens and training runtime from 127.45 to 123.26 seconds.

Raw token-level Corpus-v1 test loss increased from 3.565010 to 3.615572, but losses across tokenizers are not directly comparable. Normalized by identical source bytes, Corpus-v1 test cross-entropy improved slightly from 1.571909 to 1.567540 bits/byte (0.28%). Legacy test improved from 1.460914 to 1.460135 bits/byte, while leakage-clean legacy validation worsened from 1.413019 to 1.415718 bits/byte.

The generation comparison shows no consistent qualitative gain. Tokenizer v2 remains weak on prompt adherence, semantic continuity, repetition, entity consistency, scene persistence, and narrative progression. The experiment therefore finds a modest efficiency improvement and at most a marginal modeling improvement, not a decisive quality improvement.
