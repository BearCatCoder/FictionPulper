# FictionPulper-5M-data10M-v1

This controlled experiment changed unique fiction data while holding the 5,426,432-parameter architecture, 4,096-token smoke tokenizer, 1,024-token context, AdamW optimizer family, BF16 precision, and generation suite fixed.

## Data

- Resolved corpus stories: 1,844
- Train/validation/test stories: 1,565 / 139 / 140
- Total packed tokens: 9,906,449
- Train packed tokens: 8,780,719
- Train valid targets per epoch: 8,779,154
- Train chunks: 9,381
- Train padding: 826,990 slots (8.61%)
- Original tokenizer SHA-256: `14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012`
- Resolved corpus SHA-256: `7cd6323095a33336e4a002d96554aebd9deb26b5a6971f3d2044e67dfb035f5b`
- Split manifest SHA-256: `01e708c7454a044539921bc179ff1b9b3f31aba616d90a1b22b6ea3e628a1229`

All 31 reported near-duplicate pairs were reviewed. They comprised 22 formatting variants and 9 alternate editions; one record per pair was excluded before splitting. New Gutenberg records were grouped by source collection, with no collection crossing splits. Retained seed assignments were unchanged. The untouched original packed validation and test sets remained the legacy benchmark.

The final seed-cluster audit found one historical train/validation crossing: smoke-train `932510eaad15` and smoke-validation `162495d6f1c1` are alternate editions of the same story. Only `932510eaad15` remains in Data10M train, so the Data10M split is clean. The untouched legacy validation metric is flagged as containing known contamination. A separate leakage-clean legacy validation evaluation excludes `162495d6f1c1` and reports loss 3.566082, perplexity 35.3777, and accuracy 0.308451. The second seed pair, `e89be87f3f58` / `3650b1f0a3ad`, was train/train and does not contaminate evaluation.

`seal.json` records the final corpus, split, duplicate-resolution, packing, tokenizer, checkpoint, and run-artifact hashes. This experiment is closed; subsequent controlled experiments must not modify or rerun it.

## Schedule

- Epochs: 10 complete document/chunk passes
- Optimizer steps per epoch: 147
- Total optimizer steps: 1,470
- Valid training targets per chunk epoch: 8,779,154
- Total valid target presentations: 87,791,540
- Optimizer slot capacity per epoch: 9,633,792 (27,648 unused final-step slots)
- Warmup steps: 74
- Peak/minimum learning rate: `1e-3` / `1e-4`
- Batch size / accumulation: 16 / 4

## Results

| Metric | Smoke v1 | Data10M v1 |
|---|---:|---:|
| Parameters | 5,426,432 | 5,426,432 |
| Train packed tokens | 1,483,798 | 8,780,719 |
| Train stories | 604 | 1,565 |
| Best validation loss | 4.092338 | 3.494505 |
| Validation perplexity | 59.8797 | 32.9340 |
| Legacy validation loss | 4.092338 | 3.567081 |
| Legacy test loss | 4.195695 | 3.623500 |
| Legacy test perplexity | 66.3999 | 37.4685 |
| Legacy test accuracy | 0.238098 | 0.300748 |
| Peak VRAM GiB | 2.3412 | 2.3402 |
| Valid tokens/sec | 628,274 | 688,813 |
| Training runtime sec | 35.41 | 127.45 |

Corpus-v1 test loss was 3.565010, perplexity was 35.3398, and next-token accuracy was 0.314667. Validation improved in every epoch, so epoch 10 was selected solely by Corpus-v1 validation loss. Corpus-v1 test and legacy seed test were evaluated once after selection.

Qualitatively, sampled outputs have better local grammar and scene continuity than Smoke v1, but prompt adherence, subject persistence, character consistency, and greedy repetition remain weak.

Large artifacts remain ignored under `runs/`, `checkpoints/`, `data/corpus_v1/`, and `data/packed/`.
