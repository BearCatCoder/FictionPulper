# 5M versus 15M Quantitative Comparison

| Metric | 5M Data10M | 15M Data10M | Relative change |
|---|---:|---:|---:|
| Parameters | 5,426,432 | 15,047,040 | +177.292% |
| Training stories | 1,565 | 1,565 | 0% |
| Train targets | 8,779,154 | 8,779,154 | 0% |
| Context | 1,024 | 1,024 | 0% |
| Tokenizer SHA-256 | `14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012` | same | same |
| Optimizer steps | 1,470 | 1,470 | 0% |
| Best validation loss | 3.494505 | 3.403524 | -2.604% |
| Corpus-v1 test loss | 3.565010 | 3.484673 | -2.253% |
| Corpus-v1 test PPL | 35.339789 | 32.611754 | -7.719% |
| Corpus-v1 test accuracy | 0.314667 | 0.323956 | +2.952% |
| Legacy clean validation loss | 3.566082 | 3.465931 | -2.808% |
| Legacy test loss | 3.623500 | 3.527729 | -2.643% |
| Legacy test PPL | 37.468461 | 34.046555 | -9.133% |
| Legacy test accuracy | 0.300748 | 0.312272 | +3.832% |
| Runtime seconds | 127.453 | 278.664 | +118.640% |
| Valid targets/second | 688,813 | 315,044 | -54.263% |
| Peak VRAM GiB | 2.340 | 3.941 | +68.388% |

Direct token-level comparison is valid because both runs use identical source tokens, tokenizer-v1, context length, packed chunks, and evaluation sets.
