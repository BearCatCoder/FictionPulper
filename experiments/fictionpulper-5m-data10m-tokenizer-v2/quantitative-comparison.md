# Three-Way Quantitative Comparison

| Metric | Smoke v1 | Data10M v1 | Tokenizer v2 |
|---|---:|---:|---:|
| Parameter count | 5426432 | 5426432 | 5426432 |
| Unique corpus stories | 710 | 1844 | 1844 |
| Unique corpus words | 1162772 | 5830062 | 5830062 |
| Tokenizer SHA-256 | `14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012` | `14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012` | `108677dabadfb2888d4f8247b9377f5c074559414421b0245bd5896a9527b3c9` |
| Training tokens/word | 1.495460 | 1.707822 | 1.661748 |
| Packed train targets | 1483194 | 8779154 | 8542264 |
| Best validation loss | 4.092338 | 3.494505 | 3.495263 |
| Corpus-v1 test loss | 4.935219 | 3.565010 | 3.615572 |
| Corpus-v1 test perplexity | 139.103598 | 35.339789 | 37.172599 |
| Corpus-v1 test accuracy | 0.194690 | 0.314667 | 0.307648 |
| Corpus-v1 test bits/byte | 2.176071 | 1.571909 | 1.567540 |
| Legacy clean validation loss | 4.090146 | 3.566082 | 3.531735 |
| Legacy clean validation bits/byte | 1.620673 | 1.413019 | 1.415718 |
| Legacy test loss | 4.195695 | 3.623500 | 3.603795 |
| Legacy test perplexity | 66.399866 | 37.468461 | 36.737398 |
| Legacy test accuracy | 0.238098 | 0.300748 | 0.303313 |
| Legacy test bits/byte | 1.691610 | 1.460914 | 1.460135 |
| Training runtime seconds | 35.411171 | 127.453326 | 123.261802 |
| Valid targets/second | 628273.77 | 688813.25 | 693017.94 |
| Peak VRAM GiB | 2.341180 | 2.340203 | 2.339227 |

Smoke Corpus-v1 test and smoke leakage-clean validation are post hoc evaluations and did not affect historical checkpoint selection. Raw token-level loss, perplexity, and accuracy are segmentation-dependent; bits per source byte is the fairer v1-versus-v2 comparison.
