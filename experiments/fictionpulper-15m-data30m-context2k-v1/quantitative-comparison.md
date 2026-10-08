# Quantitative Comparison

| Metric | 15M / 1024 / Compute5 | 15M / 2048 |
|---|---:|---:|
| Parameters | 15047040 | 15047040 |
| Context | 1024 | 2048 |
| Tokenizer | v1 | v1 |
| Corpus | Data30M | Data30M |
| Unique train targets | 27177271 | 27177271 |
| Microbatch | 16 | 16 |
| Gradient accumulation | 4 | 2 |
| Allocated slots/step | 65536 | 65536 |
| Target presentations | 135886355 | 135886355 |
| Best epoch | 5 | 5 |
| Best validation loss | 3.1456916993451474 | 3.141572354898927 |
| Data30M test loss | 3.223102356171976 | 3.2157354660811417 |
| Data30M test PPL | 25.10588689210002 | 24.921614174550506 |
| Data30M test accuracy | 0.3637051897664537 | 0.3639844910219517 |
| Common held-out loss | 3.2246202187302235 | 3.2355170630697296 |
| Common held-out PPL | 25.144023113254384 | 25.419511872266778 |
| Common held-out accuracy | 0.36015224211523184 | 0.35827838942922674 |
| Legacy clean validation loss | 2.9990317794590324 | 3.028296308087274 |
| Legacy test loss | 3.0850056576449196 | 3.1134872527509714 |
| Runtime seconds | 422.8691877780075 | 564.609710828001 |
| Valid targets/second | 321343.7132036584 | 240673.07450437304 |
| Peak VRAM GiB | 3.9406185150146484 | 7.512426376342773 |

Source token streams and valid targets are identical. Chunk counts, allocated slots, and document-tail padding are packing-dependent.
