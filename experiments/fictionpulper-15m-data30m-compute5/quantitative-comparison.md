# Quantitative Comparison

| Metric | Data30M 3-Epoch | Data30M Compute5 |
|---|---:|---:|
| Parameters | 15047040 | 15047040 |
| Unique train targets | 27177271 | 27177271 |
| Context | 1024 | 1024 |
| Tokenizer | v1 | v1 |
| Epochs | 3 | 5 |
| Target presentations | 81531813 | 135886355 |
| Best epoch | 3 | 5 |
| Best validation loss | 3.2460981563125317 | 3.1456916993451474 |
| Data30M test loss | 3.322380529413267 | 3.223102356171976 |
| Data30M test PPL | 27.726275273331407 | 25.10588689210002 |
| Data30M test accuracy | 0.3489222656762679 | 0.3637051897664537 |
| Common held-out loss | 3.3169595081613577 | 3.2246202187302235 |
| Common held-out PPL | 27.57637721321671 | 25.144023113254384 |
| Common held-out accuracy | 0.3464091596612254 | 0.36015224211523184 |
| Legacy clean validation loss | 3.2052978249618653 | 2.9990317794590324 |
| Legacy test loss | 3.2799965212742994 | 3.0850056576449196 |
| Runtime seconds | 248.1865291610011 | 422.8691877780075 |
| Valid targets/second | 328510.22686694446 | 321343.7132036584 |
| Peak VRAM GiB | 3.9406185150146484 | 3.9406185150146484 |

Both runs use identical data, tokenizer, architecture, initialization seed, and evaluation definitions. The five-epoch run uses a fresh cosine schedule and does not resume three-epoch weights.
