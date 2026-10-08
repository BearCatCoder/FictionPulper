# Quantitative Comparison

The 50M training artifacts retain the accidental raw run ID `smoke-5m-20261007-212756`; `fictionpulper-50m-data30m-v1` is the intended experiment and derived-report ID.

Both models use seed 1337 for fresh random initialization. This is the same initialization seed, not the same weights, because model shapes differ. Selection used Data30M validation loss only. Existing sealed test results were not rerun.

| Metric | 15M Compute5 | 50M Capacity |
|---|---:|---:|
| Parameters | 15047040 | 50348544 |
| Parameter multiplier | 1.0 | 3.3460763047084345 |
| Context | 1024 | 1024 |
| Target presentations | 135886355 | 135886355 |
| Presentations / parameter | 9.030769839117859 | 2.698913299260451 |
| Selected epoch | 5 | 5 |
| Selected step | 2220 | 2220 |
| Best validation loss | 3.1456916993451474 | 3.007466330171598 |
| Data30M test loss | 3.223102356171976 | 3.101755504562244 |
| Data30M test PPL | 25.10588689210002 | 22.236954110994716 |
| Data30M test accuracy | 0.3637051897664537 | 0.3825981094149738 |
| Common held-out 95-doc loss | 3.2246202187302235 | 3.1131359159633174 |
| Common held-out 95-doc PPL | 25.144023113254384 | 22.491465271052853 |
| Common held-out 95-doc accuracy | 0.36015224211523184 | 0.37756224108292974 |
| Legacy clean-named validation loss (contaminated) | 2.9990317794590324 | 2.741584138020266 |
| Legacy test loss (contaminated) | 3.0850056576449196 | 2.842833104847855 |
| Runtime seconds | 422.8691877780075 | 1234.4846752800077 |
| Valid targets / second | 321343.7132036584 | 110075.36806334033 |
| Peak VRAM GiB | 3.9406185150146484 | 9.30289077758789 |

Legacy diagnostics are labeled historical continuity measurements and are partially contaminated by records seen in Data30M training.
