# Controlled Comparison

Compute5 and the candidate use the same architecture, tokenizer, seed, context, optimizer schedule, selected step, and total valid-target exposure. The controlled variable is replacement of 15% of targets with Narrative Continuity Curriculum v1.

| Metric | 15M Compute5 | 15M Narrative | Candidate change |
|---|---:|---:|---:|
| Parameters | 15047040 | 15047040 | 0 |
| Context tokens | 1024 | 1024 | 0 |
| Seed | 1337 | 1337 | 0 |
| Valid target presentations | 135886355 | 135886355 | 0 |
| Real / curriculum target mix | 100% / 0% | 85% / 15% | controlled variable |
| Selected step | 2220 | 2220 | 0 |
| Data30M validation loss | 3.1456916993451474 | 3.1482500308740273 | +0.0813% |
| Data30M test loss | 3.223102356171976 | 3.220216419984231 | -0.0895% |
| Data30M test perplexity | 25.10588689210002 | 25.033537352883737 | -0.2882% |
| Data30M test next-token accuracy | 0.3637051897664537 | 0.3643672371868933 | +0.1820% |
| Narrative facts retained, greedy | 0/13 | 0/13 | 0 |
| Narrative facts retained, sampled | 0/13 | 0/13 | 0 |
| Training seconds | 422.8691877780075 | 395.27206267598376 | -6.53% |
| Valid targets / second | 321343.7132036584 | 343779.305018554 | +6.98% |
| Peak VRAM GiB | 3.9406185150146484 | 3.9406185150146484 | +0.00% |

Checkpoint selection used Data30M validation loss only. All test, generation, repetition, and state diagnostics were post-selection.
