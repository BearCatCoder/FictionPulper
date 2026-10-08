# Teacher Forced vs Rollout

| Model | Mode | Length | Top-1 | Mean margin | Mean first divergence | Explicit contradictions | Recovery max |
|---|---|---|---|---|---|---|---|
| Compute5 | greedy | 0 | 4/13 | -0.5349 | None | 0/13 | 0.0 |
| Compute5 | greedy | 32 | 4/13 | -0.2663 | 0.6153846153846154 | 0/13 | 0.0 |
| Compute5 | greedy | 64 | 3/13 | -0.4410 | 1.0769230769230769 | 0/13 | 0.0 |
| Compute5 | greedy | 128 | 4/13 | -0.4451 | 0.46153846153846156 | 0/13 | 0.0 |
| Compute5 | greedy | 256 | 5/13 | -0.3383 | 0.07692307692307693 | 0/13 | 0.0 |
| Compute5 | sampled | 0 | 4/13 | -0.5349 | None | 0/13 | 0.0 |
| Compute5 | sampled | 32 | 4/13 | -0.4019 | 0.38461538461538464 | 0/13 | 0.0 |
| Compute5 | sampled | 64 | 6/13 | -0.1460 | 0.5384615384615384 | 0/13 | 0.0 |
| Compute5 | sampled | 128 | 4/13 | -0.5602 | 0.23076923076923078 | 0/13 | 0.0 |
| Compute5 | sampled | 256 | 5/13 | -0.2990 | 0.07692307692307693 | 0/13 | 0.0 |
| Narrative | greedy | 0 | 3/13 | -0.5611 | None | 0/13 | 0.0 |
| Narrative | greedy | 32 | 4/13 | -0.5064 | 0.38461538461538464 | 0/13 | 0.0 |
| Narrative | greedy | 64 | 1/13 | -0.6060 | 1.0769230769230769 | 0/13 | 0.0 |
| Narrative | greedy | 128 | 6/13 | -0.3756 | 0.38461538461538464 | 0/13 | 0.0 |
| Narrative | greedy | 256 | 2/13 | -0.5733 | 0.23076923076923078 | 0/13 | 0.0 |
| Narrative | sampled | 0 | 3/13 | -0.5611 | None | 0/13 | 0.0 |
| Narrative | sampled | 32 | 4/13 | -0.4482 | 0.15384615384615385 | 0/13 | 0.0 |
| Narrative | sampled | 64 | 3/13 | -0.5224 | 0.23076923076923078 | 0/13 | 0.0 |
| Narrative | sampled | 128 | 3/13 | -0.5958 | 0.15384615384615385 | 0/13 | 0.0 |
| Narrative | sampled | 256 | 4/13 | -0.4414 | 0.23076923076923078 | 0/13 | 0.0 |
| Contrastive | greedy | 0 | 4/13 | -0.2288 | None | 0/13 | 0.0 |
| Contrastive | greedy | 32 | 4/13 | -0.0866 | 0.6153846153846154 | 0/13 | 0.0 |
| Contrastive | greedy | 64 | 5/13 | 0.0361 | 1.0 | 0/13 | 0.0 |
| Contrastive | greedy | 128 | 7/13 | -0.0880 | 0.23076923076923078 | 0/13 | 0.0 |
| Contrastive | greedy | 256 | 6/13 | -0.0567 | 0.15384615384615385 | 0/13 | 0.0 |
| Contrastive | sampled | 0 | 4/13 | -0.2288 | None | 0/13 | 0.0 |
| Contrastive | sampled | 32 | 5/13 | -0.1373 | 0.46153846153846156 | 0/13 | 0.0 |
| Contrastive | sampled | 64 | 3/13 | -0.1321 | 0.46153846153846156 | 0/13 | 0.0 |
| Contrastive | sampled | 128 | 5/13 | -0.0649 | 0.23076923076923078 | 0/13 | 0.0 |
| Contrastive | sampled | 256 | 3/13 | -0.2674 | 0.07692307692307693 | 0/13 | 0.0 |
| 50M | greedy | 0 | 7/13 | 0.0056 | None | 0/13 | 0.0 |
| 50M | greedy | 32 | 5/13 | 0.0604 | 0.38461538461538464 | 0/13 | 0.0 |
| 50M | greedy | 64 | 6/13 | 0.0583 | 1.0 | 0/13 | 0.0 |
| 50M | greedy | 128 | 6/13 | 0.0341 | 0.46153846153846156 | 0/13 | 0.0 |
| 50M | greedy | 256 | 6/13 | 0.3660 | 0.07692307692307693 | 0/13 | 0.0 |
| 50M | sampled | 0 | 7/13 | 0.0056 | None | 0/13 | 0.0 |
| 50M | sampled | 32 | 7/13 | 0.1972 | 0.23076923076923078 | 0/13 | 0.0 |
| 50M | sampled | 64 | 8/13 | 0.3218 | 0.38461538461538464 | 0/13 | 0.0 |
| 50M | sampled | 128 | 6/13 | -0.0188 | 0.46153846153846156 | 0/13 | 0.0 |
| 50M | sampled | 256 | 5/13 | -0.1036 | 0.07692307692307693 | 0/13 | 0.0 |

No collapse point can be identified because Contrastive gold scoring is already poor (4/13; mean margin -0.2288). Its greedy counts at 0/32/64/128/256 are 4/4/5/7/6 and sampled counts are 4/5/3/5/3; these non-monotonic fluctuations are not evidence of improvement. Exact recovery errors are zero. The phrase-based contradiction check is conservative.
