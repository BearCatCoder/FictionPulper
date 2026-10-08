# Repetition Comparison

Metrics use tokenizer-v1 over generated continuations only.

## Historical 10-Prompt Suite

| Model | Mode | Distinct-1 | Distinct-2 | Distinct-3 | Repeated 4-gram | Repeated sentence | Longest span |
|---|---:|---:|---:|---:|---:|---:|---:|
| Compute5 | greedy | 0.2508 | 0.3913 | 0.4730 | 0.4632 | 0.1506 | 17.7 |
| Counterfactual-v1 | greedy | 0.1977 | 0.2906 | 0.3460 | 0.6056 | 0.2300 | 32.0 |
| State-Transition-v1 | greedy | 0.2195 | 0.3268 | 0.3889 | 0.5592 | 0.0911 | 22.4 |
| Compute5 | sampled | 0.4770 | 0.8282 | 0.9358 | 0.0194 | 0 | 4.9 |
| Counterfactual-v1 | sampled | 0.4605 | 0.8039 | 0.9291 | 0.0249 | 0 | 4.7 |
| State-Transition-v1 | sampled | 0.4121 | 0.7400 | 0.8547 | 0.0146 | 0 | 4.3 |

## Narrative-State 3-Prompt Suite

| Model | Mode | Distinct-1 | Distinct-2 | Distinct-3 | Repeated 4-gram | Repeated sentence | Longest span |
|---|---:|---:|---:|---:|---:|---:|---:|
| Compute5 | greedy | 0.1927 | 0.2598 | 0.3069 | 0.6507 | 0.3889 | 35.67 |
| Counterfactual-v1 | greedy | 0.1536 | 0.2362 | 0.2751 | 0.7013 | 0.2810 | 40.00 |
| State-Transition-v1 | greedy | 0.2031 | 0.3018 | 0.3677 | 0.5893 | 0.2571 | 35.67 |
| Compute5 | sampled | 0.4635 | 0.8078 | 0.9147 | 0.0369 | 0 | 5.33 |
| Counterfactual-v1 | sampled | 0.4557 | 0.8131 | 0.9331 | 0.0184 | 0 | 5.33 |
| State-Transition-v1 | sampled | 0.4727 | 0.8105 | 0.9134 | 0.0369 | 0 | 5.67 |

State-Transition-v1 is less repetitive than Counterfactual-v1 under greedy
decoding, but remains highly repetitive and is mixed against Compute5. Sampled
differences are mixed. None of these surface metrics corresponds to semantic
fact retention: both candidate modes score 0/13.
