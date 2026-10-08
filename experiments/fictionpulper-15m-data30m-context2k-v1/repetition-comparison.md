# Repetition Comparison

Metrics use tokenizer-v1 token n-grams over generated continuation text only. Values are means across prompts. Higher distinct-n is better; lower repeated rates and shorter repeated spans are better.

## Original Ten Prompts

| Model / mode | Distinct-1 | Distinct-2 | Distinct-3 | Repeated 4-gram rate | Repeated sentence rate | Longest repeated span |
|---|---:|---:|---:|---:|---:|---:|
| Compute5 greedy | 0.25078 | 0.39134 | 0.47302 | 0.46320 | 0.15061 | 17.7 |
| Context2k greedy | 0.21406 | 0.32835 | 0.39444 | 0.55760 | 0.18889 | 26.2 |
| Compute5 sampled | 0.47695 | 0.82824 | 0.93583 | 0.01937 | 0.00000 | 4.9 |
| Context2k sampled | 0.45117 | 0.79529 | 0.92244 | 0.02609 | 0.00000 | 4.9 |

Context2k is worse on every aggregate distinct-n metric and repeated 4-gram rate. Greedy degeneration is materially worse by both repeated 4-grams and longest repeated span.

## Context-Retention Prompts

| Model / mode | Distinct-1 | Distinct-2 | Distinct-3 | Repeated 4-gram rate | Repeated sentence rate | Longest repeated span |
|---|---:|---:|---:|---:|---:|---:|
| Compute5 greedy | 0.25000 | 0.38320 | 0.46032 | 0.48533 | 0.40278 | 28.0 |
| Context2k greedy | 0.21094 | 0.29921 | 0.33862 | 0.63467 | 0.33333 | 31.0 |
| Compute5 sampled | 0.46484 | 0.78562 | 0.90551 | 0.03953 | 0.00000 | 6.33 |
| Context2k sampled | 0.51172 | 0.84052 | 0.92782 | 0.03030 | 0.00000 | 5.0 |

Context2k sampled outputs are more diverse on the three retention prompts, but they retain zero of 13 explicit target facts. Diversity alone did not produce useful long-range state retention.
