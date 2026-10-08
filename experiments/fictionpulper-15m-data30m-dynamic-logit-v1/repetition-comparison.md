# Repetition Comparison

Metrics cover generated continuations only; prompts are excluded. Lower
repetition rates and shorter repeated spans are better.

| Suite | Model | Mode | Repeated 4-gram | Repeated sentence | Longest repeated span |
|---|---|---|---:|---:|---:|
| historical 10-prompt | Compute5 | greedy | 0.4632 | 0.1506 | 17.70 |
| historical 10-prompt | Counterfactual-v1 | greedy | 0.6056 | 0.2300 | 32.00 |
| historical 10-prompt | State-Transition-v1 | greedy | 0.5592 | 0.0911 | 22.40 |
| historical 10-prompt | Dynamic-logit-v1 | greedy | 0.5496 | 0.1696 | 28.40 |
| historical 10-prompt | Dynamic-logit-v1 | sampled | 0.0170 | 0.0000 | 4.70 |
| narrative-state 3-prompt | Dynamic-logit-v1 | greedy | 0.6267 | 0.3702 | 37.33 |
| narrative-state 3-prompt | Dynamic-logit-v1 | sampled | 0.0105 | 0.0000 | 4.33 |

Dynamic-logit-v1 remains strongly repetitive under greedy decoding. Sampling
greatly reduces repetition but does not improve fact retention or coherence.
