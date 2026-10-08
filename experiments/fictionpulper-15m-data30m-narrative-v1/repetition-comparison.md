# Repetition Comparison

Tokenizer-v1 metrics over generated continuations only; prompts are excluded. Lower repeated rates and higher distinct scores are better.

| Suite | Model | Mode | Distinct-1 | Distinct-2 | Distinct-3 | Repeated 4-gram | Repeated sentence | Longest repeated span |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| historical_10_prompt | compute5 | greedy | 0.25078125 | 0.39133858267716537 | 0.473015873015873 | 0.4632 | 0.1506060606060606 | 17.7 |
| historical_10_prompt | compute5 | sampled | 0.476953125 | 0.8282352941176471 | 0.9358267716535433 | 0.019367588932806323 | 0.0 | 4.9 |
| historical_10_prompt | narrative | greedy | 0.215625 | 0.33228346456692914 | 0.40476190476190477 | 0.5392 | 0.0 | 24.5 |
| historical_10_prompt | narrative | sampled | 0.505859375 | 0.8423529411764706 | 0.9598425196850394 | 0.011462450592885375 | 0.0 | 4.5 |
| narrative_state | compute5_context1024 | greedy | 0.19270833333333334 | 0.25984251968503935 | 0.30687830687830686 | 0.6506666666666667 | 0.3888888888888889 | 35.666666666666664 |
| narrative_state | compute5_context1024 | sampled | 0.4635416666666667 | 0.807843137254902 | 0.9146981627296588 | 0.03689064558629776 | 0.0 | 5.333333333333333 |
| narrative_state | narrative_context1024 | greedy | 0.203125 | 0.30708661417322836 | 0.3703703703703704 | 0.584 | 0.35714285714285715 | 37.333333333333336 |
| narrative_state | narrative_context1024 | sampled | 0.4973958333333333 | 0.8274509803921569 | 0.9278215223097113 | 0.026350461133069828 | 0.0 | 5.333333333333333 |

The candidate is modestly less repetitive on most aggregate rates, especially sampled output, but greedy repetition remains severe and this did not translate into state retention or narrative progression.
