# Repetition Comparison

Tokenizer-v1 metrics over generated continuations only; prompts excluded. Lower repetition rates and higher distinct scores are better.

| Suite | Model | Mode | Distinct-1 | Distinct-2 | Distinct-3 | Repeated 4-gram | Repeated sentence | Longest span |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| historical_10_prompt | compute5 | greedy | 0.2508 | 0.3913 | 0.4730 | 0.4632 | 0.1506 | 17.70 |
| historical_10_prompt | compute5 | sampled | 0.4770 | 0.8282 | 0.9358 | 0.0194 | 0.0000 | 4.90 |
| historical_10_prompt | narrative | greedy | 0.2156 | 0.3323 | 0.4048 | 0.5392 | 0.0000 | 24.50 |
| historical_10_prompt | narrative | sampled | 0.5059 | 0.8424 | 0.9598 | 0.0115 | 0.0000 | 4.50 |
| historical_10_prompt | contrastive_v1 | greedy | 0.2453 | 0.3858 | 0.4817 | 0.4464 | 0.0667 | 16.80 |
| historical_10_prompt | contrastive_v1 | sampled | 0.4892 | 0.8305 | 0.9472 | 0.0190 | 0.0077 | 4.80 |
| narrative_state_3_prompt | compute5_context1024 | greedy | 0.1927 | 0.2598 | 0.3069 | 0.6507 | 0.3889 | 35.67 |
| narrative_state_3_prompt | compute5_context1024 | sampled | 0.4635 | 0.8078 | 0.9147 | 0.0369 | 0.0000 | 5.33 |
| narrative_state_3_prompt | narrative_context1024 | greedy | 0.2031 | 0.3071 | 0.3704 | 0.5840 | 0.3571 | 37.33 |
| narrative_state_3_prompt | narrative_context1024 | sampled | 0.4974 | 0.8275 | 0.9278 | 0.0264 | 0.0000 | 5.33 |
| narrative_state_3_prompt | contrastive_v1_context1024 | greedy | 0.2812 | 0.4173 | 0.4947 | 0.4507 | 0.2593 | 18.00 |
| narrative_state_3_prompt | contrastive_v1_context1024 | sampled | 0.5104 | 0.8170 | 0.9081 | 0.0369 | 0.0000 | 5.33 |

On the historical suite Contrastive-v1 is close to Compute5 and mixed versus Narrative-v1. On the three narrative prompts its greedy repeated 4-gram rate is 0.4507 versus 0.6507 Compute5 and 0.5840 Narrative-v1, but repetition remains substantial and fact retention remains 0/13. Sampled differences are small and mixed.
