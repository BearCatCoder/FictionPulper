# Quantitative Comparison

## Controlled Comparison

| Metric | Compute5 | Counterfactual-v1 | State-Transition-v1 |
|---|---:|---:|---:|
| LM parameters | 15,047,040 | 15,047,040 | 15,047,040 |
| Training-only auxiliary parameters | 0 | 0 | 6,930 |
| Context | 1,024 | 1,024 | 1,024 |
| Optimizer steps | 2,220 | 2,220 | 2,220 |
| Total LM target exposure | 135,886,355 | 135,886,355 | 135,886,355 |
| Curriculum target exposure | 0 | 20,382,953 | 20,382,953 |
| Pair evaluations | 0 | 41,238 | 41,238 |
| State annotations trained | 0 | 0 | 325,416 |
| Data30M test loss | 3.223102 | 3.213676 | 3.235881 |
| Latest-state reversal, test/generalization | 0% / 0% | 0% / 0% | 0% / 0% |
| Historical forced choice | 7/13 | 6/13 | 4/13 |
| Greedy/sampled fact retention | 0/13 / 0/13 | 0/13 / 0/13 | 0/13 / 0/13 |
| Frozen-probe summary | no broad signal | no broad improvement | no broad improvement |
| Runtime seconds | 469.624 | 1,585.328 | 1,728.516 |
| Peak reserved GPU memory | 3.9406 GiB | 8.1641 GiB | 10.9609 GiB |

All three runs use the same corpus, tokenizer, split, architecture, seed, LR
schedule, batch geometry, and checkpoint-selection rule. Total LM target exposure
is unchanged; State-Transition-v1 adds supervised state-head evaluations rather
than LM targets.

## Auxiliary Head

| Split | Overall | Transition | Carry | Final current transition | Depth 2 | Depth 3 | Depth 4+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| test | 66.438% | 63.019% | 67.845% | 64.438% | 50.000% (64) | unavailable | unavailable |
| generalization | 65.946% | 62.346% | 67.430% | 62.838% | 46.429% (28) | unavailable | unavailable |

Test final-current-transition family accuracy is: causal 50.000%, character
location 43.939%, goal 51.515%, knowledge 53.030%, object location 50.000%,
ownership 45.455%, and door/physical/relationship 100%. Generalization shows
the same split: causal 50.000%, character location 43.333%, goal 46.667%,
knowledge 53.333%, object location 48.276%, ownership 36.667%, and the same
three static families at 100%. Thus the aggregate uplift is lexical/static,
not evidence of robust transition handling.

## LM Scoring

| Split | Model | Direction | Joint paired reversal | Latest-state reversal |
|---|---:|---:|---:|---:|
| test | Counterfactual-v1 | 73.404% | 48.024% | 0% |
| test | State-Transition-v1 | 70.973% | 42.249% | 0% |
| generalization | Counterfactual-v1 | 72.297% | 44.595% | 0% |
| generalization | State-Transition-v1 | 68.243% | 36.486% | 0% |

| Model | Historical forced choice | Mean correct-best-negative margin |
|---|---:|---:|
| Counterfactual-v1 | 6/13 | -0.122452 |
| State-Transition-v1 | 4/13 | -0.111185 |

## Free Generation

| Model | Greedy retained | Sampled retained |
|---|---:|---:|
| Counterfactual-v1 | 0/13 | 0/13 |
| State-Transition-v1 | 0/13 | 0/13 |

## Real Fiction

| Model | Data30M test loss | Perplexity | Accuracy |
|---|---:|---:|---:|
| Counterfactual-v1 | 3.213676 | 24.8703 | 36.6538% |
| Compute5 | 3.223102 | 25.1059 | 36.3705% |
| State-Transition-v1 | 3.235881 | 25.4288 | 36.3846% |

State-Transition-v1 is +0.6910% loss versus Counterfactual-v1 and +0.3965%
versus Compute5. This is modest, non-material one-seed degradation, not
`REAL_FICTION_DEGRADED`.

## Frozen Probes

Exact candidate interpretations from `probe-comparison.json`:

| Family | Best layer | Test | Generalization holdout |
|---|---:|---:|---:|
| object_ownership | layer_05 | `no_clear_linear_state_signal` | `weak_or_imbalanced_linear_state_signal` |
| object_location | layer_02 | `weak_or_imbalanced_linear_state_signal` | `no_clear_linear_state_signal` |
| character_location | layer_03 | `weak_or_imbalanced_linear_state_signal` | `no_clear_linear_state_signal` |
| knowledge_holder | final_norm | `weak_or_imbalanced_linear_state_signal` | `no_clear_linear_state_signal` |
| relationship_state | layer_06 | `no_clear_linear_state_signal` | `weak_or_imbalanced_linear_state_signal` |
| injury_state | layer_03 | `no_clear_linear_state_signal` | `no_clear_linear_state_signal` |
| temporal_order | layer_07 | `weak_or_imbalanced_linear_state_signal` | `weak_or_imbalanced_linear_state_signal` |
| obligation_state | layer_05 | `no_clear_linear_state_signal` | `weak_or_imbalanced_linear_state_signal` |

No family is `strong_linear_state_signal` on both holdouts. This is no broad
frozen-representation improvement.

## Training Controls

| Item | Exact value |
|---|---:|
| Base LM parameters | 15,047,040 |
| Training-only auxiliary parameters | 6,930 |
| Total training parameters | 15,053,970 |
| Optimizer steps | 2,220 |
| Real targets | 115,503,402 |
| Curriculum targets | 20,382,953 |
| Pair evaluations | 41,238 |
| Transition annotations | 89,850 |
| Carry annotations | 235,566 |
| Total state annotations | 325,416 |
| Runtime | 1,728.515979 seconds |
| Peak reserved GPU memory | 10.9609375 GiB |
