# Quantitative Comparison

## Dynamic State

| Split | N | CURRENT | PREVIOUS | NEVER_VALID | Depth-2 CURRENT | Current-prev margin |
|---|---:|---:|---:|---:|---:|---:|
| test | 1,380 | 69.783% | 4.565% | 25.652% | 50.781% | 0.003004 |
| generalization | 620 | 69.516% | 4.516% | 25.968% | 50.000% | 0.018784 |

At depth 2, PREVIOUS is also INITIAL; precedence assigns it to PREVIOUS while
the alias count preserves that interpretation. OLDER, depth 3, and depth 4+ are
unavailable in the sealed source.

| Family | Test CURRENT | Generalization CURRENT |
|---|---:|---:|
| cause/effect | 53.030% | 50.000% |
| character location | 56.061% | 55.000% |
| goal/intention | 53.788% | 55.000% |
| knowledge/secret holder | 73.485% | 80.000% |
| latest-state update | 51.042% | 50.000% |
| object door state | 100.000% | 100.000% |
| object location | 53.788% | 50.000% |
| object ownership | 65.152% | 68.333% |
| persistent physical state | 100.000% | 95.000% |
| relationship | 100.000% | 98.333% |

## Counterfactual Transfer

| Model | Test direction | Test reversal | Generalization direction | Generalization reversal |
|---|---:|---:|---:|---:|
| Compute5 | 56.839% | 13.678% | 52.703% | 5.405% |
| Counterfactual-v1 | 73.404% | 48.024% | 72.297% | 44.595% |
| State-Transition-v1 | 70.973% | 42.249% | 68.243% | 36.486% |
| Dynamic-logit-v1 | 67.629% | 35.258% | 70.270% | 40.541% |

Latest-state reversal is 0% for Dynamic-logit-v1 on both splits. Under fact
removal, balanced direction is mechanically 50%; the candidate's mean signed
margin approaches zero. The contradictory-context audit has valid opposite
labels, but preference switches only 39.855% on test and 39.032% on
generalization.

## Historical Forced Choice

| Model | Top-1 | Correct-best-negative margin |
|---|---:|---:|
| Compute5 | 4/13 | -0.534941 |
| Counterfactual-v1 | 6/13 | -0.122452 |
| State-Transition-v1 | 4/13 | -0.111185 |
| Dynamic-logit-v1 | 4/13 | -0.398496 |

## Real-Fiction LM

| Model | Data30M test loss | Clean common Data10M loss |
|---|---:|---:|
| Compute5 | 3.223102 | 3.224620 |
| Counterfactual-v1 | 3.213676 | 3.217054 |
| State-Transition-v1 | 3.235881 | 3.238363 |
| Dynamic-logit-v1 | 3.244104 | 3.245736 |

The primary held-out regression is modest and single-seed. Historical legacy
splits overlap Data30M training and are continuity diagnostics, not clean tests.
