# Quantitative Comparison

## Counterfactual Evaluation

| Split | Model | Pairs | Direction | Joint reversal | Signed margin |
|---|---:|---:|---:|---:|---:|
| test | Compute5 | 329 | 0.5684 | 0.1368 | 0.0525 |
| test | Narrative-v1 | 329 | 0.5228 | 0.0486 | 0.0287 |
| test | Contrastive-v1 | 329 | 0.5334 | 0.0760 | 0.0474 |
| test | Counterfactual-v1 | 329 | 0.7340 | 0.4802 | 2.7829 |
| generalization_holdout | Compute5 | 148 | 0.5270 | 0.0541 | 0.0620 |
| generalization_holdout | Narrative-v1 | 148 | 0.5203 | 0.0473 | 0.0347 |
| generalization_holdout | Contrastive-v1 | 148 | 0.5203 | 0.0405 | 0.0473 |
| generalization_holdout | Counterfactual-v1 | 148 | 0.7230 | 0.4459 | 2.8601 |

Joint reversal has a 0.25 random-pair reference. Direction accuracy has a 0.5 reference.

## Candidate Controls

| Split | Control | X win | Y win | X-Y margin | Absolute margin |
|---|---:|---:|---:|---:|---:|
| test | fact_removed | 0.5106 | 0.4894 | 0.0161 | 0.4880 |
| test | irrelevant_substituted | 0.5258 | 0.4742 | 0.0162 | 0.3639 |
| test | candidate_only | 0.4833 | 0.5167 | 0.0034 | n/a |
| generalization_holdout | fact_removed | 0.4662 | 0.5338 | -0.0012 | 0.5780 |
| generalization_holdout | irrelevant_substituted | 0.4730 | 0.5270 | -0.0192 | 0.4669 |
| generalization_holdout | candidate_only | 0.4797 | 0.5203 | -0.0271 | n/a |

Removed and irrelevant controls use one shared context, so paired reversal is undefined; their balanced directional accuracy is mechanically 0.5. These rows report side preference magnitude only.

## Candidate State Families

| Split | Family | N | Direction | Joint reversal | Signed margin |
|---|---:|---:|---:|---:|---:|
| test | cause_effect | 33 | 0.8788 | 0.7576 | 1.5638 |
| test | character_location | 33 | 0.5606 | 0.1212 | 0.0980 |
| test | goal_intention | 33 | 0.5000 | 0.0000 | 0.1159 |
| test | knowledge_secret_holder | 33 | 0.7273 | 0.4545 | 0.5966 |
| test | latest_state_update | 32 | 0.4375 | 0.0000 | -0.0150 |
| test | object_door_state | 33 | 1.0000 | 1.0000 | 6.4508 |
| test | object_location | 33 | 0.5758 | 0.1515 | 0.1054 |
| test | object_ownership | 33 | 0.6515 | 0.3030 | 0.5103 |
| test | persistent_physical_state | 33 | 1.0000 | 1.0000 | 9.2660 |
| test | relationship | 33 | 1.0000 | 1.0000 | 9.0526 |
| generalization_holdout | cause_effect | 14 | 0.5000 | 0.0000 | 0.0279 |
| generalization_holdout | character_location | 15 | 0.5333 | 0.0667 | 0.0387 |
| generalization_holdout | goal_intention | 15 | 0.5000 | 0.0000 | 0.0527 |
| generalization_holdout | knowledge_secret_holder | 15 | 0.9000 | 0.8000 | 1.4477 |
| generalization_holdout | latest_state_update | 14 | 0.5000 | 0.0000 | 0.0059 |
| generalization_holdout | object_door_state | 15 | 1.0000 | 1.0000 | 6.1044 |
| generalization_holdout | object_location | 15 | 0.5000 | 0.0000 | 0.0358 |
| generalization_holdout | object_ownership | 15 | 0.7667 | 0.5333 | 1.0191 |
| generalization_holdout | persistent_physical_state | 15 | 1.0000 | 1.0000 | 9.4960 |
| generalization_holdout | relationship | 15 | 1.0000 | 1.0000 | 9.9939 |

## Historical Forced Choice

| Model | Top-1 | Correct-best-negative margin |
|---|---:|---:|
| Compute5 | 4/13 | -0.5349 |
| Narrative-v1 | 3/13 | -0.5611 |
| Contrastive-v1 | 4/13 | -0.2288 |
| Capacity-50M | 7/13 | 0.0056 |
| Counterfactual-v1 | 6/13 | -0.1225 |

## Real-Fiction LM

| Split | Model | Loss | Perplexity | Accuracy | Valid tokens |
|---|---:|---:|---:|---:|---:|
| data30m_test | Compute5 | 3.223102 | 25.1059 | 0.363705 | 1546717 |
| data10m_test_data30m_train_disjoint | Compute5 | 3.224620 | 25.1440 | 0.360152 | 445606 |
| legacy_seed_validation_leakage_clean | Compute5 | 2.999032 | 20.0661 | 0.384642 | 114725 |
| legacy_seed_test | Compute5 | 3.085006 | 21.8676 | 0.373671 | 136679 |
| data30m_test | Narrative-v1 | 3.220216 | 25.0335 | 0.364367 | 1546717 |
| data10m_test_data30m_train_disjoint | Narrative-v1 | 3.220879 | 25.0501 | 0.361054 | 445606 |
| legacy_seed_validation_leakage_clean | Narrative-v1 | 3.002836 | 20.1426 | 0.382131 | 114725 |
| legacy_seed_test | Narrative-v1 | 3.093766 | 22.0600 | 0.371666 | 136679 |
| data30m_test | Contrastive-v1 | 3.218860 | 24.9996 | 0.365434 | 1546717 |
| data10m_test_data30m_train_disjoint | Contrastive-v1 | 3.221451 | 25.0645 | 0.361968 | 445606 |
| legacy_seed_validation_leakage_clean | Contrastive-v1 | 3.025733 | 20.6091 | 0.381146 | 114725 |
| legacy_seed_test | Contrastive-v1 | 3.106653 | 22.3461 | 0.369618 | 136679 |
| data30m_test | Counterfactual-v1 | 3.213676 | 24.8703 | 0.366538 | 1546717 |
| data10m_test_data30m_train_disjoint | Counterfactual-v1 | 3.217054 | 24.9545 | 0.362928 | 445606 |
| legacy_seed_validation_leakage_clean | Counterfactual-v1 | 3.047830 | 21.0696 | 0.379987 | 114725 |
| legacy_seed_test | Counterfactual-v1 | 3.128135 | 22.8313 | 0.368645 | 136679 |

Counterfactual-v1 Data30M test loss changes -0.2925% versus Compute5. The clean common Data10M subset changes -0.2346%. These small one-seed differences establish no material primary regression or gain. Legacy diagnostics regress but overlap Data30M training.
