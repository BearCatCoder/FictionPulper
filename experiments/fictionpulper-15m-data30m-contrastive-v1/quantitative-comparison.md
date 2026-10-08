# Quantitative Comparison

## Real-Fiction LM

| Split | Model | Loss | Perplexity | Accuracy | Valid tokens |
|---|---:|---:|---:|---:|---:|
| data30m_test | Compute5 | 3.223102 | 25.1059 | 0.363705 | 1546717 |
| data30m_test | Narrative-v1 | 3.220216 | 25.0335 | 0.364367 | 1546717 |
| data30m_test | Contrastive-v1 | 3.218860 | 24.9996 | 0.365434 | 1546717 |
| data10m_test_data30m_train_disjoint | Compute5 | 3.224620 | 25.1440 | 0.360152 | 445606 |
| data10m_test_data30m_train_disjoint | Narrative-v1 | 3.220879 | 25.0501 | 0.361054 | 445606 |
| data10m_test_data30m_train_disjoint | Contrastive-v1 | 3.221451 | 25.0645 | 0.361968 | 445606 |
| legacy_seed_validation_leakage_clean | Compute5 | 2.999032 | 20.0661 | 0.384642 | 114725 |
| legacy_seed_validation_leakage_clean | Narrative-v1 | 3.002836 | 20.1426 | 0.382131 | 114725 |
| legacy_seed_validation_leakage_clean | Contrastive-v1 | 3.025733 | 20.6091 | 0.381146 | 114725 |
| legacy_seed_test | Compute5 | 3.085006 | 21.8676 | 0.373671 | 136679 |
| legacy_seed_test | Narrative-v1 | 3.093766 | 22.0600 | 0.371666 | 136679 |
| legacy_seed_test | Contrastive-v1 | 3.106653 | 22.3461 | 0.369618 | 136679 |

## Held-Out Pair Ranking

| Split | Model | Pairs | Accuracy | Mean margin |
|---|---:|---:|---:|---:|
| test | Contrastive-v1 | 469 | 0.9765 | 1.1597 |
| test | Compute5 | 469 | 0.4776 | -0.0076 |
| test | Narrative-v1 | 469 | 0.8678 | 0.6820 |
| generalization | Contrastive-v1 | 207 | 0.9517 | 1.2737 |
| generalization | Compute5 | 207 | 0.5169 | 0.0840 |
| generalization | Narrative-v1 | 207 | 0.9034 | 0.7769 |

### test: state_type

| Group | Model | N | Accuracy | Margin |
|---|---:|---:|---:|---:|
| cause_effect | Contrastive-v1 | 46 | 0.8478 | 0.5921 |
| character_identity_role_relationship | Contrastive-v1 | 46 | 1.0000 | 0.5002 |
| goal_persistence_completion_failure | Contrastive-v1 | 31 | 1.0000 | 0.6825 |
| knowledge_ignorance_secret | Contrastive-v1 | 77 | 1.0000 | 1.1277 |
| ownership_transfer_hiding_retrieval | Contrastive-v1 | 63 | 1.0000 | 1.6468 |
| persistent_injury | Contrastive-v1 | 79 | 1.0000 | 1.1929 |
| physical_scene_location_movement | Contrastive-v1 | 32 | 0.8750 | 0.3916 |
| promise_debt_obligation | Contrastive-v1 | 48 | 1.0000 | 1.2311 |
| temporal_ordering | Contrastive-v1 | 47 | 1.0000 | 2.4687 |
| cause_effect | Compute5 | 46 | 0.0000 | -0.4454 |
| character_identity_role_relationship | Compute5 | 46 | 1.0000 | 0.2695 |
| goal_persistence_completion_failure | Compute5 | 31 | 0.3548 | -0.0364 |
| knowledge_ignorance_secret | Compute5 | 77 | 0.3247 | -0.0415 |
| ownership_transfer_hiding_retrieval | Compute5 | 63 | 0.9206 | 0.3670 |
| persistent_injury | Compute5 | 79 | 0.3671 | -0.0114 |
| physical_scene_location_movement | Compute5 | 32 | 0.0000 | -0.8169 |
| promise_debt_obligation | Compute5 | 48 | 0.1667 | -0.4446 |
| temporal_ordering | Compute5 | 47 | 1.0000 | 0.7262 |
| cause_effect | Narrative-v1 | 46 | 0.8478 | 0.3495 |
| character_identity_role_relationship | Narrative-v1 | 46 | 0.8696 | 0.1610 |
| goal_persistence_completion_failure | Narrative-v1 | 31 | 0.4194 | 0.0229 |
| knowledge_ignorance_secret | Narrative-v1 | 77 | 1.0000 | 0.6745 |
| ownership_transfer_hiding_retrieval | Narrative-v1 | 63 | 1.0000 | 1.0441 |
| persistent_injury | Narrative-v1 | 79 | 1.0000 | 0.6099 |
| physical_scene_location_movement | Narrative-v1 | 32 | 0.5000 | 0.1155 |
| promise_debt_obligation | Narrative-v1 | 48 | 0.6875 | 0.3964 |
| temporal_ordering | Narrative-v1 | 47 | 1.0000 | 2.2778 |

### test: distance_bucket

| Group | Model | N | Accuracy | Margin |
|---|---:|---:|---:|---:|
| d064 | Contrastive-v1 | 104 | 0.9135 | 0.7043 |
| d128 | Contrastive-v1 | 104 | 0.9808 | 0.9313 |
| d256 | Contrastive-v1 | 103 | 1.0000 | 1.0837 |
| d384 | Contrastive-v1 | 71 | 1.0000 | 1.7753 |
| d512 | Contrastive-v1 | 46 | 1.0000 | 1.5847 |
| d768 | Contrastive-v1 | 28 | 1.0000 | 1.3030 |
| d900 | Contrastive-v1 | 13 | 1.0000 | 2.0561 |
| d064 | Compute5 | 104 | 0.3654 | -0.0846 |
| d128 | Compute5 | 104 | 0.5673 | 0.0978 |
| d256 | Compute5 | 103 | 0.5922 | 0.0355 |
| d384 | Compute5 | 71 | 0.3239 | -0.1389 |
| d512 | Compute5 | 46 | 0.3261 | -0.0854 |
| d768 | Compute5 | 28 | 0.7500 | 0.1310 |
| d900 | Compute5 | 13 | 0.5385 | 0.1189 |
| d064 | Narrative-v1 | 104 | 0.7788 | 0.4821 |
| d128 | Narrative-v1 | 104 | 0.9135 | 0.5752 |
| d256 | Narrative-v1 | 103 | 0.8932 | 0.5643 |
| d384 | Narrative-v1 | 71 | 0.9859 | 1.0742 |
| d512 | Narrative-v1 | 46 | 0.6957 | 0.8743 |
| d768 | Narrative-v1 | 28 | 0.8571 | 0.6093 |
| d900 | Narrative-v1 | 13 | 1.0000 | 1.4033 |

### test: difficulty

| Group | Model | N | Accuracy | Margin |
|---|---:|---:|---:|---:|
| 1 | Contrastive-v1 | 94 | 1.0000 | 1.6805 |
| 2 | Contrastive-v1 | 94 | 1.0000 | 0.8569 |
| 3 | Contrastive-v1 | 94 | 0.9574 | 1.3901 |
| 4 | Contrastive-v1 | 94 | 1.0000 | 1.0903 |
| 5 | Contrastive-v1 | 93 | 0.9247 | 0.7764 |
| 1 | Compute5 | 94 | 0.4574 | 0.0273 |
| 2 | Compute5 | 94 | 0.4468 | -0.1112 |
| 3 | Compute5 | 94 | 0.3298 | -0.1244 |
| 4 | Compute5 | 94 | 0.4681 | 0.1176 |
| 5 | Compute5 | 93 | 0.6882 | 0.0535 |
| 1 | Narrative-v1 | 94 | 1.0000 | 1.0055 |
| 2 | Narrative-v1 | 94 | 0.6489 | 0.2403 |
| 3 | Narrative-v1 | 94 | 0.8298 | 1.0941 |
| 4 | Narrative-v1 | 94 | 1.0000 | 0.5869 |
| 5 | Narrative-v1 | 93 | 0.8602 | 0.4813 |

### generalization: state_type

| Group | Model | N | Accuracy | Margin |
|---|---:|---:|---:|---:|
| cause_effect | Contrastive-v1 | 20 | 0.7000 | 0.3915 |
| character_identity_role_relationship | Contrastive-v1 | 20 | 1.0000 | 0.6562 |
| goal_persistence_completion_failure | Contrastive-v1 | 14 | 1.0000 | 0.5417 |
| knowledge_ignorance_secret | Contrastive-v1 | 35 | 0.9714 | 1.0100 |
| ownership_transfer_hiding_retrieval | Contrastive-v1 | 27 | 1.0000 | 2.0143 |
| persistent_injury | Contrastive-v1 | 35 | 1.0000 | 1.3094 |
| physical_scene_location_movement | Contrastive-v1 | 14 | 0.7857 | 0.3933 |
| promise_debt_obligation | Contrastive-v1 | 21 | 1.0000 | 1.3016 |
| temporal_ordering | Contrastive-v1 | 21 | 1.0000 | 3.1772 |
| cause_effect | Compute5 | 20 | 0.0000 | -0.3571 |
| character_identity_role_relationship | Compute5 | 20 | 1.0000 | 0.3728 |
| goal_persistence_completion_failure | Compute5 | 14 | 0.1429 | -0.3379 |
| knowledge_ignorance_secret | Compute5 | 35 | 0.5143 | -0.0198 |
| ownership_transfer_hiding_retrieval | Compute5 | 27 | 1.0000 | 0.6759 |
| persistent_injury | Compute5 | 35 | 0.3429 | -0.0677 |
| physical_scene_location_movement | Compute5 | 14 | 0.0000 | -0.8782 |
| promise_debt_obligation | Compute5 | 21 | 0.3333 | -0.2682 |
| temporal_ordering | Compute5 | 21 | 1.0000 | 1.1686 |
| cause_effect | Narrative-v1 | 20 | 0.6500 | 0.2021 |
| character_identity_role_relationship | Narrative-v1 | 20 | 1.0000 | 0.2305 |
| goal_persistence_completion_failure | Narrative-v1 | 14 | 0.5714 | -0.0604 |
| knowledge_ignorance_secret | Narrative-v1 | 35 | 0.9714 | 0.5494 |
| ownership_transfer_hiding_retrieval | Narrative-v1 | 27 | 1.0000 | 1.0708 |
| persistent_injury | Narrative-v1 | 35 | 1.0000 | 0.7830 |
| physical_scene_location_movement | Narrative-v1 | 14 | 0.8571 | 0.3139 |
| promise_debt_obligation | Narrative-v1 | 21 | 0.8095 | 0.6822 |
| temporal_ordering | Narrative-v1 | 21 | 1.0000 | 2.7971 |

### generalization: distance_bucket

| Group | Model | N | Accuracy | Margin |
|---|---:|---:|---:|---:|
| d064 | Contrastive-v1 | 47 | 0.8511 | 0.9645 |
| d128 | Contrastive-v1 | 45 | 0.9333 | 0.8632 |
| d256 | Contrastive-v1 | 44 | 1.0000 | 1.2765 |
| d384 | Contrastive-v1 | 31 | 1.0000 | 1.7139 |
| d512 | Contrastive-v1 | 21 | 1.0000 | 1.8121 |
| d768 | Contrastive-v1 | 13 | 1.0000 | 1.5059 |
| d900 | Contrastive-v1 | 6 | 1.0000 | 2.0933 |
| d064 | Compute5 | 47 | 0.4468 | 0.0948 |
| d128 | Compute5 | 45 | 0.5556 | 0.0510 |
| d256 | Compute5 | 44 | 0.6364 | 0.2097 |
| d384 | Compute5 | 31 | 0.4194 | -0.0396 |
| d512 | Compute5 | 21 | 0.3333 | -0.0429 |
| d768 | Compute5 | 13 | 0.8462 | 0.2870 |
| d900 | Compute5 | 6 | 0.3333 | -0.0333 |
| d064 | Narrative-v1 | 47 | 0.8085 | 0.6860 |
| d128 | Narrative-v1 | 45 | 0.9333 | 0.5817 |
| d256 | Narrative-v1 | 44 | 0.9318 | 0.7112 |
| d384 | Narrative-v1 | 31 | 1.0000 | 1.0589 |
| d512 | Narrative-v1 | 21 | 0.7619 | 1.0489 |
| d768 | Narrative-v1 | 13 | 1.0000 | 0.6623 |
| d900 | Narrative-v1 | 6 | 1.0000 | 1.2724 |

### generalization: difficulty

| Group | Model | N | Accuracy | Margin |
|---|---:|---:|---:|---:|
| 1 | Contrastive-v1 | 42 | 1.0000 | 1.8779 |
| 2 | Contrastive-v1 | 42 | 1.0000 | 0.9639 |
| 3 | Contrastive-v1 | 41 | 0.9024 | 1.5944 |
| 4 | Contrastive-v1 | 41 | 1.0000 | 1.1696 |
| 5 | Contrastive-v1 | 41 | 0.8537 | 0.7556 |
| 1 | Compute5 | 42 | 0.5952 | 0.1508 |
| 2 | Compute5 | 42 | 0.4524 | -0.0877 |
| 3 | Compute5 | 41 | 0.3415 | 0.0665 |
| 4 | Compute5 | 41 | 0.5610 | 0.1720 |
| 5 | Compute5 | 41 | 0.6341 | 0.1208 |
| 1 | Narrative-v1 | 42 | 1.0000 | 1.0667 |
| 2 | Narrative-v1 | 42 | 0.8333 | 0.3653 |
| 3 | Narrative-v1 | 41 | 0.9268 | 1.3222 |
| 4 | Narrative-v1 | 41 | 0.9756 | 0.5721 |
| 5 | Narrative-v1 | 41 | 0.7805 | 0.5611 |

## Real-Fiction Regression Assessment

Primary Data30M test loss changes -0.1316% versus Compute5 and -0.0421% versus Narrative-v1. The clean common Data10M subset changes -0.0983% and +0.0178%, respectively. Legacy clean-validation and test loss regress versus both baselines; those legacy sets are historically contaminated and diagnostic only. Overall real-fiction LM behavior is mixed and near-neutral on the primary controls, not a demonstrated gain.

Pair rankings are ordered by split, then type/distance/difficulty. Near-perfect templated-pair ranking must not be read as free-generation state retention.
