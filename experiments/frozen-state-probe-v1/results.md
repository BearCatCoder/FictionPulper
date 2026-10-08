# Frozen Narrative-State Probe v1 Results

**Classification: `STATE_WEAKLY_DECODED`**

The probes find isolated linear state signals, but these are inconsistent across families and between test and generalization holdouts. The comparison does not support `STATE_NOT_DECODED` as an absolute claim, but it also does not establish robust decoding, `STATE_DECODED_BUT_NOT_USED_IN_GENERATION`, or `NARRATIVE_V1_IMPROVED_INTERNAL_REPRESENTATION`.

## Provenance

- Raw results: `runs/frozen-state-probe-v1/results.json`
- Raw results SHA-256: `94c8d615cda96df505d203816dfbbed4e70b35dfa80d49625f588b830d040ede`
- Config: `experiments/frozen-state-probe-v1/config.yaml`
- Config SHA-256: `e36c7f98b225e72476b6b8b65ef1692419dcc95b2f4a2a0d3a6f5862871e1051`
- Compute5 checkpoint: `checkpoints/fictionpulper-15m-data30m-compute5/best-validation.pt`, SHA-256 `a9b53087a5c0dee6c2a8640999ac88d54cfb62a2d94833bc737f968f267c01fe`, tag `fictionpulper-15m-data30m-compute5`, commit `979a42e7f729de69ac57ef69ec4ac8adb622cf4a`
- Narrative-v1 checkpoint: `checkpoints/fictionpulper-15m-data30m-narrative-v1/best-validation.pt`, SHA-256 `4e88c42176ab27180b3b6992c7e96d7c5d4c529434c6a2b48b67e6c451738ad5`, tag `fictionpulper-15m-data30m-narrative-v1`, commit `71b46f8235eb3fdfc77fee782786a718f4ba5095`

Values below are mean +/- population standard deviation over probe seeds 1001, 1002, and 1003. `Gen` is the generalization holdout; `Bal acc` is balanced accuracy. Chance and majority are split-specific baselines.

## Compute5

| Family | Best layer | Split | Accuracy | Bal acc | Macro-F1 | Chance | Majority |
|---|---:|---|---:|---:|---:|---:|---:|
| character_location | layer_05 | Test | 0.5417 +/- 0.0780 | 0.5265 +/- 0.0856 | 0.5219 +/- 0.0911 | 0.5000 | 0.4375 |
| character_location | layer_05 | Gen | 0.4048 +/- 0.0891 | 0.4236 +/- 0.0687 | 0.3827 +/- 0.1169 | 0.5000 | 0.5714 |
| injury_state | layer_02 | Test | 0.3544 +/- 0.0358 | 0.3596 +/- 0.0385 | 0.3365 +/- 0.0302 | 0.3333 | 0.3291 |
| injury_state | layer_02 | Gen | 0.3143 +/- 0.0404 | 0.3142 +/- 0.0501 | 0.2851 +/- 0.0444 | 0.3333 | 0.3714 |
| knowledge_holder | layer_05 | Test | 0.3117 +/- 0.0184 | 0.2885 +/- 0.0165 | 0.2654 +/- 0.0130 | 0.3333 | 0.2727 |
| knowledge_holder | layer_05 | Gen | 0.4000 +/- 0.0467 | 0.3742 +/- 0.0558 | 0.3224 +/- 0.0813 | 0.3333 | 0.3714 |
| object_location | layer_04 | Test | 0.4392 +/- 0.0198 | 0.4787 +/- 0.0199 | 0.4254 +/- 0.0246 | 0.5000 | 0.4127 |
| object_location | layer_04 | Gen | 0.5556 +/- 0.0524 | 0.5513 +/- 0.0518 | 0.5473 +/- 0.0511 | 0.5000 | 0.4815 |
| object_ownership | layer_06 | Test | 0.2434 +/- 0.0326 | 0.2853 +/- 0.0429 | 0.2298 +/- 0.0315 | 0.3333 | 0.3651 |
| object_ownership | layer_06 | Gen | 0.3704 +/- 0.0302 | 0.3799 +/- 0.0199 | 0.3344 +/- 0.0105 | 0.3333 | 0.1111 |
| obligation_state | layer_06 | Test | 0.0903 +/- 0.0260 | 0.0980 +/- 0.0221 | 0.0829 +/- 0.0265 | 0.1667 | 0.1667 |
| obligation_state | layer_06 | Gen | 0.1111 +/- 0.0594 | 0.0979 +/- 0.0559 | 0.0859 +/- 0.0455 | 0.1667 | 0.1905 |
| relationship_state | layer_06 | Test | 0.0797 +/- 0.0102 | 0.0785 +/- 0.0087 | 0.0574 +/- 0.0087 | 0.1667 | 0.1522 |
| relationship_state | layer_06 | Gen | 0.2000 +/- 0.0408 | 0.1657 +/- 0.0356 | 0.1620 +/- 0.0209 | 0.1667 | 0.1000 |
| temporal_order | layer_08 | Test | 0.1986 +/- 0.0100 | 0.1984 +/- 0.0136 | 0.1565 +/- 0.0111 | 0.3333 | 0.3404 |
| temporal_order | layer_08 | Gen | 0.4762 +/- 0.0000 | 0.4987 +/- 0.0037 | 0.4653 +/- 0.0029 | 0.3333 | 0.3333 |

## Narrative-v1

| Family | Best layer | Split | Accuracy | Bal acc | Macro-F1 | Chance | Majority |
|---|---:|---|---:|---:|---:|---:|---:|
| character_location | final_norm | Test | 0.5625 +/- 0.0442 | 0.5450 +/- 0.0346 | 0.5081 +/- 0.0640 | 0.5000 | 0.4375 |
| character_location | final_norm | Gen | 0.4286 +/- 0.0583 | 0.4375 +/- 0.0613 | 0.3602 +/- 0.0520 | 0.5000 | 0.5714 |
| injury_state | layer_07 | Test | 0.3671 +/- 0.0575 | 0.3774 +/- 0.0586 | 0.3015 +/- 0.0263 | 0.3333 | 0.3291 |
| injury_state | layer_07 | Gen | 0.4000 +/- 0.0404 | 0.4124 +/- 0.0340 | 0.3492 +/- 0.0222 | 0.3333 | 0.3714 |
| knowledge_holder | layer_08 | Test | 0.3420 +/- 0.0061 | 0.3312 +/- 0.0206 | 0.2707 +/- 0.0059 | 0.3333 | 0.2727 |
| knowledge_holder | layer_08 | Gen | 0.3048 +/- 0.0356 | 0.3199 +/- 0.0356 | 0.2471 +/- 0.0198 | 0.3333 | 0.3714 |
| object_location | layer_01 | Test | 0.5344 +/- 0.0524 | 0.4950 +/- 0.0273 | 0.4587 +/- 0.0401 | 0.5000 | 0.4127 |
| object_location | layer_01 | Gen | 0.4815 +/- 0.0000 | 0.4744 +/- 0.0072 | 0.4269 +/- 0.0721 | 0.5000 | 0.4815 |
| object_ownership | final_norm | Test | 0.3968 +/- 0.0343 | 0.3612 +/- 0.0150 | 0.3257 +/- 0.0166 | 0.3333 | 0.3651 |
| object_ownership | final_norm | Gen | 0.2099 +/- 0.0462 | 0.3931 +/- 0.0816 | 0.1950 +/- 0.0368 | 0.3333 | 0.1111 |
| obligation_state | layer_03 | Test | 0.1319 +/- 0.0354 | 0.1378 +/- 0.0314 | 0.0696 +/- 0.0144 | 0.1667 | 0.1667 |
| obligation_state | layer_03 | Gen | 0.2063 +/- 0.0224 | 0.2196 +/- 0.0037 | 0.1147 +/- 0.0250 | 0.1667 | 0.1905 |
| relationship_state | final_norm | Test | 0.1449 +/- 0.0369 | 0.1535 +/- 0.0365 | 0.1060 +/- 0.0300 | 0.1667 | 0.1522 |
| relationship_state | final_norm | Gen | 0.1333 +/- 0.0236 | 0.1435 +/- 0.0262 | 0.0898 +/- 0.0124 | 0.1667 | 0.1000 |
| temporal_order | layer_05 | Test | 0.3262 +/- 0.0100 | 0.3117 +/- 0.0112 | 0.2492 +/- 0.0058 | 0.3333 | 0.3404 |
| temporal_order | layer_05 | Gen | 0.3333 +/- 0.0000 | 0.3360 +/- 0.0037 | 0.1943 +/- 0.0348 | 0.3333 | 0.3333 |

## Family Coverage

Supported: `object_ownership`, `object_location`, `character_location`, `knowledge_holder`, `relationship_state`, `injury_state`, `temporal_order`, `obligation_state`.

Unsupported: `goal_status` and `object_retrieval_state` have no varying valid pre-resolution labels; `cause_effect` has no varying structural class label; `primary_state_type`, `difficulty`, `genre`, and `primary_attribute` measure construction metadata or template identity rather than narrative state; `primary_entity_slot` is underspecified without a state-specific semantic role.

## Limitations

- The holdouts are post-hoc for scientific interpretation. Test and generalization did not select probe epochs or layers, but the experiment-level conclusion was formed after observing both.
- Character and location labels use case-insensitive lexical-rank slots within each story. This avoids generator-order leakage but may encode lexical-order structure and is not an invariant semantic-role policy.
- Per-family held-out support is small, especially in generalization and in individual classes for the six-class relationship and obligation families. Means and standard deviations over three probe seeds do not remove this sampling uncertainty.
