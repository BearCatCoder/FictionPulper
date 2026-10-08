# Frozen Narrative-State Probe v1 Results

**Classification: `STATE_WEAKLY_DECODED`**

Signals remain isolated and inconsistent across families and held-out splits. Contrastive-v1 improves object_location on both holdouts (test macro-F1 0.5217; generalization 0.5926) but degrades or fails to broadly improve the other families, so representation change is narrow rather than broad.

## Provenance

- Raw results: `runs/frozen-state-probe-v1/results.json`
- Raw results SHA-256: `7c892105c4140cec5de8312cb30e90fa91740bc3f62726939680ac5a3e5cb269`
- Config: `experiments/frozen-state-probe-v1/config.yaml`
- Config SHA-256: `51f466eea77004ee7ae43502a24d18c4a76b5fe49ffd490bb6f6154513822fe6`
- Compute5 checkpoint: `checkpoints/fictionpulper-15m-data30m-compute5/best-validation.pt`, SHA-256 `a9b53087a5c0dee6c2a8640999ac88d54cfb62a2d94833bc737f968f267c01fe`, tag `fictionpulper-15m-data30m-compute5`, commit `979a42e7f729de69ac57ef69ec4ac8adb622cf4a`
- Narrative-v1 checkpoint: `checkpoints/fictionpulper-15m-data30m-narrative-v1/best-validation.pt`, SHA-256 `4e88c42176ab27180b3b6992c7e96d7c5d4c529434c6a2b48b67e6c451738ad5`, tag `fictionpulper-15m-data30m-narrative-v1`, commit `71b46f8235eb3fdfc77fee782786a718f4ba5095`
- Contrastive-v1 checkpoint: `checkpoints/fictionpulper-15m-data30m-contrastive-v1/best-validation.pt`, SHA-256 `5dc3bd250432334c748d125d79eeb502fb4eb7cfe4363b69a0482526de3714b4`, tag `None`, commit `dcf7c8e03702740ea0f839c1e3d17ecf92f0496b`

Values are mean +/- population standard deviation over seeds 1001, 1002, and 1003. `Gen` is the generalization holdout.

## Compute5

| Family | Best layer | Split | Accuracy | Bal acc | Macro-F1 | Chance | Majority |
|---|---:|---:|---:|---:|---:|---:|---:|
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
|---|---:|---:|---:|---:|---:|---:|---:|
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

## Contrastive-v1

| Family | Best layer | Split | Accuracy | Bal acc | Macro-F1 | Chance | Majority |
|---|---:|---:|---:|---:|---:|---:|---:|
| character_location | layer_04 | Test | 0.4479 +/- 0.0737 | 0.4061 +/- 0.0599 | 0.3419 +/- 0.0296 | 0.5000 | 0.4375 |
| character_location | layer_04 | Gen | 0.4524 +/- 0.0891 | 0.4653 +/- 0.0873 | 0.4513 +/- 0.0903 | 0.5000 | 0.5714 |
| injury_state | layer_01 | Test | 0.3291 +/- 0.0179 | 0.3314 +/- 0.0193 | 0.2397 +/- 0.0149 | 0.3333 | 0.3291 |
| injury_state | layer_01 | Gen | 0.2571 +/- 0.0404 | 0.2489 +/- 0.0336 | 0.2378 +/- 0.0273 | 0.3333 | 0.3714 |
| knowledge_holder | layer_03 | Test | 0.1991 +/- 0.0162 | 0.2206 +/- 0.0138 | 0.1652 +/- 0.0228 | 0.3333 | 0.2727 |
| knowledge_holder | layer_03 | Gen | 0.3143 +/- 0.0000 | 0.2879 +/- 0.0042 | 0.2035 +/- 0.0320 | 0.3333 | 0.3714 |
| object_location | layer_01 | Test | 0.5767 +/- 0.0270 | 0.5348 +/- 0.0346 | 0.5217 +/- 0.0445 | 0.5000 | 0.4127 |
| object_location | layer_01 | Gen | 0.6049 +/- 0.0349 | 0.6117 +/- 0.0363 | 0.5926 +/- 0.0315 | 0.5000 | 0.4815 |
| object_ownership | layer_01 | Test | 0.1958 +/- 0.0150 | 0.2247 +/- 0.0033 | 0.1624 +/- 0.0110 | 0.3333 | 0.3651 |
| object_ownership | layer_01 | Gen | 0.1852 +/- 0.0524 | 0.3519 +/- 0.0380 | 0.1465 +/- 0.0565 | 0.3333 | 0.1111 |
| obligation_state | layer_08 | Test | 0.1250 +/- 0.0170 | 0.1135 +/- 0.0215 | 0.0701 +/- 0.0070 | 0.1667 | 0.1667 |
| obligation_state | layer_08 | Gen | 0.1270 +/- 0.0594 | 0.2381 +/- 0.0506 | 0.0917 +/- 0.0480 | 0.1667 | 0.1905 |
| relationship_state | layer_07 | Test | 0.1304 +/- 0.0000 | 0.1643 +/- 0.0355 | 0.0623 +/- 0.0130 | 0.1667 | 0.1522 |
| relationship_state | layer_07 | Gen | 0.2500 +/- 0.1414 | 0.1963 +/- 0.0995 | 0.1283 +/- 0.0750 | 0.1667 | 0.1000 |
| temporal_order | layer_08 | Test | 0.3546 +/- 0.0100 | 0.3555 +/- 0.0054 | 0.2818 +/- 0.0127 | 0.3333 | 0.3404 |
| temporal_order | layer_08 | Gen | 0.3175 +/- 0.0594 | 0.3208 +/- 0.0628 | 0.2629 +/- 0.0494 | 0.3333 | 0.3333 |

## Conclusion

Contrastive-v1 changed the representation detectably only for `object_location`: its best layer is `layer_01`, with test macro-F1 `0.5217 +/- 0.0445` and generalization macro-F1 `0.5926 +/- 0.0315`. The remaining families do not show a consistent two-split improvement over both baselines. This is not broad narrative-state representation improvement.

## Limitations

- Test and generalization holdouts were evaluated post hoc after probe epoch and layer selection on validation; they were not used for selection, but the reported interpretation was formed after observing them.
- Character and location slots use case-insensitive lexical rank within each story; this policy may introduce lexical-order structure and does not identify invariant semantic roles.
- Per-family held-out support is small, especially in the generalization split and for six-class relationship and obligation labels, producing high uncertainty and unstable classwise estimates.
