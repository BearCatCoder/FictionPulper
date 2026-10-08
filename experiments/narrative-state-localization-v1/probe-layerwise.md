# Frozen Probe

Best layers were selected on validation only; displayed held-out metrics are seed means. No family is robust on both held-out splits.

| Model | State | Best layer | Test macro-F1 | Generalization macro-F1 | Classification |
|---|---|---|---|---|---|
| Compute5 | character_location | layer_05 | 0.5219 | 0.3827 | WEAKLY_DECODED |
| Compute5 | injury_state | layer_02 | 0.3365 | 0.2851 | WEAKLY_DECODED |
| Compute5 | knowledge_holder | layer_05 | 0.2654 | 0.3224 | WEAKLY_DECODED |
| Compute5 | object_location | layer_04 | 0.4254 | 0.5473 | WEAKLY_DECODED |
| Compute5 | object_ownership | layer_06 | 0.2298 | 0.3344 | WEAKLY_DECODED |
| Compute5 | obligation_state | layer_06 | 0.0829 | 0.0859 | NOT_DECODED |
| Compute5 | relationship_state | layer_06 | 0.0574 | 0.1620 | NOT_DECODED |
| Compute5 | temporal_order | layer_08 | 0.1565 | 0.4653 | WEAKLY_DECODED |
| Contrastive | character_location | layer_04 | 0.3419 | 0.4513 | NOT_DECODED |
| Contrastive | injury_state | layer_01 | 0.2397 | 0.2378 | NOT_DECODED |
| Contrastive | knowledge_holder | layer_03 | 0.1652 | 0.2035 | NOT_DECODED |
| Contrastive | object_location | layer_01 | 0.5217 | 0.5926 | CONTRASTIVE_IMPROVED |
| Contrastive | object_ownership | layer_01 | 0.1624 | 0.1465 | WEAKLY_DECODED |
| Contrastive | obligation_state | layer_08 | 0.0701 | 0.0917 | WEAKLY_DECODED |
| Contrastive | relationship_state | layer_07 | 0.0623 | 0.1283 | WEAKLY_DECODED |
| Contrastive | temporal_order | layer_08 | 0.2818 | 0.2629 | WEAKLY_DECODED |
| Narrative | character_location | final_norm | 0.5081 | 0.3602 | WEAKLY_DECODED |
| Narrative | injury_state | layer_07 | 0.3015 | 0.3492 | WEAKLY_DECODED |
| Narrative | knowledge_holder | layer_08 | 0.2707 | 0.2471 | NOT_DECODED |
| Narrative | object_location | layer_01 | 0.4587 | 0.4269 | NOT_DECODED |
| Narrative | object_ownership | final_norm | 0.3257 | 0.1950 | WEAKLY_DECODED |
| Narrative | obligation_state | layer_03 | 0.0696 | 0.1147 | WEAKLY_DECODED |
| Narrative | relationship_state | final_norm | 0.1060 | 0.0898 | NOT_DECODED |
| Narrative | temporal_order | layer_05 | 0.2492 | 0.1943 | WEAKLY_DECODED |

Contrastive `object_location` is the sole convincing relative improvement (test 0.5217, generalization 0.5926). Probe labels are lexical-rank state labels, so this isolated result is not broad narrative-state representation. Full seed-aggregated layer and distance-bucket metrics are in `probe-layerwise.json`.
