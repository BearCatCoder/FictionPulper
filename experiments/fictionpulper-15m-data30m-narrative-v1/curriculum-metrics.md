# Curriculum Metrics

Teacher-forced next-token metrics and continuation state recall measure different behavior and are reported separately.

## Test

| LM loss | Perplexity | Next-token accuracy | Valid tokens |
|---|---:|---:|---:|
| 2.593965900587492 | 13.382741118717403 | 0.5080849284478272 | 224801 |

Exact resolution recall requires complete token-ID equality from the resolution boundary. It is not teacher forced.

| Grouping | Value | Correct | Total | Accuracy |
|---|---:|---:|---:|---:|
| aggregate | all | 0 | 1405 | 0.0 |
| state_type | cause_effect | 0 | 126 | 0.0 |
| state_type | character_identity_role_relationship | 0 | 124 | 0.0 |
| state_type | goal_persistence_completion_failure | 0 | 186 | 0.0 |
| state_type | knowledge_ignorance_secret | 0 | 155 | 0.0 |
| state_type | ownership_transfer_hiding_retrieval | 0 | 219 | 0.0 |
| state_type | persistent_injury | 0 | 157 | 0.0 |
| state_type | physical_scene_location_movement | 0 | 187 | 0.0 |
| state_type | promise_debt_obligation | 0 | 127 | 0.0 |
| state_type | temporal_ordering | 0 | 124 | 0.0 |
| distance_bucket | d064 | 0 | 354 | 0.0 |
| distance_bucket | d128 | 0 | 354 | 0.0 |
| distance_bucket | d256 | 0 | 350 | 0.0 |
| distance_bucket | d384 | 0 | 194 | 0.0 |
| distance_bucket | d512 | 0 | 97 | 0.0 |
| distance_bucket | d768 | 0 | 43 | 0.0 |
| distance_bucket | d900 | 0 | 13 | 0.0 |
| difficulty | 1 | 0 | 94 | 0.0 |
| difficulty | 2 | 0 | 188 | 0.0 |
| difficulty | 3 | 0 | 282 | 0.0 |
| difficulty | 4 | 0 | 376 | 0.0 |
| difficulty | 5 | 0 | 465 | 0.0 |
| genre | adventure | 0 | 200 | 0.0 |
| genre | crime_noir | 0 | 199 | 0.0 |
| genre | fantasy_weird | 0 | 202 | 0.0 |
| genre | horror | 0 | 199 | 0.0 |
| genre | mystery | 0 | 201 | 0.0 |
| genre | science_fiction | 0 | 201 | 0.0 |
| genre | western | 0 | 203 | 0.0 |

## Generalization Holdout

| LM loss | Perplexity | Next-token accuracy | Valid tokens |
|---|---:|---:|---:|
| 2.9215267567604704 | 18.56961711646832 | 0.4811276444781996 | 100067 |

Exact resolution recall requires complete token-ID equality from the resolution boundary. It is not teacher forced.

| Grouping | Value | Correct | Total | Accuracy |
|---|---:|---:|---:|---:|
| aggregate | all | 0 | 618 | 0.0 |
| state_type | cause_effect | 0 | 55 | 0.0 |
| state_type | character_identity_role_relationship | 0 | 54 | 0.0 |
| state_type | goal_persistence_completion_failure | 0 | 82 | 0.0 |
| state_type | knowledge_ignorance_secret | 0 | 69 | 0.0 |
| state_type | ownership_transfer_hiding_retrieval | 0 | 96 | 0.0 |
| state_type | persistent_injury | 0 | 69 | 0.0 |
| state_type | physical_scene_location_movement | 0 | 83 | 0.0 |
| state_type | promise_debt_obligation | 0 | 56 | 0.0 |
| state_type | temporal_ordering | 0 | 54 | 0.0 |
| distance_bucket | d064 | 0 | 161 | 0.0 |
| distance_bucket | d128 | 0 | 154 | 0.0 |
| distance_bucket | d256 | 0 | 149 | 0.0 |
| distance_bucket | d384 | 0 | 84 | 0.0 |
| distance_bucket | d512 | 0 | 44 | 0.0 |
| distance_bucket | d768 | 0 | 20 | 0.0 |
| distance_bucket | d900 | 0 | 6 | 0.0 |
| difficulty | 1 | 0 | 42 | 0.0 |
| difficulty | 2 | 0 | 84 | 0.0 |
| difficulty | 3 | 0 | 123 | 0.0 |
| difficulty | 4 | 0 | 164 | 0.0 |
| difficulty | 5 | 0 | 205 | 0.0 |
| genre | adventure | 0 | 90 | 0.0 |
| genre | crime_noir | 0 | 87 | 0.0 |
| genre | fantasy_weird | 0 | 90 | 0.0 |
| genre | horror | 0 | 85 | 0.0 |
| genre | mystery | 0 | 86 | 0.0 |
| genre | science_fiction | 0 | 90 | 0.0 |
| genre | western | 0 | 90 | 0.0 |

## Interpretation

The low curriculum loss and roughly 48-51% teacher-forced accuracy show that the model learned synthetic curriculum token patterns. Exact recall remains zero in every state type, distance, difficulty, and genre group, including the shortest and easiest groups. The exact metric can reject valid paraphrases, but the separate human semantic audit also retained 0/13 facts under both greedy and sampled decoding. Therefore the zero is not explained only by strict wording, and the narrative-state hypothesis failed.
