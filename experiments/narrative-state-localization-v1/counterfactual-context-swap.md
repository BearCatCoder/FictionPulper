# Counterfactual Context Swap

| Model | Reversals | Direction | Signed margin | Removed |margin| | Irrelevant |margin| | Latest update |
|---|---|---|---|---|---|---|
| Compute5 | 8/56 | 0.5714 | 0.0843 | 0.5920 | 0.5928 | 0.6786 |
| Narrative | 4/56 | 0.5357 | 0.0866 | 0.6027 | 0.5996 | 0.5714 |
| Contrastive | 11/56 | 0.5982 | 0.1136 | 0.5451 | 0.5439 | 0.6250 |

| Contrastive group | Value | n | Reversal | Direction | Signed margin |
|---|---|---|---|---|---|
| family | cause_effect | 5 | 0.0000 | 0.5000 | 0.0128 |
| family | character_location | 5 | 0.0000 | 0.5000 | -0.0000 |
| family | goal | 5 | 0.0000 | 0.5000 | 0.0253 |
| family | knowledge | 7 | 0.8571 | 0.9286 | 0.4049 |
| family | object_location | 7 | 0.0000 | 0.5000 | 0.0117 |
| family | object_state | 7 | 0.0000 | 0.5000 | 0.0015 |
| family | ownership | 7 | 0.1429 | 0.5714 | 0.0341 |
| family | persistent_physical_state | 7 | 0.5714 | 0.7857 | 0.4260 |
| family | relationship | 6 | 0.0000 | 0.5000 | 0.0039 |
| distance | d064 | 9 | 0.2222 | 0.6111 | 0.1144 |
| distance | d128 | 9 | 0.1111 | 0.5556 | 0.1215 |
| distance | d256 | 9 | 0.2222 | 0.6111 | 0.1017 |
| distance | d384 | 9 | 0.2222 | 0.6111 | 0.1188 |
| distance | d512 | 9 | 0.2222 | 0.6111 | 0.1011 |
| distance | d768 | 6 | 0.0000 | 0.5000 | 0.0923 |
| distance | d900 | 5 | 0.4000 | 0.7000 | 0.1577 |

Reversal requires both context directions to rank their matching continuation, so random independent binary preferences yield 0.25; direction accuracy has chance 0.50. Removed and irrelevant controls retain large absolute preference margins, showing candidate bias rather than balanced context use. Prompts use held-out vocabulary but remain templated synthetic swaps.
