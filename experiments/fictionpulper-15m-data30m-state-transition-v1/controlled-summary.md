# Controlled Summary

**Classification: `STATIC_STATE_USE_PRESERVED_BUT_TRANSITION_FAILED`**

The state-supervised run preserves easy static-state use in the auxiliary head,
but it does not learn reliable state transitions. Aggregate auxiliary accuracy
is driven by lexical static door, physical-condition, and relationship families
at 100%; ownership, object/character location, goal, knowledge, and causal
families remain around chance. Depth-2 transitions fall to chance or below.

## Controlled Results

| Measurement level | Direct result | Counterfactual-v1 control | Interpretation |
|---|---:|---:|---|
| Auxiliary head, test transition | 63.019% | n/a | Aggregate inflated by three 100% static families |
| Auxiliary head, test carry | 67.845% | n/a | Static lexical state is carried |
| Auxiliary head, test final current transition | 64.438% | n/a | Not broad current-state updating |
| Auxiliary head, depth-2 test/generalization | 50.000% / 46.429% | n/a | Dynamic transition failed |
| LM scoring, paired reversal test/generalization | 42.249% / 36.486% | 48.024% / 44.595% | No transfer; regression from control |
| LM scoring, latest-state reversal test/generalization | 0% / 0% | 0% / 0% | Updating failure unchanged |
| LM scoring, historical forced choice | 4/13 | 6/13 | No historical-suite gain |
| Free generation, greedy fact retention | 0/13 | 0/13 | No expected fact explicitly preserved |
| Free generation, sampled fact retention | 0/13 | 0/13 | No expected fact explicitly preserved |

Depths 3 and 4+ are **unavailable**, not zero: neither holdout contains an
entity sequence with that transition depth.

## Final Questions

1. **Did the model learn immediate state transitions?** Only selectively. Aggregate transition accuracy is 63.019%, but the gain is concentrated in three trivially lexical static families; dynamic families remain around chance.
2. **Did it learn repeated state transitions?** No. Depth 2 is exactly 50.000% on test and 46.429% on generalization. Depths 3 and 4+ are unavailable.
3. **Did it learn current state, not stale state?** No. Final-current-transition accuracy is 64.438%, but latest-state LM reversal remains 0% and the apparent auxiliary gain is not broad.
4. **Did the auxiliary heads improve frozen representations?** No. No supported probe family is strong on both holdouts, and the exact family interpretations are mixed weak/no-clear signals.
5. **Did LM scoring improve over Counterfactual-v1?** No. Paired reversal is 42.249% test and 36.486% generalization versus Counterfactual-v1 at 48.024% and 44.595%.
6. **Did free generation improve?** No. Manual review of every candidate output found 0/13 retained under greedy and 0/13 under sampled decoding; raw text explicitly preserves none of the expected facts.
7. **Did entity consistency improve?** No. Generated continuations abandon or mutate the tested names and entity-linked possession, relationship, location, and goal state.
8. **Did causal or temporal progression improve?** No. Causal-state accuracy is near chance, depth-2 transitions fail, and generations do not preserve disclosed causes or goals.
9. **Was real fiction materially degraded?** No. Data30M test loss 3.235881 is modestly worse than Counterfactual-v1 3.213676 and Compute5 3.223102, but the one-seed differences are non-material; classification is not `REAL_FICTION_DEGRADED`.
10. **Is the remaining bottleneck now specifically autoregressive trajectory stability?** Not yet. Current-state updating and transfer into LM logits already failed under auxiliary and teacher-forced scoring, so autoregressive trajectory stability cannot yet be isolated as the remaining failure.

The LM-only export equivalence check is `PASS`: state heads are absent, logits
and generation are identical, and the export SHA-256 is
`a3264c8b850e86447c3eaff8191df6a78559d8adb43be3d3a12471bf6c92d72b`.
