# Controlled Summary

**Classification: `DIRECT_LOGIT_SUPERVISION_LEARNED_STATIC_STATE_BUT_TRANSITION_FAILED`**

Direct supervision on ordinary LM vocabulary logits learned easy static lexical
state distinctions, but did not establish repeated state updating or broad
narrative-state transfer. Aggregate CURRENT accuracy is inflated by door,
physical-condition, and relationship families; depth-2 discrimination is chance.

## Controlled Results

| Measurement | Dynamic-logit-v1 | Control | Interpretation |
|---|---:|---:|---|
| Dynamic CURRENT, test/generalization | 69.783% / 69.516% | n/a | Inflated by easy static families |
| Dynamic depth 1, test/generalization | 71.725% / 71.454% | n/a | Selective static discrimination |
| Dynamic depth 2, test/generalization | 50.781% / 50.000% | 48.438% / 50.000% State-Transition-v1 | Repeated update failed |
| CURRENT-minus-PREVIOUS margin, depth 2 | 0.0030 / 0.0188 | -0.0082 / 0.0052 State-Transition-v1 | Negligible separation |
| Counterfactual paired reversal, test/generalization | 35.258% / 40.541% | 48.024% / 44.595% Counterfactual-v1 | No transfer; regression |
| Latest-state paired reversal, test/generalization | 0% / 0% | 0% / 0% | Updating failure unchanged |
| Historical forced choice | 4/13 | 6/13 Counterfactual-v1 | No gain |
| Free-generation retention, greedy/sampled | 0/13 / 0/13 | 0/13 / 0/13 | No behavioral transfer |
| Data30M test loss | 3.244104 | 3.213676 Counterfactual-v1 | Modest non-material regression |

Depths 3 and 4+ are **unavailable**, not zero: the sealed source contains no
decisions at those depths.

## Final Questions

1. **Did direct vocabulary-logit supervision learn immediate state distinctions?** Selectively. Depth-1 accuracy is 71.725% test and 71.454% generalization, driven by door, physical-condition, and relationship state; several dynamic families remain near chance.
2. **Did it learn repeated state transitions?** No. Depth-2 CURRENT accuracy is 50.781% test and 50.000% generalization. Depths 3 and 4+ are unavailable.
3. **Did it prefer current over stale state?** Not reliably. Mean depth-2 current-minus-previous margins are only 0.0030 and 0.0188, and latest-state exact reversal remains 0%.
4. **Did direct logit supervision improve frozen representations?** No broad improvement. No supported family is strong on both test and generalization; most are weak or show no clear linear signal.
5. **Did ordinary LM scoring improve over Counterfactual-v1?** No. Paired reversal falls to 35.258% test and 40.541% generalization from 48.024% and 44.595%.
6. **Did historical forced choice improve?** No. Dynamic-logit-v1 scores 4/13 with mean correct-minus-best-negative margin -0.3985, versus Counterfactual-v1 at 6/13 and -0.1225.
7. **Did free generation improve?** No. Conservative review found 0/13 retained facts in greedy output and 0/13 in sampled output.
8. **Did entity consistency or causal/temporal progression improve?** No. Outputs abandon the tested entities and state, while latest-state reversal is 0%; no premise-driven progression is demonstrated.
9. **Was real fiction materially degraded?** No. Data30M test loss 3.244104 is modestly worse than State-Transition-v1 3.235881, Compute5 3.223102, and Counterfactual-v1 3.213676, but the one-seed difference is not material enough for `REAL_FICTION_DEGRADED`.
10. **Is the remaining bottleneck specifically autoregressive trajectory stability?** No. Current-versus-stale discrimination already fails under teacher-forced LM scoring, before autoregressive rollout can be isolated as the remaining bottleneck.

Fact removal collapses the mean signed margin from 3.1846 to -0.0091 on test
and from 3.2402 to -0.0060 on generalization. This confirms context sensitivity,
but opposite-world preference switches occur only 39.86% and 39.03% of the time;
context sensitivity alone is not correct dynamic updating.
