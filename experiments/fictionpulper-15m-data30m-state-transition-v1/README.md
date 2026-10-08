# FictionPulper-15M-data30M-state-transition-v1

Status: reporting complete; completion commit, tag, and final seal pending.

Primary classification: `STATIC_STATE_USE_PRESERVED_BUT_TRANSITION_FAILED`.

The locked 2,220-step run added nine training-only linear state heads to the
15,047,040-parameter LM. The heads add 6,930 parameters during training and are
absent from the equivalent 15,047,040-parameter LM-only export.

The held-out auxiliary head reports 63.019% transition, 67.845% carry, and
64.438% final-current-transition accuracy on test. Those aggregates do not
establish dynamic state updating: door, physical-condition, and relationship
families are exactly 100%, while ownership, location, goal, knowledge, and
causal families remain around chance. At transition depth 2 the head is exactly
50.000% on test and 46.429% on generalization; depths 3 and 4+ are unavailable,
not zero.

The ordinary LM regresses from Counterfactual-v1 on paired reversal
(42.249%/36.486% versus 48.024%/44.595% on test/generalization), remains 0% on
latest-state reversal, scores 4/13 rather than 6/13 forced choice, and retains
0/13 facts in both greedy and sampled generation. Real-fiction loss changes are
modest and non-material. See the tracked reports in this directory for the
controlled table, all ten final answers, exact probe interpretations, manual
fact audit, and provenance.

The selected LM-only artifact passed exact logit and generation equivalence and
has SHA-256 `a3264c8b850e86447c3eaff8191df6a78559d8adb43be3d3a12471bf6c92d72b`.

No run artifact, source, test, config, or protocol was modified while producing
these records.
