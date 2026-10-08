# Controlled Summary

**Classification: `COUNTERFACTUAL_SCORING_TRANSFER_WITHOUT_BROAD_NARRATIVE_TRANSFER`**

Counterfactual-v1 learned context-conditioned scoring beyond the sealed synthetic-shortcut baseline: joint reversal reached 48.02% on test and 44.59% on held-out templates, with 73.40% and 72.30% directional accuracy. Candidate-only X rates remained near chance, so a fixed candidate-side preference does not explain the gain.

The transfer is narrow. Latest-state reversal remained 0/32 on test and 0/14 on generalization, the historical forced-choice suite improved only to 6/13 with a negative mean margin, and conservative free-generation scoring remained 0/13 in both decoding modes. No frozen-probe family was strong on both holdouts.

Primary real-fiction losses are slightly lower than the baselines but the changes are too small and single-seed to establish a gain. Greedy repetition is worse, sampled prose still drifts, and storytelling quality did not improve.

## Decision

The experiment rejects a pure fixed-side synthetic shortcut explanation for the new benchmark result, but does not establish general narrative-state tracking. The result is best treated as generator-internal counterfactual scoring transfer with unresolved lexical/template shortcuts and no broad behavioral transfer.
