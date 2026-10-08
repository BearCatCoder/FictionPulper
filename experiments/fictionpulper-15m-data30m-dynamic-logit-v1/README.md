# FictionPulper-15M-data30M-dynamic-logit-v1

Status: reporting complete; completion commit, tag, and final seal pending.

Primary classification: `DIRECT_LOGIT_SUPERVISION_LEARNED_STATIC_STATE_BUT_TRANSITION_FAILED`.

This controlled run applied pair, dynamic-current, and stale-state losses directly
to the tied LM vocabulary logits. It preserved the locked 15,047,040-parameter
architecture, tokenizer, Data30M corpus, curriculum exposure, schedule, seed,
optimizer, and 2,220-step budget. No auxiliary state head was used.

The treatment learned easy static distinctions: door and relationship state are
100% accurate on both held-out splits, and persistent physical state is 100% on
test and 95% on generalization. It did not learn repeated updates. Depth-2
CURRENT selection is 50.781% on test and 50.000% on generalization, with mean
current-minus-previous margins of 0.0030 and 0.0188. Depths 3 and 4+ are
unavailable, not zero.

Exact latest-state paired reversal remains 0% on both splits. Static
counterfactual reversal is 35.258%/40.541%, below Counterfactual-v1 at
48.024%/44.595%. The candidate scores 4/13 forced choice and retains 0/13 facts
in both greedy and sampled generation. Real-fiction loss is modestly worse but
not materially degraded. Frozen probes show no broad two-split improvement.

See `controlled-summary.md` for all ten final answers and
`manual-fact-scoring.json` for the conservative semantic audit.

Key training command:

```bash
python -m src.train_dynamic_logit \
  --config configs/data30m-15m-dynamic-logit-v1.yaml
```

The held-out suite was run only after step 2,220 was selected by Data30M
validation loss. The selected checkpoint SHA-256 is
`1d2ac1947afd10436729b007ae5d1fbeb9ef0815fc507ee15ab1a9e8e5c43575`.
