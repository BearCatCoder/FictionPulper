# FictionPulper-15M-data30M-dynamic-logit-v1

Status: preparation complete; training not started.

This controlled experiment preserves the sealed 15,047,040-parameter model,
tokenizer-v1, Data30M fiction split, Counterfactual-v1 positive curriculum,
mixed schedule, optimization policy, and 135,886,355 positive LM target
presentations. Its only training treatment is direct dynamic-state supervision
on the tied LM vocabulary logits:

```text
L = L_CE + 0.25 L_pair + 0.25 L_dynamic + 0.10 L_stale
```

The primary objective has no auxiliary state head. Dynamic candidates are
derived deterministically from sealed Counterfactual-v1 prose and sealed
State-Transition-v1 timelines. No positive prose is regenerated.

The sealed source has depth-1 decisions across all families and depth-2
decisions for `latest_state_update`. It has no depth-3 or depth-4+ prose;
those depths are therefore unavailable rather than reported as failures.

## Baseline Taxonomy

The stale-state baseline is derived from sealed post-selection scores in
`baseline-stale-state-taxonomy.json` without rerunning inference.

| Model | Split | CURRENT | PREVIOUS | Mean current-minus-stale margin |
|---|---|---:|---:|---:|
| Counterfactual-v1 | test | 28/64 (43.75%) | 36/64 (56.25%) | -0.015015 |
| Counterfactual-v1 | generalization | 14/28 (50.00%) | 14/28 (50.00%) | 0.005894 |
| State-Transition-v1 | test | 31/64 (48.44%) | 33/64 (51.56%) | -0.008196 |
| State-Transition-v1 | generalization | 14/28 (50.00%) | 14/28 (50.00%) | 0.005210 |

At depth 2, `PREVIOUS` is also `INITIAL`; precedence assigns it to
`PREVIOUS`, while alias counts preserve the initial-state interpretation.
No older or never-valid option exists in this sealed two-candidate subset.

## Commands

```bash
python -m src.dynamic_logit_annotations \
  --expected-source-manifest-sha256 41bfec4201e36a28fa5f4bea265b326ab7a0dbc67ba95c86533a1bdf191eede8 \
  --expected-transition-manifest-sha256 36372208b3b4d35523668d4292c2884eab8ae68a239bddba4a131c3e32496593 \
  --expected-tokenizer-sha256 14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012

python -m src.train_dynamic_logit \
  --config configs/data30m-15m-dynamic-logit-v1.yaml
```
