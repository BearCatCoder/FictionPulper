# Narrative Benchmark v2

This directory freezes the Narrative Benchmark v2 specification for issue #1.
It does not claim that model evaluation or human review has occurred.

The allocation contains 112 paired scenarios: seven state families, four proof
depth buckets, and four scenarios per family/depth cell. Every scenario will be
evaluated independently as forced choice, teacher-forced continuation scoring,
and free narrative generation. These modes are never combined into a composite
score.

`protocol.json` is normative. `scenario-allocation.jsonl` freezes IDs, splits,
coverage, answer orientation, candidate order, and split-exclusive name, verb,
lexical, and template pools before scenario prose is authored. Full authored
records must satisfy `scenario.schema.json`. Test and generalization content is
post-selection only. The generalization split jointly holds out names, verbs,
and templates. Before release, its required contamination audit must also prove
those surfaces absent from all synthetic training curricula used by a model.

Validate the frozen artifacts and print balance/leakage results:

```bash
python -m src.benchmark_v2 --root benchmarks/narrative-v2
python -m unittest tests.test_benchmark_v2
```

Expected allocation validation summary: 112 scenarios, split counts 28/28/56, 28 complete
family/depth cells with four scenarios each, 56/56 candidate presentation
orders, 56/56 world-A labels, and no cross-split reuse of reserved pool or
template IDs. Actual normalized-text and pool-member leakage checks are a
mandatory release gate after scenario prose is authored.

The actual specification-validation transcript is recorded in
`validation-results.md`.

Next implementation step: author and independently review the 112 scenario
pairs in separate development, test, and generalization files, tokenize them
with a frozen tokenizer, run exact/near-duplicate and training-contamination
audits, then add a hash-locked split manifest without changing any normative v2
decision. Model runs and human reviews belong under ignored
`runs/benchmark-v2/`.
