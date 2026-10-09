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

Issue #1 stopped at this frozen allocation. Issue #2 implements its separate
authored scenario and audit artifacts without changing any normative v2
decision. Model runs and human reviews belong under ignored
`runs/benchmark-v2/`.

## Authored scenarios (issue #2)

`configs/narrative-v2-authoring.yaml` deterministically joins all 112 frozen
allocation rows and supplies split-exclusive source identifiers, names, verbs,
lexicons, and paraphrase forms. Each schema-compliant scenario contains shared
X/Y candidates, paired A/B worlds with one declared intervention, complete
scored continuations, free-generation review propositions, a fact-removed
control, and an irrelevant-substitution control. At depth greater than one the
premise is not the answer. Each subsequent family-specific event contains a
canonical two-branch mapping and depends on the preceding event; replay must
consume exactly N events to derive the allocated depth-N answer.

Reproduce or verify the authored artifacts and all audits with:

```bash
python -m src.narrative_v2_authoring \
  --config configs/narrative-v2-authoring.yaml
python -m unittest tests.test_narrative_v2_authoring
```

Generated files are `development.jsonl`, `test.jsonl`,
`generalization.jsonl`, one `<split>-candidate-controls.jsonl` file per split,
`authored-manifest.json`, `authoring-report.json`, and
`authoring-validation-results.md`. The manifest SHA-256-locks every generated
artifact except itself, explicitly labels that circular self-hashing is
omitted, and also locks the frozen inputs, configuration, generator, and tracked
tokenizer. Git commit identity remains `PENDING_UNTIL_COMMIT`. Existing complete
outputs are verified byte-for-byte; partial or changed outputs are rejected
rather than overwritten.

The release audit recursively implements every constraint used by the frozen
JSON Schema, replays intact proofs, proves fact-removed chains cannot start,
checks candidate frame cancellation and exact tokenizer token IDs, verifies
pool isolation, and audits normalized structural-template signatures across
contexts, continuations, candidates, prompts, review forms, propositions, and
controls. Exact/5-word-shingle near-duplicate checks cover those same authored
surfaces. Same-scenario paired worlds and prompt/context coupling are exempt
because they are intentional.

The contamination gate separates tracked synthetic generator/config provenance
from the actual ignored training JSONL consumed by historical candidates. Every
configured training artifact is mandatory and hash-locked; every JSONL record
is parsed and all nested rendered strings are checked for benchmark names,
verbs, normalized eight-word phrases, and template-signature shingles. It does
not scan generated binaries or copyrighted corpus data. Rebuild an absent
artifact with its documented curriculum command; hash drift fails closed and
requires explicit config review. Training artifacts remain ignored and must not
be committed. Computed facts are recorded in
`authoring-validation-results.md`; test command transcripts are deliberately
not generated. No model run, checkpoint selection, training, inference, human
review, or score is claimed.

Generation authors all three files as build-time behavior. Evaluation code must
use the restricted loader, which defaults to development and refuses test
without `--checkpoint-selection-complete`; generalization additionally requires
`--test-evaluation-complete`:

```bash
python -m src.narrative_v2_loader
python -m src.narrative_v2_loader --controls
python -m src.narrative_v2_loader \
  --split test --controls --checkpoint-selection-complete
python -m src.narrative_v2_loader \
  --split generalization \
  --checkpoint-selection-complete \
  --test-evaluation-complete
```

Development records may be used only for public scorer debugging, never to
train model weights. Authored hidden examples are not training data.
