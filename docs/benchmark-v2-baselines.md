# Benchmark v2 baseline evaluation

Issue #4 evaluates only the validation-selected, sealed Compute5 and 50M
checkpoints. The evaluator has a closed registry of checkpoint, tokenizer, and
seal hashes. It verifies the seal status and run identity, verifies the seal's
selected checkpoint and tokenizer, and for 50M verifies its recorded Compute5
parent seal. A mismatch fails before model loading. Output directories are
write-once and belong under ignored `runs/benchmark-v2/`.

## Stage order

Run the stages in this order for both models. Do not use development results to
change checkpoint identity. Test access attests that selection is complete;
generalization access additionally attests that test evaluation is complete.

1. Validate frozen benchmark artifacts and authored-data hashes.
2. Run development forced-choice and teacher-forced scoring.
3. Run test forced-choice and teacher-forced scoring.
4. Run generalization forced-choice and teacher-forced scoring.
5. Run the matched free-generation protocol for each split and model.
6. Export deterministic blinded review packets.
7. Obtain two independent human reviews plus independent adjudication where required.
8. Score completed reviews. Until then, human primary scoring is `pending`.

No cross-mode composite is permitted. Held-out fiction LM metrics remain a
separate evaluation family. Historical legacy validation/test subsets overlap
Data30M training and are exposure-contaminated continuity diagnostics, not
clean held-out baseline evidence.

## Automated modes

The command writes full token-level raw exports, separate forced-choice and
teacher-forced summaries, and a manifest containing every output hash, exact
artifact/seal/checkpoint/tokenizer identities, environment and Git provenance,
and the reproduction command.

```bash
python -m src.benchmark_v2 --root benchmarks/narrative-v2

python -m src.narrative_v2_baseline \
  --model compute5 --split development \
  --output-dir runs/benchmark-v2/compute5/development/sealed-v1
python -m src.narrative_v2_baseline \
  --model 50m --split development \
  --output-dir runs/benchmark-v2/50m/development/sealed-v1

python -m src.narrative_v2_baseline \
  --model compute5 --split test --checkpoint-selection-complete \
  --output-dir runs/benchmark-v2/compute5/test/sealed-v1
python -m src.narrative_v2_baseline \
  --model 50m --split test --checkpoint-selection-complete \
  --output-dir runs/benchmark-v2/50m/test/sealed-v1

python -m src.narrative_v2_baseline \
  --model compute5 --split generalization \
  --checkpoint-selection-complete --test-evaluation-complete \
  --output-dir runs/benchmark-v2/compute5/generalization/sealed-v1
python -m src.narrative_v2_baseline \
  --model 50m --split generalization \
  --checkpoint-selection-complete --test-evaluation-complete \
  --output-dir runs/benchmark-v2/50m/generalization/sealed-v1
```

Forced choice computes the arithmetic mean of token log probabilities over
only each complete, authored candidate span. Main A/B worlds are scored as a
coupled reversal pair. Candidate-only, fact-removed, and
irrelevant-substituted contexts are reported separately as bias controls.
Teacher forcing scores every token in each complete gold and matched
counterfactual continuation while excluding prompt tokens. Both modes report
overall, family, and depth `1`, `2`, `3`, and `4_plus` strata.

## Free generation and audit

Use the already frozen decoding implementation. The seal arguments are
mandatory. For 50M substitute its paths/hashes and use a different blinded ID.
Add the loader access attestations for test and generalization exactly as above.

```bash
python -m src.narrative_v2_free_generation generate \
  --split development \
  --checkpoint checkpoints/fictionpulper-15m-data30m-compute5/best-validation.pt \
  --checkpoint-sha256 a9b53087a5c0dee6c2a8640999ac88d54cfb62a2d94833bc737f968f267c01fe \
  --seal experiments/fictionpulper-15m-data30m-compute5/seal.json \
  --seal-sha256 10fcd7a62b819bd2af9a9fb4f5bf86df81558130f0e6f01af44c48491185db5c \
  --tokenizer data/tokenizer/tokenizer.json \
  --tokenizer-sha256 14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012 \
  --model-id fictionpulper-15m-data30m-compute5 \
  --blinded-model-id baseline-a \
  --output-dir runs/benchmark-v2/compute5/development/free-generation

python -m src.narrative_v2_free_generation export-review \
  --generations runs/benchmark-v2/compute5/development/free-generation/raw-generations.jsonl \
  --generation-manifest runs/benchmark-v2/compute5/development/free-generation/generation-manifest.json \
  --output runs/benchmark-v2/compute5/development/free-generation/review-packet.jsonl \
  --key-output runs/benchmark-v2/compute5/development/free-generation/review-key.json
```

The packet order is deterministic and hides scenario, world, seed, and model
mapping in a separate key. Automated fact-retention, causal, contradiction,
entity, and repetition measures are diagnostic only. Do not create review rows
without actual reviewers. Preserve each original row and follow the frozen
rubric before running `score-reviews` as documented in the repository README.

## Issue #4 execution record

On 2026-10-09 all three evaluation modes were run for both sealed checkpoints
on all three splits with CUDA BF16. The write-once forced-choice and
teacher-forced raw records, summaries, and manifests are under
`runs/benchmark-v2/{compute5,50m}/{development,test,generalization}/sealed-v1/`.
The full free-generation exports, diagnostic summaries, generation manifests,
blinded review packets, and separate private review keys are under the sibling
`free-generation/` directories. These files are intentionally ignored; each
manifest records its SHA-256 hashes and exact provenance. The tracked worktree
was dirty because this issue implementation was under development, and the
manifests record that fact rather than claiming a clean commit.

| Model | Split | FC world accuracy | FC paired reversal | TF world preference | TF paired preference |
|---|---:|---:|---:|---:|---:|
| Compute5 | development | 50.00% | 0.00% | 66.07% | 32.14% |
| Compute5 | test | 48.21% | 0.00% | 0.00% | 0.00% |
| Compute5 | generalization | 50.00% | 0.00% | 33.04% | 0.00% |
| 50M | development | 58.93% | 17.86% | 100.00% | 100.00% |
| 50M | test | 51.79% | 3.57% | 0.00% | 0.00% |
| 50M | generalization | 50.00% | 0.00% | 28.57% | 0.00% |

These values are baseline observations, not a composite decision. Full
family/depth and control results remain in the manifests' hashed summary
artifacts. The sharp teacher-forced split behavior is not interpreted as state
reasoning: complete continuations include split-exclusive prose surfaces, while
forced choice scores only the decision-bearing candidate.

Free generation produced 1,792 continuations, 896 for each checkpoint. Every
scenario/world has one greedy continuation and one continuation for each of the
three locked sampled seeds. The raw export hashes are:

| Model | Split | Generations | Raw export SHA-256 |
|---|---:|---:|---|
| Compute5 | development | 224 | `5751eeb5416b2d811c1618a2b67d6df4c29a46a0be655aa9039468b94a675cda` |
| Compute5 | test | 224 | `eda12cc92ab394bfd459e46e880caf77389a7bd76e34991dc6adc9318ec7ea57` |
| Compute5 | generalization | 448 | `938d36ef5db52e82fcf703064366843c507373099ec806ea3de18de340660e4c` |
| 50M | development | 224 | `1c60fda21eb72a38ab00c29d1ca59270de6770f7b5a0bdf216972daedd829544` |
| 50M | test | 224 | `9cd15c014010752f2a0b2f6a0349c3a689889434c82ee90ca10a9568ed18ae2c` |
| 50M | generalization | 448 | `1ce9016936925f4699184969cf6b7448ed14243b6d5d2fe2d42d6358cdaf0233` |

The automated free-generation diagnostics found zero overall
current-fact-only and premise-retention hits in every model/split/trial cell.
Greedy mean repeated-4-gram rates ranged from 44.73% to 74.57%; sampled rates
ranged from 1.48% to 25.36%. Named-entity retention remained at or below 9.82%
in every overall cell. These lexical checks are diagnostic and can miss both
valid paraphrases and implicit contradictions, so they are not treated as the
primary score.

The six blinded review packets contain all 1,792 continuations and are ready
for two independent reviewers. Human pair-consistency scoring and adjudication
remain `pending`; no review rows or primary human result were fabricated.

A limited manual quality-control audit inspected the first eight rows in each
deterministically shuffled development review packet (16 continuations total,
with checkpoint identities still represented only as `baseline-a` and
`baseline-b`). None of the 16 continuations stated or preserved its required
proposition, all 16 drifted away from the authored premise, and several entered
obvious phrase loops. This spot audit confirms that the zero automated
premise-retention signal is not merely a reporting accident, but it is not an
independent two-reviewer assessment, does not estimate the primary metric, and
must not be generalized as a scored sample.
