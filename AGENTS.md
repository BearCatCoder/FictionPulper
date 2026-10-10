# FictionPulper - OpenCode Project Instructions

## Current Outcome

FictionPulper is a custom decoder-only language model project for short-form
pulp fiction. The active product milestone is [issue #12]:

> Demonstrate a reproducible, coherent approximately 200-word pulp-fiction
> scene that retains opening facts and motives, advances the plot, avoids
> contradictions and loops, and has recognizable genre voice.

Track active work on the
[FictionPulper coherent-scene recovery project](https://github.com/users/BearCatCoder/projects/6/views/1).
The board status and issue dependencies are authoritative. Do not revive an old
README task as the current milestone.

[issue #12]: https://github.com/BearCatCoder/FictionPulper/issues/12

## Verified State

The following distinctions are mandatory in code, reports, and status updates:

- **Completed implementation:** the custom tokenizer/model, corpus preparation,
  document-safe packing, training, checkpointing, generation, evaluation,
  Benchmark v2 tooling, Corpus-v2 audit, Corpus-v3 pipeline, and Data100M
  preflight exist and have automated coverage.
- **Completed training:** the from-random-initialization 5M smoke, 15M, and
  50M/Data30M runs are complete. The 50M/Data30M run reached test loss `3.1018`
  and perplexity `22.24`, but showed no material storytelling improvement and
  retained `0/13` facts in its narrative-state diagnostic.
- **Completed Scene Scorecard implementation and baseline generation:** issue
  #27 froze Scene Scorecard v1 and generated the complete 40-output sealed
  50M/Data30M baseline matrix without retries, rerolls, or drops.
- **Pending human review:** Benchmark v2 generation produced 1,792
  continuations, and Scene Scorecard v1 produced 40. Independent primary human
  scoring and adjudication have not been completed for either evaluation.
  Automated lexical diagnostics and spot audits are not primary human scores.
- **Blocked future runs:** Corpus-v3 currently has 14,923,683 unsealed tokens,
  failed its genre-distribution gate, and awaits deterministic human review.
  Data100M training has not occurred and is not authorized.

Evidence is retained in:

- [`experiments/fictionpulper-5m-smoke-v1/`](experiments/fictionpulper-5m-smoke-v1/README.md)
- [`experiments/fictionpulper-50m-data30m-v1/`](experiments/fictionpulper-50m-data30m-v1/README.md)
- [`docs/benchmark-v2-baselines.md`](docs/benchmark-v2-baselines.md)
- [`docs/scene-scorecard-baseline.md`](docs/scene-scorecard-baseline.md)
- [`docs/corpus-v3.md`](docs/corpus-v3.md)
- [`experiments/fictionpulper-50m-data100m-v1/`](experiments/fictionpulper-50m-data100m-v1/README.md)

## Recovery Order

Follow this order unless the project board records a newer decision:

Issue [#27] is the completed prerequisite: Scene Scorecard v1 is frozen and the
sealed 50M/Data30M baseline generation is complete. Its independent human review
remains pending and must not be represented as complete.

1. [#8] builds a 100-300 example, rights-cleared, human-reviewed natural
   continuation pilot with reproducible split and contamination audits.
2. [#9] runs one preregistered, bounded continuation-SFT experiment on the
   sealed, validation-selected 50M/Data30M checkpoint. This is controlled
   fine-tuning of FictionPulper's own from-scratch checkpoint, not pretrained
   model use and not another random-initialization pretraining run.
3. [#12] is promoted only if independent evidence meets the frozen scene
   scorecard. Training completion, lower perplexity, or one cherry-picked output
   does not satisfy it.

Issue [#28] coordinates the outstanding independent Benchmark v2 human reviews.
It may proceed alongside the #8 pilot but remains required for primary Benchmark
v2 score claims.

The evidence-backed corpus recovery path is [#20] plus [#22], then [#21], then
[#24]. Diagnosis and lawful source discovery may overlap as allowed by those
issues. Corpus recovery is not a prerequisite for the first continuation-SFT
pilot.

Deferred work:

- [#25] Data100M training, until an approved corpus, clean comparison split,
  passing real-data preflight, and explicit experiment decision exist.
- [#10] additional dynamic-state research, until the scene pilot identifies a
  specific unresolved state-update failure.
- [#11] a general experiment registry, until a concrete reporting gap blocks
  active work.

[#8]: https://github.com/BearCatCoder/FictionPulper/issues/8
[#9]: https://github.com/BearCatCoder/FictionPulper/issues/9
[#10]: https://github.com/BearCatCoder/FictionPulper/issues/10
[#11]: https://github.com/BearCatCoder/FictionPulper/issues/11
[#20]: https://github.com/BearCatCoder/FictionPulper/issues/20
[#21]: https://github.com/BearCatCoder/FictionPulper/issues/21
[#22]: https://github.com/BearCatCoder/FictionPulper/issues/22
[#24]: https://github.com/BearCatCoder/FictionPulper/issues/24
[#25]: https://github.com/BearCatCoder/FictionPulper/issues/25
[#27]: https://github.com/BearCatCoder/FictionPulper/issues/27
[#28]: https://github.com/BearCatCoder/FictionPulper/issues/28

## Integrity Rules

1. Preserve the custom 4,096-token tokenizer, custom PyTorch model, and explicit
   configuration unless an active issue authorizes a bounded change.
2. Never load third-party pretrained transformer weights, embeddings, or a
   pretrained tokenizer. Fine-tuning a sealed FictionPulper checkpoint is
   allowed only when the active experiment protocol explicitly requires it.
3. Keep random-initialization pretraining and checkpoint-based fine-tuning
   explicitly labeled and separate.
4. Preserve document/collection/duplicate boundaries, source rights and
   provenance, canonical human-readable data, and train/validation/test
   isolation. Never train on evaluation holdouts.
5. Keep test and generalization data post-selection. Historical test sets must
   be checked against every compared model's training membership before they
   are called clean common tests.
6. Never modify or overwrite sealed corpora, manifests, splits, checkpoints,
   benchmark protocols, or experiment reports. Derived work gets a new path and
   identity.
7. Do not commit raw copyrighted data, downloaded corpora, packed binaries,
   checkpoints, run outputs, caches, credentials, or machine-local paths.
8. Report likelihood, automated diagnostics, spot audits, and independent human
   narrative scores separately. Do not claim review, training, or evaluation
   that did not occur.
9. Freeze a bounded experiment before execution. Record all outputs and failures;
   do not silently sweep data, seeds, decoding, or hyperparameters after a
   negative result.
10. Prefer correctness, reproducibility, and trustworthy data over scale.

## Development Environment

Work from the repository root with the existing CUDA-enabled environment:

```bash
source .venv/bin/activate
```

Known baseline: Python 3.14, PyTorch 2.14.1 with CUDA 13.0, NVIDIA GeForce RTX
5080. Do not reinstall CUDA or replace PyTorch without a demonstrated
compatibility problem.

The model remains a directly implemented PyTorch decoder-only Transformer using
RMSNorm, RoPE, grouped-query causal attention, SwiGLU, and tied embeddings. Do
not introduce Hugging Face `transformers`, distributed training, third-party
FlashAttention, custom CUDA kernels, a web service, or other infrastructure
unless an active issue demonstrates the need.

## Working Rules

1. Read the active issue, its dependencies, linked evidence, and current board
   status before changing files.
2. Use a focused branch and pull request. Keep changes within the issue's stated
   acceptance criteria and stop conditions.
3. Inspect existing code and artifacts before editing. Do not delete or rewrite
   functioning or historical work without explicit justification.
4. Run focused tests first and the full repository suite when the issue requires
   it. Run cheap preflight checks before expensive GPU work.
5. Keep important settings explicit and configurable. Use readable Python, type
   hints, focused modules, meaningful assertions, and clear errors.
6. Update README commands when workflows change and verify documented CLI
   options against the implementation.
7. Record out-of-scope findings as follow-up issues instead of widening work.

## Historical Smoke Milestone

FictionPulper-5M-smoke is complete, not the current task. It validated the full
data-to-generation pipeline from random initialization, including a 5,426,432
parameter model, tiny-overfit gate, 420-step BF16 run, checkpoint reload, and
recognizably fiction-like generation. Its exact historical training command,
metrics, and artifact policy remain in
[`experiments/fictionpulper-5m-smoke-v1/README.md`](experiments/fictionpulper-5m-smoke-v1/README.md).
The broader historical corpus, tokenizer, packing, and experiment reproduction
commands remain in README. Do not rerun training merely to update documentation.
