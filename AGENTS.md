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
- **Completed continuation pilot:** issue #8 produced 200 audited examples and a
  passing `single_owner_research_pilot` approval. The owner directly reviewed 50
  selected/flagged examples; the other 150 have diagnostics but no direct human
  quality review. This is not independent validation or publication-grade
  certification.
- **Completed bounded continuation-SFT pilot:** issue #9 trained one hash-locked
  candidate from the sealed 50M/Data30M checkpoint and completed post-selection
  evaluation. Continuation test NLL improved, clean-fiction loss slightly
  regressed, automated scene diagnostics remained insufficient, and the frozen
  decision is `STOP_NO_PROMOTION`. Independent matched Scene review is pending.
- **Completed Corpus-v3 shortfall diagnosis:** issue #20 reproduced the
  4,362-record, 30,004,706-token funnel and identified 14,923,683 individually
  eligible tokens plus a 21,878,226-token provenance-only scenario. That is not
  a full upper bound or training-ready corpus; Data50M/Data100M from the current
  pool is unsupported and external yield remains unknown.
- **Completed Corpus-v3 blocker disposition:** issue #22 invalidated the
  historical rights and token-accounting pass claims, corrected fail-open audit
  defects, and reran the base-only funnel. The corrected build admits zero
  records: 1,841 fail legacy schema/provenance and 2,521 fail attributable
  rights evidence. No seal exists.
- **Completed bounded source-expansion pilot:** issue #21 verified the pinned
  bytes for the two highest-ranked issue #20 catalogs and measured 174 extracted
  candidates containing 204,805 tokenizer-v1 document tokens. No attributable
  rights or high-confidence boundary review was completed, only 35 candidates
  passed downstream quality diagnostics, and the final incremental unique yield
  is zero. The frozen decision is `stop_selected_sources`; this is measured
  negative evidence, not an approved inventory or proof that all external yield
  is zero.
- **Completed Corpus-v3 seal disposition:** issue #24 reconciled the corrected
  base and bounded source-pilot evidence and recorded exact approved supply of
  zero records and tokens. The frozen decision is `no_go_data_scaling`; no
  canonical membership, approved training split, canonical seal, configured
  Data100M pack, real-data preflight, or training handoff was produced. This is
  exact for evaluated supply, not an upper bound on all external sources.
- **Pending human review:** Benchmark v2 baseline generation produced 1,792
  continuations. Issue #28 preserves its deterministic 336-row Level A packet
  and has generated the staged 80-row Stage 1 packet, but actual owner decisions
  and scoring remain pending. Strict
  independent Benchmark review is issue #38. Scene Scorecard produced 40 base
  and 40 issue #9 candidate outputs; its independent review is also pending.
  Automated lexical diagnostics and spot audits are not primary human scores.
- **Blocked future runs:** Corpus-v3 currently has zero admitted tokens under
  the corrected gates, no seal, and no packed data. Data100M training has not
  occurred and is not authorized.

Evidence is retained in:

- [`experiments/fictionpulper-5m-smoke-v1/`](experiments/fictionpulper-5m-smoke-v1/README.md)
- [`experiments/fictionpulper-50m-data30m-v1/`](experiments/fictionpulper-50m-data30m-v1/README.md)
- [`docs/benchmark-v2-baselines.md`](docs/benchmark-v2-baselines.md)
- [`docs/benchmark-v2-owner-review-v1.md`](docs/benchmark-v2-owner-review-v1.md)
- [`docs/scene-scorecard-baseline.md`](docs/scene-scorecard-baseline.md)
- [`experiments/fictionpulper-continuation-pilot-v1/`](experiments/fictionpulper-continuation-pilot-v1/README.md)
- [`experiments/fictionpulper-50m-data30m-continuation-sft-v1/`](experiments/fictionpulper-50m-data30m-continuation-sft-v1/README.md)
- [`docs/corpus-v3.md`](docs/corpus-v3.md)
- [`experiments/fictionpulper-corpus-v3-blockers-v1/`](experiments/fictionpulper-corpus-v3-blockers-v1/README.md)
- [`experiments/fictionpulper-corpus-v3-source-pilot-v1/`](experiments/fictionpulper-corpus-v3-source-pilot-v1/README.md)
- [`experiments/fictionpulper-corpus-v3-seal-disposition-v1/`](experiments/fictionpulper-corpus-v3-seal-disposition-v1/README.md)
- [`experiments/fictionpulper-50m-data100m-v1/`](experiments/fictionpulper-50m-data100m-v1/README.md)

## Recovery Order

Follow this order unless the project board records a newer decision:

Issues [#27], [#8], and [#9] are complete. Scene Scorecard v1 and its
sealed 50M/Data30M baseline generation are complete, but its independent human
review remains pending. The continuation corpus has limited single-owner pilot
approval, not independent validation.

1. [#12] remains blocked pending independent review of issue #9's corrected
   matched base/candidate Scene packet. It is promoted only if independent
   evidence meets the frozen scene
   scorecard. Training completion, lower perplexity, or one cherry-picked output
   does not satisfy it.

Issue [#28] owns the staged Level A single-owner Benchmark v2 sample. Stage 1 is
80 rows from 20 matched quartet blocks; its review is pending. It is
non-independent, exploratory evidence and cannot support agreement,
adjudication, representativeness, power, or confirmatory claims. Strict
independent Benchmark validation remains in [#38].

Issues [#20], [#22], [#21], and [#24] are complete. Issue #21 stopped its
two-source pilot with zero admitted tokens rather than weakening evidence gates.
Issue #24 then applied its stop condition and recorded `no_go_data_scaling`
without sealing or packing. Any future corpus recovery requires a new bounded
acquisition decision that supplies a nonempty approved corpus passing every
corrected gate.
Corpus recovery is not a prerequisite for the completed first continuation-SFT
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
[#38]: https://github.com/BearCatCoder/FictionPulper/issues/38

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
