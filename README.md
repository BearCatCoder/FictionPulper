# FictionPulper

FictionPulper is a custom decoder-only language model project for short-form
pulp fiction. The pipeline and its from-scratch 5M, 15M, and 50M baselines are
complete. The active product milestone is a reproducible, coherent approximately
200-word scene, measured before further model or data scaling.

The implementation covers short-story corpus preparation, deterministic document splits, tokenizer training, indexed dataset packing, the custom Transformer, training, evaluation, checkpointing, and generation. No third-party pretrained tokenizer or model weights are used. The planned first recovery experiment fine-tunes FictionPulper's own sealed, from-scratch 50M/Data30M checkpoint.

## Current Status And Next Commands

The [coherent-scene recovery board](https://github.com/users/BearCatCoder/projects/6/views/1)
is the source of truth for current work and dependencies.

- **Completed implementation:** corpus preparation, the custom tokenizer/model,
  document-safe packing, training, checkpointing, generation, evaluation,
  Benchmark v2 tooling, Corpus-v3 tooling, and Data100M preflight are built and
  tested.
- **Completed training:** the from-random-initialization 5M smoke, 15M, and
  sealed 50M/Data30M runs are complete. The 50M baseline achieved test loss
  `3.1018` and perplexity `22.24`, but its controlled report found no material
  storytelling improvement and `0/13` retained diagnostic facts.
- **Completed evaluation execution, pending primary review:** Benchmark v2
  forced-choice, teacher-forced, and free-generation baseline runs are complete.
  The 1,792 continuations still require independent human scoring and
  adjudication. Automated diagnostics and a 16-output spot audit are not a
  primary human score.
- **Prepared but blocked:** Corpus-v3 contains 14,923,683 unsealed tokens. Its
  genre-distribution gate failed and deterministic review is pending. Data100M
  training has not occurred and remains deferred.
- **Completed Scene Scorecard implementation and baseline generation:**
  [issue #27](https://github.com/BearCatCoder/FictionPulper/issues/27) froze the
  separate Scene Scorecard v1 and generated the complete sealed 50M/Data30M
  baseline matrix. Independent human review remains pending; status is recorded in
  [`docs/scene-scorecard-baseline.md`](docs/scene-scorecard-baseline.md).

Activate the existing environment and reproduce the write-once scene baseline
without rerunning training:

```bash
source .venv/bin/activate
python -m src.scene_scorecard baseline
```

The command locks the checkpoint, seal, tokenizer, prompts, decoding, and all 40
prompt/trial cells before model load and refuses to overwrite an existing run.
It does not perform training. Automated diagnostics are not human scores.

## Recovery Order

Issue [#27](https://github.com/BearCatCoder/FictionPulper/issues/27) is the
completed prerequisite: the scorecard and baseline generation are complete,
while its independent human review remains pending.

1. [#8](https://github.com/BearCatCoder/FictionPulper/issues/8) then
   [#9](https://github.com/BearCatCoder/FictionPulper/issues/9): build a
   100-300-example audited natural-continuation corpus, then run one
   preregistered SFT experiment on the existing sealed 50M/Data30M checkpoint.
2. [#12](https://github.com/BearCatCoder/FictionPulper/issues/12): promote only
   on independent evidence under the frozen scorecard, not on lower perplexity,
   completed training, or a cherry-picked generation.
3. [#20](https://github.com/BearCatCoder/FictionPulper/issues/20) plus
   [#22](https://github.com/BearCatCoder/FictionPulper/issues/22), then
   [#21](https://github.com/BearCatCoder/FictionPulper/issues/21), then
   [#24](https://github.com/BearCatCoder/FictionPulper/issues/24): diagnose
   Corpus-v3 supply and blockers, pilot lawful source expansion, and seal only
   an actually approved corpus.

Issue [#28](https://github.com/BearCatCoder/FictionPulper/issues/28) coordinates
the separate Benchmark v2 human reviews and may proceed alongside the #8 pilot.

Data100M training [#25](https://github.com/BearCatCoder/FictionPulper/issues/25),
additional dynamic-state research [#10](https://github.com/BearCatCoder/FictionPulper/issues/10),
and a general registry [#11](https://github.com/BearCatCoder/FictionPulper/issues/11)
are deferred. Corpus recovery is separate from, and does not block, the first
small continuation-SFT pilot.

## Scene Scorecard v1

The issue #27 protocol is frozen separately under
`benchmarks/scene-scorecard-v1/`. It fixes five prior-exposed regression prompts,
five reserved post-selection prompts, greedy plus sampled seeds `11337`,
`21337`, and `31337`, a 150-250-word compliance range, blinded independent
review, 70% fact-retention and 80% premise-adherence thresholds, coherent-suite
gates, uncertainty reporting, and deterministic no-reroll showcase selection.

Issue #8 must exclude all post-selection prompts while building the documented
200-example pilot. Issue #9 must compare base and candidate on the identical
protocol. Issue #12 can be promoted only under these same criteria. Full details,
the concrete issue #8 pilot specification, run paths, and current review status
are in [`docs/scene-scorecard-baseline.md`](docs/scene-scorecard-baseline.md).

## Completed Narrative Benchmark v2 Work

The generation-based success criteria are frozen under
`benchmarks/narrative-v2/`. The v2 allocation defines 112 paired scenarios
balanced across seven state families and proof depths 1, 2, 3, and 4+, with
split-exclusive names, verbs, lexical pools, and templates. It separately
specifies forced-choice, teacher-forced, and free-generation evaluation,
including decoding seeds, scorer provenance, blinded human review, paired
bootstrap uncertainty, and post-selection access controls.

Validate the specification, split isolation, and balance with:

```bash
python -m src.benchmark_v2 --root benchmarks/narrative-v2
python -m unittest tests.test_benchmark_v2
```

This command validates the frozen protocol and scenario allocation. Scenario
authoring and all baseline model runs are complete; independent primary human
scoring remains pending. Historical protocols remain immutable.

Issue #2 adds deterministic authored scenarios without changing the frozen
issue #1 artifacts. Generate all three split files and run the complete
authoring audit in one command:

```bash
python -m src.narrative_v2_authoring \
  --config configs/narrative-v2-authoring.yaml
python -m unittest tests.test_narrative_v2_authoring
```

The command creates missing outputs, but never overwrites them. On a repeated
run it reproduces every expected byte and verifies the existing files; partial
or divergent outputs fail closed. The committed manifest locks the config,
generator, frozen inputs, tracked tokenizer, authored split files, explicit
candidate-only controls, computed validation summary, and audit report. Proofs
are replayed from one world-specific initial fact through exactly the allocated
number of family-specific mapping events. Three separate renderers provide
normalized template-signature isolation across splits.

Scenario generation is build-time behavior and authored all files; it does not
by itself authorize evaluation access. The restricted loader defaults to
development:

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

Development is public for scorer implementation but not weight training. Test
and generalization remain post-selection only, and hidden authored examples are
never training data. The authoring step did not run model work; the later sealed
baseline evaluation did run inference on every split as documented below.

Issue #3 adds the free-generation runner and manual-review workflow without
changing any frozen benchmark or sealed experiment artifact. The runner reads
the locked greedy and three-seed sampled protocols directly from v2, derives
the specified per-world effective seeds, retains empty/EOS outputs, and writes
raw token IDs, stop reasons, diagnostic proposition checks, and full scorer and
model provenance. Automated checks for entities, current/stale/counterfactual
facts, causal markers, contradictions, premise state, and repetition are
diagnostic only; they are never substituted for the primary human score.

Issue #4 adds seal-bound forced-choice and teacher-forced baseline execution
for the selected Compute5 and 50M checkpoints. It reports modes separately,
scores exact authored spans, runs all three candidate controls, couples A/B
world preferences, and writes hash-manifested raw outputs under ignored
`runs/`. Exact stage ordering, checkpoint/seal identities, all split commands,
and the deterministic blinded audit workflow are documented in
[`docs/benchmark-v2-baselines.md`](docs/benchmark-v2-baselines.md). Human primary
free-generation scoring remains pending until real independent reviews exist.

Run development generation for a hash-locked selected checkpoint:

```bash
python -m src.narrative_v2_free_generation generate \
  --split development \
  --checkpoint checkpoints/<run>/best-validation.pt \
  --checkpoint-sha256 <checkpoint-sha256> \
  --seal experiments/<run>/seal.json \
  --seal-sha256 <seal-sha256> \
  --tokenizer data/tokenizer/tokenizer.json \
  --tokenizer-sha256 14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012 \
  --model-id <sealed-model-id> \
  --blinded-model-id model-a \
  --output-dir runs/benchmark-v2/model-a/development
```

Test additionally requires `--checkpoint-selection-complete`.
Generalization requires both `--checkpoint-selection-complete` and
`--test-evaluation-complete`; the access-controlled loader checks these before
opening a held-out file.

Export a shuffled anonymous packet and a separate private model key:

```bash
python -m src.narrative_v2_free_generation export-review \
  --generations runs/benchmark-v2/model-a/development/raw-generations.jsonl \
  --generation-manifest runs/benchmark-v2/model-a/development/generation-manifest.json \
  --output runs/benchmark-v2/model-a/development/review-packet.jsonl \
  --key-output runs/benchmark-v2/model-a/development/review-key.json
```

Two independent reviewers complete separate packet copies. Preserve both
original rows. A disagreement or `uncertain` label requires one independent
adjudicator row whose `label` and `adjudicated_label` contain the final label.
Evidence must quote the generation verbatim; use `<EMPTY_OUTPUT>` only for an
empty generation. Scenario, world, decoding mode, and seed stay only in the
private key; the scorer verifies the original packet and key hashes before
restoring those required report fields. Score the combined immutable review
rows with:

```bash
python -m src.narrative_v2_free_generation score-reviews \
  --generations runs/benchmark-v2/model-a/development/raw-generations.jsonl \
  --generation-manifest runs/benchmark-v2/model-a/development/generation-manifest.json \
  --review-packet runs/benchmark-v2/model-a/development/review-packet.jsonl \
  --review-key runs/benchmark-v2/model-a/development/review-key.json \
  --reviews runs/benchmark-v2/model-a/development/completed-reviews.jsonl \
  --output runs/benchmark-v2/model-a/development/human-score-report.json
python -m unittest tests.test_narrative_v2_free_generation
```

The report keeps greedy and every sampled seed separate, reports sampled-seed
mean and standard deviation, and includes strict and resolved-only pair rates,
uncertain counts, reviewer agreement, Cohen's kappa, and fixed 10,000-replicate
paired-bootstrap intervals overall and by family, depth, and family/depth. No
checkpoint inference or human review was run while implementing this workflow;
those model-specific results belong to the post-selection baseline evaluation.
Implementation validation passed the benchmark specification, authored-data,
and free-generation workflow suites (30 tests) and the then-current full
repository suite (280 tests) on 2026-10-09. This is historical implementation
evidence, not the pending independent human score.

Release generation requires the ignored, locally built synthetic training
JSONL artifacts listed under `training_contamination.training_artifacts` in the
authoring config. Their hashes are locked, and recursively extracted rendered
strings are scanned for benchmark names, verbs, normalized eight-word phrases,
and template signatures. Rebuild an absent dataset with its documented project
command; hash drift fails closed and requires explicit config review. These
training artifacts remain ignored and must not be committed.

## Environment

Activate the existing CUDA-enabled environment from the repository root:

```bash
source .venv/bin/activate
```

Confirmed development environment:

- Python 3.14
- PyTorch 2.14.1 with CUDA 13.0
- NVIDIA GeForce RTX 5080

## Prepare The Corpus

The primary smoke-test source is [`Travis-ML/ShortStory-SFT-jsonl`](https://huggingface.co/datasets/Travis-ML/ShortStory-SFT-jsonl). It contains 719 complete public-domain short stories extracted from Project Gutenberg collections.

The build uses normal, non-streaming `datasets.load_dataset()` and pins the resolved Hub commit SHA in the output statistics. Only original story fields are retained. The generated `prompt_brief` and `prompt_detailed` fields are never copied into the pretraining corpus.

```bash
python -m src.prepare_corpus --output data/corpus/stories.jsonl
```

Outputs:

```text
data/corpus/stories.jsonl
data/corpus/stories.stats.json
data/corpus/stories.splits.json
data/corpus/stories.duplicates.json
```

The preparer requires `kind == "story"`, validates source word counts, preserves each story and its paragraphs, rejects duplicate IDs and cleaned texts, and writes outputs atomically. Seed `1337` assigns the 710 unique stories at document level to 604 train, 53 validation, and 53 test documents. The nine rejected duplicate rows and their conflicting attribution are retained in the machine-readable duplicate report.

The earlier `ppirli/Gutenberg-Fiction` whole-book adapter remains available for future larger-corpus work:

```bash
python -m src.prepare_corpus \
  --source gutenberg-books \
  --output data/corpus/gutenberg-books.jsonl \
  --max-documents 20
```

`zkeown/gutenberg-corpus` was inaccessible during initial local exploration. It should be revisited later rather than treated as confirmed unavailable.

## Build Corpus v1

Corpus v1 preserves all 710 seed-story IDs and adds complete stories from Project Gutenberg's public catalog and plain-text editions. It uses the existing smoke tokenizer only to estimate size; it does not retrain or modify the tokenizer.

```bash
python -m src.corpus_v1 \
  --seed-corpus data/corpus/stories.jsonl \
  --tokenizer data/tokenizer/tokenizer.json \
  --target-tokens 10000000 \
  --max-collections 1000
```

Automatic admission is deliberately conservative. A collection must be English short fiction by an author with safely closed public-domain dates, must not carry essay or poetry subjects, and must provide a table of contents whose entries match ordered body headings. Collections with weak heading coverage or multiple non-story TOC indicators go to review; capitalization alone never creates a boundary. New stories must contain 500-10,000 words and pass prose/OCR checks.

Generated outputs under ignored `data/corpus_v1/` are:

```text
corpus.jsonl             Seed plus accepted stories
additions.jsonl          Accepted Gutenberg stories only
stats.json               Counts, hashes, distributions, and stop reason
medium-review.jsonl      Review-required extraction snippets
low-confidence.jsonl     Rejected collection diagnostics
rejections.jsonl         Per-story rejection reasons
exact-duplicates.jsonl   Automatically rejected normalized duplicates
near-duplicates.jsonl    Similarity candidates; never auto-deleted
review-samples.jsonl     Boundary evidence around representative stories
```

The verified build contains 1,875 unique stories and 10,041,019 tokens estimated with the frozen smoke tokenizer. It rejected 95 exact duplicates, reported 31 near-duplicate candidates for human review, and did not admit medium- or low-confidence extraction results. Original publication years remain `null` because the Gutenberg catalog's `Issued` field is the ebook release date, not trustworthy original-publication metadata.

## Build Corpus v2

Corpus-v2 starts from the immutable, near-duplicate-resolved 1,844-story Data10M corpus and expands it to 30 million exact tokenizer-v1 document tokens. It preserves every base ID and its existing assignment as metadata, but deliberately creates no new train/validation/test assignments and does not pack data.

```bash
python -m src.corpus_v2 --config configs/corpus-v2.yaml
```

The builder verifies the locked base and tokenizer hashes before acquisition. It may read the old Gutenberg cache, writes new raw downloads only under `data/raw/gutenberg-v2/`, and writes all Corpus-v2 data, manifests, review queues, duplicate clusters, staged 15/20/25/30M checkpoints, statistics, and audits under ignored `data/corpus_v2/`. Only high-confidence ordered TOC/body matches are admitted automatically; medium- and low-confidence evidence remains queued for review.

Live acquisition is resumable. After each source, the builder atomically updates `data/corpus_v2/state/processed-sources.json` and a deterministic per-source shard under `data/corpus_v2/state/sources/`. The ledger records source identity, URL, retrieval result, raw and shard SHA-256 values, HTTP ETag/Last-Modified when supplied, processing result, accepted IDs, review/rejection outcome, timestamp, and processing fingerprint. On restart, a completed source is replayed in catalog order only after its cached bytes, URL, raw/shard checksums, and fingerprint have been verified. Changed bytes, catalog metadata, processing code/settings, tokenizer, or reviewed decisions invalidate that source's shard and cause safe reprocessing; valid completed sources are neither fetched nor extracted again. Resume timestamps and retrieval metadata cannot affect canonical records or their ordering.

Every 15M, 20M, 25M, and 30M stage is a hard gate rather than a progress marker. Its JSON report includes accepted totals and prior-stage deltas, author/source-collection cardinality and token concentration (including the actual top-20 author share), rejection/review counts, a canonical content hash/index, and explicit schema, rights, exact/near-duplicate, boundary/chapter, OCR/prose, genre, length, concentration, and canonical-content audits. Stage-specific near-duplicate edge, cluster, and unresolved evidence is retained beside the report. A missing audit, a failed audit, or any unresolved accepted near-duplicate cluster terminates the build nonzero immediately. The final build is freeze-ready only when all required stages and final audits pass, the token count remains in the locked 28-32M range, and unresolved near-duplicate clusters are zero.

The verified Corpus-v2 build is freeze-ready with 4,362 stories and 30,004,706 exact tokenizer-v1 document tokens. Its corpus SHA-256 is `0fb5e5080b3c4ee26ea178201d349d6947052f664f2bc415b5a1b9219d496087`. Review resolved 155 near-duplicate clusters by excluding 162 duplicate editions or formatting variants; this includes three newly documented correctness exclusions from the locked base while preserving the preferred historical ID in each cluster. All accepted additions are high-confidence, all automatic schema/rights/quality/chapter/concentration gates pass, and zero near-duplicate clusters remain unresolved. Original publication era remains unknown rather than being inferred from Gutenberg ebook release dates. The acquisition-only artifact is sealed by annotated tag `fictionpulper-corpus-v2-30m`; its compact tracked record is under `experiments/fictionpulper-corpus-v2-30m/`.

## Prepare The Data30M Experiment

Create deterministic 85%/7.5%/7.5% document splits from the sealed corpus:

```bash
python -m src.prepare_data30m
```

The split treats each source collection and each transitive reviewed near-duplicate cluster as indivisible. Historical Data10M assignments are preserved unless they conflict inside one of these groups; majority assignment maximizes the number retained and every changed ID is recorded. The resulting split has 3,707 train, 327 validation, and 328 test stories. All 4,362 documents, 434 constraint groups, 4,362 unique exact-text hashes, and 168 reviewed near-duplicate edges pass the leakage audit with zero cross-split violations.

Pack the split with the unchanged tokenizer-v1 and 1,024-token context:

```bash
python -m src.tokenize_corpus --config configs/data30m-15m-v1.yaml
```

Outputs are isolated under `data/packed/corpus-v2-data30m/`; existing Data10M files are not overwritten. The train split contains 27,180,978 packed tokens, 27,177,271 valid next-token targets, and 28,412 document chunks. At batch size 16 with four-step gradient accumulation, one full chunk epoch is 444 optimizer steps. The completed primary controlled experiment ran exactly three epochs, 1,332 optimizer steps, and 67 warmup steps. This presented 81,531,813 valid targets, 1.3267 times the 61,454,078 target presentations seen by the selected epoch-7 Data10M baseline.

The three Corpus-v2 seed exclusions were all historical train duplicates. Enforcing the stricter collection boundary also changes 108 of 1,841 retained historical assignments, so Data30M validation/test are a new benchmark and must not be presented as the unchanged Data10M evaluation sets. Sealed historical artifacts and metrics remain untouched.

## Audit Corpus v2

Run the read-only, hash-grounded diversity and 1,024-token boundary audit with:

```bash
python -m src.audit_corpus_v2 --config configs/corpus-v2-audit.yaml
python -m unittest tests.test_audit_corpus_v2
```

The audit verifies the acquisition seal and independently locks the later
Data30M split, tokenizer, packed metadata, and every bin/index artifact. It
never rewrites or repacks Corpus-v2. The completed report, deterministic
24-story review, priorities, limitations, and Corpus-v3 admission rules are
documented in [`docs/corpus-v2-audit.md`](docs/corpus-v2-audit.md) and tracked
under `experiments/fictionpulper-corpus-v2-audit-v1/`.

## Build Corpus v3

Corpus-v3 is an isolated, resumable, hash-pinned acquisition and deterministic
audit pipeline targeting approximately 100M tokenizer-v1 document tokens as a
feasibility goal. Run it with:

```bash
python -m src.corpus_v3 --config configs/corpus-v3.yaml
python -m unittest tests.test_corpus_v3
```

The builder prioritizes dialogue-rich compact fiction while enforcing complete
row-level rights evidence, immutable source hashes, high-confidence story
boundaries, creator and collection concentration limits, deterministic review,
and exact/fuzzy duplicate controls. Collections and duplicate clusters cannot
cross splits. It writes stage and final reports even when supply is exhausted,
but writes `seal.json` only when every freeze gate passes. The committed config
contains no new external source inventory, and no 100M build is claimed. Exact
inventory schema, review workflow, outputs, and limitations are documented in
[`docs/corpus-v3.md`](docs/corpus-v3.md).

The base-only feasibility pass found 1,947 individually eligible stories and
exactly 14,923,683 tokenizer-v1 document tokens. It reproducibly stopped without
a seal because the 57.04% unclassified token share exceeds the 35% gate and the
deterministic review is pending. It reported zero unresolved accepted near-
duplicate clusters and zero cross-split collection/duplicate leaks.

## Prepare The 50M Data100M Experiment

Issue #7 preregisters the fresh-init 50,348,544-parameter plain-LM architecture,
five-epoch exposure policy, stable 16 x 4 batch geometry, validation-only
checkpoint selection, and clean post-selection comparisons. Run its fail-closed
readiness check with:

```bash
python -m src.data100m_preflight \
  --config configs/data100m-50m-v1.yaml \
  --require-ready
```

The experiment is currently **prepared but blocked**, not trained: Corpus-v3 has
no passing seal or packed dataset. Exact blockers, the synthetic model-only BF16
hardware-gate command, and the eventual training command are documented under
[`experiments/fictionpulper-50m-data100m-v1/`](experiments/fictionpulper-50m-data100m-v1/README.md).
No validation-selected Data100M checkpoint or Data100M evaluation is claimed.

Epoch 3 was selected at Data30M validation loss 3.2461. The primary Data30M test measured loss 3.3224, perplexity 27.73, and next-token accuracy 34.89%. On a common 95-record Data10M test subset disjoint from both training sets, Data30M improved loss from 3.4802 to 3.3170 and perplexity from 32.47 to 27.58. Validation was still improving but flattening at the locked endpoint; training was not extended. Complete results are under `experiments/fictionpulper-15m-data30m-v1/`.

## Data10M Experiment

Prepare the controlled experiment corpus by resolving reviewed near duplicates and preserving retained seed assignments:

```bash
python -m src.prepare_data10m
```

Pack it separately with the frozen smoke tokenizer:

```bash
python -m src.tokenize_corpus --config configs/data10m-5m-v1.yaml
```

The packed schedule is 147 optimizer steps per epoch, 1,470 total steps over 10 complete chunk epochs, and 74 warmup steps. Run from fresh random weights with:

```bash
python -m src.train \
  --config configs/data10m-5m-v1.yaml \
  --run-id fictionpulper-5m-data10m-v1
```

The completed run selected epoch 10 at Corpus-v1 validation loss 3.4945. Corpus-v1 test loss was 3.5650; the anchored legacy test improved from loss 4.1957 / perplexity 66.40 / accuracy 23.81% to loss 3.6235 / perplexity 37.47 / accuracy 30.07%. Full artifacts are under `runs/fictionpulper-5m-data10m-v1/`, and checkpoints are isolated under `checkpoints/fictionpulper-5m-data10m-v1/`.

## Train The Tokenizer

Train the 4096-token byte-level BPE tokenizer from scratch using only the 604 training stories:

```bash
python -m src.train_tokenizer \
  --config configs/smoke-5m.yaml
```

Outputs:

```text
data/tokenizer/tokenizer.json
data/tokenizer/tokenizer.meta.json
```

The tokenizer preserves case, Unicode, and historical spelling without Unicode normalization. It reserves all required special/control tokens and records train-only efficiency metrics, all-story length diagnostics, representative tokenization examples, inefficient passages, and exact hashes.

## Pack The Dataset

Encode all three document-disjoint splits as `uint16` with indexed story boundaries:

```bash
python -m src.tokenize_corpus --config configs/smoke-5m.yaml
```

Outputs include `train.bin`, `validation.bin`, `test.bin`, one document index per split, and `data/packed/metadata.json`. Documents use:

```text
<|story|> [genre token] <|bos|> title\n\nstory text <|eos|>
```

The packed metadata derives the full-run schedule from actual data. Each document/chunk epoch visits every training chunk once, so 420 optimizer steps represent 15 actual corpus passes, not 18.55. Per epoch there are 1,483,194 valid next-token targets, 1,809,408 allocated chunk slots, 326,214 padding slots (18.03%), and 25,600 additional unused slots in the final optimizer-step capacity. The schedule remains 28 optimizer steps per epoch, 420 total steps, and 21 warmup steps.

## Model And Overfit Gate

The custom PyTorch model implements RMSNorm, RoPE, grouped-query causal attention, SwiGLU, and tied embeddings. The configured model has exactly 5,426,432 trainable parameters.

Run the required 100-sequence memorization gate:

```bash
python -m src.tiny_overfit \
  --config configs/smoke-5m.yaml \
  --sequences 100
```

The verified fixed-subset run reduced loss from 8.3586 to 0.0581, reached 99.06% non-padding next-token accuracy, and reproduced three held-in 96-token continuations exactly. Reloading the saved checkpoint produced byte-for-byte identical greedy token sequences. Its report is written to `runs/tiny-overfit/report.json`; its checkpoint is ignored under `checkpoints/tiny-overfit.pt`.

## First Smoke Run

The first locked 15-epoch run can be reproduced with:

```bash
python -m src.train \
  --config configs/smoke-5m.yaml \
  --run-id smoke-5m-20261007-v1
```

It completed all 420 optimizer steps from fresh initialization. Validation loss improved from 8.3569 to 4.0923, and the selected epoch-15 checkpoint measured test loss 4.1957. Complete metrics, generations, hashes, runtime statistics, and checkpoint identification are under `runs/smoke-5m-20261007-v1/`. The generated text is recognizably fiction-like but still repetitive and weakly conditioned, as expected from this small smoke model and corpus.

## Counterfactual Narrative v1 audit gate

Build a deterministic development artifact without opening a full 4.5M-token
construction run:

```bash
python -m src.counterfactual_narrative \
  --config configs/counterfactual-narrative-v1.yaml \
  --max-pairs 80
```

The isolated output is `data/counterfactual_narrative_v1`. A development build
always records `status: STOP` and `safe_to_train: false`, even when its structural
and shortcut audits pass. A full build (omit `--max-pairs`) may report
`safe_to_train: true` only after every invariant, split-isolation, balance,
duplicate, control, token-total, and continuation-only classifier gate passes.
The source Narrative-v1 config and tokenizer-v1 are hash-checked and never
modified. Remove a prior disposable development output before rebuilding; the
builder refuses to overwrite any existing artifact.

Run its focused tests with:

```bash
python -m unittest tests.test_counterfactual_narrative
```

Before any training on the full artifact, run the mandatory frozen-baseline
model shortcut audit:

```bash
python -m src.counterfactual_narrative.evaluator \
  --config configs/counterfactual-pretraining-audit-v1.yaml
```

This hash-verifies the tokenizer, dataset manifest and every manifested file,
plus the sealed Compute5, Narrative-v1, and Contrastive-v1 checkpoints. It
evaluates only held-out validation, test, and generalization pairs and writes
`results.json` under the ignored
`runs/fictionpulper-15m-data30m-counterfactual-v1/pretraining-audit/` directory.
The gate stops when either candidate has a context-free win rate above 55% for
any model/split. Fact-removed and irrelevant-substituted controls have one
shared context per pair, so their paired reversal success is intentionally
undefined; the audit reports chance-balanced directional behavior and the
single context's preference magnitude instead.

After the full dataset and mandatory pretraining audit both report `PASS`, pack
only the positive train/validation worlds. The packer never opens test or
generalization records:

```bash
python -m src.pack_counterfactual_v1 \
  --config configs/data30m-15m-counterfactual-v1.yaml
```

Build the fresh deterministic 85/15 Data30M/counterfactual schedule inside the
same isolated data directory:

```bash
python -m src.mixed_schedule \
  --config configs/data30m-15m-counterfactual-v1.yaml
```

The locked run starts only from random initialization, uses symmetric paired
world ranking with `lambda=0.25`, and selects checkpoints solely by Data30M
validation loss:

```bash
python -m src.train_counterfactual \
  --config configs/data30m-15m-counterfactual-v1.yaml
```

Packing and schedule generation must be completed and their generated hashes
and exact pair-presentation exposure locked in the config before training. Do
not pass a resume checkpoint; Counterfactual-v1 explicitly forbids resume and
prior model initialization.

## Tests

```bash
python -m unittest discover -s tests
```

## License

FictionPulper is licensed under the Apache License 2.0. See `LICENSE` and `NOTICE`.

## Layout

```text
configs/        Experiment configuration
data/raw/       Downloaded source material (ignored)
data/corpus/    Human-readable canonical JSONL (ignored)
data/tokenizer/ Tokenizer artifacts
data/packed/    Tokenized binary data (ignored)
src/            Pipeline source code
tests/          Lightweight tests
checkpoints/    Model checkpoints (ignored)
runs/           TensorBoard runs (ignored)
```
