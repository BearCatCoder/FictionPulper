# FictionPulper

FictionPulper is a from-scratch decoder-only language model project for short-form pulp fiction. The 5M-parameter smoke pipeline is complete; the current controlled experiment measures the effect of expanding unique fiction data while holding the model and tokenizer fixed.

The implementation covers short-story corpus preparation, deterministic document splits, tokenizer training, indexed dataset packing, the custom Transformer, training, evaluation, checkpointing, and generation. No pretrained tokenizer or model weights are used.

## Narrative Benchmark v2

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

This freezes the protocol and scenario allocation only. No benchmark model run
or human scoring is claimed. The next step is authoring and independently
reviewing the allocated scenarios without changing the v2 protocol; historical
protocols remain immutable.

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

Generation is build-time behavior and authors all files; it does not authorize
evaluation access. The restricted loader defaults to development:

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
never training data. This dataset build did not run model training, model
evaluation, inference, checkpoint selection, or human review.

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
