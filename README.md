# FictionPulper

FictionPulper is a from-scratch decoder-only language model project for short-form pulp fiction. The current target is a roughly 5M-parameter smoke model used to validate the complete data, tokenizer, training, checkpoint, and generation pipeline.

The current implementation covers short-story corpus preparation, deterministic document splits, tokenizer training, indexed dataset packing, the custom Transformer, and the mandatory tiny-overfit gate. No pretrained tokenizer or model weights are used. Full smoke-corpus training has not started.

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

## Tests

```bash
python -m unittest discover -s tests
```

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
