# FictionPulper — OpenCode Project Instructions

## Project Mission

**FictionPulper** is a from-scratch decoder-only language model project focused on generating short-form general fiction in the spirit of classic pulp magazines.

The long-term target is a specialized ~50M parameter model capable of producing entertaining, coherent short fiction across genres such as:

- crime
- noir
- mystery
- horror
- science fiction
- western
- adventure
- fantasy
- weird fiction

The **current milestone is not the 50M model**.

The current milestone is:

> **FictionPulper-5M-smoke** — a small model trained from random initialization whose purpose is to validate the entire data → tokenizer → model → training → checkpoint → generation pipeline.

The smoke model should be genuinely trained from scratch. Do not load pretrained transformer weights, pretrained embeddings, or a pretrained tokenizer.

---

# Development Environment

Repository location:

```text
~/Dev/FictionPulper
```

Python environment:

```text
~/Dev/FictionPulper/.venv
```

Activate it with:

```bash
cd ~/Dev/FictionPulper
source .venv/bin/activate
```

Current environment is already installed and CUDA-enabled.

Known baseline:

```text
Python:         3.14
PyTorch:        2.14.1+cu130
CUDA runtime:   13.0
GPU:            NVIDIA GeForce RTX 5080
VRAM:           15.47 GiB
```

Installed project dependencies include:

```text
torch
datasets
tokenizers
huggingface_hub
numpy
pyarrow
tqdm
pyyaml
tensorboard
```

Do not reinstall CUDA or replace the existing PyTorch environment unless there is a demonstrated compatibility problem.

---

# Guiding Principles

1. Keep the first implementation simple and understandable.
2. Train all model weights from random initialization.
3. Train the tokenizer specifically for FictionPulper.
4. Preserve story/document boundaries in the source corpus.
5. Keep raw/cleaned human-readable data separate from tokenized binary data.
6. Make all important hyperparameters configurable.
7. Make experiments reproducible.
8. Validate the pipeline with a small dataset before scaling it.
9. Favor correctness and observability over premature optimization.
10. Avoid unnecessary frameworks. PyTorch should own the model and training loop.

Do not use Hugging Face `transformers` for the model implementation unless there is a compelling later reason.

---

# Current Milestone

## FictionPulper-5M-smoke

The smoke model exists to answer these questions:

- Can we download and curate a fiction corpus?
- Can we train a custom tokenizer?
- Can we tokenize and pack the dataset correctly?
- Can a custom decoder-only Transformer train successfully on the RTX 5080?
- Does training loss decrease?
- Does validation loss behave reasonably?
- Can checkpoints be saved and restored?
- Can the model generate increasingly plausible English fiction?
- Can the experiment be reproduced from configuration files?

The smoke model does **not** need to generate excellent stories.

A successful smoke test is one in which the model demonstrably learns.

---

# Target Architecture

Implement a modern decoder-only Transformer inspired by current small LLM architectures.

Initial target:

```yaml
model:
  vocab_size: 4096
  hidden_size: 256
  num_layers: 6
  num_attention_heads: 8
  num_key_value_heads: 2
  intermediate_size: 768
  max_seq_len: 1024

  activation: swiglu
  normalization: rmsnorm
  positional_encoding: rope
  tie_word_embeddings: true
  attention_bias: false
  mlp_bias: false
```

The exact parameter count does not need to be exactly 5,000,000.

Target range:

```text
4.5M–5.5M trainable parameters
```

The program must print the exact trainable parameter count at startup.

If the above dimensions fall outside the target after implementation, adjust `num_layers` or `intermediate_size`, not the vocabulary unless necessary.

---

# Model Components

Implement the model directly in PyTorch.

Required components:

```text
TokenEmbedding
RMSNorm
RotaryPositionEmbedding / RoPE
Grouped-Query Self Attention
SwiGLU MLP
TransformerBlock
Final RMSNorm
Language Model Head
```

Input embeddings and output projection should share weights.

Conceptually:

```text
tokens
  │
  ▼
embedding
  │
  ▼
┌─────────────────────┐
│ RMSNorm             │
│ GQA causal attention│
│ residual            │
│ RMSNorm             │
│ SwiGLU MLP          │
│ residual            │
└─────────────────────┘
        × N layers
  │
  ▼
RMSNorm
  │
  ▼
tied LM head
  │
  ▼
next-token logits
```

Use causal masking.

Prefer PyTorch scaled-dot-product attention when practical.

---

# Tokenizer

Train a tokenizer from scratch.

Initial tokenizer:

```yaml
tokenizer:
  type: BPE
  vocab_size: 4096
```

Required special tokens:

```text
<|pad|>
<|bos|>
<|eos|>
<|unk|>
```

Reserve FictionPulper control tokens for future conditioning:

```text
<|story|>
<|crime|>
<|noir|>
<|mystery|>
<|horror|>
<|science_fiction|>
<|western|>
<|adventure|>
<|fantasy|>
<|weird|>
```

These may be included in the smoke tokenizer even if genre metadata is incomplete initially.

The tokenizer artifacts should live under:

```text
data/tokenizer/
```

The tokenizer must be deterministic and reloadable.

---

# Dataset Strategy

The smoke test should use **public-domain English fiction**.

Preferred initial source:

```text
zkeown/gutenberg-corpus
```

Use Hugging Face `datasets` to access the corpus.

The smoke corpus should emphasize:

```text
short stories
crime
detective / mystery
horror
science fiction
western
adventure
fantasy
ghost stories
weird fiction
sea stories
```

Do not blindly dump all Gutenberg text into training.

Filter aggressively toward fiction.

---

# Smoke Corpus Size

The first useful corpus should target approximately:

```text
10 million tokenizer tokens
```

Approximate split:

```text
Train:       85%
Validation:   7.5%
Test:         7.5%
```

The split must happen at the **document/story level**, never by slicing one story across train and validation/test.

Avoid data leakage.

---

# Canonical Corpus Format

Cleaned source data should be stored as JSONL before tokenization.

Example:

```json
{
  "id": "pg-12345-story-03",
  "source": "project_gutenberg",
  "title": "Example Story",
  "author": "Example Author",
  "genres": ["mystery"],
  "publication_year": 1919,
  "rights": "public_domain_us",
  "text": "The rain began shortly after midnight..."
}
```

Use `null` for unavailable metadata.

The canonical corpus must remain human-readable.

Do not make `.bin` token files the only preserved copy of the dataset.

---

# Corpus Cleaning

Cleaning should be conservative.

Remove obvious non-story material such as:

- Gutenberg boilerplate
- tables of contents when identifiable
- scanner/OCR junk
- repeated page headers
- repeated page footers
- obvious advertisements
- navigation artifacts
- obvious legal/license boilerplate already removed by upstream processing
- duplicated documents

Preserve:

- punctuation
- dialogue
- paragraph boundaries
- scene breaks
- capitalization
- stylistic spelling
- story titles where useful

Do not aggressively "modernize" old prose.

The goal is fiction, not sanitized contemporary English.

---

# Story Boundaries

Document boundaries are important.

Each individual story should ideally be represented as:

```text
<|story|>
<genre-token-if-known>
<|bos|>

TITLE

story text...

<|eos|>
```

Collections containing multiple stories should eventually be split into individual stories when reliable boundaries can be detected.

For the smoke milestone, full collection-level documents are acceptable when clean story splitting is not reliable.

Do not invent story boundaries with brittle heuristics that silently destroy text.

---

# Tokenized Dataset

After the tokenizer is trained, convert the corpus to token IDs.

Use a compact integer representation such as:

```text
uint16
```

because a 4096-token vocabulary fits comfortably inside 16 bits.

Expected outputs:

```text
data/packed/train.bin
data/packed/validation.bin
data/packed/test.bin
```

Also save metadata describing:

- token count
- document count
- tokenizer hash/version
- split sizes
- preprocessing version

Never allow training windows to cross an `<|eos|>` boundary unless the loader intentionally includes EOS between documents.

---

# Repository Structure

Create or maintain this structure:

```text
FictionPulper/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── smoke-5m.yaml
│
├── data/
│   ├── raw/
│   ├── corpus/
│   ├── tokenizer/
│   └── packed/
│
├── src/
│   ├── prepare_corpus.py
│   ├── train_tokenizer.py
│   ├── tokenize_corpus.py
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── generate.py
│   └── evaluate.py
│
├── tests/
│   ├── test_model.py
│   ├── test_tokenizer.py
│   └── test_dataset.py
│
├── checkpoints/
└── runs/
```

Large generated artifacts should not be committed to Git.

---

# Git Ignore Requirements

`.gitignore` should at minimum exclude:

```gitignore
.venv/
__pycache__/
*.pyc

data/raw/
data/corpus/
data/packed/

checkpoints/
runs/

*.bin
*.pt
*.pth
*.ckpt

.DS_Store
```

Tokenizer configuration files may be committed if their licensing/source permits it.

Small metadata files describing dataset preparation may also be committed.

---

# Configuration

Put experiment settings in:

```text
configs/smoke-5m.yaml
```

The training scripts should not bury important values as unexplained constants.

Configuration should cover at least:

```yaml
seed:

model:
  vocab_size:
  hidden_size:
  num_layers:
  num_attention_heads:
  num_key_value_heads:
  intermediate_size:
  max_seq_len:

training:
  batch_size:
  gradient_accumulation_steps:
  max_steps:
  learning_rate:
  min_learning_rate:
  warmup_steps:
  weight_decay:
  grad_clip:
  eval_interval:
  eval_steps:
  checkpoint_interval:
  precision:

data:
  train_path:
  validation_path:
  test_path:

logging:
  tensorboard_dir:
```

---

# Training Defaults

Use sensible initial defaults rather than over-optimizing.

Suggested starting point:

```yaml
training:
  optimizer: adamw
  learning_rate: 3.0e-4
  min_learning_rate: 3.0e-5
  weight_decay: 0.1
  grad_clip: 1.0

  batch_size: 16
  gradient_accumulation_steps: 4

  warmup_steps: 200
  max_steps: 5000

  eval_interval: 250
  checkpoint_interval: 500

  precision: bf16
```

Use cosine learning-rate decay after warmup.

RTX 5080 supports BF16; use BF16 unless testing reveals a concrete reason not to.

Do not use FP16 merely because older examples do.

---

# Training Observability

Training output should report:

```text
step
training loss
validation loss
learning rate
tokens/sec
samples/sec
GPU memory allocated
GPU memory reserved
elapsed time
total tokens processed
```

TensorBoard should record at least:

```text
train/loss
validation/loss
train/learning_rate
train/tokens_per_second
train/gpu_memory_gb
```

Run TensorBoard with:

```bash
tensorboard --logdir runs
```

---

# Checkpointing

Checkpoints must contain enough information to resume training.

Save:

```text
model state
optimizer state
scheduler state
current step
configuration
random seed / RNG state where practical
tokenizer identifier/hash
```

Support:

```bash
python -m src.train --config configs/smoke-5m.yaml
```

and eventually:

```bash
python -m src.train \
  --config configs/smoke-5m.yaml \
  --resume checkpoints/step-XXXX
```

Checkpoint saving must be atomic enough that an interrupted save does not destroy the previous good checkpoint.

---

# Generation

Provide a simple CLI.

Target usage:

```bash
python -m src.generate \
  --checkpoint checkpoints/best.pt \
  --prompt "The rain began shortly after midnight."
```

Support generation controls:

```text
max_new_tokens
temperature
top_k
top_p
seed
```

Provide reasonable defaults.

The generation script must load the exact tokenizer associated with the model.

---

# Evaluation

Smoke-test evaluation does not need sophisticated literary scoring.

At minimum calculate:

```text
validation cross entropy
validation perplexity
test cross entropy
test perplexity
```

Also maintain a small fixed prompt suite so generations can be compared across checkpoints.

Example prompts:

```text
The man in the gray coat entered the station just before midnight.

There was something moving beneath the floorboards.

The rocket had been silent for three days when the signal arrived.

Nobody in Dry Creek had seen the stranger before Tuesday.

Detective Mallory looked at the locked door and knew somebody was lying.
```

Use the same prompts and generation seed when comparing checkpoints.

---

# Tests

Add lightweight tests before long training runs.

Tests should verify:

### Model

- forward pass works
- logits shape is correct
- loss can be computed
- gradients propagate
- parameter count is within expected range
- causal masking prevents future-token visibility
- tied embedding weights really are shared

### Tokenizer

- required special tokens exist
- encode/decode round trip is reasonable
- tokenizer can be saved and loaded
- vocabulary size matches configuration

### Dataset

- correct sequence length
- token IDs stay within vocabulary range
- train/validation/test are separate
- batching works
- EOS handling works

---

# First End-to-End Sanity Test

Before using the full 10M-token corpus, intentionally overfit the model on a tiny sample.

For example:

```text
100–500 training sequences
```

Train until loss becomes very low.

This proves:

- model wiring is correct
- loss is correct
- optimizer updates parameters
- dataloader is correct
- checkpointing works
- inference loads trained weights

If the model cannot overfit a tiny dataset, do **not** proceed to full training.

Find the bug first.

---

# Implementation Order

Work in this order unless a concrete dependency requires otherwise.

## Phase 1 — Repository foundation

1. Create repository directories.
2. Add `.gitignore`.
3. Add initial README.
4. Add `configs/smoke-5m.yaml`.
5. Add reproducibility seed handling.
6. Confirm CUDA from inside project environment.

## Phase 2 — Corpus preparation

1. Implement `prepare_corpus.py`.
2. Load Gutenberg corpus.
3. Inspect actual schema before assuming field names.
4. Filter English fiction.
5. Filter toward target genres.
6. Clean text conservatively.
7. Deduplicate obvious duplicate records.
8. Write canonical JSONL.
9. Produce corpus statistics.
10. Stop after enough text is gathered for smoke development.

## Phase 3 — Tokenizer

1. Implement `train_tokenizer.py`.
2. Train 4096-token BPE tokenizer.
3. Add special/control tokens.
4. Save tokenizer.
5. Produce tokenizer statistics.
6. Manually inspect tokenization examples.

## Phase 4 — Dataset packing

1. Implement `tokenize_corpus.py`.
2. Split by document.
3. Encode documents.
4. Insert BOS/EOS appropriately.
5. Pack to uint16 binary data.
6. Write metadata.
7. Implement dataset loader.

## Phase 5 — Model

1. Implement RMSNorm.
2. Implement RoPE.
3. Implement grouped-query causal attention.
4. Implement SwiGLU.
5. Implement Transformer block.
6. Implement full LM.
7. Tie embedding/output weights.
8. Print exact parameter count.
9. Add tests.

## Phase 6 — Training

1. Implement AdamW training.
2. Add BF16 autocast where appropriate.
3. Add gradient clipping.
4. Add LR warmup and cosine decay.
5. Add validation.
6. Add TensorBoard.
7. Add checkpointing.
8. Add resume support.
9. Report throughput and VRAM.

## Phase 7 — Tiny overfit test

1. Train on a tiny subset.
2. Verify loss can fall dramatically.
3. Generate from memorized prompts.
4. Fix all pipeline bugs before continuing.

## Phase 8 — Smoke training run

1. Train on the larger smoke corpus.
2. Monitor train and validation loss.
3. Save periodic checkpoints.
4. Generate fixed evaluation prompts.
5. Compare checkpoints.
6. Record results in README or experiment notes.

---

# Definition of Done — FictionPulper-5M-smoke

The milestone is complete when all of the following are true:

- [ ] corpus can be prepared reproducibly
- [ ] tokenizer is trained from scratch
- [ ] tokenizer has 4096-token vocabulary
- [ ] dataset is split by document
- [ ] tokenized train/validation/test files exist
- [ ] model is approximately 5M parameters
- [ ] model starts from random weights
- [ ] model passes unit tests
- [ ] tiny dataset can be deliberately overfit
- [ ] full smoke training runs on the RTX 5080
- [ ] training loss meaningfully decreases
- [ ] validation loss is tracked
- [ ] checkpoint can be saved
- [ ] checkpoint can be restored
- [ ] generation CLI works
- [ ] generated text becomes recognizably English prose
- [ ] TensorBoard logs are usable
- [ ] README documents reproduction steps

---

# What Not To Do Yet

Do not add these during the smoke milestone unless they are needed to fix a demonstrated problem:

- distributed training
- DeepSpeed
- FSDP
- multi-GPU support
- LoRA
- quantization-aware training
- RLHF
- DPO
- synthetic Qwen-generated corpus
- instruction tuning
- speculative decoding
- FlashAttention third-party dependencies
- custom CUDA kernels
- 50M model
- 4K+ context length
- web UI
- API server
- Hugging Face model publishing

Those are future work.

The goal right now is a clean, inspectable, reproducible training pipeline.

---

# Future Direction

After the 5M smoke milestone succeeds, the intended progression is roughly:

```text
FictionPulper-5M-smoke
        │
        ▼
improve corpus + evaluation
        │
        ▼
FictionPulper-15M experiment
        │
        ▼
architecture / context tests
        │
        ▼
FictionPulper-50M-v1
```

The eventual ~50M model will use a substantially larger and more carefully curated dataset, potentially combining:

- curated public-domain fiction
- original synthetic genre fiction
- structured narrative-planning data
- continuity/state-tracking exercises
- dialogue-heavy examples
- openings, transitions, climaxes, and endings
- genre-conditioning metadata

That is explicitly outside the current smoke milestone.

---

# Coding Style

Prefer:

- readable Python
- type hints
- dataclasses where helpful
- short focused modules
- descriptive variable names
- explicit configuration
- meaningful assertions
- clear error messages
- minimal hidden behavior

Avoid:

- giant framework abstractions
- unnecessary inheritance
- unexplained magic numbers
- premature optimization
- copying large reference implementations without understanding them

Comments should explain **why**, not narrate obvious code.

---

# OpenCode Working Rules

When working on FictionPulper:

1. Inspect existing files before editing.
2. Do not delete functioning user work without justification.
3. Keep each change scoped to the active milestone.
4. Run relevant tests after implementation.
5. Run cheap sanity checks before expensive training.
6. If outside-scope work is discovered, record it as TODO/future work rather than expanding scope automatically.
7. Do not commit large datasets, model checkpoints, caches, or generated binary artifacts.
8. Never add secrets or credentials to the repository.
9. Prefer completing one functioning vertical slice over creating many unfinished abstractions.
10. Update README instructions when workflow commands change.

---

# Immediate Task

Start with **Phase 1 and Phase 2**.

Specifically:

1. Inspect the repository as it exists.
2. Create the intended directory structure without destroying existing files.
3. Add `.gitignore`.
4. Add `configs/smoke-5m.yaml`.
5. Implement `src/prepare_corpus.py`.
6. Inspect the live schema of `zkeown/gutenberg-corpus` instead of assuming it.
7. Build a small development corpus first.
8. Print useful corpus statistics.
9. Save the cleaned output as JSONL.
10. Add tests or validation checks for malformed/empty records.
11. Document the exact command in README.
12. Stop before tokenizer/model implementation if corpus preparation has unresolved correctness problems.

When choosing between speed and trustworthy training data, choose trustworthy training data.
