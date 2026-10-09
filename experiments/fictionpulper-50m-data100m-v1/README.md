# FictionPulper-50M-data100M-v1

This issue prepares, but does not claim, a fresh-init 50,348,544-parameter
ordinary causal language-model experiment over sealed Corpus-v3 data. The model
keeps tokenizer-v1, context 1,024, BF16, and the proven 16 x 4 batch geometry.
The only intended experimental variable relative to the sealed 50M Data30M
capacity run is unique training data.

## Preregistered Contract

- Architecture: 512 hidden, 16 layers, 8 query heads, 2 key/value heads, 1,536
  SwiGLU intermediate, RMSNorm, ordinary RoPE, tied embeddings, no projection
  biases; exactly 50,348,544 trainable parameters.
- Initialization: seed 1337 from random weights; no pretrained or resume
  checkpoint.
- Objective: ordinary next-token causal language modeling with no auxiliary
  loss.
- Batch: microbatch 16, four-step accumulation, 65,536 allocated token slots
  per optimizer update.
- Exposure: five complete document/chunk epochs. Exact optimizer steps, warmup,
  valid-target presentations, and artifact hashes must be filled from the final
  packed metadata before the readiness gate can pass.
- Selection: minimum complete Corpus-v3 validation loss only. Corpus-v3 test and
  Narrative Benchmark v2 test remain closed until checkpoint selection.
- Comparison: historical Compute5 and 50M checkpoints must be evaluated on the
  final Corpus-v3 test split. The old 95-document common subset is not presumed
  clean against a new training corpus.

## Current Decision

**Stop before training.** Corpus-v3 has no seal or packed data. Its documented
base-only feasibility run reached 14,923,683 tokenizer-v1 document tokens and
stopped with audit failures: 57.04% unclassified-token share exceeded the 35%
limit and deterministic review remained pending. Consequently the exact corpus,
split, packed hashes, schedule, exposure, and common clean LM comparison cannot
yet be frozen. No Data100M model training, validation, checkpoint selection, or
held-out evaluation occurred.

The preflight deliberately fails closed until those values exist:

```bash
python -m src.data100m_preflight \
  --config configs/data100m-50m-v1.yaml \
  --require-ready
```

After Corpus-v3 passes every gate, pack it with the same config, enter the three
generated data SHA-256 locks plus exact `max_steps` and `warmup_steps`, rerun the
hardware gate, and enter that report's SHA-256 as `expected_report_sha256`.
Training must not start until preflight reports `ready_for_training`.

The model-only hardware check uses deterministic synthetic 1,024-token
sequences. It allocates the complete model, BF16 activations, gradients, and
fused AdamW state, but it neither reads Corpus-v3 nor claims a training result:

```bash
python -m src.data100m_preflight \
  --config configs/data100m-50m-v1.yaml \
  --hardware-gate \
  --output experiments/fictionpulper-50m-data100m-v1/preflight.json
```

The 2026-10-09 RTX 5080 run completed eight updates in 1.3094 seconds. Peak
allocated VRAM was 9.3107 GiB and peak reserved VRAM was 9.6523 GiB, below the
16 GiB gate without reducing microbatch 16. Loss decreased on every update from
8.4210 to 7.5507, all trainable gradients were present and finite, and reloaded
checkpoint logits matched exactly. These are synthetic stability and fit
measurements only, not Corpus-v3 throughput or training results.

After packing, record the final test binary and index hashes from metadata and
evaluate both historical baselines on that identical clean split before making
the candidate comparison:

```bash
python -m src.evaluate_checkpoint \
  --checkpoint checkpoints/fictionpulper-15m-data30m-compute5/best-validation.pt \
  --tokenizer data/tokenizer/tokenizer.json \
  --packed data/packed/corpus-v3-data100m/test.bin \
  --expected-packed-sha256 <test-bin-sha256> \
  --expected-index-sha256 <test-index-sha256> \
  --name compute5_corpus_v3_test \
  --classification preregistered_common_clean_lm \
  --output runs/fictionpulper-50m-data100m-v1/compute5-corpus-v3-test.json

python -m src.evaluate_checkpoint \
  --checkpoint checkpoints/fictionpulper-50m-data30m-v1/best-validation.pt \
  --tokenizer data/tokenizer/tokenizer.json \
  --packed data/packed/corpus-v3-data100m/test.bin \
  --expected-packed-sha256 <test-bin-sha256> \
  --expected-index-sha256 <test-index-sha256> \
  --name capacity50m_corpus_v3_test \
  --classification preregistered_common_clean_lm \
  --output runs/fictionpulper-50m-data100m-v1/capacity50m-corpus-v3-test.json
```

Once readiness and the hardware gate both pass, the intended training command
is:

```bash
python -m src.train \
  --config configs/data100m-50m-v1.yaml \
  --run-id fictionpulper-50m-data100m-v1
```

This command is preregistered, not currently authorized or claimed as run.
