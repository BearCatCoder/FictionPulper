# FictionPulper-5M-smoke-v1

The first complete smoke run validated the corpus, tokenizer, packing, custom model, training, checkpoint, validation, test, and generation pipeline.

## Reproduction

```bash
python -m src.train \
  --config configs/smoke-5m.yaml \
  --run-id smoke-5m-20261007-v1
```

The original run used seed `1337`, started from fresh random initialization, and did not load the tiny-overfit checkpoint.

## Result

- Parameters: 5,426,432
- Optimizer steps: 420
- Corpus epochs: 15
- Initial validation loss: 8.3569
- Best/final validation loss: 4.0923
- Best validation perplexity: 59.88
- Best epoch: 15
- Test loss: 4.1957
- Test perplexity: 66.40
- Valid training targets processed: 22,247,910
- Peak GPU memory: 2.34 GiB

Validation loss improved at every epoch but flattened late in training. Generated text acquired prose and dialogue structure, but remained repetitive, weakly conditioned, and semantically unstable. This is a successful pipeline smoke test, not a claim of literary quality.

`summary.json` contains the compact quantitative record. `generations.md` preserves representative best-checkpoint outputs. Large checkpoints, packed data, and TensorBoard files are intentionally excluded from Git.
