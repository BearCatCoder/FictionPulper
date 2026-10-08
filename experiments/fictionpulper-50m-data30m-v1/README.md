# FictionPulper-50M-data30M-v1

This controlled capacity experiment changes the model from 15,047,040 to 50,348,544 trainable parameters while holding Corpus-v2/Data30M membership and splits, tokenizer-v1, 1,024-token packing, context length, seed, optimizer policy, effective token batch, valid-target exposure, generation settings, and evaluation definitions fixed against sealed `fictionpulper-15m-data30m-compute5`.

The larger model uses the same RMSNorm, ordinary RoPE, grouped-query causal attention, SwiGLU, residual structure, and tied embeddings. It adds no architectural feature. Training starts from random weights with seed 1337 and does not load a prior checkpoint.

## Locked Capacity

| Dimension | Compute5 | 50M |
|---|---:|---:|
| Parameters | 15,047,040 | 50,348,544 |
| Hidden size | 384 | 512 |
| Layers | 8 | 16 |
| Attention heads | 6 | 8 |
| Key/value heads | 2 | 2 |
| Intermediate size | 1,120 | 1,536 |
| Context | 1,024 | 1,024 |

The parameter multiplier is 3.3460763047084345. At fixed exposure, valid-target presentations per parameter fall from 9.03077 to 2.69891. This experiment does not compensate by adding epochs.

## Locked Batch And Schedule

A complete BF16 training update at microbatch 16, including fused AdamW state allocation, peaked at 8.6796 GiB allocated and 8.8535 GiB reserved on the RTX 5080. The original batch policy fits safely and remains unchanged:

```text
16 microbatch x 4 accumulation x 1024 tokens = 65,536 slots/step
```

The exact Compute5 schedule is retained: five document/chunk epochs, 444 optimizer steps per epoch, 2,220 total optimizer steps, and 111 warmup steps. Both runs present exactly 135,886,355 valid training targets.

## Stability Gate

At the locked `1e-3` learning rate, an eight-update fixed-subset run reduced loss monotonically from 8.40099 to 6.37564. Every trainable parameter received a finite gradient, no NaN or Inf occurred, and checkpoint reload reproduced probe logits exactly. The gate was not used for tuning.

## Evaluation

Checkpoint selection uses only complete Data30M validation loss. Test access remains sealed until all five epochs finish. The original ten-prompt generation benchmark and decoding settings remain unchanged.

The Context2k narrative-state scenarios are reused in a separately labeled 1,024-compatible form. Both Compute5 and the 50M model receive every diagnostic fact inside their context window so the comparison tests capacity rather than truncation. Neither generation suite affects checkpoint selection.

## Results

The run completed all five locked epochs and selected epoch 5, step 2,220 using Data30M validation loss only. Training artifacts were accidentally written under the generated ID `smoke-5m-20261007-212756` because `--run-id` was omitted. The raw record is preserved under that name; derived reports use the intended experiment ID. Training and sealed test evaluation were not rerun.

| Metric | 15M Compute5 | 50M Capacity |
|---|---:|---:|
| Best validation loss | 3.1456916993 | 3.0074663302 |
| Data30M test loss | 3.2231023562 | 3.1017555046 |
| Data30M test perplexity | 25.1058868921 | 22.2369541110 |
| Data30M test accuracy | 0.3637051898 | 0.3825981094 |
| Common held-out 95-doc loss | 3.2246202187 | 3.1131359160 |
| Training seconds | 422.8692 | 1234.4847 |
| Valid targets / second | 321343.7 | 110075.4 |
| Peak VRAM GiB | 3.9406 | 9.3029 |

Capacity improves Data30M test loss by 3.76%, perplexity by 11.43%, and accuracy by 5.19%. The common held-out benchmark confirms the direction. Runtime increases 2.92x, peak VRAM increases 2.36x, and throughput falls 65.75%.

## Conclusion

**Meaningful likelihood improvement without material storytelling improvement.** Sampled generations show occasional local fluency and atmosphere, but prompt drift, contradiction, and weak event continuity remain. Both models convincingly retain 0 of 13 facts in the same-context narrative-state diagnostic. The 50M model is also more repetitive under greedy decoding on aggregate.

The larger model is a useful capacity baseline, but its extra compute and memory are not justified yet for the storytelling objective. Corpus quality, narrative supervision, and repetition control are higher-value next steps than another capacity increase.

## Commands

The actual training command omitted `--run-id`, which caused the disclosed raw artifact name:

```bash
python -m src.train --config configs/data30m-50m-v1.yaml
```

The intended command for a fresh reproduction is:

```bash
python -m src.train \
  --config configs/data30m-50m-v1.yaml \
  --run-id fictionpulper-50m-data30m-v1
```

The post-selection narrative-state comparison used:

```bash
python -m src.evaluate_context_retention \
  --baseline-checkpoint checkpoints/fictionpulper-15m-data30m-compute5/best-validation.pt \
  --candidate-checkpoint checkpoints/fictionpulper-50m-data30m-v1/best-validation.pt \
  --protocol experiments/fictionpulper-50m-data30m-v1/narrative-state-protocol.json \
  --tokenizer data/tokenizer/tokenizer.json \
  --output runs/fictionpulper-50m-data30m-v1/narrative-state-comparison.json \
  --baseline-label compute5_context1024 \
  --candidate-label capacity50m_context1024 \
  --baseline-context 1024 \
  --candidate-context 1024

python -m src.report_capacity50m
```
