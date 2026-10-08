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
