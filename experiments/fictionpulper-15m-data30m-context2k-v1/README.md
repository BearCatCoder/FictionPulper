# FictionPulper-15M-data30M-context2k-v1

This controlled context-scaling experiment changes the maximum sequence length from 1,024 to 2,048 while holding the 15,047,040-parameter architecture, Corpus-v2/Data30M stories and splits, tokenizer-v1, seed, optimizer policy, generation settings, and valid-target exposure fixed against sealed `fictionpulper-15m-data30m-compute5`.

RoPE uses its ordinary base and frequencies without scaling. Training starts from random weights with seed 1337 and does not load any prior checkpoint.

## Locked Batch

A 16 x 2,048 BF16 forward/backward with the complete model peaked at 7.3105 GiB allocated on the RTX 5080, leaving a practical margin below 15.47 GiB. Microbatch remains 16. Gradient accumulation changes from 4 to 2 so allocated capacity remains exactly 65,536 token slots per optimizer step:

```text
16 microbatch x 2 accumulation x 2048 tokens = 65,536 slots/step
```

## Locked Schedule

All candidates present the same 27,177,271 valid targets per epoch. Five epochs exactly match Compute5's 135,886,355 valid-target presentations and were selected before training.

| Epochs | Optimizer steps | Valid target presentations | Ratio vs Compute5 |
|---:|---:|---:|---:|
| 3 | 1,419 | 81,531,813 | 0.6 |
| 4 | 1,892 | 108,709,084 | 0.8 |
| 5 | 2,365 | 135,886,355 | 1.0 |

The locked schedule is five document/chunk epochs, 2,365 optimizer steps, and 118 linear-warmup steps followed by cosine decay. It will not be changed after validation is observed.

## Evaluation

Checkpoint selection uses only complete 2,048-packed Data30M validation loss. Test access remains sealed until all five epochs finish and the best validation checkpoint is selected. The historical benchmark definitions are evaluated at their original 1,024-token packing over identical source tokens.

The original ten prompts and decoding settings remain unchanged. A separate three-prompt context-retention suite places facts more than 1,024 but fewer than 2,048 tokens before the continuation point and is evaluated post hoc on both Compute5 and context2k. Neither generation suite affects checkpoint selection.
