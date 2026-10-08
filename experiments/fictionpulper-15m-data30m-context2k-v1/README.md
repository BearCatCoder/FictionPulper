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

## Results

The run completed exactly five epochs and 2,365 optimizer steps. Epoch 5 was selected by Data30M validation loss before test access. Validation loss fell from 8.38138 before training to 3.66068, 3.38637, 3.25955, 3.17106, and 3.14157. The curve remained improving but flattening; successive gains were 0.27431, 0.12682, 0.08849, and 0.02948.

At identical valid-target exposure, context2k improved native Data30M test loss from 3.22310 to 3.21574 (0.23%) and perplexity from 25.1059 to 24.9216 (0.73%). Accuracy was effectively unchanged at 36.3984% versus 36.3705%. The identical 1,024-packed 95-record common held-out test moved in the opposite direction: loss worsened from 3.22462 to 3.23552, perplexity from 25.1440 to 25.4195, and accuracy from 36.0152% to 35.8278%. Legacy test loss also worsened from 3.08501 to 3.11349. Token-level generalization therefore did not improve robustly across identical-source-token benchmarks.

The qualitative result is **no meaningful context benefit**. Context2k had access to all 13 explicit facts in the long-context suite, while Compute5 necessarily truncated them, but context2k retained none of the tested names, objects, locations, goals, relationships, secrets, or scene details. On the original prompts it produced isolated improvements for Dry Creek, crime theming, and Clara's name, but did not systematically improve prompt adherence, entity consistency, scene persistence, dialogue continuity, causality, topic drift, or narrative progression.

Repetition became worse overall. Across the ten original greedy continuations, repeated 4-gram rate rose from 0.4632 to 0.5576 and mean longest repeated span rose from 17.7 to 26.2 tokenizer tokens. Sampled repeated 4-gram rate rose from 0.01937 to 0.02609, with lower aggregate distinct-1/2/3. The retention samples were modestly more diverse, but that diversity did not preserve any target fact.

Training peak VRAM increased from 3.9406 to 7.5124 GiB, runtime increased 33.52%, and valid-target throughput fell 25.10%. The cost is not justified by the negligible, benchmark-dependent token-metric movement and absent narrative improvement.

See `quantitative-comparison.md`, `generation-comparison.md`, `context-retention-comparison.md`, `repetition-comparison.md`, and `qualitative-assessment.md`.
