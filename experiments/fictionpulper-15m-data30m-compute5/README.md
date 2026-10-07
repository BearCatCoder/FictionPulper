# FictionPulper-15M-data30M-compute5

This controlled compute-scaling experiment changes only the duration of the sealed `FictionPulper-15M-data30M-v1` setup: five fresh-initialization document/chunk epochs replace three. Model architecture, Corpus-v2 records and splits, tokenizer-v1, context, seed, optimizer policy, generation protocol, and evaluation definitions remain fixed.

The run starts from random weights with seed 1337. It does not resume or load the three-epoch checkpoint because the five-epoch cosine schedule must be designed over all 2,220 optimizer steps.

## Locked Schedule

| Epoch | Optimizer step | Cumulative valid target presentations |
|---:|---:|---:|
| 1 | 444 | 27,177,271 |
| 2 | 888 | 54,354,542 |
| 3 | 1,332 | 81,531,813 |
| 4 | 1,776 | 108,709,084 |
| 5 | 2,220 | 135,886,355 |

The five-epoch run presents 145,469,440 allocated slots and 135,886,355 valid targets. Exposure is 1.6666666666666667 times the completed Data30M-v1 run and 2.2111853179214567 times the selected Data10M baseline checkpoint.

The run completed exactly five epochs and 2,220 optimizer steps. Epoch 5 was selected by Data30M validation loss before any test data were accessed. The primary Data30M test and all frozen historical evaluations were each run once after selection.

## Results

Data30M validation loss fell from 8.37798 before training to 3.63832, 3.37687, 3.24674, 3.17016, and 3.14569 after epochs 1 through 5. The incremental improvement shrank at every epoch boundary: 0.26144, 0.13013, 0.07658, and 0.02447. The curve is therefore **improving but flattening**. The validation-minus-training loss gap increased from 0.00807 at epoch 3 to 0.16177 at epoch 5, but validation had not yet regressed when the locked schedule ended.

The selected epoch-5 checkpoint measured Data30M test loss 3.22310, perplexity 25.1059, and next-token accuracy 36.3705% over 1,546,717 valid targets. Relative to the sealed three-epoch run, this is a 2.99% loss reduction, 9.45% perplexity reduction, and 4.24% relative accuracy increase at 1.6667 times the training exposure.

On the identical 95-record historical test subset disjoint from Data30M train, loss improved from 3.31696 to 3.22462, perplexity from 27.5764 to 25.1440, and accuracy from 34.6409% to 36.0152%. Full historical metrics are retained only as exposure-contaminated continuity diagnostics because 48 old validation and 45 old test records occur in Data30M train.

Generation quality did not improve in proportion to token metrics. Sampled outputs occasionally have better local syntax, dialogue formatting, prompt-related vocabulary, and scene persistence, but almost all ten premises are abandoned. Greedy continuations still collapse into severe loops, while entity drift, weak causal linkage, and absent narrative progression remain. Extra optimization therefore improved local next-token likelihood without materially improving storytelling.

See `quantitative-comparison.md`, `generation-comparison.md`, and `qualitative-assessment.md` for the sealed comparison evidence.
