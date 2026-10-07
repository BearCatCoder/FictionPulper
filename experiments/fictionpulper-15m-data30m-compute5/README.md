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

This record is preflight-only until training and sealed evaluation complete.
