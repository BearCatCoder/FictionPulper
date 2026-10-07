# FictionPulper-15M-data30M-v1 Preparation

This controlled experiment is prepared but not trained. Relative to sealed `FictionPulper-15M-data10M-v1`, it holds model architecture, tokenizer-v1, context length, seed, optimizer, learning-rate policy, precision, and batching fixed while expanding the unique fiction corpus.

The architecture instantiates to exactly 15,047,040 trainable parameters. Corpus-v2 is anchored by annotated tag `fictionpulper-corpus-v2-30m` at commit `823ab270439fcfdbe9ba0002d275ececdadeb8e3`.

The leakage-audited split contains 3,707 train, 327 validation, and 328 test documents. Source collections and transitive reviewed duplicate clusters are indivisible. Forty historical collections contained records from multiple old splits; majority assignment preserved the maximum possible 1,733 of 1,841 retained historical assignments and changed 108. Therefore Data30M validation/test form a new benchmark rather than an unchanged continuation of Data10M evaluation.

Packing uses tokenizer-v1, `uint16`, 1,024-token document chunks, existing control/BOS/EOS formatting, and padding-masked targets. It produced 28,412 train chunks and 444 optimizer steps per epoch. Candidate schedules are:

| Epochs | Optimizer steps | Warmup steps | Real target-token visits |
|---:|---:|---:|---:|
| 3 | 1,332 | 67 | 81,531,813 |
| 5 | 2,220 | 111 | 135,886,355 |
| 7 | 3,108 | 155 | 190,240,897 |
| 10 | 4,440 | 222 | 271,772,710 |

Seven epochs is the configured candidate because the sealed 15M/Data10M run reached its validation minimum at epoch 7. This is a pre-training choice, not a Data30M result. No model initialization, optimizer step, checkpoint, or generation has been run for Data30M.
