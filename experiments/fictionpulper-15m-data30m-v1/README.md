# FictionPulper-15M-data30M-v1 Preparation

This controlled experiment is prepared but not trained. Relative to sealed `FictionPulper-15M-data10M-v1`, it holds model architecture, tokenizer-v1, context length, seed, optimizer, learning-rate policy, precision, and batching fixed while expanding the unique fiction corpus.

The architecture instantiates to exactly 15,047,040 trainable parameters. Corpus-v2 is anchored by annotated tag `fictionpulper-corpus-v2-30m` at commit `823ab270439fcfdbe9ba0002d275ececdadeb8e3`.

The leakage-audited split contains 3,707 train, 327 validation, and 328 test documents. Source collections and transitive reviewed duplicate clusters are indivisible. Forty historical collections contained records from multiple old splits; majority assignment preserved the maximum possible 1,733 of 1,841 retained historical assignments and changed 108. Therefore Data30M validation/test form a new benchmark rather than an unchanged continuation of Data10M evaluation.

Packing uses tokenizer-v1, `uint16`, 1,024-token document chunks, existing control/BOS/EOS formatting, and padding-masked targets. It produced 28,412 train chunks and 444 optimizer steps per epoch. The primary schedule is locked to:

| Epochs | Optimizer steps | Warmup steps | Real target presentations | Allocated slot presentations |
|---:|---:|---:|---:|---:|
| 3 | 1,332 | 67 | 81,531,813 | 87,281,664 |

The selected epoch-7 Data10M baseline saw 61,454,078 valid target presentations. The Data30M/Data10M-best exposure ratio is therefore 1.3267111907528741. Three epochs avoid coupling the larger unique corpus to roughly three times as much optimization exposure. Even if validation is improving at epoch 3, this run will stop; additional exposure belongs to a separate experiment. No model initialization, optimizer step, checkpoint, or generation has been run for Data30M.

`evaluation-protocol.json` freezes the post-selection evaluations. The full historical Data10M and legacy definitions are retained exactly, but are labeled as continuity diagnostics because 48 old validation records and 45 old test records occur in Data30M train after collection grouping. Separate 91-record validation and 95-record test subsets remove those exact training IDs for common held-out comparisons. The three Corpus-v2 correctness exclusions are all historical train records and remove no requested validation or test ID.
