# FictionPulper-15M-data30M-v1 Preparation

This controlled experiment is complete. Relative to sealed `FictionPulper-15M-data10M-v1`, it holds model architecture, tokenizer-v1, context length, seed, optimizer, learning-rate policy, precision, and batching fixed while expanding the unique fiction corpus.

The architecture instantiates to exactly 15,047,040 trainable parameters. Corpus-v2 is anchored by annotated tag `fictionpulper-corpus-v2-30m` at commit `823ab270439fcfdbe9ba0002d275ececdadeb8e3`.

The leakage-audited split contains 3,707 train, 327 validation, and 328 test documents. Source collections and transitive reviewed duplicate clusters are indivisible. Forty historical collections contained records from multiple old splits; majority assignment preserved the maximum possible 1,733 of 1,841 retained historical assignments and changed 108. Therefore Data30M validation/test form a new benchmark rather than an unchanged continuation of Data10M evaluation.

Packing uses tokenizer-v1, `uint16`, 1,024-token document chunks, existing control/BOS/EOS formatting, and padding-masked targets. It produced 28,412 train chunks and 444 optimizer steps per epoch. The primary schedule is locked to:

| Epochs | Optimizer steps | Warmup steps | Real target presentations | Allocated slot presentations |
|---:|---:|---:|---:|---:|
| 3 | 1,332 | 67 | 81,531,813 | 87,281,664 |

The selected epoch-7 Data10M baseline saw 61,454,078 valid target presentations. The Data30M/Data10M-best exposure ratio is therefore 1.3267111907528741. Three epochs avoid coupling the larger unique corpus to roughly three times as much optimization exposure. The run stopped exactly at epoch 3 and 1,332 optimizer steps.

`evaluation-protocol.json` freezes the post-selection evaluations. The full historical Data10M and legacy definitions are retained exactly, but are labeled as continuity diagnostics because 48 old validation records and 45 old test records occur in Data30M train after collection grouping. Separate 91-record validation and 95-record test subsets remove those exact training IDs for common held-out comparisons. The three Corpus-v2 correctness exclusions are all historical train records and remove no requested validation or test ID.

## Results

Data30M validation loss fell from 8.37798 before training to 3.62640, 3.34607, and 3.24610 after epochs 1, 2, and 3. Epoch 3 is the selected checkpoint. The primary Data30M test measured loss 3.32238, perplexity 27.7263, and next-token accuracy 34.8922% over 1,546,717 valid targets.

On the identical 95-record Data10M test subset that is disjoint from both training sets, the Data10M baseline measured loss 3.48016 / perplexity 32.4651 / accuracy 32.6248%, while Data30M measured 3.31696 / 27.5764 / 34.6409%. This is a 4.69% loss reduction, 15.06% perplexity reduction, and 6.18% relative accuracy increase. The full 140-record Data10M test also improves, but 45 of those records occurred in Data30M train and that metric is reported only as an exposure-contaminated continuity diagnostic.

Validation is classified as **improving but flattening**: the loss improvement shrank from 0.28033 between epochs 1 and 2 to 0.09997 between epochs 2 and 3. Additional optimization may help, but it was not performed in this experiment.

Generation quality remains mixed. Sampled prose has modest local fluency and dialogue gains, but prompt adherence, identity consistency, scene persistence, causal progression, and greedy repetition remain poor. See `qualitative-assessment.md` and `generation-comparison.md`.
