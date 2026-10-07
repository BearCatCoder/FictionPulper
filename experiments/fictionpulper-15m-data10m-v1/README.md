# FictionPulper-15M-data10M-v1

This controlled experiment changes only model capacity relative to sealed `FictionPulper-5M-data10M-v1`.

It reuses the exact resolved corpus, split manifest, tokenizer-v1, 1,024-token context, packed token files, seed, optimizer policy, ten chunk epochs, evaluation prompts, and generation settings. Tokenizer v2 is explicitly not used.

The requested architecture instantiates to exactly 15,047,040 trainable parameters. It uses the existing RMSNorm, RoPE, grouped-query causal attention, SwiGLU, and tied embedding implementations without architectural additions.

The isolated short sanity overfit started at loss 8.3802, near `ln(4096) = 8.3178`, and reached 5.9924 after 40 steps. All parameters received gradients, values remained finite, checkpoint reload was identical, and peak VRAM was 2.53 GiB.

## Results

The controlled run completed all 1,470 optimizer steps without numerical instability. Validation selected epoch 7 at loss 3.403524; validation then worsened through epoch 10 while training loss continued downward. The sealed Corpus-v1 test result was loss 3.484673, perplexity 32.611754, and accuracy 0.323956.

Relative to the 5M Data10M baseline, test loss improved 2.25%, test perplexity improved 7.72%, and test accuracy improved 2.95%. Legacy clean validation loss improved 2.81%, while legacy test loss improved 2.64%.

Runtime increased from 127.45 to 278.66 seconds, throughput fell 54.26%, and peak VRAM increased from 2.34 to 3.94 GiB. Generations occasionally show richer local descriptions and longer dialogue, but prompt adherence, repetition, entity consistency, and narrative progression remain weak. Capacity is a demonstrated constraint, but the epoch-7 validation minimum also indicates increasing pressure from the fixed corpus size.
