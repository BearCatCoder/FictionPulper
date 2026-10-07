# FictionPulper-15M-data10M-v1

This controlled experiment changes only model capacity relative to sealed `FictionPulper-5M-data10M-v1`.

It reuses the exact resolved corpus, split manifest, tokenizer-v1, 1,024-token context, packed token files, seed, optimizer policy, ten chunk epochs, evaluation prompts, and generation settings. Tokenizer v2 is explicitly not used.

The requested architecture instantiates to exactly 15,047,040 trainable parameters. It uses the existing RMSNorm, RoPE, grouped-query causal attention, SwiGLU, and tied embedding implementations without architectural additions.

The isolated short sanity overfit started at loss 8.3802, near `ln(4096) = 8.3178`, and reached 5.9924 after 40 steps. All parameters received gradients, values remained finite, checkpoint reload was identical, and peak VRAM was 2.53 GiB.
