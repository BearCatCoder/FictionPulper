# Packing Comparison

The underlying `uint16` token streams are byte-identical between packings. Only sequence indexes and document-tail padding change. No sequence crosses a document boundary.

| Split | Context | Documents | Chunks | Raw tokens | Valid targets | Allocated slots | Padding slots | Padding | Usable targets/chunk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 1024 | 3,707 | 28,412 | 27,180,978 | 27,177,271 | 29,093,888 | 1,916,617 | 6.5877% | 956.5420 |
| Train | 2048 | 3,707 | 15,118 | 27,180,978 | 27,177,271 | 30,961,664 | 3,784,393 | 12.2228% | 1,797.6763 |
| Validation | 1024 | 327 | 1,411 | 1,276,683 | 1,276,356 | 1,444,864 | 168,508 | 11.6626% | 904.5755 |
| Validation | 2048 | 327 | 786 | 1,276,683 | 1,276,356 | 1,609,728 | 333,372 | 20.7098% | 1,623.8626 |
| Test | 1024 | 328 | 1,672 | 1,547,045 | 1,546,717 | 1,712,128 | 165,411 | 9.6611% | 925.0700 |
| Test | 2048 | 328 | 915 | 1,547,045 | 1,546,717 | 1,873,920 | 327,203 | 17.4609% | 1,690.4011 |

Across all 4,362 stories, chunks fall from 31,495 to 16,819. Median chunks/story falls from 5 to 3, the 95th percentile from 19 to 10, and the maximum from 39 to 20. Average usable targets/chunk rises from 952.5431 to 1,783.7175.

## Context Coverage

| Full story size | Stories | Percentage |
|---|---:|---:|
| At most 1,024 tokens | 79 | 1.8111% |
| At most 2,048 tokens | 588 | 13.4801% |
| At most 4,096 tokens | 1,803 | 41.3343% |

| 2,048-token chunks/story | Stories | Percentage |
|---|---:|---:|
| 1 | 588 | 13.4801% |
| 2 | 1,215 | 27.8542% |
| 3-4 | 1,255 | 28.7712% |
| 5+ | 1,304 | 29.8945% |

The 2,048 packing exposes almost twice as many usable neighboring targets per chunk, but doubles absolute document-tail padding because each story's final partial chunk has twice the capacity.
