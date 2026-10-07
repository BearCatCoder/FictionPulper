# Quantitative Comparison

| Metric | 15M Data10M | 15M Data30M |
|---|---:|---:|
| Parameters | 15047040 | 15047040 |
| Context | 1024 | 1024 |
| Tokenizer | v1 | v1 |
| Unique training targets | 8779154 | 27177271 |
| Stories | 1565 | 3707 |
| Target presentations to selected checkpoint | 61454078 | 81531813 |
| Best native validation loss (different sets) | 3.403523859410251 | 3.2460981563125317 |
| Data10M full-test loss (45 Data30M exposures) | 3.4846727679995286 | 3.299286296108072 |
| Data10M full-test PPL (45 Data30M exposures) | 32.611753754203036 | 27.09329542832374 |
| Data10M full-test accuracy (45 Data30M exposures) | 0.3239561290183859 | 0.34718710747888787 |
| Common 95-record test loss | 3.4801647135445886 | 3.3169595081613577 |
| Common 95-record test PPL | 32.46506907209253 | 27.57637721321671 |
| Common 95-record test accuracy | 0.3262478512407822 | 0.3464091596612254 |
| Legacy clean validation loss (Data30M contaminated) | 3.465931473360209 | 3.2052978249618653 |
| Legacy test loss (Data30M contaminated) | 3.527728861968554 | 3.2799965212742994 |
| Legacy test PPL (Data30M contaminated) | 34.046555314405204 | 26.57568025021126 |
| Legacy test accuracy (Data30M contaminated) | 0.3122718193723981 | 0.34332999217143817 |
| Training runtime seconds | 278.66396146599436 | 248.1865291610011 |
| Valid targets/second | 315044.4698271947 | 328510.22686694446 |
| Peak VRAM GiB | 3.9406185150146484 | 3.9406185150146484 |

Native validation sets differ and are not directly compared as identical benchmarks. The full Data10M test uses identical records but is exposure-contaminated for Data30M. The 95-record Data10M test subset is disjoint from both training sets and is the primary direct token-level comparison.
