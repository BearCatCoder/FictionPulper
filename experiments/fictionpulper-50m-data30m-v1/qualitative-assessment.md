# Qualitative Assessment

## Classification

**Meaningful likelihood improvement without material storytelling improvement.**

The 50M model is a stronger next-token predictor at fixed data exposure, but the ten-prompt benchmark and the same-context narrative-state diagnostic do not show a commensurate improvement in story coherence, prompt adherence, or state retention.

## Original Prompt Suite

The 50M sampled continuations are occasionally more scene-like than Compute5. The floorboards, mining-camp, and crooked-forest prompts contain recognizable physical settings, dialogue, or atmosphere. These local gains are inconsistent. The rocket and Europa prompts abandon their science-fiction premises, the Dry Creek sample fixates on the word "Tuesday," and several continuations drift into unrelated family or marriage material.

Neither model produces reliable multi-paragraph narrative progression. Characters, setting, causal relationships, and goals change without support. The larger model often generates smoother sentence-level transitions, but this does not become stronger event continuity.

Greedy decoding is worse for the 50M model on aggregate. Across the ten original prompts, repeated 4-gram rate rises from 0.4632 to 0.5912, repeated-sentence rate rises from 0.1506 to 0.4073, and mean longest repeated span rises from 17.7 to 32.3 tokens. Sampled repetition is also slightly worse overall: repeated 4-gram rate rises from 0.0194 to 0.0411 and mean longest repeated span from 4.9 to 5.7 tokens.

## Narrative State

The three scenarios contain 13 disclosed facts. Every fact remains inside both models' 1,024-token context for the complete generated continuation. Assessment used four conservative outcomes: retained, contradicted, omitted, or ambiguous. A generic mention such as "key" was not counted unless the required owner, object, location, or goal relationship was preserved.

| Scenario | Facts | Compute5 retained | 50M retained | Assessment |
|---|---:|---:|---:|---|
| Martin Vale / key / hotel / room 317 | 4 | 0 | 0 | Compute5 mentions a key but relocates it to a dining room; 50M renames the man and omits the state. |
| Jonah Reed / Elise / Daniel / train / hotel | 4 | 0 | 0 | Both continuations omit the named relationship, secret, train event, and location. |
| Mara Venn / logbook / lighthouse / blue circle / Amos Bell | 5 | 0 | 0 | Both continuations omit the required state; the 50M greedy output collapses into sentence repetition. |
| **Total** | **13** | **0** | **0** | No demonstrated capacity benefit for narrative-state retention. |

This diagnostic is intentionally small and supports only a directional conclusion. Its shared filler also contains masculine-pronoun noise in Mara's scenario. That limitation was preserved and disclosed rather than changing the frozen protocol after evaluation.

## Learning Curve

Validation improves in every epoch, but the gains shrink from 0.354 loss between epochs 1 and 2 to 0.0073 between epochs 4 and 5. The final train-validation gap is 0.2395. The curve is best described as **improving but effectively plateaued by epoch 5**. The locked schedule was not extended.

## Decision

The 3.35x parameter increase produces a real quantitative gain: Data30M test loss improves 3.76%, perplexity improves 11.43%, and accuracy improves 5.19%. The disjoint 95-document benchmark confirms the direction. It costs 2.92x the training time, uses 2.36x peak VRAM, and reduces throughput by 65.75%.

That trade is not justified yet for the project's storytelling objective. Keep the 50M checkpoint as a useful capacity baseline, but prioritize data quality, narrative supervision, and decoding/repetition work before spending more compute on a larger model.
