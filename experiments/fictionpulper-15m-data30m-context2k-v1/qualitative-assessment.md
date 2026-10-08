# Qualitative Assessment

## Classification

**No meaningful context benefit.** Doubling the available context did not produce reliable long-range retrieval, state persistence, or stronger storytelling. Small prompt-level exceptions are outweighed by complete failure on the direct retention suite, unchanged topic drift and causality, and worse aggregate repetition.

## Required Questions

1. **Did 2,048 context materially reduce prompt drift?** No. Context2k better preserves the Dry Creek stranger and some crime vocabulary, but science-fiction, fantasy, horror, and mystery premises still collapse into unrelated rooms, domestic scenes, clothing descriptions, or generic dialogue.
2. **Did entity consistency improve?** No. Names and roles disappear or mutate. Context2k recalls none of Martin Vale, Jonah Reed, Elise Ward, Daniel, Mara Venn, or Amos Bell when those names remain inside its window.
3. **Did scenes persist longer?** Not systematically. Dry Creek and parts of the revolver continuation persist somewhat longer, while the station, desert camp, Europa, tower, and retention-suite settings drift as quickly or faster.
4. **Did causal or narrative progression improve?** No. Continuations rarely follow from the initiating contradiction, threat, goal, or secret. Local action sequences remain juxtaposed rather than causally connected.
5. **Did repetition improve?** No. Original-suite greedy repeated 4-grams rose from 0.4632 to 0.5576 and longest repeated span from 17.7 to 26.2 tokens. Sampled repetition also worsened modestly.
6. **Did token-level generalization improve?** Not robustly. Native 2,048-packed Data30M test loss improved 0.23%, but the identical 1,024-packed common held-out test worsened 0.34% and legacy test loss worsened 0.92%.
7. **Was the extra compute and memory justified?** No. Peak VRAM rose 90.64%, runtime rose 33.52%, and throughput fell 25.10% without material narrative improvement.

## Original Prompt Evidence

- **Station:** Compute5 at least keeps a train and station. Context2k moves to an office and contradictory descriptions of hats, hair, and differently colored beards.
- **Floorboards:** Context2k produces a somewhat atmospheric dark interior and movement upstairs, but never develops the movement beneath the floorboards.
- **Rocket and Europa:** Both models abandon the science-fiction premises almost immediately.
- **Dry Creek:** Context2k is clearly better here, preserving `the stranger` and an encounter, but loses Dry Creek and western causality.
- **Mallory locked door:** Neither model investigates the door, preserves Mallory, or develops the lie.
- **Revolver:** Context2k retains a prisoner, dead man, doctor, and interrogation-like exchange, but loses Mallory and the two-versus-three-bullet contradiction.
- **Portrait:** Context2k mentions Clara several times and begins with eyes and a shadowed figure, but never sustains the supernatural portrait event and later confuses Clara with Sara.
- **Desert camp:** Compute5 better retains a dusty stranger and entry into a dark building. Context2k drifts to a road, office, schoolhouse, marriage, and young woman.
- **Tower:** Neither model preserves the tower or absent shadow; Context2k degenerates into metacommentary about stories and books.

## Context-Retention Evidence

The three prompts contain 13 explicit facts whose prefixes occur 1,063-1,072 tokens before generation. Compute5 truncates those prefixes; context2k sees them in full.

- Context2k recalls none of Martin Vale, Blackwater Hotel, the brass key, room 317, or the midnight goal.
- Context2k recalls none of Jonah Reed, Elise Ward, Daniel, the sister-in-law relationship, the northbound train, Daniel's survival, or Ashcombe.
- Context2k recalls none of Mara Venn, Saint Orra, the oilcloth logbook, blue chalk circle, Amos Bell, or the delivery goal.

The shared intervening prose uses masculine pronouns, making Mara's pronoun-persistence subtest noisy. This limitation does not explain the failure to recover every explicit name, object, location, detail, relationship, secret, and goal.

## Interpretation

The 15M model can technically process 2,048 tokens, but it does not demonstrate useful narrative-state use at that distance. The bottleneck is not simply the size of the visible context window. At this scale and training setup, additional context mainly increases memory and compute cost without solving prompt retention, entity tracking, causal progression, or repetition.
