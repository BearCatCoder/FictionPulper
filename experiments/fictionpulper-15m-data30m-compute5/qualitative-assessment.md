# Qualitative Assessment

## Verdict

Compute5 does not materially improve storytelling over the sealed three-epoch Data30M model. Additional optimization produces occasional gains in local sampled syntax, dialogue formatting, scene persistence, and prompt-related vocabulary. Those gains are offset by unchanged greedy degeneration, persistent entity contradictions, weak causality, and abandonment of nearly every premise. The quantitative improvement is primarily better local next-token likelihood rather than stronger long-range narrative control.

| Criterion | Effect of additional compute |
|---|---|
| Grammar | Slight sampled improvement, but malformed clauses and agreement errors remain common. |
| Complete sentences | Slight improvement within sampled passages; fixed-length outputs commonly end mid-sentence. |
| Prompt adherence | No material improvement; isolated words such as `station`, `floor`, `killed`, and `revolver` survive more often than the premise. |
| Semantic continuity | Marginal sampled improvement in a few local scenes, with frequent incompatible transitions elsewhere. |
| Dialogue coherence | Turn-taking is sometimes clearer, but replies remain circular or unrelated. |
| Repetition | No improvement; greedy decoding remains dominated by loops, sometimes more severe than at epoch 3. |
| Entity consistency | No reliable improvement; age, gender, appearance, and relationships still change within passages. |
| Scene persistence | Small sampled improvement in a few outputs, usually in a setting unrelated to the prompt. |
| Causal progression | No improvement; events are juxtaposed instead of arising from the prompt or preceding actions. |
| Narrative progression | No material improvement; none of the ten premises develops into a sustained complication or consequence. |

## Prompt Evidence

- The station continuation uses train and station vocabulary more directly, but greedy decoding becomes a long `the train was coming on` loop and drops the man, gray coat, and midnight.
- The floorboards continuation maintains an ominous room and floor focus in greedy mode, but never investigates or identifies the movement; the sample abandons the premise.
- Rocket, Europa, tower, and portrait prompts still collapse into generic historical domestic prose rather than sustaining speculative or supernatural elements.
- The Dry Creek sample holds a domestic scene longer and uses more conventional dialogue, but generates a different story with no stranger or Dry Creek.
- Both Mallory prompts gain occasional crime-related words, including `killed` and `revolver`, without reasoning about the locked door, lie, firing count, or bullet contradiction.
- The desert sample retains a dusty stranger and a shadowed interior longer, but loses the desert, abandoned camp, and arrival sequence.

## Learning-Curve Context

Validation loss improves at every epoch, while each gain shrinks from 0.26144 to 0.02447. The train-validation gap reaches 0.16177 at epoch 5, up from 0.00807 at epoch 3. This is improving but flattening rather than established validation overfitting: token-level generalization still improves, but the widening gap and weak qualitative return show diminishing benefit from additional optimization under this setup.

## Interpretation

Five epochs establish that the three-epoch run was compute-limited on cross-entropy and next-token accuracy. They do not establish that more passes over the same corpus solve the model's main narrative deficiencies. The evidence supports extra compute as useful for token modeling while leaving prompt retention, state tracking, causality, repetition control, and story-level progression as the dominant bottlenecks.
