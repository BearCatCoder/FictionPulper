# Qualitative Assessment

## Verdict

Data30M is mixed. Sampled outputs show modest gains in local grammar, complete clauses, short-range action, and occasional dialogue structure. They do not show a corresponding improvement in sustained narrative control. Greedy decoding still degenerates into repeated templates, and most continuations rapidly discard the prompt's subject.

| Criterion | Assessment |
|---|---|
| Grammar | Mixed, with a slight sampled improvement; malformed descriptions remain common. |
| Complete sentences | Mixed; sampled passages finish more local sentences, but outputs still break or end mid-thought. |
| Prompt adherence | Weak; locked-door and revolver samples retain a few relevant objects, while most prompts are abandoned. |
| Semantic continuity | Slight sampled improvement over adjacent actions, without reliable causal linkage. |
| Dialogue coherence | Slight improvement in recognizable exchanges, but speakers and propositions remain unstable. |
| Repetition | No meaningful improvement; every greedy continuation enters a phrase or sentence-template loop. |
| Entity consistency | Poor and sometimes worse; age, gender, role, and relationship drift within paragraphs. |
| Scene persistence | Mixed; the locked-door scene persists longer, but most outputs jump among unrelated locations. |
| Narrative progression | Slight sampled improvement only; prompts rarely develop into setup, consequence, and resolution. |

## Persistent Weaknesses

- **Generic historical-prose collapse:** science-fiction, horror, mystery, western, and fantasy prompts still become generic houses, villages, doctors, policemen, horses, and domestic biographies.
- **Loss of prompt subject:** Mallory, Clara, Europa, the rocket, portrait, tower, signal, bullets, and floorboard movement usually disappear immediately.
- **Identity drift:** the Dry Creek sample changes a man into a boy, wife, and father; other samples similarly change gender and age.
- **Repetitive greedy loops:** examples include `I am going to tell you`, `I have been in the house`, `I'm a good man`, and `I'll go and see you` repeated without progression.
- **Weak causal progression:** generated actions seldom respond to the contradiction, threat, or mystery established by the prompt.

## Prompt Evidence

- The locked-door sampled continuation retains a door, room, occupants, and questioning longer than Data10M, but contradicts the locked premise by opening the door and loses Mallory.
- The revolver sampled continuation introduces a target, policeman, case, and police, improving genre alignment, but never reasons about the dead man or bullet count.
- The rocket sampled continuation opens with a black light in the sky and a voice, a slight atmospheric improvement over Data10M's kings and princes, then collapses into a gas lamp, dog, and kitchen.
- The desert greedy continuation regresses into `I'll go and see you`; the sampled continuation hints at scenery and a barkeeper but never reaches the mining camp.
- The Europa and shadowless-tower prompts still collapse into domestic architecture and contradictory family biography rather than preserving speculative elements.

## Interpretation

The larger unique corpus produced a clear token-level generalization improvement on the common held-out subset, but only shallow qualitative gains. It improved local sampled fluency more than prompt retention, entity tracking, scene state, or story-level causality. This experiment therefore supports more unique fiction as useful while leaving the main narrative-coherence bottleneck unresolved.
