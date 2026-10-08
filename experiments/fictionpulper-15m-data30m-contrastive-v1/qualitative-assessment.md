# Qualitative Behavior Assessment

| Category | Contrastive-v1 assessment |
|---|---|
| Grammar | Sampled clauses are often locally grammatical; agreement and malformed constructions remain. |
| Complete sentences | Similar to baselines; fixed-length continuations can stop mid-sentence. |
| Prompt adherence | No reliable improvement. Prompt entities and requested premises are abandoned. |
| Semantic continuity | No improvement: all 13 exact entity-state facts fail conservative semantic scoring in both modes. |
| Dialogue coherence | Dialogue has plausible turn formatting but speakers and subjects drift. |
| Entity consistency | No demonstrated improvement; names and entity-linked possession, relation, location, and goal states disappear. |
| Scene persistence | Generic room/house/window language persists locally but not the disclosed named scene. |
| Causal progression | Events are juxtaposed without maintaining disclosed causes, obligations, or goals. |
| Narrative progression | No sustained premise-driven arc is demonstrated. |
| Repetition | Narrative-suite greedy repetition improves versus both baselines, but severe loops remain; sampled differences are mixed and small. |

Generic overlap was not credited: for example, `key`, `station`, `house`, `window`, and family/marriage words occur without the required entity-state relationships. High synthetic-pair ranking therefore coexists with 0/13 free-generation retention. The pair benchmark is limited by synthetic templating, and the probe is limited by lexical-rank labels. Object-location is the sole convincing representational improvement; this is not broad narrative-state change.
