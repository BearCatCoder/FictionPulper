# Qualitative Behavior Assessment

| Category | Counterfactual-v1 assessment |
|---|---|
| Grammar | Sampled clauses are often locally grammatical, but malformed constructions and identity confusion remain. |
| Prompt adherence | Prompt entities and premises are usually abandoned. |
| Semantic continuity | No disclosed fact was retained under conservative free-generation scoring: 0/13 greedy and 0/13 sampled. |
| Entity consistency | Names, possession, relationships, locations, goals, and scene markers disappear. |
| Latest-state tracking | No transfer: paired reversal is 0% on both held-out splits. |
| Dialogue coherence | Turn formatting is plausible, but speakers, referents, and relationships drift. |
| Causal progression | Generated events do not preserve disclosed causes, obligations, or goals. |
| Narrative progression | No sustained premise-driven arc is demonstrated. |
| Repetition | Greedy decoding regresses substantially; sampled decoding is less repetitive but remains incoherent. |

Generic overlap was not credited. Words such as `house`, `window`, `manuscript`, or marriage language occur without the required entity-state relationship. The 6/13 forced-choice result therefore does not transfer to open-ended continuation.
