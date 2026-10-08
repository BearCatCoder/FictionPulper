# Qualitative Behavior Assessment

| Category | Assessment |
|---|---|
| Grammar | Sampled clauses are intermittently grammatical, but malformed constructions and identity confusion remain. |
| Prompt adherence | All six continuations abandon the supplied names and premises. |
| Semantic continuity | No disclosed fact is explicitly retained: 0/13 greedy and 0/13 sampled. |
| Entity consistency | Names, possession, relationships, locations, secrets, goals, and scene markers disappear. |
| Dynamic updating | Depth-2 CURRENT selection is chance and latest-state exact reversal is 0%. |
| Dialogue coherence | Dialogue formatting is plausible, but speakers and referents drift. |
| Causal progression | Outputs do not preserve disclosed causes, obligations, or goals. |
| Narrative progression | No sustained premise-driven arc is demonstrated. |
| Repetition | Greedy decoding loops heavily; sampled output is less repetitive but incoherent. |

Generic overlap was not credited. For example, one continuation mentions a
window but not the lower window's blue chalk circle; another mentions a husband
but not the specified sister-in-law relationship. See `manual-fact-scoring.json`.
