# Qualitative Assessment

| Category | Assessment |
|---|---|
| Static state | The auxiliary head cleanly recognizes lexical door, physical-condition, and relationship templates. |
| Dynamic updating | Depth-2 transition accuracy is chance/below chance; latest-state LM reversal remains 0%. |
| Prompt adherence | Continuations rapidly abandon named entities, objects, locations, and goals. |
| Fact retention | Manual review finds 0/13 greedy and 0/13 sampled; no expected fact is explicitly preserved. |
| Entity consistency | Names and entity-linked possession, relationship, location, and goal state disappear or mutate. |
| Causal progression | Causal-state auxiliary accuracy is around chance and generations do not preserve the disclosed causes or goals. |
| Grammar | Sampled text has locally sentence-like clauses but frequent identity confusion and malformed constructions. |
| Narrative progression | No sustained premise-driven progression is demonstrated. |
| Repetition | Greedy output remains highly repetitive; sampled lexical diversity does not recover semantic continuity. |

The raw candidate continuations contain generic overlap such as rooms, doors,
men, lamps, stations, or family language. None explicitly preserves the tested
entity-state relation, so none receives credit. The 4/13 forced-choice result
does not transfer to open continuation.

The evidence does not isolate autoregressive trajectory stability as the sole
remaining issue. Dynamic updating fails in the auxiliary head at depth 2, and
state information fails to improve teacher-forced LM scoring before free
generation begins.
