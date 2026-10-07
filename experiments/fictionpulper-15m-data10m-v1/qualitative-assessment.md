# Qualitative Assessment

The 5M and 15M models used the same ten prompts, tokenizer, greedy and sampled output lengths, sampling seed 11337, temperature 0.8, top-k 50, and top-p 0.95.

| Criterion | 5M Data10M | 15M Data10M |
|---|---|---|
| Grammar | Sampled text often forms sentences but has frequent agreement and construction errors | Some sampled passages sustain longer sentence structures, but agreement errors and malformed phrases remain common |
| Sentence completion | Many complete local sentences, with frequent truncation or degeneration | Slightly stronger local sentence formation; outputs still truncate at the fixed token limit and greedy text often degenerates |
| Prompt adherence | Usually loses the defining prompt event quickly | Occasionally preserves immediate scene terms longer, such as floor, door, stranger, road, or western dialect, but usually loses the central event |
| Semantic continuity | Brief local continuity followed by topic drift | Some passages maintain a local description or conversation longer, but global causal continuity remains weak |
| Dialogue coherence | Dialogue syntax is recognizable but responses are poorly connected | Longer dialogue exchanges appear, especially in sampled text, but speaker intent and responses remain inconsistent |
| Repetition | Severe greedy loops and repeated generic constructions | Severe greedy repetition remains and is not consistently improved by capacity |
| Pronoun/entity consistency | Names and roles drift rapidly | Entity descriptions are sometimes richer, but gender, age, role, and identity still change within passages |
| Scene persistence | Occasional room or landscape persistence | Modest gains in local scene vocabulary, but science-fiction and mystery scenes still collapse into generic historical prose |
| Narrative progression | Actions rarely lead to coherent consequences | Some sampled passages contain more actions and transitions, but dependable narrative progression is still absent |

The 15M model is quantitatively better and occasionally sustains richer local prose, but the qualitative gain is modest rather than transformative. Additional capacity helps language modeling, while repetition, prompt adherence, and long-range narrative coherence remain unresolved.
