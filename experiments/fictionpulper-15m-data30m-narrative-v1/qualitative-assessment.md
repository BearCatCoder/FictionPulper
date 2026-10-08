# Qualitative Assessment

## Verdict

**The curriculum learned as a language-modeling distribution but did not improve narrative state retention.** The candidate remains a locally plausible but weak story generator. Sampled outputs are less repetitive than greedy outputs, yet both models abandon premises, entities, goals, and disclosed facts. The candidate provides no material qualitative improvement over Compute5.

| Category | Candidate versus Compute5 |
|---|---|
| Grammar | Similar. Sampled passages often have locally grammatical clauses, but agreement and malformed constructions remain. |
| Complete sentences | Similar. Fixed-length continuations frequently stop mid-sentence. |
| Prompt adherence | No material improvement. Occasional prompt vocabulary survives, while core premises are usually abandoned. |
| Semantic continuity | No improvement. Local transitions can read smoothly, but events and attributes conflict or drift. |
| Dialogue coherence | No reliable improvement. Speaker turns are formatted plausibly but are circular or unrelated. |
| Repetition | Mixed. Candidate repetition improves modestly on aggregate, especially sampled output, but greedy loops remain severe. |
| Entity consistency | No demonstrated improvement; names, roles, possession, relationships, and locations are omitted or replaced. |
| Scene persistence | No reliable improvement. A scene may persist locally without preserving the requested setting or facts. |
| Causal progression | No improvement. Continuations juxtapose events rather than developing consequences from prior state. |
| Narrative progression | No material improvement. The ten prompts do not develop sustained premise-driven arcs. |

## Narrative Diagnostic

The unchanged three-prompt protocol contains 13 facts, all visible within the 1,024-token context. Manual audit found candidate greedy `0/13`, candidate sampled `0/13`, Compute5 greedy `0/13`, and Compute5 sampled `0/13`. Generic word overlap was not credited without the required entity-state relation. This semantic result independently confirms the token-exact held-out result rather than attributing failure solely to exact wording.

## Curriculum Damage

Replacing 15% of real-fiction targets caused no material real-fiction LM damage: Data30M test loss improved about 0.09%, perplexity improved about 0.29%, and next-token accuracy improved about 0.18%. Those tiny changes are effectively neutral at one run per condition and do not establish a benefit. Qualitative storytelling and semantic state retention also did not improve, so the curriculum consumed target exposure without achieving its intended behavior.
