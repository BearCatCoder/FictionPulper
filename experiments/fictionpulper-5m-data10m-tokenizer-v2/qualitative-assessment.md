# Qualitative Assessment

The same ten prompts, greedy and sampled token limits, sampling seed 11337, temperature 0.8, top-k 50, and top-p 0.95 were used for all three models.

| Criterion | Smoke v1 | Data10M v1 | Tokenizer v2 |
|---|---|---|---|
| Grammar | Frequent malformed clauses and basic agreement failures | Longer grammatical spans, but persistent agreement and syntax errors | Similar to Data10M v1; occasional longer well-formed spans but no consistent improvement |
| Complete sentences | Often collapses into fragments or loops | More complete sampled sentences | More complete sampled sentences, but many outputs still end mid-sentence at the token limit |
| Prompt adherence | Very weak | Weak; most prompts lose their defining event quickly | Weak; station and western cues sometimes persist, while rocket, Europa, portrait, and detective premises are usually abandoned |
| Semantic continuity | Minimal | Local paragraph-level continuity appears, followed by drift | Local continuity appears, but topic and causal drift remain common |
| Dialogue | Unstable quotation and speaker structure | Recognizable dialogue turns, often semantically inconsistent | Recognizable dialogue and attribution syntax, still with inconsistent speakers and responses |
| Repetition | Severe phrase loops | Severe in greedy decoding; reduced but present in sampling | Severe in greedy decoding; no reliable reduction versus Data10M v1 |
| Entity consistency | Very poor | Names and roles are introduced but rarely maintained | Names and roles are introduced but often replaced; Mallory and prompted entities usually disappear |
| Scene persistence | Rare | Occasional room, street, or landscape persistence | Occasional station, room, and camp persistence; science-fiction settings usually collapse into generic historical prose |
| Narrative progression | Little | Some sampled passages introduce actions and exchanges, but rarely form an arc | Some actions and exchanges occur, but no dependable setup-to-consequence progression |

Tokenizer v2 does not produce a clear qualitative improvement over Data10M v1. Both Data10M models are substantially more prose-like than smoke v1, but the remaining limitations appear dominated by model/data capability rather than tokenizer segmentation.
