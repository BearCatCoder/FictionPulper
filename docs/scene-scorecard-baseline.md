# Scene Scorecard v1 Baseline

## Decision

Issue #27 freezes `benchmarks/scene-scorecard-v1/` as the scene-quality contract
for the issue #8 data pilot, issue #9 bounded SFT comparison, and promotion of
issue #12. Narrative Benchmark v2 and every sealed historical artifact are
unchanged. This scorecard does not replace the outstanding Benchmark v2 human
review.

The scorecard fixes ten prompts and 40 outputs: one greedy output and sampled
outputs at exact seeds `11337`, `21337`, and `31337` for every prompt. Five
`scene-dev-*` prompts deliberately adapt the previously inspected historical
genre suite and are regression diagnostics. Five newly authored `scene-post-*`
prompts are post-selection evaluation. The latter must never enter issue #8
training/validation/test data, contamination-driven editing, issue #9 candidate
selection, or hyperparameter selection.

The selected baseline is the sealed, validation-selected, from-scratch
50M/Data30M checkpoint. Generation fails before model load unless these bytes
match:

| Artifact | SHA-256 |
|---|---|
| Checkpoint | `8da266fb064719d3f38e78d672b2e5cec0f14551c37658210322b9f46cbecd01` |
| Experiment seal | `ac9f0abf9c3c1c60f0ee02485918e86c8407e051c578e34cf1bfa7ae15451fb3` |
| Tokenizer v1 | `14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012` |
| Scene protocol | `7c438ad612d7c344f17ab32843f230e49ef3194565fc0aff45a4a06cb41f2718` |
| Prompt suite | `fe828244fec8fda482452792451ecb86d34518ee5c5695646f76b3a43f04ff16` |
| Human rubric | `cee906f1b5b3d93acb92120e3403c40776bc0da30f143453d1b3e5108bbffb8b` |

## Frozen Criteria

Both greedy and sampled generation use a 300-new-token maximum. Sampling keeps
the existing settings: temperature `0.8`, top-k `50`, and top-p `0.95`. Output
text is never truncated. EOS, maximum-token, short, compliant, and overlength
states remain in raw records. Word count applies the protocol's ASCII word
regex to the decoded continuation only; 150-250 words is compliant and 200 is
the target.

Human review covers every atomic opening fact, premise adherence, motive
preservation, contradictions, loops/repetition, plot advancement, and genre
voice. Automated phrase matching, word counts, and repetition measurements are
diagnostic only. Two genuine independent reviewers score every blinded output;
any disagreement or uncertainty requires a third independent adjudicator.

The preregistered suite gates are at least 70% retained atomic facts, at least
80% premise adherence, at least 80% coherent outputs, and at least 50% coherent
outputs for every prompt. A coherent output retains at least three of its four
facts, adheres to the premise and motive, has no contradiction or loop,
advances the plot, has genre voice, and meets the word range. Reports include
numerators, denominators, and 95% Wilson intervals. Forty outputs and 160 fact
judgments provide only wide descriptive uncertainty, especially by genre.

The eventual human report preserves output-level scores and reports the complete
40-output suite both as a clearly labeled pooled summary and as deterministic
strata. Role strata separate the 20 development/regression outputs from the 20
reserved post-selection outputs. Trial-cell strata separately report greedy and
sampled seeds `11337`, `21337`, and `31337`, each with 10 outputs and 40 atomic
fact judgments. Every stratum reports fact retention, premise adherence,
coherent outputs, and the human dimensions with numerators, denominators, and
95% Wilson intervals. These strata are descriptive: the frozen promotion gates
continue to apply only to the pooled complete 40-output suite and are not
redefined after baseline generation.

Showcase selection is unavailable unless all suite gates pass. It then considers
only coherent post-selection outputs and deterministically orders by retained
facts, distance from 200 words, prompt ID, greedy before sampled, and sample
seed. If the suite fails or none qualify, there is no showcase.

## Baseline Execution

Run once from the repository root in the existing environment:

```bash
source .venv/bin/activate
python -m src.scene_scorecard baseline
```

The write-once output directory is
`runs/scene-scorecard-v1/fictionpulper-50m-data30m-v1/`. It contains raw
generations, diagnostics, a manifest, the blinded review packet, and its private
key. These machine artifacts are ignored by Git; this compact report is the
tracked record.

Generation completed on 2026-10-10 using CUDA BF16 and produced the complete
40-output matrix. No output was retried, rerolled, truncated, or dropped. All 40
reached `max_new_tokens`; none emitted EOS. Word counts ranged from 154 to 249,
all 40 were mechanically length-compliant, and mean word count was 200.875.

Automated diagnostics found zero exact full-fact phrase mentions among 160
fact/output opportunities, mean repeated 4-gram rate 0.228872, and mean repeated
sentence rate 0.118346. Exact phrase matching misses valid paraphrases and
repetition rates do not establish narrative loops, so these are warning signals,
not human fact-retention, coherence, or promotion scores. They reinforce the
pilot emphasis on semantic fact/motive retention and non-repetitive progression.

Independent human review is **pending**. No human fact, premise, motive,
contradiction, loop, advancement, voice, coherent-output, or promotion score
exists. The suite has not been promoted and there is no showcase.

| Run artifact | SHA-256 |
|---|---|
| `raw-generations.jsonl` | `9bb73421e59ba23a1379931fbd3b9bbca169750bfb64359cc3616a1c77ccda69` |
| `diagnostic-summary.json` | `b1a7bcfd5603faf0928be50855e46a43530cb8d4338e9b7542b58f9f8c103a3a` |
| `blinded-review-packet.jsonl` | `6667c181e36522efef96fd6f409956a729ba5ce30314c459d89f1ded63588a38` |
| `private-review-key.json` | `4ed5236bc609e948a492c71c6421365c9928fcb84e9a74eab5c73aab7276bc9e` |
| `manifest.json` | `e22a50a97cba4b11ce3433ea987e11c7cc2d1f3c0677616923b84bd324e4cb50` |

## Issue #8 Pilot Specification

The baseline's established failure profile is loss of opening facts and motives,
premise drift, weak event progression, contradiction, and repetition. Issue #8
should build exactly 200 complete, rights-cleared natural-continuation examples
before any expansion decision:

- Each example contains a 60-120-word opening and a 150-250-word continuation,
  with genre, premise, protagonist/motive, and three to five atomic opening
  facts stored as audit metadata but never injected as answer labels into prose.
- Target 40 examples in each of crime/noir, horror/occult, science fiction,
  western/adventure, and weird fantasy, while varying names, settings, actions,
  sentence forms, and endings within each group.
- Include meaningful action and consequence in every continuation. During
  review, require premise and motive preservation, no opening-fact
  contradiction, no narrative loop, plot advancement, and recognizable genre
  voice under this scorecard's definitions.
- Group split by original story/source and duplicate cluster into 160 train, 20
  validation, and 20 test examples. Keep test post-selection and use validation
  only for issue #9 checkpoint selection. Report exact tokenizer-v1 token counts.
- Require documented rights/provenance for every example, two-person human
  approval of every example, adjudication of disagreements, exact-duplicate and
  normalized five-word-shingle audits, and checks against Benchmark v2 plus all
  ten scene-scorecard prompts.
- Reject any example sharing a scene-scorecard premise, names, distinctive fact
  combination, or normalized five-word shingle. The five `scene-post-*` prompts
  are categorically barred from train, validation, test, and candidate selection.
- Freeze serialization as `<|story|><|bos|>` plus opening text followed directly
  by continuation text and `<|eos|>`. Issue #9 loss applies only to continuation
  and EOS targets; opening and padding positions are masked. Audit truncation
  and EOS before approval.

Issue #9 compares the unchanged base and its one preregistered continuation-SFT
candidate on this exact 40-output matrix with no retries or rerolls. Issue #12
uses the same hashes, criteria, blinded review, uncertainty, and showcase rule.
Neither training completion nor lower perplexity can substitute for these human
promotion gates.
