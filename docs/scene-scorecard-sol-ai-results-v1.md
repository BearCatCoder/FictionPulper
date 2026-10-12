# Matched Scene GPT-6.1 Sol AI-only results v1

## Disposition and evaluator provenance

Issue #41's owner-authorized AI replacement review completed on 2026-10-12
under [the frozen amended procedure](scene-scorecard-sol-ai-evaluation-v1.md),
committed as `cc5fb4c` before individual judgments were authored and aggregated.
GPT-6.1 Sol (`openai/gpt-6.1-sol`) inspected all **80 corrected public rows** in
the OpenCode conversation and authored fact/dimension labels with evidence and
notes for every row. No external model API invocation occurred. Decisions were
locked before the private model mapping was opened for aggregation.

There was **one actual AI evaluator**, no independent humans, no second review,
and no adjudication. Prior aggregate diagnostics, candidate length failure,
Benchmark v2 AI findings, and the first 20 public rows were already visible.
Anonymous row presentation does not make this independent or uncontaminated.
The historical human roles remain unassigned and historical independent review
remains incomplete. This completes amended #41 as AI-only exploratory evidence.

**Decision remains `STOP_NO_PROMOTION`.** Both models have 0/40 AI-labeled
coherent outputs. Candidate length alone caps possible coherence at 29/40
(72.5%), below the frozen 32/40 (80%) requirement. No showcase qualifies and #12
is not promoted. No outputs were regenerated, dropped, retried or rerolled.

## AI semantic judgments

Fact labels judge the complete entity–relation–value claim, including paraphrases
and unambiguous pronouns. Incidental Mallory, Mara, Silas, Jax, Pike, key, train,
or child references do not retain the required relations. This is a semantic AI
judgment, not the separate exact-phrase diagnostic, even where totals coincide.

| Dimension / desired label | Base | SFT candidate | Candidate minus base |
|---|---:|---:|---:|
| Atomic facts retained | 0/160 | 0/160 | 0 pp |
| Premise adherence: yes | 0/40 | 0/40 | 0 pp |
| Motive preservation: yes | 0/40 | 0/40 | 0 pp |
| Contradiction absent: no | 25/40 | 21/40 | -10 pp |
| Loop/repetition absent: no | 25/40 | 27/40 | +5 pp |
| Plot advancement: yes | 0/40 | 0/40 | 0 pp |
| Genre voice: yes | 0/40 | 1/40 | +2.5 pp |
| Coherent outputs | 0/40 | 0/40 | 0 pp |

Each model has 159 `not_retained`, one `contradicted`, zero `retained`, and zero
`uncertain` atomic-fact labels. Fact uncertainty and dimension uncertainty are
reported separately; uncertainty is not converted into an affirmative judgment.

| AI dimension | Base yes / no / uncertain | Candidate yes / no / uncertain |
|---|---:|---:|
| Premise adherence | 0 / 40 / 0 | 0 / 40 / 0 |
| Motive preservation | 0 / 40 / 0 | 0 / 40 / 0 |
| Contradiction present | 7 / 25 / 8 | 3 / 21 / 16 |
| Loop/repetition present | 15 / 25 / 0 | 13 / 27 / 0 |
| Plot advancement | 0 / 40 / 0 | 0 / 36 / 4 |
| Genre voice | 0 / 38 / 2 | 1 / 35 / 4 |

Unresolved evidence comprises **10/40 base outputs, 10 dimension judgments**;
**19/40 candidate outputs, 24 dimension judgments**. Across both arms that is
29/80 outputs with at least one unresolved label and 34/480 uncertain dimension
judgments. All 320 atomic-fact judgments are resolved. There are no review
disagreements to measure because only one evaluator exists.

Fewer definite candidate contradictions (3 versus 7) do **not** establish
improvement: ambiguous contradiction labels doubled (16 versus 8), and confident
absence actually fell (21 versus 25). The single positive genre label is candidate
`review-ecd5d672f4d33cea2f68`, development Western prompt, sampled seed 21337:
horse/cartridge/Marshal/camp coloring, with none of the required opening facts,
hostage-rescue premise, motive, or plot progression. It is not a qualifying scene.

## Required strata and descriptive uncertainty

Every role/trial/prompt/genre cell has zero retained facts, premise adherence,
motive preservation, affirmative plot advancement and coherent outputs. Each
role contains 20 outputs/80 facts; each trial contains 10 outputs/40 facts; each
role-by-trial contains 5 outputs/20 facts; each prompt and genre contains 4
outputs/16 facts. All 40 matched prompt/trial pairs are retained.

In the tables below, C means contradiction **present**, L means loop **present**,
G means genre voice **yes**; `U` is the unresolved count for that dimension.
All other affirmative semantic dimensions are zero in every row.

| Arm / role (20 outputs each) | Length pass | C yes / U | L yes | G yes / U | Plot U | Unresolved outputs / judgments |
|---|---:|---:|---:|---:|---:|---:|
| Base development | 20 | 3 / 4 | 8 | 0 / 2 | 0 | 6 / 6 |
| Base post-selection | 20 | 4 / 4 | 7 | 0 / 0 | 0 | 4 / 4 |
| Candidate development | 12 | 1 / 7 | 7 | 1 / 3 | 1 | 9 / 11 |
| Candidate post-selection | 17 | 2 / 9 | 6 | 0 / 1 | 3 | 10 / 13 |

| Arm / trial (10 outputs each) | Length pass | C yes / U | L yes | G yes / U | Plot U | Unresolved outputs / judgments |
|---|---:|---:|---:|---:|---:|---:|
| Base greedy | 10 | 2 / 0 | 10 | 0 / 0 | 0 | 0 / 0 |
| Base sampled 11337 | 10 | 0 / 4 | 0 | 0 / 1 | 0 | 5 / 5 |
| Base sampled 21337 | 10 | 3 / 2 | 3 | 0 / 1 | 0 | 3 / 3 |
| Base sampled 31337 | 10 | 2 / 2 | 2 | 0 / 0 | 0 | 2 / 2 |
| Candidate greedy | 5 | 2 / 2 | 10 | 0 / 0 | 0 | 2 / 2 |
| Candidate sampled 11337 | 8 | 1 / 5 | 1 | 0 / 1 | 2 | 6 / 8 |
| Candidate sampled 21337 | 9 | 0 / 5 | 2 | 1 / 1 | 2 | 6 / 8 |
| Candidate sampled 31337 | 7 | 0 / 4 | 0 | 0 / 2 | 0 | 5 / 6 |

The retained score report includes complete role-by-trial, per-prompt and genre
summaries, exact label counts, numerators/denominators and two-sided 95% Wilson
intervals for every desired-label proportion. Key pooled intervals, in percent:

- 0/160 fact retention: [0, 2.34] for each arm.
- 0/40 premise, motive, plot advancement and coherence: [0, 8.76] each.
- Contradiction absent: base [47.03, 75.78], candidate [37.50, 67.06].
- Loop absent: base [47.03, 75.78], candidate [52.02, 79.92].
- Genre voice: base [0, 8.76], candidate [0.44, 12.88].
- Mechanical length: base [91.24, 100], candidate [57.17, 83.89].

The prompt-block paired percentile bootstrap uses all four trials within each
of ten prompt clusters, 10,000 replicates and seed 270028. Candidate-minus-base
desired-label differences and 95% intervals, in percentage points:

| Measure | Difference | Interval |
|---|---:|---:|
| Fact / premise / motive / plot / coherence (each) | 0 | [0, 0] |
| Contradiction absent | -10 | [-35, 15] |
| Loop absent | +5 | [-7.5, 17.5] |
| Genre voice | +2.5 | [0, 7.5] |
| Mechanical length compliance | -27.5 | [-42.5, -12.5] |

These intervals are conditional/descriptive. Facts within scenes are dependent;
ten prompts and small genre/role/trial cells are underpowered. Degenerate
all-failure bootstrap intervals do not establish model equivalence, population
certainty, or judge correctness. No confirmatory or causal SFT claim is supported.

## Separate mechanical, lexical and likelihood evidence

Frozen 150–250-word compliance is computed on continuation text only.

| Length classification / retained stop reason | Base | Candidate |
|---|---:|---:|
| Compliant / max tokens | 40 | 27 |
| Compliant / EOS | 0 | 2 |
| Short / EOS | 0 | 8 |
| Short / max tokens | 0 | 1 |
| Overlength / max tokens | 0 | 2 |

The 11 candidate failures are **nine short plus two overlength**, not eleven
short. Eight short outputs stopped at EOS; one exhausted the 300-token budget.
Both overlength outputs also exhausted that budget. Ten candidate outputs stop
at EOS in total. This establishes observed stopping/length categories, not why
the model chose EOS or why tokenization yielded those word counts.

The unchanged automated diagnostics report exact fact-phrase mentions 0/160 per
arm, mean repeated 4-gram rate 0.228872 → 0.211395, and repeated-sentence rate
0.118346 → 0.158970. These lexical measures are distinct from the AI semantic
loop count (15 → 13), which covers narrative loops and descriptive restatement.
Their mixed directions are not a contradiction or a narrative-quality score.

Historical issue #9 likelihood, not rerun here: continuation test NLL
4.061925 → 3.816288; clean-fiction loss 3.101755 → 3.107070. Neither improvement
in continuation likelihood nor the small genre-label difference offsets the
semantic and mechanical failures.

## Bounded development-diagnosis handoff to #42 / #43

Observed failure is **lack of dependable relation-level conditioning and
consequential continuation**, with deterministic looping in all 10/10 greedy
outputs of each arm. Sampled outputs often retain names or local nouns while
changing scene, role, referent, or objective. Development and post-selection
both fail; this is not evidence limited to one cherry-picked generation.

Development-only examples for #42 to characterize retrospectively:
`review-7c9f501bfd53d23e9c8a` stalls in a generic case intention;
`review-70176fbfa083bfc14827` retains Mallory’s name but drifts into maritime
action; `review-ecd5d672f4d33cea2f68` has Western coloring without hostage-rescue
relations. These examples illustrate dimensions, not a representative sample
or a new selection/training set.

This review **does not distinguish** weak immediate conditioning from progressive
rollout decay, serialization mismatch, or teacher-forced/free-generation mismatch.
The next action is #42's separately preregistered, bounded development-only
single-fact/motive perturbation diagnosis, with context-removed and candidate-only
teacher-forcing controls, early-versus-later free-generation measurements, and
serialization/truncation checks. Use these length/stop findings rather than
assuming all failures are early EOS. Keep all inspected Scene/Benchmark held-out
outputs retrospective and excluded from new training/selection. #43 can choose
one intervention only after that diagnosis; no further SFT sweep is authorized.

## Retained identities and reproduction

Local AI artifacts are ignored under `runs/scene-sol-ai-review-v1/`; raw outputs,
private keys, decisions and report are retained locally rather than committed.

| Artifact | SHA-256 |
|---|---|
| AI decisions `decisions.json` | `83fcde9528a77f912e83d2ab0fd6e71fd5cb60fa4b11196503836c6d5c0035fc` |
| Original AI `score-report.json` | `7821f35075f244e69222fa3a56faa62ebd2fb4bcdb09af8ff42e2884afc48bea` |
| Amended procedure | `58d216bafb230f2dd870c5e6da3a4aa96dc9aa8e2d7ae2a3f68c824b793cbe4a` |
| AI scorer `src/scene_scorecard_ai_review.py` | `cdb372cb657f6a0c43de3edf88d84cd554c78c3f4e1f926caa3ee3e1f1298e09` |
| Corrected matched packet | `69ce3baf7efd378fcc3f7cfa94e6838287bfc89ab1c11918f4575d11c3b78a5b` |
| Corrected private key | `3ef261147a5ed9cdc27b0b4c3357e3a45b9bdd6482b1afd2b3473ade193ce86c` |
| Correction report | `3521d0af111b8ebfcd151bf9b21ac0617913830e7a236a84a7b1c2ec483d163f` |
| Sealed base raw matrix | `9bb73421e59ba23a1379931fbd3b9bbca169750bfb64359cc3616a1c77ccda69` |
| Candidate raw matrix | `ee91eb1c1ddffa0795e672797731f72c4959062e76904cd11bebdc79386aa767` |
| Source post-selection report | `8f84a415bfa200aba250db660638bf471222190bfa9c3e4d70cfa9f40fcb7cdd` |
| Post-selection protocol | `f45041dcfb87e4f3dea8c0abce988beb346a8d77991305650cd8e46bebed0b87` |
| Frozen Scene protocol | `7c438ad612d7c344f17ab32843f230e49ef3194565fc0aff45a4a06cb41f2718` |
| Scene prompts | `fe828244fec8fda482452792451ecb86d34518ee5c5695646f76b3a43f04ff16` |
| Human rubric | `cee906f1b5b3d93acb92120e3403c40776bc0da30f143453d1b3e5108bbffb8b` |

Checkpoint identities from the verified retained source report/key are base
`8da266fb064719d3f38e78d672b2e5cec0f14551c37658210322b9f46cbecd01`
and candidate `7d4ef889435626304cac24a2ce651ffd20ea9d85a5f882b10d30033a6d2d3ad5`.
Checkpoint binaries need not be loaded to reproduce these judgments or scores.
Original scoring ran at `cc5fb4c`, with the new scorer/tests untracked and the
scorer hash bound above. The recorded clean tracked worktree excludes untracked
files. Retained source/decision/procedure bytes are checked before aggregation;
the packet and metadata are rebuilt in memory and compared exactly.

From the repository root, using the retained judgments rather than silently
regenerating them:

```bash
source .venv/bin/activate
sha256sum runs/scene-sol-ai-review-v1/decisions.json
python -m src.scene_scorecard_ai_review \
  --reviews runs/scene-sol-ai-review-v1/decisions.json \
  --output runs/scene-sol-ai-review-v1/score-reproduction-v1.json
python -m unittest tests.test_scene_scorecard_ai_review tests.test_scene_scorecard
python -m unittest discover -s tests
```

The output must be absent: scoring refuses overwrites. Scores and strata are
deterministic; Git provenance and report byte hash change with the checkout.
The scorer makes no evaluator calls and cannot reproduce human judgments.
Verification: **16 focused tests, 400 full-suite tests passed**. Evidence-boundary
tests reject human impersonation, independence claims, partial/duplicate rows,
wrong bindings, invented quotes, unsupported affirmations, and wrong empty-text
markers; they verify uncertainty, coherence, matched pairing, and overwrite refusal.
