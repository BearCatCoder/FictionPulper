# Benchmark v2 GPT-6.1 Sol AI-only results v1

## Disposition

Issue #28's explicitly authorized AI replacement evaluation completed on
2026-10-12 under [the amended procedure](benchmark-v2-sol-ai-evaluation-v1.md).
GPT-6.1 Sol (`openai/gpt-6.1-sol`) inspected all 80 public Stage 1 rows in the
OpenCode conversation and authored a separate decision for every row. No external
model API invocation occurred. Decisions were locked before private aggregation
metadata was loaded. The AI decision file contains no human attestation.

**Result: 0 consistent / 80 inconsistent / 0 uncertain.** All failures are
omissions of the required terminal proposition. Rows 13, 21, 55, and 65 contain
whitespace only; their bytes are nonempty, so they use omission evidence rather
than the zero-byte `<EMPTY_OUTPUT>` marker. Incidental Alden/Delia mentions do
not establish the required proposition. The original saved human row remains
separate and is not included in this evaluation.

| AI-labeled matched result | arm-1 | arm-2 |
|---|---:|---:|
| Reviewed outputs | 40 | 40 |
| Matched A/B pairs | 20 | 20 |
| Successful pairs | 0 | 0 |
| Strict pair consistency | 0% | 0% |
| Uncertain pairs | 0 | 0 |
| Resolved-only pair consistency | 0% | 0% |

The arm-2 minus arm-1 difference is **0.00**. The paired percentile bootstrap
(10,000 replicates; seed 260209) returns a 95% interval **[0.00, 0.00]** and width
0.00. This is a degenerate resampling result conditional on all failures in
these 20 deterministic blocks and this AI evaluator. It does not demonstrate
model equivalence, population certainty, or correctness of AI judgments.

## Coverage and supported comparisons

All 20 matched quartets retain both models and both A/B worlds. Per arm, every
trial and depth contains five pairs; split counts are development 6, test 7,
generalization 7. The causal-dependency family contains two pairs and each of
goal, knowledge, location, ownership, static-fact, and temporal-ordering contains
three pairs. All descriptive cells have zero successful and zero uncertain
pairs, with exact sizes retained in the local score report. Full
family-by-depth-by-split coverage is absent.

Only the overall matched difference is exploratory; arm rates and small strata
are descriptive. No powered, representative, confirmatory, agreement,
adjudication, human-owner, independent-human, or full-population claim is made.

## Adapted Stage 2 decision

**STOP_AFTER_STAGE_1**; zero additional blocks/rows authorized.

- CI width 0.00 does not exceed 0.30.
- The interval includes zero, but absolute difference 0.00 is below 0.10.
- Both uncertain-pair rates are 0.00, below the greater-than-0.10 trigger.

No Stage 2 packet was generated. This applies the historical numerical thresholds
to AI labels under the owner's amended authorization. It is not the official
human-stage decision or a new preregistered human result. Prior GPT-6 Astra
aggregate findings were visible, so this is not independent replication.

## Separate automated diagnostics

The frozen lexical diagnostics report zero premise-retention, current-fact-only,
and causal-adherence rates in each arm/trial overall cell. Explicit contradiction
and stale-fact rates are also zero: these do not rescue omission failures.

| Trial (10 outputs per arm) | arm-1 repeated 4-gram mean | arm-2 repeated 4-gram mean |
|---|---:|---:|
| greedy | 0.489572 | 0.600800 |
| sampled 11337 | 0.072727 | 0.085375 |
| sampled 21337 | 0.207905 | 0.125206 |
| sampled 31337 | 0.182609 | 0.079975 |

All automated diagnostic strata are retained separately in the score report.
Forced choice, teacher forcing, fiction likelihood, and Scene Scorecard review
were not rerun. These retrospective results may inform #42/#43's development
diagnosis, but do not authorize training on inspected holdouts or promote #12.

## Artifact identities and provenance

Local artifacts remain ignored; raw outputs and decision artifacts are not
committed. Paths below are relative to `runs/benchmark-v2/owner-review-v1/`.

| Artifact | SHA-256 |
|---|---|
| `stage-1-sol-ai-decisions-v1.json` | `848cb6d31e1c4445d39959a830ed203d764b9618b803abd062be82d088474dc9` |
| `stage-1-sol-ai-score-report-v1.json` | `f7b7702b7e82dbd3fd7a44624fa2236721b5ba3d0799d7c9e0d3c5ee3a18f936` |
| `stage-1-owner-review-packet.jsonl` | `ce11fad2c07a412c8aa7b3cbc1eccc59d6d5d4c2b982cdea4fb5696397e92a0b` |
| `stage-1-public-manifest.json` | `ac62a19faec0e47c51eb3823d21039f1c03900de188199076d7659b5452f7c15` |
| `stage-1-private-key.json` | `ca7bf26a5339c3bf6bd05d1ca3f5c65180c42a252b360b99c6756c32867ab5b1` |
| Preserved `stage-1-owner-reviews.jsonl` (one row) | `d1d3ce886f3fe923602d98b7266a57b6952c1f5c86eeb73d5af21100c82064d5` |

AI scorer: `src/narrative_v2_ai_review.py`, SHA-256
`763649562bceb8d425435a5af1de2b5327b9d0896466aaa596e47e30cc7e022b`.
Unchanged aggregation implementation: `src/narrative_v2_owner_review.py`, SHA-256
`87d0a98ca053f4940dc7229a573f2eea1b58c11023621bab2573dddd26c0efac`.
Scoring ran from parent commit `169a440298f8d03dbb8fb94a32326041eb5b5eee`, with the
new AI module untracked and hash-bound above. The report's
`tracked_worktree_clean: true` describes tracked files only, not absence of the
new untracked module. The report retains all six source identity hash bundles.
The unchanged historical validators verified all 1,792 source outputs and the
existing full/staged identities before aggregation.

## Reproduction

The historical AI judgments are an input artifact, not generated by the scorer.
Use the retained hash-locked decision file; do not silently regenerate judgments.
From the repository root:

```bash
source .venv/bin/activate
sha256sum runs/benchmark-v2/owner-review-v1/stage-1-sol-ai-decisions-v1.json
python -m src.narrative_v2_ai_review \
  --packet runs/benchmark-v2/owner-review-v1/stage-1-owner-review-packet.jsonl \
  --manifest runs/benchmark-v2/owner-review-v1/stage-1-public-manifest.json \
  --reviews runs/benchmark-v2/owner-review-v1/stage-1-sol-ai-decisions-v1.json \
  --output runs/benchmark-v2/owner-review-v1/stage-1-sol-ai-score-reproduction-v1.json \
  --checkpoint-selection-complete \
  --test-evaluation-complete
python -m unittest tests.test_narrative_v2_ai_review tests.test_narrative_v2_owner_review
python -m unittest discover -s tests
```

The output must be absent; the scorer refuses overwrites. Scoring and source
verification are deterministic, while the report's Git provenance and its byte
hash will differ if the checkout differs from the original execution commit.

Verification: 17 focused tests and **392 full-suite tests passed**. New tests
reject AI impersonation, human attestations, independence claims, partial or
duplicate decisions, wrong artifact bindings, invented quotations, and incorrect
zero-byte/whitespace evidence. The historical human workflow is unchanged.

The authorized AI disposition satisfies amended issue #28 only. The historical
human owner stage remains incomplete (one saved row, no final attestation);
strict independent validation remains pending in #38.
