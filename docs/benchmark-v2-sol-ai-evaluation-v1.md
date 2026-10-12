# Issue #28: authorized GPT-6.1 Sol AI evaluation v1

## Acceptance amendment — 2026-10-12

The owner explicitly authorized "use AI instead of human" in the execution
conversation. Issue #28 now closes on an **AI-only exploratory disposition**,
not completion of the historical single-owner human stage. The previous human
acceptance criteria remain historical requirements for any future human claim.
The one saved BearCatCoder decision is preserved and excluded from AI scoring.
The terminal human-review command and its attestation gates remain unchanged.
Independent human validation stays pending in issue #38.

## Bounded procedure (recorded before scoring)

- Evaluator: GPT-6.1 Sol (`openai/gpt-6.1-sol`) in this OpenCode conversation.
- Review all 80 existing Stage 1 public rows, with no retries or dropped rows.
- Use the unchanged rubric: consistent requires all required propositions and
  no forbidden proposition; omission is inconsistent, not uncertain.
- Record individual AI labels, confidence, omission evidence, and rationale in
  a new ignored artifact. Never impersonate BearCatCoder or attest human review.
- Review uses the public packet only. Prior AI aggregate findings and historical
  project results were already visible in this conversation; this is not an
  independent or uncontaminated replication. No private model mapping is used to
  decide labels. Private metadata is used only after decisions are locked, for
  matched aggregation and source validation.
- Revalidate all six frozen source sets and the existing full/staged artifacts.
  Preserve frozen protocols/configs, generation outputs, packets, keys, and the
  existing human review record byte-for-byte.
- Apply the existing matched-pair aggregation, 10,000 bootstrap replicates,
  seed 260209, and numerical Stage 2 thresholds to AI labels in a separate
  scorer. This is an **AI adaptation** of a human preregistration, not the
  official owner-stage decision. If triggered, keep the issue open and freeze
  a separate AI Stage 2 protocol before expansion. If not, stop after Stage 1.
- Publish provenance, exact hashes, coverage, descriptive strata, conditional
  uncertainty, separate automated diagnostics, and limitations. Do not infer
  model equivalence from a degenerate bootstrap or generalize to all 1,792 rows.
- Merge the focused PR and mark #28 Done only after the authorized AI disposition
  is complete. Human owner-stage completion remains absent; no promotion of the
  coherent-scene milestone follows.

## Acceptance checklist

- [ ] Complete 80 distinct GPT-6.1 Sol decisions in a separate hash-bound artifact.
- [ ] Validate frozen sources and score the authorized AI evidence separately.
- [ ] Publish the adapted Stage 2 decision and any required expansion disposition.
- [ ] Publish reproduction commands, hashes, coverage, uncertainty, and limitations.
- [ ] Test the AI-only boundary, merge the PR, and record the final disposition.

Results and reproduction commands will be recorded in a separate derived report.
