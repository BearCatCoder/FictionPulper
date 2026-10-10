# FictionPulper Continuation Pilot v1

```text
status: STOP
approved: false
canonical_examples: 0/200 supplied by this change
rights_clearances: pending external verification
independent_human_reviews: pending
adjudications: pending
training: not run and not authorized
```

Issue #8's deterministic prepare, audit, review-export, and finalization
workflow is implemented. It deliberately does not fabricate examples, rights
evidence, reviewers, or approvals. The production command currently stops and
reports the missing ignored inputs under `data/continuation_pilot_v1/`.
The frozen audit additionally requires 100/100 stale-state-distractor balance,
eight examples in each of 25 genre/state cells, evidence-backed event-transition
metadata, and 32/4/4 examples per genre across train/validation/test. None of
those gates is represented as passed without the real 200 examples.
Prepared token/review files are audit artifacts, not an issue #9 training
interface. Production commands additionally enforce an in-code immutable
contract and remove stale approval on STOP. The exact serialization inserts no
boundary bytes: each canonical opening itself ends in exactly two LF bytes.

Protocol and operating instructions are in
[`docs/continuation-pilot-v1.md`](../../docs/continuation-pilot-v1.md). Generated
reports and packets are ignored under `runs/continuation-pilot-v1/`. This
tracked record is not a corpus seal and is not authorization for issue #9
training.
