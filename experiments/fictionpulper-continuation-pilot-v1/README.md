# FictionPulper Continuation Pilot v1

```text
status: STOP
approved: false
canonical_examples: 0/200 supplied by this change
approval_mode: single_owner_research_pilot
rights_and_provenance: pending real owner verification
owner_review_packet: pending canonical examples
owner_human_reviews: 0
independent_human_validation: false
publication_grade_certification: false
adjudication: not claimed in owner mode
training: not run and not authorized
```

Issue #8's deterministic prepare, audit, review-export, and finalization
workflow is implemented. It deliberately does not fabricate examples, rights
evidence, owner review, verification, external reviewers, or approvals. The
production command currently stops and requests canonical examples and real
owner review/verification under ignored `data/continuation_pilot_v1/`.
Exact public-domain source bytes, rights snapshots, referenced prompts, and AI or
assessment raw outputs must remain there under approved roots; missing bytes,
unsafe paths, and hash mismatches stop preparation.
Public-domain records must additionally attest independent acquisition from the
documented URL/evidence and no copying from unsealed Corpus-v3 candidates.
Corpus-v3/pretraining source overlap is allowed only when audited and disclosed;
both approval modes bind these claims in final verification and the manifest.
The frozen audit additionally requires 100/100 stale-state-distractor balance,
eight examples in each of 25 genre/state cells, evidence-backed event-transition
metadata, and 32/4/4 examples per genre across train/validation/test. None of
those gates is represented as passed without the real 200 examples.
Prepared token/review files are audit artifacts, not an issue #9 training
interface. Production commands enforce an in-code immutable corpus contract,
allow only the formal `strict_independent` mode substitution, and remove stale
approval on STOP. Owner approval is limited to a deterministic 50-row
representative packet plus every flagged row; its manifest must disclose all
unreviewed rows and cannot claim independence or adjudication. Strict mode keeps
the original two-reviewer and distinct-adjudicator process in separate artifacts.
The exact serialization inserts no boundary bytes: each canonical opening itself
ends in exactly two LF bytes.

Protocol and operating instructions are in
[`docs/continuation-pilot-v1.md`](../../docs/continuation-pilot-v1.md). Generated
reports and packets are ignored under `runs/continuation-pilot-v1/`. This
tracked record is not a corpus seal and is not authorization for issue #9
training.
