# FictionPulper Continuation Pilot v1

```text
status: APPROVED
approved: true
canonical_examples: 200/200
approval_mode: single_owner_research_pilot
rights_and_provenance: verified by identified project owner
owner_review_packet: 50 selected/flagged examples
owner_human_reviews: 50
unreviewed_unflagged_examples: 150
independent_human_validation: false
publication_grade_certification: false
adjudication: not claimed in owner mode
training: not run and not authorized
```

Issue #8's deterministic prepare, audit, review-export, and finalization
workflow is implemented and the production corpus passed it. The ignored
`data/continuation_pilot_v1/` contains 200 canonical examples: 160 AI-assisted
adaptations from 40 independently acquired public-domain works and 40
AI-assisted originals. The identified project owner examined the retained
rights/provenance records and directly reviewed all 50 rows in the deterministic
owner packet. That packet included every one of the 40 automatically flagged
`rights_sensitive` originals plus 10 stratified public-domain adaptations.
Exact public-domain source bytes, rights snapshots, referenced prompts, and AI or
assessment raw outputs must remain there under approved roots; missing bytes,
unsafe paths, and hash mismatches stop preparation.
Public-domain records must additionally attest independent acquisition from the
documented URL/evidence and no copying from unsealed Corpus-v3 candidates.
Corpus-v3/pretraining source overlap is allowed only when audited and disclosed;
both approval modes bind these claims in final verification and the manifest.
The frozen audit passed its 100/100 stale-state-distractor balance,
eight examples in each of 25 genre/state cells, evidence-backed event-transition
metadata, and 32/4/4 examples per genre across train/validation/test. Exact,
normalized five-word, Benchmark v2, Scene Scorecard, held-out-prompt, and source
input-exclusion collision checks reported no violations. Two public-domain
sources were disclosed as identified in pretraining and 38 as not identified.
Prepared token/review files are audit artifacts, not an issue #9 training
interface. Production commands enforce an in-code immutable corpus contract,
allow only the formal `strict_independent` mode substitution, and remove stale
approval on STOP. Owner approval is limited to a deterministic 50-row
representative packet plus every flagged row; its manifest must disclose all
unreviewed rows and cannot claim independence or adjudication. Strict mode keeps
the original two-reviewer and distinct-adjudicator process in separate artifacts.
The exact serialization inserts no boundary bytes: each canonical opening itself
ends in exactly two LF bytes. It produced 95,323 tokenizer-v1 tokens with no
truncation: 76,546 train, 9,280 validation, and 9,497 test.

## Frozen Evidence

```text
examples_sha256: e3e17d0e0e2c8249158b215965ab5f4b15456db7248a198767b35721fd3f176b
prepared_sha256: 35253b224177781631bb224ef39e11b464a2aed4cef96cdafc85a2368da8b881
review_packet_sha256: 7fef84d4f8e1ce8b3720fa105ae038ff31bb973d11993bde7be8a92f773ac488
completed_reviews_sha256: c380a91a884cca0ac47aed3d801a77baace8f0c6c73bffb59b1c32d330332af8
owner_verification_sha256: ebb958db5f3bfeb9a2b4e27f568e481bcef9205eb358fc9df23c586ec029b5d9
tokenizer_sha256: 14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012
```

Reproduction and verification commands:

```bash
source .venv/bin/activate
python -m src.continuation_pilot prepare
python -m src.continuation_pilot audit
python -m src.continuation_pilot export-review
python -m src.continuation_pilot finalize
python -m unittest tests.test_continuation_pilot
python -m unittest discover -s tests
```

Protocol and operating instructions are in
[`docs/continuation-pilot-v1.md`](../../docs/continuation-pilot-v1.md). Generated
reports and packets are ignored under `runs/continuation-pilot-v1/`. This
tracked record reports the approved limited pilot but is not a corpus seal or a
training interface. Issue #9 must freeze a separate hash-bound export and
experiment protocol before training.
