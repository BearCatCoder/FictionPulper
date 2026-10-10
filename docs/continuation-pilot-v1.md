# Continuation Pilot v1

Issue #8 freezes a workflow for exactly 200 rights-cleared natural narrative
continuations. It does not provide, generate, or approve those examples. The
current status is **STOP: canonical prose, rights verification, and genuine
independent human review are pending external inputs**.

## Frozen Contract

- Exactly 200 examples: 160 train, 20 validation, and 20 post-selection test.
- Exactly 40 examples in each genre and 40 in each primary state family defined
  in `configs/continuation-pilot-v1.yaml`.
- Exactly eight examples occupy every genre x primary-state-family cell. This
  cross-balance prevents genre from becoming a one-to-one shortcut for the
  state-update task.
- Exactly 100 examples contain a stale-state distractor and 100 do not. The
  required `stale_state_distractor_present` boolean is a frozen lexical/task-
  shortcut control, not prose injected into the model input.
- Openings contain 60-120 normalized words; continuations contain 150-250.
- Premise, protagonist, motive, three to five atomic opening facts, primary
  state family, and action/consequence summaries remain metadata. They are not
  injected into prose or serialization.
- Every example has one or more structured event transitions. Each transition
  strictly records its frozen state family, before state, changing event, after
  state, and opening/continuation evidence quotes. At least one transition must
  audit the example's primary state family. Transition records and stale-state
  flags remain metadata only.
- Source IDs, original-story IDs, and duplicate clusters are transitive split
  constraints. Every transitive group must contain exactly one genre; a mixed-
  genre group stops preparation. Groups are assigned independently within each
  genre and must reach exactly 32 train, 4 validation, and 4 test examples per
  genre, as well as the overall 160/20/20 counts.
- Every example carries rights and provenance references and hashes. Their
  presence is only structural evidence, not proof of permission.
- Normalized exact and normalized five-word-shingle checks cover pilot prose and
  metadata: premise, protagonist motive, atomic facts, action/consequence
  summaries, and transition before/event/after summaries. The same checks cover
  all authored Benchmark v2 files and all ten Scene Scorecard prompts.
  Scorecard names and declared post-selection authoring inputs are independently
  rejected. These lexical checks cannot establish that a paraphrased premise or
  distinctive fact combination is semantically novel; the custodian must perform
  and attest to that human check.
- Tokenizer-v1 is fixed at SHA-256
  `14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012`.
- Serialization is exactly `<|story|><|bos|>` + opening + continuation +
  `<|eos|>`, with no inserted separator bytes. Canonical opening text must end in
  exactly two LF bytes and continuation text must begin with a non-whitespace
  byte. Thus the two LFs are part of the opening, not an inserted separator.
  This convention resolves the previously unspecified byte boundary and locks
  the exact bytes and token target boundary for issue #9. Separate and combined
  tokenization must agree or preparation fails. No genre, transition,
  stale-state, or other metadata is serialized.
- The opening and padding labels are `-100`. The first continuation token
  through EOS are targets. Every record must fit at most 1,024 tokens without
  truncation.

The JSON schema is `schemas/continuation-pilot-v1.schema.json`. Unknown fields
are rejected at every defined object level. Canonical prose and private rights
evidence belong under ignored `data/continuation_pilot_v1/`; derived packets and
reports belong under ignored `runs/continuation-pilot-v1/`.

## Commands

Run each command from the repository root:

```bash
source .venv/bin/activate
python -m src.continuation_pilot prepare
python -m src.continuation_pilot audit
python -m src.continuation_pilot export-review
python -m src.continuation_pilot finalize
```

`prepare` and `audit` produce a deterministic STOP report when canonical input
is absent. Once input exists, they validate strict schema, counts, grouping,
the 25-cell genre/state shortcut control, 100/100 stale-distractor balance,
per-genre split representation, contamination, tokenizer identity,
serialization, masking, EOS, and no truncation. `export-review` runs only after
every automated gate passes and binds its packet to prepared-content hashes.
The audit and final approval reports include exact tokenizer-v1 token totals
overall and by train, validation, and test split.
The CLI validates all critical settings against an in-code production contract:
counts, distributions, seeds, word ranges, schema identity, tokenizer identity,
serialization, and every contamination reference path/hash. Only private input,
review, custodian, and generated-output paths may be relocated. Scaled fixture
configs are helper-only and can never write `approval.json`.

Two distinct original reviewers must independently review every example for
premise and motive preservation, opening-fact consistency, no loop, plot
advancement, genre voice, meaningful action and consequence, and natural
readability. Any disagreement or uncertainty requires one distinct independent
adjudicator. Edited prose changes content and packet hashes, invalidating prior
reviews.

`finalize` also requires `custodian-verification.json`. The custodian must
explicitly record that underlying rights/provenance evidence was examined, all
rights were cleared, reviewer identities and genuine-human status were checked,
review independence was checked, adjudicators were verified, and reserved
post-selection prompts were excluded from authoring, editing, and candidate
selection. The custodian must also attest that distinctive scorecard fact
combinations were checked and that none collide. IDs, lexical scans, and self-
attestations do not establish those real-world facts; software only verifies the
custodian's hash-bound record. An adjudicator resolves only disputed or uncertain
judgments; original-reviewer consensus, including a consensus failure, is
preserved. Finalization writes `approval.json` only when every automated, human-
review, adjudication, and custodian gate passes. Prepare/audit STOP and every
anticipated finalization failure remove stale approval.

## Artifact Access

`prepared.jsonl`, its token arrays, and review artifacts are audit evidence only.
They are not an issue #9 SFT dataset API and do not authorize training. The pure
`load_prepared_records` inspection helper defaults to train plus validation and
refuses test unless `post_selection_complete=True` is explicitly supplied.
Issue #9 must define and freeze its own hash-bound training export or loader;
until then, no workflow command exposes these artifacts as training data.

## Current Result

No examples, rights approvals, reviews, or adjudications were created during
workflow implementation. The tracked report in
`experiments/fictionpulper-continuation-pilot-v1/README.md` records the pending
state. No model training is implemented or authorized by this workflow.
