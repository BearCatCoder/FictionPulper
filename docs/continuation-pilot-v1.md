# Continuation Pilot v1

Issue #8 freezes a workflow for exactly 200 rights-documented natural narrative
continuations. The active approval mode is `single_owner_research_pilot`. This is
an exploratory single-owner artifact, not independent validation, adjudication,
publication-grade certification, or evidence of general corpus fitness. The
optional `strict_independent` mode preserves the two-reviewer and adjudicator
workflow as a separate claim. The owner-mode corpus and review are complete;
strict-mode independent review has not occurred.

## Frozen Corpus Contract

- Exactly 200 examples: 160 train, 20 validation, and 20 post-selection test.
- Exactly 40 examples per frozen genre and primary state family, eight in each
  genre/state cell, and 100 examples with versus 100 without a stale-state
  distractor. Each genre contributes exactly 32/4/4 train/validation/test rows.
- Openings contain 60-120 normalized words and continuations contain 150-250.
- Premise, protagonist and motive, three to five atomic opening facts, event
  transitions, primary state family, stale-state status, meaningful action, and
  consequence are audit metadata and are never serialized.
- Source IDs, original-story IDs, and duplicate clusters form transitive split
  groups. Groups cannot cross splits or genres.
- Normalized exact and five-word-shingle checks cover prose, semantic audit
  metadata, authoring inputs, AI authoring prompts and configurations, retained
  AI raw outputs, preliminary-assessment prompts/configurations/raw outputs, and
  other candidate-selection metadata against every authored Benchmark v2
  surface and all ten Scene Scorecard prompts. Referenced prompt bytes are
  resolved and scanned. Scorecard-name collisions and any declared
  `scene-post-*` authoring input fail. A risk flag never converts an exact,
  shingle, held-out prompt, schema, grouping, serialization, or other hard-gate
  failure into a pass.
- Retained public-domain source works have a separate input-exclusion gate. It
  rejects a normalized exact reference, containment of any complete Benchmark
  v2 or Scene Scorecard prompt/premise/fact, and any shared normalized 12-word
  shingle. The ordinary five-word rule is not applied across whole source books
  because common literary phrases would make that gate unusably noisy. Complete
  reserved-prompt reuse is always a hard failure.
- Tokenizer-v1 remains fixed at SHA-256
  `14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012`.
- Serialization is `<|story|><|bos|>` + opening + continuation + `<|eos|>`.
  The canonical opening supplies exactly two trailing LF bytes; no separator is
  inserted. Separate and combined tokenization must agree. Opening and padding
  labels are `-100`; continuation through EOS are targets. The maximum is 1,024
  tokens and truncation is forbidden.

The production config and in-code contract freeze all counts, vocabulary,
seeds, hashes, contamination surfaces, serialization, and owner sample size.
Only ignored input/review/verification/output paths can move. A CLI approval-mode
override can select `strict_independent`; it cannot substitute any other
production setting. Helper configs cannot finalize or write approval.

## Provenance Contract

The strict schema is `schemas/continuation-pilot-v1.schema.json`, SHA-256
`2057ea5c6f540ac68a54acd629551f9d3a44066bae81013be1bfac5e1dbb0634`.
Unknown fields are rejected. Manual checks additionally enforce source-class
conditions and hash linkage.

Public-domain adaptations record author, title, HTTPS source URL, edition, exact
source path/hash, rights basis, retained rights-evidence path/reference/hash,
rights check date, source ID, original-story ID, acquisition method, and retained
hash-verified acquisition evidence,
and explicit declarations that bytes were independently downloaded from the
documented source and were not copied from an unsealed Corpus-v3 candidate.
Missing or false acquisition declarations fail. Authoring inputs that reference
Corpus-v3 candidate IDs or paths also fail. AI-assisted originals and
adaptations record the exact model identity, complete prompt and hash or a
hash-bound retained prompt file, complete generation configuration, generation
date, retained raw-output path/hash, editing history, and a final-content hash
equal to the canonical opening plus continuation bytes. Preliminary assessments
also retain and hash their exact raw outputs. Every source records a pretraining
exposure classification (`not_identified`, `identified`, or `uncertain`), audit
method, and disclosure. Adaptations from one original story stay grouped.
Independent acquisition is distinct from exposure: a lawfully downloaded work
is not prohibited merely because the same source appears in pretraining or
Corpus-v3. Identified or uncertain Corpus-v3 overlap requires matching
`identified`/`uncertain` pretraining exposure plus explicit overlap disclosure.

Record paths are relative to one of the approved roots in
`retained_artifact_roots`; absolute paths, traversal components, symlink escapes,
missing files, and byte-hash mismatches fail. The production roots are under
ignored `data/continuation_pilot_v1/sources/`, `rights_evidence/`, `prompts/`,
and `raw_outputs/`. Downloaded public-domain source bytes, evidence snapshots,
prompt files, and raw model/assessment outputs stay ignored and must never be
committed. Tracked records therefore contain no machine-local path.

Every example also carries a preliminary `automated` or `ai_assisted`
assessment with assessor identity, prompt provenance, configuration, date,
raw-output hash, summary, and only these risk flags:

`ambiguous`, `low_quality`, `possible_contamination`, `possible_duplicate`,
`rights_sensitive`, `provenance_incomplete`, `semantic_scorecard_collision`,
`heldout_prompt_risk`, and `other_problematic`.

The retained assessment output is canonical JSON with no unknown or missing
fields. It binds the example ID and exact opening/continuation content hash and
must exactly repeat method, assessor identity, conditional model identity,
prompt hash, generation configuration, date, risk flags, and summary. Its bytes
must be canonical and match the declared output hash. Metadata/output mismatch
fails before packet selection, ensuring every output flag reaches owner review.
These assessments are diagnostics only. They are never labeled human review.

## Owner Mode

`single_owner_research_pilot` deterministically exports a target sample of 50.
The `deterministic_stratified_all_flags_v1` selection covers every genre,
primary state family, split, represented source class, represented pretraining
exposure class, clear/flagged risk stratum, and shortest/longest corpus length.
It then hash-ranks fill rows. Every flagged row is included even when this makes
the packet exceed 50. Selection reasons are recorded per row; an unavailable
required frozen stratum stops export.

One identified genuine human owner reviews every packet row and every atomic
fact. Required yes/no/uncertain decisions cover rights/provenance, premise,
motive, opening facts, contradictions, loops, plot advancement, genre voice,
meaningful action and consequence, readability, and exploratory-SFT suitability.
Any `no`, `uncertain`, contradicted fact, missing row, changed packet hash, or
changed content blocks finalization.

`<PROVENANCE_RECORD>` is valid evidence only for the rights/provenance decision.
That decision must use exactly this sentinel. All prose and atomic-fact
dimensions require a verbatim opening/continuation quote containing at least
three normalized words; `<ABSENT>` is allowed only with a non-positive or
non-retained judgment. These structural checks remove trivial evidence bypasses,
but software cannot prove that a quoted passage is semantically relevant. The
review remains a genuine-human attestation.

Final `owner-verification.json` binds the prepared and packet hashes and records
source-rights/provenance examination, example identity and final-content linkage,
the semantic scorecard collision check, held-out prompt exclusion, no use of
unsealed Corpus-v3 candidate bytes/artifacts for authoring, editing, or selection,
and completed Corpus-v3/pretraining overlap audit/disclosure. It also records all
reviewed IDs, the unreviewed count, genuine-human status, and explicit acknowledgement
that unreviewed unflagged examples have diagnostics but no direct human quality
judgment. `approval.json` reports `single_owner_research_pilot`, false independent
validation, false publication certification, false adjudication, reviewed IDs
and counts, unreviewed count, limitations, all artifact hashes, token totals,
`unsealed_corpus_v3_candidate_bytes_or_artifacts_used: false`, and the completed
overlap-audit/disclosure claim.
Before accepting reviews, finalization freshly reconstructs the complete packet
and key from prepared records, config, and approval mode. It compares packet
content, order, sample/flag selection reasons, strict all-example coverage, key
mapping, and contract metadata exactly; coordinated packet/key tampering fails.

## Strict Mode

`strict_independent` exports all 200 examples to a separate write-once packet.
Exactly two distinct genuine human reviewers independently review each row.
Every disagreement or uncertainty requires one distinct adjudicator; consensus
failures remain failures. Strict verification binds reviewer/adjudicator identity,
independence, rights/provenance, scorecard collision, held-out prompt checks,
no unsealed Corpus-v3 candidate-artifact use, and exposure-overlap disclosure.
Its manifest claims independent human validation but still does not claim
publication-grade certification. Owner-mode reviews cannot satisfy strict mode,
and strict claims never appear in an owner manifest.

## Commands

Run from the repository root:

```bash
source .venv/bin/activate
python -m src.continuation_pilot prepare
python -m src.continuation_pilot audit
python -m src.continuation_pilot export-review
python -m src.continuation_pilot finalize

python -m src.continuation_pilot export-review --approval-mode strict_independent
python -m src.continuation_pilot finalize --approval-mode strict_independent
```

Owner artifacts use `review-packet.jsonl` and `review-key.json`; strict artifacts
use `review-packet-strict-independent.jsonl` and
`review-key-strict-independent.json`. Review exports are write-once: identical
reruns are accepted, but changed content requires a new output location rather
than overwriting evidence. All review rows bind the exact packet hash.

Canonical prose, retained source/evidence/prompt/raw-output bytes, rights records,
completed reviews, and verification belong under ignored
`data/continuation_pilot_v1/`. Derived audit artifacts belong under ignored
`runs/continuation-pilot-v1/`.

## Completed Pilot

The production corpus contains 200 examples and passed every automated gate.
Its canonical examples hash is
`e3e17d0e0e2c8249158b215965ab5f4b15456db7248a198767b35721fd3f176b`;
the prepared artifact hash is
`35253b224177781631bb224ef39e11b464a2aed4cef96cdafc85a2368da8b881`.
Serialization produced 95,323 tokenizer-v1 tokens: 76,546 train, 9,280
validation, and 9,497 test, with no truncation.

The deterministic packet hash is
`7fef84d4f8e1ce8b3720fa105ae038ff31bb973d11993bde7be8a92f773ac488`.
The identified owner reviewed and approved all 50 packet rows, including every
one of the 40 `rights_sensitive` AI-assisted originals, and completed the final
rights/provenance verification. Finalization returned `APPROVED` in
`single_owner_research_pilot` mode. The remaining 150 unflagged examples have
automated/AI-assisted diagnostics but no direct human quality judgment. No
independent validation, adjudication, publication-grade certification, or
general corpus fitness is claimed.

Prepared JSONL, token arrays, diagnostics, and packets remain audit-only.
`load_prepared_records` hides test by default and requires explicit
post-selection completion to inspect it. Issue #9 must define a separate
hash-bound training export and frozen protocol before SFT; issue #8 performed no
training.
