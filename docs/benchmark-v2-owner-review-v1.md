# Benchmark v2 Staged Owner Review v1

Issue #28 owns a staged Level A single-owner research evaluation of the six
frozen Narrative Benchmark v2 free-generation runs. It is not independent
validation, adjudication, agreement evidence, a representative sample, a power
calculation, publication-grade certification, or full-population review. Strict
independent validation remains pending in issue #38.

## Preserved Base

The original deterministic full selection remains supported and unchanged:

- 84 matched quartet blocks and 336 blind rows
- full packet SHA-256:
  `449261f23970e2228c8e9e039bf226d21c1821a73997943579febc499f0a58a9`
- full private-key SHA-256:
  `4d63657e01d6f7cb5801a1a268a20dc4a86c6a69575137fa461ddf799cf3c278`
- base config: `configs/narrative-v2-owner-review-v1.json`

The staged config
`configs/narrative-v2-owner-review-staged-v1.json` locks those identities and
the staged algorithm. Export verifies identical existing full bytes rather than
overwriting them. A mismatch fails closed. If both full artifacts are absent,
the same command can recreate them deterministically.

All six historical raw, manifest, packet, key, rubric, seal, checkpoint,
tokenizer, scenario, scorer, model-mapping, and 2-model-by-3-split identities
are validated with the unchanged historical validators before staged export or
scoring.

## Stage 1 Selection

Stage 1 uses exactly 20 existing matched quartet blocks and 80 rows. Each block
retains both models and both A/B worlds. Selection uses only structural fields,
is invariant to input order, and never inspects generated text or diagnostics.

A bounded deterministic quota search uses the frozen seed and lexicographic
hash ranks. Its exact block margins are:

| Dimension | Stage 1 block counts |
|---|---|
| Trial | greedy 5; sampled `11337` 5; `21337` 5; `31337` 5 |
| Depth | each of the four depth cells has 5 |
| Split | development 6; test 7; generalization 7 |
| Family | one hash-ranked family has 2; the other six have 3 |

This covers all seven families at least twice. It does not provide complete
family-by-depth-by-split coverage.

Stage 1 supports only:

- descriptive overall pair-consistency rates from 20 pairs per anonymous arm
- an exploratory matched arm difference across the same 20 blocks

Each trial cell has only five blocks per arm. Split, family, depth, reasoning
depth, and interaction cells are descriptive only. No comparison is described
as powered, representative, or confirmatory.

## Generated Artifacts

The write-once Stage 1 artifacts were generated from the real frozen sources on
2026-10-12. No owner decisions were generated.

| Artifact | SHA-256 |
|---|---|
| `stage-1-owner-review-packet.jsonl` | `ce11fad2c07a412c8aa7b3cbc1eccc59d6d5d4c2b982cdea4fb5696397e92a0b` |
| `stage-1-public-manifest.json` | `ac62a19faec0e47c51eb3823d21039f1c03900de188199076d7659b5452f7c15` |
| `stage-1-private-key.json` | `ca7bf26a5339c3bf6bd05d1ca3f5c65180c42a252b360b99c6756c32867ab5b1` |

The public packet and manifest contain no actual or blinded model IDs,
condition assignments, scenario/split/world/mode/seed/family/depth metadata,
source paths, or model mapping. The public manifest reports anonymized cell
sizes, supported comparisons, limitations, packet/rubric/config identities,
and rubric guidance. Exact named cells, source identities, and mapping remain
in the private key, which is unnecessary for owner review.

Export or verify the artifacts with:

```bash
source .venv/bin/activate
python -m src.narrative_v2_owner_review export \
  --config configs/narrative-v2-owner-review-staged-v1.json \
  --checkpoint-selection-complete \
  --test-evaluation-complete \
  --output-dir runs/benchmark-v2/owner-review-v1
```

Reruns verify every existing full and Stage 1 artifact byte-for-byte and write
only absent Stage 1 files. They never replace an artifact.

## Terminal Review

The terminal interface shows only public packet content and frozen rubric
guidance. It saves after every row and resumes by skipping completed rows. A
completed original judgment cannot be edited or replaced through this command.
Every newly saved row uses review-record schema version `2` and records its
packet and manifest hashes, normalized owner ID, owner role, false independence,
genuine-human attestation, label, confidence, explicit `evidence_type`, evidence,
nonempty note, and UTC `reviewed_at` timestamp. Existing schema-v1 progress rows
without `review_record_schema_version` or `evidence_type` remain readable and
resume unchanged; they retain the original quotation/empty-output semantics.

Evidence types distinguish three cases without changing the frozen rubric:

- `verbatim_quote`: required for a specific supporting statement,
  contradiction, or forbidden proposition. `evidence_quote` must occur exactly
  in the generated continuation.
- `required_proposition_absent`: permitted only with an `inconsistent` label on
  nonempty output. `evidence_quote` stays empty and `reviewer_note` gives a
  concise explanation of the omitted required fact; reviewers do not invent a
  quotation for an absence.
- `empty_output`: permitted only when the generated continuation is genuinely
  empty and binds the exact marker `<EMPTY_OUTPUT>`.

The terminal interface asks inconsistent reviews whether the evidence is a
verbatim contradiction/forbidden statement or an absent required proposition.
Consistent and uncertain reviews continue to require verbatim evidence.

## AI-Assisted Preliminary Review

An explicitly non-human GPT-6 Astra preliminary review inspected all 80 public
Stage 1 rows without the private key or model mapping. Its ignored local artifact
is `runs/benchmark-v2/owner-review-v1/stage-1-gpt-6-astra-preliminary-review-v1.md`
with SHA-256
`acedeb13b06257d671d3f23e68e6e21dae7d091455f7c72bb131dbdd8df163c7`.

The artifact reports 0 consistent, 80 inconsistent, and 0 uncertain model
judgments. Its primary diagnosis is omission: none of the continuations states
the required terminal proposition. It identifies rows 13, 21, 55, and 65 as
whitespace-only and the other 76 as nonempty outputs missing the required fact.
Structural validation confirmed that the report covers rows 1 through 80 exactly
once, every listed missing terminal value matches the corresponding packet
requirement, and the four whitespace row numbers match the packet bytes.

This is AI-assisted preliminary evidence, not genuine owner review, independent
validation, an official Stage 1 score, or a Stage 2 decision. It has no
genuine-human attestation and must not be imported into
`stage-1-owner-reviews.jsonl`. Actual owner decisions and final attestation
remain pending.

Preview without writing:

```bash
python -m src.narrative_v2_owner_review review \
  --packet runs/benchmark-v2/owner-review-v1/stage-1-owner-review-packet.jsonl \
  --manifest runs/benchmark-v2/owner-review-v1/stage-1-public-manifest.json \
  --reviews runs/benchmark-v2/owner-review-v1/stage-1-owner-reviews.jsonl \
  --reviewer-id BearCatCoder \
  --dry-run
```

Start or resume genuine review and request final verification after all rows:

```bash
python -m src.narrative_v2_owner_review review \
  --packet runs/benchmark-v2/owner-review-v1/stage-1-owner-review-packet.jsonl \
  --manifest runs/benchmark-v2/owner-review-v1/stage-1-public-manifest.json \
  --reviews runs/benchmark-v2/owner-review-v1/stage-1-owner-reviews.jsonl \
  --reviewer-id BearCatCoder \
  --final-attestation \
  --owner-verification runs/benchmark-v2/owner-review-v1/stage-1-owner-verification.json
```

If interrupted, rerun the same command. Previously saved decisions remain
immutable. The verification artifact is written once only after all 80 rows and
an explicit final attestation.

## Stage 1 Scoring

After genuine review completion:

```bash
python -m src.narrative_v2_owner_review score \
  --config configs/narrative-v2-owner-review-staged-v1.json \
  --checkpoint-selection-complete \
  --test-evaluation-complete \
  --packet runs/benchmark-v2/owner-review-v1/stage-1-owner-review-packet.jsonl \
  --manifest runs/benchmark-v2/owner-review-v1/stage-1-public-manifest.json \
  --private-key runs/benchmark-v2/owner-review-v1/stage-1-private-key.json \
  --reviews runs/benchmark-v2/owner-review-v1/stage-1-owner-reviews.jsonl \
  --owner-verification runs/benchmark-v2/owner-review-v1/stage-1-owner-verification.json \
  --output runs/benchmark-v2/owner-review-v1/stage-1-score-report.json
```

Scoring revalidates all source and staged identities and complete original
judgments. It reports partial single-owner exploratory evidence: anonymous-arm
overall rates, strict and resolved-only descriptive cells with exact sizes, an
exploratory paired arm difference, a deterministic paired bootstrap over the 20
matched blocks, separate automated diagnostics, limitations, and the Stage 2
decision. Agreement/kappa remain unavailable and adjudication remains absent.
The Stage 1 report never reveals the model mapping.

## Stage 2 Rule

Stage 2 is a preregistered decision rule, not a power calculation. It is
triggered when any condition holds:

1. Overall paired arm-difference 95% bootstrap CI width is greater than `0.30`.
2. The CI includes zero and the absolute point difference is at least `0.10`.
3. Either arm's unresolved-uncertain pair rate is greater than `0.10`.

If none holds, planned owner review stops after Stage 1. If any holds, Stage 2
authorizes 20 additional disjoint quartet blocks and 80 rows. The frozen search
balances cumulative margins to 10 blocks per trial, 10 per depth, split counts
`13/13/14`, and family counts `5/5/6/6/6/6/6`. Code and tests freeze this
selection now, but no Stage 2 packet is generated or exposed before completed
Stage 1 scoring triggers authorization. The mapping remains withheld while any
triggered stage is incomplete; the safe Stage 1 behavior is never to reveal it.

## Tests

```bash
source .venv/bin/activate
python -m unittest tests.test_narrative_v2_owner_review
```

Synthetic tests cover exact margins, ordering/text/diagnostic blindness,
disjoint cumulative Stage 2 balance, packet and manifest blinding, resumable
per-row persistence, immutable judgments, UTC timestamps, artifact bindings,
paired bootstrap repeatability, all trigger branches, descriptive cell sizes,
and absence of private mapping from public and Stage 1 score outputs.
