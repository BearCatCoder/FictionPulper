# Benchmark v2 Owner Review v1

Issue #28 owns the Level A single-owner research evaluation of the six frozen
Narrative Benchmark v2 free-generation runs. This is a deterministic sampled
review, not independent validation, a complete-population human review,
agreement evidence, adjudication, or publication-grade certification. Strict
independent review remains pending in GitHub issue #38.

## Frozen Inputs And Sample

The tracked contract is
`configs/narrative-v2-owner-review-v1.json`. It records the exact repo-relative
paths and SHA-256 identities of all six raw generation files, manifests,
historical packets, and historical private keys, plus the frozen rubric hash.
It also locks the 1,792-row population, 336-row sample, selection and packet
seeds, and 10,000-replicate paired-bootstrap settings.

Before sampling, the tool invokes the existing historical protocol and
generation validators unchanged. It verifies raw, manifest, packet, key,
rubric, seal, checkpoint, tokenizer, scenario, scorer, model-mapping, and exact
2-model-by-3-split identities and rejects an incomplete population.

The sample contains 84 matched quartet blocks. For each of the 3 splits, 7
state families, and 4 depth buckets it selects one scenario. Development and
test have one eligible scenario per cell. Generalization has two and uses a
fixed hash rank to choose one. Within each split/family group, a fixed hash
permutation assigns greedy and sampled seeds `11337`, `21337`, and `31337`
exactly once across the four depths. Every block retains both models and both
A/B worlds. Selection reads structural metadata only, never generated text or
automated diagnostics, and is invariant to source row ordering.

## Export

Held-out access attestations are mandatory because export validates test and
generalization sources:

```bash
source .venv/bin/activate
python -m src.narrative_v2_owner_review export \
  --config configs/narrative-v2-owner-review-v1.json \
  --checkpoint-selection-complete \
  --test-evaluation-complete \
  --output-dir runs/benchmark-v2/owner-review-v1
```

The command is write-once. It creates `owner-review-packet.jsonl`, which goes to
the owner, and `owner-review-private-key.json`, which the evaluation custodian
keeps from the owner until review completion.

The packet exposes no actual or blinded model ID, split, scenario, world, mode,
seed, family, depth, reasoning depth, or source path. It keeps the prompt,
generated text, semantic rubric fields, owner role, explicit non-independent
status, and empty decision fields. The packet cannot contain its own hash
without self-reference, so completed rows and separate owner verification bind
that hash.

The packet generated on 2026-10-12 has 336 rows and SHA-256
`449261f23970e2228c8e9e039bf226d21c1821a73997943579febc499f0a58a9`.
Generation does not constitute review. No owner decisions or score exist yet.

## Owner Completion

Complete exactly one copy of every packet row without consulting the private
key. Preserve all fields and add `packet_sha256` with the packet hash. Use the
same normalized, nonblank `reviewer_id` on every row. Required values are:

- `reviewer_role`: `owner`
- `independent_review_confirmed`: `false`
- `genuine_human_attestation`: `true`
- `label`: `consistent`, `inconsistent`, or `uncertain`
- `confidence`: integer 1 through 5
- `evidence_quote`: a verbatim substring of `generated_text`, or exactly
  `<EMPTY_OUTPUT>` when generated text is empty
- `reviewer_note`: nonempty owner rationale

Do not add adjudication fields or duplicate rows. Store completed rows at
`runs/benchmark-v2/owner-review-v1/completed-owner-reviews.jsonl`.

Create `runs/benchmark-v2/owner-review-v1/owner-verification.json` with exactly
this shape, replacing placeholders with the genuine identity/date and all 336
packet IDs:

```json
{
  "version": "narrative-v2-owner-review-v1",
  "review_packet_sha256": "449261f23970e2228c8e9e039bf226d21c1821a73997943579febc499f0a58a9",
  "owner_id": "<same reviewer_id used in every review row>",
  "reviewed_at": "<date>",
  "selected_owner_review_ids": ["<all packet owner_review_id values>"],
  "genuine_human_attestation": true,
  "single_owner_evidence_acknowledged": true,
  "non_independent_evidence_acknowledged": true,
  "sample_only_evidence_acknowledged": true,
  "no_agreement_available_acknowledged": true,
  "no_adjudication_performed_acknowledged": true
}
```

The software validates these claims but cannot independently prove identity.

## Score

Only after genuine owner completion, run:

```bash
source .venv/bin/activate
python -m src.narrative_v2_owner_review score \
  --config configs/narrative-v2-owner-review-v1.json \
  --checkpoint-selection-complete \
  --test-evaluation-complete \
  --packet runs/benchmark-v2/owner-review-v1/owner-review-packet.jsonl \
  --private-key runs/benchmark-v2/owner-review-v1/owner-review-private-key.json \
  --reviews runs/benchmark-v2/owner-review-v1/completed-owner-reviews.jsonl \
  --owner-verification runs/benchmark-v2/owner-review-v1/owner-verification.json \
  --output runs/benchmark-v2/owner-review-v1/owner-score-report.json
```

Scoring revalidates every historical input and reconstructs the sample and blind
packet. It then validates one genuine-human owner row per packet row, immutable
content, packet binding, one owner identity, evidence, decisions, and separate
verification. It preserves A/B pairing and reports strict and resolved-only
rates, deterministic paired bootstrap intervals, model/trial/split/family/depth/
reasoning-depth/family-depth strata, and sampled-seed summaries. Model mapping
is revealed only in the completed report.

Agreement and Cohen's kappa are `null`/not available, and adjudication is
`not_performed`; the tool never manufactures reviewers. Fact, premise,
contradiction, entity, causal, stale-state, and repetition summaries are
reported separately as automated diagnostics only.

## Tests

```bash
source .venv/bin/activate
python -m unittest tests.test_narrative_v2_owner_review
```

Synthetic fixtures cover deterministic balance and blinding, ordering
invariance, validation failures, packet and attestation binding,
no-adjudication enforcement, score strata and uncertainty, limitations, null
agreement, and required held-out CLI flags.
