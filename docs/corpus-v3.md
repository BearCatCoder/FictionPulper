# Corpus-v3 Acquisition And Audit

Corpus-v3 is a new artifact. It never rewrites the sealed Corpus-v2 corpus,
manifests, packed data, checkpoints, or experiment reports. The approximately
100 million tokenizer-v1 document-token objective is a feasibility goal, not an
admission quota: the builder reports the exact attainable compliant size and
stops rather than weakening rights, quality, diversity, or duplicate gates.

## Reproduce

From the repository root:

```bash
.venv/bin/python -m src.corpus_v3 --config configs/corpus-v3.yaml
.venv/bin/python -m src.report_corpus_v3_blockers \
  --config configs/corpus-v3-blockers-v1.yaml
.venv/bin/python -m unittest tests.test_corpus_v3 tests.test_corpus_v2 \
  tests.test_audit_corpus_v2 tests.test_dataset \
  tests.test_report_corpus_v3_blockers
```

The committed config has no external inventories. It is intentionally a safe
starting point, not a claim that a 100M build was run. It hash-locks the sealed
Corpus-v2 canonical file and tokenizer-v1, then admits only rows meeting the v3
evidence bar. To expand supply, add reviewed JSONL inventories to
`source_inventories`, record each inventory SHA-256 in the config, and update the
review-file SHA-256 after completing duplicate and sample decisions.

Each inventory row describes one already bounded story. Required fields are
`id`, `source`, `source_url`, `source_sha256`, `title`, `collection_id`,
`creator_id` or `primary_author`, `rights_status`, `rights_evidence`,
`boundary_evidence`, and `genres`. Creator, genre, and known-publication-year
claims must carry explicit evidence; an unknown publication year remains
explicitly `unknown` and is never inferred from an ebook issue date.
Controlled genres must be supported by a configured source term or by an
attributable, dated `reviewed_mappings` decision; a label alone is not evidence.
Evidence dates are checked against the config's frozen `evidence_cutoff_date`,
not the wall clock, so identical inputs retain identical audit outcomes.
`source_sha256` pins the exact downloadable
story bytes. Rights evidence contains `basis`, `evidence_url`, `checked_at`, and
`checked_by`;
boundary evidence contains a method and `confidence: high`. Whole books or
collections without trustworthy story boundaries are not silently split.

## Build Behavior

- Acquisition is resumed through `data/corpus_v3/state/acquisition-ledger.json`.
  Cached bytes, source URL, and SHA-256 must all match before reuse. Changed or
  unpinned bytes are never admitted. Any configured source acquisition failure
  blocks sealing; an incomplete run is not reported as exhausted supply.
- Exact normalized duplicates are removed before selection. Fuzzy candidates
  are deterministic transitive clusters. More than one accepted member in any
  cluster is unresolved and blocks a seal until `configs/corpus-v3-review.yaml`
  records reviewed exclusions.
- Selection interleaves normalized primary creators and favors dialogue-rich,
  compact scenes. Author, collection, genre, quality, source-hash, rights, and
  high-confidence-boundary limits remain hard freeze gates.
- Inherited Gutenberg creator strings are parsed conservatively. The first
  creator is a primary author only when its role is absent or explicitly
  `Author`; its source-scoped normalized key is recorded as an alias candidate,
  not represented as resolved identity.
- Source collections and fuzzy duplicate clusters are indivisible split groups.
  The generated split audit must report no collection or duplicate leakage.
- A deterministic sample covers every observed genre and source plus high/low
  dialogue strata. Its complete canonical records are SHA-256 locked, and every
  approval must record a reviewer, review date, decision, and notes. Missing,
  stale, duplicate, unexpected, unattributed, or content-mismatched decisions
  block freezing.
- Stage reports are written under `data/corpus_v3/stages/`. Final reports include
  `stats.json`, `audits.json`, `manifest.json`, split assignments, rejections,
  and duplicate evidence. `stats.json` records `exact_actual_token_count` and
  whether the feasibility target was reached.
- `seal.json` is written only when every required gate passes. A compliant but
  exhausted under-target supply may be sealed with
  `stop_reason: compliant_supply_exhausted`; failed gates produce reports but no
  seal. An existing seal is never overwritten.

## Current Disposition

No new-source Corpus-v3 acquisition was run for this implementation, and no
100M result is claimed. The historical issue #20 feasibility pass retained
1,947 stories and 14,923,683 tokens, with 57.04% unclassified and 32 provisional
review records. Issue #22 demonstrated that its rights gate synthesized
`inherited_source_assertion` as a review date and that its token and split audit
checks were not independent. That 14,923,683-token result remains historical
shortfall evidence, not a currently eligible corpus.

The corrected base-only rerun considered the same 4,362 candidates and admitted
none. Exactly 1,841 retain the first failure `invalid_schema`; the other 2,521
now fail `incomplete_rights_evidence`. No review approvals or genre labels were
invented, no `seal.json` was written, and gates requiring a nonempty admitted
corpus remain failed or unevaluable. The machine-readable disposition is
[`fictionpulper-corpus-v3-blockers-v1`](../experiments/fictionpulper-corpus-v3-blockers-v1/README.md).

The repository does not yet contain a reviewed, hash-pinned external inventory
large enough to establish attainable v3 supply. Issue #21 owns evidence-backed
source expansion, final membership, and the genuine review inputs needed before
issue #24 can seal or pack anything. Thresholds and safeguards remain unchanged.
