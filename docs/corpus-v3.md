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
.venv/bin/python -m unittest tests.test_corpus_v3 tests.test_corpus_v2 \
  tests.test_audit_corpus_v2 tests.test_dataset
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
`source_sha256` pins the exact downloadable
story bytes. Rights evidence contains `basis`, `evidence_url`, and `checked_at`;
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
  dialogue strata. Its IDs must exactly equal `approved_sample_ids`; pending or
  stale review blocks freezing.
- Stage reports are written under `data/corpus_v3/stages/`. Final reports include
  `stats.json`, `audits.json`, `manifest.json`, split assignments, rejections,
  and duplicate evidence. `stats.json` records `exact_actual_token_count` and
  whether the feasibility target was reached.
- `seal.json` is written only when every required gate passes. A compliant but
  exhausted under-target supply may be sealed with
  `stop_reason: compliant_supply_exhausted`; failed gates produce reports but no
  seal. An existing seal is never overwritten.

## Current Limitation

No new-source Corpus-v3 acquisition was run for this implementation, and no
100M result is claimed. A local feasibility pass using only the hash-locked
Corpus-v2 input completed twice with the same canonical SHA-256
`3c98f3f67afe395225949868d13044e3a226e87607edfd132a72344a288b88d0`.
It found 1,947 individually eligible stories and exactly 14,923,683
tokenizer-v1 document tokens. It stopped with
`candidate_supply_exhausted_with_audit_failures` and did not write a seal:
unclassified material was 57.04% of tokens versus the 35% cap, and the
deterministic 32-story review remains pending. Rights/source-hash, exact and
near-duplicate, metadata provenance, boundary, quality, author/collection
concentration, token accounting, and collection/duplicate split-leakage gates
passed; unresolved near duplicate clusters and cross-split leaks were both zero.

The repository does not yet contain a reviewed, hash-pinned external inventory
large enough to establish attainable v3 supply. The next data task is source
review and inventory construction, not relaxing the freeze gates.
