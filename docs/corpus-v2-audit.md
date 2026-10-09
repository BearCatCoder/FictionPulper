# Corpus-v2 Diversity And Boundary Audit

Issue #5 audits the immutable 30M-token Corpus-v2 acquisition artifact and the
later Data30M packing. It does not clean, rewrite, repack, or otherwise modify
either input layer.

## Reproduce

From the repository root with the existing environment:

```bash
.venv/bin/python -m src.audit_corpus_v2 \
  --config configs/corpus-v2-audit.yaml
.venv/bin/python -m unittest tests.test_audit_corpus_v2 tests.test_dataset \
  tests.test_corpus_v2 tests.test_prepare_data30m
```

The command fails before analysis if any configured input hash differs. It
checks all hashes again after writing the separate audit outputs. The
acquisition seal identifies `data/corpus_v2/corpus.jsonl`; it explicitly does
not include the split or packed dataset. The audit config therefore locks the
post-seal split, packed metadata, tokenizer, and each bin/index hash directly.
All tracked paths are repository-relative.

Outputs are under `experiments/fictionpulper-corpus-v2-audit-v1/`:

- `audit-report.json`: definitions, quantified findings, ranked issues, and
  Corpus-v3 rules.
- `representative-samples.json`: deterministic bounded excerpts and metrics.
- `sample-review.json`: the completed engineering review of all 24 samples.
- `manifest.json`: config, implementation, input/output hashes, analyzed base
  commit, dirty-state caveat, and Python/dependency versions.

The recorded run used base commit
`19a4a772ca2ac8b9ea79032077a074d7ce2bddcb`, Python 3.14.4, NumPy 2.5.3,
Tokenizers 0.23.2, and PyYAML 6.0.3. The worktree was intentionally dirty with
the uncommitted issue implementation. Generated artifacts are therefore
grounded by their file hashes and analyzed base commit, not by claiming a later
commit contains them. The focused suite passed 46 tests on 2026-10-09.
The full repository suite passed 305 tests in 100.560 seconds on the same run.

## Actual Results

The completed audit read 4,362 stories and 30,004,706 packed tokens. The
canonical corpus SHA-256 remained
`0fb5e5080b3c4ee26ea178201d349d6947052f664f2bc415b5a1b9219d496087`.

### Diversity

- The `author` field is actually a Project Gutenberg-style creator string. It
  has 247 raw and 231 mechanically normalized whole-string forms; these are not
  individual identities. There are 589 records with additional semicolon-
  delimited contributors. Recorded role occurrences include 323 illustrators,
  233 editors, 238 contributors, 136 translators, and 54 compilers.
- A defensible primary-author proxy uses only the first creator when it has no
  bracketed non-author role. It covers 4,087 stories and 28,353,003 tokens;
  275 stories remain unknown/non-author-primary. Its 224 raw forms reduce to
  206 normalized alias-candidate keys, with 18 multi-form candidate groups.
  Within covered primary-author tokens, the largest key contributes 5.15%; the
  top 10 and top 20 contribute 27.05% and 38.47%. These normalized keys are not
  verified person identities.
- Source-scoped stable collection keys produce 469 collections. The largest
  contributes 1.48% of tokens; the top 10 and top 20 contribute 7.60% and
  12.94%. Gutenberg ebook IDs are stable keys; the seed source lacks collection
  IDs, so normalized source-scoped collection titles are an explicit fallback.
- Genre is multilabel and both story- and token-weighted. Unclassified material
  is 2,337 stories and 56.05% of tokens. Fantasy is the largest known label at
  1,162 stories and 19.91% of tokens; science fiction is only 97 stories and
  2.15% of tokens.
- No record has a reliably known original publication year. All 4,362 are
  reported unknown. The 2,521 Gutenberg `catalog_ebook_issued` values remain
  separately counted and are never treated as publication dates.

### Form And Quality

- Story length is 334-19,991 words, with median 2,979, p95 10,524, and eight
  inherited stories below the Corpus-v2 500-word admission floor. Packed length
  is 439-39,137 tokens with median 5,100.5; 4,283 stories require continuation
  chunks at 1,024 tokens.
- Dialogue prevalence uses three independent style proxies. Across the corpus,
  51.48% of paragraphs open with a quote mark, there are 36.07 quote marks per
  1,000 words, and 31.88% of words occur in fully closed quote spans. The span
  parser handles conventional paragraph-reopening and nested curly quotes,
  excludes 53 stories' unclosed outer spans, and records 19,961 paragraph
  reopens. Pairing remains typographic rather than semantic speaker attribution.
- Conservative accepted-text signals occur in 642 stories (14.72%). These
  include 109 stories with Gutenberg boilerplate, 38 with transcriber/editor
  notes, 130 with line-hyphen signals, 59 with markup glyphs, and zero with the
  Unicode replacement character. Signals are review leads, not all confirmed
  OCR errors.
- Every row asserts `public_domain` or `public_domain_us`, but only the 2,521
  new Gutenberg rows (57.79%) carry complete row-level rights basis, raw-source
  hash, and source URL evidence. The inherited seed assertions remain usable
  for the sealed experiment but are insufficient as a Corpus-v3 evidence bar.
- The accepted corpus has zero exact text-hash and zero normalized alphanumeric
  text duplicate groups. It still has 23 repeated normalized title/author
  groups and nine repeated 40-word opening groups containing 18 records. Those
  are candidate patterns, not proof of duplication; this audit does not replace
  the sealed fuzzy review of 155 clusters and 162 exclusions.

### 1,024-Token Cuts

The audit reads the existing bins/indexes directly and checks metadata paths and
hash linkage, extents, split assignments, controls, sequence counts, and totals.
All 4,362 indexed documents exactly match canonical re-encoding with
`encode_document`; payload or index misattribution fails closed. It examines all
27,133 internal loader cuts at offsets `1024*n`; it does not repack.

Primary boundary classification:

| Boundary | Count | Share |
|---|---:|---:|
| UTF-8 codepoint | 6 | 0.02% |
| Paragraph | 395 | 1.46% |
| Line | 1,446 | 5.33% |
| Sentence | 1,040 | 3.83% |
| Dialogue | 7,822 | 28.83% |
| Subword | 4,266 | 15.72% |
| Other token | 12,158 | 44.81% |

No partial token side is independently decoded. Exact reversible ByteLevel
vocabulary bytes show 14 cuts adjacent to individually invalid UTF-8 token
fragments; six actually split a UTF-8 byte sequence/codepoint. Canonical text
and tokenizer offsets provide high-confidence paragraph/line/sentence cuts.
Subword and dialogue-at-boundary remain medium-confidence heuristics. Every
internal continuation starts without a repeated story/BOS/genre control token,
matching the current loader behavior.

### Representative Review

The fixed seed selects 24 stories across every observed genre, both sources,
both boundary methods, shortest/longest deciles, dialogue-opening extremes, and
OCR signal versus clean strata, then fills deterministically. All 18 mandatory
strata are explicitly recorded as covered with none uncovered; an insufficient
sample count fails. One changed dialogue-stratum sample (`"JINNY"`) was reviewed
from its bounded opening/ending and passed all three review dimensions. Review
of all bounded openings, endings, and metrics found:

- Story boundary: 17 pass, 4 concern, 3 fail.
- OCR/editorial quality: 23 pass, 1 concern.
- Prose/admission quality: 22 pass, 2 concern.

The failures include one retained Project Gutenberg footer and accepted text
ending with the next headings `CHAPTER II` or `CHAPTER XXVII`. Four concerns
end with probable next-story Roman-numeral headings. Two samples appear
biographical/historical rather than clearly short fiction. Review is a focused
engineering audit, not blinded independent literary annotation; questionable
records must be checked against full source context before Corpus-v3 admission.

## Decision

Do not mutate or retroactively reseal Corpus-v2. It remains valid historical
experiment input. Corpus-v3 should be a new artifact and should not admit data
until the following issues are addressed in order:

1. **P0:** Add evidence-backed original publication period metadata or preserve
   explicit unknown; never infer from ebook issue dates.
2. **P0:** Reduce the 56.05% unclassified token share with provenance-backed,
   multilabel genre metadata and explicit token-share caps.
3. **P1:** Replace arbitrary packing cuts with paragraph/sentence-aware cuts or
   repeat explicit continuation/document controls.
4. **P1:** Require complete row-level rights evidence, especially for inherited
   seed records.
5. **P1:** Reject boilerplate and replacement characters; quarantine other OCR
   and editorial signals for source-context review.
6. **P2:** Review title/author, opening, and fuzzy duplicate candidates before
   collection-aware splitting.

The full machine-readable admission rules additionally require stable creator
and collection IDs, explicit creator roles and primary-author provenance,
story- and token-weighted concentration caps, deterministic stratified review,
high-confidence story boundaries, and post-clustering split leakage checks.
