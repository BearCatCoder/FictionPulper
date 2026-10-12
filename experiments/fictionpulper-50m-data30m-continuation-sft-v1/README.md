# Continuation-SFT v1 Preregistration

Status: implementation complete; export, preflight, training, selection, held-out
evaluation, generation, and human review have not been run.

The first post-selection invocation stopped before scene or Benchmark v2
generation because the protocol's Data30M test-binary hash had one extra trailing
hex character. The intended artifact path was unchanged and its actual hash
matched the sealed Data30M metadata and historical experiment seals. This
clerical correction is committed and disclosed before a second invocation; no
decoding, seed, model, data membership, or metric policy changed.

This issue #9 pilot starts only from the sealed validation-selected 50M/Data30M
checkpoint. AdamW is initialized fresh. It uses seed 9009, LR `1e-5`, weight
decay `0.1`, batch 8, accumulation 2, CUDA BF16, and gradient clipping at 1.0.
There is one setting and no resume or sweep.

The canonical approved corpus is hash-bound directly. `prepared.jsonl` is never
read. The write-once export contains token IDs and continuation-only labels for
160 train and 20 validation examples; it cannot contain the 20 test examples.
The training config has no continuation-test, Scene Scorecard, Benchmark v2, or
clean-fiction test path. Those paths exist only in a separately hash-locked
protocol that the post-selection command opens after its selection gate passes.
Corpus preparation necessarily reads the approved canonical 200-row source to
reproduce grouped assignments, but it discards held-out rows before
serialization and exports no test token IDs, labels, or text. Training and
checkpoint selection consume only the sealed train/validation export.

Each of 160 SFT train examples is presented exactly twice: 320 presentations in
40 full batches. Two epochs each interleave 20 SFT batches and 114 Data30M LM
replay batches. This gives 40/268 = 14.925373% SFT batches and examples, not an
exact 15%. Exact 15% is arithmetically impossible with 40 indivisible batch-8
SFT presentations. Replay exposure is exactly 1,824 examples; token-target
counts are recorded separately and are not represented as a 15/85 ratio.

Validation runs at initialization and after each SFT epoch. Selection is minimum
continuation validation NLL, with earliest optimizer step breaking an exact tie.
Only the post-selection command can serialize or evaluate continuation test
examples, or open Data30M test, reserved Scene Scorecard prompts, and Benchmark
v2 test/generalization for model evaluation.

Preflight proves hashes, masks and padding, finite gradients, exact base-weight
reload, a real 50M CUDA BF16 update fit, and one-example loss reduction. The
memorization result is plumbing evidence only.

After selection, the immutable command evaluates continuation test and clean
fiction likelihood, runs matched Benchmark v2 forced-choice, teacher-forced,
and complete free-generation evaluation across all three splits, and generates
the exact frozen 40-output Scene Scorecard candidate
matrix, reuses rather than regenerates the sealed base raw outputs, and emits a
matched 80-output blinded base/candidate packet explicitly marked human review
pending. Independent human scoring remains pending and no narrative-quality
claim is authorized from automated results.

## Frozen identities

- Training config: `07a4c2f93a285f32a865e04db0074d8aa7582c396662b2908f57ccd0baa92523`
- Post-selection protocol: `f45041dcfb87e4f3dea8c0abce988beb346a8d77991305650cd8e46bebed0b87`
- Canonical examples: `e3e17d0e...f176b`
- Approval: `cc610034...fb5d78`
- Tokenizer: `14d4abef...f7012`
- Base checkpoint: `8da266fb...ecd01`
- Base seal: `ac9f0abf...51fb3`
- Scene protocol/prompts/rubric: `7c438ad6...f2718` / `fe828244...ff16` / `cee906f1...fb8b`
- Benchmark v2 protocol/authored manifest: `84e34abf...5c79` / `f4ed9129...079789`

## Commands

Freeze `CONFIG_SHA256` only after implementation review. Freeze
`EXPORT_SHA256` from the write-once export result before preflight.

```bash
source .venv/bin/activate
CONFIG_SHA256=$(sha256sum configs/continuation-sft-v1.yaml | cut -d' ' -f1)
python -m src.continuation_sft --config-sha256 "$CONFIG_SHA256" export
EXPORT_SHA256=$(sha256sum data/continuation_sft_v1/manifest.json | cut -d' ' -f1)
python -m src.continuation_sft --config-sha256 "$CONFIG_SHA256" preflight \
  --export-manifest data/continuation_sft_v1/manifest.json --export-sha256 "$EXPORT_SHA256"
PREFLIGHT_SHA256=$(sha256sum runs/fictionpulper-50m-data30m-continuation-sft-v1-preflight.json | cut -d' ' -f1)
python -m src.continuation_sft --config-sha256 "$CONFIG_SHA256" train \
  --export-manifest data/continuation_sft_v1/manifest.json --export-sha256 "$EXPORT_SHA256" \
  --preflight runs/fictionpulper-50m-data30m-continuation-sft-v1-preflight.json \
  --preflight-sha256 "$PREFLIGHT_SHA256"
```

Do not run until selection is complete and its selected hash has been reviewed:

```bash
SUMMARY_SHA256=$(sha256sum runs/fictionpulper-50m-data30m-continuation-sft-v1/summary.json | cut -d' ' -f1)
python -m src.continuation_sft --config-sha256 "$CONFIG_SHA256" post-selection \
  --summary runs/fictionpulper-50m-data30m-continuation-sft-v1/summary.json \
  --summary-sha256 "$SUMMARY_SHA256" \
  --checkpoint checkpoints/fictionpulper-50m-data30m-continuation-sft-v1/epoch-SELECTED.pt \
  --protocol-sha256 f45041dcfb87e4f3dea8c0abce988beb346a8d77991305650cd8e46bebed0b87
```
