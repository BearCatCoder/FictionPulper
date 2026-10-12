# Continuation-SFT v1

Status: bounded pilot and automated post-selection evaluation complete;
independent human review pending; decision `STOP_NO_PROMOTION`.

The first post-selection invocation stopped before scene or Benchmark v2
generation because the protocol's Data30M test-binary hash had one extra trailing
hex character. The intended artifact path was unchanged and its actual hash
matched the sealed Data30M metadata and historical experiment seals. This
clerical correction is committed and disclosed before a second invocation; no
decoding, seed, model, data membership, or metric policy changed.

The completed second invocation preserved all correct raw outputs, but its
Scene Scorecard packet path received the final Benchmark v2 packet because of a
local-variable reuse bug. The original run remains untouched. The tracked
correction command rebuilds only that packet and private key from the preserved
40 base plus 40 candidate scene records into a separate write-once directory.

## Result and decision

The preregistration was frozen at commit
`d7a0eab5b59e5a9cc5e4109a6307f5b8bebd9604`. Preflight passed serialization,
continuation/padding masks, finite gradients for every trainable parameter,
exact base reload, real-model CUDA BF16 fit, and tiny-example plumbing. Tiny loss
fell from `3.713713` to `0.000789`; this is not storytelling evidence.

The run completed all 134 optimizer steps. Epoch 2 was selected using only
continuation validation NLL, which fell from `4.026967` at the unchanged base to
`3.783913`. The selected checkpoint saw 320 SFT examples / 112,418 continuation
targets and 1,824 replay examples / 1,740,200 LM targets. No held-out row was
used for checkpoint selection.

Post-selection results were mixed and do not establish narrative improvement:

| Metric | Base | Candidate | Candidate minus base |
|---|---:|---:|---:|
| Continuation test NLL, 6,969 targets | 4.061925 | 3.816288 | -0.245637 |
| Continuation test perplexity | 58.0860 | 45.4352 | -12.6508 |
| Clean Data30M test loss, 1,546,717 targets | 3.101755 | 3.107070 | +0.005315 |
| Clean Data30M test perplexity | 22.2370 | 22.3555 | +0.1185 |
| Scene length-compliant outputs | 40/40 | 29/40 | -11 |
| Scene exact fact-phrase mentions | 0/160 | 0/160 | 0 |
| Scene mean repeated 4-gram rate | 0.228872 | 0.211395 | -0.017478 |
| Scene mean repeated-sentence rate | 0.118346 | 0.158970 | +0.040624 |

Benchmark v2 forced-choice paired-reversal success was unchanged on development
(`17.86%`), improved from `3.57%` to `7.14%` on test, and remained `0%` on
generalization. Teacher-forced paired preference remained `100%`, `0%`, and
`0%` respectively. Candidate free generation produced all 896 locked outputs;
automated diagnostics found zero premise-retention and current-fact-only hits in
every overall split/trial cell. These lexical diagnostics are not human scores.

The frozen decision is `STOP_NO_PROMOTION`. Independent blinded review of the
corrected 80-row matched Scene packet is still required before any narrative
quality or #12 promotion claim. No second seed, ratio, data revision, decoding
change, or hyperparameter sweep is authorized. The base received no matched
LM-only updates, so this comparison does not isolate an SFT-specific causal
effect. The next action is independent Scene review, not more training.

## Run identities

- Training export manifest: `8d38726277615e43824d021d6a01154779905de0a93fdd2973fe15bcd97e38f4`
- Preflight: `622301620e10069e2cc8e3ceb8978c1f8459e31ca9594e9db2dcc4d1adeb8d6c`
- Selection summary: `0cc0eb32f453280ddf8752d4ca3eaf15cf02624315c56a77800033576f9ef9f5`
- Selected checkpoint: `7d4ef889435626304cac24a2ce651ffd20ea9d85a5f882b10d30033a6d2d3ad5`
- Post-selection report: `8f84a415bfa200aba250db660638bf471222190bfa9c3e4d70cfa9f40fcb7cdd`
- Candidate Scene raw matrix: `ee91eb1c1ddffa0795e672797731f72c4959062e76904cd11bebdc79386aa767`
- Corrected matched Scene packet: `69ce3baf7efd378fcc3f7cfa94e6838287bfc89ab1c11918f4575d11c3b78a5b`
- Candidate Benchmark raw development/test/generalization: `3806faf1...f1577` / `18af8e54...b2f6` / `f88e631c...f9ce`

The ignored run/checkpoint outputs remain local and hash-bound. The tracked
report does not commit checkpoints, generated scenes, review keys, or raw
copyrighted data.

This issue #9 pilot starts only from the sealed validation-selected 50M/Data30M
checkpoint. AdamW is initialized fresh. It uses seed 9009, LR `1e-5`, weight
decay `0.1`, batch 8, accumulation 2, CUDA BF16, and gradient clipping at 1.0.
There is one setting and no resume or sweep.

The canonical approved corpus is hash-bound directly. `prepared.jsonl` is never
read. The write-once export contains token IDs and continuation-only labels for
160 train and 20 validation examples; it cannot contain the 20 test examples.
Its manifest hash is fixed in the loader after export review, so preflight and
training verify the sealed artifact without reopening the canonical source or
held-out rows.
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

python -m src.continuation_sft --config-sha256 "$CONFIG_SHA256" correct-scene-packet \
  --source-dir runs/fictionpulper-50m-data30m-continuation-sft-v1/post-selection \
  --source-report-sha256 8f84a415bfa200aba250db660638bf471222190bfa9c3e4d70cfa9f40fcb7cdd \
  --candidate-raw-sha256 ee91eb1c1ddffa0795e672797731f72c4959062e76904cd11bebdc79386aa767 \
  --protocol-sha256 f45041dcfb87e4f3dea8c0abce988beb346a8d77991305650cd8e46bebed0b87
```
