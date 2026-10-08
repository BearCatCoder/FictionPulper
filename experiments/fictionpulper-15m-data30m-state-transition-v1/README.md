# FictionPulper-15M-data30M-state-transition-v1

Status: prepared for the single locked training run.

## Question

Can training-only linear supervision on the current abstract state make the sealed 15M architecture update and carry narrative state without changing its inference architecture, positive LM exposure, counterfactual objective, or decoding policy?

## Immutable Baselines

| Tag | Resolved commit |
|---|---|
| `fictionpulper-corpus-v2-30m` | `823ab270439fcfdbe9ba0002d275ececdadeb8e3` |
| `fictionpulper-15m-data30m-compute5` | `979a42e7f729de69ac57ef69ec4ac8adb622cf4a` |
| `fictionpulper-15m-data30m-narrative-v1` | `71b46f8235eb3fdfc77fee782786a718f4ba5095` |
| `fictionpulper-15m-data30m-contrastive-v1` | `ad0e69b1f0342de06eef1e8eb9be46728abb11e1` |
| `fictionpulper-narrative-state-localization-v1` | `f5108d7a6b7f7070650faa694d7f74405903e395` |
| `fictionpulper-15m-data30m-counterfactual-v1` | `947cbbc86cc5c61c0ef2c5a47f890b6bd29b2401` |

At preparation, `master` and `origin/master` both resolved to the Counterfactual-v1 seal commit and the tracked worktree was clean.

## Locked Design

- Base LM: 15,047,040 parameters, freshly initialized with seed 1337.
- Auxiliary heads: nine linear classifiers, 6,930 training-only parameters.
- Representation: final Transformer hidden state after final RMSNorm.
- State loss: mean annotation CE within each world/entity sequence, then mean across sequences.
- Total curriculum loss: positive LM CE + `0.25 * paired ranking` + `0.25 * state CE`.
- Training: the exact Counterfactual-v1 2,220-step schedule and 85/15 target exposure.
- Selection: Data30M real-fiction validation loss only.
- Inference: ordinary 15,047,040-parameter LM; auxiliary heads are excluded.

State labels are sidecar metadata under ignored `data/state_transition_v1/`. They are never inserted into model tokens.

## Commands

```bash
.venv/bin/python -m src.state_transition_annotations \
  --expected-source-manifest-sha256 41bfec4201e36a28fa5f4bea265b326ab7a0dbc67ba95c86533a1bdf191eede8 \
  --expected-tokenizer-sha256 14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012

.venv/bin/python -m unittest discover -s tests

.venv/bin/python -m src.train_state_transition \
  --config configs/data30m-15m-state-transition-v1.yaml
```

Training, post-selection evaluation, completion records, and sealing must use isolated State-Transition-v1 paths and must not overwrite prior artifacts.
