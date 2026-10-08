# Narrative State Localization v1

This unsealed diagnostic separates linear state availability, context-sensitive
forced choice, and teacher-forced versus rollout degradation. It trains no
language-model weights and only loads checkpoints identified by sealed hashes.

Run stages independently:

```bash
python -m src.frozen_state_probe --config experiments/narrative-state-localization-v1/probe-config.yaml
python -m src.narrative_state_localization --config experiments/narrative-state-localization-v1/config.yaml --stage context-swap
python -m src.narrative_state_localization --config experiments/narrative-state-localization-v1/config.yaml --stage forced-choice
python -m src.narrative_state_localization --config experiments/narrative-state-localization-v1/config.yaml --stage rollout
python -m src.report_narrative_state_localization --config experiments/narrative-state-localization-v1/config.yaml
```

The tracked directory contains protocols, configuration, and compact reports.
Raw model evaluations belong under `runs/narrative-state-localization-v1/`.
The reporter creates `seal-candidate.json`, never `seal.json`.

The target counterfactual grid is 63 cells. The immutable held-out sidecars
support 56: no synthetic source is invented for seven absent long-distance
state/distance combinations.

## Final Findings (Unsealed)

The reporter validates the frozen checkpoints, tags, tokenizer, curriculum,
protocols, raw source hashes, and embedded execution provenance before writing
compact reports. Run the focused reporter tests and regenerate reports with:

```bash
python -m unittest tests.test_narrative_state_localization
python -m src.report_narrative_state_localization --config experiments/narrative-state-localization-v1/config.yaml
python -m unittest discover -s tests
```

The primary classification is **D SYNTHETIC_SHORTCUT**, with weak
representation as a contributing limitation. Contrastive-v1's 97.65% original
synthetic pair ranking does not transfer to balanced context reversal (11/56)
or original-context forced choice (4/13). Compute5, Narrative-v1,
Contrastive-v1, and 50M score 4/13, 3/13, 4/13, and 7/13 respectively.

No probe family is robust on both test and generalization. Contrastive
`object_location` is the sole convincing relative improvement (test macro-F1
0.5217, generalization 0.5926), which is not broad state representation.
Contrastive rollout begins without a gold advantage (4/13, mean margin
-0.2288), so no meaningful monotonic collapse length can be identified.
Historical free generation remains 0/13 for all four models. The experiment
and `seal-candidate.json` remain explicitly unsealed; there is no `seal.json`.
