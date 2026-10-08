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
