# FictionPulper-15M-data30M-narrative-v1

This controlled experiment compares the sealed 15M Compute5 baseline with a fresh 15M run trained at identical total valid-target exposure using 85% Data30M and 15% Narrative Continuity Curriculum v1 targets. Data30M validation loss alone selects the checkpoint. Curriculum validation is diagnostic only. No test, generalization-holdout, generation, repetition, or narrative-state result may affect selection.

## Preflight

After curriculum packing and mixed-schedule creation, generate the preflight record:

```bash
python -m src.prepare_narrative_experiment
```

The record resolves baseline tag commits and seal hashes, curriculum artifact hashes, the mixed schedule file/content hashes, the locked protocols, and explicit empty training commit/completion fields. It reads the curriculum manifest, not held-out JSONL contents.

## Post-Selection Evaluation

Only after `runs/fictionpulper-15m-data30m-narrative-v1/summary.json` reports `training_and_selection_complete` and identifies a hash-matching Data30M-validation-selected checkpoint, run the sole held-out evaluation entry point:

```bash
python -m src.narrative_post_selection
```

The CLI verifies the summary, recomputes the validation-only selection, verifies the selected checkpoint and schedule identity, and only then accesses held-out data in this order: Data30M test, curriculum test, curriculum generalization holdout, common held-out Data10M test, legacy clean validation, legacy test.

Teacher-forced loss, perplexity, and next-token accuracy are reported independently for every corpus benchmark. Curriculum state recall is not a teacher-forced metric: at every held-out sidecar resolution, greedy continuation starts exactly where `anchors[].resolved_quote` begins and generates exactly that quote's token length. A trial succeeds only if all generated tokenizer-v1 token IDs equal the metadata resolution token IDs in order. Results aggregate over anchor trials overall and by state type, distance bucket, difficulty, and genre. This intentionally strict reproducible metric can undercount semantically correct paraphrases; raw expected and generated resolutions are retained for audit.

The original ten prompts use the historical decoding settings and seed 11337 for both the candidate and the exact sealed Compute5 checkpoint. The exact 50M narrative-state protocol is run for both models, with greedy and sampled outputs left unscored and ready for separate fact scoring. Repetition outputs include distinct-1/2/3, repeated 4-gram rate, longest repeated token span, and repeated sentence rate over continuations only.

Full artifacts and their SHA-256 manifest are written under `runs/fictionpulper-15m-data30m-narrative-v1/post-selection/`. A compact `post-selection-record.json` is generated here for later review and experiment sealing. The tooling does not commit, tag, push, or modify any sealed experiment.
