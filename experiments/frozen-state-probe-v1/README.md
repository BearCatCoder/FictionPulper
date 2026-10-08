# Frozen Narrative-State Probe v1

This diagnostic freezes each transformer and trains only biased `nn.Linear`
classifiers on Narrative Curriculum v1. It captures the embedding output, every
post-transformer-layer state, and the final RMS-normalized state at the final
token strictly before the primary resolution. Test and generalization splits do
not participate in epoch or layer selection.

Each family reads only events whose evidence ends before the primary resolution:

- `object_ownership`: current holder of the primary object
- `object_location`: current hiding location of the primary object
- `character_location`: current destination shared by the moved characters
- `knowledge_holder`: character who currently knows the primary secret
- `relationship_state`: role holder plus relationship subject and target
- `injury_state`: character carrying the persistent injury
- `temporal_order`: actor in the first event
- `obligation_state`: debtor and creditor

Every family is restricted to stories whose primary state supports that task.
Character and location slots are assigned by case-insensitive lexical rank
within the story, not by `generation_values` order. Thus sampled role arguments
vary among universal labels such as `character_0` and `location_1`, including
when validation and held-out splits use exclusive lexical values. A preflight
check requires at least two training classes and rejects held-out labels not
seen in training.

Goal status, object retrieval status, and cause/effect are explicitly unsupported
because their valid pre-resolution structural labels do not vary. Curriculum
difficulty, genre, primary state type, and raw attribute/entity probes are also
unsupported because they measure construction metadata or template identity,
not narrative state.

Run from the repository root:

```bash
python -m src.frozen_state_probe --config experiments/frozen-state-probe-v1/config.yaml
```

The checkpoint list is declarative: Compute5 and Narrative-v1 are enabled now,
and a disabled Contrastive slot documents the expected future checkpoint
interface. All enabled models use the exact same probe hyperparameters and seeds
1001, 1002, and 1003. Results include hashes for the config, tokenizer,
curriculum records and sidecars, and checkpoints, plus each configured source
tag resolved to its commit. Best-layer summaries report test and generalization
accuracy, balanced accuracy, and macro-F1 as mean and population standard
deviation over seeds, together with an explicit signal interpretation.
