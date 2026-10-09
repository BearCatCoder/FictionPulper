# Validation Results

Validation was run on 2026-10-09 from the issue branch after freezing the
artifact hashes. No model inference, training, or human review was run.

```text
$ python -m src.benchmark_v2 --root benchmarks/narrative-v2
passed: true
scenario_count: 112
split_counts: development=28, test=28, generalization=56
family_depth_cells: 28
scenarios_per_family_depth_cell: 4
candidate_orders: XY=56, YX=56
world_a_labels: X=56, Y=56
cross_split_allocation_id_leakage: 0

$ python -m unittest tests.test_benchmark_v2
Ran 7 tests
OK

$ python -m unittest discover -s tests
Ran 257 tests in 5.794s
OK
```

This is an abridged result summary; the validator prints the same fields as
formatted JSON. The tests cover deterministic allocation, artifact hash drift,
stable-hash tampering, exact family/depth balance, distinct evaluation modes,
4+ depth semantics, and cross-split name-pool-ID leakage. Full authoring-time
surface contamination and near-duplicate audits remain a required gate because
scenario prose has not yet been authored.
