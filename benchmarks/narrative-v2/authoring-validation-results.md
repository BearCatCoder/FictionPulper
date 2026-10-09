# Authored Scenario Audit

This file contains deterministic generator and audit facts only. It does not claim that tests, model training, model evaluation, inference, checkpoint selection, or human review ran.

```text
status: PASS
release_ready: true
git_commit: PENDING_UNTIL_COMMIT
scenario_count: 112
split_counts: development=28, test=28, generalization=56
family_depth_cells: 28 (4 scenarios each)
reasoning_depth_counts: {"1": 28, "2": 28, "3": 28, "4": 14, "5": 14}
proof_replay: PASS (224 worlds)
template_signature_isolation: PASS
cross_split_leakage: 0
all_surface_exact_duplicates: 0
all_surface_near_duplicates_above_0.70: 0
candidate_only_controls: 112
fact_removed_replayable_chains: 0
token_boundary: PASS (14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012)
training_contamination: PASS (118252 rendered training records; 12 provenance files)
```

The manifest hashes this computed report and all other generated artifacts except the manifest itself. Test and generalization remain post-selection data; build-time authorship does not authorize evaluation-time access.
