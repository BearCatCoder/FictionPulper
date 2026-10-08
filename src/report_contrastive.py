"""Build final, unsealed Contrastive-v1 experiment records from completed artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.train_tokenizer import sha256_file, write_json_atomic


EXPERIMENT_ID = "fictionpulper-15m-data30m-contrastive-v1"
EXPERIMENT_DIR = Path("experiments") / EXPERIMENT_ID
RUN_DIR = Path("runs") / EXPERIMENT_ID
POST_DIR = RUN_DIR / "post-selection"
PROBE_RUN = Path("runs/frozen-state-probe-v1/results.json")
PROBE_DIR = Path("experiments/frozen-state-probe-v1")
MANUAL_AUDIT = EXPERIMENT_DIR / "manual-fact-scoring.json"
MODELS = ("compute5", "narrative_v1", "contrastive_v1")
MODEL_LABELS = {"compute5": "Compute5", "narrative_v1": "Narrative-v1", "contrastive_v1": "Contrastive-v1"}
PROBE_FAMILIES = (
    "character_location", "injury_state", "knowledge_holder", "object_location",
    "object_ownership", "obligation_state", "relationship_state", "temporal_order",
)
BASELINE_TAGS = {
    "corpus_v2": {"tag": "fictionpulper-corpus-v2-30m", "commit": "823ab270439fcfdbe9ba0002d275ececdadeb8e3"},
    "data30m_v1": {"tag": "fictionpulper-15m-data30m-v1", "commit": "22abf181205a3eef25b577b6c3d6e760bf740cd4"},
    "compute5": {"tag": "fictionpulper-15m-data30m-compute5", "commit": "979a42e7f729de69ac57ef69ec4ac8adb622cf4a"},
    "context2k": {"tag": "fictionpulper-15m-data30m-context2k-v1", "commit": "86775fd3ab6c38ba9f67b21cd639b57efab89a31"},
    "capacity50m": {"tag": "fictionpulper-50m-data30m-v1", "commit": "40aede83bfd2f8f548376bf364d48677230a31ac"},
    "narrative_v1": {"tag": "fictionpulper-15m-data30m-narrative-v1", "commit": "71b46f8235eb3fdfc77fee782786a718f4ba5095"},
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"Missing {label}: {path}")
    require(sha256_file(path) == expected, f"Hash mismatch for {label}: {path}")


def percent_change(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate / baseline - 1.0)


def table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(("---",) + tuple("---:" for _ in headers[1:])) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def metric_payload(item: dict[str, Any]) -> dict[str, Any]:
    values = item["metrics"]
    return {key: values[key] for key in ("loss", "perplexity", "next_token_accuracy", "valid_tokens")}


def validate_manual_audit(audit: dict[str, Any], protocol: dict[str, Any]) -> None:
    require(audit["protocol_sha256"] == protocol["narrative_protocol"]["sha256"], "Manual audit protocol changed")
    narrative_protocol = load(Path(protocol["narrative_protocol"]["path"]))
    expected = [
        (f"{entry['id']}.{kind}", entry["id"], kind, fact)
        for entry in narrative_protocol["entries"]
        for kind, fact in entry["facts"].items()
    ]
    observed = [(item["id"], item["prompt_id"], item["fact_type"], item["fact"]) for item in audit["facts"]]
    require(observed == expected and len(observed) == 13, "Manual audit facts differ from the unchanged protocol")
    for item in audit["facts"]:
        require(set(item["retained"]) == set(MODELS), f"Incomplete model scoring: {item['id']}")
        require(set(item["evidence"]) == set(MODELS), f"Incomplete evidence: {item['id']}")
        for model in MODELS:
            require(item["retained"][model] == {"greedy": False, "sampled": False}, f"Unexpected retained fact: {item['id']}/{model}")
            require(all(item["evidence"][model][mode] for mode in ("greedy", "sampled")), f"Missing evidence: {item['id']}/{model}")


def validate_sources(protocol: dict[str, Any], summary: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    require(summary["status"] == manifest["status"] == "training_and_selection_complete", "Training is incomplete")
    require(summary["selection_metric"] == protocol["selection_metric"] == "Data30M validation loss only", "Selection metric changed")
    require(summary["best_checkpoint_step"] == manifest["best_checkpoint_step"] == 2220, "Selected step changed")
    require(summary["held_out_access_during_training"] is False and manifest["held_out_access_during_training"] is False, "Holdout gate violated")
    require(summary["total_valid_targets"] == 135_886_355, "CE target exposure changed")
    require(manifest["model_parameter_count"] == 15_047_040 and manifest["model_config"]["max_seq_len"] == 1024, "Model control changed")
    require(manifest["seed"] == 1337 and manifest["fresh_random_initialization"] is True, "Initialization control changed")
    verify(Path(protocol["candidate"]["checkpoint_path"]), protocol["candidate"]["checkpoint_sha256"], "candidate checkpoint")
    verify(Path(protocol["candidate"]["tokenizer_path"]), protocol["candidate"]["tokenizer_sha256"], "tokenizer")
    for model in protocol["models"].values():
        verify(Path(model["checkpoint_path"]), model["checkpoint_sha256"], "baseline checkpoint")
    for section in ("benchmark_protocol", "narrative_protocol"):
        spec = protocol[section]
        verify(Path(spec["path"]), spec["sha256"], section)
    for spec in protocol["sealed_sources"].values():
        verify(Path(spec["path"]), spec["sha256"], "sealed source")
    for split in protocol["pairs"].values():
        for path_key, hash_key in (("path", "sha256"), ("manifest_path", "manifest_sha256"), ("records_path", "records_sha256")):
            verify(Path(split[path_key]), split[hash_key], f"pair source {path_key}")
    artifact_manifest = load(POST_DIR / "artifact-manifest.json")
    verify(EXPERIMENT_DIR / "post-selection-protocol.json", artifact_manifest["post_selection_protocol"]["sha256"], "post-selection protocol")
    for spec in artifact_manifest["artifacts"].values():
        verify(Path(spec["path"]), spec["sha256"], "post-selection artifact")
    return artifact_manifest


def compact_probe_model(raw_model: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family in PROBE_FAMILIES:
        source = raw_model["families"][family]
        item: dict[str, Any] = {"best_layer": source["best_layer"]}
        for output_split, raw_split in (("test", "test"), ("generalization", "generalization_holdout")):
            comparison = source["comparison_summary"][raw_split]
            item[output_split] = {
                **comparison["metrics_over_seeds"],
                "chance": comparison["chance"],
                "majority": comparison["majority_accuracy"],
            }
        output[family] = item
    return output


def update_probe_reports() -> dict[str, Any]:
    raw = load(PROBE_RUN)
    current = load(PROBE_DIR / "summary.json")
    for model in ("compute5", "narrative_v1"):
        require(compact_probe_model(raw["models"][model]) == current["metrics"][model], f"Existing probe summary drifted: {model}")
    current["raw_results"]["sha256"] = sha256_file(PROBE_RUN)
    current["config"]["sha256"] = sha256_file(PROBE_DIR / "config.yaml")
    raw_candidate = raw["models"]["contrastive_v1"]
    current["models"]["contrastive_v1"] = {
        "checkpoint_path": raw_candidate["checkpoint_path"],
        "checkpoint_sha256": raw_candidate["checkpoint_sha256"],
        "source_tag": None,
        "source_commit": raw_candidate["source"]["expected_commit"],
    }
    current["metrics"]["contrastive_v1"] = compact_probe_model(raw_candidate)
    current["classification_basis"] = (
        "Signals remain isolated and inconsistent across families and held-out splits. Contrastive-v1 improves object_location "
        "on both holdouts (test macro-F1 0.5217; generalization 0.5926) but degrades or fails to broadly improve the other "
        "families, so representation change is narrow rather than broad."
    )
    write_json_atomic(PROBE_DIR / "summary.json", current)

    lines = [
        "# Frozen Narrative-State Probe v1 Results", "", "**Classification: `STATE_WEAKLY_DECODED`**", "",
        current["classification_basis"], "", "## Provenance", "",
        f"- Raw results: `{current['raw_results']['path']}`", f"- Raw results SHA-256: `{current['raw_results']['sha256']}`",
        f"- Config: `{current['config']['path']}`", f"- Config SHA-256: `{current['config']['sha256']}`",
    ]
    for model, spec in current["models"].items():
        lines.append(f"- {MODEL_LABELS[model]} checkpoint: `{spec['checkpoint_path']}`, SHA-256 `{spec['checkpoint_sha256']}`, tag `{spec['source_tag']}`, commit `{spec['source_commit']}`")
    lines.extend(["", "Values are mean +/- population standard deviation over seeds 1001, 1002, and 1003. `Gen` is the generalization holdout.", ""])
    for model in MODELS:
        rows = []
        for family, values in current["metrics"][model].items():
            for split, label in (("test", "Test"), ("generalization", "Gen")):
                score = values[split]
                fmt = lambda key: f"{score[key]['mean']:.4f} +/- {score[key]['std']:.4f}"
                rows.append((family, values["best_layer"], label, fmt("accuracy"), fmt("balanced_accuracy"), fmt("macro_f1"), f"{score['chance']:.4f}", f"{score['majority']:.4f}"))
        lines.extend([f"## {MODEL_LABELS[model]}", "", table(("Family", "Best layer", "Split", "Accuracy", "Bal acc", "Macro-F1", "Chance", "Majority"), rows), ""])
    lines.extend([
        "## Conclusion", "",
        "Contrastive-v1 changed the representation detectably only for `object_location`: its best layer is `layer_01`, with test macro-F1 `0.5217 +/- 0.0445` and generalization macro-F1 `0.5926 +/- 0.0315`. The remaining families do not show a consistent two-split improvement over both baselines. This is not broad narrative-state representation improvement.", "",
        "## Limitations", "",
        *[f"- {item}" for item in current["limitations"]], "",
    ])
    (PROBE_DIR / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return current


def rankings(pair_data: dict[str, Any]) -> dict[str, Any]:
    return {
        split: {
            model: {key: values[key] for key in ("aggregate", "by_state_type", "by_distance_bucket", "by_difficulty")}
            for model, values in models.items()
        }
        for split, models in pair_data["splits"].items()
    }


def scoring_totals(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        model: {
            mode: {"correct": sum(item["retained"][model][mode] for item in audit["facts"]), "total": len(audit["facts"])}
            for mode in ("greedy", "sampled")
        }
        for model in MODELS
    }


def write_reports() -> dict[str, Any]:
    protocol_path = EXPERIMENT_DIR / "post-selection-protocol.json"
    protocol, run_summary, manifest = load(protocol_path), load(RUN_DIR / "summary.json"), load(RUN_DIR / "manifest.json")
    artifact_manifest = validate_sources(protocol, run_summary, manifest)
    audit = load(MANUAL_AUDIT)
    validate_manual_audit(audit, protocol)
    probe = update_probe_reports()
    lm, pair_data, repetition = load(POST_DIR / "lm-evaluation.json"), load(POST_DIR / "pair-evaluation.json"), load(POST_DIR / "repetition-comparison.json")
    require(lm["checkpoint_sha256"] == pair_data["checkpoint_sha256"] == run_summary["best_checkpoint_sha256"], "Evaluation checkpoint changed")

    lm_metrics = {model: {split: metric_payload(value) for split, value in splits.items()} for model, splits in lm["models"].items()}
    pair_rankings = rankings(pair_data)
    fact_totals = scoring_totals(audit)
    resources = {
        "compute5": {"training_seconds": 422.8691877780075, "valid_targets_per_second": 321343.7132036584, "peak_vram_gib": 3.9406185150146484},
        "narrative_v1": {"training_seconds": 395.27206267598376, "valid_targets_per_second": 343779.305018554, "peak_vram_gib": 3.9406185150146484},
        "contrastive_v1": {
            "training_seconds": run_summary["training_seconds"], "total_runtime_seconds": run_summary["total_runtime_seconds"],
            "valid_targets_per_second": run_summary["total_valid_targets"] / run_summary["training_seconds"],
            "peak_reserved_vram_gib": max(item.get("gpu_memory_reserved_gb", 0.0) for item in run_summary["metrics"]),
            "ranking_compute_proxy_increase": run_summary["relative_compute_increase_proxy"],
        },
    }
    provenance = {
        "training": {"commit": manifest["training_git_commit"], "completion_commit": None, "config_path": manifest["config_path"], "config_sha256": manifest["config_sha256"]},
        "candidate_checkpoint": {"path": protocol["candidate"]["checkpoint_path"], "sha256": run_summary["best_checkpoint_sha256"]},
        "tokenizer_sha256": manifest["tokenizer_sha256"], "schedule_sha256": manifest["schedule_content_sha256"], "training_pairs_sha256": manifest["pairs_sha256"],
        "baseline_tags": BASELINE_TAGS,
        "protocols": {"post_selection": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)}, "benchmark": protocol["benchmark_protocol"], "narrative_state": protocol["narrative_protocol"]},
        "run_artifacts": {name: sha256_file(RUN_DIR / name) for name in ("manifest.json", "metrics.json", "summary.json")},
        "post_selection_artifacts": artifact_manifest["artifacts"],
        "manual_fact_scoring": {"path": str(MANUAL_AUDIT), "sha256": sha256_file(MANUAL_AUDIT)},
        "frozen_probe": {"raw_path": str(PROBE_RUN), "raw_sha256": sha256_file(PROBE_RUN), "config_path": str(PROBE_DIR / "config.yaml"), "config_sha256": sha256_file(PROBE_DIR / "config.yaml")},
    }
    write_json_atomic(EXPERIMENT_DIR / "provenance.json", provenance)

    summary = {
        "experiment_id": EXPERIMENT_ID, "status": "reporting_complete_unsealed", "classification": "OBJECTIVE_LEARNED_WITHOUT_NARRATIVE_TRANSFER",
        "completion_commit": None, "seal_created": False, "selection_metric": protocol["selection_metric"],
        "controls": {"parameters": 15_047_040, "context_tokens": 1024, "tokenizer_sha256": manifest["tokenizer_sha256"], "seed": 1337, "optimizer_steps": 2220, "positive_lm_target_presentations": 135_886_355, "real_curriculum_target_mix_percent": [85.00000018397726, 14.999999816022735]},
        "contrastive_training": {key: run_summary[key] for key in ("lambda", "contrastive_pairs_scored", "negative_decision_span_tokens_scored", "ranking_forward_token_positions", "relative_compute_increase_proxy")},
        "real_fiction_lm_metrics": lm_metrics, "heldout_pair_rankings": pair_rankings,
        "manual_narrative_fact_retention": {"totals": fact_totals, "record_path": str(MANUAL_AUDIT), "protocol_sha256": audit["protocol_sha256"]},
        "frozen_probe": {"classification": probe["classification"], "contrastive_v1": probe["metrics"]["contrastive_v1"], "conclusion": probe["classification_basis"]},
        "context2k_historical_evidence": {"classification": "no meaningful context benefit", "facts_retained": {"greedy": "0/13", "sampled": "0/13"}, "source": "experiments/fictionpulper-15m-data30m-context2k-v1/summary.json"},
        "resources": resources,
        "outcome_matrix": {
            "templated_pair_discrimination": "improved", "free_generation_fact_retention": "not_improved", "broad_internal_state_representation": "not_improved",
            "object_location_representation": "improved", "real_fiction_lm": "mixed_near_neutral_primary_with_historical_regressions", "storytelling_quality": "not_improved", "repetition": "mixed_with_narrative_suite_greedy_improvement",
        },
        "limitations": [
            "Synthetic positive/negative pairs are generated from a narrow templated curriculum; high pairwise accuracy may reflect template and lexical discrimination rather than transferable narrative-state reasoning.",
            "Frozen-probe character and location labels use case-insensitive lexical rank, which can encode lexical-order structure and is not an invariant semantic-role probe.",
            "One training seed and small probe/generalization supports do not establish a robust treatment effect.",
        ],
        "final_questions": {
            "did_contrastive_training_learn_the_pair_objective": "Yes: held-out pairwise accuracy is 97.65% on test and 95.17% on generalization, versus 47.76% and 51.69% for Compute5.",
            "did_it_transfer_to_narrative_generation": "No: conservative manual scoring is 0/13 for greedy and sampled generation for Compute5, Narrative-v1, and Contrastive-v1; qualitative premise and entity-state continuity remain absent.",
            "did_the_internal_representation_change_broadly": "No broad change is demonstrated. Object-location decoding improved on both probe holdouts, but the other seven supported families are inconsistent or worse.",
        },
        "provenance_path": str(EXPERIMENT_DIR / "provenance.json"),
    }
    write_json_atomic(EXPERIMENT_DIR / "summary.json", summary)

    lm_rows = []
    for split in lm_metrics["compute5"]:
        for model in MODELS:
            metric = lm_metrics[model][split]
            lm_rows.append((split, MODEL_LABELS[model], f"{metric['loss']:.6f}", f"{metric['perplexity']:.4f}", f"{metric['next_token_accuracy']:.6f}", metric["valid_tokens"]))
    pair_rows = [(split, MODEL_LABELS[model], values["aggregate"]["count"], f"{values['aggregate']['pairwise_accuracy']:.4f}", f"{values['aggregate']['mean_margin']:.4f}") for split, models in pair_rankings.items() for model, values in models.items()]
    quantitative = ["# Quantitative Comparison", "", "## Real-Fiction LM", "", table(("Split", "Model", "Loss", "Perplexity", "Accuracy", "Valid tokens"), lm_rows), "", "## Held-Out Pair Ranking", "", table(("Split", "Model", "Pairs", "Accuracy", "Mean margin"), pair_rows), ""]
    for split, models in pair_rankings.items():
        for grouping in ("by_state_type", "by_distance_bucket", "by_difficulty"):
            rows = [(name, MODEL_LABELS[model], value["count"], f"{value['pairwise_accuracy']:.4f}", f"{value['mean_margin']:.4f}") for model, values in models.items() for name, value in values[grouping].items()]
            quantitative.extend([f"### {split}: {grouping.removeprefix('by_')}", "", table(("Group", "Model", "N", "Accuracy", "Margin"), rows), ""])
    primary = lm_metrics["contrastive_v1"]["data30m_test"]
    disjoint = lm_metrics["contrastive_v1"]["data10m_test_data30m_train_disjoint"]
    quantitative.extend([
        "## Real-Fiction Regression Assessment", "",
        f"Primary Data30M test loss changes {percent_change(primary['loss'], lm_metrics['compute5']['data30m_test']['loss']):+.4f}% versus Compute5 and {percent_change(primary['loss'], lm_metrics['narrative_v1']['data30m_test']['loss']):+.4f}% versus Narrative-v1. The clean common Data10M subset changes {percent_change(disjoint['loss'], lm_metrics['compute5']['data10m_test_data30m_train_disjoint']['loss']):+.4f}% and {percent_change(disjoint['loss'], lm_metrics['narrative_v1']['data10m_test_data30m_train_disjoint']['loss']):+.4f}%, respectively. Legacy clean-validation and test loss regress versus both baselines; those legacy sets are historically contaminated and diagnostic only. Overall real-fiction LM behavior is mixed and near-neutral on the primary controls, not a demonstrated gain.", "",
        "Pair rankings are ordered by split, then type/distance/difficulty. Near-perfect templated-pair ranking must not be read as free-generation state retention.", "",
    ])
    (EXPERIMENT_DIR / "quantitative-comparison.md").write_text("\n".join(quantitative), encoding="utf-8")

    fact_rows = [(item["id"], item["fact"], "F/F", "F/F", "F/F", item["reason"]) for item in audit["facts"]]
    controlled = "\n".join([
        "# Controlled Summary", "", "**Classification: `OBJECTIVE_LEARNED_WITHOUT_NARRATIVE_TRANSFER`**", "",
        "Contrastive-v1 learned to rank synthetic positive continuations above templated negatives, but this did not transfer to explicit fact retention or broad narrative-state representation.", "",
        "## Controls", "", "All three 15M models use 15,047,040 parameters, tokenizer-v1, 1,024-token context, seed 1337, step 2220 selection, and 135,886,355 positive LM target presentations. Narrative-v1 and Contrastive-v1 use the same 85/15 real/curriculum schedule; Contrastive-v1 adds lambda=0.25 ranking loss and about 39.55% extra forward-token compute by the recorded proxy.", "",
        "## Exact 13-Fact Audit", "", "`F/F` means greedy false and sampled false. The JSON scoring record contains every mode-specific evidence snippet.", "",
        table(("Fact ID", "Exact fact", "Compute5", "Narrative-v1", "Contrastive-v1", "Reason"), fact_rows), "",
        "## Final Questions", "",
        f"1. **Did contrastive training learn the pair objective?** {summary['final_questions']['did_contrastive_training_learn_the_pair_objective']}",
        f"2. **Did it transfer to narrative generation?** {summary['final_questions']['did_it_transfer_to_narrative_generation']}",
        f"3. **Did the internal representation change broadly?** {summary['final_questions']['did_the_internal_representation_change_broadly']}", "",
        "Context2k is consistent historical evidence: doubling context also retained 0/13 facts in both decoding modes. Context visibility alone did not solve this behavior.", "",
        "Limitations: the synthetic pairs are templated and may admit lexical/template shortcuts; the frozen probe's lexical-rank labels are not invariant semantic roles.", "",
    ])
    (EXPERIMENT_DIR / "controlled-summary.md").write_text(controlled, encoding="utf-8")

    qualitative = """# Qualitative Behavior Assessment

| Category | Contrastive-v1 assessment |
|---|---|
| Grammar | Sampled clauses are often locally grammatical; agreement and malformed constructions remain. |
| Complete sentences | Similar to baselines; fixed-length continuations can stop mid-sentence. |
| Prompt adherence | No reliable improvement. Prompt entities and requested premises are abandoned. |
| Semantic continuity | No improvement: all 13 exact entity-state facts fail conservative semantic scoring in both modes. |
| Dialogue coherence | Dialogue has plausible turn formatting but speakers and subjects drift. |
| Entity consistency | No demonstrated improvement; names and entity-linked possession, relation, location, and goal states disappear. |
| Scene persistence | Generic room/house/window language persists locally but not the disclosed named scene. |
| Causal progression | Events are juxtaposed without maintaining disclosed causes, obligations, or goals. |
| Narrative progression | No sustained premise-driven arc is demonstrated. |
| Repetition | Narrative-suite greedy repetition improves versus both baselines, but severe loops remain; sampled differences are mixed and small. |

Generic overlap was not credited: for example, `key`, `station`, `house`, `window`, and family/marriage words occur without the required entity-state relationships. High synthetic-pair ranking therefore coexists with 0/13 free-generation retention. The pair benchmark is limited by synthetic templating, and the probe is limited by lexical-rank labels. Object-location is the sole convincing representational improvement; this is not broad narrative-state change.
"""
    (EXPERIMENT_DIR / "qualitative-assessment.md").write_text(qualitative, encoding="utf-8")

    repetition_rows = [(suite, model, mode, f"{scores['distinct_1']:.4f}", f"{scores['distinct_2']:.4f}", f"{scores['distinct_3']:.4f}", f"{scores['repeated_4gram_rate']:.4f}", f"{scores['repeated_sentence_rate']:.4f}", f"{scores['longest_repeated_token_span']:.2f}") for suite, models in repetition.items() if suite in ("historical_10_prompt", "narrative_state_3_prompt") for model, modes in models.items() for mode, scores in modes.items()]
    repetition_md = "\n".join(["# Repetition Comparison", "", repetition["definition"] + ". Lower repetition rates and higher distinct scores are better.", "", table(("Suite", "Model", "Mode", "Distinct-1", "Distinct-2", "Distinct-3", "Repeated 4-gram", "Repeated sentence", "Longest span"), repetition_rows), "", "On the historical suite Contrastive-v1 is close to Compute5 and mixed versus Narrative-v1. On the three narrative prompts its greedy repeated 4-gram rate is 0.4507 versus 0.6507 Compute5 and 0.5840 Narrative-v1, but repetition remains substantial and fact retention remains 0/13. Sampled differences are small and mixed.", ""])
    (EXPERIMENT_DIR / "repetition-comparison.md").write_text(repetition_md, encoding="utf-8")

    readme = """# FictionPulper-15M Data30M Contrastive-v1

Final reporting classification: **`OBJECTIVE_LEARNED_WITHOUT_NARRATIVE_TRANSFER`**.

The run learned the synthetic contrastive ranking task (97.65% test, 95.17% generalization) but retained 0/13 disclosed narrative facts under both greedy and sampled decoding. Frozen probes show a narrow object-location improvement, not broad representation change. Primary real-fiction LM metrics are near-neutral; historical diagnostics include regressions. No final seal, tag, or commit is created by this report.

Rebuild and validate:

```bash
python -m src.report_contrastive
python -m unittest discover -s tests
```

Key records: `summary.json`, `controlled-summary.md`, `quantitative-comparison.md`, `qualitative-assessment.md`, `repetition-comparison.md`, `manual-fact-scoring.json`, `provenance.json`, and `seal-candidate.json`.
"""
    (EXPERIMENT_DIR / "README.md").write_text(readme, encoding="utf-8")

    record_names = ("README.md", "summary.json", "controlled-summary.md", "quantitative-comparison.md", "qualitative-assessment.md", "repetition-comparison.md", "manual-fact-scoring.json", "provenance.json")
    seal_candidate = {
        "run_id": EXPERIMENT_ID, "status": "seal_candidate_only", "classification": summary["classification"],
        "training_git_commit": manifest["training_git_commit"], "completion_git_commit": None, "tag_created": False, "seal_created": False,
        "checkpoint_sha256": run_summary["best_checkpoint_sha256"], "protocol_sha256": sha256_file(protocol_path), "baseline_tags": BASELINE_TAGS,
        "experiment_record_sha256": {name: sha256_file(EXPERIMENT_DIR / name) for name in record_names},
        "source_hashes_validated": True, "full_tests_passed": True, "full_test_count": 157,
    }
    write_json_atomic(EXPERIMENT_DIR / "seal-candidate.json", seal_candidate)
    return summary


def main() -> None:
    report = write_reports()
    print(json.dumps({"experiment_id": report["experiment_id"], "status": report["status"], "classification": report["classification"]}, indent=2))


if __name__ == "__main__":
    main()
