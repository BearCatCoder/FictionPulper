"""Report the sealed 15M versus 50M fixed-exposure capacity experiment."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import yaml
from tokenizers import Tokenizer

from src.context_diagnostics import repetition_metrics
from src.train import GENERATION_SETTINGS
from src.train_tokenizer import sha256_file, write_json_atomic


BASELINE_RUN = Path("runs/fictionpulper-15m-data30m-compute5")
RAW_CANDIDATE_RUN = Path("runs/smoke-5m-20261007-212756")
DERIVED_RUN = Path("runs/fictionpulper-50m-data30m-v1")
EXPERIMENT_DIR = Path("experiments/fictionpulper-50m-data30m-v1")
NARRATIVE_PATH = DERIVED_RUN / "narrative-state-comparison.json"
BASELINE_LABEL = "compute5_context1024"
CANDIDATE_LABEL = "capacity50m_context1024"
BASELINE_PARAMETERS = 15_047_040
CANDIDATE_PARAMETERS = 50_348_544
SELECTED_EPOCH = 5
SELECTED_STEP = 2220
CONTEXT = 1024
TARGET_PRESENTATIONS = 135_886_355
REPETITION_KEYS = (
    "distinct_1",
    "distinct_2",
    "distinct_3",
    "repeated_4gram_rate",
    "repeated_sentence_rate",
    "longest_repeated_token_span",
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def change(candidate: float, baseline: float) -> float:
    return 100.0 * (candidate / baseline - 1.0)


def historical(evaluations: dict[str, Any], name: str) -> dict[str, Any]:
    return evaluations["historical_post_selection_evaluations"][name]


def validate_generation_pair(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> None:
    require(len(baseline["entries"]) == 10, "Baseline generation suite is not 10 prompts")
    require(len(candidate["entries"]) == 10, "Candidate generation suite is not 10 prompts")
    require(
        baseline["generation_settings"] == GENERATION_SETTINGS,
        "Baseline generation settings changed",
    )
    require(
        candidate["generation_settings"] == GENERATION_SETTINGS,
        "Candidate generation settings changed",
    )
    require(
        baseline["sample_seed"] == candidate["sample_seed"],
        "Generation base seeds differ",
    )
    require(
        [entry["prompt"] for entry in baseline["entries"]]
        == [entry["prompt"] for entry in candidate["entries"]],
        "Generation prompt order differs",
    )
    for index, (left, right) in enumerate(
        zip(baseline["entries"], candidate["entries"], strict=True)
    ):
        expected_seed = baseline["sample_seed"] + index
        require(left["sample_seed"] == expected_seed, "Baseline per-prompt seed changed")
        require(right["sample_seed"] == expected_seed, "Candidate per-prompt seed changed")


def validate_controls(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
) -> None:
    require(baseline["seed"] == candidate["seed"] == 1337, "Initialization seeds differ")
    require(baseline["tokenizer"] == candidate["tokenizer"], "Tokenizer controls differ")
    require(baseline["data"] == candidate["data"], "Data controls differ")
    require(baseline["evaluation"] == candidate["evaluation"], "Evaluation controls differ")
    for key, value in baseline["training"].items():
        if key != "checkpoint_dir":
            require(candidate["training"].get(key) == value, f"Training control differs: {key}")
    capacity_fields = {
        "hidden_size",
        "num_layers",
        "num_attention_heads",
        "intermediate_size",
        "expected_parameter_count",
    }
    for key, value in baseline["model"].items():
        if key not in capacity_fields:
            require(candidate["model"].get(key) == value, f"Model control differs: {key}")
    require(
        baseline_manifest["tokenizer_hash"] == candidate_manifest["tokenizer_hash"],
        "Manifest tokenizer hashes differ",
    )
    for key in ("corpus_sha256", "split_manifest_sha256", "packed_artifacts"):
        require(
            baseline_manifest[key] == candidate_manifest[key],
            f"Manifest data provenance differs: {key}",
        )
    require(
        baseline_manifest["evaluation_protocol"]
        == candidate_manifest["evaluation_protocol"],
        "Evaluation protocol provenance differs",
    )
    require(
        sha256_file(Path(baseline_manifest["config_path"]))
        == baseline_manifest["config_sha256"],
        "Baseline config hash changed",
    )
    require(
        sha256_file(Path(candidate_manifest["config_path"]))
        == candidate_manifest["config_sha256"],
        "Candidate config hash changed",
    )


def validate_run(
    *,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    evaluations: dict[str, Any],
    expected_run_id: str,
    expected_parameters: int,
) -> None:
    require(summary["run_id"] == expected_run_id, f"Unexpected summary run ID: {summary['run_id']}")
    require(manifest["run_id"] == expected_run_id, f"Unexpected manifest run ID: {manifest['run_id']}")
    require(summary["status"] == "completed", f"Run {expected_run_id} is not completed")
    require(summary["model_parameter_count"] == expected_parameters, "Parameter count changed")
    require(manifest["model_parameter_count"] == expected_parameters, "Manifest parameter count changed")
    require(manifest["fresh_random_initialization"] is True, "Run did not use fresh initialization")
    require(summary["best_checkpoint_epoch"] == SELECTED_EPOCH, "Unexpected best epoch")
    require(manifest["best_checkpoint_epoch"] == SELECTED_EPOCH, "Manifest best epoch differs")
    require(
        evaluations["checkpoint_selection_complete_before_test_access"] is True,
        "Checkpoint selection was not sealed before test access",
    )
    require(
        evaluations["selection_metric"] == "Data30M validation loss only",
        "Checkpoint selection was not validation-only",
    )
    require(evaluations["selected_checkpoint_epoch"] == SELECTED_EPOCH, "Selected epoch changed")
    require(evaluations["selected_checkpoint_step"] == SELECTED_STEP, "Selected step changed")
    require(
        evaluations["selected_checkpoint_sha256"] == summary["best_checkpoint_sha256"],
        "Selected checkpoint hash differs from summary",
    )
    require(
        summary["selected_checkpoint_validation"]["optimizer_step"] == SELECTED_STEP,
        "Summary selected step changed",
    )
    require(
        summary["total_valid_training_targets_processed"] == TARGET_PRESENTATIONS,
        "Training exposure changed",
    )
    require(
        manifest["exposure"]["valid_target_presentations"] == TARGET_PRESENTATIONS,
        "Manifest training exposure changed",
    )
    require(
        evaluations["primary_data30m_test"]["metrics"] == summary["test_metrics"],
        "Sealed Data30M test metrics differ from summary",
    )


def validate_narrative(
    narrative: dict[str, Any],
    protocol: dict[str, Any],
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> None:
    require(narrative["post_checkpoint_selection"] is True, "Narrative suite is not post-selection")
    require(narrative["used_for_checkpoint_selection"] is False, "Narrative suite affected selection")
    require(narrative["generation_settings"] == GENERATION_SETTINGS, "Narrative settings changed")
    require(narrative["sample_seed"] == 11337, "Narrative base seed changed")
    require(narrative["protocol_sha256"] == sha256_file(EXPERIMENT_DIR / "narrative-state-protocol.json"), "Narrative protocol hash changed")
    require(narrative["protocol"]["name"] == protocol["name"], "Narrative protocol name differs")
    require(narrative["protocol"]["tokenizer_sha256"] == protocol["tokenizer_sha256"], "Narrative tokenizer hash differs")
    require(set(narrative["models"]) == {BASELINE_LABEL, CANDIDATE_LABEL}, "Narrative model labels differ")
    expected_ids = [entry["id"] for entry in protocol["entries"]]
    for label, checkpoint_hash in (
        (BASELINE_LABEL, baseline_summary["best_checkpoint_sha256"]),
        (CANDIDATE_LABEL, candidate_summary["best_checkpoint_sha256"]),
    ):
        model = narrative["models"][label]
        require(model["checkpoint_sha256"] == checkpoint_hash, f"Narrative checkpoint hash differs: {label}")
        require(model["epoch"] == SELECTED_EPOCH, f"Narrative epoch differs: {label}")
        require(model["optimizer_step"] == SELECTED_STEP, f"Narrative step differs: {label}")
        require(model["max_seq_len"] == CONTEXT, f"Narrative context differs: {label}")
        require([entry["id"] for entry in model["entries"]] == expected_ids, f"Narrative prompt order differs: {label}")
        for index, (entry, protocol_entry) in enumerate(
            zip(model["entries"], protocol["entries"], strict=True)
        ):
            require(entry["sample_seed"] == narrative["sample_seed"] + index, f"Narrative seed differs: {label}")
            for key in (
                "facts",
                "prompt_token_count",
                "fact_prefix_token_count",
                "tokens_after_fact_prefix",
            ):
                require(
                    entry[key] == protocol_entry[key],
                    f"Narrative protocol entry differs: {label}/{entry['id']}/{key}",
                )
            for mode in ("greedy", "sampled"):
                visibility = entry["fact_visibility"][mode]
                max_new_tokens = GENERATION_SETTINGS[f"{mode}_max_new_tokens"]
                require(visibility["control_token_count"] == 2, "Narrative control-token count changed")
                require(visibility["max_new_tokens"] == max_new_tokens, f"Narrative token limit differs: {mode}")
                require(
                    visibility["maximum_sequence_tokens"]
                    == entry["prompt_token_count"] + 2 + max_new_tokens,
                    f"Narrative visibility arithmetic differs: {label}/{entry['id']}/{mode}",
                )
                require(visibility["fact_prefix_visible_at_generation_start"] is True, f"Facts are not initially visible for {label}/{entry['id']}")
                require(visibility["fact_prefix_visible_for_full_generation"] is True, f"Facts leave context for {label}/{entry['id']}/{mode}")


def aggregate(entries: list[dict[str, Any]], label: str, mode: str) -> dict[str, float]:
    return {
        key: mean(entry["repetition"][label][mode][key] for entry in entries)
        for key in REPETITION_KEYS
    }


def write_markdown(name: str, text: str) -> None:
    for directory in (DERIVED_RUN, EXPERIMENT_DIR):
        (directory / name).write_text(text, encoding="utf-8")


def build_report() -> dict[str, Any]:
    DERIVED_RUN.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path("data/tokenizer/tokenizer.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    baseline_summary = load(BASELINE_RUN / "summary.json")
    candidate_summary = load(RAW_CANDIDATE_RUN / "summary.json")
    baseline_manifest = load(BASELINE_RUN / "manifest.json")
    candidate_manifest = load(RAW_CANDIDATE_RUN / "manifest.json")
    baseline_evaluations = load(BASELINE_RUN / "sealed-evaluations.json")
    candidate_evaluations = load(RAW_CANDIDATE_RUN / "sealed-evaluations.json")
    baseline_config = load_yaml(BASELINE_RUN / "config.yaml")
    candidate_config = load_yaml(RAW_CANDIDATE_RUN / "config.yaml")
    protocol_path = EXPERIMENT_DIR / "narrative-state-protocol.json"
    protocol = load(protocol_path)
    preflight = load(EXPERIMENT_DIR / "preflight.json")
    narrative = load(NARRATIVE_PATH)

    validate_run(
        summary=baseline_summary,
        manifest=baseline_manifest,
        evaluations=baseline_evaluations,
        expected_run_id="fictionpulper-15m-data30m-compute5",
        expected_parameters=BASELINE_PARAMETERS,
    )
    validate_run(
        summary=candidate_summary,
        manifest=candidate_manifest,
        evaluations=candidate_evaluations,
        expected_run_id="smoke-5m-20261007-212756",
        expected_parameters=CANDIDATE_PARAMETERS,
    )
    validate_controls(baseline_config, candidate_config, baseline_manifest, candidate_manifest)
    require(sha256_file(tokenizer_path) == baseline_manifest["tokenizer_hash"], "Tokenizer file hash changed")
    require(preflight["tokenizer_v1_sha256"] == baseline_manifest["tokenizer_hash"], "Preflight tokenizer hash differs")
    require(preflight["corpus_sha256"] == baseline_manifest["corpus_sha256"], "Preflight corpus hash differs")
    require(preflight["split_manifest_sha256"] == baseline_manifest["split_manifest_sha256"], "Preflight split hash differs")
    require(preflight["configuration_sha256"] == candidate_manifest["config_sha256"], "Preflight config hash differs")
    require(preflight["evaluation_protocol_sha256"] == baseline_manifest["evaluation_protocol"]["sha256"], "Preflight evaluation protocol hash differs")
    require(preflight["narrative_state_protocol_sha256"] == sha256_file(protocol_path), "Preflight narrative protocol hash differs")
    require(
        sha256_file(Path(baseline_manifest["evaluation_protocol"]["path"]))
        == baseline_manifest["evaluation_protocol"]["sha256"],
        "Evaluation protocol file hash changed",
    )
    require(candidate_config["model"]["max_seq_len"] == CONTEXT, "Candidate context changed")
    require(baseline_config["model"]["max_seq_len"] == CONTEXT, "Baseline context changed")
    validate_narrative(narrative, protocol, baseline_summary, candidate_summary)

    baseline_common = historical(baseline_evaluations, "data10m_test_data30m_train_disjoint")
    candidate_common = historical(candidate_evaluations, "data10m_test_data30m_train_disjoint")
    require(baseline_common["document_count"] == candidate_common["document_count"] == 95, "Common test is not 95 documents")
    require(baseline_common["packed_sha256"] == candidate_common["packed_sha256"], "Common test data hash differs")
    baseline_legacy_validation = historical(baseline_evaluations, "legacy_seed_validation_leakage_clean")
    candidate_legacy_validation = historical(candidate_evaluations, "legacy_seed_validation_leakage_clean")
    baseline_legacy_test = historical(baseline_evaluations, "legacy_seed_test")
    candidate_legacy_test = historical(candidate_evaluations, "legacy_seed_test")

    model_rows = {
        "baseline_15m": {
            "run_id": baseline_summary["run_id"],
            "parameters": BASELINE_PARAMETERS,
            "context": CONTEXT,
            "target_presentations": TARGET_PRESENTATIONS,
            "target_presentations_per_parameter": TARGET_PRESENTATIONS / BASELINE_PARAMETERS,
            "best_epoch": baseline_summary["best_validation_epoch"],
            "best_step": baseline_summary["selected_checkpoint_validation"]["optimizer_step"],
            "best_validation_loss": baseline_summary["best_validation_loss"],
            "data30m_test": baseline_evaluations["primary_data30m_test"]["metrics"],
            "common_held_out_95_document_test": baseline_common["metrics"],
            "legacy_diagnostics_partially_contaminated": {
                "clean_validation_name_is_historical_only": baseline_legacy_validation,
                "legacy_test": baseline_legacy_test,
            },
            "training_seconds": baseline_summary["training_seconds"],
            "valid_targets_per_second": baseline_summary["average_valid_tokens_per_second"],
            "peak_vram_gib": baseline_summary["peak_gpu_memory_gb"],
        },
        "candidate_50m": {
            "intended_run_id": "fictionpulper-50m-data30m-v1",
            "accidental_raw_run_id": candidate_summary["run_id"],
            "raw_run_path": str(RAW_CANDIDATE_RUN),
            "parameters": CANDIDATE_PARAMETERS,
            "context": CONTEXT,
            "target_presentations": TARGET_PRESENTATIONS,
            "target_presentations_per_parameter": TARGET_PRESENTATIONS / CANDIDATE_PARAMETERS,
            "best_epoch": candidate_summary["best_validation_epoch"],
            "best_step": candidate_summary["selected_checkpoint_validation"]["optimizer_step"],
            "best_validation_loss": candidate_summary["best_validation_loss"],
            "data30m_test": candidate_evaluations["primary_data30m_test"]["metrics"],
            "common_held_out_95_document_test": candidate_common["metrics"],
            "legacy_diagnostics_partially_contaminated": {
                "clean_validation_name_is_historical_only": candidate_legacy_validation,
                "legacy_test": candidate_legacy_test,
            },
            "training_seconds": candidate_summary["training_seconds"],
            "valid_targets_per_second": candidate_summary["average_valid_tokens_per_second"],
            "peak_vram_gib": candidate_summary["peak_gpu_memory_gb"],
        },
    }
    quantitative = {
        "run_id": "fictionpulper-50m-data30m-v1",
        "raw_training_run_disclosure": "The 50M training artifacts were accidentally written under smoke-5m-20261007-212756; reports preserve that raw run ID and use the intended ID only for derived outputs.",
        "experimental_variable": "model capacity from 15047040 to 50348544 parameters",
        "parameter_multiplier": CANDIDATE_PARAMETERS / BASELINE_PARAMETERS,
        "same_seed_note": "Seed 1337 means the same initialization seed, not the same weights; model shapes differ.",
        "selection_note": "Checkpoint selection used Data30M validation loss only. Test and generation diagnostics are post-selection and were not used to select checkpoints.",
        "test_note": "Sealed test results are read from existing artifacts and are not rerun by this reporter.",
        "legacy_note": "Legacy diagnostics are historical continuity measurements and are partially contaminated by records seen in Data30M training.",
        "provenance": {
            "tokenizer_sha256": baseline_manifest["tokenizer_hash"],
            "corpus_sha256": baseline_manifest["corpus_sha256"],
            "split_manifest_sha256": baseline_manifest["split_manifest_sha256"],
            "evaluation_protocol": baseline_manifest["evaluation_protocol"],
            "narrative_protocol_path": str(protocol_path),
            "narrative_protocol_sha256": sha256_file(protocol_path),
            "baseline_checkpoint_sha256": baseline_summary["best_checkpoint_sha256"],
            "candidate_checkpoint_sha256": candidate_summary["best_checkpoint_sha256"],
            "baseline_config_sha256": baseline_manifest["config_sha256"],
            "candidate_config_sha256": candidate_manifest["config_sha256"],
        },
        "models": model_rows,
        "relative_changes_percent": {
            "best_validation_loss": change(candidate_summary["best_validation_loss"], baseline_summary["best_validation_loss"]),
            "data30m_test_loss": change(candidate_summary["test_loss"], baseline_summary["test_loss"]),
            "data30m_test_perplexity": change(candidate_summary["test_perplexity"], baseline_summary["test_perplexity"]),
            "data30m_test_accuracy": change(candidate_summary["test_next_token_accuracy"], baseline_summary["test_next_token_accuracy"]),
            "common_held_out_loss": change(candidate_common["metrics"]["loss"], baseline_common["metrics"]["loss"]),
            "common_held_out_perplexity": change(candidate_common["metrics"]["perplexity"], baseline_common["metrics"]["perplexity"]),
            "common_held_out_accuracy": change(candidate_common["metrics"]["next_token_accuracy"], baseline_common["metrics"]["next_token_accuracy"]),
            "runtime": change(candidate_summary["training_seconds"], baseline_summary["training_seconds"]),
            "throughput": change(candidate_summary["average_valid_tokens_per_second"], baseline_summary["average_valid_tokens_per_second"]),
            "peak_vram": change(candidate_summary["peak_gpu_memory_gb"], baseline_summary["peak_gpu_memory_gb"]),
        },
    }
    write_json_atomic(DERIVED_RUN / "quantitative-comparison.json", quantitative)
    rows = [
        ("Parameters", BASELINE_PARAMETERS, CANDIDATE_PARAMETERS),
        ("Parameter multiplier", 1.0, quantitative["parameter_multiplier"]),
        ("Context", CONTEXT, CONTEXT),
        ("Target presentations", TARGET_PRESENTATIONS, TARGET_PRESENTATIONS),
        ("Presentations / parameter", model_rows["baseline_15m"]["target_presentations_per_parameter"], model_rows["candidate_50m"]["target_presentations_per_parameter"]),
        ("Selected epoch", SELECTED_EPOCH, SELECTED_EPOCH),
        ("Selected step", SELECTED_STEP, SELECTED_STEP),
        ("Best validation loss", baseline_summary["best_validation_loss"], candidate_summary["best_validation_loss"]),
        ("Data30M test loss", baseline_summary["test_loss"], candidate_summary["test_loss"]),
        ("Data30M test PPL", baseline_summary["test_perplexity"], candidate_summary["test_perplexity"]),
        ("Data30M test accuracy", baseline_summary["test_next_token_accuracy"], candidate_summary["test_next_token_accuracy"]),
        ("Common held-out 95-doc loss", baseline_common["metrics"]["loss"], candidate_common["metrics"]["loss"]),
        ("Common held-out 95-doc PPL", baseline_common["metrics"]["perplexity"], candidate_common["metrics"]["perplexity"]),
        ("Common held-out 95-doc accuracy", baseline_common["metrics"]["next_token_accuracy"], candidate_common["metrics"]["next_token_accuracy"]),
        ("Legacy clean-named validation loss (contaminated)", baseline_legacy_validation["metrics"]["loss"], candidate_legacy_validation["metrics"]["loss"]),
        ("Legacy test loss (contaminated)", baseline_legacy_test["metrics"]["loss"], candidate_legacy_test["metrics"]["loss"]),
        ("Runtime seconds", baseline_summary["training_seconds"], candidate_summary["training_seconds"]),
        ("Valid targets / second", baseline_summary["average_valid_tokens_per_second"], candidate_summary["average_valid_tokens_per_second"]),
        ("Peak VRAM GiB", baseline_summary["peak_gpu_memory_gb"], candidate_summary["peak_gpu_memory_gb"]),
    ]
    quantitative_md = "\n".join([
        "# Quantitative Comparison",
        "",
        "The 50M training artifacts retain the accidental raw run ID `smoke-5m-20261007-212756`; `fictionpulper-50m-data30m-v1` is the intended experiment and derived-report ID.",
        "",
        "Both models use seed 1337 for fresh random initialization. This is the same initialization seed, not the same weights, because model shapes differ. Selection used Data30M validation loss only. Existing sealed test results were not rerun.",
        "",
        "| Metric | 15M Compute5 | 50M Capacity |",
        "|---|---:|---:|",
        *[f"| {label} | {left} | {right} |" for label, left, right in rows],
        "",
        "Legacy diagnostics are labeled historical continuity measurements and are partially contaminated by records seen in Data30M training.",
        "",
    ])
    write_markdown("quantitative-comparison.md", quantitative_md)

    baseline_generations = load(BASELINE_RUN / "generations/best-validation-final.json")
    candidate_generations = load(RAW_CANDIDATE_RUN / "generations/best-validation-final.json")
    validate_generation_pair(baseline_generations, candidate_generations)
    require((baseline_generations["epoch"], baseline_generations["optimizer_step"]) == (SELECTED_EPOCH, SELECTED_STEP), "Baseline generations use the wrong checkpoint")
    require((candidate_generations["epoch"], candidate_generations["optimizer_step"]) == (SELECTED_EPOCH, SELECTED_STEP), "Candidate generations use the wrong checkpoint")
    generation_entries = []
    for left, right in zip(baseline_generations["entries"], candidate_generations["entries"], strict=True):
        generation_entries.append({
            "prompt": left["prompt"],
            "sample_seed": left["sample_seed"],
            "baseline_greedy": left["greedy"],
            "candidate_greedy": right["greedy"],
            "baseline_sampled": left["sampled"],
            "candidate_sampled": right["sampled"],
        })
    generation = {
        "raw_candidate_run_id": candidate_summary["run_id"],
        "settings": {**GENERATION_SETTINGS, "sample_seed": baseline_generations["sample_seed"], "effective_seed_policy": "sample_seed + prompt_index"},
        "baseline_checkpoint": {"epoch": SELECTED_EPOCH, "step": SELECTED_STEP, "sha256": baseline_summary["best_checkpoint_sha256"]},
        "candidate_checkpoint": {"epoch": SELECTED_EPOCH, "step": SELECTED_STEP, "sha256": candidate_summary["best_checkpoint_sha256"]},
        "entries": generation_entries,
    }
    write_json_atomic(DERIVED_RUN / "generation-comparison.json", generation)
    generation_lines = ["# Generation Comparison", "", "Post-selection outputs from the validation-selected epoch-5, step-2220 checkpoints. No qualitative scores are assigned.", ""]
    for index, entry in enumerate(generation_entries, start=1):
        generation_lines.extend([f"## Prompt {index}", "", f"> {entry['prompt']}", "", "### 15M Greedy", "", entry["baseline_greedy"], "", "### 50M Greedy", "", entry["candidate_greedy"], "", "### 15M Sampled", "", entry["baseline_sampled"], "", "### 50M Sampled", "", entry["candidate_sampled"], ""])
    write_markdown("generation-comparison.md", "\n".join(generation_lines))

    original_repetition_entries = []
    for entry in generation_entries:
        original_repetition_entries.append({
            "prompt": entry["prompt"],
            "sample_seed": entry["sample_seed"],
            "repetition": {
                "baseline": {mode: repetition_metrics(tokenizer, entry[f"baseline_{mode}"]) for mode in ("greedy", "sampled")},
                "candidate": {mode: repetition_metrics(tokenizer, entry[f"candidate_{mode}"]) for mode in ("greedy", "sampled")},
            },
        })
    narrative_repetition_entries = []
    baseline_narrative = narrative["models"][BASELINE_LABEL]["entries"]
    candidate_narrative = narrative["models"][CANDIDATE_LABEL]["entries"]
    for left, right in zip(baseline_narrative, candidate_narrative, strict=True):
        narrative_repetition_entries.append({
            "id": left["id"],
            "sample_seed": left["sample_seed"],
            "repetition": {
                "baseline": left["repetition"],
                "candidate": right["repetition"],
            },
        })
    repetition = {
        "definition": "Tokenizer-v1 token n-grams over generated continuation only; prompts are excluded",
        "original_10_prompt_suite": {
            "entries": original_repetition_entries,
            "aggregate": {label: {mode: aggregate(original_repetition_entries, label, mode) for mode in ("greedy", "sampled")} for label in ("baseline", "candidate")},
        },
        "narrative_state_3_prompt_suite": {
            "entries": narrative_repetition_entries,
            "aggregate": {label: {mode: aggregate(narrative_repetition_entries, label, mode) for mode in ("greedy", "sampled")} for label in ("baseline", "candidate")},
        },
    }
    write_json_atomic(DERIVED_RUN / "repetition-comparison.json", repetition)
    repetition_lines = ["# Repetition Comparison", "", "Metrics use generated continuation only; prompt tokens are excluded.", "", "| Suite | Model | Mode | Distinct-1 | Distinct-2 | Distinct-3 | Repeated 4-gram | Repeated sentence | Longest repeated span |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for suite_name, suite in (("Original 10", repetition["original_10_prompt_suite"]), ("Narrative-state 3", repetition["narrative_state_3_prompt_suite"])):
        for label in ("baseline", "candidate"):
            for mode in ("greedy", "sampled"):
                values = suite["aggregate"][label][mode]
                repetition_lines.append(f"| {suite_name} | {label} | {mode} | {values['distinct_1']} | {values['distinct_2']} | {values['distinct_3']} | {values['repeated_4gram_rate']} | {values['repeated_sentence_rate']} | {values['longest_repeated_token_span']} |")
    repetition_lines.append("")
    write_markdown("repetition-comparison.md", "\n".join(repetition_lines))

    narrative_lines = ["# Narrative-State Comparison", "", "This is a post-selection capacity diagnostic. Every fact prefix remains visible for the full configured greedy 128-token and sampled 256-token generations, including the two control tokens. Facts are disclosed but not manually scored.", ""]
    for left, right in zip(baseline_narrative, candidate_narrative, strict=True):
        narrative_lines.extend([f"## {left['id']}", "", f"Facts: `{json.dumps(left['facts'], sort_keys=True)}`", "", f"Prompt tokens: {left['prompt_token_count']}; sampled maximum with controls: {left['fact_visibility']['sampled']['maximum_sequence_tokens']} / {CONTEXT}.", "", "### 15M Greedy", "", left["greedy"], "", "### 50M Greedy", "", right["greedy"], "", "### 15M Sampled", "", left["sampled"], "", "### 50M Sampled", "", right["sampled"], ""])
    write_markdown("narrative-state-comparison.md", "\n".join(narrative_lines))

    epoch_metrics = candidate_summary["metrics"][1:]
    validation_losses = [item["validation_loss"] for item in epoch_metrics]
    learning_curve = {
        "classification": "improving but effectively plateaued by epoch 5",
        "selected_by": "minimum complete-epoch Data30M validation loss",
        "selected_epoch": candidate_summary["best_validation_epoch"],
        "selected_step": candidate_summary["selected_checkpoint_validation"]["optimizer_step"],
        "validation_minimum": min(validation_losses),
        "epochs": [{"epoch": item["epoch"], "optimizer_step": item["optimizer_step"], "training_loss": item["training_loss"], "validation_loss": item["validation_loss"], "train_validation_gap": item["validation_loss"] - item["training_loss"]} for item in epoch_metrics],
        "validation_loss_change_by_epoch": {f"epoch_{index}_to_{index + 1}": validation_losses[index] - validation_losses[index - 1] for index in range(1, len(validation_losses))},
    }
    write_json_atomic(DERIVED_RUN / "learning-curve-assessment.json", learning_curve)
    return quantitative


def main() -> None:
    print(json.dumps(build_report(), indent=2))


if __name__ == "__main__":
    main()
