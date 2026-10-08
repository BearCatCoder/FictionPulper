"""Post-selection evaluation for the locked 15M narrative curriculum run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from src.context_diagnostics import repetition_metrics
from src.model import FictionPulperLM, ModelConfig
from src.train import (
    FIXED_PROMPTS,
    GENERATION_SETTINGS,
    GENRE_PROMPTS,
    benchmark_excluded_ids,
    evaluate,
    generate,
)
from src.train_tokenizer import sha256_file, write_json_atomic


PROTOCOL_PATH = Path(
    "experiments/fictionpulper-15m-data30m-narrative-v1/post-selection-protocol.json"
)
RUN_DIR = Path("runs/fictionpulper-15m-data30m-narrative-v1/post-selection")
EXPERIMENT_DIR = Path("experiments/fictionpulper-15m-data30m-narrative-v1")
PROMPTS = [*FIXED_PROMPTS, *GENRE_PROMPTS.values()]
REPETITION_KEYS = (
    "distinct_1", "distinct_2", "distinct_3", "repeated_4gram_rate",
    "longest_repeated_token_span", "repeated_sentence_rate",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_protocol(protocol: dict[str, Any]) -> None:
    require(protocol["selection_metric"] == "Data30M validation loss only", "Selection metric drifted")
    require(protocol["generation"]["settings"] == GENERATION_SETTINGS, "Generation settings drifted")
    require(protocol["generation"]["sample_seed"] == 11337, "Historical generation seed drifted")
    require(len(PROMPTS) == 10, "Historical prompt suite is not exactly 10 prompts")
    for key in ("sealed_evaluation_protocol", "narrative_state_protocol"):
        item = protocol[key]
        require(sha256_file(Path(item["path"])) == item["sha256"], f"{key} hash changed")
    curriculum_manifest = load_json(Path("data/narrative_curriculum_v1/manifest.json"))
    for name in ("curriculum_test", "curriculum_generalization_holdout"):
        item = protocol["held_out"][name]
        for path_key, hash_key in (("jsonl_path", "jsonl_sha256"), ("state_path", "state_sha256")):
            filename = Path(item[path_key]).name
            require(
                curriculum_manifest["files"][filename]["sha256"] == item[hash_key],
                f"Locked curriculum manifest hash differs for {filename}",
            )


def verify_selection_gate(
    protocol: dict[str, Any],
    *,
    load_checkpoint: Callable[[Path], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Verify selection is final. This function must run before any held-out loader."""
    candidate = protocol["candidate"]
    summary = load_json(Path(candidate["summary_path"]))
    require(summary.get("status") == protocol["required_training_status"], "Training is not complete")
    require(summary.get("selection_metric") == protocol["selection_metric"], "Summary selection metric changed")
    require(summary.get("curriculum_validation_used_for_selection") is False, "Curriculum validation affected selection")
    require(summary.get("optimizer_steps") == candidate["expected_steps"], "Locked training did not complete")
    manifest = load_json(Path(candidate["manifest_path"]))
    require(manifest.get("experiment_id") == protocol["experiment_id"], "Training manifest experiment ID changed")
    require(manifest.get("completed_at") == summary.get("completed_at"), "Training completion fields differ")
    require(manifest.get("schedule_content_sha256") == summary.get("schedule_content_sha256"), "Summary schedule differs from manifest")
    eligible = [item for item in summary.get("metrics", []) if item.get("selection_eligible")]
    require(bool(eligible), "Summary has no selection-eligible checkpoints")
    expected_best = min(eligible, key=lambda item: item["data30m_validation"]["loss"])
    require(summary.get("best_checkpoint_step") == expected_best["step"], "Best checkpoint was not selected by Data30M validation")
    checkpoint_path = Path(candidate["checkpoint_path"])
    require(checkpoint_path.is_file(), "Best checkpoint is missing")
    checkpoint_hash = sha256_file(checkpoint_path)
    require(summary.get("best_checkpoint_sha256") == checkpoint_hash, "Best checkpoint hash differs from summary")
    checkpoint = load_checkpoint(checkpoint_path)
    require(checkpoint.get("step") == summary["best_checkpoint_step"], "Checkpoint step differs from summary")
    require(checkpoint.get("selection_metric") == protocol["selection_metric"], "Checkpoint selection metric changed")
    require(checkpoint.get("tokenizer_hash") == candidate["tokenizer_sha256"], "Checkpoint tokenizer hash changed")
    require(checkpoint.get("schedule_sha256") == summary.get("schedule_content_sha256"), "Checkpoint schedule hash changed")
    return summary, checkpoint, checkpoint_path


def run_after_gate(
    protocol: dict[str, Any],
    gate: Callable[[], tuple[dict[str, Any], dict[str, Any], Path]],
    held_out_evaluator: Callable[[dict[str, Any], dict[str, Any], Path], dict[str, Any]],
) -> dict[str, Any]:
    """Small injectable orchestration boundary used to prove the held-out access gate."""
    summary, checkpoint, checkpoint_path = gate()
    return held_out_evaluator(summary, checkpoint, checkpoint_path)


class CurriculumDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, records: list[dict[str, Any]], tokenizer: Tokenizer, sequence_length: int):
        self.sequence_length = sequence_length
        self.pad_id = tokenizer.token_to_id("<|pad|>")
        story = tokenizer.token_to_id("<|story|>")
        bos = tokenizer.token_to_id("<|bos|>")
        eos = tokenizer.token_to_id("<|eos|>")
        require(None not in (self.pad_id, story, bos, eos), "Tokenizer lacks curriculum control tokens")
        self.samples: list[list[int]] = []
        for record in records:
            text = str(record["text"])
            require(text.startswith(str(record["title"]) + "\n\n"), f"Noncanonical curriculum record {record['id']}")
            control = record.get("genre_control_token")
            control_ids = []
            if control is not None:
                control_id = tokenizer.token_to_id(str(control))
                require(control_id is not None, f"Unknown genre control token for {record['id']}")
                control_ids.append(int(control_id))
            ids = [int(story), *control_ids, int(bos), *tokenizer.encode(text).ids, int(eos)]
            for start in range(0, len(ids) - 1, sequence_length):
                self.samples.append(ids[start : start + sequence_length + 1])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.samples[index]
        input_ids = values[:-1]
        labels = values[1:]
        padding = self.sequence_length - len(input_ids)
        return (
            torch.tensor(input_ids + [int(self.pad_id)] * padding, dtype=torch.long),
            torch.tensor(labels + [-100] * padding, dtype=torch.long),
        )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def verify_file(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    require(observed == expected, f"{label} hash changed")
    return observed


def load_curriculum_split(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    record_path, state_path = Path(spec["jsonl_path"]), Path(spec["state_path"])
    verify_file(record_path, spec["jsonl_sha256"], "Curriculum JSONL")
    verify_file(state_path, spec["state_sha256"], "Curriculum state sidecar")
    records = load_jsonl(record_path)
    sidecars = load_jsonl(state_path)
    require(len(records) == len(sidecars) == spec["expected_documents"], "Curriculum document count changed")
    by_id = {item["id"]: item for item in sidecars}
    require(set(by_id) == {item["id"] for item in records}, "Curriculum record/sidecar IDs differ")
    return records, by_id


@torch.no_grad()
def greedy_ids(
    model: FictionPulperLM, prefix_ids: list[int], count: int, device: torch.device
) -> list[int]:
    context = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    result: list[int] = []
    model.eval()
    for _ in range(count):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(context[:, -model.config.max_seq_len :])
        token = int(logits[0, -1].argmax().item())
        result.append(token)
        context = torch.cat((context, torch.tensor([[token]], device=device)), dim=1)
    return result


def aggregate_recall(trials: list[dict[str, Any]]) -> dict[str, Any]:
    def result(items: Iterable[dict[str, Any]]) -> dict[str, int | float]:
        values = list(items)
        successes = sum(bool(item["exact_match"]) for item in values)
        return {"correct": successes, "total": len(values), "accuracy": successes / len(values) if values else 0.0}

    output: dict[str, Any] = {"aggregate": result(trials)}
    for field in ("state_type", "distance_bucket", "difficulty", "genre"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trial in trials:
            groups[str(trial[field])].append(trial)
        output[f"by_{field}"] = {key: result(groups[key]) for key in sorted(groups)}
    return output


def evaluate_state_recall(
    model: FictionPulperLM,
    tokenizer: Tokenizer,
    records: list[dict[str, Any]],
    sidecars: dict[str, dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    story, bos = tokenizer.token_to_id("<|story|>"), tokenizer.token_to_id("<|bos|>")
    if story is None or bos is None:
        raise RuntimeError("Tokenizer lacks generation controls")
    trials = []
    for record in records:
        sidecar = sidecars[record["id"]]
        control = record.get("genre_control_token")
        control_ids = []
        if control is not None:
            control_id = tokenizer.token_to_id(str(control))
            require(control_id is not None, f"Unknown genre control token for {record['id']}")
            control_ids.append(int(control_id))
        for anchor in sidecar["anchors"]:
            start = record["text"].find(anchor["resolved_quote"])
            require(start >= 0, f"Resolution quote absent for {record['id']}")
            prompt = record["text"][:start]
            expected_ids = tokenizer.encode(anchor["resolved_quote"]).ids
            generated_ids = greedy_ids(
                model,
                [int(story), *control_ids, int(bos), *tokenizer.encode(prompt).ids],
                len(expected_ids),
                device,
            )
            matching_prefix = next((i for i, pair in enumerate(zip(generated_ids, expected_ids, strict=True)) if pair[0] != pair[1]), len(expected_ids))
            fact_checks = [
                {key: event[key] for key in ("entity", "attribute", "value")}
                for event in sidecar["events"]
                if event["evidence_quote"] == anchor["resolved_quote"]
            ]
            require(bool(fact_checks), f"Resolution has no metadata fact checks for {record['id']}")
            trials.append({
                "id": record["id"], "state_type": anchor["state_type"],
                "distance_bucket": record["distance_bucket"], "difficulty": record["difficulty"],
                "genre": record["genre"], "anchor_distance_tokens": anchor["resolved_token"] - anchor["introduced_token"],
                "expected_resolution": anchor["resolved_quote"],
                "expected_fact_checks": fact_checks,
                "expected_final_state": sidecar["expected_final_state"],
                "generated_resolution": tokenizer.decode(generated_ids, skip_special_tokens=True),
                "expected_token_count": len(expected_ids), "matching_prefix_tokens": matching_prefix,
                "exact_match": generated_ids == expected_ids,
            })
    return {
        "metric": "token_exact_resolution_continuation_recall",
        "teacher_forced_lm_metric": False,
        "definition": "At every sidecar anchor, greedily generate exactly the tokenized resolved_quote length from text ending at its start; success is complete token-ID equality.",
        "trials": trials,
        **aggregate_recall(trials),
    }


def load_model(payload: dict[str, Any], device: torch.device) -> FictionPulperLM:
    model = FictionPulperLM(ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def generation_suite(model: FictionPulperLM, tokenizer: Tokenizer, device: torch.device, prompts: list[str], seed: int) -> list[dict[str, Any]]:
    entries = []
    for index, prompt in enumerate(prompts):
        greedy = generate(model, tokenizer, prompt, max_new_tokens=GENERATION_SETTINGS["greedy_max_new_tokens"], device=device, use_bf16=True)
        sampled = generate(model, tokenizer, prompt, max_new_tokens=GENERATION_SETTINGS["sampled_max_new_tokens"], device=device, use_bf16=True, temperature=GENERATION_SETTINGS["temperature"], top_k=GENERATION_SETTINGS["top_k"], top_p=GENERATION_SETTINGS["top_p"], seed=seed + index)
        entries.append({
            "prompt": prompt, "sample_seed": seed + index, "greedy": greedy, "sampled": sampled,
            "repetition": {mode: repetition_metrics(tokenizer, text) for mode, text in (("greedy", greedy), ("sampled", sampled))},
        })
    return entries


def evaluate_packed(model: FictionPulperLM, tokenizer: Tokenizer, spec: dict[str, Any], device: torch.device, *, excluded_ids: set[str] | None = None) -> dict[str, Any]:
    from src.dataset import PackedStoryDataset

    path = Path(spec["packed_path"])
    index = path.with_suffix(".index.json") if "index_path" not in spec else Path(spec["index_path"])
    verify_file(path, spec["packed_sha256"], "Packed benchmark")
    verify_file(index, spec["index_sha256"], "Packed benchmark index")
    dataset = PackedStoryDataset(path, index, sequence_length=model.config.max_seq_len, pad_token_id=tokenizer.token_to_id("<|pad|>"), exclude_document_ids=excluded_ids)
    documents = len(dataset.documents) - len(excluded_ids or ())
    require(documents == spec["expected_documents"], "Benchmark document count changed")
    metrics = evaluate(model, dataset, batch_size=16, device=device, use_bf16=True)
    require(metrics["valid_tokens"] == spec["expected_valid_targets"], "Benchmark target count changed")
    require(metrics["allocated_slots"] == spec["expected_allocated_slots"], "Benchmark slot count changed")
    return {"classification": spec.get("classification"), "document_count": documents, "excluded_ids": sorted(excluded_ids or ()), "packed_sha256": spec["packed_sha256"], "index_sha256": spec["index_sha256"], "metrics": metrics}


def repetition_aggregate(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {mode: {key: mean(entry["repetition"][mode][key] for entry in entries) for key in REPETITION_KEYS} for mode in ("greedy", "sampled")}


def write_outputs(payloads: dict[str, Any], run_dir: Path, experiment_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, payload in payloads.items():
        path = run_dir / name
        write_json_atomic(path, payload)
        artifacts[name] = {"path": str(path), "sha256": sha256_file(path)}
    manifest = {"experiment_id": "fictionpulper-15m-data30m-narrative-v1", "artifacts": artifacts}
    write_json_atomic(run_dir / "artifact-manifest.json", manifest)
    evaluation = payloads["evaluation-results.json"]
    recall = payloads["curriculum-state-recall.json"]
    lines = [
        "# Post-Selection Evaluation",
        "",
        "Checkpoint selection used Data30M validation loss only. Every result below was produced after selection and was not used to choose a checkpoint.",
        "",
        "## Teacher-Forced LM Metrics",
        "",
        "| Benchmark | Loss | Perplexity | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for name, item in evaluation["metrics"].items():
        values = item["metrics"]
        lines.append(f"| {name} | {values['loss']} | {values['perplexity']} | {values['next_token_accuracy']} |")
    lines.extend([
        "",
        "## Exact State Recall",
        "",
        "This continuation metric is distinct from teacher-forced LM metrics. Success requires token-exact reproduction of the sidecar resolution from its held-out resolution boundary.",
        "",
        "| Split | Correct | Total | Accuracy |",
        "|---|---:|---:|---:|",
    ])
    for name, item in recall.items():
        values = item["aggregate"]
        lines.append(f"| {name} | {values['correct']} | {values['total']} | {values['accuracy']} |")
    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts["report.md"] = {"path": str(report_path), "sha256": sha256_file(report_path)}
    write_json_atomic(run_dir / "artifact-manifest.json", manifest)
    compact = {
        "status": "post_selection_evaluation_complete",
        "selection_metric": "Data30M validation loss only",
        "used_holdouts_for_checkpoint_selection": False,
        "run_artifact_manifest": str(run_dir / "artifact-manifest.json"),
        "run_artifact_manifest_sha256": sha256_file(run_dir / "artifact-manifest.json"),
        "checkpoint_sha256": payloads["evaluation-results.json"]["checkpoint_sha256"],
        "metrics": evaluation["metrics"],
        "state_recall": {name: value["aggregate"] for name, value in recall.items()},
    }
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(experiment_dir / "post-selection-record.json", compact)
    return compact


def evaluate_held_out(summary: dict[str, Any], checkpoint: dict[str, Any], checkpoint_path: Path, protocol: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "Post-selection evaluation requires CUDA BF16")
    device = torch.device("cuda")
    tokenizer_path = Path(protocol["candidate"]["tokenizer_path"])
    verify_file(tokenizer_path, protocol["candidate"]["tokenizer_sha256"], "Tokenizer")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = load_model(checkpoint, device)
    held = protocol["held_out"]

    # Access order is part of the protocol and intentionally follows the selection gate.
    metrics: dict[str, Any] = {"data30m_test": evaluate_packed(model, tokenizer, held["data30m_test"], device)}
    state_payloads = {}
    for name in ("curriculum_test", "curriculum_generalization_holdout"):
        records, sidecars = load_curriculum_split(held[name])
        dataset = CurriculumDataset(records, tokenizer, model.config.max_seq_len)
        metrics[name] = {"classification": "teacher_forced_language_model", "document_count": len(records), "metrics": evaluate(model, dataset, batch_size=16, device=device, use_bf16=True)}
        state_payloads[name] = evaluate_state_recall(model, tokenizer, records, sidecars, device)

    sealed = load_json(Path(protocol["sealed_evaluation_protocol"]["path"]))
    split = load_json(Path(sealed["data30m_split_path"]))
    benchmark_by_name = {item["name"]: item for item in sealed["benchmarks"]}
    for name in held["historical_benchmarks_in_order"]:
        spec = benchmark_by_name[name]
        exclusions = benchmark_excluded_ids(spec, split["assignments"])
        metrics[name] = evaluate_packed(model, tokenizer, spec, device, excluded_ids=exclusions)

    baseline_spec = protocol["baseline"]
    verify_file(Path(baseline_spec["seal_path"]), baseline_spec["seal_sha256"], "Compute5 seal")
    verify_file(Path(baseline_spec["checkpoint_path"]), baseline_spec["checkpoint_sha256"], "Compute5 checkpoint")
    baseline_payload = torch.load(Path(baseline_spec["checkpoint_path"]), map_location=device, weights_only=False)
    baseline_model = load_model(baseline_payload, device)
    seed = protocol["generation"]["sample_seed"]
    baseline_generations = generation_suite(baseline_model, tokenizer, device, PROMPTS, seed)
    candidate_generations = generation_suite(model, tokenizer, device, PROMPTS, seed)
    generation_comparison = {
        "post_checkpoint_selection": True, "used_for_checkpoint_selection": False,
        "prompts": PROMPTS, "sample_seed": seed, "generation_settings": GENERATION_SETTINGS,
        "baseline_checkpoint_sha256": baseline_spec["checkpoint_sha256"],
        "candidate_checkpoint_sha256": sha256_file(checkpoint_path),
        "models": {"compute5": baseline_generations, "narrative": candidate_generations},
    }

    narrative_protocol = load_json(Path(protocol["narrative_state_protocol"]["path"]))
    narrative_models = {}
    for label, selected_model, selected_hash in (
        ("compute5_context1024", baseline_model, baseline_spec["checkpoint_sha256"]),
        ("narrative_context1024", model, sha256_file(checkpoint_path)),
    ):
        entries = generation_suite(selected_model, tokenizer, device, [item["prompt"] for item in narrative_protocol["entries"]], seed)
        for output, definition in zip(entries, narrative_protocol["entries"], strict=True):
            output.update({key: definition[key] for key in ("id", "facts", "prompt_token_count", "fact_prefix_token_count", "tokens_after_fact_prefix")})
        narrative_models[label] = {"checkpoint_sha256": selected_hash, "entries": entries}
    narrative = {
        "protocol_path": protocol["narrative_state_protocol"]["path"],
        "protocol_sha256": protocol["narrative_state_protocol"]["sha256"],
        "post_checkpoint_selection": True, "used_for_checkpoint_selection": False,
        "scoring_status": "unscored_outputs_ready_for_separate_greedy_and_sampled_fact_scoring",
        "sample_seed": seed, "generation_settings": GENERATION_SETTINGS, "models": narrative_models,
    }
    repetition = {
        "definition": "Tokenizer-v1 metrics over generated continuations only; prompts excluded",
        "historical_10_prompt": {label: repetition_aggregate(entries) for label, entries in generation_comparison["models"].items()},
        "narrative_state": {label: repetition_aggregate(value["entries"]) for label, value in narrative_models.items()},
    }
    evaluation = {
        "status": "complete", "post_checkpoint_selection": True,
        "used_for_checkpoint_selection": False, "selection_metric": protocol["selection_metric"],
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": summary["best_checkpoint_step"], "metrics": metrics,
        "teacher_forced_note": "All loss/PPL/accuracy results are teacher-forced next-token metrics. They are distinct from curriculum continuation state recall.",
    }
    return write_outputs({
        "evaluation-results.json": evaluation,
        "curriculum-state-recall.json": state_payloads,
        "generation-comparison.json": generation_comparison,
        "narrative-state-comparison.json": narrative,
        "repetition-comparison.json": repetition,
    }, run_dir, EXPERIMENT_DIR)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    protocol = load_json(args.protocol)
    verify_protocol(protocol)
    gate = lambda: verify_selection_gate(protocol, load_checkpoint=lambda path: torch.load(path, map_location="cpu", weights_only=False))
    result = run_after_gate(protocol, gate, lambda summary, checkpoint, path: evaluate_held_out(summary, checkpoint, path, protocol, args.run_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
