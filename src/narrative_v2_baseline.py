"""Seal-bound forced-choice and teacher-forced Narrative Benchmark v2 evaluation."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.continuity_curriculum.common import (
    canonical_json,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
    write_jsonl,
)
from src.model import FictionPulperLM, ModelConfig
from src.narrative_v2_free_generation import load_protocol
from src.narrative_v2_loader import load_authored_split


EVALUATOR_VERSION = "narrative-v2-baseline-1"
TOKENIZER_SHA256 = "14d4abefa49a742dfdf62dbcb84223016ad6a9241362ac093d624161a00f7012"
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "compute5": {
        "model_id": "fictionpulper-15m-data30m-compute5",
        "checkpoint": "checkpoints/fictionpulper-15m-data30m-compute5/best-validation.pt",
        "checkpoint_sha256": "a9b53087a5c0dee6c2a8640999ac88d54cfb62a2d94833bc737f968f267c01fe",
        "seal": "experiments/fictionpulper-15m-data30m-compute5/seal.json",
        "seal_sha256": "10fcd7a62b819bd2af9a9fb4f5bf86df81558130f0e6f01af44c48491185db5c",
    },
    "50m": {
        "model_id": "fictionpulper-50m-data30m-v1",
        "checkpoint": "checkpoints/fictionpulper-50m-data30m-v1/best-validation.pt",
        "checkpoint_sha256": "8da266fb064719d3f38e78d672b2e5cec0f14551c37658210322b9f46cbecd01",
        "seal": "experiments/fictionpulper-50m-data30m-v1/seal.json",
        "seal_sha256": "ac9f0abf9c3c1c60f0ee02485918e86c8407e051c578e34cf1bfa7ae15451fb3",
    },
}
CONTROL_CONTEXTS = ("candidate_only", "fact_removed", "irrelevant_substituted")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_sealed_model(name: str, root: Path) -> dict[str, Any]:
    """Verify the selected checkpoint against its exact immutable seal identity."""
    require(name in MODEL_REGISTRY, f"unregistered baseline model: {name}")
    specification = MODEL_REGISTRY[name]
    seal_path = root / specification["seal"]
    checkpoint_path = root / specification["checkpoint"]
    require(seal_path.is_file(), f"missing seal: {seal_path}")
    require(checkpoint_path.is_file(), f"missing checkpoint: {checkpoint_path}")
    require(sha256_file(seal_path) == specification["seal_sha256"], "seal hash mismatch")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    require(seal.get("status") == "sealed", "experiment is not sealed")
    require(seal.get("run_id") == specification["model_id"], "seal run identity mismatch")
    require(seal.get("tokenizer_sha256") == TOKENIZER_SHA256, "seal tokenizer hash mismatch")
    require(
        seal.get("checkpoint_sha256", {}).get("best_validation")
        == specification["checkpoint_sha256"],
        "seal selected-checkpoint hash mismatch",
    )
    if name == "50m":
        require(
            seal.get("baseline_seal_sha256", {}).get("compute5")
            == MODEL_REGISTRY["compute5"]["seal_sha256"],
            "50M seal is not bound to the registered Compute5 seal",
        )
    require(
        sha256_file(checkpoint_path) == specification["checkpoint_sha256"],
        "checkpoint hash mismatch",
    )
    return {
        **specification,
        "seal_path": str(seal_path),
        "checkpoint_path": str(checkpoint_path),
        "selected_epoch": seal.get("selected_epoch"),
        "selection_metric": seal.get("selection_metric"),
    }


def _prefix_ids(tokenizer: Tokenizer, text: str) -> list[int]:
    story = tokenizer.token_to_id("<|story|>")
    bos = tokenizer.token_to_id("<|bos|>")
    require(story is not None and bos is not None, "tokenizer lacks story/BOS tokens")
    return [cast(int, story), cast(int, bos), *tokenizer.encode(text).ids]


@torch.no_grad()
def score_span(
    model: FictionPulperLM,
    prefix_ids: Sequence[int],
    span_ids: Sequence[int],
    *,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, Any]:
    """Score exactly ``span_ids``; prefix tokens never enter the arithmetic mean."""
    require(bool(prefix_ids), "scored prefix must not be empty")
    require(bool(span_ids), "scored span must not be empty")
    token_ids = [*prefix_ids, *span_ids]
    require(len(token_ids) <= model.config.max_seq_len, "scored sequence exceeds model context")
    inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        logits, _ = model(inputs)
    start = len(prefix_ids) - 1
    selected = logits[0, start : start + len(span_ids)].float()
    targets = torch.tensor(span_ids, dtype=torch.long, device=device)
    token_log_probabilities = F.log_softmax(selected, dim=-1).gather(1, targets[:, None])[:, 0]
    values = [float(value) for value in token_log_probabilities.cpu().tolist()]
    return {
        "token_ids": list(span_ids),
        "token_count": len(span_ids),
        "token_log_probabilities": values,
        "mean_log_probability": sum(values) / len(values),
        "nll": -sum(values) / len(values),
    }


def preference(left: float, right: float) -> str:
    if left > right:
        return "X"
    if right > left:
        return "Y"
    return "tie"


def _base_record(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": scenario["scenario_id"],
        "scenario_stable_hash": scenario["stable_hash"],
        "split": scenario["split"],
        "state_family": scenario["state_family"],
        "reasoning_depth": scenario["reasoning_depth"],
        "depth_bucket": scenario["depth_bucket"],
        "candidate_presentation_order": scenario["candidate_presentation_order"],
    }


def evaluate_forced_choice(
    scenarios: Sequence[dict[str, Any]],
    tokenizer: Tokenizer,
    scorer: Callable[[Sequence[int], Sequence[int]], dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        candidate_ids = {}
        for label in ("X", "Y"):
            supplied = scenario["candidates"][label]["token_ids"]
            require(
                tokenizer.encode(scenario["candidates"][label]["text"]).ids == supplied,
                f"{scenario['scenario_id']}: candidate {label} token IDs changed",
            )
            candidate_ids[label] = supplied
        contexts = [
            ("main", world_name, scenario["worlds"][world_name]["context"])
            for world_name in ("A", "B")
        ] + [
            ("candidate_only", None, ""),
            ("fact_removed", None, scenario["controls"]["fact_removed_context"]),
            ("irrelevant_substituted", None, scenario["controls"]["irrelevant_substituted_context"]),
        ]
        for context_kind, world_name, context in contexts:
            scores = {
                label: scorer(_prefix_ids(tokenizer, context), candidate_ids[label])
                for label in ("X", "Y")
            }
            chosen = preference(
                scores["X"]["mean_log_probability"], scores["Y"]["mean_log_probability"]
            )
            correct = (
                scenario["worlds"][world_name]["correct_candidate"]
                if world_name is not None else None
            )
            records.append({
                **_base_record(scenario),
                "context_kind": context_kind,
                "world": world_name,
                "context": context,
                "correct_candidate": correct,
                "candidate_scores": scores,
                "preferred_candidate": chosen,
                "signed_correct_margin": (
                    scores[correct]["mean_log_probability"]
                    - scores["Y" if correct == "X" else "X"]["mean_log_probability"]
                    if correct else None
                ),
                "x_minus_y_margin": (
                    scores["X"]["mean_log_probability"] - scores["Y"]["mean_log_probability"]
                ),
                "correct": chosen == correct if correct else None,
                "tie": chosen == "tie",
            })
    return records


def evaluate_teacher_forced(
    scenarios: Sequence[dict[str, Any]],
    tokenizer: Tokenizer,
    scorer: Callable[[Sequence[int], Sequence[int]], dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for scenario in scenarios:
        for world_name in ("A", "B"):
            world = scenario["worlds"][world_name]
            prefix = _prefix_ids(tokenizer, world["context"])
            gold = scorer(prefix, tokenizer.encode(world["gold_continuation"]).ids)
            counterfactual = scorer(prefix, tokenizer.encode(world["counterfactual_continuation"]).ids)
            margin = counterfactual["nll"] - gold["nll"]
            records.append({
                **_base_record(scenario),
                "world": world_name,
                "context": world["context"],
                "gold_continuation": world["gold_continuation"],
                "counterfactual_continuation": world["counterfactual_continuation"],
                "gold": gold,
                "counterfactual": counterfactual,
                "gold_preference_margin": margin,
                "gold_preferred": margin > 0,
                "tie": margin == 0,
            })
    return records


def _strata(records: Sequence[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for dimension, value in (
            ("overall", "all"),
            ("family", str(record["state_family"])),
            ("depth", str(record["depth_bucket"])),
        ):
            groups[(dimension, value)].append(record)
    return groups


def aggregate_forced_choice(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    main = [record for record in records if record["context_kind"] == "main"]
    require(bool(main), "forced-choice records contain no main contexts")
    metrics = []
    for (dimension, value), items in sorted(_strata(main).items()):
        pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            pairs[item["scenario_id"]].append(item)
        require(all({row["world"] for row in pair} == {"A", "B"} for pair in pairs.values()), "malformed A/B forced-choice pair")
        joint = [all(row["correct"] for row in pair) for pair in pairs.values()]
        metrics.append({
            "dimension": dimension,
            "stratum": value,
            "world_count": len(items),
            "scenario_pair_count": len(pairs),
            "world_accuracy": sum(row["correct"] for row in items) / len(items),
            "joint_paired_reversal_success": sum(joint) / len(joint),
            "mean_signed_margin": sum(row["signed_correct_margin"] for row in items) / len(items),
            "tie_rate": sum(row["tie"] for row in items) / len(items),
        })
    controls = []
    for kind in CONTROL_CONTEXTS:
        selected = [record for record in records if record["context_kind"] == kind]
        require(bool(selected), f"forced-choice records lack {kind} controls")
        for (dimension, value), items in sorted(_strata(selected).items()):
            controls.append({
                "context_kind": kind,
                "dimension": dimension,
                "stratum": value,
                "scenario_count": len(items),
                "x_preference_rate": sum(row["preferred_candidate"] == "X" for row in items) / len(items),
                "y_preference_rate": sum(row["preferred_candidate"] == "Y" for row in items) / len(items),
                "tie_rate": sum(row["tie"] for row in items) / len(items),
                "mean_x_minus_y_margin": sum(row["x_minus_y_margin"] for row in items) / len(items),
            })
    return {"mode": "forced_choice", "primary_metric": "joint_paired_reversal_success", "metrics": metrics, "controls": controls}


def aggregate_teacher_forced(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    require(bool(records), "teacher-forced records are empty")
    metrics = []
    for (dimension, value), items in sorted(_strata(records).items()):
        pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            pairs[item["scenario_id"]].append(item)
        require(all({row["world"] for row in pair} == {"A", "B"} for pair in pairs.values()), "malformed A/B teacher-forced pair")
        joint = [all(row["gold_preferred"] for row in pair) for pair in pairs.values()]
        gold_nll = sum(row["gold"]["nll"] for row in items) / len(items)
        counterfactual_nll = sum(row["counterfactual"]["nll"] for row in items) / len(items)
        metrics.append({
            "dimension": dimension,
            "stratum": value,
            "world_count": len(items),
            "scenario_pair_count": len(pairs),
            "world_gold_preference_rate": sum(row["gold_preferred"] for row in items) / len(items),
            "joint_gold_over_counterfactual_preference": sum(joint) / len(joint),
            "mean_gold_nll": gold_nll,
            "mean_counterfactual_nll": counterfactual_nll,
            "mean_gold_preference_margin": sum(row["gold_preference_margin"] for row in items) / len(items),
            "gold_perplexity": math.exp(gold_nll),
            "counterfactual_perplexity": math.exp(counterfactual_nll),
            "tie_rate": sum(row["tie"] for row in items) / len(items),
        })
    return {"mode": "teacher_forced", "primary_metric": "joint_gold_over_counterfactual_preference", "metrics": metrics}


def _git_metadata(root: Path) -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, check=True, capture_output=True, text=True).stdout
    return {"git_commit": commit, "tracked_worktree_clean": not bool(status.strip())}


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    require(not args.output_dir.exists(), f"refusing to overwrite existing output directory: {args.output_dir}")
    model_identity = verify_sealed_model(args.model, root)
    protocol, benchmark_hashes = load_protocol(args.benchmark_root)
    scenarios = load_authored_split(
        args.benchmark_root, args.split,
        checkpoint_selection_complete=args.checkpoint_selection_complete,
        test_evaluation_complete=args.test_evaluation_complete,
    )
    controls = load_authored_split(
        args.benchmark_root, args.split,
        checkpoint_selection_complete=args.checkpoint_selection_complete,
        test_evaluation_complete=args.test_evaluation_complete,
        controls=True,
    )
    require([row["scenario_id"] for row in scenarios] == [row["scenario_id"] for row in controls], "candidate-control rows do not align")
    tokenizer_path = root / "data/tokenizer/tokenizer.json"
    require(sha256_file(tokenizer_path) == TOKENIZER_SHA256, "tokenizer hash mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    device = torch.device(args.device)
    use_bf16 = args.precision == "bf16"
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA is unavailable")
        require(not use_bf16 or torch.cuda.is_bf16_supported(), "CUDA BF16 is unavailable")
    else:
        require(not use_bf16, "BF16 baseline evaluation requires CUDA")
    checkpoint = torch.load(model_identity["checkpoint_path"], map_location=device, weights_only=False)
    checkpoint_tokenizer = checkpoint.get("tokenizer_hash", checkpoint.get("tokenizer_sha256"))
    require(checkpoint_tokenizer == TOKENIZER_SHA256, "checkpoint tokenizer hash mismatch")
    model = FictionPulperLM(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    scorer = lambda prefix, span: score_span(model, prefix, span, device=device, use_bf16=use_bf16)
    forced = evaluate_forced_choice(scenarios, tokenizer, scorer)
    teacher = evaluate_teacher_forced(scenarios, tokenizer, scorer)
    forced_summary = aggregate_forced_choice(forced)
    teacher_summary = aggregate_teacher_forced(teacher)
    args.output_dir.mkdir(parents=True)
    paths = {
        "forced_choice_raw": args.output_dir / "forced-choice-raw.jsonl",
        "forced_choice_summary": args.output_dir / "forced-choice-summary.json",
        "teacher_forced_raw": args.output_dir / "teacher-forced-raw.jsonl",
        "teacher_forced_summary": args.output_dir / "teacher-forced-summary.json",
    }
    write_jsonl(paths["forced_choice_raw"], forced)
    write_json_atomic(paths["forced_choice_summary"], forced_summary)
    write_jsonl(paths["teacher_forced_raw"], teacher)
    write_json_atomic(paths["teacher_forced_summary"], teacher_summary)
    source_paths = [Path(__file__).resolve()]
    manifest = {
        "status": "automated_modes_complete_free_generation_and_human_primary_pending",
        "evaluator_version": EVALUATOR_VERSION,
        "model": model_identity,
        "split": args.split,
        "scenario_count": len(scenarios),
        **benchmark_hashes,
        "tokenizer_path": str(tokenizer_path.relative_to(root)),
        "tokenizer_sha256": TOKENIZER_SHA256,
        "checkpoint_path": model_identity["checkpoint"],
        "checkpoint_sha256": model_identity["checkpoint_sha256"],
        "seal_path": model_identity["seal"],
        "seal_sha256": model_identity["seal_sha256"],
        "model_config_sha256": sha256_bytes(canonical_json(asdict(model.config)).encode("utf-8")),
        "scorer_path": str(source_paths[0].relative_to(root)),
        "scorer_sha256": sha256_file(source_paths[0]),
        "evaluator_files": {str(path.relative_to(root)): sha256_file(path) for path in source_paths},
        "precision": args.precision,
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        **_git_metadata(root),
        "artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "reproduction_command": " ".join(args.reproduction_command),
        "held_out_lm_metrics": {
            "included": False,
            "reason": "Narrative Benchmark modes are reported separately from held-out LM metrics.",
            "historical_legacy_subsets_contaminated": True,
        },
        "free_generation": {
            "status": "pending",
            "workflow": "src.narrative_v2_free_generation; automated metrics are diagnostic only; blinded human review is primary",
        },
    }
    write_json_atomic(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), required=True)
    parser.add_argument("--split", choices=("development", "test", "generalization"), required=True)
    parser.add_argument("--benchmark-root", type=Path, default=Path("benchmarks/narrative-v2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-selection-complete", action="store_true")
    parser.add_argument("--test-evaluation-complete", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args(argv)
    args.reproduction_command = ["python", "-m", "src.narrative_v2_baseline", *(argv or [])]
    return args


def main() -> None:
    args = parse_args()
    if not args.reproduction_command[3:]:
        import sys
        args.reproduction_command = ["python", "-m", "src.narrative_v2_baseline", *sys.argv[1:]]
    run(args)


if __name__ == "__main__":
    main()
