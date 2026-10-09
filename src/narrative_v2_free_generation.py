"""Run and review Narrative Benchmark v2 free-generation evaluations."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, cast

import torch
from tokenizers import Tokenizer

from src.continuity_curriculum.common import (
    canonical_json,
    load_jsonl,
    sha256_bytes,
    sha256_file,
    stable_seed,
    write_json_atomic,
    write_jsonl,
)
from src.benchmark_v2 import EXPECTED_PROTOCOL_SHA256
from src.model import FictionPulperLM, ModelConfig
from src.narrative_v2_loader import load_authored_split
from src.narrative_v2_scoring import (
    REVIEW_CONTENT_FIELDS,
    aggregate_diagnostics,
    build_review_packet,
    diagnostic_score,
    review_id,
    score_human_reviews,
)
from src.train import sample_next_token


SCORER_VERSION = "narrative-v2-free-generation-1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_once_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    write_json_atomic(path, value)


def _write_once_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    write_jsonl(path, records)


def load_protocol(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(sha256_file(protocol_path) == EXPECTED_PROTOCOL_SHA256, "frozen protocol hash mismatch")
    require(protocol["benchmark_id"] == "fictionpulper-narrative-benchmark-v2", "benchmark ID changed")
    require(protocol["protocol_version"] == 2, "protocol version changed")
    hashes = {"protocol_sha256": sha256_file(protocol_path)}
    for key, output_key in (
        ("allocation_plan", "allocation_plan_sha256"),
        ("scenario_schema", "scenario_schema_sha256"),
        ("human_review_rubric", "human_review_rubric_sha256"),
    ):
        spec = protocol["artifacts"][key]
        actual = sha256_file(root / spec["path"])
        require(actual == spec["sha256"], f"{key} hash mismatch")
        hashes[output_key] = actual
    manifest_path = root / "authored-manifest.json"
    authored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, expected in authored_manifest["locked_inputs"].items():
        require(sha256_file(root / filename) == expected, f"authored manifest lock mismatch: {filename}")
    hashes["scenario_data_manifest_sha256"] = sha256_file(manifest_path)
    decoding = protocol["decoding"]
    require(decoding["greedy"] == {"max_new_tokens": 128}, "greedy decoding changed")
    require(decoding["sampled"] == {
        "max_new_tokens": 256, "temperature": 0.8, "top_k": 50,
        "top_p": 0.95, "base_seeds": [11337, 21337, 31337],
    }, "sampled decoding changed")
    uncertainty = protocol["uncertainty"]
    require(uncertainty["bootstrap_replicates"] == 10_000, "bootstrap replicates changed")
    require(uncertainty["bootstrap_seed"] == 260_209, "bootstrap seed changed")
    return protocol, hashes


def effective_seed(base_seed: int, scenario_id: str, world: str, blinded_model_id: str) -> int:
    return stable_seed(base_seed, scenario_id, world, blinded_model_id)


@torch.no_grad()
def generate_continuation(
    model: FictionPulperLM,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    device: torch.device,
    use_bf16: bool,
    temperature: float | None = None,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    prefix = [tokenizer.token_to_id("<|story|>"), tokenizer.token_to_id("<|bos|>")]
    if any(token is None for token in prefix):
        raise ValueError("tokenizer lacks story/BOS control tokens")
    prompt_ids = tokenizer.encode(prompt).ids
    if len(prefix) + len(prompt_ids) + max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt and locked generation length exceed model context")
    eos_token_id = tokenizer.token_to_id("<|eos|>")
    if eos_token_id is None:
        raise ValueError("tokenizer lacks EOS token")
    context = torch.tensor([prefix + prompt_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    for _ in range(max_new_tokens):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits, _ = model(context)
        next_logits = logits[0, -1]
        next_token = (
            int(next_logits.argmax().item())
            if temperature is None
            else sample_next_token(
                next_logits, temperature=temperature, top_k=top_k,
                top_p=top_p, generator=generator,
            )
        )
        generated.append(next_token)
        context = torch.cat(
            [context, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1
        )
        if next_token == eos_token_id:
            break
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "generated_text": text,
        "generated_token_ids": generated,
        "actual_generated_token_count": len(generated),
        "stop_reason": "eos" if generated and generated[-1] == eos_token_id else "max_new_tokens",
        "prompt_token_count": len(prompt_ids),
    }


def _generation_id(
    split: str, scenario_id: str, world: str, mode: str,
    sampling_seed: int | None, blinded_model_id: str,
) -> str:
    payload = [split, scenario_id, world, mode, sampling_seed, blinded_model_id]
    return "generation-" + sha256_bytes(canonical_json(payload).encode("utf-8"))[:20]


def _stale_values(scenario: dict[str, Any], world_name: str) -> list[str]:
    state = str(scenario["intervention"][f"world_{world_name.casefold()}_value"])
    chain = [state]
    for event in sorted(scenario["events"], key=lambda item: int(item["sequence"])):
        if event["predicate"] == "initialize_from_intervention":
            continue
        mapping = json.loads(event["object"])
        if state not in mapping:
            raise ValueError(f"{scenario['scenario_id']} world {world_name}: broken state chain")
        state = str(mapping[state])
        chain.append(state)
    answer = str(scenario["worlds"][world_name]["proof"]["answer_value"])
    require(state == answer, f"{scenario['scenario_id']} world {world_name}: proof answer mismatch")
    return list(dict.fromkeys(value for value in chain[:-1] if value != answer))


def generate_records(
    *,
    scenarios: list[dict[str, Any]],
    protocol: dict[str, Any],
    model: FictionPulperLM,
    tokenizer: Tokenizer,
    split: str,
    blinded_model_id: str,
    device: torch.device,
    use_bf16: bool,
) -> list[dict[str, Any]]:
    records = []
    decoding = protocol["decoding"]
    for scenario in scenarios:
        for world_name in ("A", "B"):
            world = scenario["worlds"][world_name]
            other_world = scenario["worlds"]["B" if world_name == "A" else "A"]
            trials = [("greedy", None, None, decoding["greedy"])] + [
                ("sampled", base_seed, effective_seed(
                    base_seed, scenario["scenario_id"], world_name, blinded_model_id
                ), decoding["sampled"])
                for base_seed in decoding["sampled"]["base_seeds"]
            ]
            for mode, base_seed, seed, settings in trials:
                output = generate_continuation(
                    model, tokenizer, world["free_generation"]["prompt"],
                    max_new_tokens=int(settings["max_new_tokens"]),
                    device=device, use_bf16=use_bf16,
                    temperature=None if mode == "greedy" else float(settings["temperature"]),
                    top_k=0 if mode == "greedy" else int(settings["top_k"]),
                    top_p=1.0 if mode == "greedy" else float(settings["top_p"]),
                    seed=0 if seed is None else seed,
                )
                record = {
                    "generation_id": _generation_id(
                        split, scenario["scenario_id"], world_name, mode, base_seed, blinded_model_id
                    ),
                    "scenario_id": scenario["scenario_id"],
                    "scenario_stable_hash": scenario["stable_hash"],
                    "split": split,
                    "state_family": scenario["state_family"],
                    "reasoning_depth": scenario["reasoning_depth"],
                    "depth_bucket": scenario["depth_bucket"],
                    "world": world_name,
                    "generation_mode": mode,
                    "sampling_seed": base_seed,
                    "effective_seed": seed,
                    "blinded_model_id": blinded_model_id,
                    "prompt": world["free_generation"]["prompt"],
                    "review_question": world["free_generation"]["review_question"],
                    "required_propositions": world["free_generation"]["required_propositions"],
                    "forbidden_propositions": world["free_generation"]["forbidden_propositions"],
                    "entity_names": [entity["name"] for entity in scenario["entities"]],
                    "answer_value": world["proof"]["answer_value"],
                    "opposing_answer_value": other_world["proof"]["answer_value"],
                    "stale_values": _stale_values(scenario, world_name),
                    "initial_value": scenario["intervention"][f"world_{world_name.casefold()}_value"],
                    "opposing_initial_value": scenario["intervention"][
                        f"world_{'b' if world_name == 'A' else 'a'}_value"
                    ],
                    "supporting_event_ids": world["proof"]["supporting_event_ids"],
                    **output,
                }
                record["generated_text_sha256"] = sha256_bytes(
                    record["generated_text"].encode("utf-8")
                )
                record["diagnostics"] = diagnostic_score(record)
                records.append(record)
    return records


def _git_metadata() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    ).stdout
    return commit, not bool(status.strip())


def _scorer_provenance() -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), Path(__file__).with_name("narrative_v2_scoring.py").resolve()]
    files = {
        str(path.relative_to(repository_root)): sha256_file(path)
        for path in paths
    }
    return {
        "scorer_path": " + ".join(files),
        "scorer_sha256": sha256_bytes(canonical_json(files).encode("utf-8")),
        "scorer_files": files,
    }


def validate_generation_artifacts(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    scenarios: list[dict[str, Any]],
    protocol: dict[str, Any],
    *,
    generations_sha256: str,
) -> None:
    require(manifest.get("scorer_version") == SCORER_VERSION, "generation scorer version mismatch")
    require(manifest.get("raw_generations_sha256") == generations_sha256, "raw generation hash mismatch")
    require(manifest.get("decoding") == protocol["decoding"], "generation decoding protocol mismatch")
    require(manifest.get("scenario_count") == len(scenarios), "generation scenario count mismatch")
    require(manifest.get("generation_count") == len(records), "generation count mismatch")
    require(manifest.get("scorer_sha256") == _scorer_provenance()["scorer_sha256"], "generation scorer hash mismatch")
    split = str(manifest.get("split"))
    blinded_model_id = str(manifest.get("blinded_model_id"))
    scenario_by_id = {str(scenario["scenario_id"]): scenario for scenario in scenarios}
    expected_trials = {
        (scenario_id, world, mode, seed)
        for scenario_id in scenario_by_id
        for world in ("A", "B")
        for mode, seed in [
            ("greedy", None),
            *(("sampled", seed) for seed in protocol["decoding"]["sampled"]["base_seeds"]),
        ]
    }
    actual_trials: set[tuple[str, str, str, int | None]] = set()
    generation_ids: set[str] = set()
    for record in records:
        scenario_id = str(record.get("scenario_id"))
        world = str(record.get("world"))
        mode = str(record.get("generation_mode"))
        sampling_seed = record.get("sampling_seed")
        trial = (scenario_id, world, mode, sampling_seed)
        require(trial in expected_trials, f"unexpected generation trial: {trial}")
        require(trial not in actual_trials, f"duplicate generation trial: {trial}")
        if mode == "sampled" and not isinstance(sampling_seed, int):
            raise RuntimeError(f"sampled generation lacks an integer base seed: {trial}")
        actual_trials.add(trial)
        generation_id = str(record.get("generation_id"))
        require(generation_id not in generation_ids, f"duplicate generation_id: {generation_id}")
        generation_ids.add(generation_id)
        scenario = scenario_by_id[scenario_id]
        other_world = scenario["worlds"]["B" if world == "A" else "A"]
        world_data = scenario["worlds"][world]
        expected_seed = (
            None if mode == "greedy"
            else effective_seed(cast(int, sampling_seed), scenario_id, world, blinded_model_id)
        )
        expected_fields = {
            "generation_id": _generation_id(split, scenario_id, world, mode, sampling_seed, blinded_model_id),
            "scenario_stable_hash": scenario["stable_hash"],
            "split": split,
            "state_family": scenario["state_family"],
            "reasoning_depth": scenario["reasoning_depth"],
            "depth_bucket": scenario["depth_bucket"],
            "effective_seed": expected_seed,
            "blinded_model_id": blinded_model_id,
            "prompt": world_data["free_generation"]["prompt"],
            "review_question": world_data["free_generation"]["review_question"],
            "required_propositions": world_data["free_generation"]["required_propositions"],
            "forbidden_propositions": world_data["free_generation"]["forbidden_propositions"],
            "entity_names": [entity["name"] for entity in scenario["entities"]],
            "answer_value": world_data["proof"]["answer_value"],
            "opposing_answer_value": other_world["proof"]["answer_value"],
            "stale_values": _stale_values(scenario, world),
            "initial_value": scenario["intervention"][f"world_{world.casefold()}_value"],
            "opposing_initial_value": scenario["intervention"][f"world_{'b' if world == 'A' else 'a'}_value"],
            "supporting_event_ids": world_data["proof"]["supporting_event_ids"],
        }
        for field, expected in expected_fields.items():
            require(record.get(field) == expected, f"{generation_id}: {field} mismatch")
        text = record.get("generated_text")
        token_ids = record.get("generated_token_ids")
        if not isinstance(text, str):
            raise RuntimeError(f"{generation_id}: generated_text must be a string")
        if not isinstance(token_ids, list) or not all(
            isinstance(token, int) and not isinstance(token, bool) for token in token_ids
        ):
            raise RuntimeError(f"{generation_id}: generated_token_ids must be integers")
        require(record.get("generated_text_sha256") == sha256_bytes(text.encode("utf-8")), f"{generation_id}: generated text hash mismatch")
        require(record.get("actual_generated_token_count") == len(token_ids), f"{generation_id}: token count mismatch")
        maximum = int(protocol["decoding"][mode]["max_new_tokens"])
        stop_reason = record.get("stop_reason")
        require(stop_reason in {"eos", "max_new_tokens"}, f"{generation_id}: invalid stop reason")
        require(0 < len(token_ids) <= maximum, f"{generation_id}: invalid generated token count")
        if stop_reason == "max_new_tokens":
            require(len(token_ids) == maximum, f"{generation_id}: early output lacks EOS stop reason")
        require(record.get("diagnostics") == diagnostic_score(record), f"{generation_id}: diagnostics mismatch")
    require(actual_trials == expected_trials, "generation trial matrix is incomplete")


def _validated_run(
    args: argparse.Namespace, protocol: dict[str, Any], benchmark_hashes: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(args.generation_manifest.read_text(encoding="utf-8"))
    for field, expected in benchmark_hashes.items():
        require(manifest.get(field) == expected, f"generation manifest {field} mismatch")
    scenarios = load_authored_split(
        args.benchmark_root, str(manifest["split"]),
        checkpoint_selection_complete=args.checkpoint_selection_complete,
        test_evaluation_complete=args.test_evaluation_complete,
    )
    records = load_jsonl(args.generations)
    validate_generation_artifacts(
        records, manifest, scenarios, protocol,
        generations_sha256=sha256_file(args.generations),
    )
    return records, manifest


def _load_model(
    checkpoint_path: Path, expected_checkpoint_sha256: str, tokenizer_sha256: str,
    device: torch.device,
) -> tuple[FictionPulperLM, dict[str, Any]]:
    require(sha256_file(checkpoint_path) == expected_checkpoint_sha256, "checkpoint hash mismatch")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_tokenizer_hash = payload.get("tokenizer_hash", payload.get("tokenizer_sha256"))
    require(checkpoint_tokenizer_hash == tokenizer_sha256, "checkpoint tokenizer hash mismatch")
    model = FictionPulperLM(ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    model.requires_grad_(False)
    return model, payload


def run_generation(args: argparse.Namespace) -> None:
    output_paths = [
        args.output_dir / "raw-generations.jsonl",
        args.output_dir / "diagnostic-summary.json",
        args.output_dir / "generation-manifest.json",
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifacts: {existing}")
    protocol, benchmark_hashes = load_protocol(args.benchmark_root)
    scenarios = load_authored_split(
        args.benchmark_root, args.split,
        checkpoint_selection_complete=args.checkpoint_selection_complete,
        test_evaluation_complete=args.test_evaluation_complete,
    )
    tokenizer_hash = sha256_file(args.tokenizer)
    require(tokenizer_hash == args.tokenizer_sha256, "tokenizer hash mismatch")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    device = torch.device(args.device)
    use_bf16 = args.precision == "bf16"
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA is unavailable")
        if use_bf16:
            require(torch.cuda.is_bf16_supported(), "CUDA BF16 is unavailable")
    elif use_bf16:
        raise RuntimeError("BF16 benchmark generation requires CUDA")
    model, payload = _load_model(
        args.checkpoint, args.checkpoint_sha256, tokenizer_hash, device
    )
    records = generate_records(
        scenarios=scenarios, protocol=protocol, model=model, tokenizer=tokenizer,
        split=args.split, blinded_model_id=args.blinded_model_id,
        device=device, use_bf16=use_bf16,
    )
    git_commit, clean = _git_metadata()
    model_config_sha256 = sha256_bytes(
        canonical_json(asdict(model.config)).encode("utf-8")
    )
    manifest = {
        "status": "generation_complete_human_review_pending",
        "scorer_version": SCORER_VERSION,
        "benchmark_id": protocol["benchmark_id"],
        "protocol_version": protocol["protocol_version"],
        "split": args.split,
        "scenario_count": len(scenarios),
        "generation_count": len(records),
        "blinded_model_id": args.blinded_model_id,
        "model_id": args.model_id,
        **benchmark_hashes,
        **_scorer_provenance(),
        "git_commit": git_commit,
        "tracked_worktree_clean": clean,
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_sha256": tokenizer_hash,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "model_config_sha256": model_config_sha256,
        "precision": args.precision,
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "checkpoint_step": payload.get("step"),
        "checkpoint_epoch": payload.get("epoch"),
        "decoding": protocol["decoding"],
        "raw_generations_sha256": sha256_bytes(
            "".join(canonical_json(record) + "\n" for record in records).encode("utf-8")
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_once_jsonl(args.output_dir / "raw-generations.jsonl", records)
    _write_once_json(args.output_dir / "diagnostic-summary.json", aggregate_diagnostics(records))
    _write_once_json(args.output_dir / "generation-manifest.json", manifest)
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, indent=2))


def export_review(args: argparse.Namespace) -> None:
    protocol, hashes = load_protocol(args.benchmark_root)
    records, generation_manifest = _validated_run(args, protocol, hashes)
    if args.output.exists() or args.key_output.exists():
        raise FileExistsError("refusing to overwrite review packet or private key")
    rubric = json.loads(
        (args.benchmark_root / "human-review-rubric.json").read_text(encoding="utf-8")
    )
    packet = build_review_packet(
        records, randomization_seed=int(rubric["randomization_seed"])
    )
    _write_once_jsonl(args.output, packet)
    key = {
        "review_packet_sha256": sha256_file(args.output),
        "blinded_models": sorted({record["blinded_model_id"] for record in records}),
        "model_mapping": {
            generation_manifest["blinded_model_id"]: generation_manifest["model_id"]
        },
        "generation_manifest_path": str(args.generation_manifest),
        "generation_manifest_sha256": sha256_file(args.generation_manifest),
        "rubric_path": str(args.benchmark_root / "human-review-rubric.json"),
        "rubric_sha256": sha256_file(args.benchmark_root / "human-review-rubric.json"),
        "review_metadata": {
            review_id(record): {
                field: record.get(field)
                for field in (
                    "scenario_id", "world", "generation_mode", "sampling_seed",
                    "blinded_model_id", "generated_text_sha256",
                )
            }
            for record in records
        },
        "instructions": (
            "Two independent reviewers complete separate copies. Any disagreement or "
            "uncertain label requires one independent adjudicator row. Preserve original rows."
        ),
    }
    _write_once_json(args.key_output, key)
    print(json.dumps({"review_rows": len(packet), "output": str(args.output)}, indent=2))


def _validated_completed_reviews(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rubric = json.loads(
        (args.benchmark_root / "human-review-rubric.json").read_text(encoding="utf-8")
    )
    packet = load_jsonl(args.review_packet)
    key = json.loads(args.review_key.read_text(encoding="utf-8"))
    generation_manifest = json.loads(args.generation_manifest.read_text(encoding="utf-8"))
    require(key.get("review_packet_sha256") == sha256_file(args.review_packet), "review packet hash mismatch")
    require(key.get("generation_manifest_sha256") == sha256_file(args.generation_manifest), "review key generation manifest mismatch")
    require(key.get("rubric_sha256") == sha256_file(args.benchmark_root / "human-review-rubric.json"), "review key rubric mismatch")
    require(
        key.get("model_mapping") == {
            generation_manifest["blinded_model_id"]: generation_manifest["model_id"]
        },
        "review key model mapping differs from generation manifest",
    )
    expected_packet = build_review_packet(
        records, randomization_seed=int(rubric["randomization_seed"])
    )
    require(packet == expected_packet, "review packet differs from locked generations")
    packet_by_id = {str(row["review_id"]): row for row in packet}
    metadata = key.get("review_metadata")
    require(isinstance(metadata, dict) and set(metadata) == set(packet_by_id), "review key metadata is incomplete")
    completed = load_jsonl(args.reviews)
    enriched = []
    for row in completed:
        identifier = str(row.get("review_id", ""))
        require(identifier in packet_by_id, f"unknown completed review_id: {identifier}")
        original = packet_by_id[identifier]
        for field in REVIEW_CONTENT_FIELDS:
            require(row.get(field) == original.get(field), f"{identifier}: reviewed {field} differs from packet")
        hidden = metadata[identifier]
        enriched_row = dict(row)
        for field, value in hidden.items():
            if field in row:
                require(row[field] == value, f"{identifier}: hidden {field} mismatch")
            enriched_row[field] = value
        enriched.append(enriched_row)
    return enriched, key


def score_reviews(args: argparse.Namespace) -> None:
    protocol, hashes = load_protocol(args.benchmark_root)
    records, generation_manifest = _validated_run(args, protocol, hashes)
    reviews, review_key = _validated_completed_reviews(args, records)
    report = score_human_reviews(
        records, reviews,
        bootstrap_replicates=int(protocol["uncertainty"]["bootstrap_replicates"]),
        bootstrap_seed=int(protocol["uncertainty"]["bootstrap_seed"]),
    )
    report.update({
        "scorer_version": SCORER_VERSION,
        **hashes,
        **_scorer_provenance(),
        "generations_sha256": sha256_file(args.generations),
        "generation_manifest_sha256": sha256_file(args.generation_manifest),
        "reviews_sha256": sha256_file(args.reviews),
        "review_packet_sha256": sha256_file(args.review_packet),
        "review_key_sha256": sha256_file(args.review_key),
        "review_model_mapping": review_key["model_mapping"],
        "generation_provenance": {
            key: generation_manifest[key]
            for key in (
                "checkpoint_path", "checkpoint_sha256", "tokenizer_path", "tokenizer_sha256",
                "model_config_sha256", "precision", "device", "python_version", "torch_version",
                "cuda_version", "git_commit", "tracked_worktree_clean",
            )
        },
    })
    git_commit, clean = _git_metadata()
    report["git_commit"] = git_commit
    report["tracked_worktree_clean"] = clean
    _write_once_json(args.output, report)
    print(json.dumps(report, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="run locked free generation")
    generate_parser.add_argument("--benchmark-root", type=Path, default=Path("benchmarks/narrative-v2"))
    generate_parser.add_argument("--split", choices=("development", "test", "generalization"), default="development")
    generate_parser.add_argument("--checkpoint-selection-complete", action="store_true")
    generate_parser.add_argument("--test-evaluation-complete", action="store_true")
    generate_parser.add_argument("--checkpoint", type=Path, required=True)
    generate_parser.add_argument("--checkpoint-sha256", required=True)
    generate_parser.add_argument("--tokenizer", type=Path, required=True)
    generate_parser.add_argument("--tokenizer-sha256", required=True)
    generate_parser.add_argument("--model-id", required=True)
    generate_parser.add_argument("--blinded-model-id", required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--device", default="cuda")
    generate_parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    generate_parser.set_defaults(handler=run_generation)

    export_parser = subparsers.add_parser("export-review", help="export deterministic blinded review rows")
    export_parser.add_argument("--benchmark-root", type=Path, default=Path("benchmarks/narrative-v2"))
    export_parser.add_argument("--generations", type=Path, required=True)
    export_parser.add_argument("--generation-manifest", type=Path, required=True)
    export_parser.add_argument("--checkpoint-selection-complete", action="store_true")
    export_parser.add_argument("--test-evaluation-complete", action="store_true")
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--key-output", type=Path, required=True)
    export_parser.set_defaults(handler=export_review)

    score_parser = subparsers.add_parser("score-reviews", help="validate adjudication and score pair consistency")
    score_parser.add_argument("--benchmark-root", type=Path, default=Path("benchmarks/narrative-v2"))
    score_parser.add_argument("--generations", type=Path, required=True)
    score_parser.add_argument("--generation-manifest", type=Path, required=True)
    score_parser.add_argument("--checkpoint-selection-complete", action="store_true")
    score_parser.add_argument("--test-evaluation-complete", action="store_true")
    score_parser.add_argument("--reviews", type=Path, required=True)
    score_parser.add_argument("--review-packet", type=Path, required=True)
    score_parser.add_argument("--review-key", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.set_defaults(handler=score_reviews)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
