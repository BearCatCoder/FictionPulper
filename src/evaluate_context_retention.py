"""Evaluate sealed checkpoints on the context-retention continuation suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from src.context_diagnostics import repetition_metrics
from src.model import FictionPulperLM, ModelConfig
from src.train import GENERATION_SETTINGS, generate
from src.train_tokenizer import sha256_file, write_json_atomic


def load_checkpoint(path: Path, device: torch.device) -> tuple[FictionPulperLM, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = FictionPulperLM(ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def fact_visibility(
    entry: dict[str, Any], *, context: int, max_new_tokens: int
) -> dict[str, Any]:
    controls = 2
    start_tokens = entry["prompt_token_count"] + controls
    maximum_tokens = start_tokens + max_new_tokens
    visible_at_start = start_tokens <= context
    visible_for_full_generation = maximum_tokens <= context
    if visible_at_start and not visible_for_full_generation:
        raise RuntimeError(
            f"Fact prefix for {entry['id']} would leave the {context}-token context "
            f"during {max_new_tokens}-token generation "
            f"({start_tokens} prompt/control tokens + {max_new_tokens} generated tokens)"
        )
    return {
        "control_token_count": controls,
        "max_new_tokens": max_new_tokens,
        "maximum_sequence_tokens": maximum_tokens,
        "fact_prefix_visible_at_generation_start": visible_at_start,
        "fact_prefix_visible_for_full_generation": visible_for_full_generation,
    }


def generate_suite(
    *,
    model: FictionPulperLM,
    tokenizer: Tokenizer,
    protocol: dict[str, Any],
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    entries = []
    for index, item in enumerate(protocol["entries"]):
        visibility = {
            "greedy": fact_visibility(
                item,
                context=model.config.max_seq_len,
                max_new_tokens=GENERATION_SETTINGS["greedy_max_new_tokens"],
            ),
            "sampled": fact_visibility(
                item,
                context=model.config.max_seq_len,
                max_new_tokens=GENERATION_SETTINGS["sampled_max_new_tokens"],
            ),
        }
        greedy = generate(
            model,
            tokenizer,
            item["prompt"],
            max_new_tokens=GENERATION_SETTINGS["greedy_max_new_tokens"],
            device=device,
            use_bf16=True,
        )
        sampled = generate(
            model,
            tokenizer,
            item["prompt"],
            max_new_tokens=GENERATION_SETTINGS["sampled_max_new_tokens"],
            device=device,
            use_bf16=True,
            temperature=GENERATION_SETTINGS["temperature"],
            top_k=GENERATION_SETTINGS["top_k"],
            top_p=GENERATION_SETTINGS["top_p"],
            seed=seed + index,
        )
        entries.append(
            {
                "id": item["id"],
                "facts": item["facts"],
                "prompt_token_count": item["prompt_token_count"],
                "fact_prefix_token_count": item["fact_prefix_token_count"],
                "tokens_after_fact_prefix": item["tokens_after_fact_prefix"],
                "fact_prefix_visible_at_generation_start": (
                    item["prompt_token_count"] + 2 <= model.config.max_seq_len
                ),
                "fact_visibility": visibility,
                "greedy": greedy,
                "sampled": sampled,
                "sample_seed": seed + index,
                "repetition": {
                    "greedy": repetition_metrics(tokenizer, greedy),
                    "sampled": repetition_metrics(tokenizer, sampled),
                },
            }
        )
    return entries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11337)
    parser.add_argument("--baseline-label", default="compute5_context1024")
    parser.add_argument("--candidate-label", default="context2k")
    parser.add_argument("--baseline-context", type=int, default=1024)
    parser.add_argument("--candidate-context", type=int, default=2048)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Context-retention evaluation requires CUDA BF16")
    device = torch.device("cuda")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if sha256_file(args.tokenizer) != protocol["tokenizer_sha256"]:
        raise RuntimeError("Context-retention protocol tokenizer hash changed")
    if args.baseline_label == args.candidate_label:
        raise RuntimeError("Baseline and candidate labels must differ")
    if args.baseline_context <= 0 or args.candidate_context <= 0:
        raise RuntimeError("Baseline and candidate contexts must be positive")

    checkpoints = {
        args.baseline_label: (args.baseline_checkpoint, args.baseline_context),
        args.candidate_label: (args.candidate_checkpoint, args.candidate_context),
    }
    result: dict[str, Any] = {
        "protocol_path": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "protocol": {
            key: protocol[key]
            for key in (
                "name",
                "purpose",
                "source_suite",
                "minimum_prompt_tokens",
                "maximum_prompt_tokens",
                "maximum_context",
                "tokenizer_path",
                "tokenizer_sha256",
            )
            if key in protocol
        },
        "post_checkpoint_selection": True,
        "used_for_checkpoint_selection": False,
        "sample_seed": args.seed,
        "generation_settings": GENERATION_SETTINGS,
        "models": {},
    }
    for label, (path, expected_context) in checkpoints.items():
        model, payload = load_checkpoint(path, device)
        if payload["tokenizer_hash"] != protocol["tokenizer_sha256"]:
            raise RuntimeError(f"Tokenizer hash mismatch for {label}")
        if model.config.max_seq_len != expected_context:
            raise RuntimeError(
                f"{label} checkpoint context is {model.config.max_seq_len}, "
                f"expected {expected_context}"
            )
        result["models"][label] = {
            "checkpoint_path": str(path),
            "checkpoint_sha256": sha256_file(path),
            "epoch": int(payload["epoch"]),
            "optimizer_step": int(payload["step"]),
            "max_seq_len": model.config.max_seq_len,
            "entries": generate_suite(
                model=model,
                tokenizer=tokenizer,
                protocol=protocol,
                device=device,
                seed=args.seed,
            ),
        }
        del model, payload
        torch.cuda.empty_cache()
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
