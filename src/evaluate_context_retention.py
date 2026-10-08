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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Context-retention evaluation requires CUDA BF16")
    device = torch.device("cuda")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if sha256_file(args.tokenizer) != protocol["tokenizer_sha256"]:
        raise RuntimeError("Context-retention protocol tokenizer hash changed")

    checkpoints = {
        "compute5_context1024": args.baseline_checkpoint,
        "context2k": args.candidate_checkpoint,
    }
    result: dict[str, Any] = {
        "protocol_path": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "post_checkpoint_selection": True,
        "used_for_checkpoint_selection": False,
        "sample_seed": args.seed,
        "generation_settings": GENERATION_SETTINGS,
        "models": {},
    }
    for label, path in checkpoints.items():
        model, payload = load_checkpoint(path, device)
        if payload["tokenizer_hash"] != protocol["tokenizer_sha256"]:
            raise RuntimeError(f"Tokenizer hash mismatch for {label}")
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
    if result["models"]["compute5_context1024"]["max_seq_len"] != 1024:
        raise RuntimeError("Compute5 checkpoint does not have 1024 context")
    if result["models"]["context2k"]["max_seq_len"] != 2048:
        raise RuntimeError("Candidate checkpoint does not have 2048 context")
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
