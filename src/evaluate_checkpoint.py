"""Evaluate one FictionPulper checkpoint against one packed split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.dataset import PackedStoryDataset
from src.model import FictionPulperLM, ModelConfig
from src.train import evaluate
from src.train_tokenizer import sha256_file, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--exclude-id", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Checkpoint evaluation requires CUDA BF16 support")
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    tokenizer_hash = sha256_file(args.tokenizer)
    if checkpoint["tokenizer_hash"] != tokenizer_hash:
        raise RuntimeError("Checkpoint and tokenizer hashes do not match")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model_config = ModelConfig(**checkpoint["model_config"])
    model = FictionPulperLM(model_config).to("cuda")
    model.load_state_dict(checkpoint["model"])
    dataset = PackedStoryDataset(
        args.packed,
        args.packed.with_suffix(".index.json"),
        sequence_length=model_config.max_seq_len,
        pad_token_id=tokenizer.token_to_id("<|pad|>"),
        exclude_document_ids=set(args.exclude_id),
    )
    metrics = evaluate(
        model,
        dataset,
        batch_size=args.batch_size,
        device=torch.device("cuda"),
        use_bf16=True,
    )
    payload = {
        "name": args.name,
        "post_hoc": True,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_step": checkpoint["step"],
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_sha256": tokenizer_hash,
        "packed_path": str(args.packed),
        "packed_sha256": sha256_file(args.packed),
        "excluded_ids": sorted(args.exclude_id),
        "metrics": metrics,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
