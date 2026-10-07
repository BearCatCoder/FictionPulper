"""Indexed packed-token dataset that never crosses document boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedStoryDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        bin_path: str | Path,
        index_path: str | Path,
        *,
        sequence_length: int,
        pad_token_id: int,
        limit_sequences: int | None = None,
    ) -> None:
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
        self.documents: list[dict[str, Any]] = payload["documents"]
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id
        self.samples: list[tuple[int, int]] = []
        for document_index, document in enumerate(self.documents):
            if document["offset"] + document["length"] > len(self.tokens):
                raise ValueError(f"Document {document['id']} exceeds packed token file")
            for start in range(0, document["length"] - 1, sequence_length):
                self.samples.append((document_index, start))
        if limit_sequences is not None:
            self.samples = self.samples[:limit_sequences]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        document_index, relative_start = self.samples[index]
        document = self.documents[document_index]
        absolute_start = document["offset"] + relative_start
        available = min(self.sequence_length + 1, document["length"] - relative_start)
        chunk = np.asarray(
            self.tokens[absolute_start : absolute_start + available], dtype=np.int64
        )
        input_ids = np.full(self.sequence_length, self.pad_token_id, dtype=np.int64)
        labels = np.full(self.sequence_length, -100, dtype=np.int64)
        prediction_count = max(0, len(chunk) - 1)
        input_ids[:prediction_count] = chunk[:-1]
        labels[:prediction_count] = chunk[1:]
        return torch.from_numpy(input_ids), torch.from_numpy(labels)
