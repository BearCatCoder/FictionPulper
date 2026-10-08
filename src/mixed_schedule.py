"""Build and load deterministic, hashable mixed-source chunk schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping

import yaml
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from src.dataset import PackedStoryDataset
from src.train_tokenizer import sha256_file, write_json_atomic


SCHEDULE_VERSION = 1


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_targets(dataset: PackedStoryDataset, sample_index: int) -> int:
    document_index, relative_start = dataset.samples[sample_index]
    document_length = int(dataset.documents[document_index]["length"])
    return min(dataset.sequence_length, document_length - relative_start - 1)


def _sample_identity(
    dataset: PackedStoryDataset, sample_index: int
) -> tuple[str, int]:
    document_index, relative_start = dataset.samples[sample_index]
    return str(dataset.documents[document_index]["id"]), int(relative_start)


def _deterministic_samples(length: int, count: int, seed: int) -> list[int]:
    if length <= 0:
        raise ValueError("A schedule source cannot be empty")
    rng = random.Random(seed)
    result: list[int] = []
    while len(result) < count:
        epoch = list(range(length))
        rng.shuffle(epoch)
        result.extend(epoch[: count - len(result)])
    return result


def _ensure_capacity(
    dataset: PackedStoryDataset,
    sample_indices: list[int],
    target: int,
    seed: int,
) -> list[int]:
    """Deterministically replace low-capacity picks only when quota requires it."""
    capacities = [_valid_targets(dataset, index) for index in sample_indices]
    shortfall = target - sum(capacities)
    if shortfall <= 0:
        return sample_indices
    rng = random.Random(seed)
    alternatives = list(range(len(dataset)))
    rng.shuffle(alternatives)
    alternatives.sort(key=lambda index: _valid_targets(dataset, index), reverse=True)
    replacement_positions = sorted(
        range(len(sample_indices)), key=lambda position: capacities[position]
    )
    for position, replacement in zip(replacement_positions, alternatives, strict=False):
        gain = _valid_targets(dataset, replacement) - capacities[position]
        if gain <= 0:
            continue
        sample_indices[position] = replacement
        capacities[position] += gain
        shortfall -= gain
        if shortfall <= 0:
            return sample_indices
    raise ValueError(
        f"Cannot provide {target:,} targets from {len(sample_indices):,} source chunks"
    )


def _allocate_targets(capacities: list[int], target: int) -> list[int]:
    """Distribute an exact target without exceeding any selected chunk."""
    if target < len(capacities):
        raise ValueError("Target exposure must allow at least one target per chunk")
    if target > sum(capacities):
        raise ValueError(
            f"Selected chunks provide {sum(capacities):,} targets, below {target:,}"
        )
    remaining_target = target
    remaining_capacity = sum(capacities)
    allocated: list[int] = []
    for position, capacity in enumerate(capacities):
        chunks_after = len(capacities) - position - 1
        capacity_after = remaining_capacity - capacity
        proportional = round(remaining_target * capacity / remaining_capacity)
        minimum = max(1, remaining_target - capacity_after)
        maximum = min(capacity, remaining_target - chunks_after)
        take = min(max(proportional, minimum), maximum)
        allocated.append(take)
        remaining_target -= take
        remaining_capacity = capacity_after
    if remaining_target != 0:
        raise AssertionError("Target allocation did not exhaust the requested exposure")
    return allocated


def build_mixed_schedule(
    sources: Mapping[str, PackedStoryDataset],
    *,
    source_target_counts: Mapping[str, int],
    total_chunks: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    seed: int,
    source_artifacts: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Create a deterministic schedule with exact per-source valid-target totals."""
    source_names = list(source_target_counts)
    if set(source_names) != set(sources):
        raise ValueError("Sources and source_target_counts must have identical names")
    expected_chunks = batch_size * gradient_accumulation_steps * max_steps
    if total_chunks != expected_chunks:
        raise ValueError(
            f"total_chunks must equal batch_size * accumulation * max_steps "
            f"({expected_chunks:,})"
        )
    total_targets = sum(int(value) for value in source_target_counts.values())
    chunk_counts: dict[str, int] = {}
    assigned = 0
    for source_name in source_names[:-1]:
        count = round(total_chunks * source_target_counts[source_name] / total_targets)
        chunk_counts[source_name] = count
        assigned += count
    chunk_counts[source_names[-1]] = total_chunks - assigned

    source_entries: dict[str, list[dict[str, Any]]] = {}
    for source_number, source_name in enumerate(source_names):
        dataset = sources[source_name]
        sample_indices = _deterministic_samples(
            len(dataset), chunk_counts[source_name], seed + 1_000_003 * source_number
        )
        sample_indices = _ensure_capacity(
            dataset,
            sample_indices,
            int(source_target_counts[source_name]),
            seed + 2_000_003 * source_number,
        )
        capacities = [_valid_targets(dataset, index) for index in sample_indices]
        presentations = _allocate_targets(
            capacities, int(source_target_counts[source_name])
        )
        entries = []
        for sample_index, valid_targets in zip(sample_indices, presentations, strict=True):
            document_id, relative_start = _sample_identity(dataset, sample_index)
            entries.append(
                {
                    "source": source_name,
                    "sample_index": sample_index,
                    "document_id": document_id,
                    "relative_start": relative_start,
                    "valid_targets": valid_targets,
                }
            )
        source_entries[source_name] = entries

    # Deficit scheduling keeps cumulative target exposure close to the requested mix.
    positions = {name: 0 for name in source_names}
    cumulative = {name: 0 for name in source_names}
    entries: list[dict[str, Any]] = []
    for ordinal in range(total_chunks):
        available = [
            name
            for name in source_names
            if positions[name] < len(source_entries[name])
        ]
        source_name = max(
            available,
            key=lambda name: (
                (ordinal + 1) * source_target_counts[name] / total_chunks
                - cumulative[name],
                -source_names.index(name),
            ),
        )
        entry = source_entries[source_name][positions[source_name]]
        positions[source_name] += 1
        cumulative[source_name] += int(entry["valid_targets"])
        entries.append(
            {
                "ordinal": ordinal,
                "optimizer_step": ordinal
                // (batch_size * gradient_accumulation_steps)
                + 1,
                "microbatch": (ordinal // batch_size) % gradient_accumulation_steps,
                "batch_position": ordinal % batch_size,
                **entry,
            }
        )

    counts = {
        name: {
            "chunks": chunk_counts[name],
            "valid_targets": int(source_target_counts[name]),
            "valid_target_percentage": 100.0
            * int(source_target_counts[name])
            / total_targets,
        }
        for name in source_names
    }
    payload: dict[str, Any] = {
        "version": SCHEDULE_VERSION,
        "seed": seed,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_steps": max_steps,
        "sequence_length": next(iter(sources.values())).sequence_length,
        "total_chunks": total_chunks,
        "total_valid_targets": total_targets,
        "sources": counts,
        "source_artifacts": dict(source_artifacts or {}),
        "entries": entries,
    }
    payload["schedule_sha256"] = _canonical_hash(payload)
    return payload


def verify_schedule(schedule: Mapping[str, Any]) -> str:
    payload = dict(schedule)
    recorded_hash = str(payload.pop("schedule_sha256", ""))
    actual_hash = _canonical_hash(payload)
    if not recorded_hash or recorded_hash != actual_hash:
        raise RuntimeError("Mixed-source schedule SHA-256 is missing or invalid")
    if len(payload["entries"]) != int(payload["total_chunks"]):
        raise RuntimeError("Mixed-source schedule chunk count is inconsistent")
    observed: dict[str, int] = {name: 0 for name in payload["sources"]}
    observed_chunks: dict[str, int] = {name: 0 for name in payload["sources"]}
    for ordinal, entry in enumerate(payload["entries"]):
        if int(entry["ordinal"]) != ordinal:
            raise RuntimeError("Mixed-source schedule ordinals are not contiguous")
        observed[str(entry["source"])] += int(entry["valid_targets"])
        observed_chunks[str(entry["source"])] += 1
    for name, details in payload["sources"].items():
        if observed[name] != int(details["valid_targets"]):
            raise RuntimeError(f"Mixed-source target count is inconsistent for {name}")
        if observed_chunks[name] != int(details["chunks"]):
            raise RuntimeError(f"Mixed-source chunk count is inconsistent for {name}")
    if "pair_attachment" in payload:
        seen_presentations = 0
        seen_pairs: set[str] = set()
        for batch_start in range(0, len(payload["entries"]), int(payload["batch_size"])):
            microbatch_ids: list[str] = []
            for entry in payload["entries"][batch_start:batch_start + int(payload["batch_size"])]:
                for attachment in entry.get("pair_presentations", []):
                    microbatch_ids.append(str(attachment["pair_id"]))
                    seen_pairs.add(str(attachment["pair_id"]))
            if len(microbatch_ids) != len(set(microbatch_ids)):
                raise RuntimeError("Pair attachment is duplicated within a microbatch")
            seen_presentations += len(microbatch_ids)
        attachment = payload["pair_attachment"]
        if seen_presentations != int(attachment["pair_presentations"]):
            raise RuntimeError("Pair presentation exposure is inconsistent")
        if len(seen_pairs) != int(attachment["unique_pairs"]):
            raise RuntimeError("Unique attached-pair exposure is inconsistent")
    return actual_hash


class ScheduledMixedDataset(Dataset[tuple[Any, Any]]):
    """Resolve persisted schedule entries against immutable packed sources."""

    def __init__(
        self,
        sources: Mapping[str, PackedStoryDataset],
        schedule: Mapping[str, Any],
    ) -> None:
        self.sources = dict(sources)
        self.schedule = schedule
        verify_schedule(schedule)

    def __len__(self) -> int:
        return len(self.schedule["entries"])

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        entry = self.schedule["entries"][index]
        dataset = self.sources[str(entry["source"])]
        sample_index = int(entry["sample_index"])
        document_id, relative_start = _sample_identity(dataset, sample_index)
        if document_id != entry["document_id"] or relative_start != int(
            entry["relative_start"]
        ):
            raise RuntimeError("Scheduled chunk identity no longer matches its source")
        input_ids, labels = dataset[sample_index]
        valid_targets = int(entry["valid_targets"])
        available = int(labels.ne(-100).sum().item())
        if not 1 <= valid_targets <= available:
            raise RuntimeError("Scheduled valid-target count exceeds its source chunk")
        labels[valid_targets:] = -100
        return input_ids, labels


def _dataset(path: str, sequence_length: int, pad_token_id: int) -> PackedStoryDataset:
    bin_path = Path(path)
    return PackedStoryDataset(
        bin_path,
        bin_path.with_suffix(".index.json"),
        sequence_length=sequence_length,
        pad_token_id=pad_token_id,
    )


def build_schedule_from_config(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tokenizer_path = Path(config["tokenizer"]["path"])
    if sha256_file(tokenizer_path) != config["tokenizer"]["expected_sha256"]:
        raise RuntimeError("Locked tokenizer hash changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_token_id = tokenizer.token_to_id("<|pad|>")
    if pad_token_id is None:
        raise RuntimeError("Tokenizer has no pad token")
    sequence_length = int(config["model"]["max_seq_len"])
    source_config = config["mixed_data"]["training_sources"]
    sources = {
        name: _dataset(details["path"], sequence_length, int(pad_token_id))
        for name, details in source_config.items()
    }
    artifacts = {}
    for name, details in source_config.items():
        path = Path(details["path"])
        artifacts[name] = {
            "bin_path": str(path),
            "bin_sha256": sha256_file(path),
            "index_path": str(path.with_suffix(".index.json")),
            "index_sha256": sha256_file(path.with_suffix(".index.json")),
        }
    training = config["training"]
    schedule_config = config["mixed_data"]["schedule"]
    schedule = build_mixed_schedule(
        sources,
        source_target_counts={
            name: int(details["valid_targets"]) for name, details in source_config.items()
        },
        total_chunks=int(schedule_config["total_chunks"]),
        batch_size=int(training["batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        max_steps=int(training["max_steps"]),
        seed=int(config["seed"]),
        source_artifacts=artifacts,
    )
    attachment_sources = [
        (name, details)
        for name, details in source_config.items()
        if details.get("pair_attachment_index")
    ]
    if attachment_sources:
        if len(attachment_sources) != 1:
            raise ValueError("Exactly one mixed source may provide pair attachments")
        from src.pack_counterfactual_v1 import attach_pair_presentations

        name, details = attachment_sources[0]
        index_path = Path(details["pair_attachment_index"])
        if index_path != Path(details["path"]).with_suffix(".index.json"):
            raise RuntimeError("Pair attachment index must be the scheduled packed index")
        packed_index = json.loads(index_path.read_text(encoding="utf-8"))
        attach_pair_presentations(schedule, source_name=name, packed_index=packed_index)
        schedule["schedule_sha256"] = _canonical_hash(
            {key: value for key, value in schedule.items() if key != "schedule_sha256"}
        )
    output_path = Path(schedule_config["path"])
    return persist_schedule(output_path, schedule)


def persist_schedule(output_path: Path, schedule: dict[str, Any]) -> dict[str, Any]:
    """Write once, or accept an already persisted content-identical schedule."""
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        verify_schedule(existing)
        if existing == schedule:
            return existing
        raise FileExistsError(
            f"Refusing to overwrite a different mixed-source schedule: {output_path}"
        )
    write_json_atomic(output_path, schedule)
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    schedule = build_schedule_from_config(args.config)
    print(json.dumps({key: value for key, value in schedule.items() if key != "entries"}, indent=2))


if __name__ == "__main__":
    main()
