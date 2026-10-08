import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dataset import PackedStoryDataset
from src.tokenize_corpus import chunk_statistics, derive_schedule


class PackedDatasetTests(unittest.TestCase):
    def test_sequences_do_not_cross_documents_and_padding_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            bin_path = directory_path / "train.bin"
            index_path = directory_path / "train.index.json"
            np.asarray([1, 10, 11, 2, 1, 20, 2], dtype=np.uint16).tofile(bin_path)
            index_path.write_text(
                json.dumps(
                    {
                        "documents": [
                            {"id": "a", "offset": 0, "length": 4},
                            {"id": "b", "offset": 4, "length": 3},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            dataset = PackedStoryDataset(
                bin_path, index_path, sequence_length=3, pad_token_id=0
            )
            self.assertEqual(len(dataset), 2)
            first_inputs, first_labels = dataset[0]
            second_inputs, second_labels = dataset[1]
            self.assertEqual(first_inputs.tolist(), [1, 10, 11])
            self.assertEqual(first_labels.tolist(), [10, 11, 2])
            self.assertEqual(second_inputs.tolist(), [1, 20, 0])
            self.assertEqual(second_labels.tolist(), [20, 2, -100])

    def test_schedule_uses_actual_sequence_count(self):
        schedule = derive_schedule(
            {
                "token_count": 100_000,
                "valid_target_token_count": 99_000,
                "sequence_count": 125,
            },
            batch_size=16,
            sequence_length=1024,
            gradient_accumulation_steps=4,
            target_epochs=15,
        )
        self.assertEqual(schedule["allocated_token_slots_per_microbatch"], 16_384)
        self.assertEqual(schedule["allocated_token_slots_per_optimizer_step"], 65_536)
        self.assertEqual(schedule["optimizer_steps_per_epoch"], 2)
        self.assertEqual(schedule["recommended_max_steps"], 30)
        self.assertEqual(schedule["actual_corpus_passes"], 15)

    def test_schedule_uses_requested_epoch_count(self):
        schedule = derive_schedule(
            {"token_count": 100, "valid_target_token_count": 90, "sequence_count": 3},
            batch_size=2,
            sequence_length=16,
            gradient_accumulation_steps=2,
            target_epochs=10,
        )
        self.assertEqual(schedule["document_chunk_epochs"], 10)
        self.assertEqual(schedule["recommended_max_steps"], 10)
        self.assertEqual(schedule["recommended_warmup_steps"], 1)

    def test_schedule_candidate_step_counts_scale_by_epoch(self):
        candidates = {
            epochs: derive_schedule(
                {"token_count": 100, "valid_target_token_count": 90, "sequence_count": 125},
                batch_size=16,
                sequence_length=1024,
                gradient_accumulation_steps=4,
                target_epochs=epochs,
            )
            for epochs in (3, 5, 7, 10)
        }
        self.assertEqual(
            [candidates[epochs]["recommended_max_steps"] for epochs in (3, 5, 7, 10)],
            [6, 10, 14, 20],
        )

    def test_document_exclusion_removes_only_requested_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_path = root / "validation.bin"
            np.asarray([1, 2, 3, 4, 5, 6], dtype=np.uint16).tofile(bin_path)
            index_path = root / "validation.index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "documents": [
                            {"id": "keep", "offset": 0, "length": 3},
                            {"id": "exclude", "offset": 3, "length": 3},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            dataset = PackedStoryDataset(
                bin_path,
                index_path,
                sequence_length=2,
                pad_token_id=0,
                exclude_document_ids={"exclude"},
            )
            self.assertEqual(len(dataset), 1)
            inputs, labels = dataset[0]
            self.assertEqual(inputs.tolist(), [1, 2])
            self.assertEqual(labels.tolist(), [2, 3])

            with self.assertRaisesRegex(ValueError, "absent from the index"):
                PackedStoryDataset(
                    bin_path,
                    index_path,
                    sequence_length=2,
                    pad_token_id=0,
                    exclude_document_ids={"missing"},
                )

    def test_context2k_chunk_statistics(self):
        stats = chunk_statistics([100, 2050, 4098, 10242], sequence_length=2048)
        self.assertEqual(stats["median"], 2.5)
        self.assertEqual(stats["percentile_95"], 6)
        self.assertEqual(stats["maximum"], 6)
        self.assertEqual(stats["distribution"]["1"]["count"], 1)
        self.assertEqual(stats["distribution"]["2"]["count"], 1)
        self.assertEqual(stats["distribution"]["3-4"]["count"], 1)
        self.assertEqual(stats["distribution"]["5+"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
