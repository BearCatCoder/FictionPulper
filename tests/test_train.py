import math
import unittest

from src.train import learning_rate_for_step


class TrainingScheduleTests(unittest.TestCase):
    def test_linear_warmup_and_cosine_decay(self):
        values = [
            learning_rate_for_step(
                step,
                max_steps=420,
                warmup_steps=21,
                learning_rate=1e-3,
                min_learning_rate=1e-4,
            )
            for step in range(1, 421)
        ]
        self.assertAlmostEqual(values[0], 1e-3 / 21)
        self.assertAlmostEqual(values[20], 1e-3)
        self.assertAlmostEqual(values[-1], 1e-4)
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertTrue(all(left >= right for left, right in zip(values[20:], values[21:])))

    def test_schedule_rejects_invalid_step(self):
        with self.assertRaises(ValueError):
            learning_rate_for_step(
                0,
                max_steps=420,
                warmup_steps=21,
                learning_rate=1e-3,
                min_learning_rate=1e-4,
            )


if __name__ == "__main__":
    unittest.main()
