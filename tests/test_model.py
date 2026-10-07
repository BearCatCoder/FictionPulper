import unittest

import torch

from src.model import FictionPulperLM, ModelConfig
from src.tiny_overfit import count_valid_predictions


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_seq_len=32,
    )


class ModelTests(unittest.TestCase):
    def test_forward_loss_gradients_and_shape(self):
        model = FictionPulperLM(tiny_config())
        input_ids = torch.randint(0, 128, (2, 16))
        labels = torch.randint(0, 128, (2, 16))
        logits, loss = model(input_ids, labels)
        self.assertEqual(logits.shape, (2, 16, 128))
        self.assertIsNotNone(loss)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_embedding_and_lm_head_weights_are_tied(self):
        model = FictionPulperLM(tiny_config())
        self.assertEqual(model.embed_tokens.weight.data_ptr(), model.lm_head.weight.data_ptr())

    def test_model_can_move_between_devices(self):
        model = FictionPulperLM(tiny_config()).to("cpu")
        self.assertEqual(model.embed_tokens.weight.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for device transfer test")
    def test_cuda_transfer_moves_rope_buffers_and_runs_forward(self):
        model = FictionPulperLM(tiny_config()).to("cuda")
        input_ids = torch.randint(0, 128, (2, 16), device="cuda")
        logits, _ = model(input_ids)
        self.assertEqual(logits.device.type, "cuda")
        for layer in model.layers:
            self.assertEqual(layer.attention.rope.cos.device.type, "cuda")
            self.assertEqual(layer.attention.rope.sin.device.type, "cuda")

        cpu_model = FictionPulperLM(tiny_config())
        cpu_logits, _ = cpu_model(torch.randint(0, 128, (1, 8)))
        self.assertEqual(cpu_logits.device.type, "cpu")

    def test_masked_padding_targets_do_not_change_loss(self):
        torch.manual_seed(11)
        model = FictionPulperLM(tiny_config()).eval()
        input_ids = torch.randint(0, 128, (1, 8))
        labels_a = torch.randint(0, 128, (1, 8))
        labels_b = labels_a.clone()
        labels_b[:, 5:] = torch.randint(0, 128, (1, 3))
        loss_mask = torch.tensor([[True, True, True, True, True, False, False, False]])
        _, loss_a = model(input_ids, labels_a, loss_mask)
        _, loss_b = model(input_ids, labels_b, loss_mask)
        torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)

    def test_valid_logits_cannot_attend_to_trailing_padding(self):
        torch.manual_seed(13)
        model = FictionPulperLM(tiny_config()).eval()
        first = torch.randint(0, 128, (1, 10))
        second = first.clone()
        second[:, 6:] = torch.randint(0, 128, (1, 4))
        with torch.no_grad():
            first_logits, _ = model(first)
            second_logits, _ = model(second)
        torch.testing.assert_close(first_logits[:, :6], second_logits[:, :6], rtol=0, atol=1e-6)

    def test_token_accuracy_excludes_padding_targets(self):
        logits = torch.zeros(1, 4, 3)
        labels = torch.tensor([[1, -100, 2, -100]])
        logits[0, 0, 1] = 10
        logits[0, 1, 0] = 10
        logits[0, 2, 2] = 10
        logits[0, 3, 0] = 10
        self.assertEqual(count_valid_predictions(logits, labels), (2, 2))

    def test_causal_mask_blocks_future_tokens(self):
        torch.manual_seed(7)
        model = FictionPulperLM(tiny_config()).eval()
        first = torch.randint(0, 128, (1, 12))
        second = first.clone()
        second[:, 6:] = torch.randint(0, 128, (1, 6))
        with torch.no_grad():
            first_logits, _ = model(first)
            second_logits, _ = model(second)
        torch.testing.assert_close(first_logits[:, :6], second_logits[:, :6], rtol=0, atol=1e-6)

    def test_smoke_model_parameter_count(self):
        config = ModelConfig(
            vocab_size=4096,
            hidden_size=256,
            num_layers=6,
            num_attention_heads=8,
            num_key_value_heads=2,
            intermediate_size=736,
            max_seq_len=1024,
        )
        model = FictionPulperLM(config)
        self.assertEqual(model.trainable_parameter_count(), 5_426_432)
        self.assertLessEqual(model.trainable_parameter_count(), 5_500_000)
        self.assertGreaterEqual(model.trainable_parameter_count(), 4_500_000)


if __name__ == "__main__":
    unittest.main()
