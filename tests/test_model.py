import tempfile
import unittest
from pathlib import Path

import torch

from src.model import FictionPulperLM, ModelConfig
from src.tiny_overfit import count_valid_predictions
from src.train import save_checkpoint


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


def context2k_tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=16,
        num_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        max_seq_len=2048,
    )


def model_50m_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=4096,
        hidden_size=512,
        num_layers=16,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=1536,
        max_seq_len=1024,
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

    def test_15m_model_parameter_count(self):
        config = ModelConfig(
            vocab_size=4096,
            hidden_size=384,
            num_layers=8,
            num_attention_heads=6,
            num_key_value_heads=2,
            intermediate_size=1120,
            max_seq_len=1024,
        )
        model = FictionPulperLM(config)
        self.assertEqual(model.trainable_parameter_count(), 15_047_040)
        self.assertEqual(model.embed_tokens.weight.data_ptr(), model.lm_head.weight.data_ptr())

    def test_2048_token_cpu_forward_and_loss(self):
        model = FictionPulperLM(context2k_tiny_config()).eval()
        input_ids = torch.randint(0, 64, (1, 2048))
        labels = torch.randint(0, 64, (1, 2048))
        logits, loss = model(input_ids, labels)
        self.assertEqual(logits.shape, (1, 2048, 64))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.layers[0].attention.rope.cos.shape[0], 2048)

    def test_causal_mask_blocks_future_tokens_at_2048(self):
        torch.manual_seed(17)
        model = FictionPulperLM(context2k_tiny_config()).eval()
        first = torch.randint(0, 64, (1, 2048))
        second = first.clone()
        second[:, 1024:] = torch.randint(0, 64, (1, 1024))
        with torch.no_grad():
            first_logits, _ = model(first)
            second_logits, _ = model(second)
        torch.testing.assert_close(
            first_logits[:, :1024], second_logits[:, :1024], rtol=0, atol=1e-6
        )

    def test_padding_loss_mask_at_2048(self):
        torch.manual_seed(19)
        model = FictionPulperLM(context2k_tiny_config()).eval()
        input_ids = torch.randint(0, 64, (1, 2048))
        labels_a = torch.randint(0, 64, (1, 2048))
        labels_b = labels_a.clone()
        labels_b[:, 1536:] = torch.randint(0, 64, (1, 512))
        loss_mask = torch.arange(2048).unsqueeze(0) < 1536
        _, loss_a = model(input_ids, labels_a, loss_mask)
        _, loss_b = model(input_ids, labels_b, loss_mask)
        torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)

    def test_2048_checkpoint_save_and_reload(self):
        torch.manual_seed(23)
        config = context2k_tiny_config()
        model = FictionPulperLM(config).eval()
        input_ids = torch.randint(0, 64, (1, 32))
        with torch.no_grad():
            expected, _ = model(input_ids)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context2k.pt"
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                model_config=config,
                config={"model": config.__dict__},
                epoch=1,
                step=1,
                validation_loss=4.0,
                best_validation_loss=4.0,
                tokenizer_hash="test",
                metrics=[],
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            restored = FictionPulperLM(ModelConfig(**payload["model_config"])).eval()
            restored.load_state_dict(payload["model"])
            self.assertEqual(payload["epoch"], 1)
            self.assertEqual(payload["step"], 1)
            with torch.no_grad():
                actual, _ = restored(input_ids)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for context2k test")
    def test_2048_token_cuda_forward_and_rope_device(self):
        model = FictionPulperLM(context2k_tiny_config()).to("cuda").eval()
        input_ids = torch.randint(0, 64, (1, 2048), device="cuda")
        with torch.no_grad():
            logits, _ = model(input_ids)
        self.assertEqual(logits.shape, (1, 2048, 64))
        self.assertEqual(logits.device.type, "cuda")
        self.assertEqual(model.layers[0].attention.rope.cos.device.type, "cuda")
        self.assertEqual(model.layers[0].attention.rope.sin.device.type, "cuda")

    def test_15m_context2k_parameter_count_is_unchanged(self):
        config = ModelConfig(
            vocab_size=4096,
            hidden_size=384,
            num_layers=8,
            num_attention_heads=6,
            num_key_value_heads=2,
            intermediate_size=1120,
            max_seq_len=2048,
        )
        self.assertEqual(FictionPulperLM(config).trainable_parameter_count(), 15_047_040)

    def test_50m_cpu_forward_masks_gradients_tying_and_checkpoint(self):
        torch.manual_seed(29)
        config = model_50m_config()
        model = FictionPulperLM(config)
        self.assertEqual(model.trainable_parameter_count(), 50_348_544)
        self.assertEqual(model.embed_tokens.weight.data_ptr(), model.lm_head.weight.data_ptr())
        self.assertEqual(model.layers[0].attention.rope.cos.shape[0], 1024)

        input_ids = torch.randint(0, config.vocab_size, (1, 12))
        labels = torch.randint(0, config.vocab_size, (1, 12))
        logits, loss = model(input_ids, labels)
        self.assertEqual(logits.shape, (1, 12, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )

        model.eval()
        first = torch.randint(0, config.vocab_size, (1, 16))
        second = first.clone()
        second[:, 8:] = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            first_logits, _ = model(first)
            second_logits, _ = model(second)
        torch.testing.assert_close(
            first_logits[:, :8], second_logits[:, :8], rtol=0, atol=1e-6
        )

        labels_b = labels.clone()
        labels_b[:, 8:] = torch.randint(0, config.vocab_size, (1, 4))
        loss_mask = torch.arange(12).unsqueeze(0) < 8
        _, loss_a = model(input_ids, labels, loss_mask)
        _, loss_b = model(input_ids, labels_b, loss_mask)
        torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model-50m.pt"
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                model_config=config,
                config={"model": config.__dict__},
                epoch=0,
                step=0,
                validation_loss=8.3,
                best_validation_loss=8.3,
                tokenizer_hash="test",
                metrics=[],
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            restored = FictionPulperLM(ModelConfig(**payload["model_config"])).eval()
            restored.load_state_dict(payload["model"])
            with torch.no_grad():
                restored_logits, _ = restored(first)
        torch.testing.assert_close(restored_logits, first_logits, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for 50M test")
    def test_50m_cuda_forward_and_rope_device(self):
        config = model_50m_config()
        model = FictionPulperLM(config).to("cuda").eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 64), device="cuda")
        with torch.no_grad():
            logits, _ = model(input_ids)
        self.assertEqual(logits.shape, (1, 64, config.vocab_size))
        self.assertEqual(logits.device.type, "cuda")
        for layer in model.layers:
            self.assertEqual(layer.attention.rope.cos.device.type, "cuda")
            self.assertEqual(layer.attention.rope.sin.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
