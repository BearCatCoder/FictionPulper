"""FictionPulper decoder-only Transformer implemented directly in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_seq_len: int
    attention_bias: bool = False
    mlp_bias: bool = False
    rms_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float() * torch.rsqrt(
            hidden_states.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(input_dtype)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequency)
        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = query.shape[-2]
        cos = self.cos[:sequence_length].to(dtype=query.dtype)[None, None, :, :]
        sin = self.sin[:sequence_length].to(dtype=query.dtype)[None, None, :, :]
        return self.apply_rotary(query, cos, sin), self.apply_rotary(key, cos, sin)

    @staticmethod
    def apply_rotary(
        tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.groups = self.num_heads // self.num_key_value_heads
        key_value_size = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(config.hidden_size, key_value_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, key_value_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.rope = RotaryPositionEmbedding(self.head_dim, config.max_seq_len)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        query, key = self.rope(query, key)
        key = key.repeat_interleave(self.groups, dim=1)
        value = value.repeat_interleave(self.groups, dim=1)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.num_heads * self.head_dim
        )
        return self.o_proj(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(self.attention_norm(hidden_states))
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class FictionPulperLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("Input sequence exceeds max_seq_len")
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        logits = self.lm_head(self.final_norm(hidden_states))
        loss = None
        if labels is not None:
            loss_labels = labels
            if loss_mask is not None:
                if loss_mask.shape != labels.shape:
                    raise ValueError("loss_mask must have the same shape as labels")
                loss_labels = labels.masked_fill(~loss_mask.bool(), -100)
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                loss_labels.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def model_config_from_dict(values: dict[str, object]) -> ModelConfig:
    fields = ModelConfig.__dataclass_fields__
    return ModelConfig(**{key: value for key, value in values.items() if key in fields})  # type: ignore[arg-type]
