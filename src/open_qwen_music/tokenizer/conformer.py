
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

POSITION_ENCODINGS = ("rope", "sinusoidal")


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[1], device=x.device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32)
            * (-math.log(10_000.0) / self.dim)
        )
        encoding = torch.zeros(x.shape[1], self.dim, device=x.device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        return x + encoding.to(dtype=x.dtype).unsqueeze(0)


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, received {head_dim}")
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(
        self, length: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(length, device=device, dtype=torch.float32)
        angles = torch.outer(positions, self.inverse_frequency.to(device))
        emb = torch.cat([angles, angles], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to ``x`` using time-indexed cosine and sine values."""

    return x * cos + _rotate_half(x) * sin


class SelfAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be able to be num_heads={num_heads} Divisible by")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * dim))
        self.out_proj = nn.Linear(dim, dim)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)

        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        frame_mask: torch.Tensor,
        causal: bool,
        use_rotary: bool = True,
    ) -> torch.Tensor:
        batch, time, dim = x.shape
        projected = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        query, key, value = projected.chunk(3, dim=-1)
        shape = (batch, time, self.num_heads, self.head_dim)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
        if use_rotary:
            cos, sin = self.rotary(time, x.device, query.dtype)
            query = apply_rotary(query, cos, sin)
            key = apply_rotary(key, cos, sin)
        dropout = self.dropout if self.training else 0.0
        if causal:
            attended = F.scaled_dot_product_attention(
                query, key, value, dropout_p=dropout, is_causal=True
            )
        else:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=frame_mask.view(batch, 1, 1, time),
                dropout_p=dropout,
            )
        attended = attended.transpose(1, 2).reshape(batch, time, dim)
        return self.out_proj(attended)


class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Conformer The convolution kernel must be an odd number")
        self.kernel_size = kernel_size
        self.norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, 2 * dim, 1)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, groups=dim)
        self.channel_norm = nn.LayerNorm(dim)
        self.pointwise_out = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = self.norm(x).transpose(1, 2)
        y = F.glu(self.pointwise_in(y), dim=1)
        total = self.kernel_size - 1
        padding = (total, 0) if causal else (total // 2, total - total // 2)


        if frame_mask is not None:
            y = y.masked_fill(~frame_mask.unsqueeze(1), 0.0)
        y = self.depthwise(F.pad(y, padding))
        y = F.silu(self.channel_norm(y.transpose(1, 2)).transpose(1, 2))
        y = self.dropout(self.pointwise_out(y))
        return y.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        conv_kernel: int,
        dropout: float,
        use_rotary: bool = True,
    ) -> None:
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, ffn_dim, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConformerConvModule(dim, conv_kernel, dropout)
        self.ffn2 = FeedForwardModule(dim, ffn_dim, dropout)
        self.final_norm = nn.LayerNorm(dim)
        self.use_rotary = use_rotary

    def forward(
        self,
        x: torch.Tensor,
        frame_mask: torch.Tensor,
        causal: bool | None = None,
        *,
        attention_causal: bool | None = None,
        convolution_causal: bool | None = None,
    ) -> torch.Tensor:
        fallback = bool(causal) if causal is not None else False
        attention_causal = (
            fallback if attention_causal is None else bool(attention_causal)
        )
        convolution_causal = (
            fallback if convolution_causal is None else bool(convolution_causal)
        )
        x = x + 0.5 * self.ffn1(x)
        y = self.attn(
            self.attn_norm(x),
            frame_mask=frame_mask,
            causal=attention_causal,
            use_rotary=self.use_rotary,
        )
        x = x + self.attn_dropout(y)
        x = x + self.conv(
            x, causal=convolution_causal, frame_mask=frame_mask
        )
        x = self.final_norm(x + 0.5 * self.ffn2(x))
        return x.masked_fill(~frame_mask.unsqueeze(-1), 0.0)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        conv_kernel: int,
        dropout: float,
        position_encoding: str = "rope",
    ) -> None:
        super().__init__()
        if position_encoding not in POSITION_ENCODINGS:
            raise ValueError(
                f"position_encoding must be one of {POSITION_ENCODINGS}; "
                f"received {position_encoding!r}"
            )
        self.position_encoding = position_encoding
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    dim,
                    num_heads,
                    ffn_dim,
                    conv_kernel,
                    dropout,
                    use_rotary=position_encoding == "rope",
                )
                for _ in range(num_layers)
            ]
        )

    def forward_range(
        self,
        x: torch.Tensor,
        frame_mask: torch.Tensor,
        causal: bool | None = None,
        start: int = 0,
        end: int | None = None,
        *,
        attention_causal: bool | None = None,
        convolution_causal: bool | None = None,
    ) -> torch.Tensor:
        fallback = bool(causal) if causal is not None else False
        attention_causal = (
            fallback if attention_causal is None else bool(attention_causal)
        )
        convolution_causal = (
            fallback if convolution_causal is None else bool(convolution_causal)
        )
        for layer in self.layers[start:end]:
            x = layer(
                x,
                frame_mask=frame_mask,
                attention_causal=attention_causal,
                convolution_causal=convolution_causal,
            )
        return x
