"""Stable Audio 3 backbone adapted to the Renderer interface.

The implementation covers the architecture used by the released checkpoint:
128-channel 25 Hz latents, 24 transformer layers at width 1536, differential
attention, QK RMSNorm, SwiGLU, text cross-attention, frame-aligned semantic
conditioning, loudness AdaLN, and 64 memory tokens.

Parameter names and forward operations follow
``Stability-AI/stable-audio-3@a0b57f5483c4588f827f3552b7d5c6ca2a9687be``.
See ``third_party/NOTICE-STABLE_AUDIO_3.md`` for attribution and licensing.
"""

from __future__ import annotations

import copy
import math
from functools import reduce
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .dit import DiTConfig

try:  # pragma: no cover - depends on the runtime environment.
    from flash_attn import flash_attn_func
except ImportError:  # pragma: no cover - CPU and non-FlashAttention runtimes use SDPA.
    flash_attn_func = None


SA3_SOURCE_COMMIT = "a0b57f5483c4588f827f3552b7d5c6ca2a9687be"
SA3_ARCHITECTURE_TYPE = "stable_audio_3"


class ExpoFourierFeatures(nn.Module):
    """Exponential-frequency timestep features used by Stable Audio 3."""

    def __init__(
        self,
        dim: int,
        min_freq: float = 0.5,
        max_freq: float = 10_000.0,
    ) -> None:
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("ExpoFourierFeatures dim must be a positive even integer")
        self.dim = int(dim)
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        input_dtype = timestep.dtype
        value = timestep.float()
        if value.ndim == 1:
            value = value.unsqueeze(-1)
        half = self.dim // 2
        ramp = torch.linspace(
            0,
            1,
            half,
            device=value.device,
            dtype=torch.float32,
        )
        frequencies = torch.exp(
            ramp * (math.log(self.max_freq) - math.log(self.min_freq)) + math.log(self.min_freq)
        )
        arguments = value * frequencies * (2 * math.pi)
        embedding = torch.cat(
            [torch.cos(arguments), torch.sin(arguments)],
            dim=-1,
        )
        return embedding.to(input_dtype)


class SA3RMSNorm(nn.Module):
    """RMSNorm with the ``gamma`` parameter name used by Stable Audio 3."""

    def __init__(
        self,
        dim: int,
        *,
        force_fp32: bool = False,
        eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)
        self.force_fp32 = bool(force_fp32)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.force_fp32:
            return F.rms_norm(
                value,
                value.shape[-1:],
                weight=self.gamma,
                eps=self.eps,
            )
        result = F.rms_norm(
            value.float(),
            value.shape[-1:],
            weight=self.gamma.float(),
            eps=self.eps,
        )
        return result.to(value.dtype)


class RotaryEmbedding(nn.Module):
    """Partial RoPE with the checkpoint-compatible ``inv_freq`` buffer."""

    def __init__(
        self,
        dim: int,
        *,
        interpolation_factor: float = 1.0,
        base: float = 10_000.0,
        base_rescale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if dim <= 2 or dim % 2:
            raise ValueError("RoPE dim must be an even integer greater than 2")
        base *= base_rescale_factor ** (dim / (dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        if interpolation_factor < 1.0:
            raise ValueError("RoPE interpolation_factor must be at least 1")
        self.interpolation_factor = float(interpolation_factor)

    def forward_from_seq_len(
        self,
        sequence_length: int,
    ) -> tuple[torch.Tensor, float]:
        positions = torch.arange(sequence_length, device=self.inv_freq.device)
        return self.forward(positions)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        value = positions.to(torch.float32) / self.interpolation_factor
        frequencies = torch.einsum("i,j->ij", value, self.inv_freq)
        frequencies = torch.cat((frequencies, frequencies), dim=-1)
        return frequencies, 1.0


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.reshape(*value.shape[:-1], 2, -1).unbind(dim=-2)
    return torch.cat((-second, first), dim=-1)


@torch.amp.autocast("cuda", enabled=False)
def _apply_rotary_pos_emb(
    value: torch.Tensor,
    frequencies: torch.Tensor,
    scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    output_dtype = value.dtype
    compute_dtype = reduce(
        torch.promote_types,
        (value.dtype, frequencies.dtype, torch.float32),
    )
    value = value.to(compute_dtype)
    frequencies = frequencies.to(compute_dtype)[-value.shape[-2] :, :]
    rotary_dim = frequencies.shape[-1]
    rotated, unrotated = value[..., :rotary_dim], value[..., rotary_dim:]
    rotated = (
        rotated * frequencies.cos() * scale + _rotate_half(rotated) * frequencies.sin() * scale
    )
    return torch.cat((rotated.to(output_dtype), unrotated.to(output_dtype)), dim=-1)


class SA3GLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        projected, gate = self.proj(value).chunk(2, dim=-1)
        return projected * self.act(gate)


class SA3FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        mult: float = 4.0,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        linear_out = nn.Linear(inner_dim, dim)
        if zero_init_output:
            nn.init.zeros_(linear_out.weight)
            nn.init.zeros_(linear_out.bias)
        self.ff = nn.Sequential(
            SA3GLU(dim, inner_dim),
            nn.Identity(),
            linear_out,
        )

    def forward(
        self,
        value: torch.Tensor,
        *,
        varlen_metadata: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        if varlen_metadata is not None:
            raise ValueError("Padded variable-length batches are not supported")
        return self.ff(value)


def _reshape_heads(
    value: torch.Tensor,
    *,
    heads: int,
) -> torch.Tensor:
    batch, length, width = value.shape
    if width % heads:
        raise RuntimeError("Attention width must be divisible by the number of heads")
    return value.reshape(batch, length, heads, width // heads).permute(0, 2, 1, 3).contiguous()


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    if query.shape[1] != key.shape[1]:
        if query.shape[1] % key.shape[1]:
            raise RuntimeError("Grouped-query attention head counts are incompatible")
        repeat = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)

    if flash_attn_func is not None and query.device.type == "cuda":
        input_dtype = query.dtype
        query, key, value = (
            tensor.permute(0, 2, 1, 3).contiguous() for tensor in (query, key, value)
        )
        if input_dtype not in {torch.float16, torch.bfloat16}:
            query, key, value = (tensor.to(torch.float16) for tensor in (query, key, value))
        result = flash_attn_func(
            query,
            key,
            value,
            causal=causal,
            window_size=(-1, -1),
        )
        return result.to(input_dtype).permute(0, 2, 1, 3).contiguous()
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        is_causal=causal,
    )


class SA3Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        dim_heads: int = 64,
        dim_context: int | None = None,
        causal: bool = False,
        zero_init_output: bool = True,
        qk_norm_eps: float = 1.0e-6,
        qk_norm: str = "none",
        differential: bool = False,
    ) -> None:
        super().__init__()
        if qk_norm not in {"none", "rms"}:
            raise ValueError("qk_norm must be none or rms")
        self.dim = int(dim)
        self.dim_heads = int(dim_heads)
        self.differential = bool(differential)
        context_dim = self.dim if dim_context is None else int(dim_context)
        self.num_heads = self.dim // self.dim_heads
        self.kv_heads = context_dim // self.dim_heads
        if dim_context is not None:
            if self.differential:
                self.to_q = nn.Linear(self.dim, self.dim * 2, bias=False)
                self.to_kv = nn.Linear(context_dim, context_dim * 3, bias=False)
            else:
                self.to_q = nn.Linear(self.dim, self.dim, bias=False)
                self.to_kv = nn.Linear(context_dim, context_dim * 2, bias=False)
        elif self.differential:
            self.to_qkv = nn.Linear(self.dim, self.dim * 5, bias=False)
        else:
            self.to_qkv = nn.Linear(self.dim, self.dim * 3, bias=False)
        self.to_out = nn.Linear(self.dim, self.dim, bias=False)
        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)
        self.qk_norm = qk_norm
        self.qk_norm_eps = float(qk_norm_eps)
        if self.qk_norm == "rms":
            self.q_norm = SA3RMSNorm(self.dim_heads, eps=self.qk_norm_eps)
            self.k_norm = SA3RMSNorm(self.dim_heads, eps=self.qk_norm_eps)
        self.causal = bool(causal)

    def _normalize(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.qk_norm == "none":
            return query, key
        query_dtype = query.dtype
        key_dtype = key.dtype
        return (
            self.q_norm(query).to(query_dtype),
            self.k_norm(key).to(key_dtype),
        )

    def forward(
        self,
        value: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        rotary_pos_emb: tuple[torch.Tensor, float] | None = None,
        rotary_pos_emb_k: tuple[torch.Tensor, float] | None = None,
        causal: bool | None = None,
        **_: Any,
    ) -> torch.Tensor:
        heads = self.num_heads
        kv_heads = self.kv_heads
        source = value if context is None else context
        if hasattr(self, "to_q"):
            if self.differential:
                query, query_diff = self.to_q(value).chunk(2, dim=-1)
                query = torch.stack(
                    (
                        _reshape_heads(query, heads=heads),
                        _reshape_heads(query_diff, heads=heads),
                    ),
                    dim=1,
                )
                key, key_diff, values = self.to_kv(source).chunk(3, dim=-1)
                key = torch.stack(
                    (
                        _reshape_heads(key, heads=kv_heads),
                        _reshape_heads(key_diff, heads=kv_heads),
                    ),
                    dim=1,
                )
                values = _reshape_heads(values, heads=kv_heads)
            else:
                query = _reshape_heads(self.to_q(value), heads=heads)
                key, values = self.to_kv(source).chunk(2, dim=-1)
                key = _reshape_heads(key, heads=kv_heads)
                values = _reshape_heads(values, heads=kv_heads)
        elif self.differential:
            query, key, values, query_diff, key_diff = self.to_qkv(value).chunk(
                5,
                dim=-1,
            )
            query = torch.stack(
                (
                    _reshape_heads(query, heads=heads),
                    _reshape_heads(query_diff, heads=heads),
                ),
                dim=1,
            )
            key = torch.stack(
                (
                    _reshape_heads(key, heads=heads),
                    _reshape_heads(key_diff, heads=heads),
                ),
                dim=1,
            )
            values = _reshape_heads(values, heads=heads)
        else:
            query, key, values = self.to_qkv(value).chunk(3, dim=-1)
            query = _reshape_heads(query, heads=heads)
            key = _reshape_heads(key, heads=heads)
            values = _reshape_heads(values, heads=heads)

        query, key = self._normalize(query, key)
        if rotary_pos_emb is not None:
            frequencies, _ = rotary_pos_emb
            query_dtype, key_dtype = query.dtype, key.dtype
            query_frequencies = frequencies.float()
            key_frequencies = (
                rotary_pos_emb_k[0].float() if rotary_pos_emb_k is not None else query_frequencies
            )
            if rotary_pos_emb_k is None:
                if query.shape[-2] >= key.shape[-2]:
                    key_frequencies = (query.shape[-2] / key.shape[-2]) * query_frequencies
                else:
                    query_frequencies = (key.shape[-2] / query.shape[-2]) * query_frequencies
            query = _apply_rotary_pos_emb(
                query.float(),
                query_frequencies,
            ).to(query_dtype)
            key = _apply_rotary_pos_emb(
                key.float(),
                key_frequencies,
            ).to(key_dtype)
            query = query.to(values.dtype)
            key = key.to(values.dtype)

        resolved_causal = self.causal if causal is None else bool(causal)
        if query.shape[-2] == 1 and resolved_causal:
            resolved_causal = False
        if self.differential:
            query, query_diff = query.unbind(dim=1)
            key, key_diff = key.unbind(dim=1)
            attended = _attention(
                query,
                key,
                values,
                causal=resolved_causal,
            ) - _attention(
                query_diff,
                key_diff,
                values,
                causal=resolved_causal,
            )
        else:
            attended = _attention(
                query,
                key,
                values,
                causal=resolved_causal,
            )
        merged = attended.permute(0, 2, 1, 3).reshape(value.shape[0], value.shape[1], self.dim)
        return self.to_out(merged)


def _left_pad_to_match(
    embedding: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    length = embedding.shape[-2]
    if length < target_length:
        return F.pad(
            embedding,
            (0, 0, target_length - length, 0),
            value=0.0,
        )
    if length > target_length:
        return embedding[:, -target_length:, :]
    return embedding


class SA3TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        dim_heads: int,
        dim_context: int,
        global_cond_dim: int,
        local_add_cond_dim: int,
        causal: bool = False,
        zero_init_branch_outputs: bool = True,
        norm_type: str = "rms_norm",
        attn_kwargs: Mapping[str, Any] | None = None,
        ff_kwargs: Mapping[str, Any] | None = None,
        norm_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if norm_type != "rms_norm":
            raise ValueError("norm_type must be rms_norm")
        norm_options = dict(norm_kwargs or {})
        attention_options = dict(attn_kwargs or {})
        feedforward_options = dict(ff_kwargs or {})
        self.dim = int(dim)
        self.dim_heads = min(int(dim_heads), self.dim)
        self.cross_attend = True
        self.pre_norm = SA3RMSNorm(self.dim, **norm_options)
        self.self_attn = SA3Attention(
            self.dim,
            dim_heads=self.dim_heads,
            causal=causal,
            zero_init_output=zero_init_branch_outputs,
            **attention_options,
        )
        self.cross_attend_norm = SA3RMSNorm(self.dim, **norm_options)
        self.cross_attn = SA3Attention(
            self.dim,
            dim_heads=self.dim_heads,
            dim_context=dim_context,
            causal=causal,
            zero_init_output=zero_init_branch_outputs,
            **attention_options,
        )
        self.ff_norm = SA3RMSNorm(self.dim, **norm_options)
        self.ff = SA3FeedForward(
            self.dim,
            zero_init_output=zero_init_branch_outputs,
            **feedforward_options,
        )
        self.global_cond_dim = int(global_cond_dim)
        self.to_scale_shift_gate = nn.Parameter(torch.randn(6 * self.dim) / self.dim**0.5)
        self.local_add_cond_dim = int(local_add_cond_dim)
        self.to_local_embed = nn.Sequential(
            nn.Linear(self.local_add_cond_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        nn.init.zeros_(self.to_local_embed[-1].weight)
        nn.init.zeros_(self.to_local_embed[-1].bias)

    def forward(
        self,
        value: torch.Tensor,
        *,
        context: torch.Tensor,
        global_cond: torch.Tensor,
        local_add_cond: torch.Tensor,
        rotary_pos_emb: tuple[torch.Tensor, float],
        **_: Any,
    ) -> torch.Tensor:
        (
            scale_self,
            shift_self,
            gate_self,
            scale_ff,
            shift_ff,
            gate_ff,
        ) = (
            (self.to_scale_shift_gate + global_cond)
            .unsqueeze(1)
            .chunk(
                6,
                dim=-1,
            )
        )
        residual = value
        hidden = self.pre_norm(value)
        hidden = hidden * (1 + scale_self) + shift_self
        hidden = self.self_attn(hidden, rotary_pos_emb=rotary_pos_emb)
        hidden = hidden * torch.sigmoid(1 - gate_self)
        value = hidden + residual
        value = value + self.cross_attn(
            self.cross_attend_norm(value),
            context=context,
        )
        local = self.to_local_embed(local_add_cond)
        value = value + _left_pad_to_match(local, value.shape[-2])
        residual = value
        hidden = self.ff_norm(value)
        hidden = hidden * (1 + scale_ff) + shift_ff
        hidden = self.ff(hidden)
        hidden = hidden * torch.sigmoid(1 - gate_ff)
        return hidden + residual


class SA3ContinuousTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_in: int,
        dim_out: int,
        dim_heads: int,
        cond_token_dim: int,
        global_cond_dim: int,
        local_add_cond_dim: int,
        num_memory_tokens: int,
        attn_kwargs: Mapping[str, Any],
        norm_type: str,
        norm_kwargs: Mapping[str, Any],
        ff_kwargs: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.depth = int(depth)
        self.causal = False
        self.layers = nn.ModuleList([])
        self.project_in = nn.Linear(dim_in, dim, bias=False)
        self.project_out = nn.Linear(dim, dim_out, bias=False)
        self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        self.num_memory_tokens = int(num_memory_tokens)
        self.memory_tokens = nn.Parameter(torch.randn(self.num_memory_tokens, self.dim))
        self.global_cond_embedder = nn.Sequential(
            nn.Linear(global_cond_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )
        for _ in range(self.depth):
            self.layers.append(
                SA3TransformerBlock(
                    dim,
                    dim_heads=dim_heads,
                    dim_context=cond_token_dim,
                    global_cond_dim=global_cond_dim,
                    local_add_cond_dim=local_add_cond_dim,
                    causal=False,
                    zero_init_branch_outputs=True,
                    norm_type=norm_type,
                    attn_kwargs=attn_kwargs,
                    ff_kwargs=ff_kwargs,
                    norm_kwargs=norm_kwargs,
                )
            )

    def forward(
        self,
        value: torch.Tensor,
        *,
        context: torch.Tensor,
        global_cond: torch.Tensor,
        local_add_cond: torch.Tensor,
        use_checkpointing: bool = False,
        checkpoint_block_interval: int = 1,
        **_: Any,
    ) -> torch.Tensor:
        hidden = self.project_in(value)
        memory = self.memory_tokens.expand(hidden.shape[0], -1, -1)
        hidden = torch.cat((memory, hidden), dim=1)
        rotary = self.rotary_pos_emb.forward_from_seq_len(hidden.shape[1])
        global_condition = self.global_cond_embedder(global_cond)
        for index, layer in enumerate(self.layers):
            checkpoint_layer = (
                use_checkpointing
                and self.training
                and torch.is_grad_enabled()
                and index % checkpoint_block_interval == 0
            )
            if checkpoint_layer:
                hidden = checkpoint(
                    layer,
                    hidden,
                    context=context,
                    global_cond=global_condition,
                    local_add_cond=local_add_cond,
                    rotary_pos_emb=rotary,
                    use_reentrant=False,
                )
            else:
                hidden = layer(
                    hidden,
                    context=context,
                    global_cond=global_condition,
                    local_add_cond=local_add_cond,
                    rotary_pos_emb=rotary,
                )
        hidden = hidden[:, self.num_memory_tokens :, :]
        return self.project_out(hidden)


class SA3DiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        io_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        cond_token_dim: int,
        global_cond_dim: int,
        local_add_cond_dim: int,
        global_cond_type: str,
        timestep_features_type: str,
        diffusion_objective: str,
        attn_kwargs: Mapping[str, Any],
        norm_type: str,
        norm_kwargs: Mapping[str, Any],
        ff_kwargs: Mapping[str, Any],
        num_memory_tokens: int,
        **extra: Any,
    ) -> None:
        super().__init__()
        unsupported = {
            name: value
            for name, value in extra.items()
            if value
            not in {
                None,
                0,
                1,
                False,
                "continuous_transformer",
                "global",
            }
        }
        if unsupported:
            raise ValueError(f"Unsupported native architecture fields: {unsupported}")
        if global_cond_type != "adaLN":
            raise ValueError("global_cond_type must be adaLN")
        if timestep_features_type != "expo":
            raise ValueError("timestep_features_type must be expo")
        if diffusion_objective != "rectified_flow":
            raise ValueError("diffusion_objective must be rectified_flow")
        self.cond_token_dim = int(cond_token_dim)
        self.timestep_cond_type = "global"
        self.timestep_features_logsnr = False
        self.timestep_features = ExpoFourierFeatures(256, 0.5, 10_000.0)
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(256, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )
        self.diffusion_objective = diffusion_objective
        self.to_cond_embed = nn.Sequential(
            nn.Linear(cond_token_dim, embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )
        self.to_global_embed = nn.Sequential(
            nn.Linear(global_cond_dim, embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )
        self.input_concat_dim = 0
        self.patch_size = 1
        self.transformer_type = "continuous_transformer"
        self.global_cond_type = global_cond_type
        self.transformer = SA3ContinuousTransformer(
            dim=embed_dim,
            depth=depth,
            dim_heads=embed_dim // num_heads,
            dim_in=io_channels,
            dim_out=io_channels,
            cond_token_dim=embed_dim,
            global_cond_dim=embed_dim,
            local_add_cond_dim=local_add_cond_dim,
            num_memory_tokens=num_memory_tokens,
            attn_kwargs=dict(attn_kwargs),
            norm_type=norm_type,
            norm_kwargs=dict(norm_kwargs),
            ff_kwargs=dict(ff_kwargs),
        )
        self.preprocess_conv = nn.Conv1d(
            io_channels,
            io_channels,
            1,
            bias=False,
        )
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(
            io_channels,
            io_channels,
            1,
            bias=False,
        )
        nn.init.zeros_(self.postprocess_conv.weight)

    def forward(
        self,
        value: torch.Tensor,
        timestep: torch.Tensor,
        *,
        cross_attn_cond: torch.Tensor,
        local_add_cond: torch.Tensor,
        global_embed: torch.Tensor,
        cfg_scale: float = 1.0,
        cfg_dropout_prob: float = 0.0,
        use_checkpointing: bool = False,
        checkpoint_block_interval: int = 1,
        **_: Any,
    ) -> torch.Tensor:
        if cfg_scale != 1.0 or cfg_dropout_prob != 0.0:
            raise ValueError(
                "Native CFG must be disabled because guidance is applied by the pipeline"
            )
        model_dtype = next(self.parameters()).dtype
        value = value.to(model_dtype)
        timestep = timestep.float()
        context = self.to_cond_embed(cross_attn_cond.to(model_dtype))
        global_condition = self.to_global_embed(global_embed.to(model_dtype))
        local = local_add_cond.to(model_dtype).transpose(1, 2).contiguous()
        timestep_embedding = self.to_timestep_embed(
            self.timestep_features(timestep[:, None]).to(model_dtype)
        )
        global_condition = global_condition + timestep_embedding
        hidden = self.preprocess_conv(value) + value
        hidden = hidden.transpose(1, 2).contiguous()
        hidden = self.transformer(
            hidden,
            context=context,
            global_cond=global_condition,
            local_add_cond=local,
            use_checkpointing=use_checkpointing,
            checkpoint_block_interval=checkpoint_block_interval,
        )
        output = hidden.transpose(1, 2).contiguous()
        return self.postprocess_conv(output) + output


class SA3RenderDiT(nn.Module):
    """Adapt native ``[B, C, T]`` tensors to the ``[B, T, 128]`` contract."""

    architecture_type = SA3_ARCHITECTURE_TYPE
    source_commit = SA3_SOURCE_COMMIT

    def __init__(
        self,
        config: DiTConfig | Mapping[str, Any],
        *,
        native_config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.config = config if isinstance(config, DiTConfig) else DiTConfig.from_mapping(config)
        self.config.validate()
        self.native_config = copy.deepcopy(dict(native_config))
        required = {
            "io_channels": self.config.latent_dim,
            "cond_token_dim": self.config.hidden_size,
            "global_cond_dim": self.config.hidden_size,
            "local_add_cond_dim": self.config.hidden_size,
            "global_cond_type": "adaLN",
            "timestep_features_type": "expo",
            "diffusion_objective": "rectified_flow",
        }
        mismatches = {
            name: {
                "expected": expected,
                "actual": self.native_config.get(name),
            }
            for name, expected in required.items()
            if self.native_config.get(name) != expected
        }
        if mismatches:
            raise ValueError(f"Native architecture does not match the model contract: {mismatches}")
        for name in ("embed_dim", "depth", "num_heads", "num_memory_tokens"):
            value = self.native_config.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"native_config.{name} must be a positive integer")
        if int(self.native_config["embed_dim"]) % int(self.native_config["num_heads"]):
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.config.context_dim != self.config.hidden_size:
            raise ValueError("Text and semantic conditioning must have the same width")
        if self.config.adaln_conditioning != "timestep_plus_global_loudness":
            raise ValueError("The architecture requires global loudness conditioning")
        if self.native_config.get("attn_kwargs") != {
            "qk_norm": "rms",
            "differential": True,
        }:
            raise ValueError(
                "The attention configuration is incompatible with the released architecture"
            )
        if self.native_config.get("norm_type") != "rms_norm":
            raise ValueError(
                "The normalization configuration is incompatible with the released architecture"
            )
        self.native = SA3DiffusionTransformer(**self.native_config)
        self.activation_checkpoint_block_interval = 1

    def set_activation_checkpoint_block_interval(self, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("activation checkpoint block interval must be an integer")
        if value <= 0:
            raise ValueError("activation checkpoint block interval must be positive")
        self.activation_checkpoint_block_interval = value

    @property
    def latent_projection(self) -> nn.Linear:
        return self.native.transformer.project_in

    @property
    def blocks(self) -> nn.ModuleList:
        return self.native.transformer.layers

    def _validate_inputs(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        semantic_embeddings: torch.Tensor,
        frame_mask: torch.Tensor,
        text_context: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> None:
        if (
            noisy_latents.ndim != 3
            or not noisy_latents.is_floating_point()
            or noisy_latents.shape[-1] != self.config.latent_dim
        ):
            raise TypeError("noisy_latents must be a floating-point [B, T, 128] tensor")
        batch, frames, _ = noisy_latents.shape
        if batch != 1:
            raise ValueError("The Renderer currently accepts one sample per forward pass")
        if frames > self.config.max_frames:
            raise ValueError("The latent sequence exceeds max_frames")
        if semantic_embeddings.shape != (
            batch,
            frames,
            self.config.hidden_size,
        ):
            raise ValueError("Semantic embeddings must align with the latent frames")
        if frame_mask.dtype != torch.bool or frame_mask.shape != (batch, frames):
            raise TypeError("frame_mask must be a bool [B, T] tensor")
        lengths = frame_mask.sum(dim=1)
        canonical = torch.arange(
            frames,
            device=frame_mask.device,
        ).unsqueeze(0) < lengths.unsqueeze(1)
        if bool((lengths <= 0).any()) or not torch.equal(frame_mask, canonical):
            raise ValueError("frame_mask must contain a non-empty contiguous prefix")
        if timestep.ndim == 2 and timestep.shape[1] == 1:
            timestep = timestep[:, 0]
        if (
            timestep.ndim != 1
            or timestep.shape[0] != batch
            or not timestep.is_floating_point()
            or not torch.isfinite(timestep).all()
            or bool(((timestep < 0) | (timestep > 1)).any())
        ):
            raise ValueError("timestep must be finite, have shape [B], and lie in [0, 1]")
        if (
            text_context.ndim != 3
            or text_context.shape[0] != batch
            or text_context.shape[-1] != self.config.context_dim
        ):
            raise ValueError("text_context has an incompatible shape")
        if (
            text_mask.dtype != torch.bool
            or text_mask.shape != text_context.shape[:2]
            or not bool(text_mask.any(dim=1).all())
        ):
            raise ValueError("text_mask must select at least one token per sample")
        for value in (
            semantic_embeddings,
            frame_mask,
            text_context,
            text_mask,
            timestep,
        ):
            if value.device != noisy_latents.device:
                raise ValueError("All Renderer inputs must be on the same device")

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        semantic_embeddings: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        text_context: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        *,
        conditioning: Any | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        global_loudness_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if conditioning is not None:
            if any(
                value is not None
                for value in (
                    semantic_embeddings,
                    text_context,
                    text_mask,
                    context,
                    context_mask,
                )
            ):
                raise ValueError(
                    "conditioning cannot be combined with explicit conditioning tensors"
                )
            if frame_mask is not None and not torch.equal(frame_mask, conditioning.semantic_mask):
                raise ValueError("frame_mask does not match conditioning.semantic_mask")
            semantic_embeddings = conditioning.semantic_embeddings
            frame_mask = conditioning.semantic_mask
            text_context = conditioning.text_context
            text_mask = conditioning.text_mask
            if getattr(conditioning, "relative_dynamics_embeddings", None) is not None:
                raise ValueError("Relative dynamics conditioning is not supported")
        if context is not None:
            if text_context is not None:
                raise ValueError("context and text_context cannot both be provided")
            text_context = context
        if context_mask is not None:
            if text_mask is not None:
                raise ValueError("context_mask and text_mask cannot both be provided")
            text_mask = context_mask
        if any(
            value is None
            for value in (
                semantic_embeddings,
                frame_mask,
                text_context,
                text_mask,
            )
        ):
            raise ValueError("Semantic, text, and mask conditioning are required")
        assert semantic_embeddings is not None
        assert frame_mask is not None
        assert text_context is not None
        assert text_mask is not None
        self._validate_inputs(
            noisy_latents,
            timestep,
            semantic_embeddings,
            frame_mask,
            text_context,
            text_mask,
        )
        if (
            global_loudness_embedding is None
            or global_loudness_embedding.shape != (1, self.config.hidden_size)
            or global_loudness_embedding.device != noisy_latents.device
        ):
            raise ValueError("global_loudness_embedding must have shape [B, hidden_size]")
        frames = int(frame_mask.sum().item())
        latent = noisy_latents[:, :frames].transpose(1, 2).contiguous()
        semantic = semantic_embeddings[:, :frames].transpose(1, 2).contiguous()
        compact_text = text_context[:, text_mask[0], :].contiguous()
        output = self.native(
            latent,
            timestep.reshape(-1).float(),
            cross_attn_cond=compact_text,
            local_add_cond=semantic,
            global_embed=global_loudness_embedding,
            cfg_scale=1.0,
            cfg_dropout_prob=0.0,
            use_checkpointing=self.config.activation_checkpointing,
            checkpoint_block_interval=self.activation_checkpoint_block_interval,
        )
        if output.shape != (1, self.config.latent_dim, frames):
            raise RuntimeError(f"Renderer output has an incompatible shape: {tuple(output.shape)}")
        output = output.transpose(1, 2).contiguous()
        return F.pad(
            output,
            (0, 0, 0, noisy_latents.shape[1] - frames),
        )


def build_sa3_render_dit(
    config: DiTConfig | Mapping[str, Any],
    architecture: Mapping[str, Any],
) -> SA3RenderDiT:
    """Build the released Renderer architecture from its model configuration."""

    if architecture.get("type") != SA3_ARCHITECTURE_TYPE:
        raise ValueError("Unsupported Renderer architecture type")
    if architecture.get("source_commit") != SA3_SOURCE_COMMIT:
        raise ValueError("Unsupported Stable Audio 3 source commit")
    if architecture.get("profile") != "stable-audio-3-medium-compatible-v1":
        raise ValueError("Unsupported Renderer architecture profile")
    native_config = architecture.get("native_config")
    if not isinstance(native_config, Mapping):
        raise TypeError("Renderer architecture is missing native_config")
    expected_geometry = {
        "embed_dim": 1536,
        "depth": 24,
        "num_heads": 24,
        "num_memory_tokens": 64,
    }
    observed = {name: native_config.get(name) for name in expected_geometry}
    if observed != expected_geometry:
        raise ValueError(
            "Renderer geometry is incompatible with the released checkpoint: "
            f"expected={expected_geometry} actual={observed}"
        )
    return SA3RenderDiT(config, native_config=native_config)


__all__ = [
    "SA3_ARCHITECTURE_TYPE",
    "SA3_SOURCE_COMMIT",
    "SA3RenderDiT",
    "build_sa3_render_dit",
]
