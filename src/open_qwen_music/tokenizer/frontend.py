
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


CAUSAL_PADDING = "within_stride"
CAUSAL_PADDING_PRE_20260801 = "left_only"


SUBSAMPLING_CONTRACT_DEFAULTS: dict[str, Any] = {
    "frontend_type": "conv",
    "subsampling_kernel": 5,
    "causal_padding": CAUSAL_PADDING,
    "convnext_kernel": 7,
    "convnext_blocks_per_stage": 1,
    "convnext_expansion": 4,
    "convnext_layer_scale_init": 1e-6,
}




CAUSAL_SEMANTICS_DEFAULTS: dict[str, Any] = {

    "bestrq_causal_window": True,

    "bestrq_erode_loss_mask": True,

    "causal_heads": True,
}

CAUSAL_SEMANTICS_PRE_20260803_DEFAULTS: dict[str, Any] = {
    key: False for key in CAUSAL_SEMANTICS_DEFAULTS
}


POSITION_ENCODING_DEFAULTS: dict[str, Any] = {"position_encoding": "rope"}
POSITION_ENCODING_PRE_20260804_DEFAULTS: dict[str, Any] = {
    "position_encoding": "sinusoidal"
}


CAUSALITY_FIELDS = (
    "attention_causal",
    "frontend_causal",
    "conformer_conv_causal",
)


def resolve_causality(model_config: dict[str, Any]) -> dict[str, bool]:
    return {key: bool(model_config.get(key, False)) for key in CAUSALITY_FIELDS}


SUBSAMPLING_CONTRACT_DEFAULTS = {
    **SUBSAMPLING_CONTRACT_DEFAULTS,
    **CAUSAL_SEMANTICS_DEFAULTS,
    **POSITION_ENCODING_DEFAULTS,
}

SUBSAMPLING_CONTRACT_PRE_20260801_DEFAULTS: dict[str, Any] = {
    **SUBSAMPLING_CONTRACT_DEFAULTS,
    **CAUSAL_SEMANTICS_PRE_20260803_DEFAULTS,
    **POSITION_ENCODING_PRE_20260804_DEFAULTS,
    "causal_padding": CAUSAL_PADDING_PRE_20260801,
}


def resolve_subsampling_contract(
    model_config: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any]:

    resolved: dict[str, Any] = {}
    for key, fallback in defaults.items():
        value = model_config.get(key)
        if value is None:
            value = fallback
        resolved[key] = value.lower() if isinstance(value, str) else value
    return resolved


def mask_padding(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:

    if mask is None:
        return x
    return x.masked_fill(~mask.unsqueeze(-1), 0.0)


def downsample_mask(mask: torch.Tensor | None) -> torch.Tensor | None:

    return None if mask is None else mask[:, ::2]


class TimeConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, stride=stride)

    def forward(
        self, x: torch.Tensor, causal: bool, mask: torch.Tensor | None = None
    ) -> torch.Tensor:

        total = self.kernel_size - 1
        if causal:
            right = min(self.stride - 1, total)
            left = total - right
        else:
            left, right = total // 2, total - total // 2
        x = mask_padding(x, mask)
        return self.conv(F.pad(x.transpose(1, 2), (left, right))).transpose(1, 2)


class ConvSubsampling25Hz(nn.Module):

    def __init__(self, input_dim: int, model_dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TimeConv(input_dim, model_dim, kernel_size, 2),
                TimeConv(model_dim, model_dim, kernel_size, 2),
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(model_dim), nn.LayerNorm(model_dim)])
        self.activation = nn.SiLU()

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:

        for _ in range(2):
            lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        return lengths

    def forward(
        self, x: torch.Tensor, causal: bool, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer, norm in zip(self.layers, self.norms):
            x = self.activation(norm(layer(x, causal=causal, mask=mask)))
            mask = downsample_mask(mask)
        return mask_padding(x, mask)


class ConvNeXt1DBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        kernel_size: int = 7,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ConvNeXt The convolution kernel must be an odd number")
        self.kernel_size = kernel_size
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )
        self.gamma = nn.Parameter(torch.full((dim,), float(layer_scale_init)))

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = False,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        total = self.kernel_size - 1
        padding = (total, 0) if causal else (total // 2, total - total // 2)
        y = mask_padding(x, mask).transpose(1, 2)
        y = self.depthwise(F.pad(y, padding)).transpose(1, 2)
        return x + self.gamma * self.mlp(self.norm(y))


class ConvNeXtSubsampling25Hz(nn.Module):

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        *,
        kernel_size: int = 5,
        block_kernel_size: int = 7,
        blocks_per_stage: int = 1,
        expansion: int = 4,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()
        if blocks_per_stage < 1:
            raise ValueError("ConvNeXt requires at least one block")
        self.downsampling = nn.ModuleList(
            [
                TimeConv(input_dim, model_dim, kernel_size, 2),
                TimeConv(model_dim, model_dim, kernel_size, 2),
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(model_dim), nn.LayerNorm(model_dim)])
        self.stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ConvNeXt1DBlock(
                            model_dim,
                            expansion=expansion,
                            kernel_size=block_kernel_size,
                            layer_scale_init=layer_scale_init,
                        )
                        for _ in range(blocks_per_stage)
                    ]
                )
                for _ in range(2)
            ]
        )

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        for _ in range(2):
            lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
        return lengths

    def forward(
        self, x: torch.Tensor, causal: bool, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for downsampling, norm, blocks in zip(
            self.downsampling, self.norms, self.stages
        ):
            x = norm(downsampling(x, causal=causal, mask=mask))
            mask = downsample_mask(mask)
            for block in blocks:
                x = block(x, causal=causal, mask=mask)
        return mask_padding(x, mask)


def build_subsampling_frontend(
    model_config: dict[str, Any],
    *,
    input_dim: int,
    model_dim: int,
) -> nn.Module:
    contract = resolve_subsampling_contract(
        model_config, SUBSAMPLING_CONTRACT_DEFAULTS
    )
    if contract["causal_padding"] != CAUSAL_PADDING:
        raise ValueError(
            f"causal_padding must be {CAUSAL_PADDING!r}; received "
            f"{contract['causal_padding']!r}. Checkpoints trained with "
            f"{CAUSAL_PADDING_PRE_20260801!r} padding are incompatible and must be retrained."
        )
    frontend_type = str(contract["frontend_type"])
    kernel_size = int(contract["subsampling_kernel"])
    if kernel_size < 2:
        raise ValueError(
            f"subsampling_kernel must be at least stride=2; received {kernel_size}. "
            "A smaller kernel cannot cover the stride interval."
        )
    if frontend_type == "conv":
        return ConvSubsampling25Hz(input_dim, model_dim, kernel_size)
    if frontend_type == "convnext":
        return ConvNeXtSubsampling25Hz(
            input_dim,
            model_dim,
            kernel_size=kernel_size,
            block_kernel_size=int(contract["convnext_kernel"]),
            blocks_per_stage=int(contract["convnext_blocks_per_stage"]),
            expansion=int(contract["convnext_expansion"]),
            layer_scale_init=float(contract["convnext_layer_scale_init"]),
        )
    raise ValueError(f"Unsupported tokenizer frontend_type: {frontend_type}")
