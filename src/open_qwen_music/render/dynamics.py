
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from open_qwen_music.common.loudness import (
    _block_loudness,
    _integrated_from_blocks,
    k_weight,
)


SEMANTIC_VOCAB_SIZE = 32_768
SEMANTIC_FRAME_HZ = 25.0


@dataclass(frozen=True)
class DynamicsTargetConfig:
    frame_hz: float = SEMANTIC_FRAME_HZ
    local_window_seconds: float = 0.4
    target_kind: str = "relative_parent_lufs"
    relative_min_lu: float = -30.0
    relative_max_lu: float = 12.0
    activity_threshold_lu: float = -20.0
    absolute_min_lufs: float = -70.0
    absolute_max_lufs: float = 3.0
    absolute_activity_threshold_lufs: float = -50.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DynamicsTargetConfig":
        config = cls(
            frame_hz=float(value.get("frame_hz", SEMANTIC_FRAME_HZ)),
            local_window_seconds=float(value.get("local_window_seconds", 0.4)),
            target_kind=str(
                value.get("target_kind", "relative_parent_lufs")
            ),
            relative_min_lu=float(value.get("relative_min_lu", -30.0)),
            relative_max_lu=float(value.get("relative_max_lu", 12.0)),
            activity_threshold_lu=float(
                value.get("activity_threshold_lu", -20.0)
            ),
            absolute_min_lufs=float(value.get("absolute_min_lufs", -70.0)),
            absolute_max_lufs=float(value.get("absolute_max_lufs", 3.0)),
            absolute_activity_threshold_lufs=float(
                value.get("absolute_activity_threshold_lufs", -50.0)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.frame_hz != SEMANTIC_FRAME_HZ:
            raise ValueError("Dynamics targets must remain aligned with semantic tokens at 25 Hz")
        if not 0.04 <= self.local_window_seconds <= 3.0:
            raise ValueError("local_window_seconds must be in [0.04, 3.0]")
        if self.target_kind not in {
            "relative_parent_lufs",
            "relative_short_window_lufs",
        }:
            raise ValueError("target_kind is incompatible with")
        if not self.relative_min_lu < self.activity_threshold_lu:
            raise ValueError("activity threshold must exceed the relative lower bound")
        if not self.activity_threshold_lu < self.relative_max_lu:
            raise ValueError("activity threshold must be below the relative upper bound")
        if not self.absolute_min_lufs < self.absolute_activity_threshold_lufs:
            raise ValueError("absolute activity threshold must exceed the absolute lower bound")
        if not self.absolute_activity_threshold_lufs < self.absolute_max_lufs:
            raise ValueError("absolute activity threshold must be below the absolute upper bound")


def extract_relative_dynamics(
    waveform: np.ndarray,
    sample_rate: int,
    frame_count: int,
    *,
    config: DynamicsTargetConfig | None = None,
    reference_lufs: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:

    config = config or DynamicsTargetConfig()
    config.validate()
    matrix = np.atleast_2d(np.asarray(waveform, dtype=np.float32))
    if matrix.ndim != 2 or matrix.shape[0] not in {1, 2}:
        raise ValueError("waveform must be mono or stereo with shape [C, N]")
    if sample_rate <= 0 or frame_count <= 0:
        raise ValueError("sample_rate and frame_count must be positive")
    if matrix.shape[-1] <= 0 or not np.isfinite(matrix).all():
        raise ValueError("waveform must be non-empty and contain only finite values")

    filtered = k_weight(matrix, sample_rate)
    if reference_lufs is None:
        global_lufs = _integrated_from_blocks(
            _block_loudness(filtered, sample_rate, 0.4)
        )
    else:
        global_lufs = float(reference_lufs)
        if not math.isfinite(global_lufs):
            raise ValueError("reference_lufs must be finite")
    channel_weights = np.ones((filtered.shape[0], 1), dtype=np.float64)
    if filtered.shape[0] >= 6:
        channel_weights[3] = 0.0
        channel_weights[4] = channel_weights[5] = 1.41
    elif filtered.shape[0] == 5:
        channel_weights[3] = channel_weights[4] = 1.41
    power = (
        np.square(filtered, dtype=np.float64) * channel_weights
    ).sum(axis=0)
    prefix = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(power, dtype=np.float64))
    )
    window_samples = max(
        1, int(round(config.local_window_seconds * sample_rate))
    )
    half_window = window_samples / 2.0
    centers = (np.arange(frame_count, dtype=np.float64) + 0.5) * (
        sample_rate / config.frame_hz
    )
    starts = np.maximum(0, np.floor(centers - half_window).astype(np.int64))
    stops = np.minimum(
        matrix.shape[-1],
        np.ceil(centers + half_window).astype(np.int64),
    )
    local_energy = (prefix[stops] - prefix[starts]) / np.maximum(
        stops - starts, 1
    )
    local_lufs = -0.691 + 10.0 * np.log10(np.maximum(local_energy, 1.0e-12))
    if not math.isfinite(global_lufs):
        global_lufs = float(np.median(local_lufs))
    if config.target_kind in {
        "relative_parent_lufs",
        "relative_short_window_lufs",
    }:
        if (
            config.target_kind == "relative_short_window_lufs"
            and reference_lufs is None
        ):
            raise ValueError(
                "relative_short_window_lufs requires an explicit reference_lufs for the input window"
            )
        target = local_lufs - float(global_lufs)
        activity = (
            local_lufs > config.absolute_activity_threshold_lufs
            if config.target_kind == "relative_short_window_lufs"
            else target > config.activity_threshold_lu
        )
        target = np.clip(
            target,
            config.relative_min_lu,
            config.relative_max_lu,
        )
    return (
        target.astype(np.float32),
        activity.astype(np.float32),
        float(global_lufs),
    )


@dataclass(frozen=True)
class DynamicsPredictorConfig:
    semantic_vocab_size: int = SEMANTIC_VOCAB_SIZE
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ffn_dim: int = 768
    dropout: float = 0.1
    downsample_factor: int = 4
    max_frames: int = 9_000
    output_scale_lu: float = 10.0
    local_refinement_layers: int = 0
    local_refinement_kernel_size: int = 5
    local_refinement_expansion: int = 2

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "DynamicsPredictorConfig":
        config = cls(
            semantic_vocab_size=int(
                value.get("semantic_vocab_size", SEMANTIC_VOCAB_SIZE)
            ),
            hidden_size=int(value.get("hidden_size", 256)),
            num_layers=int(value.get("num_layers", 4)),
            num_heads=int(value.get("num_heads", 8)),
            ffn_dim=int(value.get("ffn_dim", 768)),
            dropout=float(value.get("dropout", 0.1)),
            downsample_factor=int(value.get("downsample_factor", 4)),
            max_frames=int(value.get("max_frames", 9_000)),
            output_scale_lu=float(value.get("output_scale_lu", 10.0)),
            local_refinement_layers=int(
                value.get("local_refinement_layers", 0)
            ),
            local_refinement_kernel_size=int(
                value.get("local_refinement_kernel_size", 5)
            ),
            local_refinement_expansion=int(
                value.get("local_refinement_expansion", 2)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.semantic_vocab_size != SEMANTIC_VOCAB_SIZE:
            raise ValueError("The semantic vocabulary size must be 32768")
        for name in (
            "hidden_size",
            "num_layers",
            "num_heads",
            "ffn_dim",
            "downsample_factor",
            "max_frames",
            "local_refinement_kernel_size",
            "local_refinement_expansion",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.output_scale_lu <= 0.0:
            raise ValueError("output_scale_lu must be positive")
        if self.local_refinement_layers < 0:
            raise ValueError("local_refinement_layers must be non-negative")
        if self.local_refinement_kernel_size % 2 != 1:
            raise ValueError("local_refinement_kernel_size must be odd")


@dataclass
class DynamicsPrediction:
    relative_loudness_lu: torch.Tensor
    activity_logits: torch.Tensor
    frame_mask: torch.Tensor
    coarse_relative_loudness_lu: torch.Tensor | None = None
    local_relative_residual_lu: torch.Tensor | None = None

    @property
    def activity_probability(self) -> torch.Tensor:
        return self.activity_logits.sigmoid()


class _LocalDynamicsBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        *,
        kernel_size: int,
        dilation: int,
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.depthwise = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
            groups=hidden_size,
        )
        self.pointwise = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * expansion, hidden_size),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = frame_mask.unsqueeze(-1)
        residual = self.norm(hidden).masked_fill(~valid, 0.0)
        residual = self.depthwise(residual.transpose(1, 2)).transpose(1, 2)
        residual = self.pointwise(residual)
        return (hidden + residual).masked_fill(~valid, 0.0)


def _sinusoidal_positions(length: int, width: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    result = torch.zeros(length, width, dtype=torch.float32)
    result[:, 0::2] = torch.sin(position * frequency)
    result[:, 1::2] = torch.cos(position * frequency[: result[:, 1::2].shape[1]])
    return result


class SemanticDynamicsPredictor(nn.Module):

    def __init__(
        self, config: DynamicsPredictorConfig | Mapping[str, Any]
    ) -> None:
        super().__init__()
        self.config = (
            config
            if isinstance(config, DynamicsPredictorConfig)
            else DynamicsPredictorConfig.from_mapping(config)
        )
        self.config.validate()
        hidden = self.config.hidden_size
        factor = self.config.downsample_factor
        self.semantic_embedding = nn.Embedding(
            self.config.semantic_vocab_size, hidden
        )
        self.downsample = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=2 * factor + 1,
            stride=factor,
            padding=factor,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.ffn_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.config.num_layers,
            norm=nn.LayerNorm(hidden),
            enable_nested_tensor=False,
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.local_refinement = nn.ModuleList(
            [
                _LocalDynamicsBlock(
                    hidden,
                    kernel_size=self.config.local_refinement_kernel_size,
                    dilation=2 ** (index % 6),
                    expansion=self.config.local_refinement_expansion,
                    dropout=self.config.dropout,
                )
                for index in range(self.config.local_refinement_layers)
            ]
        )
        self.local_output_head = (
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, 2),
            )
            if self.local_refinement
            else None
        )
        if self.local_output_head is not None:
            nn.init.zeros_(self.local_output_head[-1].weight)
            nn.init.zeros_(self.local_output_head[-1].bias)
        max_low_frames = (
            self.config.max_frames + self.config.downsample_factor - 1
        ) // self.config.downsample_factor
        self.register_buffer(
            "position_embedding",
            _sinusoidal_positions(max_low_frames, hidden),
            persistent=False,
        )

    def forward(
        self, semantic_ids: torch.Tensor, frame_mask: torch.Tensor
    ) -> DynamicsPrediction:
        if semantic_ids.dtype != torch.long or semantic_ids.ndim != 2:
            raise TypeError("semantic_ids must be an int64 tensor with shape [B, T]")
        if frame_mask.dtype != torch.bool or frame_mask.shape != semantic_ids.shape:
            raise TypeError("frame_mask must be a bool tensor with the same shape as semantic_ids")
        if semantic_ids.shape[1] > self.config.max_frames:
            raise ValueError("Semantic sequence exceeds max_frames")
        lengths = frame_mask.sum(dim=1)
        if bool((lengths <= 0).any()):
            raise ValueError("Each semantic sequence must contain at least one valid frame")
        if semantic_ids.numel() and (
            int(semantic_ids.min()) < 0
            or int(semantic_ids.max()) >= self.config.semantic_vocab_size
        ):
            raise ValueError("Semantic ID is out of bounds")

        full_hidden = self.semantic_embedding(semantic_ids)
        full_hidden = full_hidden.masked_fill(~frame_mask.unsqueeze(-1), 0.0)
        hidden = self.downsample(full_hidden.transpose(1, 2)).transpose(1, 2)
        low_lengths = torch.div(
            lengths + self.config.downsample_factor - 1,
            self.config.downsample_factor,
            rounding_mode="floor",
        )
        low_mask = (
            torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
            < low_lengths.unsqueeze(1)
        )
        hidden = hidden + self.position_embedding[: hidden.shape[1]].to(
            device=hidden.device, dtype=hidden.dtype
        )
        hidden = hidden.masked_fill(~low_mask.unsqueeze(-1), 0.0)
        hidden = self.encoder(hidden, src_key_padding_mask=~low_mask)
        low_output = self.output_head(hidden)


        batch_size, maximum_frames = semantic_ids.shape
        coarse_output = low_output.new_zeros(batch_size, maximum_frames, 2)
        long_context = (
            hidden.new_zeros(batch_size, maximum_frames, hidden.shape[-1])
            if self.local_output_head is not None
            else None
        )
        for index in range(batch_size):
            full_length = int(lengths[index])
            low_length = int(low_lengths[index])
            coarse_output[index, :full_length] = F.interpolate(
                low_output[index, :low_length].transpose(0, 1).unsqueeze(0),
                size=full_length,
                mode="linear",
                align_corners=False,
            ).squeeze(0).transpose(0, 1)
            if long_context is not None:
                long_context[index, :full_length] = F.interpolate(
                    hidden[index, :low_length].transpose(0, 1).unsqueeze(0),
                    size=full_length,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).transpose(0, 1)
        output = coarse_output
        local_output: torch.Tensor | None = None
        if self.local_output_head is not None:
            assert long_context is not None
            local_hidden = full_hidden + long_context
            for block in self.local_refinement:
                local_hidden = block(local_hidden, frame_mask)
            local_output = self.local_output_head(local_hidden)
            output = output + local_output
        relative = output[..., 0] * self.config.output_scale_lu
        activity_logits = output[..., 1]
        coarse_relative = (
            coarse_output[..., 0] * self.config.output_scale_lu
        )
        local_relative_residual = (
            local_output[..., 0] * self.config.output_scale_lu
            if local_output is not None
            else None
        )
        relative = relative.masked_fill(~frame_mask, 0.0)
        activity_logits = activity_logits.masked_fill(~frame_mask, 0.0)
        coarse_relative = coarse_relative.masked_fill(~frame_mask, 0.0)
        if local_relative_residual is not None:
            local_relative_residual = local_relative_residual.masked_fill(
                ~frame_mask, 0.0
            )
        return DynamicsPrediction(
            relative,
            activity_logits,
            frame_mask,
            coarse_relative,
            local_relative_residual,
        )


@dataclass(frozen=True)
class DynamicsLossConfig:
    relative_weight: float = 1.0
    activity_weight: float = 0.25
    derivative_weight: float = 0.2
    multiscale_weight: float = 0.3
    correlation_weight: float = 0.0
    mse_weight: float = 0.0
    scale_weight: float = 0.0
    slow_envelope_weight: float = 0.0
    slow_correlation_weight: float = 0.0
    fast_residual_weight: float = 0.0
    slow_envelope_window_frames: int = 75
    silent_relative_weight: float = 0.1
    huber_delta_lu: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DynamicsLossConfig":
        config = cls(
            relative_weight=float(value.get("relative_weight", 1.0)),
            activity_weight=float(value.get("activity_weight", 0.25)),
            derivative_weight=float(value.get("derivative_weight", 0.2)),
            multiscale_weight=float(value.get("multiscale_weight", 0.3)),
            correlation_weight=float(value.get("correlation_weight", 0.0)),
            mse_weight=float(value.get("mse_weight", 0.0)),
            scale_weight=float(value.get("scale_weight", 0.0)),
            slow_envelope_weight=float(
                value.get("slow_envelope_weight", 0.0)
            ),
            slow_correlation_weight=float(
                value.get("slow_correlation_weight", 0.0)
            ),
            fast_residual_weight=float(
                value.get("fast_residual_weight", 0.0)
            ),
            slow_envelope_window_frames=int(
                value.get("slow_envelope_window_frames", 75)
            ),
            silent_relative_weight=float(
                value.get("silent_relative_weight", 0.1)
            ),
            huber_delta_lu=float(value.get("huber_delta_lu", 1.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        for name in (
            "relative_weight",
            "activity_weight",
            "derivative_weight",
            "multiscale_weight",
            "correlation_weight",
            "mse_weight",
            "scale_weight",
            "slow_envelope_weight",
            "slow_correlation_weight",
            "fast_residual_weight",
            "silent_relative_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
        if self.slow_envelope_window_frames <= 0:
            raise ValueError("slow_envelope_window_frames must be positive")
        if self.slow_envelope_window_frames % 2 != 1:
            raise ValueError("slow_envelope_window_frames must be odd")
        if not math.isfinite(self.huber_delta_lu) or self.huber_delta_lu <= 0.0:
            raise ValueError("huber_delta_lu must be positive and finite")


def _masked_moving_average(
    value: torch.Tensor,
    mask: torch.Tensor,
    window_frames: int,
) -> torch.Tensor:
    if window_frames <= 0 or window_frames % 2 != 1:
        raise ValueError("slow_envelope_window_frames must be a positive odd number")
    if window_frames == 1:
        return value.masked_fill(~mask, 0.0)
    valid = mask.to(dtype=value.dtype)
    kernel = torch.ones(
        1,
        1,
        window_frames,
        dtype=value.dtype,
        device=value.device,
    )
    padding = window_frames // 2
    numerator = F.conv1d(
        (value * valid).unsqueeze(1),
        kernel,
        padding=padding,
    ).squeeze(1)
    denominator = F.conv1d(
        valid.unsqueeze(1),
        kernel,
        padding=padding,
    ).squeeze(1)
    return (numerator / denominator.clamp_min(1.0)).masked_fill(~mask, 0.0)


def _mean_correlation_loss(
    predicted_value: torch.Tensor,
    target_value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    correlations = []
    for index in range(mask.shape[0]):
        keep = mask[index]
        predicted = predicted_value[index, keep].float()
        target = target_value[index, keep].float()
        if predicted.numel() < 2:
            continue
        predicted_centered = predicted - predicted.mean()
        target_centered = target - target.mean()
        predicted_norm = predicted_centered.square().sum().sqrt()
        target_norm = target_centered.square().sum().sqrt()
        correlations.append(
            (predicted_centered * target_centered).sum()
            / (predicted_norm * target_norm).clamp_min(1.0e-6)
        )
    return (
        1.0 - torch.stack(correlations).mean()
        if correlations
        else predicted_value.new_zeros(())
    )


def dynamics_loss(
    prediction: DynamicsPrediction,
    target_relative_lu: torch.Tensor,
    target_activity: torch.Tensor,
    *,
    config: DynamicsLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    config = config or DynamicsLossConfig()
    config.validate()
    mask = prediction.frame_mask
    if (
        target_relative_lu.shape != mask.shape
        or target_activity.shape != mask.shape
    ):
        raise ValueError("Dynamics target must have the same shape as frame_mask")
    valid = mask.to(dtype=prediction.relative_loudness_lu.dtype)
    activity = target_activity.to(dtype=valid.dtype).clamp(0.0, 1.0)
    relative_weights = valid * (
        config.silent_relative_weight
        + (1.0 - config.silent_relative_weight) * activity
    )
    relative_element = F.huber_loss(
        prediction.relative_loudness_lu,
        target_relative_lu,
        reduction="none",
        delta=config.huber_delta_lu,
    )
    relative_loss = (relative_element * relative_weights).sum() / (
        relative_weights.sum().clamp_min(1.0)
    )
    activity_element = F.binary_cross_entropy_with_logits(
        prediction.activity_logits,
        activity,
        reduction="none",
    )
    activity_loss = (activity_element * valid).sum() / valid.sum().clamp_min(1.0)

    pair_mask = mask[:, 1:] & mask[:, :-1]
    predicted_delta = (
        prediction.relative_loudness_lu[:, 1:]
        - prediction.relative_loudness_lu[:, :-1]
    )
    target_delta = target_relative_lu[:, 1:] - target_relative_lu[:, :-1]
    derivative_element = F.huber_loss(
        predicted_delta,
        target_delta,
        reduction="none",
        delta=config.huber_delta_lu,
    )
    derivative_loss = (
        derivative_element * pair_mask.to(dtype=derivative_element.dtype)
    ).sum() / pair_mask.sum().clamp_min(1)

    multiscale_terms = []
    for factor in (4, 16, 64):
        if target_relative_lu.shape[1] < factor:
            continue
        pred_sum = F.avg_pool1d(
            (prediction.relative_loudness_lu * valid).unsqueeze(1),
            factor,
            factor,
        ).squeeze(1)
        target_sum = F.avg_pool1d(
            (target_relative_lu * valid).unsqueeze(1),
            factor,
            factor,
        ).squeeze(1)


        pooled_weight = F.avg_pool1d(
            valid.float().unsqueeze(1), factor, factor
        ).squeeze(1)
        pooled_mask = pooled_weight >= (1.0 - 1.0e-6)
        term = F.huber_loss(
            pred_sum,
            target_sum,
            reduction="none",
            delta=config.huber_delta_lu,
        )
        multiscale_terms.append(
            (term * pooled_mask.to(dtype=term.dtype)).sum()
            / pooled_mask.sum().clamp_min(1)
        )
    multiscale_loss = (
        torch.stack(multiscale_terms).mean()
        if multiscale_terms
        else relative_loss.new_zeros(())
    )
    squared_error = (
        prediction.relative_loudness_lu - target_relative_lu
    ).square()
    mse_loss = (squared_error * relative_weights).sum() / (
        relative_weights.sum().clamp_min(1.0)
    )

    scale_errors = []
    for index in range(mask.shape[0]):
        keep = mask[index]
        predicted = prediction.relative_loudness_lu[index, keep].float()
        target = target_relative_lu[index, keep].float()
        if predicted.numel() < 2:
            continue
        predicted_centered = predicted - predicted.mean()
        target_centered = target - target.mean()
        scale_errors.append(
            F.smooth_l1_loss(
                predicted_centered.square().mean().sqrt(),
                target_centered.square().mean().sqrt(),
                beta=config.huber_delta_lu,
            )
        )
    correlation_loss = _mean_correlation_loss(
        prediction.relative_loudness_lu,
        target_relative_lu,
        mask,
    )
    scale_loss = (
        torch.stack(scale_errors).mean()
        if scale_errors
        else relative_loss.new_zeros(())
    )
    slow_envelope_loss = relative_loss.new_zeros(())
    slow_correlation_loss = relative_loss.new_zeros(())
    fast_residual_loss = relative_loss.new_zeros(())
    hierarchical_weight = (
        config.slow_envelope_weight
        + config.slow_correlation_weight
        + config.fast_residual_weight
    )
    if hierarchical_weight > 0.0:
        if (
            prediction.coarse_relative_loudness_lu is None
            or prediction.local_relative_residual_lu is None
        ):
            raise ValueError("Slow-envelope and fast-residual supervision require the local refinement branch")
        slow_target = _masked_moving_average(
            target_relative_lu,
            mask,
            config.slow_envelope_window_frames,
        )
        slow_element = F.huber_loss(
            prediction.coarse_relative_loudness_lu,
            slow_target,
            reduction="none",
            delta=config.huber_delta_lu,
        )
        slow_envelope_loss = (slow_element * relative_weights).sum() / (
            relative_weights.sum().clamp_min(1.0)
        )
        slow_correlation_loss = _mean_correlation_loss(
            prediction.coarse_relative_loudness_lu,
            slow_target,
            mask,
        )
        fast_target = target_relative_lu - slow_target
        fast_element = F.huber_loss(
            prediction.local_relative_residual_lu,
            fast_target,
            reduction="none",
            delta=config.huber_delta_lu,
        )
        fast_residual_loss = (fast_element * valid).sum() / (
            valid.sum().clamp_min(1.0)
        )
    total = (
        config.relative_weight * relative_loss
        + config.activity_weight * activity_loss
        + config.derivative_weight * derivative_loss
        + config.multiscale_weight * multiscale_loss
        + config.correlation_weight * correlation_loss
        + config.mse_weight * mse_loss
        + config.scale_weight * scale_loss
        + config.slow_envelope_weight * slow_envelope_loss
        + config.slow_correlation_weight * slow_correlation_loss
        + config.fast_residual_weight * fast_residual_loss
    )
    return {
        "loss": total,
        "relative_loss": relative_loss,
        "activity_loss": activity_loss,
        "derivative_loss": derivative_loss,
        "multiscale_loss": multiscale_loss,
        "correlation_loss": correlation_loss,
        "mse_loss": mse_loss,
        "scale_loss": scale_loss,
        "slow_envelope_loss": slow_envelope_loss,
        "slow_correlation_loss": slow_correlation_loss,
        "fast_residual_loss": fast_residual_loss,
    }
