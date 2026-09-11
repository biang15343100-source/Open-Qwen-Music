
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F




TARGET_AGGREGATIONS = ("stack", "mean")


BESTRQ_CONTRACT_DEFAULTS: dict[str, Any] = {
    "target_aggregation": "stack",
    "local_window": 1,
    "whitening": True,
    "projection_init": "gaussian_column_norm",
    "mask_noise_mode": "absolute",
    "loss_mask_support": "nominal",
}


BESTRQ_CONTRACT_PRE_20260804_DEFAULTS: dict[str, Any] = {
    "target_aggregation": "mean",
    "local_window": 3,
    "whitening": False,
    "projection_init": "gaussian_column_norm",
    "mask_noise_mode": "absolute",
    "loss_mask_support": "nominal",
}


def resolve_bestrq_contract(
    bestrq_config: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any]:

    resolved: dict[str, Any] = {}
    for key, fallback in defaults.items():
        value = bestrq_config.get(key)
        if value is None:
            value = fallback
        resolved[key] = value.lower() if isinstance(value, str) else value
    return resolved


def load_whitening_stats(
    path: str | None, *, input_dim: int, aggregation: str
) -> tuple[torch.Tensor, torch.Tensor]:

    if not path:
        raise ValueError(
            "bestrq.whitening=true requires bestrq.whitening_stats. Provide a frozen "
            "statistics file produced by the offline verification tool with "
            "--with-whitening, or explicitly set bestrq.whitening=false."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source_aggregation = str(payload.get("aggregation", ""))
    if source_aggregation != aggregation:
        raise ValueError(
            "whitening statistics were computed for "
            f"target_aggregation={source_aggregation!r}, but the current configuration "
            f"uses {aggregation!r}; recompute the statistics"
        )
    source_mels = int(payload.get("n_mels", -1))
    if source_mels != input_dim:
        raise ValueError(
            f"whitening statistics use n_mels={source_mels}, but the current input uses {input_dim}"
        )
    return payload["mean"].float(), payload["matrix"].float()


class BestRQTarget(nn.Module):

    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        codebook_size: int,
        local_window: int = 1,
        seed: int = 20260719,
        distance_chunk_size: int = 2048,
        causal: bool = False,
        aggregation: str = "stack",
        projection_init: str = "gaussian_column_norm",
        whitening_mean: torch.Tensor | None = None,
        whitening_matrix: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if aggregation not in TARGET_AGGREGATIONS:
            raise ValueError(
                f"target_aggregation must be one of {TARGET_AGGREGATIONS}; "
                f"received {aggregation!r}"
            )
        self.aggregation = aggregation
        if projection_init not in ("gaussian_column_norm", "xavier_normal"):
            raise ValueError(
                "projection_init must be 'gaussian_column_norm' or 'xavier_normal'; "
                f"received {projection_init!r}"
            )
        self.projection_init = projection_init

        target_dim = input_dim * 4 if aggregation == "stack" else input_dim
        self.target_dim = target_dim
        generator = torch.Generator().manual_seed(seed)
        projection = torch.empty(target_dim, projection_dim)
        if projection_init == "xavier_normal":
            nn.init.xavier_normal_(projection, generator=generator)
        else:
            projection.normal_(generator=generator)
            projection = F.normalize(projection, dim=0)
        codebook = F.normalize(
            torch.randn(codebook_size, projection_dim, generator=generator), dim=-1
        )
        self.register_buffer("projection", projection)
        self.register_buffer("codebook", codebook)

        if whitening_mean is None:
            whitening_mean = torch.zeros(target_dim)
        if whitening_matrix is None:
            whitening_matrix = torch.eye(target_dim)
        if tuple(whitening_mean.shape) != (target_dim,):
            raise ValueError(
                f"whitening_mean must have shape ({target_dim},), received "
                f"{tuple(whitening_mean.shape)} for target_aggregation={aggregation!r}"
            )
        if tuple(whitening_matrix.shape) != (target_dim, target_dim):
            raise ValueError(
                f"whitening_matrix must have shape ({target_dim}, {target_dim}), "
                f"received {tuple(whitening_matrix.shape)}"
            )
        self.register_buffer("whitening_mean", whitening_mean.float())
        self.register_buffer("whitening_matrix", whitening_matrix.float())
        self.local_window = local_window
        self.distance_chunk_size = distance_chunk_size
        self.causal = causal

    def aggregate(
        self, clean_features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:

        return self._aggregate_25hz(clean_features, mask)

    def _aggregate_25hz(
        self, clean_features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:

        x = clean_features.transpose(1, 2)
        weight = (
            torch.ones_like(x[:, :1])
            if mask is None
            else mask.unsqueeze(1).to(x.dtype)
        )
        x = x * weight
        if self.local_window > 1:


            if self.causal:
                padding = (self.local_window - 1, 0)
            else:
                pad = self.local_window // 2
                padding = (pad, self.local_window - 1 - pad)
            x = F.avg_pool1d(F.pad(x, padding), self.local_window, stride=1)
            weight = F.avg_pool1d(F.pad(weight, padding), self.local_window, stride=1)


        if self.aggregation == "stack" and mask is not None and self.local_window > 1:
            x = x * mask.unsqueeze(1).to(x.dtype)


        pad = (-x.shape[-1]) % 4
        if pad:
            x = F.pad(x, (0, pad))
            weight = F.pad(weight, (0, pad))
        if self.aggregation == "stack":
            batch, mels, frames = x.shape

            return (
                x.reshape(batch, mels, frames // 4, 4)
                .permute(0, 2, 3, 1)
                .reshape(batch, frames // 4, 4 * mels)
            )
        x = F.avg_pool1d(x, kernel_size=4, stride=4)
        weight = F.avg_pool1d(weight, kernel_size=4, stride=4)


        return (x / weight.clamp_min(1e-6)).transpose(1, 2)

    @torch.no_grad()
    def forward(
        self, clean_features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:


        with torch.autocast(device_type=clean_features.device.type, enabled=False):
            x = self._aggregate_25hz(clean_features.float(), mask)


            x = (x - self.whitening_mean.float()) @ self.whitening_matrix.float()
            projected = F.normalize(x @ self.projection.float(), dim=-1)
            flat = projected.reshape(-1, projected.shape[-1])
            ids = []
            for chunk in flat.split(self.distance_chunk_size):
                ids.append((chunk @ self.codebook.float().T).argmax(dim=-1))
            return torch.cat(ids).view(projected.shape[:-1])


def make_span_mask(
    lengths_100hz: torch.Tensor,
    maximum_frames: int,
    span_frames: int = 40,
    probability: float = 0.3,
) -> torch.Tensor:

    batch = lengths_100hz.shape[0]
    mask = torch.zeros(batch, maximum_frames, device=lengths_100hz.device, dtype=torch.bool)
    for row in range(batch):
        length = int(lengths_100hz[row].item())
        if length <= 0:
            continue
        spans = math.ceil(length / span_frames)
        selected = torch.rand(spans, device=mask.device) < probability
        if not selected.any():
            selected[torch.randint(spans, (1,), device=mask.device)] = True
        expanded = selected.repeat_interleave(span_frames)[:length]
        mask[row, :length] = expanded
    return mask


def apply_feature_mask(
    features: torch.Tensor,
    mask: torch.Tensor,
    noise_std: float,
) -> torch.Tensor:
    noise = torch.randn_like(features) * noise_std
    return torch.where(mask.unsqueeze(-1), noise, features)


def make_waveform_span_mask(
    waveform_lengths: torch.Tensor,
    maximum_samples: int,
    *,
    span_samples: int = 9_600,
    probability: float = 0.3,
) -> torch.Tensor:

    batch = waveform_lengths.shape[0]
    mask = torch.zeros(
        batch,
        maximum_samples,
        device=waveform_lengths.device,
        dtype=torch.bool,
    )
    for row in range(batch):
        length = int(waveform_lengths[row].item())
        if length <= 0:
            continue
        spans = math.ceil(length / span_samples)
        selected = torch.rand(spans, device=mask.device) < probability
        if not selected.any():
            selected[torch.randint(spans, (1,), device=mask.device)] = True
        mask[row, :length] = selected.repeat_interleave(span_samples)[:length]
    return mask


def apply_waveform_mask(
    waveform: torch.Tensor,
    mask: torch.Tensor,
    noise_std: float | torch.Tensor,
) -> torch.Tensor:
    if isinstance(noise_std, torch.Tensor):
        scale = noise_std.to(device=waveform.device, dtype=waveform.dtype)
        while scale.ndim < waveform.ndim:
            scale = scale.unsqueeze(-1)
    else:
        scale = float(noise_std)
    noise = torch.randn_like(waveform) * scale
    return torch.where(mask, noise, waveform)


def relative_waveform_noise_std(
    waveform: torch.Tensor,
    waveform_lengths: torch.Tensor,
    *,
    noise_db: float = -20.0,
    minimum: float = 1e-5,
    maximum: float = 0.2,
) -> torch.Tensor:

    positions = torch.arange(waveform.shape[1], device=waveform.device)
    valid = positions.unsqueeze(0) < waveform_lengths.unsqueeze(1)
    weight = valid.to(waveform.dtype)
    count = weight.sum(dim=1).clamp_min(1.0)
    mean = (waveform * weight).sum(dim=1) / count
    centered = (waveform - mean.unsqueeze(1)) * weight
    rms = torch.sqrt(centered.square().sum(dim=1) / count)
    ratio = 10.0 ** (float(noise_db) / 20.0)
    return (rms * ratio).clamp(min=float(minimum), max=float(maximum))


def waveform_mask_to_feature_mask(
    mask: torch.Tensor,
    *,
    hop_length: int,
    target_length: int,
    n_fft: int | None = None,
    require_full_support: bool = False,
) -> torch.Tensor:

    if require_full_support:
        if n_fft is None:
            raise ValueError("full-support mask must provide n_fft")
        left = int(n_fft) - int(hop_length)
        padded = F.pad(mask.float(), (left, 0), value=1.0)
        frames = F.avg_pool1d(
            padded.unsqueeze(1),
            kernel_size=int(n_fft),
            stride=int(hop_length),
        ).squeeze(1)
        return (frames >= 1.0 - 1e-7)[:, :target_length]

    pad = (-mask.shape[1]) % hop_length
    if pad:
        mask = F.pad(mask, (0, pad), value=False)
    frames = F.max_pool1d(
        mask.float().unsqueeze(1),
        kernel_size=hop_length,
        stride=hop_length,
    ).squeeze(1).bool()
    return frames[:, :target_length]


def downsample_mask_100_to_25(
    mask: torch.Tensor, target_length: int, *, require_all: bool = False
) -> torch.Tensor:
    pad = (-mask.shape[1]) % 4
    if pad:
        mask = F.pad(mask, (0, pad), value=False)
    if require_all:
        pooled = F.avg_pool1d(mask.float().unsqueeze(1), 4, 4).squeeze(1)
        result = pooled >= 1.0 - 1e-7
    else:
        result = F.max_pool1d(mask.float().unsqueeze(1), 4, 4).squeeze(1).bool()
    return result[:, :target_length]


def erode_loss_mask(
    loss_mask: torch.Tensor, local_window: int, *, causal: bool
) -> torch.Tensor:

    if local_window <= 1:
        return loss_mask
    radius_left = local_window - 1 if causal else local_window // 2
    radius_right = 0 if causal else local_window - 1 - local_window // 2
    eroded = loss_mask

    if radius_left > 0:
        shifted = F.pad(loss_mask, (1, 0), value=False)[:, :-1]
        eroded = eroded & shifted
    if radius_right > 0:
        shifted = F.pad(loss_mask, (0, 1), value=False)[:, 1:]
        eroded = eroded & shifted
    return eroded


class BestRQHead(nn.Module):
    def __init__(self, model_dim: int, codebook_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(model_dim, codebook_size)

    def forward(
        self,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.proj(hidden)
        active = loss_mask & (target_ids >= 0)
        active_frames = active.sum()
        if active.any():
            loss = F.cross_entropy(logits[active], target_ids[active])
            correct_frames = (
                logits[active].argmax(dim=-1) == target_ids[active]
            ).sum()
            accuracy = correct_frames.float() / active_frames
        else:
            loss = logits.sum() * 0.0
            accuracy = logits.new_zeros(())
            correct_frames = active_frames
        return loss, {
            "bestrq_loss": loss.detach(),
            "bestrq_accuracy": accuracy.detach(),
            "bestrq_active_frames": active_frames.detach().to(loss.dtype),
            "bestrq_loss_sum": (loss.detach() * active_frames).to(loss.dtype),
            "bestrq_correct_frames": correct_frames.detach().to(loss.dtype),
        }
