
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MelodyTokenBatch:
    token_ids: torch.Tensor
    frame_mask: torch.Tensor
    frame_rate: float = 6.25
    vocab_size: int = 256
    unvoiced_id: int = 255


def _hz_to_midi(frequency_hz: torch.Tensor) -> torch.Tensor:
    return 69.0 + 12.0 * torch.log2(frequency_hz / 440.0)


class MelodyTokenizer:

    frame_rate = 6.25
    vocab_size = 256
    unvoiced_id = 255
    pool_size = 8

    def __call__(
        self,
        pitch_hz_50: torch.Tensor,
        pitch_frame_mask: torch.Tensor | None = None,
    ) -> MelodyTokenBatch:
        if pitch_hz_50.ndim == 1:
            pitch_hz_50 = pitch_hz_50.unsqueeze(0)
        if pitch_hz_50.ndim != 2:
            raise ValueError(
                f"pitch_hz_50 must have shape [B, T] or [T]; received "
                f"{tuple(pitch_hz_50.shape)}"
            )
        if pitch_frame_mask is None:
            pitch_frame_mask = torch.ones_like(pitch_hz_50, dtype=torch.bool)
        elif pitch_frame_mask.ndim == 1:
            pitch_frame_mask = pitch_frame_mask.unsqueeze(0)
        if pitch_frame_mask.shape != pitch_hz_50.shape:
            raise ValueError("pitch_frame_mask and pitch_hz_50 must have matching shapes")

        batch, frames = pitch_hz_50.shape
        pooled_frames = (frames + self.pool_size - 1) // self.pool_size
        padded_frames = pooled_frames * self.pool_size
        padded_pitch = pitch_hz_50.new_zeros(batch, padded_frames)
        padded_mask = torch.zeros(
            batch, padded_frames, device=pitch_hz_50.device, dtype=torch.bool
        )
        padded_pitch[:, :frames] = pitch_hz_50
        padded_mask[:, :frames] = pitch_frame_mask
        grouped_pitch = padded_pitch.view(batch, pooled_frames, self.pool_size)
        grouped_mask = padded_mask.view(batch, pooled_frames, self.pool_size)
        voiced = grouped_mask & torch.isfinite(grouped_pitch) & (grouped_pitch > 0)
        values = grouped_pitch.masked_fill(~voiced, float("nan"))
        pooled_pitch = torch.nanmedian(values, dim=-1).values




        window_frames = grouped_mask.sum(dim=-1)
        pooled_voiced = voiced.sum(dim=-1) * 2 > window_frames
        pooled_pitch = torch.nan_to_num(pooled_pitch, nan=0.0)

        tokens = torch.full(
            (batch, pooled_frames),
            self.unvoiced_id,
            device=pitch_hz_50.device,
            dtype=torch.long,
        )
        for row in range(batch):
            row_voiced = pooled_voiced[row]
            if not row_voiced.any():
                continue
            midi = torch.round(_hz_to_midi(pooled_pitch[row, row_voiced]))
            center = torch.median(midi)
            relative = (midi - center).clamp(-127, 127).long() + 127
            tokens[row, row_voiced] = relative


        positions_in = torch.arange(frames, device=pitch_hz_50.device)
        last_valid = torch.where(
            pitch_frame_mask,
            positions_in.unsqueeze(0).expand_as(pitch_frame_mask),
            torch.full_like(positions_in.unsqueeze(0).expand_as(pitch_frame_mask), -1),
        ).max(dim=1).values
        valid_lengths = torch.div(
            last_valid + 1 + self.pool_size - 1,
            self.pool_size,
            rounding_mode="floor",
        )
        positions = torch.arange(pooled_frames, device=pitch_hz_50.device)
        output_mask = positions.unsqueeze(0) < valid_lengths.unsqueeze(1)
        tokens = tokens.masked_fill(~output_mask, self.unvoiced_id)
        return MelodyTokenBatch(token_ids=tokens, frame_mask=output_mask)
