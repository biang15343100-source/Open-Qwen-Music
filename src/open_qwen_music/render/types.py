
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .contracts import (
    LATENT_DIM,
    SAMPLE_RATE,
    lengths_to_mask,
    sample_lengths_to_latent_lengths,
    sample_lengths_to_stft_lengths,
    validate_audio,
    validate_latents,
    validate_semantic_ids,
    validate_spectrum,
)


@dataclass
class STFTOutput:

    spectrum: Tensor
    audio_lengths: Tensor
    spectrum_lengths: Tensor
    spectrum_mask: Tensor
    waveform_dtype: torch.dtype = torch.float32
    unbatched: bool = False

    def __post_init__(self) -> None:
        validate_spectrum(self.spectrum, self.spectrum_lengths)
        if self.audio_lengths.shape != self.spectrum_lengths.shape:
            raise ValueError("audio_lengths and spectrum_lengths have different batch sizes")
        expected = lengths_to_mask(
            self.spectrum_lengths,
            self.spectrum.shape[-1],
            device=self.spectrum.device,
        )
        if not torch.equal(self.spectrum_mask, expected):
            raise ValueError("spectrum_mask and spectrum_lengths do not match")

    @property
    def values(self) -> Tensor:
        return self.spectrum


@dataclass
class LatentOutput:
    latents: Tensor
    lengths: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        validate_latents(self.latents, self.lengths)
        expected = lengths_to_mask(
            self.lengths, self.latents.shape[1], device=self.latents.device
        )
        if not torch.equal(self.mask, expected):
            raise ValueError("latent_mask and latent_lengths do not match")


@dataclass
class RenderAudioBatch:

    sample_ids: list[str]
    audio: Tensor
    audio_lengths: Tensor
    audio_mask: Tensor
    duration_seconds: Tensor
    media_bandwidth_hz: Tensor
    magnitude_max_hz: Tensor
    phase_max_hz: Tensor
    stereo_max_hz: Tensor
    adversarial_max_hz: Tensor
    waveform_adversarial_enabled: Tensor
    provenance: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        lengths = validate_audio(self.audio, self.audio_lengths)
        batch = self.audio.shape[0]
        if len(self.sample_ids) != batch:
            raise ValueError("sample_ids and audio have different batch sizes")
        if self.duration_seconds.shape != (batch,):
            raise ValueError("duration_seconds must be [B]")
        if (
            self.media_bandwidth_hz.shape != (batch,)
            or not self.media_bandwidth_hz.is_floating_point()
            or not torch.isfinite(self.media_bandwidth_hz).all()
            or bool((self.media_bandwidth_hz <= 0).any())
            or bool((self.media_bandwidth_hz > SAMPLE_RATE / 2).any())
        ):
            raise ValueError("media_bandwidth_hz must be a finite floating-point tensor in (0, 24000] with shape [B]")
        for name in (
            "magnitude_max_hz",
            "phase_max_hz",
            "stereo_max_hz",
            "adversarial_max_hz",
        ):
            value = getattr(self, name)
            if (
                value.shape != (batch,)
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
                or bool((value < 0).any())
                or bool((value > self.media_bandwidth_hz).any())
            ):
                raise ValueError(f"{name} must be in [0, media_bandwidth_hz]")
        if (
            self.waveform_adversarial_enabled.shape != (batch,)
            or self.waveform_adversarial_enabled.dtype != torch.bool
        ):
            raise ValueError("waveform_adversarial_enabled must be a bool tensor with shape [B]")
        expected = lengths_to_mask(
            lengths, self.audio.shape[-1], device=self.audio.device
        )
        if not torch.equal(self.audio_mask, expected):
            raise ValueError("audio_mask and audio_lengths do not match")
        if self.provenance and len(self.provenance) != batch:
            raise ValueError("provenance and audio have different batch sizes")

    @property
    def spectrum_lengths(self) -> Tensor:
        return sample_lengths_to_stft_lengths(self.audio_lengths)

    @property
    def latent_lengths(self) -> Tensor:
        return sample_lengths_to_latent_lengths(self.audio_lengths)

    def to(self, device: torch.device | str) -> "RenderAudioBatch":
        return RenderAudioBatch(
            sample_ids=self.sample_ids,
            audio=self.audio.to(device),
            audio_lengths=self.audio_lengths.to(device),
            audio_mask=self.audio_mask.to(device),
            duration_seconds=self.duration_seconds.to(device),
            media_bandwidth_hz=self.media_bandwidth_hz.to(device),
            magnitude_max_hz=self.magnitude_max_hz.to(device),
            phase_max_hz=self.phase_max_hz.to(device),
            stereo_max_hz=self.stereo_max_hz.to(device),
            adversarial_max_hz=self.adversarial_max_hz.to(device),
            waveform_adversarial_enabled=self.waveform_adversarial_enabled.to(device),
            provenance=self.provenance,
        )


@dataclass
class RenderBatch(RenderAudioBatch):

    semantic_ids: Tensor | None = None
    semantic_mask: Tensor | None = None
    latents: Tensor | None = None
    latent_mask: Tensor | None = None
    description_input_ids: Tensor | None = None
    description_mask: Tensor | None = None
    lyrics_input_ids: Tensor | None = None
    lyrics_mask: Tensor | None = None
    revisions: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.semantic_ids is None:
            return
        if self.semantic_mask is None:
            raise ValueError("semantic_ids and semantic_mask must be provided together")
        semantic_lengths = validate_semantic_ids(self.semantic_ids, self.semantic_mask)
        if self.latents is not None:
            if self.latent_mask is None:
                raise ValueError("latents and latent_mask must be provided together")
            latent_lengths = validate_latents(
                self.latents,
                self.latent_mask.sum(dim=1, dtype=torch.long),
            )
            if self.latents.shape[-1] != LATENT_DIM:
                raise ValueError("latent_dim must be 128")
            if not torch.equal(semantic_lengths, latent_lengths):
                raise ValueError("semantic and latent effective lengths must match for every sample")
        for name, ids, mask in (
            ("description", self.description_input_ids, self.description_mask),
            ("lyrics", self.lyrics_input_ids, self.lyrics_mask),
        ):
            if (ids is None) != (mask is None):
                raise ValueError(f"{name} IDs and masks must be provided together")
            if ids is not None and (
                ids.ndim != 2
                or ids.dtype != torch.long
                or mask is None
                or mask.shape != ids.shape
                or mask.dtype != torch.bool
            ):
                raise ValueError(f"{name} must contain int64 IDs and a bool mask with the same shape")


@dataclass
class RenderOutput:
    waveform: Tensor
    sample_rate: int = SAMPLE_RATE
    audio_lengths: Tensor | None = None
    latent_frames: int | Tensor = 0
    seed: int = 0
    num_steps: int = 0
    cfg_scale: float = 1.0
    solver: str = ""
    checkpoint_revision: str = ""

    def __post_init__(self) -> None:
        lengths = validate_audio(self.waveform, self.audio_lengths)
        if not torch.isfinite(self.waveform).all():
            raise ValueError("Renderer output waveform contains NaN/Inf")
        peak = float(self.waveform.detach().abs().max())
        if peak > 1.0:
            raise ValueError(f"Renderer output waveform peak={peak:.6f}exceeds1.0")
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError(f"Renderer output sample rate must be {SAMPLE_RATE}")
        self.audio_lengths = lengths


@dataclass(frozen=True)
class RatioMetric:

    numerator: float
    denominator: float

    @property
    def value(self) -> float:
        return self.numerator / self.denominator if self.denominator > 0 else 0.0
