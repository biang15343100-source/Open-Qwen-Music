
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .contracts import (
    AUDIO_CHANNELS,
    SAMPLE_RATE,
    STFT_BINS,
    STFT_N_FFT,
    lengths_to_mask,
    validate_spectrum,
)
from .stft import EarV2NativeStereoSTFT, StereoSTFT

ZERO_OUTPUT_INITIALIZATION = "zero_output_projection_v1"
REAL_NYQUIST_MODE = "real_only_zero_identity_v1"
EAR_VAE2_LINEAR_ADDITIVE_MAGNITUDE = "linear_additive_v1"
MID_MAGNITUDE_OUTPUT_HEADS = "band_separated_mid_magnitude_v1"
MID_MAGNITUDE_TRAINING_PROFILE = "mid_magnitude_with_trunk_v1"


@dataclass(frozen=True)
class EarVAE2PublicRefinerConfig:
    width: int = 256
    intermediate_dim: int = 1_024
    depth: int = 12
    kernel_size: int = 7
    layer_scale_init_value: float | None = None
    layer_norm_eps: float = 1.0e-5
    block_layer_norm_eps: float = 1.0e-6
    low_mid_hz: float = 1_500.0
    mid_high_hz: float = 4_000.0
    magnitude_epsilon: float = 1.0e-8
    max_log_magnitude_residual: float = 4.0
    revision: str = "open-qwen-music-refiner-v1"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.intermediate_dim <= 0 or self.depth <= 0:
            raise ValueError("Refiner width and depth must be positive")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("Refiner kernel_size must be a positive odd integer")
        if not 0.0 < self.low_mid_hz < self.mid_high_hz < self.sample_rate / 2:
            raise ValueError("Refiner band boundaries are invalid")
        for name in (
            "layer_norm_eps",
            "block_layer_norm_eps",
            "magnitude_epsilon",
            "max_log_magnitude_residual",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Refiner {name} must be finite and positive")
        if self.layer_scale_init_value is not None and (
            not math.isfinite(self.layer_scale_init_value)
            or self.layer_scale_init_value <= 0.0
        ):
            raise ValueError(
                "Refiner layer_scale_init_value must be finite and positive or None"
            )
        if self.revision != "open-qwen-music-refiner-v1":
            raise ValueError("Unsupported Refiner revision")

    @classmethod
    def from_dict(
        cls,
        values: dict[str, Any],
    ) -> "EarVAE2PublicRefinerConfig":
        values = dict(values)
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"Refiner config contains unknown fields: {sorted(unknown)}"
            )
        return cls(**values)

    @property
    def magnitude_gate_enabled(self) -> bool:
        return False

    @property
    def input_freq_bins(self) -> int:
        return STFT_BINS

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def initialization_profile(self) -> str:
        return ZERO_OUTPUT_INITIALIZATION

    @property
    def nyquist_mode(self) -> str:
        return REAL_NYQUIST_MODE

    @property
    def magnitude_residual_mode(self) -> str:
        return EAR_VAE2_LINEAR_ADDITIVE_MAGNITUDE

    @property
    def output_head_profile(self) -> str:
        return MID_MAGNITUDE_OUTPUT_HEADS

    @property
    def training_profile(self) -> str:
        return MID_MAGNITUDE_TRAINING_PROFILE


def normalize_refiner_checkpoint_config(values: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the current public Refiner schema at checkpoint load time."""
    normalized = dict(values)
    if normalized.get("revision") != "open-qwen-music-refiner-v1":
        raise ValueError("Refiner checkpoint has an unsupported revision")
    return normalized

def ear_vae2_public_band_splits(
    config: EarVAE2PublicRefinerConfig,
) -> tuple[int, int]:

    n_fft = 2 * config.input_freq_bins
    low = int(round(config.low_mid_hz * n_fft / config.sample_rate))
    high = int(round(config.mid_high_hz * n_fft / config.sample_rate))
    low = max(1, min(low, config.input_freq_bins - 2))
    high = max(low + 1, min(high, config.input_freq_bins - 1))
    return low, high


class EarVAE2PublicConvNeXt1DBlock(nn.Module):

    def __init__(
        self,
        width: int,
        intermediate_dim: int,
        *,
        kernel_size: int,
        layer_scale_init_value: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()

        self.dwconv = nn.Conv1d(
            width,
            width,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=width,
        )
        self.norm = nn.LayerNorm(width, eps=layer_norm_eps)
        self.pwconv1 = nn.Linear(width, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, width)
        self.gamma = nn.Parameter(torch.full((width,), float(layer_scale_init_value)))

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        outputs = self.dwconv(inputs).transpose(1, 2)
        outputs = self.norm(outputs)
        outputs = self.pwconv1(outputs)
        outputs = self.act(outputs)
        outputs = self.pwconv2(outputs)
        outputs = outputs * self.gamma
        return residual + outputs.transpose(1, 2)


class _EarVAE2PublicRefinerCore(nn.Module):

    def __init__(
        self,
        config: EarVAE2PublicRefinerConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = EarVAE2PublicRefinerConfig()
        elif isinstance(config, dict):
            config = EarVAE2PublicRefinerConfig.from_dict(config)
        self.config = config
        self.split_low, self.split_high = ear_vae2_public_band_splits(config)
        layer_scale = (
            1.0 / config.depth
            if config.layer_scale_init_value is None
            else config.layer_scale_init_value
        )
        self.embed = nn.Conv1d(
            2 * config.input_freq_bins,
            config.width,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2,
        )

        self.norm = nn.LayerNorm(config.width, eps=config.layer_norm_eps)
        self.convnext = nn.ModuleList(
            [
                EarVAE2PublicConvNeXt1DBlock(
                    config.width,
                    config.intermediate_dim,
                    kernel_size=config.kernel_size,
                    layer_scale_init_value=layer_scale,
                    layer_norm_eps=config.block_layer_norm_eps,
                )
                for _ in range(config.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(config.width, eps=config.layer_norm_eps)
        middle_bins = self.split_high - self.split_low
        self.low_phase_head = nn.Linear(config.width, self.split_low)
        self.middle_magnitude_head = nn.Linear(config.width, middle_bins)
        self.middle_phase_head = nn.Linear(config.width, middle_bins)
        self.high_magnitude_head = nn.Linear(
            config.width,
            config.input_freq_bins - self.split_high,
        )
        self.nyquist_head = nn.Linear(config.width, 1)
        for head in (
            self.low_phase_head,
            self.middle_magnitude_head,
            self.middle_phase_head,
            self.high_magnitude_head,
            self.nyquist_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for head in (
            self.low_phase_head,
            self.middle_phase_head,
            self.high_magnitude_head,
            self.nyquist_head,
        ):
            head.requires_grad_(False)

    def forward(self, coarse: Tensor) -> Tensor:
        validate_spectrum(coarse)
        if coarse.shape[-2] != self.config.input_freq_bins:
            raise ValueError(
                "Refiner input frequency-bin count does not match: "
                f"{coarse.shape[-2]} != {self.config.input_freq_bins}"
            )
        batch, channels, frequencies, frames = coarse.shape
        work = coarse.to(torch.complex64).reshape(
            batch * channels,
            frequencies,
            frames,
        )
        features = torch.cat((work.real, work.imag), dim=1)
        hidden = self.embed(features)
        hidden = self.norm(hidden.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            hidden = block(hidden)
        hidden = self.final_norm(hidden.transpose(1, 2))

        low = self.split_low
        high = self.split_high
        low_phase = self.low_phase_head(hidden).transpose(1, 2)
        middle_magnitude = self.middle_magnitude_head(hidden).transpose(1, 2)
        middle_phase = self.middle_phase_head(hidden).transpose(1, 2)
        high_magnitude = self.high_magnitude_head(hidden).transpose(1, 2)
        nyquist_real = self.nyquist_head(hidden).transpose(1, 2)

        magnitude = work.abs()
        phase = torch.angle(work)
        low_spectrum = torch.polar(
            magnitude[:, :low],
            phase[:, :low] + low_phase,
        )
        middle_spectrum = torch.polar(
            (magnitude[:, low:high] + middle_magnitude).clamp_min(0.0),
            phase[:, low:high] + middle_phase,
        )
        high_spectrum = torch.polar(
            (magnitude[:, high:] + high_magnitude).clamp_min(0.0),
            phase[:, high:],
        )

        low_spectrum = torch.where(
            low_phase == 0,
            work[:, :low] + low_spectrum - low_spectrum.detach(),
            low_spectrum,
        )
        middle_spectrum = torch.where(
            (middle_magnitude == 0) & (middle_phase == 0),
            work[:, low:high] + middle_spectrum - middle_spectrum.detach(),
            middle_spectrum,
        )
        high_spectrum = torch.where(
            high_magnitude == 0,
            work[:, high:] + high_spectrum - high_spectrum.detach(),
            high_spectrum,
        )
        nyquist = torch.complex(nyquist_real, torch.zeros_like(nyquist_real))
        refined = torch.cat(
            (low_spectrum, middle_spectrum, high_spectrum, nyquist),
            dim=1,
        )
        return refined.reshape(batch, channels, frequencies + 1, frames)


class EarVAE2PublicRefiner(_EarVAE2PublicRefinerCore):

    def __init__(
        self,
        config: EarVAE2PublicRefinerConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        magnitude = torch.zeros(STFT_BINS, dtype=torch.float32)
        phase = torch.zeros(STFT_BINS, dtype=torch.float32)
        magnitude[self.split_low : self.split_high] = 1.0
        self.register_buffer("_magnitude_mask", magnitude, persistent=False)
        self.register_buffer("_phase_mask", phase, persistent=False)

    @property
    def magnitude_mask(self) -> Tensor:
        return self._magnitude_mask

    @property
    def phase_mask(self) -> Tensor:
        return self._phase_mask


def refiner_config_from_mapping(values: Mapping[str, Any]) -> EarVAE2PublicRefinerConfig:
    if not isinstance(values, Mapping):
        raise TypeError("model.refiner must be a mapping")
    return EarVAE2PublicRefinerConfig.from_dict(dict(values))


def build_refiner_from_mapping(values: Mapping[str, Any]) -> EarVAE2PublicRefiner:
    return EarVAE2PublicRefiner(refiner_config_from_mapping(values))


def apply_refiner(refiner: nn.Module, coarse: Tensor) -> Tensor:
    return refiner(coarse)


def refiner_spectrum_for_loss(spectrum: Tensor) -> Tensor:

    if spectrum.ndim != 4 or spectrum.shape[1] != AUDIO_CHANNELS:
        raise ValueError("Refiner spectrum must be a complex tensor with shape [B, 2, F, T]")
    if not spectrum.is_complex():
        raise ValueError("Refiner spectrum must be complex")
    if spectrum.shape[2] == STFT_BINS:
        return spectrum
    if spectrum.shape[2] == STFT_BINS + 1:
        return spectrum[:, :, :STFT_BINS]
    raise ValueError(f"invalid Refiner spectrum frequency-bin count: {spectrum.shape[2]}")


def _refiner_audio_lengths(
    lengths: Tensor | Iterable[int] | None,
    *,
    batch: int,
    frames: int,
    hop_length: int,
    device: torch.device,
) -> Tensor:
    maximum = frames * hop_length
    if lengths is None:
        return torch.full((batch,), maximum, dtype=torch.long, device=device)
    result = (
        lengths.to(device=device, dtype=torch.long)
        if isinstance(lengths, Tensor)
        else torch.as_tensor(list(lengths), dtype=torch.long, device=device)
    )
    if (
        result.shape != (batch,)
        or bool((result <= 0).any())
        or bool((result > maximum).any())
    ):
        raise ValueError("Refiner inverse lengths are invalid")
    return result


def _inverse_full_spectrum_standard(
    stft: StereoSTFT,
    spectrum: Tensor,
    lengths: Tensor | Iterable[int] | None,
    *,
    dtype: torch.dtype | None,
) -> Tensor:
    squeeze_batch = spectrum.ndim == 3
    values = spectrum.unsqueeze(0) if squeeze_batch else spectrum
    if (
        values.ndim != 4
        or values.shape[1] != AUDIO_CHANNELS
        or values.shape[2] != STFT_BINS + 1
        or not values.is_complex()
    ):
        raise ValueError("complete Refiner spectrum must be a complex tensor with shape [B, 2, 481, T]")
    if not stft.config.drop_nyquist:
        raise ValueError("Refiner requires the STFT to preserve DC and delegates Nyquist reconstruction to the Refiner")
    batch, channels, _, frames = values.shape
    audio_lengths = _refiner_audio_lengths(
        lengths,
        batch=batch,
        frames=frames,
        hop_length=stft.config.hop_length,
        device=values.device,
    )
    expected_frames = stft._frame_lengths(audio_lengths)
    if bool((expected_frames > frames).any()):
        raise ValueError("complete Refiner spectrum has too few frames")
    frame_mask = lengths_to_mask(expected_frames, frames, device=values.device)
    work = values.to(torch.complex64) * frame_mask[:, None, None, :]
    flat = work.reshape(batch * channels, STFT_N_FFT // 2 + 1, frames)
    framed = torch.fft.irfft(
        flat.transpose(1, 2),
        n=stft.config.n_fft,
        dim=-1,
        norm="ortho" if stft.config.normalized else "backward",
    )
    window = stft.window.to(device=values.device, dtype=torch.float32)
    framed = framed.float() * window
    total_length = stft.config.n_fft + (frames - 1) * stft.config.hop_length
    waveform = F.fold(
        framed.transpose(1, 2),
        output_size=(1, total_length),
        kernel_size=(1, stft.config.n_fft),
        stride=(1, stft.config.hop_length),
    ).reshape(batch, channels, total_length)
    denominator = F.fold(
        (
            window.square()[None, :, None] * frame_mask[:, None, :].to(window.dtype)
        ).contiguous(),
        output_size=(1, total_length),
        kernel_size=(1, stft.config.n_fft),
        stride=(1, stft.config.hop_length),
    ).reshape(batch, 1, total_length)
    waveform = waveform / denominator.clamp_min(1.0e-12)
    maximum = int(audio_lengths.max().item())
    start = stft.config.explicit_left_padding
    waveform = waveform[..., start : start + maximum]
    waveform = waveform * lengths_to_mask(
        audio_lengths,
        maximum,
        device=values.device,
    ).unsqueeze(1)
    waveform = waveform.to(dtype or torch.float32)
    return waveform.squeeze(0) if squeeze_batch else waveform


def _inverse_full_spectrum_ear(
    stft: EarV2NativeStereoSTFT,
    spectrum: Tensor,
    lengths: Tensor | Iterable[int] | None,
    *,
    dtype: torch.dtype | None,
) -> Tensor:
    squeeze_batch = spectrum.ndim == 3
    values = spectrum.unsqueeze(0) if squeeze_batch else spectrum
    if (
        values.ndim != 4
        or values.shape[1] != AUDIO_CHANNELS
        or values.shape[2] != STFT_BINS + 1
        or not values.is_complex()
    ):
        raise ValueError("complete Refiner spectrum must be a complex tensor with shape [B, 2, 481, T]")
    batch, channels, _, frames = values.shape
    audio_lengths = _refiner_audio_lengths(
        lengths,
        batch=batch,
        frames=frames,
        hop_length=STFT_N_FFT // 2,
        device=values.device,
    )
    frame_lengths = stft._frame_lengths(audio_lengths)
    if not bool((frame_lengths == frames).all()):
        raise ValueError("native complete Refiner inverse requires the valid-frame count to match the spectrum frame count")
    frame_mask = lengths_to_mask(frame_lengths, frames, device=values.device)
    work = values.to(torch.complex64) * frame_mask[:, None, None, :]
    maximum = int(audio_lengths.max().item())
    waveform = torch.istft(
        work.reshape(batch * channels, STFT_N_FFT // 2 + 1, frames),
        n_fft=STFT_N_FFT,
        hop_length=STFT_N_FFT // 2,
        win_length=STFT_N_FFT,
        window=stft.window.to(device=values.device),
        center=True,
        normalized=False,
        onesided=True,
        length=maximum + stft.explicit_right_padding,
    ).reshape(batch, channels, -1)[..., :maximum]
    waveform = waveform * lengths_to_mask(
        audio_lengths,
        maximum,
        device=values.device,
    ).unsqueeze(1)
    waveform = waveform.to(dtype or torch.float32)
    return waveform.squeeze(0) if squeeze_batch else waveform


def inverse_refiner_spectrum(
    stft: nn.Module,
    spectrum: Tensor,
    lengths: Tensor | Iterable[int] | None = None,
    *,
    dtype: torch.dtype | None = None,
) -> Tensor:

    bins = spectrum.shape[-2] if spectrum.ndim in {3, 4} else -1
    if bins == STFT_BINS:
        return stft.inverse(spectrum, lengths=lengths, dtype=dtype)
    if bins != STFT_BINS + 1:
        raise ValueError(f"invalid Refiner spectrum frequency-bin count: {bins}")
    if isinstance(stft, StereoSTFT):
        return _inverse_full_spectrum_standard(
            stft,
            spectrum,
            lengths,
            dtype=dtype,
        )
    if isinstance(stft, EarV2NativeStereoSTFT):
        return _inverse_full_spectrum_ear(
            stft,
            spectrum,
            lengths,
            dtype=dtype,
        )
    raise TypeError(f"unsupported STFT type for complete 481-bin output: {type(stft).__name__}")
