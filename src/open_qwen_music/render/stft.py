
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .contracts import (
    AUDIO_CHANNELS,
    SAMPLE_RATE,
    SAMPLES_PER_LATENT_FRAME,
    SPEC_FRAMES_PER_LATENT_FRAME,
    STFT_BINS,
    STFT_HOP_LENGTH,
    STFT_LEFT_PADDING,
    STFT_N_FFT,
    STFT_WIN_LENGTH,
    STFT_CONTRACT_VERSION,
    lengths_to_mask,
    samples_to_stft_frames,
    validate_audio,
    validate_spectrum,
)
from .types import STFTOutput


@dataclass(frozen=True)
class STFTConfig:

    sample_rate: int = SAMPLE_RATE
    n_fft: int = STFT_N_FFT
    win_length: int = STFT_WIN_LENGTH
    hop_length: int = STFT_HOP_LENGTH
    center: bool = False
    normalized: bool = False
    drop_nyquist: bool = True

    explicit_left_padding: int = STFT_LEFT_PADDING
    boundary_window_floor: float = 0.0

    def __post_init__(self) -> None:
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError(f"Render STFT sampling rate must be {SAMPLE_RATE}")
        if self.n_fft != 960 or self.win_length != 960 or self.hop_length != 480:
            raise ValueError("The STFT contract fixes n_fft/win/hop at 960/960/480")
        if self.center:
            raise ValueError("Renderer STFT requires center=False and explicit padding")
        if self.explicit_left_padding != STFT_LEFT_PADDING:
            raise ValueError(f"explicit_left_padding must be {STFT_LEFT_PADDING}")
        if not 0.0 <= self.boundary_window_floor < 0.1:
            raise ValueError("boundary_window_floor must be in [0, 0.1)")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "STFTConfig":

        if not isinstance(values, Mapping):
            raise TypeError("STFT mapping must be a dict")
        required = (
            "n_fft",
            "win_length",
            "hop_length",
            "center",
            "drop_nyquist",
            "explicit_left_padding",
        )
        missing = [key for key in required if key not in values]
        if missing:
            raise ValueError(f"STFT mapping is missing field:  {missing}")
        return cls(
            sample_rate=int(values.get("sample_rate", SAMPLE_RATE)),
            n_fft=int(values["n_fft"]),
            win_length=int(values["win_length"]),
            hop_length=int(values["hop_length"]),
            center=bool(values["center"]),
            normalized=bool(values.get("normalized", False)),
            drop_nyquist=bool(values["drop_nyquist"]),
            explicit_left_padding=int(values["explicit_left_padding"]),
            boundary_window_floor=float(values.get("boundary_window_floor", 0.0)),
        )

    @property
    def bins(self) -> int:
        return self.n_fft // 2

    def frames_for_samples(self, samples: int) -> int:
        if samples < 0:
            raise ValueError("samples must not be negative")
        return samples_to_stft_frames(samples)

    def analysis_length(self, samples: int) -> int:
        frames = self.frames_for_samples(samples)
        return 0 if frames == 0 else self.n_fft + (frames - 1) * self.hop_length

    def right_padding(self, samples: int) -> int:
        frames = self.frames_for_samples(samples)
        if frames == 0:
            return 0
        return self.analysis_length(samples) - self.explicit_left_padding - samples


def _canonical_lengths(
    values: Tensor | Iterable[int] | None,
    *,
    batch: int,
    full_length: int,
    device: torch.device,
) -> Tensor:
    if values is None:
        return torch.full((batch,), full_length, dtype=torch.long, device=device)
    result = (
        values.to(device=device, dtype=torch.long)
        if isinstance(values, Tensor)
        else torch.as_tensor(list(values), dtype=torch.long, device=device)
    )
    if result.shape != (batch,):
        raise ValueError(f"lengths must have shape [{batch}],received {tuple(result.shape)}")
    if torch.any(result <= 0) or torch.any(result > full_length):
        raise ValueError("STFT only accepts 0 < length <= waveform.shape[-1]")
    return result


class StereoSTFT(nn.Module):

    def __init__(self, config: STFTConfig | None = None) -> None:
        super().__init__()
        self.config = config or STFTConfig()

        self.revision = STFT_CONTRACT_VERSION
        window = torch.hann_window(
            self.config.win_length, periodic=True, dtype=torch.float32
        )
        if self.config.boundary_window_floor:
            window = window.clamp_min(self.config.boundary_window_floor)
        self.register_buffer("window", window, persistent=False)

    def _frame_lengths(self, audio_lengths: Tensor) -> Tensor:
        latent_lengths = torch.div(
            audio_lengths + SAMPLES_PER_LATENT_FRAME - 1,
            SAMPLES_PER_LATENT_FRAME,
            rounding_mode="floor",
        )
        return latent_lengths * SPEC_FRAMES_PER_LATENT_FRAME

    def analyze(
        self,
        audio: Tensor,
        lengths: Tensor | Iterable[int] | None = None,
    ) -> STFTOutput:
        squeeze_batch = audio.ndim == 2
        if squeeze_batch:
            audio = audio.unsqueeze(0)
        if audio.ndim != 3:
            raise ValueError("audio must have shape [2, N] or [B, 2, N]")
        audio_lengths = _canonical_lengths(
            lengths,
            batch=audio.shape[0],
            full_length=audio.shape[-1],
            device=audio.device,
        )
        validate_audio(audio, audio_lengths)
        original_dtype = audio.dtype
        frame_lengths = self._frame_lengths(audio_lengths)
        max_frames = int(frame_lengths.max().item())
        required_total = self.config.n_fft + (max_frames - 1) * self.config.hop_length

        sample_mask = lengths_to_mask(
            audio_lengths, audio.shape[-1], device=audio.device
        ).unsqueeze(1)
        work = audio.float() * sample_mask
        required_audio_extent = required_total - self.config.explicit_left_padding
        right_pad = required_audio_extent - work.shape[-1]
        if right_pad < 0:

            work = work[..., :required_audio_extent]
            right_pad = 0
        work = F.pad(
            work,
            (self.config.explicit_left_padding, right_pad),
        )
        flat = work.reshape(-1, work.shape[-1])
        spectrum = torch.stft(
            flat,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window.to(device=audio.device),
            center=False,
            normalized=self.config.normalized,
            onesided=True,
            return_complex=True,
        )
        spectrum = (
            spectrum[:, : self.config.bins, :]
            if self.config.drop_nyquist
            else spectrum[:, 1:, :]
        )
        spectrum = spectrum.reshape(
            audio.shape[0], AUDIO_CHANNELS, self.config.bins, max_frames
        ).to(torch.complex64)
        frame_mask = lengths_to_mask(frame_lengths, max_frames, device=audio.device)
        spectrum = spectrum * frame_mask[:, None, None, :]
        result = STFTOutput(
            spectrum=spectrum,
            audio_lengths=audio_lengths,
            spectrum_lengths=frame_lengths,
            spectrum_mask=frame_mask,
            waveform_dtype=original_dtype,
            unbatched=squeeze_batch,
        )
        return result

    def forward(
        self,
        audio: Tensor,
        lengths: Tensor | Iterable[int] | None = None,
        *,
        return_info: bool = False,
    ) -> Tensor | STFTOutput:
        result = self.analyze(audio, lengths)
        if return_info:
            return result
        return result.spectrum.squeeze(0) if result.unbatched else result.spectrum

    def inverse(
        self,
        spectrum: Tensor | STFTOutput,
        lengths: Tensor | Iterable[int] | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> Tensor:

        output = spectrum if isinstance(spectrum, STFTOutput) else None
        values = output.spectrum if output is not None else spectrum
        squeeze_batch = output.unbatched if output is not None else values.ndim == 3
        if squeeze_batch:
            values = values.unsqueeze(0)
        if values.ndim != 4:
            raise ValueError("spectrum must have shape [2, 480, T] or [B, 2, 480, T]")
        batch, channels, bins, frames = values.shape
        validate_spectrum(values)
        if bins != self.config.bins or channels != AUDIO_CHANNELS:
            raise ValueError("Spectrum must have shape [B, 2, 480, T]")

        if lengths is None and output is not None:
            audio_lengths = output.audio_lengths.to(values.device)
        elif lengths is None:
            audio_lengths = torch.full(
                (batch,),
                frames * self.config.hop_length,
                dtype=torch.long,
                device=values.device,
            )
        else:
            max_supported = frames * self.config.hop_length
            audio_lengths = _canonical_lengths(
                lengths,
                batch=batch,
                full_length=max_supported,
                device=values.device,
            )
        expected_frames = self._frame_lengths(audio_lengths)
        if torch.any(expected_frames > frames):
            raise ValueError("Spectrum has too few frames for the requested waveform lengths")

        work = values.to(torch.complex64)
        frame_mask = lengths_to_mask(expected_frames, frames, device=values.device)
        work = work * frame_mask[:, None, None, :]
        missing = torch.zeros(
            (*work.shape[:2], 1, frames),
            dtype=work.dtype,
            device=work.device,
        )
        full = (
            torch.cat((work, missing), dim=2)
            if self.config.drop_nyquist
            else torch.cat((missing, work), dim=2)
        )
        flat = full.reshape(batch * channels, self.config.n_fft // 2 + 1, frames)
        framed = torch.fft.irfft(
            flat.transpose(1, 2),
            n=self.config.n_fft,
            dim=-1,
            norm="ortho" if self.config.normalized else "backward",
        )
        window = self.window.to(device=values.device, dtype=torch.float32)
        framed = framed.float() * window
        total_length = self.config.n_fft + (frames - 1) * self.config.hop_length
        columns = framed.transpose(1, 2)
        waveform = F.fold(
            columns,
            output_size=(1, total_length),
            kernel_size=(1, self.config.n_fft),
            stride=(1, self.config.hop_length),
        ).reshape(batch, channels, total_length)
        denominator_columns = (
            window.square()[None, :, None] * frame_mask[:, None, :].to(window.dtype)
        ).contiguous()
        denominator = F.fold(
            denominator_columns,
            output_size=(1, total_length),
            kernel_size=(1, self.config.n_fft),
            stride=(1, self.config.hop_length),
        ).reshape(batch, 1, total_length)
        waveform = waveform / denominator.clamp_min(1.0e-12)
        max_length = int(audio_lengths.max().item())
        start = self.config.explicit_left_padding
        waveform = waveform[..., start : start + max_length]
        sample_mask = lengths_to_mask(
            audio_lengths, max_length, device=values.device
        ).unsqueeze(1)
        waveform = waveform * sample_mask
        target_dtype = (
            dtype
            if dtype is not None
            else output.waveform_dtype
            if output is not None
            else torch.float32
        )
        waveform = waveform.to(target_dtype)
        return waveform.squeeze(0) if squeeze_batch else waveform

    synthesize = inverse


class EarV2NativeStereoSTFT(nn.Module):

    profile = "ear_v2_native_v1"
    explicit_left_padding = STFT_N_FFT // 2
    explicit_right_padding = STFT_HOP_LENGTH // 2

    def __init__(self) -> None:
        super().__init__()
        self.config: dict[str, Any] = {
            "profile": self.profile,
            "revision": STFT_CONTRACT_VERSION,
            "sample_rate": SAMPLE_RATE,
            "n_fft": STFT_N_FFT,
            "win_length": STFT_WIN_LENGTH,
            "hop_length": STFT_HOP_LENGTH,
            "window": "hann_periodic",
            "center": False,
            "normalized": False,
            "analysis_dtype": "float32",
            "explicit_left_padding": self.explicit_left_padding,
            "explicit_right_padding": self.explicit_right_padding,
            "keep_dc": True,
            "drop_nyquist": True,
            "bins": STFT_BINS,
        }
        self.register_buffer(
            "window",
            torch.hann_window(STFT_WIN_LENGTH, periodic=True, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _frame_lengths(audio_lengths: Tensor) -> Tensor:
        if bool((audio_lengths % SAMPLES_PER_LATENT_FRAME != 0).any()):
            raise ValueError("Native STFT diagnostics require an audio length of 1920 samples")
        if audio_lengths.numel() > 1 and not bool(
            (audio_lengths == audio_lengths[0]).all()
        ):
            raise ValueError("Native STFT diagnostics do not support ragged batches")
        return audio_lengths // STFT_HOP_LENGTH

    def analyze(
        self,
        audio: Tensor,
        lengths: Tensor | Iterable[int] | None = None,
    ) -> STFTOutput:
        squeeze_batch = audio.ndim == 2
        if squeeze_batch:
            audio = audio.unsqueeze(0)
        if audio.ndim != 3 or audio.shape[1] != AUDIO_CHANNELS:
            raise ValueError("Native STFT audio must have shape [2, N] or [B, 2, N]")
        audio_lengths = _canonical_lengths(
            lengths,
            batch=audio.shape[0],
            full_length=audio.shape[-1],
            device=audio.device,
        )
        validate_audio(audio, audio_lengths)
        frame_lengths = self._frame_lengths(audio_lengths)
        max_audio_length = int(audio_lengths.max().item())
        max_frames = int(frame_lengths.max().item())
        sample_mask = lengths_to_mask(
            audio_lengths,
            audio.shape[-1],
            device=audio.device,
        ).unsqueeze(1)
        work = (audio.float() * sample_mask)[..., :max_audio_length]
        work = F.pad(
            work,
            (self.explicit_left_padding, self.explicit_right_padding),
        )
        spectrum = torch.stft(
            work.flatten(0, 1),
            n_fft=STFT_N_FFT,
            hop_length=STFT_HOP_LENGTH,
            win_length=STFT_WIN_LENGTH,
            window=self.window.to(device=audio.device),
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        if spectrum.shape[-1] != max_frames:
            raise RuntimeError(
                f"Native STFT frame count changed unexpectedly: {spectrum.shape[-1]}!={max_frames}"
            )
        spectrum = (
            spectrum[:, :STFT_BINS]
            .reshape(
                audio.shape[0],
                AUDIO_CHANNELS,
                STFT_BINS,
                max_frames,
            )
            .to(torch.complex64)
        )
        frame_mask = lengths_to_mask(
            frame_lengths,
            max_frames,
            device=audio.device,
        )
        spectrum = spectrum * frame_mask[:, None, None, :]
        return STFTOutput(
            spectrum=spectrum,
            audio_lengths=audio_lengths,
            spectrum_lengths=frame_lengths,
            spectrum_mask=frame_mask,
            waveform_dtype=audio.dtype,
            unbatched=squeeze_batch,
        )

    def forward(
        self,
        audio: Tensor,
        lengths: Tensor | Iterable[int] | None = None,
        *,
        return_info: bool = False,
    ) -> Tensor | STFTOutput:
        result = self.analyze(audio, lengths)
        if return_info:
            return result
        return result.spectrum.squeeze(0) if result.unbatched else result.spectrum

    def inverse(
        self,
        spectrum: Tensor | STFTOutput,
        lengths: Tensor | Iterable[int] | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        output = spectrum if isinstance(spectrum, STFTOutput) else None
        values = output.spectrum if output is not None else spectrum
        squeeze_batch = output.unbatched if output is not None else values.ndim == 3
        if squeeze_batch:
            values = values.unsqueeze(0)
        validate_spectrum(values)
        batch, channels, bins, frames = values.shape
        if channels != AUDIO_CHANNELS or bins != STFT_BINS:
            raise ValueError("Native STFT spectrum must have shape [B, 2, 480, T]")
        if lengths is None and output is not None:
            audio_lengths = output.audio_lengths.to(values.device)
        elif lengths is None:
            audio_lengths = torch.full(
                (batch,),
                frames * STFT_HOP_LENGTH,
                dtype=torch.long,
                device=values.device,
            )
        else:
            audio_lengths = _canonical_lengths(
                lengths,
                batch=batch,
                full_length=frames * STFT_HOP_LENGTH,
                device=values.device,
            )
        frame_lengths = self._frame_lengths(audio_lengths)
        if not bool((frame_lengths == frames).all()):
            raise ValueError("Native inverse STFT requires every item to have the full spectrum frame count")
        frame_mask = lengths_to_mask(frame_lengths, frames, device=values.device)
        work = values.to(torch.complex64) * frame_mask[:, None, None, :]
        nyquist = torch.zeros(
            batch,
            channels,
            1,
            frames,
            dtype=work.dtype,
            device=work.device,
        )
        full = torch.cat((work, nyquist), dim=2)
        max_audio_length = int(audio_lengths.max().item())
        waveform = torch.istft(
            full.reshape(batch * channels, STFT_N_FFT // 2 + 1, frames),
            n_fft=STFT_N_FFT,
            hop_length=STFT_HOP_LENGTH,
            win_length=STFT_WIN_LENGTH,
            window=self.window.to(device=values.device),
            center=True,
            normalized=False,
            onesided=True,
            length=max_audio_length + self.explicit_right_padding,
        ).reshape(batch, channels, -1)[..., :max_audio_length]
        waveform = waveform * lengths_to_mask(
            audio_lengths,
            max_audio_length,
            device=values.device,
        ).unsqueeze(1)
        target_dtype = (
            dtype
            if dtype is not None
            else output.waveform_dtype
            if output is not None
            else torch.float32
        )
        waveform = waveform.to(target_dtype)
        return waveform.squeeze(0) if squeeze_batch else waveform

    synthesize = inverse


def build_stft_from_mapping(values: Mapping[str, Any]) -> nn.Module:

    if not isinstance(values, Mapping):
        raise TypeError("STFT mapping must be a dict")
    profile = str(values.get("profile", "open_qwen_music_v1"))
    if profile == "open_qwen_music_v1":
        return StereoSTFT(STFTConfig.from_mapping(values))
    if profile != EarV2NativeStereoSTFT.profile:
        raise ValueError(f"Unknown Renderer STFT profile:  {profile}")
    if int(values.get("sample_rate", SAMPLE_RATE)) != SAMPLE_RATE:
        raise ValueError(f"Native STFT sample rate must be {SAMPLE_RATE}")
    expected = {
        "revision": STFT_CONTRACT_VERSION,
        "n_fft": STFT_N_FFT,
        "win_length": STFT_WIN_LENGTH,
        "hop_length": STFT_HOP_LENGTH,
        "window": "hann_periodic",
        "center": False,
        "normalized": False,
        "analysis_dtype": "float32",
        "keep_dc": True,
        "drop_nyquist": True,
        "bins": STFT_BINS,
        "explicit_left_padding": STFT_N_FFT // 2,
        "explicit_right_padding": STFT_HOP_LENGTH // 2,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": values.get(key)}
        for key, expected_value in expected.items()
        if values.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Native STFT config does not match: {mismatches}")
    return EarV2NativeStereoSTFT()


RenderSTFT = StereoSTFT
ComplexSTFT = StereoSTFT
STFT = StereoSTFT


def stft(
    audio: Tensor,
    lengths: Tensor | Iterable[int] | None = None,
    *,
    config: STFTConfig | None = None,
    return_info: bool = False,
) -> Tensor | STFTOutput:
    transform = StereoSTFT(config).to(audio.device)
    return transform(audio, lengths, return_info=return_info)


def istft(
    spectrum: Tensor | STFTOutput,
    lengths: Tensor | Iterable[int] | None = None,
    *,
    config: STFTConfig | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    values = spectrum.spectrum if isinstance(spectrum, STFTOutput) else spectrum
    transform = StereoSTFT(config).to(values.device)
    return transform.inverse(spectrum, lengths, dtype=dtype)
