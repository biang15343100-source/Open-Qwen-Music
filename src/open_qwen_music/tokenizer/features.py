
from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


def _hz_to_mel(freq: torch.Tensor, scale: str = "htk") -> torch.Tensor:
    if scale == "htk":
        return 2595.0 * torch.log10(1.0 + freq / 700.0)

    f_sp = 200.0 / 3.0
    mel = freq / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    log_step = math.log(6.4) / 27.0
    return torch.where(
        freq >= min_log_hz,
        min_log_mel + torch.log((freq / min_log_hz).clamp_min(1e-12)) / log_step,
        mel,
    )


def _mel_to_hz(mel: torch.Tensor, scale: str = "htk") -> torch.Tensor:
    if scale == "htk":
        return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    log_step = math.log(6.4) / 27.0
    return torch.where(
        mel >= min_log_mel,
        min_log_hz * torch.exp(log_step * (mel - min_log_mel)),
        f_sp * mel,
    )


MEL_FILTER_NORMS = ("slaney", "peak")
MEL_SCALES = ("htk", "slaney")
LOG_MODES = ("additive", "clamp", "db", "log1p")
FEATURE_CENTERING_MODES = ("none", "frame")
TOP_DB_SCOPES = ("sample", "frame")







CHROMA_MODES = ("soft", "hard")


CHROMA_DEFAULTS = {"mode": "soft", "n_fft": 4096}
CHROMA_PRE_20260804_DEFAULTS = {"mode": "hard", "n_fft": 1024}


def resolve_chroma_config(
    chroma_config: dict | None, defaults: dict | None = None
) -> dict:

    defaults = CHROMA_DEFAULTS if defaults is None else defaults
    source = chroma_config or {}
    resolved = {}
    for key, fallback in defaults.items():
        value = source.get(key)
        if value is None:
            value = fallback
        resolved[key] = value.lower() if isinstance(value, str) else int(value)
    return resolved


FRONTEND_DEFAULTS = {
    "mel_filter_norm": "slaney",
    "mel_scale": "htk",
    "log_mode": "additive",
    "log_knee": 1e-4,
    "top_db": None,
    "top_db_scope": "sample",
    "feature_centering": "none",
}
@lru_cache(maxsize=32)
def _mel_filter_cpu(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
    norm: str = "slaney",
    mel_scale: str = "htk",
) -> torch.Tensor:

    if norm not in MEL_FILTER_NORMS:
        raise ValueError(f"mel_filter_norm must be one of {MEL_FILTER_NORMS}; received {norm!r}")
    if mel_scale not in MEL_SCALES:
        raise ValueError(f"mel_scale must be one of {MEL_SCALES}; received {mel_scale!r}")
    low = _hz_to_mel(torch.tensor(f_min), mel_scale)
    high = _hz_to_mel(torch.tensor(f_max), mel_scale)
    mel_points = torch.linspace(low, high, n_mels + 2)
    hz_points = _mel_to_hz(mel_points, mel_scale)


    fft_freqs = torch.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    left, center, right = hz_points[:-2], hz_points[1:-1], hz_points[2:]
    rising = (fft_freqs.unsqueeze(0) - left.unsqueeze(1)) / (center - left).clamp_min(
        1e-8
    ).unsqueeze(1)
    falling = (right.unsqueeze(1) - fft_freqs.unsqueeze(0)) / (right - center).clamp_min(
        1e-8
    ).unsqueeze(1)
    bank = torch.clamp(torch.minimum(rising, falling), min=0.0)
    if norm == "slaney":

        bank = bank * (2.0 / (right - left).clamp_min(1e-8)).unsqueeze(1)
    return bank


def _as_stats_buffer(value: float | Sequence[float], n_mels: int, name: str) -> torch.Tensor:

    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(torch.float32).flatten()
    elif isinstance(value, (int, float)):
        return torch.tensor(float(value))
    else:
        tensor = torch.tensor([float(item) for item in value])
    if tensor.numel() == 1:
        return tensor.reshape(())
    if tensor.numel() != n_mels:
        raise ValueError(
            f"features.{name} length must be 1 or n_mels={n_mels}; "
            f"received {tensor.numel()}"
        )
    return tensor


class LogMelFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int = 24_000,
        n_fft: int = 1024,
        hop_length: int = 240,
        win_length: int = 960,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float = 12_000.0,
        log_floor: float = 1e-9,
        mean: float | Sequence[float] = 0.0,
        std: float | Sequence[float] = 1.0,
        mel_filter_norm: str = "slaney",
        mel_scale: str = "htk",
        log_mode: str = "additive",
        log_knee: float = 1e-4,
        top_db: float | None = None,
        top_db_scope: str = "sample",
        feature_centering: str = "none",
    ) -> None:
        super().__init__()
        if log_mode not in LOG_MODES:
            raise ValueError(f"log_mode must be one of {LOG_MODES}; received {log_mode!r}")
        if mel_scale not in MEL_SCALES:
            raise ValueError(f"mel_scale must be one of {MEL_SCALES}; received {mel_scale!r}")
        if feature_centering not in FEATURE_CENTERING_MODES:
            raise ValueError(
                f"feature_centering must be one of {FEATURE_CENTERING_MODES}; "
                f"received {feature_centering!r}"
            )
        if top_db_scope not in TOP_DB_SCOPES:
            raise ValueError(
                f"top_db_scope must be one of {TOP_DB_SCOPES}; received {top_db_scope!r}"
            )
        if log_mode == "log1p" and log_knee <= 0:
            raise ValueError("log1p mode requires log_knee > 0")
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.log_floor = log_floor
        self.mel_filter_norm = str(mel_filter_norm)
        self.mel_scale = str(mel_scale)
        self.log_mode = str(log_mode)
        self.log_knee = float(log_knee)
        self.top_db = float(top_db) if top_db is not None else None
        self.top_db_scope = str(top_db_scope)
        self.feature_centering = str(feature_centering)
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

        self.register_buffer(
            "mel_filter",
            _mel_filter_cpu(
                sample_rate,
                n_fft,
                n_mels,
                f_min,
                f_max,
                self.mel_filter_norm,
                self.mel_scale,
            ).clone(),
            persistent=False,
        )
        self.register_buffer("feature_mean", _as_stats_buffer(mean, n_mels, "mean"))


        std_buffer = _as_stats_buffer(std, n_mels, "std")
        self.register_buffer(
            "feature_std", torch.where(std_buffer < 1e-3, torch.ones_like(std_buffer), std_buffer)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:

        for name in ("feature_mean", "feature_std"):
            key = prefix + name
            if key not in state_dict:
                continue
            incoming = state_dict[key]
            current = getattr(self, name)
            if not hasattr(incoming, "shape") or incoming.shape == current.shape:
                continue
            error_msgs.append(
                f"{key}: checkpoint shape {tuple(incoming.shape)} differs from model shape "
                f"{tuple(current.shape)}. Scalar and per-channel normalization statistics "
                "must use the same layout as the checkpoint; otherwise broadcasting could "
                "select channel zero. Match features.mean and features.std to the checkpoint."
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:

        waveform = F.pad(waveform, (self.n_fft - self.hop_length, 0))
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform),
            center=False,
            return_complex=True,
        )
        return spectrum.abs().square()

    def _power_mel_fp32(self, waveform: torch.Tensor) -> torch.Tensor:
        power = self._spectrogram(waveform)
        return torch.einsum(
            "mf,bft->btm", self.mel_filter.to(power.dtype), power
        ).clamp_min(0.0)

    def _forward_fp32(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self._power_mel_fp32(waveform)


        if self.log_mode == "additive":
            log_mel = torch.log(mel + self.log_floor)
        elif self.log_mode == "clamp":
            log_mel = torch.log(mel.clamp_min(self.log_floor))
        elif self.log_mode == "db":
            log_mel = 10.0 * torch.log10(mel.clamp_min(self.log_floor))
            if self.top_db is not None:
                peak_dims = (1, 2) if self.top_db_scope == "sample" else (2,)
                peak = log_mel.amax(dim=peak_dims, keepdim=True)
                log_mel = torch.maximum(log_mel, peak - self.top_db)
        else:

            log_mel = torch.log1p(mel / self.log_knee)
        if self.feature_centering == "frame":


            log_mel = log_mel - log_mel.mean(dim=-1, keepdim=True)
        return (log_mel - self.feature_mean) / self.feature_std.clamp_min(1e-6)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        self._validate_waveform(waveform)


        with torch.autocast(device_type=waveform.device.type, enabled=False):
            return self._forward_fp32(waveform.float())

    def power_mel(self, waveform: torch.Tensor) -> torch.Tensor:

        self._validate_waveform(waveform)
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            return self._power_mel_fp32(waveform.float())

    def _validate_waveform(self, waveform: torch.Tensor) -> None:
        if waveform.ndim != 2:
            raise ValueError(
                "LogMelFrontend requires two-dimensional (batch, samples) input; "
                f"received shape {tuple(waveform.shape)}"
            )
        if waveform.shape[1] < self.hop_length:
            raise ValueError(
                f"waveform requires at least hop_length={self.hop_length} samples "
                f"to produce one frame; received {waveform.shape[1]}"
            )

    def lengths(self, waveform_lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(waveform_lengths, self.hop_length, rounding_mode="floor")


@lru_cache(maxsize=8)
def _chroma_filterbank_cpu(
    sample_rate: int,
    n_fft: int,
    n_chroma: int = 12,
    center_octave: float = 5.0,
    octave_width: float = 2.0,
) -> torch.Tensor:

    n_freq = n_fft // 2 + 1

    frequencies = torch.linspace(
        0.0, float(sample_rate), n_fft, dtype=torch.float64
    )[1:]

    octaves = torch.log2(frequencies / 27.5)
    bins = n_chroma * octaves

    bins = torch.cat([bins[:1] - 1.5 * n_chroma, bins])
    widths = torch.cat(
        [torch.clamp(bins[1:] - bins[:-1], min=1.0), torch.ones(1, dtype=torch.float64)]
    )
    distance = bins.unsqueeze(0) - torch.arange(n_chroma, dtype=torch.float64).unsqueeze(1)
    half = round(n_chroma / 2)

    distance = torch.remainder(distance + half + 10 * n_chroma, n_chroma) - half
    weights = torch.exp(-0.5 * (2.0 * distance / widths.unsqueeze(0)) ** 2)
    weights = weights / weights.norm(dim=0, keepdim=True).clamp_min(1e-12)
    if octave_width is not None and octave_width > 0:
        dominance = torch.exp(
            -0.5 * ((bins / n_chroma - center_octave) / octave_width) ** 2
        )
        weights = weights * dominance.unsqueeze(0)

    weights = torch.roll(weights, -3 * (n_chroma // 12), dims=0)
    return weights[:, :n_freq].to(torch.float32).contiguous()


def chroma_from_waveform(
    waveform: torch.Tensor,
    sample_rate: int = 24_000,
    n_fft: int | None = None,
    hop_length: int = 240,
    win_length: int | None = None,
    mode: str = "soft",
    chunk_frames: int = 2048,
) -> torch.Tensor:

    if mode not in CHROMA_MODES:
        raise ValueError(f"chroma mode must be one of {CHROMA_MODES}; received {mode!r}")
    if n_fft is None:
        n_fft = 4096 if mode == "soft" else 1024
    if win_length is None:
        win_length = n_fft
    if win_length != n_fft:
        raise ValueError(
            f"chroma win_length ({win_length}) must equal n_fft ({n_fft}); "
            "torch.stft center padding for shorter windows would move each frame's "
            "right boundary beyond its hop position"
        )
    window = torch.hann_window(win_length, device=waveform.device, dtype=waveform.dtype)

    padded = F.pad(waveform, (n_fft - hop_length, 0))
    if mode == "soft":


        filterbank = _chroma_filterbank_cpu(sample_rate, n_fft).to(
            device=waveform.device, dtype=waveform.dtype
        )
        total_frames = (padded.shape[1] - n_fft) // hop_length + 1
        chunks = []
        for start in range(0, total_frames, chunk_frames):
            count = min(chunk_frames, total_frames - start)
            begin = start * hop_length
            end = begin + (count - 1) * hop_length + n_fft
            magnitude = torch.stft(
                padded[:, begin:end],
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=False,
                return_complex=True,
            ).abs()
            chunks.append(torch.einsum("cf,bft->btc", filterbank, magnitude))
        chroma = torch.cat(chunks, dim=1)
    else:
        magnitude = torch.stft(
            padded,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=False,
            return_complex=True,
        ).abs()
        frequencies = torch.fft.rfftfreq(n_fft, 1.0 / sample_rate).to(waveform.device)
        valid = frequencies >= 27.5
        midi = 69.0 + 12.0 * torch.log2(frequencies[valid] / 440.0)
        pitch_class = torch.remainder(torch.round(midi).long(), 12)
        chroma = waveform.new_zeros(waveform.shape[0], magnitude.shape[-1], 12)
        for pc in range(12):
            selected = valid.clone()
            selected[valid] = pitch_class == pc
            chroma[:, :, pc] = magnitude[:, selected, :].sum(dim=1)
    return chroma / chroma.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def lengths_to_mask(lengths: torch.Tensor, maximum: int | None = None) -> torch.Tensor:
    maximum = int(maximum if maximum is not None else lengths.max().item())
    positions = torch.arange(maximum, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)
