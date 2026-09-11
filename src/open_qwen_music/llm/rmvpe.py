"""RMVPE vocal-pitch extraction for the Melody Tokenizer.

The released pipeline uses a 128-Mel RMVPE checkpoint at its native 100 Hz
grid and reduces adjacent frames in the cent domain to the required 50 Hz
pitch curve. The curve is median-pooled by eight before melody quantization.

The neural-network classes are adapted from ``Dream-High/RMVPE`` (Apache-2.0):
https://github.com/Dream-High/RMVPE
The adaptation makes the Mel width configurable, uses modern ``torch.stft``,
loads state dicts safely with ``weights_only=True``, and adds shape/provenance
validation.  No model weights are bundled in this repository.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

RMVPE_SAMPLE_RATE = 16_000
RMVPE_PITCH_BINS = 360
RMVPE_CENTS_PER_BIN = 20.0
RMVPE_CENTS_OFFSET = 1997.3794084376191
RMVPE_FMIN_HZ = 30.0
RMVPE_FMAX_HZ = 8_000.0
RMVPE_MODEL_TIME_MULTIPLE = 32

PUBLIC_RVC_REPO_ID = "mirbox/RMVPE"
PUBLIC_RVC_FILENAME = "model.pt"
PUBLIC_RVC_SHA256 = "19dc1809cf4cdb0a18db93441816bc327e14e5644b72eeaae5220560c6736fe2"

RMVPEReductionPolicy = Literal["cent_mean", "decimate_second"]


@dataclass(frozen=True)
class RMVPEProfile:
    """All preprocessing and decoding choices that affect the F0 curve."""

    name: str
    sample_rate: int
    native_frame_rate: float
    frame_rate: float
    hop_length: int
    n_mels: int
    n_fft: int
    win_length: int
    mel_fmin: float
    mel_fmax: float
    voicing_threshold: float
    center: bool
    pad_mode: str
    implementation: str

    @classmethod
    def released(cls) -> RMVPEProfile:
        return cls(
            name="open-qwen-music-rmvpe-v1",
            sample_rate=RMVPE_SAMPLE_RATE,
            native_frame_rate=100.0,
            frame_rate=50.0,
            hop_length=160,
            n_mels=128,
            n_fft=1024,
            win_length=1024,
            mel_fmin=30.0,
            mel_fmax=8000.0,
            voicing_threshold=0.03,
            center=True,
            pad_mode="reflect",
            implementation="128-Mel RMVPE at 100 Hz with pairwise cent-mean reduction",
        )

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.hop_length <= 0:
            raise ValueError("RMVPE sample_rate and hop_length must be positive")
        actual_rate = self.sample_rate / self.hop_length
        if not math.isclose(actual_rate, self.native_frame_rate):
            raise ValueError(
                f"profile native frame rate mismatch: {self.sample_rate}/{self.hop_length} "
                f"= {actual_rate}, declared {self.native_frame_rate}"
            )
        if self.frame_rate <= 0 or self.native_frame_rate < self.frame_rate:
            raise ValueError("profile frame rates must satisfy native >= output > 0")
        if self.n_mels <= 0 or self.n_mels % 32:
            raise ValueError("RMVPE n_mels must be a positive multiple of 32")
        if not 0.0 <= self.voicing_threshold <= 1.0:
            raise ValueError("voicing_threshold must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PitchTrack:
    """One RMVPE pitch curve and enough provenance to reproduce it."""

    f0_hz: np.ndarray
    confidence: np.ndarray
    frame_rate: float
    source_num_samples: int
    source_sample_rate: int
    rmvpe_profile: dict[str, Any]
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        f0 = np.asarray(self.f0_hz)
        confidence = np.asarray(self.confidence)
        if f0.ndim != 1 or confidence.ndim != 1:
            raise ValueError("PitchTrack f0_hz and confidence must be one-dimensional")
        if f0.shape != confidence.shape:
            raise ValueError("PitchTrack f0_hz and confidence lengths differ")
        if f0.size == 0:
            raise ValueError("PitchTrack cannot be empty")
        if not np.isfinite(f0).all() or (f0 < 0).any():
            raise ValueError("PitchTrack f0_hz must be finite and non-negative")
        if not np.isfinite(confidence).all():
            raise ValueError("PitchTrack confidence must be finite")
        if ((confidence < 0) | (confidence > 1)).any():
            raise ValueError("PitchTrack confidence must be in [0, 1]")

    @property
    def num_frames(self) -> int:
        return int(self.f0_hz.size)

    @property
    def voiced_mask(self) -> np.ndarray:
        return self.f0_hz > 0

    def metadata(self) -> dict[str, Any]:
        return {
            "frame_rate": self.frame_rate,
            "num_frames": self.num_frames,
            "source_num_samples": self.source_num_samples,
            "source_sample_rate": self.source_sample_rate,
            "rmvpe_profile": self.rmvpe_profile,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


class _BiGRU(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, num_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.gru(values)[0]


class _ConvBlockRes(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(values) if self.shortcut is not None else values
        return self.conv(values) + residual


class _ResEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] | None,
        n_blocks: int,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        blocks = [_ConvBlockRes(in_channels, out_channels, momentum)]
        blocks.extend(
            _ConvBlockRes(out_channels, out_channels, momentum) for _ in range(n_blocks - 1)
        )
        self.conv = nn.ModuleList(blocks)
        self.pool = nn.AvgPool2d(kernel_size) if kernel_size is not None else None

    def forward(self, values: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        for block in self.conv:
            values = block(values)
        if self.pool is None:
            return values
        return values, self.pool(values)


class _ResDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int],
        n_blocks: int,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        output_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                3,
                stride=stride,
                padding=1,
                output_padding=output_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        blocks = [_ConvBlockRes(out_channels * 2, out_channels, momentum)]
        blocks.extend(
            _ConvBlockRes(out_channels, out_channels, momentum) for _ in range(n_blocks - 1)
        )
        self.conv2 = nn.ModuleList(blocks)

    def forward(self, values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        values = torch.cat((self.conv1(values), skip), dim=1)
        for block in self.conv2:
            values = block(values)
        return values


class _Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_size: int,
        n_encoders: int,
        kernel_size: tuple[int, int],
        n_blocks: int,
        out_channels: int = 16,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        layers: list[nn.Module] = []
        latent_channels: list[int] = []
        for _ in range(n_encoders):
            layers.append(
                _ResEncoderBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    n_blocks,
                    momentum,
                )
            )
            latent_channels.append(out_channels)
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.layers = nn.ModuleList(layers)
        self.latent_channels = latent_channels
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips: list[torch.Tensor] = []
        values = self.bn(values)
        for layer in self.layers:
            skip, values = layer(values)
            skips.append(skip)
        return values, skips


class _Intermediate(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int,
        n_blocks: int,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            _ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)
        ]
        layers.extend(
            _ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
            for _ in range(n_layers - 1)
        )
        self.layers = nn.ModuleList(layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            values = layer(values)
        return values


class _Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_decoders: int,
        stride: tuple[int, int],
        n_blocks: int,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            layers.append(_ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels
        self.layers = nn.ModuleList(layers)

    def forward(self, values: torch.Tensor, skips: Sequence[torch.Tensor]) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            values = layer(values, skips[-1 - index])
        return values


class _TimbreFilter(nn.Module):
    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_ConvBlockRes(value, value) for value in channels)

    def forward(self, values: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [layer(value) for layer, value in zip(self.layers, values, strict=True)]


class _DeepUNet(nn.Module):
    def __init__(
        self,
        n_mels: int,
        kernel_size: tuple[int, int] = (2, 2),
        n_blocks: int = 4,
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
        use_timbre_filter: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = _Encoder(
            in_channels,
            n_mels,
            en_de_layers,
            kernel_size,
            n_blocks,
            en_out_channels,
        )
        self.intermediate = _Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.tf = _TimbreFilter(self.encoder.latent_channels) if use_timbre_filter else None
        self.decoder = _Decoder(
            self.encoder.out_channel,
            en_de_layers,
            kernel_size,
            n_blocks,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values, skips = self.encoder(values)
        values = self.intermediate(values)
        if self.tf is not None:
            skips = self.tf(skips)
        return self.decoder(values, skips)


class RMVPENetwork(nn.Module):
    """Deep U-Net + BiGRU RMVPE salience network.

    The non-default architecture arguments exist for small CPU unit tests.
    Released checkpoints use ``n_blocks=4``, ``n_gru=1``, five encoder/decoder
    levels, four intermediate levels, and 16 initial channels.
    """

    def __init__(
        self,
        n_mels: int,
        *,
        n_blocks: int = 4,
        n_gru: int = 1,
        kernel_size: tuple[int, int] = (2, 2),
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
        use_timbre_filter: bool = True,
    ) -> None:
        super().__init__()
        self.n_mels = int(n_mels)
        self.unet = _DeepUNet(
            n_mels,
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
            use_timbre_filter,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, 3, padding=1)
        if n_gru:
            self.fc = nn.Sequential(
                _BiGRU(3 * n_mels, 256, n_gru),
                nn.Linear(512, RMVPE_PITCH_BINS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * n_mels, RMVPE_PITCH_BINS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim != 3 or mel.shape[1] != self.n_mels:
            raise ValueError(f"RMVPE mel must be [B,{self.n_mels},T], got {tuple(mel.shape)}")
        # U-Net pools both time and frequency five times; callers pad T to a
        # multiple of 32 and supported Mel widths are themselves multiples of 32.
        values = mel.transpose(-1, -2).unsqueeze(1)
        values = self.cnn(self.unet(values)).transpose(1, 2).flatten(-2)
        return self.fc(values)


class LogMelSpectrogram(nn.Module):
    """Log-Mel frontend matching the public RMVPE implementations."""

    def __init__(self, profile: RMVPEProfile) -> None:
        super().__init__()
        self.profile = profile
        try:
            from torchaudio.functional import melscale_fbanks
        except (ImportError, OSError) as error:
            raise ImportError(
                "RMVPE audio preprocessing needs torchaudio; install "
                "`open-qwen-music[train]` with a torchaudio build matching torch"
            ) from error
        basis = melscale_fbanks(
            n_freqs=profile.n_fft // 2 + 1,
            f_min=profile.mel_fmin,
            f_max=profile.mel_fmax,
            n_mels=profile.n_mels,
            sample_rate=profile.sample_rate,
            norm="slaney",
            mel_scale="htk",
        ).transpose(0, 1)
        self.register_buffer("mel_basis", basis.contiguous())
        self.register_buffer("window", torch.hann_window(profile.win_length))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError(f"waveform must be [B,T], got {tuple(waveform.shape)}")
        spectrum = torch.stft(
            waveform,
            n_fft=self.profile.n_fft,
            hop_length=self.profile.hop_length,
            win_length=self.profile.win_length,
            window=self.window.to(device=waveform.device, dtype=waveform.dtype),
            center=self.profile.center,
            pad_mode=self.profile.pad_mode,
            return_complex=True,
        )
        magnitude = spectrum.abs()
        basis = self.mel_basis.to(device=waveform.device, dtype=waveform.dtype)
        mel = torch.einsum("mf,bft->bmt", basis, magnitude)
        return torch.log(mel.clamp_min(1e-5))


def decode_rmvpe_salience(
    salience: np.ndarray | torch.Tensor,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode 360-bin salience using RMVPE's local ±4-bin weighted mean."""

    values = (
        salience.detach().float().cpu().numpy()
        if isinstance(salience, torch.Tensor)
        else np.asarray(salience, dtype=np.float32)
    )
    if values.ndim not in (2, 3) or values.shape[-1] != RMVPE_PITCH_BINS:
        raise ValueError(
            f"salience must be [T,{RMVPE_PITCH_BINS}] or [B,T,{RMVPE_PITCH_BINS}], "
            f"got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("salience contains NaN or Inf")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    original_shape = values.shape[:-1]
    flat = values.reshape(-1, RMVPE_PITCH_BINS)
    confidence = flat.max(axis=1)
    centers = flat.argmax(axis=1)
    cents_grid = (
        np.arange(RMVPE_PITCH_BINS, dtype=np.float64) * RMVPE_CENTS_PER_BIN + RMVPE_CENTS_OFFSET
    )
    padded_values = np.pad(flat, ((0, 0), (4, 4)))
    padded_cents = np.pad(cents_grid, (4, 4), mode="edge")
    centers = centers + 4
    offsets = np.arange(-4, 5)
    indices = centers[:, None] + offsets[None, :]
    local_values = np.take_along_axis(padded_values, indices, axis=1)
    local_cents = padded_cents[indices]
    weight_sum = local_values.sum(axis=1)
    cents = np.divide(
        (local_values * local_cents).sum(axis=1),
        weight_sum,
        out=np.zeros_like(weight_sum, dtype=np.float64),
        where=weight_sum > 0,
    )
    voiced = confidence >= threshold
    f0 = np.zeros_like(cents, dtype=np.float64)
    f0[voiced] = 10.0 * np.exp2(cents[voiced] / 1200.0)
    return (
        f0.reshape(original_shape).astype(np.float32),
        confidence.reshape(original_shape).astype(np.float32),
    )


def reduce_pitch_track(
    f0_hz: np.ndarray,
    confidence: np.ndarray,
    *,
    frames: int,
    native_frame_rate: float,
    frame_rate: float,
    centered: bool,
    policy: RMVPEReductionPolicy = "cent_mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Align an RMVPE-native curve to the exact Qwen 50 Hz time grid.

    Public RVC weights are calibrated on 100 Hz features.  Running those
    weights directly with a 20 ms hop changes the physical duration seen by
    every temporal convolution and GRU step.  Keep the native grid and reduce
    integer groups instead:

    - average voiced pitch in the cent/log-frequency domain;
    - ignore unvoiced members unless the whole group is unvoiced;
    - never interpolate across a voiced/unvoiced boundary;
    - drop the leading boundary frame added by centered STFT before grouping;
    - derive the final length from audio duration, not model padding.

    The leading-frame rule is also empirically checked on MIR-1K: native
    100->50 Hz grouping with the leading boundary removed aligns at lag 0 and
    reaches higher RPA than either direct 50 Hz inference or dropping the tail.
    """

    pitch = np.asarray(f0_hz, dtype=np.float64).reshape(-1)
    scores = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if pitch.shape != scores.shape:
        raise ValueError("f0_hz and confidence lengths differ")
    if frames <= 0:
        raise ValueError("frames must be positive")
    ratio = native_frame_rate / frame_rate
    factor = int(round(ratio))
    if factor < 1 or not math.isclose(ratio, factor):
        raise ValueError(f"only integer RMVPE frame-rate reduction is supported, got {ratio}")

    required = frames * factor
    start = 1 if centered and pitch.size >= required + 1 else 0
    pitch = pitch[start : start + required]
    scores = scores[start : start + required]
    if pitch.size < required:
        missing = required - pitch.size
        pitch = np.pad(pitch, (0, missing))
        scores = np.pad(scores, (0, missing))

    if policy not in {"cent_mean", "decimate_second"}:
        raise ValueError(f"unknown RMVPE reduction policy: {policy!r}")
    pitch_blocks = pitch.reshape(frames, factor)
    score_blocks = scores.reshape(frames, factor)
    voiced = np.isfinite(pitch_blocks) & (pitch_blocks > 0)
    if policy == "decimate_second":
        index = factor - 1
        output_pitch = np.where(voiced[:, index], pitch_blocks[:, index], 0.0)
        output_confidence = score_blocks[:, index]
        return (
            output_pitch.astype(np.float32),
            output_confidence.astype(np.float32),
        )
    counts = voiced.sum(axis=1)
    output_pitch = np.zeros(frames, dtype=np.float64)
    output_confidence = score_blocks.max(axis=1)
    has_voice = counts > 0
    log_pitch = np.where(voiced, np.log2(np.where(voiced, pitch_blocks, 1.0)), 0.0)
    output_pitch[has_voice] = np.exp2(log_pitch[has_voice].sum(axis=1) / counts[has_voice])
    voiced_scores = np.where(voiced, score_blocks, 0.0)
    output_confidence[has_voice] = voiced_scores[has_voice].sum(axis=1) / counts[has_voice]
    return output_pitch.astype(np.float32), output_confidence.astype(np.float32)


def _checkpoint_state(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError(f"RMVPE checkpoint must be a dict, got {type(payload).__name__}")
    candidate: Any = payload.get("model", payload)
    if not isinstance(candidate, dict):
        raise ValueError("RMVPE checkpoint `model` entry is not a state dict")
    state = dict(candidate)
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("RMVPE checkpoint state dict is empty or contains non-tensors")
    return state


def infer_checkpoint_mel_bins(state: dict[str, torch.Tensor]) -> int:
    """Infer the checkpoint profile from the BiGRU input width."""

    key = "fc.0.gru.weight_ih_l0"
    if key not in state:
        raise ValueError(f"RMVPE checkpoint is missing {key!r}")
    weight = state[key]
    if weight.ndim != 2 or weight.shape[1] % 3:
        raise ValueError(f"unexpected {key} shape: {tuple(weight.shape)}")
    return int(weight.shape[1] // 3)


def sha256_file(path: str | Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(block_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def download_public_rvc_checkpoint(cache_dir: str | Path) -> Path:
    """Download and verify the public compatibility checkpoint.

    Download is explicit rather than an implicit constructor side effect so
    production jobs never hang because a compute node unexpectedly needs the
    public internet.
    """

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "checkpoint download needs huggingface-hub; install `open-qwen-music[render]`"
        ) from error
    path = Path(
        hf_hub_download(
            repo_id=PUBLIC_RVC_REPO_ID,
            filename=PUBLIC_RVC_FILENAME,
            local_dir=Path(cache_dir),
        )
    )
    actual = sha256_file(path)
    if actual != PUBLIC_RVC_SHA256:
        raise RuntimeError(
            f"downloaded RMVPE checkpoint hash mismatch: {actual} != {PUBLIC_RVC_SHA256}"
        )
    return path


def _float_mono_waveform(waveform: np.ndarray | torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(waveform)
    if not values.dtype.is_floating_point:
        info = torch.iinfo(values.dtype)
        scale = float(max(abs(info.min), info.max))
        values = values.float() / scale
    else:
        values = values.float()
    if values.ndim == 2:
        # Public API is [channels, time].  Accept [time, channels] only when
        # the channel dimension is unambiguous.
        if values.shape[0] > 8 and values.shape[1] <= 8:
            values = values.transpose(0, 1)
        if values.shape[0] > 8:
            raise ValueError("2-D waveform must be [channels,time] with at most 8 channels")
        values = values.mean(dim=0)
    elif values.ndim == 1:
        pass
    else:
        raise ValueError(f"waveform must be [T] or [C,T], got {tuple(values.shape)}")
    if values.numel() == 0:
        raise ValueError("waveform is empty")
    if not torch.isfinite(values).all():
        raise ValueError("waveform contains NaN or Inf")
    peak = float(values.abs().max())
    if peak > 1.01:
        raise ValueError(
            f"floating waveform peak is {peak:.3f}; expected normalized audio in [-1,1]"
        )
    return values.clamp(-1.0, 1.0).contiguous()


def _resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {source_rate}")
    if source_rate == target_rate:
        return waveform
    try:
        from torchaudio.functional import resample
    except (ImportError, OSError) as error:
        raise ImportError(
            "resampling needs torchaudio; install `open-qwen-music[train]` "
            "with a torchaudio build matching torch"
        ) from error
    return resample(
        waveform,
        orig_freq=source_rate,
        new_freq=target_rate,
        lowpass_filter_width=64,
    )


class RMVPEPitchExtractor:
    """Checkpoint-backed audio-to-50-Hz vocal pitch extractor."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
        voicing_threshold: float | None = None,
        reduction_policy: RMVPEReductionPolicy = "cent_mean",
        chunk_frames: int | None = None,
        context_frames: int = 128,
        verify_checkpoint_hash: str | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"RMVPE checkpoint not found: {self.checkpoint}")
        self.profile = RMVPEProfile.released()
        self.device = torch.device(device)
        if self.device.type == "cpu":
            self.dtype = torch.float32
        else:
            # GPU performance/dtype validation is intentionally left to a later
            # GPU test; fp32 is the correctness-first default on every device.
            self.dtype = torch.float32
        self.voicing_threshold = (
            self.profile.voicing_threshold
            if voicing_threshold is None
            else float(voicing_threshold)
        )
        if not 0.0 <= self.voicing_threshold <= 1.0:
            raise ValueError("voicing_threshold must be in [0,1]")
        if reduction_policy not in {"cent_mean", "decimate_second"}:
            raise ValueError(f"unknown RMVPE reduction policy: {reduction_policy!r}")
        self.reduction_policy = reduction_policy
        if chunk_frames is not None:
            if chunk_frames <= 0 or chunk_frames % RMVPE_MODEL_TIME_MULTIPLE:
                raise ValueError("chunk_frames must be a positive multiple of 32")
            if context_frames < 0 or context_frames >= chunk_frames:
                raise ValueError("context_frames must be in [0, chunk_frames)")
        self.chunk_frames = chunk_frames
        self.context_frames = int(context_frames)

        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        if verify_checkpoint_hash is not None and self.checkpoint_sha256 != verify_checkpoint_hash:
            raise ValueError(
                "RMVPE checkpoint SHA256 mismatch: "
                f"{self.checkpoint_sha256} != {verify_checkpoint_hash}"
            )
        payload = torch.load(
            self.checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        state = _checkpoint_state(payload)
        state = {key: value for key, value in state.items() if not key.startswith("unet.tf.")}
        checkpoint_mels = infer_checkpoint_mel_bins(state)
        if checkpoint_mels != self.profile.n_mels:
            raise ValueError(
                f"RMVPE checkpoint has {checkpoint_mels} Mel bins; "
                f"the released pipeline requires {self.profile.n_mels}"
            )
        self.model = RMVPENetwork(
            self.profile.n_mels,
            use_timbre_filter=False,
        )
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(device=self.device, dtype=self.dtype)
        self.frontend = (
            LogMelSpectrogram(self.profile).eval().to(device=self.device, dtype=self.dtype)
        )

    @property
    def revision(self) -> str:
        return f"sha256:{self.checkpoint_sha256}"

    def provenance(self) -> dict[str, Any]:
        return {
            "implementation": "open_qwen_music.llm.rmvpe",
            "profile": self.profile.to_dict(),
            "voicing_threshold": self.voicing_threshold,
            "reduction_policy": self.reduction_policy,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "chunk_frames": self.chunk_frames,
            "context_frames": self.context_frames if self.chunk_frames else None,
            "chunking_exact": self.chunk_frames is None,
        }

    def _forward_mel(self, mel: torch.Tensor) -> torch.Tensor:
        original_frames = mel.shape[-1]
        pad = (-original_frames) % RMVPE_MODEL_TIME_MULTIPLE
        if pad:
            mel = F.pad(mel, (0, pad), mode="constant")
        with torch.inference_mode():
            salience = self.model(mel.to(device=self.device, dtype=self.dtype))
        return salience[:, :original_frames].float().cpu()

    def _salience(self, mel: torch.Tensor) -> torch.Tensor:
        frames = mel.shape[-1]
        if self.chunk_frames is None or frames <= self.chunk_frames:
            return self._forward_mel(mel)

        # The BiGRU has theoretically unbounded context, so chunked inference is
        # not bit-identical to a full-track pass.  Keeping explicit left/right
        # context makes boundary error local; provenance marks this as inexact.
        pieces: list[torch.Tensor] = []
        core = self.chunk_frames
        context = self.context_frames
        for core_start in range(0, frames, core):
            core_end = min(core_start + core, frames)
            start = max(0, core_start - context)
            end = min(frames, core_end + context)
            output = self._forward_mel(mel[..., start:end])
            pieces.append(output[:, core_start - start : core_end - start])
        result = torch.cat(pieces, dim=1)
        if result.shape[1] != frames:
            raise RuntimeError(
                f"chunked RMVPE produced {result.shape[1]} frames, expected {frames}"
            )
        return result

    def extract(
        self,
        waveform: np.ndarray | torch.Tensor,
        sample_rate: int,
    ) -> PitchTrack:
        source = _float_mono_waveform(waveform)
        source_num_samples = int(source.numel())
        resampled = _resample(source, int(sample_rate), self.profile.sample_rate)
        resampled_num_samples = int(resampled.numel())
        if self.profile.center:
            native_frames = resampled_num_samples // self.profile.hop_length + 1
        else:
            native_frames = max(
                1,
                (resampled_num_samples - self.profile.n_fft) // self.profile.hop_length + 1,
            )
        output_frames = max(
            1,
            int(round(resampled_num_samples / self.profile.sample_rate * self.profile.frame_rate)),
        )
        # reflect padding requires more samples than half the FFT.  Real songs
        # trivially satisfy this; constant-pad tiny unit-test inputs explicitly.
        minimum = self.profile.n_fft // 2 + 1
        if resampled.numel() < minimum:
            resampled = F.pad(resampled, (0, minimum - resampled.numel()))
        with torch.inference_mode():
            mel = self.frontend(resampled.to(device=self.device, dtype=self.dtype).unsqueeze(0))
        salience = self._salience(mel)[0, :native_frames]
        native_f0, native_confidence = decode_rmvpe_salience(
            salience,
            threshold=self.voicing_threshold,
        )
        f0, confidence = reduce_pitch_track(
            native_f0,
            native_confidence,
            frames=output_frames,
            native_frame_rate=self.profile.native_frame_rate,
            frame_rate=self.profile.frame_rate,
            centered=self.profile.center,
            policy=self.reduction_policy,
        )
        return PitchTrack(
            f0_hz=f0,
            confidence=confidence,
            frame_rate=self.profile.frame_rate,
            source_num_samples=source_num_samples,
            source_sample_rate=int(sample_rate),
            rmvpe_profile=self.profile.to_dict(),
            checkpoint_sha256=self.checkpoint_sha256,
        )

    def extract_file(
        self,
        path: str | Path,
        *,
        start_sec: float = 0.0,
        duration_sec: float | None = None,
    ) -> PitchTrack:
        if start_sec < 0:
            raise ValueError("start_sec cannot be negative")
        if duration_sec is not None and duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        try:
            import soundfile as sf
        except ImportError as error:
            raise ImportError(
                "audio-file input needs soundfile; install `open-qwen-music`"
            ) from error
        with sf.SoundFile(str(path)) as handle:
            start = min(round(start_sec * handle.samplerate), len(handle))
            handle.seek(start)
            frames = -1 if duration_sec is None else round(duration_sec * handle.samplerate)
            audio = handle.read(frames=frames, dtype="float32", always_2d=True)
            sample_rate = int(handle.samplerate)
        if audio.size == 0:
            raise ValueError(f"audio slice is empty: {path}")
        return self.extract(audio.T, sample_rate)

    def extract_batch(
        self,
        waveforms: Sequence[np.ndarray | torch.Tensor],
        sample_rates: Sequence[int],
    ) -> list[PitchTrack]:
        if len(waveforms) != len(sample_rates):
            raise ValueError("waveforms and sample_rates lengths differ")
        # Correctness-first variable-length API.  A padded batched fast path can
        # be added with GPU measurements; looping avoids changing each song's
        # frame count or injecting padded frames into the global MIDI median.
        return [
            self.extract(waveform, sample_rate)
            for waveform, sample_rate in zip(waveforms, sample_rates, strict=True)
        ]
