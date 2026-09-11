
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor


SAMPLE_RATE = 48_000
AUDIO_CHANNELS = 2
STEREO_CHANNELS = AUDIO_CHANNELS
STFT_N_FFT = 960
STFT_WIN_LENGTH = 960
STFT_HOP_LENGTH = 480
STFT_LEFT_PADDING = 240
STFT_BINS = 480
STFT_FRAME_HZ = SAMPLE_RATE // STFT_HOP_LENGTH
LATENT_DIM = 128
LATENT_FRAME_HZ = 25
SEMANTIC_FRAME_HZ = 25
SEMANTIC_CODEBOOK_SIZE = 32_768
SPEC_FRAMES_PER_LATENT_FRAME = STFT_FRAME_HZ // LATENT_FRAME_HZ
SAMPLES_PER_LATENT_FRAME = SAMPLE_RATE // LATENT_FRAME_HZ
SEGMENT_SECONDS = 1.28
SEGMENT_SAMPLES = 61_440
SEGMENT_STFT_FRAMES = 128
SEGMENT_LATENT_FRAMES = 32
CONTRACT_VERSION = "oqm.render.contract.v1"
STFT_CONTRACT_VERSION = "open-qwen-music-stft-v1"
LATENT_LAYOUT_FORMAT_VERSION = "oqm.render-latent-layout.v1"


class RenderContractError(ValueError):
    pass


def validate_latent_layout(value: Mapping[str, Any]) -> dict[str, Any]:

    if not isinstance(value, Mapping):
        raise TypeError("latent_layoutmust bemapping")
    allowed = {
        "format_version",
        "latent_dim",
        "frame_hz",
        "channel_semantics",
        "special_channels",
        "normalization",
    }
    unknown = set(value) - allowed
    if unknown:
        raise RenderContractError(
            f"latent_layoutcontains unknown fields:{sorted(unknown)}"
        )
    if value.get("format_version") != LATENT_LAYOUT_FORMAT_VERSION:
        raise RenderContractError("latent_layout format_versionis not compatible with")
    if value.get("latent_dim") != LATENT_DIM:
        raise RenderContractError("latent_layout latent_dimmust be128")
    if float(value.get("frame_hz", 0.0)) != float(LATENT_FRAME_HZ):
        raise RenderContractError("latent_layout frame_hzmust be25")
    if value.get("normalization") != "per_channel_affine":
        raise RenderContractError(
            "latent_layout normalizationmust beper_channel_affine"
        )
    special = value.get("special_channels")
    if not isinstance(special, list):
        raise TypeError("latent_layout.special_channelsmust be a list")
    indices: set[int] = set()
    normalized_special: list[dict[str, Any]] = []
    for position, item in enumerate(special):
        if not isinstance(item, Mapping) or set(item) != {"index", "role"}:
            raise RenderContractError(
                f"latent_layout.special_channels[{position}]must contain exactlyindex/role"
            )
        index = item["index"]
        role = item["role"]
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < LATENT_DIM
        ):
            raise RenderContractError(
                f"latent_layout.special_channels[{position}].indexIllegal"
            )
        if index in indices:
            raise RenderContractError("latent_layoutSpecial channelindexRepeat")
        if not isinstance(role, str) or not role.strip():
            raise RenderContractError(
                f"latent_layout.special_channels[{position}].rolemust be a non-empty string"
            )
        indices.add(index)
        normalized_special.append({"index": index, "role": role})
    expected_semantics = (
        "explicit_special_channels" if normalized_special else "unstructured_continuous"
    )
    if value.get("channel_semantics") != expected_semantics:
        raise RenderContractError(
            "latent_layout channel_semanticsandspecial_channelsinconsistent"
        )
    return {
        "format_version": LATENT_LAYOUT_FORMAT_VERSION,
        "latent_dim": LATENT_DIM,
        "frame_hz": float(LATENT_FRAME_HZ),
        "channel_semantics": expected_semantics,
        "special_channels": normalized_special,
        "normalization": "per_channel_affine",
    }


def ceil_div(value: int, divisor: int) -> int:

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"value must be int,received {type(value).__name__}")
    if not isinstance(divisor, int) or isinstance(divisor, bool) or divisor <= 0:
        raise ValueError(f"divisor must be a positive integer,received {divisor!r}")
    if value < 0:
        raise ValueError(f"value must be non-negative,received {value}")
    return (value + divisor - 1) // divisor


def samples_to_stft_frames(num_samples: int) -> int:

    return (
        0
        if num_samples == 0
        else SPEC_FRAMES_PER_LATENT_FRAME * samples_to_latent_frames(num_samples)
    )


def samples_to_latent_frames(num_samples: int) -> int:

    return ceil_div(num_samples, SAMPLES_PER_LATENT_FRAME)


def stft_frames_to_latent_frames(num_frames: int) -> int:
    return ceil_div(num_frames, SPEC_FRAMES_PER_LATENT_FRAME)


def latent_frames_to_samples(num_frames: int) -> int:
    if num_frames < 0:
        raise ValueError(f"num_frames must be non-negative,received {num_frames}")
    return num_frames * SAMPLES_PER_LATENT_FRAME


def latent_frames_to_stft_frames(num_frames: int) -> int:
    if num_frames < 0:
        raise ValueError(f"num_frames must be non-negative,received {num_frames}")
    return num_frames * SPEC_FRAMES_PER_LATENT_FRAME


def stft_analysis_length(num_samples: int) -> int:

    frames = samples_to_stft_frames(num_samples)
    if frames == 0:
        return 0
    return STFT_N_FFT + (frames - 1) * STFT_HOP_LENGTH


def stft_right_padding(num_samples: int) -> int:
    frames = samples_to_stft_frames(num_samples)
    if frames == 0:
        return 0
    return stft_analysis_length(num_samples) - STFT_LEFT_PADDING - num_samples


def padded_latent_samples(num_samples: int) -> int:
    return latent_frames_to_samples(samples_to_latent_frames(num_samples))


def _as_lengths_tensor(
    lengths: Tensor | Iterable[int],
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    if isinstance(lengths, Tensor):
        result = lengths.to(device=device)
    else:
        result = torch.as_tensor(list(lengths), device=device)
    if result.ndim != 1:
        raise RenderContractError(
            f"lengths must be a one-dimensional integer tensor,received shape={tuple(result.shape)}"
        )
    if result.dtype == torch.bool or result.is_floating_point():
        raise RenderContractError(f"lengths must be an integer dtype,received {result.dtype}")
    result = result.to(dtype=torch.long)
    if torch.any(result < 0):
        raise RenderContractError("lengths must not contain negative numbers")
    return result


def lengths_to_mask(
    lengths: Tensor | Iterable[int],
    max_length: int | None = None,
    *,
    device: torch.device | str | None = None,
) -> Tensor:

    values = _as_lengths_tensor(lengths, device=device)
    inferred = int(values.max().item()) if values.numel() else 0
    size = inferred if max_length is None else int(max_length)
    if size < inferred:
        raise RenderContractError(f"max_length={size} is less than the maximum effective length {inferred}")
    positions = torch.arange(size, device=values.device)
    return positions.unsqueeze(0) < values.unsqueeze(1)


def mask_to_lengths(mask: Tensor, *, require_right_padded: bool = True) -> Tensor:
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise RenderContractError(
            f"mask must be bool [B,T],received {mask.dtype} {tuple(mask.shape)}"
        )
    lengths = mask.sum(dim=1, dtype=torch.long)
    if require_right_padded:
        canonical = lengths_to_mask(lengths, mask.shape[1], device=mask.device)
        if not torch.equal(mask, canonical):
            raise RenderContractError("mask must be a continuous prefix to be valid,right padding")
    return lengths


def sample_lengths_to_stft_lengths(lengths: Tensor | Iterable[int]) -> Tensor:
    latent_lengths = sample_lengths_to_latent_lengths(lengths)
    return latent_lengths * SPEC_FRAMES_PER_LATENT_FRAME


def sample_lengths_to_latent_lengths(lengths: Tensor | Iterable[int]) -> Tensor:
    values = _as_lengths_tensor(lengths)
    return torch.div(
        values + SAMPLES_PER_LATENT_FRAME - 1,
        SAMPLES_PER_LATENT_FRAME,
        rounding_mode="floor",
    )


def stft_lengths_to_latent_lengths(lengths: Tensor | Iterable[int]) -> Tensor:
    values = _as_lengths_tensor(lengths)
    return torch.div(
        values + SPEC_FRAMES_PER_LATENT_FRAME - 1,
        SPEC_FRAMES_PER_LATENT_FRAME,
        rounding_mode="floor",
    )


def validate_audio(
    audio: Tensor,
    lengths: Tensor | Iterable[int] | None = None,
) -> Tensor:
    if audio.ndim != 3 or audio.shape[1] != AUDIO_CHANNELS:
        raise RenderContractError(
            f"audio must be [B,2,N],received shape={tuple(audio.shape)}"
        )
    if audio.is_complex() or not audio.is_floating_point():
        raise RenderContractError(f"audio must be a real floating point tensor,received {audio.dtype}")
    if lengths is None:
        result = torch.full(
            (audio.shape[0],),
            audio.shape[-1],
            dtype=torch.long,
            device=audio.device,
        )
    else:
        result = _as_lengths_tensor(lengths, device=audio.device)
    if result.numel() != audio.shape[0]:
        raise RenderContractError("audio_lengths batch and audio batch different")
    if torch.any(result > audio.shape[-1]):
        raise RenderContractError("audio_lengths Waveform tensor length exceeded")
    return result


def validate_spectrum(
    spectrum: Tensor,
    lengths: Tensor | Iterable[int] | None = None,
) -> Tensor:
    if (
        spectrum.ndim != 4
        or spectrum.shape[1] != AUDIO_CHANNELS
        or spectrum.shape[2] != STFT_BINS
    ):
        raise RenderContractError(
            f"spectrum must be complex [B,2,480,T],received shape={tuple(spectrum.shape)}"
        )
    if not spectrum.is_complex():
        raise RenderContractError(f"spectrum must be complex,received {spectrum.dtype}")
    if lengths is None:
        result = torch.full(
            (spectrum.shape[0],),
            spectrum.shape[-1],
            dtype=torch.long,
            device=spectrum.device,
        )
    else:
        result = _as_lengths_tensor(lengths, device=spectrum.device)
    if result.numel() != spectrum.shape[0] or torch.any(result > spectrum.shape[-1]):
        raise RenderContractError("spectrum_lengths and spectrum batch/time does not match")
    return result


def validate_latents(
    latents: Tensor,
    lengths: Tensor | Iterable[int] | None = None,
) -> Tensor:
    if latents.ndim != 3 or latents.shape[-1] != LATENT_DIM:
        raise RenderContractError(
            f"latents must be [B,T,128],received shape={tuple(latents.shape)}"
        )
    if not latents.is_floating_point():
        raise RenderContractError(f"latents must be floating point,received {latents.dtype}")
    if lengths is None:
        result = torch.full(
            (latents.shape[0],),
            latents.shape[1],
            dtype=torch.long,
            device=latents.device,
        )
    else:
        result = _as_lengths_tensor(lengths, device=latents.device)
    if result.numel() != latents.shape[0] or torch.any(result > latents.shape[1]):
        raise RenderContractError("latent_lengths and latent batch/time does not match")
    return result


def validate_semantic_ids(
    semantic_ids: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    if semantic_ids.ndim != 2 or semantic_ids.dtype != torch.long:
        raise RenderContractError(
            "semantic_ids must be int64 [B,T],"
            f"received {semantic_ids.dtype} {tuple(semantic_ids.shape)}"
        )
    effective = (
        torch.ones_like(semantic_ids, dtype=torch.bool) if mask is None else mask
    )
    if effective.shape != semantic_ids.shape or effective.dtype != torch.bool:
        raise RenderContractError("semantic_mask must be the same as shape of bool tensor")
    selected = semantic_ids[effective]
    if selected.numel() and (
        int(selected.min()) < 0 or int(selected.max()) >= SEMANTIC_CODEBOOK_SIZE
    ):
        raise RenderContractError(
            f"semantic token must be located at [0,{SEMANTIC_CODEBOOK_SIZE - 1}]"
        )
    return mask_to_lengths(effective)


@dataclass(frozen=True)
class LengthContract:

    samples: int
    stft_frames: int
    latent_frames: int
    stft_right_pad: int
    latent_right_pad: int

    @classmethod
    def from_samples(cls, samples: int) -> "LengthContract":
        stft_frames = samples_to_stft_frames(samples)
        latent_frames = samples_to_latent_frames(samples)
        return cls(
            samples=samples,
            stft_frames=stft_frames,
            latent_frames=latent_frames,
            stft_right_pad=stft_right_padding(samples),
            latent_right_pad=latent_frames_to_samples(latent_frames) - samples,
        )


assert STFT_FRAME_HZ == 100
assert SPEC_FRAMES_PER_LATENT_FRAME == 4
assert SAMPLES_PER_LATENT_FRAME == 1_920
assert LengthContract.from_samples(SEGMENT_SAMPLES) == LengthContract(
    samples=61_440,
    stft_frames=128,
    latent_frames=32,
    stft_right_pad=240,
    latent_right_pad=0,
)


CHANNELS = AUDIO_CHANNELS
N_FFT = STFT_N_FFT
WIN_LENGTH = STFT_WIN_LENGTH
HOP_LENGTH = STFT_HOP_LENGTH
FREQUENCY_BINS = STFT_BINS
LATENT_HZ = LATENT_FRAME_HZ
SEMANTIC_VOCAB_SIZE = SEMANTIC_CODEBOOK_SIZE
num_stft_frames = samples_to_stft_frames
num_latent_frames = samples_to_latent_frames
make_length_mask = lengths_to_mask
