
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .contracts import (
    MELODY_POOL_SIZE,
    MELODY_RELATIVE_MIDI_OFFSET,
    MELODY_RELATIVE_SEMITONE_LIMIT,
    MELODY_TOKEN_MAX,
    MELODY_TOKEN_MIN,
    MELODY_UNVOICED_ID,
)


MELODY_TOKEN_DTYPE = np.uint8


_A4_HZ = 440.0
_A4_MIDI = 69.0


def _voiced_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > 0.0)


def _median_low(values: np.ndarray) -> float:
    ordered = np.sort(values)
    return float(ordered[(ordered.size - 1) // 2])


def _round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5)


def hz_to_midi(frequency_hz: float | Sequence[float] | np.ndarray) -> np.ndarray | float:
    freq = np.asarray(frequency_hz, dtype=np.float64)
    voiced = _voiced_mask(freq)
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = _A4_MIDI + 12.0 * np.log2(np.where(voiced, freq, np.nan) / _A4_HZ)
    return midi if midi.ndim else float(midi)


def midi_to_hz(midi: float | Sequence[float] | np.ndarray) -> np.ndarray | float:
    values = np.asarray(midi, dtype=np.float64)
    freq = _A4_HZ * np.exp2((values - _A4_MIDI) / 12.0)
    return freq if freq.ndim else float(freq)


def median_pool_pitch(
    pitch_hz: Sequence[float] | np.ndarray, pool: int = MELODY_POOL_SIZE
) -> np.ndarray:
    if pool <= 0:
        raise ValueError(f"pool must be positive; received {pool}")
    pitch = np.asarray(pitch_hz, dtype=np.float64).reshape(-1)
    size = pitch.size
    if size == 0:
        return np.zeros(0, dtype=np.float64)

    pooled_frames = -(-size // pool)
    padded = np.full(pooled_frames * pool, np.nan, dtype=np.float64)
    padded[:size] = pitch
    windows = padded.reshape(pooled_frames, pool)

    voiced = _voiced_mask(windows)
    voiced_counts = voiced.sum(axis=1)

    window_sizes = np.full(pooled_frames, pool, dtype=np.int64)
    window_sizes[-1] = size - (pooled_frames - 1) * pool
    keep = (voiced_counts > 0) & (2 * voiced_counts >= window_sizes)


    ordered = np.sort(np.where(voiced, windows, np.nan), axis=1)
    low_median_index = np.maximum(voiced_counts - 1, 0) // 2
    picked = ordered[np.arange(pooled_frames), low_median_index]
    return np.where(keep, picked, 0.0)


def _center_and_quantize(rounded_midi: np.ndarray) -> np.ndarray:
    voiced = np.isfinite(rounded_midi)
    tokens = np.full(rounded_midi.shape, MELODY_UNVOICED_ID, dtype=MELODY_TOKEN_DTYPE)
    if not voiced.any():
        return tokens
    center = _median_low(rounded_midi[voiced])
    relative = np.clip(
        rounded_midi[voiced] - center,
        -MELODY_RELATIVE_SEMITONE_LIMIT,
        MELODY_RELATIVE_SEMITONE_LIMIT,
    )
    tokens[voiced] = (relative + MELODY_RELATIVE_MIDI_OFFSET).astype(MELODY_TOKEN_DTYPE)
    return tokens


def pitch_to_melody_tokens(
    pitch_hz_50hz: Sequence[float] | np.ndarray, pool: int = MELODY_POOL_SIZE
) -> np.ndarray:
    pooled = median_pool_pitch(pitch_hz_50hz, pool=pool)
    return _center_and_quantize(_round_half_up(np.asarray(hz_to_midi(pooled))))


def melody_tokens_to_relative_semitones(
    tokens: Sequence[int] | np.ndarray,
) -> np.ndarray:
    array = np.asarray(tokens)
    relative = array.astype(np.float64) - float(MELODY_RELATIVE_MIDI_OFFSET)
    relative[array == MELODY_UNVOICED_ID] = np.nan
    return relative


def recenter_melody_tokens(tokens: Sequence[int] | np.ndarray) -> np.ndarray:
    return _center_and_quantize(melody_tokens_to_relative_semitones(tokens))


def validate_melody_tokens(
    tokens: Sequence[int] | np.ndarray,
    *,
    require_uint8: bool = True,
    check_centered: bool = True,
    center_tolerance: float = 0.5,
    source: str = "melody token",
) -> np.ndarray:
    array = np.asarray(tokens)
    if array.ndim != 1:
        raise ValueError(f"{source} must be a one-dimensional sequence; received shape={array.shape}")
    if array.dtype.kind not in "iu":
        raise ValueError(f"{source} dtype must be an integer; received {array.dtype}")
    if require_uint8 and array.dtype != MELODY_TOKEN_DTYPE:
        raise ValueError(f"{source} dtype must be uint8; received {array.dtype}")
    if array.size == 0:


        raise ValueError(f"{source} is empty; Equation 3 must produce at least one frame")
    if array.min() < MELODY_TOKEN_MIN or array.max() > MELODY_TOKEN_MAX:
        raise ValueError(
            f"{source} must fall within [{MELODY_TOKEN_MIN}, {MELODY_TOKEN_MAX}]; "
            f"got [{array.min()}, {array.max()}]"
        )

    if check_centered:
        voiced = array[array != MELODY_UNVOICED_ID]
        if voiced.size:
            center = _median_low(voiced.astype(np.float64))
            if abs(center - MELODY_RELATIVE_MIDI_OFFSET) > center_tolerance:
                raise ValueError(
                    f"{source} violates the Eq. 4 invariant: voiced-token median should "
                    f"be approximately {MELODY_RELATIVE_MIDI_OFFSET} within tolerance "
                    f"{center_tolerance}; got {center}"
                )
    return array.astype(MELODY_TOKEN_DTYPE, copy=False)
