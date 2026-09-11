
from __future__ import annotations

import math

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

_SHELF_GAIN_DB = 4.0
_SHELF_Q = 1.0 / math.sqrt(2.0)
_SHELF_FC = 1681.974450955533
_HIGHPASS_Q = 0.5
_HIGHPASS_FC = 38.13547087602444
_BLOCK_SEC = 0.400
_BLOCK_OVERLAP = 0.75
_SHORT_TERM_SEC = 3.0
_ABSOLUTE_GATE_LUFS = -70.0
_INTEGRATED_RELATIVE_GATE_LU = 10.0
_RANGE_RELATIVE_GATE_LU = 20.0
_FILTER_CHUNK = 1 << 21


def _shelf_coefficients(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    k = math.tan(math.pi * _SHELF_FC / sample_rate)
    vh = 10.0 ** (_SHELF_GAIN_DB / 20.0)
    vb = vh**0.4996667741545416
    denominator = 1.0 + k / _SHELF_Q + k * k
    return (
        np.array(
            [
                (vh + vb * k / _SHELF_Q + k * k) / denominator,
                2.0 * (k * k - vh) / denominator,
                (vh - vb * k / _SHELF_Q + k * k) / denominator,
            ]
        ),
        np.array(
            [
                1.0,
                2.0 * (k * k - 1.0) / denominator,
                (1.0 - k / _SHELF_Q + k * k) / denominator,
            ]
        ),
    )


def _highpass_coefficients(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    k = math.tan(math.pi * _HIGHPASS_FC / sample_rate)
    denominator = 1.0 + k / _HIGHPASS_Q + k * k
    return (
        np.array([1.0, -2.0, 1.0]),
        np.array(
            [
                1.0,
                2.0 * (k * k - 1.0) / denominator,
                (1.0 - k / _HIGHPASS_Q + k * k) / denominator,
            ]
        ),
    )


def k_weight(waveform: np.ndarray, sample_rate: int) -> np.ndarray:

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    matrix = np.atleast_2d(np.asarray(waveform))
    if matrix.shape[-1] == 0:
        return np.empty(matrix.shape, dtype=np.float32)
    shelf_b, shelf_a = _shelf_coefficients(sample_rate)
    highpass_b, highpass_a = _highpass_coefficients(sample_rate)
    sos = np.array(
        [
            [shelf_b[0], shelf_b[1], shelf_b[2], shelf_a[0], shelf_a[1], shelf_a[2]],
            [
                highpass_b[0],
                highpass_b[1],
                highpass_b[2],
                highpass_a[0],
                highpass_a[1],
                highpass_a[2],
            ],
        ],
        dtype=np.float64,
    )
    output = np.empty(matrix.shape, dtype=np.float32)
    steady = sosfilt_zi(sos)
    first = matrix[:, 0].astype(np.float64)
    state = steady[:, None, :] * first[None, :, None]
    for start in range(0, matrix.shape[-1], _FILTER_CHUNK):
        chunk = matrix[:, start : start + _FILTER_CHUNK].astype(np.float64)
        filtered, state = sosfilt(sos, chunk, axis=-1, zi=state)
        output[:, start : start + chunk.shape[-1]] = filtered
    return output


def _channel_weights(channels: int) -> np.ndarray:
    weights = np.ones(channels)
    if channels >= 6:
        weights[3] = 0.0
        weights[4] = weights[5] = 1.41
    elif channels == 5:
        weights[3] = weights[4] = 1.41
    return weights[:, None]


def _block_loudness(
    filtered: np.ndarray,
    sample_rate: int,
    window_sec: float,
) -> np.ndarray:
    block = int(round(window_sec * sample_rate))
    divisions = max(1, int(round(1.0 / (1.0 - _BLOCK_OVERLAP))))
    step = max(1, block // divisions)
    block = step * divisions
    channels, length = filtered.shape
    if length < block:
        return np.zeros(0)
    num_segments = length // step
    segment_energy = np.empty((channels, num_segments), dtype=np.float64)
    segments_per_chunk = max(1, _FILTER_CHUNK // step)
    for begin in range(0, num_segments, segments_per_chunk):
        end = min(num_segments, begin + segments_per_chunk)
        chunk = filtered[:, begin * step : end * step].astype(np.float64)
        chunk *= chunk
        segment_energy[:, begin:end] = chunk.reshape(
            channels, end - begin, step
        ).sum(axis=-1)
    num_blocks = num_segments - divisions + 1
    if num_blocks <= 0:
        return np.zeros(0)
    prefix = np.concatenate(
        [np.zeros((channels, 1)), np.cumsum(segment_energy, axis=-1)],
        axis=-1,
    )
    starts = np.arange(num_blocks)
    per_channel = (prefix[:, starts + divisions] - prefix[:, starts]) / block
    weighted = (_channel_weights(channels) * per_channel).sum(axis=0)
    return -0.691 + 10.0 * np.log10(np.maximum(weighted, 1e-12))


def _integrated_from_blocks(loudness: np.ndarray) -> float:
    if loudness.size == 0:
        return float("-inf")
    energy = 10.0 ** ((loudness + 0.691) / 10.0)
    absolute = loudness > _ABSOLUTE_GATE_LUFS
    if not absolute.any():
        return float("-inf")
    first_pass = -0.691 + 10.0 * math.log10(float(energy[absolute].mean()))
    kept = absolute & (loudness > first_pass - _INTEGRATED_RELATIVE_GATE_LU)
    if not kept.any():
        return float(first_pass)
    return float(-0.691 + 10.0 * math.log10(float(energy[kept].mean())))


def _range_from_blocks(loudness: np.ndarray) -> float:
    if loudness.size == 0:
        return 0.0
    absolute = loudness > _ABSOLUTE_GATE_LUFS
    if not absolute.any():
        return 0.0
    energy = 10.0 ** ((loudness + 0.691) / 10.0)
    mean_loudness = -0.691 + 10.0 * math.log10(float(energy[absolute].mean()))
    kept = loudness[absolute & (loudness > mean_loudness - _RANGE_RELATIVE_GATE_LU)]
    if kept.size < 2:
        return 0.0
    return float(np.percentile(kept, 95) - np.percentile(kept, 10))


def loudness_metrics(waveform: np.ndarray, sample_rate: int) -> tuple[float, float]:

    matrix = np.atleast_2d(np.asarray(waveform))
    filtered = k_weight(matrix, sample_rate)
    integrated = _integrated_from_blocks(
        _block_loudness(filtered, sample_rate, _BLOCK_SEC)
    )
    loudness_range = _range_from_blocks(
        _block_loudness(filtered, sample_rate, _SHORT_TERM_SEC)
    )
    return integrated, loudness_range
