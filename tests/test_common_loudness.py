from __future__ import annotations

import math

import numpy as np

from open_qwen_music.common.loudness import loudness_metrics


def test_loudness_metrics_are_finite_for_tone() -> None:
    sample_rate = 24_000
    time = np.arange(sample_rate * 4, dtype=np.float64) / sample_rate
    waveform = 0.1 * np.sin(2.0 * np.pi * 440.0 * time)

    integrated, loudness_range = loudness_metrics(waveform, sample_rate)

    assert math.isfinite(integrated)
    assert loudness_range >= 0.0


def test_loudness_metrics_gate_silence() -> None:
    integrated, loudness_range = loudness_metrics(np.zeros(96_000), 24_000)

    assert integrated == float("-inf")
    assert loudness_range == 0.0
