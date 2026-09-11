
from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pytest
import torch

from open_qwen_music.tokenizer.data import TokenizerDataset


def _write_manifest(
    tmp_path: Path,
    *,
    lufs: float | None,
    amplitude: int = 4000,
    peak: float | None = None,
) -> Path:
    audio = tmp_path / f"clip_{amplitude}.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)

        frame = struct.pack("<hh", amplitude, -amplitude)
        output.writeframes(frame * 24_000 * 2)
    audio_meta: dict[str, object] = {"start_sec": 0.0, "duration_sec": 4.0}
    if lufs is not None:
        audio_meta["lufs_i"] = lufs
    if peak is not None:
        audio_meta["peak_full"] = peak
    manifest = tmp_path / f"manifest_{amplitude}_{lufs}_{peak}.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": f"clip-{amplitude}",
                "split": "train",
                "audio_path": str(audio),
                "audio": audio_meta,
                "training": {"loss_heads": {"bestrq": True}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _rms(waveform: torch.Tensor) -> float:
    return float(waveform.square().mean().sqrt())


def test_disabled_by_default_leaves_waveform_untouched(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, lufs=-30.0)
    baseline = TokenizerDataset(manifest, stage=1, max_duration_sec=4.0, random_crop=False)
    assert baseline.normalize_target_lufs is None
    plain = baseline[0]["waveform"]

    normalized = TokenizerDataset(
        manifest,
        stage=1,
        max_duration_sec=4.0,
        random_crop=False,
        normalize_target_lufs=-20.0,
    )[0]["waveform"]

    assert _rms(normalized) / _rms(plain) == pytest.approx(10 ** (10 / 20), rel=1e-3)


def test_gain_equalizes_two_clips_of_different_level(tmp_path: Path) -> None:

    quiet = _write_manifest(tmp_path, lufs=-30.0, amplitude=1000)
    loud = _write_manifest(tmp_path, lufs=-14.0, amplitude=8000)
    kwargs = dict(stage=1, max_duration_sec=4.0, random_crop=False, normalize_target_lufs=-20.0)
    quiet_rms = _rms(TokenizerDataset(quiet, **kwargs)[0]["waveform"])
    loud_rms = _rms(TokenizerDataset(loud, **kwargs)[0]["waveform"])


    assert quiet_rms > _rms(TokenizerDataset(quiet, stage=1, max_duration_sec=4.0, random_crop=False)[0]["waveform"])
    assert loud_rms < _rms(TokenizerDataset(loud, stage=1, max_duration_sec=4.0, random_crop=False)[0]["waveform"])


def test_boost_is_clamped(tmp_path: Path) -> None:

    manifest = _write_manifest(tmp_path, lufs=-60.0)
    plain = TokenizerDataset(manifest, stage=1, max_duration_sec=4.0, random_crop=False)[0]
    clamped = TokenizerDataset(
        manifest,
        stage=1,
        max_duration_sec=4.0,
        random_crop=False,
        normalize_target_lufs=-20.0,
        normalize_max_boost_db=12.0,
    )[0]
    ratio = _rms(clamped["waveform"]) / _rms(plain["waveform"])
    assert ratio == pytest.approx(10 ** (12 / 20), rel=1e-3)
    assert ratio < 10 ** (40 / 20)


def test_attenuation_is_not_clamped(tmp_path: Path) -> None:

    manifest = _write_manifest(tmp_path, lufs=-2.0)
    plain = TokenizerDataset(manifest, stage=1, max_duration_sec=4.0, random_crop=False)[0]
    normalized = TokenizerDataset(
        manifest,
        stage=1,
        max_duration_sec=4.0,
        random_crop=False,
        normalize_target_lufs=-20.0,
        normalize_max_boost_db=12.0,
    )[0]
    ratio = _rms(normalized["waveform"]) / _rms(plain["waveform"])
    assert ratio == pytest.approx(10 ** (-18 / 20), rel=1e-3)


def test_peak_guard_caps_lufs_gain_without_limiter(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, lufs=-30.0, peak=0.8)
    plain = TokenizerDataset(
        manifest, stage=1, max_duration_sec=4.0, random_crop=False
    )[0]["waveform"]
    guarded = TokenizerDataset(
        manifest,
        stage=1,
        max_duration_sec=4.0,
        random_crop=False,
        normalize_target_lufs=-20.0,
        normalize_max_peak_dbfs=-3.0,
    )[0]["waveform"]
    peak_guard_db = -3.0 - 20.0 * math.log10(0.8)
    ratio = _rms(guarded) / _rms(plain)
    assert ratio == pytest.approx(10 ** (peak_guard_db / 20), rel=1e-3)


def test_peak_guard_requires_manifest_peak(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, lufs=-30.0)
    dataset = TokenizerDataset(
        manifest,
        stage=1,
        max_duration_sec=4.0,
        random_crop=False,
        normalize_target_lufs=-20.0,
        normalize_max_peak_dbfs=-3.0,
    )
    with pytest.raises(ValueError, match="peak_full"):
        dataset[0]


def test_missing_lufs_raises_instead_of_silently_skipping(tmp_path: Path) -> None:

    manifest = _write_manifest(tmp_path, lufs=None)
    dataset = TokenizerDataset(
        manifest,
        stage=1,
        max_duration_sec=4.0,
        random_crop=False,
        normalize_target_lufs=-20.0,
    )
    with pytest.raises(ValueError, match="lufs_i"):
        dataset[0]


def test_gain_is_independent_of_crop_window(tmp_path: Path) -> None:

    manifest = _write_manifest(tmp_path, lufs=-26.0)
    gains = []
    for max_duration in (1.0, 2.0, 4.0):
        plain = TokenizerDataset(
            manifest, stage=1, max_duration_sec=max_duration, random_crop=False
        )[0]["waveform"]
        scaled = TokenizerDataset(
            manifest,
            stage=1,
            max_duration_sec=max_duration,
            random_crop=False,
            normalize_target_lufs=-20.0,
        )[0]["waveform"]
        gains.append(_rms(scaled) / _rms(plain))
    assert max(gains) - min(gains) < 1e-4
    assert gains[0] == pytest.approx(10 ** (6 / 20), rel=1e-3)
