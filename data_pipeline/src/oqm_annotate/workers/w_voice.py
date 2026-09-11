#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

from _worker import RankWriter, add_common_arguments, distribution, run


#


_FEMALE_FLOOR_HZ = 275.0
_MALE_CEILING_HZ = 185.0


_MAX_CONFIDENCE = 0.9


_F0_MIN_HZ = 70.0
_F0_MAX_HZ = 800.0

_FRAME_LENGTH = 2048

_WINDOW_SEC = 3.0
_MIN_WINDOWS = 6


#


#


#


#


#


#


_MIN_VOCAL_RATIO = 0.30


_BRIGHT_RATIO = 9.0
_DARK_RATIO = 5.0


def _configure_numba_cache(rank: int) -> str:

    path = str(
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "open-qwen-music"
        / "numba"
        / f"voice-rank-{int(rank):04d}"
    )
    os.environ.setdefault("NUMBA_CACHE_DIR", path)
    return os.environ["NUMBA_CACHE_DIR"]


def _f0_track(wave, sample_rate: int):

    import librosa
    import numpy as np

    if wave.numel() < 4 * _FRAME_LENGTH:
        return None

    f0, voiced, _ = librosa.pyin(
        wave.numpy().astype("float32"),
        fmin=_F0_MIN_HZ,
        fmax=_F0_MAX_HZ,
        sr=sample_rate,
        frame_length=_FRAME_LENGTH,
    )
    track = np.where(voiced & ~np.isnan(f0), f0, np.nan)
    if np.count_nonzero(~np.isnan(track)) < 4:
        return None
    return track


def _voiced_f0(wave, sample_rate: int, device):

    import numpy as np
    import torch

    track = _f0_track(wave, sample_rate)
    if track is None:
        return None
    values = track[~np.isnan(track)]
    return torch.from_numpy(values.astype("float32"))


def _window_pitch_profile(track, sample_rate: int) -> dict[str, float] | None:

    import numpy as np

    hop = _FRAME_LENGTH // 4
    per_window = max(1, int(_WINDOW_SEC * sample_rate / hop))
    medians = []
    for start in range(0, len(track), per_window):
        chunk = track[start : start + per_window]
        valid = chunk[~np.isnan(chunk)]

        if valid.size < per_window * 0.25:
            continue
        medians.append(float(np.median(valid)))
    if len(medians) < _MIN_WINDOWS:
        return None

    logs = np.log2(np.array(medians))
    order = np.sort(logs)


    best = None
    for cut in range(1, len(order)):
        low, high = order[:cut], order[cut:]
        cost = float(np.var(low) * low.size + np.var(high) * high.size)
        if best is None or cost < best[0]:
            best = (cost, float(high.mean() - low.mean()), cut / len(order))
    _, gap, cut_share = best
    return {
        "windows": len(medians),
        "window_p10_hz": round(float(np.quantile(medians, 0.10)), 2),
        "window_p50_hz": round(float(np.median(medians)), 2),
        "window_p90_hz": round(float(np.quantile(medians, 0.90)), 2),
        "window_spread_oct": round(
            float(np.quantile(logs, 0.90) - np.quantile(logs, 0.10)), 4
        ),
        "cluster_gap_oct": round(gap, 4),
        "cluster_minor_share": round(min(cut_share, 1.0 - cut_share), 4),
    }


def _spectral_centroid(wave, sample_rate: int) -> float:
    import torch

    window = 2048
    if wave.numel() < window:
        return 0.0
    spectrum = torch.stft(
        wave,
        n_fft=window,
        hop_length=window // 2,
        window=torch.hann_window(window),
        return_complex=True,
    ).abs()
    freqs = torch.linspace(0, sample_rate / 2, spectrum.shape[0]).unsqueeze(1)
    magnitude = spectrum.sum(dim=0, keepdim=True).clamp(min=1e-8)
    centroid = (spectrum * freqs).sum(dim=0, keepdim=True) / magnitude

    energetic = spectrum.sum(dim=0) > spectrum.sum(dim=0).max() * 0.05
    if energetic.sum() == 0:
        return 0.0
    return float(centroid[0][energetic].median())


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser())
    parser.add_argument("--max-analyze-sec", type=float, default=120.0)
    args = parser.parse_args()
    rank, local_rank, world_size = distribution()
    _configure_numba_cache(rank)

    import torch

    from _audio import load_torch

    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    if args.prewarm:
        print(f'{{"rank": {rank}, "device": "{device}", "prewarm": true}}', flush=True)
        return

    def handle(record: dict[str, Any], writer: RankWriter) -> None:
        audio_path = record["audio_path"]
        if not Path(audio_path).exists():
            raise FileNotFoundError(audio_path)

        wave, sample_rate = load_torch(audio_path)
        wave = wave.mean(dim=0)
        limit = int(args.max_analyze_sec * sample_rate)
        if wave.numel() > limit:

            start = (wave.numel() - limit) // 2
            wave = wave[start : start + limit]

        import numpy as np
        import torch as _torch

        track = _f0_track(wave, sample_rate)
        if track is None:
            writer.write(
                {
                    "sample_id": record["sample_id"],
                    "gender_hint": None,
                    "gender_confidence": 0.0,
                    "timbre_hint": None,
                    "note": "not enough voiced frames to estimate pitch",
                }
            )
            return
        f0 = _torch.from_numpy(track[~np.isnan(track)].astype("float32"))
        profile = _window_pitch_profile(track, sample_rate) or {}

        median_f0 = float(f0.median())


        q1, q3 = [float(v) for v in torch.quantile(f0, torch.tensor([0.25, 0.75]))]
        spread = q3 - q1

        vocal_ratio = record.get("vocal_ratio")
        no_voice = vocal_ratio is not None and float(vocal_ratio) < _MIN_VOCAL_RATIO
        if no_voice:


            gender, confidence = None, 0.0
        else:
            gender, confidence = _gender_from_f0(median_f0, spread)
        centroid = _spectral_centroid(wave, sample_rate)


        #


        timbre = None if no_voice else _timbre_from_centroid(centroid, median_f0)

        writer.write(
            {
                "sample_id": record["sample_id"],
                "vocal_ratio": vocal_ratio,
                **({"note": "vocal activity is too low for measurement"} if no_voice else {}),
                "median_f0_hz": round(median_f0, 2),
                "f0_iqr_hz": round(spread, 2),
                "voiced_frames": int(f0.numel()),
                "spectral_centroid_hz": round(centroid, 1),
                "gender_hint": gender,
                "gender_confidence": round(confidence, 4),
                "timbre_hint": timbre,
                "method": "pyin-f0-heuristic",
                **profile,
            }
        )

    run(args, rank=rank, world_size=world_size, handle=handle)


def _gender_from_f0(median_hz: float, iqr_hz: float) -> tuple[str | None, float]:

    if median_hz <= _F0_MIN_HZ:

        #


        #


        #


        #


        #


        return None, 0.0

    if median_hz >= _FEMALE_FLOOR_HZ:
        gender = "female"
        margin = median_hz - _FEMALE_FLOOR_HZ
    elif median_hz <= _MALE_CEILING_HZ:
        gender = "male"
        margin = _MALE_CEILING_HZ - median_hz
    else:


        return None, 0.0


    distance_score = min(1.0, margin / 60.0)
    span = _MAX_CONFIDENCE - 0.5
    return gender, 0.5 + span * distance_score


def _timbre_from_centroid(centroid_hz: float, median_f0_hz: float) -> str | None:

    if centroid_hz <= 0 or median_f0_hz <= _F0_MIN_HZ:
        return None
    ratio = centroid_hz / median_f0_hz
    if math.isnan(ratio):
        return None
    if ratio >= _BRIGHT_RATIO:
        return "bright"
    if ratio <= _DARK_RATIO:
        return "dark"
    return None


if __name__ == "__main__":
    main()
