#!/usr/bin/env python3

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "AudioProbe",
    "DURATION_MISMATCH_TOLERANCE_SEC",
    "DecodeError",
    "DurationMismatch",
    "SNDFILE_EXACT_DURATION",
    "assert_duration_matches",
    "ffmpeg_decode",
    "load_mono",
    "load_torch",
    "probe",
    "read_samples",
    "sndfile_duration_is_exact",
]


class DecodeError(RuntimeError):
    pass


class DurationMismatch(RuntimeError):
    pass


#:


#:


SNDFILE_EXACT_DURATION = frozenset(
    {"WAV", "WAVEX", "W64", "RF64", "FLAC", "OGG", "AIFF", "AU", "CAF", "SD2", "VOC"}
)


@dataclass(frozen=True, slots=True)
class AudioProbe:
    duration_sec: float
    sample_rate: int
    channels: int

    prober: str


    duration_method: str


def sndfile_duration_is_exact(fmt: str | None) -> bool:
    return str(fmt or "").upper() in SNDFILE_EXACT_DURATION


def _run_tool(
    command: list[str],
    *,
    timeout: int,
    path: Path | str | None = None,
    timeout_retries: int = 0,
    retry_delay_sec: float = 5.0,
):

    for attempt in range(timeout_retries + 1):
        try:
            return subprocess.run(command, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            if attempt >= timeout_retries:
                raise


            time.sleep(retry_delay_sec * (attempt + 1))
        except FileNotFoundError as exc:


            _warn_no_fallback(command[0], path)
            raise DecodeError(
                f"{command[0]} is not on PATH, and libsndfile could not decode the input. "
                f"(Minimum operating environment ,see TRAINING.md)"
            ) from exc
        except OSError as exc:
            raise DecodeError(
                f"{command[0]} Can\'t get up:{type(exc).__name__}: {exc}"
            ) from exc
    raise AssertionError("unreachable")


def _ffprobe_stream(path: Path | str) -> dict[str, Any]:
    proc = _run_tool(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels", "-of", "json", str(path),
        ],
        timeout=300,
        path=path,
        timeout_retries=2,
    )
    if proc.returncode != 0:
        raise DecodeError(
            f"ffprobe could not read the audio stream ({path}): "
            f"rc={proc.returncode} {proc.stderr.decode('utf-8', 'replace')[:200]}"
        )
    streams = (json.loads(proc.stdout) or {}).get("streams") or []
    if not streams:
        raise DecodeError(f"ffprobe says there is no audio stream in this file:{path}")
    return streams[0]


def ffmpeg_decode(
    path: Path | str, *, sample_rate: int | None = None, mono: bool = False
):

    import numpy as np

    stream = _ffprobe_stream(path)
    src_rate = int(stream.get("sample_rate") or 0)
    src_channels = int(stream.get("channels") or 0)
    if src_rate <= 0 or src_channels <= 0:
        raise DecodeError(f"ffprobe is/channel is not available:{stream!r}({path})")

    out_rate = int(sample_rate or src_rate)
    out_channels = 1 if mono else src_channels

    command = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le"]
    if mono:
        command += ["-ac", "1"]
    if sample_rate:
        command += ["-ar", str(out_rate)]
    command.append("-")

    proc = _run_tool(command, timeout=3600, path=path)
    if proc.returncode != 0:
        raise DecodeError(
            f"ffmpeg Decoding failed({path}):"
            f"rc={proc.returncode} {proc.stderr.decode('utf-8', 'replace')[:300]}"
        )
    flat = np.frombuffer(proc.stdout, dtype="<f4")
    if out_channels > 1:
        usable = (flat.size // out_channels) * out_channels
        flat = flat[:usable]
    return flat.reshape(-1, out_channels).astype("float32", copy=False), out_rate


def _sf_read(path: Path | str):
    import soundfile

    wave, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return wave, int(rate)


def read_samples(path: Path | str, *, always_2d: bool = False):

    try:
        wave, rate = _sf_read(path)
        used = "soundfile"
    except Exception as sf_exc:  # noqa: BLE001 - ,
        try:
            wave, rate = ffmpeg_decode(path)
        except Exception as ff_exc:  # noqa: BLE001
            raise DecodeError(
                f"Neither decoder could read the audio ({path}): "
                f"soundfile={type(sf_exc).__name__}: {sf_exc};"
                f"ffmpeg={type(ff_exc).__name__}: {ff_exc}"
            ) from sf_exc
        used = "ffmpeg"
    if used == "ffmpeg":
        _warn_fallback(path, "read_samples")
    if not always_2d and wave.ndim == 2 and wave.shape[1] == 1:
        wave = wave[:, 0]
    return wave, rate


def load_mono(path: Path | str, sample_rate: int):

    import numpy as np

    try:
        import librosa

        wave, rate = librosa.load(str(path), sr=sample_rate, mono=True)
        return np.asarray(wave, dtype="float32"), int(rate)
    except Exception as lb_exc:  # noqa: BLE001
        try:
            wave, rate = ffmpeg_decode(path, sample_rate=sample_rate, mono=True)
        except Exception as ff_exc:  # noqa: BLE001
            raise DecodeError(
                f"Neither decoder could read the audio ({path}): "
                f"librosa={type(lb_exc).__name__}: {lb_exc};"
                f"ffmpeg={type(ff_exc).__name__}: {ff_exc}"
            ) from lb_exc
    _warn_fallback(path, "load_mono")
    return wave[:, 0], rate


def load_torch(path: Path | str):

    import torch

    try:
        import torchaudio


        wave, rate = torchaudio.load(str(path), backend="soundfile")
        return wave, int(rate)
    except Exception as ta_exc:  # noqa: BLE001
        try:
            array, rate = ffmpeg_decode(path)
        except Exception as ff_exc:  # noqa: BLE001
            raise DecodeError(
                f"Neither decoder could read the audio ({path}): "
                f"torchaudio={type(ta_exc).__name__}: {ta_exc};"
                f"ffmpeg={type(ff_exc).__name__}: {ff_exc}"
            ) from ta_exc
    _warn_fallback(path, "load_torch")
    return torch.from_numpy(array.T.copy()), rate


def _sf_decoded_duration(path: Path | str) -> tuple[float, int, int]:

    import soundfile

    with soundfile.SoundFile(str(path)) as handle:
        rate = int(handle.samplerate)
        channels = int(handle.channels)
        frames = 0
        while True:
            block = handle.read(1 << 18, dtype="float32", always_2d=True)
            if not len(block):
                break
            frames += len(block)
    if rate <= 0:
        raise DecodeError(f"soundfile reported an invalid sample rate {rate} for {path}")
    return frames / rate, rate, channels


def probe(path: Path | str) -> AudioProbe:

    path = Path(path)
    info = None
    try:
        import soundfile

        info = soundfile.info(str(path))
    except Exception:  # noqa: BLE001 -  ffmpeg
        info = None

    if info is not None and sndfile_duration_is_exact(info.format):
        return AudioProbe(
            float(info.duration), int(info.samplerate), int(info.channels),
            "soundfile", "sndfile_header",
        )

    if info is not None:
        try:
            duration, rate, channels = _sf_decoded_duration(path)
            return AudioProbe(duration, rate, channels, "soundfile", "sndfile_decode")
        except Exception:  # noqa: BLE001 -  ffmpeg
            pass

    try:
        wave, rate = ffmpeg_decode(path)
    except DecodeError as exc:
        size = path.stat().st_size if path.exists() else -1


        raise DecodeError(
            f"Neither decoder could read the duration ({path.name}, {size} bytes): "
            f"soundfile {'cannot be opened' if info is None else 'Decoding failed'};{exc}"
        ) from exc
    _warn_fallback(path, "probe")
    return AudioProbe(
        wave.shape[0] / rate, rate, wave.shape[1], "ffmpeg", "ffmpeg_decode"
    )


#:


#:


#:


#:

DURATION_MISMATCH_TOLERANCE_SEC = 0.1


def assert_duration_matches(
    path: Path | str,
    *,
    declared_sec: float | None,
    measured_sec: float,
    where: str,
    tolerance: float = DURATION_MISMATCH_TOLERANCE_SEC,
) -> None:

    if declared_sec is None:
        return
    declared = float(declared_sec)


    if declared <= 0.0:
        return

    drift = abs(float(measured_sec) - declared)
    if drift <= tolerance:
        return

    raise DurationMismatch(
        f"{where} decoded duration differs for {Path(path).name}: record={declared:.3f}s, "
        f"measured={float(measured_sec):.3f}s, drift={drift:.3f}s, tolerance={tolerance}s. "
        "Confirm that the source data is stable and rerun the sample."
    )


FALLBACK_COUNTS: dict[str, int] = {}


#:


#:


FALLBACK_UNAVAILABLE: dict[str, int] = {}


FALLBACK_UNAVAILABLE_FILES: set[str] = set()


def _emit(payload: dict) -> None:
    import sys

    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)


def _warn_fallback(path: Path | str, where: str) -> None:
    FALLBACK_COUNTS[where] = FALLBACK_COUNTS.get(where, 0) + 1
    _emit({
        "warn": "libsndfile failed; falling back to ffmpeg", "where": where,
        "path": str(path), "count": FALLBACK_COUNTS[where],
    })


def _warn_no_fallback(tool: str, path: Path | str | None = None) -> None:

    FALLBACK_UNAVAILABLE[tool] = FALLBACK_UNAVAILABLE.get(tool, 0) + 1
    if path is not None:
        FALLBACK_UNAVAILABLE_FILES.add(str(path))
    _emit({
        "warn": f"{tool} is not on PATH; fallback is unavailable and the sample cannot be decoded",
        "tool": tool,
        "path": None if path is None else str(path),
        "count": FALLBACK_UNAVAILABLE[tool],
        "files": len(FALLBACK_UNAVAILABLE_FILES),
        "hint": "Install ffmpeg to decode these samples; see the training guide",
    })
