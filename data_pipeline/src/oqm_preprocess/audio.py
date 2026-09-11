
from __future__ import annotations

import io
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf


#


ANALYSIS_VERSION = 3

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
_FFMPEG_TIMEOUT = 300


_FFMPEG_ONLY_EXT = frozenset({".m4a", ".aac", ".mp4", ".wma", ".amr", ".ape", ".wv"})


_SEEK_REQUIRED_EXT = frozenset({".m4a", ".m4b", ".mp4", ".mov", ".3gp", ".3g2"})


WINDOW_SEC = 8.0
WHOLE_DECODE_MAX_SEC = 60.0


FP_PCM_SCALE = 32768.0

SILENCE_DBFS = -50.0
CLIP_AMPLITUDE = 0.999


class DecodeError(RuntimeError):
    def __init__(self, message: str, flag: str = "decode_failed") -> None:
        super().__init__(message)
        self.flag = flag


@dataclass(slots=True)
class AudioInfo:
    sample_rate: int
    channels: int
    frames: int
    format: str = ""
    subtype: str = ""
    backend: str = "soundfile"

    @property
    def duration_sec(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate > 0 else 0.0


@dataclass(slots=True)
class Analysis:

    info: AudioInfo
    peak_dbfs: float = -120.0
    rms_dbfs: float = -120.0
    near_silent_frame_ratio: float = 1.0
    clipping_ratio: float = 0.0
    channel_correlation: float = float("nan")
    effective_bandwidth_ratio: float = 0.0
    lead_silence_sec: float = 0.0
    tail_silence_sec: float = 0.0
    fingerprint: bytes = b""
    nonfinite: bool = False
    empty: bool = False
    analyzed_seconds: float = 0.0


    header_duration_sec: float = 0.0


    duration_source: str = "header"
    notes: list[str] = field(default_factory=list)


def probe(data: bytes, name_hint: str = "") -> AudioInfo:
    if not data:
        raise DecodeError("byte is empty", "zero_bytes")
    ext = Path(name_hint).suffix.lower()
    if ext not in _FFMPEG_ONLY_EXT:
        try:
            with sf.SoundFile(io.BytesIO(data)) as fh:
                return AudioInfo(
                    sample_rate=int(fh.samplerate),
                    channels=int(fh.channels),
                    frames=int(len(fh)),
                    format=str(fh.format),
                    subtype=str(fh.subtype),
                    backend="soundfile",
                )
        except (RuntimeError, sf.LibsndfileError, OSError):
            pass
    return _ffprobe(data, name_hint)


def _run_with_seek_fallback(
    build: Callable[[str], list[str]], data: bytes, name_hint: str, flag: str,
    ok: Callable[[subprocess.CompletedProcess[bytes]], bool] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    suffix = Path(name_hint).suffix[:16] or ""

    def run_file() -> subprocess.CompletedProcess[bytes]:
        with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
            handle.write(data)
            handle.flush()
            return subprocess.run(build(handle.name), capture_output=True,
                                  timeout=_FFMPEG_TIMEOUT, check=False)

    if Path(name_hint).suffix.lower() in _SEEK_REQUIRED_EXT:
        try:
            return run_file()
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DecodeError(f"execution failed {name_hint}: {exc}", flag) from exc

    try:
        proc = subprocess.run(build("pipe:0"), input=data, capture_output=True,
                              timeout=_FFMPEG_TIMEOUT, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DecodeError(f"execution failed {name_hint}: {exc}", flag) from exc
    if ok(proc) if ok else proc.returncode == 0:
        return proc
    try:
        return run_file()
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DecodeError(f"execution failed {name_hint}: {exc}", flag) from exc


def _ffprobe(data: bytes, name_hint: str) -> AudioInfo:
    import json
    from fractions import Fraction

    def build(source: str) -> list[str]:
        return [
            FFPROBE, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels,codec_name,duration,duration_ts,time_base",
            "-show_entries", "format=duration,format_name",
            "-of", "json", "-i", source,
        ]

    def parse_payload(raw: bytes) -> tuple[dict, dict, float]:
        payload = json.loads(raw or b"{}")
        stream = (payload.get("streams") or [{}])[0]
        duration = float((payload.get("format") or {}).get("duration") or
                         stream.get("duration") or 0.0)
        if duration <= 0 and stream.get("duration_ts") and stream.get("time_base"):
            duration = float(stream["duration_ts"]) * float(Fraction(stream["time_base"]))
        return payload, stream, duration

    def valid(proc: subprocess.CompletedProcess[bytes]) -> bool:
        if proc.returncode != 0:
            return False
        try:
            _, stream, duration = parse_payload(proc.stdout)
            return (int(stream.get("sample_rate") or 0) > 0
                    and int(stream.get("channels") or 0) > 0
                    and duration > 0)
        except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
            return False


    proc = _run_with_seek_fallback(
        build, data, name_hint, "probe_failed", ok=valid)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
        raise DecodeError(f"ffprobe cannot be recognized {name_hint}: {tail[0]}", "probe_failed")
    try:
        payload, stream, duration = parse_payload(proc.stdout)
        sr = int(stream.get("sample_rate") or 0)
        ch = int(stream.get("channels") or 0)
    except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
        raise DecodeError(f"ffprobe Output cannot be parsed {name_hint}: {exc}", "probe_failed") from exc
    if sr <= 0 or ch <= 0 or duration <= 0:
        raise DecodeError(f"ffprobe No valid audio stream given/Duration {name_hint}", "probe_failed")
    return AudioInfo(
        sample_rate=sr,
        channels=ch,
        frames=int(round(duration * sr)),
        format=str((payload.get("format") or {}).get("format_name") or ""),
        subtype=str(stream.get("codec_name") or ""),
        backend="ffmpeg",
    )


def decode_segment(
    data: bytes, info: AudioInfo, start_sec: float, dur_sec: float, name_hint: str = ""
) -> np.ndarray:
    if info.backend == "ffmpeg":
        return _ffmpeg_decode(data, info, start_sec, dur_sec, name_hint)
    start_frame = max(0, int(round(start_sec * info.sample_rate)))
    want = -1 if dur_sec <= 0 else max(1, int(round(dur_sec * info.sample_rate)))
    try:
        with sf.SoundFile(io.BytesIO(data)) as fh:
            if start_frame:
                fh.seek(start_frame)
            block = fh.read(want, dtype="float32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError, OSError) as exc:

        try:
            return _ffmpeg_decode(data, info, start_sec, dur_sec, name_hint)
        except DecodeError:
            raise DecodeError(f"Decoding failed {name_hint}: {exc}", "decode_failed") from exc
    return np.ascontiguousarray(block, dtype=np.float32)


def _measure_frames(data: bytes, info: AudioInfo, name_hint: str) -> int:
    if info.backend != "ffmpeg":
        try:
            with sf.SoundFile(io.BytesIO(data)) as fh:
                total = 0
                while True:
                    block = fh.read(1 << 20, dtype="float32", always_2d=True)
                    if not block.shape[0]:
                        break
                    total += int(block.shape[0])
            if total:
                return total
        except (RuntimeError, sf.LibsndfileError, OSError):
            pass
    try:
        return int(_ffmpeg_decode(data, info, 0.0, -1.0, name_hint).shape[0])
    except DecodeError:
        return 0


def _ffmpeg_decode(
    data: bytes, info: AudioInfo, start_sec: float, dur_sec: float, name_hint: str
) -> np.ndarray:
    def build(source: str) -> list[str]:
        cmd = [FFMPEG, "-v", "error", "-nostdin"]
        if start_sec > 0:
            cmd += ["-ss", f"{start_sec:.6f}"]
        cmd += ["-i", source]
        if dur_sec > 0:
            cmd += ["-t", f"{dur_sec:.6f}"]
        return cmd + ["-map", "a:0", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]

    proc = _run_with_seek_fallback(
        build, data, name_hint, "decode_failed",
        ok=lambda p: p.returncode == 0 and bool(p.stdout))
    if proc.returncode != 0 or not proc.stdout:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["no output"]
        raise DecodeError(f"ffmpeg Decoding failed {name_hint}: {tail[0]}", "decode_failed")
    flat = np.frombuffer(proc.stdout, dtype="<f4")
    ch = max(1, info.channels)
    usable = (flat.size // ch) * ch
    if usable == 0:
        raise DecodeError(f"ffmpeg The output is empty {name_hint}", "empty_audio")
    return np.ascontiguousarray(flat[:usable].reshape(-1, ch), dtype=np.float32)


def _window_starts(duration: float, window: float) -> list[float]:
    if duration <= window * 1.5:
        return [0.0]
    if duration <= window * 3.0:
        return [0.0, max(0.0, duration - window)]
    return [0.0, (duration - window) / 2.0, duration - window]


def analyze(data: bytes, info: AudioInfo, name_hint: str = "") -> Analysis:
    result = Analysis(info=info)
    duration = info.duration_sec
    result.header_duration_sec = duration
    if info.frames <= 0 or duration <= 0:

        block = decode_segment(data, info, 0.0, -1.0, name_hint)
        if block.size == 0:
            result.empty = True
            return result
        info.frames = int(block.shape[0])
        duration = info.duration_sec
        result.duration_source = "decoded"
        blocks = [(0.0, block)]
    elif duration <= WHOLE_DECODE_MAX_SEC:
        block = decode_segment(data, info, 0.0, duration + 1.0, name_hint)
        if block.size == 0:
            result.empty = True
            return result

        info.frames = int(block.shape[0])
        duration = info.duration_sec
        result.duration_source = "decoded"
        blocks = [(0.0, block)]
    else:
        blocks = []
        short = False
        for start in _window_starts(duration, WINDOW_SEC):
            try:
                block = decode_segment(data, info, start, WINDOW_SEC, name_hint)
            except DecodeError:


                if start <= 0:
                    raise
                short = True
                continue
            if block.size:
                blocks.append((start, block))
            elif start > 0:
                short = True
        if not blocks:
            result.empty = True
            return result
        if short:

            info.frames = _measure_frames(data, info, name_hint)
            duration = info.duration_sec
            result.duration_source = "decoded"
        else:


            result.duration_source = "header"

    sr = info.sample_rate
    total_frames = sum(b.shape[0] for _, b in blocks)
    result.analyzed_seconds = total_frames / sr if sr else 0.0

    if any(not np.isfinite(b).all() for _, b in blocks):
        result.nonfinite = True
        return result

    peak = 0.0
    clipped = 0
    sq_sum = 0.0
    silent_frames = 0
    total_analysis_frames = 0
    corr_num = 0.0
    corr_den_l = 0.0
    corr_den_r = 0.0
    mono_blocks: list[np.ndarray] = []

    for _, block in blocks:
        peak = max(peak, float(np.abs(block).max()))
        clipped += int((np.abs(block) >= CLIP_AMPLITUDE).sum())
        sq_sum += float(np.square(block, dtype=np.float64).sum())
        if block.shape[1] >= 2:
            left = block[:, 0].astype(np.float64)
            right = block[:, 1].astype(np.float64)
            corr_num += float((left * right).sum())
            corr_den_l += float((left * left).sum())
            corr_den_r += float((right * right).sum())
        mono = block.mean(axis=1, dtype=np.float32)
        mono_blocks.append(mono)
        n_sil, n_tot = _silence_stats(mono, sr)
        silent_frames += n_sil
        total_analysis_frames += n_tot

    n_samples = sum(b.shape[0] * b.shape[1] for _, b in blocks)
    result.peak_dbfs = _dbfs(peak)
    result.clipping_ratio = clipped / n_samples if n_samples else 0.0
    result.rms_dbfs = _dbfs(float(np.sqrt(sq_sum / n_samples)) if n_samples else 0.0)
    result.near_silent_frame_ratio = (
        silent_frames / total_analysis_frames if total_analysis_frames else 1.0
    )
    if corr_den_l > 0 and corr_den_r > 0:
        result.channel_correlation = corr_num / float(np.sqrt(corr_den_l * corr_den_r))

    if peak <= 0.0:
        result.empty = True
        return result

    first_mono = mono_blocks[0]
    result.lead_silence_sec = _edge_silence_sec(first_mono, sr, from_end=False)
    last_mono = mono_blocks[-1]
    result.tail_silence_sec = _edge_silence_sec(last_mono, sr, from_end=True)

    result.effective_bandwidth_ratio = _bandwidth_ratio(mono_blocks, sr)
    result.fingerprint = pcm_fingerprint(mono_blocks)
    return result


def _dbfs(amplitude: float) -> float:
    if amplitude <= 1e-12:
        return -120.0
    return float(20.0 * np.log10(amplitude))


def _frame_view(mono: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if mono.size < frame:
        if not mono.size:
            return np.zeros((0, frame), dtype=np.float32)
        padded = np.zeros(frame, dtype=np.float32)
        padded[: mono.size] = mono
        return padded[np.newaxis, :]
    n = 1 + (mono.size - frame) // hop
    stride = mono.strides[0]
    return np.lib.stride_tricks.as_strided(
        mono, shape=(n, frame), strides=(hop * stride, stride), writeable=False
    )


def _silence_stats(mono: np.ndarray, sr: int) -> tuple[int, int]:
    frame = max(1, int(0.02 * sr))
    frames = _frame_view(mono, frame, frame)
    if frames.shape[0] == 0:
        return 0, 0
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    threshold = 10.0 ** (SILENCE_DBFS / 20.0)
    return int((rms < threshold).sum()), int(frames.shape[0])


def _edge_silence_sec(mono: np.ndarray, sr: int, *, from_end: bool) -> float:
    frame = max(1, int(0.02 * sr))
    frames = _frame_view(mono, frame, frame)
    if frames.shape[0] == 0:
        return 0.0
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    threshold = 10.0 ** (SILENCE_DBFS / 20.0)
    loud = np.flatnonzero(rms >= threshold)
    if loud.size == 0:
        return float(frames.shape[0] * frame / sr)
    idx = (frames.shape[0] - 1 - loud[-1]) if from_end else loud[0]
    return float(idx * frame / sr)


def pcm_fingerprint(mono_blocks: list[np.ndarray]) -> bytes:
    import hashlib

    digest = hashlib.sha256()
    for index, mono in enumerate(mono_blocks):
        digest.update(index.to_bytes(2, "big"))
        if mono.size == 0:
            continue
        grid = np.clip(np.rint(mono.astype(np.float64) * FP_PCM_SCALE), -32768, 32767)
        digest.update(grid.astype("<i2").tobytes())
    return digest.digest()[:8]


_BANDWIDTH_FLOOR_DB = 80.0


def _bandwidth_ratio(mono_blocks: list[np.ndarray], sr: int) -> float:
    nyquist = sr / 2.0
    if nyquist <= 0:
        return 0.0
    sizes = [mono.size for mono in mono_blocks if mono.size]
    if not sizes:
        return 0.0


    frame = min(4096, max(256, 1 << int(np.floor(np.log2(max(min(sizes), 256))))))
    window = np.hanning(frame).astype(np.float32)
    total: np.ndarray | None = None
    for mono in mono_blocks:
        frames = _frame_view(mono, frame, frame)
        if frames.shape[0] == 0:
            continue
        spec = (np.abs(np.fft.rfft(frames * window, axis=1)) ** 2).mean(axis=0)
        total = spec if total is None else total + spec
    if total is None or total.size < 2:
        return 0.0
    peak = float(total.max())
    if peak <= 0.0:
        return 0.0
    above = np.nonzero(total > peak * 10.0 ** (-_BANDWIDTH_FLOOR_DB / 10.0))[0]
    if above.size == 0:
        return 0.0
    return float(min(1.0, int(above[-1]) / (total.size - 1)))
