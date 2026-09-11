#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import math
import os
from pathlib import Path
from typing import Any

from _worker import (
    RankWriter,
    add_common_arguments,
    distribution,
    run,
    setup_device,
)

SEPARATE_SR = 44_100
OUTPUT_SR = 16_000


_DEFAULT_CPU_THREADS: int | None = None


_WORKER_CPU_THREADS: int | None = None


def _auto_cpu_threads() -> int:

    return max(1, min(4, (os.cpu_count() or 8) // 8))


def _resolve_cpu_threads(requested: int, default_threads: int) -> int:

    if requested:
        return max(1, int(requested))
    return max(1, min(_auto_cpu_threads(), default_threads))


@contextlib.contextmanager
def _reduction_threads():

    import torch

    if _DEFAULT_CPU_THREADS is None or _WORKER_CPU_THREADS is None:
        yield
        return
    torch.set_num_threads(_DEFAULT_CPU_THREADS)
    try:
        yield
    finally:
        torch.set_num_threads(_WORKER_CPU_THREADS)


def _load_audio(path: str, target_sr: int):

    import torch
    import torchaudio

    from _audio import load_torch

    wave, sample_rate = load_torch(path)
    if sample_rate != target_sr:
        wave = torchaudio.functional.resample(wave, sample_rate, target_sr)
    if wave.shape[0] == 1:
        wave = wave.repeat(2, 1)
    elif wave.shape[0] > 2:

        #


        #


        #


        #


        wave = wave[:2]
    return wave.to(torch.float32)


def _assert_chunking_sane(
    chunk: int, overlap: int, chunk_sec: float, overlap_sec: float
) -> None:

    if chunk <= 0:
        raise ValueError(
            f"--chunk-sec {chunk_sec} produces only {chunk} samples at {SEPARATE_SR} Hz"
        )
    if overlap >= chunk:
        raise ValueError(
            f"--overlap-sec {overlap_sec} is not less than --chunk-sec {chunk_sec}"
            f" ({overlap} >= {chunk} samples); the chunk cursor would not advance"
        )
    if 0 < overlap < 2:
        raise ValueError(
            f"--overlap-sec {overlap_sec} in {SEPARATE_SR} Hz Only {overlap} "
            f"sample; use zero overlap or at least two samples for a valid fade"
        )


def _chunk_window(length: int, overlap: int, *, fade_in: bool, fade_out: bool):

    import torch

    window = torch.ones(length, dtype=torch.float32)
    if overlap <= 0:
        return window
    ramp = torch.linspace(0.0, 1.0, overlap)
    if fade_in:
        window[:overlap] = ramp[: min(overlap, length)]
    if fade_out:
        tail = min(overlap, length)
        window[-tail:] = torch.flip(ramp[:tail], dims=[0])
    return window


def _separate_chunked(model, wave, device, *, chunk_sec: float, overlap_sec: float):

    import torch

    chunk = int(chunk_sec * SEPARATE_SR)
    overlap = int(overlap_sec * SEPARATE_SR)
    _assert_chunking_sane(chunk, overlap, chunk_sec, overlap_sec)
    stride = chunk - overlap
    total = wave.shape[-1]

    output = torch.zeros_like(wave)


    accompaniment = torch.zeros_like(wave)
    weight = torch.zeros(total, dtype=torch.float32)

    reference = wave.mean(dim=0)


    with _reduction_threads():
        scale = reference.std() + 1e-8
        offset = reference.mean()
    normalized = (wave - offset) / scale

    position = 0
    while position < total:
        end = min(position + chunk, total)
        segment = normalized[:, position:end].unsqueeze(0).to(device)
        with torch.no_grad():
            sources = model(segment)

        vocals = sources[0, 3].detach().cpu()
        others = sources[0, :3].sum(dim=0).detach().cpu()

        window = _chunk_window(
            end - position, overlap, fade_in=position > 0, fade_out=end < total
        )

        output[:, position:end] += vocals * window
        accompaniment[:, position:end] += others * window
        weight[position:end] += window
        if end >= total:
            break
        position += stride

    normalizer = weight.clamp(min=1e-6)
    return output / normalizer * scale, accompaniment / normalizer * scale


FRAME_SEC = 0.05


def _rms_dbfs(mono) -> float:

    import torch

    with _reduction_threads():
        value = float(torch.sqrt(mono.pow(2).mean() + 1e-20))
    return 20.0 * math.log10(max(value, 1e-10))


def _normalize(wave, *, stem_peak_db: float, mix_peak_db: float, target_dbfs: float,
               reference: str):

    peak_db = mix_peak_db if reference == "mix" else stem_peak_db
    gain_db = target_dbfs - peak_db

    gain_db = min(gain_db, -0.1 - stem_peak_db)
    scale = 10.0 ** (gain_db / 20.0)
    return wave * scale, gain_db


def _frame_db(mono, sample_rate: int):

    import torch

    frame = int(FRAME_SEC * sample_rate)
    if mono.numel() < frame:
        return None
    frames = mono[: mono.shape[0] // frame * frame].reshape(-1, frame)
    rms = frames.pow(2).mean(dim=1).sqrt()
    return 20 * torch.log10(rms.clamp(min=1e-10))


def _activity_profile(vocals, accompaniment, sample_rate: int):

    vocal_db = _frame_db(vocals, sample_rate)
    accompaniment_db = _frame_db(accompaniment, sample_rate)
    if vocal_db is None or accompaniment_db is None:
        return None
    length = min(len(vocal_db), len(accompaniment_db))
    return vocal_db[:length] - accompaniment_db[:length]


def _intervals_from_profile(profile, *, threshold_db: float, min_sec: float):

    if profile is None or profile.numel() == 0:
        return [], 0.0
    active = profile > threshold_db

    intervals: list[list[float]] = []
    start_index: int | None = None
    for index, flag in enumerate(active.tolist()):
        if flag and start_index is None:
            start_index = index
        elif not flag and start_index is not None:
            intervals.append([start_index * 0.05, index * 0.05])
            start_index = None
    if start_index is not None:
        intervals.append([start_index * 0.05, len(active) * 0.05])

    merged: list[list[float]] = []
    for interval in intervals:

        if merged and interval[0] - merged[-1][1] < 0.3:
            merged[-1][1] = interval[1]
        else:
            merged.append(interval)
    kept = [item for item in merged if item[1] - item[0] >= min_sec]
    ratio = sum(end - start for start, end in kept) / (len(active) * 0.05)
    return kept, ratio


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser())
    parser.add_argument("--backend", default="hdemucs", choices=["hdemucs", "roformer"])
    parser.add_argument("--model-dir", default="", help="RoFormer checkpoint directory")
    parser.add_argument("--stem-dir", required=False, default="", help="Stem output directory")
    parser.add_argument("--chunk-sec", type=float, default=10.0)
    parser.add_argument("--overlap-sec", type=float, default=1.0)
    parser.add_argument(
        "--activity-threshold-db",
        type=float,
        default=-12.0,
        help="Minimum vocal-to-accompaniment level, in dB, for active vocal frames.",
    )
    parser.add_argument("--min-activity-sec", type=float, default=0.4)
    parser.add_argument("--target-dbfs", type=float, default=-3.0)
    parser.add_argument(
        "--normalize-reference",
        default="mix",
        choices=["mix", "stem"],
        help="Reference used for peak normalization. `mix` preserves the stem level "
        "relative to the mixture; `stem` normalizes the stem independently.",
    )
    parser.add_argument(
        "--save-activity-profile",
        default="false",
        help="Write frame-level activity values. This can substantially increase output size.",
    )
    parser.add_argument(
        "--keep-stem",
        default="true",
        help="Set to false to discard the full-rate stem after producing the ASR input.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="CPU tensor threads per rank. Use 0 for automatic allocation.",
    )
    args = parser.parse_args()

    rank, local_rank, world_size = distribution()

    import torch
    import torchaudio

    from _audio import assert_duration_matches

    global _DEFAULT_CPU_THREADS, _WORKER_CPU_THREADS
    _DEFAULT_CPU_THREADS = torch.get_num_threads()
    _WORKER_CPU_THREADS = _resolve_cpu_threads(args.cpu_threads, _DEFAULT_CPU_THREADS)
    torch.set_num_threads(_WORKER_CPU_THREADS)

    device = setup_device(local_rank)

    if args.backend != "hdemucs":
        raise NotImplementedError(
            "The RoFormer backend requires audio-separator; use --backend hdemucs "
            "when that dependency is unavailable"
        )
    bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model().to(device).eval()

    if args.prewarm:
        print(
            f'{{"rank": {rank}, "device": "{device}", "prewarm": true, '
            f'"cpu_threads": {_WORKER_CPU_THREADS}, '
            f'"default_cpu_threads": {_DEFAULT_CPU_THREADS}}}',
            flush=True,
        )
        return

    stem_root = Path(args.stem_dir or (Path(args.output_dir) / "stems"))
    stem_root.mkdir(parents=True, exist_ok=True)
    keep_stem = str(args.keep_stem).lower() not in {"false", "0", "no"}
    save_profile = str(args.save_activity_profile).lower() not in {"false", "0", "no"}


    made_dirs: set[Path] = set()

    def handle(record: dict[str, Any], writer: RankWriter) -> None:
        source = record["audio_path"]
        if not Path(source).exists():
            raise FileNotFoundError(source)

        wave = _load_audio(source, SEPARATE_SR)


        assert_duration_matches(
            source,
            declared_sec=record.get("duration_sec"),
            measured_sec=wave.shape[1] / SEPARATE_SR,
            where="separate",
        )
        vocals, accompaniment = _separate_chunked(
            model,
            wave,
            device,
            chunk_sec=args.chunk_sec,
            overlap_sec=args.overlap_sec,
        )

        mono = vocals.mean(dim=0)
        profile = _activity_profile(mono, accompaniment.mean(dim=0), SEPARATE_SR)
        intervals, ratio = _intervals_from_profile(
            profile,
            threshold_db=args.activity_threshold_db,
            min_sec=args.min_activity_sec,
        )


        safe_id = str(record["sample_id"]).replace(":", "_")


        target_dir = stem_root / safe_id[-4:-2] / safe_id[-2:]
        if target_dir not in made_dirs:
            target_dir.mkdir(parents=True, exist_ok=True)
            made_dirs.add(target_dir)
        stem_16k = target_dir / f"{safe_id}.vocal16k.wav"

        resampled = torchaudio.functional.resample(
            mono.unsqueeze(0), SEPARATE_SR, OUTPUT_SR
        )


        stem_rms_db = _rms_dbfs(mono)
        mix_rms_db = _rms_dbfs(wave.mean(dim=0))
        stem_to_mix_db = stem_rms_db - mix_rms_db

        resampled, applied_gain_db = _normalize(
            resampled,
            stem_peak_db=20.0 * math.log10(max(float(resampled.abs().max()), 1e-10)),
            mix_peak_db=20.0 * math.log10(max(float(wave.abs().max()), 1e-10)),
            target_dbfs=args.target_dbfs,
            reference=args.normalize_reference,
        )
        torchaudio.save(str(stem_16k), resampled, OUTPUT_SR, encoding="PCM_S", bits_per_sample=16)

        stem_full = ""
        if keep_stem:
            stem_full = str(target_dir / f"{safe_id}.vocal44k.flac")
            torchaudio.save(stem_full, vocals, SEPARATE_SR)

        payload = {
            "sample_id": record["sample_id"],
            "vocal_16k_path": str(stem_16k),
            "vocal_44k_path": stem_full,
            "vocal_intervals": [[round(a, 3), round(b, 3)] for a, b in intervals],
            "vocal_ratio": round(float(ratio), 4),


            "stem_to_mix_db": round(stem_to_mix_db, 2),
            "stem_rms_dbfs": round(stem_rms_db, 2),
            "mix_rms_dbfs": round(mix_rms_db, 2),
            "normalize_gain_db": round(applied_gain_db, 2),
            "activity_threshold_db": args.activity_threshold_db,
            "duration_sec": record.get("duration_sec", 0.0),
            "backend": args.backend,
        }
        if save_profile and profile is not None:


            payload["activity_db"] = [
                int(max(-60, min(20, round(v)))) for v in profile.tolist()
            ]
            payload["activity_fps"] = round(1.0 / FRAME_SEC, 3)
        writer.write(payload)
        del wave, vocals, accompaniment, mono
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run(args, rank=rank, world_size=world_size, handle=handle)


if __name__ == "__main__":
    main()
