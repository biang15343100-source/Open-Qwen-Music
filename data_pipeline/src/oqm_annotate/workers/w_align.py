#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _languages import to_name
from _limits import SAFE_ALIGN_SEC
from _worker import (
    RankWriter,
    add_common_arguments,
    distribution,
    resolve_model_dir,
    run,
)


#


def _usable_segments(
    record: dict[str, Any], *, text: str, duration: float
) -> tuple[list[dict[str, Any]], int]:

    segments: list[dict[str, Any]] = []
    oversize = 0
    for item in record.get("segments") or []:
        try:
            start = float(item["start"])
            end = float(item["end"])
            piece = str(item["text"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if not piece or end - start <= 0.05:
            continue
        if end - start > SAFE_ALIGN_SEC:
            oversize += 1
            continue
        segments.append({"start": start, "end": end, "text": piece})
    return segments, oversize


def _batched(items: list[dict[str, Any]], size: int):
    step = max(1, int(size))
    for start in range(0, len(items), step):
        yield items[start : start + step]


def range_violations(end_times: Any, clip_sec: float) -> tuple[int, float]:

    count = 0
    worst = 0.0
    for value in end_times:
        overshoot = float(value) - float(clip_sec)
        if overshoot > 1e-3:
            count += 1
            worst = max(worst, overshoot)
    return count, round(worst, 4)


#


#     if language == "japanese":  tokenize_japanese(text)
#     elif language == "korean":  tokenize_korean(text)
#     else:                       tokenize_space_lang(text)


#


#


_FALLBACK_LANGUAGE = "English"


def _language_name(code: str | None) -> str:

    return to_name(code) or _FALLBACK_LANGUAGE


def _monotonic_ratio(units: list[dict[str, Any]]) -> float:

    if len(units) < 2:
        return 1.0
    good = sum(
        1
        for previous, current in zip(units, units[1:])
        if current["start"] >= previous["start"] - 1e-3
    )
    return good / (len(units) - 1)


def max_backward_jump(units: list[dict[str, Any]]) -> float:

    worst = 0.0
    for previous, current in zip(units, units[1:]):
        worst = max(worst, float(previous["start"]) - float(current["start"]))
    return round(max(0.0, worst), 3)


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser())
    parser.add_argument("--model-dir", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    rank, local_rank, world_size = distribution()

    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    model_dir = resolve_model_dir(args.model_dir)
    from qwen_asr import Qwen3ForcedAligner

    model = Qwen3ForcedAligner.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=device
    )

    if args.prewarm:
        print(f'{{"rank": {rank}, "device": "{device}", "prewarm": true}}', flush=True)
        return

    import numpy

    from _audio import read_samples

    def handle(record: dict[str, Any], writer: RankWriter) -> None:
        audio_path = record["audio_path"]
        if not Path(audio_path).exists():
            raise FileNotFoundError(audio_path)


        language = _language_name(record.get("language"))
        used_fallback = to_name(record.get("language")) is None

        text = str(record.get("text") or "").strip()
        if not text:
            writer.write({"sample_id": record["sample_id"], "units": [], "skipped": "empty text"})
            return

        duration = float(record.get("duration_sec") or 0.0)
        segments, oversize = _usable_segments(record, text=text, duration=duration)
        if not segments:
            reason = (
                f"All {oversize} windows exceed {SAFE_ALIGN_SEC:.0f}s Safety upper limit"
                if oversize
                else "No windowing information available"
            )
            writer.write(
                {
                    "sample_id": record["sample_id"],
                    "units": [],
                    "oversize_segments": oversize,
                    "skipped": reason,
                }
            )
            return

        wave, sample_rate = read_samples(audio_path)
        if wave.ndim > 1:
            wave = wave.mean(axis=1)

        units: list[dict[str, Any]] = []
        failed_segments = 0


        out_of_range = 0
        max_overshoot = 0.0
        for batch in _batched(segments, args.batch_size):
            clips = [
                wave[
                    int(item["start"] * sample_rate) : int(item["end"] * sample_rate)
                ]
                for item in batch
            ]
            try:
                results = model.align(
                    audio=[(numpy.asarray(clip), sample_rate) for clip in clips],
                    text=[item["text"] for item in batch],
                    language=[language] * len(batch),
                )
            except Exception:  # noqa: BLE001


                failed_segments += len(batch)
                continue

            for item, aligned in zip(batch, results):
                offset = float(item["start"])
                kept = [
                    unit for unit in aligned if str(getattr(unit, "text", "")).strip()
                ]
                violations, worst = range_violations(
                    (getattr(unit, "end_time") for unit in kept),
                    float(item["end"]) - offset,
                )
                out_of_range += violations
                max_overshoot = max(max_overshoot, worst)
                for unit in kept:
                    unit_text = str(getattr(unit, "text", "")).strip()
                    units.append(
                        {
                            "text": unit_text,
                            "start": round(
                                offset + float(getattr(unit, "start_time")), 4
                            ),
                            "end": round(offset + float(getattr(unit, "end_time")), 4),
                        }
                    )


        monotonic = _monotonic_ratio(units)
        backward = max_backward_jump(units)
        units.sort(key=lambda unit: unit["start"])
        span = (units[-1]["end"] - units[0]["start"]) if units else 0.0
        writer.write(
            {
                "sample_id": record["sample_id"],
                "language": record.get("language"),


                "language_fallback": used_fallback,
                "units": units,
                "unit_count": len(units),
                "segment_count": len(segments),
                "failed_segments": failed_segments,


                "oversize_segments": oversize,


                "out_of_range_units": out_of_range,
                "max_overshoot_sec": round(max_overshoot, 4),
                "monotonic_ratio": round(monotonic, 4),


                "max_backward_jump_sec": backward,


                "span_ratio": round(span / duration, 4) if duration > 0 else 0.0,
                "model": args.model_dir,
            }
        )

    run(args, rank=rank, world_size=world_size, handle=handle)


if __name__ == "__main__":
    main()
