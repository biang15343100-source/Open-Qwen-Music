#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _languages import to_code, to_name
from _worker import (
    RankWriter,
    add_common_arguments,
    distribution,
    resolve_model_dir,
    run,
)

def _hint_language_name(hint: Any) -> str | None:

    return to_name(hint)


def _fixed_windows(start: float, end: float, *, window_sec: float, overlap_sec: float):

    if end - start <= window_sec:
        return [(start, end)]
    stride = max(1.0, window_sec - overlap_sec)
    spans: list[tuple[float, float]] = []
    position = start
    while position < end:
        stop = min(position + window_sec, end)
        spans.append((position, stop))
        if stop >= end:
            break
        position += stride
    return spans


def _windows(
    duration: float,
    intervals: list[tuple[float, float]],
    *,
    window_sec: float,
    overlap_sec: float,
    pad_sec: float = 0.3,
):


    #


    usable = [
        (min(max(0.0, start - pad_sec), duration), min(duration, end + pad_sec))
        for start, end in intervals
        if end > start
    ]
    usable = [(start, end) for start, end in usable if end > start]
    if not usable:


        return _fixed_windows(
            0.0, duration, window_sec=window_sec, overlap_sec=overlap_sec
        )

    usable.sort()
    merged: list[list[float]] = []
    for start, end in usable:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    spans: list[tuple[float, float]] = []
    current: list[float] | None = None
    for start, end in merged:
        if end - start > window_sec:
            if current is not None:
                spans.append((current[0], current[1]))
                current = None
            spans.extend(
                _quiet_windows(
                    start,
                    end,
                    intervals,
                    window_sec=window_sec,
                    overlap_sec=overlap_sec,
                )
            )
            continue


        if current is not None and end - current[0] <= window_sec:
            current[1] = end
        else:
            if current is not None:
                spans.append((current[0], current[1]))
            current = [start, end]
    if current is not None:
        spans.append((current[0], current[1]))


    #


    #


    #


    if spans and spans[-1][1] < duration:
        start = spans[-1][0]
        spans[-1] = (start, min(duration, start + window_sec))
    return spans


def _quiet_windows(
    block_start: float,
    block_end: float,
    intervals: list[tuple[float, float]],
    *,
    window_sec: float,
    overlap_sec: float,
) -> list[tuple[float, float]]:

    stride = window_sec - overlap_sec
    if stride <= 0.0:
        return _fixed_windows(
            block_start, block_end, window_sec=window_sec, overlap_sec=overlap_sec
        )


    cuts = sorted(
        (left_end, right_start)
        for (_, left_end), (right_start, _) in zip(intervals, intervals[1:])
        if right_start > left_end
        and block_start <= left_end
        and right_start <= block_end
    )

    spans: list[tuple[float, float]] = []
    piece_start = block_start
    while block_end - piece_start > window_sec:


        #


        candidates = [
            (gap_start, gap_end)
            for gap_start, gap_end in cuts
            if piece_start + overlap_sec <= gap_end <= piece_start + window_sec
            and block_end - gap_start >= overlap_sec
        ]
        if candidates:


            gap_start, gap_end = candidates[-1]
            spans.append((piece_start, gap_end))
            piece_start = gap_start
        else:

            spans.append((piece_start, piece_start + window_sec))
            piece_start += stride
    if block_end > piece_start:


        if spans and block_end - spans[-1][0] <= window_sec:
            spans[-1] = (spans[-1][0], block_end)
        else:
            spans.append((piece_start, block_end))
    return spans


_PUNCTUATION = set(
    " \t\n,.!?;:-\"'()"
    + "".join(
        chr(codepoint)
        for codepoint in (
            0xFF0C, 0x3002, 0xFF01, 0xFF1F, 0x3001, 0xFF1B, 0xFF1A,
            0x2026, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0xFF08, 0xFF09,
        )
    )
)

_MIN_OVERLAP_UNITS = 3


_OVERLAP_EPS = 0.05


def _strip_marks(text: str) -> tuple[str, list[int]]:

    kept: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        if char in _PUNCTUATION:
            continue
        kept.append(char)
        positions.append(index)
    return "".join(kept), positions


def _dedupe_overlap(previous: str, current: str, *, max_probe: int = 40) -> str:

    if not previous or not current:
        return current

    previous_bare, _ = _strip_marks(previous)
    current_bare, current_positions = _strip_marks(current)
    limit = min(max_probe, len(previous_bare), len(current_bare))
    for size in range(limit, _MIN_OVERLAP_UNITS - 1, -1):
        if previous_bare[-size:] == current_bare[:size]:

            if size >= len(current_positions):
                return ""
            return current[current_positions[size] :]
    return current


def repeat_length(previous: str, current: str, *, cap: int = 400) -> int:

    if not previous or not current:
        return 0
    previous_bare, _ = _strip_marks(previous)
    current_bare, _ = _strip_marks(current)
    limit = min(cap, len(previous_bare), len(current_bare))
    for size in range(limit, _MIN_OVERLAP_UNITS - 1, -1):
        if previous_bare[-size:] == current_bare[:size]:
            return size
    return 0


def _merge_pieces(
    spans: list[tuple[float, float]],
    pieces: list[str],
) -> tuple[str, list[dict[str, Any]], int]:

    merged = ""
    segments: list[dict[str, Any]] = []
    previous_end: float | None = None
    deduped = 0

    for (start, end), piece in zip(spans, pieces):
        overlapping = previous_end is not None and start < previous_end - _OVERLAP_EPS
        previous_end = end if previous_end is None else max(previous_end, end)

        if overlapping:
            addition = _dedupe_overlap(merged, piece)
            if addition != piece:
                deduped += 1
        else:
            addition = (piece or "").strip()
        if not addition:
            continue
        merged = f"{merged} {addition}".strip() if merged else addition


        if segments and start < segments[-1]["end"]:
            start = segments[-1]["end"]
        if end - start <= 0.05:


            if segments:
                segments[-1]["text"] = f"{segments[-1]['text']} {addition}".strip()
            continue
        segments.append(
            {"start": round(start, 3), "end": round(end, 3), "text": addition}
        )

    return merged, segments, deduped


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser())
    parser.add_argument("--model-dir", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--window-sec", type=float, default=28.0)
    parser.add_argument("--overlap-sec", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--context",
        default="This is a song. Transcribe the sung lyrics only.",
        help="Domain prompt passed to the ASR model.",
    )
    parser.add_argument(
        "--language-mode",
        choices=("hint", "auto"),
        default="hint",
        help="Use each record's language hint, or let the model detect language automatically.",
    )
    parser.add_argument(
        "--language",
        default="",
        help="Fallback language when a record has no language hint. Leave empty for detection.",
    )
    args = parser.parse_args()
    rank, local_rank, world_size = distribution()

    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    model_dir = resolve_model_dir(args.model_dir)
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map=device,
        max_inference_batch_size=args.batch_size,


        max_new_tokens=args.max_new_tokens,
    )


    import inspect

    transcribe_params = inspect.signature(model.transcribe).parameters
    context_key = next(
        (k for k in ("context", "prompt", "hotwords") if k in transcribe_params), None
    )
    if context_key is None and args.context:
        print(
            f'{{"rank": {rank}, "warn": "transcribe does not support realm prompts,ignored --context"}}',
            flush=True,
        )

    if args.prewarm:
        print(f'{{"rank": {rank}, "device": "{device}", "prewarm": true}}', flush=True)
        return

    import numpy as np

    from _audio import read_samples

    def handle(record: dict[str, Any], writer: RankWriter) -> None:
        audio_path = record["audio_path"]
        if not Path(audio_path).exists():
            raise FileNotFoundError(audio_path)

        wave, sample_rate = read_samples(audio_path)
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        duration = len(wave) / sample_rate

        intervals = [
            (float(a), float(b))
            for a, b in (record.get("vocal_intervals") or [])
        ]
        spans = _windows(
            duration,
            intervals,
            window_sec=args.window_sec,
            overlap_sec=args.overlap_sec,
        )
        chunks = [
            wave[int(start * sample_rate) : int(end * sample_rate)]
            for start, end in spans
        ]

        keep = [i for i, chunk in enumerate(chunks) if len(chunk) >= sample_rate * 0.3]
        if not keep:
            writer.write(
                {
                    "sample_id": record["sample_id"],
                    "text": "",
                    "language": None,
                    "duration_sec": duration,
                    "note": "no valid audio window",
                }
            )
            return


        forced_language = (
            _hint_language_name(record.get("language_hint") or args.language)
            if args.language_mode == "hint"
            else None
        )

        pieces: list[str] = []
        languages: list[str] = []
        for start in range(0, len(keep), args.batch_size):
            batch_indices = keep[start : start + args.batch_size]
            batch = [chunks[i] for i in batch_indices]
            kwargs: dict[str, Any] = {}
            if context_key and args.context:
                kwargs[context_key] = args.context
            results = model.transcribe(
                audio=[(np.asarray(c), sample_rate) for c in batch],
                language=[forced_language] * len(batch) if forced_language else None,
                **kwargs,
            )
            for item in results:
                text = getattr(item, "text", None)
                if text is None and isinstance(item, dict):
                    text = item.get("text", "")
                language = getattr(item, "language", None)
                if language is None and isinstance(item, dict):
                    language = item.get("language")
                pieces.append(str(text or "").strip())
                if language:
                    languages.append(str(language))

        merged, segments, deduped = _merge_pieces([spans[i] for i in keep], pieces)

        language_code = None
        if languages:


            language_code = max(set(languages), key=languages.count)


        residual = [
            repeat_length(a["text"], b["text"])
            for a, b in zip(segments, segments[1:])
        ]
        residual = [size for size in residual if size >= _MIN_OVERLAP_UNITS]

        writer.write(
            {
                "sample_id": record["sample_id"],
                "text": merged,
                "language": _to_code(language_code),
                "window_count": len(keep),


                "segments": segments,


                "deduped_windows": deduped,
                "residual_repeat_pairs": len(residual),
                "residual_repeat_chars": max(residual) if residual else 0,
                "forced_language": forced_language,
                "windowing": "vocal-activity" if intervals else "fixed",
                "duration_sec": round(duration, 3),
                "model": args.model_dir,
            }
        )

    run(args, rank=rank, world_size=world_size, handle=handle)


def _to_code(value: Any) -> str | None:

    return to_code(value)


if __name__ == "__main__":
    main()
