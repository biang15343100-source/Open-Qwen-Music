#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from _worker import (
    RankWriter,
    add_common_arguments,
    distribution,
    resolve_model_dir,
    run,
)

INPUT_SR = 24_000
_WINDOW_SAMPLES = 420 * INPUT_SR
_DEADLOCK_TAIL_SAMPLES = 1024

_FRAME_RATES = 8.333


def _shim_transformers_deepspeed() -> None:

    import importlib
    import sys
    import types

    try:
        importlib.import_module("transformers.deepspeed")
        return
    except Exception:  # noqa: BLE001 -
        pass

    try:
        from transformers.integrations import is_deepspeed_zero3_enabled
    except Exception:  # noqa: BLE001

        def is_deepspeed_zero3_enabled() -> bool:  # type: ignore[misc]
            return False

    module = types.ModuleType("transformers.deepspeed")
    module.is_deepspeed_zero3_enabled = is_deepspeed_zero3_enabled  # type: ignore[attr-defined]
    sys.modules["transformers.deepspeed"] = module


def _stub_msaf() -> None:

    import sys
    import types

    try:
        import msaf  # noqa: F401

        return
    except Exception:  # noqa: BLE001
        pass

    def compute_results(*_args: object, **_kwargs: object):
        raise NotImplementedError(
            "msaf is not installed. Install it to use the SongFormer backend."
        )

    package = types.ModuleType("msaf")
    package.__path__ = []  # type: ignore[attr-defined]
    evaluation = types.ModuleType("msaf.eval")
    evaluation.compute_results = compute_results  # type: ignore[attr-defined]
    package.eval = evaluation  # type: ignore[attr-defined]
    sys.modules["msaf"] = package
    sys.modules["msaf.eval"] = evaluation


def load_model(local_dir: str, device: str):
    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    _shim_transformers_deepspeed()
    _stub_msaf()

    if not local_dir or not Path(local_dir).exists():


        resolved = resolve_model_dir(local_dir or "ASLP-lab/SongFormer")
        if resolved and Path(resolved).exists():
            local_dir = resolved
        else:
            kwargs: dict[str, Any] = {
                "repo_id": local_dir or "ASLP-lab/SongFormer",
                "repo_type": "model",
                "ignore_patterns": ["SongFormer.pt", "SongFormer.safetensors"],
            }
            fallback = os.environ.get("HF_HUB_CACHE_FALLBACK")
            if fallback and Path(fallback).exists():
                try:
                    local_dir = snapshot_download(
                        **kwargs, cache_dir=fallback, local_files_only=True
                    )
                except Exception:  # noqa: BLE001
                    local_dir = snapshot_download(**kwargs)
            else:
                local_dir = snapshot_download(**kwargs)

    os.environ["SONGFORMER_LOCAL_DIR"] = local_dir
    if local_dir not in sys.path:
        sys.path.append(local_dir)
    model = AutoModel.from_pretrained(
        local_dir, trust_remote_code=True, low_cpu_mem_usage=False
    )
    return model.to(device).eval()


_LAST_STRENGTH: list[Any] = []


_PEAK_WINDOW_SEC: float | None = None


def _capture_boundary_strength() -> bool:

    try:
        import postprocessing.functional as functional
    except Exception:  # noqa: BLE001 -  transformers
        return False

    original = getattr(functional, "peak_picking", None)
    if original is None or getattr(original, "_oqm_wrapped", False):
        return original is not None

    def wrapper(*args: object, **kwargs: object):
        if _PEAK_WINDOW_SEC is not None:
            half = int(_PEAK_WINDOW_SEC * _FRAME_RATES)

            kwargs["window_past"] = half
            kwargs["window_future"] = half
        result = original(*args, **kwargs)
        _LAST_STRENGTH.clear()
        try:
            _LAST_STRENGTH.append(result.copy())
        except Exception:  # noqa: BLE001
            pass
        return result

    wrapper._oqm_wrapped = True  # type: ignore[attr-defined]
    functional.peak_picking = wrapper  # type: ignore[assignment]
    return True


def _forget_boundary_strength() -> None:

    _LAST_STRENGTH.clear()


def _boundary_strengths(sections: list[dict[str, Any]]) -> list[float | None]:

    if not _LAST_STRENGTH:
        return [None] * len(sections)
    strength = _LAST_STRENGTH[0]
    out: list[float | None] = []
    for section in sections:
        frame = int(round(float(section["start"]) * _FRAME_RATES))
        if 0 < frame < len(strength) and strength[frame] > 0:
            out.append(round(float(strength[frame]), 6))
        else:
            out.append(None)
    return out


def _trim_deadlock_tail(wave: Any) -> Any:

    tail = len(wave) % _WINDOW_SAMPLES
    if 0 < tail <= _DEADLOCK_TAIL_SAMPLES:
        return wave[: len(wave) - tail]
    return wave


def _safe_input(audio_path: str, *, declared_sec: float | None = None) -> Any:

    from _audio import DecodeError, assert_duration_matches, load_mono, probe

    try:
        measured = probe(audio_path)
    except DecodeError:


        return audio_path


    assert_duration_matches(
        audio_path,
        declared_sec=declared_sec,
        measured_sec=measured.duration_sec,
        where="structure",
    )

    if measured.duration_method == "ffmpeg_decode":
        return _trim_deadlock_tail(load_mono(audio_path, INPUT_SR)[0])


    samples = int(measured.duration_sec * INPUT_SR)
    if not 0 < samples % _WINDOW_SAMPLES <= _DEADLOCK_TAIL_SAMPLES:
        return audio_path

    return _trim_deadlock_tail(load_mono(audio_path, INPUT_SR)[0])


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser())
    parser.add_argument("--model-dir", default="ASLP-lab/SongFormer")
    parser.add_argument(
        "--no-rule-post-processing",
        action="store_true",
        help="Disable the upstream rule-based post-processing step.",
    )
    parser.add_argument(
        "--peak-window-sec",
        type=float,
        default=None,
        help=(
            "Half-window duration for peak suppression. Leave unset to use the model default."
        ),
    )
    args = parser.parse_args()
    if args.peak_window_sec is not None:
        global _PEAK_WINDOW_SEC
        _PEAK_WINDOW_SEC = args.peak_window_sec
    rank, local_rank, world_size = distribution()

    import torch

    device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

    model = load_model(args.model_dir, device)
    if args.no_rule_post_processing:
        model.config.no_rule_post_processing = True

    strength_available = _capture_boundary_strength()

    if args.prewarm:
        print(f'{{"rank": {rank}, "device": "{device}", "prewarm": true}}', flush=True)
        return

    def handle(record: dict[str, Any], writer: RankWriter) -> None:
        audio_path = record["audio_path"]
        if not Path(audio_path).exists():


            raise FileNotFoundError(audio_path)
        _forget_boundary_strength()


        sections = model(
            _safe_input(audio_path, declared_sec=record.get("duration_sec"))
        )
        strengths = _boundary_strengths(sections)
        writer.write(
            {
                "sample_id": record["sample_id"],
                "duration_sec": record.get("duration_sec", 0.0),
                "sections": [
                    {
                        "label": str(section["label"]),
                        "start": round(float(section["start"]), 4),
                        "end": round(float(section["end"]), 4),
                        "boundary_strength": strength,
                    }
                    for section, strength in zip(sections, strengths)
                ],
                "model": "songformer",
                "boundary_strength_available": strength_available,
            }
        )

    run(args, rank=rank, world_size=world_size, handle=handle)


if __name__ == "__main__":
    main()
