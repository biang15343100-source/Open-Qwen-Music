
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .direct import flatten_numeric

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_KEY_VALUE = re.compile(r"(?:^|\s)([A-Za-z0-9_./-]+)=([^\s]+)")

SEMANTIC_KEYS = (
    "plain/loss_semantic_tokens",
    "section/loss_semantic_tokens",
    "unique_section/loss_semantic_tokens",
)
PITCH_KEYS = (
    "section/loss_melody_pitch",
    "unique_section/loss_melody_pitch",
)
STRUCT_KEYS = (
    "section/loss_melody_struct",
    "unique_section/loss_melody_struct",
)


@dataclass(frozen=True)
class CurvePhase:

    label: str
    step_offset: int
    log_path: Path
    evaluation_dir: Path


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(metrics: dict[str, float], keys: Sequence[str]) -> float | None:
    values = [metrics.get(key) for key in keys]
    if any(value is None for value in values):
        return None
    return sum(float(value) for value in values) / len(values)


def summarize_eval_metrics(metrics: dict[str, Any]) -> dict[str, float]:

    numeric = flatten_numeric(metrics)
    semantic = _mean(numeric, SEMANTIC_KEYS)
    pitch = _mean(numeric, PITCH_KEYS)
    structure = _mean(numeric, STRUCT_KEYS)
    if semantic is not None:
        numeric["balanced_semantic_nll"] = semantic
    if pitch is not None:
        numeric["balanced_melody_pitch_nll"] = pitch
    if structure is not None:
        numeric["balanced_melody_struct_nll"] = structure
    if semantic is not None and pitch is not None and structure is not None:
        numeric["ubnll"] = 0.80 * semantic + 0.15 * pitch + 0.05 * structure
    return numeric


def parse_training_log(
    path: str | Path,
    *,
    phase: str,
    step_offset: int = 0,
) -> list[dict[str, Any]]:

    source = Path(path)
    points: dict[int, dict[str, Any]] = {}
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = _ANSI_ESCAPE.sub("", raw_line).strip()
            marker = line.find("train step=")
            if marker < 0:
                continue
            line = line[marker:]
            values: dict[str, float] = {}
            for key, raw_value in _KEY_VALUE.findall(line):
                number = _finite_float(raw_value)
                if number is not None:
                    values[key] = number
            if "step" not in values:
                continue
            phase_step = int(values["step"])
            global_step = int(step_offset) + phase_step
            points[global_step] = {
                "global_step": global_step,
                "phase_step": phase_step,
                "phase": str(phase),
                "metrics": values,
            }
    return [points[step] for step in sorted(points)]


def load_evaluation_curve(
    directory: str | Path,
    *,
    phase: str,
    step_offset: int = 0,
) -> list[dict[str, Any]]:

    root = Path(directory)
    points: list[dict[str, Any]] = []
    for path in sorted(root.glob("step_*.json")):
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_metrics = dict(payload.get("metrics") or payload)
        phase_step = int(payload.get("step") or raw_metrics.get("checkpoint_step") or 0)
        points.append(
            {
                "global_step": int(step_offset) + phase_step,
                "phase_step": phase_step,
                "phase": str(phase),
                "source": str(path.resolve()),
                "metrics": summarize_eval_metrics(raw_metrics),
            }
        )
    return points


def _deduplicate(points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_step = {int(point["global_step"]): point for point in points}
    return [by_step[step] for step in sorted(by_step)]


def _apply_exposure_offsets(
    phase_points: Sequence[list[dict[str, Any]]],
) -> list[dict[str, Any]]:

    output: list[dict[str, Any]] = []
    previous_max = 0.0
    key = "cumulative_global_semantic_tokens"
    for points in phase_points:
        available = [
            float(point["metrics"][key])
            for point in points
            if key in point["metrics"]
        ]
        offset = previous_max if available and available[0] < previous_max else 0.0
        for point in points:
            copied = {
                **point,
                "metrics": dict(point["metrics"]),
            }
            if key in copied["metrics"]:
                copied["metrics"][key] = float(copied["metrics"][key]) + offset
                previous_max = max(previous_max, copied["metrics"][key])
            output.append(copied)
    return output


def build_curve_series(phases: Sequence[CurvePhase]) -> dict[str, Any]:

    train_by_phase: list[list[dict[str, Any]]] = []
    eval_by_phase: list[list[dict[str, Any]]] = []
    phase_metadata: list[dict[str, Any]] = []
    for phase in phases:
        train_points = parse_training_log(
            phase.log_path,
            phase=phase.label,
            step_offset=phase.step_offset,
        )
        eval_points = load_evaluation_curve(
            phase.evaluation_dir,
            phase=phase.label,
            step_offset=phase.step_offset,
        )
        train_by_phase.append(train_points)
        eval_by_phase.append(eval_points)
        phase_metadata.append(
            {
                "label": phase.label,
                "step_offset": int(phase.step_offset),
                "log_path": str(phase.log_path.resolve()),
                "evaluation_dir": str(phase.evaluation_dir.resolve()),
                "train_points": len(train_points),
                "eval_points": len(eval_points),
            }
        )
    return {
        "phases": phase_metadata,
        "train": _deduplicate(_apply_exposure_offsets(train_by_phase)),
        "evaluation": _deduplicate(_apply_exposure_offsets(eval_by_phase)),
    }


__all__ = [
    "CurvePhase",
    "build_curve_series",
    "load_evaluation_curve",
    "parse_training_log",
    "summarize_eval_metrics",
]
