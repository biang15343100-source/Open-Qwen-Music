
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def paired_bootstrap_mean_ci(
    values: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {}
    if not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must all be finite numbers")
    resolved_weights = (
        np.ones(array.size, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)
    )
    if resolved_weights.shape != array.shape:
        raise ValueError("bootstrap weights and values different lengths")
    if np.any(~np.isfinite(resolved_weights)) or np.any(resolved_weights <= 0):
        raise ValueError("bootstrap weights must be a finite positive number")
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within(0,1)")
    resamples = int(resamples)
    if resamples <= 0:
        raise ValueError("resamples required > 0")

    mean = float(np.average(array, weights=resolved_weights))
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    sampled_values = array[indices]
    sampled_weights = resolved_weights[indices]
    estimates = (sampled_values * sampled_weights).sum(axis=1) / sampled_weights.sum(
        axis=1
    )
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    return {
        "mean": mean,
        "ci_low": float(low),
        "ci_high": float(high),
        "samples": float(array.size),
        "resamples": float(resamples),
        "confidence": confidence,
    }


def ctc_guardrail_decision(
    *,
    candidate_ci_low: float,
    baseline_mean: float,
    relative_limit: float = 1.02,
    feasible_samples: int,
    min_samples: int = 32,
) -> str:

    if int(feasible_samples) < int(min_samples):
        return "diagnostic_only"
    return (
        "fail"
        if float(candidate_ci_low)
        > float(baseline_mean) * float(relative_limit)
        else "pass"
    )


__all__ = ["ctc_guardrail_decision", "paired_bootstrap_mean_ci"]
