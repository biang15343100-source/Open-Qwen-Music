
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contracts import DISCARDED_QUALITY_BUCKET, QUALITY_BUCKETS, QUALITY_PERCENTILES


MIN_GENRE_SAMPLES_FOR_PERCENTILES = 100


POOLED_GENRE_KEY = "__pooled_small_genres__"


STRICT_PAPER_STAGE_SUPPORT: dict[str, frozenset[str]] = {
    "stage1": frozenset({"Q3", "Q4", "Q5", "Q6"}),
    "stage2": frozenset({"Q2"}),
    "stage3": frozenset({"Q1"}),
}


def bucket_percentile_ranges(
    percentiles: Sequence[float] = QUALITY_PERCENTILES,
) -> dict[str, tuple[float, float]]:
    if len(percentiles) != len(QUALITY_BUCKETS) - 1:
        raise ValueError(
            f"Requires {len(QUALITY_BUCKETS) - 1} percentile thresholds; got {len(percentiles)}"
        )
    edges = [100.0, *(float(p) for p in percentiles), 0.0]
    if any(edges[i] <= edges[i + 1] for i in range(len(edges) - 1)):
        raise ValueError(
            "Percentile thresholds must be strictly decreasing and lie within "
            f"(0, 100): {list(percentiles)}"
        )
    return {
        bucket: (edges[position + 1], edges[position])
        for position, bucket in enumerate(QUALITY_BUCKETS)
    }


def bucket_nominal_shares(
    percentiles: Sequence[float] = QUALITY_PERCENTILES,
) -> dict[str, float]:
    return {
        bucket: (high - low) / 100.0
        for bucket, (low, high) in bucket_percentile_ranges(percentiles).items()
    }


def assign_quality_buckets(
    scores: Sequence[float],
    genres: Sequence[str] | None = None,
    *,
    percentiles: Sequence[float] = QUALITY_PERCENTILES,
    min_genre_samples: int = MIN_GENRE_SAMPLES_FOR_PERCENTILES,
) -> list[str]:
    scores_array = np.asarray(scores, dtype=np.float64)
    if scores_array.ndim != 1:
        raise ValueError(f"scores must be one-dimensional; got {scores_array.shape}")
    if genres is not None and len(genres) != scores_array.size:
        raise ValueError("genres and scores length must be consistent")
    if len(percentiles) != len(QUALITY_BUCKETS) - 1:
        raise ValueError(
            f"Requires {len(QUALITY_BUCKETS) - 1} percentile thresholds; got {len(percentiles)}"
        )

    bucket_percentile_ranges(percentiles)

    buckets = ["Q7"] * scores_array.size
    groups: dict[str, list[int]] = defaultdict(list)
    if genres is None:
        groups["__all__"] = list(range(scores_array.size))
    else:
        for index, genre in enumerate(genres):
            groups[str(genre) if genre else "unknown"].append(index)
        groups = _pool_small_groups(groups, min_genre_samples=int(min_genre_samples))

    for indices in groups.values():
        if not indices:
            continue
        group_scores = scores_array[indices]
        thresholds = np.percentile(group_scores, list(percentiles))
        for position, score in zip(indices, group_scores):
            label = QUALITY_BUCKETS[-1]
            for bucket_index, threshold in enumerate(thresholds):
                if score >= threshold:
                    label = QUALITY_BUCKETS[bucket_index]
                    break
            buckets[position] = label
    return buckets


def assign_quality_buckets_ranked(
    scores: Sequence[float],
    genres: Sequence[str],
    sample_ids: Sequence[str],
    *,
    percentiles: Sequence[float] = QUALITY_PERCENTILES,
) -> list[str]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(genres) != values.size or len(sample_ids) != values.size:
        raise ValueError("scores, genres, and sample_ids must be corresponding one-dimensional sequences")
    if not np.all(np.isfinite(values)):
        raise ValueError("quality scores must all be finite")
    if len(set(str(sample_id) for sample_id in sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique so quality-score ties can be broken deterministically")
    ranges = bucket_percentile_ranges(percentiles)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, genre in enumerate(genres):
        key = str(genre).strip()
        if not key or key.lower() == "unknown":
            raise ValueError(f"primary genre is invalid: {genre!r}")
        groups[key].append(index)
    output = [""] * values.size
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: (float(values[index]), str(sample_ids[index])))
        size = len(ordered)
        for rank, index in enumerate(ordered):
            percentile = 100.0 * (rank + 0.5) / size
            label = QUALITY_BUCKETS[-1]
            for bucket in QUALITY_BUCKETS:
                low, high = ranges[bucket]
                if low <= percentile < high:
                    label = bucket
                    break
            output[index] = label
    return output


def _pool_small_groups(
    groups: dict[str, list[int]], *, min_genre_samples: int
) -> dict[str, list[int]]:
    if min_genre_samples <= 1:
        return groups
    pooled: list[int] = []
    kept: dict[str, list[int]] = {}
    for genre, indices in groups.items():
        if len(indices) < min_genre_samples:
            pooled.extend(indices)
        else:
            kept[genre] = indices
    if pooled:
        kept[POOLED_GENRE_KEY] = sorted(pooled)
    return kept


def small_genres(
    genres: Sequence[str], *, min_genre_samples: int = MIN_GENRE_SAMPLES_FOR_PERCENTILES
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for genre in genres:
        counts[str(genre) if genre else "unknown"] += 1
    return {
        genre: count for genre, count in sorted(counts.items()) if count < min_genre_samples
    }


@dataclass
class QualityBucketReport:

    counts: dict[str, int]
    shares: dict[str, float]
    nominal_shares: dict[str, float]
    max_absolute_deviation: float
    warnings: list[str] = field(default_factory=list)


def quality_bucket_report(
    buckets: Sequence[str],
    *,
    percentiles: Sequence[float] = QUALITY_PERCENTILES,
    tolerance: float = 0.02,
) -> QualityBucketReport:
    nominal = bucket_nominal_shares(percentiles)
    counts = {bucket: 0 for bucket in QUALITY_BUCKETS}
    for bucket in buckets:
        if bucket not in counts:
            raise ValueError(f"Unknown quality bucket {bucket!r}; expected {QUALITY_BUCKETS}")
        counts[bucket] += 1
    total = sum(counts.values())
    shares = {
        bucket: (count / total if total else 0.0) for bucket, count in counts.items()
    }
    warnings: list[str] = []
    worst = 0.0
    for bucket in QUALITY_BUCKETS:
        deviation = abs(shares[bucket] - nominal[bucket])
        worst = max(worst, deviation)
        if deviation > tolerance:
            warnings.append(
                f"{bucket} observed share {shares[bucket]:.4f} differs from target "
                f"{nominal[bucket]:.4f} by {deviation:.4f}; check tied MOS scores "
                "or undersized genre groups"
            )
    if total and counts[DISCARDED_QUALITY_BUCKET] == 0:
        warnings.append(
            f"No samples were assigned to {DISCARDED_QUALITY_BUCKET}; the lowest "
            "percentile may be empty when every quality group has fewer than 100 samples"
        )
    return QualityBucketReport(
        counts=counts,
        shares=shares,
        nominal_shares=nominal,
        max_absolute_deviation=worst,
        warnings=warnings,
    )


def keep_for_training(buckets: Iterable[str]) -> list[bool]:
    return [bucket != DISCARDED_QUALITY_BUCKET for bucket in buckets]


def bucket_level(bucket: str) -> int:
    return QUALITY_BUCKETS.index(bucket) + 1


def mean_quality_level(weights: Mapping[str, float]) -> float:
    total = float(sum(max(0.0, float(value)) for value in weights.values()))
    if total <= 0:
        raise ValueError(f"The sum of weights must be positive: {dict(weights)}")
    return sum(
        bucket_level(bucket) * max(0.0, float(value)) for bucket, value in weights.items()
    ) / total


class QualitySchedule:

    def __init__(self, keyframes: Sequence[tuple[int, dict[str, float]]]) -> None:
        if not keyframes:
            raise ValueError("Quality scheduling requires at least one keyframe")
        ordered = [(int(step), dict(weights)) for step, weights in keyframes]
        steps = [step for step, _ in ordered]
        if len(set(steps)) != len(steps):
            raise ValueError(f"keyframe step is duplicated: {steps}")
        if steps != sorted(steps):


            raise ValueError(f"keyframe steps must be increasing; received {steps}")
        buckets: list[str] = []
        for _, weights in ordered:
            for bucket in weights:
                if bucket not in buckets:
                    buckets.append(bucket)
        unknown = set(buckets) - set(QUALITY_BUCKETS)
        if unknown:
            raise ValueError(
                f"Unknown quality bucket: {sorted(unknown)} (expected {QUALITY_BUCKETS})"
            )
        if DISCARDED_QUALITY_BUCKET in buckets:
            raise ValueError(
                f"{DISCARDED_QUALITY_BUCKET} is discarded and cannot have a sampling weight"
            )
        self.buckets = tuple(buckets)
        self.keyframes = [
            (step, self._normalize(weights)) for step, weights in ordered
        ]


        self.stage: str | None = None
        self.strict_paper = False

    def _normalize(self, weights: dict[str, float]) -> dict[str, float]:
        values = {bucket: max(0.0, float(weights.get(bucket, 0.0))) for bucket in self.buckets}
        total = sum(values.values())
        if total <= 0:
            raise ValueError(f"The sum of keyframe weights must be positive; got {weights}")
        return {bucket: value / total for bucket, value in values.items()}

    def weights_at(self, step: int) -> dict[str, float]:
        step = int(step)
        if len(self.keyframes) == 1 or step <= self.keyframes[0][0]:
            return dict(self.keyframes[0][1])
        if step >= self.keyframes[-1][0]:
            return dict(self.keyframes[-1][1])
        for index in range(1, len(self.keyframes)):
            left_step, left = self.keyframes[index - 1]
            right_step, right = self.keyframes[index]
            if step <= right_step:
                span = max(1, right_step - left_step)
                alpha = (step - left_step) / float(span)
                return {
                    bucket: (1.0 - alpha) * left[bucket] + alpha * right[bucket]
                    for bucket in self.buckets
                }
        return dict(self.keyframes[-1][1])

    def mean_quality_level_at(self, step: int) -> float:
        return mean_quality_level(self.weights_at(step))

    def support_at(self, step: int) -> frozenset[str]:
        return frozenset(
            bucket for bucket, weight in self.weights_at(step).items() if weight > 0
        )

    @property
    def positive_support(self) -> frozenset[str]:
        return frozenset(
            bucket
            for _, weights in self.keyframes
            for bucket, weight in weights.items()
            if weight > 0
        )

    def is_monotonically_improving(self, *, tolerance: float = 1e-9) -> bool:
        levels = [mean_quality_level(weights) for _, weights in self.keyframes]
        return all(
            later <= earlier + tolerance for earlier, later in zip(levels, levels[1:])
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> QualitySchedule:
        section = dict(config.get("curriculum", {}) or {})
        raw = section.get("schedule")
        if not raw:

            buckets = [b for b in QUALITY_BUCKETS if b != DISCARDED_QUALITY_BUCKET]
            schedule = cls([(0, {bucket: 1.0 for bucket in buckets})])
        else:
            keyframes: list[tuple[int, dict[str, float]]] = []
            for entry in raw:
                if "weights" not in entry:
                    raise KeyError(f"curriculum.schedule keyframe is missing weights: {entry}")
                keyframes.append((int(entry.get("step", 0)), dict(entry["weights"])))
            schedule = cls(keyframes)

        sampler = dict(config.get("sampler", {}) or {})
        schedule.stage = str(config.get("stage", "")).strip().lower() or None
        schedule.strict_paper = bool(sampler.get("strict_paper", False))
        if schedule.strict_paper:
            _validate_strict_paper_stage(config, schedule)
        return schedule


def _validate_strict_paper_stage(
    config: Mapping[str, Any], schedule: QualitySchedule
) -> None:
    stage = schedule.stage
    expected = STRICT_PAPER_STAGE_SUPPORT.get(stage or "")


    if expected is None:
        return

    for step, weights in schedule.keyframes:
        support = frozenset(bucket for bucket, weight in weights.items() if weight > 0)
        if support != expected:
            raise ValueError(
                f"strict_paper {stage} in step={step} must be exactly "
                f"{sorted(expected)}; got {sorted(support)}"
            )

    if stage == "stage1" and not schedule.is_monotonically_improving():
        raise ValueError("strict_paper stage1 quality weights must improve monotonically with step")

    sampler = dict(config.get("sampler", {}) or {})
    promotion = dict(sampler.get("promotion", {}) or {})
    enabled = bool(promotion.get("enabled", False))
    ratio = float(promotion.get("ratio", 0.0))
    if stage == "stage3":
        if not enabled or not 0 < ratio < 1:
            raise ValueError(
                "strict_paper stage3 promotion.ratio must be in (0, 1) so both "
                "Q1 and full-corpus sampling remain enabled"
            )
        language = dict(sampler.get("language", {}) or {})
        genre = dict(sampler.get("genre", {}) or {})
        if not bool(language.get("balance", True)) or not bool(
            genre.get("balance", False)
        ):
            raise ValueError(
                "strict_paper stage3 must enable language.balance and genre.balance "
                "to keep promotion jointly auditable"
            )
    elif enabled or ratio > 0:
        raise ValueError(f"strict_paper {stage} cannot enable promotion; promotion belongs only to stage3")


def temperature_weights(
    counts: Mapping[str, int], *, temperature: float = 0.5, balance: bool = True
) -> dict[str, float]:
    if not counts:
        return {}
    if not balance:
        total = float(sum(counts.values())) or 1.0
        return {key: count / total for key, count in counts.items()}
    if not 0.0 <= temperature <= 1.0:
        raise ValueError(f"temperature must be within [0, 1]; got {temperature}")
    raw = {key: float(max(count, 0)) ** temperature for key, count in counts.items()}
    total = sum(raw.values())
    if total <= 0:
        return {key: 1.0 / len(counts) for key in counts}
    return {key: value / total for key, value in raw.items()}


def language_weights(
    counts: dict[str, int], *, temperature: float = 0.5, balance: bool = True
) -> dict[str, float]:
    return temperature_weights(counts, temperature=temperature, balance=balance)


def genre_weights(
    counts: dict[str, int], *, temperature: float = 0.5, balance: bool = True
) -> dict[str, float]:
    return temperature_weights(counts, temperature=temperature, balance=balance)


def joint_balance_weights(
    cell_counts: Mapping[tuple[str, str], int],
    *,
    row_targets: Mapping[str, float],
    col_targets: Mapping[str, float] | None = None,
    iterations: int = 64,
    tolerance: float = 1e-9,
) -> dict[tuple[str, str], float]:
    cells = {key: float(count) for key, count in cell_counts.items() if count > 0}
    if not cells:
        return {}
    rows = sorted({key[0] for key in cells})
    cols = sorted({key[1] for key in cells})
    row_target = _restricted_targets(row_targets, rows)

    if col_targets is None:
        row_totals: dict[str, float] = defaultdict(float)
        for (row, _), count in cells.items():
            row_totals[row] += count
        return {
            (row, col): count / row_totals[row] * row_target[row]
            for (row, col), count in cells.items()
        }

    col_target = _restricted_targets(col_targets, cols)
    row_index = {row: position for position, row in enumerate(rows)}
    col_index = {col: position for position, col in enumerate(cols)}
    matrix = np.zeros((len(rows), len(cols)), dtype=np.float64)
    for (row, col), count in cells.items():
        matrix[row_index[row], col_index[col]] = count
    matrix /= matrix.sum()
    row_vector = np.asarray([row_target[row] for row in rows], dtype=np.float64)
    col_vector = np.asarray([col_target[col] for col in cols], dtype=np.float64)

    for _ in range(max(1, int(iterations))):
        col_sums = matrix.sum(axis=0)
        scale = np.divide(
            col_vector, col_sums, out=np.ones_like(col_vector), where=col_sums > 0
        )
        matrix *= scale[None, :]
        row_sums = matrix.sum(axis=1)
        scale = np.divide(
            row_vector, row_sums, out=np.ones_like(row_vector), where=row_sums > 0
        )
        matrix *= scale[:, None]
        if float(np.abs(matrix.sum(axis=0) - col_vector).max()) <= tolerance:
            break

    total = matrix.sum()
    if total <= 0:  # pragma: no cover -  0 ,
        raise ValueError("IPF degenerates into an all-zero matrix")
    matrix /= total
    return {
        (row, col): float(matrix[row_index[row], col_index[col]]) for (row, col) in cells
    }


def _restricted_targets(
    targets: Mapping[str, float], keys: Sequence[str]
) -> dict[str, float]:
    restricted = {key: max(0.0, float(targets.get(key, 0.0))) for key in keys}
    total = sum(restricted.values())
    if total <= 0:


        return {key: 1.0 / len(keys) for key in keys}
    return {key: value / total for key, value in restricted.items()}


def marginal_distribution(
    cell_weights: Mapping[tuple[str, str], float], *, axis: int
) -> dict[str, float]:
    marginal: dict[str, float] = defaultdict(float)
    for key, weight in cell_weights.items():
        marginal[key[axis]] += float(weight)
    return dict(marginal)


def total_variation(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(
        abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys
    )
