
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from torch.utils.data import Sampler

from ..contracts import DISCARDED_QUALITY_BUCKET, QUALITY_BUCKETS, SequenceMode
from ..curriculum import (
    QualitySchedule,
    genre_weights,
    joint_balance_weights,
    language_weights,
    total_variation,
)
from .index import CONDITION_TOKENS_UNKNOWN, ManifestIndex
from .promotion import resolve_promotion_pool


MODE_RATIO_SCOPES = ("all", "melodic")


DISTRIBUTION_TOLERANCE = 0.01


TRAINING_QUALITY_BUCKETS = frozenset(
    bucket for bucket in QUALITY_BUCKETS if bucket != DISCARDED_QUALITY_BUCKET
)


class StepCounter:

    __slots__ = ("value",)

    def __init__(self, value: int = 0) -> None:
        self.value = int(value)

    def set(self, value: int) -> None:
        self.value = int(value)


def _normalize_ratios(ratios: dict[str, float]) -> dict[SequenceMode, float]:
    if not ratios:
        return {SequenceMode.PLAIN: 1.0}
    parsed: dict[SequenceMode, float] = {}
    for name, weight in ratios.items():
        weight = float(weight)
        if weight < 0:
            raise ValueError(f"Mode ratios cannot be negative: {name}={weight}")
        if weight == 0:
            continue
        parsed[SequenceMode(name)] = weight
    total = sum(parsed.values())
    if total <= 0:
        raise ValueError("The sum of mode ratios must be positive")
    return {mode: weight / total for mode, weight in parsed.items()}


def _normalized(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        return {}
    return {key: value / total for key, value in weights.items()}


def _normalize_modes(ratios: dict[SequenceMode, float]) -> dict[SequenceMode, float]:
    total = float(sum(ratios.values()))
    return {mode: weight / total for mode, weight in ratios.items()}


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return int(value)
    return int(-(-int(value) // int(multiple)) * int(multiple))


_CellKey = tuple[str, str]  # (language, genre)
_PoolKey = tuple[bool, str, str, str]  # (instrumental, bucket, language, genre)


@dataclass
class SamplingDiagnostics:

    step: int
    strict_paper: bool
    target_buckets: dict[str, float]
    effective_buckets: dict[str, float]
    bucket_deviation: float
    target_languages: dict[str, float]
    effective_languages: dict[str, float]
    language_deviation: float
    target_genres: dict[str, float] | None
    effective_genres: dict[str, float]
    genre_deviation: float | None
    target_instrumental: float
    effective_instrumental: float
    configured_modes: dict[str, float]
    expected_modes: dict[str, float]
    mode_deviation: float
    mode_ratio_scope: str
    forced_plain_probability: float
    melodic_vocal_plain_floor: float
    melodic_vocal_plain_probability: float
    configured_promotion_ratio: float
    effective_promotion_ratio: float
    promotion_pool_size: int

    condition_tokens_coverage: float = 1.0

    length_estimate_is_upper_bound: bool = True
    warnings: list[str] = field(default_factory=list)

    def as_flat_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "strict_paper": float(self.strict_paper),
            "bucket_tv": self.bucket_deviation,
            "language_tv": self.language_deviation,
            "genre_tv": -1.0 if self.genre_deviation is None else self.genre_deviation,
            "mode_tv": self.mode_deviation,
            "instrumental_target": self.target_instrumental,
            "instrumental_effective": self.effective_instrumental,
            "promotion_ratio": self.effective_promotion_ratio,
            "promotion_pool": self.promotion_pool_size,
            "forced_plain_p": self.forced_plain_probability,
            "melodic_vocal_plain_floor": self.melodic_vocal_plain_floor,
            "melodic_vocal_plain_p": self.melodic_vocal_plain_probability,
            "condition_len_coverage": self.condition_tokens_coverage,
            "length_upper_bound": float(self.length_estimate_is_upper_bound),
            "warnings": len(self.warnings),
        }


class CurriculumBatchSampler(Sampler[list[tuple[int, str, int]]]):

    def __init__(
        self,
        index: ManifestIndex,
        record_indices: np.ndarray,
        *,
        rank: int,
        world_size: int,
        seed: int,
        max_tokens_per_gpu: int,
        max_sequence_length: int,
        max_semantic_frames: int,
        quality_schedule: QualitySchedule,
        mode_ratios: dict[str, float],
        instrumental_ratio: float = 0.0,
        language_balance: bool = True,
        language_temperature: float = 0.5,
        genre_balance: bool = False,
        genre_temperature: float = 0.5,
        mode_ratio_scope: str = "all",
        melodic_vocal_plain_floor: float = 0.05,
        strict_paper: bool = False,
        promoted_indices: np.ndarray | None = None,
        promotion_ratio: float = 0.0,
        condition_token_estimate: int = 256,
        max_condition_tokens: int | None = None,
        use_indexed_condition_tokens: bool = True,
        pad_to_multiple_of: int = 64,
        candidate_pool_batches: int = 16,
        max_batch_size: int = 64,
        step_counter: StepCounter | None = None,
        drop_last_partial: bool = False,
        warn_on_deviation: bool = True,
        semantic_section_reanchor: bool = False,
        semantic_anchor_max_text_tokens: int = 256,
        semantic_anchor_length_fn: Callable[[int, SequenceMode], int]
        | None = None,
    ) -> None:
        self.index = index
        self.record_indices = np.asarray(record_indices, dtype=np.int64)
        if self.record_indices.size == 0:
            raise ValueError("No sample available")
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.seed = int(seed)
        self.max_tokens_per_gpu = int(max_tokens_per_gpu)
        self.max_sequence_length = int(max_sequence_length)
        self.max_semantic_frames = int(max_semantic_frames)
        self.semantic_section_reanchor = bool(semantic_section_reanchor)
        self.semantic_anchor_max_text_tokens = int(
            semantic_anchor_max_text_tokens
        )
        self.semantic_anchor_length_fn = semantic_anchor_length_fn
        self.quality_schedule = quality_schedule
        self.strict_paper = bool(strict_paper)
        self.mode_ratios = _normalize_ratios(mode_ratios)
        self.instrumental_ratio = float(instrumental_ratio)
        if not 0.0 <= self.instrumental_ratio <= 1.0:
            raise ValueError(
                f"instrumental_ratio must be in [0, 1], received {self.instrumental_ratio}"
            )
        self.language_balance = bool(language_balance)
        self.language_temperature = float(language_temperature)
        self.genre_balance = bool(genre_balance)
        self.genre_temperature = float(genre_temperature)
        if mode_ratio_scope not in MODE_RATIO_SCOPES:
            raise ValueError(
                f"mode_ratio_scope must be one of {MODE_RATIO_SCOPES}, "
                f"received {mode_ratio_scope!r}"
            )
        self.mode_ratio_scope = str(mode_ratio_scope)
        self.melodic_vocal_plain_floor = float(melodic_vocal_plain_floor)
        if not 0.0 <= self.melodic_vocal_plain_floor <= 1.0:
            raise ValueError(
                "melodic_vocal_plain_floor must be in [0, 1], received "
                f"{self.melodic_vocal_plain_floor}"
            )
        self.promotion_ratio = float(promotion_ratio)
        if not 0.0 <= self.promotion_ratio <= 1.0:
            raise ValueError(
                f"promotion_ratio must be in [0, 1], received {self.promotion_ratio}"
            )
        self.condition_token_estimate = int(condition_token_estimate)
        self.max_condition_tokens = int(
            self.condition_token_estimate if max_condition_tokens is None
            else max_condition_tokens
        )
        self.pad_to_multiple_of = max(1, int(pad_to_multiple_of))
        self.candidate_pool_batches = max(1, int(candidate_pool_batches))
        self.max_batch_size = max(1, int(max_batch_size))
        self.step_counter = step_counter or StepCounter()
        if drop_last_partial:
            raise ValueError(
                "drop_last_partial=True is unsupported because it would systematically "
                "discard the longest batch after sorting"
            )
        self.epoch = 0

        self._validate_training_records()


        self.use_indexed_condition_tokens = bool(use_indexed_condition_tokens)
        self._condition_tokens = (
            np.asarray(self.index.condition_tokens, dtype=np.int64)
            if self.use_indexed_condition_tokens
            else np.full(self.index.records, CONDITION_TOKENS_UNKNOWN, dtype=np.int64)
        )
        self._num_sections = np.asarray(self.index.num_sections, dtype=np.int64)
        self.condition_tokens_coverage = float(
            (self._condition_tokens[self.record_indices] >= 0).mean()
        )


        self._length_ceiling = _round_up(self.max_sequence_length, self.pad_to_multiple_of)
        if self.max_tokens_per_gpu < self._length_ceiling:
            raise ValueError(
                f"max_tokens_per_gpu={self.max_tokens_per_gpu} is below the padded sequence "
                f"limit {self._length_ceiling} (max_sequence_length={self.max_sequence_length}, "
                f"pad_to_multiple_of={self.pad_to_multiple_of}); the longest sample cannot "
                "fit in a batch"
            )

        self._bucket_weight_cache: dict[tuple[bool, int], tuple[list[str], np.ndarray]] = {}
        self._rng = np.random.default_rng(self.seed + 7919 * self.rank)
        self._pending_batches: list[list[tuple[int, str, int]]] = []
        self._build_pools(promoted_indices)
        self._build_mode_plan()
        if self.strict_paper:
            self._validate_strict_pool_support()
            self._validate_strict_joint_balance()
        if warn_on_deviation and self.rank == 0:
            self._warn_on_deviation()


    def _validate_training_records(self) -> None:
        unique = np.unique(self.record_indices)
        if unique.size and (int(unique[0]) < 0 or int(unique[-1]) >= self.index.records):
            raise IndexError(
                f"record_indices are out of bounds: expected [0, {self.index.records}), "
                f"received [{int(unique[0])}, {int(unique[-1])}]"
            )

        def label(record: dict[str, Any]) -> str:
            return str(record.get("sample_id", "<missing-id>"))

        def require_text(value: Any, *, field_name: str, record: dict[str, Any]) -> str:
            if (
                not isinstance(value, str)
                or not value.strip()
                or value.strip().casefold() == "unknown"
            ):
                raise ValueError(
                    f"Training sample {label(record)!r} has missing or unknown "
                    f"{field_name}: {value!r}"
                )
            return value

        with self.index.manifest_path.open("rb") as handle:
            for record_index in unique.tolist():
                handle.seek(int(self.index.offsets[record_index]))
                record = json.loads(handle.readline())
                quality = record.get("quality")
                if not isinstance(quality, dict):
                    raise ValueError(
                        f"Training sample {label(record)!r} has no valid quality object"
                    )
                bucket = quality.get("bucket")
                if bucket == DISCARDED_QUALITY_BUCKET:
                    raise ValueError(
                        f"training sample {label(record)!r} falls into the discarded "
                        f"{DISCARDED_QUALITY_BUCKET} and cannot enter the training pool"
                    )
                if not isinstance(bucket, str) or bucket not in TRAINING_QUALITY_BUCKETS:
                    raise ValueError(
                        f"Training sample {label(record)!r} has invalid quality bucket "
                        f"{bucket!r}; expected one of {sorted(TRAINING_QUALITY_BUCKETS)}"
                    )
                require_text(
                    quality.get("genre"), field_name="quality.genre", record=record
                )
                require_text(record.get("language"), field_name="language", record=record)
                instrumental = record.get("is_instrumental")
                if type(instrumental) is not bool:
                    raise ValueError(
                        f"Training sample {label(record)!r} is_instrumental must be a JSON "
                        f"boolean, received {instrumental!r} ({type(instrumental).__name__})"
                    )

    def _build_pools(self, promoted_indices: np.ndarray | None) -> None:


        instrumental_code = self.index.code_of("is_instrumental", "true")
        codes = self.index.codes("is_instrumental")
        self._instrumental_flags = (
            np.zeros(codes.size, dtype=bool)
            if instrumental_code is None
            else codes == np.uint16(instrumental_code)
        )
        bucket_table = self.index.values("quality.bucket")
        language_table = self.index.values("language")
        genre_table = self.index.values("quality.genre")
        bucket_codes = self.index.codes("quality.bucket")[self.record_indices]
        language_codes = self.index.codes("language")[self.record_indices]
        genre_codes = self.index.codes("quality.genre")[self.record_indices]
        instrumental = self._instrumental_flags[self.record_indices]

        pools: dict[_PoolKey, list[int]] = defaultdict(list)
        for position, record_index in enumerate(self.record_indices):
            bucket = bucket_table[int(bucket_codes[position])]
            key = (
                bool(instrumental[position]),
                bucket,
                language_table[int(language_codes[position])],
                genre_table[int(genre_codes[position])],
            )
            pools[key].append(int(record_index))
        self.pools = {key: np.asarray(value, dtype=np.int64) for key, value in pools.items()}

        self.available_buckets = sorted({key[1] for key in self.pools})
        language_counts: dict[str, int] = defaultdict(int)
        genre_counts: dict[str, int] = defaultdict(int)
        for key, value in self.pools.items():
            language_counts[key[2]] += int(value.size)
            genre_counts[key[3]] += int(value.size)
        self.language_counts = dict(language_counts)
        self.genre_counts = dict(genre_counts)
        self.language_weights = language_weights(
            self.language_counts,
            temperature=self.language_temperature,
            balance=self.language_balance,
        )


        self.genre_weights = (
            genre_weights(self.genre_counts, temperature=self.genre_temperature)
            if self.genre_balance
            else None
        )
        self.has_instrumental = any(key[0] for key in self.pools)
        self.has_vocal = any(not key[0] for key in self.pools)
        self._all_indices = self.record_indices


        grouped: dict[tuple[bool, str], dict[_CellKey, int]] = defaultdict(dict)
        for key, value in self.pools.items():
            grouped[(key[0], key[1])][(key[2], key[3])] = int(value.size)
        self._cell_choice: dict[tuple[bool, str], tuple[list[_CellKey], np.ndarray]] = {}
        for group_key, cell_counts in grouped.items():
            weights = joint_balance_weights(
                cell_counts,
                row_targets=self.language_weights,
                col_targets=self.genre_weights,
            )
            cells = sorted(weights)
            probs = np.asarray([weights[cell] for cell in cells], dtype=np.float64)
            total = probs.sum()
            if total <= 0:  # pragma: no cover - joint_balance_weights
                probs = np.full(len(cells), 1.0 / len(cells))
            else:
                probs = probs / total
            self._cell_choice[group_key] = (cells, probs)

        self._buckets_by_flavor: dict[bool, list[str]] = {}
        for flavor in (False, True):
            self._buckets_by_flavor[flavor] = sorted(
                bucket for (instr, bucket) in self._cell_choice if instr == flavor
            )

        self._build_promotion_pools(promoted_indices)

    def _build_promotion_pools(self, promoted_indices: np.ndarray | None) -> None:
        self.promotion_pools: dict[bool, np.ndarray] = {}
        self.promotion_excluded_discarded = 0
        if promoted_indices is None:
            self.promoted_indices = np.empty(0, dtype=np.int64)
        else:
            wanted = np.asarray(promoted_indices, dtype=np.int64)
            in_split = np.intersect1d(wanted, self.record_indices, assume_unique=False)
            bucket_table = self.index.values("quality.bucket")
            bucket_codes = self.index.codes("quality.bucket")
            keep: list[int] = []
            for record_index in in_split.tolist():
                bucket = bucket_table[int(bucket_codes[record_index])]
                if bucket == DISCARDED_QUALITY_BUCKET:
                    self.promotion_excluded_discarded += 1
                    continue
                keep.append(int(record_index))
            self.promoted_indices = np.asarray(sorted(keep), dtype=np.int64)
        if self.promoted_indices.size:
            flags = self._instrumental_flags[self.promoted_indices]
            for flavor in (False, True):
                subset = self.promoted_indices[flags == flavor]
                if subset.size:
                    self.promotion_pools[flavor] = subset

    def _required_flavors(self) -> tuple[bool, ...]:
        required: list[bool] = []
        if self.instrumental_ratio < 1.0:
            required.append(False)
        if self.instrumental_ratio > 0.0:
            required.append(True)
        return tuple(required)

    def _validate_strict_pool_support(self) -> None:
        target = set(self.quality_schedule.positive_support)
        if not target:
            raise ValueError("strict_paper schedule has no positive-weight quality buckets")
        for flavor in self._required_flavors():
            available = set(self._buckets_by_flavor[flavor])
            missing = sorted(target - available)
            if missing:
                name = "instrumental" if flavor else "vocal"
                raise ValueError(
                    f"strict_paper {name} pool is missing target buckets {missing}; "
                    f"available buckets: {sorted(available)}"
                )
            if self.promotion_ratio > 0:
                pool = self.promotion_pools.get(flavor)
                if pool is None or not pool.size:
                    name = "instrumental" if flavor else "vocal"
                    raise ValueError(
                        f"strict_paper sets promotion_ratio={self.promotion_ratio:.4f}, "
                        f"but the {name} promotion pool is empty"
                    )

    def _validate_strict_joint_balance(self) -> None:
        if self.promotion_ratio > 0 and not (
            self.language_balance and self.genre_balance
        ):
            raise ValueError(
                "strict_paper promotion requires both language_balance and genre_balance "
                "for auditable language-by-genre residuals"
            )
        if not self.genre_balance:
            return
        original_step = self.step_counter.value
        try:
            for step, _ in self.quality_schedule.keyframes:
                diagnostics = self.sampling_diagnostics(step)
                genre_residual = (
                    0.0
                    if diagnostics.genre_deviation is None
                    else diagnostics.genre_deviation
                )
                if (
                    diagnostics.language_deviation > DISTRIBUTION_TOLERANCE
                    or genre_residual > DISTRIBUTION_TOLERANCE
                ):
                    raise ValueError(
                        "strict_paper language-by-genre balance audit failed: "
                        f"step={step}, language_tv={diagnostics.language_deviation:.4f}, "
                        f"genre_tv={genre_residual:.4f}, "
                        f"tolerance={DISTRIBUTION_TOLERANCE:.4f}; sparse support or the "
                        "promotion distribution cannot satisfy both targets"
                    )
        finally:
            self.step_counter.set(original_step)

    def _build_mode_plan(self) -> None:
        instrumental_probability = self.instrumental_ratio if self.has_instrumental else 0.0
        if not self.has_vocal:
            instrumental_probability = 1.0
        melody_frames = np.asarray(self.index.melody_frames)[self.record_indices]
        vocal_mask = ~self._instrumental_flags[self.record_indices]
        vocal_total = int(vocal_mask.sum())
        vocal_without_melody = (
            float((melody_frames[vocal_mask] <= 0).sum()) / vocal_total
            if vocal_total
            else 1.0
        )
        self.forced_plain_probability = float(
            instrumental_probability
            + (1.0 - instrumental_probability) * vocal_without_melody
        )
        self.instrumental_probability = float(instrumental_probability)

        configured_plain = float(self.mode_ratios.get(SequenceMode.PLAIN, 0.0))
        p = self.forced_plain_probability
        if self.mode_ratio_scope == "melodic" or p >= 1.0 - 1e-9:
            self.melodic_mode_ratios = dict(self.mode_ratios)
        elif configured_plain >= p:
            self.melodic_mode_ratios = {
                mode: (
                    (configured_plain - p) / (1.0 - p)
                    if mode is SequenceMode.PLAIN
                    else weight / (1.0 - p)
                )
                for mode, weight in self.mode_ratios.items()
            }
        else:


            self.melodic_mode_ratios = _normalize_modes(
                {
                    mode: weight
                    for mode, weight in self.mode_ratios.items()
                    if mode is not SequenceMode.PLAIN
                }
            )
        current_plain = float(
            self.melodic_mode_ratios.get(SequenceMode.PLAIN, 0.0)
        )
        self.melodic_plain_floor_binding = (
            current_plain + 1e-12 < self.melodic_vocal_plain_floor
        )
        if self.melodic_plain_floor_binding:
            non_plain = {
                mode: weight
                for mode, weight in self.melodic_mode_ratios.items()
                if mode is not SequenceMode.PLAIN and weight > 0
            }
            non_plain_total = float(sum(non_plain.values()))
            if non_plain_total <= 0 or self.melodic_vocal_plain_floor >= 1.0:
                self.melodic_mode_ratios = {SequenceMode.PLAIN: 1.0}
            else:
                remaining = 1.0 - self.melodic_vocal_plain_floor
                self.melodic_mode_ratios = {
                    SequenceMode.PLAIN: self.melodic_vocal_plain_floor,
                    **{
                        mode: remaining * weight / non_plain_total
                        for mode, weight in non_plain.items()
                    },
                }
        self.melodic_vocal_plain_probability = float(
            self.melodic_mode_ratios.get(SequenceMode.PLAIN, 0.0)
        )
        self._melodic_modes = [
            mode for mode, weight in self.melodic_mode_ratios.items() if weight > 0
        ]
        probs = np.asarray(
            [self.melodic_mode_ratios[mode] for mode in self._melodic_modes],
            dtype=np.float64,
        )
        self._melodic_probs = probs / probs.sum()

        expected: dict[str, float] = {
            mode.value: (1.0 - p) * weight
            for mode, weight in self.melodic_mode_ratios.items()
        }
        expected[SequenceMode.PLAIN.value] = expected.get(SequenceMode.PLAIN.value, 0.0) + p
        self.expected_mode_ratios = _normalized(expected)


    def _bucket_choice(self, want_instrumental: bool) -> tuple[list[str], np.ndarray]:
        step = self.step_counter.value
        cache_key = (want_instrumental, step)
        cached = self._bucket_weight_cache.get(cache_key)
        if cached is not None:
            return cached
        available = self._buckets_by_flavor[want_instrumental]
        weights = self.quality_schedule.weights_at(step)
        usable = {b: w for b, w in weights.items() if b in available and w > 0}
        if not usable:
            name = "instrumental" if want_instrumental else "vocal"
            target = sorted(bucket for bucket, weight in weights.items() if weight > 0)
            raise ValueError(
                f"{name} pool has no current target bucket at step={step}: "
                f"target={target}, available={available}; fallback to out-of-schedule "
                "quality buckets is disabled"
            )
        buckets = sorted(usable)
        probs = np.asarray([usable[b] for b in buckets], dtype=np.float64)
        probs = probs / probs.sum()

        if len(self._bucket_weight_cache) > 4:
            self._bucket_weight_cache.clear()
        self._bucket_weight_cache[cache_key] = (buckets, probs)
        return buckets, probs

    def _draw_instrumental(self) -> bool:
        if not self.has_vocal:
            return True
        if not self.has_instrumental or self.instrumental_ratio <= 0:
            return False
        return bool(self._rng.random() < self.instrumental_ratio)

    def _draw_key(self, want_instrumental: bool) -> _PoolKey:
        buckets, bucket_probs = self._bucket_choice(want_instrumental)
        bucket = buckets[int(self._rng.choice(len(buckets), p=bucket_probs))]
        cells, cell_probs = self._cell_choice[(want_instrumental, bucket)]
        language, genre = cells[int(self._rng.choice(len(cells), p=cell_probs))]
        return (want_instrumental, bucket, language, genre)

    def _draw_index(self) -> int:
        want_instrumental = self._draw_instrumental()
        if self.promotion_ratio > 0:
            pool = self.promotion_pools.get(want_instrumental)
            if pool is not None and pool.size and self._rng.random() < self.promotion_ratio:
                return int(pool[int(self._rng.integers(pool.size))])
        key = self._draw_key(want_instrumental)
        pool = self.pools[key]
        return int(pool[int(self._rng.integers(pool.size))])

    def _draw_mode(self, record_index: int) -> SequenceMode:
        if bool(self._instrumental_flags[record_index]):
            return SequenceMode.PLAIN
        if int(self.index.melody_frames[record_index]) <= 0:
            return SequenceMode.PLAIN
        return self._melodic_modes[
            int(self._rng.choice(len(self._melodic_modes), p=self._melodic_probs))
        ]

    def draw_sample(self) -> tuple[int, SequenceMode]:
        record_index = self._draw_index()
        return record_index, self._draw_mode(record_index)

    def length_estimate_is_upper_bound(self) -> bool:
        if self.condition_tokens_coverage >= 1.0:
            return True
        return self.condition_token_estimate >= self.max_condition_tokens

    def condition_length_bound(self, record_index: int) -> int:
        known = int(self._condition_tokens[record_index])
        if known < 0:
            return self.condition_token_estimate
        return min(known, self.max_condition_tokens)

    def estimated_length(self, record_index: int, mode: SequenceMode) -> int:
        semantic = min(int(self.index.semantic_frames[record_index]), self.max_semantic_frames)
        melody = 0
        if mode.has_melody:
            melody_frames = int(self.index.melody_frames[record_index])
            if melody_frames > 0:
                melody = melody_frames + 2 * int(self._num_sections[record_index]) + 2
        anchor = 0
        if (
            self.semantic_section_reanchor
            and mode.has_melody
            and not mode.is_unique_section
        ):
            if self.semantic_anchor_length_fn is not None:
                anchor = int(
                    self.semantic_anchor_length_fn(record_index, mode)
                )
                if anchor < 0:
                    raise ValueError("semantic_anchor_length_fn returns a negative number")
            else:

                anchor = int(self._num_sections[record_index]) * (
                    self.semantic_anchor_max_text_tokens + 3
                )
        total = (
            semantic
            + melody
            + self.condition_length_bound(record_index)
            + anchor
            + 8
        )
        return min(_round_up(total, self.pad_to_multiple_of), self._length_ceiling)


    def effective_promotion_ratio(self) -> float:
        if self.promotion_ratio <= 0:
            return 0.0
        reachable = 0.0
        for flavor, flavor_probability in (
            (True, self.instrumental_probability),
            (False, 1.0 - self.instrumental_probability),
        ):
            pool = self.promotion_pools.get(flavor)
            if pool is not None and pool.size:
                reachable += flavor_probability
        return self.promotion_ratio * reachable

    def sampling_diagnostics(self, step: int | None = None) -> SamplingDiagnostics:
        if step is not None:
            self.step_counter.set(step)
        current_step = self.step_counter.value

        bucket_marginal: dict[str, float] = defaultdict(float)
        language_marginal: dict[str, float] = defaultdict(float)
        genre_marginal: dict[str, float] = defaultdict(float)
        instrumental_mass = 0.0
        fields = (
            ("quality.bucket", bucket_marginal),
            ("language", language_marginal),
            ("quality.genre", genre_marginal),
        )

        for flavor, flavor_probability in (
            (True, self.instrumental_probability),
            (False, 1.0 - self.instrumental_probability),
        ):
            if flavor_probability <= 0 or not self._buckets_by_flavor[flavor]:
                continue
            if flavor:
                instrumental_mass += flavor_probability
            pool = self.promotion_pools.get(flavor)
            promotion = (
                self.promotion_ratio if (pool is not None and pool.size) else 0.0
            )
            if promotion > 0:

                share = flavor_probability * promotion / float(pool.size)
                for field_name, marginal in fields:
                    table = self.index.values(field_name)
                    codes = self.index.codes(field_name)
                    for record_index in pool.tolist():
                        marginal[table[int(codes[record_index])]] += share
            buckets, bucket_probs = self._bucket_choice(flavor)
            remaining = flavor_probability * (1.0 - promotion)
            for bucket, bucket_probability in zip(buckets, bucket_probs.tolist()):
                mass = remaining * bucket_probability
                bucket_marginal[bucket] += mass
                cells, cell_probs = self._cell_choice[(flavor, bucket)]
                for (language, genre), cell_probability in zip(cells, cell_probs.tolist()):
                    language_marginal[language] += mass * cell_probability
                    genre_marginal[genre] += mass * cell_probability

        effective_buckets = _normalized(dict(bucket_marginal))
        effective_languages = _normalized(dict(language_marginal))
        effective_genres = _normalized(dict(genre_marginal))
        target_buckets = _normalized(
            {
                bucket: weight
                for bucket, weight in self.quality_schedule.weights_at(current_step).items()
                if weight > 0
            }
        )
        target_genres = dict(self.genre_weights) if self.genre_weights else None

        warnings: list[str] = []
        bucket_deviation = total_variation(target_buckets, effective_buckets)
        if bucket_deviation > DISTRIBUTION_TOLERANCE:
            missing = sorted(set(target_buckets) - set(effective_buckets))
            warnings.append(
                f"Quality-bucket distribution differs from its schedule: TV={bucket_deviation:.4f}"
                + (f"; missing buckets: {missing}" if missing else "")
            )
        language_deviation = total_variation(self.language_weights, effective_languages)
        if language_deviation > DISTRIBUTION_TOLERANCE:
            warnings.append(
                f"Language distribution differs from its target: TV={language_deviation:.4f}"
            )
        genre_deviation: float | None = None
        if target_genres is not None:
            genre_deviation = total_variation(target_genres, effective_genres)
            if genre_deviation > DISTRIBUTION_TOLERANCE:
                warnings.append(
                    f"Genre distribution differs from its target: TV={genre_deviation:.4f}; "
                    "the language-by-genre support is too sparse to satisfy both marginals"
                )
        configured_plain = float(self.mode_ratios.get(SequenceMode.PLAIN, 0.0))
        mode_deviation = total_variation(
            {mode.value: weight for mode, weight in self.mode_ratios.items()},
            self.expected_mode_ratios,
        )
        if mode_deviation > DISTRIBUTION_TOLERANCE:
            if self.mode_ratio_scope == "all":
                if self.melodic_plain_floor_binding:
                    warnings.append(
                        f"Mode distribution differs from its global target: TV={mode_deviation:.4f}; "
                        f"the melodic-vocal PLAIN floor raises its probability from "
                        f"{self.melodic_vocal_plain_floor:.4f} to "
                        f"{self.melodic_vocal_plain_probability:.4f}"
                    )
                else:
                    warnings.append(
                        f"Mode distribution cannot reach its target: TV={mode_deviation:.4f}; "
                        f"forced PLAIN probability {self.forced_plain_probability:.4f} exceeds "
                        f"the configured value {configured_plain:.4f}"
                    )
            else:
                warnings.append(
                    "mode_ratio_scope=melodic applies only to melody-conditioned samples; "
                    f"the overall mode distribution differs by TV={mode_deviation:.4f}"
                )
        effective_instrumental = instrumental_mass
        if abs(effective_instrumental - self.instrumental_ratio) > DISTRIBUTION_TOLERANCE:
            warnings.append(
                f"Observed instrumental ratio {effective_instrumental:.4f} differs from "
                f"the configured ratio {self.instrumental_ratio:.4f}"
            )
        effective_promotion = self.effective_promotion_ratio()
        if self.promotion_ratio > 0 and effective_promotion <= 0:
            warnings.append(
                f"promotion.ratio={self.promotion_ratio:.4f}, but the promotion pool is empty"
            )
        if self.promotion_excluded_discarded:
            warnings.append(
                f"Excluded {self.promotion_excluded_discarded} promotion samples from "
                f"discard bucket {DISCARDED_QUALITY_BUCKET}"
            )


        upper_bound = self.length_estimate_is_upper_bound()
        if not upper_bound:
            warnings.append(
                f"Condition-length estimates are not upper bounds: "
                f"{1.0 - self.condition_tokens_coverage:.1%} of samples use the fallback "
                f"condition_token_estimate={self.condition_token_estimate}, which can "
                f"underestimate each sample by up to "
                f"{self.max_condition_tokens - self.condition_token_estimate} tokens. "
                "Build the index with the text tokenizer or set "
                f"sampler.condition_token_estimate={self.max_condition_tokens}."
            )

        return SamplingDiagnostics(
            step=current_step,
            strict_paper=self.strict_paper,
            target_buckets=target_buckets,
            effective_buckets=effective_buckets,
            bucket_deviation=bucket_deviation,
            target_languages=dict(self.language_weights),
            effective_languages=effective_languages,
            language_deviation=language_deviation,
            target_genres=target_genres,
            effective_genres=effective_genres,
            genre_deviation=genre_deviation,
            target_instrumental=self.instrumental_ratio,
            effective_instrumental=effective_instrumental,
            configured_modes={
                mode.value: weight for mode, weight in self.mode_ratios.items()
            },
            expected_modes=dict(self.expected_mode_ratios),
            mode_deviation=mode_deviation,
            mode_ratio_scope=self.mode_ratio_scope,
            forced_plain_probability=self.forced_plain_probability,
            melodic_vocal_plain_floor=self.melodic_vocal_plain_floor,
            melodic_vocal_plain_probability=self.melodic_vocal_plain_probability,
            configured_promotion_ratio=self.promotion_ratio,
            effective_promotion_ratio=effective_promotion,
            promotion_pool_size=int(self.promoted_indices.size),
            condition_tokens_coverage=self.condition_tokens_coverage,
            length_estimate_is_upper_bound=upper_bound,
            warnings=warnings,
        )

    def _warn_on_deviation(self) -> None:
        diagnostics = self.sampling_diagnostics()
        if not diagnostics.warnings:
            return
        for message in diagnostics.warnings:
            print(f"[sampler][warn] {message}", flush=True)


    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._rng = np.random.default_rng(self.seed + 7919 * self.rank + 104729 * self.epoch)
        self._pending_batches = []

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": "oqm.llm.sampler-state.v3",
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "epoch": self.epoch,
            "step": self.step_counter.value,
            "manifest_sha256": str(
                self.index.metadata.get("source_sha256") or ""
            ),
            "index_revision": self.index.revision,
            "record_indices_sha256": hashlib.sha256(
                np.ascontiguousarray(self.record_indices, dtype="<i8").tobytes()
            ).hexdigest(),
            "rng": self._rng.bit_generator.state,


            "pending_batches": [
                [
                    [int(record_index), str(mode), int(epoch)]
                    for record_index, mode, epoch in batch
                ]
                for batch in self._pending_batches
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        version = state.get("format_version")
        if version != "oqm.llm.sampler-state.v3":
            raise ValueError(
                f"Unsupported sampler state format: {state.get('format_version')}"
            )
        for name, expected in (
            ("seed", self.seed),
            ("rank", self.rank),
            ("world_size", self.world_size),
        ):
            if int(state.get(name, -1)) != int(expected):
                raise ValueError(
                    f"Sampler state {name}={state.get(name)} does not match current={expected}; "
                    "exact resume is unavailable"
                )
        identities = {
            "manifest_sha256": str(
                self.index.metadata.get("source_sha256") or ""
            ),
            "index_revision": self.index.revision,
            "record_indices_sha256": hashlib.sha256(
                np.ascontiguousarray(self.record_indices, dtype="<i8").tobytes()
            ).hexdigest(),
        }
        for name, expected in identities.items():
            if str(state.get(name) or "") != expected:
                raise ValueError(
                    f"Sampler state {name} does not match the current corpus or index; "
                    "resume is unavailable"
                )
        self.epoch = int(state["epoch"])
        self.step_counter.set(int(state["step"]))
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = dict(state["rng"])
        self._pending_batches = [
            [
                (int(record_index), str(mode), int(epoch))
                for record_index, mode, epoch in batch
            ]
            for batch in state.get("pending_batches", ())
        ]

    def __iter__(self) -> Iterator[list[tuple[int, str, int]]]:
        while True:
            if not self._pending_batches:
                approximate_batch = max(
                    1, self.max_tokens_per_gpu // max(1, self.max_sequence_length // 2)
                )
                pool_size = min(
                    4096, max(8, approximate_batch * self.candidate_pool_batches)
                )
                candidates: list[tuple[int, int, SequenceMode]] = []
                for _ in range(pool_size):
                    record_index, mode = self.draw_sample()
                    candidates.append(
                        (self.estimated_length(record_index, mode), record_index, mode)
                    )
                candidates.sort(key=lambda item: item[0])

                batches: list[list[tuple[int, str, int]]] = []
                batch: list[tuple[int, str, int]] = []
                longest = 0
                for length, record_index, mode in candidates:
                    candidate_longest = max(longest, length)
                    if batch and (
                        candidate_longest * (len(batch) + 1) > self.max_tokens_per_gpu
                        or len(batch) >= self.max_batch_size
                    ):
                        batches.append(batch)
                        batch = []
                        longest = 0
                        candidate_longest = length
                    batch.append((record_index, mode.value, self.epoch))
                    longest = candidate_longest

                if batch:
                    batches.append(batch)
                self._pending_batches = batches
            yield self._pending_batches.pop(0)

    def __len__(self) -> int:  # pragma: no cover -  sampler
        raise TypeError("CurriculumBatchSampler is infinite and has no length")


def build_sampler(
    config: dict[str, Any],
    index: ManifestIndex,
    record_indices: np.ndarray,
    *,
    rank: int,
    world_size: int,
    step_counter: StepCounter,
    semantic_anchor_length_fn: Callable[[int, SequenceMode], int]
    | None = None,
) -> CurriculumBatchSampler:
    sampler_config = dict(config.get("sampler", {}) or {})
    sequence_config = dict(config.get("sequence", {}) or {})
    train_config = dict(config.get("train", {}) or {})
    language_config = dict(sampler_config.get("language", {}) or {})
    genre_config = dict(sampler_config.get("genre", {}) or {})
    promotion_config = dict(sampler_config.get("promotion", {}) or {})

    condition_ceiling = int(sequence_config.get("max_condition_tokens", 1024))

    promoted_indices: np.ndarray | None = None
    promotion_ratio = float(promotion_config.get("ratio", 0.0))
    if promotion_config.get("enabled", False) and promotion_ratio > 0:
        promoted_indices = resolve_promotion_pool(
            index,
            from_manifest_flag=bool(promotion_config.get("from_manifest_flag", True)),
            ids=promotion_config.get("ids"),
            ids_file=promotion_config.get("ids_file"),
            strict=bool(promotion_config.get("strict", True)),
        )
    else:
        promotion_ratio = 0.0

    return CurriculumBatchSampler(
        index,
        record_indices,
        rank=rank,
        world_size=world_size,
        seed=int(train_config.get("seed", 0)),
        max_tokens_per_gpu=int(train_config.get("max_tokens_per_gpu", 8192)),
        max_sequence_length=int(sequence_config.get("max_sequence_length", 4096)),
        max_semantic_frames=int(sequence_config.get("max_semantic_frames", 2250)),
        quality_schedule=QualitySchedule.from_config(config),
        mode_ratios=sampler_config.get("modes", {"plain": 1.0}),
        instrumental_ratio=float(sampler_config.get("instrumental_ratio", 0.0)),
        language_balance=bool(language_config.get("balance", True)),
        language_temperature=float(language_config.get("temperature", 0.5)),
        genre_balance=bool(genre_config.get("balance", False)),
        genre_temperature=float(genre_config.get("temperature", 0.5)),
        mode_ratio_scope=str(sampler_config.get("mode_ratio_scope", "all")),
        melodic_vocal_plain_floor=float(
            sampler_config.get("melodic_vocal_plain_floor", 0.05)
        ),
        strict_paper=bool(sampler_config.get("strict_paper", False)),
        promoted_indices=promoted_indices,
        promotion_ratio=promotion_ratio,


        #  TRAINING.md P1).
        condition_token_estimate=int(
            sampler_config.get("condition_token_estimate", condition_ceiling)
        ),
        max_condition_tokens=condition_ceiling,

        pad_to_multiple_of=int(train_config.get("pad_to_multiple_of", 64)),
        candidate_pool_batches=int(sampler_config.get("candidate_pool_batches", 16)),
        max_batch_size=int(train_config.get("max_batch_size", 64)),
        step_counter=step_counter,
        semantic_section_reanchor=bool(
            sequence_config.get("semantic_section_reanchor", False)
        ),
        semantic_anchor_max_text_tokens=int(
            sequence_config.get("semantic_anchor_max_text_tokens", 256)
        ),
        semantic_anchor_length_fn=semantic_anchor_length_fn,
    )


def uniform_eval_batches(
    index: ManifestIndex,
    record_indices: Sequence[int],
    *,
    mode: SequenceMode,
    max_tokens: int,
    max_sequence_length: int,
    max_semantic_frames: int,
    condition_token_estimate: int = 256,
    max_condition_tokens: int | None = None,
    pad_to_multiple_of: int = 64,
    max_batch_size: int = 32,
    semantic_section_reanchor: bool = False,
    semantic_anchor_max_text_tokens: int = 256,
    semantic_anchor_length_fn: Callable[[int, SequenceMode], int]
    | None = None,
) -> list[list[tuple[int, str, int]]]:
    ceiling = int(
        condition_token_estimate if max_condition_tokens is None else max_condition_tokens
    )
    length_ceiling = _round_up(max_sequence_length, pad_to_multiple_of)
    condition_tokens = np.asarray(index.condition_tokens, dtype=np.int64)
    num_sections = np.asarray(index.num_sections, dtype=np.int64)
    lengths: list[tuple[int, int]] = []
    for record_index in record_indices:
        semantic = min(int(index.semantic_frames[record_index]), max_semantic_frames)
        melody_frames = int(index.melody_frames[record_index])
        melody = 0
        if mode.has_melody and melody_frames > 0:
            melody = melody_frames + 2 * int(num_sections[record_index]) + 2
        known = int(condition_tokens[record_index])
        condition = condition_token_estimate if known < 0 else min(known, ceiling)
        anchor = 0
        if (
            semantic_section_reanchor
            and mode.has_melody
            and not mode.is_unique_section
        ):
            anchor = (
                int(semantic_anchor_length_fn(int(record_index), mode))
                if semantic_anchor_length_fn is not None
                else int(num_sections[record_index])
                * (int(semantic_anchor_max_text_tokens) + 3)
            )
        total = semantic + melody + condition + anchor + 8
        lengths.append(
            (min(_round_up(total, pad_to_multiple_of), length_ceiling), int(record_index))
        )
    lengths.sort()
    batches: list[list[tuple[int, str, int]]] = []
    batch: list[tuple[int, str, int]] = []
    longest = 0
    for length, record_index in lengths:
        candidate_longest = max(longest, length)
        if batch and (
            candidate_longest * (len(batch) + 1) > max_tokens or len(batch) >= max_batch_size
        ):
            batches.append(batch)
            batch = []
            candidate_longest = length
        batch.append((record_index, mode.value, 0))
        longest = candidate_longest
    if batch:
        batches.append(batch)
    return batches
