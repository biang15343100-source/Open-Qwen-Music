
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import MELODY_FRAME_RATE, SEMANTIC_FRAME_RATE


def duration_aligned_lyrics(
    record: dict[str, Any],
    duration_sec: float,
) -> tuple[str, int]:

    limit = max(0.0, float(duration_sec))
    chunks: list[str] = []
    used = 0
    for section in record.get("sections") or ():
        text = str(section.get("lyrics") or "").strip()
        if not text:
            continue
        start = section.get("start_sec")
        end = section.get("end_sec")
        if type(start) not in (int, float) or type(end) not in (int, float):
            continue
        start_value, end_value = float(start), float(end)
        if (
            not math.isfinite(start_value)
            or not math.isfinite(end_value)
            or end_value <= start_value
            or start_value >= limit
        ):
            continue
        covered = min(1.0, max(0.0, (limit - start_value) / (end_value - start_value)))
        if covered <= 0.0:
            continue
        if covered < 1.0:
            keep = max(1, int(math.ceil(len(text) * covered)))
            text = text[:keep].rstrip()
        if text:
            chunks.append(text)
            used += 1
    return "\n".join(chunks), used


def semantic_frequency_buckets(
    counts: Sequence[int] | np.ndarray,
) -> np.ndarray:

    values = np.asarray(counts, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ValueError("Semantic frequency counts cannot be empty")
    if np.any(values < 0):
        raise ValueError("Semantic frequency counts cannot be negative")
    output = np.full(values.size, 3, dtype=np.int8)
    active = np.flatnonzero(values > 0)
    if active.size == 0:
        return output

    order = active[np.lexsort((active, -values[active]))]
    groups = np.array_split(order, 3)
    for bucket, indices in enumerate(groups):
        output[indices] = bucket
    return output


def load_semantic_frequency_counts(
    path: str | Path,
    *,
    expected_size: int,
    expected_corpus_revision: str | None = None,
) -> np.ndarray:

    path = Path(path)
    if path.suffix == ".npy":
        counts = np.load(path, allow_pickle=False)
        metadata: dict[str, Any] = {}
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "counts" not in archive:
                raise KeyError(f"{path} is missing counts")
            counts = archive["counts"]
            metadata = (
                json.loads(str(archive["metadata"].item()))
                if "metadata" in archive.files
                else {}
            )
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        counts = payload["counts"] if isinstance(payload, dict) else payload
        metadata = payload if isinstance(payload, dict) else {}
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    if counts.size != int(expected_size):
        raise ValueError(
            f"Semantic frequency counts length {counts.size} != {expected_size}"
        )
    if np.any(counts < 0):
        raise ValueError("Semantic frequency counts cannot be negative")
    if expected_corpus_revision is not None:
        actual = str(metadata.get("corpus_revision") or "")
        if actual != str(expected_corpus_revision):
            raise RuntimeError(
                "Semantic frequency counts corpus revision mismatch:"
                f"counts={actual!r}, expected={expected_corpus_revision!r}"
            )
        if str(metadata.get("split") or "") != "train":
            raise RuntimeError(
                f"Semantic frequency counts must come from the train split; got "
                f"{metadata.get('split')!r}"
            )
    return counts


def semantic_distribution_metrics(
    sequences: Sequence[Sequence[int] | np.ndarray],
    *,
    codebook_size: int = 32768,
) -> dict[str, float]:

    arrays = [np.asarray(sequence, dtype=np.int64).reshape(-1) for sequence in sequences]
    nonempty = [array for array in arrays if array.size]
    if not nonempty:
        return {
            "semantic_tokens": 0.0,
            "unique_semantic_codes": 0.0,
            "codebook_coverage": 0.0,
            "unigram_entropy_nats": 0.0,
            "normalized_unigram_entropy": 0.0,
            "effective_codebook_size": 0.0,
            "max_code_fraction": 0.0,
            "repeated_4gram_fraction": 0.0,
            "longest_constant_run": 0.0,
            "local_lag_match_fraction": 0.0,
            "periodic_repeat_fraction": 0.0,
        }
    flat = np.concatenate(nonempty)
    if flat.min() < 0 or flat.max() >= codebook_size:
        raise ValueError("Freely generated Semantic local ID out of bounds")
    counts = np.bincount(flat, minlength=codebook_size).astype(np.float64)
    probabilities = counts[counts > 0] / float(flat.size)
    entropy = float(-(probabilities * np.log(probabilities)).sum())

    repeated_ngrams = 0
    total_ngrams = 0
    longest_run = 0
    periodic_positions = 0
    periodic_total = 0
    for array in arrays:
        if array.size:
            changes = np.flatnonzero(np.diff(array) != 0) + 1
            boundaries = np.concatenate(([0], changes, [array.size]))
            longest_run = max(longest_run, int(np.diff(boundaries).max(initial=0)))
        if array.size >= 4:
            grams = [
                tuple(int(value) for value in array[index : index + 4])
                for index in range(array.size - 3)
            ]
            frequencies = Counter(grams)
            repeated_ngrams += sum(count - 1 for count in frequencies.values())
            total_ngrams += len(grams)

        if array.size >= 2:
            repeated = np.zeros(array.size, dtype=bool)
            for lag in range(1, min(16, array.size - 1) + 1):
                repeated[lag:] |= array[lag:] == array[:-lag]
            periodic_positions += int(repeated[1:].sum())
            periodic_total += int(array.size - 1)
    local_lag_match = periodic_positions / max(periodic_total, 1)
    return {
        "semantic_tokens": float(flat.size),
        "unique_semantic_codes": float((counts > 0).sum()),
        "codebook_coverage": float((counts > 0).sum()) / float(codebook_size),
        "unigram_entropy_nats": entropy,
        "normalized_unigram_entropy": entropy / math.log(float(codebook_size)),
        "effective_codebook_size": float(math.exp(entropy)),
        "max_code_fraction": float(counts.max()) / float(flat.size),
        "repeated_4gram_fraction": repeated_ngrams / max(total_ngrams, 1),
        "longest_constant_run": float(longest_run),
        "local_lag_match_fraction": local_lag_match,

        "periodic_repeat_fraction": local_lag_match,
    }


def per_sample_distribution_metrics(
    sequences: Sequence[Sequence[int] | np.ndarray],
    *,
    codebook_size: int,
    repeated_4gram_threshold: float,
    constant_run_threshold: int,
    max_code_fraction_threshold: float,
) -> dict[str, float]:

    if not 0.0 <= repeated_4gram_threshold <= 1.0:
        raise ValueError("repeated_4gram_thresholdmust be within[0,1]")
    if constant_run_threshold < 1:
        raise ValueError("constant_run_thresholdrequired>=1")
    if not 0.0 <= max_code_fraction_threshold <= 1.0:
        raise ValueError("max_code_fraction_thresholdmust be within[0,1]")
    rows = [
        semantic_distribution_metrics([sequence], codebook_size=codebook_size)
        for sequence in sequences
        if np.asarray(sequence).size
    ]
    if not rows:
        return {"samples": 0.0, "catastrophic_rate": 0.0}
    output: dict[str, float] = {"samples": float(len(rows))}
    for name in (
        "effective_codebook_size",
        "max_code_fraction",
        "repeated_4gram_fraction",
        "longest_constant_run",
        "local_lag_match_fraction",
    ):
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        output[f"{name}_mean"] = float(values.mean())
        output[f"{name}_median"] = float(np.median(values))
        output[f"{name}_p90"] = float(np.percentile(values, 90))
        output[f"{name}_p95"] = float(np.percentile(values, 95))
        output[f"{name}_max"] = float(values.max())
    catastrophic = [
        float(row["repeated_4gram_fraction"]) > repeated_4gram_threshold
        or float(row["longest_constant_run"]) > constant_run_threshold
        or float(row["max_code_fraction"]) > max_code_fraction_threshold
        for row in rows
    ]
    output["catastrophic_rate"] = float(np.mean(catastrophic))
    output["threshold/repeated_4gram_fraction"] = float(
        repeated_4gram_threshold
    )
    output["threshold/longest_constant_run"] = float(constant_run_threshold)
    output["threshold/max_code_fraction"] = float(max_code_fraction_threshold)
    return output


def token_distribution_distances(
    generated: Sequence[Sequence[int] | np.ndarray],
    reference: Sequence[Sequence[int] | np.ndarray],
    *,
    codebook_size: int,
    ordinal_ids: bool,
) -> dict[str, float]:

    generated_nonempty = [
        np.asarray(values, dtype=np.int64).reshape(-1)
        for values in generated
        if np.asarray(values).size
    ]
    reference_nonempty = [
        np.asarray(values, dtype=np.int64).reshape(-1)
        for values in reference
        if np.asarray(values).size
    ]
    if not generated_nonempty or not reference_nonempty:
        return {"matched_samples": 0.0}
    generated_flat = np.concatenate(generated_nonempty)
    reference_flat = np.concatenate(reference_nonempty)
    if (
        generated_flat.min() < 0
        or reference_flat.min() < 0
        or generated_flat.max() >= codebook_size
        or reference_flat.max() >= codebook_size
    ):
        raise ValueError("Token-distribution local ID is out of bounds")
    p = np.bincount(generated_flat, minlength=codebook_size).astype(np.float64)
    q = np.bincount(reference_flat, minlength=codebook_size).astype(np.float64)
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        active = left > 0
        return float((left[active] * np.log(left[active] / right[active])).sum())

    output = {
        "matched_samples": float(
            min(len(generated_nonempty), len(reference_nonempty))
        ),
        "total_variation": float(0.5 * np.abs(p - q).sum()),
        "jensen_shannon_nats": 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint),
    }
    if ordinal_ids:
        output["wasserstein_1d_normalized"] = float(
            np.abs(np.cumsum(p) - np.cumsum(q)).sum()
            / max(codebook_size - 1, 1)
        )
    return output


def generation_length_metrics(
    semantic_sequences: Sequence[Sequence[int] | np.ndarray],
    melody_sequences: Sequence[Sequence[int] | np.ndarray | None],
    stop_reasons: Sequence[str],
    *,
    max_semantic_frames: int | None = None,
    max_melody_tokens: int | None = None,
) -> dict[str, float]:

    if not (
        len(semantic_sequences) == len(melody_sequences) == len(stop_reasons)
    ):
        raise ValueError("Semantic, melody, and stop_reasons entry counts must match")

    semantic_lengths = np.asarray(
        [np.asarray(values).size for values in semantic_sequences],
        dtype=np.float64,
    )
    melody_lengths = np.asarray(
        [
            0 if values is None else np.asarray(values).size
            for values in melody_sequences
        ],
        dtype=np.float64,
    )
    semantic_seconds = semantic_lengths / SEMANTIC_FRAME_RATE
    melody_seconds = melody_lengths / MELODY_FRAME_RATE

    def describe(prefix: str, values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {
                f"{prefix}_count": 0.0,
                f"{prefix}_mean": 0.0,
                f"{prefix}_std": 0.0,
                f"{prefix}_min": 0.0,
                f"{prefix}_p10": 0.0,
                f"{prefix}_p50": 0.0,
                f"{prefix}_p90": 0.0,
                f"{prefix}_max": 0.0,
            }
        return {
            f"{prefix}_count": float(values.size),
            f"{prefix}_mean": float(values.mean()),
            f"{prefix}_std": float(values.std()),
            f"{prefix}_min": float(values.min()),
            f"{prefix}_p10": float(np.percentile(values, 10)),
            f"{prefix}_p50": float(np.percentile(values, 50)),
            f"{prefix}_p90": float(np.percentile(values, 90)),
            f"{prefix}_max": float(values.max()),
        }

    output: dict[str, float] = {}
    output.update(describe("length/semantic_frames", semantic_lengths))
    output.update(describe("length/melody_tokens", melody_lengths))
    output.update(describe("length/semantic_seconds", semantic_seconds))
    output.update(describe("length/melody_seconds", melody_seconds))

    paired = melody_lengths > 0
    token_ratios = semantic_lengths[paired] / melody_lengths[paired]
    duration_ratios = semantic_seconds[paired] / melody_seconds[paired]
    duration_differences = semantic_seconds[paired] - melody_seconds[paired]
    absolute_differences = np.abs(duration_differences)
    relative_differences = absolute_differences / np.maximum(
        np.maximum(semantic_seconds[paired], melody_seconds[paired]),
        1e-9,
    )
    output.update(describe("length/semantic_to_melody_token_ratio", token_ratios))
    output.update(describe("length/semantic_to_melody_duration_ratio", duration_ratios))
    output.update(describe("length/semantic_minus_melody_seconds", duration_differences))
    output.update(describe("length/absolute_duration_gap_seconds", absolute_differences))
    output["length/paired_stream_rate"] = float(
        paired.mean() if paired.size else 0.0
    )
    output["length/duration_within_1s_rate"] = float(
        np.mean(absolute_differences <= 1.0) if absolute_differences.size else 0.0
    )
    output["length/duration_within_10pct_rate"] = float(
        np.mean(relative_differences <= 0.10) if relative_differences.size else 0.0
    )

    reasons = [str(value) for value in stop_reasons]
    denominator = max(len(reasons), 1)
    for reason in ("natural_eos", "frame_cap", "max_new_tokens", "unfinished"):
        output[f"termination/{reason}_rate"] = float(
            sum(value == reason for value in reasons) / denominator
        )
    semantic_cap_hits = (
        semantic_lengths >= int(max_semantic_frames)
        if max_semantic_frames is not None
        else np.asarray([value == "frame_cap" for value in reasons], dtype=bool)
    )
    melody_cap_hits = (
        melody_lengths >= int(max_melody_tokens)
        if max_melody_tokens is not None
        else np.zeros_like(semantic_cap_hits)
    )
    any_cap_hits = semantic_cap_hits | melody_cap_hits
    raw_natural = np.asarray(
        [value == "natural_eos" for value in reasons],
        dtype=bool,
    )
    output["termination/semantic_cap_rate"] = float(
        semantic_cap_hits.mean() if semantic_cap_hits.size else 0.0
    )
    output["termination/melody_cap_rate"] = float(
        melody_cap_hits.mean() if melody_cap_hits.size else 0.0
    )
    output["termination/any_cap_rate"] = float(
        any_cap_hits.mean() if any_cap_hits.size else 0.0
    )
    output["termination/fully_natural_eos_rate"] = float(
        np.mean(raw_natural & ~any_cap_hits) if raw_natural.size else 0.0
    )
    return output


def section_sequence_metrics(
    expected: Sequence[Sequence[str]],
    produced: Sequence[Sequence[str]],
) -> dict[str, float]:

    if len(expected) != len(produced):
        raise ValueError("Expected and produced item counts must match")
    true_positive = false_positive = false_negative = edits = reference_count = 0
    count_error = 0
    exact = 0
    for reference, hypothesis in zip(expected, produced):
        reference = list(reference)
        hypothesis = list(hypothesis)
        exact += int(reference == hypothesis)
        reference_count += len(reference)
        count_error += abs(len(reference) - len(hypothesis))

        table = np.zeros((len(reference) + 1, len(hypothesis) + 1), dtype=np.int32)
        for row in range(1, len(reference) + 1):
            for column in range(1, len(hypothesis) + 1):
                if reference[row - 1] == hypothesis[column - 1]:
                    table[row, column] = table[row - 1, column - 1] + 1
                else:
                    table[row, column] = max(
                        table[row - 1, column], table[row, column - 1]
                    )
        matched = int(table[-1, -1])
        true_positive += matched
        false_positive += len(hypothesis) - matched
        false_negative += len(reference) - matched

        previous = list(range(len(hypothesis) + 1))
        for row, value in enumerate(reference, start=1):
            current = [row]
            for column, candidate in enumerate(hypothesis, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[column] + 1,
                        previous[column - 1] + int(value != candidate),
                    )
                )
            previous = current
        edits += previous[-1]
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    samples = len(expected)
    return {
        "section_exact_match_rate": exact / max(samples, 1),
        "section_precision": precision,
        "section_recall": recall,
        "section_f1": f1,
        "section_edit_distance_per_reference": edits / max(reference_count, 1),
        "section_count_mae": count_error / max(samples, 1),
    }


def finalize_calibration_metrics(raw_sums: dict[str, float]) -> dict[str, float]:

    bins: dict[str, dict[str, float]] = {}
    for name, value in raw_sums.items():
        if not name.startswith("_calibration_semantic_bin_"):
            continue
        prefix, field = name.rsplit("_", 1)
        if field == "sum":
            prefix, quantity = prefix.rsplit("_", 1)
            field = f"{quantity}_sum"
        bins.setdefault(prefix, {})[field] = float(value)
    total = sum(values.get("count", 0.0) for values in bins.values())
    if total <= 0:
        return {}
    ece = 0.0
    maximum_gap = 0.0
    for values in bins.values():
        count = values.get("count", 0.0)
        if count <= 0:
            continue
        confidence = values.get("confidence_sum", 0.0) / count
        accuracy = values.get("correct_sum", 0.0) / count
        gap = abs(accuracy - confidence)
        ece += gap * count / total
        maximum_gap = max(maximum_gap, gap)
    return {
        "ece_semantic_tokens": ece,
        "max_calibration_gap_semantic_tokens": maximum_gap,
        "calibration_semantic_tokens": total,
    }


def long_range_forgetting_metrics(metrics: dict[str, float]) -> dict[str, float]:

    output: dict[str, float] = {}
    loss_gaps: list[float] = []
    tail_gaps: list[float] = []
    accuracy_drops: list[float] = []
    for mode in ("plain", "section", "unique_section"):
        prefix = f"{mode}/"
        q1 = metrics.get(prefix + "loss_semantic_position_q1")
        q4 = metrics.get(prefix + "loss_semantic_position_q4")
        tail = metrics.get(prefix + "loss_semantic_position_tail500")
        acc_q1 = metrics.get(prefix + "acc_semantic_position_q1")
        acc_q4 = metrics.get(prefix + "acc_semantic_position_q4")
        if q1 is not None and q4 is not None:
            gap = float(q4) - float(q1)
            output[f"{mode}/long_range_loss_q4_minus_q1"] = gap
            loss_gaps.append(gap)
        if q1 is not None and tail is not None:
            gap = float(tail) - float(q1)
            output[f"{mode}/long_range_loss_tail500_minus_q1"] = gap
            tail_gaps.append(gap)
        if acc_q1 is not None and acc_q4 is not None:
            drop = float(acc_q1) - float(acc_q4)
            output[f"{mode}/long_range_acc_q1_minus_q4"] = drop
            accuracy_drops.append(drop)
    if loss_gaps:
        output["long_range/loss_q4_minus_q1_mean"] = float(np.mean(loss_gaps))
    if tail_gaps:
        output["long_range/loss_tail500_minus_q1_mean"] = float(
            np.mean(tail_gaps)
        )
    if accuracy_drops:
        output["long_range/acc_q1_minus_q4_mean"] = float(
            np.mean(accuracy_drops)
        )
    return output


def flatten_numeric(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:

    output: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            output.update(flatten_numeric(value, name))
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            output[name] = float(value)
    return output


__all__ = [
    "finalize_calibration_metrics",
    "flatten_numeric",
    "load_semantic_frequency_counts",
    "long_range_forgetting_metrics",
    "section_sequence_metrics",
    "semantic_distribution_metrics",
    "semantic_frequency_buckets",
]
