
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .contracts import SEMANTIC_CODEBOOK_SIZE, SequenceMode
from .grammar import (
    AllowedTokenSets,
    GrammarConfig,
    GrammarRowState,
    advance_state,
    allowed_kind_for_state,
    initial_state,
    token_is_allowed,
)
from .melody import melody_tokens_to_relative_semitones, recenter_melody_tokens
from .registry import TokenRegistry


melody_tokens_to_semitones = melody_tokens_to_relative_semitones


def dtw_mean_absolute_error(
    reference: np.ndarray, hypothesis: np.ndarray, *, band_ratio: float | None = None
) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    hyp = np.asarray(hypothesis, dtype=np.float64)
    if ref.size == 0 or hyp.size == 0:
        return float("nan")
    if np.all(np.isnan(ref)) or np.all(np.isnan(hyp)):
        return float("nan")

    n, m = ref.size, hyp.size
    band = None
    if band_ratio is not None:
        band = max(1, int(round(band_ratio * max(n, m))))


    difference = np.abs(ref[:, None] - hyp[None, :])
    valid = ~np.isnan(difference)
    cost = np.where(valid, difference, 0.0)

    infinity = float("inf")
    accumulated = np.full((n + 1, m + 1), infinity, dtype=np.float64)
    counts = np.zeros((n + 1, m + 1), dtype=np.float64)
    accumulated[0, 0] = 0.0
    for i in range(1, n + 1):
        lower, upper = 1, m
        if band is not None:
            center = int(round(i * m / n))
            lower = max(1, center - band)
            upper = min(m, center + band)
        for j in range(lower, upper + 1):
            candidates = (
                accumulated[i - 1, j - 1],
                accumulated[i - 1, j],
                accumulated[i, j - 1],
            )
            best = int(np.argmin(candidates))
            previous = candidates[best]
            if previous == infinity:
                continue
            previous_counts = (
                counts[i - 1, j - 1],
                counts[i - 1, j],
                counts[i, j - 1],
            )[best]
            accumulated[i, j] = previous + cost[i - 1, j - 1]
            counts[i, j] = previous_counts + (1.0 if valid[i - 1, j - 1] else 0.0)
    if accumulated[n, m] == infinity or counts[n, m] <= 0:
        return float("nan")
    return float(accumulated[n, m] / counts[n, m])


def melody_mae(
    reference_tokens: Sequence[int] | np.ndarray,
    hypothesis_tokens: Sequence[int] | np.ndarray,
    *,
    band_ratio: float | None = 0.2,
    recenter: bool = False,
) -> float:
    reference: Sequence[int] | np.ndarray = reference_tokens
    hypothesis: Sequence[int] | np.ndarray = hypothesis_tokens
    if recenter:
        reference = recenter_melody_tokens(reference)
        hypothesis = recenter_melody_tokens(hypothesis)
    return dtw_mean_absolute_error(
        melody_tokens_to_relative_semitones(reference),
        melody_tokens_to_relative_semitones(hypothesis),
        band_ratio=band_ratio,
    )


@dataclass
class LegalityReport:
    total_tokens: int
    illegal_tokens: int
    complete_sequences: int
    total_sequences: int
    unique_semantic_codes: int
    semantic_tokens: int
    legal_sequences: int
    grammar_complete_sequences: int
    finished_mismatches: int

    @property
    def legality_rate(self) -> float:
        return 1.0 if self.total_tokens == 0 else 1.0 - self.illegal_tokens / self.total_tokens

    @property
    def namespace_legality_rate(self) -> float:
        return self.legality_rate

    @property
    def completion_rate(self) -> float:
        return (
            0.0
            if self.total_sequences == 0
            else self.complete_sequences / self.total_sequences
        )

    @property
    def codebook_coverage(self) -> float:
        return self.unique_semantic_codes / float(SEMANTIC_CODEBOOK_SIZE)

    @property
    def sequence_legality_rate(self) -> float:
        return (
            0.0
            if self.total_sequences == 0
            else self.legal_sequences / self.total_sequences
        )

    @property
    def grammar_completion_rate(self) -> float:
        return (
            0.0
            if self.total_sequences == 0
            else self.grammar_complete_sequences / self.total_sequences
        )

    @property
    def finished_mismatch_rate(self) -> float:
        return self.finished_mismatches / max(self.total_sequences, 1)

    def as_dict(self) -> dict[str, float]:
        return {
            "legality_rate": self.legality_rate,
            "namespace_legality_rate": self.legality_rate,
            "sequence_legality_rate": self.sequence_legality_rate,
            "completion_rate": self.completion_rate,
            "grammar_completion_rate": self.grammar_completion_rate,
            "finished_mismatch_rate": self.finished_mismatch_rate,
            "codebook_coverage": self.codebook_coverage,
            "unique_semantic_codes": float(self.unique_semantic_codes),
            "semantic_tokens": float(self.semantic_tokens),
        }


def legality_report(
    raw_sequences: Sequence[np.ndarray],
    registry: TokenRegistry,
    *,
    finished: Sequence[bool] | None = None,
) -> LegalityReport:
    total = 0
    illegal = 0
    semantic_total = 0
    codes: set[int] = set()
    legal_sequences = 0
    grammar_complete = 0
    finished_mismatches = 0
    max_raw_length = max(
        (len(np.asarray(sequence)) for sequence in raw_sequences),
        default=1,
    )
    sets = AllowedTokenSets(registry, torch.device("cpu"))
    grammar_config = GrammarConfig(
        min_semantic_frames=0,
        max_semantic_frames=max(1, max_raw_length),
        max_melody_tokens=max(1, max_raw_length),
        max_melody_segments=max(1, max_raw_length),
        min_melody_segments=0,
        strict_mode=False,
    )
    for sequence_index, sequence in enumerate(raw_sequences):
        values = [int(token) for token in np.asarray(sequence).tolist()]
        mode = (
            SequenceMode.SECTION
            if values and values[0] == registry.control("melody_bos")
            else SequenceMode.PLAIN
        )
        row = GrammarRowState(state=initial_state(mode))
        sequence_legal = bool(values)
        for token in values:
            if row.finished:
                sequence_legal = False
                break
            kind = allowed_kind_for_state(row, grammar_config, mode)
            if not token_is_allowed(kind, token, registry):
                sequence_legal = False
                break
            advance_state(row, token, sets, registry)
        if sequence_legal:
            legal_sequences += 1
        if row.finished:
            grammar_complete += 1
        if finished is not None and bool(finished[sequence_index]) != bool(row.finished):
            finished_mismatches += 1
        for token in np.asarray(sequence).tolist():
            token = int(token)
            total += 1
            namespace = registry.namespace_of(token)
            if namespace in ("text", "reserved"):
                illegal += 1
            elif namespace == "semantic":
                semantic_total += 1
                codes.add(token - registry.semantic_base)
    complete = (
        int(sum(bool(f) for f in finished))
        if finished is not None
        else grammar_complete
    )
    return LegalityReport(
        total_tokens=total,
        illegal_tokens=illegal,
        complete_sequences=complete,
        total_sequences=len(raw_sequences),
        unique_semantic_codes=len(codes),
        semantic_tokens=semantic_total,
        legal_sequences=legal_sequences,
        grammar_complete_sequences=grammar_complete,
        finished_mismatches=finished_mismatches,
    )


def perplexity(mean_loss_nats: float) -> float:
    return float(np.exp(min(float(mean_loss_nats), 50.0)))


def section_following_rate(
    expected_sections: Sequence[Sequence[str]], produced_sections: Sequence[Sequence[str]]
) -> float:
    if len(expected_sections) != len(produced_sections):
        raise ValueError(
            "Expected and produced section-sequence counts differ: "
            f"{len(expected_sections)} vs {len(produced_sections)}"
        )
    if not expected_sections:
        return float("nan")
    matched = 0
    for expected, produced in zip(expected_sections, produced_sections):
        if list(expected) == list(produced):
            matched += 1
    return matched / float(len(expected_sections))


def phoneme_error_rate(
    reference_lyrics: Sequence[str], transcribed_lyrics: Sequence[str]
) -> float:  # pragma: no cover -  ASR
    raise NotImplementedError(
        "PER requires an ASR transcript, a phoneticizer, and rendered audio; "
        "see open_qwen_music.llm.eval.PhonemeErrorRate and TRAINING.md"
    )
