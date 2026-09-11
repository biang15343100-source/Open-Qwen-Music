
from __future__ import annotations

from collections.abc import Sequence
import math

import torch


def collapse_ctc(ids: Sequence[int], blank_id: int = 0) -> list[int]:

    output: list[int] = []
    previous: int | None = None
    for value in ids:
        value = int(value)
        if value != blank_id and value != previous:
            output.append(value)
        previous = value
    return output


def edit_distance(reference: Sequence[object], hypothesis: Sequence[object]) -> int:

    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_value in enumerate(reference, start=1):
        current = [row]
        for column, hyp_value in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_value != hyp_value),
                )
            )
        previous = current
    return previous[-1]


def edit_operations(
    reference: Sequence[object], hypothesis: Sequence[object]
) -> dict[str, int]:

    # (cost, substitutions, deletions, insertions)
    previous = [(column, 0, 0, column) for column in range(len(hypothesis) + 1)]
    for row, ref_value in enumerate(reference, start=1):
        current = [(row, 0, row, 0)]
        for column, hyp_value in enumerate(hypothesis, start=1):
            diagonal = previous[column - 1]
            mismatch = int(ref_value != hyp_value)
            substitution = (
                diagonal[0] + mismatch,
                diagonal[1] + mismatch,
                diagonal[2],
                diagonal[3],
            )
            deletion = (
                previous[column][0] + 1,
                previous[column][1],
                previous[column][2] + 1,
                previous[column][3],
            )
            insertion = (
                current[-1][0] + 1,
                current[-1][1],
                current[-1][2],
                current[-1][3] + 1,
            )


            candidates = (substitution, deletion, insertion)
            best_cost = min(candidate[0] for candidate in candidates)
            current.append(
                next(candidate for candidate in candidates if candidate[0] == best_cost)
            )
        previous = current
    cost, substitutions, deletions, insertions = previous[-1]
    assert cost == substitutions + deletions + insertions
    return {
        "edits": cost,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def _log_add(*values: float) -> float:
    finite = [value for value in values if value != -math.inf]
    if not finite:
        return -math.inf
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def ctc_prefix_beam_search(
    log_probs: torch.Tensor,
    *,
    blank_id: int = 0,
    beam_size: int = 10,
    token_topk: int = 32,
) -> list[int]:

    if log_probs.ndim != 2:
        raise ValueError(
            f"log_probs must have shape [T, V]; received {tuple(log_probs.shape)}"
        )
    beam: dict[tuple[int, ...], tuple[float, float]] = {
        (): (0.0, -math.inf)
    }
    topk = min(int(token_topk), log_probs.shape[-1])
    for frame in log_probs.detach().float().cpu():
        values, indices = torch.topk(frame, k=topk)
        candidates = list(zip(indices.tolist(), values.tolist()))
        if blank_id not in indices:
            candidates.append((blank_id, float(frame[blank_id])))
        next_beam: dict[tuple[int, ...], tuple[float, float]] = {}

        def update(
            prefix: tuple[int, ...],
            blank_score: float = -math.inf,
            nonblank_score: float = -math.inf,
        ) -> None:
            old_blank, old_nonblank = next_beam.get(
                prefix, (-math.inf, -math.inf)
            )
            next_beam[prefix] = (
                _log_add(old_blank, blank_score),
                _log_add(old_nonblank, nonblank_score),
            )

        for prefix, (prob_blank, prob_nonblank) in beam.items():
            total = _log_add(prob_blank, prob_nonblank)
            for token, token_log_prob in candidates:
                if token == blank_id:
                    update(prefix, blank_score=total + token_log_prob)
                    continue
                end_token = prefix[-1] if prefix else None
                if token == end_token:

                    update(
                        prefix,
                        nonblank_score=prob_nonblank + token_log_prob,
                    )

                    repeated = (*prefix, token)
                    update(
                        repeated,
                        nonblank_score=prob_blank + token_log_prob,
                    )
                else:
                    extended = (*prefix, token)
                    update(
                        extended,
                        nonblank_score=total + token_log_prob,
                    )
        beam = dict(
            sorted(
                next_beam.items(),
                key=lambda item: _log_add(*item[1]),
                reverse=True,
            )[:beam_size]
        )
    best = max(beam.items(), key=lambda item: _log_add(*item[1]))[0]
    return list(best)
