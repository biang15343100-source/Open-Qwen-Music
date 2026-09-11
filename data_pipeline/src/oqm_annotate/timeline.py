
from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Iterable, Sequence


@dataclass(slots=True)
class TextUnit:

    text: str
    start_sec: float
    end_sec: float

    @property
    def midpoint(self) -> float:
        return (self.start_sec + self.end_sec) / 2.0

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass(slots=True)
class Interval:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:

    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_units_to_sections(
    units: Sequence[TextUnit],
    sections: Sequence[Interval],
) -> list[list[int]]:

    buckets: list[list[int]] = [[] for _ in sections]
    if not sections:
        return buckets

    for index, unit in enumerate(units):
        target = _locate(unit.midpoint, sections)
        if target is not None:
            buckets[target].append(index)
    return buckets


_ORPHAN_TOLERANCE = 2.0


def _locate(position: float, sections: Sequence[Interval]) -> int | None:

    best_index: int | None = None
    best_distance = float("inf")
    for index, section in enumerate(sections):
        if section.start_sec <= position < section.end_sec:
            return index
        if position < section.start_sec:
            distance = section.start_sec - position
        else:
            distance = position - section.end_sec
        if distance < best_distance:
            best_distance = distance
            best_index = index
    if best_index is not None and best_distance <= _ORPHAN_TOLERANCE:
        return best_index
    return None


def coverage_ratio(units: Iterable[TextUnit], start_sec: float, end_sec: float) -> float:

    span = max(0.0, end_sec - start_sec)
    if span <= 0:
        return 0.0
    clipped: list[tuple[float, float]] = []
    for unit in units:
        left = max(unit.start_sec, start_sec)
        right = min(unit.end_sec, end_sec)
        if right > left:
            clipped.append((left, right))
    if not clipped:
        return 0.0
    clipped.sort()
    merged_total = 0.0
    current_start, current_end = clipped[0]
    for left, right in clipped[1:]:
        if left > current_end:
            merged_total += current_end - current_start
            current_start, current_end = left, right
        else:
            current_end = max(current_end, right)
    merged_total += current_end - current_start
    return min(1.0, merged_total / span)


_SCRIPTLESS_JOIN_LANGS = frozenset({"zh", "ja", "ko", "yue"})


#


_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x2E80, 0x2FDF),
    (0x3040, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),
    (0xF900, 0xFAFF),
    (0xFF66, 0xFF9F),
    (0x20000, 0x3FFFF),
)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def _unit_is_cjk(text: str, *, scriptless_is_cjk: bool) -> bool:

    if any(_is_cjk(char) for char in text):
        return True
    if any(char.isalpha() for char in text):
        return False
    return scriptless_is_cjk


def join_units(units: Sequence[TextUnit], *, language: str) -> str:

    texts = [unit.text.strip() for unit in units if unit.text.strip()]
    if not texts:
        return ""
    scriptless_is_cjk = language in _SCRIPTLESS_JOIN_LANGS
    flags = [_unit_is_cjk(text, scriptless_is_cjk=scriptless_is_cjk) for text in texts]
    pieces = [texts[0]]
    for index in range(1, len(texts)):


        if not flags[index - 1] and not flags[index]:
            pieces.append(" ")
        pieces.append(texts[index])
    return "".join(pieces)


def silent_ratio(
    voiced: Sequence[tuple[float, float]], start: float, end: float
) -> float:

    span = end - start
    if span <= 0:
        return 0.0
    sung = sum(
        max(0.0, min(end, right) - max(start, left)) for left, right in voiced
    )
    return max(0.0, min(1.0, (span - sung) / span))


def group_into_phrases(
    units: Sequence[TextUnit],
    *,
    gap_sec: float = 0.7,
    max_units_per_line: int = 24,
    voiced: Sequence[tuple[float, float]] | None = None,
    min_silent_ratio: float = 0.5,
) -> list[list[TextUnit]]:

    if not units:
        return []
    phrases: list[list[TextUnit]] = []
    current: list[TextUnit] = [units[0]]

    def breakable(previous: TextUnit, unit: TextUnit) -> bool:
        if unit.start_sec - previous.end_sec < gap_sec:
            return False
        if not voiced:
            return True
        return (
            silent_ratio(voiced, previous.end_sec, unit.start_sec) >= min_silent_ratio
        )

    for previous, unit in zip(units, units[1:]):
        if breakable(previous, unit):
            phrases.append(current)
            current = [unit]
            continue
        current.append(unit)
        if len(current) <= max_units_per_line:
            continue


        #


        lower = max(1, max_units_per_line // 2)

        def split_score(index: int) -> tuple[float, float, int]:
            left, right = current[index - 1].end_sec, current[index].start_sec
            silent = (
                silent_ratio(voiced, left, right) * (right - left) if voiced else 0.0
            )
            return (silent, right - left, index)

        split_at = max(range(lower, len(current)), key=split_score)
        phrases.append(current[:split_at])
        current = current[split_at:]
    phrases.append(current)
    return [phrase for phrase in phrases if phrase]


def phrase_gaps(
    units: Sequence[TextUnit],
    *,
    min_gap_sec: float = 0.12,
) -> list[tuple[float, float]]:

    gaps: list[tuple[float, float]] = []
    for current, following in zip(units, units[1:]):
        length = following.start_sec - current.end_sec
        if length >= min_gap_sec:
            gaps.append(((current.end_sec + following.start_sec) / 2.0, length))
    return gaps


def adaptive_gap_threshold(
    units: Sequence[TextUnit],
    *,
    percentile: float = 0.75,
    floor_sec: float = 0.30,
    ceiling_sec: float = 1.50,
    voiced: Sequence[tuple[float, float]] | None = None,
    min_silent_ratio: float = 0.5,
) -> float:

    gaps = phrase_gaps(units)
    if voiced:

        gaps = [
            (midpoint, length)
            for midpoint, length in gaps
            if silent_ratio(voiced, midpoint - length / 2, midpoint + length / 2)
            >= min_silent_ratio
        ]
    lengths = sorted(length for _, length in gaps)
    if not lengths:
        return floor_sec
    index = min(len(lengths) - 1, max(0, int(round(percentile * (len(lengths) - 1)))))
    return max(floor_sec, min(ceiling_sec, lengths[index]))


def snap_to_quietest_gap(
    boundaries: Sequence[float],
    gaps: Sequence[tuple[float, float]],
    *,
    tolerance_sec: float = 1.2,
) -> list[float]:

    if not gaps:
        return list(boundaries)
    snapped: list[float] = []
    for boundary in boundaries:
        nearby = [
            (length, midpoint)
            for midpoint, length in gaps
            if abs(midpoint - boundary) <= tolerance_sec
        ]
        snapped.append(max(nearby)[1] if nearby else boundary)
    return snapped


def choose_boundary(current: float, *, low: float, high: float) -> float:

    if high < low:
        return current
    if current < low:
        return low
    return min(current, high)


def strength_at(
    anchors: Sequence[tuple[float, float]],
    moment: float,
    *,
    tolerance_sec: float,
) -> float | None:

    best: float | None = None
    best_distance = tolerance_sec
    for anchor, value in anchors:
        distance = abs(anchor - moment)
        if distance <= best_distance:
            best, best_distance = value, distance
    return best


def _assertable(
    previous: Sequence[int],
    index: int,
    bounds: Sequence[tuple[float, float]],
    unit_spans: Sequence[tuple[float, float] | None] | None,
    reconcile_max_sec: float,
) -> bool:

    if unit_spans is None:
        return True

    right = unit_spans[index]
    left_ends = [
        unit_spans[i][1] for i in previous if unit_spans[i] is not None  # type: ignore[index]
    ]
    if right is None or not left_ends:
        return False

    last_end = max(left_ends)
    next_start = right[0]
    if last_end > next_start:
        return False

    boundary = bounds[index][0]
    if last_end <= boundary <= next_start:
        return True
    return max(last_end - boundary, boundary - next_start) <= reconcile_max_sec


def group_same_label_runs(
    labels: Sequence[str],
    bounds: Sequence[tuple[float, float]],
    *,
    max_merged_sec: float,
    capped_labels: Collection[str] = (),
    strengths: Sequence[float | None] | None = None,
    max_strength: float | None = None,
    weak_strength: float | None = None,
    unit_spans: Sequence[tuple[float, float] | None] | None = None,
    reconcile_max_sec: float = 0.0,
) -> list[list[int]]:

    groups: list[list[int]] = []
    for index, label in enumerate(labels):
        if groups:
            previous = groups[-1]
            if labels[previous[0]] == label:
                merged = bounds[index][1] - bounds[previous[0]][0]
                short_enough = label not in capped_labels or merged <= max_merged_sec
                value = (
                    strengths[index]
                    if strengths is not None and index < len(strengths)
                    else None
                )
                if (
                    value is not None
                    and max_strength is not None
                    and value >= max_strength
                    and _assertable(
                        previous,
                        index,
                        bounds,
                        unit_spans,
                        reconcile_max_sec,
                    )
                ):
                    merge = False
                elif value is not None and weak_strength is not None and value <= weak_strength:
                    merge = True
                else:
                    merge = short_enough
                if merge:
                    previous.append(index)
                    continue
        groups.append([index])
    return groups


def merge_short_sections(
    sections: list[tuple[str, float, float]],
    *,
    min_duration_sec: float = 2.0,
) -> list[tuple[str, float, float]]:

    if not sections:
        return []
    result = [list(item) for item in sections]
    changed = True
    while changed and len(result) > 1:
        changed = False
        for index, (_, start, end) in enumerate(result):
            if end - start >= min_duration_sec:
                continue
            previous = result[index - 1] if index > 0 else None
            following = result[index + 1] if index + 1 < len(result) else None
            if previous is None and following is None:
                break
            if previous is None:
                target = index + 1
            elif following is None:
                target = index - 1
            else:
                prev_len = previous[2] - previous[1]
                next_len = following[2] - following[1]
                target = index - 1 if prev_len >= next_len else index + 1
            if target < index:
                result[target][2] = result[index][2]
            else:
                result[target][1] = result[index][1]
            result.pop(index)
            changed = True
            break
    return [(str(label), float(start), float(end)) for label, start, end in result]
