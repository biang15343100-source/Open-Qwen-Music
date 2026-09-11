
from __future__ import annotations

from typing import Any, Iterable

from .base import LocalStage
from ..schema import (
    SECTION_LABELS,
    VOCAL_SECTION_LABELS,
    Section,
    serialize_structured_lyrics,
)
from ..timeline import (
    Interval,
    TextUnit,
    assign_units_to_sections,
    choose_boundary,
    coverage_ratio,
    join_units,
    adaptive_gap_threshold,
    group_into_phrases,
    group_same_label_runs,
    merge_short_sections,
    snap_to_quietest_gap,
    strength_at,
)


_MIN_SPAN_SEC = 1e-3


_LABEL_ALIASES: dict[str, str] = {
    "intro": "intro",
    "verse": "verse",
    "pre-chorus": "verse",
    "prechorus": "verse",
    "pre_chorus": "verse",
    "chorus": "chorus",
    "post-chorus": "chorus",
    "postchorus": "chorus",
    "post_chorus": "chorus",
    "hook": "chorus",
    "refrain": "chorus",
    "bridge": "bridge",
    "inst": "inst",
    "instrumental": "inst",
    "solo": "inst",
    "break": "inst",
    "interlude": "inst",
    "breakdown": "inst",
    "outro": "outro",
    "end": "outro",
    "ending": "outro",
    "silence": "silence",
    "start": "silence",
}


def normalize_label(raw: str) -> str:

    token = str(raw or "").strip().lower().replace("_", "-")

    stripped = token.rstrip("0123456789 -")
    for candidate in (token, stripped, token.replace("-", "")):
        if candidate in _LABEL_ALIASES:
            return _LABEL_ALIASES[candidate]
    return "inst"


class SectionsStage(LocalStage):
    name = "sections"


    depends_on = ("structure", "align", "lyrics", "separate", "index")

    def iter_inputs(self) -> Iterable[dict[str, Any]]:
        index = self.upstream("index")
        structure = self.upstream("structure")
        align = self.upstream("align")
        lyrics = self.upstream("lyrics")
        separate = self.upstream("separate")

        for sample_id, record in sorted(structure.items()):
            yield {
                "sample_id": sample_id,
                "structure": record,
                "align": align.get(sample_id) or {},
                "lyrics": lyrics.get(sample_id) or {},
                "separate": separate.get(sample_id) or {},
                "index": index.get(sample_id) or {},
            }

    def process(self, item: dict[str, Any]) -> dict[str, Any] | None:
        duration = float(
            item["index"].get("duration_sec")
            or item["structure"].get("duration_sec")
            or 0.0
        )
        if duration <= 0:
            return {"sample_id": item["sample_id"], "error": "audio duration is missing"}

        raw_sections = item["structure"].get("sections") or []
        spans = self._normalize_spans(raw_sections, duration)
        if not spans:
            return {"sample_id": item["sample_id"], "error": "structure model produced no sections"}

        units, dropped_units = self._text_units(item["align"])
        language = (
            item["lyrics"].get("language")
            or item["index"].get("language_hint")
            or "en"
        )
        vocal_intervals = self._vocal_intervals(item["separate"])
        anchors = self._strength_anchors(raw_sections)

        sections = self._build_sections(
            spans, units, language, vocal_intervals, anchors
        )
        structured = serialize_structured_lyrics(sections)

        vocal_sections = [s for s in sections if s.has_lyrics]
        placed = sum(s.unit_count for s in sections)


        outside = sum(
            1 for unit in units if unit.midpoint < 0.0 or unit.midpoint > duration
        )
        return {
            "sample_id": item["sample_id"],
            "duration_sec": round(duration, 3),
            "language": language,
            "sections": [s.to_dict() for s in sections],
            "structured_lyrics": structured,
            "section_count": len(sections),
            "lyric_section_count": len(vocal_sections),


            "align_units": len(units),
            "placed_units": placed,
            "dropped_units": dropped_units,


            "orphan_units": len(units) - placed,


            "units_outside_duration": outside,
            "distinct_labels": sorted({s.label for s in sections}),


            "lyric_coverage": round(
                sum(s.lyric_coverage * s.duration_sec for s in sections)
                / max(duration, 1e-6),
                4,
            ),
            "lyrics_tier": item["lyrics"].get("tier", "none"),
            "align_available": bool(units),
        }

    def _normalize_spans(
        self, raw_sections: list[dict[str, Any]], duration: float
    ) -> list[tuple[str, float, float]]:

        spans: list[tuple[str, float, float]] = []
        for entry in raw_sections:
            try:
                start = max(0.0, float(entry["start"]))
                end = min(duration, float(entry["end"]))
            except (KeyError, TypeError, ValueError):
                continue
            if end - start <= 1e-3:
                continue
            spans.append((normalize_label(entry.get("label", "")), start, end))

        if not spans:
            return []
        spans.sort(key=lambda item: item[1])


        cleaned: list[tuple[str, float, float]] = []
        for label, start, end in spans:
            if cleaned and start < cleaned[-1][2]:
                start = cleaned[-1][2]
            if end - start <= 1e-3:
                continue
            cleaned.append((label, start, end))

        return merge_short_sections(
            cleaned,
            min_duration_sec=float(self.config.get("min_section_sec", 2.0)),
        )

    def _text_units(self, align: dict[str, Any]) -> tuple[list[TextUnit], int]:

        units: list[TextUnit] = []
        dropped = 0
        for entry in align.get("units") or []:
            try:
                text = str(entry["text"]).strip()
                start = float(entry["start"])
                end = float(entry["end"])
            except (KeyError, TypeError, ValueError):
                dropped += 1
                continue
            if not text or end < start:
                dropped += 1
                continue
            units.append(TextUnit(text=text, start_sec=start, end_sec=end))
        units.sort(key=lambda unit: unit.start_sec)
        return units, dropped

    def _vocal_intervals(self, separate: dict[str, Any]) -> list[tuple[float, float]]:

        intervals: list[tuple[float, float]] = []
        for entry in separate.get("vocal_intervals") or []:
            try:
                start, end = float(entry[0]), float(entry[1])
            except (TypeError, ValueError, IndexError):
                continue
            if end > start:
                intervals.append((start, end))
        return intervals

    def _strength_anchors(
        self, raw_sections: list[dict[str, Any]]
    ) -> list[tuple[float, float]]:

        anchors: list[tuple[float, float]] = []
        for entry in raw_sections:
            value = entry.get("boundary_strength")
            if value is None:
                continue
            try:
                anchors.append((float(entry["start"]), float(value)))
            except (KeyError, TypeError, ValueError):
                continue
        return anchors

    def _snap_spans_to_phrases(
        self,
        spans: list[tuple[str, float, float]],
        vocal_intervals: list[tuple[float, float]],
    ) -> list[tuple[str, float, float]]:

        tolerance = float(self.config.get("boundary_snap_sec", 1.2))
        if tolerance <= 0 or len(spans) < 2:
            return spans

        silences = [
            ((current[1] + following[0]) / 2.0, following[0] - current[1])
            for current, following in zip(vocal_intervals, vocal_intervals[1:])
            if following[0] > current[1]
        ]
        if not silences:
            return spans

        internal = [start for _, start, _ in spans[1:]]
        snapped = snap_to_quietest_gap(internal, silences, tolerance_sec=tolerance)


        result: list[tuple[str, float, float]] = []
        previous_end = spans[0][1]
        for index, (label, _, end) in enumerate(spans):
            start = previous_end if index == 0 else snapped[index - 1]
            start = max(start, previous_end)
            limit = spans[index][2]
            if start >= limit:
                start = previous_end
            result.append((label, start, max(end, start)))
            previous_end = result[-1][2]


        for index in range(len(result) - 1):
            label, start, _ = result[index]
            result[index] = (label, start, result[index + 1][1])
        return [item for item in result if item[2] > item[1]]

    def _reconcile_boundaries(
        self,
        blocks: list[tuple[str, float, float, list[list[TextUnit]]]],
    ) -> list[tuple[str, float, float, list[list[TextUnit]]]]:

        limit = float(self.config.get("boundary_reconcile_max_sec", 0.0) or 0.0)
        if limit <= 0 or len(blocks) < 2:
            return blocks

        working = [list(block) for block in blocks]
        for index in range(len(working) - 1):
            left, right = working[index], working[index + 1]
            left_units = [unit for phrase in left[3] for unit in phrase]
            right_units = [unit for phrase in right[3] for unit in phrase]
            if not left_units or not right_units:
                continue
            if not self._has_lyrics(left_units, left[1], left[2]):
                continue
            if not self._has_lyrics(right_units, right[1], right[2]):
                continue

            last_end = max(unit.end_sec for unit in left_units)
            next_start = min(unit.start_sec for unit in right_units)
            if last_end > next_start:
                continue

            boundary = float(left[2])
            if last_end <= boundary <= next_start:
                continue

            low = max(last_end, boundary - limit, left[1] + _MIN_SPAN_SEC)
            high = min(next_start, boundary + limit, right[2] - _MIN_SPAN_SEC)
            if high < low:
                continue

            moved = choose_boundary(boundary, low=low, high=high)
            left[2] = moved
            right[1] = moved

        return [tuple(block) for block in working]  # type: ignore[misc]

    def _rescue_starved_sections(
        self,
        phrases: list[list[TextUnit]],
        phrase_spans: list[TextUnit],
        intervals: list[Interval],
        buckets: list[list[int]],
    ) -> tuple[list[list[TextUnit]], list[TextUnit], list[list[int]]]:

        if not intervals:
            return phrases, phrase_spans, buckets

        min_units = int(self.config.get("min_section_units", 2))
        min_coverage = float(self.config.get("min_section_coverage", 0.05))


        for _ in range(4):
            units = [unit for phrase in phrases for unit in phrase]
            cuts: dict[int, set[float]] = {}
            for index, occupants in enumerate(buckets):
                if occupants:
                    continue
                start, end = intervals[index].start_sec, intervals[index].end_sec
                inside = [unit for unit in units if start <= unit.midpoint < end]
                if len(inside) < min_units or coverage_ratio(inside, start, end) < min_coverage:
                    continue
                for position, span in enumerate(phrase_spans):
                    if span.start_sec >= end or span.end_sec <= start:
                        continue
                    owner = next(
                        (k for k, members in enumerate(buckets) if position in members),
                        None,
                    )
                    if owner is not None:
                        owned = [
                            unit for member in buckets[owner] for unit in phrases[member]
                        ]
                        remainder = [
                            unit
                            for unit in owned
                            if not (start <= unit.midpoint < end)
                        ]


                        if self._has_lyrics(
                            owned,
                            intervals[owner].start_sec,
                            intervals[owner].end_sec,
                        ) and not self._has_lyrics(
                            remainder,
                            intervals[owner].start_sec,
                            intervals[owner].end_sec,
                        ):
                            continue
                    cuts.setdefault(position, set()).update(
                        value
                        for value in (start, end)
                        if span.start_sec < value < span.end_sec
                    )
            if not cuts:
                break

            rebuilt: list[list[TextUnit]] = []
            for position, phrase in enumerate(phrases):
                rebuilt.extend(_split_phrase(phrase, cuts[position]) if position in cuts else [phrase])
            if len(rebuilt) == len(phrases):
                break
            phrases = rebuilt
            phrase_spans = [
                TextUnit(text="", start_sec=phrase[0].start_sec, end_sec=phrase[-1].end_sec)
                for phrase in phrases
            ]
            buckets = assign_units_to_sections(phrase_spans, intervals)

        return phrases, phrase_spans, buckets

    def _build_sections(
        self,
        spans: list[tuple[str, float, float]],
        units: list[TextUnit],
        language: str,
        vocal_intervals: list[tuple[float, float]],
        anchors: list[tuple[float, float]] | None = None,
    ) -> list[Section]:
        spans = self._snap_spans_to_phrases(spans, vocal_intervals)
        intervals = [Interval(start, end) for _, start, end in spans]


        min_silent = float(self.config.get("break_min_silent_ratio", 0.5))
        gap = adaptive_gap_threshold(
            units,
            percentile=float(self.config.get("line_gap_percentile", 0.75)),
            floor_sec=float(self.config.get("line_gap_sec", 0.30)),
            ceiling_sec=float(self.config.get("line_gap_max_sec", 1.50)),
            voiced=vocal_intervals,
            min_silent_ratio=min_silent,
        )


        phrases = group_into_phrases(
            units, gap_sec=gap, voiced=vocal_intervals, min_silent_ratio=min_silent
        )

        #


        #


        #


        if self.config.get("split_phrases_at_boundaries", False):
            cuts = {start for _, start, _ in spans[1:]}
            min_units = int(self.config.get("min_section_units", 2))
            split: list[list[TextUnit]] = []
            for phrase in phrases:
                parts = _split_phrase(phrase, cuts)
                if len(parts) > 1 and all(len(part) >= min_units for part in parts):
                    split.extend(parts)
                else:
                    split.append(phrase)
            phrases = split
        phrase_spans = [
            TextUnit(text="", start_sec=phrase[0].start_sec, end_sec=phrase[-1].end_sec)
            for phrase in phrases
        ]
        buckets = assign_units_to_sections(phrase_spans, intervals)
        phrases, phrase_spans, buckets = self._rescue_starved_sections(
            phrases, phrase_spans, intervals, buckets
        )

        vocal_energy_threshold = float(self.config.get("min_vocal_overlap", 0.15))


        label_threshold = float(self.config.get("min_vocal_label_overlap", 0.5))


        labels: list[str] = []
        for (label, start, end), indices in zip(spans, buckets):
            member_units = [unit for i in indices for unit in phrases[i]]


            sung = self._has_lyrics(member_units, start, end) or (
                _overlap_ratio(vocal_intervals, start, end) >= label_threshold
            )


            if label in VOCAL_SECTION_LABELS and not sung:
                label = "inst"
            labels.append(label if label in SECTION_LABELS else "inst")


        unit_spans: list[tuple[float, float] | None] = []
        for indices in buckets:
            owned = [unit for i in indices for unit in phrases[i]]
            unit_spans.append(
                (
                    min(unit.start_sec for unit in owned),
                    max(unit.end_sec for unit in owned),
                )
                if owned
                else None
            )

        groups = self._merge_groups(
            labels, [(s, e) for _, s, e in spans], anchors or [], unit_spans
        )

        blocks = [
            (
                labels[group[0]],
                spans[group[0]][1],
                spans[group[-1]][2],
                [phrases[i] for g in group for i in buckets[g]],
            )
            for group in groups
        ]
        blocks = self._reconcile_boundaries(blocks)
        blocks = self._split_silent_tails(blocks, vocal_intervals)
        blocks = self._split_silent_heads(blocks, vocal_intervals)

        sections: list[Section] = []
        for position, (label, start, end, member_phrases) in enumerate(blocks):
            member_units = [unit for phrase in member_phrases for unit in phrase]
            coverage = coverage_ratio(member_units, start, end)
            has_lyrics = self._has_lyrics(member_units, start, end)

            text = (
                "\n".join(
                    line
                    for line in (
                        join_units(phrase, language=language)
                        for phrase in member_phrases
                    )
                    if line
                )
                if has_lyrics
                else ""
            )

            is_vocal = has_lyrics or (
                _overlap_ratio(vocal_intervals, start, end) >= vocal_energy_threshold
            )

            sections.append(
                Section(
                    section_id=f"s{position:03d}",
                    label=label,
                    start_sec=start,
                    end_sec=end,
                    has_lyrics=has_lyrics,
                    is_vocal=is_vocal,
                    lyrics=text,
                    unit_count=len(member_units),
                    lyric_coverage=coverage,
                )
            )
        return sections

    def _split_silent_heads(
        self,
        blocks: list[tuple[str, float, float, list[list[TextUnit]]]],
        vocal_intervals: list[tuple[float, float]],
    ) -> list[tuple[str, float, float, list[list[TextUnit]]]]:

        min_head = float(self.config.get("min_section_sec", 2.0))
        if min_head <= 0 or not blocks:
            return blocks

        working = [list(block) for block in blocks]


        first = next(
            (
                i
                for i, block in enumerate(working)
                if [u for phrase in block[3] for u in phrase]
                and self._has_lyrics(
                    [u for phrase in block[3] for u in phrase], block[1], block[2]
                )
            ),
            None,
        )
        if first is None:
            return blocks

        label, start, end, member_phrases = working[first]
        units = [unit for phrase in member_phrases for unit in phrase]
        voiced_start = min(
            (
                max(interval_start, start)
                for interval_start, interval_end in vocal_intervals
                if interval_end > start and interval_start < end
            ),
            default=float("inf"),
        )
        sing_start = min(min(unit.start_sec for unit in units), voiced_start)

        if sing_start - start >= min_head:
            if first > 0:
                working[first][1] = sing_start
                working[first - 1][2] = sing_start
            else:
                working[first][1] = sing_start
                working.insert(first, ["intro", start, sing_start, []])

        return [tuple(block) for block in working]  # type: ignore[misc]

    def _split_silent_tails(
        self,
        blocks: list[tuple[str, float, float, list[list[TextUnit]]]],
        vocal_intervals: list[tuple[float, float]],
    ) -> list[tuple[str, float, float, list[list[TextUnit]]]]:

        min_tail = float(self.config.get("min_section_sec", 2.0))
        if min_tail <= 0 or not blocks:
            return blocks

        working = [list(block) for block in blocks]
        index = 0
        while index < len(working):
            label, start, end, member_phrases = working[index]
            units = [unit for phrase in member_phrases for unit in phrase]
            if not units or not self._has_lyrics(units, start, end):
                index += 1
                continue

            voiced_end = max(
                (
                    min(interval_end, end)
                    for interval_start, interval_end in vocal_intervals
                    if interval_end > start and interval_start < end
                ),
                default=0.0,
            )
            last_word = max(max(unit.end_sec for unit in units), voiced_end)

            following = working[index + 1] if index + 1 < len(working) else None
            following_sung = following is not None and self._has_lyrics(
                [u for phrase in following[3] for u in phrase], following[1], following[2]
            )

            is_final_sung = not any(
                self._has_lyrics(
                    [u for phrase in later[3] for u in phrase], later[1], later[2]
                )
                for later in working[index + 1 :]
            )
            if (
                last_word > end
                and not following_sung
                and is_final_sung
                and following is not None
                and following[0] in {"outro", "silence"}
            ):


                if following is None:
                    index += 1
                    continue
                working[index][2] = min(last_word, following[2])
                following[1] = min(last_word, following[2])
                index += 1
                continue


            floor = 0.0 if (is_final_sung and following is not None) else min_tail
            if end - last_word < floor or last_word <= start:
                index += 1
                continue

            if following is not None:
                if following_sung:

                    index += 1
                    continue
                working[index][2] = last_word
                following[1] = last_word
            else:
                working[index][2] = last_word
                working.insert(index + 1, ["outro", last_word, end, []])
                index += 1
            index += 1


        return [tuple(block) for block in working if block[2] > block[1]]  # type: ignore[misc]

    def _has_lyrics(self, units: list[TextUnit], start: float, end: float) -> bool:

        min_units = int(self.config.get("min_section_units", 2))
        min_coverage = float(self.config.get("min_section_coverage", 0.05))
        return len(units) >= min_units and coverage_ratio(units, start, end) >= min_coverage

    def _merge_groups(
        self,
        labels: list[str],
        bounds: list[tuple[float, float]],
        anchors: list[tuple[float, float]],
        unit_spans: list[tuple[float, float] | None] | None = None,
    ) -> list[list[int]]:

        if not self.config.get("merge_same_label", True):
            return [[index] for index in range(len(labels))]

        max_strength = self.config.get("same_label_max_strength")
        weak_strength = self.config.get("same_label_weak_strength")
        strengths: list[float | None] | None = None
        if (max_strength is not None or weak_strength is not None) and anchors:
            tolerance = float(self.config.get("boundary_snap_sec", 1.2)) + 0.3
            strengths = [
                strength_at(anchors, start, tolerance_sec=tolerance)
                for start, _ in bounds
            ]

        return group_same_label_runs(
            labels,
            bounds,
            max_merged_sec=float(self.config.get("same_label_max_sec", 30.0)),
            capped_labels=VOCAL_SECTION_LABELS,
            strengths=strengths,
            max_strength=None if max_strength is None else float(max_strength),
            weak_strength=None if weak_strength is None else float(weak_strength),
            unit_spans=unit_spans,
            reconcile_max_sec=float(
                self.config.get("boundary_reconcile_max_sec", 0.0) or 0.0
            ),
        )


def _split_phrase(phrase: list[TextUnit], cuts: set[float]) -> list[list[TextUnit]]:

    parts: list[list[TextUnit]] = []
    current: list[TextUnit] = []
    ordered = sorted(cuts)
    position = 0
    for unit in phrase:
        while position < len(ordered) and unit.midpoint >= ordered[position]:
            if current:
                parts.append(current)
                current = []
            position += 1
        current.append(unit)
    if current:
        parts.append(current)
    return [part for part in parts if part]


def _overlap_ratio(
    intervals: list[tuple[float, float]], start: float, end: float
) -> float:
    span = max(0.0, end - start)
    if span <= 0 or not intervals:
        return 0.0
    total = 0.0
    for left, right in intervals:
        total += max(0.0, min(end, right) - max(start, left))
    return min(1.0, total / span)
