
from __future__ import annotations

import os
import statistics
from functools import lru_cache
from typing import Any, Iterable

from .base import LocalStage, WorkerStage
from .index import VOICING_HUMAN, VOICING_INSTRUMENT
from ..text import (
    detect_language,
    error_rate,
    is_cjk_text,
    looping_ratio,
    repeat_ngram_ratio,
    script_evidence,
    tokenize,
)


class AsrVocalStage(WorkerStage):

    name = "asr.vocal"
    depends_on = ("separate",)
    script = "w_asr.py"

    def select_inputs(self) -> list[dict[str, Any]]:
        index = self.upstream("index")
        rows = self.upstream("separate")
        items: list[dict[str, Any]] = []
        for sample_id, record in sorted(rows.items()):
            stem = record.get("vocal_16k_path")
            if not stem:
                continue
            meta = index.get(sample_id, {})
            items.append(
                {
                    "sample_id": sample_id,
                    "audio_path": stem,
                    "duration_sec": record.get("duration_sec", 0.0),
                    "language_hint": meta.get("language_hint"),
                    "vocal_ratio": record.get("vocal_ratio"),

                    "vocal_intervals": record.get("vocal_intervals") or [],
                }
            )
        return items


class AsrMixStage(WorkerStage):

    name = "asr.mix"


    depends_on = ("index", "separate")
    script = "w_asr.py"

    def select_inputs(self) -> list[dict[str, Any]]:
        rows = self.upstream("index")
        separate = self.upstream("separate")
        return [
            {
                "sample_id": sample_id,
                "audio_path": record["audio_path"],
                "duration_sec": record.get("duration_sec", 0.0),
                "language_hint": record.get("language_hint"),
                "vocal_intervals": (separate.get(sample_id) or {}).get(
                    "vocal_intervals"
                )
                or [],
            }
            for sample_id, record in sorted(rows.items())
        ]


_NO_SPACE_LANGUAGES = frozenset({"zh", "ja", "ko", "yue"})


def _strong_script_evidence(text: str) -> str | None:

    return script_evidence(text)


def _resolve_language(
    text: str, primary: dict[str, Any], secondary: dict[str, Any]
) -> str:

    def no_space(code: str) -> bool:
        return str(code).strip().lower()[:3] in _NO_SPACE_LANGUAGES

    reported = str(primary.get("language") or "").strip()
    second = str(secondary.get("language") or "").strip()
    if not text.strip():
        return reported or second or detect_language(text)


    evidence = _strong_script_evidence(text)
    if evidence:
        return evidence

    if not reported:
        return detect_language(text)


    if not second:
        return reported

    by_text = is_cjk_text(text)
    if by_text != no_space(reported) and no_space(second) == by_text:

        return second
    return reported


def _char_count(text: str) -> int:

    return len(tokenize(text))


def voiced_sec(span: tuple[float, float], intervals: Any) -> float:

    start, end = span
    total = 0.0
    for item in intervals or []:
        try:
            left, right = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        low, high = max(start, left), min(end, right)
        if high > low:
            total += high - low
    return total


def _span_of(segment: dict[str, Any]) -> tuple[float, float] | None:
    try:
        return float(segment["start"]), float(segment["end"])
    except (KeyError, TypeError, ValueError):
        return None


def pair_windows(
    primary: list[dict[str, Any]], secondary: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:

    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for segment in primary:
        span = _span_of(segment)
        if span is None:
            pairs.append((segment, None))
            continue
        best: dict[str, Any] | None = None
        best_overlap = 0.0
        for other in secondary:
            other_span = _span_of(other)
            if other_span is None:
                continue
            low = max(span[0], other_span[0])
            high = min(span[1], other_span[1])
            overlap = high - low
            if overlap <= 0:
                continue
            shorter = min(span[1] - span[0], other_span[1] - other_span[0])
            if shorter <= 0 or overlap < 0.5 * shorter:
                continue
            if overlap > best_overlap:
                best, best_overlap = other, overlap
        pairs.append((segment, best))
    return pairs


def window_densities(
    segments: list[dict[str, Any]], intervals: Any, *, min_voiced_sec: float
) -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []
    for segment in segments:
        span = _span_of(segment)
        if span is None:
            continue
        voiced = voiced_sec(span, intervals)
        chars = _char_count(str(segment.get("text") or ""))
        rows.append(
            {
                "start": span[0],
                "end": span[1],
                "voiced_sec": round(voiced, 2),
                "chars": chars,
                "density": (chars / voiced) if voiced > 0 else None,
                "usable": voiced >= min_voiced_sec,
            }
        )
    return rows


#


#


#


#

# (`'Ooh.'` -> `'Ah ah'`,`'Ah ah'` -> `'Ah ah ah ah ah ah ah ah ah.'`),


_WINDOW_PATH_MAX_LOOPING = 0.5


#


#

#               (`'Ah ah'` -> `"I've thought enough. Should I turn this around?"`


#


#


_WINDOW_PATH_MIN_GAIN_MULTIPLE = 3.0


def select_window_path(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    intervals: Any,
    *,
    mode: str,
    flagged: set[tuple[float, float]] | None = None,
    max_looping: float = _WINDOW_PATH_MAX_LOOPING,
    min_gain_multiple: float = _WINDOW_PATH_MIN_GAIN_MULTIPLE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:

    if mode == "off" or not secondary:
        return list(primary), {
            "switched_windows": 0,
            "switched_chars": 0,
            "looping_blocked": 0,
            "thin_gain_blocked": 0,
        }

    switched = 0
    gained = 0
    blocked = 0
    thin = 0
    out: list[dict[str, Any]] = []
    for segment, other in pair_windows(primary, secondary):
        span = _span_of(segment)
        take = False
        if other is not None and span is not None:
            mine = _char_count(str(segment.get("text") or ""))
            theirs = _char_count(str(other.get("text") or ""))
            if theirs > mine:
                if mode == "always":
                    take = True
                elif mode == "undertranscribed" and flagged is not None:
                    take = (round(span[0], 2), round(span[1], 2)) in flagged
            if take and theirs < min_gain_multiple * max(mine, 1):


                take = False
                thin += 1
            if take and looping_ratio(str(other.get("text") or "")) > max_looping:


                take = False
                blocked += 1
        if take:
            switched += 1
            gained += _char_count(str(other.get("text") or "")) - _char_count(
                str(segment.get("text") or "")
            )


            merged = dict(segment)
            merged["text"] = str(other.get("text") or "").strip()
            merged["path"] = "mix"
            out.append(merged)
        else:
            out.append(segment)
    return out, {
        "switched_windows": switched,
        "switched_chars": gained,
        "looping_blocked": blocked,
        "thin_gain_blocked": thin,
    }


def filter_degenerate_windows(
    segments: list[dict[str, Any]],
    *,
    min_char_rate: float,
    min_duration_sec: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for segment in segments:
        try:
            start = float(segment["start"])
            end = float(segment["end"])
            text = str(segment.get("text") or "")
        except (KeyError, TypeError, ValueError):
            continue
        span = end - start
        if span >= min_duration_sec and span > 0:
            if _char_count(text) / span < min_char_rate:
                dropped.append(segment)
                continue
        kept.append(segment)
    return kept, dropped


#

#


#


#

#


#

#     0.20      10          5        50.0%   50.0%
#     0.30      10          5        50.0%   50.0%
#     0.40      16          5        31.2%   50.0%
#     0.60      30          7        23.3%   70.0%
#


#


#


#


_UNDERTRANSCRIBED_MAX_RATIO = 0.30
_UNDERTRANSCRIBED_MIN_VOICED_SEC = 8.0


def _finalize_primary_text(
    primary: dict[str, Any], *, min_char_rate: float, min_duration_sec: float
) -> str:

    segments = list(primary.get("segments") or [])
    if not segments:
        return (primary.get("text") or "").strip()
    kept, _ = filter_degenerate_windows(
        segments, min_char_rate=min_char_rate, min_duration_sec=min_duration_sec
    )
    return " ".join(
        str(s.get("text") or "").strip()
        for s in kept
        if str(s.get("text") or "").strip()
    ).strip()


def undertranscribed_windows(
    rows: list[dict[str, Any]], *, max_ratio: float, min_voiced_sec: float
) -> tuple[list[dict[str, Any]], float | None]:

    usable = [r for r in rows if r.get("usable") and r.get("density") is not None]
    if len(usable) < 3:
        return [], None
    median = statistics.median(r["density"] for r in usable)
    if median <= 0:
        return [], median
    hits = [
        dict(r, ratio=round(r["density"] / median, 4))
        for r in usable
        if r["density"] / median <= max_ratio
    ]
    return hits, median


#


#


#   zh     190   2.32   5.17   8.73   27.50
#   ja      23   2.41   6.08  11.98   11.98
#   ko      19   2.27   3.75   3.75    3.75
#   hi      12   6.37  18.72  18.72   18.72


#


#


_CHAR_RATE_P99 = {
    "en": 33.78,
    "zh": 8.73,
    "ja": 11.98,
    "ko": 3.75,
    "hi": 18.72,
    "pl": 15.22,
    "de": 16.00,
    "it": 17.92,
    "nl": 12.75,
    "pt": 10.61,
}
_POOL_CHAR_RATE_P99 = 33.33


def untranscribed_voiced_sec(
    spans: Iterable[dict[str, Any]], start: float, end: float
) -> float:

    total = 0.0
    for span in spans or []:
        try:
            left, right = float(span["start"]), float(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        low, high = max(start, left), min(end, right)
        if high <= low:
            continue
        voiced = span.get("voiced_sec")
        window = right - left
        if voiced is None or window <= 0:
            total += high - low
            continue


        total += float(voiced) * (high - low) / window
    return total


@lru_cache(maxsize=1)
def _reference_window_sec() -> float:

    from ..config import _safe_align_sec

    return _safe_align_sec()


def window_char_cap(language: str | None, window_sec: float) -> float:

    rate = _CHAR_RATE_P99.get((language or "").lower())
    return window_sec * max(rate or 0.0, _POOL_CHAR_RATE_P99)


def hallucinated_windows(
    segments: list[dict[str, Any]], language: str | None, window_sec: float
) -> list[dict[str, Any]]:

    cap = window_char_cap(language, window_sec)
    hits: list[dict[str, Any]] = []
    for segment in segments:
        try:
            chars = len(str(segment.get("text") or ""))
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if chars > cap:
            hits.append(
                {"start": start, "end": end, "chars": chars, "cap": round(cap, 1)}
            )
    return hits


#


#


#     vocal_ratio     0.0617       0.7062            1.0000
#     stem_to_mix_db  -22.26       -3.78              0.01
#


#


#


#


#


#


#


#


#


#


_VOICE_MIN_RATIO = 0.70
_VOICE_MIN_STEM_DB = -6.0


_VOICE_NO_WORDS_MIN_RATIO = _VOICE_MIN_RATIO
_VOICE_NO_WORDS_MIN_STEM_DB = _VOICE_MIN_STEM_DB


#


#


#


#


_STEM_IS_MIX_DB = -3.0


def _stem_is_the_mix(separate: dict[str, Any]) -> bool:

    stem_db = separate.get("stem_to_mix_db")
    if not isinstance(stem_db, (int, float)):
        return False
    return float(stem_db) > _STEM_IS_MIX_DB


def has_voice(separate: dict[str, Any], config: Any = None) -> bool:

    ratio = separate.get("vocal_ratio")
    stem_db = separate.get("stem_to_mix_db")
    if not isinstance(ratio, (int, float)) or not isinstance(stem_db, (int, float)):
        return False

    min_ratio = _VOICE_MIN_RATIO
    min_stem_db = _VOICE_MIN_STEM_DB
    if config is not None:
        min_ratio = float(config.get("voice_min_ratio", min_ratio))
        min_stem_db = float(config.get("voice_min_stem_db", min_stem_db))
    return float(ratio) >= min_ratio and float(stem_db) >= min_stem_db


has_voice_without_words = has_voice


class LyricsFusionStage(LocalStage):

    name = "lyrics"
    depends_on = ("asr.vocal", "asr.mix", "separate", "index")

    def iter_inputs(self) -> Iterable[dict[str, Any]]:
        primary = self.upstream("asr.vocal")
        secondary = self.upstream("asr.mix")
        separate = self.upstream("separate")
        index = self.upstream("index")
        for sample_id, record in sorted(primary.items()):
            yield {
                "sample_id": sample_id,
                "primary": record,
                "secondary": secondary.get(sample_id) or {},


                "separate": separate.get(sample_id) or {},


                "index": index.get(sample_id) or {},
            }

    def process(self, item: dict[str, Any]) -> dict[str, Any] | None:
        primary = item["primary"]
        secondary = item["secondary"]

        segments = list(primary.get("segments") or [])
        intervals = (item.get("separate") or {}).get("vocal_intervals") or []
        min_voiced = float(
            self.config.get(
                "undertranscribed_min_voiced_sec", _UNDERTRANSCRIBED_MIN_VOICED_SEC
            )
        )
        max_ratio = float(
            self.config.get("undertranscribed_max_ratio", _UNDERTRANSCRIBED_MAX_RATIO)
        )


        path_mode = str(self.config.get("window_path_select", "off"))
        pre_rows = window_densities(segments, intervals, min_voiced_sec=min_voiced)
        pre_hits, _ = undertranscribed_windows(
            pre_rows, max_ratio=max_ratio, min_voiced_sec=min_voiced
        )
        segments, path_stats = select_window_path(
            segments,
            list(secondary.get("segments") or []),
            intervals,
            mode=path_mode,
            flagged={(round(h["start"], 2), round(h["end"], 2)) for h in pre_hits},
            max_looping=float(
                self.config.get("window_path_max_looping", _WINDOW_PATH_MAX_LOOPING)
            ),
            min_gain_multiple=float(
                self.config.get(
                    "window_path_min_gain_multiple", _WINDOW_PATH_MIN_GAIN_MULTIPLE
                )
            ),
        )
        dropped: list[dict[str, Any]] = []
        min_char_rate = float(self.config.get("min_window_char_rate", 0.25))
        min_rate_sec = float(self.config.get("min_window_rate_sec", 2.0))
        if segments:
            segments, dropped = filter_degenerate_windows(
                segments,
                min_char_rate=min_char_rate,
                min_duration_sec=min_rate_sec,
            )


            text = " ".join(
                str(s.get("text") or "").strip()
                for s in segments
                if str(s.get("text") or "").strip()
            ).strip()
        else:

            text = (primary.get("text") or "").strip()
        other = (secondary.get("text") or "").strip()

        min_units = int(self.config.get("min_text_units", 3))

        units = len(tokenize(text))


        cross_text = text
        if path_stats["switched_windows"]:
            cross_text = _finalize_primary_text(
                primary,
                min_char_rate=min_char_rate,
                min_duration_sec=min_rate_sec,
            )
        pair_error = error_rate(cross_text, other) if (cross_text and other) else 1.0

        repeat = repeat_ngram_ratio(text)
        looping = looping_ratio(text)
        language = _resolve_language(text, primary, secondary)


        #


        separate = item.get("separate") or {}
        index_row = item.get("index") or {}
        wordless = bool(index_row.get("wordless"))

        voicing = str(index_row.get("wordless_voicing") or "")
        vocal_ratio = separate.get("vocal_ratio")
        other_units = len(tokenize(other))


        secondary_ran = bool(secondary)

        if units < min_units and (other_units < min_units or not secondary_ran):


            #


            #


            #


            #


            #


            if not has_voice(separate, self.config):
                tier, reason, status = "instrumental", "neither transcript contains lyrics", "no_vocals"
            elif wordless and voicing == VOICING_HUMAN:
                tier, reason, status = (
                    "vocal_no_lyrics",
                    "neither transcript contains lyrics; audio contains vocals and the "
                    "source declares non-lexical vocalization",
                    "vocals_no_lyrics",
                )
            elif wordless:
                tier, status = "instrumental", "no_human_voicing"
                reason = (
                    "Neither transcript contains lyrics. The audio resembles a human "
                    "voice, but the source identifies it as an instrument sample."
                    if voicing == VOICING_INSTRUMENT
                    else "Neither transcript contains lyrics. The audio resembles a human "
                    "voice, but the source does not identify the sound as singing."
                )
            else:
                tier, reason, status = (
                    "reject",
                    "neither transcript contains lyrics, but vocals are present and the words are unknown",
                    "vocals_untranscribed",
                )
        elif units < min_units:


            tier, reason, status = "reject", "mixture has lyrics but the separated track has no transcript", "primary_empty"
        elif not other and secondary_ran and _stem_is_the_mix(separate):


            #


            #            (38-vocalset 7 + 47-annotated-vocalset 6),


            #


            if wordless and voicing == VOICING_HUMAN:
                tier, reason, status = (
                    "vocal_no_lyrics",
                    "vocal and mixture energy match without accompaniment; the source "
                    "declares non-lexical vocalization",
                    "vocals_no_lyrics",
                )
            else:
                tier, reason, status = (
                    "reject",
                    "vocal and mixture energy match without accompaniment; only the "
                    "separated track has text, so lyrics remain uncertain",
                    "vocals_untranscribed",
                )
        elif not other and secondary_ran:


            tier, reason, status = "instrumental", "only the separated track has text; treated as a separation artifact", "stem_artifact"
        elif not other:


            tier, reason, status = "bronze", "second transcript is unavailable for cross-checking", "no_cross_check"
        else:


            #


            #


            tier, reason, status = "checked", "", "ok"


        #


        #

        confidence = max(0.0, 1.0 - pair_error) if (text and other) else 0.0


        #


        borderline = bool(text and other and confidence < _WORKING_POINTS["pretrain"])


        #


        #


        hallucinated = hallucinated_windows(segments, language, _reference_window_sec())
        if hallucinated:
            borderline = True


        #


        final_rows = window_densities(segments, intervals, min_voiced_sec=min_voiced)
        under_hits, window_median = undertranscribed_windows(
            final_rows, max_ratio=max_ratio, min_voiced_sec=min_voiced
        )


        dropped_under = [
            {
                "start": row["start"],
                "end": row["end"],
                "voiced_sec": row["voiced_sec"],
                "chars": row["chars"],
                "density": None if row["density"] is None else round(row["density"], 4),
                "ratio": 0.0,
                "dropped": True,
            }
            for row in window_densities(dropped, intervals, min_voiced_sec=min_voiced)
            if row["usable"]
        ]
        under_spans = sorted(
            [
                {
                    "start": round(h["start"], 3),
                    "end": round(h["end"], 3),
                    "voiced_sec": h["voiced_sec"],
                    "chars": h["chars"],
                    "density": None
                    if h["density"] is None
                    else round(h["density"], 4),
                    "ratio": h.get("ratio", 0.0),
                    "dropped": bool(h.get("dropped")),
                }
                for h in list(under_hits) + dropped_under
            ],
            key=lambda h: h["start"],
        )


        trusted = tier in {"checked", "bronze"}
        return {
            "sample_id": item["sample_id"],
            "text": text if trusted else "",
            "tier": tier,


            "lyrics_status": status,
            "reason": reason,
            "borderline": borderline,


            "lyrics_confidence": round(confidence, 4),
            "vocal_ratio": vocal_ratio,


            "stem_to_mix_db": separate.get("stem_to_mix_db"),


            "wordless_declared": wordless,


            "wordless_voicing": voicing,


            "untrusted_text": "" if trusted else text,
            "language": language,
            "pair_error": round(pair_error, 4),
            "repeat_ratio": round(repeat, 4),


            #


            #


            "looping_ratio": round(looping, 4),


            "hallucinated_windows": len(hallucinated),
            "hallucinated_chars": max((h["chars"] for h in hallucinated), default=0),


            # `pair_error=0.077` / `looping_ratio=0` / `hallucinated_windows=0`

            #


            "undertranscribed_windows": len(under_spans),
            "undertranscribed_voiced_sec": round(
                sum(s["voiced_sec"] for s in under_spans), 2
            ),
            "undertranscribed_spans": under_spans,


            "window_density_median": None
            if window_median is None
            else round(window_median, 4),


            "path_switched_windows": path_stats["switched_windows"],
            "path_switched_chars": path_stats["switched_chars"],
            "path_looping_blocked": path_stats["looping_blocked"],
            "path_thin_gain_blocked": path_stats["thin_gain_blocked"],
            "text_units": units,


            "segments": segments if trusted else [],
            "dropped_windows": len(dropped),
            "dropped_window_sec": round(
                sum(float(s["end"]) - float(s["start"]) for s in dropped), 2
            ),
            "duration_sec": primary.get("duration_sec", 0.0),
        }


#


#

#   pretrain      95%   0.723     92.8%          51.3%      [+0.010, +0.024]
#   melody_cot    70%   0.905     94.5%          70.5%      [+0.032, +0.084]
#   sft           50%   0.945     96.0%          75.8%      [+0.062, +0.133]
#


#


_WORKING_POINTS = {
    "pretrain": 0.723,
    "melody_cot": 0.905,
    "sft": 0.945,
}


def working_point(config: dict[str, Any], name: str) -> float:

    return float((config.get("working_points") or {}).get(name, _WORKING_POINTS[name]))


class AlignStage(WorkerStage):

    name = "align"
    depends_on = ("lyrics", "asr.vocal", "separate", "index")
    script = "w_align.py"


    #


    #


    #

    max_zero_duration_ratio = 0.40


    #


    min_units_for_zero_guard = 200
    min_songs_for_zero_guard = 10

    def _finish(self, report: Any) -> Any:
        units = zero = songs = 0


        out_of_range = 0
        out_of_range_songs = 0


        nnodes = int(os.environ.get("NNODES", "1"))
        records = (
            self.store.iter_records(ranks=self._node_ranks())
            if nnodes > 1
            else self.store.iter_records()
        )
        for record in records:
            current = record.get("units") or []
            if current:
                songs += 1
            violations = int(record.get("out_of_range_units") or 0)
            if violations:
                out_of_range += violations
                out_of_range_songs += 1
            for unit in current:
                units += 1
                if float(unit["end"]) - float(unit["start"]) <= 1e-6:
                    zero += 1
        if not units:
            return super()._finish(report)

        ratio = zero / units
        report.notes["zero_duration_ratio"] = round(ratio, 4)
        report.notes["zero_duration_units"] = zero
        report.notes["align_units"] = units


        report.notes["out_of_range_units"] = out_of_range
        report.notes["out_of_range_songs"] = out_of_range_songs
        report.notes["align_guard_scope"] = (
            f"node:{os.environ.get('NODE_RANK', '0')}" if nnodes > 1 else "global"
        )

        limit = float(
            self.config.get("max_zero_duration_ratio", self.max_zero_duration_ratio)
        )
        min_units = int(
            self.config.get("min_units_for_zero_guard", self.min_units_for_zero_guard)
        )
        min_songs = int(
            self.config.get("min_songs_for_zero_guard", self.min_songs_for_zero_guard)
        )
        if units < min_units or songs < min_songs:


            report.notes["zero_duration_guard"] = (
                f"insufficient samples ({units} units / {songs} songs; "
                f"requires at least {min_units} and {min_songs})"
            )
        elif ratio > limit:

            raise RuntimeError(
                f"Alignment zero-duration rate {ratio:.1%} ({zero}/{units} units across "
                f"{songs} songs) exceeds the {limit:.0%} limit. "
                "Check the ASR window-sec settings and alignment inputs."
            )
        return super()._finish(report)

    def select_inputs(self) -> list[dict[str, Any]]:
        separate = self.upstream("separate")
        vocal = self.upstream("asr.vocal")
        index = self.upstream("index")
        rows = self.upstream("lyrics")
        items: list[dict[str, Any]] = []


        accepted = set(self.config.get("accept_tiers") or ("checked",))


        source = str(self.config.get("audio_source", "stem"))
        for sample_id, record in sorted(rows.items()):
            if record.get("tier") not in accepted:
                continue
            text = (record.get("text") or "").strip()
            if not text:
                continue
            if source == "mix":
                stem = (index.get(sample_id) or {}).get("audio_path")
            else:
                stem = (separate.get(sample_id) or {}).get("vocal_16k_path")
            if not stem:
                continue
            items.append(
                {
                    "sample_id": sample_id,
                    "audio_path": stem,
                    "text": text,


                    #


                    "segments": record.get("segments")
                    or (vocal.get(sample_id) or {}).get("segments")
                    or [],
                    "language": record.get("language") or "en",
                    "duration_sec": record.get("duration_sec", 0.0),
                }
            )
        return items
