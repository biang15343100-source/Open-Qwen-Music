
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from .contracts import SECTION_LABELS


CONDITION_TEMPLATE_VERSION = "oqm.llm.condition.v2"


TAG_FIELD_ORDER: tuple[str, ...] = (
    "genre",
    "mood",
    "instruments",
    "vocal_gender",
    "vocal_timbre",
    "language",
    "bpm",
    "key",
)


#: "rates each generated song on a 1-10 scale across genre, mood, instrument,


EVAL_TAG_DIMENSIONS: tuple[str, ...] = (
    "genre",
    "mood",
    "instruments",
    "vocal_gender",
    "vocal_timbre",
)


LIST_TAG_FIELDS: frozenset[str] = frozenset({"genre", "mood", "instruments"})


TAG_FIELD_ALIASES: dict[str, str] = {
    "genres": "genre",
    "style": "genre",
    "styles": "genre",
    "moods": "mood",
    "emotion": "mood",
    "emotions": "mood",
    "instrument": "instruments",
    "instrumentation": "instruments",
    "arrangement": "instruments",
    "gender": "vocal_gender",
    "singer_gender": "vocal_gender",
    "voice_gender": "vocal_gender",
    "timbre": "vocal_timbre",
    "voice": "vocal_timbre",
    "vocal": "vocal_timbre",
    "singer": "vocal_timbre",
    "singer_timbre": "vocal_timbre",
    "vocal_characteristics": "vocal_timbre",
    "lang": "language",
    "tempo": "bpm",
    "tonality": "key",
}


@dataclass(frozen=True)
class ConditionRenderConfig:

    tag_fields: tuple[str, ...] = TAG_FIELD_ORDER
    include_instrumental_language: bool = True
    include_instrumental_marker: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ConditionRenderConfig:
        section = dict(config.get("condition", {}) or {})
        known = {
            "tag_fields",
            "include_instrumental_language",
            "include_instrumental_marker",
        }
        unknown = set(section) - known
        if unknown:
            raise KeyError(f"condition is configured with unknown fields:{sorted(unknown)}")
        raw_fields = section.get("tag_fields", TAG_FIELD_ORDER)
        fields = tuple(str(field) for field in raw_fields)
        if len(set(fields)) != len(fields):
            raise ValueError("condition.tag_fields cannot be repeated")
        invalid = sorted(set(fields) - set(TAG_FIELD_ORDER))
        if invalid:
            raise ValueError(
                f"condition.tag_fields contains non-whitelist fields:{invalid};"
                f"is {list(TAG_FIELD_ORDER)}"
            )
        return cls(
            tag_fields=fields,
            include_instrumental_language=bool(
                section.get("include_instrumental_language", True)
            ),
            include_instrumental_marker=bool(
                section.get("include_instrumental_marker", True)
            ),
        )

    @property
    def revision(self) -> str:
        if self == ConditionRenderConfig():
            return CONDITION_TEMPLATE_VERSION
        payload = json.dumps(
            {
                "tag_fields": self.tag_fields,
                "include_instrumental_language": self.include_instrumental_language,
                "include_instrumental_marker": self.include_instrumental_marker,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
        return f"{CONDITION_TEMPLATE_VERSION}+policy-{digest}"


_DURATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{1,3}:\d{2}(?::\d{2})?\b"),
    re.compile(
        r"\b\d+(?:\.\d+)?\s*"
        r"(?:ms|msec|msecs|sec|secs|second|seconds|min|mins|minute|minutes|hr|hrs|hour|hours)"
        r"\b",
        re.IGNORECASE,
    ),

    re.compile(r"\b\d+(?:\.\d+)?\s*(?:bars?|beats?|measures?|frames?)\b", re.IGNORECASE),
    re.compile(r"\d+(?:\.\d+)?\s*(?:\u6beb\u79d2|\u79d2\u949f|\u79d2|\u5206\u949f|\u5206|\u5c0f\u65f6|\u5c0f\u8282|\u62cd|\u5e27)"),
)


def find_duration_mentions(text: str) -> list[str]:
    found: list[str] = []
    for pattern in _DURATION_PATTERNS:
        found.extend(match.group(0) for match in pattern.finditer(text))
    return found


def strip_duration_mentions(text: str) -> str:
    for pattern in _DURATION_PATTERNS:
        text = pattern.sub(" ", text)
    return " ".join(text.split())


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.split())


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if float(value).is_integer() else f"{value:g}"
    return str(value)


def normalize_tag_value(value: Any) -> str:
    return strip_duration_mentions(normalize_text(_format_scalar(value)).lower())


def canonical_tag_key(key: Any) -> str:
    normalized = normalize_text(key).lower().replace(" ", "_").replace("-", "_")
    return TAG_FIELD_ALIASES.get(normalized, normalized)


def canonicalize_tags(tags: dict[str, Any]) -> dict[str, Any]:
    candidates: dict[str, list[tuple[str, Any]]] = {}
    for raw_key, value in tags.items():
        key = canonical_tag_key(raw_key)
        if key not in TAG_FIELD_ORDER:
            continue
        candidates.setdefault(key, []).append((str(raw_key), value))
    resolved: dict[str, Any] = {}
    for key, entries in candidates.items():
        exact = [value for raw_key, value in entries if raw_key == key]
        resolved[key] = exact[0] if exact else min(entries, key=lambda item: item[0])[1]
    return resolved


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


@dataclass(frozen=True)
class SectionTiming:

    start_sec: float | None = None
    end_sec: float | None = None
    melody_start_frame: int | None = None
    melody_end_frame: int | None = None

    @property
    def is_empty(self) -> bool:
        return (
            self.start_sec is None
            and self.end_sec is None
            and self.melody_start_frame is None
            and self.melody_end_frame is None
        )


class Section:


    CONDITION_VISIBLE_FIELDS: tuple[str, ...] = ("label", "lyrics")

    __slots__ = ("label", "lyrics", "timing")

    def __init__(
        self,
        label: str,
        lyrics: str = "",
        *,
        start_sec: float | None = None,
        end_sec: float | None = None,
        melody_start_frame: int | None = None,
        melody_end_frame: int | None = None,
        timing: SectionTiming | None = None,
    ) -> None:
        canonical = normalize_tag_value(label).replace(" ", "_").replace("-", "_")
        if canonical not in SECTION_LABELS:
            raise ValueError(
                f"Unknown section label: {label!r} -> {canonical!r}; "
                f"expected {SECTION_LABELS}. Declare taxonomy aliases explicitly in "
                "the annotation adapter."
            )
        self.label = canonical
        self.lyrics = normalize_text(lyrics) if lyrics else ""
        self.timing = timing or SectionTiming(
            start_sec=start_sec,
            end_sec=end_sec,
            melody_start_frame=melody_start_frame,
            melody_end_frame=melody_end_frame,
        )

    @property
    def start_sec(self) -> float | None:
        return self.timing.start_sec

    @property
    def end_sec(self) -> float | None:
        return self.timing.end_sec

    @property
    def melody_start_frame(self) -> int | None:
        return self.timing.melody_start_frame

    @property
    def melody_end_frame(self) -> int | None:
        return self.timing.melody_end_frame

    @property
    def is_vocal(self) -> bool:
        return bool(self.lyrics)

    def render(self) -> str:
        return f"[{self.label}] {self.lyrics}" if self.lyrics else f"[{self.label}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Section):
            return NotImplemented
        return (
            self.label == other.label
            and self.lyrics == other.lyrics
            and self.timing == other.timing
        )

    def __repr__(self) -> str:
        return f"Section(label={self.label!r}, lyrics={self.lyrics!r}, timing={self.timing!r})"


@dataclass
class MusicCondition:

    tags: dict[str, Any] = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)
    language: str | None = None
    is_instrumental: bool = False

    #: "The text portion, consisting of tags and lyrics, is used as conditioning context."


    description: str | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> MusicCondition:
        sections = [
            Section(**section) if isinstance(section, dict) else section
            for section in (record.get("sections") or [])
        ]
        return cls(
            tags=dict(record.get("tags") or {}),
            sections=sections,
            language=record.get("language"),
            is_instrumental=bool(record.get("is_instrumental", False)),
            description=record.get("description"),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "tags": self.tags,
            "sections": [
                {
                    "label": s.label,
                    "lyrics": s.lyrics,
                    "start_sec": s.start_sec,
                    "end_sec": s.end_sec,
                    "melody_start_frame": s.melody_start_frame,
                    "melody_end_frame": s.melody_end_frame,
                }
                for s in self.sections
            ],
            "language": self.language,
            "is_instrumental": self.is_instrumental,
            "description": self.description,
        }

    @property
    def vocal_sections(self) -> list[Section]:
        return [s for s in self.sections if s.is_vocal]

    @property
    def has_lyric_annotation(self) -> bool:
        return any(s.is_vocal for s in self.sections)


    def tag_fields(
        self, render_config: ConditionRenderConfig | None = None
    ) -> list[str]:
        render_config = render_config or ConditionRenderConfig()
        parts: list[str] = []
        tags = canonicalize_tags(self.tags)
        if self.is_instrumental and not render_config.include_instrumental_language:
            tags.pop("language", None)
        elif self.language and "language" not in tags:
            tags["language"] = self.language
        for name in render_config.tag_fields:
            if name not in tags:
                continue
            raw = tags[name]
            if raw is None:
                continue
            if name in LIST_TAG_FIELDS:
                if isinstance(raw, (set, frozenset)):

                    values: list[Any] = sorted(raw, key=str)
                elif isinstance(raw, (list, tuple)):
                    values = list(raw)
                else:
                    values = [raw]
                cleaned_values = _dedupe_preserving_order(
                    [normalize_tag_value(v) for v in values]
                )
                if not cleaned_values:
                    continue
                parts.append(f"{name}: {', '.join(cleaned_values)}")
            else:
                cleaned = normalize_tag_value(raw)
                if not cleaned:
                    continue
                parts.append(f"{name}: {cleaned}")
        if self.is_instrumental and render_config.include_instrumental_marker:


            parts.append("instrumental: true")
        return parts

    def render_tags(
        self, render_config: ConditionRenderConfig | None = None
    ) -> str:
        return " | ".join(self.tag_fields(render_config))

    def render_lyric_lines(self) -> list[str]:
        return [section.render() for section in self.sections]

    def render_lyrics(self) -> str:
        return "\n".join(self.render_lyric_lines())

    def render(
        self, render_config: ConditionRenderConfig | None = None
    ) -> str:
        blocks: list[str] = []
        tags = self.render_tags(render_config)
        if tags:
            blocks.append(f"[tags] {tags}")
        lyrics = self.render_lyrics()
        if lyrics:
            blocks.append(f"[lyrics]\n{lyrics}")
        return "\n".join(blocks)


def render_condition(condition: MusicCondition | dict[str, Any]) -> str:
    if isinstance(condition, dict):
        condition = MusicCondition.from_record(condition)
    return condition.render()
