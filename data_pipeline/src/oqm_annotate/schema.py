
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "oqm.annotation.v1"


SECTION_LABELS: tuple[str, ...] = (
    "intro",
    "verse",
    "chorus",
    "bridge",
    "inst",
    "outro",
    "silence",
)


VOCAL_SECTION_LABELS: frozenset[str] = frozenset({"verse", "chorus", "bridge"})

VOCAL_GENDERS: tuple[str, ...] = ("female", "male", "mixed", "instrumental")


@dataclass(slots=True)
class Section:

    section_id: str
    label: str
    start_sec: float
    end_sec: float
    has_lyrics: bool
    is_vocal: bool
    lyrics: str = ""

    unit_count: int = 0


    lyric_coverage: float = 0.0

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "label": self.label,
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "has_lyrics": self.has_lyrics,
            "is_vocal": self.is_vocal,
            "lyrics": self.lyrics,
            "unit_count": self.unit_count,
            "lyric_coverage": round(self.lyric_coverage, 4),
        }


@dataclass(slots=True)
class MusicalTags:

    genre: list[str] = field(default_factory=list)
    mood: list[str] = field(default_factory=list)
    instrument: list[str] = field(default_factory=list)
    vocal_gender: str | None = None
    vocal_timbre: list[str] = field(default_factory=list)
    confidence: dict[str, float] = field(default_factory=dict)
    sources: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "genre": self.genre,
            "mood": self.mood,
            "instrument": self.instrument,
            "vocal_gender": self.vocal_gender,
            "vocal_timbre": self.vocal_timbre,
            "confidence": {k: round(v, 4) for k, v in self.confidence.items()},
            "sources": self.sources,
        }


def serialize_structured_lyrics(sections: list[Section]) -> str:

    blocks: list[str] = []
    for section in sections:
        if not section.has_lyrics:
            continue
        text = section.lyrics.strip()
        if not text:
            continue
        blocks.append(f"[{section.label}]\n{text}")
    return "\n\n".join(blocks)


def validate_annotation(record: dict[str, Any]) -> list[str]:

    problems: list[str] = []

    if record.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version must be {SCHEMA_VERSION}; got {record.get('schema_version')!r}"
        )
    if not record.get("sample_id"):
        problems.append("sample_id is missing")

    duration = float(record.get("audio", {}).get("duration_sec") or 0.0)
    if duration <= 0:
        problems.append("audio.duration_sec must be positive")

    tags = record.get("tags") or {}
    gender = tags.get("vocal_gender")
    if gender is not None and gender not in VOCAL_GENDERS:
        problems.append(f"vocal_gender {gender!r} is not in {VOCAL_GENDERS}")
    for key in ("genre", "mood", "instrument", "vocal_timbre"):
        value = tags.get(key)
        if value is not None and not isinstance(value, list):
            problems.append(f"tags.{key} must be a list; got {type(value).__name__}")

    sections = record.get("sections") or []
    previous_end = -1e-6
    for index, section in enumerate(sections):
        label = section.get("label")
        if label not in SECTION_LABELS:
            problems.append(f"sections[{index}].label {label!r} is not in {SECTION_LABELS}")
        start = float(section.get("start_sec", -1))
        end = float(section.get("end_sec", -1))
        if start < 0 or end <= start:
            problems.append(f"sections[{index}] has invalid bounds [{start}, {end}]")

        if start + 1e-3 < previous_end:
            problems.append(f"sections[{index}] overlaps the previous section")
        if end > duration + 0.5:
            problems.append(f"sections[{index}].end_sec {end} exceeds audio duration {duration}")
        previous_end = end
        if section.get("has_lyrics") and not (section.get("lyrics") or "").strip():
            problems.append(f"sections[{index}] has_lyrics=True but lyrics is empty")

    structured = record.get("structured_lyrics")
    if structured is None:
        problems.append("structured_lyrics is missing; use an empty string when no lyrics are present")

    return problems
