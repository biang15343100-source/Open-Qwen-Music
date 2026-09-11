
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator

UNKNOWN = "unknown"


class CodeBook:

    __slots__ = ("name", "_names", "_codes")

    def __init__(self, name: str, names: Iterable[str]) -> None:
        ordered = list(names)
        if ordered[0] != UNKNOWN:
            ordered = [UNKNOWN, *ordered]
        if len(set(ordered)) != len(ordered):
            dupes = sorted({x for x in ordered if ordered.count(x) > 1})
            raise ValueError(f"code table {name} Duplicate exists: {dupes}")
        self.name = name
        self._names: tuple[str, ...] = tuple(ordered)
        self._codes: dict[str, int] = {v: i for i, v in enumerate(ordered)}

    def code(self, value: str | None, *, strict: bool = False) -> int:
        if value is None:
            return 0
        got = self._codes.get(value)
        if got is None:
            if strict:
                raise KeyError(f"code table {self.name} No value registered {value!r};optional: {self._names}")
            return 0
        return got

    def name_of(self, code: int) -> str:
        if not 0 <= code < len(self._names):
            raise IndexError(f"code table {self.name} out-of-bounds code value {code}")
        return self._names[code]

    def has(self, value: str) -> bool:
        return value in self._codes

    def fingerprint(self) -> str:
        payload = "\x00".join(self._names).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def as_dict(self) -> dict[int, str]:
        return dict(enumerate(self._names))

    def __len__(self) -> int:
        return len(self._names)

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)


STORAGE_CLASS = CodeBook(
    "storage_class",
    ["loose", "zip", "tar", "targz", "parquet", "sqlite", "external"],
)


CODEC = CodeBook(
    "codec",
    [
        "pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_u8", "pcm_f32le", "pcm_f64le",
        "mp3", "aac", "vorbis", "opus", "flac", "alac", "wmav2", "amr_nb", "gsm",
    ],
)


CONTENT_TYPE = CodeBook(
    "content_type",
    [
        "music_mix",
        "music_stem",
        "vocal_solo",
        "instrument_solo",
        "note",
        "speech",
        "sound_event",
        "noise",
    ],
)

GRANULARITY = CodeBook("granularity", ["whole_song", "clip", "phrase", "note"])

DOMAIN = CodeBook("domain", ["music", "singing", "speech", "audio_event"])

STEM_ROLE = CodeBook(
    "stem_role",
    ["mixture", "vocals", "drums", "bass", "guitar", "piano", "strings",
     "wind", "synth", "percussion", "other", "accompaniment"],
)

SYNTHETIC_MODEL = CodeBook(
    "synthetic_model",
    ["suno", "udio", "riffusion", "acestep", "mureka", "musicgen", "heartmula", "other_ai"],
)


LYRICS_FORMAT = CodeBook(
    "lyrics_format",
    ["none", "plain", "lrc", "structured", "phoneme", "word_timed"],
)

ALIGN_LEVEL = CodeBook("lyrics_align_level", ["none", "section", "line", "word", "phoneme"])

TEXT_SOURCE = CodeBook(
    "text_source",
    ["official", "human_annotation", "asr", "generation_prompt", "llm", "template", "filename"],
)


LANGUAGE = CodeBook(
    "language",
    [
        "zh", "en", "zh_en", "ja", "ko", "es", "fr", "de", "it", "pt", "ru", "other",
        "instrumental",
        "language_neutral",
    ],
)

CONFIDENCE = CodeBook("confidence", ["low", "medium", "high"])

LANGUAGE_EVIDENCE = CodeBook(
    "language_evidence",
    [
        "record_field",
        "dataset_scope",
        "path_pattern",
        "lyrics_script",
        "transcript_script",
        "instrumental_rule",
        "no_evidence",
    ],
)

LICENSE_FAMILY = CodeBook(
    "license_family",
    ["cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa", "cc-by-nc-nd",
     "research_only", "proprietary", "public_domain"],
)


STATUS = CodeBook(
    "status",
    [
        "accepted",
        "rejected",
        "alias",
        "no_local_audio",
        "pending_probe",
    ],
)

DUP_STATUS = CodeBook("dup_status", ["canonical", "alias"])


DUP_METHOD = CodeBook("dup_method",
                      ["uri", "byte_hash", "external_id", "fingerprint", "transitive"])

SPLIT = CodeBook("split", ["train", "valid", "test"])

SPLIT_SOURCE = CodeBook(
    "split_source",
    ["official_field", "official_path", "stable_group_hash", "group_promoted"],
)

DURATION_SOURCE = CodeBook("duration_source", ["decoded", "declared", "header"])


ALL_CODEBOOKS: dict[str, CodeBook] = {
    book.name: book
    for book in (
        STORAGE_CLASS, CODEC, CONTENT_TYPE, GRANULARITY, DOMAIN, STEM_ROLE,
        SYNTHETIC_MODEL, LYRICS_FORMAT, ALIGN_LEVEL, TEXT_SOURCE, LANGUAGE,
        CONFIDENCE, LANGUAGE_EVIDENCE, LICENSE_FAMILY, STATUS, DUP_STATUS,
        DUP_METHOD, SPLIT, SPLIT_SOURCE, DURATION_SOURCE,
    )
}


def codebooks_fingerprint() -> str:
    joined = "|".join(f"{name}:{book.fingerprint()}" for name, book in sorted(ALL_CODEBOOKS.items()))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def export_codebooks() -> dict[str, object]:
    return {
        "fingerprint": codebooks_fingerprint(),
        "codebooks": {name: book.as_dict() for name, book in sorted(ALL_CODEBOOKS.items())},
    }
