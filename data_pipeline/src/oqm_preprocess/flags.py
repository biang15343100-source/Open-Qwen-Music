
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    REJECT = "reject"
    WARN = "warn"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class FlagSpec:
    name: str
    bit: int
    severity: Severity
    desc: str


def _spec(name: str, bit: int, severity: Severity, desc: str) -> FlagSpec:
    return FlagSpec(name, bit, severity, desc)


_SPECS: tuple[FlagSpec, ...] = (
    _spec("container_missing", 0, Severity.REJECT, "Archive or directory is missing"),
    _spec("member_missing", 1, Severity.REJECT, "Archive member is missing or its index is stale"),
    _spec("probe_failed", 2, Severity.REJECT, "Container or codec probe failed"),
    _spec("decode_failed", 3, Severity.REJECT, "Audio decoding failed"),
    _spec("empty_audio", 4, Severity.REJECT, "Decoded audio has no frames"),
    _spec("nonfinite_audio", 5, Severity.REJECT, "Audio contains NaN or Inf samples"),
    _spec("truncated", 6, Severity.REJECT, "Audio is shorter than declared and appears truncated"),
    _spec("not_audio", 7, Severity.REJECT, "Enumerated member is not audio"),
    _spec("mac_junk", 8, Severity.REJECT, "macOS metadata file"),
    _spec("zero_bytes", 9, Severity.REJECT, "Member has zero bytes"),
    _spec("unsupported_codec", 10, Severity.REJECT, "Codec is not supported"),
    _spec("too_short", 16, Severity.REJECT, "Duration is below the configured minimum"),
    _spec("too_long", 17, Severity.WARN, "Duration exceeds the configured maximum"),
    _spec("duration_mismatch", 18, Severity.WARN, "Declared and decoded durations differ"),
    _spec("duration_unknown", 19, Severity.WARN, "Duration is unknown"),
    _spec("near_silent", 24, Severity.REJECT, "Audio is nearly silent"),
    _spec("severe_clipping", 25, Severity.REJECT, "Clipping exceeds the configured limit"),
    _spec("fake_stereo", 26, Severity.WARN, "Stereo channels are effectively identical"),
    _spec("upsampled_lowband", 27, Severity.WARN, "Bandwidth suggests low-rate transcoding"),
    _spec("low_bitrate", 28, Severity.WARN, "Bitrate is below the configured minimum"),
    _spec("sample_rate_below_target", 29, Severity.WARN, "Sample rate is below 24 kHz"),
    _spec("heavy_leading_silence", 30, Severity.WARN, "Leading or trailing silence is excessive"),
    _spec("dc_offset_high", 31, Severity.WARN, "DC offset exceeds the configured limit"),
    _spec("lyrics_audio_mismatch", 36, Severity.WARN, "Lyrics length does not match audio duration"),
    _spec("lyrics_is_prompt", 37, Severity.INFO, "Lyrics field was moved to prompt_text"),
    _spec("lyrics_lang_not_zh_en", 38, Severity.WARN, "Lyrics language is outside the supported set"),
    _spec("lyrics_empty_claimed", 39, Severity.WARN, "Lyrics are declared but empty"),
    _spec("transcript_needs_normalization", 40, Severity.WARN, "Transcript contains placeholders"),
    _spec("text_from_filename", 41, Severity.INFO, "Text was inferred from the filename"),
    _spec("eval_holdout", 48, Severity.INFO, "Evaluation holdout; exclude from training"),
    _spec("license_restricted", 49, Severity.INFO, "License restricts use"),
    _spec("license_unknown", 50, Severity.INFO, "License is unknown"),
    _spec("has_lyrics_timeline", 56, Severity.INFO, "Lyrics timeline is available"),
    _spec("has_stems", 57, Severity.INFO, "Separated stems are available"),
    _spec("has_raw_meta", 58, Severity.INFO, "Raw metadata is available"),
    _spec("expected_count_mismatch", 59, Severity.INFO, "Enumerated count differs from the expected count"),
)

BY_NAME: dict[str, FlagSpec] = {s.name: s for s in _SPECS}
BY_BIT: dict[int, FlagSpec] = {s.bit: s for s in _SPECS}

if len(BY_NAME) != len(_SPECS) or len(BY_BIT) != len(_SPECS):
    raise RuntimeError("flag Duplicate code point or name")

REJECT_MASK: int = 0
WARN_MASK: int = 0
INFO_MASK: int = 0
for _s in _SPECS:
    if _s.severity is Severity.REJECT:
        REJECT_MASK |= 1 << _s.bit
    elif _s.severity is Severity.WARN:
        WARN_MASK |= 1 << _s.bit
    else:
        INFO_MASK |= 1 << _s.bit


def bit(name: str) -> int:
    try:
        return 1 << BY_NAME[name].bit
    except KeyError:
        raise KeyError(f"Unregistered flag {name!r};optional: {sorted(BY_NAME)}") from None


def mask_of(*names: str) -> int:
    out = 0
    for n in names:
        out |= bit(n)
    return out


def decode(mask: int) -> list[str]:
    return [BY_BIT[b].name for b in sorted(BY_BIT) if mask & (1 << b)]


def is_rejected(mask: int) -> bool:
    return bool(mask & REJECT_MASK)


def reject_reasons(mask: int) -> list[str]:
    return [BY_BIT[b].name for b in sorted(BY_BIT) if mask & (1 << b) and (REJECT_MASK & (1 << b))]


def fingerprint() -> str:
    payload = "|".join(f"{s.bit}:{s.name}:{s.severity.value}" for s in _SPECS)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def export_codebook() -> dict[str, object]:
    return {
        "fingerprint": fingerprint(),
        "reject_mask": REJECT_MASK,
        "warn_mask": WARN_MASK,
        "info_mask": INFO_MASK,
        "flags": [
            {"bit": s.bit, "name": s.name, "severity": s.severity.value, "desc": s.desc}
            for s in _SPECS
        ],
    }
