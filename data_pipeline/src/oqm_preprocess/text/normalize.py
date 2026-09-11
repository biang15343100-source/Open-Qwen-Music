
from __future__ import annotations

import re
import unicodedata


_LRC_TIME = re.compile(r"\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]")
_LRC_META = re.compile(r"^\[(ar|ti|al|by|offset|re|ve|length)\s*:[^\]]*\]\s*$", re.IGNORECASE)

_CJK_SECTION_LABELS = "|".join(
    "".join(chr(codepoint) for codepoint in points)
    for points in (
        (0x4E3B, 0x6B4C),
        (0x526F, 0x6B4C),
        (0x524D, 0x594F),
        (0x95F4, 0x594F),
        (0x5C3E, 0x594F),
        (0x6865, 0x6BB5),
        (0x8FC7, 0x95E8),
    )
)
_SECTION = re.compile(
    r"^\s*[\[\(【]\s*(verse|chorus|bridge|intro|outro|pre-?chorus|hook|refrain|interlude|"
    + _CJK_SECTION_LABELS
    + r")[^\]\)】]*[\]\)】]\s*$",
    re.IGNORECASE,
)
_CJK = re.compile(
    f"[{chr(0x4E00)}-{chr(0x9FFF)}{chr(0x3400)}-{chr(0x4DBF)}"
    f"{chr(0xF900)}-{chr(0xFAFF)}]"
)
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
_KANA = re.compile(f"[{chr(0x3040)}-{chr(0x30FF)}]")
_HANGUL = re.compile(f"[{chr(0xAC00)}-{chr(0xD7AF)}{chr(0x1100)}-{chr(0x11FF)}]")
_CYRILLIC = re.compile(f"[{chr(0x0400)}-{chr(0x04FF)}]")
_CYRILLIC_WORD = re.compile(f"[{chr(0x0400)}-{chr(0x04FF)}]+")


_PROMPT_MARKERS = re.compile(
    r"\b(bpm|tempo|genre|instrumental|male vocal|female vocal|lo-?fi|edm|synthwave|"
    r"prompt|style of|in the style|acoustic guitar|808|reverb)\b",
    re.IGNORECASE,
)
_PLACEHOLDER = re.compile(r"[<\[\{](unk|noise|laughter|music|sil|silence|nsn|spn)[>\]\}]", re.IGNORECASE)


def clean(text: str | None) -> str:
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = out.replace("​", "").replace("﻿", "").replace("　", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in out.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def is_lrc(text: str) -> bool:
    return bool(_LRC_TIME.search(text))


def parse_lrc(text: str) -> tuple[str, list[tuple[float, str]]]:
    timeline: list[tuple[float, str]] = []
    plain: list[str] = []
    for line in text.split("\n"):
        if _LRC_META.match(line):
            continue
        stamps = _LRC_TIME.findall(line)
        body = _LRC_TIME.sub("", line).strip()
        if not stamps:
            if body:
                plain.append(body)
            continue
        if not body:
            continue
        plain.append(body)
        for minute, second, frac in stamps:
            fraction = float(f"0.{frac}") if frac else 0.0
            timeline.append((int(minute) * 60 + int(second) + fraction, body))
    timeline.sort(key=lambda x: x[0])
    return "\n".join(plain), timeline


def strip_sections(text: str) -> tuple[str, list[str]]:
    body: list[str] = []
    labels: list[str] = []
    for line in text.split("\n"):
        if _SECTION.match(line):
            labels.append(line.strip(" []()【】"))
            continue
        body.append(line)
    return "\n".join(body).strip(), labels


def singable_units(text: str) -> int:
    return (len(_CJK.findall(text)) + len(_LATIN_WORD.findall(text))
            + len(_HANGUL.findall(text)) + len(_KANA.findall(text))
            + len(_CYRILLIC_WORD.findall(text)))


def classify_script(text: str) -> str:
    if not text:
        return "unknown"
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN_WORD.findall(text))
    kana = len(_KANA.findall(text))
    hangul = len(_HANGUL.findall(text))
    cyrillic = len(_CYRILLIC.findall(text))
    total = cjk + latin + kana + hangul + cyrillic
    if total == 0:
        return "unknown"

    if kana >= max(3, total * 0.05):
        return "ja"
    if hangul >= total * 0.3:
        return "ko"
    if cyrillic >= total * 0.3:
        return "ru"
    zh_ratio = cjk / total
    en_ratio = latin / total
    if zh_ratio >= 0.7:
        return "zh"
    if en_ratio >= 0.85:
        return "en"
    if zh_ratio >= 0.1 and en_ratio >= 0.1:
        return "zh_en"
    return "other"


def looks_like_prompt(text: str) -> bool:
    if not text:
        return False
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) > 3:
        return False
    commas = text.count(",") + text.count("，")
    if commas < 2:
        return False
    return bool(_PROMPT_MARKERS.search(text))


def has_placeholders(text: str) -> bool:
    return bool(_PLACEHOLDER.search(text))


def normalize_lyrics(raw: str | None) -> dict[str, object]:
    text = clean(raw)
    if not text:
        return {"text": "", "format": "none", "timeline": [], "labels": [],
                "units": 0, "script": "unknown", "is_prompt": False}

    timeline: list[tuple[float, str]] = []
    fmt = "plain"
    if is_lrc(text):
        text, timeline = parse_lrc(text)
        fmt = "lrc"
    body, labels = strip_sections(text)
    if labels:
        fmt = "structured" if fmt == "plain" else fmt
    text = body or text

    return {
        "text": text,
        "format": fmt,
        "timeline": timeline,
        "labels": labels,
        "units": singable_units(text),
        "script": classify_script(text),
        "is_prompt": looks_like_prompt(text),
    }
