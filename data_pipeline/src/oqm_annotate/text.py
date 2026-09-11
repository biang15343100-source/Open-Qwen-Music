
from __future__ import annotations

import re
import unicodedata


def _range(start: int, end: int) -> str:
    return f"{chr(start)}-{chr(end)}"


_CJK = re.compile(
    f"[{_range(0x3040, 0x30FF)}{_range(0x3400, 0x9FFF)}"
    f"{_range(0xF900, 0xFAFF)}{_range(0xAC00, 0xD7AF)}]"
)
_KANA = re.compile(f"[{_range(0x3040, 0x30FF)}]")
_HANGUL = re.compile(f"[{_range(0xAC00, 0xD7AF)}]")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def is_cjk_text(text: str, *, threshold: float = 0.2) -> bool:

    stripped = _SPACES.sub("", text)
    if not stripped:
        return False
    cjk_count = len(_CJK.findall(stripped))
    return cjk_count / len(stripped) >= threshold


def normalize_for_compare(text: str) -> str:

    normalized = unicodedata.normalize("NFKC", text or "").lower()
    normalized = _PUNCT.sub(" ", normalized)
    return _SPACES.sub(" ", normalized).strip()


def tokenize(text: str) -> list[str]:

    normalized = normalize_for_compare(text)
    if not normalized:
        return []
    if not is_cjk_text(normalized):
        return normalized.split()

    tokens: list[str] = []
    buffer: list[str] = []
    for char in normalized:
        if _CJK.match(char):
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            tokens.append(char)
        elif char.isspace():
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
        else:
            buffer.append(char)
    if buffer:
        tokens.append("".join(buffer))
    return tokens


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:

    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str) -> float:

    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    denominator = max(len(ref_tokens), len(hyp_tokens))
    if denominator == 0:
        return 0.0
    return edit_distance(ref_tokens, hyp_tokens) / denominator


def repeat_ngram_ratio(text: str, *, n: int = 4) -> float:

    tokens = tokenize(text)
    if len(tokens) < n * 2:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    unique = len(set(grams))
    return 1.0 - unique / len(grams)


def looping_ratio(text: str, *, max_period: int = 20, min_cycles: int = 4) -> float:

    tokens = tokenize(text)
    length = len(tokens)
    if length < 8:
        return 0.0

    covered = 0
    for period in range(1, min(max_period, length // 2) + 1):
        run = 0
        threshold = period * (min_cycles - 1)
        for index in range(length - period):
            if tokens[index] == tokens[index + period]:
                run += 1
                if run >= threshold:
                    covered = max(covered, run + period)
            else:
                run = 0
    return min(1.0, covered / length)


_HAN = re.compile(f"[{_range(0x4E00, 0x9FFF)}]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
STRONG_HAN_SHARE = 0.5


#


#


#


#


_SYLLABARY_MIN_SHARE = 0.25


def script_evidence(text: str) -> str | None:

    kana = len(_KANA.findall(text))
    hangul = len(_HANGUL.findall(text))
    han = len(_HAN.findall(text))
    cjk = kana + hangul + han
    if cjk and (kana + hangul) / cjk >= _SYLLABARY_MIN_SHARE:

        return "ja" if kana >= hangul else "ko"
    total = han + len(_LATIN_LETTER.findall(text))
    if total and han / total >= STRONG_HAN_SHARE:
        return "zh"
    return None


def detect_language(text: str) -> str:

    if not text.strip():
        return "unknown"
    if _HANGUL.search(text):
        return "ko"
    if _KANA.search(text):
        return "ja"
    if is_cjk_text(text):
        return "zh"
    latin = sum(1 for char in text if "a" <= char.lower() <= "z")
    return "en" if latin >= len(text.strip()) * 0.3 else "other"
