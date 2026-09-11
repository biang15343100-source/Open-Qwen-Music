#!/usr/bin/env python3

from __future__ import annotations

from typing import Any


CODE_TO_NAME: dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}

NAME_TO_CODE: dict[str, str] = {
    name.lower(): code for code, name in CODE_TO_NAME.items()
}


_ALIASES: dict[str, str] = {
    "cmn": "zh", "zho": "zh", "chi": "zh", "mandarin": "zh",
    "eng": "en",
    "can": "yue", "zh-yue": "yue",
    "ara": "ar", "deu": "de", "ger": "de", "fra": "fr", "fre": "fr",
    "spa": "es", "por": "pt", "ind": "id", "ita": "it", "kor": "ko",
    "rus": "ru", "tha": "th", "vie": "vi", "jpn": "ja", "jap": "ja",
    "tur": "tr", "hin": "hi", "msa": "ms", "may": "ms",
    "nld": "nl", "dut": "nl", "swe": "sv", "dan": "da", "fin": "fi",
    "pol": "pl", "ces": "cs", "cze": "cs", "tl": "fil", "tgl": "fil",
    "fas": "fa", "per": "fa", "ell": "el", "gre": "el",
    "ron": "ro", "rum": "ro", "hun": "hu", "mkd": "mk", "mac": "mk",
}


def to_code(value: Any) -> str | None:

    if not value:
        return None
    text = str(value).strip().lower().replace("_", "-")
    if not text:
        return None
    if text in CODE_TO_NAME:
        return text
    if text in NAME_TO_CODE:
        return NAME_TO_CODE[text]
    if text in _ALIASES:
        return _ALIASES[text]

    head = text.split("-", 1)[0]
    if head != text:
        return to_code(head)
    return None


def to_name(value: Any) -> str | None:

    code = to_code(value)
    return CODE_TO_NAME.get(code) if code else None
