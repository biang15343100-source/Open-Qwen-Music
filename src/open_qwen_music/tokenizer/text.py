
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path


CTC_PHONEME_SPEC_VERSION = "oqm.ctc-phoneme.zh-en.v1"
CTC_SUBWORD_SPEC_VERSION = "oqm.ctc-subword.multilingual.v1"


ENGLISH_PHONEMES = [
    "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2",
    "AO0", "AO1", "AO2", "AW1", "AW2", "AY0", "AY1", "AY2",
    "B", "CH", "D", "DH", "EH0", "EH1", "EH2", "ER0", "ER1",
    "ER2", "EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1",
    "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N",
    "NG", "OW0", "OW1", "OW2", "OY1", "P", "R", "S", "SH", "T",
    "TH", "UH0", "UH1", "UH2", "UW0", "UW1", "UW2", "V", "W",
    "Y", "Z", "ZH",
]
CHINESE_PHONEMES = [
    "a", "ai", "an", "ang", "ao", "b", "c", "ch", "d", "e", "ei",
    "en", "eng", "er", "f", "g", "h", "i", "ia", "ian", "iang",
    "iao", "ie", "in", "ing", "iong", "iou", "j", "k", "l", "m",
    "n", "o", "ong", "ou", "p", "q", "r", "s", "sh", "t", "u",
    "ua", "uai", "uan", "uang", "uei", "uen", "uo", "v", "van",
    "ve", "vn", "x", "z", "zh",
]
FIXED_ZH_EN_PHONEME_VOCAB = [
    "<blank>",
    "<unk>",
    *ENGLISH_PHONEMES,
    *CHINESE_PHONEMES,
]
FIXED_ZH_EN_PHONEME_SET = set(FIXED_ZH_EN_PHONEME_VOCAB[2:])




ENGLISH_PHONEME_FALLBACKS = {
    "AA0": "AA1",
    "AW0": "AW1",
    "OY0": "OY1",
    "OY2": "OY1",
}

_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _canonical_english_phoneme(unit: str) -> str | None:

    candidate = unit.upper()
    if candidate in FIXED_ZH_EN_PHONEME_SET:
        return candidate
    return ENGLISH_PHONEME_FALLBACKS.get(candidate)


def normalize_lyrics(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(text.split())


def is_cjk(char: str) -> bool:
    return len(char) == 1 and 0x3400 <= ord(char) <= 0x9FFF


@lru_cache(maxsize=1)
def _cmu_dictionary() -> dict[str, list[list[str]]]:
    try:
        import cmudict
    except ImportError as exc:
        raise RuntimeError(
            "English G2P requires cmudict; install the project dependencies"
        ) from exc
    return cmudict.dict()


def english_to_phonemes(text: str) -> tuple[list[str], int]:

    dictionary = _cmu_dictionary()
    units: list[str] = []
    unknown = 0
    for word in _ENGLISH_WORD_RE.findall(text):
        pronunciations = dictionary.get(word.casefold())
        if not pronunciations:
            units.append("<unk>")
            unknown += 1
            continue


        pronunciation = []
        for unit in pronunciations[0]:
            canonical = _canonical_english_phoneme(unit)
            if canonical is None:
                pronunciation.append("<unk>")
                unknown += 1
            else:
                pronunciation.append(canonical)
        if pronunciation:
            units.extend(pronunciation)
        else:
            units.append("<unk>")
            unknown += 1
    return units, unknown


def chinese_to_phonemes(text: str) -> tuple[list[str], int]:

    try:
        from pypinyin import Style, pinyin
    except ImportError as exc:
        raise RuntimeError(
            "Chinese G2P requires pypinyin; install the project dependencies"
        ) from exc
    chars = [char for char in text if is_cjk(char)]
    if not chars:
        return [], 0
    initials = pinyin(
        chars,
        style=Style.INITIALS,
        strict=True,
        errors=lambda value: ["" for _ in value],
    )
    finals = pinyin(
        chars,
        style=Style.FINALS,
        strict=True,
        errors=lambda value: ["" for _ in value],
    )
    units: list[str] = []
    unknown = 0
    for initial, final in zip(initials, finals):
        syllable_units = []
        if initial and initial[0] in FIXED_ZH_EN_PHONEME_SET:
            syllable_units.append(initial[0])
        if final and final[0] in FIXED_ZH_EN_PHONEME_SET:
            syllable_units.append(final[0])
        if syllable_units:
            units.extend(syllable_units)
        else:
            units.append("<unk>")
            unknown += 1
    return units, unknown


def lyrics_to_phonemes(text: str) -> tuple[list[str], dict[str, int | str]]:

    normalized = unicodedata.normalize("NFKC", text)
    units: list[str] = []
    unknown_words = 0
    unknown_chars = 0
    cursor = 0
    for match in _ENGLISH_WORD_RE.finditer(normalized):
        prefix = normalized[cursor : match.start()]
        zh_units, zh_unknown = chinese_to_phonemes(prefix)
        en_units, en_unknown = english_to_phonemes(match.group(0))
        units.extend(zh_units)
        units.extend(en_units)
        unknown_chars += zh_unknown
        unknown_words += en_unknown
        cursor = match.end()
    zh_units, zh_unknown = chinese_to_phonemes(normalized[cursor:])
    units.extend(zh_units)
    unknown_chars += zh_unknown
    return units, {
        "spec_version": CTC_PHONEME_SPEC_VERSION,
        "unknown_english_words": unknown_words,
        "unknown_chinese_chars": unknown_chars,
    }


def normalize_existing_phonemes(
    units: list[str], language: str
) -> tuple[list[str], int]:

    normalized: list[str] = []
    unknown = 0
    english = "english" in language.casefold() or language.casefold().startswith("en")
    for value in units:
        unit = unicodedata.normalize("NFKC", str(value)).strip()
        if not unit or unit.upper() in {"SP", "AP", "<SP>", "<AP>"}:
            continue
        candidate = unit.upper() if english else unit.lower()
        if candidate not in FIXED_ZH_EN_PHONEME_SET and english:


            candidate = ENGLISH_PHONEME_FALLBACKS.get(candidate, candidate)
        if candidate in FIXED_ZH_EN_PHONEME_SET:
            normalized.append(candidate)
        else:
            normalized.append("<unk>")
            unknown += 1
    return normalized, unknown


class CharacterTokenizer:

    def __init__(
        self,
        tokens: list[str],
        *,
        kind: str = "units",
        tokenizer_json: str | Path | None = None,
    ) -> None:
        if not tokens or tokens[0] != "<blank>":
            raise ValueError("CTC vocabulary item 0 must be <blank>")
        self.tokens = tokens
        self.token_to_id = {token: index for index, token in enumerate(tokens)}
        self.unk_id = self.token_to_id.get("<unk>", 1)
        self.kind = kind
        self.tokenizer_json = (
            Path(tokenizer_json) if tokenizer_json is not None else None
        )
        self._backend = None

    @classmethod
    def from_file(cls, path: str | Path) -> "CharacterTokenizer":
        path = Path(path)
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return cls(payload)
            if payload.get("format_version") == "oqm.ctc-subword-vocab.v1":
                tokenizer_json = Path(payload["tokenizer_json"])
                if not tokenizer_json.is_absolute():
                    tokenizer_json = path.parent / tokenizer_json
                return cls(
                    list(payload["tokens"]),
                    kind="subword",
                    tokenizer_json=tokenizer_json,
                )
            raise ValueError(f"UnsupportedCTCvocabulary format: {path}")
        return cls(
            [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
        )

    def _subword_backend(self):
        if self.kind != "subword" or self.tokenizer_json is None:
            raise RuntimeError("currentCTCvocabulary is notsubword tokenizer")
        if self._backend is None:
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "runtime textsubwordencoding/Decoding requires tokenizers>=0.20"
                ) from exc
            self._backend = Tokenizer.from_file(str(self.tokenizer_json))
        return self._backend

    def encode(self, text: str) -> list[int]:
        if self.kind == "subword":
            if not text.strip():
                return []

            return [
                index + 1
                for index in self._subword_backend().encode(
                    normalize_lyrics(text)
                ).ids
            ]
        return [self.token_to_id.get(char, self.unk_id) for char in normalize_lyrics(text)]

    def encode_units(self, units: list[str | int]) -> list[int]:
        ids: list[int] = []
        for unit in units:
            if isinstance(unit, bool):
                raise TypeError("CTC unit must not be bool")
            if isinstance(unit, int):
                if unit <= 0 or unit >= len(self):
                    raise ValueError(f"CTC token ID out of bounds or use blank: {unit}")
                ids.append(unit)
                continue
            ids.append(self.token_to_id.get(str(unit), self.unk_id))
        return ids

    def token(self, index: int) -> str:
        if 0 <= index < len(self.tokens):
            return self.tokens[index]
        return f"<invalid:{index}>"

    def decode_units(self, ids: list[int]) -> list[str]:
        return [self.token(index) for index in ids]

    def decode_text(self, ids: list[int]) -> str:
        if self.kind == "subword":
            backend_ids = [index - 1 for index in ids if index > 0]
            return normalize_lyrics(
                self._subword_backend().decode(
                    backend_ids, skip_special_tokens=True
                )
            )
        return " ".join(self.decode_units(ids))

    def __len__(self) -> int:
        return len(self.tokens)
