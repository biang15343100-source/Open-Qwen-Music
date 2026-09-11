
from __future__ import annotations

import re
import unicodedata

GENRES: frozenset[str] = frozenset(
    """
    pop rock indie folk country blues jazz soul funk disco rnb hiphop rap trap
    electronic house techno trance dubstep drum-and-bass ambient downtempo
    synthwave lofi edm dance classical orchestral opera choral piano-solo
    metal punk hardcore grunge alternative post-rock progressive
    reggae ska latin salsa bossa-nova tango flamenco afrobeat world
    mandopop cantopop c-pop j-pop k-pop city-pop chinese-folk chinese-traditional
    gospel soundtrack cinematic new-age experimental noise industrial
    ballad acoustic singer-songwriter musical-theatre childrens
    """.split()
)

MOODS: frozenset[str] = frozenset(
    """
    happy sad melancholic nostalgic romantic dreamy peaceful calm relaxing
    uplifting hopeful triumphant epic dramatic dark tense anxious angry
    aggressive energetic upbeat playful funny quirky sexy sensual
    lonely bittersweet longing warm cold ethereal mysterious spiritual
    motivational confident rebellious groovy chill soothing sentimental
    """.split()
)

INSTRUMENTS: frozenset[str] = frozenset(
    """
    acoustic-guitar electric-guitar bass-guitar double-bass piano electric-piano
    organ synthesizer keyboard accordion harmonica
    drums drum-machine percussion tambourine
    violin viola cello strings harp
    trumpet trombone saxophone flute clarinet oboe brass woodwind
    erhu guzheng pipa dizi suona guqin
    banjo mandolin ukulele sitar
    choir vocals male-vocals female-vocals
    """.split()
)

TIMBRES: frozenset[str] = frozenset(
    """
    warm bright dark husky raspy breathy airy smooth silky clear crisp
    powerful soft gentle thin full rich nasal gritty rough sweet
    deep high-pitched low-pitched mellow sharp resonant ethereal
    """.split()
)


ALIASES: dict[str, str] = {
    # genre
    "hip-hop": "hiphop",
    "r&b": "rnb",
    "r-b": "rnb",
    "rhythm-and-blues": "rnb",
    "drum-n-bass": "drum-and-bass",
    "dnb": "drum-and-bass",
    "d&b": "drum-and-bass",
    "electronica": "electronic",
    "electro": "electronic",
    "electronic-dance-music": "edm",
    "lo-fi": "lofi",
    "lo-fi-hip-hop": "lofi",
    "trip-hop": "downtempo",
    "heavy-metal": "metal",
    "death-metal": "metal",
    "black-metal": "metal",
    "alt-rock": "alternative",
    "alternative-rock": "alternative",
    "prog-rock": "progressive",
    "progressive-rock": "progressive",
    "post-punk": "punk",
    "film-score": "soundtrack",
    "film-music": "soundtrack",
    "score": "soundtrack",
    "orchestra": "orchestral",
    "symphonic": "orchestral",
    "mando-pop": "mandopop",
    "mandarin-pop": "mandopop",
    "chinese-pop": "c-pop",
    "canto-pop": "cantopop",
    "cantonese-pop": "cantopop",

    "citypop": "city-pop",
    "j-rock": "rock",
    "bossa": "bossa-nova",
    "afro-beat": "afrobeat",
    "world-music": "world",
    "singer-songwriter-folk": "singer-songwriter",
    "power-ballad": "ballad",
    "slow-ballad": "ballad",
    "childrens-music": "childrens",
    "children": "childrens",
    "kids": "childrens",
    "musical": "musical-theatre",
    "broadway": "musical-theatre",
    "new-wave": "alternative",
    "chillout": "chill",

    "joyful": "happy",
    "cheerful": "happy",
    "sorrowful": "sad",
    "somber": "melancholic",
    "melancholy": "melancholic",
    "wistful": "nostalgic",
    "reflective": "nostalgic",
    "loving": "romantic",
    "tender": "gentle",
    "serene": "peaceful",
    "tranquil": "peaceful",
    "laid-back": "chill",
    "inspiring": "uplifting",
    "optimistic": "hopeful",
    "victorious": "triumphant",
    "heroic": "epic",
    "theatrical": "dramatic",
    "moody": "dark",
    "gloomy": "dark",
    "suspenseful": "tense",
    "nervous": "anxious",
    "furious": "angry",
    "intense": "aggressive",
    "lively": "energetic",
    "driving": "energetic",
    "cheeky": "playful",
    "humorous": "funny",
    "quirky-fun": "quirky",
    "seductive": "sensual",
    "isolated": "lonely",
    "yearning": "longing",
    "otherworldly": "ethereal",
    "meditative": "spiritual",
    "empowering": "motivational",
    "inspirational": "motivational",
    "defiant": "rebellious",
    "funky": "groovy",
    "calming": "soothing",


    "emotional": "sentimental",
    # instrument
    "guitar": "acoustic-guitar",
    "e-guitar": "electric-guitar",
    "electricguitar": "electric-guitar",
    "distorted-guitar": "electric-guitar",
    "bass": "bass-guitar",
    "upright-bass": "double-bass",
    "contrabass": "double-bass",
    "grand-piano": "piano",
    "rhodes": "electric-piano",
    "wurlitzer": "electric-piano",
    "hammond-organ": "organ",
    "synth": "synthesizer",
    "synths": "synthesizer",
    "pad": "synthesizer",
    "keys": "keyboard",
    "drum-kit": "drums",
    "drum-set": "drums",
    "drumkit": "drums",
    "808": "drum-machine",
    "beats": "drum-machine",
    "congas": "percussion",
    "bongos": "percussion",
    "shaker": "percussion",
    "string-section": "strings",
    "string-ensemble": "strings",
    "fiddle": "violin",
    "violoncello": "cello",
    "sax": "saxophone",
    "horns": "brass",
    "horn-section": "brass",
    "brass-section": "brass",
    "woodwinds": "woodwind",
    "guzheng-zither": "guzheng",
    "chinese-flute": "dizi",
    "bamboo-flute": "dizi",
    "choir-vocals": "choir",
    "backing-vocals": "vocals",
    "lead-vocals": "vocals",
    "voice": "vocals",
    "female-voice": "female-vocals",
    "male-voice": "male-vocals",
    # timbre
    "warm-tone": "warm",
    "bright-tone": "bright",
    "hoarse": "husky",
    "gravelly": "raspy",
    "whispery": "breathy",
    "light": "airy",
    "velvety": "silky",
    "strong": "powerful",
    "belting": "powerful",
    "delicate": "soft",
    "thin-tone": "thin",
    "full-bodied": "full",
    "grainy": "gritty",
    "coarse": "rough",
    "honeyed": "sweet",
    "low": "low-pitched",
    "high": "high-pitched",


    #


    #


    #


    "bass-voice": "vocals",
    "piercing": "sharp",
}

VOCABULARIES: dict[str, frozenset[str]] = {
    "genre": GENRES,
    "mood": MOODS,
    "instrument": INSTRUMENTS,
    "vocal_timbre": TIMBRES,
}

_PUNCT = re.compile(r"[^\w\s&+-]", re.UNICODE)
_SPACE = re.compile(r"[\s_/]+")
_DASHES = re.compile(r"-{2,}")

_GENDER_ALIASES: dict[str, str] = {
    "female": "female",
    "f": "female",
    "woman": "female",
    "women": "female",
    "female-vocals": "female",
    "female-vocal": "female",
    "female-singer": "female",
    "soprano": "female",
    "male": "male",
    "m": "male",
    "man": "male",
    "men": "male",
    "male-vocals": "male",
    "male-vocal": "male",
    "male-singer": "male",
    "baritone": "male",
    "tenor": "male",
    "mixed": "mixed",
    "duet": "mixed",
    "both": "mixed",
    "male-and-female": "mixed",
    "mixed-vocals": "mixed",
    "group": "mixed",
    "choir": "mixed",
    "instrumental": "instrumental",
    "none": "instrumental",
    "no-vocals": "instrumental",
    "no-vocal": "instrumental",
}


def _normalize_token(raw: str) -> str:

    text = unicodedata.normalize("NFKC", str(raw)).strip().lower()
    text = _SPACE.sub("-", text)
    text = _PUNCT.sub("", text)
    text = _DASHES.sub("-", text).strip("-")
    return text


def normalize_tag(raw: str, field: str) -> str | None:

    vocabulary = VOCABULARIES.get(field)
    if vocabulary is None:
        return None
    token = _normalize_token(raw)
    if not token:
        return None
    token = ALIASES.get(token, token)
    if token in vocabulary:
        return token

    parts = token.split("-")
    for size in range(len(parts) - 1, 0, -1):
        for start in range(0, len(parts) - size + 1):
            candidate = "-".join(parts[start : start + size])
            candidate = ALIASES.get(candidate, candidate)
            if candidate in vocabulary:
                return candidate
    return None


def normalize_tag_list_detailed(
    values: object, field: str, *, limit: int = 8
) -> tuple[list[str], list[str], list[str]]:

    if values is None:
        return [], [], []
    if isinstance(values, str):

        separators = ",;" + "".join(chr(codepoint) for codepoint in (0x3001, 0xFF0C, 0xFF1B))
        items: list[str] = re.split(f"[{re.escape(separators)}]", values)
    elif isinstance(values, (list, tuple, set)):
        items = [str(v) for v in values]
    else:
        return [], [], []

    mapped: list[str] = []
    unmapped: list[str] = []
    truncated: list[str] = []
    seen: set[str] = set()
    for item in items:
        raw = str(item).strip()
        if not raw:
            continue
        canonical = normalize_tag(raw, field)
        if canonical is None:
            if raw.lower() not in {u.lower() for u in unmapped}:
                unmapped.append(raw)
            continue
        if canonical in seen:
            continue
        seen.add(canonical)


        if len(mapped) < limit:
            mapped.append(canonical)
        else:
            truncated.append(canonical)
    return mapped, unmapped, truncated


def normalize_tag_list(
    values: object, field: str, *, limit: int = 8
) -> tuple[list[str], list[str]]:

    mapped, unmapped, _truncated = normalize_tag_list_detailed(
        values, field, limit=limit
    )
    return mapped, unmapped


def normalize_gender(raw: object) -> str | None:

    if raw is None:
        return None
    token = _normalize_token(str(raw))
    if not token:
        return None
    if token in _GENDER_ALIASES:
        return _GENDER_ALIASES[token]

    has_female = "female" in token or "woman" in token
    has_male = re.search(r"(?<!fe)male|(?<!wo)man", token) is not None
    if has_female and has_male:
        return "mixed"
    if has_female:
        return "female"
    if has_male:
        return "male"
    if "instrumental" in token:
        return "instrumental"
    return None
