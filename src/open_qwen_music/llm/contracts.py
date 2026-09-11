
from __future__ import annotations

from enum import Enum


SEMANTIC_SAMPLE_RATE = 24_000
SEMANTIC_FRAME_RATE = 25.0
SEMANTIC_FRAME_SAMPLES = 960
SEMANTIC_CODEBOOK_SIZE = 32_768
SEMANTIC_TOKEN_MIN = 0
SEMANTIC_TOKEN_MAX = SEMANTIC_CODEBOOK_SIZE - 1
SEMANTIC_BITRATE_BPS = 375.0


# Eq.(3)  f~ = MedianPool8(f),"unvoiced frames are ignored when computing the median
#         and a pooled frame is set to 0 if the corresponding window is mostly unvoiced"
# Eq.(4)  m_bar = median{round(hz2midi(f~_i)) | f~_i > 0}


MELODY_SOURCE_FRAME_RATE = 50.0
MELODY_POOL_SIZE = 8
MELODY_FRAME_RATE = 6.25
MELODY_VOCAB_SIZE = 256
MELODY_UNVOICED_ID = 255
MELODY_RELATIVE_MIDI_OFFSET = 127


MELODY_RELATIVE_SEMITONE_LIMIT = 127
MELODY_TOKEN_MIN = 0
MELODY_TOKEN_MAX = MELODY_VOCAB_SIZE - 1


MELODY_VOICED_TOKEN_MAX = MELODY_UNVOICED_ID - 1


SEMANTIC_FRAMES_PER_MELODY_FRAME = 4


#: ``TASK_T2M + MODE_SECTION|MODE_UNIQUE_SECTION + MELODY_BOS...MELODY_EOS``.

SEQUENCE_PROTOCOL_REVISION = "oqm.llm.sequence.v2"


class SequenceMode(str, Enum):

    PLAIN = "plain"
    SECTION = "section"
    UNIQUE_SECTION = "unique_section"
    COVER_SECTION = "cover_section"
    COVER_UNIQUE_SECTION = "cover_unique_section"

    @property
    def is_cover(self) -> bool:
        return self in (SequenceMode.COVER_SECTION, SequenceMode.COVER_UNIQUE_SECTION)

    @property
    def has_melody(self) -> bool:
        return self is not SequenceMode.PLAIN

    @property
    def is_unique_section(self) -> bool:
        return self in (SequenceMode.UNIQUE_SECTION, SequenceMode.COVER_UNIQUE_SECTION)


SECTION_LABELS: tuple[str, ...] = (
    "intro",
    "verse",
    "chorus",
    "bridge",
    "instrumental",
    "outro",
    "silence",
)


#: and silence are omitted from the Melody-CoT sequence."
NON_VOCAL_SECTION_LABELS: frozenset[str] = frozenset({"intro", "instrumental", "outro", "silence"})


QUALITY_PERCENTILES: tuple[float, ...] = (90.0, 75.0, 50.0, 25.0, 5.0, 1.0)
QUALITY_BUCKETS: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7")
DISCARDED_QUALITY_BUCKET = "Q7"


def semantic_frames_for_seconds(seconds: float) -> int:
    if seconds < 0:
        raise ValueError(f"duration cannot be negative:{seconds}")
    return int(round(seconds * SEMANTIC_FRAME_RATE))


def melody_frames_for_seconds(seconds: float) -> int:
    if seconds < 0:
        raise ValueError(f"duration cannot be negative:{seconds}")
    return int(round(seconds * MELODY_FRAME_RATE))


def validate_semantic_contract(
    frame_rate: float, codebook_size: int, *, source: str = "semantic token"
) -> None:
    if float(frame_rate) != SEMANTIC_FRAME_RATE:
        raise ValueError(f"{source} frame rate must be {SEMANTIC_FRAME_RATE}; got {frame_rate}")
    if int(codebook_size) != SEMANTIC_CODEBOOK_SIZE:
        raise ValueError(f"{source} codebook must be {SEMANTIC_CODEBOOK_SIZE}; got {codebook_size}")


def validate_melody_contract(
    frame_rate: float, vocab_size: int, unvoiced_id: int, *, source: str = "melody token"
) -> None:
    if float(frame_rate) != MELODY_FRAME_RATE:
        raise ValueError(f"{source} frame rate must be {MELODY_FRAME_RATE}; got {frame_rate}")
    if int(vocab_size) != MELODY_VOCAB_SIZE:
        raise ValueError(f"{source} vocabulary must be {MELODY_VOCAB_SIZE}; got {vocab_size}")
    if int(unvoiced_id) != MELODY_UNVOICED_ID:
        raise ValueError(f"{source} unvoiced ID must be {MELODY_UNVOICED_ID}; got {unvoiced_id}")
