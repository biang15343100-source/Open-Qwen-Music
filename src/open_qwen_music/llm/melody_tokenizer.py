"""Complete audio -> RMVPE F0 -> Qwen-Music melody-token pipeline.

``open_qwen_music.llm.melody`` is the executable Eq.(3)/(4) reference.  This module
adds the missing neural F0 extractor and keeps the intermediate pitch curve,
confidence, and checkpoint provenance alongside the final uint8 tokens.

Important section-level rule
----------------------------
Eq.(4)'s MIDI median is global over the supplied pitch sequence.  For the
section-level Melody-CoT pattern, tokenize the *whole song once* and then slice
the resulting 6.25 Hz tokens by section boundaries.  Tokenizing every section
independently changes its center and does not reproduce the paper equation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import (
    MELODY_FRAME_RATE,
    MELODY_SOURCE_FRAME_RATE,
    MELODY_UNVOICED_ID,
    MELODY_VOCAB_SIZE,
)
from .melody import pitch_to_melody_tokens, validate_melody_tokens
from .rmvpe import PitchTrack, RMVPEPitchExtractor

MELODY_ALGORITHM_REVISION = "oqm.melody.eq3-4.v1"


@dataclass(frozen=True)
class MelodyTokenization:
    """A final token sequence plus auditable intermediate RMVPE output."""

    token_ids: np.ndarray
    frame_mask: np.ndarray
    pitch: PitchTrack
    tokenizer_revision: str

    def __post_init__(self) -> None:
        tokens = np.asarray(self.token_ids)
        mask = np.asarray(self.frame_mask)
        validate_melody_tokens(tokens)
        if mask.dtype != np.bool_:
            raise ValueError(f"frame_mask dtype must be bool, got {mask.dtype}")
        if mask.ndim != 1 or mask.shape != tokens.shape:
            raise ValueError("frame_mask must be one-dimensional and match token_ids")
        if not mask.all():
            raise ValueError(
                "single-item MelodyTokenization has no padding; frame_mask must be all true"
            )
        if not self.tokenizer_revision.startswith("sha256:"):
            raise ValueError("tokenizer_revision must be a sha256 revision")

    @property
    def frame_rate(self) -> float:
        return MELODY_FRAME_RATE

    @property
    def vocab_size(self) -> int:
        return MELODY_VOCAB_SIZE

    @property
    def unvoiced_id(self) -> int:
        return MELODY_UNVOICED_ID

    @property
    def num_frames(self) -> int:
        return int(self.token_ids.size)

    @property
    def has_voiced_melody(self) -> bool:
        return bool((self.token_ids != MELODY_UNVOICED_ID).any())

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": "oqm.melody.tokens.v1",
            "algorithm_revision": MELODY_ALGORITHM_REVISION,
            "tokenizer_revision": self.tokenizer_revision,
            "frame_rate": self.frame_rate,
            "vocab_size": self.vocab_size,
            "unvoiced_id": self.unvoiced_id,
            "num_frames": self.num_frames,
            "has_voiced_melody": self.has_voiced_melody,
            "pitch": self.pitch.metadata(),
        }

    def slice_seconds(self, start_sec: float, end_sec: float) -> np.ndarray:
        """Slice already-centered whole-song tokens without re-centering them."""

        if start_sec < 0 or end_sec <= start_sec:
            raise ValueError("expected 0 <= start_sec < end_sec")
        start = max(0, int(np.floor(start_sec * self.frame_rate)))
        end = min(self.num_frames, int(np.ceil(end_sec * self.frame_rate)))
        return self.token_ids[start:end].copy()

    def save(self, output: str | Path) -> tuple[Path, Path]:
        """Write compressed arrays and a JSON provenance sidecar."""

        path = Path(output)
        if path.suffix != ".npz":
            raise ValueError(f"MelodyTokenization output must end in .npz, got {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            token_ids=self.token_ids,
            frame_mask=self.frame_mask,
            pitch_hz_50=self.pitch.f0_hz.astype(np.float32, copy=False),
            pitch_confidence=self.pitch.confidence.astype(np.float32, copy=False),
        )
        sidecar = path.with_suffix(path.suffix + ".json")
        sidecar.write_text(
            json.dumps(self.metadata(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path, sidecar


class MelodyTokenizer:
    """Qwen-Music §2.3.1 Melody Tokenizer.

    The supplied ``RMVPEPitchExtractor`` fixes the otherwise unpublished
    checkpoint, preprocessing profile, voicing threshold, and chunking policy.
    Its provenance is part of ``tokenizer_revision``.
    """

    def __init__(self, pitch_extractor: RMVPEPitchExtractor) -> None:
        self.pitch_extractor = pitch_extractor
        revision_payload = {
            "algorithm_revision": MELODY_ALGORITHM_REVISION,
            "rmvpe": pitch_extractor.provenance(),
            # Paper-underspecified choices frozen by open_qwen_music.llm.melody:
            "median_even": "lower",
            "mostly_unvoiced": "strict_majority",
            "tail_window": "ceil_actual_length",
            "round_ties": "half_up_toward_positive_infinity",
            "hz_to_midi": "69+12*log2(hz/440)",
        }
        canonical = json.dumps(
            revision_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self.revision_payload = revision_payload
        self.revision = f"sha256:{hashlib.sha256(canonical).hexdigest()}"

    def encode_pitch(self, pitch: PitchTrack) -> MelodyTokenization:
        if not np.isclose(pitch.frame_rate, MELODY_SOURCE_FRAME_RATE):
            raise ValueError(
                f"Qwen Melody Tokenizer requires {MELODY_SOURCE_FRAME_RATE} Hz F0, "
                f"got {pitch.frame_rate}"
            )
        tokens = pitch_to_melody_tokens(pitch.f0_hz)
        validate_melody_tokens(tokens)
        return MelodyTokenization(
            token_ids=tokens,
            frame_mask=np.ones(tokens.shape, dtype=np.bool_),
            pitch=pitch,
            tokenizer_revision=self.revision,
        )

    def encode_waveform(
        self,
        waveform: np.ndarray | torch.Tensor,
        sample_rate: int,
    ) -> MelodyTokenization:
        return self.encode_pitch(self.pitch_extractor.extract(waveform, sample_rate))

    def encode_file(
        self,
        path: str | Path,
        *,
        start_sec: float = 0.0,
        duration_sec: float | None = None,
    ) -> MelodyTokenization:
        pitch = self.pitch_extractor.extract_file(
            path,
            start_sec=start_sec,
            duration_sec=duration_sec,
        )
        return self.encode_pitch(pitch)

    def encode_batch(
        self,
        waveforms: Sequence[np.ndarray | torch.Tensor],
        sample_rates: Sequence[int],
    ) -> list[MelodyTokenization]:
        return [
            self.encode_pitch(pitch)
            for pitch in self.pitch_extractor.extract_batch(waveforms, sample_rates)
        ]

    def provenance(self) -> dict[str, Any]:
        return {
            **self.revision_payload,
            "tokenizer_revision": self.revision,
        }
