
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .condition import MusicCondition
from .contracts import (
    MELODY_FRAME_RATE,
    MELODY_UNVOICED_ID,
    MELODY_VOCAB_SIZE,
    SEMANTIC_CODEBOOK_SIZE,
    SEMANTIC_FRAME_RATE,
    SEMANTIC_SAMPLE_RATE,
    SEQUENCE_PROTOCOL_REVISION,
    SequenceMode,
)

RENDER_REQUEST_SCHEMA_VERSION = "oqm.render.request.v1"
VALID_STOP_REASONS = frozenset(
    {"natural_eos", "frame_cap", "max_new_tokens", "unfinished"}
)


def prompt_token_sha256(token_ids: Sequence[int] | np.ndarray) -> str:
    values = np.asarray(token_ids, dtype="<i8").reshape(-1)
    return "sha256:" + hashlib.sha256(values.tobytes()).hexdigest()


def semantic_token_sha256(token_ids: Sequence[int] | np.ndarray) -> str:
    values = np.asarray(token_ids, dtype="<u2").reshape(-1)
    return "sha256:" + hashlib.sha256(values.tobytes()).hexdigest()


def _strict_integer_ids(values: Any, *, name: str) -> np.ndarray:
    if isinstance(values, (list, tuple)) and any(
        type(value) is not int for value in values
    ):
        raise ValueError(f"{name} must contain only JSON integers, not bools, floats, or strings")
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise ValueError(f"{name} must be an integer array")
    return array.astype(np.int64, copy=False)


def _envelope_sha256(payload: dict[str, Any]) -> str:
    canonical = {
        key: value for key, value in payload.items() if key != "envelope_sha256"
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RenderRequest:
    sample_id: str
    condition: MusicCondition
    semantic_ids: np.ndarray
    mode: str
    stop_reason: str
    finished: bool
    semantic_tokenizer_revision: str
    registry_revision: str
    condition_template_version: str
    prompt_sha256: str
    checkpoint: dict[str, Any]
    generation_config: dict[str, Any]
    melody_ids: np.ndarray | None = None
    melody_sections: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        semantic = _strict_integer_ids(self.semantic_ids, name="semantic_ids")
        melody = (
            None
            if self.melody_ids is None
            else _strict_integer_ids(self.melody_ids, name="melody_ids")
        )
        object.__setattr__(self, "semantic_ids", semantic)
        object.__setattr__(self, "melody_ids", melody)
        object.__setattr__(
            self,
            "melody_sections",
            tuple(str(value) for value in self.melody_sections),
        )
        if not self.sample_id:
            raise ValueError("RenderRequest sample_id cannot be empty")
        if semantic.size == 0:
            raise ValueError(f"{self.sample_id}: semantic_ids is empty")
        if bool((semantic < 0).any()) or bool(
            (semantic >= SEMANTIC_CODEBOOK_SIZE).any()
        ):
            raise ValueError(f"{self.sample_id}: semantic local ID out of bounds")
        if melody is not None and (
            bool((melody < 0).any()) or bool((melody >= MELODY_VOCAB_SIZE).any())
        ):
            raise ValueError(f"{self.sample_id}: melody local ID out of bounds")
        if not self.semantic_tokenizer_revision or (
            self.semantic_tokenizer_revision.lower() == "unknown"
        ):
            raise ValueError(f"{self.sample_id}: semantic tokenizer revision not frozen")
        if not self.registry_revision or not self.condition_template_version:
            raise ValueError(f"{self.sample_id}: registry/condition revision not frozen")
        if self.stop_reason not in VALID_STOP_REASONS:
            raise ValueError(f"{self.sample_id}: stop_reason={self.stop_reason!r} is invalid")
        if self.mode not in {mode.value for mode in SequenceMode}:
            raise ValueError(f"{self.sample_id}: mode={self.mode!r} is invalid")
        if type(self.finished) is not bool:
            raise ValueError(f"{self.sample_id}: finished must be a JSON bool")
        should_be_finished = self.stop_reason in {"natural_eos", "frame_cap"}
        if bool(self.finished) != should_be_finished:
            raise ValueError(
                f"{self.sample_id}: finished={self.finished} and "
                f"stop_reason={self.stop_reason!r} Contradiction"
            )
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.prompt_sha256):
            raise ValueError(f"{self.sample_id}: prompt_sha256 has an invalid format")

    def to_json(self) -> dict[str, Any]:
        payload = {
            "schema_version": RENDER_REQUEST_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "condition": self.condition.to_record(),
            "mode": self.mode,
            "semantic": {
                "ids": self.semantic_ids.tolist(),
                "num_frames": int(self.semantic_ids.size),
                "frame_rate": SEMANTIC_FRAME_RATE,
                "sample_rate": SEMANTIC_SAMPLE_RATE,
                "codebook_size": SEMANTIC_CODEBOOK_SIZE,
                "sha256": semantic_token_sha256(self.semantic_ids),
                "tokenizer_revision": self.semantic_tokenizer_revision,
            },
            "melody_plan": None
            if self.melody_ids is None
            else {
                "ids": self.melody_ids.tolist(),
                "num_frames": int(self.melody_ids.size),
                "frame_rate": MELODY_FRAME_RATE,
                "vocab_size": MELODY_VOCAB_SIZE,
                "unvoiced_id": MELODY_UNVOICED_ID,
                "sections": list(self.melody_sections),
            },
            "generation": {
                "finished": bool(self.finished),
                "stop_reason": self.stop_reason,
                "config": self.generation_config,
            },
            "provenance": {
                "registry_revision": self.registry_revision,
                "sequence_protocol_revision": SEQUENCE_PROTOCOL_REVISION,
                "condition_template_version": self.condition_template_version,
                "prompt_sha256": self.prompt_sha256,
                "checkpoint": self.checkpoint,
            },
        }
        payload["envelope_sha256"] = _envelope_sha256(payload)
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RenderRequest:
        version = payload.get("schema_version")
        if version != RENDER_REQUEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported RenderRequest version: {version!r}; "
                f"expects{RENDER_REQUEST_SCHEMA_VERSION!r}"
            )
        semantic = dict(payload.get("semantic") or {})
        melody = payload.get("melody_plan")
        generation = dict(payload.get("generation") or {})
        provenance = dict(payload.get("provenance") or {})
        if provenance.get("sequence_protocol_revision") != SEQUENCE_PROTOCOL_REVISION:
            raise ValueError("RenderRequest sequence protocol revision mismatch")
        expected_semantic = {
            "frame_rate": SEMANTIC_FRAME_RATE,
            "sample_rate": SEMANTIC_SAMPLE_RATE,
            "codebook_size": SEMANTIC_CODEBOOK_SIZE,
        }
        for name, expected in expected_semantic.items():
            if semantic.get(name) != expected:
                raise ValueError(
                    f"RenderRequest semantic.{name}={semantic.get(name)!r} != {expected!r}"
                )
        semantic_ids = _strict_integer_ids(
            semantic.get("ids") or [],
            name="semantic.ids",
        )
        semantic_num_frames = semantic.get("num_frames")
        if type(semantic_num_frames) is not int:
            raise ValueError("RenderRequest semantic.num_frames must be an integer")
        if semantic_num_frames != int(semantic_ids.size):
            raise ValueError("RenderRequest semantic.num_frames does not match the number of IDs")
        if melody is not None:
            melody_payload = dict(melody)
            expected_melody = {
                "frame_rate": MELODY_FRAME_RATE,
                "vocab_size": MELODY_VOCAB_SIZE,
                "unvoiced_id": MELODY_UNVOICED_ID,
            }
            for name, expected in expected_melody.items():
                if melody_payload.get(name) != expected:
                    raise ValueError(
                        f"RenderRequest melody_plan.{name}="
                        f"{melody_payload.get(name)!r} != {expected!r}"
                    )
            melody_ids = _strict_integer_ids(
                melody_payload.get("ids") or [],
                name="melody_plan.ids",
            )
            melody_num_frames = melody_payload.get("num_frames")
            if type(melody_num_frames) is not int:
                raise ValueError("RenderRequest melody num_frames must be an integer")
            if melody_num_frames != int(melody_ids.size):
                raise ValueError("RenderRequest melody num_frames does not match the number of IDs")
        finished = generation.get("finished", False)
        if type(finished) is not bool:
            raise ValueError("RenderRequest generation.finished must be a JSON bool")
        request = cls(
            sample_id=str(payload.get("sample_id") or ""),
            condition=MusicCondition.from_record(dict(payload.get("condition") or {})),
            semantic_ids=semantic_ids,
            melody_ids=(
                None
                if melody is None
                else melody_ids
            ),
            melody_sections=(
                () if melody is None else tuple(dict(melody).get("sections") or ())
            ),
            mode=str(payload.get("mode") or ""),
            stop_reason=str(generation.get("stop_reason") or "unfinished"),
            finished=finished,
            semantic_tokenizer_revision=str(
                semantic.get("tokenizer_revision") or "unknown"
            ),
            registry_revision=str(provenance.get("registry_revision") or ""),
            condition_template_version=str(
                provenance.get("condition_template_version") or ""
            ),
            prompt_sha256=str(provenance.get("prompt_sha256") or ""),
            checkpoint=dict(provenance.get("checkpoint") or {}),
            generation_config=dict(generation.get("config") or {}),
        )
        expected_semantic_hash = str(semantic.get("sha256") or "")
        if expected_semantic_hash != semantic_token_sha256(request.semantic_ids):
            raise ValueError(f"{request.sample_id}: semantic token SHA-256 mismatch")
        if str(payload.get("envelope_sha256") or "") != _envelope_sha256(payload):
            raise ValueError(f"{request.sample_id}: RenderRequest envelope SHA-256 mismatch")
        return request


def write_render_requests(
    path: str | Path,
    requests: Sequence[RenderRequest],
) -> Path:
    if not requests:
        raise ValueError("RenderRequest list cannot be empty")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for request in requests:
                handle.write(json.dumps(request.to_json(), ensure_ascii=False) + "\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_render_requests(
    path: str | Path,
    *,
    expected_tokenizer_revision: str | None = None,
    expected_condition_template_version: str | None = None,
    expected_registry_revision: str | None = None,
) -> list[RenderRequest]:
    requests: list[RenderRequest] = []
    seen_sample_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                request = RenderRequest.from_json(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number} invalid RenderRequest: {error}") from error
            if (
                expected_tokenizer_revision is not None
                and request.semantic_tokenizer_revision
                != str(expected_tokenizer_revision)
            ):
                raise RuntimeError(
                    f"{request.sample_id}: Render tokenizer revision mismatch"
                )
            if (
                expected_condition_template_version is not None
                and request.condition_template_version
                != str(expected_condition_template_version)
            ):
                raise RuntimeError(
                    f"{request.sample_id}: Render condition revision mismatch"
                )
            if (
                expected_registry_revision is not None
                and request.registry_revision != str(expected_registry_revision)
            ):
                raise RuntimeError(
                    f"{request.sample_id}: Render registry revision mismatch"
                )
            if request.sample_id in seen_sample_ids:
                raise RuntimeError(f"RenderRequest sample_id is duplicated: {request.sample_id}")
            seen_sample_ids.add(request.sample_id)
            requests.append(request)
    if not requests:
        raise ValueError(f"{path} contains no RenderRequest records")
    return requests
