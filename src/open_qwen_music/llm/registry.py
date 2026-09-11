
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import (
    MELODY_UNVOICED_ID,
    MELODY_VOCAB_SIZE,
    SECTION_LABELS,
    SEMANTIC_CODEBOOK_SIZE,
    SEQUENCE_PROTOCOL_REVISION,
)

REGISTRY_FORMAT_VERSION = "oqm.llm.token_registry.v2"


CONTROL_TOKEN_NAMES: tuple[str, ...] = (
    "bos",
    "eos",
    "pad",
    "task_t2m",
    "mode_plain",
    "mode_section",
    "mode_unique_section",
    "cond_bos",
    "cond_eos",
    "melody_bos",
    "melody_eos",
    "music_bos",
    "music_eos",
    "seg_end",
) + tuple(f"section_{label}" for label in SECTION_LABELS)


def control_token_text(name: str) -> str:
    return f"<|oqm_{name}|>"


@dataclass(frozen=True)
class TokenRegistry:

    text_vocab_size: int
    control_base: int
    num_control: int
    semantic_base: int
    semantic_size: int
    melody_base: int
    melody_size: int
    total_vocab_size: int
    control_names: tuple[str, ...] = field(default=CONTROL_TOKEN_NAMES)


    @classmethod
    def build(
        cls,
        *,
        text_vocab_size: int,
        semantic_size: int = SEMANTIC_CODEBOOK_SIZE,
        melody_size: int = MELODY_VOCAB_SIZE,
        control_names: tuple[str, ...] = CONTROL_TOKEN_NAMES,
        pad_to_multiple_of: int = 256,
    ) -> TokenRegistry:
        if text_vocab_size <= 0:
            raise ValueError(f"text_vocab_size must be positive; got {text_vocab_size}")
        if len(set(control_names)) != len(control_names):
            raise ValueError("Control token names must be unique")
        control_base = text_vocab_size
        semantic_base = control_base + len(control_names)
        melody_base = semantic_base + semantic_size
        raw_total = melody_base + melody_size
        if pad_to_multiple_of > 1:
            blocks = (raw_total + pad_to_multiple_of - 1) // pad_to_multiple_of
            total = blocks * pad_to_multiple_of
        else:
            total = raw_total
        return cls(
            text_vocab_size=text_vocab_size,
            control_base=control_base,
            num_control=len(control_names),
            semantic_base=semantic_base,
            semantic_size=semantic_size,
            melody_base=melody_base,
            melody_size=melody_size,
            total_vocab_size=total,
            control_names=tuple(control_names),
        )


    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": REGISTRY_FORMAT_VERSION,
            "sequence_protocol_revision": SEQUENCE_PROTOCOL_REVISION,
            "text_vocab_size": self.text_vocab_size,
            "control_base": self.control_base,
            "num_control": self.num_control,
            "semantic_base": self.semantic_base,
            "semantic_size": self.semantic_size,
            "melody_base": self.melody_base,
            "melody_size": self.melody_size,
            "total_vocab_size": self.total_vocab_size,
            "control_names": list(self.control_names),
            "section_labels": list(SECTION_LABELS),
            "melody_unvoiced_local_id": MELODY_UNVOICED_ID,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TokenRegistry:
        version = payload.get("format_version")
        if version != REGISTRY_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported registry format version: {version} "
                f"(expected {REGISTRY_FORMAT_VERSION})"
            )
        sequence_revision = payload.get("sequence_protocol_revision")
        if sequence_revision != SEQUENCE_PROTOCOL_REVISION:
            raise ValueError(
                f"Unsupported registry sequence protocol: {sequence_revision} "
                f"(expected {SEQUENCE_PROTOCOL_REVISION})"
            )
        section_labels = tuple(payload.get("section_labels", ()))
        if section_labels != SECTION_LABELS:
            raise ValueError(
                f"Registry section taxonomy does not match: {section_labels} "
                f"(expected {SECTION_LABELS})"
            )
        melody_unvoiced_id = int(payload.get("melody_unvoiced_local_id", -1))
        if melody_unvoiced_id != MELODY_UNVOICED_ID:
            raise ValueError(
                "registry melody unvoiced id does not match:"
                f"{melody_unvoiced_id}(expects {MELODY_UNVOICED_ID})"
            )
        control_names = tuple(payload["control_names"])
        retired = {"task_cover", "ref_melody_bos", "ref_melody_eos"} & set(control_names)
        if retired:
            raise ValueError(f"registry contains retired cover control tokens: {sorted(retired)}")
        num_control = int(payload["num_control"])
        if num_control != len(control_names):
            raise ValueError(
                f"registry num_control={num_control}, but control_names has "
                f"{len(control_names)} entries"
            )
        return cls(
            text_vocab_size=int(payload["text_vocab_size"]),
            control_base=int(payload["control_base"]),
            num_control=num_control,
            semantic_base=int(payload["semantic_base"]),
            semantic_size=int(payload["semantic_size"]),
            melody_base=int(payload["melody_base"]),
            melody_size=int(payload["melody_size"]),
            total_vocab_size=int(payload["total_vocab_size"]),
            control_names=control_names,
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["revision"] = self.revision
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> TokenRegistry:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        stored = payload.pop("revision", None)
        registry = cls.from_dict(payload)
        if stored is not None and stored != registry.revision:
            raise ValueError(
                "registry.json content does not match its revision: "
                f"stored={stored}, calculated={registry.revision}"
            )
        return registry

    @property
    def revision(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()[:32]


    def control(self, name: str) -> int:
        try:
            return self.control_base + self.control_names.index(name)
        except ValueError as error:
            raise KeyError(f"Unknown control token: {name}") from error

    def section_control(self, label: str) -> int:
        if label not in SECTION_LABELS:
            raise KeyError(f"Unknown section label: {label} (expected {SECTION_LABELS})")
        return self.control(f"section_{label}")

    @property
    def control_ids(self) -> tuple[int, ...]:
        return tuple(range(self.control_base, self.control_base + self.num_control))

    @property
    def section_control_ids(self) -> tuple[int, ...]:
        return tuple(self.section_control(label) for label in SECTION_LABELS)


    @property
    def bos_id(self) -> int:
        return self.control("bos")

    @property
    def eos_id(self) -> int:
        return self.control("eos")

    @property
    def pad_id(self) -> int:

        return self.control("pad")


    def semantic_to_global(self, local_id: int) -> int:
        if not 0 <= local_id < self.semantic_size:
            raise ValueError(f"semantic local id out of bounds:{local_id}")
        return self.semantic_base + local_id

    def melody_to_global(self, local_id: int) -> int:
        if not 0 <= local_id < self.melody_size:
            raise ValueError(f"melody local id out of bounds:{local_id}")
        return self.melody_base + local_id

    def is_text(self, global_id: int) -> bool:
        return 0 <= global_id < self.text_vocab_size

    def is_control(self, global_id: int) -> bool:
        return self.control_base <= global_id < self.control_base + self.num_control

    def is_semantic(self, global_id: int) -> bool:
        return self.semantic_base <= global_id < self.semantic_base + self.semantic_size

    def is_melody(self, global_id: int) -> bool:
        return self.melody_base <= global_id < self.melody_base + self.melody_size

    def namespace_of(self, global_id: int) -> str:
        if self.is_text(global_id):
            return "text"
        if self.is_control(global_id):
            return "control"
        if self.is_semantic(global_id):
            return "semantic"
        if self.is_melody(global_id):
            return "melody"
        return "reserved"

    def local_of(self, global_id: int) -> int | None:
        if self.is_semantic(global_id):
            return global_id - self.semantic_base
        if self.is_melody(global_id):
            return global_id - self.melody_base
        if self.is_control(global_id):
            return global_id - self.control_base
        return None

    def describe(self, global_id: int) -> str:
        namespace = self.namespace_of(global_id)
        if namespace == "control":
            return control_token_text(self.control_names[global_id - self.control_base])
        if namespace == "semantic":
            return f"<|oqm_sem_{global_id - self.semantic_base:05d}|>"
        if namespace == "melody":
            local = global_id - self.melody_base
            suffix = "_unvoiced" if local == MELODY_UNVOICED_ID else ""
            return f"<|oqm_mel_{local:03d}{suffix}|>"
        if namespace == "text":
            return f"<text:{global_id}>"
        return f"<reserved:{global_id}>"


def build_registry_from_config(config: dict[str, Any], text_vocab_size: int) -> TokenRegistry:
    section = config.get("registry", {}) or {}
    resolved_path = section.get("resolved_path")
    if resolved_path:
        registry = TokenRegistry.load(Path(str(resolved_path)).resolve(strict=True))
        if registry.text_vocab_size != int(text_vocab_size):
            raise RuntimeError(
                "Resolved registry text_vocab_size does not match the base model: "
                f"{registry.text_vocab_size} != {text_vocab_size}"
            )
        expected_revision = str(section.get("resolved_revision") or "")
        if expected_revision and registry.revision != expected_revision:
            raise RuntimeError(
                "resolved registry revision mismatch:"
                f"{registry.revision} != {expected_revision}"
            )
        return registry
    semantic_size = int(section.get("semantic_size", SEMANTIC_CODEBOOK_SIZE))
    melody_size = int(section.get("melody_size", MELODY_VOCAB_SIZE))
    if semantic_size != SEMANTIC_CODEBOOK_SIZE:
        raise ValueError(
            "registry.semantic_size does not match the frozen semantic contract: "
            f"{semantic_size} != {SEMANTIC_CODEBOOK_SIZE}."
            " If the tokenizer codebook size changes, update the contracts and registry "
            "protocol before changing this value."
        )
    if melody_size != MELODY_VOCAB_SIZE:
        raise ValueError(
            "registry.melody_size does not match the frozen melody contract: "
            f"{melody_size} != {MELODY_VOCAB_SIZE}."
            " Update the contracts and registry protocol before changing this value."
        )
    return TokenRegistry.build(
        text_vocab_size=text_vocab_size,
        semantic_size=semantic_size,
        melody_size=melody_size,
        pad_to_multiple_of=int(section.get("pad_to_multiple_of", 256)),
    )
