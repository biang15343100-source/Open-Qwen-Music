
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import io
import json
import os
import sqlite3
import tempfile
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml

from .conditioning import (
    DESCRIPTION_MAX_TOKENS,
    LYRICS_MAX_TOKENS,
    REWRITER_SCHEMA_VERSION,
    SUPPORTED_LYRICS_MAX_TOKENS,
    FrozenQwenEmbeddingAdapter,
    TextEncoderProvenance,
    validate_cache_provenance,
)

CONFIG_FORMAT_VERSION = "oqm.render-text-cache-config.v1"
CACHE_FORMAT_VERSION = "oqm.render-text-cache.v2"
RENDER_SAMPLE_SCHEMA = "oqm.render-sample.v1"
RENDER_CONDITION_SCHEMA = "oqm.render-condition.v1"
TEXT_CACHE_READY_SCHEMA = "oqm.render-text-cache-ready.v1"
TEXT_CACHE_READY_STATUS = "TEXT_CACHE_READY"
RENDERER_DATA_TEXT_REFERENCE_SCHEMA = "oqm.render.renderer_data-text-cache-reference.v1"
QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

_STORAGE_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def canonical_json_bytes(value: Any) -> bytes:

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    digest = str(value)
    if len(digest) != 64:
        raise ValueError(f"{name} must be 64 bit SHA-256")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be SHA-256") from exc
    return digest.lower()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


@dataclass(frozen=True)
class RenderTextCacheConfig:

    cache_revision: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    hidden_size: int
    local_path: str = ""
    asset_lock: str = ""
    description_max_tokens: int = DESCRIPTION_MAX_TOKENS
    lyrics_max_tokens: int = LYRICS_MAX_TOKENS
    local_files_only: bool = True
    trust_remote_code: bool = False
    hidden_state_selection: str = "last_hidden_state"
    position_id_policy: str = "attention_mask_cumsum"
    encoder_use_cache: bool = False
    empty_text_policy: str = "zero_valid_tokens"
    frozen_eval_mode: bool = True
    use_fast_tokenizer: bool = True
    padding: bool = True
    truncation: bool = True
    truncation_policy: str = "reject"
    add_special_tokens: bool = True
    padding_side: str = "left"
    truncation_side: str = "right"
    output_format: str = "auto"
    storage_dtype: str = "float32"
    device: str = "cpu"
    distributed_backend: str = "gloo"
    distributed_timeout_seconds: float = 300.0
    format_version: str = CONFIG_FORMAT_VERSION

    def validate(self) -> None:
        string_fields = (
            "format_version",
            "cache_revision",
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "local_path",
            "asset_lock",
            "hidden_state_selection",
            "position_id_policy",
            "empty_text_policy",
            "truncation_policy",
            "padding_side",
            "truncation_side",
            "output_format",
            "storage_dtype",
            "device",
            "distributed_backend",
        )
        for name in string_fields:
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"text cache config {name} must be the string")
        integer_fields = (
            "hidden_size",
            "description_max_tokens",
            "lyrics_max_tokens",
        )
        for name in integer_fields:
            if not isinstance(getattr(self, name), int) or isinstance(
                getattr(self, name), bool
            ):
                raise TypeError(f"text cache config {name} must be an integer")
        for name in (
            "local_files_only",
            "trust_remote_code",
            "encoder_use_cache",
            "frozen_eval_mode",
            "use_fast_tokenizer",
            "padding",
            "truncation",
            "add_special_tokens",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"text cache config {name} must bebool")
        if (
            isinstance(self.distributed_timeout_seconds, bool)
            or not isinstance(self.distributed_timeout_seconds, (int, float))
        ):
            raise TypeError(
                "text cache config distributed_timeout_seconds must be a finite value"
            )
        if self.format_version != CONFIG_FORMAT_VERSION:
            raise ValueError(
                f"text cache config format is not compatible with:{self.format_version!r}"
            )
        self.provenance.validate()
        if self.hidden_size <= 0:
            raise ValueError("text encoder hidden_size must be positive")
        if self.description_max_tokens != DESCRIPTION_MAX_TOKENS:
            raise ValueError("description max token hard contract is 256")
        if self.lyrics_max_tokens not in SUPPORTED_LYRICS_MAX_TOKENS:
            raise ValueError(
                "lyrics max tokenonly supportsRendererData short=1536or"
                "Losslessfull parent=1792"
            )
        if self.local_files_only is not True:
            raise ValueError("Qwen Text cache prohibits implicit downloads,local_files_only must be true")
        if self.trust_remote_code is not False:
            raise ValueError("Qwen Text caching is not allowed trust_remote_code")
        if self.hidden_state_selection != "last_hidden_state":
            raise ValueError(
                "Text cachehidden_state_selectiononly supportslast_hidden_state"
            )
        if self.position_id_policy != "attention_mask_cumsum":
            raise ValueError(
                "Text cacheposition_id_policyonly supportsattention_mask_cumsum"
            )
        if self.encoder_use_cache is not False:
            raise ValueError("Text cacheencoder_use_cachemust befalse")
        if self.empty_text_policy != "zero_valid_tokens":
            raise ValueError(
                "Text cacheempty_text_policyonly supportszero_valid_tokens"
            )
        if self.frozen_eval_mode is not True:
            raise ValueError("Text cache frozenencodermust remainevalmode")
        if self.padding is not True or self.truncation is not True:
            raise ValueError("Text caching must be explicitly enabled padding and truncation")
        if self.truncation_policy not in {"reject", "allow"}:
            raise ValueError("truncation_policycan only bereject/allow")
        if self.add_special_tokens is not True:
            raise ValueError("Text cache must be retained tokenizer special tokens")
        if self.padding_side not in {"left", "right"}:
            raise ValueError("padding_side can only be left/right")
        if self.truncation_side not in {"left", "right"}:
            raise ValueError("truncation_side can only be left/right")
        if self.output_format not in {"auto", "safetensors", "npy"}:
            raise ValueError("output_format can only be auto/safetensors/npy")
        if self.storage_dtype not in _STORAGE_DTYPES:
            raise ValueError("storage_dtype can only be float16/float32/bfloat16")
        if not str(self.device).strip():
            raise ValueError("runtime.device cannot be empty")
        if self.distributed_backend != "gloo":
            raise ValueError("Text cache startup consensus fixed use gloo backend")
        if not np.isfinite(self.distributed_timeout_seconds) or (
            self.distributed_timeout_seconds <= 0
        ):
            raise ValueError("distributed_timeout_seconds must be a finite positive number")

    @property
    def provenance(self) -> TextEncoderProvenance:
        return TextEncoderProvenance(
            model_id=self.model_id,
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            cache_revision=self.cache_revision,
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "format_version": self.format_version,
            "cache": {
                "revision": self.cache_revision,
                "output_format": self.output_format,
                "storage_dtype": self.storage_dtype,
            },
            "text_encoder": {
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "tokenizer_revision": self.tokenizer_revision,
                "hidden_size": self.hidden_size,
                "local_path": self.local_path,
                "asset_lock": self.asset_lock,
                "local_files_only": self.local_files_only,
                "trust_remote_code": self.trust_remote_code,
                "hidden_state_selection": self.hidden_state_selection,
                "position_id_policy": self.position_id_policy,
                "encoder_use_cache": self.encoder_use_cache,
                "empty_text_policy": self.empty_text_policy,
                "frozen_eval_mode": self.frozen_eval_mode,
            },
            "tokenization": {
                "description_max_tokens": self.description_max_tokens,
                "lyrics_max_tokens": self.lyrics_max_tokens,
                "use_fast_tokenizer": self.use_fast_tokenizer,
                "padding": self.padding,
                "truncation": self.truncation,
                "truncation_policy": self.truncation_policy,
                "add_special_tokens": self.add_special_tokens,
                "padding_side": self.padding_side,
                "truncation_side": self.truncation_side,
            },
            "runtime": {
                "device": self.device,
                "distributed_backend": self.distributed_backend,
                "distributed_timeout_seconds": self.distributed_timeout_seconds,
            },
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RenderTextCacheConfig":
        if not isinstance(value, Mapping):
            raise TypeError("text cache config must be mapping")
        allowed_top = {
            "format_version",
            "cache",
            "text_encoder",
            "tokenization",
            "runtime",
        }
        unknown_top = set(value) - allowed_top
        if unknown_top:
            raise ValueError(f"text cache configUnknown top-level field:{sorted(unknown_top)}")
        cache = value.get("cache") or {}
        encoder = value.get("text_encoder") or {}
        tokenization = value.get("tokenization") or {}
        runtime = value.get("runtime") or {}
        for name, section in (
            ("cache", cache),
            ("text_encoder", encoder),
            ("tokenization", tokenization),
            ("runtime", runtime),
        ):
            if not isinstance(section, Mapping):
                raise TypeError(f"text cache config {name} must be mapping")
        allowed_sections = {
            "cache": {"revision", "output_format", "storage_dtype"},
            "text_encoder": {
                "model_id",
                "model_revision",
                "tokenizer_revision",
                "hidden_size",
                "local_path",
                "asset_lock",
                "local_files_only",
                "trust_remote_code",
                "hidden_state_selection",
                "position_id_policy",
                "encoder_use_cache",
                "empty_text_policy",
                "frozen_eval_mode",
            },
            "tokenization": {
                "description_max_tokens",
                "lyrics_max_tokens",
                "use_fast_tokenizer",
                "padding",
                "truncation",
                "truncation_policy",
                "add_special_tokens",
                "padding_side",
                "truncation_side",
            },
            "runtime": {
                "device",
                "distributed_backend",
                "distributed_timeout_seconds",
            },
        }
        for name, section in (
            ("cache", cache),
            ("text_encoder", encoder),
            ("tokenization", tokenization),
            ("runtime", runtime),
        ):
            unknown = set(section) - allowed_sections[name]
            if unknown:
                raise ValueError(f"text cache config {name}Unknown field:{sorted(unknown)}")

        def integer(name: str, candidate: Any) -> int:
            if not isinstance(candidate, int) or isinstance(candidate, bool):
                raise TypeError(f"text cache config {name} must be an integer")
            return candidate

        def number(name: str, candidate: Any) -> float:
            if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
                raise TypeError(f"text cache config {name} must be a numeric value")
            return float(candidate)

        result = cls(
            format_version=value.get("format_version", CONFIG_FORMAT_VERSION),
            cache_revision=cache.get("revision", value.get("cache_revision", "")),
            model_id=encoder.get("model_id", value.get("model_id", "")),
            model_revision=encoder.get(
                "model_revision", value.get("model_revision", "")
            ),
            tokenizer_revision=(
                encoder.get(
                    "tokenizer_revision",
                    value.get("tokenizer_revision", ""),
                )
            ),
            hidden_size=integer(
                "hidden_size",
                encoder.get("hidden_size", value.get("hidden_size", 0)),
            ),
            local_path=encoder.get("local_path", value.get("local_path", "")),
            asset_lock=encoder.get("asset_lock", value.get("asset_lock", "")),
            description_max_tokens=integer(
                "description_max_tokens",
                tokenization.get(
                    "description_max_tokens",
                    value.get("description_max_tokens", DESCRIPTION_MAX_TOKENS),
                ),
            ),
            lyrics_max_tokens=integer(
                "lyrics_max_tokens",
                tokenization.get(
                    "lyrics_max_tokens",
                    value.get("lyrics_max_tokens", LYRICS_MAX_TOKENS),
                ),
            ),
            local_files_only=encoder.get(
                "local_files_only", value.get("local_files_only", True)
            ),
            trust_remote_code=encoder.get(
                "trust_remote_code", value.get("trust_remote_code", False)
            ),
            hidden_state_selection=encoder.get(
                "hidden_state_selection",
                value.get("hidden_state_selection", "last_hidden_state"),
            ),
            position_id_policy=encoder.get(
                "position_id_policy",
                value.get("position_id_policy", "attention_mask_cumsum"),
            ),
            encoder_use_cache=encoder.get(
                "encoder_use_cache",
                value.get("encoder_use_cache", False),
            ),
            empty_text_policy=encoder.get(
                "empty_text_policy",
                value.get("empty_text_policy", "zero_valid_tokens"),
            ),
            frozen_eval_mode=encoder.get(
                "frozen_eval_mode",
                value.get("frozen_eval_mode", True),
            ),
            use_fast_tokenizer=tokenization.get(
                "use_fast_tokenizer",
                value.get("use_fast_tokenizer", True),
            ),
            padding=tokenization.get("padding", value.get("padding", True)),
            truncation=tokenization.get("truncation", value.get("truncation", True)),
            truncation_policy=(
                tokenization.get(
                    "truncation_policy",
                    value.get("truncation_policy", "reject"),
                )
            ),
            add_special_tokens=tokenization.get(
                "add_special_tokens",
                value.get("add_special_tokens", True),
            ),
            padding_side=(
                tokenization.get("padding_side", value.get("padding_side", "left"))
            ),
            truncation_side=(
                tokenization.get(
                    "truncation_side",
                    value.get("truncation_side", "right"),
                )
            ),
            output_format=(
                cache.get("output_format", value.get("output_format", "auto"))
            ),
            storage_dtype=(
                cache.get("storage_dtype", value.get("storage_dtype", "float32"))
            ),
            device=runtime.get("device", value.get("device", "cpu")),
            distributed_backend=(
                runtime.get(
                    "distributed_backend",
                    value.get("distributed_backend", "gloo"),
                )
            ),
            distributed_timeout_seconds=number(
                "distributed_timeout_seconds",
                runtime.get(
                    "distributed_timeout_seconds",
                    value.get("distributed_timeout_seconds", 300.0),
                ),
            ),
        )
        result.validate()
        return result


def load_text_cache_config(
    path: str | Path,
) -> tuple[RenderTextCacheConfig, dict[str, Any]]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError("text cache YAML The root node must be mapping")
    return RenderTextCacheConfig.from_mapping(raw), raw


@dataclass(frozen=True)
class RenderTextCondition:
    sample_id: str
    description: str
    lyrics: str
    rewriter_revision: str
    text_tokenizer_revision: str
    schema_version: str = REWRITER_SCHEMA_VERSION
    _source_condition_sha256: str | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("condition record is missing sample_id")
        if not isinstance(self.description, str):
            raise TypeError(f"{self.sample_id} description must be the string")
        if not isinstance(self.lyrics, str):
            raise TypeError(f"{self.sample_id} lyrics must be the string")
        self._require_pinned_revision("rewriter_revision", self.rewriter_revision)
        self._require_pinned_revision(
            "text_tokenizer_revision", self.text_tokenizer_revision
        )
        if self.schema_version != REWRITER_SCHEMA_VERSION:
            raise ValueError(
                f"condition schemamust be{REWRITER_SCHEMA_VERSION}"
            )
        if self._source_condition_sha256 is not None:
            _require_sha256("source_condition_sha256", self._source_condition_sha256)

    @staticmethod
    def _require_pinned_revision(name: str, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name}must be the stringrevision")
        revision = value.strip()
        if revision.lower().startswith("pin-") or revision.lower() in {
            "",
            "main",
            "master",
            "latest",
            "head",
            "unknown",
            "unresolved",
            "none",
            "null",
            "required",
            "required_pinned_revision",
        }:
            raise ValueError(f"{name}must be fixedrevision,cannot be used{value!r}")
        return revision

    @property
    def source_condition_sha256(self) -> str:
        return self._source_condition_sha256 or canonical_sha256(
            {
                "schema_version": self.schema_version,
                "description": self.description,
                "lyrics": self.lyrics,
                "rewriter_revision": self.rewriter_revision,
                "text_tokenizer_revision": self.text_tokenizer_revision,
            }
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RenderTextCondition":
        if not isinstance(value, Mapping):
            raise TypeError("condition JSONL record must be mapping")
        outer_schema = value.get("schema_version")
        if outer_schema not in {RENDER_SAMPLE_SCHEMA, RENDER_CONDITION_SCHEMA}:
            raise ValueError(
                "conditionrecordschema_versionmust be"
                f"{RENDER_SAMPLE_SCHEMA}or{RENDER_CONDITION_SCHEMA}"
            )
        sample_id = value.get("sample_id", value.get("prompt_id", value.get("id")))
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("condition record missing string sample_id")
        condition = value.get("condition")
        if not isinstance(condition, Mapping):
            raise TypeError("Versioningconditionrecord must containcondition mapping")
        allowed_condition_fields = {
            "schema_version",
            "description",
            "lyrics",
            "rewriter_revision",
            "text_tokenizer_revision",
        }
        unknown_condition_fields = set(condition) - allowed_condition_fields
        if unknown_condition_fields:
            raise ValueError(
                "conditioncontains unknown fields:"
                f"{sorted(unknown_condition_fields)}"
            )
        if condition.get("schema_version") != REWRITER_SCHEMA_VERSION:
            raise ValueError(
                f"condition.schema_versionmust be{REWRITER_SCHEMA_VERSION}"
            )
        rewriter_revision = cls._require_pinned_revision(
            "condition.rewriter_revision",
            condition.get("rewriter_revision"),
        )
        text_tokenizer_revision = cls._require_pinned_revision(
            "condition.text_tokenizer_revision",
            condition.get("text_tokenizer_revision"),
        )
        return cls(
            sample_id=sample_id,
            description=condition.get("description"),
            lyrics=condition.get("lyrics"),
            rewriter_revision=rewriter_revision,
            text_tokenizer_revision=text_tokenizer_revision,
            schema_version=str(condition["schema_version"]),
            _source_condition_sha256=canonical_sha256(dict(condition)),
        )


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


class IndexedConditionManifest(Sequence[RenderTextCondition]):

    def __init__(
        self,
        path: str | Path,
        *,
        check_duplicate_ids: bool = True,
    ) -> None:
        if not isinstance(check_duplicate_ids, bool):
            raise TypeError("check_duplicate_idsmust bebool")
        self.manifest = Path(path).resolve()
        self._offsets = array("Q")
        self._sizes = array("Q")
        self._lines = array("Q")
        self._fd: int | None = None
        self._fd_pid: int | None = None
        source_digest = hashlib.sha256()
        condition_digest = hashlib.sha256()
        condition_digest.update(b"[")
        first_condition = True
        seen: set[str] | None = set() if check_duplicate_ids else None
        try:
            with self.manifest.open("rb") as handle:
                initial_identity = _stat_identity(os.fstat(handle.fileno()))
                line_number = 0
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    line_number += 1
                    source_digest.update(line)
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            f"{self.manifest}:{line_number} JSON cannot be parsed"
                        ) from exc
                    try:
                        condition = RenderTextCondition.from_mapping(value)
                    except Exception as exc:
                        raise type(exc)(
                            f"{self.manifest}:{line_number} condition Illegal record:{exc}"
                        ) from exc
                    if seen is not None:
                        if condition.sample_id in seen:
                            raise ValueError(
                                "condition manifest contains duplicate sample_id"
                            )
                        seen.add(condition.sample_id)
                    self._offsets.append(offset)
                    self._sizes.append(len(line))
                    self._lines.append(line_number)
                    if not first_condition:
                        condition_digest.update(b",")
                    condition_digest.update(
                        canonical_json_bytes(
                            {
                                "sample_id_sha256": canonical_sha256(
                                    condition.sample_id
                                ),
                                "source_condition_sha256": (
                                    condition.source_condition_sha256
                                ),
                            }
                        )
                    )
                    first_condition = False
                final_identity = _stat_identity(os.fstat(handle.fileno()))
        except OSError as exc:
            raise FileNotFoundError(
                f"condition manifestdoes not exist or cannot be read:{self.manifest}"
            ) from exc
        if not self._offsets:
            raise ValueError(f"condition manifest is empty:{self.manifest}")
        if final_identity != initial_identity:
            raise RuntimeError("condition manifestdrifted during indexing")
        try:
            path_identity = _stat_identity(self.manifest.stat())
        except OSError as exc:
            raise FileNotFoundError(
                f"condition manifestnot visible after indexing:{self.manifest}"
            ) from exc
        if path_identity != initial_identity:
            raise RuntimeError("condition manifestwas replaced by")
        condition_digest.update(b"]")
        self._identity = initial_identity
        self.source_manifest_sha256 = source_digest.hexdigest()
        self.condition_set_sha256 = condition_digest.hexdigest()

    def __len__(self) -> int:
        return len(self._offsets)

    def _open_fd(self) -> int:
        current_pid = os.getpid()
        if self._fd is not None and self._fd_pid != current_pid:
            os.close(self._fd)
            self._fd = None
        if self._fd is None:
            descriptor = os.open(self.manifest, os.O_RDONLY)
            descriptor_identity = _stat_identity(os.fstat(descriptor))
            try:
                path_identity = _stat_identity(self.manifest.stat())
            except OSError:
                path_identity = ()
            if (
                descriptor_identity != self._identity
                or path_identity != self._identity
            ):
                os.close(descriptor)
                raise RuntimeError("condition manifestIdentity drifts after initialization")
            self._fd = descriptor
            self._fd_pid = current_pid
        return self._fd

    def __getitem__(
        self,
        index: int | slice,
    ) -> RenderTextCondition | list[RenderTextCondition]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        descriptor = self._open_fd()
        try:
            path_identity = _stat_identity(self.manifest.stat())
        except OSError:
            path_identity = ()
        if (
            _stat_identity(os.fstat(descriptor)) != self._identity
            or path_identity != self._identity
        ):
            raise RuntimeError("condition manifestcontent drifts after initialization")
        payload = os.pread(
            descriptor,
            int(self._sizes[index]),
            int(self._offsets[index]),
        )
        if len(payload) != int(self._sizes[index]):
            raise RuntimeError("condition manifestIndex short read")
        try:
            value = json.loads(payload.decode("utf-8"))
            return RenderTextCondition.from_mapping(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{self.manifest}:{int(self._lines[index])} JSON cannot be parsed"
            ) from exc

    def __iter__(self) -> Iterator[RenderTextCondition]:
        for index in range(len(self)):
            yield self[index]

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_fd"] = None
        state["_fd_pid"] = None
        return state

    def __del__(self) -> None:
        descriptor = getattr(self, "_fd", None)
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
            self._fd = None


def source_condition_sha256(
    value: RenderTextCondition | Mapping[str, Any],
) -> str:

    if isinstance(value, RenderTextCondition):
        return value.source_condition_sha256
    return RenderTextCondition.from_mapping(value).source_condition_sha256


def load_condition_jsonl(path: str | Path) -> list[RenderTextCondition]:
    return list(IndexedConditionManifest(path))


def rank_shard_indices(count: int, *, rank: int, world_size: int) -> list[int]:
    if count < 0:
        raise ValueError("count must not be negative")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank/world_size Illegal")
    return list(range(rank, count, world_size))


def shard_condition_records(
    records: Sequence[RenderTextCondition],
    *,
    rank: int,
    world_size: int,
) -> list[RenderTextCondition]:
    return [
        records[index]
        for index in rank_shard_indices(len(records), rank=rank, world_size=world_size)
    ]


def _observed_commit(value: Any) -> str | None:
    direct = getattr(value, "_commit_hash", None)
    if direct:
        return str(direct)
    config = getattr(value, "config", None)
    configured = getattr(config, "_commit_hash", None)
    if configured:
        return str(configured)
    kwargs = getattr(value, "init_kwargs", None)
    if isinstance(kwargs, Mapping) and kwargs.get("_commit_hash"):
        return str(kwargs["_commit_hash"])
    return None


def _validate_loaded_revision(name: str, value: Any, expected: str) -> None:
    observed = _observed_commit(value)
    if observed is not None and observed != expected:
        raise RuntimeError(
            f"local {name} revision Drift:expected={expected} actual={observed}"
        )


def build_frozen_qwen_adapter(
    config: RenderTextCacheConfig,
    *,
    encoder: nn.Module | None = None,
    tokenizer: Any | None = None,
    device: torch.device | str | None = None,
) -> FrozenQwenEmbeddingAdapter:

    config.validate()
    target_device = torch.device(device or config.device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configuration request CUDA,but the current interpreter is not available CUDA")
    if tokenizer is None or encoder is None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Official text caching requires transformers;Please inject fake encoder/tokenizer"
            ) from exc
        runtime_local_path = os.environ.get(
            "OQM_TEXT_ENCODER_LOCAL_PATH", ""
        ).strip()
        load_source = runtime_local_path or config.local_path or config.model_id
        if config.local_path and not Path(load_source).is_dir():
            raise FileNotFoundError(f"QwenThe local weight directory does not exist:{load_source}")
        if config.local_path:
            validate_local_text_encoder_asset(
                config,
                runtime_local_path=runtime_local_path or None,
            )
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                load_source,
                revision=(config.tokenizer_revision if not config.local_path else None),
                local_files_only=True,
                trust_remote_code=False,
                use_fast=config.use_fast_tokenizer,
            )
        if encoder is None:
            encoder = AutoModel.from_pretrained(
                load_source,
                revision=config.model_revision if not config.local_path else None,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            )
    if not isinstance(encoder, nn.Module):
        raise TypeError("text encoder must be torch.nn.Module")
    if tokenizer is None or not callable(tokenizer):
        raise TypeError("text tokenizer must be callable")
    _validate_loaded_revision("Qwen encoder", encoder, config.model_revision)
    _validate_loaded_revision("Qwen tokenizer", tokenizer, config.tokenizer_revision)
    try:
        tokenizer.padding_side = config.padding_side
        tokenizer.truncation_side = config.truncation_side
    except (AttributeError, TypeError) as exc:
        raise TypeError("tokenizer must support fixed padding/truncation side") from exc
    inferred = getattr(encoder, "hidden_size", None)
    if inferred is None:
        inferred = getattr(getattr(encoder, "config", None), "hidden_size", None)
    if inferred is not None and int(inferred) != config.hidden_size:
        raise RuntimeError(
            "text encoder hidden_size and cache config inconsistent:"
            f"{int(inferred)}!={config.hidden_size}"
        )
    encoder.to(device=target_device)
    return FrozenQwenEmbeddingAdapter(
        model_id=config.model_id,
        revision=config.model_revision,
        tokenizer_revision=config.tokenizer_revision,
        cache_revision=config.cache_revision,
        encoder=encoder,
        tokenizer=tokenizer,
        hidden_size=config.hidden_size,
        local_files_only=True,
        use_fast_tokenizer=config.use_fast_tokenizer,
        padding_side=config.padding_side,
        truncation_side=config.truncation_side,
        truncation_policy=config.truncation_policy,
        add_special_tokens=config.add_special_tokens,
        trust_remote_code=config.trust_remote_code,
        hidden_state_selection=config.hidden_state_selection,
        position_id_policy=config.position_id_policy,
        encoder_use_cache=config.encoder_use_cache,
        empty_text_policy=config.empty_text_policy,
        frozen_eval_mode=config.frozen_eval_mode,
    )


def validate_local_text_encoder_asset(
    config: RenderTextCacheConfig,
    *,
    runtime_local_path: str | Path | None = None,
) -> None:

    if not config.asset_lock:
        raise ValueError("Official localQwenloading is missingasset_lock")
    lock_path = Path(config.asset_lock)
    if not lock_path.is_absolute():
        lock_path = Path(__file__).resolve().parents[3] / lock_path
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    component = (payload.get("components") or {}).get("text_encoder")
    if not isinstance(component, Mapping):
        raise ValueError("Render asset lockis missingtext_encoder")
    model_revision = component.get("model_revision", component.get("revision"))
    tokenizer_revision = component.get(
        "tokenizer_revision", component.get("revision")
    )
    expected = {
        "repository": f"https://huggingface.co/{config.model_id}",
        "local_path": config.local_path,
    }
    mismatches = {
        key: (value, component.get(key))
        for key, value in expected.items()
        if component.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Render text encoder lockinconsistent:{mismatches}")
    if config.model_revision != str(model_revision or ""):
        raise RuntimeError(
            "Render text encoder model revision and local assets lock inconsistent:"
            f"expected={config.model_revision} actual={model_revision}"
        )
    if config.tokenizer_revision != str(tokenizer_revision or ""):
        raise RuntimeError(
            "Render text tokenizer revision and local assets lock inconsistent:"
            f"expected={config.tokenizer_revision} "
            f"actual={tokenizer_revision}"
        )
    files = component.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("Render text encoder lockMissing fileSHA")
    root = (
        Path(runtime_local_path).resolve()
        if runtime_local_path is not None
        else Path(config.local_path)
    )
    for relative, expected_sha in files.items():
        path = root / str(relative)
        if not path.is_file() or file_sha256(path) != _require_sha256(
            f"text_encoder.files.{relative}", expected_sha
        ):
            raise RuntimeError(f"Qwenlocal fileSHAinconsistent:{relative}")


@dataclass(frozen=True)
class _EncodedText:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    hidden: torch.Tensor
    original_token_count: int
    truncated: bool


def _untruncated_token_count(
    tokenizer: Any,
    text: str,
    *,
    config: RenderTextCacheConfig,
    max_length: int,
) -> int:
    kwargs: dict[str, Any] = {
        "padding": False,
        "truncation": False,
        "return_tensors": "pt",
    }
    try:
        signature = inspect.signature(tokenizer.__call__)
    except (TypeError, ValueError):
        signature = None
    accepts_kwargs = signature is None or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs or (
        signature is not None and "add_special_tokens" in signature.parameters
    ):
        kwargs["add_special_tokens"] = config.add_special_tokens
    try:
        encoded = tokenizer([text], **kwargs)
    except TypeError:

        kwargs["max_length"] = max(max_length * 16, len(text) * 4 + 16)
        encoded = tokenizer([text], **kwargs)
    if not isinstance(encoded, Mapping):
        raise TypeError("tokenizerlength detection output must bemapping")
    ids = torch.as_tensor(encoded.get("input_ids"))
    mask_value = encoded.get("attention_mask")
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("tokenizerlength detection must return[1,L] input_ids")
    if mask_value is None:
        return int(ids.shape[1])
    mask = torch.as_tensor(mask_value, dtype=torch.bool)
    if mask.shape != ids.shape:
        raise ValueError("tokenizerlength detectionattention_maskShape error")
    return int(mask.sum().item())


def _encode_one_text(
    adapter: FrozenQwenEmbeddingAdapter,
    text: str,
    *,
    max_length: int,
    config: RenderTextCacheConfig,
    device: torch.device,
) -> _EncodedText:
    tokenizer = adapter._lazy_load_tokenizer()
    is_empty = not text.strip()
    original_token_count = (
        0
        if is_empty
        else _untruncated_token_count(
            tokenizer,
            text,
            config=config,
            max_length=max_length,
        )
    )
    if original_token_count > max_length and config.truncation_policy == "reject":
        raise ValueError(
            f"texttokennumber{original_token_count}exceeds the upper limit{max_length},"
            "OfficialcacheDisable silent truncation"
        )
    tokenization_kwargs: dict[str, Any] = {
        "padding": config.padding,
        "truncation": config.truncation,
        "max_length": max_length,
        "return_tensors": "pt",
    }
    try:
        signature = inspect.signature(tokenizer.__call__)
    except (TypeError, ValueError):
        signature = None
    if signature is None or (
        "add_special_tokens" in signature.parameters
        or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    ):
        tokenization_kwargs["add_special_tokens"] = config.add_special_tokens
    encoded = tokenizer([text], **tokenization_kwargs)
    if not isinstance(encoded, Mapping):
        raise TypeError("tokenizer output must be mapping")
    input_ids = torch.as_tensor(encoded.get("input_ids"), dtype=torch.long)
    attention_mask = torch.as_tensor(encoded.get("attention_mask"), dtype=torch.bool)
    if (
        input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or attention_mask.shape != input_ids.shape
    ):
        raise ValueError("tokenizer must return the same shape [1,L] input_ids/attention_mask")
    if input_ids.shape[1] > max_length:
        raise RuntimeError("tokenizer declared max_length truncate")
    if input_ids.numel() and int(input_ids.min()) < 0:
        raise ValueError("tokenizer input_ids must not contain negative numbers")
    if is_empty:
        attention_mask.zero_()
    input_ids = input_ids.to(device=device)
    attention_mask = attention_mask.to(device=device)
    hidden = adapter(input_ids, attention_mask)
    if hidden.shape != (
        1,
        input_ids.shape[1],
        config.hidden_size,
    ):
        raise RuntimeError(
            "Qwen must return by token hidden feature,Prohibited pooling:"
            f"received {tuple(hidden.shape)}"
        )
    if not hidden.is_floating_point() or not torch.isfinite(hidden).all():
        raise RuntimeError("Qwen token-level hidden feature dtype/Illegal value")
    storage_dtype = _STORAGE_DTYPES[config.storage_dtype]
    return _EncodedText(
        input_ids=input_ids[0].detach().cpu().contiguous(),
        attention_mask=attention_mask[0].detach().cpu().contiguous(),
        hidden=hidden[0].detach().to(device="cpu", dtype=storage_dtype).contiguous(),
        original_token_count=original_token_count,
        truncated=original_token_count > max_length,
    )


def _safetensors_available() -> bool:
    return importlib.util.find_spec("safetensors") is not None


def _resolve_output_format(config: RenderTextCacheConfig) -> str:
    if config.output_format == "auto":
        return "safetensors" if _safetensors_available() else "npy"
    if config.output_format == "safetensors" and not _safetensors_available():
        raise RuntimeError(
            "Configuration requirements safetensors,but the environment is not installed;can be explicitly changed to none pickle of npy"
        )
    if config.output_format == "npy" and config.storage_dtype == "bfloat16":
        raise RuntimeError("NumPy does not support bfloat16 cache,Please use safetensors")
    return config.output_format


def _temporary_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(name)


def _sync_and_replace(temporary: Path, target: Path) -> None:
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    _fsync_directory(target.parent)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_rank_manifest(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:

    target = Path(path)
    temporary = _temporary_path(target)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                json.dump(
                    record,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _field_metadata(
    name: str,
    encoded: _EncodedText,
    *,
    max_length: int,
    config: RenderTextCacheConfig,
) -> dict[str, Any]:
    ids = encoded.input_ids.tolist()
    mask = encoded.attention_mask.tolist()
    metadata: dict[str, Any] = {
        "input_ids": ids,
        "attention_mask": mask,
        "input_ids_sha256": canonical_sha256(ids),
        "attention_mask_sha256": canonical_sha256(mask),
        "token_count": int(encoded.attention_mask.sum().item()),
        "original_token_count": int(encoded.original_token_count),
        "truncated": bool(encoded.truncated),
        "max_length": max_length,
        "shape": list(encoded.hidden.shape),
        "dtype": _dtype_name(encoded.hidden.dtype),
        "pooled": False,
        "feature_stage": "qwen_token_hidden",
        "trainable_lyrics_encoder_layers_applied": 0,
    }
    if name == "lyrics":
        metadata["next_stage"] = "six_layer_trainable_rope_encoder"
    metadata["tokenization_sha256"] = canonical_sha256(
        {
            "input_ids": ids,
            "attention_mask": mask,
            "max_length": max_length,
            "padding": config.padding,
            "truncation": config.truncation,
            "truncation_policy": config.truncation_policy,
            "add_special_tokens": config.add_special_tokens,
            "padding_side": config.padding_side,
            "truncation_side": config.truncation_side,
        }
    )
    return metadata


def _artifact_identity(
    condition: RenderTextCondition, config: RenderTextCacheConfig
) -> str:
    return canonical_sha256(
        {
            "sample_id": condition.sample_id,
            "source_condition_sha256": condition.source_condition_sha256,
            "cache_config_sha256": config.sha256,
            "provenance": config.provenance.to_dict(),
        }
    )


def cache_condition_record(
    condition: RenderTextCondition | Mapping[str, Any],
    *,
    adapter: FrozenQwenEmbeddingAdapter,
    config: RenderTextCacheConfig,
    output_dir: str | Path,
    cache_config_file_sha256: str | None = None,
    source_manifest_sha256: str | None = None,
    source_index: int | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:

    config.validate()
    item = (
        condition
        if isinstance(condition, RenderTextCondition)
        else RenderTextCondition.from_mapping(condition)
    )
    if item.text_tokenizer_revision != config.tokenizer_revision:
        raise RuntimeError(
            f"{item.sample_id} condition text tokenizer revisionandcacheConfiguration is inconsistent"
        )
    validate_cache_provenance(adapter.provenance, config.provenance)
    config_file_sha = _require_sha256(
        "cache_config_file_sha256",
        cache_config_file_sha256 or config.sha256,
    )
    manifest_sha = (
        _require_sha256("source_manifest_sha256", source_manifest_sha256)
        if source_manifest_sha256 is not None
        else None
    )
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank/world_size Illegal")
    if source_index is not None and source_index % world_size != rank:
        raise RuntimeError("source_index does not belong to the current rank,Reject interleaved write caching")
    encoder = adapter._lazy_load_encoder()
    device = next(
        iter(encoder.parameters()),
        next(iter(encoder.buffers()), torch.empty(0)),
    ).device
    description = _encode_one_text(
        adapter,
        item.description,
        max_length=config.description_max_tokens,
        config=config,
        device=device,
    )
    lyrics = _encode_one_text(
        adapter,
        item.lyrics,
        max_length=config.lyrics_max_tokens,
        config=config,
        device=device,
    )
    output_format = _resolve_output_format(config)
    identity = _artifact_identity(item, config)
    artifact_dir = Path(output_dir).resolve() / "artifacts" / identity[:2]
    suffix = ".safetensors" if output_format == "safetensors" else ".npy"
    artifact_path = artifact_dir / f"{identity}{suffix}"
    tensors = {
        "description_embeddings": description.hidden,
        "description_input_ids": description.input_ids,
        "description_mask": description.attention_mask,
        "lyrics_embeddings": lyrics.hidden,
        "lyrics_input_ids": lyrics.input_ids,
        "lyrics_mask": lyrics.attention_mask,
    }
    header = {
        "format_version": CACHE_FORMAT_VERSION,
        "sample_id_sha256": hashlib.sha256(item.sample_id.encode("utf-8")).hexdigest(),
        "source_condition_sha256": item.source_condition_sha256,
        "rewriter_revision": item.rewriter_revision,
        "text_tokenizer_revision": item.text_tokenizer_revision,
        "cache_config_sha256": config.sha256,
        "provenance_fingerprint": config.provenance.fingerprint,
        "feature_stage": "qwen_token_hidden_before_trainable_lyrics_encoder",
    }
    if output_format == "safetensors":
        from safetensors.torch import save_file

        temporary = _temporary_path(artifact_path)
        try:
            save_file(
                {name: tensor.contiguous() for name, tensor in tensors.items()},
                str(temporary),
                metadata=header,
            )
            _sync_and_replace(temporary, artifact_path)
        finally:
            temporary.unlink(missing_ok=True)
        artifact_shape: list[int] | None = None
    else:
        combined = torch.cat((description.hidden, lyrics.hidden), dim=0).numpy()
        temporary = _temporary_path(artifact_path)
        try:
            with temporary.open("wb") as handle:
                np.save(handle, combined, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, artifact_path)
        finally:
            temporary.unlink(missing_ok=True)
        artifact_shape = list(combined.shape)
    artifact_sha = file_sha256(artifact_path)
    provenance = config.provenance.to_dict()
    record: dict[str, Any] = {
        "schema_version": CACHE_FORMAT_VERSION,
        "sample_id": item.sample_id,
        "sample_id_sha256": header["sample_id_sha256"],
        "source_condition_sha256": item.source_condition_sha256,
        "rewriter_revision": item.rewriter_revision,
        "text_tokenizer_revision": item.text_tokenizer_revision,
        "source_manifest_sha256": manifest_sha,
        "source_index": source_index,
        "rank": rank,
        "world_size": world_size,
        **provenance,
        "provenance": provenance,
        "provenance_fingerprint": config.provenance.fingerprint,
        "cache_config": config.to_dict(),
        "cache_config_sha256": config.sha256,
        "cache_config_file_sha256": config_file_sha,
        "description": _field_metadata(
            "description",
            description,
            max_length=config.description_max_tokens,
            config=config,
        ),
        "lyrics": _field_metadata(
            "lyrics",
            lyrics,
            max_length=config.lyrics_max_tokens,
            config=config,
        ),
        "artifact": {
            "uri": str(artifact_path),
            "format": output_format,
            "sha256": artifact_sha,
            "shape": artifact_shape,
            "dtype": config.storage_dtype,
        },
        "file_sha256": artifact_sha,
    }
    if output_format == "npy":
        sidecar_path = artifact_path.with_suffix(".json")
        _atomic_write_json(sidecar_path, record)
        record["metadata_uri"] = str(sidecar_path)
        record["metadata_sha256"] = file_sha256(sidecar_path)
    return record


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSONcannot read:{path}") from exc
    if not isinstance(value, dict):
        raise TypeError("cache JSON sidecar must be mapping")
    return value


def _validate_field_metadata(
    name: str,
    value: Any,
    *,
    expected_max_length: int,
    config: RenderTextCacheConfig,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if not isinstance(value, Mapping):
        raise TypeError(f"cache {name} metadata must be mapping")
    if value.get("max_length") != expected_max_length:
        raise RuntimeError(f"cache {name} max length Contract Drift")
    if value.get("pooled") is not False:
        raise RuntimeError(f"cache {name} Prohibited use pooled feature")
    if value.get("feature_stage") != "qwen_token_hidden":
        raise RuntimeError(f"cache {name} feature stage not Qwen token hidden")
    if value.get("trainable_lyrics_encoder_layers_applied") != 0:
        raise RuntimeError("cache is prohibited from containing trainable lyrics encoder output")
    if name == "lyrics" and (
        value.get("next_stage") != "six_layer_trainable_rope_encoder"
    ):
        raise RuntimeError("lyrics cache is not bound and can be trained on the subsequent six layers. encoder")
    ids = value.get("input_ids")
    mask = value.get("attention_mask")
    if not isinstance(ids, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in ids
    ):
        raise TypeError(f"cache {name} input_ids Illegal")
    if not isinstance(mask, list) or not all(isinstance(item, bool) for item in mask):
        raise TypeError(f"cache {name} attention_mask Illegal")
    if len(ids) != len(mask) or len(ids) > expected_max_length:
        raise RuntimeError(f"cache {name} ids/mask Illegal length")
    expected_shape = [len(ids), config.hidden_size]
    if value.get("shape") != expected_shape:
        raise RuntimeError(f"cache {name} shape Contract inconsistent")
    if value.get("dtype") != config.storage_dtype:
        raise RuntimeError(f"cache {name} dtype Contract inconsistent")
    if value.get("token_count") != sum(mask):
        raise RuntimeError(f"cache {name} token_count and mask inconsistent")
    original_count = value.get("original_token_count")
    truncated = value.get("truncated")
    if (
        not isinstance(original_count, int)
        or isinstance(original_count, bool)
        or original_count < value["token_count"]
        or not isinstance(truncated, bool)
        or truncated != (original_count > expected_max_length)
    ):
        raise RuntimeError(f"cache {name}Truncation evidence is inconsistent")
    if truncated and config.truncation_policy == "reject":
        raise RuntimeError(f"cache {name}violatesrejecttruncation strategy")
    if value.get("input_ids_sha256") != canonical_sha256(ids):
        raise RuntimeError(f"cache {name} input_ids SHA has been tampered with")
    if value.get("attention_mask_sha256") != canonical_sha256(mask):
        raise RuntimeError(f"cache {name} attention mask SHA has been tampered with")
    tokenization_sha = canonical_sha256(
        {
            "input_ids": ids,
            "attention_mask": mask,
            "max_length": expected_max_length,
            "padding": config.padding,
            "truncation": config.truncation,
            "truncation_policy": config.truncation_policy,
            "add_special_tokens": config.add_special_tokens,
            "padding_side": config.padding_side,
            "truncation_side": config.truncation_side,
        }
    )
    if value.get("tokenization_sha256") != tokenization_sha:
        raise RuntimeError(f"cache {name} tokenization Evidence has been tampered with")
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.bool),
        expected_shape,
    )


@dataclass(frozen=True)
class LoadedTextCacheEntry:
    sample_id: str
    description_embeddings: torch.Tensor
    description_input_ids: torch.Tensor
    description_mask: torch.Tensor
    lyrics_embeddings: torch.Tensor
    lyrics_input_ids: torch.Tensor
    lyrics_mask: torch.Tensor
    cache_provenance: TextEncoderProvenance
    source_condition_sha256: str

    def conditioner_kwargs(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        target = torch.device(device) if device is not None else None

        def move(value: torch.Tensor, *, feature: bool) -> torch.Tensor:
            return value.to(
                device=target,
                dtype=dtype if feature and dtype is not None else value.dtype,
            )

        return {
            "description_embeddings": move(
                self.description_embeddings.unsqueeze(0), feature=True
            ),
            "description_mask": move(self.description_mask.unsqueeze(0), feature=False),
            "lyrics_embeddings": move(
                self.lyrics_embeddings.unsqueeze(0), feature=True
            ),
            "lyrics_mask": move(self.lyrics_mask.unsqueeze(0), feature=False),
            "cache_provenance": self.cache_provenance,
        }


def read_renderer_data_content_addressed_text_pair(
    reference: Mapping[str, Any],
    *,
    condition: RenderTextCondition | Mapping[str, Any],
    expected_provenance: TextEncoderProvenance | Mapping[str, Any],
    expected_cache_config_sha256: str,
    expected_cache_config_file_sha256: str,
    base_dir: str | Path | None = None,
    tags_base_dir: str | Path | None = None,
    lyrics_base_dir: str | Path | None = None,
) -> LoadedTextCacheEntry:

    if not isinstance(reference, Mapping):
        raise TypeError("RendererData text reference must be a mapping")
    if reference.get("schema_version") != RENDERER_DATA_TEXT_REFERENCE_SCHEMA:
        raise RuntimeError("Unsupported RendererData text reference schema")
    parsed = (
        condition
        if isinstance(condition, RenderTextCondition)
        else RenderTextCondition.from_mapping(condition)
    )
    tags_sha = _require_sha256(
        "tags_content_sha256", reference.get("tags_content_sha256")
    )
    lyrics_sha = _require_sha256(
        "lyrics_content_sha256", reference.get("lyrics_content_sha256")
    )
    if hashlib.sha256(parsed.description.encode("utf-8")).hexdigest() != tags_sha:
        raise RuntimeError("RendererData tags content does not match its SHA-256")
    if hashlib.sha256(parsed.lyrics.encode("utf-8")).hexdigest() != lyrics_sha:
        raise RuntimeError("RendererData lyrics content does not match its SHA-256")
    if reference.get("source_condition_sha256") != parsed.source_condition_sha256:
        raise RuntimeError("RendererData text reference does not match the condition")
    tags_record = reference.get("tags_record")
    lyrics_record = reference.get("lyrics_record")
    if not isinstance(tags_record, Mapping) or not isinstance(lyrics_record, Mapping):
        raise TypeError("RendererData text reference must embed tags and lyrics records")
    if tags_record.get("sample_id") != f"renderer_data-tags:{tags_sha}":
        raise RuntimeError(
            "RendererData tags cache sample_id does not match its content SHA"
        )
    if lyrics_record.get("sample_id") != f"renderer_data-lyrics:{lyrics_sha}":
        raise RuntimeError(
            "RendererData lyrics cache sample_id does not match its content SHA"
        )
    tags_entry = read_text_cache_entry(
        tags_record,
        expected_provenance=expected_provenance,
        expected_cache_config_sha256=expected_cache_config_sha256,
        expected_cache_config_file_sha256=expected_cache_config_file_sha256,
        base_dir=tags_base_dir if tags_base_dir is not None else base_dir,
    )
    lyrics_entry = read_text_cache_entry(
        lyrics_record,
        expected_provenance=expected_provenance,
        expected_cache_config_sha256=expected_cache_config_sha256,
        expected_cache_config_file_sha256=expected_cache_config_file_sha256,
        base_dir=lyrics_base_dir if lyrics_base_dir is not None else base_dir,
    )
    if bool(tags_entry.lyrics_mask.any()) or bool(
        torch.count_nonzero(tags_entry.lyrics_embeddings)
    ):
        raise RuntimeError("RendererData tags cacheoflyrics counterpartrequiredzero+false")
    if bool(lyrics_entry.description_mask.any()) or bool(
        torch.count_nonzero(lyrics_entry.description_embeddings)
    ):
        raise RuntimeError(
            "RendererData lyrics cacheofdescription counterpartrequiredzero+false"
        )
    if tags_entry.cache_provenance != lyrics_entry.cache_provenance:
        raise RuntimeError("RendererData tags/lyrics cache provenanceinconsistent")
    return LoadedTextCacheEntry(
        sample_id=parsed.sample_id,
        description_embeddings=tags_entry.description_embeddings,
        description_input_ids=tags_entry.description_input_ids,
        description_mask=tags_entry.description_mask,
        lyrics_embeddings=lyrics_entry.lyrics_embeddings,
        lyrics_input_ids=lyrics_entry.lyrics_input_ids,
        lyrics_mask=lyrics_entry.lyrics_mask,
        cache_provenance=tags_entry.cache_provenance,
        source_condition_sha256=parsed.source_condition_sha256,
    )


def _validate_artifact_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    description_ids: torch.Tensor,
    description_mask: torch.Tensor,
    lyrics_ids: torch.Tensor,
    lyrics_mask: torch.Tensor,
    config: RenderTextCacheConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    required = {
        "description_embeddings",
        "description_input_ids",
        "description_mask",
        "lyrics_embeddings",
        "lyrics_input_ids",
        "lyrics_mask",
    }
    if set(tensors) != required:
        raise RuntimeError("safetensors cache tensor The collection does not comply with the contract")
    for name in ("description_input_ids", "lyrics_input_ids"):
        if tensors[name].dtype != torch.long:
            raise RuntimeError(f"{name} artifact dtype must be int64")
    for name in ("description_mask", "lyrics_mask"):
        if tensors[name].dtype != torch.bool:
            raise RuntimeError(f"{name} artifact dtype must be bool")
    if not torch.equal(tensors["description_input_ids"], description_ids):
        raise RuntimeError("description input_ids and artifact inconsistent")
    if not torch.equal(tensors["description_mask"], description_mask):
        raise RuntimeError("description mask and artifact inconsistent")
    if not torch.equal(tensors["lyrics_input_ids"], lyrics_ids):
        raise RuntimeError("lyrics input_ids and artifact inconsistent")
    if not torch.equal(tensors["lyrics_mask"], lyrics_mask):
        raise RuntimeError("lyrics mask and artifact inconsistent")
    description = tensors["description_embeddings"]
    lyrics = tensors["lyrics_embeddings"]
    expected_dtype = _STORAGE_DTYPES[config.storage_dtype]
    for name, hidden, mask in (
        ("description", description, description_mask),
        ("lyrics", lyrics, lyrics_mask),
    ):
        if hidden.shape != (mask.shape[0], config.hidden_size):
            raise RuntimeError(f"{name} artifact shape does not comply with the contract")
        if hidden.dtype != expected_dtype:
            raise RuntimeError(f"{name} artifact dtype does not comply with the contract")
        if not torch.isfinite(hidden).all():
            raise RuntimeError(f"{name} artifact contains non-finite")
        if (~mask).any() and torch.count_nonzero(hidden[~mask]):
            raise RuntimeError(f"{name} padding hidden must be zero")
    return description.contiguous(), lyrics.contiguous()


def read_text_cache_entry(
    record: Mapping[str, Any],
    *,
    expected_provenance: TextEncoderProvenance | Mapping[str, Any],
    expected_cache_config_sha256: str,
    expected_source_condition_sha256: str | None = None,
    expected_condition: RenderTextCondition | Mapping[str, Any] | None = None,
    expected_cache_config_file_sha256: str | None = None,
    base_dir: str | Path | None = None,
) -> LoadedTextCacheEntry:

    if not isinstance(record, Mapping):
        raise TypeError("text cache record must be mapping")
    if record.get("schema_version") != CACHE_FORMAT_VERSION:
        raise RuntimeError("text cache schema_version is not compatible with")
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("text cache record is missing sample_id")
    expected_sample_hash = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    if record.get("sample_id_sha256") != expected_sample_hash:
        raise RuntimeError("text cache sample_id SHA has been tampered with")
    expected = (
        expected_provenance
        if isinstance(expected_provenance, TextEncoderProvenance)
        else TextEncoderProvenance.from_mapping(expected_provenance)
    )
    nested = record.get("provenance")
    if not isinstance(nested, Mapping):
        raise RuntimeError("text cache is missing provenance")
    top_level = TextEncoderProvenance.from_mapping(
        {
            name: record.get(name)
            for name in TextEncoderProvenance.__dataclass_fields__
        }
    )
    nested_value = TextEncoderProvenance.from_mapping(nested)
    if top_level != nested_value:
        raise RuntimeError("text cache Top-level and nested provenance inconsistent")
    validate_cache_provenance(nested_value, expected)
    if record.get("provenance_fingerprint") != expected.fingerprint:
        raise RuntimeError("text cache provenance fingerprint has been tampered with")
    cache_mapping = record.get("cache_config")
    if not isinstance(cache_mapping, Mapping):
        raise RuntimeError("text cache is missing cache_config")
    embedded_config_sha = canonical_sha256(cache_mapping)
    config = RenderTextCacheConfig.from_mapping(cache_mapping)
    declared_config_sha = _require_sha256(
        "cache_config_sha256", record.get("cache_config_sha256")
    )
    expected_config_sha = _require_sha256(
        "expected_cache_config_sha256", expected_cache_config_sha256
    )


    if embedded_config_sha != declared_config_sha:
        raise RuntimeError("text cache config content and SHA inconsistent")
    if declared_config_sha != expected_config_sha:
        raise RuntimeError("text cache config SHA Drift")
    validate_cache_provenance(config.provenance, expected)
    declared_file_config_sha = _require_sha256(
        "cache_config_file_sha256",
        record.get("cache_config_file_sha256"),
    )
    if expected_cache_config_file_sha256 is not None and (
        declared_file_config_sha
        != _require_sha256(
            "expected_cache_config_file_sha256",
            expected_cache_config_file_sha256,
        )
    ):
        raise RuntimeError("text cache YAML file SHA Drift")
    source_hash = _require_sha256(
        "source_condition_sha256",
        record.get("source_condition_sha256"),
    )
    RenderTextCondition._require_pinned_revision(
        "text cache rewriter_revision",
        str(record.get("rewriter_revision") or ""),
    )
    if str(record.get("text_tokenizer_revision") or "") != expected.tokenizer_revision:
        raise RuntimeError("text cache condition tokenizer revisioninconsistent")
    if expected_condition is not None:
        condition_value = (
            expected_condition
            if isinstance(expected_condition, RenderTextCondition)
            else RenderTextCondition.from_mapping(expected_condition)
        )
        condition_hash = condition_value.source_condition_sha256
        if (
            expected_source_condition_sha256 is not None
            and condition_hash
            != _require_sha256(
                "expected_source_condition_sha256",
                expected_source_condition_sha256,
            )
        ):
            raise ValueError("Caller condition and explicit condition SHA inconsistent")
        expected_source_condition_sha256 = condition_hash
        if record.get("rewriter_revision") != condition_value.rewriter_revision:
            raise RuntimeError("text cache rewriter revisionandconditioninconsistent")
    if expected_source_condition_sha256 is not None and source_hash != (
        _require_sha256(
            "expected_source_condition_sha256",
            expected_source_condition_sha256,
        )
    ):
        raise RuntimeError("text cache source condition SHA Drift")
    description_ids, description_mask, _ = _validate_field_metadata(
        "description",
        record.get("description"),
        expected_max_length=DESCRIPTION_MAX_TOKENS,
        config=config,
    )
    lyrics_ids, lyrics_mask, _ = _validate_field_metadata(
        "lyrics",
        record.get("lyrics"),
        expected_max_length=config.lyrics_max_tokens,
        config=config,
    )
    artifact = record.get("artifact")
    if not isinstance(artifact, Mapping):
        raise RuntimeError("text cache is missing artifact metadata")
    uri = artifact.get("uri")
    if not isinstance(uri, str) or not uri:
        raise RuntimeError("text cache artifact uri Illegal")
    path = _resolve_cache_artifact(
        uri,
        base_dir=base_dir,
        field="text cache artifact URI",
    )
    try:
        artifact_bytes = path.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"text cache artifact does not exist or cannot be read:{path}") from exc
    declared_file_sha = _require_sha256("artifact.sha256", artifact.get("sha256"))
    if record.get("file_sha256") != declared_file_sha:
        raise RuntimeError("text cache top level and artifact file SHA inconsistent")
    if hashlib.sha256(artifact_bytes).hexdigest() != declared_file_sha:
        raise RuntimeError("text cache artifact file SHA inconsistent")
    storage_format = artifact.get("format")
    if artifact.get("dtype") != config.storage_dtype:
        raise RuntimeError("text cache artifact dtype metadata Drift")
    if storage_format == "safetensors":
        if path.suffix != ".safetensors":
            raise RuntimeError("safetensors cache Illegal extension")
        if not _safetensors_available():
            raise RuntimeError("Read safetensors cache requires safetensors")
        from safetensors.torch import load as load_safetensors

        if len(artifact_bytes) < 8:
            raise RuntimeError("safetensors cache headerdamaged")
        header_length = int.from_bytes(artifact_bytes[:8], "little")
        if header_length <= 0 or header_length > len(artifact_bytes) - 8:
            raise RuntimeError("safetensors cache headerIllegal length")
        try:
            raw_header = json.loads(
                artifact_bytes[8 : 8 + header_length].decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("safetensors cache headercannot be parsed") from exc
        if not isinstance(raw_header, Mapping):
            raise RuntimeError("safetensors cache headermust bemapping")
        header = raw_header.get("__metadata__")
        if not isinstance(header, Mapping):
            raise RuntimeError("safetensors cacheis missingmetadata")
        expected_header = {
            "format_version": CACHE_FORMAT_VERSION,
            "sample_id_sha256": expected_sample_hash,
            "source_condition_sha256": source_hash,
            "rewriter_revision": str(record.get("rewriter_revision") or ""),
            "text_tokenizer_revision": str(
                record.get("text_tokenizer_revision") or ""
            ),
            "cache_config_sha256": declared_config_sha,
            "provenance_fingerprint": expected.fingerprint,
            "feature_stage": ("qwen_token_hidden_before_trainable_lyrics_encoder"),
        }
        if header != expected_header:
            raise RuntimeError("safetensors header Contract or hash inconsistent")
        try:
            tensors = load_safetensors(artifact_bytes)
        except Exception as exc:  # noqa: BLE001 - .
            raise RuntimeError("safetensors cacheUnable to deserialize") from exc
        description, lyrics = _validate_artifact_tensors(
            tensors,
            description_ids=description_ids,
            description_mask=description_mask,
            lyrics_ids=lyrics_ids,
            lyrics_mask=lyrics_mask,
            config=config,
        )
    elif storage_format == "npy":
        if path.suffix != ".npy":
            raise RuntimeError("npy cache Illegal extension")
        metadata_uri = record.get("metadata_uri")
        metadata_sha = record.get("metadata_sha256")
        if not isinstance(metadata_uri, str) or not metadata_uri:
            raise RuntimeError("npy cache is missing JSON sidecar")
        metadata_path = _resolve_cache_artifact(
            metadata_uri,
            base_dir=base_dir,
            field="npy metadata URI",
        )
        try:
            metadata_bytes = metadata_path.read_bytes()
        except OSError as exc:
            raise FileNotFoundError(
                f"npy cache JSON sidecardoes not exist or cannot be read:{metadata_path}"
            ) from exc
        if hashlib.sha256(metadata_bytes).hexdigest() != _require_sha256(
            "metadata_sha256", metadata_sha
        ):
            raise RuntimeError("npy cache JSON sidecar SHA inconsistent")
        try:
            sidecar = json.loads(metadata_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("npy cache JSON sidecarcannot be parsed") from exc
        if not isinstance(sidecar, Mapping):
            raise RuntimeError("npy cache JSON sidecarmust bemapping")
        rank_record_without_sidecar = {
            key: value
            for key, value in record.items()
            if key not in {"metadata_uri", "metadata_sha256"}
        }
        if sidecar != rank_record_without_sidecar:
            raise RuntimeError("npy cache rank JSONL and sample-by-sample JSON inconsistent")
        try:
            array = np.load(io.BytesIO(artifact_bytes), allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RuntimeError("npy cacheUnable to deserialize") from exc
        expected_shape = [
            description_mask.shape[0] + lyrics_mask.shape[0],
            config.hidden_size,
        ]
        if artifact.get("shape") != expected_shape or list(array.shape) != (
            expected_shape
        ):
            raise RuntimeError("npy cache shape does not comply with the contract")
        if str(array.dtype) != config.storage_dtype:
            raise RuntimeError("npy cache dtype does not comply with the contract")
        values = torch.from_numpy(np.array(array, copy=True))
        split = description_mask.shape[0]
        description = values[:split]
        lyrics = values[split:]
        description, lyrics = _validate_artifact_tensors(
            {
                "description_embeddings": description,
                "description_input_ids": description_ids,
                "description_mask": description_mask,
                "lyrics_embeddings": lyrics,
                "lyrics_input_ids": lyrics_ids,
                "lyrics_mask": lyrics_mask,
            },
            description_ids=description_ids,
            description_mask=description_mask,
            lyrics_ids=lyrics_ids,
            lyrics_mask=lyrics_mask,
            config=config,
        )
    else:
        raise RuntimeError(f"Unknown text cache artifact format:{storage_format!r}")
    return LoadedTextCacheEntry(
        sample_id=sample_id,
        description_embeddings=description,
        description_input_ids=description_ids,
        description_mask=description_mask,
        lyrics_embeddings=lyrics,
        lyrics_input_ids=lyrics_ids,
        lyrics_mask=lyrics_mask,
        cache_provenance=expected,
        source_condition_sha256=source_hash,
    )


def load_text_cache_record(
    record: Mapping[str, Any],
    *,
    expected_provenance: TextEncoderProvenance | Mapping[str, Any],
    expected_cache_config_sha256: str,
    expected_source_condition_sha256: str | None = None,
    expected_condition: RenderTextCondition | Mapping[str, Any] | None = None,
    expected_cache_config_file_sha256: str | None = None,
    base_dir: str | Path | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:

    entry = read_text_cache_entry(
        record,
        expected_provenance=expected_provenance,
        expected_cache_config_sha256=expected_cache_config_sha256,
        expected_source_condition_sha256=expected_source_condition_sha256,
        expected_condition=expected_condition,
        expected_cache_config_file_sha256=(expected_cache_config_file_sha256),
        base_dir=base_dir,
    )
    return entry.conditioner_kwargs(device=device, dtype=dtype)


def _read_rank_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} cache JSON cannot be parsed") from exc
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} cache record must be mapping")
            records.append(value)
    return records


def _decode_cache_jsonl_line(
    path: Path,
    line_number: int,
    payload: bytes,
) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}:{line_number} cache JSON cannot be parsed") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{path}:{line_number} cache record must be mapping")
    return value


def _resolve_within(root: Path, raw: Any, *, field: str) -> Path:
    value = str(raw or "")
    if not value or "://" in value:
        raise RuntimeError(f"{field}must be a local path")
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{field}escapetext cache releaseTable of Contents") from exc
    return resolved


def _artifact_root(base_dir: str | Path) -> Path:

    root = Path(base_dir).resolve()
    return root.parent if root.name == "ranks" else root


def _resolve_cache_artifact(
    raw: Any,
    *,
    base_dir: str | Path | None,
    field: str,
) -> Path:
    value = str(raw or "")
    if not value or "://" in value:
        raise RuntimeError(f"{field}must be a local path")
    candidate = Path(value)
    if base_dir is None:
        if not candidate.is_absolute():
            raise RuntimeError(f"relative to{field}is missingbase_dir")
        return candidate.resolve()
    return _resolve_within(_artifact_root(base_dir), candidate, field=field)


class RenderTextCacheLoader:

    def __init__(
        self,
        manifests: (str | Path | Sequence[str | Path] | Sequence[Mapping[str, Any]]),
        *,
        expected_provenance: TextEncoderProvenance | Mapping[str, Any],
        expected_cache_config_sha256: str,
        expected_cache_config_file_sha256: str | None = None,
    ) -> None:
        self.expected_provenance = (
            expected_provenance
            if isinstance(expected_provenance, TextEncoderProvenance)
            else TextEncoderProvenance.from_mapping(expected_provenance)
        )
        self.expected_cache_config_sha256 = _require_sha256(
            "expected_cache_config_sha256",
            expected_cache_config_sha256,
        )
        self.expected_cache_config_file_sha256 = (
            _require_sha256(
                "expected_cache_config_file_sha256",
                expected_cache_config_file_sha256,
            )
            if expected_cache_config_file_sha256 is not None
            else None
        )
        self._indexed_manifest: Path | None = None
        self._indexed_identity: tuple[int, int, int, int, int] | None = None
        self._indexed_fd: int | None = None
        self._indexed_fd_pid: int | None = None
        self._index_database_path: Path | None = None
        self._index_database_owner_pid: int | None = None
        self._index_connection: sqlite3.Connection | None = None
        self._index_connection_pid: int | None = None
        self._indexed_count = 0
        self._indexed_base_dir: Path | None = None
        rows_with_bases: list[tuple[dict[str, Any], Path | None]]
        if isinstance(manifests, (str, Path)):
            path = Path(manifests).resolve()
            if path.is_dir():
                ready_path = path / "READY"
                if not ready_path.is_file():
                    raise FileNotFoundError(f"text cachedirectory is missingREADY:{ready_path}")
                ready = _load_json_object(ready_path)
                if (
                    ready.get("schema_version") != TEXT_CACHE_READY_SCHEMA
                    or ready.get("status") != TEXT_CACHE_READY_STATUS
                ):
                    raise RuntimeError("text cache READY schema/statusis not compatible with")
                manifest_path = _resolve_within(
                    path,
                    ready.get("manifest"),
                    field="text cache READY.manifest",
                )
                if not manifest_path.is_file():
                    raise FileNotFoundError(
                        f"text cache READYquotedmanifestdoes not exist:{manifest_path}"
                    )
                if ready.get("cache_config_sha256") != self.expected_cache_config_sha256:
                    raise RuntimeError("text cache READY config SHAinconsistent")
                if self.expected_cache_config_file_sha256 is not None and (
                    ready.get("cache_config_file_sha256")
                    != self.expected_cache_config_file_sha256
                ):
                    raise RuntimeError("text cache READY configfileSHAinconsistent")
                declared_rank_manifests = ready.get("rank_manifests")
                world_size = ready.get("world_size")
                if (
                    not isinstance(world_size, int)
                    or isinstance(world_size, bool)
                    or world_size <= 0
                    or not isinstance(declared_rank_manifests, Mapping)
                ):
                    raise RuntimeError("text cache READY world/rank metadataIllegal")
                expected_names = {
                    f"rank_{rank:04d}.jsonl" for rank in range(world_size)
                }
                if set(declared_rank_manifests) != expected_names:
                    raise RuntimeError("text cache READY rank manifestThe collection is incomplete")
                rank_dir = path / "ranks"
                ready_records = ready.get("records")
                if (
                    not isinstance(ready_records, int)
                    or isinstance(ready_records, bool)
                    or ready_records <= 0
                ):
                    raise RuntimeError("text cache READYThe number of records is inconsistent")
                source_manifest_sha = _require_sha256(
                    "READY.source_manifest_sha256",
                    ready.get("source_manifest_sha256"),
                )
                rank_paths = {
                    rank: _resolve_within(
                        path,
                        rank_dir / f"rank_{rank:04d}.jsonl",
                        field=f"text cache rank manifest {rank}",
                    )
                    for rank in range(world_size)
                }
                self._initialize_directory_index(
                    manifest_path=manifest_path,
                    expected_manifest_sha256=_require_sha256(
                        "READY.manifest_sha256",
                        ready.get("manifest_sha256"),
                    ),
                    rank_paths=rank_paths,
                    rank_sha256={
                        rank: _require_sha256(
                            f"READY.rank_manifests.rank_{rank:04d}.jsonl",
                            declared_rank_manifests[f"rank_{rank:04d}.jsonl"],
                        )
                        for rank in range(world_size)
                    },
                    world_size=world_size,
                    expected_records=ready_records,
                    expected_source_manifest_sha256=source_manifest_sha,
                    expected_condition_set_sha256=_require_sha256(
                        "READY.condition_set_sha256",
                        ready.get("condition_set_sha256"),
                    ),
                )
                return
            else:
                rows = _read_rank_jsonl(path)
                rows_with_bases = [(row, path.parent) for row in rows]
        else:
            values = list(manifests)
            if not values:
                raise ValueError("text cache manifests must not be empty")
            if all(isinstance(item, Mapping) for item in values):
                rows = [dict(item) for item in values]  # type: ignore[arg-type]
                rows_with_bases = [(row, None) for row in rows]
            elif all(isinstance(item, (str, Path)) for item in values):
                rows_with_bases = []
                for item in values:
                    manifest_path = Path(item).resolve()
                    rows_with_bases.extend(
                        (row, manifest_path.parent)
                        for row in _read_rank_jsonl(manifest_path)
                    )
                rows = [row for row, _ in rows_with_bases]
            else:
                raise TypeError("manifests must be all paths or all cache records")
        if not rows:
            raise ValueError("text cache manifests No sample")
        identifiers = [row.get("sample_id") for row in rows]
        if any(not isinstance(value, str) or not value for value in identifiers):
            raise ValueError("text cache manifest is illegal sample_id")
        if len(set(identifiers)) != len(identifiers):
            raise RuntimeError("text cache manifests contains duplicate sample_id/rank cross write")
        worlds = {row.get("world_size") for row in rows}
        if len(worlds) != 1:
            raise RuntimeError("text cache manifests mixed with different world_size")
        world_size = next(iter(worlds))
        if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
            raise RuntimeError("text cache world_sizeIllegal")
        ranks = {row.get("rank") for row in rows}
        if ranks != set(range(world_size)):
            raise RuntimeError(
                "text cache rankThe collection is incomplete:"
                f"expected={list(range(world_size))} actual={sorted(ranks)}"
            )
        source_manifests = {row.get("source_manifest_sha256") for row in rows}
        if len(source_manifests) != 1:
            raise RuntimeError("text cache manifestsmixed with differentsource manifest")
        _require_sha256(
            "source_manifest_sha256",
            next(iter(source_manifests)),
        )
        rewriter_revisions = {row.get("rewriter_revision") for row in rows}
        if len(rewriter_revisions) != 1:
            raise RuntimeError("text cache manifestsmixed with differentrewriter revision")
        RenderTextCondition._require_pinned_revision(
            "text cache rewriter_revision",
            str(next(iter(rewriter_revisions)) or ""),
        )
        condition_tokenizers = {row.get("text_tokenizer_revision") for row in rows}
        if condition_tokenizers != {self.expected_provenance.tokenizer_revision}:
            raise RuntimeError("text cache manifestsofcondition tokenizer revisioninconsistent")
        config_hashes = {row.get("cache_config_sha256") for row in rows}
        if config_hashes != {self.expected_cache_config_sha256}:
            raise RuntimeError("text cache manifestsofcache config SHAinconsistent")
        config_file_hashes = {row.get("cache_config_file_sha256") for row in rows}
        if len(config_file_hashes) != 1:
            raise RuntimeError("text cache manifestsmixed with differentcache configfileSHA")
        observed_config_file_sha = _require_sha256(
            "cache_config_file_sha256",
            next(iter(config_file_hashes)),
        )
        if (
            self.expected_cache_config_file_sha256 is not None
            and observed_config_file_sha != self.expected_cache_config_file_sha256
        ):
            raise RuntimeError("text cache manifestofconfigfileSHAinconsistent")
        for row in rows:
            rank = row.get("rank")
            world = row.get("world_size")
            index = row.get("source_index")
            if (
                not isinstance(rank, int)
                or not isinstance(world, int)
                or world <= 0
                or not 0 <= rank < world
            ):
                raise RuntimeError("text cache rank/world metadata Illegal")
            if index is not None and (
                not isinstance(index, int) or index < 0 or index % world != rank
            ):
                raise RuntimeError("text cache source_index/rank Fragments are not mutually exclusive")
        source_indices = [
            int(row["source_index"])
            for row in rows
            if row.get("source_index") is not None
        ]
        if len(source_indices) != len(set(source_indices)):
            raise RuntimeError("text cache manifests Duplicate statement source_index")
        if len(source_indices) != len(rows) or sorted(source_indices) != list(
            range(len(rows))
        ):
            raise RuntimeError("text cache source_indexIncomplete continuous coverage input")
        self._records = {str(row["sample_id"]): row for row in rows}
        self._record_bases = {
            str(row["sample_id"]): base for row, base in rows_with_bases
        }

    def _validate_indexed_row(
        self,
        row: Mapping[str, Any],
        *,
        source_index: int,
        rank: int,
        world_size: int,
        source_manifest_sha256: str,
        observed_config_file_sha256: str | None,
    ) -> str:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("text cache manifest is illegal sample_id")
        if (
            row.get("source_index") != source_index
            or row.get("rank") != rank
            or row.get("world_size") != world_size
            or source_index % world_size != rank
        ):
            raise RuntimeError("text cache source_index/rank Fragments are not mutually exclusive")
        if row.get("source_manifest_sha256") != source_manifest_sha256:
            raise RuntimeError("text cache READYandmanifest source SHAinconsistent")
        RenderTextCondition._require_pinned_revision(
            "text cache rewriter_revision",
            str(row.get("rewriter_revision") or ""),
        )
        if (
            row.get("text_tokenizer_revision")
            != self.expected_provenance.tokenizer_revision
        ):
            raise RuntimeError(
                "text cache manifestsofcondition tokenizer revisioninconsistent"
            )
        if row.get("cache_config_sha256") != self.expected_cache_config_sha256:
            raise RuntimeError("text cache manifestsofcache config SHAinconsistent")
        config_file_sha = _require_sha256(
            "cache_config_file_sha256",
            row.get("cache_config_file_sha256"),
        )
        if (
            observed_config_file_sha256 is not None
            and config_file_sha != observed_config_file_sha256
        ):
            raise RuntimeError("text cache manifestsmixed with differentcache configfileSHA")
        if (
            self.expected_cache_config_file_sha256 is not None
            and config_file_sha != self.expected_cache_config_file_sha256
        ):
            raise RuntimeError("text cache manifestofconfigfileSHAinconsistent")
        return config_file_sha

    def _initialize_directory_index(
        self,
        *,
        manifest_path: Path,
        expected_manifest_sha256: str,
        rank_paths: Mapping[int, Path],
        rank_sha256: Mapping[int, str],
        world_size: int,
        expected_records: int,
        expected_source_manifest_sha256: str,
        expected_condition_set_sha256: str,
    ) -> None:
        descriptor, database_name = tempfile.mkstemp(
            prefix="oqm-render-text-index-",
            suffix=".sqlite3",
        )
        os.close(descriptor)
        database_path = Path(database_name)
        os.chmod(database_path, 0o600)
        connection = sqlite3.connect(database_path)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute(
            "CREATE TABLE records ("
            "sample_id TEXT PRIMARY KEY, "
            "source_index INTEGER NOT NULL UNIQUE, "
            "byte_offset INTEGER NOT NULL, "
            "byte_size INTEGER NOT NULL"
            ") WITHOUT ROWID"
        )
        condition_digest = hashlib.sha256()
        condition_digest.update(b"[")
        first_condition = True
        rank_handles: dict[int, Any] = {}
        rank_identities: dict[int, tuple[int, int, int, int, int]] = {}
        rank_digests = {rank: hashlib.sha256() for rank in rank_paths}
        rank_lines = {rank: 0 for rank in rank_paths}
        try:
            for rank, rank_path in rank_paths.items():
                try:
                    handle = rank_path.open("rb")
                except OSError as exc:
                    raise FileNotFoundError(
                        f"text cache rank manifestdoes not exist:{rank_path}"
                    ) from exc
                rank_handles[rank] = handle
                rank_identities[rank] = _stat_identity(os.fstat(handle.fileno()))
            with manifest_path.open("rb") as global_handle:
                manifest_identity = _stat_identity(os.fstat(global_handle.fileno()))
                manifest_digest = hashlib.sha256()
                record_index = 0
                observed_config_file_sha: str | None = None
                global_line_number = 0
                while True:
                    offset = global_handle.tell()
                    line = global_handle.readline()
                    if not line:
                        break
                    global_line_number += 1
                    manifest_digest.update(line)
                    if not line.strip():
                        continue
                    row = _decode_cache_jsonl_line(
                        manifest_path,
                        global_line_number,
                        line,
                    )
                    rank = record_index % world_size
                    observed_config_file_sha = self._validate_indexed_row(
                        row,
                        source_index=record_index,
                        rank=rank,
                        world_size=world_size,
                        source_manifest_sha256=(
                            expected_source_manifest_sha256
                        ),
                        observed_config_file_sha256=observed_config_file_sha,
                    )
                    sample_id = str(row["sample_id"])
                    try:
                        connection.execute(
                            "INSERT INTO records"
                            "(sample_id, source_index, byte_offset, byte_size) "
                            "VALUES (?, ?, ?, ?)",
                            (sample_id, record_index, offset, len(line)),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise RuntimeError(
                            "text cache manifests contains duplicate sample_id/rank cross write"
                        ) from exc

                    rank_handle = rank_handles[rank]
                    while True:
                        rank_line = rank_handle.readline()
                        if not rank_line:
                            raise RuntimeError(
                                "text cache rank/globalThe number of records is inconsistent"
                            )
                        rank_lines[rank] += 1
                        rank_digests[rank].update(rank_line)
                        if rank_line.strip():
                            break
                    rank_row = _decode_cache_jsonl_line(
                        rank_paths[rank],
                        rank_lines[rank],
                        rank_line,
                    )
                    if row != rank_row:
                        raise RuntimeError("text cache rank/globalThe record content is inconsistent")

                    if not first_condition:
                        condition_digest.update(b",")
                    condition_digest.update(
                        canonical_json_bytes(
                            {
                                "sample_id_sha256": canonical_sha256(sample_id),
                                "source_condition_sha256": row.get(
                                    "source_condition_sha256"
                                ),
                            }
                        )
                    )
                    first_condition = False
                    record_index += 1
                for rank, handle in rank_handles.items():
                    for remaining in handle:
                        rank_lines[rank] += 1
                        rank_digests[rank].update(remaining)
                        if remaining.strip():
                            raise RuntimeError(
                                "text cache rank/globalThe number of records is inconsistent"
                            )
                final_manifest_identity = _stat_identity(
                    os.fstat(global_handle.fileno())
                )
            connection.commit()
        except Exception:
            connection.close()
            database_path.unlink(missing_ok=True)
            raise
        finally:
            for handle in rank_handles.values():
                handle.close()
        connection.close()
        if record_index != expected_records or record_index <= 0:
            database_path.unlink(missing_ok=True)
            raise RuntimeError("text cache READYThe number of records is inconsistent")
        if (
            manifest_digest.hexdigest() != expected_manifest_sha256
            or final_manifest_identity != manifest_identity
            or _stat_identity(manifest_path.stat()) != manifest_identity
        ):
            raise RuntimeError("text cache READY manifest SHAInconsistency or drift during read")
        for rank, rank_path in rank_paths.items():
            if (
                rank_digests[rank].hexdigest() != rank_sha256[rank]
                or _stat_identity(rank_path.stat()) != rank_identities[rank]
            ):
                raise RuntimeError(
                    f"text cache rank manifest SHAinconsistent:{rank_path.name}"
                )
        condition_digest.update(b"]")
        if condition_digest.hexdigest() != expected_condition_set_sha256:
            database_path.unlink(missing_ok=True)
            raise RuntimeError("text cache READY condition set SHAinconsistent")
        self._indexed_manifest = manifest_path
        self._indexed_identity = manifest_identity
        self._index_database_path = database_path
        self._index_database_owner_pid = os.getpid()
        self._indexed_count = record_index
        self._indexed_base_dir = manifest_path.parent.resolve()
        self._records = {}
        self._record_bases = {}

    def __len__(self) -> int:
        if self._index_database_path is not None:
            return self._indexed_count
        return len(self._records)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        if self._index_database_path is not None:
            connection = self._open_index_database()
            return tuple(
                str(row[0])
                for row in connection.execute(
                    "SELECT sample_id FROM records ORDER BY source_index"
                )
            )
        return tuple(self._records)

    def _open_index_database(self) -> sqlite3.Connection:
        if self._index_database_path is None:
            raise RuntimeError("text cache SQLiteIndex not initialized")
        current_pid = os.getpid()
        if (
            self._index_connection is not None
            and self._index_connection_pid != current_pid
        ):
            self._index_connection.close()
            self._index_connection = None
        if self._index_connection is None:
            uri = f"file:{self._index_database_path}?mode=ro"
            self._index_connection = sqlite3.connect(uri, uri=True)
            self._index_connection_pid = current_pid
        return self._index_connection

    def _open_indexed_manifest(self) -> int:
        if self._indexed_manifest is None or self._indexed_identity is None:
            raise RuntimeError("text cacheIndex not initialized")
        current_pid = os.getpid()
        if self._indexed_fd is not None and self._indexed_fd_pid != current_pid:
            os.close(self._indexed_fd)
            self._indexed_fd = None
        if self._indexed_fd is None:
            descriptor = os.open(self._indexed_manifest, os.O_RDONLY)
            descriptor_identity = _stat_identity(os.fstat(descriptor))
            try:
                path_identity = _stat_identity(self._indexed_manifest.stat())
            except OSError:
                path_identity = ()
            if (
                descriptor_identity != self._indexed_identity
                or path_identity != self._indexed_identity
            ):
                os.close(descriptor)
                raise RuntimeError("text cache manifestIdentity drifts after indexing")
            self._indexed_fd = descriptor
            self._indexed_fd_pid = current_pid
        return self._indexed_fd

    def record(self, sample_id: str) -> Mapping[str, Any]:
        if self._index_database_path is not None:
            row = self._open_index_database().execute(
                "SELECT byte_offset, byte_size FROM records WHERE sample_id = ?",
                (sample_id,),
            ).fetchone()
            if row is None:
                raise KeyError(
                    f"text cache does not contain sample_id={sample_id!r}"
                )
            offset, size = int(row[0]), int(row[1])
            descriptor = self._open_indexed_manifest()
            try:
                path_identity = _stat_identity(self._indexed_manifest.stat())
            except OSError:
                path_identity = ()
            if (
                _stat_identity(os.fstat(descriptor)) != self._indexed_identity
                or path_identity != self._indexed_identity
            ):
                raise RuntimeError("text cache manifestContent drifts after indexing")
            payload = os.pread(descriptor, size, offset)
            if len(payload) != size:
                raise RuntimeError("text cache manifestIndex short read")
            return _decode_cache_jsonl_line(
                self._indexed_manifest,
                0,
                payload,
            )
        try:
            return self._records[sample_id]
        except KeyError as exc:
            raise KeyError(f"text cache does not contain sample_id={sample_id!r}") from exc

    def record_base(self, sample_id: str) -> Path:

        if self._index_database_path is not None:

            self.record(sample_id)
            if self._indexed_base_dir is None:
                raise RuntimeError("text cacheindex is missingmanifest base_dir")
            return self._indexed_base_dir
        try:
            return self._record_bases[sample_id]
        except KeyError as exc:
            raise KeyError(f"text cache does not contain sample_id={sample_id!r}") from exc

    def load_entry(
        self,
        sample_id: str,
        *,
        expected_source_condition_sha256: str | None = None,
        expected_condition: RenderTextCondition | Mapping[str, Any] | None = None,
    ) -> LoadedTextCacheEntry:
        base_dir = (
            self._indexed_base_dir
            if self._index_database_path is not None
            else self._record_bases[sample_id]
        )
        return read_text_cache_entry(
            self.record(sample_id),
            expected_provenance=self.expected_provenance,
            expected_cache_config_sha256=self.expected_cache_config_sha256,
            expected_source_condition_sha256=(expected_source_condition_sha256),
            expected_condition=expected_condition,
            expected_cache_config_file_sha256=(self.expected_cache_config_file_sha256),
            base_dir=base_dir,
        )

    def close(self) -> None:
        descriptor = self._indexed_fd
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
            self._indexed_fd = None
            self._indexed_fd_pid = None
        connection = self._index_connection
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error:
                pass
            self._index_connection = None
            self._index_connection_pid = None
        database_path = self._index_database_path
        if (
            database_path is not None
            and self._index_database_owner_pid == os.getpid()
        ):
            database_path.unlink(missing_ok=True)
            self._index_database_owner_pid = None

    def __enter__(self) -> "RenderTextCacheLoader":
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_indexed_fd"] = None
        state["_indexed_fd_pid"] = None
        state["_index_connection"] = None
        state["_index_connection_pid"] = None
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(dict(state))

        self._index_database_owner_pid = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def load(
        self,
        sample_id: str,
        *,
        expected_source_condition_sha256: str | None = None,
        expected_condition: RenderTextCondition | Mapping[str, Any] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        return self.load_entry(
            sample_id,
            expected_source_condition_sha256=(expected_source_condition_sha256),
            expected_condition=expected_condition,
        ).conditioner_kwargs(device=device, dtype=dtype)

    def load_batch(
        self,
        sample_ids: Sequence[str],
        *,
        expected_source_condition_sha256: (
            Mapping[str, str] | Sequence[str] | None
        ) = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        if not sample_ids:
            raise ValueError("text cache batch must not be empty")
        if isinstance(expected_source_condition_sha256, Mapping):
            missing = [
                sample_id
                for sample_id in sample_ids
                if sample_id not in expected_source_condition_sha256
            ]
            if missing:
                raise ValueError("condition SHA mapping incomplete coverage sample_ids")
            hashes = [
                expected_source_condition_sha256[sample_id] for sample_id in sample_ids
            ]
        elif expected_source_condition_sha256 is None:
            hashes = [None] * len(sample_ids)
        else:
            hashes = list(expected_source_condition_sha256)
            if len(hashes) != len(sample_ids):
                raise ValueError("condition SHA quantity and sample_ids different")
        entries = [
            self.load_entry(
                sample_id,
                expected_source_condition_sha256=hashes[index],
            )
            for index, sample_id in enumerate(sample_ids)
        ]
        hidden_size = entries[0].description_embeddings.shape[-1]
        storage_dtype = entries[0].description_embeddings.dtype
        for entry in entries:
            if (
                entry.description_embeddings.shape[-1] != hidden_size
                or entry.lyrics_embeddings.shape[-1] != hidden_size
                or entry.description_embeddings.dtype != storage_dtype
                or entry.lyrics_embeddings.dtype != storage_dtype
            ):
                raise RuntimeError("text cache batch hidden shape/dtype inconsistent")

        def pad(
            values: Sequence[torch.Tensor],
            masks: Sequence[torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            compact: list[torch.Tensor] = []
            for value, valid in zip(values, masks, strict=True):
                if (
                    valid.dtype != torch.bool
                    or valid.ndim != 1
                    or value.shape[0] != valid.shape[0]
                ):
                    raise RuntimeError("text cache value/maskInconsistent shape")
                selected = value[valid]
                compact.append(selected)
            maximum = max(value.shape[0] for value in compact)
            hidden = torch.zeros(len(values), maximum, hidden_size, dtype=storage_dtype)
            mask = torch.zeros(len(values), maximum, dtype=torch.bool)
            for index, value in enumerate(compact):
                hidden[index, : value.shape[0]] = value
                mask[index, : value.shape[0]] = True
            return hidden, mask

        description, description_mask = pad(
            [entry.description_embeddings for entry in entries],
            [entry.description_mask for entry in entries],
        )
        lyrics, lyrics_mask = pad(
            [entry.lyrics_embeddings for entry in entries],
            [entry.lyrics_mask for entry in entries],
        )
        target = torch.device(device) if device is not None else None
        return {
            "description_embeddings": description.to(
                device=target, dtype=dtype or description.dtype
            ),
            "description_mask": description_mask.to(device=target),
            "lyrics_embeddings": lyrics.to(device=target, dtype=dtype or lyrics.dtype),
            "lyrics_mask": lyrics_mask.to(device=target),
            "cache_provenance": self.expected_provenance,
        }

    __call__ = load_batch
