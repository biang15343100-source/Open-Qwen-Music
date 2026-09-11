from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from open_qwen_music.common.config import load_config

from .audio import (
    AudioCredentialError,
    audio_password_env,
    load_audio,
    password_from_env,
)
from .contracts import CODEBOOK_SIZE, FRAME_RATE, SAMPLE_RATE
from .frontend import ConvSubsampling25Hz
from .text import CharacterTokenizer

REPORT_SCHEMA = "oqm.tokenizer-manifest-validation.v1"
INDEX_SCHEMA = "oqm.manifest-index.v3"
MANIFEST_SCHEMA = "oqm.manifest.v1"
AUDIO_ASSETS_SCHEMA = "oqm.manifest-audio-assets.v1"
PERMANENT_TAR_STORAGE = "permanent_indexed_tar_v1"
INDEX_CATEGORICAL_FIELDS = (
    "training.sampling_group",
    "training.source_sampling_group",
    "training.ctc_group",
    "training.quality_tier",
    "training.io_group",
)
LOSS_HEADS = ("bestrq", "ctc", "mel", "chroma", "vq")
_READY_STATES = frozenset({"READY", "TRAINING_READY"})
_UNKNOWN_VALUES = frozenset({"", "unknown", "null"})


@lru_cache(maxsize=512)
def _sha256_file_identity(
    path_text: str,
    device: int,
    inode: int,
    size: int,
    mtime_ns: int,
    ctime_ns: int,
) -> str:
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        before = os.fstat(handle.fileno())
        expected = (device, inode, size, mtime_ns, ctime_ns)
        actual_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if actual_before != expected:
            raise OSError(f"File identity changed before SHA-256 reading: {path_text}")
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
        actual_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if actual_after != expected:
            raise OSError(f"File identity changed during SHA-256 reading: {path_text}")
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    file_stat = candidate.stat()
    return _sha256_file_identity(
        str(candidate),
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _nested(value: Mapping[str, Any], field: str, default: Any = None) -> Any:
    current: Any = value
    for key in field.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _first(value: Mapping[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        candidate = _nested(value, field)
        if candidate is not None:
            return candidate
    return None


def _resolve_path(value: str | Path, *, config_path: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists() or config_path is None:
        return cwd_candidate
    return (config_path.parent / path).resolve()


def resolve_audio_uri(record: Mapping[str, Any], manifest_dir: Path) -> str:

    value = (
        record.get("audio_path")
        or _nested(record, "audio.path")
        or _nested(record, "audio.uri")
        or _nested(record, "source.uri")
    )
    if value is None or not str(value).strip():
        raise ValueError("is missing audio_path, audio.path, audio.uri, or source.uri")
    uri = str(value)
    if uri.startswith(("tar://", "zip://")):
        scheme, payload = uri.split("://", 1)
        if payload.count("::") != 1:
            raise ValueError(f"{scheme} URI must contain an archive::member pair")
        archive, member = payload.split("::", 1)
        if not archive or not member:
            raise ValueError(f"{scheme} URI archive/member must not be empty")
        archive_path = Path(archive)
        if not archive_path.is_absolute():
            archive_path = manifest_dir / archive_path
        return f"{scheme}://{archive_path.resolve()}::{member}"
    if uri.startswith("parquet://"):
        body = uri.removeprefix("parquet://")
        if body.count("::") != 1:
            raise ValueError("parquet URI must contain a path::parameters pair")
        parquet_path, parameters = body.split("::", 1)
        path = Path(parquet_path)
        if not path.is_absolute():
            raise ValueError("parquet URI must use absolute path")
        parsed: dict[str, str] = {}
        for chunk in parameters.split("&"):
            key, separator, parameter = chunk.partition("=")
            if not separator or not key or not parameter or key in parsed:
                raise ValueError(f"parquet URI contains an invalid or repeated parameter: {chunk!r}")
            parsed[key] = parameter
        if set(parsed) != {"rg", "row", "col"}:
            raise ValueError("parquet URI parameters must be exactly rg, row, and col")
        if any(
            not parsed[key].isdigit() or (len(parsed[key]) > 1 and parsed[key].startswith("0"))
            for key in ("rg", "row")
        ):
            raise ValueError("parquet URI rg/row must be a canonical non-negative integer")
        column = parsed["col"]
        if not (
            column.isascii()
            and (column[0].isalpha() or column[0] == "_")
            and all(character.isalnum() or character in "_.-" for character in column)
        ):
            raise ValueError("parquet URI has an invalid col parameter")
        return uri
    if uri.startswith("file://"):
        path = Path(uri.removeprefix("file://"))
        if not path.is_absolute():
            raise ValueError("file URI must use absolute path")
        return f"file://{path}"
    if "://" in uri:
        raise ValueError(f"Unsupported audio URI scheme: {uri.split('://', 1)[0]}")
    path = Path(uri)
    return str(path if path.is_absolute() else (manifest_dir / path).resolve())


def _storage_class(uri: str) -> str:
    for scheme in ("parquet", "tar", "zip", "file"):
        if uri.startswith(f"{scheme}://"):
            return scheme
    return "file"


@dataclass(frozen=True, slots=True)
class _Issue:
    code: str
    message: str
    line: int | None = None
    sample_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.line is not None:
            value["line"] = self.line
        if self.sample_id is not None:
            value["sample_id"] = self.sample_id
        return value


class _Issues:
    def __init__(self, *, detail_limit: int = 1000) -> None:
        self.detail_limit = detail_limit
        self.errors: list[_Issue] = []
        self.warnings: list[_Issue] = []
        self.error_counts: Counter[str] = Counter()
        self.warning_counts: Counter[str] = Counter()

    def error(
        self,
        code: str,
        message: str,
        *,
        line: int | None = None,
        sample_id: str | None = None,
    ) -> None:
        self.error_counts[code] += 1
        if len(self.errors) < self.detail_limit:
            self.errors.append(_Issue(code, message, line, sample_id))

    def warning(
        self,
        code: str,
        message: str,
        *,
        line: int | None = None,
        sample_id: str | None = None,
    ) -> None:
        self.warning_counts[code] += 1
        if len(self.warnings) < self.detail_limit:
            self.warnings.append(_Issue(code, message, line, sample_id))

    @property
    def error_count(self) -> int:
        return sum(self.error_counts.values())

    @property
    def warning_count(self) -> int:
        return sum(self.warning_counts.values())


@dataclass(frozen=True, slots=True)
class _RecordLink:
    sample_id: str
    split: str
    group_id: str
    license_id: str
    commercial_ok: bool | None
    start_sec: float
    duration_sec: float


@dataclass(frozen=True, slots=True)
class _DerivedLink:
    sample_id: str
    parent_id: str
    child: _RecordLink
    line: int


@dataclass(frozen=True, slots=True)
class _DecodeCandidate:
    key: str
    sample_id: str
    uri: str
    storage: str
    dataset: str
    tier: str
    split: str
    start_sec: float
    duration_sec: float
    archive_offset: int | None
    archive_size: int | None
    password_env: str | None
    asset_id: str | None
    payload_sha256: str | None
    asset_revision: str | None
    shard_sha256: str | None


class _DecodeSampler:
    def __init__(self, limit: int) -> None:
        self.limit = max(0, int(limit))
        self.best_by_credential: dict[str, _DecodeCandidate] = {}
        self.best_by_asset_revision: dict[str, _DecodeCandidate] = {}
        self.best_by_dimension: dict[tuple[str, str], _DecodeCandidate] = {}
        self.global_best: list[_DecodeCandidate] = []

    def add(self, candidate: _DecodeCandidate) -> None:
        if self.limit <= 0:
            return
        if candidate.password_env is not None:
            previous = self.best_by_credential.get(candidate.password_env)
            if previous is None or candidate.key < previous.key:
                self.best_by_credential[candidate.password_env] = candidate
        if candidate.asset_revision is not None:
            previous = self.best_by_asset_revision.get(candidate.asset_revision)
            if previous is None or candidate.key < previous.key:
                self.best_by_asset_revision[candidate.asset_revision] = candidate
        for dimension, value in (
            ("storage", candidate.storage),
            ("dataset", candidate.dataset),
            ("tier", candidate.tier),
        ):
            key = (dimension, value)
            previous = self.best_by_dimension.get(key)
            if previous is None or candidate.key < previous.key:
                self.best_by_dimension[key] = candidate
        self.global_best.append(candidate)
        self.global_best.sort(key=lambda item: item.key)
        del self.global_best[self.limit :]

    def selected(self) -> list[_DecodeCandidate]:
        if self.limit <= 0:
            return []
        result: dict[str, _DecodeCandidate] = {}

        for candidate in sorted(self.best_by_credential.values(), key=lambda item: item.key):
            result.setdefault(candidate.sample_id, candidate)
            if len(result) >= self.limit:
                return list(result.values())

        for candidate in sorted(self.best_by_asset_revision.values(), key=lambda item: item.key):
            result.setdefault(candidate.sample_id, candidate)
            if len(result) >= self.limit:
                return list(result.values())
        for candidate in sorted(self.best_by_dimension.values(), key=lambda item: item.key):
            result.setdefault(candidate.sample_id, candidate)
            if len(result) >= self.limit:
                return list(result.values())
        for candidate in self.global_best:
            result.setdefault(candidate.sample_id, candidate)
            if len(result) >= self.limit:
                break
        return list(result.values())


@dataclass(frozen=True, slots=True)
class _ObservedAudioAsset:
    asset_id: str
    asset_revision: str
    shard_sha256: str
    path: Path
    size_bytes: int | None


_PRIMED_AUDIO_ASSET_PATH_SIZES: dict[Path, int] = {}


class _ManifestAudioAssets:
    def __init__(self, issues: _Issues) -> None:
        self.issues = issues
        self.records = 0
        self.assets: dict[str, _ObservedAudioAsset] = {}
        self.path_assets: dict[Path, str] = {}
        self.path_sizes: dict[Path, int | None] = dict(_PRIMED_AUDIO_ASSET_PATH_SIZES)
        self.resolved_paths: dict[str, Path] = {str(path): path for path in self.path_sizes}
        self.payloads: set[tuple[str, int, int, str]] = set()
        self.manifest_payload_bytes = 0
        self.revisions: set[str] = set()
        self.catalog_sha256_values: set[str] = set()
        self.content_manifest_sha256_values: set[str] = set()
        self.storage_counts: Counter[str] = Counter()
        self.digest = hashlib.sha256()

    def observe(
        self,
        *,
        asset_id: str,
        asset_revision: str,
        shard_sha256: str,
        payload_sha256: str,
        uri: str,
        archive_offset: int,
        archive_size: int,
        catalog_sha256: str | None,
        content_manifest_sha256: str | None,
        line: int,
        sample_id: str,
    ) -> None:
        body = uri.removeprefix("tar://")
        archive_text, _member = body.split("::", 1)
        path = self.resolved_paths.get(archive_text)
        if path is None:
            path = Path(archive_text).resolve()
            self.resolved_paths[archive_text] = path
        if path not in self.path_sizes:
            try:
                file_stat = path.stat()
                if not path.is_file():
                    raise OSError("is not an ordinary file")
                self.path_sizes[path] = int(file_stat.st_size)
            except OSError as exc:
                self.path_sizes[path] = None
                self.issues.error(
                    "audio_asset_shard_missing",
                    f"permanent TAR shard is unreadable: {path}: {exc}",
                    line=line,
                    sample_id=sample_id,
                )
        size_bytes = self.path_sizes[path]
        if size_bytes is not None and (
            archive_offset > size_bytes or archive_size > size_bytes - archive_offset
        ):
            self.issues.error(
                "audio_archive_bounds",
                "audio.archive_offset/archive_size exceeds the shard file range: "
                f"offset={archive_offset} size={archive_size} "
                f"file_size={size_bytes}",
                line=line,
                sample_id=sample_id,
            )

        observed = _ObservedAudioAsset(
            asset_id=asset_id,
            asset_revision=asset_revision,
            shard_sha256=shard_sha256,
            path=path,
            size_bytes=size_bytes,
        )
        previous = self.assets.setdefault(asset_id, observed)
        if previous != observed:
            self.issues.error(
                "audio_asset_id_drift",
                f"audio.asset_id={asset_id!r} is bound to multiple revision/shard/path/size",
                line=line,
                sample_id=sample_id,
            )
        previous_asset_id = self.path_assets.setdefault(path, asset_id)
        if previous_asset_id != asset_id:
            self.issues.error(
                "audio_asset_path_alias",
                f"same as shard path is bound to multiple asset_id: "
                f"{previous_asset_id!r}/{asset_id!r}",
                line=line,
                sample_id=sample_id,
            )
        payload_key = (
            asset_id,
            archive_offset,
            archive_size,
            payload_sha256,
        )
        if payload_key not in self.payloads:
            self.payloads.add(payload_key)
            self.manifest_payload_bytes += archive_size
        self.records += 1
        self.revisions.add(asset_revision)
        if catalog_sha256 is not None:
            self.catalog_sha256_values.add(catalog_sha256)
        if content_manifest_sha256 is not None:
            self.content_manifest_sha256_values.add(content_manifest_sha256)
        self.storage_counts["tar"] += 1
        self.digest.update(
            json.dumps(
                {
                    "asset_id": asset_id,
                    "asset_revision": asset_revision,
                    "archive": str(path),
                    "archive_offset": archive_offset,
                    "archive_size": archive_size,
                    "payload_sha256": payload_sha256,
                    "shard_sha256": shard_sha256,
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )

    def summary(self) -> dict[str, Any] | None:
        if self.records == 0:
            return None
        if len(self.revisions) != 1:
            self.issues.error(
                "audio_asset_revision_mixed",
                f"Manifest contains multiple audio.asset_revision values: {sorted(self.revisions)[:8]}",
            )
        if len(self.catalog_sha256_values) > 1:
            self.issues.error(
                "audio_assets_catalog_mixed",
                "Manifest mixed with multiple asset catalog SHA",
            )
        if len(self.content_manifest_sha256_values) > 1:
            self.issues.error(
                "audio_assets_content_manifest_mixed",
                "Manifest mixed with multiple asset CONTENT_MANIFEST SHA",
            )
        result = {
            "schema_version": AUDIO_ASSETS_SCHEMA,
            "audio_copied": True,
            "storage_format": PERMANENT_TAR_STORAGE,
            "asset_revision": (next(iter(self.revisions)) if len(self.revisions) == 1 else None),
            "records": self.records,
            "assets": len(self.assets),
            "payloads": len(self.payloads),
            "manifest_payload_bytes": self.manifest_payload_bytes,
            "storage_counts": dict(sorted(self.storage_counts.items())),
            "manifest_assets_sha256": self.digest.hexdigest(),
        }
        if len(self.catalog_sha256_values) == 1:
            result["catalog_sha256"] = next(iter(self.catalog_sha256_values))
        if len(self.content_manifest_sha256_values) == 1:
            result["content_manifest_sha256"] = next(iter(self.content_manifest_sha256_values))
        return result


class _IndexAudit:
    def __init__(
        self,
        index_dir: Path,
        manifest: Path,
        issues: _Issues,
        *,
        expected_metadata_sha256: str | None = None,
    ) -> None:
        self.index_dir = index_dir
        self.manifest = manifest
        self.issues = issues
        self.metadata: dict[str, Any] | None = None
        self.metadata_sha256: str | None = None
        self.records: int | None = None
        self.arrays: dict[str, np.memmap[Any, Any]] = {}
        self.categorical: dict[str, tuple[np.memmap[Any, Any], list[str]]] = {}
        self._prepare(expected_metadata_sha256)

    def _prepare(self, expected_metadata_sha256: str | None) -> None:
        metadata_path = self.index_dir / "metadata.json"
        ready_path = self.index_dir / "READY"
        if not metadata_path.is_file() or not ready_path.is_file():
            self.issues.error(
                "index_not_ready",
                f"index is missing metadata.json or READY: {self.index_dir}",
            )
            return
        try:
            metadata_bytes = metadata_path.read_bytes()
            ready_digest = ready_path.read_text(encoding="utf-8").strip()
            metadata = json.loads(metadata_bytes)
        except (OSError, json.JSONDecodeError) as exc:
            self.issues.error("index_unreadable", f"index is unreadable: {exc}")
            return
        digest = hashlib.sha256(metadata_bytes).hexdigest()
        self.metadata_sha256 = digest
        if ready_digest != digest:
            self.issues.error(
                "index_ready_drift", "index READY digest does not match metadata.json"
            )
        if expected_metadata_sha256 and digest != expected_metadata_sha256:
            self.issues.error(
                "index_lineage_sha_mismatch",
                "config lineage index SHA-256 does not match metadata.json",
            )
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema_version") != INDEX_SCHEMA
            or metadata.get("complete") is not True
        ):
            self.issues.error(
                "index_schema",
                f"index must be complete {INDEX_SCHEMA}",
            )
            return
        self.metadata = metadata
        try:
            self.records = int(metadata["records"])
        except (KeyError, TypeError, ValueError):
            self.issues.error("index_records", "index record count is invalid")
            return
        actual_manifest_sha = sha256_file(self.manifest)
        if metadata.get("manifest_sha256") != actual_manifest_sha:
            self.issues.error(
                "index_manifest_sha_drift", "index manifest SHA-256 binding does not match"
            )
        try:
            source_size = int(metadata.get("source_size", -1))
        except (TypeError, ValueError):
            source_size = -1
        if source_size != self.manifest.stat().st_size:
            self.issues.error(
                "index_manifest_size_drift", "index manifest size binding does not match"
            )
        specifications = metadata.get("arrays")
        if not isinstance(specifications, Mapping):
            self.issues.error("index_arrays", "index arrays must be an object")
            return
        for filename, specification in specifications.items():
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not isinstance(specification, Mapping)
            ):
                self.issues.error("index_array_spec", f"invalid array specification: {filename!r}")
                continue
            try:
                dtype = np.dtype(specification["dtype"])
                length = int(specification["length"])
                size = int(specification["size_bytes"])
                expected_sha = str(specification["sha256"])
            except (KeyError, TypeError, ValueError) as exc:
                self.issues.error(
                    "index_array_spec",
                    f"array {filename} is missing SHA, size, dtype, or length: {exc}",
                )
                continue
            path = self.index_dir / filename
            if length < 0 or size != length * dtype.itemsize:
                self.issues.error(
                    "index_array_shape", f"array specification is inconsistent: {filename}"
                )
                continue
            if not path.is_file() or path.stat().st_size != size:
                self.issues.error(
                    "index_array_size_drift",
                    f"array is missing or has an unexpected size: {path}",
                )
                continue
            if sha256_file(path) != expected_sha:
                self.issues.error("index_array_sha_drift", f"array SHA-256 does not match: {path}")
                continue
            try:
                self.arrays[filename] = np.memmap(path, mode="r", dtype=dtype, shape=(length,))
            except (OSError, ValueError) as exc:
                self.issues.error(
                    "index_array_unreadable",
                    f"array cannot be memory-mapped: {path}: {exc}",
                )
        for filename, dtype in (
            ("offsets.i64", np.dtype("<i8")),
            ("durations.f32", np.dtype("<f4")),
            ("splits.u16", np.dtype("<u2")),
        ):
            array = self.arrays.get(filename)
            if array is None or array.dtype != dtype or len(array) != self.records:
                self.issues.error(
                    "index_required_array",
                    f"{filename} must be {dtype} and length=records",
                )
        categorical = metadata.get("categorical_fields")
        if not isinstance(categorical, Mapping):
            self.issues.error("index_categorical", "index is missing categorical_fields")
            return
        for field in INDEX_CATEGORICAL_FIELDS:
            specification = categorical.get(field)
            if not isinstance(specification, Mapping):
                self.issues.error(
                    "index_categorical_missing", f"index is missing categorical field {field}"
                )
                continue
            filename = specification.get("file")
            names = specification.get("names")
            array = self.arrays.get(str(filename))
            if (
                array is None
                or array.dtype != np.dtype("<u2")
                or len(array) != self.records
                or not isinstance(names, list)
                or not names
                or not all(isinstance(name, str) for name in names)
            ):
                self.issues.error(
                    "index_categorical_spec",
                    f"invalid categorical field specification: {field}",
                )
                continue
            if int(array.max(initial=0)) >= len(names):
                self.issues.error(
                    "index_categorical_ids", f"categorical field ID is out of bounds: {field}"
                )
                continue
            self.categorical[field] = (array, list(names))

    def observe(
        self,
        ordinal: int,
        offset: int,
        record: Mapping[str, Any],
        *,
        line: int,
        sample_id: str,
    ) -> None:
        if self.records is None or ordinal >= self.records:
            self.issues.error(
                "index_record_count",
                "manifest contains more records than the index",
                line=line,
                sample_id=sample_id,
            )
            return
        offsets = self.arrays.get("offsets.i64")
        if offsets is not None and int(offsets[ordinal]) != offset:
            self.issues.error(
                "index_offset_mismatch",
                "index offset does not match the JSONL record boundary",
                line=line,
                sample_id=sample_id,
            )
        durations = self.arrays.get("durations.f32")
        declared = _nested(record, "audio.duration_sec", 0.0)
        try:
            duration = float(declared)
        except (TypeError, ValueError):
            duration = 0.0
        if durations is not None and not math.isclose(
            float(durations[ordinal]), duration, rel_tol=1e-5, abs_tol=1e-4
        ):
            self.issues.error(
                "index_duration_mismatch",
                "index duration does not match the manifest",
                line=line,
                sample_id=sample_id,
            )
        splits = self.arrays.get("splits.u16")
        split_names = self.metadata.get("split_names", []) if self.metadata else []
        split = str(record.get("split", "train"))
        if splits is not None:
            split_id = int(splits[ordinal])
            indexed = (
                split_names[split_id]
                if isinstance(split_names, list)
                and split_id < len(split_names)
                and isinstance(split_names[split_id], str)
                else None
            )
            if indexed != split:
                self.issues.error(
                    "index_split_mismatch",
                    "index split does not match the manifest",
                    line=line,
                    sample_id=sample_id,
                )
        for field, (array, names) in self.categorical.items():
            expected = _group_value(record, field)
            value_id = int(array[ordinal])
            indexed = names[value_id] if value_id < len(names) else None
            if indexed != expected:
                self.issues.error(
                    "index_categorical_mismatch",
                    f"index {field}={indexed!r}, manifest={expected!r}",
                    line=line,
                    sample_id=sample_id,
                )

    def finish(self, observed_records: int) -> None:
        if self.records is not None and observed_records != self.records:
            self.issues.error(
                "index_record_count",
                f"manifest records={observed_records}, index records={self.records}",
            )


def _group_value(record: Mapping[str, Any], field: str) -> str:
    if field == "training.io_group":
        value = _first(record, ("training.io_group", "audio.io_group"))
    else:
        value = _nested(record, field)
    return str(value) if value is not None and str(value) else "unknown"


def _finite_nonnegative(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0.0


def _is_weight_field(name: str) -> bool:
    key = name.casefold()
    return (
        key in {"weight", "weights"}
        or key.startswith("weight_")
        or key.startswith("weights_")
        or key.endswith("_weight")
        or key.endswith("_weights")
    )


def _record_weights(
    value: Any,
    prefix: str = "training",
    *,
    inside_weights: bool = False,
) -> list[tuple[str, Any]]:

    result: list[tuple[str, Any]] = []
    if not isinstance(value, Mapping):
        return result
    for key, child in value.items():
        field = f"{prefix}.{key}"
        weighted = inside_weights or _is_weight_field(str(key))
        if isinstance(child, Mapping):
            result.extend(_record_weights(child, field, inside_weights=weighted))
        elif isinstance(child, list) and weighted:
            result.extend((f"{field}[{index}]", item) for index, item in enumerate(child))
        elif weighted:
            result.append((field, child))
    return result


def _lyrics(record: Mapping[str, Any]) -> str:
    if "lyrics" in record:
        value = record["lyrics"]
        if not isinstance(value, str):
            raise TypeError("lyrics must be a string")
        return value
    text = record.get("text") or {}
    if not isinstance(text, Mapping):
        raise TypeError("text must be an object")
    if "lyrics" in text:
        value = text["lyrics"]
        if not isinstance(value, str):
            raise TypeError("text.lyrics must be a string")
        return value
    sections = text.get("sections") or []
    if not isinstance(sections, list):
        raise TypeError("text.sections must be an array")
    return "\n".join(
        str(section.get("lyrics", "")) for section in sections if isinstance(section, Mapping)
    ).strip()


def _ctc_units(record: Mapping[str, Any]) -> list[str | int] | None:
    value = record.get("ctc_units")
    if value is None:
        value = _nested(record, "text.phonemes")
    if value is None:
        return None
    if isinstance(value, str):
        units: list[Any] = value.split()
    elif isinstance(value, list):
        units = value
    else:
        raise TypeError("ctc_units must be a string or array")
    if not units:
        return None
    if any(isinstance(unit, bool) or not isinstance(unit, (str, int)) for unit in units):
        raise TypeError("ctc_units can contain only strings or integer token IDs")
    return list(units)


def _encode_ctc(
    record: Mapping[str, Any],
    tokenizer: CharacterTokenizer | None,
) -> tuple[list[int], list[str | int] | None]:
    units = _ctc_units(record)
    if units is not None:
        if tokenizer is None:
            if not all(isinstance(unit, int) for unit in units):
                raise ValueError(
                    "String ctc_units require --ctc-vocab or config.data.vocab during validation"
                )
            return [int(unit) for unit in units], units
        ids: list[int] = []
        for unit in units:
            if isinstance(unit, int):
                token_id = unit
                if token_id <= 0 or token_id >= len(tokenizer):
                    raise ValueError(f"CTC token ID is out of bounds or uses the blank ID: {token_id}")
                ids.append(token_id)
                continue
            if unit not in tokenizer.token_to_id:
                raise ValueError(f"ctc_units contains units outside the vocabulary: {unit!r}")
            token_id = tokenizer.token_to_id[unit]
            if token_id == 0:
                raise ValueError("ctc_units must not explicitly include CTC blank")
            ids.append(token_id)
        return ids, units
    lyrics = _lyrics(record)
    if not lyrics.strip():
        raise ValueError("CTC is enabled, but lyrics are empty")
    if tokenizer is None:
        raise ValueError("Encoding lyrics requires --ctc-vocab or config.data.vocab")
    ids = tokenizer.encode(lyrics)
    if not ids:
        raise ValueError("The frozen CharacterTokenizer produced an empty encoding")
    if any(token_id <= 0 or token_id >= len(tokenizer) for token_id in ids):
        raise ValueError("The frozen CharacterTokenizer produced an out-of-bounds or blank token")
    return ids, None


def _ctc_required_frames(ids: list[int]) -> int:
    return len(ids) + sum(left == right for left, right in zip(ids, ids[1:]))


def _actual_output_frames(duration_sec: float, config: Mapping[str, Any]) -> int:
    features_value = config.get("features")
    features = features_value if isinstance(features_value, Mapping) else {}
    try:
        sample_rate = int(features.get("sample_rate", SAMPLE_RATE))
        hop_length = int(features.get("hop_length", sample_rate // 100))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0
    if sample_rate <= 0 or hop_length <= 0:
        return 0
    samples = round(duration_sec * sample_rate)
    feature_frames = torch.tensor([samples // hop_length], dtype=torch.long)
    return int(ConvSubsampling25Hz.output_lengths(feature_frames)[0].item())


def _is_derived(record: Mapping[str, Any]) -> bool:
    return bool(
        record.get("is_derived")
        or _nested(record, "source.is_derived")
        or record.get("derived")
        or record.get("segment_kind") == "ctc_section"
        or _nested(record, "training.segment_kind") == "ctc_section"
    )


def _parent_id(record: Mapping[str, Any]) -> str:
    value = _first(
        record,
        (
            "parent_sample_id",
            "parent_uid_hex",
            "derived.parent_sample_id",
            "derived.parent_id",
            "source.parent_sample_id",
            "source.parent_uid_hex",
        ),
    )
    return "" if value is None else str(value)


def _content_hash(record: Mapping[str, Any]) -> str:
    value = _first(
        record,
        (
            "source.content_sha256",
            "source.content_hash_l1",
            "source.sha256",
            "content_sha256",
            "content_hash_l1",
        ),
    )
    return "" if value is None else str(value)


def _commercial_ok(record: Mapping[str, Any]) -> bool | None:
    value = _first(
        record,
        (
            "source.commercial_ok",
            "commercial_ok",
            "training.license_policy_pass",
        ),
    )
    return value if isinstance(value, bool) else None


def _is_canonical(record: Mapping[str, Any]) -> bool | None:
    value = _first(record, ("source.is_canonical", "is_canonical", "canonical"))
    if isinstance(value, bool):
        return value
    status = _first(record, ("source.dup_status", "dup_status"))
    if status is not None:
        return str(status).casefold() == "canonical"
    canonical_id = _first(
        record,
        ("source.canonical_uid_hex", "canonical_uid_hex", "source.canonical_id"),
    )
    own_id = _first(record, ("source.uid_hex", "uid_hex", "sample_id"))
    if canonical_id is not None and own_id is not None:
        return str(canonical_id) == str(own_id)
    return None


def _is_holdout(record: Mapping[str, Any]) -> bool:
    if any(
        _nested(record, field) is True
        for field in (
            "eval_holdout",
            "training.eval_holdout",
            "training.is_holdout",
            "source.eval_holdout",
        )
    ):
        return True
    reasons = _first(record, ("training.reason_codes", "reason_codes"))
    return isinstance(reasons, list) and "eval_holdout" in reasons


def _revision(record: Mapping[str, Any]) -> str:
    value = _first(
        record,
        (
            "corpus_revision",
            "lineage.corpus_revision",
            "source.corpus_revision",
            "source.release_revision",
        ),
    )
    return "" if value is None else str(value)


def _license_view(config: Mapping[str, Any]) -> str:
    value = _first(
        config,
        (
            "data.release_view",
            "data.license_view",
            "data.view",
            "lineage.release_view",
        ),
    )
    return "" if value is None else str(value)


def _expected_sha(lineage: Mapping[str, Any], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = lineage.get(name)
        if value is not None:
            return str(value)
    return None


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class _AudioAssetsDeclaration:
    label: str
    storage_format: str | None
    asset_revision: str | None
    catalog_path: Path | None
    catalog_sha256: str | None
    total_payload_bytes: int | None

    def identity(self) -> dict[str, Any]:
        return {
            "storage_format": self.storage_format,
            "asset_revision": self.asset_revision,
            "catalog_sha256": self.catalog_sha256,
            "total_payload_bytes": self.total_payload_bytes,
        }


def _direct_audio_assets_values(
    value: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    result: list[tuple[str, Mapping[str, Any]]] = []
    for field in ("audio_assets", "data.audio_assets", "lineage.audio_assets"):
        candidate = _nested(value, field)
        if isinstance(candidate, Mapping):
            result.append((field, candidate))
    return result


def _parse_audio_assets_declaration(
    value: Mapping[str, Any],
    *,
    label: str,
    context_path: Path | None,
    issues: _Issues,
    require_catalog_path: bool,
) -> _AudioAssetsDeclaration:
    schema = value.get("schema_version")
    if schema is not None and schema != AUDIO_ASSETS_SCHEMA:
        issues.error(
            "audio_assets_schema",
            f"{label}.schema_version must be {AUDIO_ASSETS_SCHEMA}",
        )
    storage_value = _first(
        value,
        ("storage_format", "storage_layout", "format", "storage"),
    )
    storage_format = str(storage_value) if storage_value is not None else None
    if storage_format != PERMANENT_TAR_STORAGE:
        issues.error(
            "audio_assets_storage",
            f"{label}.storage_format must be {PERMANENT_TAR_STORAGE}",
        )
    revision_value = _first(
        value,
        ("asset_revision", "revision", "release_revision"),
    )
    asset_revision = (
        str(revision_value) if isinstance(revision_value, str) and revision_value.strip() else None
    )
    if asset_revision is None:
        issues.error(
            "audio_assets_revision",
            f"{label}.asset_revision must be a non-empty string",
        )
    catalog_value = _first(
        value,
        ("catalog_path", "catalog", "asset_catalog", "path"),
    )
    catalog_path: Path | None = None
    if isinstance(catalog_value, str) and catalog_value.strip():
        catalog_path = _resolve_path(
            catalog_value,
            config_path=context_path,
        )
    elif require_catalog_path:
        issues.error(
            "audio_assets_catalog_path",
            f"{label} must declare asset catalog path",
        )
    catalog_sha_value = _first(
        value,
        ("catalog_sha256", "asset_catalog_sha256"),
    )
    catalog_sha256 = str(catalog_sha_value) if catalog_sha_value is not None else None
    if not _is_sha256(catalog_sha256):
        issues.error(
            "audio_assets_catalog_sha",
            f"{label}.catalog_sha256 must be 64 lowercase hexadecimal SHA-256",
        )
        catalog_sha256 = None
    total_value = _first(
        value,
        ("total_payload_bytes", "payload_bytes"),
    )
    if total_value is None:
        total_payload_bytes = None
    elif isinstance(total_value, bool) or not isinstance(total_value, int) or total_value < 0:
        issues.error(
            "audio_assets_total_bytes",
            f"{label}.total_payload_bytes must be a non-negative integer",
        )
        total_payload_bytes = None
    else:
        total_payload_bytes = total_value
    return _AudioAssetsDeclaration(
        label=label,
        storage_format=storage_format,
        asset_revision=asset_revision,
        catalog_path=catalog_path,
        catalog_sha256=catalog_sha256,
        total_payload_bytes=total_payload_bytes,
    )


def _audio_assets_declarations(
    value: Mapping[str, Any] | None,
    *,
    label: str,
    context_path: Path | None,
    issues: _Issues,
    require_catalog_path: bool,
) -> list[_AudioAssetsDeclaration]:
    if value is None:
        return []
    direct = _direct_audio_assets_values(value)
    result = [
        _parse_audio_assets_declaration(
            candidate,
            label=f"{label}.{field}",
            context_path=context_path,
            issues=issues,
            require_catalog_path=require_catalog_path,
        )
        for field, candidate in direct
    ]
    if result:
        expected = result[0].identity()
        for candidate in result[1:]:
            if candidate.identity() != expected:
                issues.error(
                    "audio_assets_conflicting_binding",
                    f"{label} contains audio_assets declarations with different identities",
                )
    return result


def _catalog_asset_rows(catalog: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = catalog.get("assets")
    if value is None:
        value = catalog.get("shards")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        rows: list[Mapping[str, Any]] = []
        for asset_id, item in value.items():
            if not isinstance(item, Mapping):
                continue
            row = dict(item)
            row.setdefault("asset_id", str(asset_id))
            rows.append(row)
        return rows
    return []


_PRIMED_AUDIO_ASSET_CATALOGS: dict[
    tuple[str, str],
    tuple[
        Mapping[str, Any],
        tuple[tuple[str, Path, str, int], ...],
        tuple[tuple[str, str], ...],
    ],
] = {}


def _prime_audio_asset_catalog_cache(
    path: Path,
    expected_sha: str,
    catalog: Mapping[str, Any],
    assets: Mapping[str, tuple[Path, str, int]],
) -> None:

    resolved = path.resolve()
    if sha256_file(resolved) != expected_sha:
        raise ValueError("audio asset catalog SHA-256 changed while priming the cache")
    normalized = tuple(
        (
            str(asset_id),
            Path(asset_path),
            str(asset_sha),
            int(asset_size),
        )
        for asset_id, (asset_path, asset_sha, asset_size) in sorted(assets.items())
    )
    _PRIMED_AUDIO_ASSET_CATALOGS[(str(resolved), expected_sha)] = (
        catalog,
        normalized,
        (),
    )
    _PRIMED_AUDIO_ASSET_PATH_SIZES.update(
        {asset_path: asset_size for _, asset_path, _, asset_size in normalized}
    )
    _audit_audio_asset_catalog_once.cache_clear()


def _audit_audio_asset_catalog_once(
    path_text: str,
    expected_sha: str,
    device: int | None = None,
    inode: int | None = None,
    size: int | None = None,
    mtime_ns: int | None = None,
    ctime_ns: int | None = None,
) -> tuple[
    Mapping[str, Any] | None,
    tuple[tuple[str, Path, str, int], ...],
    tuple[tuple[str, str], ...],
]:

    path = Path(path_text).resolve()
    if None in (device, inode, size, mtime_ns, ctime_ns):
        try:
            file_stat = path.stat()
        except OSError:
            device, inode, size, mtime_ns, ctime_ns = (0, 0, 0, 0, 0)
        else:
            device, inode, size, mtime_ns, ctime_ns = (
                file_stat.st_dev,
                file_stat.st_ino,
                file_stat.st_size,
                file_stat.st_mtime_ns,
                file_stat.st_ctime_ns,
            )
    return _audit_audio_asset_catalog_for_identity(
        str(path),
        expected_sha,
        int(device),
        int(inode),
        int(size),
        int(mtime_ns),
        int(ctime_ns),
    )


@lru_cache(maxsize=16)
def _audit_audio_asset_catalog_for_identity(
    path_text: str,
    expected_sha: str,
    device: int,
    inode: int,
    size: int,
    mtime_ns: int,
    ctime_ns: int,
) -> tuple[
    Mapping[str, Any] | None,
    tuple[tuple[str, Path, str, int], ...],
    tuple[tuple[str, str], ...],
]:

    path = Path(path_text)
    primed = _PRIMED_AUDIO_ASSET_CATALOGS.get((str(path), expected_sha))
    if primed is not None:
        if not path.is_file() or sha256_file(path) != expected_sha:
            return (
                None,
                (),
                (("audio_assets_catalog_sha", f"asset catalog SHA mismatch: {path}"),),
            )
        return primed
    errors: list[tuple[str, str]] = []
    if not path.is_file():
        return (
            None,
            (),
            (("audio_assets_catalog_missing", f"asset catalog does not exist: {path}"),),
        )
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        return (
            None,
            (),
            (("audio_assets_catalog_sha", f"asset catalog SHA mismatch: {path}"),),
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return (
            None,
            (),
            (
                (
                    "audio_assets_catalog_json",
                    f"asset catalog could not be parsed: {path}: {exc}",
                ),
            ),
        )
    if not isinstance(value, Mapping):
        return (
            None,
            (),
            (("audio_assets_catalog_json", "asset catalog must be an object"),),
        )
    rows = _catalog_asset_rows(value)
    if not rows:
        return (
            value,
            (),
            (("audio_assets_catalog_assets", "asset catalog is missing assets/shards"),),
        )
    catalog_assets: dict[str, tuple[Path, str, int]] = {}
    for row in rows:
        asset_id_value = _first(row, ("asset_id", "shard_id", "id"))
        path_value = _first(
            row,
            (
                "path",
                "relative_path",
                "relpath",
                "tar_path",
                "tar_relpath",
                "shard_path",
            ),
        )
        sha_value = _first(row, ("sha256", "shard_sha256"))
        size_value = _first(row, ("size_bytes", "bytes", "shard_size"))
        if (
            not isinstance(asset_id_value, str)
            or not asset_id_value
            or not isinstance(path_value, str)
            or not path_value
            or not _is_sha256(sha_value)
            or isinstance(size_value, bool)
            or not isinstance(size_value, int)
            or size_value <= 0
        ):
            errors.append(
                (
                    "audio_assets_catalog_entry",
                    f"invalid asset catalog shard entry: asset_id={asset_id_value!r}",
                )
            )
            continue
        shard_path = Path(path_value)
        if not shard_path.is_absolute():
            shard_path = path.parent / shard_path
        shard_path = shard_path.resolve()
        if asset_id_value in catalog_assets:
            errors.append(
                (
                    "audio_assets_catalog_duplicate",
                    f"duplicate asset_id in asset catalog: {asset_id_value}",
                )
            )
            continue
        catalog_assets[asset_id_value] = (
            shard_path,
            str(sha_value),
            size_value,
        )
        try:
            actual_size = shard_path.stat().st_size
        except OSError as exc:
            errors.append(
                (
                    "audio_assets_catalog_shard_missing",
                    f"catalog shard is not readable: {shard_path}: {exc}",
                )
            )
        else:
            if actual_size != size_value:
                errors.append(
                    (
                        "audio_assets_catalog_shard_size",
                        f"catalog shard size mismatch: {shard_path} {actual_size} != {size_value}",
                    )
                )
    assets = tuple(
        (asset_id, path_value, sha_value, size_value)
        for asset_id, (path_value, sha_value, size_value) in catalog_assets.items()
    )
    return value, assets, tuple(errors)


_audit_audio_asset_catalog_once.cache_clear = (  # type: ignore[attr-defined]
    _audit_audio_asset_catalog_for_identity.cache_clear
)
_audit_audio_asset_catalog_once.cache_info = (  # type: ignore[attr-defined]
    _audit_audio_asset_catalog_for_identity.cache_info
)


def _validate_audio_asset_catalog(
    declaration: _AudioAssetsDeclaration,
    observed: _ManifestAudioAssets,
    issues: _Issues,
) -> None:
    path = declaration.catalog_path
    expected_sha = declaration.catalog_sha256
    if path is None or expected_sha is None:
        return
    try:
        file_stat = path.resolve().stat()
        identity = (
            file_stat.st_dev,
            file_stat.st_ino,
            file_stat.st_size,
            file_stat.st_mtime_ns,
            file_stat.st_ctime_ns,
        )
    except OSError:
        identity = (0, 0, 0, 0, 0)
    value, cached_assets, cached_errors = _audit_audio_asset_catalog_once(
        str(path),
        expected_sha,
        *identity,
    )
    for code, message in cached_errors:
        issues.error(code, message)
    if value is None or not cached_assets:
        return
    revision = _first(
        value,
        ("asset_revision", "release_revision", "revision"),
    )
    if declaration.asset_revision is not None and str(revision) != declaration.asset_revision:
        issues.error(
            "audio_assets_catalog_revision",
            "asset catalog revision and config/VIEW/index Inconsistent binding",
        )
    catalog_total = _first(
        value,
        (
            "total_payload_bytes",
            "payload_bytes",
            "statistics.total_payload_bytes",
        ),
    )
    if (
        declaration.total_payload_bytes is not None
        and catalog_total != declaration.total_payload_bytes
    ):
        issues.error(
            "audio_assets_catalog_total_bytes",
            "asset catalog total_payload_bytes is inconsistent with binding",
        )
    catalog_assets = {
        asset_id: (asset_path, asset_sha, asset_size)
        for asset_id, asset_path, asset_sha, asset_size in cached_assets
    }
    for asset_id, asset in observed.assets.items():
        expected = catalog_assets.get(asset_id)
        if expected is None:
            issues.error(
                "audio_assets_catalog_membership",
                f"Manifest asset_id is not in catalog: {asset_id}",
            )
            continue
        catalog_path, catalog_sha, catalog_size = expected
        if (
            asset.path != catalog_path
            or asset.shard_sha256 != catalog_sha
            or (asset.size_bytes is not None and asset.size_bytes != catalog_size)
        ):
            issues.error(
                "audio_assets_catalog_binding",
                f"Manifest and catalog shard Inconsistent identity: {asset_id}",
            )


def _load_json_mapping(
    path: Path,
    *,
    label: str,
    issues: _Issues,
) -> Mapping[str, Any] | None:
    if not path.is_file():
        issues.error("audio_assets_view_missing", f"{label} does not exist: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        issues.error(
            "audio_assets_view_json",
            f"{label} cannot be parsed: {path}: {exc}",
        )
        return None
    if not isinstance(value, Mapping):
        issues.error("audio_assets_view_json", f"{label} must be an object")
        return None
    return value


def _validate_manifest_audio_asset_bindings(
    observed: _ManifestAudioAssets,
    summary: Mapping[str, Any],
    *,
    strict: bool,
    config: Mapping[str, Any] | None,
    config_path: Path | None,
    index_audit: _IndexAudit | None,
    issues: _Issues,
) -> None:
    index_value = (
        index_audit.metadata.get("audio_assets")
        if index_audit is not None and index_audit.metadata is not None
        else None
    )
    if index_audit is not None:
        if not isinstance(index_value, Mapping):
            issues.error(
                "index_audio_assets_missing",
                "permanent-asset manifest v3 index is missing an audio_assets binding",
            )
        else:
            for key, expected in summary.items():
                if index_value.get(key) != expected:
                    issues.error(
                        "index_audio_assets_mismatch",
                        f"index audio_assets.{key} does not match the manifest-derived value",
                    )

    if not strict:
        return
    assert config is not None
    config_declarations = _audio_assets_declarations(
        config,
        label="config",
        context_path=config_path,
        issues=issues,
        require_catalog_path=True,
    )
    if not config_declarations:
        issues.error(
            "config_audio_assets_missing",
            "strict permanent-asset config must declare an audio_assets binding",
        )
    config_audio_copied = _first(
        config,
        (
            "audio_copied",
            "data.audio_copied",
            "lineage.audio_copied",
            "audio_assets.audio_copied",
            "data.audio_assets.audio_copied",
        ),
    )
    if config_audio_copied is not True:
        issues.error(
            "config_audio_copied",
            "permanent asset config must be explicitly declared audio_copied=true",
        )

    descriptor_path = _descriptor_path(config, config_path=config_path)
    descriptor = (
        _load_json_mapping(
            descriptor_path,
            label="VIEW descriptor",
            issues=issues,
        )
        if descriptor_path is not None
        else None
    )
    if descriptor_path is None:
        issues.error(
            "audio_assets_view_missing",
            "strict permanent-asset config must bind a VIEW through ready_descriptor",
        )
    view_declarations = _audio_assets_declarations(
        descriptor,
        label="VIEW",
        context_path=descriptor_path,
        issues=issues,
        require_catalog_path=False,
    )
    if descriptor is not None and not view_declarations:
        issues.error(
            "view_audio_assets_missing",
            "permanent asset VIEW must declare audio_assets binding",
        )
    if descriptor is not None:
        view_audio_copied = _first(
            descriptor,
            ("audio_copied", "audio_assets.audio_copied"),
        )
        if view_audio_copied is not True:
            issues.error(
                "view_audio_copied",
                "permanent asset VIEW must be explicitly declared audio_copied=true",
            )

    index_declarations = (
        [
            _parse_audio_assets_declaration(
                index_value,
                label="index.audio_assets",
                context_path=None,
                issues=issues,
                require_catalog_path=False,
            )
        ]
        if isinstance(index_value, Mapping)
        else []
    )
    declarations = [
        *config_declarations,
        *view_declarations,
        *index_declarations,
    ]
    if declarations:
        expected_identity = declarations[0].identity()
        for declaration in declarations[1:]:
            if declaration.identity() != expected_identity:
                issues.error(
                    "audio_assets_cross_binding",
                    "catalog, config, VIEW, and index audio_assets identities differ",
                )
        if declarations[0].asset_revision != summary.get("asset_revision"):
            issues.error(
                "audio_assets_manifest_revision",
                "audio.asset_revision conflicts with the config, VIEW, or index binding",
            )
        catalog_declaration = next(
            (
                declaration
                for declaration in config_declarations
                if declaration.catalog_path is not None
            ),
            None,
        )
        if catalog_declaration is not None:
            _validate_audio_asset_catalog(
                catalog_declaration,
                observed,
                issues,
            )


def _validate_config(
    config: Mapping[str, Any],
    *,
    config_path: Path | None,
    manifest: Path,
    actual_manifest_sha: str,
    issues: _Issues,
) -> tuple[int, float, bool, Path | None, CharacterTokenizer | None]:
    try:
        raw_stage = config["stage"]
        if isinstance(raw_stage, bool):
            raise ValueError
        stage = int(raw_stage)
    except (KeyError, TypeError, ValueError):
        issues.error("config_stage", "config.stage must be in [1, 4]")
        stage = 0
    if stage not in (1, 2, 3, 4):
        issues.error("config_stage", f"config.stage is invalid: {stage}")
    data = config.get("data")
    if not isinstance(data, Mapping):
        issues.error("config_data", "config.data must be an object")
        data = {}
    configured_manifest = data.get("manifest")
    if configured_manifest is None:
        issues.error("config_manifest", "config.data.manifest is missing")
    elif _resolve_path(str(configured_manifest), config_path=config_path) != manifest:
        issues.error(
            "config_manifest_path",
            "CLI manifest does not match the expanded config.data.manifest",
        )
    try:
        raw_max_duration = data["max_duration_sec"]
        if isinstance(raw_max_duration, bool):
            raise ValueError
        max_duration = float(raw_max_duration)
        if not math.isfinite(max_duration) or max_duration <= 0:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        issues.error(
            "config_max_duration", "config.data.max_duration_sec must be a positive finite number"
        )
        max_duration = 0.0
    random_crop = data.get("random_crop")
    if not isinstance(random_crop, bool):
        issues.error("config_random_crop", "config.data.random_crop must be explicitly bool")
        random_crop = bool(random_crop)

    semantic_value = config.get("semantic_contract")
    semantic = semantic_value if isinstance(semantic_value, Mapping) else {}
    if not isinstance(semantic_value, Mapping):
        issues.error("config_semantic", "config.semantic_contract must be an object")
    try:
        semantic_frame_rate = float(semantic.get("frame_rate", FRAME_RATE))
    except (TypeError, ValueError):
        semantic_frame_rate = math.nan
    if semantic_frame_rate != FRAME_RATE:
        issues.error("semantic_frame_rate", f"Semantic frame rate must remain {FRAME_RATE} Hz")
    try:
        semantic_vocab = int(semantic.get("codebook_size", CODEBOOK_SIZE))
    except (TypeError, ValueError):
        semantic_vocab = -1
    if semantic_vocab != CODEBOOK_SIZE:
        issues.error("semantic_vocab", f"Semantic vocab must remain {CODEBOOK_SIZE}")
    features_value = config.get("features")
    features = features_value if isinstance(features_value, Mapping) else {}
    if not isinstance(features_value, Mapping):
        issues.error("config_features", "config.features must be an object")
    try:
        sample_rate = int(features["sample_rate"])
        hop_length = int(features["hop_length"])
    except (KeyError, TypeError, ValueError):
        sample_rate = hop_length = -1
    if sample_rate != SAMPLE_RATE:
        issues.error(
            "feature_sample_rate",
            f"Tokenizer input sample rate must remain {SAMPLE_RATE}",
        )
    if hop_length <= 0 or sample_rate != 100 * hop_length:
        issues.error(
            "feature_frame_rate",
            "features.hop_length must produce 100 Hz features so 4x subsampling yields 25 Hz",
        )

    lineage = config.get("lineage") or {}
    if not isinstance(lineage, Mapping):
        issues.error("config_lineage", "config.lineage must be an object")
        lineage = {}
    manifest_source = lineage.get("manifest_source")
    if (
        manifest_source is not None
        and _resolve_path(str(manifest_source), config_path=config_path) != manifest
    ):
        issues.error(
            "lineage_manifest_path",
            "lineage.manifest_source does not match data.manifest",
        )
    expected_manifest_sha = _expected_sha(lineage, ("manifest_sha256",))
    if not expected_manifest_sha:
        issues.error(
            "lineage_manifest_sha_missing",
            "strict config must declare lineage.manifest_sha256",
        )
    elif expected_manifest_sha != actual_manifest_sha:
        issues.error(
            "lineage_manifest_sha",
            "lineage.manifest_sha256 does not match the manifest file",
        )

    tokenizer: CharacterTokenizer | None = None
    vocab_path: Path | None = None
    if data.get("vocab") is None:
        issues.error("config_vocab", "config.data.vocab is missing")
    else:
        vocab_path = _resolve_path(str(data["vocab"]), config_path=config_path)
        try:
            tokenizer = CharacterTokenizer.from_file(vocab_path)
        except Exception as exc:  # noqa: BLE001
            issues.error("config_vocab", f"frozen character tokenizer cannot be loaded: {exc}")
    if vocab_path is not None and vocab_path.is_file():
        actual_vocab_sha = sha256_file(vocab_path)
        expected_vocab_sha = _expected_sha(
            lineage,
            ("vocab_sha256", "ctc_vocab_sha256", "ctc_subword_vocab_sha256"),
        )
        if not expected_vocab_sha:
            issues.error(
                "lineage_vocab_sha_missing",
                "strict config must declare the frozen CTC vocabulary SHA-256",
            )
        elif expected_vocab_sha != actual_vocab_sha:
            issues.error("lineage_vocab_sha", "lineage CTC vocabulary SHA-256 does not match")
        if tokenizer is not None:
            heads_value = config.get("heads")
            heads = heads_value if isinstance(heads_value, Mapping) else {}
            if heads_value is not None and not isinstance(heads_value, Mapping):
                issues.error("config_heads", "config.heads must be an object")
            try:
                expected_size = int(heads.get("ctc_vocab_size", len(tokenizer)))
            except (TypeError, ValueError):
                expected_size = -1
            if expected_size != len(tokenizer):
                issues.error(
                    "config_vocab_size",
                    f"heads.ctc_vocab_size={expected_size}, actual vocabulary size={len(tokenizer)}",
                )
            if tokenizer.tokenizer_json is not None:
                expected_backend_sha = _expected_sha(
                    lineage,
                    ("tokenizer_json_sha256", "ctc_tokenizer_json_sha256"),
                )
                if not expected_backend_sha:
                    issues.error(
                        "lineage_tokenizer_json_sha_missing",
                        "subword CharacterTokenizer must be bound to tokenizer.json SHA",
                    )
                else:
                    actual_backend_sha = sha256_file(tokenizer.tokenizer_json)
                    if expected_backend_sha != actual_backend_sha:
                        issues.error(
                            "lineage_tokenizer_json_sha",
                            "lineage subword tokenizer JSON SHA-256 does not match",
                        )
    return stage, max_duration, random_crop, vocab_path, tokenizer


def _validate_intervals(
    record: Mapping[str, Any],
    *,
    duration: float,
    units: list[str | int] | None,
    issues: _Issues,
    line: int,
    sample_id: str,
) -> bool:
    intervals = record.get("ctc_unit_intervals")
    if intervals is None:
        return False
    if not isinstance(intervals, list) or not intervals:
        issues.error(
            "ctc_intervals",
            "ctc_unit_intervals must be a non-empty array",
            line=line,
            sample_id=sample_id,
        )
        return False
    if units is not None and len(intervals) != len(units):
        issues.error(
            "ctc_intervals_length",
            "ctc_unit_intervals and ctc_units have different lengths",
            line=line,
            sample_id=sample_id,
        )
    previous_end = 0.0
    for interval in intervals:
        if not isinstance(interval, Mapping):
            issues.error(
                "ctc_intervals",
                "CTC interval must be an object",
                line=line,
                sample_id=sample_id,
            )
            continue
        try:
            start = float(interval["start_sec"])
            end = float(interval["end_sec"])
        except (KeyError, TypeError, ValueError):
            start = end = math.nan
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or end > duration + 1e-6
            or start < previous_end - 1e-6
        ):
            issues.error(
                "ctc_interval_bounds",
                f"invalid CTC interval: start={start} end={end} duration={duration}",
                line=line,
                sample_id=sample_id,
            )
        previous_end = max(previous_end, end)
    return True


def _validate_sampler_coverage(
    config: Mapping[str, Any],
    observed_groups: Mapping[str, set[str]],
    ctc_groups: set[str],
    issues: _Issues,
) -> None:
    data_value = config.get("data")
    data = data_value if isinstance(data_value, Mapping) else {}
    balanced = data.get("balanced_sampler")
    if not isinstance(balanced, Mapping):
        issues.error(
            "sampler_missing",
            "strict verification requires config.data.balanced_sampler",
        )
        return
    balance_key = str(balanced.get("balance_key", "training.sampling_group"))
    if balance_key == "audio.io_group":
        balance_key = "training.io_group"
    if balance_key not in observed_groups:
        issues.error("sampler_balance_key", f"Unknown balance_key: {balance_key}")
    else:
        weights = balanced.get("weights") or {}
        if not isinstance(weights, Mapping):
            issues.error("sampler_weights", "balanced_sampler.weights must be an object")
            weights = {}
        default = balanced.get("default_weight", 0.0)
        if not _finite_nonnegative(default):
            issues.error("sampler_weight_invalid", "balanced_sampler.default_weight is invalid")
            default = 0.0
        for name, value in weights.items():
            if not _finite_nonnegative(value):
                issues.error(
                    "sampler_weight_invalid",
                    f"balanced_sampler.weights.{name} must be a non-negative finite number",
                )
        missing = {
            value
            for value in observed_groups.get(balance_key, set())
            if value not in weights and float(default) <= 0.0
        }
        if missing:
            issues.error(
                "sampler_group_uncovered",
                f"{balance_key} There is a group that has not been sampler covers: {sorted(missing)[:16]}",
            )

    ctc_weights = data.get("ctc_group_weights") or {}
    ctc_default = data.get("ctc_group_default_weight", 0.0)
    if not isinstance(ctc_weights, Mapping):
        issues.error("ctc_sampler_weights", "data.ctc_group_weights must be an object")
        ctc_weights = {}
    if not _finite_nonnegative(ctc_default):
        issues.error("ctc_sampler_weight_invalid", "ctc_group_default_weight is invalid")
        ctc_default = 0.0
    for name, value in ctc_weights.items():
        if not _finite_nonnegative(value):
            issues.error(
                "ctc_sampler_weight_invalid",
                f"ctc_group_weights.{name} must be a non-negative finite number",
            )
    missing_ctc = {
        value for value in ctc_groups if value not in ctc_weights and float(ctc_default) <= 0.0
    }
    if missing_ctc:
        issues.error(
            "ctc_group_uncovered",
            f"CTC group is not covered by the weight configuration: {sorted(missing_ctc)[:16]}",
        )

    locality_key = balanced.get("locality_key")
    if locality_key not in {"training.io_group", "audio.io_group"}:
        issues.error(
            "io_group_uncovered",
            "balanced_sampler.locality_key must override training/audio.io_group",
        )

    aliases = {
        "training.quality_tier": ("quality_tier_weights", "tier_weights"),
        "training.source_sampling_group": ("source_sampling_group_weights",),
        "training.sampling_group": ("sampling_group_weights",),
        "training.io_group": ("io_group_weights",),
    }
    for field, names in aliases.items():
        table = next((data.get(name) for name in names if data.get(name) is not None), None)
        if table is None:
            continue
        if not isinstance(table, Mapping):
            issues.error("sampler_weights", f"{names[0]} must be an object")
            continue
        for key, value in table.items():
            if not _finite_nonnegative(value):
                issues.error(
                    "sampler_weight_invalid",
                    f"{names[0]}.{key} must be a non-negative finite number",
                )
        missing = observed_groups[field] - {str(key) for key in table}
        if missing:
            issues.error(
                "sampler_group_uncovered",
                f"{field} is not covered by the independent weight table: {sorted(missing)[:16]}",
            )


def _dimension_report(
    counts: Mapping[str, int],
    durations: Mapping[str, float],
) -> dict[str, Any]:
    return {
        key: {
            "records": int(counts[key]),
            "hours": round(float(durations.get(key, 0.0)) / 3600.0, 6),
        }
        for key in sorted(counts)
    }


def _decode_candidates(
    candidates: list[_DecodeCandidate],
    issues: _Issues,
) -> dict[str, Any]:
    by_storage: Counter[str] = Counter()
    succeeded = 0
    failed = 0
    samples: list[dict[str, Any]] = []
    for candidate in candidates:
        status = "ok"
        detail = ""
        credential_present = False
        try:
            password = None
            if candidate.password_env is not None:
                password = password_from_env(candidate.password_env)
                credential_present = True
            waveform, sample_rate = load_audio(
                candidate.uri,
                start_sec=candidate.start_sec,
                duration_sec=min(1.0, candidate.duration_sec),
                archive_offset=candidate.archive_offset,
                archive_size=candidate.archive_size,
                password=password,
                asset_id=candidate.asset_id,
                payload_sha256=candidate.payload_sha256,
                asset_revision=candidate.asset_revision,
                shard_sha256=candidate.shard_sha256,
            )
            if waveform.numel() == 0 or sample_rate <= 0:
                raise ValueError("Decoded audio is empty or has an invalid sample rate")
            succeeded += 1
            by_storage[candidate.storage] += 1
        except AudioCredentialError as exc:
            failed += 1
            status = "error"
            detail = f"{type(exc).__name__}: {exc}"
            issues.error(
                "credential_decode_failed",
                f"Encryption with deterministic sampling ZIP Credentials/Reading failed: {detail}",
                sample_id=candidate.sample_id,
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            status = "error"
            detail = f"{type(exc).__name__}: {exc}"
            issues.error(
                "decode_failed",
                f"Deterministic sampling decoding failed: {detail}",
                sample_id=candidate.sample_id,
            )
        samples.append(
            {
                "sample_id": candidate.sample_id,
                "storage": candidate.storage,
                "dataset": candidate.dataset,
                "tier": candidate.tier,
                "split": candidate.split,
                "status": status,
                "detail": detail,
                "password_env": candidate.password_env,
                "credential_present": credential_present,
            }
        )
    return {
        "attempted": len(candidates),
        "succeeded": succeeded,
        "failed": failed,
        "by_storage_succeeded": dict(sorted(by_storage.items())),
        "samples": samples,
    }


def _resolve_index_dir(
    manifest: Path,
    *,
    index: str | Path | None,
    config: Mapping[str, Any] | None,
    config_path: Path | None,
) -> Path | None:
    value: str | Path | None = index
    if value is None and config is not None:
        value = _first(config, ("data.index", "data.manifest_index"))
    if value is not None:
        path = _resolve_path(value, config_path=config_path)
        return path.parent if path.name == "metadata.json" else path
    adjacent = Path(str(manifest.resolve()) + ".index")
    return adjacent if adjacent.exists() else None


def validate_tokenizer_manifest(
    manifest: str | Path,
    *,
    config: str | Path | Mapping[str, Any] | None = None,
    index: str | Path | None = None,
    output: str | Path | None = None,
    decode_samples: int = 16,
    ctc_vocab: str | Path | None = None,
    ctc_frame_rate: float = FRAME_RATE,
) -> dict[str, Any]:

    manifest_path = Path(manifest).expanduser().resolve()
    issues = _Issues()
    config_path: Path | None = None
    expanded_config: Mapping[str, Any] | None
    if isinstance(config, Mapping):
        expanded_config = config
    elif config is not None:
        config_path = Path(config).expanduser().resolve()
        try:
            expanded_config = load_config(config_path)
        except Exception as exc:  # noqa: BLE001
            expanded_config = None
            issues.error("config_load", f"Tokenizer config Expansion failed: {exc}")
    else:
        expanded_config = None
    strict = expanded_config is not None
    if not manifest_path.is_file():
        issues.error("manifest_missing", f"Manifest does not exist: {manifest_path}")
        report = _final_report(
            manifest_path,
            issues,
            strict=strict,
            statistics={},
            index_report=None,
            config_report=None,
        )
        _write_report(report, output)
        return report

    actual_manifest_sha = sha256_file(manifest_path)
    stage = 0
    max_duration = math.inf
    random_crop = False
    tokenizer: CharacterTokenizer | None = None
    vocab_path: Path | None = None
    if expanded_config is not None:
        stage, max_duration, random_crop, vocab_path, tokenizer = _validate_config(
            expanded_config,
            config_path=config_path,
            manifest=manifest_path,
            actual_manifest_sha=actual_manifest_sha,
            issues=issues,
        )
    elif ctc_vocab is not None:
        vocab_path = Path(ctc_vocab).expanduser().resolve()
        try:
            tokenizer = CharacterTokenizer.from_file(vocab_path)
        except Exception as exc:  # noqa: BLE001
            issues.error("ctc_vocab", f"CTC vocab cannot be loaded: {exc}")
    if strict and not math.isclose(float(ctc_frame_rate), FRAME_RATE):
        issues.error("ctc_frame_rate", f"Strict configuration must use {FRAME_RATE}Hz")

    index_dir = _resolve_index_dir(
        manifest_path,
        index=index,
        config=expanded_config,
        config_path=config_path,
    )
    index_audit: _IndexAudit | None = None
    expected_index_sha = None
    if expanded_config is not None:
        lineage_value = expanded_config.get("lineage")
        lineage = lineage_value if isinstance(lineage_value, Mapping) else {}
        expected_index_sha = _expected_sha(
            lineage,
            ("manifest_index_sha256", "index_sha256", "index_metadata_sha256"),
        )
        if not expected_index_sha:
            issues.error(
                "lineage_index_sha_missing",
                "strict config must declare Manifest index metadata SHA",
            )
    if index_dir is None:
        if strict:
            issues.error(
                "index_missing", "strict config Verification requirements v3 READY Manifest index"
            )
        else:
            issues.warning("index_missing", "No adjacent manifest index was found")
    else:
        index_audit = _IndexAudit(
            index_dir,
            manifest_path,
            issues,
            expected_metadata_sha256=expected_index_sha,
        )

    sample_ids: set[str] = set()
    revisions: set[str] = set()
    links: dict[str, _RecordLink] = {}
    aliases: dict[str, str] = {}
    derived_links: list[_DerivedLink] = []
    group_splits: dict[str, set[str]] = defaultdict(set)
    parent_splits: dict[str, set[str]] = defaultdict(set)
    content_splits: dict[str, set[str]] = defaultdict(set)
    observed_groups = {field: set() for field in INDEX_CATEGORICAL_FIELDS}
    ctc_groups: set[str] = set()
    count_by: dict[str, Counter[str]] = {
        name: Counter() for name in ("storage", "source", "split", "tier", "head")
    }
    duration_by: dict[str, Counter[str]] = {
        name: Counter() for name in ("storage", "source", "split", "tier", "head")
    }
    credential_states: dict[str, bool] = {}
    credential_fatal = False
    decode_sampler = _DecodeSampler(decode_samples)
    audio_assets_audit = _ManifestAudioAssets(issues)
    records = 0
    total_duration = 0.0
    target_stage = f"tokenizer_s{stage}" if stage else ""
    license_view = _license_view(expanded_config or {})

    with manifest_path.open("rb") as handle:
        physical_line = 0
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            physical_line += 1
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                issues.error(
                    "manifest_json",
                    f"JSONL Parsing failed: {exc}",
                    line=physical_line,
                )
                continue
            if not isinstance(record, Mapping):
                issues.error(
                    "manifest_record",
                    "Manifest Each line must be JSON object",
                    line=physical_line,
                )
                continue
            ordinal = records
            records += 1
            sample_id = str(record.get("sample_id") or "")
            if record.get("schema_version") != MANIFEST_SCHEMA:
                issues.error(
                    "manifest_schema",
                    f"schema_version must be {MANIFEST_SCHEMA}",
                    line=physical_line,
                    sample_id=sample_id or None,
                )
            if not sample_id:
                issues.error("sample_id_missing", "is missing sample_id", line=physical_line)
                sample_id = f"<line-{physical_line}>"
            elif sample_id in sample_ids:
                issues.error(
                    "sample_id_duplicate",
                    "sample_id Repeat",
                    line=physical_line,
                    sample_id=sample_id,
                )
            sample_ids.add(sample_id)

            revision = _revision(record)
            if revision.strip().casefold() not in {
                "",
                "unknown",
                "none",
                "null",
            }:
                revisions.add(revision)
            elif strict:
                issues.error(
                    "corpus_revision_missing",
                    "Every record in a strict manifest must bind to a corpus revision",
                    line=physical_line,
                    sample_id=sample_id,
                )

            training = record.get("training")
            if not isinstance(training, Mapping):
                issues.error(
                    "training_missing",
                    "training must be an object",
                    line=physical_line,
                    sample_id=sample_id,
                )
                training = {}
            eligible = training.get("eligible_stages")
            if strict and (
                not isinstance(eligible, list)
                or target_stage not in {str(item) for item in eligible}
            ):
                issues.error(
                    "stage_ineligible",
                    f"training.eligible_stages does not contain {target_stage}",
                    line=physical_line,
                    sample_id=sample_id,
                )
            heads = training.get("loss_heads")
            if not isinstance(heads, Mapping):
                issues.error(
                    "loss_heads_missing",
                    "training.loss_heads must be an object",
                    line=physical_line,
                    sample_id=sample_id,
                )
                heads = {}
            if strict:
                missing_heads = sorted(set(LOSS_HEADS) - set(heads))
                if missing_heads:
                    issues.error(
                        "loss_heads_incomplete",
                        f"loss_heads missing field: {missing_heads}",
                        line=physical_line,
                        sample_id=sample_id,
                    )
            if any(not isinstance(value, bool) for value in heads.values()):
                issues.error(
                    "loss_head_type",
                    "loss_heads must be bool",
                    line=physical_line,
                    sample_id=sample_id,
                )
            if not any(value is True for value in heads.values()):
                issues.error(
                    "loss_heads_empty",
                    "At least one loss head must be enabled",
                    line=physical_line,
                    sample_id=sample_id,
                )
            if strict and stage in (1, 2) and heads.get("bestrq") is not True:
                issues.error(
                    "bestrq_declaration",
                    f"Stage {stage} records must declare loss_heads.bestrq=true",
                    line=physical_line,
                    sample_id=sample_id,
                )
            for field, value in _record_weights(training):
                if not _finite_nonnegative(value):
                    issues.error(
                        "record_weight_invalid",
                        f"{field} must be a non-negative finite number; received {value!r}",
                        line=physical_line,
                        sample_id=sample_id,
                    )

            for field in INDEX_CATEGORICAL_FIELDS:
                value = _group_value(record, field)
                observed_groups[field].add(value)
                normalized_group = value.strip().casefold()
                if strict and (
                    normalized_group in _UNKNOWN_VALUES
                    or (normalized_group == "none" and field != "training.ctc_group")
                ):
                    issues.error(
                        "group_missing",
                        f"{field} must not be empty/unknown",
                        line=physical_line,
                        sample_id=sample_id,
                    )
            if not strict and _nested(record, "training.sampling_group") is None:
                issues.warning(
                    "missing_sampling_group",
                    "is missing training.sampling_group",
                    line=physical_line,
                    sample_id=sample_id,
                )

            audio = record.get("audio")
            if not isinstance(audio, Mapping):
                issues.error(
                    "audio_missing",
                    "audio must be an object",
                    line=physical_line,
                    sample_id=sample_id,
                )
                audio = {}
            try:
                raw_start = audio.get("start_sec", 0.0)
                raw_duration = audio["duration_sec"]
                if isinstance(raw_start, bool) or isinstance(raw_duration, bool):
                    raise ValueError
                start = float(raw_start)
                duration = float(raw_duration)
            except (KeyError, TypeError, ValueError):
                start = duration = math.nan
            if not math.isfinite(start) or start < 0:
                issues.error(
                    "audio_start",
                    f"audio.start_sec must be a non-negative finite number; received {start}",
                    line=physical_line,
                    sample_id=sample_id,
                )
            if not math.isfinite(duration) or duration <= 0:
                issues.error(
                    "audio_duration",
                    f"audio.duration_sec must be a positive finite number; received {duration}",
                    line=physical_line,
                    sample_id=sample_id,
                )
                duration = 0.0
            try:
                uri = resolve_audio_uri(record, manifest_path.parent)
            except (KeyError, TypeError, ValueError) as exc:
                uri = ""
                issues.error(
                    "audio_uri",
                    f"audio URI is invalid: {exc}",
                    line=physical_line,
                    sample_id=sample_id,
                )
            storage = _storage_class(uri) if uri else "unknown"
            password_env: str | None = None
            credential_present = False
            try:
                password_env = audio_password_env(record)
                if password_env is not None:
                    if storage != "zip":
                        raise AudioCredentialError(
                            f"Audio credential environment variable {password_env} can be used only with a zip:// audio URI"
                        )
                    try:
                        password_from_env(password_env)
                    except AudioCredentialError as exc:
                        credential_states[password_env] = False
                        issues.error(
                            "credential_missing",
                            str(exc),
                            line=physical_line,
                            sample_id=sample_id,
                        )
                        credential_fatal = True
                        break
                    else:
                        credential_present = True
                        credential_states[password_env] = True
            except AudioCredentialError as exc:
                issues.error(
                    "audio_password_env",
                    str(exc),
                    line=physical_line,
                    sample_id=sample_id,
                )
                credential_fatal = True
                break
            archive_offset: int | None = None
            archive_size: int | None = None
            raw_archive_offset = audio.get("archive_offset")
            raw_archive_size = audio.get("archive_size")
            if (raw_archive_offset is None) != (raw_archive_size is None):
                issues.error(
                    "audio_archive_range",
                    "audio.archive_offset/archive_size must be declared in pairs",
                    line=physical_line,
                    sample_id=sample_id,
                )
            elif raw_archive_offset is not None:
                if (
                    isinstance(raw_archive_offset, bool)
                    or isinstance(raw_archive_size, bool)
                    or not isinstance(raw_archive_offset, int)
                    or not isinstance(raw_archive_size, int)
                    or raw_archive_offset < 0
                    or raw_archive_size <= 0
                ):
                    issues.error(
                        "audio_archive_range",
                        "audio.archive_offset must be non-negative and archive_size must be a positive integer",
                        line=physical_line,
                        sample_id=sample_id,
                    )
                else:
                    archive_offset = raw_archive_offset
                    archive_size = raw_archive_size
            if archive_offset is not None and storage != "tar":
                issues.error(
                    "audio_archive_storage",
                    "audio.archive_offset/archive_size can only be used with tar:// URI",
                    line=physical_line,
                    sample_id=sample_id,
                )

            asset_values = {
                field: audio.get(field)
                for field in (
                    "asset_id",
                    "payload_sha256",
                    "asset_revision",
                    "shard_sha256",
                )
            }
            has_asset_fields = any(value is not None for value in asset_values.values())
            valid_asset_fields = has_asset_fields
            if has_asset_fields and not all(value is not None for value in asset_values.values()):
                valid_asset_fields = False
                issues.error(
                    "audio_asset_fields",
                    "Permanent assets must also be declared audio.asset_id/payload_sha256/"
                    "asset_revision/shard_sha256",
                    line=physical_line,
                    sample_id=sample_id,
                )
            if has_asset_fields:
                for field in ("asset_id", "asset_revision"):
                    value = asset_values[field]
                    if (
                        not isinstance(value, str)
                        or not value
                        or value != value.strip()
                        or len(value) > 1024
                    ):
                        valid_asset_fields = False
                        issues.error(
                            "audio_asset_field",
                            f"audio.{field} must be a non-empty specification string",
                            line=physical_line,
                            sample_id=sample_id,
                        )
                for field in ("payload_sha256", "shard_sha256"):
                    if not _is_sha256(asset_values[field]):
                        valid_asset_fields = False
                        issues.error(
                            "audio_asset_sha",
                            f"audio.{field} must be 64 lowercase hexadecimal SHA-256",
                            line=physical_line,
                            sample_id=sample_id,
                        )
                if storage != "tar":
                    valid_asset_fields = False
                    issues.error(
                        "audio_asset_storage",
                        "Permanent audio assets must use tar:// URI",
                        line=physical_line,
                        sample_id=sample_id,
                    )
                if archive_offset is None or archive_size is None:
                    valid_asset_fields = False
                    issues.error(
                        "audio_asset_archive_range",
                        "Permanent audio assets must be declared valid archive_offset/archive_size",
                        line=physical_line,
                        sample_id=sample_id,
                    )
            if (
                valid_asset_fields
                and uri
                and archive_offset is not None
                and archive_size is not None
            ):
                asset_source_value = record.get("source")
                asset_source = asset_source_value if isinstance(asset_source_value, Mapping) else {}
                locator_value = asset_source.get("locator_overlay")
                locator = locator_value if isinstance(locator_value, Mapping) else {}
                locator_catalog_sha = locator.get("catalog_sha256")
                locator_content_manifest_sha = locator.get("content_manifest_sha256")
                audio_assets_audit.observe(
                    asset_id=str(asset_values["asset_id"]),
                    asset_revision=str(asset_values["asset_revision"]),
                    shard_sha256=str(asset_values["shard_sha256"]),
                    payload_sha256=str(asset_values["payload_sha256"]),
                    uri=uri,
                    archive_offset=archive_offset,
                    archive_size=archive_size,
                    catalog_sha256=(
                        str(locator_catalog_sha) if _is_sha256(locator_catalog_sha) else None
                    ),
                    content_manifest_sha256=(
                        str(locator_content_manifest_sha)
                        if _is_sha256(locator_content_manifest_sha)
                        else None
                    ),
                    line=physical_line,
                    sample_id=sample_id,
                )

            if strict:
                declared_max = training.get("max_duration_sec")
                if not _finite_nonnegative(declared_max) or not math.isclose(
                    float(declared_max), max_duration, rel_tol=0.0, abs_tol=1e-6
                ):
                    issues.error(
                        "record_max_duration_mismatch",
                        "training.max_duration_sec does not match config.data.max_duration_sec",
                        line=physical_line,
                        sample_id=sample_id,
                    )
                if training.get("random_crop") is not random_crop:
                    issues.error(
                        "record_random_crop_mismatch",
                        "training.random_crop does not match config.data.random_crop",
                        line=physical_line,
                        sample_id=sample_id,
                    )

            split = str(record.get("split", "train"))
            source = record.get("source") or {}
            if not isinstance(source, Mapping):
                source = {}
                issues.error(
                    "source_type",
                    "source must be an object",
                    line=physical_line,
                    sample_id=sample_id,
                )
            dataset = str(source.get("dataset") or source.get("dataset_key") or "unknown")
            tier = _group_value(record, "training.quality_tier")
            group_id = str(source.get("group_id") or "")
            license_id = str(source.get("license_id") or "UNKNOWN")
            commercial_ok = _commercial_ok(record)
            if strict and not group_id.strip():
                issues.error(
                    "group_id_missing",
                    "source.group_id must not be empty",
                    line=physical_line,
                    sample_id=sample_id,
                )
            if group_id:
                group_splits[group_id].add(split)
            content_hash = _content_hash(record)
            if strict and not content_hash.strip():
                issues.error(
                    "content_hash_missing",
                    "is missing canonical content hash",
                    line=physical_line,
                    sample_id=sample_id,
                )
            if content_hash:
                content_splits[content_hash].add(split)
            canonical = _is_canonical(record)
            if strict and canonical is not True:
                issues.error(
                    "canonical_required",
                    "training records must be explicitly canonical",
                    line=physical_line,
                    sample_id=sample_id,
                )
            license_unknown = license_id.strip().casefold() in {
                "",
                "unknown",
                "none",
                "null",
            }
            if license_unknown:
                if license_view == "research-max":
                    issues.warning(
                        "license_unknown_research",
                        "research-max includes an unknown license; review usage rights",
                        line=physical_line,
                        sample_id=sample_id,
                    )
                elif license_view == "license-audited":
                    issues.warning(
                        "license_unknown_audited",
                        "license-audited relies on upstream commercial_ok, but an unknown "
                        "license still requires rights review",
                        line=physical_line,
                        sample_id=sample_id,
                    )
                elif strict:
                    issues.error(
                        "license_unknown",
                        "strict training manifest contains an unknown license",
                        line=physical_line,
                        sample_id=sample_id,
                    )
                else:
                    issues.warning(
                        "unknown_license",
                        "license_id Unknown",
                        line=physical_line,
                        sample_id=sample_id,
                    )
            elif license_view == "research-max" and commercial_ok is not True:
                issues.warning(
                    "license_restricted_research",
                    "research-max includes a record without commercial_ok=true",
                    line=physical_line,
                    sample_id=sample_id,
                )
            if license_view == "license-audited" and commercial_ok is not True:
                issues.error(
                    "license_audited_commercial_ok",
                    "license-audited View can only contain commercial_ok=true",
                    line=physical_line,
                    sample_id=sample_id,
                )
            if _is_holdout(record) and split == "train":
                issues.error(
                    "eval_holdout_in_train",
                    "eval_holdout records cannot appear in the train split",
                    line=physical_line,
                    sample_id=sample_id,
                )

            link = _RecordLink(
                sample_id=sample_id,
                split=split,
                group_id=group_id,
                license_id=license_id,
                commercial_ok=commercial_ok,
                start_sec=start if math.isfinite(start) else 0.0,
                duration_sec=duration,
            )
            links[sample_id] = link
            for alias in (
                record.get("uid_hex"),
                source.get("uid_hex"),
                record.get("canonical_uid_hex"),
            ):
                if alias is not None:
                    aliases[str(alias)] = sample_id
            if _is_derived(record):
                parent_id = _parent_id(record)
                if not parent_id:
                    issues.error(
                        "derived_parent_missing",
                        "derived section is missing its parent identifier",
                        line=physical_line,
                        sample_id=sample_id,
                    )
                else:
                    parent_splits[parent_id].add(split)
                    derived_links.append(_DerivedLink(sample_id, parent_id, link, physical_line))

            ctc_enabled = heads.get("ctc") is True
            if ctc_enabled:
                ctc_groups.add(_group_value(record, "training.ctc_group"))
                encoded: list[int] = []
                units: list[str | int] | None = None
                try:
                    encoded, units = _encode_ctc(record, tokenizer)
                except (TypeError, ValueError, RuntimeError) as exc:
                    issues.error(
                        "ctc_target",
                        str(exc),
                        line=physical_line,
                        sample_id=sample_id,
                    )
                has_intervals = _validate_intervals(
                    record,
                    duration=duration,
                    units=units,
                    issues=issues,
                    line=physical_line,
                    sample_id=sample_id,
                )
                derived = _is_derived(record)
                ctc_training_unit = training.get("ctc_training_unit")
                if ctc_training_unit == "full_track":
                    if derived or record.get("segment_kind") != "track":
                        issues.error(
                            "ctc_full_track_scope",
                            "full_track CTC must directly use the non-derived full track record",
                            line=physical_line,
                            sample_id=sample_id,
                        )
                    if record.get("target_source") not in {
                        "annotation_full_track",
                        "canonical_structured_lyrics",
                    }:
                        issues.error(
                            "ctc_full_track_target_source",
                            "full_track CTC must declare an accepted complete-track target source",
                            line=physical_line,
                            sample_id=sample_id,
                        )
                    if duration > 300.0 + 1e-6:
                        issues.error(
                            "ctc_full_track_duration",
                            "full_track CTC duration must not exceed 300 seconds",
                            line=physical_line,
                            sample_id=sample_id,
                        )
                    if random_crop:
                        issues.error(
                            "ctc_full_track_random_crop",
                            "full_track CTC requires random_crop=false",
                            line=physical_line,
                            sample_id=sample_id,
                        )
                elif ctc_training_unit == "section" and not derived:
                    issues.error(
                        "ctc_section_scope",
                        "section CTC must use an explicitly derived segment",
                        line=physical_line,
                        sample_id=sample_id,
                    )
                elif ctc_training_unit not in {None, "full_track", "section"}:
                    issues.error(
                        "ctc_training_unit",
                        f"Unsupported CTC training unit: {ctc_training_unit!r}",
                        line=physical_line,
                        sample_id=sample_id,
                    )
                if strict and duration > max_duration + 1e-6:
                    if not has_intervals and not derived:
                        issues.error(
                            "ctc_crop_mismatch",
                            "CTC audio would be cropped without an interval-aligned or derived segment; runtime disabling is not allowed",
                            line=physical_line,
                            sample_id=sample_id,
                        )
                    if random_crop and not derived:
                        issues.error(
                            "ctc_random_crop_mismatch",
                            "non-derived CTC records must not use random cropping",
                            line=physical_line,
                            sample_id=sample_id,
                        )
                if encoded:
                    effective = min(duration, max_duration) if strict else duration
                    available = (
                        _actual_output_frames(effective, expanded_config or {})
                        if strict
                        else int(effective * float(ctc_frame_rate))
                    )
                    required = _ctc_required_frames(encoded)
                    if required > available:
                        issues.error(
                            "ctc_infeasible",
                            f"CTC target requires at least {required} frames, but the "
                            f"frontend produces only {available}",
                            line=physical_line,
                            sample_id=sample_id,
                        )
                    declared_required = record.get("ctc_minimum_frames")
                    if declared_required is not None:
                        try:
                            declared_frames = int(declared_required)
                        except (TypeError, ValueError):
                            declared_frames = -1
                        if declared_frames != required:
                            issues.error(
                                "ctc_declared_frames_mismatch",
                                "ctc_minimum_frames does not match the encoded target and adjacent repeats",
                                line=physical_line,
                                sample_id=sample_id,
                            )

            if index_audit is not None:
                index_audit.observe(
                    ordinal,
                    offset,
                    record,
                    line=physical_line,
                    sample_id=sample_id,
                )
            total_duration += duration
            for dimension, value in (
                ("storage", storage),
                ("source", dataset),
                ("split", split),
                ("tier", tier),
            ):
                count_by[dimension][value] += 1
                duration_by[dimension][value] += duration
            for head in LOSS_HEADS:
                if heads.get(head) is True:
                    count_by["head"][head] += 1
                    duration_by["head"][head] += duration
            if uri and (password_env is None or credential_present):
                key = hashlib.sha256(
                    (
                        f"oqm-tokenizer-validator-v1\0{storage}\0{dataset}\0{tier}\0{sample_id}"
                    ).encode()
                ).hexdigest()
                decode_sampler.add(
                    _DecodeCandidate(
                        key=key,
                        sample_id=sample_id,
                        uri=uri,
                        storage=storage,
                        dataset=dataset,
                        tier=tier,
                        split=split,
                        start_sec=start if math.isfinite(start) else 0.0,
                        duration_sec=duration,
                        archive_offset=archive_offset,
                        archive_size=archive_size,
                        password_env=password_env,
                        asset_id=(str(asset_values["asset_id"]) if valid_asset_fields else None),
                        payload_sha256=(
                            str(asset_values["payload_sha256"]) if valid_asset_fields else None
                        ),
                        asset_revision=(
                            str(asset_values["asset_revision"]) if valid_asset_fields else None
                        ),
                        shard_sha256=(
                            str(asset_values["shard_sha256"]) if valid_asset_fields else None
                        ),
                    )
                )

    if credential_fatal:
        statistics = {
            "records": records,
            "duration_hours": round(total_duration / 3600.0, 6),
            "decode": {
                "attempted": 0,
                "succeeded": 0,
                "failed": 0,
                "by_storage_succeeded": {},
                "samples": [],
            },
            "credentials": [
                {
                    "password_env": password_env,
                    "credential_present": credential_states[password_env],
                }
                for password_env in sorted(credential_states)
            ],
            "reject": {
                "errors": issues.error_count,
                "by_code": dict(sorted(issues.error_counts.items())),
            },
        }
        report = _final_report(
            manifest_path,
            issues,
            strict=strict,
            statistics=statistics,
            index_report=None,
            config_report=None,
            manifest_sha256=actual_manifest_sha,
        )
        _write_report(report, output)
        return report

    if index_audit is not None:
        index_audit.finish(records)
    audio_assets_summary = audio_assets_audit.summary()
    if audio_assets_summary is not None:
        _validate_manifest_audio_asset_bindings(
            audio_assets_audit,
            audio_assets_summary,
            strict=strict,
            config=expanded_config,
            config_path=config_path,
            index_audit=index_audit,
            issues=issues,
        )
    elif (
        index_audit is not None
        and index_audit.metadata is not None
        and index_audit.metadata.get("audio_assets") is not None
    ):
        issues.error(
            "index_audio_assets_unexpected",
            "index declares audio_assets, but the manifest has no permanent-asset fields",
        )
    if strict and records == 0:
        issues.error("manifest_empty", "Strict training manifest must not be empty")
    if len(revisions) > 1:
        issues.error(
            "corpus_revision_mixed",
            f"Manifest contains multiple corpus revisions: {sorted(revisions)[:8]}",
        )
    if strict:
        expected_revision = _first(
            expanded_config or {},
            ("lineage.corpus_revision", "lineage.release_revision"),
        )
        if expected_revision is not None and revisions != {str(expected_revision)}:
            issues.error(
                "corpus_revision_lineage",
                f"Manifest corpus revision={sorted(revisions)}, lineage={expected_revision}",
            )

    for group, splits in group_splits.items():
        if len(splits) > 1:
            issues.error(
                "group_split_leak",
                f"source.group_id={group!r} appears across splits: {sorted(splits)}",
            )
    for parent, splits in parent_splits.items():
        if len(splits) > 1:
            issues.error(
                "parent_split_leak",
                f"parent={parent!r} appears across splits: {sorted(splits)}",
            )
    for digest, splits in content_splits.items():
        if len(splits) > 1:
            issues.error(
                "content_hash_split_leak",
                f"content hash={digest!r} appears across splits: {sorted(splits)}",
            )
    for derived in derived_links:
        parent_sample = aliases.get(derived.parent_id, derived.parent_id)
        parent = links.get(parent_sample)
        if parent is None:
            issues.error(
                "derived_parent_not_found",
                f"derived section not found parent={derived.parent_id!r}",
                line=derived.line,
                sample_id=derived.sample_id,
            )
            continue
        child = derived.child
        for field, child_value, parent_value in (
            ("group_id", child.group_id, parent.group_id),
            ("split", child.split, parent.split),
            ("license_id", child.license_id, parent.license_id),
            ("commercial_ok", child.commercial_ok, parent.commercial_ok),
        ):
            if child_value != parent_value:
                issues.error(
                    "derived_inheritance",
                    f"derived section does not inherit parent {field}: "
                    f"{child_value!r} != {parent_value!r}",
                    line=derived.line,
                    sample_id=derived.sample_id,
                )
        child_end = child.start_sec + child.duration_sec
        parent_end = parent.start_sec + parent.duration_sec
        if child.start_sec < parent.start_sec - 1e-6 or child_end > parent_end + 1e-6:
            issues.error(
                "derived_bounds",
                "derived section exceeds parent audio interval",
                line=derived.line,
                sample_id=derived.sample_id,
            )

    if strict:
        _validate_sampler_coverage(
            expanded_config or {},
            observed_groups,
            ctc_groups,
            issues,
        )
    decode_report = _decode_candidates(decode_sampler.selected(), issues)
    statistics = {
        "records": records,
        "duration_hours": round(total_duration / 3600.0, 6),
        "audio_assets": audio_assets_summary,
        "decode": decode_report,
        "credentials": [
            {
                "password_env": password_env,
                "credential_present": credential_states[password_env],
            }
            for password_env in sorted(credential_states)
        ],
        "reject": {
            "errors": issues.error_count,
            "by_code": dict(sorted(issues.error_counts.items())),
        },
        "head": _dimension_report(count_by["head"], duration_by["head"]),
        "tier": _dimension_report(count_by["tier"], duration_by["tier"]),
        "source": _dimension_report(count_by["source"], duration_by["source"]),
        "split": _dimension_report(count_by["split"], duration_by["split"]),
        "storage": _dimension_report(count_by["storage"], duration_by["storage"]),
    }
    index_report = (
        {
            "path": str(index_dir),
            "schema_version": (
                index_audit.metadata.get("schema_version")
                if index_audit and index_audit.metadata
                else None
            ),
            "metadata_sha256": (index_audit.metadata_sha256 if index_audit is not None else None),
            "ready": bool(
                index_audit is not None
                and index_audit.metadata is not None
                and not any(code.startswith("index_") for code in issues.error_counts)
            ),
        }
        if index_dir is not None
        else None
    )
    raw_config_lineage = expanded_config.get("lineage") if expanded_config is not None else None
    config_lineage = raw_config_lineage if isinstance(raw_config_lineage, Mapping) else {}
    config_report = (
        {
            "path": str(config_path) if config_path is not None else None,
            "stage": stage,
            "manifest": str(manifest_path),
            "max_duration_sec": max_duration,
            "random_crop": random_crop,
            "vocab": str(vocab_path) if vocab_path is not None else None,
            "lineage": {
                "manifest_sha256": _expected_sha(config_lineage, ("manifest_sha256",)),
                "vocab_sha256": _expected_sha(
                    config_lineage,
                    (
                        "vocab_sha256",
                        "ctc_vocab_sha256",
                        "ctc_subword_vocab_sha256",
                    ),
                ),
                "manifest_index_sha256": expected_index_sha,
                "actual_vocab_sha256": (
                    sha256_file(vocab_path)
                    if vocab_path is not None and vocab_path.is_file()
                    else None
                ),
            },
        }
        if expanded_config is not None
        else None
    )
    report = _final_report(
        manifest_path,
        issues,
        strict=strict,
        statistics=statistics,
        index_report=index_report,
        config_report=config_report,
        manifest_sha256=actual_manifest_sha,
    )
    _write_report(report, output)
    return report


def _final_report(
    manifest: Path,
    issues: _Issues,
    *,
    strict: bool,
    statistics: Mapping[str, Any],
    index_report: Mapping[str, Any] | None,
    config_report: Mapping[str, Any] | None,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    ready_reasons: list[str] = []
    if not strict:
        ready_reasons.append("a training configuration was not provided")
    if index_report is None or index_report.get("ready") is not True:
        ready_reasons.append("manifest-index v3 did not pass READY validation")
    if issues.error_count:
        ready_reasons.append(f"{issues.error_count} validation error(s)")
    ready_candidate = not ready_reasons
    decoded = statistics.get("decode") or {}
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS" if issues.error_count == 0 else "FAIL",
        "manifest": {
            "path": str(manifest),
            "sha256": manifest_sha256,
            "size_bytes": manifest.stat().st_size if manifest.is_file() else None,
        },
        "config": config_report,
        "index": index_report,
        "ready_candidate": {
            "eligible": ready_candidate,
            "decision": "READY_CANDIDATE" if ready_candidate else "NOT_READY",
            "reasons": ready_reasons,
        },
        "statistics": dict(statistics),
        "num_samples": int(statistics.get("records", 0)),
        "duration_hours": float(statistics.get("duration_hours", 0.0)),
        "decoded_samples": int(decoded.get("attempted", 0)),
        "errors": [issue.message for issue in issues.errors],
        "warnings": dict(sorted(issues.warning_counts.items())),
        "issues": {
            "error_count": issues.error_count,
            "warning_count": issues.warning_count,
            "error_counts": dict(sorted(issues.error_counts.items())),
            "warning_counts": dict(sorted(issues.warning_counts.items())),
            "errors": [issue.as_dict() for issue in issues.errors],
            "warnings": [issue.as_dict() for issue in issues.warnings],
            "details_truncated": (
                issues.error_count > len(issues.errors)
                or issues.warning_count > len(issues.warnings)
            ),
        },
    }


def _write_report(report: Mapping[str, Any], output: str | Path | None) -> None:
    if output is None:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _descriptor_path(
    config: Mapping[str, Any],
    *,
    config_path: Path | None = None,
) -> Path | None:
    value = _first(
        config,
        (
            "data.ready_descriptor",
            "data.release_descriptor",
            "lineage.ready_descriptor",
            "lineage.release_descriptor",
        ),
    )
    if value is None:
        return None
    path = _resolve_path(str(value), config_path=config_path)
    return path / "READY" if path.is_dir() else path


def _descriptor_readiness(
    descriptor: Mapping[str, Any],
    *,
    view: str,
) -> str:
    if view:
        view_value = _nested(descriptor, f"views.{view}")
        if isinstance(view_value, Mapping):
            state = view_value.get("training_readiness") or view_value.get("status")
            if state is not None:
                return str(state)
    state = (
        descriptor.get("training_readiness")
        or descriptor.get("status")
        or descriptor.get("readiness")
    )
    if state is not None:
        return str(state)
    return "READY" if descriptor.get("ready") is True else ""


def validate_ready_release(
    config: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:

    data = config.get("data") or {}
    if not bool(data.get("require_ready_release", False)):
        return {"required": False, "status": "NOT_REQUIRED"}
    issues = _Issues(detail_limit=100)
    path_context = Path(config_path).expanduser().resolve() if config_path is not None else None
    lineage = config.get("lineage") or {}
    if not isinstance(lineage, Mapping):
        lineage = {}
        issues.error("config_lineage", "require_ready_release requires lineage object")
    manifest_value = data.get("manifest")
    if manifest_value is None:
        issues.error("config_manifest", "require_ready_release requires data.manifest")
        manifest = Path("<missing>")
    else:
        manifest = _resolve_path(str(manifest_value), config_path=path_context)
    expected_manifest_sha = _expected_sha(lineage, ("manifest_sha256",))
    if not expected_manifest_sha:
        issues.error(
            "lineage_manifest_sha",
            "require_ready_release requires lineage.manifest_sha256",
        )
    elif not manifest.is_file() or sha256_file(manifest) != expected_manifest_sha:
        issues.error(
            "lineage_manifest_sha",
            "manifest is missing or its SHA-256 does not match lineage",
        )

    index_dir = _resolve_index_dir(
        manifest,
        index=None,
        config=config,
        config_path=path_context,
    )
    expected_index_sha = _expected_sha(
        lineage,
        ("manifest_index_sha256", "index_sha256", "index_metadata_sha256"),
    )
    if not expected_index_sha:
        issues.error(
            "lineage_index_sha",
            "require_ready_release requires lineage.manifest_index_sha256",
        )
    index_audit = None
    if index_dir is None or not manifest.is_file():
        issues.error(
            "index_missing", "require_ready_release requires v3 READY adjacent/explicit index"
        )
    else:
        index_audit = _IndexAudit(
            index_dir,
            manifest,
            issues,
            expected_metadata_sha256=expected_index_sha,
        )

    descriptor_path = _descriptor_path(config, config_path=path_context)
    expected_descriptor_sha = _expected_sha(
        lineage,
        ("ready_descriptor_sha256", "release_descriptor_sha256"),
    )
    descriptor: Mapping[str, Any] | None = None
    if descriptor_path is None:
        issues.error(
            "ready_descriptor_missing",
            "require_ready_release requires READY/release descriptor path",
        )
    elif not descriptor_path.is_file():
        issues.error("ready_descriptor_missing", f"descriptor does not exist: {descriptor_path}")
    else:
        actual_descriptor_sha = sha256_file(descriptor_path)
        if not expected_descriptor_sha:
            issues.error(
                "lineage_descriptor_sha",
                "require_ready_release requires lineage.ready_descriptor_sha256",
            )
        elif actual_descriptor_sha != expected_descriptor_sha:
            issues.error("lineage_descriptor_sha", "READY descriptor SHA-256 does not match")
        try:
            value = json.loads(descriptor_path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping):
                raise TypeError("descriptor is not an object")
            descriptor = value
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            issues.error("ready_descriptor", f"READY descriptor cannot be parsed: {exc}")
    readiness = ""
    if descriptor is not None:
        readiness = _descriptor_readiness(
            descriptor,
            view=_license_view(config),
        )
        if readiness not in _READY_STATES:
            issues.error(
                "release_not_ready",
                f"release readiness={readiness or '<missing>'}; expected READY or TRAINING_READY",
            )
        descriptor_manifest_sha = _first(
            descriptor,
            ("manifest_sha256", "manifest.sha256", "artifacts.manifest.sha256"),
        )
        if (
            descriptor_manifest_sha is not None
            and expected_manifest_sha is not None
            and str(descriptor_manifest_sha) != expected_manifest_sha
        ):
            issues.error(
                "descriptor_manifest_sha",
                "descriptor manifest SHA-256 does not match lineage",
            )
        descriptor_index_sha = _first(
            descriptor,
            (
                "manifest_index_sha256",
                "index_sha256",
                "index.sha256",
                "artifacts.index.sha256",
            ),
        )
        if (
            descriptor_index_sha is not None
            and expected_index_sha is not None
            and str(descriptor_index_sha) != expected_index_sha
        ):
            issues.error(
                "descriptor_index_sha",
                "descriptor index SHA-256 does not match lineage",
            )

    index_audio_assets = (
        index_audit.metadata.get("audio_assets")
        if index_audit is not None and index_audit.metadata is not None
        else None
    )
    config_audio_assets = _audio_assets_declarations(
        config,
        label="config",
        context_path=path_context,
        issues=issues,
        require_catalog_path=True,
    )
    descriptor_audio_assets = _audio_assets_declarations(
        descriptor,
        label="VIEW",
        context_path=descriptor_path,
        issues=issues,
        require_catalog_path=False,
    )

    permanent_assets = bool(config_audio_assets or isinstance(index_audio_assets, Mapping))
    if permanent_assets:
        if not config_audio_assets:
            issues.error(
                "config_audio_assets_missing",
                "permanent-asset startup verification requires config.audio_assets",
            )
        if not descriptor_audio_assets:
            issues.error(
                "view_audio_assets_missing",
                "permanent-asset startup verification requires VIEW.audio_assets",
            )
        index_declarations = (
            [
                _parse_audio_assets_declaration(
                    index_audio_assets,
                    label="index.audio_assets",
                    context_path=None,
                    issues=issues,
                    require_catalog_path=False,
                )
            ]
            if isinstance(index_audio_assets, Mapping)
            else []
        )
        if not index_declarations:
            issues.error(
                "index_audio_assets_missing",
                "permanent-asset startup verification requires index.audio_assets",
            )
        declarations = [
            *config_audio_assets,
            *descriptor_audio_assets,
            *index_declarations,
        ]
        if declarations:
            identity = declarations[0].identity()
            if any(declaration.identity() != identity for declaration in declarations[1:]):
                issues.error(
                    "audio_assets_cross_binding",
                    "catalog, config, VIEW, and index audio_assets identities differ at startup",
                )
        if (
            _first(
                config,
                (
                    "audio_copied",
                    "data.audio_copied",
                    "lineage.audio_copied",
                    "audio_assets.audio_copied",
                    "data.audio_assets.audio_copied",
                ),
            )
            is not True
        ):
            issues.error(
                "config_audio_copied",
                "permanent asset config must declare audio_copied=true",
            )
        if (
            descriptor is None
            or _first(
                descriptor,
                ("audio_copied", "audio_assets.audio_copied"),
            )
            is not True
        ):
            issues.error(
                "view_audio_copied",
                "permanent asset VIEW must declare audio_copied=true",
            )
        catalog_declaration = next(
            (
                declaration
                for declaration in config_audio_assets
                if declaration.catalog_path is not None
            ),
            None,
        )
        if catalog_declaration is not None:
            _validate_audio_asset_catalog(
                catalog_declaration,
                _ManifestAudioAssets(issues),
                issues,
            )

    result = {
        "required": True,
        "status": "PASS" if issues.error_count == 0 else "FAIL",
        "manifest": str(manifest),
        "index": str(index_dir) if index_dir is not None else None,
        "index_metadata_sha256": (index_audit.metadata_sha256 if index_audit is not None else None),
        "descriptor": str(descriptor_path) if descriptor_path is not None else None,
        "readiness": readiness,
        "issues": [issue.as_dict() for issue in issues.errors],
    }
    if issues.error_count:
        summary = "; ".join(issue.message for issue in issues.errors[:8])
        raise RuntimeError(f"Tokenizer READY startup verification failed: {summary}")
    return result
