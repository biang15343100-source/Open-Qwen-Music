
from __future__ import annotations

import hashlib
import io
import math
import os
import re
import stat
import subprocess
import tarfile
import wave
import zipfile
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable

import torch
import torch.nn.functional as F

_ZIP_CACHE: dict[str, zipfile.ZipFile] = {}
_BINARY_CACHE: OrderedDict[str, "_BoundArchiveFD"] = OrderedDict()
_ARCHIVE_FD_CACHE_SIZE = 16
_ARCHIVE_PAYLOAD_CACHE: OrderedDict[Hashable, bytes] = OrderedDict()
_ARCHIVE_PAYLOAD_CACHE_ENV = "OQM_ARCHIVE_PAYLOAD_CACHE_BYTES"
_DEFAULT_ARCHIVE_PAYLOAD_CACHE_BYTES = 64 * 1024 * 1024
_ARCHIVE_PAYLOAD_CACHE_MAX_BYTES: int | None = None
_ARCHIVE_PAYLOAD_CACHE_BYTES = 0
_PREAD_CHUNK_BYTES = 8 * 1024 * 1024
_CACHE_PID = os.getpid()

_CREDENTIAL_FINGERPRINT_KEY = os.urandom(32)
_PASSWORD_ENV_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

_FFMPEG_SAMPLE_RATE = 24_000
_FFMPEG_BYTES_PER_SAMPLE = 4
_FFMPEG_NO_DURATION_MAX_SEC = 600.0
_FFMPEG_NO_DURATION_PROBE_SEC = 0.05
_FFMPEG_OUTPUT_SLACK_FRAMES = 1024
_FFMPEG_STDERR_MAX_CHARS = 1024
_FFMPEG_TIMEOUT_MIN_SEC = 15.0
_FFMPEG_TIMEOUT_MAX_SEC = 120.0

_PARQUET_CACHE_ENV = "OQM_PARQUET_ROW_GROUP_CACHE_SIZE"
_PARQUET_LOCAL_MIRROR_ENV = "OQM_PARQUET_LOCAL_MIRROR"
_DEFAULT_PARQUET_ROW_GROUP_CACHE_SIZE = 8
_PARQUET_ROW_GROUP_CACHE_SIZE: int | None = None
_PARQUET_ROW_GROUP_CACHE: OrderedDict[
    tuple[str, int, str, int, int], "_CachedParquetRowGroup"
] = OrderedDict()

_PARQUET_PREFETCHED_ROWS: dict[
    str, tuple[tuple[str, int, str, int, int], "_ParquetAudioSource"]
] = {}


class AudioCredentialError(RuntimeError):
    pass


class EncryptedZipReadError(AudioCredentialError):
    pass


class FFmpegAudioDecodeError(RuntimeError):
    pass


class AudioAssetIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_FileIdentity":
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            size=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
        )

    def cache_key(self) -> tuple[int, int, int, int, int]:
        return (
            self.device,
            self.inode,
            self.size,
            self.mtime_ns,
            self.ctime_ns,
        )


@dataclass(slots=True)
class _BoundArchiveFD:
    path: str
    descriptor: int
    identity: _FileIdentity

    def close(self) -> None:
        try:
            os.close(self.descriptor)
        except OSError:
            pass


def audio_password_env(record: Mapping[str, Any]) -> str | None:

    audio = record.get("audio")
    source = record.get("source")
    audio_fields = audio if isinstance(audio, Mapping) else {}
    source_fields = source if isinstance(source, Mapping) else {}
    if "password" in audio_fields or "password" in source_fields:
        raise AudioCredentialError(
            "Manifest records must not contain audio/source.password; use password_env instead"
        )
    values = [
        value
        for value in (
            audio_fields.get("password_env"),
            source_fields.get("password_env"),
        )
        if value is not None
    ]
    if not values:
        return None
    if len(values) == 2 and values[0] != values[1]:
        raise AudioCredentialError(
            "audio.password_env and source.password_env must be consistent with"
        )
    value = values[0]
    if (
        not isinstance(value, str)
        or len(value) > 255
        or _PASSWORD_ENV_RE.fullmatch(value) is None
    ):

        raise AudioCredentialError(
            "password_env must be a valid environment variable name"
        )
    return value


def password_from_env(password_env: str) -> bytes:

    if (
        not isinstance(password_env, str)
        or len(password_env) > 255
        or _PASSWORD_ENV_RE.fullmatch(password_env) is None
    ):
        raise AudioCredentialError(
            "password_env must be a valid environment variable name"
        )
    value = os.environ.get(password_env)
    if value is None or value == "":
        raise AudioCredentialError(
            f"Audio credential environment variable {password_env} is missing or empty"
        )
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError:
        raise AudioCredentialError(
            f"Audio credential environment variable {password_env} cannot be encoded as UTF-8"
        ) from None


def _credential_fingerprint(password: bytes | None) -> str:
    if password is None:
        return "none"

    return hashlib.blake2b(
        password,
        key=_CREDENTIAL_FINGERPRINT_KEY,
        digest_size=16,
    ).hexdigest()


def _close_container_caches() -> None:
    for archive in _ZIP_CACHE.values():
        try:
            archive.close()
        except Exception:  # noqa: BLE001
            pass
    for handle in _BINARY_CACHE.values():
        handle.close()
    _ZIP_CACHE.clear()
    _BINARY_CACHE.clear()


def _clear_archive_payload_cache() -> None:
    global _ARCHIVE_PAYLOAD_CACHE_BYTES
    _ARCHIVE_PAYLOAD_CACHE.clear()
    _ARCHIVE_PAYLOAD_CACHE_BYTES = 0


def _ensure_cache_process() -> None:

    global _CACHE_PID
    pid = os.getpid()
    if pid == _CACHE_PID:
        return
    _close_container_caches()
    _clear_archive_payload_cache()
    _PARQUET_ROW_GROUP_CACHE.clear()
    _PARQUET_PREFETCHED_ROWS.clear()
    _CACHE_PID = pid


def reset_audio_io_state() -> None:

    global _CACHE_PID
    _close_container_caches()
    _clear_archive_payload_cache()
    _PARQUET_ROW_GROUP_CACHE.clear()
    _PARQUET_PREFETCHED_ROWS.clear()
    _CACHE_PID = os.getpid()


def configure_parquet_row_group_cache(max_entries: int) -> None:

    global _PARQUET_ROW_GROUP_CACHE_SIZE
    try:
        size = int(max_entries)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Parquet row-group cache limit must be a non-negative integer: {max_entries!r}"
        ) from exc
    if size < 0:
        raise ValueError(f"Parquet row-group cache limit cannot be negative: {size}")
    _ensure_cache_process()
    _PARQUET_ROW_GROUP_CACHE_SIZE = size
    while len(_PARQUET_ROW_GROUP_CACHE) > size:
        _PARQUET_ROW_GROUP_CACHE.popitem(last=False)


def configure_archive_payload_cache(max_bytes: int) -> None:

    global _ARCHIVE_PAYLOAD_CACHE_MAX_BYTES
    if isinstance(max_bytes, bool):
        raise ValueError("Archive payload cache budget must be a non-negative integer")
    try:
        budget = int(max_bytes)
    except (TypeError, ValueError) as exc:
        raise ValueError("Archive payload cache budget must be a non-negative integer") from exc
    if budget < 0:
        raise ValueError("Archive payload cache budget cannot be negative")
    _ensure_cache_process()
    _ARCHIVE_PAYLOAD_CACHE_MAX_BYTES = budget
    _trim_archive_payload_cache(budget)


def _resolve_archive_payload_cache_bytes() -> int:
    global _ARCHIVE_PAYLOAD_CACHE_MAX_BYTES
    if _ARCHIVE_PAYLOAD_CACHE_MAX_BYTES is None:
        raw = os.environ.get(
            _ARCHIVE_PAYLOAD_CACHE_ENV,
            str(_DEFAULT_ARCHIVE_PAYLOAD_CACHE_BYTES),
        )
        try:
            configure_archive_payload_cache(int(raw))
        except ValueError as exc:
            raise ValueError(
                f"environment variable {_ARCHIVE_PAYLOAD_CACHE_ENV} must be a "
                f"non-negative integer; received {raw!r}"
            ) from exc
    assert _ARCHIVE_PAYLOAD_CACHE_MAX_BYTES is not None
    return _ARCHIVE_PAYLOAD_CACHE_MAX_BYTES


def _trim_archive_payload_cache(max_bytes: int) -> None:
    global _ARCHIVE_PAYLOAD_CACHE_BYTES
    while (
        _ARCHIVE_PAYLOAD_CACHE
        and _ARCHIVE_PAYLOAD_CACHE_BYTES > max_bytes
    ):
        _, evicted = _ARCHIVE_PAYLOAD_CACHE.popitem(last=False)
        _ARCHIVE_PAYLOAD_CACHE_BYTES -= len(evicted)


def _resolve_parquet_cache_size(override: int | None) -> int:
    if override is not None:
        configure_parquet_row_group_cache(override)
    elif _PARQUET_ROW_GROUP_CACHE_SIZE is None:
        raw = os.environ.get(
            _PARQUET_CACHE_ENV, str(_DEFAULT_PARQUET_ROW_GROUP_CACHE_SIZE)
        )
        try:
            configure_parquet_row_group_cache(int(raw))
        except ValueError as exc:
            raise ValueError(
                f"environment variable {_PARQUET_CACHE_ENV} must be a non-negative "
                f"integer; received {raw!r}"
            ) from exc
    assert _PARQUET_ROW_GROUP_CACHE_SIZE is not None
    return _PARQUET_ROW_GROUP_CACHE_SIZE


def _cache_payload(key: Hashable, loader: Callable[[], bytes]) -> bytes:
    global _ARCHIVE_PAYLOAD_CACHE_BYTES
    _ensure_cache_process()
    budget = _resolve_archive_payload_cache_bytes()
    payload = _ARCHIVE_PAYLOAD_CACHE.get(key)
    if payload is not None:
        _ARCHIVE_PAYLOAD_CACHE.move_to_end(key)
        return payload
    payload = loader()
    if not isinstance(payload, bytes):
        payload = bytes(payload)
    if budget > 0 and len(payload) <= budget:
        previous = _ARCHIVE_PAYLOAD_CACHE.pop(key, None)
        if previous is not None:
            _ARCHIVE_PAYLOAD_CACHE_BYTES -= len(previous)
        _ARCHIVE_PAYLOAD_CACHE[key] = payload
        _ARCHIVE_PAYLOAD_CACHE_BYTES += len(payload)
        _trim_archive_payload_cache(budget)
    return payload


def _archive_path_identity(path: str) -> tuple[str, _FileIdentity]:
    normalized = os.path.abspath(os.path.expanduser(path))
    try:
        file_stat = os.stat(normalized)
    except OSError as exc:
        raise AudioAssetIntegrityError(
            f"TAR asset is unreadable: {normalized} ({type(exc).__name__})"
        ) from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise AudioAssetIntegrityError(f"TAR asset is not a regular file: {normalized}")
    return normalized, _FileIdentity.from_stat(file_stat)


def _open_bound_archive(path: str) -> _BoundArchiveFD:
    normalized = os.path.abspath(os.path.expanduser(path))
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    for _attempt in range(3):
        try:
            descriptor = os.open(normalized, flags)
        except OSError as exc:
            raise AudioAssetIntegrityError(
                f"cannot open TAR asset: {normalized} ({type(exc).__name__})"
            ) from exc
        try:
            descriptor_stat = os.fstat(descriptor)
            path_stat = os.stat(normalized)
            if not stat.S_ISREG(descriptor_stat.st_mode) or not stat.S_ISREG(
                path_stat.st_mode
            ):
                raise AudioAssetIntegrityError(
                    f"TAR asset is not a regular file: {normalized}"
                )
            descriptor_identity = _FileIdentity.from_stat(descriptor_stat)
            path_identity = _FileIdentity.from_stat(path_stat)
            if descriptor_identity == path_identity:
                return _BoundArchiveFD(
                    path=normalized,
                    descriptor=descriptor,
                    identity=descriptor_identity,
                )
        except AudioAssetIntegrityError:
            os.close(descriptor)
            raise
        except OSError:
            pass
        os.close(descriptor)
    raise AudioAssetIntegrityError(
        f"TAR path identity kept changing while opening the asset: {normalized}"
    )


def _bound_archive(path: str) -> _BoundArchiveFD:

    _ensure_cache_process()
    normalized = os.path.abspath(os.path.expanduser(path))
    cached = _BINARY_CACHE.get(normalized)
    if cached is not None:
        try:
            descriptor_stat = os.fstat(cached.descriptor)
            path_stat = os.stat(normalized)
            descriptor_identity = _FileIdentity.from_stat(descriptor_stat)
            path_identity = _FileIdentity.from_stat(path_stat)
        except OSError:
            descriptor_identity = path_identity = None
        if (
            descriptor_identity == cached.identity
            and path_identity == cached.identity
            and descriptor_identity is not None
        ):
            _BINARY_CACHE.move_to_end(normalized)
            return cached

        _BINARY_CACHE.pop(normalized, None)
        cached.close()

    opened = _open_bound_archive(normalized)
    _BINARY_CACHE[normalized] = opened
    while len(_BINARY_CACHE) > _ARCHIVE_FD_CACHE_SIZE:
        _, evicted = _BINARY_CACHE.popitem(last=False)
        evicted.close()
    return opened


def _assert_bound_archive_stable(
    archive: _BoundArchiveFD,
    *,
    operation: str,
) -> None:
    try:
        descriptor_stat = os.fstat(archive.descriptor)
        path_stat = os.stat(archive.path)
    except OSError as exc:
        raise AudioAssetIntegrityError(
            f"{operation}Period TAR The asset is unreadable: {archive.path}"
        ) from exc
    descriptor_identity = _FileIdentity.from_stat(descriptor_stat)
    path_identity = _FileIdentity.from_stat(path_stat)
    if (
        descriptor_identity != archive.identity
        or path_identity != archive.identity
    ):
        raise AudioAssetIntegrityError(
            f"TAR asset identity drift during {operation}: "
            f"{archive.path}"
        )


def _pread_exact(
    archive: _BoundArchiveFD,
    *,
    offset: int,
    size: int,
) -> bytes:
    if (
        isinstance(offset, bool)
        or isinstance(size, bool)
        or not isinstance(offset, int)
        or not isinstance(size, int)
        or offset < 0
        or size <= 0
    ):
        raise AudioAssetIntegrityError(
            "TAR archive_offset must be a non-negative integer and archive_size must be a positive integer"
        )
    file_size = archive.identity.size
    if offset > file_size or size > file_size - offset:
        raise AudioAssetIntegrityError(
            "TAR payload range is out of bounds: "
            f"offset={offset} size={size} file_size={file_size} "
            f"path={archive.path}"
        )
    _assert_bound_archive_stable(archive, operation="before pread")
    chunks: list[bytes] = []
    consumed = 0
    while consumed < size:
        request = min(_PREAD_CHUNK_BYTES, size - consumed)
        try:
            chunk = os.pread(
                archive.descriptor,
                request,
                offset + consumed,
            )
        except InterruptedError:
            continue
        except OSError as exc:
            raise AudioAssetIntegrityError(
                f"TAR payload pread failed: path={archive.path} "
                f"offset={offset + consumed}({type(exc).__name__})"
            ) from exc
        if not chunk:
            raise AudioAssetIntegrityError(
                "TAR payload short read: "
                f"expected={size} actual={consumed} offset={offset} "
                f"path={archive.path}"
            )
        chunks.append(chunk)
        consumed += len(chunk)
    _assert_bound_archive_stable(archive, operation="pread")
    payload = b"".join(chunks)
    if len(payload) != size:
        raise AudioAssetIntegrityError(
            f"TAR payload short read: expected={size} actual={len(payload)} "
            f"offset={offset} path={archive.path}"
        )
    return payload


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_tar_asset_metadata(
    *,
    asset_id: str | None,
    payload_sha256: str | None,
    asset_revision: str | None,
    shard_sha256: str | None,
    archive_offset: int | None,
    archive_size: int | None,
) -> tuple[str, str, str, str] | None:
    values = (asset_id, payload_sha256, asset_revision, shard_sha256)
    if not any(value is not None for value in values):
        return None
    if not all(value is not None for value in values):
        raise AudioAssetIntegrityError(
            "permanent TAR assets must declare asset_id/payload_sha256/"
            "asset_revision/shard_sha256"
        )
    assert (
        asset_id is not None
        and payload_sha256 is not None
        and asset_revision is not None
        and shard_sha256 is not None
    )
    if (
        not isinstance(asset_id, str)
        or not asset_id
        or asset_id != asset_id.strip()
        or len(asset_id) > 1024
    ):
        raise AudioAssetIntegrityError("audio.asset_id must be a non-empty string")
    if (
        not isinstance(asset_revision, str)
        or not asset_revision
        or asset_revision != asset_revision.strip()
        or len(asset_revision) > 1024
    ):
        raise AudioAssetIntegrityError(
            "audio.asset_revision must be a non-empty string"
        )
    if not _is_sha256(payload_sha256):
        raise AudioAssetIntegrityError(
            "audio.payload_sha256 must be 64 lowercase hexadecimal SHA-256"
        )
    if not _is_sha256(shard_sha256):
        raise AudioAssetIntegrityError(
            "audio.shard_sha256 must be 64 lowercase hexadecimal SHA-256"
        )
    if archive_offset is None or archive_size is None:
        raise AudioAssetIntegrityError(
            "permanent TAR assets must also be declared archive_offset/archive_size"
        )
    return asset_id, payload_sha256, asset_revision, shard_sha256


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    channels: int
    sample_width: int
    num_frames: int

    @property
    def duration_sec(self) -> float:
        return self.num_frames / self.sample_rate


@dataclass(frozen=True)
class _ParquetAudioRef:
    uri: str
    path: Path
    row_group: int
    row: int
    column: str


@dataclass(frozen=True)
class _CachedParquetRowGroup:
    column: Any
    num_rows: int
    value_kind: str


@dataclass(frozen=True)
class _ParquetAudioSource:
    payload: bytes | None = None
    path: str | None = None


_PARQUET_INTEGER_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_PARQUET_COLUMN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")


def _resolve_parquet_storage_path(path: Path) -> Path:

    raw = os.environ.get(_PARQUET_LOCAL_MIRROR_ENV, "").strip()
    if not raw:
        return path
    if raw.count("::") != 1:
        raise ValueError(
            f"{_PARQUET_LOCAL_MIRROR_ENV} must use /source::/mirror"
        )
    source_text, mirror_text = raw.split("::", 1)
    source_root = Path(source_text)
    mirror_root = Path(mirror_text)
    if not source_root.is_absolute() or not mirror_root.is_absolute():
        raise ValueError(
            f"{_PARQUET_LOCAL_MIRROR_ENV} source and mirror must be absolute paths"
        )
    normalized_path = path.resolve(strict=False)
    normalized_source = source_root.resolve(strict=False)
    normalized_mirror = mirror_root.resolve(strict=False)
    try:
        relative = normalized_path.relative_to(normalized_source)
    except ValueError:
        return path
    local_path = (normalized_mirror / relative).resolve(strict=False)
    try:
        local_path.relative_to(normalized_mirror)
    except ValueError as exc:
        raise ValueError("Parquet local mirror path is outside the configured root") from exc
    if not local_path.is_file():
        raise FileNotFoundError(
            f"Parquet local mirror is missing; refusing to fall back to remote storage: {local_path}"
        )
    return local_path


def _parse_parquet_uri(uri: str) -> _ParquetAudioRef:
    prefix = "parquet://"
    if not uri.startswith(prefix):
        raise ValueError(f"expected a Parquet audio URI; received {uri!r}")
    body = uri.removeprefix(prefix)
    if body.count("::") != 1:
        raise ValueError(
            "Parquet URI must be parquet:///abs/file.parquet::"
            f"rg=N&row=M&col=audio; received {uri!r}"
        )
    path_text, parameter_text = body.split("::", 1)
    path = Path(path_text)
    if not path_text or not path.is_absolute():
        raise ValueError(f"Parquet URI file path must be absolute: {uri!r}")
    if not parameter_text:
        raise ValueError(f"Parquet URI is missing a parameter: {uri!r}")
    parameters: dict[str, str] = {}
    for chunk in parameter_text.split("&"):
        key, separator, value = chunk.partition("=")
        if not chunk or not separator or not key or not value:
            raise ValueError(f"Parquet URI has an invalid parameter: {chunk!r} in {uri!r}")
        if key in parameters:
            raise ValueError(f"Parquet URI contains a duplicate parameter {key!r}: {uri!r}")
        parameters[key] = value
    expected = {"rg", "row", "col"}
    if set(parameters) != expected:
        missing = sorted(expected - set(parameters))
        unknown = sorted(set(parameters) - expected)
        raise ValueError(
            f"Parquet URI parameters must be exactly rg, row, and col: missing={missing} "
            f"unknown={unknown} uri={uri!r}"
        )
    for key in ("rg", "row"):
        if not _PARQUET_INTEGER_RE.fullmatch(parameters[key]):
            raise ValueError(
                f"Parquet URI parameter {key} must be a canonical non-negative integer: "
                f"{parameters[key]!r} in {uri!r}"
            )
    column = parameters["col"]
    if not _PARQUET_COLUMN_RE.fullmatch(column):
        raise ValueError(f"Parquet URI has an invalid col parameter: {column!r} in {uri!r}")
    return _ParquetAudioRef(
        uri=uri,
        path=_resolve_parquet_storage_path(path),
        row_group=int(parameters["rg"]),
        row=int(parameters["row"]),
        column=column,
    )


def _parquet_cache_key(
    ref: _ParquetAudioRef,
) -> tuple[str, int, str, int, int]:
    try:
        stat = ref.path.stat()
    except OSError as exc:
        raise FileNotFoundError(f"Parquet file is unreadable: {ref.path}") from exc
    if not ref.path.is_file():
        raise FileNotFoundError(f"Parquet path is not a regular file: {ref.path}")
    return (
        str(ref.path),
        ref.row_group,
        ref.column,
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _read_parquet_row_group(ref: _ParquetAudioRef) -> _CachedParquetRowGroup:

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Reading parquet:// audio requires the pyarrow training dependency"
        ) from exc

    parquet_file = None
    try:
        parquet_file = pq.ParquetFile(ref.path)
        if ref.row_group >= parquet_file.num_row_groups:
            raise IndexError(
                f"Parquet row group out of bounds: rg={ref.row_group} "
                f"num_row_groups={parquet_file.num_row_groups} path={ref.path}"
            )
        expected_rows = int(
            parquet_file.metadata.row_group(ref.row_group).num_rows
        )
        table = parquet_file.read_row_group(
            ref.row_group,
            columns=[ref.column],
            use_threads=False,
        )
    except (IndexError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read Parquet row group: path={ref.path} rg={ref.row_group} "
            f"col={ref.column}"
        ) from exc
    finally:
        if parquet_file is not None:
            close = getattr(parquet_file, "close", None)
            if close is not None:
                close()
    if ref.column not in table.column_names:
        raise KeyError(f"Parquet column does not exist: {ref.column!r} path={ref.path}")
    column = table.column(ref.column)
    if len(column) != expected_rows:
        raise RuntimeError(
            f"Parquet row-group count changed: metadata={expected_rows} "
            f"loaded={len(column)} path={ref.path} rg={ref.row_group}"
        )
    value_type = column.type
    if pa.types.is_binary(value_type) or pa.types.is_large_binary(value_type):
        value_kind = "binary"
    elif pa.types.is_struct(value_type):
        fields = {field.name: field.type for field in value_type}
        if set(fields) != {"bytes", "path"}:
            raise TypeError(
                "Parquet audio struct must contain exactly bytes and path fields; "
                f"received {sorted(fields)}: path={ref.path} col={ref.column}"
            )
        if not (
            pa.types.is_binary(fields["bytes"])
            or pa.types.is_large_binary(fields["bytes"])
        ):
            raise TypeError(
                f"Parquet audio struct.bytes must be binary: {fields['bytes']}"
            )
        if not (
            pa.types.is_string(fields["path"])
            or pa.types.is_large_string(fields["path"])
        ):
            raise TypeError(
                f"Parquet audio struct.path must be a string: {fields['path']}"
            )
        value_kind = "struct"
    else:
        raise TypeError(
            "Parquet audio column supports only HF binary or struct<bytes,path>; "
            f"received {value_type}: path={ref.path} col={ref.column}"
        )
    return _CachedParquetRowGroup(
        column=column,
        num_rows=expected_rows,
        value_kind=value_kind,
    )


def _cached_parquet_row_group(
    ref: _ParquetAudioRef,
    key: tuple[str, int, str, int, int],
    *,
    cache_size: int,
) -> _CachedParquetRowGroup:
    cached = _PARQUET_ROW_GROUP_CACHE.get(key)
    if cached is not None:
        _PARQUET_ROW_GROUP_CACHE.move_to_end(key)
        return cached
    cached = _read_parquet_row_group(ref)

    logical_key = key[:3]
    for old_key in list(_PARQUET_ROW_GROUP_CACHE):
        if old_key[:3] == logical_key and old_key != key:
            _PARQUET_ROW_GROUP_CACHE.pop(old_key)
    if cache_size > 0:
        _PARQUET_ROW_GROUP_CACHE[key] = cached
        while len(_PARQUET_ROW_GROUP_CACHE) > cache_size:
            _PARQUET_ROW_GROUP_CACHE.popitem(last=False)
    return cached


def _extract_parquet_audio_source(
    ref: _ParquetAudioRef,
    row_group: _CachedParquetRowGroup,
) -> _ParquetAudioSource:
    if ref.row >= row_group.num_rows:
        raise IndexError(
            f"Parquet row out of bounds: row={ref.row} rows={row_group.num_rows} "
            f"path={ref.path} rg={ref.row_group}"
        )
    scalar = row_group.column[ref.row]
    if not scalar.is_valid:
        raise ValueError(
            f"Parquet audio value is null: path={ref.path} rg={ref.row_group} "
            f"row={ref.row} col={ref.column}"
        )
    value = scalar.as_py()
    if row_group.value_kind == "binary":
        if value is None:
            raise ValueError(
                f"Parquet binary audio value is null: path={ref.path} row={ref.row}"
            )
        return _ParquetAudioSource(payload=bytes(value))
    if not isinstance(value, dict):
        raise TypeError(
            f"Parquet audio struct decoded to an invalid type: {type(value).__name__}"
        )
    payload = value.get("bytes")
    external_path = value.get("path")
    if payload is not None:
        return _ParquetAudioSource(payload=bytes(payload))
    if external_path is None or not isinstance(external_path, str) or not external_path:
        raise ValueError(
            "Parquet audio struct has neither bytes nor a path: "
            f"path={ref.path} rg={ref.row_group} row={ref.row}"
        )
    if external_path.startswith("file://"):
        resolved_path = external_path
    else:
        candidate = Path(external_path)
        resolved_path = str(
            candidate if candidate.is_absolute() else ref.path.parent / candidate
        )
    return _ParquetAudioSource(path=resolved_path)


def prefetch_parquet_audio(
    paths: Iterable[str | Path],
    *,
    row_group_cache_size: int | None = None,
) -> None:

    _ensure_cache_process()
    cache_size = _resolve_parquet_cache_size(row_group_cache_size)
    references = [
        _parse_parquet_uri(str(path))
        for path in paths
        if str(path).startswith("parquet://")
    ]
    _PARQUET_PREFETCHED_ROWS.clear()
    grouped: dict[
        tuple[str, int, str, int, int], list[_ParquetAudioRef]
    ] = {}
    for ref in references:
        grouped.setdefault(_parquet_cache_key(ref), []).append(ref)
    for key, refs in grouped.items():
        row_group = _cached_parquet_row_group(
            refs[0], key, cache_size=cache_size
        )
        for ref in refs:
            source = _extract_parquet_audio_source(ref, row_group)
            _PARQUET_PREFETCHED_ROWS[ref.uri] = (key, source)


def _decode_pcm(raw: bytes, sample_width: int) -> torch.Tensor:
    if sample_width == 1:
        x = torch.frombuffer(bytearray(raw), dtype=torch.uint8).float()
        return (x - 128.0) / 128.0
    if sample_width == 2:
        return torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
    if sample_width == 3:
        packed = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(-1, 3).to(torch.int32)
        values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        values = torch.where(values >= (1 << 23), values - (1 << 24), values)
        return values.float() / float(1 << 23)
    if sample_width == 4:
        return torch.frombuffer(bytearray(raw), dtype=torch.int32).float() / 2147483648.0
    raise ValueError(f"Unsupported PCM sample width: {sample_width}")


def _open_wave(source: str | Path | io.BytesIO) -> wave.Wave_read:
    return wave.open(str(source) if isinstance(source, (str, Path)) else source, "rb")


def probe_audio(path: str | Path) -> AudioInfo:
    path_text = str(path)
    if path_text.startswith("file://"):
        path_text = path_text.removeprefix("file://")
        if not Path(path_text).is_absolute() or "::" in path_text:
            raise ValueError(f"file URI must point to the absolute path: {path!r}")
    path = Path(path_text)
    if path.suffix.lower() == ".wav":
        try:
            with _open_wave(path) as f:
                return AudioInfo(
                    sample_rate=f.getframerate(),
                    channels=f.getnchannels(),
                    sample_width=f.getsampwidth(),
                    num_frames=f.getnframes(),
                )
        except wave.Error:
            pass
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(f"Unable to detect {path}; non-PCM WAV decoding requires soundfile") from exc
    info = sf.info(str(path))
    return AudioInfo(
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        sample_width=0,
        num_frames=int(info.frames),
    )


def _read_wave(
    source: str | Path | io.BytesIO,
    *,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
) -> tuple[torch.Tensor, int]:
    with _open_wave(source) as f:
        channels = f.getnchannels()
        sample_rate = f.getframerate()
        sample_width = f.getsampwidth()
        start_frame = min(f.getnframes(), max(0, round(start_sec * sample_rate)))
        f.setpos(start_frame)
        frames_to_read = (
            f.getnframes() - start_frame
            if duration_sec is None
            else min(f.getnframes() - start_frame, round(duration_sec * sample_rate))
        )


        if frames_to_read <= 0:
            return torch.zeros(0), sample_rate
        frames = f.readframes(frames_to_read)
    waveform = _decode_pcm(frames, sample_width).view(-1, channels).mean(dim=1)
    return waveform.contiguous(), sample_rate


def _ffmpeg_stderr_summary(stderr: bytes | str | None) -> str:

    if isinstance(stderr, bytes):
        text = stderr.decode("utf-8", errors="replace")
    elif isinstance(stderr, str):
        text = stderr
    else:
        text = ""
    sensitive_values: list[str] = []
    for key, value in os.environ.items():
        normalized_key = key.upper()
        if value and any(
            marker in normalized_key
            for marker in ("PASSWORD", "SECRET", "TOKEN", "CREDENTIAL")
        ):
            sensitive_values.append(value)
            text = text.replace(value, "<redacted>")
    text = " ".join(text.split())
    for value in sensitive_values:
        normalized_value = " ".join(value.split())
        if normalized_value:
            text = text.replace(normalized_value, "<redacted>")
    if not text:
        return "<empty>"
    if len(text) <= _FFMPEG_STDERR_MAX_CHARS:
        return text
    return (
        text[:_FFMPEG_STDERR_MAX_CHARS]
        + f"...<truncated:{len(text) - _FFMPEG_STDERR_MAX_CHARS} chars>"
    )


def _format_ffmpeg_seconds(value: float) -> str:
    return f"{value:.9f}".rstrip("0").rstrip(".")


def _read_audio_bytes_ffmpeg(
    payload: bytes,
    *,
    start_sec: float,
    duration_sec: float | None,
    previous_error: BaseException,
) -> tuple[torch.Tensor, int]:

    previous_name = type(previous_error).__name__
    try:
        normalized_start = float(start_sec)
    except (TypeError, ValueError):
        normalized_start = math.nan
    if not math.isfinite(normalized_start):
        raise FFmpegAudioDecodeError("ffmpeg fallback of start_sec must be a finite number")
    normalized_start = max(0.0, normalized_start)

    no_duration = duration_sec is None
    if no_duration:
        requested_duration = _FFMPEG_NO_DURATION_MAX_SEC
        command_duration = _FFMPEG_NO_DURATION_MAX_SEC + _FFMPEG_NO_DURATION_PROBE_SEC
        safe_frames = math.ceil(_FFMPEG_NO_DURATION_MAX_SEC * _FFMPEG_SAMPLE_RATE)
        max_output_bytes = safe_frames * _FFMPEG_BYTES_PER_SAMPLE
    else:
        try:
            requested_duration = float(duration_sec)
        except (TypeError, ValueError):
            requested_duration = math.nan
        if not math.isfinite(requested_duration) or requested_duration <= 0.0:
            raise FFmpegAudioDecodeError(
                "ffmpeg fallback of duration_sec must be a positive finite number"
            )
        if requested_duration > _FFMPEG_NO_DURATION_MAX_SEC:
            raise FFmpegAudioDecodeError(
                "ffmpeg fallback duration exceeds the safety limit of "
                f"{_FFMPEG_NO_DURATION_MAX_SEC:g}s"
            )
        command_duration = requested_duration
        safe_frames = (
            math.ceil(requested_duration * _FFMPEG_SAMPLE_RATE)
            + _FFMPEG_OUTPUT_SLACK_FRAMES
        )
        max_output_bytes = safe_frames * _FFMPEG_BYTES_PER_SAMPLE


    seekable_container = b"ftyp" in payload[:64]
    input_name = "pipe:0"
    input_payload: bytes | None = payload
    pass_fds: tuple[int, ...] = ()
    memfd: int | None = None
    if seekable_container and hasattr(os, "memfd_create"):
        memfd = os.memfd_create("oqm-audio-bytes", flags=0)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            written += os.write(memfd, view[written:])
        os.lseek(memfd, 0, os.SEEK_SET)
        input_name = f"/proc/self/fd/{memfd}"
        input_payload = None
        pass_fds = (memfd,)

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        input_name,
    ]
    if normalized_start > 0.0:
        command.extend(["-ss", _format_ffmpeg_seconds(normalized_start)])
    command.extend(
        [
            "-t",
            _format_ffmpeg_seconds(command_duration),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            str(_FFMPEG_SAMPLE_RATE),
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    timeout_sec = min(
        _FFMPEG_TIMEOUT_MAX_SEC,
        max(
            _FFMPEG_TIMEOUT_MIN_SEC,
            5.0 + (normalized_start + requested_duration) * 0.25,
        ),
    )
    process_env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "LANG": "C",
        "LC_ALL": "C",
    }
    if os.environ.get("LD_LIBRARY_PATH"):
        process_env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    try:
        result = subprocess.run(
            command,
            input=input_payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_sec,
            env=process_env,
            pass_fds=pass_fds,
        )
    except subprocess.TimeoutExpired:
        raise FFmpegAudioDecodeError(
            "Built-in byte decoder failed "
            f"({previous_name}); ffmpeg fallback timed out "
            f"({timeout_sec:g}s)"
        ) from None
    except FileNotFoundError:
        raise FFmpegAudioDecodeError(
            f"Built-in byte decoder failed ({previous_name}); ffmpeg was not found"
        ) from None
    except OSError as exc:
        raise FFmpegAudioDecodeError(
            "Built-in byte decoder failed "
            f"({previous_name}); ffmpeg could not start ({type(exc).__name__})"
        ) from None
    finally:
        if memfd is not None:
            os.close(memfd)

    stderr_summary = _ffmpeg_stderr_summary(result.stderr)
    if result.returncode != 0:
        raise FFmpegAudioDecodeError(
            "Built-in byte decoder failed "
            f"({previous_name}); ffmpeg fallback returned non-zero status "
            f"{result.returncode}; stderr={stderr_summary}"
        )
    output = result.stdout
    if not isinstance(output, (bytes, bytearray)):
        raise FFmpegAudioDecodeError("ffmpeg fallback stdout must contain raw bytes")
    output_size = len(output)
    if output_size == 0:
        raise FFmpegAudioDecodeError(
            f"ffmpeg fallback returned empty audio; stderr={stderr_summary}"
        )
    if output_size > max_output_bytes:
        raise FFmpegAudioDecodeError(
            "ffmpeg fallback output exceeds the safety limit: "
            f"{output_size} > {max_output_bytes} bytes"
        )
    if no_duration and output_size >= max_output_bytes:
        raise FFmpegAudioDecodeError(
            "ffmpeg fallback reached the safety limit without duration_sec: "
            f"{_FFMPEG_NO_DURATION_MAX_SEC:g}s; provide an explicit crop duration"
        )
    if output_size % _FFMPEG_BYTES_PER_SAMPLE != 0:
        raise FFmpegAudioDecodeError(
            "ffmpeg fallback returned an f32le byte count that is not a multiple of 4"
        )
    waveform = torch.frombuffer(
        bytearray(output),
        dtype=torch.float32,
    ).clone()
    return waveform, _FFMPEG_SAMPLE_RATE


def _read_audio_bytes(
    payload: bytes,
    *,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
) -> tuple[torch.Tensor, int]:
    buffer = io.BytesIO(payload)
    try:
        return _read_wave(buffer, start_sec=start_sec, duration_sec=duration_sec)
    except (wave.Error, EOFError) as exc:
        wave_error: BaseException = exc
        buffer.seek(0)


    if b"ftyp" in payload[:64]:
        return _read_audio_bytes_ffmpeg(
            payload,
            start_sec=start_sec,
            duration_sec=duration_sec,
            previous_error=wave_error,
        )
    try:
        import soundfile as sf
    except ImportError as exc:
        soundfile_error: BaseException = exc
    else:
        try:
            try:
                sound_file = sf.SoundFile(buffer)
            except sf.LibsndfileError as initial_error:


                scan_start = 0
                if payload.startswith(b"ID3") and len(payload) >= 10:
                    size = (
                        ((payload[6] & 0x7F) << 21)
                        | ((payload[7] & 0x7F) << 14)
                        | ((payload[8] & 0x7F) << 7)
                        | (payload[9] & 0x7F)
                    )
                    scan_start = 10 + size
                frame_start = -1
                for index in range(
                    scan_start,
                    min(len(payload) - 1, scan_start + 2_000_000),
                ):
                    first, second = payload[index], payload[index + 1]
                    if (
                        first == 0xFF
                        and (second & 0xE0) == 0xE0
                        and (second & 0x18) != 0x08
                    ):
                        frame_start = index
                        break
                if frame_start < 0:
                    raise initial_error
                sound_file = sf.SoundFile(io.BytesIO(payload[frame_start:]))
            with sound_file as f:
                f.seek(min(len(f), round(start_sec * f.samplerate)))
                frames = (
                    -1 if duration_sec is None else round(duration_sec * f.samplerate)
                )
                array = f.read(
                    frames=frames,
                    always_2d=True,
                    dtype="float32",
                )
                sample_rate = f.samplerate
            return torch.from_numpy(array).mean(dim=1), int(sample_rate)
        except Exception as exc:  # noqa: BLE001
            soundfile_error = exc
    return _read_audio_bytes_ffmpeg(
        payload,
        start_sec=start_sec,
        duration_sec=duration_sec,
        previous_error=soundfile_error,
    )


def load_audio(
    path: str | Path,
    *,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    archive_offset: int | None = None,
    archive_size: int | None = None,
    parquet_row_group_cache_size: int | None = None,
    password: bytes | None = None,
    asset_id: str | None = None,
    payload_sha256: str | None = None,
    asset_revision: str | None = None,
    shard_sha256: str | None = None,
) -> tuple[torch.Tensor, int]:

    _ensure_cache_process()
    path_text = str(path)
    has_asset_metadata = any(
        value is not None
        for value in (
            asset_id,
            payload_sha256,
            asset_revision,
            shard_sha256,
        )
    )
    if has_asset_metadata and not path_text.startswith("tar://"):
        raise AudioAssetIntegrityError(
            "audio.asset_id, payload_sha256, asset_revision, and shard_sha256 "
            "can be used only with permanent tar:// assets"
        )
    if password is not None and (
        not isinstance(password, bytes) or len(password) == 0
    ):
        raise AudioCredentialError("ZIP password must be non-null bytes")
    if password is not None and not path_text.startswith("zip://"):
        raise AudioCredentialError("ZIP password can be used only with a zip:// audio URI")
    if path_text.startswith("parquet://"):
        cache_size = _resolve_parquet_cache_size(
            parquet_row_group_cache_size
        )
        ref = _parse_parquet_uri(path_text)
        key = _parquet_cache_key(ref)
        prefetched = _PARQUET_PREFETCHED_ROWS.get(path_text)
        if prefetched is not None and prefetched[0] == key:
            source = prefetched[1]
        else:
            row_group = _cached_parquet_row_group(
                ref, key, cache_size=cache_size
            )
            source = _extract_parquet_audio_source(ref, row_group)
        if source.payload is not None:
            return _read_audio_bytes(
                source.payload,
                start_sec=start_sec,
                duration_sec=duration_sec,
            )
        assert source.path is not None
        return load_audio(
            source.path,
            start_sec=start_sec,
            duration_sec=duration_sec,
            parquet_row_group_cache_size=parquet_row_group_cache_size,
        )
    if path_text.startswith("tar://"):
        body = path_text.removeprefix("tar://")
        if body.count("::") != 1:
            raise ValueError("tar URI must contain exactly one archive::member pair")
        archive_text, member = body.split("::", 1)
        if not archive_text or not member:
            raise ValueError("tar URI of archive/member must not be empty")
        asset_metadata = _validate_tar_asset_metadata(
            asset_id=asset_id,
            payload_sha256=payload_sha256,
            asset_revision=asset_revision,
            shard_sha256=shard_sha256,
            archive_offset=archive_offset,
            archive_size=archive_size,
        )
        if (archive_offset is None) != (archive_size is None):
            raise AudioAssetIntegrityError(
                "TAR archive_offset/archive_size must be declared in pairs"
            )

        if archive_offset is not None and archive_size is not None:
            bound = _bound_archive(archive_text)
            key = (
                "tar-indexed-v1",
                bound.path,
                member,
                asset_id,
                asset_revision,
                shard_sha256,
                payload_sha256,
                *bound.identity.cache_key(),
                archive_offset,
                archive_size,
            )

            def read_tar() -> bytes:
                return _pread_exact(
                    bound,
                    offset=archive_offset,
                    size=archive_size,
                )

            payload = _cache_payload(key, read_tar)
            _assert_bound_archive_stable(bound, operation="payload cache read")
        else:
            if asset_metadata is not None:
                raise AudioAssetIntegrityError(
                    "Permanent TAR assets cannot fall back to a tarfile member scan"
                )
            normalized_archive, identity = _archive_path_identity(archive_text)
            key = (
                "tar-member-v1",
                normalized_archive,
                member,
                *identity.cache_key(),
            )

            def read_tar() -> bytes:
                with tarfile.open(normalized_archive, "r:*") as archive:
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise FileNotFoundError(f"tar member does not exist: {member}")
                    result = extracted.read()
                _, final_identity = _archive_path_identity(normalized_archive)
                if final_identity != identity:
                    raise AudioAssetIntegrityError(
                        "TAR asset identity drift while reading a member: "
                        f"{normalized_archive}"
                    )
                return result

            payload = _cache_payload(key, read_tar)
            _, current_identity = _archive_path_identity(normalized_archive)
            if current_identity != identity:
                raise AudioAssetIntegrityError(
                    "TAR payload file identity changed on a cache hit: "
                    f"{normalized_archive}"
                )
        if payload_sha256 is not None:
            actual_payload_sha256 = hashlib.sha256(payload).hexdigest()
            if actual_payload_sha256 != payload_sha256:
                raise AudioAssetIntegrityError(
                    "TAR payload SHA-256 mismatch: "
                    f"asset_id={asset_id} expected={payload_sha256} "
                    f"actual={actual_payload_sha256}"
                )
        return _read_audio_bytes(
            payload, start_sec=start_sec, duration_sec=duration_sec
        )
    if path_text.startswith("zip://"):
        archive_text, member = path_text.removeprefix("zip://").split("::", 1)
        key = (
            f"zip://{archive_text}::{member}::"
            f"credential={_credential_fingerprint(password)}"
        )

        def read_zip() -> bytes:
            archive = _ZIP_CACHE.get(archive_text)
            if archive is None:
                archive = zipfile.ZipFile(archive_text)
                _ZIP_CACHE[archive_text] = archive
            if password is None:
                return archive.read(member)
            try:
                return archive.read(member, pwd=password)
            except Exception as exc:  # noqa: BLE001


                raise EncryptedZipReadError(
                    "Encrypted ZIP member read failed "
                    f"({type(exc).__name__}); check credentials and archive integrity"
                ) from None

        payload = _cache_payload(key, read_zip)
        return _read_audio_bytes(payload, start_sec=start_sec, duration_sec=duration_sec)

    if path_text.startswith("file://"):
        path_text = path_text.removeprefix("file://")
        if not Path(path_text).is_absolute() or "::" in path_text:
            raise ValueError(f"file URI must point to the absolute path: {path!r}")
    path = Path(path_text)
    if path.suffix.lower() == ".wav":
        try:
            return _read_wave(path, start_sec=start_sec, duration_sec=duration_sec)
        except wave.Error:
            pass

    soundfile_error: BaseException
    try:
        import soundfile as sf
    except ImportError as exc:
        soundfile_error = exc
    else:
        try:
            with sf.SoundFile(str(path)) as f:
                f.seek(round(start_sec * f.samplerate))
                frames = (
                    -1
                    if duration_sec is None
                    else round(duration_sec * f.samplerate)
                )
                array = f.read(frames=frames, always_2d=True, dtype="float32")
                sample_rate = f.samplerate
            return torch.from_numpy(array).mean(dim=1), int(sample_rate)
        except Exception as exc:  # noqa: BLE001
            soundfile_error = exc
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"The audio file is unreadable:{path}: {type(exc).__name__}") from exc
    return _read_audio_bytes_ffmpeg(
        payload,
        start_sec=start_sec,
        duration_sec=duration_sec,
        previous_error=soundfile_error,
    )


RESAMPLE_FILTER_WIDTH = 64
RESAMPLE_ROLLOFF = 0.99
RESAMPLE_KAISER_BETA = 14.769656459379492


RESAMPLE_MAX_POLYPHASE = 1024


@lru_cache(maxsize=16)
def _sinc_resample_kernel(
    source_rate: int, target_rate: int
) -> tuple[torch.Tensor, int]:

    divisor = math.gcd(source_rate, target_rate)
    source = source_rate // divisor
    target = target_rate // divisor


    if max(source, target) > RESAMPLE_MAX_POLYPHASE:
        raise ValueError(
            f"unsupported {source_rate} Hz -> {target_rate} Hz resampling: "
            f"reduced ratio {source}/{target} is too large for a practical polyphase "
            "kernel; check the audio sample rate"
        )
    base_freq = min(source, target) * RESAMPLE_ROLLOFF
    width = math.ceil(RESAMPLE_FILTER_WIDTH * source / base_freq)

    idx = torch.arange(-width, width + source, dtype=torch.float64)[None, None] / source
    phases = torch.arange(0, -target, -1, dtype=torch.float64)[:, None, None] / target
    t = ((phases + idx) * base_freq).clamp_(-RESAMPLE_FILTER_WIDTH, RESAMPLE_FILTER_WIDTH)
    window = torch.i0(
        RESAMPLE_KAISER_BETA * torch.sqrt(1 - (t / RESAMPLE_FILTER_WIDTH) ** 2)
    ) / torch.i0(torch.tensor(RESAMPLE_KAISER_BETA, dtype=torch.float64))
    t *= math.pi
    kernel = torch.where(t == 0, torch.ones_like(t), torch.sin(t) / t)
    kernel = kernel * window * (base_freq / source)
    return kernel.to(torch.float32), width


def resample_mono(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:

    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    waveform = waveform.float()
    source_rate, target_rate = int(source_rate), int(target_rate)
    if source_rate == target_rate:
        return waveform
    length = waveform.numel()
    if length == 0:
        return waveform

    kernel, width = _sinc_resample_kernel(source_rate, target_rate)
    divisor = math.gcd(source_rate, target_rate)
    source = source_rate // divisor
    target_length = math.ceil(target_rate * length / source_rate)

    padded = F.pad(waveform.view(1, 1, -1), (width, width + source))
    resampled = F.conv1d(padded, kernel.to(padded), stride=source)
    return resampled.transpose(1, 2).reshape(-1)[:target_length]
