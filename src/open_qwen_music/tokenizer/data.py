
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import math
import os
import random
import secrets
import socket
import stat
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler, Sampler

from .audio import (
    AudioAssetIntegrityError,
    AudioCredentialError,
    EncryptedZipReadError,
    audio_password_env,
    load_audio,
    password_from_env,
    prefetch_parquet_audio,
    probe_audio,
    resample_mono,
    reset_audio_io_state,
)
from .contracts import SAMPLE_RATE
from .features import (
    LogMelFrontend,
    chroma_from_waveform,
    lengths_to_mask,
    resolve_chroma_config,
)
from .frontend import ConvSubsampling25Hz
from .text import CharacterTokenizer


def _get_audio_path(record: dict[str, Any], manifest_dir: Path) -> str:
    path = (
        record.get("audio_path")
        or record.get("audio", {}).get("path")
        or record.get("source", {}).get("uri")
    )
    if path is None:
        raise KeyError(
            f"Sample {record.get('sample_id')} is missing audio_path, audio.path, "
            "or source.uri"
        )
    path_text = str(path)
    if path_text.startswith(("tar://", "zip://")):
        scheme, payload = path_text.split("://", 1)
        archive, member = payload.split("::", 1)
        archive_path = Path(archive)
        if not archive_path.is_absolute():
            archive_path = manifest_dir / archive_path
        return f"{scheme}://{archive_path}::{member}"
    if path_text.startswith("parquet://"):


        return path_text
    if path_text.startswith("file://"):
        file_path = Path(path_text.removeprefix("file://"))
        if not file_path.is_absolute():
            raise ValueError(f"file URI must point to the absolute path: {path_text!r}")
        return f"file://{file_path}"
    path = Path(path)
    return str(path if path.is_absolute() else manifest_dir / path)


def _get_audio_password(
    record: dict[str, Any],
    audio_path: str,
) -> tuple[str | None, bytes | None]:
    password_env = audio_password_env(record)
    if password_env is None:
        return None, None
    if not audio_path.startswith("zip://"):
        raise AudioCredentialError(
            f"Audio credential environment variable {password_env} can be used only with a zip:// audio URI"
        )
    return password_env, password_from_env(password_env)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_MANIFEST_INDEX_SCHEMA = "oqm.manifest-index.v3"
_INDEX_READY_FILE = "READY"
_INDEX_HASH_VERIFY_MODES = ("always", "node_once")
_INDEX_NODE_SENTINEL_SCHEMA = "oqm.index-node-sha-verification.v1"
_INDEX_NODE_CACHE_ROOT = "open-qwen-music-index-verification"
_INDEX_NODE_LOCK_TIMEOUT_ENV = "OQM_INDEX_VERIFY_LOCK_TIMEOUT_SEC"
_INDEX_NODE_LOCK_TIMEOUT_SEC = 1800.0
_INDEX_NODE_SENTINEL_MAX_BYTES = 4 * 1024 * 1024
_IO_GROUP_FIELDS = frozenset({"training.io_group", "audio.io_group"})
_INDEX_REQUIRED_CATEGORICAL_FIELDS = frozenset(
    {
        "training.sampling_group",
        "training.source_sampling_group",
        "training.ctc_group",
        "training.quality_tier",
        "training.io_group",
    }
)


class _IndexSnapshotChanged(RuntimeError):
    pass


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _index_lock_timeout_sec() -> float:
    raw = os.environ.get(_INDEX_NODE_LOCK_TIMEOUT_ENV)
    if raw is None:
        return _INDEX_NODE_LOCK_TIMEOUT_SEC
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"{_INDEX_NODE_LOCK_TIMEOUT_ENV} must be a positive finite number of seconds"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(
            f"{_INDEX_NODE_LOCK_TIMEOUT_ENV} must be a positive finite number of seconds"
        )
    return value


def _private_cache_root() -> Path:

    node_digest = hashlib.sha256(
        socket.gethostname().encode("utf-8", errors="surrogatepass")
    ).hexdigest()[:16]
    root = Path("/tmp") / (
        f"{_INDEX_NODE_CACHE_ROOT}-{os.getuid()}-{node_digest}"
    )
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        pass
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise RuntimeError(f"Unable to access the node index verification cache directory: {root}") from exc
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or root_stat.st_uid != os.getuid()
        or stat.S_IMODE(root_stat.st_mode) != 0o700
    ):
        raise RuntimeError(
            f"The node index cache must be owned by the current user and use mode 0700: {root}"
        )
    return root


def _open_private_lock(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError(f"Unable to open the node index verification lock: {path}") from exc
    lock_stat = os.fstat(descriptor)
    if (
        not stat.S_ISREG(lock_stat.st_mode)
        or lock_stat.st_uid != os.getuid()
        or stat.S_IMODE(lock_stat.st_mode) != 0o600
        or lock_stat.st_nlink != 1
    ):
        os.close(descriptor)
        raise RuntimeError(
            f"The node index lock must be owned by the current user and use mode 0600: {path}"
        )
    return descriptor


@contextmanager
def _exclusive_node_lock(path: Path, *, label: str):
    descriptor = _open_private_lock(path)
    timeout_sec = _index_lock_timeout_sec()
    deadline = time.monotonic() + timeout_sec
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(
                        f"Timed out after {timeout_sec:g}s waiting for node {label} lock: {path}"
                    ) from None
                time.sleep(min(0.05, remaining))
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _read_private_file(path: Path, *, max_bytes: int) -> bytes | None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        file_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_uid != os.getuid()
            or stat.S_IMODE(file_stat.st_mode) != 0o600
            or file_stat.st_nlink != 1
            or file_stat.st_size > max_bytes
        ):
            return None
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        return payload if len(payload) <= max_bytes else None
    finally:
        os.close(descriptor)


def _atomic_write_private(path: Path, payload: bytes) -> None:
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    replaced = False
    try:
        descriptor = os.open(temporary, flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Writing the node index verification file returned zero bytes")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        replaced = True
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if replaced:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def _node_authentication_key(root: Path) -> bytes:
    key_path = root / "authentication.key"
    with _exclusive_node_lock(
        root / "authentication.lock",
        label="index verification authentication key",
    ):
        key = _read_private_file(key_path, max_bytes=32)
        if key is None:
            if key_path.exists() or key_path.is_symlink():
                raise RuntimeError(
                    f"Node index verification authentication key has invalid permissions or content: {key_path}"
                )
            key = secrets.token_bytes(32)
            _atomic_write_private(key_path, key)
        if len(key) != 32:
            raise RuntimeError(f"Node index verification authentication key has an invalid length: {key_path}")
        return key


def _file_binding(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    try:
        file_stat = path.stat()
    except OSError as exc:
        raise RuntimeError(f"Index verification file is unreadable: {path}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise RuntimeError(f"Index verification target is not a regular file: {path}")
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "size": int(file_stat.st_size),
        "mtime_ns": int(file_stat.st_mtime_ns),
        "ctime_ns": int(file_stat.st_ctime_ns),
        "inode": int(file_stat.st_ino),
        "device": int(file_stat.st_dev),
    }


def _node_verification_paths(
    *,
    root: Path,
    metadata_sha256: str,
    manifest_binding: dict[str, Any],
) -> tuple[str, Path, Path]:
    key_payload = {
        "schema_version": _INDEX_NODE_SENTINEL_SCHEMA,
        "metadata_sha256": metadata_sha256,
        "manifest": manifest_binding,
    }
    cache_key = hashlib.sha256(_canonical_json_bytes(key_payload)).hexdigest()
    return (
        cache_key,
        root / f"{cache_key}.lock",
        root / f"{cache_key}.json",
    )


def _node_sentinel_valid(
    path: Path,
    *,
    expected: dict[str, Any],
    authentication_key: bytes,
) -> bool:
    payload = _read_private_file(
        path,
        max_bytes=_INDEX_NODE_SENTINEL_MAX_BYTES,
    )
    if payload is None:
        return False
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(decoded, dict):
        return False
    supplied_mac = decoded.pop("hmac_sha256", None)
    if not _is_sha256(supplied_mac):
        return False
    expected_mac = hmac.new(
        authentication_key,
        _canonical_json_bytes(decoded),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(supplied_mac, expected_mac) and decoded == expected


def _write_node_sentinel(
    path: Path,
    *,
    payload: dict[str, Any],
    authentication_key: bytes,
) -> None:
    encoded_payload = dict(payload)
    encoded_payload["hmac_sha256"] = hmac.new(
        authentication_key,
        _canonical_json_bytes(payload),
        hashlib.sha256,
    ).hexdigest()
    encoded = _canonical_json_bytes(encoded_payload) + b"\n"
    if len(encoded) > _INDEX_NODE_SENTINEL_MAX_BYTES:
        raise RuntimeError(
            "Node index verification sentinel exceeds the safe size limit: "
            f"{len(encoded)} > {_INDEX_NODE_SENTINEL_MAX_BYTES} bytes"
        )
    _atomic_write_private(path, encoded)


def _get_lyrics(record: dict[str, Any]) -> str:
    if "lyrics" in record:
        lyrics = record["lyrics"]


        if not isinstance(lyrics, str):
            raise TypeError(
                f"Sample {record.get('sample_id')} lyrics must be a string; "
                f"received {type(lyrics).__name__}: {lyrics!r}"
            )
        return lyrics
    text = record.get("text") or {}


    if "lyrics" in text:
        lyrics = text["lyrics"]
        if not isinstance(lyrics, str):
            raise TypeError(
                f"Sample {record.get('sample_id')} text.lyrics must be a string; "
                f"received {type(lyrics).__name__}: {lyrics!r}"
            )
        return lyrics
    sections = text.get("sections") or []
    return "\n".join(str(section.get("lyrics", "")) for section in sections).strip()


def _get_ctc_units(record: dict[str, Any]) -> list[str | int] | None:
    units = record.get("ctc_units")
    if units is None:
        units = record.get("text", {}).get("phonemes")
    if units is None:
        return None
    if isinstance(units, str):
        units = units.split()


    if not units:
        return None
    if any(
        isinstance(unit, bool) or not isinstance(unit, (str, int))
        for unit in units
    ):
        raise TypeError(
            f"Sample {record.get('sample_id')} of ctc_units "
            "may contain only strings or integer token IDs"
        )


    return list(units)


def ctc_required_frames(token_ids: list[int]) -> int:

    return len(token_ids) + sum(
        left == right for left, right in zip(token_ids, token_ids[1:])
    )


def _get_ctc_unit_intervals(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    intervals = record.get("ctc_unit_intervals")
    if intervals is None:
        return None
    return [dict(interval) for interval in intervals]


def _subsampled_lengths(feature_lengths: torch.Tensor) -> torch.Tensor:

    return ConvSubsampling25Hz.output_lengths(feature_lengths)


CTC_CROP_MISMATCH_POLICIES = ("error", "disable")




MAX_CONSECUTIVE_DECODE_FAILURES = 64


def _describe_exception(exc: BaseException) -> str:
    try:
        return f"{type(exc).__name__}: {exc}"
    except Exception:  # noqa: BLE001
        return type(exc).__name__


class TokenizerDraw(int):

    epoch: int
    draw_id: int

    def __new__(cls, sample_index: int, epoch: int, draw_id: int) -> TokenizerDraw:
        instance = super().__new__(cls, int(sample_index))
        instance.epoch = int(epoch)
        instance.draw_id = int(draw_id)
        return instance

    def __reduce__(self):

        return type(self), (int(self), self.epoch, self.draw_id)


def _sampler_draw(
    dataset: Dataset,
    sample_index: int,
    *,
    epoch: int,
    draw_id: int,
) -> int:

    if getattr(dataset, "crop_seed", None) is None or not bool(
        getattr(dataset, "random_crop", False)
    ):
        return int(sample_index)
    return TokenizerDraw(sample_index, epoch, draw_id)


class TokenizerDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        *,
        stage: int,
        max_duration_sec: float,
        random_crop: bool,
        crop_seed: int | None = None,
        split: str | None = "train",
        duration_buckets_sec: list[float] | None = None,
        ctc_on_crop_mismatch: str = "error",
        normalize_target_lufs: float | None = None,
        normalize_max_boost_db: float = 12.0,
        normalize_max_peak_dbfs: float | None = None,
        ctc_group_weights: dict[str, float] | None = None,
        ctc_group_default_weight: float = 1.0,
        parquet_row_group_cache_size: int | None = None,
        verify_index_hashes: str = "node_once",
    ) -> None:
        if ctc_on_crop_mismatch not in CTC_CROP_MISMATCH_POLICIES:
            raise ValueError(
                f"ctc_on_crop_mismatch must be one of {CTC_CROP_MISMATCH_POLICIES}; "
                f"received {ctc_on_crop_mismatch!r}"
            )
        self.manifest = Path(manifest)
        self.manifest_dir = self.manifest.parent
        self.stage = stage
        self.max_samples = round(max_duration_sec * SAMPLE_RATE)
        self.random_crop = random_crop
        if crop_seed is not None and (
            isinstance(crop_seed, bool) or not isinstance(crop_seed, int)
        ):
            raise ValueError(
                f"crop_seed must be an integer or null; received {crop_seed!r}"
            )

        self.crop_seed = crop_seed
        self.duration_buckets_sec = sorted(duration_buckets_sec or [])
        self.ctc_on_crop_mismatch = ctc_on_crop_mismatch


        self.normalize_target_lufs = (
            float(normalize_target_lufs) if normalize_target_lufs is not None else None
        )

        self.normalize_max_boost_db = float(normalize_max_boost_db)
        self.normalize_max_peak_dbfs = (
            float(normalize_max_peak_dbfs)
            if normalize_max_peak_dbfs is not None
            else None
        )
        self.ctc_group_weights = {
            str(key): float(value)
            for key, value in (ctc_group_weights or {}).items()
        }
        self.ctc_group_default_weight = float(ctc_group_default_weight)
        if (
            parquet_row_group_cache_size is not None
            and int(parquet_row_group_cache_size) < 0
        ):
            raise ValueError(
                "parquet_row_group_cache_size must be a non-negative integer; "
                f"received {parquet_row_group_cache_size}"
            )
        self.parquet_row_group_cache_size = (
            int(parquet_row_group_cache_size)
            if parquet_row_group_cache_size is not None
            else None
        )
        if verify_index_hashes not in _INDEX_HASH_VERIFY_MODES:
            raise ValueError(
                f"verify_index_hashes must be one of {_INDEX_HASH_VERIFY_MODES}; "
                f"received {verify_index_hashes!r}"
            )
        self.verify_index_hashes = verify_index_hashes
        self._index_verification_sentinel: Path | None = None
        self._index_verification_lock: Path | None = None
        invalid_ctc_weights = {
            key: value
            for key, value in self.ctc_group_weights.items()
            if not math.isfinite(value) or value < 0.0
        }
        if invalid_ctc_weights or not math.isfinite(
            self.ctc_group_default_weight
        ) or self.ctc_group_default_weight < 0.0:
            raise ValueError(
                "CTC group weight must be a non-negative finite number: "
                f"invalid={invalid_ctc_weights} "
                f"default={self.ctc_group_default_weight}"
            )
        self._crop_mismatch_warned = False
        self._empty_crop_warned = False
        self._decode_failures = 0
        self._manifest_handle = None
        self._owner_pid = os.getpid()
        self._index_dir = Path(str(self.manifest.resolve()) + ".index")
        index_present = (
            self._index_dir.exists() or self._index_dir.is_symlink()
        )
        if not index_present and any(
            self._index_dir.parent.glob(
                f".{self._index_dir.name}.backup-*"
            )
        ):
            raise RuntimeError(
                "The manifest index is being published or a previous publish was interrupted; "
                "refusing to read an incomplete index: "
                f"{self._index_dir}"
            )
        self._lazy = index_present
        self.records: list[dict[str, Any]] | None = None
        self._selected_indices = None
        self._durations = None
        self._categorical_indices: dict[str, np.memmap] = {}
        self._categorical_names: dict[str, list[str]] = {}
        if self._lazy:
            self._load_index(split)
        else:
            records = [
                json.loads(line)
                for line in self.manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]


            self.records = (
                [
                    record
                    for record in records
                    if record.get("split", "train") == split
                ]
                if split is not None
                else records
            )
        if len(self) == 0:
            raise ValueError(f"Manifest is empty: {self.manifest}")

    def _read_index_metadata(
        self,
    ) -> tuple[dict[str, Any], str, Path, Path]:
        metadata_path = self._index_dir / "metadata.json"
        ready_path = self._index_dir / _INDEX_READY_FILE
        if not metadata_path.is_file() or not ready_path.is_file():
            raise RuntimeError(
                "The manifest index is missing metadata.json or READY; "
                "refusing to read an incomplete index: "
                f"{self._index_dir}"
            )
        try:
            metadata_bytes = metadata_path.read_bytes()
            ready_digest = ready_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(f"Manifest index is unreadable: {self._index_dir}") from exc
        metadata_digest = hashlib.sha256(metadata_bytes).hexdigest()
        if ready_digest != metadata_digest:
            raise RuntimeError(
                f"Manifest index READY and metadata.json differ: {self._index_dir}"
            )
        try:
            metadata = json.loads(metadata_bytes)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Manifest index metadata is invalid: {metadata_path}"
            ) from exc
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema_version") != _MANIFEST_INDEX_SCHEMA
            or metadata.get("complete") is not True
            or not metadata.get("manifest_sha256")
            or not isinstance(metadata.get("arrays"), dict)
        ):
            raise RuntimeError(
                "Manifest index has an unsupported version or is missing a content hash; "
                f"rebuild it: {self._index_dir}"
            )
        if not _is_sha256(metadata.get("manifest_sha256")):
            raise RuntimeError(
                f"Manifest index manifest_sha256 is invalid; rebuild it: {self._index_dir}"
            )
        return metadata, metadata_digest, metadata_path, ready_path

    @staticmethod
    def _assert_metadata_snapshot(
        metadata_path: Path,
        ready_path: Path,
        metadata_digest: str,
    ) -> None:
        try:
            current_metadata = metadata_path.read_bytes()
            current_ready = ready_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise _IndexSnapshotChanged(
                "metadata.json or READY became unreadable while waiting for the node verification lock"
            ) from exc
        current_digest = hashlib.sha256(current_metadata).hexdigest()
        if current_digest != metadata_digest or current_ready != metadata_digest:
            raise _IndexSnapshotChanged(
                "metadata.json or READY changed during node verification"
            )

    def _verify_stable_sha(
        self,
        path: Path,
        binding: dict[str, Any],
        *,
        label: str,
    ) -> None:
        before = _file_binding(path, expected_sha256=str(binding["sha256"]))
        if before != binding:
            raise RuntimeError(
                f"{label} file identity changed before content verification: {path}"
            )
        actual_sha = _sha256_file(path)
        after = _file_binding(path, expected_sha256=str(binding["sha256"]))
        if before != after:
            raise RuntimeError(f"{label} changed during verification: {path}")
        if actual_sha != str(binding["sha256"]):
            if label == "Manifest":
                raise RuntimeError(
                    f"The manifest SHA-256 does not match the index; rebuild {self._index_dir}"
                )
            raise RuntimeError(f"Manifest index array SHA-256 mismatch: {path}")

    def _verify_all_index_hashes(
        self,
        *,
        manifest_binding: dict[str, Any],
        array_bindings: dict[str, dict[str, Any]],
    ) -> None:
        self._verify_stable_sha(
            Path(str(manifest_binding["path"])),
            manifest_binding,
            label="Manifest",
        )
        for filename in sorted(array_bindings):
            binding = array_bindings[filename]
            self._verify_stable_sha(
                Path(str(binding["path"])),
                binding,
                label=f"index array {filename}",
            )

    def _verify_index_node_once(
        self,
        *,
        metadata_digest: str,
        metadata_path: Path,
        ready_path: Path,
        manifest_binding: dict[str, Any],
        array_bindings: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        root = _private_cache_root()
        authentication_key = _node_authentication_key(root)
        cache_key, lock_path, sentinel_path = _node_verification_paths(
            root=root,
            metadata_sha256=metadata_digest,
            manifest_binding=manifest_binding,
        )
        self._index_verification_lock = lock_path
        self._index_verification_sentinel = sentinel_path
        with _exclusive_node_lock(lock_path, label="manifest/index content SHA verification"):
            self._assert_metadata_snapshot(
                metadata_path,
                ready_path,
                metadata_digest,
            )
            current_manifest = _file_binding(
                Path(str(manifest_binding["path"])),
                expected_sha256=str(manifest_binding["sha256"]),
            )
            if current_manifest != manifest_binding:
                raise _IndexSnapshotChanged(
                    "Manifest file identity changed while waiting for the node verification lock"
                )

            current_arrays: dict[str, dict[str, Any]] = {}
            for filename, initial_binding in sorted(array_bindings.items()):
                current = _file_binding(
                    Path(str(initial_binding["path"])),
                    expected_sha256=str(initial_binding["sha256"]),
                )
                if current["size"] != initial_binding["size"]:
                    raise RuntimeError(
                        "Manifest index array is missing or has an unexpected size: "
                        f"{initial_binding['path']}"
                    )
                current_arrays[filename] = current
            sentinel_payload = {
                "schema_version": _INDEX_NODE_SENTINEL_SCHEMA,
                "cache_key": cache_key,
                "metadata_sha256": metadata_digest,
                "manifest": current_manifest,
                "arrays": current_arrays,
            }
            if _node_sentinel_valid(
                sentinel_path,
                expected=sentinel_payload,
                authentication_key=authentication_key,
            ):
                return current_arrays
            if sentinel_path.exists() or sentinel_path.is_symlink():
                try:
                    sentinel_path.unlink()
                except OSError as exc:
                    raise RuntimeError(
                        f"Unable to remove an invalid node index verification marker: {sentinel_path}"
                    ) from exc

            self._verify_all_index_hashes(
                manifest_binding=current_manifest,
                array_bindings=current_arrays,
            )
            self._assert_metadata_snapshot(
                metadata_path,
                ready_path,
                metadata_digest,
            )
            _write_node_sentinel(
                sentinel_path,
                payload=sentinel_payload,
                authentication_key=authentication_key,
            )
            return current_arrays

    def _load_index(self, split: str | None) -> None:
        last_snapshot_error: _IndexSnapshotChanged | None = None
        for _attempt in range(3):
            try:
                self._load_index_snapshot(split)
                return
            except _IndexSnapshotChanged as exc:
                last_snapshot_error = exc
        assert last_snapshot_error is not None
        raise RuntimeError(
            "Manifest or index kept changing during node verification; retries exhausted: "
            f"{last_snapshot_error}"
        ) from None

    def _load_index_snapshot(self, split: str | None) -> None:
        (
            metadata,
            metadata_digest,
            metadata_path,
            ready_path,
        ) = self._read_index_metadata()

        manifest_path = self.manifest.resolve()
        manifest_binding = _file_binding(
            manifest_path,
            expected_sha256=str(metadata["manifest_sha256"]),
        )
        if int(metadata.get("source_size", -1)) != manifest_binding["size"]:
            raise RuntimeError(
                f"Manifest index source size mismatch; rebuild it: {self._index_dir}"
            )

        try:
            count = int(metadata["records"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Manifest index record count is invalid: {self._index_dir}"
            ) from exc
        if count <= 0:
            raise ValueError(f"Manifest is empty: {self.manifest}")
        arrays: dict[str, Any] = metadata["arrays"]
        validated: dict[str, tuple[Path, np.dtype[Any], int]] = {}
        array_bindings: dict[str, dict[str, Any]] = {}
        for filename, specification in arrays.items():
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not isinstance(specification, dict)
            ):
                raise RuntimeError(
                    f"Manifest index array specification is invalid: {filename!r}"
                )
            try:
                dtype = np.dtype(specification["dtype"])
                length = int(specification["length"])
                expected_size = int(specification["size_bytes"])
                expected_sha = str(specification["sha256"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Manifest index array specification is missing "
                    f"sha256, size_bytes, dtype, or length: {filename}"
                ) from exc
            if (
                length < 0
                or expected_size != length * dtype.itemsize
                or not _is_sha256(expected_sha)
            ):
                raise RuntimeError(
                    f"Manifest index array specification is inconsistent: {filename}"
                )
            path = self._index_dir / filename
            if not path.is_file():
                raise RuntimeError(
                    f"Manifest index array is missing or has an unexpected size: {path}"
                )
            try:
                resolved_path = path.resolve(strict=True)
            except OSError as exc:
                raise RuntimeError(
                    f"Manifest index array path cannot be resolved: {path}"
                ) from exc
            binding = _file_binding(
                resolved_path,
                expected_sha256=expected_sha,
            )
            if binding["size"] != expected_size:
                raise RuntimeError(
                    f"Manifest index array is missing or has an unexpected size: {path}"
                )
            validated[filename] = (resolved_path, dtype, length)
            array_bindings[filename] = binding

        if self.verify_index_hashes == "always":
            self._verify_all_index_hashes(
                manifest_binding=manifest_binding,
                array_bindings=array_bindings,
            )
            self._assert_metadata_snapshot(
                metadata_path,
                ready_path,
                metadata_digest,
            )
            verified_array_bindings = array_bindings
        else:
            verified_array_bindings = self._verify_index_node_once(
                metadata_digest=metadata_digest,
                metadata_path=metadata_path,
                ready_path=ready_path,
                manifest_binding=manifest_binding,
                array_bindings=array_bindings,
            )

        current_manifest = _file_binding(
            manifest_path,
            expected_sha256=str(manifest_binding["sha256"]),
        )
        if current_manifest != manifest_binding:
            raise RuntimeError("Manifest changed after content verification")
        self._assert_metadata_snapshot(
            metadata_path,
            ready_path,
            metadata_digest,
        )
        for filename, binding in verified_array_bindings.items():
            current = _file_binding(
                Path(str(binding["path"])),
                expected_sha256=str(binding["sha256"]),
            )
            if current != binding:
                raise RuntimeError(
                    f"Index array changed after content verification: {filename}"
                )

        required = {
            "offsets.i64": np.dtype("<i8"),
            "durations.f32": np.dtype("<f4"),
            "splits.u16": np.dtype("<u2"),
        }
        mapped: dict[str, np.memmap] = {}
        for filename, expected_dtype in required.items():
            if filename not in validated:
                raise RuntimeError(f"Manifest index missing array: {filename}")
            path, dtype, length = validated[filename]
            if dtype != expected_dtype or length != count:
                raise RuntimeError(
                    f"Manifest index array has an invalid dtype or length: {filename} "
                    f"dtype={dtype} length={length} records={count}"
                )
            mapped[filename] = np.memmap(
                path, mode="r", dtype=dtype, shape=(count,)
            )

        offsets = mapped["offsets.i64"]
        self._durations = mapped["durations.f32"]
        split_ids = mapped["splits.u16"]
        split_names = list(metadata.get("split_names") or [])
        if (
            not all(isinstance(name, str) for name in split_names)
            or not split_names
            or int(split_ids.max()) >= len(split_names)
        ):
            raise RuntimeError(
                f"Manifest index split names and IDs differ: {self._index_dir}"
            )
        if split is None:
            selected = np.arange(count, dtype=np.int64)
        elif split not in split_names:
            selected = np.empty(0, dtype=np.int64)
        else:
            selected = np.flatnonzero(
                split_ids == split_names.index(split)
            ).astype(np.int64)
        self._offsets = offsets
        self._selected_indices = selected

        categorical_fields = metadata.get("categorical_fields")
        if (
            not isinstance(categorical_fields, dict)
            or not _INDEX_REQUIRED_CATEGORICAL_FIELDS.issubset(
                categorical_fields
            )
        ):
            raise RuntimeError(
                f"Manifest index categorical_fields is invalid: {self._index_dir}"
            )
        for field, specification in categorical_fields.items():
            if not isinstance(specification, dict):
                raise RuntimeError(
                    f"Manifest index categorical field is invalid: {field}"
                )
            filename = specification.get("file")
            names = specification.get("names")
            if (
                not isinstance(filename, str)
                or filename not in validated
                or not isinstance(names, list)
                or not all(isinstance(name, str) for name in names)
            ):
                raise RuntimeError(
                    f"Manifest index categorical field specification is invalid: {field}"
                )
            path, dtype, length = validated[filename]
            if dtype != np.dtype("<u2") or length != count:
                raise RuntimeError(
                    f"Manifest index categorical array has an invalid dtype or length: {field}"
                )
            values = np.memmap(
                path, mode="r", dtype=dtype, shape=(count,)
            )
            if not names or int(values.max()) >= len(names):
                raise RuntimeError(
                    f"Manifest index category names and IDs differ for {field}"
                )
            self._categorical_indices[str(field)] = values
            self._categorical_names[str(field)] = list(names)

    def __len__(self) -> int:
        return (
            len(self._selected_indices)
            if self._lazy
            else len(self.records or [])
        )

    def _record(self, index: int) -> dict[str, Any]:
        self._ensure_process_state()
        if not self._lazy:
            return self.records[index]  # type: ignore[index]
        if self._manifest_handle is None:
            self._manifest_handle = self.manifest.open("rb")
        global_index = int(self._selected_indices[index])
        self._manifest_handle.seek(int(self._offsets[global_index]))
        return json.loads(self._manifest_handle.readline())

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_manifest_handle"] = None
        state["_owner_pid"] = None
        return state

    def _ensure_process_state(self) -> None:

        pid = os.getpid()
        if self._owner_pid == pid:
            return
        if self._manifest_handle is not None:
            try:
                self._manifest_handle.close()
            except Exception:  # noqa: BLE001
                pass
        self._manifest_handle = None
        reset_audio_io_state()
        self._owner_pid = pid

    def reset_manifest_handle(self) -> None:
        if self._manifest_handle is not None:
            self._manifest_handle.close()
            self._manifest_handle = None

    def duration_sec(self, index: int) -> float:
        if self._lazy:
            global_index = int(self._selected_indices[index])
            declared = float(self._durations[global_index])


            if declared <= 0.0:
                return self.max_samples / SAMPLE_RATE
            return min(declared, self.max_samples / SAMPLE_RATE)
        record = self.records[index]  # type: ignore[index]
        declared = record.get("audio", {}).get("duration_sec")
        if declared is None:
            return self.max_samples / SAMPLE_RATE
        return min(float(declared), self.max_samples / SAMPLE_RATE)

    def padding_duration_sec(self, index: int) -> float | None:
        if not self.duration_buckets_sec:
            return None
        duration = self.duration_sec(index)
        for boundary in self.duration_buckets_sec:
            if duration <= boundary:
                return boundary
        return self.max_samples / SAMPLE_RATE

    def sampling_value(
        self, index: int, field: str, default: str = "unknown"
    ) -> str:
        indexed_field = (
            "training.io_group" if field in _IO_GROUP_FIELDS else field
        )
        if self._lazy and indexed_field in self._categorical_indices:
            global_index = int(self._selected_indices[index])
            value_id = int(
                self._categorical_indices[indexed_field][global_index]
            )
            names = self._categorical_names[indexed_field]
            return names[value_id] if value_id < len(names) else default
        record = self._record(index)
        candidates = (
            (field,)
            if field not in _IO_GROUP_FIELDS
            else (field, *sorted(_IO_GROUP_FIELDS - {field}))
        )
        for candidate in candidates:
            value: Any = record
            for key in candidate.split("."):
                if not isinstance(value, dict) or key not in value:
                    break
                value = value[key]
            else:
                if value is not None and str(value):
                    return str(value)
        return default

    def _apply_loudness_gain(
        self, waveform: torch.Tensor, record: dict[str, Any], audio_meta: dict[str, Any]
    ) -> torch.Tensor:

        if self.normalize_target_lufs is None:
            return waveform
        lufs = audio_meta.get("lufs_i")
        if lufs is None:
            raise ValueError(
                f"Sample {record.get('sample_id')} is missing audio.lufs_i, but the configuration enables "
                f"data.normalize_target_lufs={self.normalize_target_lufs}. "
                "Measure loudness with the offline validation tool and merge it "
                "into the manifest first."
            )
        gain_db = min(
            self.normalize_target_lufs - float(lufs), self.normalize_max_boost_db
        )
        if self.normalize_max_peak_dbfs is not None:
            peak = audio_meta.get("peak_full")
            if peak is None or float(peak) <= 0:
                raise ValueError(
                    f"Sample {record.get('sample_id')} is missing audio.peak_full, but the configuration enables "
                    f"data.normalize_max_peak_dbfs={self.normalize_max_peak_dbfs}."
                )
            peak_guard_db = self.normalize_max_peak_dbfs - 20.0 * math.log10(
                float(peak)
            )
            gain_db = min(gain_db, peak_guard_db)
        if abs(gain_db) < 1e-6:
            return waveform
        return waveform * float(10.0 ** (gain_db / 20.0))

    def _crop_fraction(
        self,
        record: dict[str, Any],
        draw: TokenizerDraw | None,
    ) -> float:
        if self.crop_seed is None:
            return random.random()
        if draw is None:
            raise RuntimeError(
                "Deterministic cropping is enabled with data.crop_seed, but the dataset "
                "received a plain integer index. Use the resumable sampler so each draw "
                "includes a TokenizerDraw identity."
            )
        sample_uid = next(
            (
                str(record[field])
                for field in ("sample_uid", "sample_id", "uid", "id")
                if record.get(field) is not None and str(record[field])
            ),
            None,
        )
        if sample_uid is None:
            raise ValueError(
                "Deterministic cropping requires a non-empty "
                "sample_uid, sample_id, uid, or id in the manifest"
            )
        payload = {
            "schema_version": "oqm.tokenizer-deterministic-crop.v1",
            "crop_seed": self.crop_seed,
            "stage": self.stage,
            "sample_uid": sample_uid,
            "epoch": draw.epoch,
            "draw_id": draw.draw_id,
        }
        digest = hashlib.sha256(_canonical_json_bytes(payload)).digest()

        return (int.from_bytes(digest[:8], "big") >> 11) / float(1 << 53)

    def __getitems__(
        self, indices: list[int | TokenizerDraw]
    ) -> list[dict[str, Any]]:

        self._ensure_process_state()
        draws = [
            index if isinstance(index, TokenizerDraw) else int(index)
            for index in indices
        ]
        materialized = [int(index) for index in draws]
        records = [self._record(index) for index in materialized]
        paths = [
            _get_audio_path(record, self.manifest_dir)
            for record in records
        ]

        for record, path in zip(records, paths):
            _get_audio_password(record, path)
        prefetch_parquet_audio(
            paths,
            row_group_cache_size=self.parquet_row_group_cache_size,
        )
        return [self.__getitem__(index) for index in draws]

    def __getitem__(self, index: int | TokenizerDraw) -> dict[str, Any]:
        self._ensure_process_state()
        draw = index if isinstance(index, TokenizerDraw) else None
        index = int(index)
        record = self._record(index)
        audio_path = _get_audio_path(record, self.manifest_dir)
        audio_meta = record.get("audio", {})
        permanent_asset = any(
            audio_meta.get(field) is not None
            for field in (
                "asset_id",
                "payload_sha256",
                "asset_revision",
                "shard_sha256",
            )
        )
        password_env, password = _get_audio_password(record, audio_path)
        declared_start = float(audio_meta.get("start_sec", 0.0))
        declared_duration = audio_meta.get("duration_sec")
        if declared_duration is None and not audio_path.startswith(
            ("tar://", "zip://", "parquet://")
        ):
            declared_duration = probe_audio(audio_path).duration_sec - declared_start
        declared_duration = float(declared_duration) if declared_duration is not None else None
        crop_duration = min(
            self.max_samples / SAMPLE_RATE,
            declared_duration if declared_duration is not None else self.max_samples / SAMPLE_RATE,
        )
        crop_start = declared_start
        if (
            self.random_crop
            and declared_duration is not None
            and declared_duration > crop_duration
        ):
            crop_start += self._crop_fraction(record, draw) * (
                declared_duration - crop_duration
            )

        def _try_decode(start_sec: float):
            try:
                candidate, rate = load_audio(
                    audio_path,
                    start_sec=start_sec,
                    duration_sec=crop_duration,
                    archive_offset=audio_meta.get("archive_offset"),
                    archive_size=audio_meta.get("archive_size"),
                    parquet_row_group_cache_size=(
                        self.parquet_row_group_cache_size
                    ),
                    password=password,
                    asset_id=audio_meta.get("asset_id"),
                    payload_sha256=audio_meta.get("payload_sha256"),
                    asset_revision=audio_meta.get("asset_revision"),
                    shard_sha256=audio_meta.get("shard_sha256"),
                )
            except EncryptedZipReadError:
                assert password_env is not None
                raise AudioCredentialError(
                    f"Sample {record.get('sample_id', index)} ZIP member could not "
                    f"be read; check environment variable {password_env} and archive integrity"
                ) from None
            except AudioCredentialError:
                raise
            except AudioAssetIntegrityError as exc:
                if permanent_asset:


                    raise
                return None, _describe_exception(exc)
            except Exception as exc:  # noqa: BLE001
                if permanent_asset:
                    raise AudioAssetIntegrityError(
                        "Permanent audio asset decoding failed:"
                        f"{_describe_exception(exc)}"
                    ) from exc


                return None, _describe_exception(exc)
            if candidate.numel() == 0:
                if permanent_asset:
                    raise AudioAssetIntegrityError(
                        "Permanent audio asset decoder returned zero samples"
                    )
                return None, "decoder returned zero samples"
            return (candidate, rate), None

        decoded, failure = _try_decode(crop_start)
        if decoded is None and crop_start > declared_start:


            retried, retry_failure = _try_decode(declared_start)
            if retried is not None:
                crop_start = declared_start
                decoded, failure = retried, None
                if not self._empty_crop_warned:
                    self._empty_crop_warned = True
                    warnings.warn(
                        "Random crop start exceeded the actual end of the audio because "
                        "the manifest duration was too large; retrying from start_sec. "
                        "First sample_id="
                        f"{record.get('sample_id')} crop_start={crop_start:.3f} "
                        f"declared_duration={declared_duration}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            else:
                failure = f"{failure}; retry from start_sec also failed: {retry_failure}"

        if decoded is None:


            self._decode_failures += 1
            detail = (
                f"sample_id={record.get('sample_id')} path={audio_path} "
                f"start={crop_start} duration={crop_duration}: {failure}"
            )
            if self._decode_failures > MAX_CONSECUTIVE_DECODE_FAILURES:


                raise RuntimeError(
                    f"{self._decode_failures} consecutive samples failed to decode; "
                    f"a storage or archive failure is likely. Last failure: {detail}"
                )
            warnings.warn(
                f"Sample decode failed; replacing audio with silence and disabling its "
                f"supervision (consecutive failure {self._decode_failures}): {detail}",
                RuntimeWarning,
                stacklevel=2,
            )
            padding_sec = self.padding_duration_sec(index)
            placeholder_samples = round(
                (padding_sec if padding_sec is not None else crop_duration)
                * SAMPLE_RATE
            )
            placeholder_samples = max(
                960, min(placeholder_samples, self.max_samples)
            )
            return {
                "sample_id": str(record.get("sample_id", index)),
                "waveform": torch.zeros(placeholder_samples),
                "lyrics": "",
                "ctc_units": None,
                "ctc_unit_intervals": None,
                "language": str(record.get("text", {}).get("language", "unknown")),
                "source_dataset": str(
                    record.get("source", {}).get("dataset", "unknown")
                ),
                "sampling_group": str(
                    record.get("training", {}).get(
                        "sampling_group",
                        record.get("source", {}).get("dataset", "unknown"),
                    )
                ),
                "is_synthetic": bool(
                    record.get("source", {}).get("is_synthetic", False)
                ),

                "sample_weight": 0.0,
                "ctc_sample_weight": 0.0,
                "mel_sample_weight": 0.0,
                "chroma_sample_weight": 0.0,
                "vq_sample_weight": 0.0,
                "pad_to_num_samples": (
                    round(padding_sec * SAMPLE_RATE)
                    if padding_sec is not None
                    else None
                ),
                "ctc_enabled": False,
                "mel_enabled": False,
                "chroma_enabled": False,
                "vq_enabled": False,


                "decode_failed": True,
            }
        self._decode_failures = 0
        waveform, source_rate = decoded
        waveform = resample_mono(waveform, source_rate, SAMPLE_RATE)
        waveform = self._apply_loudness_gain(waveform, record, audio_meta)




        crop_unknown = declared_duration is None
        crop_truncated = (
            declared_duration is not None
            and declared_duration > crop_duration + 1e-3
        ) or waveform.numel() > self.max_samples
        if waveform.numel() > self.max_samples:
            waveform = waveform[: self.max_samples]
        if waveform.numel() < 960:
            waveform = torch.nn.functional.pad(waveform, (0, 960 - waveform.numel()))

        training = record.get("training") or {}
        loss_heads = training.get("loss_heads") or {}
        lyrics = _get_lyrics(record)
        ctc_units = _get_ctc_units(record)
        ctc_unit_intervals = _get_ctc_unit_intervals(record)
        if ctc_unit_intervals is not None:
            crop_offset = crop_start - declared_start
            adjusted_intervals = []
            crop_end = crop_offset + waveform.numel() / SAMPLE_RATE
            for interval in ctc_unit_intervals:
                start = float(interval["start_sec"])
                end = float(interval["end_sec"])
                if end <= crop_offset or start >= crop_end:
                    continue
                adjusted_intervals.append(
                    {
                        "unit": str(interval["unit"]),
                        "start_sec": max(0.0, start - crop_offset),
                        "end_sec": min(crop_end, end) - crop_offset,
                    }
                )
            ctc_unit_intervals = adjusted_intervals
        ctc_enabled = bool(loss_heads.get("ctc", bool(ctc_units or lyrics)))

        if self.stage >= 3 and ctc_enabled and crop_unknown:
            raise ValueError(
                f"Sample {record.get('sample_id', index)} uses an archive URI, but the manifest has no "
                "audio.duration_sec, so the loader cannot determine whether the audio "
                "was trimmed "
                f"{self.max_samples / SAMPLE_RATE}s. The CTC target covers the full "
                "declared segment, so cropping would make it inconsistent with the "
                "audio. Add duration_sec to the manifest or disable CTC supervision."
            )
        if self.stage >= 3 and ctc_enabled and crop_truncated:
            sample_id = record.get("sample_id", index)
            detail = (
                f"sample_id={sample_id} declared {declared_duration}s but was cropped to "
                f"{waveform.numel() / SAMPLE_RATE:.3f}s"
                f" (max_duration_sec={self.max_samples / SAMPLE_RATE}, "
                f"random_crop={self.random_crop})"
            )
            if self.ctc_on_crop_mismatch == "error":
                raise ValueError(
                    f"CTC sample was cropped, so its lyrics or phoneme target no longer "
                    f"matches the audio: {detail}. Segment the source consistently or set "
                    "data.ctc_on_crop_mismatch=disable."
                )
            ctc_enabled = False
            if not self._crop_mismatch_warned:
                self._crop_mismatch_warned = True
                warnings.warn(
                    f"CTC supervision was disabled for a cropped sample (first occurrence: {detail})",
                    RuntimeWarning,
                    stacklevel=2,
                )
        sampling_group = str(
            training.get(
                "sampling_group",
                training.get(
                    "source_sampling_group",
                    record.get("source", {}).get("dataset", "unknown"),
                ),
            )
        )
        try:
            sample_weight = float(training.get("sample_weight", 1.0))
            ctc_weight = float(training.get("ctc_weight", sample_weight))
            mel_weight = float(training.get("mel_weight", sample_weight))
            chroma_weight = float(training.get("chroma_weight", sample_weight))
            vq_weight = float(training.get("vq_weight", sample_weight))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Sample {record.get('sample_id')} must be a non-negative finite number"
            ) from exc


        ctc_group = str(training.get("ctc_group", sampling_group))
        ctc_group_weight = self.ctc_group_weights.get(
            ctc_group, self.ctc_group_default_weight
        )
        weights = {
            "sample_weight": sample_weight,
            "ctc_weight": ctc_weight,
            "mel_weight": mel_weight,
            "chroma_weight": chroma_weight,
            "vq_weight": vq_weight,
        }
        invalid_weights = {
            name: value
            for name, value in weights.items()
            if not math.isfinite(value) or value < 0.0
        }
        if invalid_weights:
            raise ValueError(
                f"Sample {record.get('sample_id')} weight must be a non-negative finite number: "
                f"{invalid_weights}"
            )
        mel_enabled = bool(loss_heads.get("mel", True))
        chroma_enabled = bool(loss_heads.get("chroma", True))


        vq_enabled = bool(loss_heads.get("vq", True))
        ctc_sample_weight = (
            ctc_weight * ctc_group_weight if ctc_enabled else 0.0
        )
        return {
            "sample_id": str(record.get("sample_id", index)),
            "waveform": waveform,
            "lyrics": lyrics,
            "ctc_units": ctc_units,
            "ctc_unit_intervals": ctc_unit_intervals,
            "language": str(record.get("text", {}).get("language", "unknown")),
            "source_dataset": str(record.get("source", {}).get("dataset", "unknown")),
            "sampling_group": sampling_group,
            "is_synthetic": bool(
                record.get("source", {}).get("is_synthetic", False)
            ),
            "sample_weight": sample_weight,
            "ctc_sample_weight": ctc_sample_weight,
            "mel_sample_weight": mel_weight if mel_enabled else 0.0,
            "chroma_sample_weight": (
                chroma_weight if chroma_enabled else 0.0
            ),
            "vq_sample_weight": vq_weight if vq_enabled else 0.0,
            "pad_to_num_samples": (
                round(self.padding_duration_sec(index) * SAMPLE_RATE)
                if self.padding_duration_sec(index) is not None
                else None
            ),
            "ctc_enabled": ctc_enabled,
            "mel_enabled": mel_enabled,
            "chroma_enabled": chroma_enabled,
            "vq_enabled": vq_enabled,
            "decode_failed": False,
        }


class ResumableDistributedSampler(DistributedSampler):

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size_per_rank: int,
        seed: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(dataset, seed=seed, **kwargs)
        self._batch_size_per_rank = max(1, int(batch_size_per_rank))
        self.skip_batches = 0

    def set_skip_batches(self, count: int) -> None:
        self.skip_batches = max(0, int(count))

    def _skipped_samples(self) -> int:
        return self.skip_batches * self._batch_size_per_rank

    def __iter__(self):
        indices = list(super().__iter__())
        draws = [
            _sampler_draw(
                self.dataset,
                index,
                epoch=self.epoch,
                draw_id=local_position * self.num_replicas + self.rank,
            )
            for local_position, index in enumerate(indices)
        ]
        return iter(draws[self._skipped_samples() :])

    def __len__(self) -> int:
        return max(0, super().__len__() - self._skipped_samples())


class DistributedDurationBucketBatchSampler(Sampler[list[int]]):

    def __init__(
        self,
        dataset: TokenizerDataset,
        *,
        batch_size_per_rank: int,
        rank: int,
        world_size: int,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        if not dataset.duration_buckets_sec:
            raise ValueError("duration bucket sampler requires duration_buckets_sec")
        self.dataset = dataset
        self.batch_size = batch_size_per_rank
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.skip_batches = 0
        self._length = self._count_batches()

    def _bucket_key(self, index: int) -> float:
        return float(self.dataset.padding_duration_sec(index))

    def _grouped_indices(self) -> dict[float, list[int]]:
        groups: dict[float, list[int]] = {}
        for index in range(len(self.dataset)):
            groups.setdefault(self._bucket_key(index), []).append(index)
        return groups

    def _count_batches(self) -> int:
        global_batch = self.batch_size * self.world_size
        return sum(
            math.ceil(len(indices) / global_batch)
            for indices in self._grouped_indices().values()
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_skip_batches(self, count: int) -> None:

        self.skip_batches = max(0, int(count))

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        global_batch = self.batch_size * self.world_size
        global_batches: list[list[int]] = []
        for _, indices in sorted(self._grouped_indices().items()):
            indices = list(indices)
            if self.shuffle:
                generator.shuffle(indices)
            target = math.ceil(len(indices) / global_batch) * global_batch
            if len(indices) < target:
                original = list(indices)
                repeats = target - len(indices)
                indices.extend(
                    original[index % len(original)] for index in range(repeats)
                )
            for start in range(0, len(indices), global_batch):
                global_batches.append(indices[start : start + global_batch])
        if self.shuffle:
            generator.shuffle(global_batches)
        local_start = self.rank * self.batch_size
        for batch_index, batch in enumerate(global_batches):
            if batch_index < self.skip_batches:
                continue
            yield [
                _sampler_draw(
                    self.dataset,
                    batch[slot],
                    epoch=self.epoch,
                    draw_id=batch_index * global_batch + slot,
                )
                for slot in range(local_start, local_start + self.batch_size)
            ]

    def __len__(self) -> int:
        return self._length


class DistributedBalancedDurationBucketBatchSampler(Sampler[list[int]]):

    def __init__(
        self,
        dataset: TokenizerDataset,
        *,
        batch_size_per_rank: int,
        rank: int,
        world_size: int,
        seed: int,
        balance_key: str,
        weights: dict[str, float],
        default_weight: float = 0.0,
        shuffle: bool = True,
        bucket_by_duration: bool = True,
        locality_key: str | None = None,
    ) -> None:
        if bucket_by_duration and not dataset.duration_buckets_sec:
            raise ValueError("balanced duration sampler requires duration_buckets_sec")
        if not balance_key:
            raise ValueError("balanced duration sampler requires balance_key")
        if not weights and default_weight <= 0:
            raise ValueError("balanced duration sampler requires at least one positive weight")
        if any(float(value) < 0 for value in weights.values()):
            raise ValueError("balanced sampler weights cannot be negative")
        self.dataset = dataset
        self.batch_size = batch_size_per_rank
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.balance_key = balance_key
        self.weights = {str(key): float(value) for key, value in weights.items()}
        self.default_weight = float(default_weight)
        self.shuffle = shuffle
        self.bucket_by_duration = bool(bucket_by_duration)
        self.locality_key = str(locality_key) if locality_key else None
        self.epoch = 0
        self.skip_batches = 0
        self._groups = self._grouped_indices()
        present = {
            key
            for by_key in self._groups.values()
            for key, indices in by_key.items()
            if indices
        }
        absent = sorted(
            key
            for key, weight in self.weights.items()
            if weight > 0 and key not in present
        )
        if absent:
            raise ValueError(
                "Balanced sampler assigns positive weights to groups with no samples: "
                f"{absent}. Add data for those groups or remove their weight entries. "
                f"Available groups: {sorted(present)}"
            )


        reset_manifest_handle = getattr(self.dataset, "reset_manifest_handle", None)
        if reset_manifest_handle is not None:
            reset_manifest_handle()
        self._length = sum(
            math.ceil(
                sum(
                    len(indices)
                    for key, indices in by_key.items()
                    if self._weight(key) > 0
                )
                / (self.batch_size * self.world_size)
            )
            for by_key in self._groups.values()
        )

    def _grouped_indices(self) -> dict[float, dict[str, list[int]]]:
        groups: dict[float, dict[str, list[int]]] = {}
        for index in range(len(self.dataset)):
            bucket = (
                float(self.dataset.padding_duration_sec(index))
                if self.bucket_by_duration
                else 0.0
            )
            key = self.dataset.sampling_value(
                index, self.balance_key, default="unknown"
            )
            groups.setdefault(bucket, {}).setdefault(key, []).append(index)
        return groups

    def _weight(self, key: str) -> float:
        return self.weights.get(key, self.default_weight)

    def _order_pool_by_locality(
        self, values: list[int], generator: random.Random
    ) -> list[int]:

        assert self.locality_key is not None
        blocks: dict[str, list[int]] = {}
        for index in values:
            locality = self.dataset.sampling_value(
                index, self.locality_key, default="unknown"
            )
            blocks.setdefault(locality, []).append(index)
        block_keys = sorted(blocks)
        if self.shuffle:
            for key in block_keys:
                generator.shuffle(blocks[key])
            generator.shuffle(block_keys)
        return [
            index
            for key in block_keys
            for index in blocks[key]
        ]

    @staticmethod
    def _allocate_counts(
        keys: list[str],
        weights: list[float],
        total: int,
        credit: dict[str, float] | None = None,
    ) -> dict[str, int]:

        weight_sum = sum(weights)
        if weight_sum <= 0:
            raise ValueError(f"current duration bucket has no positive weight group: {keys}")
        shares = [weight / weight_sum * total for weight in weights]
        if credit is None:
            demand = shares
        else:
            demand = [credit.get(key, 0.0) + share for key, share in zip(keys, shares)]


        counts = [max(0, math.floor(value)) for value in demand]
        order = sorted(
            range(len(keys)),
            key=lambda index: (demand[index] - math.floor(demand[index]), keys[index]),
            reverse=True,
        )


        remainder = total - sum(counts)
        position = 0
        while remainder > 0:
            counts[order[position % len(order)]] += 1
            remainder -= 1
            position += 1
        while remainder < 0:
            for index in reversed(order):
                if remainder == 0:
                    break
                if counts[index] > 0:
                    counts[index] -= 1
                    remainder += 1
        if credit is not None:


            for key, want, got in zip(keys, demand, counts):
                credit[key] = want - got
        assert sum(counts) == total and all(count >= 0 for count in counts)
        return dict(zip(keys, counts))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_skip_batches(self, count: int) -> None:

        self.skip_batches = max(0, int(count))

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        global_batch_size = self.batch_size * self.world_size
        global_batches: list[list[int]] = []
        for _, source_groups in sorted(self._groups.items()):
            available = sorted(
                key
                for key, indices in source_groups.items()
                if indices and self._weight(key) > 0
            )
            if not available:
                raise ValueError(
                    f"bucket has no sample group for balance_key={self.balance_key}"
                )
            weights = [self._weight(key) for key in available]
            pools = {key: list(source_groups[key]) for key in available}
            if self.locality_key is None:

                for values in pools.values():
                    if self.shuffle:
                        generator.shuffle(values)
            else:
                pools = {
                    key: self._order_pool_by_locality(values, generator)
                    for key, values in pools.items()
                }
            cursors = {key: 0 for key in available}


            total_samples = sum(
                len(source_groups[key]) for key in available
            )
            num_batches = math.ceil(total_samples / global_batch_size)
            credit: dict[str, float] = {}
            for _ in range(num_batches):
                allocation = self._allocate_counts(
                    available, weights, global_batch_size, credit
                )
                batch: list[int] = []
                for key in available:
                    need = allocation[key]
                    for _ in range(need):
                        pool = pools[key]
                        cursor = cursors[key]
                        if cursor >= len(pool):
                            cursor = 0
                            if self.locality_key is not None:
                                pool[:] = self._order_pool_by_locality(
                                    pool, generator
                                )
                            elif self.shuffle:
                                generator.shuffle(pool)
                        batch.append(pool[cursor])
                        cursors[key] = cursor + 1


                if self.shuffle and self.locality_key is None:
                    generator.shuffle(batch)
                global_batches.append(batch)
        if self.shuffle and self.locality_key is None:
            generator.shuffle(global_batches)
        local_start = self.rank * self.batch_size
        for batch_index, batch in enumerate(global_batches):
            if batch_index < self.skip_batches:
                continue
            yield [
                _sampler_draw(
                    self.dataset,
                    batch[slot],
                    epoch=self.epoch,
                    draw_id=batch_index * global_batch_size + slot,
                )
                for slot in range(local_start, local_start + self.batch_size)
            ]

    def __len__(self) -> int:
        return self._length


class TokenizerCollator:
    def __init__(
        self,
        *,
        stage: int,
        feature_config: dict[str, Any],
        text_tokenizer: CharacterTokenizer,
        pad_to_num_samples: int | None = None,
        chroma_config: dict[str, Any] | None = None,
        mel_target_mode: str = "frontend",
    ) -> None:
        if mel_target_mode not in {"frontend", "power"}:
            raise ValueError(
                "mel_target_mode must be 'frontend' or 'power'; "
                f"received {mel_target_mode!r}"
            )
        self.stage = stage
        self.feature_extractor = LogMelFrontend(**feature_config)
        self.text_tokenizer = text_tokenizer
        self.pad_to_num_samples = pad_to_num_samples
        self.mel_target_mode = mel_target_mode


        self.chroma_config = resolve_chroma_config(chroma_config)

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        lengths = torch.tensor([sample["waveform"].numel() for sample in samples], dtype=torch.long)
        maximum = int(lengths.max().item())
        sample_padding = [
            int(sample["pad_to_num_samples"])
            for sample in samples
            if sample.get("pad_to_num_samples") is not None
        ]
        if sample_padding:
            if len(set(sample_padding)) != 1:
                raise ValueError(
                    f"a batch contains multiple duration buckets: {sorted(set(sample_padding))}"
                )
            maximum = max(maximum, sample_padding[0])
        if self.pad_to_num_samples is not None:
            if maximum > self.pad_to_num_samples:
                raise ValueError(
                    f"sample length {maximum} exceeds fixed padding length "
                    f"{self.pad_to_num_samples}"
                )
            maximum = self.pad_to_num_samples
        waveform = torch.zeros(len(samples), maximum)
        for row, sample in enumerate(samples):
            waveform[row, : sample["waveform"].numel()] = sample["waveform"]
        batch: dict[str, Any] = {
            "waveform": waveform,
            "waveform_num_samples": lengths,
            "audio_mask": lengths_to_mask(lengths, maximum),
            "sample_ids": [sample["sample_id"] for sample in samples],
            "lyrics_texts": [sample["lyrics"] for sample in samples],
            "languages": [sample["language"] for sample in samples],
            "source_datasets": [sample["source_dataset"] for sample in samples],
            "sampling_groups": [
                sample.get("sampling_group", sample.get("source_dataset", "unknown"))
                for sample in samples
            ],
            "is_synthetic": torch.tensor(
                [sample.get("is_synthetic", False) for sample in samples],
                dtype=torch.bool,
            ),
            "sample_weights": torch.tensor(
                [sample.get("sample_weight", 1.0) for sample in samples],
                dtype=torch.float32,
            ),
            "ctc_sample_weights": torch.tensor(
                [
                    sample.get(
                        "ctc_sample_weight",
                        sample.get("sample_weight", 1.0),
                    )
                    for sample in samples
                ],
                dtype=torch.float32,
            ),
            "mel_sample_weights": torch.tensor(
                [
                    sample.get(
                        "mel_sample_weight",
                        sample.get("sample_weight", 1.0),
                    )
                    for sample in samples
                ],
                dtype=torch.float32,
            ),
            "chroma_sample_weights": torch.tensor(
                [
                    sample.get(
                        "chroma_sample_weight",
                        sample.get("sample_weight", 1.0),
                    )
                    for sample in samples
                ],
                dtype=torch.float32,
            ),
            "vq_sample_weights": torch.tensor(
                [
                    sample.get(
                        "vq_sample_weight",
                        sample.get("sample_weight", 1.0),
                    )
                    for sample in samples
                ],
                dtype=torch.float32,
            ),


            "decode_failed": torch.tensor(
                [bool(sample.get("decode_failed", False)) for sample in samples],
                dtype=torch.bool,
            ),
            "vq_enabled": torch.tensor(
                [bool(sample.get("vq_enabled", True)) for sample in samples],
                dtype=torch.bool,
            ),
        }
        if self.stage < 3:
            return batch

        with torch.no_grad():
            mel = (
                self.feature_extractor.power_mel(waveform)
                if self.mel_target_mode == "power"
                else self.feature_extractor(waveform)
            )
            feature_lengths = self.feature_extractor.lengths(lengths).clamp_max(mel.shape[1])
            chroma = chroma_from_waveform(
                waveform,
                sample_rate=int(self.feature_extractor.sample_rate),
                n_fft=int(self.chroma_config["n_fft"]),
                hop_length=int(self.feature_extractor.hop_length),
                mode=str(self.chroma_config["mode"]),
            )
        encoded = [
            (
                self.text_tokenizer.encode_units(sample["ctc_units"])
                if sample["ctc_units"] is not None
                else self.text_tokenizer.encode(sample["lyrics"])
            )
            for sample in samples
        ]
        lyric_lengths = torch.tensor([len(ids) for ids in encoded], dtype=torch.long)
        maximum_lyrics = max(1, int(lyric_lengths.max().item()))
        lyric_ids = torch.zeros(len(samples), maximum_lyrics, dtype=torch.long)
        for row, ids in enumerate(encoded):
            if ids:
                lyric_ids[row, : len(ids)] = torch.tensor(ids)
        frame_lengths_25hz = _subsampled_lengths(feature_lengths)
        adjacent_repeats = torch.tensor(
            [
                ctc_required_frames(ids) - len(ids)
                for ids in encoded
            ],
            dtype=torch.long,
        )
        required_ctc_frames = lyric_lengths + adjacent_repeats
        requested_ctc = torch.tensor(
            [bool(sample["ctc_enabled"]) for sample in samples],
            dtype=torch.bool,
        )
        ctc_feasible = (lyric_lengths > 0) & (
            frame_lengths_25hz >= required_ctc_frames
        )
        ctc_infeasible = requested_ctc & ~ctc_feasible
        ctc_enabled = requested_ctc & ctc_feasible
        maximum_frames_25hz = int(frame_lengths_25hz.max().item())
        ctc_frame_token_ids = torch.zeros(
            len(samples), maximum_frames_25hz, dtype=torch.long
        )


        ctc_frame_target_mask = torch.zeros(
            len(samples), maximum_frames_25hz, dtype=torch.bool
        )
        ctc_alignment_enabled = torch.zeros(len(samples), dtype=torch.bool)
        for row, sample in enumerate(samples):
            intervals = sample["ctc_unit_intervals"]
            if not intervals:
                ctc_frame_target_mask[row] = False
                continue
            ctc_alignment_enabled[row] = True
            frame_count = int(frame_lengths_25hz[row].item())
            previous_unit: str | None = None
            for interval in intervals:
                unit = str(interval["unit"])
                token_id = self.text_tokenizer.encode_units([unit])[0]
                start_frame = max(0, round(float(interval["start_sec"]) * 25.0))
                end_frame = min(
                    frame_count,
                    max(start_frame + 1, round(float(interval["end_sec"]) * 25.0)),
                )
                if start_frame >= frame_count:
                    continue

                if unit == previous_unit and end_frame - start_frame > 1:
                    start_frame += 1
                ctc_frame_token_ids[row, start_frame:end_frame] = token_id
                ctc_frame_target_mask[row, start_frame:end_frame] = True
                previous_unit = unit
        batch.update(
            lyrics_token_ids=lyric_ids,
            lyrics_lengths=lyric_lengths,
            mel_target=mel,
            mel_target_mask=lengths_to_mask(feature_lengths, mel.shape[1]),
            chroma_target=chroma,
            chroma_target_mask=lengths_to_mask(feature_lengths, chroma.shape[1]),
            ctc_enabled=ctc_enabled,
            ctc_infeasible=ctc_infeasible,
            ctc_required_frames=required_ctc_frames,
            ctc_frame_token_ids=ctc_frame_token_ids,
            ctc_frame_target_mask=ctc_frame_target_mask,
            ctc_alignment_enabled=ctc_alignment_enabled,
            mel_enabled=torch.tensor([sample["mel_enabled"] for sample in samples]),
            chroma_enabled=torch.tensor([sample["chroma_enabled"] for sample in samples]),
        )
        return batch
