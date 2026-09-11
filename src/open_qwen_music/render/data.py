
from __future__ import annotations

import bisect
import fcntl
import hashlib
import hmac
import io
import json
import math
import os
import secrets
import sys
import tarfile
import threading
import time
import zipfile
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from itertools import accumulate, islice
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import soundfile as sf
import soxr
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from .contracts import (
    AUDIO_CHANNELS,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    SAMPLES_PER_LATENT_FRAME,
    lengths_to_mask,
)
from .types import RenderAudioBatch


RESAMPLE_CONTEXT_SECONDS = 0.1
_AUDIO_VERIFY_SENTINEL_SCHEMA = "oqm.render-audio-node-verification.v1"
_AUDIO_VERIFY_CACHE_ENV = "OQM_RENDER_AUDIO_VERIFY_CACHE"


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _audio_verify_root() -> Path:
    configured = os.environ.get(_AUDIO_VERIFY_CACHE_ENV)
    root = (
        Path(configured)
        if configured
        else Path("/tmp") / f"open-qwen-music-render-audio-{os.getuid()}"
    )
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink():
        raise RuntimeError("Render audio verification cache directory must not be a symlink")
    root.chmod(0o700)
    return root


def _audio_verify_key(root: Path) -> bytes:
    path = root / "authentication.key"
    lock_path = root / "authentication.key.lock"
    with lock_path.open("a+b") as lock:
        os.chmod(lock_path, 0o600)


        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not path.exists():
            temporary = root / f".authentication.key.{os.getpid()}.tmp"
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(secrets.token_bytes(32))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        if path.is_symlink() or path.stat().st_mode & 0o077:
            raise RuntimeError("Render audio verification key has insecure permissions")
        key = path.read_bytes()
        if len(key) != 32:
            raise RuntimeError("Render audio verification key is invalid")
        return key


def _stat_identity(path: Path) -> dict[str, int | str]:
    value = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
        "inode": int(value.st_ino),
        "device": int(value.st_dev),
    }


def _verified_audio_sentinel(
    path: Path,
    *,
    expected: Mapping[str, Any],
    authentication_key: bytes,
) -> bool:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 64 * 1024:
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(value, dict):
        return False
    supplied = value.pop("hmac_sha256", None)
    if not isinstance(supplied, str):
        return False
    actual = hmac.new(
        authentication_key,
        _canonical_json_bytes(value),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(supplied, actual) and value == expected


def _write_audio_sentinel(
    path: Path,
    *,
    payload: Mapping[str, Any],
    authentication_key: bytes,
) -> None:
    value = dict(payload)
    value["hmac_sha256"] = hmac.new(
        authentication_key,
        _canonical_json_bytes(payload),
        hashlib.sha256,
    ).hexdigest()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(value) + b"\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_audio_sha256_node_once(
    *,
    uri: str,
    container: Path,
    payload: bytes | None,
    expected_sha256: str,
    payload_offset: int | None = None,
    payload_size: int | None = None,
) -> None:
    if len(expected_sha256) != 64:
        raise RuntimeError("Render audio SHA-256 must be 64 hexadecimal characters")
    int(expected_sha256, 16)
    root = _audio_verify_root()
    authentication_key = _audio_verify_key(root)
    cache_key = hashlib.sha256(f"{uri}\0{expected_sha256}".encode("utf-8")).hexdigest()
    lock_path = root / f"{cache_key}.lock"
    sentinel_path = root / f"{cache_key}.json"
    with lock_path.open("a+b") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        initial_stat = _stat_identity(container)
        sentinel_payload: dict[str, Any] = {
            "schema_version": _AUDIO_VERIFY_SENTINEL_SCHEMA,
            "uri": uri,
            "expected_sha256": expected_sha256,
            "container": initial_stat,
        }
        if payload_offset is not None or payload_size is not None:
            if payload_offset is None or payload_size is None:
                raise RuntimeError("Indexed audio verification requires both offset and size")
            if payload_offset < 0 or payload_size < 0:
                raise RuntimeError("Indexed audio verification range is invalid")
            if payload_offset + payload_size > int(initial_stat["size"]):
                raise RuntimeError("Indexed audio verification range exceeds TAR bounds")
            sentinel_payload["payload_range"] = {
                "offset": payload_offset,
                "size": payload_size,
            }
        if _verified_audio_sentinel(
            sentinel_path,
            expected=sentinel_payload,
            authentication_key=authentication_key,
        ):
            return
        digest = hashlib.sha256()
        if payload is not None:
            digest.update(payload)
        elif payload_offset is not None and payload_size is not None:
            descriptor = os.open(container, os.O_RDONLY)
            try:
                position = payload_offset
                remaining = payload_size
                while remaining:
                    chunk = os.pread(
                        descriptor, min(8 * 1024 * 1024, remaining), position
                    )
                    if not chunk:
                        raise RuntimeError(f"Indexed audio verification read is incomplete: {uri}")
                    digest.update(chunk)
                    position += len(chunk)
                    remaining -= len(chunk)
            finally:
                os.close(descriptor)
        else:
            with container.open("rb") as source:
                while chunk := source.read(8 * 1024 * 1024):
                    digest.update(chunk)
        final_stat = _stat_identity(container)
        if final_stat != initial_stat:
            raise RuntimeError(f"Render audio file changed during verification: {container}")
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"Render audio SHA-256 mismatch: {uri}")
        _write_audio_sentinel(
            sentinel_path,
            payload=sentinel_payload,
            authentication_key=authentication_key,
        )


class _IndexedTarMember(io.RawIOBase):

    def __init__(self, path: Path, *, offset: int, size: int) -> None:
        super().__init__()
        if offset < 0 or size < 0:
            raise ValueError("Indexed TAR member range is invalid")
        self._descriptor = os.open(path, os.O_RDONLY)
        container_size = os.fstat(self._descriptor).st_size
        if offset + size > container_size:
            os.close(self._descriptor)
            self._descriptor = -1
            raise RuntimeError("Indexed TAR member crosses the container boundary")
        self._offset = offset
        self._size = size
        self._position = 0

    def _ensure_open(self) -> None:
        if self.closed or self._descriptor < 0:
            raise ValueError("Cannot perform I/O on a closed indexed TAR member")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        self._ensure_open()
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        self._ensure_open()
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"Unsupported seek whence={whence}")
        if position < 0:
            raise ValueError("Indexed TAR member cannot seek to a negative position")
        self._position = min(position, self._size)
        return self._position

    def readinto(self, buffer: Any) -> int:
        self._ensure_open()
        remaining = self._size - self._position
        if remaining <= 0:
            return 0
        view = memoryview(buffer).cast("B")
        requested = min(len(view), remaining)
        chunk = os.pread(
            self._descriptor,
            requested,
            self._offset + self._position,
        )
        view[: len(chunk)] = chunk
        self._position += len(chunk)
        return len(chunk)

    def close(self) -> None:
        if getattr(self, "_descriptor", -1) >= 0:
            os.close(self._descriptor)
            self._descriptor = -1
        super().close()


@dataclass(frozen=True)
class RenderAudioRecord:
    sample_id: str
    audio_path: str
    audio_sha256: str | None = None
    source_sha256: str | None = None
    data_release_sha256: str | None = None
    split: str = "train"
    repeat_count: int = 1
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RenderAudioRecord":
        if value.get("schema_version") != "oqm.acoustic-audio-view.v1":
            raise ValueError(
                "Render manifest records must use schema_version="
                "'oqm.acoustic-audio-view.v1'"
            )
        sample_id = value.get("sample_id", value.get("id"))
        audio = value.get("audio")
        if isinstance(audio, Mapping):
            audio_path = audio.get("uri")
            audio_sha256 = audio.get("sha256")
            source_sha256 = audio.get("source_sha256")
            metadata_source = {
                **dict(value),
                **dict(audio),
            }
        else:
            audio_path = value.get("audio_path", value.get("path"))
            audio_sha256 = value.get("audio_sha256", value.get("source_sha256"))
            source_sha256 = value.get("source_sha256")
            metadata_source = dict(value)
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Render manifest record is missing sample_id")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError(f"{sample_id} is missing audio_path")
        repeat_count = value.get("repeat_count")
        if repeat_count is None:
            sampling = value.get("sampling")
            repeat_count = (
                sampling.get("repeat_count") if isinstance(sampling, Mapping) else None
            )
        if repeat_count is None:
            repeat_count = 1
        return cls(
            sample_id=sample_id,
            audio_path=audio_path,
            audio_sha256=audio_sha256,
            source_sha256=source_sha256,
            data_release_sha256=value.get("data_release_sha256"),
            split=str(
                value.get("split")
                or (value.get("groups") or {}).get("split")
                or "train"
            ),
            repeat_count=int(repeat_count),
            metadata={
                key: item
                for key, item in metadata_source.items()
                if key
                not in {
                    "sample_id",
                    "id",
                    "audio_path",
                    "path",
                    "audio",
                    "audio_sha256",
                    "source_sha256",
                    "data_release_sha256",
                }
            },
        )


def load_render_manifest(path: str | Path) -> list[RenderAudioRecord]:
    manifest = Path(path)
    records: list[RenderAudioRecord] = []
    with manifest.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest}:{line_number} JSON cannot be parsed") from exc
            record = RenderAudioRecord.from_mapping(value)
            audio_path = Path(record.audio_path)
            if "://" not in record.audio_path and not audio_path.is_absolute():
                record = RenderAudioRecord(
                    **{
                        **record.__dict__,
                        "audio_path": str((manifest.parent / audio_path).resolve()),
                    }
                )
            records.append(record)
    if not records:
        raise ValueError(f"Render manifest is empty: {manifest}")
    identifiers = [record.sample_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Render manifest contains duplicate sample_id")
    return records


class RenderAudioDataset(Dataset[dict[str, Any]]):

    def __init__(
        self,
        manifest: str | Path | Sequence[RenderAudioRecord],
        *,
        segment_samples: int | None = SEGMENT_SAMPLES,
        random_crop: bool = True,
        seed: int = 0,
        verify_audio_sha256: bool = False,
        audio_integrity_mode: str | None = None,
        verify_source_sha256: bool | None = None,
        allow_source_audio: bool = False,
        expected_release_sha256: str | None = None,
        expected_release_revision: str | None = None,
        split: str = "train",
    ) -> None:
        if (
            expected_release_sha256 is not None
            and expected_release_revision is not None
            and expected_release_sha256 != expected_release_revision
        ):
            raise ValueError("Conflicting expected release revision parameters")
        expected_release_revision = (
            expected_release_revision
            if expected_release_revision is not None
            else expected_release_sha256
        )
        records = (
            list(manifest)
            if not isinstance(manifest, (str, Path))
            else load_render_manifest(manifest)
        )
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be 'train', 'valid', or 'test'")
        self.records = [record for record in records if record.split == split]
        if not self.records:
            raise ValueError("RenderAudioDataset records cannot be empty")
        if segment_samples is not None and segment_samples <= 0:
            raise ValueError("segment_samples must be positive or None")
        self.segment_samples = segment_samples
        self.random_crop = bool(random_crop)
        self.seed = int(seed)
        self.epoch = 0
        if verify_source_sha256 is not None:
            if verify_audio_sha256 and not verify_source_sha256:
                raise ValueError("Conflicting audio/source SHA-256 verification parameters")
            verify_audio_sha256 = bool(verify_source_sha256)
        if audio_integrity_mode is None:
            audio_integrity_mode = "node_once" if verify_audio_sha256 else "none"
        if audio_integrity_mode not in {"none", "node_once"}:
            raise ValueError("audio_integrity_mode must be 'none' or 'node_once'")
        if verify_audio_sha256 and audio_integrity_mode == "none":
            raise ValueError("verify_audio_sha256=true cannot be used with integrity mode 'none'")
        self.audio_integrity_mode = audio_integrity_mode
        self.verify_audio_sha256 = audio_integrity_mode != "none"
        self.allow_source_audio = bool(allow_source_audio)
        release_revisions = {
            record.data_release_sha256
            for record in self.records
            if record.data_release_sha256 is not None
        }
        self.data_release_sha256 = (
            next(iter(release_revisions)) if len(release_revisions) == 1 else None
        )
        if expected_release_revision is not None:
            if len(expected_release_revision) != 64:
                raise ValueError("expected_release_revision must be 64 hexadecimal characters")
            int(expected_release_revision, 16)
            mismatches = []
            for record in self.records:
                if record.data_release_sha256 != expected_release_revision:
                    mismatches.append(record.sample_id)
            if mismatches:
                raise ValueError(f"Manifest release revision mismatch: {mismatches[:8]}")
        bad_repeats = [
            record.sample_id for record in self.records if record.repeat_count <= 0
        ]
        if bad_repeats:
            raise ValueError(f"repeat_count must be positive: {bad_repeats[:8]}")
        self._cumulative_repeats = list(
            accumulate(record.repeat_count for record in self.records)
        )

    def __len__(self) -> int:
        return self._cumulative_repeats[-1]

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must not be negative")
        self.epoch = int(epoch)

    def _record_slot(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        record_index = bisect.bisect_right(self._cumulative_repeats, index)
        previous = (
            0 if record_index == 0 else self._cumulative_repeats[record_index - 1]
        )
        return record_index, index - previous

    def _crop_start(
        self,
        record: RenderAudioRecord,
        *,
        slot: int,
        attempt: int,
        available: int,
        required: int,
        grid: int,
        edge_fraction: float,
    ) -> int:
        if self.segment_samples is None or available <= required:
            return 0
        edge = min(int(available * edge_fraction), max(0, available - required))
        lower = edge
        upper = max(lower, available - edge - required)
        if not self.random_crop:
            sampling = (record.metadata or {}).get("sampling")
            sampling = sampling if isinstance(sampling, Mapping) else {}
            signed = sampling.get("fixed_source_start_samples")
            if isinstance(signed, list) and signed:
                start = int(signed[(slot + attempt) % len(signed)])
                if not 0 <= start <= max(0, available - required):
                    raise ValueError(f"{record.sample_id} signed crop is out of bounds")
                return start
            if record.repeat_count == 1:
                return (lower + upper) // 2
            fraction = (slot + 1) / (record.repeat_count + 1)
            return int(round(lower + fraction * (upper - lower)))
        slots = max(1, (upper - lower) // max(1, grid) + 1)
        prefix = f"{self.seed}:{self.epoch}:{record.sample_id}:{attempt}"
        offset = (
            int.from_bytes(
                hashlib.sha256(f"{prefix}:offset".encode()).digest()[:8], "little"
            )
            % slots
        )
        stride = (
            int.from_bytes(
                hashlib.sha256(f"{prefix}:stride".encode()).digest()[:8], "little"
            )
            % slots
        )
        stride = max(1, stride)
        while math.gcd(stride, slots) != 1:
            stride += 1
            if stride >= slots:
                stride = 1
        position = (offset + slot * stride) % slots
        return lower + position * max(1, grid)

    @staticmethod
    def _uri_parts(uri: str) -> tuple[str, Path, str]:
        if "://" not in uri:
            return "file", Path(uri), ""
        scheme, _, rest = uri.partition("://")
        container, separator, member = rest.partition("::")
        if scheme == "file":
            return scheme, Path(container), ""
        if scheme not in {"tar", "targz", "zip"} or not separator or not member:
            raise ValueError(f"Unsupported published audio URI: {uri}")
        return scheme, Path(container), member

    @staticmethod
    def _read_hints(record: RenderAudioRecord) -> Mapping[str, Any]:
        metadata = record.metadata or {}
        direct = metadata.get("source_read_hints")
        if isinstance(direct, Mapping):
            return direct
        source = metadata.get("source")
        if isinstance(source, Mapping) and isinstance(
            source.get("read_hints"), Mapping
        ):
            return source["read_hints"]
        return {}

    @classmethod
    def _indexed_tar_range(cls, record: RenderAudioRecord) -> tuple[int, int] | None:
        scheme, _, _ = cls._uri_parts(record.audio_path)
        if scheme != "tar":
            return None
        hints = cls._read_hints(record)
        offset = int(hints.get("data_offset", -1))
        size = int(hints.get("size", -1))
        return (offset, size) if offset >= 0 and size >= 0 else None

    def _audio_payload(self, record: RenderAudioRecord) -> bytes | None:
        scheme, container, member = self._uri_parts(record.audio_path)
        if scheme == "file":
            return None
        if scheme == "zip":
            with zipfile.ZipFile(container) as archive:
                return archive.read(member)
        if self._indexed_tar_range(record) is not None:


            return None
        with tarfile.open(container, "r:gz" if scheme == "targz" else "r:") as archive:
            handle = archive.extractfile(member)
            if handle is None:
                raise FileNotFoundError(f"{container}::{member}")
            return handle.read()

    @staticmethod
    def _segment_valid(
        audio: np.ndarray,
        policy: Mapping[str, Any],
    ) -> bool:
        if audio.ndim != 2 or audio.shape[1] != AUDIO_CHANNELS:
            return False
        if not np.isfinite(audio).all():
            return False
        work = audio.astype(np.float64)
        rms = math.sqrt(float(np.square(work).mean()))
        rms_dbfs = -160.0 if rms <= 1.0e-8 else 20.0 * math.log10(rms)
        if rms_dbfs < float(policy.get("min_rms_dbfs", -160.0)):
            return False
        if float((np.abs(work) >= 1.0).mean()) > float(
            policy.get("max_clipping_ratio", 1.0)
        ):
            return False
        difference = work[:, 0] - work[:, 1]
        if float((difference == 0.0).mean()) >= float(
            policy.get("max_lr_equal_ratio", 1.0)
        ):
            return False
        side_to_total = float(np.square(difference).mean()) / max(
            float(np.square(work).mean()),
            1.0e-12,
        )
        if side_to_total < float(policy.get("min_side_to_total_ratio", 0.0)):
            return False
        left_std = float(work[:, 0].std())
        right_std = float(work[:, 1].std())
        if left_std <= 1.0e-8 or right_std <= 1.0e-8:
            return False
        correlation = float(np.corrcoef(work[:, 0], work[:, 1])[0, 1])
        return math.isfinite(correlation) and abs(correlation) < float(
            policy.get("max_abs_channel_correlation", 1.0)
        )

    @staticmethod
    def _read_segment_with_resample_context(
        handle: sf.SoundFile,
        *,
        start: int,
        source_samples: int,
        target_samples: int,
        sample_rate: int,
    ) -> tuple[np.ndarray, int]:
        if sample_rate == SAMPLE_RATE:
            handle.seek(start)
            candidate = handle.read(
                target_samples,
                dtype="float32",
                always_2d=True,
            )
        else:
            context = max(1, int(math.ceil(sample_rate * RESAMPLE_CONTEXT_SECONDS)))
            read_start = max(0, start - context)
            read_stop = min(
                int(handle.frames),
                start + source_samples + context,
            )
            handle.seek(read_start)
            contextual = handle.read(
                max(0, read_stop - read_start),
                dtype="float32",
                always_2d=True,
            )
            converted = soxr.resample(
                contextual,
                sample_rate,
                SAMPLE_RATE,
                quality="VHQ",
            )
            target_offset = int(round((start - read_start) * SAMPLE_RATE / sample_rate))
            candidate = converted[target_offset : target_offset + target_samples]
        candidate = np.ascontiguousarray(candidate[:target_samples], dtype=np.float32)
        valid_length = int(candidate.shape[0])
        if valid_length < target_samples:
            candidate = np.pad(
                candidate,
                ((0, target_samples - valid_length), (0, 0)),
            )
        return np.ascontiguousarray(candidate, dtype=np.float32), valid_length

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, slot = self._record_slot(index)
        record = self.records[record_index]
        scheme, path, _ = self._uri_parts(record.audio_path)
        indexed_range = self._indexed_tar_range(record)
        payload = self._audio_payload(record)
        if self.audio_integrity_mode == "node_once":
            if not record.audio_sha256:
                raise RuntimeError(f"{record.sample_id} is missing the derived-audio SHA-256")
            _verify_audio_sha256_node_once(
                uri=record.audio_path,
                container=path,
                payload=payload,
                expected_sha256=record.audio_sha256,
                payload_offset=(indexed_range[0] if indexed_range else None),
                payload_size=(indexed_range[1] if indexed_range else None),
            )
        backing: io.IOBase | str
        if indexed_range is not None:
            backing = _IndexedTarMember(
                path,
                offset=indexed_range[0],
                size=indexed_range[1],
            )
        else:
            backing = io.BytesIO(payload) if payload is not None else str(path)
        metadata = record.metadata or {}
        sampling = metadata.get("sampling")
        sampling = sampling if isinstance(sampling, Mapping) else {}
        crop_policy = sampling.get("segment_qc")
        crop_policy = crop_policy if isinstance(crop_policy, Mapping) else {}
        attempts = int(sampling.get("attempts_per_segment", 1))
        edge_fraction = float(sampling.get("crop_edge_fraction", 0.0))
        if not 0.0 <= edge_fraction < 0.5 or attempts <= 0:
            raise ValueError(f"{record.sample_id} dynamic crop policy is invalid")
        start = 0
        audio: np.ndarray | None = None
        audio_valid_length: int | None = None
        try:
            with sf.SoundFile(backing) as handle:
                sample_rate = int(handle.samplerate)
                if sample_rate != SAMPLE_RATE and not self.allow_source_audio:
                    raise ValueError(
                        f"{record.sample_id} must be materialized at {SAMPLE_RATE} Hz; "
                        "published source tracks require explicit allow_source_audio"
                    )
                if int(handle.channels) != AUDIO_CHANNELS:
                    raise ValueError(
                        f"{record.sample_id} must be stereo; received {handle.channels}ch"
                    )
                if self.segment_samples is None:
                    start = 0
                    handle.seek(0)
                    audio = handle.read(dtype="float32", always_2d=True)
                    if sample_rate != SAMPLE_RATE:
                        audio = soxr.resample(
                            audio,
                            sample_rate,
                            SAMPLE_RATE,
                            quality="VHQ",
                        )
                    audio = np.ascontiguousarray(audio, dtype=np.float32)
                    audio_valid_length = int(audio.shape[0])
                else:
                    source_samples = (
                        self.segment_samples
                        if sample_rate == SAMPLE_RATE
                        else math.ceil(self.segment_samples * sample_rate / SAMPLE_RATE)
                    )
                    grid = max(1, int(round(sample_rate / 25.0)))
                    for attempt in range(attempts):
                        start = self._crop_start(
                            record,
                            slot=slot,
                            attempt=attempt,
                            available=int(handle.frames),
                            required=source_samples,
                            grid=grid,
                            edge_fraction=edge_fraction,
                        )
                        candidate, candidate_valid_length = (
                            self._read_segment_with_resample_context(
                                handle,
                                start=start,
                                source_samples=source_samples,
                                target_samples=self.segment_samples,
                                sample_rate=sample_rate,
                            )
                        )


                        if not crop_policy or self._segment_valid(
                            candidate, crop_policy
                        ):
                            audio = candidate
                            audio_valid_length = candidate_valid_length
                            break
        finally:
            if isinstance(backing, io.IOBase):
                backing.close()
        if audio is None or audio_valid_length is None:
            raise RuntimeError(
                f"{record.sample_id} failed segment quality checks after "
                f"{attempts} deterministic crop attempts"
            )
        waveform = torch.from_numpy(np.ascontiguousarray(audio.T))
        selected = waveform
        valid_length = audio_valid_length
        if (
            self.segment_samples is not None
            and selected.shape[-1] < self.segment_samples
        ):
            selected = F.pad(
                selected,
                (0, self.segment_samples - selected.shape[-1]),
            )
        media_bandwidth_hz = metadata.get(
            "media_bandwidth_hz",
            float(metadata.get("sample_rate_hz", sample_rate)) / 2.0,
        )
        media_bandwidth_hz = float(media_bandwidth_hz)
        if (
            not math.isfinite(media_bandwidth_hz)
            or media_bandwidth_hz <= 0
            or media_bandwidth_hz > SAMPLE_RATE / 2
        ):
            raise ValueError(f"{record.sample_id} media_bandwidth_hz is invalid")
        return {
            "sample_id": record.sample_id,
            "audio": selected,
            "audio_length": valid_length,
            "duration_seconds": valid_length / SAMPLE_RATE,
            "media_bandwidth_hz": media_bandwidth_hz,
            "magnitude_max_hz": media_bandwidth_hz,
            "phase_max_hz": media_bandwidth_hz,
            "stereo_max_hz": media_bandwidth_hz,
            "adversarial_max_hz": media_bandwidth_hz,
            "waveform_adversarial_enabled": True,
            "provenance": {
                "audio_path": str(path),
                "audio_uri": record.audio_path,
                "crop_start_samples": start,
                "crop_slot": slot,
                "source_sample_rate_hz": sample_rate,
                "source_storage_scheme": scheme,
                "source_sha256": record.source_sha256,
                "audio_sha256": record.audio_sha256,
                "data_release_sha256": record.data_release_sha256,
                **(record.metadata or {}),
            },
        }


class RenderCollator:
    def __init__(
        self,
        *,
        fixed_samples: int | None = None,
        pad_to_latent_frame: bool = True,
        return_dataclass: bool = False,
    ) -> None:
        if fixed_samples is not None and fixed_samples <= 0:
            raise ValueError("fixed_samples must be positive")
        self.fixed_samples = fixed_samples
        self.pad_to_latent_frame = bool(pad_to_latent_frame)
        self.return_dataclass = bool(return_dataclass)

    def __call__(
        self, samples: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any] | RenderAudioBatch:
        if not samples:
            raise ValueError("RenderCollator does not accept an empty batch")
        lengths = torch.tensor(
            [int(sample["audio_length"]) for sample in samples],
            dtype=torch.long,
        )
        maximum = (
            self.fixed_samples
            if self.fixed_samples is not None
            else max(int(sample["audio"].shape[-1]) for sample in samples)
        )
        if self.pad_to_latent_frame:
            maximum = (
                (maximum + SAMPLES_PER_LATENT_FRAME - 1)
                // SAMPLES_PER_LATENT_FRAME
                * SAMPLES_PER_LATENT_FRAME
            )
        if int(lengths.max()) > maximum:
            raise ValueError("fixed_samples is shorter than an effective batch item")
        audio_values: list[Tensor] = []
        for sample in samples:
            audio = sample["audio"].float()
            if audio.shape[0] != AUDIO_CHANNELS:
                raise ValueError("collator audio must be [2,N]")
            if audio.shape[-1] > maximum:
                audio = audio[..., :maximum]
            audio_values.append(F.pad(audio, (0, maximum - audio.shape[-1])))
        audio = torch.stack(audio_values)
        mask = lengths_to_mask(lengths, maximum)
        duration = lengths.float() / SAMPLE_RATE
        media_bandwidth_hz = torch.tensor(
            [float(sample["media_bandwidth_hz"]) for sample in samples],
            dtype=torch.float32,
        )
        magnitude_max_hz = torch.tensor(
            [float(sample["magnitude_max_hz"]) for sample in samples],
            dtype=torch.float32,
        )
        phase_max_hz = torch.tensor(
            [float(sample["phase_max_hz"]) for sample in samples],
            dtype=torch.float32,
        )
        stereo_max_hz = torch.tensor(
            [float(sample["stereo_max_hz"]) for sample in samples],
            dtype=torch.float32,
        )
        adversarial_max_hz = torch.tensor(
            [float(sample["adversarial_max_hz"]) for sample in samples],
            dtype=torch.float32,
        )
        waveform_adversarial_enabled = torch.tensor(
            [bool(sample["waveform_adversarial_enabled"]) for sample in samples],
            dtype=torch.bool,
        )
        batch = RenderAudioBatch(
            sample_ids=[str(sample["sample_id"]) for sample in samples],
            audio=audio,
            audio_lengths=lengths,
            audio_mask=mask,
            duration_seconds=duration,
            media_bandwidth_hz=media_bandwidth_hz,
            magnitude_max_hz=magnitude_max_hz,
            phase_max_hz=phase_max_hz,
            stereo_max_hz=stereo_max_hz,
            adversarial_max_hz=adversarial_max_hz,
            waveform_adversarial_enabled=waveform_adversarial_enabled,
            provenance=[dict(sample.get("provenance", {})) for sample in samples],
        )
        if self.return_dataclass:
            return batch
        return {
            "sample_ids": batch.sample_ids,
            "audio": batch.audio,
            "audio_lengths": batch.audio_lengths,
            "audio_mask": batch.audio_mask,
            "duration_seconds": batch.duration_seconds,
            "media_bandwidth_hz": batch.media_bandwidth_hz,
            "magnitude_max_hz": batch.magnitude_max_hz,
            "phase_max_hz": batch.phase_max_hz,
            "stereo_max_hz": batch.stereo_max_hz,
            "adversarial_max_hz": batch.adversarial_max_hz,
            "waveform_adversarial_enabled": batch.waveform_adversarial_enabled,
            "provenance": batch.provenance,
        }


class ThreadedBatchLoader:

    def __init__(
        self,
        dataset: Dataset[dict[str, Any]],
        *,
        sampler: Sampler[int],
        batch_size: int,
        num_workers: int,
        collate_fn: Callable[[Sequence[dict[str, Any]]], Any],
        prefetch_batches: int = 2,
        slow_item_warning_seconds: float = 0.0,
    ) -> None:
        if batch_size <= 0 or num_workers <= 0 or prefetch_batches <= 0:
            raise ValueError("Threaded loader batch, workers, and prefetch values must be positive")
        if slow_item_warning_seconds < 0:
            raise ValueError("Threaded loader slow-sample threshold must not be negative")
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.collate_fn = collate_fn
        self.prefetch_batches = int(prefetch_batches)
        self.slow_item_warning_seconds = float(slow_item_warning_seconds)

    def __iter__(self) -> Iterator[Any]:
        return _ThreadedBatchIterator(self)

    def __len__(self) -> int:
        return math.ceil(len(self.sampler) / self.batch_size)


@dataclass
class _PendingThreadedSample:
    index: int
    started: threading.Event
    started_at: list[float | None]
    future: Future[dict[str, Any]]


def _load_threaded_sample(
    dataset: Dataset[dict[str, Any]],
    index: int,
    started: threading.Event,
    started_at: list[float | None],
) -> dict[str, Any]:


    started_at[0] = time.monotonic()
    started.set()
    return dataset.__getitem__(index)


class _ThreadedBatchIterator:
    def __init__(self, loader: ThreadedBatchLoader) -> None:
        self.loader = loader
        self.indices = iter(loader.sampler)
        self.executor = ThreadPoolExecutor(
            max_workers=loader.num_workers,
            thread_name_prefix="oqm-render-audio",
        )
        self.pending: deque[list[_PendingThreadedSample]] = deque()
        self.exhausted = False
        self.closed = False
        for _ in range(loader.prefetch_batches):
            self._schedule()

    def _schedule(self) -> None:
        if self.exhausted:
            return
        indices = list(islice(self.indices, self.loader.batch_size))
        if not indices:
            self.exhausted = True
            return
        if len(indices) != self.loader.batch_size:
            self.close()
            raise RuntimeError("Threaded loader received an incomplete batch")
        scheduled: list[_PendingThreadedSample] = []
        for index in indices:
            started = threading.Event()
            started_at: list[float | None] = [None]
            future = self.executor.submit(
                _load_threaded_sample,
                self.loader.dataset,
                int(index),
                started,
                started_at,
            )
            scheduled.append(
                _PendingThreadedSample(
                    index=int(index),
                    started=started,
                    started_at=started_at,
                    future=future,
                )
            )
        self.pending.append(scheduled)

    def _await_sample(
        self,
        item: _PendingThreadedSample,
    ) -> dict[str, Any]:
        warning_seconds = self.loader.slow_item_warning_seconds
        if warning_seconds <= 0:
            return item.future.result()
        while not item.started.is_set():
            try:
                return item.future.result(timeout=min(0.1, warning_seconds))
            except FutureTimeoutError:
                if item.future.done():
                    return item.future.result()
        started_at = item.started_at[0]
        if started_at is None:
            raise RuntimeError("Threaded loader worker startup state is invalid")
        warning_count = 0
        next_warning = warning_seconds
        rank = dist.get_rank() if dist.is_initialized() else 0
        while True:
            elapsed = time.monotonic() - started_at
            if not item.future.done() and elapsed >= next_warning:
                warning_count += 1
                print(
                    json.dumps(
                        {
                            "event": "render_threaded_loader_slow_item",
                            "rank": rank,
                            "dataset_index": item.index,
                            "elapsed_seconds": round(elapsed, 3),
                            "warning_count": warning_count,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                while next_warning <= elapsed:
                    next_warning += warning_seconds
            timeout = max(0.001, next_warning - (time.monotonic() - started_at))
            try:
                sample = item.future.result(timeout=timeout)
            except FutureTimeoutError:


                if item.future.done():
                    return item.future.result()
                continue
            if warning_count:
                print(
                    json.dumps(
                        {
                            "event": "render_threaded_loader_slow_item_recovered",
                            "rank": rank,
                            "dataset_index": item.index,
                            "elapsed_seconds": round(
                                time.monotonic() - started_at,
                                3,
                            ),
                            "warning_count": warning_count,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            return sample

    def __iter__(self) -> _ThreadedBatchIterator:
        return self

    def __next__(self) -> Any:
        if not self.pending:
            self.close()
            raise StopIteration
        futures = self.pending.popleft()
        try:
            samples = [self._await_sample(item) for item in futures]
        except Exception:
            self.close()
            raise
        self._schedule()
        return self.loader.collate_fn(samples)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for futures in self.pending:
            for item in futures:
                item.future.cancel()
        self.pending.clear()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self) -> None:
        self.close()


def _feistel_permute_index(
    index: int,
    *,
    size: int,
    seed: int,
    epoch: int,
) -> int:

    if not 0 <= index < size:
        raise ValueError("Permutation index is out of bounds")
    if size <= 1:
        return index
    bits = (size - 1).bit_length()
    if bits % 2:
        bits += 1
    half = bits // 2
    mask = (1 << half) - 1
    key = f"{seed}:{epoch}:{size}".encode()

    def permute_once(value: int) -> int:
        left = (value >> half) & mask
        right = value & mask
        for round_index in range(6):
            digest = hashlib.blake2b(
                key
                + round_index.to_bytes(1, "big")
                + right.to_bytes((half + 7) // 8, "big"),
                digest_size=8,
            ).digest()
            function = int.from_bytes(digest, "big") & mask
            left, right = right, left ^ function
        return (left << half) | right

    value = index
    for _ in range(128):
        value = permute_once(value)
        if value < size:
            return value
    raise RuntimeError("Feistel cycle walk did not return to the valid domain within 128 rounds")


class DistributedResumableSampler(Sampler[int]):

    FORMAT_VERSION = "oqm.render.distributed-sampler.v1"

    def __init__(
        self,
        data_source: Dataset[Any] | Sequence[Any],
        *,
        rank: int | None = None,
        world_size: int | None = None,
        batch_size: int = 1,
        seed: int = 0,
        shuffle: bool = True,
        order_mode: str = "torch_randperm",
    ) -> None:
        inferred_world = dist.get_world_size() if dist.is_initialized() else 1
        inferred_rank = dist.get_rank() if dist.is_initialized() else 0
        self.rank = inferred_rank if rank is None else int(rank)
        self.world_size = inferred_world if world_size is None else int(world_size)
        self.batch_size = int(batch_size)
        self.data_source = data_source
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.order_mode = str(order_mode)
        if self.order_mode not in {"torch_randperm", "feistel_cycle_walk_v1"}:
            raise ValueError("sampler order_mode is invalid")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("sampler rank/world_size is invalid")
        if self.batch_size <= 0:
            raise ValueError("sampler batch_size must be positive")
        self.global_batch_size = self.world_size * self.batch_size
        self.usable_samples = (
            len(self.data_source) // self.global_batch_size
        ) * self.global_batch_size
        if self.usable_samples == 0:
            raise ValueError(
                f"dataset_size={len(self.data_source)} is less than one global batch "
                f"{self.global_batch_size}"
            )
        self.samples_per_rank = self.usable_samples // self.world_size
        self.epoch = 0
        self.cursor = 0
        if hasattr(self.data_source, "set_epoch"):
            self.data_source.set_epoch(0)

    @property
    def topology(self) -> dict[str, Any]:
        topology = {
            "format_version": self.FORMAT_VERSION,
            "dataset_size": len(self.data_source),
            "world_size": self.world_size,
            "batch_size_per_rank": self.batch_size,
            "global_batch_size": self.global_batch_size,
            "usable_samples_per_epoch": self.usable_samples,
            "samples_per_rank": self.samples_per_rank,
            "seed": self.seed,
            "shuffle": self.shuffle,
        }
        if self.order_mode != "torch_randperm":
            topology["order_mode"] = self.order_mode
        return topology

    @property
    def config_hash(self) -> str:
        payload = json.dumps(
            self.topology,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def _global_order(self, *, epoch: int | None = None) -> list[int]:
        current_epoch = self.epoch if epoch is None else int(epoch)
        if current_epoch < 0:
            raise ValueError("Distributed sampler epoch must not be negative")
        if not self.shuffle:
            order = list(range(len(self.data_source)))
        elif self.order_mode == "feistel_cycle_walk_v1":
            order = [
                _feistel_permute_index(
                    position,
                    size=len(self.data_source),
                    seed=self.seed,
                    epoch=current_epoch,
                )
                for position in range(self.usable_samples)
            ]
        else:
            generator = torch.Generator().manual_seed(self.seed + current_epoch)
            order = torch.randperm(len(self.data_source), generator=generator).tolist()
        return order[: self.usable_samples]

    def local_order(self) -> list[int]:
        return self._global_order()[self.rank : self.usable_samples : self.world_size]

    def __len__(self) -> int:
        return self.samples_per_rank - self.cursor

    def __iter__(self) -> Iterator[int]:
        if self.shuffle and self.order_mode == "feistel_cycle_walk_v1":
            while self.cursor < self.samples_per_rank:
                global_position = self.rank + self.cursor * self.world_size
                index = _feistel_permute_index(
                    global_position,
                    size=len(self.data_source),
                    seed=self.seed,
                    epoch=self.epoch,
                )
                self.cursor += 1
                yield index
            self.epoch += 1
            self.cursor = 0
            if hasattr(self.data_source, "set_epoch"):
                self.data_source.set_epoch(self.epoch)
            return
        order = self.local_order()
        while self.cursor < len(order):
            index = order[self.cursor]
            self.cursor += 1
            yield index
        self.epoch += 1
        self.cursor = 0
        if hasattr(self.data_source, "set_epoch"):
            self.data_source.set_epoch(self.epoch)

    def _state_dict_at(self, *, epoch: int, cursor: int) -> dict[str, Any]:
        if epoch < 0 or not 0 <= cursor <= self.samples_per_rank:
            raise ValueError("Distributed sampler epoch/cursor is invalid")
        if self.shuffle and self.order_mode == "feistel_cycle_walk_v1":
            order_sha256 = hashlib.sha256(
                json.dumps(
                    {
                        "algorithm": self.order_mode,
                        "seed": self.seed,
                        "epoch": epoch,
                        "dataset_size": len(self.data_source),
                        "usable_samples": self.usable_samples,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        else:
            order = self._global_order(epoch=epoch)
            order_sha256 = hashlib.sha256(
                json.dumps(order, separators=(",", ":")).encode()
            ).hexdigest()
        return {
            **self.topology,
            "config_hash": self.config_hash,
            "rank": self.rank,
            "epoch": int(epoch),
            "cursor": int(cursor),
            "order_sha256": order_sha256,
        }

    def state_dict(self) -> dict[str, Any]:
        return self._state_dict_at(epoch=self.epoch, cursor=self.cursor)

    def state_dict_at_batches_consumed(
        self,
        batches_consumed: int,
    ) -> dict[str, Any]:

        if batches_consumed < 0:
            raise ValueError("batches_consumed must not be negative")
        samples_consumed = int(batches_consumed) * self.batch_size
        epoch, cursor = divmod(samples_consumed, self.samples_per_rank)
        return self._state_dict_at(epoch=epoch, cursor=cursor)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("Distributed sampler format_version is unsupported")
        expected = {
            **self.topology,
            "config_hash": self.config_hash,
            "rank": self.rank,
        }
        mismatches = {
            key: {"checkpoint": state.get(key), "current": value}
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Distributed sampler topology/config mismatch: {mismatches}")
        epoch = int(state["epoch"])
        cursor = int(state["cursor"])
        if epoch < 0 or not 0 <= cursor <= self.samples_per_rank:
            raise ValueError("Distributed sampler epoch/cursor is invalid")
        self.epoch = epoch
        self.cursor = cursor
        current_order_sha256 = self._state_dict_at(
            epoch=epoch,
            cursor=cursor,
        )["order_sha256"]
        if state.get("order_sha256") != current_order_sha256:
            raise ValueError("Distributed sampler epoch-order hash mismatch")
        if hasattr(self.data_source, "set_epoch"):
            self.data_source.set_epoch(epoch)


class ResumableRandomSampler(Sampler[int]):

    def __init__(
        self,
        data_source: Dataset[Any] | Sequence[Any],
        *,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.position = 0

    def __len__(self) -> int:
        return max(0, len(self.data_source) - self.position)

    def _order(self) -> list[int]:
        if not self.shuffle:
            return list(range(len(self.data_source)))
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.data_source), generator=generator).tolist()

    def __iter__(self) -> Iterator[int]:
        order = self._order()
        while self.position < len(order):
            index = order[self.position]
            self.position += 1
            yield index
        self.epoch += 1
        self.position = 0
        if hasattr(self.data_source, "set_epoch"):
            self.data_source.set_epoch(self.epoch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": "oqm.render.sampler.v1",
            "dataset_size": len(self.data_source),
            "seed": self.seed,
            "shuffle": self.shuffle,
            "epoch": self.epoch,
            "position": self.position,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("format_version") != "oqm.render.sampler.v1":
            raise ValueError("Sampler format_version is unsupported")
        expected = {
            "dataset_size": len(self.data_source),
            "seed": self.seed,
            "shuffle": self.shuffle,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"sampler {key} mismatch: {state.get(key)!r}!={value!r}")
        epoch = int(state["epoch"])
        position = int(state["position"])
        if epoch < 0 or not 0 <= position <= len(self.data_source):
            raise ValueError("sampler epoch/position is invalid")
        self.epoch = epoch
        self.position = position
        if hasattr(self.data_source, "set_epoch"):
            self.data_source.set_epoch(epoch)


class SyntheticRenderDataset(Dataset[dict[str, Any]]):

    def __init__(self, *, count: int = 8, samples: int = SEGMENT_SAMPLES) -> None:
        if count <= 0 or samples <= 0:
            raise ValueError("count/samples must be positive")
        self.count = count
        self.samples = samples

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, Any]:
        time = torch.arange(self.samples, dtype=torch.float32) / SAMPLE_RATE
        frequency = 110.0 * (index + 1)
        left = 0.1 * torch.sin(2 * math.pi * frequency * time)
        right = 0.08 * torch.sin(2 * math.pi * (frequency + 3.0) * time + 0.2)
        return {
            "sample_id": f"synthetic-{index:04d}",
            "audio": torch.stack((left, right)),
            "audio_length": self.samples,
            "duration_seconds": self.samples / SAMPLE_RATE,
            "media_bandwidth_hz": SAMPLE_RATE / 2,
            "magnitude_max_hz": SAMPLE_RATE / 2,
            "phase_max_hz": SAMPLE_RATE / 2,
            "stereo_max_hz": SAMPLE_RATE / 2,
            "adversarial_max_hz": SAMPLE_RATE / 2,
            "waveform_adversarial_enabled": True,
            "provenance": {"synthetic": True},
        }
