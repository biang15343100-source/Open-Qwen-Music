from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import (
    MELODY_FRAME_RATE,
    MELODY_UNVOICED_ID,
    MELODY_VOCAB_SIZE,
    SEMANTIC_CODEBOOK_SIZE,
    SEMANTIC_FRAME_RATE,
    validate_melody_contract,
    validate_semantic_contract,
)

CORPUS_FORMAT_VERSION = "oqm.llm.corpus.v1"
SEMANTIC_DTYPE = np.dtype("<u2")  # uint16 little-endian
MELODY_DTYPE = np.dtype("<u1")
DEFAULT_SHARD_BYTES = 512 * 1024 * 1024
RESERVED_CORPUS_METADATA_KEYS = frozenset(
    {
        "format_version",
        "records",
        "total_semantic_frames",
        "total_hours",
        "semantic",
        "melody",
        "tokenizer_revision",
        "manifest",
        "manifest_sha256",
        "corpus_revision",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shard_identities(directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(directory.glob("*.bin"))
    ]


@dataclass(frozen=True)
class TokenSpan:
    shard: str
    frame_offset: int
    num_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard": self.shard,
            "frame_offset": self.frame_offset,
            "num_frames": self.num_frames,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TokenSpan:
        return cls(
            shard=str(payload["shard"]),
            frame_offset=int(payload["frame_offset"]),
            num_frames=int(payload["num_frames"]),
        )


class _StreamWriter:
    def __init__(
        self, directory: Path, prefix: str, dtype: np.dtype, shard_bytes: int
    ) -> None:
        self._directory = directory
        self._prefix = prefix
        self._dtype = dtype
        self._shard_bytes = shard_bytes
        self._directory.mkdir(parents=True, exist_ok=True)
        self._shard_index = 0
        self._handle = None
        self._frames_written = 0
        self._open_shard()

    def _shard_name(self) -> str:
        return f"{self._prefix}-{self._shard_index:05d}.bin"

    def _open_shard(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._frames_written = 0
        self._handle = (self._directory / self._shard_name()).open("wb")

    def append(self, values: np.ndarray) -> TokenSpan:
        array = np.ascontiguousarray(values.astype(self._dtype, copy=False))
        if self._frames_written > 0 and (
            (self._frames_written + array.size) * self._dtype.itemsize
            > self._shard_bytes
        ):
            self._shard_index += 1
            self._open_shard()
        span = TokenSpan(
            shard=self._shard_name(),
            frame_offset=self._frames_written,
            num_frames=int(array.size),
        )
        assert self._handle is not None
        self._handle.write(array.tobytes())
        self._frames_written += int(array.size)
        return span

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None


class CorpusWriter:
    def __init__(
        self,
        root: str | Path,
        *,
        tokenizer_revision: str,
        shard_bytes: int = DEFAULT_SHARD_BYTES,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        self._extra_metadata = dict(extra_metadata or {})
        reserved = sorted(
            RESERVED_CORPUS_METADATA_KEYS.intersection(self._extra_metadata)
        )
        if reserved:
            raise ValueError(
                f"extra_metadata cannot overwrite reserved CorpusWriter fields: {reserved}"
            )
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_handle = (self.root / ".writer.lock").open("a+")
        try:
            fcntl.flock(
                self._lock_handle,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            self._lock_handle.close()
            raise RuntimeError(
                f"Another CorpusWriter is already writing to {self.root}"
            ) from error
        if (self.root / "corpus.json").exists():
            self._release_lock()
            raise FileExistsError(f"A completed corpus already exists at {self.root}")
        residual = [
            path
            for path in (
                self.root / "manifest.jsonl",
                *(self.root / "semantic").glob("*.bin"),
                *(self.root / "melody").glob("*.bin"),
            )
            if path.exists()
        ]
        if residual:
            self._release_lock()
            raise FileExistsError(
                f"The corpus directory contains unfinished writer output: {residual[:4]}"
            )
        self.tokenizer_revision = tokenizer_revision
        self._semantic = _StreamWriter(
            self.root / "semantic", "sem", SEMANTIC_DTYPE, shard_bytes
        )
        self._melody = _StreamWriter(
            self.root / "melody", "mel", MELODY_DTYPE, shard_bytes
        )
        self._manifest = (self.root / "manifest.jsonl").open("w", encoding="utf-8")
        self._count = 0
        self._total_semantic_frames = 0
        self._sample_ids: set[str] = set()
        self._closed = False

    def add(
        self,
        *,
        sample_id: str,
        semantic: np.ndarray,
        record: dict[str, Any],
        melody: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("CorpusWriter is closed")
        if not sample_id:
            raise ValueError("sample_id cannot be empty")
        if sample_id in self._sample_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        self._sample_ids.add(sample_id)
        semantic = np.asarray(semantic)
        if semantic.ndim != 1:
            raise ValueError(
                f"semantic must be one-dimensional; received {semantic.shape}"
            )
        if semantic.size == 0:
            raise ValueError(f"Sample {sample_id} of semantic token is empty")
        if semantic.min() < 0 or semantic.max() >= SEMANTIC_CODEBOOK_SIZE:
            raise ValueError(
                f"Sample {sample_id} of semantic token out of bounds:"
                f"[{int(semantic.min())}, {int(semantic.max())}] is not in [0, {SEMANTIC_CODEBOOK_SIZE})"
            )
        payload = dict(record)
        payload["schema_version"] = "oqm.llm.sample.v1"
        payload["sample_id"] = sample_id
        payload["semantic"] = self._semantic.append(semantic).to_dict()
        if melody is not None:
            melody = np.asarray(melody)
            if melody.ndim != 1:
                raise ValueError(
                    f"melody must be one-dimensional; received {melody.shape}"
                )
            if melody.size and (melody.min() < 0 or melody.max() >= MELODY_VOCAB_SIZE):
                raise ValueError(
                    f"Sample {sample_id} has melody tokens outside "
                    f"[0, {MELODY_VOCAB_SIZE})"
                )
            payload["melody"] = self._melody.append(melody).to_dict()
        else:
            payload["melody"] = None
        self._manifest.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._count += 1
        self._total_semantic_frames += int(semantic.size)
        return payload

    def close(self) -> Path:
        if self._closed:
            path = self.root / "corpus.json"
            if not path.exists():
                raise RuntimeError(
                    "CorpusWriter was aborted before corpus.json was published"
                )
            return path
        if self._count <= 0:
            self.abort()
            raise RuntimeError("Cannot publish an empty LLM corpus")
        self._semantic.close()
        self._melody.close()
        self._manifest.flush()
        self._manifest.close()
        manifest_path = self.root / "manifest.jsonl"
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        metadata = {
            "format_version": CORPUS_FORMAT_VERSION,
            "records": self._count,
            "total_semantic_frames": self._total_semantic_frames,
            "total_hours": self._total_semantic_frames / SEMANTIC_FRAME_RATE / 3600.0,
            "semantic": {
                "frame_rate": SEMANTIC_FRAME_RATE,
                "codebook_size": SEMANTIC_CODEBOOK_SIZE,
                "dtype": SEMANTIC_DTYPE.str,
                "shards": _shard_identities(self.root / "semantic"),
            },
            "melody": {
                "frame_rate": MELODY_FRAME_RATE,
                "vocab_size": MELODY_VOCAB_SIZE,
                "unvoiced_id": MELODY_UNVOICED_ID,
                "dtype": MELODY_DTYPE.str,
                "shards": _shard_identities(self.root / "melody"),
            },
            "tokenizer_revision": self.tokenizer_revision,
            "manifest": "manifest.jsonl",
            "manifest_sha256": manifest_sha256,
            **self._extra_metadata,
        }
        revision_payload = json.dumps(
            metadata, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
        metadata["corpus_revision"] = (
            "sha256:" + hashlib.sha256(revision_payload).hexdigest()
        )
        path = self.root / "corpus.json"
        temporary = self.root / ".corpus.json.tmp"
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
        self._closed = True
        self._release_lock()
        return path

    def abort(self) -> None:
        if self._closed:
            return
        self._semantic.close()
        self._melody.close()
        if not self._manifest.closed:
            self._manifest.flush()
            self._manifest.close()
        (self.root / ".corpus.json.tmp").unlink(missing_ok=True)
        (self.root / "corpus.json").unlink(missing_ok=True)
        self._closed = True
        self._release_lock()

    def _release_lock(self) -> None:
        handle = getattr(self, "_lock_handle", None)
        if handle is None or handle.closed:
            return
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()

    def __enter__(self) -> CorpusWriter:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if exc_type is None:
            try:
                self.close()
            except Exception:
                self.abort()
                raise
        else:
            self.abort()


class ShardReader:
    def __init__(self, directory: str | Path, dtype: np.dtype) -> None:
        self.directory = Path(directory)
        self.dtype = dtype
        self._maps: dict[str, np.memmap] = {}

    def _map(self, shard: str) -> np.memmap:
        cached = self._maps.get(shard)
        if cached is None:
            path = self.directory / shard
            if not path.exists():
                raise FileNotFoundError(f"Missing shard: {path}")
            cached = np.memmap(path, dtype=self.dtype, mode="r")
            self._maps[shard] = cached
        return cached

    def read(self, span: TokenSpan) -> np.ndarray:
        data = self._map(span.shard)
        start = span.frame_offset
        stop = start + span.num_frames
        if stop > data.size:
            raise ValueError(
                f"shard {span.shard} out-of-bounds read:[{start}, {stop}) exceeds {data.size}"
            )
        return np.asarray(data[start:stop])

    def close(self) -> None:
        self._maps.clear()


class CorpusReader:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        metadata_path = self.root / "corpus.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"LLM corpus is missing corpus.json: {self.root}")
        self.metadata: dict[str, Any] = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        if self.metadata.get("format_version") != CORPUS_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported corpus format {self.metadata.get('format_version')!r}; "
                f"expected {CORPUS_FORMAT_VERSION!r}"
            )
        semantic = self.metadata["semantic"]
        validate_semantic_contract(
            semantic["frame_rate"],
            semantic["codebook_size"],
            source=f"corpus {self.root}",
        )
        melody = self.metadata["melody"]
        validate_melody_contract(
            melody["frame_rate"],
            melody["vocab_size"],
            melody["unvoiced_id"],
            source=f"corpus {self.root}",
        )
        self._semantic_reader: ShardReader | None = None
        self._melody_reader: ShardReader | None = None

    @property
    def manifest_path(self) -> Path:
        return self.root / str(self.metadata.get("manifest", "manifest.jsonl"))

    @property
    def tokenizer_revision(self) -> str:
        return str(self.metadata.get("tokenizer_revision", "unknown"))

    @property
    def records(self) -> int:
        return int(self.metadata.get("records", 0))

    def validate_shard_hashes(self, *, require: bool = False) -> None:

        for namespace in ("semantic", "melody"):
            spec = dict(self.metadata.get(namespace) or {})
            identities = spec.get("shards")
            if not isinstance(identities, list):
                if require:
                    raise RuntimeError(
                        f"Corpus {namespace} metadata is missing shard SHA-256 identities"
                    )
                continue
            directory = self.root / namespace
            expected_files = [
                str(identity.get("file") or "") for identity in identities
            ]
            if len(expected_files) != len(set(expected_files)):
                raise RuntimeError(
                    f"Corpus {namespace} shard inventory contains duplicates"
                )
            if any(Path(name).name != name or not name for name in expected_files):
                raise RuntimeError(f"Corpus {namespace} shard name contains a path")
            actual_files = sorted(path.name for path in directory.glob("*.bin"))
            if sorted(expected_files) != actual_files:
                raise RuntimeError(
                    f"Corpus {namespace} shard inventory does not match: "
                    f"metadata={sorted(expected_files)}, actual={actual_files}"
                )
            for identity in identities:
                path = directory / str(identity.get("file") or "")
                if not path.is_file():
                    raise FileNotFoundError(f"Corpus shard does not exist: {path}")
                actual_size = path.stat().st_size
                if actual_size != int(identity.get("bytes", -1)):
                    raise RuntimeError(
                        f"Corpus shard size mismatch: {path} "
                        f"{actual_size}!={identity.get('bytes')}"
                    )
                actual_sha = _sha256_file(path)
                if actual_sha != str(identity.get("sha256") or ""):
                    raise RuntimeError(f"Corpus shard SHA-256 mismatch: {path}")

    def validate_manifest_identity(self, *, require: bool = False) -> None:
        manifest = self.manifest_path
        if manifest.resolve().parent != self.root.resolve():
            raise RuntimeError(
                f"Corpus manifest must be inside the corpus root: {manifest}"
            )
        if not manifest.is_file():
            raise FileNotFoundError(f"Corpus manifest does not exist: {manifest}")
        expected_sha = str(self.metadata.get("manifest_sha256") or "")
        if not expected_sha:
            if require:
                raise RuntimeError("Corpus metadata is missing manifest_sha256")
        elif _sha256_file(manifest) != expected_sha:
            raise RuntimeError("Corpus manifest SHA-256 mismatch")
        expected_revision = str(self.metadata.get("corpus_revision") or "")
        metadata_without_revision = dict(self.metadata)
        metadata_without_revision.pop("corpus_revision", None)
        actual_revision = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    metadata_without_revision,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
        if not expected_revision:
            if require:
                raise RuntimeError("Corpus metadata is missing corpus_revision")
        elif expected_revision != actual_revision:
            raise RuntimeError(
                f"Corpus revision mismatch: {expected_revision} != {actual_revision}"
            )

    def validate_manifest_spans(self) -> None:
        inventories: dict[str, dict[str, int]] = {}
        for namespace, dtype in (
            ("semantic", SEMANTIC_DTYPE),
            ("melody", MELODY_DTYPE),
        ):
            identities = list((self.metadata.get(namespace) or {}).get("shards") or ())
            inventories[namespace] = {}
            for identity in identities:
                name = str(identity.get("file") or "")
                size = int(identity.get("bytes", -1))
                if size < 0 or size % dtype.itemsize:
                    raise RuntimeError(
                        f"{namespace}/{name} byte count is incompatible with its dtype"
                    )
                inventories[namespace][name] = size // dtype.itemsize

        intervals: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
        sample_ids: set[str] = set()
        records = 0
        semantic_frames = 0
        with self.manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = str(record.get("sample_id") or "")
                if not sample_id or sample_id in sample_ids:
                    raise RuntimeError(
                        f"manifest:{line_number} has an empty or duplicate sample_id: "
                        f"{sample_id!r}"
                    )
                sample_ids.add(sample_id)
                records += 1
                for namespace in ("semantic", "melody"):
                    payload = record.get(namespace)
                    if payload is None:
                        continue
                    span = TokenSpan.from_dict(dict(payload))
                    if (
                        Path(span.shard).name != span.shard
                        or span.shard not in inventories[namespace]
                    ):
                        raise RuntimeError(
                            f"{sample_id}: {namespace} references an unregistered shard "
                            f"{span.shard!r}"
                        )
                    if span.frame_offset < 0 or span.num_frames < 0:
                        raise RuntimeError(f"{sample_id}: {namespace} span is negative")
                    stop = span.frame_offset + span.num_frames
                    if stop > inventories[namespace][span.shard]:
                        raise RuntimeError(
                            f"{sample_id}: {namespace} span exceeds the shard"
                        )
                    intervals.setdefault((namespace, span.shard), []).append(
                        (span.frame_offset, stop, sample_id)
                    )
                    if namespace == "semantic":
                        if span.num_frames <= 0:
                            raise RuntimeError(f"{sample_id}: semantic span is empty")
                        semantic_frames += span.num_frames
        if records != self.records:
            raise RuntimeError(
                f"manifest records={records} != corpus records={self.records}"
            )
        if semantic_frames != int(self.metadata.get("total_semantic_frames", -1)):
            raise RuntimeError(
                "Manifest semantic frame count does not match corpus metadata"
            )
        for namespace, inventory in inventories.items():
            for shard, total_frames in inventory.items():
                cursor = 0
                for start, stop, sample_id in sorted(
                    intervals.get((namespace, shard), ())
                ):
                    if start != cursor:
                        kind = "overlap" if start < cursor else "gap"
                        raise RuntimeError(
                            f"{namespace}/{shard} contains a span {kind}: "
                            f"cursor={cursor}, next={start}, sample={sample_id}"
                        )
                    cursor = stop
                if cursor != total_frames:
                    raise RuntimeError(
                        f"{namespace}/{shard} spans cover {cursor} frames; "
                        f"expected {total_frames}"
                    )

    def validate_integrity(self, *, require: bool = False) -> None:
        self.validate_manifest_identity(require=require)
        self.validate_shard_hashes(require=require)
        self.validate_manifest_spans()

    def read_semantic(self, span: TokenSpan) -> np.ndarray:
        if self._semantic_reader is None:
            self._semantic_reader = ShardReader(self.root / "semantic", SEMANTIC_DTYPE)
        return self._semantic_reader.read(span)

    def read_melody(self, span: TokenSpan) -> np.ndarray:
        if self._melody_reader is None:
            self._melody_reader = ShardReader(self.root / "melody", MELODY_DTYPE)
        return self._melody_reader.read(span)

    def close(self) -> None:
        if self._semantic_reader is not None:
            self._semantic_reader.close()
            self._semantic_reader = None
        if self._melody_reader is not None:
            self._melody_reader.close()
            self._melody_reader = None
