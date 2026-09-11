
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..condition import ConditionRenderConfig, MusicCondition
from ..contracts import DISCARDED_QUALITY_BUCKET, QUALITY_BUCKETS


INDEX_FORMAT_VERSION = "oqm.llm.manifest-index.v5"


CONDITION_TOKENS_UNKNOWN = -1
_TRAINING_QUALITY_BUCKETS = frozenset(QUALITY_BUCKETS) - {DISCARDED_QUALITY_BUCKET}


_UNBOUNDED_CONDITION_BUDGET = 1 << 30


_FINGERPRINT_PROBE = "[tags] pop | warm | piano\n[lyrics]\nmidnight light 123"


def condition_encoder_fingerprint(
    text_encoder: Any,
    condition_config: ConditionRenderConfig | None = None,
) -> str:
    condition_config = condition_config or ConditionRenderConfig()
    ids = list(text_encoder.encode(_FINGERPRINT_PROBE, add_special_tokens=False))
    payload = "|".join(
        [
            condition_config.revision,
            type(text_encoder).__name__,
            str(getattr(text_encoder, "_oqm_artifact_revision", "unknown")),
            ",".join(str(int(token)) for token in ids),
        ]
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _condition_length_fn(
    text_encoder: Any,
    condition_config: ConditionRenderConfig | None = None,
):
    from ..sequence import SequenceBuilder, SequenceConfig

    builder = SequenceBuilder(
        None,  # type: ignore[arg-type]
        text_encoder,
        SequenceConfig(max_condition_tokens=_UNBOUNDED_CONDITION_BUDGET),
        condition_config or ConditionRenderConfig(),
    )

    def length_of(record: dict[str, Any]) -> int:
        ids, truncated = builder.encode_condition(MusicCondition.from_record(record))
        if truncated:  # pragma: no cover -  2^30,
            raise RuntimeError("Conditional length detection actually triggered truncation,_UNBOUNDED_CONDITION_BUDGET invalid")
        return len(ids)

    return length_of


CATEGORICAL_FIELDS: tuple[str, ...] = (
    "split",
    "quality.bucket",
    "quality.genre",
    "language",
    "is_instrumental",
    "has_melody",


    "quality.promoted",
)


def _dotted(record: dict[str, Any], path: str) -> Any:
    node: Any = record
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _field_value(record: dict[str, Any], field: str) -> str:
    if field == "has_melody":
        return "true" if record.get("melody") else "false"
    if field == "is_instrumental":
        return "true" if record.get("is_instrumental") else "false"
    if field == "quality.promoted":

        return "true" if (record.get("quality") or {}).get("promoted") else "false"
    value = _dotted(record, field)
    if value is None or value == "":
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def index_dir(manifest: str | Path) -> Path:
    return Path(str(manifest) + ".index")


def build_index(
    manifest: str | Path,
    *,
    force: bool = False,
    text_encoder: Any | None = None,
    strict_metadata: bool = False,
    condition_config: ConditionRenderConfig | None = None,
) -> Path:
    manifest_path = Path(manifest)
    directory = index_dir(manifest_path)
    stat = manifest_path.stat()
    fingerprint = (
        None
        if text_encoder is None
        else condition_encoder_fingerprint(text_encoder, condition_config)
    )
    if not force and _index_is_fresh(
        directory, stat, fingerprint, strict_metadata=bool(strict_metadata)
    ):
        return directory

    directory.mkdir(parents=True, exist_ok=True)


    #


    with (directory / ".build.lock").open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)

        if not force and _index_is_fresh(
            directory, stat, fingerprint, strict_metadata=bool(strict_metadata)
        ):
            return directory
        return _build_index_locked(
            manifest_path,
            directory,
            stat,
            fingerprint,
            text_encoder=text_encoder,
            strict_metadata=bool(strict_metadata),
            condition_config=condition_config,
        )


def _index_is_fresh(
    directory: Path,
    stat: os.stat_result,
    fingerprint: str | None,
    *,
    strict_metadata: bool,
) -> bool:
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    fresh = (
        existing.get("format_version") == INDEX_FORMAT_VERSION
        and existing.get("source_size") == stat.st_size
        and existing.get("source_mtime_ns") == stat.st_mtime_ns
    )
    return (
        fresh
        and (not strict_metadata or existing.get("strict_metadata") is True)
        and (fingerprint is None or existing.get("condition_fingerprint") == fingerprint)
    )


def _atomic_write(path: Path, payload: np.ndarray | str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if isinstance(payload, str):
        temporary.write_text(payload, encoding="utf-8")
    else:
        payload.tofile(temporary)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_index_locked(
    manifest_path: Path,
    directory: Path,
    stat: os.stat_result,
    fingerprint: str | None,
    *,
    text_encoder: Any | None,
    strict_metadata: bool,
    condition_config: ConditionRenderConfig | None,
) -> Path:
    condition_length = (
        None
        if text_encoder is None
        else _condition_length_fn(text_encoder, condition_config)
    )
    offsets: list[int] = []
    semantic_frames: list[int] = []
    melody_frames: list[int] = []
    condition_tokens: list[int] = []
    num_sections: list[int] = []
    vocabularies: dict[str, dict[str, int]] = {field: {} for field in CATEGORICAL_FIELDS}
    codes: dict[str, list[int]] = {field: [] for field in CATEGORICAL_FIELDS}
    seen_sample_ids: set[str] = set()
    source_digest = hashlib.sha256()

    with manifest_path.open("rb") as handle:
        offset = 0
        for raw in handle:
            source_digest.update(raw)
            stripped = raw.strip()
            if stripped:
                record = json.loads(stripped)
                sample_id = str(record.get("sample_id") or "")
                if not sample_id:
                    raise ValueError("Manifest record is missing sample_id")
                if sample_id in seen_sample_ids:
                    raise ValueError(f"Duplicate manifest sample_id: {sample_id}")
                seen_sample_ids.add(sample_id)
                if strict_metadata:
                    split = record.get("split")
                    if split not in {"train", "valid", "test"}:
                        raise ValueError(f"{sample_id}: invalid split={split!r}")
                    quality = record.get("quality")
                    if not isinstance(quality, dict):
                        raise ValueError(f"{sample_id}: quality must be an object")
                    bucket = quality.get("bucket")
                    if bucket not in _TRAINING_QUALITY_BUCKETS:
                        raise ValueError(
                            f"{sample_id}: quality.bucket={bucket!r}; expected Q1 through Q6"
                        )
                    if str(quality.get("genre") or "").strip().lower() in {"", "unknown"}:
                        raise ValueError(f"{sample_id}: quality.genre cannot be empty/unknown")
                    if str(record.get("language") or "").strip().lower() in {"", "unknown"}:
                        raise ValueError(f"{sample_id}: language cannot be empty/unknown")
                    if type(record.get("is_instrumental")) is not bool:
                        raise ValueError(f"{sample_id}: is_instrumental must be true bool")
                offsets.append(offset)
                semantic_frames.append(int((record.get("semantic") or {}).get("num_frames", 0)))
                melody_frames.append(int((record.get("melody") or {}).get("num_frames", 0)))
                num_sections.append(len(record.get("sections") or ()))
                condition_tokens.append(
                    CONDITION_TOKENS_UNKNOWN
                    if condition_length is None
                    else condition_length(record)
                )
                for field in CATEGORICAL_FIELDS:
                    value = _field_value(record, field)
                    table = vocabularies[field]
                    if value not in table:
                        if len(table) >= 65535:
                            raise ValueError(
                                f"Categorical field {field} has more than 65535 values and "
                                "cannot use uint16 encoding"
                            )
                        table[value] = len(table)
                    codes[field].append(table[value])
            offset += len(raw)

    _atomic_write(directory / "offsets.i64", np.asarray(offsets, dtype=np.int64))
    _atomic_write(
        directory / "semantic_frames.i32", np.asarray(semantic_frames, dtype=np.int32)
    )
    _atomic_write(directory / "melody_frames.i32", np.asarray(melody_frames, dtype=np.int32))
    _atomic_write(
        directory / "condition_tokens.i32", np.asarray(condition_tokens, dtype=np.int32)
    )
    _atomic_write(directory / "num_sections.i32", np.asarray(num_sections, dtype=np.int32))
    for field in CATEGORICAL_FIELDS:
        _atomic_write(
            directory / f"{_safe(field)}.u16", np.asarray(codes[field], dtype=np.uint16)
        )
    array_files = [
        "offsets.i64",
        "semantic_frames.i32",
        "melody_frames.i32",
        "condition_tokens.i32",
        "num_sections.i32",
        *(f"{_safe(field)}.u16" for field in CATEGORICAL_FIELDS),
    ]

    known = np.asarray(condition_tokens, dtype=np.int64)
    known = known[known >= 0]
    metadata = {
        "format_version": INDEX_FORMAT_VERSION,
        "manifest": str(manifest_path),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "source_sha256": source_digest.hexdigest(),
        "records": len(offsets),
        "condition_fingerprint": fingerprint,
        "strict_metadata": strict_metadata,


        "condition_tokens_stats": None
        if known.size == 0
        else {
            "known": int(known.size),
            "mean": float(known.mean()),
            "p50": float(np.percentile(known, 50)),
            "p90": float(np.percentile(known, 90)),
            "p99": float(np.percentile(known, 99)),
            "max": int(known.max()),
        },
        "categorical_fields": {
            field: {"file": f"{_safe(field)}.u16", "values": list(vocabularies[field])}
            for field in CATEGORICAL_FIELDS
        },
        "array_files": {
            name: {
                "bytes": (directory / name).stat().st_size,
                "sha256": _sha256_file(directory / name),
            }
            for name in array_files
        },
    }


    _atomic_write(
        directory / "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2)
    )
    return directory


def _safe(field: str) -> str:
    return field.replace(".", "__")


class ManifestIndex:

    def __init__(self, manifest: str | Path) -> None:
        self.manifest_path = Path(manifest)
        self.directory = index_dir(self.manifest_path)
        with (self.directory / ".build.lock").open("r") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_SH)
            self._open_index()

    def _open_index(self) -> None:
        metadata_path = self.directory / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Manifest index is missing at {self.directory}; run build_index() first"
            )
        self.metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("format_version") != INDEX_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported index format version: {self.metadata.get('format_version')}"
            )
        stat = self.manifest_path.stat()
        if (
            self.metadata.get("source_size") != stat.st_size
            or self.metadata.get("source_mtime_ns") != stat.st_mtime_ns
        ):
            raise RuntimeError(
                f"Manifest index is stale because {self.manifest_path} changed; "
                "rebuild it with build_index(..., force=True)"
            )
        self.records = int(self.metadata["records"])
        shape = (self.records,)
        self.offsets = np.memmap(
            self.directory / "offsets.i64", dtype=np.int64, mode="r", shape=shape
        )
        self.semantic_frames = np.memmap(
            self.directory / "semantic_frames.i32", dtype=np.int32, mode="r", shape=shape
        )
        self.melody_frames = np.memmap(
            self.directory / "melody_frames.i32", dtype=np.int32, mode="r", shape=shape
        )
        self.condition_tokens = np.memmap(
            self.directory / "condition_tokens.i32", dtype=np.int32, mode="r", shape=shape
        )
        self.num_sections = np.memmap(
            self.directory / "num_sections.i32", dtype=np.int32, mode="r", shape=shape
        )
        self.condition_fingerprint: str | None = self.metadata.get("condition_fingerprint")
        self._categorical: dict[str, np.memmap] = {}
        self._vocabularies: dict[str, list[str]] = {}
        for field, spec in self.metadata["categorical_fields"].items():
            self._categorical[field] = np.memmap(
                self.directory / spec["file"], dtype=np.uint16, mode="r", shape=shape
            )
            self._vocabularies[field] = list(spec["values"])

    def validate_integrity(
        self,
        *,
        expected_manifest_sha256: str | None = None,
    ) -> None:
        actual_manifest_sha = _sha256_file(self.manifest_path)
        recorded_manifest_sha = str(self.metadata.get("source_sha256") or "")
        if not recorded_manifest_sha or actual_manifest_sha != recorded_manifest_sha:
            raise RuntimeError("Manifest index source SHA-256 does not match")
        if (
            expected_manifest_sha256 is not None
            and actual_manifest_sha != str(expected_manifest_sha256)
        ):
            raise RuntimeError("Manifest index is not bound to the current corpus manifest")
        array_files = self.metadata.get("array_files")
        if not isinstance(array_files, dict) or not array_files:
            raise RuntimeError("Manifest index is missing array content identities")
        expected_names = set(array_files)
        actual_names = {
            path.name
            for path in self.directory.iterdir()
            if path.is_file()
            and path.name not in {"metadata.json", ".build.lock"}
            and not path.name.endswith(".tmp")
        }
        if expected_names != actual_names:
            raise RuntimeError(
                f"Manifest index array inventory does not match: {sorted(expected_names)} != "
                f"{sorted(actual_names)}"
            )
        for name, identity in array_files.items():
            path = self.directory / name
            if path.stat().st_size != int(identity.get("bytes", -1)):
                raise RuntimeError(f"Manifest index array size does not match: {name}")
            if _sha256_file(path) != str(identity.get("sha256") or ""):
                raise RuntimeError(f"Manifest index array SHA-256 does not match: {name}")

    def condition_tokens_known(self) -> np.ndarray:
        return np.asarray(self.condition_tokens) >= 0

    @property
    def revision(self) -> str:
        payload = json.dumps(
            self.metadata,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def condition_tokens_coverage(self) -> float:
        if self.records == 0:  # pragma: no cover - ManifestIndex
            return 0.0
        return float(self.condition_tokens_known().mean())

    def values(self, field: str) -> list[str]:
        if field not in self._vocabularies:
            raise KeyError(
                f"Index has no field {field}; available fields: {sorted(self._vocabularies)}"
            )
        return self._vocabularies[field]

    def codes(self, field: str) -> np.ndarray:
        if field not in self._categorical:
            raise KeyError(
                f"Index has no field {field}; available fields: {sorted(self._categorical)}"
            )
        return np.asarray(self._categorical[field])

    def code_of(self, field: str, value: str) -> int | None:
        table = self.values(field)
        try:
            return table.index(value)
        except ValueError:
            return None

    def field_values(self, field: str) -> np.ndarray:
        table = np.asarray(self.values(field), dtype=object)
        return table[self.codes(field)]

    def indices_where(self, field: str, allowed: set[str]) -> np.ndarray:
        codes = {self.code_of(field, value) for value in allowed}
        codes.discard(None)
        if not codes:
            return np.empty(0, dtype=np.int64)
        mask = np.isin(self.codes(field), np.asarray(sorted(codes), dtype=np.uint16))
        return np.nonzero(mask)[0].astype(np.int64)
