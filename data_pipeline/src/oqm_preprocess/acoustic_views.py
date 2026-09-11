from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from .uri import AudioRef, parse


VIEW_SCHEMA = "oqm.acoustic-audio-view.v1"
SPLITS = ("train", "valid", "test")
CONSUMERS = ("vae-stage1", "vae-stage2", "refiner")


class AcousticViewError(RuntimeError):
    pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_identity(handle: BinaryIO, *, size: int | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    remaining = size
    consumed = 0
    while remaining is None or remaining > 0:
        requested = 8 * 1024 * 1024 if remaining is None else min(8 * 1024 * 1024, remaining)
        chunk = handle.read(requested)
        if not chunk:
            break
        digest.update(chunk)
        consumed += len(chunk)
        if remaining is not None:
            remaining -= len(chunk)
    if remaining not in (None, 0):
        raise AcousticViewError(f"Audio payload ended {remaining} bytes early")
    return digest.hexdigest(), consumed


def _read_hints(record: dict[str, Any]) -> dict[str, Any]:
    source = record.get("source")
    if not isinstance(source, dict):
        return {}
    hints = source.get("read_hints")
    return dict(hints) if isinstance(hints, dict) else {}


def _audio_identity(reference: AudioRef, record: dict[str, Any]) -> tuple[str, int]:
    if reference.is_clip:
        raise AcousticViewError("Acoustic training views require whole-track audio URIs")
    if reference.storage_class == "loose":
        return _file_sha256(reference.container), reference.container.stat().st_size
    if reference.storage_class == "zip":
        with zipfile.ZipFile(reference.container) as archive, archive.open(reference.member) as handle:
            return _stream_identity(handle)
    if reference.storage_class in {"tar", "targz"}:
        hints = _read_hints(record)
        offset = int(hints.get("data_offset", -1))
        size = int(hints.get("size", -1))
        if reference.storage_class == "tar" and offset >= 0 and size >= 0:
            with reference.container.open("rb") as handle:
                handle.seek(offset)
                return _stream_identity(handle, size=size)
        mode = "r:gz" if reference.storage_class == "targz" else "r:"
        with tarfile.open(reference.container, mode) as archive:
            handle = archive.extractfile(reference.member)
            if handle is None:
                raise FileNotFoundError(f"{reference.container}::{reference.member}")
            with handle:
                return _stream_identity(handle)
    raise AcousticViewError(
        f"Acoustic training does not support {reference.storage_class!r} audio URIs"
    )


def _records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AcousticViewError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise AcousticViewError(f"Expected an object at {path}:{line_number}")
            yield line_number, value


def _convert_record(record: dict[str, Any], *, source: Path, line_number: int) -> dict[str, Any]:
    sample_id = record.get("sample_id")
    audio = record.get("audio")
    split = record.get("split")
    if not isinstance(sample_id, str) or not sample_id:
        raise AcousticViewError(f"Missing sample_id at {source}:{line_number}")
    if not isinstance(audio, dict) or not isinstance(audio.get("path"), str):
        raise AcousticViewError(f"Missing audio.path at {source}:{line_number}")
    if split not in SPLITS:
        raise AcousticViewError(f"Unsupported split {split!r} at {source}:{line_number}")
    duration = audio.get("duration_sec")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
        raise AcousticViewError(f"Invalid audio.duration_sec at {source}:{line_number}")
    sample_rate = audio.get("sample_rate_hz")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise AcousticViewError(f"Invalid audio.sample_rate_hz at {source}:{line_number}")
    if audio.get("channels") != 2:
        raise AcousticViewError(
            f"Acoustic training requires stereo audio at {source}:{line_number}"
        )

    uri = str(audio["path"])
    reference = parse(uri)
    digest, size_bytes = _audio_identity(reference, record)
    source_metadata = dict(record.get("source") or {})
    source_metadata["cached_source_sha256"] = digest
    source_metadata["file_bytes"] = size_bytes
    return {
        "schema_version": VIEW_SCHEMA,
        "sample_id": sample_id,
        "audio": {
            "uri": uri,
            "sha256": digest,
            "duration_sec": float(duration),
            "sample_rate_hz": sample_rate,
            "channels": 2,
        },
        "duration_sec": float(duration),
        "split": split,
        "source": source_metadata,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_acoustic_views(
    tokenizer_manifest: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    source = Path(tokenizer_manifest).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Tokenizer manifest does not exist: {source}")
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Acoustic view root is not a directory: {output}")
    generated_names = (*CONSUMERS, "ACOUSTIC_VIEW.json", "ACOUSTIC_READY")
    existing = [output / name for name in generated_names if (output / name).exists()]
    if existing and not force:
        raise FileExistsError(
            f"Acoustic view outputs already exist: {existing}; pass --force to rebuild them"
        )

    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    sample_ids: set[str] = set()
    for line_number, record in _records(source):
        converted = _convert_record(record, source=source, line_number=line_number)
        sample_id = converted["sample_id"]
        if sample_id in sample_ids:
            raise AcousticViewError(f"Duplicate sample_id in {source}: {sample_id}")
        sample_ids.add(sample_id)
        by_split[converted["split"]].append(converted)
    missing = [split for split, records in by_split.items() if not records]
    if missing:
        raise AcousticViewError(f"Tokenizer manifest has no records for splits: {', '.join(missing)}")

    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        artifacts: dict[str, dict[str, Any]] = {}
        for consumer in CONSUMERS:
            directory = staging / consumer
            directory.mkdir()
            for split in SPLITS:
                path = directory / f"{split}.jsonl"
                _write_jsonl(path, by_split[split])
                artifacts[f"{consumer}/{path.name}"] = {
                    "records": len(by_split[split]),
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
        descriptor = {
            "schema_version": VIEW_SCHEMA,
            "status": "READY",
            "source_manifest": str(source),
            "source_manifest_sha256": _file_sha256(source),
            "splits": {split: len(records) for split, records in by_split.items()},
            "consumers": list(CONSUMERS),
            "audio_integrity": "sha256",
            "artifacts": artifacts,
        }
        _write_json(staging / "ACOUSTIC_VIEW.json", descriptor)
        _write_json(staging / "ACOUSTIC_READY", descriptor)
        output.mkdir(parents=True, exist_ok=True)
        for target in existing:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        for name in generated_names:
            os.replace(staging / name, output / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return descriptor
