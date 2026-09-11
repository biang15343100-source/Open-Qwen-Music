
from __future__ import annotations

import hashlib
import json
import math
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import yaml

from open_qwen_music.common.archive import read_archive_bytes
from open_qwen_music.common.checkpoint import file_sha256
from open_qwen_music.tokenizer.audio import (
    AudioCredentialError,
    EncryptedZipReadError,
    audio_password_env,
    load_audio,
    password_from_env,
    resample_mono,
)
from open_qwen_music.tokenizer.contracts import SAMPLE_RATE

SEMANTIC_SCHEMA_VERSION = "oqm.semantic-token.v2"
SEMANTIC_RELEASE_SCHEMA_VERSION = "oqm.semantic-token-release.v1"
SEMANTIC_FRAME_HZ = 25.0
SEMANTIC_CODEBOOK_SIZE = 32_768
SEMANTIC_STORAGE_DTYPE = "uint16"


class SemanticMaterializationError(ValueError):
    pass


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticMaterializationError(f"{name} must be an object")
    return value


def require_sha256(value: Any, name: str) -> str:
    digest = str(value or "")
    if len(digest) != 64:
        raise SemanticMaterializationError(f"{name} must be a 64-character SHA-256 digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise SemanticMaterializationError(f"{name} must be a hexadecimal SHA-256 digest") from exc
    return digest.lower()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class FrozenTokenizerBinding:

    binding_path: Path
    binding_sha256: str
    artifact_path: Path
    artifact_sha256: str
    artifact_size_bytes: int
    sidecar_path: Path
    sidecar_sha256: str
    semantic_extractor_revision: str
    sample_rate: int
    frame_rate: float
    codebook_size: int
    codebooks: int

    def identity(self) -> dict[str, Any]:
        return {
            "binding": {
                "path": str(self.binding_path),
                "sha256": self.binding_sha256,
            },
            "artifact": {
                "path": str(self.artifact_path),
                "sha256": self.artifact_sha256,
                "size_bytes": self.artifact_size_bytes,
            },
            "artifact_sidecar": {
                "path": str(self.sidecar_path),
                "sha256": self.sidecar_sha256,
            },
            "tokenizer_revision": self.artifact_sha256,
            "semantic_extractor_revision": self.semantic_extractor_revision,
            "semantic_contract": {
                "sample_rate": self.sample_rate,
                "frame_rate": self.frame_rate,
                "codebook_size": self.codebook_size,
                "codebooks": self.codebooks,
                "storage_dtype": SEMANTIC_STORAGE_DTYPE,
            },
        }


def load_frozen_tokenizer_binding(
    binding_path: str | Path,
    *,
    artifact_override: str | Path | None = None,
) -> FrozenTokenizerBinding:

    binding_path = Path(binding_path).resolve()
    if not binding_path.is_file():
        raise FileNotFoundError(f"Tokenizer binding file does not exist: {binding_path}")
    payload = yaml.safe_load(binding_path.read_text(encoding="utf-8"))
    root = _require_mapping(payload, "binding")
    if root.get("format_version") != "oqm.llm.tokenizer-binding.v2":
        raise SemanticMaterializationError(
            "Tokenizer binding format must be oqm.llm.tokenizer-binding.v2"
        )
    if root.get("status") != "DOWNSTREAM_FROZEN":
        raise SemanticMaterializationError("Tokenizer binding status must be DOWNSTREAM_FROZEN")

    tokenizer = _require_mapping(root.get("tokenizer"), "binding.tokenizer")
    contract = _require_mapping(
        root.get("semantic_contract"), "binding.semantic_contract"
    )
    expected_revision = require_sha256(
        tokenizer.get("revision"), "binding.tokenizer.revision"
    )
    semantic_extractor_revision = require_sha256(
        tokenizer.get("semantic_extractor_revision"),
        "binding.tokenizer.semantic_extractor_revision",
    )
    declared_artifact = Path(str(tokenizer.get("artifact") or ""))
    if not declared_artifact.is_absolute():
        declared_artifact = binding_path.parent / declared_artifact
    artifact_path = (
        Path(artifact_override).resolve()
        if artifact_override is not None
        else declared_artifact.resolve()
    )
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Tokenizer deployment artifact does not exist: {artifact_path}")
    actual_revision = file_sha256(artifact_path)
    if actual_revision != expected_revision:
        raise SemanticMaterializationError(
            f"Tokenizer deployment artifact SHA-256 does not match the binding: "
            f"{actual_revision} != {expected_revision}"
        )

    sidecar_path = artifact_path.with_suffix(artifact_path.suffix + ".json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Tokenizer deployment artifact sidecar does not exist: {sidecar_path}")
    sidecar = _require_mapping(
        json.loads(sidecar_path.read_text(encoding="utf-8")), "artifact sidecar"
    )
    if sidecar.get("format_version") != "oqm.tokenizer.deploy.v1":
        raise SemanticMaterializationError(
            "Tokenizer deployment sidecar format must be oqm.tokenizer.deploy.v1"
        )
    if (
        require_sha256(sidecar.get("tokenizer_revision"), "sidecar.tokenizer_revision")
        != expected_revision
    ):
        raise SemanticMaterializationError(
            "Tokenizer deployment sidecar tokenizer_revision does not match the binding"
        )
    source_identity = _require_mapping(
        sidecar.get("source_checkpoint_identity"),
        "sidecar.source_checkpoint_identity",
    )
    if (
        require_sha256(
            source_identity.get("sha256"), "sidecar.source_checkpoint_identity.sha256"
        )
        != semantic_extractor_revision
    ):
        raise SemanticMaterializationError(
            "Tokenizer deployment sidecar checkpoint identity does not match "
            "semantic_extractor_revision"
        )

    sample_rate = int(contract.get("sample_rate", -1))
    frame_rate = float(contract.get("frame_rate", -1.0))
    codebook_size = int(contract.get("codebook_size", -1))
    codebooks = int(contract.get("codebooks", -1))
    storage_dtype = str(contract.get("storage_dtype") or "")
    if (
        sample_rate != SAMPLE_RATE
        or frame_rate != SEMANTIC_FRAME_HZ
        or codebook_size != SEMANTIC_CODEBOOK_SIZE
        or codebooks != 1
        or storage_dtype != SEMANTIC_STORAGE_DTYPE
    ):
        raise SemanticMaterializationError(
            "Tokenizer binding violates the 24 kHz, 25 Hz, single-codebook, "
            "32768-entry, uint16 contract"
        )
    sidecar_contract = _require_mapping(
        sidecar.get("semantic_contract"), "sidecar.semantic_contract"
    )
    if (
        int(sidecar_contract.get("sample_rate", -1)) != sample_rate
        or float(sidecar_contract.get("frame_rate", -1.0)) != frame_rate
        or int(sidecar_contract.get("codebook_size", -1)) != codebook_size
    ):
        raise SemanticMaterializationError(
            "Tokenizer binding and deployment sidecar semantic contracts do not match"
        )

    return FrozenTokenizerBinding(
        binding_path=binding_path,
        binding_sha256=file_sha256(binding_path),
        artifact_path=artifact_path,
        artifact_sha256=actual_revision,
        artifact_size_bytes=artifact_path.stat().st_size,
        sidecar_path=sidecar_path,
        sidecar_sha256=file_sha256(sidecar_path),
        semantic_extractor_revision=semantic_extractor_revision,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        codebook_size=codebook_size,
        codebooks=codebooks,
    )


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SemanticMaterializationError(
                    f"{path}:{line_number} is not valid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise SemanticMaterializationError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            yield line_number, value


def require_sample_id(record: Mapping[str, Any], *, location: str) -> str:
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise SemanticMaterializationError(f"{location} is missing a non-empty sample_id")
    return sample_id


def _resolve_archive_uri(value: str, base_dir: Path) -> str:
    for scheme in ("tar://", "zip://"):
        if value.startswith(scheme):
            body = value[len(scheme) :]
            if "::" not in body:
                raise SemanticMaterializationError(
                    f"Archive URI is missing the '::' member separator: {value}"
                )
            archive, member = body.split("::", 1)
            archive_path = Path(archive)
            if not archive_path.is_absolute():
                archive_path = (base_dir / archive_path).resolve()
            return f"{scheme}{archive_path}::{member}"
    return value


def resolve_audio_uri(record: Mapping[str, Any], *, base_dir: Path) -> str:
    audio = record.get("audio")
    source = record.get("source")
    audio_fields = audio if isinstance(audio, Mapping) else {}
    source_fields = source if isinstance(source, Mapping) else {}
    value = (
        record.get("render_audio_path")
        or record.get("wav_path")
        or record.get("audio_path")
        or record.get("audio_uri")
        or audio_fields.get("uri")
        or audio_fields.get("path")
        or source_fields.get("uri")
    )
    if not isinstance(value, str) or not value:
        raise SemanticMaterializationError(
            "Record is missing render_audio_path, wav_path, audio_path, audio_uri, audio.uri, audio.path, or source.uri"
        )
    value = _resolve_archive_uri(value, base_dir)
    if value.startswith(("tar://", "zip://")):
        return value
    if value.startswith("file://"):
        path = Path(value.removeprefix("file://"))
        if not path.is_absolute() or "::" in str(path):
            raise SemanticMaterializationError(
                f"file URI must point to an absolute standalone file: {value}"
            )
        return f"file://{path}"
    if "://" in value:
        raise SemanticMaterializationError(f"Unsupported audio URI: {value}")
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _archive_uri_parts(uri: str) -> tuple[str, Path, str]:
    for scheme in ("tar", "zip"):
        prefix = f"{scheme}://"
        if uri.startswith(prefix):
            body = uri.removeprefix(prefix)
            if body.count("::") != 1:
                raise SemanticMaterializationError(
                    f"{scheme} URI must contain exactly one archive::member pair"
                )
            archive, member = body.split("::", 1)
            if not archive or not member:
                raise SemanticMaterializationError(
                    f"{scheme} URI archive and member values must not be empty"
                )
            return scheme, Path(archive), member
    raise SemanticMaterializationError(f"Expected a TAR or ZIP URI: {uri}")


def _optional_integer(
    value: Any,
    name: str,
    *,
    minimum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise SemanticMaterializationError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise SemanticMaterializationError(f"{name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise SemanticMaterializationError(f"{name} must be an integer")
    if result < minimum:
        raise SemanticMaterializationError(f"{name} must be at least {minimum}")
    return result


def _hash_file_slice(path: Path, *, offset: int, size: int) -> str:

    if not path.is_file():
        raise FileNotFoundError(f"Input archive file does not exist: {path}")
    before = path.stat()
    if offset > before.st_size or size > before.st_size - offset:
        raise SemanticMaterializationError(
            f"TAR payload range is out of bounds: offset={offset}, size={size}, "
            f"archive_size={before.st_size}"
        )
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as source:
        source.seek(offset)
        while remaining:
            chunk = source.read(min(32 * 1024 * 1024, remaining))
            if not chunk:
                raise SemanticMaterializationError("Indexed TAR payload ended before the declared size")
            digest.update(chunk)
            remaining -= len(chunk)
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise SemanticMaterializationError(
            "TAR file identity changed while computing the payload SHA"
        )
    return digest.hexdigest()


def _audio_asset_sha256(
    uri: str,
    *,
    archive_offset: int | None,
    archive_size: int | None,
    password: bytes | None,
) -> str:
    if uri.startswith(("tar://", "zip://")):
        scheme, archive_path, member = _archive_uri_parts(uri)
        if (archive_offset is None) != (archive_size is None):
            raise SemanticMaterializationError(
                "audio.archive_offset and audio.archive_size must be declared together"
            )
        if scheme == "tar" and archive_offset is not None:
            assert archive_size is not None
            return _hash_file_slice(
                archive_path,
                offset=archive_offset,
                size=archive_size,
            )
        if archive_offset is not None:
            raise SemanticMaterializationError(
                "audio.archive_offset and audio.archive_size apply only to tar:// URIs"
            )
        if scheme == "tar":
            return hashlib.sha256(read_archive_bytes(uri)).hexdigest()
        digest = hashlib.sha256()
        try:
            with (
                zipfile.ZipFile(archive_path) as archive,
                archive.open(
                    member,
                    "r",
                    pwd=password,
                ) as source,
            ):
                while chunk := source.read(32 * 1024 * 1024):
                    digest.update(chunk)
        except RuntimeError:
            raise EncryptedZipReadError(
                "ZIP member read failed; check the password_env reference and archive integrity"
            ) from None
        return digest.hexdigest()
    path = Path(uri.removeprefix("file://"))
    if not path.is_file():
        raise FileNotFoundError(f"Input audio file does not exist: {path}")
    return file_sha256(path)


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "a finite positive number" if positive else "a finite number"
        raise SemanticMaterializationError(f"{name} must be {qualifier}")
    return result


def validate_semantic_array(
    values: np.ndarray,
    *,
    expected_frames: int | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.uint16:
        raise SemanticMaterializationError(
            f"Semantic dtype must be uint16, received {array.dtype}"
        )
    if array.ndim != 1:
        raise SemanticMaterializationError(
            f"Semantic array must have shape [T], received {array.shape}"
        )
    if expected_frames is not None and int(array.shape[0]) != int(expected_frames):
        raise SemanticMaterializationError(
            f"Semantic frame count {array.shape[0]} does not match {expected_frames}"
        )
    if array.size and int(array.max()) >= SEMANTIC_CODEBOOK_SIZE:
        raise SemanticMaterializationError("Semantic token ID is outside [0, 32768)")
    return array


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as sink:
        np.save(sink, values, allow_pickle=False)
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, path)


def encode_semantic_record(
    *,
    model: Any,
    record: Mapping[str, Any],
    token_path: Path,
    device: torch.device,
    binding: FrozenTokenizerBinding,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    source_manifest_line_number: int,
    materializer_revision: str,
    max_seconds: float = 0.0,
) -> dict[str, Any]:

    sample_id = require_sample_id(
        record, location=f"{source_manifest_path}:{source_manifest_line_number}"
    )
    source_manifest_sha256 = require_sha256(
        source_manifest_sha256, "source_manifest_sha256"
    )
    materializer_revision = require_sha256(
        materializer_revision, "materializer_revision"
    )
    audio = record.get("audio")
    audio_fields = audio if isinstance(audio, Mapping) else {}
    uri = resolve_audio_uri(record, base_dir=source_manifest_path.parent)
    start_value = record.get("start_sec", audio_fields.get("start_sec", 0.0))
    start_sec = _finite_float(start_value or 0.0, "start_sec")
    if start_sec < 0.0:
        raise SemanticMaterializationError("start_sec must be non-negative")
    duration_value = record.get("duration_sec", audio_fields.get("duration_sec"))
    duration_sec = (
        None
        if duration_value in (None, "")
        else _finite_float(duration_value, "duration_sec", positive=True)
    )
    if max_seconds > 0.0:
        duration_sec = (
            max_seconds if duration_sec is None else min(duration_sec, max_seconds)
        )

    archive_offset = _optional_integer(
        audio_fields.get("archive_offset"),
        "audio.archive_offset",
        minimum=0,
    )
    archive_size = _optional_integer(
        audio_fields.get("archive_size"),
        "audio.archive_size",
        minimum=1,
    )
    password_env = audio_password_env(record)
    password: bytes | None = None
    if password_env is not None:
        if not uri.startswith("zip://"):
            raise AudioCredentialError(
                f"Audio credential environment variable {password_env} can only be used "
                "with a zip:// audio URI"
            )
        password = password_from_env(password_env)

    actual_input_sha256 = _audio_asset_sha256(
        uri,
        archive_offset=archive_offset,
        archive_size=archive_size,
        password=password,
    )
    declared_input_sha256s = (
        ("input_audio_sha256", record.get("input_audio_sha256")),
        ("audio.sha256", audio_fields.get("sha256")),
        ("audio.payload_sha256", audio_fields.get("payload_sha256")),
    )
    for field, declared_value in declared_input_sha256s:
        if declared_value is None:
            continue
        declared_sha256 = require_sha256(declared_value, field)
        if declared_sha256 != actual_input_sha256:
            raise SemanticMaterializationError(
                f"{field} does not match the SHA of the decoded audio asset"
            )

    waveform, source_rate = load_audio(
        uri,
        start_sec=start_sec,
        duration_sec=duration_sec,
        archive_offset=archive_offset,
        archive_size=archive_size,
        password=password,
        asset_id=audio_fields.get("asset_id") if uri.startswith("tar://") else None,
        payload_sha256=(
            audio_fields.get("payload_sha256") if uri.startswith("tar://") else None
        ),
        asset_revision=(
            audio_fields.get("asset_revision") if uri.startswith("tar://") else None
        ),
        shard_sha256=(
            audio_fields.get("shard_sha256") if uri.startswith("tar://") else None
        ),
    )
    waveform = resample_mono(waveform, source_rate, binding.sample_rate)
    if waveform.ndim != 1 or waveform.numel() <= 0:
        raise SemanticMaterializationError(
            "Resampled audio must be a non-empty mono tensor with shape [N]"
        )
    actual_duration_sec = waveform.shape[-1] / binding.sample_rate
    expected_frames = int(round(actual_duration_sec * binding.frame_rate))
    if expected_frames <= 0:
        raise SemanticMaterializationError("Audio is shorter than one semantic frame")

    batch_waveform = waveform.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(batch_waveform, dtype=torch.bool)
    with torch.inference_mode():
        result = model.encode_audio(
            batch_waveform,
            binding.sample_rate,
            attention_mask=attention_mask,
        )
    if float(result.frame_rate) != binding.frame_rate:
        raise SemanticMaterializationError("frame_rate violates the frozen tokenizer contract")
    if int(result.codebook_size) != binding.codebook_size:
        raise SemanticMaterializationError("codebook_size violates the frozen tokenizer contract")
    if str(result.tokenizer_revision) != binding.artifact_sha256:
        raise SemanticMaterializationError("tokenizer_revision does not match the artifact")
    ids = result.token_ids[0].detach().cpu()
    mask = result.frame_mask[0].detach().cpu().bool()
    if ids.shape != mask.shape or ids.ndim != 1:
        raise SemanticMaterializationError(
            "token_ids and frame_mask must be one-dimensional tensors with equal shape"
        )
    if ids.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise SemanticMaterializationError("token_ids must be an integer tensor")
    valid_ids = ids[mask].to(torch.int64).numpy()
    if valid_ids.size and (
        int(valid_ids.min()) < 0 or int(valid_ids.max()) >= binding.codebook_size
    ):
        raise SemanticMaterializationError("Semantic token ID is outside [0, 32768)")
    values = valid_ids.astype(np.uint16, copy=False)
    validate_semantic_array(values, expected_frames=expected_frames)

    source_duration_value = record.get(
        "source_duration_sec",
        audio_fields.get("source_duration_sec", actual_duration_sec),
    )
    source_duration_sec = _finite_float(
        source_duration_value, "source_duration_sec", positive=True
    )
    if duration_sec is not None:
        source_duration_sec = min(source_duration_sec, duration_sec)
    if int(round(source_duration_sec * binding.frame_rate)) != expected_frames:
        raise SemanticMaterializationError(
            "source_duration_sec does not match the decoded 25 Hz frame count"
        )
    source_start_value = record.get(
        "source_start_sec", audio_fields.get("source_start_sec", start_sec)
    )
    source_start_sec = _finite_float(source_start_value or 0.0, "source_start_sec")
    if source_start_sec < 0.0:
        raise SemanticMaterializationError("source_start_sec must be non-negative")
    source_audio_sha256 = require_sha256(
        record.get("source_audio_sha256")
        or audio_fields.get("source_sha256")
        or actual_input_sha256,
        "source_audio_sha256",
    )

    _atomic_save_npy(token_path, values)
    token_sha256 = file_sha256(token_path)
    return {
        "schema_version": SEMANTIC_SCHEMA_VERSION,
        "sample_id": sample_id,
        "token_uri": str(token_path.resolve()),
        "token_sha256": token_sha256,
        "shape": [int(values.shape[0])],
        "dtype": SEMANTIC_STORAGE_DTYPE,
        "num_frames": int(values.shape[0]),
        "frame_hz": binding.frame_rate,
        "codebook_size": binding.codebook_size,
        "tokenizer_revision": binding.artifact_sha256,
        "semantic_extractor_revision": binding.semantic_extractor_revision,
        "tokenizer_checkpoint_sha256": binding.semantic_extractor_revision,
        "tokenizer_artifact_uri": str(binding.artifact_path),
        "tokenizer_artifact_sha256": binding.artifact_sha256,
        "tokenizer_artifact_sidecar_sha256": binding.sidecar_sha256,
        "tokenizer_binding_sha256": binding.binding_sha256,
        "materializer_revision": materializer_revision,
        "source_manifest_uri": str(source_manifest_path.resolve()),
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest_line_number": int(source_manifest_line_number),
        "input_audio_sha256": actual_input_sha256,
        "source_audio_sha256": source_audio_sha256,
        "source_start_sec": source_start_sec,
        "source_duration_sec": source_duration_sec,
        "read_start_sec": start_sec,
        "read_duration_sec": actual_duration_sec,
    }


def validate_materialized_record(
    record: Mapping[str, Any],
    *,
    binding: FrozenTokenizerBinding,
    source_manifest_sha256: str,
    token_root: Path,
    materializer_revision: str | None = None,
) -> tuple[Path, np.ndarray]:

    if record.get("schema_version") != SEMANTIC_SCHEMA_VERSION:
        raise SemanticMaterializationError("Semantic record schema_version is invalid")
    require_sample_id(record, location="semantic record")
    expected = {
        "tokenizer_revision": binding.artifact_sha256,
        "semantic_extractor_revision": binding.semantic_extractor_revision,
        "tokenizer_checkpoint_sha256": binding.semantic_extractor_revision,
        "tokenizer_artifact_sha256": binding.artifact_sha256,
        "tokenizer_artifact_sidecar_sha256": binding.sidecar_sha256,
        "tokenizer_binding_sha256": binding.binding_sha256,
        "source_manifest_sha256": source_manifest_sha256,
    }
    if materializer_revision is not None:
        expected["materializer_revision"] = require_sha256(
            materializer_revision, "materializer_revision"
        )
    for field, value in expected.items():
        if record.get(field) != value:
            raise SemanticMaterializationError(
                f"Semantic record {field} does not match: {record.get(field)!r} != {value!r}"
            )
    for field in (
        "token_sha256",
        "input_audio_sha256",
        "source_audio_sha256",
        "materializer_revision",
    ):
        require_sha256(record.get(field), field)
    if record.get("dtype") != SEMANTIC_STORAGE_DTYPE:
        raise SemanticMaterializationError("Semantic record dtype must be uint16")
    if float(record.get("frame_hz", -1.0)) != binding.frame_rate:
        raise SemanticMaterializationError("Semantic record frame_hz is invalid")
    if int(record.get("codebook_size", -1)) != binding.codebook_size:
        raise SemanticMaterializationError("Semantic record codebook_size is invalid")
    shape = record.get("shape")
    num_frames = int(record.get("num_frames", -1))
    if shape != [num_frames] or num_frames <= 0:
        raise SemanticMaterializationError("Semantic record shape and num_frames are invalid")
    line_number = int(record.get("source_manifest_line_number", -1))
    if line_number <= 0:
        raise SemanticMaterializationError("source_manifest_line_number must be positive")
    duration = _finite_float(
        record.get("source_duration_sec"),
        "source_duration_sec",
        positive=True,
    )
    if int(round(duration * binding.frame_rate)) != num_frames:
        raise SemanticMaterializationError(
            "Semantic frame count does not match source_duration_sec"
        )

    token_path = Path(str(record.get("token_uri") or "")).resolve()
    resolved_root = token_root.resolve()
    try:
        token_path.relative_to(resolved_root)
    except ValueError as exc:
        raise SemanticMaterializationError(
            f"token_uri escapes the output root: {token_path}"
        ) from exc
    if not token_path.is_file():
        raise FileNotFoundError(f"Semantic token file does not exist: {token_path}")
    if file_sha256(token_path) != record.get("token_sha256"):
        raise SemanticMaterializationError("Semantic token file SHA does not match")
    values = validate_semantic_array(
        np.load(token_path, allow_pickle=False),
        expected_frames=num_frames,
    )
    return token_path, values
