
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from array import array
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from .conditioning import REWRITER_SCHEMA_VERSION, TextEncoderProvenance
from .contracts import (
    AUDIO_CHANNELS,
    LATENT_DIM,
    LATENT_FRAME_HZ,
    SAMPLE_RATE,
    SEMANTIC_CODEBOOK_SIZE,
    samples_to_latent_frames,
    validate_latent_layout,
)
from .text_cache import (
    RENDERER_DATA_TEXT_REFERENCE_SCHEMA,
    RenderTextCacheLoader,
    RenderTextCondition,
    read_renderer_data_content_addressed_text_pair,
    read_text_cache_entry,
)

RENDER_SAMPLE_SCHEMA = "oqm.render-sample.v1"
RENDER_SAMPLE_READY_SCHEMA = "oqm.render-sample-ready.v1"
RENDER_SAMPLE_READY_STATUS = "RENDER_SAMPLE_READY"
RENDER_SAMPLE_INDEX_SCHEMA = "oqm.render-sample-index.v1"
RENDER_SAMPLE_INDEX_STATUS = "RENDER_SAMPLE_INDEX_READY"
RENDER_SAMPLE_INDEX_PAYLOAD_SCHEMA = "oqm.render-sample-index-npy.v1"
RENDER_SAMPLE_INDEX_PAYLOAD_DTYPE = np.dtype(
    [
        ("offset", "<u8"),
        ("size", "<u8"),
        ("line_number", "<u8"),
        ("frame_length", "<u4"),
    ]
)
MAX_RENDER_FRAMES = 9_000
RENDER_SAMPLE_REQUIRED_CHECKS = {
    "sample_sets_equal",
    "source_audio_sha_equal",
    "derived_audio_sha_equal",
    "time_ranges_equal",
    "semantic_latent_frames_equal",
    "quality_profile_pass",
    "artifact_sha_verified",
    "split_groups_disjoint",
}
RENDER_SAMPLE_REQUIRED_GROUP_FIELDS = (
    "recording_group_id",
    "performance_group_id",
    "composition_group_id",
    "song_group_id",
)
POSTERIOR_SEED_DERIVATION = "sha256-little-endian-63-v1"
RENDERER_DATA_DEFERRED_PARENT_TEXT_SCHEMA = (
    "oqm.render.renderer_data-deferred-parent-text.v1"
)


def _content_addressed_record(
    loader: RenderTextCacheLoader,
    *,
    role: str,
    content_sha256: str,
) -> tuple[str, Mapping[str, Any]]:
    record_id = f"renderer_data-{role}:{content_sha256}"
    return record_id, loader.record(record_id)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field}must be a mapping")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field}must be a SHA-256 string")
    digest = value.lower()
    if len(digest) != 64:
        raise ValueError(f"{field}must be a 64-character SHA-256")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"{field}is not hexadecimal SHA-256") from exc
    return digest


def _integer(value: Any, *, field: str, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field}must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field}must be >= {minimum}")
    return value


def _number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field}must be a finite value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field}must be a finite value")
    return result


def _artifact_path(
    value: Mapping[str, Any],
    *,
    base_dir: Path,
    field: str,
) -> Path:
    raw_uri = value.get("uri")
    if not isinstance(raw_uri, (str, os.PathLike)):
        raise ValueError(f"{field}.uri must be a local file path")
    uri = os.fspath(raw_uri)
    if not uri or "://" in uri:
        raise ValueError(f"{field}.uri must be a local file path")
    path = Path(uri)
    if path.is_absolute():
        return path.resolve()
    root = base_dir.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field}.uri escapes the manifest directory: {uri!r}") from exc
    return resolved


def _seconds_to_samples(value: Any, *, field: str, positive: bool) -> int:
    seconds = _number(value, field=field)
    if not math.isfinite(seconds) or (seconds <= 0.0 if positive else seconds < 0.0):
        qualifier = " " if positive else "  "
        raise ValueError(f"{field}must be a finite {qualifier}number")
    samples = round(seconds * SAMPLE_RATE)
    if not math.isclose(
        seconds * SAMPLE_RATE,
        float(samples),
        rel_tol=0.0,
        abs_tol=1.0e-4,
    ):
        raise ValueError(f"{field}must fall exactly on the {SAMPLE_RATE} Hz sampling grid")
    return int(samples)


def _expected_posterior_sample_seed(
    *,
    base_seed: int,
    sample_id: str,
    derived_audio_sha256: str,
) -> int:
    digest = _sha256(
        derived_audio_sha256,
        field=f"{sample_id}.latent.derived_audio_sha256",
    )
    payload = f"{base_seed}:0:{sample_id}:{digest}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def _json_object_from_bytes(payload: bytes, *, source: Path) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON cannot be parsed: {source}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be a mapping: {source}")
    return value


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_bound_latent_layout(
    layout: Any,
    layout_sha256: Any,
    *,
    field: str,
) -> dict[str, Any]:

    if (layout is None) != (layout_sha256 is None):
        raise ValueError(f"{field}content and SHA-256 must either both exist or both be absent")
    if layout is None:
        raise ValueError(f"{field} is missing")
    normalized = validate_latent_layout(
        _mapping(layout, field=field)
    )
    declared_sha256 = _sha256(layout_sha256, field=f"{field}_sha256")
    if declared_sha256 != _canonical_sha(normalized):
        raise ValueError(f"{field} SHA does not match its content")
    return normalized


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_object(
    payload: bytes,
    *,
    source: Path,
    line_number: int,
) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"JSONL is not UTF-8: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}:{line_number} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{source}:{line_number} must be an object")
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


class _IndexedCanonicalRecords(Sequence[dict[str, Any]]):

    def __init__(self, dataset: "CanonicalRenderSampleDataset") -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(
        self,
        index: int | slice,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [
                self._dataset._read_record(item)
                for item in range(*index.indices(len(self)))
            ]
        return self._dataset._read_record(index)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self._dataset.iter_records()


class CanonicalRenderSampleDataset(Dataset[dict[str, Any]]):

    def __init__(
        self,
        manifest: str | Path,
        *,
        expected_revisions: Mapping[str, str],
        expected_artifacts: Mapping[str, Any] | None = None,
        text_model_id: str,
        split: str = "train",
        required_quality_profile: str | None = None,
        expected_manifest_sha256: str | None = None,
        expected_records: int | None = None,
        ready_path: str | Path | None = None,
        expected_ready_sha256: str | None = None,
        expected_release_revision: str | None = None,
        expected_ready_text_cache_revision: str | None = None,
        index_path: str | Path | None = None,
        expected_index_sha256: str | None = None,
        verify_manifest_sha256: bool = True,
        validate_index_records: bool = True,
        tags_text_cache_loader: RenderTextCacheLoader | None = None,
        lyrics_text_cache_loader: RenderTextCacheLoader | None = None,
        allow_renderer_data_deferred_parent_text: bool = False,
        resolve_renderer_data_deferred_parent_text: bool = False,
    ) -> None:
        self.manifest = Path(manifest).resolve()
        if (tags_text_cache_loader is None) != (lyrics_text_cache_loader is None):
            raise ValueError(
                "RendererData tags and lyrics content-cache loaders must be provided together"
            )
        self.tags_text_cache_loader = tags_text_cache_loader
        self.lyrics_text_cache_loader = lyrics_text_cache_loader
        if not isinstance(allow_renderer_data_deferred_parent_text, bool):
            raise TypeError("allow_renderer_data_deferred_parent_text must be bool")
        self.allow_renderer_data_deferred_parent_text = (
            allow_renderer_data_deferred_parent_text
        )
        if not isinstance(resolve_renderer_data_deferred_parent_text, bool):
            raise TypeError("resolve_renderer_data_deferred_parent_text must be bool")
        if resolve_renderer_data_deferred_parent_text and (
            not allow_renderer_data_deferred_parent_text
            or tags_text_cache_loader is None
            or lyrics_text_cache_loader is None
        ):
            raise ValueError(
                "Deferred full text in renderer data must be explicitly enabled with paired content cache"
            )
        self.resolve_renderer_data_deferred_parent_text = (
            resolve_renderer_data_deferred_parent_text
        )
        if not isinstance(expected_revisions, Mapping) or any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in expected_revisions.items()
        ):
            raise TypeError("expected_revisions must map strings to strings")
        if expected_artifacts is not None and (
            not isinstance(expected_artifacts, Mapping)
            or any(not isinstance(name, str) for name in expected_artifacts)
        ):
            raise TypeError("expected_artifacts must be a string-keyed mapping")
        self.expected_revisions = dict(expected_revisions)
        if expected_ready_text_cache_revision is not None and (
            not isinstance(expected_ready_text_cache_revision, str)
            or not expected_ready_text_cache_revision
        ):
            raise TypeError("expected_ready_text_cache_revision must be a non-empty string")
        self.expected_ready_text_cache_revision = (
            expected_ready_text_cache_revision
        )
        self.expected_artifacts = dict(expected_artifacts or {})
        if required_quality_profile is not None and (
            not isinstance(required_quality_profile, str)
            or not required_quality_profile
        ):
            raise TypeError("required_quality_profile must be a non-empty string")
        self.required_quality_profile = required_quality_profile
        self._dataset_text_cache_identity: dict[str, str] = {}
        if expected_records is not None and (
            not isinstance(expected_records, int)
            or isinstance(expected_records, bool)
            or expected_records <= 0
        ):
            raise ValueError("expected_records must be a positive integer")
        if (ready_path is None) != (expected_ready_sha256 is None):
            raise ValueError("ready_path and expected_ready_sha256 must be provided together")
        if (index_path is None) != (expected_index_sha256 is None):
            raise ValueError("index_path and expected_index_sha256 must be provided together")
        if index_path is not None and ready_path is None:
            raise ValueError("Published sample index must be bound to READY")
        if not isinstance(verify_manifest_sha256, bool):
            raise TypeError("verify_manifest_sha256 must be bool")
        if not isinstance(validate_index_records, bool):
            raise TypeError("validate_index_records must be bool")
        ready: Mapping[str, Any] | None = None
        ready_file: Path | None = None
        expected_ready_sha: str | None = None
        ready_latent_layout: dict[str, Any] | None = None
        if ready_path is not None:
            ready_file = Path(ready_path).resolve()
            try:
                ready_payload = ready_file.read_bytes()
            except OSError as exc:
                raise FileNotFoundError(
                    f"Render sample READY file does not exist or cannot be read: {ready_file}"
                ) from exc
            expected_ready_sha = _sha256(
                expected_ready_sha256,
                field="expected_ready_sha256",
            )
            if hashlib.sha256(ready_payload).hexdigest() != expected_ready_sha:
                raise RuntimeError("Render sample READY SHA-256 mismatch")
            ready = _json_object_from_bytes(ready_payload, source=ready_file)
            if (
                not isinstance(ready, Mapping)
                or ready.get("schema_version") != RENDER_SAMPLE_READY_SCHEMA
                or ready.get("status") != RENDER_SAMPLE_READY_STATUS
            ):
                raise RuntimeError("Render sample READY schema or status is unsupported")
            declared_manifest = Path(str(ready.get("manifest") or ""))
            if not declared_manifest.is_absolute():
                declared_manifest = ready_file.parent / declared_manifest
            declared_manifest = declared_manifest.resolve()
            if declared_manifest != self.manifest:
                raise RuntimeError("Render sample READY references a different manifest")
            if expected_release_revision is not None and (
                not isinstance(expected_release_revision, str)
                or ready.get("revision") != expected_release_revision
            ):
                raise RuntimeError("Render sample release revision mismatch")
            if ready.get("split") != split:
                raise RuntimeError("Render sample READY split mismatch")
            if (
                required_quality_profile is not None
                and ready.get("required_quality_profile")
                != required_quality_profile
            ):
                raise RuntimeError("Render sample READY quality-profile mismatch")
            checks = ready.get("checks")
            if (
                not isinstance(checks, Mapping)
                or not RENDER_SAMPLE_REQUIRED_CHECKS.issubset(checks)
                or any(
                    checks.get(name) is not True
                    for name in RENDER_SAMPLE_REQUIRED_CHECKS
                )
            ):
                raise RuntimeError("Render sample READY checks did not all pass")
            ready_records = ready.get("records")
            if (
                not isinstance(ready_records, int)
                or isinstance(ready_records, bool)
                or ready_records <= 0
            ):
                raise RuntimeError("Render sample READY record count is invalid")
            if expected_records is None:
                expected_records = ready_records
            elif expected_records != ready_records:
                raise RuntimeError("Configured expected_records does not match READY")
            ready_identities = ready.get("identities")
            if not isinstance(ready_identities, Mapping):
                raise RuntimeError("Render sample READY is missing identities")
            ready_expected: dict[str, Any] = {
                "tokenizer_revision": self.expected_revisions.get(
                    "tokenizer_revision"
                ),
                "vae_revision": self.expected_revisions.get("vae_revision"),
                "latent_cache_revision": self.expected_revisions.get(
                    "latent_cache_revision"
                ),
                "latent_stats_sha256": self.expected_revisions.get(
                    "latent_stats_sha256"
                ),
                "text_encoder_revision": self.expected_revisions.get(
                    "text_encoder_revision"
                ),
                "text_tokenizer_revision": self.expected_revisions.get(
                    "text_tokenizer_revision"
                ),
                "text_cache_revision": self.expected_revisions.get(
                    "text_cache_revision"
                ),
                "rewriter_revision": self.expected_revisions.get(
                    "rewriter_revision"
                ),
            }
            if self.expected_ready_text_cache_revision is not None:
                ready_expected["text_cache_revision"] = (
                    self.expected_ready_text_cache_revision
                )
            ready_artifact_names = {
                "posterior_mode": "latent_posterior_mode",
                "posterior_base_seed": "latent_posterior_base_seed",
                "posterior_seed_derivation": "latent_posterior_seed_derivation",
                "posterior_epsilon_draw_layout": (
                    "latent_posterior_epsilon_draw_layout"
                ),
            }
            for name, expected in (expected_artifacts or {}).items():
                ready_expected[ready_artifact_names.get(name, name)] = expected
            mismatches = {
                name: {
                    "expected": expected,
                    "actual": ready_identities.get(name),
                }
                for name, expected in ready_expected.items()
                if ready_identities.get(name) != expected
            }
            if mismatches:
                raise RuntimeError(
                    f"Render sample READY revisions do not match: {mismatches}"
                )
            ready_layout = ready_identities.get("latent_layout")
            ready_layout_sha256 = ready_identities.get("latent_layout_sha256")
            if ready_layout is not None or ready_layout_sha256 is not None:
                ready_latent_layout = _validate_bound_latent_layout(
                    ready_layout,
                    ready_layout_sha256,
                    field="Render sample READY latent_layout",
                )
                expected_special_channels = self.expected_artifacts.get(
                    "latent_special_channels"
                )
                if (
                    expected_special_channels is not None
                    and ready_latent_layout["special_channels"]
                    != expected_special_channels
                ):
                    raise RuntimeError(
                        "Render sample READY latent side channel does not match the config"
                    )
        self.ready = dict(ready) if ready is not None else None
        self.latent_layout = ready_latent_layout
        self.ready_sha256 = expected_ready_sha
        self.ready_path = ready_file
        self.split_groups_disjoint_verified = bool(
            ready is not None
            and isinstance(ready.get("checks"), Mapping)
            and ready["checks"].get("split_groups_disjoint") is True
        )
        ready_inputs = ready.get("inputs") if ready is not None else None
        audio_input = (
            ready_inputs.get("audio_manifest")
            if isinstance(ready_inputs, Mapping)
            else None
        )
        self.source_audio_manifest_sha256 = (
            _sha256(
                audio_input.get("sha256"),
                field="READY.inputs.audio_manifest.sha256",
            )
            if isinstance(audio_input, Mapping)
            else None
        )
        self.base_dir = self.manifest.parent
        required_revisions = {
            "tokenizer_revision",
            "vae_revision",
            "text_encoder_revision",
            "text_tokenizer_revision",
            "text_cache_revision",
            "rewriter_revision",
            "latent_cache_revision",
            "latent_stats_sha256",
        }
        missing = required_revisions - set(self.expected_revisions)
        if missing:
            raise ValueError(f"expected_revisions is missing {sorted(missing)}")
        _sha256(
            self.expected_revisions["latent_stats_sha256"],
            field="expected latent_stats_sha256",
        )
        self.text_provenance = TextEncoderProvenance(
            model_id=str(text_model_id),
            model_revision=self.expected_revisions["text_encoder_revision"],
            tokenizer_revision=self.expected_revisions["text_tokenizer_revision"],
            cache_revision=self.expected_revisions["text_cache_revision"],
        )
        self.text_provenance.validate()
        for name, loader in (
            ("tags", self.tags_text_cache_loader),
            ("lyrics", self.lyrics_text_cache_loader),
        ):
            if loader is not None and loader.expected_provenance != self.text_provenance:
                raise RuntimeError(
                    f"Renderer data {name} text-cache provenance mismatch"
                )
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be 'train', 'valid', or 'test'")
        self.split = split
        self._record_offsets = array("Q")
        self._record_sizes = array("Q")
        self._record_lines = array("Q")
        self._frame_lengths = array("I")
        self._manifest_fd: int | None = None
        self._manifest_fd_pid: int | None = None
        self._validated_sample_ids: tuple[str, ...] | None = None
        self.index_path: Path | None = None
        self.index_sha256: str | None = None
        self.index_revision: str | None = None
        self.uses_published_index = index_path is not None
        if index_path is not None:
            assert ready is not None
            assert ready_file is not None
            assert expected_ready_sha is not None
            self._load_published_index(
                index_path=Path(index_path).resolve(),
                expected_index_sha256=_sha256(
                    expected_index_sha256,
                    field="expected_index_sha256",
                ),
                ready=ready,
                ready_path=ready_file,
                ready_sha256=expected_ready_sha,
                expected_manifest_sha256=expected_manifest_sha256,
                expected_records=expected_records,
                expected_release_revision=expected_release_revision,
                verify_manifest_sha256=verify_manifest_sha256,
            )
            initial_identity = self._manifest_identity
            final_identity = initial_identity
            digest_value = self.manifest_sha256
            if validate_index_records:
                self._validate_published_index_records()
        else:
            try:
                with self.manifest.open("rb") as source:
                    initial_identity = _stat_identity(os.fstat(source.fileno()))
                    digest = hashlib.sha256()


                    seen: set[str] | None = set() if ready is None else None
                    line_number = 0
                    while True:
                        offset = source.tell()
                        line = source.readline()
                        if not line:
                            break
                        line_number += 1
                        digest.update(line)
                        if not line.strip():
                            continue
                        raw = _jsonl_object(
                            line,
                            source=self.manifest,
                            line_number=line_number,
                        )
                        if raw.get("schema_version") != RENDER_SAMPLE_SCHEMA:
                            raise ValueError(
                                f"{self.manifest}:{line_number} schemais not"
                                f"{RENDER_SAMPLE_SCHEMA}"
                            )
                        if raw.get("split") != split:
                            raise ValueError(
                                f"{self.manifest}:{line_number} "
                                f"split={raw.get('split')!r},requires singlesplit={split!r}"
                            )
                        sample_id = raw.get("sample_id")
                        if not isinstance(sample_id, str) or not sample_id:
                            raise ValueError(f"sample_id is empty or duplicated: {sample_id!r}")
                        if seen is not None:
                            if sample_id in seen:
                                raise ValueError(
                                    f"sample_id is empty or duplicated: {sample_id!r}"
                                )
                            seen.add(sample_id)
                        frames = self._validate_metadata(raw, sample_id=sample_id)
                        self._record_offsets.append(offset)
                        self._record_sizes.append(len(line))
                        self._record_lines.append(line_number)
                        self._frame_lengths.append(frames)
                    final_identity = _stat_identity(os.fstat(source.fileno()))
                    digest_value = digest.hexdigest()
            except OSError as exc:
                raise FileNotFoundError(
                    f"Render sample manifest does not exist or cannot be read: {self.manifest}"
                ) from exc
            if final_identity != initial_identity:
                raise RuntimeError("Render sample manifest changed during indexing")
        try:
            path_identity = _stat_identity(self.manifest.stat())
        except OSError as exc:
            raise FileNotFoundError(
                f"Render sample manifest is not visible after indexing: {self.manifest}"
            ) from exc
        if path_identity != initial_identity:
            raise RuntimeError("Render sample manifest was replaced by another file")
        self._manifest_identity = initial_identity
        self.manifest_sha256 = digest_value
        if expected_manifest_sha256 is not None:
            expected_manifest_sha256 = _sha256(
                expected_manifest_sha256,
                field="expected_manifest_sha256",
            )
            if self.manifest_sha256 != expected_manifest_sha256:
                raise RuntimeError(
                    "Render sample manifest SHA-256 mismatch: "
                    f"expected={expected_manifest_sha256} "
                    f"actual={self.manifest_sha256}"
                )
        if ready is not None and self.manifest_sha256 != _sha256(
            ready.get("manifest_sha256"),
            field="READY.manifest_sha256",
        ):
            raise RuntimeError("Render sample READY manifest SHA-256 mismatch")
        if not self._record_offsets:
            raise ValueError(f"{self.manifest} with split={split} has no samples")
        if expected_records is not None and len(self) != expected_records:
            raise RuntimeError(
                f"{self.manifest} with split={split} record count mismatch: "
                f"expected={expected_records} actual={len(self)}"
            )

    def _validate_published_index_records(self) -> None:
        seen: set[str] = set()
        sample_ids: list[str] = []
        try:
            with self.manifest.open("rb") as source:
                initial_identity = _stat_identity(os.fstat(source.fileno()))
                for index in range(len(self)):
                    offset = source.tell()
                    line = source.readline()
                    expected_offset = int(self._record_offsets[index])
                    expected_size = int(self._record_sizes[index])
                    expected_line = int(self._record_lines[index])
                    if (
                        offset != expected_offset
                        or len(line) != expected_size
                        or expected_line != index + 1
                    ):
                        raise RuntimeError(
                            "Render sample index and manifest byte boundaries differ"
                        )
                    raw = _jsonl_object(
                        line,
                        source=self.manifest,
                        line_number=expected_line,
                    )
                    if raw.get("schema_version") != RENDER_SAMPLE_SCHEMA:
                        raise RuntimeError("Render sample index record schema mismatch")
                    if raw.get("split") != self.split:
                        raise RuntimeError("Render sample index record split mismatch")
                    sample_id = raw.get("sample_id")
                    if (
                        not isinstance(sample_id, str)
                        or not sample_id
                        or sample_id in seen
                    ):
                        raise RuntimeError(
                            f"Render sample index record sample_id is invalid: {sample_id!r}"
                        )
                    seen.add(sample_id)
                    sample_ids.append(sample_id)
                    frames = self._validate_metadata(raw, sample_id=sample_id)
                    if frames != int(self._frame_lengths[index]):
                        raise RuntimeError(
                            f"{sample_id} sample-index frame length mismatch"
                        )
                if source.read(1):
                    raise RuntimeError("Render sample index does not cover the manifest tail")
                final_identity = _stat_identity(os.fstat(source.fileno()))
        except OSError as exc:
            raise FileNotFoundError(
                "Render sample manifest cannot be verified against the published index: "
                f"{self.manifest}"
            ) from exc
        if initial_identity != self._manifest_identity or final_identity != initial_identity:
            raise RuntimeError("Render sample manifest changed while verifying the published index")
        expected_ids_sha = _sha256(
            self._index_metadata.get("sample_ids_sha256"),
            field="sample index sample_ids_sha256",
        )
        if _canonical_sha(sorted(sample_ids)) != expected_ids_sha:
            raise RuntimeError("Render sample index sample_id collection SHA-256 mismatch")
        self._validated_sample_ids = tuple(sample_ids)

    def _load_published_index(
        self,
        *,
        index_path: Path,
        expected_index_sha256: str,
        ready: Mapping[str, Any],
        ready_path: Path,
        ready_sha256: str,
        expected_manifest_sha256: str | None,
        expected_records: int | None,
        expected_release_revision: str | None,
        verify_manifest_sha256: bool,
    ) -> None:
        try:
            index_payload = index_path.read_bytes()
        except OSError as exc:
            raise FileNotFoundError(
                f"Render sample index does not exist or cannot be read: {index_path}"
            ) from exc
        if hashlib.sha256(index_payload).hexdigest() != expected_index_sha256:
            raise RuntimeError("Render sample index SHA-256 mismatch")
        metadata = _json_object_from_bytes(index_payload, source=index_path)
        if (
            metadata.get("schema_version") != RENDER_SAMPLE_INDEX_SCHEMA
            or metadata.get("status") != RENDER_SAMPLE_INDEX_STATUS
            or metadata.get("payload_schema")
            != RENDER_SAMPLE_INDEX_PAYLOAD_SCHEMA
        ):
            raise RuntimeError("Render sample index schema or status is unsupported")
        declared_revision = _sha256(
            metadata.get("revision"),
            field="sample index revision",
        )
        unsigned_metadata = {
            name: value for name, value in metadata.items() if name != "revision"
        }
        if _canonical_sha(unsigned_metadata) != declared_revision:
            raise RuntimeError("Render sample index revision verification failed")
        ready_index = _mapping(
            ready.get("index"),
            field="READY.index",
        )
        if (
            ready_index.get("schema_version") != RENDER_SAMPLE_INDEX_SCHEMA
            or ready_index.get("required") is not True
        ):
            raise RuntimeError("Render sample READY does not declare the required published index")
        declared_index_path = Path(str(ready_index.get("metadata") or ""))
        if not declared_index_path.is_absolute():
            declared_index_path = ready_path.parent / declared_index_path
        if declared_index_path.resolve() != index_path:
            raise RuntimeError("Render sample READY references different index metadata")
        manifest_identity = _mapping(
            metadata.get("manifest"),
            field="sample index manifest",
        )
        declared_manifest_path = Path(str(manifest_identity.get("path") or ""))
        if not declared_manifest_path.is_absolute():
            declared_manifest_path = index_path.parent / declared_manifest_path
        if declared_manifest_path.resolve() != self.manifest:
            raise RuntimeError("Render sample index references a different manifest")
        manifest_sha = _sha256(
            manifest_identity.get("sha256"),
            field="sample index manifest.sha256",
        )
        ready_manifest_sha = _sha256(
            ready.get("manifest_sha256"),
            field="READY.manifest_sha256",
        )
        configured_manifest_sha = (
            _sha256(
                expected_manifest_sha256,
                field="expected_manifest_sha256",
            )
            if expected_manifest_sha256 is not None
            else ready_manifest_sha
        )
        if manifest_sha != ready_manifest_sha or manifest_sha != configured_manifest_sha:
            raise RuntimeError("Render sample index manifest SHA-256 binding mismatch")
        try:
            manifest_stat = self.manifest.stat()
        except OSError as exc:
            raise FileNotFoundError(
                f"Render sample manifest does not exist or cannot be read: {self.manifest}"
            ) from exc
        manifest_size = _integer(
            manifest_identity.get("size_bytes"),
            field="sample index manifest.size_bytes",
            minimum=1,
        )
        if int(manifest_stat.st_size) != manifest_size:
            raise RuntimeError("Render sample index manifest size mismatch")
        if verify_manifest_sha256 and _file_sha256(self.manifest) != manifest_sha:
            raise RuntimeError("Render sample index manifest content SHA-256 mismatch")
        ready_identity = _mapping(
            metadata.get("ready"),
            field="sample index ready",
        )
        declared_ready_path = Path(str(ready_identity.get("path") or ""))
        if not declared_ready_path.is_absolute():
            declared_ready_path = index_path.parent / declared_ready_path
        if declared_ready_path.resolve() != ready_path:
            raise RuntimeError("Render sample index references a different READY file")
        if (
            _sha256(
                ready_identity.get("sha256"),
                field="sample index ready.sha256",
            )
            != ready_sha256
        ):
            raise RuntimeError("Render sample index READY SHA-256 binding mismatch")
        records = _integer(
            metadata.get("records"),
            field="sample index records",
            minimum=1,
        )
        ready_records = _integer(
            ready.get("records"),
            field="READY.records",
            minimum=1,
        )
        if records != ready_records or (
            expected_records is not None and records != expected_records
        ):
            raise RuntimeError("Render sample index record-count binding mismatch")
        if metadata.get("split") != self.split:
            raise RuntimeError("Render sample index split mismatch")
        if metadata.get("required_quality_profile") != self.required_quality_profile:
            raise RuntimeError("Render sample index quality-profile mismatch")
        release_revision = str(ready.get("revision") or "")
        if (
            metadata.get("release_revision") != release_revision
            or (
                expected_release_revision is not None
                and release_revision != expected_release_revision
            )
        ):
            raise RuntimeError("Render sample index release revision mismatch")
        ready_identities = _mapping(
            ready.get("identities"),
            field="READY.identities",
        )
        if metadata.get("identities") != ready_identities:
            raise RuntimeError("Render sample index identities do not match READY")
        _sha256(
            metadata.get("sample_ids_sha256"),
            field="sample index sample_ids_sha256",
        )
        producer = _mapping(metadata.get("producer"), field="sample index producer")
        _sha256(
            producer.get("code_sha256"),
            field="sample index producer.code_sha256",
        )
        payload_identity = _mapping(
            metadata.get("payload"),
            field="sample index payload",
        )
        raw_payload_path = payload_identity.get("path")
        if not isinstance(raw_payload_path, (str, os.PathLike)):
            raise RuntimeError("Render sample index payload.path is invalid")
        payload_path = Path(os.fspath(raw_payload_path))
        if not payload_path.is_absolute():
            payload_root = index_path.parent.resolve()
            payload_path = (payload_root / payload_path).resolve()
            try:
                payload_path.relative_to(payload_root)
            except ValueError as exc:
                raise RuntimeError("Render sample index payload path escapes its root") from exc
        else:
            payload_path = payload_path.resolve()
        try:
            payload_stat = payload_path.stat()
        except OSError as exc:
            raise FileNotFoundError(
                f"Render sample index payload does not exist: {payload_path}"
            ) from exc
        payload_size = _integer(
            payload_identity.get("size_bytes"),
            field="sample index payload.size_bytes",
            minimum=1,
        )
        if int(payload_stat.st_size) != payload_size:
            raise RuntimeError("Render sample index payload size mismatch")
        payload_sha = _sha256(
            payload_identity.get("sha256"),
            field="sample index payload.sha256",
        )
        if _file_sha256(payload_path) != payload_sha:
            raise RuntimeError("Render sample index payload SHA-256 mismatch")
        ready_payload_path = Path(str(ready_index.get("payload") or ""))
        if not ready_payload_path.is_absolute():
            ready_payload_path = ready_path.parent / ready_payload_path
        if (
            ready_payload_path.resolve() != payload_path
            or ready_index.get("payload_sha256") != payload_sha
            or ready_index.get("payload_size_bytes") != payload_size
            or ready_index.get("records") != records
        ):
            raise RuntimeError("Render sample READY and index payload bindings differ")
        expected_arrays = {
            "offset": "uint64",
            "size": "uint64",
            "line_number": "uint64",
            "frame_length": "uint32",
        }
        if payload_identity.get("arrays") != {
            name: dtype for name, dtype in expected_arrays.items()
        }:
            raise RuntimeError("Render sample index payload array contract mismatch")
        try:
            payload = np.load(
                payload_path,
                allow_pickle=False,
                mmap_mode="r",
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError("Render sample index payload cannot be parsed") from exc
        if (
            not isinstance(payload, np.ndarray)
            or payload.ndim != 1
            or payload.dtype != RENDER_SAMPLE_INDEX_PAYLOAD_DTYPE
            or len(payload) != records
        ):
            raise RuntimeError("Render sample index payload shape or dtype mismatch")
        offsets = payload["offset"]
        sizes = payload["size"]
        line_numbers = payload["line_number"]
        frame_lengths = payload["frame_length"]
        if (
            int(offsets[0]) != 0
            or np.any(sizes == 0)
            or np.any(frame_lengths == 0)
            or np.any(frame_lengths > MAX_RENDER_FRAMES)
            or not np.array_equal(
                line_numbers,
                np.arange(1, records + 1, dtype=np.uint64),
            )
            or (
                records > 1
                and not np.array_equal(
                    offsets[1:],
                    offsets[:-1] + sizes[:-1],
                )
            )
            or int(offsets[-1] + sizes[-1]) != manifest_size
        ):
            raise RuntimeError("Render sample index offset/frame ranges overlap")
        self._record_offsets = array("Q", offsets.tolist())
        self._record_sizes = array("Q", sizes.tolist())
        self._record_lines = array("Q", line_numbers.tolist())
        self._frame_lengths = array("I", frame_lengths.tolist())
        self._manifest_identity = _stat_identity(manifest_stat)
        self.manifest_sha256 = manifest_sha
        self.index_path = index_path
        self.index_sha256 = expected_index_sha256
        self.index_revision = declared_revision
        self._index_metadata = metadata

    def _validate_metadata(self, raw: Mapping[str, Any], *, sample_id: str) -> int:
        audio = _mapping(raw.get("audio"), field=f"{sample_id}.audio")
        semantic = _mapping(raw.get("semantic"), field=f"{sample_id}.semantic")
        latent = _mapping(raw.get("latent"), field=f"{sample_id}.latent")
        condition = _mapping(raw.get("condition"), field=f"{sample_id}.condition")
        text_cache = _mapping(raw.get("text_cache"), field=f"{sample_id}.text_cache")
        is_renderer_data_text_reference = (
            text_cache.get("schema_version") == RENDERER_DATA_TEXT_REFERENCE_SCHEMA
        )
        is_renderer_data_deferred_parent_text = (
            text_cache.get("schema_version")
            == RENDERER_DATA_DEFERRED_PARENT_TEXT_SCHEMA
        )
        text_record = (
            None
            if is_renderer_data_text_reference or is_renderer_data_deferred_parent_text
            else _mapping(
                text_cache.get("record"), field=f"{sample_id}.text_cache.record"
            )
        )
        quality = _mapping(raw.get("quality"), field=f"{sample_id}.quality")
        groups = _mapping(raw.get("groups"), field=f"{sample_id}.groups")
        if groups.get("split") != raw.get("split"):
            raise RuntimeError(f"{sample_id} groups.split does not match the record split")
        for name in RENDER_SAMPLE_REQUIRED_GROUP_FIELDS:
            value = groups.get(name)
            if not isinstance(value, str) or not value:
                raise RuntimeError(f"{sample_id} groups.{name} is missing")
        if self.required_quality_profile is not None and quality.get(
            self.required_quality_profile
        ) is not True:
            raise RuntimeError(
                f"{sample_id} failed quality profile={self.required_quality_profile}"
            )
        audio_path = _artifact_path(
            audio, base_dir=self.base_dir, field=f"{sample_id}.audio"
        )
        if not audio_path.is_file():
            raise FileNotFoundError(f"{sample_id} audio artifact does not exist: {audio_path}")
        audio_sha = _sha256(audio.get("sha256"), field=f"{sample_id}.audio.sha256")
        source_sha = _sha256(
            audio.get("source_sha256"),
            field=f"{sample_id}.audio.source_sha256",
        )
        sample_rate = _integer(
            audio.get("sample_rate", audio.get("sample_rate_hz")),
            field=f"{sample_id}.audio.sample_rate",
            minimum=1,
        )
        channels = _integer(
            audio.get("channels"),
            field=f"{sample_id}.audio.channels",
            minimum=1,
        )
        if sample_rate != SAMPLE_RATE or channels != AUDIO_CHANNELS:
            raise ValueError(
                f"{sample_id} audio must be {SAMPLE_RATE}Hz/{AUDIO_CHANNELS}ch"
            )
        audio_start_samples = _seconds_to_samples(
            audio.get("start_sec"),
            field=f"{sample_id}.audio.start_sec",
            positive=False,
        )
        audio_duration_samples = _seconds_to_samples(
            audio.get("duration_sec"),
            field=f"{sample_id}.audio.duration_sec",
            positive=True,
        )
        if (
            audio_start_samples % (SAMPLE_RATE // LATENT_FRAME_HZ)
            or audio_duration_samples % (SAMPLE_RATE // LATENT_FRAME_HZ)
        ):
            raise ValueError(f"{sample_id} audio time range must fall on the 25 Hz grid")
        semantic_shape = semantic.get("shape")
        latent_shape = latent.get("shape")
        shapes_are_integer = (
            isinstance(semantic_shape, list)
            and len(semantic_shape) == 1
            and isinstance(latent_shape, list)
            and len(latent_shape) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (*semantic_shape, *latent_shape)
            )
        )
        if (
            not shapes_are_integer
            or latent_shape[1] != LATENT_DIM
            or semantic_shape[0] != latent_shape[0]
            or latent_shape[0] <= 0
        ):
            raise ValueError(f"{sample_id} semantic and latent shapes do not match")
        frames = latent_shape[0]
        if frames > MAX_RENDER_FRAMES:
            raise ValueError(
                f"{sample_id} length {frames} exceeds the six-minute limit of "
                f"{MAX_RENDER_FRAMES} frames"
            )
        if samples_to_latent_frames(audio_duration_samples) != frames:
            raise ValueError(
                f"{sample_id} audio duration does not match semantic/latent frame count"
            )
        if (
            _number(
                semantic.get("frame_hz"),
                field=f"{sample_id}.semantic.frame_hz",
            )
            != LATENT_FRAME_HZ
            or _number(
                latent.get("frame_hz"),
                field=f"{sample_id}.latent.frame_hz",
            )
            != LATENT_FRAME_HZ
        ):
            raise ValueError(f"{sample_id} semantic and latent data must both be 25 Hz")
        if _integer(
            semantic.get("codebook_size"),
            field=f"{sample_id}.semantic.codebook_size",
            minimum=1,
        ) != SEMANTIC_CODEBOOK_SIZE:
            raise ValueError(f"{sample_id} semantic codebook size must be 32768")
        if semantic.get("dtype") != "uint16":
            raise ValueError(f"{sample_id} semantic dtype must be uint16")
        if latent.get("dtype") not in {"float16", "float32"}:
            raise ValueError(f"{sample_id} latent dtype must be 'float16' or 'float32'")
        for owner_name, owner in (("semantic", semantic), ("latent", latent)):
            _sha256(owner.get("sha256"), field=f"{sample_id}.{owner_name}.sha256")
            _artifact_path(
                owner,
                base_dir=self.base_dir,
                field=f"{sample_id}.{owner_name}",
            )
        semantic_source = _sha256(
            semantic.get("source_audio_sha256"),
            field=f"{sample_id}.semantic.source_audio_sha256",
        )
        latent_source = _sha256(
            latent.get("source_audio_sha256"),
            field=f"{sample_id}.latent.source_audio_sha256",
        )
        if semantic_source != latent_source:
            raise ValueError(f"{sample_id} semantic and latent source-audio SHA-256 values differ")
        if semantic_source != source_sha:
            raise ValueError(f"{sample_id} audio, semantic, and latent source-audio SHA-256 values differ")
        semantic_input = _sha256(
            semantic.get("input_audio_sha256"),
            field=f"{sample_id}.semantic.input_audio_sha256",
        )
        latent_input = _sha256(
            latent.get("derived_audio_sha256", latent.get("input_audio_sha256")),
            field=f"{sample_id}.latent.derived_audio_sha256",
        )
        semantic_input_basis = str(
            semantic.get("input_audio_basis") or "canonical_audio"
        )
        if semantic_input_basis == "canonical_audio":
            semantic_input_matches = semantic_input == audio_sha
        else:
            raise ValueError(
                f"{sample_id} semantic input_audio_basis must be canonical_audio"
            )
        if not semantic_input_matches or latent_input != audio_sha:
            raise ValueError(f"{sample_id} semantic and latent audio identities differ")
        for owner_name, owner in (("semantic", semantic), ("latent", latent)):
            if str(owner.get("sample_id") or "") != sample_id:
                raise ValueError(f"{sample_id} {owner_name}.sample_id mismatch")
            start_samples = _seconds_to_samples(
                owner.get("source_start_sec"),
                field=f"{sample_id}.{owner_name}.source_start_sec",
                positive=False,
            )
            duration_samples = _seconds_to_samples(
                owner.get("source_duration_sec"),
                field=f"{sample_id}.{owner_name}.source_duration_sec",
                positive=True,
            )
            if (
                start_samples != audio_start_samples
                or duration_samples != audio_duration_samples
            ):
                raise ValueError(f"{sample_id} {owner_name} and audio time ranges differ")
        expected = self.expected_revisions
        observed = {
            "tokenizer_revision": semantic.get("tokenizer_revision"),
            "vae_revision": latent.get("vae_revision"),
            "latent_cache_revision": latent.get("cache_revision"),
            "latent_stats_sha256": latent.get("latent_stats_sha256"),
        }
        mismatches = {
            name: {"expected": expected[name], "actual": value}
            for name, value in observed.items()
            if str(value or "") != expected[name]
        }
        if mismatches:
            raise RuntimeError(f"{sample_id} artifact revision mismatch: {mismatches}")
        posterior_mode = latent.get("posterior_mode")
        posterior_base_seed = latent.get("posterior_base_seed")
        posterior_seed = latent.get("posterior_sample_seed")
        posterior_seed_derivation = latent.get("posterior_seed_derivation")
        posterior_epsilon_draw_layout = latent.get(
            "posterior_epsilon_draw_layout", "contiguous_bdt"
        )
        if posterior_epsilon_draw_layout != "contiguous_bdt":
            raise ValueError(
                f"{sample_id} latent posterior epsilon layout is invalid"
            )
        if posterior_mode not in {"mean", "sample"}:
            raise ValueError(f"{sample_id} latent posterior_mode is invalid")
        v2_sample_seed_contract = (
            "posterior_base_seed" in latent
            or "posterior_seed_derivation" in latent
        )
        if posterior_mode == "sample":
            if v2_sample_seed_contract:
                if (
                    not isinstance(posterior_base_seed, int)
                    or isinstance(posterior_base_seed, bool)
                    or posterior_seed_derivation != POSTERIOR_SEED_DERIVATION
                ):
                    raise ValueError(
                        f"{sample_id} sample posterior base seed or derivation is invalid"
                    )
                expected_seed = _expected_posterior_sample_seed(
                    base_seed=posterior_base_seed,
                    sample_id=sample_id,
                    derived_audio_sha256=latent_input,
                )
                if posterior_seed != expected_seed:
                    raise ValueError(
                        f"{sample_id} sample posterior seed does not match its base seed"
                    )
            elif not isinstance(posterior_seed, int) or isinstance(
                posterior_seed, bool
            ):
                raise ValueError(f"{sample_id} sample posterior is missing a fixed seed")
        elif any(
            value is not None
            for value in (
                posterior_base_seed,
                posterior_seed,
                posterior_seed_derivation,
            )
        ):
            raise ValueError(f"{sample_id} mean posterior must not declare a seed")
        latent_layout = latent.get("latent_layout")
        latent_layout_sha256 = latent.get("latent_layout_sha256")
        normalized_layout: dict[str, Any] | None = None
        if latent_layout is not None or latent_layout_sha256 is not None:
            normalized_layout = _validate_bound_latent_layout(
                latent_layout,
                latent_layout_sha256,
                field=f"{sample_id}.latent.latent_layout",
            )
            if (
                self.latent_layout is not None
                and normalized_layout != self.latent_layout
            ):
                raise RuntimeError(
                    f"{sample_id} latent layout does not match READY metadata"
                )
        elif self.latent_layout is not None:
            raise RuntimeError(f"{sample_id} is missing the latent layout declared by READY")
        if self.expected_artifacts:
            artifact_observed = {
                "tokenizer_checkpoint_sha256": semantic.get(
                    "tokenizer_checkpoint_sha256"
                ),
                "semantic_extractor_revision": semantic.get(
                    "semantic_extractor_revision"
                ),
                "tokenizer_artifact_sha256": semantic.get(
                    "tokenizer_artifact_sha256"
                ),
                "tokenizer_artifact_sidecar_sha256": semantic.get(
                    "tokenizer_artifact_sidecar_sha256"
                ),
                "tokenizer_binding_sha256": semantic.get(
                    "tokenizer_binding_sha256"
                ),
                "semantic_materializer_revision": semantic.get(
                    "materializer_revision"
                ),
                "tokenizer_config_sha256": semantic.get("tokenizer_config_sha256"),
                "semantic_encoder_code_sha256": semantic.get("encoder_code_sha256"),
                "vae_checkpoint_sha256": latent.get("vae_checkpoint_sha256"),
                "stft_revision": latent.get("stft_revision"),
                "stft_config_sha256": latent.get("stft_config_sha256"),
                "posterior_mode": latent.get("posterior_mode"),
                "posterior_base_seed": posterior_base_seed,
                "posterior_seed_derivation": posterior_seed_derivation,
                "posterior_epsilon_draw_layout": (
                    posterior_epsilon_draw_layout
                ),
                "latent_layout_sha256": latent_layout_sha256,
                "latent_special_channels": (
                    normalized_layout["special_channels"]
                    if normalized_layout is not None
                    else None
                ),
                "text_cache_config_sha256": text_cache.get(
                    "cache_config_sha256"
                ),
                "text_cache_config_file_sha256": text_cache.get(
                    "cache_config_file_sha256"
                ),
            }


            dataset_only_artifacts = {
                "description_contract_sha256",
                "duration_contract",
            }
            artifact_mismatches = {
                name: {
                    "expected": expected,
                    "actual": artifact_observed.get(name),
                }
                for name, expected in self.expected_artifacts.items()
                if name not in dataset_only_artifacts
                and artifact_observed.get(name) != expected
            }
            if artifact_mismatches:
                raise RuntimeError(
                    f"{sample_id} latent artifact identity mismatch: {artifact_mismatches}"
                )
        _sha256(
            latent.get("latent_stats_sha256"),
            field=f"{sample_id}.latent.latent_stats_sha256",
        )
        if not isinstance(condition.get("description"), str) or not isinstance(
            condition.get("lyrics"), str
        ):
            raise TypeError(f"{sample_id} condition is missing description or lyrics")
        if (
            str(condition.get("rewriter_revision") or "")
            != self.expected_revisions["rewriter_revision"]
        ):
            raise RuntimeError(f"{sample_id} rewriter revision mismatch")
        if condition.get("schema_version") != REWRITER_SCHEMA_VERSION:
            raise RuntimeError(f"{sample_id} rewriter schema mismatch")
        if (
            str(condition.get("text_tokenizer_revision") or "")
            != self.expected_revisions["text_tokenizer_revision"]
        ):
            raise RuntimeError(f"{sample_id} condition text-tokenizer revision mismatch")
        if is_renderer_data_text_reference or is_renderer_data_deferred_parent_text:
            parsed_condition = RenderTextCondition.from_mapping(raw)
            tags_sha = _sha256(
                text_cache.get("tags_content_sha256"),
                field=f"{sample_id}.text_cache.tags_content_sha256",
            )
            lyrics_sha = _sha256(
                text_cache.get("lyrics_content_sha256"),
                field=f"{sample_id}.text_cache.lyrics_content_sha256",
            )
            if hashlib.sha256(parsed_condition.description.encode("utf-8")).hexdigest() != tags_sha:
                raise RuntimeError(f"{sample_id} renderer-data tags content SHA-256 mismatch")
            if hashlib.sha256(parsed_condition.lyrics.encode("utf-8")).hexdigest() != lyrics_sha:
                raise RuntimeError(f"{sample_id} renderer-data lyrics content SHA-256 mismatch")
            if text_cache.get("source_condition_sha256") != (
                parsed_condition.source_condition_sha256
            ):
                raise RuntimeError(f"{sample_id} renderer-data complete-condition SHA-256 mismatch")
            if is_renderer_data_deferred_parent_text:
                if not self.allow_renderer_data_deferred_parent_text:
                    raise RuntimeError(
                        f"{sample_id} deferred full-track text requires an explicitly "
                        "enabled RendererData view"
                    )
                if (
                    text_cache.get("use_policy") != "renderer_data_short_window_only"
                    or text_cache.get("full_parent_text_ready") is not False
                ):
                    raise RuntimeError(
                        f"{sample_id} has an incompatible deferred full-track text policy"
                    )
                text_records = ()
            else:
                if (
                    self.tags_text_cache_loader is None
                    or self.lyrics_text_cache_loader is None
                ):
                    raise RuntimeError(
                        f"{sample_id} RendererData content-addressed text reference "
                        "requires both loaders"
                    )
                text_records = (
                    _content_addressed_record(
                        self.tags_text_cache_loader,
                        role="tags",
                        content_sha256=tags_sha,
                    )[1],
                    _content_addressed_record(
                        self.lyrics_text_cache_loader,
                        role="lyrics",
                        content_sha256=lyrics_sha,
                    )[1],
                )
        else:
            assert text_record is not None
            if str(text_record.get("sample_id") or "") != sample_id:
                raise RuntimeError(f"{sample_id} text-cache sample_id mismatch")
            text_records = (text_record,)
        for current_record in text_records:
            observed_text_provenance = TextEncoderProvenance.from_mapping(
                {
                    name: current_record.get(name)
                    for name in TextEncoderProvenance.__dataclass_fields__
                }
            )
            if observed_text_provenance != self.text_provenance:
                raise RuntimeError(
                    f"{sample_id} text cache provenance does not match"
                )
        text_cache_identity = {
            "cache_config_sha256": _sha256(
                text_cache.get("cache_config_sha256"),
                field=f"{sample_id}.text_cache.cache_config_sha256",
            ),
            "cache_config_file_sha256": _sha256(
                text_cache.get("cache_config_file_sha256"),
                field=f"{sample_id}.text_cache.cache_config_file_sha256",
            ),
        }
        if not self._dataset_text_cache_identity:
            self._dataset_text_cache_identity = text_cache_identity
        elif text_cache_identity != self._dataset_text_cache_identity:
            raise RuntimeError(f"{sample_id} text-cache config differs from the first dataset record")
        return frames

    def __len__(self) -> int:
        return len(self._record_offsets)

    @property
    def records(self) -> Sequence[dict[str, Any]]:
        return _IndexedCanonicalRecords(self)

    @property
    def frame_lengths(self) -> list[int]:
        return list(self._frame_lengths)

    def frame_length(self, index: int) -> int:
        return int(self._frame_lengths[index])

    def _open_manifest_fd(self) -> int:
        current_pid = os.getpid()
        if (
            self._manifest_fd is not None
            and self._manifest_fd_pid != current_pid
        ):

            os.close(self._manifest_fd)
            self._manifest_fd = None
        if self._manifest_fd is None:
            descriptor = os.open(self.manifest, os.O_RDONLY)
            descriptor_identity = _stat_identity(os.fstat(descriptor))
            try:
                path_identity = _stat_identity(self.manifest.stat())
            except OSError:
                path_identity = ()
            if (
                descriptor_identity != self._manifest_identity
                or path_identity != self._manifest_identity
            ):
                os.close(descriptor)
                raise RuntimeError("Render sample manifest identity changed after initialization")
            self._manifest_fd = descriptor
            self._manifest_fd_pid = current_pid
        return self._manifest_fd

    def _read_record(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        descriptor = self._open_manifest_fd()
        try:
            path_identity = _stat_identity(self.manifest.stat())
        except OSError:
            path_identity = ()
        if (
            _stat_identity(os.fstat(descriptor)) != self._manifest_identity
            or path_identity != self._manifest_identity
        ):
            raise RuntimeError("Render sample manifest content changed after initialization")
        offset = int(self._record_offsets[index])
        size = int(self._record_sizes[index])
        payload = os.pread(descriptor, size, offset)
        if len(payload) != size:
            raise RuntimeError("Render sample manifest index returned a short read")
        return _jsonl_object(
            payload,
            source=self.manifest,
            line_number=int(self._record_lines[index]),
        )

    def iter_records(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self._read_record(index)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        if self._validated_sample_ids is not None:
            return self._validated_sample_ids
        return tuple(str(record["sample_id"]) for record in self.iter_records())

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_manifest_fd"] = None
        state["_manifest_fd_pid"] = None
        return state

    def __del__(self) -> None:
        descriptor = getattr(self, "_manifest_fd", None)
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
            self._manifest_fd = None

    def _load_array(
        self,
        artifact: Mapping[str, Any],
        *,
        sample_id: str,
        name: str,
    ) -> np.ndarray:
        path = _artifact_path(
            artifact,
            base_dir=self.base_dir,
            field=f"{sample_id}.{name}",
        )
        expected_sha = _sha256(
            artifact.get("sha256"), field=f"{sample_id}.{name}.sha256"
        )
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise FileNotFoundError(
                f"{sample_id} {name} artifact does not exist or cannot be read: {path}"
            ) from exc
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise RuntimeError(f"{sample_id} {name} artifact SHA-256 mismatch")
        try:
            values = np.load(io.BytesIO(payload), allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"{sample_id} {name} artifact cannot be parsed") from exc
        if (
            list(values.shape) != artifact["shape"]
            or str(values.dtype) != artifact["dtype"]
        ):
            raise RuntimeError(f"{sample_id} {name} artifact shape or dtype changed")
        if not np.isfinite(values).all():
            raise RuntimeError(f"{sample_id} {name} artifact contains NaN or Inf")
        return np.array(values, copy=True)

    def _load_acoustic_item(
        self, index: int
    ) -> tuple[dict[str, Any], RenderTextCondition, dict[str, Any]]:
        raw = self._read_record(index)
        sample_id = str(raw["sample_id"])
        semantic_meta = _mapping(raw["semantic"], field=f"{sample_id}.semantic")
        latent_meta = _mapping(raw["latent"], field=f"{sample_id}.latent")
        semantic = self._load_array(semantic_meta, sample_id=sample_id, name="semantic")
        latent = self._load_array(latent_meta, sample_id=sample_id, name="latent")
        if semantic.size and int(semantic.max()) >= SEMANTIC_CODEBOOK_SIZE:
            raise ValueError(f"{sample_id} semantic token is out of bounds")
        condition = RenderTextCondition.from_mapping(raw)
        frames = int(self._frame_lengths[index])
        item = {
            "sample_id": sample_id,
            "latents": torch.from_numpy(latent),
            "latent_mask": torch.ones(frames, dtype=torch.bool),
            "semantic_ids": torch.from_numpy(
                semantic.astype(np.int64, copy=False)
            ),
            "semantic_mask": torch.ones(frames, dtype=torch.bool),
            "duration_seconds": frames / LATENT_FRAME_HZ,
            "revisions": dict(self.expected_revisions),
            "provenance": {
                "manifest": str(self.manifest),
                "latent_sha256": latent_meta["sha256"],
                "semantic_sha256": semantic_meta["sha256"],
                "source_audio_sha256": latent_meta["source_audio_sha256"],
                "source_condition_sha256": (
                    condition.source_condition_sha256
                ),
            },
        }
        return raw, condition, item

    def load_renderer_data_parent_assets(self, index: int) -> dict[str, Any]:

        _, _, item = self._load_acoustic_item(index)
        return item

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw, condition, item = self._load_acoustic_item(index)
        sample_id = str(raw["sample_id"])
        text_cache = _mapping(raw["text_cache"], field=f"{sample_id}.text_cache")
        deferred_parent_text = (
            text_cache.get("schema_version")
            == RENDERER_DATA_DEFERRED_PARENT_TEXT_SCHEMA
        )
        if deferred_parent_text and not self.resolve_renderer_data_deferred_parent_text:
            raise RuntimeError(
                f"{sample_id} deferred parent text cannot be read directly in the full view"
            )
        if deferred_parent_text:
            assert self.tags_text_cache_loader is not None
            assert self.lyrics_text_cache_loader is not None
            text_cache_config_sha256 = (
                self.tags_text_cache_loader.expected_cache_config_sha256
            )
            text_cache_config_file_sha256 = (
                self.tags_text_cache_loader.expected_cache_config_file_sha256
            )
            if (
                text_cache_config_sha256
                != self.lyrics_text_cache_loader.expected_cache_config_sha256
                or text_cache_config_file_sha256
                != self.lyrics_text_cache_loader.expected_cache_config_file_sha256
                or text_cache_config_file_sha256 is None
            ):
                raise RuntimeError(
                    f"{sample_id} RendererData full tags/lyrics cacheInconsistent identity"
                )
        else:
            text_cache_config_sha256 = _sha256(
                text_cache.get("cache_config_sha256"),
                field=f"{sample_id}.text_cache.cache_config_sha256",
            )
            text_cache_config_file_sha256 = str(
                text_cache.get("cache_config_file_sha256") or ""
            )
        record_base_value = text_cache.get("record_base_dir")
        if record_base_value is None:
            text_record_base = self.base_dir
        else:
            if not isinstance(record_base_value, (str, os.PathLike)):
                raise TypeError(f"{sample_id} text_cache.record_base_dir must be a path")
            text_record_base = Path(record_base_value)
            if not text_record_base.is_absolute():
                text_record_base = self.base_dir / text_record_base
            text_record_base = text_record_base.resolve()
            if not text_record_base.is_dir():
                raise FileNotFoundError(
                    f"{sample_id} text_cache.record_base_dirdoes not exist:"
                    f"{text_record_base}"
                )
        if deferred_parent_text or (
            text_cache.get("schema_version") == RENDERER_DATA_TEXT_REFERENCE_SCHEMA
        ):
            if (
                self.tags_text_cache_loader is None
                or self.lyrics_text_cache_loader is None
            ):
                raise RuntimeError(
                    f"{sample_id} RendererData content-addressed text reference "
                    "requires both loaders"
                )
            tags_sha = str(text_cache.get("tags_content_sha256") or "")
            lyrics_sha = str(text_cache.get("lyrics_content_sha256") or "")
            tags_sample_id, tags_record = _content_addressed_record(
                self.tags_text_cache_loader,
                role="tags",
                content_sha256=tags_sha,
            )
            lyrics_sample_id, lyrics_record = _content_addressed_record(
                self.lyrics_text_cache_loader,
                role="lyrics",
                content_sha256=lyrics_sha,
            )
            text_entry = read_renderer_data_content_addressed_text_pair(
                {
                    **dict(text_cache),
                    "schema_version": RENDERER_DATA_TEXT_REFERENCE_SCHEMA,
                    "tags_record": tags_record,
                    "lyrics_record": lyrics_record,
                },
                condition=condition,
                expected_provenance=self.text_provenance,
                expected_cache_config_sha256=text_cache_config_sha256,
                expected_cache_config_file_sha256=text_cache_config_file_sha256,
                tags_base_dir=self.tags_text_cache_loader.record_base(
                    tags_sample_id
                ),
                lyrics_base_dir=self.lyrics_text_cache_loader.record_base(
                    lyrics_sample_id
                ),
            )
        else:
            text_record = _mapping(
                text_cache.get("record"), field=f"{sample_id}.text_cache.record"
            )
            if str(text_record.get("sample_id") or "") != sample_id:
                raise RuntimeError(f"{sample_id} text-cache sample_id mismatch")
            text_entry = read_text_cache_entry(
                text_record,
                expected_provenance=self.text_provenance,
                expected_cache_config_sha256=text_cache_config_sha256,
                expected_condition=condition,
                expected_cache_config_file_sha256=text_cache.get(
                    "cache_config_file_sha256"
                ),
                base_dir=text_record_base,
            )
        return {
            **item,
            "description_embeddings": text_entry.description_embeddings,
            "description_input_ids": text_entry.description_input_ids,
            "description_mask": text_entry.description_mask,
            "lyrics_embeddings": text_entry.lyrics_embeddings,
            "lyrics_input_ids": text_entry.lyrics_input_ids,
            "lyrics_mask": text_entry.lyrics_mask,
        }
