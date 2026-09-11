
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from .contracts import LATENT_FRAME_HZ
from .sample_data import CanonicalRenderSampleDataset
from .text_cache import (
    RENDER_CONDITION_SCHEMA,
    RENDERER_DATA_TEXT_REFERENCE_SCHEMA,
    RenderTextCacheLoader,
    RenderTextCondition,
    read_renderer_data_content_addressed_text_pair,
    read_text_cache_entry,
)

RENDERER_DATA_CROP_ROW_SCHEMA = "oqm.render.renderer_data-short-crop.v1"
RENDERER_DATA_CROP_READY_SCHEMA = "oqm.render.renderer_data-short-crop-ready.v1"
RENDERER_DATA_CROP_READY_STATUS = "RENDERER_DATA_SHORT_CROP_READY"
RENDERER_DATA_PARENT_SAMPLER_STATE_VERSION = "oqm.render.renderer_data-parent-sampler.v1"
RENDERER_DATA_PARENT_SUBSET_SCHEMA = "oqm.render.renderer_data-parent-subset.v1"
RENDERER_DATA_WINDOW_SUBSET_SCHEMA = "oqm.render.renderer_data-window-subset.v1"
RENDERER_DATA_RNG_BYTES_SCHEMA = "oqm.render.renderer_data-rng-bytes.v1"
RENDERER_DATA_LOUDNESS_ROW_SCHEMA = "oqm.render.renderer_data-window-loudness.v1"
RENDERER_DATA_LOUDNESS_READY_SCHEMA = "oqm.render.renderer_data-window-loudness-ready.v1"
RENDERER_DATA_LOUDNESS_READY_STATUS = "RENDERER_DATA_WINDOW_LOUDNESS_READY"
def _matches_current(value: Any, current: str) -> bool:
    return value == current


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a lowercase hexadecimal SHA-256") from exc
    if value != value.lower():
        raise ValueError(f"{field} must be a lowercase hexadecimal SHA-256")
    return value


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _resolve_path(value: Any, *, base_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError(f"{field} must be a local file path")
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    root = base_dir.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} must remain within the crop manifest directory") from exc
    return resolved


def encode_renderer_data_rng_state(value: torch.Tensor) -> dict[str, Any]:

    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.ndim != 1
    ):
        raise RuntimeError("RendererData parent order RNG state must be a one-dimensional uint8 tensor")
    payload = bytes(value.detach().to(device="cpu").contiguous().tolist())
    return {
        "schema_version": RENDERER_DATA_RNG_BYTES_SCHEMA,
        "num_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "base64": base64.b64encode(payload).decode("ascii"),
    }


def decode_renderer_data_rng_state(value: Any) -> torch.Tensor:

    if isinstance(value, torch.Tensor):
        if value.dtype != torch.uint8 or value.ndim != 1:
            raise RuntimeError("checkpoint RendererData parent order RNG tensor is invalid")
        return value.detach().to(device="cpu").contiguous()
    expected_fields = {"schema_version", "num_bytes", "sha256", "base64"}
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise RuntimeError("checkpoint RendererData parent order RNG encoding is invalid")
    if not _matches_current(value.get("schema_version"), RENDERER_DATA_RNG_BYTES_SCHEMA):
        raise RuntimeError("checkpoint RendererData parent order RNG schema is incompatible with")
    num_bytes = _require_int(
        value.get("num_bytes"), field="rng_state.num_bytes", minimum=1
    )
    expected_sha256 = _require_sha256(value.get("sha256"), field="rng_state.sha256")
    encoded = value.get("base64")
    if not isinstance(encoded, str) or not encoded:
        raise RuntimeError("checkpoint RendererData parent order RNG base64 value is invalid")
    try:
        payload = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError, binascii.Error) as exc:
        raise RuntimeError(
            "checkpoint RendererData parent order RNG base64 value could not be parsed"
        ) from exc
    if len(payload) != num_bytes:
        raise RuntimeError("checkpoint RendererData parent order RNG byte count does not match")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RuntimeError("checkpoint RendererData parent-order RNG SHA does not match")
    return torch.tensor(list(payload), dtype=torch.uint8)


@dataclass(frozen=True)
class RendererDataCropWindow:
    sample_id: str
    window_index: int
    start_frame: int
    end_frame: int
    condition: Mapping[str, Any]
    text_cache: Mapping[str, Any]

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class RendererDataCropParent:
    parent_index: int
    parent_sample_id: str
    parent_latent_sha256: str
    parent_semantic_sha256: str
    parent_condition_sha256: str
    windows: tuple[RendererDataCropWindow, ...]


class RendererDataWindowLoudnessCache:

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        expected_manifest_sha256: str,
        ready_path: str | Path,
        expected_ready_sha256: str,
        expected_crop_manifest_sha256: str,
        evaluation_parent_windows: Mapping[int, Sequence[str]] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        self.ready_path = Path(ready_path).resolve(strict=True)
        self.manifest_sha256 = _require_sha256(
            expected_manifest_sha256,
            field="expected_loudness_manifest_sha256",
        )
        if (
            evaluation_parent_windows is None
            and _file_sha256(self.manifest_path) != self.manifest_sha256
        ):
            raise RuntimeError("RendererData loudness manifest SHA does not match")
        ready_sha256 = _require_sha256(
            expected_ready_sha256,
            field="expected_loudness_ready_sha256",
        )
        if _file_sha256(self.ready_path) != ready_sha256:
            raise RuntimeError("RendererData loudness READY SHA does not match")
        try:
            ready = json.loads(self.ready_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("RendererData loudness READY could not be parsed") from exc
        if not isinstance(ready, Mapping):
            raise TypeError("RendererData loudness READY must be a JSON object")
        if (
            not _matches_current(
                ready.get("schema_version"), RENDERER_DATA_LOUDNESS_READY_SCHEMA
            )
            or not _matches_current(
                ready.get("status"), RENDERER_DATA_LOUDNESS_READY_STATUS
            )
        ):
            raise RuntimeError("RendererData loudness READY schema or status is incompatible with")
        declared_manifest = _resolve_path(
            ready.get("manifest"),
            base_dir=self.ready_path.parent,
            field="loudness READY.manifest",
        )
        if declared_manifest != self.manifest_path:
            raise RuntimeError("RendererData loudness READY references a different manifest")
        if ready.get("manifest_sha256") != self.manifest_sha256:
            raise RuntimeError("RendererData loudness READY manifest SHA does not match")
        if ready.get("source_crop_manifest_sha256") != (expected_crop_manifest_sha256):
            raise RuntimeError("RendererData loudness data is not bound to the current crop manifest")

        selected_windows: dict[int, frozenset[str]] | None = None
        if evaluation_parent_windows is not None:
            selected_windows = {}
            for parent_index, sample_ids in evaluation_parent_windows.items():
                if (
                    not isinstance(parent_index, int)
                    or isinstance(parent_index, bool)
                    or parent_index < 0
                    or not isinstance(sample_ids, Sequence)
                    or isinstance(sample_ids, (str, bytes))
                    or not sample_ids
                ):
                    raise ValueError("RendererData sparse loudness parent or window selection is invalid")
                frozen_ids = frozenset(str(sample_id) for sample_id in sample_ids)
                if len(frozen_ids) != len(sample_ids) or any(
                    not sample_id for sample_id in frozen_ids
                ):
                    raise ValueError("RendererData sparse loudness window sample_id is empty or duplicated")
                selected_windows[parent_index] = frozen_ids
            if not selected_windows:
                raise ValueError("RendererData sparse loudness selection must not be empty")

        selected_max_parent = max(selected_windows) if selected_windows else None
        values: dict[str, float] = {}
        parents = 0
        logical_parent_index = 0
        try:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    parent_index = logical_parent_index
                    logical_parent_index += 1
                    if (
                        selected_windows is not None
                        and parent_index not in selected_windows
                    ):
                        if parent_index >= selected_max_parent:
                            break
                        continue
                    row = json.loads(line)
                    if (
                        not isinstance(row, Mapping)
                        or not _matches_current(
                            row.get("schema_version"), RENDERER_DATA_LOUDNESS_ROW_SCHEMA
                        )
                    ):
                        raise RuntimeError(
                            f"RendererData loudness manifest: {line_number} schema is incompatible with"
                        )
                    windows = row.get("windows")
                    if (
                        not isinstance(windows, Sequence)
                        or isinstance(windows, (str, bytes))
                        or not windows
                    ):
                        raise RuntimeError(
                            f"RendererData loudness manifest: {line_number} windows are invalid"
                        )
                    if row.get("parent_index") != parent_index:
                        raise RuntimeError("RendererData loudness parent_index does not match the line number")
                    parents += 1
                    wanted_ids = (
                        selected_windows[parent_index]
                        if selected_windows is not None
                        else None
                    )
                    for window in windows:
                        if not isinstance(window, Mapping):
                            raise TypeError("RendererData loudness window must be a mapping")
                        sample_id = window.get("sample_id")
                        value = window.get("integrated_lufs")
                        if wanted_ids is not None and sample_id not in wanted_ids:
                            continue
                        if (
                            not isinstance(sample_id, str)
                            or not sample_id
                            or sample_id in values
                            or isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(float(value))
                        ):
                            raise RuntimeError("RendererData loudness window has an invalid identity or value")
                        values[sample_id] = float(value)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("RendererData loudness manifest could not be read") from exc
        if selected_windows is None:
            if parents != ready.get("parents") or len(values) != ready.get("windows"):
                raise RuntimeError("RendererData loudness READY counts do not match the manifest")
        else:
            expected_sample_ids = frozenset(
                sample_id
                for sample_ids in selected_windows.values()
                for sample_id in sample_ids
            )
            if frozenset(values) != expected_sample_ids:
                missing = sorted(expected_sample_ids - values.keys())
                raise RuntimeError(f"RendererData sparse loudness does not cover the frozen window: {missing[:8]}")
        self.values = values
        self.ready = dict(ready)
        self.ready_sha256 = ready_sha256

    def value(self, sample_id: str) -> float:
        try:
            return self.values[sample_id]
        except KeyError as exc:
            raise RuntimeError(f"RendererData loudness data is missing a crop window: {sample_id}") from exc


class RendererDataShortWindowDataset(Dataset[dict[str, Any]]):

    def __init__(
        self,
        parent_dataset: CanonicalRenderSampleDataset,
        *,
        crop_manifest_path: str | Path,
        expected_crop_manifest_sha256: str,
        crop_ready_path: str | Path,
        expected_crop_ready_sha256: str,
        verify_parent_records: bool = True,
        tags_text_cache_loader: RenderTextCacheLoader | None = None,
        lyrics_text_cache_loader: RenderTextCacheLoader | None = None,
        parent_subset: Mapping[str, Any] | None = None,
        window_subset: Mapping[str, Any] | None = None,
        loudness_manifest_path: str | Path | None = None,
        expected_loudness_manifest_sha256: str | None = None,
        loudness_ready_path: str | Path | None = None,
        expected_loudness_ready_sha256: str | None = None,
        evaluation_selection: Mapping[str, Any] | None = None,
    ) -> None:
        self.parent_dataset = parent_dataset
        self.crop_manifest_path = Path(crop_manifest_path).resolve(strict=True)
        self.crop_ready_path = Path(crop_ready_path).resolve(strict=True)
        self.base_dir = self.crop_manifest_path.parent
        if (tags_text_cache_loader is None) != (lyrics_text_cache_loader is None):
            raise ValueError("RendererData tags and lyrics content cache loaders must be provided together")
        self.tags_text_cache_loader = tags_text_cache_loader
        self.lyrics_text_cache_loader = lyrics_text_cache_loader
        self.crop_manifest_sha256 = _require_sha256(
            expected_crop_manifest_sha256,
            field="expected_crop_manifest_sha256",
        )
        if (
            evaluation_selection is None
            and _file_sha256(self.crop_manifest_path) != self.crop_manifest_sha256
        ):
            raise RuntimeError("RendererData crop manifest SHA does not match")
        ready_sha256 = _require_sha256(
            expected_crop_ready_sha256,
            field="expected_crop_ready_sha256",
        )
        if _file_sha256(self.crop_ready_path) != ready_sha256:
            raise RuntimeError("RendererData crop READY SHA does not match")
        try:
            ready = json.loads(self.crop_ready_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("RendererData crop READY could not be parsed") from exc
        if not isinstance(ready, Mapping):
            raise TypeError("RendererData crop READY must be a JSON object")
        if (
            not _matches_current(
                ready.get("schema_version"), RENDERER_DATA_CROP_READY_SCHEMA
            )
            or not _matches_current(ready.get("status"), RENDERER_DATA_CROP_READY_STATUS)
        ):
            raise RuntimeError("RendererData crop READY schema or status is incompatible with")
        declared_manifest = _resolve_path(
            ready.get("manifest"),
            base_dir=self.crop_ready_path.parent,
            field="crop READY.manifest",
        )
        if declared_manifest != self.crop_manifest_path:
            raise RuntimeError("RendererData crop READY references a different manifest")
        if ready.get("manifest_sha256") != self.crop_manifest_sha256:
            raise RuntimeError("RendererData crop READY manifest SHA does not match")
        if ready.get("parent_manifest_sha256") != parent_dataset.manifest_sha256:
            raise RuntimeError("RendererData crop data is not bound to the current parent manifest")
        if ready.get("split") != parent_dataset.split:
            raise RuntimeError("RendererData crop split does not match the parent manifest")

        selected_parent_windows = self._parse_evaluation_selection(
            evaluation_selection,
            parent_dataset=parent_dataset,
            crop_manifest_sha256=self.crop_manifest_sha256,
            crop_ready_sha256=ready_sha256,
            crop_rows_sha256=ready.get("rows_sha256"),
        )
        rows = self._load_crop_rows(
            selected_parent_windows=selected_parent_windows,
        )
        if selected_parent_windows is None:
            if ready.get("rows_sha256") != _canonical_sha256(rows):
                raise RuntimeError("RendererData crop READY rows SHA does not match")
            if ready.get("parents") != len(rows):
                raise RuntimeError("RendererData crop READY parent count does not match")
            if len(rows) != len(parent_dataset):
                raise RuntimeError("RendererData crop data must cover every parent record")

        parent_sample_ids = (
            parent_dataset.sample_ids if selected_parent_windows is None else None
        )
        parents: list[RendererDataCropParent] = []
        seen_window_ids: set[str] = set()
        window_count = 0
        expected_parent_indices = (
            range(len(rows))
            if selected_parent_windows is None
            else selected_parent_windows.keys()
        )
        for expected_parent_index, raw in zip(
            expected_parent_indices,
            rows,
            strict=True,
        ):
            expected_parent_sample_id = (
                parent_sample_ids[expected_parent_index]
                if parent_sample_ids is not None
                else str(parent_dataset.records[expected_parent_index]["sample_id"])
            )
            parent = self._parse_parent(
                raw,
                expected_parent_index=expected_parent_index,
                expected_parent_sample_id=expected_parent_sample_id,
                expected_parent_frames=parent_dataset.frame_length(
                    expected_parent_index
                ),
                expected_parent_manifest_sha256=parent_dataset.manifest_sha256,
                expected_split=parent_dataset.split,
            )
            duplicate_windows = seen_window_ids.intersection(
                window.sample_id for window in parent.windows
            )
            if duplicate_windows:
                raise RuntimeError(
                    f"RendererData crop window sample_id is duplicated: {sorted(duplicate_windows)[:8]}"
                )
            seen_window_ids.update(window.sample_id for window in parent.windows)
            if verify_parent_records:
                self._verify_parent_record(parent)
            if selected_parent_windows is not None:
                requested = selected_parent_windows[expected_parent_index]
                available = {window.window_index: window for window in parent.windows}
                missing = sorted(set(requested) - available.keys())
                if missing:
                    raise RuntimeError(
                        "RendererData sparse evaluation window does not exist:"
                        f"parent_index={expected_parent_index} windows={missing}"
                    )
                parent = RendererDataCropParent(
                    parent_index=parent.parent_index,
                    parent_sample_id=parent.parent_sample_id,
                    parent_latent_sha256=parent.parent_latent_sha256,
                    parent_semantic_sha256=parent.parent_semantic_sha256,
                    parent_condition_sha256=parent.parent_condition_sha256,
                    windows=tuple(available[index] for index in requested),
                )
            parents.append(parent)
            window_count += len(parent.windows)
        if selected_parent_windows is None and ready.get("windows") != window_count:
            raise RuntimeError("RendererData crop READY window count does not match")

        if selected_parent_windows is not None and (
            parent_subset is not None or window_subset is not None
        ):
            raise ValueError("RendererData sparse evaluation cannot overlap the training parent or window subset")
        self.parent_subset = self._parse_parent_subset(
            parent_subset,
            parents=parents,
            source_manifest_sha256=parent_dataset.manifest_sha256,
        )
        if self.parent_subset is None:
            selected_parents = parents
        else:
            selected_parents = [
                parents[index] for index in self.parent_subset["parent_indices"]
            ]
        self.window_subset = self._parse_window_subset(
            window_subset,
            parents=selected_parents,
            source_crop_manifest_sha256=self.crop_manifest_sha256,
        )
        if self.window_subset is not None:
            selected_parents = [
                RendererDataCropParent(
                    parent_index=parent.parent_index,
                    parent_sample_id=parent.parent_sample_id,
                    parent_latent_sha256=parent.parent_latent_sha256,
                    parent_semantic_sha256=parent.parent_semantic_sha256,
                    parent_condition_sha256=parent.parent_condition_sha256,
                    windows=(parent.windows[window_index],),
                )
                for parent, window_index in zip(
                    selected_parents,
                    self.window_subset["window_indices"],
                    strict=True,
                )
            ]
        self.parents = tuple(selected_parents)
        self.ready = dict(ready)
        self.ready_path = self.crop_ready_path
        self.ready_sha256 = ready_sha256
        self.index_path = self.crop_manifest_path
        self.index_sha256 = self.crop_manifest_sha256
        self.split_groups_disjoint_verified = (
            parent_dataset.split_groups_disjoint_verified
        )
        self._sample_ids = tuple(parent.parent_sample_id for parent in self.parents)
        self._window_frame_lengths = tuple(
            tuple(window.frames for window in parent.windows) for parent in self.parents
        )


        self._frame_lengths = tuple(
            max(values) for values in self._window_frame_lengths
        )
        loudness_fields = (
            loudness_manifest_path,
            expected_loudness_manifest_sha256,
            loudness_ready_path,
            expected_loudness_ready_sha256,
        )
        if any(value is not None for value in loudness_fields) and not all(
            value is not None for value in loudness_fields
        ):
            raise ValueError("RendererData loudness manifest and READY paths and SHAs must be fully declared")
        self.loudness_cache = (
            RendererDataWindowLoudnessCache(
                manifest_path=loudness_manifest_path,
                expected_manifest_sha256=str(expected_loudness_manifest_sha256),
                ready_path=loudness_ready_path,
                expected_ready_sha256=str(expected_loudness_ready_sha256),
                expected_crop_manifest_sha256=self.crop_manifest_sha256,
                evaluation_parent_windows=(
                    {
                        parent.parent_index: tuple(
                            window.sample_id for window in parent.windows
                        )
                        for parent in self.parents
                    }
                    if selected_parent_windows is not None
                    else None
                ),
            )
            if all(value is not None for value in loudness_fields)
            else None
        )

    def _load_crop_rows(
        self,
        *,
        selected_parent_windows: Mapping[int, Sequence[int]] | None,
    ) -> list[dict[str, Any]]:
        rows_by_parent: dict[int, dict[str, Any]] = {}
        rows: list[dict[str, Any]] = []
        selected_max_parent = (
            max(selected_parent_windows) if selected_parent_windows else None
        )
        logical_parent_index = 0
        try:
            with self.crop_manifest_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    parent_index = logical_parent_index
                    logical_parent_index += 1
                    if (
                        selected_parent_windows is not None
                        and parent_index not in selected_parent_windows
                    ):
                        if parent_index >= selected_max_parent:
                            break
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise TypeError(f"crop manifest line  {line_number} must be a JSON object")
                    if selected_parent_windows is None:
                        rows.append(value)
                    else:
                        rows_by_parent[parent_index] = value
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("RendererData crop manifest could not be parsed") from exc
        if selected_parent_windows is None:
            if not rows:
                raise ValueError("RendererData crop manifest must not be empty")
            return rows
        missing = sorted(set(selected_parent_windows) - rows_by_parent.keys())
        if missing:
            raise RuntimeError(f"RendererData sparse evaluation parent was replaced by crop coverage: {missing[:8]}")
        return [rows_by_parent[index] for index in selected_parent_windows]

    @staticmethod
    def _parse_evaluation_selection(
        value: Mapping[str, Any] | None,
        *,
        parent_dataset: CanonicalRenderSampleDataset,
        crop_manifest_sha256: str,
        crop_ready_sha256: str,
        crop_rows_sha256: Any,
    ) -> dict[int, tuple[int, ...]] | None:

        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError("RendererData evaluation selection must be a mapping")
        expected = {
            "source_expected_records": len(parent_dataset),
            "source_manifest_sha256": parent_dataset.manifest_sha256,
            "source_crop_manifest_sha256": crop_manifest_sha256,
            "source_crop_ready_sha256": crop_ready_sha256,
            "source_crop_rows_sha256": crop_rows_sha256,
        }
        mismatches = {
            name: {"expected": expected_value, "actual": value.get(name)}
            for name, expected_value in expected.items()
            if value.get(name) != expected_value
        }
        rows = value.get("rows")
        if (
            mismatches
            or not isinstance(rows, Sequence)
            or isinstance(rows, (str, bytes))
            or not rows
        ):
            raise RuntimeError(
                f"RendererData evaluation selection has a mismatched source identity or invalid rows: {mismatches}"
            )
        if value.get("rows_sha256") != _canonical_sha256(rows):
            raise RuntimeError("RendererData evaluation selection rows SHA does not match")
        selected: dict[int, list[int]] = {}
        for position, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TypeError(f"RendererData evaluation selection row {position} is invalid")
            parent_index = row.get("parent_index")
            window_index = row.get("window_index")
            if (
                not isinstance(parent_index, int)
                or isinstance(parent_index, bool)
                or not 0 <= parent_index < len(parent_dataset)
                or not isinstance(window_index, int)
                or isinstance(window_index, bool)
                or window_index < 0
            ):
                raise ValueError("RendererData evaluation selection parent or window is invalid")
            selected.setdefault(parent_index, []).append(window_index)
        if any(len(indices) != len(set(indices)) for indices in selected.values()):
            raise ValueError("RendererData evaluation selection contains duplicate parent or window entries")
        return {index: tuple(indices) for index, indices in selected.items()}

    @staticmethod
    def _parse_parent_subset(
        value: Mapping[str, Any] | None,
        *,
        parents: Sequence[RendererDataCropParent],
        source_manifest_sha256: str,
    ) -> dict[str, Any] | None:

        if value is None:
            return None
        required = {
            "schema_version",
            "source_expected_records",
            "source_manifest_sha256",
            "parent_indices",
            "parent_sample_ids",
            "ordered_parent_sample_ids_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("RendererData parent subset fields are incomplete or contain unknown entries")
        if not _matches_current(
            value.get("schema_version"), RENDERER_DATA_PARENT_SUBSET_SCHEMA
        ):
            raise RuntimeError("RendererData parent subset schema is incompatible with")
        source_expected_records = _require_int(
            value.get("source_expected_records"),
            field="parent_subset.source_expected_records",
            minimum=1,
        )
        if source_expected_records != len(parents):
            raise RuntimeError("RendererData parent subset source record count does not match")
        if value.get("source_manifest_sha256") != source_manifest_sha256:
            raise RuntimeError("RendererData parent subset is bound to a different parent manifest")
        raw_indices = value.get("parent_indices")
        raw_sample_ids = value.get("parent_sample_ids")
        if (
            not isinstance(raw_indices, Sequence)
            or isinstance(raw_indices, (str, bytes))
            or not raw_indices
            or not isinstance(raw_sample_ids, Sequence)
            or isinstance(raw_sample_ids, (str, bytes))
            or len(raw_indices) != len(raw_sample_ids)
        ):
            raise ValueError(
                "RendererData parent subset indices and sample_ids must be non-empty sequences of equal length"
            )
        indices = tuple(
            _require_int(index, field="parent_subset.parent_indices")
            for index in raw_indices
        )
        if len(set(indices)) != len(indices) or any(
            index >= len(parents) for index in indices
        ):
            raise ValueError("RendererData parent subset index is duplicated or out of bounds")
        sample_ids = tuple(str(sample_id) for sample_id in raw_sample_ids)
        if any(not sample_id for sample_id in sample_ids) or len(
            set(sample_ids)
        ) != len(sample_ids):
            raise ValueError("RendererData parent subset sample_id is empty or duplicated")
        actual_sample_ids = tuple(parents[index].parent_sample_id for index in indices)
        if sample_ids != actual_sample_ids:
            raise RuntimeError(
                "RendererData parent subset index/sample_id does not match the parent manifest"
            )
        declared_order_sha256 = _require_sha256(
            value.get("ordered_parent_sample_ids_sha256"),
            field="parent_subset.ordered_parent_sample_ids_sha256",
        )
        if declared_order_sha256 != _canonical_sha256(sample_ids):
            raise RuntimeError("RendererData ordered parent subset sample IDs SHA does not match")
        return {
            **dict(value),
            "parent_indices": indices,
            "parent_sample_ids": sample_ids,
        }

    @staticmethod
    def _parse_window_subset(
        value: Mapping[str, Any] | None,
        *,
        parents: Sequence[RendererDataCropParent],
        source_crop_manifest_sha256: str,
    ) -> dict[str, Any] | None:

        if value is None:
            return None
        required = {
            "schema_version",
            "source_crop_manifest_sha256",
            "parent_sample_ids",
            "window_indices",
            "window_sample_ids",
            "ordered_parent_windows_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("RendererData window subset fields are incomplete or contain unknown entries")
        if not _matches_current(
            value.get("schema_version"), RENDERER_DATA_WINDOW_SUBSET_SCHEMA
        ):
            raise RuntimeError("RendererData window subset schema is incompatible with")
        if value.get("source_crop_manifest_sha256") != source_crop_manifest_sha256:
            raise RuntimeError("RendererData window subset is bound to a different crop manifest")
        raw_parent_ids = value.get("parent_sample_ids")
        raw_indices = value.get("window_indices")
        raw_window_ids = value.get("window_sample_ids")
        if any(
            not isinstance(raw, Sequence) or isinstance(raw, (str, bytes))
            for raw in (raw_parent_ids, raw_indices, raw_window_ids)
        ):
            raise ValueError("RendererData window subset list field is invalid")
        assert isinstance(raw_parent_ids, Sequence)
        assert isinstance(raw_indices, Sequence)
        assert isinstance(raw_window_ids, Sequence)
        if (
            not len(parents)
            == len(raw_parent_ids)
            == len(raw_indices)
            == len(raw_window_ids)
        ):
            raise ValueError("RendererData window subset must map one-to-one to parent records")
        parent_ids = tuple(str(value) for value in raw_parent_ids)
        expected_parent_ids = tuple(parent.parent_sample_id for parent in parents)
        if parent_ids != expected_parent_ids:
            raise RuntimeError("RendererData window subset parent sequence or identity does not match")
        indices = tuple(
            _require_int(index, field="window_subset.window_indices")
            for index in raw_indices
        )
        if any(
            index >= len(parent.windows)
            for parent, index in zip(parents, indices, strict=True)
        ):
            raise ValueError("RendererData window subset index is out of bounds")
        window_ids = tuple(str(value) for value in raw_window_ids)
        expected_window_ids = tuple(
            parent.windows[index].sample_id
            for parent, index in zip(parents, indices, strict=True)
        )
        if window_ids != expected_window_ids:
            raise RuntimeError("RendererData window subset identity does not match")
        ordered = [
            {
                "parent_sample_id": parent_id,
                "window_index": index,
                "window_sample_id": window_id,
            }
            for parent_id, index, window_id in zip(
                parent_ids, indices, window_ids, strict=True
            )
        ]
        declared_sha256 = _require_sha256(
            value.get("ordered_parent_windows_sha256"),
            field="window_subset.ordered_parent_windows_sha256",
        )
        if declared_sha256 != _canonical_sha256(ordered):
            raise RuntimeError("RendererData ordered window subset parent/window SHA does not match")
        return {**dict(value), "window_indices": indices}

    @staticmethod
    def _parse_parent(
        raw: Mapping[str, Any],
        *,
        expected_parent_index: int,
        expected_parent_sample_id: str,
        expected_parent_frames: int,
        expected_parent_manifest_sha256: str,
        expected_split: str,
    ) -> RendererDataCropParent:
        if not _matches_current(raw.get("schema_version"), RENDERER_DATA_CROP_ROW_SCHEMA):
            raise RuntimeError("RendererData crop row schema is incompatible with")
        parent_index = _require_int(raw.get("parent_index"), field="crop.parent_index")
        parent_sample_id = raw.get("parent_sample_id")
        if (
            parent_index != expected_parent_index
            or parent_sample_id != expected_parent_sample_id
        ):
            raise RuntimeError("RendererData crop parent sequence or identity does not match the parent manifest")
        if raw.get("parent_manifest_sha256") != expected_parent_manifest_sha256:
            raise RuntimeError("RendererData crop row is bound to a different parent manifest")
        if raw.get("split") != expected_split:
            raise RuntimeError("RendererData crop row split does not match")
        parent_latent_sha256 = _require_sha256(
            raw.get("parent_latent_sha256"), field="crop.parent_latent_sha256"
        )
        parent_semantic_sha256 = _require_sha256(
            raw.get("parent_semantic_sha256"), field="crop.parent_semantic_sha256"
        )
        parent_condition_sha256 = _require_sha256(
            raw.get("parent_condition_sha256"),
            field="crop.parent_condition_sha256",
        )
        raw_windows = raw.get("windows")
        if (
            not isinstance(raw_windows, Sequence)
            or isinstance(raw_windows, (str, bytes))
            or not 1 <= len(raw_windows) <= 4
            or not all(isinstance(value, Mapping) for value in raw_windows)
        ):
            raise ValueError("each RendererData parent must have 1-4 windows")
        windows: list[RendererDataCropWindow] = []
        cursor = 0
        for window_index, value in enumerate(raw_windows):
            assert isinstance(value, Mapping)
            declared_index = _require_int(
                value.get("window_index"), field="crop.window_index"
            )
            start_frame = _require_int(
                value.get("start_frame"), field="crop.start_frame"
            )
            end_frame = _require_int(
                value.get("end_frame"), field="crop.end_frame", minimum=1
            )
            sample_id = value.get("sample_id")
            condition = value.get("condition")
            text_cache = value.get("text_cache")
            if declared_index != window_index:
                raise RuntimeError("RendererData crop window_index must start at 0 and increase consecutively")
            if start_frame != cursor or end_frame <= start_frame:
                raise RuntimeError(
                    "RendererData crop windows must be contiguous and non-overlapping, "
                    "without gaps"
                )
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("RendererData crop window.sample_id must be a non-empty string")
            if not isinstance(condition, Mapping):
                raise TypeError("RendererData crop window.condition must be a mapping")
            if not isinstance(text_cache, Mapping):
                raise TypeError("RendererData crop window.text_cache must be a mapping")
            condition_record = {
                "schema_version": RENDER_CONDITION_SCHEMA,
                "sample_id": sample_id,
                "condition": dict(condition),
            }
            parsed_condition = RenderTextCondition.from_mapping(condition_record)
            if text_cache.get("source_condition_sha256") != (
                parsed_condition.source_condition_sha256
            ):
                raise RuntimeError("RendererData crop text cache is not bound to the current window condition")
            windows.append(
                RendererDataCropWindow(
                    sample_id=sample_id,
                    window_index=window_index,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    condition=dict(condition),
                    text_cache=dict(text_cache),
                )
            )
            cursor = end_frame
        if cursor != expected_parent_frames:
            raise RuntimeError("RendererData crop data must cover the entire parent timeline")
        return RendererDataCropParent(
            parent_index=parent_index,
            parent_sample_id=str(parent_sample_id),
            parent_latent_sha256=parent_latent_sha256,
            parent_semantic_sha256=parent_semantic_sha256,
            parent_condition_sha256=parent_condition_sha256,
            windows=tuple(windows),
        )

    def _verify_parent_record(self, parent: RendererDataCropParent) -> None:
        raw = self.parent_dataset.records[parent.parent_index]
        condition = RenderTextCondition.from_mapping(raw)
        mismatches = {
            "latent": (
                parent.parent_latent_sha256,
                (raw.get("latent") or {}).get("sha256"),
            ),
            "semantic": (
                parent.parent_semantic_sha256,
                (raw.get("semantic") or {}).get("sha256"),
            ),
            "condition": (
                parent.parent_condition_sha256,
                condition.source_condition_sha256,
            ),
        }
        drift = {
            name: {"expected": expected, "actual": actual}
            for name, (expected, actual) in mismatches.items()
            if expected != actual
        }
        if drift:
            raise RuntimeError(
                f"RendererData crop parent artifact identity does not match: {drift}"
            )

    def __len__(self) -> int:
        return len(self.parents)

    @property
    def sample_ids(self) -> tuple[str, ...]:

        return self._sample_ids

    @property
    def frame_lengths(self) -> list[int]:
        return list(self._frame_lengths)

    @property
    def window_frame_lengths(self) -> tuple[tuple[int, ...], ...]:
        return self._window_frame_lengths

    @property
    def window_counts(self) -> tuple[int, ...]:
        return tuple(len(parent.windows) for parent in self.parents)

    def __getitem__(self, index: Any) -> dict[str, Any]:
        if (
            not isinstance(index, tuple)
            or len(index) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) for value in index
            )
        ):
            raise TypeError(
                "RendererData short dataset requires input from the parent-first sampler"
                "(parent_index, window_index)"
            )
        parent_index, window_index = index
        parent = self.parents[parent_index]
        window = parent.windows[window_index]
        load_parent_assets = getattr(
            self.parent_dataset, "load_renderer_data_parent_assets", None
        )
        item = (
            load_parent_assets(parent.parent_index)
            if callable(load_parent_assets)
            else self.parent_dataset[parent.parent_index]
        )
        if item.get("sample_id") != parent.parent_sample_id:
            raise RuntimeError("RendererData runtime parent sample_id does not match")
        provenance = item.get("provenance")
        if not isinstance(provenance, Mapping):
            raise RuntimeError("RendererData runtime parent provenance is missing")
        if (
            provenance.get("latent_sha256") != parent.parent_latent_sha256
            or provenance.get("semantic_sha256") != parent.parent_semantic_sha256
            or provenance.get("source_condition_sha256")
            != parent.parent_condition_sha256
        ):
            raise RuntimeError("RendererData runtime parent artifact identity does not match")

        text_cache = window.text_cache
        condition_record = {
            "schema_version": RENDER_CONDITION_SCHEMA,
            "sample_id": window.sample_id,
            "condition": dict(window.condition),
        }
        condition = RenderTextCondition.from_mapping(condition_record)
        record_base_value = text_cache.get("record_base_dir")
        text_record_base = (
            self.base_dir
            if record_base_value is None
            else _resolve_path(
                record_base_value,
                base_dir=self.base_dir,
                field="crop.text_cache.record_base_dir",
            )
        )
        if text_cache.get("schema_version") == RENDERER_DATA_TEXT_REFERENCE_SCHEMA:
            if (
                self.tags_text_cache_loader is None
                or self.lyrics_text_cache_loader is None
            ):
                raise RuntimeError(
                    "RendererData content-addressed text reference is missing the tags or lyrics cache loader"
                )
            tags_sha = str(text_cache.get("tags_content_sha256") or "")
            lyrics_sha = str(text_cache.get("lyrics_content_sha256") or "")
            tags_sample_id = f"renderer_data-tags:{tags_sha}"
            lyrics_sample_id = f"renderer_data-lyrics:{lyrics_sha}"
            content_reference = {
                **dict(text_cache),
                "tags_record": self.tags_text_cache_loader.record(tags_sample_id),
                "lyrics_record": self.lyrics_text_cache_loader.record(lyrics_sample_id),
            }
            text_entry = read_renderer_data_content_addressed_text_pair(
                content_reference,
                condition=condition,
                expected_provenance=self.parent_dataset.text_provenance,
                expected_cache_config_sha256=str(
                    text_cache.get("cache_config_sha256") or ""
                ),
                expected_cache_config_file_sha256=str(
                    text_cache.get("cache_config_file_sha256") or ""
                ),
                tags_base_dir=self.tags_text_cache_loader.record_base(tags_sample_id),
                lyrics_base_dir=self.lyrics_text_cache_loader.record_base(
                    lyrics_sample_id
                ),
            )
        else:
            text_record = text_cache.get("record")
            if not isinstance(text_record, Mapping):
                raise TypeError("RendererData crop text_cache.record must be a mapping")
            text_entry = read_text_cache_entry(
                text_record,
                expected_provenance=self.parent_dataset.text_provenance,
                expected_cache_config_sha256=str(
                    text_cache.get("cache_config_sha256") or ""
                ),
                expected_cache_config_file_sha256=str(
                    text_cache.get("cache_config_file_sha256") or ""
                ),
                expected_condition=condition,
                base_dir=text_record_base,
            )
        start = window.start_frame
        stop = window.end_frame
        result = dict(item)
        for name in ("latents", "latent_mask", "semantic_ids", "semantic_mask"):
            result[name] = item[name][start:stop]
        result.update(
            sample_id=window.sample_id,
            parent_sample_id=parent.parent_sample_id,
            renderer_data_window_index=window.window_index,
            description_embeddings=text_entry.description_embeddings,
            description_input_ids=text_entry.description_input_ids,
            description_mask=text_entry.description_mask,
            lyrics_embeddings=text_entry.lyrics_embeddings,
            lyrics_input_ids=text_entry.lyrics_input_ids,
            lyrics_mask=text_entry.lyrics_mask,
            duration_seconds=window.frames / LATENT_FRAME_HZ,
            provenance={
                **dict(provenance),
                "renderer_data_crop_manifest": str(self.crop_manifest_path),
                "renderer_data_crop_manifest_sha256": self.crop_manifest_sha256,
                "parent_sample_id": parent.parent_sample_id,
                "crop_start_frame": start,
                "crop_end_frame": stop,
                "crop_source_condition_sha256": (condition.source_condition_sha256),
            },
        )
        if self.loudness_cache is not None:
            result["global_loudness_lufs"] = self.loudness_cache.value(window.sample_id)
        return result


class RendererDataFixedValidationDataset(Dataset[dict[str, Any]]):

    def __init__(self, source: RendererDataShortWindowDataset, *, seed: int) -> None:
        self.source = source
        self.sample_ids = source.sample_ids
        self.frame_lengths = []
        self.window_indices: list[int] = []
        for sample_id, values in zip(
            self.sample_ids, source.window_frame_lengths, strict=True
        ):
            window_index = int.from_bytes(
                hashlib.sha256(
                    f"{int(seed)}:{sample_id}:renderer-data-fixed-valid-window".encode()
                ).digest()[:8],
                byteorder="big",
            ) % len(values)
            self.window_indices.append(window_index)
            self.frame_lengths.append(values[window_index])

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.source[(index, self.window_indices[index])]


class RendererDataParentBatchSampler(Sampler[list[int | tuple[int, int]]]):

    def __init__(
        self,
        *,
        sample_ids: Sequence[str],
        rank: int,
        world_size: int,
        batch_size_per_rank: int,
        gradient_accumulation_steps: int,
        global_parent_batch_size: int,
        seed: int,
        training_config_hash: str,
        view: str,
        window_counts: Sequence[int] | None = None,
        window_frame_lengths: Sequence[Sequence[int]] | None = None,
    ) -> None:
        self.sample_ids = tuple(str(value) for value in sample_ids)
        if not self.sample_ids or any(not value for value in self.sample_ids):
            raise ValueError("RendererData parent sampler requires non-null sample_ids")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("RendererData parent sample_ids must be unique")
        self.rank = _require_int(rank, field="sampler.rank")
        self.world_size = _require_int(
            world_size, field="sampler.world_size", minimum=1
        )
        if self.rank >= self.world_size:
            raise ValueError("RendererData parent sampler rank must be less than world_size")
        self.batch_size_per_rank = _require_int(
            batch_size_per_rank,
            field="sampler.batch_size_per_rank",
            minimum=1,
        )
        self.gradient_accumulation_steps = _require_int(
            gradient_accumulation_steps,
            field="sampler.gradient_accumulation_steps",
            minimum=1,
        )
        self.global_parent_batch_size = _require_int(
            global_parent_batch_size,
            field="sampler.global_parent_batch_size",
            minimum=1,
        )
        expected_global = (
            self.world_size
            * self.batch_size_per_rank
            * self.gradient_accumulation_steps
        )
        if self.global_parent_batch_size != expected_global:
            raise ValueError(
                "RendererData global parent batch must equal"
                "world_size×per-rank batch×gradient accumulation:"
                f"{self.global_parent_batch_size}!={expected_global}"
            )
        if len(self.sample_ids) < self.global_parent_batch_size:
            raise ValueError(
                "RendererData parent count must not be smaller than the global parent batch, which would repeat a parent within one update"
            )
        if view not in {"short", "full"}:
            raise ValueError("RendererData sampler view must be short or full")
        self.view = view
        self.seed = int(seed)
        if not str(training_config_hash):
            raise ValueError("RendererData sampler must be bound to the training config hash")
        self.training_config_hash = str(training_config_hash)

        if view == "short":
            if window_counts is None or window_frame_lengths is None:
                raise ValueError("RendererData short sampler is missing window counts or lengths")
            if len(window_counts) != len(self.sample_ids) or len(
                window_frame_lengths
            ) != len(self.sample_ids):
                raise ValueError("RendererData window metadata and parent counts do not match")
            counts = tuple(int(value) for value in window_counts)
            lengths = tuple(
                tuple(int(frame) for frame in values) for values in window_frame_lengths
            )
            if any(not 1 <= value <= 4 for value in counts):
                raise ValueError("each RendererData parent must have 1-4 windows")
            if any(
                len(values) != count or any(frame <= 0 for frame in values)
                for count, values in zip(counts, lengths, strict=True)
            ):
                raise ValueError("RendererData window counts or lengths are invalid")
        else:
            if window_counts is not None or window_frame_lengths is not None:
                raise ValueError("RendererData full sampler must not contain short-window metadata")
            counts = tuple(1 for _ in self.sample_ids)
            lengths = tuple((1,) for _ in self.sample_ids)
        self.window_counts = counts
        self.window_frame_lengths = lengths
        self.window_start_offsets = tuple(
            int.from_bytes(
                hashlib.sha256(
                    f"{self.seed}:{sample_id}:renderer-data-window-start".encode()
                ).digest()[:8],
                byteorder="big",
            )
            % count
            for sample_id, count in zip(
                self.sample_ids, self.window_counts, strict=True
            )
        )

        self.parent_order_generator = torch.Generator(device="cpu")
        self.parent_order_generator.manual_seed(self.seed)
        self.parent_order = tuple(
            torch.randperm(
                len(self.sample_ids), generator=self.parent_order_generator
            ).tolist()
        )
        self.epoch = 0
        self.optimizer_batch_cursor = 0
        self.microstep_cursor = 0
        self.parent_sampler_cursor = 0
        self.parent_cycles = 0
        self.window_cursors = [0 for _ in self.sample_ids]
        self.global_step = 0
        self.topology = {
            "format_version": RENDERER_DATA_PARENT_SAMPLER_STATE_VERSION,
            "world_size": self.world_size,
            "rank_independent": True,
            "batch_size_per_rank": self.batch_size_per_rank,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "global_parent_batch_size": self.global_parent_batch_size,
            "dataset_size": len(self.sample_ids),
            "sample_ids_sha256": _canonical_sha256(self.sample_ids),
            "seed": self.seed,
            "view": self.view,
            "window_counts_sha256": _canonical_sha256(self.window_counts),
            "window_frame_lengths_sha256": _canonical_sha256(self.window_frame_lengths),
            "window_start_offsets_sha256": _canonical_sha256(self.window_start_offsets),
            "parent_order_sha256": _canonical_sha256(self.parent_order),
            "training_config_hash": self.training_config_hash,
        }
        self.sampler_config_hash = _canonical_sha256(self.topology)

    @property
    def num_optimizer_batches_per_epoch(self) -> int:
        return (
            len(self.sample_ids) + self.global_parent_batch_size - 1
        ) // self.global_parent_batch_size

    @property
    def num_batches_per_epoch(self) -> int:
        return self.num_optimizer_batches_per_epoch * self.gradient_accumulation_steps

    @property
    def cursor(self) -> int:
        return (
            self.optimizer_batch_cursor * self.gradient_accumulation_steps
            + self.microstep_cursor
        )

    @property
    def curriculum_telemetry(self) -> dict[str, Any]:
        return {
            "protocol": "renderer_data_parent_first",
            "view": self.view,
            "global_step": self.global_step,
            "global_parent_batch_size": self.global_parent_batch_size,
            "parent_sampler_cursor": self.parent_sampler_cursor,
            "parent_cycles": self.parent_cycles,
            "optimizer_batch_cursor": self.optimizer_batch_cursor,
            "microstep_cursor": self.microstep_cursor,
        }

    def set_global_step(self, global_step: int) -> bool:
        if int(global_step) < 0:
            raise ValueError("RendererData sampler global_step must not be negative")
        self.global_step = int(global_step)
        return False

    def _global_parents(self, parent_cursor: int) -> list[int]:
        size = len(self.parent_order)
        return [
            self.parent_order[(parent_cursor + offset) % size]
            for offset in range(self.global_parent_batch_size)
        ]

    def _window_index(self, parent_index: int, window_cursor: int) -> int:
        return (
            self.window_start_offsets[parent_index] + window_cursor
        ) % self.window_counts[parent_index]

    def _global_microsteps(
        self,
        *,
        parent_cursor: int,
        window_cursors: Sequence[int],
    ) -> list[list[int | tuple[int, int]]]:
        parents = self._global_parents(parent_cursor)
        if self.view == "short":
            selected: list[int | tuple[int, int]] = [
                (
                    parent_index,
                    self._window_index(
                        parent_index, int(window_cursors[parent_index])
                    ),
                )
                for parent_index in parents
            ]


            selected.sort(
                key=lambda value: (
                    self.window_frame_lengths[value[0]][value[1]],  # type: ignore[index]
                    hashlib.sha256(
                        f"{self.seed}:{self.sample_ids[value[0]]}:batch-order".encode()  # type: ignore[index]
                    ).hexdigest(),
                )
            )
        else:
            selected = list(parents)
        global_microbatch = self.world_size * self.batch_size_per_rank
        return [
            selected[start : start + global_microbatch]
            for start in range(0, self.global_parent_batch_size, global_microbatch)
        ]

    def _local_slice(
        self, global_microstep: Sequence[int | tuple[int, int]]
    ) -> list[int | tuple[int, int]]:
        start = self.rank * self.batch_size_per_rank
        stop = start + self.batch_size_per_rank
        result = list(global_microstep[start:stop])
        if len(result) != self.batch_size_per_rank:
            raise AssertionError("RendererData sampler produced an incomplete per-rank microbatch")
        return result

    def __iter__(self) -> Iterator[list[int | tuple[int, int]]]:
        parent_cursor = self.parent_sampler_cursor
        window_cursors = list(self.window_cursors)
        first_microstep = self.microstep_cursor
        for optimizer_index in range(
            self.optimizer_batch_cursor,
            self.num_optimizer_batches_per_epoch,
        ):
            del optimizer_index
            parents = self._global_parents(parent_cursor)
            microsteps = self._global_microsteps(
                parent_cursor=parent_cursor,
                window_cursors=window_cursors,
            )
            for microstep_index in range(first_microstep, len(microsteps)):
                yield self._local_slice(microsteps[microstep_index])
            for parent_index in parents:
                window_cursors[parent_index] += 1
            parent_cursor = (parent_cursor + self.global_parent_batch_size) % len(
                self.parent_order
            )
            first_microstep = 0

    def __len__(self) -> int:
        return max(0, self.num_batches_per_epoch - self.cursor)

    def _advance_one(self) -> None:
        if self.cursor >= self.num_batches_per_epoch:
            raise RuntimeError("RendererData sampler cursor has reached the end of the current epoch")
        self.microstep_cursor += 1
        if self.microstep_cursor < self.gradient_accumulation_steps:
            return
        parents = self._global_parents(self.parent_sampler_cursor)
        for parent_index in parents:
            self.window_cursors[parent_index] += 1
        total_cursor = self.parent_sampler_cursor + self.global_parent_batch_size
        cycles, self.parent_sampler_cursor = divmod(
            total_cursor, len(self.parent_order)
        )
        self.parent_cycles += cycles
        self.optimizer_batch_cursor += 1
        self.microstep_cursor = 0

    def advance(self, count: int = 1) -> None:
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("RendererData sampler advance must be a non-negative integer")
        for _ in range(count):
            self._advance_one()

    def set_epoch(self, epoch: int, *, reset_cursor: bool = True) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("RendererData sampler epoch must be a non-negative integer")
        if reset_cursor and self.cursor not in {0, self.num_batches_per_epoch}:
            raise RuntimeError("RendererData sampler epoch must not reset while the batch cursor is active")
        self.epoch = epoch
        if reset_cursor:
            self.optimizer_batch_cursor = 0
            self.microstep_cursor = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": RENDERER_DATA_PARENT_SAMPLER_STATE_VERSION,
            "epoch": self.epoch,
            "optimizer_batch_cursor": self.optimizer_batch_cursor,
            "microstep_cursor": self.microstep_cursor,
            "parent_sampler_cursor": self.parent_sampler_cursor,
            "parent_cycles": self.parent_cycles,
            "window_cursors": list(self.window_cursors),
            "parent_order_rng_state": self.parent_order_generator.get_state(),
            "global_step": self.global_step,
            "topology": dict(self.topology),
            "config_hash": self.sampler_config_hash,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not _matches_current(
            state.get("format_version"), RENDERER_DATA_PARENT_SAMPLER_STATE_VERSION
        ):
            raise RuntimeError("checkpoint RendererData sampler format_version is incompatible with")
        stored_topology = state.get("topology")
        if not isinstance(stored_topology, Mapping):
            raise RuntimeError("checkpoint RendererData sampler is missing topology")
        if _canonical_sha256(stored_topology) != state.get("config_hash"):
            raise RuntimeError("checkpoint RendererData sampler topology or hash does not match")
        if dict(stored_topology) != self.topology:
            raise RuntimeError(
                "resume RendererData parent, window, or topology identity does not match"
            )
        integers = {
            name: int(state.get(name, -1))
            for name in (
                "epoch",
                "optimizer_batch_cursor",
                "microstep_cursor",
                "parent_sampler_cursor",
                "parent_cycles",
                "global_step",
            )
        }
        if any(value < 0 for value in integers.values()):
            raise RuntimeError("checkpoint RendererData sampler cursor or global_step is invalid")
        if integers["optimizer_batch_cursor"] > self.num_optimizer_batches_per_epoch:
            raise RuntimeError("checkpoint RendererData optimizer batch cursor is out of bounds")
        if not 0 <= integers["microstep_cursor"] < self.gradient_accumulation_steps:
            raise RuntimeError("checkpoint RendererData microstep cursor is out of bounds")
        if integers["parent_sampler_cursor"] >= len(self.parent_order):
            raise RuntimeError("checkpoint RendererData parent sampler cursor is out of bounds")
        if (
            integers["optimizer_batch_cursor"] == self.num_optimizer_batches_per_epoch
            and integers["microstep_cursor"] != 0
        ):
            raise RuntimeError("checkpoint RendererData epoch ended with an unfinished microstep")
        raw_window_cursors = state.get("window_cursors")
        if (
            not isinstance(raw_window_cursors, Sequence)
            or isinstance(raw_window_cursors, (str, bytes))
            or len(raw_window_cursors) != len(self.sample_ids)
        ):
            raise RuntimeError("checkpoint RendererData window cursors are missing or have the wrong length")
        window_cursors = [int(value) for value in raw_window_cursors]
        if any(value < 0 for value in window_cursors):
            raise RuntimeError("checkpoint RendererData window cursor is invalid")
        rng_state = decode_renderer_data_rng_state(state.get("parent_order_rng_state"))

        self.epoch = integers["epoch"]
        self.optimizer_batch_cursor = integers["optimizer_batch_cursor"]
        self.microstep_cursor = integers["microstep_cursor"]
        self.parent_sampler_cursor = integers["parent_sampler_cursor"]
        self.parent_cycles = integers["parent_cycles"]
        self.window_cursors = window_cursors
        self.global_step = integers["global_step"]
        self.parent_order_generator.set_state(rng_state.cpu())
