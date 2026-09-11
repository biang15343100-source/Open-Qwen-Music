from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import open_qwen_music.render.renderer_data as renderer_data_module
from open_qwen_music.render.renderer_data import (
    RENDERER_DATA_CROP_READY_SCHEMA,
    RENDERER_DATA_CROP_READY_STATUS,
    RENDERER_DATA_CROP_ROW_SCHEMA,
    RENDERER_DATA_PARENT_SUBSET_SCHEMA,
    RENDERER_DATA_WINDOW_SUBSET_SCHEMA,
    RendererDataFixedValidationDataset,
    RendererDataParentBatchSampler,
    RendererDataShortWindowDataset,
    decode_renderer_data_rng_state,
    encode_renderer_data_rng_state,
)
from open_qwen_music.render.text_cache import (
    RENDER_CONDITION_SCHEMA,
    REWRITER_SCHEMA_VERSION,
    RenderTextCondition,
)


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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _condition(sample_id: str, *, lyrics: str) -> tuple[dict[str, Any], str]:
    condition = {
        "schema_version": REWRITER_SCHEMA_VERSION,
        "description": "Genre: acoustic; Mood: calm",
        "lyrics": lyrics,
        "rewriter_revision": "rewriter-commit-1234",
        "text_tokenizer_revision": "tokenizer-commit-5678",
    }
    parsed = RenderTextCondition.from_mapping(
        {
            "schema_version": RENDER_CONDITION_SCHEMA,
            "sample_id": sample_id,
            "condition": condition,
        }
    )
    return condition, parsed.source_condition_sha256


class _FakeParentDataset:
    manifest_sha256 = "a" * 64
    split = "train"
    split_groups_disjoint_verified = True
    text_provenance = SimpleNamespace(name="fake-text-provenance")

    def __init__(self) -> None:
        self.assets_only_calls = 0
        self._sample_ids = ("parent-a", "parent-b")
        self._frame_lengths = (6, 4)
        self.records: list[dict[str, Any]] = []
        self.items: list[dict[str, Any]] = []
        for parent_index, (sample_id, frames) in enumerate(
            zip(self._sample_ids, self._frame_lengths, strict=True)
        ):
            condition, condition_sha256 = _condition(sample_id, lyrics="la la la")
            latent_sha256 = f"{parent_index + 1:x}" * 64
            semantic_sha256 = f"{parent_index + 3:x}" * 64
            self.records.append(
                {
                    "schema_version": "oqm.render-sample.v1",
                    "sample_id": sample_id,
                    "condition": condition,
                    "latent": {"sha256": latent_sha256},
                    "semantic": {"sha256": semantic_sha256},
                }
            )
            values = torch.arange(frames, dtype=torch.float32) + parent_index * 100
            self.items.append(
                {
                    "sample_id": sample_id,
                    "latents": values[:, None],
                    "latent_mask": torch.ones(frames, dtype=torch.bool),
                    "semantic_ids": values.to(dtype=torch.long),
                    "semantic_mask": torch.ones(frames, dtype=torch.bool),
                    "description_embeddings": torch.full((1, 2), -1.0),
                    "description_input_ids": torch.tensor([1]),
                    "description_mask": torch.ones(1, dtype=torch.bool),
                    "lyrics_embeddings": torch.full((1, 2), -1.0),
                    "lyrics_input_ids": torch.tensor([2]),
                    "lyrics_mask": torch.ones(1, dtype=torch.bool),
                    "duration_seconds": frames / 25.0,
                    "provenance": {
                        "latent_sha256": latent_sha256,
                        "semantic_sha256": semantic_sha256,
                        "source_condition_sha256": condition_sha256,
                    },
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self._sample_ids

    def frame_length(self, index: int) -> int:
        return self._frame_lengths[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]

    def load_renderer_data_parent_assets(self, index: int) -> dict[str, Any]:
        self.assets_only_calls += 1
        return self.items[index]


def _window(sample_id: str, index: int, start: int, end: int) -> dict[str, Any]:
    condition, condition_sha256 = _condition(sample_id, lyrics=f" {index} ")
    return {
        "sample_id": sample_id,
        "window_index": index,
        "start_frame": start,
        "end_frame": end,
        "condition": condition,
        "text_cache": {
            "source_condition_sha256": condition_sha256,
            "record": {"sample_id": sample_id},
            "cache_config_sha256": "c" * 64,
            "cache_config_file_sha256": "d" * 64,
        },
    }


def _publish_crop_contract(
    tmp_path: Path,
    parent_dataset: _FakeParentDataset,
    *,
    first_windows: list[dict[str, Any]] | None = None,
) -> tuple[Path, str, Path, str]:
    rows: list[dict[str, Any]] = []
    window_sets = (
        first_windows
        or [
            _window("parent-a#w0", 0, 0, 2),
            _window("parent-a#w1", 1, 2, 6),
        ],
        [_window("parent-b#w0", 0, 0, 4)],
    )
    for parent_index, windows in enumerate(window_sets):
        record = parent_dataset.records[parent_index]
        rows.append(
            {
                "schema_version": RENDERER_DATA_CROP_ROW_SCHEMA,
                "parent_index": parent_index,
                "parent_sample_id": record["sample_id"],
                "parent_manifest_sha256": parent_dataset.manifest_sha256,
                "split": parent_dataset.split,
                "parent_latent_sha256": record["latent"]["sha256"],
                "parent_semantic_sha256": record["semantic"]["sha256"],
                "parent_condition_sha256": RenderTextCondition.from_mapping(
                    record
                ).source_condition_sha256,
                "windows": windows,
            }
        )
    manifest = tmp_path / "renderer_data-crops.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_sha256 = _file_sha256(manifest)
    ready = tmp_path / "RENDERER_DATA_CROPS_READY.json"
    ready.write_text(
        json.dumps(
            {
                "schema_version": RENDERER_DATA_CROP_READY_SCHEMA,
                "status": RENDERER_DATA_CROP_READY_STATUS,
                "manifest": manifest.name,
                "manifest_sha256": manifest_sha256,
                "parent_manifest_sha256": parent_dataset.manifest_sha256,
                "split": parent_dataset.split,
                "rows_sha256": _canonical_sha256(rows),
                "parents": 2,
                "windows": sum(len(value) for value in window_sets),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest, manifest_sha256, ready, _file_sha256(ready)


def _build_dataset(
    tmp_path: Path,
    parent_dataset: _FakeParentDataset,
    *,
    first_windows: list[dict[str, Any]] | None = None,
    parent_subset: dict[str, Any] | None = None,
    window_subset: dict[str, Any] | None = None,
) -> RendererDataShortWindowDataset:
    manifest, manifest_sha256, ready, ready_sha256 = _publish_crop_contract(
        tmp_path,
        parent_dataset,
        first_windows=first_windows,
    )
    return RendererDataShortWindowDataset(
        parent_dataset,  # type: ignore[arg-type]
        crop_manifest_path=manifest,
        expected_crop_manifest_sha256=manifest_sha256,
        crop_ready_path=ready,
        expected_crop_ready_sha256=ready_sha256,
        parent_subset=parent_subset,
        window_subset=window_subset,
    )


def test_renderer_data_short_dataset_is_parent_weighted_and_slices_selected_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_dataset = _FakeParentDataset()
    dataset = _build_dataset(tmp_path, parent_dataset)
    text_entry = SimpleNamespace(
        description_embeddings=torch.full((2, 3), 11.0),
        description_input_ids=torch.tensor([11, 12]),
        description_mask=torch.ones(2, dtype=torch.bool),
        lyrics_embeddings=torch.full((3, 3), 22.0),
        lyrics_input_ids=torch.tensor([21, 22, 23]),
        lyrics_mask=torch.ones(3, dtype=torch.bool),
    )
    monkeypatch.setattr(
        renderer_data_module,
        "read_text_cache_entry",
        lambda *args, **kwargs: text_entry,
    )

    assert len(dataset) == 2
    assert dataset.sample_ids == ("parent-a", "parent-b")
    assert dataset.window_counts == (2, 1)
    assert dataset.window_frame_lengths == ((2, 4), (4,))
    with pytest.raises(TypeError, match="parent-first sampler"):
        dataset[0]

    item = dataset[(0, 1)]
    assert parent_dataset.assets_only_calls == 1
    assert item["sample_id"] == "parent-a#w1"
    assert item["parent_sample_id"] == "parent-a"
    assert item["latents"].squeeze(-1).tolist() == [2.0, 3.0, 4.0, 5.0]
    assert item["semantic_ids"].tolist() == [2, 3, 4, 5]
    assert item["description_embeddings"].eq(11.0).all()
    assert item["lyrics_embeddings"].eq(22.0).all()
    assert item["provenance"]["crop_start_frame"] == 2
    assert item["provenance"]["crop_end_frame"] == 6


def test_renderer_data_short_dataset_supports_fail_closed_parent_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_dataset = _FakeParentDataset()
    sample_ids = ["parent-b"]
    subset = {
        "schema_version": RENDERER_DATA_PARENT_SUBSET_SCHEMA,
        "source_expected_records": 2,
        "source_manifest_sha256": parent_dataset.manifest_sha256,
        "parent_indices": [1],
        "parent_sample_ids": sample_ids,
        "ordered_parent_sample_ids_sha256": _canonical_sha256(sample_ids),
    }
    dataset = _build_dataset(
        tmp_path,
        parent_dataset,
        parent_subset=subset,
    )
    text_entry = SimpleNamespace(
        description_embeddings=torch.full((2, 3), 11.0),
        description_input_ids=torch.tensor([11, 12]),
        description_mask=torch.ones(2, dtype=torch.bool),
        lyrics_embeddings=torch.full((3, 3), 22.0),
        lyrics_input_ids=torch.tensor([21, 22, 23]),
        lyrics_mask=torch.ones(3, dtype=torch.bool),
    )
    monkeypatch.setattr(
        renderer_data_module,
        "read_text_cache_entry",
        lambda *args, **kwargs: text_entry,
    )

    assert len(dataset) == 1
    assert dataset.sample_ids == ("parent-b",)
    assert dataset.window_counts == (1,)
    item = dataset[(0, 0)]
    assert item["parent_sample_id"] == "parent-b"
    assert item["latents"].squeeze(-1).tolist() == [100.0, 101.0, 102.0, 103.0]

    drifted = dict(subset)
    drifted["parent_sample_ids"] = ["parent-a"]
    drifted["ordered_parent_sample_ids_sha256"] = _canonical_sha256(["parent-a"])
    drifted_root = tmp_path / "drifted"
    drifted_root.mkdir()
    with pytest.raises(RuntimeError, match="index/sample_id"):
        _build_dataset(
            drifted_root,
            parent_dataset,
            parent_subset=drifted,
        )


def test_renderer_data_short_dataset_supports_fail_closed_fixed_window_subset(
    tmp_path: Path,
) -> None:
    parent_dataset = _FakeParentDataset()
    parent_ids = ["parent-a"]
    parent_subset = {
        "schema_version": RENDERER_DATA_PARENT_SUBSET_SCHEMA,
        "source_expected_records": 2,
        "source_manifest_sha256": parent_dataset.manifest_sha256,
        "parent_indices": [0],
        "parent_sample_ids": parent_ids,
        "ordered_parent_sample_ids_sha256": _canonical_sha256(parent_ids),
    }
    ordered = [
        {
            "parent_sample_id": "parent-a",
            "window_index": 0,
            "window_sample_id": "parent-a#w0",
        }
    ]

    source_root = tmp_path / "source"
    source_root.mkdir()
    manifest, manifest_sha, _, _ = _publish_crop_contract(source_root, parent_dataset)
    del manifest
    window_subset = {
        "schema_version": RENDERER_DATA_WINDOW_SUBSET_SCHEMA,
        "source_crop_manifest_sha256": manifest_sha,
        "parent_sample_ids": parent_ids,
        "window_indices": [0],
        "window_sample_ids": ["parent-a#w0"],
        "ordered_parent_windows_sha256": _canonical_sha256(ordered),
    }

    dataset = RendererDataShortWindowDataset(
        parent_dataset,  # type: ignore[arg-type]
        crop_manifest_path=source_root / "renderer_data-crops.jsonl",
        expected_crop_manifest_sha256=manifest_sha,
        crop_ready_path=source_root / "RENDERER_DATA_CROPS_READY.json",
        expected_crop_ready_sha256=_file_sha256(
            source_root / "RENDERER_DATA_CROPS_READY.json"
        ),
        parent_subset=parent_subset,
        window_subset=window_subset,
    )
    assert dataset.sample_ids == ("parent-a",)
    assert dataset.window_counts == (1,)
    assert dataset.window_frame_lengths == ((2,),)
    sampler = RendererDataParentBatchSampler(
        sample_ids=dataset.sample_ids,
        rank=0,
        world_size=1,
        batch_size_per_rank=1,
        gradient_accumulation_steps=1,
        global_parent_batch_size=1,
        seed=7,
        training_config_hash="a" * 64,
        view="short",
        window_counts=dataset.window_counts,
        window_frame_lengths=dataset.window_frame_lengths,
    )
    first_five = []
    for step in range(5):
        if step:
            sampler.set_epoch(step)
        first_five.append(next(iter(sampler))[0])
        sampler.advance()
    assert first_five == [(0, 0)] * 5

    drifted = dict(window_subset)
    drifted["window_sample_ids"] = ["parent-a#w1"]
    with pytest.raises(RuntimeError, match="window subset identity"):
        RendererDataShortWindowDataset(
            parent_dataset,  # type: ignore[arg-type]
            crop_manifest_path=source_root / "renderer_data-crops.jsonl",
            expected_crop_manifest_sha256=manifest_sha,
            crop_ready_path=source_root / "RENDERER_DATA_CROPS_READY.json",
            expected_crop_ready_sha256=_file_sha256(
                source_root / "RENDERER_DATA_CROPS_READY.json"
            ),
            parent_subset=parent_subset,
            window_subset=drifted,
        )


def test_renderer_data_sparse_evaluation_reads_only_frozen_parent_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_dataset = _FakeParentDataset()
    manifest, manifest_sha, ready, ready_sha = _publish_crop_contract(
        tmp_path,
        parent_dataset,
    )
    selection_rows = [{"parent_index": 1, "window_index": 0}]
    selection = {
        "source_expected_records": 2,
        "source_manifest_sha256": parent_dataset.manifest_sha256,
        "source_crop_manifest_sha256": manifest_sha,
        "source_crop_ready_sha256": ready_sha,
        "source_crop_rows_sha256": json.loads(ready.read_text(encoding="utf-8"))[
            "rows_sha256"
        ],
        "rows": selection_rows,
        "rows_sha256": _canonical_sha256(selection_rows),
    }
    original_file_sha256 = renderer_data_module._file_sha256
    hashed_paths: list[Path] = []

    def observe_hash(path: Path) -> str:
        hashed_paths.append(path)
        return original_file_sha256(path)

    monkeypatch.setattr(renderer_data_module, "_file_sha256", observe_hash)
    dataset = RendererDataShortWindowDataset(
        parent_dataset,  # type: ignore[arg-type]
        crop_manifest_path=manifest,
        expected_crop_manifest_sha256=manifest_sha,
        crop_ready_path=ready,
        expected_crop_ready_sha256=ready_sha,
        evaluation_selection=selection,
    )

    assert manifest not in hashed_paths
    assert dataset.sample_ids == ("parent-b",)
    assert dataset.window_counts == (1,)
    assert dataset.parents[0].parent_index == 1
    assert dataset.parents[0].windows[0].window_index == 0


def test_renderer_data_sparse_evaluation_rejects_selection_rows_sha_drift(
    tmp_path: Path,
) -> None:
    parent_dataset = _FakeParentDataset()
    manifest, manifest_sha, ready, ready_sha = _publish_crop_contract(
        tmp_path,
        parent_dataset,
    )
    selection = {
        "source_expected_records": 2,
        "source_manifest_sha256": parent_dataset.manifest_sha256,
        "source_crop_manifest_sha256": manifest_sha,
        "source_crop_ready_sha256": ready_sha,
        "source_crop_rows_sha256": json.loads(ready.read_text(encoding="utf-8"))[
            "rows_sha256"
        ],
        "rows": [{"parent_index": 1, "window_index": 0}],
        "rows_sha256": "f" * 64,
    }
    with pytest.raises(RuntimeError, match="selection rows SHA"):
        RendererDataShortWindowDataset(
            parent_dataset,  # type: ignore[arg-type]
            crop_manifest_path=manifest,
            expected_crop_manifest_sha256=manifest_sha,
            crop_ready_path=ready,
            expected_crop_ready_sha256=ready_sha,
            evaluation_selection=selection,
        )


def test_renderer_data_short_dataset_rejects_crop_gap_or_incomplete_parent(
    tmp_path: Path,
) -> None:
    parent_dataset = _FakeParentDataset()
    windows = [
        _window("parent-a#w0", 0, 0, 2),
        _window("parent-a#w1", 1, 3, 6),
    ]
    with pytest.raises(RuntimeError, match="crop windows must be contiguous"):
        _build_dataset(tmp_path, parent_dataset, first_windows=windows)


def test_renderer_data_short_dataset_rejects_parent_artifact_drift(tmp_path: Path) -> None:
    parent_dataset = _FakeParentDataset()
    contract = _publish_crop_contract(tmp_path, parent_dataset)
    parent_dataset.records[0]["latent"]["sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="parent artifact identity"):
        RendererDataShortWindowDataset(
            parent_dataset,  # type: ignore[arg-type]
            crop_manifest_path=contract[0],
            expected_crop_manifest_sha256=contract[1],
            crop_ready_path=contract[2],
            expected_crop_ready_sha256=contract[3],
        )


def _short_sampler(*, rank: int = 0, seed: int = 17) -> RendererDataParentBatchSampler:
    return RendererDataParentBatchSampler(
        sample_ids=[f"parent-{index}" for index in range(8)],
        rank=rank,
        world_size=2,
        batch_size_per_rank=4,
        gradient_accumulation_steps=1,
        global_parent_batch_size=8,
        seed=seed,
        training_config_hash="training-config-sha",
        view="short",
        window_counts=[2] * 8,
        window_frame_lengths=[(100 + index, 200 + index) for index in range(8)],
    )


def _full_sampler(*, rank: int = 0, seed: int = 23) -> RendererDataParentBatchSampler:
    return RendererDataParentBatchSampler(
        sample_ids=[f"parent-{index}" for index in range(8)],
        rank=rank,
        world_size=4,
        batch_size_per_rank=1,
        gradient_accumulation_steps=2,
        global_parent_batch_size=8,
        seed=seed,
        training_config_hash="training-config-sha",
        view="full",
    )


def test_renderer_data_parent_sampler_does_not_weight_by_window_count_and_rotates() -> None:
    rank0 = _short_sampler(rank=0)
    rank1 = _short_sampler(rank=1)
    first_global = list(iter(rank0))[0] + list(iter(rank1))[0]
    assert len(first_global) == 8
    assert len({value[0] for value in first_global}) == 8
    first_windows = {parent: window for parent, window in first_global}

    rank0.advance()
    rank1.advance()
    assert rank0.window_cursors == [1] * 8
    assert rank0.parent_cycles == 1
    rank0.set_epoch(1)
    rank1.set_epoch(1)
    second_global = list(iter(rank0))[0] + list(iter(rank1))[0]
    second_windows = {parent: window for parent, window in second_global}
    assert set(second_windows) == set(first_windows)
    assert all(
        second_windows[parent] != first for parent, first in first_windows.items()
    )


def test_renderer_data_parent_sampler_exact_resume_restores_mid_update_state() -> None:
    sampler = _full_sampler()
    iterator = iter(sampler)
    first_microstep = next(iterator)
    sampler.advance()
    expected_second_microstep = next(iterator)
    state = sampler.state_dict()

    resumed = _full_sampler()
    resumed.load_state_dict(state)
    assert resumed.cursor == 1
    assert resumed.microstep_cursor == 1
    assert list(iter(resumed))[0] == expected_second_microstep
    assert first_microstep != expected_second_microstep
    assert torch.equal(
        resumed.parent_order_generator.get_state(),
        state["parent_order_rng_state"],
    )

    resumed.advance()
    assert resumed.cursor == resumed.num_batches_per_epoch
    assert resumed.window_cursors == [1] * 8
    assert resumed.parent_cycles == 1


def test_renderer_data_parent_sampler_refuses_topology_drift() -> None:
    state = _short_sampler(seed=17).state_dict()
    with pytest.raises(RuntimeError, match="topology identity does not match"):
        _short_sampler(seed=18).load_state_dict(state)


def test_renderer_data_fixed_validation_maps_each_parent_to_stable_window() -> None:
    source = SimpleNamespace(
        sample_ids=("a", "b"),
        window_frame_lengths=((10, 20), (30,)),
        __len__=lambda: 2,
    )

    class _Source:
        sample_ids = source.sample_ids
        window_frame_lengths = source.window_frame_lengths

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: tuple[int, int]) -> dict[str, Any]:
            return {"selected": index}

    first = RendererDataFixedValidationDataset(_Source(), seed=7)  # type: ignore[arg-type]
    second = RendererDataFixedValidationDataset(_Source(), seed=7)  # type: ignore[arg-type]
    assert first.window_indices == second.window_indices
    assert [first[index] for index in range(2)] == [
        {"selected": (index, first.window_indices[index])} for index in range(2)
    ]


def test_renderer_data_rng_encoding_round_trip() -> None:
    sampler = _short_sampler(seed=17)
    rng_state = sampler.state_dict()["parent_order_rng_state"]
    encoded = encode_renderer_data_rng_state(rng_state)
    assert encoded["num_bytes"] > 0
    assert torch.equal(decode_renderer_data_rng_state(encoded), rng_state)
    assert torch.equal(decode_renderer_data_rng_state(rng_state), rng_state)

    drifted = dict(encoded)
    drifted["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="RNG SHA does not match"):
        decode_renderer_data_rng_state(drifted)
