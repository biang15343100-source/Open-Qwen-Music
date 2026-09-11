from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from open_qwen_music.common.checkpoint import file_sha256
from open_qwen_music.render.conditioning import (
    FrozenQwenEmbeddingAdapter,
    RenderConditioner,
)
from open_qwen_music.render.dit import DiTConfig
from open_qwen_music.render.flow import FlowConfig
from open_qwen_music.render.sample_data import CanonicalRenderSampleDataset
from open_qwen_music.render.sa3 import SA3RenderDiT
from open_qwen_music.render.text_cache import (
    QWEN_MODEL_ID,
    RenderTextCacheConfig,
    RenderTextCacheLoader,
    RenderTextCondition,
    build_frozen_qwen_adapter,
    cache_condition_record,
)
from open_qwen_music.render.trainer_dit import (
    RenderDiTTrainer,
    RevisionContract,
    collate_cached_render_batch,
)


class FakeTokenizer:
    padding_side = "right"
    truncation_side = "right"

    def __call__(
        self,
        texts,
        *,
        padding,
        truncation,
        max_length,
        return_tensors,
        add_special_tokens=True,
    ):
        sequences = []
        for text in texts:
            values = [index % 31 + 1 for index, _ in enumerate(text)]
            if add_special_tokens:
                values.append(63)
            sequences.append(values[:max_length])
        width = max(len(value) for value in sequences)
        ids = [value + [0] * (width - len(value)) for value in sequences]
        masks = [
            [True] * len(value) + [False] * (width - len(value))
            for value in sequences
        ]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class FakeEncoder(nn.Module):
    hidden_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        values = input_ids.float()
        hidden = torch.stack((values, values + 1, values * 2, -values), dim=-1)
        return SimpleNamespace(last_hidden_state=hidden + self.anchor * 0)


def _text_config() -> RenderTextCacheConfig:
    return RenderTextCacheConfig.from_mapping(
        {
            "format_version": "oqm.render-text-cache-config.v1",
            "cache": {
                "revision": "text-cache-v1",
                "output_format": "npy",
                "storage_dtype": "float32",
            },
            "text_encoder": {
                "model_id": QWEN_MODEL_ID,
                "model_revision": "text-model-v1",
                "tokenizer_revision": "text-tokenizer-v1",
                "hidden_size": 4,
                "local_files_only": True,
                "trust_remote_code": False,
            },
            "tokenization": {
                "description_max_tokens": 256,
                "lyrics_max_tokens": 1536,
                "use_fast_tokenizer": True,
                "padding": True,
                "truncation": True,
                "add_special_tokens": True,
                "padding_side": "right",
                "truncation_side": "right",
            },
            "runtime": {
                "device": "cpu",
                "distributed_backend": "gloo",
                "distributed_timeout_seconds": 30,
            },
        }
    )


def _revisions() -> dict[str, str]:
    return {
        "tokenizer_revision": "semantic-v1",
        "vae_revision": "vae-v1",
        "text_encoder_revision": "text-model-v1",
        "text_tokenizer_revision": "text-tokenizer-v1",
        "text_cache_revision": "text-cache-v1",
        "rewriter_revision": "rewriter-v1",
        "latent_cache_revision": "latent-cache-v1",
        "latent_stats_sha256": "6" * 64,
    }


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _latent_layout() -> dict[str, object]:
    return {
        "format_version": "oqm.render-latent-layout.v1",
        "latent_dim": 128,
        "frame_hz": 25.0,
        "channel_semantics": "unstructured_continuous",
        "special_channels": [],
        "normalization": "per_channel_affine",
    }


def _manifest(
    tmp_path: Path,
    *,
    include_latent_layout: bool = False,
) -> Path:
    audio_path = tmp_path / "audio.bin"
    semantic_path = tmp_path / "semantic.npy"
    latent_path = tmp_path / "latent.npy"
    audio_path.write_bytes(b"canonical-render-audio")
    np.save(semantic_path, np.array([1, 2, 3], dtype=np.uint16))
    np.save(latent_path, np.ones((3, 128), dtype=np.float16))
    audio_sha256 = file_sha256(audio_path)
    source_audio_sha256 = "a" * 64
    config = _text_config()
    adapter = build_frozen_qwen_adapter(
        config,
        encoder=FakeEncoder(),
        tokenizer=FakeTokenizer(),
    )
    condition = RenderTextCondition(
        "sample-1",
        "warm piano with soft strings",
        "sing beneath the evening sky",
        "rewriter-v1",
        "text-tokenizer-v1",
    )
    condition_mapping = {
        "sample_id": condition.sample_id,
        "condition": {
            "description": condition.description,
            "lyrics": condition.lyrics,
            "rewriter_revision": condition.rewriter_revision,
            "text_tokenizer_revision": condition.text_tokenizer_revision,
            "schema_version": condition.schema_version,
        },
    }
    text_record = cache_condition_record(
        condition,
        adapter=adapter,
        config=config,
        output_dir=tmp_path / "text",
        source_index=0,
    )
    row = {
        "schema_version": "oqm.render-sample.v1",
        "sample_id": "sample-1",
        "split": "train",
        "audio": {
            "uri": str(audio_path),
            "sha256": audio_sha256,
            "source_sha256": source_audio_sha256,
            "sample_rate": 48_000,
            "channels": 2,
            "start_sec": 0.0,
            "duration_sec": 3 / 25,
        },
        "semantic": {
            "sample_id": "sample-1",
            "uri": str(semantic_path),
            "sha256": file_sha256(semantic_path),
            "dtype": "uint16",
            "shape": [3],
            "frame_hz": 25.0,
            "codebook_size": 32768,
            "tokenizer_revision": "semantic-v1",
            "input_audio_sha256": audio_sha256,
            "source_audio_sha256": source_audio_sha256,
            "source_start_sec": 0.0,
            "source_duration_sec": 3 / 25,
        },
        "latent": {
            "sample_id": "sample-1",
            "uri": str(latent_path),
            "sha256": file_sha256(latent_path),
            "dtype": "float16",
            "shape": [3, 128],
            "frame_hz": 25.0,
            "vae_revision": "vae-v1",
            "cache_revision": "latent-cache-v1",
            "latent_stats_sha256": "6" * 64,
            "posterior_mode": "mean",
            "derived_audio_sha256": audio_sha256,
            "source_audio_sha256": source_audio_sha256,
            "source_start_sec": 0.0,
            "source_duration_sec": 3 / 25,
        },
        "condition": condition_mapping["condition"],
        "groups": {
            "recording_group_id": "recording-1",
            "performance_group_id": "performance-1",
            "composition_group_id": "composition-1",
            "song_group_id": "song-1",
            "split": "train",
        },
        "text_cache": {
            "cache_config_sha256": config.sha256,
            "cache_config_file_sha256": text_record[
                "cache_config_file_sha256"
            ],
            "record": text_record,
        },
        "quality": {"render_broad": True},
    }
    if include_latent_layout:
        layout = _latent_layout()
        row["latent"]["latent_layout"] = layout
        row["latent"]["latent_layout_sha256"] = _canonical_sha256(layout)
    path = tmp_path / "render-sample.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _published_manifest(
    tmp_path: Path,
) -> tuple[Path, Path, str]:
    manifest = _manifest(tmp_path, include_latent_layout=True)
    layout = _latent_layout()
    layout_sha256 = _canonical_sha256(layout)
    ready = {
        "schema_version": "oqm.render-sample-ready.v1",
        "status": "RENDER_SAMPLE_READY",
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "revision": "render-sample-v1",
        "split": "train",
        "records": 1,
        "checks": {
            "sample_sets_equal": True,
            "source_audio_sha_equal": True,
            "derived_audio_sha_equal": True,
            "time_ranges_equal": True,
            "semantic_latent_frames_equal": True,
            "quality_profile_pass": True,
            "artifact_sha_verified": True,
            "split_groups_disjoint": True,
        },
        "identities": {
            **_revisions(),
            "latent_layout": layout,
            "latent_layout_sha256": layout_sha256,
            "latent_special_channels": [],
        },
    }
    ready_path = tmp_path / "READY"
    ready_path.write_text(
        json.dumps(ready, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest, ready_path, layout_sha256


def test_canonical_render_sample_loads_public_manifest(
    tmp_path: Path,
) -> None:
    dataset = CanonicalRenderSampleDataset(
        _manifest(tmp_path),
        expected_revisions=_revisions(),
        text_model_id=QWEN_MODEL_ID,
    )
    assert dataset.frame_lengths == [3]
    item = dataset[0]
    assert item["latents"].shape == (3, 128)
    assert item["semantic_ids"].tolist() == [1, 2, 3]
    assert item["description_embeddings"].shape[-1] == 4
    batch = collate_cached_render_batch([item])
    assert batch["latents"].shape == (1, 3, 128)
    assert batch["semantic_ids"].shape == (1, 3)
    assert batch["revisions"][0]["latent_stats_sha256"] == "6" * 64


def test_canonical_render_sample_loads_renderer_data_content_addressed_text_pair(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    row = json.loads(manifest.read_text(encoding="utf-8"))
    condition = RenderTextCondition.from_mapping(row)
    config = _text_config()
    adapter = build_frozen_qwen_adapter(
        config,
        encoder=FakeEncoder(),
        tokenizer=FakeTokenizer(),
    )
    tags_sha = hashlib.sha256(condition.description.encode("utf-8")).hexdigest()
    lyrics_sha = hashlib.sha256(condition.lyrics.encode("utf-8")).hexdigest()
    tags_record = cache_condition_record(
        RenderTextCondition(
            f"renderer_data-tags:{tags_sha}",
            condition.description,
            "",
            condition.rewriter_revision,
            condition.text_tokenizer_revision,
        ),
        adapter=adapter,
        config=config,
        output_dir=tmp_path / "tags",
        source_manifest_sha256="b" * 64,
        source_index=0,
    )
    lyrics_record = cache_condition_record(
        RenderTextCondition(
            f"renderer_data-lyrics:{lyrics_sha}",
            "",
            condition.lyrics,
            condition.rewriter_revision,
            condition.text_tokenizer_revision,
        ),
        adapter=adapter,
        config=config,
        output_dir=tmp_path / "lyrics",
        source_manifest_sha256="c" * 64,
        source_index=0,
    )
    row["text_cache"] = {
        "schema_version": "oqm.render.renderer_data-text-cache-reference.v1",
        "tags_content_sha256": tags_sha,
        "lyrics_content_sha256": lyrics_sha,
        "source_condition_sha256": condition.source_condition_sha256,
        "cache_config_sha256": config.sha256,
        "cache_config_file_sha256": config.sha256,
    }
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    tags_loader = RenderTextCacheLoader(
        [tags_record],
        expected_provenance=config.provenance,
        expected_cache_config_sha256=config.sha256,
        expected_cache_config_file_sha256=config.sha256,
    )
    lyrics_loader = RenderTextCacheLoader(
        [lyrics_record],
        expected_provenance=config.provenance,
        expected_cache_config_sha256=config.sha256,
        expected_cache_config_file_sha256=config.sha256,
    )
    dataset = CanonicalRenderSampleDataset(
        manifest,
        expected_revisions=_revisions(),
        text_model_id=QWEN_MODEL_ID,
        tags_text_cache_loader=tags_loader,
        lyrics_text_cache_loader=lyrics_loader,
    )
    item = dataset[0]
    assert item["description_mask"].any()
    assert item["lyrics_mask"].any()
    assert item["description_embeddings"].shape[-1] == 4
    assert item["lyrics_embeddings"].shape[-1] == 4


def test_canonical_renderer_data_full_resolves_deferred_parent_with_new_cache(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    row = json.loads(manifest.read_text(encoding="utf-8"))
    condition = RenderTextCondition.from_mapping(row)
    mapping = _text_config().to_dict()
    mapping["tokenization"]["lyrics_max_tokens"] = 1792
    config = RenderTextCacheConfig.from_mapping(mapping)
    adapter = build_frozen_qwen_adapter(
        config,
        encoder=FakeEncoder(),
        tokenizer=FakeTokenizer(),
    )
    tags_sha = hashlib.sha256(condition.description.encode()).hexdigest()
    lyrics_sha = hashlib.sha256(condition.lyrics.encode()).hexdigest()
    tags_record = cache_condition_record(
        RenderTextCondition(
            f"renderer_data-tags:{tags_sha}",
            condition.description,
            "",
            condition.rewriter_revision,
            condition.text_tokenizer_revision,
        ),
        adapter=adapter,
        config=config,
        output_dir=tmp_path / "full-tags",
        source_manifest_sha256="d" * 64,
        source_index=0,
    )
    lyrics_record = cache_condition_record(
        RenderTextCondition(
            f"renderer_data-lyrics:{lyrics_sha}",
            "",
            condition.lyrics,
            condition.rewriter_revision,
            condition.text_tokenizer_revision,
        ),
        adapter=adapter,
        config=config,
        output_dir=tmp_path / "full-lyrics",
        source_manifest_sha256="e" * 64,
        source_index=0,
    )
    row["text_cache"] = {
        "schema_version": "oqm.render.renderer_data-deferred-parent-text.v1",
        "use_policy": "renderer_data_short_window_only",
        "full_parent_text_ready": False,
        "tags_content_sha256": tags_sha,
        "lyrics_content_sha256": lyrics_sha,
        "source_condition_sha256": condition.source_condition_sha256,


        "cache_config_sha256": "f" * 64,
        "cache_config_file_sha256": "a" * 64,
    }
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    tags_loader = RenderTextCacheLoader(
        [tags_record],
        expected_provenance=config.provenance,
        expected_cache_config_sha256=config.sha256,
        expected_cache_config_file_sha256=config.sha256,
    )
    lyrics_loader = RenderTextCacheLoader(
        [lyrics_record],
        expected_provenance=config.provenance,
        expected_cache_config_sha256=config.sha256,
        expected_cache_config_file_sha256=config.sha256,
    )
    unresolved = CanonicalRenderSampleDataset(
        manifest,
        expected_revisions=_revisions(),
        text_model_id=QWEN_MODEL_ID,
        tags_text_cache_loader=tags_loader,
        lyrics_text_cache_loader=lyrics_loader,
        allow_renderer_data_deferred_parent_text=True,
    )
    with pytest.raises(RuntimeError, match="cannot be read directly in the full view"):
        _ = unresolved[0]

    resolved = CanonicalRenderSampleDataset(
        manifest,
        expected_revisions=_revisions(),
        text_model_id=QWEN_MODEL_ID,
        tags_text_cache_loader=tags_loader,
        lyrics_text_cache_loader=lyrics_loader,
        allow_renderer_data_deferred_parent_text=True,
        resolve_renderer_data_deferred_parent_text=True,
    )
    item = resolved[0]
    assert item["description_mask"].any()
    assert item["lyrics_mask"].any()
    assert item["lyrics_embeddings"].shape[0] == len(condition.lyrics) + 1


def test_canonical_renderer_data_text_loaders_must_be_paired(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="loaders must be provided together"):
        CanonicalRenderSampleDataset(
            _manifest(tmp_path),
            expected_revisions=_revisions(),
            text_model_id=QWEN_MODEL_ID,
            tags_text_cache_loader=object(),  # type: ignore[arg-type]
        )


def test_canonical_float16_batch_runs_dit_training_graph(tmp_path: Path) -> None:
    dataset = CanonicalRenderSampleDataset(
        _manifest(tmp_path),
        expected_revisions=_revisions(),
        text_model_id=QWEN_MODEL_ID,
    )
    batch = collate_cached_render_batch([dataset[0]])
    batch["global_loudness_lufs"] = torch.tensor([-14.0])
    adapter = FrozenQwenEmbeddingAdapter(
        model_id=QWEN_MODEL_ID,
        revision="text-model-v1",
        tokenizer_revision="text-tokenizer-v1",
        cache_revision="text-cache-v1",
        encoder=FakeEncoder(),
        tokenizer=FakeTokenizer(),
        hidden_size=4,
    )
    conditioner = RenderConditioner(
        hidden_size=16,
        text_encoder=adapter,
        text_encoder_dim=4,
        lyrics_num_heads=2,
        lyrics_head_dim=8,
        lyrics_ffn_expansion=2,
        global_loudness_conditioning=True,
    )
    model = SA3RenderDiT(
        DiTConfig(hidden_size=16, context_dim=16, max_frames=16),
        native_config={
            "io_channels": 128,
            "embed_dim": 64,
            "depth": 1,
            "num_heads": 2,
            "cond_token_dim": 16,
            "global_cond_dim": 16,
            "local_add_cond_dim": 16,
            "global_cond_type": "adaLN",
            "timestep_features_type": "expo",
            "diffusion_objective": "rectified_flow",
            "attn_kwargs": {"qk_norm": "rms", "differential": True},
            "norm_type": "rms_norm",
            "norm_kwargs": {"force_fp32": True},
            "ff_kwargs": {"mult": 2.0},
            "num_memory_tokens": 2,
        },
    )
    trainer = RenderDiTTrainer(
        model=model,
        conditioner=conditioner,
        flow_config=FlowConfig(),
        revisions=RevisionContract.from_mapping(_revisions()),
        learning_rate=1.0e-3,
        weight_decay=0.0,
        max_steps=1,
        text_drop_probability=0.0,
        require_cached_text=True,
    )
    output = trainer.compute_loss(batch)
    assert output.prediction.dtype == torch.float32
    assert torch.isfinite(output.loss)


def test_canonical_render_sample_rejects_stats_or_file_drift(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    wrong = _revisions()
    wrong["latent_stats_sha256"] = "7" * 64
    with pytest.raises(RuntimeError, match="revision"):
        CanonicalRenderSampleDataset(
            manifest,
            expected_revisions=wrong,
            text_model_id=QWEN_MODEL_ID,
        )
    dataset = CanonicalRenderSampleDataset(
        manifest,
        expected_revisions=_revisions(),
        text_model_id=QWEN_MODEL_ID,
    )
    latent_path = tmp_path / "latent.npy"
    np.save(latent_path, np.zeros((3, 128), dtype=np.float16))
    with pytest.raises(RuntimeError, match="SHA"):
        dataset[0]


def test_canonical_render_sample_binds_ready_and_record_latent_layout(
    tmp_path: Path,
) -> None:
    manifest, ready_path, layout_sha256 = _published_manifest(tmp_path)
    dataset = CanonicalRenderSampleDataset(
        manifest,
        expected_revisions=_revisions(),
        expected_artifacts={
            "latent_layout_sha256": layout_sha256,
            "latent_special_channels": [],
        },
        text_model_id=QWEN_MODEL_ID,
        ready_path=ready_path,
        expected_ready_sha256=file_sha256(ready_path),
        expected_release_revision="render-sample-v1",
        expected_manifest_sha256=file_sha256(manifest),
        expected_records=1,
    )
    assert dataset.latent_layout == _latent_layout()
    assert dataset[0]["latents"].shape == (3, 128)


def test_canonical_render_sample_separates_ready_and_runtime_text_revision(
    tmp_path: Path,
) -> None:
    manifest, ready_path, _ = _published_manifest(tmp_path)
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    ready["identities"]["text_cache_revision"] = "parent-text-cache-v1"
    ready_path.write_text(
        json.dumps(ready, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime_revisions = {
        **_revisions(),
        "text_cache_revision": "full-parent-text-cache-1792-v1",
    }
    common = {
        "text_model_id": QWEN_MODEL_ID,
        "ready_path": ready_path,
        "expected_ready_sha256": file_sha256(ready_path),
        "expected_release_revision": "render-sample-v1",
        "expected_manifest_sha256": file_sha256(manifest),
        "expected_records": 1,
    }
    with pytest.raises(RuntimeError, match="READY revisions do not match"):
        CanonicalRenderSampleDataset(
            manifest,
            expected_revisions=runtime_revisions,
            **common,
        )


    with pytest.raises(RuntimeError, match="text cache provenance does not match"):
        CanonicalRenderSampleDataset(
            manifest,
            expected_revisions=runtime_revisions,
            expected_ready_text_cache_revision="parent-text-cache-v1",
            **common,
        )


def test_canonical_render_sample_binds_dataset_duration_and_condition_description(
    tmp_path: Path,
) -> None:
    manifest, ready_path, _ = _published_manifest(tmp_path)
    description_contract = "d" * 64
    duration_contract = "oqm.render.duration-buckets.v2-native-360"
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    ready["identities"]["description_contract_sha256"] = description_contract
    ready["identities"]["duration_contract"] = duration_contract
    ready_path.write_text(
        json.dumps(ready, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    dataset = CanonicalRenderSampleDataset(
        manifest,
        expected_revisions=_revisions(),
        expected_artifacts={
            "description_contract_sha256": description_contract,
            "duration_contract": duration_contract,
        },
        text_model_id=QWEN_MODEL_ID,
        ready_path=ready_path,
        expected_ready_sha256=file_sha256(ready_path),
        expected_release_revision="render-sample-v1",
        expected_manifest_sha256=file_sha256(manifest),
        expected_records=1,
    )
    assert dataset[0]["latents"].shape == (3, 128)


def test_canonical_render_sample_rejects_record_layout_different_from_ready(
    tmp_path: Path,
) -> None:
    manifest, ready_path, layout_sha256 = _published_manifest(tmp_path)
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row_layout = {
        **_latent_layout(),
        "channel_semantics": "explicit_special_channels",
        "special_channels": [
            {"index": 0, "role": "amplitude_sidechannel"}
        ],
    }
    row["latent"]["latent_layout"] = row_layout
    row["latent"]["latent_layout_sha256"] = _canonical_sha256(row_layout)
    manifest.write_text(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    ready["manifest_sha256"] = file_sha256(manifest)
    ready_path.write_text(
        json.dumps(ready, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="latent layout does not match READY"):
        CanonicalRenderSampleDataset(
            manifest,
            expected_revisions=_revisions(),
            expected_artifacts={
                "latent_layout_sha256": layout_sha256,
                "latent_special_channels": [],
            },
            text_model_id=QWEN_MODEL_ID,
            ready_path=ready_path,
            expected_ready_sha256=file_sha256(ready_path),
            expected_release_revision="render-sample-v1",
            expected_manifest_sha256=file_sha256(manifest),
            expected_records=1,
        )


def test_canonical_render_sample_rejects_record_layout_sha_drift(
    tmp_path: Path,
) -> None:
    manifest, ready_path, layout_sha256 = _published_manifest(tmp_path)
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row["latent"]["latent_layout_sha256"] = "f" * 64
    manifest.write_text(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    ready["manifest_sha256"] = file_sha256(manifest)
    ready_path.write_text(
        json.dumps(ready, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="latent_layout SHA does not match"):
        CanonicalRenderSampleDataset(
            manifest,
            expected_revisions=_revisions(),
            expected_artifacts={
                "latent_layout_sha256": layout_sha256,
                "latent_special_channels": [],
            },
            text_model_id=QWEN_MODEL_ID,
            ready_path=ready_path,
            expected_ready_sha256=file_sha256(ready_path),
            expected_release_revision="render-sample-v1",
            expected_manifest_sha256=file_sha256(manifest),
            expected_records=1,
        )
