from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

pytest.importorskip("torch")

from open_qwen_music.render.materialize import _best_partition, materialize_renderer
from open_qwen_music.render.cache import load_frozen_latent_stats
from open_qwen_music.render.conditioning import TextEncoderProvenance
from open_qwen_music.render.renderer_data import RendererDataShortWindowDataset
from open_qwen_music.render.sample_data import CanonicalRenderSampleDataset
from open_qwen_music.render.text_cache import RenderTextCacheLoader


REV = "a" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class FakeAdapters:
    identity = {
        "tokenizer_revision": REV,
        "vae_revision": REV,
        "text_encoder_revision": REV,
        "rewriter_revision": REV,
        "text_tokenizer_revision": REV,
        "text_cache_revision": REV,
        "latent_cache_revision": REV,
        "latent_stats_sha256": REV,
        "vae_checkpoint_sha256": REV,
        "stft_revision": "open-qwen-music-stft-v1",
        "stft_config_sha256": REV,
        "latent_layout": {
            "format_version": "oqm.render-latent-layout.v1",
            "latent_dim": 128,
            "frame_hz": 25.0,
            "channel_semantics": "unstructured_continuous",
            "special_channels": [],
            "normalization": "per_channel_affine",
        },
        "text_provenance": {
            "model_id": "public-test-encoder",
            "model_revision": REV,
            "tokenizer_revision": REV,
            "cache_revision": REV,
        },
        "text_cache_config": {
            "format_version": "oqm.render-text-cache-config.v1",
            "cache": {"revision": REV, "output_format": "npy", "storage_dtype": "float32"},
            "text_encoder": {
                "model_id": "public-test-encoder", "model_revision": REV,
                "tokenizer_revision": REV, "hidden_size": 4, "local_path": "/fixture",
                "asset_lock": "", "local_files_only": True, "trust_remote_code": False,
                "hidden_state_selection": "last_hidden_state",
                "position_id_policy": "attention_mask_cumsum", "encoder_use_cache": False,
                "empty_text_policy": "zero_valid_tokens", "frozen_eval_mode": True,
            },
            "tokenization": {
                "description_max_tokens": 256, "lyrics_max_tokens": 1536,
                "use_fast_tokenizer": True, "padding": True, "truncation": True,
                "truncation_policy": "reject", "add_special_tokens": True,
                "padding_side": "left", "truncation_side": "right",
            },
            "runtime": {"device": "cpu", "distributed_backend": "gloo", "distributed_timeout_seconds": 300.0},
        },
        "latent_identity": {
            "vae_revision": REV,
            "cache_revision": REV,
            "latent_stats_sha256": REV,
            "posterior_mode": "mean",
        },
    }

    def encode_latent(self, audio, sample_rate, *, sample_id, audio_sha256=None):
        assert sample_rate == 48_000 and audio.shape == (48_000, 2)
        assert audio_sha256
        frames = np.linspace(-1.0, 1.0, 25, dtype=np.float32)[:, None]
        channels = np.linspace(0.5, 1.5, 128, dtype=np.float32)[None, :]
        return frames * channels

    def encode_text(self, text, *, role, max_tokens):
        length = 0 if not text else min(2, max_tokens)
        return {
            "embeddings": np.ones((length, 4), dtype=np.float32),
            "input_ids": np.arange(length, dtype=np.int64),
            "attention_mask": np.ones(length, dtype=np.bool_),
            "original_token_count": length,
            "truncated": False,
        }

    def semantic_codebook(self):
        values = np.zeros((32768, 2), dtype=np.float32)
        values[:, 0] = 1.0
        return values

    def semantic_neighbors(self, top_k):
        assert top_k == 1
        ids = (np.arange(32768, dtype=np.int32)[:, None] + 1) % 32768
        return ids, np.ones((32768, 1), dtype=np.float32)


class FailingAdapters(FakeAdapters):
    def encode_latent(self, audio, sample_rate, *, sample_id, audio_sha256=None):
        raise RuntimeError("fixture encoder failure")


def test_section_partition_is_contiguous_and_prefers_nearby_boundaries() -> None:
    partition = _best_partition(
        (0, 2_000, 4_500),
        frames=4_500,
        target_frames=2_250,
        target_k=2,
    )
    assert partition == (0, 2_000, 4_500)
    assert sum(end - start for start, end in zip(partition, partition[1:])) == 4_500


def test_stage34_mono_audio_materializes_complete_renderer_release(tmp_path: Path) -> None:
    source_audio = tmp_path / "mono-16k.wav"
    sf.write(source_audio, np.linspace(-0.1, 0.1, 16_000, dtype=np.float32), 16_000)
    source_sha = _sha(source_audio)
    source = tmp_path / "stage34.jsonl"
    row = {
        "sample_id": "track-1", "split": "validation",
        "audio": {"path": source_audio.name, "sha256": source_sha},
        "description": "Acoustic song", "tags": {"genre": ["folk"]},
        "text": {"lyrics": "[verse]\nhello"},
        "sections": [{"label": "verse", "lyrics": "hello", "start_sec": 0.0, "end_sec": 1.0}],
        "training": {"sampling_group": "public"},
    }
    train_row = {
        **row,
        "sample_id": "track-2",
        "split": "train",
    }
    source.write_text(
        json.dumps(row) + "\n" + json.dumps(train_row) + "\n",
        encoding="utf-8",
    )
    semantic = tmp_path / "semantics"
    artifact = semantic / "artifacts" / "track.npy"
    artifact.parent.mkdir(parents=True)
    np.save(artifact, np.arange(25, dtype=np.uint16), allow_pickle=False)
    manifest = semantic / "manifest.jsonl"
    semantic_rows = [
        {
            "schema_version": "oqm.renderer-semantic-token.v1",
            "sample_id": sample_id,
            "split": split,
            "source_audio_sha256": source_sha,
            "artifact": {
                "path": "artifacts/track.npy",
                "sha256": _sha(artifact),
                "dtype": "uint16",
                "shape": [25],
                "frame_hz": 25.0,
                "codebook_size": 32768,
            },
        }
        for sample_id, split in (("track-1", "valid"), ("track-2", "train"))
    ]
    manifest.write_text(
        "".join(json.dumps(item) + "\n" for item in semantic_rows),
        encoding="utf-8",
    )
    _json(semantic / "READY", {
        "schema_version": "oqm.renderer-semantic-release.v1", "status": "READY",
        "manifest": "manifest.jsonl", "manifest_sha256": _sha(manifest), "records": 2,
        "tokenizer_revision": REV,
    })
    calibration = tmp_path / "calibration.json"
    _json(
        calibration,
        {
            "schema_version": "oqm.render-semantic-error-calibration.v1",
            "status": "RENDER_SEMANTIC_ERROR_CALIBRATION_READY",
            "teacher_forced": True,
            "semantic_contract": {
                "frame_hz": 25.0,
                "codebook_size": 32768,
                "codebooks": 1,
            },
            "metric": "semantic_teacher_forced_accuracy_at_1",
            "accuracy_at_1": 0.5,
            "negative_log_likelihood_sum_nats": 2.0,
            "cross_entropy_nats": 1.0,
            "correct_tokens": 1,
            "evaluated_tokens": 2,
            "tokenizer_revision": REV,
            "token_registry_sha256": REV,
            "llm_checkpoint_sha256": REV,
            "evaluation_dataset_revision": "public-validation-v1",
            "evaluation_manifest_sha256": REV,
            "evaluation_split": "valid",
        },
    )
    output = tmp_path / "renderer"
    adapters = FakeAdapters()
    adapters.renderer_config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/train/renderer.yaml").read_text()
    )
    materialize_renderer(
        source,
        semantic,
        output,
        adapters=adapters,
        calibration_report=calibration,
        semantic_top_k=1,
    )
    for relative in (
        "READY", "samples/valid/READY", "latents/valid/READY", "crops/valid/READY",
        "loudness/valid/READY", "text/tags/READY", "text/short_lyrics/READY",
        "semantic_embedding/READY", "semantic_corruption/READY",
        "renderer.resolved.yaml",
    ):
        assert (output / relative).is_file(), relative
    release_ready = json.loads((output / "READY").read_text())
    resolved = output / release_ready["resolved_renderer_config"]["path"]
    assert _sha(resolved) == release_ready["resolved_renderer_config"]["sha256"]
    resolved_config = yaml.safe_load(resolved.read_text())
    assert resolved_config["data"]["expected_records"] == 1
    assert resolved_config["validation"]["expected_records"] == 1
    assert resolved_config["semantic_corruption"]["top_k"] == 1
    assert "upstream_revisions" not in resolved_config
    assert "required_upstream_sha_keys" not in resolved_config
    sample = json.loads((output / "samples/valid/manifest.jsonl").read_text())
    canonical = (output / "samples/valid" / sample["audio"]["uri"]).resolve()
    assert canonical.is_relative_to(output)
    waveform, rate = sf.read(canonical, always_2d=True)
    assert rate == 48_000 and waveform.shape == (48_000, 2)

    provenance = TextEncoderProvenance(
        model_id="public-test-encoder",
        model_revision=REV,
        tokenizer_revision=REV,
        cache_revision=REV,
    )
    text_loader_kwargs = {
        "expected_provenance": provenance,
        "expected_cache_config_sha256": FakeAdapters.identity["text_cache_config"][
            "cache"
        ]["revision"],
    }
    tags_ready = json.loads((output / "text/tags/READY").read_text())
    text_loader_kwargs["expected_cache_config_sha256"] = tags_ready[
        "cache_config_sha256"
    ]
    text_loader_kwargs["expected_cache_config_file_sha256"] = tags_ready[
        "cache_config_file_sha256"
    ]
    tags_loader = RenderTextCacheLoader(output / "text/tags", **text_loader_kwargs)
    lyrics_loader = RenderTextCacheLoader(
        output / "text/short_lyrics", **text_loader_kwargs
    )
    sample_root = output / "samples/valid"
    sample_ready_path = sample_root / "READY"
    sample_ready = json.loads(sample_ready_path.read_text())
    parent = CanonicalRenderSampleDataset(
        sample_root / "manifest.jsonl",
        expected_revisions={
            name: resolved_config["revisions"][name]
            for name in (
                "tokenizer_revision",
                "vae_revision",
                "text_encoder_revision",
                "text_tokenizer_revision",
                "text_cache_revision",
                "rewriter_revision",
                "latent_cache_revision",
                "latent_stats_sha256",
            )
        },
        expected_artifacts={"posterior_mode": "mean"},
        text_model_id="public-test-encoder",
        split="valid",
        required_quality_profile="render_broad",
        expected_manifest_sha256=sample_ready["manifest_sha256"],
        expected_records=1,
        ready_path=sample_ready_path,
        expected_ready_sha256=_sha(sample_ready_path),
        expected_release_revision=sample_ready["revision"],
        index_path=sample_root / "manifest.index.json",
        expected_index_sha256=_sha(sample_root / "manifest.index.json"),
        tags_text_cache_loader=tags_loader,
        lyrics_text_cache_loader=lyrics_loader,
        allow_renderer_data_deferred_parent_text=True,
    )
    crop_root = output / "crops/valid"
    loudness_root = output / "loudness/valid"
    windows = RendererDataShortWindowDataset(
        parent,
        crop_manifest_path=crop_root / "manifest.jsonl",
        expected_crop_manifest_sha256=_sha(crop_root / "manifest.jsonl"),
        crop_ready_path=crop_root / "READY",
        expected_crop_ready_sha256=_sha(crop_root / "READY"),
        tags_text_cache_loader=tags_loader,
        lyrics_text_cache_loader=lyrics_loader,
        loudness_manifest_path=loudness_root / "manifest.jsonl",
        expected_loudness_manifest_sha256=_sha(loudness_root / "manifest.jsonl"),
        loudness_ready_path=loudness_root / "READY",
        expected_loudness_ready_sha256=_sha(loudness_root / "READY"),
    )
    item = windows[(0, 0)]
    assert item["latents"].shape == (25, 128)
    assert np.allclose(item["latents"].mean(axis=0), 0.0, atol=1.0e-6)
    assert (output / "latents/stats.json").is_file()
    assert (output / "latents/stats.json.sha256.json").is_file()
    frozen_stats = load_frozen_latent_stats(
        output / "latents/stats.json",
        expected_sha256=resolved_config["revisions"]["latent_stats_sha256"],
        expected_vae_checkpoint_sha256=REV,
        expected_vae_revision=REV,
        expected_stft_revision="open-qwen-music-stft-v1",
        expected_stft_config_sha256=REV,
        expected_posterior_mode="mean",
        expected_manifest_sha256=_sha(source),
        expected_cache_config_sha256=resolved_config["revisions"][
            "latent_cache_revision"
        ],
        expected_posterior_epsilon_draw_layout="contiguous_bdt",
    )
    assert frozen_stats.mean.shape == (128,)
    assert item["semantic_ids"].shape == (25,)
    assert math.isfinite(item["global_loudness_lufs"])
    release_sha = _sha(output / "READY")
    with pytest.raises(FileExistsError):
        materialize_renderer(
            source,
            semantic,
            output,
            adapters=adapters,
            calibration_report=calibration,
            semantic_top_k=1,
        )
    assert _sha(output / "READY") == release_sha

    failed_output = tmp_path / "renderer-failed"
    failed_adapters = FailingAdapters()
    failed_adapters.renderer_config = adapters.renderer_config
    with pytest.raises(RuntimeError, match="fixture encoder failure"):
        materialize_renderer(
            source,
            semantic,
            failed_output,
            adapters=failed_adapters,
            calibration_report=calibration,
            semantic_top_k=1,
        )
    assert not failed_output.exists()
    assert not list(tmp_path.glob(".renderer-failed.staging-*"))
