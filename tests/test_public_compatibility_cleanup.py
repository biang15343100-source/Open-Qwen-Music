from __future__ import annotations

import pytest

from open_qwen_music.render.data import RenderAudioDataset, RenderAudioRecord
from open_qwen_music.tokenizer.trainer import _validate_resume_topology
from scripts.train_acoustic import _validate_config


def _audio_record() -> RenderAudioRecord:
    return RenderAudioRecord(sample_id="sample", audio_path="/tmp/sample.wav")


@pytest.mark.parametrize("mode", ["none", "node_once"])
def test_render_dataset_accepts_public_integrity_modes(mode: str) -> None:
    dataset = RenderAudioDataset([_audio_record()], audio_integrity_mode=mode)
    assert dataset.audio_integrity_mode == mode


def test_render_dataset_rejects_unsupported_integrity_mode() -> None:
    with pytest.raises(ValueError, match="none.*node_once"):
        RenderAudioDataset([_audio_record()], audio_integrity_mode="per_read")


def test_acoustic_config_rejects_unsupported_integrity_mode() -> None:
    config = {
        "stage": 1,
        "component": "spec_vae",
        "loss": {"recipe": "open_qwen_music_vae_stage1_v1"},
        "data": {"audio_integrity_mode": "per_read"},
        "train": {"max_steps": 1},
    }
    with pytest.raises(ValueError, match="audio_integrity_mode"):
        _validate_config(config, smoke=True)


def test_resume_without_topology_is_rejected() -> None:
    config = {
        "data": {},
        "train": {
            "batch_size_per_rank": 1,
            "gradient_accumulation_steps": 1,
        },
    }
    with pytest.raises(RuntimeError, match="missing distributed_state"):
        _validate_resume_topology(config, None)
