from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from open_qwen_music.render.dit import DiTConfig
from open_qwen_music.render.sa3 import (
    SA3_SOURCE_COMMIT,
    SA3RenderDiT,
    build_sa3_render_dit,
)


def _small_model(*, activation_checkpointing: bool = False) -> SA3RenderDiT:
    model_config = DiTConfig(
        hidden_size=64,
        context_dim=64,
        adaln_conditioning="timestep_plus_global_loudness",
        activation_checkpointing=activation_checkpointing,
    )
    native_config = {
        "io_channels": 128,
        "embed_dim": 64,
        "depth": 2,
        "num_heads": 2,
        "cond_token_dim": 64,
        "global_cond_dim": 64,
        "local_add_cond_dim": 64,
        "global_cond_type": "adaLN",
        "timestep_features_type": "expo",
        "diffusion_objective": "rectified_flow",
        "attn_kwargs": {"qk_norm": "rms", "differential": True},
        "norm_type": "rms_norm",
        "norm_kwargs": {"force_fp32": True},
        "ff_kwargs": {"mult": 4.0},
        "num_memory_tokens": 2,
    }
    return SA3RenderDiT(model_config, native_config=native_config)


def _inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(1, 7, 128),
        torch.tensor([0.6]),
        torch.randn(1, 7, 64),
        torch.tensor([[True] * 5 + [False] * 2]),
        torch.randn(1, 6, 64),
        torch.tensor([[False, True, True, False, True, False]]),
        torch.randn(1, 64),
    )


def test_sa3_forward_masks_padding() -> None:
    torch.manual_seed(20260906)
    model = _small_model().eval()
    latent, timestep, semantic, frame_mask, context, text_mask, loudness = _inputs()
    output = model(
        latent,
        timestep,
        semantic,
        frame_mask,
        context,
        text_mask,
        global_loudness_embedding=loudness,
    )
    assert output.shape == latent.shape
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[:, 5:]) == 0


def test_sa3_activation_checkpointing_runs_backward() -> None:
    model = _small_model(activation_checkpointing=True).train()
    model.set_activation_checkpoint_block_interval(2)
    latent, timestep, semantic, frame_mask, context, text_mask, loudness = _inputs()
    latent.requires_grad_()
    model(
        latent,
        timestep,
        semantic,
        frame_mask,
        context,
        text_mask,
        global_loudness_embedding=loudness,
    ).square().mean().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()


def test_sa3_builder_requires_pinned_source_commit() -> None:
    config = DiTConfig(adaln_conditioning="timestep_plus_global_loudness")
    architecture = {
        "type": "stable_audio_3",
        "source_commit": "0" * 40,
        "profile": "stable-audio-3-medium-compatible-v1",
        "native_config": {},
    }
    with pytest.raises(ValueError, match="source commit"):
        build_sa3_render_dit(config, architecture)
    assert SA3_SOURCE_COMMIT == "a0b57f5483c4588f827f3552b7d5c6ca2a9687be"


def test_renderer_recipe_matches_released_architecture() -> None:
    config_path = Path(__file__).parents[1] / "configs/train/renderer.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    architecture = config["architecture"]
    assert architecture["type"] == "stable_audio_3"
    assert architecture["source_commit"] == SA3_SOURCE_COMMIT
    assert architecture["native_config"] == {
        "io_channels": 128,
        "embed_dim": 1536,
        "depth": 24,
        "num_heads": 24,
        "cond_token_dim": 1024,
        "global_cond_dim": 1024,
        "local_add_cond_dim": 1024,
        "global_cond_type": "adaLN",
        "timestep_features_type": "expo",
        "diffusion_objective": "rectified_flow",
        "attn_kwargs": {"qk_norm": "rms", "differential": True},
        "norm_type": "rms_norm",
        "norm_kwargs": {"force_fp32": True},
        "ff_kwargs": {"mult": 4.0},
        "num_memory_tokens": 64,
    }
