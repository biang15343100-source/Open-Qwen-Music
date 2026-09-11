from __future__ import annotations

import pytest
import torch

from open_qwen_music.render.spec_vae import (
    LATENT_DIM,
    SpecVAE,
    SpecVAEConfig,
    audit_spec_vae_parameter_count,
)


def _config() -> SpecVAEConfig:
    return SpecVAEConfig(
        encoder_base_channels=1,
        decoder_base_channels=1,
        weight_norm=False,
    )


def _spectrum(*, batch: int = 1, frames: int = 8) -> torch.Tensor:
    real = torch.randn(batch, 2, 480, frames)
    imaginary = torch.randn(batch, 2, 480, frames)
    return torch.complex(real, imaginary)


def test_config_exposes_only_the_released_architecture() -> None:
    config = SpecVAEConfig.from_dict(
        {
            "latent_dim": 128,
            "posterior_variance_epsilon": 1.0e-6,
            "encoder_base_channels": 64,
            "decoder_base_channels": 128,
            "weight_norm": True,
            "revision": "open-qwen-music-acoustic-vae-v1",
        }
    )
    assert config.latent_dim == LATENT_DIM
    with pytest.raises(ValueError, match="unknown fields"):
        SpecVAEConfig.from_dict({"architecture_profile": "base_v3"})


def test_encode_decode_preserves_contract_and_ragged_masks() -> None:
    model = SpecVAE(_config()).eval()
    spectrum = _spectrum(batch=2)
    lengths = torch.tensor([8, 4])

    output = model(spectrum, lengths, sample_posterior=False)

    assert output.reconstruction.shape == spectrum.shape
    assert output.latents.shape == (2, 2, LATENT_DIM)
    assert output.latent_lengths.tolist() == [2, 1]
    assert not output.latent_mask[1, 1]
    assert torch.count_nonzero(output.latents[1, 1]) == 0
    assert torch.count_nonzero(output.reconstruction[1, :, :, 4:]) == 0
    assert torch.isfinite(output.posterior.kl())


def test_posterior_sampling_is_seeded_and_uses_contiguous_layout() -> None:
    model = SpecVAE(_config()).eval()
    encoded = model.encode(_spectrum())
    first = encoded.posterior.sample(
        torch.Generator().manual_seed(7), contiguous_source_layout=True
    )
    second = encoded.posterior.sample(
        torch.Generator().manual_seed(7), contiguous_source_layout=True
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, encoded.posterior.mode())


def test_parameter_audit_partitions_the_model() -> None:
    audit = audit_spec_vae_parameter_count(SpecVAE(_config()))
    assert audit["revision"] == "open-qwen-music-acoustic-vae-v1"
    assert audit["total"] == audit["encoder"] + audit["decoder"] + audit["other"]
    assert audit["trainable"] == audit["total"]
