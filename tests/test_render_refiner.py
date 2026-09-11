from __future__ import annotations

import pytest
import torch

from open_qwen_music.render.refiner import (
    EarVAE2PublicRefiner,
    EarVAE2PublicRefinerConfig,
    build_refiner_from_mapping,
    inverse_refiner_spectrum,
    refiner_spectrum_for_loss,
)
from open_qwen_music.render.stft import STFTConfig, StereoSTFT


def _config() -> EarVAE2PublicRefinerConfig:
    return EarVAE2PublicRefinerConfig(
        width=4,
        intermediate_dim=8,
        depth=1,
        kernel_size=3,
    )


def _spectrum(frames: int = 3) -> torch.Tensor:
    real = torch.randn(1, 2, 480, frames)
    imaginary = torch.randn(1, 2, 480, frames)
    return torch.complex(real, imaginary)


def test_mapping_exposes_only_the_released_refiner() -> None:
    model = build_refiner_from_mapping(
        {
            "width": 4,
            "intermediate_dim": 8,
            "depth": 1,
            "kernel_size": 3,
        }
    )
    assert isinstance(model, EarVAE2PublicRefiner)
    with pytest.raises(ValueError, match="unknown fields"):
        build_refiner_from_mapping({"architecture_profile": "band_mode_v1"})


def test_zero_initialized_refiner_preserves_the_observed_spectrum() -> None:
    model = EarVAE2PublicRefiner(_config()).eval()
    coarse = _spectrum()
    refined = model(coarse)
    assert refined.shape == (1, 2, 481, 3)
    assert torch.allclose(refiner_spectrum_for_loss(refined), coarse, atol=1.0e-6)
    assert torch.count_nonzero(refined[:, :, -1]) == 0


def test_only_middle_magnitude_path_is_trainable() -> None:
    model = EarVAE2PublicRefiner(_config())
    assert model.magnitude_mask.sum() == model.split_high - model.split_low
    assert torch.count_nonzero(model.phase_mask) == 0
    assert all(parameter.requires_grad for parameter in model.middle_magnitude_head.parameters())
    assert all(not parameter.requires_grad for parameter in model.low_phase_head.parameters())
    assert all(not parameter.requires_grad for parameter in model.middle_phase_head.parameters())
    assert all(not parameter.requires_grad for parameter in model.high_magnitude_head.parameters())
    assert all(not parameter.requires_grad for parameter in model.nyquist_head.parameters())


def test_inverse_accepts_refiner_nyquist_bin() -> None:
    stft = StereoSTFT(STFTConfig())
    spectrum = EarVAE2PublicRefiner(_config()).eval()(_spectrum(frames=4))
    audio = inverse_refiner_spectrum(stft, spectrum, torch.tensor([960]))
    assert audio.shape == (1, 2, 960)
    assert torch.isfinite(audio).all()
