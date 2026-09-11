from __future__ import annotations

import pytest
import torch
from torch import nn

from open_qwen_music.render.discriminators import DiscriminatorOutput
from open_qwen_music.render.losses import (
    feature_matching_loss,
    lsgan_discriminator_loss,
    lsgan_generator_loss,
)
from open_qwen_music.render.trainer_vae import (
    _split_discriminator_output,
    discriminator_step,
    generator_adversarial_terms,
)
from scripts.train_acoustic import _waveform_discriminator_kwargs


def test_family_weighted_sum_applies_to_d_g_and_fm() -> None:

    per_scale = torch.tensor([1.0, 3.0, 4.0], dtype=torch.float64)
    expected = torch.tensor(2.0 + 0.375 * 4.0, dtype=torch.float64)
    family_sizes = (2, 1)
    family_weights = (1.0, 0.375)

    generator = lsgan_generator_loss(
        [1.0 - value.sqrt() for value in per_scale],
        family_sizes=family_sizes,
        family_weights=family_weights,
        compute_dtype=torch.float64,
    )
    discriminator = lsgan_discriminator_loss(
        [torch.ones(()) for _ in per_scale],
        [(2.0 * value).sqrt() for value in per_scale],
        family_sizes=family_sizes,
        family_weights=family_weights,
        compute_dtype=torch.float64,
    )
    feature_matching = feature_matching_loss(
        [[torch.zeros(())] for _ in per_scale],
        [[value] for value in per_scale],
        family_sizes=family_sizes,
        family_weights=family_weights,
        compute_dtype=torch.float64,
    )

    torch.testing.assert_close(generator, expected)
    torch.testing.assert_close(discriminator, expected)
    torch.testing.assert_close(feature_matching, expected)


    equal = lsgan_generator_loss(
        [1.0 - value.sqrt() for value in per_scale],
        family_sizes=family_sizes,
        compute_dtype=torch.float64,
    )
    flat = lsgan_generator_loss(
        [1.0 - value.sqrt() for value in per_scale],
        compute_dtype=torch.float64,
    )
    assert equal == pytest.approx(3.0)
    assert flat == pytest.approx(8.0 / 3.0)


class _WeightedToyDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, waveform: torch.Tensor) -> DiscriminatorOutput:
        base = waveform.mean(dim=-1, keepdim=True) * self.scale
        return DiscriminatorOutput(
            logits=[base, base * 2.0, base * 4.0],
            features=[[base], [base * 2.0], [base * 4.0]],
            family_names=("stft", "cqt"),
            family_sizes=(2, 1),
            family_weights=(1.0, 0.375),
            loss_reduction="family_weighted_sum_v1",
            loss_compute_dtype="float64",
            family_diagnostics_enabled=False,
        )


def test_trainer_uses_weighted_sum_for_d_g_and_fm() -> None:
    model = _WeightedToyDiscriminator()
    real = torch.ones(1, 2, 8)
    fake = torch.zeros(1, 2, 8, requires_grad=True)

    generator = generator_adversarial_terms(model, real, fake)
    assert generator["adversarial"].detach().item() == pytest.approx(1.375)
    assert generator["feature_matching"].detach().item() == pytest.approx(3.0)
    (generator["adversarial"] + generator["feature_matching"]).backward()
    assert fake.grad is not None
    assert torch.isfinite(fake.grad).all()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    discriminator = discriminator_step(model, optimizer, real, fake)
    assert discriminator["discriminator"] == pytest.approx(1.9375)


def test_family_weights_survive_detach_and_real_fake_split() -> None:
    base = torch.arange(4.0, requires_grad=True).reshape(4, 1)
    output = DiscriminatorOutput(
        logits=[base, base + 1.0],
        features=[[base], [base + 1.0]],
        family_names=("stft", "cqt"),
        family_sizes=(1, 1),
        family_weights=(1.0, 0.375),
        loss_reduction="family_weighted_sum_v1",
        loss_compute_dtype="float64",
    )

    detached = output.detached()
    real, fake = _split_discriminator_output(output, batch_size=2)
    assert detached.family_weights == (1.0, 0.375)
    assert all(not value.requires_grad for value in detached.logits)
    assert real.family_weights == (1.0, 0.375)
    assert fake.family_weights == (1.0, 0.375)
    assert real.loss_reduction == fake.loss_reduction == "family_weighted_sum_v1"


@pytest.mark.parametrize(
    ("cqt_enabled", "weights"),
    [
        (False, [1.0]),
        (True, [1.0, 0.375]),
    ],
)
def test_yaml_loss_family_weights_are_mapped_to_constructor(
    cqt_enabled: bool,
    weights: list[float],
) -> None:
    config = {
        "discriminator": {
            "cqt_enabled": cqt_enabled,
            "stft_scales": [[128, 32, 128]],
            "cqt_scales": [],
            "base_channels": 2,
            "depth": 1,
            "loss_reduction": "family_weighted_sum_v1",
            "loss_family_weights": weights,
        },
        "smoke": {},
    }

    kwargs = _waveform_discriminator_kwargs(config, smoke_model=False)
    assert kwargs["loss_reduction"] == "family_weighted_sum_v1"
    assert kwargs["family_weights"] == weights
