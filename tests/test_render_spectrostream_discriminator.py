import pytest
import torch

from open_qwen_music.render.discriminators import (
    GlobalLayerNorm2D,
    SpectralDiscriminator,
    SpectroStreamMultiScaleSTFTDiscriminator,
    SpectroStreamPatchDiscriminator2D,
    _stft,
)


def test_global_layer_norm_normalizes_each_record() -> None:
    layer = GlobalLayerNorm2D(3)
    inputs = torch.randn(2, 3, 7, 11)
    output = layer(inputs)
    torch.testing.assert_close(
        output.mean(dim=(1, 2, 3)),
        torch.zeros(2),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output.var(dim=(1, 2, 3), unbiased=False),
        torch.ones(2),
        atol=3.0e-5,
        rtol=0.0,
    )


def test_spectrostream_patch_discriminator_shape_and_stereo_fusion() -> None:
    model = SpectroStreamPatchDiscriminator2D(128, base_channels=2)
    spectrum = torch.randn(3, 2, 128, 17, dtype=torch.complex64)
    logits, features = model(spectrum)
    assert logits.shape == (3, 1, 3, 1)
    assert len(features) == 6
    assert all(value.shape[0] == 3 for value in features)


def test_spectrostream_multiscale_returns_one_branch_per_scale() -> None:
    model = SpectroStreamMultiScaleSTFTDiscriminator(
        scales=((128, 32, 128), (256, 64, 256)),
        base_channels=2,
    )
    audio = torch.randn(2, 2, 2_048)
    output = model(audio)
    assert len(output.logits) == 2
    assert len(output.features) == 2
    assert all(value.shape[0] == 2 for value in output.logits)
    assert all(len(features) == 6 for features in output.features)


def test_spectrostream_band_balanced_features_reuse_existing_weights() -> None:
    control = SpectroStreamMultiScaleSTFTDiscriminator(
        scales=((512, 128, 512),),
        base_channels=2,
    )
    candidate = SpectroStreamMultiScaleSTFTDiscriminator(
        scales=((512, 128, 512),),
        base_channels=2,
        feature_bands_hz=((4_000.0, 8_000.0), (8_000.0, 12_000.0), (12_000.0, 20_000.0)),
    )
    candidate.load_state_dict(control.state_dict(), strict=True)
    audio = torch.randn(2, 2, 2_048)
    control_output = control(audio)
    candidate_output = candidate(audio)
    torch.testing.assert_close(candidate_output.logits[0], control_output.logits[0])
    assert len(control_output.features[0]) == 6
    assert candidate_output.highband_features is not None
    assert len(candidate_output.features[0]) == len(control_output.features[0])

    assert len(candidate_output.highband_features[0]) > 0
    assert sum(p.numel() for p in candidate.parameters()) == sum(
        p.numel() for p in control.parameters()
    )


def test_spectrostream_band_balanced_features_reject_invalid_band() -> None:
    with pytest.raises(ValueError, match="Feature band"):
        SpectroStreamMultiScaleSTFTDiscriminator(
            scales=((128, 64, 128),),
            base_channels=2,
            feature_bands_hz=((12_000.0, 25_000.0),),
        )


def test_spectrostream_public_frontend_uses_no_tail_padding() -> None:
    audio = torch.randn(1, 2, 2_048)
    public = _stft(
        audio,
        128,
        64,
        128,
        frontend_profile="spectrostream_public_v1",
    )
    assert public.shape[-1] == 31


def test_spectrostream_public_frontend_rejects_short_input() -> None:
    with pytest.raises(ValueError, match="does not pad short input"):
        _stft(
            torch.randn(1, 2, 64),
            128,
            64,
            128,
            frontend_profile="spectrostream_public_v1",
        )


def test_source_backed_spectrostream_spectral_discriminator_uses_waveform() -> None:
    model = SpectralDiscriminator(
        stft_scales=((128, 64, 128), (256, 128, 256)),
        stft_frontend_profile="spectrostream_public_v1",
        base_channels=1,
    )
    audio = torch.randn(1, 2, 2_048, requires_grad=True)
    output = model(audio, lengths=torch.tensor([2_048]))
    loss = sum(value.square().mean() for value in output.logits)
    gradient = torch.autograd.grad(loss, audio)[0]
    assert model.input_domain == "waveform"
    assert len(output.logits) == 2
    assert all(len(value) == 6 for value in output.features)
    assert torch.isfinite(gradient).all()
    assert int(torch.count_nonzero(gradient)) > 0
