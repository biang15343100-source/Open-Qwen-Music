import math

import pytest
import torch

from open_qwen_music.render.contracts import SAMPLE_RATE
from open_qwen_music.render.stft import STFTConfig, StereoSTFT


def _stereo_signals(samples: int) -> dict[str, torch.Tensor]:
    time = torch.arange(samples, dtype=torch.float32) / SAMPLE_RATE
    sine = 0.1 * torch.sin(2 * math.pi * 440 * time)
    chirp = 0.08 * torch.sin(
        2 * math.pi * (80 * time + 0.5 * 6_000 * time.square())
    )
    impulse = torch.zeros(samples)
    impulse[samples // 2] = 1
    noise = 0.03 * torch.randn(
        samples, generator=torch.Generator().manual_seed(7)
    )
    return {
        "silence": torch.zeros(2, samples),
        "impulse": torch.stack((impulse, impulse)),
        "sine": torch.stack((sine, 0.7 * sine)),
        "chirp": torch.stack((chirp, torch.roll(chirp, 11))),
        "noise": torch.stack((noise, torch.roll(noise, 5))),
        "left_only": torch.stack((sine, torch.zeros_like(sine))),
        "dual_mono": torch.stack((sine, sine)),
        "antiphase": torch.stack((sine, -sine)),
    }


@pytest.mark.parametrize(
    "samples",
    [1, 479, 480, 481, 1_919, 1_920, 1_921, 3_840, 3_841, 61_440],
)
def test_stft_arbitrary_length_is_exactly_restored(samples: int) -> None:
    transform = StereoSTFT()
    audio = torch.randn(
        1, 2, samples, generator=torch.Generator().manual_seed(samples)
    ) * 0.01
    result = transform.analyze(audio)
    restored = transform.inverse(result)
    assert result.spectrum.dtype == torch.complex64
    assert result.spectrum.shape == (
        1,
        2,
        480,
        4 * math.ceil(samples / 1_920),
    )
    assert restored.shape == audio.shape
    assert torch.isfinite(result.spectrum).all()
    assert torch.isfinite(restored).all()

    assert (restored - audio).abs().mean() < 5e-4


def test_stft_signal_matrix_lr_and_phase() -> None:
    transform = StereoSTFT()
    for name, signal in _stereo_signals(61_440).items():
        result = transform.analyze(signal.unsqueeze(0))
        restored = transform.inverse(result)[0]
        error = (restored - signal).abs()
        assert torch.isfinite(restored).all(), name
        if name == "noise":
            assert error.mean() < 1e-3
        elif name == "impulse":
            assert error.max() < 2e-3
        else:
            assert error.max() < 1e-4
    left_only = _stereo_signals(3_840)["left_only"]
    restored = transform.inverse(transform.analyze(left_only.unsqueeze(0)))[0]
    assert restored[0].abs().max() > 0.05
    assert restored[1].abs().max() == 0


@pytest.mark.parametrize("samples", [1_921, 61_440])
@pytest.mark.parametrize("position_name", ["first", "middle", "last"])
def test_stft_impulse_roundtrip_has_zero_sample_delay(
    samples: int,
    position_name: str,
) -> None:
    transform = StereoSTFT()
    positions = {
        "first": 0,
        "middle": samples // 2,
        "last": samples - 1,
    }
    position = positions[position_name]
    audio = torch.zeros(1, 2, samples)
    audio[:, :, position] = 1.0

    restored = transform.inverse(transform.analyze(audio), dtype=torch.float32)

    assert restored.shape == audio.shape
    assert int(restored[0, 0].abs().argmax()) == position
    assert int(restored[0, 1].abs().argmax()) == position


def test_stft_varying_lengths_mask_and_dtype_do_not_drift() -> None:
    transform = StereoSTFT()
    audio = torch.randn(2, 2, 3_841, dtype=torch.float64)
    lengths = torch.tensor([3_841, 1_920])
    short_audio = audio[1:2, :, :1_920].clone()
    short_spectrum = transform.analyze(short_audio)
    short_restored = transform.inverse(short_spectrum)
    audio[1, :, 1_920:] = 1000.0
    result = transform.analyze(audio, lengths)
    restored = transform.inverse(result)
    assert result.spectrum.dtype == torch.complex64
    assert restored.dtype == torch.float64
    assert restored.shape == audio.shape
    assert torch.count_nonzero(result.spectrum[1, :, :, 4:]) == 0
    assert torch.count_nonzero(restored[1, :, 1_920:]) == 0
    assert torch.equal(result.spectrum[1:2, :, :, :4], short_spectrum.spectrum)
    assert torch.allclose(
        restored[1:2, :, :1_920],
        short_restored,
        atol=1.0e-7,
        rtol=1.0e-7,
    )


def test_nyquist_is_explicitly_dropped_not_aliased_into_bin_479() -> None:
    transform = StereoSTFT()
    samples = 3_840
    nyquist = torch.where(
        torch.arange(samples) % 2 == 0,
        torch.ones(samples),
        -torch.ones(samples),
    ).float()
    result = transform.analyze(torch.stack((nyquist, nyquist)).unsqueeze(0))
    assert result.spectrum.shape[2] == 480

    full = torch.stft(
        torch.nn.functional.pad(nyquist, (240, 240)),
        n_fft=960,
        hop_length=480,
        win_length=960,
        window=torch.hann_window(960),
        center=False,
        return_complex=True,
    )
    assert full.shape[0] == 481
    assert full[-1].abs().max() > 1


def test_drop_dc_keep_nyquist_layout_preserves_480_bin_shape() -> None:
    transform = StereoSTFT(STFTConfig(drop_nyquist=False))
    samples = 3_840
    nyquist = torch.where(
        torch.arange(samples) % 2 == 0,
        torch.ones(samples),
        -torch.ones(samples),
    ).float()
    audio = torch.stack((nyquist, nyquist)).unsqueeze(0)
    result = transform.analyze(audio)
    assert result.spectrum.shape == (1, 2, 480, 8)
    assert result.spectrum[:, :, -1].abs().max() > 1
    restored = transform.inverse(result)
    assert (restored - audio).abs().mean() < 1e-4


def test_explicit_padding_contract() -> None:
    config = STFTConfig()
    assert config.center is False
    assert config.explicit_left_padding == 240
    assert config.right_padding(61_440) == 240
    assert config.frames_for_samples(61_440) == 128


def test_each_four_stft_frames_center_on_one_semantic_frame() -> None:
    config = STFTConfig()
    for latent_index in range(8):
        frame_indices = torch.arange(4 * latent_index, 4 * latent_index + 4)
        centers = (
            frame_indices * config.hop_length
            + config.n_fft // 2
            - config.explicit_left_padding
        )
        expected = (latent_index + 0.5) * 1_920
        assert float(centers.float().mean()) == expected


def test_stft_from_mapping_matches_default_and_rejects_drop_dc() -> None:
    mapping = {
        "n_fft": 960,
        "win_length": 960,
        "hop_length": 480,
        "center": False,
        "normalized": False,
        "drop_nyquist": True,
        "explicit_left_padding": 240,
        "boundary_window_floor": 0.0,
        "revision": "open-qwen-music-stft-v1",
        "window": "hann_periodic",
        "keep_dc": True,
        "bins": 480,
    }
    config = STFTConfig.from_mapping(mapping)
    assert config == STFTConfig()
    dropped = dict(mapping)
    dropped["n_fft"] = 1024
    with pytest.raises(ValueError, match="960/960/480"):
        STFTConfig.from_mapping(dropped)
