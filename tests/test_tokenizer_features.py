
from __future__ import annotations

import math

import pytest
import torch

from open_qwen_music.tokenizer.audio import (
    RESAMPLE_FILTER_WIDTH,
    RESAMPLE_KAISER_BETA,
    RESAMPLE_ROLLOFF,
    resample_mono,
)
from open_qwen_music.tokenizer.bestrq import BestRQTarget
from open_qwen_music.tokenizer.features import (
    LogMelFrontend,
    _mel_filter_cpu,
    chroma_from_waveform,
)
from open_qwen_music.tokenizer.heads import spectral_loss

SAMPLE_RATE = 24_000
N_FFT = 1024
HOP = 240
N_MELS = 128


def _tone(frequency: float, seconds: float = 2.0, amplitude: float = 0.5) -> torch.Tensor:
    time = torch.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * torch.sin(2 * math.pi * frequency * time)).unsqueeze(0)


def _music_like(batch: int = 2, seconds: float = 4.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    time = torch.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    waveform = torch.zeros(batch, time.numel())
    for index in range(batch):
        for _ in range(5):
            f0 = float(torch.empty(1).uniform_(80, 800, generator=generator))
            for harmonic in range(1, 6):
                waveform[index] += (0.3 / harmonic) * torch.sin(
                    2 * math.pi * f0 * harmonic * time
                )
        waveform[index] += 0.01 * torch.randn(time.numel(), generator=generator)
    return waveform / waveform.abs().amax() * 0.7


@pytest.mark.parametrize("norm", ["slaney", "peak"])
def test_mel_filter_has_no_dead_channel(norm: str) -> None:
    bank = _mel_filter_cpu(SAMPLE_RATE, N_FFT, N_MELS, 0.0, 12_000.0, norm)
    row_sum = bank.sum(dim=1)
    assert bank.shape == (N_MELS, N_FFT // 2 + 1)
    assert int((row_sum == 0).sum()) == 0, f"{norm} A constant zero channel appears"


def test_slaney_normalization_flattens_row_sum() -> None:
    slaney = _mel_filter_cpu(SAMPLE_RATE, N_FFT, N_MELS, 0.0, 12_000.0, "slaney")
    slaney_ratio = slaney.sum(dim=1).max() / slaney.sum(dim=1).min()
    assert slaney_ratio < 2.0


def test_unknown_mel_filter_norm_rejected() -> None:
    with pytest.raises(ValueError, match="mel_filter_norm"):
        _mel_filter_cpu(SAMPLE_RATE, N_FFT, N_MELS, 0.0, 12_000.0, "bogus")


def test_filter_bank_buffer_is_not_the_lru_cache_object() -> None:

    frontend = LogMelFrontend()
    cached = _mel_filter_cpu(SAMPLE_RATE, N_FFT, N_MELS, 0.0, 12_000.0, "slaney")
    assert frontend.mel_filter is not cached
    assert torch.equal(frontend.mel_filter, cached)
    assert "mel_filter" not in frontend.state_dict()


def test_frontend_is_causal() -> None:

    frontend = LogMelFrontend()
    generator = torch.Generator().manual_seed(0)
    waveform = torch.randn(1, SAMPLE_RATE, generator=generator)
    perturbed = waveform.clone()
    perturbed[:, 12_000:] += 1.0
    with torch.no_grad():
        delta = (frontend(waveform) - frontend(perturbed)).abs().sum(dim=-1)[0]
    changed = (delta > 1e-6).nonzero().flatten()
    assert int(changed[0]) == 12_000 // HOP


@pytest.mark.parametrize(
    ("num_samples", "expected_frames"),
    [(24_000, 100), (24_001, 100), (24_239, 100), (24_240, 101)],
)
def test_frame_count_matches_lengths(num_samples: int, expected_frames: int) -> None:
    frontend = LogMelFrontend()
    with torch.no_grad():
        frames = frontend(torch.zeros(1, num_samples)).shape[1]
    reported = int(frontend.lengths(torch.tensor([num_samples]))[0])
    assert frames == expected_frames
    assert reported == expected_frames


def test_per_channel_stats_are_accepted_and_applied() -> None:
    mean = torch.arange(N_MELS, dtype=torch.float32)
    std = torch.full((N_MELS,), 2.0)
    frontend = LogMelFrontend(mean=mean.tolist(), std=std.tolist())
    assert frontend.feature_mean.shape == (N_MELS,)
    baseline = LogMelFrontend(mean=0.0, std=1.0)
    waveform = _music_like(batch=1)
    with torch.no_grad():
        expected = (baseline(waveform) - mean) / std
        assert torch.allclose(frontend(waveform), expected, atol=1e-5)


def test_scalar_stats_stay_zero_dimensional() -> None:
    frontend = LogMelFrontend(mean=-3.4, std=5.2)
    assert frontend.feature_mean.shape == ()
    assert "feature_mean" in frontend.state_dict()


def test_degenerate_std_does_not_amplify_float_noise() -> None:

    std = [1.0] * N_MELS
    mean = [0.0] * N_MELS
    for channel in (2, 6, 13):
        std[channel] = 0.0
    frontend = LogMelFrontend(mel_filter_norm="slaney", mean=mean, std=std)
    assert torch.equal(frontend.feature_std[torch.tensor([2, 6, 13])], torch.ones(3))

    with torch.no_grad():
        features = LogMelFrontend(
            mel_filter_norm="slaney",
            mean=[0.0] * N_MELS,
            std=[0.0] * N_MELS,
        )(_music_like(batch=1))
    assert torch.isfinite(features).all()


def test_wrong_length_stats_rejected() -> None:
    with pytest.raises(ValueError, match="features.mean"):
        LogMelFrontend(mean=[0.0, 1.0, 2.0])


def test_per_channel_cmvn_raises_bestrq_target_diversity() -> None:

    waveform = _music_like(batch=4, seconds=6.0)
    with torch.no_grad():
        features = LogMelFrontend(mean=-3.396, std=5.210)(waveform)
    flat = features.reshape(-1, N_MELS)
    per_channel = (features - flat.mean(0)) / flat.std(0).clamp_min(1e-6)

    def distinct_codes(x: torch.Tensor) -> int:
        target = BestRQTarget(
            input_dim=N_MELS,
            projection_dim=16,
            codebook_size=8192,
            local_window=3,
            seed=20260719,
        )
        return int(target(x).unique().numel())

    assert distinct_codes(per_channel) > distinct_codes(features)


def test_bestrq_target_entropy_stays_above_floor() -> None:

    waveform = _music_like(batch=8, seconds=10.0)
    frontend = LogMelFrontend(mel_filter_norm="slaney", log_mode="additive", log_floor=1e-9)
    with torch.no_grad():
        features = frontend(waveform)
    flat = features.reshape(-1, N_MELS)
    normalized = (features - flat.mean(0)) / flat.std(0).clamp_min(1e-6)
    codes = BestRQTarget(
        input_dim=N_MELS,
        projection_dim=16,
        codebook_size=8192,
        local_window=3,
        seed=20260719,
    )(normalized)

    counts = torch.bincount(codes.flatten())
    probabilities = counts[counts > 0].double() / counts.sum()
    entropy = float(-(probabilities * probabilities.log()).sum())
    used = int((counts > 0).sum())


    assert entropy > 4.2, f"BestRQ target entropy collapse to {entropy:.3f} nats"
    assert used > 280, f"BestRQ only uses {used} code words"


def test_mel_filter_matches_librosa() -> None:

    librosa = pytest.importorskip("librosa", reason="librosa Non-training environment dependency")
    reference = torch.from_numpy(
        librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            n_mels=N_MELS,
            fmin=0.0,
            fmax=12_000.0,
            htk=True,
            norm="slaney",
        )
    ).float()
    ours = _mel_filter_cpu(SAMPLE_RATE, N_FFT, N_MELS, 0.0, 12_000.0, "slaney")
    assert torch.allclose(ours, reference, atol=1e-6)


def test_additive_log_leaves_no_pinned_channel() -> None:


    waveform = _music_like(batch=2, seconds=4.0) * 0.02
    clamped = LogMelFrontend(log_mode="clamp", log_floor=1e-5)
    additive = LogMelFrontend(log_mode="additive", log_floor=1e-9)
    with torch.no_grad():
        clamped_features = clamped(waveform).reshape(-1, N_MELS)
        additive_features = additive(waveform).reshape(-1, N_MELS)

    floor = math.log(1e-5)
    pinned = (clamped_features <= floor + 1e-6).float().mean(dim=0)
    assert float(pinned.max()) > 0.3, "The constructed signal did not trigger clamp,The test is meaningless"
    assert float(additive_features.std(dim=0).min()) > 1e-3, "Additive log "


def test_frame_centered_db_is_gain_invariant() -> None:

    generator = torch.Generator().manual_seed(20260807)
    waveform = torch.randn(2, 48_000, generator=generator) * 0.1
    frontend = LogMelFrontend(
        n_fft=2048,
        win_length=2048,
        mel_scale="slaney",
        mel_filter_norm="slaney",
        log_mode="db",
        log_floor=1e-12,
        feature_centering="frame",
    )
    with torch.no_grad():
        baseline = frontend(waveform)
        quieter = frontend(waveform * 0.25)
        louder = frontend(waveform * 4.0)
    torch.testing.assert_close(baseline, quieter, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(baseline, louder, rtol=1e-5, atol=1e-5)


def test_clamp_log_mode_matches_direct_computation() -> None:

    waveform = _music_like(batch=1)
    frontend = LogMelFrontend(mel_filter_norm="slaney", log_mode="clamp", log_floor=1e-5)
    with torch.no_grad():
        actual = frontend(waveform)
        bank = _mel_filter_cpu(SAMPLE_RATE, N_FFT, N_MELS, 0.0, 12_000.0, "slaney")
        power = frontend._spectrogram(waveform)
        expected = torch.log(
            torch.einsum("mf,bft->btm", bank, power).clamp_min(1e-5)
        )
    assert torch.equal(actual, expected)


def test_unknown_log_mode_rejected() -> None:
    with pytest.raises(ValueError, match="log_mode"):
        LogMelFrontend(log_mode="bogus")


def test_resample_suppresses_aliasing() -> None:

    source_rate = 44_100
    time = torch.arange(source_rate) / source_rate
    for frequency in (16_000.0, 18_000.0, 20_000.0):
        tone = torch.sin(2 * math.pi * frequency * time)
        resampled = resample_mono(tone, source_rate, SAMPLE_RATE)
        magnitude = torch.fft.rfft(resampled * torch.hann_window(resampled.numel())).abs()
        decibels = 20 * math.log10(float(magnitude.max()) / (resampled.numel() / 4) + 1e-20)
        assert decibels < -40.0, f"{frequency:.0f} Hz Aliasing residue {decibels:.1f} dB"


def test_resample_preserves_passband() -> None:
    source_rate = 44_100
    time = torch.arange(source_rate) / source_rate
    for frequency in (440.0, 4_000.0, 8_000.0):
        tone = torch.sin(2 * math.pi * frequency * time)
        resampled = resample_mono(tone, source_rate, SAMPLE_RATE)
        magnitude = torch.fft.rfft(resampled * torch.hann_window(resampled.numel())).abs()
        peak_hz = float(
            torch.fft.rfftfreq(resampled.numel(), 1.0 / SAMPLE_RATE)[int(magnitude.argmax())]
        )
        decibels = 20 * math.log10(float(magnitude.max()) / (resampled.numel() / 4) + 1e-20)
        assert abs(peak_hz - frequency) < 2.0
        assert decibels > -1.0, f"{frequency:.0f} Hz Passband attenuation {decibels:.1f} dB"


def test_resample_matches_torchaudio() -> None:

    torchaudio = pytest.importorskip("torchaudio")
    generator = torch.Generator().manual_seed(3)
    waveform = torch.randn(44_100, generator=generator)
    for source_rate in (44_100, 48_000, 16_000):
        ours = resample_mono(waveform, source_rate, SAMPLE_RATE)
        reference = torchaudio.functional.resample(
            waveform,
            source_rate,
            SAMPLE_RATE,
            lowpass_filter_width=RESAMPLE_FILTER_WIDTH,
            rolloff=RESAMPLE_ROLLOFF,
            resampling_method="sinc_interp_kaiser",
            beta=RESAMPLE_KAISER_BETA,
        )
        assert ours.shape == reference.shape
        assert torch.allclose(ours, reference, atol=1e-5), f"{source_rate} Hz and torchaudio inconsistent"


@pytest.mark.parametrize("source_rate", [44_100, 48_000, 22_050, 16_000, 8_000])
def test_resample_output_length(source_rate: int) -> None:
    waveform = torch.zeros(source_rate * 3)
    resampled = resample_mono(waveform, source_rate, SAMPLE_RATE)
    assert resampled.numel() == SAMPLE_RATE * 3


def test_resample_is_identity_at_same_rate() -> None:
    waveform = torch.randn(1024)
    assert torch.equal(resample_mono(waveform, SAMPLE_RATE, SAMPLE_RATE), waveform)


def test_resample_downmixes_stereo() -> None:
    stereo = torch.stack([torch.ones(2048), torch.full((2048,), 3.0)])
    assert torch.allclose(
        resample_mono(stereo, SAMPLE_RATE, SAMPLE_RATE), torch.full((2048,), 2.0)
    )


def test_frontend_stays_fp32_under_autocast() -> None:

    frontend = LogMelFrontend(mean=-3.396, std=5.210)
    waveform = _music_like(batch=1)
    with torch.no_grad():
        reference = frontend(waveform)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with torch.no_grad():
            under_autocast = frontend(waveform)
    assert under_autocast.dtype == torch.float32
    assert torch.equal(under_autocast, reference)


def test_bestrq_target_is_bitwise_stable_under_autocast() -> None:
    frontend = LogMelFrontend(mean=-3.396, std=5.210)
    waveform = _music_like(batch=2)
    target = BestRQTarget(
        input_dim=N_MELS,
        projection_dim=16,
        codebook_size=8192,
        local_window=3,
        seed=20260719,
    )
    with torch.no_grad():
        reference = target(frontend(waveform))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with torch.no_grad():
            observed = target(frontend(waveform))
    assert torch.equal(observed, reference)


def test_short_input_reports_clear_error() -> None:
    frontend = LogMelFrontend()
    with pytest.raises(ValueError, match="hop_length"):
        frontend(torch.zeros(1, HOP - 1))


def test_one_dimensional_input_reports_clear_error() -> None:
    frontend = LogMelFrontend()
    with pytest.raises(ValueError, match="two-dimensional"):
        frontend(torch.zeros(SAMPLE_RATE))


def test_exactly_one_frame_is_accepted() -> None:
    frontend = LogMelFrontend()
    with torch.no_grad():
        assert frontend(torch.zeros(1, HOP)).shape[1] == 1


def test_power_mel_is_nonnegative_and_frame_aligned() -> None:
    frontend = LogMelFrontend()
    waveform = _music_like(batch=2, seconds=1.0)
    power = frontend.power_mel(waveform)
    log_mel = frontend(waveform)
    assert power.shape == log_mel.shape
    assert bool((power >= 0).all())


# --- Chroma -------------------------------------------------------------------


def test_chroma_resolves_pitch_class_above_400hz() -> None:

    for frequency, pitch_class in ((440.0, 9), (523.25, 0), (659.26, 4)):
        chroma = chroma_from_waveform(_tone(frequency))
        assert int(chroma[0, 100].argmax()) == pitch_class


def test_chroma_frames_are_normalized() -> None:
    chroma = chroma_from_waveform(_tone(440.0))
    assert torch.allclose(chroma[0, 100].sum(), torch.tensor(1.0), atol=1e-5)


def test_silent_clip_yields_zero_chroma_target() -> None:

    chroma = chroma_from_waveform(torch.zeros(1, 2 * SAMPLE_RATE))
    assert float(chroma.abs().sum()) == 0.0


def test_spectral_loss_ignores_all_zero_target() -> None:
    chroma = chroma_from_waveform(torch.zeros(1, 2 * SAMPLE_RATE))
    mask = torch.ones(1, chroma.shape[1], dtype=torch.bool)
    prediction = torch.randn_like(chroma) * 0.1
    loss, convergence, _ = spectral_loss(prediction, chroma, mask)
    assert float(loss) == 0.0
    assert float(convergence) == 0.0


def test_spectral_loss_keeps_normal_sample_when_batch_has_silence() -> None:

    voiced = chroma_from_waveform(_tone(440.0))
    silent = torch.zeros_like(voiced)
    mask = torch.ones(1, voiced.shape[1], dtype=torch.bool)
    prediction = torch.randn_like(voiced) * 0.1

    alone, _, _ = spectral_loss(prediction, voiced, mask)
    mixed, _, _ = spectral_loss(
        torch.cat([prediction, prediction]),
        torch.cat([voiced, silent]),
        torch.cat([mask, mask]),
    )
    assert torch.allclose(alone, mixed, atol=1e-5)
    assert float(mixed) < 100.0


def test_nonnegative_spectral_loss_uses_log_magnitude() -> None:
    target = torch.tensor([[[1.0, 4.0]]])
    prediction = torch.tensor([[[2.0, 8.0]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    loss, convergence, magnitude = spectral_loss(
        prediction, target, mask, mode="nonnegative_log"
    )
    expected_magnitude = math.log(2.0)
    assert float(magnitude) == pytest.approx(expected_magnitude, rel=1e-5)
    assert float(loss) == pytest.approx(
        float(convergence) + expected_magnitude, rel=1e-5
    )


def test_nonnegative_spectral_loss_rejects_signed_target() -> None:
    with pytest.raises(ValueError, match="negative target"):
        spectral_loss(
            torch.ones(1, 1, 2),
            torch.tensor([[[-1.0, 1.0]]]),
            torch.ones(1, 1, dtype=torch.bool),
            mode="nonnegative_log",
        )


def test_soft_chroma_resolves_bass_register() -> None:

    def harmonic_tone(midi: float) -> torch.Tensor:
        frequency = 440.0 * 2 ** ((midi - 69) / 12)
        t = torch.arange(SAMPLE_RATE) / SAMPLE_RATE
        wave = sum(
            amplitude * torch.sin(2 * torch.pi * frequency * k * t)
            for k, amplitude in enumerate((1.0, 0.5, 0.3), start=1)
        )
        return (wave / wave.abs().max() * 0.5).unsqueeze(0)

    soft_correct = hard_correct = 0
    for midi in range(24, 85):  # C1..C6
        expected = midi % 12
        tone = harmonic_tone(midi)
        soft = chroma_from_waveform(tone, n_fft=4096, mode="soft").mean(dim=1)
        hard = chroma_from_waveform(tone, n_fft=1024, mode="hard").mean(dim=1)
        soft_correct += int(int(soft.argmax()) == expected)
        hard_correct += int(int(hard.argmax()) == expected)
    assert soft_correct == 61, soft_correct
    assert hard_correct < soft_correct


def test_soft_chroma_has_no_pitch_class_bias_on_white_noise() -> None:

    torch.manual_seed(0)
    noise = torch.randn(1, 4 * SAMPLE_RATE) * 0.1
    soft = chroma_from_waveform(noise, n_fft=4096, mode="soft").mean(dim=1)[0]
    hard = chroma_from_waveform(noise, n_fft=1024, mode="hard").mean(dim=1)[0]
    assert float(soft.max() / soft.min()) < 1.4
    assert float(hard.max() / hard.min()) > 1.7


def test_chroma_frame_count_matches_mel_frontend() -> None:

    frontend = LogMelFrontend()
    for samples in (SAMPLE_RATE // 2, 3 * SAMPLE_RATE, 171_120):
        waveform = torch.randn(1, samples) * 0.1
        expected = frontend(waveform).shape[1]
        for mode, n_fft in (("hard", 1024), ("soft", 4096)):
            chroma = chroma_from_waveform(waveform, n_fft=n_fft, mode=mode)
            assert chroma.shape[1] == expected, (samples, mode)


def test_soft_chroma_is_causal() -> None:

    torch.manual_seed(0)
    waveform = torch.randn(1, 3 * SAMPLE_RATE) * 0.1
    frame = 200
    reference = chroma_from_waveform(waveform, n_fft=4096, mode="soft")
    disturbed = waveform.clone()
    tail = (frame + 1) * HOP
    disturbed[0, tail:] = torch.randn(disturbed.shape[1] - tail) * 0.5
    after = chroma_from_waveform(disturbed, n_fft=4096, mode="soft")
    torch.testing.assert_close(reference[:, : frame + 1], after[:, : frame + 1])


def test_soft_chroma_chunking_is_exact() -> None:

    torch.manual_seed(0)
    waveform = torch.randn(2, 7 * SAMPLE_RATE) * 0.1
    chunked = chroma_from_waveform(waveform, n_fft=4096, mode="soft", chunk_frames=512)
    whole = chroma_from_waveform(
        waveform, n_fft=4096, mode="soft", chunk_frames=10**9
    )
    torch.testing.assert_close(chunked, whole)


def test_chroma_rejects_window_shorter_than_n_fft() -> None:

    with pytest.raises(ValueError, match="win_length"):
        chroma_from_waveform(torch.randn(1, SAMPLE_RATE), n_fft=4096, win_length=960)
