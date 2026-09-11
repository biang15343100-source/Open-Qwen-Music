
from __future__ import annotations

import math

import pytest
import torch

from open_qwen_music.tokenizer.bestrq import (
    BESTRQ_CONTRACT_DEFAULTS,
    BESTRQ_CONTRACT_PRE_20260804_DEFAULTS,
    BestRQTarget,
    apply_waveform_mask,
    downsample_mask_100_to_25,
    load_whitening_stats,
    make_waveform_span_mask,
    relative_waveform_noise_std,
    resolve_bestrq_contract,
    waveform_mask_to_feature_mask,
)
from open_qwen_music.tokenizer.features import LogMelFrontend

N_MELS = 128


def _music_like(batch: int = 4, seconds: float = 6.0) -> torch.Tensor:

    generator = torch.Generator().manual_seed(11)
    sample_rate = 24_000
    time = torch.arange(int(seconds * sample_rate)) / sample_rate
    waveform = torch.zeros(batch, time.numel())
    for row in range(batch):
        fundamental = 110.0 * (1 + row)
        for harmonic in range(1, 8):
            amplitude = 0.3 / harmonic
            waveform[row] += amplitude * torch.sin(
                2 * math.pi * fundamental * harmonic * time
            )
        envelope = 0.5 + 0.5 * torch.sin(2 * math.pi * 1.7 * time)
        waveform[row] *= envelope
    waveform += 0.01 * torch.randn(waveform.shape, generator=generator)
    return waveform / waveform.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)


def _normalized_features() -> torch.Tensor:
    frontend = LogMelFrontend(
        mel_filter_norm="slaney", log_mode="additive", log_floor=1e-9
    )
    with torch.no_grad():
        features = frontend(_music_like())
    flat = features.reshape(-1, N_MELS)
    return (features - flat.mean(0)) / flat.std(0).clamp_min(1e-6)


def _reverse_within_group(features: torch.Tensor) -> torch.Tensor:
    batch, frames, mels = features.shape
    trim = frames - frames % 4
    head = features[:, :trim].reshape(batch, trim // 4, 4, mels).flip(dims=[2])
    return torch.cat([head.reshape(batch, trim, mels), features[:, trim:]], dim=1)


def _make_target(**overrides) -> BestRQTarget:
    arguments = dict(
        input_dim=N_MELS,
        projection_dim=16,
        codebook_size=8192,
        local_window=1,
        seed=20260719,
    )
    arguments.update(overrides)
    return BestRQTarget(**arguments)


def test_stack_preserves_intra_frame_order_but_mean_does_not() -> None:

    features = _normalized_features()
    reversed_features = _reverse_within_group(features)

    stacked = _make_target(aggregation="stack")
    pooled = _make_target(aggregation="mean")
    with torch.no_grad():
        stack_change = (stacked(features) != stacked(reversed_features)).float().mean()
        mean_change = (pooled(features) != pooled(reversed_features)).float().mean()

    assert float(mean_change) == 0.0, "Mean pooling should not be sensitive to frame order,The test signal or implementation is incorrect"
    assert float(stack_change) > 0.05, (
        f"stack is only as sensitive to intra-group frame sequence as {float(stack_change):.3f},"
        "indicates that the splicing sequence has not actually entered target"
    )


def test_stack_projection_matches_four_frame_input() -> None:
    stacked = _make_target(aggregation="stack")
    pooled = _make_target(aggregation="mean")
    assert stacked.target_dim == 4 * N_MELS
    assert stacked.projection.shape == (4 * N_MELS, 16)
    assert pooled.target_dim == N_MELS
    assert pooled.projection.shape == (N_MELS, 16)


def test_unknown_aggregation_is_rejected() -> None:
    with pytest.raises(ValueError, match="target_aggregation"):
        _make_target(aggregation="max")


@pytest.mark.parametrize("aggregation", ["stack", "mean"])
def test_target_ignores_batch_padding(aggregation: str) -> None:

    torch.manual_seed(5)
    target = _make_target(
        input_dim=16, projection_dim=8, codebook_size=64, aggregation=aggregation
    )
    valid = 37
    features = torch.randn(1, valid, 16)
    mask = torch.ones(1, valid, dtype=torch.bool)
    padded = torch.nn.functional.pad(features, (0, 0, 0, 23), value=-11.5)
    padded_mask = torch.nn.functional.pad(mask, (0, 23), value=False)

    with torch.no_grad():
        alone = target(features, mask)
        batched = target(padded, padded_mask)

    keep = valid // 4
    torch.testing.assert_close(alone[:, :keep], batched[:, :keep])


# --- whitening ------------------------------------------------------------------


def _participation_ratio(matrix: torch.Tensor) -> float:

    centered = (matrix - matrix.mean(dim=0)).double()
    covariance = centered.T @ centered / centered.shape[0]
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    return float(eigenvalues.sum() ** 2 / (eigenvalues**2).sum())


def _fit_whitening(pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = pooled.reshape(-1, pooled.shape[-1]).double()
    mean = flat.mean(dim=0)
    centered = flat - mean
    covariance = centered.T @ centered / centered.shape[0]
    values, vectors = torch.linalg.eigh(covariance)
    whitening = vectors @ torch.diag((values.clamp_min(0) + 1e-3).rsqrt()) @ vectors.T
    return mean.float(), whitening.float()


def test_whitening_makes_projection_isotropic() -> None:

    features = _normalized_features()
    baseline = _make_target(aggregation="stack")
    with torch.no_grad():
        pooled = baseline.aggregate(features)
    mean, whitening = _fit_whitening(pooled)
    whitened = _make_target(
        aggregation="stack", whitening_mean=mean, whitening_matrix=whitening
    )

    def projected_ratio(target: BestRQTarget) -> float:
        with torch.no_grad():
            aggregated = target.aggregate(features).reshape(-1, target.target_dim)
            centered = aggregated - target.whitening_mean
            unit = torch.nn.functional.normalize(
                centered @ target.whitening_matrix @ target.projection, dim=-1
            )
        return _participation_ratio(unit)

    before = projected_ratio(baseline)
    after = projected_ratio(whitened)

    assert before < 10.0, f" participation ratio unexpectedly high({before:.2f})"
    assert after > 14.0, (
        f"whitening after participation ratio only {after:.2f},Data is still concentrated in a few directions"
    )


def test_whitening_increases_active_codes() -> None:

    features = _normalized_features()
    baseline = _make_target(aggregation="stack")
    with torch.no_grad():
        mean, whitening = _fit_whitening(baseline.aggregate(features))
        whitened = _make_target(
            aggregation="stack", whitening_mean=mean, whitening_matrix=whitening
        )
        before = int(baseline(features).unique().numel())
        after = int(whitened(features).unique().numel())
    assert after > before, f"whitening starts from {before} downgraded to {after}"


def test_identity_whitening_is_a_no_op() -> None:

    features = _normalized_features()
    default = _make_target(aggregation="stack")
    explicit = _make_target(
        aggregation="stack",
        whitening_mean=torch.zeros(4 * N_MELS),
        whitening_matrix=torch.eye(4 * N_MELS),
    )
    with torch.no_grad():
        assert torch.equal(default(features), explicit(features))


def test_whitening_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="whitening_matrix must have shape"):
        _make_target(
            aggregation="stack",
            whitening_matrix=torch.eye(N_MELS),
        )


def test_missing_whitening_stats_reports_actionable_error() -> None:

    with pytest.raises(ValueError, match="whitening_stats"):
        load_whitening_stats(None, input_dim=N_MELS, aggregation="stack")


def test_whitening_stats_aggregation_must_match(tmp_path) -> None:
    path = tmp_path / "whitening.pt"
    torch.save(
        {
            "aggregation": "mean",
            "n_mels": N_MELS,
            "mean": torch.zeros(N_MELS),
            "matrix": torch.eye(N_MELS),
        },
        path,
    )
    with pytest.raises(ValueError, match="target_aggregation"):
        load_whitening_stats(str(path), input_dim=N_MELS, aggregation="stack")


def test_contract_defaults_describe_new_and_old_behaviour() -> None:
    assert BESTRQ_CONTRACT_DEFAULTS == {
        "target_aggregation": "stack",
        "local_window": 1,
        "whitening": True,
        "projection_init": "gaussian_column_norm",
        "mask_noise_mode": "absolute",
        "loss_mask_support": "nominal",
    }
    assert BESTRQ_CONTRACT_PRE_20260804_DEFAULTS == {
        "target_aggregation": "mean",
        "local_window": 3,
        "whitening": False,
        "projection_init": "gaussian_column_norm",
        "mask_noise_mode": "absolute",
        "loss_mask_support": "nominal",
    }


def test_xavier_projection_is_frozen_and_deterministic() -> None:
    first = BestRQTarget(
        input_dim=8,
        projection_dim=4,
        codebook_size=32,
        seed=17,
        projection_init="xavier_normal",
    )
    second = BestRQTarget(
        input_dim=8,
        projection_dim=4,
        codebook_size=32,
        seed=17,
        projection_init="xavier_normal",
    )
    assert torch.equal(first.projection, second.projection)
    assert first.projection.requires_grad is False
    assert first.codebook.requires_grad is False


def test_relative_waveform_noise_uses_ac_rms_not_dc() -> None:
    time = torch.arange(24_000) / 24_000
    tone = 0.1 * torch.sin(2 * math.pi * 440 * time)
    waveform = torch.stack([tone, tone + 0.8])
    scales = relative_waveform_noise_std(
        waveform,
        torch.tensor([24_000, 24_000]),
        noise_db=-20.0,
        minimum=0.0,
        maximum=1.0,
    )
    torch.testing.assert_close(scales[0], scales[1])
    expected = tone.square().mean().sqrt() * 0.1
    torch.testing.assert_close(scales[0], expected)


def test_relative_noise_and_frame_db_make_lufs_gain_redundant() -> None:

    generator = torch.Generator().manual_seed(41)
    waveform = torch.randn(2, 48_000, generator=generator) * 0.1
    lengths = torch.tensor([48_000, 48_000])
    mask = make_waveform_span_mask(
        lengths, 48_000, span_samples=9_600, probability=0.3
    )
    scale = relative_waveform_noise_std(
        waveform, lengths, noise_db=-20.0, minimum=0.0, maximum=1.0
    )
    gain = 10.0 ** (9.0 / 20.0)
    gained = waveform * gain
    gained_scale = relative_waveform_noise_std(
        gained, lengths, noise_db=-20.0, minimum=0.0, maximum=1.0
    )
    torch.testing.assert_close(gained_scale, scale * gain)

    torch.manual_seed(99)
    masked = apply_waveform_mask(waveform, mask, scale)
    torch.manual_seed(99)
    gained_masked = apply_waveform_mask(gained, mask, gained_scale)
    torch.testing.assert_close(gained_masked, masked * gain)

    frontend = LogMelFrontend(
        n_fft=2048,
        win_length=2048,
        mel_scale="slaney",
        mel_filter_norm="slaney",
        log_mode="db",
        log_floor=1e-12,
        top_db=60.0,
        top_db_scope="frame",
        feature_centering="frame",
    )
    with torch.no_grad():
        torch.testing.assert_close(
            frontend(masked), frontend(gained_masked), rtol=1e-5, atol=1e-5
        )


@pytest.mark.parametrize(
    ("n_fft", "expected_targets"),
    [(1024, list(range(11, 20))), (2048, list(range(12, 20)))],
)
def test_full_stft_support_drops_visible_left_boundary(
    n_fft: int, expected_targets: list[int]
) -> None:
    waveform_mask = torch.zeros(1, 24_000, dtype=torch.bool)
    waveform_mask[:, 9_600:19_200] = True
    feature_mask = waveform_mask_to_feature_mask(
        waveform_mask,
        hop_length=240,
        target_length=100,
        n_fft=n_fft,
        require_full_support=True,
    )
    target_mask = downsample_mask_100_to_25(
        feature_mask, 25, require_all=True
    )
    assert target_mask[0].nonzero().flatten().tolist() == expected_targets


def test_contract_resolution_uses_side_specific_defaults() -> None:

    empty_config = resolve_bestrq_contract({}, BESTRQ_CONTRACT_DEFAULTS)
    empty_checkpoint = resolve_bestrq_contract(
        {}, BESTRQ_CONTRACT_PRE_20260804_DEFAULTS
    )
    assert empty_config["target_aggregation"] == "stack"
    assert empty_checkpoint["target_aggregation"] == "mean"
    assert empty_config != empty_checkpoint
