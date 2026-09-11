
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .contracts import (
    AUDIO_CHANNELS,
    SAMPLE_RATE,
    STFT_N_FFT,
    lengths_to_mask,
    validate_audio,
)

EPSILON = 1.0e-8


@dataclass
class ReducedValue:
    value: Tensor
    numerator: Tensor
    denominator: Tensor


def masked_reduce(
    values: Tensor,
    mask: Tensor | None = None,
    *,
    weights: Tensor | None = None,
    sample_equal: bool = False,
) -> ReducedValue:

    work = values.float()
    effective = torch.ones_like(work) if mask is None else mask.to(work.dtype)
    effective = torch.broadcast_to(effective, work.shape)
    if weights is not None:
        effective = effective * torch.broadcast_to(weights.float(), work.shape)
    if sample_equal:
        reduce_dims = tuple(range(1, work.ndim))
        numerator = (work * effective).sum(dim=reduce_dims)
        denominator = effective.sum(dim=reduce_dims)
        per_sample = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1.0),
            numerator * 0.0,
        )
        value = per_sample.mean()
        return ReducedValue(value, numerator.sum(), denominator.sum())
    numerator = (work * effective).sum()
    denominator = effective.sum()
    value = torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1.0),
        numerator * 0.0,
    )
    return ReducedValue(value, numerator, denominator)


def waveform_l1_loss(
    estimate: Tensor,
    target: Tensor,
    lengths: Tensor,
    *,
    sample_equal: bool = False,
) -> Tensor:
    if estimate.shape != target.shape:
        raise ValueError("waveform L1 estimate and target shapes must match")
    validate_audio(estimate, lengths)
    validate_audio(target, lengths)
    mask = lengths_to_mask(
        lengths,
        estimate.shape[-1],
        device=estimate.device,
    ).unsqueeze(1)
    return masked_reduce(
        (estimate.float() - target.float()).abs(),
        mask,
        sample_equal=sample_equal,
    ).value


@dataclass(frozen=True)
class HighbandWaveformL1Components:

    local_loss_sum: Tensor
    local_eligible_records: Tensor
    per_record_loss: Tensor
    per_band_l1: Tensor
    target_band_rms: Tensor
    target_band_energy_ratio: Tensor
    eligible_band_mask: Tensor
    supervision_max_hz: Tensor


def _raised_cosine_ramp(
    frequencies: Tensor,
    *,
    start_hz: float,
    end_hz: float,
    rising: bool,
) -> Tensor:
    if not math.isfinite(start_hz) or not math.isfinite(end_hz):
        raise ValueError("raised-cosine boundary must be finite")
    if end_hz <= start_hz:
        raise ValueError("raised-cosine end_hz must be greater than start_hz")
    position = ((frequencies - start_hz) / (end_hz - start_hz)).clamp(0.0, 1.0)
    rising_value = 0.5 - 0.5 * torch.cos(math.pi * position)
    return rising_value if rising else 1.0 - rising_value


def _complementary_highband_masks(
    frequencies: Tensor,
    *,
    bands_hz: Sequence[tuple[float, float]],
    transition_hz: float,
) -> Tensor:

    bands = tuple((float(lower), float(upper)) for lower, upper in bands_hz)
    if not bands:
        raise ValueError("high-band waveform L1 requires at least one band")
    if not math.isfinite(transition_hz) or transition_hz <= 0.0:
        raise ValueError("high-band transition_hz must be finite and positive")
    nyquist = SAMPLE_RATE / 2
    for index, (lower, upper) in enumerate(bands):
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower < 0.0
            or upper <= lower
            or upper > nyquist
        ):
            raise ValueError(f"invalid high-band frequency range: {(lower, upper)}")
        if index and not math.isclose(
            bands[index - 1][1],
            lower,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError("High-band frequency bands must be contiguous and cover the full range")
    if transition_hz >= min(upper - lower for lower, upper in bands):
        raise ValueError("high-band transition_hz must be less than each band width")

    masks: list[Tensor] = []
    half_transition = transition_hz / 2.0
    for index, (lower, upper) in enumerate(bands):
        if index == 0:
            lower_weight = _raised_cosine_ramp(
                frequencies,
                start_hz=lower,
                end_hz=lower + transition_hz,
                rising=True,
            )
        else:
            lower_weight = _raised_cosine_ramp(
                frequencies,
                start_hz=lower - half_transition,
                end_hz=lower + half_transition,
                rising=True,
            )
        if index == len(bands) - 1:
            upper_weight = _raised_cosine_ramp(
                frequencies,
                start_hz=upper - transition_hz,
                end_hz=upper,
                rising=False,
            )
        else:
            upper_weight = _raised_cosine_ramp(
                frequencies,
                start_hz=upper - half_transition,
                end_hz=upper + half_transition,
                rising=False,
            )
        masks.append(lower_weight * upper_weight)
    return torch.stack(masks)


def multiband_filtered_waveform_l1_components(
    estimate: Tensor,
    target: Tensor,
    lengths: Tensor,
    media_bandwidth_hz: Tensor,
    magnitude_max_hz: Tensor,
    phase_max_hz: Tensor,
    stereo_max_hz: Tensor,
    *,
    sample_rate: int = SAMPLE_RATE,
    bands_hz: Sequence[tuple[float, float]] = (
        (4_000.0, 8_000.0),
        (8_000.0, 12_000.0),
        (12_000.0, 20_000.0),
    ),
    transition_hz: float = 250.0,
    reflect_padding_samples: int = 2_048,
    target_band_rms_min: float = 5.0e-4,
    target_band_energy_ratio_min: float = 1.0e-4,
    hard_max_hz: float = 20_000.0,
) -> HighbandWaveformL1Components:

    if estimate.shape != target.shape:
        raise ValueError("high-band waveform L1 estimate and target shapes must match")
    resolved_lengths = validate_audio(estimate, lengths)
    validate_audio(target, resolved_lengths)
    batch_size = estimate.shape[0]
    limits = []
    for name, value in (
        ("media_bandwidth_hz", media_bandwidth_hz),
        ("magnitude_max_hz", magnitude_max_hz),
        ("phase_max_hz", phase_max_hz),
        ("stereo_max_hz", stereo_max_hz),
    ):
        if value.shape != (batch_size,):
            raise ValueError(f"{name} must have shape [B]")
        current = value.to(device=estimate.device, dtype=torch.float32)
        if not bool(torch.isfinite(current).all()) or bool((current < 0.0).any()):
            raise ValueError(f"{name} must be finite and non-negative")
        limits.append(current)
    if sample_rate <= 0:
        raise ValueError("high-band sample_rate must be positive")
    if (
        reflect_padding_samples <= 0
        or not math.isfinite(target_band_rms_min)
        or target_band_rms_min < 0.0
        or not math.isfinite(target_band_energy_ratio_min)
        or target_band_energy_ratio_min < 0.0
        or not math.isfinite(hard_max_hz)
        or not 0.0 < hard_max_hz <= sample_rate / 2
    ):
        raise ValueError("high-band filter or eligibility configuration is invalid")

    supervision_max_hz = torch.stack(
        [*limits, torch.full_like(limits[0], float(hard_max_hz))]
    ).amin(dim=0)
    per_band_l1_rows: list[Tensor] = []
    target_band_rms_rows: list[Tensor] = []
    target_band_ratio_rows: list[Tensor] = []
    eligible_rows: list[Tensor] = []
    per_record_losses: list[Tensor] = []
    eligible_records: list[Tensor] = []

    for record_index in range(batch_size):
        length = int(resolved_lengths[record_index].item())
        if length <= reflect_padding_samples:
            raise ValueError(
                "high-band waveform L1 must exceed reflect padding:"
                f"{length}<={reflect_padding_samples}"
            )
        estimate_record = estimate[record_index, :, :length].float()
        target_record = target[record_index, :, :length].float()
        residual = estimate_record - target_record
        padded_target = F.pad(
            target_record,
            (reflect_padding_samples, reflect_padding_samples),
            mode="reflect",
        )
        padded_residual = F.pad(
            residual,
            (reflect_padding_samples, reflect_padding_samples),
            mode="reflect",
        )
        padded_length = padded_target.shape[-1]
        frequencies = torch.fft.rfftfreq(
            padded_length,
            d=1.0 / float(sample_rate),
            device=estimate.device,
        )
        base_masks = _complementary_highband_masks(
            frequencies,
            bands_hz=bands_hz,
            transition_hz=transition_hz,
        )
        ceiling = supervision_max_hz[record_index]
        ceiling_weight = _raised_cosine_ramp(
            frequencies,
            start_hz=float(ceiling.detach().item()) - transition_hz,
            end_hz=float(ceiling.detach().item()),
            rising=False,
        )
        ceiling_weight = torch.where(
            frequencies <= ceiling,
            ceiling_weight,
            torch.zeros_like(ceiling_weight),
        )
        masks = torch.minimum(base_masks, ceiling_weight.unsqueeze(0))
        target_spectrum = torch.fft.rfft(padded_target, dim=-1)
        residual_spectrum = torch.fft.rfft(padded_residual, dim=-1)
        crop = slice(reflect_padding_samples, reflect_padding_samples + length)
        band_targets = torch.fft.irfft(
            target_spectrum.unsqueeze(0) * masks[:, None, :],
            n=padded_length,
            dim=-1,
        )[..., crop]
        band_residuals = torch.fft.irfft(
            residual_spectrum.unsqueeze(0) * masks[:, None, :],
            n=padded_length,
            dim=-1,
        )[..., crop]
        band_l1 = band_residuals.abs().mean(dim=(1, 2))
        band_energy = band_targets.square().mean(dim=(1, 2))
        band_rms = _zero_preserving_rms(band_energy)
        full_energy = target_record.square().mean().detach()
        energy_ratio = band_energy / full_energy.clamp_min(
            torch.finfo(torch.float32).tiny
        )
        mask_nonempty = masks.amax(dim=1) > 0.0
        eligible = (
            mask_nonempty
            & (band_rms.detach() >= target_band_rms_min)
            & (energy_ratio.detach() >= target_band_energy_ratio_min)
        )
        eligible_count = eligible.sum()
        per_record_loss = torch.where(
            eligible_count > 0,
            (band_l1 * eligible.to(band_l1.dtype)).sum()
            / eligible_count.clamp_min(1).to(band_l1.dtype),
            band_l1.sum() * 0.0,
        )
        per_band_l1_rows.append(band_l1)
        target_band_rms_rows.append(band_rms)
        target_band_ratio_rows.append(energy_ratio)
        eligible_rows.append(eligible)
        per_record_losses.append(per_record_loss)
        eligible_records.append(eligible_count > 0)

    per_record_loss_tensor = torch.stack(per_record_losses)
    eligible_record_mask = torch.stack(eligible_records)
    local_loss_sum = (
        per_record_loss_tensor * eligible_record_mask.to(per_record_loss_tensor.dtype)
    ).sum()
    return HighbandWaveformL1Components(
        local_loss_sum=local_loss_sum,
        local_eligible_records=eligible_record_mask.sum(),
        per_record_loss=per_record_loss_tensor,
        per_band_l1=torch.stack(per_band_l1_rows),
        target_band_rms=torch.stack(target_band_rms_rows),
        target_band_energy_ratio=torch.stack(target_band_ratio_rows),
        eligible_band_mask=torch.stack(eligible_rows),
        supervision_max_hz=supervision_max_hz,
    )


def _validate_waveform_triplet(
    estimate: Tensor,
    baseline: Tensor,
    target: Tensor,
    lengths: Tensor,
    *,
    name: str,
) -> Tensor:
    if estimate.shape != baseline.shape or estimate.shape != target.shape:
        raise ValueError(f"{name} estimate, baseline, and target shapes must match")
    resolved = validate_audio(estimate, lengths)
    validate_audio(baseline, resolved)
    validate_audio(target, resolved)
    return resolved


def _waveform_target_rms(
    target: Tensor,
    mask: Tensor,
    *,
    floor: float,
) -> Tensor:
    expanded = mask.expand_as(target)
    energy = (target.float().square() * expanded).sum(dim=(1, 2))
    count = expanded.sum(dim=(1, 2)).clamp_min(1.0)
    return (energy / count).sqrt().clamp_min(floor)


def _zero_preserving_rms(squared_mean: Tensor) -> Tensor:

    if not squared_mean.is_floating_point():
        raise TypeError("zero-preserving RMS requires a floating-point tensor")
    tiny = torch.finfo(squared_mean.dtype).tiny
    return (squared_mean + tiny).sqrt() - math.sqrt(tiny)


def waveform_peak_envelope_overshoot_loss(
    estimate: Tensor,
    target: Tensor,
    lengths: Tensor,
    *,
    window_size: int = 63,
    hop_size: int = 16,
    rms_floor: float = 1.0e-4,
) -> Tensor:

    if estimate.shape != target.shape:
        raise ValueError("peak envelope estimate and target shapes must match")
    resolved = validate_audio(estimate, lengths)
    validate_audio(target, resolved)
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("peak envelope window_size must be a positive odd number")
    if hop_size <= 0:
        raise ValueError("peak envelope hop_size must be positive")
    if not math.isfinite(rms_floor) or rms_floor <= 0:
        raise ValueError("peak envelope rms_floor must be finite and positive")
    mask = lengths_to_mask(
        resolved, estimate.shape[-1], device=estimate.device
    ).unsqueeze(1)
    estimate_abs = estimate.float().abs() * mask
    target_abs = target.float().abs() * mask
    scale = _waveform_target_rms(target.float(), mask, floor=rms_floor)

    estimate_peak = estimate_abs.amax(dim=(1, 2))
    target_peak = target_abs.amax(dim=(1, 2))
    global_excess = F.relu(estimate_peak - target_peak) / scale

    padding = window_size // 2
    estimate_envelope = F.max_pool1d(
        estimate_abs,
        kernel_size=window_size,
        stride=hop_size,
        padding=padding,
    )
    target_envelope = F.max_pool1d(
        target_abs,
        kernel_size=window_size,
        stride=hop_size,
        padding=padding,
    )
    envelope_mask = F.max_pool1d(
        mask.float(),
        kernel_size=window_size,
        stride=hop_size,
        padding=padding,
    ).bool()
    local_excess = F.relu(estimate_envelope - target_envelope) / scale[:, None, None]
    local = masked_reduce(
        local_excess,
        envelope_mask,
        sample_equal=True,
    ).value
    return global_excess.mean() + local


def waveform_global_peak_ceiling_loss(
    estimate: Tensor,
    target: Tensor,
    lengths: Tensor,
    *,
    ceiling: float = 1.0,
    rms_floor: float = 1.0e-4,
    smooth_l1_beta: float = 1.0,
) -> Tensor:

    if estimate.shape != target.shape:
        raise ValueError("global peak estimate and target shapes must match")
    resolved = validate_audio(estimate, lengths)
    validate_audio(target, resolved)
    for name, value in (
        ("ceiling", ceiling),
        ("rms_floor", rms_floor),
        ("smooth_l1_beta", smooth_l1_beta),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"global peak {name} must be finite and positive")
    mask = lengths_to_mask(
        resolved, estimate.shape[-1], device=estimate.device
    ).unsqueeze(1)
    estimate_peak = (estimate.float().abs() * mask).amax(dim=(1, 2))
    target_peak = (target.float().abs() * mask).amax(dim=(1, 2))
    permitted_peak = target_peak.clamp_max(float(ceiling))
    scale = _waveform_target_rms(target.float(), mask, floor=rms_floor)
    normalized_excess = F.relu(estimate_peak - permitted_peak) / scale
    return F.smooth_l1_loss(
        normalized_excess,
        torch.zeros_like(normalized_excess),
        beta=float(smooth_l1_beta),
        reduction="mean",
    )


def waveform_residual_energy_overshoot_loss(
    estimate: Tensor,
    baseline: Tensor,
    target: Tensor,
    lengths: Tensor,
    *,
    rms_floor: float = 1.0e-4,
) -> Tensor:

    resolved = _validate_waveform_triplet(
        estimate,
        baseline,
        target,
        lengths,
        name="residual energy",
    )
    if not math.isfinite(rms_floor) or rms_floor <= 0:
        raise ValueError("residual energy rms_floor must be finite and positive")
    mask = lengths_to_mask(
        resolved, estimate.shape[-1], device=estimate.device
    ).unsqueeze(1)
    expanded = mask.expand_as(estimate)
    residual = (estimate.float() - baseline.float()) * expanded
    oracle = (target.float() - baseline.float()) * expanded
    scale = _waveform_target_rms(target.float(), mask, floor=rms_floor)

    pointwise_excess = F.relu(residual.abs() - oracle.abs()) / scale[:, None, None]
    pointwise = masked_reduce(
        pointwise_excess,
        mask,
        sample_equal=True,
    ).value
    count = expanded.sum(dim=(1, 2)).clamp_min(1.0)
    residual_rms = _zero_preserving_rms(residual.square().sum(dim=(1, 2)) / count)
    oracle_rms = _zero_preserving_rms(oracle.square().sum(dim=(1, 2)) / count)
    global_excess = F.relu(residual_rms - oracle_rms) / scale
    return pointwise + global_excess.mean()


def waveform_si_sdr_no_regression_loss(
    estimate: Tensor,
    baseline: Tensor,
    target: Tensor,
    lengths: Tensor,
    *,
    margin_db: float = 0.0,
    epsilon: float = 1.0e-8,
) -> Tensor:

    resolved = _validate_waveform_triplet(
        estimate,
        baseline,
        target,
        lengths,
        name="SI-SDR guard",
    )
    if not math.isfinite(margin_db) or margin_db < 0:
        raise ValueError("SI-SDR guard margin_db must be finite and non-negative")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("SI-SDR guard epsilon must be finite and positive")
    mask = lengths_to_mask(
        resolved, estimate.shape[-1], device=estimate.device
    ).unsqueeze(1)

    def lrms(values: Tensor) -> Tensor:
        work = values.float() * mask
        return torch.cat((work, lr_to_ms(work)), dim=1)

    target_lrms = lrms(target)
    stream_mask = mask.expand(-1, target_lrms.shape[1], -1)

    def score(values: Tensor) -> tuple[Tensor, Tensor]:
        work = lrms(values)
        target_energy = (target_lrms.square() * stream_mask).sum(dim=-1)
        scale = (work * target_lrms * stream_mask).sum(
            dim=-1
        ) / target_energy.clamp_min(epsilon)
        projected = scale.unsqueeze(-1) * target_lrms
        noise = (work - projected) * stream_mask
        projected_energy = projected.square().sum(dim=-1).clamp_min(epsilon)
        noise_energy = noise.square().sum(dim=-1).clamp_min(epsilon)
        return 10.0 * torch.log10(projected_energy / noise_energy), target_energy

    estimate_score, target_energy = score(estimate)
    baseline_score, _ = score(baseline)
    valid = target_energy > epsilon
    penalties = F.relu(baseline_score.detach() - estimate_score + margin_db)
    if not bool(valid.any()):
        return estimate.sum() * 0.0
    return penalties.masked_select(valid).mean()


def multiresolution_ccpc(
    estimate: Tensor,
    target: Tensor,
    lengths: Tensor | None = None,
    media_bandwidth_hz: Tensor | None = None,
    *,
    fft_sizes: Sequence[int] = (512, 1024, 2048, 4096),
    magnitude_threshold: float = 1.0e-6,
    epsilon: float = 1.0e-8,
    detach_estimate_weights: bool = True,
) -> Tensor:

    if estimate.shape != target.shape:
        raise ValueError("CCPC estimate and target shapes must match")
    resolved_lengths = validate_audio(estimate, lengths)
    validate_audio(target, resolved_lengths)
    if bool((resolved_lengths != estimate.shape[-1]).any()):
        values = [
            multiresolution_ccpc(
                estimate[index : index + 1, :, : int(length)],
                target[index : index + 1, :, : int(length)],
                None,
                (
                    None
                    if media_bandwidth_hz is None
                    else media_bandwidth_hz[index : index + 1]
                ),
                fft_sizes=fft_sizes,
                magnitude_threshold=magnitude_threshold,
                epsilon=epsilon,
                detach_estimate_weights=detach_estimate_weights,
            )
            for index, length in enumerate(resolved_lengths)
        ]
        return torch.stack(values).mean()
    resolution_scores: list[Tensor] = []
    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(
            n_fft,
            dtype=torch.float32,
            device=estimate.device,
        )

        def transform(values: Tensor) -> Tensor:
            padding = n_fft // 2
            if values.shape[-1] > padding:
                left = values[..., 1 : padding + 1].flip(-1)
                right = values[..., -padding - 1 : -1].flip(-1)
                transformed_values = torch.cat((left, values, right), dim=-1)
                center = False
            else:
                transformed_values = values
                center = True
            spectrum = torch.stft(
                transformed_values.float().flatten(0, 1),
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=window,
                center=center,
                pad_mode="constant",
                return_complex=True,
            )
            return spectrum.reshape(
                values.shape[0],
                2,
                spectrum.shape[-2],
                spectrum.shape[-1],
            )

        estimate_stft = transform(estimate)
        target_stft = transform(target)
        estimate_left, estimate_left_valid = _unit(estimate_stft[:, 0])
        estimate_right, estimate_right_valid = _unit(estimate_stft[:, 1])
        target_left, target_left_valid = _unit(target_stft[:, 0])
        target_right, target_right_valid = _unit(target_stft[:, 1])
        estimate_ipd = estimate_left * estimate_right.conj()
        target_ipd = target_left * target_right.conj()
        phase_error = estimate_ipd * target_ipd.conj()
        weights = estimate_stft[:, 0].abs() * estimate_stft[:, 1].abs()
        if detach_estimate_weights:
            weights = weights.detach()
        valid = (
            estimate_left_valid
            & estimate_right_valid
            & target_left_valid
            & target_right_valid
            & (weights > magnitude_threshold)
        )
        if media_bandwidth_hz is not None:
            frequency_mask = spectral_bandwidth_mask(
                media_bandwidth_hz,
                bins=estimate_stft.shape[-2],
                bin_hz=SAMPLE_RATE / n_fft,
                device=estimate_stft.device,
            )
            valid = valid & frequency_mask.unsqueeze(-1)
        weights = weights * valid
        frame_vector = (weights * phase_error).sum(dim=1)
        frame_energy = weights.sum(dim=1)
        frame_resultant = torch.complex(
            frame_vector.real.float(),
            frame_vector.imag.float(),
        ).abs()
        frame_coherence = frame_resultant / frame_energy.clamp_min(epsilon)
        sample_energy = frame_energy.sum(dim=1)
        sample_coherence = (frame_coherence * frame_energy).sum(
            dim=1
        ) / sample_energy.clamp_min(epsilon)
        resolution_scores.append(sample_coherence)
    return torch.stack(resolution_scores).mean().clamp(0.0, 1.0)


def multiresolution_ccpc_no_regression_loss(
    estimate: Tensor,
    baseline: Tensor,
    target: Tensor,
    lengths: Tensor,
    media_bandwidth_hz: Tensor | None = None,
    *,
    margin: float = 0.0,
    allowed_regression: float = 0.0,
    fft_sizes: Sequence[int] = (512, 1024, 2048, 4096),
    magnitude_threshold: float = 1.0e-6,
    epsilon: float = 1.0e-8,
) -> Tensor:

    resolved = _validate_waveform_triplet(
        estimate,
        baseline,
        target,
        lengths,
        name="CCPC no-regression",
    )
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("CCPC no-regression margin must be finite and non-negative")
    if (
        not math.isfinite(allowed_regression)
        or allowed_regression < 0
        or allowed_regression >= 1
    ):
        raise ValueError("CCPC no-regression allowed_regression must be a finite value in [0, 1)")
    penalties = []
    for index, length in enumerate(resolved):
        length_value = int(length)
        bandwidth = (
            None
            if media_bandwidth_hz is None
            else media_bandwidth_hz[index : index + 1]
        )
        estimate_score = multiresolution_ccpc(
            estimate[index : index + 1, :, :length_value],
            target[index : index + 1, :, :length_value],
            media_bandwidth_hz=bandwidth,
            fft_sizes=fft_sizes,
            magnitude_threshold=magnitude_threshold,
            epsilon=epsilon,
            detach_estimate_weights=False,
        )
        with torch.no_grad():
            baseline_score = multiresolution_ccpc(
                baseline[index : index + 1, :, :length_value],
                target[index : index + 1, :, :length_value],
                media_bandwidth_hz=bandwidth,
                fft_sizes=fft_sizes,
                magnitude_threshold=magnitude_threshold,
                epsilon=epsilon,
            )
        penalties.append(
            F.relu(baseline_score - estimate_score + margin - allowed_regression)
        )
    return torch.stack(penalties).mean()


def _time_mask(mask: Tensor | None, values: Tensor) -> Tensor | None:
    if mask is None:
        return None
    if mask.ndim != 2 or mask.shape != (values.shape[0], values.shape[-1]):
        raise ValueError("time mask must be bool with shape [B, T]")
    shape = [mask.shape[0]] + [1] * (values.ndim - 2) + [mask.shape[1]]
    return mask.reshape(shape)


def spectral_bandwidth_mask(
    media_bandwidth_hz: Tensor,
    *,
    bins: int,
    bin_hz: float,
    device: torch.device,
) -> Tensor:
    if (
        media_bandwidth_hz.ndim != 1
        or not media_bandwidth_hz.is_floating_point()
        or not torch.isfinite(media_bandwidth_hz).all()
        or bool((media_bandwidth_hz < 0).any())
        or bool((media_bandwidth_hz > SAMPLE_RATE / 2).any())
    ):
        raise ValueError("Band upper limit must be a finite floating-point tensor in [0, 24,000] with shape [B]")
    frequencies = torch.arange(bins, device=device, dtype=torch.float32) * float(bin_hz)
    limits = media_bandwidth_hz.to(device=device, dtype=torch.float32).unsqueeze(1)
    return (frequencies.unsqueeze(0) <= limits) & (limits > 0)


def k_weighting_response(
    *,
    bins: int,
    bin_hz: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:

    frequencies = torch.arange(bins, device=device, dtype=torch.float64) * float(bin_hz)
    z1 = torch.exp(
        torch.complex(
            torch.zeros_like(frequencies),
            -2.0 * math.pi * frequencies / SAMPLE_RATE,
        )
    )

    def response(
        b0: float,
        b1: float,
        b2: float,
        a1: float,
        a2: float,
    ) -> Tensor:
        numerator = b0 + b1 * z1 + b2 * z1.square()
        denominator = 1.0 + a1 * z1 + a2 * z1.square()
        return numerator / denominator

    shelf = response(
        1.53512485958697,
        -2.69169618940638,
        1.19839281085285,
        -1.69065929318241,
        0.73248077421585,
    )
    high_pass = response(
        1.0,
        -2.0,
        1.0,
        -1.99004745483398,
        0.99007225036621,
    )
    return (shelf * high_pass).abs().to(dtype=dtype)


def _spectral_mask(
    time_mask: Tensor | None,
    frequency_mask: Tensor | None,
    values: Tensor,
) -> Tensor | None:
    result = _time_mask(time_mask, values)
    if frequency_mask is not None:
        if frequency_mask.dtype != torch.bool or frequency_mask.shape != (
            values.shape[0],
            values.shape[-2],
        ):
            raise ValueError("frequency_mask must be bool with shape [B, F]")
        shaped = frequency_mask.reshape(
            frequency_mask.shape[0],
            *([1] * (values.ndim - 3)),
            frequency_mask.shape[1],
            1,
        )
        result = shaped if result is None else result & shaped
    return result


def safe_magnitude(spectrum: Tensor, epsilon: float = EPSILON) -> Tensor:

    if not spectrum.is_complex():
        raise ValueError("safe_magnitude requires a complex tensor")
    del epsilon


    return spectrum.to(torch.complex64).abs().float()


def refiner_oracle_log_magnitude_loss(
    estimate: Tensor,
    baseline: Tensor,
    target: Tensor,
    time_mask: Tensor | None = None,
    *,
    magnitude_max_hz: Tensor | None = None,
    magnitude_mode_weights: Tensor | None = None,
    max_abs_log_residual: float = 4.0,
    smooth_l1_beta: float = 0.1,
    sample_equal: bool = True,
    epsilon: float = EPSILON,
) -> Tensor:

    if estimate.shape != baseline.shape or estimate.shape != target.shape:
        raise ValueError(
            "Refiner oracle magnitude estimate, baseline, and target shapes must match"
        )
    if (
        not estimate.is_complex()
        or not baseline.is_complex()
        or not target.is_complex()
    ):
        raise ValueError("Refiner oracle magnitude requires a complex spectrum")
    if not math.isfinite(max_abs_log_residual) or max_abs_log_residual <= 0:
        raise ValueError("max_abs_log_residual must be finite and positive")
    if not math.isfinite(smooth_l1_beta) or smooth_l1_beta <= 0:
        raise ValueError("smooth_l1_beta must be finite and positive")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    bins = estimate.shape[-2]
    frequency_mask = None
    if magnitude_max_hz is not None:
        frequency_mask = spectral_bandwidth_mask(
            magnitude_max_hz,
            bins=bins,
            bin_hz=SAMPLE_RATE / STFT_N_FFT,
            device=estimate.device,
        )
    valid = _spectral_mask(time_mask, frequency_mask, estimate.real)

    if magnitude_mode_weights is None:
        mode_weights = torch.ones(
            bins,
            dtype=torch.float32,
            device=estimate.device,
        )
    else:
        if (
            magnitude_mode_weights.ndim != 1
            or magnitude_mode_weights.shape[0] != bins
            or not magnitude_mode_weights.is_floating_point()
            or not torch.isfinite(magnitude_mode_weights).all()
            or bool((magnitude_mode_weights < 0).any())
            or bool((magnitude_mode_weights > 1).any())
        ):
            raise ValueError("magnitude_mode_weights must be a finite floating-point tensor in [0, 1] with shape [F]")
        mode_weights = magnitude_mode_weights.to(
            device=estimate.device,
            dtype=torch.float32,
        )
    mode_weights = mode_weights[None, None, :, None]

    baseline_magnitude = safe_magnitude(baseline)
    estimate_magnitude = safe_magnitude(estimate)
    target_magnitude = safe_magnitude(target)
    predicted = (
        torch.log(estimate_magnitude.clamp_min(epsilon))
        - torch.log(baseline_magnitude.clamp_min(epsilon))
    ).clamp(-max_abs_log_residual, max_abs_log_residual)
    oracle = (
        torch.log(target_magnitude.clamp_min(epsilon))
        - torch.log(baseline_magnitude.clamp_min(epsilon))
    ).clamp(-max_abs_log_residual, max_abs_log_residual)
    oracle = oracle * mode_weights
    difference = F.smooth_l1_loss(
        predicted,
        oracle,
        beta=smooth_l1_beta,
        reduction="none",
    )
    audibility = (baseline_magnitude * target_magnitude).clamp_min(
        0.0
    ).sqrt().detach() * mode_weights
    return masked_reduce(
        difference,
        valid,
        weights=audibility,
        sample_equal=sample_equal,
    ).value


def _spectral_sample_rms_scale(
    target: Tensor,
    *,
    mask: Tensor | None,
    frequency_mask: Tensor | None,
    floor: float,
) -> Tensor:

    valid = _spectral_mask(mask, frequency_mask, target)
    if valid is None:
        valid = torch.ones_like(target.real, dtype=torch.bool)
    else:
        valid = valid.expand_as(target.real)
    count = valid.sum(dim=(1, 2, 3))
    if bool((count == 0).any()):
        raise ValueError("sample RMS normalization received empty supervision band")
    power = target.abs().square().masked_fill(~valid, 0.0).sum(dim=(1, 2, 3)) / count
    return power.sqrt().clamp_min(float(floor))


def spectral_linear_loss(
    estimate: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    frequency_mask: Tensor | None = None,
    *,
    sample_equal: bool = False,
) -> Tensor:
    difference = (safe_magnitude(estimate) - safe_magnitude(target)).abs()
    return masked_reduce(
        difference,
        _spectral_mask(mask, frequency_mask, difference),
        sample_equal=sample_equal,
    ).value


def spectral_complex_l1_loss(
    estimate: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    frequency_mask: Tensor | None = None,
    *,
    sample_equal: bool = False,
) -> Tensor:

    if not estimate.is_complex() or not target.is_complex():
        raise ValueError("complex L1 requires complex estimate and target tensors")
    difference = 0.5 * (
        (estimate.real.float() - target.real.float()).abs()
        + (estimate.imag.float() - target.imag.float()).abs()
    )
    return masked_reduce(
        difference,
        _spectral_mask(mask, frequency_mask, difference),
        sample_equal=sample_equal,
    ).value


def spectral_log1p_loss(
    estimate: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    frequency_mask: Tensor | None = None,
    *,
    sample_equal: bool = False,
) -> Tensor:
    estimate_log = torch.log1p(safe_magnitude(estimate))
    target_log = torch.log1p(safe_magnitude(target))
    difference = (estimate_log - target_log).abs()
    return masked_reduce(
        difference,
        _spectral_mask(mask, frequency_mask, difference),
        sample_equal=sample_equal,
    ).value


def adaptive_log_magnitude_loss(
    estimate: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    frequency_mask: Tensor | None = None,
    *,
    sample_equal: bool = False,
) -> Tensor:

    estimate_magnitude = safe_magnitude(estimate)
    target_magnitude = safe_magnitude(target)
    valid = _spectral_mask(mask, frequency_mask, estimate_magnitude)
    if valid is None:
        valid_values = torch.ones_like(estimate_magnitude)
    else:
        valid_values = torch.broadcast_to(valid, estimate_magnitude.shape).float()
    count = valid_values.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)

    def masked_std(values: Tensor) -> Tensor:
        mean = (values * valid_values).sum(dim=(-2, -1), keepdim=True) / count
        variance = ((values - mean).square() * valid_values).sum(
            dim=(-2, -1), keepdim=True
        ) / count
        return variance.clamp_min(0.0).sqrt()

    sigma = (
        torch.sqrt(
            masked_std(target_magnitude).square()
            + masked_std(estimate_magnitude).square()
        )
        .detach()
        .clamp_min(EPSILON)
    )
    difference = (
        torch.log1p(estimate_magnitude / sigma) - torch.log1p(target_magnitude / sigma)
    ).abs()
    return masked_reduce(difference, valid, sample_equal=sample_equal).value


def _waveform_stft(
    waveform: Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    lengths: Tensor | None,
) -> tuple[Tensor, Tensor]:
    values = waveform.float()
    if lengths is None:
        lengths = torch.full(
            (values.shape[0],),
            values.shape[-1],
            dtype=torch.long,
            device=values.device,
        )
    validate_audio(values, lengths)
    waveform_mask = lengths_to_mask(
        lengths, values.shape[-1], device=values.device
    ).unsqueeze(1)
    values = values * waveform_mask
    frame_lengths = torch.div(
        lengths + hop_length - 1, hop_length, rounding_mode="floor"
    )
    frames = int(frame_lengths.max().item())
    analysis_length = n_fft + (frames - 1) * hop_length
    if analysis_length < values.shape[-1]:
        values = values[..., :analysis_length]
    else:
        values = F.pad(values, (0, analysis_length - values.shape[-1]))
    window = torch.hann_window(
        win_length, dtype=torch.float32, device=values.device
    ).clamp_min(1.0e-3)
    spectrum = torch.stft(
        values.reshape(-1, values.shape[-1]),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        return_complex=True,
    )
    spectrum = spectrum.reshape(
        values.shape[0], values.shape[1], spectrum.shape[-2], spectrum.shape[-1]
    )
    frame_mask = lengths_to_mask(
        frame_lengths, spectrum.shape[-1], device=values.device
    )
    return spectrum, frame_mask


class MultiResolutionSTFTLoss(nn.Module):

    def __init__(
        self,
        resolutions: Sequence[Sequence[int]] = (
            (512, 128, 512),
            (1024, 256, 1024),
            (2048, 512, 2048),
        ),
        *,
        linear_weight: float = 1.0,
        log_weight: float = 1.0,
        k_weighting: bool = True,
        adaptive_log_magnitude: bool = True,
        sample_rms_normalization: bool = False,
        sample_rms_floor: float = 1.0e-4,
        sample_equal_reduction: bool = False,
        objective_profile: str = "oqm_linear_adaptive_log_v1",
        spectral_convergence_epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        parsed = tuple(tuple(int(value) for value in item) for item in resolutions)
        if not parsed or any(len(item) != 3 for item in parsed):
            raise ValueError("resolutions must be a non-empty list of (n_fft, hop, win_length) tuples")
        for n_fft, hop, win in parsed:
            if min(n_fft, hop, win) <= 0 or win > n_fft:
                raise ValueError(f"Invalid STFT resolution: {(n_fft, hop, win)}")
        self.resolutions = parsed
        self.linear_weight = float(linear_weight)
        self.log_weight = float(log_weight)
        self.k_weighting = bool(k_weighting)
        self.adaptive_log_magnitude = bool(adaptive_log_magnitude)
        self.sample_rms_normalization = bool(sample_rms_normalization)
        self.sample_rms_floor = float(sample_rms_floor)
        self.sample_equal_reduction = bool(sample_equal_reduction)
        self.objective_profile = str(objective_profile)
        self.spectral_convergence_epsilon = float(spectral_convergence_epsilon)
        if not math.isfinite(self.sample_rms_floor) or self.sample_rms_floor <= 0:
            raise ValueError("sample_rms_floor must be finite and positive")
        if self.objective_profile not in {
            "oqm_linear_adaptive_log_v1",
            "yamamoto_spectral_convergence_logmag_v1",
        }:
            raise ValueError("MR-STFT objective_profile is invalid")
        if (
            not math.isfinite(self.spectral_convergence_epsilon)
            or self.spectral_convergence_epsilon <= 0
        ):
            raise ValueError("MR-STFT spectral_convergence_epsilon must be finite and positive")
        if self.objective_profile == "yamamoto_spectral_convergence_logmag_v1" and (
            self.adaptive_log_magnitude or self.sample_rms_normalization
        ):
            raise ValueError(
                "Yamamoto MR-STFT does not support adaptive-log or sample-RMS normalization"
            )

    def components(
        self,
        estimate: Tensor,
        target: Tensor,
        lengths: Tensor | None = None,
        media_bandwidth_hz: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if estimate.shape != target.shape:
            raise ValueError("MR-STFT estimate and target shapes must match")
        linear_terms: list[Tensor] = []
        log_terms: list[Tensor] = []
        for n_fft, hop, win in self.resolutions:
            estimate_spectrum, frame_mask = _waveform_stft(
                estimate,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                lengths=lengths,
            )
            target_spectrum, _ = _waveform_stft(
                target,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                lengths=lengths,
            )
            if self.k_weighting:
                response = k_weighting_response(
                    bins=estimate_spectrum.shape[-2],
                    bin_hz=SAMPLE_RATE / n_fft,
                    device=estimate_spectrum.device,
                )[None, None, :, None]
                estimate_spectrum = estimate_spectrum * response
                target_spectrum = target_spectrum * response
            frequency_mask = (
                None
                if media_bandwidth_hz is None
                else spectral_bandwidth_mask(
                    media_bandwidth_hz,
                    bins=estimate_spectrum.shape[-2],
                    bin_hz=SAMPLE_RATE / n_fft,
                    device=estimate_spectrum.device,
                )
            )
            if self.sample_rms_normalization:
                scale = _spectral_sample_rms_scale(
                    target_spectrum,
                    mask=frame_mask,
                    frequency_mask=frequency_mask,
                    floor=self.sample_rms_floor,
                )
                estimate_spectrum = estimate_spectrum / scale[:, None, None, None]
                target_spectrum = target_spectrum / scale[:, None, None, None]
            if self.objective_profile == "yamamoto_spectral_convergence_logmag_v1":
                estimate_magnitude = safe_magnitude(estimate_spectrum)
                target_magnitude = safe_magnitude(target_spectrum)
                valid = _spectral_mask(
                    frame_mask,
                    frequency_mask,
                    estimate_magnitude,
                )
                if valid is None:
                    valid = torch.ones_like(estimate_magnitude, dtype=torch.bool)
                expanded = torch.broadcast_to(valid, estimate_magnitude.shape)
                difference = (estimate_magnitude - target_magnitude).masked_fill(
                    ~expanded,
                    0.0,
                )
                target_valid = target_magnitude.masked_fill(~expanded, 0.0)
                convergence = (
                    torch.linalg.vector_norm(difference, dim=(-2, -1))
                    / (
                        torch.linalg.vector_norm(target_valid, dim=(-2, -1))
                        + self.spectral_convergence_epsilon
                    )
                ).mean()
                log_magnitude = masked_reduce(
                    (
                        torch.log(
                            estimate_magnitude.clamp_min(
                                self.spectral_convergence_epsilon
                            )
                        )
                        - torch.log(
                            target_magnitude.clamp_min(
                                self.spectral_convergence_epsilon
                            )
                        )
                    ).abs(),
                    valid,
                    sample_equal=self.sample_equal_reduction,
                ).value
                linear_terms.append(convergence)
                log_terms.append(log_magnitude)
            else:
                linear_terms.append(
                    spectral_linear_loss(
                        estimate_spectrum,
                        target_spectrum,
                        frame_mask,
                        frequency_mask,
                        sample_equal=self.sample_equal_reduction,
                    )
                )
                log_terms.append(
                    adaptive_log_magnitude_loss(
                        estimate_spectrum,
                        target_spectrum,
                        frame_mask,
                        frequency_mask,
                        sample_equal=self.sample_equal_reduction,
                    )
                    if self.adaptive_log_magnitude
                    else spectral_log1p_loss(
                        estimate_spectrum,
                        target_spectrum,
                        frame_mask,
                        frequency_mask,
                        sample_equal=self.sample_equal_reduction,
                    )
                )
        linear = torch.stack(linear_terms).mean()
        log = torch.stack(log_terms).mean()
        if self.objective_profile == "yamamoto_spectral_convergence_logmag_v1":
            return {
                "mr_stft_spectral_convergence": linear,
                "mr_stft_log_magnitude": log,
                "mr_stft": self.linear_weight * linear + self.log_weight * log,
            }
        return {
            "mr_stft_linear": linear,
            "mr_stft_log1p": log,
            "mr_stft": self.linear_weight * linear + self.log_weight * log,
        }

    def forward(
        self,
        estimate: Tensor,
        target: Tensor,
        lengths: Tensor | None = None,
        media_bandwidth_hz: Tensor | None = None,
    ) -> Tensor:
        return self.components(
            estimate,
            target,
            lengths,
            media_bandwidth_hz,
        )["mr_stft"]


class SourceAlignedMultiResolutionSTFTLoss(nn.Module):

    def __init__(
        self,
        resolutions: Sequence[Sequence[int]],
        *,
        spectral_convergence_weight: float = 1.0,
        adaptive_log_weight: float = 1.0,
        spectral_convergence_epsilon: float = 1.0e-8,
        spectral_convergence_relative_floor: float = 0.0,
        window_floor: float = 1.0e-3,
        k_weighting: bool = True,
        require_media_bandwidth: bool = True,
        compute_dtype: str = "float64",
        view_mode: str = "lr_ms_equal",
        scale_reduction: str = "equal_mean",
        view_reduction: str = "equal_mean",
        sample_reduction: str = "equal_mean",
    ) -> None:
        super().__init__()
        parsed = tuple(tuple(int(value) for value in item) for item in resolutions)
        if not parsed or any(len(item) != 3 for item in parsed):
            raise ValueError("source MR-STFT resolutions must be a non-empty list of (n_fft, hop, win_length) tuples")
        for n_fft, hop, win in parsed:
            if min(n_fft, hop, win) <= 0 or win > n_fft:
                raise ValueError(f"Invalid source MR-STFT resolution: {(n_fft, hop, win)}")
            if hop != n_fft // 4 or win != n_fft:
                raise ValueError("source MR-STFT requires 75% overlap and win_length=n_fft")
        if (
            spectral_convergence_weight < 0
            or adaptive_log_weight < 0
            or spectral_convergence_weight + adaptive_log_weight <= 0
        ):
            raise ValueError("The two source MR-STFT weights must be non-negative and at least one of them must be positive")
        if spectral_convergence_epsilon <= 0 or window_floor <= 0:
            raise ValueError("source MR-STFT epsilon and window floor must be positive")
        if (
            not math.isfinite(spectral_convergence_relative_floor)
            or spectral_convergence_relative_floor < 0.0
        ):
            raise ValueError("source MR-STFT relative floor must be finite and non-negative")
        if compute_dtype != "float64":
            raise ValueError("source MR-STFT must use float64")
        if view_mode != "lr_ms_equal":
            raise ValueError("source MR-STFT must use equal L/R and M/S weights")
        if {scale_reduction, view_reduction, sample_reduction} != {"equal_mean"}:
            raise ValueError("source MR-STFT reduction must use equal_mean")
        self.resolutions = parsed
        self.spectral_convergence_weight = float(spectral_convergence_weight)
        self.adaptive_log_weight = float(adaptive_log_weight)
        self.spectral_convergence_epsilon = float(spectral_convergence_epsilon)
        self.spectral_convergence_relative_floor = float(
            spectral_convergence_relative_floor
        )
        self.window_floor = float(window_floor)
        self.k_weighting = bool(k_weighting)
        self.require_media_bandwidth = bool(require_media_bandwidth)
        self.compute_dtype = torch.float64
        self.view_mode = view_mode

    def _stft(
        self,
        waveform: Tensor,
        *,
        lengths: Tensor,
        n_fft: int,
        hop: int,
        win: int,
    ) -> tuple[Tensor, Tensor]:
        values = waveform.to(dtype=self.compute_dtype)
        validate_audio(values, lengths)
        sample_mask = lengths_to_mask(
            lengths, values.shape[-1], device=values.device
        ).unsqueeze(1)
        values = values * sample_mask
        frame_lengths = torch.div(
            lengths + hop - 1,
            hop,
            rounding_mode="floor",
        )
        frames = int(frame_lengths.max().item())
        analysis_length = n_fft + (frames - 1) * hop
        if analysis_length < values.shape[-1]:
            values = values[..., :analysis_length]
        else:
            values = F.pad(values, (0, analysis_length - values.shape[-1]))
        window = torch.hann_window(
            win,
            periodic=True,
            dtype=self.compute_dtype,
            device=values.device,
        ).clamp_min(self.window_floor)
        spectrum = torch.stft(
            values.flatten(0, 1),
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        ).reshape(values.shape[0], values.shape[1], n_fft // 2 + 1, frames)
        frame_mask = lengths_to_mask(
            frame_lengths,
            frames,
            device=values.device,
        )
        return spectrum, frame_mask

    @staticmethod
    def _per_sample_channel_mean(values: Tensor, valid: Tensor) -> Tensor:
        expanded = torch.broadcast_to(valid, values.shape)
        denominator = expanded.sum(dim=(-2, -1))
        if bool((denominator == 0).any()):
            raise ValueError("source MR-STFT received empty sample or channel supervision")
        numerator = values.masked_fill(~expanded, 0.0).sum(dim=(-2, -1))
        return (numerator / denominator).mean()

    def _resolution_components(
        self,
        estimate: Tensor,
        target: Tensor,
        lengths: Tensor,
        media_bandwidth_hz: Tensor,
        *,
        n_fft: int,
        hop: int,
        win: int,
        relative_floor_reference: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        estimate_spectrum, frame_mask = self._stft(
            estimate,
            lengths=lengths,
            n_fft=n_fft,
            hop=hop,
            win=win,
        )
        target_spectrum, _ = self._stft(
            target,
            lengths=lengths,
            n_fft=n_fft,
            hop=hop,
            win=win,
        )
        if self.spectral_convergence_relative_floor > 0.0:
            if relative_floor_reference is None:

                raw_target_magnitude = target_spectrum.abs().masked_fill(
                    ~frame_mask[:, None, None, :],
                    0.0,
                )
                raw_lr_norm = torch.linalg.vector_norm(
                    raw_target_magnitude,
                    dim=(-2, -1),
                )
                relative_floor_reference = raw_lr_norm.square().sum(
                    dim=1
                ).sqrt() * math.sqrt(0.5)
            elif relative_floor_reference.shape != (target.shape[0],):
                raise ValueError("source MR-STFT relative floor reference must have shape [B]")
        if self.k_weighting:
            response = k_weighting_response(
                bins=estimate_spectrum.shape[-2],
                bin_hz=SAMPLE_RATE / n_fft,
                device=estimate.device,
                dtype=self.compute_dtype,
            )[None, None, :, None]
            estimate_spectrum = estimate_spectrum * response
            target_spectrum = target_spectrum * response
        frequency_mask = spectral_bandwidth_mask(
            media_bandwidth_hz.float(),
            bins=estimate_spectrum.shape[-2],
            bin_hz=SAMPLE_RATE / n_fft,
            device=estimate.device,
        )
        valid = frame_mask[:, None, None, :] & frequency_mask[:, None, :, None]
        estimate_magnitude = estimate_spectrum.abs()
        target_magnitude = target_spectrum.abs()
        expanded = torch.broadcast_to(valid, estimate_magnitude.shape)
        difference = (estimate_magnitude - target_magnitude).masked_fill(~expanded, 0.0)
        target_masked = target_magnitude.masked_fill(~expanded, 0.0)


        target_norm = torch.linalg.vector_norm(target_masked, dim=(-2, -1))
        if self.spectral_convergence_relative_floor > 0.0:
            assert relative_floor_reference is not None
            target_norm = torch.maximum(
                target_norm,
                self.spectral_convergence_relative_floor
                * relative_floor_reference[:, None],
            )
        convergence = (
            torch.linalg.vector_norm(difference, dim=(-2, -1))
            / (target_norm + self.spectral_convergence_epsilon)
        ).mean()

        count = expanded.sum(dim=(-2, -1), keepdim=True)
        if bool((count == 0).any()):
            raise ValueError("source adaptive-log received empty sample or channel supervision")

        def standard_deviation(values: Tensor) -> Tensor:
            mean = (
                values.masked_fill(~expanded, 0.0).sum(dim=(-2, -1), keepdim=True)
                / count
            )
            variance = (values - mean).square().masked_fill(~expanded, 0.0).sum(
                dim=(-2, -1), keepdim=True
            ) / count
            return variance.clamp_min(0.0).sqrt()

        sigma = (
            (
                standard_deviation(estimate_magnitude).square()
                + standard_deviation(target_magnitude).square()
            )
            .sqrt()
            .detach()
            .clamp_min(self.spectral_convergence_epsilon)
        )
        adaptive = self._per_sample_channel_mean(
            (
                torch.log1p(estimate_magnitude / sigma)
                - torch.log1p(target_magnitude / sigma)
            ).abs(),
            valid,
        )
        return convergence, adaptive, relative_floor_reference

    def components(
        self,
        estimate: Tensor,
        target: Tensor,
        lengths: Tensor,
        media_bandwidth_hz: Tensor | None,
    ) -> dict[str, Tensor]:
        if estimate.shape != target.shape:
            raise ValueError("source MR-STFT estimate and target shapes must match")
        if media_bandwidth_hz is None and self.require_media_bandwidth:
            raise ValueError("source MR-STFT requires media_bandwidth_hz")
        if media_bandwidth_hz is None:
            media_bandwidth_hz = torch.full(
                (estimate.shape[0],),
                SAMPLE_RATE / 2,
                device=estimate.device,
                dtype=torch.float32,
            )
        views = (
            (estimate, target),
            (lr_to_ms(estimate), lr_to_ms(target)),
        )
        convergence_terms: list[Tensor] = []
        adaptive_terms: list[Tensor] = []
        relative_floor_references: dict[tuple[int, int, int], Tensor] = {}
        for view_index, (view_estimate, view_target) in enumerate(views):
            for n_fft, hop, win in self.resolutions:
                convergence, adaptive, relative_floor_reference = (
                    self._resolution_components(
                        view_estimate,
                        view_target,
                        lengths,
                        media_bandwidth_hz,
                        n_fft=n_fft,
                        hop=hop,
                        win=win,
                        relative_floor_reference=(
                            relative_floor_references.get((n_fft, hop, win))
                            if view_index > 0
                            else None
                        ),
                    )
                )
                if relative_floor_reference is not None:
                    relative_floor_references[(n_fft, hop, win)] = (
                        relative_floor_reference
                    )
                convergence_terms.append(convergence)
                adaptive_terms.append(adaptive)
        convergence = torch.stack(convergence_terms).mean()
        adaptive = torch.stack(adaptive_terms).mean()
        total = (
            self.spectral_convergence_weight * convergence
            + self.adaptive_log_weight * adaptive
        )
        return {
            "source_spectral_convergence": convergence,
            "source_adaptive_log_magnitude": adaptive,
            "source_reconstruction": total,
            "mr_stft": total,
        }

    def forward(
        self,
        estimate: Tensor,
        target: Tensor,
        lengths: Tensor,
        media_bandwidth_hz: Tensor | None,
    ) -> Tensor:
        return self.components(
            estimate,
            target,
            lengths,
            media_bandwidth_hz,
        )["source_reconstruction"]


class SpectroStreamMixedScaleMelLoss(nn.Module):

    _FROZEN_RESOLUTIONS = (64, 128, 256, 512, 1024, 2048)

    def __init__(
        self,
        resolutions: Sequence[int] = _FROZEN_RESOLUTIONS,
        *,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = 64,
        f_min: float = 0.0,
        f_max: float = 24_000.0,
        log_floor: float = 1.0e-5,
        mel_scale: str = "htk",
        mel_norm: str = "area_hz",
        scale_reduction: str = "sum",
        stereo_reduction: str = "raw_lr_equal_mean",
        sample_reduction: str = "record_equal",
        compute_dtype: str = "float32",
        require_media_bandwidth: bool = True,
    ) -> None:
        super().__init__()
        parsed = tuple(int(value) for value in resolutions)
        if parsed != self._FROZEN_RESOLUTIONS:
            raise ValueError(
                "SpectroStream mixed-scale requires six scales: 64, 128, 256, 512, 1024, and 2048"
            )
        if sample_rate != SAMPLE_RATE:
            raise ValueError("SpectroStream mixed-scale requires 48 kHz")
        if n_mels != 64:
            raise ValueError("SpectroStream mixed-scale requires 64 Mel filter")
        if f_min != 0.0 or f_max != SAMPLE_RATE / 2:
            raise ValueError("SpectroStream mixed-scale requires a Mel band of [0, 24,000] Hz")
        if not math.isfinite(log_floor) or log_floor <= 0:
            raise ValueError("SpectroStream mixed-scale log floor must be finite and positive")
        if mel_scale != "htk" or mel_norm != "area_hz":
            raise ValueError("SpectroStream mixed-scale requires an HTK, Hz-area-normalized frontend")
        if scale_reduction != "sum":
            raise ValueError("SpectroStream mixed-scale requires cross-scale summation")
        if stereo_reduction != "raw_lr_equal_mean":
            raise ValueError("SpectroStream mixed-scale uses raw equal L/R weights")
        if sample_reduction != "record_equal":
            raise ValueError("SpectroStream mixed-scale uses equal per-record weights")
        if compute_dtype != "float32":
            raise ValueError("SpectroStream mixed-scale requires float32")
        self.resolutions = parsed
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.log_floor = float(log_floor)
        self.require_media_bandwidth = bool(require_media_bandwidth)
        self.compute_dtype = torch.float32
        for n_fft in self.resolutions:
            self.register_buffer(
                f"mel_bank_{n_fft}",
                self._build_mel_bank(n_fft),
                persistent=True,
            )

    def _build_mel_bank(self, n_fft: int) -> Tensor:

        dtype = self.compute_dtype
        lower_mel = 2595.0 * math.log10(1.0 + self.f_min / 700.0)
        upper_mel = 2595.0 * math.log10(1.0 + self.f_max / 700.0)
        mel_points = torch.linspace(
            lower_mel,
            upper_mel,
            self.n_mels + 2,
            dtype=dtype,
        )
        hz_points = 700.0 * (torch.pow(10.0, mel_points / 2595.0) - 1.0)
        frequencies = torch.arange(n_fft // 2 + 1, dtype=dtype) * (
            self.sample_rate / n_fft
        )
        lower = hz_points[:-2, None]
        center = hz_points[1:-1, None]
        upper = hz_points[2:, None]
        rising = (frequencies[None, :] - lower) / (center - lower)
        falling = (upper - frequencies[None, :]) / (upper - center)
        triangle = torch.minimum(rising, falling).clamp_min(0.0)
        area = 2.0 / (upper - lower)
        bank = triangle * area
        if bank.shape != (self.n_mels, n_fft // 2 + 1):
            raise RuntimeError("SpectroStream mixed-scale Mel bank shape mismatch")
        if not torch.isfinite(bank).all():
            raise RuntimeError("SpectroStream mixed-scale Mel bank contains NaN/Inf")
        return bank

    def mel_bank(self, n_fft: int) -> Tensor:
        if n_fft not in self.resolutions:
            raise ValueError(f"Unsupported mixed-scale n_fft={n_fft}")
        return getattr(self, f"mel_bank_{n_fft}")

    def _stft(
        self,
        waveform: Tensor,
        *,
        lengths: Tensor,
        n_fft: int,
    ) -> tuple[Tensor, Tensor]:
        values = waveform.to(dtype=self.compute_dtype)
        validate_audio(values, lengths)
        sample_mask = lengths_to_mask(
            lengths,
            values.shape[-1],
            device=values.device,
        ).unsqueeze(1)
        values = values * sample_mask
        hop = n_fft // 4
        frame_lengths = torch.div(lengths + hop - 1, hop, rounding_mode="floor")
        frames = int(frame_lengths.max().item())
        analysis_length = n_fft + (frames - 1) * hop
        if analysis_length < values.shape[-1]:
            values = values[..., :analysis_length]
        else:
            values = F.pad(values, (0, analysis_length - values.shape[-1]))
        window = torch.hann_window(
            n_fft,
            periodic=True,
            dtype=self.compute_dtype,
            device=values.device,
        )
        spectrum = torch.stft(
            values.flatten(0, 1),
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        ).reshape(values.shape[0], values.shape[1], n_fft // 2 + 1, frames)
        frame_mask = lengths_to_mask(frame_lengths, frames, device=values.device)
        return spectrum, frame_mask

    @staticmethod
    def _record_lr_equal_mean(values: Tensor, frame_mask: Tensor) -> Tensor:
        if values.ndim != 4 or values.shape[1] != AUDIO_CHANNELS:
            raise ValueError("mixed-scale reduction requires shape [B, 2, Mel, T]")
        valid = frame_mask[:, None, None, :].expand_as(values)
        denominator = valid.sum(dim=(-2, -1))
        if bool((denominator == 0).any()):
            raise ValueError("mixed-scale received empty record or channel supervision")
        per_record_channel = (
            values.masked_fill(~valid, 0.0).sum(dim=(-2, -1)) / denominator
        )
        return per_record_channel.mean(dim=1).mean(dim=0)

    def components(
        self,
        estimate: Tensor,
        target: Tensor,
        lengths: Tensor,
        media_bandwidth_hz: Tensor | None,
    ) -> dict[str, Tensor]:
        if estimate.shape != target.shape:
            raise ValueError("mixed-scale estimate and target shapes must match")
        resolved_lengths = validate_audio(estimate, lengths)
        validate_audio(target, resolved_lengths)
        if media_bandwidth_hz is None:
            if self.require_media_bandwidth:
                raise ValueError("mixed-scale requires media_bandwidth_hz")
            media_bandwidth_hz = torch.full(
                (estimate.shape[0],),
                self.f_max,
                dtype=torch.float32,
                device=estimate.device,
            )

        result: dict[str, Tensor] = {}
        linear_terms: list[Tensor] = []
        log_squared_terms: list[Tensor] = []
        weighted_log_terms: list[Tensor] = []
        scale_terms: list[Tensor] = []
        for n_fft in self.resolutions:
            estimate_spectrum, frame_mask = self._stft(
                estimate,
                lengths=resolved_lengths,
                n_fft=n_fft,
            )
            target_spectrum, _ = self._stft(
                target,
                lengths=resolved_lengths,
                n_fft=n_fft,
            )
            frequency_mask = spectral_bandwidth_mask(
                media_bandwidth_hz.float(),
                bins=n_fft // 2 + 1,
                bin_hz=self.sample_rate / n_fft,
                device=estimate.device,
            )[:, None, :, None]
            estimate_magnitude = estimate_spectrum.abs() * frequency_mask
            target_magnitude = target_spectrum.abs() * frequency_mask
            bank = self.mel_bank(n_fft).to(device=estimate.device)
            estimate_mel = torch.einsum("mf,bcft->bcmt", bank, estimate_magnitude)
            target_mel = torch.einsum("mf,bcft->bcmt", bank, target_magnitude)
            linear = self._record_lr_equal_mean(
                (estimate_mel - target_mel).abs(),
                frame_mask,
            )
            log_squared = self._record_lr_equal_mean(
                (
                    estimate_mel.clamp_min(self.log_floor).log()
                    - target_mel.clamp_min(self.log_floor).log()
                ).square(),
                frame_mask,
            )
            alpha = math.sqrt(n_fft / 2.0)
            weighted_log = log_squared * alpha
            scale_total = linear + weighted_log
            result[f"mixed_scale_linear_mel_s{n_fft}"] = linear
            result[f"mixed_scale_log_mel_squared_s{n_fft}"] = log_squared
            result[f"mixed_scale_log_mel_squared_weighted_s{n_fft}"] = weighted_log
            result[f"mixed_scale_s{n_fft}"] = scale_total
            linear_terms.append(linear)
            log_squared_terms.append(log_squared)
            weighted_log_terms.append(weighted_log)
            scale_terms.append(scale_total)

        linear_sum = torch.stack(linear_terms).sum()
        log_squared_sum = torch.stack(log_squared_terms).sum()
        weighted_log_sum = torch.stack(weighted_log_terms).sum()
        total = torch.stack(scale_terms).sum()
        result.update(
            {
                "mixed_scale_linear_mel": linear_sum,
                "mixed_scale_log_mel_squared": log_squared_sum,
                "mixed_scale_log_mel_squared_weighted": weighted_log_sum,
                "mixed_scale_spectral": total,
            }
        )
        return result

    def forward(
        self,
        estimate: Tensor,
        target: Tensor,
        lengths: Tensor,
        media_bandwidth_hz: Tensor | None,
    ) -> Tensor:
        return self.components(
            estimate,
            target,
            lengths,
            media_bandwidth_hz,
        )["mixed_scale_spectral"]


def _hz_to_mel(frequency: Tensor) -> Tensor:
    return 2595.0 * torch.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel: Tensor) -> Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def mel_filter_bank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
    device: torch.device | str | None = None,
) -> Tensor:

    maximum = sample_rate / 2 if f_max is None else float(f_max)
    if not 0 <= f_min < maximum <= sample_rate / 2:
        raise ValueError("Mel invalid frequency range")
    mel_edges = torch.linspace(
        _hz_to_mel(torch.tensor(float(f_min))),
        _hz_to_mel(torch.tensor(maximum)),
        n_mels + 2,
        dtype=torch.float32,
        device=device,
    )
    hz_edges = _mel_to_hz(mel_edges)
    frequencies = torch.linspace(
        0.0,
        sample_rate / 2,
        n_fft // 2 + 1,
        dtype=torch.float32,
        device=device,
    )
    lower = hz_edges[:-2, None]
    center = hz_edges[1:-1, None]
    upper = hz_edges[2:, None]
    rising = (frequencies - lower) / (center - lower).clamp_min(EPSILON)
    falling = (upper - frequencies) / (upper - center).clamp_min(EPSILON)
    filters = torch.minimum(rising, falling).clamp_min(0.0)

    filters *= 2.0 / (upper - lower).clamp_min(EPSILON)
    return filters


class MelSpectralLoss(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float | None = 24_000.0,
        linear_weight: float = 1.0,
        log_weight: float = 1.0,
        sample_rms_normalization: bool = False,
        sample_rms_floor: float = 1.0e-4,
        sample_equal_reduction: bool = False,
    ) -> None:
        super().__init__()
        bank = mel_filter_bank(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self.register_buffer("bank", bank, persistent=False)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.linear_weight = float(linear_weight)
        self.log_weight = float(log_weight)
        self.sample_rms_normalization = bool(sample_rms_normalization)
        self.sample_rms_floor = float(sample_rms_floor)
        self.sample_equal_reduction = bool(sample_equal_reduction)
        if not math.isfinite(self.sample_rms_floor) or self.sample_rms_floor <= 0:
            raise ValueError("sample_rms_floor must be finite and positive")

    def components(
        self,
        estimate: Tensor,
        target: Tensor,
        lengths: Tensor | None = None,
        media_bandwidth_hz: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if estimate.shape != target.shape:
            raise ValueError("Mel estimate and target shapes must match")
        estimate_spectrum, mask = _waveform_stft(
            estimate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            lengths=lengths,
        )
        target_spectrum, _ = _waveform_stft(
            target,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            lengths=lengths,
        )
        frequency_mask = (
            None
            if media_bandwidth_hz is None
            else spectral_bandwidth_mask(
                media_bandwidth_hz,
                bins=estimate_spectrum.shape[-2],
                bin_hz=SAMPLE_RATE / self.n_fft,
                device=estimate_spectrum.device,
            )
        )
        if self.sample_rms_normalization:
            scale = _spectral_sample_rms_scale(
                target_spectrum,
                mask=mask,
                frequency_mask=frequency_mask,
                floor=self.sample_rms_floor,
            )
            estimate_spectrum = estimate_spectrum / scale[:, None, None, None]
            target_spectrum = target_spectrum / scale[:, None, None, None]
        bank = self.bank.to(estimate.device)
        estimate_mel = torch.einsum(
            "mf,bcft->bcmt", bank, safe_magnitude(estimate_spectrum)
        )
        target_mel = torch.einsum(
            "mf,bcft->bcmt", bank, safe_magnitude(target_spectrum)
        )
        time_mask = _time_mask(mask, estimate_mel)
        if frequency_mask is not None:
            mel_valid = torch.einsum(
                "mf,bf->bm", bank.ne(0).float(), frequency_mask.float()
            ) >= bank.ne(0).sum(dim=1).unsqueeze(0)
            mel_mask = mel_valid[:, None, :, None]
            time_mask = mel_mask if time_mask is None else time_mask & mel_mask
        linear = masked_reduce(
            (estimate_mel - target_mel).abs(),
            time_mask,
            sample_equal=self.sample_equal_reduction,
        ).value
        log = masked_reduce(
            (torch.log1p(estimate_mel) - torch.log1p(target_mel)).abs(),
            time_mask,
            sample_equal=self.sample_equal_reduction,
        ).value
        return {
            "mel_linear": linear,
            "mel_log1p": log,
            "mel": self.linear_weight * linear + self.log_weight * log,
        }

    def forward(
        self,
        estimate: Tensor,
        target: Tensor,
        lengths: Tensor | None = None,
        media_bandwidth_hz: Tensor | None = None,
    ) -> Tensor:
        return self.components(
            estimate,
            target,
            lengths,
            media_bandwidth_hz,
        )["mel"]


class MultiResolutionMelSpectralLoss(nn.Module):

    def __init__(
        self,
        resolutions: Sequence[Sequence[int]],
        *,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = 64,
        f_min: float = 0.0,
        f_max: float | None = 24_000.0,
        linear_weight: float = 1.0,
        log_weight: float = 1.0,
        sample_rms_normalization: bool = False,
        sample_rms_floor: float = 1.0e-4,
        sample_equal_reduction: bool = False,
    ) -> None:
        super().__init__()
        parsed = tuple(tuple(int(value) for value in item) for item in resolutions)
        if not parsed or any(len(item) != 3 for item in parsed):
            raise ValueError("Mel resolutions must be a non-empty list of (n_fft, hop, win_length) tuples")
        self.losses = nn.ModuleList(
            [
                MelSpectralLoss(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop,
                    win_length=win,
                    n_mels=n_mels,
                    f_min=f_min,
                    f_max=f_max,
                    linear_weight=linear_weight,
                    log_weight=log_weight,
                    sample_rms_normalization=sample_rms_normalization,
                    sample_rms_floor=sample_rms_floor,
                    sample_equal_reduction=sample_equal_reduction,
                )
                for n_fft, hop, win in parsed
            ]
        )
        self.resolutions = parsed
        self.linear_weight = float(linear_weight)
        self.log_weight = float(log_weight)

    def components(
        self,
        estimate: Tensor,
        target: Tensor,
        lengths: Tensor | None = None,
        media_bandwidth_hz: Tensor | None = None,
    ) -> dict[str, Tensor]:
        values = [
            loss.components(estimate, target, lengths, media_bandwidth_hz)
            for loss in self.losses
        ]
        linear = torch.stack([value["mel_linear"] for value in values]).mean()
        log = torch.stack([value["mel_log1p"] for value in values]).mean()
        return {
            "mel_linear": linear,
            "mel_log1p": log,
            "mel": self.linear_weight * linear + self.log_weight * log,
        }

    def forward(
        self,
        estimate: Tensor,
        target: Tensor,
        lengths: Tensor | None = None,
        media_bandwidth_hz: Tensor | None = None,
    ) -> Tensor:
        return self.components(
            estimate,
            target,
            lengths,
            media_bandwidth_hz,
        )["mel"]


def diagonal_gaussian_kl(
    mean: Tensor,
    logvar: Tensor,
    mask: Tensor | None = None,
    *,
    start_dim: int = 0,
    reduction: str = "element_mean",
    sample_equal: bool = False,
) -> Tensor:
    mean_term, variance_term = diagonal_gaussian_kl_components(
        mean,
        logvar,
        mask,
        start_dim=start_dim,
        reduction=reduction,
        sample_equal=sample_equal,
    )
    return mean_term + variance_term


def diagonal_gaussian_kl_components(
    mean: Tensor,
    logvar: Tensor,
    mask: Tensor | None = None,
    *,
    start_dim: int = 0,
    reduction: str = "element_mean",
    sample_equal: bool = False,
) -> tuple[Tensor, Tensor]:

    if mean.shape != logvar.shape:
        raise ValueError("KL mean and logvar shapes must match")
    if not 0 <= start_dim < mean.shape[-1]:
        raise ValueError("KL start_dim exceeds latent dimension")
    if reduction not in {"element_mean", "channel_sum_mean"}:
        raise ValueError("KL reduction must be element_mean/channel_sum_mean")
    work_mean = mean[..., start_dim:].float()
    work_logvar = logvar[..., start_dim:].float().clamp(-30.0, 20.0)
    mean_values = 0.5 * work_mean.square()
    variance_values = 0.5 * (torch.exp(work_logvar) - 1.0 - work_logvar)

    def reduce(values: Tensor) -> Tensor:
        if reduction == "channel_sum_mean":
            return masked_reduce(
                values.sum(dim=-1),
                mask,
                sample_equal=sample_equal,
            ).value
        expanded = None if mask is None else mask.unsqueeze(-1)
        return masked_reduce(values, expanded, sample_equal=sample_equal).value

    return reduce(mean_values), reduce(variance_values)


def lr_to_ms(values: Tensor) -> Tensor:
    if values.shape[1] != AUDIO_CHANNELS:
        raise ValueError("LR/MS input must have two channels")
    scale = math.sqrt(0.5)
    middle = (values[:, 0] + values[:, 1]) * scale
    side = (values[:, 0] - values[:, 1]) * scale
    return torch.stack((middle, side), dim=1)


def lr_ms_magnitude_loss(
    estimate: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    frequency_mask: Tensor | None = None,
    *,
    sample_equal: bool = False,
) -> Tensor:
    estimate_ms = safe_magnitude(lr_to_ms(estimate))
    target_ms = safe_magnitude(lr_to_ms(target))
    ms = (estimate_ms - target_ms).abs()
    return masked_reduce(
        ms,
        _spectral_mask(mask, frequency_mask, ms),
        sample_equal=sample_equal,
    ).value


def _unit(values: Tensor, epsilon: float = EPSILON) -> tuple[Tensor, Tensor]:
    magnitude = safe_magnitude(values, epsilon)
    valid = magnitude > math.sqrt(epsilon)
    unit = values / magnitude.clamp_min(math.sqrt(epsilon))
    return unit.to(torch.complex64), valid


def circular_distance(
    estimate_unit: Tensor,
    target_unit: Tensor,
    *,
    absolute: bool = False,
    valid: Tensor | None = None,
) -> Tensor:
    relative = estimate_unit * target_unit.conj()
    if absolute:
        if valid is not None:
            if valid.dtype != torch.bool or valid.shape != relative.shape:
                raise ValueError("circular distance valid mask must match input shape")


            relative = torch.where(valid, relative, torch.ones_like(relative))
        return torch.atan2(relative.imag.float(), relative.real.float()).abs()
    return (1.0 - relative.real.float()).clamp_min(0.0)


def unit_phasor_if_gd_loss(
    estimate: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    frequency_mask: Tensor | None = None,
    *,
    magnitude_weighted: bool = False,
    sample_equal: bool = False,
    objective: str = "unit_phasor_cosine_v1",
) -> dict[str, Tensor]:

    if objective not in {"unit_phasor_cosine_v1", "ear_wrapped_l1_v1"}:
        raise ValueError("IF/GD objective is invalid")
    absolute = objective == "ear_wrapped_l1_v1"

    estimate_unit, estimate_valid = _unit(estimate)
    target_unit, target_valid = _unit(target)
    valid = estimate_valid & target_valid
    if mask is not None:
        valid = valid & _time_mask(mask, valid)
    if frequency_mask is not None:
        valid = valid & _spectral_mask(None, frequency_mask, valid)
    phase_weights = (
        (safe_magnitude(estimate) * safe_magnitude(target)).sqrt().detach()
        if magnitude_weighted
        else None
    )

    estimate_if = estimate_unit[..., 1:] * estimate_unit[..., :-1].conj()
    target_if = target_unit[..., 1:] * target_unit[..., :-1].conj()
    if_valid = valid[..., 1:] & valid[..., :-1]
    if_weights = (
        phase_weights[..., 1:] * phase_weights[..., :-1]
        if phase_weights is not None
        else None
    )
    if_reduced = masked_reduce(
        circular_distance(
            estimate_if,
            target_if,
            absolute=absolute,
            valid=if_valid,
        ),
        if_valid,
        weights=if_weights,
        sample_equal=sample_equal,
    )

    estimate_gd = estimate_unit[..., 1:, :] * estimate_unit[..., :-1, :].conj()
    target_gd = target_unit[..., 1:, :] * target_unit[..., :-1, :].conj()
    gd_valid = valid[..., 1:, :] & valid[..., :-1, :]
    gd_weights = (
        phase_weights[..., 1:, :] * phase_weights[..., :-1, :]
        if phase_weights is not None
        else None
    )
    gd_reduced = masked_reduce(
        circular_distance(
            estimate_gd,
            target_gd,
            absolute=absolute,
            valid=gd_valid,
        ),
        gd_valid,
        weights=gd_weights,
        sample_equal=sample_equal,
    )
    return {
        "if": if_reduced.value,
        "gd": gd_reduced.value,
        "if_gd": if_reduced.value + gd_reduced.value,
        "if_numerator": if_reduced.numerator,
        "if_denominator": if_reduced.denominator,
        "gd_numerator": gd_reduced.numerator,
        "gd_denominator": gd_reduced.denominator,
    }


def _ipd(values: Tensor) -> tuple[Tensor, Tensor]:
    unit, valid = _unit(values)
    both = valid[:, 0] & valid[:, 1]
    return unit[:, 0] * unit[:, 1].conj(), both


def ipd_cosine_similarity(
    estimate: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    frequency_mask: Tensor | None = None,
    *,
    magnitude_weighted: bool = False,
    sample_equal: bool = False,
) -> Tensor:

    estimate_ipd, estimate_valid = _ipd(estimate)
    target_ipd, target_valid = _ipd(target)
    valid = estimate_valid & target_valid
    if mask is not None:
        valid = valid & _time_mask(mask, valid)
    if frequency_mask is not None:
        if frequency_mask.shape != valid.shape[:2]:
            raise ValueError("IPD cosine frequency_mask must have shape [B, F]")
        valid = valid & frequency_mask.unsqueeze(-1)
    coherence = (estimate_ipd * target_ipd.conj()).real.float()
    weights = (
        (
            safe_magnitude(estimate[:, 0])
            * safe_magnitude(estimate[:, 1])
            * safe_magnitude(target[:, 0])
            * safe_magnitude(target[:, 1])
        )
        .sqrt()
        .detach()
        if magnitude_weighted
        else None
    )
    if sample_equal:
        effective = valid.to(coherence.dtype)
        if weights is not None:
            effective = effective * weights.float()
        numerator = (coherence * effective).sum(dim=(1, 2))
        denominator = effective.sum(dim=(1, 2))
        return torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1.0),
            torch.ones_like(numerator),
        ).mean()
    reduced = masked_reduce(coherence, valid, weights=weights)
    return torch.where(
        reduced.denominator > 0,
        reduced.value,
        torch.ones_like(reduced.value),
    )


def ipd_cosine_loss(
    estimate: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    frequency_mask: Tensor | None = None,
    *,
    magnitude_weighted: bool = False,
    sample_equal: bool = False,
) -> Tensor:
    return 1.0 - ipd_cosine_similarity(
        estimate,
        target,
        mask,
        frequency_mask,
        magnitude_weighted=magnitude_weighted,
        sample_equal=sample_equal,
    )


def ccpc(
    estimate: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    frequency_mask: Tensor | None = None,
) -> Tensor:

    return ipd_cosine_similarity(estimate, target, mask, frequency_mask)


def ccpc_loss(
    estimate: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    frequency_mask: Tensor | None = None,
) -> Tensor:

    return ipd_cosine_loss(estimate, target, mask, frequency_mask)


def absolute_circular_ipd_error(
    estimate: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    frequency_mask: Tensor | None = None,
    *,
    magnitude_weighted: bool = False,
    normalize_by_pi: bool = True,
    sample_equal: bool = False,
) -> Tensor:
    estimate_ipd, estimate_valid = _ipd(estimate)
    target_ipd, target_valid = _ipd(target)
    valid = estimate_valid & target_valid
    if mask is not None:
        valid = valid & _time_mask(mask, valid)
    if frequency_mask is not None:
        if frequency_mask.shape != valid.shape[:2]:
            raise ValueError("IPD frequency_mask must have shape [B, F]")
        valid = valid & frequency_mask.unsqueeze(-1)
    relative = estimate_ipd * target_ipd.conj()

    relative = torch.where(
        valid,
        relative,
        torch.ones_like(relative),
    )
    values = torch.atan2(relative.imag.float(), relative.real.float()).abs()
    if normalize_by_pi:
        values = values / math.pi
    weights = (
        (
            safe_magnitude(estimate[:, 0])
            * safe_magnitude(estimate[:, 1])
            * safe_magnitude(target[:, 0])
            * safe_magnitude(target[:, 1])
        )
        .sqrt()
        .detach()
        if magnitude_weighted
        else None
    )
    return masked_reduce(
        values,
        valid,
        weights=weights,
        sample_equal=sample_equal,
    ).value


def spectral_pan_error(
    estimate: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    frequency_mask: Tensor | None = None,
    *,
    energy_floor: float = 1.0e-8,
    sample_equal: bool = False,
) -> Tensor:
    estimate_power = safe_magnitude(estimate).square()
    target_power = safe_magnitude(target).square()
    estimate_total = estimate_power.sum(dim=1)
    target_total = target_power.sum(dim=1)
    estimate_pan = estimate_power[:, 0] / estimate_total.clamp_min(energy_floor)
    target_pan = target_power[:, 0] / target_total.clamp_min(energy_floor)
    valid = target_total > energy_floor
    if mask is not None:
        valid = valid & _time_mask(mask, valid)
    if frequency_mask is not None:
        if frequency_mask.shape != valid.shape[:2]:
            raise ValueError("pan frequency_mask must have shape [B, F]")
        valid = valid & frequency_mask.unsqueeze(-1)
    return masked_reduce(
        (estimate_pan - target_pan).abs(),
        valid,
        sample_equal=sample_equal,
    ).value


def lsgan_discriminator_loss(
    real_logits: Sequence[Tensor],
    fake_logits: Sequence[Tensor],
    *,
    family_sizes: Sequence[int] | None = None,
    family_weights: Sequence[float] | None = None,
    compute_dtype: torch.dtype = torch.float32,
) -> Tensor:
    if len(real_logits) != len(fake_logits) or not real_logits:
        raise ValueError("LSGAN real/fake logits must have the same non-zero length")
    if compute_dtype not in {torch.float32, torch.float64}:
        raise ValueError("LSGAN compute_dtype must be float32 or float64")
    terms = [
        0.5
        * (
            (real.to(compute_dtype) - 1.0).square().mean()
            + fake.to(compute_dtype).square().mean()
        )
        for real, fake in zip(real_logits, fake_logits)
    ]
    return _reduce_discriminator_families(
        terms,
        family_sizes=family_sizes,
        family_weights=family_weights,
    )


def lsgan_generator_loss(
    fake_logits: Sequence[Tensor],
    family_sizes: Sequence[int] | None = None,
    *,
    family_weights: Sequence[float] | None = None,
    compute_dtype: torch.dtype = torch.float32,
) -> Tensor:
    if not fake_logits:
        raise ValueError("LSGAN fake logits must be non-empty")
    if compute_dtype not in {torch.float32, torch.float64}:
        raise ValueError("LSGAN compute_dtype must be float32 or float64")
    terms = [
        (value.to(compute_dtype) - 1.0).square().mean()
        for value in fake_logits
    ]
    return _reduce_discriminator_families(
        terms,
        family_sizes=family_sizes,
        family_weights=family_weights,
    )


def _reduce_discriminator_families(
    terms: Sequence[Tensor],
    *,
    family_sizes: Sequence[int] | None,
    family_weights: Sequence[float] | None = None,
) -> Tensor:

    if not terms:
        raise ValueError("discriminator loss terms must be non-empty")
    if family_sizes is None:
        if family_weights is not None:
            raise ValueError("family_weights requires family_sizes as well")
        return torch.stack(tuple(terms)).mean()
    parsed = tuple(int(value) for value in family_sizes)
    if not parsed or any(value <= 0 for value in parsed) or sum(parsed) != len(terms):
        raise ValueError("discriminator family_sizes and term count mismatch")
    offsets = [0]
    for size in parsed:
        offsets.append(offsets[-1] + size)
    families = [
        torch.stack(tuple(terms[start:stop])).mean()
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
    ]
    if family_weights is not None:
        weights = tuple(float(value) for value in family_weights)
        if (
            len(weights) != len(families)
            or any(not math.isfinite(value) or value < 0.0 for value in weights)
            or not any(value > 0.0 for value in weights)
        ):
            raise ValueError(
                "family_weights must match the family count, be non-negative, and contain at least one positive value"
            )
        return torch.stack(
            tuple(family * weight for family, weight in zip(families, weights, strict=True))
        ).sum()
    return torch.stack(families).mean()


def feature_matching_loss(
    real_features: Sequence[Sequence[Tensor]],
    fake_features: Sequence[Sequence[Tensor]],
    family_sizes: Sequence[int] | None = None,
    *,
    family_weights: Sequence[float] | None = None,
    compute_dtype: torch.dtype = torch.float32,
) -> Tensor:
    if len(real_features) != len(fake_features):
        raise ValueError("FM real and fake scale counts differ")
    if compute_dtype not in {torch.float32, torch.float64}:
        raise ValueError("FM compute_dtype must be float32 or float64")
    scale_terms: list[Tensor] = []
    flat_terms: list[Tensor] = []
    for real_scale, fake_scale in zip(real_features, fake_features):
        if len(real_scale) != len(fake_scale):
            raise ValueError("FM feature layer counts differ")
        layer_terms = [
            (
                real.detach().to(compute_dtype)
                - fake.to(compute_dtype)
            )
            .abs()
            .mean()
            for real, fake in zip(real_scale, fake_scale, strict=True)
        ]
        flat_terms.extend(layer_terms)
        if layer_terms:
            scale_terms.append(torch.stack(layer_terms).mean())
    if not flat_terms:
        device = (
            fake_features[0][0].device
            if fake_features and fake_features[0]
            else torch.device("cpu")
        )
        return torch.zeros((), device=device, dtype=compute_dtype)
    if family_sizes is None:
        if family_weights is not None:
            raise ValueError("FM family_weights requires family_sizes as well")
        return torch.stack(flat_terms).mean()
    return _reduce_discriminator_families(
        scale_terms,
        family_sizes=family_sizes,
        family_weights=family_weights,
    )


@dataclass(frozen=True)
class RenderLossConfig:
    k_weighting: bool = True
    adaptive_log_magnitude: bool = True
    sample_rms_normalization: bool = False
    sample_rms_floor: float = 1.0e-4
    sample_equal_reduction: bool = False
    spectrum_complex: float = 1.0
    spectrum_linear: float = 1.0
    spectrum_log1p: float = 1.0
    lr_ms: float = 1.0
    if_gd: float = 0.0
    ccpc: float = 1.0
    absolute_ipd: float = 1.0
    spectral_pan: float = 1.0
    phase_weighting: str = "binary"
    if_gd_objective: str = "unit_phasor_cosine_v1"
    kl: float = 1.0e-6
    kl_start_dim: int = 0
    kl_reduction: str = "element_mean"
    kl_mean_scale: float = 1.0
    kl_variance_scale: float = 1.0


class SpectralReconstructionLoss(nn.Module):

    def __init__(self, config: RenderLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or RenderLossConfig()
        if (
            not math.isfinite(self.config.sample_rms_floor)
            or self.config.sample_rms_floor <= 0
        ):
            raise ValueError("sample_rms_floor must be finite and positive")
        if self.config.kl_start_dim < 0:
            raise ValueError("kl_start_dim must not be negative")
        if self.config.phase_weighting not in {"binary", "magnitude"}:
            raise ValueError("phase_weighting must be binary or magnitude")
        if self.config.if_gd_objective not in {
            "unit_phasor_cosine_v1",
            "ear_wrapped_l1_v1",
        }:
            raise ValueError("if_gd_objective is invalid")
        if (
            self.config.if_gd_objective == "ear_wrapped_l1_v1"
            and self.config.phase_weighting != "binary"
        ):
            raise ValueError("EAR wrapped-L1 IF/GD only supports binary valid-bin reduction")
        if self.config.kl_reduction not in {
            "element_mean",
            "channel_sum_mean",
        }:
            raise ValueError("kl_reduction is invalid")
        for name, value in (
            ("kl_mean_scale", self.config.kl_mean_scale),
            ("kl_variance_scale", self.config.kl_variance_scale),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def forward(
        self,
        estimate: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
        *,
        posterior_mean: Tensor | None = None,
        posterior_logvar: Tensor | None = None,
        latent_mask: Tensor | None = None,
        media_bandwidth_hz: Tensor | None = None,
        magnitude_max_hz: Tensor | None = None,
        phase_max_hz: Tensor | None = None,
        stereo_max_hz: Tensor | None = None,
        magnitude_mode_mask: Tensor | None = None,
        phase_mode_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        sample_equal = self.config.sample_equal_reduction
        magnitude_limit = (
            magnitude_max_hz if magnitude_max_hz is not None else media_bandwidth_hz
        )
        phase_limit = phase_max_hz if phase_max_hz is not None else media_bandwidth_hz
        stereo_limit = (
            stereo_max_hz if stereo_max_hz is not None else media_bandwidth_hz
        )

        def make_frequency_mask(limit: Tensor | None) -> Tensor | None:
            return (
                None
                if limit is None
                else spectral_bandwidth_mask(
                    limit,
                    bins=estimate.shape[-2],
                    bin_hz=SAMPLE_RATE / STFT_N_FFT,
                    device=estimate.device,
                )
            )

        magnitude_mask = make_frequency_mask(magnitude_limit)
        phase_mask = make_frequency_mask(phase_limit)
        stereo_mask = make_frequency_mask(stereo_limit)

        def apply_mode_mask(
            bandwidth_mask: Tensor | None,
            mode_mask: Tensor | None,
            *,
            name: str,
        ) -> Tensor | None:
            if mode_mask is None:
                return bandwidth_mask
            if (
                mode_mask.shape != (estimate.shape[-2],)
                or mode_mask.dtype != torch.bool
            ):
                raise ValueError(f"{name} must be bool with shape [F]")
            expanded = mode_mask.to(device=estimate.device).unsqueeze(0)
            return expanded if bandwidth_mask is None else bandwidth_mask & expanded

        magnitude_mask = apply_mode_mask(
            magnitude_mask,
            magnitude_mode_mask,
            name="magnitude_mode_mask",
        )
        phase_mask = apply_mode_mask(
            phase_mask,
            phase_mode_mask,
            name="phase_mode_mask",
        )
        complex_mask = (
            phase_mask
            if magnitude_mask is None
            else magnitude_mask
            if phase_mask is None
            else magnitude_mask & phase_mask
        )
        frequency_mask = None if magnitude_mask is None else magnitude_mask
        weighted_estimate = estimate
        weighted_target = target
        if self.config.k_weighting:
            response = k_weighting_response(
                bins=estimate.shape[-2],
                bin_hz=SAMPLE_RATE / STFT_N_FFT,
                device=estimate.device,
            )[None, None, :, None]
            weighted_estimate = estimate * response
            weighted_target = target * response
        if self.config.sample_rms_normalization:
            valid = _spectral_mask(mask, magnitude_mask, weighted_target)
            if valid is None:
                valid = torch.ones_like(weighted_target.real, dtype=torch.bool)
            else:
                valid = valid.expand_as(weighted_target.real)
            count = valid.sum(dim=(1, 2, 3))


            empty = count == 0
            power = weighted_target.abs().square().masked_fill(~valid, 0.0).sum(
                dim=(1, 2, 3)
            ) / count.clamp_min(1)
            scale = power.sqrt().clamp_min(float(self.config.sample_rms_floor))
            scale = torch.where(empty, torch.ones_like(scale), scale)
            weighted_estimate = weighted_estimate / scale[:, None, None, None]
            weighted_target = weighted_target / scale[:, None, None, None]
        linear = spectral_linear_loss(
            weighted_estimate,
            weighted_target,
            mask,
            frequency_mask,
            sample_equal=sample_equal,
        )
        complex_l1 = spectral_complex_l1_loss(
            weighted_estimate,
            weighted_target,
            mask,
            complex_mask,
            sample_equal=sample_equal,
        )
        log1p = (
            adaptive_log_magnitude_loss(
                weighted_estimate,
                weighted_target,
                mask,
                frequency_mask,
                sample_equal=sample_equal,
            )
            if self.config.adaptive_log_magnitude
            else spectral_log1p_loss(
                weighted_estimate,
                weighted_target,
                mask,
                frequency_mask,
                sample_equal=sample_equal,
            )
        )
        lr_ms = lr_ms_magnitude_loss(
            weighted_estimate,
            weighted_target,
            mask,
            stereo_mask,
            sample_equal=sample_equal,
        )
        magnitude_weighted_phase = self.config.phase_weighting == "magnitude"
        phase = unit_phasor_if_gd_loss(
            estimate,
            target,
            mask,
            phase_mask,
            magnitude_weighted=magnitude_weighted_phase,
            sample_equal=sample_equal,
            objective=self.config.if_gd_objective,
        )
        coherence = ipd_cosine_loss(
            estimate,
            target,
            mask,
            stereo_mask,
            magnitude_weighted=magnitude_weighted_phase,
            sample_equal=sample_equal,
        )
        coherence_metric = 1.0 - coherence
        if stereo_mask is not None:
            if sample_equal:


                supervised_fraction = stereo_mask.any(dim=1).float().mean()
                coherence_metric = supervised_fraction - coherence
            elif not bool(stereo_mask.any()):
                coherence_metric = coherence * 0.0
        ipd = absolute_circular_ipd_error(
            estimate,
            target,
            mask,
            stereo_mask,
            magnitude_weighted=magnitude_weighted_phase,
            sample_equal=sample_equal,
        )
        pan = spectral_pan_error(
            estimate,
            target,
            mask,
            stereo_mask,
            sample_equal=sample_equal,
        )
        if posterior_mean is None or posterior_logvar is None:
            kl = linear * 0.0
            kl_mean = kl
            kl_variance = kl
        else:
            kl_mean, kl_variance = diagonal_gaussian_kl_components(
                posterior_mean,
                posterior_logvar,
                latent_mask,
                start_dim=self.config.kl_start_dim,
                reduction=self.config.kl_reduction,
                sample_equal=sample_equal,
            )
            kl = kl_mean + kl_variance
        kl_regularized = (
            self.config.kl_mean_scale * kl_mean
            + self.config.kl_variance_scale * kl_variance
        )
        total = (
            self.config.spectrum_complex * complex_l1
            + self.config.spectrum_linear * linear
            + self.config.spectrum_log1p * log1p
            + self.config.lr_ms * lr_ms
            + self.config.if_gd * phase["if_gd"]
            + self.config.ccpc * coherence
            + self.config.absolute_ipd * ipd
            + self.config.spectral_pan * pan
            + self.config.kl * kl_regularized
        )
        return {
            "total": total,
            "spectrum_linear": linear,
            "spectrum_complex": complex_l1,
            "spectrum_log1p": log1p,
            "lr_ms": lr_ms,
            "if": phase["if"],
            "gd": phase["gd"],
            "if_gd": phase["if_gd"],
            "ccpc_loss": coherence,
            "ccpc": coherence_metric,
            "ipd_cosine_loss": coherence,
            "ipd_cosine_similarity": coherence_metric,
            "absolute_ipd": ipd,
            "spectral_pan": pan,
            "kl": kl,
            "kl_mean": kl_mean,
            "kl_variance": kl_variance,
            "kl_regularized": kl_regularized,
        }


MultiResolutionSpectralLoss = MultiResolutionSTFTLoss
MelLoss = MelSpectralLoss
absolute_ipd_error = absolute_circular_ipd_error
