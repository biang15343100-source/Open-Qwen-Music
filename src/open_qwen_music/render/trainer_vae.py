
from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any, Callable, Mapping, TypeVar

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from open_qwen_music.common.distributed import (
    assert_distributed_consensus,
    raise_if_any_rank_error,
    reduce_scalar_sum,
)

from .contracts import SAMPLE_RATE, STFT_N_FFT
from .discriminators import (
    DiscriminatorOutput,
    SpectroStreamFeatureBandView,
)
from .losses import (
    HighbandWaveformL1Components,
    MelSpectralLoss,
    MultiResolutionSTFTLoss,
    SourceAlignedMultiResolutionSTFTLoss,
    SpectralReconstructionLoss,
    SpectroStreamMixedScaleMelLoss,
    feature_matching_loss,
    lsgan_discriminator_loss,
    lsgan_generator_loss,
    multiband_filtered_waveform_l1_components,
    multiresolution_ccpc,
    multiresolution_ccpc_no_regression_loss,
    refiner_oracle_log_magnitude_loss,
    spectral_bandwidth_mask,
    spectral_complex_l1_loss,
    unit_phasor_if_gd_loss,
    waveform_l1_loss,
    waveform_peak_envelope_overshoot_loss,
    waveform_residual_energy_overshoot_loss,
    waveform_si_sdr_no_regression_loss,
)
from .refiner import (
    EarVAE2PublicRefiner,
    apply_refiner,
    inverse_refiner_spectrum,
    refiner_spectrum_for_loss,
)
from .spec_vae import SpecVAE
from .stft import StereoSTFT
from .trainer_common import finite_gradient_norm, temporary_requires_grad

T = TypeVar("T")
_MISSING = object()


def _discriminator_input_domain(module: nn.Module | None) -> str | None:
    if module is None:
        return None
    unwrapped = module.module if isinstance(module, DistributedDataParallel) else module
    domain = str(getattr(unwrapped, "input_domain", "spectrum"))
    if domain not in {"spectrum", "waveform"}:
        raise ValueError(f"unsupported spectral discriminator input_domain: {domain}")
    return domain


@dataclass(frozen=True)
class AdversarialWeights:
    adversarial: float = 1.0
    feature_matching: float = 100.0


@dataclass
class TrainStepResult:
    loss: Tensor
    metrics: dict[str, float]
    numerators: dict[str, float]
    denominators: dict[str, float]
    discriminator_updated: bool
    local_audio_seconds: float
    global_audio_seconds: float
    backward_scale: float


@dataclass(frozen=True)
class HighbandFeatureMatchingComponents:

    local_band_loss_sums: Tensor
    local_band_eligible_records: Tensor
    per_record_band_loss: Tensor
    eligible_band_mask: Tensor


@dataclass(frozen=True, eq=False)
class GeneratorAdversarialResult(Mapping[str, Tensor]):
    terms: dict[str, Tensor]
    highband_components: HighbandFeatureMatchingComponents | None = None

    def __getitem__(self, key: str) -> Tensor:
        return self.terms[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.terms)

    def __len__(self) -> int:
        return len(self.terms)


def distributed_eligible_record_mean_loss(
    components: HighbandWaveformL1Components,
    *,
    backward_scale: float,
) -> tuple[Tensor, Tensor]:

    if not math.isfinite(backward_scale) or backward_scale <= 0.0:
        raise ValueError("eligible-record DDP reduction requires a finite positive backward_scale")
    global_count = components.local_eligible_records.detach().to(
        device=components.local_loss_sum.device,
        dtype=torch.float64,
    )
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if dist.is_initialized():
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    if float(global_count.item()) <= 0.0:
        return components.local_loss_sum * 0.0, global_count
    loss = (
        components.local_loss_sum
        * float(world_size)
        / global_count.to(dtype=components.local_loss_sum.dtype)
        / float(backward_scale)
    )
    return loss, global_count


def distributed_highband_feature_matching_loss(
    components: HighbandFeatureMatchingComponents,
    *,
    backward_scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:

    if not math.isfinite(backward_scale) or backward_scale <= 0.0:
        raise ValueError("high-band feature-matching DDP reduction requires a finite positive backward_scale")
    local_sums = components.local_band_loss_sums
    local_counts = components.local_band_eligible_records
    if (
        local_sums.ndim != 1
        or local_counts.shape != local_sums.shape
        or local_sums.numel() == 0
    ):
        raise ValueError("high-band feature-matching sums and counts must have matching shapes")
    if not bool(torch.isfinite(local_sums).all()):
        raise FloatingPointError("high-band feature-matching local loss sum contains NaN or infinity")
    if bool((local_counts < 0).any()):
        raise ValueError("high-band feature-matching local eligible count must not be negative")
    locally_eligible = local_counts > 0
    masked_local_sums = torch.where(
        locally_eligible,
        local_sums,
        local_sums * 0.0,
    )

    telemetry = torch.cat(
        (
            masked_local_sums.detach().double(),
            local_counts.detach().double(),
        )
    )
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if dist.is_initialized():
        dist.all_reduce(telemetry, op=dist.ReduceOp.SUM)
    band_count = local_sums.numel()
    global_sums = telemetry[:band_count]
    global_counts = telemetry[band_count:]
    globally_eligible = global_counts > 0.0
    globally_eligible_band_count = globally_eligible.sum()
    safe_counts = global_counts.clamp_min(1.0).to(dtype=local_sums.dtype)
    local_backward_bands = (
        masked_local_sums * float(world_size) / safe_counts / float(backward_scale)
    )
    local_backward_bands = torch.where(
        globally_eligible.to(device=local_sums.device),
        local_backward_bands,
        masked_local_sums * 0.0,
    )
    backward = torch.where(
        globally_eligible_band_count > 0,
        local_backward_bands.sum()
        / globally_eligible_band_count.clamp_min(1).to(local_sums.dtype),
        masked_local_sums.sum() * 0.0,
    )
    global_band_means = torch.where(
        globally_eligible,
        global_sums / global_counts.clamp_min(1.0),
        torch.zeros_like(global_sums),
    ).to(device=local_sums.device, dtype=local_sums.dtype)
    global_counts = global_counts.to(device=local_sums.device, dtype=local_sums.dtype)
    return (
        backward,
        global_band_means,
        global_counts,
        globally_eligible_band_count.to(
            device=local_sums.device,
            dtype=local_sums.dtype,
        ),
    )


def mixed_scale_weight_at_step(
    *,
    initial_weight: float,
    schedule: Mapping[str, Any] | None,
    global_step: int,
) -> float:

    fixed = float(initial_weight)
    if not math.isfinite(fixed) or fixed < 0.0:
        raise ValueError("mixed-scale initial weight must be finite and non-negative")
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise TypeError("mixed-scale schedule global_step must be a non-negative integer")
    step = global_step
    if step < 0:
        raise ValueError("mixed-scale schedule global_step must be a non-negative integer")
    if schedule is None:
        return fixed
    expected_keys = {
        "mode",
        "w_start",
        "hold_through_step",
        "ramp_end_step",
        "target_weight",
    }
    if set(schedule) != expected_keys:
        raise ValueError(
            "mixed-scale schedule field mismatch: "
            f"missing={sorted(expected_keys - set(schedule))}, "
            f"extra={sorted(set(schedule) - expected_keys)}"
        )
    if schedule["mode"] != "safe_start_geometric_v1":
        raise ValueError("mixed-scale schedule supports only safe_start_geometric_v1")
    w_start = float(schedule["w_start"])
    target = float(schedule["target_weight"])
    hold_value = schedule["hold_through_step"]
    ramp_end_value = schedule["ramp_end_step"]
    if (
        isinstance(hold_value, bool)
        or not isinstance(hold_value, int)
        or isinstance(ramp_end_value, bool)
        or not isinstance(ramp_end_value, int)
    ):
        raise TypeError("mixed-scale schedule step boundary must be an integer")
    hold = hold_value
    ramp_end = ramp_end_value
    if any(not math.isfinite(value) or value <= 0.0 for value in (w_start, target)):
        raise ValueError("mixed-scale schedule weight must be finite and positive")
    if fixed != w_start:
        raise ValueError("mixed_scale_spectral_weight must match schedule.w_start")
    if hold < 0 or ramp_end <= hold:
        raise ValueError("mixed-scale schedule must satisfy 0 <= hold < ramp_end")
    if step <= hold:
        return w_start
    if step >= ramp_end:
        return target
    progress = (step - hold) / (ramp_end - hold)
    return math.exp(
        math.log(w_start) + progress * (math.log(target) - math.log(w_start))
    )


class LossTermGradientDominanceError(RuntimeError):

    def __init__(
        self,
        *,
        term_gradients: Mapping[str, Mapping[str, float]],
        violations: Mapping[str, Mapping[str, float | None]],
        dominance_ratio: float,
    ) -> None:
        first = next(iter(violations))
        super().__init__(
            f"loss gradient audit {first} decoder         "
            f"{dominance_ratio:g} "
        )
        self.term_gradients = {
            name: dict(values) for name, values in term_gradients.items()
        }
        self.violations = {name: dict(values) for name, values in violations.items()}
        self.dominance_ratio = dominance_ratio


def audit_loss_term_gradients(
    *,
    raw_terms: Mapping[str, Tensor],
    weighted_terms: Mapping[str, Tensor],
    decoder_output: Tensor,
    named_parameters: list[tuple[str, nn.Parameter]],
    dominance_ratio: float = 100.0,
) -> dict[str, dict[str, float]]:

    if not raw_terms or set(raw_terms) != set(weighted_terms):
        raise ValueError("loss gradient audit raw and weighted term collections must be identical and non-empty")
    if not math.isfinite(dominance_ratio) or dominance_ratio <= 0:
        raise ValueError("loss gradient audit dominance_ratio must be finite and positive")
    parameters = [value for _, value in named_parameters if value.requires_grad]
    if not parameters:
        raise ValueError("loss gradient audit found no trainable parameters")
    result: dict[str, dict[str, float]] = {}
    for name in sorted(raw_terms):
        raw = raw_terms[name]
        weighted = weighted_terms[name]
        if raw.numel() != 1 or weighted.numel() != 1:
            raise ValueError(f"loss gradient audit {name} must be scalar")
        if not bool(torch.isfinite(raw)) or not bool(torch.isfinite(weighted)):
            raise FloatingPointError(f"loss gradient audit {name} is not finite")
        gradients = torch.autograd.grad(
            weighted,
            [decoder_output, *parameters],
            retain_graph=True,
            allow_unused=True,
        )

        def l2(values: list[Tensor | None]) -> Tensor:
            total = torch.zeros((), dtype=torch.float64, device=decoder_output.device)
            for value in values:
                if value is not None:


                    total = total + value.detach().abs().double().square().sum()
            return total.sqrt()

        decoder_l2 = l2([gradients[0]])
        parameter_l2 = l2(list(gradients[1:]))
        if not bool(torch.isfinite(decoder_l2)) or not bool(
            torch.isfinite(parameter_l2)
        ):
            raise FloatingPointError(f"loss gradient audit {name} gradient is not finite")
        if float(decoder_l2) == 0.0 or float(parameter_l2) == 0.0:
            raise RuntimeError(f"loss gradient audit {name} gradient is zero")
        result[name] = {
            "raw": float(raw.detach().float()),
            "weighted": float(weighted.detach().float()),
            "decoder_output_gradient_l2": float(decoder_l2),
            "parameter_gradient_l2": float(parameter_l2),
        }
    decoder_norms = {
        name: values["decoder_output_gradient_l2"] for name, values in result.items()
    }
    violations: dict[str, dict[str, float | None]] = {}
    for name, value in decoder_norms.items():
        other_sum = sum(
            other for other_name, other in decoder_norms.items() if other_name != name
        )
        if value > dominance_ratio * other_sum:
            violations[name] = {
                "decoder_output_gradient_l2": value,
                "other_decoder_output_gradient_l2_sum": other_sum,
                "ratio_to_other_sum": value / other_sum if other_sum > 0.0 else None,
                "dominance_ratio_limit": dominance_ratio,
            }
    if violations:
        raise LossTermGradientDominanceError(
            term_gradients=result,
            violations=violations,
            dominance_ratio=dominance_ratio,
        )
    return result


@dataclass(frozen=True)
class SourceAnchoredStage1LossAssembly:
    total: Tensor
    terms: dict[str, Tensor]
    reconstruction_raw_terms: dict[str, Tensor]
    reconstruction_weighted_terms: dict[str, Tensor]


@dataclass(frozen=True)
class Stage3IncrementalLossAssembly:

    total: Tensor
    terms: dict[str, Tensor]


def posterior_forward_arguments(
    *,
    posterior_mode: str,
    posterior_generator: torch.Generator | None,
    posterior_sample_layout: str,
) -> dict[str, Any]:

    if posterior_mode not in {"mean", "sample"}:
        raise ValueError("posterior_mode must be mean or sample")
    if posterior_sample_layout != "contiguous_bdt":
        raise ValueError("posterior_sample_layout must be contiguous_bdt")
    if posterior_mode == "mean" and posterior_generator is not None:
        raise ValueError("posterior mean mode does not accept a sampling generator")
    if (
        posterior_mode == "sample"
        and posterior_sample_layout == "contiguous_bdt"
        and posterior_generator is None
    ):
        raise ValueError("contiguous_bdt sampling requires an independent generator")
    return {
        "sample_posterior": posterior_mode == "sample",
        "generator": posterior_generator if posterior_mode == "sample" else None,
        "posterior_sample_layout": posterior_sample_layout,
    }


def assemble_source_anchored_stage1_loss(
    *,
    source_reconstruction_loss: SourceAlignedMultiResolutionSTFTLoss,
    source_mixed_scale_loss: (
        SpectroStreamMixedScaleMelLoss | None
    ),
    source_mixed_scale_weight: float,
    source_direct_complex_weight: float,
    waveform_l1_weight: float,
    stft_weight: float,
    kl_weight: float,
    output: Any,
    target_stft: Any,
    fake_audio: Tensor,
    target_audio: Tensor,
    audio_lengths: Tensor,
    media_bandwidth_hz: Tensor,
    magnitude_max_hz: Tensor,
    phase_max_hz: Tensor,
    waveform_adversarial_enabled: Tensor,
    source_if_gd_weight: float = 0.0,
    source_if_gd_min_hz: float = 4_000.0,
    source_if_gd_magnitude_weighted: bool = True,
) -> SourceAnchoredStage1LossAssembly:

    source_terms = source_reconstruction_loss.components(
        fake_audio,
        target_audio.float(),
        audio_lengths,
        media_bandwidth_hz,
    )
    posterior = output.posterior
    if not hasattr(posterior, "source_surrogate_components"):
        raise TypeError("source-anchored loss requires a softplus-scale posterior")
    mean_component, variance_component = posterior.source_surrogate_components()


    source_no_half = posterior.source_surrogate_kl()
    source_half = 0.5 * source_no_half
    actual_half = posterior.actual_sample_q_half_kl()
    source_mr_raw = source_terms["source_reconstruction"]
    source_mr_weighted = stft_weight * source_mr_raw
    kl_weighted = kl_weight * source_no_half
    total = source_mr_weighted + kl_weighted
    valid_std = posterior.std.float()[posterior.mask]
    valid_mean = posterior.mean.float()[posterior.mask]
    valid_stabilized_std = posterior.variance.float().sqrt()[posterior.mask]
    if valid_std.numel() == 0:
        raise ValueError("source posterior has no valid frames")
    terms = dict(source_terms)
    terms.update(
        {
            "kl": source_no_half,
            "source_surrogate_no_half": source_no_half,
            "source_surrogate_half": source_half,
            "actual_sample_q_half_kl": actual_half,
            "source_mr_raw": source_mr_raw,
            "source_mr_weighted": source_mr_weighted,
            "kl_surrogate_no_half_raw": source_no_half,
            "kl_surrogate_half_raw": source_half,
            "kl_surrogate_weighted": kl_weighted,
            "kl_actual_q_half_raw": actual_half,
            "kl_surrogate_mean_component_raw": mean_component,
            "kl_surrogate_variance_component_raw": variance_component,
            "kl_mean": mean_component,
            "kl_variance": variance_component,
            "kl_regularized": source_no_half,
            "posterior_sample_std_mean": valid_std.mean(),
            "posterior_stabilized_kl_std_mean": valid_stabilized_std.mean(),
            "posterior_noise_to_mean_rms_ratio": (
                valid_std.square().mean().sqrt()
                / valid_mean.square().mean().sqrt().clamp_min(1.0e-20)
            ),
        }
    )
    quantile_levels = torch.tensor(
        [0.01, 0.10, 0.50, 0.90, 0.99],
        device=valid_std.device,
        dtype=torch.float32,
    )
    quantiles = torch.quantile(valid_std.reshape(-1), quantile_levels)
    for name, value in zip(
        ("p01", "p10", "p50", "p90", "p99"),
        quantiles,
        strict=True,
    ):
        terms[f"posterior_sample_std_{name}"] = value
    terms["posterior_actual_variance_tiny_boundary_rate"] = (
        (
            posterior.sample_variance.float()[posterior.mask]
            <= torch.finfo(torch.float32).tiny
        )
        .float()
        .mean()
    )

    raw_audit = {"source_mr": source_mr_raw}
    weighted_audit = {"source_mr": source_mr_weighted}
    if source_mixed_scale_weight:
        if source_mixed_scale_loss is None:
            raise ValueError("enabled mixed-scale objective is missing its Equation 6 implementation")
        mixed_terms = source_mixed_scale_loss.components(
            fake_audio,
            target_audio.float(),
            audio_lengths,
            media_bandwidth_hz,
        )
        terms.update(mixed_terms)
        mixed_raw = mixed_terms["mixed_scale_spectral"]
        mixed_weighted = source_mixed_scale_weight * mixed_raw
        terms["mixed_scale_spectral_weighted"] = mixed_weighted
        terms["mixed_scale_spectral_applied_weight"] = torch.as_tensor(
            source_mixed_scale_weight,
            device=mixed_raw.device,
            dtype=mixed_raw.dtype,
        )
        total = total + mixed_weighted
        raw_audit["mixed_scale_spectral"] = mixed_raw
        weighted_audit["mixed_scale_spectral"] = mixed_weighted
    if source_direct_complex_weight:
        complex_max_hz = torch.minimum(
            torch.minimum(media_bandwidth_hz.float(), magnitude_max_hz.float()),
            phase_max_hz.float(),
        )
        phase_frequency_mask = spectral_bandwidth_mask(
            complex_max_hz,
            bins=output.reconstruction.shape[-2],
            bin_hz=SAMPLE_RATE / STFT_N_FFT,
            device=output.reconstruction.device,
        )
        direct_raw = spectral_complex_l1_loss(
            output.reconstruction,
            target_stft.spectrum,
            target_stft.spectrum_mask,
            phase_frequency_mask,
            sample_equal=True,
        )
        direct_weighted = source_direct_complex_weight * direct_raw
        terms["source_direct_complex"] = direct_raw
        terms["source_direct_complex_weighted"] = direct_weighted
        total = total + direct_weighted
        raw_audit["source_direct_complex"] = direct_raw
        weighted_audit["source_direct_complex"] = direct_weighted
    if source_if_gd_weight:
        complex_max_hz = torch.minimum(
            torch.minimum(media_bandwidth_hz.float(), magnitude_max_hz.float()),
            phase_max_hz.float(),
        )
        frequencies = torch.arange(
            output.reconstruction.shape[-2],
            device=output.reconstruction.device,
            dtype=torch.float32,
        ) * (SAMPLE_RATE / STFT_N_FFT)
        phase_frequency_mask = (
            (frequencies[None, :] >= float(source_if_gd_min_hz))
            & (frequencies[None, :] <= complex_max_hz[:, None])
            & (complex_max_hz[:, None] > 0.0)
        )
        phase_terms = unit_phasor_if_gd_loss(
            output.reconstruction,
            target_stft.spectrum,
            target_stft.spectrum_mask,
            phase_frequency_mask,
            magnitude_weighted=source_if_gd_magnitude_weighted,
            sample_equal=True,
        )
        phase_raw = phase_terms["if_gd"]
        phase_weighted = source_if_gd_weight * phase_raw
        terms["source_if"] = phase_terms["if"]
        terms["source_gd"] = phase_terms["gd"]
        terms["source_if_gd"] = phase_raw
        terms["source_if_gd_weighted"] = phase_weighted
        total = total + phase_weighted
        raw_audit["source_if_gd"] = phase_raw
        weighted_audit["source_if_gd"] = phase_weighted
    if waveform_l1_weight:
        if not bool(waveform_adversarial_enabled.all()):
            raise ValueError("source waveform L1 requires explicitly enabled waveform supervision")
        waveform_raw = waveform_l1_loss(
            fake_audio,
            target_audio.float(),
            audio_lengths,
            sample_equal=True,
        )
        waveform_weighted = waveform_l1_weight * waveform_raw
        terms["waveform_l1"] = waveform_raw
        terms["waveform_l1_weighted"] = waveform_weighted
        total = total + waveform_weighted
        raw_audit["waveform_l1"] = waveform_raw
        weighted_audit["waveform_l1"] = waveform_weighted
    terms["source_reconstruction_bucket_weighted"] = sum(weighted_audit.values())
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("source-anchored Spec-VAE generator loss contains NaN or infinity")
    return SourceAnchoredStage1LossAssembly(
        total=total,
        terms=terms,
        reconstruction_raw_terms=raw_audit,
        reconstruction_weighted_terms=weighted_audit,
    )


def assemble_stage3_incremental_loss(
    *,
    source_reconstruction_loss: SourceAlignedMultiResolutionSTFTLoss,
    source_mixed_scale_loss: (
        SpectroStreamMixedScaleMelLoss | None
    ),
    source_mixed_scale_weight: float,
    stft_weight: float,
    if_gd_weight: float,
    if_gd_objective: str,
    if_gd_sample_equal: bool,
    refined_spectrum: Tensor,
    target_stft: Any,
    fake_audio: Tensor,
    target_audio: Tensor,
    audio_lengths: Tensor,
    media_bandwidth_hz: Tensor,
    magnitude_max_hz: Tensor,
    phase_max_hz: Tensor,
    phase_mode_mask: Tensor,
) -> Stage3IncrementalLossAssembly:

    for name, value in (
        ("source_mixed_scale_weight", source_mixed_scale_weight),
        ("stft_weight", stft_weight),
        ("if_gd_weight", if_gd_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Refiner incremental {name} must be finite and non-negative")
    if source_mixed_scale_weight and source_mixed_scale_loss is None:
        raise ValueError("Refiner training requires the inherited mixed-scale parent loss instance")
    if if_gd_objective != "ear_wrapped_l1_v1":
        raise ValueError("Refiner IF/GD supports only EAR wrapped-phase L1")
    if phase_mode_mask.shape != (refined_spectrum.shape[-2],):
        raise ValueError("Refiner IF/GD phase-mode mask must have shape [F]")

    magnitude_limit_hz = torch.minimum(
        media_bandwidth_hz.float(),
        magnitude_max_hz.float(),
    )
    source_terms = source_reconstruction_loss.components(
        fake_audio,
        target_audio.float(),
        audio_lengths,
        magnitude_limit_hz,
    )
    source_mr_raw = source_terms["source_reconstruction"]
    source_mr_weighted = stft_weight * source_mr_raw
    total = source_mr_weighted
    terms = dict(source_terms)
    terms["source_mr_raw"] = source_mr_raw
    terms["source_mr_weighted"] = source_mr_weighted

    if source_mixed_scale_weight:
        assert source_mixed_scale_loss is not None
        mixed_terms = source_mixed_scale_loss.components(
            fake_audio,
            target_audio.float(),
            audio_lengths,
            magnitude_limit_hz,
        )
        terms.update(mixed_terms)
        mixed_raw = mixed_terms["mixed_scale_spectral"]
        mixed_weighted = source_mixed_scale_weight * mixed_raw
        terms["mixed_scale_spectral_weighted"] = mixed_weighted
        terms["mixed_scale_spectral_applied_weight"] = torch.as_tensor(
            source_mixed_scale_weight,
            device=mixed_raw.device,
            dtype=mixed_raw.dtype,
        )
        total = total + mixed_weighted

    phase_limit_hz = torch.minimum(
        media_bandwidth_hz.float(),
        phase_max_hz.float(),
    )
    frequencies = torch.arange(
        refined_spectrum.shape[-2],
        device=refined_spectrum.device,
        dtype=torch.float32,
    ) * (SAMPLE_RATE / STFT_N_FFT)
    phase_frequency_mask = (
        (frequencies[None, :] <= phase_limit_hz[:, None])
        & (phase_limit_hz[:, None] > 0.0)
        & phase_mode_mask.to(device=refined_spectrum.device, dtype=torch.bool)[None, :]
    )
    phase_terms = unit_phasor_if_gd_loss(
        refined_spectrum,
        target_stft.spectrum,
        target_stft.spectrum_mask,
        phase_frequency_mask,
        magnitude_weighted=False,
        sample_equal=if_gd_sample_equal,
        objective=if_gd_objective,
    )
    phase_raw = phase_terms["if_gd"]
    phase_weighted = if_gd_weight * phase_raw
    terms["if"] = phase_terms["if"]
    terms["gd"] = phase_terms["gd"]
    terms["if_gd"] = phase_raw
    terms["if_gd_weighted"] = phase_weighted
    total = total + phase_weighted
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Refiner incremental generator loss contains NaN or infinity")
    return Stage3IncrementalLossAssembly(total=total, terms=terms)


def _root_module(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


def synchronized_call(action: str, function: Callable[[], T]) -> T:

    result: T | object = _MISSING
    local_error: str | None = None
    try:
        result = function()
    except Exception as exc:  # noqa: BLE001 -  rank.
        local_error = f"{type(exc).__name__}: {exc}"
    raise_if_any_rank_error(local_error, action=action)
    assert result is not _MISSING
    return result  # type: ignore[return-value]


def _split_discriminator_output(
    output: DiscriminatorOutput,
    batch_size: int,
) -> tuple[DiscriminatorOutput, DiscriminatorOutput]:
    def split_tensor(value: Tensor) -> tuple[Tensor, Tensor]:
        if value.shape[0] != 2 * batch_size:
            raise ValueError(
                "discriminator merged-batch output has invalid first dimension: "
                f"{value.shape[0]} != {2 * batch_size}"
            )
        return value[:batch_size], value[batch_size:]

    real_logits: list[Tensor] = []
    fake_logits: list[Tensor] = []
    for value in output.logits:
        real, fake = split_tensor(value)
        real_logits.append(real)
        fake_logits.append(fake)
    real_features: list[list[Tensor]] = []
    fake_features: list[list[Tensor]] = []
    for scale in output.features:
        current_real: list[Tensor] = []
        current_fake: list[Tensor] = []
        for value in scale:
            real, fake = split_tensor(value)
            current_real.append(real)
            current_fake.append(fake)
        real_features.append(current_real)
        fake_features.append(current_fake)
    real_highband_features: list[list[Tensor]] | None = None
    fake_highband_features: list[list[Tensor]] | None = None
    if output.highband_features is not None:
        real_highband_features = []
        fake_highband_features = []
        for scale in output.highband_features:
            current_real = []
            current_fake = []
            for value in scale:
                real, fake = split_tensor(value)
                current_real.append(real)
                current_fake.append(fake)
            real_highband_features.append(current_real)
            fake_highband_features.append(current_fake)
    real_highband_feature_views: list[SpectroStreamFeatureBandView] | None = None
    fake_highband_feature_views: list[SpectroStreamFeatureBandView] | None = None
    if output.highband_feature_views is not None:
        real_highband_feature_views = []
        fake_highband_feature_views = []
        for view in output.highband_feature_views:
            real, fake = split_tensor(view.tensor)
            real_highband_feature_views.append(
                SpectroStreamFeatureBandView(spec=view.spec, tensor=real)
            )
            fake_highband_feature_views.append(
                SpectroStreamFeatureBandView(spec=view.spec, tensor=fake)
            )
    return (
        DiscriminatorOutput(
            real_logits,
            real_features,
            real_highband_features,
            real_highband_feature_views,
            family_names=output.family_names,
            family_sizes=output.family_sizes,
            family_weights=output.family_weights,
            loss_reduction=output.loss_reduction,
            loss_compute_dtype=output.loss_compute_dtype,
            family_diagnostics_enabled=output.family_diagnostics_enabled,
        ),
        DiscriminatorOutput(
            fake_logits,
            fake_features,
            fake_highband_features,
            fake_highband_feature_views,
            family_names=output.family_names,
            family_sizes=output.family_sizes,
            family_weights=output.family_weights,
            loss_reduction=output.loss_reduction,
            loss_compute_dtype=output.loss_compute_dtype,
            family_diagnostics_enabled=output.family_diagnostics_enabled,
        ),
    )


def _adversarial_reduction_arguments(
    output: DiscriminatorOutput,
) -> tuple[tuple[int, ...] | None, torch.dtype]:
    compute_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }.get(output.loss_compute_dtype)
    if compute_dtype is None:
        raise ValueError(f"unsupported discriminator loss_compute_dtype: {output.loss_compute_dtype!r}")
    if output.loss_reduction == "flat_feature_mean_v1":
        if output.family_weights is not None:
            raise ValueError("flat_feature_mean_v1 does not accept family_weights")
        return None, compute_dtype
    if output.loss_reduction in {
        "family_equal_mean_v1",
        "family_weighted_sum_v1",
    }:
        names = output.family_names
        sizes = output.family_sizes
        if (
            names is None
            or sizes is None
            or not names
            or len(names) != len(sizes)
            or len(set(names)) != len(names)
            or any(not name for name in names)
            or any(int(size) <= 0 for size in sizes)
            or sum(int(size) for size in sizes) != len(output.logits)
        ):
            raise ValueError("family-based reduction requires complete, unique, positive family identities")
        weights = output.family_weights
        if output.loss_reduction == "family_equal_mean_v1":
            if weights is not None:
                raise ValueError("family_equal_mean_v1 does not accept family_weights")
        elif (
            weights is None
            or len(weights) != len(sizes)
            or any(not math.isfinite(value) or value < 0.0 for value in weights)
            or not any(value > 0.0 for value in weights)
        ):
            raise ValueError(
                "family_weighted_sum_v1requires complete,non-negative and non-all-zerofamily_weights"
            )
        return sizes, compute_dtype
    raise ValueError(f"unsupported discriminator loss_reduction: {output.loss_reduction!r}")


def _adversarial_family_terms(
    output: DiscriminatorOutput,
    *,
    real_output: DiscriminatorOutput,
    fake_output: DiscriminatorOutput,
    compute_dtype: torch.dtype,
) -> dict[str, Tensor]:

    enabled = output.family_diagnostics_enabled
    if enabled is None:
        enabled = output.loss_reduction in {
            "family_equal_mean_v1",
            "family_weighted_sum_v1",
        }
    if not enabled:
        return {}
    if output.family_names is None or output.family_sizes is None:
        return {}
    if len(output.family_names) != len(output.family_sizes):
        raise ValueError("discriminator family names and sizes must have matching lengths")
    terms: dict[str, Tensor] = {}
    offset = 0
    for name, size in zip(output.family_names, output.family_sizes, strict=True):
        stop = offset + int(size)
        if size <= 0 or stop > len(real_output.logits):
            raise ValueError("discriminator family slice is out of bounds")
        terms[f"adversarial_{name}"] = lsgan_generator_loss(
            [value.detach() for value in fake_output.logits[offset:stop]],
            compute_dtype=compute_dtype,
        )
        terms[f"feature_matching_{name}"] = feature_matching_loss(
            [
                [value.detach() for value in scale]
                for scale in real_output.features[offset:stop]
            ],
            [
                [value.detach() for value in scale]
                for scale in fake_output.features[offset:stop]
            ],
            compute_dtype=compute_dtype,
        )
        offset = stop
    if offset != len(real_output.logits):
        raise ValueError("discriminator families do not cover every scale")
    return terms


def distributed_audio_weight(
    batch: Mapping[str, Any],
) -> tuple[float, float, float]:

    def local_audio_seconds() -> float:
        value = _audio_seconds(batch)
        if value < 0:
            raise ValueError("local audio-seconds must not be negative")
        return value


    local_seconds = synchronized_call("audio-seconds contract", local_audio_seconds)
    global_seconds = reduce_scalar_sum(local_seconds)
    if not global_seconds > 0:
        raise ValueError(
            "local audio seconds may be zero but cannot be negative; global audio seconds must be positive; "
            f"local={local_seconds}, global={global_seconds}"
        )
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    backward_scale = world_size * local_seconds / global_seconds
    return local_seconds, global_seconds, backward_scale


def discriminator_step(
    discriminator: nn.Module,
    optimizer: torch.optim.Optimizer,
    real: Tensor,
    fake: Tensor,
    *,
    lengths: Tensor | None = None,
    max_grad_norm: float | None = None,
    backward_scale: float = 1.0,
) -> dict[str, Tensor]:

    optimizer.zero_grad(set_to_none=True)
    batch_size = real.shape[0]

    def validate_inputs() -> bool:
        if fake.shape[0] != batch_size or real.shape[1:] != fake.shape[1:]:
            raise ValueError("discriminator real and fake outputs must have matching shapes")
        return True

    synchronized_call("discriminator input contract", validate_inputs)

    def forward_and_loss() -> tuple[Tensor, dict[str, Tensor]]:
        combined_lengths = (
            torch.cat((lengths, lengths), dim=0) if lengths is not None else None
        )
        combined = torch.cat((real, fake.detach()), dim=0)
        output = (
            discriminator(combined)
            if combined_lengths is None
            else discriminator(combined, lengths=combined_lengths)
        )
        if not isinstance(output, DiscriminatorOutput):
            raise TypeError("discriminator must return DiscriminatorOutput")
        real_output, fake_output = _split_discriminator_output(output, batch_size)
        family_sizes, compute_dtype = _adversarial_reduction_arguments(output)
        loss = lsgan_discriminator_loss(
            real_output.logits,
            fake_output.logits,
            family_sizes=family_sizes,
            family_weights=output.family_weights,
            compute_dtype=compute_dtype,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("discriminator loss is NaN/Inf")
        family_metrics: dict[str, Tensor] = {}
        family_diagnostics_enabled = output.family_diagnostics_enabled
        if family_diagnostics_enabled is None:
            family_diagnostics_enabled = output.loss_reduction in {
                "family_equal_mean_v1",
                "family_weighted_sum_v1",
            }
        if (
            family_diagnostics_enabled
            and output.family_names is not None
            and output.family_sizes is not None
        ):
            offset = 0
            for family_name, family_size in zip(
                output.family_names,
                output.family_sizes,
                strict=True,
            ):
                stop = offset + int(family_size)
                real_family_logits = real_output.logits[offset:stop]
                fake_family_logits = fake_output.logits[offset:stop]
                family_metrics[f"discriminator_{family_name}"] = (
                    lsgan_discriminator_loss(
                        [value.detach() for value in real_family_logits],
                        [value.detach() for value in fake_family_logits],
                        compute_dtype=compute_dtype,
                    )
                )
                if output.family_diagnostics_enabled is True:
                    real_logit_mean = torch.stack(
                        [
                            value.detach().to(compute_dtype).mean()
                            for value in real_family_logits
                        ]
                    ).mean()
                    fake_logit_mean = torch.stack(
                        [
                            value.detach().to(compute_dtype).mean()
                            for value in fake_family_logits
                        ]
                    ).mean()
                    family_metrics[f"discriminator_{family_name}_real_logit_mean"] = (
                        real_logit_mean
                    )
                    family_metrics[f"discriminator_{family_name}_fake_logit_mean"] = (
                        fake_logit_mean
                    )
                    family_metrics[
                        f"discriminator_{family_name}_real_fake_logit_gap"
                    ] = real_logit_mean - fake_logit_mean
                offset = stop
            if offset != len(real_output.logits):
                raise ValueError("discriminator families do not cover every scale")
        return loss, family_metrics

    loss, family_metrics = synchronized_call(
        "discriminator forward/loss",
        forward_and_loss,
    )
    (loss * float(backward_scale)).backward()

    def optimize() -> Tensor:
        if max_grad_norm is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(), max_grad_norm, error_if_nonfinite=True
            )
        else:
            gradient_norm = finite_gradient_norm(discriminator.parameters())
        optimizer.step()
        return torch.as_tensor(gradient_norm).detach()

    gradient_norm = synchronized_call("discriminator optimizer step", optimize)
    return {
        "discriminator": loss.detach(),
        "discriminator_grad_norm": torch.as_tensor(gradient_norm).detach(),
        **{name: value.detach() for name, value in family_metrics.items()},
    }


def spectrostream_highband_feature_matching_loss(
    real_views: list[SpectroStreamFeatureBandView],
    fake_views: list[SpectroStreamFeatureBandView],
    *,
    adversarial_max_hz: Tensor,
    expected_bands_hz: tuple[tuple[float, float], ...],
) -> HighbandFeatureMatchingComponents:

    expected_band_count = len(expected_bands_hz)
    if expected_band_count <= 0:
        raise ValueError("high-band feature-matching expected_bands_hz must not be empty")
    if len(real_views) != len(fake_views) or not real_views:
        raise ValueError("high-band feature-matching real and fake views must be non-empty and have equal lengths")
    batch_size = int(real_views[0].tensor.shape[0])
    if adversarial_max_hz.shape != (batch_size,):
        raise ValueError("high-band feature-matching adversarial_max_hz must have shape [B]")
    if not bool(torch.isfinite(adversarial_max_hz).all()) or bool(
        ((adversarial_max_hz < 0.0) | (adversarial_max_hz > SAMPLE_RATE / 2)).any()
    ):
        raise ValueError("high-band feature-matching adversarial_max_hz must be in [0, Nyquist]")
    if any(
        not 0.0 <= lower < upper <= SAMPLE_RATE / 2
        for lower, upper in expected_bands_hz
    ) or any(
        expected_bands_hz[index][1] > expected_bands_hz[index + 1][0]
        for index in range(expected_band_count - 1)
    ):
        raise ValueError("high-band feature-matching expected_bands_hz must be ordered and non-overlapping")
    band_terms: list[list[Tensor]] = [[] for _ in expected_bands_hz]
    for real, fake in zip(real_views, fake_views, strict=True):
        if real.spec != fake.spec:
            raise ValueError("high-band feature-matching real and fake view identities do not match")
        if real.tensor.shape != fake.tensor.shape or real.tensor.shape[0] != batch_size:
            raise ValueError("high-band feature-matching real and fake view shapes do not match")
        if not 0 <= real.spec.band_index < expected_band_count:
            raise ValueError("high-band feature-matching band index is out of bounds")
        if real.spec.requested_hz != expected_bands_hz[real.spec.band_index]:
            raise ValueError("high-band feature-matching view band identity changed")
        band_terms[real.spec.band_index].append(
            (real.tensor.detach().float() - fake.tensor.float())
            .abs()
            .mean(dim=tuple(range(1, fake.tensor.ndim)))
        )
    if any(not values for values in band_terms):
        raise ValueError("high-band feature-matching has no available feature views")
    per_record_band_loss = torch.stack(
        [torch.stack(values).mean(dim=0) for values in band_terms],
        dim=1,
    )
    if not bool(torch.isfinite(per_record_band_loss).all()):
        raise FloatingPointError("high-band feature-matching per-record loss contains NaN or infinity")
    upper_hz = torch.tensor(
        [upper for _lower, upper in expected_bands_hz],
        device=adversarial_max_hz.device,
        dtype=adversarial_max_hz.dtype,
    )
    eligible = adversarial_max_hz[:, None] >= upper_hz[None, :]
    local_band_loss_sums = (
        per_record_band_loss * eligible.to(per_record_band_loss.dtype)
    ).sum(dim=0)
    local_band_eligible_records = eligible.sum(dim=0)
    return HighbandFeatureMatchingComponents(
        local_band_loss_sums=local_band_loss_sums,
        local_band_eligible_records=local_band_eligible_records,
        per_record_band_loss=per_record_band_loss,
        eligible_band_mask=eligible,
    )


def local_highband_feature_matching_loss(
    components: HighbandFeatureMatchingComponents,
) -> tuple[Tensor, Tensor, Tensor]:

    sums = components.local_band_loss_sums
    counts = components.local_band_eligible_records
    eligible = counts > 0
    band_means = torch.where(
        eligible,
        sums / counts.clamp_min(1).to(sums.dtype),
        torch.zeros_like(sums),
    )
    eligible_band_count = eligible.sum()
    loss = torch.where(
        eligible_band_count > 0,
        band_means.sum() / eligible_band_count.clamp_min(1).to(sums.dtype),
        sums.sum() * 0.0,
    )
    return loss, band_means, counts.to(dtype=sums.dtype)


def generator_adversarial_terms(
    discriminator: nn.Module,
    real: Tensor,
    fake: Tensor,
    *,
    lengths: Tensor | None = None,
    highband_adversarial_max_hz: Tensor | None = None,
    highband_bands_hz: tuple[tuple[float, float], ...] = (),
) -> GeneratorAdversarialResult:

    root = _root_module(discriminator)
    batch_size = real.shape[0]
    if fake.shape[0] != batch_size or real.shape[1:] != fake.shape[1:]:
        raise ValueError("generator adversarial real and fake outputs must have matching shapes")

    with temporary_requires_grad(root, False):
        combined_lengths = (
            torch.cat((lengths, lengths), dim=0) if lengths is not None else None
        )
        combined = torch.cat((real, fake), dim=0)
        output = (
            root(combined)
            if combined_lengths is None
            else root(combined, lengths=combined_lengths)
        )
    if not isinstance(output, DiscriminatorOutput):
        raise TypeError("discriminator must return DiscriminatorOutput")
    real_output, fake_output = _split_discriminator_output(output, batch_size)
    family_sizes, compute_dtype = _adversarial_reduction_arguments(output)
    terms: dict[str, Tensor] = {
        "adversarial": lsgan_generator_loss(
            fake_output.logits,
            family_sizes=family_sizes,
            family_weights=output.family_weights,
            compute_dtype=compute_dtype,
        ),
        "feature_matching": feature_matching_loss(
            real_output.features,
            fake_output.features,
            family_sizes=family_sizes,
            family_weights=output.family_weights,
            compute_dtype=compute_dtype,
        ),
    }
    terms.update(
        _adversarial_family_terms(
            output,
            real_output=real_output,
            fake_output=fake_output,
            compute_dtype=compute_dtype,
        )
    )
    highband_components = None
    if (
        real_output.highband_features is not None
        and fake_output.highband_features is not None
    ):
        if (
            real_output.highband_feature_views is not None
            and fake_output.highband_feature_views is not None
        ):
            band_count = 1 + max(
                view.spec.band_index for view in real_output.highband_feature_views
            )
            if highband_adversarial_max_hz is None:
                raise ValueError(
                    "structured high-band feature views require per-record "
                    "adversarial_max_hz"
                )
            if len(highband_bands_hz) != band_count:
                raise ValueError("structured high-band feature-view count changed")
            highband_components = spectrostream_highband_feature_matching_loss(
                real_output.highband_feature_views,
                fake_output.highband_feature_views,
                adversarial_max_hz=highband_adversarial_max_hz,
                expected_bands_hz=highband_bands_hz,
            )
        else:
            terms["highband_feature_matching"] = feature_matching_loss(
                real_output.highband_features,
                fake_output.highband_features,
            )
    return GeneratorAdversarialResult(
        terms=terms,
        highband_components=highband_components,
    )


def _bandlimit_waveform(
    waveform: Tensor,
    max_hz: Tensor,
    *,
    sample_rate: int = 48_000,
    mode: str = "hard_rfft_v1",
    transition_hz: float = 0.0,
    reflect_padding_samples: int = 0,
) -> Tensor:

    if max_hz.shape != (waveform.shape[0],):
        raise ValueError("adversarial_max_hz must have shape [B]")
    if mode not in {"none_v1", "hard_rfft_v1", "reflect_cosine_v1"}:
        raise ValueError("unsupported adversarial band-limit mode")
    if mode == "none_v1":
        if transition_hz != 0.0 or reflect_padding_samples != 0:
            raise ValueError("none_v1 does not allow a transition band or reflection padding")
        return waveform.float()
    if mode == "hard_rfft_v1":
        if transition_hz != 0.0 or reflect_padding_samples != 0:
            raise ValueError("hard_rfft_v1 does not allow a transition band or reflection padding")
        frequencies = torch.fft.rfftfreq(
            waveform.shape[-1], d=1.0 / sample_rate, device=waveform.device
        )
        limits = max_hz.to(
            device=waveform.device,
            dtype=frequencies.dtype,
        ).view(-1, 1, 1)
        mask = (frequencies.view(1, 1, -1) <= limits) & (limits > 0)
        spectrum = torch.fft.rfft(waveform.float(), dim=-1)
        return torch.fft.irfft(
            spectrum * mask.to(spectrum.dtype),
            n=waveform.shape[-1],
            dim=-1,
        )

    if (
        not math.isfinite(transition_hz)
        or transition_hz <= 0.0
        or reflect_padding_samples <= 0
        or waveform.shape[-1] <= reflect_padding_samples
    ):
        raise ValueError("reflect_cosine_v1 requires a valid transition band and reflection padding")


    values = waveform.float()
    left = values[..., 1 : reflect_padding_samples + 1].flip(-1)
    right = values[..., -reflect_padding_samples - 1 : -1].flip(-1)
    padded = torch.cat(
        (left, values, right),
        dim=-1,
    )
    frequencies = torch.fft.rfftfreq(
        padded.shape[-1],
        d=1.0 / sample_rate,
        device=waveform.device,
    )
    limits = max_hz.to(device=waveform.device, dtype=frequencies.dtype).view(-1, 1, 1)
    frequency_grid = frequencies.view(1, 1, -1)
    ramp_start = limits - float(transition_hz)
    position = ((frequency_grid - ramp_start) / float(transition_hz)).clamp(0.0, 1.0)
    mask = 0.5 + 0.5 * torch.cos(math.pi * position)
    mask = torch.where(frequency_grid <= ramp_start, torch.ones_like(mask), mask)
    mask = torch.where(frequency_grid >= limits, torch.zeros_like(mask), mask)
    mask = torch.where(limits > 0.0, mask, torch.zeros_like(mask))
    spectrum = torch.fft.rfft(padded, dim=-1)
    filtered = torch.fft.irfft(
        spectrum * mask.to(spectrum.dtype),
        n=padded.shape[-1],
        dim=-1,
    )
    return filtered[
        ...,
        reflect_padding_samples : reflect_padding_samples + waveform.shape[-1],
    ]


def _bandlimit_spectrum(spectrum: Tensor, max_hz: Tensor) -> Tensor:

    if max_hz.shape != (spectrum.shape[0],):
        raise ValueError("adversarial_max_hz must have shape [B]")
    bin_hz = 48_000 / 960
    frequencies = (
        torch.arange(spectrum.shape[-2], device=spectrum.device, dtype=torch.float32)
        * bin_hz
    )
    limits = max_hz.to(device=spectrum.device, dtype=frequencies.dtype).view(
        -1, 1, 1, 1
    )
    mask = (frequencies.view(1, 1, -1, 1) <= limits) & (limits > 0)
    return spectrum * mask.to(spectrum.dtype)


def _audio_seconds(batch: Mapping[str, Any]) -> float:
    duration = batch.get("duration_seconds")
    if isinstance(duration, Tensor):
        return float(duration.detach().double().sum().item())
    lengths = batch["audio_lengths"]
    return float(lengths.detach().double().sum().item() / 48_000)


class SpecVAETrainer:

    def __init__(
        self,
        *,
        model: SpecVAE,
        stft: StereoSTFT,
        generator_optimizer: torch.optim.Optimizer,
        reconstruction_loss: SpectralReconstructionLoss | None = None,
        mr_stft_loss: MultiResolutionSTFTLoss | None = None,
        mel_spectral_loss: MelSpectralLoss | None = None,
        source_reconstruction_loss: SourceAlignedMultiResolutionSTFTLoss | None = None,
        source_mixed_scale_loss: (
            SpectroStreamMixedScaleMelLoss | None
        ) = None,
        source_mixed_scale_weight: float = 0.0,
        source_mixed_scale_weight_schedule: Mapping[str, Any] | None = None,
        source_direct_complex_weight: float = 0.0,
        source_if_gd_weight: float = 0.0,
        source_if_gd_min_hz: float = 4_000.0,
        source_if_gd_magnitude_weighted: bool = True,
        source_highband_waveform_l1_weight: float = 0.0,
        source_highband_waveform_l1_bands_hz: tuple[tuple[float, float], ...] = (
            (4_000.0, 8_000.0),
            (8_000.0, 12_000.0),
            (12_000.0, 20_000.0),
        ),
        source_highband_waveform_l1_transition_hz: float = 250.0,
        source_highband_waveform_l1_reflect_padding_samples: int = 2_048,
        source_highband_waveform_l1_target_band_rms_min: float = 5.0e-4,
        source_highband_waveform_l1_target_band_energy_ratio_min: float = 1.0e-4,
        source_highband_waveform_l1_hard_max_hz: float = 20_000.0,
        discriminator: nn.Module | None = None,
        discriminator_optimizer: torch.optim.Optimizer | None = None,
        generator_scheduler: Any | None = None,
        discriminator_scheduler: Any | None = None,
        stage: int = 1,
        discriminator_warmup_steps: int = 0,
        posterior_mode: str = "sample",
        posterior_generator: torch.Generator | None = None,
        posterior_sample_layout: str = "contiguous_bdt",
        stft_weight: float = 1.0,
        mr_stft_weight: float = 0.0,
        mel_weight: float = 0.0,
        waveform_l1_weight: float = 0.0,
        stft_consistency_weight: float = 0.0,
        waveform_peak_envelope_weight: float = 0.0,
        waveform_peak_envelope_window: int = 63,
        waveform_peak_envelope_hop: int = 16,
        waveform_peak_envelope_rms_floor: float = 1.0e-4,
        adversarial_weights: AdversarialWeights | None = None,
        highband_feature_matching_weight: float = 0.0,
        highband_feature_matching_bands_hz: tuple[tuple[float, float], ...] = (),
        adversarial_bandlimit_mode: str = "hard_rfft_v1",
        adversarial_bandlimit_transition_hz: float = 0.0,
        adversarial_bandlimit_reflect_padding_samples: int = 0,
        max_grad_norm: float | None = 1.0,
        model_autocast: bool = False,
        model_autocast_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if stage not in {1, 2}:
            raise ValueError("SpecVAETrainer stage must be 1 or 2")
        if stage == 2 and (discriminator is None or discriminator_optimizer is None):
            raise ValueError("Stage 2 must provide discriminator and optimizer")
        self.model = model
        self.stft = stft
        self.generator_optimizer = generator_optimizer
        self.reconstruction_loss = reconstruction_loss or SpectralReconstructionLoss()
        self.mr_stft_loss = mr_stft_loss
        self.mel_spectral_loss = mel_spectral_loss
        self.source_reconstruction_loss = source_reconstruction_loss
        self.source_mixed_scale_loss = source_mixed_scale_loss
        self.source_mixed_scale_weight = float(source_mixed_scale_weight)
        if (
            not math.isfinite(self.source_mixed_scale_weight)
            or self.source_mixed_scale_weight < 0
        ):
            raise ValueError("source_mixed_scale_weight must be finite and non-negative")
        if self.source_mixed_scale_weight and self.source_mixed_scale_loss is None:
            raise ValueError("mixed-scale weight requires a SpectroStream Equation 6 loss instance")
        if source_mixed_scale_weight_schedule is not None and not isinstance(
            source_mixed_scale_weight_schedule, Mapping
        ):
            raise TypeError("source_mixed_scale_weight_schedule must be a mapping")
        self.source_mixed_scale_weight_schedule = (
            None
            if source_mixed_scale_weight_schedule is None
            else dict(source_mixed_scale_weight_schedule)
        )

        mixed_scale_weight_at_step(
            initial_weight=self.source_mixed_scale_weight,
            schedule=self.source_mixed_scale_weight_schedule,
            global_step=0,
        )
        self.source_direct_complex_weight = float(source_direct_complex_weight)
        if (
            not math.isfinite(self.source_direct_complex_weight)
            or self.source_direct_complex_weight < 0
        ):
            raise ValueError("source_direct_complex_weight must be finite and non-negative")
        if (
            self.source_direct_complex_weight
            and self.source_reconstruction_loss is None
        ):
            raise ValueError("direct-complex weight requires the source reconstruction recipe")
        self.source_if_gd_weight = float(source_if_gd_weight)
        self.source_if_gd_min_hz = float(source_if_gd_min_hz)
        self.source_if_gd_magnitude_weighted = bool(source_if_gd_magnitude_weighted)
        if (
            not math.isfinite(self.source_if_gd_weight)
            or self.source_if_gd_weight < 0.0
            or not math.isfinite(self.source_if_gd_min_hz)
            or not 0.0 <= self.source_if_gd_min_hz < SAMPLE_RATE / 2
        ):
            raise ValueError("invalid source IF/GD configuration")
        if self.source_if_gd_weight and self.source_reconstruction_loss is None:
            raise ValueError("source IF/GD weight requires the source reconstruction recipe")
        self.source_highband_waveform_l1_weight = float(
            source_highband_waveform_l1_weight
        )
        self.source_highband_waveform_l1_bands_hz = tuple(
            (float(lower), float(upper))
            for lower, upper in source_highband_waveform_l1_bands_hz
        )
        self.source_highband_waveform_l1_transition_hz = float(
            source_highband_waveform_l1_transition_hz
        )
        self.source_highband_waveform_l1_reflect_padding_samples = int(
            source_highband_waveform_l1_reflect_padding_samples
        )
        self.source_highband_waveform_l1_target_band_rms_min = float(
            source_highband_waveform_l1_target_band_rms_min
        )
        self.source_highband_waveform_l1_target_band_energy_ratio_min = float(
            source_highband_waveform_l1_target_band_energy_ratio_min
        )
        self.source_highband_waveform_l1_hard_max_hz = float(
            source_highband_waveform_l1_hard_max_hz
        )
        if (
            not math.isfinite(self.source_highband_waveform_l1_weight)
            or self.source_highband_waveform_l1_weight < 0.0
        ):
            raise ValueError("source high-band waveform L1 weight must be finite and non-negative")
        if (
            self.source_highband_waveform_l1_weight
            and self.source_reconstruction_loss is None
        ):
            raise ValueError("source high-band waveform L1 requires the source reconstruction recipe")
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_scheduler = generator_scheduler
        self.discriminator_scheduler = discriminator_scheduler
        self.stage = stage
        if posterior_mode not in {"mean", "sample"}:
            raise ValueError("posterior_mode must be mean or sample")
        self.posterior_mode = posterior_mode
        self.posterior_generator = posterior_generator
        if posterior_sample_layout != "contiguous_bdt":
            raise ValueError("posterior_sample_layout must be contiguous_bdt")
        self.posterior_sample_layout = posterior_sample_layout
        self.discriminator_warmup_steps = int(discriminator_warmup_steps)
        if self.discriminator_warmup_steps < 0:
            raise ValueError("discriminator_warmup_steps must not be negative")
        self.stft_weight = float(stft_weight)
        self.mr_stft_weight = float(mr_stft_weight)
        self.mel_weight = float(mel_weight)
        self.waveform_l1_weight = float(waveform_l1_weight)
        self.stft_consistency_weight = float(stft_consistency_weight)
        if self.stft_consistency_weight < 0:
            raise ValueError("stft_consistency_weight must not be negative")
        self.waveform_peak_envelope_weight = float(waveform_peak_envelope_weight)
        self.waveform_peak_envelope_window = int(waveform_peak_envelope_window)
        self.waveform_peak_envelope_hop = int(waveform_peak_envelope_hop)
        self.waveform_peak_envelope_rms_floor = float(waveform_peak_envelope_rms_floor)
        if (
            not math.isfinite(self.waveform_peak_envelope_weight)
            or self.waveform_peak_envelope_weight < 0
        ):
            raise ValueError("waveform_peak_envelope_weight must be finite and non-negative")
        if self.waveform_peak_envelope_window <= 0 or (
            self.waveform_peak_envelope_window % 2 == 0
        ):
            raise ValueError("waveform_peak_envelope_window must be a positive odd integer")
        if self.waveform_peak_envelope_hop <= 0:
            raise ValueError("waveform_peak_envelope_hop must be positive")
        if (
            not math.isfinite(self.waveform_peak_envelope_rms_floor)
            or self.waveform_peak_envelope_rms_floor <= 0
        ):
            raise ValueError("waveform_peak_envelope_rms_floor must be finite and positive")
        self.adversarial_weights = adversarial_weights or AdversarialWeights()
        self.highband_feature_matching_weight = float(highband_feature_matching_weight)
        self.highband_feature_matching_bands_hz = tuple(
            (float(lower), float(upper))
            for lower, upper in highband_feature_matching_bands_hz
        )
        if (
            not math.isfinite(self.highband_feature_matching_weight)
            or self.highband_feature_matching_weight < 0.0
        ):
            raise ValueError("high-band feature-matching weight must be finite and non-negative")
        if self.highband_feature_matching_bands_hz and any(
            not 0.0 <= lower < upper <= SAMPLE_RATE / 2
            for lower, upper in self.highband_feature_matching_bands_hz
        ):
            raise ValueError("invalid high-band feature-matching frequency configuration")
        self.adversarial_bandlimit_mode = str(adversarial_bandlimit_mode)
        self.adversarial_bandlimit_transition_hz = float(
            adversarial_bandlimit_transition_hz
        )
        self.adversarial_bandlimit_reflect_padding_samples = int(
            adversarial_bandlimit_reflect_padding_samples
        )
        if self.adversarial_bandlimit_mode not in {
            "none_v1",
            "hard_rfft_v1",
            "reflect_cosine_v1",
        }:
            raise ValueError("unsupported adversarial band-limit mode")
        if self.adversarial_bandlimit_mode in {"none_v1", "hard_rfft_v1"}:
            if (
                self.adversarial_bandlimit_transition_hz != 0.0
                or self.adversarial_bandlimit_reflect_padding_samples != 0
            ):
                raise ValueError("none_v1 and hard_rfft_v1 do not allow a transition band or reflection padding")
        elif (
            not math.isfinite(self.adversarial_bandlimit_transition_hz)
            or self.adversarial_bandlimit_transition_hz <= 0.0
            or self.adversarial_bandlimit_reflect_padding_samples <= 0
        ):
            raise ValueError("invalid reflect_cosine_v1 configuration")
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.model_autocast = bool(model_autocast)
        self.model_autocast_dtype = model_autocast_dtype

    def train_step(
        self,
        batch: Mapping[str, Any],
        *,
        global_step: int,
    ) -> TrainStepResult:
        source_mixed_scale_weight = mixed_scale_weight_at_step(
            initial_weight=self.source_mixed_scale_weight,
            schedule=self.source_mixed_scale_weight_schedule,
            global_step=global_step,
        )
        assert_distributed_consensus(
            "render_stage12_update_plan",
            {
                "stage": self.stage,
                "global_step": int(global_step),
                "source_mixed_scale_weight": source_mixed_scale_weight,
                "discriminator_update": (
                    self.stage == 2 and global_step >= self.discriminator_warmup_steps
                ),
            },
        )
        required_batch_fields = (
            "audio",
            "audio_lengths",
            "duration_seconds",
            "media_bandwidth_hz",
            "magnitude_max_hz",
            "phase_max_hz",
            "stereo_max_hz",
            "adversarial_max_hz",
            "waveform_adversarial_enabled",
        )

        def validate_batch_contract() -> bool:
            missing = [name for name in required_batch_fields if name not in batch]
            if missing:
                raise KeyError(f"render batch is missing fields: {missing}")
            audio_value = batch["audio"]
            lengths_value = batch["audio_lengths"]
            if not isinstance(audio_value, Tensor) or audio_value.ndim != 3:
                raise TypeError("audio must be a tensor with shape [B, 2, N]")
            if audio_value.shape[1] != 2:
                raise ValueError("render batch audio must be stereo")
            if not isinstance(lengths_value, Tensor) or lengths_value.shape != (
                audio_value.shape[0],
            ):
                raise ValueError("audio_lengths must have shape [B]")
            if bool((lengths_value <= 0).any()) or bool(
                (lengths_value > audio_value.shape[-1]).any()
            ):
                raise ValueError("audio_lengths is out of bounds")
            for name in required_batch_fields[3:]:
                value = batch[name]
                if not isinstance(value, Tensor) or value.shape != (
                    audio_value.shape[0],
                ):
                    raise ValueError(f"{name} must be a tensor with shape [B]")
            if self.stage == 2 and not bool(
                batch["waveform_adversarial_enabled"].all()
            ):
                raise ValueError("Stage 2 batch contains samples with waveform adversarial training disabled")
            return True

        synchronized_call("Spec-VAE batch contract", validate_batch_contract)
        local_seconds, global_seconds, backward_scale = distributed_audio_weight(batch)
        audio = batch["audio"]
        audio_lengths = batch["audio_lengths"]
        media_bandwidth_hz = batch["media_bandwidth_hz"]
        magnitude_max_hz = batch["magnitude_max_hz"]
        phase_max_hz = batch["phase_max_hz"]
        stereo_max_hz = batch["stereo_max_hz"]
        adversarial_max_hz = batch["adversarial_max_hz"]
        waveform_adversarial_enabled = batch["waveform_adversarial_enabled"]

        def acoustic_forward() -> tuple[Any, Any, Tensor]:
            target = self.stft.analyze(audio, audio_lengths)
            with torch.autocast(
                device_type=audio.device.type,
                dtype=self.model_autocast_dtype,
                enabled=self.model_autocast and audio.device.type == "cuda",
            ):
                posterior_arguments = posterior_forward_arguments(
                    posterior_mode=self.posterior_mode,
                    posterior_generator=self.posterior_generator,
                    posterior_sample_layout=self.posterior_sample_layout,
                )
                model_output = self.model(
                    target.spectrum,
                    target.spectrum_lengths,
                    **posterior_arguments,
                )
            generated_audio = self.stft.inverse(
                model_output.reconstruction,
                audio_lengths,
                dtype=torch.float32,
            )
            return target, model_output, generated_audio

        target_stft, output, fake_audio = synchronized_call(
            "Spec-VAE acoustic forward", acoustic_forward
        )
        gan_real_audio = audio.float()
        gan_fake_audio = fake_audio
        if self.stage == 2 and self.adversarial_bandlimit_mode != "none_v1":
            gan_real_audio = _bandlimit_waveform(
                gan_real_audio,
                adversarial_max_hz,
                mode=self.adversarial_bandlimit_mode,
                transition_hz=self.adversarial_bandlimit_transition_hz,
                reflect_padding_samples=(
                    self.adversarial_bandlimit_reflect_padding_samples
                ),
            )
            gan_fake_audio = _bandlimit_waveform(
                gan_fake_audio,
                adversarial_max_hz,
                mode=self.adversarial_bandlimit_mode,
                transition_hz=self.adversarial_bandlimit_transition_hz,
                reflect_padding_samples=(
                    self.adversarial_bandlimit_reflect_padding_samples
                ),
            )
        discriminator_metrics: dict[str, Tensor] = {}
        discriminator_active = (
            self.stage == 2 and global_step >= self.discriminator_warmup_steps
        )
        if discriminator_active:
            assert self.discriminator is not None
            assert self.discriminator_optimizer is not None
            discriminator_metrics = discriminator_step(
                self.discriminator,
                self.discriminator_optimizer,
                gan_real_audio,
                gan_fake_audio,
                lengths=audio_lengths,
                max_grad_norm=self.max_grad_norm,
                backward_scale=backward_scale,
            )
            if self.discriminator_scheduler is not None:
                self.discriminator_scheduler.step()

        self.generator_optimizer.zero_grad(set_to_none=True)
        highband_components = None
        highband_backward = None
        highband_global_eligible = None
        if self.source_highband_waveform_l1_weight:
            highband_components = synchronized_call(
                "Spec-VAE highband waveform L1",
                lambda: multiband_filtered_waveform_l1_components(
                    fake_audio,
                    audio.float(),
                    audio_lengths,
                    media_bandwidth_hz,
                    magnitude_max_hz,
                    phase_max_hz,
                    stereo_max_hz,
                    bands_hz=self.source_highband_waveform_l1_bands_hz,
                    transition_hz=self.source_highband_waveform_l1_transition_hz,
                    reflect_padding_samples=(
                        self.source_highband_waveform_l1_reflect_padding_samples
                    ),
                    target_band_rms_min=(
                        self.source_highband_waveform_l1_target_band_rms_min
                    ),
                    target_band_energy_ratio_min=(
                        self.source_highband_waveform_l1_target_band_energy_ratio_min
                    ),
                    hard_max_hz=self.source_highband_waveform_l1_hard_max_hz,
                ),
            )
            highband_backward, highband_global_eligible = (
                distributed_eligible_record_mean_loss(
                    highband_components,
                    backward_scale=backward_scale,
                )
            )

        precomputed_adversarial_result: GeneratorAdversarialResult | None = None
        highband_fm_reduction: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
        if discriminator_active and self.highband_feature_matching_bands_hz:
            assert self.discriminator is not None
            precomputed_adversarial_result = synchronized_call(
                "Spec-VAE generator adversarial terms",
                lambda: generator_adversarial_terms(
                    self.discriminator,
                    gan_real_audio,
                    gan_fake_audio,
                    lengths=audio_lengths,
                    highband_adversarial_max_hz=adversarial_max_hz,
                    highband_bands_hz=self.highband_feature_matching_bands_hz,
                ),
            )
            if precomputed_adversarial_result.highband_components is not None:


                highband_fm_reduction = distributed_highband_feature_matching_loss(
                    precomputed_adversarial_result.highband_components,
                    backward_scale=backward_scale,
                )

        def generator_loss() -> tuple[Tensor, dict[str, Tensor]]:
            if self.source_reconstruction_loss is not None:
                assembly = assemble_source_anchored_stage1_loss(
                    source_reconstruction_loss=self.source_reconstruction_loss,
                    source_mixed_scale_loss=self.source_mixed_scale_loss,
                    source_mixed_scale_weight=source_mixed_scale_weight,
                    source_direct_complex_weight=self.source_direct_complex_weight,
                    source_if_gd_weight=self.source_if_gd_weight,
                    source_if_gd_min_hz=self.source_if_gd_min_hz,
                    source_if_gd_magnitude_weighted=(
                        self.source_if_gd_magnitude_weighted
                    ),
                    waveform_l1_weight=self.waveform_l1_weight,
                    stft_weight=self.stft_weight,
                    kl_weight=self.reconstruction_loss.config.kl,
                    output=output,
                    target_stft=target_stft,
                    fake_audio=fake_audio,
                    target_audio=audio,
                    audio_lengths=audio_lengths,
                    media_bandwidth_hz=media_bandwidth_hz,
                    magnitude_max_hz=magnitude_max_hz,
                    phase_max_hz=phase_max_hz,
                    waveform_adversarial_enabled=waveform_adversarial_enabled,
                )
                total = assembly.total
                terms = assembly.terms
                if highband_components is not None:
                    assert highband_backward is not None
                    assert highband_global_eligible is not None
                    local_count = highband_components.local_eligible_records.to(
                        dtype=highband_components.local_loss_sum.dtype
                    )
                    highband_local_mean = torch.where(
                        local_count > 0,
                        highband_components.local_loss_sum / local_count.clamp_min(1.0),
                        highband_components.local_loss_sum * 0.0,
                    )
                    highband_weighted = (
                        self.source_highband_waveform_l1_weight * highband_backward
                    )
                    total = total + highband_weighted
                    terms["source_highband_waveform_l1"] = highband_local_mean
                    terms["source_highband_waveform_l1_backward"] = highband_backward
                    terms["source_highband_waveform_l1_weighted"] = highband_weighted
                    terms["source_highband_waveform_l1_local_eligible_records"] = (
                        highband_components.local_eligible_records
                    )
                    terms["source_highband_waveform_l1_global_eligible_records"] = (
                        highband_global_eligible
                    )
                    for band_index, eligible in enumerate(
                        highband_components.eligible_band_mask.unbind(dim=1)
                    ):
                        eligible_float = eligible.to(
                            highband_components.per_band_l1.dtype
                        )
                        eligible_count = eligible_float.sum()
                        band_mean = torch.where(
                            eligible_count > 0,
                            (
                                highband_components.per_band_l1[:, band_index]
                                * eligible_float
                            ).sum()
                            / eligible_count.clamp_min(1.0),
                            highband_components.per_band_l1[:, band_index].sum() * 0.0,
                        )
                        terms[f"source_highband_waveform_l1_band_{band_index}"] = (
                            band_mean
                        )
                        terms[
                            f"source_highband_waveform_l1_band_{band_index}_eligible"
                        ] = eligible_count
            else:
                reconstruction = self.reconstruction_loss(
                    output.reconstruction,
                    target_stft.spectrum,
                    target_stft.spectrum_mask,
                    posterior_mean=output.posterior.mean,
                    posterior_logvar=output.posterior.logvar,
                    latent_mask=output.latent_mask,
                    media_bandwidth_hz=media_bandwidth_hz,
                    magnitude_max_hz=magnitude_max_hz,
                    phase_max_hz=phase_max_hz,
                    stereo_max_hz=stereo_max_hz,
                )
                total = self.stft_weight * reconstruction["total"]
                terms = dict(reconstruction)
                if self.waveform_l1_weight:
                    if not bool(waveform_adversarial_enabled.all()):
                        raise ValueError("waveform L1 requires explicitly enabled waveform supervision")
                    waveform_l1 = waveform_l1_loss(
                        fake_audio,
                        audio.float(),
                        audio_lengths,
                    )
                    terms["waveform_l1"] = waveform_l1
                    total = total + self.waveform_l1_weight * waveform_l1
                if self.waveform_peak_envelope_weight:
                    if not bool(waveform_adversarial_enabled.all()):
                        raise ValueError(
                            "waveform peak envelope requires waveform supervision "
                            "to be enabled for the sample"
                        )
                    peak_envelope = waveform_peak_envelope_overshoot_loss(
                        fake_audio,
                        audio.float(),
                        audio_lengths,
                        window_size=self.waveform_peak_envelope_window,
                        hop_size=self.waveform_peak_envelope_hop,
                        rms_floor=self.waveform_peak_envelope_rms_floor,
                    )
                    terms["waveform_peak_envelope_overshoot"] = peak_envelope
                    total = total + self.waveform_peak_envelope_weight * peak_envelope
                if self.mr_stft_loss is not None and self.mr_stft_weight:
                    mr_stft = self.mr_stft_loss(
                        fake_audio,
                        audio.float(),
                        audio_lengths,
                        magnitude_max_hz,
                    )
                    terms["mr_stft"] = mr_stft
                    total = total + self.stft_weight * self.mr_stft_weight * mr_stft
                if self.mel_spectral_loss is not None and self.mel_weight:
                    mel = self.mel_spectral_loss(
                        fake_audio,
                        audio.float(),
                        audio_lengths,
                        magnitude_max_hz,
                    )
                    terms["mel"] = mel
                    total = total + self.mel_weight * mel
                if self.stft_consistency_weight:
                    projected = self.stft.analyze(fake_audio, audio_lengths)
                    consistency = spectral_complex_l1_loss(
                        output.reconstruction,
                        projected.spectrum,
                        target_stft.spectrum_mask,
                    )
                    terms["stft_consistency"] = consistency
                    total = total + self.stft_consistency_weight * consistency
            if discriminator_active:
                assert self.discriminator is not None
                adversarial_result = (
                    precomputed_adversarial_result
                    if precomputed_adversarial_result is not None
                    else generator_adversarial_terms(
                        self.discriminator,
                        gan_real_audio,
                        gan_fake_audio,
                        lengths=audio_lengths,
                    )
                )
                adversarial = adversarial_result.terms
                terms.update(adversarial)
                if (
                    self.highband_feature_matching_weight
                    and adversarial_result.highband_components is None
                ):
                    raise RuntimeError(
                        "high-band feature matching is enabled, but the "
                        "discriminator returned no band features"
                    )
                total = (
                    total
                    + self.adversarial_weights.adversarial * adversarial["adversarial"]
                    + self.adversarial_weights.feature_matching
                    * adversarial["feature_matching"]
                )
                if adversarial_result.highband_components is not None:
                    assert highband_fm_reduction is not None
                    (
                        highband_fm_backward,
                        highband_band_means,
                        highband_global_counts,
                        highband_global_eligible_band_count,
                    ) = highband_fm_reduction
                    total = (
                        total
                        + self.highband_feature_matching_weight * highband_fm_backward
                    )
                    globally_eligible = highband_global_counts > 0
                    highband_global_mean = torch.where(
                        highband_global_eligible_band_count > 0,
                        (
                            highband_band_means
                            * globally_eligible.to(highband_band_means.dtype)
                        ).sum()
                        / highband_global_eligible_band_count.clamp_min(1.0),
                        highband_band_means.sum() * 0.0,
                    )
                    terms["highband_feature_matching"] = highband_global_mean
                    terms["highband_feature_matching_weighted"] = (
                        self.highband_feature_matching_weight * highband_global_mean
                    )
                    terms[
                        "highband_feature_matching_global_eligible_record_band_pairs"
                    ] = highband_global_counts.sum()
                    terms["highband_feature_matching_global_eligible_bands"] = (
                        highband_global_eligible_band_count
                    )
                    for band_index, (band_mean, eligible_count) in enumerate(
                        zip(
                            highband_band_means,
                            highband_global_counts,
                            strict=True,
                        )
                    ):
                        terms[f"highband_feature_matching_band_{band_index}"] = (
                            band_mean
                        )
                        terms[
                            f"highband_feature_matching_band_{band_index}_eligible"
                        ] = eligible_count
            if not torch.isfinite(total):
                raise FloatingPointError("Spec-VAE generator loss is NaN/Inf")
            return total, terms

        total, terms = synchronized_call("Spec-VAE generator loss", generator_loss)
        (total * backward_scale).backward()

        def optimize_generator() -> Tensor:
            gradient_norm = (
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                    error_if_nonfinite=True,
                )
                if self.max_grad_norm is not None
                else finite_gradient_norm(self.model.parameters())
            )
            self.generator_optimizer.step()
            return torch.as_tensor(gradient_norm).detach()

        generator_grad_norm = synchronized_call(
            "Spec-VAE generator optimizer step",
            optimize_generator,
        )
        if self.generator_scheduler is not None:
            synchronized_call(
                "Spec-VAE generator scheduler step",
                self.generator_scheduler.step,
            )
        terms.update(discriminator_metrics)
        terms["generator"] = total.detach()
        terms["generator_grad_norm"] = torch.as_tensor(generator_grad_norm).detach()
        metrics = {
            name: float(value.detach().float().item())
            for name, value in terms.items()
            if value.numel() == 1
        }
        return TrainStepResult(
            loss=total.detach(),
            metrics=metrics,
            numerators={name: value * local_seconds for name, value in metrics.items()},
            denominators={name: local_seconds for name in metrics},
            discriminator_updated=discriminator_active,
            local_audio_seconds=local_seconds,
            global_audio_seconds=global_seconds,
            backward_scale=backward_scale,
        )


class RefinerTrainer:

    def __init__(
        self,
        *,
        vae: SpecVAE,
        refiner: EarVAE2PublicRefiner,
        stft: StereoSTFT,
        generator_optimizer: torch.optim.Optimizer,
        reconstruction_loss: SpectralReconstructionLoss | None = None,
        mr_stft_loss: MultiResolutionSTFTLoss | None = None,
        mel_spectral_loss: MelSpectralLoss | None = None,
        source_reconstruction_loss: SourceAlignedMultiResolutionSTFTLoss | None = None,
        source_mixed_scale_loss: (
            SpectroStreamMixedScaleMelLoss | None
        ) = None,
        source_mixed_scale_weight: float = 0.0,
        incremental_if_gd_weight: float = 0.0,
        incremental_if_gd_objective: str = "ear_wrapped_l1_v1",
        incremental_if_gd_sample_equal: bool = True,
        mr_stft_weight: float = 0.0,
        mel_weight: float = 0.0,
        waveform_l1_weight: float = 0.0,
        stft_consistency_weight: float = 0.0,
        reconstruction_domain: str = "raw",
        mr_ccpc_weight: float = 0.0,
        mr_ccpc_no_regression_weight: float = 0.0,
        mr_ccpc_no_regression_margin: float = 0.0,
        mr_ccpc_no_regression_allowed_regression: float = 0.0,
        waveform_peak_envelope_weight: float = 0.0,
        waveform_peak_envelope_window: int = 63,
        waveform_peak_envelope_hop: int = 16,
        waveform_peak_envelope_rms_floor: float = 1.0e-4,
        refiner_residual_energy_weight: float = 0.0,
        refiner_residual_energy_rms_floor: float = 1.0e-4,
        refiner_oracle_log_magnitude_weight: float = 0.0,
        refiner_oracle_log_magnitude_beta: float = 0.1,
        refiner_oracle_log_magnitude_sample_equal: bool = True,
        refiner_oracle_log_magnitude_target: str = "final",
        si_sdr_no_regression_weight: float = 0.0,
        si_sdr_no_regression_margin_db: float = 0.0,
        waveform_discriminator: nn.Module | None = None,
        waveform_discriminator_optimizer: torch.optim.Optimizer | None = None,
        spectral_discriminator: nn.Module | None = None,
        spectral_discriminator_optimizer: torch.optim.Optimizer | None = None,
        generator_scheduler: Any | None = None,
        waveform_discriminator_scheduler: Any | None = None,
        spectral_discriminator_scheduler: Any | None = None,
        waveform_weights: AdversarialWeights = AdversarialWeights(0.175, 20.0),
        spectral_weights: AdversarialWeights = AdversarialWeights(0.5, 35.0),
        generator_steps_per_discriminator: int = 4,
        spectral_warmup_steps: int = 15_000,
        waveform_warmup_steps: int = 0,
        vae_posterior_mode: str = "mean",
        stft_weight: float = 0.5,
        max_grad_norm: float = 1.0,
    ) -> None:
        if generator_steps_per_discriminator <= 0:
            raise ValueError("generator-to-discriminator update ratio must be positive")
        self.vae = vae.eval()
        for parameter in self.vae.parameters():
            parameter.requires_grad_(False)
        self.refiner = refiner
        refiner_module = refiner.module if hasattr(refiner, "module") else refiner
        if not isinstance(refiner_module, EarVAE2PublicRefiner):
            raise TypeError("refiner must wrap EarVAE2PublicRefiner")
        self.refiner_magnitude_mode_mask = refiner_module.magnitude_mask.detach() >= 0.5
        self.refiner_magnitude_mode_weights = (
            refiner_module.magnitude_mask.detach().float()
        )
        self.refiner_phase_mode_mask = refiner_module.phase_mask.detach() >= 0.5
        self.refiner_max_abs_log_magnitude_residual = float(
            refiner_module.config.max_log_magnitude_residual
        )
        self.refiner_magnitude_gate_enabled = bool(
            refiner_module.config.magnitude_gate_enabled
        )
        self.stft = stft
        self.generator_optimizer = generator_optimizer
        self.reconstruction_loss = reconstruction_loss or SpectralReconstructionLoss()
        self.mr_stft_loss = mr_stft_loss
        self.mel_spectral_loss = mel_spectral_loss
        self.source_reconstruction_loss = source_reconstruction_loss
        self.source_mixed_scale_loss = source_mixed_scale_loss
        self.source_mixed_scale_weight = float(source_mixed_scale_weight)
        self.incremental_if_gd_weight = float(incremental_if_gd_weight)
        self.incremental_if_gd_objective = str(incremental_if_gd_objective)
        self.incremental_if_gd_sample_equal = bool(incremental_if_gd_sample_equal)
        for name, value in (
            ("source_mixed_scale_weight", self.source_mixed_scale_weight),
            ("incremental_if_gd_weight", self.incremental_if_gd_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.source_mixed_scale_weight and self.source_mixed_scale_loss is None:
            raise ValueError("Refiner mixed-scale weight requires a parent loss instance")
        if (
            self.source_mixed_scale_loss is not None
            and source_reconstruction_loss is None
        ):
            raise ValueError("Refiner mixed-scale loss must be rebuilt only by inheriting the parent loss")
        if self.incremental_if_gd_weight and source_reconstruction_loss is None:
            raise ValueError("Refiner incremental IF/GD requires rebuilding the parent loss")
        if source_reconstruction_loss is not None and (
            self.incremental_if_gd_objective != "ear_wrapped_l1_v1"
        ):
            raise ValueError("Refiner incremental IF/GD target changed unexpectedly")
        self.mr_stft_weight = float(mr_stft_weight)
        self.mel_weight = float(mel_weight)
        self.waveform_l1_weight = float(waveform_l1_weight)
        self.stft_consistency_weight = float(stft_consistency_weight)
        if self.stft_consistency_weight < 0:
            raise ValueError("stft_consistency_weight must not be negative")
        if reconstruction_domain not in {"raw", "projected"}:
            raise ValueError("reconstruction_domain must be raw or projected")
        self.reconstruction_domain = reconstruction_domain
        self.mr_ccpc_weight = float(mr_ccpc_weight)
        if self.mr_ccpc_weight < 0:
            raise ValueError("mr_ccpc_weight must not be negative")
        self.mr_ccpc_no_regression_weight = float(mr_ccpc_no_regression_weight)
        self.mr_ccpc_no_regression_margin = float(mr_ccpc_no_regression_margin)
        self.mr_ccpc_no_regression_allowed_regression = float(
            mr_ccpc_no_regression_allowed_regression
        )
        self.waveform_peak_envelope_weight = float(waveform_peak_envelope_weight)
        self.waveform_peak_envelope_window = int(waveform_peak_envelope_window)
        self.waveform_peak_envelope_hop = int(waveform_peak_envelope_hop)
        self.waveform_peak_envelope_rms_floor = float(waveform_peak_envelope_rms_floor)
        self.refiner_residual_energy_weight = float(refiner_residual_energy_weight)
        self.refiner_residual_energy_rms_floor = float(
            refiner_residual_energy_rms_floor
        )
        self.refiner_oracle_log_magnitude_weight = float(
            refiner_oracle_log_magnitude_weight
        )
        if self.refiner_oracle_log_magnitude_weight:
            raise ValueError(
                "linear-amplitude Refiner does not support log-magnitude oracle loss"
            )
        self.refiner_oracle_log_magnitude_beta = float(
            refiner_oracle_log_magnitude_beta
        )
        self.refiner_oracle_log_magnitude_sample_equal = bool(
            refiner_oracle_log_magnitude_sample_equal
        )
        if refiner_oracle_log_magnitude_target != "final":
            raise ValueError("refiner_oracle_log_magnitude_target must be final")
        self.refiner_oracle_log_magnitude_target = refiner_oracle_log_magnitude_target
        self.si_sdr_no_regression_weight = float(si_sdr_no_regression_weight)
        self.si_sdr_no_regression_margin_db = float(si_sdr_no_regression_margin_db)
        for name, value in (
            ("waveform_peak_envelope_weight", self.waveform_peak_envelope_weight),
            (
                "mr_ccpc_no_regression_weight",
                self.mr_ccpc_no_regression_weight,
            ),
            (
                "mr_ccpc_no_regression_margin",
                self.mr_ccpc_no_regression_margin,
            ),
            (
                "mr_ccpc_no_regression_allowed_regression",
                self.mr_ccpc_no_regression_allowed_regression,
            ),
            ("refiner_residual_energy_weight", self.refiner_residual_energy_weight),
            (
                "refiner_oracle_log_magnitude_weight",
                self.refiner_oracle_log_magnitude_weight,
            ),
            ("si_sdr_no_regression_weight", self.si_sdr_no_regression_weight),
            ("si_sdr_no_regression_margin_db", self.si_sdr_no_regression_margin_db),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.waveform_peak_envelope_window <= 0 or (
            self.waveform_peak_envelope_window % 2 == 0
        ):
            raise ValueError("waveform_peak_envelope_window must be a positive odd integer")
        if self.waveform_peak_envelope_hop <= 0:
            raise ValueError("waveform_peak_envelope_hop must be positive")
        for name, value in (
            (
                "waveform_peak_envelope_rms_floor",
                self.waveform_peak_envelope_rms_floor,
            ),
            (
                "refiner_residual_energy_rms_floor",
                self.refiner_residual_energy_rms_floor,
            ),
            (
                "refiner_oracle_log_magnitude_beta",
                self.refiner_oracle_log_magnitude_beta,
            ),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.needs_coarse_audio = bool(
            self.refiner_residual_energy_weight
            or self.si_sdr_no_regression_weight
            or self.mr_ccpc_no_regression_weight
        )
        self.waveform_discriminator = waveform_discriminator
        self.waveform_discriminator_optimizer = waveform_discriminator_optimizer
        self.spectral_discriminator = spectral_discriminator
        self.spectral_discriminator_input_domain = _discriminator_input_domain(
            spectral_discriminator
        )
        self.spectral_discriminator_optimizer = spectral_discriminator_optimizer
        self.generator_scheduler = generator_scheduler
        self.waveform_discriminator_scheduler = waveform_discriminator_scheduler
        self.spectral_discriminator_scheduler = spectral_discriminator_scheduler
        self.waveform_weights = waveform_weights
        self.spectral_weights = spectral_weights
        self.generator_steps_per_discriminator = generator_steps_per_discriminator
        self.spectral_warmup_steps = spectral_warmup_steps
        if vae_posterior_mode not in {"mean", "sample"}:
            raise ValueError("vae_posterior_mode must be mean or sample")
        self.vae_posterior_mode = vae_posterior_mode
        self.waveform_warmup_steps = int(waveform_warmup_steps)
        if self.waveform_warmup_steps < 0:
            raise ValueError("waveform_warmup_steps must not be negative")
        self.stft_weight = stft_weight
        self.max_grad_norm = max_grad_norm

    def train_step(
        self,
        batch: Mapping[str, Any],
        *,
        global_step: int,
    ) -> TrainStepResult:
        update_discriminator = global_step % self.generator_steps_per_discriminator == 0
        spectral_active = global_step >= self.spectral_warmup_steps
        waveform_active = global_step >= self.waveform_warmup_steps
        assert_distributed_consensus(
            "render_stage3_update_plan",
            {
                "global_step": int(global_step),
                "waveform_discriminator_update": (
                    update_discriminator and waveform_active
                ),
                "spectral_active": spectral_active,
                "spectral_discriminator_update": (
                    update_discriminator and spectral_active
                ),
            },
        )

        def validate_batch_contract() -> bool:
            required = (
                "audio",
                "audio_lengths",
                "duration_seconds",
                "media_bandwidth_hz",
                "magnitude_max_hz",
                "phase_max_hz",
                "stereo_max_hz",
                "adversarial_max_hz",
                "waveform_adversarial_enabled",
            )
            missing = [name for name in required if name not in batch]
            if missing:
                raise KeyError(f"Refiner batch is missing fields: {missing}")
            audio_value = batch["audio"]
            lengths_value = batch["audio_lengths"]
            if (
                not isinstance(audio_value, Tensor)
                or audio_value.ndim != 3
                or audio_value.shape[1] != 2
            ):
                raise ValueError("Refiner audio must have shape [B, 2, N]")
            if not isinstance(lengths_value, Tensor) or lengths_value.shape != (
                audio_value.shape[0],
            ):
                raise ValueError("Refiner audio_lengths must have shape [B]")
            if bool((lengths_value <= 0).any()) or bool(
                (lengths_value > audio_value.shape[-1]).any()
            ):
                raise ValueError("Refiner audio_lengths is out of bounds")
            return True

        synchronized_call("Refiner batch contract", validate_batch_contract)
        local_seconds, global_seconds, backward_scale = distributed_audio_weight(batch)
        audio = batch["audio"].float()
        lengths = batch["audio_lengths"]
        media_bandwidth_hz = batch["media_bandwidth_hz"]
        magnitude_max_hz = batch["magnitude_max_hz"]
        phase_max_hz = batch["phase_max_hz"]
        stereo_max_hz = batch["stereo_max_hz"]
        adversarial_max_hz = batch["adversarial_max_hz"]
        waveform_adversarial_enabled = batch["waveform_adversarial_enabled"]

        def acoustic_forward() -> tuple[
            Any,
            Tensor,
            Tensor,
            Tensor,
            Tensor | None,
            Tensor | None,
            Tensor | None,
        ]:
            target = self.stft.analyze(audio, lengths)
            with torch.no_grad():
                vae_output = self.vae(
                    target.spectrum,
                    target.spectrum_lengths,
                    sample_posterior=self.vae_posterior_mode == "sample",
                )
                coarse = vae_output.reconstruction
            refined_with_nyquist = apply_refiner(self.refiner, coarse)
            proposal_spectrum = None
            magnitude_gate = None
            refined_spectrum = refiner_spectrum_for_loss(refined_with_nyquist)
            generated_audio = inverse_refiner_spectrum(
                self.stft,
                (
                    refined_spectrum
                    if refined_with_nyquist is None
                    else refined_with_nyquist
                ),
                lengths,
                dtype=torch.float32,
            )
            coarse_audio = (
                self.stft.inverse(coarse, lengths, dtype=torch.float32).detach()
                if self.needs_coarse_audio
                else None
            )
            return (
                target,
                coarse,
                refined_spectrum,
                generated_audio,
                coarse_audio,
                proposal_spectrum,
                magnitude_gate,
            )

        (
            target_stft,
            coarse_spectrum,
            refined,
            fake_audio,
            coarse_audio,
            proposal_spectrum,
            magnitude_gate,
        ) = synchronized_call("Refiner acoustic forward", acoustic_forward)
        gan_real_audio = audio
        gan_fake_audio = fake_audio
        if self.waveform_discriminator is not None:
            gan_real_audio = _bandlimit_waveform(gan_real_audio, adversarial_max_hz)
            gan_fake_audio = _bandlimit_waveform(gan_fake_audio, adversarial_max_hz)
        gan_real_spectral_input = target_stft.spectrum
        gan_fake_spectral_input = refined
        spectral_input_lengths = target_stft.spectrum_lengths
        if self.spectral_discriminator is not None:
            if self.spectral_discriminator_input_domain == "waveform":
                gan_real_spectral_input = _bandlimit_waveform(
                    audio,
                    adversarial_max_hz,
                )
                gan_fake_spectral_input = _bandlimit_waveform(
                    fake_audio,
                    adversarial_max_hz,
                )
                spectral_input_lengths = lengths
            else:
                gan_real_spectral_input = _bandlimit_spectrum(
                    target_stft.spectrum,
                    adversarial_max_hz,
                )
                gan_fake_spectral_input = _bandlimit_spectrum(
                    refined,
                    adversarial_max_hz,
                )
        d_terms: dict[str, Tensor] = {}
        if (
            update_discriminator
            and waveform_active
            and self.waveform_discriminator is not None
            and self.waveform_discriminator_optimizer is not None
        ):
            if not bool(waveform_adversarial_enabled.all()):
                raise ValueError("waveform GAN batch contains samples with waveform supervision disabled")
            result = discriminator_step(
                self.waveform_discriminator,
                self.waveform_discriminator_optimizer,
                gan_real_audio,
                gan_fake_audio,
                lengths=lengths,
                max_grad_norm=self.max_grad_norm,
                backward_scale=backward_scale,
            )
            d_terms.update({f"waveform_{key}": value for key, value in result.items()})
            if self.waveform_discriminator_scheduler is not None:
                self.waveform_discriminator_scheduler.step()
        if (
            update_discriminator
            and spectral_active
            and self.spectral_discriminator is not None
            and self.spectral_discriminator_optimizer is not None
        ):
            result = discriminator_step(
                self.spectral_discriminator,
                self.spectral_discriminator_optimizer,
                gan_real_spectral_input,
                gan_fake_spectral_input,
                lengths=spectral_input_lengths,
                max_grad_norm=self.max_grad_norm,
                backward_scale=backward_scale,
            )
            d_terms.update({f"spectral_{key}": value for key, value in result.items()})
            if self.spectral_discriminator_scheduler is not None:
                self.spectral_discriminator_scheduler.step()

        self.generator_optimizer.zero_grad(set_to_none=True)

        def generator_loss() -> tuple[Tensor, dict[str, Tensor]]:
            if self.source_reconstruction_loss is not None:
                assembly = assemble_stage3_incremental_loss(
                    source_reconstruction_loss=self.source_reconstruction_loss,
                    source_mixed_scale_loss=self.source_mixed_scale_loss,
                    source_mixed_scale_weight=self.source_mixed_scale_weight,
                    stft_weight=self.stft_weight,
                    if_gd_weight=self.incremental_if_gd_weight,
                    if_gd_objective=self.incremental_if_gd_objective,
                    if_gd_sample_equal=self.incremental_if_gd_sample_equal,
                    refined_spectrum=refined,
                    target_stft=target_stft,
                    fake_audio=fake_audio,
                    target_audio=audio,
                    audio_lengths=lengths,
                    media_bandwidth_hz=media_bandwidth_hz,
                    magnitude_max_hz=magnitude_max_hz,
                    phase_max_hz=phase_max_hz,
                    phase_mode_mask=self.refiner_phase_mode_mask,
                )
                total = assembly.total
                terms = dict(assembly.terms)
            else:
                reconstruction_estimate = refined
                if self.reconstruction_domain == "projected":
                    reconstruction_estimate = self.stft.analyze(
                        fake_audio,
                        lengths,
                    ).spectrum
                reconstruction = self.reconstruction_loss(
                    reconstruction_estimate,
                    target_stft.spectrum,
                    target_stft.spectrum_mask,
                    media_bandwidth_hz=media_bandwidth_hz,
                    magnitude_max_hz=magnitude_max_hz,
                    phase_max_hz=phase_max_hz,
                    stereo_max_hz=stereo_max_hz,
                    magnitude_mode_mask=self.refiner_magnitude_mode_mask,
                    phase_mode_mask=self.refiner_phase_mode_mask,
                )
                total = self.stft_weight * reconstruction["total"]
                terms = dict(reconstruction)
            if magnitude_gate is not None:
                active_gate = magnitude_gate[:, :, self.refiner_magnitude_mode_mask, :]
                terms["refiner_magnitude_gate_mean"] = active_gate.mean()
                terms["refiner_magnitude_gate_std"] = active_gate.std(unbiased=False)
                terms["refiner_magnitude_gate_saturation_fraction"] = (
                    ((active_gate < 0.05) | (active_gate > 0.95)).float().mean()
                )
            if self.refiner_oracle_log_magnitude_weight:
                oracle_estimate = refined
                if self.refiner_oracle_log_magnitude_target == "proposal":
                    if proposal_spectrum is None:
                        raise RuntimeError("proposal oracle target requires a spectrum proposal")
                    oracle_estimate = proposal_spectrum
                oracle_log_magnitude = refiner_oracle_log_magnitude_loss(
                    oracle_estimate,
                    coarse_spectrum,
                    target_stft.spectrum,
                    target_stft.spectrum_mask,
                    magnitude_max_hz=magnitude_max_hz,
                    magnitude_mode_weights=self.refiner_magnitude_mode_weights,
                    max_abs_log_residual=(self.refiner_max_abs_log_magnitude_residual),
                    smooth_l1_beta=self.refiner_oracle_log_magnitude_beta,
                    sample_equal=self.refiner_oracle_log_magnitude_sample_equal,
                )
                terms["refiner_oracle_log_magnitude"] = oracle_log_magnitude
                total = total + (
                    self.refiner_oracle_log_magnitude_weight * oracle_log_magnitude
                )
            if self.waveform_l1_weight:
                if not bool(waveform_adversarial_enabled.all()):
                    raise ValueError("waveform L1 requires explicitly enabled waveform supervision")
                waveform_l1 = waveform_l1_loss(fake_audio, audio, lengths)
                terms["waveform_l1"] = waveform_l1
                total = total + self.waveform_l1_weight * waveform_l1
            if self.mr_ccpc_weight:
                mr_ccpc_loss = 1.0 - multiresolution_ccpc(
                    fake_audio,
                    audio,
                    lengths,
                )
                terms["mr_ccpc_loss"] = mr_ccpc_loss
                total = total + self.mr_ccpc_weight * mr_ccpc_loss
            if self.mr_ccpc_no_regression_weight:
                if coarse_audio is None:
                    raise RuntimeError("CCPC no-regression guard requires the coarse waveform")
                mr_ccpc_guard = multiresolution_ccpc_no_regression_loss(
                    fake_audio,
                    coarse_audio,
                    audio,
                    lengths,
                    stereo_max_hz,
                    margin=self.mr_ccpc_no_regression_margin,
                    allowed_regression=(self.mr_ccpc_no_regression_allowed_regression),
                )
                terms["mr_ccpc_no_regression"] = mr_ccpc_guard
                total = total + (self.mr_ccpc_no_regression_weight * mr_ccpc_guard)
            if self.waveform_peak_envelope_weight:
                peak_envelope = waveform_peak_envelope_overshoot_loss(
                    fake_audio,
                    audio,
                    lengths,
                    window_size=self.waveform_peak_envelope_window,
                    hop_size=self.waveform_peak_envelope_hop,
                    rms_floor=self.waveform_peak_envelope_rms_floor,
                )
                terms["waveform_peak_envelope_overshoot"] = peak_envelope
                total = total + self.waveform_peak_envelope_weight * peak_envelope
            if self.refiner_residual_energy_weight:
                if coarse_audio is None:
                    raise RuntimeError("Refiner residual-energy metric requires the coarse waveform")
                residual_energy = waveform_residual_energy_overshoot_loss(
                    fake_audio,
                    coarse_audio,
                    audio,
                    lengths,
                    rms_floor=self.refiner_residual_energy_rms_floor,
                )
                terms["refiner_residual_energy_overshoot"] = residual_energy
                total = total + (self.refiner_residual_energy_weight * residual_energy)
            if self.si_sdr_no_regression_weight:
                if coarse_audio is None:
                    raise RuntimeError("SI-SDR guard requires the coarse waveform")
                si_sdr_guard = waveform_si_sdr_no_regression_loss(
                    fake_audio,
                    coarse_audio,
                    audio,
                    lengths,
                    margin_db=self.si_sdr_no_regression_margin_db,
                )
                terms["si_sdr_no_regression"] = si_sdr_guard
                total = total + self.si_sdr_no_regression_weight * si_sdr_guard
            if (
                self.source_reconstruction_loss is None
                and self.mr_stft_loss is not None
                and self.mr_stft_weight
            ):
                mr_stft = self.mr_stft_loss(
                    fake_audio,
                    audio,
                    lengths,
                    magnitude_max_hz,
                )
                terms["mr_stft"] = mr_stft
                total = total + self.stft_weight * self.mr_stft_weight * mr_stft
            if self.mel_spectral_loss is not None and self.mel_weight:
                mel = self.mel_spectral_loss(
                    fake_audio,
                    audio,
                    lengths,
                    magnitude_max_hz,
                )
                terms["mel"] = mel
                total = total + self.mel_weight * mel
            if self.stft_consistency_weight:
                projected = self.stft.analyze(fake_audio, lengths)
                consistency = spectral_complex_l1_loss(
                    refined,
                    projected.spectrum,
                    target_stft.spectrum_mask,
                )
                terms["stft_consistency"] = consistency
                total = total + self.stft_consistency_weight * consistency
            if waveform_active and self.waveform_discriminator is not None:
                if not bool(waveform_adversarial_enabled.all()):
                    raise ValueError("waveform GAN batch contains samples with waveform supervision disabled")
                adversarial = generator_adversarial_terms(
                    self.waveform_discriminator,
                    gan_real_audio,
                    gan_fake_audio,
                    lengths=lengths,
                )
                terms.update(
                    {f"waveform_{key}": value for key, value in adversarial.items()}
                )
                total = (
                    total
                    + self.waveform_weights.adversarial * adversarial["adversarial"]
                    + self.waveform_weights.feature_matching
                    * adversarial["feature_matching"]
                )
            if spectral_active and self.spectral_discriminator is not None:
                adversarial = generator_adversarial_terms(
                    self.spectral_discriminator,
                    gan_real_spectral_input,
                    gan_fake_spectral_input,
                    lengths=spectral_input_lengths,
                )
                terms.update(
                    {f"spectral_{key}": value for key, value in adversarial.items()}
                )
                total = (
                    total
                    + self.spectral_weights.adversarial * adversarial["adversarial"]
                    + self.spectral_weights.feature_matching
                    * adversarial["feature_matching"]
                )
            if not torch.isfinite(total):
                raise FloatingPointError("Refiner generator loss is NaN/Inf")
            return total, terms

        total, terms = synchronized_call("Refiner generator loss", generator_loss)
        (total * backward_scale).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.refiner.parameters(),
            self.max_grad_norm,
            error_if_nonfinite=True,
        )
        synchronized_call(
            "Refiner generator optimizer step",
            self.generator_optimizer.step,
        )
        if self.generator_scheduler is not None:
            self.generator_scheduler.step()
        terms.update(d_terms)
        terms["generator"] = total.detach()
        terms["generator_grad_norm"] = torch.as_tensor(gradient_norm).detach()
        metrics = {
            name: float(value.detach().float().item())
            for name, value in terms.items()
            if value.numel() == 1
        }
        return TrainStepResult(
            loss=total.detach(),
            metrics=metrics,
            numerators={name: value * local_seconds for name, value in metrics.items()},
            denominators={name: local_seconds for name in metrics},
            discriminator_updated=bool(d_terms),
            local_audio_seconds=local_seconds,
            global_audio_seconds=global_seconds,
            backward_scale=backward_scale,
        )


VAETrainer = SpecVAETrainer
