
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist
from torch import Tensor


FLOW_FORMAT_VERSION = "oqm.render.flow.v2"
SUPPORTED_PATHS = ("linear",)
SUPPORTED_SOURCES = ("gaussian",)
SUPPORTED_TIMESTEPS = (
    "uniform",
    "logit_normal",
    "truncated_logit_normal_rescaled",
    "uniform_logit_normal_25_75",
    "uniform_logit_normal_50_50",
)
SUPPORTED_SOURCE_COUPLINGS = ("independent", "minibatch_ot")
SUPPORTED_SOURCE_COUPLING_SCOPES = ("local", "global")
SUPPORTED_TIMESTEP_SHIFTS = ("none", "length_logistic")
SUPPORTED_TARGETS = ("velocity",)
SUPPORTED_WEIGHTINGS = ("uniform",)
SUPPORTED_REDUCTIONS = ("valid_frame_mean", "sample_mean")
SUPPORTED_COMPUTE_DTYPES = ("float32",)
SUPPORTED_TIME_DIRECTIONS = ("noise_to_data", "data_to_noise")
SUPPORTED_SOLVERS = ("euler", "heun")
CONSISTENCY_FLOW_MATCHING_FORMAT_VERSION = (
    "oqm.render.consistency-flow-matching.v1"
)


@dataclass(frozen=True)
class FlowConfig:
    format_version: str = FLOW_FORMAT_VERSION
    path: str = "linear"
    source_distribution: str = "gaussian"
    timestep_distribution: str = "uniform"
    prediction_target: str = "velocity"
    loss_weighting: str = "uniform"
    loss_reduction: str = "valid_frame_mean"
    source_compute_dtype: str = "float32"
    timestep_compute_dtype: str = "float32"
    loss_compute_dtype: str = "float32"
    time_direction: str = "noise_to_data"
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    truncated_logit_normal_left: float = 0.075
    timestep_epsilon: float = 1.0e-5
    source_coupling: str = "independent"
    source_coupling_scope: str = "local"
    source_coupling_sinkhorn_iterations: int = 20
    timestep_shift: str = "none"
    timestep_shift_min_frames: int = 750
    timestep_shift_max_frames: int = 9_000
    timestep_shift_base: float = 0.5
    timestep_shift_max: float = 1.15

    def validate(self) -> None:
        if self.format_version != FLOW_FORMAT_VERSION:
            raise ValueError(
                f"flow format_version must be {FLOW_FORMAT_VERSION},"
                f"received {self.format_version}"
            )
        choices = (
            ("path", self.path, SUPPORTED_PATHS),
            (
                "source_distribution",
                self.source_distribution,
                SUPPORTED_SOURCES,
            ),
            (
                "source_coupling",
                self.source_coupling,
                SUPPORTED_SOURCE_COUPLINGS,
            ),
            (
                "source_coupling_scope",
                self.source_coupling_scope,
                SUPPORTED_SOURCE_COUPLING_SCOPES,
            ),
            (
                "timestep_distribution",
                self.timestep_distribution,
                SUPPORTED_TIMESTEPS,
            ),
            ("prediction_target", self.prediction_target, SUPPORTED_TARGETS),
            ("loss_weighting", self.loss_weighting, SUPPORTED_WEIGHTINGS),
            ("loss_reduction", self.loss_reduction, SUPPORTED_REDUCTIONS),
            (
                "source_compute_dtype",
                self.source_compute_dtype,
                SUPPORTED_COMPUTE_DTYPES,
            ),
            (
                "timestep_compute_dtype",
                self.timestep_compute_dtype,
                SUPPORTED_COMPUTE_DTYPES,
            ),
            (
                "loss_compute_dtype",
                self.loss_compute_dtype,
                SUPPORTED_COMPUTE_DTYPES,
            ),
            ("time_direction", self.time_direction, SUPPORTED_TIME_DIRECTIONS),
            ("timestep_shift", self.timestep_shift, SUPPORTED_TIMESTEP_SHIFTS),
        )
        for name, value, supported in choices:
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if value not in supported:
                raise ValueError(f"unsupported {name}={value!r}; expected one of {supported}")
        for name, value in {
            "logit_normal_mean": self.logit_normal_mean,
            "logit_normal_std": self.logit_normal_std,
            "truncated_logit_normal_left": self.truncated_logit_normal_left,
            "timestep_epsilon": self.timestep_epsilon,
            "timestep_shift_base": self.timestep_shift_base,
            "timestep_shift_max": self.timestep_shift_max,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a numeric value")
        mean = self.logit_normal_mean
        std = self.logit_normal_std
        truncation = self.truncated_logit_normal_left
        epsilon = self.timestep_epsilon
        if not math_is_finite(mean):
            raise ValueError("logit_normal_mean must be finite")
        if not math_is_finite(std) or std <= 0:
            raise ValueError("logit_normal_std must be finite and positive")
        if not math_is_finite(truncation) or not 0.0 < truncation < 1.0:
            raise ValueError("truncated_logit_normal_left must be in (0, 1)")
        if not math_is_finite(epsilon) or not 0.0 < epsilon < 0.5:
            raise ValueError("timestep_epsilon must be finite and in (0, 0.5)")
        integer_values = {
            "source_coupling_sinkhorn_iterations": (
                self.source_coupling_sinkhorn_iterations
            ),
            "timestep_shift_min_frames": self.timestep_shift_min_frames,
            "timestep_shift_max_frames": self.timestep_shift_max_frames,
        }
        for name, value in integer_values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.timestep_shift_max_frames <= self.timestep_shift_min_frames:
            raise ValueError("timestep_shift_max_frames must be greater than timestep_shift_min_frames")
        if (
            not math_is_finite(self.timestep_shift_base)
            or not math_is_finite(self.timestep_shift_max)
            or self.timestep_shift_base < 0.0
            or self.timestep_shift_max < self.timestep_shift_base
        ):
            raise ValueError("timestep shift requires 0 <= base <= max, with finite values")
        if self.timestep_shift != "none" and self.time_direction != "data_to_noise":
            raise ValueError("length timestep shift is defined only for the data_to_noise time direction")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FlowConfig":
        normalized = dict(value)

        if (
            "source_distribution" not in normalized
            and "noise_distribution" in normalized
        ):
            normalized["source_distribution"] = normalized.pop("noise_distribution")
        elif "noise_distribution" in normalized:
            if normalized["source_distribution"] != normalized["noise_distribution"]:
                raise ValueError(
                    "flow config aliases source_distribution and noise_distribution conflict"
                )
            normalized.pop("noise_distribution")
        unknown = set(normalized) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"flow config contains unknown fields: {sorted(unknown)}")
        config = cls(
            **{
                key: normalized[key]
                for key in cls.__dataclass_fields__
                if key in normalized
            }
        )
        config.validate()
        return config


@dataclass(frozen=True)
class ConsistencyFlowMatchingConfig:

    format_version: str = CONSISTENCY_FLOW_MATCHING_FORMAT_VERSION
    enabled: bool = False
    delta: float = 1.0e-3
    num_segments: int = 2
    boundary: float = 0.0
    boundary_zero_steps: int = 0
    velocity_weight: float = 1.0e-5

    def validate(self) -> None:
        if self.format_version != CONSISTENCY_FLOW_MATCHING_FORMAT_VERSION:
            raise ValueError(
                "consistency flow matching format_version must be "
                f"{CONSISTENCY_FLOW_MATCHING_FORMAT_VERSION}"
            )
        if not isinstance(self.enabled, bool):
            raise TypeError("consistency flow matching enabled must be a bool")
        if (
            not isinstance(self.num_segments, int)
            or isinstance(self.num_segments, bool)
            or self.num_segments <= 0
        ):
            raise ValueError("consistency flow matching num_segments must be a positive integer")
        if (
            not isinstance(self.boundary_zero_steps, int)
            or isinstance(self.boundary_zero_steps, bool)
            or self.boundary_zero_steps < 0
        ):
            raise ValueError(
                "consistency flow matching boundary_zero_steps must be a non-negative integer"
            )
        for name, value in {
            "delta": self.delta,
            "boundary": self.boundary,
            "velocity_weight": self.velocity_weight,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TypeError(f"consistency flow matching {name} must be finite")
        if not 0.0 < float(self.delta) < 1.0:
            raise ValueError("consistency flow matching delta must be in (0, 1)")
        if not 0.0 <= float(self.boundary) <= 1.0:
            raise ValueError("consistency flow matching boundary must be in [0, 1]")
        if float(self.velocity_weight) < 0.0:
            raise ValueError("consistency flow matching velocity_weight must be non-negative")

    def for_step(self, global_step: int) -> "ConsistencyFlowMatchingConfig":

        self.validate()
        if not isinstance(global_step, int) or isinstance(global_step, bool):
            raise TypeError("consistency flow matching global_step must be an integer")
        if global_step < 0:
            raise ValueError("consistency flow matching global_step must be non-negative")
        if global_step < self.boundary_zero_steps:
            return replace(self, boundary=0.0)
        return self

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "ConsistencyFlowMatchingConfig":
        if value is None:
            config = cls()
        else:
            if not isinstance(value, Mapping):
                raise TypeError("train.consistency_flow_matching must be a mapping")
            unknown = set(value) - set(cls.__dataclass_fields__)
            if unknown:
                raise ValueError(
                    "train.consistency_flow_matching contains unknown fields: "
                    f"{sorted(unknown)}"
                )
            config = cls(
                **{
                    key: value[key]
                    for key in cls.__dataclass_fields__
                    if key in value
                }
            )
        config.validate()
        return config


def _generator_device(generator: torch.Generator | None) -> torch.device:
    if generator is None:
        return torch.device("cpu")
    try:
        return torch.device(generator.device)
    except AttributeError:
        return torch.device("cpu")


def _rand(
    shape: tuple[int, ...],
    *,
    reference: torch.Tensor,
    generator: torch.Generator | None,
    normal: bool,
) -> torch.Tensor:

    target_device = reference.device
    function = torch.randn if normal else torch.rand
    if generator is None:
        return function(
            shape,
            device=target_device,
            dtype=reference.dtype,
        )
    generator_device = _generator_device(generator)
    if generator_device != target_device:
        value = function(
            shape,
            device=generator_device,
            dtype=torch.float32,
            generator=generator,
        )
        return value.to(device=target_device, dtype=reference.dtype)
    return function(
        shape,
        device=target_device,
        dtype=reference.dtype,
        generator=generator,
    )


def sample_source_like(
    target: torch.Tensor,
    config: FlowConfig,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    config.validate()
    if not target.is_floating_point():
        raise TypeError("flow target must be floating point tensor")
    if config.source_distribution != "gaussian":
        raise AssertionError(
            "FlowConfig.validate should have rejected the unknown source distribution"
        )


    source_reference = torch.empty(
        (),
        device=target.device,
        dtype=torch.float32,
    )
    source = _rand(
        tuple(target.shape),
        reference=source_reference,
        generator=generator,
        normal=True,
    )
    return source.to(dtype=target.dtype)


def _effective_lengths_tensor(
    effective_lengths: torch.Tensor | list[int] | tuple[int, ...],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    lengths = torch.as_tensor(effective_lengths, device=device)
    if lengths.shape != (batch_size,):
        raise ValueError(
            "effective_lengths must have shape [B]; "
            f"received {tuple(lengths.shape)} for batch size {batch_size}"
        )
    if lengths.dtype == torch.bool or lengths.is_floating_point():
        raise TypeError("effective_lengths must be an integer tensor")
    lengths = lengths.to(dtype=torch.int64)
    if bool((lengths <= 0).any()):
        raise ValueError("effective_lengths must all be positive")
    return lengths


def shift_timesteps_by_length(
    timestep: torch.Tensor,
    effective_lengths: torch.Tensor | list[int] | tuple[int, ...],
    config: FlowConfig,
) -> torch.Tensor:

    config.validate()
    if config.timestep_shift == "none":
        return timestep
    if not isinstance(timestep, torch.Tensor) or not timestep.is_floating_point():
        raise TypeError("pending timestep shift values must be a floating-point tensor")
    if timestep.ndim not in {1, 2}:
        raise ValueError("pending timestep shift values must have shape [B] or [B, S]")
    batch_size = int(timestep.shape[0])
    lengths = _effective_lengths_tensor(
        effective_lengths,
        batch_size=batch_size,
        device=timestep.device,
    ).to(dtype=torch.float32)
    if not torch.isfinite(timestep).all() or bool(
        ((timestep < 0.0) | (timestep > 1.0)).any()
    ):
        raise ValueError("pending shifted timesteps must be finite and in [0, 1]")
    minimum = float(config.timestep_shift_min_frames)
    maximum = float(config.timestep_shift_max_frames)
    fraction = (lengths.clamp(minimum, maximum) - minimum) / (maximum - minimum)
    shift = float(config.timestep_shift_base) + fraction * (
        float(config.timestep_shift_max) - float(config.timestep_shift_base)
    )
    alpha = torch.exp(shift)
    if timestep.ndim == 2:
        alpha = alpha.unsqueeze(1)
    value = timestep.float()
    shifted = alpha * value / (1.0 + (alpha - 1.0) * value)

    shifted = torch.where(value == 0.0, torch.zeros_like(shifted), shifted)
    shifted = torch.where(value == 1.0, torch.ones_like(shifted), shifted)
    if not torch.isfinite(shifted).all():
        raise FloatingPointError("length timestep shift produced NaN or infinity")
    return shifted


def _truncated_logit_normal_rescaled(
    normal: torch.Tensor,
    *,
    left: float,
    mean: float,
    std: float,
) -> torch.Tensor:

    if normal.dtype != torch.float32:
        normal = normal.float()
    sqrt_two = math.sqrt(2.0)
    cdf = 0.5 * (1.0 + torch.erf(normal / sqrt_two))
    left_logit = math.log(float(left) / (1.0 - float(left)))
    standardized_left = (left_logit - float(mean)) / float(std)
    lower = 0.5 * (1.0 + math.erf(standardized_left / sqrt_two))
    truncated_cdf = lower + (1.0 - lower) * cdf

    finfo = torch.finfo(torch.float32)
    truncated_cdf = truncated_cdf.clamp(min=finfo.eps, max=1.0 - finfo.eps)
    truncated_normal = sqrt_two * torch.erfinv(2.0 * truncated_cdf - 1.0)
    truncated = torch.sigmoid(truncated_normal * float(std) + float(mean))
    return (truncated - float(left)) / (1.0 - float(left))


def sample_timesteps(
    batch_size: int,
    config: FlowConfig,
    *,
    reference: torch.Tensor,
    generator: torch.Generator | None = None,
    effective_lengths: torch.Tensor | list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    config.validate()
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not isinstance(reference, torch.Tensor) or not reference.is_floating_point():
        raise TypeError("timestep reference must be a floating-point tensor")
    if config.timestep_compute_dtype != "float32":
        raise AssertionError("FlowConfig.validate should have rejected a non-FP32 timestep")


    timestep_reference = torch.empty(
        (),
        device=reference.device,
        dtype=torch.float32,
    )
    if config.timestep_distribution == "uniform":
        timestep = _rand(
            (batch_size,),
            reference=timestep_reference,
            generator=generator,
            normal=False,
        )
    else:
        normal = _rand(
            (batch_size,),
            reference=timestep_reference,
            generator=generator,
            normal=True,
        )
        logit_normal = torch.sigmoid(
            normal * config.logit_normal_std + config.logit_normal_mean
        )
        if config.timestep_distribution == "logit_normal":
            timestep = logit_normal
        elif config.timestep_distribution == "truncated_logit_normal_rescaled":
            timestep = _truncated_logit_normal_rescaled(
                normal,
                left=config.truncated_logit_normal_left,
                mean=config.logit_normal_mean,
                std=config.logit_normal_std,
            )
        else:


            uniform_probability = (
                0.25
                if config.timestep_distribution == "uniform_logit_normal_25_75"
                else 0.5
            )
            choose_uniform = (
                _rand(
                    (batch_size,),
                    reference=timestep_reference,
                    generator=generator,
                    normal=False,
                )
                < uniform_probability
            )
            uniform = _rand(
                (batch_size,),
                reference=timestep_reference,
                generator=generator,
                normal=False,
            )
            timestep = torch.where(choose_uniform, uniform, logit_normal)
    if config.timestep_shift != "none":
        if effective_lengths is None:
            raise ValueError("length timestep shift requires explicit effective_lengths")
        timestep = shift_timesteps_by_length(timestep, effective_lengths, config)
    return timestep.clamp(
        min=config.timestep_epsilon, max=1.0 - config.timestep_epsilon
    )


def _broadcast_timestep(timestep: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if timestep.ndim == 0:
        timestep = timestep.expand(target.shape[0])
    if timestep.shape != (target.shape[0],):
        raise ValueError(f"timestep must be scalar or have shape [B]; received {tuple(timestep.shape)}")
    if not timestep.is_floating_point():
        raise TypeError("timestep must be a floating-point tensor")
    if not torch.isfinite(timestep).all():
        raise ValueError("timestep contains NaN/Inf")
    if bool(((timestep < 0.0) | (timestep > 1.0)).any()):
        raise ValueError("timestep must be in [0, 1]")
    return timestep.reshape(target.shape[0], *([1] * (target.ndim - 1))).to(
        device=target.device, dtype=target.dtype
    )


def linear_flow_path(
    source: torch.Tensor,
    target: torch.Tensor,
    timestep: torch.Tensor,
    *,
    time_direction: str = "noise_to_data",
) -> tuple[torch.Tensor, torch.Tensor]:

    if source.shape != target.shape:
        raise ValueError("flow source and target must have the same shape")
    if (
        not source.is_floating_point()
        or not target.is_floating_point()
        or source.dtype != target.dtype
        or source.device != target.device
    ):
        raise TypeError("flow source and target must be floating-point tensors with the same dtype and device")
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError("flow source/target contains NaN/Inf")
    if time_direction not in SUPPORTED_TIME_DIRECTIONS:
        raise ValueError(
            f"unsupported time_direction={time_direction!r}; "
            f"expected one of {SUPPORTED_TIME_DIRECTIONS}"
        )
    time = _broadcast_timestep(timestep, target)
    if time_direction == "noise_to_data":
        noisy = (1.0 - time) * source + time * target
        velocity = target - source
    else:
        noisy = (1.0 - time) * target + time * source
        velocity = source - target
    if not torch.isfinite(noisy).all() or not torch.isfinite(velocity).all():
        raise FloatingPointError(
            "linear flow path produced NaN or infinity from finite inputs; check latent magnitude and dtype"
        )
    return noisy, velocity


@dataclass
class SourceCouplingResult:
    source: torch.Tensor
    permutation: torch.Tensor
    cost_before: torch.Tensor
    cost_after: torch.Tensor
    batch_size: int
    scope: str


def _validate_ot_inputs(
    target: torch.Tensor,
    source: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    if target.ndim != 3 or source.shape != target.shape:
        raise ValueError("minibatch OT requires target and source tensors with the same shape [B, T, D]")
    if (
        not target.is_floating_point()
        or not source.is_floating_point()
        or source.dtype != target.dtype
        or source.device != target.device
    ):
        raise TypeError(
            "minibatch OT requires floating-point target and source tensors "
            "with the same dtype and device"
        )
    if frame_mask.dtype != torch.bool or frame_mask.shape != target.shape[:2]:
        raise TypeError("minibatch OT requires a bool frame_mask with shape [B, T]")
    if frame_mask.device != target.device:
        raise ValueError("minibatch OT frame_mask and latents must be on the same device")
    lengths = frame_mask.sum(dim=1, dtype=torch.int64)
    if bool((lengths <= 0).any()):
        raise ValueError("minibatch OT requires at least one valid frame per sample")
    expected_mask = torch.arange(target.shape[1], device=target.device).unsqueeze(
        0
    ) < lengths.unsqueeze(1)
    if not torch.equal(frame_mask, expected_mask):
        raise ValueError("minibatch OT accepts only right-padded frame masks")
    checked_target = torch.where(
        frame_mask.unsqueeze(-1), target, torch.zeros_like(target)
    )
    if not torch.isfinite(checked_target).all() or not torch.isfinite(source).all():
        raise FloatingPointError("valid minibatch OT target or source values contain NaN or infinity")
    return lengths


@torch.no_grad()
def couple_minibatch_source(
    target: torch.Tensor,
    source: torch.Tensor,
    frame_mask: torch.Tensor,
    *,
    sinkhorn_iterations: int = 20,
    scope: str = "local",
) -> SourceCouplingResult:

    lengths = _validate_ot_inputs(target, source, frame_mask)
    if (
        not isinstance(sinkhorn_iterations, int)
        or isinstance(sinkhorn_iterations, bool)
        or sinkhorn_iterations <= 0
    ):
        raise ValueError("sinkhorn_iterations must be a positive integer")
    if scope not in SUPPORTED_SOURCE_COUPLING_SCOPES:
        raise ValueError(f"OT scope must be one of {SUPPORTED_SOURCE_COUPLING_SCOPES}")
    batch = int(target.shape[0])
    if batch < 2:
        raise RuntimeError("minibatch OT rejects an effective batch<2")
    target_fp32 = target.float()
    source_fp32 = source.float()
    cost = torch.empty(batch, batch, device=target.device, dtype=torch.float32)

    for length_tensor in torch.unique(lengths, sorted=True):
        length = int(length_tensor.item())
        rows = torch.nonzero(lengths == length_tensor, as_tuple=False).flatten()
        target_flat = target_fp32[rows, :length].reshape(rows.numel(), -1)
        source_flat = source_fp32[:, :length].reshape(batch, -1)
        target_norm = target_flat.square().sum(dim=1, keepdim=True)
        source_norm = source_flat.square().sum(dim=1).unsqueeze(0)
        row_cost = target_norm + source_norm - 2.0 * (target_flat @ source_flat.T)
        cost[rows] = row_cost.clamp_min_(0.0) / float(target_flat.shape[1])
    if not torch.isfinite(cost).all():
        raise FloatingPointError("minibatch OT cost contains NaN or infinity")
    scale = cost.detach().mean().clamp_min(torch.finfo(torch.float32).tiny)
    log_probability = -cost / scale
    for _ in range(sinkhorn_iterations):
        log_probability = log_probability - torch.logsumexp(
            log_probability, dim=1, keepdim=True
        )
        log_probability = log_probability - torch.logsumexp(
            log_probability, dim=0, keepdim=True
        )
    probability = log_probability.exp()
    permutation = torch.empty(batch, dtype=torch.long, device=target.device)
    used = torch.zeros(batch, dtype=torch.bool, device=target.device)

    for row in range(batch):
        scores = probability[row].masked_fill(used, -1.0)
        selected = scores.argmax()
        permutation[row] = selected
        used[selected] = True
    row_indices = torch.arange(batch, device=target.device)
    cost_before = cost.diagonal().mean()
    cost_after = cost[row_indices, permutation].mean()
    return SourceCouplingResult(
        source=source[permutation],
        permutation=permutation,
        cost_before=cost_before,
        cost_after=cost_after,
        batch_size=batch,
        scope=scope,
    )


def _distributed_shapes(target: torch.Tensor) -> list[tuple[int, int, int]]:
    local = torch.tensor(target.shape, dtype=torch.int64, device=target.device)
    if not dist.is_initialized():
        return [tuple(int(value) for value in local.tolist())]
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return [tuple(int(value) for value in shape.tolist()) for shape in gathered]


@torch.no_grad()
def _globally_coupled_source(
    target: torch.Tensor,
    frame_mask: torch.Tensor,
    config: FlowConfig,
    *,
    source_generator: torch.Generator | None,
) -> SourceCouplingResult:

    shapes = _distributed_shapes(target)
    if any(len(shape) != 3 or shape[2] != target.shape[2] for shape in shapes):
        raise RuntimeError(f"global minibatch OT received incompatible latent shapes across ranks: {shapes}")
    maximum_batch = max(shape[0] for shape in shapes)
    maximum_frames = max(shape[1] for shape in shapes)
    local_batch, local_frames, channels = target.shape
    source_reference = torch.empty(
        local_batch,
        maximum_frames,
        channels,
        device=target.device,
        dtype=target.dtype,
    )
    full_source = sample_source_like(
        source_reference,
        config,
        generator=source_generator,
    )
    padded_target = torch.zeros_like(source_reference).new_zeros(
        maximum_batch, maximum_frames, channels
    )
    padded_source = torch.zeros_like(padded_target)
    padded_mask = torch.zeros(
        maximum_batch,
        maximum_frames,
        dtype=torch.uint8,
        device=target.device,
    )
    padded_target[:local_batch, :local_frames] = target
    padded_source[:local_batch] = full_source
    padded_mask[:local_batch, :local_frames] = frame_mask.to(torch.uint8)
    if dist.is_initialized():
        world_size = dist.get_world_size()
        target_rows = [torch.empty_like(padded_target) for _ in range(world_size)]
        source_rows = [torch.empty_like(padded_source) for _ in range(world_size)]
        mask_rows = [torch.empty_like(padded_mask) for _ in range(world_size)]
        dist.all_gather(target_rows, padded_target.contiguous())
        dist.all_gather(source_rows, padded_source.contiguous())
        dist.all_gather(mask_rows, padded_mask.contiguous())
        rank = dist.get_rank()
    else:
        target_rows = [padded_target]
        source_rows = [padded_source]
        mask_rows = [padded_mask]
        rank = 0
    global_targets = torch.cat(
        [value[: shape[0]] for value, shape in zip(target_rows, shapes, strict=True)],
        dim=0,
    )
    global_sources = torch.cat(
        [value[: shape[0]] for value, shape in zip(source_rows, shapes, strict=True)],
        dim=0,
    )
    global_masks = torch.cat(
        [
            value[: shape[0]].to(torch.bool)
            for value, shape in zip(mask_rows, shapes, strict=True)
        ],
        dim=0,
    )
    coupled = couple_minibatch_source(
        global_targets,
        global_sources,
        global_masks,
        sinkhorn_iterations=config.source_coupling_sinkhorn_iterations,
        scope="global",
    )
    local_offset = sum(shape[0] for shape in shapes[:rank])
    local_source = coupled.source[
        local_offset : local_offset + local_batch, :local_frames
    ].contiguous()
    return SourceCouplingResult(
        source=local_source,
        permutation=coupled.permutation,
        cost_before=coupled.cost_before,
        cost_after=coupled.cost_after,
        batch_size=coupled.batch_size,
        scope=coupled.scope,
    )


@dataclass
class FlowTrainingSample:
    source: torch.Tensor
    timestep: torch.Tensor
    noisy_latents: torch.Tensor
    target: torch.Tensor
    target_velocity: torch.Tensor
    source_coupling_cost_before: torch.Tensor | None = None
    source_coupling_cost_after: torch.Tensor | None = None
    source_coupling_batch_size: int = 1
    source_coupling_scope: str = "independent"


@dataclass(frozen=True)
class ConsistencyFlowMatchingPair:

    neighbor_timestep: torch.Tensor
    neighbor_latents: torch.Tensor
    segment_endpoint_timestep: torch.Tensor
    segment_endpoint_latents: torch.Tensor
    use_predicted_neighbor_endpoint: torch.Tensor
    velocity_consistency_active: torch.Tensor
    requires_neighbor_prediction: bool


def make_consistency_flow_matching_pair(
    sample: FlowTrainingSample,
    flow_config: FlowConfig,
    consistency_config: ConsistencyFlowMatchingConfig,
) -> ConsistencyFlowMatchingPair:

    flow_config.validate()
    consistency_config.validate()
    if not consistency_config.enabled:
        raise ValueError("consistency flow matching pairing requires enabled=true")
    if flow_config.path != "linear" or flow_config.prediction_target != "velocity":
        raise ValueError("consistency flow matching currently supports only linear velocity flow")
    timestep = sample.timestep.float()
    if timestep.shape != (sample.target.shape[0],):
        raise ValueError("consistency flow matching timestep must have shape [B]")
    progress = (
        timestep
        if flow_config.time_direction == "noise_to_data"
        else 1.0 - timestep
    )
    neighbor_progress = (progress + float(consistency_config.delta)).clamp_max(1.0)
    boundaries = torch.linspace(
        0.0,
        1.0,
        int(consistency_config.num_segments) + 1,
        device=timestep.device,
        dtype=torch.float32,
    )
    segment_index = torch.searchsorted(
        boundaries,
        progress.contiguous(),
        right=False,
    ).clamp(min=1, max=int(consistency_config.num_segments))
    endpoint_progress = boundaries[segment_index]
    if flow_config.time_direction == "noise_to_data":
        neighbor_timestep = neighbor_progress
        endpoint_timestep = endpoint_progress
    else:
        neighbor_timestep = 1.0 - neighbor_progress
        endpoint_timestep = 1.0 - endpoint_progress
    neighbor_latents, _ = linear_flow_path(
        sample.source,
        sample.target,
        neighbor_timestep,
        time_direction=flow_config.time_direction,
    )
    endpoint_latents, _ = linear_flow_path(
        sample.source,
        sample.target,
        endpoint_timestep,
        time_direction=flow_config.time_direction,
    )
    use_predicted = neighbor_progress < float(consistency_config.boundary)
    velocity_active = (progress < float(consistency_config.boundary)) & (
        endpoint_progress - progress
        > 1.01 * float(consistency_config.delta)
    )


    requires_neighbor = float(consistency_config.boundary) > 0.0
    return ConsistencyFlowMatchingPair(
        neighbor_timestep=neighbor_timestep,
        neighbor_latents=neighbor_latents,
        segment_endpoint_timestep=endpoint_timestep,
        segment_endpoint_latents=endpoint_latents,
        use_predicted_neighbor_endpoint=use_predicted,
        velocity_consistency_active=velocity_active,
        requires_neighbor_prediction=requires_neighbor,
    )


def make_flow_training_sample(
    target: torch.Tensor,
    config: FlowConfig,
    *,
    source_generator: torch.Generator | None = None,
    timestep_generator: torch.Generator | None = None,
    source: torch.Tensor | None = None,
    timestep: torch.Tensor | None = None,
    frame_mask: torch.Tensor | None = None,
    apply_source_coupling: bool = True,
) -> FlowTrainingSample:
    config.validate()
    if target.ndim < 2 or not target.is_floating_point():
        raise TypeError("target latents must be a floating-point tensor with at least two dimensions")
    if not torch.isfinite(target).all():
        raise ValueError("target latent contains NaN/Inf")
    if frame_mask is None:
        frame_mask = torch.ones(
            target.shape[:2], dtype=torch.bool, device=target.device
        )
    if frame_mask.dtype != torch.bool or frame_mask.shape != target.shape[:2]:
        raise TypeError("flow frame_mask must be a bool tensor with shape [B, T]")
    if (
        source is not None
        and apply_source_coupling
        and config.source_coupling == "minibatch_ot"
        and config.source_coupling_scope == "global"
    ):
        raise ValueError(
            "global minibatch OT does not accept an explicit local source; "
            "training samples a shared source across ranks, and fixed validation "
            "must set apply_source_coupling=False"
        )
    coupling: SourceCouplingResult | None = None
    if (
        source is None
        and apply_source_coupling
        and config.source_coupling == "minibatch_ot"
        and config.source_coupling_scope == "global"
    ):
        coupling = _globally_coupled_source(
            target,
            frame_mask,
            config,
            source_generator=source_generator,
        )
        source = coupling.source
    else:
        source = (
            sample_source_like(target, config, generator=source_generator)
            if source is None
            else source
        )
    if not isinstance(source, torch.Tensor):
        raise TypeError("explicit flow source must be a tensor")
    if (
        apply_source_coupling
        and config.source_coupling == "minibatch_ot"
        and coupling is None
    ):
        coupling = couple_minibatch_source(
            target,
            source,
            frame_mask,
            sinkhorn_iterations=config.source_coupling_sinkhorn_iterations,
            scope=config.source_coupling_scope,
        )
        source = coupling.source
    if timestep is not None and not isinstance(timestep, torch.Tensor):
        raise TypeError("explicit timestep must be a tensor")
    timestep = (
        sample_timesteps(
            target.shape[0],
            config,
            reference=target,
            generator=timestep_generator,
            effective_lengths=frame_mask.sum(dim=1, dtype=torch.int64),
        )
        if timestep is None
        else timestep.to(device=target.device)
    )
    if config.path != "linear" or config.prediction_target != "velocity":
        raise AssertionError(
            "FlowConfig.validate should have rejected the unknown path or target"
        )
    noisy, velocity = linear_flow_path(
        source,
        target,
        timestep,
        time_direction=config.time_direction,
    )
    return FlowTrainingSample(
        source=source,
        timestep=timestep,
        noisy_latents=noisy,
        target=target,
        target_velocity=velocity,
        source_coupling_cost_before=(
            coupling.cost_before if coupling is not None else None
        ),
        source_coupling_cost_after=(
            coupling.cost_after if coupling is not None else None
        ),
        source_coupling_batch_size=(
            coupling.batch_size if coupling is not None else int(target.shape[0])
        ),
        source_coupling_scope=(
            coupling.scope if coupling is not None else "independent"
        ),
    )


@dataclass
class MaskedLoss:
    loss: torch.Tensor
    numerator: torch.Tensor
    denominator: torch.Tensor
    valid_frames: torch.Tensor

    def __iter__(self):

        yield self.loss
        yield self.numerator
        yield self.denominator


def masked_flow_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    frame_mask: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    loss_weighting: str = "uniform",
    loss_reduction: str = "valid_frame_mean",
    compute_dtype: str = "float32",
) -> MaskedLoss:

    if loss_weighting not in SUPPORTED_WEIGHTINGS:
        raise ValueError(
            f"unsupported loss_weighting={loss_weighting!r}; "
            f"expected one of {SUPPORTED_WEIGHTINGS}"
        )
    if loss_reduction not in SUPPORTED_REDUCTIONS:
        raise ValueError(
            f"unsupported loss_reduction={loss_reduction!r}; "
            f"expected one of {SUPPORTED_REDUCTIONS}"
        )
    if compute_dtype not in SUPPORTED_COMPUTE_DTYPES:
        raise ValueError(
            f"unsupported loss_compute_dtype={compute_dtype!r}; "
            f"expected one of {SUPPORTED_COMPUTE_DTYPES}"
        )
    if compute_dtype != "float32":
        raise AssertionError("flow loss currently supports only FP32 computation")
    if prediction.shape != target.shape or prediction.ndim < 3:
        raise ValueError("prediction and target must have the same shape with at least [B, T, D]")
    if (
        not prediction.is_floating_point()
        or not target.is_floating_point()
        or prediction.device != target.device
    ):
        raise TypeError("prediction and target must be floating-point tensors on the same device")
    if frame_mask.dtype != torch.bool or frame_mask.shape != prediction.shape[:2]:
        raise TypeError("frame_mask must be a bool tensor with shape [B, T]")
    if frame_mask.device != prediction.device:
        raise ValueError("frame_mask, prediction, and target must be on the same device")
    weight: torch.Tensor
    sample_reduction_weight: torch.Tensor | None = None
    if sample_weight is None:
        weight = torch.ones(
            prediction.shape[:2],
            device=prediction.device,
            dtype=torch.float32,
        )
    else:
        if not isinstance(sample_weight, torch.Tensor):
            raise TypeError("sample_weight must be a tensor")
        if sample_weight.dtype == torch.bool or not (
            sample_weight.is_floating_point()
            or sample_weight.dtype
            in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
        ):
            raise TypeError("sample_weight must be a numeric tensor")
        if sample_weight.ndim == 1 and sample_weight.shape[0] == prediction.shape[0]:
            sample_reduction_weight = sample_weight.to(
                device=prediction.device,
                dtype=torch.float32,
            )
            weight = sample_reduction_weight[:, None].expand(prediction.shape[:2])
        elif sample_weight.shape == prediction.shape[:2]:
            weight = sample_weight
        else:
            raise ValueError("sample_weight must have shape [B] or [B, T]")
        weight = weight.to(device=prediction.device, dtype=torch.float32)
        if not torch.isfinite(weight).all() or bool((weight < 0).any()):
            raise ValueError("sample_weight must be finite and non-negative")


    active = frame_mask & (weight > 0)
    expanded_active = active
    for _ in range(prediction.ndim - 2):
        expanded_active = expanded_active.unsqueeze(-1)
    delta = torch.where(
        expanded_active,
        prediction.float() - target.float(),
        torch.zeros((), device=prediction.device, dtype=torch.float32),
    )
    if not torch.isfinite(delta).all():
        raise FloatingPointError("flow loss contains NaN/Inf")
    squared = delta.square()
    if not torch.isfinite(squared).all():
        raise FloatingPointError("flow loss overflowed to NaN or infinity")


    frame_error = squared.mean(dim=tuple(range(2, squared.ndim)))
    weighted_mask = frame_mask.to(torch.float32) * weight
    if loss_reduction == "valid_frame_mean":

        numerator = (frame_error * weighted_mask).sum()
        denominator = weighted_mask.sum()
    else:


        per_sample_denominator = weighted_mask.sum(dim=1)
        active_samples = per_sample_denominator > 0
        if not bool(active_samples.any()):
            raise ValueError("flow loss requires at least one valid sample with positive weight")
        per_sample_numerator = (frame_error * weighted_mask).sum(dim=1)
        per_sample_loss = torch.where(
            active_samples,
            per_sample_numerator
            / per_sample_denominator.clamp_min(torch.finfo(torch.float32).tiny),
            torch.zeros_like(per_sample_numerator),
        )


        reduction_weight = (
            sample_reduction_weight
            if sample_reduction_weight is not None
            else active_samples.to(torch.float32)
        )
        reduction_weight = torch.where(
            active_samples,
            reduction_weight,
            torch.zeros_like(reduction_weight),
        )
        numerator = (per_sample_loss * reduction_weight).sum()
        denominator = reduction_weight.sum()
    if not torch.isfinite(numerator) or not torch.isfinite(denominator):
        raise FloatingPointError("flow loss numerator or denominator contains NaN or infinity")
    if not bool(denominator > 0):
        raise ValueError("flow loss requires at least one valid frame with positive weight")
    loss = numerator / denominator
    if not torch.isfinite(loss):
        raise FloatingPointError("flow loss contains NaN or infinity")
    return MaskedLoss(
        loss=loss,
        numerator=numerator,
        denominator=denominator,
        valid_frames=frame_mask.sum(),
    )


masked_flow_matching_loss = masked_flow_loss


@dataclass(frozen=True)
class ConsistencyFlowMatchingLoss:

    combined: MaskedLoss
    endpoint: MaskedLoss
    velocity: MaskedLoss


def _broadcast_batch_coefficient(
    value: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:

    if value.shape != (reference.shape[0],) or not value.is_floating_point():
        raise ValueError("CFM trajectory coefficients must be a floating-point tensor with shape [B]")
    if value.device != reference.device or not torch.isfinite(value).all():
        raise ValueError("CFM trajectory coefficients must be finite and on the same device as the state")
    return value.reshape(
        reference.shape[0], *([1] * (reference.ndim - 1))
    ).float()


def consistency_flow_matching_loss(
    primary_prediction: torch.Tensor,
    neighbor_prediction: torch.Tensor | None,
    sample: FlowTrainingSample,
    pair: ConsistencyFlowMatchingPair,
    frame_mask: torch.Tensor,
    *,
    flow_config: FlowConfig,
    consistency_config: ConsistencyFlowMatchingConfig,
) -> ConsistencyFlowMatchingLoss:

    flow_config.validate()
    consistency_config.validate()
    if not consistency_config.enabled:
        raise ValueError("consistency flow matching loss requires enabled=true")
    if flow_config.path != "linear" or flow_config.prediction_target != "velocity":
        raise ValueError("consistency flow matching currently supports only linear velocity flow")
    if primary_prediction.shape != sample.noisy_latents.shape:
        raise ValueError("CFM primary prediction and flow state must have the same shape")
    if frame_mask.dtype != torch.bool or frame_mask.shape != sample.target.shape[:2]:
        raise TypeError("CFM frame_mask must be a bool tensor with shape [B, T]")
    batch = int(sample.target.shape[0])
    vector_fields = {
        "neighbor_timestep": pair.neighbor_timestep,
        "segment_endpoint_timestep": pair.segment_endpoint_timestep,
        "use_predicted_neighbor_endpoint": pair.use_predicted_neighbor_endpoint,
        "velocity_consistency_active": pair.velocity_consistency_active,
    }
    for name, value in vector_fields.items():
        if value.shape != (batch,) or value.device != sample.target.device:
            raise ValueError(f"CFM {name} must have shape [B] and be on the same device as the flow state")
    for name, value in {
        "neighbor_latents": pair.neighbor_latents,
        "segment_endpoint_latents": pair.segment_endpoint_latents,
    }.items():
        if value.shape != sample.target.shape or value.device != sample.target.device:
            raise ValueError(f"CFM {name} must have the same shape and device as the target")
    if pair.requires_neighbor_prediction:
        if neighbor_prediction is None:
            raise ValueError("CFM is missing the second-stage neighbor prediction")
        if neighbor_prediction.shape != primary_prediction.shape:
            raise ValueError("CFM neighbor and primary predictions must have the same shape")
    elif neighbor_prediction is not None:
        raise ValueError("CFM received an unexpected neighbor prediction")

    primary_coefficient = _broadcast_batch_coefficient(
        pair.segment_endpoint_timestep - sample.timestep,
        sample.noisy_latents,
    )
    primary_endpoint = (
        sample.noisy_latents.float()
        + primary_coefficient * primary_prediction.float()
    )
    true_endpoint = pair.segment_endpoint_latents.float()
    if neighbor_prediction is None:
        endpoint_target = true_endpoint

        zero_prediction = primary_prediction.float() * 0.0
        zero_target = primary_prediction.detach().float() * 0.0
        velocity = masked_flow_loss(
            zero_prediction,
            zero_target,
            frame_mask,
            loss_weighting=flow_config.loss_weighting,
            loss_reduction=flow_config.loss_reduction,
            compute_dtype=flow_config.loss_compute_dtype,
        )
    else:
        detached_neighbor = neighbor_prediction.detach().float()
        neighbor_coefficient = _broadcast_batch_coefficient(
            pair.segment_endpoint_timestep - pair.neighbor_timestep,
            pair.neighbor_latents,
        )
        predicted_neighbor_endpoint = (
            pair.neighbor_latents.float()
            + neighbor_coefficient * detached_neighbor
        )
        endpoint_selector = pair.use_predicted_neighbor_endpoint
        for _ in range(primary_prediction.ndim - 1):
            endpoint_selector = endpoint_selector.unsqueeze(-1)
        endpoint_target = torch.where(
            endpoint_selector,
            predicted_neighbor_endpoint,
            true_endpoint,
        )
        velocity_selector = pair.velocity_consistency_active
        for _ in range(primary_prediction.ndim - 1):
            velocity_selector = velocity_selector.unsqueeze(-1)
        zero = torch.zeros(
            (), device=primary_prediction.device, dtype=torch.float32
        )
        velocity_primary = torch.where(
            velocity_selector,
            primary_prediction.float(),
            zero,
        )
        velocity_target = torch.where(
            velocity_selector,
            detached_neighbor,
            zero,
        )
        velocity = masked_flow_loss(
            velocity_primary,
            velocity_target,
            frame_mask,
            loss_weighting=flow_config.loss_weighting,
            loss_reduction=flow_config.loss_reduction,
            compute_dtype=flow_config.loss_compute_dtype,
        )

    endpoint = masked_flow_loss(
        primary_endpoint,
        endpoint_target.detach(),
        frame_mask,
        loss_weighting=flow_config.loss_weighting,
        loss_reduction=flow_config.loss_reduction,
        compute_dtype=flow_config.loss_compute_dtype,
    )
    if not torch.equal(endpoint.denominator, velocity.denominator):
        raise RuntimeError("CFM endpoint and velocity reduction denominators are inconsistent")
    numerator = endpoint.numerator + (
        float(consistency_config.velocity_weight) * velocity.numerator
    )
    denominator = endpoint.denominator
    combined_loss = numerator / denominator
    if not torch.isfinite(combined_loss):
        raise FloatingPointError("consistency flow matching total loss contains NaN or infinity")
    combined = MaskedLoss(
        loss=combined_loss,
        numerator=numerator,
        denominator=denominator,
        valid_frames=endpoint.valid_frames,
    )
    return ConsistencyFlowMatchingLoss(
        combined=combined,
        endpoint=endpoint,
        velocity=velocity,
    )


@torch.no_grad()
def timestep_binned_mse(
    prediction: Tensor,
    target: Tensor,
    frame_mask: Tensor,
    timestep: Tensor,
    *,
    num_bins: int = 10,
) -> tuple[Tensor, Tensor]:

    if not isinstance(num_bins, int) or isinstance(num_bins, bool):
        raise TypeError("num_bins must be a positive integer")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    if (
        prediction.shape != target.shape
        or prediction.ndim < 3
        or frame_mask.shape != prediction.shape[:2]
        or timestep.shape != (prediction.shape[0],)
    ):
        raise ValueError("timestep bucket input shapes are inconsistent")
    if frame_mask.dtype != torch.bool or not timestep.is_floating_point():
        raise TypeError("timestep bucketing requires a bool frame_mask and floating-point timestep")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("timestep bucketing requires floating-point prediction and target tensors")
    if (
        prediction.device != target.device
        or frame_mask.device != prediction.device
        or timestep.device != prediction.device
    ):
        raise ValueError("all timestep bucket tensors must be on the same device")
    if not torch.isfinite(timestep).all() or bool(
        ((timestep < 0.0) | (timestep > 1.0)).any()
    ):
        raise ValueError("timestep bucketing requires finite timesteps in [0, 1]")
    expanded_mask = frame_mask
    for _ in range(prediction.ndim - 2):
        expanded_mask = expanded_mask.unsqueeze(-1)
    delta = torch.where(
        expanded_mask,
        prediction.float() - target.float(),
        torch.zeros((), device=prediction.device, dtype=torch.float32),
    )
    if not torch.isfinite(delta).all():
        raise FloatingPointError("valid timestep bucket values contain NaN or infinity")
    squared = delta.square()
    if not torch.isfinite(squared).all():
        raise FloatingPointError("timestep-binned MSE overflowed to NaN or infinity")
    frame_error = squared.mean(dim=tuple(range(2, delta.ndim)))
    if not torch.isfinite(frame_error).all():
        raise FloatingPointError("timestep-binned frame MSE contains NaN or infinity")
    indices = torch.clamp((timestep.float() * num_bins).long(), 0, num_bins - 1)
    numerators = torch.zeros(num_bins, device=prediction.device, dtype=torch.float64)
    denominators = torch.zeros_like(numerators)
    for index in range(num_bins):
        selected = (indices == index).unsqueeze(1) & frame_mask
        numerators[index] = frame_error.double().masked_fill(~selected, 0.0).sum()
        denominators[index] = selected.sum(dtype=torch.float64)
    if not torch.isfinite(numerators).all() or not torch.isfinite(denominators).all():
        raise FloatingPointError("timestep-binned MSE numerator or denominator contains NaN or infinity")
    return numerators, denominators


@torch.no_grad()
def channelwise_mse_sums(
    prediction: Tensor,
    target: Tensor,
    frame_mask: Tensor,
) -> tuple[Tensor, Tensor]:

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("per-channel MSE requires prediction and target tensors with shape [B, T, D]")
    if frame_mask.dtype != torch.bool or frame_mask.shape != prediction.shape[:2]:
        raise TypeError("per-channel MSE requires a bool frame_mask with shape [B, T]")
    if (
        not prediction.is_floating_point()
        or not target.is_floating_point()
        or prediction.device != target.device
        or frame_mask.device != prediction.device
    ):
        raise TypeError("per-channel MSE requires floating-point prediction and target tensors on the same device")
    delta = torch.where(
        frame_mask.unsqueeze(-1),
        prediction.float() - target.float(),
        torch.zeros((), device=prediction.device, dtype=torch.float32),
    )
    if not torch.isfinite(delta).all():
        raise FloatingPointError("valid per-channel MSE values contain NaN or infinity")
    numerators = delta.square().sum(dim=(0, 1), dtype=torch.float64)
    denominator = frame_mask.sum(dtype=torch.float64)
    if not torch.isfinite(numerators).all() or not bool(denominator > 0):
        raise FloatingPointError("per-channel MSE reduction produced a non-finite value")
    return numerators, denominator


@torch.no_grad()
def timestep_channelwise_mse_sums(
    prediction: Tensor,
    target: Tensor,
    frame_mask: Tensor,
    timestep: Tensor,
    *,
    num_bins: int = 10,
) -> tuple[Tensor, Tensor]:

    if not isinstance(num_bins, int) or isinstance(num_bins, bool) or num_bins <= 0:
        raise ValueError("num_bins must be a positive integer")
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("timestepper-channel MSE requires prediction and target tensors with shape [B, T, D]")
    if frame_mask.dtype != torch.bool or frame_mask.shape != prediction.shape[:2]:
        raise TypeError("timestepper-channel MSE requires a bool frame_mask with shape [B, T]")
    if timestep.shape != (prediction.shape[0],) or not timestep.is_floating_point():
        raise TypeError("timestep-binned per-channel MSE requires a floating-point timestep with shape [B]")
    if (
        not prediction.is_floating_point()
        or not target.is_floating_point()
        or prediction.device != target.device
        or frame_mask.device != prediction.device
        or timestep.device != prediction.device
    ):
        raise TypeError(
            "timestep-binned per-channel MSE requires all tensors on the same device "
            "and floating-point prediction and target tensors"
        )
    if not torch.isfinite(timestep).all() or bool(
        ((timestep < 0.0) | (timestep > 1.0)).any()
    ):
        raise ValueError("timestep-binned per-channel MSE requires finite timesteps in [0, 1]")
    delta = torch.where(
        frame_mask.unsqueeze(-1),
        prediction.float() - target.float(),
        torch.zeros((), device=prediction.device, dtype=torch.float32),
    )
    if not torch.isfinite(delta).all():
        raise FloatingPointError("timestepvalid per-channel MSE values contain NaN or infinity")
    squared = delta.square()
    indices = torch.clamp((timestep.float() * num_bins).long(), 0, num_bins - 1)
    numerators = torch.zeros(
        num_bins,
        prediction.shape[-1],
        dtype=torch.float64,
        device=prediction.device,
    )
    denominators = torch.zeros(
        num_bins,
        dtype=torch.float64,
        device=prediction.device,
    )
    for index in range(num_bins):
        selected = frame_mask & (indices == index).unsqueeze(1)
        numerators[index] = torch.where(
            selected.unsqueeze(-1),
            squared,
            torch.zeros((), device=prediction.device, dtype=squared.dtype),
        ).sum(dim=(0, 1), dtype=torch.float64)
        denominators[index] = selected.sum(dtype=torch.float64)
    if not torch.isfinite(numerators).all() or not torch.isfinite(denominators).all():
        raise FloatingPointError(
            "timestep-binned per-channel MSE reduction produced NaN or infinity"
        )
    return numerators, denominators


def classifier_free_guidance(
    conditional_velocity: torch.Tensor,
    null_velocity: torch.Tensor,
    scale: float | torch.Tensor,
) -> torch.Tensor:

    if conditional_velocity.shape != null_velocity.shape:
        raise ValueError("CFG conditional and null velocities must have the same shape")
    if (
        not conditional_velocity.is_floating_point()
        or not null_velocity.is_floating_point()
        or conditional_velocity.dtype != null_velocity.dtype
        or conditional_velocity.device != null_velocity.device
    ):
        raise TypeError(
            "CFG conditional and null velocities must be floating-point tensors with the same dtype and device"
        )
    if (
        not torch.isfinite(conditional_velocity).all()
        or not torch.isfinite(null_velocity).all()
    ):
        raise FloatingPointError("CFG conditional/null velocity contains NaN/Inf")
    if isinstance(scale, torch.Tensor):
        if scale.dtype == torch.bool or not (
            scale.is_floating_point()
            or scale.dtype
            in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
        ):
            raise TypeError("CFG tensor scale must be a numeric scalar or have shape [B]")
        if scale.ndim == 0:
            pass
        elif scale.ndim == 1 and scale.shape[0] == conditional_velocity.shape[0]:
            pass
        else:
            raise ValueError("CFG tensor scale must be scalar or have shape [B]")
        value = scale.to(
            device=conditional_velocity.device,
            dtype=conditional_velocity.dtype,
        )
        if not torch.isfinite(value).all() or bool((value < 0).any()):
            raise ValueError("CFG scale must be finite and non-negative")
        if value.numel() == 1 and float(value) == 0.0:
            return null_velocity
        if value.numel() == 1 and float(value) == 1.0:
            return conditional_velocity
        while value.ndim < conditional_velocity.ndim:
            value = value.unsqueeze(-1)
    else:
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise TypeError("CFG scale must be a numeric scalar or have shape [B]")
        value = float(scale)
        if not math_is_finite(value) or value < 0:
            raise ValueError("CFG scale must be finite and non-negative")
        if value == 0.0:
            return null_velocity
        if value == 1.0:
            return conditional_velocity
    guided = null_velocity + value * (conditional_velocity - null_velocity)
    if not torch.isfinite(guided).all():
        raise FloatingPointError("CFG output contains NaN/Inf")
    return guided


cfg_combine = classifier_free_guidance


def math_is_finite(value: float) -> bool:

    return value == value and value not in (float("inf"), float("-inf"))


VelocityFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class SolverOutput:
    sample: torch.Tensor
    nfe: int
    solver: str
    num_steps: int
    t_start: float
    t_end: float


def _validate_solver_inputs(
    initial: torch.Tensor,
    *,
    num_steps: int,
    t_start: float,
    t_end: float,
    frame_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if not initial.is_floating_point() or initial.ndim < 2:
        raise TypeError("ODE initial state must be a floating-point tensor with at least two dimensions")
    if not isinstance(num_steps, int) or isinstance(num_steps, bool):
        raise TypeError("num_steps must be a positive integer")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if (
        isinstance(t_start, bool)
        or not isinstance(t_start, (int, float))
        or isinstance(t_end, bool)
        or not isinstance(t_end, (int, float))
    ):
        raise TypeError("t_start and t_end must be numeric")
    if not math_is_finite(float(t_start)) or not math_is_finite(float(t_end)):
        raise ValueError("t_start and t_end must be finite")
    if (
        not 0.0 <= float(t_start) <= 1.0
        or not 0.0 <= float(t_end) <= 1.0
        or float(t_start) == float(t_end)
    ):
        raise ValueError("flow ODE time range must be within [0, 1] and have distinct endpoints")
    if frame_mask is not None:
        if frame_mask.dtype != torch.bool or frame_mask.shape != initial.shape[:2]:
            raise TypeError("frame_mask must be a bool tensor with shape [B, T]")
        if frame_mask.device != initial.device:
            raise ValueError("frame_mask and ODE state must be on the same device")
        if not bool(frame_mask.any(dim=1).all()):
            raise ValueError("ODE integration requires at least one valid frame per sample")
        valid = frame_mask
        for _ in range(initial.ndim - 2):
            valid = valid.unsqueeze(-1)
        checked = torch.where(valid, initial, torch.zeros_like(initial))
    else:
        checked = initial
    if not torch.isfinite(checked).all():
        raise FloatingPointError("ODE initial state contains NaN/Inf")
    return frame_mask


def _time_batch(state: torch.Tensor, time: float | torch.Tensor) -> torch.Tensor:
    dtype = torch.float64 if state.dtype == torch.float64 else torch.float32
    if isinstance(time, torch.Tensor):
        if not time.is_floating_point() or time.ndim not in {0, 1}:
            raise TypeError("ODE timestep must be a floating-point scalar or have shape [B]")
        if time.ndim == 0:
            time = time.expand(state.shape[0])
        if time.shape != (state.shape[0],):
            raise ValueError("per-sample ODE timestep must have shape [B]")
        value = time.to(device=state.device, dtype=dtype)
        if not torch.isfinite(value).all():
            raise ValueError("ODE timestep contains NaN or infinity")
        return value
    return torch.full(
        (state.shape[0],),
        float(time),
        device=state.device,
        dtype=dtype,
    )


def _checked_velocity(
    function: VelocityFunction,
    state: torch.Tensor,
    time: float | torch.Tensor,
    frame_mask: torch.Tensor | None,
    *,
    evaluation_dtype: torch.dtype,
) -> torch.Tensor:
    evaluation_state = state.to(dtype=evaluation_dtype)
    velocity = function(evaluation_state, _time_batch(state, time))
    if not isinstance(velocity, torch.Tensor) or velocity.shape != state.shape:
        raise RuntimeError("velocity function must return a tensor with the same shape as the state")
    if not velocity.is_floating_point() or velocity.device != state.device:
        raise TypeError("velocity must be a floating-point tensor on the same device as the state")
    if frame_mask is not None:
        valid = frame_mask
        for _ in range(state.ndim - 2):
            valid = valid.unsqueeze(-1)
        velocity = velocity.masked_fill(~valid, 0.0)
    if not torch.isfinite(velocity).all():
        if isinstance(time, torch.Tensor):
            time_label = f"[{float(time.min()):.6g},{float(time.max()):.6g}]"
        else:
            time_label = f"{time:.6g}"
        raise FloatingPointError(f"velocity in t={time_label} appears NaN/Inf")


    return velocity.to(dtype=state.dtype)


def _initial_solver_state(
    initial: torch.Tensor, frame_mask: torch.Tensor | None
) -> torch.Tensor:

    dtype = (
        torch.float32
        if initial.dtype in {torch.float16, torch.bfloat16}
        else initial.dtype
    )
    state = initial.to(dtype=dtype).clone()
    if frame_mask is not None:
        valid = frame_mask
        for _ in range(state.ndim - 2):
            valid = valid.unsqueeze(-1)
        state = state.masked_fill(~valid, 0.0)
    return state


def _checked_state(state: torch.Tensor, *, time: float | torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(state).all():
        if isinstance(time, torch.Tensor):
            time_label = f"[{float(time.min()):.6g},{float(time.max()):.6g}]"
        else:
            time_label = f"{time:.6g}"
        raise FloatingPointError(f"ODE state in t={time_label} appears NaN/Inf")
    return state


def _validated_sampling_schedule(
    schedule: torch.Tensor | None,
    initial: torch.Tensor,
    *,
    num_steps: int,
    t_start: float,
    t_end: float,
) -> torch.Tensor | None:
    if schedule is None:
        return None
    if not isinstance(schedule, torch.Tensor) or not schedule.is_floating_point():
        raise TypeError("ODE schedule must be a floating-point tensor")
    if schedule.ndim == 1:
        if schedule.shape != (num_steps + 1,):
            raise ValueError("one-dimensional ODE schedule must contain num_steps + 1 points")
    elif schedule.ndim == 2:
        if schedule.shape != (initial.shape[0], num_steps + 1):
            raise ValueError("2D ODE schedule must have shape [B, num_steps + 1]")
    else:
        raise ValueError("ODE schedule must be one- or two-dimensional")
    value = schedule.to(device=initial.device, dtype=torch.float32)
    if not torch.isfinite(value).all() or bool(((value < 0.0) | (value > 1.0)).any()):
        raise ValueError("ODE schedule must be finite and in [0, 1]")
    first = value[..., 0]
    last = value[..., -1]
    if not torch.equal(
        first, torch.full_like(first, float(t_start))
    ) or not torch.equal(last, torch.full_like(last, float(t_end))):
        raise ValueError("ODE schedule endpoints must exactly match t_start and t_end")
    deltas = value[..., 1:] - value[..., :-1]
    if t_end > t_start:
        valid_direction = deltas > 0
    else:
        valid_direction = deltas < 0
    if not bool(valid_direction.all()):
        raise ValueError("ODE schedule must be strictly monotonic along the integration direction")
    return value


def _schedule_step(
    schedule: torch.Tensor | None,
    *,
    index: int,
    num_steps: int,
    t_start: float,
    t_end: float,
) -> tuple[float | torch.Tensor, float | torch.Tensor, float | torch.Tensor]:
    if schedule is None:
        step_size = (float(t_end) - float(t_start)) / int(num_steps)
        time = float(t_start) + index * step_size
        next_time = (
            float(t_end)
            if index == int(num_steps) - 1
            else float(t_start) + (index + 1) * step_size
        )
        return time, next_time, step_size
    if schedule.ndim == 1:
        time = float(schedule[index])
        next_time = float(schedule[index + 1])
        return time, next_time, next_time - time
    time = schedule[:, index]
    next_time = schedule[:, index + 1]
    return time, next_time, next_time - time


def _broadcast_step_size(
    step_size: float | torch.Tensor,
    state: torch.Tensor,
) -> float | torch.Tensor:
    if not isinstance(step_size, torch.Tensor):
        return step_size
    value = step_size.to(device=state.device, dtype=state.dtype)
    return value.reshape(value.shape[0], *([1] * (state.ndim - 1)))


def euler_solve(
    function: VelocityFunction,
    initial: torch.Tensor,
    *,
    num_steps: int,
    t_start: float = 0.0,
    t_end: float = 1.0,
    frame_mask: torch.Tensor | None = None,
    schedule: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_solver_inputs(
        initial,
        num_steps=num_steps,
        t_start=t_start,
        t_end=t_end,
        frame_mask=frame_mask,
    )
    schedule = _validated_sampling_schedule(
        schedule,
        initial,
        num_steps=num_steps,
        t_start=t_start,
        t_end=t_end,
    )
    state = _initial_solver_state(initial, frame_mask)
    for index in range(int(num_steps)):
        time, next_time, step_size = _schedule_step(
            schedule,
            index=index,
            num_steps=num_steps,
            t_start=t_start,
            t_end=t_end,
        )
        broadcast_step = _broadcast_step_size(step_size, state)
        state = _checked_state(
            state
            + broadcast_step
            * _checked_velocity(
                function,
                state,
                time,
                frame_mask,
                evaluation_dtype=initial.dtype,
            ),
            time=next_time,
        )
    return state


def heun_solve(
    function: VelocityFunction,
    initial: torch.Tensor,
    *,
    num_steps: int,
    t_start: float = 0.0,
    t_end: float = 1.0,
    frame_mask: torch.Tensor | None = None,
    schedule: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_solver_inputs(
        initial,
        num_steps=num_steps,
        t_start=t_start,
        t_end=t_end,
        frame_mask=frame_mask,
    )
    schedule = _validated_sampling_schedule(
        schedule,
        initial,
        num_steps=num_steps,
        t_start=t_start,
        t_end=t_end,
    )
    state = _initial_solver_state(initial, frame_mask)
    for index in range(int(num_steps)):
        time, next_time, step_size = _schedule_step(
            schedule,
            index=index,
            num_steps=num_steps,
            t_start=t_start,
            t_end=t_end,
        )
        broadcast_step = _broadcast_step_size(step_size, state)
        first = _checked_velocity(
            function,
            state,
            time,
            frame_mask,
            evaluation_dtype=initial.dtype,
        )
        predictor = _checked_state(
            state + broadcast_step * first,
            time=next_time,
        )
        second = _checked_velocity(
            function,
            predictor,
            next_time,
            frame_mask,
            evaluation_dtype=initial.dtype,
        )
        state = _checked_state(
            state + 0.5 * broadcast_step * (first + second),
            time=next_time,
        )
    return state


euler_solver = euler_solve
heun_solver = heun_solve


def solve_flow(
    function: VelocityFunction,
    initial: torch.Tensor,
    *,
    solver: str,
    num_steps: int,
    t_start: float = 0.0,
    t_end: float = 1.0,
    frame_mask: torch.Tensor | None = None,
    schedule: torch.Tensor | None = None,
) -> SolverOutput:
    if solver == "euler":
        sample = euler_solve(
            function,
            initial,
            num_steps=num_steps,
            t_start=t_start,
            t_end=t_end,
            frame_mask=frame_mask,
            schedule=schedule,
        )
        nfe = int(num_steps)
    elif solver == "heun":
        sample = heun_solve(
            function,
            initial,
            num_steps=num_steps,
            t_start=t_start,
            t_end=t_end,
            frame_mask=frame_mask,
            schedule=schedule,
        )
        nfe = 2 * int(num_steps)
    else:
        raise ValueError(f"unsupported solver={solver!r}; expected one of {SUPPORTED_SOLVERS}")
    return SolverOutput(
        sample=sample,
        nfe=nfe,
        solver=solver,
        num_steps=int(num_steps),
        t_start=float(t_start),
        t_end=float(t_end),
    )


def sampling_interval(config: FlowConfig | Mapping[str, Any]) -> tuple[float, float]:

    resolved = (
        config if isinstance(config, FlowConfig) else FlowConfig.from_mapping(config)
    )
    resolved.validate()
    if resolved.time_direction == "noise_to_data":
        return 0.0, 1.0
    return 1.0, 0.0


def sampling_schedule(
    config: FlowConfig | Mapping[str, Any],
    *,
    num_steps: int,
    reference: torch.Tensor,
    effective_lengths: torch.Tensor | list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:

    resolved = (
        config if isinstance(config, FlowConfig) else FlowConfig.from_mapping(config)
    )
    resolved.validate()
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0:
        raise ValueError("sampling schedule num_steps must be a positive integer")
    if not isinstance(reference, torch.Tensor) or not reference.is_floating_point():
        raise TypeError("sampling schedule reference must be a floating-point tensor")
    start, end = sampling_interval(resolved)
    base = torch.linspace(
        start,
        end,
        num_steps + 1,
        device=reference.device,
        dtype=torch.float32,
    )
    if resolved.timestep_shift == "none":
        return base
    if effective_lengths is None:
        raise ValueError("length-aware sampling schedule requires effective_lengths")
    lengths = torch.as_tensor(effective_lengths, device=reference.device)
    if lengths.ndim != 1:
        raise ValueError("sampling effective_lengths must have shape [B]")
    grid = base.unsqueeze(0).expand(int(lengths.shape[0]), -1).clone()
    shifted = shift_timesteps_by_length(grid, lengths, resolved)
    shifted[:, 0] = float(start)
    shifted[:, -1] = float(end)
    return shifted


class FlowMatchingObjective:

    def __init__(self, config: FlowConfig | Mapping[str, Any]) -> None:
        self.config = (
            config
            if isinstance(config, FlowConfig)
            else FlowConfig.from_mapping(config)
        )
        self.config.validate()

    def prepare(
        self,
        target: torch.Tensor,
        *,
        source_generator: torch.Generator | None = None,
        timestep_generator: torch.Generator | None = None,
        source: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        apply_source_coupling: bool = True,
    ) -> FlowTrainingSample:
        return make_flow_training_sample(
            target,
            self.config,
            source_generator=source_generator,
            timestep_generator=timestep_generator,
            source=source,
            timestep=timestep,
            frame_mask=frame_mask,
            apply_source_coupling=apply_source_coupling,
        )

    def loss(
        self,
        prediction: torch.Tensor,
        target_velocity: torch.Tensor,
        frame_mask: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> MaskedLoss:
        return masked_flow_loss(
            prediction,
            target_velocity,
            frame_mask,
            sample_weight=sample_weight,
            loss_weighting=self.config.loss_weighting,
            loss_reduction=self.config.loss_reduction,
            compute_dtype=self.config.loss_compute_dtype,
        )
