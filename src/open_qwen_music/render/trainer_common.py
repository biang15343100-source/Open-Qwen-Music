
from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from open_qwen_music.common.distributed import reduce_scalar_sum

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


class MetricAccumulator:

    def __init__(self) -> None:
        self._values: dict[str, list[float]] = {}

    def add(
        self,
        name: str,
        numerator: float | Tensor,
        denominator: float | Tensor = 1.0,
    ) -> None:
        num = (
            float(numerator.detach().double().item())
            if isinstance(numerator, Tensor)
            else float(numerator)
        )
        den = (
            float(denominator.detach().double().item())
            if isinstance(denominator, Tensor)
            else float(denominator)
        )
        if not math.isfinite(num) or not math.isfinite(den) or den < 0:
            raise FloatingPointError(
                f"Invalid metric {name}: numerator={num}, denominator={den}"
            )
        current = self._values.setdefault(name, [0.0, 0.0])
        current[0] += num
        current[1] += den

    def update(self, values: Mapping[str, tuple[float, float]]) -> None:
        for name, (numerator, denominator) in values.items():
            self.add(name, numerator, denominator)

    def compute(self, *, distributed: bool = False) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for name, (numerator, denominator) in self._values.items():
            if distributed:
                numerator = reduce_scalar_sum(numerator)
                denominator = reduce_scalar_sum(denominator)
            result[name] = {
                "value": numerator / denominator if denominator > 0 else 0.0,
                "numerator": numerator,
                "denominator": denominator,
            }
        return result

    def reset(self) -> None:
        self._values.clear()

    def state_dict(self) -> dict[str, list[float]]:
        return {name: list(values) for name, values in self._values.items()}

    def load_state_dict(self, state: Mapping[str, Iterable[float]]) -> None:
        self._values = {}
        for name, values in state.items():
            pair = list(values)
            if len(pair) != 2:
                raise ValueError("MetricAccumulator state must be [num,den]")
            self._values[name] = [float(pair[0]), float(pair[1])]


class JsonlLogger:

    def __init__(self, path: str | Path, *, fsync: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = bool(fsync)

    def log(self, event: str, **payload: Any) -> None:
        record = {
            "time_unix": time.time(),
            "event": str(event),
            **payload,
        }
        line = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        with self.path.open("a", encoding="utf-8") as output:
            output.write(line + "\n")
            output.flush()
            if self.fsync:
                os.fsync(output.fileno())

    __call__ = log


@torch.no_grad()
def zeropower_via_newton_schulz(
    gradient: Tensor,
    *,
    steps: int = 5,
    epsilon: float = 1.0e-7,
) -> Tensor:

    if gradient.ndim < 2:
        raise ValueError("Newton-Schulz only applies to matrix-like parameters")
    matrix = gradient.float().reshape(gradient.shape[0], -1)
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.transpose(0, 1)
    matrix = matrix / matrix.norm().clamp_min(epsilon)

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        gram = matrix @ matrix.transpose(0, 1)
        matrix = a * matrix + (b * gram + c * (gram @ gram)) @ matrix
    if transposed:
        matrix = matrix.transpose(0, 1)
    matrix *= math.sqrt(max(1.0, matrix.shape[0] / matrix.shape[1]))
    return matrix.reshape_as(gradient).to(gradient.dtype)


def zeropower_via_newton_schulz_ear_v2(
    gradient: Tensor,
    *,
    steps: int = 5,
    epsilon: float = 1.0e-7,
) -> Tensor:

    if gradient.ndim != 2:
        raise ValueError("Newton-Schulz only accepts two-dimensional matrices")
    matrix = gradient.bfloat16()
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.transpose(0, 1)
    matrix = matrix / (matrix.norm() + float(epsilon))
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        gram = matrix @ matrix.transpose(0, 1)


        polynomial = b * gram + c * gram @ gram
        matrix = a * matrix + polynomial @ matrix
    if transposed:
        matrix = matrix.transpose(0, 1)
    return matrix


class Muon(Optimizer):

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 1.0e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
        non_matrix_rule: str = "momentum_sgd",
        matrix_nesterov_rule: str = "source_add",
        auxiliary_betas: tuple[float, float] | list[float] = (0.9, 0.95),
        auxiliary_eps: float = 1.0e-8,
        implementation: str = "oqm_v1",
    ) -> None:
        if lr <= 0 or not 0 <= momentum < 1 or weight_decay < 0 or ns_steps <= 0:
            raise ValueError("Muon lr, momentum, weight_decay, or ns_steps is invalid")
        if non_matrix_rule not in {"momentum_sgd", "adamw"}:
            raise ValueError("Muon non_matrix_rule must be momentum_sgd or adamw")
        if matrix_nesterov_rule not in {
            "standard_lerp",
            "source_add",
        }:
            raise ValueError("Muon matrix_nesterov_rule is invalid")
        if len(auxiliary_betas) != 2:
            raise ValueError("Muon auxiliary_betas must contain two values")
        if implementation not in {"oqm_v1", "ear_v2_source_v1"}:
            raise ValueError("Muon implementation is invalid")
        beta1, beta2 = (float(value) for value in auxiliary_betas)
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or auxiliary_eps <= 0:
            raise ValueError("Muon auxiliary AdamW parameters are invalid")
        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
            "nesterov": bool(nesterov),
            "ns_steps": int(ns_steps),
            "orthogonalize": None,
            "non_matrix_rule": str(non_matrix_rule),
            "matrix_nesterov_rule": str(matrix_nesterov_rule),
            "auxiliary_betas": (beta1, beta2),
            "auxiliary_eps": float(auxiliary_eps),
            "implementation": str(implementation),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        gradients_by_device: dict[torch.device, list[Tensor]] = {}
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    if parameter.grad.is_sparse:
                        raise RuntimeError("Muon does not support sparse gradient")
                    gradients_by_device.setdefault(parameter.grad.device, []).append(
                        parameter.grad
                    )


        for device, gradients in gradients_by_device.items():
            nonfinite = torch.zeros((), dtype=torch.bool, device=device)
            for gradient in gradients:
                nonfinite.logical_or_(~torch.isfinite(gradient).all())
            if bool(nonfinite):
                raise FloatingPointError("Muon received NaN/Inf gradient")
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradient")
                orthogonalize = group.get("orthogonalize")
                use_orthogonal = (
                    parameter.ndim >= 2
                    if orthogonalize is None
                    else bool(orthogonalize)
                )
                state = self.state[parameter]
                if (
                    group.get("implementation", "oqm_v1") == "ear_v2_source_v1"
                    and use_orthogonal
                ):
                    original_shape = gradient.shape
                    matrix = (
                        gradient.reshape(gradient.shape[0], -1)
                        if gradient.ndim > 2
                        else gradient
                    )
                    momentum_buffer = state.setdefault(
                        "momentum_buffer",
                        torch.zeros_like(matrix),
                    )
                    momentum_buffer.mul_(group["momentum"]).add_(matrix)
                    update = (
                        matrix.add(
                            momentum_buffer,
                            alpha=group["momentum"],
                        )
                        if group["nesterov"]
                        else momentum_buffer
                    )
                    update = zeropower_via_newton_schulz_ear_v2(
                        update,
                        steps=group["ns_steps"],
                    )
                    if gradient.ndim > 2:
                        update = update.reshape(original_shape)
                    adjusted_lr = (
                        group["lr"]
                        * 0.2
                        * math.sqrt(max(parameter.shape[0], parameter.shape[1]))
                    )
                    if group["weight_decay"]:
                        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-adjusted_lr)
                    continue
                if not use_orthogonal and group["non_matrix_rule"] == "adamw":
                    if group.get("implementation", "oqm_v1") == "ear_v2_source_v1":
                        if group["weight_decay"]:
                            parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                        beta1, beta2 = group["auxiliary_betas"]
                        state["step"] = int(state.get("step", 0)) + 1
                        moment1 = state.setdefault(
                            "moment1",
                            torch.zeros_like(parameter),
                        )
                        moment2 = state.setdefault(
                            "moment2",
                            torch.zeros_like(parameter),
                        )
                        moment1.lerp_(gradient, 1.0 - beta1)
                        moment2.lerp_(gradient.square(), 1.0 - beta2)
                        normalized = moment1 / (group["auxiliary_eps"] + moment2.sqrt())
                        scale = (1.0 - beta1 ** state["step"]) / math.sqrt(
                            1.0 - beta2 ** state["step"]
                        )
                        parameter.add_(
                            normalized,
                            alpha=-group["lr"] / scale,
                        )
                        continue
                    if group["weight_decay"]:
                        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    beta1, beta2 = group["auxiliary_betas"]
                    state["step"] = int(state.get("step", 0)) + 1
                    exp_avg = state.setdefault(
                        "exp_avg",
                        torch.zeros_like(parameter),
                    )
                    exp_avg_sq = state.setdefault(
                        "exp_avg_sq",
                        torch.zeros_like(parameter),
                    )
                    exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(
                        gradient,
                        gradient,
                        value=1.0 - beta2,
                    )
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    denominator = (
                        exp_avg_sq.sqrt()
                        .div_(math.sqrt(bias_correction2))
                        .add_(group["auxiliary_eps"])
                    )
                    parameter.addcdiv_(
                        exp_avg,
                        denominator,
                        value=-group["lr"] / bias_correction1,
                    )
                    continue
                if group["weight_decay"]:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                momentum_buffer = state.setdefault(
                    "momentum_buffer",
                    torch.zeros_like(parameter),
                )
                momentum_buffer.mul_(group["momentum"]).add_(
                    gradient,
                    alpha=1.0 - group["momentum"],
                )
                if not group["nesterov"]:
                    update = momentum_buffer
                elif group["matrix_nesterov_rule"] == "standard_lerp":
                    update = torch.lerp(
                        gradient,
                        momentum_buffer,
                        group["momentum"],
                    )
                else:
                    update = gradient + group["momentum"] * momentum_buffer
                if use_orthogonal:
                    update = zeropower_via_newton_schulz(
                        update, steps=group["ns_steps"]
                    )
                parameter.add_(update, alpha=-group["lr"])
        return loss


def build_optimizer(
    parameters: Iterable[Tensor] | Iterable[dict[str, Any]],
    config: Mapping[str, Any],
) -> Optimizer:
    name = str(config.get("name", "")).lower()
    kwargs = {
        key: value
        for key, value in config.items()
        if key not in {"name", "parameter_grouping"}
    }
    if name == "muon":
        return Muon(parameters, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)
    if name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    raise ValueError(f"Unknown optimizer={name!r}; Muon will not silently fall back to Adam when unavailable")


def partition_named_parameters_for_muon(
    module: nn.Module,
    *,
    policy: str,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:

    if policy not in {
        "oqm_v1",
        "ear_v2_source_v1",
        "ear_v2_discriminator_v1",
    }:
        raise ValueError("Muon parameter grouping policy is invalid")
    matrix_parameters: list[Tensor] = []
    matrix_names: list[str] = []
    auxiliary_parameters: list[Tensor] = []
    auxiliary_names: list[str] = []
    oqm_auxiliary_markers = (
        "parametrizations.weight.original0",
        ".alpha",
        ".beta",
        ".bias",
        ".norm.",
        "layer_scale",
    )
    seen: set[int] = set()
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        identity = id(parameter)
        if identity in seen:
            raise RuntimeError(f"Optimizer contains a duplicate parameter object: {name}")
        seen.add(identity)
        if policy == "ear_v2_source_v1":
            use_muon = (
                parameter.ndim >= 2
                and "project_out" not in name
                and "conv_post" not in name
            )
        elif policy == "ear_v2_discriminator_v1":


            path_parts = name.split(".")
            is_normalization = any(part.startswith("norm") for part in path_parts[:-1])
            is_post_convolution = "output" in path_parts[:-1]
            use_muon = (
                parameter.ndim >= 2 and not is_post_convolution and not is_normalization
            )
        else:
            use_muon = parameter.ndim >= 2 and not any(
                marker in name for marker in oqm_auxiliary_markers
            )
        if use_muon:
            matrix_parameters.append(parameter)
            matrix_names.append(name)
        else:
            auxiliary_parameters.append(parameter)
            auxiliary_names.append(name)
    expected = {
        id(parameter) for parameter in module.parameters() if parameter.requires_grad
    }
    if seen != expected:
        raise RuntimeError("Optimizer parameter groups do not cover all trainable parameters")
    groups: list[dict[str, Any]] = []
    if matrix_parameters:
        groups.append({"params": matrix_parameters, "orthogonalize": True})
    if auxiliary_parameters:
        groups.append({"params": auxiliary_parameters, "orthogonalize": False})
    return groups, {
        "muon": matrix_names,
        "adamw_like": auxiliary_names,
    }


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    *,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    if not 0 <= warmup_steps < max_steps:
        raise ValueError("must satisfy 0 <= warmup_steps < max_steps")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(1.0e-8, step / max(1, warmup_steps))
        progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def seed_everything(seed: int, *, rank: int = 0, step: int = 0) -> int:
    resolved = int(seed) + int(rank) + 7_919 * int(step)
    random.seed(resolved)
    if np is not None:
        np.random.seed(resolved % (2**32))
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(resolved)
    return resolved


def move_to_device(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if hasattr(value, "to") and value.__class__.__module__.startswith(
        "open_qwen_music"
    ):
        return value.to(device)
    return value


@contextmanager
def temporary_requires_grad(
    module: nn.Module,
    enabled: bool,
) -> Iterator[None]:
    previous = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)
        yield
    finally:
        for parameter, value in zip(module.parameters(), previous):
            parameter.requires_grad_(value)


def autocast_context(
    device: torch.device,
    *,
    enabled: bool,
    dtype: torch.dtype = torch.bfloat16,
):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def finite_gradient_norm(parameters: Iterable[Tensor]) -> Tensor:
    gradients = [
        parameter.grad.detach().float().norm()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return torch.zeros(())
    norm = torch.stack(gradients).norm()
    if not torch.isfinite(norm):
        raise FloatingPointError("gradient norm is NaN/Inf")
    return norm
