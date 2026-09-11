
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def log_frequency_prior(
    frequency_bins: int,
    *,
    min_relative_frequency: float = 1.0 / 480.0,
    strength: float = 0.5,
    epsilon: float = 1.0e-6,
    mode: str = "scaled",
    device: torch.device | str | None = None,
) -> Tensor:

    if frequency_bins <= 0:
        raise ValueError("frequency_bins must be positive")
    if min_relative_frequency <= 0 or epsilon <= 0:
        raise ValueError("min_relative_frequency must be positive")
    if mode not in {"scaled", "eq6"}:
        raise ValueError("prior mode must be scaled or eq6")
    relative = torch.linspace(
        0.0, 1.0, frequency_bins, dtype=torch.float32, device=device
    )
    reference = relative.mean()
    if mode == "eq6":
        return torch.log(relative / reference + epsilon)
    relative = relative.clamp_min(min_relative_frequency)
    return strength * torch.log(relative / relative.mean())


class SpecSnakeBeta(nn.Module):

    def __init__(
        self,
        channels: int,
        frequency_bins: int,
        *,
        alpha_init: float = 0.0,
        beta_init: float = 0.0,
        prior_strength: float = 0.5,
        epsilon: float = 1.0e-6,
        log_parameter_limit: float = 12.0,
        denominator_mode: str = "add_epsilon",
        parameterization: str = "channel_frequency",
        prior_mode: str = "scaled",
        finite_check: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0 or frequency_bins <= 0:
            raise ValueError("channels/frequency_bins must be positive")
        if epsilon <= 0 or log_parameter_limit <= 0:
            raise ValueError("epsilon/log_parameter_limit must be positive")
        if denominator_mode != "add_epsilon":
            raise ValueError("denominator_mode must be add_epsilon")
        if parameterization not in {"channel_frequency", "frequency_only"}:
            raise ValueError("parameterizationmust bechannel_frequency/frequency_only")
        prior = log_frequency_prior(
            frequency_bins,
            strength=prior_strength,
            epsilon=epsilon,
            mode=prior_mode,
        ).view(1, 1, frequency_bins, 1)
        if prior_mode == "eq6":


            prior_extent = float(prior.detach().abs().max().cpu())
            log_parameter_limit = max(
                float(log_parameter_limit),
                prior_extent + torch.finfo(torch.float32).eps,
            )
        parameter_channels = 1 if parameterization == "frequency_only" else channels
        self.alpha = nn.Parameter(
            prior.expand(1, parameter_channels, frequency_bins, 1).clone() + alpha_init
        )
        self.beta = nn.Parameter(
            torch.full(
                (1, parameter_channels, frequency_bins, 1),
                float(beta_init),
            )
        )
        self.channels = int(channels)
        self.frequency_bins = int(frequency_bins)
        self.epsilon = float(epsilon)
        self.log_parameter_limit = float(log_parameter_limit)
        self.denominator_mode = denominator_mode
        self.parameterization = parameterization
        self.prior_mode = prior_mode
        self.finite_check = bool(finite_check)

    def _parameters_for(self, bins: int) -> tuple[Tensor, Tensor]:
        if bins == self.frequency_bins:
            return self.alpha, self.beta
        alpha = F.interpolate(
            self.alpha.squeeze(-1),
            size=bins,
            mode="linear",
            align_corners=True,
        ).unsqueeze(-1)
        beta = F.interpolate(
            self.beta.squeeze(-1),
            size=bins,
            mode="linear",
            align_corners=True,
        ).unsqueeze(-1)
        return alpha, beta

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError(
                f"SpecSnakeBeta requires [B,{self.channels},F,T],"
                f"received {tuple(inputs.shape)}"
            )
        original_dtype = inputs.dtype
        work = inputs.float()
        alpha, beta = self._parameters_for(inputs.shape[2])
        alpha = alpha.float().clamp(-self.log_parameter_limit, self.log_parameter_limit)
        beta = beta.float().clamp(-self.log_parameter_limit, self.log_parameter_limit)
        frequency = torch.exp(alpha)
        denominator = torch.exp(beta)
        denominator = (
            denominator + self.epsilon
            if self.denominator_mode == "add_epsilon"
            else denominator.clamp_min(self.epsilon)
        )
        modulation = torch.sin(work * frequency).square() / denominator
        result = work + modulation

        if self.finite_check and not torch.isfinite(result).all():
            raise FloatingPointError("SpecSnakeBeta produces NaN/Inf")
        return result.to(original_dtype)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, frequency_bins={self.frequency_bins}, "
            f"epsilon={self.epsilon:g}"
        )


Spec_SnakeBeta = SpecSnakeBeta
