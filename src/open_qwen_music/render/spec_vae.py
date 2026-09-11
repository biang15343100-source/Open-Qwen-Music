from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .contracts import (
    LATENT_DIM,
    SPEC_FRAMES_PER_LATENT_FRAME,
    lengths_to_mask,
    validate_spectrum,
)
from .source_spec_vae import SourceSpecDecoder, SourceSpecEncoder


VAE_REVISION = "open-qwen-music-acoustic-vae-v1"


@dataclass(frozen=True)
class SpecVAEConfig:
    """Configuration for the released acoustic VAE."""

    latent_dim: int = LATENT_DIM
    posterior_variance_epsilon: float = 1.0e-6
    encoder_base_channels: int = 64
    decoder_base_channels: int = 128
    weight_norm: bool = True
    revision: str = VAE_REVISION

    def __post_init__(self) -> None:
        if self.latent_dim != LATENT_DIM:
            raise ValueError(f"latent_dim must be {LATENT_DIM}")
        if not isfinite(self.posterior_variance_epsilon) or self.posterior_variance_epsilon <= 0:
            raise ValueError("posterior_variance_epsilon must be finite and positive")
        if self.encoder_base_channels <= 0 or self.decoder_base_channels <= 0:
            raise ValueError("VAE base channel counts must be positive")
        if self.revision != VAE_REVISION:
            raise ValueError(f"revision must be {VAE_REVISION!r}")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SpecVAEConfig":
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Spec-VAE config contains unknown fields: {sorted(unknown)}")
        return cls(**values)


def normalize_spec_vae_checkpoint_config(values: dict[str, Any]) -> dict[str, Any]:
    """Validate the released checkpoint schema at the loading boundary."""
    normalized = dict(values)
    if normalized.get("revision") != VAE_REVISION:
        raise ValueError("Spec-VAE checkpoint has an unsupported revision")
    return normalized


def masked_reduce_time(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape != mask.shape:
        raise ValueError("Time reduction values and mask must have the same shape")
    numerator = torch.where(mask, values, torch.zeros_like(values)).sum()
    return numerator / mask.sum().clamp_min(1)


@dataclass
class SoftplusDiagonalGaussianPosterior:
    mean: Tensor
    raw_scale: Tensor
    mask: Tensor
    variance_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.mean.shape != self.raw_scale.shape:
            raise ValueError("Posterior mean and raw_scale must have the same shape")
        if self.mean.ndim != 3 or self.mean.shape[-1] != LATENT_DIM:
            raise ValueError(f"Posterior tensors must have shape [B, T, {LATENT_DIM}]")
        if self.mask.shape != self.mean.shape[:2] or self.mask.dtype != torch.bool:
            raise ValueError("Posterior mask must be a boolean [B, T] tensor")
        if not isfinite(self.variance_epsilon) or self.variance_epsilon <= 0:
            raise ValueError("Posterior variance epsilon must be finite and positive")

    @property
    def std(self) -> Tensor:
        return F.softplus(self.raw_scale.float()).to(self.mean.dtype)

    @property
    def sample_variance(self) -> Tensor:
        return self.std.float().square().to(self.mean.dtype)

    @property
    def variance(self) -> Tensor:
        return (self.sample_variance.float() + self.variance_epsilon).to(self.mean.dtype)

    @property
    def logvar(self) -> Tensor:
        return self.variance.float().log().to(self.mean.dtype)

    @property
    def actual_sample_logvar(self) -> Tensor:
        return (
            self.sample_variance.float()
            .clamp_min(torch.finfo(torch.float32).tiny)
            .log()
            .to(self.mean.dtype)
        )

    def sample(
        self,
        generator: torch.Generator | None = None,
        *,
        contiguous_source_layout: bool = False,
    ) -> Tensor:
        source_layout = self.mean.transpose(1, 2)
        noise_source_layout = (
            torch.empty(
                source_layout.shape,
                dtype=source_layout.dtype,
                device=source_layout.device,
            )
            if contiguous_source_layout
            else torch.empty_like(source_layout)
        )
        noise = noise_source_layout.normal_(generator=generator).transpose(1, 2)
        return (self.mean + self.std * noise) * self.mask.unsqueeze(-1)

    def mode(self) -> Tensor:
        return self.mean * self.mask.unsqueeze(-1)

    def source_surrogate_kl(self, *, include_half: bool = False) -> Tensor:
        variance = self.variance.float()
        values = (self.mean.float().square() + variance - variance.log() - 1.0).sum(
            dim=-1
        )
        reduced = masked_reduce_time(values, self.mask)
        return reduced * (0.5 if include_half else 1.0)

    def source_surrogate_components(self) -> tuple[Tensor, Tensor]:
        mean_values = self.mean.float().square().sum(dim=-1)
        variance_values = (self.variance.float() - self.logvar.float() - 1.0).sum(
            dim=-1
        )
        return (
            masked_reduce_time(mean_values, self.mask),
            masked_reduce_time(variance_values, self.mask),
        )

    def actual_sample_q_half_kl(self) -> Tensor:
        variance = self.sample_variance.float().clamp_min(
            torch.finfo(torch.float32).tiny
        )
        values = 0.5 * (
            self.mean.float().square() + variance - variance.log() - 1.0
        ).sum(dim=-1)
        return masked_reduce_time(values, self.mask)

    def kl(self, *, reduction: str = "mean") -> Tensor:
        values = 0.5 * (
            self.mean.float().square()
            + self.variance.float()
            - self.logvar.float()
            - 1.0
        )
        masked = values * self.mask.unsqueeze(-1)
        if reduction == "mean":
            valid_dims = self.mask.sum().clamp_min(1) * self.mean.shape[-1]
            return masked.sum() / valid_dims
        if reduction == "sum":
            return masked.sum()
        if reduction == "none":
            return masked
        raise ValueError(f"Unknown KL reduction: {reduction}")


@dataclass
class SpecVAEEncodeOutput:
    posterior: SoftplusDiagonalGaussianPosterior
    latent_lengths: Tensor
    latent_mask: Tensor

    @property
    def mean(self) -> Tensor:
        return self.posterior.mean


@dataclass
class SpecVAEOutput:
    reconstruction: Tensor
    posterior: SoftplusDiagonalGaussianPosterior
    latents: Tensor
    spectrum_lengths: Tensor
    spectrum_mask: Tensor
    latent_lengths: Tensor
    latent_mask: Tensor

    @property
    def sample(self) -> Tensor:
        return self.reconstruction

    @property
    def latent(self) -> Tensor:
        return self.latents


class SpecVAE(nn.Module):
    """The released complex-spectrogram VAE."""

    def __init__(self, config: SpecVAEConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        if config is None:
            config = SpecVAEConfig()
        elif isinstance(config, dict):
            config = SpecVAEConfig.from_dict(config)
        self.config = config
        self.encoder = SourceSpecEncoder(
            base_channels=config.encoder_base_channels,
            latent_dim=config.latent_dim,
            weight_norm_enabled=config.weight_norm,
        )
        self.decoder = SourceSpecDecoder(
            base_channels=config.decoder_base_channels,
            latent_dim=config.latent_dim,
            weight_norm_enabled=config.weight_norm,
        )

    def encode(
        self,
        spectrum: Tensor,
        spectrum_lengths: Tensor | Iterable[int] | None = None,
    ) -> SpecVAEEncodeOutput:
        if spectrum.ndim == 3:
            spectrum = spectrum.unsqueeze(0)
        lengths = validate_spectrum(spectrum, spectrum_lengths)
        canonical_frames = (
            (spectrum.shape[-1] + SPEC_FRAMES_PER_LATENT_FRAME - 1)
            // SPEC_FRAMES_PER_LATENT_FRAME
            * SPEC_FRAMES_PER_LATENT_FRAME
        )
        if canonical_frames != spectrum.shape[-1]:
            spectrum = F.pad(spectrum, (0, canonical_frames - spectrum.shape[-1]))
        moments, latent_lengths, latent_mask = self.encoder(spectrum, lengths)
        mean, raw_scale = moments.chunk(2, dim=-1)
        posterior = SoftplusDiagonalGaussianPosterior(
            mean=mean * latent_mask.unsqueeze(-1),
            raw_scale=raw_scale * latent_mask.unsqueeze(-1),
            mask=latent_mask,
            variance_epsilon=self.config.posterior_variance_epsilon,
        )
        return SpecVAEEncodeOutput(
            posterior=posterior,
            latent_lengths=latent_lengths,
            latent_mask=latent_mask,
        )

    def decode(
        self,
        latents: Tensor,
        *,
        spectrum_frames: int | None = None,
        latent_lengths: Tensor | None = None,
        spectrum_lengths: Tensor | None = None,
    ) -> Tensor:
        if latents.ndim == 2:
            latents = latents.unsqueeze(0)
        return self.decoder(
            latents,
            spectrum_frames=spectrum_frames,
            latent_lengths=latent_lengths,
            spectrum_lengths=spectrum_lengths,
        )

    def forward(
        self,
        spectrum: Tensor,
        spectrum_lengths: Tensor | Iterable[int] | None = None,
        *,
        sample_posterior: bool | None = None,
        generator: torch.Generator | None = None,
        posterior_sample_layout: str = "contiguous_bdt",
    ) -> SpecVAEOutput:
        if spectrum.ndim == 3:
            spectrum = spectrum.unsqueeze(0)
        lengths = validate_spectrum(spectrum, spectrum_lengths)
        encoded = self.encode(spectrum, lengths)
        should_sample = self.training if sample_posterior is None else sample_posterior
        if posterior_sample_layout != "contiguous_bdt":
            raise ValueError("posterior_sample_layout must be contiguous_bdt")
        latents = (
            encoded.posterior.sample(generator, contiguous_source_layout=True)
            if should_sample
            else encoded.posterior.mode()
        )
        reconstruction = self.decode(
            latents,
            spectrum_frames=spectrum.shape[-1],
            latent_lengths=encoded.latent_lengths,
            spectrum_lengths=lengths,
        )
        spectrum_mask = lengths_to_mask(
            lengths, spectrum.shape[-1], device=spectrum.device
        )
        return SpecVAEOutput(
            reconstruction=reconstruction,
            posterior=encoded.posterior,
            latents=latents,
            spectrum_lengths=lengths,
            spectrum_mask=spectrum_mask,
            latent_lengths=encoded.latent_lengths,
            latent_mask=encoded.latent_mask,
        )


SpecVAEModel = SpecVAE


def audit_spec_vae_parameter_count(model: SpecVAE) -> dict[str, Any]:
    def count(module: nn.Module) -> int:
        return sum(parameter.numel() for parameter in module.parameters())

    encoder = count(model.encoder)
    decoder = count(model.decoder)
    total = count(model)
    return {
        "revision": model.config.revision,
        "total": total,
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "encoder": encoder,
        "decoder": decoder,
        "other": total - encoder - decoder,
    }


def nonfinite_spec_vae_parameter_names(model: SpecVAE) -> list[str]:
    bad = [
        name
        for name, value in model.state_dict().items()
        if isinstance(value, Tensor)
        and (value.is_floating_point() or value.is_complex())
        and not torch.isfinite(value).all()
    ]
    weighted_types = (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)
    for name, module in model.named_modules():
        if isinstance(module, weighted_types) and not torch.isfinite(module.weight).all():
            bad.append(f"{name}.effective_weight")
    return sorted(set(bad))
