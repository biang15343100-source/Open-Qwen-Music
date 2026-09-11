"""Configuration contract for the released Renderer backbone."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .contracts import LATENT_DIM, LATENT_FRAME_HZ


MAX_RENDER_FRAMES = 9_000
ADALN_CONDITIONING_MODES = ("timestep", "timestep_plus_global_loudness")


@dataclass(frozen=True)
class DiTConfig:
    """Settings that connect the Renderer to the training pipeline.

    Transformer topology lives in the checkpoint-compatible ``architecture``
    section so training and inference share one architecture source of truth.
    """

    latent_dim: int = LATENT_DIM
    latent_frame_hz: int = LATENT_FRAME_HZ
    hidden_size: int = 1_024
    context_dim: int = 1_024
    max_frames: int = MAX_RENDER_FRAMES
    adaln_conditioning: str = "timestep_plus_global_loudness"
    activation_checkpointing: bool = False

    def validate(self) -> None:
        if self.latent_dim != LATENT_DIM:
            raise ValueError(f"latent_dim must be {LATENT_DIM}")
        if self.latent_frame_hz != LATENT_FRAME_HZ:
            raise ValueError(f"latent_frame_hz must be {LATENT_FRAME_HZ}")
        for name in ("hidden_size", "context_dim", "max_frames"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.context_dim != self.hidden_size:
            raise ValueError("context_dim must equal hidden_size")
        if self.adaln_conditioning not in ADALN_CONDITIONING_MODES:
            raise ValueError(
                f"adaln_conditioning must be one of {ADALN_CONDITIONING_MODES}"
            )
        if not isinstance(self.activation_checkpointing, bool):
            raise TypeError("activation_checkpointing must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DiTConfig":
        if not isinstance(value, Mapping):
            raise TypeError("Renderer model configuration must be a mapping")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown Renderer model settings: {sorted(unknown)}")
        config = cls(**dict(value))
        config.validate()
        return config


__all__ = ["DiTConfig", "MAX_RENDER_FRAMES"]
