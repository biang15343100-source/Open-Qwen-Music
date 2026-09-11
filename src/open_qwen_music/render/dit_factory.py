from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn

from .conditioning import RenderConditioner, build_render_conditioner_from_config
from .dit import DiTConfig
from .sa3 import build_sa3_render_dit


def build_render_dit_components_from_config(
    config: Mapping[str, Any],
    *,
    encoder: nn.Module | None = None,
    text_cache_revision_override: str | None = None,
) -> tuple[nn.Module, RenderConditioner]:

    if not isinstance(config, Mapping):
        raise TypeError("Renderer configuration must be a mapping")
    model_mapping = config.get("model")
    condition_mapping = config.get("conditioning")
    revisions = config.get("revisions")
    if not isinstance(model_mapping, Mapping):
        raise TypeError("Renderer configuration is missing the model section")
    if not isinstance(condition_mapping, Mapping):
        raise TypeError("Renderer configuration is missing the conditioning section")
    if not isinstance(revisions, Mapping):
        raise TypeError("Renderer configuration is missing the revisions section")

    model_config = DiTConfig.from_mapping(model_mapping)
    text_cache_revision = (
        text_cache_revision_override
        if text_cache_revision_override is not None
        else revisions.get("text_cache_revision")
    )
    revision_values = {
        "text_encoder_revision": revisions.get("text_encoder_revision"),
        "text_tokenizer_revision": revisions.get("text_tokenizer_revision"),
        "text_cache_revision": text_cache_revision,
        "semantic_tokenizer_revision": revisions.get("tokenizer_revision"),
    }
    invalid_revisions = {
        name: value
        for name, value in revision_values.items()
        if not isinstance(value, str) or not value
    }
    if invalid_revisions:
        raise ValueError(
            "Renderer configuration is missing revisions: "
            f"{sorted(invalid_revisions)}"
        )

    conditioner = build_render_conditioner_from_config(
        hidden_size=model_config.hidden_size,
        condition_config=condition_mapping,
        text_encoder_revision=revision_values["text_encoder_revision"],
        text_tokenizer_revision=revision_values["text_tokenizer_revision"],
        text_cache_revision=revision_values["text_cache_revision"],
        semantic_tokenizer_revision=revision_values["semantic_tokenizer_revision"],
        encoder=encoder,
    )
    architecture = config.get("architecture")
    if not isinstance(architecture, Mapping):
        raise TypeError("Renderer configuration is missing the architecture section")
    model = build_sa3_render_dit(model_config, architecture)
    conditioner.initialize_semantic_projection_from_latent(model.latent_projection)
    return model, conditioner
