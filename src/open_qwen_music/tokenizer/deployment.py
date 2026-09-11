
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .conformer import ConformerEncoder, SinusoidalPositionalEncoding
from .contracts import CODEBOOK_SIZE, FRAME_RATE, SAMPLE_RATE
from .features import LogMelFrontend, lengths_to_mask
from .frontend import (
    CAUSAL_PADDING,
    build_subsampling_frontend,
    resolve_causality,
)
from .model import SemanticTokenBatch
from .quantizer import build_quantizer


def _validate_artifact_contract(model_config: dict[str, Any]) -> None:

    padding = model_config.get("causal_padding")
    if padding is None:
        raise ValueError(
            "The deployment artifact is missing model.causal_padding. "
            "Export it again with the current tokenizer code."
        )
    if str(padding).lower() != CAUSAL_PADDING:
        raise ValueError(
            f"The deployment artifact uses causal_padding={padding!r}; "
            f"the runtime requires {CAUSAL_PADDING!r}. Export the tokenizer again."
        )
    causality = resolve_causality(model_config)
    noncausal = [key for key, value in causality.items() if not value]
    if noncausal:
        raise ValueError(
            "The deployment graph requires causal attention, subsampling, and "
            f"Conformer convolutions; non-causal fields: {noncausal}."
        )
    if model_config.get("position_encoding") is None:
        raise ValueError(
            "The deployment artifact is missing model.position_encoding. "
            "Export it again so the positional encoding contract is explicit."
        )


class DeploymentMusicTokenizer(nn.Module):
    def __init__(self, artifact: dict[str, Any], revision: str = "unknown") -> None:
        super().__init__()
        self.revision = revision
        feature_config = artifact["features"]
        model_config = artifact["model"]
        _validate_artifact_contract(model_config)
        dim = int(model_config["dim"])
        insertion = int(model_config["quantizer_insertion_layer"])
        self.feature_extractor = LogMelFrontend(**feature_config)
        self.subsampling = build_subsampling_frontend(
            model_config,
            input_dim=int(feature_config["n_mels"]),
            model_dim=dim,
        )
        position_encoding = str(model_config.get("position_encoding", "rope"))
        self.position_encoding = (
            SinusoidalPositionalEncoding(dim)
            if position_encoding == "sinusoidal"
            else nn.Identity()
        )
        self.encoder = ConformerEncoder(
            dim=dim,
            num_layers=insertion,
            num_heads=int(model_config["num_heads"]),
            ffn_dim=int(model_config["ffn_dim"]),
            conv_kernel=int(model_config["conv_kernel"]),
            dropout=float(model_config["dropout"]),
            position_encoding=position_encoding,
        )
        self.quantizer = build_quantizer(
            {
                **artifact["quantizer"],
                "codebook_size": int(artifact["semantic_contract"]["codebook_size"]),
            },
            input_dim=dim,
        )
        self.load_state_dict(artifact["state_dict"], strict=True)


        self.eval()

    @classmethod
    def from_file(
        cls, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> "DeploymentMusicTokenizer":
        path = Path(path)
        artifact = torch.load(path, map_location=map_location, weights_only=False)
        revision = "unknown"
        sidecar = path.with_suffix(path.suffix + ".json")
        if sidecar.exists():
            import json

            revision = json.loads(sidecar.read_text(encoding="utf-8"))["tokenizer_revision"]
        return cls(artifact, revision=revision)

    @torch.no_grad()
    def encode_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> SemanticTokenBatch:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"deploy Tokenizer requires {SAMPLE_RATE} Hz")
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if attention_mask is None:
            lengths = torch.full(
                (waveform.shape[0],), waveform.shape[1], device=waveform.device, dtype=torch.long
            )
        else:
            lengths = attention_mask.long().sum(dim=1)
        features = self.feature_extractor(waveform)
        feature_lengths = self.feature_extractor.lengths(lengths).clamp_max(features.shape[1])
        hidden = self.position_encoding(
            self.subsampling(
                features,
                causal=True,
                mask=lengths_to_mask(feature_lengths, features.shape[1]),
            )
        )
        frame_lengths = self.subsampling.output_lengths(feature_lengths).clamp_max(hidden.shape[1])
        frame_mask = lengths_to_mask(frame_lengths, hidden.shape[1])
        hidden = self.encoder.forward_range(
            hidden, frame_mask, causal=True, start=0, end=None
        )
        _, token_ids, _, _ = self.quantizer(hidden, mask=frame_mask)
        return SemanticTokenBatch(
            token_ids=token_ids.masked_fill(~frame_mask, 0),
            frame_mask=frame_mask,
            frame_rate=FRAME_RATE,
            codebook_size=CODEBOOK_SIZE,
            tokenizer_revision=self.revision,
        )
