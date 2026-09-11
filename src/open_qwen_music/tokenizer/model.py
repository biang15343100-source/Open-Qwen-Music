
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.func import functional_call

from .bestrq import (
    BESTRQ_CONTRACT_DEFAULTS,
    BESTRQ_CONTRACT_PRE_20260804_DEFAULTS,
    BestRQHead,
    BestRQTarget,
    apply_feature_mask,
    apply_waveform_mask,
    downsample_mask_100_to_25,
    erode_loss_mask,
    load_whitening_stats,
    make_span_mask,
    make_waveform_span_mask,
    relative_waveform_noise_std,
    resolve_bestrq_contract,
    waveform_mask_to_feature_mask,
)
from .conformer import ConformerEncoder, SinusoidalPositionalEncoding
from .contracts import CODEBOOK_SIZE, FRAME_RATE, SAMPLE_RATE, validate_contract
from .features import LogMelFrontend, _mel_filter_cpu, lengths_to_mask
from .frontend import (
    CAUSALITY_FIELDS,
    CAUSAL_SEMANTICS_DEFAULTS,
    POSITION_ENCODING_DEFAULTS,
    SUBSAMPLING_CONTRACT_DEFAULTS,
    SUBSAMPLING_CONTRACT_PRE_20260801_DEFAULTS,
    build_subsampling_frontend,
    resolve_causality,
    resolve_subsampling_contract,
)
from .heads import MultiTaskHeads, distributed_weighted_mean, masked_l1
from .quantizer import build_quantizer, distributed_quantizer_diversity


def _guard_distributed_quantizer_diversity(diversity_beta: float) -> None:
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1 and float(diversity_beta) != 0.0:


        raise RuntimeError(
            "distributed_quantizer_frame_mean cannot be used with diversity_beta != 0; "
            "use the global-moment collective"
        )


def distributed_quantizer_frame_mean(
    local_mean: torch.Tensor,
    local_frames: torch.Tensor,
    *,
    diversity_beta: float,
) -> torch.Tensor:

    _guard_distributed_quantizer_diversity(diversity_beta)

    weight_sum = local_frames.to(
        device=local_mean.device, dtype=local_mean.dtype
    )
    return distributed_weighted_mean(local_mean * weight_sum, weight_sum)


@dataclass
class SemanticTokenBatch:
    token_ids: torch.Tensor
    frame_mask: torch.Tensor
    frame_rate: float = FRAME_RATE
    codebook_size: int = CODEBOOK_SIZE
    tokenizer_revision: str = "unfrozen"


def _require_causal_model(causal: bool, entry: str) -> None:

    if not causal:
        raise ValueError(
            f"{entry} requires a causal model, but model.causal=false. "
            "Tokens would see future frames and cannot be used for semantic-token "
            "output or quantizer codebook initialization."
        )


class MusicTokenizer(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.stage = int(config["stage"])
        model_config = config["model"]
        causality = resolve_causality(model_config)
        self.attention_causal = causality["attention_causal"]
        self.frontend_causal = causality["frontend_causal"]
        self.conformer_conv_causal = causality["conformer_conv_causal"]

        self.causal = self.attention_causal
        self.strictly_causal = all(causality.values())


        causal_semantics = resolve_subsampling_contract(
            config["model"], CAUSAL_SEMANTICS_DEFAULTS
        )
        self.bestrq_causal_window = bool(causal_semantics["bestrq_causal_window"])
        self.bestrq_erode_loss_mask = bool(causal_semantics["bestrq_erode_loss_mask"])
        self.causal_heads = bool(causal_semantics["causal_heads"])
        feature_config = dict(config["features"])
        self.feature_stats_pending = any(
            feature_config.get(name) is None for name in ("mean", "std")
        )
        if self.feature_stats_pending:
            experiment = config.get("experiment") or {}
            if not (
                experiment.get("readiness") in {"BLOCKED", "INITIALIZATION_ONLY"}
                and bool(experiment.get("long_train_blocked"))
            ):
                raise ValueError(
                    "features.mean and features.std may be omitted only when the "
                    "configuration explicitly blocks long training"
                )

            if feature_config.get("mean") is None:
                feature_config["mean"] = 0.0
            if feature_config.get("std") is None:
                feature_config["std"] = 1.0
        validate_contract(
            int(feature_config["sample_rate"]),
            float(config["semantic_contract"]["frame_rate"]),
            int(config["semantic_contract"]["codebook_size"]),
        )
        self.feature_extractor = LogMelFrontend(**feature_config)
        if self.feature_extractor.mel_filter.is_meta:


            _mel_filter_cpu.cache_clear()
        dim = int(model_config["dim"])
        self.subsampling = build_subsampling_frontend(
            model_config,
            input_dim=int(feature_config["n_mels"]),
            model_dim=dim,
        )
        position_encoding = str(
            resolve_subsampling_contract(model_config, POSITION_ENCODING_DEFAULTS)[
                "position_encoding"
            ]
        )


        self.position_encoding = (
            SinusoidalPositionalEncoding(dim)
            if position_encoding == "sinusoidal"
            else nn.Identity()
        )
        self.encoder = ConformerEncoder(
            dim=dim,
            num_layers=int(model_config["num_layers"]),
            num_heads=int(model_config["num_heads"]),
            ffn_dim=int(model_config["ffn_dim"]),
            conv_kernel=int(model_config["conv_kernel"]),
            dropout=float(model_config["dropout"]),
            position_encoding=position_encoding,
        )
        self.insertion_layer = int(model_config["quantizer_insertion_layer"])
        if not 1 <= self.insertion_layer <= int(model_config["num_layers"]):
            raise ValueError("quantizer_insertion_layer must be in [1, num_layers - 1]")

        bestrq = config["bestrq"]
        bestrq_contract = resolve_bestrq_contract(bestrq, BESTRQ_CONTRACT_DEFAULTS)
        self.bestrq_contract = bestrq_contract
        if str(bestrq_contract["mask_noise_mode"]) not in ("absolute", "relative_rms"):
            raise ValueError(
                "bestrq.mask_noise_mode must be 'absolute' or 'relative_rms'"
            )
        if str(bestrq_contract["loss_mask_support"]) not in ("nominal", "full_stft"):
            raise ValueError(
                "bestrq.loss_mask_support must be 'nominal' or 'full_stft'"
            )
        aggregation = str(bestrq_contract["target_aggregation"])
        whitening_mean = whitening_matrix = None


        if bestrq_contract["whitening"] and self.stage in (1, 2):


            whitening_mean, whitening_matrix = load_whitening_stats(
                bestrq.get("whitening_stats"),
                input_dim=int(feature_config["n_mels"]),
                aggregation=aggregation,
            )
        self.bestrq_target = BestRQTarget(
            input_dim=int(feature_config["n_mels"]),
            projection_dim=int(bestrq["projection_dim"]),
            codebook_size=int(bestrq["target_codebook_size"]),
            local_window=int(bestrq_contract["local_window"]),
            seed=int(bestrq["seed"]),
            distance_chunk_size=int(bestrq.get("distance_chunk_size", 2048)),
            causal=self.causal and self.bestrq_causal_window,
            aggregation=aggregation,
            projection_init=str(bestrq_contract["projection_init"]),
            whitening_mean=whitening_mean,
            whitening_matrix=whitening_matrix,
        )
        self.bestrq_head = BestRQHead(dim, int(bestrq["target_codebook_size"]))

        heads = config["heads"]
        self.heads = MultiTaskHeads(
            model_dim=dim,
            ctc_vocab_size=int(heads["ctc_vocab_size"]),
            mel_bins=int(feature_config["n_mels"]),
            chroma_bins=12,
            convnext_blocks=int(heads.get("convnext_blocks", 2)),
            ctc_blank_id=int(heads.get("ctc_blank_id", 0)),
            ctc_normalize_by_target_length=bool(
                heads.get("ctc_normalize_by_target_length", True)
            ),
            mel_loss_mode=str(heads.get("mel_loss_mode", "signed_l1")),
            chroma_loss_mode=str(
                heads.get("chroma_loss_mode", "signed_l1")
            ),
        )
        self.quantizer = build_quantizer(
            {
                **config["quantizer"],
                "codebook_size": int(config["semantic_contract"]["codebook_size"]),
            },
            input_dim=dim,
        )
        self._configure_stage_trainability()

    def _mask_noise_std(
        self, waveform: torch.Tensor, waveform_lengths: torch.Tensor
    ) -> float | torch.Tensor:
        if str(self.bestrq_contract["mask_noise_mode"]) == "relative_rms":
            return relative_waveform_noise_std(
                waveform,
                waveform_lengths,
                noise_db=float(self.config["bestrq"].get("mask_noise_db", -20.0)),
                minimum=float(
                    self.config["bestrq"].get("mask_noise_min_std", 1e-5)
                ),
                maximum=float(
                    self.config["bestrq"].get("mask_noise_max_std", 0.2)
                ),
            )
        return float(self.config["bestrq"]["noise_std"])

    def validate_checkpoint_contract(
        self, checkpoint_config: dict[str, Any], *, source: str = "checkpoint"
    ) -> None:

        source_model = (checkpoint_config or {}).get("model")
        if not source_model:
            return
        target = resolve_subsampling_contract(
            self.config["model"], SUBSAMPLING_CONTRACT_DEFAULTS
        )
        incoming = resolve_subsampling_contract(
            source_model, SUBSAMPLING_CONTRACT_PRE_20260801_DEFAULTS
        )
        mismatches = {
            key: (target[key], incoming[key])
            for key in SUBSAMPLING_CONTRACT_DEFAULTS
            if target[key] != incoming[key]
        }
        if mismatches:
            raise RuntimeError(
                f"The downsampling frontend contract differs from {source} "
                "(configured, checkpoint): "
                f"{mismatches}"
            )


        target_model = self.config["model"]
        explicit_causality = any(
            key in target_model or key in source_model for key in CAUSALITY_FIELDS
        )
        if explicit_causality:
            target_causality = resolve_causality(target_model)
            incoming_causality = resolve_causality(source_model)
            source_stage = int((checkpoint_config or {}).get("stage", 0) or 0)
            diagnostic_probe_override = bool(
                self.config.get("probe", {}).get(
                    "allow_stage1_attention_causal_override", False
                )
            )
            allowed_attention_change = source_stage == 1 and (
                self.stage == 2
                or (self.stage == 3 and diagnostic_probe_override)
            )
            checked_fields = (
                ("frontend_causal", "conformer_conv_causal")
                if allowed_attention_change
                else CAUSALITY_FIELDS
            )
            causality_mismatches = {
                key: (target_causality[key], incoming_causality[key])
                for key in checked_fields
                if target_causality[key] != incoming_causality[key]
            }
            if causality_mismatches:
                raise RuntimeError(
                    f"The attention/convolution causality contract differs from {source} "
                    "(configured, checkpoint): "
                    f"{causality_mismatches}"
                )

        source_stage = int((checkpoint_config or {}).get("stage", 0) or 0)
        if source_stage >= 3 and self.stage >= 3:
            defaults = {
                "mel_target_mode": "frontend",
                "mel_loss_mode": "signed_l1",
                "chroma_loss_mode": "signed_l1",
            }
            source_heads = checkpoint_config.get("heads") or {}
            target_heads = self.config.get("heads") or {}
            head_mismatches = {
                key: (
                    target_heads.get(key, default),
                    source_heads.get(key, default),
                )
                for key, default in defaults.items()
                if target_heads.get(key, default)
                != source_heads.get(key, default)
            }
            source_weights = checkpoint_config.get("loss_weights") or {}
            reconstruction_disabled = (
                float(source_weights.get("mel", 0.0)) == 0.0
                and float(source_weights.get("chroma", 0.0)) == 0.0
            )
            if head_mismatches and not reconstruction_disabled:
                raise RuntimeError(
                    f"The mel/chroma target-loss contract differs from {source}: "
                    f"{head_mismatches}"
                )


        source_bestrq = (checkpoint_config or {}).get("bestrq")
        if not source_bestrq:
            return
        target_bestrq = resolve_bestrq_contract(
            self.config["bestrq"], BESTRQ_CONTRACT_DEFAULTS
        )
        incoming_bestrq = resolve_bestrq_contract(
            source_bestrq, BESTRQ_CONTRACT_PRE_20260804_DEFAULTS
        )
        bestrq_mismatches = {
            key: (target_bestrq[key], incoming_bestrq[key])
            for key in BESTRQ_CONTRACT_DEFAULTS
            if target_bestrq[key] != incoming_bestrq[key]
        }
        if bestrq_mismatches:
            raise RuntimeError(
                f"The BestRQ target contract differs from {source} "
                "(configured, checkpoint): "
                f"{bestrq_mismatches}"
            )

    def _configure_stage_trainability(self) -> None:

        if self.stage in (1, 2):
            disabled = (self.heads, self.quantizer)
        elif self.stage == 3:
            disabled = (self.bestrq_head, self.quantizer)
        else:
            disabled = (self.bestrq_head,)
        for module in disabled:
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def _encode_features(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        *,
        quantize: bool,
        gate_alpha: float,
        quantizer_sample_weights: torch.Tensor | None = None,
        bottleneck_mode: str = "quantized",
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        dict,
        torch.Tensor,
    ]:
        if bottleneck_mode not in {
            "identity",
            "radius_projected",
            "projected",
            "quantized",
        }:
            raise ValueError(
                "bottleneck_mode must be identity, radius_projected, projected, "
                f"or quantized; received {bottleneck_mode!r}"
            )
        feature_mask = lengths_to_mask(feature_lengths, features.shape[1])
        hidden = self.subsampling(
            features, causal=self.frontend_causal, mask=feature_mask
        )
        hidden = self.position_encoding(hidden)
        frame_lengths = self.subsampling.output_lengths(feature_lengths).clamp_max(
            hidden.shape[1]
        )
        frame_mask = lengths_to_mask(frame_lengths, hidden.shape[1])
        hidden = self.encoder.forward_range(
            hidden,
            frame_mask,
            start=0,
            end=self.insertion_layer,
            attention_causal=self.attention_causal,
            convolution_causal=self.conformer_conv_causal,
        )
        token_ids = None
        quantizer_loss = hidden.sum() * 0.0
        quantizer_metrics: dict[str, torch.Tensor] = {}
        if quantize:
            quantizer_loss_mask = frame_mask
            quantizer_frame_weights = None
            if quantizer_sample_weights is not None:
                if quantizer_sample_weights.shape != (features.shape[0],):
                    raise ValueError(
                        "quantizer_sample_weights must have shape [B]; received "
                        f"{tuple(quantizer_sample_weights.shape)}"
                    )
                sample_weights = quantizer_sample_weights.to(
                    device=hidden.device, dtype=hidden.dtype
                )
                quantizer_frame_weights = (
                    frame_mask.to(hidden.dtype) * sample_weights.unsqueeze(1)
                )
                quantizer_loss_mask = frame_mask & (
                    sample_weights > 0.0
                ).unsqueeze(1)
            if bottleneck_mode == "radius_projected":
                hidden = self.quantizer.project_radius_preserving(hidden)
            elif bottleneck_mode == "projected":
                hidden = self.quantizer.project_continuous(hidden)
            else:
                quantized, token_ids, quantizer_loss, quantizer_metrics = self.quantizer(
                    hidden,
                    mask=frame_mask,
                    loss_mask=quantizer_loss_mask,
                    loss_weights=quantizer_frame_weights,
                )
                hidden = hidden + float(gate_alpha) * (quantized - hidden)
        bottleneck_hidden = hidden
        hidden = self.encoder.forward_range(
            hidden,
            frame_mask,
            start=self.insertion_layer,
            end=None,
            attention_causal=self.attention_causal,
            convolution_causal=self.conformer_conv_causal,
        )
        return (
            hidden,
            frame_mask,
            token_ids,
            quantizer_loss,
            quantizer_metrics,
            bottleneck_hidden,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        gate_alpha: float = 1.0,
        loss_weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        waveform = batch["waveform"]
        waveform_lengths = batch["waveform_num_samples"]
        clean_features = self.feature_extractor(waveform)
        feature_lengths = self.feature_extractor.lengths(waveform_lengths).clamp_max(
            clean_features.shape[1]
        )

        if self.stage in (1, 2):
            probability = float(self.config["bestrq"]["mask_probability"])
            full_support = (
                str(self.bestrq_contract["loss_mask_support"]) == "full_stft"
            )
            if self.config["bestrq"].get("mask_domain", "feature") == "waveform":
                waveform_mask = make_waveform_span_mask(
                    waveform_lengths,
                    waveform.shape[1],
                    span_samples=round(
                        float(self.config["bestrq"].get("mask_span_sec", 0.4))
                        * SAMPLE_RATE
                    ),
                    probability=probability,
                )
                features = self.feature_extractor(
                    apply_waveform_mask(
                        waveform,
                        waveform_mask,
                        self._mask_noise_std(waveform, waveform_lengths),
                    )
                )
                mask = waveform_mask_to_feature_mask(
                    waveform_mask,
                    hop_length=int(self.feature_extractor.hop_length),
                    target_length=clean_features.shape[1],
                    n_fft=int(self.feature_extractor.n_fft),
                    require_full_support=full_support,
                )
            else:
                mask = make_span_mask(
                    feature_lengths,
                    clean_features.shape[1],
                    span_frames=int(self.config["bestrq"]["mask_span_frames"]),
                    probability=probability,
                )
                features = apply_feature_mask(
                    clean_features,
                    mask,
                    float(self.config["bestrq"]["noise_std"]),
                )
            hidden, frame_mask, _, _, _, _ = self._encode_features(
                features, feature_lengths, quantize=False, gate_alpha=0.0
            )
            targets = self.bestrq_target(
                clean_features,
                lengths_to_mask(feature_lengths, clean_features.shape[1]),
            )
            targets = targets[:, : hidden.shape[1]]
            loss_mask = downsample_mask_100_to_25(
                mask, hidden.shape[1], require_all=full_support
            ) & frame_mask
            if self.bestrq_erode_loss_mask:
                loss_mask = erode_loss_mask(
                    loss_mask,
                    self.bestrq_target.local_window,
                    causal=self.bestrq_target.causal,
                )


            decode_failed = batch.get("decode_failed")
            if decode_failed is not None and bool(decode_failed.any()):
                loss_mask = loss_mask & ~decode_failed.to(loss_mask.device).view(-1, 1)
            local_loss, metrics = self.bestrq_head(hidden, targets, loss_mask)




            if self.training:
                active_frames = metrics["bestrq_active_frames"].to(
                    device=local_loss.device, dtype=local_loss.dtype
                )
                loss = distributed_weighted_mean(
                    local_loss * active_frames,
                    active_frames,
                )
                metrics["bestrq_local_loss_mean"] = metrics["bestrq_loss"]


                metrics["bestrq_loss"] = loss.detach()
            else:


                loss = local_loss
            if bool(batch.get("return_bestrq_target_counts", False)):
                active_targets = targets[loss_mask & (targets >= 0)]
                metrics["bestrq_target_counts"] = torch.bincount(
                    active_targets,
                    minlength=int(self.config["bestrq"]["target_codebook_size"]),
                ).detach()
            metrics["decode_failed_samples"] = (
                loss.new_zeros(())
                if decode_failed is None
                else decode_failed.sum().to(loss.dtype)
            )
            metrics["loss"] = loss.detach()
            return loss, metrics

        mask_probability = float(self.config["bestrq"].get("mask_probability", 0.0))
        training_features = clean_features
        if self.training and mask_probability > 0.0:
            if self.config["bestrq"].get("mask_domain", "feature") == "waveform":
                waveform_mask = make_waveform_span_mask(
                    waveform_lengths,
                    waveform.shape[1],
                    span_samples=round(
                        float(self.config["bestrq"].get("mask_span_sec", 0.4))
                        * SAMPLE_RATE
                    ),
                    probability=mask_probability,
                )
                training_features = self.feature_extractor(
                    apply_waveform_mask(
                        waveform,
                        waveform_mask,
                        self._mask_noise_std(waveform, waveform_lengths),
                    )
                )
            else:
                mask = make_span_mask(
                    feature_lengths,
                    clean_features.shape[1],
                    span_frames=int(self.config["bestrq"]["mask_span_frames"]),
                    probability=mask_probability,
                )
                training_features = apply_feature_mask(
                    clean_features,
                    mask,
                    float(self.config["bestrq"]["noise_std"]),
                )
        decode_failed = batch.get("decode_failed")
        quantizer_sample_weights = None
        if self.stage == 4:
            batch_size = training_features.shape[0]
            failed = (
                torch.zeros(batch_size, dtype=torch.bool, device=feature_lengths.device)
                if decode_failed is None
                else decode_failed.to(
                    device=feature_lengths.device, dtype=torch.bool
                )
            )
            declared_vq = batch.get("vq_enabled")
            vq_enabled = (
                torch.ones(batch_size, dtype=torch.bool, device=feature_lengths.device)
                if declared_vq is None
                else declared_vq.to(
                    device=feature_lengths.device, dtype=torch.bool
                )
            )
            declared_vq_weights = batch.get(
                "vq_sample_weights",
                batch.get("sample_weights"),
            )
            vq_weights = (
                torch.ones(
                    batch_size,
                    dtype=training_features.dtype,
                    device=feature_lengths.device,
                )
                if declared_vq_weights is None
                else declared_vq_weights.to(
                    device=feature_lengths.device,
                    dtype=training_features.dtype,
                )
            )
            if vq_weights.shape != (batch_size,):
                raise ValueError(
                    "vq_sample_weights must have shape [B]; received "
                    f"{tuple(vq_weights.shape)}"
                )
            quantizer_sample_weights = vq_weights * (
                ~failed & vq_enabled
            ).to(vq_weights.dtype)
        (
            hidden,
            frame_mask,
            _,
            quantizer_loss,
            quantizer_metrics,
            bottleneck_hidden,
        ) = self._encode_features(
            training_features,
            feature_lengths,
            quantize=self.stage == 4,
            gate_alpha=gate_alpha,
            quantizer_sample_weights=quantizer_sample_weights,
        )
        if self.stage == 4:
            local_quantizer_loss = quantizer_loss
            local_vq_frames = quantizer_metrics["codebook_batch_frames"]
            local_vq_weight_sum = quantizer_metrics["codebook_batch_weight"]
            diversity_beta = float(
                getattr(self.quantizer, "diversity_beta", 0.0)
            )
            local_base_loss = local_quantizer_loss
            if diversity_beta != 0.0:
                private_names = (
                    "_diversity_local_loss",
                    "_diversity_weighted_sum",
                    "_diversity_weighted_second_moment",
                    "_diversity_weight_sum",
                )
                missing = [
                    name for name in private_names if name not in quantizer_metrics
                ]
                if missing:
                    raise RuntimeError(
                        "diversity_beta is non-zero, but the quantizer did not return "
                        "complete global-moment statistics: "
                        f"{missing}"
                    )
                local_diversity = quantizer_metrics.pop(
                    "_diversity_local_loss"
                )
                diversity_sum = quantizer_metrics.pop(
                    "_diversity_weighted_sum"
                )
                diversity_second = quantizer_metrics.pop(
                    "_diversity_weighted_second_moment"
                )
                diversity_weight = quantizer_metrics.pop(
                    "_diversity_weight_sum"
                )
                local_base_loss = (
                    local_quantizer_loss - diversity_beta * local_diversity
                )
            quantizer_loss = distributed_quantizer_frame_mean(
                local_base_loss,
                local_vq_weight_sum,
                diversity_beta=0.0,
            )
            if diversity_beta != 0.0:
                global_diversity = distributed_quantizer_diversity(
                    diversity_sum,
                    diversity_second,
                    diversity_weight,
                    distributed=self.training,
                )
                quantizer_loss = (
                    quantizer_loss + diversity_beta * global_diversity
                )
                quantizer_metrics["diversity_loss"] = (
                    global_diversity.detach()
                )
            quantizer_metrics["quantizer_local_loss_mean"] = (
                local_quantizer_loss.detach()
            )

            quantizer_metrics["quantizer_local_frames"] = (
                local_vq_frames.detach()
            )
            quantizer_metrics["quantizer_local_weight_sum"] = (
                local_vq_weight_sum.detach()
            )
            quantizer_metrics["quantizer_loss"] = quantizer_loss.detach()
        frame_lengths = frame_mask.sum(dim=1)
        component_losses, metrics = self.heads(
            hidden,
            frame_lengths,
            batch,
            causal=self.attention_causal and self.causal_heads,
        )
        weights = loss_weights or self.config["loss_weights"]
        loss = (
            float(weights["ctc"]) * component_losses["ctc"]
            + float(weights.get("ctc_alignment", 0.0))
            * component_losses["ctc_alignment"]
            + float(weights["mel"]) * component_losses["mel"]
            + float(weights["chroma"]) * component_losses["chroma"]
        )
        if self.stage == 4:
            loss = loss + float(weights["quantizer"]) * quantizer_loss
            metrics.update(quantizer_metrics)
            metrics["diversity_beta"] = loss.new_tensor(
                float(getattr(self.quantizer, "diversity_beta", 0.0))
            )
            local_mel_weight = float(weights.get("local_mel", 0.0))
            if local_mel_weight > 0.0:


                detached_state = {
                    name: value.detach()
                    for name, value in self.heads.mel.named_parameters()
                }
                detached_state.update(
                    {
                        name: value.detach()
                        for name, value in self.heads.mel.named_buffers()
                    }
                )
                local_mel_prediction = functional_call(
                    self.heads.mel,
                    detached_state,
                    (
                        bottleneck_hidden,
                        batch["mel_target"].shape[1],
                    ),
                    {
                        "mask": frame_mask,
                        "causal": self.attention_causal and self.causal_heads,
                    },
                )
                local_mel_mask = (
                    batch["mel_target_mask"]
                    & batch["mel_enabled"].unsqueeze(1)
                )
                local_mel_loss = masked_l1(
                    local_mel_prediction,
                    batch["mel_target"],
                    local_mel_mask,
                    sample_weights=batch.get(
                        "mel_sample_weights",
                        batch.get("sample_weights"),
                    ),
                )
                loss = loss + local_mel_weight * local_mel_loss
                metrics["local_mel_loss"] = local_mel_loss.detach()
            else:
                metrics["local_mel_loss"] = loss.new_zeros(())
            metrics["local_mel_weight"] = loss.new_tensor(local_mel_weight)
        metrics["loss"] = loss.detach()
        metrics["gate_alpha"] = loss.new_tensor(float(gate_alpha))
        metrics["ctc_weight"] = loss.new_tensor(float(weights["ctc"]))
        metrics["ctc_alignment_weight"] = loss.new_tensor(
            float(weights.get("ctc_alignment", 0.0))
        )
        metrics["decode_failed_samples"] = (
            loss.new_zeros(())
            if decode_failed is None
            else decode_failed.sum().to(loss.dtype)
        )
        return loss, metrics

    @torch.no_grad()
    def predict_multitask_details(
        self,
        batch: dict[str, torch.Tensor],
        *,
        gate_alpha: float = 1.0,
        bottleneck_mode: str | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:

        if self.stage < 3:
            raise ValueError("predict_multitask applies only to Stage 3 or 4")
        resolved_mode = bottleneck_mode or (
            "quantized" if self.stage == 4 else "identity"
        )
        if resolved_mode not in {
            "identity",
            "radius_projected",
            "projected",
            "quantized",
        }:
            raise ValueError(
                "bottleneck_mode must be identity, radius_projected, projected, "
                f"or quantized; received {resolved_mode!r}"
            )
        if self.stage != 4 and resolved_mode != "identity":
            raise ValueError(
                f"Stage{self.stage} has no trained VQ; bottleneck_mode must be identity"
            )
        waveform = batch["waveform"]
        waveform_lengths = batch["waveform_num_samples"]
        features = self.feature_extractor(waveform)
        feature_lengths = self.feature_extractor.lengths(waveform_lengths).clamp_max(
            features.shape[1]
        )
        hidden, frame_mask, token_ids, _, _, _ = self._encode_features(
            features,
            feature_lengths,
            quantize=self.stage == 4 and resolved_mode != "identity",
            gate_alpha=gate_alpha,
            bottleneck_mode=resolved_mode,
        )
        predictions = self.heads.predict(
            hidden,
            mel_target_length=batch["mel_target"].shape[1],
            chroma_target_length=batch["chroma_target"].shape[1],
            mask=frame_mask,
            causal=self.attention_causal and self.causal_heads,
        )
        return predictions, frame_mask.sum(dim=1), token_ids

    @torch.no_grad()
    def predict_multitask(
        self,
        batch: dict[str, torch.Tensor],
        *,
        gate_alpha: float = 1.0,
        bottleneck_mode: str | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        predictions, frame_lengths, _ = self.predict_multitask_details(
            batch,
            gate_alpha=gate_alpha,
            bottleneck_mode=bottleneck_mode,
        )
        return predictions, frame_lengths

    @torch.no_grad()
    def encode_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        attention_mask: torch.Tensor | None = None,
        tokenizer_revision: str = "unfrozen",
    ) -> SemanticTokenBatch:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"encode_audio requires {SAMPLE_RATE} Hz; received {sample_rate}")
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if attention_mask is None:
            lengths = torch.full(
                (waveform.shape[0],), waveform.shape[1], device=waveform.device, dtype=torch.long
            )
        else:
            lengths = attention_mask.long().sum(dim=1)
        _require_causal_model(self.strictly_causal, "encode_audio")
        features = self.feature_extractor(waveform)
        feature_lengths = self.feature_extractor.lengths(lengths).clamp_max(features.shape[1])
        hidden = self.subsampling(
            features,
            causal=self.frontend_causal,
            mask=lengths_to_mask(feature_lengths, features.shape[1]),
        )
        hidden = self.position_encoding(hidden)
        frame_lengths = self.subsampling.output_lengths(feature_lengths).clamp_max(hidden.shape[1])
        frame_mask = lengths_to_mask(frame_lengths, hidden.shape[1])
        hidden = self.encoder.forward_range(
            hidden,
            frame_mask,
            start=0,
            end=self.insertion_layer,
            attention_causal=self.attention_causal,
            convolution_causal=self.conformer_conv_causal,
        )
        _, token_ids, _, _ = self.quantizer(hidden, mask=frame_mask)
        return SemanticTokenBatch(
            token_ids=token_ids.masked_fill(~frame_mask, 0),
            frame_mask=frame_mask,
            tokenizer_revision=tokenizer_revision,
        )

    @torch.no_grad()
    def extract_insertion_embeddings(
        self,
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(waveform)
        feature_lengths = self.feature_extractor.lengths(waveform_lengths).clamp_max(
            features.shape[1]
        )
        hidden = self.position_encoding(
            self.subsampling(
                features,
                causal=self.frontend_causal,
                mask=lengths_to_mask(feature_lengths, features.shape[1]),
            )
        )
        frame_lengths = self.subsampling.output_lengths(feature_lengths).clamp_max(
            hidden.shape[1]
        )
        frame_mask = lengths_to_mask(frame_lengths, hidden.shape[1])
        hidden = self.encoder.forward_range(
            hidden,
            frame_mask,
            start=0,
            end=self.insertion_layer,
            attention_causal=self.attention_causal,
            convolution_causal=self.conformer_conv_causal,
        )
        return hidden, frame_mask

    @torch.no_grad()
    def extract_quantizer_inputs(
        self,
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        _require_causal_model(self.strictly_causal, "extract_quantizer_inputs")
        return self.extract_insertion_embeddings(waveform, waveform_lengths)

    @torch.no_grad()
    def extract_quantized_embeddings(
        self,
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.stage != 4:
            raise ValueError("extract_quantized_embeddings requires a Stage 4 model")
        hidden, frame_mask = self.extract_quantizer_inputs(
            waveform, waveform_lengths
        )
        quantized, token_ids, _, _ = self.quantizer(hidden, mask=frame_mask)
        return quantized, frame_mask, token_ids
