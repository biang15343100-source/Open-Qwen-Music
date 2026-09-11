
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from open_qwen_music.common.distributed import reduce_scalar_sum

from .features import lengths_to_mask
from .frontend import ConvNeXt1DBlock


def distributed_weighted_mean(
    local_numerator: torch.Tensor,
    local_denominator: torch.Tensor,
) -> torch.Tensor:

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size <= 1:
        return local_numerator / local_denominator.clamp_min(1e-8)
    global_denominator = reduce_scalar_sum(
        float(local_denominator.detach().item())
    )
    return (
        local_numerator
        * float(world_size)
        / max(global_denominator, 1e-8)
    )


class FramePredictionHead(nn.Module):

    def __init__(
        self,
        model_dim: int,
        output_dim: int,
        num_blocks: int = 2,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ConvNeXt1DBlock(model_dim, layer_scale_init=layer_scale_init)
                for _ in range(num_blocks)
            ]
        )
        self.output = nn.Linear(model_dim, output_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        target_length: int,
        mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        x = hidden.repeat_interleave(4, dim=1)
        if mask is not None:
            mask = mask.repeat_interleave(4, dim=1)
        for block in self.blocks:
            x = block(x, causal=causal, mask=mask)
        return self.output(x)[:, :target_length]


def masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    active = mask.unsqueeze(-1).expand_as(target)
    difference = (prediction - target).abs() * active
    per_sample_count = active.sum(dim=(1, 2)).clamp_min(1)
    per_sample = difference.sum(dim=(1, 2)) / per_sample_count
    enabled = mask.any(dim=1)
    if sample_weights is None:
        weights = enabled.to(per_sample.dtype)
    else:
        weights = sample_weights.to(per_sample.dtype) * enabled
    return distributed_weighted_mean(
        (per_sample * weights).sum(),
        weights.sum(),
    )


def spectral_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    min_target_norm: float = 1e-3,
    mode: str = "signed_l1",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode not in {"signed_l1", "nonnegative_log"}:
        raise ValueError(f"Unknown spectral loss mode: {mode}")
    if mode == "nonnegative_log" and bool((target < 0).any()):
        raise ValueError("nonnegative_log spectral loss received negative target")
    active = mask.unsqueeze(-1).to(prediction.dtype)
    difference = (prediction - target) * active
    target_active = target * active
    target_norm = torch.linalg.vector_norm(target_active.flatten(1), dim=1)


    convergence_per_sample = torch.linalg.vector_norm(
        difference.flatten(1), dim=1
    ) / (target_norm.clamp_min(min_target_norm) + epsilon)
    if mode == "nonnegative_log":
        log_difference = (
            torch.log(prediction.clamp_min(0.0) + epsilon)
            - torch.log(target + epsilon)
        ).abs() * active
        magnitude_per_sample = log_difference.sum(dim=(1, 2)) / (
            active.sum(dim=(1, 2)).clamp_min(1.0) * prediction.shape[-1]
        )
    else:
        magnitude_per_sample = difference.abs().sum(dim=(1, 2)) / (
            active.sum(dim=(1, 2)).clamp_min(1.0) * prediction.shape[-1]
        )
    enabled = mask.any(dim=1) & (target_norm > min_target_norm)
    if sample_weights is None:
        weights = enabled.to(prediction.dtype)
    else:
        weights = sample_weights.to(prediction.dtype) * enabled


    denominator = weights.sum()
    convergence = distributed_weighted_mean(
        (convergence_per_sample * weights).sum(),
        denominator,
    )
    magnitude = distributed_weighted_mean(
        (magnitude_per_sample * weights).sum(),
        denominator,
    )
    return convergence + magnitude, convergence.detach(), magnitude.detach()


class MultiTaskHeads(nn.Module):
    def __init__(
        self,
        model_dim: int,
        ctc_vocab_size: int,
        mel_bins: int = 128,
        chroma_bins: int = 12,
        convnext_blocks: int = 2,
        ctc_blank_id: int = 0,
        ctc_normalize_by_target_length: bool = True,
        mel_loss_mode: str = "signed_l1",
        chroma_loss_mode: str = "signed_l1",
    ) -> None:
        super().__init__()
        self.ctc_blank_id = ctc_blank_id
        self.ctc_normalize_by_target_length = (
            ctc_normalize_by_target_length
        )
        if mel_loss_mode not in {"signed_l1", "nonnegative_log"}:
            raise ValueError(f"Unknown mel_loss_mode={mel_loss_mode!r}")
        if chroma_loss_mode not in {"signed_l1", "nonnegative_log"}:
            raise ValueError(f"Unknown chroma_loss_mode={chroma_loss_mode!r}")
        self.mel_loss_mode = mel_loss_mode
        self.chroma_loss_mode = chroma_loss_mode
        self.ctc = nn.Linear(model_dim, ctc_vocab_size)
        self.mel = FramePredictionHead(model_dim, mel_bins, convnext_blocks)
        self.chroma = FramePredictionHead(model_dim, chroma_bins, convnext_blocks)

    def predict(
        self,
        hidden: torch.Tensor,
        *,
        mel_target_length: int,
        chroma_target_length: int,
        mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> dict[str, torch.Tensor]:
        predictions = {
            "ctc_logits": self.ctc(hidden),
            "mel": self.mel(hidden, mel_target_length, mask=mask, causal=causal),
            "chroma": self.chroma(
                hidden, chroma_target_length, mask=mask, causal=causal
            ),
        }
        if self.mel_loss_mode == "nonnegative_log":
            predictions["mel"] = F.softplus(predictions["mel"])
        if self.chroma_loss_mode == "nonnegative_log":
            predictions["chroma"] = F.softplus(predictions["chroma"])
        return predictions

    def compute_losses(
        self,
        predictions: dict[str, torch.Tensor],
        frame_lengths: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}

        ctc_enabled = batch["ctc_enabled"]
        sample_weights = batch.get("sample_weights")
        ctc_sample_weights = batch.get("ctc_sample_weights", sample_weights)
        mel_sample_weights = batch.get("mel_sample_weights", sample_weights)
        chroma_sample_weights = batch.get(
            "chroma_sample_weights", sample_weights
        )
        if ctc_enabled.any():
            logits = predictions["ctc_logits"][ctc_enabled]
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            targets = batch["lyrics_token_ids"][ctc_enabled]
            target_lengths = batch["lyrics_lengths"][ctc_enabled]
            flat_targets = torch.cat(
                [row[: int(length.item())] for row, length in zip(targets, target_lengths)]
            )
            ctc_loss = F.ctc_loss(
                log_probs,
                flat_targets,
                frame_lengths[ctc_enabled],
                target_lengths,
                blank=self.ctc_blank_id,
                zero_infinity=True,
                reduction="none",
            )


            if self.ctc_normalize_by_target_length:
                ctc_loss = ctc_loss / target_lengths.to(
                    ctc_loss.dtype
                ).clamp_min(1.0)
            if ctc_sample_weights is not None:
                active_weights = ctc_sample_weights[ctc_enabled].to(
                    ctc_loss.dtype
                )
            else:
                active_weights = torch.ones_like(ctc_loss)
            ctc_numerator = (ctc_loss * active_weights).sum()
            ctc_denominator = active_weights.sum()
        else:
            ctc_numerator = predictions["ctc_logits"].sum() * 0.0
            ctc_denominator = ctc_numerator.detach()
        ctc_loss = distributed_weighted_mean(
            ctc_numerator, ctc_denominator
        )
        losses["ctc"] = ctc_loss
        metrics["ctc_loss"] = ctc_loss.detach()
        metrics["ctc_loss_numerator"] = ctc_numerator.detach()
        metrics["ctc_weight_sum"] = ctc_denominator.detach()
        metrics["ctc_active_samples"] = ctc_enabled.sum().to(ctc_loss.dtype)
        metrics["ctc_infeasible_samples"] = batch.get(
            "ctc_infeasible", torch.zeros_like(ctc_enabled)
        ).sum().to(ctc_loss.dtype)

        alignment_mask = batch.get("ctc_frame_target_mask")
        if alignment_mask is not None:
            alignment_mask = (
                alignment_mask[:, : predictions["ctc_logits"].shape[1]]
                & batch["ctc_alignment_enabled"].unsqueeze(1)
            )
        if alignment_mask is not None and alignment_mask.any():
            aligned_logits = predictions["ctc_logits"][:, : alignment_mask.shape[1]]
            ctc_alignment_per_frame = F.cross_entropy(
                aligned_logits[alignment_mask],
                batch["ctc_frame_token_ids"][:, : alignment_mask.shape[1]][
                    alignment_mask
                ],
                reduction="none",
            )
            if ctc_sample_weights is not None:
                frame_weights = ctc_sample_weights.unsqueeze(1).expand_as(
                    alignment_mask
                )[alignment_mask].to(ctc_alignment_per_frame.dtype)
            else:
                frame_weights = torch.ones_like(ctc_alignment_per_frame)
            ctc_alignment_numerator = (
                ctc_alignment_per_frame * frame_weights
            ).sum()
            ctc_alignment_denominator = frame_weights.sum()
            ctc_alignment_accuracy = (
                aligned_logits[alignment_mask].argmax(dim=-1)
                == batch["ctc_frame_token_ids"][:, : alignment_mask.shape[1]][
                    alignment_mask
                ]
            ).float().mean()
        else:
            ctc_alignment_numerator = (
                predictions["ctc_logits"].square().sum() * 0.0
            )
            ctc_alignment_denominator = ctc_alignment_numerator.detach()
            ctc_alignment_accuracy = (
                ctc_alignment_numerator.detach().new_zeros(())
            )
        ctc_alignment_loss = distributed_weighted_mean(
            ctc_alignment_numerator,
            ctc_alignment_denominator,
        )
        losses["ctc_alignment"] = ctc_alignment_loss
        metrics["ctc_alignment_loss"] = ctc_alignment_loss.detach()
        metrics["ctc_alignment_accuracy"] = ctc_alignment_accuracy.detach()

        mel_prediction = predictions["mel"]
        mel_mask = batch["mel_target_mask"] & batch["mel_enabled"].unsqueeze(1)
        mel_loss, mel_sc, mel_mag = spectral_loss(
            mel_prediction,
            batch["mel_target"],
            mel_mask,
            sample_weights=mel_sample_weights,
            mode=self.mel_loss_mode,
        )
        losses["mel"] = mel_loss
        metrics.update(
            mel_loss=mel_loss.detach(),
            mel_spectral_convergence=mel_sc,
            mel_magnitude=mel_mag,
            mel_l1=masked_l1(
                mel_prediction,
                batch["mel_target"],
                mel_mask,
                sample_weights=mel_sample_weights,
            ).detach(),
        )

        chroma_prediction = predictions["chroma"]
        chroma_mask = batch["chroma_target_mask"] & batch["chroma_enabled"].unsqueeze(1)
        if self.chroma_loss_mode == "nonnegative_log":
            chroma_mask = chroma_mask & (
                batch["chroma_target"].sum(dim=-1) > 1e-6
            )
        chroma_loss, chroma_sc, chroma_mag = spectral_loss(
            chroma_prediction,
            batch["chroma_target"],
            chroma_mask,
            sample_weights=chroma_sample_weights,
            mode=self.chroma_loss_mode,
        )
        losses["chroma"] = chroma_loss
        metrics.update(
            chroma_loss=chroma_loss.detach(),
            chroma_spectral_convergence=chroma_sc,
            chroma_magnitude=chroma_mag,
        )
        return losses, metrics

    def forward(
        self,
        hidden: torch.Tensor,
        frame_lengths: torch.Tensor,
        batch: dict[str, torch.Tensor],
        causal: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        predictions = self.predict(
            hidden,
            mel_target_length=batch["mel_target"].shape[1],
            chroma_target_length=batch["chroma_target"].shape[1],
            mask=lengths_to_mask(frame_lengths, hidden.shape[1]),
            causal=causal,
        )
        return self.compute_losses(predictions, frame_lengths, batch)
