
from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from open_qwen_music.common.distributed import auxiliary_cuda_group


def _zero_with_parameter_graph(
    module: nn.Module, hidden: torch.Tensor
) -> torch.Tensor:

    zero = hidden.sum() * 0.0
    for parameter in module.parameters():
        if parameter.requires_grad:
            zero = zero + parameter.sum() * 0.0
    return zero


def _weighted_isotropy_statistics(
    flat: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    weighted_sum = torch.einsum("n,ni->i", weights, flat)
    weighted_second_moment = torch.einsum(
        "n,ni,nj->ij",
        weights,
        flat,
        flat,
    )
    return weighted_sum, weighted_second_moment, weights.sum()


def distributed_quantizer_diversity(
    local_weighted_sum: torch.Tensor,
    local_weighted_second_moment: torch.Tensor,
    local_weight_sum: torch.Tensor,
    *,
    distributed: bool = True,
) -> torch.Tensor:

    if local_weighted_sum.ndim != 1:
        raise ValueError("diversity first moment must have shape [D]")
    dimension = int(local_weighted_sum.shape[0])
    if local_weighted_second_moment.shape != (dimension, dimension):
        raise ValueError("diversity second moment must have shape [D,D]")
    if local_weight_sum.numel() != 1:
        raise ValueError("diversity weight sum must be a scalar")

    use_collective = (
        distributed
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )
    world_size = dist.get_world_size() if use_collective else 1
    if use_collective:
        packed = torch.cat(
            [
                local_weighted_sum.detach().reshape(-1),
                local_weighted_second_moment.detach().reshape(-1),
                local_weight_sum.detach().reshape(1),
            ]
        )
        group = auxiliary_cuda_group() if packed.is_cuda else None
        dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=group)
        split = dimension
        global_sum_detached = packed[:split].reshape_as(local_weighted_sum)
        global_second_detached = packed[
            split : split + dimension * dimension
        ].reshape_as(local_weighted_second_moment)
        global_weight_sum = packed[-1]
        weighted_sum = local_weighted_sum + (
            global_sum_detached - local_weighted_sum.detach()
        )
        weighted_second_moment = local_weighted_second_moment + (
            global_second_detached
            - local_weighted_second_moment.detach()
        )
    else:
        weighted_sum = local_weighted_sum
        weighted_second_moment = local_weighted_second_moment
        global_weight_sum = local_weight_sum

    if float(global_weight_sum.detach().item()) <= 0.0:
        return weighted_sum.sum() * 0.0 + weighted_second_moment.sum() * 0.0
    denominator = global_weight_sum.clamp_min(1e-8)
    mean = weighted_sum / denominator
    second_moment = weighted_second_moment / denominator
    target = torch.eye(
        dimension,
        device=weighted_sum.device,
        dtype=weighted_sum.dtype,
    ) / float(dimension)
    diversity = mean.square().sum() + (second_moment - target).square().sum()
    if world_size == 1:
        return diversity
    gradient_scaled = diversity * float(world_size)
    return diversity.detach() + gradient_scaled - gradient_scaled.detach()


def codebook_metrics(
    ids: torch.Tensor,
    codebook_size: int,
    weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:

    flat_ids = ids.reshape(-1)
    if weights is None:
        flat_weights = torch.ones(
            flat_ids.shape,
            device=flat_ids.device,
            dtype=torch.float32,
        )
    else:
        if weights.shape != ids.shape:
            raise ValueError(
                "codebook statistics weights must have the same shape as ids; received "
                f"{tuple(weights.shape)} != {tuple(ids.shape)}"
            )
        flat_weights = weights.reshape(-1).to(
            device=flat_ids.device, dtype=torch.float32
        )
    metrics_device = flat_ids.device
    if flat_ids.is_cuda and torch.are_deterministic_algorithms_enabled():


        counts = torch.bincount(
            flat_ids.cpu(),
            weights=flat_weights.cpu(),
            minlength=codebook_size,
        ).to(metrics_device)
    else:
        counts = torch.bincount(
            flat_ids,
            weights=flat_weights,
            minlength=codebook_size,
        )
    total = counts.sum()
    probabilities = counts / total.clamp_min(1e-8)
    active = counts > 0
    entropy_bits = -(probabilities[active] * probabilities[active].log2()).sum()
    return {
        "codebook_utilization": active.float().mean(),
        "codebook_perplexity": torch.pow(2.0, entropy_bits),
        "codebook_entropy_bits": entropy_bits,
        "dead_code_rate": 1.0 - active.float().mean(),
        "codebook_batch_frames": counts.new_tensor(float(flat_ids.numel())),
        "codebook_batch_weight": counts.sum(),
    }


def _resolve_loss_weights(
    hidden: torch.Tensor,
    mask: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    loss_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:

    if mask is None:
        assignment_mask = torch.ones(
            hidden.shape[:2], dtype=torch.bool, device=hidden.device
        )
    else:
        assignment_mask = mask.to(device=hidden.device, dtype=torch.bool)
        if assignment_mask.shape != hidden.shape[:2]:
            raise ValueError(
                "mask must match the leading dimensions of hidden; received "
                f"{tuple(assignment_mask.shape)} != {tuple(hidden.shape[:2])}"
            )
    if loss_mask is None:
        eligible_mask = assignment_mask
    else:
        eligible_mask = loss_mask.to(device=hidden.device, dtype=torch.bool)
        if eligible_mask.shape != assignment_mask.shape:
            raise ValueError(
                f"loss_mask must have the same shape as mask; received "
                f"{tuple(eligible_mask.shape)} != {tuple(assignment_mask.shape)}"
            )
        if bool((eligible_mask & ~assignment_mask).any()):
            raise ValueError("loss_mask cannot select frames outside mask")
    if loss_weights is None:
        frame_weights = eligible_mask.to(dtype=hidden.dtype)
    else:
        frame_weights = loss_weights.to(device=hidden.device, dtype=hidden.dtype)
        if frame_weights.shape != assignment_mask.shape:
            raise ValueError(
                f"loss_weights must have the same shape as mask; received "
                f"{tuple(frame_weights.shape)} != {tuple(assignment_mask.shape)}"
            )
        if bool((~torch.isfinite(frame_weights)).any()) or bool(
            (frame_weights < 0.0).any()
        ):
            raise ValueError("loss_weights must be a non-negative finite number")
        if bool(((frame_weights > 0.0) & ~eligible_mask).any()):
            raise ValueError(
                "loss_weights must not fall on loss_mask or mask except"
            )
        frame_weights = frame_weights * eligible_mask
    return assignment_mask, frame_weights


class CosineVectorQuantizer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        code_dim: int,
        codebook_size: int,
        commitment_beta: float = 0.25,
        commitment_scale: float = 1.0,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 1.0,
        dead_code_threshold_relative: float | None = None,
        dead_code_warmup_steps: int = 100,
        dead_code_max_replacements: int = 1024,
        dead_code_replacement_noise: float = 0.01,
        diversity_beta: float = 0.0,
        update_mode: str = "ema",
        codebook_beta: float = 1.0,
        freeze_input_projection: bool = False,
        ema_statistics_mode: str = "rank0_scaled",
        distance_chunk_size: int = 4096,
        dead_code_candidate_mode: str = "rank0_local",
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, code_dim)
        self.output_proj = nn.Linear(code_dim, input_dim)
        codebook = F.normalize(torch.randn(codebook_size, code_dim), dim=-1)
        if update_mode == "gradient":
            self.codebook = nn.Parameter(codebook)
        elif update_mode == "ema":
            self.register_buffer("codebook", codebook)
        else:
            raise ValueError(f"Unknown VQ update_mode={update_mode}")
        self.register_buffer("ema_count", torch.ones(codebook_size))
        self.register_buffer("ema_sum", codebook.clone())
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))
        self.codebook_size = codebook_size
        self.commitment_beta = commitment_beta
        if commitment_scale <= 0:
            raise ValueError(
                f"commitment_scale must be positive; received {commitment_scale}"
            )
        self.commitment_scale = commitment_scale
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold
        self.dead_code_threshold_relative = dead_code_threshold_relative
        self.dead_code_warmup_steps = dead_code_warmup_steps


        self._last_total_frames = float(codebook_size)
        self.dead_code_max_replacements = dead_code_max_replacements
        self.dead_code_replacement_noise = dead_code_replacement_noise
        self.diversity_beta = diversity_beta
        self.update_mode = update_mode
        self.codebook_beta = codebook_beta
        if freeze_input_projection:
            for parameter in self.input_proj.parameters():
                parameter.requires_grad_(False)
        if ema_statistics_mode not in {"rank0_scaled", "all_reduce"}:
            raise ValueError(
                "ema_statistics_mode must be 'rank0_scaled' or 'all_reduce'; received "
                f"{ema_statistics_mode!r}"
            )
        self.ema_statistics_mode = ema_statistics_mode
        if dead_code_candidate_mode not in {"rank0_local", "all_gather_balanced"}:
            raise ValueError(
                "dead_code_candidate_mode must be 'rank0_local' or "
                "'all_gather_balanced'; received "
                f"{dead_code_candidate_mode!r}"
            )
        if (
            dead_code_candidate_mode == "all_gather_balanced"
            and ema_statistics_mode != "all_reduce"
        ):
            raise ValueError(
                "all_gather_balanced dead-code candidates require ema_statistics_mode=all_reduce"
            )
        self.dead_code_candidate_mode = dead_code_candidate_mode
        self.distance_chunk_size = distance_chunk_size
        self._pending_count: torch.Tensor | None = None
        self._pending_sum: torch.Tensor | None = None
        self._pending_candidates: torch.Tensor | None = None
        self._pending_candidate_priorities: torch.Tensor | None = None

    def effective_codebook(self) -> torch.Tensor:

        return F.normalize(self.codebook, dim=-1)

    def _nearest(self, flat: torch.Tensor) -> torch.Tensor:

        normalized_codebook = self.effective_codebook()
        result = []
        for chunk in flat.split(self.distance_chunk_size):
            result.append((chunk @ normalized_codebook.T).argmax(dim=-1))
        return torch.cat(result)

    @torch.no_grad()
    def _accumulate_ema(
        self,
        flat: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        counts = torch.bincount(
            ids,
            weights=weights,
            minlength=self.codebook_size,
        ).to(flat.dtype)

        sums = None
        if self.update_mode == "ema":
            sums = flat.new_zeros(self.codebook_size, flat.shape[-1])
            sums.index_add_(0, ids, flat * weights.unsqueeze(-1))
        if self._pending_count is None:
            self._pending_count = counts
            self._pending_sum = sums
        else:
            self._pending_count.add_(counts)
            if sums is not None and self._pending_sum is not None:
                self._pending_sum.add_(sums)


        priorities = torch.rand(
            weights.shape,
            device=weights.device,
            dtype=weights.dtype,
        ).clamp_min(torch.finfo(weights.dtype).tiny)
        priorities = priorities.log() / weights
        candidate_count = min(flat.shape[0], self.dead_code_max_replacements)
        if flat.shape[0] > candidate_count:
            priorities, candidate_ids = priorities.topk(candidate_count)
            candidates = flat[candidate_ids]
        else:
            candidates = flat
        if self._pending_candidates is None:
            self._pending_candidates = candidates
            self._pending_candidate_priorities = priorities
            return
        assert self._pending_candidate_priorities is not None
        combined = torch.cat([self._pending_candidates, candidates], dim=0)
        combined_priorities = torch.cat(
            [self._pending_candidate_priorities, priorities], dim=0
        )
        if combined.shape[0] > self.dead_code_max_replacements:
            combined_priorities, keep = combined_priorities.topk(
                self.dead_code_max_replacements
            )
            combined = combined[keep]
        self._pending_candidates = combined
        self._pending_candidate_priorities = combined_priorities

    def _dead_threshold(self) -> float:

        if self.dead_code_threshold_relative is None:
            return float(self.dead_code_threshold)
        uniform = float(self._last_total_frames) / float(self.codebook_size)
        return float(self.dead_code_threshold_relative) * uniform

    @torch.no_grad()
    def _revive_dead_codes(self, candidates: torch.Tensor | None) -> int:

        if self.num_updates.item() < self.dead_code_warmup_steps:
            return 0
        if candidates is None or candidates.shape[0] == 0:
            return 0
        dead = self.ema_count < self._dead_threshold()
        if not bool(dead.any()):
            return 0


        replacement_count = min(
            int(dead.sum().item()),
            self.dead_code_max_replacements,
            candidates.shape[0],
        )
        if replacement_count <= 0:
            return 0
        dead_pool = dead.nonzero(as_tuple=False).flatten()
        dead_ids = dead_pool[
            torch.randperm(dead_pool.shape[0], device=dead_pool.device)[
                :replacement_count
            ]
        ]
        replacement_ids = torch.randperm(
            candidates.shape[0], device=candidates.device
        )[:replacement_count]
        replacements = candidates[replacement_ids]
        if self.dead_code_replacement_noise > 0:
            replacements = F.normalize(
                replacements
                + self.dead_code_replacement_noise * torch.randn_like(replacements),
                dim=-1,
            )

        replacement_mass = max(self.dead_code_threshold * 2.0, 1.0)
        self.ema_sum[dead_ids] = replacements * replacement_mass
        self.ema_count[dead_ids] = replacement_mass
        if self.update_mode == "gradient":


            self.codebook.data[dead_ids] = replacements
        return replacement_count

    @torch.no_grad()
    def _broadcast_codebook(self) -> None:

        if not (dist.is_available() and dist.is_initialized()):
            return
        if self.update_mode != "gradient":
            return
        group = auxiliary_cuda_group() if self.codebook.is_cuda else None
        dist.broadcast(self.codebook.data, src=0, group=group)

    @torch.no_grad()
    def _gather_balanced_revival_candidates(
        self,
        candidates: torch.Tensor | None,
        priorities: torch.Tensor | None,
    ) -> torch.Tensor | None:

        distributed = dist.is_available() and dist.is_initialized()
        if not distributed:
            return candidates
        world_size = dist.get_world_size()
        quota = max(
            1,
            (self.dead_code_max_replacements + world_size - 1) // world_size,
        )
        packed = self.codebook.new_zeros(
            (quota, int(self.codebook.shape[-1]) + 1)
        )
        packed[:, -1] = -torch.inf
        if candidates is not None and candidates.shape[0] > 0:
            if priorities is None or priorities.shape[0] != candidates.shape[0]:
                raise RuntimeError(
                    "dead-code candidates and weighted-reservoir priorities "
                    "must have the same length"
                )
            take = min(quota, int(candidates.shape[0]))
            if candidates.shape[0] > take:
                local_priorities, keep = priorities.topk(take)
                local_candidates = candidates[keep]
            else:
                local_candidates = candidates
                local_priorities = priorities
            packed[:take, :-1] = local_candidates[:take]
            packed[:take, -1] = local_priorities[:take]

        gathered = [torch.empty_like(packed) for _ in range(world_size)]
        group = auxiliary_cuda_group() if packed.is_cuda else None
        dist.all_gather(gathered, packed, group=group)
        if dist.get_rank() != 0:
            return None
        global_packed = torch.cat(gathered, dim=0)
        valid = torch.isfinite(global_packed[:, -1])
        if not bool(valid.any()):
            return None
        global_packed = global_packed[valid]
        if global_packed.shape[0] > self.dead_code_max_replacements:
            _, keep = global_packed[:, -1].topk(self.dead_code_max_replacements)
            global_packed = global_packed[keep]
        return global_packed[:, :-1]

    @torch.no_grad()
    def synchronize_ema(self) -> int:

        distributed = dist.is_available() and dist.is_initialized()
        exact_global = distributed and self.ema_statistics_mode == "all_reduce"
        if self._pending_count is None and not exact_global:

            self._broadcast_codebook()
            return 0


        counts = (
            self._pending_count
            if self._pending_count is not None
            else self.ema_count.new_zeros(self.codebook_size)
        )
        sums = self._pending_sum
        if exact_global and self.update_mode == "ema" and sums is None:
            sums = self.ema_sum.new_zeros(self.ema_sum.shape)
        candidates = self._pending_candidates
        candidate_priorities = self._pending_candidate_priorities
        self._pending_count = None
        self._pending_sum = None
        self._pending_candidates = None
        self._pending_candidate_priorities = None
        if exact_global:
            group = auxiliary_cuda_group() if counts.is_cuda else None
            dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
            if sums is not None:
                dist.all_reduce(sums, op=dist.ReduceOp.SUM, group=group)
        if (
            distributed
            and self.dead_code_candidate_mode == "all_gather_balanced"
            and int(self.num_updates.item()) + 1 >= self.dead_code_warmup_steps
        ):
            candidates = self._gather_balanced_revival_candidates(
                candidates,
                candidate_priorities,
            )
        if distributed and dist.get_rank() != 0:
            self._broadcast_codebook()
            return 0
        if distributed and not exact_global:


            world_size = dist.get_world_size()
            counts = counts * world_size
            if sums is not None:
                sums = sums * world_size

        self._last_total_frames = float(counts.sum().item())
        self.ema_count.mul_(self.ema_decay).add_(counts, alpha=1.0 - self.ema_decay)
        if sums is not None:
            self.ema_sum.mul_(self.ema_decay).add_(sums, alpha=1.0 - self.ema_decay)
        self.num_updates.add_(1)


        replacement_count = self._revive_dead_codes(candidates)
        if self.update_mode == "ema":
            self.codebook.copy_(
                F.normalize(
                    self.ema_sum / self.ema_count.unsqueeze(-1).clamp_min(1e-5), dim=-1
                )
            )
        self._broadcast_codebook()
        return replacement_count

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        loss_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:


        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self._forward_fp32(
                hidden.float(),
                mask,
                loss_mask,
                loss_weights,
            )

    def project_continuous(self, hidden: torch.Tensor) -> torch.Tensor:

        with torch.autocast(device_type=hidden.device.type, enabled=False):
            projected = F.normalize(self.input_proj(hidden.float()), dim=-1)
            return self.output_proj(projected)

    def project_radius_preserving(self, hidden: torch.Tensor) -> torch.Tensor:

        with torch.autocast(device_type=hidden.device.type, enabled=False):
            projected = self.input_proj(hidden.float())
            projected = projected / math.sqrt(projected.shape[-1])
            return self.output_proj(projected)

    def _forward_fp32(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        loss_mask: torch.Tensor | None = None,
        loss_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        projected = F.normalize(self.input_proj(hidden), dim=-1)
        mask, frame_weights = _resolve_loss_weights(
            hidden,
            mask,
            loss_mask,
            loss_weights,
        )
        assignment_flat = projected[mask]
        if assignment_flat.shape[0] == 0:
            zero = _zero_with_parameter_graph(self, hidden)
            empty_weights = frame_weights[mask]
            diversity_sum, diversity_second, diversity_weight = (
                _weighted_isotropy_statistics(
                    assignment_flat,
                    empty_weights,
                )
            )
            ids = torch.zeros(hidden.shape[:2], dtype=torch.long, device=hidden.device)
            metrics = codebook_metrics(
                ids.new_empty(0),
                self.codebook_size,
                weights=hidden.new_empty(0),
            )
            metrics.update(
                commitment_loss=zero.detach(),
                commitment_scale=zero.detach().new_tensor(self.commitment_scale),
                codebook_loss=zero.detach(),
                diversity_loss=zero.detach(),
            )
            if self.diversity_beta != 0.0:
                metrics.update(
                    _diversity_local_loss=zero,
                    _diversity_weighted_sum=diversity_sum,
                    _diversity_weighted_second_moment=diversity_second,
                    _diversity_weight_sum=diversity_weight,
                )
            return hidden, ids, zero, metrics
        with torch.no_grad():
            assignment_ids = self._nearest(assignment_flat)
        ids = torch.zeros(hidden.shape[:2], dtype=torch.long, device=hidden.device)
        ids[mask] = assignment_ids
        normalized_codebook = self.effective_codebook()
        quantized = F.embedding(ids, normalized_codebook)
        positive_weight = frame_weights > 0.0
        eligible_flat = projected[positive_weight]
        eligible_ids = ids[positive_weight]
        eligible_weights = frame_weights[positive_weight]
        if self.training and eligible_flat.shape[0] > 0:


            self._accumulate_ema(
                eligible_flat.detach(),
                eligible_ids,
                eligible_weights.detach(),
            )
        straight_through = projected + (quantized - projected).detach()
        output = self.output_proj(straight_through)
        if eligible_flat.shape[0] == 0:


            zero = _zero_with_parameter_graph(self, hidden)
            diversity_sum, diversity_second, diversity_weight = (
                _weighted_isotropy_statistics(
                    eligible_flat,
                    eligible_weights,
                )
            )
            metrics = codebook_metrics(
                eligible_ids,
                self.codebook_size,
                weights=eligible_weights,
            )
            metrics.update(
                commitment_loss=zero.detach(),
                commitment_scale=zero.detach().new_tensor(self.commitment_scale),
                codebook_loss=zero.detach(),
                diversity_loss=zero.detach(),
            )
            if self.diversity_beta != 0.0:
                metrics.update(
                    _diversity_local_loss=zero,
                    _diversity_weighted_sum=diversity_sum,
                    _diversity_weighted_second_moment=diversity_second,
                    _diversity_weight_sum=diversity_weight,
                )
            return output, ids, zero, metrics
        weight_sum = eligible_weights.sum()
        commitment_per_frame = (
            projected[positive_weight] - quantized.detach()[positive_weight]
        ).square().mean(dim=-1)
        commitment = self.commitment_beta * self.commitment_scale * (
            commitment_per_frame * eligible_weights
        ).sum() / weight_sum
        if self.update_mode == "gradient":
            codebook_per_frame = (
                projected.detach()[positive_weight] - quantized[positive_weight]
            ).square().mean(dim=-1)
            codebook_loss = self.codebook_beta * (
                codebook_per_frame * eligible_weights
            ).sum() / weight_sum
        else:
            codebook_loss = commitment.new_zeros(())


        diversity_sum, diversity_second, diversity_weight = (
            _weighted_isotropy_statistics(
                eligible_flat,
                eligible_weights,
            )
        )
        mean = diversity_sum / weight_sum
        second_moment = diversity_second / weight_sum
        target = torch.eye(
            eligible_flat.shape[-1],
            device=eligible_flat.device,
            dtype=eligible_flat.dtype,
        )
        target = target / eligible_flat.shape[-1]
        diversity = mean.square().sum() + (
            second_moment - target
        ).square().sum()
        quantizer_loss = (
            commitment
            + codebook_loss
            + self.diversity_beta * diversity
        )
        metrics = codebook_metrics(
            eligible_ids,
            self.codebook_size,
            weights=eligible_weights,
        )
        metrics["commitment_loss"] = commitment.detach()
        metrics["commitment_scale"] = commitment.new_tensor(self.commitment_scale)
        metrics["codebook_loss"] = codebook_loss.detach()
        metrics["diversity_loss"] = diversity.detach()
        if self.diversity_beta != 0.0:
            metrics.update(
                _diversity_local_loss=diversity,
                _diversity_weighted_sum=diversity_sum,
                _diversity_weighted_second_moment=diversity_second,
                _diversity_weight_sum=diversity_weight,
            )
        return output, ids, quantizer_loss, metrics


class CosineSimVQ(CosineVectorQuantizer):

    def __init__(
        self,
        input_dim: int,
        code_dim: int,
        codebook_size: int,
        commitment_beta: float = 0.25,
        commitment_scale: float = 1.0,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 1.0,
        dead_code_threshold_relative: float | None = None,
        dead_code_warmup_steps: int = 100,
        dead_code_max_replacements: int = 1024,
        dead_code_replacement_noise: float = 0.01,
        diversity_beta: float = 0.0,
        update_mode: str = "gradient",
        codebook_beta: float = 1.0,
        freeze_input_projection: bool = False,
        distance_chunk_size: int = 4096,
        dead_code_revival: bool = False,
    ) -> None:
        if update_mode != "gradient":
            raise ValueError(
                "cosine_simvq requires update_mode='gradient'; "
                f"received update_mode={update_mode!r}"
            )
        if dead_code_revival:
            raise ValueError(
                "cosine_simvq uses a fixed base_codebook and does not support "
                "dead-code revival"
            )
        super().__init__(
            input_dim=input_dim,
            code_dim=code_dim,
            codebook_size=codebook_size,
            commitment_beta=commitment_beta,
            commitment_scale=commitment_scale,
            ema_decay=ema_decay,
            dead_code_threshold=dead_code_threshold,
            dead_code_threshold_relative=dead_code_threshold_relative,
            dead_code_warmup_steps=dead_code_warmup_steps,
            dead_code_max_replacements=dead_code_max_replacements,
            dead_code_replacement_noise=dead_code_replacement_noise,
            diversity_beta=diversity_beta,
            update_mode="gradient",
            codebook_beta=codebook_beta,
            freeze_input_projection=freeze_input_projection,
            distance_chunk_size=distance_chunk_size,
        )
        base_codebook = self.codebook.detach().clone()
        del self.codebook
        self.register_buffer("base_codebook", base_codebook)
        self.codebook_transform = nn.Linear(code_dim, code_dim, bias=False)
        nn.init.eye_(self.codebook_transform.weight)
        self.dead_code_revival = False

    def effective_codebook(self) -> torch.Tensor:

        weight = self.codebook_transform.weight
        identity = torch.eye(
            weight.shape[0], device=weight.device, dtype=weight.dtype
        )


        transformed = self.base_codebook + F.linear(
            self.base_codebook, weight - identity
        )
        return F.normalize(transformed, dim=-1)

    @torch.no_grad()
    def _accumulate_ema(
        self,
        flat: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:

        counts = torch.bincount(
            ids,
            weights=weights,
            minlength=self.codebook_size,
        ).to(flat.dtype)
        if self._pending_count is None:
            self._pending_count = counts
        else:
            self._pending_count.add_(counts)
        self._pending_sum = None
        self._pending_candidates = None
        self._pending_candidate_priorities = None

    @torch.no_grad()
    def _revive_dead_codes(self, candidates: torch.Tensor | None) -> int:

        return 0

    @torch.no_grad()
    def _broadcast_codebook(self) -> None:

        return

    def _forward_fp32(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        loss_mask: torch.Tensor | None = None,
        loss_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        output, ids, loss, metrics = super()._forward_fp32(
            hidden, mask, loss_mask, loss_weights
        )
        identity = torch.eye(
            self.codebook_transform.weight.shape[0],
            device=self.codebook_transform.weight.device,
            dtype=self.codebook_transform.weight.dtype,
        )
        metrics["simvq_transform_delta_norm"] = (
            self.codebook_transform.weight.detach() - identity
        ).norm()
        metrics["dead_code_revival_enabled"] = loss.detach().new_zeros(())
        return output, ids, loss, metrics


class FiniteScalarQuantizer(nn.Module):

    def __init__(self, input_dim: int, levels: list[int]) -> None:
        super().__init__()
        if math.prod(levels) != 32_768:
            raise ValueError(f"the product of FSQ levels must be 32768; received {levels}")
        self.levels = levels
        self.codebook_size = math.prod(levels)
        self.input_proj = nn.Linear(input_dim, len(levels))
        self.output_proj = nn.Linear(len(levels), input_dim)
        basis = [1]
        for level in levels[:-1]:
            basis.append(basis[-1] * level)
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.long))
        self.register_buffer("levels_tensor", torch.tensor(levels, dtype=torch.float32))

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        loss_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        bounded = torch.tanh(self.input_proj(hidden))
        levels = self.levels_tensor.to(bounded)
        indices = torch.round((bounded + 1.0) * 0.5 * (levels - 1.0))
        quantized = indices / (levels - 1.0) * 2.0 - 1.0
        straight_through = bounded + (quantized - bounded).detach()
        ids = (indices.long() * self.basis).sum(dim=-1)
        mask, frame_weights = _resolve_loss_weights(
            hidden,
            mask,
            loss_mask,
            loss_weights,
        )
        assignment_ids = ids[mask]
        positive_weight = frame_weights > 0.0
        valid_ids = ids[positive_weight]
        valid_weights = frame_weights[positive_weight]
        if assignment_ids.numel() == 0:
            zero = _zero_with_parameter_graph(self, hidden)
            empty_ids = torch.zeros(
                hidden.shape[:2], dtype=torch.long, device=hidden.device
            )
            metrics = codebook_metrics(
                valid_ids,
                self.codebook_size,
                weights=valid_weights,
            )
            metrics["commitment_loss"] = zero.detach()
            metrics["codebook_loss"] = zero.detach()
            metrics["diversity_loss"] = zero.detach()
            return hidden, empty_ids, zero, metrics
        output = self.output_proj(straight_through)
        zero = _zero_with_parameter_graph(self, hidden)
        metrics = codebook_metrics(
            valid_ids,
            self.codebook_size,
            weights=valid_weights,
        )
        metrics["commitment_loss"] = zero.detach()
        metrics["codebook_loss"] = zero.detach()
        metrics["diversity_loss"] = zero.detach()
        return output, ids, zero, metrics

    def synchronize_ema(self) -> int:
        return 0


def build_quantizer(config: dict, input_dim: int) -> nn.Module:
    quantizer_type = config.get("type", "cosine_vq")
    if quantizer_type == "cosine_vq":
        return CosineVectorQuantizer(
            input_dim=input_dim,
            code_dim=int(config.get("code_dim", 16)),
            codebook_size=int(config["codebook_size"]),
            commitment_beta=float(config.get("commitment_beta", 0.25)),
            commitment_scale=float(config.get("commitment_scale", 1.0)),
            ema_decay=float(config.get("ema_decay", 0.99)),
            dead_code_threshold=float(config.get("dead_code_threshold", 1.0)),
            dead_code_threshold_relative=(
                None
                if config.get("dead_code_threshold_relative") is None
                else float(config["dead_code_threshold_relative"])
            ),
            dead_code_warmup_steps=int(config.get("dead_code_warmup_steps", 100)),
            dead_code_max_replacements=int(
                config.get("dead_code_max_replacements", 1024)
            ),
            dead_code_replacement_noise=float(
                config.get("dead_code_replacement_noise", 0.01)
            ),
            diversity_beta=float(config.get("diversity_beta", 0.0)),
            update_mode=str(config.get("update_mode", "ema")),
            codebook_beta=float(config.get("codebook_beta", 1.0)),
            freeze_input_projection=bool(
                config.get("freeze_input_projection", False)
            ),
            ema_statistics_mode=str(
                config.get("ema_statistics_mode", "rank0_scaled")
            ),
            dead_code_candidate_mode=str(
                config.get("dead_code_candidate_mode", "rank0_local")
            ),
            distance_chunk_size=int(config.get("distance_chunk_size", 4096)),
        )
    if quantizer_type == "cosine_simvq":
        return CosineSimVQ(
            input_dim=input_dim,
            code_dim=int(config.get("code_dim", 16)),
            codebook_size=int(config["codebook_size"]),
            commitment_beta=float(config.get("commitment_beta", 0.25)),
            commitment_scale=float(config.get("commitment_scale", 1.0)),
            ema_decay=float(config.get("ema_decay", 0.99)),
            dead_code_threshold=float(config.get("dead_code_threshold", 1.0)),
            dead_code_threshold_relative=(
                None
                if config.get("dead_code_threshold_relative") is None
                else float(config["dead_code_threshold_relative"])
            ),
            dead_code_warmup_steps=int(config.get("dead_code_warmup_steps", 100)),
            dead_code_max_replacements=int(
                config.get("dead_code_max_replacements", 1024)
            ),
            dead_code_replacement_noise=float(
                config.get("dead_code_replacement_noise", 0.01)
            ),
            diversity_beta=float(config.get("diversity_beta", 0.0)),
            update_mode=str(config.get("update_mode", "gradient")),
            codebook_beta=float(config.get("codebook_beta", 1.0)),
            freeze_input_projection=bool(
                config.get("freeze_input_projection", False)
            ),
            distance_chunk_size=int(config.get("distance_chunk_size", 4096)),
            dead_code_revival=bool(config.get("dead_code_revival", False)),
        )
    if quantizer_type == "fsq":
        return FiniteScalarQuantizer(input_dim=input_dim, levels=list(config["levels"]))
    raise ValueError(f"Unknown quantizer type: {quantizer_type}")
