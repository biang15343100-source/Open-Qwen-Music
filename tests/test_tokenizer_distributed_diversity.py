from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from open_qwen_music.tokenizer.quantizer import (
    distributed_quantizer_diversity,
)

class _DiversityProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(
                [[0.9, 0.2], [-0.1, 1.1]],
                dtype=torch.float64,
            )
        )

    def forward(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        *,
        distributed: bool,
    ) -> torch.Tensor:
        projected = torch.nn.functional.normalize(
            inputs @ self.weight.T,
            dim=-1,
        )
        weighted_sum = torch.einsum("n,ni->i", weights, projected)
        weighted_second = torch.einsum(
            "n,ni,nj->ij",
            weights,
            projected,
            projected,
        )
        return distributed_quantizer_diversity(
            weighted_sum,
            weighted_second,
            weights.sum(),
            distributed=distributed,
        )


def _rank_inputs(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if rank == 0:
        return (
            torch.tensor([[0.8, 0.6], [1.0, 0.0]], dtype=torch.float64),
            torch.tensor([0.5, 1.5], dtype=torch.float64),
        )
    return (
        torch.tensor(
            [[0.0, 1.0], [-0.6, 0.8], [-1.0, 0.0]],
            dtype=torch.float64,
        ),
        torch.tensor([1.0, 0.25, 0.75], dtype=torch.float64),
    )


def _worker(rank: int, world_size: int, output: str) -> None:
    store = dist.FileStore(str(Path(output) / "rendezvous"), world_size)
    dist.init_process_group(
        "gloo",
        store=store,
        rank=rank,
        world_size=world_size,
    )
    try:
        model = DistributedDataParallel(_DiversityProjection())
        inputs, weights = _rank_inputs(rank)
        loss = model(inputs, weights, distributed=True)
        loss.backward()
        first_gradient = model.module.weight.grad.detach().clone()
        model.zero_grad(set_to_none=True)
        if rank == 0:
            zero_inputs = torch.empty(0, 2, dtype=torch.float64)
            zero_weights = torch.empty(0, dtype=torch.float64)
        else:
            zero_inputs, zero_weights = _rank_inputs(rank)
        zero_rank_loss = model(
            zero_inputs,
            zero_weights,
            distributed=True,
        )
        zero_rank_loss.backward()
        torch.save(
            {
                "loss": loss.detach(),
                "gradient": first_gradient,
                "zero_rank_loss": zero_rank_loss.detach(),
                "zero_rank_gradient": model.module.weight.grad.detach(),
            },
            Path(output) / f"rank_{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_diversity_matches_single_process_concat(
    tmp_path: Path,
) -> None:
    world_size = 2
    mp.spawn(
        _worker,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    expected_model = _DiversityProjection()
    inputs0, weights0 = _rank_inputs(0)
    inputs1, weights1 = _rank_inputs(1)
    expected = expected_model(
        torch.cat([inputs0, inputs1]),
        torch.cat([weights0, weights1]),
        distributed=False,
    )
    expected.backward()
    expected_zero_rank_model = _DiversityProjection()
    expected_zero_rank_inputs, expected_zero_rank_weights = _rank_inputs(1)
    expected_zero_rank = expected_zero_rank_model(
        expected_zero_rank_inputs,
        expected_zero_rank_weights,
        distributed=False,
    )
    expected_zero_rank.backward()
    for rank in range(world_size):
        payload = torch.load(
            tmp_path / f"rank_{rank}.pt",
            map_location="cpu",
            weights_only=True,
        )
        torch.testing.assert_close(payload["loss"], expected.detach())
        torch.testing.assert_close(
            payload["gradient"],
            expected_model.weight.grad,
        )
        torch.testing.assert_close(
            payload["zero_rank_loss"],
            expected_zero_rank.detach(),
        )
        torch.testing.assert_close(
            payload["zero_rank_gradient"],
            expected_zero_rank_model.weight.grad,
        )
