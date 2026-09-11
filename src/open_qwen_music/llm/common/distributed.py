
from __future__ import annotations

import os

import torch
import torch.distributed as dist

_TELEMETRY_GROUP: dist.ProcessGroup | None = None
_AUX_CUDA_GROUP: dist.ProcessGroup | None = None


def init_distributed() -> tuple[int, int, int, torch.device]:
    global _TELEMETRY_GROUP, _AUX_CUDA_GROUP
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        kwargs = {"device_id": device} if device.type == "cuda" else {}
        dist.init_process_group(backend=backend, init_method="env://", **kwargs)
        _TELEMETRY_GROUP = dist.new_group(backend="gloo")
        if device.type == "cuda":
            _AUX_CUDA_GROUP = dist.new_group(backend="nccl")
    return rank, local_rank, world_size, device


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _telemetry_group() -> dist.ProcessGroup:
    if _TELEMETRY_GROUP is None:
        raise RuntimeError("distributed telemetry group not initialized")
    return _TELEMETRY_GROUP


def reduce_metrics(metrics: dict[str, torch.Tensor | float]) -> dict[str, float]:
    if not dist.is_initialized():
        return {
            name: float(value.detach().item() if isinstance(value, torch.Tensor) else value)
            for name, value in metrics.items()
        }

    group = _telemetry_group()
    gathered: list[list[str] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, sorted(metrics), group=group)
    names = sorted({name for names in gathered if names for name in names})
    if not names:
        return {}

    buffer = torch.zeros((2, len(names)), dtype=torch.float64)
    for position, name in enumerate(names):
        if name not in metrics:
            continue
        value = metrics[name]
        buffer[0, position] = float(
            value.detach().item() if isinstance(value, torch.Tensor) else value
        )
        buffer[1, position] = 1.0
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM, group=group)
    return {
        name: float(buffer[0, position] / max(buffer[1, position].item(), 1.0))
        for position, name in enumerate(names)
    }


def reduce_weighted_metrics(
    numerators: dict[str, float], denominators: dict[str, float]
) -> dict[str, float]:
    local_names = sorted(set(numerators) | set(denominators))
    if not dist.is_initialized():
        return {
            name: float(numerators.get(name, 0.0) / denominators[name])
            for name in local_names
            if denominators.get(name, 0.0) > 0.0
        }
    group = _telemetry_group()
    gathered: list[list[str] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_names, group=group)
    names = sorted({name for values in gathered if values for name in values})
    if not names:
        return {}
    buffer = torch.zeros((2, len(names)), dtype=torch.float64)
    for position, name in enumerate(names):
        buffer[0, position] = float(numerators.get(name, 0.0))
        buffer[1, position] = float(denominators.get(name, 0.0))
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM, group=group)
    return {
        name: float(buffer[0, position] / buffer[1, position])
        for position, name in enumerate(names)
        if buffer[1, position].item() > 0.0
    }


def reduce_scalar_sum(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=_telemetry_group())
    return float(tensor.item())


def all_gather_int(value: int) -> list[int]:
    if not dist.is_initialized():
        return [int(value)]
    group = _telemetry_group()
    source = torch.tensor([int(value)], dtype=torch.int64)
    gathered = [torch.zeros_like(source) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, source, group=group)
    return [int(item.item()) for item in gathered]


def all_gather_objects(value: object) -> list[object]:
    if not dist.is_initialized():
        return [value]
    gathered: list[object | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value, group=_telemetry_group())
    return list(gathered)


def require_equal_across_ranks(value: int, *, name: str) -> int:
    values = all_gather_int(value)
    if len(set(values)) != 1:
        raise RuntimeError(
            f"{name} differs across ranks: {values}. This can desynchronize collective "
            "operations and deadlock training. Fix sampling or batching, or set "
            "runtime split_parts=0."
        )
    return values[0]


def broadcast_object(obj: object, src: int = 0) -> object:
    if not dist.is_initialized():
        return obj
    holder = [obj]
    dist.broadcast_object_list(holder, src=src, group=_telemetry_group())
    return holder[0]


def auxiliary_cuda_group() -> dist.ProcessGroup | None:
    return _AUX_CUDA_GROUP


def barrier() -> None:
    if dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def cleanup_distributed() -> None:
    global _TELEMETRY_GROUP, _AUX_CUDA_GROUP
    if dist.is_initialized():
        dist.destroy_process_group()
    _TELEMETRY_GROUP = None
    _AUX_CUDA_GROUP = None
