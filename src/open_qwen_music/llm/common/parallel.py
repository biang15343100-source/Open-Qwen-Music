
from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


PARALLEL_STRATEGIES = ("ddp", "fsdp2")


def resolve_strategy(config: dict[str, Any], *, world_size: int) -> str:
    train_section = dict(config.get("train", {}) or {})
    strategy = str(train_section.get("parallel_strategy", "ddp")).lower()
    if strategy not in PARALLEL_STRATEGIES:
        raise ValueError(
            f"Unknown train.parallel_strategy={strategy!r}; expected one of "
            f"{list(PARALLEL_STRATEGIES)}"
        )
    if world_size <= 1:
        return "ddp"
    return strategy


def validate_precision(config: dict[str, Any], strategy: str) -> None:
    if strategy != "fsdp2":
        return
    dtype = str((config.get("model", {}) or {}).get("dtype", "bf16")).lower()
    if dtype != "fp32":
        raise ValueError(
            f"train.parallel_strategy=fsdp2 requires model.dtype=fp32 (got {dtype!r}). "
            "Sharded parameters retain their load dtype, so loading bf16 would make the "
            "optimizer update bf16 weights. Compute precision is controlled by "
            "train.fsdp_param_dtype, which defaults to bf16."
        )


def wrap_model(
    model: nn.Module,
    *,
    strategy: str,
    config: dict[str, Any],
    device: torch.device,
    local_rank: int,
    world_size: int,
) -> nn.Module:
    if world_size <= 1:
        return model
    train_section = dict(config.get("train", {}) or {})
    if strategy == "ddp":
        return DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            bucket_cap_mb=float(train_section.get("ddp_bucket_cap_mb", 25.0)),
            gradient_as_bucket_view=bool(
                train_section.get("ddp_gradient_as_bucket_view", True)
            ),
            static_graph=bool(train_section.get("ddp_static_graph", False)),
            find_unused_parameters=False,
        )
    return _wrap_fsdp2(model, train_section=train_section, device=device)


def _dtype_from_name(name: str) -> torch.dtype:
    table = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if name not in table:
        raise ValueError(f"Unknown dtype {name!r}; expected one of {list(table)}")
    return table[name]


def _wrap_fsdp2(
    model: nn.Module, *, train_section: dict[str, Any], device: torch.device
) -> nn.Module:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh(device.type, (dist.get_world_size(),))
    policy = MixedPrecisionPolicy(
        param_dtype=_dtype_from_name(str(train_section.get("fsdp_param_dtype", "bf16"))),


        reduce_dtype=_dtype_from_name(str(train_section.get("fsdp_reduce_dtype", "fp32"))),
    )
    reshard_after_forward = bool(train_section.get("fsdp_reshard_after_forward", True))


    blocks = _transformer_blocks(model)
    for block in blocks:
        fully_shard(
            block, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward
        )
    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward)
    return model


def _transformer_blocks(model: nn.Module) -> list[nn.Module]:
    inner = getattr(model, "backbone", model)
    for path in ("model.layers", "layers", "transformer.h"):
        current: nn.Module | None = inner
        for part in path.split("."):
            current = getattr(current, part, None)
            if current is None:
                break
        if isinstance(current, nn.ModuleList) and len(current) > 0:
            return list(current)
    raise RuntimeError(
        "FSDP2 could not find transformer blocks at model.layers, layers, or "
        "transformer.h. Root-only sharding would all-gather the entire model at once."
    )


@contextmanager
def gradient_sync(model: nn.Module, *, strategy: str, sync: bool) -> Iterator[None]:
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        yield
        return
    if strategy == "ddp":
        assert isinstance(model, DistributedDataParallel)
        with nullcontext() if sync else model.no_sync():
            yield
        return
    setter = getattr(model, "set_requires_gradient_sync", None)
    if setter is None:


        raise RuntimeError("FSDP2 strategy lacks set_requires_gradient_sync; wrapping failed")
    setter(sync)
    try:
        yield
    finally:
        setter(True)


def scale_gradients(model: nn.Module, factor: float) -> None:
    value = float(factor)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"Gradient scale must be finite and non-negative; got {factor}")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(value)


def clip_grad_norm(model: nn.Module, *, strategy: str, max_norm: float) -> float:
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    if hasattr(total, "full_tensor"):  # DTensor
        total = total.full_tensor()
    return float(total)


def is_sharded(model: nn.Module) -> bool:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover -  torch
        return False
    for parameter in model.parameters():
        return isinstance(parameter.data, DTensor) or isinstance(parameter, DTensor)
    return False


def _use_fsdp_path(model: nn.Module, strategy: str) -> bool:
    return (
        strategy == "fsdp2"
        and dist.is_initialized()
        and dist.get_world_size() > 1
        and is_sharded(model)
    )


def model_state_dict(model: nn.Module, *, strategy: str) -> dict[str, Any]:
    from .checkpoint import unwrap_model

    if not _use_fsdp_path(model, strategy):
        return unwrap_model(model).state_dict()
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    return get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )


def load_model_state_dict(
    model: nn.Module, state: dict[str, Any], *, strategy: str, strict: bool = False
) -> tuple[list[str], list[str]]:
    from .checkpoint import unwrap_model

    if not _use_fsdp_path(model, strategy):
        result = unwrap_model(model).load_state_dict(state, strict=strict)
        return list(result.missing_keys), list(result.unexpected_keys)
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    set_model_state_dict(
        model,
        state,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=strict),
    )
    return [], []


def optimizer_state_dict(
    model: nn.Module, optimizer: torch.optim.Optimizer, *, strategy: str
) -> dict[str, Any]:
    if not _use_fsdp_path(model, strategy):
        return optimizer.state_dict()
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
    )

    return get_optimizer_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )


def load_optimizer_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    *,
    strategy: str,
) -> None:
    if not _use_fsdp_path(model, strategy):
        optimizer.load_state_dict(state)
        return
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_optimizer_state_dict,
    )

    set_optimizer_state_dict(
        model,
        optimizer,
        state,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
    )
