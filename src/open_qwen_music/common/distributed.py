
from __future__ import annotations

import json
import os
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


_TELEMETRY_GROUP: dist.ProcessGroup | None = None
_AUX_CUDA_GROUP: dist.ProcessGroup | None = None


# status(64) + action(128).
_RANK_STATUS_PROTOCOL_VERSION = 1
_RANK_STATUS_MAX_UTF8_BYTES = 64
_RANK_ACTION_MAX_UTF8_BYTES = 128
_RANK_STATUS_HEADER_BYTES = 6
_RANK_STATUS_WIRE_BYTES = (
    _RANK_STATUS_HEADER_BYTES
    + _RANK_STATUS_MAX_UTF8_BYTES
    + _RANK_ACTION_MAX_UTF8_BYTES
)


DEFAULT_DIST_TIMEOUT_SEC = 300.0
DEFAULT_TRACE_BUFFER_SIZE = 2000


def _configure_nccl_diagnostics() -> None:


    os.environ.setdefault("TORCH_FR_BUFFER_SIZE", str(DEFAULT_TRACE_BUFFER_SIZE))
    os.environ.setdefault(
        "TORCH_NCCL_TRACE_BUFFER_SIZE", str(DEFAULT_TRACE_BUFFER_SIZE)
    )
    os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    os.environ.setdefault("TORCH_NCCL_DESYNC_DEBUG", "1")
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "1")


def init_distributed(
    *, create_aux_cuda_group: bool = False
) -> tuple[int, int, int, torch.device]:
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
        timeout_sec = float(
            os.environ.get("OQM_DIST_TIMEOUT_SEC", DEFAULT_DIST_TIMEOUT_SEC)
        )
        timeout = timedelta(seconds=timeout_sec)
        if backend == "nccl":
            _configure_nccl_diagnostics()
        dist.init_process_group(
            backend=backend, init_method="env://", timeout=timeout, **kwargs
        )


        _TELEMETRY_GROUP = dist.new_group(backend="gloo", timeout=timeout)
        if device.type == "cuda" and create_aux_cuda_group:
            _AUX_CUDA_GROUP = dist.new_group(backend="nccl", timeout=timeout)
    return rank, local_rank, world_size, device


def assert_distributed_consensus(name: str, payload: Any) -> None:

    if not dist.is_initialized():
        return
    if _TELEMETRY_GROUP is None:
        raise RuntimeError("distributed telemetry group not initialized")
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


    encoded = canonical.encode("utf-8")
    length = torch.tensor(len(encoded), dtype=torch.int64)
    maximum = length.clone()
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=_TELEMETRY_GROUP)
    local_payload = torch.zeros(int(maximum.item()), dtype=torch.uint8)
    if encoded:
        local_payload[: len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8)
    gathered_lengths = [torch.empty_like(length) for _ in range(dist.get_world_size())]
    gathered_payloads = [
        torch.empty_like(local_payload) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered_lengths, length, group=_TELEMETRY_GROUP)
    dist.all_gather(gathered_payloads, local_payload, group=_TELEMETRY_GROUP)
    if all(
        int(item_length) == len(encoded)
        and torch.equal(item_payload[: len(encoded)], local_payload[: len(encoded)])
        for item_length, item_payload in zip(
            gathered_lengths, gathered_payloads, strict=True
        )
    ):
        return
    gathered: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, canonical, group=_TELEMETRY_GROUP)
    reference = gathered[0]
    mismatches = [
        {"rank": rank, "payload": value}
        for rank, value in enumerate(gathered)
        if value != reference
    ]
    if mismatches:
        raise RuntimeError(
            f"Startup state differs across ranks for {name}: "
            f"rank0={reference}; mismatches={mismatches[:8]}"
        )


def raise_if_rank0_error(error: str | None, *, action: str) -> None:

    if not dist.is_initialized():
        if error:
            raise RuntimeError(f"{action} failed:{error}")
        return
    if _TELEMETRY_GROUP is None:
        raise RuntimeError("distributed telemetry group not initialized")
    message = [error if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(message, src=0, group=_TELEMETRY_GROUP)
    if message[0]:
        raise RuntimeError(f"{action} failed:{message[0]}")


def gather_object_to_rank0(payload: Any) -> list[Any] | None:

    if not dist.is_initialized():
        return [payload]
    if _TELEMETRY_GROUP is None:
        raise RuntimeError("distributed telemetry group not initialized")
    gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(
        payload,
        object_gather_list=gathered,
        dst=0,
        group=_TELEMETRY_GROUP,
    )
    return gathered


def raise_if_any_rank_error(error: str | None, *, action: str) -> None:

    if not dist.is_initialized():
        if error:
            raise RuntimeError(f"{action} failed:{error}")
        return
    if _TELEMETRY_GROUP is None:
        raise RuntimeError("distributed telemetry group not initialized")
    failed = torch.tensor(1 if error else 0, dtype=torch.int64)
    dist.all_reduce(failed, op=dist.ReduceOp.SUM, group=_TELEMETRY_GROUP)
    if int(failed.item()) == 0:
        return
    gathered: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, error, group=_TELEMETRY_GROUP)
    failures = [
        f"rank{rank}={message}"
        for rank, message in enumerate(gathered)
        if message is not None
    ]
    raise RuntimeError(f"{action} failed:{'; '.join(failures)}")


def _rank_status_text(
    value: object,
    *,
    field: str,
    maximum: int,
) -> tuple[str, bytes, str | None]:

    if not isinstance(value, str):
        return (
            repr(value),
            b"",
            f"{field} must be a string, received {type(value).__name__}",
        )
    safe_value = value.encode("utf-8", errors="backslashreplace").decode("utf-8")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        return safe_value, b"", f"{field} is not valid UTF-8: {exc}"
    if len(encoded) > maximum:
        return (
            safe_value,
            b"",
            f"{field} UTF-8 length {len(encoded)} exceeds the protocol limit {maximum}",
        )
    return safe_value, encoded, None


def _rank_status_wire(
    *,
    status: object,
    error: object,
    action: object,
) -> tuple[torch.Tensor, dict[str, str | None]]:
    safe_status, encoded_status, status_error = _rank_status_text(
        status,
        field="status",
        maximum=_RANK_STATUS_MAX_UTF8_BYTES,
    )
    safe_action, encoded_action, action_error = _rank_status_text(
        action,
        field="action",
        maximum=_RANK_ACTION_MAX_UTF8_BYTES,
    )
    effective_errors = [value for value in (status_error, action_error) if value]
    if error is not None:
        safe_error = (
            error.encode("utf-8", errors="backslashreplace").decode("utf-8")
            if isinstance(error, str)
            else repr(error)
        )
        effective_errors.insert(0, safe_error)
    effective_error = "; ".join(effective_errors) if effective_errors else None

    wire = torch.zeros(_RANK_STATUS_WIRE_BYTES, dtype=torch.uint8, device="cpu")
    wire[0] = _RANK_STATUS_PROTOCOL_VERSION
    wire[1] = 1 if effective_error is not None else 0
    status_length = len(encoded_status)
    action_length = len(encoded_action)
    wire[2] = status_length & 0xFF
    wire[3] = status_length >> 8
    wire[4] = action_length & 0xFF
    wire[5] = action_length >> 8
    status_start = _RANK_STATUS_HEADER_BYTES
    action_start = status_start + _RANK_STATUS_MAX_UTF8_BYTES
    if encoded_status:
        wire[status_start : status_start + status_length] = torch.tensor(
            list(encoded_status), dtype=torch.uint8
        )
    if encoded_action:
        wire[action_start : action_start + action_length] = torch.tensor(
            list(encoded_action), dtype=torch.uint8
        )
    return wire, {
        "action": safe_action,
        "status": safe_status,
        "error": effective_error,
    }


def _decode_rank_status_wire(wire: torch.Tensor) -> tuple[str, str, str | None]:
    if (
        wire.device.type != "cpu"
        or wire.dtype != torch.uint8
        or wire.ndim != 1
        or wire.numel() != _RANK_STATUS_WIRE_BYTES
    ):
        return "", "", "invalid wire tensor"
    if int(wire[0]) != _RANK_STATUS_PROTOCOL_VERSION:
        return "", "", f"invalid wire version: {int(wire[0])}"
    if int(wire[1]) not in {0, 1}:
        return "", "", f"invalid wire error flag: {int(wire[1])}"
    status_length = int(wire[2]) | (int(wire[3]) << 8)
    action_length = int(wire[4]) | (int(wire[5]) << 8)
    if status_length > _RANK_STATUS_MAX_UTF8_BYTES:
        return "", "", f"invalid wire status length: {status_length}"
    if action_length > _RANK_ACTION_MAX_UTF8_BYTES:
        return "", "", f"invalid wire action length: {action_length}"
    status_start = _RANK_STATUS_HEADER_BYTES
    action_start = status_start + _RANK_STATUS_MAX_UTF8_BYTES
    status_padding = wire[
        status_start + status_length : status_start + _RANK_STATUS_MAX_UTF8_BYTES
    ]
    action_padding = wire[
        action_start + action_length : action_start + _RANK_ACTION_MAX_UTF8_BYTES
    ]
    if bool(torch.any(status_padding)) or bool(torch.any(action_padding)):
        return "", "", "wire padding is not zero"
    try:
        status = bytes(wire[status_start : status_start + status_length].tolist()).decode(
            "utf-8"
        )
        action = bytes(wire[action_start : action_start + action_length].tolist()).decode(
            "utf-8"
        )
    except UnicodeDecodeError as exc:
        return "", "", f"wire payload is not valid UTF-8: {exc}"
    return status, action, None


def synchronize_rank_status(
    *,
    status: str,
    error: str | None,
    action: str,
) -> str:

    wire, local_payload = _rank_status_wire(
        status=status,
        error=error,
        action=action,
    )
    if not dist.is_initialized():
        _, _, wire_error = _decode_rank_status_wire(wire)
        if local_payload["error"] is not None or wire_error is not None:
            raise RuntimeError(
                "Rank status synchronization failed: "
                + json.dumps(
                    {**local_payload, "wire_error": wire_error},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        return str(local_payload["status"])
    if _TELEMETRY_GROUP is None:
        raise RuntimeError("distributed telemetry group not initialized")

    world_size = dist.get_world_size(group=_TELEMETRY_GROUP)
    gathered_wires = [torch.empty_like(wire) for _ in range(world_size)]

    dist.all_gather(gathered_wires, wire, group=_TELEMETRY_GROUP)
    decoded = [_decode_rank_status_wire(value) for value in gathered_wires]
    reference_status, reference_action, _ = decoded[0]
    requires_details = any(
        wire_error is not None
        or int(gathered_wires[rank][1]) != 0
        or gathered_status != reference_status
        or gathered_action != reference_action
        for rank, (gathered_status, gathered_action, wire_error) in enumerate(decoded)
    )
    if not requires_details:
        return reference_status

    gathered_payloads: list[dict[str, str | None] | None] = [None] * world_size
    dist.all_gather_object(
        gathered_payloads,
        local_payload,
        group=_TELEMETRY_GROUP,
    )
    wire_issues = [
        {"rank": rank, "error": wire_error}
        for rank, (_, _, wire_error) in enumerate(decoded)
        if wire_error is not None
    ]
    raise RuntimeError(
        "Rank status synchronization failed across ranks: "
        + json.dumps(
            {
                "ranks": gathered_payloads,
                "wire_issues": wire_issues,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    reduced: dict[str, float] = {}
    for name, value in metrics.items():
        tensor = value.detach().float().cpu().clone()
        if dist.is_initialized():
            if _TELEMETRY_GROUP is None:
                raise RuntimeError("distributed telemetry group not initialized")
            dist.all_reduce(
                tensor, op=dist.ReduceOp.SUM, group=_TELEMETRY_GROUP
            )
            tensor /= dist.get_world_size()
        reduced[name] = float(tensor.item())
    return reduced


def reduce_scalar_sum(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64)
    if dist.is_initialized():
        if _TELEMETRY_GROUP is None:
            raise RuntimeError("distributed telemetry group not initialized")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=_TELEMETRY_GROUP)
    return float(tensor.item())


def reduce_tensor_sum(value: torch.Tensor) -> torch.Tensor:

    if not isinstance(value, torch.Tensor):
        raise TypeError("reduce_tensor_sum requires a tensor")
    tensor = value.detach().to(device="cpu", dtype=torch.float64).clone()
    if not torch.isfinite(tensor).all():
        raise ValueError("reduce_tensor_sum input contains NaN or Inf")
    if dist.is_initialized():
        if _TELEMETRY_GROUP is None:
            raise RuntimeError("distributed telemetry group not initialized")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=_TELEMETRY_GROUP)
    return tensor


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
        if _AUX_CUDA_GROUP is not None:
            dist.destroy_process_group(_AUX_CUDA_GROUP)
        if _TELEMETRY_GROUP is not None:
            dist.destroy_process_group(_TELEMETRY_GROUP)
        dist.destroy_process_group()
    _TELEMETRY_GROUP = None
    _AUX_CUDA_GROUP = None
