
from __future__ import annotations

import importlib

import pytest
import torch


@pytest.fixture
def distributed_module(monkeypatch: pytest.MonkeyPatch):
    for key in (
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
        "TORCH_NCCL_TRACE_BUFFER_SIZE",
        "TORCH_FR_BUFFER_SIZE",
        "TORCH_NCCL_DUMP_ON_TIMEOUT",
        "TORCH_NCCL_DESYNC_DEBUG",
        "TORCH_NCCL_ENABLE_MONITORING",
        "OQM_DIST_TIMEOUT_SEC",
    ):
        monkeypatch.delenv(key, raising=False)
    module = importlib.import_module("open_qwen_music.common.distributed")
    return importlib.reload(module)


def test_default_collective_timeout_is_fail_fast(distributed_module) -> None:
    assert distributed_module.DEFAULT_DIST_TIMEOUT_SEC == pytest.approx(300.0)


def test_diagnostics_do_not_rewrite_heartbeat(
    distributed_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = distributed_module
    monkeypatch.setenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "120")
    module._configure_nccl_diagnostics()
    assert module.os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] == "120"


def test_diagnostics_enable_flight_recorder(distributed_module) -> None:
    module = distributed_module
    module._configure_nccl_diagnostics()
    assert module.os.environ["TORCH_FR_BUFFER_SIZE"] == "2000"
    assert module.os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "2000"
    assert module.os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "1"
    assert module.os.environ["TORCH_NCCL_DESYNC_DEBUG"] == "1"
    assert module.os.environ["TORCH_NCCL_ENABLE_MONITORING"] == "1"


def test_explicit_diagnostics_are_preserved(
    distributed_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = distributed_module
    monkeypatch.setenv("TORCH_NCCL_TRACE_BUFFER_SIZE", "99")
    monkeypatch.setenv("TORCH_FR_BUFFER_SIZE", "98")
    module._configure_nccl_diagnostics()
    assert module.os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "99"
    assert module.os.environ["TORCH_FR_BUFFER_SIZE"] == "98"


def test_startup_consensus_rejects_rank_drift(
    distributed_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = distributed_module
    monkeypatch.setattr(module, "_TELEMETRY_GROUP", object())
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)

    calls = 0

    def reduce(value, **kwargs):
        value.fill_(64)

    def gather(output, value, *, group):
        nonlocal calls
        calls += 1
        if calls == 1:
            output[0].copy_(value)
            output[1].copy_(value).add_(1)
        else:
            output[0].copy_(value)
            output[1].zero_()

    monkeypatch.setattr(module.dist, "all_reduce", reduce)
    monkeypatch.setattr(module.dist, "all_gather", gather)
    monkeypatch.setattr(
        module.dist,
        "all_gather_object",
        lambda output, value, *, group: output.__setitem__(
            slice(None), [value, '{"step": 2}']
        ),
    )
    with pytest.raises(RuntimeError, match="Startup state differs across ranks"):
        module.assert_distributed_consensus("resume", {"step": 1})


def test_startup_consensus_success_avoids_python_object_collective(
    distributed_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = distributed_module
    monkeypatch.setattr(module, "_TELEMETRY_GROUP", object())
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(module.dist, "all_reduce", lambda tensor, **kwargs: None)

    def gather(output, value, *, group):
        for item in output:
            item.copy_(value)

    monkeypatch.setattr(module.dist, "all_gather", gather)
    monkeypatch.setattr(
        module.dist,
        "all_gather_object",
        lambda *args, **kwargs: pytest.fail("matching state should not gather Python objects"),
    )

    module.assert_distributed_consensus("resume", {"state": "matching"})


def test_rank0_error_is_raised_on_every_rank(
    distributed_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = distributed_module
    monkeypatch.setattr(module, "_TELEMETRY_GROUP", object())
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 1)

    def broadcast(message, *, src, group):
        message[0] = "ENOSPC"

    monkeypatch.setattr(module.dist, "broadcast_object_list", broadcast)
    with pytest.raises(RuntimeError, match="ENOSPC"):
        module.raise_if_rank0_error(None, action="checkpoint")


def test_any_rank_error_success_uses_only_scalar_collective(
    distributed_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = distributed_module
    monkeypatch.setattr(module, "_TELEMETRY_GROUP", object())
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "all_reduce", lambda tensor, **kwargs: None)
    monkeypatch.setattr(
        module.dist,
        "all_gather_object",
        lambda *args, **kwargs: pytest.fail("success path should not gather Python objects"),
    )
    module.raise_if_any_rank_error(None, action="train step")


def test_any_rank_error_collects_failure_details(
    distributed_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = distributed_module
    monkeypatch.setattr(module, "_TELEMETRY_GROUP", object())
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)

    def reduce(tensor, **kwargs):
        tensor.fill_(1)

    monkeypatch.setattr(module.dist, "all_reduce", reduce)

    def gather(output, value, *, group):
        output[:] = [None, "boom"]

    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="rank1=boom"):
        module.raise_if_any_rank_error("local", action="train step")


def test_rank_status_uninitialized_validates_error_and_protocol(
    distributed_module,
) -> None:
    module = distributed_module
    assert (
        module.synchronize_rank_status(
            status="Batch",
            error=None,
            action="Render dataloader step=1",
        )
        == "Batch"
    )
    with pytest.raises(RuntimeError, match='"error": ""'):
        module.synchronize_rank_status(status="batch", error="", action="load")
    with pytest.raises(RuntimeError, match="protocol limit"):
        module.synchronize_rank_status(
            status="x" * 65,
            error=None,
            action="load",
        )


def test_rank_status_success_uses_one_fixed_tensor_collective(
    distributed_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = distributed_module
    telemetry_group = object()
    monkeypatch.setattr(module, "_TELEMETRY_GROUP", telemetry_group)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)

    def world_size(*, group):
        assert group is telemetry_group
        return 2

    calls = {"tensor": 0, "object": 0}

    def gather(output, value, *, group):
        assert group is telemetry_group
        assert value.dtype == torch.uint8
        assert value.device.type == "cpu"
        assert value.shape == (module._RANK_STATUS_WIRE_BYTES,)
        calls["tensor"] += 1
        for item in output:
            item.copy_(value)

    monkeypatch.setattr(module.dist, "get_world_size", world_size)
    monkeypatch.setattr(module.dist, "all_gather", gather)
    monkeypatch.setattr(
        module.dist,
        "all_reduce",
        lambda *args, **kwargs: pytest.fail("status synchronization must not call all_reduce"),
    )
    monkeypatch.setattr(
        module.dist,
        "all_gather_object",
        lambda *args, **kwargs: calls.__setitem__("object", calls["object"] + 1),
    )
    assert (
        module.synchronize_rank_status(
            status="batch",
            error=None,
            action="Render dataloader step=248",
        )
        == "batch"
    )
    assert calls == {"tensor": 1, "object": 0}


@pytest.mark.parametrize(
    ("remote_status", "remote_action"),
    [
        ("batch ", "Render dataloader step=248"),
        ("batch", "Render dataloader step=249"),
    ],
)
def test_rank_status_drift_uses_one_object_fallback(
    distributed_module,
    monkeypatch: pytest.MonkeyPatch,
    remote_status: str,
    remote_action: str,
) -> None:
    module = distributed_module
    telemetry_group = object()
    monkeypatch.setattr(module, "_TELEMETRY_GROUP", telemetry_group)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        module.dist,
        "get_world_size",
        lambda *, group: 2,
    )
    calls = {"tensor": 0, "object": 0}
    remote_wire, remote_payload = module._rank_status_wire(
        status=remote_status,
        error=None,
        action=remote_action,
    )

    def gather(output, value, *, group):
        calls["tensor"] += 1
        output[0].copy_(value)
        output[1].copy_(remote_wire)

    def gather_object(output, value, *, group):
        calls["object"] += 1
        output[:] = [value, remote_payload]

    monkeypatch.setattr(module.dist, "all_gather", gather)
    monkeypatch.setattr(module.dist, "all_gather_object", gather_object)
    with pytest.raises(RuntimeError, match="Rank status synchronization failed across ranks") as raised:
        module.synchronize_rank_status(
            status="batch",
            error=None,
            action="Render dataloader step=248",
        )
    assert remote_status in str(raised.value) or remote_action in str(raised.value)
    assert calls == {"tensor": 1, "object": 1}


def test_rank_status_local_protocol_error_still_enters_fixed_collective(
    distributed_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = distributed_module
    monkeypatch.setattr(module, "_TELEMETRY_GROUP", object())
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda *, group: 2)
    calls = {"tensor": 0, "object": 0}

    def gather(output, value, *, group):
        calls["tensor"] += 1
        for item in output:
            item.copy_(value)

    def gather_object(output, value, *, group):
        calls["object"] += 1
        output[:] = [value, value]

    monkeypatch.setattr(module.dist, "all_gather", gather)
    monkeypatch.setattr(module.dist, "all_gather_object", gather_object)
    with pytest.raises(RuntimeError, match="protocol limit"):
        module.synchronize_rank_status(
            status="x" * 65,
            error=None,
            action="load",
        )
    assert calls == {"tensor": 1, "object": 1}


def test_rank_status_tensor_timeout_does_not_attempt_fallback(
    distributed_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = distributed_module
    monkeypatch.setattr(module, "_TELEMETRY_GROUP", object())
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda *, group: 2)

    def failed_gather(*args, **kwargs):
        raise RuntimeError("Gloo 900s timeout")

    monkeypatch.setattr(module.dist, "all_gather", failed_gather)
    monkeypatch.setattr(
        module.dist,
        "all_gather_object",
        lambda *args, **kwargs: pytest.fail("must not fall back after tensor collective failure"),
    )
    with pytest.raises(RuntimeError, match="Gloo 900s timeout"):
        module.synchronize_rank_status(
            status="batch",
            error=None,
            action="load",
        )
