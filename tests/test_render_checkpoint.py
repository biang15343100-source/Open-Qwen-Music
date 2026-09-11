import json
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from open_qwen_music.render.checkpoint import (
    FORMAT_VERSION,
    RenderCheckpointApplyError,
    RenderCheckpointError,
    inspect_render_checkpoint,
    load_render_checkpoint,
    read_render_checkpoint_state,
    save_render_checkpoint,
)
from open_qwen_music.common.checkpoint import file_sha256
from open_qwen_music.render.trainer_common import Muon


class StateObject:
    def __init__(self, value: int) -> None:
        self.value = value

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.value = int(state["value"])


class MutatingFailModel(torch.nn.Linear):
    def load_state_dict(
        self,
        state_dict,
        strict: bool = True,
        assign: bool = False,
    ):
        del state_dict, strict, assign
        with torch.no_grad():
            self.weight.fill_(123.0)
        raise RuntimeError("synthetic late apply failure")


def _objects() -> tuple[
    dict[str, torch.nn.Module],
    dict[str, torch.optim.Optimizer],
    dict[str, object],
]:
    models = {
        "encoder": torch.nn.Linear(4, 3),
        "decoder": torch.nn.Linear(3, 4),
    }
    optimizers = {
        "encoder": torch.optim.AdamW(models["encoder"].parameters(), lr=1e-3),
        "decoder": torch.optim.SGD(models["decoder"].parameters(), lr=2e-3),
    }
    schedulers = {
        name: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        for name, optimizer in optimizers.items()
    }

    loss = models["encoder"](torch.ones(2, 4)).square().mean()
    loss.backward()
    optimizers["encoder"].step()
    optimizers["encoder"].zero_grad(set_to_none=True)
    return models, optimizers, schedulers


def _revisions() -> dict[str, str]:
    return {
        "data_release_sha256": "a" * 64,
        "stft_config_sha256": "b" * 64,
    }


def test_checkpoint_exact_multi_object_resume_and_rng(tmp_path: Path) -> None:
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    models, optimizers, schedulers = _objects()
    ema = {"decoder": StateObject(7)}
    scaler = StateObject(8)
    sampler = StateObject(9)
    path = tmp_path / "render.pt"
    save_render_checkpoint(
        path,
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        ema=ema,
        scaler=scaler,
        sampler=sampler,
        config={"stage": 2, "candidate": "v1"},
        upstream_revisions=_revisions(),
        required_upstream_sha_keys=tuple(_revisions()),
        global_step=123,
        epoch=4,
        batches_consumed=567,
        training_audio_seconds=89.25,
        extra_state={
            "validation_state": {
                "best_valid_loss": 0.25,
                "valid_without_improvement": 2,
            }
        },
    )
    expected_rng = (random.random(), np.random.rand(), torch.rand(4))
    expected_parameters = {
        name: {key: value.clone() for key, value in model.state_dict().items()}
        for name, model in models.items()
    }
    with torch.no_grad():
        for model in models.values():
            for parameter in model.parameters():
                parameter.zero_()
    ema["decoder"].value = -1
    scaler.value = -1
    sampler.value = -1
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    resume = load_render_checkpoint(
        path,
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        ema=ema,
        scaler=scaler,
        sampler=sampler,
        expected_config={"stage": 2, "candidate": "v1"},
        expected_upstream_revisions=_revisions(),
        required_upstream_sha_keys=tuple(_revisions()),
    )
    assert resume.global_step == 123
    assert resume.epoch == 4
    assert resume.batches_consumed == 567
    assert resume.training_audio_seconds == 89.25
    assert resume.extra_state["validation_state"] == {
        "best_valid_loss": 0.25,
        "valid_without_improvement": 2,
    }
    assert ema["decoder"].value == 7
    assert scaler.value == 8
    assert sampler.value == 9
    for name, model in models.items():
        for key, value in model.state_dict().items():
            assert torch.equal(value, expected_parameters[name][key])
    actual_rng = (random.random(), np.random.rand(), torch.rand(4))
    assert actual_rng[0] == expected_rng[0]
    assert actual_rng[1] == expected_rng[1]
    assert torch.equal(actual_rng[2], expected_rng[2])
    metadata = inspect_render_checkpoint(path)
    assert metadata["format_version"] == FORMAT_VERSION
    assert metadata["model_names"] == ["decoder", "encoder"]
    assert not list(tmp_path.glob("*.tmp.*"))


def test_checkpoint_rejects_revision_mismatch_and_tampering(
    tmp_path: Path,
) -> None:
    models, optimizers, schedulers = _objects()
    path = tmp_path / "render.pt"
    save_render_checkpoint(
        path,
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        config={"stage": 1},
        upstream_revisions=_revisions(),
        required_upstream_sha_keys=tuple(_revisions()),
        global_step=1,
    )
    mismatch = dict(_revisions())
    mismatch["data_release_sha256"] = "c" * 64
    with pytest.raises(RenderCheckpointError, match="upstream revision or SHA"):
        load_render_checkpoint(
            path,
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            expected_config={"stage": 1},
            expected_upstream_revisions=mismatch,
            required_upstream_sha_keys=tuple(_revisions()),
        )

    state = torch.load(path, map_location="cpu", weights_only=False)
    state["upstream_revisions"]["data_release_sha256"] = "d" * 64
    torch.save(state, path)
    with pytest.raises(RenderCheckpointError, match="SHA|  "):
        load_render_checkpoint(path, resume=False)
    with pytest.raises(RenderCheckpointError, match="revision manifest"):
        load_render_checkpoint(path, resume=False, strict_sidecar=False)


def test_checkpoint_strict_failures_are_not_silently_accepted(
    tmp_path: Path,
) -> None:
    models, optimizers, schedulers = _objects()
    path = tmp_path / "render.pt"
    save_render_checkpoint(
        path,
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        config={"stage": 2},
        upstream_revisions=_revisions(),
        required_upstream_sha_keys=tuple(_revisions()),
        global_step=2,
    )
    with pytest.raises(RenderCheckpointError, match="config"):
        load_render_checkpoint(
            path,
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            expected_config={"stage": 3},
        )
    with pytest.raises(RenderCheckpointError, match="optimizer names do not match"):
        load_render_checkpoint(
            path,
            models=models,
            optimizers={"encoder": optimizers["encoder"]},
            schedulers=schedulers,
        )
    assert path.is_symlink()
    resolved = path.resolve()
    resolved.with_suffix(resolved.suffix + ".sha256.json").unlink()
    with pytest.raises(RenderCheckpointError, match="sidecar"):
        load_render_checkpoint(path, resume=False)


def test_checkpoint_public_pointer_switches_between_immutable_generations(
    tmp_path: Path,
) -> None:
    models, optimizers, schedulers = _objects()
    path = tmp_path / "last.pt"
    common = {
        "models": models,
        "optimizers": optimizers,
        "schedulers": schedulers,
        "config": {"stage": 1},
        "upstream_revisions": _revisions(),
        "required_upstream_sha_keys": tuple(_revisions()),
    }
    save_render_checkpoint(path, global_step=1, **common)
    first = path.resolve()
    assert first.is_file()
    assert first.with_suffix(first.suffix + ".sha256.json").is_file()
    save_render_checkpoint(path, global_step=2, **common)
    second = path.resolve()
    assert second != first
    assert first.is_file()
    assert inspect_render_checkpoint(path)["training_state"]["global_step"] == 2


def test_checkpoint_load_uses_launcher_prestaged_render_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models, optimizers, schedulers = _objects()
    path = tmp_path / "last.pt"
    save_render_checkpoint(
        path,
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        config={"stage": 2},
        upstream_revisions=_revisions(),
        required_upstream_sha_keys=tuple(_revisions()),
        global_step=2,
    )
    canonical = path.resolve()
    cache_dir = tmp_path / "cache"
    cached = cache_dir / "render.pt"
    cache_dir.mkdir()
    shutil.copyfile(canonical, cached)
    shutil.copyfile(
        canonical.with_suffix(canonical.suffix + ".sha256.json"),
        cached.with_suffix(cached.suffix + ".sha256.json"),
    )
    monkeypatch.setenv("OQM_PRESTAGED_CHECKPOINT", str(path))
    monkeypatch.setenv("OQM_PRESTAGED_CHECKPOINT_CANONICAL", str(canonical))
    monkeypatch.setenv("OQM_PRESTAGED_CHECKPOINT_LOCAL", str(cached))

    state = read_render_checkpoint_state(path)

    assert state["training_state"]["global_step"] == 2
    assert cached != canonical
    assert Path(f"{cached}.sha256.json").is_file()


def test_checkpoint_init_allows_only_explicit_config_field_replacement(
    tmp_path: Path,
) -> None:
    models, optimizers, schedulers = _objects()
    path = tmp_path / "render.pt"
    source_config = {"flow": {"mean": 1.0, "std": 1.0}}
    save_render_checkpoint(
        path,
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        config=source_config,
        upstream_revisions=_revisions(),
        required_upstream_sha_keys=tuple(_revisions()),
        global_step=2,
    )
    load_render_checkpoint(
        path,
        expected_config={"flow": {"mean": 0.0, "std": 1.0}},
        expected_config_sections=("flow",),
        expected_config_replacements=("flow.mean",),
        resume=False,
    )
    with pytest.raises(RenderCheckpointError, match="config section"):
        load_render_checkpoint(
            path,
            expected_config={"flow": {"mean": 0.0, "std": 0.5}},
            expected_config_sections=("flow",),
            expected_config_replacements=("flow.mean",),
            resume=False,
        )
    with pytest.raises(RenderCheckpointError, match="replacement path does not exist"):
        load_render_checkpoint(
            path,
            expected_config={"flow": {"mean": 0.0, "std": 1.0}},
            expected_config_sections=("flow",),
            expected_config_replacements=("flow.missing",),
            resume=False,
        )


def test_checkpoint_single_read_binds_expected_file_sha(tmp_path: Path) -> None:
    models, optimizers, schedulers = _objects()
    path = tmp_path / "last.pt"
    common = {
        "models": models,
        "optimizers": optimizers,
        "schedulers": schedulers,
        "config": {"stage": 1},
        "upstream_revisions": _revisions(),
        "required_upstream_sha_keys": tuple(_revisions()),
    }
    save_render_checkpoint(path, global_step=1, **common)
    first_sha = file_sha256(path.resolve())
    first = read_render_checkpoint_state(
        path,
        expected_checkpoint_sha256=first_sha,
    )
    assert first["training_state"]["global_step"] == 1

    save_render_checkpoint(path, global_step=2, **common)
    second_sha = file_sha256(path.resolve())
    assert second_sha != first_sha
    with pytest.raises(RenderCheckpointError, match="content SHA does not match"):
        read_render_checkpoint_state(
            path,
            expected_checkpoint_sha256=first_sha,
        )
    second = read_render_checkpoint_state(
        path,
        expected_checkpoint_sha256=second_sha,
    )
    assert second["training_state"]["global_step"] == 2


def test_checkpoint_mmap_single_read_binds_expected_file_sha(tmp_path: Path) -> None:
    models, optimizers, schedulers = _objects()
    path = tmp_path / "last.pt"
    save_render_checkpoint(
        path,
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        config={"stage": 1},
        upstream_revisions=_revisions(),
        required_upstream_sha_keys=tuple(_revisions()),
        global_step=1,
    )
    digest = file_sha256(path.resolve())
    state = read_render_checkpoint_state(
        path,
        expected_checkpoint_sha256=digest,
        mmap=True,
    )
    assert state["training_state"]["global_step"] == 1


@pytest.mark.parametrize("mmap", [False, True])
def test_checkpoint_loads_verified_private_snapshot_if_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mmap: bool,
) -> None:
    first_model = torch.nn.Linear(2, 2)
    second_model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        first_model.weight.fill_(1.0)
        first_model.bias.fill_(1.0)
        second_model.weight.fill_(9.0)
        second_model.bias.fill_(9.0)
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    common = {
        "config": {"stage": 1},
        "upstream_revisions": _revisions(),
        "required_upstream_sha_keys": tuple(_revisions()),
        "global_step": 1,
    }
    save_render_checkpoint(first, models={"model": first_model}, **common)
    save_render_checkpoint(second, models={"model": second_model}, **common)
    first_generation = first.resolve()
    second_generation = second.resolve()
    first_sha = file_sha256(first_generation)
    real_load = torch.load
    replaced = False

    def replace_source_then_load(*args, **kwargs):
        nonlocal replaced
        if not replaced:
            shutil.copyfile(second_generation, first_generation)
            replaced = True
        return real_load(*args, **kwargs)

    monkeypatch.setattr(
        "open_qwen_music.render.checkpoint.torch.load",
        replace_source_then_load,
    )
    state = read_render_checkpoint_state(
        first,
        expected_checkpoint_sha256=first_sha,
        mmap=mmap,
    )
    assert replaced
    assert torch.equal(
        state["models"]["model"]["weight"],
        torch.ones_like(state["models"]["model"]["weight"]),
    )


def test_checkpoint_component_is_checked_before_loading_models(tmp_path: Path) -> None:
    models, optimizers, schedulers = _objects()
    path = tmp_path / "wrong-component.pt"
    save_render_checkpoint(
        path,
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        config={"stage": 1},
        upstream_revisions=_revisions(),
        required_upstream_sha_keys=tuple(_revisions()),
        global_step=1,
        component="spec_vae",
    )
    before = {
        name: {key: value.clone() for key, value in model.state_dict().items()}
        for name, model in models.items()
    }
    with pytest.raises(RenderCheckpointError, match="component"):
        load_render_checkpoint(
            path,
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            expected_component="render_dit",
        )
    for name, model in models.items():
        for key, value in model.state_dict().items():
            assert torch.equal(value, before[name][key])


def test_checkpoint_resume_preflight_does_not_mutate_model_on_optimizer_error(
    tmp_path: Path,
) -> None:
    source_models, source_optimizers, source_schedulers = _objects()
    path = tmp_path / "resume.pt"
    save_render_checkpoint(
        path,
        models=source_models,
        optimizers=source_optimizers,
        schedulers=source_schedulers,
        config={"stage": 1},
        upstream_revisions=_revisions(),
        required_upstream_sha_keys=tuple(_revisions()),
        global_step=1,
        component="render_dit",
    )
    target_models, target_optimizers, target_schedulers = _objects()
    before = {
        name: {key: value.clone() for key, value in model.state_dict().items()}
        for name, model in target_models.items()
    }
    with pytest.raises(RenderCheckpointError, match="optimizer names do not match"):
        load_render_checkpoint(
            path,
            models=target_models,
            optimizers={"encoder": target_optimizers["encoder"]},
            schedulers=target_schedulers,
            expected_component="render_dit",
        )
    for name, model in target_models.items():
        for key, value in model.state_dict().items():
            assert torch.equal(value, before[name][key])


def test_checkpoint_rejects_optimizer_tensor_shape_before_model_mutation(
    tmp_path: Path,
) -> None:
    source_models, source_optimizers, source_schedulers = _objects()
    path = tmp_path / "bad-optimizer-shape.pt"
    save_render_checkpoint(
        path,
        models=source_models,
        optimizers=source_optimizers,
        schedulers=source_schedulers,
        config={"stage": 1},
        upstream_revisions=_revisions(),
        required_upstream_sha_keys=tuple(_revisions()),
        global_step=1,
        component="render_dit",
    )
    generation = path.resolve()
    payload = torch.load(generation, map_location="cpu", weights_only=False)
    optimizer_state = next(iter(payload["optimizers"]["encoder"]["state"].values()))
    optimizer_state["exp_avg"] = torch.zeros(1)
    torch.save(payload, generation)
    sidecar_path = generation.with_suffix(generation.suffix + ".sha256.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["checkpoint_sha256"] = file_sha256(generation)
    sidecar["checkpoint_size_bytes"] = generation.stat().st_size
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    target_models, target_optimizers, target_schedulers = _objects()
    before = {
        name: {key: value.clone() for key, value in model.state_dict().items()}
        for name, model in target_models.items()
    }
    with pytest.raises(RenderCheckpointError, match="tensor state shape"):
        load_render_checkpoint(
            path,
            models=target_models,
            optimizers=target_optimizers,
            schedulers=target_schedulers,
            expected_component="render_dit",
        )
    for name, model in target_models.items():
        for key, value in model.state_dict().items():
            assert torch.equal(value, before[name][key])


def test_checkpoint_allows_only_ear_v2_flattened_muon_momentum(
    tmp_path: Path,
) -> None:
    source = torch.nn.Conv2d(2, 3, kernel_size=(2, 2), bias=False)
    source_optimizer = Muon(
        [{"params": [source.weight], "orthogonalize": True}],
        lr=1.0e-4,
        implementation="ear_v2_source_v1",
    )
    source(torch.ones(1, 2, 3, 3)).square().mean().backward()
    source_optimizer.step()
    momentum = source_optimizer.state[source.weight]["momentum_buffer"]
    assert momentum.shape == (source.weight.shape[0], source.weight[0].numel())

    path = tmp_path / "ear-v2-muon.pt"
    save_render_checkpoint(
        path,
        models={"model": source},
        optimizers={"optimizer": source_optimizer},
        config={"stage": 1},
        upstream_revisions=_revisions(),
        required_upstream_sha_keys=tuple(_revisions()),
        global_step=1,
        component="spec_vae",
    )
    target = torch.nn.Conv2d(2, 3, kernel_size=(2, 2), bias=False)
    target_optimizer = Muon(
        [{"params": [target.weight], "orthogonalize": True}],
        lr=1.0e-4,
        implementation="ear_v2_source_v1",
    )
    load_render_checkpoint(
        path,
        models={"model": target},
        optimizers={"optimizer": target_optimizer},
        expected_config={"stage": 1},
        expected_upstream_revisions=_revisions(),
        required_upstream_sha_keys=tuple(_revisions()),
        expected_component="spec_vae",
    )
    assert target_optimizer.state[target.weight]["momentum_buffer"].shape == momentum.shape
    assert torch.equal(
        target_optimizer.state[target.weight]["momentum_buffer"],
        momentum,
    )


def test_checkpoint_rejects_wrong_layout_with_same_numel_for_ear_v2_muon(
    tmp_path: Path,
) -> None:
    source = torch.nn.Conv2d(2, 3, kernel_size=(2, 2), bias=False)
    source_optimizer = Muon(
        [{"params": [source.weight], "orthogonalize": True}],
        lr=1.0e-4,
        implementation="ear_v2_source_v1",
    )
    source(torch.ones(1, 2, 3, 3)).square().mean().backward()
    source_optimizer.step()
    path = tmp_path / "bad-ear-v2-muon-layout.pt"
    save_render_checkpoint(
        path,
        models={"model": source},
        optimizers={"optimizer": source_optimizer},
        config={"stage": 1},
        upstream_revisions=_revisions(),
        required_upstream_sha_keys=tuple(_revisions()),
        global_step=1,
        component="spec_vae",
    )
    generation = path.resolve()
    payload = torch.load(generation, map_location="cpu", weights_only=False)
    state = next(iter(payload["optimizers"]["optimizer"]["state"].values()))
    state["momentum_buffer"] = state["momentum_buffer"].reshape(1, 3, 8)
    torch.save(payload, generation)
    sidecar_path = generation.with_suffix(generation.suffix + ".sha256.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["checkpoint_sha256"] = file_sha256(generation)
    sidecar["checkpoint_size_bytes"] = generation.stat().st_size
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    target = torch.nn.Conv2d(2, 3, kernel_size=(2, 2), bias=False)
    target_optimizer = Muon(
        [{"params": [target.weight], "orthogonalize": True}],
        lr=1.0e-4,
        implementation="ear_v2_source_v1",
    )
    before = {key: value.clone() for key, value in target.state_dict().items()}
    with pytest.raises(RenderCheckpointError, match="tensor state shape"):
        load_render_checkpoint(
            path,
            models={"model": target},
            optimizers={"optimizer": target_optimizer},
            expected_config={"stage": 1},
            expected_upstream_revisions=_revisions(),
            required_upstream_sha_keys=tuple(_revisions()),
            expected_component="spec_vae",
        )
    for key, value in target.state_dict().items():
        assert torch.equal(value, before[key])


def test_checkpoint_pre_apply_validator_runs_before_state_mutation(
    tmp_path: Path,
) -> None:
    source_models, source_optimizers, source_schedulers = _objects()
    path = tmp_path / "pre-apply.pt"
    save_render_checkpoint(
        path,
        models=source_models,
        optimizers=source_optimizers,
        schedulers=source_schedulers,
        config={"stage": 1},
        upstream_revisions=_revisions(),
        required_upstream_sha_keys=tuple(_revisions()),
        global_step=1,
        component="render_dit",
    )
    target_models, target_optimizers, target_schedulers = _objects()
    before = {
        name: {key: value.clone() for key, value in model.state_dict().items()}
        for name, model in target_models.items()
    }

    def reject(_state) -> None:
        raise RuntimeError("component metadata mismatch")

    with pytest.raises(RenderCheckpointError, match="pre-apply"):
        load_render_checkpoint(
            path,
            models=target_models,
            optimizers=target_optimizers,
            schedulers=target_schedulers,
            expected_component="render_dit",
            pre_apply_validator=reject,
        )
    for name, model in target_models.items():
        for key, value in model.state_dict().items():
            assert torch.equal(value, before[name][key])


def test_checkpoint_late_apply_failure_is_explicitly_nonrecoverable(
    tmp_path: Path,
) -> None:
    source = torch.nn.Linear(4, 3)
    path = tmp_path / "late-apply.pt"
    save_render_checkpoint(
        path,
        models={"model": source},
        config={"stage": 1},
        upstream_revisions=_revisions(),
        required_upstream_sha_keys=tuple(_revisions()),
        global_step=1,
        component="render_dit",
    )
    target = MutatingFailModel(4, 3)
    with pytest.raises(
        RenderCheckpointApplyError,
        match="apply failed.*discard the target objects",
    ):
        load_render_checkpoint(
            path,
            models={"model": target},
            resume=False,
            expected_component="render_dit",
        )
    assert torch.count_nonzero(target.weight == 123.0) == target.weight.numel()


def test_checkpoint_rejects_revision_without_sha(tmp_path: Path) -> None:
    with pytest.raises(RenderCheckpointError, match="missing sha256"):
        save_render_checkpoint(
            tmp_path / "bad.pt",
            models=torch.nn.Linear(2, 2),
            config={},
            upstream_revisions={"tokenizer": {"revision": "v1"}},
            global_step=0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("global_step", True),
        ("epoch", 1.5),
        ("batches_consumed", -1),
        ("training_audio_seconds", float("nan")),
    ],
)
def test_checkpoint_rejects_invalid_progress_types(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    kwargs = {
        "path": tmp_path / "bad-progress.pt",
        "models": torch.nn.Linear(2, 2),
        "config": {},
        "upstream_revisions": _revisions(),
        "global_step": 0,
        "epoch": 0,
        "batches_consumed": 0,
        "training_audio_seconds": 0.0,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        save_render_checkpoint(**kwargs)


def test_checkpoint_rejects_nonfinite_optimizer_state(tmp_path: Path) -> None:
    models, optimizers, schedulers = _objects()
    optimizer_state = next(iter(optimizers["encoder"].state.values()))
    optimizer_state["exp_avg"].fill_(float("nan"))
    with pytest.raises(RenderCheckpointError, match="optimizers.*NaN/Inf"):
        save_render_checkpoint(
            tmp_path / "nonfinite-optimizer.pt",
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            config={"stage": 1},
            upstream_revisions=_revisions(),
            required_upstream_sha_keys=tuple(_revisions()),
            global_step=1,
        )


def test_checkpoint_allows_uninitialized_best_valid_loss_sentinel(
    tmp_path: Path,
) -> None:
    path = tmp_path / "valid-sentinel.pt"
    save_render_checkpoint(
        path,
        models=torch.nn.Linear(2, 2),
        config={"stage": 1},
        upstream_revisions=_revisions(),
        required_upstream_sha_keys=tuple(_revisions()),
        global_step=0,
        extra_state={
            "validation_state": {
                "best_valid_loss": float("inf"),
                "valid_without_improvement": 0,
            }
        },
    )
    state = read_render_checkpoint_state(path)
    assert state["extra_state"]["validation_state"]["best_valid_loss"] == float(
        "inf"
    )


def test_checkpoint_rejects_positive_infinity_outside_valid_sentinel(
    tmp_path: Path,
) -> None:
    with pytest.raises(RenderCheckpointError, match="extra_state.*NaN/Inf"):
        save_render_checkpoint(
            tmp_path / "bad-extra-state.pt",
            models=torch.nn.Linear(2, 2),
            config={"stage": 1},
            upstream_revisions=_revisions(),
            required_upstream_sha_keys=tuple(_revisions()),
            global_step=0,
            extra_state={"other_metric": float("inf")},
        )
