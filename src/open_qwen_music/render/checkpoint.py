
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import torch
from torch import nn

from open_qwen_music.common.checkpoint import (
    config_hash,
    file_sha256,
    unwrap_model,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover -  numpy.
    np = None


FORMAT_VERSION = "oqm.render.ckpt.v1"


class RenderCheckpointError(RuntimeError):
    pass


class RenderCheckpointApplyError(RenderCheckpointError):
    pass


@dataclass(frozen=True)
class RenderResumeState:
    config_hash: str
    config: dict[str, Any]
    global_step: int
    epoch: int
    batches_consumed: int
    training_audio_seconds: float
    sampler_state: dict[str, Any] | None
    distributed_state: dict[str, Any] | None
    upstream_revisions: dict[str, Any]
    required_upstream_sha_keys: tuple[str, ...]
    extra_state: dict[str, Any]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _validate_sha256(value: Any, *, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise RenderCheckpointError(f"{field} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RenderCheckpointError(f"{field} is not hexadecimal SHA-256") from exc


def validate_upstream_revisions(
    revisions: Mapping[str, Any],
    *,
    required_sha_keys: Sequence[str] = (),
) -> dict[str, Any]:

    if not isinstance(revisions, Mapping) or not revisions:
        raise RenderCheckpointError("upstream_revisions must not be empty")
    normalized = json.loads(json.dumps(revisions, sort_keys=True, default=str))
    for key in required_sha_keys:
        if key not in normalized:
            raise RenderCheckpointError(f"Missing required upstream SHA: {key}")
    sha_count = 0
    for name, value in normalized.items():
        if name.endswith("_sha256"):
            _validate_sha256(value, field=name)
            sha_count += 1
            continue
        if isinstance(value, Mapping):
            if "revision" in value and "sha256" not in value:
                raise RenderCheckpointError(
                    f"upstream {name} declares a revision but is missing sha256"
                )
            if "sha256" in value:
                _validate_sha256(value["sha256"], field=f"{name}.sha256")
                sha_count += 1
        elif name.endswith("_revision"):
            sha_key = name[: -len("_revision")] + "_sha256"
            if sha_key not in normalized:
                raise RenderCheckpointError(f"{name} must also provide {sha_key}")
    if sha_count == 0:
        raise RenderCheckpointError("upstream_revisions must contain at least one SHA-256")
    return normalized


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state() if np is not None else None,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if state.get("python") is not None:
        random.setstate(state["python"])
    if np is not None and state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        cuda_states = state["torch_cuda"]
        if len(cuda_states) != torch.cuda.device_count():
            raise RenderCheckpointError(
                "CUDA RNG device count does not match the current process; exact resume is unavailable"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def _named_objects(
    value: Any,
    *,
    default_name: str,
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        result = {default_name: value}
    if not all(isinstance(name, str) and name for name in result):
        raise ValueError("Checkpoint object name must be a non-empty string")
    return result


def _state_dicts(objects: Mapping[str, Any], *, unwrap: bool = False) -> dict[str, Any]:
    states: dict[str, Any] = {}
    for name, value in objects.items():
        target = (
            unwrap_model(value) if unwrap and isinstance(value, nn.Module) else value
        )
        if not hasattr(target, "state_dict"):
            raise TypeError(f"{name} No state_dict()")
        states[name] = target.state_dict()
    return states


def _assert_finite_models(states: Mapping[str, Mapping[str, Any]]) -> None:
    bad: list[str] = []
    for model_name, state in states.items():
        if not isinstance(model_name, str) or not model_name:
            raise RenderCheckpointError("Checkpoint model name must be a non-empty string")
        if not isinstance(state, Mapping):
            raise RenderCheckpointError(
                f"checkpoint model {model_name!r} state must be a mapping"
            )
        for key, value in state.items():
            if (
                isinstance(value, torch.Tensor)
                and (value.is_floating_point() or value.is_complex())
                and not torch.isfinite(value).all()
            ):
                bad.append(f"{model_name}.{key}")
    if bad:
        raise RenderCheckpointError(
            f"Refusing to save a Renderer checkpoint containing NaN/Inf: {bad[:8]}"
        )


_ALLOWED_POSITIVE_INFINITY_PATHS = frozenset(
    {


        "extra_state.validation_state.best_valid_loss",
    }
)


def _find_nonfinite_values(value: Any, *, prefix: str) -> list[str]:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(
            value
        ).all():
            return [prefix]
        return []
    if np is not None and isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            return [prefix]
        return []
    if isinstance(value, float):
        if (
            math.isinf(value)
            and value > 0.0
            and prefix in _ALLOWED_POSITIVE_INFINITY_PATHS
        ):
            return []
        return [] if math.isfinite(value) else [prefix]
    if isinstance(value, Mapping):
        return [
            item
            for name, nested in value.items()
            for item in _find_nonfinite_values(
                nested,
                prefix=f"{prefix}.{name}" if prefix else str(name),
            )
        ]
    if isinstance(value, (list, tuple)):
        return [
            item
            for index, nested in enumerate(value)
            for item in _find_nonfinite_values(
                nested,
                prefix=f"{prefix}[{index}]",
            )
        ]
    return []


def _assert_finite_state(value: Any, *, label: str) -> None:
    bad = _find_nonfinite_values(value, prefix=label)
    if bad:
        raise RenderCheckpointError(f"Render checkpoint {label}containsNaN/Inf: {bad[:8]}")


def _sampler_state(sampler: Any) -> dict[str, Any] | None:
    if sampler is None:
        return None
    if isinstance(sampler, Mapping):
        return dict(sampler)
    if hasattr(sampler, "state_dict"):
        return sampler.state_dict()
    raise TypeError("sampler must be a mapping or implement state_dict()")


def _integrity_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format_version": state.get("format_version"),
        "component": state.get("component"),
        "global_step": state.get("training_state", {}).get("global_step"),
        "config_hash": state.get("config_hash"),
        "upstream_revisions": state.get("upstream_revisions"),
        "required_upstream_sha_keys": state.get("required_upstream_sha_keys"),
        "revision_manifest_hash": state.get("revision_manifest_hash"),
        "model_names": sorted(state.get("models", {})),
        "optimizer_names": sorted(state.get("optimizers", {})),
        "scheduler_names": sorted(state.get("schedulers", {})),
    }


def _require_mapping_field(
    state: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    value = state.get(name)
    if not isinstance(value, Mapping):
        raise RenderCheckpointError(f"checkpoint {name} must be a mapping")
    return value


def _validate_non_negative_int(value: Any, *, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RenderCheckpointError(f"{field} must be a non-negative integer")


def _validate_checkpoint_structure(state: Mapping[str, Any]) -> None:
    required = {
        "format_version",
        "component",
        "models",
        "optimizers",
        "schedulers",
        "ema",
        "scaler",
        "config",
        "config_hash",
        "upstream_revisions",
        "required_upstream_sha_keys",
        "revision_manifest_hash",
        "training_state",
        "rng_states",
        "extra_state",
        "integrity_hash",
    }
    missing = required - set(state)
    if missing:
        raise RenderCheckpointError(f"checkpoint is missing field: {sorted(missing)}")
    for name in (
        "models",
        "optimizers",
        "schedulers",
        "ema",
        "config",
        "training_state",
        "rng_states",
        "extra_state",
    ):
        _require_mapping_field(state, name)
    if not isinstance(state.get("component"), str) or not state["component"]:
        raise RenderCheckpointError("checkpoint component must be a non-empty string")
    _validate_sha256(state.get("config_hash"), field="checkpoint.config_hash")
    _validate_sha256(
        state.get("revision_manifest_hash"),
        field="checkpoint.revision_manifest_hash",
    )
    _validate_sha256(state.get("integrity_hash"), field="checkpoint.integrity_hash")
    required_upstream = state.get("required_upstream_sha_keys")
    if (
        not isinstance(required_upstream, Sequence)
        or isinstance(required_upstream, (str, bytes))
        or not all(isinstance(value, str) and value for value in required_upstream)
        or len(set(required_upstream)) != len(required_upstream)
    ):
        raise RenderCheckpointError(
            "checkpoint required_upstream_sha_keys must be a sequence of unique strings"
        )
    training = _require_mapping_field(state, "training_state")
    for name in ("global_step", "epoch", "batches_consumed"):
        _validate_non_negative_int(
            training.get(name),
            field=f"checkpoint.training_state.{name}",
        )
    audio_seconds = training.get("training_audio_seconds")
    if (
        isinstance(audio_seconds, bool)
        or not isinstance(audio_seconds, (int, float))
        or not math.isfinite(float(audio_seconds))
        or float(audio_seconds) < 0
    ):
        raise RenderCheckpointError(
            "checkpoint.training_state.training_audio_seconds must be finite and non-negative"
        )
    _assert_finite_models(state["models"])
    for name in ("optimizers", "schedulers", "ema", "scaler", "extra_state"):
        _assert_finite_state(state.get(name), label=name)


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256.json")


def _generation_root(path: Path) -> Path:
    return path.parent / f".{path.name}.generations"


def _resolved_checkpoint_path(path: Path) -> Path:
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise RenderCheckpointError(f"checkpoint does not exist or its link is invalid: {path}") from exc


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_render_checkpoint(
    path: str | Path,
    *,
    models: Mapping[str, nn.Module] | nn.Module,
    optimizers: Mapping[str, torch.optim.Optimizer]
    | torch.optim.Optimizer
    | None = None,
    schedulers: Mapping[str, Any] | Any | None = None,
    ema: Mapping[str, Any] | Any | None = None,
    scaler: Any | None = None,
    sampler: Any | None = None,
    config: Mapping[str, Any],
    upstream_revisions: Mapping[str, Any],
    global_step: int,
    epoch: int = 0,
    batches_consumed: int = 0,
    training_audio_seconds: float = 0.0,
    component: str = "render",
    distributed_state: Mapping[str, Any] | None = None,
    required_upstream_sha_keys: Sequence[str] = (),
    extra_state: Mapping[str, Any] | None = None,
) -> None:

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(component, str) or not component:
        raise ValueError("component must be a non-empty string")
    for name, value in {
        "global_step": global_step,
        "epoch": epoch,
        "batches_consumed": batches_consumed,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if (
        isinstance(training_audio_seconds, bool)
        or not isinstance(training_audio_seconds, (int, float))
        or not math.isfinite(float(training_audio_seconds))
        or training_audio_seconds < 0
    ):
        raise ValueError("training_audio_seconds must be finite and non-negative")
    if not isinstance(config, Mapping):
        raise TypeError("checkpoint config must be a mapping")
    if distributed_state is not None and not isinstance(distributed_state, Mapping):
        raise TypeError("checkpoint distributed_state must be a mapping or None")
    if extra_state is not None and not isinstance(extra_state, Mapping):
        raise TypeError("checkpoint extra_state must be a mapping or None")
    normalized_revisions = validate_upstream_revisions(
        upstream_revisions, required_sha_keys=required_upstream_sha_keys
    )
    normalized_required_keys = sorted(set(required_upstream_sha_keys))
    named_models = _named_objects(models, default_name="model")
    if not named_models:
        raise ValueError("Renderer checkpoint requires at least one model")
    named_optimizers = _named_objects(optimizers, default_name="optimizer")
    named_schedulers = _named_objects(schedulers, default_name="scheduler")
    named_ema = _named_objects(ema, default_name="ema")
    model_states = _state_dicts(named_models, unwrap=True)
    _assert_finite_models(model_states)
    state: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "component": str(component),
        "models": model_states,
        "optimizers": _state_dicts(named_optimizers),
        "schedulers": _state_dicts(named_schedulers),
        "ema": _state_dicts(named_ema),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": dict(config),
        "config_hash": config_hash(dict(config)),
        "upstream_revisions": normalized_revisions,
        "required_upstream_sha_keys": normalized_required_keys,
        "revision_manifest_hash": _sha256_json(normalized_revisions),
        "training_state": {
            "global_step": int(global_step),
            "epoch": int(epoch),
            "batches_consumed": int(batches_consumed),
            "training_audio_seconds": float(training_audio_seconds),
            "sampler_state": _sampler_state(sampler),
            "distributed_state": dict(distributed_state)
            if distributed_state is not None
            else None,
        },
        "rng_states": capture_rng_state(),
        "extra_state": dict(extra_state or {}),
    }
    for name in ("optimizers", "schedulers", "ema", "scaler", "extra_state"):
        _assert_finite_state(state[name], label=name)
    state["integrity_hash"] = _sha256_json(_integrity_payload(state))
    generation_root = _generation_root(destination)
    generation_root.mkdir(parents=True, exist_ok=True)
    generation = generation_root / (
        f"step-{int(global_step):012d}-pid-{os.getpid()}-{os.urandom(8).hex()}"
    )
    generation.mkdir()
    checkpoint_path = generation / "checkpoint.pt"
    checkpoint_sidecar = _sidecar_path(checkpoint_path)
    try:
        with checkpoint_path.open("xb") as output:
            torch.save(state, output)
            output.flush()
            os.fsync(output.fileno())
        _atomic_json(
            checkpoint_sidecar,
            {
                "format_version": FORMAT_VERSION,
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                "global_step": global_step,
                "config_hash": state["config_hash"],
                "revision_manifest_hash": state["revision_manifest_hash"],
                "integrity_hash": state["integrity_hash"],
            },
        )
        for directory in (generation, generation_root, destination.parent):
            try:
                descriptor = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError:
                pass


        public_sidecar = _sidecar_path(destination)
        relative_sidecar = os.path.relpath(checkpoint_sidecar, destination.parent)
        temporary_sidecar_link = public_sidecar.with_name(
            f".{public_sidecar.name}.tmp-link.{os.getpid()}.{os.urandom(4).hex()}"
        )
        os.symlink(relative_sidecar, temporary_sidecar_link)
        os.replace(temporary_sidecar_link, public_sidecar)
        relative_checkpoint = os.path.relpath(checkpoint_path, destination.parent)
        temporary_link = destination.with_name(
            f".{destination.name}.tmp-link.{os.getpid()}.{os.urandom(4).hex()}"
        )
        os.symlink(relative_checkpoint, temporary_link)
        os.replace(temporary_link, destination)


        try:
            descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    except Exception:

        if not destination.is_symlink() or destination.resolve() != checkpoint_path:
            checkpoint_sidecar.unlink(missing_ok=True)
            checkpoint_path.unlink(missing_ok=True)
            try:
                generation.rmdir()
            except OSError:
                pass
        raise


def _load_checkpoint_payload(
    path: Path,
    *,
    strict_sidecar: bool,
    map_location: str | torch.device,
    expected_checkpoint_sha256: str | None = None,
    mmap: bool = False,
) -> Mapping[str, Any]:
    if not isinstance(mmap, bool):
        raise TypeError("checkpoint mmap must be a boolean")
    expected_digest: str | None = None
    if expected_checkpoint_sha256 is not None:
        _validate_sha256(
            expected_checkpoint_sha256,
            field="expected_checkpoint_sha256",
        )
        expected_digest = expected_checkpoint_sha256.lower()
    source = _resolved_checkpoint_path(path)
    sidecar_path = _sidecar_path(source)
    prestaged_canonical = os.environ.get("OQM_PRESTAGED_CHECKPOINT_CANONICAL")
    prestaged_local = os.environ.get("OQM_PRESTAGED_CHECKPOINT_LOCAL")
    if prestaged_canonical and prestaged_local:
        requested = path.resolve(strict=True)
        declared = Path(prestaged_canonical)
        if requested == declared:
            source = Path(prestaged_local)
            if not source.is_file():
                raise RenderCheckpointError(
                    f"declared local checkpoint does not exist: {source}"
                )
            sidecar_path = _sidecar_path(source)
    sidecar: Mapping[str, Any] | None = None
    if strict_sidecar:
        if not sidecar_path.is_file():
            raise RenderCheckpointError(f"strict checkpoint is missing sidecar: {sidecar_path}")
        try:
            sidecar_bytes = sidecar_path.read_bytes()
            value = json.loads(sidecar_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RenderCheckpointError("checkpoint sidecar cannot be parsed") from exc
        if not isinstance(value, Mapping):
            raise RenderCheckpointError("checkpoint sidecar must be a mapping")
        sidecar = value
        if sidecar.get("format_version") != FORMAT_VERSION:
            raise RenderCheckpointError("checkpoint sidecar format_version is incompatible with")
        _validate_sha256(
            sidecar.get("checkpoint_sha256"),
            field="checkpoint sidecar.checkpoint_sha256",
        )
    snapshot_directory = os.environ.get("OQM_CHECKPOINT_SNAPSHOT_DIR")
    snapshot_fd, snapshot_name = tempfile.mkstemp(
        prefix="oqm-render-checkpoint-",
        suffix=".pt",
        dir=snapshot_directory or None,
    )
    snapshot_path = Path(snapshot_name)
    try:
        digest = hashlib.sha256()
        copied_size = 0
        try:

            with (
                os.fdopen(snapshot_fd, "wb") as snapshot,
                source.open("rb") as source_handle,
            ):
                opened_stat = os.fstat(source_handle.fileno())
                while block := source_handle.read(8 << 20):
                    digest.update(block)
                    snapshot.write(block)
                    copied_size += len(block)
                snapshot.flush()
                os.fsync(snapshot.fileno())
                closed_stat = os.fstat(source_handle.fileno())
        except OSError as exc:
            raise RenderCheckpointError(
                f"checkpoint could not be copied to a private verification snapshot: {source}"
            ) from exc
        if (
            opened_stat.st_dev != closed_stat.st_dev
            or opened_stat.st_ino != closed_stat.st_ino
            or opened_stat.st_size != closed_stat.st_size
            or opened_stat.st_mtime_ns != closed_stat.st_mtime_ns
            or opened_stat.st_ctime_ns != closed_stat.st_ctime_ns
            or copied_size != opened_stat.st_size
        ):
            raise RenderCheckpointError("checkpoint changed while its checksum was being computed")
        actual_digest = digest.hexdigest()
        if sidecar is not None:
            if int(sidecar.get("checkpoint_size_bytes", -1)) != copied_size:
                raise RenderCheckpointError(
                    "checkpoint size or SHA does not match its sidecar"
                )
            if actual_digest != str(sidecar["checkpoint_sha256"]).lower():
                raise RenderCheckpointError(
                    "checkpoint content SHA does not match its sidecar"
                )
        if expected_digest is not None and actual_digest != expected_digest:
            raise RenderCheckpointError(
                "checkpoint content SHA does not match the expected value: "
                f"expected={expected_digest} actual={actual_digest}"
            )
        try:
            state = torch.load(
                snapshot_path,
                map_location=map_location,
                weights_only=False,
                mmap=mmap,
            )
        except Exception as exc:  # noqa: BLE001 - checkpoint.
            raise RenderCheckpointError("checkpoint could not be deserialized") from exc
    finally:
        snapshot_path.unlink(missing_ok=True)
    if not isinstance(state, Mapping):
        raise RenderCheckpointError("checkpoint root object must be a mapping")
    if sidecar is not None:
        training_state = state.get("training_state")
        expected_sidecar = {
            "global_step": (
                training_state.get("global_step")
                if isinstance(training_state, Mapping)
                else None
            ),
            "config_hash": state.get("config_hash"),
            "revision_manifest_hash": state.get("revision_manifest_hash"),
            "integrity_hash": state.get("integrity_hash"),
        }
        mismatches = {
            name: {
                "sidecar": sidecar.get(name),
                "checkpoint": expected,
            }
            for name, expected in expected_sidecar.items()
            if sidecar.get(name) != expected
        }
        if mismatches:
            raise RenderCheckpointError(
                f"checkpoint sidecar metadata does not match the payload: {mismatches}"
            )
    return state


def _verify_sidecar(path: Path) -> None:

    _load_checkpoint_payload(path, strict_sidecar=True, map_location="cpu")


def read_render_checkpoint_state(
    path: str | Path,
    *,
    strict_sidecar: bool = True,
    map_location: str | torch.device = "cpu",
    expected_checkpoint_sha256: str | None = None,
    expected_component: str | None = None,
    mmap: bool = False,
) -> Mapping[str, Any]:

    state = _load_checkpoint_payload(
        Path(path),
        strict_sidecar=strict_sidecar,
        map_location=map_location,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        mmap=mmap,
    )
    _validate_loaded_state(
        state,
        expected_component=expected_component,
        expected_upstream_revisions=None,
        expected_config=None,
        expected_config_sections=None,
        expected_config_replacements=None,
        expected_config_normalizer=None,
        required_upstream_sha_keys=None,
    )
    return state


def _validate_loaded_state(
    state: Mapping[str, Any],
    *,
    expected_component: str | None,
    expected_upstream_revisions: Mapping[str, Any] | None,
    expected_config: Mapping[str, Any] | None,
    expected_config_sections: Sequence[str] | None,
    expected_config_replacements: Sequence[str] | None,
    expected_config_normalizer: (
        Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
    ),
    required_upstream_sha_keys: Sequence[str] | None,
) -> None:
    _validate_checkpoint_structure(state)
    if state.get("format_version") != FORMAT_VERSION:
        raise RenderCheckpointError(
            f"checkpoint format_version={state.get('format_version')!r},"
            f"requires {FORMAT_VERSION}"
        )
    if expected_component is not None:
        if not isinstance(expected_component, str) or not expected_component:
            raise ValueError("expected_component must be a non-empty string")
        if state.get("component") != expected_component:
            raise RenderCheckpointError(
                "checkpoint component does not match:"
                f"expected={expected_component!r} actual={state.get('component')!r}"
            )
    revisions = validate_upstream_revisions(
        state.get("upstream_revisions", {}),
        required_sha_keys=state.get("required_upstream_sha_keys", ()),
    )
    if _sha256_json(revisions) != state.get("revision_manifest_hash"):
        raise RenderCheckpointError("upstream revision manifest has been tampered with")
    if _sha256_json(_integrity_payload(state)) != state.get("integrity_hash"):
        raise RenderCheckpointError("checkpoint metadata integrity hash does not match")
    if config_hash(dict(state.get("config", {}))) != state.get("config_hash"):
        raise RenderCheckpointError("checkpoint config has been tampered with")
    stored_config = state.get("config")
    if not isinstance(stored_config, Mapping):
        raise RenderCheckpointError("checkpoint config must be a mapping")
    comparable_stored = stored_config
    comparable_expected = expected_config
    if expected_config_normalizer is not None:
        comparable_stored = expected_config_normalizer(stored_config)
        if expected_config is not None:
            comparable_expected = expected_config_normalizer(expected_config)
    replacements = tuple(expected_config_replacements or ())
    if replacements:
        if comparable_expected is None:
            raise ValueError("expected_config_replacements also requires expected_config")
        if not all(isinstance(value, str) and value for value in replacements) or len(
            set(replacements)
        ) != len(replacements):
            raise TypeError("expected_config_replacements must be a sequence of unique non-empty field paths")
        comparable_expected = deepcopy(comparable_expected)
        for field_path in replacements:
            parts = field_path.split(".")
            expected_cursor: Any = comparable_expected
            stored_cursor: Any = comparable_stored
            for part in parts[:-1]:
                if (
                    not isinstance(expected_cursor, MutableMapping)
                    or not isinstance(stored_cursor, Mapping)
                    or part not in expected_cursor
                    or part not in stored_cursor
                ):
                    raise RenderCheckpointError(
                        f"Allowed config replacement path does not exist: {field_path}"
                    )
                expected_cursor = expected_cursor[part]
                stored_cursor = stored_cursor[part]
            leaf = parts[-1]
            if (
                not isinstance(expected_cursor, MutableMapping)
                or not isinstance(stored_cursor, Mapping)
                or leaf not in expected_cursor
                or leaf not in stored_cursor
            ):
                raise RenderCheckpointError(
                    f"Allowed config replacement path does not exist: {field_path}"
                )
            expected_cursor[leaf] = deepcopy(stored_cursor[leaf])
    if (
        comparable_expected is not None
        and expected_config_sections is None
        and config_hash(dict(comparable_expected))
        != config_hash(dict(comparable_stored))
    ):
        raise RenderCheckpointError("resume config hash does not match")
    if expected_config_sections is not None:
        if comparable_expected is None:
            raise ValueError("expected_config_sections also requires expected_config")
        mismatches = {
            section: {
                "expected": comparable_expected.get(section),
                "actual": comparable_stored.get(section),
            }
            for section in expected_config_sections
            if comparable_expected.get(section) != comparable_stored.get(section)
        }
        if mismatches:
            raise RenderCheckpointError(
                f"checkpoint config section does not match:  {mismatches}"
            )
    if expected_upstream_revisions is not None:
        expected = validate_upstream_revisions(
            expected_upstream_revisions,
            required_sha_keys=required_upstream_sha_keys or (),
        )
        if revisions != expected:
            raise RenderCheckpointError(
                "upstream revision or SHA does not match; refusing to resume"
            )
    if required_upstream_sha_keys is not None:
        stored_required = sorted(state.get("required_upstream_sha_keys", ()))
        if sorted(set(required_upstream_sha_keys)) != stored_required:
            raise RenderCheckpointError("required upstream SHA field set does not match")


def _require_same_names(
    target: Mapping[str, Any],
    stored: Mapping[str, Any],
    *,
    kind: str,
) -> None:
    if set(target) != set(stored):
        raise RenderCheckpointError(
            f"{kind} names do not match: current={sorted(target)}, stored={sorted(stored)}"
        )


def _preflight_model_state(
    name: str,
    model: nn.Module,
    stored_state: Any,
) -> None:
    if not isinstance(stored_state, Mapping):
        raise RenderCheckpointError(f"model {name} state must be a mapping")
    current = unwrap_model(model).state_dict()
    if set(current) != set(stored_state):
        missing = sorted(set(current) - set(stored_state))
        unexpected = sorted(set(stored_state) - set(current))
        raise RenderCheckpointError(
            f"model {name} state keys do not match:"
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    mismatched = []
    for key, current_value in current.items():
        stored_value = stored_state[key]
        if isinstance(current_value, torch.Tensor):
            if not isinstance(stored_value, torch.Tensor):
                mismatched.append(
                    f"{key}: expected tensor, actual={type(stored_value).__name__}"
                )
            elif current_value.shape != stored_value.shape:
                mismatched.append(
                    f"{key}: expected shape={tuple(current_value.shape)}, "
                    f"actual={tuple(stored_value.shape)}"
                )
    if mismatched:
        raise RenderCheckpointError(
            f"model {name} state shapes or types do not match: {mismatched[:8]}"
        )


def _preflight_optimizer_state(
    name: str,
    optimizer: torch.optim.Optimizer,
    stored_state: Any,
) -> None:
    if not isinstance(stored_state, Mapping):
        raise RenderCheckpointError(f"optimizer {name} state must be a mapping")
    stored_groups = stored_state.get("param_groups")
    stored_values = stored_state.get("state")
    if not isinstance(stored_groups, list) or not isinstance(stored_values, Mapping):
        raise RenderCheckpointError(f"optimizer {name} state is missing param_groups or state")
    current_groups = optimizer.param_groups
    if len(stored_groups) != len(current_groups):
        raise RenderCheckpointError(f"optimizer {name} parameter-group count does not match")
    parameter_by_stored_id: dict[Any, torch.Tensor] = {}
    group_by_stored_id: dict[Any, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for index, (stored_group, current_group) in enumerate(
        zip(stored_groups, current_groups, strict=True)
    ):
        if not isinstance(stored_group, Mapping):
            raise RenderCheckpointError(
                f"optimizer {name} param_groups[{index}] must be a mapping"
            )
        stored_parameters = stored_group.get("params")
        current_parameters = current_group.get("params")
        if (
            not isinstance(stored_parameters, Sequence)
            or isinstance(stored_parameters, (str, bytes))
            or not isinstance(current_parameters, Sequence)
            or len(stored_parameters) != len(current_parameters)
        ):
            raise RenderCheckpointError(
                f"optimizer {name} param_groups[{index}] parameter count does not match"
            )
        for stored_id, parameter in zip(
            stored_parameters, current_parameters, strict=True
        ):
            if stored_id in parameter_by_stored_id:
                raise RenderCheckpointError(
                    f"optimizer {name} checkpoint contains duplicate parameter identifiers"
                )
            if not isinstance(parameter, torch.Tensor):
                raise RenderCheckpointError(
                    f"optimizer {name} current parameter object is not a tensor"
                )
            parameter_by_stored_id[stored_id] = parameter
            group_by_stored_id[stored_id] = (stored_group, current_group)
    unknown_state = set(stored_values) - set(parameter_by_stored_id)
    if unknown_state:
        raise RenderCheckpointError(
            f"optimizer {name} state refers to unknown parameters: {list(unknown_state)[:8]}"
        )
    mismatched = []
    for parameter_id, parameter_state in stored_values.items():
        if not isinstance(parameter_state, Mapping):
            raise RenderCheckpointError(f"optimizer {name} parameter state must be a mapping")
        parameter = parameter_by_stored_id[parameter_id]
        stored_group, current_group = group_by_stored_id[parameter_id]
        for field, value in parameter_state.items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape != parameter.shape
            ):


                expected_matrix_shape = (
                    tuple(parameter.shape)
                    if parameter.ndim <= 2
                    else (parameter.shape[0], parameter.numel() // parameter.shape[0])
                )
                is_ear_v2_flattened_muon_momentum = (
                    optimizer.__class__.__name__ == "Muon"
                    and stored_group.get("implementation") == "ear_v2_source_v1"
                    and current_group.get("implementation")
                    == "ear_v2_source_v1"
                    and bool(stored_group.get("orthogonalize"))
                    and bool(current_group.get("orthogonalize"))
                    and field == "momentum_buffer"
                    and parameter.ndim > 2
                    and tuple(value.shape) == expected_matrix_shape
                )
                if is_ear_v2_flattened_muon_momentum:
                    continue
                mismatched.append(
                    f"{parameter_id}.{field}: expected shape={tuple(parameter.shape)}, "
                    f"actual={tuple(value.shape)}"
                )
    if mismatched:
        raise RenderCheckpointError(
            f"optimizer {name} tensor state shape does not match: {mismatched[:8]}"
        )


def _preflight_state_dict(
    name: str,
    current: Any,
    stored: Any,
) -> None:

    if isinstance(current, Mapping):
        if not isinstance(stored, Mapping) or set(current) != set(stored):
            raise RenderCheckpointError(f"{name} state fields do not match")
        for key, value in current.items():
            _preflight_state_dict(f"{name}.{key}", value, stored[key])
        return
    if isinstance(current, (list, tuple)):
        if not isinstance(stored, type(current)) or len(current) != len(stored):
            raise RenderCheckpointError(f"{name}  state sequence structure does not match")
        for index, (left, right) in enumerate(zip(current, stored, strict=True)):
            _preflight_state_dict(f"{name}[{index}]", left, right)
        return
    if isinstance(current, torch.Tensor):
        if not isinstance(stored, torch.Tensor) or current.shape != stored.shape:
            raise RenderCheckpointError(f"{name}  tensor shape or type does not match")


def _validate_rng_state_payload(state: Any) -> None:
    if not isinstance(state, Mapping):
        raise RenderCheckpointError("checkpoint rng_states must be a mapping")
    expected = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != expected:
        raise RenderCheckpointError("checkpoint rng_states fields are incomplete")
    try:
        if state["python"] is not None:
            random.Random().setstate(state["python"])
        if np is not None and state["numpy"] is not None:
            probe = np.random.RandomState()
            probe.set_state(state["numpy"])
        if state["torch_cpu"] is not None:
            torch.Generator(device="cpu").set_state(state["torch_cpu"].cpu())
        cuda_states = state["torch_cuda"]
        if cuda_states is not None:
            if not isinstance(cuda_states, Sequence) or isinstance(
                cuda_states, (str, bytes)
            ):
                raise TypeError("CUDA RNG state must be a sequence")
            if (
                torch.cuda.is_available()
                and len(cuda_states) != torch.cuda.device_count()
            ):
                raise ValueError("CUDA RNG device count does not match the current process")
            for value in cuda_states:
                if not isinstance(value, torch.Tensor):
                    raise TypeError("CUDA RNG state must be a tensor sequence")
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RenderCheckpointError("checkpoint RNG state is incompatible with") from exc


def _resume_state_from_payload(state: Mapping[str, Any]) -> RenderResumeState:
    training = _require_mapping_field(state, "training_state")
    sampler_state = training.get("sampler_state")
    distributed_state = training.get("distributed_state")
    if sampler_state is not None and not isinstance(sampler_state, Mapping):
        raise RenderCheckpointError(
            "checkpoint training_state.sampler_state must be a mapping or None"
        )
    if distributed_state is not None and not isinstance(distributed_state, Mapping):
        raise RenderCheckpointError(
            "checkpoint training_state.distributed_state must be a mapping or None"
        )
    return RenderResumeState(
        config_hash=str(state["config_hash"]),
        config=dict(_require_mapping_field(state, "config")),
        global_step=int(training["global_step"]),
        epoch=int(training.get("epoch", 0)),
        batches_consumed=int(training.get("batches_consumed", 0)),
        training_audio_seconds=float(training.get("training_audio_seconds", 0.0)),
        sampler_state=(
            dict(sampler_state) if isinstance(sampler_state, Mapping) else None
        ),
        distributed_state=(
            dict(distributed_state) if isinstance(distributed_state, Mapping) else None
        ),
        upstream_revisions=dict(_require_mapping_field(state, "upstream_revisions")),
        required_upstream_sha_keys=tuple(state["required_upstream_sha_keys"]),
        extra_state=dict(_require_mapping_field(state, "extra_state")),
    )


def load_render_checkpoint(
    path: str | Path,
    *,
    models: Mapping[str, nn.Module] | nn.Module | None = None,
    optimizers: Mapping[str, torch.optim.Optimizer]
    | torch.optim.Optimizer
    | None = None,
    schedulers: Mapping[str, Any] | Any | None = None,
    ema: Mapping[str, Any] | Any | None = None,
    scaler: Any | None = None,
    sampler: Any | None = None,
    expected_config: Mapping[str, Any] | None = None,
    expected_config_sections: Sequence[str] | None = None,
    expected_config_replacements: Sequence[str] | None = None,
    expected_config_normalizer: (
        Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
    ) = None,
    expected_checkpoint_sha256: str | None = None,
    expected_component: str | None = None,
    expected_upstream_revisions: Mapping[str, Any] | None = None,
    required_upstream_sha_keys: Sequence[str] | None = None,
    resume: bool = True,
    restore_rng: bool = True,
    strict_sidecar: bool = True,
    map_location: str | torch.device = "cpu",
    pre_apply_validator: Callable[[RenderResumeState], None] | None = None,
) -> RenderResumeState:

    if pre_apply_validator is not None and not callable(pre_apply_validator):
        raise TypeError("pre_apply_validator must be callable or None")
    if expected_config_normalizer is not None and not callable(
        expected_config_normalizer
    ):
        raise TypeError("expected_config_normalizer must be callable or None")

    source = Path(path)
    state = _load_checkpoint_payload(
        source,
        strict_sidecar=strict_sidecar,
        map_location=map_location,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    _validate_loaded_state(
        state,
        expected_component=expected_component,
        expected_upstream_revisions=expected_upstream_revisions,
        expected_config=expected_config,
        expected_config_sections=expected_config_sections,
        expected_config_replacements=expected_config_replacements,
        expected_config_normalizer=expected_config_normalizer,
        required_upstream_sha_keys=required_upstream_sha_keys,
    )
    named_models = _named_objects(models, default_name="model")
    if named_models:
        _require_same_names(named_models, state["models"], kind="model")
        for name, model in named_models.items():
            _preflight_model_state(name, model, state["models"][name])
    if resume:
        named_optimizers = _named_objects(optimizers, default_name="optimizer")
        named_schedulers = _named_objects(schedulers, default_name="scheduler")
        named_ema = _named_objects(ema, default_name="ema")
        for kind, current, stored in (
            ("optimizer", named_optimizers, state["optimizers"]),
            ("scheduler", named_schedulers, state["schedulers"]),
            ("EMA", named_ema, state["ema"]),
        ):
            _require_same_names(current, stored, kind=kind)
        stored_scaler = state.get("scaler")
        if (scaler is None) != (stored_scaler is None):
            raise RenderCheckpointError("GradScaler state does not match")
        for name, optimizer in named_optimizers.items():
            _preflight_optimizer_state(name, optimizer, state["optimizers"][name])
        for kind, current, stored in (
            ("scheduler", named_schedulers, state["schedulers"]),
            ("EMA", named_ema, state["ema"]),
        ):
            for name, value in current.items():
                _preflight_state_dict(
                    f"{kind} {name}",
                    value.state_dict(),
                    stored[name],
                )
        if scaler is not None:
            _preflight_state_dict(
                "GradScaler",
                scaler.state_dict(),
                stored_scaler,
            )
        sampler_state = state["training_state"].get("sampler_state")
        if sampler is not None and sampler_state is None:
            raise RenderCheckpointError("A sampler is required, but the checkpoint has no sampler state")
        if restore_rng:
            _validate_rng_state_payload(state["rng_states"])
    resume_state = _resume_state_from_payload(state)
    if pre_apply_validator is not None:
        try:
            pre_apply_validator(resume_state)
        except RenderCheckpointError:
            raise
        except Exception as exc:
            raise RenderCheckpointError(
                f"checkpoint component pre-apply validation failed: {exc}"
            ) from exc

    if named_models:
        for name, model in named_models.items():
            try:
                unwrap_model(model).load_state_dict(state["models"][name], strict=True)
            except Exception as exc:
                raise RenderCheckpointApplyError(
                    "checkpoint apply failed after mutation began; discard the target "
                    f"objects and exit: model {name} state is incompatible"
                ) from exc
    if resume:
        for kind, current, stored in (
            ("optimizer", named_optimizers, state["optimizers"]),
            ("scheduler", named_schedulers, state["schedulers"]),
            ("EMA", named_ema, state["ema"]),
        ):
            for name, value in current.items():
                try:
                    value.load_state_dict(stored[name])
                except Exception as exc:
                    raise RenderCheckpointApplyError(
                        "checkpoint apply failed after mutation began; discard the target "
                        f"objects and exit: {kind} {name} state is incompatible"
                    ) from exc
        if scaler is not None:
            try:
                scaler.load_state_dict(stored_scaler)
            except Exception as exc:
                raise RenderCheckpointApplyError(
                    "checkpoint apply failed after mutation began; discard the target "
                    "objects and exit: GradScaler state is incompatible"
                ) from exc
        try:
            if sampler is not None:
                if isinstance(sampler, MutableMapping):
                    sampler.clear()
                    sampler.update(sampler_state)
                elif hasattr(sampler, "load_state_dict"):
                    sampler.load_state_dict(sampler_state)
                else:
                    raise TypeError("sampler must implement load_state_dict()")
            elif sampler_state is not None:

                pass
        except Exception as exc:
            raise RenderCheckpointApplyError(
                "checkpoint apply failed after mutation began; discard the target "
                "objects and exit: sampler state is incompatible"
            ) from exc
        if restore_rng:
            try:
                restore_rng_state(state["rng_states"])
            except Exception as exc:
                raise RenderCheckpointApplyError(
                    "checkpoint applyfailed,processRNGmay have been partially modified,"
                    "must exit because the RNG state is incompatible with"
                ) from exc
    return resume_state


def inspect_render_checkpoint(
    path: str | Path,
    *,
    verify_sidecar: bool = True,
) -> dict[str, Any]:
    source = Path(path)
    state = _load_checkpoint_payload(
        source,
        strict_sidecar=verify_sidecar,
        map_location="cpu",
    )
    _validate_loaded_state(
        state,
        expected_component=None,
        expected_upstream_revisions=None,
        expected_config=None,
        expected_config_sections=None,
        expected_config_replacements=None,
        expected_config_normalizer=None,
        required_upstream_sha_keys=None,
    )
    return {
        "format_version": state["format_version"],
        "component": state["component"],
        "config_hash": state["config_hash"],
        "upstream_revisions": state["upstream_revisions"],
        "training_state": state["training_state"],
        "model_names": sorted(state["models"]),
        "optimizer_names": sorted(state["optimizers"]),
        "scheduler_names": sorted(state["schedulers"]),
    }


save_checkpoint = save_render_checkpoint
load_checkpoint = load_render_checkpoint
ResumeState = RenderResumeState
