
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.distributed as dist


class ResumeState(NamedTuple):

    global_step: int = 0
    epoch: int = 0
    batches_consumed: int = 0
    distributed_state: dict[str, Any] | None = None


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _pinned_code_revision_identity() -> dict[str, Any] | None:

    raw_path = os.environ.get("OQM_CODE_REVISION_PATH", "")
    expected_sha256 = os.environ.get("OQM_CODE_REVISION_SHA256", "")
    if not raw_path and not expected_sha256:
        return None
    if not raw_path or not expected_sha256:
        raise RuntimeError(
            "OQM_CODE_REVISION_PATH and OQM_CODE_REVISION_SHA256 must be provided together"
        )
    if os.environ.get("OQM_CODE_REVISION_B64"):
        raise RuntimeError(
            "OQM_CODE_REVISION_B64 cannot be combined with path-based code revision binding"
        )
    if (
        len(expected_sha256) != 64
        or expected_sha256 != expected_sha256.lower()
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise RuntimeError(
            "OQM_CODE_REVISION_SHA256 must be a 64-character lowercase SHA-256 digest"
        )

    requested_path = Path(raw_path)
    try:
        identity_path = requested_path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Pinned code identity file cannot be read: {raw_path}: {exc}") from exc
    if not requested_path.is_absolute() or raw_path != str(identity_path):
        raise RuntimeError("OQM_CODE_REVISION_PATH must be a canonical absolute path")
    if not identity_path.is_file():
        raise RuntimeError(f"Pinned code identity is not a regular file: {identity_path}")
    if identity_path.stat().st_mode & 0o222:
        raise RuntimeError(f"Pinned code identity file must be read-only: {identity_path}")

    try:
        raw_identity = identity_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Pinned code identity file could not be read: {identity_path}: {exc}") from exc
    actual_sha256 = hashlib.sha256(raw_identity).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Pinned code identity file SHA-256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    try:
        identity = json.loads(raw_identity.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Pinned code identity file is not valid JSON: {identity_path}") from exc
    if not isinstance(identity, dict):
        raise TypeError("Pinned code identity JSON must be an object")

    expected_workspace_sha256 = os.environ.get("OQM_CODE_REVISION", "")
    if expected_workspace_sha256 and (
        identity.get("workspace_sha256") != expected_workspace_sha256
    ):
        raise RuntimeError(
            "Pinned code identity workspace_sha256 does not match the launcher binding"
        )
    return identity


@lru_cache(maxsize=4)
def code_revision_identity(repo_root: str | Path | None = None) -> dict[str, Any]:

    pinned_identity = _pinned_code_revision_identity()
    if pinned_identity is not None:
        return pinned_identity

    encoded = os.environ.get("OQM_CODE_REVISION_B64")
    if encoded:
        try:
            return json.loads(base64.b64decode(encoded).decode("utf-8"))
        except (ValueError, json.JSONDecodeError) as exc:
            return {
                "available": False,
                "error": f"OQM_CODE_REVISION_B64 could not be decoded: {exc}",
            }
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )

    def git(*arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "-C",
                str(root),
                *arguments,
            ],
            capture_output=True,
            check=False,
        )

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    diff = git("diff", "--binary", "HEAD", "--")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z")
    failed = [
        result
        for result in (head, status, diff, untracked)
        if result.returncode != 0
    ]
    if failed:
        message = failed[0].stderr.decode("utf-8", errors="replace").strip()
        return {
            "available": False,
            "repo_root": str(root),
            "error": message or f"git rc={failed[0].returncode}",
        }
    untracked_paths = [
        Path(value.decode("utf-8", errors="surrogateescape"))
        for value in untracked.stdout.split(b"\0")
        if value
    ]
    digest = hashlib.sha256()
    commit = head.stdout.decode().strip()
    digest.update(f"commit\0{commit}\0".encode())
    digest.update(b"tracked-diff\0")
    digest.update(diff.stdout)
    for relative in sorted(untracked_paths, key=lambda item: str(item)):
        digest.update(b"untracked\0")
        digest.update(os.fsencode(relative))
        digest.update(b"\0")
        path = root / relative
        if path.is_file():
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
        elif path.is_symlink():
            digest.update(os.fsencode(os.readlink(path)))
        digest.update(b"\0")
    return {
        "available": True,
        "repo_root": str(root),
        "commit": commit,
        "dirty": bool(status.stdout),
        "workspace_sha256": digest.hexdigest(),
        "tracked_diff_bytes": len(diff.stdout),
        "untracked_files": len(untracked_paths),
    }


def resume_config_hash(config: dict[str, Any]) -> str:

    payload = {key: value for key, value in config.items() if key != "lineage"}
    return config_hash(payload)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _sampler_kind(config: dict[str, Any]) -> str:
    data = config.get("data", {})
    if data.get("balanced_sampler"):
        return (
            "balanced_duration_bucket"
            if data.get("duration_buckets_sec")
            else "balanced_global"
        )
    if data.get("duration_buckets_sec"):
        return "duration_bucket"
    return "distributed"


def distributed_training_state(config: dict[str, Any]) -> dict[str, Any]:

    manifest = Path(str(config.get("data", {}).get("manifest", "")))
    manifest_stat: dict[str, Any] | None = None
    if manifest.is_file():
        stat = manifest.stat()
        manifest_stat = {
            "path": str(manifest.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return {
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "batch_size_per_rank": int(
            config.get("train", {}).get("batch_size_per_rank", 1)
        ),
        "gradient_accumulation_steps": int(
            config.get("train", {}).get("gradient_accumulation_steps", 1)
        ),
        "sampler_kind": _sampler_kind(config),
        "manifest": manifest_stat,
    }


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(32 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".cache.json")


def _seal_checkpoint_cache_timestamp(path: Path) -> os.stat_result:

    stat = path.stat()
    sealed_mtime_ns = max(0, stat.st_mtime_ns - 1_000_000_000)
    os.utime(path, ns=(stat.st_atime_ns, sealed_mtime_ns))
    return path.stat()


def checkpoint_content_fingerprint(path: str | Path) -> str:

    checkpoint = Path(path)
    metadata_path = _cache_metadata_path(checkpoint)
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            stat = checkpoint.stat()
            if (
                int(metadata["size"]) == stat.st_size
                and int(metadata["cached_device"]) == stat.st_dev
                and int(metadata["cached_mtime_ns"]) == stat.st_mtime_ns
                and int(metadata["cached_ctime_ns"]) == stat.st_ctime_ns
                and int(metadata["cached_inode"]) == stat.st_ino
            ):
                return str(metadata["sha256"])
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            pass
    return file_sha256(checkpoint)


def checkpoint_runtime_identity(path: str | Path) -> dict[str, Any]:

    checkpoint = Path(path)
    state = torch.load(
        checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if state.get("format_version") == "oqm.render.ckpt.v1":
        config = state.get("config")
        training_state = state.get("training_state")
        if not isinstance(config, dict) or not isinstance(training_state, dict):
            raise RuntimeError("Render checkpoint is missing config or training_state")
        return {
            "sha256": checkpoint_content_fingerprint(checkpoint),
            "size": checkpoint.stat().st_size,
            "format_version": state.get("format_version"),
            "stage": int(config.get("stage", 0)),
            "global_step": int(training_state.get("global_step", -1)),

            "config_hash": state.get("config_hash"),
            "resume_config_hash": state.get("config_hash"),
            "distributed_state": training_state.get("distributed_state"),
            "provenance": state.get("upstream_revisions"),
        }
    return {
        "sha256": checkpoint_content_fingerprint(checkpoint),
        "size": checkpoint.stat().st_size,
        "format_version": state.get("format_version"),
        "stage": int(state.get("stage", 0)),
        "global_step": int(state.get("global_step", 0)),
        "config_hash": state.get("config_hash"),
        "resume_config_hash": state.get("resume_config_hash")
        or resume_config_hash(state.get("config", {})),
        "distributed_state": state.get("distributed_state"),
        "provenance": state.get("provenance"),
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: dict[str, Any],
    global_step: int,
    epoch: int = 0,
    batches_consumed: int = 0,
    provenance: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = unwrap_model(model).state_dict()


    nonfinite = [
        key
        for key, value in model_state.items()
        if value.is_floating_point() and not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise RuntimeError(
            f"Refusing to save checkpoint with NaN or Inf values at step {global_step}; "
            f"{len(nonfinite)} tensors are affected, including {nonfinite[:5]}. "
            f"The previous checkpoint remains at {path}."
        )
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    distributed_state = distributed_training_state(config)
    state = {
        "format_version": "oqm.tokenizer.ckpt.v1",
        "stage": int(config["stage"]),
        "global_step": global_step,
        "epoch": epoch,
        "batches_consumed": batches_consumed,
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "config_hash": config_hash(config),
        "resume_config_hash": resume_config_hash(config),
        "distributed_state": distributed_state,
        "provenance": provenance,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = {
        "format_version": state["format_version"],
        "stage": state["stage"],
        "global_step": global_step,
        "epoch": epoch,
        "batches_consumed": batches_consumed,
        "semantic_contract": config["semantic_contract"],
        "feature_config": config["features"],
        "model_config": config["model"],
        "quantizer_config": config["quantizer"],
        "config_hash": state["config_hash"],
        "resume_config_hash": state["resume_config_hash"],
        "distributed_state": distributed_state,
        "provenance": provenance,
        "checkpoint_size_bytes": path.stat().st_size,
    }
    sidecar_path = path.with_suffix(path.suffix + ".json")
    sidecar_temporary = sidecar_path.with_suffix(
        sidecar_path.suffix + f".tmp.{os.getpid()}"
    )
    sidecar_temporary.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(sidecar_temporary, sidecar_path)


def retain_checkpoint(source: str | Path, destination: str | Path) -> None:

    source = Path(source)
    destination = Path(destination)
    source_sidecar = source.with_suffix(source.suffix + ".json")
    destination_sidecar = destination.with_suffix(destination.suffix + ".json")
    if destination.exists() or destination_sidecar.exists():
        raise FileExistsError(
            f"Milestone checkpoint already exists and will not be overwritten: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    sidecar_temporary = destination_sidecar.with_suffix(
        destination_sidecar.suffix + f".tmp.{os.getpid()}"
    )
    try:
        os.link(source, temporary)
        os.link(source_sidecar, sidecar_temporary)
        os.replace(temporary, destination)
        os.replace(sidecar_temporary, destination_sidecar)
    finally:
        temporary.unlink(missing_ok=True)
        sidecar_temporary.unlink(missing_ok=True)


def stage_checkpoint_locally(
    path: str | Path,
    *,
    cache_dir: str | Path = "/tmp/open_qwen_music/checkpoints",
    retries: int = 5,
    expected_sha256: str | None = None,
) -> Path:

    if expected_sha256 is not None:
        if (
            len(expected_sha256) != 64
            or expected_sha256 != expected_sha256.lower()
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError(
                "expected_sha256 must be a 64-character lowercase hexadecimal digest"
            )
    source = Path(path).resolve()
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    source_key = (
        expected_sha256[:20]
        if expected_sha256 is not None
        else hashlib.sha256(str(source).encode()).hexdigest()[:20]
    )
    lock_path = directory / f"{source.stem}.{source_key}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for attempt in range(1, retries + 1):
            before = source.stat()
            source_identity = {
                "path": str(source),
                "device": before.st_dev,
                "size": before.st_size,
                "mtime_ns": before.st_mtime_ns,
                "ctime_ns": before.st_ctime_ns,
                "inode": before.st_ino,
            }
            fingerprint = (
                expected_sha256[:20]
                if expected_sha256 is not None
                else hashlib.sha256(
                    json.dumps(source_identity, sort_keys=True).encode()
                ).hexdigest()[:20]
            )
            local = directory / f"{source.stem}.{fingerprint}{source.suffix}"
            metadata_path = _cache_metadata_path(local)
            if local.exists() and local.stat().st_size == before.st_size:
                try:
                    state = torch.load(
                        local, map_location="cpu", weights_only=False, mmap=True
                    )
                    valid_tokenizer = (
                        "model" in state and "format_version" in state
                    )
                    valid_render = (
                        "models" in state
                        and isinstance(state.get("config"), dict)
                        and isinstance(state.get("training_state"), dict)
                        and state.get("format_version") == "oqm.render.ckpt.v1"
                    )
                    if valid_tokenizer or valid_render:
                        if (
                            expected_sha256 is not None
                            and checkpoint_content_fingerprint(local)
                            != expected_sha256
                        ):
                            raise RuntimeError("Local checkpoint cache SHA does not match")
                        if not metadata_path.is_file():
                            local_stat = _seal_checkpoint_cache_timestamp(local)
                            metadata = {
                                "source": source_identity,
                                "size": local_stat.st_size,
                                "cached_device": local_stat.st_dev,
                                "cached_mtime_ns": local_stat.st_mtime_ns,
                                "cached_ctime_ns": local_stat.st_ctime_ns,
                                "cached_inode": local_stat.st_ino,
                                "sha256": file_sha256(local),
                            }
                            metadata_path.write_text(
                                json.dumps(metadata, ensure_ascii=False, indent=2)
                                + "\n",
                                encoding="utf-8",
                            )
                        return local
                except Exception:
                    local.unlink(missing_ok=True)
                    metadata_path.unlink(missing_ok=True)
            temporary = directory / (
                f".{local.name}.{os.getpid()}.{attempt}.tmp"
            )
            try:
                digest = hashlib.sha256()
                with source.open("rb") as src, temporary.open("wb") as dst:
                    while chunk := src.read(32 * 1024 * 1024):
                        digest.update(chunk)
                        dst.write(chunk)
                    dst.flush()
                    os.fsync(dst.fileno())
                after = source.stat()
                after_identity = {
                    "path": str(source),
                    "device": after.st_dev,
                    "size": after.st_size,
                    "mtime_ns": after.st_mtime_ns,
                    "ctime_ns": after.st_ctime_ns,
                    "inode": after.st_ino,
                }
                if source_identity != after_identity:
                    raise RuntimeError(
                        "Checkpoint changed while being staged; retrying"
                    )
                if temporary.stat().st_size != before.st_size:
                    raise IOError(
                        f"checkpoint size mismatch "
                        f"{temporary.stat().st_size}!={before.st_size}"
                    )
                state = torch.load(
                    temporary, map_location="cpu", weights_only=False, mmap=True
                )
                valid_tokenizer = "model" in state and "format_version" in state
                valid_render = (
                    "models" in state
                    and isinstance(state.get("config"), dict)
                    and isinstance(state.get("training_state"), dict)
                    and state.get("format_version") == "oqm.render.ckpt.v1"
                )
                if not (valid_tokenizer or valid_render):
                    raise RuntimeError("Checkpoint is missing required fields")
                if (
                    expected_sha256 is not None
                    and digest.hexdigest() != expected_sha256
                ):
                    raise RuntimeError("Checkpoint content SHA does not match")
                os.replace(temporary, local)
                local_stat = _seal_checkpoint_cache_timestamp(local)
                metadata = {
                    "source": source_identity,
                    "size": local_stat.st_size,
                    "cached_device": local_stat.st_dev,
                    "cached_mtime_ns": local_stat.st_mtime_ns,
                    "cached_ctime_ns": local_stat.st_ctime_ns,
                    "cached_inode": local_stat.st_ino,
                    "sha256": digest.hexdigest(),
                }
                metadata_temporary = metadata_path.with_suffix(
                    metadata_path.suffix + f".tmp.{os.getpid()}"
                )
                metadata_temporary.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(metadata_temporary, metadata_path)
                return local
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt >= retries:
                    raise
                time.sleep(float(attempt) * 2.0)
        raise RuntimeError(f"cannot stage checkpoint: {source}")


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    resume: bool = False,
    load_model: bool = True,
    require_stage: int | None = None,
    restore_rng: bool = True,
    allow_cross_stage: bool = False,
) -> ResumeState:
    if allow_cross_stage and require_stage is None:
        raise ValueError("allow_cross_stage must also declare require_stage")
    started = time.monotonic()
    state = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    print(
        f"checkpoint_load rank={os.environ.get('RANK', '0')} "
        f"path={path} seconds={time.monotonic() - started:.3f}",
        flush=True,
    )
    if load_model:
        model_started = time.monotonic()
        source_stage = int(state.get("stage", 0))
        target_stage = int(unwrap_model(model).stage)


        if require_stage is not None and source_stage != require_stage:
            raise RuntimeError(
                f"Checkpoint stage is {source_stage}, but this entry point requires stage "
                f"{require_stage}. Cross-stage loading would leave head or quantizer weights "
                "uninitialized."
            )
        if allow_cross_stage and source_stage >= target_stage:
            raise RuntimeError(
                "allow_cross_stage only supports initialization from an earlier stage: "
                f"source_stage={source_stage}, target_stage={target_stage}"
            )


        validator = getattr(
            unwrap_model(model), "validate_checkpoint_contract", None
        )
        if callable(validator):
            validator(state.get("config", {}), source=str(path))
        model_state = state["model"]
        if source_stage < 3 <= target_stage:
            model_state = {
                key: value
                for key, value in model_state.items()
                if not key.startswith("heads.")
            }
        if source_stage < 4 <= target_stage:
            model_state = {
                key: value
                for key, value in model_state.items()
                if not key.startswith("quantizer.")
            }
        missing, unexpected = unwrap_model(model).load_state_dict(
            model_state, strict=False
        )
        if allow_cross_stage:


            allowed_missing_list: list[str] = []
            if source_stage < 3 <= target_stage:
                allowed_missing_list.append("heads.")
            if source_stage < 4 <= target_stage:
                allowed_missing_list.append("quantizer.")
            allowed_missing = tuple(allowed_missing_list)
        elif require_stage is not None and source_stage == require_stage:
            allowed_missing = ()
        else:
            allowed_missing = ("heads.", "quantizer.")
        bad_missing = [
            key for key in missing if not key.startswith(allowed_missing)
        ]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"Checkpoint is incompatible: missing={bad_missing}, unexpected={unexpected}"
            )
        print(
            f"checkpoint_model_state rank={os.environ.get('RANK', '0')} "
            f"seconds={time.monotonic() - model_started:.3f}",
            flush=True,
        )
    if resume:
        if optimizer is None or scheduler is None:
            raise ValueError("resume requires optimizer and scheduler")
        optimizer_started = time.monotonic()
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        print(
            f"checkpoint_optimizer_state rank={os.environ.get('RANK', '0')} "
            f"seconds={time.monotonic() - optimizer_started:.3f}",
            flush=True,
        )


        if restore_rng and state.get("rng_state") is not None:
            torch.set_rng_state(state["rng_state"])
        if (
            restore_rng
            and torch.cuda.is_available()
            and state.get("cuda_rng_state") is not None
        ):
            states = state["cuda_rng_state"]
            local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
            selected = states[local_rank % len(states)]


            torch.cuda.set_rng_state(selected, device=torch.cuda.current_device())


        return ResumeState(
            global_step=int(state["global_step"]),
            epoch=int(state.get("epoch", 0)),
            batches_consumed=int(state.get("batches_consumed", 0)),
            distributed_state=state.get("distributed_state"),
        )
    return ResumeState()
