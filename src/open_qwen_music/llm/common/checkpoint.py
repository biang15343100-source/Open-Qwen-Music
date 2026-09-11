
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .logging import source_tree_revision

CHECKPOINT_FORMAT_VERSION = "oqm.llm.ckpt.v2"


SHARDED_META_NAME = "meta.pt"


def checkpoint_path(directory: str | Path, name: str, *, sharded: bool) -> Path:
    return Path(directory) / (name if sharded else f"{name}.pt")


def is_sharded_checkpoint(path: str | Path) -> bool:
    directory = Path(path)
    return directory.is_dir() and (directory / SHARDED_META_NAME).exists()


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def resolve_base_model_revision(config: dict[str, Any]) -> str:
    model = dict(config.get("model", {}) or {})
    explicit = str(model.get("base_model_revision") or "")
    if explicit and explicit != "unknown":
        return explicit
    path = model.get("base_model_path")
    if not path:
        return "unknown"
    manifest = Path(str(path)) / "oqm_model_manifest.json"
    if not manifest.exists():
        return "unknown"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    if payload.get("format_version") != "oqm.llm.base-model-artifact.v1":
        return "unknown"
    if (payload.get("verification") or {}).get("passed") is not True:
        return "unknown"
    return str(payload.get("artifact_revision") or "unknown")


def resolve_text_tokenizer_revision(config: dict[str, Any]) -> str:

    model = dict(config.get("model", {}) or {})
    explicit = str(model.get("text_tokenizer_revision") or "")
    root_value = model.get("tokenizer_path") or model.get("base_model_path")
    if not root_value:
        return explicit if explicit and explicit != "unknown" else "unknown"
    root = Path(str(root_value))
    patterns = (
        "tokenizer*",
        "special_tokens*",
        "added_tokens*",
        "vocab*",
        "merges*",
        "spiece*",
        "sentencepiece*",
        "*.tiktoken",
    )
    files = sorted(
        {
            path
            for pattern in patterns
            for path in root.glob(pattern)
            if path.is_file()
        },
        key=lambda path: path.name,
    )
    if not files:
        return explicit if explicit and explicit != "unknown" else "unknown"
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    actual = "sha256:" + digest.hexdigest()
    if explicit and explicit != "unknown" and explicit != actual:
        raise RuntimeError(
            "model.text_tokenizer_revision does not match the tokenizer artifact: "
            f"expected={explicit}, actual={actual}"
        )
    return actual


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def build_provenance(
    config: dict[str, Any],
    *,
    registry_revision: str,
    tokenizer_revision: str,
    condition_template_version: str,
    sequence_protocol_revision: str | None = None,
    corpus_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    corpus = dict(corpus_metadata or {})
    return {
        "registry_revision": registry_revision,
        "source": {"revision": source_tree_revision()},
        "sequence_protocol_revision": sequence_protocol_revision or "unknown",
        "text_tokenizer": {
            "revision": resolve_text_tokenizer_revision(config),
        },
        "semantic_tokenizer": {
            "revision": tokenizer_revision,
            "semantic_extractor_revision": str(
                corpus.get("semantic_extractor_revision")
                or (config.get("registry") or {}).get(
                    "semantic_extractor_revision"
                )
                or "unknown"
            ),
            "frame_rate": 25.0,
            "codebook_size": 32768,
        },
        "melody_quantizer": {
            "version": "rmvpe-relative-midi-v1",
            "vocab_size": 256,
            "unvoiced_id": 255,
        },
        "condition_template_version": condition_template_version,
        "condition_policy": dict(config.get("condition", {}) or {}),
        "corpus": {
            "revision": corpus.get("corpus_revision", "unknown"),
            "manifest_sha256": corpus.get("manifest_sha256", "unknown"),
            "adapter": corpus.get("annotation_adapter", "unknown"),
            "quality_scorer": corpus.get("quality_scorer", "unknown"),
            "source_release": corpus.get("source_release", "unknown"),
            "source_lineage": corpus.get("source_lineage", "unknown"),
            "melody_tokenizer": corpus.get("melody_tokenizer", "unknown"),
        },
        "base_model": {
            "path": config.get("model", {}).get("base_model_path"),
            "revision": resolve_base_model_revision(config),
        },
        "sequence_config": {
            "max_sequence_length": config.get("sequence", {}).get("max_sequence_length"),
            "max_semantic_frames": config.get("sequence", {}).get("max_semantic_frames"),


            "max_condition_tokens": config.get("sequence", {}).get("max_condition_tokens"),


            "melody_section_policy": config.get("sequence", {}).get(
                "melody_section_policy", "lyrics_only"
            ),
            "min_section_melody_frames": config.get("sequence", {}).get(
                "min_section_melody_frames"
            ),
            "semantic_section_reanchor": config.get("sequence", {}).get(
                "semantic_section_reanchor", False
            ),
            "semantic_anchor_max_text_tokens": config.get("sequence", {}).get(
                "semantic_anchor_max_text_tokens", 256
            ),
            "semantic_anchor_boundary_policy": config.get("sequence", {}).get(
                "semantic_anchor_boundary_policy", "oracle_section_start"
            ),
            "semantic_anchor_require_complete_plan": config.get(
                "sequence", {}
            ).get("semantic_anchor_require_complete_plan", False),
            "mode_ratios": config.get("sampler", {}).get("modes"),
            "mode_ratio_scope": config.get("sampler", {}).get("mode_ratio_scope"),
            "melodic_vocal_plain_floor": config.get("sampler", {}).get(
                "melodic_vocal_plain_floor"
            ),
            "instrumental_ratio": config.get("sampler", {}).get(
                "instrumental_ratio"
            ),
        },


        "embedding": {
            "init_mode": config.get("model", {}).get("embedding_init_mode"),
            "untie_word_embeddings": config.get("model", {}).get("untie_word_embeddings"),
            "partitioned_embeddings": config.get("model", {}).get(
                "partitioned_embeddings", False
            ),
            "tie_music_embeddings": config.get("model", {}).get(
                "tie_music_embeddings", False
            ),
            "tie_embedding_namespaces": config.get("model", {}).get(
                "tie_embedding_namespaces"
            ),
            "semantic_output_residual_rank": config.get("model", {}).get(
                "semantic_output_residual_rank", 0
            ),
            "modality_type_embeddings": config.get("model", {}).get(
                "modality_type_embeddings", False
            ),
        },
        "grammar_constrained_loss": config.get("model", {}).get(
            "grammar_constrained_loss", False
        ),
        "grammar_constrained_metrics": config.get("model", {}).get(
            "grammar_constrained_metrics", False
        ),
        "region_loss_weights": config.get("model", {}).get("region_loss_weights"),
        "training": {
            "semantic_history_corruption_rate": (
                config.get("train", {}) or {}
            ).get("semantic_history_corruption_rate", 0.0),
        },


        "optimizer": {
            "embedding_lr_scale": (config.get("optimizer", {}) or {}).get(
                "embedding_lr_scale", 1.0
            ),
            "text_embedding_lr_scale": (config.get("optimizer", {}) or {}).get(
                "text_embedding_lr_scale"
            ),
            "new_embedding_lr_scale": (config.get("optimizer", {}) or {}).get(
                "new_embedding_lr_scale"
            ),
            "type_embedding_lr_scale": (config.get("optimizer", {}) or {}).get(
                "type_embedding_lr_scale"
            ),
        },
        "curriculum_stage": config.get("stage"),
    }


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    config: dict[str, Any],
    global_step: int,
    provenance: dict[str, Any],
    extra: dict[str, Any] | None = None,
    strategy: str = "ddp",
    sharded: bool = False,
) -> None:
    from .distributed import all_gather_objects, is_main_process
    from .parallel import model_state_dict, optimizer_state_dict

    rng_states_by_rank = all_gather_objects(_capture_rng_state())
    extra_by_rank = all_gather_objects(extra or {})
    meta = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": config.get("stage"),
        "global_step": global_step,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "config_hash": config_hash(config),
        "provenance": provenance,

        "extra": extra_by_rank[0],
        "extra_by_rank": extra_by_rank,
        "rng_states_by_rank": rng_states_by_rank,

        "rng_state": torch.get_rng_state(),
    }
    if sharded:
        _save_sharded(path, model=model, optimizer=optimizer, meta=meta)
        return

    gathered_model = model_state_dict(model, strategy=strategy)
    gathered_optimizer = (
        optimizer_state_dict(model, optimizer, strategy=strategy)
        if optimizer is not None
        else None
    )
    if not is_main_process():
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {**meta, "model": gathered_model, "optimizer": gathered_optimizer}
    versions = path.parent / f".{path.name}.versions"
    versions.mkdir(parents=True, exist_ok=True)
    step = int(global_step)
    generation = versions / f"step-{step:09d}-{time.time_ns()}{path.suffix}"
    staging = versions / f".{generation.name}.tmp"
    torch.save(state, staging)
    staging_sidecar = Path(str(staging) + ".json")
    _write_sidecar(staging, meta, size_bytes=staging.stat().st_size)
    generation_sidecar = Path(str(generation) + ".json")
    os.replace(staging, generation)
    os.replace(staging_sidecar, generation_sidecar)

    link_tmp = path.parent / f".{path.name}.link-{os.getpid()}"
    link_tmp.unlink(missing_ok=True)
    os.symlink(os.path.relpath(generation, path.parent), link_tmp)
    if path.exists() and not path.is_symlink():
        previous = versions / f"previous-{time.time_ns()}{path.suffix}"
        os.replace(path, previous)
    os.replace(link_tmp, path)

    compatibility = Path(str(path) + ".json")
    compatibility_tmp = Path(str(compatibility) + ".tmp")
    shutil.copy2(generation_sidecar, compatibility_tmp)
    os.replace(compatibility_tmp, compatibility)

    completed = sorted(
        (
            item
            for item in versions.glob(f"*{path.suffix}")
            if not item.name.startswith(".") and Path(str(item) + ".json").exists()
        ),
        key=lambda item: item.stat().st_mtime_ns,
    )
    for obsolete in completed[:-2]:
        obsolete.unlink(missing_ok=True)
        Path(str(obsolete) + ".json").unlink(missing_ok=True)


def _save_sharded(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    meta: dict[str, Any],
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_state_dict

    from .distributed import barrier, is_main_process

    path = Path(path)
    versions = path.parent / f".{path.name}.versions"
    step = int(meta.get("global_step") or 0)
    generation_names: list[str | None] = [
        f"step-{step:09d}-{time.time_ns()}" if is_main_process() else None
    ]


    torch.distributed.broadcast_object_list(generation_names, src=0)
    generation_name = generation_names[0]
    if not generation_name:
        raise RuntimeError("rank 0 did not broadcast a DCP generation name")
    generation = versions / generation_name
    staging = versions / f".{generation_name}.tmp"
    if is_main_process():
        path.parent.mkdir(parents=True, exist_ok=True)
        versions.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
    barrier()

    if optimizer is None:
        payload = {"model": get_model_state_dict(model)}
    else:
        model_state, optimizer_state = get_state_dict(model, optimizer)
        payload = {"model": model_state, "optimizer": optimizer_state}
    dcp.save(payload, checkpoint_id=str(staging))

    if is_main_process():
        torch.save(meta, staging / SHARDED_META_NAME)
        total = sum(f.stat().st_size for f in staging.iterdir() if f.is_file())

        _write_sidecar(staging, meta, size_bytes=total)
        os.replace(staging, generation)

        link_tmp = path.parent / f".{path.name}.link-{os.getpid()}"
        link_tmp.unlink(missing_ok=True)
        relative = os.path.relpath(generation, path.parent)
        os.symlink(relative, link_tmp, target_is_directory=True)
        if path.exists() and not path.is_symlink():

            previous = versions / f"previous-{time.time_ns()}"
            os.replace(path, previous)
        os.replace(link_tmp, path)


        compatibility = Path(str(path) + ".json")
        compatibility_tmp = Path(str(compatibility) + ".tmp")
        shutil.copy2(generation / "sidecar.json", compatibility_tmp)
        os.replace(compatibility_tmp, compatibility)


        completed = sorted(
            (
                item
                for item in versions.iterdir()
                if item.is_dir()
                and (item / SHARDED_META_NAME).exists()
                and (item / "sidecar.json").exists()
                and item != generation
            ),
            key=lambda item: item.stat().st_mtime_ns,
        )
        for obsolete in completed[:-1]:
            shutil.rmtree(obsolete, ignore_errors=True)
    barrier()


def _write_sidecar(path: Path, meta: dict[str, Any], *, size_bytes: int) -> None:
    sidecar = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": meta.get("stage"),
        "global_step": meta.get("global_step"),
        "config_hash": meta.get("config_hash"),
        "provenance": meta.get("provenance"),
        "extra": meta.get("extra") or {},
        "extra_by_rank": meta.get("extra_by_rank") or [],
        "sharded": is_sharded_checkpoint(path),
        "checkpoint_size_bytes": size_bytes,
    }
    target = path / "sidecar.json" if is_sharded_checkpoint(path) else Path(str(path) + ".json")
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def stage_checkpoint_locally(
    path: str | Path,
    *,
    cache_dir: str | Path = "/tmp/open_qwen_music.llm/checkpoints",
    retries: int = 5,
) -> Path:
    source = Path(path).resolve()
    stat = source.stat()
    fingerprint = hashlib.sha256(
        f"{source}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()[:20]
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    local = directory / f"{source.stem}.{fingerprint}{source.suffix}"
    lock_path = directory / f"{local.name}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for attempt in range(1, retries + 1):
            if local.exists() and local.stat().st_size == stat.st_size:
                try:
                    state = torch.load(
                        local, map_location="cpu", weights_only=False, mmap=True
                    )
                    if "model" in state and "format_version" in state:
                        return local
                except Exception:
                    local.unlink(missing_ok=True)
            temporary = directory / f".{local.name}.{os.getpid()}.{attempt}.tmp"
            try:
                with source.open("rb") as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=32 * 1024 * 1024)
                    dst.flush()
                    os.fsync(dst.fileno())
                if temporary.stat().st_size != stat.st_size:
                    raise OSError(
                        f"checkpoint size does not match {temporary.stat().st_size}!={stat.st_size}"
                    )
                state = torch.load(
                    temporary, map_location="cpu", weights_only=False, mmap=True
                )
                if "model" not in state or "format_version" not in state:
                    raise RuntimeError("Checkpoint is missing a required field")
                os.replace(temporary, local)
                return local
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt >= retries:
                    raise
                time.sleep(float(attempt) * 2.0)
        raise RuntimeError(f"cannot stage checkpoint: {source}")


#:


#:


PROVENANCE_WARN_ONLY: tuple[str, ...] = (
    "curriculum_stage",
    "sequence_config.max_sequence_length",
    "sequence_config.max_semantic_frames",
    "sequence_config.max_condition_tokens",
    "sequence_config.mode_ratios",
    "sequence_config.mode_ratio_scope",
    "sequence_config.melodic_vocal_plain_floor",
    "sequence_config.instrumental_ratio",
    "region_loss_weights",
    "optimizer.embedding_lr_scale",
    "optimizer.text_embedding_lr_scale",
    "optimizer.new_embedding_lr_scale",
    "optimizer.type_embedding_lr_scale",
    "training.semantic_history_corruption_rate",
    "grammar_constrained_metrics",


    "corpus.revision",
    "corpus.manifest_sha256",
)


PROVENANCE_NEVER_BYPASS: tuple[str, ...] = (
    "registry_revision",
    "source.revision",
    "text_tokenizer.revision",
    "semantic_tokenizer.revision",
    "semantic_tokenizer.semantic_extractor_revision",
    "semantic_tokenizer.frame_rate",
    "semantic_tokenizer.codebook_size",
)


PROVENANCE_BACKFILL_DEFAULTS: dict[str, Any] = {
    "semantic_tokenizer.semantic_extractor_revision": "unknown",
    "embedding.partitioned_embeddings": False,
    "embedding.tie_music_embeddings": False,
    "embedding.tie_embedding_namespaces": None,
    "embedding.semantic_output_residual_rank": 0,
    "embedding.modality_type_embeddings": False,
    "grammar_constrained_loss": False,
    "grammar_constrained_metrics": False,
    "sequence_config.semantic_section_reanchor": False,
    "sequence_config.semantic_anchor_max_text_tokens": 256,
    "sequence_config.semantic_anchor_boundary_policy": "oracle_section_start",
    "sequence_config.semantic_anchor_require_complete_plan": False,
    "training.semantic_history_corruption_rate": 0.0,
    "optimizer.text_embedding_lr_scale": None,
    "optimizer.new_embedding_lr_scale": None,
    "optimizer.type_embedding_lr_scale": None,
}


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            flat.update(_flatten(value, f"{prefix}{key}."))
    else:
        flat[prefix.rstrip(".")] = payload
    return flat


def _compare_provenance(
    expected: dict[str, Any], found: dict[str, Any]
) -> tuple[list[str], list[str]]:
    fatal: list[str] = []
    warnings: list[str] = []
    expected_flat = _flatten(expected)
    found_flat = _flatten(found)
    for path in sorted(set(expected_flat) | set(found_flat)):
        default = PROVENANCE_BACKFILL_DEFAULTS.get(path, "<missing>")
        mine = expected_flat.get(path, default)
        theirs = found_flat.get(path, default)
        if mine == theirs:
            continue
        message = f"{path} differs: current {mine!r} vs checkpoint {theirs!r}"
        if any(
            path == allowed or path.startswith(allowed + ".")
            for allowed in PROVENANCE_WARN_ONLY
        ):
            warnings.append(message)
        else:
            fatal.append(message)
    return fatal, warnings


def _check_provenance(
    state: dict[str, Any],
    expected_provenance: dict[str, Any] | None,
    allow_provenance_mismatch: bool,
    *,
    resume: bool = False,
) -> None:
    if expected_provenance is None:
        return
    problems, warnings = _compare_provenance(
        expected_provenance, state.get("provenance") or {}
    )
    if not resume:


        problems = [
            message
            for message in problems
            if not message.startswith("parent_checkpoint.")
        ]
        warnings = [
            message
            for message in warnings
            if not message.startswith("parent_checkpoint.")
        ]
    resume_fatal: list[str] = []
    if resume:
        resume_fatal = [
            message for message in warnings if message.startswith("corpus.")
        ]
        problems.extend(resume_fatal)
        warnings = [
            message for message in warnings if not message.startswith("corpus.")
        ]
    if resume_fatal:
        raise RuntimeError(
            "Resume checkpoint and current corpus differ; the sampler cursor cannot be interpreted safely:\n  - "
            + "\n  - ".join(resume_fatal)
        )
    if warnings:


        print(
            "Checkpoint provenance differences (warning only):\n  - "
            + "\n  - ".join(warnings),
            flush=True,
        )
    if not problems:
        return
    never_bypass = [
        problem
        for problem in problems
        if any(
            problem.startswith(path + " differs:")
            for path in PROVENANCE_NEVER_BYPASS
        )
    ]
    if never_bypass:
        raise RuntimeError(
            "Checkpoint token provenance does not match:\n  - "
            + "\n  - ".join(never_bypass)
            + "\nStart from fresh weights with the matching tokenizer and registry; "
            "train.allow_provenance_mismatch cannot bypass this error."
        )
    message = (
        "Checkpoint provenance verification failed:\n  - "
        + "\n  - ".join(problems)
        + "\nThese fields change the meaning of IDs stored in the checkpoint"
        " (vocabulary layout, tokenizer scale, conditioning template, or base weights)."
        " To continue, set train.allow_provenance_mismatch=true."
    )
    if not allow_provenance_mismatch:
        raise RuntimeError(message)
    print(f"WARNING {message}", flush=True)


def _check_checkpoint_format(state: dict[str, Any], *, resume: bool) -> None:
    version = str(state.get("format_version") or "")
    if version == CHECKPOINT_FORMAT_VERSION:
        return
    raise RuntimeError(
        f"checkpoint format={version!r} does not support the current {'resume' if resume else 'init'};"
        f"expects {CHECKPOINT_FORMAT_VERSION!r}"
    )


def _restore_rng(state: dict[str, Any]) -> None:
    per_rank = state.get("rng_states_by_rank")
    if per_rank:
        current_rank = int(os.environ.get("RANK", "0"))
        current_world = int(os.environ.get("WORLD_SIZE", "1"))
        if len(per_rank) != current_world:
            raise RuntimeError(
                f"Checkpoint saved RNG state for {len(per_rank)} ranks, but current "
                f"world_size={current_world}. DCP weights can load across world sizes, "
                "but exact optimizer and RNG continuation cannot. Use --init-from to "
                "load weights only, or keep the original world size."
            )
        selected = per_rank[current_rank]
        random.setstate(selected["python"])
        np.random.set_state(selected["numpy"])
        torch.set_rng_state(selected["torch_cpu"].cpu().to(torch.uint8))
        if torch.cuda.is_available() and selected.get("torch_cuda") is not None:
            torch.cuda.set_rng_state(
                selected["torch_cuda"].cpu().to(torch.uint8),
                device=torch.cuda.current_device(),
            )
        return
    if state.get("rng_state") is not None:
        torch.set_rng_state(state["rng_state"].cpu().to(torch.uint8))
    if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
        states = state["cuda_rng_state"]
        local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))


        torch.cuda.set_rng_state(
            states[local_rank % len(states)], device=torch.cuda.current_device()
        )


def _load_sharded(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    resume: bool,
    load_model: bool,
    expected_provenance: dict[str, Any] | None,
    allow_provenance_mismatch: bool,
) -> int:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_state_dict,
        set_model_state_dict,
        set_state_dict,
    )

    started = time.monotonic()
    meta = torch.load(path / SHARDED_META_NAME, map_location="cpu", weights_only=False)
    _check_checkpoint_format(meta, resume=resume)
    _check_provenance(
        meta,
        expected_provenance,
        allow_provenance_mismatch,
        resume=resume,
    )

    if resume:
        if optimizer is None:
            raise ValueError("resume requires optimizer")
        model_state, optimizer_state = get_state_dict(model, optimizer)
        payload = {"model": model_state, "optimizer": optimizer_state}
        dcp.load(payload, checkpoint_id=str(path))
        set_state_dict(
            model,
            optimizer,
            model_state_dict=payload["model"],
            optim_state_dict=payload["optimizer"],
        )
    elif load_model:
        payload = {"model": get_model_state_dict(model)}
        dcp.load(payload, checkpoint_id=str(path))
        set_model_state_dict(model, payload["model"])

    print(
        f"checkpoint_load rank={os.environ.get('RANK', '0')} path={path} sharded=1 "
        f"resume={int(resume)} seconds={time.monotonic() - started:.3f}",
        flush=True,
    )
    if not resume:
        return 0
    if scheduler is not None and meta.get("scheduler") is not None:
        scheduler.load_state_dict(meta["scheduler"])
    _restore_rng(meta)
    return int(meta["global_step"])


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    resume: bool = False,
    load_model: bool = True,
    expected_provenance: dict[str, Any] | None = None,
    allow_provenance_mismatch: bool = False,
    strategy: str = "ddp",
) -> int:
    from .parallel import load_model_state_dict, load_optimizer_state_dict

    if is_sharded_checkpoint(path):
        return _load_sharded(
            Path(path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            resume=resume,
            load_model=load_model,
            expected_provenance=expected_provenance,
            allow_provenance_mismatch=allow_provenance_mismatch,
        )

    started = time.monotonic()
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    _check_checkpoint_format(state, resume=resume)
    print(
        f"checkpoint_load rank={os.environ.get('RANK', '0')} path={path} "
        f"seconds={time.monotonic() - started:.3f}",
        flush=True,
    )
    _check_provenance(
        state,
        expected_provenance,
        allow_provenance_mismatch,
        resume=resume,
    )

    if load_model:
        model_started = time.monotonic()
        missing, unexpected = load_model_state_dict(
            model, state["model"], strategy=strategy, strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint model state is incompatible: missing={list(missing)[:8]}, "
                f"unexpected={list(unexpected)[:8]}"
            )
        print(
            f"checkpoint_model_state rank={os.environ.get('RANK', '0')} "
            f"seconds={time.monotonic() - model_started:.3f}",
            flush=True,
        )

    if resume:
        if optimizer is None:
            raise ValueError("resume requires optimizer")
        load_optimizer_state_dict(model, optimizer, state["optimizer"], strategy=strategy)
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        _restore_rng(state)
        return int(state["global_step"])
    return 0


def read_sidecar(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    internal = checkpoint / "sidecar.json"
    resolved_sidecar = (
        Path(str(checkpoint.resolve()) + ".json") if checkpoint.is_symlink() else None
    )
    sidecar = (
        internal
        if internal.exists()
        else resolved_sidecar
        if resolved_sidecar is not None and resolved_sidecar.exists()
        else Path(str(checkpoint) + ".json")
    )
    if not sidecar.exists():
        raise FileNotFoundError(f"Missing sidecar: {sidecar}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    expected_size = payload.get("checkpoint_size_bytes")
    if checkpoint.is_file() and expected_size is not None:
        actual = checkpoint.stat().st_size
        if actual != int(expected_size):
            raise ValueError(
                f"Checkpoint and sidecar identities differ: {checkpoint} size={actual}, "
                f"sidecar={expected_size}"
            )
    return payload
