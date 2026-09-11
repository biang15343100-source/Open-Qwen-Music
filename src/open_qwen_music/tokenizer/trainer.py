
from __future__ import annotations

import json
import importlib
import importlib.metadata
import math
import os
import platform
import random
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from open_qwen_music.common.checkpoint import (
    checkpoint_runtime_identity,
    code_revision_identity,
    config_hash,
    distributed_training_state,
    file_sha256,
    load_checkpoint,
    retain_checkpoint,
    resume_config_hash,
    save_checkpoint,
    stage_checkpoint_locally,
)
from open_qwen_music.common.distributed import (
    assert_distributed_consensus,
    barrier,
    cleanup_distributed,
    init_distributed,
    is_main_process,
    raise_if_rank0_error,
    reduce_metrics,
    reduce_scalar_sum,
)

from .data import (
    DistributedBalancedDurationBucketBatchSampler,
    DistributedDurationBucketBatchSampler,
    ResumableDistributedSampler,
    TokenizerCollator,
    TokenizerDataset,
)
from .contracts import SAMPLE_RATE
from .features import (
    CHROMA_DEFAULTS,
    CHROMA_PRE_20260804_DEFAULTS,
    FRONTEND_DEFAULTS,
    resolve_chroma_config,
)
from .frontend import (
    CAUSALITY_FIELDS,
    SUBSAMPLING_CONTRACT_DEFAULTS,
    SUBSAMPLING_CONTRACT_PRE_20260801_DEFAULTS,
    resolve_causality,
    resolve_subsampling_contract,
)
from .manifest_validation import validate_ready_release
from .model import MusicTokenizer
from .initialization import validate_initialization_artifact
from .text import CharacterTokenizer


class _RankWatchdog:

    def __init__(
        self,
        *,
        output_dir: Path,
        rank: int,
        local_rank: int,
        timeout_sec: float,
        poll_sec: float,
    ) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self.timeout_sec = timeout_sec
        self.poll_sec = max(1.0, poll_sec)
        self.directory = output_dir / "watchdog"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / f"rank_{rank:04d}.json"
        self.stack_path = self.directory / f"rank_{rank:04d}.stacks.log"
        self._lock = threading.Lock()
        self._last_heartbeat = time.monotonic()
        self._last_dump_heartbeat = -1.0
        self._state: dict[str, Any] = {
            "rank": rank,
            "local_rank": local_rank,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "phase": "initializing",
            "step": 0,
            "micro_step": 0,
            "sample_ids": [],
        }
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if timeout_sec > 0:
            self._thread = threading.Thread(
                target=self._run,
                name=f"tokenizer-watchdog-rank-{rank}",
                daemon=True,
            )
            self._thread.start()

    def heartbeat(
        self,
        phase: str,
        *,
        step: int,
        micro_step: int,
        sample_ids: list[str] | None = None,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            self._last_heartbeat = now
            self._state.update(
                phase=phase,
                step=step,
                micro_step=micro_step,
                sample_ids=list(sample_ids or []),
                heartbeat_unix=time.time(),
            )

    def _snapshot(self) -> tuple[float, dict[str, Any]]:
        with self._lock:
            return self._last_heartbeat, dict(self._state)

    def _run(self) -> None:
        while not self._stop.wait(self.poll_sec):
            heartbeat, state = self._snapshot()
            stalled_for = time.monotonic() - heartbeat
            if stalled_for < self.timeout_sec or heartbeat == self._last_dump_heartbeat:
                continue
            self._last_dump_heartbeat = heartbeat
            state.update(stalled_for_sec=stalled_for, detected_unix=time.time())
            temporary = self.state_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temporary, self.state_path)
            with self.stack_path.open("a", encoding="utf-8") as output:
                output.write(
                    "\n" + "=" * 80 + f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"rank={self.rank} stalled_for={stalled_for:.1f}s "
                    f"phase={state['phase']} step={state['step']} "
                    f"sample_ids={state['sample_ids']}\n"
                )
                import faulthandler

                faulthandler.dump_traceback(file=output, all_threads=True)
            print(
                f"[watchdog] rank={self.rank} has had no heartbeat for more than {self.timeout_sec:.0f}s; "
                f"phase={state['phase']} step={state['step']} "
                f"sample_ids={state['sample_ids']}; diagnostic written to {self.state_path}",
                file=sys.stderr,
                flush=True,
            )

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_sec + 1.0)


def _seed_everything(seed: int, rank: int, step: int = 0) -> None:

    seed += rank + 7919 * step
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _configure_sdp_backend(config: dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        return
    backend = str(config["train"].get("sdp_backend", "auto"))
    if backend == "auto":
        return
    if backend != "math":
        raise ValueError(f"sdp_backend must be auto or math; received {backend}")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def _move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    *,
    peak_lr: float,
    min_lr: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    if peak_lr <= 0:
        raise ValueError(f"optimizer.lr must be greater than zero; received {peak_lr}")
    if min_lr < 0 or min_lr > peak_lr:
        raise ValueError(
            "optimizer.min_lr must satisfy 0 <= min_lr <= lr; "
            f"received min_lr={min_lr}, lr={peak_lr}"
        )
    floor = min_lr / peak_lr

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _gate_alpha(config: dict[str, Any], step: int) -> float:
    if int(config["stage"]) != 4:
        return 0.0
    warmup = int(config["quantizer"].get("gate_warmup_steps", 0))
    if warmup <= 0:
        return 1.0
    return min(1.0, step / warmup)


def _loss_weights(config: dict[str, Any], step: int) -> dict[str, float]:
    weights = {key: float(value) for key, value in config["loss_weights"].items()}
    alignment_only_steps = int(config.get("probe", {}).get("alignment_only_steps", 0))
    if int(config["stage"]) == 3 and step < alignment_only_steps:
        weights["ctc"] = 0.0
        weights["ctc_alignment"] = float(
            config.get("probe", {}).get("alignment_warmup_weight", 1.0)
        )
    elif int(config["stage"]) == 3 and alignment_only_steps > 0:


        weights["ctc_alignment"] = float(
            config.get("probe", {}).get("alignment_post_weight", 0.0)
        )
    return weights


def _clear_frozen_stage4_gradients(
    model: torch.nn.Module, config: dict[str, Any], step: int
) -> None:
    if int(config["stage"]) != 4:
        return
    insertion_layer = int(config["model"]["quantizer_insertion_layer"])
    root = model.module if hasattr(model, "module") else model
    frozen_lower_steps = int(config["quantizer"].get("frozen_lower_steps", 0))
    if step < frozen_lower_steps:
        for name, parameter in root.named_parameters():
            freeze = name.startswith(("feature_extractor.", "subsampling."))
            if name.startswith("encoder.layers."):
                layer_index = int(name.split(".")[2])
                freeze = freeze or layer_index < insertion_layer
            if freeze:
                parameter.grad = None
    projection_steps = int(config["quantizer"].get("freeze_input_projection_steps", 0))
    if step < projection_steps:
        for parameter in root.quantizer.input_proj.parameters():
            parameter.grad = None


def _module_gradient_norm(module: torch.nn.Module) -> torch.Tensor:

    total: torch.Tensor | None = None
    fallback: torch.Tensor | None = None
    for parameter in module.parameters():
        if fallback is None:
            fallback = parameter.detach().new_zeros((), dtype=torch.float32)
        if parameter.grad is None:
            continue
        squared = parameter.grad.detach().float().square().sum()
        total = squared if total is None else total + squared
    if total is None:
        return fallback if fallback is not None else torch.zeros(())
    return total.sqrt()


def _head_gradient_metrics(model: torch.nn.Module) -> dict[str, torch.Tensor]:

    root = model.module if hasattr(model, "module") else model
    heads = root.heads
    return {
        "grad_norm_ctc_head": _module_gradient_norm(heads.ctc),
        "grad_norm_mel_head": _module_gradient_norm(heads.mel),
        "grad_norm_chroma_head": _module_gradient_norm(heads.chroma),
    }


def _configure_stage4_quantizer_step(
    model: torch.nn.Module, config: dict[str, Any], step: int
) -> None:
    if int(config["stage"]) != 4:
        return
    root = model.module if hasattr(model, "module") else model
    configured_beta = float(config["quantizer"].get("diversity_beta", 0.0))
    start_step = int(config["quantizer"].get("diversity_start_step", 0))
    root.quantizer.diversity_beta = configured_beta if step >= start_step else 0.0


def _configure_probe_trainability(
    model: MusicTokenizer, config: dict[str, Any]
) -> None:
    probe = config.get("probe", {})
    freeze_backbone = bool(probe.get("freeze_backbone", False))
    configured_heads = probe.get("trainable_heads")
    if configured_heads is not None and not freeze_backbone:
        raise ValueError("probe.trainable_heads is allowed only when freeze_backbone=true")
    if not freeze_backbone:
        return
    available_heads = {
        "ctc": model.heads.ctc,
        "mel": model.heads.mel,
        "chroma": model.heads.chroma,
    }
    if configured_heads is None:
        trainable_heads = set(available_heads)
    else:
        if not isinstance(configured_heads, list) or not configured_heads:
            raise ValueError("probe.trainable_heads must be a non-empty list")
        trainable_heads = {str(name) for name in configured_heads}
        unknown = sorted(trainable_heads - set(available_heads))
        if unknown:
            raise ValueError(f"probe.trainable_heads contains an unknown head: {unknown}")
    for module in (model.feature_extractor, model.subsampling, model.encoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)


    for name, module in available_heads.items():
        for parameter in module.parameters():
            parameter.requires_grad_(name in trainable_heads)


def _configure_probe_module_modes(
    model: MusicTokenizer | DistributedDataParallel,
    config: dict[str, Any],
) -> None:

    if not bool(config.get("probe", {}).get("freeze_backbone", False)):
        return
    root = model.module if isinstance(model, DistributedDataParallel) else model
    for module in (root.feature_extractor, root.subsampling, root.encoder):
        module.eval()
    root.heads.train()
    configured_heads = config.get("probe", {}).get("trainable_heads")
    if configured_heads is not None:
        trainable_heads = {str(name) for name in configured_heads}
        for name in ("ctc", "mel", "chroma"):
            if name not in trainable_heads:
                getattr(root.heads, name).eval()


def _write_run_metadata(
    output_dir: Path,
    config: dict[str, Any],
    world_size: int,
    *,
    provenance: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "stage": int(config["stage"]),
        "world_size": world_size,
        "host_num": os.environ.get("HOST_NUM"),
        "host_gpu_num": os.environ.get("HOST_GPU_NUM"),
        "node_ip_list": os.environ.get("NODE_IP_LIST"),
        "selected_node_indices": os.environ.get("OQM_SELECTED_NODE_INDICES"),
        "selected_node_ips": os.environ.get("OQM_SELECTED_NODE_IPS"),
        "master_addr": os.environ.get("MASTER_ADDR"),
        "master_port": os.environ.get("MASTER_PORT"),
        "run_id": os.environ.get("OQM_RUN_ID"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "config_hash": config_hash(config),
        "resume_config_hash": resume_config_hash(config),
        "provenance": provenance,
        "config": config,
    }
    (output_dir / "run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _lineage_asset_specs(config: dict[str, Any]) -> list[tuple[str, Path, str]]:
    lineage = config.get("lineage", {})
    specs: list[tuple[str, Path, str]] = []
    manifest_source = lineage.get("manifest_source") or config.get("data", {}).get(
        "manifest"
    )
    manifest_sha = lineage.get("manifest_sha256")
    if manifest_source and manifest_sha:
        configured_manifest = config.get("data", {}).get("manifest")
        if (
            configured_manifest
            and Path(configured_manifest).resolve() != Path(manifest_source).resolve()
        ):
            raise RuntimeError(
                "lineage.manifest_source and data.manifest differ: "
                f"{manifest_source} != {configured_manifest}"
            )
        specs.append(("manifest", Path(manifest_source).resolve(), str(manifest_sha)))
    stats_source = lineage.get("feature_stats_source")
    stats_sha = lineage.get("feature_stats_sha256")
    if stats_source and stats_sha:
        specs.append(("feature_stats", Path(stats_source).resolve(), str(stats_sha)))
    return specs


def _asset_runtime_identity(config: dict[str, Any]) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    for name, path, expected_sha in _lineage_asset_specs(config):
        stat = path.stat()
        identity[name] = {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": expected_sha,
        }
    return identity


def _verify_lineage_assets(config: dict[str, Any]) -> None:
    specs = _lineage_asset_specs(config)
    if bool(config["train"].get("require_asset_sha256", False)) and len(specs) < 2:
        raise RuntimeError(
            "Training requires SHA-256 lineage for both the manifest and feature_stats"
        )
    mismatches = []
    for name, path, expected_sha in specs:
        if not path.is_file():
            mismatches.append(f"{name}: file does not exist: {path}")
            continue
        actual_sha = file_sha256(path)
        if actual_sha != expected_sha:
            mismatches.append(
                f"{name}: SHA-256 mismatch: expected={expected_sha} actual={actual_sha}"
            )
    if mismatches:
        raise RuntimeError("Training asset identity verification failed:" + "; ".join(mismatches))


def _validate_distributed_contract(config: dict[str, Any], *, world_size: int) -> None:
    train_config = config["train"]
    target_global_batch = train_config.get("target_global_batch_size")
    if target_global_batch is not None:
        per_update = world_size * int(train_config["batch_size_per_rank"])
        target_global_batch = int(target_global_batch)
        if target_global_batch < per_update or target_global_batch % per_update:
            raise RuntimeError(
                "train.target_global_batch_size must be divisible by "
                "world_size * batch_size_per_rank: "
                f"target={target_global_batch} divisor={per_update}"
            )
        required_accumulation = target_global_batch // per_update
        configured = train_config.get("gradient_accumulation_steps", "auto")
        if configured not in (None, "auto", required_accumulation):
            raise RuntimeError(
                "gradient_accumulation_steps conflicts with target_global_batch_size: "
                f"configured={configured} required={required_accumulation}"
            )
        train_config["gradient_accumulation_steps"] = required_accumulation
    expected_world = train_config.get("expected_world_size")
    if expected_world is not None and int(expected_world) != world_size:
        raise RuntimeError(
            f"world size differs from the configuration: expected={expected_world} actual={world_size}"
        )
    global_batch = (
        world_size
        * int(train_config["batch_size_per_rank"])
        * int(train_config.get("gradient_accumulation_steps", 1))
    )
    expected_global_batch = train_config.get("expected_global_batch_size")
    if expected_global_batch is not None and int(expected_global_batch) != global_batch:
        raise RuntimeError(
            "global batch size differs from the configured target: "
            f"expected={expected_global_batch} actual={global_batch}"
        )


def _validate_code_identity(config: dict[str, Any], identity: dict[str, Any]) -> None:
    if not bool(config["train"].get("require_clean_code", False)):
        return
    if not identity.get("available"):
        raise RuntimeError(f"Unable to read the code revision: {identity.get('error')}")
    if identity.get("dirty"):
        raise RuntimeError(
            "Training requires a clean workspace; current state: "
            f"tracked_diff_bytes={identity.get('tracked_diff_bytes')} "
            f"untracked_files={identity.get('untracked_files')}"
        )


def _environment_identity(config: dict[str, Any]) -> dict[str, Any]:
    required = [
        str(name) for name in config["train"].get("required_python_modules", [])
    ]
    missing = []
    for name in required:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise RuntimeError(f"The training environment is missing Python modules: {missing}")
    packages = {}
    for distribution in (
        "torch",
        "PyYAML",
        "numpy",
        "soundfile",
        "tokenizers",
        "pyarrow",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    libsndfile = None
    if "soundfile" in required:
        soundfile = importlib.import_module("soundfile")
        libsndfile = getattr(soundfile, "__libsndfile_version__", None)
    nccl_version = None
    if torch.cuda.is_available():
        try:
            nccl_version = torch.cuda.nccl.version()
        except (AttributeError, RuntimeError):
            nccl_version = None
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "packages": packages,
        "libsndfile_version": libsndfile,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "nccl_version": nccl_version,
    }


def _distributed_runtime_identity() -> dict[str, Any]:

    communication_keys = (
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_SL",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "NCCL_NET_GDR_LEVEL",
        "NCCL_IB_QPS_PER_CONNECTION",
        "NCCL_IB_TC",
        "NCCL_PXN_DISABLE",
        "UCX_NET_DEVICES",
        "GLOO_SOCKET_IFNAME",
    )
    return {
        "launch_backend": os.environ.get("OQM_LAUNCH_BACKEND", "direct"),
        "nnodes": os.environ.get("OQM_NNODES"),
        "nproc_per_node": os.environ.get("OQM_NPROC_PER_NODE"),
        "master_addr": os.environ.get("OQM_MASTER_ADDR", os.environ.get("MASTER_ADDR")),
        "communication": {key: os.environ.get(key) for key in communication_keys},
    }


def _validate_checkpoint_provenance(
    config: dict[str, Any],
    *,
    current: dict[str, Any],
    checkpoint_identity: dict[str, Any],
) -> None:
    if not bool(config["train"].get("require_checkpoint_provenance", False)):
        return
    saved = checkpoint_identity.get("provenance")
    if not saved:
        raise RuntimeError(
            "checkpoint is missing provenance required for an exact resume"
        )
    expected_workspace = saved.get("code", {}).get("workspace_sha256")
    actual_workspace = current.get("code", {}).get("workspace_sha256")
    if not expected_workspace or expected_workspace != actual_workspace:
        raise RuntimeError(
            "checkpoint code revision differs from the current workspace: "
            f"checkpoint={expected_workspace} current={actual_workspace}"
        )
    if saved.get("assets") != current.get("assets"):
        raise RuntimeError("checkpoint training assets do not match the current manifest and CMVN")
    if saved.get("environment") != current.get("environment"):
        raise RuntimeError("checkpoint Python, CUDA, or NCCL metadata differs from this runtime")


def _validate_init_checkpoint_identity(
    config: dict[str, Any], checkpoint_identity: dict[str, Any]
) -> None:

    expected = (config.get("lineage") or {}).get("curriculum_parent_checkpoint_sha256")
    if expected is None:
        return
    actual = checkpoint_identity.get("sha256")
    if actual != expected:
        raise RuntimeError(
            "The initialization checkpoint SHA-256 differs from the configured lineage: "
            f"current={actual} expected={expected}"
        )


def _checkpoint_lineage_record(
    source_path: str | Path,
    checkpoint_identity: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:

    if mode not in {"init", "resume"}:
        raise ValueError(f"Unsupported checkpoint lineage mode={mode!r}")
    required = ("sha256", "size", "stage", "global_step")
    missing = [key for key in required if checkpoint_identity.get(key) is None]
    if missing:
        raise ValueError(f"checkpoint runtime identity missing field:{missing}")
    return {
        "mode": mode,
        "source_path": str(Path(source_path).resolve()),
        "sha256": str(checkpoint_identity["sha256"]),
        "size": int(checkpoint_identity["size"]),
        "format_version": checkpoint_identity.get("format_version"),
        "stage": int(checkpoint_identity["stage"]),
        "global_step": int(checkpoint_identity["global_step"]),
        "config_hash": checkpoint_identity.get("config_hash"),
        "resume_config_hash": checkpoint_identity.get("resume_config_hash"),
    }


def _validate_resume_topology(
    config: dict[str, Any], saved: dict[str, Any] | None
) -> None:
    current = distributed_training_state(config)
    if saved is None:
        raise RuntimeError(
            "checkpoint is missing distributed_state, so an exact resume of the world "
            "size, batch state, sampler, and manifest cannot be verified."
        )
    checked = (
        "world_size",
        "batch_size_per_rank",
        "gradient_accumulation_steps",
        "sampler_kind",
        "manifest",
    )
    mismatches = {
        key: (current.get(key), saved.get(key))
        for key in checked
        if current.get(key) != saved.get(key)
    }
    if mismatches:
        raise RuntimeError(
            "The resume topology differs from the checkpoint; batches_consumed "
            "cannot be applied to the new sampler: "
            f"{mismatches}"
        )


def _validate_resume_config_hash(
    config: dict[str, Any], checkpoint_identity: dict[str, Any]
) -> None:
    expected = checkpoint_identity["resume_config_hash"]
    actual = resume_config_hash(config)
    if expected == actual:
        return
    if bool(config["train"].get("allow_resume_config_mismatch", False)):
        return
    raise RuntimeError(
        "The resume configuration differs from the checkpoint: "
        f"current={actual} checkpoint={expected}. Changes to data, learning rate, or "
        "max_steps cannot reuse the saved optimizer and scheduler state. Set "
        "train.allow_resume_config_mismatch=true only when this limitation is acceptable."
    )


def _stats_to_config(value: Any) -> float | list[float]:

    if isinstance(value, torch.Tensor):
        return float(value.item()) if value.numel() == 1 else value.flatten().tolist()
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return float(value)


def _validate_subsampling_contract_from_checkpoint(
    config: dict[str, Any], checkpoint: str | None
) -> None:

    if not checkpoint:
        return
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    source_config = state.get("config", {}).get("model")
    if not source_config:
        raise RuntimeError(f"upstream checkpoint is missing model config: {checkpoint}")
    target = resolve_subsampling_contract(
        config["model"], SUBSAMPLING_CONTRACT_DEFAULTS
    )
    source = resolve_subsampling_contract(
        source_config, SUBSAMPLING_CONTRACT_PRE_20260801_DEFAULTS
    )
    mismatches = {
        key: (target[key], source[key])
        for key in SUBSAMPLING_CONTRACT_DEFAULTS
        if target[key] != source[key]
    }
    if mismatches:
        raise RuntimeError(
            f"Subsampling frontend contract mismatch "
            f"(configuration value, checkpoint value): {mismatches}"
        )
    target_model = config["model"]
    explicit_causality = any(
        key in target_model or key in source_config for key in CAUSALITY_FIELDS
    )
    if explicit_causality:
        target_causality = resolve_causality(target_model)
        source_causality = resolve_causality(source_config)
        source_stage = int(
            (state.get("config") or {}).get("stage", state.get("stage", 0)) or 0
        )
        diagnostic_probe_override = bool(
            config.get("probe", {}).get("allow_stage1_attention_causal_override", False)
        )
        allowed_attention_change = source_stage == 1 and (
            int(config["stage"]) == 2
            or (int(config["stage"]) == 3 and diagnostic_probe_override)
        )
        checked_fields = (
            ("frontend_causal", "conformer_conv_causal")
            if allowed_attention_change
            else CAUSALITY_FIELDS
        )
        causality_mismatches = {
            key: (target_causality[key], source_causality[key])
            for key in checked_fields
            if target_causality[key] != source_causality[key]
        }
        if causality_mismatches:
            raise RuntimeError(
                "Attention/convolution causality contract mismatch "
                "(configuration value, checkpoint value): "
                f"{causality_mismatches}"
            )


def _inherit_feature_config_from_checkpoint(
    config: dict[str, Any],
    checkpoint: str | None,
    *,
    lineage_checkpoint: str | None = None,
) -> None:

    if not checkpoint:
        return
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    source_config = state.get("config", {}).get("features")
    if not source_config:
        raise RuntimeError(f"upstream checkpoint is missing features config: {checkpoint}")
    immutable = (
        "sample_rate",
        "n_fft",
        "hop_length",
        "win_length",
        "n_mels",
        "f_min",
        "f_max",
        "log_floor",
        "mel_filter_norm",
        "mel_scale",
        "log_mode",
        "log_knee",
        "top_db",
        "top_db_scope",
        "feature_centering",
    )


    def resolved(mapping: dict[str, Any], key: str, defaults: dict[str, Any]) -> Any:
        value = mapping.get(key)
        return defaults[key] if value is None and key in defaults else value

    mismatches = {
        key: (
            resolved(config["features"], key, FRONTEND_DEFAULTS),
            resolved(source_config, key, FRONTEND_DEFAULTS),
        )
        for key in immutable
        if resolved(config["features"], key, FRONTEND_DEFAULTS)
        != resolved(source_config, key, FRONTEND_DEFAULTS)
    }
    if mismatches:
        raise RuntimeError(f"Upstream frontend contract mismatch: {mismatches}")


    target_chroma = resolve_chroma_config(config.get("chroma"), CHROMA_DEFAULTS)
    source_chroma = resolve_chroma_config(
        (state.get("config") or {}).get("chroma"), CHROMA_PRE_20260804_DEFAULTS
    )
    chroma_mismatches = {
        key: (target_chroma[key], source_chroma[key])
        for key in CHROMA_DEFAULTS
        if target_chroma[key] != source_chroma[key]
    }
    if chroma_mismatches:
        raise RuntimeError(
            "The upstream chroma target contract differs "
            f"(configured, checkpoint): {chroma_mismatches}"
        )


    source_stage = int(
        (state.get("config") or {}).get("stage", state.get("stage", 0)) or 0
    )
    if source_stage >= 3 and int(config["stage"]) >= 3:
        source_full_config = state.get("config") or {}
        target_vocab = str(config["data"]["vocab"])
        source_vocab = str((source_full_config.get("data") or {}).get("vocab", ""))
        if source_vocab and source_vocab != target_vocab:
            raise RuntimeError(
                f"The CTC vocabulary differs from the Stage {source_stage} checkpoint; "
                "loading it would change the head weights. "
                f"Configured vocabulary: {target_vocab}; checkpoint vocabulary: "
                f"{source_vocab}. Vocabulary changes must start from a Stage 2 "
                "checkpoint and repeat head warmup."
            )
        target_heads = config.get("heads") or {}
        source_heads = source_full_config.get("heads") or {}
        mode_defaults = {
            "mel_target_mode": "frontend",
            "mel_loss_mode": "signed_l1",
            "chroma_loss_mode": "signed_l1",
        }
        mode_mismatches = {
            key: (
                target_heads.get(key, default),
                source_heads.get(key, default),
            )
            for key, default in mode_defaults.items()
            if target_heads.get(key, default) != source_heads.get(key, default)
        }
        source_weights = source_full_config.get("loss_weights") or {}
        source_reconstruction_disabled = (
            float(source_weights.get("mel", 0.0)) == 0.0
            and float(source_weights.get("chroma", 0.0)) == 0.0
        )
        if mode_mismatches and not source_reconstruction_disabled:
            raise RuntimeError(
                "The Stage 3/4 mel/chroma target-loss contract differs from the checkpoint: "
                f"{mode_mismatches}. Only a checkpoint with disabled reconstruction "
                "heads may change these modes during CTC warmup."
            )

    if int(config["stage"]) < 3:
        return


    model_state = state.get("model", {})
    mean = model_state.get("feature_extractor.feature_mean")
    std = model_state.get("feature_extractor.feature_std")
    stats_source = "model_buffers"
    if mean is None or std is None:

        mean = source_config.get("mean")
        std = source_config.get("std")
        stats_source = "checkpoint_config"
    if mean is None or std is None:
        raise RuntimeError(f"upstream checkpoint is missing feature mean/std: {checkpoint}")
    config["features"]["mean"] = _stats_to_config(mean)
    config["features"]["std"] = _stats_to_config(std)


    config.setdefault("lineage", {})["feature_config_inherited_from"] = str(
        lineage_checkpoint or checkpoint
    )


    config["lineage"]["feature_stats_value_source"] = stats_source


def _validate_stage_lineage(
    config: dict[str, Any], init_from: str | None, resume_from: str | None
) -> None:
    if init_from and resume_from:
        raise ValueError("Pass only one of init_from and resume_from")
    phase = str(config.get("phase") or "")
    if not phase:
        raise ValueError("Tokenizer training config must declare phase")
    parent_phase = config.get("parent_phase")
    source = resume_from or init_from
    if source is None:
        if parent_phase is not None:
            raise ValueError(
                f"Tokenizer phase {phase} requires --init-from from {parent_phase}"
            )
        return
    if resume_from:
        state = torch.load(
            resume_from, map_location="cpu", weights_only=False, mmap=True
        )
        source_phase = str((state.get("config") or {}).get("phase") or "")
        if source_phase != phase:
            raise ValueError(
                f"Tokenizer resume checkpoint phase mismatch: expected={phase} "
                f"actual={source_phase or '<missing>'}"
            )
        return
    if parent_phase is None:
        raise ValueError(f"Tokenizer phase {phase} must start from fresh initialization")
    if phase == "stage4":
        validate_initialization_artifact(init_from, config)
        return
    state = torch.load(init_from, map_location="cpu", weights_only=False, mmap=True)
    source_phase = str((state.get("config") or {}).get("phase") or "")
    if source_phase != str(parent_phase):
        raise ValueError(
            f"Tokenizer initialization checkpoint phase mismatch: "
            f"expected={parent_phase} actual={source_phase or '<missing>'}"
        )


def _validate_training_readiness(config: dict[str, Any]) -> None:
    experiment = config.get("experiment", {})
    if not bool(experiment.get("long_train_blocked", False)):
        return
    max_steps = int(config["train"]["max_steps"])
    allowed_steps = int(experiment.get("max_provisional_steps", 0))
    if max_steps <= allowed_steps:
        return
    reasons = (
        ", ".join(str(reason) for reason in experiment.get("blocked_reasons", []))
        or "Not recorded"
    )
    raise RuntimeError(
        f"Current configuration readiness={experiment.get('readiness', 'unknown')} and "
        f"allows at most {allowed_steps} steps; received {max_steps}; "
        f"blocking reasons: {reasons}. Complete the required quality checks and "
        "generate a new frozen configuration."
    )


def _reduce_training_metrics(
    metrics: dict[str, torch.Tensor],
) -> dict[str, float]:

    reduced = reduce_metrics(metrics)
    required = {
        "bestrq_loss_sum",
        "bestrq_correct_frames",
        "bestrq_active_frames",
    }
    if required.issubset(reduced):
        active_frames = reduced["bestrq_active_frames"]
        if active_frames > 0.0:
            global_loss = reduced["bestrq_loss_sum"] / active_frames
            reduced["bestrq_loss"] = global_loss
            reduced["bestrq_accuracy"] = (
                reduced["bestrq_correct_frames"] / active_frames
            )

            if "loss" in reduced:
                reduced["loss"] = global_loss
        else:
            reduced["bestrq_loss"] = 0.0
            reduced["bestrq_accuracy"] = 0.0
            if "loss" in reduced:
                reduced["loss"] = 0.0
    return reduced


def _validate_ready_release(config: dict[str, Any]) -> dict[str, Any]:
    data = config.get("data") or {}
    if data.get("view_format") == "public_tokenizer_view":
        descriptor_path = Path(str(data["ready_descriptor"]))
        if not descriptor_path.is_file():
            raise FileNotFoundError(f"Tokenizer view descriptor does not exist: {descriptor_path}")
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        if (
            descriptor.get("schema_version") != "oqm.public-tokenizer-view.v1"
            or descriptor.get("status") != "READY"
        ):
            raise RuntimeError("Tokenizer view descriptor is not READY or has an unsupported schema")
        artifacts = descriptor.get("artifacts") or {}
        for key in ("manifest", "vocab"):
            path = Path(str(data[key]))
            recorded = artifacts.get(path.name) or {}
            if not path.is_file() or recorded.get("sha256") != file_sha256(path):
                raise RuntimeError(f"Tokenizer view {key} is missing or does not match VIEW.json")
        return {"required": True, "status": "READY", "schema_version": descriptor["schema_version"]}
    return validate_ready_release(config)


def train(
    config: dict[str, Any],
    *,
    init_from: str | None = None,
    resume_from: str | None = None,
) -> Path:
    _validate_training_readiness(config)
    rank, local_rank, world_size, device = init_distributed(
        create_aux_cuda_group=int(config["stage"]) == 4
    )
    _validate_distributed_contract(config, world_size=world_size)
    ready_error = None
    if is_main_process():
        try:


            _validate_ready_release(config)
        except Exception as exc:
            ready_error = f"{type(exc).__name__}: {exc}"
    raise_if_rank0_error(ready_error, action="READY release validation")
    _seed_everything(int(config["train"]["seed"]), rank)
    _configure_sdp_backend(config)

    output_dir = Path(config["train"]["output_dir"])
    watchdog = _RankWatchdog(
        output_dir=output_dir,
        rank=rank,
        local_rank=local_rank,
        timeout_sec=float(config["train"].get("watchdog_timeout_sec", 0.0)),
        poll_sec=float(config["train"].get("watchdog_poll_sec", 30.0)),
    )
    watchdog.heartbeat("runtime_provenance", step=0, micro_step=0)
    code_identity = code_revision_identity()
    _validate_code_identity(config, code_identity)
    asset_error = None
    if is_main_process():
        try:
            _verify_lineage_assets(config)
        except Exception as exc:
            asset_error = f"{type(exc).__name__}: {exc}"
    raise_if_rank0_error(asset_error, action="lineage asset verification")
    distributed_runtime = _distributed_runtime_identity()
    provenance = {
        "code": code_identity,
        "assets": _asset_runtime_identity(config),
        "environment": _environment_identity(config),
        "distributed_runtime": distributed_runtime,
    }
    watchdog.heartbeat("checkpoint_staging", step=0, micro_step=0)
    cache_dir = config["train"].get(
        "local_checkpoint_cache_dir", "/tmp/open_qwen_music/checkpoints"
    )

    source_checkpoint = resume_from or init_from
    if world_size > 1:
        if init_from:
            init_from = str(stage_checkpoint_locally(init_from, cache_dir=cache_dir))
        if resume_from:
            resume_from = str(
                stage_checkpoint_locally(resume_from, cache_dir=cache_dir)
            )
        barrier()
    checkpoint = resume_from or init_from
    watchdog.heartbeat("checkpoint_contract", step=0, micro_step=0)
    _validate_stage_lineage(config, init_from, resume_from)
    _validate_subsampling_contract_from_checkpoint(config, resume_from or init_from)
    _inherit_feature_config_from_checkpoint(
        config,
        resume_from or init_from,
        lineage_checkpoint=source_checkpoint,
    )


    checkpoint_identity = (
        checkpoint_runtime_identity(checkpoint) if checkpoint is not None else None
    )
    if init_from and checkpoint_identity is not None:
        _validate_init_checkpoint_identity(config, checkpoint_identity)
    if resume_from and checkpoint_identity is not None:
        _validate_resume_config_hash(config, checkpoint_identity)
        _validate_checkpoint_provenance(
            config,
            current=provenance,
            checkpoint_identity=checkpoint_identity,
        )
    if checkpoint_identity is not None and source_checkpoint is not None:
        provenance["startup_checkpoint"] = _checkpoint_lineage_record(
            source_checkpoint,
            checkpoint_identity,
            mode="resume" if resume_from else "init",
        )
    startup_contract = {
        "config_hash": config_hash(config),
        "provenance": provenance,
        "checkpoint": checkpoint_identity,
        "distributed_state": distributed_training_state(config),
        "distributed_runtime": distributed_runtime,
        "stage": int(config["stage"]),
        "selected_node_indices": os.environ.get("OQM_SELECTED_NODE_INDICES"),
        "selected_node_ips": os.environ.get("OQM_SELECTED_NODE_IPS"),
        "master_addr": os.environ.get("MASTER_ADDR"),
        "master_port": os.environ.get("MASTER_PORT"),
        "run_id": os.environ.get("OQM_RUN_ID"),
    }
    watchdog.heartbeat("startup_consensus", step=0, micro_step=0)
    assert_distributed_consensus("tokenizer_startup", startup_contract)
    if is_main_process():
        _write_run_metadata(output_dir, config, world_size, provenance=provenance)
    barrier()

    max_steps = int(config["train"]["max_steps"])
    completed_step = (
        int(checkpoint_identity["global_step"])
        if resume_from and checkpoint_identity is not None
        else 0
    )
    if max_steps <= completed_step:
        final_checkpoint = output_dir / "last.pt"
        watchdog.heartbeat("cleanup", step=completed_step, micro_step=0)
        watchdog.close()
        cleanup_distributed()
        return final_checkpoint

    tokenizer = CharacterTokenizer.from_file(config["data"]["vocab"])
    expected_vocab = int(config["heads"]["ctc_vocab_size"])
    if len(tokenizer) > expected_vocab:
        raise ValueError(
            f"Vocabulary size {len(tokenizer)} exceeds ctc_vocab_size={expected_vocab}"
        )
    dataset = TokenizerDataset(
        config["data"]["manifest"],
        stage=int(config["stage"]),
        max_duration_sec=float(config["data"]["max_duration_sec"]),
        random_crop=bool(config["data"].get("random_crop", True)),
        crop_seed=config["data"].get("crop_seed"),
        split=config["data"].get("split", "train"),
        duration_buckets_sec=list(config["data"].get("duration_buckets_sec", [])),
        ctc_on_crop_mismatch=str(config["data"].get("ctc_on_crop_mismatch", "error")),
        normalize_target_lufs=config["data"].get("normalize_target_lufs"),
        normalize_max_boost_db=float(
            config["data"].get("normalize_max_boost_db", 12.0)
        ),
        normalize_max_peak_dbfs=config["data"].get("normalize_max_peak_dbfs"),
        ctc_group_weights={
            str(key): float(value)
            for key, value in (config["data"].get("ctc_group_weights") or {}).items()
        },
        ctc_group_default_weight=float(
            config["data"].get("ctc_group_default_weight", 1.0)
        ),
        parquet_row_group_cache_size=config["data"].get("parquet_row_group_cache_size"),
        verify_index_hashes=str(config["data"].get("verify_index_hashes", "node_once")),
    )
    batch_size_per_rank = int(config["train"]["batch_size_per_rank"])
    bucket_sampler = None
    sampler = None
    duration_buckets = list(config["data"].get("duration_buckets_sec", []))
    balanced = config["data"].get("balanced_sampler")
    if balanced:
        bucket_sampler = DistributedBalancedDurationBucketBatchSampler(
            dataset,
            batch_size_per_rank=batch_size_per_rank,
            rank=rank,
            world_size=world_size,
            seed=int(config["train"]["seed"]),
            balance_key=str(balanced.get("balance_key", "training.sampling_group")),
            weights={
                str(key): float(value)
                for key, value in balanced.get("weights", {}).items()
            },
            default_weight=float(balanced.get("default_weight", 0.0)),
            shuffle=True,
            bucket_by_duration=bool(duration_buckets),
            locality_key=balanced.get("locality_key"),
        )
    elif duration_buckets:
        bucket_sampler = DistributedDurationBucketBatchSampler(
            dataset,
            batch_size_per_rank=batch_size_per_rank,
            rank=rank,
            world_size=world_size,
            seed=int(config["train"]["seed"]),
            shuffle=True,
        )

    elif world_size > 1 or (dataset.random_crop and dataset.crop_seed is not None):
        sampler = ResumableDistributedSampler(
            dataset,
            batch_size_per_rank=batch_size_per_rank,
            seed=int(config["train"]["seed"]),
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
    collator = TokenizerCollator(
        stage=int(config["stage"]),
        feature_config=config["features"],
        chroma_config=config.get("chroma"),
        text_tokenizer=tokenizer,
        mel_target_mode=str(config.get("heads", {}).get("mel_target_mode", "frontend")),
        pad_to_num_samples=(
            round(float(config["data"]["max_duration_sec"]) * SAMPLE_RATE)
            if bool(config["data"].get("pad_to_max_duration", False))
            else None
        ),
    )
    num_workers = int(config["data"]["num_workers"])
    loader_kwargs = {
        "dataset": dataset,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collator,
        "persistent_workers": num_workers > 0,
        "multiprocessing_context": (
            str(config["data"]["worker_start_method"])
            if num_workers > 0 and config["data"].get("worker_start_method")
            else None
        ),
    }
    if bucket_sampler is not None:
        loader = DataLoader(batch_sampler=bucket_sampler, **loader_kwargs)
    else:
        loader = DataLoader(
            batch_size=batch_size_per_rank,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=False,
            **loader_kwargs,
        )

    watchdog.heartbeat("model_init", step=0, micro_step=0)
    model = MusicTokenizer(config).to(device)
    _configure_probe_trainability(model, config)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["optimizer"]["lr"]),
        betas=tuple(config["optimizer"]["betas"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    peak_lr = float(config["optimizer"]["lr"])
    scheduler = _build_scheduler(
        optimizer,
        int(config["optimizer"]["warmup_steps"]),
        max_steps,
        peak_lr=peak_lr,
        min_lr=float(config["optimizer"].get("min_lr", 0.0)),
    )


    watchdog.heartbeat("model_checkpoint_load", step=0, micro_step=0)
    if resume_from:
        load_checkpoint(
            resume_from,
            model=model,
            resume=False,
            require_stage=int(config["stage"]),
        )
    elif init_from:
        parent_stage = int(config["parent_stage"])
        load_checkpoint(
            init_from,
            model=model,
            resume=False,
            require_stage=parent_stage,
            allow_cross_stage=parent_stage < int(config["stage"]),
        )

    if world_size > 1:
        watchdog.heartbeat("ddp_init", step=0, micro_step=0)
        static_graph = bool(config["train"].get("ddp_static_graph", True))




        accumulation_steps = int(config["train"].get("gradient_accumulation_steps", 1))
        if static_graph and accumulation_steps > 1:
            raise ValueError(
                f"ddp_static_graph=True and gradient_accumulation_steps="
                f"{accumulation_steps} is incompatible: no_sync() is ineffective with static_graph, "
                f"intermediate microbatches would also all-reduce and produce incorrect accumulated gradients. "
                f"Set ddp_static_graph=false or gradient_accumulation_steps=1."
            )


        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,


            broadcast_buffers=True,
            bucket_cap_mb=float(config["train"].get("ddp_bucket_cap_mb", 25.0)),
            gradient_as_bucket_view=bool(
                config["train"].get("ddp_gradient_as_bucket_view", True)
            ),
            static_graph=static_graph,


            init_sync=not bool(resume_from),
        )


        if device.type == "cuda":
            torch.cuda.synchronize(device)
        barrier()
        watchdog.heartbeat("ddp_ready", step=0, micro_step=0)

    start_step = 0
    start_epoch = 0
    start_batches_consumed = 0
    if resume_from:
        watchdog.heartbeat("optimizer_resume", step=0, micro_step=0)
        resumed = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            resume=True,
            load_model=False,
            restore_rng=False,
        )
        _validate_resume_topology(config, resumed.distributed_state)
        assert_distributed_consensus(
            "tokenizer_resume_state",
            {
                "global_step": resumed.global_step,
                "epoch": resumed.epoch,
                "batches_consumed": resumed.batches_consumed,
                "distributed_state": resumed.distributed_state,
            },
        )
        start_step = resumed.global_step
        start_epoch = resumed.epoch
        start_batches_consumed = resumed.batches_consumed


        _seed_everything(int(config["train"]["seed"]), rank, step=start_step)
        if device.type == "cuda":


            torch.cuda.synchronize(device)
        barrier()
        watchdog.heartbeat(
            "optimizer_ready",
            step=start_step,
            micro_step=start_batches_consumed,
        )

    precision = str(config["train"].get("precision", "bf16"))
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision == "fp16"
    )
    accumulation = int(config["train"].get("gradient_accumulation_steps", 1))
    clip_norm = float(config["train"].get("gradient_clip_norm", 1.0))
    log_every = int(config["train"].get("log_every_steps", 10))
    save_every = int(config["train"].get("save_every_steps", 1000))
    retained_steps = {
        int(value) for value in config["train"].get("retain_checkpoint_steps", [])
    }
    invalid_retained = [
        value for value in retained_steps if value <= 0 or value > max_steps
    ]
    if invalid_retained:
        raise ValueError(
            f"retain_checkpoint_steps must be in [1, {max_steps}]: {invalid_retained}"
        )
    stop_after_checkpoint_step = int(
        os.environ.get("OQM_STOP_AFTER_CHECKPOINT_STEP", "0")
    )
    if stop_after_checkpoint_step < 0 or stop_after_checkpoint_step > max_steps:
        raise ValueError(
            f"OQM_STOP_AFTER_CHECKPOINT_STEP must be in [0, {max_steps}]; "
            f"received {stop_after_checkpoint_step}"
        )
    model.train()
    _configure_probe_module_modes(model, config)
    optimizer.zero_grad(set_to_none=True)
    step = start_step


    epoch = start_epoch
    pending_skip_batches = start_batches_consumed


    micro_step = 0
    nonfinite_grad_steps = 0
    max_nonfinite_grad_steps = int(config["train"].get("max_nonfinite_grad_steps", 20))
    started = time.time()
    processed_audio_seconds = 0.0
    pending_audio_seconds = 0.0
    dead_code_replacements_segment_total = 0
    final_checkpoint = output_dir / "last.pt"
    stop_requested = False
    while step < max_steps and not stop_requested:
        active_sampler = bucket_sampler if bucket_sampler is not None else sampler
        if active_sampler is not None:
            active_sampler.set_epoch(epoch)


        skip_batches = pending_skip_batches
        pending_skip_batches = 0
        if skip_batches and hasattr(active_sampler, "set_skip_batches"):
            active_sampler.set_skip_batches(skip_batches)
        elif hasattr(active_sampler, "set_skip_batches"):
            active_sampler.set_skip_batches(0)
        loader_iterator = iter(loader)
        micro_step = skip_batches
        while True:
            watchdog.heartbeat(
                "dataloader_next",
                step=step,
                micro_step=micro_step,
            )
            try:
                batch = next(loader_iterator)
            except StopIteration:
                break
            sample_ids = list(batch.get("sample_ids", []))
            watchdog.heartbeat(
                "move_to_device",
                step=step,
                micro_step=micro_step,
                sample_ids=sample_ids,
            )
            batch = _move_to_device(batch, device)
            pending_audio_seconds += (
                float(batch["waveform_num_samples"].sum().item()) / SAMPLE_RATE
            )
            _configure_stage4_quantizer_step(model, config, step)
            gate_alpha = _gate_alpha(config, step)
            loss_weights = _loss_weights(config, step)
            sync_step = (micro_step + 1) % accumulation == 0
            context = (
                model.no_sync()
                if isinstance(model, DistributedDataParallel) and not sync_step
                else torch.autograd.profiler.record_function("ddp_sync")
            )
            with context:
                watchdog.heartbeat(
                    "forward",
                    step=step,
                    micro_step=micro_step,
                    sample_ids=sample_ids,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=use_autocast,
                ):
                    loss, metrics = model(
                        batch,
                        gate_alpha=gate_alpha,
                        loss_weights=loss_weights,
                    )
                    scaled_loss = loss / accumulation
                watchdog.heartbeat(
                    "backward",
                    step=step,
                    micro_step=micro_step,
                    sample_ids=sample_ids,
                )
                scaler.scale(scaled_loss).backward()
            if not sync_step:
                micro_step += 1
                continue

            watchdog.heartbeat(
                "optimizer",
                step=step,
                micro_step=micro_step,
                sample_ids=sample_ids,
            )
            scaler.unscale_(optimizer)
            _clear_frozen_stage4_gradients(model, config, step)
            should_log_step = (step + 1) % log_every == 0 or step == 0
            if int(config["stage"]) >= 3 and should_log_step:
                metrics.update(_head_gradient_metrics(model))
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            if should_log_step:
                metrics["grad_norm_total"] = grad_norm.detach()


            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
            else:
                nonfinite_grad_steps += 1
                if is_main_process():
                    print(
                        f"nonfinite_grad step={step + 1} "
                        f"count={nonfinite_grad_steps}/{max_nonfinite_grad_steps} "
                        f"grad_norm={grad_norm} skipped=1",
                        flush=True,
                    )
                if nonfinite_grad_steps > max_nonfinite_grad_steps:
                    raise RuntimeError(
                        f"Non-finite gradients occurred for {nonfinite_grad_steps} steps, "
                        f"exceeds max_nonfinite_grad_steps={max_nonfinite_grad_steps}."
                        " No effective updates were observed; check the data and learning rate."
                    )


            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            scheduler.step()
            if int(config["stage"]) == 4:
                root = model.module if hasattr(model, "module") else model
                replacement_count = int(root.quantizer.synchronize_ema())
                dead_code_replacements_segment_total += replacement_count
                distributed_world = (
                    torch.distributed.get_world_size()
                    if torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                    else 1
                )


                rank0_scale = distributed_world if is_main_process() else 0
                metrics["dead_code_replacements"] = loss.detach().new_tensor(
                    replacement_count * rank0_scale
                )
                metrics["dead_code_replacements_segment_total"] = (
                    loss.detach().new_tensor(
                        dead_code_replacements_segment_total * rank0_scale
                    )
                )
            step += 1
            processed_audio_seconds += pending_audio_seconds
            pending_audio_seconds = 0.0

            if step % log_every == 0 or step == 1:
                watchdog.heartbeat(
                    "metrics_all_reduce",
                    step=step,
                    micro_step=micro_step,
                    sample_ids=sample_ids,
                )
                reduced = _reduce_training_metrics(metrics)
                global_audio_seconds = reduce_scalar_sum(processed_audio_seconds)
                if is_main_process():
                    elapsed = max(time.time() - started, 1e-6)
                    run_steps = step - start_step
                    fields = " ".join(
                        f"{key}={value:.5g}" for key, value in sorted(reduced.items())
                    )
                    print(
                        f"stage={config['stage']} step={step}/{max_steps} "
                        f"lr={scheduler.get_last_lr()[0]:.3e} "
                        f"steps_per_sec={run_steps / elapsed:.3f} "
                        f"audio_sec_per_sec={global_audio_seconds / elapsed:.2f} "
                        f"{fields}",
                        flush=True,
                    )
            if (
                step % save_every == 0
                or step >= max_steps
                or step in retained_steps
                or step == stop_after_checkpoint_step
            ):
                watchdog.heartbeat(
                    "checkpoint_save",
                    step=step,
                    micro_step=micro_step,
                    sample_ids=sample_ids,
                )
                save_error = None
                if is_main_process():
                    try:
                        save_checkpoint(
                            final_checkpoint,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            config=config,
                            global_step=step,
                            epoch=epoch,
                            batches_consumed=micro_step + 1,
                            provenance=provenance,
                        )
                        if step in retained_steps:
                            retain_checkpoint(
                                final_checkpoint,
                                output_dir / f"step_{step:06d}.pt",
                            )
                    except Exception as exc:
                        save_error = f"{type(exc).__name__}: {exc}"
                raise_if_rank0_error(save_error, action=f"checkpoint save step={step}")
                watchdog.heartbeat(
                    "checkpoint_barrier",
                    step=step,
                    micro_step=micro_step,
                    sample_ids=sample_ids,
                )
                barrier()
                if (
                    stop_after_checkpoint_step > 0
                    and step >= stop_after_checkpoint_step
                ):
                    if is_main_process():
                        print(
                            "graceful_stop_after_checkpoint "
                            f"step={step} requested={stop_after_checkpoint_step}",
                            flush=True,
                        )
                    stop_requested = True
            if step >= max_steps:
                break
            if stop_requested:
                break
            watchdog.heartbeat(
                "step_complete",
                step=step,
                micro_step=micro_step,
                sample_ids=sample_ids,
            )
            micro_step += 1
        if stop_requested:
            break
        epoch += 1

    watchdog.heartbeat("cleanup", step=step, micro_step=micro_step)
    watchdog.close()
    cleanup_distributed()
    return final_checkpoint
