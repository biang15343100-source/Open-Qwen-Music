
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .common import (
    all_gather_objects,
    barrier,
    build_provenance,
    checkpoint_path,
    cleanup_distributed,
    config_hash,
    init_distributed,
    is_main_process,
    is_sharded_checkpoint,
    load_checkpoint,
    log_kv,
    read_sidecar,
    reduce_metrics,
    reduce_scalar_sum,
    reduce_weighted_metrics,
    require_equal_across_ranks,
    resolve_base_model_revision,
    resolve_text_tokenizer_revision,
    save_checkpoint,
    seed_everything,
    stage_checkpoint_locally,
    write_run_metadata,
)
from .common.distributed import auxiliary_cuda_group
from .common.logging import Stopwatch
from .common.parallel import (
    clip_grad_norm,
    gradient_sync,
    resolve_strategy,
    scale_gradients,
    validate_precision,
    wrap_model,
)
from .condition import ConditionRenderConfig
from .contracts import SEMANTIC_FRAME_RATE, SEMANTIC_FRAMES_PER_MELODY_FRAME
from .data import (
    LlmCollator,
    LlmSampleDataset,
    StepCounter,
    build_sampler,
)
from .eval.curves import summarize_eval_metrics
from .eval.direct import (
    duration_aligned_lyrics,
    finalize_calibration_metrics,
    generation_length_metrics,
    load_semantic_frequency_counts,
    long_range_forgetting_metrics,
    per_sample_distribution_metrics,
    section_sequence_metrics,
    semantic_distribution_metrics,
    semantic_frequency_buckets,
    token_distribution_distances,
)
from .eval.stage4_aggregate import paired_bootstrap_mean_ci
from .eval.stage4_probe import (
    Stage4AudioTargetStore,
    Stage4SemanticProbe,
    score_stage4_pair_batch,
)
from .grammar import GrammarConfig
from .model import (
    MUSIC_EMBEDDING_NAMESPACES,
    MusicLLM,
    PartitionedVocabularyEmbedding,
    build_model,
    vocabulary_weight_range,
)
from .registry import TokenRegistry, build_registry_from_config
from .sequence import IGNORE_LABEL, SequenceConfig
from .training_contracts import (
    initial_data_step,
    validate_production_tokenizer_identity,
)

_STAGE_PARENT: dict[str, str] = {
    "stage2": "stage1",
    "stage3": "stage2",
}
_TRAINING_STAGES = frozenset({"stage1", "stage2", "stage3"})


def checkpoint_identity(path: str | Path) -> dict[str, Any]:
    sidecar = read_sidecar(path)
    provenance = sidecar.get("provenance") or {}
    return {
        "path": str(Path(path).resolve()),
        "stage": sidecar.get("stage"),
        "global_step": sidecar.get("global_step"),
        "config_hash": sidecar.get("config_hash"),
        "registry_revision": provenance.get("registry_revision"),
        "semantic_tokenizer_revision": (
            provenance.get("semantic_tokenizer") or {}
        ).get("revision"),
        "parent_checkpoint": provenance.get("parent_checkpoint"),
    }


def validate_stage_entry(
    config: dict[str, Any],
    *,
    init_from: str | None,
    resume_from: str | None,
    check_output: bool = True,
) -> dict[str, Any] | None:
    if init_from and resume_from:
        raise ValueError("--init-from and --resume-from are mutually exclusive")
    stage = str(config.get("stage") or "")
    explicit_parent = config.get("stage_parent")
    parent = (
        str(explicit_parent)
        if explicit_parent not in (None, "")
        else _STAGE_PARENT.get(stage)
    )
    train_section = dict(config.get("train", {}) or {})
    production = bool(
        (config.get("data", {}) or {}).get("require_production_contract", False)
    )
    if (
        production
        and stage not in {"stage1", "stage2", "stage3"}
        and parent is None
        and not bool(train_section.get("allow_root_stage", False))
    ):
        raise ValueError(
            f"Custom training stage {stage!r} must declare stage_parent, or set "
            "train.allow_root_stage=true for an intentional fresh start"
        )
    upstream = resume_from or init_from
    identity: dict[str, Any] | None = None

    if parent and not upstream:
        raise ValueError(
            f"{stage} must provide --init-from <{parent} checkpoint> or "
            "--resume-from <same-stage checkpoint>; silent base-model restart is disabled"
        )
    if upstream:
        identity = checkpoint_identity(upstream)
        found_stage = str(identity.get("stage") or "")
        expected = stage if resume_from else parent
        if expected and found_stage != expected:
            mode = "resume" if resume_from else "init"
            raise ValueError(
                f"{stage} of {mode} checkpoint stage must be {expected!r},"
                f"received {found_stage!r}:{upstream}"
            )
        if resume_from:
            current_hash = config_hash(config)
            found_hash = str(identity.get("config_hash") or "")
            allow_resume_drift = bool(
                (config.get("train", {}) or {}).get("allow_resume_config_mismatch", False)
            )
            if found_hash != current_hash and not allow_resume_drift:
                raise ValueError(
                    "Resume config does not match the checkpoint: "
                    f"current={current_hash} checkpoint={found_hash}. "
                    "Use the original config, or set train.allow_resume_config_mismatch=true "
                    "to accept an inexact recovery."
                )

    if check_output and not resume_from:
        output = Path(str(train_section.get("output_dir", "runs/llm/default")))
        allow_existing = bool(train_section.get("allow_existing_output", False))
        if output.exists() and any(output.iterdir()) and not allow_existing:
            raise FileExistsError(
                f"Output directory is not empty: {output}. Use a new output_dir, or set "
                "train.allow_existing_output=true to accept possible overwrites."
            )
    return identity


def validate_training_preconditions(config: dict[str, Any]) -> None:
    stage = str(config.get("stage") or "")
    data = dict(config.get("data", {}) or {})


    if stage not in _TRAINING_STAGES and not bool(
        data.get("require_production_contract", False)
    ):
        return
    problems: list[str] = []
    generation = dict(config.get("generation", {}) or {})
    sampler = dict(config.get("sampler", {}) or {})
    sequence = dict(config.get("sequence", {}) or {})
    evaluation = dict(config.get("evaluation", {}) or {})
    revision = resolve_base_model_revision(config)
    if revision == "unknown":
        problems.append("model.base_model_revision is not pinned")
    if resolve_text_tokenizer_revision(config) == "unknown":
        problems.append("model text tokenizer revision is not pinned")
    if not bool(data.get("require_production_contract", False)):
        problems.append("data.require_production_contract must be true")
    if int(data.get("num_workers", 0)) != 0:
        problems.append(
            "data.num_workers must be 0 so checkpoints capture the exact sampler cursor; "
            "prefetched batches would otherwise be skipped after resume"
        )
    if not bool(sampler.get("strict_paper", False)):
        problems.append("sampler.strict_paper must be true")
    if bool(sequence.get("random_crop", False)):
        problems.append("sequence.random_crop must be false; use a windowed manifest")
    if str(sequence.get("overflow_policy", "")) != "error":
        problems.append("sequence.overflow_policy must be error to prevent silent truncation")
    semantic_frames = int(sequence.get("max_semantic_frames", 0))
    melody_frames = int(generation.get("max_melody_tokens", 0))
    required_melody = math.ceil(
        semantic_frames / float(SEMANTIC_FRAMES_PER_MELODY_FRAME)
    )
    if melody_frames < required_melody:
        problems.append(
            f"generation.max_melody_tokens={melody_frames} is below the {required_melody} "
            "frames represented by the training window"
        )
    melody_segments = int(generation.get("max_melody_segments", 0))
    required_new_tokens = semantic_frames + 3 + melody_frames + 2 * melody_segments + 2
    max_new_tokens = int(generation.get("max_new_tokens", 0))
    if max_new_tokens < required_new_tokens:
        problems.append(
            f"generation.max_new_tokens={max_new_tokens} is below the required "
            f"{required_new_tokens} tokens"
        )
    max_sequence_length = int(sequence.get("max_sequence_length", 0))
    max_condition_tokens = int(sequence.get("max_condition_tokens", 0))
    if 5 + max_condition_tokens + required_new_tokens > max_sequence_length:
        problems.append(
            "The maximum condition and completion exceed sequence.max_sequence_length: "
            f"5 + {max_condition_tokens} + {required_new_tokens} > {max_sequence_length}"
        )
    eval_every = int(evaluation.get("every_steps", 0))
    full_every = int(evaluation.get("full_every_steps", 0))
    free_generation = dict(evaluation.get("generation", {}) or {})
    generation_every = int(free_generation.get("every_steps", 0))
    if full_every > 0 and (eval_every <= 0 or full_every % eval_every != 0):
        problems.append(
            "evaluation.full_every_steps must be a positive multiple of evaluation.every_steps"
        )
    if generation_every > 0 and (
        eval_every <= 0 or generation_every % eval_every != 0
    ):
        problems.append(
            "evaluation.generation.every_steps must be a positive multiple of evaluation.every_steps"
        )
    if problems:
        raise RuntimeError(
            f"{stage} training prerequisites are not satisfied:\n  - "
            + "\n  - ".join(problems)
        )


def build_text_encoder(config: dict[str, Any]):
    from transformers import AutoTokenizer

    section = dict(config.get("model", {}) or {})
    path = section.get("tokenizer_path") or section.get("base_model_path")
    if not path:
        raise KeyError("requires model.tokenizer_path or model.base_model_path to load text tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    setattr(
        tokenizer,
        "_oqm_artifact_revision",
        resolve_text_tokenizer_revision(config),
    )
    return tokenizer


def validate_text_encoder_namespace(
    text_encoder: Any,
    registry: TokenRegistry,
) -> None:

    size = len(text_encoder) if hasattr(text_encoder, "__len__") else None
    if size is not None and int(size) > registry.control_base:
        raise RuntimeError(
            f"Text tokenizer size {size} exceeds the text namespace limit "
            f"{registry.control_base}; added tokens would overlap control or semantic IDs"
        )
    get_vocab = getattr(text_encoder, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        max_id = max((int(value) for value in vocab.values()), default=-1)
        if max_id >= registry.control_base:
            raise RuntimeError(
                f"Text tokenizer maximum ID {max_id} enters the music/control namespace "
                f"[{registry.control_base}, ...)"
            )


def resolve_registry(config: dict[str, Any]) -> TokenRegistry:
    from transformers import AutoConfig

    section = dict(config.get("model", {}) or {})
    explicit = (config.get("registry", {}) or {}).get("text_vocab_size")
    if explicit is not None:
        return build_registry_from_config(config, int(explicit))
    path = section.get("base_model_path") or section.get("tokenizer_path")
    if not path:
        raise KeyError(
            "Cannot determine text_vocab_size; set registry.text_vocab_size or model.base_model_path"
        )
    base_config = AutoConfig.from_pretrained(str(path))
    return build_registry_from_config(config, int(base_config.vocab_size))


def _embedding_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    inner = model.module if hasattr(model, "module") else model
    grouped = getattr(inner, "embedding_parameters_by_role", None)
    if callable(grouped):
        return [
            parameter
            for parameters in grouped().values()
            for parameter in parameters
        ]
    found: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in (
        inner.backbone.get_input_embeddings(),
        inner.backbone.get_output_embeddings(),
    ):
        if module is None:
            continue
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                found.append(parameter)
    return found


def build_optimizer(
    model: torch.nn.Module, config: dict[str, Any]
) -> torch.optim.Optimizer:
    section = dict(config.get("optimizer", {}) or {})
    weight_decay = float(section.get("weight_decay", 0.01))
    embedding_lr_scale = float(section.get("embedding_lr_scale", 1.0))
    raw_text_scale = section.get("text_embedding_lr_scale")
    raw_new_scale = section.get("new_embedding_lr_scale")
    raw_type_scale = section.get("type_embedding_lr_scale")
    text_embedding_lr_scale = (
        embedding_lr_scale
        if raw_text_scale is None
        else float(raw_text_scale)
    )
    new_embedding_lr_scale = (
        embedding_lr_scale
        if raw_new_scale is None
        else float(raw_new_scale)
    )
    type_embedding_lr_scale = (
        new_embedding_lr_scale
        if raw_type_scale is None
        else float(raw_type_scale)
    )
    base_lr = float(section.get("lr", 2e-4))
    scales = {
        "embedding_lr_scale": embedding_lr_scale,
        "text_embedding_lr_scale": text_embedding_lr_scale,
        "new_embedding_lr_scale": new_embedding_lr_scale,
        "type_embedding_lr_scale": type_embedding_lr_scale,
    }
    invalid = {name: value for name, value in scales.items() if value <= 0.0}
    if invalid:
        raise ValueError(
            f"Embedding learning-rate scales must be positive; got {invalid}. "
            "Freeze embeddings with requires_grad instead of setting a zero scale."
        )

    inner = model.module if hasattr(model, "module") else model
    grouped_getter = getattr(inner, "embedding_parameters_by_role", None)
    embedding_roles = (
        grouped_getter()
        if callable(grouped_getter)
        else {"all": _embedding_parameters(model)}
    )
    if "all" in embedding_roles and (
        text_embedding_lr_scale != new_embedding_lr_scale
    ):
        raise ValueError(
            "Different learning rates for base and new embedding rows require "
            "model.partitioned_embeddings=true. Scaling gradients within one parameter "
            "is not equivalent under Adam because its moments cancel that scaling."
        )
    embedding_params = [
        parameter
        for parameters in embedding_roles.values()
        for parameter in parameters
    ]
    embedding_ids = {id(parameter) for parameter in embedding_params}
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in embedding_ids:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    groups: list[dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay, "lr": base_lr})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0, "lr": base_lr})
    role_scales = {
        "all": (
            text_embedding_lr_scale
            if text_embedding_lr_scale == new_embedding_lr_scale
            else embedding_lr_scale
        ),
        "text": text_embedding_lr_scale,
        "new": new_embedding_lr_scale,
        "type": type_embedding_lr_scale,
    }
    for role, parameters in embedding_roles.items():
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        if not trainable:
            continue
        if role not in role_scales:
            raise KeyError(f"Unknown embedding parameter role={role!r}")
        groups.append(
            {
                "params": trainable,
                "weight_decay": 0.0,
                "lr": base_lr * role_scales[role],
                "embedding_role": role,
            }
        )
    return torch.optim.AdamW(
        groups,
        lr=base_lr,
        betas=tuple(section.get("betas", (0.9, 0.95))),
        eps=float(section.get("eps", 1e-8)),
    )


def lr_multiplier_fn(config: dict[str, Any]) -> Callable[[int], float]:
    section = dict(config.get("optimizer", {}) or {})
    train_section = dict(config.get("train", {}) or {})
    warmup = int(section.get("warmup_steps", 1000))
    max_steps = int(train_section.get("max_steps", 100000))
    decay = str(section.get("decay", "cosine"))
    min_ratio = float(section.get("min_lr_ratio", 0.1))
    if decay not in ("cosine", "linear", "constant"):
        raise ValueError(
            f"optimizer.decay must be cosine, linear, or constant; got {decay}"
        )
    if warmup >= max_steps and decay != "constant":
        raise ValueError(
            f"optimizer.warmup_steps={warmup} is at least train.max_steps={max_steps}; "
            "the schedule would never leave warmup"
        )

    def multiplier(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        if decay == "constant":
            return 1.0
        progress = min(
            1.0, max(0.0, (step - warmup) / max(1.0, float(max_steps - warmup)))
        )
        if decay == "linear":
            factor = 1.0 - progress
        else:
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * factor

    return multiplier


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict[str, Any]
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier_fn(config))


def lr_endpoints(config: dict[str, Any]) -> tuple[float, float]:
    base_lr = float((config.get("optimizer", {}) or {}).get("lr", 2e-4))
    max_steps = int((config.get("train", {}) or {}).get("max_steps", 100000))
    multiplier = lr_multiplier_fn(config)
    return base_lr, base_lr * multiplier(max_steps - 1)


VOCAL_TIME_SHARE = 0.62
TYPICAL_SECTION_COUNT = 9
CONTROL_TOKEN_OVERHEAD = 10  # bos/task/mode/cond_bos/cond_eos/melody_bos/eos/music_bos/eos/eos


def sequence_budget(config: dict[str, Any]) -> dict[str, float]:
    sequence = dict(config.get("sequence", {}) or {})
    sampler = dict(config.get("sampler", {}) or {})
    train = dict(config.get("train", {}) or {})

    semantic = int(sequence.get("max_semantic_frames", 2250))
    condition = int(sequence.get("max_condition_tokens", 1024))
    max_length = int(sequence.get("max_sequence_length", 4096))
    duration = semantic / 25.0
    melody_section = duration * 6.25 * VOCAL_TIME_SHARE + 2 * TYPICAL_SECTION_COUNT
    melody_whole_song = duration * 6.25 + 16
    longest = condition + melody_section + semantic + CONTROL_TOKEN_OVERHEAD


    condition_estimate = int(sampler.get("condition_token_estimate", condition))
    estimated = min(
        semantic + melody_whole_song + condition_estimate + 8, float(max_length)
    )
    budget = int(train.get("max_tokens_per_gpu", 8192))
    batch_at_longest = max(
        1, min(int(train.get("max_batch_size", 64)), int(budget // estimated))
    )

    worst_batch = max(1, int(train.get("max_batch_size", 64)))
    gap = max(0, condition - condition_estimate)
    worst_tokens = float(min(budget + worst_batch * gap, worst_batch * max_length))
    return {
        "duration_sec": duration,
        "semantic": float(semantic),
        "melody_section": melody_section,
        "condition": float(condition),
        "longest_sample": longest,
        "max_sequence_length": float(max_length),
        "estimated_longest": estimated,
        "batch_at_budget": float(batch_at_longest),
        "condition_underestimate": float(gap),
        "worst_case_batch_size": float(worst_batch),
        "worst_case_batch_tokens": worst_tokens,
        "max_tokens_per_gpu": float(budget),
        "overshoot_ratio": worst_tokens / float(budget),

        "forward_ratio_after_split": 1.0,
    }


def _infinite(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def split_batch_to_token_budget(
    batch: dict[str, Any], *, budget: int, pad_to_multiple_of: int = 64
) -> list[dict[str, Any]]:
    input_ids = batch["input_ids"]
    width = int(input_ids.shape[-1])
    size = int(input_ids.shape[0])
    if size * width <= budget or size == 1:

        return [batch]

    multiple = max(1, int(pad_to_multiple_of))
    real_lengths = batch["attention_mask"].sum(dim=-1).tolist()
    order = sorted(range(size), key=lambda row: -int(real_lengths[row]))

    groups: list[list[int]] = []
    group_width = 0
    for row in order:
        padded = min(width, ((int(real_lengths[row]) + multiple - 1) // multiple) * multiple)
        if not groups or (len(groups[-1]) + 1) * group_width > budget:
            groups.append([row])
            group_width = padded
        else:
            groups[-1].append(row)
    return [_slice_batch(batch, rows, multiple) for rows in groups]


def merge_part_metrics(
    part_metrics: dict[str, list[tuple[float, float]]],
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for name, samples in part_metrics.items():
        if name.startswith("tokens_") or name == "supervised_tokens":
            merged[name] = sum(value for value, _ in samples)
            continue
        weight = sum(share for _, share in samples)
        if weight <= 0.0:
            merged[name] = sum(value for value, _ in samples) / max(len(samples), 1)
        else:
            merged[name] = sum(value * share for value, share in samples) / weight
    return merged


def metric_token_weight(
    name: str,
    values: dict[str, float],
    *,
    num_loss_tokens: int,
    accuracy_limit: int,
) -> float | None:
    if name.startswith("tokens_") or name == "supervised_tokens":
        return None
    if name == "loss_weighted":
        return float(num_loss_tokens)
    if name == "loss":
        return float(values.get("supervised_tokens", 0.0))
    suffix: str | None = None
    mean_prefixes = (
        "normalized_entropy_",
        "confidence_",
        "entropy_",
        "brier_",
        "mrr_",
        "loss_",
        "acc_",
    )
    for prefix in mean_prefixes:
        if name.startswith(prefix):
            suffix = name.removeprefix(prefix)
            break
    if suffix is None:
        return None
    count = float(values.get(f"tokens_{suffix}", 0.0))
    if name.startswith("acc_"):
        count = min(count, float(max(1, accuracy_limit)))
    return count


def _slice_batch(batch: dict[str, Any], rows: list[int], multiple: int) -> dict[str, Any]:
    index = torch.tensor(sorted(rows), dtype=torch.long)
    attention_mask = batch["attention_mask"].index_select(0, index)
    width = int(batch["input_ids"].shape[-1])
    longest = int(attention_mask.sum(dim=-1).max())
    keep = min(width, ((longest + multiple - 1) // multiple) * multiple)
    labels = batch["labels"].index_select(0, index)[:, :keep]
    part = {
        "input_ids": batch["input_ids"].index_select(0, index)[:, :keep],
        "labels": labels,
        "attention_mask": attention_mask[:, :keep],
        "num_semantic_tokens": batch["num_semantic_tokens"].index_select(0, index),
        "real_tokens": int(attention_mask.sum()),
        "padded_tokens": len(rows) * keep,

        "supervised_tokens": int((labels != IGNORE_LABEL).sum()),
    }
    if "constraint_kinds" in batch:
        part["constraint_kinds"] = batch["constraint_kinds"].index_select(
            0, index
        )[:, :keep]
    return part


def _reduce_max(value: float, device: torch.device) -> float:
    group = auxiliary_cuda_group()
    if not dist.is_initialized() or group is None:
        return value
    tensor = torch.tensor([float(value)], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return float(tensor.item())


def _log_sampling_diagnostics(sampler: Any, step: int) -> None:
    if not is_main_process() or not hasattr(sampler, "sampling_diagnostics"):
        return


    diagnostics = sampler.sampling_diagnostics()
    log_kv("sampling", diagnostics.as_flat_dict())
    for warning in diagnostics.warnings:
        print(f"[sampler][warn] step={step} {warning}", flush=True)


class NonFiniteStepGuard:

    def __init__(self, *, limit: int, checkpoint_hint: str = "") -> None:
        self.limit = int(limit)
        self.checkpoint_hint = checkpoint_hint
        self.consecutive = 0
        self.skipped = 0

    def observe(self, grad_norm: float) -> bool:
        if math.isfinite(grad_norm):
            self.consecutive = 0
            return True
        self.skipped += 1
        self.consecutive += 1
        if self.limit > 0 and self.consecutive >= self.limit:
            raise RuntimeError(
                f"Gradient norm was non-finite for {self.consecutive} consecutive steps "
                f"(grad_norm={grad_norm}), reaching train.max_consecutive_nonfinite_steps="
                f"{self.limit}. Check the data and learning rate before continuing."
                + self.checkpoint_hint
            )
        return False

    def state_dict(self) -> dict[str, int]:
        return {
            "limit": self.limit,
            "consecutive": self.consecutive,
            "skipped": self.skipped,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("limit", self.limit)) != self.limit:
            raise RuntimeError("NonFiniteStepGuard limit does not match the checkpoint")
        self.consecutive = int(state.get("consecutive", 0))
        self.skipped = int(state.get("skipped", 0))


def _skip_replayed_batches(
    sampler: Any,
    *,
    resume_extra: dict[str, Any] | None,
    step: int,
    epoch: int,
    steps_per_epoch: int,
    accumulation: int,
) -> int:
    if steps_per_epoch <= 1:
        return 0
    recorded = (resume_extra or {}).get("epoch_batches")
    consumed_steps = step - epoch * steps_per_epoch
    if recorded is None:
        if consumed_steps > 0:
            log_kv(
                "resume_replay_warning",
                {
                    "step": step,
                    "epoch": epoch,
                    "replayed_steps": consumed_steps,
                    "reason": "checkpoint does not contain epoch_batches",
                },
            )
        return consumed_steps * accumulation
    to_skip = int(recorded)
    if to_skip <= 0:
        return 0
    started = time.monotonic()
    iterator = iter(sampler)
    for _ in range(to_skip):
        next(iterator)
    log_kv(
        "resume_skip",
        {
            "step": step,
            "epoch": epoch,
            "skipped_batches": to_skip,
            "seconds": time.monotonic() - started,
        },
    )
    return to_skip


def metric_improved(
    value: float,
    best: float | None,
    *,
    mode: str,
    min_delta: float,
) -> bool:
    if mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'")
    if min_delta < 0:
        raise ValueError("min_delta cannot be negative")
    if best is None:
        return True
    return (
        float(value) < float(best) - min_delta
        if mode == "min"
        else float(value) > float(best) + min_delta
    )


def require_consistent_selection_metric(
    values: list[float | None],
    *,
    metric: str,
) -> float:
    if not values or any(value is None for value in values):
        raise RuntimeError(
            f"evaluation.selection.metric={metric!r} is missing on a rank"
        )
    numbers = [float(value) for value in values if value is not None]
    if any(not math.isfinite(value) for value in numbers):
        raise RuntimeError(f"evaluation.selection.metric={metric!r} contains a non-finite value")
    if max(numbers) - min(numbers) > 1e-9 * max(1.0, max(map(abs, numbers))):
        raise RuntimeError(
            f"evaluation.selection.metric={metric!r} differs across ranks: {numbers}"
        )
    return numbers[0]


def _warn_batch_budget_risk(config: dict[str, Any]) -> None:
    sequence = dict(config.get("sequence", {}) or {})
    sampler = dict(config.get("sampler", {}) or {})
    ceiling = int(sequence.get("max_condition_tokens", 1024))


    estimate = int(sampler.get("condition_token_estimate", ceiling))
    if estimate >= ceiling:
        return
    weights = dict((config.get("model", {}) or {}).get("region_loss_weights", {}) or {})
    zero_weight = [name for name, value in weights.items() if float(value) == 0.0]
    log_kv(
        "batch_budget_warning",
        {
            "condition_token_estimate": estimate,
            "max_condition_tokens": ceiling,
            "per_sample_underestimate": ceiling - estimate,
            "max_batch_size": (config.get("train", {}) or {}).get("max_batch_size"),
            "mitigation": "trainer_splits_batches",
            "requested_fix": f"sampler.condition_token_estimate>={ceiling}",
            "split_normalization_exact": not zero_weight,
        },
    )
    print(
        "[budget][warn] sampler.condition_token_estimate="
        f"{estimate} < sequence.max_condition_tokens={ceiling}: "
        f"the sampler underestimates each sample by up to {ceiling - estimate} tokens; "
        "the trainer will split batches into sub-batches. Set "
        f"sampler.condition_token_estimate to at least {ceiling} to avoid the split.",
        flush=True,
    )
    if zero_weight:
        print(
            f"[budget][warn] region_loss_weights entries {zero_weight} are zero; batch "
            "splitting may slightly change normalization. Use small positive weights or "
            "increase max_tokens_per_gpu to avoid splitting.",
            flush=True,
        )


class _ProgressWatchdog:

    def __init__(self, *, rank: int, warn_seconds: float, poll_seconds: float = 5.0) -> None:
        self.rank = int(rank)
        self.warn_seconds = float(warn_seconds)
        self.poll_seconds = float(poll_seconds)
        self._step = 0
        self._phase = "startup"
        self._at = time.monotonic()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def beat(self, step: int, phase: str = "train") -> None:
        with self._lock:
            self._step = int(step)
            self._phase = phase
            self._at = time.monotonic()

    def start(self) -> None:
        if self.warn_seconds <= 0:
            return
        self._thread = threading.Thread(target=self._run, name="oqm-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_seconds * 2)

    def _run(self) -> None:
        last_beat = 0.0
        warned_at = 0.0
        while not self._stop.wait(self.poll_seconds):
            with self._lock:
                step, phase, at = self._step, self._phase, self._at
            if at != last_beat:
                last_beat, warned_at = at, 0.0
            stalled = time.monotonic() - at
            if stalled < self.warn_seconds or stalled - warned_at < self.warn_seconds:
                continue
            warned_at = stalled
            print(
                f"train_stall rank={self.rank} step={step} phase={phase} "
                f"stalled_seconds={stalled:.1f} threshold={self.warn_seconds:.0f}",
                flush=True,
            )


def validate_training_topology(config: Mapping[str, Any], *, world_size: int) -> None:
    train = config.get("train") or {}
    expected_world_size = int(train.get("expected_world_size", world_size))
    if world_size != expected_world_size:
        raise RuntimeError(
            "LLM world size does not match the training recipe: "
            f"expected={expected_world_size} actual={world_size}"
        )
    effective_batch_size = (
        world_size
        * int(train.get("max_batch_size", 1))
        * int(train.get("gradient_accumulation_steps", 1))
    )
    expected_batch_size = int(
        train.get("expected_global_batch_size", effective_batch_size)
    )
    if effective_batch_size != expected_batch_size:
        raise RuntimeError(
            "LLM global batch size does not match the training recipe: "
            f"expected={expected_batch_size} actual={effective_batch_size}"
        )


def train(
    config: dict[str, Any],
    *,
    init_from: str | None = None,
    resume_from: str | None = None,
) -> Path:
    rank, local_rank, world_size, device = init_distributed()
    train_section = dict(config.get("train", {}) or {})
    validate_training_topology(config, world_size=world_size)
    seed = int(train_section.get("seed", 20260729))
    semantic_history_corruption_rate = float(
        train_section.get("semantic_history_corruption_rate", 0.0)
    )
    if not 0.0 <= semantic_history_corruption_rate < 1.0:
        raise ValueError("train.semantic_history_corruption_rate must be within [0,1)")
    seed_everything(seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


    strategy = resolve_strategy(config, world_size=world_size)
    validate_precision(config, strategy)
    validate_training_preconditions(config)

    output_dir = Path(str(train_section.get("output_dir", "runs/llm/default")))
    upstream_identity = validate_stage_entry(
        config,
        init_from=init_from,
        resume_from=resume_from,
        check_output=True,
    )


    # ``common/checkpoint._save_sharded``.
    checkpoint_format = str(train_section.get("checkpoint_format", "auto")).lower()
    if checkpoint_format not in ("auto", "sharded", "single_file"):
        raise ValueError(
            f"Unknown train.checkpoint_format={checkpoint_format!r}; expected one of "
            "auto, sharded, or single_file"
        )
    sharded_checkpoints = (
        strategy == "fsdp2" and world_size > 1
        if checkpoint_format == "auto"
        else checkpoint_format == "sharded" and world_size > 1
    )
    last_path = checkpoint_path(output_dir, "last", sharded=sharded_checkpoints)
    final_path = checkpoint_path(output_dir, "final", sharded=sharded_checkpoints)
    best_path = checkpoint_path(output_dir, "best", sharded=sharded_checkpoints)


    watchdog = _ProgressWatchdog(
        rank=rank, warn_seconds=float(train_section.get("stall_warn_seconds", 300.0))
    )
    watchdog.start()
    watchdog.beat(0, phase="startup")


    resume_extra: dict[str, Any] | None = None
    data_parent_step = 0
    inherit_data_state = bool(
        init_from and train_section.get("inherit_data_state_from_parent", False)
    )
    data_state_path = resume_from or (init_from if inherit_data_state else None)
    if data_state_path:
        try:
            resume_sidecar = read_sidecar(data_state_path)
            data_parent_step = int(resume_sidecar.get("global_step", 0))
            per_rank_extra = list(resume_sidecar.get("extra_by_rank") or ())
            if per_rank_extra:
                current_rank = int(os.environ.get("RANK", "0"))
                current_world = int(os.environ.get("WORLD_SIZE", "1"))
                if len(per_rank_extra) != current_world:
                    raise RuntimeError(
                        f"Checkpoint saved per-rank state for {len(per_rank_extra)} ranks, "
                        f"but current world_size={current_world}; exact resume is unavailable"
                    )
                resume_extra = dict(per_rank_extra[current_rank] or {})
            else:

                resume_extra = dict(resume_sidecar.get("extra") or {})
        except (FileNotFoundError, ValueError):
            if inherit_data_state:
                raise RuntimeError(
                    "train.inherit_data_state_from_parent=true, but the parent checkpoint "
                    "has no readable sampler sidecar"
                )
            resume_extra = None
    if resume_from and (resume_extra or {}).get("resume_supported") is False:
        raise RuntimeError(
            "This model-selection checkpoint supports --init-from but not --resume-from"
        )


    cache_dir = str(train_section.get("local_checkpoint_cache_dir", "/tmp/open_qwen_music.llm/ckpt"))
    if world_size > 1:
        if init_from and not is_sharded_checkpoint(init_from):
            init_from = str(stage_checkpoint_locally(init_from, cache_dir=cache_dir))
        if resume_from and not is_sharded_checkpoint(resume_from):
            resume_from = str(stage_checkpoint_locally(resume_from, cache_dir=cache_dir))
        barrier()

    watchdog.beat(0, phase="registry")
    registry = resolve_registry(config)
    text_encoder = build_text_encoder(config)
    validate_text_encoder_namespace(text_encoder, registry)
    sequence_config = SequenceConfig.from_config(config)
    condition_config = ConditionRenderConfig.from_config(config)
    grammar_config = GrammarConfig.from_config(config)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        registry.save(output_dir / "token_registry.json")
        write_run_metadata(
            output_dir,
            config,
            world_size=world_size,
            extra={
                "registry_revision": registry.revision,
                "total_vocab_size": registry.total_vocab_size,
            },
        )
    barrier()

    watchdog.beat(0, phase="dataset")
    data_section = dict(config.get("data", {}) or {})
    corpus_root = data_section.get("corpus")
    if not corpus_root:
        raise KeyError("data.corpus not configured")
    dataset = LlmSampleDataset(
        corpus_root,
        registry,
        text_encoder,
        sequence_config=sequence_config,
        split=data_section.get("split", "train"),
        seed=seed,
        strict_metadata=bool(data_section.get("require_production_contract", False)),
        local_cache_dir=data_section.get("local_cache_dir"),
        condition_config=condition_config,
        grammar_config=grammar_config,
    )
    validate_production_tokenizer_identity(
        config,
        tokenizer_revision=dataset.corpus.tokenizer_revision,
        semantic_extractor_revision=dataset.corpus.metadata.get(
            "semantic_extractor_revision"
        ),
        source=f"corpus {corpus_root}",
    )
    if bool(data_section.get("require_production_contract", False)):
        shard_validation_error: str | None = None
        if local_rank == 0:
            try:
                dataset.corpus.validate_integrity(require=True)
                dataset.index.validate_integrity(
                    expected_manifest_sha256=str(
                        dataset.corpus.metadata.get("manifest_sha256") or ""
                    )
                )
            except Exception as error:
                shard_validation_error = f"{type(error).__name__}: {error}"
        errors = [
            value
            for value in all_gather_objects(shard_validation_error)
            if value is not None
        ]
        if errors:
            raise RuntimeError(f"Corpus payload verification failed: {errors[0]}")
    base_provenance = build_provenance(
        config,
        registry_revision=registry.revision,
        tokenizer_revision=dataset.corpus.tokenizer_revision,
        condition_template_version=dataset.builder.condition_template_version,
        sequence_protocol_revision=dataset.builder.sequence_protocol_revision,
        corpus_metadata=dataset.corpus.metadata,
    )
    provenance = dict(base_provenance)
    if resume_from:

        previous_parent = (
            upstream_identity.get("parent_checkpoint") if upstream_identity is not None else None
        )
        if previous_parent is not None:
            provenance["parent_checkpoint"] = previous_parent
    elif upstream_identity is not None:
        provenance["parent_checkpoint"] = upstream_identity

    step_counter = StepCounter()
    sampler = build_sampler(
        config,
        dataset.index,
        dataset.record_indices,
        rank=rank,
        world_size=world_size,
        step_counter=step_counter,
        semantic_anchor_length_fn=dataset.semantic_anchor_token_bound,
    )
    collator = LlmCollator(
        pad_token_id=registry.pad_id,
        pad_to_multiple_of=int(train_section.get("pad_to_multiple_of", 64)),
        max_sequence_length=sequence_config.max_sequence_length,
    )
    num_workers = int(data_section.get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=(
            int(data_section.get("prefetch_factor", 4))
            if num_workers > 0
            else None
        ),
    )

    watchdog.beat(0, phase="build_model")


    model = build_model(config, registry, device=device).to(device)
    log_kv(
        "model",
        {
            "params_m": model.num_parameters() / 1e6,
            "vocab": registry.total_vocab_size,
            "registry_revision": registry.revision,
            "tokenizer_revision": dataset.corpus.tokenizer_revision,
        },
    )


    allow_mismatch = bool(train_section.get("allow_provenance_mismatch", False))
    upstream = resume_from or init_from
    upstream_is_sharded = bool(upstream) and is_sharded_checkpoint(upstream)
    upstream_expected_provenance = provenance if resume_from else base_provenance
    if upstream and not upstream_is_sharded:
        load_checkpoint(
            upstream,
            model=model,
            resume=False,
            expected_provenance=upstream_expected_provenance,
            allow_provenance_mismatch=allow_mismatch,
        )

    if world_size > 1:


        watchdog.beat(0, phase=f"{strategy}_init")
        log_kv(
            "startup",
            {
                "phase": f"{strategy}_init",
                "strategy": strategy,
                "params_m": model.num_parameters() / 1e6,
            },
        )
        model = wrap_model(
            model,
            strategy=strategy,
            config=config,
            device=device,
            local_rank=local_rank,
            world_size=world_size,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        barrier()


    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    start_step = 0
    if init_from and upstream_is_sharded:

        load_checkpoint(
            init_from,
            model=model,
            resume=False,
            expected_provenance=base_provenance,
            allow_provenance_mismatch=allow_mismatch,
            strategy=strategy,
        )
    if resume_from:
        start_step = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            resume=True,


            load_model=upstream_is_sharded,
            expected_provenance=provenance,
            allow_provenance_mismatch=allow_mismatch,
            strategy=strategy,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        barrier()


    # `model.eval()`(modeling_utils.py:"The model is set in evaluation mode by default"),


    model.train()

    precision = str(train_section.get("precision", "bf16"))
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]
    scaler = torch.amp.GradScaler(enabled=precision == "fp16")
    if resume_from and (resume_extra or {}).get("grad_scaler_state") is not None:
        scaler.load_state_dict(dict((resume_extra or {})["grad_scaler_state"]))

    accumulation = max(1, int(train_section.get("gradient_accumulation_steps", 1)))
    max_steps = int(train_section.get("max_steps", 100000))
    log_every = int(train_section.get("log_every_steps", 10))
    save_every = int(train_section.get("save_every_steps", 1000))
    milestone_every = int(train_section.get("milestone_every_steps", 0))
    if milestone_every < 0:
        raise ValueError("train.milestone_every_steps required >= 0")
    save_final_checkpoint = bool(train_section.get("save_final_checkpoint", True))
    evaluation_section = dict(config.get("evaluation", {}) or {})
    selection = dict(evaluation_section.get("selection", {}) or {})
    selection_enabled = bool(selection.get("enabled", False))
    selection_metric = str(selection.get("metric", "ubnll"))
    selection_mode = str(selection.get("mode", "min"))
    selection_min_delta = float(selection.get("min_delta", 0.0))
    if selection_mode not in {"min", "max"}:
        raise ValueError("evaluation.selection.mode must be min/max")
    if selection_min_delta < 0:
        raise ValueError("evaluation.selection.min_delta cannot be negative")
    best_metric = (
        float((resume_extra or {}).get("best_metric"))
        if resume_from and (resume_extra or {}).get("best_metric") is not None
        else None
    )
    best_step = int((resume_extra or {}).get("best_step", -1)) if resume_from else -1
    if resume_from and best_metric is not None:
        saved_metric = (resume_extra or {}).get("selection_metric")
        saved_mode = (resume_extra or {}).get("selection_mode")
        if saved_metric != selection_metric or saved_mode != selection_mode:
            print(
                "Checkpoint selection contract changed; resetting previous best: "
                f"saved=({saved_metric},{saved_mode}) "
                f"current=({selection_metric},{selection_mode})",
                flush=True,
            )
            best_metric = None
            best_step = -1
    optional_selection_slots: dict[str, dict[str, Any]] = {}
    saved_optional_slots = dict(
        (resume_extra or {}).get("selection_slots") or {}
    ) if resume_from else {}
    for raw_name, raw_spec in dict(selection.get("slots") or {}).items():
        name = str(raw_name)
        if not name or any(not (char.isalnum() or char in "_-") for char in name):
            raise ValueError(
                "evaluation.selection.slots names may contain only letters, digits, "
                "underscores, and hyphens"
            )
        spec = dict(raw_spec or {})
        metric = str(spec.get("metric") or "")
        mode = str(spec.get("mode", "min"))
        min_delta = float(spec.get("min_delta", 0.0))
        if not metric:
            raise ValueError(f"selection slot {name!r} is missing metric")
        if mode not in {"min", "max"}:
            raise ValueError(f"selection slot {name!r} mode must be 'min' or 'max'")
        if min_delta < 0:
            raise ValueError(f"selection slot {name!r} min_delta cannot be negative")
        saved = dict(saved_optional_slots.get(name) or {})
        if saved.get("metric") != metric or saved.get("mode") != mode:
            saved = {}
        optional_selection_slots[name] = {
            "metric": metric,
            "mode": mode,
            "min_delta": min_delta,
            "best_metric": (
                float(saved["best_metric"])
                if saved.get("best_metric") is not None
                else None
            ),
            "best_step": int(saved.get("best_step", -1)),
        }
    eval_every = int(evaluation_section.get("every_steps", 0))
    if eval_every < 0:
        raise ValueError("evaluation.every_steps required >= 0")
    if eval_every > 0 and eval_every % max(log_every, 1) != 0:
        raise ValueError(
            "evaluation.every_steps must be a multiple of train.log_every_steps so CUDA "
            "peak-memory resets do not split a logging window"
        )
    full_eval_every = int(evaluation_section.get("full_every_steps", 0))
    if full_eval_every < 0:
        raise ValueError("evaluation.full_every_steps required >= 0")
    if full_eval_every > 0 and eval_every <= 0:
        raise ValueError("evaluation.full_every_steps requires periodic fast evaluation")
    if full_eval_every > 0 and eval_every > 0 and full_eval_every % eval_every != 0:
        raise ValueError(
            "evaluation.full_every_steps must be evaluation.every_steps "
        )
    generation_every_steps = int(
        (evaluation_section.get("generation") or {}).get("every_steps", 0)
    )
    if generation_every_steps < 0:
        raise ValueError("evaluation.generation.every_steps required >= 0")
    if generation_every_steps > 0 and eval_every <= 0:
        raise ValueError("evaluation.generation.every_steps requires periodic evaluation")
    if (
        generation_every_steps > 0
        and eval_every > 0
        and generation_every_steps % eval_every != 0
    ):
        raise ValueError(
            "evaluation.generation.every_steps must be evaluation.every_steps "
        )
    clip_norm = float(train_section.get("gradient_clip_norm", 1.0))
    steps_per_epoch = max(1, int(train_section.get("steps_per_epoch", 10000)))


    max_consecutive_bad = int(train_section.get("max_consecutive_nonfinite_steps", 5))
    diagnostics_every = int(train_section.get("sampling_diagnostics_every_steps", 0))

    step = start_step
    data_step = initial_data_step(
        optimizer_step=step,
        data_state_active=data_state_path is not None,
        parent_step=data_parent_step,
        extra=resume_extra,
    )
    step_counter.set(data_step)
    epoch = data_step // steps_per_epoch
    saved_sampler_state = (resume_extra or {}).get("sampler_state")
    if saved_sampler_state is not None:
        if num_workers != 0:
            raise RuntimeError(
                "An exact sampler-state resume requires data.num_workers=0 because "
                "DataLoader prefetch can move the saved cursor past consumed data"
            )
        sampler.load_state_dict(dict(saved_sampler_state))
        if step_counter.value != data_step:
            raise RuntimeError(
                f"sampler state step={step_counter.value} != data step={data_step}"
            )
        epoch = sampler.epoch
        epoch_batches = int((resume_extra or {}).get("epoch_batches", 0))
    else:
        if inherit_data_state:
            raise RuntimeError(
                "Parent checkpoint is missing sampler state required to continue the data cursor"
            )
        sampler.set_epoch(epoch)
        epoch_batches = (
            _skip_replayed_batches(
                sampler,
                resume_extra=resume_extra,
                step=step,
                epoch=epoch,
                steps_per_epoch=steps_per_epoch,
                accumulation=max(
                    1,
                    int(
                        train_section.get(
                            "gradient_accumulation_steps", 1
                        )
                    ),
                ),
            )
        )
    iterator = _infinite(loader)
    stopwatch = Stopwatch()
    cumulative_global_semantic_tokens = float(
        (resume_extra or {}).get("cumulative_global_semantic_tokens", 0.0)
    )
    cumulative_global_loss_tokens = float(
        (resume_extra or {}).get("cumulative_global_loss_tokens", 0.0)
    )
    cumulative_global_melody_pitch_tokens = float(
        (resume_extra or {}).get("cumulative_global_melody_pitch_tokens", 0.0)
    )
    cumulative_global_melody_struct_tokens = float(
        (resume_extra or {}).get("cumulative_global_melody_struct_tokens", 0.0)
    )
    window: dict[str, float] = {}
    window_counts: dict[str, int] = {}
    window_metric_numerators: dict[str, float] = {}
    window_metric_denominators: dict[str, float] = {}
    window_tokens = 0.0
    window_semantic = 0.0
    window_semantic_history_corrupted = 0.0
    window_melody_pitch = 0.0
    window_melody_struct = 0.0
    window_padded = 0.0
    window_data_seconds = 0.0
    window_global_loss_tokens = 0.0
    window_local_loss_tokens = 0.0
    window_grad_norms: list[float] = []
    latest_stability_snapshot: dict[str, float] = {}
    accum_local_loss_tokens = 0.0
    micro = 0
    nonfinite_micro_batches = 0
    split_batches = 0
    split_parts = 0


    token_budget = int(train_section.get("max_tokens_per_gpu", 8192))
    pad_multiple = int(train_section.get("pad_to_multiple_of", 64))


    split_to_budget = bool(train_section.get("split_batches_to_budget", True))
    guard = NonFiniteStepGuard(
        limit=max_consecutive_bad,
        checkpoint_hint=f"The latest checkpoint is {output_dir / 'last.pt'}.",
    )
    if resume_from and (resume_extra or {}).get("nonfinite_guard_state") is not None:
        guard.load_state_dict(dict((resume_extra or {})["nonfinite_guard_state"]))
    inner_model = model.module if hasattr(model, "module") else model
    periodic_evaluator: _LoadedModelEvaluator | None = None
    periodic_eval_runs = 0
    if eval_every > 0:
        watchdog.beat(step, phase="build_evaluator")
        periodic_evaluator = _LoadedModelEvaluator(
            config,
            registry,
            text_encoder,
            split=str(evaluation_section.get("split", "valid")),
            max_batches=int(evaluation_section.get("max_batches", 8)),
            rank=rank,
            world_size=world_size,
            device=device,
        )

        stopwatch.lap()
        watchdog.beat(step)

    peak_lr, final_lr = lr_endpoints(config)
    log_kv(
        "train_start",
        {
            "step": step,
            "data_step": data_step,
            "max_steps": max_steps,
            "world_size": world_size,
            "accumulation": accumulation,
            "max_tokens_per_gpu": train_section.get("max_tokens_per_gpu"),
            "max_sequence_length": sequence_config.max_sequence_length,
            "max_semantic_frames": sequence_config.max_semantic_frames,
            "max_condition_tokens": sequence_config.max_condition_tokens,
            "precision": precision,
            "peak_lr": peak_lr,
            "final_lr": final_lr,
            "warmup_steps": (config.get("optimizer", {}) or {}).get("warmup_steps"),
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
            "partitioned_embeddings": isinstance(
                inner_model.backbone.get_input_embeddings(),
                PartitionedVocabularyEmbedding,
            ),
            "tie_music_embeddings": (
                tuple(getattr(inner_model.lm_head, "tie_namespaces", ()))
                == MUSIC_EMBEDDING_NAMESPACES
            ),
            "tie_embedding_namespaces": ",".join(
                getattr(inner_model.lm_head, "tie_namespaces", ())
            ),
            "semantic_output_residual_rank": int(
                getattr(inner_model.lm_head, "semantic_output_residual_rank", 0)
            ),
            "modality_type_embeddings": (
                inner_model.modality_type_embeddings is not None
            ),
            "grammar_constrained_loss": inner_model.grammar_constrained_loss,
            "grammar_constrained_metrics": (
                inner_model.grammar_constrained_metrics
            ),
            "melody_unvoiced_id": inner_model.melody_unvoiced_id,
            "melody_unvoiced_weight": inner_model.melody_unvoiced_weight,
            "grad_clip": clip_norm,
            "loss_chunk_size": (config.get("model", {}) or {}).get("loss_chunk_size"),
            "gradient_checkpointing": (config.get("model", {}) or {}).get(
                "gradient_checkpointing"
            ),
            "semantic_history_corruption_rate": semantic_history_corruption_rate,


            "gc_active": bool(getattr(inner_model.backbone, "is_gradient_checkpointing", False))
            and inner_model.training,
            "training_mode": inner_model.training,
            "attn": getattr(inner_model.backbone.config, "_attn_implementation", "unknown"),
            "eval_every_steps": eval_every,
            "full_eval_every_steps": full_eval_every,
            "generation_every_steps": generation_every_steps,
            "eval_split": (
                str(evaluation_section.get("split", "valid"))
                if periodic_evaluator is not None
                else "disabled"
            ),
        },
    )
    _log_sampling_diagnostics(sampler, data_step)
    _warn_batch_budget_risk(config)


    watchdog.beat(0, phase="first_batch")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while step < max_steps:
        data_started = time.monotonic()
        batch = next(iterator)
        window_data_seconds += time.monotonic() - data_started
        micro += 1
        epoch_batches += 1
        is_sync_step = micro % accumulation == 0
        want_metrics = is_sync_step and ((step + 1) % log_every == 0)


        parts = (
            split_batch_to_token_budget(
                batch, budget=token_budget, pad_to_multiple_of=pad_multiple
            )
            if split_to_budget
            else [batch]
        )


        if strategy == "fsdp2" and world_size > 1:
            require_equal_across_ranks(len(parts), name="FSDP2 split part count")
        if len(parts) > 1:
            split_batches += 1
            split_parts += len(parts)
            if split_batches <= 3 or split_batches % 200 == 0:


                log_kv(
                    "batch_split",
                    {
                        "step": step,
                        "parts": len(parts),
                        "batch": int(batch["input_ids"].shape[0]),
                        "padded_tokens": int(batch["padded_tokens"]),
                        "budget": token_budget,
                        "ratio": float(batch["padded_tokens"]) / max(token_budget, 1),
                        "split_batches_total": split_batches,
                    },
                    main_only=False,
                )

        part_supervised = [
            float(part.get("supervised_tokens", int((part["labels"] != IGNORE_LABEL).sum())))
            for part in parts
        ]
        supervised_total = max(sum(part_supervised), 1.0)

        micro_finite = True
        part_metrics: dict[str, list[tuple[float, float]]] = {}
        for position, part in enumerate(parts):
            input_ids = part["input_ids"].to(device, non_blocking=True)
            labels = part["labels"].to(device, non_blocking=True)
            attention_mask = part["attention_mask"].to(device, non_blocking=True)
            input_ids, corrupted_history = corrupt_semantic_history(
                input_ids,
                attention_mask,
                registry=registry,
                rate=semantic_history_corruption_rate,
                seed=(
                    seed
                    + rank * 1_000_003
                    + step * 10_007
                    + (micro % accumulation) * 101
                    + position
                ),
            )
            window_semantic_history_corrupted += float(corrupted_history)
            constraint_kinds = (
                part["constraint_kinds"].to(device, non_blocking=True)
                if "constraint_kinds" in part
                else None
            )
            share = part_supervised[position] / supervised_total


            sync_now = is_sync_step and position == len(parts) - 1
            with gradient_sync(model, strategy=strategy, sync=sync_now):
                if autocast_dtype is None:
                    output = _forward(
                        model,
                        input_ids,
                        labels,
                        attention_mask,
                        want_metrics,
                        constraint_kinds=constraint_kinds,
                    )
                else:
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                        output = _forward(
                            model,
                            input_ids,
                            labels,
                            attention_mask,
                            want_metrics,
                            constraint_kinds=constraint_kinds,
                        )


                loss_tokens = int(output.num_loss_tokens or 0)
                scaler.scale(output.loss * loss_tokens).backward()
                accum_local_loss_tokens += float(loss_tokens)


            numeric_metrics = {name: float(value) for name, value in output.metrics.items()}
            for name, numeric in numeric_metrics.items():
                if not math.isfinite(numeric):
                    micro_finite = False
                    continue
                part_metrics.setdefault(name, []).append((numeric, share))
                weight = metric_token_weight(
                    name,
                    numeric_metrics,
                    num_loss_tokens=int(output.num_loss_tokens or 0),
                    accuracy_limit=int(inner_model.accuracy_sample_positions),
                )
                if weight is not None and weight > 0.0:
                    window_metric_numerators[name] = (
                        window_metric_numerators.get(name, 0.0) + numeric * weight
                    )
                    window_metric_denominators[name] = (
                        window_metric_denominators.get(name, 0.0) + weight
                    )
            window_tokens += float(part["real_tokens"])
            window_padded += float(part["padded_tokens"])
            window_semantic += float(part["num_semantic_tokens"].sum())
            window_melody_pitch += numeric_metrics.get("tokens_melody_pitch", 0.0)
            window_melody_struct += numeric_metrics.get("tokens_melody_struct", 0.0)

        for name, value in merge_part_metrics(part_metrics).items():
            window[name] = window.get(name, 0.0) + value
            window_counts[name] = window_counts.get(name, 0) + 1
        part_metrics.clear()

        if not micro_finite:
            nonfinite_micro_batches += 1
            log_kv(
                "train_anomaly",
                {
                    "step": step,
                    "micro": micro,
                    "kind": "nonfinite_loss",
                    "longest_seq": int(batch["input_ids"].shape[-1]),
                    "batch": int(batch["input_ids"].shape[0]),
                },
                main_only=False,
            )

        if not is_sync_step:
            continue

        if precision == "fp16":
            scaler.unscale_(optimizer)
        global_loss_tokens = reduce_scalar_sum(accum_local_loss_tokens)
        if global_loss_tokens <= 0:
            watchdog.stop()
            raise RuntimeError("Entire accumulation window contains no tokens that contribute to loss")

        scale_gradients(model, world_size / global_loss_tokens)
        window_global_loss_tokens += global_loss_tokens
        cumulative_global_loss_tokens += global_loss_tokens
        window_local_loss_tokens += accum_local_loss_tokens
        accum_local_loss_tokens = 0.0
        grad_norm = clip_grad_norm(model, strategy=strategy, max_norm=clip_norm)
        try:
            apply_step = guard.observe(grad_norm)
        except RuntimeError:
            watchdog.stop()
            raise
        if apply_step:
            scaler.step(optimizer)
            window_grad_norms.append(grad_norm)
        else:
            log_kv(
                "train_anomaly",
                {
                    "step": step + 1,
                    "kind": "nonfinite_grad_skipped",
                    "grad_norm": grad_norm,
                    "consecutive": guard.consecutive,
                    "skipped_total": guard.skipped,
                },
                main_only=False,
            )
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1
        data_step += 1
        step_counter.set(data_step)
        watchdog.beat(step)
        if data_step // steps_per_epoch != epoch:


            epoch = data_step // steps_per_epoch
            sampler.set_epoch(epoch)
            epoch_batches = 0

        if step % log_every == 0:
            elapsed = stopwatch.lap()
            averaged = average_window(window, window_counts)
            reduced = reduce_metrics(averaged)

            reduced.update(
                reduce_weighted_metrics(
                    window_metric_numerators,
                    window_metric_denominators,
                )
            )
            total_tokens = reduce_scalar_sum(window_tokens)
            total_semantic = reduce_scalar_sum(window_semantic)
            total_semantic_history_corrupted = reduce_scalar_sum(
                window_semantic_history_corrupted
            )
            total_melody_pitch = reduce_scalar_sum(window_melody_pitch)
            total_melody_struct = reduce_scalar_sum(window_melody_struct)
            total_padded = reduce_scalar_sum(window_padded)
            cumulative_global_semantic_tokens += total_semantic
            cumulative_global_melody_pitch_tokens += total_melody_pitch
            cumulative_global_melody_struct_tokens += total_melody_struct
            norms = sorted(window_grad_norms) or [float("nan")]
            latest_stability_snapshot = {
                "training_stability/grad_norm_p50": float(
                    np.quantile(norms, 0.50)
                ),
                "training_stability/grad_norm_p90": float(
                    np.quantile(norms, 0.90)
                ),
                "training_stability/grad_norm_max": float(norms[-1]),
                "training_stability/clip_fraction": float(
                    sum(1 for value in window_grad_norms if value > clip_norm)
                    / max(len(window_grad_norms), 1)
                ),
                "training_stability/nonfinite_micro_total": float(
                    nonfinite_micro_batches
                ),
                "training_stability/skipped_steps_total": float(guard.skipped),
            }
            peak_gib = 0.0
            if device.type == "cuda":
                peak_gib = _reduce_max(
                    torch.cuda.max_memory_allocated(device) / 2**30, device
                )
                torch.cuda.reset_peak_memory_stats(device)
            log_kv(
                "train",
                {
                    "step": step,
                    "data_step": data_step,
                    "max_steps": max_steps,
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],


                    "grad_norm": norms[len(norms) // 2],
                    "grad_norm_max": norms[-1],


                    "clip_frac": sum(1 for n in window_grad_norms if n > clip_norm)
                    / max(len(window_grad_norms), 1),
                    "steps_per_sec": log_every / max(elapsed, 1e-6),
                    "tokens_per_sec": total_tokens / max(elapsed, 1e-6),
                    "audio_sec_per_sec": total_semantic / 25.0 / max(elapsed, 1e-6),
                    "pad_waste": 1.0 - total_tokens / max(total_padded, 1.0),


                    "data_wait_frac": window_data_seconds / max(elapsed, 1e-6),
                    "mem_gib": peak_gib,
                    "skipped": guard.skipped,
                    "nonfinite_micro": nonfinite_micro_batches,


                    "split_batches": split_batches,
                    "split_parts": split_parts,
                    **{key: value for key, value in reduced.items() if key.startswith("loss")},
                    **{key: value for key, value in reduced.items() if key.startswith("acc")},
                    "sup_tokens_rank0_local": window_local_loss_tokens,
                    "sup_tokens_global": window_global_loss_tokens,
                    "semantic_history_corrupted_tokens": (
                        total_semantic_history_corrupted
                    ),
                    "semantic_history_corruption_effective_rate": (
                        total_semantic_history_corrupted
                        / max(total_semantic, 1.0)
                    ),
                    "cumulative_global_loss_tokens": cumulative_global_loss_tokens,
                    "cumulative_global_semantic_tokens": (
                        cumulative_global_semantic_tokens
                    ),
                    "cumulative_global_melody_pitch_tokens": (
                        cumulative_global_melody_pitch_tokens
                    ),
                    "cumulative_global_melody_struct_tokens": (
                        cumulative_global_melody_struct_tokens
                    ),
                    "cumulative_sampled_audio_hours": (
                        cumulative_global_semantic_tokens / 25.0 / 3600.0
                    ),
                },
            )
            window = {}
            window_counts = {}
            window_metric_numerators = {}
            window_metric_denominators = {}
            window_tokens = 0.0
            window_padded = 0.0
            window_semantic = 0.0
            window_semantic_history_corrupted = 0.0
            window_melody_pitch = 0.0
            window_melody_struct = 0.0
            window_data_seconds = 0.0
            window_global_loss_tokens = 0.0
            window_local_loss_tokens = 0.0
            window_grad_norms = []

        if diagnostics_every > 0 and step % diagnostics_every == 0:
            _log_sampling_diagnostics(sampler, data_step)

        if periodic_evaluator is not None and step % eval_every == 0:
            watchdog.beat(step, phase="evaluation")
            eval_started = time.monotonic()
            was_training = model.training
            model.eval()
            embedding_every = int(
                evaluation_section.get("embedding_norm_every_evals", 0)
            )
            include_embeddings = (
                embedding_every > 0
                and (periodic_eval_runs + 1) % embedding_every == 0
            )
            try:
                with torch.no_grad():
                    eval_results = periodic_evaluator.run(
                        model,
                        step=step,
                        include_embedding_norms=include_embeddings,
                    )
                    if full_eval_every > 0 and step % full_eval_every == 0:
                        watchdog.beat(step, phase="full_evaluation")
                        full_started = time.monotonic()
                        full_results = periodic_evaluator.run(
                            model,
                            step=step,
                            include_embedding_norms=False,
                            full=True,
                        )
                        full_seconds = time.monotonic() - full_started
                        if device.type == "cuda":
                            full_seconds = _reduce_max(full_seconds, device)
                        eval_results.update(
                            {
                                f"full/{name}": value
                                for name, value in full_results.items()
                            }
                        )
                        eval_results["full/evaluation_seconds"] = full_seconds
                    eval_results["cumulative_global_loss_tokens"] = (
                        cumulative_global_loss_tokens
                    )
                    eval_results["cumulative_global_semantic_tokens"] = (
                        cumulative_global_semantic_tokens
                    )
                    eval_results["cumulative_global_melody_pitch_tokens"] = (
                        cumulative_global_melody_pitch_tokens
                    )
                    eval_results["cumulative_global_melody_struct_tokens"] = (
                        cumulative_global_melody_struct_tokens
                    )
                    eval_results["cumulative_sampled_audio_hours"] = (
                        cumulative_global_semantic_tokens / 25.0 / 3600.0
                    )
                    eval_results.update(latest_stability_snapshot)
            finally:
                model.train(was_training)
            periodic_eval_runs += 1
            eval_seconds = time.monotonic() - eval_started
            if device.type == "cuda":
                eval_seconds = _reduce_max(eval_seconds, device)
            eval_results["evaluation_seconds"] = eval_seconds
            if device.type == "cuda":
                eval_results["evaluation_peak_mem_gib"] = _reduce_max(
                    torch.cuda.max_memory_allocated(device) / 2**30,
                    device,
                )


                torch.cuda.reset_peak_memory_stats(device)
            new_best = False
            improved_optional_slots: list[str] = []
            if selection_enabled:
                selected_value = eval_results.get(selection_metric)
                local_value = (
                    float(selected_value)
                    if selected_value is not None
                    and math.isfinite(float(selected_value))
                    else None
                )
                value = require_consistent_selection_metric(
                    all_gather_objects(local_value),
                    metric=selection_metric,
                )
                new_best = metric_improved(
                    value,
                    best_metric,
                    mode=selection_mode,
                    min_delta=selection_min_delta,
                )
                if new_best:
                    best_metric = value
                    best_step = step
                eval_results["selection/best_metric"] = float(best_metric)
                eval_results["selection/best_step"] = float(best_step)
                eval_results["selection/is_new_best"] = float(new_best)
                for slot_name, slot in optional_selection_slots.items():
                    slot_metric = str(slot["metric"])
                    selected = eval_results.get(slot_metric)
                    local_slot_value = (
                        float(selected)
                        if selected is not None and math.isfinite(float(selected))
                        else None
                    )
                    gathered_slot_values = all_gather_objects(local_slot_value)
                    if all(value is None for value in gathered_slot_values):
                        continue
                    slot_value = require_consistent_selection_metric(
                        gathered_slot_values,
                        metric=slot_metric,
                    )
                    slot_improved = metric_improved(
                        slot_value,
                        slot["best_metric"],
                        mode=str(slot["mode"]),
                        min_delta=float(slot["min_delta"]),
                    )
                    if slot_improved:
                        slot["best_metric"] = slot_value
                        slot["best_step"] = step
                        improved_optional_slots.append(slot_name)
                    prefix = f"selection_slots/{slot_name}"
                    eval_results[f"{prefix}/best_metric"] = float(
                        slot["best_metric"]
                    )
                    eval_results[f"{prefix}/best_step"] = float(slot["best_step"])
                    eval_results[f"{prefix}/is_new_best"] = float(slot_improved)
            log_kv(
                "periodic_eval",
                {
                    "step": step,
                    "split": str(evaluation_section.get("split", "valid")),
                    "seconds": eval_seconds,
                    **{
                        key: value
                        for key, value in eval_results.items()
                        if key
                        in {
                            "plain/loss_semantic",
                            "section/loss_semantic",
                            "unique_section/loss_semantic",
                            "section/unpaired_loss_difference_vs_plain",
                            "unique_section/unpaired_loss_difference_vs_plain",
                            "stage4/ctc/hypothesis/token_error_rate",
                            "stage4/ctc/delta/token_error_rate",
                            "stage4/mel/hypothesis_to_reference/l1",
                            "stage4/chroma/hypothesis_to_reference/frame_cosine",
                            "full/ubnll",
                            "full/balanced_semantic_nll",
                            "full/balanced_melody_pitch_nll",
                            "full/balanced_melody_struct_nll",
                            "full/evaluation_seconds",
                        }
                    },
                },
            )
            if is_main_process() and bool(
                evaluation_section.get("write_step_json", True)
            ):
                evaluation_dir = output_dir / "evaluations"
                evaluation_dir.mkdir(parents=True, exist_ok=True)
                destination = evaluation_dir / f"step_{step:08d}.json"
                temporary = destination.with_name(
                    f".{destination.name}.tmp-{os.getpid()}"
                )
                serializable = {
                    key: (float(value) if math.isfinite(float(value)) else None)
                    for key, value in sorted(eval_results.items())
                }
                temporary.write_text(
                    json.dumps(
                        {
                            "format_version": "oqm.llm.periodic-eval.v1",
                            "step": step,
                            "split": str(evaluation_section.get("split", "valid")),
                            "metadata": periodic_evaluator.metadata(),
                            "metrics": serializable,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                os.replace(temporary, destination)
            if new_best:
                watchdog.beat(step, phase="best_checkpoint")
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=None,
                    scheduler=scheduler,
                    config=config,
                    global_step=step,
                    provenance=provenance,
                    extra={
                        "selection_metric": selection_metric,
                        "selection_mode": selection_mode,
                        "best_metric": best_metric,
                        "best_step": best_step,
                        "selection_slots": optional_selection_slots,
                        "data_step": data_step,
                        "resume_exact": False,
                        "resume_supported": False,
                    },
                    strategy=strategy,
                    sharded=sharded_checkpoints,
                )
                log_kv(
                    "best_checkpoint",
                    {
                        "step": step,
                        "metric": selection_metric,
                        "value": best_metric,
                        "path": str(best_path),
                    },
                )
            for slot_name in improved_optional_slots:
                slot = optional_selection_slots[slot_name]
                slot_path = checkpoint_path(
                    output_dir,
                    f"best_{slot_name}",
                    sharded=sharded_checkpoints,
                )
                watchdog.beat(step, phase=f"best_checkpoint_{slot_name}")
                save_checkpoint(
                    slot_path,
                    model=model,
                    optimizer=None,
                    scheduler=scheduler,
                    config=config,
                    global_step=step,
                    provenance=provenance,
                    extra={
                        "selection_slot": slot_name,
                        "selection_metric": slot["metric"],
                        "selection_mode": slot["mode"],
                        "best_metric": slot["best_metric"],
                        "best_step": slot["best_step"],
                        "selection_slots": optional_selection_slots,
                        "data_step": data_step,
                        "resume_exact": False,
                        "resume_supported": False,
                    },
                    strategy=strategy,
                    sharded=sharded_checkpoints,
                )
                log_kv(
                    "best_checkpoint",
                    {
                        "step": step,
                        "slot": slot_name,
                        "metric": slot["metric"],
                        "value": slot["best_metric"],
                        "path": str(slot_path),
                    },
                )
            barrier()


            stopwatch.lap()
            watchdog.beat(step)

        if milestone_every > 0 and step % milestone_every == 0:
            milestone_path = checkpoint_path(
                output_dir,
                f"milestone_{step:06d}",
                sharded=sharded_checkpoints,
            )
            watchdog.beat(step, phase="milestone_checkpoint")
            save_checkpoint(
                milestone_path,
                model=model,
                optimizer=None,
                scheduler=scheduler,
                config=config,
                global_step=step,
                provenance=provenance,
                extra={
                    "checkpoint_role": "fixed_milestone",
                    "data_step": data_step,
                    "selection_metric": (
                        selection_metric if selection_enabled else None
                    ),
                    "selection_mode": selection_mode if selection_enabled else None,
                    "best_metric": best_metric,
                    "best_step": best_step,
                    "selection_slots": optional_selection_slots,
                    "resume_exact": False,
                    "resume_supported": False,
                },
                strategy=strategy,
                sharded=sharded_checkpoints,
            )
            log_kv(
                "milestone_checkpoint",
                {
                    "step": step,
                    "path": str(milestone_path),
                },
            )
            barrier()

        if save_every > 0 and step % save_every == 0:


            watchdog.beat(step, phase="checkpoint")


            save_started = time.monotonic()
            save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                global_step=step,
                provenance=provenance,
                extra={
                    "total_vocab_size": registry.total_vocab_size,
                    "data_step": data_step,
                    "selection_metric": selection_metric if selection_enabled else None,
                    "selection_mode": selection_mode if selection_enabled else None,
                    "best_metric": best_metric,
                    "best_step": best_step,
                    "selection_slots": optional_selection_slots,

                    "epoch_batches": epoch_batches,
                    "sampler_state": sampler.state_dict() if num_workers == 0 else None,

                    "resume_exact": False,
                    "resume_state_exact": (
                        num_workers == 0
                        and not bool(train_section.get("allow_resume_config_mismatch", False))
                    ),
                    "grad_scaler_state": scaler.state_dict(),
                    "nonfinite_guard_state": guard.state_dict(),
                    "cumulative_global_loss_tokens": cumulative_global_loss_tokens,
                    "cumulative_global_semantic_tokens": (
                        cumulative_global_semantic_tokens
                    ),
                    "cumulative_global_melody_pitch_tokens": (
                        cumulative_global_melody_pitch_tokens
                    ),
                    "cumulative_global_melody_struct_tokens": (
                        cumulative_global_melody_struct_tokens
                    ),
                },
                strategy=strategy,
                sharded=sharded_checkpoints,
            )


            log_kv(
                "checkpoint",
                {
                    "step": step,
                    "seconds": time.monotonic() - save_started,
                    "path": str(last_path),
                },
            )
            barrier()
            watchdog.beat(step)

    _log_sampling_diagnostics(sampler, data_step)
    if save_final_checkpoint:
        save_checkpoint(
            final_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            global_step=step,
            provenance=provenance,
            extra={
                "total_vocab_size": registry.total_vocab_size,
                "data_step": data_step,
                "selection_metric": selection_metric if selection_enabled else None,
                "selection_mode": selection_mode if selection_enabled else None,
                "best_metric": best_metric,
                "best_step": best_step,
                "selection_slots": optional_selection_slots,
                "epoch_batches": epoch_batches,
                "sampler_state": sampler.state_dict() if num_workers == 0 else None,
                "resume_exact": False,
                "resume_state_exact": (
                    num_workers == 0
                    and not bool(train_section.get("allow_resume_config_mismatch", False))
                ),
                "grad_scaler_state": scaler.state_dict(),
                "nonfinite_guard_state": guard.state_dict(),
                "cumulative_global_loss_tokens": cumulative_global_loss_tokens,
                "cumulative_global_semantic_tokens": (
                    cumulative_global_semantic_tokens
                ),
                "cumulative_global_melody_pitch_tokens": (
                    cumulative_global_melody_pitch_tokens
                ),
                "cumulative_global_melody_struct_tokens": (
                    cumulative_global_melody_struct_tokens
                ),
            },
            strategy=strategy,
            sharded=sharded_checkpoints,
        )
    if is_main_process():
        log_kv(
            "train_done",
            {
                "step": step,
                "skipped_steps": guard.skipped,
                "nonfinite_micro": nonfinite_micro_batches,
                "split_batches": split_batches,
                "split_parts": split_parts,
                "path": str(final_path) if save_final_checkpoint else "not_saved",
                "save_final_checkpoint": save_final_checkpoint,
                "cumulative_global_loss_tokens": cumulative_global_loss_tokens,
                "cumulative_global_semantic_tokens": (
                    cumulative_global_semantic_tokens
                ),
                "cumulative_global_melody_pitch_tokens": (
                    cumulative_global_melody_pitch_tokens
                ),
                "cumulative_global_melody_struct_tokens": (
                    cumulative_global_melody_struct_tokens
                ),
            },
        )
    barrier()
    watchdog.stop()
    dataset.reset_handles()
    if periodic_evaluator is not None:
        periodic_evaluator.close()
    del inner_model
    cleanup_distributed()
    return final_path


def average_window(
    totals: dict[str, float], counts: dict[str, int]
) -> dict[str, float]:
    return {key: value / max(1, counts.get(key, 1)) for key, value in totals.items()}


def corrupt_semantic_history(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    registry: TokenRegistry,
    rate: float,
    seed: int,
) -> tuple[torch.Tensor, int]:

    rate = float(rate)
    if rate <= 0.0:
        return input_ids, 0
    if rate >= 1.0:
        raise ValueError("semantic_history_corruption_rate required < 1")
    semantic = (
        (input_ids >= registry.semantic_base)
        & (input_ids < registry.semantic_base + registry.semantic_size)
        & attention_mask.bool()
    )
    if not bool(semantic.any()):
        return input_ids, 0
    generator = torch.Generator(device=input_ids.device).manual_seed(int(seed))
    selected = semantic & (
        torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
        < rate
    )
    count = int(selected.sum().item())
    if count <= 0:
        return input_ids, 0
    output = input_ids.clone()
    original_local = output[selected] - registry.semantic_base

    offsets = torch.randint(
        1,
        registry.semantic_size,
        (count,),
        device=input_ids.device,
        generator=generator,
    )
    replacement_local = (original_local + offsets) % registry.semantic_size
    output[selected] = replacement_local + registry.semantic_base
    return output, count


def _forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    compute_accuracy: bool,
    *,
    constraint_kinds: torch.Tensor | None = None,
    compute_diagnostics: bool = False,
    return_semantic_predictions: bool = False,
):
    if isinstance(model, (DistributedDataParallel,)):
        return model(
            input_ids,
            labels,
            attention_mask,
            constraint_kinds,
            compute_accuracy=compute_accuracy,
            compute_diagnostics=compute_diagnostics,
            return_semantic_predictions=return_semantic_predictions,
        )
    assert isinstance(model, MusicLLM)
    return model(
        input_ids,
        labels,
        attention_mask,
        constraint_kinds,
        compute_accuracy=compute_accuracy,
        compute_diagnostics=compute_diagnostics,
        return_semantic_predictions=return_semantic_predictions,
    )


def _reference_sensitivity(
    model: MusicLLM,
    dataset: LlmSampleDataset,
    collator: LlmCollator,
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    limit: int = 32,
) -> dict[str, float]:
    from .contracts import SequenceMode
    from .data.dataset import _to_item
    from .data.shards import TokenSpan

    candidates = [
        int(index)
        for index in dataset.record_indices
        if int(dataset.index.melody_frames[int(index)]) > 0
    ]
    if len(candidates) < 2:
        return {}
    if len(candidates) > limit:
        positions = np.linspace(0, len(candidates) - 1, limit).round().astype(int)
        candidates = [candidates[int(position)] for position in sorted(set(positions.tolist()))]
    local = candidates[rank::world_size]


    rounds = math.ceil(len(candidates) / max(world_size, 1))
    output: dict[str, float] = {}
    for mode in (SequenceMode.COVER_SECTION, SequenceMode.COVER_UNIQUE_SECTION):
        matched_sum = 0.0
        mismatched_sum = 0.0
        token_count = 0.0
        scored = 0
        for position in range(rounds):
            active = position < len(local)
            record_index = local[position] if active else candidates[0]
            donor_index = candidates[(candidates.index(record_index) + 1) % len(candidates)]
            record = dataset.raw_record(record_index)
            donor = dataset.raw_record(donor_index)
            semantic = dataset.corpus.read_semantic(TokenSpan.from_dict(record["semantic"]))
            matched_melody = dataset.corpus.read_melody(TokenSpan.from_dict(record["melody"]))
            donor_melody = dataset.corpus.read_melody(TokenSpan.from_dict(donor["melody"]))
            built_matched = dataset.builder.build(
                record,
                mode=mode,
                semantic=semantic,
                melody=matched_melody,
                sample_index=record_index,
                seed=dataset.seed,
            )
            built_mismatched = dataset.builder.build(
                record,
                mode=mode,
                semantic=semantic,
                melody=donor_melody,
                sample_index=record_index,
                seed=dataset.seed,
            )
            pair = collator(
                [_to_item(built_matched, record), _to_item(built_mismatched, record)]
            )
            with torch.no_grad():
                first = model(
                    pair["input_ids"][:1].to(device),
                    pair["labels"][:1].to(device),
                    pair["attention_mask"][:1].to(device),
                )
                second = model(
                    pair["input_ids"][1:].to(device),
                    pair["labels"][1:].to(device),
                    pair["attention_mask"][1:].to(device),
                )
            if (
                not active
                or built_matched.mode is not mode
                or built_mismatched.mode is not mode
            ):
                continue
            count = float(first.metrics.get("tokens_semantic", 0.0))
            if count <= 0:
                continue
            matched_sum += float(first.metrics["loss_semantic"]) * count
            mismatched_sum += float(second.metrics["loss_semantic"]) * count
            token_count += count
            scored += 1
        reduced = reduce_weighted_metrics(
            {
                "matched": matched_sum,
                "mismatched": mismatched_sum,
            },
            {
                "matched": token_count,
                "mismatched": token_count,
            },
        )
        if "matched" in reduced:
            prefix = f"{mode.value}/reference"
            output[f"{prefix}_matched_loss_semantic"] = reduced["matched"]
            output[f"{prefix}_mismatched_loss_semantic"] = reduced["mismatched"]
            output[f"{prefix}_nll_gap"] = reduced["mismatched"] - reduced["matched"]
            output[f"{prefix}_samples"] = reduce_scalar_sum(float(scored))
    return output


@torch.no_grad()
def _embedding_norm_metrics(model: MusicLLM) -> dict[str, float]:

    registry = model.registry
    ranges = {
        "text": (0, registry.text_vocab_size),
        "control": (registry.control_base, registry.control_base + registry.num_control),
        "semantic": (registry.semantic_base, registry.semantic_base + registry.semantic_size),
        "melody": (registry.melody_base, registry.melody_base + registry.melody_size),
        "reserved": (
            registry.melody_base + registry.melody_size,
            registry.total_vocab_size,
        ),
    }
    matrices = {
        "input_embedding": model.backbone.get_input_embeddings(),
        "lm_head": model.lm_head,
    }
    output: dict[str, float] = {}
    for matrix_name, module in matrices.items():
        for namespace, (start, end) in ranges.items():
            if end <= start:
                continue
            weight = vocabulary_weight_range(module, start, end)

            if hasattr(weight, "full_tensor"):
                weight = weight.full_tensor()
            elif hasattr(weight.data, "full_tensor"):
                weight = weight.data.full_tensor()
            norms = torch.linalg.vector_norm(
                weight.detach().to(torch.float32),
                dim=1,
            )
            prefix = f"embedding_norm/{matrix_name}/{namespace}"
            output[f"{prefix}_mean"] = float(norms.mean().item())
            output[f"{prefix}_p50"] = float(norms.median().item())
            output[f"{prefix}_p95"] = float(torch.quantile(norms, 0.95).item())
    return output


def _reduce_sum_mapping(local: dict[str, float]) -> dict[str, float]:

    gathered = all_gather_objects(local)
    names = sorted(
        {
            str(name)
            for payload in gathered
            if isinstance(payload, dict)
            for name in payload
        }
    )
    return {
        name: sum(
            float(payload.get(name, 0.0))
            for payload in gathered
            if isinstance(payload, dict)
        )
        for name in names
    }


def _evenly_select(values: list[Any], limit: int) -> list[Any]:
    if limit <= 0 or len(values) <= limit:
        return list(values)
    positions = np.linspace(0, len(values) - 1, limit).round().astype(int)
    return [values[int(position)] for position in sorted(set(positions.tolist()))]


def _replace_token_namespace(
    source: np.ndarray,
    donor: np.ndarray,
    *,
    predicate: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:

    output = np.asarray(source, dtype=np.int64).copy()
    source_positions = np.flatnonzero(predicate(output))
    donor_tokens = np.asarray(donor, dtype=np.int64)[predicate(np.asarray(donor))]
    if source_positions.size == 0 or donor_tokens.size == 0:
        return output
    replacement = np.resize(donor_tokens, source_positions.size)
    output[source_positions] = replacement
    return output


def _replace_token_namespace_equal_length(
    source: np.ndarray,
    donor: np.ndarray,
    *,
    predicate: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray | None:
    output = np.asarray(source, dtype=np.int64).copy()
    source_positions = np.flatnonzero(predicate(output))
    donor_values = np.asarray(donor, dtype=np.int64)
    donor_tokens = donor_values[predicate(donor_values)]
    if source_positions.size == 0 or donor_tokens.size != source_positions.size:
        return None
    output[source_positions] = donor_tokens
    return output


def _replace_changed_condition_tokens(
    source: np.ndarray,
    variant_condition_ids: list[int],
    *,
    text_vocab_size: int,
) -> np.ndarray:

    output = np.asarray(source, dtype=np.int64).copy()
    positions = np.flatnonzero((output >= 0) & (output < text_vocab_size))
    source_ids = output[positions].tolist()
    variant_ids = [int(value) for value in variant_condition_ids]
    matcher = SequenceMatcher(a=source_ids, b=variant_ids, autojunk=False)
    changed = False
    for operation, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if operation == "equal" or a_end <= a_start:
            continue
        donor = variant_ids[b_start:b_end]
        if not donor:
            donor = variant_ids or source_ids[::-1]
        if not donor:
            continue
        output[positions[a_start:a_end]] = np.resize(
            np.asarray(donor, dtype=np.int64), a_end - a_start
        )
        changed = True

    if not changed and variant_ids != source_ids and positions.size:
        output[positions] = np.resize(
            np.asarray(variant_ids or source_ids[::-1], dtype=np.int64),
            positions.size,
        )
    return output


def _condition_ids_for_record(dataset: LlmSampleDataset, record: dict[str, Any]) -> list[int]:
    from .condition import MusicCondition

    text = MusicCondition.from_record(record).render(
        dataset.builder.condition_config
    )
    return (
        list(dataset.text_encoder.encode(text, add_special_tokens=False))
        if text
        else []
    )


def _condition_ablation_records(
    record: dict[str, Any], donor: dict[str, Any]
) -> dict[str, dict[str, Any]]:

    from .contracts import SECTION_LABELS

    tags = dict(record)
    tags["tags"] = dict(donor.get("tags") or {})
    tags["language"] = donor.get("language")

    donor_lyrics = [
        str(section.get("lyrics") or "")
        for section in (donor.get("sections") or [])
        if str(section.get("lyrics") or "").strip()
    ]
    lyrics = dict(record)
    lyrics["sections"] = [
        {
            **dict(section),
            "lyrics": (
                donor_lyrics[index % len(donor_lyrics)]
                if donor_lyrics
                else ""
            ),
        }
        for index, section in enumerate(record.get("sections") or [])
    ]

    donor_labels = [
        str(section.get("label") or "")
        for section in (donor.get("sections") or [])
        if str(section.get("label") or "")
    ]
    source_labels = [
        str(section.get("label") or "")
        for section in (record.get("sections") or [])
    ]
    proposed_labels = [
        (
            donor_labels[index % len(donor_labels)]
            if donor_labels
            else source_label
        )
        for index, source_label in enumerate(source_labels)
    ]
    if proposed_labels == source_labels and source_labels:
        rotated = source_labels[1:] + source_labels[:1]
        if rotated != source_labels:
            proposed_labels = rotated
        else:
            proposed_labels = [
                SECTION_LABELS[(SECTION_LABELS.index(label) + 1) % len(SECTION_LABELS)]
                for label in source_labels
            ]
    section_labels = dict(record)
    section_labels["sections"] = [
        {
            **dict(section),
            "label": proposed_labels[index],
        }
        for index, section in enumerate(record.get("sections") or [])
    ]
    return {
        "tags_donor": tags,
        "lyrics_donor": lyrics,
        "section_label_donor": section_labels,
    }


def _semantic_loss_for_item(
    model: torch.nn.Module,
    item: dict[str, Any],
    collator: LlmCollator,
    *,
    device: torch.device,
) -> tuple[float, float]:
    batch = collator([item])
    output = _forward(
        model,
        batch["input_ids"].to(device),
        batch["labels"].to(device),
        batch["attention_mask"].to(device),
        False,
        constraint_kinds=(
            batch["constraint_kinds"].to(device)
            if "constraint_kinds" in batch
            else None
        ),
    )
    return (
        float(output.metrics["loss_semantic"]),
        float(output.metrics["tokens_semantic"]),
    )


def _conditioning_sensitivity(
    model: torch.nn.Module,
    dataset: LlmSampleDataset,
    collator: LlmCollator,
    *,
    registry: TokenRegistry,
    device: torch.device,
    rank: int,
    world_size: int,
    limit: int,
    donors: int,
) -> dict[str, float]:

    from .contracts import SequenceMode
    from .data.dataset import _to_item
    from .data.shards import TokenSpan

    raw_candidates = [
        int(index)
        for index in dataset.record_indices
        if int(dataset.index.melody_frames[int(index)]) > 0
    ]
    raw_records = {
        index: dataset.raw_record(index) for index in raw_candidates
    }
    raw_candidates = [
        index
        for index in raw_candidates
        if not bool(raw_records[index].get("is_instrumental", False))
        and any(
            str(section.get("lyrics") or "").strip()
            for section in (raw_records[index].get("sections") or [])
        )
    ]
    candidates = _evenly_select(raw_candidates, limit)
    if len(candidates) < 2:
        return {}
    records_by_index = {
        index: raw_records[index] for index in candidates
    }

    def group_identity(record: dict[str, Any]) -> str:
        return str(
            record.get("group_id")
            or record.get("canonical_uid")
            or record.get("uid")
            or record.get("sample_id")
        )

    identities = {
        index: group_identity(record)
        for index, record in records_by_index.items()
    }
    requested_donors = min(max(int(donors), 1), len(candidates) - 1)
    donor_indices: dict[int, list[int]] = {}
    for position, index in enumerate(candidates):
        eligible: list[int] = []
        for offset in range(1, len(candidates)):
            candidate = candidates[(position + offset) % len(candidates)]
            if identities[candidate] == identities[index]:
                continue
            eligible.append(candidate)
            if len(eligible) >= requested_donors:
                break
        donor_indices[index] = eligible
    donor_count = min(len(values) for values in donor_indices.values())
    if donor_count <= 0:
        return {}
    local = candidates[rank::world_size]
    rounds = math.ceil(len(candidates) / max(world_size, 1))

    def text_predicate(values: np.ndarray) -> np.ndarray:
        return (values >= 0) & (values < registry.text_vocab_size)

    def melody_predicate(values: np.ndarray) -> np.ndarray:
        return (values >= registry.melody_base) & (
            values < registry.melody_base + registry.melody_size
        )

    output: dict[str, float] = {}

    for mode in (
        SequenceMode.PLAIN,
        SequenceMode.SECTION,
        SequenceMode.UNIQUE_SECTION,
    ):
        gap_values: dict[str, list[float]] = {
            "text_donor": [],
            "tags_donor": [],
            "lyrics_donor": [],
            "section_label_donor": [],
            "melody_donor": [],
        }
        scored_by_kind: Counter[str] = Counter()
        for position in range(rounds):
            active = position < len(local)
            record_index = local[position] if active else candidates[0]
            donor_index = donor_indices[record_index][0]
            record = records_by_index[record_index]
            donor_record = records_by_index[donor_index]
            semantic = dataset.corpus.read_semantic(TokenSpan.from_dict(record["semantic"]))
            melody = dataset.corpus.read_melody(TokenSpan.from_dict(record["melody"]))
            donor_semantic = dataset.corpus.read_semantic(
                TokenSpan.from_dict(donor_record["semantic"])
            )
            donor_melody = dataset.corpus.read_melody(
                TokenSpan.from_dict(donor_record["melody"])
            )
            built = dataset.builder.build(
                record,
                mode=mode,
                semantic=semantic,
                melody=melody,
                sample_index=record_index,
                seed=dataset.seed,
            )
            built_donor = dataset.builder.build(
                donor_record,
                mode=mode,
                semantic=donor_semantic,
                melody=donor_melody,
                sample_index=donor_index,
                seed=dataset.seed,
            )
            matched = _to_item(built, record)
            text_item = dict(matched)
            text_item["input_ids"] = _replace_token_namespace(
                built.input_ids,
                built_donor.input_ids,
                predicate=text_predicate,
            )
            component_items: dict[str, dict[str, Any]] = {}
            for kind, variant_record in _condition_ablation_records(
                record, donor_record
            ).items():
                item = dict(matched)
                item["input_ids"] = _replace_changed_condition_tokens(
                    built.input_ids,
                    _condition_ids_for_record(dataset, variant_record),
                    text_vocab_size=registry.text_vocab_size,
                )
                component_items[kind] = item
            melody_item = dict(matched)
            melody_item["input_ids"] = _replace_token_namespace(
                built.input_ids,
                built_donor.input_ids,
                predicate=melody_predicate,
            )
            matched_loss, _ = _semantic_loss_for_item(
                model, matched, collator, device=device
            )
            text_loss, _ = _semantic_loss_for_item(
                model, text_item, collator, device=device
            )
            component_losses = {
                kind: _semantic_loss_for_item(
                    model, item, collator, device=device
                )[0]
                for kind, item in component_items.items()
            }
            melody_loss, _ = _semantic_loss_for_item(
                model, melody_item, collator, device=device
            )
            if not active:
                continue
            text_changed = not np.array_equal(
                text_item["input_ids"], matched["input_ids"]
            )
            if text_changed:
                gap_values["text_donor"].append(text_loss - matched_loss)
                scored_by_kind["text_donor"] += 1
            for kind, item in component_items.items():
                if np.array_equal(item["input_ids"], matched["input_ids"]):
                    continue
                gap_values[kind].append(component_losses[kind] - matched_loss)
                scored_by_kind[kind] += 1
            melody_changed = not np.array_equal(
                melody_item["input_ids"], matched["input_ids"]
            )
            if mode.has_melody and built.mode is mode and melody_changed:
                gap_values["melody_donor"].append(melody_loss - matched_loss)
                scored_by_kind["melody_donor"] += 1
        for kind, gaps in gap_values.items():
            scored = scored_by_kind[kind]
            gathered = all_gather_objects(gaps)
            values = [
                float(value)
                for rank_values in gathered
                if isinstance(rank_values, list)
                for value in rank_values
            ]
            if not values:
                continue
            prefix = f"{mode.value}/conditioning/{kind}"
            output[f"{prefix}_nll_gap_mean"] = float(np.mean(values))
            output[f"{prefix}_nll_gap_p10"] = float(np.quantile(values, 0.10))
            output[f"{prefix}_positive_rate"] = float(np.mean(np.asarray(values) > 0))
            output[f"{prefix}_samples"] = reduce_scalar_sum(float(scored))

    for mode in (SequenceMode.COVER_SECTION, SequenceMode.COVER_UNIQUE_SECTION):
        all_pair_gaps: list[float] = []
        hard_gaps: list[float] = []
        retrieval: list[float] = []
        semantic_tokens = 0.0
        scored = 0
        attempted_donor_pairs = 0
        equal_length_donor_pairs = 0
        for position in range(rounds):
            active = position < len(local)
            record_index = local[position] if active else candidates[0]
            record = records_by_index[record_index]
            semantic = dataset.corpus.read_semantic(TokenSpan.from_dict(record["semantic"]))
            matched_melody = dataset.corpus.read_melody(
                TokenSpan.from_dict(record["melody"])
            )
            built_matched = dataset.builder.build(
                record,
                mode=mode,
                semantic=semantic,
                melody=matched_melody,
                sample_index=record_index,
                seed=dataset.seed,
            )
            matched_item = _to_item(built_matched, record)
            matched_loss, matched_tokens = _semantic_loss_for_item(
                model,
                matched_item,
                collator,
                device=device,
            )
            donor_losses: list[float] = []
            donor_modes_valid = True
            for donor_index in donor_indices[record_index][:donor_count]:
                donor_record = records_by_index[donor_index]
                donor_semantic = dataset.corpus.read_semantic(
                    TokenSpan.from_dict(donor_record["semantic"])
                )
                donor_melody = dataset.corpus.read_melody(
                    TokenSpan.from_dict(donor_record["melody"])
                )
                built_donor = dataset.builder.build(
                    donor_record,
                    mode=mode,
                    semantic=donor_semantic,
                    melody=donor_melody,
                    sample_index=donor_index,
                    seed=dataset.seed,
                )
                replacement = _replace_token_namespace_equal_length(
                    built_matched.input_ids,
                    built_donor.input_ids,
                    predicate=melody_predicate,
                )
                equal_length = replacement is not None
                if active:
                    attempted_donor_pairs += 1
                    equal_length_donor_pairs += int(equal_length)
                mismatched_item = dict(matched_item)
                mismatched_item["input_ids"] = (
                    replacement
                    if replacement is not None
                    else built_matched.input_ids.copy()
                )
                donor_loss, _ = _semantic_loss_for_item(
                    model,
                    mismatched_item,
                    collator,
                    device=device,
                )
                if equal_length:
                    donor_losses.append(donor_loss)
                donor_modes_valid = donor_modes_valid and built_donor.mode is mode
            if (
                not active
                or built_matched.mode is not mode
                or not donor_losses
                or not donor_modes_valid
            ):
                continue
            gaps = [loss - matched_loss for loss in donor_losses]
            all_pair_gaps.extend(gaps)
            hard_gaps.append(min(gaps))
            retrieval.append(float(all(gap > 0.0 for gap in gaps)))
            semantic_tokens += matched_tokens
            scored += 1
        gathered_pairs = all_gather_objects(all_pair_gaps)
        gathered_hard = all_gather_objects(hard_gaps)
        gathered_retrieval = all_gather_objects(retrieval)
        pair_values = [
            float(value)
            for values in gathered_pairs
            if isinstance(values, list)
            for value in values
        ]
        hard_values = [
            float(value)
            for values in gathered_hard
            if isinstance(values, list)
            for value in values
        ]
        retrieval_values = [
            float(value)
            for values in gathered_retrieval
            if isinstance(values, list)
            for value in values
        ]
        prefix = f"{mode.value}/reference_multi_donor"
        attempted_total = reduce_scalar_sum(float(attempted_donor_pairs))
        equal_total = reduce_scalar_sum(float(equal_length_donor_pairs))
        output[f"{prefix}_equal_length_pair_coverage"] = (
            equal_total / max(attempted_total, 1.0)
        )
        if pair_values:
            output[f"{prefix}_nll_gap_mean"] = float(np.mean(pair_values))
            output[f"{prefix}_nll_gap_p10"] = float(np.quantile(pair_values, 0.10))
            output[f"{prefix}_positive_pair_rate"] = float(
                np.mean(np.asarray(pair_values) > 0)
            )
            output[f"{prefix}_hardest_gap_mean"] = float(np.mean(hard_values))
            output[f"{prefix}_retrieval_top1_rate"] = float(
                np.mean(retrieval_values)
            )
            output[f"{prefix}_samples"] = reduce_scalar_sum(float(scored))
            output[f"{prefix}_donors_per_sample"] = len(pair_values) / max(
                reduce_scalar_sum(float(scored)),
                1.0,
            )
            output[f"{prefix}_same_group_donor_rate"] = 0.0
    return output


def _mismatch_semantic_anchor_ids(
    input_ids: np.ndarray, registry: TokenRegistry
) -> np.ndarray | None:

    values = np.asarray(input_ids, dtype=np.int64)
    music_positions = np.flatnonzero(values == registry.control("music_bos"))
    if music_positions.size != 1:
        return None
    start = int(music_positions[0]) + 1
    cond_bos = registry.control("cond_bos")
    cond_eos = registry.control("cond_eos")
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < values.size:
        if int(values[cursor]) != cond_bos:
            cursor += 1
            continue
        ends = np.flatnonzero(values[cursor + 1 :] == cond_eos)
        if ends.size == 0:
            return None
        end = cursor + 1 + int(ends[0])
        spans.append((cursor + 1, end))
        cursor = end + 1
    if len(spans) < 2:
        return None
    output = values.copy()
    contents = [values[left:right].copy() for left, right in spans]
    for index, (left, right) in enumerate(spans):
        donor = contents[(index + 1) % len(contents)]
        if donor.size <= 0:
            return None
        target_size = right - left
        output[left:right] = np.resize(donor, target_size)
    return output


@torch.no_grad()
def _semantic_anchor_sensitivity(
    model: torch.nn.Module,
    dataset: LlmSampleDataset,
    collator: LlmCollator,
    *,
    registry: TokenRegistry,
    device: torch.device,
    rank: int,
    world_size: int,
    limit: int,
) -> dict[str, float]:

    from .contracts import SequenceMode

    candidates = _evenly_select(
        [int(index) for index in dataset.record_indices], limit
    )
    local = candidates[rank::world_size]
    rounds = math.ceil(len(candidates) / max(world_size, 1)) if candidates else 0
    gaps: list[float] = []
    for position in range(rounds):
        active = position < len(local)
        record_index = local[position] if active else candidates[0]
        item = dataset[(record_index, SequenceMode.SECTION.value, 0)]
        mismatched_ids = _mismatch_semantic_anchor_ids(
            item["input_ids"], registry
        )
        scoreable = (
            item["mode"] == SequenceMode.SECTION.value
            and mismatched_ids is not None
        )
        mismatched = dict(item)
        if mismatched_ids is not None:
            mismatched["input_ids"] = mismatched_ids
        matched_loss, _ = _semantic_loss_for_item(
            model, item, collator, device=device
        )
        mismatched_loss, _ = _semantic_loss_for_item(
            model, mismatched, collator, device=device
        )
        if active and scoreable:
            gaps.append(mismatched_loss - matched_loss)
    gathered = all_gather_objects(gaps)
    values = [
        float(value)
        for payload in gathered
        if isinstance(payload, list)
        for value in payload
    ]
    if not values:
        return {}
    return {
        "section/anchor_mismatch_nll_gap_mean": float(np.mean(values)),
        "section/anchor_mismatch_nll_gap_p10": float(
            np.quantile(values, 0.10)
        ),
        "section/anchor_mismatch_positive_rate": float(
            np.mean(np.asarray(values) > 0.0)
        ),
        "section/anchor_mismatch_samples": float(len(values)),
    }


def _stage4_head_metrics(
    model: torch.nn.Module,
    dataset: LlmSampleDataset,
    collator: LlmCollator,
    probe: Stage4SemanticProbe,
    target_store: Stage4AudioTargetStore | None,
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    limit: int,
    bootstrap_resamples: int = 2000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 0,
) -> dict[str, float]:

    from .contracts import SequenceMode

    candidates = _evenly_select([int(index) for index in dataset.record_indices], limit)
    if not candidates:
        return {}
    local = candidates[rank::world_size]
    rounds = math.ceil(len(candidates) / max(world_size, 1))
    numerators: dict[str, float] = {}
    denominators: dict[str, float] = {}
    local_ctc_deltas: list[tuple[float, float, bool]] = []
    local_samples = 0
    for position in range(rounds):
        active = position < len(local)
        record_index = local[position] if active else candidates[0]
        batch = collator([dataset[(record_index, SequenceMode.PLAIN.value, 0)]])
        llm_output = _forward(
            model,
            batch["input_ids"].to(device),
            batch["labels"].to(device),
            batch["attention_mask"].to(device),
            False,
            constraint_kinds=(
                batch["constraint_kinds"].to(device)
                if "constraint_kinds" in batch
                else None
            ),
            return_semantic_predictions=True,
        )
        predicted = llm_output.semantic_predictions
        if predicted is None:
            raise RuntimeError("Stage 4 probe requested semantic predictions, but the model returned none")
        labels = batch["labels"].to(device)
        reference_ids, hypothesis_ids = _stage4_semantic_local_pair(
            labels,
            predicted,
            dataset.registry,
        )
        if bool((hypothesis_ids < 0).any()):
            raise RuntimeError("Stage 4 semantic predictions contain an unfilled position")
        frame_count = int(reference_ids.numel())
        if frame_count <= 0:
            continue
        token_ids = torch.stack([reference_ids, hypothesis_ids], dim=0)
        frame_mask = torch.ones_like(token_ids, dtype=torch.bool)
        head_outputs = probe(token_ids, frame_mask)
        reference = {name: value[:1] for name, value in head_outputs.items()}
        hypothesis = {name: value[1:] for name, value in head_outputs.items()}
        record = dataset.raw_record(record_index)
        lyric_text = "\n".join(
            str(section.get("lyrics") or "")
            for section in (record.get("sections") or [])
            if str(section.get("lyrics") or "").strip()
        )
        targets = [probe.encode_lyrics(lyric_text)]
        audio_targets = (
            None
            if target_store is None
            else target_store.get(str(record.get("sample_id") or ""))
        )
        mel_targets = (
            None
            if target_store is None
            else [None if audio_targets is None else torch.from_numpy(audio_targets[0])]
        )
        chroma_targets = (
            None
            if target_store is None
            else [None if audio_targets is None else torch.from_numpy(audio_targets[1])]
        )
        pair_num, pair_den = score_stage4_pair_batch(
            reference,
            hypothesis,
            frame_lengths=torch.tensor([frame_count], device=device),
            lyric_targets=targets,
            blank_id=int(getattr(probe.heads, "ctc_blank_id", 0)),
            mel_targets=mel_targets,
            chroma_targets=chroma_targets,
        )
        if not active:
            continue
        reference_key = "ctc/reference/token_error_rate"
        hypothesis_key = "ctc/hypothesis/token_error_rate"
        reference_den = float(pair_den.get(reference_key, 0.0))
        hypothesis_den = float(pair_den.get(hypothesis_key, 0.0))
        if reference_den > 0.0 and hypothesis_den > 0.0:
            reference_rate = float(pair_num[reference_key]) / reference_den
            hypothesis_rate = float(pair_num[hypothesis_key]) / hypothesis_den
            feasible_den = float(
                pair_den.get("ctc/target_feasible_rate", 0.0)
            )
            feasible = (
                feasible_den > 0.0
                and float(pair_num.get("ctc/target_feasible_rate", 0.0))
                / feasible_den
                >= 0.5
            )
            local_ctc_deltas.append(
                (
                    hypothesis_rate - reference_rate,
                    min(reference_den, hypothesis_den),
                    feasible,
                )
            )
        for name, value in pair_num.items():
            numerators[name] = numerators.get(name, 0.0) + value
        for name, value in pair_den.items():
            denominators[name] = denominators.get(name, 0.0) + value
        local_samples += 1
    reduced = reduce_weighted_metrics(numerators, denominators)
    output = {f"stage4/{name}": value for name, value in reduced.items()}
    for suffix in (
        "token_error_rate",
        "substitution_rate",
        "deletion_rate",
        "insertion_rate",
        "nll_per_target_token",
        "blank_ratio",
        "posterior_entropy",
    ):
        reference_key = f"stage4/ctc/reference/{suffix}"
        hypothesis_key = f"stage4/ctc/hypothesis/{suffix}"
        if reference_key in output and hypothesis_key in output:
            output[f"stage4/ctc/delta/{suffix}"] = (
                output[hypothesis_key] - output[reference_key]
            )
    for family, suffixes in (
        ("mel", ("l1", "spectral_convergence", "frame_cosine")),
        (
            "chroma",
            (
                "l1",
                "spectral_convergence",
                "frame_cosine",
                "dominant_pitch_class_accuracy",
            ),
        ),
    ):
        for suffix in suffixes:
            reference_key = f"stage4/{family}/reference_to_audio/{suffix}"
            hypothesis_key = f"stage4/{family}/hypothesis_to_audio/{suffix}"
            if reference_key in output and hypothesis_key in output:
                output[f"stage4/{family}/delta_to_audio/{suffix}"] = (
                    output[hypothesis_key] - output[reference_key]
                )
    output["stage4/samples"] = reduce_scalar_sum(float(local_samples))
    gathered_deltas = all_gather_objects(local_ctc_deltas)
    paired = [
        (float(delta), float(weight), bool(feasible))
        for payload in gathered_deltas
        if isinstance(payload, list)
        for delta, weight, feasible in payload
    ]
    feasible_paired = [
        (delta, weight) for delta, weight, feasible in paired if feasible
    ]
    output["stage4/ctc/delta/paired_samples"] = float(len(paired))
    output["stage4/ctc/delta/paired_feasible_samples"] = float(
        len(feasible_paired)
    )
    if feasible_paired:
        bootstrap = paired_bootstrap_mean_ci(
            [value for value, _ in feasible_paired],
            weights=[weight for _, weight in feasible_paired],
            confidence=bootstrap_confidence,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        for name, value in bootstrap.items():
            output[
                f"stage4/ctc/delta/token_error_rate_paired_feasible_{name}"
            ] = value
    output["stage4/reference_target_is_frozen_head_output"] = 1.0
    output["stage4/ctc_vocab_available"] = float(probe.ctc_tokenizer is not None)
    output["stage4/audio_target_store_configured"] = float(target_store is not None)
    return output


@torch.no_grad()
def _duration_aligned_lyrics(
    record: dict[str, Any],
    duration_sec: float,
) -> tuple[str, int]:

    return duration_aligned_lyrics(record, duration_sec)


def _mode_adherence_metrics(
    raw_token_ids: list[np.ndarray],
    *,
    mode,
    registry: TokenRegistry,
) -> dict[str, float]:

    melody_bos = registry.control("melody_bos")
    selected_melody_plan = [
        bool(
            np.asarray(raw).size
            and int(np.asarray(raw).reshape(-1)[0]) == melody_bos
        )
        for raw in raw_token_ids
    ]
    return {
        "generated_melody_plan_rate": float(
            np.mean(selected_melody_plan) if selected_melody_plan else 0.0
        ),
        "mode_adherence_rate": float(
            np.mean(
                [
                    selected if mode.has_melody else not selected
                    for selected in selected_melody_plan
                ]
            )
            if selected_melody_plan
            else 0.0
        ),
    }


def _load_fixed_generation_prompts(
    generation_section: dict[str, Any],
    *,
    mode: Any,
    limit: int,
) -> tuple[list[dict[str, Any]], int] | None:

    raw_path = generation_section.get("prompts")
    if not raw_path:
        return None
    path = Path(str(raw_path)).resolve(strict=True)
    content = path.read_bytes()
    expected_sha256 = str(generation_section.get("prompts_sha256") or "")
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Free-generation prompt-suite SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )
    records = [
        json.loads(line)
        for line in content.decode("utf-8").splitlines()
        if line.strip()
    ]
    sample_ids = [str(record.get("sample_id") or "") for record in records]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("Every fixed free-generation record must have sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Fixed free-generation suite has duplicate sample_id values")
    mode_records = [
        record
        for record in records
        if str(record.get("generation_mode") or mode.value) == mode.value
    ]
    if not mode_records:
        raise ValueError(f"Fixed free-generation suite does not contain mode={mode.value}")
    selected = _evenly_select(mode_records, min(int(limit), len(mode_records)))
    return [dict(record) for record in selected], len(mode_records)


@torch.no_grad()
def _stage4_free_generation_metrics(
    model: MusicLLM,
    dataset: LlmSampleDataset,
    probe: Stage4SemanticProbe,
    *,
    device: torch.device,
    limit: int,
    generation_section: dict[str, Any],
    step: int,
) -> dict[str, float]:

    from .contracts import SequenceMode
    from .data.shards import TokenSpan
    from .generate import GenerationConfig, generate_semantic, parse_generated_tokens

    target = max(1, int(limit))
    mode = SequenceMode(str(generation_section.get("mode", SequenceMode.SECTION.value)))
    if mode.is_cover:
        raise ValueError("Training-time free-generation evaluation does not support cover mode; provide a reference explicitly")
    target_duration_sec = (
        int(generation_section.get("max_semantic_frames", 512))
        / SEMANTIC_FRAME_RATE
    )
    fixed_prompts = _load_fixed_generation_prompts(
        generation_section,
        mode=mode,
        limit=target,
    )
    prompt_pool_size = 0
    if fixed_prompts is not None:
        records, prompt_pool_size = fixed_prompts
    else:
        candidate_indices = _evenly_select(
            [int(index) for index in dataset.record_indices],
            min(len(dataset.record_indices), target * 64),
        )
        records = []
        for index in candidate_indices:
            record = dataset.raw_record(index)
            if bool(record.get("is_instrumental", False)) or not any(
                str(section.get("lyrics") or "").strip()
                for section in (record.get("sections") or [])
            ):
                continue


            aligned_text, _ = _duration_aligned_lyrics(record, target_duration_sec)
            if not aligned_text:
                continue
            records.append(record)
            if len(records) >= target:
                break
    if not records:
        return {}
    prompts = [
        dataset.builder.build_prompt(record, mode=mode)
        for record in records
    ]
    config = GenerationConfig(
        max_new_tokens=int(generation_section.get("max_new_tokens", 1024)),
        temperature=float(generation_section.get("temperature", 0.0)),
        top_p=float(generation_section.get("top_p", 1.0)),
        top_k=int(generation_section.get("top_k", 0)),
        min_semantic_frames=int(
            generation_section.get("min_semantic_frames", 25)
        ),
        max_semantic_frames=int(
            generation_section.get("max_semantic_frames", 512)
        ),
        max_melody_tokens=int(
            generation_section.get("max_melody_tokens", 256)
        ),
        max_melody_segments=int(
            generation_section.get("max_melody_segments", 16)
        ),
        min_melody_segments=1,
        semantic_repetition_penalty=float(
            generation_section.get("semantic_repetition_penalty", 0.0)
        ),
        semantic_repetition_window=int(
            generation_section.get("semantic_repetition_window", 64)
        ),
        plain_semantic_repetition_penalty=float(
            generation_section.get("plain_semantic_repetition_penalty", 0.0)
        ),
        melody_repetition_penalty=float(
            generation_section.get("melody_repetition_penalty", 0.0)
        ),
        melody_repetition_window=int(
            generation_section.get("melody_repetition_window", 32)
        ),
        melody_unvoiced_run_threshold=int(
            generation_section.get("melody_unvoiced_run_threshold", 0)
        ),
        melody_unvoiced_run_penalty=float(
            generation_section.get("melody_unvoiced_run_penalty", 0.0)
        ),
        plain_semantic_eos_bias=float(
            generation_section.get("plain_semantic_eos_bias", 0.0)
        ),
        plain_semantic_eos_bias_start_frames=int(
            generation_section.get("plain_semantic_eos_bias_start_frames", 0)
        ),
        plain_semantic_eos_bias_interval_frames=int(
            generation_section.get("plain_semantic_eos_bias_interval_frames", 625)
        ),
        seed=int(generation_section.get("seed", 0)),
        strict_mode=bool(generation_section.get("strict_mode", True)),
    )
    anchor_boundary = str(generation_section.get("anchor_boundary", "generated"))
    anchor_schedules = None
    anchor_pools = None
    if dataset.builder.config.semantic_section_reanchor:
        if anchor_boundary == "oracle":
            anchor_schedules = [
                dataset.builder.oracle_section_anchor_schedule(record)
                for record in records
            ]
        elif anchor_boundary == "generated":
            anchor_pools = [
                dataset.builder.section_anchor_pools(record) for record in records
            ]
        elif anchor_boundary != "none":
            raise ValueError(
                "evaluation.generation.anchor_boundary must be "
                "generated/oracle/none"
            )


    fsdp2_root = hasattr(model, "unshard") and hasattr(model, "reshard")
    if fsdp2_root:
        model.unshard(async_op=False)  # type: ignore[attr-defined]
    try:
        generated = generate_semantic(
            model,
            prompts,
            mode=mode,
            config=config,
            tokenizer_revision=dataset.corpus.tokenizer_revision,
            device=device,
            semantic_anchor_schedules=anchor_schedules,
            semantic_anchor_pools=anchor_pools,
        )
    finally:
        if fsdp2_root:
            model.reshard()  # type: ignore[attr-defined]
    numerators: dict[str, float] = {}
    denominators: dict[str, float] = {}
    aligned_numerators: dict[str, float] = {}
    aligned_denominators: dict[str, float] = {}
    aligned_target_samples = 0
    aligned_target_sections = 0
    fixed_numerators: dict[str, float] = {}
    fixed_denominators: dict[str, float] = {}
    fixed_target_samples = 0
    fixed_target_sections = 0
    scored_sequences: list[np.ndarray] = []
    reference_matched_hypotheses: list[np.ndarray] = []
    reference_sequences: list[np.ndarray] = []
    generated_melody_matched: list[np.ndarray] = []
    reference_melody_sequences: list[np.ndarray] = []
    for record_index, (record, semantic, generated_melody) in enumerate(
        zip(
            records,
            generated.semantic_ids,
            generated.generated_melody_ids,
        )
    ):
        values = np.asarray(semantic, dtype=np.int64).reshape(-1)
        if values.size <= 0:
            continue
        scored_sequences.append(values)
        semantic_span = record.get("semantic")
        if isinstance(semantic_span, dict):
            reference = dataset.corpus.read_semantic(
                TokenSpan.from_dict(semantic_span)
            )
            if int(reference.size) >= int(values.size):
                reference_matched_hypotheses.append(values)
                reference_sequences.append(
                    np.asarray(reference[: values.size], dtype=np.int64).reshape(-1)
                )
            melody_span = record.get("melody")
            if (
                mode.has_melody
                and isinstance(melody_span, dict)
                and generated_melody is not None
            ):
                reference_melody = dataset.corpus.read_melody(
                    TokenSpan.from_dict(melody_span)
                )
                built_reference = dataset.builder.build(
                    record,
                    mode=mode,
                    semantic=reference,
                    melody=reference_melody,
                    epoch=0,
                    sample_index=record_index,
                    seed=int(dataset.seed or 0),
                )
                parsed_reference = parse_generated_tokens(
                    np.asarray(built_reference.input_ids),
                    dataset.registry,
                )
                effective_reference = np.asarray(
                    parsed_reference["melody_ids"],
                    dtype=np.int64,
                )
                hypothesis_melody = np.asarray(
                    generated_melody,
                    dtype=np.int64,
                )
                matched_length = min(
                    int(effective_reference.size),
                    int(hypothesis_melody.size),
                )
                if matched_length > 0:
                    reference_melody_sequences.append(
                        effective_reference[:matched_length]
                    )
                    generated_melody_matched.append(
                        hypothesis_melody[:matched_length]
                    )
        token_ids = torch.as_tensor(values, dtype=torch.long, device=device)[None]
        frame_mask = torch.ones_like(token_ids, dtype=torch.bool)
        predictions = probe(token_ids, frame_mask)
        lyric_text = "\n".join(
            str(section.get("lyrics") or "")
            for section in (record.get("sections") or [])
            if str(section.get("lyrics") or "").strip()
        )
        if not lyric_text:
            continue
        pair_num, pair_den = score_stage4_pair_batch(
            predictions,
            predictions,
            frame_lengths=torch.tensor([values.size], device=device),
            lyric_targets=[probe.encode_lyrics(lyric_text)],
            blank_id=int(getattr(probe.heads, "ctc_blank_id", 0)),
        )
        for name, value in pair_num.items():
            if name.startswith("ctc/hypothesis/") or name.startswith(
                "ctc/target_"
            ):
                numerators[name] = numerators.get(name, 0.0) + value
        for name, value in pair_den.items():
            if name in numerators or name.startswith("ctc/target_"):
                denominators[name] = denominators.get(name, 0.0) + value
        aligned_text, aligned_sections = _duration_aligned_lyrics(
            record,
            values.size / SEMANTIC_FRAME_RATE,
        )
        aligned_target = probe.encode_lyrics(aligned_text)
        if aligned_target:
            aligned_target_samples += 1
            aligned_target_sections += aligned_sections
            aligned_num, aligned_den = score_stage4_pair_batch(
                predictions,
                predictions,
                frame_lengths=torch.tensor([values.size], device=device),
                lyric_targets=[aligned_target],
                blank_id=int(getattr(probe.heads, "ctc_blank_id", 0)),
            )
            for name, value in aligned_num.items():
                if name.startswith("ctc/hypothesis/") or name.startswith(
                    "ctc/target_"
                ):
                    aligned_numerators[name] = (
                        aligned_numerators.get(name, 0.0) + value
                    )
            for name, value in aligned_den.items():
                if name in aligned_numerators or name.startswith("ctc/target_"):
                    aligned_denominators[name] = (
                        aligned_denominators.get(name, 0.0) + value
                    )
        fixed_text, fixed_sections = _duration_aligned_lyrics(
            record,
            config.max_semantic_frames / SEMANTIC_FRAME_RATE,
        )
        fixed_target = probe.encode_lyrics(fixed_text)
        if fixed_target:
            fixed_target_samples += 1
            fixed_target_sections += fixed_sections
            fixed_num, fixed_den = score_stage4_pair_batch(
                predictions,
                predictions,
                frame_lengths=torch.tensor([values.size], device=device),
                lyric_targets=[fixed_target],
                blank_id=int(getattr(probe.heads, "ctc_blank_id", 0)),
            )
            for name, value in fixed_num.items():
                if name.startswith("ctc/hypothesis/") or name.startswith(
                    "ctc/target_"
                ):
                    fixed_numerators[name] = fixed_numerators.get(name, 0.0) + value
            for name, value in fixed_den.items():
                if name in fixed_numerators or name.startswith("ctc/target_"):
                    fixed_denominators[name] = fixed_denominators.get(name, 0.0) + value
    reduced = {
        name: numerators[name] / denominators[name]
        for name in sorted(numerators)
        if denominators.get(name, 0.0) > 0.0
    }

    output: dict[str, float] = {}
    output.update(
        {
            "stage4_free/config/natural_unlocked": 1.0,
            "stage4_free/config/min_semantic_frames": float(
                config.min_semantic_frames
            ),
            "stage4_free/config/max_semantic_frames": float(
                config.max_semantic_frames
            ),
            "stage4_free/config/max_melody_tokens": float(
                config.max_melody_tokens
            ),
            "stage4_free/config/max_new_tokens": float(config.max_new_tokens),
            "stage4_free/config/fixed_prompt_suite": float(
                fixed_prompts is not None
            ),
            "stage4_free/config/prompt_pool_size": float(prompt_pool_size),
            "stage4_free/config/selected_prompts": float(len(records)),
        }
    )
    length_metrics = generation_length_metrics(
        generated.semantic_ids,
        generated.generated_melody_ids,
        generated.stop_reasons,
        max_semantic_frames=config.max_semantic_frames,
        max_melody_tokens=config.max_melody_tokens,
    )
    output.update(
        {f"stage4_free/{name}": value for name, value in length_metrics.items()}
    )
    output["stage4_free/natural_eos_rate"] = length_metrics[
        "termination/fully_natural_eos_rate"
    ]
    output["stage4_free/cap_rate"] = length_metrics["termination/any_cap_rate"]
    output["stage4_free/max_new_tokens_rate"] = length_metrics[
        "termination/max_new_tokens_rate"
    ]
    output["stage4_free/unfinished_rate"] = length_metrics[
        "termination/unfinished_rate"
    ]
    output.update(
        {
            f"stage4_free/ctc_full_target/{name.removeprefix('ctc/')}": value
            for name, value in reduced.items()
            if name.startswith("ctc/")
        }
    )
    aligned_reduced = {
        name: aligned_numerators[name] / aligned_denominators[name]
        for name in sorted(aligned_numerators)
        if aligned_denominators.get(name, 0.0) > 0.0
    }
    output.update(
        {
            f"stage4_free/ctc_aligned/{name.removeprefix('ctc/')}": value
            for name, value in aligned_reduced.items()
            if name.startswith("ctc/")
        }
    )
    output["stage4_free/ctc_aligned/target_samples"] = float(
        aligned_target_samples
    )
    output["stage4_free/ctc_aligned/target_sections"] = float(
        aligned_target_sections
    )
    fixed_reduced = {
        name: fixed_numerators[name] / fixed_denominators[name]
        for name in sorted(fixed_numerators)
        if fixed_denominators.get(name, 0.0) > 0.0
    }
    output.update(
        {
            f"stage4_free/ctc_fixed_horizon/{name.removeprefix('ctc/')}": value
            for name, value in fixed_reduced.items()
            if name.startswith("ctc/")
        }
    )
    output["stage4_free/ctc_fixed_horizon/target_samples"] = float(
        fixed_target_samples
    )
    output["stage4_free/ctc_fixed_horizon/target_sections"] = float(
        fixed_target_sections
    )
    output["stage4_free/ctc_fixed_horizon/horizon_frames"] = float(
        config.max_semantic_frames
    )
    distribution = semantic_distribution_metrics(scored_sequences)
    output.update(
        {f"stage4_free/semantic/{name}": value for name, value in distribution.items()}
    )
    semantic_per_sample = per_sample_distribution_metrics(
        scored_sequences,
        codebook_size=dataset.registry.semantic_size,
        repeated_4gram_threshold=float(
            generation_section.get(
                "semantic_catastrophic_repeated_4gram_threshold", 0.10
            )
        ),
        constant_run_threshold=int(
            generation_section.get(
                "semantic_catastrophic_constant_run_threshold", 1024
            )
        ),
        max_code_fraction_threshold=float(
            generation_section.get(
                "semantic_catastrophic_max_code_fraction_threshold", 0.05
            )
        ),
    )
    output.update(
        {
            f"stage4_free/semantic_per_sample/{name}": value
            for name, value in semantic_per_sample.items()
        }
    )
    if semantic_per_sample.get("samples", 0.0) > 0.0:
        output["stage4_free/selection/semantic_risk"] = max(
            semantic_per_sample["repeated_4gram_fraction_p95"]
            / max(
                semantic_per_sample[
                    "threshold/repeated_4gram_fraction"
                ],
                1e-12,
            ),
            semantic_per_sample["longest_constant_run_p95"]
            / max(
                semantic_per_sample["threshold/longest_constant_run"],
                1e-12,
            ),
            semantic_per_sample["max_code_fraction_p95"]
            / max(
                semantic_per_sample["threshold/max_code_fraction"],
                1e-12,
            ),
        )
    reference_distribution = semantic_distribution_metrics(reference_sequences)
    matched_distribution = semantic_distribution_metrics(
        reference_matched_hypotheses
    )
    semantic_distances = token_distribution_distances(
        reference_matched_hypotheses,
        reference_sequences,
        codebook_size=dataset.registry.semantic_size,
        ordinal_ids=False,
    )
    output.update(
        {
            f"stage4_free/reference_semantic/{name}": value
            for name, value in reference_distribution.items()
        }
    )
    output.update(
        {
            f"stage4_free/reference_distance/semantic_{name}": value
            for name, value in semantic_distances.items()
        }
    )
    for name in (
        "codebook_coverage",
        "effective_codebook_size",
        "max_code_fraction",
        "local_lag_match_fraction",
        "repeated_4gram_fraction",
        "longest_constant_run",
    ):
        hypothesis_value = matched_distribution.get(name)
        reference_value = reference_distribution.get(name)
        if (
            hypothesis_value is not None
            and reference_value is not None
            and float(reference_value) > 0.0
        ):
            output[f"stage4_free/reference_ratio/{name}"] = float(
                hypothesis_value
            ) / float(reference_value)
    output["stage4_free/reference_ratio/matched_samples"] = float(
        len(reference_sequences)
    )
    output["stage4_free/reference_ratio/coverage"] = float(
        len(reference_sequences) / max(len(scored_sequences), 1)
    )
    output["stage4_free/requested_samples"] = float(len(records))
    output["stage4_free/scored_samples"] = float(len(scored_sequences))
    output["stage4_free/mean_semantic_frames"] = float(
        np.mean([values.size for values in scored_sequences])
        if scored_sequences
        else 0.0
    )
    output["stage4_free/completion_rate"] = float(
        np.mean(generated.finished) if generated.finished else 0.0
    )

    output["stage4_free/forced_stop_rate"] = length_metrics[
        "termination/any_cap_rate"
    ]
    no_melody_plan_rate = float(
        np.mean([int(not sections) for sections in generated.melody_sections])
    )
    output["stage4_free/no_melody_plan_rate"] = no_melody_plan_rate

    output["stage4_free/empty_plan_rate"] = (
        no_melody_plan_rate if mode.has_melody else 0.0
    )
    output.update(
        {
            f"stage4_free/{name}": value
            for name, value in _mode_adherence_metrics(
                generated.raw_token_ids,
                mode=mode,
                registry=dataset.registry,
            ).items()
        }
    )
    if mode.has_melody:
        expected_sections = [
            [
                str(section.get("label") or "")
                for section in (record.get("sections") or [])
                if str(section.get("lyrics") or "").strip()
            ]
            for record in records
        ]
        section_metrics = section_sequence_metrics(
            expected_sections,
            generated.melody_sections,
        )
        output.update(
            {
                f"stage4_free/section/{name}": value
                for name, value in section_metrics.items()
            }
        )
    output["stage4_free/requested_mode_plain"] = float(
        mode is SequenceMode.PLAIN
    )
    output["stage4_free/requested_mode_section"] = float(
        mode is SequenceMode.SECTION
    )
    output["stage4_free/strict_mode"] = float(config.strict_mode)
    output["stage4_free/semantic_anchor_tokens"] = float(
        sum(generated.semantic_anchor_tokens)
    )
    melody_lengths = [
        0 if values is None else int(np.asarray(values).size)
        for values in generated.generated_melody_ids
    ]
    melody_nonempty = [
        np.asarray(values, dtype=np.int64)
        for values in generated.generated_melody_ids
        if values is not None and np.asarray(values).size
    ]
    melody_flat = (
        np.concatenate(melody_nonempty)
        if melody_nonempty
        else np.zeros(0, dtype=np.int64)
    )
    melody_distribution = semantic_distribution_metrics(
        melody_nonempty,
        codebook_size=dataset.registry.melody_size,
    )
    melody_per_sample = per_sample_distribution_metrics(
        melody_nonempty,
        codebook_size=dataset.registry.melody_size,
        repeated_4gram_threshold=float(
            generation_section.get(
                "melody_catastrophic_repeated_4gram_threshold", 0.65
            )
        ),
        constant_run_threshold=int(
            generation_section.get(
                "melody_catastrophic_constant_run_threshold", 128
            )
        ),
        max_code_fraction_threshold=float(
            generation_section.get(
                "melody_catastrophic_max_code_fraction_threshold", 0.55
            )
        ),
    )
    melody_distribution = {
        (
            "melody_tokens"
            if name == "semantic_tokens"
            else "unique_melody_codes"
            if name == "unique_semantic_codes"
            else name
        ): value
        for name, value in melody_distribution.items()
    }
    output.update(
        {
            f"stage4_free/melody/{name}": value
            for name, value in melody_distribution.items()
        }
    )
    output.update(
        {
            f"stage4_free/melody_per_sample/{name}": value
            for name, value in melody_per_sample.items()
        }
    )
    melody_distances = token_distribution_distances(
        generated_melody_matched,
        reference_melody_sequences,
        codebook_size=dataset.registry.melody_size,
        ordinal_ids=True,
    )
    output.update(
        {
            f"stage4_free/reference_distance/melody_{name}": value
            for name, value in melody_distances.items()
        }
    )
    if melody_per_sample.get("samples", 0.0) > 0.0:
        output["stage4_free/selection/melody_risk"] = max(
            melody_per_sample["repeated_4gram_fraction_p95"]
            / max(
                melody_per_sample["threshold/repeated_4gram_fraction"],
                1e-12,
            ),
            melody_per_sample["longest_constant_run_p95"]
            / max(
                melody_per_sample["threshold/longest_constant_run"],
                1e-12,
            ),
            melody_per_sample["max_code_fraction_p95"]
            / max(
                melody_per_sample["threshold/max_code_fraction"],
                1e-12,
            ),
        )
    semantic_sequence_keys = {
        np.asarray(values, dtype=np.int64).tobytes()
        for values in generated.semantic_ids
    }
    semantic_prefix_keys = {
        np.asarray(values, dtype=np.int64)[:32].tobytes()
        for values in generated.semantic_ids
    }
    melody_sequence_keys = {
        np.asarray(values, dtype=np.int64).tobytes()
        for values in generated.generated_melody_ids
        if values is not None
    }
    melody_prefix_keys = {
        np.asarray(values, dtype=np.int64)[:16].tobytes()
        for values in generated.generated_melody_ids
        if values is not None
    }
    output.update(
        {
            "stage4_free/cross_sample/unique_semantic_sequence_rate": float(
                len(semantic_sequence_keys) / max(len(generated.semantic_ids), 1)
            ),
            "stage4_free/cross_sample/unique_semantic_prefix32_rate": float(
                len(semantic_prefix_keys) / max(len(generated.semantic_ids), 1)
            ),
            "stage4_free/cross_sample/unique_melody_sequence_rate": float(
                len(melody_sequence_keys)
                / max(len(generated.generated_melody_ids), 1)
            ),
            "stage4_free/cross_sample/unique_melody_prefix16_rate": float(
                len(melody_prefix_keys)
                / max(len(generated.generated_melody_ids), 1)
            ),
        }
    )
    output["stage4_free/melody_tokens"] = float(sum(melody_lengths))
    output["stage4_free/mean_melody_tokens"] = float(np.mean(melody_lengths))
    output["stage4_free/mean_melody_sections"] = float(
        np.mean([len(values) for values in generated.melody_sections])
    )
    output["stage4_free/unique_melody_tokens"] = float(
        np.unique(melody_flat).size
    )
    output["stage4_free/melody_unvoiced_rate"] = float(
        np.mean(melody_flat == 255) if melody_flat.size else 0.0
    )
    output["stage4_free/anchor_boundary_generated"] = float(
        anchor_boundary == "generated"
    )
    output["stage4_free/anchor_boundary_oracle"] = float(
        anchor_boundary == "oracle"
    )
    artifact = generation_section.get("output")
    if artifact and is_main_process():
        destination = Path(str(artifact).format(step=int(step))).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        semantic_lengths = np.asarray(
            [np.asarray(values).size for values in generated.semantic_ids],
            dtype=np.int64,
        )
        raw_lengths = np.asarray(
            [np.asarray(values).size for values in generated.raw_token_ids],
            dtype=np.int64,
        )
        stop_reasons = np.asarray(generated.stop_reasons, dtype=object)
        temporary = destination.with_name(
            f".{destination.stem}.tmp-{os.getpid()}.npz"
        )
        np.savez_compressed(
            temporary,
            sample_ids=np.asarray(
                [str(record.get("sample_id") or "") for record in records],
                dtype=object,
            ),
            semantic_lengths=semantic_lengths,
            semantic_ids=np.concatenate(generated.semantic_ids),
            melody_lengths=np.asarray(melody_lengths, dtype=np.int64),
            melody_ids=melody_flat,
            finished=np.asarray(generated.finished, dtype=bool),
            raw_lengths=raw_lengths,
            raw_token_ids=np.concatenate(generated.raw_token_ids),
            stop_reasons=stop_reasons,
        )
        os.replace(temporary, destination)
        destination.with_suffix(".summary.json").write_text(
            json.dumps(
                {
                    "output": str(destination),
                    "num_samples": len(records),
                    "mode": mode.value,
                    "tokenizer_revision": dataset.corpus.tokenizer_revision,
                    "generation_config": config.to_dict(),
                    "ctc_target_protocol": "section-time-proportional-prefix-v1",
                    "sections": generated.melody_sections,
                    "stop_reasons": stop_reasons.tolist(),
                    "stop_reason_counts": dict(
                        sorted(Counter(stop_reasons.tolist()).items())
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        output["stage4_free/output_saved"] = 1.0
    return output


def _stage4_semantic_local_pair(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    registry: TokenRegistry,
) -> tuple[torch.Tensor, torch.Tensor]:

    if labels.shape != predictions.shape:
        raise ValueError(
            f"Stage4 label and prediction shapes differ: {labels.shape}/{predictions.shape}"
        )
    semantic_mask = (labels >= registry.semantic_base) & (
        labels < registry.semantic_base + registry.semantic_size
    )
    reference = labels[semantic_mask] - registry.semantic_base
    hypothesis = predictions[semantic_mask] - registry.semantic_base
    if bool((hypothesis < 0).any()) or bool(
        (hypothesis >= registry.semantic_size).any()
    ):
        raise RuntimeError(
            "Stage4 teacher-forced prediction falls outside the semantic namespace. "
            "Use constrained semantic argmax and subtract semantic_base when converting "
            "global IDs to local IDs."
        )
    return reference, hypothesis


class _LoadedModelEvaluator:

    def __init__(
        self,
        config: dict[str, Any],
        registry: TokenRegistry,
        text_encoder: Any,
        *,
        split: str,
        max_batches: int,
        rank: int,
        world_size: int,
        device: torch.device,
    ) -> None:
        from .contracts import SequenceMode
        from .data.sampler import uniform_eval_batches

        self.config = config
        self.registry = registry
        validate_text_encoder_namespace(text_encoder, registry)
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.split = str(split)
        self.max_batches = int(max_batches)
        self.completed_runs = 0
        self.eval_section = dict(config.get("evaluation", {}) or {})
        self.sequence_config = SequenceConfig.from_config(config)
        self.condition_config = ConditionRenderConfig.from_config(config)
        self.grammar_config = GrammarConfig.from_config(config)
        self.sequence_config.random_crop = False
        self.sequence_config.resample_unique_sections = False
        data_section = dict(config.get("data", {}) or {})
        self.dataset = LlmSampleDataset(
            data_section["corpus"],
            registry,
            text_encoder,
            sequence_config=self.sequence_config,
            split=split,
            seed=int((config.get("train", {}) or {}).get("seed", 0)),
            strict_metadata=bool(data_section.get("require_production_contract", False)),
            local_cache_dir=data_section.get("local_cache_dir"),
            condition_config=self.condition_config,
            grammar_config=self.grammar_config,
        )
        sample_ids_path = self.eval_section.get("sample_ids_path")
        if sample_ids_path:
            allowlist_path = Path(str(sample_ids_path)).resolve(strict=True)
            content = allowlist_path.read_bytes()
            expected_sha256 = str(
                self.eval_section.get("sample_ids_sha256") or ""
            )
            actual_sha256 = hashlib.sha256(content).hexdigest()
            if expected_sha256 and actual_sha256 != expected_sha256:
                raise RuntimeError(
                    "Evaluation sample-allowlist SHA-256 mismatch: "
                    f"{actual_sha256} != {expected_sha256}"
                )
            allowed = {
                str(json.loads(line).get("sample_id") or "")
                for line in content.decode("utf-8").splitlines()
                if line.strip()
            }
            if "" in allowed:
                raise ValueError("Evaluation sample allowlist contains a blank sample_id")
            selected_indices = [
                int(index)
                for index in self.dataset.record_indices
                if str(self.dataset.raw_record(int(index)).get("sample_id") or "")
                in allowed
            ]
            selected_ids = {
                str(self.dataset.raw_record(index).get("sample_id") or "")
                for index in selected_indices
            }
            missing = allowed - selected_ids
            if missing:
                raise ValueError(
                    f"Evaluation allowlist has {len(missing)} entries absent from corpus split={split}"
                )
            self.dataset.record_indices = np.asarray(
                selected_indices,
                dtype=np.int64,
            )
            self.eval_sample_ids_revision = f"sha256:{actual_sha256}"
        else:
            self.eval_sample_ids_revision = "all-split-records"
        train_section = dict(config.get("train", {}) or {})
        self.collator = LlmCollator(
            pad_token_id=registry.pad_id,
            pad_to_multiple_of=int(train_section.get("pad_to_multiple_of", 64)),
            max_sequence_length=self.sequence_config.max_sequence_length,
        )
        self.modes = (
            SequenceMode.PLAIN,
            SequenceMode.SECTION,
            SequenceMode.UNIQUE_SECTION,
            SequenceMode.COVER_SECTION,
            SequenceMode.COVER_UNIQUE_SECTION,
        )
        self.all_batches_by_mode: dict[Any, list[list[Any]]] = {}
        self.batches_by_mode: dict[Any, list[list[Any]]] = {}
        for mode in self.modes:
            batches = uniform_eval_batches(
                self.dataset.index,
                [int(i) for i in self.dataset.record_indices],
                mode=mode,
                max_tokens=int(train_section.get("max_tokens_per_gpu", 8192)),
                max_sequence_length=self.sequence_config.max_sequence_length,
                max_semantic_frames=self.sequence_config.max_semantic_frames,
                condition_token_estimate=int(
                    (config.get("sampler") or {}).get(
                        "condition_token_estimate",
                        self.sequence_config.max_condition_tokens,
                    )
                ),
                max_condition_tokens=self.sequence_config.max_condition_tokens,
                pad_to_multiple_of=int(train_section.get("pad_to_multiple_of", 64)),
                max_batch_size=int(train_section.get("max_batch_size", 32)),
                semantic_section_reanchor=(
                    self.sequence_config.semantic_section_reanchor
                ),
                semantic_anchor_max_text_tokens=(
                    self.sequence_config.semantic_anchor_max_text_tokens
                ),
                semantic_anchor_length_fn=(
                    self.dataset.semantic_anchor_token_bound
                ),
            )
            self.all_batches_by_mode[mode] = list(batches)
            self.batches_by_mode[mode] = _evenly_select(batches, max_batches)
        eval_plan_payload = json.dumps(
            {
                mode.value: [
                    [list(item) for item in batch]
                    for batch in self.batches_by_mode[mode]
                ]
                for mode in self.modes
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self.eval_plan_revision = (
            "sha256:"
            + hashlib.sha256(eval_plan_payload.encode()).hexdigest()
        )
        full_plan_payload = json.dumps(
            {
                mode.value: [
                    [list(item) for item in batch]
                    for batch in self.all_batches_by_mode[mode]
                ]
                for mode in self.modes
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self.full_eval_plan_revision = (
            "sha256:"
            + hashlib.sha256(full_plan_payload.encode()).hexdigest()
        )

        frequency_path = self.eval_section.get("semantic_frequency_counts")
        self.frequency_counts_path = (
            str(Path(frequency_path).resolve()) if frequency_path else None
        )
        self.frequency_buckets: torch.Tensor | None = None
        if frequency_path:
            counts = load_semantic_frequency_counts(
                frequency_path,
                expected_size=registry.semantic_size,
                expected_corpus_revision=str(
                    self.dataset.corpus.metadata.get("corpus_revision") or ""
                ),
            )
            self.frequency_buckets = torch.from_numpy(
                semantic_frequency_buckets(counts)
            )

        stage4 = dict(self.eval_section.get("stage4", {}) or {})
        self.stage4_probe: Stage4SemanticProbe | None = None
        self.stage4_target_store: Stage4AudioTargetStore | None = None
        self.stage4_artifact_path: str | None = None
        if bool(stage4.get("enabled", False)):
            artifact = stage4.get("artifact")
            repository = stage4.get("repository")
            if not artifact or not repository:
                raise KeyError(
                    "evaluation.stage4.enabled=true must be configured when artifact/repository"
                )
            self.stage4_artifact_path = str(Path(artifact).resolve())
            local = stage_checkpoint_locally(
                artifact,
                cache_dir=str(
                    stage4.get(
                        "local_cache_dir", "/tmp/open_qwen_music.llm/stage4-probe"
                    )
                ),
            )
            self.stage4_probe = Stage4SemanticProbe.from_artifact(
                local,
                repository=repository,
                device=device,
                precision=str(stage4.get("precision", "bf16")),
                ctc_vocab_path=stage4.get("ctc_vocab_path"),
                expected_tokenizer_revision=self.dataset.corpus.tokenizer_revision,
            )
            if stage4.get("audio_targets"):
                self.stage4_target_store = Stage4AudioTargetStore(
                    stage4["audio_targets"]
                )
                target_revision = self.stage4_target_store.metadata.get(
                    "semantic_tokenizer_revision"
                )
                if (
                    target_revision
                    and str(target_revision)
                    != self.dataset.corpus.tokenizer_revision
                ):
                    raise RuntimeError(
                        "Stage4 audio-target tokenizer revision does not match the corpus: "
                        f"target={target_revision}, "
                        f"corpus={self.dataset.corpus.tokenizer_revision}"
                    )

    def run(
        self,
        model: torch.nn.Module,
        *,
        step: int,
        include_embedding_norms: bool,
        full: bool = False,
    ) -> dict[str, float]:
        inner_model = model.module if isinstance(model, DistributedDataParallel) else model
        assert isinstance(inner_model, MusicLLM)
        inner_model.set_semantic_frequency_buckets(self.frequency_buckets)
        direct_enabled = bool(self.eval_section.get("direct_metrics", True))
        results: dict[str, float] = {"checkpoint_step": float(step)}
        for mode in self.modes:
            selected = (
                self.all_batches_by_mode[mode]
                if full
                else self.batches_by_mode[mode]
            )
            batches = selected[self.rank :: self.world_size]
            rounds = math.ceil(len(selected) / max(self.world_size, 1)) if selected else 0
            numerators: dict[str, float] = {}
            denominators: dict[str, float] = {}
            calibration_sums: dict[str, float] = {}
            effective = Counter()
            local_samples = 0
            local_batches = 0
            local_anchor_sections = 0
            local_anchor_tokens = 0
            local_anchor_truncated = 0
            for position in range(rounds):
                active = position < len(batches)
                indices = batches[position] if active else selected[0]
                batch = self.collator([self.dataset[key] for key in indices])
                output = _forward(
                    model,
                    batch["input_ids"].to(self.device),
                    batch["labels"].to(self.device),
                    batch["attention_mask"].to(self.device),
                    True,
                    constraint_kinds=(
                        batch["constraint_kinds"].to(self.device)
                        if "constraint_kinds" in batch
                        else None
                    ),
                    compute_diagnostics=direct_enabled,
                )
                if not active:
                    continue
                values = {name: float(value) for name, value in output.metrics.items()}
                for name, value in values.items():
                    if name.startswith("_calibration_"):
                        calibration_sums[name] = calibration_sums.get(name, 0.0) + value
                        continue
                    weight = metric_token_weight(
                        name,
                        values,
                        num_loss_tokens=int(output.num_loss_tokens or 0),
                        accuracy_limit=int(inner_model.accuracy_sample_positions),
                    )
                    if weight is None or weight <= 0:
                        continue
                    numerators[name] = numerators.get(name, 0.0) + value * weight
                    denominators[name] = denominators.get(name, 0.0) + weight
                effective.update(str(value) for value in batch["modes"])
                local_samples += int(batch["input_ids"].shape[0])
                local_batches += 1
                local_anchor_sections += int(batch["semantic_anchor_sections"].sum())
                local_anchor_tokens += int(batch["semantic_anchor_tokens"].sum())
                local_anchor_truncated += int(
                    batch["semantic_anchor_text_truncated"].sum()
                )
            reduced = reduce_weighted_metrics(numerators, denominators)
            for name, value in reduced.items():
                key = f"{mode.value}/{name}"
                results[key] = value
                if name.startswith("loss") and math.isfinite(value):
                    results[f"{mode.value}/ppl{name.removeprefix('loss')}"] = math.exp(
                        min(value, 50.0)
                    )
            calibration = finalize_calibration_metrics(
                _reduce_sum_mapping(calibration_sums)
            )
            results.update(
                {f"{mode.value}/{name}": value for name, value in calibration.items()}
            )
            gathered_effective = all_gather_objects(dict(effective))
            effective_total: Counter[str] = Counter()
            for payload in gathered_effective:
                if isinstance(payload, dict):
                    effective_total.update(
                        {str(name): int(value) for name, value in payload.items()}
                    )
            requested = sum(effective_total.values())
            results[f"{mode.value}/requested_samples"] = float(requested)
            results[f"{mode.value}/effective_mode_rate"] = (
                effective_total.get(mode.value, 0) / max(requested, 1)
            )
            results[f"{mode.value}/fallback_rate"] = (
                1.0 - results[f"{mode.value}/effective_mode_rate"]
            )
            for effective_mode, count in sorted(effective_total.items()):
                results[
                    f"{mode.value}/effective_as_{effective_mode}_rate"
                ] = count / max(requested, 1)
            results[f"{mode.value}/samples"] = reduce_scalar_sum(float(local_samples))
            results[f"{mode.value}/batches"] = reduce_scalar_sum(float(local_batches))
            results[f"{mode.value}/semantic_anchor_sections"] = reduce_scalar_sum(
                float(local_anchor_sections)
            )
            results[f"{mode.value}/semantic_anchor_tokens"] = reduce_scalar_sum(
                float(local_anchor_tokens)
            )
            results[
                f"{mode.value}/semantic_anchor_text_truncated"
            ] = reduce_scalar_sum(float(local_anchor_truncated))

        plain_loss = results.get("plain/loss_semantic")
        if plain_loss is not None:
            for mode_name in ("section", "unique_section"):
                conditioned = results.get(f"{mode_name}/loss_semantic")
                if conditioned is not None:


                    results[
                        f"{mode_name}/unpaired_loss_difference_vs_plain"
                    ] = plain_loss - conditioned

        conditioning = dict(self.eval_section.get("conditioning", {}) or {})
        if not full and bool(conditioning.get("enabled", True)):
            results.update(
                _conditioning_sensitivity(
                    model,
                    self.dataset,
                    self.collator,
                    registry=self.registry,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    limit=int(conditioning.get("max_samples", 8)),
                    donors=int(conditioning.get("donors", 3)),
                )
            )
        anchor_sensitivity = dict(
            self.eval_section.get("anchor_sensitivity", {}) or {}
        )
        if not full and self.sequence_config.semantic_section_reanchor and bool(
            anchor_sensitivity.get("enabled", True)
        ):
            results.update(
                _semantic_anchor_sensitivity(
                    model,
                    self.dataset,
                    self.collator,
                    registry=self.registry,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    limit=int(anchor_sensitivity.get("max_samples", 8)),
                )
            )
        if not full and self.stage4_probe is not None:
            stage4 = dict(self.eval_section.get("stage4", {}) or {})
            results.update(
                _stage4_head_metrics(
                    model,
                    self.dataset,
                    self.collator,
                    self.stage4_probe,
                    self.stage4_target_store,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    limit=int(stage4.get("max_samples", 8)),
                    bootstrap_resamples=int(
                        stage4.get("bootstrap_resamples", 2000)
                    ),
                    bootstrap_confidence=float(
                        stage4.get("bootstrap_confidence", 0.95)
                    ),
                    bootstrap_seed=int(
                        stage4.get(
                            "bootstrap_seed",
                            (self.config.get("train", {}) or {}).get("seed", 0),
                        )
                    ),
                )
            )
            generation = dict(self.eval_section.get("generation", {}) or {})
            every = max(1, int(generation.get("every_evals", 1)))
            generation_every_steps = int(generation.get("every_steps", 0))
            generation_due = (
                step % generation_every_steps == 0
                if generation_every_steps > 0
                else (self.completed_runs + 1) % every == 0
            )
            if bool(generation.get("enabled", False)) and generation_due:
                vocal_metrics = _stage4_free_generation_metrics(
                    inner_model,
                    self.dataset,
                    self.stage4_probe,
                    device=self.device,
                    limit=int(generation.get("max_samples", 2)),
                    generation_section=generation,
                    step=step,
                )
                results.update(vocal_metrics)
                instrumental = dict(generation.get("instrumental", {}) or {})
                if bool(instrumental.get("enabled", False)):
                    instrumental_generation = {
                        key: value
                        for key, value in generation.items()
                        if key != "instrumental"
                    }
                    instrumental_generation.update(instrumental)
                    instrumental_metrics = _stage4_free_generation_metrics(
                        inner_model,
                        self.dataset,
                        self.stage4_probe,
                        device=self.device,
                        limit=int(instrumental_generation.get("max_samples", 2)),
                        generation_section=instrumental_generation,
                        step=step,
                    )
                    results.update(
                        {
                            key.replace(
                                "stage4_free/",
                                "stage4_free_instrumental/",
                                1,
                            ): value
                            for key, value in instrumental_metrics.items()
                        }
                    )
                    semantic_risks = [
                        value
                        for value in (
                            vocal_metrics.get(
                                "stage4_free/selection/semantic_risk"
                            ),
                            instrumental_metrics.get(
                                "stage4_free/selection/semantic_risk"
                            ),
                        )
                        if value is not None
                    ]
                    if semantic_risks:
                        results["stage4_free/selection/semantic_risk"] = max(
                            map(float, semantic_risks)
                        )
        if include_embedding_norms:
            results.update(_embedding_norm_metrics(inner_model))
        results.update(long_range_forgetting_metrics(results))
        results.update(
            {
                key: value
                for key, value in summarize_eval_metrics(results).items()
                if key
                in {
                    "balanced_semantic_nll",
                    "balanced_melody_pitch_nll",
                    "balanced_melody_struct_nll",
                    "ubnll",
                }
            }
        )
        if not full:
            self.completed_runs += 1
        return results

    def close(self) -> None:
        self.dataset.reset_handles()
        self.stage4_probe = None
        self.stage4_target_store = None

    def metadata(self) -> dict[str, Any]:
        stage4: dict[str, Any] = {
            "enabled": self.stage4_probe is not None,
            "artifact": self.stage4_artifact_path,
            "reference_target": (
                "audio_mel_chroma_plus_reference_token_ceiling"
                if self.stage4_target_store is not None
                else "reference_semantic_head_output"
            ),
        }
        if self.stage4_probe is not None:
            stage4.update(
                {
                    "protocol": self.stage4_probe.info.protocol,
                    "semantic_tokenizer_revision": (
                        self.stage4_probe.info.semantic_tokenizer_revision
                    ),
                    "source_checkpoint": self.stage4_probe.info.source_checkpoint,
                    "ctc_vocab_path": self.stage4_probe.info.vocab_path,
                    "ctc_vocab_kind": self.stage4_probe.info.vocab_kind,
                    "upper_layers": self.stage4_probe.info.upper_layers,
                }
            )
        return {
            "split": self.split,
            "sample_ids_revision": self.eval_sample_ids_revision,
            "selected_records": int(len(self.dataset.record_indices)),
            "max_batches_per_view": self.max_batches,
            "eval_plan_revision": self.eval_plan_revision,
            "full_eval_plan_revision": self.full_eval_plan_revision,
            "full_batches_per_view": {
                mode.value: len(self.all_batches_by_mode[mode])
                for mode in self.modes
            },
            "full_every_steps": int(
                self.eval_section.get("full_every_steps", 0)
            ),
            "direct_metrics": bool(self.eval_section.get("direct_metrics", True)),
            "condition_template_version": (
                self.dataset.builder.condition_template_version
            ),
            "condition_policy": dict(self.config.get("condition", {}) or {}),
            "semantic_frequency_counts": self.frequency_counts_path,
            "stage4": stage4,
            "generation": dict(self.eval_section.get("generation", {}) or {}),
        }


def evaluate(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    split: str = "valid",
    max_batches: int = 50,
) -> dict[str, float]:
    rank, local_rank, world_size, device = init_distributed()
    registry = resolve_registry(config)
    text_encoder = build_text_encoder(config)
    strategy = resolve_strategy(config, world_size=world_size)
    validate_precision(config, strategy)
    checkpoint_is_sharded = is_sharded_checkpoint(checkpoint_path)
    model: torch.nn.Module = build_model(config, registry, device=device).to(device)
    sidecar = read_sidecar(checkpoint_path)
    if not checkpoint_is_sharded:
        load_checkpoint(checkpoint_path, model=model, resume=False)
    if world_size > 1:
        model = wrap_model(
            model,
            strategy=strategy,
            config=config,
            device=device,
            local_rank=local_rank,
            world_size=world_size,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        barrier()
    if checkpoint_is_sharded:
        load_checkpoint(
            checkpoint_path,
            model=model,
            resume=False,
            strategy=strategy,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        barrier()
    model.eval()
    runtime = _LoadedModelEvaluator(
        config,
        registry,
        text_encoder,
        split=split,
        max_batches=max_batches,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    from .inference import validate_generation_checkpoint

    try:
        validate_generation_checkpoint(
            sidecar.get("provenance") or {},
            config=config,
            registry_revision=registry.revision,
            condition_template_version=runtime.dataset.builder.condition_template_version,
            semantic_tokenizer_revision=runtime.dataset.corpus.tokenizer_revision,
        )
        with torch.no_grad():
            results = runtime.run(
                model,
                step=int(sidecar.get("global_step") or 0),
                include_embedding_norms=True,
            )
        log_kv(
            "eval",
            {
                key: value
                for key, value in results.items()
                if "loss" in key
                or "acc" in key
                or "nll_gap" in key
                or key.startswith("stage4/")
            },
        )
        return results
    finally:
        runtime.close()
        cleanup_distributed()
