
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import re
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from open_qwen_music.common.checkpoint import (
    checkpoint_content_fingerprint,
    config_hash,
    file_sha256,
    unwrap_model,
)
from open_qwen_music.common.distributed import (
    assert_distributed_consensus,
    barrier,
    cleanup_distributed,
    gather_object_to_rank0,
    init_distributed,
    is_main_process,
    raise_if_rank0_error,
    reduce_scalar_sum,
    reduce_tensor_sum,
)

from .conditioning import (
    SUPPORTED_LYRICS_MAX_TOKENS,
    RenderConditioner,
    TextEncoderProvenance,
)
from .contracts import (
    LATENT_DIM,
    LATENT_FRAME_HZ,
    SAMPLE_RATE,
    SAMPLES_PER_LATENT_FRAME,
    SEMANTIC_CODEBOOK_SIZE,
    mask_to_lengths,
)
from .dit import DiTConfig
from .dit_factory import build_render_dit_components_from_config
from .flow import (
    ConsistencyFlowMatchingConfig,
    FlowConfig,
    FlowMatchingObjective,
    MaskedLoss,
    channelwise_mse_sums,
    consistency_flow_matching_loss,
    make_consistency_flow_matching_pair,
    sample_source_like,
    sample_timesteps,
    timestep_binned_mse,
    timestep_channelwise_mse_sums,
)
from .sample_data import RENDER_SAMPLE_SCHEMA, CanonicalRenderSampleDataset
from .semantic_corruption import (
    SemanticCorruptionConfig,
    SemanticDistractorTable,
    load_semantic_distractor_table,
    verify_semantic_error_calibration,
)
from .renderer_data import (
    RendererDataFixedValidationDataset,
    RendererDataParentBatchSampler,
    RendererDataShortWindowDataset,
    encode_renderer_data_rng_state,
)
from .text_cache import RenderTextCacheLoader

DURATION_CONTRACT_V1_360 = "oqm.render.duration-buckets.v1-360"
DURATION_CONTRACT_V2_NATIVE_360 = "oqm.render.duration-buckets.v2-native-360"
DURATION_BUCKET_PROTOCOLS = {
    DURATION_CONTRACT_V1_360: (30, 90, 180, 360),


    DURATION_CONTRACT_V2_NATIVE_360: (30, 90, 180, 360),
}

DURATION_BUCKETS_SECONDS = DURATION_BUCKET_PROTOCOLS[DURATION_CONTRACT_V1_360]
CHECKPOINT_ADAPTER_VERSION = "oqm.render.dit-trainer.ckpt-adapter.v5"
SUPPORTED_CHECKPOINT_ADAPTER_VERSIONS = frozenset(
    {
        "oqm.render.dit-trainer.ckpt-adapter.v4",
        CHECKPOINT_ADAPTER_VERSION,
    }
)
DIT_EMA_FORMAT_VERSION = "oqm.render.dit-ema.v1"
DIT_EMA_MODES = ("none", "parameter_fp32")
SAMPLER_STATE_VERSION = "oqm.render.distributed-duration-sampler.v1"
TRAINING_TOPOLOGY_VERSION = "oqm.render.dit-training-topology.v1"
TRAINING_PRECISIONS = ("fp32", "bf16")
AUDIO_SECONDS_UPDATE_POLICIES = ("minimum", "exact", "telemetry_only")
INIT_ADDABLE_UPSTREAMS: frozenset[str] = frozenset()
INIT_REPLACEABLE_UPSTREAMS: frozenset[str] = frozenset()
INIT_REPLACEABLE_CONFIG_FIELDS: frozenset[str] = frozenset()
DIT_CONFIG_FORMAT_VERSION = "oqm.render.dit.config.v1"
DIT_TEST_CONFIG_FORMAT_VERSION = "oqm.render.dit.config.test.v1"
DIT_EVALUATION_CONFIG_SECTIONS = (
    "experiment",
    "model",
    "conditioning",
    "flow",
    "semantic_corruption",
    "revisions",
    "data",
    "validation",
    "optimizer",
    "train",
)
DIT_DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS = (":4096:8", ":16:8")
DURATION_CURRICULUM_FORMAT_VERSION = "oqm.render.duration-curriculum.v1"
DURATION_CURRICULUM_MODES = ("nested_shortest",)


def _require_explicit_fields(
    owner: Mapping[str, Any],
    required: frozenset[str],
    *,
    section: str,
) -> None:

    missing = sorted(required - set(owner))
    if missing:
        raise RuntimeError(f"{section} must declare every required field; missing={missing}")


_MISSING = object()
_NO_DEFAULT = object()


def _batch_get(batch: Any, name: str, default: Any = _NO_DEFAULT) -> Any:
    if isinstance(batch, Mapping):
        if name in batch:
            return batch[name]
    elif hasattr(batch, name):
        return getattr(batch, name)
    if default is _NO_DEFAULT:
        raise KeyError(f"Render DiT batch is missing field {name!r}")
    return default


def _move_value(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_value(item, device) for key, item in value.items()}
    return value


def move_batch_to_device(batch: Any, device: torch.device) -> dict[str, Any]:
    names = (
        "sample_ids",
        "semantic_ids",
        "semantic_mask",
        "latents",
        "latent_mask",
        "description_input_ids",
        "description_mask",
        "lyrics_input_ids",
        "lyrics_mask",
        "description_embeddings",
        "lyrics_embeddings",
        "duration_seconds",
        "global_loudness_lufs",
        "revisions",
        "provenance",
    )
    result = {}
    for name in names:
        value = _batch_get(batch, name, _MISSING)
        if value is not _MISSING and value is not None:
            result[name] = _move_value(value, device)
    return result


@dataclass(frozen=True)
class RevisionContract:
    tokenizer_revision: str
    vae_revision: str
    text_encoder_revision: str
    text_tokenizer_revision: str
    text_cache_revision: str
    rewriter_revision: str
    latent_cache_revision: str
    latent_stats_sha256: str

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a revision string")
            normalized = value.strip().lower()
            if normalized.startswith("pin-") or normalized in {
                "",
                "main",
                "master",
                "latest",
                "head",
                "unknown",
                "unresolved",
                "none",
                "null",
                "required",
                "required_pinned_revision",
            }:
                raise ValueError(f"{name} must match the required revision; received {value!r}")
        if len(self.latent_stats_sha256) != 64:
            raise ValueError("latent_stats_sha256 must be a 64-character SHA-256")
        try:
            int(self.latent_stats_sha256, 16)
        except ValueError as exc:
            raise ValueError("latent_stats_sha256 is not a hexadecimal SHA-256") from exc

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RevisionContract:
        aliases = {
            "semantic_tokenizer_revision": "tokenizer_revision",
            "spec_vae_revision": "vae_revision",
            "model_revision": "text_encoder_revision",
            "tokenizer_revision_text": "text_tokenizer_revision",
            "cache_revision": "text_cache_revision",
        }
        normalized = dict(value)
        for old, new in aliases.items():
            if old in normalized:
                if new in normalized and normalized[new] != normalized[old]:
                    raise ValueError(f"revision alias conflict: {old}/{new}")
                normalized.setdefault(new, normalized[old])
                del normalized[old]


        for name in ("refiner_revision", "checkpoint_revision"):
            normalized.pop(name, None)
        unknown = set(normalized) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"revision config contains unknown fields: {sorted(unknown)}")
        result = cls(
            **{name: normalized.get(name, "") for name in cls.__dataclass_fields__}
        )
        result.validate()
        return result

    def text_provenance(self, model_id: str) -> TextEncoderProvenance:
        return TextEncoderProvenance(
            model_id=model_id,
            model_revision=self.text_encoder_revision,
            tokenizer_revision=self.text_tokenizer_revision,
            cache_revision=self.text_cache_revision,
        )


def _normalize_dit_checkpoint_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:

    normalized = json.loads(json.dumps(config, sort_keys=True, default=str))

    relocatable_suffixes = (
        "_path",
        "_dir",
        "_root",
        "_manifest",
        "_artifact",
        "_checkpoint",
        "_file",
        "_repository",
    )
    relocatable_names = {
        "asset_lock",
        "cache_manifest",
        "corpus",
        "index",
        "local_path",
        "manifest",
        "output",
        "output_dir",
        "path",
        "prompts",
        "ready_path",
        "repository",
        "root",
        "runtime_root",
        "source_root",
        "vocab",
    }

    def normalize_paths(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {
                name: normalize_paths(item, str(name))
                for name, item in value.items()
            }
        if isinstance(value, list):
            return [normalize_paths(item, key) for item in value]
        if isinstance(value, str) and (
            key in relocatable_names or key.endswith(relocatable_suffixes)
        ):
            return f"<relocatable:{key}>"
        return value

    normalized = normalize_paths(normalized)
    normalized.pop("experiment", None)
    model = normalized.get("model")
    if isinstance(model, dict):
        if "num_kv_heads" not in model and isinstance(model.get("num_heads"), int):
            model["num_kv_heads"] = model["num_heads"]
        model.setdefault("normalization_type", "layernorm")
        model.setdefault("qk_norm", "none")
        model.setdefault("qk_norm_eps", 1.0e-6)
        model.setdefault("qk_norm_affine", False)
        model.setdefault("cross_attention_gate", "adaln")
    conditioning = normalized.get("conditioning")
    if isinstance(conditioning, dict):
        conditioning.setdefault("lyrics_norm_type", "layernorm")
        conditioning.setdefault("null_context_layout", "single_token")
    optimizer = normalized.get("optimizer")
    if isinstance(optimizer, dict):


        optimizer.setdefault("ema_decay", None)
    train = normalized.get("train")
    if isinstance(train, dict):
        train.setdefault("activation_checkpoint_block_interval", 1)





    return normalized


_EMDC_UPSTREAM_SHA_KEYS = frozenset(
    {
        "semantic_distractor_ready_sha256",
        "semantic_distractor_artifact_sha256",
        "semantic_distractor_ids_sha256",
        "semantic_distractor_scores_sha256",
        "semantic_error_llm_checkpoint_sha256",
        "semantic_error_token_registry_sha256",
        "semantic_error_evaluation_manifest_sha256",
        "semantic_error_evaluation_report_sha256",
    }
)












def _normalized_checkpoint_config_value(value: Any) -> Any:

    if isinstance(value, Mapping):
        return {
            str(key): _normalized_checkpoint_config_value(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_normalized_checkpoint_config_value(item) for item in value]
    return value


def validate_dit_checkpoint_evaluation_identity(
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    expected_step: int,
    context: str = "  ",
) -> None:

    if not isinstance(state, Mapping):
        raise TypeError("DiT checkpoint state must be a mapping")
    if not isinstance(config, Mapping):
        raise TypeError("DiT review config must be a mapping")
    if (
        not isinstance(expected_step, int)
        or isinstance(expected_step, bool)
        or expected_step < 0
    ):
        raise ValueError("DiT review expected_step must be a non-negative integer")
    if not isinstance(context, str) or not context.strip():
        raise TypeError("DiT review context must be a non-empty string")
    stored_config = state.get("config")
    if not isinstance(stored_config, Mapping):
        raise RuntimeError("DiT checkpoint is missing config")
    stored_config = _normalize_dit_checkpoint_config(stored_config)
    config = _normalize_dit_checkpoint_config(config)
    mismatches = {
        section: {
            "expected": config.get(section),
            "actual": stored_config.get(section),
        }
        for section in DIT_EVALUATION_CONFIG_SECTIONS
        if _normalized_checkpoint_config_value(stored_config.get(section))
        != _normalized_checkpoint_config_value(config.get(section))
    }
    if mismatches:
        raise RuntimeError(f"DiT checkpoint and {context} configuration mismatch: {mismatches}")
    training_state = state.get("training_state")
    if not isinstance(training_state, Mapping):
        raise RuntimeError("DiT checkpoint is missing training_state")
    actual_step = training_state.get("global_step")
    if (
        not isinstance(actual_step, int)
        or isinstance(actual_step, bool)
        or actual_step != expected_step
    ):
        raise RuntimeError(
            f"DiT checkpoint step mismatch: expected={expected_step} actual={actual_step}"
        )


def _normalize_revision_rows(
    batch: Mapping[str, Any], batch_size: int
) -> list[Mapping[str, Any]]:
    revisions = batch.get("revisions")
    if revisions is None:
        provenance = batch.get("provenance")
        revisions = provenance
    if isinstance(revisions, Mapping):
        return [revisions] * batch_size
    if isinstance(revisions, Sequence) and not isinstance(revisions, (str, bytes)):
        rows = list(revisions)
        if len(rows) != batch_size or not all(isinstance(row, Mapping) for row in rows):
            raise ValueError("revisions must be a list of B mappings")
        return rows
    raise RuntimeError("Precaching DiT batch must carry revisions")


def validate_batch_revisions(
    batch: Mapping[str, Any], expected: RevisionContract
) -> list[RevisionContract]:
    expected.validate()
    batch_size = int(batch["semantic_ids"].shape[0])
    observed = [
        RevisionContract.from_mapping(row)
        for row in _normalize_revision_rows(batch, batch_size)
    ]
    mismatches = []
    for index, revision in enumerate(observed):
        if revision != expected:
            fields = {
                key: (getattr(expected, key), getattr(revision, key))
                for key in asdict(expected)
                if getattr(expected, key) != getattr(revision, key)
            }
            mismatches.append((index, fields))
    if mismatches:
        raise RuntimeError(f"batch cache revision does not match the training contract: {mismatches}")
    return observed


def validate_dit_batch(
    batch: Mapping[str, Any],
    *,
    expected_revisions: RevisionContract | None = None,
) -> dict[str, Any]:
    required = (
        "latents",
        "latent_mask",
        "semantic_ids",
        "semantic_mask",
    )
    for name in required:
        if name not in batch:
            raise KeyError(f"DiT batch is missing {name}")
    latents = batch["latents"]
    latent_mask = batch["latent_mask"]
    semantic_ids = batch["semantic_ids"]
    semantic_mask = batch["semantic_mask"]
    if (
        not isinstance(latents, torch.Tensor)
        or latents.ndim != 3
        or latents.shape[-1] != LATENT_DIM
        or not latents.is_floating_point()
    ):
        raise TypeError("latents must be floating point with shape [B, T, 128]")
    if (
        not isinstance(semantic_ids, torch.Tensor)
        or semantic_ids.dtype != torch.long
        or semantic_ids.shape != latents.shape[:2]
    ):
        raise TypeError("semantic_ids must be latent-aligned int64 with shape [B, T]")
    for name, mask in (
        ("latent_mask", latent_mask),
        ("semantic_mask", semantic_mask),
    ):
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.shape != latents.shape[:2]
        ):
            raise TypeError(f"{name} must be bool with shape [B, T]")
        mask_to_lengths(mask, require_right_padded=True)
    if not torch.equal(latent_mask, semantic_mask):
        raise ValueError("semantic and latent masks must match frame by frame")
    lengths = semantic_mask.sum(dim=1)
    if latents.shape[0] <= 0 or latents.shape[1] <= 0 or bool((lengths <= 0).any()):
        raise ValueError("DiT batch: each sample must contain at least one valid frame")
    if semantic_ids.numel() and (
        int(semantic_ids.min()) < 0 or int(semantic_ids.max()) >= SEMANTIC_CODEBOOK_SIZE
    ):
        raise ValueError("semantic IDs, including padding, must be in [0, 32767]")
    if not torch.isfinite(latents).all():
        raise ValueError("cached latents contain NaN/Inf")
    for name, maximum in (("description", 256), ("lyrics", 1_536)):
        ids = batch.get(f"{name}_input_ids")
        embeddings = batch.get(f"{name}_embeddings")
        mask = batch.get(f"{name}_mask")
        if mask is None or (ids is None) == (embeddings is None):
            raise ValueError(
                f"{name} must provide mask,and input_ids/cached embeddings There must be exactly one"
            )
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"{name}_mask must be a tensor")
        if mask.dtype != torch.bool or mask.ndim != 2:
            raise TypeError(f"{name}_mask must be bool with shape [B, L]")
        if mask.shape[0] != latents.shape[0] or mask.shape[1] > maximum:
            raise ValueError(f"{name} batch has an invalid length")
        if ids is not None:
            if not isinstance(ids, torch.Tensor):
                raise TypeError(f"{name}_input_ids must be a tensor")
            if ids.dtype != torch.long or ids.shape != mask.shape:
                raise TypeError(f"{name}_input_ids must be int64 with the same shape as its mask")
            if ids.device != latents.device or mask.device != latents.device:
                raise ValueError(f"{name} IDs, mask, and latents must be on the same device")
        if embeddings is not None:
            if not isinstance(embeddings, torch.Tensor):
                raise TypeError(f"{name}_embeddings must be a tensor")
            if (
                embeddings.ndim != 3
                or embeddings.shape[:2] != mask.shape
                or not embeddings.is_floating_point()
            ):
                raise ValueError(f"{name}_embeddings must be floating point with shape [B, L, D]")
            if embeddings.device != latents.device or mask.device != latents.device:
                raise ValueError(f"{name} embeddings, mask, and latents must be on the same device")
            if not torch.isfinite(embeddings).all():
                raise ValueError(f"{name}_embeddings contains NaN/Inf")
            if (~mask).any() and torch.count_nonzero(embeddings[~mask]):
                raise ValueError(f"{name}_embeddings at padding positions must be zero")
    duration = batch.get("duration_seconds")
    if duration is not None:
        if (
            not isinstance(duration, torch.Tensor)
            or duration.shape != (latents.shape[0],)
            or not duration.is_floating_point()
            or not torch.isfinite(duration).all()
        ):
            raise ValueError("duration_seconds must be finite floating point with shape [B]")


        observed_duration = duration.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        canonical_samples = (
            lengths.detach().to(device="cpu", dtype=torch.long)
            * SAMPLES_PER_LATENT_FRAME
        )
        canonical_duration = canonical_samples.to(torch.float64) / SAMPLE_RATE
        if not torch.equal(observed_duration, canonical_duration):
            raise ValueError(
                "duration_seconds Assertion only,must be strictly equal to "
                "semantic_frames * 1920 / 48000 of canonical value"
            )
    global_loudness = batch.get("global_loudness_lufs")
    if global_loudness is not None and (
        not isinstance(global_loudness, torch.Tensor)
        or global_loudness.shape != (latents.shape[0],)
        or not global_loudness.is_floating_point()
        or not torch.isfinite(global_loudness).all()
        or global_loudness.device != latents.device
    ):
        raise ValueError("global_loudness_lufs must be a finite floating-point tensor with shape [B] on the latent device")
    if expected_revisions is not None:
        validate_batch_revisions(batch, expected_revisions)
    return dict(batch)


def freeze_module(module: nn.Module, *, name: str) -> nn.Module:
    module.requires_grad_(False)
    module.eval()
    module._oqm_frozen_module_name = str(name)
    return module


def assert_module_frozen(module: nn.Module, *, name: str) -> None:
    trainable = [
        parameter_name
        for parameter_name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    gradients = [
        parameter_name
        for parameter_name, parameter in module.named_parameters()
        if parameter.grad is not None
    ]
    if trainable or gradients:
        raise RuntimeError(
            f"Frozen module {name} has an invalid trainable state: {trainable[:5]} "
            f"gradients={gradients[:5]}"
        )


@dataclass(frozen=True)
class DurationBucketPolicy:
    buckets_seconds: tuple[int, ...] = DURATION_BUCKETS_SECONDS
    latent_frame_hz: int = LATENT_FRAME_HZ

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in self.buckets_seconds
        ):
            raise TypeError("duration buckets must be an integer")
        if (
            not isinstance(self.latent_frame_hz, int)
            or isinstance(self.latent_frame_hz, bool)
            or self.latent_frame_hz != LATENT_FRAME_HZ
        ):
            raise ValueError(f"Render latent frame rate must be {LATENT_FRAME_HZ} Hz")
        buckets = tuple(self.buckets_seconds)
        if not buckets or buckets != tuple(sorted(set(buckets))):
            raise ValueError("duration buckets must be strictly incremental and non-repeating")
        if any(value <= 0 for value in buckets):
            raise ValueError("duration bucket must be positive")
        supported = {
            value
            for protocol_buckets in DURATION_BUCKET_PROTOCOLS.values()
            for value in protocol_buckets
        }
        unsupported = set(buckets) - supported
        if unsupported:
            raise ValueError(
                "Renderer length courses only support versionedduration buckets,"
                f"received {sorted(unsupported)}"
            )

    def bucket_for_frames(self, frames: int) -> int:
        if not isinstance(frames, int) or isinstance(frames, bool):
            raise TypeError("valid frame count must be an integer")
        if frames <= 0:
            raise ValueError("valid frame count must be positive")
        seconds = int(math.ceil(frames / self.latent_frame_hz))
        for bucket in self.buckets_seconds:
            if seconds <= bucket:
                return bucket
        raise ValueError(f"Sample {seconds}s exceeds maximum bucket {self.buckets_seconds[-1]}s")

    def frames_for_bucket(self, bucket_seconds: int) -> int:
        if not isinstance(bucket_seconds, int) or isinstance(bucket_seconds, bool):
            raise TypeError("duration bucket must be an integer")
        if bucket_seconds not in self.buckets_seconds:
            raise ValueError(f"Unknown duration bucket {bucket_seconds}")
        return bucket_seconds * self.latent_frame_hz

    @staticmethod
    def audio_seconds(frame_mask: torch.Tensor) -> float:
        if frame_mask.dtype != torch.bool or frame_mask.ndim != 2:
            raise TypeError("frame_mask must be bool with shape [B, T]")
        return float(frame_mask.sum().item()) / LATENT_FRAME_HZ

    def batch_size_for_global_audio_seconds(
        self,
        bucket_seconds: int,
        *,
        global_audio_seconds_per_update: float,
        world_size: int = 1,
        gradient_accumulation_steps: int = 1,
    ) -> int:
        if (
            not isinstance(bucket_seconds, int)
            or isinstance(bucket_seconds, bool)
            or not isinstance(world_size, int)
            or isinstance(world_size, bool)
            or not isinstance(gradient_accumulation_steps, int)
            or isinstance(gradient_accumulation_steps, bool)
        ):
            raise TypeError("bucket/world_size/gradient_accumulation_steps must be an integer")
        if (
            isinstance(global_audio_seconds_per_update, bool)
            or not isinstance(global_audio_seconds_per_update, (int, float))
            or not math.isfinite(global_audio_seconds_per_update)
        ):
            raise TypeError("global_audio_seconds_per_update must be a finite value")
        denominator = bucket_seconds * world_size * gradient_accumulation_steps
        if denominator <= 0 or global_audio_seconds_per_update <= 0:
            raise ValueError("audio-seconds/update and topology must be positive")
        ratio = float(global_audio_seconds_per_update) / denominator
        batch = round(ratio)
        if batch <= 0:
            raise ValueError(
                "target audio-seconds/update is smaller than one bucket and cannot preserve optimizer semantics"
            )
        if not math.isclose(ratio, batch, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                "target audio-seconds/update cannot be represented by the current bucket, world-size, and accumulation topology"
            )
        return int(batch)


@dataclass(frozen=True)
class DurationCurriculumStage:

    start_step: int
    active_records: int
    batch_size_per_rank: int


@dataclass(frozen=True)
class DurationCurriculum:

    stages: tuple[DurationCurriculumStage, ...]
    ordering: str = "frame_length_then_sample_id_sha256"
    mode: str = "nested_shortest"
    format_version: str = DURATION_CURRICULUM_FORMAT_VERSION

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        dataset_size: int | None = None,
    ) -> DurationCurriculum:
        if value.get("format_version") != DURATION_CURRICULUM_FORMAT_VERSION:
            raise ValueError("duration_curriculum.format_version is incompatible")
        mode = str(value.get("mode") or "")
        if mode not in DURATION_CURRICULUM_MODES:
            raise ValueError(
                f"duration_curriculum.mode must be{DURATION_CURRICULUM_MODES}"
            )
        ordering = str(value.get("ordering") or "")
        if ordering != "frame_length_then_sample_id_sha256":
            raise ValueError(
                "duration_curriculum.ordering only supports frame_length_then_sample_id_sha256"
            )
        raw_stages = value.get("stages")
        if (
            not isinstance(raw_stages, Sequence)
            or isinstance(raw_stages, (str, bytes))
            or not raw_stages
        ):
            raise TypeError("duration_curriculum.stages must be a non-empty sequence of objects")
        stages: list[DurationCurriculumStage] = []
        for raw in raw_stages:
            if not isinstance(raw, Mapping):
                raise TypeError("duration_curriculum stage must be an object")
            unknown = set(raw) - {
                "start_step",
                "active_records",
                "batch_size_per_rank",
            }
            if unknown:
                raise ValueError(
                    f"duration_curriculum stagecontains unknown fields: {sorted(unknown)}"
                )
            values: dict[str, int] = {}
            for name in ("start_step", "active_records", "batch_size_per_rank"):
                item = raw.get(name)
                if not isinstance(item, int) or isinstance(item, bool):
                    raise TypeError(f"duration_curriculum stage.{name} must be an integer")
                values[name] = item
            if values["start_step"] < 0:
                raise ValueError("duration_curriculum stage.start_step must not be negative")
            if values["active_records"] <= 0:
                raise ValueError("duration_curriculum stage.active_records must be positive")
            if values["batch_size_per_rank"] <= 0:
                raise ValueError(
                    "duration_curriculum stage.batch_size_per_rank must be positive"
                )
            stages.append(DurationCurriculumStage(**values))
        if stages[0].start_step != 0:
            raise ValueError("duration_curriculum: the first stage must start at step 0")
        if any(
            stages[index].start_step <= stages[index - 1].start_step
            or stages[index].active_records <= stages[index - 1].active_records
            for index in range(1, len(stages))
        ):
            raise ValueError(
                "duration_curriculum start_step and active_records must increase strictly"
            )
        if dataset_size is not None:
            if stages[-1].active_records != int(dataset_size):
                raise ValueError(
                    "duration_curriculum: the  final stage must cover the complete data set:"
                    f"expected={dataset_size} actual={stages[-1].active_records}"
                )
            if any(stage.active_records > int(dataset_size) for stage in stages):
                raise ValueError("duration_curriculum active_records exceeds the dataset size")
        return cls(stages=tuple(stages), ordering=ordering, mode=mode)

    def stage_index_for_step(self, global_step: int) -> int:
        if int(global_step) < 0:
            raise ValueError("duration curriculum global_step must not be negative")
        result = 0
        for index, stage in enumerate(self.stages):
            if stage.start_step > int(global_step):
                break
            result = index
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "mode": self.mode,
            "ordering": self.ordering,
            "stages": [asdict(stage) for stage in self.stages],
        }


def validate_exact_bucket_frame_lengths(
    frame_lengths: Sequence[int],
    *,
    policy: DurationBucketPolicy,
) -> None:

    allowed = {policy.frames_for_bucket(bucket) for bucket in policy.buckets_seconds}
    invalid = [
        {"index": index, "frames": int(frames)}
        for index, frames in enumerate(frame_lengths)
        if int(frames) not in allowed
    ]
    if invalid:
        raise RuntimeError(
            f"exact bucket data must be cropped to declared duration boundaries: {invalid[:8]}"
        )


class DurationBucketBatchSampler:

    def __init__(
        self,
        frame_lengths: Sequence[int],
        *,
        policy: DurationBucketPolicy,
        batch_size_by_bucket: Mapping[int, int],
        seed: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        if not frame_lengths:
            raise ValueError("duration bucket sampler requires non-empty length")
        self.policy = policy
        self.frame_lengths = [int(value) for value in frame_lengths]
        self.batch_size_by_bucket = {
            int(key): int(value) for key, value in batch_size_by_bucket.items()
        }
        missing = set(policy.buckets_seconds) - set(self.batch_size_by_bucket)
        if missing or any(value <= 0 for value in self.batch_size_by_bucket.values()):
            raise ValueError(
                f"each duration bucket must have positive batch size;missing={sorted(missing)}"
            )
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.groups: dict[int, list[int]] = {
            bucket: [] for bucket in policy.buckets_seconds
        }
        for index, frames in enumerate(self.frame_lengths):
            self.groups[policy.bucket_for_frames(frames)].append(index)

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("sampler epoch must not be negative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator(device="cpu").manual_seed(self.seed + self.epoch)
        batches: list[list[int]] = []
        for bucket in self.policy.buckets_seconds:
            indices = list(self.groups[bucket])
            if self.shuffle and indices:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[index] for index in order]
            batch_size = self.batch_size_by_bucket[bucket]
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                if len(batch) == batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle and batches:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in order]
        yield from batches

    def __len__(self) -> int:
        total = 0
        for bucket, indices in self.groups.items():
            batch_size = self.batch_size_by_bucket[bucket]
            total += (
                len(indices) // batch_size
                if self.drop_last
                else math.ceil(len(indices) / batch_size)
            )
        return total

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.set_epoch(int(state["epoch"]))


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class FixedCropSelectionDataset(Dataset[dict[str, Any]]):

    def __init__(
        self,
        dataset: CanonicalRenderSampleDataset,
        *,
        selection_path: str | Path,
        expected_selection_sha256: str,
        expected_source_manifest_sha256: str,
    ) -> None:
        self.dataset = dataset
        self.selection_path = Path(selection_path).expanduser().resolve(strict=True)
        actual_selection_sha256 = file_sha256(self.selection_path)
        if actual_selection_sha256 != expected_selection_sha256:
            raise RuntimeError(
                "fixed crop selection SHA mismatch:"
                f"expected={expected_selection_sha256} "
                f"actual={actual_selection_sha256}"
            )
        try:
            selection = json.loads(self.selection_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("fixed crop selection cannot read") from exc
        if not isinstance(selection, Mapping):
            raise TypeError("fixed crop selection must be a JSON object")
        if selection.get("schema_version") != ("oqm.render-dit-vae-gap-selection.v1"):
            raise RuntimeError("fixed crop selection schema is incompatible")
        if selection.get("split") != dataset.split:
            raise RuntimeError("fixed crop selection split and canonical release mismatch")
        if selection.get("source_manifest_sha256") != expected_source_manifest_sha256:
            raise RuntimeError("fixed crop selection is not bound to the current canonical manifest")
        rows = selection.get("rows")
        if (
            not isinstance(rows, Sequence)
            or isinstance(rows, (str, bytes))
            or not rows
            or not all(isinstance(row, Mapping) for row in rows)
        ):
            raise TypeError("fixed crop selection rows must be a non-empty sequence of objects")
        if selection.get("rows_sha256") != _sha256_json(rows):
            raise RuntimeError("fixed crop selection rows SHA mismatch")

        source_sample_ids = dataset.sample_ids
        source_frame_lengths = dataset.frame_lengths
        normalized_rows: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        selected_indices: set[int] = set()
        for raw in rows:
            index = int(raw.get("target_index", -1))
            sample_id = str(raw.get("sample_id") or "")
            crop_start = int(raw.get("crop_start_frame", -1))
            crop_frames = int(raw.get("crop_frames", 0))
            if not 0 <= index < len(dataset):
                raise ValueError(f"fixed crop target_index out of bounds: {index}")
            if index in selected_indices or sample_id in selected_ids:
                raise RuntimeError("fixed crop selection contains duplicate sample IDs or target_index values")
            if source_sample_ids[index] != sample_id:
                raise RuntimeError("fixed crop selection sample_id and target_index mismatch")
            if crop_start < 0 or crop_frames <= 0:
                raise ValueError("fixed crop start or frame count is invalid")
            if crop_start + crop_frames > source_frame_lengths[index]:
                raise RuntimeError("fixed crop window exceeds the canonical sample")
            normalized_rows.append(
                {
                    **dict(raw),
                    "target_index": index,
                    "sample_id": sample_id,
                    "crop_start_frame": crop_start,
                    "crop_frames": crop_frames,
                }
            )
            selected_indices.add(index)
            selected_ids.add(sample_id)

        self.rows = tuple(normalized_rows)
        self._sample_ids = tuple(str(row["sample_id"]) for row in self.rows)
        self._frame_lengths = tuple(int(row["crop_frames"]) for row in self.rows)

        self.index_sha256 = actual_selection_sha256
        self.ready_sha256 = dataset.ready_sha256
        self.ready_path = dataset.ready_path
        self.split_groups_disjoint_verified = dataset.split_groups_disjoint_verified

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self._sample_ids

    @property
    def frame_lengths(self) -> list[int]:
        return list(self._frame_lengths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        item = self.dataset[int(row["target_index"])]
        sample_id = str(row["sample_id"])
        if item["sample_id"] != sample_id:
            raise RuntimeError("fixed crop runtime sample_id mismatch")
        provenance = item["provenance"]
        expected_hashes = {
            "latent_sha256": row["latent_sha256"],
            "semantic_sha256": row["semantic_sha256"],
        }
        mismatches = {
            name: {"expected": expected, "actual": provenance.get(name)}
            for name, expected in expected_hashes.items()
            if provenance.get(name) != expected
        }
        if mismatches:
            raise RuntimeError(f"fixed crop upstream artifact identity mismatch: {mismatches}")
        start = int(row["crop_start_frame"])
        stop = start + int(row["crop_frames"])
        result = dict(item)
        for name in ("latents", "latent_mask", "semantic_ids", "semantic_mask"):
            result[name] = item[name][start:stop]
        result["duration_seconds"] = int(row["crop_frames"]) / LATENT_FRAME_HZ
        result["provenance"] = {
            **dict(provenance),
            "fixed_crop_selection": str(self.selection_path),
            "fixed_crop_selection_sha256": self.index_sha256,
            "crop_start_frame": start,
            "crop_frames": int(row["crop_frames"]),
        }
        return result


class DistributedDurationBucketBatchSampler:

    def __init__(
        self,
        frame_lengths: Sequence[int],
        *,
        policy: DurationBucketPolicy,
        batch_size_by_bucket: Mapping[int, int],
        rank: int,
        world_size: int,
        seed: int,
        training_config_hash: str,
        sample_ids: Sequence[str] | None = None,
        duration_curriculum: Mapping[str, Any] | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        if not frame_lengths:
            raise ValueError("distributed duration sampler requires non-empty length")
        if int(world_size) <= 0 or not 0 <= int(rank) < int(world_size):
            raise ValueError("distributed sampler rank or world_size is invalid")
        if not str(training_config_hash):
            raise ValueError("distributed sampler must be bound to training config hash")
        self.policy = policy
        self.frame_lengths = [int(value) for value in frame_lengths]
        if any(value <= 0 for value in self.frame_lengths):
            raise ValueError("distributed sampler frame lengths must all be positive")
        self.batch_size_by_bucket = {
            int(key): int(value) for key, value in batch_size_by_bucket.items()
        }
        missing = set(policy.buckets_seconds) - set(self.batch_size_by_bucket)
        if missing or any(value <= 0 for value in self.batch_size_by_bucket.values()):
            raise ValueError(
                "each duration bucket must have positive per-rank batch size;"
                f"missing={sorted(missing)}"
            )
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.training_config_hash = str(training_config_hash)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.cursor = 0
        if sample_ids is None:
            self.sample_ids = tuple(
                str(index) for index in range(len(self.frame_lengths))
            )
        else:
            if len(sample_ids) != len(self.frame_lengths):
                raise ValueError("sample_ids and frame_lengths differ in length")
            self.sample_ids = tuple(str(value) for value in sample_ids)
            if any(not value for value in self.sample_ids):
                raise ValueError("sample_ids must be non-empty")
            if len(set(self.sample_ids)) != len(self.sample_ids):
                raise ValueError("sample_ids must be unique")
        self.duration_curriculum = (
            DurationCurriculum.from_mapping(
                duration_curriculum,
                dataset_size=len(self.frame_lengths),
            )
            if duration_curriculum is not None
            else None
        )
        self.curriculum_order = tuple(
            sorted(
                range(len(self.frame_lengths)),
                key=lambda index: (
                    self.frame_lengths[index],
                    hashlib.sha256(
                        f"{self.seed}:{self.sample_ids[index]}".encode()
                    ).hexdigest(),
                ),
            )
        )
        self.curriculum_global_step = 0
        self.curriculum_stage_index = 0
        self.groups: dict[int, list[int]] = {
            bucket: [] for bucket in policy.buckets_seconds
        }
        for index, frames in enumerate(self.frame_lengths):
            self.groups[policy.bucket_for_frames(frames)].append(index)
        self.topology = {
            "format_version": SAMPLER_STATE_VERSION,
            "world_size": self.world_size,
            "buckets_seconds": list(policy.buckets_seconds),
            "latent_frame_hz": int(policy.latent_frame_hz),
            "batch_size_by_bucket": {
                str(key): self.batch_size_by_bucket[key]
                for key in sorted(self.batch_size_by_bucket)
            },
            "seed": self.seed,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "dataset_size": len(self.frame_lengths),
            "frame_lengths_sha256": _sha256_json(self.frame_lengths),
            "training_config_hash": self.training_config_hash,
        }
        if self.duration_curriculum is not None:
            self.topology["duration_curriculum"] = {
                **self.duration_curriculum.to_dict(),
                "sample_ids_sha256": _sha256_json(self.sample_ids),
                "curriculum_order_sha256": _sha256_json(self.curriculum_order),
            }
        self.sampler_config_hash = _sha256_json(self.topology)
        if self.duration_curriculum is not None:
            for stage in self.duration_curriculum.stages:
                global_capacity = stage.batch_size_per_rank * self.world_size
                if stage.active_records % global_capacity != 0:
                    raise ValueError(
                        "duration curriculumper stageactive_records must be able to beglobal batchDivisible by:"
                        f"active={stage.active_records} capacity={global_capacity}"
                    )
        if self.num_batches_per_epoch == 0:
            raise ValueError(
                "The current data is in distributed duration sampler Not available under global batch;"
                "each end bucket requires at least world_size bar sample"
            )

    def _global_batches(self, epoch: int) -> list[tuple[int, list[int]]]:
        generator = torch.Generator(device="cpu").manual_seed(self.seed + int(epoch))
        if self.duration_curriculum is not None:
            stage = self.duration_curriculum.stages[self.curriculum_stage_index]
            indices = list(self.curriculum_order[: stage.active_records])
            global_capacity = stage.batch_size_per_rank * self.world_size
            batches: list[tuple[int, list[int]]] = []
            for start in range(0, len(indices), global_capacity):
                batch = indices[start : start + global_capacity]
                if self.shuffle:
                    order = torch.randperm(len(batch), generator=generator).tolist()
                    batch = [batch[index] for index in order]
                bucket = self.policy.bucket_for_frames(
                    max(self.frame_lengths[index] for index in batch)
                )
                batches.append((bucket, batch))
            if self.shuffle and batches:
                order = torch.randperm(len(batches), generator=generator).tolist()
                batches = [batches[index] for index in order]
            return batches
        batches: list[tuple[int, list[int]]] = []
        for bucket in self.policy.buckets_seconds:
            indices = list(self.groups[bucket])
            if self.shuffle and indices:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[index] for index in order]
            global_capacity = self.batch_size_by_bucket[bucket] * self.world_size
            for start in range(0, len(indices), global_capacity):
                batch = indices[start : start + global_capacity]
                if len(batch) < global_capacity and self.drop_last:
                    continue
                if len(batch) < self.world_size:
                    continue
                batches.append((bucket, batch))
        if self.shuffle and batches:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in order]
        return batches

    def _local_slice(self, global_batch: Sequence[int]) -> list[int]:
        size = len(global_batch)
        quotient, remainder = divmod(size, self.world_size)
        local_size = quotient + int(self.rank < remainder)
        start = self.rank * quotient + min(self.rank, remainder)
        result = list(global_batch[start : start + local_size])
        if not result:
            raise AssertionError("sampler should not produce null rank batch")
        return result

    @property
    def curriculum_telemetry(self) -> dict[str, Any] | None:
        if self.duration_curriculum is None:
            return None
        stage = self.duration_curriculum.stages[self.curriculum_stage_index]
        active = self.curriculum_order[: stage.active_records]
        return {
            "stage_index": self.curriculum_stage_index,
            "start_step": stage.start_step,
            "active_records": stage.active_records,
            "batch_size_per_rank": stage.batch_size_per_rank,
            "max_duration_seconds": max(self.frame_lengths[index] for index in active)
            / self.policy.latent_frame_hz,
            "global_step": self.curriculum_global_step,
        }

    def set_global_step(self, global_step: int) -> bool:

        if int(global_step) < 0:
            raise ValueError("sampler global_step must not be negative")
        self.curriculum_global_step = int(global_step)
        if self.duration_curriculum is None:
            return False
        stage_index = self.duration_curriculum.stage_index_for_step(global_step)
        if stage_index == self.curriculum_stage_index:
            return False
        self.curriculum_stage_index = stage_index
        self.cursor = 0
        return True

    @property
    def num_batches_per_epoch(self) -> int:
        return len(self._global_batches(self.epoch))

    def set_epoch(self, epoch: int, *, reset_cursor: bool = True) -> None:
        if int(epoch) < 0:
            raise ValueError("sampler epoch must not be negative")
        self.epoch = int(epoch)
        if reset_cursor:
            self.cursor = 0

    def advance(self, count: int = 1) -> None:
        value = self.cursor + int(count)
        if int(count) < 0 or value > self.num_batches_per_epoch:
            raise RuntimeError(
                "distributed sampler cursor exceeds the current epoch batch number:"
                f"cursor={self.cursor} advance={count} "
                f"total={self.num_batches_per_epoch}"
            )
        self.cursor = value

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._global_batches(self.epoch)
        for _, global_batch in batches[self.cursor :]:
            yield self._local_slice(global_batch)

    def __len__(self) -> int:
        return max(0, self.num_batches_per_epoch - self.cursor)

    def state_dict(self) -> dict[str, Any]:
        state = {
            "format_version": SAMPLER_STATE_VERSION,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "topology": dict(self.topology),
            "config_hash": self.sampler_config_hash,
        }
        if self.duration_curriculum is not None:
            state["duration_curriculum_state"] = {
                "global_step": self.curriculum_global_step,
                "stage_index": self.curriculum_stage_index,
            }
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("format_version") != SAMPLER_STATE_VERSION:
            raise RuntimeError("checkpoint sampler format_version is incompatible")
        stored_topology = state.get("topology")
        if not isinstance(stored_topology, Mapping):
            raise RuntimeError("checkpoint sampler is missing topology")
        if _sha256_json(stored_topology) != state.get("config_hash"):
            raise RuntimeError("checkpoint sampler topology/config hash has been tampered with")
        if dict(stored_topology) != self.topology:
            raise RuntimeError(
                "resume sampler world/batch data/config Topology is mismatch:"
                f"current={self.topology} checkpoint={dict(stored_topology)}"
            )
        epoch = int(state.get("epoch", -1))
        cursor = int(state.get("cursor", -1))
        if epoch < 0 or cursor < 0:
            raise RuntimeError("checkpoint sampler epoch or cursor is invalid")
        self.epoch = epoch
        curriculum_state = state.get("duration_curriculum_state")
        if self.duration_curriculum is None:
            if curriculum_state is not None:
                raise RuntimeError(
                    "Non-coursesampler checkpointresidualduration curriculumstatus"
                )
        else:
            if not isinstance(curriculum_state, Mapping):
                raise RuntimeError("Curriculum sampler checkpoint is missing duration curriculum state")
            global_step = int(curriculum_state.get("global_step", -1))
            stage_index = int(curriculum_state.get("stage_index", -1))
            if (
                global_step < 0
                or stage_index
                != self.duration_curriculum.stage_index_for_step(global_step)
            ):
                raise RuntimeError("checkpoint duration curriculum stage is invalid")
            self.curriculum_global_step = global_step
            self.curriculum_stage_index = stage_index
        total = self.num_batches_per_epoch
        if cursor > total:
            raise RuntimeError(
                "checkpoint sampler cursor exceeds the current epoch:"
                f"cursor={cursor} total={total}"
            )
        self.cursor = cursor


class StructuredJSONLLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Mapping[str, Any]) -> None:
        serializable = {}
        for key, value in record.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise TypeError(f"Log field {key} is not a scalar")
                value = value.detach().cpu().item()
            if isinstance(value, Path):
                value = str(value)
            serializable[key] = value
        line = json.dumps(
            serializable, ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        with self.path.open("a", encoding="utf-8") as output:
            output.write(line + "\n")
            output.flush()


def build_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    max_steps: int,
    scheduler: str = "cosine",
    min_lr_ratio: float = 1.0,
    step_indexing: str = "optimizer_update_zero_based_final_inclusive",
) -> torch.optim.lr_scheduler.LambdaLR:
    if warmup_steps < 0 or max_steps <= 0 or warmup_steps > max_steps:
        raise ValueError("warmup_steps must be in [0, max_steps]")
    if scheduler not in {"constant", "cosine"}:
        raise ValueError("scheduler must be constant or cosine")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if scheduler == "constant" and min_lr_ratio != 1.0:
        raise ValueError("constant scheduler requires min_lr_ratio=1.0")
    if step_indexing != "optimizer_update_zero_based_final_inclusive":
        raise ValueError(
            "scheduler step_indexingonly supportsoptimizer_update_zero_based_final_inclusive"
        )

    def schedule(step: int) -> float:


        if warmup_steps and step < warmup_steps:
            return max(1.0e-8, float(step + 1) / warmup_steps)
        if scheduler == "constant":
            return 1.0
        decay_updates = max_steps - warmup_steps


        progress = (
            1.0 if decay_updates <= 1 else (step - warmup_steps) / (decay_updates - 1)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


@dataclass
class TrainingLossOutput:
    masked_loss: MaskedLoss
    prediction: torch.Tensor
    target_velocity: torch.Tensor
    text_drop_mask: torch.Tensor
    timestep: torch.Tensor
    timestep_bin_numerators: torch.Tensor | None = None
    timestep_bin_denominators: torch.Tensor | None = None
    channel_mse_numerators: torch.Tensor | None = None
    channel_mse_denominator: torch.Tensor | None = None
    timestep_channel_mse_numerators: torch.Tensor | None = None
    timestep_channel_mse_denominators: torch.Tensor | None = None
    semantic_replacement_mask: torch.Tensor | None = None
    semantic_replacement_probability: float = 0.0
    source_coupling_cost_before: torch.Tensor | None = None
    source_coupling_cost_after: torch.Tensor | None = None
    source_coupling_batch_size: int = 1
    source_coupling_scope: str = "independent"
    training_objective: str = "flow_matching"
    endpoint_consistency_numerator: torch.Tensor | None = None
    endpoint_consistency_denominator: torch.Tensor | None = None
    velocity_consistency_numerator: torch.Tensor | None = None
    velocity_consistency_denominator: torch.Tensor | None = None

    @property
    def loss(self) -> torch.Tensor:
        return self.masked_loss.loss


class RenderDiTTrainingGraph(nn.Module):

    def __init__(
        self,
        *,
        model: nn.Module,
        conditioner: RenderConditioner,
    ) -> None:
        super().__init__()
        self.conditioner = conditioner
        self.model = model

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        semantic_ids: torch.Tensor,
        frame_mask: torch.Tensor,
        description_input_ids: torch.Tensor | None = None,
        description_mask: torch.Tensor,
        lyrics_input_ids: torch.Tensor | None = None,
        lyrics_mask: torch.Tensor,
        description_embeddings: torch.Tensor | None = None,
        lyrics_embeddings: torch.Tensor | None = None,
        cache_provenance: TextEncoderProvenance | None = None,
        text_drop_mask: torch.Tensor | None = None,
        global_loudness_lufs: torch.Tensor | None = None,
        neighbor_latents: torch.Tensor | None = None,
        neighbor_timestep: torch.Tensor | None = None,
        share_model_rng: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (neighbor_latents is None) != (neighbor_timestep is None):
            raise ValueError("neighbor_latents and neighbor_timestep must also be provided")
        if not isinstance(share_model_rng, bool):
            raise TypeError("share_model_rng must be bool")
        if neighbor_latents is None and share_model_rng:
            raise ValueError("Neighbor state is missing")
        if neighbor_latents is not None:
            if neighbor_latents.shape != noisy_latents.shape:
                raise ValueError("neighbor_latents must have the same shape as primary latents")
            if neighbor_timestep.shape != timestep.shape:
                raise ValueError("neighbor_timestep must have the same shape as primary timestep")
        conditioning = self.conditioner(
            semantic_ids,
            frame_mask,
            description_input_ids=description_input_ids,
            description_mask=description_mask,
            lyrics_input_ids=lyrics_input_ids,
            lyrics_mask=lyrics_mask,
            description_embeddings=description_embeddings,
            lyrics_embeddings=lyrics_embeddings,
            cache_provenance=cache_provenance,
            text_drop_mask=text_drop_mask,
        )
        embed_global_loudness = getattr(
            self.conditioner,
            "embed_global_loudness",
            None,
        )
        global_loudness_embedding = (
            embed_global_loudness(global_loudness_lufs)
            if callable(embed_global_loudness)
            else None
        )
        model_loudness_kwargs = (
            {"global_loudness_embedding": global_loudness_embedding}
            if global_loudness_embedding is not None
            else {}
        )
        if neighbor_latents is None:
            return self.model(
                noisy_latents,
                timestep,
                conditioning=conditioning,
                frame_mask=frame_mask,
                **model_loudness_kwargs,
            )

        cpu_rng_before = torch.get_rng_state() if share_model_rng else None
        cuda_rng_before = (
            torch.cuda.get_rng_state(noisy_latents.device)
            if share_model_rng and noisy_latents.device.type == "cuda"
            else None
        )
        primary = self.model(
            noisy_latents,
            timestep,
            conditioning=conditioning,
            frame_mask=frame_mask,
            **model_loudness_kwargs,
        )
        cpu_rng_after = torch.get_rng_state() if share_model_rng else None
        cuda_rng_after = (
            torch.cuda.get_rng_state(noisy_latents.device)
            if share_model_rng and noisy_latents.device.type == "cuda"
            else None
        )
        try:
            if cpu_rng_before is not None:
                torch.set_rng_state(cpu_rng_before)
            if cuda_rng_before is not None:
                torch.cuda.set_rng_state(cuda_rng_before, noisy_latents.device)
            with torch.no_grad():
                neighbor = self.model(
                    neighbor_latents,
                    neighbor_timestep,
                    conditioning=conditioning,
                    frame_mask=frame_mask,
                    **model_loudness_kwargs,
                )
        finally:
            if cpu_rng_after is not None:
                torch.set_rng_state(cpu_rng_after)
            if cuda_rng_after is not None:
                torch.cuda.set_rng_state(cuda_rng_after, noisy_latents.device)
        return primary, neighbor


class TrainableParameterEMA:

    def __init__(
        self,
        model: nn.Module,
        conditioner: RenderConditioner,
        *,
        decay: float,
    ) -> None:
        if (
            isinstance(decay, bool)
            or not isinstance(decay, (int, float))
            or not math.isfinite(decay)
            or not 0.0 < float(decay) < 1.0
        ):
            raise ValueError("EMA decay must be a finite value in (0, 1)")
        self.decay = float(decay)
        self.num_updates = 0
        self._parameters = self._named_trainable_parameters(model, conditioner)
        if not self._parameters:
            raise ValueError("EMA has no trainable parameters")
        self.shadow = {
            name: parameter.detach().float().clone()
            for name, parameter in self._parameters.items()
        }

    @staticmethod
    def _named_trainable_parameters(
        model: nn.Module,
        conditioner: RenderConditioner,
    ) -> dict[str, nn.Parameter]:
        result: dict[str, nn.Parameter] = {}
        for prefix, module in (("dit", model), ("conditioner", conditioner)):
            for name, parameter in module.named_parameters():
                if parameter.requires_grad:
                    result[f"{prefix}.{name}"] = parameter
        return result

    @torch.no_grad()
    def reset_from_parameters(self) -> None:
        for name, parameter in self._parameters.items():
            self.shadow[name].copy_(parameter.detach().float())
        self.num_updates = 0

    @torch.no_grad()
    def update(self) -> None:
        one_minus_decay = 1.0 - self.decay
        for name, parameter in self._parameters.items():
            self.shadow[name].lerp_(parameter.detach().float(), one_minus_decay)
        self.num_updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": DIT_EMA_FORMAT_VERSION,
            "mode": "parameter_fp32",
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": dict(self.shadow),
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("EMA state must be a mapping")
        if state.get("format_version") != DIT_EMA_FORMAT_VERSION:
            raise RuntimeError("EMA state format_version is incompatible")
        if state.get("mode") != "parameter_fp32":
            raise RuntimeError("EMA state mode is incompatible")
        if float(state.get("decay", -1.0)) != self.decay:
            raise RuntimeError("EMA checkpoint decay does not match the current configuration")
        num_updates = state.get("num_updates")
        if (
            not isinstance(num_updates, int)
            or isinstance(num_updates, bool)
            or num_updates < 0
        ):
            raise RuntimeError("EMA checkpoint num_updates is invalid")
        shadow = state.get("shadow")
        if not isinstance(shadow, Mapping) or set(shadow) != set(self.shadow):
            raise RuntimeError("EMA checkpoint parameter set does not match")
        for name, target in self.shadow.items():
            value = shadow[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != target.shape
                or value.dtype != torch.float32
                or not torch.isfinite(value).all()
            ):
                raise RuntimeError(f"EMA checkpoint parameter {name} is invalid")
        for name, target in self.shadow.items():
            target.copy_(shadow[name].to(device=target.device))
        self.num_updates = int(num_updates)


def apply_trainable_parameter_ema_state(
    model: nn.Module,
    conditioner: RenderConditioner,
    state: Mapping[str, Any],
) -> dict[str, Any]:

    decay = state.get("decay") if isinstance(state, Mapping) else None
    ema = TrainableParameterEMA(model, conditioner, decay=float(decay))
    ema.load_state_dict(state)
    with torch.no_grad():
        for name, parameter in ema._parameters.items():
            parameter.copy_(ema.shadow[name].to(dtype=parameter.dtype))
    return {
        "format_version": DIT_EMA_FORMAT_VERSION,
        "mode": "parameter_fp32",
        "decay": ema.decay,
        "num_updates": ema.num_updates,
        "parameter_count": len(ema.shadow),
    }


class CheckpointAdapter(Protocol):
    format_version: str

    def save(
        self,
        path: str | Path,
        *,
        model: nn.Module,
        conditioner: RenderConditioner,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        ema: TrainableParameterEMA | None,
        trainer_state: Mapping[str, Any],
        config: Mapping[str, Any],
        revisions: Mapping[str, str],
        rng_states: Mapping[str, Any],
        distributed_state: Mapping[str, Any],
    ) -> None: ...

    def load(
        self,
        path: str | Path,
        *,
        model: nn.Module,
        conditioner: RenderConditioner,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any,
        ema: TrainableParameterEMA | None,
        expected_config: Mapping[str, Any],
        expected_revisions: Mapping[str, str],
        resume: bool,
        expected_checkpoint_sha256: str | None = None,
        pre_apply_validator: (Callable[[Mapping[str, Any]], None] | None) = None,
    ) -> Mapping[str, Any]: ...


class _TrainableConditionerCheckpointView(nn.Module):

    _FROZEN_PREFIX = "text_encoder.encoder."
    _FROZEN_DYNAMICS_PREFIX = "relative_dynamics_conditioning.predictor."

    def __init__(self, conditioner: RenderConditioner) -> None:
        super().__init__()

        object.__setattr__(self, "_conditioner", conditioner)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        state = self._conditioner.state_dict(*args, **kwargs)
        frozen_prefixes = [
            self._FROZEN_PREFIX,
            self._FROZEN_DYNAMICS_PREFIX,
        ]
        if not self._conditioner.semantic_embedding.weight.requires_grad:
            frozen_prefixes.append("semantic_embedding.")
        return {
            key: value
            for key, value in state.items()
            if not any(key.startswith(prefix) for prefix in frozen_prefixes)
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        forbidden_prefixes = (
            self._FROZEN_PREFIX,
            self._FROZEN_DYNAMICS_PREFIX,
        )
        forbidden = [
            key
            for key in state_dict
            if any(key.startswith(prefix) for prefix in forbidden_prefixes)
        ]
        if forbidden:
            raise RuntimeError(
                f"DiT checkpoint must not contain embedded language-model weights:{forbidden[:3]}"
            )
        result = self._conditioner.load_state_dict(
            state_dict, strict=False, assign=assign
        )
        allowed_missing_prefixes = [
            self._FROZEN_PREFIX,
            self._FROZEN_DYNAMICS_PREFIX,
        ]
        if not self._conditioner.semantic_embedding.weight.requires_grad:
            allowed_missing_prefixes.append("semantic_embedding.")
        bad_missing = [
            key
            for key in result.missing_keys
            if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        if bad_missing or result.unexpected_keys:
            raise RuntimeError(
                "conditioner checkpoint state is incompatible:"
                f"missing={bad_missing}, unexpected={result.unexpected_keys}"
            )
        return result


TrainableConditionerCheckpointView = _TrainableConditionerCheckpointView


class StrictRenderCheckpointAdapter:

    format_version = CHECKPOINT_ADAPTER_VERSION

    def __init__(
        self,
        upstream_revisions: Mapping[str, Any],
        *,
        required_upstream_sha_keys: Sequence[str] = (),
    ) -> None:
        from .checkpoint import validate_upstream_revisions

        self.upstream_revisions = validate_upstream_revisions(
            upstream_revisions,
            required_sha_keys=required_upstream_sha_keys,
        )
        self.required_upstream_sha_keys = tuple(required_upstream_sha_keys)

    def _assert_logical_revisions(self, expected: Mapping[str, str]) -> None:
        component_names = {
            "tokenizer_revision": "tokenizer",
            "vae_revision": "vae",
            "text_encoder_revision": "text_encoder",
            "text_tokenizer_revision": "text_tokenizer",
            "text_cache_revision": "text_cache",
            "rewriter_revision": "rewriter",
            "latent_cache_revision": "latent_cache",
            "latent_stats_sha256": "latent_stats",
        }
        mismatches = {}
        for field, value in expected.items():
            observed = self.upstream_revisions.get(field)
            if observed is None:
                component = self.upstream_revisions.get(component_names.get(field, ""))
                if isinstance(component, Mapping):
                    observed = component.get("revision")
            if observed != value:
                mismatches[field] = (value, observed)
        if mismatches:
            raise RuntimeError(
                f"checkpoint adapter upstream revision and trainer mismatch:{mismatches}"
            )

    def validate_revisions(self, expected: Mapping[str, str]) -> None:
        self._assert_logical_revisions(expected)

    def _validate_init_upstream_revisions(
        self,
        observed: Mapping[str, Any],
        *,
        allowed_replacements: Sequence[str],
    ) -> None:
        if (
            not isinstance(allowed_replacements, Sequence)
            or isinstance(allowed_replacements, (str, bytes))
            or not all(
                isinstance(value, str) and value for value in allowed_replacements
            )
            or len(set(allowed_replacements)) != len(allowed_replacements)
        ):
            raise TypeError(
                "train.init_allowed_upstream_replacements must be a unique string sequence"
            )
        allowed = set(allowed_replacements)
        unknown = allowed - INIT_REPLACEABLE_UPSTREAMS
        if unknown:
            raise RuntimeError(f"init_from replacement of these upstream identities are not allowed:  {sorted(unknown)}")
        source = dict(observed)
        target = self.upstream_revisions
        source_only = set(source) - set(target)
        target_only = set(target) - set(source)
        if source_only or target_only - INIT_ADDABLE_UPSTREAMS or target_only - allowed:
            raise RuntimeError(
                "init_from checkpoint upstream fields do not match: "
                f"source={sorted(source)} target={sorted(target)}"
            )
        missing_allowed = allowed - (set(source) | set(target))
        if missing_allowed:
            raise RuntimeError(
                f"init_from replacement field does not exist: {sorted(missing_allowed)}"
            )
        mismatches = {
            name: {
                "source": source[name],
                "target": target[name],
            }
            for name in set(source) & set(target)
            if name not in allowed and source[name] != target[name]
        }
        if mismatches:
            raise RuntimeError(f"init_from checkpoint contains unauthorized upstream changes: {mismatches}")


    def save(
        self,
        path: str | Path,
        *,
        model: nn.Module,
        conditioner: RenderConditioner,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        ema: TrainableParameterEMA | None,
        trainer_state: Mapping[str, Any],
        config: Mapping[str, Any],
        revisions: Mapping[str, str],
        rng_states: Mapping[str, Any],
        distributed_state: Mapping[str, Any],
    ) -> None:
        from .checkpoint import save_render_checkpoint

        self._assert_logical_revisions(revisions)
        save_render_checkpoint(
            path,
            models={
                "dit": model,
                "conditioner": _TrainableConditionerCheckpointView(conditioner),
            },
            optimizers={"adamw": optimizer},
            schedulers={"lr": scheduler},
            ema={"trainable_parameters": ema} if ema is not None else None,
            config=config,
            upstream_revisions=self.upstream_revisions,
            required_upstream_sha_keys=self.required_upstream_sha_keys,
            global_step=int(trainer_state["global_step"]),
            epoch=int(trainer_state.get("epoch", 0)),
            batches_consumed=int(trainer_state.get("batches_consumed", 0)),
            training_audio_seconds=float(trainer_state["consumed_audio_seconds"]),
            sampler=trainer_state.get("sampler_state"),
            distributed_state=distributed_state,
            component="render_dit",
            extra_state={
                "dit_trainer_adapter_version": self.format_version,
                "rank_rng_states": dict(rng_states),
                "validation_state": dict(trainer_state.get("validation_state") or {}),
                "training_extension": trainer_state.get("training_extension"),
            },
        )

    def load(
        self,
        path: str | Path,
        *,
        model: nn.Module,
        conditioner: RenderConditioner,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any,
        ema: TrainableParameterEMA | None,
        expected_config: Mapping[str, Any],
        expected_revisions: Mapping[str, str],
        resume: bool,
        expected_checkpoint_sha256: str | None = None,
        pre_apply_validator: (Callable[[Mapping[str, Any]], None] | None) = None,
    ) -> Mapping[str, Any]:
        from .checkpoint import load_render_checkpoint

        self._assert_logical_revisions(expected_revisions)
        source = Path(path).resolve(strict=True)
        def normalize(resumed: Any) -> dict[str, Any]:
            extra = resumed.extra_state
            observed_adapter_version = extra.get("dit_trainer_adapter_version")
            if observed_adapter_version not in SUPPORTED_CHECKPOINT_ADAPTER_VERSIONS:
                raise RuntimeError("checkpoint is missing matching DiT trainer adapter metadata")
            if (
                resume
                and ema is not None
                and observed_adapter_version != self.format_version
            ):
                raise RuntimeError("EMA resume only accepts v5 checkpoint")
            distributed_state = (
                dict(resumed.distributed_state)
                if isinstance(resumed.distributed_state, Mapping)
                else resumed.distributed_state
            )
            sampler_state = None
            if isinstance(resumed.sampler_state, Mapping):
                if (
                    resumed.sampler_state.get("format_version")
                    == "oqm.render.renderer_data-parent-sampler.v1"
                ):


                    rng_state = resumed.sampler_state.get("parent_order_rng_state")
                    without_rng = {
                        key: value
                        for key, value in resumed.sampler_state.items()
                        if key != "parent_order_rng_state"
                    }
                    sampler_state = json.loads(json.dumps(without_rng, sort_keys=True))
                    sampler_state["parent_order_rng_state"] = encode_renderer_data_rng_state(
                        rng_state
                    )
                else:
                    sampler_state = json.loads(
                        json.dumps(
                            resumed.sampler_state,
                            sort_keys=True,
                            default=str,
                        )
                    )
            else:
                sampler_state = resumed.sampler_state


            return {
                "global_step": resumed.global_step,
                "epoch": resumed.epoch,
                "batches_consumed": resumed.batches_consumed,
                "sampler_state": sampler_state,
                "distributed_state": distributed_state,
                "consumed_audio_seconds": resumed.training_audio_seconds,
                "rank_rng_states": extra.get("rank_rng_states"),
                "validation_state": extra.get("validation_state"),
                "training_extension": extra.get("training_extension"),
            }

        def validate_before_apply(resumed: Any) -> None:
            normalized = normalize(resumed)
            if not resume:
                train_config = expected_config.get("train") or {}
                self._validate_init_upstream_revisions(
                    resumed.upstream_revisions,
                    allowed_replacements=train_config.get(
                        "init_allowed_upstream_replacements",
                        (),
                    ),
                )
            if pre_apply_validator is not None:
                pre_apply_validator(normalized)

        resumed = load_render_checkpoint(
            source,
            models={
                "dit": model,
                "conditioner": _TrainableConditionerCheckpointView(conditioner),
            },
            optimizers={"adamw": optimizer} if resume else None,
            schedulers={"lr": scheduler} if resume else None,
            ema=(
                {"trainable_parameters": ema}
                if resume and ema is not None
                else None
            ),
            expected_config=expected_config,
            expected_config_sections=(
                None
                if resume
                else (
                    "model",
                    "conditioning",
                    "flow",
                    "semantic_corruption",
                )
            ),
            expected_config_replacements=(
                () if resume
                else tuple(
                    (expected_config.get("train") or {}).get(
                        "init_allowed_config_replacements",
                        (),
                    )
                )
            ),
            expected_config_normalizer=_normalize_dit_checkpoint_config,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_component="render_dit",
            expected_upstream_revisions=(
                self.upstream_revisions if resume else None
            ),


            required_upstream_sha_keys=(
                self.required_upstream_sha_keys if resume else None
            ),
            resume=resume,

            restore_rng=False,
            strict_sidecar=True,
            pre_apply_validator=validate_before_apply,
        )
        return normalize(resumed)


def validate_checkpoint_adapter(adapter: CheckpointAdapter) -> None:
    if getattr(adapter, "format_version", None) != CHECKPOINT_ADAPTER_VERSION:
        raise TypeError(
            "checkpoint adapter format_version does not match:"
            f"expected={CHECKPOINT_ADAPTER_VERSION} "
            f"actual={getattr(adapter, 'format_version', None)}"
        )
    required_signatures = {
        "save": {
            "path",
            "model",
            "conditioner",
            "optimizer",
            "scheduler",
            "ema",
            "trainer_state",
            "config",
            "revisions",
            "rng_states",
            "distributed_state",
        },
        "load": {
            "path",
            "model",
            "conditioner",
            "optimizer",
            "scheduler",
            "ema",
            "expected_config",
            "expected_revisions",
            "resume",
            "expected_checkpoint_sha256",
            "pre_apply_validator",
        },
    }
    for method_name, required in required_signatures.items():
        method = getattr(adapter, method_name, None)
        if not callable(method):
            raise TypeError(f"checkpoint adapter is missing {method_name}()")
        parameters = set(inspect.signature(method).parameters)
        missing = required - parameters
        if missing:
            raise TypeError(
                f"checkpoint adapter {method_name} is missing required parameters {sorted(missing)}"
            )


class RenderDiTTrainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        conditioner: RenderConditioner,
        flow_config: FlowConfig | Mapping[str, Any],
        revisions: RevisionContract,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.95),
        optimizer_foreach: bool | None = None,
        optimizer_fused: bool | None = None,
        optimizer_eps: float = 1.0e-8,
        optimizer_amsgrad: bool = False,
        optimizer_maximize: bool = False,
        optimizer_capturable: bool = False,
        optimizer_differentiable: bool = False,
        optimizer_parameter_grouping: str = "all_trainable_single_group",
        optimizer_weight_decay_exclusions: Sequence[str] = (),
        optimizer_zero_grad_set_to_none: bool = True,
        ema_mode: str = "none",
        ema_decay: float | None = None,
        warmup_steps: int = 0,
        max_steps: int = 1,
        scheduler: str = "cosine",
        scheduler_step_indexing: str = ("optimizer_update_zero_based_final_inclusive"),
        gradient_clip_norm: float = 1.0,
        gradient_clip_error_if_nonfinite: bool = True,
        gradient_clip_foreach: bool = False,
        text_drop_probability: float = 0.15,
        semantic_embedding_freeze_steps: int = 0,
        semantic_embedding_freeze_policy: str = "first_n_optimizer_updates",
        semantic_corruption_config: (
            SemanticCorruptionConfig | Mapping[str, Any] | None
        ) = None,
        semantic_distractor_table: SemanticDistractorTable | None = None,
        semantic_error_calibration_provenance: Mapping[str, Any] | None = None,
        seed: int = 0,
        rank_seed_stride: int = 1_000_003,
        flow_source_seed_offset: int = 11,
        flow_timestep_seed_offset: int = 23,
        text_drop_seed_offset: int = 37,
        min_lr_ratio: float = 1.0,
        frozen_modules: Mapping[str, nn.Module] | None = None,
        logger: StructuredJSONLLogger | None = None,
        checkpoint_adapter: CheckpointAdapter | None = None,
        config: Mapping[str, Any] | None = None,
        require_cached_text: bool = False,
        rank: int = 0,
        world_size: int = 1,
        distributed_state: Mapping[str, Any] | None = None,
        device: torch.device | str | None = None,
        precision: str = "fp32",
    ) -> None:
        numeric_values = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_clip_norm": gradient_clip_norm,
            "text_drop_probability": text_drop_probability,
            "min_lr_ratio": min_lr_ratio,
            "optimizer_eps": optimizer_eps,
        }
        if ema_decay is not None:
            numeric_values["ema_decay"] = ema_decay
        for name, value in numeric_values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise TypeError(f"{name} must be a finite value")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("AdamW learning_rate must be positive and weight_decay must be non-negative")
        if float(gradient_clip_norm) != 1.0:
            raise ValueError("The Renderer training contract requires gradient_clip=1.0")
        if not isinstance(gradient_clip_error_if_nonfinite, bool):
            raise TypeError("gradient_clip_error_if_nonfinite must be bool")
        if not isinstance(gradient_clip_foreach, bool):
            raise TypeError("gradient_clip_foreach must be bool")
        if not 0.0 <= text_drop_probability <= 1.0:
            raise ValueError("text_drop_probability must be in [0, 1]")
        if (
            not isinstance(semantic_embedding_freeze_steps, int)
            or isinstance(semantic_embedding_freeze_steps, bool)
            or semantic_embedding_freeze_steps < 0
        ):
            raise ValueError("semantic_embedding_freeze_steps must be a non-negative integer")
        if semantic_embedding_freeze_policy != "first_n_optimizer_updates":
            raise ValueError(
                "semantic_embedding_freeze_policyonly supportsfirst_n_optimizer_updates"
            )
        rng_offsets = {
            "flow_source_seed_offset": flow_source_seed_offset,
            "flow_timestep_seed_offset": flow_timestep_seed_offset,
            "text_drop_seed_offset": text_drop_seed_offset,
        }
        if (
            not isinstance(rank_seed_stride, int)
            or isinstance(rank_seed_stride, bool)
            or rank_seed_stride <= 0
        ):
            raise ValueError("rank_seed_stride must be a positive integer")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in rng_offsets.values()
        ):
            raise ValueError("random stream seed offset must be a non-negative integer")
        if len(set(rng_offsets.values())) != len(rng_offsets):
            raise ValueError("Independent random stream seed offset must be unique")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise TypeError("Render trainer rank must be an integer")
        if not isinstance(world_size, int) or isinstance(world_size, bool):
            raise TypeError("Render trainer world_size must be an integer")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("Render trainer rank or world_size is invalid")
        if dist.is_initialized() and (
            int(rank) != dist.get_rank() or int(world_size) != dist.get_world_size()
        ):
            raise RuntimeError("Render trainer rank/world_size and process group mismatch")
        if int(rank) != 0 and logger is not None:
            raise ValueError("structured logger can only be used on rank 0")
        precision = str(precision).lower()
        if precision not in TRAINING_PRECISIONS:
            raise ValueError(f"precision must be {TRAINING_PRECISIONS}")
        revisions.validate()
        if len(betas) != 2 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value < 1.0
            for value in betas
        ):
            raise ValueError("AdamW betas must be two finite values in [0, 1)")
        for name, value in {
            "optimizer_foreach": optimizer_foreach,
            "optimizer_fused": optimizer_fused,
            "optimizer_amsgrad": optimizer_amsgrad,
            "optimizer_maximize": optimizer_maximize,
            "optimizer_capturable": optimizer_capturable,
            "optimizer_differentiable": optimizer_differentiable,
        }.items():
            if name in {"optimizer_foreach", "optimizer_fused"}:
                if value is not None and not isinstance(value, bool):
                    raise TypeError(f"{name} must be bool or null")
            elif not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
        if optimizer_foreach is True and optimizer_fused is True:
            raise ValueError("AdamW foreach and fused cannot both be true")
        if optimizer_eps <= 0:
            raise ValueError("AdamW eps must be a finite positive number")
        if optimizer_parameter_grouping != "all_trainable_single_group":
            raise ValueError(
                "The current optimizer_parameter_groupingonly supportsall_trainable_single_group"
            )
        if (
            not isinstance(optimizer_weight_decay_exclusions, Sequence)
            or isinstance(optimizer_weight_decay_exclusions, (str, bytes))
            or list(optimizer_weight_decay_exclusions)
        ):
            raise ValueError("The current optimizer_weight_decay_exclusions must be an empty list")
        if ema_mode not in DIT_EMA_MODES:
            raise ValueError(f"ema_mode must be one of {DIT_EMA_MODES}")
        if ema_mode == "none":
            if ema_decay is not None:
                raise ValueError("ema_mode=none must not be set when ema_decay is set")
        elif ema_decay is None or not 0.0 < float(ema_decay) < 1.0:
            raise ValueError("parameter_fp32 EMA requires ema_decay in (0, 1)")
        if not isinstance(optimizer_zero_grad_set_to_none, bool):
            raise TypeError("optimizer_zero_grad_set_to_none must be bool")
        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.conditioner = conditioner
        if device is not None:
            target_device = torch.device(device)
            self.model.to(target_device)
            self.conditioner.to(target_device)
        self.training_graph: nn.Module = RenderDiTTrainingGraph(
            model=self.model,
            conditioner=self.conditioner,
        )
        self.flow = FlowMatchingObjective(flow_config)
        self.revisions = revisions
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.gradient_clip_error_if_nonfinite = gradient_clip_error_if_nonfinite
        self.gradient_clip_foreach = gradient_clip_foreach
        self.optimizer_zero_grad_set_to_none = optimizer_zero_grad_set_to_none
        self.text_drop_probability = float(text_drop_probability)
        self.semantic_embedding_freeze_steps = int(semantic_embedding_freeze_steps)
        self.semantic_embedding_freeze_policy = semantic_embedding_freeze_policy
        self.semantic_corruption = (
            semantic_corruption_config
            if isinstance(semantic_corruption_config, SemanticCorruptionConfig)
            else SemanticCorruptionConfig.from_mapping(semantic_corruption_config)
        )
        self.semantic_corruption.validate()
        if self.semantic_corruption.mode == "emdc_knn":
            if not isinstance(semantic_distractor_table, SemanticDistractorTable):
                raise ValueError("emdc_knn training must be injected with a semantic distractor table")
            _, verified_calibration = verify_semantic_error_calibration(
                self.semantic_corruption.calibration,
                expected_tokenizer_revision=revisions.tokenizer_revision,
            )
            if (
                semantic_error_calibration_provenance is not None
                and dict(semantic_error_calibration_provenance) != verified_calibration
            ):
                raise RuntimeError("Injected semantic calibration provenance does not match the verified report")
            if (
                semantic_distractor_table.tokenizer_revision
                != revisions.tokenizer_revision
            ):
                raise RuntimeError(
                    "Semantic distractor tableand trainingTokenizer revision mismatch"
                )
            if self.semantic_corruption.top_k > semantic_distractor_table.top_k:
                raise ValueError("semantic corruption top_k exceeds distractor asset capacity")
            semantic_embedding_provenance = (
                conditioner.semantic_embedding_asset_provenance
            )
            if (
                not isinstance(semantic_embedding_provenance, Mapping)
                or semantic_embedding_provenance.get("source")
                != "tokenizer_effective_codebook"
                or semantic_distractor_table.provenance.get(
                    "source_embedding_tensor_sha256"
                )
                != semantic_embedding_provenance.get("tensor_sha256")
            ):
                raise RuntimeError(
                    "The EMDC neighbor table must be bound to the tokenizer codebook "
                    "used by the current conditioner"
                )
            semantic_error_calibration_provenance = verified_calibration
        elif semantic_distractor_table is not None:
            raise ValueError("semantic_corruption.mode=none must not receive a distractor table")
        elif semantic_error_calibration_provenance is not None:
            raise ValueError("semantic_corruption.mode=none must not receive semantic calibration provenance")
        self.semantic_distractor_table = semantic_distractor_table
        self.semantic_error_calibration_provenance = (
            dict(semantic_error_calibration_provenance)
            if semantic_error_calibration_provenance is not None
            else None
        )
        self.logger = logger
        self.config = dict(config or {})
        self.consistency_flow_matching = ConsistencyFlowMatchingConfig.from_mapping(
            (self.config.get("train") or {}).get("consistency_flow_matching")
        )
        fixed_flow_training = self.config.get("train", {}).get("fixed_flow_training")
        self.fixed_flow_training = (
            dict(fixed_flow_training)
            if isinstance(fixed_flow_training, Mapping)
            else None
        )
        self.distributed_state = dict(
            distributed_state
            or {
                "format_version": TRAINING_TOPOLOGY_VERSION,
                "world_size": self.world_size,
                "config_hash": config_hash(self.config),
            }
        )
        self.require_cached_text = bool(require_cached_text)
        self.precision = precision
        self.deterministic_manual_gradient_reduce = bool(
            self.config.get("train", {}).get(
                "deterministic_manual_gradient_reduce",
                False,
            )
        )
        self.frozen_modules = dict(frozen_modules or {})
        self.frozen_modules.setdefault("qwen_text_encoder", conditioner.text_encoder)
        for name, module in self.frozen_modules.items():
            freeze_module(module, name=name)
        parameters = [
            parameter
            for module in (self.model, self.conditioner)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("DiT trainer has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(learning_rate),
            betas=tuple(float(value) for value in betas),
            eps=float(optimizer_eps),
            weight_decay=float(weight_decay),
            amsgrad=optimizer_amsgrad,
            maximize=optimizer_maximize,
            foreach=optimizer_foreach,
            capturable=optimizer_capturable,
            differentiable=optimizer_differentiable,
            fused=optimizer_fused,
        )
        self.scheduler = build_linear_warmup_scheduler(
            self.optimizer,
            warmup_steps=int(warmup_steps),
            max_steps=int(max_steps),
            scheduler=str(scheduler),
            min_lr_ratio=float(min_lr_ratio),
            step_indexing=str(scheduler_step_indexing),
        )
        self.ema = (
            TrainableParameterEMA(
                self.model,
                self.conditioner,
                decay=float(ema_decay),
            )
            if ema_mode == "parameter_fp32"
            else None
        )
        self.assert_optimizer_parameter_identity()

        rank_seed_offset = self.rank * int(rank_seed_stride)
        self.source_generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + rank_seed_offset + int(flow_source_seed_offset)
        )
        self.timestep_generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + rank_seed_offset + int(flow_timestep_seed_offset)
        )
        self.text_drop_generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + rank_seed_offset + int(text_drop_seed_offset)
        )
        self.global_step = 0
        self.consumed_audio_seconds = 0.0
        self.epoch = 0
        self.batches_consumed = 0
        self.sampler_state: dict[str, Any] | None = None
        self.validation_runs = 0
        self.best_valid_loss = float("inf")
        self.best_valid_step: int | None = None
        self.valid_without_improvement = 0
        self.last_valid_loss: float | None = None
        self.training_extension: dict[str, Any] | None = None
        self._pending_post_ddp_rng_state: dict[str, Any] | None = None
        self._checkpoint_load_failed = False
        self.checkpoint_adapter = checkpoint_adapter
        if checkpoint_adapter is not None:
            validate_checkpoint_adapter(checkpoint_adapter)
            revision_validator = getattr(checkpoint_adapter, "validate_revisions", None)
            if callable(revision_validator):
                revision_validator(asdict(self.revisions))

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def unwrapped_model(self) -> nn.Module:
        return self.model

    @property
    def unwrapped_conditioner(self) -> RenderConditioner:
        if not isinstance(self.conditioner, RenderConditioner):
            raise TypeError("trainer conditioner root must be RenderConditioner")
        return self.conditioner

    @property
    def unwrapped_training_graph(self) -> RenderDiTTrainingGraph:
        graph = unwrap_model(self.training_graph)
        if not isinstance(graph, RenderDiTTrainingGraph):
            raise TypeError("trainer training graph root has an invalid type")
        if (
            graph.model is not self.unwrapped_model
            or graph.conditioner is not self.unwrapped_conditioner
        ):
            raise RuntimeError("training graph and checkpoint roots have different identities")
        return graph

    @property
    def ddp_wrapped(self) -> bool:
        return isinstance(self.training_graph, DistributedDataParallel)

    def assert_optimizer_parameter_identity(self) -> None:

        expected = {
            id(parameter)
            for module in (self.unwrapped_model, self.unwrapped_conditioner)
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        observed_list = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]
        observed = {id(parameter) for parameter in observed_list}
        if len(observed) != len(observed_list):
            raise RuntimeError("optimizer parameter list contains duplicate objects")
        if observed != expected:
            raise RuntimeError(
                "optimizer parameters and current DiT training graph mismatch;"
                "model may be in optimizer occurred after creation device/dtype parameter replacement"
            )

    def _autocast_context(self):
        if self.precision == "fp32":
            return nullcontext()
        if self.precision != "bf16":
            raise AssertionError("Construction phase should have rejected unknown precision")
        if self.device.type not in {"cpu", "cuda"}:
            raise RuntimeError(f"Device {self.device.type!r} does not support the current BF16 autocast")
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)

    def _assert_checkpoint_state_usable(self) -> None:
        if self._checkpoint_load_failed:
            raise RuntimeError(
                "Checkpoint loading previously failed; discard this trainer and "
                "restart the process"
            )

    def wrap_distributed(self, *, local_rank: int, resume: bool) -> None:
        if self.world_size == 1:
            return
        if not dist.is_initialized():
            raise RuntimeError("world_size > 1 but process group not initialized")
        if (
            isinstance(self.training_graph, DistributedDataParallel)
            or isinstance(self.model, DistributedDataParallel)
            or isinstance(self.conditioner, DistributedDataParallel)
        ):
            raise RuntimeError(
                "Render training graph must not be duplicated or wrapped by multiple DDP instances"
            )
        device = self.device
        common: dict[str, Any] = {
            "device_ids": [int(local_rank)] if device.type == "cuda" else None,
            "output_device": int(local_rank) if device.type == "cuda" else None,
            "find_unused_parameters": bool(
                self.config.get("train", {}).get(
                    "ddp_find_unused_parameters",
                    False,
                )
            ),
            "broadcast_buffers": bool(
                self.config.get("train", {}).get(
                    "ddp_broadcast_buffers",
                    False,
                )
            ),
            "bucket_cap_mb": float(
                self.config.get("train", {}).get("ddp_bucket_cap_mb", 25.0)
            ),
            "gradient_as_bucket_view": bool(
                self.config.get("train", {}).get("ddp_gradient_as_bucket_view", True)
            ),
            "static_graph": bool(
                self.config.get("train", {}).get("ddp_static_graph", False)
            ),
        }
        if (
            common["static_graph"]
            and int(self.config.get("train", {}).get("gradient_accumulation_steps", 1))
            > 1
        ):
            raise ValueError(
                "ddp_static_graph=Trueandgradient_accumulation_steps>1 is incompatible"
            )
        if (
            "init_sync"
            in inspect.signature(DistributedDataParallel.__init__).parameters
        ):
            init_sync_policy = str(
                self.config.get("train", {}).get(
                    "ddp_init_sync_policy",
                    "fresh_only",
                )
            )
            if init_sync_policy != "fresh_only":
                raise ValueError("ddp_init_sync_policy only supports fresh_only")
            common["init_sync"] = not bool(resume)


        self.training_graph = DistributedDataParallel(self.training_graph, **common)
        if self.ema is not None and not resume:


            self.ema.reset_from_parameters()
        self.assert_optimizer_parameter_identity()

    def _cache_provenance(
        self, batch: Mapping[str, Any]
    ) -> TextEncoderProvenance | None:
        if "description_embeddings" not in batch and "lyrics_embeddings" not in batch:
            return None
        return self.revisions.text_provenance(
            self.unwrapped_conditioner.provenance.model_id
        )

    @staticmethod
    def _stable_validation_seed(
        base_seed: int,
        *,
        sample_id: str,
        stream: str,
    ) -> int:
        payload = f"{int(base_seed)}:{stream}:{sample_id}".encode()

        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)

    def _fixed_validation_inputs(
        self,
        values: Mapping[str, Any],
        *,
        validation_seed: int,
        text_drop_probability: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_ids = values.get("sample_ids")
        batch = int(values["latents"].shape[0])
        if (
            not isinstance(sample_ids, Sequence)
            or isinstance(sample_ids, (str, bytes))
            or len(sample_ids) != batch
            or not all(isinstance(value, str) and value for value in sample_ids)
        ):
            raise ValueError("fixed validation requires each sample to carry a non-empty sample_id")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("fixed validation batch sample_id must be unique")
        source = torch.zeros_like(values["latents"])
        timestep = torch.empty(
            batch,
            device=values["latents"].device,
            dtype=torch.float32,
        )
        text_drop = torch.empty(
            batch,
            device=values["latents"].device,
            dtype=torch.bool,
        )
        lengths = values["semantic_mask"].sum(dim=1).detach().cpu().tolist()
        for index, (sample_id, length) in enumerate(
            zip(sample_ids, lengths, strict=True)
        ):
            source_generator = torch.Generator(device="cpu").manual_seed(
                self._stable_validation_seed(
                    validation_seed,
                    sample_id=sample_id,
                    stream="flow_source",
                )
            )
            source_row = sample_source_like(
                values["latents"][index : index + 1, : int(length)],
                self.flow.config,
                generator=source_generator,
            )
            source[index, : int(length)] = source_row[0]
            timestep_generator = torch.Generator(device="cpu").manual_seed(
                self._stable_validation_seed(
                    validation_seed,
                    sample_id=sample_id,
                    stream="flow_timestep",
                )
            )
            timestep[index] = sample_timesteps(
                1,
                self.flow.config,
                reference=values["latents"][index : index + 1],
                generator=timestep_generator,
                effective_lengths=torch.tensor(
                    [int(length)],
                    dtype=torch.int64,
                    device=values["latents"].device,
                ),
            )[0]
            text_generator = torch.Generator(device="cpu").manual_seed(
                self._stable_validation_seed(
                    validation_seed,
                    sample_id=sample_id,
                    stream="text_drop",
                )
            )
            text_drop[index] = bool(
                torch.rand((), generator=text_generator) < float(text_drop_probability)
            )
        return source, timestep, text_drop

    def _fixed_training_inputs(
        self,
        values: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        fixed = self.fixed_flow_training
        if fixed is None:
            raise RuntimeError("fixed_flow_training is not enabled")
        sample_ids = values.get("sample_ids")
        batch = int(values["latents"].shape[0])
        if (
            not isinstance(sample_ids, Sequence)
            or isinstance(sample_ids, (str, bytes))
            or len(sample_ids) != batch
            or not all(isinstance(value, str) and value for value in sample_ids)
        ):
            raise ValueError("fixed flow training requires each sample to carry a non-empty sample_id")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("fixed flow training batch sample_id must be unique")
        grid = tuple(float(value) for value in fixed["timestep_grid"])
        source = torch.zeros_like(values["latents"])
        timestep = torch.empty(
            batch,
            device=values["latents"].device,
            dtype=torch.float32,
        )
        text_drop = torch.zeros(
            batch,
            device=values["latents"].device,
            dtype=torch.bool,
        )
        lengths = values["semantic_mask"].sum(dim=1).detach().cpu().tolist()
        for index, (sample_id, length) in enumerate(
            zip(sample_ids, lengths, strict=True)
        ):
            source_generator = torch.Generator(device="cpu").manual_seed(
                self._stable_validation_seed(
                    int(fixed["source_seed"]),
                    sample_id=sample_id,
                    stream=str(fixed["source_stream"]),
                )
            )
            source_row = sample_source_like(
                values["latents"][index : index + 1, : int(length)],
                self.flow.config,
                generator=source_generator,
            )
            source[index, : int(length)] = source_row[0]
            offset = self._stable_validation_seed(
                int(fixed["source_seed"]),
                sample_id=sample_id,
                stream="fixed_flow_timestep_grid_offset",
            ) % len(grid)
            timestep[index] = grid[(self.global_step + offset) % len(grid)]
        return source, timestep, text_drop

    def compute_loss(
        self,
        batch: Any,
        *,
        validation_seed: int | None = None,
        validation_text_drop_probability: float | None = None,
        use_distributed_graph: bool | None = None,
    ) -> TrainingLossOutput:
        self._assert_checkpoint_state_usable()
        source_semantic_ids = _batch_get(batch, "semantic_ids")
        source_semantic_mask = _batch_get(batch, "semantic_mask")
        values = validate_dit_batch(
            move_batch_to_device(batch, self.device),
            expected_revisions=self.revisions,
        )
        model_dtype = next(self.unwrapped_model.parameters()).dtype
        values["latents"] = values["latents"].to(dtype=model_dtype)
        text_dtype = self.unwrapped_conditioner.description_projection.weight.dtype
        for name in ("description_embeddings", "lyrics_embeddings"):
            if name in values:
                values[name] = values[name].to(dtype=text_dtype)
        if self.require_cached_text and (
            "description_embeddings" not in values or "lyrics_embeddings" not in values
        ):
            raise RuntimeError(
                "DiT training requires precomputed description and lyrics embeddings; "
                "online text encoding is unavailable in the training loop"
            )
        frame_mask = values["semantic_mask"]
        semantic_ids = values["semantic_ids"]
        semantic_replacement_mask = torch.zeros_like(
            semantic_ids,
            dtype=torch.bool,
        )
        semantic_replacement_probability = 0.0
        if validation_seed is None:
            if self.fixed_flow_training is None:
                flow_sample = self.flow.prepare(
                    values["latents"],
                    source_generator=self.source_generator,
                    timestep_generator=self.timestep_generator,
                    frame_mask=frame_mask,
                )
                text_drop_mask = self.unwrapped_conditioner.sample_text_drop_mask(
                    values["latents"].shape[0],
                    self.text_drop_probability,
                    generator=self.text_drop_generator,
                    device=self.device,
                )
            else:
                source, timestep, text_drop_mask = self._fixed_training_inputs(values)
                flow_sample = self.flow.prepare(
                    values["latents"],
                    source=source,
                    timestep=timestep,
                    frame_mask=frame_mask,
                    apply_source_coupling=False,
                )
            if self.semantic_corruption.mode == "emdc_knn":
                if self.semantic_distractor_table is None:
                    raise AssertionError("emdc_knn is missing a verified distractor table")


                corruption_ids = (
                    source_semantic_ids
                    if isinstance(source_semantic_ids, torch.Tensor)
                    and source_semantic_ids.device.type == "cpu"
                    else semantic_ids
                )
                corruption_mask = (
                    source_semantic_mask
                    if isinstance(source_semantic_mask, torch.Tensor)
                    and source_semantic_mask.device == corruption_ids.device
                    else frame_mask
                )
                corruption = self.semantic_distractor_table.corrupt(
                    corruption_ids,
                    corruption_mask,
                    sample_ids=values["sample_ids"],
                    global_step=self.global_step,
                    config=self.semantic_corruption,
                )
                semantic_ids = corruption.semantic_ids.to(
                    device=self.device,
                    non_blocking=True,
                )
                semantic_replacement_mask = corruption.replacement_mask.to(
                    device=self.device,
                    non_blocking=True,
                )
                semantic_replacement_probability = corruption.probability
        else:
            probability = (
                self.text_drop_probability
                if validation_text_drop_probability is None
                else validation_text_drop_probability
            )
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0.0 <= probability <= 1.0
            ):
                raise ValueError("validation_text_drop_probability must be a finite value in [0, 1]")
            source, timestep, text_drop_mask = self._fixed_validation_inputs(
                values,
                validation_seed=validation_seed,
                text_drop_probability=float(probability),
            )
            flow_sample = self.flow.prepare(
                values["latents"],
                source=source,
                timestep=timestep,
                frame_mask=frame_mask,


                apply_source_coupling=False,
            )
        if use_distributed_graph is None:
            use_distributed_graph = not self.deterministic_manual_gradient_reduce
        graph = (
            self.training_graph
            if use_distributed_graph
            else self.unwrapped_training_graph
        )
        effective_consistency_config = self.consistency_flow_matching.for_step(
            self.global_step
        )
        consistency_pair = (
            make_consistency_flow_matching_pair(
                flow_sample,
                self.flow.config,
                effective_consistency_config,
            )
            if validation_seed is None and self.consistency_flow_matching.enabled
            else None
        )
        with self._autocast_context():
            graph_output = graph(
                flow_sample.noisy_latents,
                flow_sample.timestep,
                semantic_ids=semantic_ids,
                frame_mask=frame_mask,
                description_input_ids=values.get("description_input_ids"),
                description_mask=values["description_mask"],
                lyrics_input_ids=values.get("lyrics_input_ids"),
                lyrics_mask=values["lyrics_mask"],
                description_embeddings=values.get("description_embeddings"),
                lyrics_embeddings=values.get("lyrics_embeddings"),
                cache_provenance=self._cache_provenance(values),
                text_drop_mask=text_drop_mask,
                global_loudness_lufs=values.get("global_loudness_lufs"),
                neighbor_latents=(
                    consistency_pair.neighbor_latents
                    if consistency_pair is not None
                    and consistency_pair.requires_neighbor_prediction
                    else None
                ),
                neighbor_timestep=(
                    consistency_pair.neighbor_timestep
                    if consistency_pair is not None
                    and consistency_pair.requires_neighbor_prediction
                    else None
                ),
                share_model_rng=(
                    consistency_pair is not None
                    and consistency_pair.requires_neighbor_prediction
                ),
            )
        neighbor_prediction: torch.Tensor | None
        if isinstance(graph_output, tuple):
            prediction, neighbor_prediction = graph_output
        else:
            prediction = graph_output
            neighbor_prediction = None
        if consistency_pair is None:
            masked = self.flow.loss(prediction, flow_sample.target_velocity, frame_mask)
            endpoint_consistency_numerator = None
            endpoint_consistency_denominator = None
            velocity_consistency_numerator = None
            velocity_consistency_denominator = None
            training_objective = "flow_matching"
        else:
            consistency_loss = consistency_flow_matching_loss(
                prediction,
                neighbor_prediction,
                flow_sample,
                consistency_pair,
                frame_mask,
                flow_config=self.flow.config,
                consistency_config=effective_consistency_config,
            )
            masked = consistency_loss.combined
            endpoint_consistency_numerator = consistency_loss.endpoint.numerator
            endpoint_consistency_denominator = consistency_loss.endpoint.denominator
            velocity_consistency_numerator = consistency_loss.velocity.numerator
            velocity_consistency_denominator = consistency_loss.velocity.denominator
            training_objective = "consistency_flow_matching"

        with torch.no_grad():
            bin_numerators, bin_denominators = timestep_binned_mse(
                prediction.detach(),
                flow_sample.target_velocity.detach(),
                frame_mask,
                flow_sample.timestep.detach(),
            )
            if validation_seed is not None:
                channel_numerators, channel_denominator = channelwise_mse_sums(
                    prediction.detach(),
                    flow_sample.target_velocity.detach(),
                    frame_mask,
                )
                (
                    timestep_channel_numerators,
                    timestep_channel_denominators,
                ) = timestep_channelwise_mse_sums(
                    prediction.detach(),
                    flow_sample.target_velocity.detach(),
                    frame_mask,
                    flow_sample.timestep.detach(),
                )
            else:
                channel_numerators = None
                channel_denominator = None
                timestep_channel_numerators = None
                timestep_channel_denominators = None
        return TrainingLossOutput(
            masked_loss=masked,
            prediction=prediction,
            target_velocity=flow_sample.target_velocity,
            text_drop_mask=text_drop_mask,
            timestep=flow_sample.timestep,
            timestep_bin_numerators=bin_numerators,
            timestep_bin_denominators=bin_denominators,
            channel_mse_numerators=channel_numerators,
            channel_mse_denominator=channel_denominator,
            timestep_channel_mse_numerators=timestep_channel_numerators,
            timestep_channel_mse_denominators=timestep_channel_denominators,
            semantic_replacement_mask=semantic_replacement_mask,
            semantic_replacement_probability=semantic_replacement_probability,
            source_coupling_cost_before=flow_sample.source_coupling_cost_before,
            source_coupling_cost_after=flow_sample.source_coupling_cost_after,
            source_coupling_batch_size=flow_sample.source_coupling_batch_size,
            source_coupling_scope=flow_sample.source_coupling_scope,
            training_objective=training_objective,
            endpoint_consistency_numerator=endpoint_consistency_numerator,
            endpoint_consistency_denominator=endpoint_consistency_denominator,
            velocity_consistency_numerator=velocity_consistency_numerator,
            velocity_consistency_denominator=velocity_consistency_denominator,
        )

    @torch.no_grad()
    def validate(
        self,
        batches: Iterable[Any],
        *,
        validation_seed: int,
        text_drop_probability: float = 0.0,
        expected_sample_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:

        self._assert_checkpoint_state_usable()
        if not isinstance(validation_seed, int) or isinstance(validation_seed, bool):
            raise TypeError("validation_seed must be an integer")
        frozen_sample_ids: tuple[str, ...] | None = None
        frozen_sample_ids_sha256: str | None = None
        if expected_sample_ids is not None:
            if (
                not isinstance(expected_sample_ids, Sequence)
                or isinstance(expected_sample_ids, (str, bytes))
                or not expected_sample_ids
                or not all(
                    isinstance(value, str) and value for value in expected_sample_ids
                )
            ):
                raise TypeError("expected_sample_ids must be a non-empty string sequence")
            frozen_sample_ids = tuple(expected_sample_ids)
            if len(set(frozen_sample_ids)) != len(frozen_sample_ids):
                raise ValueError("Frozen validation sample IDs contain duplicates")
            frozen_sample_ids_sha256 = _sha256_json(sorted(frozen_sample_ids))
        previous_training = self.training_graph.training
        self.training_graph.eval()
        local_sample_ids: list[str] = []
        local_seen_sample_ids: set[str] = set()
        local_numerator = 0.0
        local_denominator = 0.0
        local_valid_frames = 0.0
        local_text_drop_count = 0.0
        local_batch_size = 0.0
        local_timestep_sum = 0.0
        local_timestep_count = 0.0
        local_batches = 0.0
        local_timestep_bin_numerators = [0.0] * 10
        local_timestep_bin_denominators = [0.0] * 10
        local_channel_numerators = torch.zeros(LATENT_DIM, dtype=torch.float64)
        local_channel_denominator = torch.zeros((), dtype=torch.float64)
        local_timestep_channel_numerators = torch.zeros(
            10, LATENT_DIM, dtype=torch.float64
        )
        local_timestep_channel_denominators = torch.zeros(10, dtype=torch.float64)
        validation_error: str | None = None
        try:
            for batch in batches:
                sample_ids = _batch_get(batch, "sample_ids")
                if (
                    not isinstance(sample_ids, Sequence)
                    or isinstance(sample_ids, (str, bytes))
                    or not sample_ids
                    or not all(isinstance(value, str) and value for value in sample_ids)
                ):
                    raise ValueError("fixed validation requires each batch to contain a non-empty sample_id")
                repeated = sorted(set(sample_ids) & local_seen_sample_ids)
                if repeated:
                    raise ValueError(
                        f"fixed validation contains duplicate sample IDs across batches: {repeated[:8]}"
                    )
                local_seen_sample_ids.update(sample_ids)
                local_sample_ids.extend(sample_ids)
                output = self.compute_loss(
                    batch,
                    validation_seed=validation_seed,
                    validation_text_drop_probability=text_drop_probability,
                    use_distributed_graph=False,
                )
                local_numerator += float(output.masked_loss.numerator)
                local_denominator += float(output.masked_loss.denominator)
                local_valid_frames += float(output.masked_loss.valid_frames)
                local_text_drop_count += float(output.text_drop_mask.sum())
                local_batch_size += float(output.text_drop_mask.numel())
                local_timestep_sum += float(output.timestep.float().sum())
                local_timestep_count += float(output.timestep.numel())
                local_batches += 1.0
                assert output.timestep_bin_numerators is not None
                assert output.timestep_bin_denominators is not None
                assert output.channel_mse_numerators is not None
                assert output.channel_mse_denominator is not None
                assert output.timestep_channel_mse_numerators is not None
                assert output.timestep_channel_mse_denominators is not None
                local_channel_numerators += output.channel_mse_numerators.cpu()
                local_channel_denominator += output.channel_mse_denominator.cpu()
                local_timestep_channel_numerators += (
                    output.timestep_channel_mse_numerators.cpu()
                )
                local_timestep_channel_denominators += (
                    output.timestep_channel_mse_denominators.cpu()
                )
                for bin_index in range(10):
                    local_timestep_bin_numerators[bin_index] += float(
                        output.timestep_bin_numerators[bin_index]
                    )
                    local_timestep_bin_denominators[bin_index] += float(
                        output.timestep_bin_denominators[bin_index]
                    )
        except Exception as exc:
            validation_error = f"{type(exc).__name__}: {exc}"
        finally:
            self.training_graph.train(previous_training)
            for module in self.frozen_modules.values():
                module.eval()
        assert_distributed_consensus(
            "render_fixed_validation",
            {"status": "error" if validation_error else "ok"},
        )
        if validation_error is not None:
            raise RuntimeError(f"Render fixed validation failed: {validation_error}")

        gathered_sample_ids = gather_object_to_rank0(
            {"rank": self.rank, "sample_ids": local_sample_ids}
        )
        sample_identity_error: str | None = None
        if self.rank == 0:
            try:
                assert gathered_sample_ids is not None
                expected_ranks = set(range(self.world_size))
                observed_ranks = {int(item["rank"]) for item in gathered_sample_ids}
                if observed_ranks != expected_ranks:
                    raise RuntimeError(
                        "fixed validation sample_id collectrank is incomplete:"
                        f"expected={sorted(expected_ranks)} "
                        f"actual={sorted(observed_ranks)}"
                    )
                global_sample_ids = [
                    sample_id
                    for item in gathered_sample_ids
                    for sample_id in item["sample_ids"]
                ]
                counts: dict[str, int] = {}
                for sample_id in global_sample_ids:
                    counts[sample_id] = counts.get(sample_id, 0) + 1
                duplicates = sorted(
                    sample_id for sample_id, count in counts.items() if count != 1
                )
                if duplicates:
                    raise RuntimeError(
                        f"fixed validation contains duplicate sample IDs across ranks: {duplicates[:8]}"
                    )
                if frozen_sample_ids is not None and (
                    set(global_sample_ids) != set(frozen_sample_ids)
                    or len(global_sample_ids) != len(frozen_sample_ids)
                ):
                    missing = sorted(set(frozen_sample_ids) - set(global_sample_ids))
                    unexpected = sorted(set(global_sample_ids) - set(frozen_sample_ids))
                    raise RuntimeError(
                        "fixed validationInaccurate coverage freezesample_idCollection:"
                        f"expected={len(frozen_sample_ids)} "
                        f"actual={len(global_sample_ids)} "
                        f"missing={missing[:8]} unexpected={unexpected[:8]}"
                    )
            except Exception as exc:
                sample_identity_error = f"{type(exc).__name__}: {exc}"
        raise_if_rank0_error(
            sample_identity_error,
            action="fixed validation sample identity",
        )

        global_numerator = reduce_scalar_sum(local_numerator)
        global_denominator = reduce_scalar_sum(local_denominator)
        global_valid_frames = reduce_scalar_sum(local_valid_frames)
        global_text_drop_count = reduce_scalar_sum(local_text_drop_count)
        global_batch_size = reduce_scalar_sum(local_batch_size)
        global_timestep_sum = reduce_scalar_sum(local_timestep_sum)
        global_timestep_count = reduce_scalar_sum(local_timestep_count)
        global_batches = reduce_scalar_sum(local_batches)
        global_bin_numerators = [
            reduce_scalar_sum(value) for value in local_timestep_bin_numerators
        ]
        global_bin_denominators = [
            reduce_scalar_sum(value) for value in local_timestep_bin_denominators
        ]
        global_channel_numerators = reduce_tensor_sum(local_channel_numerators)
        global_channel_denominator = reduce_tensor_sum(local_channel_denominator)
        global_timestep_channel_numerators = reduce_tensor_sum(
            local_timestep_channel_numerators
        )
        global_timestep_channel_denominators = reduce_tensor_sum(
            local_timestep_channel_denominators
        )
        if (
            global_denominator <= 0
            or global_timestep_count <= 0
            or global_batch_size <= 0
        ):
            raise RuntimeError("fixed validation: there are no valid samples or frames to summarize")
        metrics: dict[str, Any] = {
            "event": "valid",
            "global_step": self.global_step,
            "validation_run": self.validation_runs + 1,
            "loss": global_numerator / global_denominator,
            "loss_numerator": global_numerator,
            "loss_denominator": global_denominator,
            "valid_frames": int(global_valid_frames),
            "batch_size": int(global_batch_size),
            "batches": int(global_batches),
            "text_drop_count": int(global_text_drop_count),
            "text_drop_probability": float(text_drop_probability),
            "timestep_mean": global_timestep_sum / global_timestep_count,
            "validation_seed": validation_seed,
            "precision": self.precision,
            "flow_format_version": self.flow.config.format_version,
            "world_size": self.world_size,
        }
        channel_mse = global_channel_numerators / global_channel_denominator
        metrics["channel_velocity_mse"] = [float(value) for value in channel_mse]
        metrics["worst_channel_velocity_mse"] = float(channel_mse.max())
        metrics["worst_channel_index"] = int(channel_mse.argmax())
        metrics["timestep_channel_velocity_mse"] = {
            f"{bin_index / 10:.1f}_{(bin_index + 1) / 10:.1f}": (
                [
                    float(value)
                    for value in (
                        global_timestep_channel_numerators[bin_index]
                        / global_timestep_channel_denominators[bin_index]
                    )
                ]
                if global_timestep_channel_denominators[bin_index] > 0
                else None
            )
            for bin_index in range(10)
        }
        if frozen_sample_ids is not None:
            metrics["sample_ids_count"] = len(frozen_sample_ids)
            metrics["sample_ids_sha256"] = frozen_sample_ids_sha256
        for bin_index, (numerator, denominator) in enumerate(
            zip(global_bin_numerators, global_bin_denominators)
        ):
            metrics[
                f"timestep_loss_{bin_index / 10:.1f}_{(bin_index + 1) / 10:.1f}"
            ] = numerator / denominator if denominator > 0 else None
            metrics[
                f"timestep_frames_{bin_index / 10:.1f}_{(bin_index + 1) / 10:.1f}"
            ] = int(denominator)
        self.validation_runs += 1
        self.last_valid_loss = float(metrics["loss"])
        return metrics

    def _preflight_batch(self, batch: Any) -> None:
        values = validate_dit_batch(
            move_batch_to_device(batch, torch.device("cpu")),
            expected_revisions=self.revisions,
        )
        if self.require_cached_text and (
            "description_embeddings" not in values or "lyrics_embeddings" not in values
        ):
            raise RuntimeError("DiT training requires cached description and lyrics embeddings")

    def _backward_context(self, *, synchronize: bool) -> ExitStack:
        stack = ExitStack()
        if self.world_size > 1:
            if not self.ddp_wrapped:
                raise RuntimeError(
                    "world_size > 1 when training graph must go through a single DDP wrapping"
                )
            if self.deterministic_manual_gradient_reduce or not synchronize:


                stack.enter_context(self.training_graph.no_sync())
        return stack

    def _manual_all_reduce_gradients(self, parameters: Sequence[nn.Parameter]) -> None:

        if not self.deterministic_manual_gradient_reduce or self.world_size == 1:
            return
        if not dist.is_initialized():
            raise RuntimeError("Manual gradient merging requires an initialized distributed process group")
        missing = [
            index
            for index, parameter in enumerate(parameters)
            if parameter.requires_grad and parameter.grad is None
        ]
        if missing:
            raise RuntimeError(
                "Manual gradient merging requires dense gradients for all trainable parameters:"
                f"missing_indices={missing[:8]}"
            )
        groups: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("Manual gradient regression does not support sparse gradients")
            groups.setdefault(
                (gradient.device, gradient.dtype),
                [],
            ).append(gradient)
        for gradients in groups.values():
            flattened = torch.cat(
                [gradient.detach().reshape(-1) for gradient in gradients]
            )
            dist.all_reduce(flattened, op=dist.ReduceOp.SUM)
            offset = 0
            for gradient in gradients:
                elements = gradient.numel()
                gradient.copy_(flattened[offset : offset + elements].view_as(gradient))
                offset += elements
            if offset != flattened.numel():
                raise AssertionError("Manual gradient merge flat-buffer segmentation is incomplete")

    def _apply_semantic_embedding_freeze(self) -> bool:

        if self.semantic_embedding_freeze_policy != "first_n_optimizer_updates":
            raise AssertionError("Construction phase should have rejected unknown semantic freeze strategy")
        frozen = self.global_step < self.semantic_embedding_freeze_steps
        embedding = self.unwrapped_conditioner.semantic_embedding.weight
        if frozen and embedding.requires_grad:

            embedding.grad = None
        return frozen

    def train_update(self, microbatches: Iterable[Any]) -> dict[str, Any]:

        self._assert_checkpoint_state_usable()
        batches = list(microbatches)
        preflight_error: str | None = None
        try:
            if not batches:
                raise ValueError("train_update requires at least one micro-batch")
            if self.world_size > 1 and not self.ddp_wrapped:
                raise RuntimeError(
                    "distributed train_update is missing the single DDP training graph"
                )
            for batch in batches:
                self._preflight_batch(batch)
        except Exception as exc:
            preflight_error = f"{type(exc).__name__}: {exc}"

        assert_distributed_consensus(
            "render_train_update_preflight",
            {
                "microbatch_count": len(batches),
                "error": preflight_error,
            },
        )
        if preflight_error is not None:
            raise RuntimeError(f"Render train_update preflight failed: {preflight_error}")

        self.training_graph.train()
        self.assert_optimizer_parameter_identity()
        for module in self.frozen_modules.values():
            module.eval()
        self.optimizer.zero_grad(set_to_none=self.optimizer_zero_grad_set_to_none)
        local_numerator = 0.0
        local_denominator = 0.0
        local_valid_frames = 0.0
        local_audio_seconds = 0.0
        local_text_drop_count = 0.0
        local_batch_size = 0.0
        local_timestep_sum = 0.0
        local_timestep_count = 0.0
        local_semantic_replacements = 0.0
        local_semantic_eligible = 0.0
        semantic_replacement_probability: float | None = None
        local_timestep_bin_numerators = [0.0] * 10
        local_timestep_bin_denominators = [0.0] * 10
        local_source_coupling_cost_before = 0.0
        local_source_coupling_cost_after = 0.0
        local_source_coupling_count = 0.0
        local_endpoint_consistency_numerator = 0.0
        local_endpoint_consistency_denominator = 0.0
        local_velocity_consistency_numerator = 0.0
        local_velocity_consistency_denominator = 0.0
        training_objective = (
            "consistency_flow_matching"
            if self.consistency_flow_matching.enabled
            else "flow_matching"
        )
        effective_consistency_config = self.consistency_flow_matching.for_step(
            self.global_step
        )
        source_coupling_batch_size: int | None = None
        source_coupling_scope: str | None = None
        for index, batch in enumerate(batches):
            synchronize = index + 1 == len(batches)
            with self._backward_context(synchronize=synchronize):
                output = self.compute_loss(batch)


                output.masked_loss.numerator.backward()
            if output.training_objective != training_objective:
                raise RuntimeError("Within one optimizer update, the training objective must be consistent")
            local_numerator += float(output.masked_loss.numerator.detach())
            local_denominator += float(output.masked_loss.denominator.detach())
            local_valid_frames += float(output.masked_loss.valid_frames.detach())
            frame_mask = _batch_get(batch, "semantic_mask")
            local_audio_seconds += DurationBucketPolicy.audio_seconds(frame_mask)
            local_text_drop_count += float(output.text_drop_mask.sum().detach())
            local_batch_size += float(output.text_drop_mask.numel())
            local_timestep_sum += float(output.timestep.detach().float().sum())
            local_timestep_count += float(output.timestep.numel())
            consistency_fields = (
                output.endpoint_consistency_numerator,
                output.endpoint_consistency_denominator,
                output.velocity_consistency_numerator,
                output.velocity_consistency_denominator,
            )
            if training_objective == "consistency_flow_matching":
                if any(value is None for value in consistency_fields):
                    raise RuntimeError("CFM training is missing composition-loss telemetry")
                local_endpoint_consistency_numerator += float(
                    output.endpoint_consistency_numerator.detach()
                )
                local_endpoint_consistency_denominator += float(
                    output.endpoint_consistency_denominator.detach()
                )
                local_velocity_consistency_numerator += float(
                    output.velocity_consistency_numerator.detach()
                )
                local_velocity_consistency_denominator += float(
                    output.velocity_consistency_denominator.detach()
                )
            elif any(value is not None for value in consistency_fields):
                raise RuntimeError("Standard FM training unexpectedly produced CFM telemetry")
            if output.source_coupling_cost_before is not None:
                if output.source_coupling_cost_after is None:
                    raise RuntimeError("source coupling cost field is incomplete")


                if output.source_coupling_scope != "global" or self.rank == 0:
                    local_source_coupling_cost_before += float(
                        output.source_coupling_cost_before.detach()
                    )
                    local_source_coupling_cost_after += float(
                        output.source_coupling_cost_after.detach()
                    )
                    local_source_coupling_count += 1.0
                if source_coupling_batch_size is None:
                    source_coupling_batch_size = int(output.source_coupling_batch_size)
                    source_coupling_scope = output.source_coupling_scope
                elif (
                    source_coupling_batch_size != output.source_coupling_batch_size
                    or source_coupling_scope != output.source_coupling_scope
                ):
                    raise RuntimeError("Within one optimizer update, the OT scope must be consistent")
            if output.semantic_replacement_mask is not None:
                local_semantic_replacements += float(
                    output.semantic_replacement_mask.sum().detach()
                )
            local_semantic_eligible += float(_batch_get(batch, "semantic_mask").sum())
            if semantic_replacement_probability is None:
                semantic_replacement_probability = float(
                    output.semantic_replacement_probability
                )
            elif semantic_replacement_probability != float(
                output.semantic_replacement_probability
            ):
                raise RuntimeError(
                    "Within one optimizer update, semantic corruption probabilities must be consistent"
                )
            if (
                output.timestep_bin_numerators is None
                or output.timestep_bin_denominators is None
            ):
                bin_numerators, bin_denominators = timestep_binned_mse(
                    output.prediction,
                    output.target_velocity,
                    _batch_get(batch, "semantic_mask"),
                    output.timestep,
                )
            else:
                bin_numerators = output.timestep_bin_numerators
                bin_denominators = output.timestep_bin_denominators
            for bin_index in range(10):
                local_timestep_bin_numerators[bin_index] += float(
                    bin_numerators[bin_index].detach()
                )
                local_timestep_bin_denominators[bin_index] += float(
                    bin_denominators[bin_index].detach()
                )


        global_numerator = reduce_scalar_sum(local_numerator)
        global_denominator = reduce_scalar_sum(local_denominator)
        global_valid_frames = reduce_scalar_sum(local_valid_frames)
        global_audio_seconds = reduce_scalar_sum(local_audio_seconds)
        global_text_drop_count = reduce_scalar_sum(local_text_drop_count)
        global_batch_size = reduce_scalar_sum(local_batch_size)
        global_timestep_sum = reduce_scalar_sum(local_timestep_sum)
        global_timestep_count = reduce_scalar_sum(local_timestep_count)
        global_endpoint_consistency_numerator = reduce_scalar_sum(
            local_endpoint_consistency_numerator
        )
        global_endpoint_consistency_denominator = reduce_scalar_sum(
            local_endpoint_consistency_denominator
        )
        global_velocity_consistency_numerator = reduce_scalar_sum(
            local_velocity_consistency_numerator
        )
        global_velocity_consistency_denominator = reduce_scalar_sum(
            local_velocity_consistency_denominator
        )
        global_source_coupling_cost_before = reduce_scalar_sum(
            local_source_coupling_cost_before
        )
        global_source_coupling_cost_after = reduce_scalar_sum(
            local_source_coupling_cost_after
        )
        global_source_coupling_count = reduce_scalar_sum(local_source_coupling_count)
        global_semantic_replacements = reduce_scalar_sum(local_semantic_replacements)
        global_semantic_eligible = reduce_scalar_sum(local_semantic_eligible)
        global_timestep_bin_numerators = [
            reduce_scalar_sum(value) for value in local_timestep_bin_numerators
        ]
        global_timestep_bin_denominators = [
            reduce_scalar_sum(value) for value in local_timestep_bin_denominators
        ]
        if global_denominator <= 0 or global_timestep_count <= 0:
            raise RuntimeError("global flow denominator and timestep count must be positive")

        parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]
        self._manual_all_reduce_gradients(parameters)
        semantic_embedding_frozen = self._apply_semantic_embedding_freeze()
        semantic_embedding = self.unwrapped_conditioner.semantic_embedding.weight
        semantic_embedding_optimizer_member = any(
            parameter is semantic_embedding
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        )
        semantic_unfreeze_transition = (
            self.semantic_embedding_freeze_steps > 0
            and not semantic_embedding_frozen
            and self.global_step == self.semantic_embedding_freeze_steps
        )
        semantic_embedding_before = None
        semantic_embedding_pre_update_sha256 = None
        if semantic_unfreeze_transition:
            if not semantic_embedding_optimizer_member:
                raise RuntimeError("Semantic embedding was not present when thawing the optimizer param group")
            if (
                semantic_embedding.grad is None
                or not torch.isfinite(semantic_embedding.grad).all()
            ):
                raise RuntimeError("Semantic embedding: the first step of thawing does not have a finite gradient")
            semantic_embedding_before = semantic_embedding.detach().clone()
            semantic_embedding_pre_update_sha256 = hashlib.sha256(
                semantic_embedding_before.contiguous()
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()


        gradient_scale = (
            1.0 if self.deterministic_manual_gradient_reduce else self.world_size
        ) / global_denominator
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(gradient_scale)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            self.gradient_clip_norm,
            error_if_nonfinite=self.gradient_clip_error_if_nonfinite,
            foreach=self.gradient_clip_foreach,
        )
        dynamics_module = self.unwrapped_conditioner.relative_dynamics_conditioning
        dynamics_adapter_gradient_norm = None
        if dynamics_module is not None:
            dynamics_gradient_square = sum(
                parameter.grad.detach().float().square().sum()
                for parameter in dynamics_module.adapter.parameters()
                if parameter.grad is not None
            )
            dynamics_adapter_gradient_norm = float(
                dynamics_gradient_square.sqrt().cpu()
            )
        applied_learning_rate = float(self.optimizer.param_groups[0]["lr"])
        self.optimizer.step()
        semantic_embedding_post_update_sha256 = None
        semantic_embedding_max_abs_update = None
        if semantic_unfreeze_transition:
            assert semantic_embedding_before is not None
            if not torch.isfinite(semantic_embedding).all():
                raise RuntimeError("Semantic embedding: a non-finite value appears after the first step of unfreezing.")
            semantic_embedding_post_update_sha256 = hashlib.sha256(
                semantic_embedding.detach()
                .contiguous()
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
            semantic_embedding_max_abs_update = float(
                (
                    semantic_embedding.detach().float()
                    - semantic_embedding_before.float()
                )
                .abs()
                .max()
                .cpu()
            )
            if (
                semantic_embedding_pre_update_sha256
                == semantic_embedding_post_update_sha256
                or not math.isfinite(semantic_embedding_max_abs_update)
                or semantic_embedding_max_abs_update <= 0
            ):
                raise RuntimeError("A bounded semantic embedding update did not occur in the first step of unfreezing")
        if self.ema is not None:
            self.ema.update()
        self.scheduler.step()
        self.global_step += 1
        self.consumed_audio_seconds += global_audio_seconds
        for name, module in self.frozen_modules.items():
            assert_module_frozen(module, name=name)
        metrics: dict[str, Any] = {
            "event": "train_step",
            "global_step": self.global_step,
            "loss": global_numerator / global_denominator,
            "loss_numerator": global_numerator,
            "loss_denominator": global_denominator,
            "valid_frames": int(global_valid_frames),
            "audio_seconds_per_update": global_audio_seconds,
            "consumed_audio_seconds": self.consumed_audio_seconds,
            "microbatch_count": len(batches),
            "text_drop_count": int(global_text_drop_count),
            "batch_size": int(global_batch_size),
            "semantic_embedding_frozen": semantic_embedding_frozen,
            "semantic_embedding_freeze_steps": self.semantic_embedding_freeze_steps,
            "semantic_embedding_optimizer_member": semantic_embedding_optimizer_member,
            "semantic_embedding_unfreeze_transition": semantic_unfreeze_transition,
            "semantic_embedding_pre_update_sha256": semantic_embedding_pre_update_sha256,
            "semantic_embedding_post_update_sha256": semantic_embedding_post_update_sha256,
            "semantic_embedding_max_abs_update": semantic_embedding_max_abs_update,
            "semantic_corruption_mode": self.semantic_corruption.mode,
            "semantic_robust_sample_probability": (
                float(self.semantic_corruption.robust_sample_probability)
                if self.semantic_corruption.mode == "emdc_knn"
                else 0.0
            ),
            "semantic_conditional_replacement_probability": float(
                self.semantic_corruption.probability_for_step(self.global_step - 1)
            ),
            "semantic_replacement_probability": float(
                semantic_replacement_probability or 0.0
            ),
            "semantic_replacement_count": int(global_semantic_replacements),
            "semantic_replacement_ratio": (
                global_semantic_replacements / global_semantic_eligible
                if global_semantic_eligible > 0
                else 0.0
            ),
            "timestep_mean": global_timestep_sum / global_timestep_count,
            "gradient_norm": float(gradient_norm.detach()),
            "dynamics_conditioning_enabled": dynamics_module is not None,
            "dynamics_adapter_gradient_norm": (dynamics_adapter_gradient_norm),
            "learning_rate": applied_learning_rate,
            "next_learning_rate": float(self.scheduler.get_last_lr()[0]),
            "ema_mode": "parameter_fp32" if self.ema is not None else "none",
            "ema_decay": self.ema.decay if self.ema is not None else None,
            "ema_num_updates": (self.ema.num_updates if self.ema is not None else 0),
            "precision": self.precision,
            "flow_format_version": self.flow.config.format_version,
            "training_objective": training_objective,
            "consistency_flow_matching_format_version": (
                self.consistency_flow_matching.format_version
                if self.consistency_flow_matching.enabled
                else None
            ),
            "consistency_delta": (
                self.consistency_flow_matching.delta
                if self.consistency_flow_matching.enabled
                else None
            ),
            "consistency_num_segments": (
                self.consistency_flow_matching.num_segments
                if self.consistency_flow_matching.enabled
                else None
            ),
            "consistency_boundary": (
                effective_consistency_config.boundary
                if self.consistency_flow_matching.enabled
                else None
            ),
            "consistency_target_boundary": (
                self.consistency_flow_matching.boundary
                if self.consistency_flow_matching.enabled
                else None
            ),
            "consistency_boundary_zero_steps": (
                self.consistency_flow_matching.boundary_zero_steps
                if self.consistency_flow_matching.enabled
                else None
            ),
            "consistency_velocity_weight": (
                self.consistency_flow_matching.velocity_weight
                if self.consistency_flow_matching.enabled
                else None
            ),
            "endpoint_consistency_loss": (
                global_endpoint_consistency_numerator
                / global_endpoint_consistency_denominator
                if global_endpoint_consistency_denominator > 0
                else None
            ),
            "velocity_consistency_loss": (
                global_velocity_consistency_numerator
                / global_velocity_consistency_denominator
                if global_velocity_consistency_denominator > 0
                else None
            ),
            "source_coupling": self.flow.config.source_coupling,
            "source_coupling_scope": (source_coupling_scope or "independent"),
            "source_coupling_batch_size": int(
                source_coupling_batch_size or global_batch_size
            ),
            "source_coupling_cost_before": (
                global_source_coupling_cost_before / global_source_coupling_count
                if global_source_coupling_count > 0
                else None
            ),
            "source_coupling_cost_after": (
                global_source_coupling_cost_after / global_source_coupling_count
                if global_source_coupling_count > 0
                else None
            ),
            "timestep_shift": self.flow.config.timestep_shift,
            "fixed_flow_training": self.fixed_flow_training is not None,
            "fixed_flow_source_seed": (
                int(self.fixed_flow_training["source_seed"])
                if self.fixed_flow_training is not None
                else None
            ),
            "fixed_flow_timestep_grid_size": (
                len(self.fixed_flow_training["timestep_grid"])
                if self.fixed_flow_training is not None
                else None
            ),
            "world_size": self.world_size,
        }
        target_audio_seconds = float(
            self.config.get("train", {}).get(
                "global_audio_seconds_per_update",
                0.0,
            )
        )
        audio_policy = str(
            self.config.get("train", {}).get(
                "audio_seconds_update_policy",
                "minimum",
            )
        )
        metrics["target_audio_seconds_per_update"] = target_audio_seconds
        metrics["audio_seconds_update_policy"] = audio_policy
        metrics["audio_seconds_overshoot"] = (
            global_audio_seconds - target_audio_seconds
            if target_audio_seconds > 0
            else 0.0
        )
        for bin_index, (numerator, denominator) in enumerate(
            zip(
                global_timestep_bin_numerators,
                global_timestep_bin_denominators,
            )
        ):
            metrics[
                f"timestep_loss_{bin_index / 10:.1f}_{(bin_index + 1) / 10:.1f}"
            ] = numerator / denominator if denominator > 0 else None
            metrics[
                f"timestep_frames_{bin_index / 10:.1f}_{(bin_index + 1) / 10:.1f}"
            ] = int(denominator)
        log_error = None
        if self.rank == 0 and self.logger is not None:
            try:
                self.logger.write(metrics)
            except Exception as exc:
                log_error = f"{type(exc).__name__}: {exc}"
        raise_if_rank0_error(
            log_error, action=f"structured log step={self.global_step}"
        )
        return metrics

    def train_step(self, batch: Any) -> dict[str, Any]:
        return self.train_update((batch,))

    def rng_states(self) -> dict[str, torch.Tensor]:
        return {
            "flow_source": self.source_generator.get_state(),
            "flow_timestep": self.timestep_generator.get_state(),
            "text_drop": self.text_drop_generator.get_state(),
        }

    def restore_rng_states(self, states: Mapping[str, torch.Tensor]) -> None:
        expected = {"flow_source", "flow_timestep", "text_drop"}
        if set(states) != expected:
            raise RuntimeError(f"checkpoint RNG streams must be exactly {sorted(expected)}")
        self.source_generator.set_state(states["flow_source"].cpu())
        self.timestep_generator.set_state(states["flow_timestep"].cpu())
        self.text_drop_generator.set_state(states["text_drop"].cpu())

    def rank_rng_state(self) -> dict[str, Any]:
        return {
            "independent": self.rng_states(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda"
                else None
            ),
        }

    def restore_rank_rng_state(self, state: Mapping[str, Any]) -> None:
        if set(state) != {
            "independent",
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda",
        }:
            raise RuntimeError("checkpoint rank RNG payload fields are incomplete")
        independent = state["independent"]
        if not isinstance(independent, Mapping):
            raise RuntimeError("checkpoint independent RNG payload must be a mapping")
        self.restore_rng_states(independent)
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"].cpu())
        cuda_state = state["torch_cuda"]
        if self.device.type == "cuda":
            if cuda_state is None:
                raise RuntimeError("CUDA resume checkpoint is missing the current rank CUDA RNG")
            torch.cuda.set_rng_state(cuda_state.cpu(), self.device)
        elif cuda_state is not None:
            raise RuntimeError("CPU resume must not restore a CUDA RNG payload")

    def restore_rng_after_ddp_init(self) -> None:

        if self._pending_post_ddp_rng_state is None:
            return
        self.restore_rank_rng_state(self._pending_post_ddp_rng_state)
        self._pending_post_ddp_rng_state = None

    def save_checkpoint(self, path: str | Path) -> None:
        self._assert_checkpoint_state_usable()
        if self.checkpoint_adapter is None:
            raise RuntimeError("The strict Render checkpoint adapter is required")
        progress = {
            "global_step": self.global_step,
            "consumed_audio_seconds": self.consumed_audio_seconds,
            "epoch": self.epoch,
            "batches_consumed": self.batches_consumed,
            "sampler_state": self.sampler_state,
            "distributed_state": self.distributed_state,
            "validation_state": {
                "validation_runs": self.validation_runs,
                "best_valid_loss": (
                    self.best_valid_loss
                    if math.isfinite(self.best_valid_loss)
                    else None
                ),
                "best_valid_step": self.best_valid_step,
                "valid_without_improvement": self.valid_without_improvement,
                "last_valid_loss": self.last_valid_loss,
            },
            "training_extension": self.training_extension,
        }
        assert_distributed_consensus("render_checkpoint_progress", progress)
        gathered = gather_object_to_rank0(
            {"rank": self.rank, "rng_state": self.rank_rng_state()}
        )
        save_error = None
        if self.rank == 0:
            try:
                assert gathered is not None
                by_rank = {
                    str(int(item["rank"])): item["rng_state"] for item in gathered
                }
                expected_ranks = {str(rank) for rank in range(self.world_size)}
                if set(by_rank) != expected_ranks:
                    raise RuntimeError(
                        "checkpoint rank RNG The collection is incomplete:"
                        f"expected={sorted(expected_ranks)} "
                        f"actual={sorted(by_rank)}"
                    )
                self.checkpoint_adapter.save(
                    path,
                    model=self.unwrapped_model,
                    conditioner=self.unwrapped_conditioner,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    ema=self.ema,
                    trainer_state=progress,
                    config=self.config,
                    revisions=asdict(self.revisions),
                    rng_states=by_rank,
                    distributed_state=self.distributed_state,
                )
            except Exception as exc:
                save_error = f"{type(exc).__name__}: {exc}"
        raise_if_rank0_error(
            save_error, action=f"Render checkpoint save step={self.global_step}"
        )
        barrier()

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        resume: bool,
        expected_checkpoint_sha256: str | None = None,
    ) -> None:
        self._assert_checkpoint_state_usable()
        if self.checkpoint_adapter is None:
            raise RuntimeError("The strict Render checkpoint adapter is required")
        state: Mapping[str, Any] | None = None
        load_error: str | None = None
        summary: dict[str, Any] | None = None

        def validate_before_apply(candidate: Mapping[str, Any]) -> None:
            nonlocal summary
            if not resume:
                summary = {"resume": False}
                return
            required = {
                "global_step",
                "epoch",
                "batches_consumed",
                "sampler_state",
                "distributed_state",
                "consumed_audio_seconds",
                "rank_rng_states",
                "validation_state",
            }
            missing = required - set(candidate)
            if missing:
                raise RuntimeError(f"checkpoint adapter response is missing fields {sorted(missing)}")
            if candidate["distributed_state"] != self.distributed_state:
                raise RuntimeError(
                    "resume world/batch config/cache topology mismatch:"
                    f"current={self.distributed_state} "
                    f"checkpoint={candidate['distributed_state']}"
                )
            rank_rng_states = candidate["rank_rng_states"]
            if not isinstance(rank_rng_states, Mapping):
                raise RuntimeError("checkpoint is missing per-rank RNG mapping")
            expected_ranks = {str(rank) for rank in range(self.world_size)}
            if set(rank_rng_states) != expected_ranks:
                raise RuntimeError(
                    "checkpoint per-rank RNG Topology is mismatch:"
                    f"expected={sorted(expected_ranks)} "
                    f"actual={sorted(rank_rng_states)}"
                )
            validation_state = candidate["validation_state"]
            if not isinstance(validation_state, Mapping):
                raise RuntimeError("checkpoint is missing validation_state mapping")
            training_extension = candidate.get("training_extension")
            if training_extension is not None and not isinstance(
                training_extension, Mapping
            ):
                raise RuntimeError("checkpoint training_extension must be a mapping or null")
            validation_runs = validation_state.get("validation_runs")
            best_valid_loss = validation_state.get("best_valid_loss")
            best_valid_step = validation_state.get("best_valid_step")
            valid_without_improvement = validation_state.get(
                "valid_without_improvement"
            )
            last_valid_loss = validation_state.get("last_valid_loss")
            if (
                not isinstance(validation_runs, int)
                or isinstance(validation_runs, bool)
                or validation_runs < 0
                or not isinstance(valid_without_improvement, int)
                or isinstance(valid_without_improvement, bool)
                or valid_without_improvement < 0
                or (
                    best_valid_step is not None
                    and (
                        not isinstance(best_valid_step, int)
                        or isinstance(best_valid_step, bool)
                        or best_valid_step < 0
                    )
                )
                or (
                    best_valid_loss is not None
                    and (
                        isinstance(best_valid_loss, bool)
                        or not isinstance(best_valid_loss, (int, float))
                        or not math.isfinite(best_valid_loss)
                        or best_valid_loss < 0
                    )
                )
                or (
                    last_valid_loss is not None
                    and (
                        isinstance(last_valid_loss, bool)
                        or not isinstance(last_valid_loss, (int, float))
                        or not math.isfinite(last_valid_loss)
                        or last_valid_loss < 0
                    )
                )
            ):
                raise RuntimeError("checkpoint validation_state is invalid")
            current_rank_state = rank_rng_states.get(str(self.rank))
            if not isinstance(current_rank_state, Mapping):
                raise RuntimeError("checkpoint is missing the current rank RNG state")
            if set(current_rank_state) != {
                "independent",
                "python",
                "numpy",
                "torch_cpu",
                "torch_cuda",
            }:
                raise RuntimeError("checkpoint rank RNG payload fields are incomplete")
            independent = current_rank_state["independent"]
            if not isinstance(independent, Mapping) or set(independent) != {
                "flow_source",
                "flow_timestep",
                "text_drop",
            }:
                raise RuntimeError("checkpoint independent RNG streams are incomplete")
            for name, value in independent.items():
                if not isinstance(value, torch.Tensor):
                    raise RuntimeError(f"checkpoint RNG stream {name} is not a tensor")
                torch.Generator(device="cpu").set_state(value.cpu())
            try:
                random.Random().setstate(current_rank_state["python"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError("checkpoint Python RNG state is incompatible") from exc
            try:
                numpy_probe = np.random.RandomState()
                numpy_probe.set_state(current_rank_state["numpy"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError("checkpoint NumPy RNG state is incompatible") from exc
            torch_cpu = current_rank_state["torch_cpu"]
            if not isinstance(torch_cpu, torch.Tensor):
                raise RuntimeError("checkpoint torch_cpu RNG is not a tensor")
            torch.Generator(device="cpu").set_state(torch_cpu.cpu())
            torch_cuda = current_rank_state["torch_cuda"]
            if self.device.type == "cuda":
                if not isinstance(torch_cuda, torch.Tensor):
                    raise RuntimeError("CUDA resume checkpoint is missing the current rank CUDA RNG")
            elif torch_cuda is not None:
                raise RuntimeError("CPU resume must not restore a CUDA RNG payload")
            summary = {
                "resume": True,
                "global_step": int(candidate["global_step"]),
                "epoch": int(candidate["epoch"]),
                "batches_consumed": int(candidate["batches_consumed"]),
                "consumed_audio_seconds": float(candidate["consumed_audio_seconds"]),
                "sampler_state": candidate["sampler_state"],
                "distributed_state": candidate["distributed_state"],
                "rng_ranks": sorted(rank_rng_states),
                "validation_state": dict(validation_state),
                "training_extension": (
                    dict(training_extension) if training_extension is not None else None
                ),
            }

        try:
            state = self.checkpoint_adapter.load(
                path,
                model=self.unwrapped_model,
                conditioner=self.unwrapped_conditioner,
                optimizer=self.optimizer if resume else None,
                scheduler=self.scheduler if resume else None,
                ema=self.ema if resume else None,
                expected_config=self.config,
                expected_revisions=asdict(self.revisions),
                resume=resume,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                pre_apply_validator=validate_before_apply,
            )


            if summary is None:
                validate_before_apply(state)
        except Exception as exc:
            load_error = f"{type(exc).__name__}: {exc}"
            self._checkpoint_load_failed = True
        try:
            assert_distributed_consensus(
                "render_checkpoint_load",
                {"error": load_error, "summary": summary},
            )
            raise_if_rank0_error(
                load_error if self.rank == 0 else None,
                action=f"Render checkpoint load resume={resume}",
            )
        except Exception:


            self._checkpoint_load_failed = True
            raise
        if load_error is not None or state is None:

            raise RuntimeError(f"Render checkpoint load failed: {load_error}")
        if resume:
            self.global_step = int(state["global_step"])
            self.epoch = int(state["epoch"])
            self.batches_consumed = int(state["batches_consumed"])
            self.sampler_state = (
                dict(state["sampler_state"])
                if state["sampler_state"] is not None
                else None
            )
            self.consumed_audio_seconds = float(state["consumed_audio_seconds"])
            validation_state = state["validation_state"]
            self.validation_runs = int(validation_state["validation_runs"])
            stored_best = validation_state["best_valid_loss"]
            self.best_valid_loss = (
                float(stored_best) if stored_best is not None else float("inf")
            )
            self.best_valid_step = validation_state["best_valid_step"]
            self.valid_without_improvement = int(
                validation_state["valid_without_improvement"]
            )
            stored_last = validation_state["last_valid_loss"]
            self.last_valid_loss = (
                float(stored_last) if stored_last is not None else None
            )
            stored_extension = state.get("training_extension")
            self.training_extension = (
                dict(stored_extension)
                if isinstance(stored_extension, Mapping)
                else None
            )
            rank_rng_states = state["rank_rng_states"]
            current_rank_rng = dict(rank_rng_states[str(self.rank)])
            self.restore_rank_rng_state(current_rank_rng)


            self._pending_post_ddp_rng_state = current_rank_rng
        elif self.ema is not None:


            self.ema.reset_from_parameters()
        barrier()

class CachedRenderDataset(Dataset[dict[str, Any]]):

    def __init__(self, manifest: str | Path) -> None:
        self.manifest = Path(manifest)
        self.records = []
        self._frame_lengths: list[int | None] = []
        with self.manifest.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                cache_path = record.get("cache_path")
                if not cache_path:
                    raise ValueError(f"{self.manifest}:{line_number} is missing cache_path")
                path = Path(cache_path)
                if not path.is_absolute():
                    path = self.manifest.parent / path
                self.records.append((record, path))
                frames = record.get("latent_frames")
                self._frame_lengths.append(int(frames) if frames is not None else None)
        if not self.records:
            raise ValueError(f"Render cache manifest is empty:{self.manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record, path = self.records[index]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise TypeError(f"Render cache must be a mapping:{path}")
        result = dict(payload)
        result.setdefault("sample_id", record.get("sample_id", str(index)))
        result.setdefault("revisions", record.get("revisions"))
        return result

    def frame_length(self, index: int) -> int:
        cached = self._frame_lengths[index]
        if cached is not None:
            return cached
        payload = self[index]
        mask = payload.get("semantic_mask", payload.get("latent_mask"))
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 2 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.ndim != 1 or mask.dtype != torch.bool:
                raise ValueError("cache semantic and latent masks must be bool with shape [T]")
            frames = int(mask.sum())
        else:
            semantic = payload.get("semantic_ids")
            latents = payload.get("latents")
            value = semantic if isinstance(semantic, torch.Tensor) else latents
            if not isinstance(value, torch.Tensor):
                raise ValueError("cache is missing tensor frame length metadata")
            if value.ndim >= 2 and value.shape[0] == 1:
                value = value[0]
            frames = int(value.shape[0])
        if frames <= 0:
            raise ValueError("cache latent and semantic must be non-empty")
        self._frame_lengths[index] = frames
        return frames

    @property
    def frame_lengths(self) -> list[int]:
        return [self.frame_length(index) for index in range(len(self))]


def _unpadded_length(value: torch.Tensor, mask: torch.Tensor | None) -> int:
    if mask is not None:
        if mask.ndim != 1 or mask.dtype != torch.bool:
            raise ValueError("single sample cache mask must be bool with shape [T]")
        mask_to_lengths(mask.unsqueeze(0), require_right_padded=True)
        return int(mask.sum().item())
    return int(value.shape[0])


def _pad_1d_or_2d(
    values: Sequence[torch.Tensor],
    *,
    pad_value: float,
) -> torch.Tensor:
    if not values:
        raise ValueError("cannot pad an empty tensor sequence")
    maximum = max(int(value.shape[0]) for value in values)
    tail = tuple(values[0].shape[1:])
    dtype = values[0].dtype
    device = values[0].device
    if any(
        tuple(value.shape[1:]) != tail or value.dtype != dtype or value.device != device
        for value in values
    ):
        raise ValueError("pending padded tensor tail shape, dtype, and device must match")
    result = values[0].new_full((len(values), maximum, *tail), pad_value)
    for index, value in enumerate(values):
        result[index, : value.shape[0]] = value
    return result


def collate_cached_render_batch(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot collate empty Render batch")
    latent_values = []
    semantic_values = []
    lengths = []
    for record in records:
        latent = record["latents"]
        semantic = record["semantic_ids"]
        if not isinstance(latent, torch.Tensor) or not isinstance(
            semantic, torch.Tensor
        ):
            raise TypeError("cached latent and semantic values must be tensors")
        if latent.ndim == 2:
            pass
        elif latent.ndim == 3 and latent.shape[0] == 1:
            latent = latent[0]
        else:
            raise ValueError("single sample cached latent must have shape [T, 128]")
        if semantic.ndim == 2 and semantic.shape[0] == 1:
            semantic = semantic[0]
        if semantic.ndim != 1:
            raise ValueError("single sample cached semantic must have shape [T]")
        if semantic.dtype != torch.long:
            raise TypeError("single sample cached semantic must be int64; lossy conversion is prohibited")
        mask = record.get("latent_mask")
        if isinstance(mask, torch.Tensor) and mask.ndim == 2:
            mask = mask[0]
        length = _unpadded_length(latent, mask)
        semantic_mask = record.get("semantic_mask")
        if isinstance(semantic_mask, torch.Tensor) and semantic_mask.ndim == 2:
            semantic_mask = semantic_mask[0]
        semantic_length = _unpadded_length(semantic, semantic_mask)
        if semantic_length != length:
            raise ValueError("cache semantic and latent effective lengths must match exactly")
        if semantic.shape[0] < length or latent.shape[0] < length:
            raise ValueError("cache mask exceeds tensor length")
        latent_values.append(latent[:length])
        semantic_values.append(semantic[:length])
        lengths.append(length)
    latents = _pad_1d_or_2d(latent_values, pad_value=0.0)
    semantics = _pad_1d_or_2d(semantic_values, pad_value=0)
    positions = torch.arange(latents.shape[1])
    frame_mask = positions.unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    batch: dict[str, Any] = {
        "sample_ids": [
            str(record.get("sample_id", index)) for index, record in enumerate(records)
        ],
        "latents": latents,
        "latent_mask": frame_mask,
        "semantic_ids": semantics.long(),
        "semantic_mask": frame_mask.clone(),


        "duration_seconds": (
            torch.tensor(lengths, dtype=torch.long) * SAMPLES_PER_LATENT_FRAME
        ).to(torch.float64)
        / SAMPLE_RATE,
        "revisions": [record.get("revisions") for record in records],
    }
    loudness_presence = ["global_loudness_lufs" in record for record in records]
    if any(loudness_presence):
        if not all(loudness_presence):
            raise ValueError("A batch cannot mix missing and present global loudness values")
        values = [float(record["global_loudness_lufs"]) for record in records]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("global loudness batch contains NaN/Inf")
        batch["global_loudness_lufs"] = torch.tensor(values, dtype=torch.float32)
    for name, maximum in (("description", 256), ("lyrics", 1_536)):
        masks = []
        ids_values = []
        embedding_values = []
        has_embeddings = [
            record.get(f"{name}_embeddings") is not None for record in records
        ]
        has_ids = [record.get(f"{name}_input_ids") is not None for record in records]


        if all(has_embeddings):
            use_embeddings = True
        elif all(has_ids) and not any(has_embeddings):
            use_embeddings = False
        else:
            raise ValueError(f"{name} batch cannot mix cached embeddings and input_ids")
        for record in records:
            key = f"{name}_embeddings" if use_embeddings else f"{name}_input_ids"
            value = record[key]
            if (
                use_embeddings
                and value.ndim == 3
                and value.shape[0] == 1
                or not use_embeddings
                and value.ndim == 2
                and value.shape[0] == 1
            ):
                value = value[0]
            expected_ndim = 2 if use_embeddings else 1
            if value.ndim != expected_ndim:
                expected = "[T,D] [1,T,D]" if use_embeddings else "[T] [1,T]"
                raise ValueError(f"{name} cache must be {expected}")
            mask = record.get(f"{name}_mask")
            if isinstance(mask, torch.Tensor) and mask.ndim == 2:
                mask = mask[0]
            if mask is not None:
                if (
                    mask.ndim != 1
                    or mask.dtype != torch.bool
                    or mask.shape[0] != value.shape[0]
                ):
                    raise ValueError(f"{name} cache mask must be bool with shape [T]")
                value = value[mask]
            length = int(value.shape[0])
            if length > maximum:
                raise ValueError(f"{name} cache length {length}>{maximum}")
            (embedding_values if use_embeddings else ids_values).append(value)
            masks.append(length)
        values = embedding_values if use_embeddings else ids_values
        padded = _pad_1d_or_2d(values, pad_value=0.0 if use_embeddings else 0)
        mask = torch.arange(padded.shape[1]).unsqueeze(0) < torch.tensor(
            masks
        ).unsqueeze(1)
        batch[f"{name}_embeddings" if use_embeddings else f"{name}_input_ids"] = padded
        batch[f"{name}_mask"] = mask
    return batch


def build_trainer_from_config(
    config: Mapping[str, Any],
    *,
    fake_text_encoder: nn.Module | None = None,
    frozen_vae: nn.Module | None = None,
    frozen_tokenizer: nn.Module | None = None,
    checkpoint_adapter: CheckpointAdapter | None = None,
    rank: int = 0,
    world_size: int = 1,
    distributed_state: Mapping[str, Any] | None = None,
    device: torch.device | str | None = None,
) -> RenderDiTTrainer:
    validate_dit_launch_config(config)
    condition_config = config["conditioning"]
    revision_config = RevisionContract.from_mapping(config["revisions"])
    optimizer_config = config["optimizer"]
    if str(optimizer_config.get("name", "AdamW")).lower() != "adamw":
        raise ValueError("Qwen-Music DiT training entry point only supports AdamW")
    semantic_corruption_config = SemanticCorruptionConfig.from_mapping(
        config.get("semantic_corruption")
    )
    semantic_error_calibration_provenance = None
    semantic_distractor_table = None
    semantic_corruption_load_error: str | None = None
    try:
        if semantic_corruption_config.mode == "emdc_knn":
            _, semantic_error_calibration_provenance = (
                verify_semantic_error_calibration(
                    semantic_corruption_config.calibration,
                    expected_tokenizer_revision=revision_config.tokenizer_revision,
                )
            )
            semantic_distractor_table = load_semantic_distractor_table(
                semantic_corruption_config.asset or {},
                expected_tokenizer_revision=revision_config.tokenizer_revision,
            )
    except Exception as exc:
        semantic_corruption_load_error = f"{type(exc).__name__}: {exc}"
    assert_distributed_consensus(
        "render_semantic_corruption_assets",
        {
            "error": semantic_corruption_load_error,
            "mode": semantic_corruption_config.mode,
            "calibration": semantic_error_calibration_provenance,
            "distractor": (
                semantic_distractor_table.provenance
                if semantic_distractor_table is not None
                else None
            ),
        },
    )
    if semantic_corruption_load_error is not None:
        raise RuntimeError(
            f"Render Semantic corruptionAsset loading failed: {semantic_corruption_load_error}"
        )
    model, conditioner = build_render_dit_components_from_config(
        config,
        encoder=fake_text_encoder,
    )
    train_config = config["train"]
    model.set_activation_checkpoint_block_interval(
        _config_int(
            train_config,
            "activation_checkpoint_block_interval",
            default=1,
            minimum=1,
        )
    )
    frozen = {}
    if frozen_vae is not None:
        frozen["spec_vae"] = frozen_vae
    if frozen_tokenizer is not None:
        frozen["semantic_tokenizer"] = frozen_tokenizer
    if (
        conditioner.semantic_embedding_asset_provenance is not None
        and not conditioner.semantic_embedding.weight.requires_grad
    ):
        frozen["semantic_embedding_asset"] = conditioner.semantic_embedding
    if conditioner.relative_dynamics_conditioning is not None:
        frozen["relative_dynamics_predictor"] = (
            conditioner.relative_dynamics_conditioning.predictor
        )
    output_dir = Path(train_config.get("output_dir", "outputs/render_dit"))
    logger = (
        StructuredJSONLLogger(output_dir / "metrics.jsonl") if int(rank) == 0 else None
    )
    if checkpoint_adapter is None:
        checkpoint_adapter = StrictRenderCheckpointAdapter(asdict(revision_config))
    return RenderDiTTrainer(
        model=model,
        conditioner=conditioner,
        flow_config=FlowConfig.from_mapping(config["flow"]),
        revisions=revision_config,
        learning_rate=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
        betas=tuple(optimizer_config.get("betas", (0.9, 0.95))),
        optimizer_foreach=optimizer_config.get("foreach"),
        optimizer_fused=optimizer_config.get("fused"),
        optimizer_eps=float(optimizer_config.get("eps", 1.0e-8)),
        optimizer_amsgrad=bool(optimizer_config.get("amsgrad", False)),
        optimizer_maximize=bool(optimizer_config.get("maximize", False)),
        optimizer_capturable=bool(optimizer_config.get("capturable", False)),
        optimizer_differentiable=bool(optimizer_config.get("differentiable", False)),
        optimizer_parameter_grouping=str(
            optimizer_config.get(
                "parameter_grouping",
                "all_trainable_single_group",
            )
        ),
        optimizer_weight_decay_exclusions=tuple(
            optimizer_config.get("weight_decay_exclusions", ())
        ),
        optimizer_zero_grad_set_to_none=bool(
            optimizer_config.get("zero_grad_set_to_none", True)
        ),
        ema_mode=str(optimizer_config.get("ema_mode", "none")),
        ema_decay=(
            float(optimizer_config["ema_decay"])
            if "ema_decay" in optimizer_config
            else None
        ),
        warmup_steps=int(optimizer_config.get("warmup_steps", 0)),
        max_steps=int(train_config["max_steps"]),
        scheduler=str(optimizer_config.get("scheduler", "cosine")),
        scheduler_step_indexing=str(
            optimizer_config.get(
                "scheduler_step_indexing",
                "optimizer_update_zero_based_final_inclusive",
            )
        ),
        gradient_clip_norm=float(train_config.get("gradient_clip_norm", 1.0)),
        gradient_clip_error_if_nonfinite=bool(
            train_config.get("gradient_clip_error_if_nonfinite", True)
        ),
        gradient_clip_foreach=bool(train_config.get("gradient_clip_foreach", False)),
        text_drop_probability=float(condition_config["text_drop_probability"]),
        semantic_embedding_freeze_steps=int(
            train_config.get("semantic_embedding_freeze_steps", 0)
        ),
        semantic_embedding_freeze_policy=str(
            train_config.get(
                "semantic_embedding_freeze_policy",
                "first_n_optimizer_updates",
            )
        ),
        semantic_corruption_config=semantic_corruption_config,
        semantic_distractor_table=semantic_distractor_table,
        semantic_error_calibration_provenance=(semantic_error_calibration_provenance),
        seed=int(train_config["seed"]),
        rank_seed_stride=int(train_config.get("rank_seed_stride", 1_000_003)),
        flow_source_seed_offset=int(train_config.get("flow_source_seed_offset", 11)),
        flow_timestep_seed_offset=int(
            train_config.get("flow_timestep_seed_offset", 23)
        ),
        text_drop_seed_offset=int(train_config.get("text_drop_seed_offset", 37)),
        min_lr_ratio=float(optimizer_config.get("min_lr_ratio", 1.0)),
        frozen_modules=frozen,
        logger=logger,
        checkpoint_adapter=checkpoint_adapter,
        config=config,
        require_cached_text=bool(config["data"].get("require_cached_text", True)),
        rank=int(rank),
        world_size=int(world_size),
        distributed_state=distributed_state,
        device=device,
        precision=str(train_config.get("precision", "fp32")),
    )


def _config_int(
    owner: Mapping[str, Any],
    name: str,
    *,
    default: Any = _NO_DEFAULT,
    minimum: int | None = None,
) -> int:
    value = owner.get(name, default)
    if value is _NO_DEFAULT:
        raise ValueError(f"configuration is missing {name}")
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _config_float(
    owner: Mapping[str, Any],
    name: str,
    *,
    default: Any = _NO_DEFAULT,
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_inclusive: bool = True,
) -> float:
    value = owner.get(name, default)
    if value is _NO_DEFAULT:
        raise ValueError(f"configuration is missing {name}")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise TypeError(f"{name} must be a finite value")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and (
        result > maximum if maximum_inclusive else result >= maximum
    ):
        operator = "<=" if maximum_inclusive else "<"
        raise ValueError(f"{name} must be {operator} {maximum}")
    return result


def _config_bool(
    owner: Mapping[str, Any],
    name: str,
    *,
    default: Any = _NO_DEFAULT,
) -> bool:
    value = owner.get(name, default)
    if value is _NO_DEFAULT:
        raise ValueError(f"configuration is missing {name}")
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value




def validate_dit_model_and_conditioning(config: Mapping[str, Any]) -> None:

    if not isinstance(config, Mapping):
        raise TypeError("DiT config must be a mapping")
    for name in ("model", "conditioning"):
        if not isinstance(config.get(name), Mapping):
            raise ValueError(f"DiT configuration is missing mapping section: {name}")
    model_config = DiTConfig.from_mapping(config["model"])
    if model_config.context_dim != model_config.hidden_size:
        raise ValueError("RenderConditioner requires context_dim to equal hidden_size")
    conditioning = config["conditioning"]
    allowed_conditioning = {
        "description_max_tokens",
        "description_projection_bias",
        "semantic_vocab_size",
        "initialization",
        "lyrics_max_tokens",
        "lyrics_encoder_layers",
        "lyrics_encoder_heads",
        "lyrics_head_dim",
        "lyrics_ffn_expansion",
        "lyrics_ffn_activation",
        "lyrics_gelu_approximation",
        "lyrics_qkv_bias",
        "lyrics_non_qkv_linear_bias",
        "lyrics_rope_base",
        "lyrics_rope_style",
        "lyrics_norm_eps",
        "lyrics_norm_type",
        "lyrics_attention_direction",
        "lyrics_norm_style",
        "lyrics_projection_order",
        "lyrics_projection_bias",
        "null_context_init_std",
        "text_drop_granularity",
        "text_drop_scope",
        "null_context_tokens",
        "null_context_layout",
        "text_context_composition",
        "text_context_layout",
        "text_context_separator",
        "text_context_segment_embedding",
        "text_compaction_policy",
        "dropout",
        "text_drop_probability",
        "semantic_embedding_init_std",
        "semantic_embedding_asset",
        "semantic_projection_init",
        "semantic_source_dim",
        "semantic_projection_bias",
        "global_loudness_conditioning",
        "global_loudness_mean_lufs",
        "global_loudness_std_lu",
        "global_loudness_clamp_std",
        "global_loudness_initialization_seed",
        "dynamics_conditioning",
        "dynamics_checkpoint",
        "dynamics_checkpoint_sha256",
        "dynamics_checkpoint_step",
        "dynamics_target_kind",
        "dynamics_freeze_predictor",
        "dynamics_include_activity",
        "dynamics_relative_scale_lu",
        "dynamics_initialization_seed",
        "text_encoder",
    }
    unknown_conditioning = set(conditioning) - allowed_conditioning
    if unknown_conditioning:
        raise ValueError(f"conditioning contains unknown fields: {sorted(unknown_conditioning)}")
    loudness_enabled = _config_bool(
        conditioning,
        "global_loudness_conditioning",
        default=False,
    )
    expected_adaln = "timestep_plus_global_loudness" if loudness_enabled else "timestep"
    if model_config.adaln_conditioning != expected_adaln:
        raise ValueError(
            "global loudnessswitch andmodel.adaln_conditioning mismatch:"
            f"expected={expected_adaln} actual={model_config.adaln_conditioning}"
        )
    for name, default in (
        ("global_loudness_mean_lufs", -14.0),
        ("global_loudness_std_lu", 5.0),
        ("global_loudness_clamp_std", 4.0),
    ):
        value = _config_float(conditioning, name, default=default)
        if name != "global_loudness_mean_lufs" and value <= 0.0:
            raise ValueError(f"conditioning.{name} must be positive")
    _config_int(
        conditioning,
        "global_loudness_initialization_seed",
        default=20260903,
        minimum=0,
    )
    dynamics_enabled = _config_bool(
        conditioning,
        "dynamics_conditioning",
        default=False,
    )
    dynamics_checkpoint = conditioning.get("dynamics_checkpoint")
    dynamics_checkpoint_sha256 = conditioning.get("dynamics_checkpoint_sha256")
    if dynamics_enabled:
        if not isinstance(dynamics_checkpoint, (str, Path)) or not str(
            dynamics_checkpoint
        ):
            raise ValueError("conditioning.dynamics_checkpoint must be a non-empty path")
        if not isinstance(dynamics_checkpoint_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", dynamics_checkpoint_sha256
        ):
            raise ValueError("conditioning.dynamics_checkpoint_sha256 must be a lowercase SHA-256")
        if not _config_bool(
            conditioning,
            "dynamics_freeze_predictor",
            default=True,
        ):
            raise ValueError("The current DiT dynamics predictor must be frozen")
        _config_bool(
            conditioning,
            "dynamics_include_activity",
            default=True,
        )
        if (
            _config_float(
                conditioning,
                "dynamics_relative_scale_lu",
                default=10.0,
            )
            <= 0.0
        ):
            raise ValueError("conditioning.dynamics_relative_scale_lu must be positive")
        _config_int(
            conditioning,
            "dynamics_checkpoint_step",
            default=8_250,
            minimum=0,
        )
        dynamics_target_kind = str(
            conditioning.get(
                "dynamics_target_kind",
                "relative_parent_lufs",
            )
        )
        if dynamics_target_kind not in {
            "relative_parent_lufs",
            "relative_short_window_lufs",
        }:
            raise ValueError(
                "conditioning.dynamics_target_kindonly supports"
                "relative_parent_lufs/relative_short_window_lufs"
            )
        _config_int(
            conditioning,
            "dynamics_initialization_seed",
            default=2026090304,
            minimum=0,
        )
    elif dynamics_checkpoint is not None or dynamics_checkpoint_sha256 is not None:
        raise ValueError("Disabled dynamics conditioning must not retain a checkpoint")
    if (
        _config_int(
            conditioning,
            "semantic_vocab_size",
            default=SEMANTIC_CODEBOOK_SIZE,
            minimum=1,
        )
        != SEMANTIC_CODEBOOK_SIZE
    ):
        raise ValueError(f"semantic_vocab_size must be {SEMANTIC_CODEBOOK_SIZE}")
    if (
        _config_int(
            conditioning,
            "description_max_tokens",
            default=256,
            minimum=1,
        )
        != 256
    ):
        raise ValueError("description_max_tokens must be 256")
    if _config_bool(
        conditioning,
        "description_projection_bias",
        default=False,
    ):
        raise ValueError("description_projection_bias must be false")
    if conditioning.get("initialization", "dit_xavier") != "dit_xavier":
        raise ValueError("conditioning.initialization must be dit_xavier")
    lyrics_max_tokens = _config_int(
        conditioning,
        "lyrics_max_tokens",
        default=1_536,
        minimum=1,
    )
    if lyrics_max_tokens not in SUPPORTED_LYRICS_MAX_TOKENS:
        raise ValueError(
            "lyrics_max_tokens must be 1536 for RendererData short views or 1792 for lossless full views"
        )
    if (
        _config_int(
            conditioning,
            "lyrics_encoder_layers",
            default=6,
            minimum=1,
        )
        != 6
    ):
        raise ValueError("lyrics_encoder_layers must be 6")
    _config_int(conditioning, "lyrics_encoder_heads", default=8, minimum=1)
    lyrics_head_dim = conditioning.get("lyrics_head_dim")
    if lyrics_head_dim is not None:
        lyrics_head_dim = _config_int(
            conditioning,
            "lyrics_head_dim",
            minimum=1,
        )
        if lyrics_head_dim % 2:
            raise ValueError("lyrics_head_dim must be an even number")
    elif model_config.hidden_size % _config_int(
        conditioning,
        "lyrics_encoder_heads",
        default=8,
        minimum=1,
    ):
        raise ValueError("When lyrics_head_dim is omitted, hidden_size must be divisible by the number of lyrics heads")
    _config_float(
        conditioning,
        "lyrics_ffn_expansion",
        default=4.0,
        minimum=0.0,
    )
    if float(conditioning.get("lyrics_ffn_expansion", 4.0)) <= 0:
        raise ValueError("lyrics_ffn_expansion must be positive")
    if conditioning.get("lyrics_ffn_activation", "gelu") not in {
        "gelu",
        "swiglu",
    }:
        raise ValueError("lyrics_ffn_activation must be gelu or swiglu")
    if conditioning.get("lyrics_gelu_approximation", "tanh") not in {
        "none",
        "tanh",
    }:
        raise ValueError("lyrics_gelu_approximation must be none or tanh")
    if (
        conditioning.get("lyrics_ffn_activation", "gelu") != "gelu"
        and conditioning.get("lyrics_gelu_approximation", "none") != "none"
    ):
        raise ValueError("Non-GELU lyrics FFN requires lyrics_gelu_approximation=none")
    _config_bool(conditioning, "lyrics_qkv_bias", default=True)
    _config_bool(conditioning, "lyrics_non_qkv_linear_bias", default=True)
    _config_float(
        conditioning,
        "lyrics_rope_base",
        default=10_000.0,
        minimum=1.0,
        maximum_inclusive=False,
    )
    if conditioning.get("lyrics_rope_style", "half_split") != "half_split":
        raise ValueError("lyrics_rope_style only supports half_split")
    _config_float(
        conditioning,
        "lyrics_norm_eps",
        default=1.0e-5,
        minimum=0.0,
        maximum_inclusive=False,
    )
    if conditioning.get("lyrics_norm_type", "layernorm") not in {
        "layernorm",
        "rmsnorm",
    }:
        raise ValueError("lyrics_norm_type must be layernorm or rmsnorm")
    if (
        conditioning.get("lyrics_attention_direction", "bidirectional")
        != "bidirectional"
    ):
        raise ValueError("lyrics_attention_direction only supports bidirectional")
    if (
        conditioning.get("lyrics_norm_style", "pre_norm_with_final_norm")
        != "pre_norm_with_final_norm"
    ):
        raise ValueError("lyrics_norm_style only supports pre_norm_with_final_norm")
    if (
        conditioning.get("lyrics_projection_order", "before_encoder")
        != "before_encoder"
    ):
        raise ValueError("lyrics_projection_order only supports before_encoder")
    _config_bool(conditioning, "lyrics_projection_bias", default=False)
    _config_float(
        conditioning,
        "null_context_init_std",
        default=0.02,
        minimum=0.0,
        maximum_inclusive=False,
    )
    if conditioning.get("text_drop_granularity", "sample") != "sample":
        raise ValueError("text_drop_granularity only supports sample")
    if (
        conditioning.get(
            "text_drop_scope",
            "joint_description_lyrics",
        )
        != "joint_description_lyrics"
    ):
        raise ValueError("text_drop_scope only supports joint_description_lyrics")
    if (
        _config_int(
            conditioning,
            "null_context_tokens",
            default=1,
            minimum=1,
        )
        != 1
    ):
        raise ValueError("null_context_tokens only supports 1")
    if conditioning.get("null_context_layout", "single_token") not in {
        "single_token",
        "preserve_text_mask",
    }:
        raise ValueError("null_context_layout must be single_token or preserve_text_mask")
    if conditioning.get("text_context_composition", "concatenate") != "concatenate":
        raise ValueError("text_context_composition only supports concatenate")
    if (
        conditioning.get(
            "text_context_layout",
            "description_then_lyrics",
        )
        != "description_then_lyrics"
    ):
        raise ValueError("text_context_layout only supports description_then_lyrics")
    if conditioning.get("text_context_separator", "none") != "none":
        raise ValueError("text_context_separator only supports none")
    if conditioning.get("text_context_segment_embedding", "none") != "none":
        raise ValueError("text_context_segment_embedding only supports none")
    if (
        conditioning.get(
            "text_compaction_policy",
            "stable_valid_tokens_right_padded",
        )
        != "stable_valid_tokens_right_padded"
    ):
        raise ValueError("text_compaction_policy only supports stable_valid_tokens_right_padded")
    _config_float(
        conditioning,
        "dropout",
        default=0.0,
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    _config_float(
        conditioning,
        "text_drop_probability",
        minimum=0.0,
        maximum=1.0,
    )
    semantic_embedding_init_std = conditioning.get("semantic_embedding_init_std")
    if semantic_embedding_init_std is not None:
        _config_float(
            conditioning,
            "semantic_embedding_init_std",
            minimum=0.0,
        )
        if float(semantic_embedding_init_std) <= 0:
            raise ValueError("semantic_embedding_init_std must be positive")
    semantic_embedding_asset = conditioning.get("semantic_embedding_asset")
    if semantic_embedding_asset is not None:
        if not isinstance(semantic_embedding_asset, Mapping):
            raise TypeError("semantic_embedding_asset must be a mapping")
        required_asset_fields = {
            "ready_path",
            "ready_sha256",
            "artifact_sha256",
            "asset_revision",
            "embedding_key",
            "tensor_sha256",
            "trainable",
        }
        allowed_asset_fields = required_asset_fields | {"expected_source"}
        missing_asset_fields = required_asset_fields - set(semantic_embedding_asset)
        unknown_asset_fields = set(semantic_embedding_asset) - allowed_asset_fields
        if missing_asset_fields or unknown_asset_fields:
            raise ValueError(
                "semantic_embedding_assetfield is incomplete or unknown:"
                f"missing={sorted(missing_asset_fields)} "
                f"unknown={sorted(unknown_asset_fields)}"
            )
        for name in (
            "ready_sha256",
            "artifact_sha256",
            "asset_revision",
            "tensor_sha256",
        ):
            value = semantic_embedding_asset[name]
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"semantic_embedding_asset.{name} must be a 64-character SHA-256")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(
                    f"semantic_embedding_asset.{name} must be a hexadecimal SHA-256"
                ) from exc
        if not isinstance(semantic_embedding_asset["ready_path"], str):
            raise TypeError("semantic_embedding_asset.ready_path must be a string")
        if semantic_embedding_asset["embedding_key"] not in {
            "tokenizer_codebook",
            "ridge_latent",
        }:
            raise ValueError("semantic_embedding_asset.embedding_key is invalid")
        if not isinstance(semantic_embedding_asset["trainable"], bool):
            raise TypeError("semantic_embedding_asset.trainable must be bool")
        expected_source = semantic_embedding_asset.get("expected_source")
        if expected_source not in {None, "tokenizer_effective_codebook"}:
            raise ValueError("semantic_embedding_asset.expected_source is invalid")
        if semantic_embedding_init_std is not None:
            raise ValueError(
                "semantic_embedding_assetandsemantic_embedding_init_stdMutually exclusive"
            )
    semantic_projection_init = conditioning.get(
        "semantic_projection_init",
        "default",
    )
    if semantic_projection_init != "default":
        raise ValueError("conditioning.semantic_projection_init must be default")
    semantic_source_dim = conditioning.get("semantic_source_dim")
    if semantic_source_dim is not None:
        _config_int(
            conditioning,
            "semantic_source_dim",
            minimum=1,
        )
        if semantic_embedding_asset is None and semantic_embedding_init_std is None:
            raise ValueError(
                "Random low-dimensionalsemantic embedding requires semantic_embedding_init_std"
            )
    _config_bool(
        conditioning,
        "semantic_projection_bias",
        default=False,
    )
    flow_config = FlowConfig.from_mapping(config.get("flow") or {})
    if flow_config.source_coupling == "minibatch_ot":
        data_config = config.get("data") or {}
        train_config = config.get("train") or {}
        configured_batches = data_config.get("batch_size_by_bucket")
        if isinstance(configured_batches, Mapping) and configured_batches:
            per_rank_minimum = min(int(value) for value in configured_batches.values())
        else:
            per_rank_minimum = int(train_config.get("batch_size_per_rank", 0))
        expected_world_size = int(train_config.get("expected_world_size", 1))
        coupling_batch = (
            per_rank_minimum * expected_world_size
            if flow_config.source_coupling_scope == "global"
            else per_rank_minimum
        )
        if coupling_batch < 2:
            raise ValueError(
                "minibatch OTconfigured at minimumduration bucketbatch<2;"
                "Rejection of writing a runtime-identical pseudo-alignment configuration"
            )
    text_encoder = conditioning.get("text_encoder")
    if not isinstance(text_encoder, Mapping):
        raise TypeError("conditioning.text_encoder must be a mapping")
    allowed_text_encoder = {
        "model_id",
        "revision",
        "tokenizer_revision",
        "cache_revision",
        "hidden_size",
        "lazy",
        "local_path",
        "asset_lock",
        "local_files_only",
        "use_fast_tokenizer",
        "padding_side",
        "truncation_side",
        "truncation_policy",
        "add_special_tokens",
        "trust_remote_code",
        "hidden_state_selection",
        "position_id_policy",
        "encoder_use_cache",
        "empty_text_policy",
        "frozen_eval_mode",
    }
    unknown_text = set(text_encoder) - allowed_text_encoder
    if unknown_text:
        raise ValueError(f"text_encoder contains unknown fields: {sorted(unknown_text)}")
    if (
        not isinstance(text_encoder.get("model_id"), str)
        or not str(text_encoder["model_id"]).strip()
    ):
        raise ValueError("text_encoder.model_id cannot be empty")
    _config_int(text_encoder, "hidden_size", minimum=1)
    for name, default in (
        ("lazy", True),
        ("local_files_only", True),
        ("use_fast_tokenizer", True),
        ("add_special_tokens", True),
        ("trust_remote_code", False),
        ("encoder_use_cache", False),
        ("frozen_eval_mode", True),
    ):
        _config_bool(text_encoder, name, default=default)
    if text_encoder.get("local_files_only", True) is not True:
        raise ValueError("The DiT text encoder does not allow implicit downloads")
    if text_encoder.get("padding_side", "left") not in {"left", "right"}:
        raise ValueError("text_encoder.padding_side must be left/right")
    if text_encoder.get("truncation_side", "right") not in {"left", "right"}:
        raise ValueError("text_encoder.truncation_side must be left/right")
    if text_encoder.get("truncation_policy", "allow") not in {"allow", "reject"}:
        raise ValueError("text_encoder.truncation_policy must be allow/reject")
    if text_encoder.get("hidden_state_selection", "last_hidden_state") != (
        "last_hidden_state"
    ):
        raise ValueError("text_encoder.hidden_state_selection only supports last_hidden_state")
    if text_encoder.get("position_id_policy", "attention_mask_cumsum") != (
        "attention_mask_cumsum"
    ):
        raise ValueError("text_encoder.position_id_policy only supports attention_mask_cumsum")
    if text_encoder.get("empty_text_policy", "zero_valid_tokens") != (
        "zero_valid_tokens"
    ):
        raise ValueError("text_encoder.empty_text_policy only supports zero_valid_tokens")
    for name in ("local_path", "asset_lock"):
        value = text_encoder.get(name)
        if value is not None and not isinstance(value, str):
            raise TypeError(f"text_encoder.{name} must be a string")





def validate_runtime_training_topology(
    config: Mapping[str, Any],
    *,
    world_size: int,
) -> dict[str, int]:
    train_config = config.get("train") or {}
    configured_world_size = train_config.get("expected_world_size")
    expected_world_size = (
        int(world_size)
        if configured_world_size is None
        else int(configured_world_size)
    )
    if int(world_size) != expected_world_size:
        raise RuntimeError(
            "Renderer world size does not match train.expected_world_size: "
            f"expected={expected_world_size} actual={world_size}"
        )
    batch_size_per_rank = int(train_config.get("batch_size_per_rank", 1))
    accumulation_steps = int(train_config.get("gradient_accumulation_steps", 1))
    effective_global_batch_size = (
        int(world_size) * batch_size_per_rank * accumulation_steps
    )
    expected_global_batch_size = int(
        train_config.get("global_parent_batch_size", effective_global_batch_size)
    )
    if effective_global_batch_size != expected_global_batch_size:
        raise RuntimeError(
            "Renderer global batch size does not match the training recipe: "
            f"expected={expected_global_batch_size} actual={effective_global_batch_size}"
        )
    return {
        "expected_world_size": expected_world_size,
        "actual_world_size": int(world_size),
        "global_batch_size": effective_global_batch_size,
    }











def validate_dit_launch_config(config: Mapping[str, Any]) -> None:
    """Validate the final Renderer recipe before model construction."""
    if not isinstance(config, Mapping):
        raise TypeError("Renderer config must be a mapping")

    allowed_sections = {
        "format_version",
        "architecture",
        "model",
        "conditioning",
        "flow",
        "semantic_corruption",
        "revisions",
        "data",
        "validation",
        "optimizer",
        "train",
    }
    unknown = set(config) - allowed_sections
    if unknown:
        raise ValueError(f"Unknown Renderer config sections: {sorted(unknown)}")
    if config.get("format_version") not in {
        DIT_CONFIG_FORMAT_VERSION,
        DIT_TEST_CONFIG_FORMAT_VERSION,
    }:
        raise ValueError("Unsupported Renderer config format_version")

    required = ("model", "conditioning", "flow", "revisions", "data", "optimizer", "train")
    missing = [name for name in required if not isinstance(config.get(name), Mapping)]
    if missing:
        raise ValueError(f"Renderer config is missing sections: {missing}")

    validate_dit_model_and_conditioning(config)
    FlowConfig.from_mapping(config["flow"])
    revisions = RevisionContract.from_mapping(config["revisions"])
    corruption = SemanticCorruptionConfig.from_mapping(config.get("semantic_corruption"))

    if corruption.mode == "emdc_knn":
        calibration = corruption.calibration or {}
        asset = corruption.asset or {}
        if calibration.get("tokenizer_revision") != revisions.tokenizer_revision:
            raise ValueError("Semantic corruption and Tokenizer revisions differ")
        embedding = config["conditioning"].get("semantic_embedding_asset")
        if not isinstance(embedding, Mapping) or (
            asset.get("source_embedding_tensor_sha256") != embedding.get("tensor_sha256")
        ):
            raise ValueError("Semantic corruption must use the published Tokenizer embedding")

    optimizer = config["optimizer"]
    if str(optimizer.get("name", "AdamW")).lower() != "adamw":
        raise ValueError("Renderer training uses AdamW")
    learning_rate = _config_float(optimizer, "lr", minimum=0.0)
    if learning_rate <= 0:
        raise ValueError("optimizer.lr must be positive")
    _config_float(optimizer, "weight_decay", default=0.0, minimum=0.0)
    betas = optimizer.get("betas", (0.9, 0.95))
    if (
        not isinstance(betas, Sequence)
        or isinstance(betas, (str, bytes))
        or len(betas) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) < 1.0
            for value in betas
        )
    ):
        raise ValueError("optimizer.betas must contain two values in [0, 1)")
    warmup_steps = _config_int(optimizer, "warmup_steps", default=0, minimum=0)
    scheduler = str(optimizer.get("scheduler", "constant"))
    if scheduler not in {"constant", "cosine"}:
        raise ValueError("optimizer.scheduler must be constant or cosine")
    ema_mode = str(optimizer.get("ema_mode", "none"))
    if ema_mode not in DIT_EMA_MODES:
        raise ValueError(f"optimizer.ema_mode must be one of {DIT_EMA_MODES}")
    if ema_mode != "none":
        decay = _config_float(optimizer, "ema_decay", minimum=0.0, maximum=1.0)
        if not 0.0 < decay < 1.0:
            raise ValueError("optimizer.ema_decay must be in (0, 1)")

    train_config = config["train"]
    _config_int(train_config, "seed", minimum=0)
    _config_int(train_config, "batch_size_per_rank", minimum=1)
    _config_int(train_config, "gradient_accumulation_steps", default=1, minimum=1)
    max_steps = _config_int(train_config, "max_steps", minimum=1)
    _config_int(train_config, "save_every_steps", default=max_steps, minimum=1)
    _config_int(train_config, "semantic_embedding_freeze_steps", default=0, minimum=0)
    _config_int(train_config, "activation_checkpoint_block_interval", default=1, minimum=1)
    if warmup_steps > max_steps:
        raise ValueError("optimizer.warmup_steps cannot exceed train.max_steps")
    if str(train_config.get("precision", "fp32")).lower() not in TRAINING_PRECISIONS:
        raise ValueError(f"train.precision must be one of {TRAINING_PRECISIONS}")
    _config_float(train_config, "gradient_clip_norm", default=1.0, minimum=0.0)
    if corruption.mode == "emdc_knn" and (
        corruption.clean_steps >= max_steps
        or corruption.clean_steps + corruption.ramp_steps > max_steps
    ):
        raise ValueError("The corruption warmup and ramp must finish before training ends")

    data = config["data"]
    manifest = data.get("cache_manifest")
    if not isinstance(manifest, (str, Path)) or not str(manifest):
        raise ValueError("data.cache_manifest must point to the materialized training manifest")
    split = str(data.get("split", "train"))
    if split != "train":
        raise ValueError("data.split must be train")
    _config_int(data, "num_workers", default=0, minimum=0)
    _config_bool(data, "drop_last", default=False)
    _config_bool(data, "shuffle", default=True)
    buckets = data.get("duration_buckets_seconds", DURATION_BUCKETS_SECONDS)
    policy = DurationBucketPolicy(tuple(buckets))
    batch_sizes = data.get("batch_size_by_bucket")
    if batch_sizes is not None:
        normalized = {int(key): int(value) for key, value in batch_sizes.items()}
        if set(normalized) != set(policy.buckets_seconds) or any(
            value <= 0 for value in normalized.values()
        ):
            raise ValueError("data.batch_size_by_bucket must cover every duration bucket")

    if data.get("sampler_mode") is not None:
        if data.get("sampler_mode") != "renderer_data_parent_first":
            raise ValueError("The final Renderer recipe uses parent-first sampling")
        if data.get("renderer_data_view") != "short":
            raise ValueError("The final Renderer recipe uses short windows")
        materialized_fields = (
            "ready_path",
            "ready_sha256",
            "index_path",
            "index_sha256",
            "manifest_sha256",
            "expected_records",
            "renderer_data_crop_manifest",
            "renderer_data_crop_manifest_sha256",
            "renderer_data_crop_ready_path",
            "renderer_data_crop_ready_sha256",
            "renderer_data_content_text_cache",
        )
        missing = [name for name in materialized_fields if data.get(name) is None]
        if missing:
            raise ValueError(f"Materialized Renderer data is missing: {missing}")
        if config["conditioning"].get("global_loudness_conditioning"):
            loudness_fields = (
                "renderer_data_loudness_manifest",
                "renderer_data_loudness_manifest_sha256",
                "renderer_data_loudness_ready_path",
                "renderer_data_loudness_ready_sha256",
            )
            missing = [name for name in loudness_fields if data.get(name) is None]
            if missing:
                raise ValueError(f"Materialized loudness data is missing: {missing}")

    validation = config.get("validation")
    if validation is not None:
        if not isinstance(validation, Mapping):
            raise TypeError("validation must be a mapping")
        enabled = _config_bool(validation, "enabled", default=False)
        if enabled:
            if validation.get("split", "valid") != "valid":
                raise ValueError("validation.split must be valid")
            if not validation.get("cache_manifest"):
                raise ValueError("validation.cache_manifest is required")
            _config_int(validation, "every_steps", minimum=1)
            _config_int(validation, "batch_size_per_rank", default=1, minimum=1)
            _config_int(validation, "num_workers", default=0, minimum=0)

    audio_policy = str(train_config.get("audio_seconds_update_policy", "minimum"))
    if audio_policy not in AUDIO_SECONDS_UPDATE_POLICIES:
        raise ValueError(f"audio_seconds_update_policy must be one of {AUDIO_SECONDS_UPDATE_POLICIES}")

def validate_dit_experiment(config: Mapping[str, Any]) -> None:
    """Reject retired experiment metadata through the retained CLI API."""
    if "experiment" in config:
        raise ValueError("The final Renderer config does not accept experiment metadata")


def _runtime_file_identity(
    value: str | Path | None,
    *,
    checkpoint: bool = False,
    expected_sha256: str | None = None,
    verify_sha256: bool = True,
) -> dict[str, Any] | None:
    if value is None:
        return None
    path = Path(value)
    try:
        stat = path.stat()
        if expected_sha256 is not None:
            if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
                raise ValueError("expected_sha256 must be a 64-character SHA-256")
            int(expected_sha256, 16)
        if checkpoint:
            digest = checkpoint_content_fingerprint(path)
        elif verify_sha256 or expected_sha256 is None:
            digest = file_sha256(path)
        else:
            digest = expected_sha256
        if expected_sha256 is not None and digest != expected_sha256:
            raise RuntimeError(
                f"file SHA does not match: expected={expected_sha256} actual={digest}"
            )
        return {
            "path": str(path.resolve()),
            "size": int(stat.st_size),
            "sha256": digest,
        }
    except Exception as exc:
        return {
            "path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _canonical_expected_artifacts(
    config: Mapping[str, Any],
    *,
    identity_config: Mapping[str, Any],
) -> dict[str, Any]:
    # READY files are the single source of truth for materialized artifacts.
    # Logical component revisions are checked separately by the dataset loader.
    return {}


def _canonical_dataset(
    *,
    config: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
    identity_config: Mapping[str, Any],
    revisions: RevisionContract,
    text_model_id: str,
    default_split: str,
    tags_text_cache_loader: RenderTextCacheLoader | None = None,
    lyrics_text_cache_loader: RenderTextCacheLoader | None = None,
    resolve_renderer_data_deferred_parent_text: bool = False,
    verify_published_manifest_records: bool = True,
) -> CanonicalRenderSampleDataset:
    expected_ready_text_cache_revision = None
    if resolve_renderer_data_deferred_parent_text:
        ready_path = Path(str(dataset_config.get("ready_path") or "")).resolve(
            strict=True
        )
        expected_ready_sha = str(dataset_config.get("ready_sha256") or "")
        if file_sha256(ready_path) != expected_ready_sha:
            raise RuntimeError("RendererData parent sample READY SHA-256 mismatch")
        try:
            parent_ready = json.loads(ready_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("RendererData parent sample READY cannot be parsed") from exc
        parent_identities = (
            parent_ready.get("identities")
            if isinstance(parent_ready, Mapping)
            else None
        )
        expected_ready_text_cache_revision = (
            parent_identities.get("text_cache_revision")
            if isinstance(parent_identities, Mapping)
            else None
        )
        if (
            not isinstance(expected_ready_text_cache_revision, str)
            or not expected_ready_text_cache_revision
        ):
            raise RuntimeError("RendererData parent sample READY is missing text cache revision")
    return CanonicalRenderSampleDataset(
        dataset_config["cache_manifest"],
        expected_revisions=asdict(revisions),
        expected_artifacts=_canonical_expected_artifacts(
            config,
            identity_config=identity_config,
        ),
        text_model_id=text_model_id,
        split=str(dataset_config.get("split", default_split)),
        required_quality_profile=dataset_config.get("required_quality_profile"),
        expected_manifest_sha256=dataset_config.get("manifest_sha256"),
        expected_records=dataset_config.get("expected_records"),
        ready_path=dataset_config.get("ready_path"),
        expected_ready_sha256=dataset_config.get("ready_sha256"),
        expected_release_revision=dataset_config.get("sample_release_revision"),
        expected_ready_text_cache_revision=(expected_ready_text_cache_revision),
        index_path=dataset_config.get("index_path"),
        expected_index_sha256=dataset_config.get("index_sha256"),


        verify_manifest_sha256=(
            verify_published_manifest_records and is_main_process()
        ),


        validate_index_records=(
            verify_published_manifest_records and is_main_process()
        ),
        tags_text_cache_loader=tags_text_cache_loader,
        lyrics_text_cache_loader=lyrics_text_cache_loader,
        allow_renderer_data_deferred_parent_text=(
            dataset_config.get("sampler_mode") == "renderer_data_parent_first"
            and dataset_config.get("renderer_data_view") in {"short", "full"}
        ),
        resolve_renderer_data_deferred_parent_text=(resolve_renderer_data_deferred_parent_text),
    )


def _validate_renderer_data_full_text_ready(
    dataset_config: Mapping[str, Any],
    *,
    content_cache: Mapping[str, Any],
) -> dict[str, Any]:
    reference = dataset_config.get("renderer_data_full_text_ready")
    if not isinstance(reference, Mapping):
        raise RuntimeError("RendererData full is missing a full-text READY reference")
    path = Path(str(reference.get("path") or "")).resolve(strict=True)
    expected_sha = str(reference.get("sha256") or "")
    if file_sha256(path) != expected_sha:
        raise RuntimeError("RendererData full-text READY SHA-256 mismatch")
    try:
        ready = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("RendererData full-text READY cannot be parsed") from exc
    split = str(dataset_config.get("split") or "")
    expected_records = dataset_config.get("expected_records")
    split_counts = ready.get("split_counts") if isinstance(ready, Mapping) else None
    base_splits = ready.get("base_splits") if isinstance(ready, Mapping) else None
    base_split = base_splits.get(split) if isinstance(base_splits, Mapping) else None
    valid_split_counts = (
        isinstance(split_counts, Mapping)
        and split in split_counts
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in split_counts.values()
        )
    )
    declared_manifest = (
        Path(str(base_split.get("manifest") or "")).resolve()
        if isinstance(base_split, Mapping)
        else None
    )
    configured_manifest = Path(
        str(dataset_config.get("cache_manifest") or "")
    ).resolve()
    if (
        not isinstance(ready, Mapping)
        or ready.get("schema_version") != "oqm.render.renderer_data-full-text-view-ready.v1"
        or ready.get("status") != "RENDERER_DATA_FULL_TEXT_VIEW_READY"
        or not valid_split_counts
        or ready.get("records") != sum(split_counts.values())
        or split_counts.get(split) != expected_records
        or not isinstance(base_split, Mapping)
        or base_split.get("records") != expected_records
        or declared_manifest != configured_manifest
        or base_split.get("manifest_sha256") != dataset_config.get("manifest_sha256")
        or base_split.get("ready_sha256") != dataset_config.get("ready_sha256")
        or ready.get("lyrics_max_tokens") != 1_792
    ):
        raise RuntimeError("RendererData full-text READY schema, split, count, or length is incompatible")
    text_identity = ready.get("text_cache")
    if not isinstance(text_identity, Mapping):
        raise RuntimeError("RendererData full-text READY is missing text cache identity")
    expected = {
        "cache_config_sha256": content_cache.get("cache_config_sha256"),
        "cache_config_file_sha256": content_cache.get("cache_config_file_sha256"),
        "tags_ready_sha256": (content_cache.get("tags") or {}).get("ready_sha256"),
        "lyrics_ready_sha256": (content_cache.get("lyrics") or {}).get("ready_sha256"),
    }
    mismatches = {
        name: {"expected": value, "actual": text_identity.get(name)}
        for name, value in expected.items()
        if text_identity.get(name) != value
    }
    checks = ready.get("checks")
    if (
        mismatches
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise RuntimeError(f"RendererData full-text READY cache or checks are incomplete: {mismatches}")
    return dict(ready)


def _renderer_data_content_text_loaders(
    dataset_config: Mapping[str, Any],
    *,
    expected_provenance: TextEncoderProvenance,
) -> tuple[RenderTextCacheLoader | None, RenderTextCacheLoader | None]:
    content_cache = dataset_config.get("renderer_data_content_text_cache")
    if content_cache is None:
        return None, None
    if not isinstance(content_cache, Mapping):
        raise TypeError("renderer_data_content_text_cache must be a mapping")
    tags_ref = content_cache["tags"]
    lyrics_ref = content_cache["lyrics"]
    assert isinstance(tags_ref, Mapping) and isinstance(lyrics_ref, Mapping)
    tags_root = Path(str(tags_ref["root"])).resolve(strict=True)
    lyrics_root = Path(str(lyrics_ref["root"])).resolve(strict=True)
    for name, root, expected_sha in (
        ("tags", tags_root, tags_ref["ready_sha256"]),
        ("lyrics", lyrics_root, lyrics_ref["ready_sha256"]),
    ):
        ready_path = root / "READY"
        if not ready_path.is_file() or file_sha256(ready_path) != expected_sha:
            raise RuntimeError(f"RendererData {name} text cache READY is missing or has a SHA mismatch")
    if dataset_config.get("renderer_data_view") == "full":
        _validate_renderer_data_full_text_ready(
            dataset_config,
            content_cache=content_cache,
        )
    loader_kwargs = {
        "expected_provenance": expected_provenance,
        "expected_cache_config_sha256": str(content_cache["cache_config_sha256"]),
        "expected_cache_config_file_sha256": str(
            content_cache["cache_config_file_sha256"]
        ),
    }
    return (
        RenderTextCacheLoader(tags_root, **loader_kwargs),
        RenderTextCacheLoader(lyrics_root, **loader_kwargs),
    )


def _build_sample_dataset(
    *,
    config: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
    identity_config: Mapping[str, Any],
    revisions: RevisionContract,
    text_model_id: str,
    default_split: str,
    renderer_data_evaluation_selection: Mapping[str, Any] | None = None,
) -> (
    CachedRenderDataset
    | CanonicalRenderSampleDataset
    | FixedCropSelectionDataset
    | RendererDataShortWindowDataset
):
    manifest_format = str(
        dataset_config.get(
            "manifest_format",
            identity_config.get("manifest_format", RENDER_SAMPLE_SCHEMA),
        )
    )
    if manifest_format == RENDER_SAMPLE_SCHEMA:
        text_loaders = (
            _renderer_data_content_text_loaders(
                dataset_config,
                expected_provenance=revisions.text_provenance(text_model_id),
            )
            if dataset_config.get("sampler_mode") == "renderer_data_parent_first"
            else (None, None)
        )
        dataset = _canonical_dataset(
            config=config,
            dataset_config=dataset_config,
            identity_config=identity_config,
            revisions=revisions,
            text_model_id=text_model_id,
            default_split=default_split,
            tags_text_cache_loader=text_loaders[0],
            lyrics_text_cache_loader=text_loaders[1],
            resolve_renderer_data_deferred_parent_text=(
                dataset_config.get("sampler_mode") == "renderer_data_parent_first"
                and dataset_config.get("renderer_data_view") == "full"
            ),
            verify_published_manifest_records=(renderer_data_evaluation_selection is None),
        )
        if dataset_config.get("sampler_mode") == "renderer_data_parent_first":
            view = str(dataset_config.get("renderer_data_view") or "")
            if view == "full":
                return dataset
            if view != "short":
                raise RuntimeError("RendererData parent-first is missing the short or full view")
            return RendererDataShortWindowDataset(
                dataset,
                crop_manifest_path=dataset_config["renderer_data_crop_manifest"],
                expected_crop_manifest_sha256=str(
                    dataset_config["renderer_data_crop_manifest_sha256"]
                ),
                crop_ready_path=dataset_config["renderer_data_crop_ready_path"],
                expected_crop_ready_sha256=str(
                    dataset_config["renderer_data_crop_ready_sha256"]
                ),
                verify_parent_records=is_main_process(),
                tags_text_cache_loader=text_loaders[0],
                lyrics_text_cache_loader=text_loaders[1],
                parent_subset=dataset_config.get("renderer_data_parent_subset"),
                window_subset=dataset_config.get("renderer_data_window_subset"),
                loudness_manifest_path=dataset_config.get("renderer_data_loudness_manifest"),
                expected_loudness_manifest_sha256=dataset_config.get(
                    "renderer_data_loudness_manifest_sha256"
                ),
                loudness_ready_path=dataset_config.get("renderer_data_loudness_ready_path"),
                expected_loudness_ready_sha256=dataset_config.get(
                    "renderer_data_loudness_ready_sha256"
                ),
                evaluation_selection=renderer_data_evaluation_selection,
            )
        selection_path = dataset_config.get("fixed_crop_selection_path")
        if selection_path is None:
            return dataset
        return FixedCropSelectionDataset(
            dataset,
            selection_path=selection_path,
            expected_selection_sha256=str(
                dataset_config["fixed_crop_selection_sha256"]
            ),
            expected_source_manifest_sha256=str(dataset_config["manifest_sha256"]),
        )
    raise ValueError(f"Unknown DiT manifest_format={manifest_format!r}")


def _assert_train_valid_disjoint(
    train_dataset: Dataset[Any],
    valid_dataset: Dataset[Any],
) -> None:
    if isinstance(train_dataset, RendererDataShortWindowDataset):
        train_dataset = train_dataset.parent_dataset
    if isinstance(train_dataset, RendererDataFixedValidationDataset):
        train_dataset = train_dataset.source.parent_dataset
    if isinstance(valid_dataset, RendererDataShortWindowDataset):
        valid_dataset = valid_dataset.parent_dataset
    if isinstance(valid_dataset, RendererDataFixedValidationDataset):
        valid_dataset = valid_dataset.source.parent_dataset
    if not isinstance(
        train_dataset,
        CanonicalRenderSampleDataset,
    ) or not isinstance(valid_dataset, CanonicalRenderSampleDataset):
        return
    if (
        train_dataset.split_groups_disjoint_verified
        and valid_dataset.split_groups_disjoint_verified
        and train_dataset.source_audio_manifest_sha256 is not None
        and train_dataset.source_audio_manifest_sha256
        == valid_dataset.source_audio_manifest_sha256
    ):


        return
    identity_fields = (
        ("source_audio_sha256", "audio", "source_audio_sha256"),
        ("derived_audio_sha256", "audio", "sha256"),
        ("recording_group_id", "groups", "recording_group_id"),
        ("performance_group_id", "groups", "performance_group_id"),
        ("composition_group_id", "groups", "composition_group_id"),
        ("song_group_id", "groups", "song_group_id"),
        ("variant_group_id", "groups", "variant_group_id"),
        ("artist_group_id", "groups", "artist_group_id"),
        ("album_group_id", "groups", "album_group_id"),
        ("leakage_group_id", "groups", "leakage_group_id"),
    )

    def identities(
        dataset: CanonicalRenderSampleDataset,
        section: str,
        field: str,
    ) -> set[str]:
        result = set()
        for row in dataset.records:
            owner = row.get(section)
            if not isinstance(owner, Mapping):
                raise RuntimeError(f"canonical record is missing {section} mapping")
            value = owner.get(field)
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise RuntimeError(f"canonical {section}.{field} is invalid")
                result.add(value)
        return result

    valid_ids = set(valid_dataset.sample_ids)
    valid_identities = {
        name: identities(valid_dataset, section, field)
        for name, section, field in identity_fields
    }
    sample_overlap: list[str] = []
    overlaps: dict[str, list[str]] = {}
    for row in train_dataset.iter_records():
        sample_id = str(row["sample_id"])
        if sample_id in valid_ids and len(sample_overlap) < 8:
            sample_overlap.append(sample_id)
        for name, section, field in identity_fields:
            owner = row.get(section)
            if not isinstance(owner, Mapping):
                raise RuntimeError(f"canonical record is missing {section} mapping")
            value = owner.get(field)
            if value is None:
                continue
            if not isinstance(value, str) or not value:
                raise RuntimeError(f"canonical {section}.{field} is invalid")
            if (
                value in valid_identities[name]
                and len(overlaps.setdefault(name, [])) < 8
            ):
                overlaps[name].append(value)
    overlaps = {name: values for name, values in overlaps.items() if values}
    if sample_overlap or overlaps:
        raise RuntimeError(
            "DiT training and validation sets must isolate samples, audio, and groups: "
            f"sample_overlap={sample_overlap[:8]} "
            f"identity_overlap="
            f"{ {name: values[:8] for name, values in overlaps.items()} }"
        )


def _distributed_training_state(
    config: Mapping[str, Any],
    *,
    world_size: int,
    manifest_identity: Mapping[str, Any] | None,
    validation_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    data = config["data"]
    train_config = config["train"]
    buckets = tuple(
        int(value)
        for value in data.get("duration_buckets_seconds", DURATION_BUCKETS_SECONDS)
    )
    configured = data.get("batch_size_by_bucket")
    if configured:
        batch_sizes = {str(int(key)): int(value) for key, value in configured.items()}
    else:
        batch_sizes = {
            str(bucket): int(train_config["batch_size_per_rank"]) for bucket in buckets
        }
    return {
        "format_version": TRAINING_TOPOLOGY_VERSION,
        "world_size": int(world_size),
        "batch_size_per_rank": int(train_config["batch_size_per_rank"]),
        "global_parent_batch_size": (
            int(train_config["global_parent_batch_size"])
            if train_config.get("global_parent_batch_size") is not None
            else None
        ),
        "batch_size_by_bucket": {key: batch_sizes[key] for key in sorted(batch_sizes)},
        "gradient_accumulation_steps": int(
            train_config.get("gradient_accumulation_steps", 1)
        ),
        "global_audio_seconds_per_update": float(
            train_config.get("global_audio_seconds_per_update", 0.0)
        ),
        "audio_seconds_update_policy": str(
            train_config.get("audio_seconds_update_policy", "minimum")
        ),
        "duration_buckets_seconds": list(buckets),
        "sampler_drop_last": bool(data.get("drop_last", False)),
        "sampler_mode": str(data.get("sampler_mode", "duration_bucket")),
        "renderer_data_view": data.get("renderer_data_view"),
        "renderer_data_crop_manifest_sha256": data.get("renderer_data_crop_manifest_sha256"),
        "renderer_data_crop_ready_sha256": data.get("renderer_data_crop_ready_sha256"),
        "manifest": dict(manifest_identity or {}),
        "validation": dict(validation_identity or {}),
        "config_hash": config_hash(dict(config)),
    }


def _seed_process(
    seed: int,
    rank: int,
    device: torch.device,
    *,
    rank_seed_stride: int = 1_000_003,
) -> None:
    if (
        not isinstance(rank_seed_stride, int)
        or isinstance(rank_seed_stride, bool)
        or rank_seed_stride <= 0
    ):
        raise ValueError("rank_seed_stride must be a positive integer")
    value = int(seed) + int(rank) * rank_seed_stride
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if device.type == "cuda":
        torch.cuda.manual_seed(value)


def _configure_deterministic_execution(enabled: bool) -> dict[str, Any]:
    if not isinstance(enabled, bool):
        raise TypeError("train.deterministic must be bool")
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = enabled
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if enabled:


            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        else:

            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_cudnn_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
    return {
        "enabled": enabled,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": (
            bool(torch.backends.cuda.matmul.allow_tf32)
            if torch.cuda.is_available()
            else None
        ),
        "cudnn_allow_tf32": (
            bool(torch.backends.cudnn.allow_tf32) if torch.cuda.is_available() else None
        ),
        "flash_sdp_enabled": (
            bool(torch.backends.cuda.flash_sdp_enabled())
            if torch.cuda.is_available()
            else None
        ),
        "mem_efficient_sdp_enabled": (
            bool(torch.backends.cuda.mem_efficient_sdp_enabled())
            if torch.cuda.is_available()
            else None
        ),
        "math_sdp_enabled": (
            bool(torch.backends.cuda.math_sdp_enabled())
            if torch.cuda.is_available()
            else None
        ),
        "cudnn_sdp_enabled": (
            bool(torch.backends.cuda.cudnn_sdp_enabled())
            if torch.cuda.is_available()
            else None
        ),
    }


def configure_dit_evaluation_determinism(
    device: torch.device | str,
) -> dict[str, Any]:

    resolved_device = torch.device(device)
    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if (
        resolved_device.type == "cuda"
        and workspace_config not in DIT_DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS
    ):
        raise RuntimeError(
            "CUDAbit by bitDiTReview requestCUBLAS_WORKSPACE_CONFIGis"
            f"{DIT_DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS},received"
            f"{workspace_config!r}"
        )
    state = _configure_deterministic_execution(True)
    return {
        **state,
        "cublas_workspace_config": workspace_config,
        "required_for_exact_replay": True,
    }


def _next_loader_batch(
    iterator: Iterator[Any], *, event: str
) -> tuple[Any | None, bool]:
    batch = None
    status = "ok"
    error = None
    try:
        batch = next(iterator)
    except StopIteration:
        status = "end"
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    assert_distributed_consensus(event, {"status": status, "error": error})
    if error is not None:
        raise RuntimeError(f"Render DataLoader failed: {error}")
    return batch, status == "end"


def resolve_initialization_paths(
    config: Mapping[str, Any],
    *,
    init_from: str | Path | None,
    resume_from: str | Path | None,
) -> tuple[str | Path | None, str | Path | None]:

    train_config = config["train"]
    configured_init = train_config.get("init_from")
    configured_resume = train_config.get("resume_from")
    if configured_init is not None and configured_resume is not None:
        raise ValueError("train.init_from and train.resume_from cannot be configured at the same time")
    if init_from is not None and resume_from is not None:
        raise ValueError("--init-from and --resume-from cannot be provided together")
    if resume_from is not None:
        if configured_resume is not None and str(resume_from) != str(configured_resume):
            raise ValueError("CLI --resume-from and train.resume_from mismatch")
        resolved_init = None
        resolved_resume = resume_from
    elif init_from is not None:
        if configured_init is not None and str(init_from) != str(configured_init):
            raise ValueError("CLI --init-from and train.init_from mismatch")
        if configured_resume is not None:
            raise ValueError("CLI --init-from and train.resume_from conflict")
        resolved_init = init_from
        resolved_resume = None
    else:
        resolved_init = configured_init
        resolved_resume = configured_resume
    return resolved_init, resolved_resume


def train(
    config: Mapping[str, Any],
    *,
    fake_text_encoder: nn.Module | None = None,
    checkpoint_adapter: CheckpointAdapter | None = None,
    init_from: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> Path:

    process_started_at = time.perf_counter()

    def validate_local_launch() -> None:
        validate_dit_launch_config(config)
        return None


    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        runtime_code_identity = validate_local_launch()
    else:
        runtime_code_identity = None
    rank, local_rank, world_size, distributed_device = init_distributed()
    try:
        if world_size > 1:
            launch_validation_error: str | None = None
            try:
                runtime_code_identity = validate_local_launch()
            except Exception as exc:
                launch_validation_error = f"{type(exc).__name__}: {exc}"
            assert_distributed_consensus(
                "render_launch_config",
                {"error": launch_validation_error},
            )
            if launch_validation_error is not None:
                raise RuntimeError(
                    f"Render DiTStatic configuration verification failed: {launch_validation_error}"
                )
        train_config = config["train"]
        runtime_training_topology = validate_runtime_training_topology(
            config,
            world_size=world_size,
        )
        init_from, resume_from = resolve_initialization_paths(
            config,
            init_from=init_from,
            resume_from=resume_from,
        )
        configured_device = torch.device(
            str(
                train_config.get(
                    "device",
                    "cuda" if torch.cuda.is_available() else "cpu",
                )
            )
        )
        if world_size > 1:
            if configured_device.type != distributed_device.type:
                raise RuntimeError(
                    "train.device and common.distributed selected backend/device "
                    f"inconsistent:config={configured_device} runtime={distributed_device}"
                )
            device = distributed_device
        else:
            device = configured_device
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("Configuration requests CUDA, but no GPU is available to this process")
        deterministic_execution = _configure_deterministic_execution(
            bool(train_config.get("deterministic", False))
        )

        manifest_identity = {
            "manifest": _runtime_file_identity(
                config["data"]["cache_manifest"],
                expected_sha256=config["data"].get("manifest_sha256"),
                verify_sha256=False,
            ),
            "ready": _runtime_file_identity(
                config["data"].get("ready_path"),
                expected_sha256=config["data"].get("ready_sha256"),
            ),
            "index": _runtime_file_identity(
                config["data"].get("index_path"),
                expected_sha256=config["data"].get("index_sha256"),
            ),
        }
        validation_config = config.get("validation") or {}
        validation_enabled = bool(validation_config.get("enabled", False))
        validation_identity = (
            {
                "manifest": _runtime_file_identity(
                    validation_config.get("cache_manifest"),
                    expected_sha256=validation_config.get("manifest_sha256"),
                    verify_sha256=False,
                ),
                "ready": _runtime_file_identity(
                    validation_config.get("ready_path"),
                    expected_sha256=validation_config.get("ready_sha256"),
                ),
                "index": _runtime_file_identity(
                    validation_config.get("index_path"),
                    expected_sha256=validation_config.get("index_sha256"),
                ),
            }
            if validation_enabled
            else None
        )
        distributed_state = _distributed_training_state(
            config,
            world_size=world_size,
            manifest_identity=manifest_identity,
            validation_identity=validation_identity,
        )
        checkpoint_path = resume_from or init_from
        resolved_checkpoint_path: Path | None = None
        checkpoint_resolution_error: str | None = None
        if checkpoint_path is not None:
            try:
                resolved_checkpoint_path = Path(checkpoint_path).resolve(strict=True)
            except Exception as exc:
                checkpoint_resolution_error = f"{type(exc).__name__}: {exc}"
        stop_after_checkpoint_step = int(
            os.environ.get("OQM_STOP_AFTER_CHECKPOINT_STEP", "0")
        )
        startup_contract = {
            "config_hash": config_hash(dict(config)),
            "cache_contract": {
                "revisions": config.get("revisions"),
                "manifest": manifest_identity,
                "validation": validation_identity,
            },
            "topology": distributed_state,
            "checkpoint": _runtime_file_identity(
                resolved_checkpoint_path,
                checkpoint=True,
            ),
            "checkpoint_resolution_error": checkpoint_resolution_error,
            "runtime_code_identity": runtime_code_identity,
            "runtime_training_topology": runtime_training_topology,
            "deterministic_execution": deterministic_execution,
            "init_mode": (
                "resume"
                if resume_from is not None
                else "init"
                if init_from is not None
                else "fresh"
            ),
            "stop_after_checkpoint_step": stop_after_checkpoint_step,
        }

        assert_distributed_consensus("render_startup", startup_contract)
        if checkpoint_resolution_error is not None:
            raise RuntimeError(
                f"Renderer checkpoint could not be parsed: {checkpoint_resolution_error}"
            )
        checkpoint_identity = startup_contract["checkpoint"]
        expected_checkpoint_sha256 = (
            str(checkpoint_identity["sha256"])
            if isinstance(checkpoint_identity, Mapping)
            and isinstance(checkpoint_identity.get("sha256"), str)
            else None
        )
        configured_checkpoint_sha = (
            train_config.get("resume_from_sha256")
            if resume_from is not None
            else train_config.get("init_from_sha256")
            if init_from is not None
            else None
        )
        if (
            configured_checkpoint_sha is not None
            and expected_checkpoint_sha256 != configured_checkpoint_sha
        ):
            raise RuntimeError(
                "Render checkpoint SHAis mismatch with the training configuration:"
                f"expected={configured_checkpoint_sha} "
                f"actual={expected_checkpoint_sha256}"
            )

        output_dir = Path(train_config["output_dir"])
        output_error = None
        if is_main_process():
            try:
                if (
                    resume_from is None
                    and output_dir.exists()
                    and any(output_dir.iterdir())
                ):
                    raise FileExistsError(
                        "fresh/initTraining refuses to reuse non-nulloutput_dir;"
                        "Please use a new directory or explicit--resume-from"
                    )
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                output_error = f"{type(exc).__name__}: {exc}"
        raise_if_rank0_error(output_error, action="Render output directory")
        barrier()

        _seed_process(
            int(train_config["seed"]),
            rank,
            device,
            rank_seed_stride=int(train_config.get("rank_seed_stride", 1_000_003)),
        )
        trainer = build_trainer_from_config(
            config,
            fake_text_encoder=fake_text_encoder,
            checkpoint_adapter=checkpoint_adapter,
            rank=rank,
            world_size=world_size,
            distributed_state=distributed_state,
            device=device,
        )
        if resume_from is not None:
            assert resolved_checkpoint_path is not None
            trainer.load_checkpoint(
                resolved_checkpoint_path,
                resume=True,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
            )
        elif init_from is not None:
            assert resolved_checkpoint_path is not None
            trainer.load_checkpoint(
                resolved_checkpoint_path,
                resume=False,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
            )
        if trainer.checkpoint_adapter is None:
            raise RuntimeError("Renderer train() requires the strict checkpoint adapter")

        dataset: (
            CachedRenderDataset
            | CanonicalRenderSampleDataset
            | FixedCropSelectionDataset
            | RendererDataShortWindowDataset
            | None
        ) = None
        valid_dataset: (
            CachedRenderDataset
            | CanonicalRenderSampleDataset
            | FixedCropSelectionDataset
            | RendererDataShortWindowDataset
            | None
        ) = None
        sampler: (
            DistributedDurationBucketBatchSampler | RendererDataParentBatchSampler | None
        ) = None
        data_error = None
        data_init_started_at = time.perf_counter()
        try:
            dataset = _build_sample_dataset(
                config=config,
                dataset_config=config["data"],
                identity_config=config["data"],
                revisions=trainer.revisions,
                text_model_id=trainer.unwrapped_conditioner.provenance.model_id,
                default_split="train",
            )
            if validation_enabled:
                valid_dataset = _build_sample_dataset(
                    config=config,
                    dataset_config=validation_config,
                    identity_config=config["data"],
                    revisions=trainer.revisions,
                    text_model_id=trainer.unwrapped_conditioner.provenance.model_id,
                    default_split="valid",
                )
                _assert_train_valid_disjoint(dataset, valid_dataset)
            policy = DurationBucketPolicy(
                tuple(
                    int(value)
                    for value in config["data"].get(
                        "duration_buckets_seconds",
                        DURATION_BUCKETS_SECONDS,
                    )
                )
            )
            sampler_mode = str(config["data"].get("sampler_mode", "duration_bucket"))
            if bool(config["data"].get("require_exact_bucket_frames", False)):
                validate_exact_bucket_frame_lengths(
                    dataset.frame_lengths,
                    policy=policy,
                )
            configured_batch_sizes = config["data"].get("batch_size_by_bucket")
            if configured_batch_sizes:
                batch_size_by_bucket = {
                    int(key): int(value)
                    for key, value in configured_batch_sizes.items()
                }
            else:
                batch_size_by_bucket = {
                    bucket: int(train_config["batch_size_per_rank"])
                    for bucket in policy.buckets_seconds
                }
            if sampler_mode == "renderer_data_parent_first":
                renderer_data_view = str(config["data"]["renderer_data_view"])
                if renderer_data_view == "short" and not isinstance(
                    dataset, RendererDataShortWindowDataset
                ):
                    raise RuntimeError("RendererData short view is not a constructed short-window dataset")
                if renderer_data_view == "full" and not isinstance(
                    dataset, CanonicalRenderSampleDataset
                ):
                    raise RuntimeError("RendererData full must consume the canonical parent directly")
                sampler = RendererDataParentBatchSampler(
                    sample_ids=dataset.sample_ids,
                    rank=rank,
                    world_size=world_size,
                    batch_size_per_rank=int(train_config["batch_size_per_rank"]),
                    gradient_accumulation_steps=int(
                        train_config.get("gradient_accumulation_steps", 1)
                    ),
                    global_parent_batch_size=int(
                        train_config["global_parent_batch_size"]
                    ),
                    seed=int(train_config["seed"]),
                    training_config_hash=distributed_state["config_hash"],
                    view=renderer_data_view,
                    window_counts=(
                        dataset.window_counts
                        if isinstance(dataset, RendererDataShortWindowDataset)
                        else None
                    ),
                    window_frame_lengths=(
                        dataset.window_frame_lengths
                        if isinstance(dataset, RendererDataShortWindowDataset)
                        else None
                    ),
                )
            else:
                sampler = DistributedDurationBucketBatchSampler(
                    dataset.frame_lengths,
                    policy=policy,
                    batch_size_by_bucket=batch_size_by_bucket,
                    rank=rank,
                    world_size=world_size,
                    seed=int(train_config["seed"]),
                    training_config_hash=distributed_state["config_hash"],
                    sample_ids=getattr(dataset, "sample_ids", None),
                    duration_curriculum=config["data"].get("duration_curriculum"),
                    shuffle=bool(config["data"].get("shuffle", True)),
                    drop_last=bool(config["data"].get("drop_last", False)),
                )
            if trainer.sampler_state is not None:
                sampler.load_state_dict(trainer.sampler_state)
                if (
                    sampler.epoch != trainer.epoch
                    or sampler.cursor != trainer.batches_consumed
                ):
                    raise RuntimeError(
                        "checkpoint training_state and sampler epoch/cursor mismatch"
                    )
            else:
                sampler.set_epoch(trainer.epoch)
                if trainer.batches_consumed != 0:
                    raise RuntimeError("checkpoint declares a batch position but is missing sampler_state")
            if sampler.set_global_step(trainer.global_step):
                trainer.batches_consumed = sampler.cursor
            trainer.sampler_state = sampler.state_dict()
        except Exception as exc:
            data_error = f"{type(exc).__name__}: {exc}"
        assert_distributed_consensus(
            "render_data_sampler_init",
            {
                "error": data_error,
                "sampler_state": (
                    sampler.state_dict() if sampler is not None else None
                ),
                "validation_records": (
                    len(valid_dataset) if valid_dataset is not None else None
                ),
                "train_index_sha256": getattr(dataset, "index_sha256", None),
                "valid_index_sha256": (
                    getattr(valid_dataset, "index_sha256", None)
                    if valid_dataset is not None
                    else None
                ),
            },
        )
        if data_error is not None or dataset is None or sampler is None:
            raise RuntimeError(f"Render data/sampler initialization failed: {data_error}")
        local_data_init_seconds = time.perf_counter() - data_init_started_at

        trainer.wrap_distributed(local_rank=local_rank, resume=resume_from is not None)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        barrier()

        num_workers = int(config["data"].get("num_workers", 0))
        rank_seed_stride = int(train_config.get("rank_seed_stride", 1_000_003))
        loader_generator = torch.Generator(device="cpu").manual_seed(
            int(train_config["seed"])
            + rank * rank_seed_stride
            + int(train_config.get("data_loader_seed_offset", 53))
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_cached_render_batch,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
            generator=loader_generator,
        )
        iterator: Iterator[Any] = iter(loader)
        valid_loader: DataLoader[Any] | None = None
        expected_validation_sample_ids: tuple[str, ...] | None = None
        if valid_dataset is not None:
            if isinstance(valid_dataset, RendererDataShortWindowDataset):
                valid_dataset = RendererDataFixedValidationDataset(
                    valid_dataset,
                    seed=int(validation_config["seed"]),
                )
            if (
                isinstance(valid_dataset, CanonicalRenderSampleDataset)
                and is_main_process()
            ):
                expected_validation_sample_ids = valid_dataset.sample_ids
            valid_indices = list(range(rank, len(valid_dataset), world_size))
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=int(validation_config.get("batch_size_per_rank", 1)),
                sampler=valid_indices,
                shuffle=False,
                drop_last=False,
                num_workers=int(validation_config.get("num_workers", 0)),
                collate_fn=collate_cached_render_batch,
                pin_memory=device.type == "cuda",
                persistent_workers=int(validation_config.get("num_workers", 0)) > 0,
                generator=torch.Generator(device="cpu").manual_seed(
                    int(validation_config["seed"])
                    + rank * rank_seed_stride
                    + int(
                        train_config.get(
                            "validation_loader_seed_offset",
                            71,
                        )
                    )
                ),
            )


        trainer.restore_rng_after_ddp_init()
        startup_timings = gather_object_to_rank0(
            {
                "rank": rank,
                "data_init_seconds": local_data_init_seconds,
                "total_startup_seconds": time.perf_counter() - process_started_at,
            }
        )
        startup_log_error = None
        if rank == 0 and trainer.logger is not None:
            try:
                assert startup_timings is not None
                trainer.logger.write(
                    {
                        "event": "startup",
                        "world_size": world_size,
                        "rank_data_init_seconds": [
                            float(value["data_init_seconds"])
                            for value in startup_timings
                        ],
                        "max_data_init_seconds": max(
                            float(value["data_init_seconds"])
                            for value in startup_timings
                        ),
                        "rank_total_startup_seconds": [
                            float(value["total_startup_seconds"])
                            for value in startup_timings
                        ],
                        "max_total_startup_seconds": max(
                            float(value["total_startup_seconds"])
                            for value in startup_timings
                        ),
                        "uses_published_index": bool(
                            isinstance(dataset, CanonicalRenderSampleDataset)
                            and dataset.uses_published_index
                        ),
                        "train_index_sha256": getattr(
                            dataset,
                            "index_sha256",
                            None,
                        ),
                        "valid_index_sha256": getattr(
                            valid_dataset,
                            "index_sha256",
                            None,
                        ),
                        "deterministic_execution": deterministic_execution,
                        "semantic_corruption": (trainer.semantic_corruption.to_dict()),
                        "semantic_error_calibration": (
                            trainer.semantic_error_calibration_provenance
                        ),
                        "semantic_distractor_asset": (
                            trainer.semantic_distractor_table.provenance
                            if trainer.semantic_distractor_table is not None
                            else None
                        ),
                        "duration_curriculum": sampler.curriculum_telemetry,
                    }
                )
            except Exception as exc:
                startup_log_error = f"{type(exc).__name__}: {exc}"
        raise_if_rank0_error(
            startup_log_error,
            action="Render startup telemetry",
        )
        configured_max_steps = int(train_config["max_steps"])
        max_steps = configured_max_steps
        save_every = int(train_config.get("save_every_steps", configured_max_steps))
        validation_patience = int(validation_config.get("patience", 0))
        target_audio_seconds = float(
            train_config.get("global_audio_seconds_per_update", 0.0)
        )
        audio_seconds_policy = str(
            train_config.get("audio_seconds_update_policy", "minimum")
        )
        minimum_microbatches = int(train_config.get("gradient_accumulation_steps", 1))
        if (
            configured_max_steps <= 0
            or max_steps <= 0
            or save_every <= 0
            or not math.isfinite(target_audio_seconds)
            or target_audio_seconds < 0
            or minimum_microbatches <= 0
        ):
            raise ValueError("Training step, save interval, audio interval, or gradient accumulation configuration is invalid")
        if audio_seconds_policy not in AUDIO_SECONDS_UPDATE_POLICIES:
            raise ValueError(
                f"audio_seconds_update_policy must be{AUDIO_SECONDS_UPDATE_POLICIES}"
            )
        if (
            stop_after_checkpoint_step < 0
            or stop_after_checkpoint_step > max_steps
            or (
                stop_after_checkpoint_step > 0
                and stop_after_checkpoint_step <= trainer.global_step
            )
        ):
            raise ValueError(
                "OQM_STOP_AFTER_CHECKPOINT_STEP must be located at the current step and the operating upper limit"
            )

        final_checkpoint = output_dir / "last.pt"


        stop_requested = (
            validation_enabled
            and validation_patience > 0
            and trainer.valid_without_improvement >= validation_patience
        )
        while trainer.global_step < max_steps and not stop_requested:
            curriculum_transition = sampler.set_global_step(trainer.global_step)
            if curriculum_transition:
                iterator = iter(loader)
                trainer.epoch = sampler.epoch
                trainer.batches_consumed = sampler.cursor
                trainer.sampler_state = sampler.state_dict()
                transition_log_error = None
                if rank == 0 and trainer.logger is not None:
                    try:
                        trainer.logger.write(
                            {
                                "event": "duration_curriculum_transition",
                                "global_step": trainer.global_step,
                                "duration_curriculum": sampler.curriculum_telemetry,
                            }
                        )
                    except Exception as exc:
                        transition_log_error = f"{type(exc).__name__}: {exc}"
                raise_if_rank0_error(
                    transition_log_error,
                    action=(
                        "Render duration curriculum transition "
                        f"step={trainer.global_step}"
                    ),
                )
            microbatches: list[Any] = []
            accumulated_global_audio_seconds = 0.0
            while len(microbatches) < minimum_microbatches or (
                target_audio_seconds > 0
                and accumulated_global_audio_seconds < target_audio_seconds
            ):
                batch, ended = _next_loader_batch(
                    iterator,
                    event=(
                        "render_dataloader_next:"
                        f"epoch={sampler.epoch}:cursor={sampler.cursor}"
                    ),
                )
                if ended:
                    sampler.set_epoch(sampler.epoch + 1)
                    iterator = iter(loader)
                    continue
                assert batch is not None
                sampler.advance()
                microbatches.append(batch)
                local_audio_seconds = DurationBucketPolicy.audio_seconds(
                    batch["semantic_mask"]
                )
                accumulated_global_audio_seconds += reduce_scalar_sum(
                    local_audio_seconds
                )

            if (
                audio_seconds_policy == "exact"
                and target_audio_seconds > 0
                and not math.isclose(
                    accumulated_global_audio_seconds,
                    target_audio_seconds,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
            ):
                raise RuntimeError(
                    "exact audio-seconds/update cannot be replaced by the currentglobal microbatchconsists of:"
                    f"target={target_audio_seconds} "
                    f"actual={accumulated_global_audio_seconds}"
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            update_started_at = time.perf_counter()
            train_metrics = trainer.train_update(microbatches)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            update_seconds = time.perf_counter() - update_started_at
            runtime_rows = gather_object_to_rank0(
                {
                    "rank": rank,
                    "update_seconds": update_seconds,
                    "cuda_max_memory_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else 0
                    ),
                    "cuda_max_memory_reserved_bytes": (
                        int(torch.cuda.max_memory_reserved(device))
                        if device.type == "cuda"
                        else 0
                    ),
                }
            )
            runtime_log_error = None
            if rank == 0 and trainer.logger is not None:
                try:
                    assert runtime_rows is not None
                    max_update_seconds = max(
                        float(value["update_seconds"]) for value in runtime_rows
                    )
                    trainer.logger.write(
                        {
                            "event": "train_step_runtime",
                            "global_step": trainer.global_step,
                            "world_size": world_size,
                            "max_update_seconds": max_update_seconds,
                            "audio_seconds_per_update": float(
                                train_metrics["audio_seconds_per_update"]
                            ),
                            "audio_seconds_per_wall_second": float(
                                train_metrics["audio_seconds_per_update"]
                            )
                            / max(max_update_seconds, 1.0e-12),
                            "max_cuda_memory_allocated_bytes": max(
                                int(value["cuda_max_memory_allocated_bytes"])
                                for value in runtime_rows
                            ),
                            "max_cuda_memory_reserved_bytes": max(
                                int(value["cuda_max_memory_reserved_bytes"])
                                for value in runtime_rows
                            ),
                            "rank_runtime": runtime_rows,
                            "duration_curriculum": sampler.curriculum_telemetry,
                        }
                    )
                except Exception as exc:
                    runtime_log_error = f"{type(exc).__name__}: {exc}"
            raise_if_rank0_error(
                runtime_log_error,
                action=f"Render step runtime log step={trainer.global_step}",
            )
            trainer.epoch = sampler.epoch
            trainer.batches_consumed = sampler.cursor
            trainer.sampler_state = sampler.state_dict()
            stopped_early = False
            external_stop_due = (
                stop_after_checkpoint_step > 0
                and trainer.global_step >= stop_after_checkpoint_step
            )
            if valid_loader is not None and (
                trainer.global_step % int(validation_config["every_steps"]) == 0
                or trainer.global_step == max_steps
                or external_stop_due
            ):
                valid_metrics = trainer.validate(
                    valid_loader,
                    validation_seed=int(validation_config["seed"]),
                    text_drop_probability=float(
                        validation_config.get("text_drop_probability", 0.0)
                    ),
                    expected_sample_ids=expected_validation_sample_ids,
                )
                if bool(validation_config.get("require_full_coverage", True)) and int(
                    valid_metrics["batch_size"]
                ) != len(valid_dataset):
                    raise RuntimeError(
                        "Fixed validation coverage is incomplete: "
                        f"expected={len(valid_dataset)} "
                        f"actual={valid_metrics['batch_size']}"
                    )
                minimum_delta = float(validation_config.get("min_delta", 0.0))
                improved = (
                    valid_metrics["loss"] < trainer.best_valid_loss - minimum_delta
                )
                if improved:
                    trainer.best_valid_loss = float(valid_metrics["loss"])
                    trainer.best_valid_step = trainer.global_step
                    trainer.valid_without_improvement = 0
                else:
                    trainer.valid_without_improvement += 1
                valid_log_error = None
                if trainer.rank == 0 and trainer.logger is not None:
                    try:
                        trainer.logger.write(
                            {
                                **valid_metrics,
                                "best_valid_loss": trainer.best_valid_loss,
                                "best_valid_step": trainer.best_valid_step,
                                "valid_without_improvement": (
                                    trainer.valid_without_improvement
                                ),
                                "improved": improved,
                            }
                        )
                    except Exception as exc:
                        valid_log_error = f"{type(exc).__name__}: {exc}"
                raise_if_rank0_error(
                    valid_log_error,
                    action=f"fixed validation log step={trainer.global_step}",
                )
                if improved and bool(validation_config.get("save_best", True)):
                    trainer.save_checkpoint(output_dir / "best.pt")
                stopped_early = (
                    validation_patience > 0
                    and trainer.valid_without_improvement >= validation_patience
                )
            should_stop = external_stop_due or stopped_early
            should_save = (
                trainer.global_step % save_every == 0
                or trainer.global_step == max_steps
                or should_stop
            )
            if should_save:
                trainer.save_checkpoint(final_checkpoint)
            stop_requested = should_stop
        barrier()
        return final_checkpoint
    finally:
        cleanup_distributed()
