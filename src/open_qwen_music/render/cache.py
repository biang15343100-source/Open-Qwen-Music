
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
from torch import Tensor, nn

from open_qwen_music.common.archive import read_archive_bytes
from open_qwen_music.common.checkpoint import file_sha256

from .checkpoint import FORMAT_VERSION as CHECKPOINT_FORMAT_VERSION
from .checkpoint import read_render_checkpoint_state
from .contracts import (
    AUDIO_CHANNELS,
    LATENT_DIM,
    LATENT_FRAME_HZ,
    LATENT_LAYOUT_FORMAT_VERSION,
    SAMPLE_RATE,
    SPEC_FRAMES_PER_LATENT_FRAME,
    STFT_BINS,
    STFT_CONTRACT_VERSION,
    samples_to_latent_frames,
    validate_latent_layout,
)
from .spec_vae import (
    SoftplusDiagonalGaussianPosterior,
    SpecVAE,
    SpecVAEConfig,
    normalize_spec_vae_checkpoint_config,
    nonfinite_spec_vae_parameter_names,
)
from .stft import StereoSTFT, STFTConfig

CACHE_CONFIG_FORMAT_VERSION = "oqm.render.latent-cache.config.v2"
CACHE_RUNTIME_FORMAT_VERSION = "oqm.render.latent-cache.v2"
RENDER_AUDIO_SCHEMA = "oqm.render-audio.v1"
LATENT_STATS_FORMAT_VERSION = "oqm.render.latent-stats.v3"
LATENT_STATS_SIDECAR_FORMAT_VERSION = "oqm.render.latent-stats.sha256.v1"
LATENT_STATS_SAMPLER_FORMAT_VERSION = "oqm.render.latent-stats-sampler.v1"
LATENT_ARTIFACT_FORMAT_VERSION = "oqm.render.latent-artifact.v3"
LATENT_CACHE_READY_FORMAT_VERSION = "oqm.render-latent-cache-ready.v3"
POSTERIOR_SEED_DERIVATION = "sha256-little-endian-63-v1"
POSTERIOR_MODES = frozenset({"mean", "sample"})
POSTERIOR_EPSILON_DRAW_LAYOUT = "contiguous_bdt"
POSTERIOR_EPSILON_DRAW_LAYOUTS = frozenset({POSTERIOR_EPSILON_DRAW_LAYOUT})
CACHE_DTYPES = frozenset({"float16", "float32"})
LATENT_STATS_DURATION_BUCKETS_SECONDS = (30, 90, 180, 360)


def canonical_json_bytes(value: Any) -> bytes:

    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def build_latent_layout(checkpoint_config: Mapping[str, Any]) -> dict[str, Any]:
    model = checkpoint_config.get("model")
    vae = model.get("spec_vae") if isinstance(model, Mapping) else None
    if not isinstance(vae, Mapping):
        raise ValueError("Spec-VAE checkpoint is missing model.spec_vae")
    SpecVAEConfig.from_dict(dict(vae))
    return {
        "format_version": LATENT_LAYOUT_FORMAT_VERSION,
        "latent_dim": LATENT_DIM,
        "frame_hz": float(LATENT_FRAME_HZ),
        "channel_semantics": "unstructured_continuous",
        "special_channels": [],
        "normalization": "per_channel_affine",
    }


def require_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").lower()
    if len(digest) != 64:
        raise ValueError(f"{field} must be 64 bit SHA-256")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"{field} is not hexadecimal SHA-256") from exc
    return digest


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, value: Any) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    atomic_write_bytes(path, payload)


def atomic_save_npy(path: str | Path, values: np.ndarray) -> str:

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as output:
            np.save(output, values, allow_pickle=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    persisted = np.load(destination, allow_pickle=False)
    if persisted.dtype != values.dtype or persisted.shape != values.shape:
        raise RuntimeError(
            f"{destination} dtype or shape changed after making the tensor contiguous:"
            f"{persisted.dtype}/{persisted.shape}"
        )
    if not np.isfinite(persisted).all():
        raise RuntimeError(f"{destination} includes NaN/Inf")
    return file_sha256(destination)


def _resolve_local_path(uri: Any, *, base_dir: Path, field: str) -> Path:
    text = str(uri or "")
    if not text:
        raise ValueError(f"{field} is empty")
    if text.startswith("file://"):
        raw_path = text.removeprefix("file://")
        if not raw_path.startswith("/"):
            raise ValueError(f"{field}  file URI must contain an absolute path")
        return Path(raw_path)
    if "://" in text:
        raise ValueError(f"{field} must be a local file URI; received {text!r}")
    path = Path(text)
    return path if path.is_absolute() else base_dir / path


@dataclass(frozen=True)
class RenderAudioInput:

    index: int
    sample_id: str
    split: str
    audio_path: Path
    derived_audio_sha256: str
    source_audio_sha256: str
    source_start_sec: float
    source_duration_sec: float
    num_samples: int
    latent_frames: int
    raw_record: dict[str, Any]


@dataclass(frozen=True)
class RenderAudioManifest:
    path: Path
    sha256: str
    records: tuple[RenderAudioInput, ...]


def _validate_exact_audio_length(
    *,
    sample_id: str,
    declared_duration: float,
    actual_frames: int,
) -> None:
    if not math.isfinite(declared_duration) or declared_duration <= 0.0:
        raise ValueError(f"{sample_id} audio.duration_sec must be a finite positive number")
    declared_frames = declared_duration * SAMPLE_RATE
    if not math.isclose(
        declared_frames,
        float(actual_frames),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise ValueError(
            f"{sample_id} Declared duration {declared_duration}s="
            f"{declared_frames}  samples, but the file contains {actual_frames}  samples"
        )


def load_render_audio_manifest(
    path: str | Path,
    *,
    verify_audio_files: bool = True,
) -> RenderAudioManifest:

    if not isinstance(verify_audio_files, bool):
        raise TypeError("verify_audio_files must be a boolean")

    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Render manifest does not exist: {manifest_path}")
    manifest_sha = file_sha256(manifest_path)
    records: list[RenderAudioInput] = []
    seen_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{manifest_path}: {line_number} JSON cannot be parsed"
                ) from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{manifest_path}: {line_number} must be a JSON object")
            if raw.get("schema_version") != RENDER_AUDIO_SCHEMA:
                raise ValueError(
                    f"{manifest_path}: {line_number} schema_version must equal "
                    f"{RENDER_AUDIO_SCHEMA}"
                )
            audio = raw.get("audio")
            if not isinstance(audio, Mapping):
                raise ValueError(f"{manifest_path}: {line_number} audio must be an object")
            sample_id = str(raw.get("sample_id") or "")
            if not sample_id:
                raise ValueError(f"{manifest_path}: {line_number} sample_id must not be empty")
            if sample_id in seen_ids:
                raise ValueError(f"Render manifest contains a duplicate sample_id: {sample_id}")
            seen_ids.add(sample_id)
            split = str(raw.get("split") or "")
            if split not in {"train", "valid", "test"}:
                raise ValueError(
                    f"{sample_id} split must be train, valid, or test; received {split!r}"
                )
            audio_path = _resolve_local_path(
                audio["uri"],
                base_dir=manifest_path.parent,
                field=f"{sample_id}.audio.uri",
            )
            if verify_audio_files:
                audio_path = audio_path.resolve()
            derived_sha = require_sha256(
                audio.get("sha256"), field=f"{sample_id}.audio.sha256"
            )
            source_sha = require_sha256(
                audio.get("source_sha256"),
                field=f"{sample_id}.audio.source_sha256",
            )
            if verify_audio_files:
                if not audio_path.is_file():
                    raise FileNotFoundError(
                        f"{sample_id} Audio file does not exist: {audio_path}"
                    )
                if file_sha256(audio_path) != derived_sha:
                    raise ValueError(f"{sample_id} Derived audio SHA does not match")
                source_uri = str(audio.get("source_uri") or "")
                if source_uri.startswith(("tar://", "zip://")):
                    actual_source_sha = hashlib.sha256(
                        read_archive_bytes(source_uri)
                    ).hexdigest()
                elif "://" in source_uri:
                    raise ValueError(
                        f"{sample_id}.audio.source_uri uses URI:"
                        f"{source_uri!r}"
                    )
                else:
                    source_path = _resolve_local_path(
                        source_uri,
                        base_dir=manifest_path.parent,
                        field=f"{sample_id}.audio.source_uri",
                    ).resolve()
                    if not source_path.is_file():
                        raise FileNotFoundError(
                            f"{sample_id} source audio does not exist: {source_path}"
                        )
                    actual_source_sha = file_sha256(source_path)
                if actual_source_sha != source_sha:
                    raise ValueError(f"{sample_id} Source audio SHA does not match")
                with sf.SoundFile(str(audio_path)) as handle:
                    sample_rate = int(handle.samplerate)
                    channels = int(handle.channels)
                    frames = int(len(handle))
            else:
                sample_rate = int(
                    audio.get("sample_rate", audio.get("sample_rate_hz", -1))
                )
                channels = int(audio.get("channels", -1))
                declared_duration = float(audio.get("duration_sec", 0.0))
                frames = int(
                    audio.get(
                        "samples",
                        round(declared_duration * SAMPLE_RATE),
                    )
                )
            if sample_rate != SAMPLE_RATE or channels != AUDIO_CHANNELS:
                raise ValueError(
                    f"{sample_id} must be {SAMPLE_RATE} Hz stereo,"
                    f"received {sample_rate} Hz/{channels}ch"
                )
            if frames <= 0:
                raise ValueError(f"{sample_id} Audio file is empty")
            duration = float(audio["duration_sec"])
            _validate_exact_audio_length(
                sample_id=sample_id,
                declared_duration=duration,
                actual_frames=frames,
            )
            start = float(audio["start_sec"])
            if not math.isfinite(start) or start < 0.0:
                raise ValueError(f"{sample_id} audio.start_sec must be finite and non-negative")
            latent_frames = samples_to_latent_frames(frames)
            if latent_frames <= 0:
                raise ValueError(f"{sample_id} is not valid latent frame")
            if latent_frames > 360 * LATENT_FRAME_HZ:
                raise ValueError(
                    f"{sample_id} exceeds the 360-second Renderer limit; implicit clipping is disabled"
                )
            records.append(
                RenderAudioInput(
                    index=len(records),
                    sample_id=sample_id,
                    split=split,
                    audio_path=audio_path,
                    derived_audio_sha256=derived_sha,
                    source_audio_sha256=source_sha,
                    source_start_sec=start,
                    source_duration_sec=duration,
                    num_samples=frames,
                    latent_frames=latent_frames,
                    raw_record=raw,
                )
            )
    if not records:
        raise ValueError(f"Render manifest is empty: {manifest_path}")
    return RenderAudioManifest(
        path=manifest_path,
        sha256=manifest_sha,
        records=tuple(records),
    )


def validate_stft_mapping(
    value: Mapping[str, Any],
    *,
    expected_revision: str,
) -> str:

    mapping = normalize_stft_checkpoint_config(value)
    if str(expected_revision) != STFT_CONTRACT_VERSION:
        raise ValueError(
            f"STFT revision={expected_revision!r},Current requirement {STFT_CONTRACT_VERSION!r}"
        )
    required = {
        "revision": STFT_CONTRACT_VERSION,
        "n_fft": 960,
        "win_length": 960,
        "hop_length": 480,
        "window": "hann_periodic",
        "center": False,
        "explicit_left_padding": 240,
        "explicit_right_padding": True,
        "bins": STFT_BINS,
        "analysis_dtype": "float32",
    }
    mismatches = {
        key: {"expected": expected, "actual": mapping.get(key)}
        for key, expected in required.items()
        if mapping.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"STFT config is incompatible with the current revision: {mismatches}")
    if bool(mapping.get("normalized", False)):
        raise ValueError("the current STFT revision requires normalized=false")
    if float(mapping.get("boundary_window_floor", 0.0)) != 0.0:
        raise ValueError("the current STFT revision requires boundary_window_floor=0")
    drop_nyquist = mapping.get("drop_nyquist")
    keep_dc = mapping.get("keep_dc")
    if not isinstance(drop_nyquist, bool) or keep_dc is not drop_nyquist:
        raise ValueError(
            "The 480-bin layout must use keep_dc=true with drop_nyquist=true, "
            "or keep_dc=false with drop_nyquist=false"
        )
    unknown = (
        set(mapping)
        - set(required)
        - {
            "normalized",
            "boundary_window_floor",
            "keep_dc",
            "drop_nyquist",
            "profile",
        }
    )
    if unknown:
        raise ValueError(f"STFT config contains unsupported fields: {sorted(unknown)}")
    profile = mapping.get("profile")
    if profile is not None and profile != "open_qwen_music_v1":
        raise ValueError(
            "STFT profile must be open_qwen_music_v1 when specified"
        )
    STFTConfig.from_mapping(mapping)
    return json_sha256(mapping)


def normalize_stft_checkpoint_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the current public STFT checkpoint schema at the load boundary."""
    normalized = dict(value)
    if normalized.get("revision") != STFT_CONTRACT_VERSION:
        raise ValueError("STFT checkpoint has an unsupported revision")
    return normalized


@dataclass(frozen=True)
class FrozenSpecVAE:
    model: nn.Module
    checkpoint_sha256: str
    vae_revision: str
    stft_revision: str
    stft_config: dict[str, Any]
    stft_config_sha256: str
    checkpoint_config: dict[str, Any]
    posterior_epsilon_draw_layout: str = POSTERIOR_EPSILON_DRAW_LAYOUT

    def __post_init__(self) -> None:
        if self.posterior_epsilon_draw_layout not in POSTERIOR_EPSILON_DRAW_LAYOUTS:
            raise ValueError("Spec-VAE posterior epsilon draw layout is invalid")


def resolve_checkpoint_posterior_epsilon_draw_layout(
    checkpoint_config: Mapping[str, Any],
) -> str:

    train = checkpoint_config.get("train")
    if train is not None and not isinstance(train, Mapping):
        raise ValueError("Spec-VAE checkpoint train config must be a mapping")
    train = train if isinstance(train, Mapping) else {}
    raw_layout = train.get("posterior_epsilon_draw_layout")
    layout = POSTERIOR_EPSILON_DRAW_LAYOUT if raw_layout is None else str(raw_layout)
    if layout != POSTERIOR_EPSILON_DRAW_LAYOUT:
        raise ValueError("checkpoint posterior epsilon draw layout must be contiguous_bdt")
    return layout


def posterior_epsilon_draw_identity(
    checkpoint_config: Mapping[str, Any],
) -> dict[str, str]:

    layout = resolve_checkpoint_posterior_epsilon_draw_layout(checkpoint_config)
    return {
        "posterior_epsilon_draw_layout": layout,
        "posterior_epsilon_draw_order": "contiguous_[B,D,T]_then_transpose_[B,T,D]",
    }


def _assert_checkpoint_contract(config: Mapping[str, Any]) -> None:
    contract = config.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Spec-VAE checkpoint is missing contract")
    expected = {
        "sample_rate": SAMPLE_RATE,
        "channels": AUDIO_CHANNELS,
        "latent_frame_hz": LATENT_FRAME_HZ,
        "latent_dim": LATENT_DIM,
    }
    mismatches = {
        key: {"expected": wanted, "actual": contract.get(key)}
        for key, wanted in expected.items()
        if contract.get(key) != wanted
    }
    if mismatches:
        raise ValueError(f"Spec-VAE checkpoint shape contract does not match: {mismatches}")


def load_frozen_spec_vae(
    checkpoint: str | Path,
    *,
    expected_checkpoint_sha256: str,
    expected_vae_revision: str,
    expected_stft_config: Mapping[str, Any],
    expected_stft_config_sha256: str,
    expected_stft_revision: str,
    device: torch.device | str = "cpu",
    model_factory: Callable[[SpecVAEConfig], nn.Module] = SpecVAE,
) -> FrozenSpecVAE:

    checkpoint_path = Path(checkpoint).resolve()
    expected_checkpoint_sha = require_sha256(
        expected_checkpoint_sha256, field="vae_checkpoint_sha256"
    )
    actual_checkpoint_sha = file_sha256(checkpoint_path)
    if actual_checkpoint_sha != expected_checkpoint_sha:
        raise ValueError(
            "Spec-VAE checkpoint SHA does not match:"
            f"expected={expected_checkpoint_sha}, actual={actual_checkpoint_sha}"
        )
    expected_stft_sha = require_sha256(
        expected_stft_config_sha256, field="stft_config_sha256"
    )
    current_stft_sha = validate_stft_mapping(
        expected_stft_config,
        expected_revision=expected_stft_revision,
    )
    if current_stft_sha != expected_stft_sha:
        raise ValueError(
            "Runtime STFT config SHA does not match:"
            f"declared={expected_stft_sha}, actual={current_stft_sha}"
        )


    state = read_render_checkpoint_state(
        checkpoint_path,
        strict_sidecar=True,
        map_location="cpu",
        expected_checkpoint_sha256=expected_checkpoint_sha,
    )
    if state.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Spec-VAE checkpoint format_version is incompatible with")
    if state.get("component") != "spec_vae":
        raise ValueError(
            f"Checkpoint component must be spec_vae; received {state.get('component')!r}"
        )
    checkpoint_config = state.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("Spec-VAE checkpoint is missing complete config")
    checkpoint_config = dict(checkpoint_config)
    if checkpoint_config.get("format_version") != "oqm.render.config.v1":
        raise ValueError("Spec-VAE checkpoint config format_version is incompatible with")
    _assert_checkpoint_contract(checkpoint_config)
    checkpoint_stft = checkpoint_config.get("stft")
    if not isinstance(checkpoint_stft, Mapping):
        raise ValueError("Spec-VAE checkpoint is missing STFT config")
    raw_checkpoint_stft = dict(checkpoint_stft)
    checkpoint_stft = normalize_stft_checkpoint_config(raw_checkpoint_stft)
    expected_stft_config = normalize_stft_checkpoint_config(expected_stft_config)
    checkpoint_stft_sha = validate_stft_mapping(
        checkpoint_stft,
        expected_revision=expected_stft_revision,
    )
    if checkpoint_stft != expected_stft_config:
        raise ValueError("Checkpoint STFT config differs from the cache config")
    if checkpoint_stft_sha != expected_stft_sha:
        raise ValueError("Checkpoint STFT config SHA does not match the cache config")
    upstream_stft_sha = (state.get("upstream_revisions") or {}).get(
        "stft_config_sha256"
    )
    if upstream_stft_sha not in {
        expected_stft_sha,
        json_sha256(raw_checkpoint_stft),
    }:
        raise ValueError("Checkpoint upstream stft_config_sha256 does not match the actual config")

    model_section = checkpoint_config.get("model")
    if not isinstance(model_section, Mapping):
        raise ValueError("Spec-VAE checkpoint is missing model config")
    model_values = model_section.get("spec_vae")
    if not isinstance(model_values, Mapping):
        raise ValueError("Spec-VAE checkpoint is missing model.spec_vae")
    model_values = normalize_spec_vae_checkpoint_config(dict(model_values))
    posterior_layout = resolve_checkpoint_posterior_epsilon_draw_layout(
        checkpoint_config
    )
    if model_values.get("revision") != expected_vae_revision:
        raise ValueError(
            "Spec-VAE revision does not match:"
            f"checkpoint={model_values.get('revision')!r}, "
            f"expected={expected_vae_revision!r}"
        )
    vae_config = SpecVAEConfig.from_dict(model_values)
    if vae_config.revision != expected_vae_revision:
        raise ValueError("Spec-VAE config revision does not match the declared revision")
    model_states = state.get("models")
    if not isinstance(model_states, Mapping) or "vae" not in model_states:
        raise ValueError("Spec-VAE checkpoint models must contain a vae entry")
    model = model_factory(vae_config)
    model.load_state_dict(model_states["vae"], strict=True)
    non_finite = nonfinite_spec_vae_parameter_names(model)
    if non_finite:
        raise ValueError(f"Spec-VAE checkpoint parameter contains NaN/Inf: {non_finite[:8]}")
    model.requires_grad_(False)
    model.to(torch.device(device)).eval()
    return FrozenSpecVAE(
        model=model,
        checkpoint_sha256=actual_checkpoint_sha,
        vae_revision=expected_vae_revision,
        stft_revision=expected_stft_revision,
        stft_config=checkpoint_stft,
        stft_config_sha256=checkpoint_stft_sha,
        checkpoint_config=checkpoint_config,
        posterior_epsilon_draw_layout=posterior_layout,
    )


def load_audio_tensor(
    record: RenderAudioInput,
    *,
    device: torch.device | str,
) -> Tensor:
    if not record.audio_path.is_file():
        raise FileNotFoundError(
            f"{record.sample_id} Audio file does not exist: {record.audio_path}"
        )
    actual_derived_sha = file_sha256(record.audio_path)
    if actual_derived_sha != record.derived_audio_sha256:
        raise ValueError(f"{record.sample_id} Derived audio SHA does not match")
    audio = record.raw_record.get("audio")
    if not isinstance(audio, Mapping):
        raise ValueError(f"{record.sample_id} Audio metadata is invalid")
    source_uri = str(audio.get("source_uri") or "")
    if source_uri.startswith(("tar://", "zip://")):
        actual_source_sha = hashlib.sha256(read_archive_bytes(source_uri)).hexdigest()
    elif "://" in source_uri:
        source_path = _resolve_local_path(
            source_uri,
            base_dir=record.audio_path.parent,
            field=f"{record.sample_id}.audio.source_uri",
        ).resolve()
        if source_path == record.audio_path:
            actual_source_sha = actual_derived_sha
        elif not source_path.is_file():
            raise FileNotFoundError(
                f"{record.sample_id} source audio does not exist: {source_path}"
            )
        else:
            actual_source_sha = file_sha256(source_path)
    else:
        source_path = _resolve_local_path(
            source_uri,
            base_dir=record.audio_path.parent,
            field=f"{record.sample_id}.audio.source_uri",
        ).resolve()
        if source_path == record.audio_path:
            actual_source_sha = actual_derived_sha
        elif not source_path.is_file():
            raise FileNotFoundError(
                f"{record.sample_id} source audio does not exist: {source_path}"
            )
        else:
            actual_source_sha = file_sha256(source_path)
    if actual_source_sha != record.source_audio_sha256:
        raise ValueError(f"{record.sample_id} Source audio SHA does not match")
    values, sample_rate = sf.read(
        record.audio_path,
        dtype="float32",
        always_2d=True,
    )
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"{record.sample_id} Decoded sample rate does not match")
    if values.shape != (record.num_samples, AUDIO_CHANNELS):
        raise ValueError(
            f"{record.sample_id} Decoded shape={values.shape},"
            f"requires ({record.num_samples},{AUDIO_CHANNELS})"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{record.sample_id} audio contains NaN/Inf")
    return torch.from_numpy(np.ascontiguousarray(values.T)).unsqueeze(0).to(device)


def posterior_seed(
    *,
    base_seed: int,
    record: RenderAudioInput,
    occurrence: int = 0,
) -> int:
    result = expected_posterior_sample_seed(
        posterior_mode="sample",
        posterior_base_seed=base_seed,
        sample_id=record.sample_id,
        derived_audio_sha256=record.derived_audio_sha256,
        occurrence=occurrence,
    )
    assert result is not None
    return result


def expected_posterior_sample_seed(
    *,
    posterior_mode: str,
    posterior_base_seed: int | None,
    sample_id: str,
    derived_audio_sha256: str,
    occurrence: int = 0,
) -> int | None:

    if posterior_mode == "mean":
        if posterior_base_seed is not None:
            raise ValueError("Mean posterior must not declare posterior_base_seed")
        return None
    if posterior_mode != "sample":
        raise ValueError(f"posterior_mode must be mean or sample; received {posterior_mode!r}")
    if not isinstance(posterior_base_seed, int) or isinstance(
        posterior_base_seed, bool
    ):
        raise TypeError("Sample posterior must declare an integer posterior_base_seed")
    derived_sha = require_sha256(
        derived_audio_sha256,
        field="derived_audio_sha256",
    )
    payload = (
        f"{posterior_base_seed}:{int(occurrence)}:{sample_id}:{derived_sha}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def _posterior_generator(
    *,
    mode: str,
    device: torch.device,
    base_seed: int,
    record: RenderAudioInput,
    occurrence: int,
) -> tuple[torch.Generator | None, int | None]:
    if mode == "mean":
        return None, None
    if mode != "sample":
        raise ValueError(f"posterior_mode must be mean or sample; received {mode!r}")
    seed = expected_posterior_sample_seed(
        posterior_mode=mode,
        posterior_base_seed=base_seed,
        sample_id=record.sample_id,
        derived_audio_sha256=record.derived_audio_sha256,
        occurrence=occurrence,
    )
    assert seed is not None
    generator = torch.Generator(device=device).manual_seed(seed)
    return generator, seed


def sample_posterior_with_layout(
    posterior: Any,
    *,
    generator: torch.Generator,
    posterior_epsilon_draw_layout: str,
) -> Tensor:

    if posterior_epsilon_draw_layout == POSTERIOR_EPSILON_DRAW_LAYOUT:
        if not isinstance(posterior, SoftplusDiagonalGaussianPosterior):
            raise ValueError("contiguous_bdt requires a softplus source posterior")
        return posterior.sample(
            generator=generator,
            contiguous_source_layout=True,
        )
    raise ValueError("posterior_epsilon_draw_layout must be contiguous_bdt")


@torch.inference_mode()
def encode_record_latents(
    *,
    model: nn.Module,
    stft: StereoSTFT,
    record: RenderAudioInput,
    device: torch.device | str,
    posterior_mode: str,
    sample_seed: int,
    occurrence: int = 0,
    posterior_epsilon_draw_layout: str = POSTERIOR_EPSILON_DRAW_LAYOUT,
) -> tuple[Tensor, int | None]:

    if posterior_mode not in POSTERIOR_MODES:
        raise ValueError("posterior_mode must be explicitly mean or sample")
    resolved_device = torch.device(device)
    waveform = load_audio_tensor(record, device=resolved_device)
    lengths = torch.tensor(
        [record.num_samples], dtype=torch.long, device=resolved_device
    )
    analyzed = stft.analyze(waveform, lengths)
    expected_spectrum_frames = record.latent_frames * SPEC_FRAMES_PER_LATENT_FRAME
    if analyzed.spectrum.shape != (
        1,
        AUDIO_CHANNELS,
        STFT_BINS,
        expected_spectrum_frames,
    ):
        raise ValueError(
            f"{record.sample_id} STFT shape={tuple(analyzed.spectrum.shape)},"
            f"requires (1,2,{STFT_BINS},{expected_spectrum_frames})"
        )
    if int(analyzed.spectrum_lengths[0]) != expected_spectrum_frames:
        raise ValueError(f"{record.sample_id} STFT length does not satisfy the contract")
    if not torch.isfinite(analyzed.spectrum).all():
        raise ValueError(f"{record.sample_id} STFT contains NaN/Inf")
    encoded = model.encode(analyzed.spectrum, analyzed.spectrum_lengths)
    posterior = getattr(encoded, "posterior", None)
    if posterior is None:
        raise TypeError("Spec-VAE encode output is missing posterior")
    generator, used_seed = _posterior_generator(
        mode=posterior_mode,
        device=resolved_device,
        base_seed=sample_seed,
        record=record,
        occurrence=occurrence,
    )
    if posterior_mode == "mean":
        mean = getattr(posterior, "mean", None)
        if not isinstance(mean, Tensor):
            raise TypeError("Spec-VAE posterior is missing mean tensor")
        latents = mean
    else:
        assert generator is not None
        latents = sample_posterior_with_layout(
            posterior,
            generator=generator,
            posterior_epsilon_draw_layout=posterior_epsilon_draw_layout,
        )
    latent_lengths = getattr(encoded, "latent_lengths", None)
    latent_mask = getattr(encoded, "latent_mask", None)
    if not isinstance(latent_lengths, Tensor) or latent_lengths.shape != (1,):
        raise ValueError("Spec-VAE encode latent_lengths must have shape [1]")
    if int(latent_lengths[0]) != record.latent_frames:
        raise ValueError(
            f"{record.sample_id} latent length={int(latent_lengths[0])},"
            f"requires ceil({record.num_samples}/1920)={record.latent_frames}"
        )
    if (
        not isinstance(latent_mask, Tensor)
        or latent_mask.dtype != torch.bool
        or latent_mask.shape != (1, record.latent_frames)
        or not bool(latent_mask.all())
    ):
        raise ValueError(f"{record.sample_id} Latent mask and shape do not satisfy the full-prefix contract")
    if latents.shape != (1, record.latent_frames, LATENT_DIM):
        raise ValueError(
            f"{record.sample_id} latent shape={tuple(latents.shape)},"
            f"requires (1,{record.latent_frames},{LATENT_DIM})"
        )
    if not latents.is_floating_point() or not torch.isfinite(latents).all():
        raise ValueError(f"{record.sample_id} Latent dtype or values are invalid")
    return latents[0].detach().float().cpu(), used_seed


@dataclass
class ChannelStats:

    sum: np.ndarray
    sumsq: np.ndarray
    count: np.ndarray

    @classmethod
    def empty(cls) -> "ChannelStats":
        return cls(
            sum=np.zeros(LATENT_DIM, dtype=np.float64),
            sumsq=np.zeros(LATENT_DIM, dtype=np.float64),
            count=np.zeros(LATENT_DIM, dtype=np.int64),
        )

    def update(self, latents: Tensor | np.ndarray) -> None:
        values = (
            latents.detach().cpu().numpy()
            if isinstance(latents, Tensor)
            else np.asarray(latents)
        )
        if values.ndim != 2 or values.shape[1] != LATENT_DIM:
            raise ValueError(
                f"stats latent must have shape [T, {LATENT_DIM}],received {values.shape}"
            )
        if values.shape[0] <= 0 or not np.isfinite(values).all():
            raise ValueError("stats latent must be non-empty and all finite")
        work = values.astype(np.float64, copy=False)
        self.sum += work.sum(axis=0, dtype=np.float64)
        self.sumsq += np.square(work).sum(axis=0, dtype=np.float64)
        self.count += values.shape[0]

    def merge(self, other: "ChannelStats") -> None:
        self.sum += other.sum
        self.sumsq += other.sumsq
        self.count += other.count

    def copy(self) -> "ChannelStats":
        return ChannelStats(
            sum=self.sum.copy(),
            sumsq=self.sumsq.copy(),
            count=self.count.copy(),
        )

    def validate(self) -> None:
        expected = (LATENT_DIM,)
        if (
            self.sum.shape != expected
            or self.sumsq.shape != expected
            or self.count.shape != expected
        ):
            raise ValueError("channel statistics must all have shape [128]")
        if (
            not np.isfinite(self.sum).all()
            or not np.isfinite(self.sumsq).all()
            or np.any(self.count < 0)
        ):
            raise ValueError("Channel statistics contain invalid values or counts")

    def moments(
        self,
        *,
        minimum_std: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.validate()
        if not math.isfinite(minimum_std) or minimum_std <= 0.0:
            raise ValueError("minimum_std must be a finite positive number")
        if np.any(self.count <= 0):
            raise ValueError("each latent channel must have statistical samples")
        count = self.count.astype(np.float64)
        mean = self.sum / count
        second = self.sumsq / count
        variance = second - np.square(mean)
        tolerance = 1.0e-12 * np.maximum(
            1.0, np.maximum(np.abs(second), np.square(mean))
        )
        if np.any(variance < -tolerance):
            bad = np.flatnonzero(variance < -tolerance)[:8].tolist()
            raise ValueError(f"Latent variance is invalid for channels={bad}")
        variance = np.maximum(variance, 0.0)
        std = np.sqrt(variance)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("latent mean/std contains NaN/Inf")
        if np.any(std < minimum_std):
            bad = np.flatnonzero(std < minimum_std)[:8].tolist()
            raise ValueError(
                f"latent std is less than minimum_std={minimum_std}; channels={bad}"
            )
        return mean, std


def all_reduce_channel_stats(
    local: ChannelStats,
    *,
    device: torch.device | str = "cpu",
) -> ChannelStats:

    local.validate()
    result = local.copy()
    if not dist.is_initialized():
        return result
    backend = str(dist.get_backend()).lower()
    reduction_device = (
        torch.device(device) if "nccl" in backend else torch.device("cpu")
    )
    sum_tensor = torch.from_numpy(result.sum.copy()).to(reduction_device)
    sumsq_tensor = torch.from_numpy(result.sumsq.copy()).to(reduction_device)
    count_tensor = torch.from_numpy(result.count.copy()).to(reduction_device)
    dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(sumsq_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    result = ChannelStats(
        sum=sum_tensor.cpu().numpy().astype(np.float64, copy=True),
        sumsq=sumsq_tensor.cpu().numpy().astype(np.float64, copy=True),
        count=count_tensor.cpu().numpy().astype(np.int64, copy=True),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class StatsSamplerConfig:

    split: str = "train"
    epochs: int = 1
    seed: int = 0
    shuffle: bool = True
    duration_buckets_seconds: tuple[int, ...] = LATENT_STATS_DURATION_BUCKETS_SECONDS
    max_records: int | None = None
    strategy: str = "global_shuffle"

    def __post_init__(self) -> None:
        if not self.split:
            raise ValueError("stats sampler split must not be empty")
        if int(self.epochs) <= 0:
            raise ValueError("stats sampler epochs must be positive")
        if self.max_records is not None and (
            not isinstance(self.max_records, int)
            or isinstance(self.max_records, bool)
            or self.max_records <= 0
        ):
            raise ValueError("stats sampler max_records must be null or a positive integer")
        buckets = tuple(int(value) for value in self.duration_buckets_seconds)
        if buckets != LATENT_STATS_DURATION_BUCKETS_SECONDS:
            raise ValueError(
                "stats sampler duration buckets must match the Renderer "
                "30/90/180/360-second training contract"
            )
        if self.strategy not in {
            "global_shuffle",
            "proportional_dataset_family_duration",
        }:
            raise ValueError("stats sampler strategy is invalid")
        if self.strategy == "proportional_dataset_family_duration":
            if self.epochs != 1 or not self.shuffle or self.max_records is None:
                raise ValueError(
                    "dataset-family and duration stratification requires epochs=1, shuffle=true, and max_records"
                )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StatsSamplerConfig":
        return cls(
            split=str(value.get("split", "train")),
            epochs=int(value.get("epochs", 1)),
            seed=int(value.get("seed", 0)),
            shuffle=bool(value.get("shuffle", True)),
            duration_buckets_seconds=tuple(
                int(item)
                for item in value.get(
                    "duration_buckets_seconds",
                    LATENT_STATS_DURATION_BUCKETS_SECONDS,
                )
            ),
            max_records=(
                int(value["max_records"])
                if value.get("max_records") is not None
                else None
            ),
            strategy=str(value.get("strategy", "global_shuffle")),
        )

    def _stratum(self, record: RenderAudioInput) -> tuple[str, int]:
        source = record.raw_record.get("source")
        family = source.get("dataset_family") if isinstance(source, Mapping) else None
        if family not in {"source_b", "source_a"}:
            raise ValueError(
                f"{record.sample_id} is missing dataset_family"
            )
        for upper in self.duration_buckets_seconds:
            if record.latent_frames <= int(upper) * LATENT_FRAME_HZ:
                return str(family), int(upper)
        raise ValueError(f"{record.sample_id} exceeds the maximum statistics duration")

    def _proportional_order(
        self,
        records: Sequence[RenderAudioInput],
        eligible: Sequence[int],
    ) -> list[int]:
        assert self.max_records is not None
        target = min(int(self.max_records), len(eligible))
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        permutation = torch.randperm(len(eligible), generator=generator).tolist()
        shuffled = [eligible[index] for index in permutation]
        strata: dict[tuple[str, int], list[int]] = {}
        for index in eligible:
            strata.setdefault(self._stratum(records[index]), []).append(index)
        quotas = {
            key: target * len(indices) // len(eligible)
            for key, indices in strata.items()
        }
        remainder_order = sorted(
            strata,
            key=lambda key: (
                -(target * len(strata[key]) % len(eligible)),
                key,
            ),
        )
        for key in remainder_order[: target - sum(quotas.values())]:
            quotas[key] += 1
        used = {key: 0 for key in strata}
        selected: list[int] = []
        for index in shuffled:
            key = self._stratum(records[index])
            if used[key] < quotas[key]:
                selected.append(index)
                used[key] += 1
        if len(selected) != target:
            raise RuntimeError("Statistics quota allocation did not close exactly")
        return selected

    def global_assignments(
        self,
        records: Sequence[RenderAudioInput],
    ) -> list[tuple[int, int]]:
        eligible = [record.index for record in records if record.split == self.split]
        if not eligible:
            raise ValueError(f"stats split={self.split!r} No sample")
        maximum_frames = self.duration_buckets_seconds[-1] * LATENT_FRAME_HZ
        too_long = [
            records[index].sample_id
            for index in eligible
            if records[index].latent_frames > maximum_frames
        ]
        if too_long:
            raise ValueError(
                f"Statistics sample exceeds the 360-second training bucket; implicit clipping is disabled: {too_long[:8]}"
            )
        assignments: list[tuple[int, int]] = []
        if self.strategy == "proportional_dataset_family_duration":
            return [
                (0, index)
                for index in self._proportional_order(records, eligible)
            ]
        for epoch in range(self.epochs):
            order = list(eligible)
            if self.shuffle:
                generator = torch.Generator(device="cpu").manual_seed(self.seed + epoch)
                permutation = torch.randperm(len(order), generator=generator).tolist()
                order = [order[index] for index in permutation]
            if self.max_records is not None:
                order = order[: self.max_records]
            assignments.extend((epoch, index) for index in order)
        return assignments

    def local_assignments(
        self,
        records: Sequence[RenderAudioInput],
        *,
        rank: int,
        world_size: int,
    ) -> list[tuple[int, int]]:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("stats sampler rank or world_size is invalid")
        global_plan = self.global_assignments(records)
        return global_plan[rank::world_size]

    def provenance(
        self,
        records: Sequence[RenderAudioInput],
    ) -> dict[str, Any]:
        plan = self.global_assignments(records)
        eligible_indices = [
            record.index for record in records if record.split == self.split
        ]
        payload: dict[str, Any] = {
            "format_version": LATENT_STATS_SAMPLER_FORMAT_VERSION,
            "split": self.split,
            "epochs": self.epochs,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "duration_buckets_seconds": list(self.duration_buckets_seconds),
            "max_records": self.max_records,
            "strategy": self.strategy,
            "drop_last": False,
            "world_invariant_rank_sharding": True,
            "sample_exposure": "once_per_eligible_record_per_epoch",
            "frame_weighting": "valid_latent_frames",
            "eligible_records": len(eligible_indices),
            "sampled_records": len(plan),
        }
        if self.strategy == "proportional_dataset_family_duration":
            eligible_strata: dict[str, int] = {}
            selected_strata: dict[str, int] = {}
            for index in eligible_indices:
                key = "/".join(map(str, self._stratum(records[index])))
                eligible_strata[key] = eligible_strata.get(key, 0) + 1
            for _epoch, index in plan:
                key = "/".join(map(str, self._stratum(records[index])))
                selected_strata[key] = selected_strata.get(key, 0) + 1
            payload["eligible_strata"] = dict(sorted(eligible_strata.items()))
            payload["selected_strata"] = dict(sorted(selected_strata.items()))
        return payload


def collect_local_stats(
    *,
    records: Sequence[RenderAudioInput],
    assignments: Iterable[tuple[int, int]],
    model: nn.Module,
    stft: StereoSTFT,
    device: torch.device | str,
    posterior_mode: str,
    sample_seed: int,
    posterior_epsilon_draw_layout: str = POSTERIOR_EPSILON_DRAW_LAYOUT,
) -> ChannelStats:
    accumulator = ChannelStats.empty()
    for epoch, index in assignments:
        latents, _used_seed = encode_record_latents(
            model=model,
            stft=stft,
            record=records[index],
            device=device,
            posterior_mode=posterior_mode,
            sample_seed=sample_seed,
            occurrence=epoch,
            posterior_epsilon_draw_layout=posterior_epsilon_draw_layout,
        )
        accumulator.update(latents)
    return accumulator


def build_latent_stats_payload(
    stats: ChannelStats,
    *,
    minimum_std: float,
    manifest_sha256: str,
    cache_config_sha256: str,
    vae_checkpoint_sha256: str,
    vae_revision: str,
    stft_revision: str,
    stft_config_sha256: str,
    posterior_mode: str,
    sample_seed: int,
    sampler: Mapping[str, Any],
    latent_layout: Mapping[str, Any],
    stage2_vae_binding: Mapping[str, Any] | None = None,
    posterior_epsilon_draw_layout: str = POSTERIOR_EPSILON_DRAW_LAYOUT,
) -> dict[str, Any]:
    if posterior_mode not in POSTERIOR_MODES:
        raise ValueError("stats posterior_mode must be explicitly mean or sample")
    if posterior_epsilon_draw_layout not in POSTERIOR_EPSILON_DRAW_LAYOUTS:
        raise ValueError("stats posterior_epsilon_draw_layout is invalid")
    mean, std = stats.moments(minimum_std=minimum_std)
    normalized_layout = validate_latent_layout(latent_layout)
    payload = {
        "format_version": LATENT_STATS_FORMAT_VERSION,
        "latent_dim": LATENT_DIM,
        "frame_hz": float(LATENT_FRAME_HZ),
        "dtype": "float64",
        "variance_estimator": "population",
        "minimum_std": float(minimum_std),
        "count": [int(value) for value in stats.count],
        "sum": [float(value) for value in stats.sum],
        "sumsq": [float(value) for value in stats.sumsq],
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "manifest_sha256": require_sha256(manifest_sha256, field="manifest_sha256"),
        "cache_config_sha256": require_sha256(
            cache_config_sha256, field="cache_config_sha256"
        ),
        "vae_checkpoint_sha256": require_sha256(
            vae_checkpoint_sha256, field="vae_checkpoint_sha256"
        ),
        "vae_revision": str(vae_revision),
        "stft_revision": str(stft_revision),
        "stft_config_sha256": require_sha256(
            stft_config_sha256, field="stft_config_sha256"
        ),
        "posterior_mode": posterior_mode,
        "posterior_base_seed": int(sample_seed)
        if posterior_mode == "sample"
        else None,
        "posterior_seed_derivation": (
            POSTERIOR_SEED_DERIVATION if posterior_mode == "sample" else None
        ),
        "posterior_epsilon_draw_layout": posterior_epsilon_draw_layout,
        "latent_layout": normalized_layout,
        "latent_layout_sha256": json_sha256(normalized_layout),
        "sampler": dict(sampler),
    }
    if stage2_vae_binding is not None:
        payload["stage2_vae_binding"] = dict(stage2_vae_binding)
    return payload


def latent_stats_sidecar_path(path: str | Path) -> Path:
    source = Path(path)
    return source.with_suffix(source.suffix + ".sha256.json")


def freeze_latent_stats(path: str | Path, payload: Mapping[str, Any]) -> str:

    destination = Path(path)
    validate_latent_stats_payload(payload)
    atomic_write_json(destination, dict(payload))
    digest = file_sha256(destination)
    atomic_write_json(
        latent_stats_sidecar_path(destination),
        {
            "format_version": LATENT_STATS_SIDECAR_FORMAT_VERSION,
            "stats_sha256": digest,
            "stats_size_bytes": destination.stat().st_size,
        },
    )
    return digest


@dataclass(frozen=True)
class FrozenLatentStats:
    path: Path
    sha256: str
    payload: dict[str, Any]
    mean: np.ndarray
    std: np.ndarray


def _stats_array(
    payload: Mapping[str, Any],
    field: str,
    *,
    dtype: np.dtype[Any],
) -> np.ndarray:
    value = np.asarray(payload.get(field), dtype=dtype)
    if value.shape != (LATENT_DIM,):
        raise ValueError(f"latent stats {field} must be [128]")
    return value


def validate_latent_stats_payload(payload: Mapping[str, Any]) -> None:
    format_version = payload.get("format_version")
    if format_version != LATENT_STATS_FORMAT_VERSION:
        raise ValueError("latent stats format_version is incompatible with")
    if payload.get("latent_dim") != LATENT_DIM:
        raise ValueError("latent stats latent_dim must be 128")
    if float(payload.get("frame_hz", 0.0)) != float(LATENT_FRAME_HZ):
        raise ValueError("latent stats frame_hz must be 25")
    if payload.get("dtype") != "float64":
        raise ValueError("latent stats accumulation dtype must be float64")
    if payload.get("variance_estimator") != "population":
        raise ValueError("latent stats variance_estimator must be population")
    posterior_mode = str(payload.get("posterior_mode") or "")
    if posterior_mode not in POSTERIOR_MODES:
        raise ValueError("latent stats posterior_mode must be mean or sample")
    seed_field = "posterior_base_seed"
    posterior_base_seed = payload.get(seed_field)
    if posterior_mode == "sample" and (
        not isinstance(posterior_base_seed, int)
        or isinstance(posterior_base_seed, bool)
    ):
        raise ValueError(f"sample statistics must record an integer {seed_field}")
    if posterior_mode == "mean" and posterior_base_seed is not None:
        raise ValueError(f"mean statistics must not record {seed_field}")
    expected_derivation = (
        POSTERIOR_SEED_DERIVATION if posterior_mode == "sample" else None
    )
    if payload.get("posterior_seed_derivation") != expected_derivation:
        raise ValueError("latent stats posterior seed derivation does not match")
    posterior_layout = payload.get(
        "posterior_epsilon_draw_layout",
        POSTERIOR_EPSILON_DRAW_LAYOUT,
    )
    if posterior_layout not in POSTERIOR_EPSILON_DRAW_LAYOUTS:
        raise ValueError("latent stats posterior epsilon draw layout is invalid")
    if format_version == LATENT_STATS_FORMAT_VERSION:
        layout = validate_latent_layout(payload.get("latent_layout"))
        if payload.get("latent_layout_sha256") != json_sha256(layout):
            raise ValueError("latent stats latent_layout_sha256 does not match")
    stage2_binding = payload.get("stage2_vae_binding")
    if stage2_binding is not None:
        if not isinstance(stage2_binding, Mapping):
            raise ValueError("latent stats stage2_vae_binding must be a mapping")
        expected_fields = {
            "schema_version",
            "binding_path",
            "binding_sha256",
            "binding_revision",
            "posterior_decision_path",
            "posterior_decision_sha256",
            "posterior_mode",
            "posterior_base_seed",
            "posterior_seed_derivation",
            "posterior_epsilon_draw_layout",
        }
        if set(stage2_binding) != expected_fields:
            raise ValueError("latent stats stage2_vae_binding fields are incomplete or contain unknown entries")
        if (
            stage2_binding.get("schema_version")
            != "oqm.render.latent-stage2-vae-binding-ref.v1"
        ):
            raise ValueError("latent stats stage2_vae_binding schema is incompatible with")
        for field in (
            "binding_sha256",
            "binding_revision",
            "posterior_decision_sha256",
        ):
            require_sha256(
                stage2_binding.get(field), field=f"stage2_vae_binding.{field}"
            )
        if stage2_binding.get("posterior_mode") != posterior_mode:
            raise ValueError("latent stats binding posterior mode does not match")
        if stage2_binding.get("posterior_base_seed") != posterior_base_seed:
            raise ValueError("latent stats binding posterior base seed does not match")
        if (
            stage2_binding.get("posterior_seed_derivation")
            != payload.get("posterior_seed_derivation")
        ):
            raise ValueError("latent stats binding posterior seed derivation does not match")
        if stage2_binding.get("posterior_epsilon_draw_layout") != posterior_layout:
            raise ValueError("latent stats binding posterior epsilon layout does not match")
        for field in ("binding_path", "posterior_decision_path"):
            value = stage2_binding.get(field)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError(f"stage2_vae_binding.{field} must be an absolute path")
    for field in (
        "manifest_sha256",
        "cache_config_sha256",
        "vae_checkpoint_sha256",
        "stft_config_sha256",
    ):
        require_sha256(payload.get(field), field=field)
    if payload.get("stft_revision") != STFT_CONTRACT_VERSION:
        raise ValueError("latent stats STFT revision is incompatible with")
    if not str(payload.get("vae_revision") or ""):
        raise ValueError("latent stats is missing VAE revision")
    sampler = payload.get("sampler")
    if (
        not isinstance(sampler, Mapping)
        or sampler.get("format_version") != LATENT_STATS_SAMPLER_FORMAT_VERSION
    ):
        raise ValueError("latent stats sampler provenance is incompatible with")
    raw_count = payload.get("count")
    if (
        not isinstance(raw_count, list)
        or len(raw_count) != LATENT_DIM
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in raw_count
        )
    ):
        raise ValueError("latent stats count must be 128 positive integers")
    count = _stats_array(payload, "count", dtype=np.int64)
    sums = _stats_array(payload, "sum", dtype=np.float64)
    sumsq = _stats_array(payload, "sumsq", dtype=np.float64)
    declared_mean = _stats_array(payload, "mean", dtype=np.float64)
    declared_std = _stats_array(payload, "std", dtype=np.float64)
    minimum_std = float(payload.get("minimum_std", 0.0))
    reconstructed = ChannelStats(
        sum=sums,
        sumsq=sumsq,
        count=count,
    )
    mean, std = reconstructed.moments(minimum_std=minimum_std)
    if not np.allclose(declared_mean, mean, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("latent stats mean does not match sum/count")
    if not np.allclose(declared_std, std, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("latent stats standard deviation does not match sum/sumsq/count")


def load_frozen_latent_stats(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_vae_checkpoint_sha256: str,
    expected_vae_revision: str,
    expected_stft_revision: str,
    expected_stft_config_sha256: str,
    expected_posterior_mode: str,
    expected_sample_seed: int | None = None,
    expected_posterior_base_seed: int | None = None,
    expected_manifest_sha256: str | None = None,
    expected_cache_config_sha256: str | None = None,
    expected_posterior_epsilon_draw_layout: str | None = None,
    expected_stage2_vae_binding_sha256: str | None = None,
    expected_posterior_decision_sha256: str | None = None,
) -> FrozenLatentStats:
    source = Path(path).resolve()
    expected_sha = require_sha256(expected_sha256, field="latent_stats_sha256")
    try:
        payload_bytes = source.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"Latent statistics do not exist or could not be read: {source}") from exc
    actual_sha = hashlib.sha256(payload_bytes).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(
            f"latent stats SHA does not match: expected={expected_sha}, actual={actual_sha}"
        )
    sidecar_path = latent_stats_sidecar_path(source)
    if not sidecar_path.is_file():
        raise ValueError(f"Latent statistics are missing a sidecar: {sidecar_path}")
    try:
        sidecar_bytes = sidecar_path.read_bytes()
        sidecar = json.loads(sidecar_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("latent stats sidecar could not be parsed") from exc
    if not isinstance(sidecar, Mapping):
        raise ValueError("latent stats sidecar must be an object")
    if sidecar.get("format_version") != LATENT_STATS_SIDECAR_FORMAT_VERSION:
        raise ValueError("latent stats sidecar format_version is incompatible with")
    if (
        sidecar.get("stats_sha256") != actual_sha
        or int(sidecar.get("stats_size_bytes", -1)) != len(payload_bytes)
    ):
        raise ValueError("latent stats sidecar SHA or size does not match")
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("latent stats JSON could not be parsed") from exc
    if not isinstance(payload, dict):
        raise ValueError("latent stats JSON must be an object")
    validate_latent_stats_payload(payload)
    expected_values = {
        "vae_checkpoint_sha256": require_sha256(
            expected_vae_checkpoint_sha256,
            field="vae_checkpoint_sha256",
        ),
        "vae_revision": str(expected_vae_revision),
        "stft_revision": str(expected_stft_revision),
        "stft_config_sha256": require_sha256(
            expected_stft_config_sha256,
            field="stft_config_sha256",
        ),
        "posterior_mode": str(expected_posterior_mode),
    }
    if expected_manifest_sha256 is not None:
        expected_values["manifest_sha256"] = require_sha256(
            expected_manifest_sha256,
            field="manifest_sha256",
        )
    if expected_cache_config_sha256 is not None:
        expected_values["cache_config_sha256"] = require_sha256(
            expected_cache_config_sha256,
            field="cache_config_sha256",
        )
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected_values.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"latent stats revision, hash, or mode does not match: {mismatches}")
    seed_field = "posterior_base_seed"
    if (
        expected_sample_seed is not None
        and expected_posterior_base_seed is not None
        and expected_sample_seed != expected_posterior_base_seed
    ):
        raise ValueError(
            "expected_sample_seed conflicts with expected_posterior_base_seed"
        )
    if expected_posterior_mode == "sample":
        configured_base_seed = (
            expected_posterior_base_seed
            if expected_posterior_base_seed is not None
            else expected_sample_seed
        )
        if not isinstance(configured_base_seed, int) or isinstance(
            configured_base_seed, bool
        ):
            raise ValueError("sample statistics must provide expected_posterior_base_seed")
        expected_base_seed = int(configured_base_seed)
    else:


        if expected_posterior_base_seed is not None:
            raise ValueError("mean statistics must not provide expected_posterior_base_seed")
        expected_base_seed = None
    if payload.get(seed_field) != expected_base_seed:
        raise ValueError("latent stats posterior base seed does not match")
    actual_layout = str(
        payload.get(
            "posterior_epsilon_draw_layout",
            POSTERIOR_EPSILON_DRAW_LAYOUT,
        )
    )
    if expected_posterior_epsilon_draw_layout is not None:
        if expected_posterior_epsilon_draw_layout not in POSTERIOR_EPSILON_DRAW_LAYOUTS:
            raise ValueError("expected posterior epsilon draw layout is invalid")
        if actual_layout != expected_posterior_epsilon_draw_layout:
            raise ValueError("latent stats posterior epsilon draw layout does not match")
    stage2_binding = payload.get("stage2_vae_binding")
    if expected_stage2_vae_binding_sha256 is not None:
        if not isinstance(stage2_binding, Mapping):
            raise ValueError("latent stats are missing the Stage 2 VAE binding")
        if stage2_binding.get("binding_sha256") != require_sha256(
            expected_stage2_vae_binding_sha256,
            field="stage2_vae_binding_sha256",
        ):
            raise ValueError("latent stats Stage 2 VAE binding SHA does not match")
    if expected_posterior_decision_sha256 is not None:
        if not isinstance(stage2_binding, Mapping):
            raise ValueError("latent stats are missing the posterior decision")
        if stage2_binding.get("posterior_decision_sha256") != require_sha256(
            expected_posterior_decision_sha256,
            field="posterior_decision_sha256",
        ):
            raise ValueError("latent stats posterior decision SHA does not match")
    return FrozenLatentStats(
        path=source,
        sha256=actual_sha,
        payload=payload,
        mean=_stats_array(payload, "mean", dtype=np.float64),
        std=_stats_array(payload, "std", dtype=np.float64),
    )


def standardize_latents(
    latents: Tensor | np.ndarray,
    *,
    stats: FrozenLatentStats,
    dtype: str,
) -> np.ndarray:
    if dtype not in CACHE_DTYPES:
        raise ValueError("cache dtype must be float16 or float32")
    values = (
        latents.detach().cpu().numpy()
        if isinstance(latents, Tensor)
        else np.asarray(latents)
    )
    if values.ndim != 2 or values.shape[1] != LATENT_DIM:
        raise ValueError("Latent to standardize must have shape [T, 128]")
    if not np.isfinite(values).all():
        raise ValueError("Latent to standardize contains NaN/Inf")


    work = values.astype(np.float32, copy=False)
    mean = stats.mean.astype(np.float32, copy=False)
    std = stats.std.astype(np.float32, copy=False)
    normalized = (work - mean[None, :]) / std[None, :]
    if not np.isfinite(normalized).all():
        raise ValueError("Standardized latent contains NaN/Inf")
    target_dtype = np.float16 if dtype == "float16" else np.float32
    result = normalized.astype(target_dtype, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"latent to {dtype} appear later NaN/Inf")
    return result


def destandardize_latents(
    latents: Tensor | np.ndarray,
    *,
    stats: FrozenLatentStats,
) -> Tensor | np.ndarray:

    if isinstance(latents, Tensor):
        if (
            latents.ndim not in {2, 3}
            or latents.shape[-1] != LATENT_DIM
            or not latents.is_floating_point()
            or not torch.isfinite(latents).all()
        ):
            raise ValueError("Latent to denormalize must be a finite floating-point tensor with shape [T, 128] or [B, T, 128]")
        mean = torch.as_tensor(
            stats.mean,
            device=latents.device,
            dtype=torch.float32,
        )
        std = torch.as_tensor(
            stats.std,
            device=latents.device,
            dtype=torch.float32,
        )
        restored_tensor = latents.float() * std + mean
        if not torch.isfinite(restored_tensor).all():
            raise ValueError("Denormalized latent contains NaN/Inf")
        return restored_tensor

    values = np.asarray(latents)
    if values.ndim not in {2, 3} or values.shape[-1] != LATENT_DIM:
        raise ValueError(
            f"Latent to denormalize must have shape [T, 128] or [B, T, 128]; received {values.shape}"
        )
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("Latent to denormalize must contain finite floating-point values")
    work = values.astype(np.float32, copy=False)
    mean = stats.mean.astype(np.float32, copy=False)
    std = stats.std.astype(np.float32, copy=False)
    restored = work * std + mean
    if not np.isfinite(restored).all():
        raise ValueError("Denormalized latent contains NaN/Inf")
    return restored


def artifact_path_for_record(
    output_dir: str | Path,
    record: RenderAudioInput,
) -> Path:
    safe_id = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in record.sample_id
    )[:96]
    suffix = hashlib.sha256(record.sample_id.encode("utf-8")).hexdigest()[:12]
    return Path(output_dir) / "latents" / suffix[:2] / f"{safe_id}-{suffix}.npy"


def latent_artifact_sidecar_path(path: str | Path) -> Path:
    source = Path(path)
    return source.with_suffix(source.suffix + ".json")


def _latent_artifact_record(
    *,
    record: RenderAudioInput,
    destination: Path,
    digest: str,
    frozen_vae: FrozenSpecVAE,
    stats: FrozenLatentStats,
    posterior_mode: str,
    sample_seed: int,
    dtype: str,
    cache_config_sha256: str,
) -> dict[str, Any]:
    layout = stats.payload.get("latent_layout")
    normalized_layout = (
        validate_latent_layout(layout) if isinstance(layout, Mapping) else None
    )
    used_seed = expected_posterior_sample_seed(
        posterior_mode=posterior_mode,
        posterior_base_seed=sample_seed if posterior_mode == "sample" else None,
        sample_id=record.sample_id,
        derived_audio_sha256=record.derived_audio_sha256,
        occurrence=0,
    )
    row = {
        "artifact_format_version": LATENT_ARTIFACT_FORMAT_VERSION,
        "sample_id": record.sample_id,
        "latent_uri": str(destination.resolve()),
        "latent_sha256": digest,
        "file_sha256": digest,
        "shape": [record.latent_frames, LATENT_DIM],
        "dtype": dtype,
        "num_frames": record.latent_frames,
        "frame_hz": float(LATENT_FRAME_HZ),
        "vae_checkpoint_sha256": frozen_vae.checkpoint_sha256,
        "vae_revision": frozen_vae.vae_revision,
        "latent_stats_sha256": stats.sha256,
        "posterior_mode": posterior_mode,
        "posterior_base_seed": (
            int(sample_seed) if posterior_mode == "sample" else None
        ),
        "posterior_sample_seed": used_seed,
        "posterior_seed_derivation": (
            POSTERIOR_SEED_DERIVATION if posterior_mode == "sample" else None
        ),
        "posterior_epsilon_draw_layout": (
            frozen_vae.posterior_epsilon_draw_layout
        ),
        **(
            {
                "latent_layout": normalized_layout,
                "latent_layout_sha256": stats.payload["latent_layout_sha256"],
            }
            if normalized_layout is not None
            else {}
        ),
        "stft_revision": frozen_vae.stft_revision,
        "stft_config_sha256": frozen_vae.stft_config_sha256,
        "source_audio_sha256": record.source_audio_sha256,
        "derived_audio_sha256": record.derived_audio_sha256,
        "input_audio_sha256": record.derived_audio_sha256,
        "source_start_sec": record.source_start_sec,
        "source_duration_sec": record.source_duration_sec,
        "read_start_sec": 0.0,
        "read_duration_sec": record.num_samples / SAMPLE_RATE,
        "cache_config_sha256": require_sha256(
            cache_config_sha256, field="cache_config_sha256"
        ),
        "cache_revision": require_sha256(
            cache_config_sha256, field="cache_revision"
        ),
    }
    stage2_binding = stats.payload.get("stage2_vae_binding")
    if isinstance(stage2_binding, Mapping):
        row.update(
            {
                "stage2_vae_binding_sha256": stage2_binding["binding_sha256"],
                "stage2_vae_binding_revision": stage2_binding["binding_revision"],
                "posterior_decision_sha256": stage2_binding[
                    "posterior_decision_sha256"
                ],
            }
        )
    return row


def load_cached_latent_record(
    *,
    record: RenderAudioInput,
    frozen_vae: FrozenSpecVAE,
    stats: FrozenLatentStats,
    posterior_mode: str,
    sample_seed: int,
    dtype: str,
    output_path: str | Path,
    cache_config_sha256: str,
) -> dict[str, Any]:
    destination = Path(output_path)
    if not destination.is_file():
        raise FileNotFoundError(f"Latent to restore does not exist: {destination}")
    values = np.load(destination, allow_pickle=False)
    expected_dtype = np.dtype(dtype)
    if (
        values.shape != (record.latent_frames, LATENT_DIM)
        or values.dtype != expected_dtype
        or not np.isfinite(values).all()
    ):
        raise RuntimeError(
            f"{record.sample_id} Latent to restore has an invalid shape, dtype, or value"
        )
    digest = file_sha256(destination)
    expected = _latent_artifact_record(
        record=record,
        destination=destination,
        digest=digest,
        frozen_vae=frozen_vae,
        stats=stats,
        posterior_mode=posterior_mode,
        sample_seed=sample_seed,
        dtype=dtype,
        cache_config_sha256=cache_config_sha256,
    )
    sidecar = latent_artifact_sidecar_path(destination)
    if sidecar.is_file():
        try:
            observed = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"{record.sample_id} latent sidecar could not be parsed"
            ) from exc
        if canonical_json_bytes(observed) != canonical_json_bytes(expected):
            raise RuntimeError(
                f"{record.sample_id} latent sidecar does not match the current artifact identity"
            )
    else:
        raise RuntimeError(
            f"{record.sample_id} Latent already exists without an identity sidecar; implicit reuse is not allowed"
        )
    return expected


def cache_latent_record(
    *,
    record: RenderAudioInput,
    model: nn.Module,
    stft: StereoSTFT,
    device: torch.device | str,
    frozen_vae: FrozenSpecVAE,
    stats: FrozenLatentStats,
    posterior_mode: str,
    sample_seed: int,
    dtype: str,
    output_path: str | Path,
    cache_config_sha256: str,
) -> dict[str, Any]:
    stats_layout = str(
        stats.payload.get(
            "posterior_epsilon_draw_layout",
            POSTERIOR_EPSILON_DRAW_LAYOUT,
        )
    )
    if stats_layout != frozen_vae.posterior_epsilon_draw_layout:
        raise ValueError("latent stats and checkpoint posterior epsilon draw layouts do not match")
    destination = Path(output_path)
    if destination.exists() or latent_artifact_sidecar_path(destination).exists():
        raise FileExistsError(
            f"{record.sample_id} Latent or sidecar already exists; explicit resume verification is required"
        )
    raw_latents, _used_seed = encode_record_latents(
        model=model,
        stft=stft,
        record=record,
        device=device,
        posterior_mode=posterior_mode,
        sample_seed=sample_seed,
        occurrence=0,
        posterior_epsilon_draw_layout=(
            frozen_vae.posterior_epsilon_draw_layout
        ),
    )
    values = standardize_latents(raw_latents, stats=stats, dtype=dtype)
    if values.shape != (record.latent_frames, LATENT_DIM):
        raise ValueError(f"{record.sample_id} Standardized latent shape changed unexpectedly")
    digest = atomic_save_npy(destination, values)
    row = _latent_artifact_record(
        record=record,
        destination=destination,
        digest=digest,
        frozen_vae=frozen_vae,
        stats=stats,
        posterior_mode=posterior_mode,
        sample_seed=sample_seed,
        dtype=dtype,
        cache_config_sha256=cache_config_sha256,
    )
    persisted = np.load(destination, allow_pickle=False)
    if (
        list(persisted.shape) != row["shape"]
        or str(persisted.dtype) != row["dtype"]
        or file_sha256(destination) != row["latent_sha256"]
        or not np.isfinite(persisted).all()
    ):
        raise RuntimeError(f"{record.sample_id} Latent artifact readback verification failed")
    atomic_write_json(latent_artifact_sidecar_path(destination), row)
    return row


def exclusive_record_indices(
    size: int,
    *,
    rank: int,
    world_size: int,
) -> range:
    if size <= 0:
        raise ValueError("cache record size must be positive")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("cache rank or world_size is invalid")
    return range(rank, size, world_size)
