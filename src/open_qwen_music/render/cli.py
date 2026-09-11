
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
import soxr
import torch
from torch import nn

from open_qwen_music.common.checkpoint import config_hash, file_sha256
from open_qwen_music.common.config import load_config

from .cache import load_frozen_latent_stats, validate_stft_mapping
from .checkpoint import read_render_checkpoint_state
from .conditioning import (
    REWRITER_SCHEMA_VERSION,
    RenderConditioner,
)
from .contracts import (
    LATENT_FRAME_HZ,
    SEMANTIC_CODEBOOK_SIZE,
    STFT_CONTRACT_VERSION,
    mask_to_lengths,
)
from .dit_factory import build_render_dit_components_from_config
from .flow import FlowConfig
from .inference import InferenceRevisions, RenderInferencePipeline
from .losses import (
    MelSpectralLoss,
    MultiResolutionSTFTLoss,
    absolute_circular_ipd_error,
    ccpc_loss,
    lr_to_ms,
    multiresolution_ccpc,
    spectral_bandwidth_mask,
    spectral_pan_error,
)
from .refiner import build_refiner_from_mapping, normalize_refiner_checkpoint_config
from .spec_vae import (
    SpecVAE,
    SpecVAEConfig,
    normalize_spec_vae_checkpoint_config,
)
from .stft import STFTConfig, StereoSTFT
from .trainer_dit import (
    DIT_EVALUATION_CONFIG_SECTIONS,
    TrainableConditionerCheckpointView,
    _normalize_dit_checkpoint_config,
    _normalized_checkpoint_config_value,
    train,
    validate_dit_experiment,
    validate_dit_model_and_conditioning,
)

RENDER_EVALUATOR_VERSION = "oqm.render.evaluator.v4"
RECONSTRUCTION_METRIC_PROTOCOL = (
    "oqm.render.reconstruction-mr-v3-bandlimited-stable-ccpc"
)
RENDER_OUTPUT_FORMAT_VERSION = "oqm.render.output.v1"


def _resolve_inference_posterior_identity(
    preset: Mapping[str, Any],
) -> tuple[str, int | None]:

    posterior_mode = preset.get("latent_posterior_mode")
    if posterior_mode not in {"mean", "sample"}:
        raise ValueError("inference.latent_posterior_mode must be 'mean' or 'sample'")
    posterior_base_seed = preset.get("latent_posterior_base_seed")
    if posterior_mode == "sample":
        if (
            not isinstance(posterior_base_seed, int)
            or isinstance(posterior_base_seed, bool)
            or posterior_base_seed < 0
        ):
            raise ValueError(
                "inference sample posterior requires a fixed non-negative "
                "latent_posterior_base_seed"
            )
        return posterior_mode, int(posterior_base_seed)
    if posterior_base_seed is not None:
        raise ValueError("inference mean posterior must not declare a posterior seed")
    return posterior_mode, None


def _strict_payload(
    path: str | Path,
    expected_sha256: Any,
    *,
    name: str,
    expected_component: str,
) -> tuple[dict[str, Any], str]:
    if not isinstance(expected_sha256, str):
        raise RuntimeError(f"{name} SHA-256 must be a string")
    value = expected_sha256.lower()
    if len(value) != 64:
        raise RuntimeError(
            f"{name} must be a fixed 64-character SHA-256 in the inference preset"
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise RuntimeError(f"{name} SHA-256 is not hexadecimal") from exc
    payload = read_render_checkpoint_state(
        path,
        strict_sidecar=True,
        map_location="cpu",
        expected_checkpoint_sha256=value,
        expected_component=expected_component,
        mmap=True,
    )
    if not isinstance(payload, Mapping):
        raise RuntimeError("Render checkpoint payload must be dict")
    return dict(payload), value


def _validate_inference_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config, Mapping):
        raise TypeError("Render inference config must be a mapping")
    for name in ("model", "conditioning", "flow", "revisions", "inference", "artifacts"):
        if not isinstance(config.get(name), Mapping):
            raise TypeError(f"Render inference config is missing mapping section: {name}")
    validate_dit_experiment(config)
    validate_dit_model_and_conditioning(config)
    FlowConfig.from_mapping(config["flow"])
    revisions = InferenceRevisions.from_mapping(config["revisions"])
    preset = config["inference"]
    allowed_preset = {
        "preset_version",
        "solver",
        "num_steps",
        "nfe",
        "precision",
        "cfg_scale",
        "cfg_rescale",
        "use_refiner",
        "require_input_tokenizer_revision",
        "latent_posterior_mode",
        "latent_posterior_base_seed",

        "latent_sample_seed",
        "wav_subtype",
        "flac_subtype",
        "peak_policy",
        "max_output_peak",
        "max_output_true_peak",
        "source_labels",
        "claim",
    }
    unknown = set(preset) - allowed_preset
    if unknown:
        raise ValueError(f"inference contains unknown fields: {sorted(unknown)}")
    if not isinstance(preset.get("preset_version"), str) or not preset[
        "preset_version"
    ].strip():
        raise ValueError("inference.preset_version must be a non-empty string")
    solver = preset.get("solver")
    if solver not in {"euler", "heun"}:
        raise ValueError("inference.solver must be 'euler' or 'heun'")
    for name in ("num_steps", "nfe"):
        value = preset.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"inference.{name} must be an integer")
        if value <= 0:
            raise ValueError(f"inference.{name} must be positive")
    expected_nfe = (
        preset["num_steps"] if solver == "euler" else 2 * preset["num_steps"]
    )
    if preset["nfe"] != expected_nfe:
        raise ValueError(
            "inference.nfe is inconsistent with solver/num_steps: "
            f"expected={expected_nfe} actual={preset['nfe']}"
        )
    for name in ("cfg_rescale", "use_refiner", "require_input_tokenizer_revision"):
        if not isinstance(preset.get(name), bool):
            raise TypeError(f"inference.{name} must be bool")
    if preset["cfg_rescale"] is not False:
        raise NotImplementedError(
            "cfg_rescale is unsupported; the inference implementation requires false"
        )
    if preset.get("precision") not in {"fp32", "bf16"}:
        raise ValueError("inference.precision must be 'fp32' or 'bf16'")
    cfg_scale = preset.get("cfg_scale")
    if (
        isinstance(cfg_scale, bool)
        or not isinstance(cfg_scale, (int, float))
        or not np.isfinite(cfg_scale)
        or cfg_scale < 0
    ):
        raise ValueError("inference.cfg_scale must be a finite non-negative number")
    _resolve_inference_posterior_identity(preset)
    if preset.get("peak_policy") != "reject":
        raise ValueError("Published inference supports only peak_policy='reject'")
    for name in ("max_output_peak", "max_output_true_peak"):
        value = preset.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not 0 < value <= 1
        ):
            raise ValueError(f"inference.{name} must be in (0, 1]")
    for name in ("wav_subtype", "flac_subtype"):
        if preset.get(name) != "PCM_24":
            raise ValueError(f"inference.{name} must be 'PCM_24'")
    artifacts = config["artifacts"]
    expected_artifacts = {
        "dit_sha256",
        "vae_sha256",
        "refiner_sha256",
        "latent_stats_sha256",
    }
    if set(artifacts) != expected_artifacts:
        raise ValueError(
            "inference artifacts are incomplete or contain unknown fields: "
            f"expected={sorted(expected_artifacts)} actual={sorted(artifacts)}"
        )
    normalized_artifacts: dict[str, str] = {}
    for name in sorted(expected_artifacts):
        value = artifacts[name]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"inference artifacts.{name} must be a 64-character SHA-256")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(
                f"inference artifacts.{name} must be hexadecimal SHA-256"
            ) from exc
        normalized_artifacts[name] = value.lower()
    if (
        revisions.latent_stats_sha256.lower()
        != normalized_artifacts["latent_stats_sha256"]
    ):
        raise RuntimeError(
            "Inference revisions and artifacts disagree on the latent-stats SHA-256"
        )


def _validate_local_text_asset(
    text_config: Mapping[str, Any],
    revisions: InferenceRevisions,
) -> None:
    lock_value = text_config.get("asset_lock")
    local_value = text_config.get("local_path")
    if not lock_value or not local_value:
        raise RuntimeError("Published inference requires text_encoder.local_path and asset_lock")
    lock_path = Path(str(lock_value))
    if not lock_path.is_absolute():
        lock_path = Path(__file__).resolve().parents[3] / lock_path
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    component = (payload.get("components") or {}).get("text_encoder")
    if not isinstance(component, Mapping):
        raise RuntimeError("Render asset lock is missing text_encoder")
    model_revision = component.get("model_revision", component.get("revision"))
    tokenizer_revision = component.get(
        "tokenizer_revision", component.get("revision")
    )
    expected = {
        "repository": f"https://huggingface.co/{text_config['model_id']}",
        "local_path": str(local_value),
    }
    mismatches = {
        name: {"expected": value, "actual": component.get(name)}
        for name, value in expected.items()
        if component.get(name) != value
    }
    if mismatches:
        raise RuntimeError(f"Render text-encoder asset lock mismatch: {mismatches}")
    if str(model_revision or "") != revisions.text_encoder_revision:
        raise RuntimeError("Render text-encoder model revision does not match the asset lock")
    if str(tokenizer_revision or "") != revisions.text_tokenizer_revision:
        raise RuntimeError("Render text-tokenizer revision does not match the asset lock")
    files = component.get("files")
    if not isinstance(files, Mapping) or not files:
        raise RuntimeError("Render text-encoder asset lock is missing file SHA-256 values")
    root = Path(str(local_value))
    for relative, digest in files.items():
        path = root / str(relative)
        if not path.is_file() or file_sha256(path) != str(digest):
            raise RuntimeError(f"Local Qwen text-model file SHA-256 mismatch: {relative}")


def _select_model_state(
    payload: Mapping[str, Any],
    *,
    preferred_names: Sequence[str],
    component: str,
) -> Mapping[str, Any]:
    models = payload.get("models")
    if not isinstance(models, Mapping) or not models:
        raise RuntimeError(f"{component} checkpoint is missing models")
    for name in preferred_names:
        if name in models:
            return models[name]
    if len(models) == 1:
        return next(iter(models.values()))
    raise RuntimeError(
        f"{component} checkpoint contains multiple model,Unable to determine target:{sorted(models)}"
    )


def _component_config(
    payload: Mapping[str, Any], names: Sequence[str]
) -> Mapping[str, Any]:
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError("checkpoint is missing config")
    for name in names:
        candidate = config.get(name)
        if isinstance(candidate, Mapping):
            return candidate
    model_section = config.get("model")
    if isinstance(model_section, Mapping):
        for name in names:
            candidate = model_section.get(name)
            if isinstance(candidate, Mapping):
                return candidate
        return model_section
    return config


def _build_dit_and_conditioner(
    config: Mapping[str, Any],
) -> tuple[nn.Module, RenderConditioner]:
    condition_config = config["conditioning"]
    text_config = condition_config["text_encoder"]
    revisions = InferenceRevisions.from_mapping(config["revisions"])
    _validate_local_text_asset(text_config, revisions)
    return build_render_dit_components_from_config(config)


def _assert_dit_checkpoint_config(
    payload: Mapping[str, Any], inference_config: Mapping[str, Any]
) -> None:
    stored = payload.get("config")
    if not isinstance(stored, Mapping):
        raise RuntimeError("DiT checkpoint is missing config")
    stored = _normalize_dit_checkpoint_config(stored)
    inference_config = _normalize_dit_checkpoint_config(inference_config)
    for section in DIT_EVALUATION_CONFIG_SECTIONS:
        if _normalized_checkpoint_config_value(
            stored.get(section)
        ) != _normalized_checkpoint_config_value(
            inference_config.get(section)
        ):
            raise RuntimeError(
                f"DiT checkpoint of {section} and inference preset inconsistent"
            )
    stored_revisions = stored.get("revisions")
    expected_revisions = inference_config.get("revisions")
    if not isinstance(stored_revisions, Mapping) or not isinstance(
        expected_revisions, Mapping
    ):
        raise RuntimeError("DiT checkpoint or inference preset is missing revisions")
    required = (
        "checkpoint_revision",
        "tokenizer_revision",
        "vae_revision",
        "text_encoder_revision",
        "text_tokenizer_revision",
        "text_cache_revision",
        "rewriter_revision",
        "latent_cache_revision",
        "latent_stats_sha256",
    )
    mismatches = {
        name: {
            "checkpoint": stored_revisions.get(name),
            "inference": expected_revisions.get(name),
        }
        for name in required
        if stored_revisions.get(name) != expected_revisions.get(name)
    }
    if mismatches:
        raise RuntimeError(f"DiT checkpoint and inference upstream revisions differ: {mismatches}")


def _assert_dit_upstream_contract(
    payload: Mapping[str, Any],
    inference_config: Mapping[str, Any],
    *,
    vae_checkpoint_sha256: str,
    refiner_checkpoint_sha256: str,
    latent_stats_sha256: str,
) -> None:
    stored = payload.get("upstream_revisions")
    expected = inference_config.get("upstream_revisions")
    if not isinstance(stored, Mapping) or not isinstance(expected, Mapping):
        raise RuntimeError("DiT checkpoint or inference preset is missing upstream_revisions")
    if dict(stored) != dict(expected):
        raise RuntimeError("DiT checkpoint and inference preset use different SHA-256 lists")
    stored_required = payload.get("required_upstream_sha_keys")
    expected_required = inference_config.get("required_upstream_sha_keys")
    normalized_required: dict[str, set[str]] = {}
    for name, value in {
        "checkpoint": stored_required,
        "inference": expected_required,
    }.items():
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or not all(isinstance(item, str) and item for item in value)
            or len(set(value)) != len(value)
        ):
            raise RuntimeError(
                "DiT inference required_upstream_sha_keys is invalid: "
                f"{name}={value!r}"
            )
        normalized_required[name] = set(value)
    if normalized_required["checkpoint"] != normalized_required["inference"]:
        raise RuntimeError(
            "DiT checkpoint and inference preset use different required_upstream_sha_keys"
        )
    vae = stored.get("vae")
    latent_stats = stored.get("latent_stats")
    if (
        not isinstance(vae, Mapping)
        or str(vae.get("sha256") or "") != vae_checkpoint_sha256
    ):
        raise RuntimeError("DiT checkpoint Spec-VAE SHA-256 does not match the file")
    if (
        not isinstance(latent_stats, Mapping)
        or str(latent_stats.get("sha256") or "") != latent_stats_sha256
    ):
        raise RuntimeError("DiT checkpoint latent-stats SHA-256 does not match the file")
    refiner = stored.get("refiner")
    if (
        not isinstance(refiner, Mapping)
        or str(refiner.get("sha256") or "") != refiner_checkpoint_sha256
    ):
        raise RuntimeError("DiT checkpoint refiner SHA-256 does not match the file")
def build_inference_pipeline(
    config: Mapping[str, Any],
    *,
    dit_checkpoint: str | Path,
    vae_checkpoint: str | Path,
    refiner_checkpoint: str | Path,
    latent_stats: str | Path,
    device: torch.device | str,
) -> RenderInferencePipeline:
    _validate_inference_config(config)
    artifacts = config["artifacts"]
    dit_payload, dit_checkpoint_sha256 = _strict_payload(
        dit_checkpoint,
        artifacts.get("dit_sha256", ""),
        name="DiT",
        expected_component="render_dit",
    )
    vae_payload, vae_checkpoint_sha256 = _strict_payload(
        vae_checkpoint,
        artifacts.get("vae_sha256", ""),
        name="Spec-VAE",
        expected_component="spec_vae",
    )
    refiner_payload, refiner_checkpoint_sha256 = _strict_payload(
        refiner_checkpoint,
        artifacts.get("refiner_sha256", ""),
        name="Refiner",
        expected_component="refiner",
    )
    _assert_dit_checkpoint_config(dit_payload, config)

    model, conditioner = _build_dit_and_conditioner(config)
    models = dit_payload.get("models") or {}
    if set(models) != {"dit", "conditioner"}:
        raise RuntimeError(
            "DiT checkpoint model names must be strictly dit/conditioner,"
            f"received {sorted(models)}"
        )
    model.load_state_dict(models["dit"], strict=True)
    TrainableConditionerCheckpointView(conditioner).load_state_dict(
        models["conditioner"], strict=True
    )

    revisions = InferenceRevisions.from_mapping(config["revisions"])
    vae_config = SpecVAEConfig.from_dict(normalize_spec_vae_checkpoint_config(
        dict(_component_config(vae_payload, ("spec_vae", "vae")))
    ))
    if vae_config.revision != revisions.vae_revision:
        raise RuntimeError("Inference preset vae_revision does not match the checkpoint config")
    upstream = vae_payload.get("upstream_revisions")
    if not isinstance(upstream, Mapping):
        raise RuntimeError("Spec-VAE checkpoint is missing upstream_revisions")
    stft_config_sha256 = str(upstream.get("stft_config_sha256") or "")
    checkpoint_stft = (vae_payload.get("config") or {}).get("stft")
    if not isinstance(checkpoint_stft, Mapping):
        raise RuntimeError("Spec-VAE checkpoint is missing the stft mapping")
    current_stft_sha256 = validate_stft_mapping(
        checkpoint_stft,
        expected_revision=STFT_CONTRACT_VERSION,
    )
    if current_stft_sha256 != stft_config_sha256:
        raise RuntimeError("Current StereoSTFT config does not match the Spec-VAE checkpoint")
    inference_config = config["inference"]
    solver = str(inference_config["solver"])
    num_steps = inference_config["num_steps"]
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0:
        raise ValueError("inference.num_steps must be a positive integer")
    expected_nfe = num_steps if solver == "euler" else 2 * num_steps if solver == "heun" else None
    if expected_nfe is None:
        raise ValueError("inference.solver must be 'euler' or 'heun'")
    if int(inference_config.get("nfe", -1)) != expected_nfe:
        raise ValueError(
            "inference.nfe is inconsistent with solver/num_steps: "
            f"expected={expected_nfe} actual={inference_config.get('nfe')}"
        )
    preset_version = str(inference_config.get("preset_version") or "")
    if not preset_version:
        raise ValueError("inference.preset_version cannot be empty")
    precision = str(inference_config.get("precision", "fp32")).lower()
    if precision not in {"fp32", "bf16"}:
        raise ValueError("inference.precision must be 'fp32' or 'bf16'")
    posterior_mode, posterior_base_seed = (
        _resolve_inference_posterior_identity(inference_config)
    )
    frozen_latent_stats = load_frozen_latent_stats(
        latent_stats,
        expected_sha256=str(artifacts.get("latent_stats_sha256") or ""),
        expected_vae_checkpoint_sha256=vae_checkpoint_sha256,
        expected_vae_revision=vae_config.revision,
        expected_stft_revision=STFT_CONTRACT_VERSION,
        expected_stft_config_sha256=stft_config_sha256,
        expected_posterior_mode=posterior_mode,
        expected_posterior_base_seed=posterior_base_seed,
    )
    if frozen_latent_stats.sha256 != revisions.latent_stats_sha256:
        raise RuntimeError("Inference revisions and latent-stats artifact SHA-256 differ")
    _assert_dit_upstream_contract(
        dit_payload,
        config,
        vae_checkpoint_sha256=vae_checkpoint_sha256,
        refiner_checkpoint_sha256=refiner_checkpoint_sha256,
        latent_stats_sha256=frozen_latent_stats.sha256,
    )
    vae = SpecVAE(vae_config)
    vae.load_state_dict(
        _select_model_state(
            vae_payload,
            preferred_names=("spec_vae", "vae", "generator", "model"),
            component="Spec-VAE",
        ),
        strict=True,
    )
    refiner_values = normalize_refiner_checkpoint_config(
        dict(_component_config(refiner_payload, ("refiner",)))
    )
    refiner = build_refiner_from_mapping(refiner_values)
    refiner_config = refiner.config
    if refiner_config.revision != revisions.refiner_revision:
        raise RuntimeError(
            "Inference preset refiner_revision does not match the checkpoint config"
        )
    refiner_upstream = refiner_payload.get("upstream_revisions")
    if not isinstance(refiner_upstream, Mapping):
        raise RuntimeError("Refiner checkpoint is missing upstream_revisions")
    if refiner_upstream.get("spec_vae_sha256") != vae_checkpoint_sha256:
        raise RuntimeError("Refiner checkpoint Spec-VAE SHA-256 does not match the file")
    if refiner_upstream.get("stft_config_sha256") != stft_config_sha256:
        raise RuntimeError("Refiner checkpoint STFT SHA-256 does not match the VAE")
    refiner.load_state_dict(
        _select_model_state(
            refiner_payload,
            preferred_names=("refiner", "generator", "model"),
            component="Refiner",
        ),
        strict=True,
    )
    vae.revision = vae_config.revision
    refiner.revision = refiner_config.revision
    target_device = torch.device(device)
    model.to(
        device=target_device,
        dtype=torch.bfloat16 if precision == "bf16" else torch.float32,
    ).eval()
    conditioner.to(target_device).eval()
    vae.to(target_device).eval()
    refiner.to(target_device).eval()
    inverse_stft = StereoSTFT(
        STFTConfig(
            sample_rate=48_000,
            n_fft=int(checkpoint_stft["n_fft"]),
            win_length=int(checkpoint_stft["win_length"]),
            hop_length=int(checkpoint_stft["hop_length"]),
            center=bool(checkpoint_stft["center"]),
            normalized=bool(checkpoint_stft.get("normalized", False)),
            drop_nyquist=bool(checkpoint_stft["drop_nyquist"]),
            explicit_left_padding=int(checkpoint_stft["explicit_left_padding"]),
            boundary_window_floor=float(
                checkpoint_stft.get("boundary_window_floor", 0.0)
            ),
        )
    ).to(target_device)
    return RenderInferencePipeline(
        model=model,
        conditioner=conditioner,
        spec_decoder=vae,
        refiner=refiner,
        inverse_stft=inverse_stft,
        latent_stats=frozen_latent_stats,
        flow_config=config["flow"],
        revisions=revisions,
        require_input_tokenizer_revision=bool(
            config["inference"].get("require_input_tokenizer_revision", True)
        ),
        artifact_identities={
            "dit_checkpoint_sha256": dit_checkpoint_sha256,
            "vae_checkpoint_sha256": vae_checkpoint_sha256,
            "refiner_checkpoint_sha256": refiner_checkpoint_sha256,
            "latent_stats_sha256": frozen_latent_stats.sha256,
        },
        inverse_stft_revision=STFT_CONTRACT_VERSION,
        inference_preset=inference_config,
        inference_preset_sha256=config_hash(dict(inference_config)),
        precision=precision,
    )


def train_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the acoustic renderer with torchrun"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-from")
    parser.add_argument("--resume-from")


    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if (
        args.local_rank is not None
        and int(os.environ.get("LOCAL_RANK", args.local_rank)) != args.local_rank
    ):
        raise RuntimeError("--local-rank does not match LOCAL_RANK")
    config = load_config(args.config)
    train(
        config,
        init_from=args.init_from,
        resume_from=args.resume_from,
    )


def _read_text_with_identity(
    direct: str | None,
    path: str | None,
    *,
    name: str,
) -> tuple[str, dict[str, Any]]:
    if (direct is None) == (path is None):
        raise ValueError(f"{name} must provide exactly one of direct text or file input")
    if direct is not None:
        value = str(direct)
        payload = value.encode("utf-8")
        return value, {
            "kind": "inline_utf8",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    source = Path(str(path)).resolve(strict=True)
    try:
        payload = source.read_bytes()
        value = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} file must be UTF-8") from exc
    return value, {
        "kind": "utf8_file",
        "uri": str(source),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _read_text(direct: str | None, path: str | None, *, name: str) -> str:
    return _read_text_with_identity(direct, path, name=name)[0]


def _resolve_contained_artifact(
    raw_path: str,
    *,
    descriptor: Path,
    trusted_root: str | Path | None,
) -> tuple[Path, Path]:
    if "://" in raw_path:
        raise ValueError("semantic token_uri must be a local file path")
    root = (
        Path(trusted_root).resolve(strict=True)
        if trusted_root is not None
        else descriptor.parent.resolve(strict=True)
    )
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = descriptor.parent / candidate
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "semantic token_uriescapetrusted root:"
            f"root={root} artifact={resolved}"
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"semantic token_uri is not a regular file: {resolved}")
    return resolved, root


def _semantic_artifact(
    source: Path,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    str | None,
    str,
    list[int],
    str,
]:
    try:
        payload_bytes = source.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"Semantic artifact does not exist or cannot be read: {source}") from exc
    digest = hashlib.sha256(payload_bytes).hexdigest()
    if source.suffix.lower() == ".npy":
        try:
            values = np.load(io.BytesIO(payload_bytes), allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError("Semantic .npy file cannot be parsed") from exc
        if values.dtype != np.uint16:
            raise TypeError(f"Semantic .npy dtype must be uint16; received {values.dtype}")
        if values.ndim not in {1, 2} or any(int(size) <= 0 for size in values.shape):
            raise ValueError("Semantic .npy must be a non-empty [T] or [B, T] array")
        artifact_shape = list(values.shape)
        ids = torch.from_numpy(values.astype(np.int64, copy=False))
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.numel() and int(ids.max()) >= SEMANTIC_CODEBOOK_SIZE:
            raise ValueError("Semantic .npy token IDs must be in 0..32767")
        return ids, None, None, digest, artifact_shape, str(values.dtype)
    try:
        payload = torch.load(
            io.BytesIO(payload_bytes),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:  # noqa: BLE001 - .
        raise ValueError("Semantic Torch artifact cannot be deserialized safely") from exc
    if isinstance(payload, torch.Tensor):
        ids = payload
        mask = None
        revision = None
    elif isinstance(payload, Mapping):
        ids = payload.get("semantic_ids", payload.get("token_ids"))
        mask = payload.get("semantic_mask", payload.get("frame_mask"))
        revision = payload.get("tokenizer_revision")
    else:
        raise TypeError("Semantic file must contain a tensor or mapping")
    if not isinstance(ids, torch.Tensor):
        raise ValueError("Semantic file is missing semantic_ids or token_ids")
    artifact_shape = list(ids.shape)
    if ids.dtype != torch.long:
        raise TypeError(
            "Semantic Torch artifact token dtype must be int64; "
            f"received {str(ids.dtype).removeprefix('torch.')}"
        )
    if ids.ndim not in {1, 2} or any(int(size) <= 0 for size in ids.shape):
        raise ValueError("Semantic Torch artifact must be a non-empty [T] or [B, T] tensor")
    if ids.numel() and (
        int(ids.min()) < 0 or int(ids.max()) >= SEMANTIC_CODEBOOK_SIZE
    ):
        raise ValueError("Semantic Torch artifact token IDs must be in 0..32767")
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            raise TypeError("Semantic Torch artifact mask must be a tensor")
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.dtype != torch.bool or mask.shape != ids.shape:
            raise TypeError("Semantic Torch artifact mask must be a same-shape bool tensor")
        mask_to_lengths(mask, require_right_padded=True)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Each semantic Torch artifact sample needs at least one valid frame")
    if isinstance(payload, Mapping):
        frame_rate = payload.get("frame_rate")
        if frame_rate is not None and (
            isinstance(frame_rate, bool)
            or not isinstance(frame_rate, (int, float))
            or not np.isfinite(frame_rate)
            or float(frame_rate) != float(LATENT_FRAME_HZ)
        ):
            raise ValueError("Semantic Torch artifact frame_rate must be 25")
        codebook_size = payload.get("codebook_size")
        if codebook_size is not None and (
            not isinstance(codebook_size, int)
            or isinstance(codebook_size, bool)
            or codebook_size != SEMANTIC_CODEBOOK_SIZE
        ):
            raise ValueError("Semantic Torch artifact codebook_size must be 32768")
    return (
        ids,
        mask,
        str(revision) if revision is not None else None,
        digest,
        artifact_shape,
        "int64",
    )


def _load_semantic_with_identity(
    path: str | Path,
    *,
    trusted_root: str | Path | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    str | None,
    dict[str, Any],
]:
    source = Path(path).resolve(strict=True)
    if source.suffix.lower() == ".json":
        descriptor_bytes = source.read_bytes()
        try:
            descriptor = json.loads(descriptor_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Semantic JSON descriptor cannot be parsed") from exc
        if not isinstance(descriptor, Mapping):
            raise TypeError("Semantic JSON descriptor must be a mapping")
        token_uri = descriptor.get("token_uri")
        alias_uri = descriptor.get("uri")
        if token_uri is not None and alias_uri is not None and token_uri != alias_uri:
            raise ValueError("Semantic descriptor contains conflicting token_uri and uri values")
        token_uri = token_uri if token_uri is not None else alias_uri
        if not isinstance(token_uri, str) or not token_uri:
            raise ValueError("Semantic JSON descriptor is missing token_uri")
        token_path, resolved_root = _resolve_contained_artifact(
            token_uri,
            descriptor=source,
            trusted_root=trusted_root,
        )
        (
            ids,
            mask,
            nested_revision,
            actual_sha,
            artifact_shape,
            artifact_dtype,
        ) = _semantic_artifact(token_path)
        expected_sha = descriptor.get("token_sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError("Semantic descriptor must include a 64-character token_sha256")
        try:
            int(expected_sha, 16)
        except ValueError as exc:
            raise ValueError("Semantic descriptor token_sha256 is not hexadecimal") from exc
        if actual_sha != expected_sha.lower():
            raise RuntimeError("Semantic token artifact SHA-256 mismatch")
        if descriptor.get("dtype") != artifact_dtype:
            raise RuntimeError(
                "Semantic descriptor dtype does not match the artifact: "
                f"declared={descriptor.get('dtype')!r} actual={artifact_dtype!r}"
            )
        shape = descriptor.get("shape")
        if shape != artifact_shape:
            raise RuntimeError("Semantic descriptor shape does not match the artifact")
        frame_hz = descriptor.get("frame_hz")
        if (
            isinstance(frame_hz, bool)
            or not isinstance(frame_hz, (int, float))
            or not np.isfinite(frame_hz)
            or float(frame_hz) != float(LATENT_FRAME_HZ)
        ):
            raise RuntimeError("Semantic descriptor frame_hz must be 25")
        codebook_size = descriptor.get("codebook_size")
        if (
            not isinstance(codebook_size, int)
            or isinstance(codebook_size, bool)
            or codebook_size != SEMANTIC_CODEBOOK_SIZE
        ):
            raise RuntimeError("Semantic descriptor codebook_size must be 32768")
        num_frames = descriptor.get("num_frames")
        if (
            not isinstance(num_frames, int)
            or isinstance(num_frames, bool)
            or num_frames != ids.shape[1]
        ):
            raise RuntimeError("Semantic descriptor num_frames does not match the artifact")
        revision = descriptor.get("tokenizer_revision")
        if not isinstance(revision, str) or not revision.strip():
            raise RuntimeError("Semantic descriptor is missing tokenizer_revision")
        if (
            nested_revision is not None
            and revision != nested_revision
        ):
            raise RuntimeError(
                "Semantic descriptor tokenizer revision does not match the nested artifact"
            )
        return ids, mask, revision, {
            "kind": "semantic_descriptor",
            "descriptor_uri": str(source),
            "descriptor_sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
            "descriptor_size_bytes": len(descriptor_bytes),
            "trusted_root": str(resolved_root),
            "artifact_uri": str(token_path),
            "artifact_sha256": actual_sha,
            "artifact_shape": artifact_shape,
            "artifact_dtype": artifact_dtype,
        }
    ids, mask, revision, digest, artifact_shape, artifact_dtype = (
        _semantic_artifact(source)
    )
    return ids, mask, revision, {
        "kind": "semantic_artifact",
        "artifact_uri": str(source),
        "artifact_sha256": digest,
        "artifact_size_bytes": source.stat().st_size,
        "artifact_shape": artifact_shape,
        "artifact_dtype": artifact_dtype,
    }


def _load_semantic(
    path: str | Path,
    *,
    trusted_root: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, str | None]:
    ids, mask, revision, _ = _load_semantic_with_identity(
        path,
        trusted_root=trusted_root,
    )
    return ids, mask, revision


def _atomic_write_audio(
    destination: Path,
    waveform: np.ndarray,
    *,
    sample_rate: int,
    subtype: str,
) -> None:
    temporary = destination.with_name(
        f".{destination.stem}.tmp.{os.getpid()}{destination.suffix}"
    )
    try:
        sf.write(
            temporary,
            waveform,
            sample_rate,
            format=destination.suffix.removeprefix(".").upper(),
            subtype=subtype,
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _new_staging_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.staging.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:

        pass


def _stage_audio(
    path: Path,
    waveform: np.ndarray,
    *,
    sample_rate: int,
    subtype: str,
    output_suffix: str,
) -> None:
    sf.write(
        path,
        waveform,
        sample_rate,
        format=output_suffix.removeprefix(".").upper(),
        subtype=subtype,
    )
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _stage_torch_output(
    path: Path,
    *,
    output: Any,
    waveform: torch.Tensor,
    metadata: Mapping[str, Any],
) -> None:
    with path.open("wb") as handle:
        torch.save(
            {
                "waveform": waveform.cpu(),
                "audio_lengths": output.audio_lengths.detach().cpu(),
                "sample_rate": output.sample_rate,
                "metadata": dict(metadata),
            },
            handle,
        )
        handle.flush()
        os.fsync(handle.fileno())


def _stage_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _publish_render_output(
    destination: Path,
    *,
    output: Any,
    preset: Mapping[str, Any],
) -> dict[str, Any]:

    destination = destination.absolute()
    sidecar = destination.with_suffix(destination.suffix + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _path_lexists(destination) or _path_lexists(sidecar):
        raise FileExistsError(
            "Published render output will not overwrite an existing artifact or sidecar: "
            f"{destination} / {sidecar}"
        )
    suffix = destination.suffix.lower()
    if suffix not in {".wav", ".flac", ".pt", ".pth"}:
        raise ValueError("Render output suffix must be .wav, .flac, .pt, or .pth")
    waveform = output.waveform.detach().float()
    if not torch.isfinite(waveform).all():
        raise FloatingPointError("Render output contains NaN or Inf")
    output_peak = float(waveform.abs().max())
    output_true_peak = max(
        float(
            np.abs(
                soxr.resample(
                    item.transpose(0, 1).cpu().numpy(),
                    output.sample_rate,
                    output.sample_rate * 4,
                    quality="VHQ",
                )
            ).max(initial=0.0)
        )
        for item in waveform
    )
    output_rms = float(waveform.double().square().mean().sqrt())
    output_dc = float(waveform.double().mean())
    peak_policy = str(preset.get("peak_policy", "reject"))
    if peak_policy != "reject":
        raise ValueError("Published inference supports only peak_policy='reject'")
    if output_peak > float(preset.get("max_output_peak", 1.0)):
        raise ValueError(
            f"Renderoutputpeak={output_peak:.6f}exceeds the configured limit; integer PCM would clip"
        )
    if output_true_peak > float(
        preset.get("max_output_true_peak", preset.get("max_output_peak", 1.0))
    ):
        raise ValueError(
            f"Renderoutput4x true peak={output_true_peak:.6f}exceeds the configured true-peak limit"
        )

    base_metadata = dict(getattr(output, "metadata", {}))
    staged_output = _new_staging_path(destination)
    staged_sidecar = _new_staging_path(sidecar)
    output_published = False
    sidecar_published = False
    try:
        if suffix in {".wav", ".flac"}:
            if waveform.shape[0] != 1:
                raise ValueError("Audio file output currently only accepts batch=1")
            length = int(output.audio_lengths[0])
            subtype_key = "wav_subtype" if suffix == ".wav" else "flac_subtype"
            subtype = str(preset[subtype_key])
            _stage_audio(
                staged_output,
                waveform[0, :, :length].cpu().transpose(0, 1).numpy(),
                sample_rate=output.sample_rate,
                subtype=subtype,
                output_suffix=suffix,
            )
        else:
            subtype = "torch"
            _stage_torch_output(
                staged_output,
                output=output,
                waveform=waveform,
                metadata=base_metadata,
            )
        metadata = {
            **base_metadata,
            "format_version": RENDER_OUTPUT_FORMAT_VERSION,
            "status": "COMMITTED",
            "output": {
                "uri": str(destination),
                "sha256": file_sha256(staged_output),
                "size_bytes": staged_output.stat().st_size,
                "subtype": subtype,
                "peak": output_peak,
                "true_peak_4x": output_true_peak,
                "rms": output_rms,
                "dc": output_dc,
                "peak_policy": peak_policy,
            },
        }

        _stage_json(staged_sidecar, metadata)

        os.link(staged_output, destination)
        output_published = True
        os.link(staged_sidecar, sidecar)
        sidecar_published = True
        _fsync_directory(destination.parent)
    except Exception:
        if sidecar_published:
            sidecar.unlink(missing_ok=True)
        if output_published:
            destination.unlink(missing_ok=True)
        _fsync_directory(destination.parent)
        raise
    finally:
        staged_output.unlink(missing_ok=True)
        staged_sidecar.unlink(missing_ok=True)
    output.metadata = metadata
    return metadata


def render_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run acoustic renderer inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dit-checkpoint", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--refiner-checkpoint", required=True)
    parser.add_argument("--latent-stats", required=True)
    parser.add_argument("--semantic", required=True)
    parser.add_argument(
        "--semantic-root",
        help="Trusted root for files referenced by a semantic JSON descriptor",
    )
    parser.add_argument(
        "--tokenizer-revision",
        help="Tokenizer revision for a bare semantic .npy file",
    )
    parser.add_argument(
        "--rewriter-revision",
        required=True,
        help="Revision of the description and lyrics rewriter",
    )
    parser.add_argument(
        "--rewriter-schema-version",
        required=True,
        choices=(REWRITER_SCHEMA_VERSION,),
    )
    parser.add_argument("--description")
    parser.add_argument("--description-file")
    parser.add_argument("--lyrics")
    parser.add_argument("--lyrics-file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument(
        "--global-loudness-lufs",
        type=float,
        help="Target integrated LUFS for loudness-conditioned checkpoints",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solver", choices=("euler", "heun"))
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--cfg-scale", type=float)
    parser.add_argument("--no-refiner", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    preset = config["inference"]
    solver = args.solver or str(preset["solver"])
    num_steps = (
        args.num_steps if args.num_steps is not None else int(preset["num_steps"])
    )
    cfg_scale = (
        args.cfg_scale if args.cfg_scale is not None else float(preset["cfg_scale"])
    )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = build_inference_pipeline(
        config,
        dit_checkpoint=args.dit_checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        refiner_checkpoint=args.refiner_checkpoint,
        latent_stats=args.latent_stats,
        device=device,
    )
    (
        semantic_ids,
        semantic_mask,
        tokenizer_revision,
        semantic_identity,
    ) = _load_semantic_with_identity(
        args.semantic,
        trusted_root=args.semantic_root,
    )
    if args.tokenizer_revision is not None:
        if (
            tokenizer_revision is not None
            and tokenizer_revision != args.tokenizer_revision
        ):
            raise RuntimeError(
                "CLI tokenizer revision does not match the semantic descriptor"
            )
        tokenizer_revision = args.tokenizer_revision
    description, description_identity = _read_text_with_identity(
        args.description, args.description_file, name="description"
    )
    lyrics, lyrics_identity = _read_text_with_identity(
        args.lyrics, args.lyrics_file, name="lyrics"
    )
    output = pipeline.render(
        semantic_ids,
        description,
        lyrics,
        semantic_mask=semantic_mask,
        duration_seconds=args.duration_seconds,
        global_loudness_lufs=args.global_loudness_lufs,
        tokenizer_revision=tokenizer_revision,
        rewriter_revision=args.rewriter_revision,
        rewriter_schema_version=args.rewriter_schema_version,
        seed=args.seed,
        solver=solver,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        use_refiner=not args.no_refiner,
        input_artifact_identities={
            "semantic": semantic_identity,
            "description": description_identity,
            "lyrics": lyrics_identity,
        },
    )
    destination = Path(args.output)
    _publish_render_output(destination, output=output, preset=preset)


def _load_waveform(path: str | Path) -> tuple[torch.Tensor, int]:
    source = Path(path)
    if source.suffix.lower() in {".wav", ".flac"}:
        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        return torch.from_numpy(audio.T), int(sample_rate)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    waveform = payload["waveform"] if isinstance(payload, Mapping) else payload
    sample_rate = (
        int(payload.get("sample_rate", 48_000))
        if isinstance(payload, Mapping)
        else 48_000
    )
    if waveform.ndim == 3 and waveform.shape[0] == 1:
        waveform = waveform[0]
    return waveform.float(), sample_rate


def evaluation_ccpc(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    media_bandwidth_hz: torch.Tensor | None = None,
    fft_sizes: Sequence[int] = (512, 1024, 2048, 4096),
    magnitude_threshold: float = 1.0e-6,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:

    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(0)
    if reference.ndim == 2:
        reference = reference.unsqueeze(0)
    return multiresolution_ccpc(
        prediction,
        reference,
        media_bandwidth_hz=media_bandwidth_hz,
        fft_sizes=fft_sizes,
        magnitude_threshold=magnitude_threshold,
        epsilon=epsilon,
    )


def evaluate_waveforms(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    sample_rate: int,
    magnitude_max_hz: float | None = None,
    stereo_max_hz: float | None = None,
) -> dict[str, float | str]:
    if sample_rate != 48_000:
        raise ValueError("Render only accepts 48 kHz")
    if prediction.shape != reference.shape or prediction.ndim != 2:
        raise ValueError("prediction/reference must be the same shape as [2,N]")
    if prediction.shape[0] != 2:
        raise ValueError("Render only accepts stereo")
    resolved_magnitude_max_hz = (
        sample_rate / 2 if magnitude_max_hz is None else float(magnitude_max_hz)
    )
    resolved_stereo_max_hz = (
        sample_rate / 2 if stereo_max_hz is None else float(stereo_max_hz)
    )
    for name, value in (
        ("magnitude_max_hz", resolved_magnitude_max_hz),
        ("stereo_max_hz", resolved_stereo_max_hz),
    ):
        if not np.isfinite(value) or not 0.0 < value <= sample_rate / 2:
            raise ValueError(f"{name}must be in (0,{sample_rate / 2:g}] Hz")
    prediction = prediction.double()
    reference = reference.double()
    error = prediction - reference
    mse = error.square().mean()

    def si_sdr_value(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_energy = target.square().sum().clamp_min(1.0e-12)
        scale = (estimate * target).sum() / target_energy
        projected = scale * target
        noise = estimate - projected
        return 10.0 * torch.log10(
            projected.square().sum().clamp_min(1.0e-12)
            / noise.square().sum().clamp_min(1.0e-12)
        )

    joint_si_sdr = si_sdr_value(prediction, reference)
    prediction_lrms = torch.cat(
        (prediction, lr_to_ms(prediction.unsqueeze(0)).squeeze(0)),
        dim=0,
    )
    reference_lrms = torch.cat(
        (reference, lr_to_ms(reference.unsqueeze(0)).squeeze(0)),
        dim=0,
    )
    per_stream_si_sdr = torch.stack(
        [
            si_sdr_value(prediction_lrms[index], reference_lrms[index])
            for index in range(4)
        ]
    )
    prediction_batch = prediction.float().unsqueeze(0)
    reference_batch = reference.float().unsqueeze(0)
    metric_device = prediction.device
    lengths = torch.tensor(
        [prediction.shape[-1]],
        dtype=torch.long,
        device=metric_device,
    )
    magnitude_bandwidth = torch.tensor(
        [resolved_magnitude_max_hz],
        dtype=torch.float32,
        device=metric_device,
    )
    stereo_bandwidth = torch.tensor(
        [resolved_stereo_max_hz],
        dtype=torch.float32,
        device=metric_device,
    )
    transform = StereoSTFT().to(metric_device)
    prediction_stft = transform.analyze(prediction_batch, lengths)
    reference_stft = transform.analyze(reference_batch, lengths)
    evaluation_stft = MultiResolutionSTFTLoss(
        resolutions=(
            (128, 32, 128),
            (256, 64, 256),
            (512, 128, 512),
            (1024, 256, 1024),
            (2048, 512, 2048),
            (4096, 1024, 4096),
        ),
        k_weighting=False,
        adaptive_log_magnitude=False,
    )
    stft_components = evaluation_stft.components(
        prediction_batch,
        reference_batch,
        lengths,
        magnitude_bandwidth,
    )
    mel_components = [
        MelSpectralLoss(
            n_fft=n_fft,
            hop_length=n_fft // 4,
            win_length=n_fft,
            n_mels=64,
        ).to(metric_device).components(
            prediction_batch,
            reference_batch,
            lengths,
            magnitude_bandwidth,
        )
        for n_fft in (512, 1024, 2048, 4096)
    ]
    mel_linear = torch.stack(
        [value["mel_linear"] for value in mel_components]
    ).mean()
    mel_log = torch.stack([value["mel_log1p"] for value in mel_components]).mean()
    stereo_frequency_mask = spectral_bandwidth_mask(
        stereo_bandwidth,
        bins=prediction_stft.spectrum.shape[-2],
        bin_hz=sample_rate / (prediction_stft.spectrum.shape[-2] * 2),
        device=prediction_stft.spectrum.device,
    )
    coherence_loss = ccpc_loss(
        prediction_stft.spectrum,
        reference_stft.spectrum,
        reference_stft.spectrum_mask,
        stereo_frequency_mask,
    )
    ccpc_value = evaluation_ccpc(
        prediction_batch,
        reference_batch,
        media_bandwidth_hz=stereo_bandwidth,
    )
    pan = spectral_pan_error(
        prediction_stft.spectrum,
        reference_stft.spectrum,
        reference_stft.spectrum_mask,
        stereo_frequency_mask,
    )
    ipd = absolute_circular_ipd_error(
        prediction_stft.spectrum,
        reference_stft.spectrum,
        reference_stft.spectrum_mask,
        stereo_frequency_mask,
    )
    band_edges = {
        "0_4khz": (0, 80),
        "4_12khz": (80, 240),
        "12_21khz": (240, 420),
        "21_24khz": (420, 480),
    }
    band_complex_l1 = {
        name: float(
            (
                prediction_stft.spectrum[:, :, start:stop]
                - reference_stft.spectrum[:, :, start:stop]
            )
            .abs()
            .mean()
        )
        for name, (start, stop) in band_edges.items()
    }
    prediction_true_peak = float(
        np.abs(
            soxr.resample(
                prediction.transpose(0, 1).cpu().numpy(),
                sample_rate,
                sample_rate * 4,
                quality="VHQ",
            )
        ).max(initial=0.0)
    )
    reference_true_peak = float(
        np.abs(
            soxr.resample(
                reference.transpose(0, 1).cpu().numpy(),
                sample_rate,
                sample_rate * 4,
                quality="VHQ",
            )
        ).max(initial=0.0)
    )
    return {
        "evaluator_version": RENDER_EVALUATOR_VERSION,
        "metric_protocol": RECONSTRUCTION_METRIC_PROTOCOL,
        "magnitude_max_hz": resolved_magnitude_max_hz,
        "stereo_max_hz": resolved_stereo_max_hz,
        "mse": float(mse),
        "si_sdr_db": float(per_stream_si_sdr.mean()),
        "si_sdr_joint_stereo_db": float(joint_si_sdr),
        "si_sdr_l_db": float(per_stream_si_sdr[0]),
        "si_sdr_r_db": float(per_stream_si_sdr[1]),
        "si_sdr_m_db": float(per_stream_si_sdr[2]),
        "si_sdr_s_db": float(per_stream_si_sdr[3]),
        "stft_distance": float(stft_components["mr_stft_linear"]),
        "stft_log1p": float(stft_components["mr_stft_log1p"]),
        "mel_distance": float(mel_linear),
        "mel_log1p": float(mel_log),
        "ccpc": float(ccpc_value),
        "ipd_cosine_similarity": float(1.0 - coherence_loss),
        "spectral_pan_error": float(pan),
        "absolute_ipd_error": float(ipd),
        "band_complex_l1": band_complex_l1,
        "prediction_peak": float(prediction.abs().max()),
        "reference_peak": float(reference.abs().max()),
        "prediction_true_peak_4x": prediction_true_peak,
        "reference_true_peak_4x": reference_true_peak,
        "prediction_rms": float(prediction.square().mean().sqrt()),
        "reference_rms": float(reference.square().mean().sqrt()),
        "prediction_dc": float(prediction.mean()),
        "reference_dc": float(reference.mean()),
        "duration_seconds": prediction.shape[-1] / sample_rate,
    }


def eval_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a rendered waveform")
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--magnitude-max-hz", type=float)
    parser.add_argument("--stereo-max-hz", type=float)
    args = parser.parse_args(argv)
    prediction, prediction_rate = _load_waveform(args.prediction)
    reference, reference_rate = _load_waveform(args.reference)
    if prediction_rate != reference_rate:
        raise ValueError("Prediction and reference sample rates differ")
    metrics = evaluate_waveforms(
        prediction,
        reference,
        sample_rate=prediction_rate,
        magnitude_max_hz=args.magnitude_max_hz,
        stereo_max_hz=args.stereo_max_hz,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Open-Qwen-Music Render")
    parser.add_argument("command", choices=("train", "render", "eval"))
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return
    command = arguments[0]
    if command not in {"train", "render", "eval"}:
        parser.error(f"unknown command: {command!r}")
    remaining = arguments[1:]
    if command == "train":
        train_main(remaining)
    elif command == "render":
        render_main(remaining)
    else:
        eval_main(remaining)


if __name__ == "__main__":
    main()
