#!/usr/bin/env python3
"""Train the public Acoustic VAE or bandwidth refiner recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from open_qwen_music.common.checkpoint import file_sha256
from open_qwen_music.common.config import load_config
from open_qwen_music.common.distributed import (
    barrier,
    cleanup_distributed,
    gather_object_to_rank0,
    init_distributed,
    is_main_process,
)
from open_qwen_music.render.checkpoint import (
    capture_rng_state,
    load_render_checkpoint,
    read_render_checkpoint_state,
    restore_rng_state,
    save_render_checkpoint,
)
from open_qwen_music.render.data import (
    DistributedResumableSampler,
    RenderAudioDataset,
    RenderCollator,
    SyntheticRenderDataset,
)
from open_qwen_music.render.discriminators import (
    AcousticDiscriminator,
    SpectralDiscriminator,
)
from open_qwen_music.render.losses import (
    MelSpectralLoss,
    MultiResolutionSTFTLoss,
    RenderLossConfig,
    SourceAlignedMultiResolutionSTFTLoss,
    SpectralReconstructionLoss,
    SpectroStreamMixedScaleMelLoss,
)
from open_qwen_music.render.refiner import build_refiner_from_mapping
from open_qwen_music.render.spec_vae import (
    SpecVAE,
    SpecVAEConfig,
    normalize_spec_vae_checkpoint_config,
)
from open_qwen_music.render.stft import build_stft_from_mapping
from open_qwen_music.render.trainer_common import (
    JsonlLogger,
    MetricAccumulator,
    build_optimizer,
    build_warmup_cosine_scheduler,
    move_to_device,
    partition_named_parameters_for_muon,
    seed_everything,
)
from open_qwen_music.render.trainer_vae import (
    AdversarialWeights,
    RefinerTrainer,
    SpecVAETrainer,
)


PUBLIC_RECIPES = {
    1: ("spec_vae", "open_qwen_music_vae_stage1_v1"),
    2: ("spec_vae", "open_qwen_music_vae_stage2_v1"),
    3: ("refiner", "open_qwen_music_refiner_v1"),
}


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _device(
    name: str, *, distributed_device: torch.device, world_size: int
) -> torch.device:
    if name == "auto":
        return (
            distributed_device
            if world_size > 1
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    device = torch.device(name)
    if world_size > 1 and device != distributed_device:
        raise ValueError(
            f"Distributed training assigned {distributed_device}, but --device requested {device}"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _validate_config(config: Mapping[str, Any], *, smoke: bool) -> tuple[int, int]:
    stage = int(config.get("stage", 0))
    if stage not in PUBLIC_RECIPES:
        raise ValueError("stage must be 1, 2, or 3")
    component, recipe = PUBLIC_RECIPES[stage]
    if config.get("component") != component:
        raise ValueError(f"Stage {stage} requires component={component}")
    if str((config.get("loss") or {}).get("recipe", "")) != recipe:
        raise ValueError(f"Stage {stage} requires loss.recipe={recipe}")
    if "experiment" in config:
        raise ValueError(
            "Public acoustic configurations must not contain experiment branches"
        )
    data = config.get("data") or {}
    if data.get("audio_integrity_mode") not in {None, "none", "node_once"}:
        raise ValueError("data.audio_integrity_mode must be 'none' or 'node_once'")
    train = config.get("train") or {}
    max_steps = int(train.get("max_steps", 0))
    if max_steps <= 0:
        raise ValueError("train.max_steps must be positive")
    if not smoke and (
        int(train.get("expected_world_size", 0)) <= 0
        or int(train.get("expected_global_batch_size", 0)) <= 0
    ):
        raise ValueError(
            "The public recipe requires explicit world and global batch sizes"
        )
    return stage, max_steps


def _quality_gate(
    config: Mapping[str, Any], requested: int | None, *, smoke: bool
) -> int:
    maximum = int(config["train"]["max_steps"])
    if requested is None:
        if config["train"].get("require_explicit_stop_at_step") and not smoke:
            raise ValueError("This recipe requires --stop-at-step")
        return maximum
    stop = int(requested)
    gates = [int(value) for value in config["train"].get("quality_gate_steps", [])]
    if not 0 < stop <= maximum:
        raise ValueError("--stop-at-step must be within the configured training budget")
    if gates and stop not in gates and not smoke:
        raise ValueError(
            f"--stop-at-step must be one of the configured quality gates: {gates}"
        )
    return stop


def _prepare_output(path: Path, *, resume: bool, require_empty: bool) -> None:
    if require_empty and not resume and path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Fresh-run output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _ddp(
    module: nn.Module, *, device: torch.device, local_rank: int, world_size: int
) -> nn.Module:
    if world_size == 1:
        return module
    return DistributedDataParallel(
        module,
        device_ids=[local_rank] if device.type == "cuda" else None,
        output_device=local_rank if device.type == "cuda" else None,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )


def _optimizer_parameters(module: nn.Module, config: Mapping[str, Any]) -> Any:
    trainable = [
        (name, value)
        for name, value in module.named_parameters()
        if value.requires_grad
    ]
    if not trainable:
        raise ValueError("The optimizer has no trainable parameters")
    if str(config.get("name", "")).lower() != "muon":
        return [value for _, value in trainable]
    return partition_named_parameters_for_muon(
        module, policy=str(config.get("parameter_grouping", "oqm_v1"))
    )[0]


def _scheduler(
    optimizer: torch.optim.Optimizer, train: Mapping[str, Any], *, maximum: int
):
    schedule_steps = min(int(train.get("scheduler_max_steps", maximum)), maximum)
    if schedule_steps <= 0:
        raise ValueError("scheduler_max_steps must be positive")
    return build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=min(int(train.get("warmup_steps", 0)), max(0, schedule_steps - 1)),
        max_steps=schedule_steps,
        min_lr_ratio=float(train.get("min_lr_ratio", 0.0)),
    )


def _reconstruction_loss(config: Mapping[str, Any]) -> SpectralReconstructionLoss:
    loss = config["loss"]
    weights = config["loss_weights"]
    scale = max(float(weights["stft"]), 1.0e-12)
    return SpectralReconstructionLoss(
        RenderLossConfig(
            k_weighting=bool(loss["k_weighting"]),
            adaptive_log_magnitude=bool(loss["adaptive_log_magnitude"]),
            sample_rms_normalization=bool(loss.get("sample_rms_normalization", False)),
            sample_rms_floor=float(loss.get("sample_rms_floor", 1.0e-4)),
            sample_equal_reduction=bool(loss.get("sample_equal_reduction", False)),
            spectrum_complex=float(loss.get("spectrum_complex", 0.0)),
            spectrum_linear=float(loss.get("spectrum_linear", 0.0)),
            spectrum_log1p=float(loss.get("spectrum_log1p", 0.0)),
            lr_ms=float(loss.get("lr_ms", 0.0)),
            if_gd=float(weights.get("if_gd", 0.0)) / scale,
            ccpc=float(loss.get("ccpc", 0.0)),
            absolute_ipd=float(loss.get("absolute_ipd", 0.0)),
            spectral_pan=float(loss.get("spectral_pan", 0.0)),
            phase_weighting=str(loss.get("phase_weighting", "binary")),
            if_gd_objective=str(loss.get("if_gd_objective", "unit_phasor_cosine_v1")),
            kl=float(weights.get("kl", 0.0)) / scale,
            kl_start_dim=int(loss.get("kl_start_dim", 0)),
            kl_reduction=str(loss.get("kl_reduction", "element_mean")),
            kl_mean_scale=float(loss.get("kl_mean_scale", 1.0)),
            kl_variance_scale=float(loss.get("kl_variance_scale", 1.0)),
        )
    )


def _multi_resolution_loss(config: Mapping[str, Any], device: torch.device):
    loss = config["loss"]
    return MultiResolutionSTFTLoss(
        loss["mr_stft_resolutions"],
        k_weighting=bool(loss.get("mr_stft_k_weighting", loss["k_weighting"])),
        adaptive_log_magnitude=bool(
            loss.get("mr_stft_adaptive_log_magnitude", loss["adaptive_log_magnitude"])
        ),
        sample_rms_normalization=bool(
            loss.get("mr_stft_sample_rms_normalization", False)
        ),
        sample_rms_floor=float(loss.get("sample_rms_floor", 1.0e-4)),
        sample_equal_reduction=bool(loss.get("mr_stft_sample_equal_reduction", True)),
        objective_profile=str(
            loss.get("mr_stft_objective_profile", "oqm_linear_adaptive_log_v1")
        ),
        spectral_convergence_epsilon=float(
            loss.get("mr_stft_spectral_convergence_epsilon", 1.0e-8)
        ),
    ).to(device)


def _source_multi_resolution_loss(config: Mapping[str, Any], device: torch.device):
    loss = config["loss"]
    return SourceAlignedMultiResolutionSTFTLoss(
        loss["mr_stft_resolutions"],
        spectral_convergence_weight=float(loss["mr_stft_spectral_convergence_weight"]),
        adaptive_log_weight=float(loss["mr_stft_adaptive_log_weight"]),
        spectral_convergence_epsilon=float(
            loss["mr_stft_spectral_convergence_epsilon"]
        ),
        spectral_convergence_relative_floor=float(
            loss.get("mr_stft_spectral_convergence_relative_floor", 0.0)
        ),
        window_floor=float(loss["mr_stft_window_floor"]),
        k_weighting=bool(loss["mr_stft_k_weighting"]),
        require_media_bandwidth=bool(loss["mr_stft_require_media_bandwidth"]),
        compute_dtype=str(loss["mr_stft_compute_dtype"]),
        view_mode=str(loss["mr_stft_view_mode"]),
        scale_reduction=str(loss["mr_stft_scale_reduction"]),
        view_reduction=str(loss["mr_stft_view_reduction"]),
        sample_reduction="equal_mean"
        if loss.get("mr_stft_sample_equal_reduction")
        else "global_mean",
    ).to(device)


def _mixed_scale_loss(config: Mapping[str, Any], device: torch.device):
    loss = config["loss"]
    if float(loss.get("mixed_scale_spectral_weight", 0.0)) == 0.0:
        return None
    return SpectroStreamMixedScaleMelLoss(
        loss["mixed_scale_resolutions"],
        sample_rate=int(loss["mixed_scale_sample_rate"]),
        n_mels=int(loss["mixed_scale_n_mels"]),
        f_min=float(loss["mixed_scale_f_min"]),
        f_max=float(loss["mixed_scale_f_max"]),
        log_floor=float(loss["mixed_scale_log_floor"]),
        mel_scale=str(loss["mixed_scale_mel_scale"]),
        mel_norm=str(loss["mixed_scale_mel_norm"]),
        scale_reduction=str(loss["mixed_scale_scale_reduction"]),
        stereo_reduction=str(loss["mixed_scale_stereo_reduction"]),
        sample_reduction=str(loss["mixed_scale_sample_reduction"]),
        compute_dtype=str(loss["mixed_scale_compute_dtype"]),
        require_media_bandwidth=bool(loss["mixed_scale_require_media_bandwidth"]),
    ).to(device)


def _mel_loss(config: Mapping[str, Any], device: torch.device):
    loss = config["loss"]
    if float(loss.get("mel_weight", 0.0)) == 0.0:
        return None
    mel = loss["mel"]
    return MelSpectralLoss(
        n_fft=int(mel["n_fft"]),
        hop_length=int(mel["hop_length"]),
        win_length=int(mel["win_length"]),
        n_mels=int(mel["n_mels"]),
        f_min=float(mel["f_min"]),
        f_max=float(mel["f_max"]),
        sample_rms_normalization=bool(mel.get("sample_rms_normalization", False)),
        sample_rms_floor=float(mel.get("sample_rms_floor", 1.0e-4)),
        sample_equal_reduction=bool(mel.get("sample_equal_reduction", True)),
    ).to(device)


def _waveform_discriminator_kwargs(
    config: Mapping[str, Any], *, smoke_model: bool
) -> dict[str, Any]:
    discriminator, smoke = config["discriminator"], config["smoke"]
    return {
        "cqt_enabled": False
        if smoke_model
        else bool(discriminator.get("cqt_enabled", False)),
        "stft_scales": smoke["stft_scales"]
        if smoke_model
        else discriminator["stft_scales"],
        "cqt_scales": () if smoke_model else discriminator.get("cqt_scales", ()),
        "base_channels": int(
            smoke["discriminator_base_channels"]
            if smoke_model
            else discriminator["base_channels"]
        ),
        "depth": int(
            smoke["discriminator_depth"] if smoke_model else discriminator["depth"]
        ),
        "stft_frontend_profile": "spectrostream_public_v1",
        "stft_feature_bands_hz": ()
        if smoke_model
        else discriminator.get("stft_feature_bands_hz", ()),
        "stft_feature_band_layers": (0, 1, 2)
        if smoke_model
        else discriminator.get("stft_feature_band_layers", (0, 1, 2)),
        "stft_feature_band_minimum_bins": 1,
        "stft_feature_band_require_complete_layer": False,
        "stft_feature_band_max_receptive_field_hz": None,
        "loss_reduction": str(
            discriminator.get("loss_reduction", "flat_feature_mean_v1")
        ),
        "family_weights": discriminator.get("loss_family_weights"),
        "loss_compute_dtype": str(discriminator.get("loss_compute_dtype", "float32")),
        "family_diagnostics_enabled": discriminator.get("family_diagnostics_enabled"),
    }


def _load_vae_parent(path: Path, vae: SpecVAE) -> None:
    state = read_render_checkpoint_state(path)
    if state.get("component") != "spec_vae":
        raise RuntimeError("The parent checkpoint is not an Acoustic VAE checkpoint")
    source_model = ((state.get("config") or {}).get("model") or {}).get("spec_vae")
    if (
        not isinstance(source_model, Mapping)
        or SpecVAEConfig.from_dict(
            normalize_spec_vae_checkpoint_config(dict(source_model))
        ) != vae.config
    ):
        raise RuntimeError(
            "The parent Acoustic VAE architecture does not match this recipe"
        )
    weights = (state.get("models") or {}).get("vae")
    if not isinstance(weights, Mapping):
        raise RuntimeError(
            "The parent checkpoint does not contain Acoustic VAE weights"
        )
    vae.load_state_dict(weights, strict=True)


def _parse_upstream(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        key, separator, digest = value.partition("=")
        if not separator or not key or len(digest) != 64:
            raise ValueError("--upstream must use NAME=64_HEX_DIGEST")
        int(digest, 16)
        result[key] = digest.lower()
    return result


def _upstream_revisions(config, supplied, *, smoke, manifest, vae_checkpoint):
    required = tuple(config["checkpoint"]["required_upstream_sha_keys"])
    revisions = dict(supplied)
    actual = {
        "data_manifest_sha256": file_sha256(manifest) if manifest is not None else None,
        "stft_config_sha256": _json_sha256(config["stft"]),
        "stage1_spec_vae_sha256": file_sha256(vae_checkpoint)
        if vae_checkpoint
        else None,
        "spec_vae_sha256": file_sha256(vae_checkpoint) if vae_checkpoint else None,
    }
    for name, digest in actual.items():
        if name in required and digest is not None:
            if name in revisions and revisions[name] != digest:
                raise ValueError(f"{name} does not match the loaded artifact")
            revisions[name] = digest
    code_revision = os.environ.get("OQM_CODE_REVISION")
    if "code_sha256" in required and code_revision:
        if len(code_revision) != 64:
            raise ValueError("OQM_CODE_REVISION must be a 64-character SHA-256")
        int(code_revision, 16)
        revisions["code_sha256"] = code_revision.lower()
    if smoke:
        for index, name in enumerate(required, 1):
            revisions.setdefault(name, f"{index:x}" * 64)
    missing = [name for name in required if name not in revisions]
    if missing:
        raise ValueError("Missing upstream identities: " + ", ".join(missing))
    return revisions, required


def _distributed_checkpoint_state(
    *, stage: int, sampler: DistributedResumableSampler
) -> dict[str, Any]:
    return {
        "format_version": "oqm.render.ddp-state.v1",
        "stage": stage,
        "world_size": sampler.world_size,
        "batch_size_per_rank": sampler.batch_size,
        "sampler_config_hash": getattr(
            sampler, "consensus_config_hash", sampler.config_hash
        ),
    }


def _build_models(config, *, stage, smoke, device, vae_checkpoint):
    model_values = dict(config["model"]["spec_vae"])
    if smoke:
        smoke_config = config["smoke"]
        model_values["encoder_base_channels"] = int(
            smoke_config["encoder_base_channels"]
        )
        model_values["decoder_base_channels"] = int(
            smoke_config["decoder_base_channels"]
        )
    vae = SpecVAE(SpecVAEConfig.from_dict(model_values)).to(device)
    if stage > 1 and not smoke:
        if vae_checkpoint is None:
            raise ValueError("Stage 2 and refiner training require --vae-checkpoint")
        _load_vae_parent(vae_checkpoint, vae)
    refiner = None
    if stage == 3:
        values = dict(config["model"]["refiner"])
        if smoke:
            width = int(config["smoke"]["refiner_width"])
            values.update(
                width=width,
                intermediate_dim=max(8, width * 4),
                depth=int(config["smoke"]["refiner_depth"]),
                kernel_size=3,
            )
        refiner = build_refiner_from_mapping(values).to(device)
        vae.eval()
        for parameter in vae.parameters():
            parameter.requires_grad_(False)
    elif stage == 2:
        for parameter in vae.encoder.parameters():
            parameter.requires_grad_(False)
    return vae, refiner


def _train(arguments, *, rank, local_rank, world_size, distributed_device):
    config = load_config(arguments.config)
    smoke = bool(arguments.smoke)
    stage, _ = _validate_config(config, smoke=smoke)
    if arguments.max_steps is not None:
        if not smoke:
            raise ValueError("--max-steps is reserved for smoke checks")
        config["train"]["max_steps"] = int(arguments.max_steps)
    maximum = int(config["train"]["max_steps"])
    stop = _quality_gate(config, arguments.stop_at_step, smoke=smoke)
    device = _device(
        arguments.device, distributed_device=distributed_device, world_size=world_size
    )
    seed_everything(int(config["train"]["seed"]), rank=rank)
    posterior_generator = None
    if str(config["train"].get("posterior_mode", "sample")) == "sample":
        posterior_generator = torch.Generator(device=device)
        posterior_generator.manual_seed(int(config["train"]["seed"]) + rank)
    deterministic = bool(config["train"].get("deterministic", True))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)
    batch_size = 1 if smoke else int(config["train"]["batch_size_per_rank"])
    if not smoke and world_size != int(config["train"]["expected_world_size"]):
        raise ValueError(
            f"This recipe requires world_size={config['train']['expected_world_size']}"
        )
    if not smoke and world_size * batch_size != int(
        config["train"]["expected_global_batch_size"]
    ):
        raise ValueError("Runtime global batch does not match this recipe")
    config["runtime"] = {
        "format_version": "oqm.render.acoustic-runtime.v1",
        "smoke": smoke,
        "world_size": world_size,
        "batch_size_per_rank": batch_size,
    }
    output = Path(arguments.output_dir or config["train"]["output_dir"])
    if is_main_process():
        _prepare_output(
            output,
            resume=bool(arguments.resume),
            require_empty=bool(config["train"].get("require_empty_output_dir", False)),
        )
    barrier()
    logger = JsonlLogger(output / "train.jsonl") if is_main_process() else None
    vae_checkpoint = (
        Path(arguments.vae_checkpoint) if arguments.vae_checkpoint else None
    )
    vae, refiner = _build_models(
        config, stage=stage, smoke=smoke, device=device, vae_checkpoint=vae_checkpoint
    )
    stft = build_stft_from_mapping(config["stft"]).to(device)
    if smoke:
        dataset = SyntheticRenderDataset(
            count=max(int(config["smoke"]["count"]), world_size * batch_size * maximum),
            samples=int(config["data"]["segment_samples"]),
        )
        manifest_path = None
    else:
        manifest_path = Path(str(config["data"]["manifest"]))
        dataset = RenderAudioDataset(
            manifest_path,
            segment_samples=int(config["data"]["segment_samples"]),
            random_crop=bool(config["data"]["random_crop"]),
            seed=int(
                config["train"].get("data_crop_base_seed", config["train"]["seed"])
            ),
            verify_audio_sha256=bool(config["data"].get("verify_audio_sha256", False)),
            audio_integrity_mode=config["data"].get("audio_integrity_mode"),
            allow_source_audio=bool(config["data"].get("allow_source_audio", True)),
            split=str(config["data"].get("split", "train")),
        )
    sampler = DistributedResumableSampler(
        dataset,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        seed=int(config["train"]["seed"]),
        shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0 if smoke else int(config["train"].get("num_workers", 0)),
        collate_fn=RenderCollator(fixed_samples=int(config["data"]["segment_samples"])),
    )
    waveform = (
        AcousticDiscriminator(
            **_waveform_discriminator_kwargs(config, smoke_model=smoke)
        ).to(device)
        if stage >= 2
        else None
    )
    spectral = None
    if stage == 3 and config["discriminator"].get("spectral_enabled", True):
        values = config["discriminator"]
        spectral = SpectralDiscriminator(
            base_channels=int(
                config["smoke"]["discriminator_base_channels"]
                if smoke
                else values["base_channels"]
            ),
            stft_scales=config["smoke"]["stft_scales"]
            if smoke
            else values["spectral_stft_scales"],
            stft_frontend_profile="spectrostream_public_v1",
        ).to(device)
    generator = refiner if stage == 3 else vae
    generator_optimizer = build_optimizer(
        _optimizer_parameters(generator, config["train"]["generator_optimizer"]),
        config["train"]["generator_optimizer"],
    )
    generator_scheduler = _scheduler(
        generator_optimizer, config["train"], maximum=maximum
    )
    discriminator_optimizers, discriminator_schedulers = {}, {}
    for name, module, key in (
        (
            "waveform_discriminator",
            waveform,
            "discriminator_optimizer"
            if stage == 2
            else "waveform_discriminator_optimizer",
        ),
        ("spectral_discriminator", spectral, "spectral_discriminator_optimizer"),
    ):
        if module is None:
            continue
        optimizer = build_optimizer(
            _optimizer_parameters(module, config["train"][key]), config["train"][key]
        )
        discriminator_optimizers[name] = optimizer
        discriminator_schedulers[name] = _scheduler(
            optimizer, {**config["train"], "warmup_steps": 0}, maximum=maximum
        )
    train_vae = _ddp(vae, device=device, local_rank=local_rank, world_size=world_size)
    train_refiner = (
        _ddp(refiner, device=device, local_rank=local_rank, world_size=world_size)
        if refiner is not None
        else None
    )
    train_waveform = (
        _ddp(waveform, device=device, local_rank=local_rank, world_size=world_size)
        if waveform is not None
        else None
    )
    train_spectral = (
        _ddp(spectral, device=device, local_rank=local_rank, world_size=world_size)
        if spectral is not None
        else None
    )
    if stage <= 2:
        trainer = SpecVAETrainer(
            model=train_vae,
            stft=stft,
            generator_optimizer=generator_optimizer,
            generator_scheduler=generator_scheduler,
            reconstruction_loss=_reconstruction_loss(config),
            mr_stft_loss=_multi_resolution_loss(config, device),
            source_reconstruction_loss=_source_multi_resolution_loss(config, device),
            source_mixed_scale_loss=_mixed_scale_loss(config, device),
            source_mixed_scale_weight=float(
                config["loss"].get("mixed_scale_spectral_weight", 0.0)
            ),
            mel_spectral_loss=_mel_loss(config, device),
            mr_stft_weight=float(config["loss"].get("mr_stft_weight", 0.0)),
            mel_weight=float(config["loss"].get("mel_weight", 0.0)),
            waveform_l1_weight=float(config["loss"].get("waveform_l1_weight", 0.0)),
            discriminator=train_waveform,
            discriminator_optimizer=discriminator_optimizers.get(
                "waveform_discriminator"
            ),
            discriminator_scheduler=discriminator_schedulers.get(
                "waveform_discriminator"
            ),
            stage=stage,
            posterior_mode=str(config["train"].get("posterior_mode", "sample")),
            posterior_generator=posterior_generator,
            posterior_sample_layout=str(
                config["train"].get("posterior_epsilon_draw_layout", "contiguous_bdt")
            ),
            discriminator_warmup_steps=int(
                config["discriminator"].get("warmup_steps", 0)
            ),
            stft_weight=float(config["loss_weights"]["stft"]),
            adversarial_weights=AdversarialWeights(
                float(config["loss_weights"].get("waveform_adversarial", 0.0)),
                float(config["loss_weights"].get("waveform_feature_matching", 0.0)),
            ),
            max_grad_norm=float(config["train"].get("max_grad_norm", 1.0)),
        )
    else:
        trainer = RefinerTrainer(
            vae=vae,
            refiner=train_refiner,
            stft=stft,
            generator_optimizer=generator_optimizer,
            generator_scheduler=generator_scheduler,
            reconstruction_loss=_reconstruction_loss(config),
            mr_stft_loss=_multi_resolution_loss(config, device),
            mel_spectral_loss=_mel_loss(config, device),
            mr_stft_weight=float(config["loss"].get("mr_stft_weight", 0.0)),
            mel_weight=float(config["loss"].get("mel_weight", 0.0)),
            waveform_l1_weight=float(config["loss"].get("waveform_l1_weight", 0.0)),
            stft_consistency_weight=float(
                config["loss"].get("stft_consistency_weight", 0.0)
            ),
            reconstruction_domain=str(
                config["loss"].get("refiner_reconstruction_domain", "raw")
            ),
            waveform_discriminator=train_waveform,
            waveform_discriminator_optimizer=discriminator_optimizers.get(
                "waveform_discriminator"
            ),
            spectral_discriminator=train_spectral,
            spectral_discriminator_optimizer=discriminator_optimizers.get(
                "spectral_discriminator"
            ),
            waveform_discriminator_scheduler=discriminator_schedulers.get(
                "waveform_discriminator"
            ),
            spectral_discriminator_scheduler=discriminator_schedulers.get(
                "spectral_discriminator"
            ),
            waveform_weights=AdversarialWeights(
                float(config["loss_weights"]["waveform_adversarial"]),
                float(config["loss_weights"]["waveform_feature_matching"]),
            ),
            spectral_weights=AdversarialWeights(
                float(config["loss_weights"]["spectral_adversarial"]),
                float(config["loss_weights"]["spectral_feature_matching"]),
            ),
            generator_steps_per_discriminator=int(
                config["train"]["generator_steps_per_discriminator"]
            ),
            spectral_warmup_steps=int(config["discriminator"]["spectral_warmup_steps"]),
            waveform_warmup_steps=int(
                config["discriminator"].get("waveform_warmup_steps", 0)
            ),
            vae_posterior_mode=str(config["train"].get("vae_posterior_mode", "mean")),
            stft_weight=float(config["loss_weights"]["stft"]),
            max_grad_norm=float(config["train"].get("max_grad_norm", 1.0)),
        )
    revisions, required = _upstream_revisions(
        config,
        _parse_upstream(arguments.upstream),
        smoke=smoke,
        manifest=manifest_path,
        vae_checkpoint=vae_checkpoint,
    )
    models = {
        "vae": train_vae,
        **({"refiner": train_refiner} if train_refiner else {}),
        **({"waveform_discriminator": train_waveform} if train_waveform else {}),
        **({"spectral_discriminator": train_spectral} if train_spectral else {}),
    }
    optimizers = {"generator": generator_optimizer, **discriminator_optimizers}
    schedulers = {"generator": generator_scheduler, **discriminator_schedulers}
    distributed_state = _distributed_checkpoint_state(stage=stage, sampler=sampler)
    step = batches_consumed = 0
    audio_seconds = 0.0
    if arguments.resume:
        state = load_render_checkpoint(
            arguments.resume,
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            expected_config=config,
            expected_component=str(config["component"]),
            expected_upstream_revisions=revisions,
            required_upstream_sha_keys=required,
            restore_rng=False,
        )
        bundle = state.sampler_state or {}
        states = bundle.get("states")
        if bundle.get(
            "format_version"
        ) != "oqm.render.distributed-sampler-bundle.v1" or not isinstance(states, list):
            raise RuntimeError(
                "The checkpoint does not contain a distributed sampler bundle"
            )
        sampler.load_state_dict(
            next(value for value in states if value["rank"] == rank)
        )
        rank_states = state.extra_state.get("rank_rng_states")
        if not isinstance(rank_states, list):
            raise RuntimeError("The checkpoint does not contain per-rank random state")
        restore_rng_state(
            next(value for value in rank_states if value["rank"] == rank)["rng"]
        )
        local_rank_state = next(value for value in rank_states if value["rank"] == rank)
        if posterior_generator is not None:
            generator_state = local_rank_state.get("posterior_generator_state")
            if not isinstance(generator_state, torch.Tensor):
                raise RuntimeError(
                    "The checkpoint does not contain the posterior generator state"
                )
            posterior_generator.set_state(generator_state)
        step, batches_consumed, audio_seconds = (
            state.global_step,
            state.batches_consumed,
            state.training_audio_seconds,
        )
    accumulator = MetricAccumulator()

    def save(path: Path) -> None:
        gathered = gather_object_to_rank0(
            {
                "rank": rank,
                "sampler": sampler.state_dict_at_batches_consumed(batches_consumed),
                "rng": capture_rng_state(),
                "posterior_generator_state": None
                if posterior_generator is None
                else posterior_generator.get_state(),
            }
        )
        if is_main_process():
            gathered.sort(key=lambda value: int(value["rank"]))
            save_render_checkpoint(
                path,
                models=models,
                optimizers=optimizers,
                schedulers=schedulers,
                sampler={
                    "format_version": "oqm.render.distributed-sampler-bundle.v1",
                    "world_size": world_size,
                    "states": [value["sampler"] for value in gathered],
                },
                config=config,
                upstream_revisions=revisions,
                required_upstream_sha_keys=required,
                global_step=step,
                epoch=sampler.epoch,
                batches_consumed=batches_consumed,
                training_audio_seconds=audio_seconds,
                component=str(config["component"]),
                distributed_state=distributed_state,
                extra_state={
                    "rank_rng_states": [
                        {
                            "rank": value["rank"],
                            "rng": value["rng"],
                            "posterior_generator_state": value[
                                "posterior_generator_state"
                            ],
                        }
                        for value in gathered
                    ]
                },
            )
        barrier()

    iterator = iter(loader)
    while step < stop:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        result = trainer.train_step(move_to_device(batch, device), global_step=step)
        step += 1
        batches_consumed += 1
        audio_seconds += result.global_audio_seconds
        for name, numerator in result.numerators.items():
            accumulator.add(name, numerator, result.denominators[name])
        if step == 1 or step % int(config["train"]["log_every_steps"]) == 0:
            if logger:
                logger.log(
                    "train",
                    step=step,
                    training_audio_seconds=audio_seconds,
                    metrics=accumulator.compute(distributed=True),
                )
            accumulator.reset()
        if step % int(config["train"]["save_every_steps"]) == 0 or step in set(
            config["train"].get("checkpoint_steps", [])
        ):
            save(output / f"step_{step:06d}.pt")
    save(output / "last.pt")
    if logger:
        logger.log("complete" if step >= maximum else "paused", step=step)


def train(arguments: argparse.Namespace) -> None:
    rank, local_rank, world_size, distributed_device = init_distributed()
    try:
        _train(
            arguments,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            distributed_device=distributed_device,
        )
        barrier()
    finally:
        cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Restore a complete interrupted run")
    parser.add_argument(
        "--vae-checkpoint", help="Frozen VAE parent for Stage 2 or refiner training"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-steps", type=int, help="Override the budget for smoke checks"
    )
    parser.add_argument("--stop-at-step", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--upstream", action="append", default=[], metavar="NAME=SHA256"
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
