
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_qwen_music.common.checkpoint import file_sha256

from .contracts import CODEBOOK_SIZE, FRAME_RATE


T39_CHECKPOINT_FORMAT = "oqm.tokenizer.t39-continuous-bottleneck.v1"
T39_NONLINEAR_KIND = "nonlinear_d32"
T39_PCA_KIND = "pca32_spherical"


def _build_activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"activation must be silu or gelu; received {name!r}")


def _build_normalization(name: str, dim: int) -> nn.Module:
    if name == "layer_norm":
        return nn.LayerNorm(dim)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"normalization must be layer_norm or none; received {name!r}")


class NonlinearContinuousBottleneck(nn.Module):

    def __init__(
        self,
        *,
        input_dim: int,
        expansion_dim: int,
        bottleneck_dim: int,
        activation: str,
        normalization: str,
        bias: bool,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or expansion_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("input/expansion/bottleneck dim must be a positive integer")
        if bottleneck_dim >= input_dim:
            raise ValueError("The diagnostic bottleneck dimension must be strictly smaller than the input dimension")
        self.input_dim = int(input_dim)
        self.expansion_dim = int(expansion_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.activation_name = str(activation)
        self.normalization_name = str(normalization)
        self.use_bias = bool(bias)

        self.encoder_input = nn.Linear(input_dim, expansion_dim, bias=bias)
        self.encoder_norm = _build_normalization(normalization, expansion_dim)
        self.encoder_activation = _build_activation(activation)
        self.encoder_output = nn.Linear(
            expansion_dim, bottleneck_dim, bias=bias
        )
        self.decoder_input = nn.Linear(
            bottleneck_dim, expansion_dim, bias=bias
        )
        self.decoder_norm = _build_normalization(normalization, expansion_dim)
        self.decoder_activation = _build_activation(activation)
        self.decoder_output = nn.Linear(expansion_dim, input_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.encoder_input,
            self.encoder_output,
            self.decoder_input,
            self.decoder_output,
        ):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def architecture(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "expansion_dim": self.expansion_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "activation": self.activation_name,
            "normalization": self.normalization_name,
            "bias": self.use_bias,
            "residual_skip": False,
            "latent_normalization": "none",
        }

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.encoder_output(
            self.encoder_activation(
                self.encoder_norm(self.encoder_input(hidden))
            )
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder_output(
            self.decoder_activation(
                self.decoder_norm(self.decoder_input(latent))
            )
        )

    def forward_with_latent(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(hidden)
        return self.decode(latent), latent

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        reconstructed, _ = self.forward_with_latent(hidden)
        return reconstructed


class PCAContinuousBottleneck(nn.Module):

    def __init__(
        self,
        *,
        input_weight: torch.Tensor,
        input_bias: torch.Tensor,
        output_weight: torch.Tensor,
        output_bias: torch.Tensor,
        eigenvalues: torch.Tensor,
        eigenvalue_floor: float,
    ) -> None:
        super().__init__()
        if input_weight.ndim != 2 or output_weight.ndim != 2:
            raise ValueError("PCA input/output weight must be a two-dimensional tensor")
        code_dim, input_dim = input_weight.shape
        if output_weight.shape != (input_dim, code_dim):
            raise ValueError("PCA input/output weight shape is not an inverse mapping of each other")
        if input_bias.shape != (code_dim,) or output_bias.shape != (input_dim,):
            raise ValueError("PCA bias shape is incompatible with the weight")
        if eigenvalues.shape != (code_dim,):
            raise ValueError("PCA eigenvalue shape is incompatible with code_dim")
        if eigenvalue_floor <= 0:
            raise ValueError("eigenvalue_floor must be a positive number")
        self.input_dim = int(input_dim)
        self.bottleneck_dim = int(code_dim)
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.register_buffer("input_weight", input_weight.detach().clone())
        self.register_buffer("input_bias", input_bias.detach().clone())
        self.register_buffer("output_weight", output_weight.detach().clone())
        self.register_buffer("output_bias", output_bias.detach().clone())
        self.register_buffer("eigenvalues", eigenvalues.detach().clone())

    @classmethod
    def fit(
        cls,
        mean: torch.Tensor,
        covariance: torch.Tensor,
        *,
        bottleneck_dim: int,
        eigenvalue_floor: float,
    ) -> "PCAContinuousBottleneck":
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be a square matrix")
        input_dim = int(covariance.shape[0])
        if mean.shape != (input_dim,):
            raise ValueError("mean shape is incompatible with the covariance")
        if not 0 < bottleneck_dim < input_dim:
            raise ValueError("PCA bottleneck_dim must be in [1, input_dim - 1]")
        if eigenvalue_floor <= 0:
            raise ValueError("eigenvalue_floor must be a positive number")
        covariance = (covariance.double() + covariance.double().T) * 0.5
        mean = mean.double()
        if not torch.isfinite(covariance).all() or not torch.isfinite(mean).all():
            raise RuntimeError("PCA statistics include NaN/Inf")
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)[:bottleneck_dim]
        selected_values = eigenvalues[order]
        tolerance = (
            torch.finfo(selected_values.dtype).eps
            * input_dim
            * max(float(selected_values.abs().max()), 1.0)
        )
        if float(selected_values.min()) < -tolerance:
            raise RuntimeError(
                "PCA top-D Significant negative eigenvalues appear:"
                f"{float(selected_values.min()):.3e}"
            )
        selected_vectors = eigenvectors[:, order]
        standard_deviations = selected_values.clamp_min(
            eigenvalue_floor
        ).sqrt()
        input_weight = selected_vectors.T / standard_deviations.unsqueeze(1)
        input_bias = -(input_weight @ mean)
        output_weight = (
            selected_vectors
            * standard_deviations.unsqueeze(0)
            * math.sqrt(bottleneck_dim)
        )
        output_bias = mean
        return cls(
            input_weight=input_weight.float(),
            input_bias=input_bias.float(),
            output_weight=output_weight.float(),
            output_bias=output_bias.float(),
            eigenvalues=selected_values.float(),
            eigenvalue_floor=eigenvalue_floor,
        )

    @property
    def architecture(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "projection": "top-D PCA whiten",
            "latent_normalization": "per-frame L2",
            "decoder_compensation": f"sqrt({self.bottleneck_dim})",
            "eigenvalue_floor": self.eigenvalue_floor,
            "residual_skip": False,
        }

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = F.linear(hidden, self.input_weight, self.input_bias)
        return F.normalize(projected, dim=-1)

    def forward_with_latent(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(hidden)
        reconstructed = F.linear(
            latent, self.output_weight, self.output_bias
        )
        return reconstructed, latent

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        reconstructed, _ = self.forward_with_latent(hidden)
        return reconstructed


class StreamingHiddenMoments:

    def __init__(
        self,
        dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.count = 0
        self.mean = torch.zeros(dim, device=device, dtype=dtype)
        self.m2 = torch.zeros(dim, dim, device=device, dtype=dtype)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != self.mean.numel():
            raise ValueError(
                f"hidden must have shape [N,{self.mean.numel()}]; "
                f"received {tuple(values.shape)}"
            )
        if values.shape[0] == 0:
            return
        values = values.to(device=self.mean.device, dtype=self.mean.dtype)
        if not torch.isfinite(values).all():
            raise RuntimeError("hidden sample contains NaN/Inf")
        batch_count = int(values.shape[0])
        batch_mean = values.mean(dim=0)
        centered = values - batch_mean
        batch_m2 = centered.T @ centered
        if self.count == 0:
            self.count = batch_count
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            return
        previous = self.count
        total = previous + batch_count
        delta = batch_mean - self.mean
        self.m2.add_(batch_m2 + (previous * batch_count / total) * torch.outer(delta, delta))
        self.mean.add_(delta, alpha=batch_count / total)
        self.count = total

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count < 2:
            raise RuntimeError(f"at least two frames are required; received {self.count}")
        return (
            self.mean.double().cpu(),
            (self.m2 / (self.count - 1)).double().cpu(),
        )


@dataclass
class CroppedHiddenBatch:
    hidden: torch.Tensor
    frame_mask: torch.Tensor
    ctc_enabled: torch.Tensor


def crop_hidden_batch(
    hidden: torch.Tensor,
    frame_mask: torch.Tensor,
    ctc_enabled: torch.Tensor,
    *,
    crop_frames: int,
    generator: torch.Generator,
) -> CroppedHiddenBatch:

    if hidden.ndim != 3 or frame_mask.shape != hidden.shape[:2]:
            raise ValueError("hidden and frame_mask must have shapes [B, T, D] and [B, T]")
    if ctc_enabled.shape != (hidden.shape[0],):
        raise ValueError("ctc_enabled must be [B]")
    if crop_frames <= 0:
        raise ValueError("crop_frames must be a positive integer")
    lengths = frame_mask.sum(dim=1).long().cpu().tolist()
    rows: list[torch.Tensor] = []
    kept_ctc = []
    for row, length in enumerate(lengths):
        if length <= 0:
            continue
        selected_length = min(int(length), crop_frames)
        maximum_start = int(length) - selected_length
        start = (
            int(
                torch.randint(
                    maximum_start + 1, (), generator=generator
                ).item()
            )
            if maximum_start > 0
            else 0
        )
        rows.append(hidden[row, start : start + selected_length])
        kept_ctc.append(ctc_enabled[row])
    if not rows:
        raise RuntimeError("batch is not valid layer13 frame")
    maximum = max(int(row.shape[0]) for row in rows)
    output = hidden.new_zeros((len(rows), maximum, hidden.shape[-1]))
    output_mask = torch.zeros(
        len(rows), maximum, dtype=torch.bool, device=hidden.device
    )
    for row_index, values in enumerate(rows):
        length = int(values.shape[0])
        output[row_index, :length] = values
        output_mask[row_index, :length] = True
    return CroppedHiddenBatch(
        hidden=output,
        frame_mask=output_mask,
        ctc_enabled=torch.stack(kept_ctc).to(
            device=hidden.device, dtype=torch.bool
        ),
    )


def freeze_teacher(model: nn.Module) -> None:

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def _run_upper(
    model: Any, hidden: torch.Tensor, frame_mask: torch.Tensor
) -> torch.Tensor:
    return model.encoder.forward_range(
        hidden,
        frame_mask,
        start=model.insertion_layer,
        end=None,
        attention_causal=model.attention_causal,
        convolution_causal=model.conformer_conv_causal,
    )


def _output_mask(frame_mask: torch.Tensor, target_length: int) -> torch.Tensor:
    repeated = frame_mask.repeat_interleave(4, dim=1)
    if repeated.shape[1] >= target_length:
        return repeated[:, :target_length]
    return F.pad(repeated, (0, target_length - repeated.shape[1]), value=False)


@torch.no_grad()
def predict_with_continuous_bottleneck(
    model: Any,
    bottleneck: nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:

    hidden, frame_mask = model.extract_quantizer_inputs(
        batch["waveform"], batch["waveform_num_samples"]
    )
    reconstructed = bottleneck(hidden)
    reconstructed = reconstructed.masked_fill(~frame_mask.unsqueeze(-1), 0.0)
    upper = _run_upper(model, reconstructed, frame_mask)
    predictions = model.heads.predict(
        upper,
        mel_target_length=batch["mel_target"].shape[1],
        chroma_target_length=batch["chroma_target"].shape[1],
        mask=frame_mask,
        causal=model.attention_causal and model.causal_heads,
    )
    return predictions, frame_mask.sum(dim=1)


def _masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    active = mask.unsqueeze(-1).expand_as(prediction)
    if not bool(active.any()):
        return prediction.sum() * 0.0
    return (prediction.float() - target.float()).abs()[active].mean()


def _masked_ctc_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("CTC distillation temperature must be a positive number")
    if not bool(mask.any()):
        return student_logits.sum() * 0.0
    student = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher = F.softmax(teacher_logits.float() / temperature, dim=-1)
    per_frame = F.kl_div(student, teacher, reduction="none").sum(dim=-1)
    return per_frame[mask].mean() * (temperature * temperature)


def _masked_normalized_hidden_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    active = mask.unsqueeze(-1).expand_as(prediction)
    if not bool(active.any()):
        return prediction.sum() * 0.0
    prediction_values = prediction.float()[active]
    target_values = target.float()[active]
    mse = (prediction_values - target_values).square().mean()
    target_power = target_values.square().mean().detach().clamp_min(1e-6)
    return mse / target_power


def distillation_step(
    model: Any,
    bottleneck: NonlinearContinuousBottleneck,
    hidden: torch.Tensor,
    frame_mask: torch.Tensor,
    ctc_enabled: torch.Tensor,
    *,
    loss_weights: dict[str, float],
    ctc_temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    required = ("ctc_kl", "mel_l1", "chroma_l1", "hidden_reconstruction")
    missing = [key for key in required if key not in loss_weights]
    if missing:
        raise ValueError(f"loss_weights is missing fields: {missing}")
    if any(float(loss_weights[key]) < 0 for key in required):
        raise ValueError("all continuous bottleneck loss weights must be non-negative")
    if hidden.ndim != 3 or frame_mask.shape != hidden.shape[:2]:
        raise ValueError("hidden and frame_mask shapes are incompatible")
    if ctc_enabled.shape != (hidden.shape[0],):
        raise ValueError("ctc_enabled must be [B]")

    hidden = hidden.detach()
    target_length = 4 * hidden.shape[1]
    with torch.no_grad():
        teacher_upper = _run_upper(model, hidden, frame_mask)
        teacher_predictions = model.heads.predict(
            teacher_upper,
            mel_target_length=target_length,
            chroma_target_length=target_length,
            mask=frame_mask,
            causal=model.attention_causal and model.causal_heads,
        )

    reconstructed, latent = bottleneck.forward_with_latent(hidden)
    reconstructed = reconstructed.masked_fill(
        ~frame_mask.unsqueeze(-1), 0.0
    )
    student_upper = _run_upper(model, reconstructed, frame_mask)
    student_predictions = model.heads.predict(
        student_upper,
        mel_target_length=target_length,
        chroma_target_length=target_length,
        mask=frame_mask,
        causal=model.attention_causal and model.causal_heads,
    )

    ctc_mask = frame_mask & ctc_enabled.to(
        device=frame_mask.device, dtype=torch.bool
    ).unsqueeze(1)
    acoustic_mask = _output_mask(frame_mask, target_length)
    losses = {
        "ctc_kl": _masked_ctc_kl(
            student_predictions["ctc_logits"],
            teacher_predictions["ctc_logits"],
            ctc_mask,
            temperature=ctc_temperature,
        ),
        "mel_l1": _masked_l1(
            student_predictions["mel"],
            teacher_predictions["mel"],
            acoustic_mask,
        ),
        "chroma_l1": _masked_l1(
            student_predictions["chroma"],
            teacher_predictions["chroma"],
            acoustic_mask,
        ),
        "hidden_reconstruction": _masked_normalized_hidden_mse(
            reconstructed, hidden, frame_mask
        ),
    }
    total = sum(
        float(loss_weights[key]) * value for key, value in losses.items()
    )
    latent_values = latent.float()[frame_mask]
    metrics = {
        **{f"loss_{key}": value.detach() for key, value in losses.items()},
        "loss_total": total.detach(),
        "latent_rms": latent_values.square().mean().sqrt().detach(),
        "reconstruction_l1": _masked_l1(
            reconstructed, hidden, frame_mask
        ).detach(),
        "active_frames": frame_mask.sum().to(total.dtype).detach(),
        "ctc_active_frames": ctc_mask.sum().to(total.dtype).detach(),
    }
    return total, metrics


@dataclass
class LoadedContinuousBottleneck:
    module: NonlinearContinuousBottleneck | PCAContinuousBottleneck
    metadata: dict[str, Any]


def load_continuous_bottleneck_checkpoint(
    path: str | Path,
    *,
    expected_teacher_sha256: str | None,
    expected_insertion_layer: int,
    expected_input_dim: int,
    device: torch.device,
) -> LoadedContinuousBottleneck:

    checkpoint_path = Path(path).resolve()
    state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if state.get("format_version") != T39_CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"{checkpoint_path} is not a supported continuous bottleneck checkpoint"
        )
    hard_contract = state.get("hard_contract") or {}
    expected_contract = {
        "frame_rate": FRAME_RATE,
        "single_codebook_size": CODEBOOK_SIZE,
        "insertion_layer": expected_insertion_layer,
        "input_dim": expected_input_dim,
        "bottleneck_dim": 32,
    }
    mismatches = {
        key: (hard_contract.get(key), expected)
        for key, expected in expected_contract.items()
        if hard_contract.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"The continuous bottleneck checkpoint contract differs: {mismatches}")
    teacher = state.get("teacher") or {}
    if (
        expected_teacher_sha256 is not None
        and teacher.get("sha256") != expected_teacher_sha256
    ):
        raise RuntimeError(
            "The checkpoint does not match the bound Stage 3 teacher SHA-256: "
            f"{teacher.get('sha256')} != {expected_teacher_sha256}"
        )

    kind = str(state.get("kind"))
    architecture = state.get("architecture") or {}
    if kind == T39_NONLINEAR_KIND:
        module: NonlinearContinuousBottleneck | PCAContinuousBottleneck = (
            NonlinearContinuousBottleneck(
                input_dim=int(architecture["input_dim"]),
                expansion_dim=int(architecture["expansion_dim"]),
                bottleneck_dim=int(architecture["bottleneck_dim"]),
                activation=str(architecture["activation"]),
                normalization=str(architecture["normalization"]),
                bias=bool(architecture["bias"]),
            )
        )
    elif kind == T39_PCA_KIND:
        model_state = state["model"]
        module = PCAContinuousBottleneck(
            input_weight=model_state["input_weight"],
            input_bias=model_state["input_bias"],
            output_weight=model_state["output_weight"],
            output_bias=model_state["output_bias"],
            eigenvalues=model_state["eigenvalues"],
            eigenvalue_floor=float(architecture["eigenvalue_floor"]),
        )
    else:
        raise RuntimeError(f"unsupported continuous bottleneck checkpoint kind={kind!r}")
    module.load_state_dict(state["model"], strict=True)
    module.to(device).eval()
    metadata = {
        "path": str(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "kind": kind,
        "global_step": int(state.get("global_step", 0)),
        "architecture": architecture,
        "teacher": teacher,
        "hard_contract": hard_contract,
        "fit": state.get("fit"),
    }
    return LoadedContinuousBottleneck(module=module, metadata=metadata)
