
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

import torch
import torch.nn as nn

from open_qwen_music.common.checkpoint import config_hash

from .cache import FrozenLatentStats, destandardize_latents
from .conditioning import (
    REWRITER_SCHEMA_VERSION,
    ConditioningOutput,
    RenderConditioner,
)
from .contracts import (
    LATENT_DIM,
    LATENT_FRAME_HZ,
    SAMPLE_RATE,
    SAMPLES_PER_LATENT_FRAME,
    SPEC_FRAMES_PER_LATENT_FRAME,
    STFT_BINS,
    lengths_to_mask,
    mask_to_lengths,
)
from .flow import (
    FLOW_FORMAT_VERSION,
    SUPPORTED_SOLVERS,
    FlowConfig,
    classifier_free_guidance,
    sample_source_like,
    sampling_interval,
    sampling_schedule,
    solve_flow,
)
from .types import LatentOutput, RenderOutput, STFTOutput


@dataclass(frozen=True)
class InferenceRevisions:
    checkpoint_revision: str
    tokenizer_revision: str
    vae_revision: str
    refiner_revision: str
    text_encoder_revision: str
    text_tokenizer_revision: str
    text_cache_revision: str
    rewriter_revision: str
    latent_cache_revision: str
    latent_stats_sha256: str
    flow_format_version: str = FLOW_FORMAT_VERSION

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str):
                raise TypeError(f"inference revision {name} must be the string")
            normalized = value.strip().lower()
            if normalized.startswith(("pin-", "new-", "unresolved:")) or normalized in {
                "",
                "main",
                "master",
                "latest",
                "head",
                "unknown",
                "unresolved",
                "none",
            }:
                raise ValueError(f"inference revision {name} must be fixed,received {value!r}")
        if self.flow_format_version != FLOW_FORMAT_VERSION:
            raise ValueError(
                "inference flow revision does not match:"
                f"{self.flow_format_version}!={FLOW_FORMAT_VERSION}"
            )
        if len(self.latent_stats_sha256) != 64:
            raise ValueError("latent_stats_sha256 must be64bitSHA-256")
        try:
            int(self.latent_stats_sha256, 16)
        except ValueError as exc:
            raise ValueError("latent_stats_sha256 is not hexadecimalSHA-256") from exc

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InferenceRevisions":
        if not isinstance(value, Mapping):
            raise TypeError("inference revisions must be mapping")
        normalized = dict(value)
        aliases = {
            "dit_revision": "checkpoint_revision",
            "semantic_tokenizer_revision": "tokenizer_revision",
            "spec_vae_revision": "vae_revision",
            "model_revision": "text_encoder_revision",
            "cache_revision": "text_cache_revision",
        }
        for old, new in aliases.items():
            if old in normalized:
                if new in normalized and normalized[new] != normalized[old]:
                    raise ValueError(f"inference revisionAlias conflict:{old}/{new}")
                normalized.setdefault(new, normalized[old])
                del normalized[old]
        unknown = set(normalized) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"inference revisionscontains unknown fields:{sorted(unknown)}")
        result = cls(
            **{
                name: normalized.get(
                    name,
                    FLOW_FORMAT_VERSION if name == "flow_format_version" else "",
                )
                for name in cls.__dataclass_fields__
            }
        )
        result.validate()
        return result


class SpecDecoder(Protocol):
    def decode(
        self, latents: torch.Tensor | LatentOutput, **kwargs: Any
    ) -> torch.Tensor | STFTOutput: ...


class SpectrumRefiner(Protocol):
    def __call__(
        self, spectrum: torch.Tensor | STFTOutput, **kwargs: Any
    ) -> torch.Tensor | STFTOutput: ...


class InverseSTFT(Protocol):
    def inverse(
        self,
        spectrum: torch.Tensor | STFTOutput,
        lengths: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor: ...


def _module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _callable_signature(function: Any) -> inspect.Signature | None:
    target = function.forward if isinstance(function, nn.Module) else function
    try:
        return inspect.signature(target)
    except (TypeError, ValueError):
        return None


def _call_with_supported_kwargs(function: Any, positional: Any, **kwargs: Any) -> Any:
    signature = _callable_signature(function)
    if signature is None:
        return function(positional)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    selected = (
        kwargs
        if accepts_kwargs
        else {
            key: value for key, value in kwargs.items() if key in signature.parameters
        }
    )
    return function(positional, **selected)


def _dependency_revision(dependency: Any) -> str | None:
    for name in (
        "revision",
        "checkpoint_revision",
        "model_revision",
        "format_revision",
    ):
        value = getattr(dependency, name, None)
        if value is not None:
            return str(value)
    return None


def _validate_dependency_revision(name: str, dependency: Any, expected: str) -> None:
    observed = _dependency_revision(dependency)
    if observed is None:
        raise RuntimeError(f"{name} dependencyMissing verifiablerevision,Reject unbound dependencies")
    if observed != expected:
        raise RuntimeError(
            f"{name} dependency revision does not match:expected={expected} actual={observed}"
        )


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_identity(value: torch.Tensor) -> dict[str, Any]:
    cpu = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    dtype = str(cpu.dtype).removeprefix("torch.")
    shape = [int(size) for size in cpu.shape]
    digest.update(
        json.dumps(
            {"dtype": dtype, "shape": shape},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(cpu.view(torch.uint8).numpy().tobytes(order="C"))
    return {
        "sha256": digest.hexdigest(),
        "dtype": dtype,
        "shape": shape,
    }


def _raw_input_identity(
    value: str | Sequence[str] | torch.Tensor,
) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {"kind": "token_ids", **_tensor_identity(value)}
    values = [value] if isinstance(value, str) else list(value)
    return {
        "kind": "utf8_text",
        "sha256": _sha256_json(values),
        "items": len(values),
    }


def _normalize_text_input(
    value: str | Sequence[str] | torch.Tensor,
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    name: str,
    conditioner: RenderConditioner,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = (
        conditioner.description_max_tokens
        if name == "description"
        else conditioner.lyrics_max_tokens
    )
    if isinstance(value, torch.Tensor):
        if mask is None:
            raise ValueError(f"{name} tensorinput must be provided explicitlymask")
        input_ids = value.to(device=device)
        if input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise TypeError(f"{name} tensor must be int64[B,L]")
        attention_mask = mask.to(device=device)
    else:
        if mask is not None:
            raise ValueError(f"{name}String input must not be provided in additionmask")
        values = [value] if isinstance(value, str) else list(value)
        if len(values) == 1 and batch_size > 1:
            values = values * batch_size
        if len(values) != batch_size or not all(
            isinstance(item, str) for item in values
        ):
            raise ValueError(f"{name} The number of strings must equal batch size")
        input_ids, attention_mask = conditioner.text_encoder.tokenize(
            values, max_length=maximum, device=device
        )
    if input_ids.shape[0] != batch_size:
        raise ValueError(f"{name} batch size does not match")
    if attention_mask.dtype != torch.bool or attention_mask.shape != input_ids.shape:
        raise TypeError(f"{name}_mask must be the same shape as bool tensor")
    if input_ids.shape[1] > maximum:
        raise ValueError(f"{name} is longer than {maximum}")
    if input_ids.numel() and int(input_ids.min()) < 0:
        raise ValueError(f"{name} input_idsmust not contain negative numbers")
    return input_ids, attention_mask


def _extract_spectrum(
    value: torch.Tensor | STFTOutput,
) -> torch.Tensor:
    return value.spectrum if isinstance(value, STFTOutput) else value


class RenderInferencePipeline:
    def __init__(
        self,
        *,
        model: nn.Module,
        conditioner: RenderConditioner,
        spec_decoder: Any,
        refiner: Any,
        inverse_stft: Any,
        latent_stats: FrozenLatentStats,
        flow_config: FlowConfig | Mapping[str, Any],
        revisions: InferenceRevisions | Mapping[str, Any],
        require_input_tokenizer_revision: bool = False,
        artifact_identities: Mapping[str, str] | None = None,
        inference_preset: Mapping[str, Any] | None = None,
        inference_preset_sha256: str | None = None,
        precision: str = "fp32",
        inverse_stft_revision: str | None = None,
    ) -> None:
        self.model = model
        self.conditioner = conditioner
        self.spec_decoder = spec_decoder
        self.refiner = refiner
        self.inverse_stft = inverse_stft
        self.latent_stats = latent_stats
        self.flow_config = (
            flow_config
            if isinstance(flow_config, FlowConfig)
            else FlowConfig.from_mapping(flow_config)
        )
        self.flow_config.validate()
        self.revisions = (
            revisions
            if isinstance(revisions, InferenceRevisions)
            else InferenceRevisions.from_mapping(revisions)
        )
        self.revisions.validate()
        if self.revisions.flow_format_version != self.flow_config.format_version:
            raise RuntimeError("inference revisionsandflow configVersion inconsistent")
        if self.latent_stats.sha256 != self.revisions.latent_stats_sha256:
            raise RuntimeError(
                "latent stats revisiondoes not match:"
                f"expected={self.revisions.latent_stats_sha256} "
                f"actual={self.latent_stats.sha256}"
            )
        if not isinstance(require_input_tokenizer_revision, bool):
            raise TypeError("require_input_tokenizer_revisionmust bebool")
        self.require_input_tokenizer_revision = require_input_tokenizer_revision
        precision = str(precision).lower()
        if precision not in {"fp32", "bf16"}:
            raise ValueError("inference precisionmust befp32/bf16")
        self.precision = precision
        self.artifact_identities = {
            str(name): str(value) for name, value in (artifact_identities or {}).items()
        }
        for name, value in self.artifact_identities.items():
            if len(value) != 64:
                raise ValueError(f"artifact identity {name}must be64bitSHA-256")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(f"artifact identity {name}is notSHA-256") from exc
        self.inference_preset = dict(inference_preset or {})
        self.inference_preset_sha256 = (
            str(inference_preset_sha256)
            if inference_preset_sha256 is not None
            else None
        )
        if bool(self.inference_preset) != (self.inference_preset_sha256 is not None):
            raise ValueError(
                "inference_presetandinference_preset_sha256must be provided or both"
            )
        if self.inference_preset_sha256 is not None:
            if len(self.inference_preset_sha256) != 64:
                raise ValueError("inference_preset_sha256must be64bitSHA-256")
            try:
                int(self.inference_preset_sha256, 16)
            except ValueError as exc:
                raise ValueError("inference_preset_sha256is notSHA-256") from exc
            actual_preset_sha256 = config_hash(self.inference_preset)
            if self.inference_preset_sha256.lower() != actual_preset_sha256:
                raise RuntimeError("inference_preset_sha256andbase presetThe content is inconsistent")
        if (
            self.conditioner.provenance.model_revision
            != self.revisions.text_encoder_revision
            or self.conditioner.provenance.tokenizer_revision
            != self.revisions.text_tokenizer_revision
            or self.conditioner.provenance.cache_revision
            != self.revisions.text_cache_revision
        ):
            raise RuntimeError(
                "conditioner of encoder/tokenizer/cache revision and reasoning preset inconsistent"
            )
        _validate_dependency_revision(
            "Spec decoder", spec_decoder, self.revisions.vae_revision
        )
        _validate_dependency_revision(
            "Refiner", refiner, self.revisions.refiner_revision
        )
        if inverse_stft_revision is not None:
            _validate_dependency_revision(
                "Inverse STFT",
                inverse_stft,
                inverse_stft_revision,
            )
        model_device = _module_device(self.model)
        self.conditioner.to(device=model_device)
        for dependency in (self.spec_decoder, self.refiner, self.inverse_stft):
            if isinstance(dependency, nn.Module):
                dependency.to(device=model_device)
        self.model.eval()
        self.conditioner.eval()
        if isinstance(self.spec_decoder, nn.Module):
            self.spec_decoder.eval()
        if isinstance(self.refiner, nn.Module):
            self.refiner.eval()
        if isinstance(self.inverse_stft, nn.Module):
            self.inverse_stft.eval()
        self._validate_model_precision()

    @property
    def device(self) -> torch.device:
        return _module_device(self.model)

    def _validate_model_precision(self) -> None:
        expected = torch.bfloat16 if self.precision == "bf16" else torch.float32
        floating = {
            parameter.dtype
            for parameter in self.model.parameters()
            if parameter.is_floating_point()
        }
        if floating and floating != {expected}:
            raise RuntimeError(
                "DiTparameterdtypeandinference precisioninconsistent:"
                f"expected={expected} actual={sorted(map(str, floating))}"
            )

    def _validate_semantic(
        self,
        semantic_ids: torch.Tensor,
        semantic_mask: torch.Tensor | None,
        *,
        tokenizer_revision: str | None,
        duration_seconds: float | Sequence[float] | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        semantic_ids = semantic_ids.to(device=self.device)
        if semantic_ids.dtype != torch.long or semantic_ids.ndim != 2:
            raise TypeError("semantic_ids must be int64[B,T]")
        if semantic_mask is None:
            semantic_mask = torch.ones_like(semantic_ids, dtype=torch.bool)
        else:
            semantic_mask = semantic_mask.to(device=self.device)
        if (
            semantic_mask.dtype != torch.bool
            or semantic_mask.shape != semantic_ids.shape
        ):
            raise TypeError("semantic_mask must be the same shape as bool tensor")
        lengths = mask_to_lengths(semantic_mask, require_right_padded=True)
        if bool((lengths <= 0).any()):
            raise ValueError("semantic sequence cannot be empty")
        max_frames = getattr(getattr(self.model, "config", None), "max_frames", None)
        if max_frames is not None and semantic_ids.shape[1] > int(max_frames):
            raise ValueError(
                f"semanticlength{semantic_ids.shape[1]}exceedsDiT max_frames={max_frames}"
            )

        if semantic_ids.numel() and (
            int(semantic_ids.min()) < 0 or int(semantic_ids.max()) >= 32_768
        ):
            raise ValueError("semantic id(contains padding)must be located at 0..32767")
        if tokenizer_revision is None:
            if self.require_input_tokenizer_revision:
                raise RuntimeError("semantic input is missing tokenizer revision")
        elif str(tokenizer_revision) != self.revisions.tokenizer_revision:
            raise RuntimeError(
                "semantic tokenizer revision does not match:"
                f"expected={self.revisions.tokenizer_revision} "
                f"actual={tokenizer_revision}"
            )
        if duration_seconds is not None:
            duration = torch.as_tensor(
                duration_seconds,
                dtype=torch.float64,
                device=self.device,
            )
            if duration.ndim == 0:
                duration = duration.expand(semantic_ids.shape[0])
            if duration.shape != (semantic_ids.shape[0],):
                raise ValueError("duration_seconds must be scalar or [B]")
            if not torch.isfinite(duration).all() or bool((duration <= 0).any()):
                raise ValueError("duration_seconds required finite and >0")
            frame_positions = duration * LATENT_FRAME_HZ
            rounded = torch.round(frame_positions)
            if not torch.allclose(
                frame_positions,
                rounded,
                rtol=0.0,
                atol=1.0e-5,
            ):
                raise ValueError(
                    "duration_secondsmust strictly fall within25HzTime Grid,"
                    "cannot rely onroundTolerate half-frame deviation"
                )
            asserted = rounded.long()
            if not torch.equal(asserted, lengths):
                raise ValueError(
                    "semantic length is the source of truth;duration_seconds is inconsistent with its assertion"
                )
        return semantic_ids, semantic_mask, lengths

    def _conditioning_pair(
        self,
        semantic_ids: torch.Tensor,
        semantic_mask: torch.Tensor,
        description_ids: torch.Tensor,
        description_mask: torch.Tensor,
        lyrics_ids: torch.Tensor,
        lyrics_mask: torch.Tensor,
    ) -> tuple[ConditioningOutput, ConditioningOutput]:
        conditional = self.conditioner(
            semantic_ids,
            semantic_mask,
            description_input_ids=description_ids,
            description_mask=description_mask,
            lyrics_input_ids=lyrics_ids,
            lyrics_mask=lyrics_mask,
        )
        null = self.conditioner.null_from_semantic(
            conditional.semantic_embeddings,
            conditional.semantic_mask,
            text_mask=conditional.text_mask,
            relative_dynamics_embeddings=(
                conditional.relative_dynamics_embeddings
            ),
        )

        if not torch.equal(
            conditional.semantic_embeddings, null.semantic_embeddings
        ) or not torch.equal(conditional.semantic_mask, null.semantic_mask):
            raise AssertionError("CFG null branch incorrectly changed semantic condition")
        if (
            conditional.relative_dynamics_embeddings is None
        ) != (null.relative_dynamics_embeddings is None) or (
            conditional.relative_dynamics_embeddings is not None
            and not torch.equal(
                conditional.relative_dynamics_embeddings,
                null.relative_dynamics_embeddings,
            )
        ):
            raise AssertionError("CFG null branch incorrectly changed relative dynamics condition")
        model_parameter = next(self.model.parameters(), None)
        if model_parameter is not None:
            target_device = model_parameter.device
            target_dtype = model_parameter.dtype

            def convert(value: ConditioningOutput) -> ConditioningOutput:
                return ConditioningOutput(
                    semantic_embeddings=value.semantic_embeddings.to(
                        device=target_device, dtype=target_dtype
                    ),
                    semantic_mask=value.semantic_mask.to(device=target_device),
                    text_context=value.text_context.to(
                        device=target_device, dtype=target_dtype
                    ),
                    text_mask=value.text_mask.to(device=target_device),
                    text_drop_mask=value.text_drop_mask.to(device=target_device),
                    provenance=dict(value.provenance),
                    relative_dynamics_embeddings=(
                        value.relative_dynamics_embeddings.to(
                            device=target_device,
                            dtype=target_dtype,
                        )
                        if value.relative_dynamics_embeddings is not None
                        else None
                    ),
                )

            conditional = convert(conditional)
            null = convert(null)
        return conditional, null

    def _decode_spectrum(
        self,
        latent_output: LatentOutput,
        *,
        audio_lengths: torch.Tensor,
    ) -> torch.Tensor | STFTOutput:
        decoder = getattr(self.spec_decoder, "decode", self.spec_decoder)
        raw_latents = destandardize_latents(
            latent_output.latents,
            stats=self.latent_stats,
        )
        if not isinstance(raw_latents, torch.Tensor):
            raise TypeError("latentDenormalization must returnTensor")
        raw_latents = raw_latents.masked_fill(~latent_output.mask.unsqueeze(-1), 0.0)
        decoded = _call_with_supported_kwargs(
            decoder,
            raw_latents,
            latent_mask=latent_output.mask,
            latent_lengths=latent_output.lengths,
            spectrum_frames=latent_output.latents.shape[1]
            * SPEC_FRAMES_PER_LATENT_FRAME,
            audio_lengths=audio_lengths,
        )
        if not isinstance(decoded, (torch.Tensor, STFTOutput)):
            raise TypeError("Spec decoder must return complex tensor or STFTOutput")
        spectrum = _extract_spectrum(decoded)
        if not torch.isfinite(spectrum).all():
            raise FloatingPointError("Spec decoderoutput containsNaN/Inf")
        expected_frames = latent_output.lengths * SPEC_FRAMES_PER_LATENT_FRAME
        if (
            spectrum.ndim != 4
            or spectrum.shape[:3] != (latent_output.latents.shape[0], 2, STFT_BINS)
            or not spectrum.is_complex()
        ):
            raise ValueError(
                "Spec decoder output must be identical tolatent batchconsistent complex [B,2,480,T]"
            )
        if spectrum.device != raw_latents.device:
            raise ValueError("Spec decoderoutput must matchlatentis located in the samedevice")
        expected_width = latent_output.latents.shape[1] * SPEC_FRAMES_PER_LATENT_FRAME
        if spectrum.shape[-1] != expected_width:
            raise ValueError(
                "Spec decoder spectrum width must be the same as padded latent strictly corresponds to:"
                f"expected={expected_width} actual={spectrum.shape[-1]}"
            )
        if isinstance(decoded, STFTOutput):
            if not torch.equal(
                decoded.spectrum_lengths.to(expected_frames.device),
                expected_frames,
            ) or not torch.equal(
                decoded.audio_lengths.to(audio_lengths.device), audio_lengths
            ):
                raise ValueError(
                    "Spec decoder STFTOutput length metadata and semantic Contract inconsistent"
                )
            return decoded
        spectrum_mask = lengths_to_mask(
            expected_frames,
            spectrum.shape[-1],
            device=spectrum.device,
        )
        return STFTOutput(
            spectrum=spectrum.masked_fill(~spectrum_mask[:, None, None, :], 0.0),
            audio_lengths=audio_lengths,
            spectrum_lengths=expected_frames,
            spectrum_mask=spectrum_mask,
        )

    def _refine(
        self,
        spectrum: torch.Tensor | STFTOutput,
        *,
        use_refiner: bool,
    ) -> torch.Tensor | STFTOutput:
        if not use_refiner:
            return spectrum
        refiner = getattr(self.refiner, "refine", self.refiner)
        coarse_values = _extract_spectrum(spectrum)
        if isinstance(spectrum, STFTOutput):
            lengths = spectrum.spectrum_lengths
            padded_frames = coarse_values.shape[-1]
            signature = _callable_signature(refiner)
            accepts_mask = signature is not None and (
                "spectrum_mask" in signature.parameters
                or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
            )
            if not accepts_mask and not bool((lengths == padded_frames).all()):


                rows: list[torch.Tensor] = []
                for index, length_value in enumerate(lengths.tolist()):
                    length = int(length_value)
                    item = refiner(coarse_values[index : index + 1, ..., :length])
                    item_values = _extract_spectrum(item)
                    if (
                        not isinstance(item_values, torch.Tensor)
                        or item_values.shape != (1, 2, STFT_BINS, length)
                        or not item_values.is_complex()
                        or item_values.device != coarse_values.device
                        or not torch.isfinite(item_values).all()
                    ):
                        raise ValueError(
                            "Nonemask Refinershape/dtype/deviceIllegal"
                        )
                    rows.append(item_values)
                padded = torch.zeros_like(coarse_values)
                for index, item in enumerate(rows):
                    padded[index : index + 1, ..., : item.shape[-1]] = item
                return STFTOutput(
                    spectrum=padded,
                    audio_lengths=spectrum.audio_lengths,
                    spectrum_lengths=spectrum.spectrum_lengths,
                    spectrum_mask=spectrum.spectrum_mask,
                    waveform_dtype=spectrum.waveform_dtype,
                )
        refined = _call_with_supported_kwargs(
            refiner,
            coarse_values,
            spectrum_mask=(
                spectrum.spectrum_mask if isinstance(spectrum, STFTOutput) else None
            ),
        )
        if isinstance(refined, STFTOutput):
            if isinstance(spectrum, STFTOutput) and (
                not torch.equal(refined.audio_lengths, spectrum.audio_lengths)
                or not torch.equal(refined.spectrum_lengths, spectrum.spectrum_lengths)
                or refined.spectrum.shape != spectrum.spectrum.shape
            ):
                raise ValueError("Refiner must not be changed spectrum/audio length contract")
            if not torch.isfinite(refined.spectrum).all():
                raise FloatingPointError("Refineroutput containsNaN/Inf")
            return refined
        if not isinstance(refined, torch.Tensor):
            raise TypeError("Refiner must return complex tensor or STFTOutput")
        if not torch.isfinite(refined).all():
            raise FloatingPointError("Refineroutput containsNaN/Inf")
        base = spectrum
        if isinstance(base, STFTOutput):
            if refined.shape != base.spectrum.shape or not refined.is_complex():
                raise ValueError("Refiner output spectrum shape/dtype change")
            return STFTOutput(
                spectrum=refined,
                audio_lengths=base.audio_lengths,
                spectrum_lengths=base.spectrum_lengths,
                spectrum_mask=base.spectrum_mask,
                waveform_dtype=base.waveform_dtype,
            )
        if refined.shape != base.shape or not refined.is_complex():
            raise ValueError("Refiner output spectrum shape/dtype change")
        return refined

    def _synthesize(
        self,
        spectrum: torch.Tensor | STFTOutput,
        audio_lengths: torch.Tensor,
    ) -> torch.Tensor:
        inverse = getattr(
            self.inverse_stft,
            "inverse",
            getattr(self.inverse_stft, "synthesize", self.inverse_stft),
        )
        waveform = _call_with_supported_kwargs(
            inverse,
            spectrum,
            lengths=audio_lengths,
            audio_lengths=audio_lengths,
            dtype=torch.float32,
        )
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("iSTFT must return waveform tensor")
        if waveform.is_complex() or not waveform.is_floating_point():
            raise TypeError("iSTFT must return real floating point waveform tensor")
        spectrum_values = _extract_spectrum(spectrum)
        if waveform.device != spectrum_values.device:
            raise ValueError("iSTFTThe output must be on the same spectrum as the inputdevice")
        if waveform.ndim != 3 or waveform.shape[:2] != (
            audio_lengths.shape[0],
            2,
        ):
            raise ValueError("iSTFT output must be [B,2,N]")
        expected_max = int(audio_lengths.max())
        if waveform.shape[-1] != expected_max:
            raise ValueError(
                "iSTFT The output length must be determined by semantic frame Accurate decision:"
                f"expected={expected_max} actual={waveform.shape[-1]}"
            )
        sample_mask = lengths_to_mask(
            audio_lengths, expected_max, device=waveform.device
        )
        return waveform.float().masked_fill(~sample_mask.unsqueeze(1), 0.0)

    @torch.no_grad()
    def render(
        self,
        semantic_ids: torch.Tensor,
        description: str | Sequence[str] | torch.Tensor,
        lyrics: str | Sequence[str] | torch.Tensor,
        *,
        semantic_mask: torch.Tensor | None = None,
        description_mask: torch.Tensor | None = None,
        lyrics_mask: torch.Tensor | None = None,
        duration_seconds: float | Sequence[float] | torch.Tensor | None = None,
        tokenizer_revision: str | None = None,
        text_tokenizer_revision: str | None = None,
        rewriter_revision: str | None = None,
        rewriter_schema_version: str | None = None,
        global_loudness_lufs: float
        | Sequence[float]
        | torch.Tensor
        | None = None,
        seed: int,
        num_steps: int,
        cfg_scale: float,
        solver: str,
        use_refiner: bool = True,
        input_artifact_identities: Mapping[str, Any] | None = None,
    ) -> RenderOutput:

        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be int")
        if not -(2**63) <= seed <= 2**64 - 1:
            raise ValueError("seedexceedstorch.Generator.manual_seedSupport range")
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, int)
            or num_steps <= 0
        ):
            raise ValueError("num_stepsmust be a positive integer")
        if solver not in SUPPORTED_SOLVERS:
            raise ValueError(f"solvermust be{SUPPORTED_SOLVERS}")
        if (
            isinstance(cfg_scale, bool)
            or not isinstance(cfg_scale, (int, float))
            or not math_is_finite(cfg_scale)
            or cfg_scale < 0
        ):
            raise ValueError("cfg_scale required finite and >=0")
        if not isinstance(use_refiner, bool):
            raise TypeError("use_refinermust bebool")
        self._validate_model_precision()
        if rewriter_revision is None:
            raise RuntimeError("description/lyricsinput is missingrewriter revision")
        if str(rewriter_revision) != self.revisions.rewriter_revision:
            raise RuntimeError(
                "rewriter revisiondoes not match:"
                f"expected={self.revisions.rewriter_revision} "
                f"actual={rewriter_revision}"
            )
        if rewriter_schema_version != REWRITER_SCHEMA_VERSION:
            raise RuntimeError(
                "rewriter schemadoes not match:"
                f"expected={REWRITER_SCHEMA_VERSION} "
                f"actual={rewriter_schema_version!r}"
            )
        if text_tokenizer_revision is not None and (
            str(text_tokenizer_revision) != self.revisions.text_tokenizer_revision
        ):
            raise RuntimeError(
                "text tokenizer revisiondoes not match:"
                f"expected={self.revisions.text_tokenizer_revision} "
                f"actual={text_tokenizer_revision}"
            )
        if isinstance(description, torch.Tensor) or isinstance(lyrics, torch.Tensor):
            if text_tokenizer_revision is None:
                raise RuntimeError("TensorText input is missingtext tokenizer revision")
        semantic_ids, semantic_mask, lengths = self._validate_semantic(
            semantic_ids,
            semantic_mask,
            tokenizer_revision=tokenizer_revision,
            duration_seconds=duration_seconds,
        )
        batch_size = semantic_ids.shape[0]
        description_ids, description_mask = _normalize_text_input(
            description,
            description_mask,
            batch_size=batch_size,
            name="description",
            conditioner=self.conditioner,
            device=self.device,
        )
        lyrics_ids, lyrics_mask = _normalize_text_input(
            lyrics,
            lyrics_mask,
            batch_size=batch_size,
            name="lyrics",
            conditioner=self.conditioner,
            device=self.device,
        )
        try:
            external_input_identities = json.loads(
                json.dumps(
                    dict(input_artifact_identities or {}),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("input_artifact_identitiesmust be finiteJSON mapping") from exc
        semantic_identity = {
            "ids": _tensor_identity(semantic_ids),
            "mask": _tensor_identity(semantic_mask),
            "combined_sha256": _sha256_json(
                {
                    "ids": _tensor_identity(semantic_ids),
                    "mask": _tensor_identity(semantic_mask),
                }
            ),
            "tokenizer_revision": tokenizer_revision,
        }
        description_identity = {
            "raw": _raw_input_identity(description),
            "token_ids": _tensor_identity(description_ids),
            "mask": _tensor_identity(description_mask),
        }
        description_identity["combined_sha256"] = _sha256_json(description_identity)
        lyrics_identity = {
            "raw": _raw_input_identity(lyrics),
            "token_ids": _tensor_identity(lyrics_ids),
            "mask": _tensor_identity(lyrics_mask),
        }
        lyrics_identity["combined_sha256"] = _sha256_json(lyrics_identity)
        self.model.eval()
        self.conditioner.eval()
        if isinstance(self.spec_decoder, nn.Module):
            self.spec_decoder.eval()
        if isinstance(self.refiner, nn.Module):
            self.refiner.eval()
        if isinstance(self.inverse_stft, nn.Module):
            self.inverse_stft.eval()
        conditional, null = self._conditioning_pair(
            semantic_ids,
            semantic_mask,
            description_ids,
            description_mask,
            lyrics_ids,
            lyrics_mask,
        )
        global_loudness_tensor = (
            torch.as_tensor(
                global_loudness_lufs,
                dtype=torch.float32,
                device=self.device,
            )
            if global_loudness_lufs is not None
            else None
        )
        if global_loudness_tensor is not None:
            if global_loudness_tensor.ndim == 0:
                global_loudness_tensor = global_loudness_tensor.expand(
                    batch_size
                )
            if (
                global_loudness_tensor.shape != (batch_size,)
                or not torch.isfinite(global_loudness_tensor).all()
            ):
                raise ValueError(
                    "global_loudness_lufsmust befinite scalaror[B]"
                )
        global_loudness_embedding = self.conditioner.embed_global_loudness(
            global_loudness_tensor
        )
        model_loudness_kwargs = (
            {"global_loudness_embedding": global_loudness_embedding}
            if global_loudness_embedding is not None
            else {}
        )
        initial_reference = torch.empty(
            batch_size,
            semantic_ids.shape[1],
            LATENT_DIM,
            device=self.device,
            dtype=torch.float32,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial = sample_source_like(
            initial_reference, self.flow_config, generator=generator
        )
        initial = initial.masked_fill(~semantic_mask.unsqueeze(-1), 0.0)

        def guided_velocity(
            latent: torch.Tensor, timestep: torch.Tensor
        ) -> torch.Tensor:

            model_dtype = next(self.model.parameters()).dtype
            model_latent = latent.to(dtype=model_dtype)
            conditional_velocity = self.model(
                model_latent,
                timestep,
                conditioning=conditional,
                frame_mask=semantic_mask,
                **model_loudness_kwargs,
            )
            null_velocity = self.model(
                model_latent,
                timestep,
                conditioning=null,
                frame_mask=semantic_mask,
                **model_loudness_kwargs,
            )
            return classifier_free_guidance(
                conditional_velocity.float(),
                null_velocity.float(),
                cfg_scale,
            )

        t_start, t_end = sampling_interval(self.flow_config)
        schedule = sampling_schedule(
            self.flow_config,
            num_steps=num_steps,
            reference=initial,
            effective_lengths=lengths,
        )
        solved = solve_flow(
            guided_velocity,
            initial,
            solver=solver,
            num_steps=num_steps,
            t_start=t_start,
            t_end=t_end,
            frame_mask=semantic_mask,
            schedule=schedule,
        )
        latent_output = LatentOutput(
            latents=solved.sample.masked_fill(~semantic_mask.unsqueeze(-1), 0.0),
            lengths=lengths,
            mask=semantic_mask,
        )
        audio_lengths = lengths * SAMPLES_PER_LATENT_FRAME
        coarse = self._decode_spectrum(latent_output, audio_lengths=audio_lengths)
        refined = self._refine(coarse, use_refiner=use_refiner)
        waveform = self._synthesize(refined, audio_lengths)
        output = RenderOutput(
            waveform=waveform,
            sample_rate=SAMPLE_RATE,
            audio_lengths=audio_lengths,
            latent_frames=(
                int(lengths[0])
                if bool((lengths == lengths[0]).all())
                else lengths.detach().clone()
            ),
            seed=seed,
            num_steps=int(num_steps),
            cfg_scale=float(cfg_scale),
            solver=solver,
            checkpoint_revision=self.revisions.checkpoint_revision,
        )

        output.nfe = solved.nfe
        output.dit_forward_evaluations = 2 * solved.nfe
        output.revisions = asdict(self.revisions)
        effective_preset = dict(self.inference_preset)
        effective_preset.update(
            {
                "solver": solver,
                "num_steps": int(num_steps),
                "nfe": solved.nfe,
                "cfg_scale": float(cfg_scale),
                "cfg_rescale": False,
                "use_refiner": bool(use_refiner),
                "precision": self.precision,
            }
        )
        output.metadata = {
            "semantic_frame_hz": LATENT_FRAME_HZ,
            "latent_lengths": lengths.detach().cpu().tolist(),
            "audio_lengths": audio_lengths.detach().cpu().tolist(),
            "seed": seed,
            "solver": solver,
            "num_steps": int(num_steps),
            "nfe": solved.nfe,
            "dit_forward_evaluations": 2 * solved.nfe,
            "cfg_scale": float(cfg_scale),
            "cfg_rescale": False,
            "use_refiner": bool(use_refiner),
            "flow_config": self.flow_config.to_dict(),
            "t_start": solved.t_start,
            "t_end": solved.t_end,
            "seed_scope": "batch",
            "precision": self.precision,
            "artifact_identities": dict(self.artifact_identities),

            "inference_preset": dict(self.inference_preset),
            "inference_preset_sha256": self.inference_preset_sha256,
            "base_inference_preset": dict(self.inference_preset),
            "base_inference_preset_sha256": self.inference_preset_sha256,
            "effective_inference_preset": effective_preset,
            "effective_inference_preset_sha256": config_hash(effective_preset),
            "input_identities": {
                "semantic": semantic_identity,
                "description": description_identity,
                "lyrics": lyrics_identity,
                "artifacts": external_input_identities,
                "global_loudness_lufs": (
                    global_loudness_tensor.detach().cpu().tolist()
                    if global_loudness_tensor is not None
                    else None
                ),
            },
            "revisions": asdict(self.revisions),
        }
        return output


RenderPipeline = RenderInferencePipeline


def math_is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def render(
    pipeline: RenderInferencePipeline,
    semantic_ids: torch.Tensor,
    description: str | Sequence[str] | torch.Tensor,
    lyrics: str | Sequence[str] | torch.Tensor,
    **kwargs: Any,
) -> RenderOutput:

    return pipeline.render(semantic_ids, description, lyrics, **kwargs)
