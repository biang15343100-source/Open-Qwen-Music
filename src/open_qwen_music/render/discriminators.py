from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .contracts import (
    AUDIO_CHANNELS,
    SAMPLE_RATE,
    validate_audio,
)


DEFAULT_STFT_SCALES = (
    (256, 64, 256),
    (384, 96, 384),
    (512, 128, 512),
    (768, 192, 768),
    (1024, 256, 1024),
    (1536, 384, 1536),
    (2048, 512, 2048),
    (4096, 1024, 4096),
)
SPECTROSTREAM_PUBLIC_DISCRIMINATOR_SCALES = (
    (128, 64, 128),
    (256, 128, 256),
    (512, 256, 512),
    (1024, 512, 1024),
    (2048, 1024, 2048),
    (4096, 2048, 4096),
)


@dataclass(frozen=True)
class SpectroStreamFeatureBandViewSpec:
    scale_index: int
    n_fft: int
    layer_index: int
    band_index: int
    requested_hz: tuple[float, float]
    realized_center_hz: tuple[float, float]
    receptive_field_hz: float
    feature_frequency_bins: int
    start_bin: int
    stop_bin: int


@dataclass(frozen=True)
class SpectroStreamFeatureBandView:
    spec: SpectroStreamFeatureBandViewSpec
    tensor: Tensor

    def detached(self) -> "SpectroStreamFeatureBandView":
        return SpectroStreamFeatureBandView(
            spec=self.spec,
            tensor=self.tensor.detach(),
        )


@dataclass
class DiscriminatorOutput:
    logits: list[Tensor]
    features: list[list[Tensor]]
    highband_features: list[list[Tensor]] | None = None
    highband_feature_views: list[SpectroStreamFeatureBandView] | None = None
    family_names: tuple[str, ...] | None = None
    family_sizes: tuple[int, ...] | None = None
    loss_reduction: str = "flat_feature_mean_v1"
    loss_compute_dtype: str = "float32"
    family_diagnostics_enabled: bool | None = None
    family_weights: tuple[float, ...] | None = None

    def detached(self) -> "DiscriminatorOutput":
        return DiscriminatorOutput(
            logits=[value.detach() for value in self.logits],
            features=[[value.detach() for value in scale] for scale in self.features],
            highband_features=(
                None
                if self.highband_features is None
                else [[value.detach() for value in scale] for scale in self.highband_features]
            ),
            highband_feature_views=(
                None
                if self.highband_feature_views is None
                else [value.detached() for value in self.highband_feature_views]
            ),
            family_names=self.family_names,
            family_sizes=self.family_sizes,
            family_weights=self.family_weights,
            loss_reduction=self.loss_reduction,
            loss_compute_dtype=self.loss_compute_dtype,
            family_diagnostics_enabled=self.family_diagnostics_enabled,
        )


def spectrostream_feature_frequency_centers_hz(
    *,
    input_frequency_bins: int,
    layer_index: int,
    sample_rate: int = SAMPLE_RATE,
) -> Tensor:

    if input_frequency_bins <= 0:
        raise ValueError("input_frequency_bins must be positive")
    if layer_index < 0 or layer_index >= 6:
        raise ValueError("layer_index must be between 0 and 5")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    stride = 2 ** (layer_index + 1)
    output_bins = (input_frequency_bins + stride - 1) // stride
    centers = (torch.arange(output_bins, dtype=torch.float64) + 0.5) * stride - 0.5
    return centers * (float(sample_rate) / (2.0 * input_frequency_bins))


def spectrostream_feature_band_slice(
    *,
    input_frequency_bins: int,
    layer_index: int,
    lower_hz: float,
    upper_hz: float,
    minimum_bins: int = 1,
    require_receptive_field_containment: bool = False,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[int, int, tuple[float, float]] | None:

    if (
        not math.isfinite(lower_hz)
        or not math.isfinite(upper_hz)
        or not 0.0 <= lower_hz < upper_hz <= sample_rate / 2
    ):
        raise ValueError("The feature band must lie within the Nyquist range")
    if minimum_bins <= 0:
        raise ValueError("minimum_bins must be positive")
    centers = spectrostream_feature_frequency_centers_hz(
        input_frequency_bins=input_frequency_bins,
        layer_index=layer_index,
        sample_rate=sample_rate,
    )
    if require_receptive_field_containment:
        half_receptive_field_hz = 0.5 * spectrostream_feature_receptive_field_hz(
            input_frequency_bins=input_frequency_bins,
            layer_index=layer_index,
            sample_rate=sample_rate,
        )
        selected_mask = (centers - half_receptive_field_hz >= float(lower_hz)) & (
            centers + half_receptive_field_hz < float(upper_hz)
        )
    else:
        selected_mask = (centers >= float(lower_hz)) & (centers < float(upper_hz))
    selected = torch.nonzero(selected_mask, as_tuple=False).flatten()
    if int(selected.numel()) < minimum_bins:
        return None
    start = int(selected[0])
    stop = int(selected[-1]) + 1
    if stop - start != int(selected.numel()):
        raise RuntimeError("The selected feature band is not contiguous")
    return start, stop, (float(centers[start]), float(centers[stop - 1]))


def spectrostream_feature_receptive_field_hz(
    *,
    input_frequency_bins: int,
    layer_index: int,
    sample_rate: int = SAMPLE_RATE,
) -> float:

    if input_frequency_bins <= 0:
        raise ValueError("input_frequency_bins must be positive")
    if layer_index < 0 or layer_index >= 6:
        raise ValueError("layer_index must be between 0 and 5")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    receptive_field_bins = 5 * (2 ** (layer_index + 1)) + 2
    return receptive_field_bins * (float(sample_rate) / (2.0 * input_frequency_bins))


def spectrostream_feature_band_layout(
    *,
    scales: Sequence[Sequence[int]],
    feature_bands_hz: Sequence[Sequence[float]],
    feature_band_layers: Sequence[int],
    feature_band_minimum_bins: int,
    feature_band_require_complete_layer: bool,
    feature_band_max_receptive_field_hz: float | None,
) -> tuple[tuple[SpectroStreamFeatureBandViewSpec, ...], ...]:
    resolved_scales = tuple(tuple(int(value) for value in scale) for scale in scales)
    resolved_bands = tuple((float(lower), float(upper)) for lower, upper in feature_bands_hz)
    resolved_layers = tuple(int(value) for value in feature_band_layers)
    if not resolved_scales or any(len(scale) != 3 for scale in resolved_scales):
        raise ValueError("STFT scales must contain (n_fft, hop, win) triples")
    nyquist_hz = SAMPLE_RATE / 2
    if any(not 0.0 <= lower < upper <= nyquist_hz for lower, upper in resolved_bands):
        raise ValueError("Feature bands must lie within the Nyquist range")
    if (
        not resolved_layers
        or any(value < 0 or value >= 6 for value in resolved_layers)
        or len(set(resolved_layers)) != len(resolved_layers)
    ):
        raise ValueError("Feature band layers must be unique indices from 0 through 5")
    if feature_band_minimum_bins <= 0:
        raise ValueError("feature_band_minimum_bins must be positive")
    if feature_band_max_receptive_field_hz is not None and (
        not math.isfinite(feature_band_max_receptive_field_hz)
        or feature_band_max_receptive_field_hz <= 0.0
    ):
        raise ValueError("The maximum receptive field must be finite and positive")

    layouts: list[tuple[SpectroStreamFeatureBandViewSpec, ...]] = []
    complete_layer_counts = {layer: 0 for layer in resolved_layers}
    band_counts = [0 for _ in resolved_bands]
    for scale_index, (n_fft, _hop, _win) in enumerate(resolved_scales):
        input_frequency_bins = n_fft // 2
        scale_specs: list[SpectroStreamFeatureBandViewSpec] = []
        for layer_index in resolved_layers:
            layer_specs: list[SpectroStreamFeatureBandViewSpec] = []
            receptive_field_hz = spectrostream_feature_receptive_field_hz(
                input_frequency_bins=input_frequency_bins,
                layer_index=layer_index,
            )
            if (
                feature_band_max_receptive_field_hz is not None
                and receptive_field_hz > feature_band_max_receptive_field_hz
            ):
                continue
            centers = spectrostream_feature_frequency_centers_hz(
                input_frequency_bins=input_frequency_bins,
                layer_index=layer_index,
            )
            for band_index, (lower_hz, upper_hz) in enumerate(resolved_bands):
                resolved = spectrostream_feature_band_slice(
                    input_frequency_bins=input_frequency_bins,
                    layer_index=layer_index,
                    lower_hz=lower_hz,
                    upper_hz=upper_hz,
                    minimum_bins=feature_band_minimum_bins,
                    require_receptive_field_containment=True,
                )
                if resolved is None:
                    continue
                first, last, realized = resolved
                layer_specs.append(
                    SpectroStreamFeatureBandViewSpec(
                        scale_index=scale_index,
                        n_fft=n_fft,
                        layer_index=layer_index,
                        band_index=band_index,
                        requested_hz=(lower_hz, upper_hz),
                        realized_center_hz=realized,
                        receptive_field_hz=receptive_field_hz,
                        feature_frequency_bins=int(centers.numel()),
                        start_bin=first,
                        stop_bin=last,
                    )
                )
            if feature_band_require_complete_layer and len(layer_specs) != len(resolved_bands):
                layer_specs = []
            if layer_specs:
                if len(layer_specs) == len(resolved_bands):
                    complete_layer_counts[layer_index] += 1
                for spec in layer_specs:
                    band_counts[spec.band_index] += 1
                scale_specs.extend(layer_specs)
        layouts.append(tuple(scale_specs))
    if resolved_bands:
        if not any(layouts):
            raise ValueError("No discriminator feature view covers the requested bands")
        if feature_band_require_complete_layer and any(
            count == 0 for count in complete_layer_counts.values()
        ):
            raise ValueError("A requested layer does not cover every feature band")
        if any(count == 0 for count in band_counts):
            raise ValueError("A requested feature band has no available view")
    return tuple(layouts)


class PatchDiscriminator2D(nn.Module):
    def __init__(
        self,
        input_channels: int = 4,
        *,
        base_channels: int = 16,
        max_channels: int = 256,
        depth: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_channels
        for index in range(depth):
            output = min(max_channels, base_channels * (2**index))
            layers.append(
                nn.Conv2d(
                    current,
                    output,
                    kernel_size=(3, 5),
                    stride=(1 if index == 0 else 2, 2),
                    padding=(1, 2),
                )
            )
            current = output
        self.layers = nn.ModuleList(layers)
        self.output = nn.Conv2d(current, 1, kernel_size=3, padding=1)

    def forward(self, inputs: Tensor) -> tuple[Tensor, list[Tensor]]:
        features: list[Tensor] = []
        outputs = inputs.float()
        for layer in self.layers:
            outputs = F.leaky_relu(layer(outputs), negative_slope=0.2)
            features.append(outputs)
        logits = self.output(outputs)
        return logits, features


class GlobalLayerNorm2D(nn.Module):
    def __init__(self, channels: int, *, epsilon: float = 1.0e-5) -> None:
        super().__init__()
        if channels <= 0 or epsilon <= 0:
            raise ValueError("channels and epsilon must be positive")
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.epsilon = float(epsilon)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.weight.shape[1]:
            raise ValueError("GlobalLayerNorm2D input shape does not match its channels")
        work = inputs.float()
        mean = work.mean(dim=(1, 2, 3), keepdim=True)
        variance = (work - mean).square().mean(dim=(1, 2, 3), keepdim=True)
        return (work - mean) * torch.rsqrt(variance + self.epsilon) * self.weight + self.bias


def _same_pad_2d(
    inputs: Tensor,
    *,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
) -> Tensor:
    pads: list[tuple[int, int]] = []
    for size, kernel, current_stride in zip(inputs.shape[-2:], kernel_size, stride):
        output = (int(size) + current_stride - 1) // current_stride
        total = max(0, (output - 1) * current_stride + kernel - int(size))
        pads.append((total // 2, total - total // 2))
    return F.pad(inputs, (pads[1][0], pads[1][1], pads[0][0], pads[0][1]))


class SpectroStreamDiscriminatorBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        stride: tuple[int, int],
    ) -> None:
        super().__init__()
        self.stride = tuple(int(value) for value in stride)
        self.norm1 = GlobalLayerNorm2D(input_channels)
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3)
        self.norm2 = GlobalLayerNorm2D(input_channels)
        kernel = tuple(max(3, 2 * value) for value in self.stride)
        self.conv2 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=kernel,
            stride=self.stride,
        )
        self.projection = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        shortcut = F.avg_pool2d(
            inputs,
            kernel_size=self.stride,
            stride=self.stride,
            ceil_mode=True,
        )
        shortcut = self.projection(shortcut)
        outputs = F.leaky_relu(self.norm1(inputs), negative_slope=0.2)
        outputs = self.conv1(_same_pad_2d(outputs, kernel_size=(3, 3), stride=(1, 1)))
        outputs = F.leaky_relu(self.norm2(outputs), negative_slope=0.2)
        outputs = self.conv2(
            _same_pad_2d(
                outputs,
                kernel_size=self.conv2.kernel_size,
                stride=self.stride,
            )
        )
        if outputs.shape[-2:] != shortcut.shape[-2:]:
            raise RuntimeError("SpectroStream residual paths produced different shapes")
        return outputs + shortcut


class SpectroStreamPatchDiscriminator2D(nn.Module):
    def __init__(self, frequency_bins: int, *, base_channels: int = 32) -> None:
        super().__init__()
        if frequency_bins <= 0:
            raise ValueError("frequency_bins must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        c0 = int(base_channels)
        self.input_projection = nn.Conv2d(3, c0, kernel_size=7, padding=3)
        self.pre_fusion_blocks = nn.ModuleList(
            [
                SpectroStreamDiscriminatorBlock(c0, 2 * c0, stride=(1, 2)),
                SpectroStreamDiscriminatorBlock(2 * c0, 4 * c0, stride=(2, 2)),
                SpectroStreamDiscriminatorBlock(4 * c0, 4 * c0, stride=(1, 2)),
                SpectroStreamDiscriminatorBlock(4 * c0, 8 * c0, stride=(2, 2)),
            ]
        )
        self.post_fusion_blocks = nn.ModuleList(
            [
                SpectroStreamDiscriminatorBlock(16 * c0, 8 * c0, stride=(1, 2)),
                SpectroStreamDiscriminatorBlock(8 * c0, 16 * c0, stride=(2, 2)),
            ]
        )
        final_frequency_bins = (frequency_bins + 63) // 64
        self.output = nn.Conv2d(
            16 * c0,
            1,
            kernel_size=(1, final_frequency_bins),
            stride=(1, final_frequency_bins),
        )

    def forward(self, spectrum: Tensor) -> tuple[Tensor, list[Tensor]]:
        if spectrum.ndim != 4 or spectrum.shape[1] != AUDIO_CHANNELS or not spectrum.is_complex():
            raise ValueError("SpectroStream discriminator input must be complex [B, 2, F, T]")
        batch, stereo, frequencies, frames = spectrum.shape
        values = torch.stack(
            (spectrum.real.float(), spectrum.imag.float(), spectrum.abs().float()),
            dim=2,
        ).permute(0, 1, 2, 4, 3)
        outputs = self.input_projection(values.reshape(batch * stereo, 3, frames, frequencies))
        features: list[Tensor] = []
        for block in self.pre_fusion_blocks:
            outputs = block(outputs)
            features.append(
                outputs.reshape(
                    batch,
                    stereo * outputs.shape[1],
                    outputs.shape[2],
                    outputs.shape[3],
                )
            )
        outputs = outputs.reshape(
            batch,
            stereo * outputs.shape[1],
            outputs.shape[2],
            outputs.shape[3],
        )
        for block in self.post_fusion_blocks:
            outputs = block(outputs)
            features.append(outputs)
        logits = self.output(outputs)
        if logits.shape[-1] != 1:
            raise RuntimeError(
                "SpectroStreamDiscriminator does not collapse the complete frequency axis"
            )
        return logits, features


def _complex_channels(spectrum: Tensor) -> Tensor:
    if not spectrum.is_complex():
        raise ValueError("The discriminator spectrum input must be complex")
    values = torch.view_as_real(spectrum.to(torch.complex64))
    return values.permute(0, 1, 4, 2, 3).flatten(1, 2)


def _stft(
    audio: Tensor,
    n_fft: int,
    hop: int,
    win: int,
    *,
    frontend_profile: str = "spectrostream_public_v1",
) -> Tensor:
    if audio.ndim != 3 or audio.shape[1] != AUDIO_CHANNELS:
        raise ValueError("Discriminator audio must have shape [B, 2, N]")
    if frontend_profile != "spectrostream_public_v1":
        raise ValueError("The STFT discriminator uses spectrostream_public_v1")
    values = audio.float()
    if values.shape[-1] < n_fft:
        raise ValueError(
            "The SpectroStream frontend does not pad short inputs; "
            "input length must be at least n_fft"
        )
    window = torch.hann_window(
        win,
        periodic=True,
        dtype=torch.float32,
        device=audio.device,
    )
    spectrum = torch.stft(
        values.flatten(0, 1),
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        center=False,
        return_complex=True,
    )
    return spectrum.reshape(audio.shape[0], AUDIO_CHANNELS, spectrum.shape[-2], spectrum.shape[-1])


class SpectroStreamMultiScaleSTFTDiscriminator(nn.Module):
    def __init__(
        self,
        scales: Sequence[Sequence[int]] = DEFAULT_STFT_SCALES,
        *,
        base_channels: int = 32,
        frontend_profile: str = "spectrostream_public_v1",
        feature_bands_hz: Sequence[Sequence[float]] = (),
        feature_band_layers: Sequence[int] = (0, 1, 2),
        feature_band_minimum_bins: int = 1,
        feature_band_require_complete_layer: bool = False,
        feature_band_max_receptive_field_hz: float | None = None,
    ) -> None:
        super().__init__()
        self.scales = tuple(tuple(int(value) for value in scale) for scale in scales)
        self.frontend_profile = str(frontend_profile)
        self.feature_bands_hz = tuple(
            (float(lower), float(upper)) for lower, upper in feature_bands_hz
        )
        self.feature_band_layers = tuple(int(value) for value in feature_band_layers)
        self.feature_band_minimum_bins = int(feature_band_minimum_bins)
        self.feature_band_require_complete_layer = bool(feature_band_require_complete_layer)
        self.feature_band_max_receptive_field_hz = (
            None
            if feature_band_max_receptive_field_hz is None
            else float(feature_band_max_receptive_field_hz)
        )
        if not self.scales or any(len(scale) != 3 for scale in self.scales):
            raise ValueError("STFT scales must be (n_fft,hop,win) list")
        if self.frontend_profile != "spectrostream_public_v1":
            raise ValueError("SpectroStream requires the public STFT frontend")
        self.feature_band_layout = spectrostream_feature_band_layout(
            scales=self.scales,
            feature_bands_hz=self.feature_bands_hz,
            feature_band_layers=self.feature_band_layers,
            feature_band_minimum_bins=self.feature_band_minimum_bins,
            feature_band_require_complete_layer=(self.feature_band_require_complete_layer),
            feature_band_max_receptive_field_hz=(self.feature_band_max_receptive_field_hz),
        )
        self.discriminators = nn.ModuleList(
            [
                SpectroStreamPatchDiscriminator2D(
                    n_fft // 2,
                    base_channels=base_channels,
                )
                for n_fft, _hop, _win in self.scales
            ]
        )

    def forward(
        self,
        audio: Tensor,
        lengths: Tensor | None = None,
    ) -> DiscriminatorOutput:
        lengths = validate_audio(audio, lengths)
        if bool((lengths != audio.shape[-1]).any()):
            raise ValueError(
                "SpectroStreamMultiScaleSTFTDiscriminator does not support ragged batches"
            )
        logits: list[Tensor] = []
        features: list[list[Tensor]] = []
        highband_features: list[list[Tensor]] = []
        highband_feature_views: list[SpectroStreamFeatureBandView] = []
        for scale_index, (scale, discriminator) in enumerate(
            zip(self.scales, self.discriminators, strict=True)
        ):
            spectrum = _stft(
                audio,
                *scale,
                frontend_profile=self.frontend_profile,
            )[..., :-1, :]
            current_logits, current_features = discriminator(spectrum)
            current_highband_features: list[Tensor] = []
            for spec in self.feature_band_layout[scale_index]:
                value = current_features[spec.layer_index]
                if int(value.shape[-1]) != spec.feature_frequency_bins:
                    raise RuntimeError("SpectroStream feature frequency shape changed")
                view = value[..., spec.start_bin : spec.stop_bin]
                current_highband_features.append(view)
                highband_feature_views.append(
                    SpectroStreamFeatureBandView(
                        spec=spec,
                        tensor=view,
                    )
                )
            logits.append(current_logits)
            features.append(current_features)
            highband_features.append(current_highband_features)
        return DiscriminatorOutput(
            logits,
            features,
            highband_features if self.feature_bands_hz else None,
            highband_feature_views if highband_feature_views else None,
        )


class CQTDiscriminator(nn.Module):
    def __init__(
        self,
        scales: Sequence[Mapping[str, Any]] = (),
        *,
        base_channels: int = 16,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.scales = tuple(dict(scale) for scale in scales)
        self.transforms = nn.ModuleList()
        if self.scales:
            try:
                from nnAudio.features import CQT2010v2
            except ImportError as exc:
                raise RuntimeError(
                    "The CQT discriminator requires the nnAudio render dependency"
                ) from exc
            for index, scale in enumerate(self.scales):
                required = {"hop_length", "fmin", "n_bins", "bins_per_octave"}
                if set(scale) != required:
                    raise ValueError(f"CQT scale {index} must contain exactly {sorted(required)}")
                self.transforms.append(
                    CQT2010v2(
                        sr=SAMPLE_RATE,
                        hop_length=int(scale["hop_length"]),
                        fmin=float(scale["fmin"]),
                        n_bins=int(scale["n_bins"]),
                        bins_per_octave=int(scale["bins_per_octave"]),
                        pad_mode="constant",
                        trainable=False,
                        output_format="Complex",
                        verbose=False,
                    )
                )
        self.discriminators = nn.ModuleList(
            [PatchDiscriminator2D(4, base_channels=base_channels, depth=depth) for _ in self.scales]
        )

    def forward(
        self,
        audio: Tensor,
        lengths: Tensor | None = None,
    ) -> DiscriminatorOutput:
        lengths = validate_audio(audio, lengths)
        if bool((lengths != audio.shape[-1]).any()):
            raise ValueError("CQTDiscriminator does not support ragged batches")
        logits: list[Tensor] = []
        features: list[list[Tensor]] = []
        for transform, discriminator in zip(self.transforms, self.discriminators):
            values = transform(audio.flatten(0, 1))
            if values.is_complex():
                spectrum = values
            elif values.shape[-1] == 2:
                spectrum = torch.view_as_complex(values.float().contiguous())
            else:
                raise RuntimeError("nnAudio CQT output must be complex or end in [..., 2]")
            spectrum = spectrum.reshape(
                audio.shape[0],
                AUDIO_CHANNELS,
                spectrum.shape[-2],
                spectrum.shape[-1],
            ).to(torch.complex64)
            current_logits, current_features = discriminator(_complex_channels(spectrum))
            logits.append(current_logits)
            features.append(current_features)
        return DiscriminatorOutput(logits, features)


class SpectralDiscriminator(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int = 16,
        stft_scales: Sequence[Sequence[int]] = (SPECTROSTREAM_PUBLIC_DISCRIMINATOR_SCALES),
        stft_frontend_profile: str = "spectrostream_public_v1",
    ) -> None:
        super().__init__()
        self.input_domain = "waveform"
        self.discriminator = SpectroStreamMultiScaleSTFTDiscriminator(
            scales=stft_scales,
            base_channels=base_channels,
            frontend_profile=stft_frontend_profile,
        )

    def forward(
        self,
        audio: Tensor,
        lengths: Tensor | None = None,
    ) -> DiscriminatorOutput:
        return self.discriminator(audio, lengths)


class AcousticDiscriminator(nn.Module):
    def __init__(
        self,
        *,
        stft_scales: Sequence[Sequence[int]] = DEFAULT_STFT_SCALES,
        cqt_scales: Sequence[Mapping[str, Any]] = (),
        base_channels: int = 16,
        depth: int = 4,
        stft_frontend_profile: str = "spectrostream_public_v1",
        stft_feature_bands_hz: Sequence[Sequence[float]] = (),
        stft_feature_band_layers: Sequence[int] = (0, 1, 2),
        stft_feature_band_minimum_bins: int = 1,
        stft_feature_band_require_complete_layer: bool = False,
        stft_feature_band_max_receptive_field_hz: float | None = None,
        cqt_enabled: bool = True,
        loss_reduction: str = "flat_feature_mean_v1",
        family_weights: Sequence[float] | None = None,
        loss_compute_dtype: str = "float32",
        family_diagnostics_enabled: bool | None = None,
    ) -> None:
        super().__init__()
        allowed_reductions = {
            "flat_feature_mean_v1",
            "family_equal_mean_v1",
            "family_weighted_sum_v1",
        }
        if loss_reduction not in allowed_reductions:
            raise ValueError(f"loss_reduction must be one of {sorted(allowed_reductions)}")
        if loss_compute_dtype not in {"float32", "float64"}:
            raise ValueError("loss_compute_dtype must be float32 or float64")
        parsed_family_weights = (
            None if family_weights is None else tuple(float(value) for value in family_weights)
        )
        if loss_reduction == "family_weighted_sum_v1":
            expected_family_count = 2 if cqt_enabled else 1
            if (
                parsed_family_weights is None
                or len(parsed_family_weights) != expected_family_count
                or any(not math.isfinite(value) or value < 0.0 for value in parsed_family_weights)
                or not any(value > 0.0 for value in parsed_family_weights)
            ):
                raise ValueError(
                    "family_weights must contain one non-negative value per "
                    "enabled discriminator family and at least one positive value"
                )
        elif parsed_family_weights is not None:
            raise ValueError("family_weights is only valid with family_weighted_sum_v1")
        self.loss_reduction = str(loss_reduction)
        self.family_weights = parsed_family_weights
        self.loss_compute_dtype = str(loss_compute_dtype)
        self.family_diagnostics_enabled = family_diagnostics_enabled
        self.stft = SpectroStreamMultiScaleSTFTDiscriminator(
            stft_scales,
            base_channels=base_channels,
            frontend_profile=stft_frontend_profile,
            feature_bands_hz=stft_feature_bands_hz,
            feature_band_layers=stft_feature_band_layers,
            feature_band_minimum_bins=stft_feature_band_minimum_bins,
            feature_band_require_complete_layer=(stft_feature_band_require_complete_layer),
            feature_band_max_receptive_field_hz=(stft_feature_band_max_receptive_field_hz),
        )
        self.cqt = (
            CQTDiscriminator(
                cqt_scales,
                base_channels=base_channels,
                depth=depth,
            )
            if cqt_enabled
            else None
        )

    def forward(
        self,
        audio: Tensor,
        lengths: Tensor | None = None,
    ) -> DiscriminatorOutput:
        stft_output = self.stft(audio, lengths)
        if self.cqt is None:
            cqt_logits: list[Tensor] = []
            cqt_features: list[list[Tensor]] = []
            family_names = ("stft",)
            family_sizes = (len(stft_output.logits),)
        else:
            cqt_output = self.cqt(audio, lengths)
            cqt_logits = cqt_output.logits
            cqt_features = cqt_output.features
            family_names = ("stft", "cqt")
            family_sizes = (len(stft_output.logits), len(cqt_logits))
        return DiscriminatorOutput(
            logits=stft_output.logits + cqt_logits,
            features=stft_output.features + cqt_features,
            highband_features=stft_output.highband_features,
            highband_feature_views=stft_output.highband_feature_views,
            family_names=family_names,
            family_sizes=family_sizes,
            family_weights=self.family_weights,
            loss_reduction=self.loss_reduction,
            loss_compute_dtype=self.loss_compute_dtype,
            family_diagnostics_enabled=self.family_diagnostics_enabled,
        )


MultiScaleSTFTCQTDiscriminator = AcousticDiscriminator
