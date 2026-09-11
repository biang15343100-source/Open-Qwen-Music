"""\u03b5ar-VAE2 base topology adapted to the OQM complex-spectrum API.

Provenance:
    Repository: https://github.com/Eps-Acoustic-Revolution-Lab/EAR_VAE2
    Commit: bcbd9e1dacebb494ed9c8a4e9f36338c9de5d03b
    Source file: ear_vae2/model.py
    Source SHA-256:
      126b91338d2b2e268a10a4ea58396b579ddadd206a8685dce0980170a683eaa6
    License: Apache-2.0

Only the base encoder/decoder path is adapted here. Open-Qwen-Music supplies its
own versioned STFT, refiner, ragged-time masks, and
``[B,2,F,T] complex <-> [B, T, 128]`` interface.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from .contracts import (
    LATENT_DIM,
    SPEC_FRAMES_PER_LATENT_FRAME,
    STFT_BINS,
    lengths_to_mask,
    stft_lengths_to_latent_lengths,
    validate_latents,
    validate_spectrum,
)


SOURCE_RATIOS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (1, 2),
    (1, 3),
    (1, 2),
    (1, 2),
    (2, 2),
    (2, 1),
)
SOURCE_FREQUENCIES: tuple[int, ...] = (480, 240, 120, 40, 20, 10, 5, 5)


def _mask_time(values: Tensor, lengths: Tensor) -> Tensor:
    mask = lengths_to_mask(lengths, values.shape[-2], device=values.device)
    return values * mask[:, None, :, None]


def _length_path(lengths: Tensor) -> list[Tensor]:
    result = [lengths.to(dtype=torch.long)]
    for time_stride, _ in SOURCE_RATIOS:
        result.append(
            torch.div(
                result[-1] + time_stride - 1,
                time_stride,
                rounding_mode="floor",
            )
        )
    return result


class SourceSnakeBeta2d(nn.Module):

    def __init__(self, channels: int, frequency_bins: int) -> None:
        super().__init__()
        frequencies = torch.linspace(0, 24_000, frequency_bins + 1)[:-1]
        prior = torch.log(frequencies / frequencies.mean() + 1.0e-6)
        self.alpha = nn.Parameter(prior.clone())
        self.beta = nn.Parameter(torch.zeros(frequency_bins))
        self.channels = int(channels)
        self.frequency_bins = int(frequency_bins)
        self.epsilon = 1.0e-9

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError(
                f"SourceSnakeBeta2d expects [B, C, T, F], received {tuple(inputs.shape)}"
            )
        if inputs.shape[-1] != self.frequency_bins:
            raise ValueError("SourceSnakeBeta2d received an unexpected frequency dimension")
        alpha = self.alpha.view(1, 1, -1, 1).exp()
        beta = self.beta.view(1, 1, -1, 1).exp()
        work = inputs.permute(0, 1, 3, 2)
        output = work + torch.sin(work * alpha).square() / (beta + self.epsilon)
        return output.permute(0, 1, 3, 2)


class SourceSnakeBeta1d(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.channels = int(channels)
        self.epsilon = 1.0e-9

    def forward(self, inputs: Tensor) -> Tensor:
        alpha = self.alpha.view(1, -1, 1).exp()
        beta = self.beta.view(1, -1, 1).exp()
        return inputs + torch.sin(inputs * alpha).square() / (beta + self.epsilon)


class SourceEncoderBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: tuple[int, int],
        frequency_bins: int,
        weight_norm_enabled: bool,
    ) -> None:
        super().__init__()
        self.act = SourceSnakeBeta2d(out_channels, frequency_bins)
        time_stride, frequency_stride = stride

        def norm(module: nn.Module) -> nn.Module:
            return weight_norm(module) if weight_norm_enabled else module

        self.conv1 = norm(nn.Conv2d(in_channels, in_channels, kernel_size=3))
        kernel_time = max(3, 2 * time_stride)
        kernel_frequency = max(3, 2 * frequency_stride)
        self.conv2 = norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_time, kernel_frequency),
                stride=stride,
            )
        )
        self.has_shortcut = in_channels != out_channels or stride != (1, 1)
        if self.has_shortcut:
            self.projection = norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.avg_pool = (
                nn.AvgPool2d(kernel_size=stride, stride=stride)
                if stride != (1, 1)
                else nn.Identity()
            )

    def forward(self, inputs: Tensor) -> Tensor:
        shortcut = (
            self.projection(self.avg_pool(inputs)) if self.has_shortcut else inputs
        )
        outputs = self.act(inputs)
        outputs = self.conv1(F.pad(outputs, (1, 1, 1, 1)))
        kernel_time, kernel_frequency = self.conv2.kernel_size
        time_stride = self.conv2.stride[0]
        pad_time = kernel_time - time_stride
        before = pad_time // 2
        after = pad_time - before
        pad_frequency = (kernel_frequency - 1) // 2
        outputs = self.act(outputs)
        outputs = self.conv2(
            F.pad(
                outputs,
                (pad_frequency, pad_frequency, before, after),
            )
        )
        if outputs.shape != shortcut.shape:
            raise RuntimeError(
                "Source encoder residual shape mismatch: "
                f"{tuple(outputs.shape)} != {tuple(shortcut.shape)}"
            )
        return outputs + shortcut


class SourceDecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: tuple[int, int],
        frequency_bins: int,
        weight_norm_enabled: bool,
    ) -> None:
        super().__init__()
        self.act = SourceSnakeBeta2d(out_channels, frequency_bins)
        time_stride, frequency_stride = stride

        def norm(module: nn.Module) -> nn.Module:
            return weight_norm(module) if weight_norm_enabled else module

        self.conv1 = norm(nn.Conv2d(in_channels, out_channels, kernel_size=3))
        kernel_time = max(3, 2 * time_stride)
        kernel_frequency = max(3, 2 * frequency_stride)
        padding_time = (kernel_time - time_stride) // 2
        padding_frequency = (kernel_frequency - frequency_stride) // 2
        self.transposed_conv = norm(
            nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_time, kernel_frequency),
                padding=(padding_time, padding_frequency),
                stride=stride,
            )
        )
        self.has_shortcut = in_channels != out_channels or stride != (1, 1)
        if self.has_shortcut:
            self.projection = norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.upsample = (
                nn.Upsample(scale_factor=stride, mode="nearest")
                if stride != (1, 1)
                else nn.Identity()
            )

    def forward(self, inputs: Tensor) -> Tensor:
        shortcut = (
            self.projection(self.upsample(inputs)) if self.has_shortcut else inputs
        )
        outputs = self.act(inputs)
        outputs = self.conv1(F.pad(outputs, (1, 1, 1, 1)))
        outputs = self.transposed_conv(self.act(outputs))
        if outputs.shape[-2] != shortcut.shape[-2]:
            difference = outputs.shape[-2] - shortcut.shape[-2]
            start = difference // 2
            outputs = outputs[..., start : start + shortcut.shape[-2], :]
        if outputs.shape[-1] != shortcut.shape[-1]:
            difference = outputs.shape[-1] - shortcut.shape[-1]
            start = difference // 2
            outputs = outputs[..., start : start + shortcut.shape[-1]]
        if outputs.shape != shortcut.shape:
            raise RuntimeError(
                "Source decoder residual shape mismatch: "
                f"{tuple(outputs.shape)} != {tuple(shortcut.shape)}"
            )
        return outputs + shortcut


class SourceBottleneck1D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        weight_norm_enabled: bool,
    ) -> None:
        super().__init__()
        hidden = max(in_channels, out_channels)

        def norm(module: nn.Module) -> nn.Module:
            return weight_norm(module) if weight_norm_enabled else module

        self.act = SourceSnakeBeta1d(in_channels)
        self.conv1 = norm(nn.Conv1d(in_channels, hidden, kernel_size=1))
        self.act2 = SourceSnakeBeta1d(hidden)
        self.conv2 = norm(nn.Conv1d(hidden, out_channels, kernel_size=1))
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else norm(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        )

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.shortcut(inputs)
        outputs = self.conv1(self.act(inputs))
        outputs = self.conv2(self.act2(outputs))
        return outputs + residual


class SourceSpecEncoder(nn.Module):

    def __init__(
        self,
        *,
        base_channels: int = 64,
        latent_dim: int = LATENT_DIM,
        weight_norm_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.C0 = int(base_channels)
        self.D = 2 * int(latent_dim)

        def norm(module: nn.Module) -> nn.Module:
            return weight_norm(module) if weight_norm_enabled else module

        self.input_conv = norm(nn.Conv2d(2, self.C0, kernel_size=(7, 7)))
        specs = (
            (1, 2, (1, 2), 1),
            (2, 2, (1, 2), 2),
            (2, 4, (1, 3), 4),
            (4, 4, (1, 2), 12),
            (4, 4, (1, 2), 24),
            (4, 8, (2, 2), 48),
            (8, 8, (2, 1), 96),
        )
        self.encoder_stream = nn.ModuleList(
            SourceEncoderBlock(
                in_multiplier * self.C0,
                out_multiplier * self.C0,
                stride=stride,
                frequency_bins=STFT_BINS // divisor,
                weight_norm_enabled=weight_norm_enabled,
            )
            for in_multiplier, out_multiplier, stride, divisor in specs
        )
        self.post_concat_block = SourceEncoderBlock(
            16 * self.C0,
            8 * self.C0,
            stride=(1, 1),
            frequency_bins=5,
            weight_norm_enabled=weight_norm_enabled,
        )
        self.bottleneck_block = SourceBottleneck1D(
            40 * self.C0,
            self.D,
            weight_norm_enabled=weight_norm_enabled,
        )

    @staticmethod
    def _to_channels(spectrum: Tensor) -> Tensor:
        components = torch.stack((spectrum.real, spectrum.imag), dim=1)
        return components.permute(0, 1, 3, 2).float()

    def _run_stream(self, values: Tensor) -> Tensor:
        outputs = self.input_conv(F.pad(values, (3, 3, 3, 3)))
        for block in self.encoder_stream:
            outputs = block(outputs)
        return outputs

    def forward(
        self,
        spectrum: Tensor,
        spectrum_lengths: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        validate_spectrum(spectrum, spectrum_lengths)
        frame_mask = lengths_to_mask(
            spectrum_lengths,
            spectrum.shape[-1],
            device=spectrum.device,
        )
        spectrum = spectrum * frame_mask[:, None, None, :]
        left = self._run_stream(self._to_channels(spectrum[:, 0]))
        right = self._run_stream(self._to_channels(spectrum[:, 1]))
        outputs = self.post_concat_block(torch.cat((left, right), dim=1))
        batch, channels, time, frequency = outputs.shape
        if frequency != 5:
            raise RuntimeError(f"Source encoder expected 5 frequency bins, received {frequency}")
        moments = self.bottleneck_block(
            outputs.permute(0, 2, 1, 3)
            .reshape(batch, time, channels * frequency)
            .transpose(1, 2)
        ).transpose(1, 2)
        latent_lengths = stft_lengths_to_latent_lengths(spectrum_lengths)
        if moments.shape[1] != int(
            (spectrum.shape[-1] + SPEC_FRAMES_PER_LATENT_FRAME - 1)
            // SPEC_FRAMES_PER_LATENT_FRAME
        ):
            raise RuntimeError("Source encoder output does not satisfy the 25 Hz contract")
        latent_mask = lengths_to_mask(
            latent_lengths,
            moments.shape[1],
            device=moments.device,
        )
        return moments * latent_mask.unsqueeze(-1), latent_lengths, latent_mask


class SourceSpecDecoder(nn.Module):

    def __init__(
        self,
        *,
        base_channels: int = 128,
        latent_dim: int = LATENT_DIM,
        weight_norm_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.C0 = int(base_channels)
        self.D = int(latent_dim)
        self.bottleneck_block = SourceBottleneck1D(
            self.D,
            40 * self.C0,
            weight_norm_enabled=weight_norm_enabled,
        )
        self.pre_split_block = SourceDecoderBlock(
            8 * self.C0,
            16 * self.C0,
            stride=(1, 1),
            frequency_bins=5,
            weight_norm_enabled=weight_norm_enabled,
        )
        specs: Sequence[tuple[int, int, tuple[int, int], int]] = (
            (8, 8, (2, 1), 96),
            (8, 4, (2, 2), 96),
            (4, 4, (1, 2), 48),
            (4, 4, (1, 2), 24),
            (4, 2, (1, 3), 12),
            (2, 2, (1, 2), 4),
            (2, 1, (1, 2), 2),
        )
        self.decoder_stream = nn.ModuleList(
            SourceDecoderBlock(
                in_multiplier * self.C0,
                out_multiplier * self.C0,
                stride=stride,
                frequency_bins=STFT_BINS // divisor,
                weight_norm_enabled=weight_norm_enabled,
            )
            for in_multiplier, out_multiplier, stride, divisor in specs
        )
        output = nn.Conv2d(self.C0, 2, kernel_size=(7, 7))
        self.output_conv = weight_norm(output) if weight_norm_enabled else output
        self.latent_crest_scale_head: nn.Module | None = None

    @staticmethod
    def _to_complex(components: Tensor) -> Tensor:
        values = components.permute(0, 2, 3, 1).float()
        return torch.complex(values[..., 0], values[..., 1]).permute(0, 2, 1)

    def forward(
        self,
        latents: Tensor,
        *,
        spectrum_frames: int | None = None,
        latent_lengths: Tensor | None = None,
        spectrum_lengths: Tensor | None = None,
    ) -> Tensor:
        validate_latents(latents, latent_lengths)
        batch, latent_frames, _ = latents.shape
        target_frames = (
            latent_frames * SPEC_FRAMES_PER_LATENT_FRAME
            if spectrum_frames is None
            else int(spectrum_frames)
        )
        if target_frames <= 0:
            raise ValueError("spectrum_frames must be positive")
        if latent_frames * SPEC_FRAMES_PER_LATENT_FRAME != target_frames:
            raise ValueError("Source decoder requires spectrum_frames == 4 * latent_frames")
        if latent_lengths is None:
            latent_lengths = torch.full(
                (batch,), latent_frames, dtype=torch.long, device=latents.device
            )
        else:
            latent_lengths = latent_lengths.to(device=latents.device, dtype=torch.long)
        latents = latents * lengths_to_mask(
            latent_lengths, latent_frames, device=latents.device
        ).unsqueeze(-1)
        outputs = self.bottleneck_block(latents.transpose(1, 2))
        outputs = outputs.transpose(1, 2).reshape(
            batch, latent_frames, 8 * self.C0, 5
        ).permute(0, 2, 1, 3)
        outputs = self.pre_split_block(outputs)
        left, right = torch.chunk(outputs, 2, dim=1)
        for block in self.decoder_stream:
            left = block(left)
            right = block(right)
        left = self.output_conv(F.pad(left, (3, 3, 3, 3)))
        right = self.output_conv(F.pad(right, (3, 3, 3, 3)))
        result = torch.stack(
            (self._to_complex(left), self._to_complex(right)),
            dim=1,
        )
        result = result[..., :target_frames]
        if spectrum_lengths is None:
            spectrum_lengths = (latent_lengths * SPEC_FRAMES_PER_LATENT_FRAME).clamp(
                max=target_frames
            )
        mask = lengths_to_mask(
            spectrum_lengths.to(device=result.device, dtype=torch.long),
            target_frames,
            device=result.device,
        )
        return (result * mask[:, None, None, :]).to(torch.complex64)
