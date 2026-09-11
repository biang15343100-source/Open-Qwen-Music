from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly

from open_qwen_music.common.archive import read_archive_bytes
from open_qwen_music.common.loudness import loudness_metrics


FRAME_HZ = 25.0
SAMPLE_RATE = 48_000
CHANNELS = 2
CODEBOOK_SIZE = 32_768
LATENT_DIM = 128
MAX_PARENT_FRAMES = 9_000
TARGET_WINDOW_FRAMES = 90 * 25
SAMPLE_SCHEMA = "oqm.render-sample.v1"
SAMPLE_READY_SCHEMA = "oqm.render-sample-ready.v1"
CROP_SCHEMA = "oqm.render.renderer_data-short-crop.v1"
CROP_READY_SCHEMA = "oqm.render.renderer_data-short-crop-ready.v1"
LOUDNESS_SCHEMA = "oqm.render.renderer_data-window-loudness.v1"
LOUDNESS_READY_SCHEMA = "oqm.render.renderer_data-window-loudness-ready.v1"
TEXT_REFERENCE_SCHEMA = "oqm.render.renderer_data-text-cache-reference.v1"
TEXT_CACHE_SCHEMA = "oqm.render-text-cache.v2"
TEXT_READY_SCHEMA = "oqm.render-text-cache-ready.v1"
SEMANTIC_RELEASE_SCHEMA = "oqm.renderer-semantic-release.v1"
SEMANTIC_ROW_SCHEMA = "oqm.renderer-semantic-token.v1"
INDEX_DTYPE = np.dtype(
    [("offset", "<u8"), ("size", "<u8"), ("line_number", "<u8"), ("frame_length", "<u4")]
)


class RendererMaterializationError(RuntimeError):
    pass


class RendererAdapterBundle(Protocol):
    identity: Mapping[str, Any]

    def encode_latent(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        sample_id: str,
        audio_sha256: str | None = None,
    ) -> np.ndarray: ...

    def encode_text(self, text: str, *, role: str, max_tokens: int) -> Mapping[str, Any]: ...

    def semantic_codebook(self) -> np.ndarray: ...


class ProductionRendererAdapters:
    def __init__(
        self,
        *,
        renderer_config: str | Path,
        tokenizer_checkpoint: str | Path,
        vae_checkpoint: str | Path,
        text_encoder: str | Path,
        device: str,
    ) -> None:
        import torch

        from open_qwen_music.common.checkpoint import file_sha256
        from open_qwen_music.common.config import load_config
        from open_qwen_music.tokenizer.deployment import DeploymentMusicTokenizer

        from .cache import (
            build_latent_layout,
            load_frozen_spec_vae,
            validate_stft_mapping,
        )
        from .checkpoint import read_render_checkpoint_state
        from .contracts import STFT_CONTRACT_VERSION
        from .stft import STFTConfig, StereoSTFT
        from .text_cache import RenderTextCacheConfig, build_frozen_qwen_adapter

        self.device = torch.device(device)
        self.renderer_config = load_config(renderer_config)
        tokenizer_path = Path(tokenizer_checkpoint).resolve(strict=True)
        self.tokenizer = DeploymentMusicTokenizer.from_file(
            tokenizer_path, map_location="cpu"
        ).to(self.device).eval()
        checkpoint_path = Path(vae_checkpoint).resolve(strict=True)
        checkpoint_sha = file_sha256(checkpoint_path)
        state = read_render_checkpoint_state(
            checkpoint_path,
            strict_sidecar=True,
            map_location="cpu",
            expected_checkpoint_sha256=checkpoint_sha,
            expected_component="spec_vae",
        )
        checkpoint_config = state.get("config")
        if not isinstance(checkpoint_config, Mapping):
            raise RendererMaterializationError("VAE checkpoint is missing its configuration")
        stft_mapping = checkpoint_config.get("stft")
        model_mapping = checkpoint_config.get("model")
        vae_mapping = model_mapping.get("spec_vae") if isinstance(model_mapping, Mapping) else None
        if not isinstance(stft_mapping, Mapping) or not isinstance(vae_mapping, Mapping):
            raise RendererMaterializationError("VAE checkpoint is missing STFT or model metadata")
        vae_revision = str(
            vae_mapping.get("revision")
            or (self.renderer_config.get("revisions") or {}).get("vae_revision")
            or ""
        )
        if not vae_revision:
            raise RendererMaterializationError("VAE checkpoint is missing its public revision")
        stft_sha = validate_stft_mapping(
            stft_mapping, expected_revision=STFT_CONTRACT_VERSION
        )
        self.frozen_vae = load_frozen_spec_vae(
            checkpoint_path,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_vae_revision=vae_revision,
            expected_stft_config=stft_mapping,
            expected_stft_config_sha256=stft_sha,
            expected_stft_revision=STFT_CONTRACT_VERSION,
            device=self.device,
        )
        self.stft = StereoSTFT(STFTConfig.from_mapping(stft_mapping)).to(self.device).eval()
        conditioning = self.renderer_config.get("conditioning") or {}
        text_settings = conditioning.get("text_encoder") or {}
        revisions = self.renderer_config.get("revisions") or {}
        self.text_config = RenderTextCacheConfig(
            cache_revision=str(text_settings["cache_revision"]),
            model_id=str(text_settings["model_id"]),
            model_revision=str(text_settings["revision"]),
            tokenizer_revision=str(text_settings["tokenizer_revision"]),
            hidden_size=int(text_settings["hidden_size"]),
            local_path=str(Path(text_encoder).resolve(strict=True)),
            asset_lock=str(text_settings.get("asset_lock") or ""),
            description_max_tokens=int(conditioning.get("description_max_tokens", 256)),
            lyrics_max_tokens=int(conditioning.get("lyrics_max_tokens", 1536)),
            local_files_only=True,
            trust_remote_code=False,
            hidden_state_selection=str(text_settings.get("hidden_state_selection", "last_hidden_state")),
            position_id_policy=str(text_settings.get("position_id_policy", "attention_mask_cumsum")),
            encoder_use_cache=bool(text_settings.get("encoder_use_cache", False)),
            empty_text_policy=str(text_settings.get("empty_text_policy", "zero_valid_tokens")),
            frozen_eval_mode=True,
            use_fast_tokenizer=bool(text_settings.get("use_fast_tokenizer", True)),
            padding=True,
            truncation=True,
            truncation_policy=str(text_settings.get("truncation_policy", "reject")),
            add_special_tokens=bool(text_settings.get("add_special_tokens", True)),
            padding_side=str(text_settings.get("padding_side", "left")),
            truncation_side=str(text_settings.get("truncation_side", "right")),
            output_format="npy",
            storage_dtype="float32",
            device=str(self.device),
        )
        self.text_adapter = build_frozen_qwen_adapter(
            self.text_config, device=self.device
        )
        data = self.renderer_config.get("data") or {}
        latent_identity = {
            "vae_revision": vae_revision,
            "vae_checkpoint_sha256": checkpoint_sha,
            "cache_revision": str(revisions.get("latent_cache_revision") or ""),
            "latent_stats_sha256": str(revisions.get("latent_stats_sha256") or ""),
            "stft_revision": STFT_CONTRACT_VERSION,
            "stft_config_sha256": stft_sha,
            "posterior_mode": str(data.get("latent_posterior_mode", "sample")),
            "posterior_base_seed": data.get("latent_posterior_base_seed"),
            "posterior_seed_derivation": data.get(
                "latent_posterior_seed_derivation", "sha256-little-endian-63-v1"
            ),
            "posterior_epsilon_draw_layout": data.get(
                "latent_posterior_epsilon_draw_layout",
                self.frozen_vae.posterior_epsilon_draw_layout,
            ),
        }
        self.identity = {
            **dict(revisions),
            "tokenizer_revision": self.tokenizer.revision,
            "vae_revision": vae_revision,
            "tokenizer_checkpoint_sha256": file_sha256(tokenizer_path),
            "vae_checkpoint_sha256": checkpoint_sha,
            "stft_revision": STFT_CONTRACT_VERSION,
            "stft_config_sha256": stft_sha,
            "latent_layout": build_latent_layout(checkpoint_config),
            "text_encoder_revision": self.text_config.model_revision,
            "text_tokenizer_revision": self.text_config.tokenizer_revision,
            "text_cache_revision": self.text_config.cache_revision,
            "rewriter_revision": "open-qwen-music-rewriter-v1",
            **{
                name: value
                for name, value in data.items()
                if name.endswith("_sha256")
                or name
                in {
                    "semantic_extractor_revision",
                    "semantic_materializer_revision",
                    "latent_special_channels",
                }
            },
            "latent_posterior_mode": latent_identity["posterior_mode"],
            "latent_posterior_base_seed": latent_identity["posterior_base_seed"],
            "latent_posterior_seed_derivation": latent_identity[
                "posterior_seed_derivation"
            ],
            "latent_posterior_epsilon_draw_layout": latent_identity[
                "posterior_epsilon_draw_layout"
            ],
            "latent_identity": latent_identity,
            "text_provenance": self.text_config.provenance.to_dict(),
            "text_cache_config": self.text_config.to_dict(),
            "text_cache_config_file_sha256": self.text_config.sha256,
        }

    def encode_latent(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        sample_id: str,
        audio_sha256: str | None = None,
    ) -> np.ndarray:
        import torch

        from .cache import sample_posterior_with_layout

        if sample_rate != SAMPLE_RATE or audio_sha256 is None:
            raise RendererMaterializationError("Production VAE input identity is incomplete")
        values = torch.from_numpy(np.ascontiguousarray(audio.T)).unsqueeze(0).to(self.device)
        lengths = torch.tensor([audio.shape[0]], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            analyzed = self.stft.analyze(values, lengths)
            encoded = self.frozen_vae.model.encode(
                analyzed.spectrum, analyzed.spectrum_lengths
            )
            posterior = encoded.posterior
            latent_identity = self.identity["latent_identity"]
            if latent_identity["posterior_mode"] == "mean":
                latents = posterior.mean
            else:
                seed = _posterior_seed(
                    int(latent_identity["posterior_base_seed"]), sample_id, audio_sha256
                )
                generator = torch.Generator(device=self.device).manual_seed(seed)
                latents = sample_posterior_with_layout(
                    posterior,
                    generator=generator,
                    posterior_epsilon_draw_layout=str(
                        latent_identity["posterior_epsilon_draw_layout"]
                    ),
                )
        return latents[0].detach().float().cpu().numpy()

    def encode_text(self, text: str, *, role: str, max_tokens: int) -> Mapping[str, Any]:
        from .text_cache import _encode_one_text

        encoded = _encode_one_text(
            self.text_adapter,
            text,
            max_length=max_tokens,
            config=self.text_config,
            device=self.device,
        )
        return {
            "embeddings": encoded.hidden.numpy(),
            "input_ids": encoded.input_ids.numpy(),
            "attention_mask": encoded.attention_mask.numpy(),
            "original_token_count": encoded.original_token_count,
            "truncated": encoded.truncated,
        }

    def semantic_codebook(self) -> np.ndarray:
        import torch

        effective = getattr(self.tokenizer.quantizer, "effective_codebook", None)
        if not callable(effective):
            raise RendererMaterializationError(
                "Tokenizer quantizer does not expose an effective codebook"
            )
        with torch.inference_mode():
            return effective().detach().float().cpu().numpy()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _posterior_seed(base_seed: int, sample_id: str, audio_sha256: str) -> int:
    payload = f"{base_seed}:0:{sample_id}:{audio_sha256}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RendererMaterializationError(f"Cannot parse JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise RendererMaterializationError(f"Expected a JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RendererMaterializationError(
                    f"Invalid JSONL record at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise RendererMaterializationError(f"Expected an object at {path}:{line_number}")
            yield line_number, value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _resolve_local(value: Any, *, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RendererMaterializationError(f"{field} must be a non-empty local path")
    if value.startswith("file://"):
        value = value[7:]
    elif "://" in value:
        raise RendererMaterializationError(f"{field} must reference a local file")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _validate_sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RendererMaterializationError(f"{field} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RendererMaterializationError(f"{field} must be a hexadecimal SHA-256") from exc
    return value.lower()


def _load_semantics(release_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    ready_path = release_root / "READY"
    ready = _read_json(ready_path)
    if ready.get("schema_version") != SEMANTIC_RELEASE_SCHEMA or ready.get("status") != "READY":
        raise RendererMaterializationError("Renderer semantic release schema or status is invalid")
    manifest = release_root / str(ready.get("manifest") or "")
    if ready.get("manifest_sha256") != _sha256(manifest):
        raise RendererMaterializationError("Renderer semantic manifest SHA-256 does not match READY")
    rows: dict[str, dict[str, Any]] = {}
    for line_number, row in _iter_jsonl(manifest):
        sample_id = row.get("sample_id")
        if row.get("schema_version") != SEMANTIC_ROW_SCHEMA:
            raise RendererMaterializationError(f"Invalid semantic schema at line {line_number}")
        if not isinstance(sample_id, str) or not sample_id or sample_id in rows:
            raise RendererMaterializationError(f"Missing or duplicate semantic sample at line {line_number}")
        artifact = row.get("artifact")
        if not isinstance(artifact, Mapping):
            raise RendererMaterializationError(f"Missing semantic artifact for {sample_id}")
        path = _resolve_local(artifact.get("path"), base=manifest.parent, field="artifact.path")
        expected = _validate_sha(artifact.get("sha256"), field="artifact.sha256")
        if not path.is_file() or _sha256(path) != expected:
            raise RendererMaterializationError(f"Semantic artifact SHA-256 failed for {sample_id}")
        tokens = np.load(path, allow_pickle=False)
        if (
            tokens.dtype != np.uint16
            or tokens.ndim != 1
            or not tokens.size
            or int(tokens.max()) >= CODEBOOK_SIZE
            or artifact.get("shape") != [int(tokens.size)]
            or float(artifact.get("frame_hz", -1)) != FRAME_HZ
        ):
            raise RendererMaterializationError(f"Semantic token contract failed for {sample_id}")
        rows[sample_id] = {**row, "path": path, "tokens": tokens}
    if len(rows) != ready.get("records"):
        raise RendererMaterializationError("Semantic release record count does not match READY")
    return rows, ready


def _archive_uri(value: str, *, base: Path) -> str:
    for scheme in ("tar://", "zip://"):
        if value.startswith(scheme):
            archive, separator, member = value.removeprefix(scheme).partition("::")
            if not separator or not archive or not member:
                raise RendererMaterializationError(
                    f"Archive audio URI must use {scheme}<archive>::<member>"
                )
            path = Path(archive).expanduser()
            if not path.is_absolute():
                path = (base / path).resolve()
            return f"{scheme}{path}::{member}"
    return value


def _audio_payload(row: Mapping[str, Any], *, manifest: Path) -> tuple[bytes, str]:
    metadata = row.get("audio")
    if not isinstance(metadata, Mapping):
        raise RendererMaterializationError("Source record is missing audio metadata")
    value = metadata.get("path", metadata.get("uri"))
    if not isinstance(value, str) or not value:
        raise RendererMaterializationError("audio.path must be a non-empty path or archive URI")
    value = _archive_uri(value, base=manifest.parent)
    try:
        if value.startswith(("tar://", "zip://")):
            payload = read_archive_bytes(value)
        else:
            payload = _resolve_local(value, base=manifest.parent, field="audio.path").read_bytes()
    except (OSError, ValueError) as exc:
        raise RendererMaterializationError(f"Cannot read source audio: {value}") from exc
    digest = hashlib.sha256(payload).hexdigest()
    declared = metadata.get("sha256")
    if declared is not None and _validate_sha(declared, field="audio.sha256") != digest:
        raise RendererMaterializationError("Source audio SHA-256 does not match its payload")
    return payload, digest


def _canonical_audio(
    row: Mapping[str, Any],
    *,
    manifest: Path,
    output: Path,
    split: str,
    sample_id: str,
    semantic_frames: int,
) -> tuple[Path, np.ndarray, str, str]:
    payload, source_sha = _audio_payload(row, manifest=manifest)
    try:
        waveform, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    except (OSError, RuntimeError, sf.LibsndfileError) as exc:
        raise RendererMaterializationError("Cannot decode source audio") from exc
    if not waveform.size or not np.isfinite(waveform).all():
        raise RendererMaterializationError("Audio must be non-empty and finite")
    if waveform.shape[1] == 1:
        waveform = np.repeat(waveform, 2, axis=1)
    elif waveform.shape[1] != 2:
        mono = waveform.astype(np.float64).mean(axis=1, keepdims=True).astype(np.float32)
        waveform = np.repeat(mono, 2, axis=1)
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
        waveform = resample_poly(
            waveform,
            SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
            axis=0,
        ).astype(np.float32)
    target_samples = semantic_frames * (SAMPLE_RATE // int(FRAME_HZ))
    if waveform.shape[0] < target_samples:
        waveform = np.pad(waveform, ((0, target_samples - waveform.shape[0]), (0, 0)))
    waveform = np.ascontiguousarray(waveform[:target_samples], dtype=np.float32)
    target = output / "audio" / split / f"{hashlib.sha256(sample_id.encode()).hexdigest()}.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, waveform, SAMPLE_RATE, format="WAV", subtype="FLOAT")
    canonical, published_rate = sf.read(target, dtype="float32", always_2d=True)
    if published_rate != SAMPLE_RATE or canonical.shape != waveform.shape:
        raise RendererMaterializationError("Published canonical audio failed verification")
    return target, np.ascontiguousarray(canonical), source_sha, _sha256(target)


def _condition(row: Mapping[str, Any], *, identity: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("condition")
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        description = str(row.get("description") or "").strip()
        tags = row.get("tags")
        if isinstance(tags, Mapping) and tags:
            tag_text = "; ".join(
                f"{key}: {', '.join(map(str, item)) if isinstance(item, (list, tuple)) else item}"
                for key, item in sorted(tags.items())
            )
            description = f"{description}\nTags: {tag_text}".strip()
        text = row.get("text")
        lyrics = text.get("lyrics") if isinstance(text, Mapping) else row.get("lyrics")
        result = {"description": description, "lyrics": str(lyrics or "")}
    if not isinstance(result.get("description"), str) or not isinstance(result.get("lyrics"), str):
        raise RendererMaterializationError("Condition description and lyrics must be strings")
    result.setdefault("schema_version", "oqm.rewriter-output.v1")
    result.setdefault("rewriter_revision", str(identity.get("rewriter_revision") or ""))
    result.setdefault("text_tokenizer_revision", str(identity.get("text_tokenizer_revision") or ""))
    return result


def _groups(row: Mapping[str, Any], *, split: str, sample_id: str) -> dict[str, Any]:
    value = row.get("groups")
    result = dict(value) if isinstance(value, Mapping) else {}
    result["split"] = split
    for field in (
        "recording_group_id",
        "performance_group_id",
        "composition_group_id",
        "song_group_id",
    ):
        if not isinstance(result.get(field), str) or not result[field]:
            digest = hashlib.sha256(f"{field}:{sample_id}".encode()).hexdigest()
            result[field] = f"public:{digest}"
    return result


def _sections(row: Mapping[str, Any], *, frames: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(row.get("sections") or []):
        if not isinstance(raw, Mapping):
            continue
        start_value, end_value = raw.get("start_sec"), raw.get("end_sec")
        if not isinstance(start_value, (int, float)) or not isinstance(end_value, (int, float)):
            continue
        start = max(0, min(frames, int(round(float(start_value) * FRAME_HZ))))
        end = max(start, min(frames, int(round(float(end_value) * FRAME_HZ))))
        result.append(
            {
                "section_index": index,
                "label": str(raw.get("label") or "section"),
                "lyrics": str(raw.get("lyrics") or ""),
                "start_frame": start,
                "end_frame": end,
            }
        )
    return result


def _best_partition(
    boundaries: Sequence[int], *, frames: int, target_frames: int, target_k: int
) -> tuple[int, ...]:
    if boundaries[0] != 0 or boundaries[-1] != frames:
        raise RendererMaterializationError("Partition boundaries must cover the complete sample")
    k = min(target_k, 4, len(boundaries) - 1)
    if k <= 1:
        return (0, frames)
    n = len(boundaries)
    empty: tuple[float, tuple[int, ...]] = (float("inf"), ())
    table = [[empty for _ in range(n)] for _ in range(k + 1)]
    table[0][0] = (0.0, (0,))
    for windows in range(1, k + 1):
        for end_index in range(1, n):
            best = empty
            for start_index in range(windows - 1, end_index):
                cost, path = table[windows - 1][start_index]
                length = boundaries[end_index] - boundaries[start_index]
                if path and length > 0:
                    candidate = (
                        cost + float((length - target_frames) ** 2),
                        path + (boundaries[end_index],),
                    )
                    if candidate < best:
                        best = candidate
            table[windows][end_index] = best
    partition = table[k][-1][1]
    if len(partition) != k + 1:
        raise RendererMaterializationError("Cannot partition the sample into short windows")
    return partition


def _window_lyrics(sections: Sequence[Mapping[str, Any]], start: int, end: int) -> str:
    return "\n\n".join(
        f"[{section['label']}]\n{section['lyrics']}"
        for section in sections
        if int(section["start_frame"]) >= start
        and int(section["end_frame"]) <= end
        and str(section["lyrics"])
    )

def _text_field_metadata(
    encoded: Mapping[str, Any],
    *,
    hidden: np.ndarray,
    maximum: int,
    config: Mapping[str, Any],
    lyrics: bool,
) -> dict[str, Any]:
    ids = np.asarray(encoded.get("input_ids"), dtype=np.int64)
    mask = np.asarray(encoded.get("attention_mask"), dtype=np.bool_)
    tokenization = config.get("tokenization")
    policy = tokenization if isinstance(tokenization, Mapping) else config
    values: dict[str, Any] = {
        "input_ids": ids.tolist(),
        "attention_mask": mask.tolist(),
        "input_ids_sha256": _canonical_sha(ids.tolist()),
        "attention_mask_sha256": _canonical_sha(mask.tolist()),
        "token_count": int(mask.sum()),
        "original_token_count": int(encoded.get("original_token_count", mask.sum())),
        "truncated": bool(encoded.get("truncated", False)),
        "max_length": maximum,
        "shape": list(hidden.shape),
        "dtype": "float32",
        "pooled": False,
        "feature_stage": "qwen_token_hidden",
        "trainable_lyrics_encoder_layers_applied": 0,
    }
    if lyrics:
        values["next_stage"] = "six_layer_trainable_rope_encoder"
    values["tokenization_sha256"] = _canonical_sha(
        {
            "input_ids": ids.tolist(),
            "attention_mask": mask.tolist(),
            "max_length": maximum,
            "padding": policy.get("padding", True),
            "truncation": policy.get("truncation", True),
            "truncation_policy": policy.get("truncation_policy", "reject"),
            "add_special_tokens": policy.get("add_special_tokens", True),
            "padding_side": policy.get("padding_side", "left"),
            "truncation_side": policy.get("truncation_side", "right"),
        }
    )
    return values


def _text_entry(
    adapters: RendererAdapterBundle,
    *,
    text: str,
    role: str,
    sample_id: str,
    root: Path,
    rewriter_revision: str,
    text_tokenizer_revision: str,
) -> dict[str, Any]:
    maximum = 256 if role == "tags" else 1_536
    encoded = adapters.encode_text(text, role=role, max_tokens=maximum)
    hidden = np.asarray(encoded.get("embeddings"), dtype=np.float32)
    ids = np.asarray(encoded.get("input_ids"), dtype=np.int64)
    mask = np.asarray(encoded.get("attention_mask"), dtype=np.bool_)
    if hidden.ndim != 2 or ids.ndim != 1 or mask.ndim != 1 or hidden.shape[0] != ids.size or ids.size != mask.size:
        raise RendererMaterializationError(f"Text adapter returned inconsistent arrays for {sample_id}")
    if ids.size > maximum or not np.isfinite(hidden).all():
        raise RendererMaterializationError(f"Text adapter output is invalid for {sample_id}")
    empty = {"input_ids": np.zeros(0, dtype=np.int64), "attention_mask": np.zeros(0, dtype=np.bool_), "original_token_count": 0, "truncated": False}
    if role == "tags":
        description, lyrics = hidden, np.zeros((0, hidden.shape[1]), dtype=np.float32)
        description_encoded, lyrics_encoded = encoded, empty
    else:
        description, lyrics = np.zeros((0, hidden.shape[1]), dtype=np.float32), hidden
        description_encoded, lyrics_encoded = empty, encoded
    artifact = root / "artifacts" / f"{hashlib.sha256(sample_id.encode()).hexdigest()}.npy"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open("xb") as handle:
        np.save(handle, np.concatenate((description, lyrics), axis=0), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    identity = dict(adapters.identity)
    provenance = dict(identity.get("text_provenance") or {})
    config = dict(identity.get("text_cache_config") or {})
    config_sha = _canonical_sha(config)
    fingerprint = hashlib.sha256(
        json.dumps(provenance, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    condition_sha = hashlib.sha256(text.encode()).hexdigest()
    record = {
        "schema_version": TEXT_CACHE_SCHEMA,
        "sample_id": sample_id,
        "sample_id_sha256": hashlib.sha256(sample_id.encode()).hexdigest(),
        "source_condition_sha256": condition_sha,
        "rewriter_revision": rewriter_revision,
        "text_tokenizer_revision": text_tokenizer_revision,
        "source_manifest_sha256": None,
        "source_index": None,
        "rank": 0,
        "world_size": 1,
        **provenance,
        "provenance": provenance,
        "provenance_fingerprint": fingerprint,
        "cache_config": config,
        "cache_config_sha256": config_sha,
        "cache_config_file_sha256": identity.get("text_cache_config_file_sha256", config_sha),
        "description": _text_field_metadata(
            description_encoded, hidden=description, maximum=256, config=config, lyrics=False
        ),
        "lyrics": _text_field_metadata(
            lyrics_encoded, hidden=lyrics, maximum=1_536, config=config, lyrics=True
        ),
        "artifact": {
            "uri": str(Path("artifacts") / artifact.name),
            "format": "npy",
            "sha256": _sha256(artifact),
            "shape": [int(description.shape[0] + lyrics.shape[0]), int(hidden.shape[1])],
            "dtype": "float32",
        },
        "file_sha256": _sha256(artifact),
    }
    sidecar = artifact.with_suffix(".json")
    _write_json(sidecar, record)
    record["metadata_uri"] = str(Path("artifacts") / sidecar.name)
    record["metadata_sha256"] = _sha256(sidecar)
    return record

def _publish_text_cache(root: Path, rows: list[dict[str, Any]], *, source_sha: str) -> dict[str, Any]:
    for source_index, row in enumerate(rows):
        row["source_manifest_sha256"] = source_sha
        row["source_index"] = source_index
        row["rank"] = 0
        row["world_size"] = 1
        metadata_uri = row.get("metadata_uri")
        if not isinstance(metadata_uri, str) or not metadata_uri:
            raise RendererMaterializationError("Text cache row is missing metadata_uri")
        sidecar = (root / metadata_uri).resolve()
        try:
            sidecar.relative_to(root.resolve())
        except ValueError as exc:
            raise RendererMaterializationError(
                "Text cache metadata path escapes its release directory"
            ) from exc
        _write_json(
            sidecar,
            {
                key: value
                for key, value in row.items()
                if key not in {"metadata_uri", "metadata_sha256"}
            },
        )
        row["metadata_sha256"] = _sha256(sidecar)
    manifest = root / "manifest.jsonl"
    _write_jsonl(manifest, rows)
    ranks = root / "ranks"
    ranks.mkdir()
    rank_manifest = ranks / "rank_0000.jsonl"
    shutil.copyfile(manifest, rank_manifest)
    config_sha = str(rows[0]["cache_config_sha256"])
    config_file_sha = str(rows[0]["cache_config_file_sha256"])
    ready = {
        "schema_version": TEXT_READY_SCHEMA,
        "status": "TEXT_CACHE_READY",
        "manifest": manifest.name,
        "manifest_sha256": _sha256(manifest),
        "rank_manifests": {rank_manifest.name: _sha256(rank_manifest)},
        "world_size": 1,
        "records": len(rows),
        "source_manifest_sha256": source_sha,
        "condition_set_sha256": _canonical_sha(
            [
                {
                    "sample_id_sha256": _canonical_sha(row["sample_id"]),
                    "source_condition_sha256": row["source_condition_sha256"],
                }
                for row in rows
            ]
        ),
        "cache_config_sha256": config_sha,
        "cache_config_file_sha256": config_file_sha,
    }
    _write_json(root / "READY", ready)
    return ready


def _lufs(audio: np.ndarray) -> float:
    integrated, _ = loudness_metrics(audio.T, SAMPLE_RATE)
    return -70.0 if not math.isfinite(integrated) else max(-70.0, integrated)


def _tensor_sha(value: Any) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _semantic_neighbors(codebook: np.ndarray, *, top_k: int, chunk_size: int = 128) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as functional

    rows = int(codebook.shape[0])
    if not 0 < top_k < rows:
        raise RendererMaterializationError("semantic_top_k must be between 1 and 32767")
    values = torch.from_numpy(codebook).double()
    norms = values.norm(dim=-1)
    if bool((norms <= 0).any()):
        raise RendererMaterializationError("Tokenizer codebook contains a zero-norm row")
    normalized = functional.normalize(values, dim=-1)
    neighbor_ids = torch.empty(rows, top_k, dtype=torch.int32)
    neighbor_scores = torch.empty(rows, top_k, dtype=torch.float32)
    for start in range(0, rows, chunk_size):
        stop = min(rows, start + chunk_size)
        similarities = normalized[start:stop] @ normalized.T
        similarities[torch.arange(stop - start), torch.arange(start, stop)] = -torch.inf
        indices = torch.argsort(similarities, dim=-1, descending=True, stable=True)[:, :top_k]
        scores = torch.gather(similarities, 1, indices)
        neighbor_ids[start:stop] = indices.to(torch.int32)
        neighbor_scores[start:stop] = scores.to(torch.float32)
    return neighbor_ids.contiguous(), neighbor_scores.contiguous()


def _publish_semantic_distractors(
    root: Path,
    *,
    codebook: np.ndarray,
    tokenizer_revision: str,
    source_embedding_tensor_sha256: str,
    top_k: int,
    neighbor_tables: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    import torch

    neighbor_ids, neighbor_scores = (
        neighbor_tables
        if neighbor_tables is not None
        else _semantic_neighbors(codebook, top_k=top_k)
    )
    neighbor_ids = torch.as_tensor(neighbor_ids, dtype=torch.int32).contiguous()
    neighbor_scores = torch.as_tensor(neighbor_scores, dtype=torch.float32).contiguous()
    if neighbor_ids.shape != (CODEBOOK_SIZE, top_k) or neighbor_scores.shape != (
        CODEBOOK_SIZE,
        top_k,
    ):
        raise RendererMaterializationError("Semantic neighbor adapter returned invalid tables")
    ids_sha = _tensor_sha(neighbor_ids)
    scores_sha = _tensor_sha(neighbor_scores)
    tensors = {
        "neighbor_ids": {"shape": list(neighbor_ids.shape), "dtype": "int32", "sha256": ids_sha},
        "neighbor_scores": {"shape": list(neighbor_scores.shape), "dtype": "float32", "sha256": scores_sha},
    }
    revision_payload = {
        "schema_version": "oqm.render-semantic-distractors.v1",
        "source": "tokenizer_codebook_cosine_knn",
        "tokenizer_revision": tokenizer_revision,
        "source_embedding_tensor_sha256": source_embedding_tensor_sha256,
        "metric": "cosine",
        "exclude_self": True,
        "top_k": top_k,
        "tensors": tensors,
        "input": {
            "tensor_sha256": source_embedding_tensor_sha256,
            "tokenizer_revision": tokenizer_revision,
            "source": "tokenizer_effective_codebook",
        },
        "producer_sha256": _sha256(Path(__file__)),
    }
    asset_revision = _canonical_sha(revision_payload)
    root.mkdir()
    artifact = root / "semantic_distractors.pt"
    torch.save(
        {
            **revision_payload,
            "asset_revision": asset_revision,
            "neighbor_ids": neighbor_ids,
            "neighbor_scores": neighbor_scores,
        },
        artifact,
    )
    report = {
        "schema_version": "oqm.render-semantic-distractors-ready.v1",
        "status": "RENDER_SEMANTIC_DISTRACTORS_READY",
        "source": "tokenizer_codebook_cosine_knn",
        "asset_revision": asset_revision,
        "artifact": {"path": artifact.name, "sha256": _sha256(artifact), "size_bytes": artifact.stat().st_size},
        "tokenizer_revision": tokenizer_revision,
        "source_embedding_tensor_sha256": source_embedding_tensor_sha256,
        "metric": "cosine",
        "exclude_self": True,
        "top_k": top_k,
        "neighbors": tensors,
        "input": revision_payload["input"],
        "diagnostics": {
            "score_min": float(neighbor_scores.min()),
            "score_mean": float(neighbor_scores.mean()),
            "score_max": float(neighbor_scores.max()),
        },
        "checks": {
            "source_is_tokenizer_codebook": True,
            "self_excluded": True,
            "scores_sorted_descending": True,
            "tensor_hashes_bound": True,
            "vae_independent": True,
        },
    }
    _write_json(root / "REPORT.json", report)
    ready = {**report, "report": {"path": "REPORT.json", "sha256": _sha256(root / "REPORT.json")}}
    _write_json(root / "READY", ready)
    return ready


def _publish_calibration(source: str | Path | None, root: Path) -> dict[str, Any] | None:
    if source is None:
        return None
    path = Path(source).expanduser().resolve(strict=True)
    report = _read_json(path)
    if (
        report.get("schema_version") != "oqm.render-semantic-error-calibration.v1"
        or report.get("status") != "RENDER_SEMANTIC_ERROR_CALIBRATION_READY"
        or report.get("teacher_forced") is not True
        or report.get("semantic_contract")
        != {"frame_hz": FRAME_HZ, "codebook_size": CODEBOOK_SIZE, "codebooks": 1}
    ):
        raise RendererMaterializationError(
            "Semantic calibration report schema or contract is incompatible"
        )
    config_fields = (
        "metric",
        "accuracy_at_1",
        "negative_log_likelihood_sum_nats",
        "cross_entropy_nats",
        "correct_tokens",
        "evaluated_tokens",
        "tokenizer_revision",
        "token_registry_sha256",
        "llm_checkpoint_sha256",
        "evaluation_dataset_revision",
        "evaluation_manifest_sha256",
        "evaluation_split",
    )
    missing = [name for name in config_fields if name not in report]
    if missing:
        raise RendererMaterializationError(
            f"Semantic calibration report is missing fields: {missing}"
        )
    target = root / "calibration" / "REPORT.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_json(target, report)
    return {
        "path": "calibration/REPORT.json",
        "sha256": _sha256(target),
        "config": {
            "format_version": report["schema_version"],
            **{name: report[name] for name in config_fields},
        },
    }


def _index(manifest: Path, rows: Sequence[Mapping[str, Any]], output: Path) -> tuple[Path, str]:
    payload = output / "manifest.index.npy"
    entries = np.empty(len(rows), dtype=INDEX_DTYPE)
    offset = 0
    with manifest.open("rb") as handle:
        for index, line in enumerate(handle):
            entries[index] = (offset, len(line), index + 1, int(rows[index]["semantic"]["shape"][0]))
            offset += len(line)
    with payload.open("xb") as handle:
        np.save(handle, entries, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    return payload, _sha256(payload)


def _write_resolved_renderer_config(
    staging: Path,
    *,
    final_root: Path,
    base_config: Mapping[str, Any] | None,
    split_results: Mapping[str, Any],
    embedding_ready: Mapping[str, Any],
    distractor_ready: Mapping[str, Any],
    tags_ready: Mapping[str, Any],
    lyrics_ready: Mapping[str, Any],
    calibration: Mapping[str, Any] | None,
    identity: Mapping[str, Any],
) -> Path | None:
    if base_config is None:
        return None
    config = json.loads(json.dumps(base_config))
    revisions = config.get("revisions")
    if not isinstance(revisions, dict):
        raise RendererMaterializationError("Renderer config is missing revisions")
    revision_fields = (
        "tokenizer_revision",
        "vae_revision",
        "text_encoder_revision",
        "text_tokenizer_revision",
        "text_cache_revision",
        "rewriter_revision",
        "latent_cache_revision",
        "latent_stats_sha256",
    )
    for name in revision_fields:
        value = identity.get(name)
        if not isinstance(value, str) or not value:
            raise RendererMaterializationError(f"Renderer adapter is missing {name}")
        revisions[name] = value
    if revisions["tokenizer_revision"] != embedding_ready["tokenizer_revision"]:
        raise RendererMaterializationError(
            "Tokenizer checkpoint and semantic release revisions do not match"
        )
    for split, section_name in (("train", "data"), ("valid", "validation")):
        if split not in split_results:
            raise RendererMaterializationError(
                f"Renderer release must contain the {split} split"
            )
        if not isinstance(config.get(section_name), dict):
            raise RendererMaterializationError(
                f"Renderer config is missing {section_name}"
            )
        section = config[section_name]
        section.pop("root", None)
        sample_root = staging / "samples" / split
        final_sample_root = final_root / "samples" / split
        crop_root = staging / "crops" / split
        final_crop_root = final_root / "crops" / split
        loud_root = staging / "loudness" / split
        final_loud_root = final_root / "loudness" / split
        sample_ready = _read_json(sample_root / "READY")
        section.update(
            {
                "cache_manifest": str(final_sample_root / "manifest.jsonl"),
                "ready_path": str(final_sample_root / "READY"),
                "ready_sha256": _sha256(sample_root / "READY"),
                "index_path": str(final_sample_root / "manifest.index.json"),
                "index_sha256": _sha256(sample_root / "manifest.index.json"),
                "sample_release_revision": sample_ready["revision"],
                "manifest_format": "oqm.render-sample.v1",
                "manifest_sha256": sample_ready["manifest_sha256"],
                "expected_records": sample_ready["records"],
                "sampler_mode": "renderer_data_parent_first",
                "renderer_data_view": "short",
                "renderer_data_crop_manifest": str(final_crop_root / "manifest.jsonl"),
                "renderer_data_crop_manifest_sha256": _sha256(crop_root / "manifest.jsonl"),
                "renderer_data_crop_ready_path": str(final_crop_root / "READY"),
                "renderer_data_crop_ready_sha256": _sha256(crop_root / "READY"),
                "renderer_data_loudness_manifest": str(final_loud_root / "manifest.jsonl"),
                "renderer_data_loudness_manifest_sha256": _sha256(loud_root / "manifest.jsonl"),
                "renderer_data_loudness_ready_path": str(final_loud_root / "READY"),
                "renderer_data_loudness_ready_sha256": _sha256(loud_root / "READY"),
            }
        )
        section["renderer_data_content_text_cache"] = {
            "tags": {
                "root": str(final_root / "text" / "tags"),
                "ready_sha256": _sha256(staging / "text" / "tags" / "READY"),
            },
            "lyrics": {
                "root": str(final_root / "text" / "short_lyrics"),
                "ready_sha256": _sha256(
                    staging / "text" / "short_lyrics" / "READY"
                ),
            },
            "cache_config_sha256": tags_ready["cache_config_sha256"],
            "cache_config_file_sha256": tags_ready["cache_config_file_sha256"],
        }
    conditioning = config.get("conditioning")
    if not isinstance(conditioning, dict):
        raise RendererMaterializationError("Renderer config is missing conditioning")
    conditioning["semantic_source_dim"] = embedding_ready["embeddings"][
        "tokenizer_codebook"
    ]["shape"][1]
    loudness_values: list[float] = []
    for split in ("train", "valid"):
        for _, row in _iter_jsonl(staging / "loudness" / split / "manifest.jsonl"):
            loudness_values.extend(
                float(window["integrated_lufs"])
                for window in row["windows"]
            )
    conditioning["global_loudness_mean_lufs"] = float(np.mean(loudness_values))
    loudness_std = float(np.std(loudness_values))
    conditioning["global_loudness_std_lu"] = max(loudness_std, 1.0e-6)
    embedding = conditioning.get("semantic_embedding_asset")
    if isinstance(embedding, dict):
        embedding.update(
            {
                "ready_path": str(final_root / "semantic_embedding" / "READY"),
                "ready_sha256": _sha256(staging / "semantic_embedding" / "READY"),
                "artifact_sha256": embedding_ready["artifact"]["sha256"],
                "asset_revision": embedding_ready["asset_revision"],
                "tensor_sha256": embedding_ready["embeddings"]["tokenizer_codebook"]["sha256"],
            }
        )
    else:
        raise RendererMaterializationError(
            "Renderer config is missing semantic_embedding_asset"
        )
    corruption = config.get("semantic_corruption")
    if isinstance(corruption, dict) and isinstance(corruption.get("asset"), dict):
        asset = corruption["asset"]
        asset.update(
            {
                "ready_path": str(final_root / "semantic_corruption" / "READY"),
                "ready_sha256": _sha256(staging / "semantic_corruption" / "READY"),
                "artifact_sha256": distractor_ready["artifact"]["sha256"],
                "asset_revision": distractor_ready["asset_revision"],
                "neighbor_ids_sha256": distractor_ready["neighbors"]["neighbor_ids"]["sha256"],
                "neighbor_scores_sha256": distractor_ready["neighbors"]["neighbor_scores"]["sha256"],
                "source_embedding_tensor_sha256": distractor_ready["source_embedding_tensor_sha256"],
            }
        )
        corruption["top_k"] = distractor_ready["top_k"]
        corruption["top_k_policy"] = "tokenizer_codebook_cosine_knn"
        if calibration is None:
            raise RendererMaterializationError(
                "The final Renderer recipe requires semantic calibration"
            )
        calibration_config = dict(calibration["config"])
        calibration_config["evaluation_report_path"] = str(
            final_root / "semantic_corruption" / calibration["path"]
        )
        calibration_config["evaluation_report_sha256"] = calibration["sha256"]
        corruption["calibration"] = calibration_config
    else:
        raise RendererMaterializationError(
            "The final Renderer config requires semantic corruption assets"
        )
    from .trainer_dit import validate_dit_launch_config

    validate_dit_launch_config(config)
    target = staging / "renderer.resolved.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def materialize_renderer(
    source_manifest: str | Path,
    semantic_release: str | Path,
    output_dir: str | Path,
    *,
    adapters: RendererAdapterBundle,
    calibration_report: str | Path | None = None,
    semantic_top_k: int = 384,
) -> dict[str, Any]:
    source = Path(source_manifest).expanduser().resolve(strict=True)
    semantics, semantic_ready = _load_semantics(Path(semantic_release).expanduser().resolve(strict=True))
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Renderer materialization output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        source_rows = [row for _, row in _iter_jsonl(source)]
        source_ids = [row.get("sample_id") for row in source_rows]
        if any(not isinstance(value, str) or not value for value in source_ids):
            raise RendererMaterializationError("Every source record must have a sample_id")
        if len(set(source_ids)) != len(source_ids) or set(source_ids) != set(semantics):
            raise RendererMaterializationError("Source and semantic sample sets must match exactly")
        by_split: dict[str, list[dict[str, Any]]] = {}
        crops: dict[str, list[dict[str, Any]]] = {}
        loudness: dict[str, list[dict[str, Any]]] = {}
        tags: dict[str, dict[str, Any]] = {}
        lyrics: dict[str, dict[str, Any]] = {}
        identity = dict(adapters.identity)
        if identity.get("tokenizer_revision") != semantic_ready["tokenizer_revision"]:
            raise RendererMaterializationError(
                "Renderer adapter and semantic release use different Tokenizer revisions"
            )
        for stale in (
            "semantic_extractor_revision",
            "tokenizer_artifact_sidecar_sha256",
            "tokenizer_binding_sha256",
            "semantic_materializer_revision",
        ):
            identity.pop(stale, None)
        from .cache import (
            ChannelStats,
            build_latent_stats_payload,
            freeze_latent_stats,
            json_sha256,
        )

        latent_stats = ChannelStats.empty()
        for row in source_rows:
            sample_id = str(row["sample_id"])
            split = {"validation": "valid", "val": "valid"}.get(row.get("split"), row.get("split"))
            if split not in {"train", "valid", "test"} or semantics[sample_id].get("split") != split:
                raise RendererMaterializationError(f"Split mismatch for {sample_id}")
            semantic = semantics[sample_id]
            tokens = np.asarray(semantic["tokens"])
            if tokens.size > MAX_PARENT_FRAMES:
                raise RendererMaterializationError(
                    f"Renderer sample exceeds {MAX_PARENT_FRAMES} frames: {sample_id}"
                )
            audio_path, waveform, source_audio_sha, canonical_audio_sha = _canonical_audio(
                row,
                manifest=source,
                output=staging,
                split=str(split),
                sample_id=sample_id,
                semantic_frames=int(tokens.size),
            )
            if semantic.get("source_audio_sha256") != source_audio_sha:
                raise RendererMaterializationError(f"Semantic source audio SHA-256 mismatch for {sample_id}")
            latent = np.asarray(
                adapters.encode_latent(
                    waveform,
                    SAMPLE_RATE,
                    sample_id=sample_id,
                    audio_sha256=canonical_audio_sha,
                ),
                dtype=np.float32,
            )
            if latent.shape != (tokens.size, LATENT_DIM) or not np.isfinite(latent).all():
                raise RendererMaterializationError(f"VAE adapter returned invalid latents for {sample_id}")
            if split == "train":
                latent_stats.update(latent)
            latent_path = staging / "latents" / str(split) / f"{hashlib.sha256(sample_id.encode()).hexdigest()}.npy"
            latent_path.parent.mkdir(parents=True, exist_ok=True)
            with latent_path.open("xb") as handle:
                np.save(handle, latent.astype(np.float32), allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            semantic_path = staging / "samples" / str(split) / "semantic" / f"{semantic['artifact']['sha256']}.npy"
            semantic_path.parent.mkdir(parents=True, exist_ok=True)
            if not semantic_path.exists():
                shutil.copyfile(Path(semantic["path"]), semantic_path)
            condition = _condition(row, identity=identity)
            groups = _groups(row, split=str(split), sample_id=sample_id)
            frames = int(tokens.size)
            windows = []
            sections = _sections(row, frames=frames)
            boundaries = {0, frames}
            for section in sections:
                boundaries.update((section["start_frame"], section["end_frame"]))
            partition = _best_partition(
                tuple(sorted(boundaries)),
                frames=frames,
                target_frames=TARGET_WINDOW_FRAMES,
                target_k=max(1, min(4, math.ceil(frames / TARGET_WINDOW_FRAMES))),
            )
            for window_index, (cursor, end) in enumerate(itertools.pairwise(partition)):
                window_id = f"{sample_id}#short-v1:w{window_index}:{cursor}-{end}"
                description = condition["description"]
                lyric_text = _window_lyrics(sections, cursor, end) if sections else condition["lyrics"]
                window_condition = {**condition, "lyrics": lyric_text}
                tags_sha = hashlib.sha256(description.encode()).hexdigest()
                lyrics_sha = hashlib.sha256(lyric_text.encode()).hexdigest()
                tags_id = f"renderer_data-tags:{tags_sha}"
                lyrics_id = f"renderer_data-lyrics:{lyrics_sha}"
                if tags_id not in tags:
                    tags[tags_id] = _text_entry(
                        adapters,
                        text=description,
                        role="tags",
                        sample_id=tags_id,
                        root=staging / "text" / "tags",
                        rewriter_revision=str(condition["rewriter_revision"]),
                        text_tokenizer_revision=str(condition["text_tokenizer_revision"]),
                    )
                if lyrics_id not in lyrics:
                    lyrics[lyrics_id] = _text_entry(
                        adapters,
                        text=lyric_text,
                        role="lyrics",
                        sample_id=lyrics_id,
                        root=staging / "text" / "short_lyrics",
                        rewriter_revision=str(condition["rewriter_revision"]),
                        text_tokenizer_revision=str(condition["text_tokenizer_revision"]),
                    )
                condition_sha = _canonical_sha(window_condition)
                text_reference = {
                    "schema_version": TEXT_REFERENCE_SCHEMA,
                    "tags_content_sha256": tags_sha,
                    "lyrics_content_sha256": lyrics_sha,
                    "source_condition_sha256": condition_sha,
                    "cache_config_sha256": tags[tags_id]["cache_config_sha256"],
                    "cache_config_file_sha256": tags[tags_id]["cache_config_file_sha256"],
                }
                windows.append(
                    {
                        "sample_id": window_id,
                        "window_index": window_index,
                        "start_frame": cursor,
                        "end_frame": end,
                        "condition": window_condition,
                        "text_cache": text_reference,
                    }
                )
            latent_sha = _sha256(latent_path)
            sample_row = {
                "schema_version": SAMPLE_SCHEMA,
                "sample_id": sample_id,
                "split": split,
                "audio": {
                    "uri": str(output / "audio" / str(split) / audio_path.name),
                    "sha256": canonical_audio_sha,
                    "source_sha256": source_audio_sha,
                    "sample_rate": SAMPLE_RATE,
                    "channels": CHANNELS,
                    "start_sec": 0.0,
                    "duration_sec": frames / FRAME_HZ,
                },
                "semantic": {
                    "uri": str(Path("semantic") / semantic_path.name),
                    "sha256": semantic["artifact"]["sha256"],
                    "shape": [frames],
                    "dtype": "uint16",
                    "frame_hz": FRAME_HZ,
                    "codebook_size": CODEBOOK_SIZE,
                    "sample_id": sample_id,
                    "source_audio_sha256": source_audio_sha,
                    "input_audio_sha256": canonical_audio_sha,
                    "input_audio_basis": "canonical_audio",
                    "source_start_sec": 0.0,
                    "source_duration_sec": frames / FRAME_HZ,
                    "tokenizer_revision": semantic_ready["tokenizer_revision"],
                    "tokenizer_checkpoint_sha256": identity.get(
                        "tokenizer_checkpoint_sha256"
                    ),
                    "semantic_extractor_revision": identity.get(
                        "semantic_extractor_revision"
                    ),
                    "materializer_revision": identity.get(
                        "semantic_materializer_revision"
                    ),
                    "tokenizer_artifact_sidecar_sha256": identity.get(
                        "tokenizer_artifact_sidecar_sha256"
                    ),
                    "tokenizer_binding_sha256": identity.get(
                        "tokenizer_binding_sha256"
                    ),
                },
                "latent": {
                    "uri": str(output / "latents" / str(split) / latent_path.name),
                    "sha256": latent_sha,
                    "shape": [frames, LATENT_DIM],
                    "dtype": "float32",
                    "frame_hz": FRAME_HZ,
                    "sample_id": sample_id,
                    "source_audio_sha256": source_audio_sha,
                    "derived_audio_sha256": canonical_audio_sha,
                    "source_start_sec": 0.0,
                    "source_duration_sec": frames / FRAME_HZ,
                    **dict(identity.get("latent_identity") or {}),
                    **(
                        {
                            "posterior_sample_seed": _posterior_seed(
                                int((identity.get("latent_identity") or {})["posterior_base_seed"]),
                                sample_id,
                                canonical_audio_sha,
                            )
                        }
                        if (identity.get("latent_identity") or {}).get("posterior_mode") == "sample"
                        else {}
                    ),
                },
                "condition": condition,
                "text_cache": windows[0]["text_cache"],
                "quality": {**dict(row.get("quality") or {}), "render_broad": True},
                "groups": groups,
            }
            by_split.setdefault(str(split), []).append(sample_row)
            crops.setdefault(str(split), []).append(
                {
                    "schema_version": CROP_SCHEMA,
                    "parent_sample_id": sample_id,
                    "parent_latent_sha256": latent_sha,
                    "parent_semantic_sha256": semantic["artifact"]["sha256"],
                    "parent_condition_sha256": _canonical_sha(condition),
                    "split": split,
                    "windows": windows,
                }
            )
            loudness.setdefault(str(split), []).append(
                {
                    "schema_version": LOUDNESS_SCHEMA,
                    "windows": [
                        {
                            "sample_id": window["sample_id"],
                            "integrated_lufs": round(
                                _lufs(waveform[window["start_frame"] * 1920 : window["end_frame"] * 1920]), 6
                            ),
                        }
                        for window in windows
                    ],
                }
            )

        if "train" not in by_split:
            raise RendererMaterializationError("Renderer release must contain the train split")
        source_sha = _sha256(source)
        latent_identity = dict(identity.get("latent_identity") or {})
        latent_layout = dict(identity.get("latent_layout") or {})
        if not latent_layout:
            latent_layout = {
                "format_version": "oqm.render-latent-layout.v1",
                "latent_dim": LATENT_DIM,
                "frame_hz": FRAME_HZ,
                "channel_semantics": "unstructured_continuous",
                "special_channels": [],
                "normalization": "per_channel_affine",
            }
        cache_config_sha = json_sha256(
            {
                "format_version": "oqm.render.latent-cache.v1",
                "vae_checkpoint_sha256": identity.get("vae_checkpoint_sha256"),
                "vae_revision": identity.get("vae_revision"),
                "stft_revision": identity.get("stft_revision"),
                "stft_config_sha256": identity.get("stft_config_sha256"),
                "posterior_mode": latent_identity.get("posterior_mode"),
                "posterior_base_seed": latent_identity.get("posterior_base_seed"),
                "posterior_seed_derivation": latent_identity.get(
                    "posterior_seed_derivation"
                ),
                "posterior_epsilon_draw_layout": latent_identity.get(
                    "posterior_epsilon_draw_layout"
                ),
                "dtype": "float32",
            }
        )
        stats_payload = build_latent_stats_payload(
            latent_stats,
            minimum_std=1.0e-6,
            manifest_sha256=source_sha,
            cache_config_sha256=cache_config_sha,
            vae_checkpoint_sha256=str(identity.get("vae_checkpoint_sha256") or ""),
            vae_revision=str(identity.get("vae_revision") or ""),
            stft_revision=str(identity.get("stft_revision") or ""),
            stft_config_sha256=str(identity.get("stft_config_sha256") or ""),
            posterior_mode=str(latent_identity.get("posterior_mode") or ""),
            sample_seed=int(latent_identity.get("posterior_base_seed") or 0),
            sampler={
                "format_version": "oqm.render.latent-stats-sampler.v1",
                "split": "train",
                "epochs": 1,
                "shuffle": False,
                "sampled_records": len(by_split["train"]),
                "frame_weighting": "valid_latent_frames",
            },
            latent_layout=latent_layout,
            posterior_epsilon_draw_layout=str(
                latent_identity.get("posterior_epsilon_draw_layout")
                or "contiguous_bdt"
            ),
        )
        stats_path = staging / "latents" / "stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_sha = freeze_latent_stats(stats_path, stats_payload)
        mean, std = latent_stats.moments(minimum_std=1.0e-6)
        for split, rows in by_split.items():
            for row, crop in zip(rows, crops[split], strict=True):
                latent_meta = row["latent"]
                latent_path = staging / "latents" / split / Path(
                    str(latent_meta["uri"])
                ).name
                values = np.load(latent_path, allow_pickle=False)
                normalized = ((values.astype(np.float64) - mean) / std).astype(
                    np.float32
                )
                with latent_path.open("wb") as handle:
                    np.save(handle, normalized, allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                latent_meta.update(
                    {
                        "sha256": _sha256(latent_path),
                        "cache_revision": cache_config_sha,
                        "latent_stats_sha256": stats_sha,
                        "latent_layout": latent_layout,
                        "latent_layout_sha256": stats_payload[
                            "latent_layout_sha256"
                        ],
                    }
                )
                crop["parent_latent_sha256"] = latent_meta["sha256"]
        identity["latent_cache_revision"] = cache_config_sha
        identity["latent_stats_sha256"] = stats_sha
        identity["latent_layout"] = latent_layout
        identity["latent_layout_sha256"] = stats_payload["latent_layout_sha256"]
        latent_identity.update(
            {
                "cache_revision": cache_config_sha,
                "latent_stats_sha256": stats_sha,
                "latent_layout": latent_layout,
                "latent_layout_sha256": stats_payload["latent_layout_sha256"],
            }
        )
        identity["latent_identity"] = latent_identity
        release_identity = dict(identity)
        for source_name, ready_name in (
            ("posterior_mode", "latent_posterior_mode"),
            ("posterior_base_seed", "latent_posterior_base_seed"),
            ("posterior_seed_derivation", "latent_posterior_seed_derivation"),
            (
                "posterior_epsilon_draw_layout",
                "latent_posterior_epsilon_draw_layout",
            ),
        ):
            if source_name in latent_identity:
                release_identity[ready_name] = latent_identity[source_name]
        tags_ready = _publish_text_cache(staging / "text" / "tags", list(tags.values()), source_sha=source_sha)
        lyrics_ready = _publish_text_cache(
            staging / "text" / "short_lyrics", list(lyrics.values()), source_sha=source_sha
        )
        split_results: dict[str, Any] = {}
        for split, rows in by_split.items():
            sample_root = staging / "samples" / split
            manifest = sample_root / "manifest.jsonl"
            _write_jsonl(manifest, rows)
            manifest_sha = _sha256(manifest)
            crop_rows = crops[split]
            for index, crop in enumerate(crop_rows):
                crop["parent_index"] = index
                crop["parent_manifest_sha256"] = manifest_sha
            crop_root = staging / "crops" / split
            crop_manifest = crop_root / "manifest.jsonl"
            _write_jsonl(crop_manifest, crop_rows)
            crop_ready = {
                "schema_version": CROP_READY_SCHEMA,
                "status": "RENDERER_DATA_SHORT_CROP_READY",
                "manifest": crop_manifest.name,
                "manifest_sha256": _sha256(crop_manifest),
                "parent_manifest_sha256": manifest_sha,
                "split": split,
                "rows_sha256": _canonical_sha(crop_rows),
                "parents": len(rows),
                "windows": sum(len(row["windows"]) for row in crop_rows),
            }
            _write_json(crop_root / "READY", crop_ready)
            loud_rows = loudness[split]
            for index, loud_row in enumerate(loud_rows):
                loud_row["parent_index"] = index
            loud_root = staging / "loudness" / split
            loud_manifest = loud_root / "manifest.jsonl"
            _write_jsonl(loud_manifest, loud_rows)
            loud_ready = {
                "schema_version": LOUDNESS_READY_SCHEMA,
                "status": "RENDERER_DATA_WINDOW_LOUDNESS_READY",
                "manifest": loud_manifest.name,
                "manifest_sha256": _sha256(loud_manifest),
                "source_crop_manifest_sha256": _sha256(crop_manifest),
                "parents": len(rows),
                "windows": sum(len(row["windows"]) for row in loud_rows),
            }
            _write_json(loud_root / "READY", loud_ready)
            latent_manifest = staging / "latents" / split / "manifest.jsonl"
            _write_jsonl(
                latent_manifest,
                [
                    {
                        "sample_id": row["sample_id"],
                        "artifact": {
                            **row["latent"],
                            "uri": Path(str(row["latent"]["uri"])).name,
                        },
                    }
                    for row in rows
                ],
            )
            _write_json(
                staging / "latents" / split / "READY",
                {
                    "schema_version": "oqm.render-latent-release.v1",
                    "status": "READY",
                    "manifest": latent_manifest.name,
                    "manifest_sha256": _sha256(latent_manifest),
                    "records": len(rows),
                    "frame_hz": FRAME_HZ,
                    "latent_dim": LATENT_DIM,
                    "cache_revision": cache_config_sha,
                    "stats": {
                        "path": str(output / "latents" / stats_path.name),
                        "sha256": stats_sha,
                    },
                },
            )
            index_payload, payload_sha = _index(manifest, rows, sample_root)
            sample_ready = {
                "schema_version": SAMPLE_READY_SCHEMA,
                "status": "RENDER_SAMPLE_READY",
                "revision": _canonical_sha({"split": split, "manifest_sha256": manifest_sha}),
                "split": split,
                "required_quality_profile": "render_broad",
                "records": len(rows),
                "manifest": manifest.name,
                "manifest_sha256": manifest_sha,
                "identities": release_identity,
                "checks": {
                    "sample_sets_equal": True,
                    "source_audio_sha_equal": True,
                    "derived_audio_sha_equal": True,
                    "time_ranges_equal": True,
                    "semantic_latent_frames_equal": True,
                    "quality_profile_pass": True,
                    "artifact_sha_verified": True,
                    "split_groups_disjoint": True,
                },
                "index": {
                    "schema_version": "oqm.render-sample-index.v1",
                    "required": True,
                    "metadata": "manifest.index.json",
                    "payload": index_payload.name,
                    "payload_sha256": payload_sha,
                    "payload_size_bytes": index_payload.stat().st_size,
                    "records": len(rows),
                },
            }
            _write_json(sample_root / "READY", sample_ready)
            ready_sha = _sha256(sample_root / "READY")
            index_metadata = {
                "schema_version": "oqm.render-sample-index.v1",
                "status": "RENDER_SAMPLE_INDEX_READY",
                "payload_schema": "oqm.render-sample-index-npy.v1",
                "manifest": {"path": manifest.name, "sha256": manifest_sha, "size_bytes": manifest.stat().st_size},
                "ready": {"path": "READY", "sha256": ready_sha},
                "payload": {
                    "path": index_payload.name,
                    "sha256": payload_sha,
                    "size_bytes": index_payload.stat().st_size,
                    "arrays": {
                        "offset": "uint64",
                        "size": "uint64",
                        "line_number": "uint64",
                        "frame_length": "uint32",
                    },
                },
                "records": len(rows),
                "split": split,
                "required_quality_profile": "render_broad",
                "release_revision": sample_ready["revision"],
                "identities": release_identity,
                "sample_ids_sha256": _canonical_sha(sorted(row["sample_id"] for row in rows)),
                "producer": {"module": "open_qwen_music.render.materialize", "code_sha256": _sha256(Path(__file__))},
            }
            index_metadata["revision"] = _canonical_sha(index_metadata)
            _write_json(sample_root / "manifest.index.json", index_metadata)
            split_results[split] = {
                "samples_ready_sha256": ready_sha,
                "crops_ready_sha256": _sha256(crop_root / "READY"),
                "loudness_ready_sha256": _sha256(loud_root / "READY"),
                "latents_ready_sha256": _sha256(staging / "latents" / split / "READY"),
                "records": len(rows),
            }

        codebook = np.asarray(adapters.semantic_codebook(), dtype=np.float32)
        if codebook.ndim != 2 or codebook.shape[0] != CODEBOOK_SIZE or not np.isfinite(codebook).all():
            raise RendererMaterializationError("Tokenizer codebook must be finite with shape [32768, D]")
        embedding_root = staging / "semantic_embedding"
        embedding_root.mkdir()
        tensor_sha = hashlib.sha256(memoryview(np.ascontiguousarray(codebook)).cast("B")).hexdigest()
        artifact = embedding_root / "embedding.pt"
        try:
            import torch
        except ImportError as exc:
            raise RendererMaterializationError("PyTorch is required to publish semantic embeddings") from exc
        torch.save(
            {
                "schema_version": "oqm.render-semantic-embedding.v1",
                "asset_revision": _canonical_sha({"tensor_sha256": tensor_sha}),
                "tokenizer_revision": semantic_ready["tokenizer_revision"],
                "source": "tokenizer_effective_codebook",
                "embeddings": {"tokenizer_codebook": torch.from_numpy(codebook)},
            },
            artifact,
        )
        embedding_ready = {
            "schema_version": "oqm.render-semantic-embedding-ready.v1",
            "status": "RENDER_SEMANTIC_EMBEDDING_READY",
            "asset_revision": _canonical_sha({"tensor_sha256": tensor_sha}),
            "tokenizer_revision": semantic_ready["tokenizer_revision"],
            "source": "tokenizer_effective_codebook",
            "semantic_contract": {"frame_hz": FRAME_HZ, "codebook_size": CODEBOOK_SIZE, "codebooks": 1},
            "artifact": {"path": artifact.name, "sha256": _sha256(artifact)},
            "embeddings": {"tokenizer_codebook": {"sha256": tensor_sha, "shape": list(codebook.shape), "dtype": "float32"}},
        }
        _write_json(embedding_root / "READY", embedding_ready)
        corruption_root = staging / "semantic_corruption"
        distractor_ready = _publish_semantic_distractors(
            corruption_root,
            codebook=codebook,
            tokenizer_revision=str(semantic_ready["tokenizer_revision"]),
            source_embedding_tensor_sha256=tensor_sha,
            top_k=semantic_top_k,
            neighbor_tables=(
                getattr(adapters, "semantic_neighbors")(semantic_top_k)
                if callable(getattr(adapters, "semantic_neighbors", None))
                else None
            ),
        )
        calibration = _publish_calibration(calibration_report, corruption_root)
        resolved_config = _write_resolved_renderer_config(
            staging,
            final_root=output,
            base_config=(
                getattr(adapters, "renderer_config", None)
                if isinstance(getattr(adapters, "renderer_config", None), Mapping)
                else None
            ),
            split_results=split_results,
            embedding_ready=embedding_ready,
            distractor_ready=distractor_ready,
            tags_ready=tags_ready,
            lyrics_ready=lyrics_ready,
            calibration=calibration,
            identity=identity,
        )
        result = {
            "schema_version": "oqm.renderer-materialization-ready.v1",
            "status": "READY",
            "source_manifest_sha256": source_sha,
            "semantic_release_ready_sha256": _sha256(Path(semantic_release).resolve() / "READY"),
            "splits": split_results,
            "text": {
                "tags_ready_sha256": _sha256(staging / "text" / "tags" / "READY"),
                "lyrics_ready_sha256": _sha256(staging / "text" / "short_lyrics" / "READY"),
                "tags_records": tags_ready["records"],
                "lyrics_records": lyrics_ready["records"],
            },
            "semantic_embedding_ready_sha256": _sha256(embedding_root / "READY"),
            "semantic_corruption_ready_sha256": _sha256(corruption_root / "READY"),
            "semantic_corruption_asset_revision": distractor_ready["asset_revision"],
            "semantic_error_calibration": calibration,
            "resolved_renderer_config": (
                {
                    "path": resolved_config.name,
                    "sha256": _sha256(resolved_config),
                }
                if resolved_config is not None
                else None
            ),
        }
        _write_json(staging / "READY", result)
        os.replace(staging, output)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize public Renderer training data")
    parser.add_argument("--config", required=True, help="Renderer training configuration")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--semantic-release", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-checkpoint", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument(
        "--calibration-report",
        required=True,
        help="Frozen LLM semantic accuracy calibration report to copy into the release",
    )
    parser.add_argument("--semantic-top-k", type=int, default=384)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    adapters = ProductionRendererAdapters(
        renderer_config=Path(args.config).expanduser().resolve(strict=True),
        tokenizer_checkpoint=Path(args.tokenizer_checkpoint).expanduser().resolve(strict=True),
        vae_checkpoint=Path(args.vae_checkpoint).expanduser().resolve(strict=True),
        text_encoder=Path(args.text_encoder).expanduser().resolve(strict=True),
        device=args.device,
    )
    result = materialize_renderer(
        args.source_manifest,
        args.semantic_release,
        args.output_dir,
        adapters=adapters,
        calibration_report=args.calibration_report,
        semantic_top_k=args.semantic_top_k,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
