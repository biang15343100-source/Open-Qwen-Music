"""Run the Open-Qwen-Music community inference pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

import torch

from open_qwen_music.common.config import load_config
from open_qwen_music.llm.inference import run_generation
from open_qwen_music.llm.render_contract import read_render_requests, semantic_token_sha256
from open_qwen_music.render.cli import _publish_render_output
from open_qwen_music.render.conditioning import REWRITER_SCHEMA_VERSION
from open_qwen_music.release import (
    DEFAULT_REPO_ID,
    DEFAULT_REVISION,
    OpenQwenMusicWeightBundle,
)


PIPELINE_FORMAT = "oqm.inference.v2"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _resolve(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def run_pipeline(
    *,
    config_path: str | Path,
    prompts_path: str | Path,
    output_dir: str | Path,
    model_source: str | Path | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    text_encoder_source: str | Path | None = None,
    device: str | None = None,
    seed: int = 0,
) -> Path:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    if config.get("format_version") != PIPELINE_FORMAT:
        raise ValueError(f"Inference configuration must be {PIPELINE_FORMAT}")

    model_section = _mapping(config.get("model"), name="model")
    bundle = OpenQwenMusicWeightBundle.from_pretrained(
        model_source or str(model_section.get("repo_id") or DEFAULT_REPO_ID),
        revision=revision or str(model_section.get("revision") or DEFAULT_REVISION),
        cache_dir=cache_dir,
    )
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    llm = _mapping(config.get("llm"), name="llm")
    if llm.get("mode") != "plain":
        raise ValueError("The published language model uses plain sequence generation")
    temperature = float(llm.get("temperature", 0.7))
    if temperature <= 0:
        raise ValueError("Inference temperature must be positive")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    semantic_npz = output_dir / "semantic.npz"
    requests_path = output_dir / "render-requests.jsonl"
    llm_config = load_config(_resolve(config_path, llm["config"]))
    llm_dir = bundle.component_dir("language_model")
    llm_config.setdefault("model", {})["tokenizer_path"] = str(llm_dir)
    llm_config["model"]["base_model_path"] = str(llm_dir)

    from open_qwen_music.llm.trainer import resolve_registry

    registry = resolve_registry(llm_config)
    precision = str(llm_config.get("generation", {}).get("precision", "bf16"))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        precision
    ]
    language_model = bundle.load_language_model(
        registry,
        device=target_device,
        dtype=dtype,
        model_options=llm_config.get("model"),
    )
    run_generation(
        llm_config,
        checkpoint=llm_dir,
        model=language_model,
        semantic_tokenizer_revision=bundle.semantic_tokenizer_revision,
        prompts_path=prompts_path,
        output_path=semantic_npz,
        mode="plain",
        temperature=temperature,
        top_p=float(llm.get("top_p", 0.98)),
        semantic_repetition_penalty=float(llm.get("semantic_repetition_penalty", 0.1)),
        semantic_repetition_window=int(llm.get("semantic_repetition_window", 64)),
        seed=seed,
        batch_size=int(llm.get("batch_size", 1)),
        strict_mode=True,
        device=str(target_device),
        render_requests_path=requests_path,
    )

    if text_encoder_source is None:
        text_encoder = bundle.resolve_text_encoder(cache_dir=cache_dir)
    else:
        text_encoder = Path(text_encoder_source).expanduser().resolve()
        if not text_encoder.is_dir():
            raise FileNotFoundError(f"Text encoder directory not found: {text_encoder}")
    renderer, preset = bundle.load_render_pipeline(
        device=target_device,
        text_encoder_dir=text_encoder,
    )
    requests = read_render_requests(requests_path)
    render_section = _mapping(config.get("render"), name="render")
    manifest: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        semantic_ids = torch.as_tensor(
            request.semantic_ids, dtype=torch.long, device=target_device
        ).unsqueeze(0)
        semantic_mask = torch.ones_like(semantic_ids, dtype=torch.bool)
        result = renderer.render(
            semantic_ids,
            description=[request.condition.description or request.condition.render_tags()],
            lyrics=[request.condition.render_lyrics()],
            semantic_mask=semantic_mask,
            duration_seconds=[request.semantic_ids.size / 25.0],
            global_loudness_lufs=float(render_section["global_loudness_lufs"]),
            tokenizer_revision=request.semantic_tokenizer_revision,
            rewriter_revision=renderer.revisions.rewriter_revision,
            rewriter_schema_version=REWRITER_SCHEMA_VERSION,
            seed=seed + index,
            solver=str(preset["solver"]),
            num_steps=int(preset["num_steps"]),
            cfg_scale=float(preset["cfg_scale"]),
            use_refiner=bool(preset["use_refiner"]),
            input_artifact_identities={
                "kind": "render_request",
                "semantic_sha256": semantic_token_sha256(request.semantic_ids),
                "request": str(requests_path),
            },
        )
        name = _SAFE_NAME.sub("_", request.sample_id).strip("._") or f"sample-{index:04d}"
        destination = output_dir / f"{name}.wav"
        metadata = _publish_render_output(destination, output=result, preset=preset)
        manifest.append(
            {
                "sample_id": request.sample_id,
                "output": str(destination),
                "metadata": str(destination) + ".json",
                "sha256": metadata["output"]["sha256"],
            }
        )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path
