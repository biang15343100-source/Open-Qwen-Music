from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from open_qwen_music.common.checkpoint import (
    checkpoint_runtime_identity,
    file_sha256,
    load_checkpoint,
)
from open_qwen_music.common.config import load_config

from .data import TokenizerDataset
from .model import MusicTokenizer


INITIALIZATION_FORMAT = "oqm.tokenizer.stage4-initialization.v1"
INITIALIZATION_ROLE = "tokenizer-stage4-data-dependent-initialization"


def spherical_kmeans(
    samples: torch.Tensor,
    *,
    clusters: int,
    iterations: int,
    chunk_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if samples.ndim != 2 or samples.shape[0] < clusters:
        raise ValueError(
            f"K-means requires at least {clusters} two-dimensional samples; "
            f"received {tuple(samples.shape)}"
        )
    if iterations < 1 or chunk_size < 1:
        raise ValueError("iterations and chunk_size must be positive")
    samples = F.normalize(samples.float(), dim=-1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    initial = torch.randperm(samples.shape[0], generator=generator)[:clusters]
    centroids = samples[initial.to(samples.device)].clone()
    counts = torch.zeros(clusters, device=samples.device, dtype=torch.float32)
    for iteration in range(iterations):
        sums = torch.zeros_like(centroids)
        counts.zero_()
        for start in range(0, samples.shape[0], chunk_size):
            batch = samples[start : start + chunk_size]
            assignments = (batch @ centroids.T).argmax(dim=-1)
            sums.index_add_(0, assignments, batch)
            counts.index_add_(
                0,
                assignments,
                torch.ones(assignments.shape[0], device=samples.device),
            )
        populated = counts > 0
        centroids[populated] = F.normalize(sums[populated], dim=-1)
        empty = (~populated).nonzero(as_tuple=False).flatten()
        if empty.numel():
            offset = (iteration * clusters) % samples.shape[0]
            replacements = (
                torch.arange(empty.numel(), device=samples.device) + offset
            ) % samples.shape[0]
            centroids[empty] = samples[replacements]
    return centroids, counts


def _checkpoint_phase(state: dict[str, Any]) -> str:
    config = state.get("config") or {}
    return str(config.get("phase") or "")


def validate_initialization_artifact(
    path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    artifact_path = Path(path)
    state = torch.load(
        artifact_path, map_location="cpu", weights_only=False, mmap=True
    )
    if state.get("format_version") != INITIALIZATION_FORMAT:
        raise ValueError(
            "Tokenizer Stage 4 requires a data-dependent initialization artifact; "
            f"received format={state.get('format_version')!r}"
        )
    if state.get("artifact_role") != INITIALIZATION_ROLE:
        raise ValueError("Tokenizer Stage 4 initialization artifact has the wrong role")
    contract = state.get("initialization_contract")
    if not isinstance(contract, dict):
        raise ValueError("Tokenizer Stage 4 initialization artifact has no contract")
    quantizer = config["quantizer"]
    expected = {
        "source_phase": "stage3",
        "sample_frames": int(quantizer["init_sample_frames"]),
        "kmeans_iterations": int(quantizer["init_kmeans_iterations"]),
        "codebook_size": int(config["semantic_contract"]["codebook_size"]),
        "code_dim": int(quantizer["code_dim"]),
        "seed": int(config["train"]["seed"]),
    }
    mismatches = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Tokenizer Stage 4 initialization contract mismatch: {mismatches}"
        )
    source_sha = str(contract.get("source_checkpoint_sha256") or "")
    if len(source_sha) != 64 or any(value not in "0123456789abcdef" for value in source_sha):
        raise ValueError("Tokenizer Stage 4 initialization source checkpoint hash is invalid")
    manifest = Path(str(quantizer.get("initialization_manifest") or ""))
    if not manifest.is_file() or file_sha256(manifest) != contract.get("manifest_sha256"):
        raise ValueError(
            "Tokenizer Stage 4 initialization manifest is missing or does not match the artifact"
        )
    model_state = state.get("model") or {}
    codebook = model_state.get("quantizer.codebook")
    shape = (expected["codebook_size"], expected["code_dim"])
    if not isinstance(codebook, torch.Tensor) or tuple(codebook.shape) != shape:
        raise ValueError(
            f"Tokenizer Stage 4 initialization codebook must have shape {shape}"
        )
    if not torch.isfinite(codebook).all():
        raise ValueError("Tokenizer Stage 4 initialization codebook contains NaN or infinity")
    norms = codebook.float().norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
        raise ValueError("Tokenizer Stage 4 initialization codebook is not normalized")
    if _checkpoint_phase(state) != "stage4_initialization" or int(
        state.get("stage", 0)
    ) != 4:
        raise ValueError("Tokenizer Stage 4 initialization artifact has invalid phase metadata")
    return state


def build_initialization_artifact(
    config: dict[str, Any],
    *,
    stage3_checkpoint: str | Path,
    manifest: str | Path,
    output: str | Path,
    device: str | None = None,
) -> Path:
    if str(config.get("phase")) != "stage4" or int(config.get("stage", 0)) != 4:
        raise ValueError("Codebook initialization requires the Stage 4 configuration")
    quantizer_config = config["quantizer"]
    if quantizer_config.get("initialization") != "data_dependent":
        raise ValueError("Stage 4 quantizer.initialization must be data_dependent")
    checkpoint_path = Path(stage3_checkpoint).expanduser().resolve()
    checkpoint_state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if int(checkpoint_state.get("stage", 0)) != 3 or _checkpoint_phase(
        checkpoint_state
    ) != "stage3":
        raise ValueError("Codebook initialization requires a completed Stage 3 checkpoint")

    runtime_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(int(config["train"]["seed"]))
    model = MusicTokenizer(config).to(runtime_device)
    load_checkpoint(
        checkpoint_path,
        model=model,
        require_stage=3,
        allow_cross_stage=True,
    )
    model.eval()
    manifest_path = Path(manifest).expanduser().resolve()
    data = config["data"]
    dataset = TokenizerDataset(
        manifest_path,
        stage=4,
        max_duration_sec=float(data["max_duration_sec"]),
        random_crop=False,
        split=data.get("split", "train"),
        ctc_on_crop_mismatch="disable",
        verify_index_hashes=str(data.get("verify_index_hashes", "node_once")),
    )
    requested_frames = int(quantizer_config["init_sample_frames"])
    vectors: list[torch.Tensor] = []
    collected = 0
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            if sample.get("decode_failed"):
                continue
            waveform = sample["waveform"].unsqueeze(0).to(runtime_device)
            lengths = torch.tensor(
                [waveform.shape[-1]], device=runtime_device, dtype=torch.long
            )
            hidden, mask = model.extract_quantizer_inputs(waveform, lengths)
            projected = F.normalize(model.quantizer.input_proj(hidden.float()), dim=-1)
            available = projected[mask]
            take = min(requested_frames - collected, available.shape[0])
            if take:
                vectors.append(available[:take])
                collected += take
            if collected == requested_frames:
                break
    if collected != requested_frames:
        raise RuntimeError(
            f"Stage 4 initialization requested {requested_frames} frames but only "
            f"collected {collected}"
        )
    samples = torch.cat(vectors, dim=0)
    centroids, counts = spherical_kmeans(
        samples,
        clusters=int(config["semantic_contract"]["codebook_size"]),
        iterations=int(quantizer_config["init_kmeans_iterations"]),
        chunk_size=min(
            int(quantizer_config["init_kmeans_batch_frames"]),
            int(quantizer_config.get("distance_chunk_size", 4096)),
        ),
        seed=int(config["train"]["seed"]),
    )
    with torch.no_grad():
        model.quantizer.codebook.copy_(centroids)
        if hasattr(model.quantizer, "ema_count"):
            model.quantizer.ema_count.copy_(counts.clamp_min(1.0))
        if hasattr(model.quantizer, "ema_sum"):
            model.quantizer.ema_sum.copy_(
                centroids * counts.clamp_min(1.0).unsqueeze(-1)
            )

    source_identity = checkpoint_runtime_identity(checkpoint_path)
    contract = {
        "source_phase": "stage3",
        "source_checkpoint_sha256": source_identity["sha256"],
        "manifest_sha256": file_sha256(manifest_path),
        "sample_frames": requested_frames,
        "kmeans_iterations": int(quantizer_config["init_kmeans_iterations"]),
        "codebook_size": int(config["semantic_contract"]["codebook_size"]),
        "code_dim": int(quantizer_config["code_dim"]),
        "seed": int(config["train"]["seed"]),
    }
    artifact_config = copy.deepcopy(config)
    artifact_config["phase"] = "stage4_initialization"
    artifact = {
        "format_version": INITIALIZATION_FORMAT,
        "artifact_role": INITIALIZATION_ROLE,
        "stage": 4,
        "global_step": 0,
        "model": {key: value.cpu() for key, value in model.state_dict().items()},
        "config": artifact_config,
        "initialization_contract": contract,
    }
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(artifact, temporary)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    _ = validate_initialization_artifact(output_path, config)
    sidecar = {
        "format_version": INITIALIZATION_FORMAT,
        "artifact_role": INITIALIZATION_ROLE,
        "sha256": file_sha256(output_path),
        "initialization_contract": contract,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the data-dependent Tokenizer Stage 4 codebook"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage3-checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    manifest = args.manifest or config["quantizer"].get("initialization_manifest")
    output = args.output or config["quantizer"].get("initialization_path")
    if not manifest or not output:
        parser.error(
            "--manifest and --output are required unless the Stage 4 config declares "
            "quantizer.initialization_manifest and quantizer.initialization_path"
        )
    path = build_initialization_artifact(
        config,
        stage3_checkpoint=args.stage3_checkpoint,
        manifest=manifest,
        output=output,
        device=args.device,
    )
    print(path)


if __name__ == "__main__":
    main()
