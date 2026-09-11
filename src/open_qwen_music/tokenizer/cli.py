
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from open_qwen_music.common.checkpoint import (
    checkpoint_runtime_identity,
    code_revision_identity,
    file_sha256,
    load_checkpoint,
    stage_checkpoint_locally,
)
from open_qwen_music.common.config import load_config

from .audio import load_audio, resample_mono
from .contracts import SAMPLE_RATE
from .frontend import (
    SUBSAMPLING_CONTRACT_DEFAULTS,
    resolve_subsampling_contract,
)
from .model import MusicTokenizer
from .trainer import (
    _inherit_feature_config_from_checkpoint,
    _stats_to_config,
    _validate_subsampling_contract_from_checkpoint,
    train,
)


def _runtime_checkpoint(
    checkpoint: str, local_checkpoint_cache_dir: str | None
) -> tuple[str, str]:
    source = str(Path(checkpoint).resolve())
    runtime = (
        str(
            stage_checkpoint_locally(
                source, cache_dir=local_checkpoint_cache_dir
            )
        )
        if local_checkpoint_cache_dir
        else source
    )
    return source, runtime


def _inherit_checkpoint_contract(
    config: dict, *, source_checkpoint: str, runtime_checkpoint: str
) -> dict:
    _validate_subsampling_contract_from_checkpoint(config, runtime_checkpoint)
    _inherit_feature_config_from_checkpoint(
        config,
        runtime_checkpoint,
        lineage_checkpoint=source_checkpoint,
    )
    identity = checkpoint_runtime_identity(runtime_checkpoint)
    if int(identity["stage"]) != 4:
        raise ValueError(
            f"Semantic encoding and export require a Stage 4 checkpoint; "
            f"received stage={identity['stage']}"
        )
    return identity


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Training Open-Qwen-Music Tokenizer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-from")
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    config = load_config(args.config)
    train(config, init_from=args.init_from, resume_from=args.resume_from)


def encode_main() -> None:
    parser = argparse.ArgumentParser(description="encoding 24kHz semantic token")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--local-checkpoint-cache-dir")
    args = parser.parse_args()
    config = load_config(args.config)


    config["stage"] = 4
    config["model"]["causal"] = True
    source_checkpoint, runtime_checkpoint = _runtime_checkpoint(
        args.checkpoint, args.local_checkpoint_cache_dir
    )
    checkpoint_identity = _inherit_checkpoint_contract(
        config,
        source_checkpoint=source_checkpoint,
        runtime_checkpoint=runtime_checkpoint,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicTokenizer(config).to(device)
    load_checkpoint(runtime_checkpoint, model=model, require_stage=4)
    model.eval()
    waveform, sample_rate = load_audio(args.audio)
    waveform = resample_mono(waveform, sample_rate, SAMPLE_RATE).to(device)
    revision = str(checkpoint_identity["sha256"])
    result = model.encode_audio(waveform, SAMPLE_RATE, tokenizer_revision=revision)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "token_ids": result.token_ids.cpu(),
            "frame_mask": result.frame_mask.cpu(),
            "frame_rate": result.frame_rate,
            "codebook_size": result.codebook_size,
            "tokenizer_revision": result.tokenizer_revision,
        },
        args.output,
    )


def export_main() -> None:
    parser = argparse.ArgumentParser(description="Export the Stage 4 deployment subgraph")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--local-checkpoint-cache-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    if int(config["stage"]) != 4:
        raise ValueError("is allowed only when exporting a Stage 4 configuration")
    source_checkpoint, runtime_checkpoint = _runtime_checkpoint(
        args.checkpoint, args.local_checkpoint_cache_dir
    )
    checkpoint_identity = _inherit_checkpoint_contract(
        config,
        source_checkpoint=source_checkpoint,
        runtime_checkpoint=runtime_checkpoint,
    )
    model = MusicTokenizer(config)
    load_checkpoint(runtime_checkpoint, model=model, require_stage=4)
    insertion = int(config["model"]["quantizer_insertion_layer"])
    state = {}
    for key, value in model.state_dict().items():
        keep = key.startswith(("feature_extractor.", "subsampling.", "quantizer."))
        if key.startswith("encoder.layers."):
            layer_index = int(key.split(".")[2])
            keep = layer_index < insertion
        if keep:
            state[key] = value.cpu()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)


    features = dict(config["features"])
    features["mean"] = _stats_to_config(model.feature_extractor.feature_mean)
    features["std"] = _stats_to_config(model.feature_extractor.feature_std)


    model_config = {
        **config["model"],
        **resolve_subsampling_contract(config["model"], SUBSAMPLING_CONTRACT_DEFAULTS),
    }
    artifact = {
        "format_version": "oqm.tokenizer.deploy.v1",
        "semantic_contract": config["semantic_contract"],
        "features": features,
        "model": model_config,
        "quantizer": config["quantizer"],
        "source_checkpoint_identity": checkpoint_identity,
        "export_code_identity": code_revision_identity(),
        "state_dict": state,
    }
    torch.save(artifact, output)
    revision = file_sha256(output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "format_version": artifact["format_version"],
                "tokenizer_revision": revision,
                "artifact_size_bytes": output.stat().st_size,
                "semantic_contract": config["semantic_contract"],
                "quantizer": config["quantizer"],
                "source_checkpoint": source_checkpoint,
                "source_checkpoint_identity": checkpoint_identity,
                "export_code_identity": artifact["export_code_identity"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Exported {output} tokenizer_revision={revision}")


if __name__ == "__main__":
    train_main()
