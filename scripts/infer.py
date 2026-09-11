#!/usr/bin/env python3
"""Run the fixed end-to-end Open-Qwen-Music inference pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, help="condition JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/inference/pipeline.yaml"),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Hugging Face model repository ID or a local model snapshot",
    )
    parser.add_argument("--revision", default=None, help="Model repository revision")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory")
    parser.add_argument(
        "--text-encoder",
        default=None,
        help="Local renderer text-encoder snapshot (required for fully offline inference)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from open_qwen_music.pipeline import run_pipeline

    manifest = run_pipeline(
        config_path=args.config,
        prompts_path=args.prompts,
        output_dir=args.output_dir,
        model_source=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        text_encoder_source=args.text_encoder,
        device=args.device,
        seed=args.seed,
    )
    print(f"Inference completed: {manifest}")


if __name__ == "__main__":
    main()
