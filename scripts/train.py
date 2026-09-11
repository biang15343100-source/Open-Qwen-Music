#!/usr/bin/env python3
"""Run a final Open-Qwen-Music training stage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = {
    "tokenizer-stage1a": ROOT / "configs/train/tokenizer-stage1a.yaml",
    "tokenizer-stage1b": ROOT / "configs/train/tokenizer-stage1b.yaml",
    "tokenizer-stage2b": ROOT / "configs/train/tokenizer-stage2b.yaml",
    "tokenizer-stage3-head-warmup": ROOT
    / "configs/train/tokenizer-stage3-head-warmup.yaml",
    "tokenizer-stage3": ROOT / "configs/train/tokenizer-stage3.yaml",
    "tokenizer-stage4": ROOT / "configs/train/tokenizer-stage4.yaml",
    "llm-stage1": ROOT / "configs/train/llm.yaml",
    "llm-stage2": ROOT / "configs/train/llm-stage2.yaml",
    "vae-stage1": ROOT / "configs/train/vae-stage1.yaml",
    "vae-stage2": ROOT / "configs/train/vae-stage2.yaml",
    "renderer": ROOT / "configs/train/renderer.yaml",
    "refiner": ROOT / "configs/train/refiner.yaml",
}


def _command(component: str, arguments: list[str]) -> list[str]:
    if "--config" not in arguments:
        arguments = ["--config", str(DEFAULT_CONFIGS[component]), *arguments]
    if component.startswith("tokenizer-"):
        return [sys.executable, "-m", "open_qwen_music.tokenizer.cli", *arguments]
    if component.startswith("llm-"):
        return [sys.executable, "-m", "open_qwen_music.llm.cli", "train", *arguments]
    if component == "renderer":
        return [sys.executable, "-m", "open_qwen_music.render.cli", "train", *arguments]
    return [sys.executable, str(ROOT / "scripts/train_acoustic.py"), *arguments]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=tuple(DEFAULT_CONFIGS))
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return
    component = sys.argv[1]
    if component not in DEFAULT_CONFIGS:
        parser.error(
            f"argument component: invalid choice: {component!r} "
            f"(choose from {', '.join(DEFAULT_CONFIGS)})"
        )
    command = _command(component, sys.argv[2:])
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
