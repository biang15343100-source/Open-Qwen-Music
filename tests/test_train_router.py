from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import train


ROOT = Path(__file__).resolve().parents[1]


def test_language_model_stages_use_distinct_configs() -> None:
    stage1 = train._command("llm-stage1", [])
    stage2 = train._command("llm-stage2", ["--init-from", "stage1.pt"])

    assert stage1 == [
        sys.executable,
        "-m",
        "open_qwen_music.llm.cli",
        "train",
        "--config",
        str(train.DEFAULT_CONFIGS["llm-stage1"]),
    ]
    assert stage2 == [
        sys.executable,
        "-m",
        "open_qwen_music.llm.cli",
        "train",
        "--config",
        str(train.DEFAULT_CONFIGS["llm-stage2"]),
        "--init-from",
        "stage1.pt",
    ]


def test_explicit_config_is_not_replaced() -> None:
    command = train._command(
        "llm-stage2",
        ["--config", "custom.yaml", "--resume-from", "last.pt"],
    )
    assert command[-4:] == ["--config", "custom.yaml", "--resume-from", "last.pt"]


def test_component_help_is_forwarded_to_the_selected_trainer() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "scripts/train.py", "tokenizer-stage1a", "--help"],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--init-from" in result.stdout
    assert "--resume-from" in result.stdout
