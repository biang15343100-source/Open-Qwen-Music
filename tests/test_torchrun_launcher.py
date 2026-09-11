from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/launch_torchrun.sh"


def _dry_run(*arguments: str, **environment: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1", **environment},
        text=True,
        capture_output=True,
        check=False,
    )


def test_single_node_uses_standard_standalone_torchrun() -> None:
    result = _dry_run("scripts/train.py", "llm-stage1", NPROC_PER_NODE="4")
    assert result.returncode == 0, result.stderr
    assert "torch.distributed.run" in result.stdout
    assert "--standalone" in result.stdout
    assert "--nproc-per-node=4" in result.stdout
    assert "scripts/train.py llm-stage1" in result.stdout


def test_multi_node_uses_rendezvous_without_remote_shell() -> None:
    result = _dry_run(
        "--module",
        "open_qwen_music.llm.cli",
        "train",
        NNODES="2",
        NODE_RANK="1",
        NPROC_PER_NODE="8",
        MASTER_ADDR="10.0.0.10",
        MASTER_PORT="29400",
    )
    assert result.returncode == 0, result.stderr
    assert "--nnodes=2" in result.stdout
    assert "--node-rank=1" in result.stdout
    assert "--rdzv-endpoint=10.0.0.10:29400" in result.stdout
    assert "--module open_qwen_music.llm.cli train" in result.stdout
    assert "ssh" not in result.stdout


def test_multi_node_requires_master_address() -> None:
    environment = os.environ.copy()
    environment.pop("MASTER_ADDR", None)
    result = subprocess.run(
        ["bash", str(LAUNCHER), "scripts/train.py", "renderer"],
        cwd=ROOT,
        env={
            **environment,
            "DRY_RUN": "1",
            "NNODES": "2",
            "NODE_RANK": "0",
            "NPROC_PER_NODE": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "MASTER_ADDR" in result.stderr
