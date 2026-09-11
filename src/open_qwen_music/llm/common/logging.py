
from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .distributed import is_main_process


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_kv(prefix: str, values: dict[str, Any], *, main_only: bool = True) -> None:
    if main_only and not is_main_process():
        return
    parts = []
    for key, value in values.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.6g}")
        else:
            parts.append(f"{key}={value}")
    print(f"{prefix} " + " ".join(parts), flush=True)


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


_SOURCE_TREE_REVISION_CACHE: str | None = None


def source_tree_revision(root: str | Path | None = None) -> str:
    global _SOURCE_TREE_REVISION_CACHE
    if root is None and _SOURCE_TREE_REVISION_CACHE is not None:
        return _SOURCE_TREE_REVISION_CACHE
    source_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = [path for path in source_root.glob("*.py") if path.is_file()]
    scan_names = ("common", "data", "eval", "scripts", "configs")
    for name in scan_names:
        directory = source_root / name
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    for path in sorted(paths, key=lambda item: item.relative_to(source_root).as_posix()):
        relative = path.relative_to(source_root)
        if (
            "__pycache__" in relative.parts
            or ".pytest_cache" in relative.parts
            or path.suffix == ".pyc"
        ):
            continue
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    revision = "sha256:" + digest.hexdigest()
    if root is None:
        _SOURCE_TREE_REVISION_CACHE = revision
    return revision


def _source_state(
    root: Path,
    output_dir: Path,
    *,
    snapshot_name: str = "source_snapshot.tar.gz",
) -> dict[str, Any]:
    try:
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain=v1", "-z"],
            stderr=subprocess.DEVNULL,
        )
        diff = subprocess.check_output(
            ["git", "-C", str(root), "diff", "--binary", "HEAD", "--", "."],
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        status, diff = b"", b""

    snapshot = output_dir / snapshot_name

    def filter_member(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        if "__pycache__" in parts or ".pytest_cache" in parts or info.name.endswith(".pyc"):
            return None
        return info

    with tarfile.open(snapshot, "w:gz") as archive:
        for name in (
            "__init__.py",
            "cli.py",
            "condition.py",
            "contracts.py",
            "curriculum.py",
            "generate.py",
            "grammar.py",
            "inference.py",
            "melody.py",
            "melody_tokenizer.py",
            "metrics.py",
            "model.py",
            "registry.py",
            "render_contract.py",
            "rmvpe.py",
            "sequence.py",
            "trainer.py",
            "common",
            "data",
            "eval",
            "scripts",
            "configs",
            "tests",
            "README.md",
        ):
            path = root / name
            if path.exists():
                archive.add(path, arcname=name, recursive=True, filter=filter_member)
    snapshot_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    return {
        "git_dirty": bool(status),
        "git_status_sha256": hashlib.sha256(status).hexdigest(),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "snapshot": snapshot.name,
        "snapshot_sha256": snapshot_hash,
        "snapshot_bytes": snapshot.stat().st_size,
        "tree_revision": source_tree_revision(root),
    }


def write_run_metadata(
    output_dir: str | Path,
    config: dict[str, Any],
    *,
    world_size: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    primary_path = directory / "run.json"
    if primary_path.exists():
        invocation_dir = directory / "invocations"
        invocation_dir.mkdir(parents=True, exist_ok=True)
        stamp = (
            time.strftime("%Y%m%dT%H%M%S")
            + f"-{time.time_ns()}-pid{os.getpid()}"
        )
        path = invocation_dir / f"{stamp}.json"
        snapshot_name = f"source_snapshot-{stamp}.tar.gz"
    else:
        path = primary_path
        snapshot_name = "source_snapshot.tar.gz"
    root = Path(__file__).resolve().parents[1]
    source = _source_state(root, directory, snapshot_name=snapshot_name)
    payload = {
        "schema_version": "oqm.llm.run.v2",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "world_size": world_size,
        "cluster": {
            key: os.environ.get(key)
            for key in (
                "HOST_NUM",
                "HOST_GPU_NUM",
                "INDEX",
                "CHIEF_IP",
                "MASTER_ADDR",
                "MASTER_PORT",
                "NODE_IP_LIST",
                "RUN_ID",
            )
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "git_commit": _git_commit(root),
        "source": source,
        "runtime": {
            "python": os.sys.version,
            "container_image": os.environ.get("IMAGE_FULL_NAME")
            or os.environ.get("CONTAINER_IMAGE"),
        },
        "config": config,
        "extra": extra or {},
    }
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


class Stopwatch:

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._start = time.monotonic()

    def lap(self) -> float:
        now = time.monotonic()
        elapsed = now - self._start
        self._start = now
        return elapsed
