
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from .base import MemberInfo, ReadError, is_audio_name, is_mac_junk


def walk(root: Path, *, audio_only: bool = True, follow_symlinks: bool = False,
         member_glob: str | None = None) -> Iterator[MemberInfo]:
    if not root.is_dir():
        raise ReadError(f"directory does not exist {root}", "container_missing")
    files = _by_glob(root, member_glob) if member_glob else _by_walk(root, follow_symlinks)

    collected: list[MemberInfo] = []
    for full in files:
        name = full.name
        if is_mac_junk(name) or (audio_only and not is_audio_name(name)):
            continue
        try:
            st = full.stat()
        except OSError:
            continue
        collected.append(
            MemberInfo(
                name=str(full.relative_to(root)),
                size=st.st_size,
                data_offset=0,
                compressed_size=st.st_size,
                mtime=st.st_mtime_ns // 1_000_000_000,
            )
        )
    collected.sort(key=lambda m: m.name)
    yield from collected


def _by_glob(root: Path, pattern: str) -> Iterator[Path]:
    for path in root.glob(pattern):
        if path.is_file():
            yield path


def _by_walk(root: Path, follow_symlinks: bool) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirnames[:] = sorted(d for d in dirnames if d != "__MACOSX")
        base = Path(dirpath)
        for fname in filenames:
            yield base / fname


def read_file(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise ReadError(f"File does not exist: {path}", "container_missing") from exc
    except OSError as exc:
        raise ReadError(f"File reading failed {path}: {exc}", "decode_failed") from exc
