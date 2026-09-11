
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..uri import AudioRef
from . import loose, parquet_reader, sqlite_reader, tar_reader, zip_reader
from .base import (
    AUDIO_EXTENSIONS,
    HANDLES,
    MemberInfo,
    ReadError,
    ReadHints,
    is_audio_name,
    is_mac_junk,
)

__all__ = [
    "AUDIO_EXTENSIONS",
    "HANDLES",
    "MemberInfo",
    "ReadError",
    "ReadHints",
    "enumerate_members",
    "is_audio_name",
    "is_mac_junk",
    "loose",
    "parquet_reader",
    "read_bytes",
    "sqlite_reader",
    "tar_reader",
    "zip_reader",
]


def enumerate_members(
    storage_class: str, container: Path, **options: Any
) -> Iterator[MemberInfo]:
    audio_only = options.pop("audio_only", True)
    if storage_class == "loose":
        return loose.walk(container, audio_only=audio_only, **options)
    if storage_class == "zip":
        return zip_reader.list_members(container, audio_only=audio_only, **options)
    if storage_class == "tar":
        return tar_reader.list_members(container, compressed=False, audio_only=audio_only, **options)
    if storage_class == "targz":
        return tar_reader.list_members(container, compressed=True, audio_only=audio_only, **options)
    if storage_class == "parquet":
        return parquet_reader.list_members(container, **options)
    if storage_class == "sqlite":
        return sqlite_reader.list_members(container, **options)
    raise ReadError(f"storage_class {storage_class!r} does not support enumeration", "probe_failed")


def read_bytes(ref: AudioRef, hints: ReadHints | None = None,
               password: bytes | None = None) -> bytes:
    sc = ref.storage_class
    if sc == "loose":
        return loose.read_file(ref.container)
    if sc == "zip":
        return zip_reader.read_member(ref.container, ref.member, hints, password)
    if sc in ("tar", "targz"):
        return tar_reader.read_member(ref.container, ref.member, hints)
    if sc == "parquet":
        return parquet_reader.read_member(ref.container, ref.params or {})
    if sc == "sqlite":
        return sqlite_reader.read_member(ref.container, ref.params or {})
    raise ReadError(f"storage_class {sc!r} does not support byte reads", "probe_failed")
