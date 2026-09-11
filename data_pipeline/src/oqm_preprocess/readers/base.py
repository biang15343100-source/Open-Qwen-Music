
from __future__ import annotations

import contextlib
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any


class ReadError(RuntimeError):

    def __init__(self, message: str, flag: str = "decode_failed") -> None:
        super().__init__(message)
        self.flag = flag


@dataclass(slots=True)
class MemberInfo:

    name: str
    size: int = -1
    data_offset: int = -1
    header_offset: int = -1
    compressed_size: int = -1
    compress_method: int = 0
    mtime: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def seekable(self) -> bool:
        return self.data_offset >= 0 or self.header_offset >= 0


@dataclass(frozen=True, slots=True)
class ReadHints:

    data_offset: int = -1
    header_offset: int = -1
    size: int = -1
    compressed_size: int = -1
    compress_method: int = 0

    @classmethod
    def from_member(cls, m: MemberInfo) -> ReadHints:
        return cls(
            m.data_offset, m.header_offset, m.size, m.compressed_size, m.compress_method
        )

    @property
    def usable(self) -> bool:
        located = self.data_offset >= 0 or self.header_offset >= 0
        return located and self.compressed_size >= 0


class HandleCache:

    def __init__(self, capacity: int = 8) -> None:
        self._capacity = capacity
        self._items: OrderedDict[tuple[str, int, int], IO[bytes]] = OrderedDict()

        self._lock = threading.RLock()

    def get(self, path: Path) -> IO[bytes]:
        try:
            st = os.stat(path)
        except OSError as exc:
            raise ReadError(f"The container is inaccessible {path}: {exc}", "container_missing") from exc
        key = (str(path), st.st_size, st.st_mtime_ns)
        with self._lock:
            handle = self._items.get(key)
            if handle is not None and not handle.closed:
                self._items.move_to_end(key)
                return handle
            try:
                handle = open(path, "rb")  # noqa: SIM115 -
            except OSError as exc:
                raise ReadError(f"Container opening failed {path}: {exc}", "container_missing") from exc
            self._items[key] = handle
            self._items.move_to_end(key)
            while len(self._items) > self._capacity:
                _, stale = self._items.popitem(last=False)
                with contextlib.suppress(OSError):
                    stale.close()
            return handle

    def dup_fd(self, path: Path) -> int:
        with self._lock:
            return os.dup(self.get(path).fileno())

    def configure(self, capacity: int) -> None:
        with self._lock:
            self._capacity = max(1, capacity)

    def clear(self) -> None:
        with self._lock:
            for handle in self._items.values():
                with contextlib.suppress(OSError):
                    handle.close()
            self._items.clear()


HANDLES = HandleCache()


def pread_upto(path: Path, offset: int, size: int) -> bytes:
    if size <= 0:
        return b""
    fd = HANDLES.dup_fd(path)
    chunks: list[bytes] = []
    got = 0
    try:
        while got < size:
            chunk = os.pread(fd, size - got, offset + got)
            if not chunk:
                break
            chunks.append(chunk)
            got += len(chunk)
    except OSError as exc:
        raise ReadError(f"Reading failed {path}@{offset}+{size}: {exc}", "member_missing") from exc
    finally:
        os.close(fd)
    return b"".join(chunks)


def pread(path: Path, offset: int, size: int) -> bytes:
    if size < 0:
        raise ReadError(f"Invalid read length {size} for {path}", "member_missing")
    if size == 0:
        return b""
    data = pread_upto(path, offset, size)
    if len(data) != size:
        raise ReadError(
            f"short read {path}@{offset}: expects {size} Real gain {len(data)}", "truncated"
        )
    return data


AUDIO_EXTENSIONS = frozenset({
    ".wav", ".wave", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac",
    ".aif", ".aiff", ".aifc", ".wma", ".au", ".snd", ".caf", ".w64", ".mp4",
})


def is_audio_name(name: str) -> bool:
    dot = name.rfind(".")
    if dot < 0:
        return False
    return name[dot:].lower() in AUDIO_EXTENSIONS


def is_mac_junk(name: str) -> bool:
    if "__MACOSX/" in name or name.startswith("__MACOSX"):
        return True
    base = name.rsplit("/", 1)[-1]
    return base.startswith("._") or base == ".DS_Store"
