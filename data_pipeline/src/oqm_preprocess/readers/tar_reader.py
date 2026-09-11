
from __future__ import annotations

import contextlib
import tarfile
import zlib
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .base import MemberInfo, ReadError, ReadHints, is_audio_name, is_mac_junk, pread

try:
    from isal import igzip as _igzip
except ImportError:  # pragma: no cover -
    _igzip = None


_STREAM_MODES = {
    ".gz": "r|gz", ".tgz": "r|gz",
    ".bz2": "r|bz2", ".tbz": "r|bz2", ".tbz2": "r|bz2",
    ".xz": "r|xz", ".txz": "r|xz",
}
_GZIP_SUFFIXES = frozenset({".gz", ".tgz"})


def _stream_mode(container: Path) -> str:
    return _STREAM_MODES.get(container.suffix.lower(), "r|*")


@contextlib.contextmanager
def _open_stream(container: Path) -> Iterator[tarfile.TarFile]:
    raw = None
    try:
        if _igzip is not None and container.suffix.lower() in _GZIP_SUFFIXES:
            raw = _igzip.IGzipFile(container, "rb")
            archive = tarfile.open(fileobj=raw, mode="r|")  # noqa: SIM115 -  finally
        else:
            archive = tarfile.open(container, mode=_stream_mode(container))  # noqa: SIM115
    except (OSError, EOFError, tarfile.TarError) as exc:
        if raw is not None:
            raw.close()
        raise ReadError(f"tar.gz cannot be opened {container}: {exc}", "container_missing") from exc
    try:
        with archive:
            yield archive
    finally:
        if raw is not None:
            raw.close()


def list_members(
    container: Path, *, compressed: bool, audio_only: bool = True
) -> Iterator[MemberInfo]:
    if compressed:
        yield from _list_stream(container, audio_only=audio_only)
    else:
        yield from _list_seekable(container, audio_only=audio_only)


_BLOCK = 512
_USTAR_MAGIC = b"ustar"
_PLAIN_FILE = frozenset({b"0", b"\x00", b"7"})
_SKIPPABLE = frozenset({b"1", b"2", b"5", b"6"})
_GNU_LONGNAME = b"L"
_GNU_LONGLINK = b"K"
_PAX_LOCAL = b"x"
_PAX_GLOBAL = b"g"

_MAX_META_BYTES = 1 << 20


def _octal(field: bytes) -> int:
    text = field.split(b"\x00")[0].strip()
    return int(text, 8) if text else 0


def _pax_path(blob: bytes) -> str | None:
    pos = 0
    while pos < len(blob):
        end = blob.find(b" ", pos)
        if end < 0:
            return None
        try:
            length = int(blob[pos:end])
        except ValueError:
            return None
        if length <= 0 or pos + length > len(blob):
            return None
        record = blob[end + 1: pos + length].rstrip(b"\n")
        key, _, value = record.partition(b"=")
        if key == b"path":
            return value.decode("utf-8", "surrogateescape")
        pos += length
    return None


def _list_seekable(container: Path, *, audio_only: bool) -> Iterator[MemberInfo]:
    import os

    try:
        fd = os.open(container, os.O_RDONLY)
    except OSError as exc:
        raise ReadError(f"tar cannot be opened {container}: {exc}", "container_missing") from exc
    try:
        offset = 0
        pending_name: str | None = None
        while True:
            head = os.pread(fd, _BLOCK, offset)
            if len(head) < _BLOCK:
                return
            if not head.strip(b"\x00"):
                return
            if head[257:262] != _USTAR_MAGIC:
                raise ReadError(
                    f"TAR member header is invalid in {container} at offset {offset}",
                    "container_missing",
                )
            kind = head[156:157]
            try:
                size = _octal(head[124:136])
                mtime = _octal(head[136:148])
            except ValueError as exc:
                raise ReadError(
                    f"tar The header field is broken {container} at {offset}: {exc}",
                    "container_missing") from exc
            data_offset = offset + _BLOCK
            offset = data_offset + (size + _BLOCK - 1) // _BLOCK * _BLOCK

            if kind in (_GNU_LONGNAME, _PAX_LOCAL, _PAX_GLOBAL, _GNU_LONGLINK):
                if size > _MAX_META_BYTES:
                    raise ReadError(
                        f"tar The extension header is abnormally large {container} at {data_offset}: {size}",
                        "container_missing")
                blob = os.pread(fd, size, data_offset) if size else b""
                if kind == _GNU_LONGNAME:
                    pending_name = blob.split(b"\x00")[0].decode("utf-8", "surrogateescape")
                elif kind == _PAX_LOCAL:
                    got = _pax_path(blob)
                    if got:
                        pending_name = got

                continue

            if kind in _SKIPPABLE:
                pending_name = None
                continue
            if kind not in _PLAIN_FILE:
                raise ReadError(
                    f"tar Header type {kind!r} does not currently support {container} at {offset}",
                    "container_missing")

            if pending_name is not None:
                member = pending_name
                pending_name = None
            else:

                name = head[0:100].split(b"\x00")[0]
                prefix = head[345:500].split(b"\x00")[0]
                full = (prefix + b"/" + name) if prefix else name
                member = full.decode("utf-8", "surrogateescape")
            if is_mac_junk(member):
                continue
            if audio_only and not is_audio_name(member):
                continue
            yield MemberInfo(
                name=member,
                size=size,
                data_offset=data_offset,
                compressed_size=size,
                compress_method=0,
                mtime=mtime,
            )
    finally:
        os.close(fd)


def _list_seekable_tarfile(container: Path, *, audio_only: bool) -> Iterator[MemberInfo]:
    try:
        archive = tarfile.open(container, mode="r:")  # noqa: SIM115 -  with
    except (OSError, tarfile.TarError) as exc:
        raise ReadError(f"tar cannot be opened {container}: {exc}", "container_missing") from exc
    with archive:
        for info in archive:
            if not info.isfile():
                continue
            if is_mac_junk(info.name):
                continue
            if audio_only and not is_audio_name(info.name):
                continue
            yield MemberInfo(
                name=info.name,
                size=info.size,
                data_offset=info.offset_data,
                compressed_size=info.size,
                compress_method=0,
                mtime=int(info.mtime),
            )


_STREAM_ERRORS = (tarfile.TarError, EOFError, OSError, zlib.error)


def _guard(stream: Iterator[Any], container: Path) -> Iterator[Any]:
    try:
        yield from stream
    except _STREAM_ERRORS as exc:


        raise ReadError(f"tar Stream interrupted {container}: {exc}", "member_missing") from exc


def _list_stream(container: Path, *, audio_only: bool) -> Iterator[MemberInfo]:
    yield from _guard(_iter_stream_members(container, audio_only=audio_only), container)


def _iter_stream_members(container: Path, *, audio_only: bool) -> Iterator[MemberInfo]:
    with _open_stream(container) as archive:
        for seq, info in enumerate(archive):
            if not info.isfile():
                continue
            if is_mac_junk(info.name):
                continue
            if audio_only and not is_audio_name(info.name):
                continue
            yield MemberInfo(
                name=info.name,
                size=info.size,
                data_offset=-1,
                compressed_size=-1,
                mtime=int(info.mtime),
                extra={"seq": seq},
            )


def read_member(container: Path, member: str, hints: ReadHints | None = None) -> bytes:
    if hints is not None and hints.data_offset >= 0 and hints.size >= 0:
        return pread(container, hints.data_offset, hints.size)
    return _read_via_tarfile(container, member)


def _read_via_tarfile(container: Path, member: str) -> bytes:
    try:
        with tarfile.open(container, mode="r:*") as archive:
            fh = archive.extractfile(member)
            if fh is None:
                raise ReadError(f"tar member is not an ordinary file {container}::{member}", "member_missing")
            with fh:
                return fh.read()
    except KeyError as exc:
        raise ReadError(f"tar has no such member {container}::{member}", "member_missing") from exc
    except (OSError, tarfile.TarError) as exc:
        raise ReadError(f"tar Reading failed {container}::{member}: {exc}", "decode_failed") from exc


def stream_members(
    container: Path, wanted: Iterable[str] | None = None, *, audio_only: bool = True
) -> Iterator[tuple[str, bytes]]:
    yield from _guard(_iter_stream_bytes(container, wanted, audio_only=audio_only), container)


def _iter_stream_bytes(
    container: Path, wanted: Iterable[str] | None, *, audio_only: bool
) -> Iterator[tuple[str, bytes]]:
    remaining = set(wanted) if wanted is not None else None
    if remaining is not None and not remaining:
        return
    with _open_stream(container) as archive:
        for info in archive:
            if not info.isfile() or is_mac_junk(info.name):
                continue
            if remaining is not None:
                if info.name not in remaining:
                    continue
            elif audio_only and not is_audio_name(info.name):
                continue
            fh = archive.extractfile(info)
            if fh is None:
                continue
            with fh:
                yield info.name, fh.read()
            if remaining is not None:
                remaining.discard(info.name)
                if not remaining:
                    return
