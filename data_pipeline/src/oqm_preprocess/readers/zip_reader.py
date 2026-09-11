
from __future__ import annotations

import struct
import zipfile
import zlib
from collections.abc import Iterator
from pathlib import Path

from .base import (
    MemberInfo,
    ReadError,
    ReadHints,
    is_audio_name,
    is_mac_junk,
    pread,
    pread_upto,
)

_LOCAL_SIG = b"PK\x03\x04"
_LOCAL_HEADER_LEN = 30

_EXTRA_PROBE = 256

METHOD_STORED = 0
METHOD_DEFLATE = 8


_FLAG_ENCRYPTED = 0x1
_CRYPT_HEADER_LEN = 12


def _crc_table() -> list[int]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ (0xEDB88320 if crc & 1 else 0)
        table.append(crc)
    return table


_CRCTABLE = _crc_table()


class _ZipCrypto:

    __slots__ = ("_k0", "_k1", "_k2")

    def __init__(self, password: bytes) -> None:
        self._k0, self._k1, self._k2 = 305419896, 591751049, 878082192
        for byte in password:
            self._update(byte)

    def _update(self, char: int) -> None:
        k0 = (self._k0 >> 8) ^ _CRCTABLE[(self._k0 ^ char) & 0xFF]
        k1 = (self._k1 + (k0 & 0xFF)) & 0xFFFFFFFF
        k1 = (k1 * 134775813 + 1) & 0xFFFFFFFF
        self._k0 = k0
        self._k1 = k1
        self._k2 = (self._k2 >> 8) ^ _CRCTABLE[(self._k2 ^ (k1 >> 24)) & 0xFF]

    def decrypt(self, data: bytes) -> bytes:
        out = bytearray(len(data))
        table = _CRCTABLE
        k0, k1, k2 = self._k0, self._k1, self._k2
        for i, cipher in enumerate(data):
            k = k2 | 2
            plain = cipher ^ (((k * (k ^ 1)) >> 8) & 0xFF)
            out[i] = plain
            k0 = (k0 >> 8) ^ table[(k0 ^ plain) & 0xFF]
            k1 = ((k1 + (k0 & 0xFF)) & 0xFFFFFFFF)
            k1 = (k1 * 134775813 + 1) & 0xFFFFFFFF
            k2 = (k2 >> 8) ^ table[(k2 ^ (k1 >> 24)) & 0xFF]
        self._k0, self._k1, self._k2 = k0, k1, k2
        return bytes(out)


def _decrypt(raw: bytes, password: bytes, container: Path, member: str) -> bytes:
    if len(raw) < _CRYPT_HEADER_LEN:
        raise ReadError(f"zip Encryption header truncation {container}::{member}", "member_missing")
    return _ZipCrypto(password).decrypt(raw)[_CRYPT_HEADER_LEN:]


def list_members(container: Path, *, audio_only: bool = True) -> Iterator[MemberInfo]:
    try:
        archive = zipfile.ZipFile(container)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReadError(f"zip cannot be opened {container}: {exc}", "container_missing") from exc
    with archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if is_mac_junk(name):
                continue
            if audio_only and not is_audio_name(name):
                continue
            yield MemberInfo(
                name=name,
                size=info.file_size,
                header_offset=info.header_offset,
                compressed_size=info.compress_size,
                compress_method=info.compress_type,
                mtime=int(info.date_time[0]) if info.date_time else 0,
            )


def read_member(container: Path, member: str, hints: ReadHints | None = None,
                password: bytes | None = None) -> bytes:
    if hints is not None and hints.usable:
        return _read_fast(container, member, hints, password)
    return _read_via_zipfile(container, member, password)


def _read_fast(container: Path, member: str, hints: ReadHints,
               password: bytes | None = None) -> bytes:
    if hints.data_offset >= 0:
        raw = pread(container, hints.data_offset, hints.compressed_size)

        if password:
            raw = _decrypt(raw, password, container, member)
        return _inflate(raw, hints.compress_method, hints.size, container, member)

    name_len = len(member.encode("utf-8"))
    want = _LOCAL_HEADER_LEN + name_len + _EXTRA_PROBE + hints.compressed_size
    blob = _pread_upto(container, hints.header_offset, want)
    if len(blob) < _LOCAL_HEADER_LEN:
        raise ReadError(f"zip Local header truncation {container}::{member}", "member_missing")
    if blob[:4] != _LOCAL_SIG:
        raise ReadError(f"zip The local header signature does not match {container}::{member}", "member_missing")

    flag_bits = struct.unpack("<H", blob[6:8])[0]
    hdr_name_len, hdr_extra_len = struct.unpack("<HH", blob[26:30])
    data_start = _LOCAL_HEADER_LEN + hdr_name_len + hdr_extra_len
    data_end = data_start + hints.compressed_size
    if len(blob) < data_end:

        raw = pread(container, hints.header_offset + data_start, hints.compressed_size)
    else:
        raw = blob[data_start:data_end]
    if flag_bits & _FLAG_ENCRYPTED:
        if not password:
            raise ReadError(
                f"zip Member is encrypted but no password is configured {container}::{member}", "decode_failed")
        raw = _decrypt(raw, password, container, member)
    return _inflate(raw, hints.compress_method, hints.size, container, member)


def _pread_upto(container: Path, offset: int, size: int) -> bytes:
    return pread_upto(container, offset, size)


def _inflate(raw: bytes, method: int, expect_size: int, container: Path, member: str,
             password: bytes | None = None) -> bytes:
    if method == METHOD_STORED:
        data = raw
    elif method == METHOD_DEFLATE:
        try:
            data = zlib.decompressobj(-zlib.MAX_WBITS).decompress(raw)
        except zlib.error as exc:
            raise ReadError(f"zip Decompression failed {container}::{member}: {exc}", "decode_failed") from exc
    else:


        return _read_via_zipfile(container, member, password)

    if expect_size >= 0 and len(data) != expect_size:
        raise ReadError(
            f"zip member length does not match {container}::{member}: expects {expect_size} Real gain {len(data)}",
            "truncated",
        )
    return data


def _read_via_zipfile(container: Path, member: str,
                      password: bytes | None = None) -> bytes:
    try:
        with zipfile.ZipFile(container) as archive:
            if password:
                archive.setpassword(password)
            with archive.open(member) as fh:
                return fh.read()
    except KeyError as exc:
        raise ReadError(f"zip has no such member {container}::{member}", "member_missing") from exc
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReadError(f"zip Reading failed {container}::{member}: {exc}", "decode_failed") from exc
