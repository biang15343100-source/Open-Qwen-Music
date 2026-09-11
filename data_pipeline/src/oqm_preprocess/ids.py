
from __future__ import annotations

import hashlib

UID_BYTES = 8
HASH_BYTES = 16

_HEAD_TAIL = 1 << 20


def make_uid(dataset_slug: str, local_id: str) -> bytes:
    digest = hashlib.sha1(f"{dataset_slug}\x00{local_id}".encode(), usedforsecurity=False)
    return digest.digest()[:UID_BYTES]


def uid_hex(uid: bytes) -> str:
    return uid.hex()


def make_group_id(group_key: str) -> int:
    digest = hashlib.sha256(group_key.encode()).digest()
    return int.from_bytes(digest[:8], "big")


def content_hash_l1(data: bytes) -> bytes:
    size = len(data)
    h = hashlib.sha256()
    h.update(size.to_bytes(8, "big"))
    if size <= 2 * _HEAD_TAIL:
        h.update(data)
    else:
        h.update(data[:_HEAD_TAIL])
        h.update(data[-_HEAD_TAIL:])
    return h.digest()[:HASH_BYTES]


def split_bucket(group_id: int) -> int:
    digest = hashlib.sha256(group_id.to_bytes(8, "big")).digest()
    return int.from_bytes(digest[:8], "big") % 100
