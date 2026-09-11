
from __future__ import annotations

from pathlib import Path
from typing import Any

from .enums import STORAGE_CLASS
from .readers.base import ReadHints
from .store.containers import ContainerTable
from .uri import AudioRef, build


def to_ref(row: dict[str, Any], containers: ContainerTable) -> AudioRef:
    storage_class = STORAGE_CLASS.name_of(int(row["storage_class"]))
    container = containers.path_of(int(row["container_id"]))
    member = row.get("member") or ""

    if storage_class == "loose":

        return AudioRef(storage_class="loose", container=container / member)

    extra = parse_params(row.get("locator_params"))
    params: dict[str, str] | None = None
    if storage_class == "parquet":
        params = {
            "rg": str(int_field(row, "parquet_row_group")),
            "row": str(int_field(row, "parquet_row_index")),
            **extra,
        }
        member = ""
    elif storage_class == "sqlite":
        params = {"id": str(member), **extra}
        member = ""

    start = row.get("clip_start_sec")
    end = row.get("clip_end_sec")
    return AudioRef(
        storage_class=storage_class,
        container=container,
        member=member,
        params=params,
        clip_start_sec=float(start) if start is not None else None,
        clip_end_sec=float(end) if end is not None else None,
    )


def int_field(row: dict[str, Any], key: str, default: int = -1) -> int:
    value = row.get(key)
    return default if value is None else int(value)


def to_hints(row: dict[str, Any]) -> ReadHints:
    return ReadHints(
        data_offset=int_field(row, "member_offset"),
        header_offset=int_field(row, "member_header_offset"),
        size=int_field(row, "member_size"),
        compressed_size=_compressed_size(row),
        compress_method=int_field(row, "member_compress", 0),
    )


def _compressed_size(row: dict[str, Any]) -> int:
    explicit = int_field(row, "member_compressed_size")
    if explicit >= 0:
        return explicit
    return int_field(row, "member_size") if int_field(row, "member_compress", 0) == 0 else -1


def parse_params(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    out: dict[str, str] = {}
    for chunk in text.split("&"):
        key, eq, value = chunk.partition("=")
        if eq:
            out[key] = value
    return out


def format_params(params: dict[str, str]) -> str | None:
    if not params:
        return None
    return "&".join(f"{k}={v}" for k, v in sorted(params.items()))


def to_uri(row: dict[str, Any], containers: ContainerTable) -> str:
    ref = to_ref(row, containers)
    return build(
        ref.storage_class, ref.container, ref.member,
        params=ref.params,
        clip_start_sec=ref.clip_start_sec, clip_end_sec=ref.clip_end_sec,
    )


def name_hint(row: dict[str, Any], containers: ContainerTable) -> str:
    storage_class = STORAGE_CLASS.name_of(int(row["storage_class"]))
    if storage_class == "loose":
        return str(Path(row.get("member") or "").name)
    member = row.get("member") or ""
    if member:
        return member.rsplit("/", 1)[-1]
    return containers.path_of(int(row["container_id"])).name
