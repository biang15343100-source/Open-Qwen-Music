
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .base import MemberInfo, ReadError


_BYTES_FIELD = "bytes"
_PATH_FIELD = "path"
_AUDIO_COLUMN_HINTS = ("audio", "wav", "waveform", "speech", "mp3", "flac", "audio_bytes")


def _open(container: Path) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(container)
    except (OSError, pa.ArrowInvalid) as exc:
        raise ReadError(f"parquet cannot be opened {container}: {exc}", "container_missing") from exc


def detect_audio_column(container: Path) -> str:
    schema = _open(container).schema_arrow
    binary_cols: list[str] = []
    for field in schema:
        t = field.type
        if pa.types.is_struct(t):
            names = {t.field(i).name for i in range(t.num_fields)}
            if _BYTES_FIELD in names:
                return field.name
        elif pa.types.is_binary(t) or pa.types.is_large_binary(t):
            binary_cols.append(field.name)
    for hint in _AUDIO_COLUMN_HINTS:
        if hint in binary_cols:
            return hint
    if len(binary_cols) == 1:
        return binary_cols[0]
    raise ReadError(
        f"Unable to determine the Parquet audio column in {container}; "
        f"binary columns: {binary_cols}",
        "probe_failed",
    )


def list_members(
    container: Path,
    *,
    audio_column: str | None = None,
    meta_columns: list[str] | None = None,
) -> Iterator[MemberInfo]:
    handle = _open(container)
    column = audio_column or detect_audio_column(container)
    schema = handle.schema_arrow
    if column not in schema.names:
        raise ReadError(f"Parquet column {column!r} is missing from {container}", "probe_failed")

    want = [c for c in (meta_columns or []) if c in schema.names]

    path_proj = f"{column}.{_PATH_FIELD}"
    audio_type = schema.field(column).type
    has_path = pa.types.is_struct(audio_type) and any(
        audio_type.field(i).name == _PATH_FIELD for i in range(audio_type.num_fields)
    )

    for rg in range(handle.num_row_groups):
        rg_meta = handle.metadata.row_group(rg)
        n_rows = rg_meta.num_rows
        cols = list(want)
        if has_path:
            cols.append(path_proj)
        table = handle.read_row_group(rg, columns=cols) if cols else None
        paths = _extract_paths(table, path_proj, column) if (table is not None and has_path) else None
        meta_rows = _extract_meta(table, want) if (table is not None and want) else None

        for row in range(n_rows):
            inner = paths[row] if paths is not None else None
            name = inner or f"rg{rg}/row{row}"
            yield MemberInfo(
                name=name,
                size=-1,
                data_offset=-1,
                compressed_size=-1,
                extra={
                    "rg": rg,
                    "row": row,
                    "col": column,
                    **(meta_rows[row] if meta_rows is not None else {}),
                },
            )


def _extract_paths(table: pa.Table, path_proj: str, column: str) -> list[str | None] | None:
    for candidate in (path_proj, column):
        if candidate in table.column_names:
            col = table.column(candidate)
            if pa.types.is_struct(col.type):
                col = pc.struct_field(col, _PATH_FIELD)
            return col.to_pylist()
    return None


def _extract_meta(table: pa.Table, want: list[str]) -> list[dict[str, Any]]:
    present = [c for c in want if c in table.column_names]
    if not present:
        return [{} for _ in range(table.num_rows)]
    cols = {c: table.column(c).to_pylist() for c in present}
    return [{c: cols[c][i] for c in present} for i in range(table.num_rows)]


def stream_rowgroup(
    container: Path, row_group: int, column: str
) -> Iterator[tuple[int, bytes | None]]:
    handle = _open(container)
    if not 0 <= row_group < handle.num_row_groups:
        raise ReadError(f"parquet row_group out of bounds {row_group} @ {container}", "member_missing")
    try:
        table = handle.read_row_group(row_group, columns=[column])
    except (OSError, pa.ArrowInvalid, KeyError) as exc:
        raise ReadError(f"parquet Reading failed {container} rg={row_group}: {exc}", "decode_failed") from exc
    col = table.column(column)
    if pa.types.is_struct(col.type):
        col = pc.struct_field(col, _BYTES_FIELD)
    for index in range(len(col)):
        value = col[index].as_py()
        yield index, value
        del value


def read_member(container: Path, params: dict[str, str]) -> bytes:
    try:
        rg = int(params["rg"])
        row = int(params["row"])
    except (KeyError, ValueError) as exc:
        raise ReadError(f"Missing or invalid Parquet parameters: {params}", "member_missing") from exc
    column = params.get("col") or detect_audio_column(container)
    for idx, value in stream_rowgroup(container, rg, column):
        if idx == row:
            if value is None:
                raise ReadError(f"parquet Audio is empty {container} rg={rg} row={row}", "empty_audio")
            return value
    raise ReadError(f"parquet line crossed {container} rg={rg} row={row}", "member_missing")
