
from __future__ import annotations

import contextlib
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from .base import MemberInfo, ReadError


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ident(value: str, what: str) -> str:
    if not _IDENT_RE.match(value):
        raise ReadError(f"Invalid {what} identifier {value!r}", "member_missing")
    return value


_TUNING = ("mmap_size=2147483648", "cache_size=-131072", "temp_store=MEMORY")


def connect_ro(container: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{container}?mode=ro", uri=True)
    for pragma in _TUNING:

        with contextlib.suppress(sqlite3.Error):
            conn.execute(f"PRAGMA {pragma}")
    return conn


def _connect(container: Path) -> sqlite3.Connection:
    if not container.exists():
        raise ReadError(f"SQLite file does not exist: {container}", "container_missing")
    try:
        return connect_ro(container)
    except sqlite3.Error as exc:
        raise ReadError(f"sqlite cannot be opened {container}: {exc}", "container_missing") from exc


def list_members(
    container: Path, *, table: str, id_column: str, blob_column: str
) -> Iterator[MemberInfo]:
    table = _ident(table, "table")
    id_column = _ident(id_column, "id_column")
    blob_column = _ident(blob_column, "blob_column")
    conn = _connect(container)
    try:
        sql = (
            f"SELECT {id_column}, LENGTH({blob_column}) FROM {table} "
            f"WHERE {blob_column} IS NOT NULL ORDER BY {id_column}"
        )
        for row_id, size in conn.execute(sql):
            yield MemberInfo(
                name=str(row_id),
                size=int(size or 0),
                extra={"table": table, "id": str(row_id), "col": blob_column, "idcol": id_column},
            )
    except sqlite3.Error as exc:
        raise ReadError(f"sqlite Query failed {container}: {exc}", "probe_failed") from exc
    finally:
        conn.close()


def read_member(container: Path, params: dict[str, str]) -> bytes:
    try:
        table = _ident(params["table"], "table")
        blob_column = _ident(params["col"], "col")
        id_column = _ident(params.get("idcol", "id"), "idcol")
        row_id = params["id"]
    except KeyError as exc:
        raise ReadError(f"Missing SQLite parameters: {params}", "member_missing") from exc

    conn = _connect(container)
    try:
        cur = conn.execute(
            f"SELECT {blob_column} FROM {table} WHERE {id_column}=? LIMIT 1", (row_id,)
        )
        row = cur.fetchone()
    except sqlite3.Error as exc:
        raise ReadError(f"sqlite Reading failed {container}: {exc}", "decode_failed") from exc
    finally:
        conn.close()

    if row is None:
        raise ReadError(f"sqlite No such line {container} {table}.{id_column}={row_id}", "member_missing")
    if row[0] is None:
        raise ReadError(f"sqlite BLOB is empty {container} id={row_id}", "empty_audio")
    return bytes(row[0])
