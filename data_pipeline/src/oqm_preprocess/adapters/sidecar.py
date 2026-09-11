
from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections.abc import Iterator
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from ..readers.sqlite_reader import connect_ro
from ..registry import TEXTDIR_CONTENT, EnrichSpec, JoinSpec
from ..runtime.log import get

log = get("sidecar")


def load_records(spec: EnrichSpec) -> Iterator[dict[str, Any]]:
    if spec.format == "jsonl":
        for path in spec.sidecar_paths():
            yield from _read_jsonl(path)
    elif spec.format in ("csv", "tsv"):
        delimiter = "\t" if spec.format == "tsv" else ","
        for path in spec.sidecar_paths():
            yield from _read_delimited(path, delimiter)
    elif spec.format == "parquet":
        for path in spec.sidecar_paths():
            yield from _read_parquet(path)
    elif spec.format == "sqlite":
        for path in spec.sidecar_paths():
            yield from _read_sqlite(path, str(spec.table))
    elif spec.format == "textdir":
        for path in spec.sidecar_paths():
            yield from _read_textdir(path, spec)
    else:
        raise ValueError(f"load_records does not process {spec.format}")


def _read_textdir(path: Path, spec: EnrichSpec) -> Iterator[dict[str, Any]]:
    excludes = [re.compile(p) for p in spec.exclude_regex]

    def emit(name: str, raw: bytes) -> dict[str, Any] | None:
        if any(pattern.search(name) for pattern in excludes):
            return None
        base = name.rsplit("/", 1)[-1]
        return {
            TEXTDIR_CONTENT: raw.decode("utf-8", "replace"),
            "__name": name,
            "__basename": base,
            "__stem": base.rsplit(".", 1)[0] if "." in base else base,
        }

    suffix = path.suffix.lower()
    if path.is_dir():
        for item in sorted(path.rglob(spec.member_glob)):
            if not item.is_file():
                continue
            record = emit(str(item.relative_to(path)), item.read_bytes())
            if record is not None:
                yield record
    elif suffix in (".zip", ".jar"):
        import zipfile
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not fnmatch(info.filename, spec.member_glob):
                    continue
                record = emit(info.filename, archive.read(info))
                if record is not None:
                    yield record
    elif suffix in (".tar", ".gz", ".tgz", ".bz2", ".xz"):
        import tarfile
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                if not member.isfile() or not fnmatch(member.name, spec.member_glob):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                with handle:
                    record = emit(member.name, handle.read())
                if record is not None:
                    yield record
    else:
        raise ValueError(f"textdir Unknown container {path}")


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    bad = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                if bad <= 3:
                    log.warning("%s:%d JSON parse failed; skipping", path.name, lineno)
                continue
            if isinstance(payload, dict):
                yield payload
    if bad:
        log.warning("%s had %d JSON lines that failed to parse", path.name, bad)


def _read_delimited(path: Path, delimiter: str) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh, delimiter=delimiter):
            yield dict(row)


def _read_parquet(path: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(path)
    for batch in handle.iter_batches(batch_size=8192):
        yield from batch.to_pylist()


def _read_sqlite(path: Path, table: str) -> Iterator[dict[str, Any]]:
    if not table.isidentifier():
        raise ValueError(f"Invalid table name {table!r}")
    conn = connect_ro(path)
    conn.row_factory = sqlite3.Row
    try:
        for row in conn.execute(f"SELECT * FROM {table}"):
            yield {k: v for k, v in dict(row).items() if not isinstance(v, bytes)}
    except sqlite3.Error as exc:
        log.warning("SQLite sidecar read failed for %s.%s: %s", path.name, table, exc)
    finally:
        conn.close()


def read_sqlite_meta(container: Path, table: str, id_column: str,
                     columns: list[str]) -> dict[str, dict[str, Any]]:
    wanted = [c for c in dict.fromkeys([id_column, *columns]) if c.isidentifier()]
    if not wanted:
        return {}
    conn = connect_ro(container)
    try:
        cur = conn.execute(f"SELECT {', '.join(wanted)} FROM {table}")
        names = [d[0] for d in cur.description]
        return {str(row[0]): dict(zip(names, row, strict=True)) for row in cur}
    except sqlite3.Error as exc:
        log.warning("SQLite metadata read failed for %s: %s", container, exc)
        return {}
    finally:
        conn.close()


def normalize_key(value: Any, join: JoinSpec) -> str:
    key = str(value).strip().replace("\\", "/")
    if join.strip_prefix and key.startswith(join.strip_prefix):
        key = key[len(join.strip_prefix):]
    if join.strip_suffix and key.endswith(join.strip_suffix):
        key = key[: -len(join.strip_suffix)]
    return key.lower() if join.lower else key


def record_keys(member: str, local_id: str, join: JoinSpec) -> list[str]:
    member = (member or "").replace("\\", "/")
    base = member.rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0] if "." in base else base
    member_stem = member.rsplit(".", 1)[0] if "." in member else member

    if join.match == "local_id":
        candidates = [local_id]
    elif join.match == "member":
        candidates = [member, member_stem]
    elif join.match == "stem":
        candidates = [member_stem, stem]
    elif join.match == "basename":
        candidates = [base]
    else:  # basename_stem
        candidates = [stem]

    if join.member_pattern:


        pattern = re.compile(join.member_pattern)
        extracted = []
        for candidate in candidates:
            match = pattern.search(candidate)
            if match and (value := next((g for g in match.groups() if g), None)):
                extracted.append(value)
        candidates = extracted

    out: list[str] = []
    for candidate in candidates:
        key = normalize_key(candidate, join)
        if key and key not in out:
            out.append(key)
    return out


def build_index(spec: EnrichSpec) -> tuple[dict[str, dict[str, Any]], int]:
    if spec.join is None:
        return {}, 0
    index: dict[str, dict[str, Any]] = {}
    collisions = 0
    seen = 0
    missing_key = 0
    columns: list[str] = []
    for row in load_records(spec):
        seen += 1
        if not columns:
            columns = list(row)
        raw_key = row.get(spec.join.key_field)
        if raw_key is None:
            missing_key += 1
            continue
        key = normalize_key(raw_key, spec.join)

        if key in index:
            collisions += 1
            continue
        index[key] = row

        if "." in key.rsplit("/", 1)[-1]:
            stem = key.rsplit(".", 1)[0]
            index.setdefault(stem, row)

    if seen and missing_key == seen:
        log.error("Bypass %s of %d There are no records join key %r,optional column:%s",
                  spec.format, seen, spec.join.key_field, columns[:30])
    return index, collisions
