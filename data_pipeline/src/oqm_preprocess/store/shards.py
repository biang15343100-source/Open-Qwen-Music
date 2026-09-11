
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ..schema import records_to_table

SUCCESS_FILE = "_SUCCESS.json"


IO_THREADS = 32


def atomic_write_table(table: pa.Table, path: Path, *, compression: str, level: int,
                       row_group_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(
            table, tmp,
            compression=compression,
            compression_level=level if compression == "zstd" else None,
            row_group_size=row_group_size,
            use_dictionary=True,
            write_statistics=True,
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


@dataclass(slots=True)
class ShardResult:
    key: str
    path: Path
    rows: int
    skipped: bool = False
    elapsed_sec: float = 0.0


class StageStore:

    def __init__(self, root: Path, schema: pa.Schema, *, config_fingerprint: str,
                 compression: str = "zstd", compression_level: int = 7,
                 row_group_size: int = 50000,
                 extra_roots: Sequence[Path] = ()) -> None:
        self.root = Path(root)
        self.schema = schema
        self.config_fingerprint = config_fingerprint
        self.compression = compression
        self.compression_level = compression_level
        self.row_group_size = row_group_size


        self.extra_roots = [Path(p) for p in extra_roots]
        self.root.mkdir(parents=True, exist_ok=True)
        self._names: dict[str, Path] | None = None
        self._parts: list[Path] | None = None

    def _read_roots(self) -> list[Path]:
        return [self.root, *self.extra_roots]

    def listing(self) -> dict[str, Path]:
        if self._names is None:
            def entries(root: Path) -> list[Path]:
                try:
                    return sorted(root.iterdir())
                except FileNotFoundError:
                    return []
            roots = self._read_roots()
            found: dict[str, Path] = {}
            parts: list[Path] = []
            with ThreadPoolExecutor(min(IO_THREADS, len(roots))) as pool:
                for listed in pool.map(entries, roots):
                    for path in listed:
                        found.setdefault(path.name, path)
                        if path.name.startswith("part-") and path.name.endswith(".parquet"):
                            parts.append(path)
            self._names = found
            self._parts = parts
        return self._names

    def _invalidate(self) -> None:
        self._names = None
        self._parts = None

    def shard_path(self, key: str) -> Path:
        return self.root / f"part-{key}.parquet"

    def _marker_path(self, key: str) -> Path:
        return self.root / f".part-{key}.done.json"

    def _find(self, name: str) -> Path | None:
        return self.listing().get(name)

    def read_marker(self, key: str) -> dict[str, Any] | None:
        return self._read_marker_at(self._find(f".part-{key}.done.json"))

    @staticmethod
    def _read_marker_at(marker: Path | None) -> dict[str, Any] | None:
        if marker is None:
            return None
        try:
            return json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def is_done(self, key: str, *, lineage: str | None = None) -> bool:
        payload = self.read_marker(key)
        if payload is None:
            return False
        if payload.get("config_fingerprint") != self.config_fingerprint:
            return False
        if lineage is not None and payload.get("lineage") != lineage:
            return False
        return all(self._find(name) is not None for name in payload.get("parts", []))

    def write_shard(self, key: str, records: list[dict[str, Any]], *,
                    extra: dict[str, Any] | None = None, mark: bool = True,
                    lineage: str | None = None) -> ShardResult:
        started = time.time()
        table = records_to_table(records, self.schema)
        path = self.shard_path(key)
        atomic_write_table(
            table, path,
            compression=self.compression, level=self.compression_level,
            row_group_size=self.row_group_size,
        )
        elapsed = time.time() - started
        self._invalidate()
        if mark:
            self.mark_done(key, rows=table.num_rows, parts=[path.name],
                           elapsed_sec=elapsed, extra=extra, lineage=lineage)
        return ShardResult(key=key, path=path, rows=table.num_rows, elapsed_sec=elapsed)

    def mark_done(self, key: str, *, rows: int, parts: list[str],
                  elapsed_sec: float = 0.0, extra: dict[str, Any] | None = None,
                  lineage: str | None = None) -> None:
        atomic_write_json(
            {
                "key": key,
                "rows": rows,
                "parts": parts,
                "config_fingerprint": self.config_fingerprint,
                "lineage": lineage,
                "elapsed_sec": round(elapsed_sec, 3),
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                **(extra or {}),
            },
            self._marker_path(key),
        )
        self._invalidate()


    def _marker_items(self, prefix: str | None = None) -> list[tuple[str, Path]]:
        head, tail = ".part-", ".done.json"
        items = []
        for name, path in self.listing().items():
            if not (name.startswith(head) and name.endswith(tail)):
                continue
            key = name[len(head):-len(tail)]
            if prefix is None or key == prefix or key.startswith(f"{prefix}-"):
                items.append((key, path))
        return sorted(items)

    def marker_keys(self, prefix: str | None = None) -> list[str]:
        return [key for key, _ in self._marker_items(prefix)]

    def lineage_of(self, prefix: str | None = None) -> str:
        items = self._marker_items(prefix)
        pieces = []
        if items:
            with ThreadPoolExecutor(min(IO_THREADS, len(items))) as pool:
                payloads = pool.map(self._read_marker_at, [p for _, p in items])
                for (key, _), payload in zip(items, payloads, strict=True):
                    payload = payload or {}
                    pieces.append(
                        f"{key}|{payload.get('rows', -1)}|{payload.get('written_at', '-')}")
        if not pieces:
            return "empty"
        digest = hashlib.sha1("\n".join(pieces).encode("utf-8"), usedforsecurity=False)
        return digest.hexdigest()[:16]

    def clear(self) -> int:
        removed = 0
        for path in list(self.root.glob("part-*.parquet")) + \
                list(self.root.glob(".part-*.done.json")):
            path.unlink(missing_ok=True)
            removed += 1
        (self.root / SUCCESS_FILE).unlink(missing_ok=True)
        self._invalidate()
        return removed

    def prune_prefix(self, prefix: str, keep: set[str]) -> int:
        removed = 0
        for key in self.marker_keys(prefix):
            if key in keep:
                continue
            payload = self.read_marker(key) or {}
            for name in payload.get("parts", []):
                (self.root / name).unlink(missing_ok=True)
            self.shard_path(key).unlink(missing_ok=True)
            self._marker_path(key).unlink(missing_ok=True)
            removed += 1


        for name, path in list(self.listing().items()):
            if path.parent != self.root:
                continue
            if not (name.startswith(f"part-{prefix}") and name.endswith(".parquet")):
                continue
            key = name[len("part-"):-len(".parquet")]
            if key != prefix and not key.startswith(f"{prefix}-"):
                continue
            if key not in keep:
                path.unlink(missing_ok=True)
                removed += 1
        if removed:
            self._invalidate()
        return removed

    def shard_paths(self) -> list[Path]:
        self.listing()
        order = {root: i for i, root in enumerate(self._read_roots())}
        return sorted(self._parts or [], key=lambda p: (p.name, order.get(p.parent, 0)))

    def read_all(self, columns: list[str] | None = None) -> pa.Table:
        paths = self.shard_paths()
        if not paths:
            return self.schema.empty_table() if columns is None else pa.table({})
        return pq.read_table(paths, columns=columns)

    def iter_batches(self, columns: list[str] | None = None,
                     batch_size: int = 65536) -> Iterator[pa.RecordBatch]:
        paths = self.shard_paths()
        with ThreadPoolExecutor(IO_THREADS) as pool:
            remaining = iter(paths)
            window = deque(pool.submit(pq.read_table, p, columns=columns)
                           for p in islice(remaining, IO_THREADS))
            for path in remaining:
                yield from window.popleft().result().to_batches(max_chunksize=batch_size)
                window.append(pool.submit(pq.read_table, path, columns=columns))
            while window:
                yield from window.popleft().result().to_batches(max_chunksize=batch_size)

    def count_rows(self) -> int:
        paths = self.shard_paths()
        if not paths:
            return 0
        with ThreadPoolExecutor(min(IO_THREADS, len(paths))) as pool:
            return sum(pool.map(lambda p: pq.ParquetFile(p).metadata.num_rows, paths))

    def finalize(self, stats: dict[str, Any], *, lineage: str | None = None) -> None:
        atomic_write_json(
            {
                "config_fingerprint": self.config_fingerprint,
                "lineage": lineage,
                "shards": len(self.shard_paths()),
                "rows": self.count_rows(),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                **stats,
            },
            self.root / SUCCESS_FILE,
        )

    def success(self) -> dict[str, Any] | None:
        path = self.root / SUCCESS_FILE
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def is_stage_complete(self, *, lineage: str | None = None) -> bool:
        payload = self.success()
        if not payload or payload.get("config_fingerprint") != self.config_fingerprint:
            return False
        return lineage is None or payload.get("lineage") == lineage

    def stage_lineage(self) -> str:
        payload = self.success()
        if not payload:
            return "empty"
        seed = f"{payload.get('rows', -1)}|{payload.get('shards', -1)}|{payload.get('finished_at', '-')}"
        return hashlib.sha1(seed.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
