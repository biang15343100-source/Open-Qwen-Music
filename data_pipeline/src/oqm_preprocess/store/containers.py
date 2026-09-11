
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..enums import STORAGE_CLASS
from ..schema import CONTAINER_TABLE_SCHEMA


class ContainerTable:

    def __init__(self) -> None:
        self._by_path: dict[str, int] = {}
        self._rows: list[dict[str, object]] = []
        self._lock = threading.Lock()

    def intern(self, path: Path | str, storage_class: str) -> int:
        key = str(path)
        with self._lock:
            got = self._by_path.get(key)
            if got is not None:
                return got
            cid = len(self._rows)
            size = -1
            mtime = -1
            try:
                st = os.stat(key)
                size, mtime = st.st_size, st.st_mtime_ns
            except OSError:
                pass
            self._by_path[key] = cid
            self._rows.append({
                "container_id": cid,
                "path": key,
                "storage_class": STORAGE_CLASS.code(storage_class, strict=True),
                "size_bytes": size,
                "mtime_ns": mtime,
            })
            return cid

    def path_of(self, container_id: int) -> Path:
        if 0 <= container_id < len(self._rows):
            return Path(str(self._rows[container_id]["path"]))
        raise KeyError(
            f"Unknown container_id {container_id}; table contains {len(self._rows)} containers. "
            "container_table.parquet may be out of sync with stage output; "
            "rerun discover --force to rebuild it."
        )

    def storage_class_of(self, container_id: int) -> int:
        return int(self._rows[container_id]["storage_class"])  # type: ignore[arg-type]

    def to_table(self) -> pa.Table:
        from ..schema import records_to_table

        return records_to_table(self._rows, CONTAINER_TABLE_SCHEMA)

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._assert_extends(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(self.to_table(), tmp, compression="zstd")
        os.replace(tmp, path)

    def _assert_extends(self, path: Path) -> None:
        if not path.exists():
            return
        on_disk = pq.read_table(path, columns=["container_id", "path"]).to_pylist()
        if len(on_disk) > len(self._rows):
            raise ValueError(
                f"Cannot write container_table: disk has {len(on_disk)} containers but "
                f"memory has {len(self._rows)}. Another discover process may have used this "
                "work_dir; overwriting would invalidate existing container_id references."
            )
        for row in on_disk:
            idx = int(row["container_id"])
            mine = str(self._rows[idx]["path"])
            if mine != str(row["path"]):
                raise ValueError(
                    f"Cannot write container_table: container_id {idx} is {row['path']} on "
                    f"disk but {mine} in memory; existing stage outputs would reference the "
                    "wrong container."
                )

    @classmethod
    def read(cls, path: Path) -> ContainerTable:
        table = cls()
        if not Path(path).exists():
            return table
        for row in pq.read_table(path).to_pylist():
            table._by_path[str(row["path"])] = int(row["container_id"])
            table._rows.append(row)

        for idx, row in enumerate(table._rows):
            if int(row["container_id"]) != idx:
                raise ValueError(f"container_table sequence is invalid at row {idx}, id={row['container_id']}")
        return table

    def stats(self) -> dict[str, int]:
        return {"containers": len(self._rows)}

    def __len__(self) -> int:
        return len(self._rows)

    def dump_json(self) -> str:
        return json.dumps(self._rows, ensure_ascii=False)
