
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Iterator


def _bad_json(path: Path, line_number: int, exc: json.JSONDecodeError) -> ValueError:
    return ValueError(f"{path}:{line_number} is not legal JSON:{exc}")


class StageStore:

    def __init__(self, root: Path | str, stage: str):
        self.stage = stage
        self.dir = Path(root) / "stages" / stage
        self.dir.mkdir(parents=True, exist_ok=True)

    def shard_path(self, rank: int) -> Path:
        return self.dir / f"rank_{rank:04d}.jsonl"

    def shard_paths(self, ranks: Iterable[int] | None = None) -> list[Path]:

        if ranks is None:
            return sorted(self.dir.glob("rank_*.jsonl"))
        return [
            path
            for path in (self.shard_path(rank) for rank in sorted(set(ranks)))
            if path.exists()
        ]

    @property
    def done_marker(self) -> Path:
        return self.dir / "_SUCCESS.json"

    def iter_records(
        self, *, ranks: Iterable[int] | None = None, strict: bool = True
    ) -> Iterator[dict[str, Any]]:

        for path in self.shard_paths(ranks):
            with path.open("r", encoding="utf-8") as handle:
                broken: tuple[int, json.JSONDecodeError] | None = None
                for line_number, line in enumerate(handle, start=1):
                    if broken is not None:

                        raise _bad_json(path, *broken)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        if strict:
                            raise _bad_json(path, line_number, exc) from exc
                        broken = (line_number, exc)
                        continue
                    yield record

    def load_by_id(self, *, include_errors: bool = False) -> dict[str, dict[str, Any]]:

        result: dict[str, dict[str, Any]] = {}
        for record in self.iter_records():
            sample_id = record.get("sample_id")
            if not sample_id:
                continue


            if record.get("error") and not include_errors:
                continue
            result[str(sample_id)] = record
        return result

    def completed_ids(
        self, *, ranks: Iterable[int] | None = None, strict: bool = True
    ) -> set[str]:

        done: set[str] = set()
        for record in self.iter_records(ranks=ranks, strict=strict):
            sample_id = record.get("sample_id")
            if sample_id:
                done.add(str(sample_id))
        return done


    #


    #


    # `oqm-annotate resume-snapshot`).
    #


    @property
    def resume_snapshot_path(self) -> Path:
        return self.dir / "_resume_ids.json"

    def write_resume_snapshot(self) -> dict[str, Any]:

        ids = sorted(self.completed_ids())
        payload = {
            "stage": self.stage,
            "taken_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "shards": len(self.shard_paths()),
            "count": len(ids),
            "sample_ids": ids,
        }
        _atomic_write_json(self.resume_snapshot_path, payload)
        return payload

    def read_resume_snapshot(self) -> set[str] | None:

        path = self.resume_snapshot_path
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            payload = json.loads(raw)
            ids = payload["sample_ids"]
            if not isinstance(ids, list):
                raise TypeError(f"sample_ids is not a list but {type(ids).__name__}")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Resume snapshot {path} is invalid: {exc}. Delete it and run "
                "`oqm-annotate resume-snapshot` again."
            ) from exc
        return {str(sample_id) for sample_id in ids}

    def reset(self) -> None:
        for path in self.dir.glob("rank_*.jsonl"):
            path.unlink()
        self.done_marker.unlink(missing_ok=True)


        self.resume_snapshot_path.unlink(missing_ok=True)

    def counts(
        self, *, ranks: Iterable[int] | None = None, strict: bool = True
    ) -> tuple[int, int]:

        ok = bad = 0
        for record in self.iter_records(ranks=ranks, strict=strict):
            if record.get("error"):
                bad += 1
            else:
                ok += 1
        return ok, bad

    def mark_done(self, payload: dict[str, Any]) -> None:
        _atomic_write_json(self.done_marker, payload)

    def clear_done(self) -> None:

        self.done_marker.unlink(missing_ok=True)

    def is_done(self) -> bool:
        return self.done_marker.exists()

    def read_done(self) -> dict[str, Any]:

        try:
            payload = json.loads(self.done_marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def product_size(self) -> int:

        total = 0
        for path in self.shard_paths():
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total


class ShardWriter:

    def __init__(self, store: StageStore, rank: int):
        self.store = store
        self.rank = rank
        self.path = store.shard_path(rank)
        self._handle = self.path.open("a", encoding="utf-8")
        self.written = 0
        self.failed = 0

    def write(self, record: dict[str, Any]) -> None:
        if not record.get("sample_id"):
            raise ValueError(f"{self.store.stage} is missing sample_id:{record!r}")
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        if record.get("error"):
            self.failed += 1
        else:
            self.written += 1

    def write_error(self, sample_id: str, exc: BaseException) -> None:

        self.write({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path | str, records: list[dict[str, Any]]) -> None:

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False, suffix=".tmp"
    )
    try:
        with handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    )
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
