
from __future__ import annotations

import errno
import fcntl
import json
import os
import socket
import sys
import time
from pathlib import Path
from types import TracebackType


class WorkDirBusy(RuntimeError):
    pass


class WorkDirLock:
    def __init__(self, work_dir: Path, *, purpose: str) -> None:
        self.path = Path(work_dir) / ".oqm.lock"
        self.purpose = purpose
        self._fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            holder = self._read_holder(fd)
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise WorkDirBusy(
                    f"work_dir is occupied by another process:{holder}."
                    f"Confirm that it has ended before running,or change to work_dir"
                ) from None
            raise
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "purpose": self.purpose,
            "argv": " ".join(sys.argv),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, ensure_ascii=False).encode("utf-8"))
        os.fsync(fd)
        self._fd = fd

    @staticmethod
    def _read_holder(fd: int) -> str:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            raw = os.read(fd, 4096).decode("utf-8", "replace").strip()
            payload = json.loads(raw)
        except (OSError, ValueError):
            return "<unknown process>"
        return (f"pid={payload.get('pid')} host={payload.get('host')} "
                f"started at {payload.get('started_at')} command {payload.get('argv')}")

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> WorkDirLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.release()
