
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

_CONFIGURED = False


def setup(level: str = "INFO", log_file: Path | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-5s [%(name)s] %(message)s", datefmt="%H:%M:%S"
    )
    root = logging.getLogger("oqm")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(fmt)
        root.addHandler(handler)

    root.propagate = False
    _CONFIGURED = True


def get(name: str) -> logging.Logger:
    if not _CONFIGURED:
        setup(os.environ.get("OQM_LOG_LEVEL", "INFO"))
    return logging.getLogger(f"oqm.{name}")


def mute_native_stderr() -> None:
    global _CONFIGURED
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
    finally:
        os.close(devnull)

    stream = os.fdopen(saved, "w", buffering=1, errors="replace")
    sys.stderr = stream
    root = logging.getLogger("oqm")
    root.handlers.clear()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-5s [%(name)s] %(message)s", datefmt="%H:%M:%S"
    ))
    root.addHandler(handler)
    root.setLevel(getattr(logging, os.environ.get("OQM_LOG_LEVEL", "INFO").upper(),
                          logging.INFO))
    root.propagate = False
    _CONFIGURED = True


class Progress:

    def __init__(self, logger: logging.Logger, label: str, total: int | None = None,
                 interval_sec: float = 30.0) -> None:
        self._log = logger
        self._label = label
        self._total = total
        self._interval = interval_sec
        self._count = 0
        self._started = time.time()
        self._last = self._started

    def advance(self, n: int = 1) -> None:
        self._count += n
        now = time.time()
        if now - self._last >= self._interval:
            self._last = now
            self._emit(now)

    def _emit(self, now: float) -> None:
        elapsed = max(now - self._started, 1e-6)
        rate = self._count / elapsed
        if self._total:
            pct = 100.0 * self._count / self._total
            eta = (self._total - self._count) / rate if rate > 0 else float("inf")
            self._log.info(
                "%s %d/%d (%.1f%%) %.0f/s used %.0fs Expected remaining %.0fs",
                self._label, self._count, self._total, pct, rate, elapsed, eta,
            )
        else:
            self._log.info("%s %d rows %.0f/s elapsed %.0fs", self._label, self._count, rate, elapsed)

    def done(self) -> float:
        elapsed = time.time() - self._started
        self._log.info("%s completed %d rows in %.1fs", self._label, self._count, elapsed)
        return elapsed

    @property
    def count(self) -> int:
        return self._count
