
from __future__ import annotations

import fcntl
import os
import shutil
import time
from pathlib import Path

DEFAULT_CACHE_ROOT = "/dev/shm/oqm-model-cache"


_STAGED_SUFFIXES = (".safetensors", ".json", ".txt", ".model", ".bin")


def _payload_files(source: Path) -> list[Path]:
    return sorted(
        p
        for p in source.iterdir()
        if p.is_file() and p.suffix in _STAGED_SUFFIXES
    )


def _signature(files: list[Path]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _model_signature_once_per_distributed_run(
    source: Path,
    files: list[Path],
    cache_root: Path,
    *,
    wait_seconds: float,
) -> str:
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return _signature(files)
    import hashlib

    session = "|".join(
        (
            str(source.resolve()),
            os.environ.get("RUN_ID", ""),
            os.environ.get("TORCHELASTIC_RUN_ID", ""),
            os.environ.get("MASTER_ADDR", ""),
            os.environ.get("MASTER_PORT", ""),
        )
    )
    key = hashlib.sha256(session.encode()).hexdigest()[:16]
    stamp = cache_root / f".{source.name}-{key}.signature"
    lock_path = cache_root / f".{source.name}-{key}.signature.lock"
    started = time.monotonic()
    with lock_path.open("w") as lock_file:
        _acquire(lock_file, deadline=started + wait_seconds, target=stamp)
        if stamp.is_file():
            return stamp.read_text(encoding="utf-8").strip()
        signature = _signature(files)
        temporary = stamp.with_name(f".{stamp.name}.tmp-{os.getpid()}")
        temporary.write_text(signature + "\n", encoding="utf-8")
        os.replace(temporary, stamp)
        return signature


def _free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def stage_model_dir(
    source: str | Path,
    *,
    cache_root: str | Path | None = DEFAULT_CACHE_ROOT,
    wait_seconds: float = 1800.0,
    log: bool = True,
) -> Path:
    source = Path(source)
    if cache_root is None or not source.is_dir():
        return source

    cache_root = Path(cache_root)
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        _log(log, f"model_stage skipped: cannot create {cache_root} ({error})")
        return source

    files = _payload_files(source)
    if not files:
        return source
    total_bytes = sum(p.stat().st_size for p in files)

    signature = _model_signature_once_per_distributed_run(
        source,
        files,
        cache_root,
        wait_seconds=wait_seconds,
    )
    target = cache_root / f"{source.name}-{signature}"
    done_marker = target / ".staged"
    if done_marker.exists():
        return target


    if _free_bytes(cache_root) < total_bytes * 1.25:
        _log(
            log,
            f"model_stage skipped: insufficient space in {cache_root} "
            f"(requires {total_bytes / 1e9:.1f} GB times 1.25)",
        )
        return source

    lock_path = cache_root / f"{target.name}.lock"
    started = time.monotonic()
    with lock_path.open("w") as lock_file:
        _acquire(lock_file, deadline=started + wait_seconds, target=target)

        if done_marker.exists():
            return target
        _copy(source, target, files, log=log, total_bytes=total_bytes, started=started)
        if _signature(_payload_files(target)) != signature:
            shutil.rmtree(target, ignore_errors=True)
            raise RuntimeError("Model source file is still in staging; refusing a mixed-generation cache")
    return target


def stage_corpus_dir(
    source: str | Path,
    *,
    cache_root: str | Path | None,
    wait_seconds: float = 1800.0,
    log: bool = True,
) -> Path:

    source = Path(source)
    if cache_root is None or not source.is_dir():
        return source
    metadata = source / "corpus.json"
    manifest = source / "manifest.jsonl"
    if not metadata.is_file() or not manifest.is_file():
        return source
    files = [metadata, manifest]
    for relative in ("semantic", "melody", "manifest.jsonl.index"):
        directory = source / relative
        if directory.is_dir():
            files.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    if len(files) <= 2:
        return source

    cache_root = Path(cache_root)
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        _log(log, f"corpus_stage skipped: cannot create {cache_root} ({error})")
        return source
    total_bytes = sum(path.stat().st_size for path in files)
    signature = _tree_signature(source, files)
    target = cache_root / f"{source.name}-{signature}"
    done_marker = target / ".staged"
    if done_marker.exists():
        return target
    if _free_bytes(cache_root) < total_bytes * 1.25:
        _log(
            log,
            f"corpus_stage skipped: insufficient space in {cache_root} "
            f"(requires {total_bytes / 1e9:.1f} GB times 1.25)",
        )
        return source

    lock_path = cache_root / f"{target.name}.lock"
    started = time.monotonic()
    with lock_path.open("w") as lock_file:
        _acquire(lock_file, deadline=started + wait_seconds, target=target)
        if done_marker.exists():
            return target
        staging = target.with_name(target.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        try:
            for path in files:
                destination = staging / path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)


                shutil.copy2(path, destination)
            shutil.rmtree(target, ignore_errors=True)
            os.replace(staging, target)
            (target / ".staged").write_text("ok\n", encoding="utf-8")
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(target, ignore_errors=True)
            raise
    elapsed = max(time.monotonic() - started, 1e-6)
    _log(
        log,
        f"corpus_stage done target={target} gb={total_bytes / 1e9:.2f} "
        f"seconds={elapsed:.1f} mb_per_s={total_bytes / 1e6 / elapsed:.0f}",
    )
    return target


def _tree_signature(root: Path, files: list[Path]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _acquire(lock_file, *, deadline: float, target: Path) -> None:
    while True:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for {target}. If the lock owner terminated, "
                    f"remove {target}.lock and retry."
                ) from None
            time.sleep(1.0)


def _copy(
    source: Path,
    target: Path,
    files: list[Path],
    *,
    log: bool,
    total_bytes: int,
    started: float,
) -> None:


    staging = target.with_name(target.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        for path in files:
            shutil.copyfile(path, staging / path.name)


        shutil.rmtree(target, ignore_errors=True)
        os.replace(staging, target)
        (target / ".staged").write_text("ok\n", encoding="utf-8")
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(target, ignore_errors=True)
        raise
    elapsed = max(time.monotonic() - started, 1e-6)
    _log(
        log,
        f"model_stage done target={target} gb={total_bytes / 1e9:.2f} "
        f"seconds={elapsed:.1f} mb_per_s={total_bytes / 1e6 / elapsed:.0f}",
    )


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)
