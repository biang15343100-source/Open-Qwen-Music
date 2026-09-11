
from __future__ import annotations

import os
import re
import signal
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .. import audio as audio_mod
from .. import flags, ids, locator
from ..audio import DecodeError
from ..config import PipelineConfig
from ..enums import CODEC, DURATION_SOURCE, STATUS, STORAGE_CLASS
from ..readers import parquet_reader, read_bytes, tar_reader
from ..readers.base import HANDLES, ReadError
from ..runtime.context import Context
from ..runtime.log import Progress, get, mute_native_stderr
from ..runtime.reaper import die_with_parent
from ..schema import WORK_SCHEMA

log = get("probe")

STAGE = "s2_probe"
INPUT_STAGE = "s1_discover"


TRUNCATION_TOLERANCE = 0.02


@dataclass(slots=True)
class ProbeTask:

    key: str
    dataset_id: int
    container_id: int
    storage_class: str
    units: list[tuple[str, int]] = field(default_factory=list)


    bytes_hint: int = 0

    def __len__(self) -> int:
        return len(self.units)

    def cost(self) -> int:
        return max(self.bytes_hint, len(self.units) * _ASSUMED_UNKNOWN_BYTES)


def lineage_for(ctx: Context, slug: str) -> str:
    src = ctx.store(INPUT_STAGE, WORK_SCHEMA)
    return f"{src.lineage_of(slug)}+analysis-v{audio_mod.ANALYSIS_VERSION}"


def _passwords_of(ctx: Context, datasets: list[str] | None) -> dict[int, bytes]:
    out: dict[int, bytes] = {}
    for spec in ctx.registry.select(datasets):
        for source in spec.sources:
            secret = source.secret()
            if secret:
                out[spec.dataset_id] = secret
    return out


_ENCRYPTED_CHUNK = 2000


_ASSUMED_UNKNOWN_BYTES = 1 << 20


def assign_shard(tasks: list[ProbeTask], index: int, total: int) -> list[ProbeTask]:
    if total <= 1:
        return tasks
    order = sorted(tasks, key=lambda t: (-t.cost(), t.key))
    loads = [0] * total
    buckets: list[list[ProbeTask]] = [[] for _ in range(total)]
    for task in order:
        pick = min(range(total), key=lambda b: (loads[b], b))
        buckets[pick].append(task)
        loads[pick] += task.cost()
    chosen = sorted(buckets[index], key=lambda t: t.key)
    log.info(
        "Sharding %d/%d:%d tasks,about %.1f GB(each piece %s GB)",
        index, total, len(chosen), loads[index] / 1e9,
        " ".join(f"{x/1e9:.0f}" for x in loads),
    )
    return chosen


def build_tasks(ctx: Context, datasets: list[str] | None) -> list[ProbeTask]:
    src = ctx.store(INPUT_STAGE, WORK_SCHEMA)
    wanted_ids = {spec.dataset_id for spec in ctx.registry.select(datasets)}
    encrypted_ids = set(_passwords_of(ctx, datasets))
    shard_size = ctx.cfg.runtime.shard_size
    task_bytes = ctx.cfg.runtime.probe_task_bytes


    member_regex = os.environ.get("OQM_PROBE_MEMBER_REGEX")
    member_pattern = re.compile(member_regex, re.IGNORECASE) if member_regex else None


    buckets: dict[tuple[int, int, int], list[tuple[str, int, int]]] = defaultdict(list)
    storage_of: dict[tuple[int, int, int], str] = {}

    for path in src.shard_paths():
        columns = ["dataset_id", "container_id", "storage_class", "parquet_row_group",
                   "member_size"]
        if member_pattern:
            columns.append("member")
        table = pq.read_table(path, columns=columns)
        ds_col = table.column("dataset_id").to_pylist()
        cid_col = table.column("container_id").to_pylist()
        sc_col = table.column("storage_class").to_pylist()
        rg_col = table.column("parquet_row_group").to_pylist()
        size_col = table.column("member_size").to_pylist()
        member_col = (table.column("member").to_pylist() if member_pattern
                      else [None] * table.num_rows)
        for row, (ds, cid, sc, rg, msize, member) in enumerate(
                zip(ds_col, cid_col, sc_col, rg_col, size_col, member_col, strict=True)):
            if ds not in wanted_ids:
                continue
            if member_pattern and not member_pattern.search(member or ""):
                continue
            storage = STORAGE_CLASS.name_of(int(sc))

            key = (int(ds), int(cid), int(rg) if storage == "parquet" else -1)
            buckets[key].append((path.name, row, int(msize or 0)))
            storage_of[key] = storage

    tasks: list[ProbeTask] = []
    for key, entries in sorted(buckets.items()):
        dataset_id, container_id, row_group = key
        storage = storage_of[key]
        entries.sort()
        units = [(name, row) for name, row, _ in entries]
        sizes = [size for _, _, size in entries]

        if storage == "targz":
            chunk = len(units)
        elif dataset_id in encrypted_ids:
            chunk = _ENCRYPTED_CHUNK
        else:
            chunk = shard_size
        spans = (_spans_by_count(len(units), chunk) if storage == "targz" or not task_bytes
                 else _spans_by_bytes(sizes, chunk, task_bytes))
        for part, (start, stop) in enumerate(spans):
            tasks.append(ProbeTask(
                key=f"d{dataset_id:03d}-c{container_id:06d}-g{row_group + 1:05d}-{part:04d}",
                dataset_id=dataset_id,
                container_id=container_id,
                storage_class=storage,
                units=units[start:stop],
                bytes_hint=sum(sizes[start:stop]),
            ))
    return tasks


def _spans_by_count(total: int, chunk: int) -> list[tuple[int, int]]:
    return [(start, min(start + chunk, total)) for start in range(0, total, chunk)]


def _spans_by_bytes(sizes: list[int], chunk: int, budget: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    acc = 0
    for i, size in enumerate(sizes):


        weight = size or _ASSUMED_UNKNOWN_BYTES
        taken = i - start
        if taken and (acc + weight > budget or taken >= chunk):
            spans.append((start, i))
            start, acc = i, 0
        acc += weight
    if start < len(sizes):
        spans.append((start, len(sizes)))
    return spans


class ShardCache:

    def __init__(self, root: Path, capacity: int = 2) -> None:
        self._root = root
        self._capacity = capacity
        self._cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    def rows(self, shard_name: str) -> list[dict[str, Any]]:
        got = self._cache.get(shard_name)
        if got is not None:
            self._cache.move_to_end(shard_name)
            return got
        rows = pq.read_table(self._root / shard_name).to_pylist()
        self._cache[shard_name] = rows
        self._cache.move_to_end(shard_name)
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return rows


class _Timeout(Exception):
    pass


def _on_alarm(signum: int, frame: Any) -> None:
    raise _Timeout()


class _Deadline:

    def __init__(self, seconds: int) -> None:
        self._seconds = seconds
        self._enabled = seconds > 0 and hasattr(signal, "SIGALRM")
        self._previous: Any = None

    def __enter__(self) -> _Deadline:
        if self._enabled:
            self._previous = signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(self._seconds)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._enabled:
            signal.alarm(0)
            if self._previous is not None:
                signal.signal(signal.SIGALRM, self._previous)


def _mark(record: dict[str, Any], *flag_names: str) -> None:
    for name in flag_names:
        record["flags"] = int(record.get("flags", 0)) | flags.bit(name)


def probe_one(record: dict[str, Any], payload: bytes, name_hint: str,
              timeout_sec: int = 0) -> dict[str, Any]:
    record["file_bytes"] = len(payload)
    if not payload:
        _mark(record, "zero_bytes")
        return record

    record["content_hash_l1"] = ids.content_hash_l1(payload)

    try:
        with _Deadline(timeout_sec):
            info = audio_mod.probe(payload, name_hint)
            analysis = audio_mod.analyze(payload, info, name_hint)
    except _Timeout:
        _mark(record, "decode_failed")
        return record
    except DecodeError as exc:
        _mark(record, exc.flag)
        return record
    except (MemoryError, ValueError, RuntimeError, OSError) as exc:
        log.warning("Detection exception %s: %s", name_hint, exc)
        _mark(record, "decode_failed")
        return record

    info = analysis.info
    record["sample_rate_hz"] = info.sample_rate
    record["channels"] = min(127, info.channels)
    record["codec"] = CODEC.code(_codec_name(info))
    record["duration_sec"] = float(info.duration_sec)
    record["duration_source"] = DURATION_SOURCE.code(analysis.duration_source, strict=True)


    header = analysis.header_duration_sec
    if header > 0 and info.duration_sec < header * (1.0 - TRUNCATION_TOLERANCE):
        _mark(record, "truncated")

    if analysis.empty:
        _mark(record, "empty_audio")
        return record
    if analysis.nonfinite:
        _mark(record, "nonfinite_audio")
        return record

    record["peak_dbfs"] = analysis.peak_dbfs
    record["rms_dbfs"] = analysis.rms_dbfs
    record["near_silent_frame_ratio"] = analysis.near_silent_frame_ratio
    record["clipping_ratio"] = analysis.clipping_ratio
    record["channel_correlation"] = (
        None if analysis.channel_correlation != analysis.channel_correlation
        else analysis.channel_correlation
    )
    record["effective_bandwidth_ratio"] = analysis.effective_bandwidth_ratio
    record["lead_silence_sec"] = analysis.lead_silence_sec
    record["tail_silence_sec"] = analysis.tail_silence_sec
    record["fingerprint_l2"] = analysis.fingerprint or None
    return record


_SUBTYPE_TO_CODEC = {
    "PCM_16": "pcm_s16le", "PCM_24": "pcm_s24le", "PCM_32": "pcm_s32le",
    "PCM_U8": "pcm_u8", "FLOAT": "pcm_f32le", "DOUBLE": "pcm_f64le",
    "MPEG_LAYER_III": "mp3", "VORBIS": "vorbis", "OPUS": "opus", "ALAC_16": "alac",
}


def _codec_name(info: audio_mod.AudioInfo) -> str:
    if info.backend == "ffmpeg":
        return str(info.subtype).lower()
    if info.format == "FLAC":
        return "flac"
    return _SUBTYPE_TO_CODEC.get(info.subtype, str(info.subtype).lower())


_WORKER: dict[str, Any] = {}


def _init_worker(work_dir: str, input_dir: str, container_path: str, timeout: int,
                 handle_cache: int, prefetch: int = 1,
                 prefetch_bytes: int = 268435456,
                 passwords: dict[int, bytes] | None = None) -> None:
    from ..store.containers import ContainerTable

    HANDLES.configure(handle_cache)
    _WORKER["shards"] = ShardCache(Path(input_dir))
    _WORKER["containers"] = ContainerTable.read(Path(container_path))
    _WORKER["timeout"] = timeout
    _WORKER["work_dir"] = work_dir
    _WORKER["prefetch"] = prefetch
    _WORKER["prefetch_bytes"] = prefetch_bytes
    _WORKER["passwords"] = passwords or {}


def _init_pool_worker(*args: Any) -> None:
    die_with_parent()
    mute_native_stderr()
    _init_worker(*args)


def run_task(task: ProbeTask) -> tuple[list[dict[str, Any]], float]:
    started = time.time()
    shards: ShardCache = _WORKER["shards"]
    containers = _WORKER["containers"]
    timeout = int(_WORKER["timeout"])

    records: list[dict[str, Any]] = []
    for shard_name, row in task.units:
        records.append(dict(shards.rows(shard_name)[row]))

    container = containers.path_of(task.container_id)
    if not container.exists():
        for record in records:
            _mark(record, "container_missing")
        return [_finalize(r) for r in records], time.time() - started

    if task.storage_class == "parquet":
        _run_parquet(task, records, container, timeout)
    elif task.storage_class == "targz":
        _run_targz(task, records, container, timeout)
    else:
        _run_generic(records, containers, timeout, int(_WORKER.get("prefetch", 1)),
                     int(_WORKER.get("prefetch_bytes", 268435456)),
                     _WORKER.get("passwords", {}).get(task.dataset_id))
    return [_finalize(r) for r in records], time.time() - started


def _finalize(record: dict[str, Any]) -> dict[str, Any]:
    mask = int(record.get("flags", 0))
    record["status"] = STATUS.code("rejected" if flags.is_rejected(mask) else "pending_probe")
    return record


_ASSUMED_MEMBER_BYTES = 8 << 20


def _run_generic(records: list[dict[str, Any]], containers: Any, timeout: int,
                 prefetch: int = 1, budget_bytes: int = 256 << 20,
                 password: bytes | None = None) -> None:
    def fetch(record: dict[str, Any]) -> tuple[bytes, str]:
        ref = locator.to_ref(record, containers)
        hints = locator.to_hints(record)
        return read_bytes(ref, hints, password), locator.name_hint(record, containers)

    def consume(record: dict[str, Any], result: Any) -> None:
        try:
            payload, hint_name = result()
        except ReadError as exc:
            log.debug("Reading failed %s: %s", locator.name_hint(record, containers), exc)
            _mark(record, exc.flag)
            return
        probe_one(record, payload, hint_name, timeout)

    if prefetch <= 1 or len(records) < 2:
        for record in records:
            consume(record, lambda r=record: fetch(r))
        return

    def size_of(record: dict[str, Any]) -> int:
        size = locator.int_field(record, "member_size")
        return size if size > 0 else _ASSUMED_MEMBER_BYTES

    with ThreadPoolExecutor(max_workers=prefetch, thread_name_prefix="probe-read") as pool:
        window: deque[tuple[dict[str, Any], int, Future[tuple[bytes, str]]]] = deque()
        remaining = iter(records)
        upcoming = next(remaining, None)
        inflight = 0
        while True:
            while upcoming is not None and len(window) < prefetch:
                size = size_of(upcoming)

                if window and inflight + size > budget_bytes:
                    break
                window.append((upcoming, size, pool.submit(fetch, upcoming)))
                inflight += size
                upcoming = next(remaining, None)
            if not window:
                break
            record, size, future = window.popleft()
            inflight -= size
            consume(record, future.result)


def _run_parquet(task: ProbeTask, records: list[dict[str, Any]], container: Path,
                 timeout: int) -> None:
    by_rowgroup: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_rowgroup[locator.int_field(record, "parquet_row_group")].append(record)

    for row_group, group in by_rowgroup.items():
        column = locator.parse_params(group[0].get("locator_params")).get("col")
        if not column:
            for record in group:
                _mark(record, "probe_failed")
            continue
        wanted = {locator.int_field(r, "parquet_row_index"): r for r in group}
        try:
            for index, payload in parquet_reader.stream_rowgroup(container, row_group, column):
                record = wanted.pop(index, None)
                if record is None:
                    continue


                probe_one(record, payload or b"", f"rg{row_group}/row{index}.wav", timeout)
        except ReadError as exc:
            log.warning("parquet Group read failed %s rg=%s: %s", container, row_group, exc)
            for record in wanted.values():
                _mark(record, exc.flag)
            continue
        for record in wanted.values():
            _mark(record, "member_missing")


def _run_targz(task: ProbeTask, records: list[dict[str, Any]], container: Path,
               timeout: int) -> None:
    wanted = {str(r.get("member") or ""): r for r in records}
    try:
        for name, payload in tar_reader.stream_members(container, list(wanted)):
            record = wanted.pop(name, None)
            if record is None:
                continue
            probe_one(record, payload, name.rsplit("/", 1)[-1], timeout)
    except ReadError as exc:
        log.warning("targz Streaming read failed %s: %s", container, exc)
        for record in wanted.values():
            _mark(record, exc.flag)
        return
    for record in wanted.values():
        _mark(record, "member_missing")


def run(ctx: Context, datasets: list[str] | None = None, *, force: bool = False,
        workers: int | None = None, shard: tuple[int, int] | None = None) -> dict[str, Any]:
    cfg: PipelineConfig = ctx.cfg
    src = ctx.store(INPUT_STAGE, WORK_SCHEMA)
    if not src.shard_paths():
        raise RuntimeError(f"{INPUT_STAGE} has no output; run discover first")

    store = ctx.store(STAGE, WORK_SCHEMA)
    tasks = build_tasks(ctx, datasets)
    lineages = {
        spec.dataset_id: lineage_for(ctx, spec.slug)
        for spec in ctx.registry.select(datasets)
    }
    if shard is not None:
        tasks = assign_shard(tasks, *shard)


        log.info("Sharded mode; skipping stale-fragment cleanup")
    else:
        stale = _prune_stale(store, tasks, ctx, datasets)
        if stale:
            log.warning("Removed %d probe fragments with changed or missing upstream inputs", stale)
    pending = [t for t in tasks
               if force or not store.is_done(t.key, lineage=lineages.get(t.dataset_id))]
    total_units = sum(len(t) for t in pending)
    log.info(
        "S2 probe:%d tasks(%d Completed Skip),total %d items to be detected",
        len(pending), len(tasks) - len(pending), total_units,
    )
    if not pending:
        store.finalize({"tasks": len(tasks), "skipped": len(tasks)})
        return {"tasks": len(tasks), "probed": 0, "skipped": len(tasks)}

    n_workers = workers if workers is not None else cfg.runtime.workers
    init_args = (
        str(cfg.work_dir), str(src.root), str(Context.container_path(cfg)),
        cfg.runtime.probe_timeout_sec, cfg.runtime.handle_cache,
        cfg.runtime.probe_prefetch, cfg.runtime.probe_prefetch_bytes,
        _passwords_of(ctx, datasets),
    )
    progress = Progress(log, "S2 probe", total=total_units, interval_sec=20.0)
    probed = 0
    failures = 0

    slowest: list[tuple[float, str, int]] = []
    for task, (records, task_sec) in _execute(pending, n_workers, init_args):
        if records is None:
            failures += 1
            continue
        store.write_shard(task.key, records, extra={
            "dataset_id": task.dataset_id, "container_id": task.container_id,
            "probe_sec": round(task_sec, 3),
            "items_per_sec": round(len(records) / task_sec, 2) if task_sec > 0 else None,
        }, lineage=lineages.get(task.dataset_id))
        slowest.append((task_sec, task.key, len(records)))
        probed += len(records)
        progress.advance(len(records))
    progress.done()


    slowest.sort(reverse=True)
    for seconds, key, rows in slowest[:5]:
        log.info("Slowest task %s: %d items in %.1fs (%.1f items/s)",
                 key, rows, seconds, rows / seconds if seconds > 0 else 0.0)

    stats = {
        "tasks": len(tasks), "probed": probed,
        "skipped": len(tasks) - len(pending), "task_failures": failures,
        "slowest_tasks": [
            {"key": k, "rows": n, "sec": round(s, 1)} for s, k, n in slowest[:10]
        ],
    }
    store.finalize(stats)
    if failures:
        raise RuntimeError(f"{failures} probe tasks failed; output is incomplete and cannot proceed")
    return stats


def _prune_stale(store: Any, tasks: list[ProbeTask], ctx: Context,
                 datasets: list[str] | None) -> int:
    keep: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        keep[f"d{task.dataset_id:03d}"].add(task.key)
    removed = 0
    for spec in ctx.registry.select(datasets):
        prefix = f"d{spec.dataset_id:03d}"
        removed += store.prune_prefix(prefix, keep.get(prefix, set()))
    return removed


def _execute(
    tasks: list[ProbeTask], n_workers: int, init_args: tuple[Any, ...],
) -> Iterator[tuple[ProbeTask, tuple[list[dict[str, Any]] | None, float]]]:
    if n_workers <= 1:
        _init_worker(*init_args)
        for task in tasks:
            yield task, _guard(task)
        return

    import multiprocessing as mp


    context = mp.get_context("spawn")
    with context.Pool(n_workers, initializer=_init_pool_worker, initargs=init_args) as pool:
        for task, records in pool.imap_unordered(_guard_pair, tasks, chunksize=1):
            yield task, records


def _guard(task: ProbeTask) -> tuple[list[dict[str, Any]] | None, float]:
    try:
        return run_task(task)
    except Exception:  # noqa: BLE001 -
        log.exception("The overall detection task failed key=%s container_id=%s", task.key, task.container_id)
        return None, 0.0


def _guard_pair(task: ProbeTask) -> tuple[ProbeTask, tuple[list[dict[str, Any]] | None, float]]:
    return task, _guard(task)
