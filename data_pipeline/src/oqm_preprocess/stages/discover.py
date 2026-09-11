
from __future__ import annotations

import multiprocessing
import re
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import flags, ids, readers
from ..enums import (
    CONFIDENCE,
    CONTENT_TYPE,
    DOMAIN,
    GRANULARITY,
    LANGUAGE,
    LANGUAGE_EVIDENCE,
    LICENSE_FAMILY,
    STATUS,
    STEM_ROLE,
    STORAGE_CLASS,
    SYNTHETIC_MODEL,
)
from ..locator import format_params
from ..readers import parquet_reader
from ..readers.base import MemberInfo, ReadError
from ..registry import DatasetSpec, SourceSpec
from ..runtime.context import Context
from ..runtime.log import Progress, get
from ..runtime.reaper import die_with_parent
from ..schema import WORK_SCHEMA, new_record

log = get("discover")

STAGE = "s1_discover"


class ContractError(RuntimeError):
    pass


@dataclass(slots=True)
class DatasetReport:
    dataset_id: int
    slug: str
    containers_total: int = 0
    containers_missing: int = 0
    items: int = 0
    skipped_non_audio: int = 0
    skipped_by_regex: int = 0
    uid_collisions: int = 0
    expected: int | None = None
    count_ok: bool = True
    elapsed_sec: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "slug": self.slug,
            "containers_total": self.containers_total,
            "containers_missing": self.containers_missing,
            "items": self.items,
            "skipped_non_audio": self.skipped_non_audio,
            "skipped_by_regex": self.skipped_by_regex,
            "uid_collisions": self.uid_collisions,
            "expected": self.expected,
            "count_ok": self.count_ok,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "errors": self.errors[:20],
        }


def _container_label(container: Path, source: SourceSpec) -> str:
    if source.root:
        try:
            return str(container.relative_to(source.root))
        except ValueError:
            pass
    return container.name


def _make_local_id(storage_class: str, label: str, member: MemberInfo) -> str:
    if storage_class == "loose":
        return member.name
    if storage_class == "parquet":
        rg = member.extra.get("rg", -1)
        row = member.extra.get("row", -1)

        return f"{label}::rg{rg}/row{row}"
    if storage_class == "sqlite":
        return f"{label}::{member.extra.get('id', member.name)}"
    return f"{label}::{member.name}"


def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns]


def enumerate_container(
    spec: DatasetSpec, source: SourceSpec, container: Path, container_id: int,
    license_code: int,
) -> tuple[list[dict[str, Any]], int, str]:
    records: list[dict[str, Any]] = []
    skipped = 0
    try:
        for item in _member_records(spec, source, container, container_id, license_code):
            if "__skip__" in item:
                skipped += 1
            else:
                records.append(item)
    except ReadError as exc:
        return records, skipped, f"{container}: {exc}"
    return records, skipped, ""


def _member_records(
    spec: DatasetSpec, source: SourceSpec, container: Path, container_id: int,
    license_code: int,
) -> Iterator[dict[str, Any]]:
    label = _container_label(container, source)
    storage_class = source.storage_class
    include = _compile(source.include_regex)
    exclude = _compile(source.exclude_regex)

    options: dict[str, Any] = {}
    locator_params: dict[str, str] = {}
    if storage_class == "parquet":
        column = source.audio_column or parquet_reader.detect_audio_column(container)
        options = {"audio_column": column, "meta_columns": source.meta_columns}
        locator_params = {"col": column}
    elif storage_class == "sqlite":
        options = {
            "table": source.table,
            "id_column": source.id_column,
            "blob_column": source.blob_column,
        }
        locator_params = {
            "table": str(source.table),
            "idcol": str(source.id_column),
            "col": str(source.blob_column),
        }
    else:
        options = {"audio_only": source.audio_only}
        if storage_class == "loose" and source.member_glob:
            options["member_glob"] = source.member_glob
    params_text = format_params(locator_params)

    for member in readers.enumerate_members(storage_class, container, **options):
        name = member.name
        if exclude and any(p.search(name) for p in exclude):
            yield {"__skip__": "regex"}
            continue
        if include and not any(p.search(name) for p in include):
            yield {"__skip__": "regex"}
            continue
        yield _to_record(
            spec, member, container_id, label, storage_class, license_code, params_text
        )


def _to_record(
    spec: DatasetSpec, member: MemberInfo, container_id: int, label: str,
    storage_class: str, license_code: int, locator_params: str | None,
) -> dict[str, Any]:
    local_id = _make_local_id(storage_class, label, member)
    record = new_record(
        uid=ids.make_uid(spec.slug, local_id),
        dataset_id=spec.dataset_id,
        local_id=local_id,
        storage_class=STORAGE_CLASS.code(storage_class, strict=True),
        container_id=container_id,
        member=member.name,
        member_offset=member.data_offset,
        member_header_offset=member.header_offset,
        member_size=member.size,
        member_compressed_size=(
            member.compressed_size if member.compress_method != 0 else None
        ),
        member_compress=int(member.compress_method) & 0x7F,
        sequential_only=storage_class == "targz",
        locator_params=locator_params,
        file_bytes=member.size if member.size >= 0 else None,
        content_type=CONTENT_TYPE.code(spec.content_type),
        granularity=GRANULARITY.code(spec.granularity),
        domain=DOMAIN.code(spec.domain),
        stem_role=STEM_ROLE.code(spec.stem_role),
        is_synthetic=spec.is_synthetic,
        synthetic_model=SYNTHETIC_MODEL.code(spec.synthetic_model),
        is_derived=spec.is_derived,
        language=LANGUAGE.code(spec.language.default),
        language_confidence=CONFIDENCE.code(spec.language.confidence),
        language_evidence=LANGUAGE_EVIDENCE.code(spec.language.evidence),
        license_id=license_code,
        license_family=LICENSE_FAMILY.code(spec.license.family),
        commercial_ok=spec.license.commercial_ok,
        status=STATUS.code("pending_probe", strict=True),
        flags=0,
    )
    if storage_class == "parquet":
        record["parquet_row_group"] = int(member.extra.get("rg", -1))
        record["parquet_row_index"] = int(member.extra.get("row", -1))
    if not spec.license.commercial_ok:
        record["flags"] |= flags.bit("license_restricted")
    if spec.license.family == "unknown":
        record["flags"] |= flags.bit("license_unknown")
    return record


@dataclass
class _Plan:

    spec: DatasetSpec
    report: DatasetReport
    lineage: str
    license_code: int
    tasks: list[tuple[SourceSpec, Path, int]]
    started: float


def _plan(ctx: Context, spec: DatasetSpec, *, force: bool) -> _Plan | DatasetReport:
    started = time.time()
    report = DatasetReport(dataset_id=spec.dataset_id, slug=spec.slug,
                           expected=spec.expected_item_count)
    store = ctx.store(STAGE, WORK_SCHEMA)
    lineage = spec.fingerprint(scope="source")

    if spec.metadata_only:
        store.mark_done(spec.slug, rows=0, parts=[], extra={"metadata_only": True},
                        lineage=lineage)
        report.elapsed_sec = time.time() - started
        log.info("[%s] Metadata only; no local audio, skipping enumeration", spec.slug)
        return report

    if not force and store.is_done(spec.slug, lineage=lineage):
        marker = store.read_marker(spec.slug) or {}
        report.items = int(marker.get("rows", 0))
        report.containers_total = int(marker.get("containers_total", 0))
        report.count_ok = spec.count_ok(report.items)
        report.elapsed_sec = time.time() - started
        log.info("[%s] already complete; skipping (%d rows)", spec.slug, report.items)
        return report

    tasks: list[tuple[SourceSpec, Path, int]] = []
    for source in spec.sources:
        containers = source.containers()
        report.containers_total += len(containers)
        for container in containers:
            if not container.exists():
                report.containers_missing += 1
                report.errors.append(f"container is missing: {container}")
                log.error("[%s] Container missing %s", spec.slug, container)
                continue
            tasks.append((source, container, ctx.containers.intern(container, source.storage_class)))

    return _Plan(spec=spec, report=report, lineage=lineage,
                 license_code=ctx.license_code(spec.license.id), tasks=tasks, started=started)


def _consume(ctx: Context, plan: _Plan,
             results: Iterable[tuple[list[dict[str, Any]], int, str]]) -> DatasetReport:
    spec, report = plan.spec, plan.report
    store = ctx.store(STAGE, WORK_SCHEMA)
    shard_size = ctx.cfg.runtime.shard_size

    buffer: list[dict[str, Any]] = []
    parts: list[str] = []
    keys: set[str] = set()
    uids: list[int] = []
    part_seq = 0
    progress = Progress(log, f"[{spec.slug}] enumerating", total=spec.expected_item_count)

    def flush() -> None:
        nonlocal part_seq, buffer
        if not buffer:
            return
        key = f"{spec.slug}-{part_seq:05d}"
        result = store.write_shard(key, buffer, mark=False)
        parts.append(result.path.name)
        keys.add(key)
        part_seq += 1
        buffer = []

    for records, skipped, error in results:
        report.skipped_by_regex += skipped
        if error:
            report.errors.append(error)
            log.error("[%s] Enumeration failed %s", spec.slug, error)
        for item in records:
            buffer.append(item)
            uids.append(int.from_bytes(item["uid"], "big"))
            report.items += 1
            progress.advance()
            if len(buffer) >= shard_size:
                flush()

    flush()
    progress.done()

    report.uid_collisions = _count_uid_collisions(uids)
    if report.uid_collisions:
        raise ContractError(
            f"[{spec.slug}] found {report.uid_collisions} UID collisions; "
            "local_id must be unique within each dataset"
        )

    report.count_ok = spec.count_ok(report.items)
    report.elapsed_sec = time.time() - plan.started
    stale = store.prune_prefix(spec.slug, keys | {spec.slug})
    if stale:
        log.warning("[%s] removed %d stale shards", spec.slug, stale)


    ctx.save_containers()
    store.mark_done(
        spec.slug, rows=report.items, parts=parts,
        elapsed_sec=report.elapsed_sec, extra=report.as_dict(), lineage=plan.lineage,
    )
    if not report.count_ok:
        log.error(
            "[%s] enumerator %d and expected %s Deviation exceeds %.1f%%",
            spec.slug, report.items, spec.expected_item_count, spec.expected_tolerance * 100,
        )
    return report


def discover_dataset(ctx: Context, spec: DatasetSpec, *, force: bool = False) -> DatasetReport:
    plan = _plan(ctx, spec, force=force)
    if isinstance(plan, DatasetReport):
        return plan
    results = _enumerate_all(plan.tasks, spec, plan.license_code,
                             workers=ctx.cfg.runtime.discover_workers)
    return _consume(ctx, plan, results)


def _enumerate_all(
    tasks: list[tuple[SourceSpec, Path, int]], spec: DatasetSpec, license_code: int,
    *, workers: int,
) -> Iterator[tuple[list[dict[str, Any]], int, str]]:
    if not tasks:
        return
    if workers <= 1 or len(tasks) == 1:
        for source, container, container_id in tasks:
            yield enumerate_container(spec, source, container, container_id, license_code)
        return


    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=context,
                             initializer=die_with_parent) as pool:
        futures = [
            pool.submit(enumerate_container, spec, source, container, container_id, license_code)
            for source, container, container_id in tasks
        ]
        for future, (_, container, _) in zip(futures, tasks, strict=True):
            try:
                yield future.result()
            except Exception as exc:
                yield [], 0, f"{container}: enumeration subprocess failed {exc!r}"


def _count_uid_collisions(uid_ints: list[int]) -> int:
    if len(uid_ints) < 2:
        return 0
    arr = np.fromiter(uid_ints, dtype=np.uint64, count=len(uid_ints))
    arr.sort()
    return int((arr[1:] == arr[:-1]).sum())


def _run_all(ctx: Context, specs: list[DatasetSpec], *, force: bool) -> list[DatasetReport]:
    plans: list[_Plan] = []
    done: dict[str, DatasetReport] = {}
    for spec in specs:
        planned = _plan(ctx, spec, force=force)
        if isinstance(planned, DatasetReport):
            done[spec.slug] = planned
        else:
            plans.append(planned)

    pending = [p for p in plans if p.tasks]
    total_tasks = sum(len(p.tasks) for p in pending)
    workers = min(ctx.cfg.runtime.discover_workers, max(total_tasks, 1))

    if total_tasks and workers > 1:
        log.info("Enumerating %d containers (%d datasets, %d workers)",
                 total_tasks, len(pending), workers)
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                 initializer=die_with_parent) as pool:
            futures = {
                plan.spec.slug: [
                    pool.submit(enumerate_container, plan.spec, source, container,
                                container_id, plan.license_code)
                    for source, container, container_id in plan.tasks
                ]
                for plan in pending
            }


            remaining = list(pending)
            while remaining:
                ready = [p for p in remaining
                         if all(f.done() for f in futures[p.spec.slug])]
                if not ready:


                    unfinished = [f for p in remaining for f in futures[p.spec.slug]
                                  if not f.done()]
                    if not unfinished:
                        continue
                    wait(unfinished, return_when=FIRST_COMPLETED)
                    continue
                for plan in ready:
                    done[plan.spec.slug] = _consume(
                        ctx, plan, _collect(futures.pop(plan.spec.slug), plan.tasks)
                    )
                    remaining.remove(plan)
    else:
        for plan in pending:
            done[plan.spec.slug] = _consume(
                ctx, plan,
                _enumerate_all(plan.tasks, plan.spec, plan.license_code, workers=1),
            )

    for plan in plans:
        if plan.spec.slug not in done:
            done[plan.spec.slug] = _consume(ctx, plan, [])
    return [done[spec.slug] for spec in specs]


def _collect(futures: list[Future], tasks: list[tuple[SourceSpec, Path, int]],
             ) -> Iterator[tuple[list[dict[str, Any]], int, str]]:
    for future, (_, container, _) in zip(futures, tasks, strict=True):
        try:
            yield future.result()
        except Exception as exc:
            yield [], 0, f"{container}: enumeration subprocess failed {exc!r}"


def run(ctx: Context, datasets: list[str] | None = None, *,
        force: bool = False, strict: bool = True) -> list[DatasetReport]:
    specs = ctx.registry.select(datasets)
    log.info("S1 discover started: %d datasets", len(specs))
    reports: list[DatasetReport] = _run_all(ctx, specs, force=force)
    ctx.save_containers()

    store = ctx.store(STAGE, WORK_SCHEMA)
    total = sum(r.items for r in reports)
    violations = [r.slug for r in reports if not r.count_ok]
    missing = [r.slug for r in reports if r.containers_missing]
    store.finalize({
        "datasets": len(reports),
        "items": total,
        "count_violations": violations,
        "containers_missing": missing,
        "per_dataset": [r.as_dict() for r in reports],
    })
    log.info("S1 discover completed: %d datasets, %d items", len(reports), total)
    if violations:
        message = f"The following data set enumeration numbers do not match the declaration: {violations}"
        if strict:
            raise ContractError(message)
        log.warning(message)
    return reports
