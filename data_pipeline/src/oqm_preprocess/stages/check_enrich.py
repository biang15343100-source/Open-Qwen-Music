
from __future__ import annotations

from typing import Any

import pyarrow.parquet as pq

from ..adapters import sidecar
from ..registry import DatasetSpec, EnrichSpec
from ..runtime.context import Context
from ..runtime.log import get
from ..schema import WORK_SCHEMA

log = get("check-enrich")

INPUT_STAGE = "s1_discover"

SAMPLE_ROWS = 3000


def run(ctx: Context, datasets: list[str] | None = None) -> dict[str, Any]:
    src = ctx.store(INPUT_STAGE, WORK_SCHEMA)
    reports: list[dict[str, Any]] = []

    for spec in ctx.registry.select(datasets):
        if not spec.enrich or spec.metadata_only:
            continue


        sample = _sample_units(src, spec)
        for enrich_spec in spec.enrich:
            reports.append(_check_one(ctx, spec, enrich_spec, sample))

    bad = [r for r in reports
           if not r.get("pending") and (r.get("join_rate", 1.0) < 0.5 or r.get("status"))]
    pending = [r for r in reports if r.get("pending")]
    return {"checked": len(reports), "suspicious": len(bad),
            "pending": len(pending), "reports": reports}


def _sample_units(src: Any, spec: DatasetSpec) -> list[dict[str, Any]]:
    prefix = f"part-{spec.slug}-"
    rows: list[dict[str, Any]] = []
    for path in src.shard_paths():
        if not path.name.startswith(prefix):
            continue
        table = pq.read_table(path, columns=["member", "local_id", "storage_class"])
        rows.extend(table.to_pylist())
        if len(rows) >= SAMPLE_ROWS:
            break
    return rows[:SAMPLE_ROWS]


def _check_one(ctx: Context, spec: DatasetSpec, enrich_spec: EnrichSpec,
               sample: list[dict[str, Any]]) -> dict[str, Any]:
    label = f"{spec.slug}/{enrich_spec.format}"
    if enrich_spec.format.startswith("inline_"):
        return _check_inline(ctx, spec, enrich_spec, label)

    try:
        index, collisions = sidecar.build_index(enrich_spec)
    except (OSError, ValueError) as exc:
        return {"slug": spec.slug, "spec": enrich_spec.format,
                "status": f"sidecar read failed: {exc}"}

    if not index:
        columns = _peek_columns(enrich_spec)
        return {"slug": spec.slug, "spec": enrich_spec.format, "status": "index is empty",
                "join_rate": 0.0, "available_columns": columns,
                "hint": f"join.key_field={enrich_spec.join.key_field!r} may not be present in the sidecar columns"
                        if enrich_spec.join else "join declaration is missing"}

    if not sample:
        return {"slug": spec.slug, "spec": enrich_spec.format,
                "sidecar_records": len(index), "key_collisions": collisions,
                "target_fields": sorted(enrich_spec.fields),
                "status": "sidecar is valid; join coverage will be checked after discovery", "pending": True,
                "index_key_examples": list(index)[:3]}

    hit = 0
    misses: list[str] = []
    for row in sample:
        keys = sidecar.record_keys(str(row.get("member") or ""),
                                   str(row.get("local_id") or ""),
                                   enrich_spec.join)  # type: ignore[arg-type]
        if any(k in index for k in keys):
            hit += 1
        elif len(misses) < 3:
            misses.append(keys[0] if keys else "")

    rate = hit / len(sample)
    report = {
        "slug": spec.slug, "spec": enrich_spec.format,
        "sidecar_records": len(index), "key_collisions": collisions,
        "sampled": len(sample), "join_rate": round(rate, 4),
        "target_fields": sorted(enrich_spec.fields),
    }

    floor = enrich_spec.expected_join_rate - 0.02 if enrich_spec.expected_join_rate else 0.5
    if rate < floor:

        report["miss_examples"] = misses
        report["index_key_examples"] = list(index)[:3]
        if enrich_spec.expected_join_rate:
            report["expected_join_rate"] = enrich_spec.expected_join_rate
            report["status"] = (f"join rate {rate:.1%} is below the declared minimum "
                                f"{enrich_spec.expected_join_rate:.1%}")
    return report


def _check_inline(ctx: Context, spec: DatasetSpec, enrich_spec: EnrichSpec,
                  label: str) -> dict[str, Any]:
    wanted = set(enrich_spec.fields.values()) | set(enrich_spec.external_ids)
    columns: list[str] = []
    if enrich_spec.format == "inline_parquet":
        source = next((s for s in spec.sources if s.storage_class == "parquet"), None)
        containers = source.containers() if source else []
        if containers:
            columns = list(pq.ParquetFile(containers[0]).schema_arrow.names)
    elif enrich_spec.format == "inline_sqlite":
        source = next((s for s in spec.sources if s.storage_class == "sqlite"), None)
        containers = source.containers() if source else []
        if containers:
            columns = _sqlite_columns(containers[0], source.table if source else None)

    missing = sorted(wanted - set(columns)) if columns else []
    report = {"slug": spec.slug, "spec": enrich_spec.format,
              "target_fields": sorted(enrich_spec.fields)}
    if not columns:
        report["status"] = "container header could not be read"
    elif missing:
        report["status"] = f"columns are missing: {missing}"
        report["available_columns"] = columns[:30]
        report["join_rate"] = 0.0
    return report


def _sqlite_columns(path: Any, table: str | None) -> list[str]:
    import sqlite3

    from ..readers.sqlite_reader import connect_ro

    if not table:
        return []
    try:
        conn = connect_ro(path)
        try:
            cur = conn.execute(f"SELECT * FROM {table} LIMIT 1")
            return [d[0] for d in cur.description]
        finally:
            conn.close()
    except sqlite3.Error:
        return []


def _peek_columns(enrich_spec: EnrichSpec) -> list[str]:
    try:
        for row in sidecar.load_records(enrich_spec):
            return list(row)[:30]
    except (OSError, ValueError):
        return []
    return []
