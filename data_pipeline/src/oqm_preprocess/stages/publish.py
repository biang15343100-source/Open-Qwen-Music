
from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .. import enums, flags
from ..enums import STATUS, codebooks_fingerprint
from ..runtime.context import Context
from ..runtime.log import get
from ..schema import (
    CORPUS_SCHEMA,
    SCHEMA_VERSION,
    WORK_SCHEMA,
    project,
    records_to_table,
)
from ..store import atomic_write_json, atomic_write_table
from .normalize import verify_no_leakage

log = get("publish")

STAGE = "s7_publish"
INPUT_STAGE = "s6_normalize"

READY_FILE = "READY"


KEEP_BACKUPS = 2


_SPLIT_CHECK_MIN_GROUPS = 500

_SIDECAR_SOURCES = {
    "lyrics_timeline": "s3_lyrics_timeline",
    "raw_meta": "s3_raw_meta",
    "dup_edges": "s5_dup_edges",
}


class ValidationError(RuntimeError):
    pass


def run(ctx: Context, *, force: bool = False) -> dict[str, Any]:
    src = ctx.store(INPUT_STAGE, WORK_SCHEMA)
    if not src.shard_paths():
        if src.success() is None:
            raise RuntimeError(f"{INPUT_STAGE} has no output; run normalize first")

        raise RuntimeError(
            f"{INPUT_STAGE} completed with no rows; refusing to publish an empty release"
        )

    root = ctx.cfg.release_dir / f"oqm-corpus-{ctx.cfg.version}"
    lineage = src.lineage_of()
    if root.exists() and (root / READY_FILE).exists() and not force:
        if _ready_lineage(root) == lineage:
            log.info("Release already exists with READY: %s (--force can overwrite it)", root)
            return {"release": str(root), "skipped": True}
        log.warning("Published release was built from an older normalize output; rebuilding")

    staging = root.with_name(root.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "corpus").mkdir(parents=True)
    (staging / "meta").mkdir(parents=True)

    stats = _write_corpus(ctx, src, staging)
    _write_sidecars(ctx, staging)
    _write_meta(ctx, staging, stats)

    checks = _validate(ctx, staging, stats)
    stats["checks"] = checks
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValidationError(f"Publishing verification failed: {failed};product stays in {staging} for troubleshooting")

    _write_version(ctx, staging, stats)
    _write_report(ctx, staging, stats)

    if root.exists():
        backup = root.with_name(f"{root.name}.old-{int(time.time())}")
        os.replace(root, backup)
        log.info("Previous release moved to %s", backup)
        _trim_backups(root)
    os.replace(staging, root)

    (root / READY_FILE).write_text(
        json.dumps({"schema_version": SCHEMA_VERSION,
                    "lineage": lineage,
                    "published_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}),
        encoding="utf-8",
    )
    log.info("Release completed: %s", root)
    stats["release"] = str(root)
    return stats


def _trim_backups(root: Path, keep: int = KEEP_BACKUPS) -> list[Path]:
    backups = sorted(root.parent.glob(f"{root.name}.old-*"))
    stale = backups[:-keep] if keep else backups
    for path in stale:
        shutil.rmtree(path, ignore_errors=True)
        log.info("Removed old backup %s", path.name)
    return stale


def _ready_lineage(root: Path) -> str | None:
    try:
        return json.loads((root / READY_FILE).read_text(encoding="utf-8")).get("lineage")
    except (OSError, ValueError):
        return None


def _write_corpus(ctx: Context, src: Any, staging: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    per_dataset: dict[int, Counter[str]] = {}
    rows_written = 0
    duration_by_split: Counter[str] = Counter()

    for path in src.shard_paths():
        table = pq.read_table(path)
        records = table.to_pylist()
        by_ds: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            by_ds.setdefault(int(record["dataset_id"]), []).append(record)
            status = STATUS.name_of(int(record["status"]))
            counts[status] += 1
            per_dataset.setdefault(int(record["dataset_id"]), Counter())[status] += 1
            if status == "accepted" and record.get("duration_sec"):
                duration_by_split[enums.SPLIT.name_of(int(record["split"] or 0))] += float(
                    record["duration_sec"]
                )
        for dataset_id, chunk in sorted(by_ds.items()):
            slug = ctx.registry.get(dataset_id).slug
            out = staging / "corpus" / f"{slug}--{path.stem}.parquet"
            atomic_write_table(
                project(records_to_table(chunk, WORK_SCHEMA), CORPUS_SCHEMA), out,
                compression=ctx.cfg.runtime.compression,
                level=ctx.cfg.runtime.compression_level,
                row_group_size=ctx.cfg.runtime.row_group_size,
            )
            rows_written += len(chunk)

    return {
        "rows": rows_written,
        "status_counts": dict(counts),
        "per_dataset": {
            ctx.registry.get(ds).slug: dict(c) for ds, c in sorted(per_dataset.items())
        },
        "accepted_hours": {k: round(v / 3600.0, 2) for k, v in sorted(duration_by_split.items())},
    }


def _write_sidecars(ctx: Context, staging: Path) -> None:
    for name, stage in _SIDECAR_SOURCES.items():
        stage_dir = ctx.cfg.stage_dir(stage)
        shards = sorted(stage_dir.glob("part-*.parquet")) if stage_dir.exists() else []
        if not shards:
            continue
        out_dir = staging / "sidecar" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for shard in shards:
            shutil.copy2(shard, out_dir / shard.name)
        log.info("Side table %s: %d shards", name, len(shards))


def _write_meta(ctx: Context, staging: Path, stats: dict[str, Any]) -> None:
    meta = staging / "meta"
    container_src = Context.container_path(ctx.cfg)
    if container_src.exists():
        shutil.copy2(container_src, meta / "container_table.parquet")

    atomic_write_json(
        {"license_codes": ctx.licenses, "by_code": {str(v): k for k, v in ctx.licenses.items()}},
        meta / "license_codes.json",
    )
    atomic_write_json(enums.export_codebooks(), meta / "codebooks.json")
    atomic_write_json(flags.export_codebook(), meta / "flags.json")
    atomic_write_json(
        {"datasets": ctx.registry.as_rows(), "count": len(ctx.registry)},
        meta / "dataset_registry.json",
    )
    atomic_write_json(ctx.cfg.raw_dict(), meta / "pipeline_config.json")
    atomic_write_json(stats, meta / "stage_stats.json")


def _validate(ctx: Context, staging: Path, stats: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    shards = sorted((staging / "corpus").glob("*.parquet"))
    checks["corpus_not_empty"] = bool(shards)
    if not shards:
        return checks

    table = pq.read_table(shards, columns=["uid", "status", "canonical_uid", "duration_sec",
                                           "split", "group_id", "dataset_id"])
    n = table.num_rows
    checks["row_count_matches"] = n == stats["rows"]


    uids = np.array(
        [int.from_bytes(bytes(v), "big") for v in table.column("uid").to_pylist()],
        dtype=np.uint64,
    )
    checks["uid_unique"] = len(np.unique(uids)) == n

    schema_ok = True
    for shard in shards[:5]:
        got = pq.ParquetFile(shard).schema_arrow
        if got.names != CORPUS_SCHEMA.names:
            schema_ok = False
            log.error("Sharding schema does not match: %s", shard)
    checks["schema_stable"] = schema_ok

    status = np.asarray(table.column("status").to_numpy(zero_copy_only=False))
    accepted = status == STATUS.code("accepted")
    checks["has_accepted_rows"] = bool(accepted.any())


    durations = table.column("duration_sec").to_numpy(zero_copy_only=False)
    checks["accepted_have_duration"] = bool(
        np.all(np.isfinite(durations[accepted]) & (durations[accepted] > 0))
    )


    alias_mask = status == STATUS.code("alias")
    if alias_mask.any():
        canon = [
            int.from_bytes(bytes(v), "big") if v else -1
            for v in table.column("canonical_uid").to_pylist()
        ]
        known = set(uids.tolist())
        checks["alias_targets_exist"] = all(
            canon[i] in known for i in np.flatnonzero(alias_mask)
        )
    else:
        checks["alias_targets_exist"] = True


    leakage = verify_no_leakage(ctx, root=staging)
    checks["no_split_leakage"] = leakage["leaking_groups"] == 0
    checks["alias_split_matches_canonical"] = leakage["alias_split_mismatch"] == 0
    stats["leakage"] = leakage


    zeroed = [
        slug for slug, counts in stats.get("per_dataset", {}).items()
        if sum(counts.values()) > 0 and counts.get("accepted", 0) == 0
    ]
    stats["zeroed_datasets"] = zeroed
    for slug in zeroed:
        log.warning("Dataset %s has no accepted rows; confirm that this is expected", slug)

    splits = np.asarray(table.column("split").to_numpy(zero_copy_only=False))[accepted]
    present = {enums.SPLIT.name_of(int(s)) for s in np.unique(splits)} if splits.size else set()
    complete = {"train", "valid", "test"}.issubset(present)


    leakage_groups = leakage["groups"]
    if leakage_groups >= _SPLIT_CHECK_MIN_GROUPS:
        checks["all_splits_present"] = complete
    elif not complete:
        log.warning("Missing split; found only %s (%d groups, too few for strict verification)",
                    sorted(present), leakage_groups)

    for name, ok in checks.items():
        log.info("Verification %-24s %s", name, "by" if ok else "failed")
    return checks


def _write_version(ctx: Context, staging: Path, stats: dict[str, Any]) -> None:
    atomic_write_json(
        {
            "schema_version": SCHEMA_VERSION,
            "release_version": ctx.cfg.version,
            "config_fingerprint": ctx.cfg.fingerprint(),
            "codebooks_fingerprint": codebooks_fingerprint(),
            "flags_fingerprint": flags.fingerprint(),
            "datasets": len(ctx.registry),
            "rows": stats["rows"],
            "status_counts": stats["status_counts"],
            "accepted_hours": stats.get("accepted_hours", {}),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        staging / "VERSION.json",
    )


def _write_report(ctx: Context, staging: Path, stats: dict[str, Any]) -> None:
    lines: list[str] = [
        f"# oqm-corpus-{ctx.cfg.version} build report",
        "",
        f"- schema: `{SCHEMA_VERSION}`",
        f"- configuration fingerprint: `{ctx.cfg.fingerprint()}`",
        f"- total rows: {stats['rows']:,}",
        f"- status counts: {stats['status_counts']}",
        f"- accepted duration (hours): {stats.get('accepted_hours', {})}",
        "",
        "## by dataset",
        "",
        "| dataset | accepted | rejected | alias | other |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for slug, counts in stats.get("per_dataset", {}).items():
        other = sum(v for k, v in counts.items() if k not in ("accepted", "rejected", "alias"))
        lines.append(
            f"| {slug} | {counts.get('accepted', 0):,} | {counts.get('rejected', 0):,} "
            f"| {counts.get('alias', 0):,} | {other:,} |"
        )

    lines += ["", "## Validation", ""]
    for name, ok in stats.get("checks", {}).items():
        lines.append(f"- {'PASS' if ok else 'FAIL'} `{name}`")
    leakage = stats.get("leakage")
    if leakage:
        lines.append(f"- groups {leakage['groups']:,}; cross-split leakage {leakage['leaking_groups']}")
    zeroed = stats.get("zeroed_datasets") or []
    if zeroed:
        lines += ["", "## datasets with no accepted rows", "",
                  "These datasets contain discovered items but no accepted rows; verify that this is expected:", ""]
        lines += [f"- `{slug}`" for slug in zeroed]

    (staging / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_release(root: Path, columns: list[str] | None = None) -> pa.Table:
    root = Path(root)
    if not (root / READY_FILE).exists():
        raise RuntimeError(f"{root} has no READY marker and may be incomplete")
    return pq.read_table(root / "corpus", columns=columns)
