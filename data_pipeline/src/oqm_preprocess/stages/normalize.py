
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .. import flags, ids
from ..enums import SPLIT, SPLIT_SOURCE, STATUS
from ..runtime.context import Context
from ..runtime.log import Progress, get
from ..schema import WORK_SCHEMA

log = get("normalize")

STAGE = "s6_normalize"
INPUT_STAGE = "s5_dedup"


def assign_split(group_id: int, train_pct: int, valid_pct: int) -> str:
    bucket = ids.split_bucket(int(group_id))
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + valid_pct:
        return "valid"
    return "test"


def _canonical_splits(ctx: Context, train_pct: int, valid_pct: int,
                      ) -> dict[bytes, tuple[int, int]]:
    src = ctx.store(INPUT_STAGE, WORK_SCHEMA)
    columns = ["uid", "group_id", "split", "split_source", "status", "canonical_uid"]
    table = pq.read_table(src.shard_paths(), columns=columns)

    uids = table.column("uid").to_pylist()
    canonicals = table.column("canonical_uid").to_pylist()
    wanted = {
        bytes(c) for u, c in zip(uids, canonicals, strict=True)
        if c is not None and u is not None and bytes(c) != bytes(u)
    }
    if not wanted:
        return {}

    groups = table.column("group_id").to_pylist()
    splits = table.column("split").to_pylist()
    sources = table.column("split_source").to_pylist()
    statuses = table.column("status").to_pylist()

    out: dict[bytes, tuple[int, int]] = {}
    for uid, group, split, source, status in zip(
        uids, groups, splits, sources, statuses, strict=True
    ):
        if uid is None:
            continue
        key = bytes(uid)
        if key not in wanted:
            continue
        out[key] = (int(group or 0),
                    _split_of(int(group or 0), int(split or 0), int(source or 0),
                              int(status or 0), train_pct, valid_pct))
    return out


def _split_of(group_id: int, split: int, split_source: int, status: int,
              train_pct: int, valid_pct: int) -> int:
    if status == STATUS.code("rejected"):
        return SPLIT.code("unknown")
    if split_source == SPLIT_SOURCE.code("official_field"):
        return split
    return SPLIT.code(assign_split(group_id, train_pct, valid_pct))


def run(ctx: Context, *, force: bool = False) -> dict[str, Any]:
    src = ctx.store(INPUT_STAGE, WORK_SCHEMA)
    store = ctx.store(STAGE, WORK_SCHEMA)
    lineage = src.lineage_of()
    if not src.shard_paths():


        if src.success() is None:
            raise RuntimeError(f"{INPUT_STAGE} has no output; run dedup first")
        log.warning("%s is empty: no upstream rows remain, so this stage has nothing to do", INPUT_STAGE)
        store.clear()
        store.finalize({"split_counts": {}, "per_dataset": {}}, lineage=lineage)
        return {"split_counts": {}, "per_dataset": {}}
    if not force and store.is_stage_complete(lineage=lineage):
        log.info("S6 normalize already complete; skipping")
        return store.success() or {}
    store.clear()

    split_cfg = ctx.cfg.split
    train_pct, valid_pct = split_cfg.train_pct, split_cfg.valid_pct
    if not 0 < train_pct < 100 or not 0 <= valid_pct < 100 or train_pct + valid_pct >= 100:
        raise ValueError(f"Invalid split configuration: train={train_pct} valid={valid_pct}")

    canonical_of = _canonical_splits(ctx, train_pct, valid_pct)
    holdout_slugs = {s.slug for s in ctx.registry if s.holdout}
    holdout_ids = {s.dataset_id for s in ctx.registry if s.holdout}

    counts: Counter[str] = Counter()
    by_dataset: dict[int, Counter[str]] = {}
    progress = Progress(log, "S6 normalize", total=src.count_rows())

    for path in src.shard_paths():
        key = path.stem[len("part-"):]
        records = pq.read_table(path).to_pylist()
        for record in records:
            _assign(record, canonical_of, train_pct, valid_pct, holdout_ids)
            counts[SPLIT.name_of(int(record["split"]))] += 1
            by_dataset.setdefault(int(record["dataset_id"]), Counter())[
                SPLIT.name_of(int(record["split"]))
            ] += 1
        store.write_shard(key, records)
        progress.advance(len(records))
    progress.done()

    stats = {
        "split_counts": dict(counts),
        "holdout_datasets": sorted(holdout_slugs),
        "per_dataset": {
            ctx.registry.get(ds).slug: dict(c) for ds, c in sorted(by_dataset.items())
        },
    }
    store.finalize(stats, lineage=lineage)
    log.info("S6 normalize completed: %s", dict(counts))
    return stats


def _assign(record: dict[str, Any], canonical_of: dict[bytes, tuple[int, int]],
            train_pct: int, valid_pct: int, holdout_ids: set[int]) -> None:
    if int(record.get("dataset_id", -1)) in holdout_ids:
        record["flags"] = int(record.get("flags", 0)) | flags.bit("eval_holdout")


    if int(record.get("status") or 0) == STATUS.code("rejected"):
        record["split"] = SPLIT.code("unknown")
        record["split_source"] = 0
        return


    canonical = record.get("canonical_uid")
    if canonical is not None and bytes(canonical) != bytes(record["uid"]):
        inherited = canonical_of.get(bytes(canonical))
        if inherited is not None:
            record["group_id"], record["split"] = inherited
            record["split_source"] = SPLIT_SOURCE.code("group_promoted")
            return


    if int(record.get("split_source") or 0) == SPLIT_SOURCE.code("official_field"):
        return

    group_id = int(record.get("group_id") or 0)
    record["split"] = SPLIT.code(assign_split(group_id, train_pct, valid_pct))
    record["split_source"] = SPLIT_SOURCE.code("stable_group_hash")


def verify_no_leakage(ctx: Context, root: Path | None = None) -> dict[str, Any]:
    columns = ["uid", "group_id", "split", "status", "canonical_uid"]
    if root is not None:
        table = pq.read_table(root / "corpus", columns=columns)
    else:
        table = pq.read_table(ctx.store(STAGE, WORK_SCHEMA).shard_paths(), columns=columns)

    status = np.asarray(table.column("status").to_numpy(zero_copy_only=False))
    splits = np.asarray(table.column("split").to_numpy(zero_copy_only=False))
    groups = np.asarray(table.column("group_id").to_numpy(zero_copy_only=False))

    keep = (status == STATUS.code("accepted")) | (status == STATUS.code("alias"))

    result = {"groups": 0, "leaking_groups": 0, "alias_pairs": 0, "alias_split_mismatch": 0}
    kept_groups, kept_splits = groups[keep], splits[keep]
    if kept_groups.size:
        order = np.lexsort((kept_splits, kept_groups))
        g, s = kept_groups[order], kept_splits[order]
        result["groups"] = int(np.flatnonzero(g[1:] != g[:-1]).size + 1)
        result["leaking_groups"] = int(((g[1:] == g[:-1]) & (s[1:] != s[:-1])).sum())

    split_of = {
        bytes(u): int(sp)
        for u, sp in zip(table.column("uid").to_pylist(), splits, strict=True)
        if u is not None
    }
    mismatch = 0
    pairs = 0
    for uid, canonical, split, keep_row in zip(
        table.column("uid").to_pylist(), table.column("canonical_uid").to_pylist(),
        splits, keep, strict=True,
    ):
        if not keep_row or uid is None or canonical is None:
            continue
        key = bytes(canonical)
        if key == bytes(uid):
            continue
        pairs += 1
        if split_of.get(key, int(split)) != int(split):
            mismatch += 1
    result["alias_pairs"] = pairs
    result["alias_split_mismatch"] = mismatch
    return result
