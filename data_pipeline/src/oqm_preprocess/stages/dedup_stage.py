
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .. import flags
from ..dedup import UnionFind, compute_scores, pick_canonical
from ..enums import DUP_METHOD, DUP_STATUS, STATUS
from ..runtime.context import Context
from ..runtime.log import Progress, get
from ..schema import DUP_EDGES_SCHEMA, WORK_SCHEMA

log = get("dedup")

STAGE = "s5_dedup"
INPUT_STAGE = "s4_filter"

_LOAD_COLUMNS = [
    "uid", "dataset_id", "status", "flags", "uri_hash", "content_hash_l1",
    "fingerprint_l2", "duration_bucket", "duration_sec", "sample_rate_hz",
    "codec", "is_synthetic", "external_ids_json",
]


def _as_uint64(column: pa.ChunkedArray | pa.Array, width: int) -> np.ndarray:
    values = column.to_pylist()
    out = np.zeros(len(values), dtype=np.uint64)
    for i, value in enumerate(values):
        if value:


            out[i] = (int.from_bytes(bytes(value)[:width], "big", signed=False) + 1) & 0xFFFFFFFFFFFFFFFF
    return out


def _load(ctx: Context) -> dict[str, Any]:
    src = ctx.store(INPUT_STAGE, WORK_SCHEMA)
    paths = src.shard_paths()
    if not paths:
        raise RuntimeError(f"{INPUT_STAGE} has no output; run the filter stage first")

    table = pq.read_table(paths, columns=_LOAD_COLUMNS)


    allowed = ctx.registry.enabled_ids
    seen = set(table.column("dataset_id").to_pylist())
    keep_mask: np.ndarray | None = None
    if not seen <= allowed:
        dropped = sorted(seen - allowed)
        mask = pc.is_in(table.column("dataset_id"), value_set=pa.array(sorted(allowed), pa.int32()))
        keep_mask = np.asarray(mask.to_numpy(zero_copy_only=False), dtype=bool)
        before = table.num_rows
        table = table.filter(mask)
        log.warning(
            "S5 dedup ignored %d rows from disabled datasets %s",
            before - table.num_rows,
            dropped,
        )


    uids = np.array(
        [int.from_bytes(bytes(v), "big") for v in table.column("uid").to_pylist()],
        dtype=np.uint64,
    )

    table, uids, keep_mask, duplicate_rows = _collapse_duplicate_uid(
        table, uids, keep_mask,
        _as_uint64(table.column("content_hash_l1"), 8),
    )

    n = table.num_rows
    log.info("S5 dedup loaded %d rows", n)
    dataset_id = np.asarray(table.column("dataset_id").to_numpy(zero_copy_only=False), dtype=np.int32)
    status = np.asarray(table.column("status").to_numpy(zero_copy_only=False), dtype=np.int8)
    flags_col = np.asarray(table.column("flags").to_numpy(zero_copy_only=False), dtype=np.uint64)

    return {
        "n": n,
        "uid": uids,
        "dataset_id": dataset_id,
        "status": status,
        "flags": flags_col,
        "uri": _as_uint64(table.column("uri_hash"), 8),


        "l1": _as_uint64(table.column("content_hash_l1"), 8),
        "l2": _as_uint64(table.column("fingerprint_l2"), 8),
        "bucket": np.nan_to_num(
            table.column("duration_bucket").to_numpy(zero_copy_only=False).astype(np.float64), nan=-1.0
        ),
        "duration": np.nan_to_num(
            table.column("duration_sec").to_numpy(zero_copy_only=False).astype(np.float64), nan=0.0
        ),
        "sample_rate": np.nan_to_num(
            table.column("sample_rate_hz").to_numpy(zero_copy_only=False).astype(np.float64), nan=0.0
        ),
        "codec": np.nan_to_num(
            table.column("codec").to_numpy(zero_copy_only=False).astype(np.float64), nan=0.0
        ).astype(np.int32),
        "is_synthetic": np.asarray(
            [bool(v) for v in table.column("is_synthetic").to_pylist()], dtype=bool
        ),
        "external": table.column("external_ids_json").to_pylist(),
        "paths": paths,

        "keep_mask": keep_mask,
        "duplicate_uid_rows": duplicate_rows,
    }


def _collapse_duplicate_uid(
    table: pa.Table, uids: np.ndarray, keep_mask: np.ndarray | None,
    content: np.ndarray,
) -> tuple[pa.Table, np.ndarray, np.ndarray | None, int]:
    unique, counts = np.unique(uids, return_counts=True)
    if counts.max(initial=0) <= 1:
        return table, uids, keep_mask, 0

    keep = np.ones(uids.size, dtype=bool)
    conflicts = 0
    for value in unique[counts > 1]:
        rows = np.flatnonzero(uids == value)


        hashed = content[rows][content[rows] != 0]
        if np.unique(hashed).size > 1:
            conflicts += 1
            continue

        best = rows[0] if hashed.size == 0 else rows[np.argmax(content[rows] != 0)]
        keep[rows] = False
        keep[best] = True
    if conflicts:
        raise RuntimeError(
            f"{conflicts} UIDs refer to entries with different content; "
            "make local_id unique in the discovery stage"
        )
    dropped = int(uids.size - keep.sum())

    log.warning("S5 dedup discarded %d duplicate UID rows", dropped)
    if keep_mask is None:
        keep_mask = keep
    else:

        keep_mask = keep_mask.copy()
        keep_mask[np.flatnonzero(keep_mask)[~keep]] = False
    return table.filter(pa.array(keep)), uids[keep], keep_mask, dropped


def _merge_by_key(uf: UnionFind, keys: np.ndarray, eligible: np.ndarray, label: str) -> int:
    valid = np.flatnonzero(eligible & (keys != 0))
    if valid.size < 2:
        return 0
    order = valid[np.argsort(keys[valid], kind="stable")]
    merged = uf.union_sorted_groups(order, keys[order])
    log.info("  %s: %d candidates, %d merges", label, valid.size, merged)
    return merged


def _merge_l2(uf: UnionFind, data: dict[str, Any], eligible: np.ndarray) -> int:
    keys = data["l2"]
    bucket = data["bucket"]
    valid = np.flatnonzero(eligible & (keys != 0))
    if valid.size < 2:
        return 0

    order = valid[np.lexsort((bucket[valid], keys[valid]))]
    same = (keys[order][1:] == keys[order][:-1]) & (bucket[order][1:] == bucket[order][:-1])
    merged = 0
    for idx in np.flatnonzero(same):
        if uf.union(int(order[idx]), int(order[idx + 1])):
            merged += 1
    log.info("  L2 fingerprint: %d candidates, %d merges", valid.size, merged)
    return merged


def _merge_external_ids(uf: UnionFind, data: dict[str, Any], eligible: np.ndarray) -> int:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in np.flatnonzero(eligible):
        payload = data["external"][index]
        if not payload:
            continue
        try:
            pairs = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            continue
        for key, value in pairs.items():
            if value:
                groups[(key, str(value))].append(int(index))

    merged = 0
    for members in groups.values():
        for other in members[1:]:
            if uf.union(members[0], other):
                merged += 1
    log.info("  L1.5 external ID: %d ID groups, %d merges", len(groups), merged)
    return merged


def run(ctx: Context, *, force: bool = False) -> dict[str, Any]:
    store = ctx.store(STAGE, WORK_SCHEMA)
    edges_store = ctx.store("s5_dup_edges", DUP_EDGES_SCHEMA)
    cfg = ctx.cfg.dedup

    lineage = ctx.store(INPUT_STAGE, WORK_SCHEMA).lineage_of()
    if not force and store.is_stage_complete(lineage=lineage):
        log.info("S5 dedup already complete; skipping")
        return store.success() or {}
    store.clear()
    edges_store.clear()

    data = _load(ctx)
    n = data["n"]
    if n == 0:
        store.finalize({"rows": 0}, lineage=lineage)
        return {"rows": 0}


    eligible = data["status"] == STATUS.code("accepted")
    log.info("Rows participating in deduplication: %d / %d", int(eligible.sum()), n)

    uf = UnionFind(n)
    merged = {}
    if cfg.enable_l0_uri:
        merged["l0_uri"] = _merge_by_key(uf, data["uri"], eligible, "L0 identical location")
    if cfg.enable_l1_bytes:
        merged["l1_bytes"] = _merge_by_key(uf, data["l1"], eligible, "L1 byte hash")
    if cfg.enable_l15_external_id:
        merged["l15_external_id"] = _merge_external_ids(uf, data, eligible)
    if cfg.enable_l2_fingerprint:
        merged["l2_fingerprint"] = _merge_l2(uf, data, eligible)

    roots = uf.roots()
    priority = _priority_array(ctx, data["dataset_id"])
    has_lyrics, has_meta = _text_presence(ctx, n, data["keep_mask"], data["flags"])
    scores = compute_scores(
        codec=data["codec"], sample_rate=data["sample_rate"], duration=data["duration"],
        has_lyrics=has_lyrics, has_meta=has_meta, is_synthetic=data["is_synthetic"],
        priority=priority, cfg=cfg,
    )

    roots = np.where(eligible, roots, np.arange(n, dtype=np.int32))
    representative = pick_canonical(roots, scores, data["uid"])

    is_alias = (representative != np.arange(n)) & eligible
    method = _dominant_method(data, representative, is_alias)
    log.info("Deduplication result: %d alias rows, %d duplicate groups", int(is_alias.sum()),
             int(len({int(r) for r in representative[is_alias]})))

    stats = _write_output(ctx, store, edges_store, data, representative, is_alias, method, force)
    stats["merged"] = merged
    stats["alias_rows"] = int(is_alias.sum())
    stats["eligible_rows"] = int(eligible.sum())
    store.finalize(stats, lineage=lineage)
    edges_store.finalize({"rows": edges_store.count_rows()}, lineage=lineage)
    return stats


def _priority_array(ctx: Context, dataset_id: np.ndarray) -> np.ndarray:
    lookup = {spec.dataset_id: float(spec.priority) for spec in ctx.registry}
    out = np.zeros(dataset_id.size, dtype=np.float32)
    for ds, value in lookup.items():
        out[dataset_id == ds] = value
    return out


def _text_presence(ctx: Context, n: int, keep_mask: np.ndarray | None,
                   raw_flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = ctx.store(INPUT_STAGE, WORK_SCHEMA)
    table = pq.read_table(src.shard_paths(), columns=["lyrics_text"])
    lyrics = np.array([bool(v) for v in table.column("lyrics_text").to_pylist()], dtype=bool)
    if keep_mask is not None:
        lyrics = lyrics[keep_mask]
    if lyrics.size != n:
        raise RuntimeError(f"Column lengths are inconsistent: {lyrics.size} vs {n}")


    meta = (raw_flags & flags.bit("has_raw_meta")) != 0
    return lyrics, meta


def _dominant_method(data: dict[str, Any], representative: np.ndarray,
                     is_alias: np.ndarray) -> np.ndarray:
    n = representative.size
    out = np.zeros(n, dtype=np.int8)
    external = _external_pairs(data)
    for index in np.flatnonzero(is_alias):
        rep = int(representative[index])
        if data["uri"][index] and data["uri"][index] == data["uri"][rep]:
            out[index] = DUP_METHOD.code("uri")
        elif data["l1"][index] and data["l1"][index] == data["l1"][rep]:
            out[index] = DUP_METHOD.code("byte_hash")
        elif data["l2"][index] and data["l2"][index] == data["l2"][rep]:
            out[index] = DUP_METHOD.code("fingerprint")
        elif external[index] & external[rep]:
            out[index] = DUP_METHOD.code("external_id")
        else:
            out[index] = DUP_METHOD.code("transitive")
    return out


def _external_pairs(data: dict[str, Any]) -> list[set[tuple[str, str]]]:
    out: list[set[tuple[str, str]]] = []
    for payload in data["external"]:
        pairs: set[tuple[str, str]] = set()
        if payload:
            try:
                for key, value in json.loads(payload).items():
                    if value:
                        pairs.add((str(key), str(value)))
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
        out.append(pairs)
    return out


def _write_output(ctx: Context, store: Any, edges_store: Any, data: dict[str, Any],
                  representative: np.ndarray, is_alias: np.ndarray, method: np.ndarray,
                  force: bool) -> dict[str, Any]:
    uid_array = data["uid"]
    canonical_uid = uid_array[representative]
    alias_status = STATUS.code("alias")

    def as_bytes(value: np.uint64) -> bytes:
        return int(value).to_bytes(8, "big")

    per_dataset_alias: Counter[int] = Counter()
    edges: list[dict[str, Any]] = []

    keep_mask = data.get("keep_mask")
    raw_index = 0
    index = 0
    dropped = 0
    edge_part = 0
    shard_size = ctx.cfg.runtime.shard_size
    progress = Progress(log, "S5 writeback", total=data["n"])
    for path in data["paths"]:
        key = path.stem[len("part-"):]
        records = pq.read_table(path).to_pylist()
        kept: list[dict[str, Any]] = []
        for local, record in enumerate(records):
            if keep_mask is not None and not keep_mask[raw_index]:
                raw_index += 1
                dropped += 1
                continue
            raw_index += 1
            kept.append(record)
            if int.from_bytes(bytes(record["uid"]), "big") != int(uid_array[index]):
                raise RuntimeError(
                    f"Fragment row order is inconsistent with loading @ {path.name}:{local},The deduplication conclusion cannot be written back safely"
                )
            if is_alias[index]:
                record["dup_status"] = DUP_STATUS.code("alias")
                record["canonical_uid"] = as_bytes(canonical_uid[index])
                record["dup_method"] = int(method[index])
                record["status"] = alias_status
                per_dataset_alias[int(record["dataset_id"])] += 1
                edges.append({
                    "uid": record["uid"],
                    "canonical_uid": as_bytes(canonical_uid[index]),
                    "method": int(method[index]),
                    "similarity": 1.0,
                })
            else:
                record["dup_status"] = DUP_STATUS.code("canonical")
                record["canonical_uid"] = record["uid"]
                record["dup_method"] = 0
            index += 1
        store.write_shard(key, kept)
        progress.advance(len(kept))

        if len(edges) >= shard_size:
            edges_store.write_shard(f"edges-{edge_part:05d}", edges)
            edges = []
            edge_part += 1
    progress.done()

    if edges:
        edges_store.write_shard(f"edges-{edge_part:05d}", edges)
    if index != data["n"]:
        raise RuntimeError(f"Written row count {index} does not match loaded count {data['n']}; deduplication results may be misassigned")
    duplicates = int(data.get("duplicate_uid_rows") or 0)
    return {
        "rows": index,
        "dropped_disabled_rows": dropped - duplicates,
        "dropped_duplicate_uid_rows": duplicates,
        "alias_per_dataset": {
            ctx.registry.get(ds).slug: count for ds, count in sorted(per_dataset_alias.items())
        },
    }
