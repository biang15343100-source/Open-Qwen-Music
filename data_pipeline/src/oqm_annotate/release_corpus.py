
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .corpus import (
    GRANULARITY_NOTE,
    GRANULARITY_PHRASE,
    GRANULARITY_WHOLE,
    ArchiveError,
)


CORPUS_SCHEMA_VERSION = "oqm.corpus.v1"


LANGUAGE_FROM_RELEASE = "@release"


GRANULARITY_FROM_RELEASE: dict[str, str] = {
    "whole_song": GRANULARITY_WHOLE,
    "phrase": GRANULARITY_PHRASE,
    "note": GRANULARITY_NOTE,
}


RELEASE_GRANULARITIES = ("whole_song", "clip", "phrase", "note")


#:


_NON_LANGUAGE_CODES = frozenset(
    {"unknown", "other", "instrumental", "language_neutral", "zh_en"}
)


class ReleaseError(RuntimeError):
    pass


def _preprocess_src_candidates(hint: str | Path | None) -> list[Path]:
    if hint:
        return [Path(str(hint))]
    here = Path(__file__).resolve()


    return [
        here.parents[1],
        here.parents[3] / "data_preprocess" / "src",
    ]


def import_preprocess(hint: str | Path | None = None) -> Any:

    tried: list[Path] = []
    for candidate in _preprocess_src_candidates(hint):
        tried.append(candidate)
        if not (candidate / "oqm_preprocess" / "__init__.py").exists():
            continue
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        try:
            from oqm_preprocess import flags as _flags  # noqa: PLC0415
            from oqm_preprocess.enums import (  # noqa: PLC0415
                CONTENT_TYPE,
                DUP_STATUS,
                DURATION_SOURCE,
                GRANULARITY,
                LANGUAGE,
                LICENSE_FAMILY,
                SPLIT,
                STATUS,
                STORAGE_CLASS,
            )
            from oqm_preprocess.locator import to_hints, to_ref  # noqa: PLC0415
            from oqm_preprocess.readers import read_bytes  # noqa: PLC0415
            from oqm_preprocess.readers import tar_reader  # noqa: PLC0415
            from oqm_preprocess.stages.publish import read_release  # noqa: PLC0415
            from oqm_preprocess.store import ContainerTable  # noqa: PLC0415
            from oqm_preprocess.registry import Registry  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover -
            raise ReleaseError(
                f"Found the preprocessing package at {candidate}, but importing it failed: {exc}\n"
                "Install pyarrow in the annotation environment because the upstream "
                "release table uses Parquet."
            ) from exc
        return _Preprocess(
            flags=_flags,
            CONTENT_TYPE=CONTENT_TYPE,
            DUP_STATUS=DUP_STATUS,
            DURATION_SOURCE=DURATION_SOURCE,
            GRANULARITY=GRANULARITY,
            LANGUAGE=LANGUAGE,
            LICENSE_FAMILY=LICENSE_FAMILY,
            SPLIT=SPLIT,
            STATUS=STATUS,
            STORAGE_CLASS=STORAGE_CLASS,
            to_hints=to_hints,
            to_ref=to_ref,
            read_bytes=read_bytes,
            tar_reader=tar_reader,
            read_release=read_release,
            ContainerTable=ContainerTable,
            Registry=Registry,
        )
    raise ReleaseError(
        f"The source code directory of the preprocessing pipeline cannot be found,tried:{[str(p) for p in tried]}\n"
        f"  Write `preprocess_src: /path/to/data_preprocess/src` specifies.\n"
        f"  This layer deliberately reuses the upstream `locator` / `readers` without copying the pronunciation yourself:"
        f"The performance of copying errors is to get the bytes of other members,Decoding is still successful,The duration is still reasonable."
    )


@dataclass(frozen=True, slots=True)
class _Preprocess:

    flags: Any
    CONTENT_TYPE: Any
    DUP_STATUS: Any
    DURATION_SOURCE: Any
    GRANULARITY: Any
    LANGUAGE: Any
    LICENSE_FAMILY: Any
    SPLIT: Any
    STATUS: Any
    STORAGE_CLASS: Any
    to_hints: Any
    to_ref: Any
    read_bytes: Any
    tar_reader: Any
    read_release: Any
    ContainerTable: Any
    Registry: Any


_NEEDED_COLUMNS = (

    "uid", "dataset_id", "local_id",

    "storage_class", "container_id", "member",
    "member_offset", "member_header_offset", "member_size",
    "member_compressed_size", "member_compress", "sequential_only",
    "parquet_row_group", "parquet_row_index", "locator_params",
    "clip_start_sec", "clip_end_sec",

    "duration_sec", "duration_source", "sample_rate_hz", "channels",
    "granularity", "content_type", "language", "status", "split", "flags",
    "dup_status", "canonical_uid",


    "license_family", "commercial_ok",
)


@dataclass
class Release:

    root: Path
    table: Any
    containers: Any
    pre: _Preprocess
    version: dict[str, Any]


    upstream_duration: dict[str, list[float]] = field(default_factory=dict)

    @property
    def lineage(self) -> str:

        return str(self.version.get("lineage") or "")


    dataset_slugs: dict[int, str] = field(default_factory=dict)


    passwords: dict[int, bytes] = field(default_factory=dict)


def open_release(
    root: str | Path, *, preprocess_src: str | Path | None = None
) -> Release:

    pre = import_preprocess(preprocess_src)
    path = Path(str(root))
    if not path.is_dir():
        raise ReleaseError(f"Release directory does not exist: {path}")
    try:
        table = pre.read_release(path)
    except Exception as exc:  # noqa: BLE001 - ,
        raise ReleaseError(
            f"Failed to read release: {path}\n"
            f"  {type(exc).__name__}: {exc}\n"
            "  A missing READY marker means publishing validation has not completed. "
            "Do not consume an incomplete release."
        ) from exc

    schema_version = _schema_version(table)
    if schema_version != CORPUS_SCHEMA_VERSION:
        raise ReleaseError(
            f"Release {path} uses schema_version={schema_version!r}; "
            f"this adapter requires {CORPUS_SCHEMA_VERSION!r}. "
            "Review schema.py before adding support for another version."
        )

    missing = [c for c in _NEEDED_COLUMNS if c not in table.column_names]
    if missing:
        raise ReleaseError(
            f"Release {path} is missing required columns: {missing}. "
            "Review oqm_preprocess/schema.py for upstream schema changes."
        )

    version = _read_json(path / "VERSION.json")
    version.update(_read_json(path / "READY"))
    containers = pre.ContainerTable.read(path / "meta" / "container_table.parquet")
    return Release(
        root=path,
        table=table,
        containers=containers,
        pre=pre,
        version=version,
        upstream_duration=_upstream_duration(path),
        dataset_slugs=_dataset_slugs(path),
        passwords=_release_passwords(path, pre),
    )


def _schema_version(table: Any) -> str:
    metadata = table.schema.metadata or {}
    raw = metadata.get(b"schema_version") or metadata.get("schema_version")
    if isinstance(raw, bytes):
        return raw.decode()
    return str(raw or "")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _upstream_duration(root: Path) -> dict[str, list[float]]:
    config = _read_json(root / "meta" / "pipeline_config.json")
    duration = ((config.get("filters") or {}).get("duration")) or {}
    out: dict[str, list[float]] = {}
    for key, value in duration.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            out[str(key)] = [float(value[0]), float(value[1])]
    return out


def _dataset_slugs(root: Path) -> dict[int, str]:

    payload = _read_json(root / "meta" / "dataset_registry.json")
    entries: list[Any]
    if isinstance(payload.get("datasets"), list):
        entries = payload["datasets"]
    elif isinstance(payload.get("datasets"), dict):
        entries = list(payload["datasets"].values())
    else:
        entries = [v for v in payload.values() if isinstance(v, dict)]
    out: dict[int, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        did, slug = entry.get("dataset_id"), entry.get("slug")
        if did is None or not slug:
            continue
        try:
            out[int(did)] = str(slug)
        except (TypeError, ValueError):
            continue
    return out


def _release_passwords(root: Path, pre: _Preprocess) -> dict[int, bytes]:
    config = _read_json(root / "meta" / "pipeline_config.json")
    raw = config.get("registry_dir")
    if not raw:
        return {}
    registry_dir = Path(str(raw))
    if not registry_dir.is_dir():
        raise ReleaseError(
            f"release recorded registry_dir does not exist:{registry_dir}."
            "Encrypted container(as music4all)therefore cannot be read;Don\'t silently degrade into no password."
        )
    registry = pre.Registry.load(registry_dir)
    out: dict[int, bytes] = {}
    for spec in registry:
        secrets = {source.secret() for source in spec.sources if source.secret()}
        if len(secrets) > 1:
            raise ReleaseError(f"data set {spec.slug} declares multiple different zip Password")
        if secrets:
            out[int(spec.dataset_id)] = next(iter(secrets))
    return out


@dataclass(frozen=True, slots=True)
class Selection:


    release_granularity: str = "whole_song"

    datasets: tuple[str, ...] = ()


    statuses: tuple[str, ...] = ("accepted",)

    splits: tuple[str, ...] = ()


    exclude_flags: tuple[str, ...] = ("eval_holdout",)


    exclude_warn: bool = False

    #:


    include_uids: frozenset[bytes] | None = None
    sample_count: int | None = None
    sample_seed: str = "oqm-release-v1"


def select_rows(release: Release, selection: Selection) -> tuple[list[dict[str, Any]], dict[str, Any]]:

    pre = release.pre
    if selection.release_granularity not in RELEASE_GRANULARITIES:
        raise ReleaseError(
            f"release_granularity={selection.release_granularity!r} is not the upstream"
            f"The value in the code table,optional {list(RELEASE_GRANULARITIES)}."
        )

    want_gran = pre.GRANULARITY.code(selection.release_granularity, strict=True)
    want_status = {pre.STATUS.code(s, strict=True) for s in selection.statuses}
    want_split = {pre.SPLIT.code(s, strict=True) for s in selection.splits}
    exclude_mask = pre.flags.mask_of(*selection.exclude_flags)
    if selection.exclude_warn:
        exclude_mask |= pre.flags.WARN_MASK
    keep_datasets = set(selection.datasets)

    audit = {
        "total_rows": int(release.table.num_rows),
        "dropped_uid_filter": 0,
        "dropped_granularity": 0,
        "dropped_status": 0,
        "dropped_split": 0,
        "dropped_flags": 0,
        "dropped_dataset": 0,
    }
    table = release.table
    if selection.include_uids is not None:


        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.compute as pc  # noqa: PLC0415

        uid_type = table.schema.field("uid").type
        wanted = pa.array(sorted(selection.include_uids), type=uid_type)
        before = int(table.num_rows)
        table = table.filter(pc.is_in(table.column("uid"), value_set=wanted))
        audit["dropped_uid_filter"] = before - int(table.num_rows)
        audit["uid_filter_requested"] = len(selection.include_uids)
        audit["uid_filter_found"] = int(table.num_rows)

    slugs = release.dataset_slugs
    rows: list[dict[str, Any]] = []
    for row in table.select(list(_NEEDED_COLUMNS)).to_pylist():
        if row.get("granularity") != want_gran:
            audit["dropped_granularity"] += 1
            continue
        if row.get("status") not in want_status:
            audit["dropped_status"] += 1
            continue
        if want_split and row.get("split") not in want_split:
            audit["dropped_split"] += 1
            continue
        if exclude_mask and (int(row.get("flags") or 0) & exclude_mask):
            audit["dropped_flags"] += 1
            continue
        slug = slugs.get(int(row.get("dataset_id") or -1), "")
        if keep_datasets and slug not in keep_datasets:
            audit["dropped_dataset"] += 1
            continue
        row["_slug"] = slug
        rows.append(row)

    audit["selected_before_sampling"] = len(rows)
    if selection.sample_count is not None:
        rows = sample_rows(rows, selection.sample_count, seed=selection.sample_seed)
        audit["sample_requested"] = int(selection.sample_count)
        audit["sample_seed"] = selection.sample_seed
    audit["selected"] = len(rows)
    return rows, audit


def sample_rows(
    rows: list[dict[str, Any]], count: int, *, seed: str = "oqm-release-v1"
) -> list[dict[str, Any]]:

    if count <= 0:
        return []
    ordered = sorted(rows, key=lambda r: _selection_key(seed, r["uid"]))
    return ordered[:count]


def _selection_key(seed: str, uid: bytes) -> bytes:
    return hashlib.sha256(seed.encode() + b"\x00" + bytes(uid)).digest()


def duration_gate_preflight(
    rows: list[dict[str, Any]],
    gate: tuple[float, float],
    release_granularity: str,
    upstream_duration: dict[str, list[float]],
) -> dict[str, Any]:

    low, high = gate
    upstream = upstream_duration.get(release_granularity)
    durations = [r.get("duration_sec") for r in rows]
    known = [float(d) for d in durations if d is not None]

    too_short = sum(1 for d in known if d < low)
    too_long = sum(1 for d in known if d > high)
    result: dict[str, Any] = {
        "release_granularity": release_granularity,
        "gate": [low, high],
        "upstream_gate": upstream,
        "candidates": len(rows),
        "duration_known": len(known),
        "duration_unknown": len(rows) - len(known),
        "predicted_too_short": too_short,
        "predicted_too_long": too_long,
        "predicted_kept": len(known) - too_short - too_long,
    }
    if known:
        result["predicted_drop_rate"] = round((too_short + too_long) / len(known), 4)

    if upstream is None:
        result["basis"] = "no upstream threshold"
        result["stricter_than_upstream"] = None
        result["detail"] = (
            "The release does not record an upstream duration filter, so strictness "
            "cannot be compared. Review the predicted record counts."
        )
        return result

    up_low, up_high = upstream
    stricter = low > up_low or high < up_high
    result["basis"] = "upstream filters.duration"
    result["stricter_than_upstream"] = stricter


    gap = sum(1 for d in known if (d < low and d >= up_low) or (d > high and d <= up_high))
    result["dropped_from_upstream_accepted"] = gap
    if not stricter:
        result["detail"] = (
            f"The current gate [{low}, {high}] is not stricter than upstream "
            f"[{up_low}, {up_high}] and does not discard upstream-accepted records."
        )
    else:
        if known:
            result["detail"] = (
                f"The current gate [{low}, {high}] is stricter than upstream "
                f"[{up_low}, {up_high}] and discards {gap} upstream-accepted records"
                f" ({gap / len(known):.1%} of {len(known)})."
            )
        else:
            result["detail"] = (
                f"The current gate [{low}, {high}] is stricter than upstream "
                f"[{up_low}, {up_high}], but this node has no matching records."
            )
    return result


def assert_preflight_acknowledged(
    dataset: str, preflight: dict[str, Any], ack: str
) -> None:

    gap = int(preflight.get("dropped_from_upstream_accepted") or 0)
    if gap <= 0:
        return
    if str(ack or "").strip():
        return
    low, high = preflight.get("gate") or [None, None]
    upstream = preflight.get("upstream_gate") or [None, None]
    known = int(preflight.get("duration_known") or 0)
    raise ReleaseError(
        f"Dataset {dataset} uses duration gate [{low}, {high}], which is stricter than "
        f"upstream [{upstream[0]}, {upstream[1]}] and drops {gap} "
        f"({gap / known:.1%} of {known}) upstream-accepted records.\n"
        f"  Predicted: too short {preflight.get('predicted_too_short')} / "
        f"too long {preflight.get('predicted_too_long')} / "
        f"kept {preflight.get('predicted_kept')}.\n"
        "  Align min_duration_sec and max_duration_sec with the upstream gate, or set "
        "duration_gate_ack with the reason for the stricter policy."
    )


_SAFE = re.compile(r"[^0-9A-Za-z._-]+")


_FALLBACK_SUFFIX = ".bin"


def uid_hex(uid: bytes) -> str:

    return bytes(uid).hex()


def cache_name(row: dict[str, Any], suffix: str) -> str:

    stem = _SAFE.sub("_", uid_hex(row["uid"]))
    return f"{stem}{suffix or _FALLBACK_SUFFIX}"


def member_suffix(row: dict[str, Any], storage_class: str) -> str:

    member = str(row.get("member") or "")
    if member:
        suffix = Path(member.rsplit("/", 1)[-1]).suffix.lower()
        if suffix:
            return suffix
    if storage_class == "parquet":


        return ".bin"
    return _FALLBACK_SUFFIX


def group_by_container(rows: list[dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:

    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(int(row["container_id"]), []).append(row)
    for group in buckets.values():
        group.sort(key=lambda r: (int(r.get("member_offset") or -1), str(r.get("member") or "")))
    return sorted(buckets.items())


def materialize_rows(
    release: Release,
    rows: list[dict[str, Any]],
    cache_dir: Path,
    stats: dict[str, Any] | None = None,
    *,
    workers: int = 1,
) -> Iterator[dict[str, Any]]:

    pre = release.pre
    stats = stats if stats is not None else {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    groups = group_by_container(rows)
    if workers <= 1:
        for container_id, group in groups:
            storage_class = pre.STORAGE_CLASS.name_of(
                int(group[0]["storage_class"])
            )
            if storage_class == "targz":
                yield from _materialize_targz(
                    release, container_id, group, cache_dir, stats
                )
                continue
            for row in group:
                payload = _materialize_one(release, row, cache_dir, stats)
                if payload is not None:
                    yield payload
        return


    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    random_rows: list[dict[str, Any]] = []
    for container_id, group in groups:
        storage_class = pre.STORAGE_CLASS.name_of(
            int(group[0]["storage_class"])
        )
        if storage_class == "targz":
            local: dict[str, Any] = {}
            yield from _materialize_targz(
                release, container_id, group, cache_dir, local
            )
            _merge_materialize_stats(stats, local)
            continue
        random_rows.extend(group)

    def one(row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        local: dict[str, Any] = {}
        return _materialize_one(release, row, cache_dir, local), local

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for payload, local in pool.map(one, random_rows):
            _merge_materialize_stats(stats, local)
            if payload is not None:
                yield payload


def _merge_materialize_stats(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("materialized", "materialize_failed", "targz_streams"):
        if source.get(key):
            target[key] = int(target.get(key, 0)) + int(source[key])
    if source.get("materialize_errors"):
        target.setdefault("materialize_errors", []).extend(source["materialize_errors"])


def _target_path(release: Release, row: dict[str, Any], cache_dir: Path) -> Path:
    storage_class = release.pre.STORAGE_CLASS.name_of(int(row["storage_class"]))
    return cache_dir / cache_name(row, member_suffix(row, storage_class))


def _record_failure(stats: dict[str, Any], row: dict[str, Any], exc: BaseException) -> None:
    stats["materialize_failed"] = int(stats.get("materialize_failed", 0)) + 1
    stats.setdefault("materialize_errors", []).append(
        f"{uid_hex(row['uid'])}: {type(exc).__name__}: {exc}"
    )


def _materialize_one(
    release: Release, row: dict[str, Any], cache_dir: Path, stats: dict[str, Any]
) -> dict[str, Any] | None:
    pre = release.pre
    storage_class = pre.STORAGE_CLASS.name_of(int(row["storage_class"]))
    try:
        ref = pre.to_ref(row, release.containers)
    except Exception as exc:  # noqa: BLE001 -
        _record_failure(stats, row, exc)
        return None

    if storage_class == "loose":
        path = Path(ref.container)
        if not path.exists():
            _record_failure(stats, row, ArchiveError(f"audio does not exist：{path}"))
            return None
        return _payload(release, row, path, storage_class, "n/a")

    target = _target_path(release, row, cache_dir)
    declared = int(row.get("member_size") or -1)
    if target.exists() and declared > 0 and target.stat().st_size == declared:


        return _payload(release, row, target, storage_class, "cached")

    try:
        data = pre.read_bytes(
            ref,
            pre.to_hints(row),
            release.passwords.get(int(row.get("dataset_id") or -1)),
        )
    except Exception as exc:  # noqa: BLE001 -  ReadError  OSError
        _record_failure(stats, row, exc)
        return None
    _atomic_write(target, data)
    stats["materialized"] = int(stats.get("materialized", 0)) + 1
    return _payload(release, row, target, storage_class, "extracted")


def _materialize_targz(
    release: Release,
    container_id: int,
    group: list[dict[str, Any]],
    cache_dir: Path,
    stats: dict[str, Any],
) -> Iterator[dict[str, Any]]:

    container = release.containers.path_of(container_id)
    pending: dict[str, list[dict[str, Any]]] = {}
    for row in group:
        target = _target_path(release, row, cache_dir)
        declared = int(row.get("member_size") or -1)
        if target.exists() and declared > 0 and target.stat().st_size == declared:
            yield _payload(release, row, target, "targz", "cached")
            continue

        pending.setdefault(str(row.get("member") or ""), []).append(row)
    if not pending:
        return

    stats["targz_streams"] = int(stats.get("targz_streams", 0)) + 1
    got: set[str] = set()
    try:
        for name, data in release.pre.tar_reader.stream_members(
            container, wanted=list(pending)
        ):
            for row in pending.get(name, ()):
                target = _target_path(release, row, cache_dir)
                _atomic_write(target, data)
                stats["materialized"] = int(stats.get("materialized", 0)) + 1
                got.add(name)
                yield _payload(release, row, target, "targz", "streamed")
    except Exception as exc:  # noqa: BLE001 -
        for rows_for_name in pending.values():
            for row in rows_for_name:
                _record_failure(stats, row, exc)
        return
    for name, rows_for_name in pending.items():
        if name in got:
            continue
        for row in rows_for_name:
            _record_failure(
                stats, row, ArchiveError(f"tar.gz stream does not contain member {name!r}：{container}")
            )


def _atomic_write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(target)


def _payload(
    release: Release,
    row: dict[str, Any],
    path: Path,
    storage_class: str,
    origin: str,
) -> dict[str, Any]:

    pre = release.pre
    language = pre.LANGUAGE.name_of(int(row["language"])) if row.get("language") is not None else "unknown"
    return {
        "path": path,


        "release_sample_id": f"oqm:{uid_hex(row['uid'])}",


        "manifest_duration_sec": row.get("duration_sec"),
        "sample_rate_hint": row.get("sample_rate_hz"),
        "channels_hint": row.get("channels"),
        "crop_sec": None,
        "language_hint": _language_hint(language),
        "extra": {


            "release_uid": uid_hex(row["uid"]),
            "release_root": str(release.root),
            "release_lineage": release.lineage,
            "release_local_id": row.get("local_id"),
            "release_dataset_slug": row.get("_slug") or "",
            "release_granularity": pre.GRANULARITY.name_of(int(row["granularity"])),
            "release_content_type": (
                pre.CONTENT_TYPE.name_of(int(row["content_type"]))
                if row.get("content_type") is not None
                else None
            ),
            "release_language": language,
            "release_status": pre.STATUS.name_of(int(row["status"])),
            "release_split": pre.SPLIT.name_of(int(row["split"]))
            if row.get("split") is not None else None,
            "release_flags": pre.flags.decode(int(row.get("flags") or 0)),
            "release_duration_sec": row.get("duration_sec"),


            "release_duration_source": (
                pre.DURATION_SOURCE.name_of(int(row["duration_source"]))
                if row.get("duration_source") is not None
                else None
            ),
            "release_dup_status": (
                pre.DUP_STATUS.name_of(int(row["dup_status"]))
                if row.get("dup_status") is not None
                else None
            ),
            "release_license_family": (
                pre.LICENSE_FAMILY.name_of(int(row["license_family"]))
                if row.get("license_family") is not None
                else None
            ),
            "release_commercial_ok": row.get("commercial_ok"),
            "encapsulation": storage_class,
            "archive_identity": origin,
            "language_source": "release" if _language_hint(language) else "none",
        },
    }


def _language_hint(language: str) -> str | None:

    if language in _NON_LANGUAGE_CODES:
        return None
    return language or None
