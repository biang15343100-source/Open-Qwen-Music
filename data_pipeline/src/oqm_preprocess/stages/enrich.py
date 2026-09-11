
from __future__ import annotations

import contextlib
import hashlib
import json
import re
from collections import defaultdict, deque
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

from .. import adapters, flags, ids, text
from ..adapters import sidecar
from ..enums import (
    ALIGN_LEVEL,
    CONFIDENCE,
    CONTENT_TYPE,
    LANGUAGE,
    LANGUAGE_EVIDENCE,
    LYRICS_FORMAT,
    SPLIT,
    SPLIT_SOURCE,
    STEM_ROLE,
    TEXT_SOURCE,
)
from ..registry import DatasetSpec, EnrichSpec
from ..runtime.context import Context
from ..runtime.log import Progress, get
from ..schema import LYRICS_TIMELINE_SCHEMA, RAW_META_SCHEMA, WORK_SCHEMA

log = get("enrich")

STAGE = "s3_enrich"
INPUT_STAGE = "s2_probe"

_SPLIT_ALIASES = {
    "train": "train", "training": "train", "tr": "train",
    "valid": "valid", "validation": "valid", "val": "valid", "dev": "valid",
    "test": "test", "eval": "test", "testing": "test",
}


@dataclass(slots=True)
class EnrichReport:
    dataset_id: int
    slug: str
    rows: int = 0
    joined: int = 0
    sidecar_records: int = 0
    key_collisions: int = 0
    with_lyrics: int = 0
    with_transcript: int = 0
    with_timeline: int = 0
    language_resolved: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def join_rate(self) -> float:
        return self.joined / self.rows if self.rows else 0.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any], spec: DatasetSpec) -> EnrichReport:
        fields = {"rows", "joined", "sidecar_records", "key_collisions", "with_lyrics",
                  "with_transcript", "with_timeline", "language_resolved"}
        return cls(
            dataset_id=spec.dataset_id, slug=spec.slug,
            **{k: int(payload.get(k, 0) or 0) for k in fields},
            notes=list(payload.get("notes") or []),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id, "slug": self.slug, "rows": self.rows,
            "joined": self.joined, "join_rate": round(self.join_rate, 4),
            "sidecar_records": self.sidecar_records, "key_collisions": self.key_collisions,
            "with_lyrics": self.with_lyrics, "with_transcript": self.with_transcript,
            "with_timeline": self.with_timeline,
            "language_resolved": self.language_resolved,
            "notes": self.notes[:20],
        }


def _apply_fields(record: dict[str, Any], meta: dict[str, Any], spec: EnrichSpec,
                  scratch: dict[str, Any]) -> None:
    for target, column in spec.fields.items():
        value = meta.get(column)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if target == "lyrics_text":
            _set_lyrics(record, str(value), spec, scratch)
        elif target in ("transcript_text", "caption_text", "prompt_text"):
            cleaned = text.clean(str(value))
            if cleaned:
                record[target] = cleaned
                if target == "transcript_text":
                    record["transcript_source"] = TEXT_SOURCE.code(spec.transcript_source)
                    if text.has_placeholders(cleaned):
                        record["flags"] |= flags.bit("transcript_needs_normalization")
                elif target == "caption_text":
                    record["caption_source"] = TEXT_SOURCE.code(spec.caption_source)
        elif target == "language":
            scratch["language_field"] = str(value).strip()
        elif target == "declared_duration_sec":
            with contextlib.suppress(TypeError, ValueError):
                record["declared_duration_sec"] = float(value)
        elif target == "content_type":
            code = CONTENT_TYPE.code(str(value))
            if code:
                record["content_type"] = code
        elif target == "stem_role":
            code = STEM_ROLE.code(str(value))
            if code:
                record["stem_role"] = code
        else:
            scratch[target] = str(value)


def _set_lyrics(record: dict[str, Any], raw: str, spec: EnrichSpec,
                scratch: dict[str, Any]) -> None:
    parsed = text.normalize_lyrics(raw)
    body = str(parsed["text"])
    if not body:
        return


    if spec.lyrics_is_prompt or parsed["is_prompt"]:
        record["prompt_text"] = body
        record["flags"] |= flags.bit("lyrics_is_prompt")
        return

    record["lyrics_text"] = body
    record["lyrics_source"] = TEXT_SOURCE.code(spec.lyrics_source)
    fmt = str(parsed["format"])
    declared = spec.lyrics_format

    record["lyrics_format"] = LYRICS_FORMAT.code(fmt if fmt != "plain" else declared)
    scratch["lyrics_units"] = parsed["units"]
    scratch["lyrics_script"] = parsed["script"]

    timeline = parsed["timeline"]
    if timeline:
        scratch["timeline"] = timeline
        record["lyrics_align_level"] = ALIGN_LEVEL.code("line")
        record["flags"] |= flags.bit("has_lyrics_timeline")
    elif parsed["labels"]:
        record["lyrics_align_level"] = ALIGN_LEVEL.code("section")


def _resolve_language(record: dict[str, Any], spec: DatasetSpec,
                      scratch: dict[str, Any]) -> bool:
    declared = scratch.get("language_field")
    if declared:
        code = LANGUAGE.code(_canon_language(declared))
        if code:
            record["language"] = code
            record["language_evidence"] = LANGUAGE_EVIDENCE.code("record_field")
            record["language_confidence"] = CONFIDENCE.code("high")
            return True

    script = scratch.get("lyrics_script")
    if script and script != "unknown":
        code = LANGUAGE.code(script)
        if code:
            record["language"] = code
            record["language_evidence"] = LANGUAGE_EVIDENCE.code("lyrics_script")
            record["language_confidence"] = CONFIDENCE.code("medium")
            return True


    return spec.language.default != "unknown"


_LANGUAGE_ALIASES = {
    "chinese": "zh", "mandarin": "zh", "cn": "zh", "zh-cn": "zh", "zh_cn": "zh", "cmn": "zh",
    "english": "en", "eng": "en", "en-us": "en", "en_us": "en",
    "japanese": "ja", "jp": "ja", "jpn": "ja",
    "korean": "ko", "kr": "ko", "kor": "ko",
    "instrumental": "instrumental", "inst": "instrumental", "none": "instrumental",
    "mixed": "zh_en", "zh-en": "zh_en", "bilingual": "zh_en",
}


def _canon_language(value: str) -> str:
    lowered = value.strip().lower()
    return _LANGUAGE_ALIASES.get(lowered, lowered)


def _apply_path_fields(record: dict[str, Any], patterns: dict[str, re.Pattern[str]],
                       scratch: dict[str, Any]) -> None:
    member = str(record.get("member") or "")
    for name, pattern in patterns.items():
        match = pattern.search(member)
        if not match:
            continue
        if not match.groups():
            scratch[name] = match.group(0)
            continue

        value = next((g for g in match.groups() if g), None)
        if value:
            scratch[name] = value


def _group_key(record: dict[str, Any], spec: DatasetSpec, scratch: dict[str, Any]) -> str:
    if spec.group_key_template:
        try:
            rendered = spec.group_key_template.format_map(_Missing(scratch))
        except (KeyError, IndexError):
            rendered = ""
        if rendered and "{" not in rendered:
            return f"{spec.slug}|{rendered}"

    for keys in (("artist", "album"), ("artist",), ("album",)):
        values = [str(scratch[k]) for k in keys if scratch.get(k)]
        if len(values) == len(keys):
            return f"{spec.slug}|{'|'.join(values)}"


    member = str(record.get("member") or record.get("local_id") or "")
    parent = member.rsplit("/", 1)[0] if "/" in member else ""
    if parent:
        return f"{spec.slug}|dir:{parent}"
    return f"{spec.slug}|uid:{bytes(record['uid']).hex()}"


class _Missing(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _apply_split(record: dict[str, Any], meta: dict[str, Any], spec: EnrichSpec) -> bool:
    if not spec.split_field:
        return False
    raw = meta.get(spec.split_field)
    if raw is None:
        return False
    value = spec.split_value_map.get(str(raw), str(raw)).strip().lower()
    canonical = _SPLIT_ALIASES.get(value)
    if canonical is None:
        return False
    record["split"] = SPLIT.code(canonical)
    record["split_source"] = SPLIT_SOURCE.code("official_field")
    return True


class DatasetEnricher:

    def __init__(self, ctx: Context, spec: DatasetSpec) -> None:
        self.ctx = ctx
        self.spec = spec
        self.report = EnrichReport(dataset_id=spec.dataset_id, slug=spec.slug)
        self.dedup_keys = set(ctx.cfg.dedup.external_id_keys)
        self.patterns = {k: re.compile(v) for k, v in spec.path_fields.items()}

        self.file_indexes: list[tuple[EnrichSpec, dict[str, dict[str, Any]]]] = []
        self.inline_specs: list[EnrichSpec] = []
        for enrich_spec in spec.enrich:
            if enrich_spec.format.startswith("inline_"):
                self.inline_specs.append(enrich_spec)
                continue
            index, collisions = sidecar.build_index(enrich_spec)
            self.report.key_collisions += collisions
            self.report.sidecar_records += len(index)
            self.file_indexes.append((enrich_spec, index))
            if not index:
                self.report.notes.append(f"{enrich_spec.format} sidecar is empty")

    @property
    def has_metadata(self) -> bool:
        return bool(self.file_indexes or self.inline_specs)

    def process(self, records: list[dict[str, Any]],
                ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        spec = self.spec
        report = self.report
        report.rows += len(records)
        timeline_rows: list[dict[str, Any]] = []
        raw_meta_rows: list[dict[str, Any]] = []

        indexes = list(self.file_indexes)
        for enrich_spec in self.inline_specs:
            index = _inline_index(self.ctx, spec, enrich_spec, records)
            report.sidecar_records += len(index)
            indexes.append((enrich_spec, index))

        for record in records:
            scratch: dict[str, Any] = {}
            external: dict[str, str] = {}
            raw_payloads: list[dict[str, Any]] = []
            matched = False

            _apply_path_fields(record, self.patterns, scratch)

            for enrich_spec, index in indexes:
                meta = _lookup(record, enrich_spec, index)
                if meta is None:
                    continue
                matched = True
                _apply_fields(record, meta, enrich_spec, scratch)
                _apply_split(record, meta, enrich_spec)
                for column in enrich_spec.external_ids:
                    value = meta.get(column)
                    if value not in (None, ""):
                        external[column] = str(value)
                for column in enrich_spec.group_key_fields:
                    value = meta.get(column)
                    if value not in (None, ""):
                        scratch.setdefault(column, str(value))
                if enrich_spec.keep_raw:
                    raw_payloads.append(meta)

            if matched:
                report.joined += 1
            if _resolve_language(record, spec, scratch):
                report.language_resolved += 1
            if record.get("lyrics_text"):
                report.with_lyrics += 1
            if record.get("transcript_text"):
                report.with_transcript += 1

            group_key = _group_key(record, spec, scratch)
            record["group_key"] = group_key
            record["group_id"] = ids.make_group_id(group_key)


            joinable = {k: v for k, v in external.items() if k in self.dedup_keys}
            if joinable:
                record["external_ids_json"] = json.dumps(
                    joinable, sort_keys=True, ensure_ascii=False
                )

            timeline = scratch.get("timeline")
            if timeline:
                report.with_timeline += 1
                timeline_rows.extend(_timeline_rows(record, timeline))

            if external or raw_payloads:
                record["flags"] |= flags.bit("has_raw_meta")
                raw_meta_rows.append({
                    "uid": record["uid"],
                    "dataset_id": record["dataset_id"],
                    "external_ids": list(external.items()) or None,
                    "tags_raw": _tags(scratch),
                    "raw_json": json.dumps(raw_payloads, ensure_ascii=False, default=str)
                    if raw_payloads else None,
                })

        if spec.adapter and adapters.has(spec.adapter):
            extra = adapters.get(spec.adapter)(spec, records)
            report.notes.append(f"adapter {spec.adapter}: {extra}")
        return timeline_rows, raw_meta_rows

    def finish(self) -> EnrichReport:
        report = self.report
        if self.has_metadata and report.rows and report.join_rate < 0.5:
            report.notes.append(f"join rate is only {report.join_rate:.1%}; the join key may be incorrect")
            log.warning("[%s] metadata hook rate only %.1f%%(%d/%d)",
                        self.spec.slug, report.join_rate * 100, report.joined, report.rows)
        return report


def _tags(scratch: dict[str, Any]) -> list[tuple[str, str]] | None:
    items = [(k, str(v)) for k, v in sorted(scratch.items())
             if k not in ("timeline", "lyrics_units", "lyrics_script") and v not in (None, "")]
    return items or None


def _lookup(record: dict[str, Any], spec: EnrichSpec,
            index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if spec.join is None:

        return index.get(str(record.get("local_id") or ""))
    member = str(record.get("member") or "")
    local_id = str(record.get("local_id") or "")
    for key in sidecar.record_keys(member, local_id, spec.join):
        got = index.get(key)
        if got is not None:
            return got
    return None


def _timeline_rows(record: dict[str, Any], timeline: list[tuple[float, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    duration = record.get("duration_sec")
    for seq, (start, line) in enumerate(timeline):
        end = timeline[seq + 1][0] if seq + 1 < len(timeline) else duration
        rows.append({
            "uid": record["uid"], "seq": seq,
            "start_sec": float(start),
            "end_sec": float(end) if end is not None else None,
            "text": line, "label": None,
        })
    return rows


def _inline_index(ctx: Context, spec: DatasetSpec, enrich_spec: EnrichSpec,
                  records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if enrich_spec.format == "inline_parquet":
        return _inline_parquet(ctx, spec, enrich_spec, records)
    if enrich_spec.format == "inline_sqlite":
        return _inline_sqlite(ctx, spec, enrich_spec, records)
    return {}


def _inline_parquet(ctx: Context, spec: DatasetSpec, enrich_spec: EnrichSpec,
                    records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq

    from ..stages.discover import _container_label

    columns = sorted({*enrich_spec.fields.values(), *enrich_spec.external_ids,
                      *enrich_spec.group_key_fields,
                      *([enrich_spec.split_field] if enrich_spec.split_field else [])})
    by_container: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_container[int(record["container_id"])].append(record)

    index: dict[str, dict[str, Any]] = {}
    source = next((s for s in spec.sources if s.storage_class == "parquet"), None)
    for container_id in by_container:
        path = ctx.containers.path_of(container_id)
        if not path.exists():
            continue
        handle = pq.ParquetFile(path)
        present = [c for c in columns if c in handle.schema_arrow.names]
        if not present:
            continue
        label = _container_label(path, source) if source else path.name
        for rg in range(handle.num_row_groups):
            table = handle.read_row_group(rg, columns=present)
            for row, payload in enumerate(table.to_pylist()):
                index[f"{label}::rg{rg}/row{row}"] = payload
    return index


def _inline_sqlite(ctx: Context, spec: DatasetSpec, enrich_spec: EnrichSpec,
                   records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from ..stages.discover import _container_label

    source = next((s for s in spec.sources if s.storage_class == "sqlite"), None)
    if source is None or not source.table or not source.id_column:
        return {}
    columns = sorted({*enrich_spec.fields.values(), *enrich_spec.external_ids,
                      *enrich_spec.group_key_fields})
    index: dict[str, dict[str, Any]] = {}
    for container_id in {int(r["container_id"]) for r in records}:
        path = ctx.containers.path_of(container_id)
        if not path.exists():
            continue
        label = _container_label(path, source)
        rows = sidecar.read_sqlite_meta(path, source.table, source.id_column, columns)
        for row_id, payload in rows.items():
            index[f"{label}::{row_id}"] = payload
    return index


def _mix(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8"),
                        usedforsecurity=False).hexdigest()[:16]


def dataset_shards(src: Any, dataset_id: int) -> list[Path]:
    prefix = f"part-d{dataset_id:03d}-"
    return [p for p in src.shard_paths() if p.name.startswith(prefix)]


def _reports_on_disk(ctx: Context, store: Any,
                     fresh: list[EnrichReport]) -> list[EnrichReport]:
    by_slug = {r.slug: r for r in fresh}
    for spec in ctx.registry:
        if spec.slug in by_slug:
            continue
        payload = store.read_marker(spec.slug)
        if payload:
            by_slug[spec.slug] = EnrichReport.from_dict(payload, spec)
    return [by_slug[k] for k in sorted(by_slug)]


def _read_records(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _prefetch(paths: Sequence[Path], workers: int) -> Iterator[list[dict[str, Any]]]:
    if workers <= 1:
        for path in paths:
            yield _read_records(path)
        return
    with ThreadPoolExecutor(workers) as pool:
        remaining = iter(paths)
        window = deque(pool.submit(_read_records, p)
                       for p in islice(remaining, workers * 2))
        for path in remaining:
            yield window.popleft().result()
            window.append(pool.submit(_read_records, path))
        while window:
            yield window.popleft().result()


def run(ctx: Context, datasets: list[str] | None = None, *, force: bool = False,
        workers: int | None = None) -> dict[str, Any]:
    src = ctx.store(INPUT_STAGE, WORK_SCHEMA)
    if not src.shard_paths():
        raise RuntimeError(f"{INPUT_STAGE} has no output; run probe first")

    store = ctx.store(STAGE, WORK_SCHEMA)
    timeline_store = ctx.store("s3_lyrics_timeline", LYRICS_TIMELINE_SCHEMA)
    raw_store = ctx.store("s3_raw_meta", RAW_META_SCHEMA)
    specs = ctx.registry.select(datasets)

    reports: list[EnrichReport] = []
    n_workers = workers or ctx.cfg.runtime.workers
    shard_rows = ctx.cfg.runtime.shard_size
    progress = Progress(log, "S3 enrich", total=src.count_rows())
    for spec in specs:
        shards = dataset_shards(src, spec.dataset_id)
        if not shards:
            continue

        lineage = _mix(src.lineage_of(f"d{spec.dataset_id:03d}"), spec.fingerprint(scope="full"))
        if not force and store.is_done(spec.slug, lineage=lineage):
            log.info("[%s] enrich already complete; skipping", spec.slug)
            reports.append(EnrichReport.from_dict(store.read_marker(spec.slug) or {}, spec))
            continue

        enricher = DatasetEnricher(ctx, spec)
        parts: list[str] = []
        keys: set[str] = {spec.slug}
        timeline_batch: list[dict[str, Any]] = []
        raw_batch: list[dict[str, Any]] = []
        total = 0
        part = 0
        batch: list[dict[str, Any]] = []
        timeline_keys: set[str] = set()
        raw_keys: set[str] = set()

        def flush() -> None:
            nonlocal part, batch
            if not batch:
                return
            key = f"{spec.slug}-{part:05d}"
            result = store.write_shard(key, batch, mark=False)
            parts.append(result.path.name)
            keys.add(key)
            if timeline_batch:
                timeline_store.write_shard(key, timeline_batch)
                timeline_keys.add(key)
                timeline_batch.clear()
            if raw_batch:
                raw_store.write_shard(key, raw_batch)
                raw_keys.add(key)
                raw_batch.clear()
            part += 1
            batch = []


        for records in _prefetch(shards, n_workers):
            timeline_rows, raw_rows = enricher.process(records)
            batch.extend(records)
            timeline_batch.extend(timeline_rows)
            raw_batch.extend(raw_rows)
            total += len(records)
            progress.advance(len(records))
            if len(batch) >= shard_rows:
                flush()
        flush()

        report = enricher.finish()
        reports.append(report)
        store.prune_prefix(spec.slug, keys)


        timeline_store.prune_prefix(spec.slug, timeline_keys)
        raw_store.prune_prefix(spec.slug, raw_keys)
        store.mark_done(spec.slug, rows=total, parts=parts, extra=report.as_dict(),
                        lineage=lineage)
        log.info("[%s] enrich completed: %d rows, hook rate %.1f%%, lyrics %d rows",
                 spec.slug, report.rows, report.join_rate * 100, report.with_lyrics)
    progress.done()


    all_reports = _reports_on_disk(ctx, store, reports)

    stats = {
        "datasets": len(all_reports),
        "with_lyrics": sum(r.with_lyrics for r in all_reports),
        "with_transcript": sum(r.with_transcript for r in all_reports),
        "per_dataset": [r.as_dict() for r in all_reports],
    }
    store.finalize(stats)
    timeline_store.finalize({"rows": timeline_store.count_rows()})
    raw_store.finalize({"rows": raw_store.count_rows()})
    return stats
