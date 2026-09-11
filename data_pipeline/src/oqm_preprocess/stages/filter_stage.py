
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from .. import flags, text
from ..config import FilterConfig
from ..enums import CONTENT_TYPE, GRANULARITY, LANGUAGE, STATUS
from ..locator import int_field
from ..runtime.context import Context
from ..runtime.log import Progress, get
from ..schema import WORK_SCHEMA

log = get("filter")

STAGE = "s4_filter"
INPUT_STAGE = "s3_enrich"


RULES_VERSION = 2

_ZH_EN = frozenset({"zh", "en", "zh_en", "instrumental", "language_neutral", "unknown"})


def uri_hash(record: dict[str, Any]) -> bytes:
    payload = "|".join((
        str(int_field(record, "container_id")),
        str(record.get("member") or ""),
        str(int_field(record, "parquet_row_group")),
        str(int_field(record, "parquet_row_index")),
        f"{record.get('clip_start_sec')}:{record.get('clip_end_sec')}",
    ))
    return hashlib.sha1(payload.encode(), usedforsecurity=False).digest()[:8]


def apply_row(record: dict[str, Any], cfg: FilterConfig, bucket_sec: float) -> dict[str, Any]:
    mask = int(record.get("flags", 0))
    record["uri_hash"] = uri_hash(record)


    if flags.is_rejected(mask):
        record["status"] = STATUS.code("rejected")
        return record

    mask |= _duration_flags(record, cfg)
    mask |= _acoustic_flags(record, cfg)
    mask |= _text_flags(record, cfg)

    record["flags"] = mask
    record["status"] = STATUS.code("rejected" if flags.is_rejected(mask) else "accepted")

    duration = record.get("duration_sec")
    if duration is not None and bucket_sec > 0:
        record["duration_bucket"] = round(float(duration) / bucket_sec) * bucket_sec
    return record


def _duration_flags(record: dict[str, Any], cfg: FilterConfig) -> int:
    duration = record.get("duration_sec")
    if duration is None or duration <= 0:
        return flags.bit("duration_unknown")

    duration = float(duration)
    granularity = GRANULARITY.name_of(int(record.get("granularity") or 0))
    low, high = cfg.duration.bounds(granularity)
    mask = 0
    if duration < low:
        mask |= flags.bit("too_short")
    if duration > high:
        mask |= flags.bit("too_long")

    declared = record.get("declared_duration_sec")
    if declared:
        declared = float(declared)
        delta = abs(declared - duration)
        if (delta > cfg.duration_absolute_error_sec
                and delta / max(declared, 1e-6) > cfg.duration_relative_error):
            mask |= flags.bit("duration_mismatch")
    return mask


def _acoustic_flags(record: dict[str, Any], cfg: FilterConfig) -> int:
    mask = 0
    silent = record.get("near_silent_frame_ratio")
    if silent is not None and float(silent) >= cfg.near_silent_ratio:
        mask |= flags.bit("near_silent")

    clipping = record.get("clipping_ratio")
    if clipping is not None and float(clipping) >= cfg.clipping_ratio:
        mask |= flags.bit("severe_clipping")

    channels = record.get("channels")
    correlation = record.get("channel_correlation")
    if (channels and int(channels) >= 2 and correlation is not None
            and abs(float(correlation)) >= cfg.fake_stereo_correlation):
        mask |= flags.bit("fake_stereo")

    bandwidth = record.get("effective_bandwidth_ratio")
    if bandwidth is not None and 0 < float(bandwidth) < cfg.min_effective_bandwidth_ratio:
        mask |= flags.bit("upsampled_lowband")

    sample_rate = record.get("sample_rate_hz")
    if sample_rate and int(sample_rate) < cfg.target_sample_rate_hz:
        mask |= flags.bit("sample_rate_below_target")

    duration = record.get("duration_sec")
    lead = record.get("lead_silence_sec")
    if duration and lead and float(lead) / float(duration) >= cfg.heavy_leading_silence_ratio:
        mask |= flags.bit("heavy_leading_silence")


    file_bytes = record.get("file_bytes")
    codec = record.get("codec")
    if file_bytes and duration and codec and float(duration) > 0:
        kbps = float(file_bytes) * 8.0 / 1000.0 / float(duration)
        if kbps < cfg.min_bitrate_kbps:
            mask |= flags.bit("low_bitrate")
    return mask


def _text_flags(record: dict[str, Any], cfg: FilterConfig) -> int:
    mask = 0
    lyrics = record.get("lyrics_text")
    duration = record.get("duration_sec")
    if not lyrics:
        return mask

    units = text.singable_units(lyrics)
    if units < cfg.lyrics_min_chars:
        mask |= flags.bit("lyrics_empty_claimed")
    elif duration and float(duration) > 0:
        per_sec = units / float(duration)


        if per_sec < cfg.lyrics_min_chars_per_sec or per_sec > cfg.lyrics_max_chars_per_sec:
            mask |= flags.bit("lyrics_audio_mismatch")
        else:
            record["lyrics_coverage"] = min(1.0, per_sec / cfg.lyrics_max_chars_per_sec)

    script = text.classify_script(lyrics)
    if script not in _ZH_EN and script != "unknown":
        mask |= flags.bit("lyrics_lang_not_zh_en")
    return mask


def run(ctx: Context, datasets: list[str] | None = None, *, force: bool = False) -> dict[str, Any]:
    import pyarrow.parquet as pq

    src = ctx.store(INPUT_STAGE, WORK_SCHEMA)
    if not src.shard_paths():
        raise RuntimeError(f"{INPUT_STAGE} has no output; run enrich first")

    store = ctx.store(STAGE, WORK_SCHEMA)
    cfg = ctx.cfg.filters
    bucket = ctx.cfg.dedup.duration_bucket_sec
    wanted = {s.slug for s in ctx.registry.select(datasets)}

    lineages = {s.slug: f"{src.lineage_of(s.slug)}+rules-v{RULES_VERSION}"
                for s in ctx.registry.select(datasets)}
    upstream_keys = {p.stem[len("part-"):] for p in src.shard_paths()}
    for slug in wanted:
        store.prune_prefix(slug, upstream_keys)

    counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    per_dataset: dict[int, Counter[str]] = {}
    progress = Progress(log, "S4 filter", total=src.count_rows())

    def recover(key: str) -> None:
        marker = store.read_marker(key) or {}
        counts.update(marker.get("flag_counts", {}))
        status_counts.update(marker.get("status_counts", {}))
        for ds, per_status in (marker.get("per_dataset") or {}).items():
            per_dataset.setdefault(int(ds), Counter()).update(per_status)

    for path in src.shard_paths():
        key = path.stem[len("part-"):]
        slug = key.rsplit("-", 1)[0]
        if slug not in wanted:
            recover(key)
            continue
        if not force and store.is_done(key, lineage=lineages.get(slug)):
            recover(key)
            continue
        records = pq.read_table(path).to_pylist()
        shard_flags: Counter[str] = Counter()
        shard_status: Counter[str] = Counter()
        shard_dataset: dict[int, Counter[str]] = {}
        for record in records:
            apply_row(record, cfg, bucket)
            for name in flags.decode(int(record["flags"])):
                shard_flags[name] += 1
            status = STATUS.name_of(int(record["status"]))
            shard_status[status] += 1
            shard_dataset.setdefault(int(record["dataset_id"]), Counter())[status] += 1
        store.write_shard(key, records, lineage=lineages.get(slug), extra={
            "flag_counts": dict(shard_flags),
            "status_counts": dict(shard_status),
            "per_dataset": {str(ds): dict(c) for ds, c in shard_dataset.items()},
        })
        counts.update(shard_flags)
        status_counts.update(shard_status)
        for ds, per_status in shard_dataset.items():
            per_dataset.setdefault(ds, Counter()).update(per_status)
        progress.advance(len(records))
    progress.done()

    stats = {
        "flag_counts": dict(counts.most_common()),
        "status_counts": dict(status_counts),
        "per_dataset": {
            ctx.registry.get(ds).slug: dict(c) for ds, c in sorted(per_dataset.items())
        },
    }
    store.finalize(stats)
    log.info("S4 filter completed: %s", dict(status_counts))
    return stats


def audit_content_types(records: list[dict[str, Any]]) -> Counter[str]:
    out: Counter[str] = Counter()
    for record in records:
        out[CONTENT_TYPE.name_of(int(record.get("content_type") or 0))] += 1
        out[f"lang:{LANGUAGE.name_of(int(record.get('language') or 0))}"] += 1
    return out
