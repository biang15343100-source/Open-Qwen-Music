
from __future__ import annotations

from typing import Any

import pyarrow as pa

from .enums import STATUS

SCHEMA_VERSION = "oqm.corpus.v1"

UID = pa.binary(8)          # sha1(dataset_slug \0 local_id)[:8]
HASH16 = pa.binary(16)      # content_hash_l1


_IDENTITY = [
    pa.field("uid", UID, nullable=False),
    pa.field("dataset_id", pa.int16(), nullable=False),
    pa.field("local_id", pa.string()),
]


_LOCATOR = [
    pa.field("storage_class", pa.int8(), nullable=False),
    pa.field("container_id", pa.int32(), nullable=False),
    pa.field("member", pa.string()),
    pa.field("member_offset", pa.int64()),
    pa.field("member_header_offset", pa.int64()),
    pa.field("member_size", pa.int64()),

    pa.field("member_compressed_size", pa.int64()),
    pa.field("member_compress", pa.int8()),
    pa.field("sequential_only", pa.bool_()),
    pa.field("parquet_row_group", pa.int32()),
    pa.field("parquet_row_index", pa.int64()),


    pa.field("locator_params", pa.string()),
    pa.field("clip_start_sec", pa.float32()),
    pa.field("clip_end_sec", pa.float32()),
]

_AUDIO = [
    pa.field("duration_sec", pa.float32()),


    pa.field("duration_source", pa.int8()),
    pa.field("sample_rate_hz", pa.int32()),
    pa.field("channels", pa.int8()),
    pa.field("codec", pa.int8()),
    pa.field("file_bytes", pa.int64()),
]


_QUALITY = [
    pa.field("peak_dbfs", pa.float32()),
    pa.field("rms_dbfs", pa.float32()),
    pa.field("near_silent_frame_ratio", pa.float32()),
    pa.field("clipping_ratio", pa.float32()),
    pa.field("channel_correlation", pa.float32()),
    pa.field("effective_bandwidth_ratio", pa.float32()),
    pa.field("lead_silence_sec", pa.float32()),
    pa.field("tail_silence_sec", pa.float32()),
]

_CONTENT = [
    pa.field("content_type", pa.int8()),
    pa.field("granularity", pa.int8()),
    pa.field("domain", pa.int8()),
    pa.field("stem_role", pa.int8()),
    pa.field("is_synthetic", pa.bool_()),
    pa.field("synthetic_model", pa.int8()),
    pa.field("is_derived", pa.bool_()),
    pa.field("parent_uid", UID),
]

_TEXT = [
    pa.field("lyrics_text", pa.string()),
    pa.field("lyrics_format", pa.int8()),
    pa.field("lyrics_align_level", pa.int8()),
    pa.field("lyrics_source", pa.int8()),
    pa.field("lyrics_coverage", pa.float32()),
    pa.field("transcript_text", pa.string()),
    pa.field("transcript_source", pa.int8()),
    pa.field("caption_text", pa.string()),
    pa.field("caption_source", pa.int8()),
    pa.field("prompt_text", pa.string()),
]

_LANG_LICENSE = [
    pa.field("language", pa.int8()),
    pa.field("language_confidence", pa.int8()),
    pa.field("language_evidence", pa.int8()),
    pa.field("license_id", pa.int16()),
    pa.field("license_family", pa.int8()),
    pa.field("commercial_ok", pa.bool_()),
]

_DEDUP = [
    pa.field("content_hash_l1", HASH16),
    pa.field("dup_status", pa.int8()),
    pa.field("canonical_uid", UID),
    pa.field("dup_method", pa.int8()),
]

_SPLIT = [
    pa.field("group_id", pa.uint64()),
    pa.field("split", pa.int8()),
    pa.field("split_source", pa.int8()),
]

_STATUS = [
    pa.field("status", pa.int8(), nullable=False),
    pa.field("flags", pa.uint64(), nullable=False),
]

CORPUS_SCHEMA = pa.schema(
    [
        *_IDENTITY, *_LOCATOR, *_AUDIO, *_QUALITY, *_CONTENT,
        *_TEXT, *_LANG_LICENSE, *_DEDUP, *_SPLIT, *_STATUS,
    ],
    metadata={b"schema_version": SCHEMA_VERSION.encode()},
)


_WORK_ONLY = [
    pa.field("fingerprint_l2", pa.binary(8)),


    pa.field("uri_hash", pa.binary(8)),
    pa.field("duration_bucket", pa.float32()),     # L2 blocking key
    pa.field("declared_duration_sec", pa.float32()),
    pa.field("canonical_score", pa.float32()),
    pa.field("group_key", pa.string()),
    pa.field("external_ids_json", pa.string()),
]

WORK_SCHEMA = pa.schema(
    [*CORPUS_SCHEMA, *_WORK_ONLY],
    metadata={b"schema_version": f"{SCHEMA_VERSION}.work".encode()},
)


LYRICS_TIMELINE_SCHEMA = pa.schema([
    pa.field("uid", UID, nullable=False),
    pa.field("seq", pa.int32(), nullable=False),
    pa.field("start_sec", pa.float32()),
    pa.field("end_sec", pa.float32()),
    pa.field("text", pa.string()),
    pa.field("label", pa.string()),
])


STEMS_SCHEMA = pa.schema([
    pa.field("uid", UID, nullable=False),
    pa.field("role", pa.int8(), nullable=False),
    pa.field("storage_class", pa.int8(), nullable=False),
    pa.field("container_id", pa.int32(), nullable=False),
    pa.field("member", pa.string()),
    pa.field("member_offset", pa.int64()),
    pa.field("member_size", pa.int64()),
])

DUP_EDGES_SCHEMA = pa.schema([
    pa.field("uid", UID, nullable=False),
    pa.field("canonical_uid", UID, nullable=False),
    pa.field("method", pa.int8(), nullable=False),
    pa.field("similarity", pa.float32()),
])

RAW_META_SCHEMA = pa.schema([
    pa.field("uid", UID, nullable=False),
    pa.field("dataset_id", pa.int16(), nullable=False),
    pa.field("external_ids", pa.map_(pa.string(), pa.string())),
    pa.field("tags_raw", pa.map_(pa.string(), pa.string())),
    pa.field("raw_json", pa.string()),
])

CATALOG_EXTERNAL_SCHEMA = pa.schema([
    pa.field("dataset_id", pa.int16(), nullable=False),
    pa.field("local_id", pa.string(), nullable=False),
    pa.field("source_url", pa.string()),
    pa.field("title", pa.string()),
    pa.field("artist", pa.string()),
    pa.field("duration_sec", pa.float32()),
    pa.field("language", pa.int8()),
    pa.field("external_ids", pa.map_(pa.string(), pa.string())),
    pa.field("raw_json", pa.string()),
])

CONTAINER_TABLE_SCHEMA = pa.schema([
    pa.field("container_id", pa.int32(), nullable=False),
    pa.field("path", pa.string(), nullable=False),
    pa.field("storage_class", pa.int8(), nullable=False),
    pa.field("size_bytes", pa.int64()),
    pa.field("mtime_ns", pa.int64()),
])

REJECT_LOG_SCHEMA = pa.schema([
    pa.field("uid", UID, nullable=False),
    pa.field("dataset_id", pa.int16(), nullable=False),
    pa.field("flags", pa.uint64(), nullable=False),
    pa.field("stage", pa.string()),
    pa.field("detail", pa.string()),
])

SIDECAR_SCHEMAS: dict[str, pa.Schema] = {
    "lyrics_timeline": LYRICS_TIMELINE_SCHEMA,
    "stems": STEMS_SCHEMA,
    "dup_edges": DUP_EDGES_SCHEMA,
    "raw_meta": RAW_META_SCHEMA,
    "catalog_external": CATALOG_EXTERNAL_SCHEMA,
    "container_table": CONTAINER_TABLE_SCHEMA,
    "reject_log": REJECT_LOG_SCHEMA,
}


_DEFAULTS: dict[str, Any] = {
    "dataset_id": -1,
    "storage_class": 0,
    "container_id": -1,
    "member_offset": -1,
    "member_header_offset": -1,
    "member_size": -1,
    "member_compress": 0,
    "sequential_only": False,
    "parquet_row_group": -1,
    "parquet_row_index": -1,
    "is_synthetic": False,
    "is_derived": False,
    "commercial_ok": False,


    "status": STATUS.code("pending_probe", strict=True),
    "flags": 0,
    "dup_status": 0,
    "split": 0,
    "group_id": 0,
}

_WORK_FIELD_NAMES = tuple(WORK_SCHEMA.names)


def new_record(**overrides: Any) -> dict[str, Any]:
    unknown = set(overrides) - set(_WORK_FIELD_NAMES)
    if unknown:
        raise KeyError(f"Unknown field: {sorted(unknown)}")
    record: dict[str, Any] = dict.fromkeys(_WORK_FIELD_NAMES)
    record.update(_DEFAULTS)
    record.update(overrides)
    return record


def records_to_table(records: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    if not records:
        return schema.empty_table()
    columns = []
    for field in schema:
        values = [r.get(field.name) for r in records]
        try:
            columns.append(pa.array(values, type=field.type))
        except (pa.ArrowInvalid, pa.ArrowTypeError, OverflowError) as exc:
            sample = next((v for v in values if v is not None), None)
            raise TypeError(
                f"column {field.name} cannot be converted to {field.type}(Sample {sample!r}): {exc}"
            ) from exc
    return pa.Table.from_arrays(columns, schema=schema)


def project(table: pa.Table, schema: pa.Schema) -> pa.Table:
    columns = [
        table.column(f.name) if f.name in table.column_names
        else pa.nulls(table.num_rows, f.type)
        for f in schema
    ]
    return pa.Table.from_arrays(columns, schema=schema)
