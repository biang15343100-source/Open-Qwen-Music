from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from . import enums, locator
from .stages.publish import read_release
from .store.containers import ContainerTable


VIEW_SCHEMA = "oqm.public-tokenizer-view.v1"
MAX_CTC_VOCAB_SIZE = 4096
_SECTION_ALIASES = {"inst": "instrumental"}
_SECTION_LABELS = {
    "intro",
    "verse",
    "chorus",
    "bridge",
    "instrumental",
    "outro",
    "silence",
}


class TokenizerViewError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _dataset_names(release: Path) -> dict[int, str]:
    path = release / "meta" / "dataset_registry.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    values = payload.get("datasets", []) if isinstance(payload, dict) else []
    if isinstance(values, dict):
        values = list(values.values())
    return {
        int(item["dataset_id"]): str(item["slug"])
        for item in values
        if isinstance(item, dict) and item.get("dataset_id") is not None and item.get("slug")
    }


def _vocabulary(records: list[dict[str, Any]]) -> list[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(str(record.get("lyrics") or ""))
    ordered = sorted(counts, key=lambda value: (-counts[value], ord(value)))
    return ["<blank>", "<unk>", *ordered[: MAX_CTC_VOCAB_SIZE - 2]]


def _primary_genre(tags: dict[str, Any]) -> str:
    value = tags.get("genre")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (list, tuple)):
        for item in value:
            if str(item).strip():
                return str(item).strip()
    return "music"


def _llm_sections(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in annotation.get("sections") or []:
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label") or "").strip().lower().replace("-", "_")
        label = _SECTION_ALIASES.get(label, label)
        if label not in _SECTION_LABELS:
            continue
        section: dict[str, Any] = {
            "label": label,
            "lyrics": str(raw.get("lyrics") or "").strip(),
        }
        for field in ("start_sec", "end_sec"):
            value = raw.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                section[field] = float(value)
        result.append(section)
    return result


def _training_record(
    row: dict[str, Any],
    *,
    containers: ContainerTable,
    datasets: dict[int, str],
    annotations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    uid = bytes(row["uid"]).hex()
    dataset = datasets.get(int(row.get("dataset_id") or -1), "community")
    annotation = annotations.get(f"oqm:{uid}") or annotations.get(uid) or {}
    lyrics = str(annotation.get("structured_lyrics") or row.get("lyrics_text") or "")
    language = str(
        annotation.get("language") or enums.LANGUAGE.name_of(int(row.get("language") or 0))
    )
    split = enums.SPLIT.name_of(int(row.get("split") or 0))
    read_hints = {
        "data_offset": int(row.get("member_offset") or -1),
        "header_offset": int(row.get("member_header_offset") or -1),
        "size": int(row.get("member_size") or -1),
        "compressed_size": int(row.get("member_compressed_size") or -1),
        "compress_method": int(row.get("member_compress") or 0),
    }
    audio = {
        "path": locator.to_uri(row, containers),
        "duration_sec": float(row["duration_sec"]),
    }
    if row.get("sample_rate_hz") is not None:
        audio["sample_rate_hz"] = int(row["sample_rate_hz"])
    if row.get("channels") is not None:
        audio["channels"] = int(row["channels"])
    tags = dict(annotation.get("tags") or {})
    sections = _llm_sections(annotation)
    vocal_gender = str(tags.get("vocal_gender") or "").casefold()
    annotated_instrumental = annotation.get("is_instrumental")
    is_instrumental = (
        annotated_instrumental
        if type(annotated_instrumental) is bool
        else vocal_gender == "instrumental" or not lyrics.strip()
    )
    return {
        "sample_id": uid,
        "audio": audio,
        "source": {
            "dataset": dataset,
            "is_synthetic": bool(row.get("is_synthetic", False)),
            "file_bytes": int(row.get("file_bytes") or -1),
            "read_hints": read_hints,
        },
        "text": {"language": language, "lyrics": lyrics},
        "description": str(annotation.get("description") or "").strip(),
        "tags": tags,
        "sections": sections,
        "language": language,
        "is_instrumental": is_instrumental,
        "quality": {
            "bucket": "Q3",
            "genre": _primary_genre(tags),
        },
        "split": split,
        "training": {
            "sampling_group": dataset,
            "source_sampling_group": dataset,
            "ctc_group": "lyrics" if lyrics else "no_lyrics",
            "quality_tier": "accepted",
            "io_group": dataset,
            "sample_weight": 1.0,
            "ctc_weight": 1.0 if lyrics else 0.0,
            "mel_weight": 1.0,
            "chroma_weight": 1.0,
            "vq_weight": 1.0,
            "loss_heads": {
                "ctc": bool(lyrics),
                "mel": True,
                "chroma": True,
                "vq": True,
            },
        },
    }


def build_tokenizer_views(
    release: str | Path,
    output_dir: str | Path,
    *,
    annotations: str | Path | None = None,
    max_full_track_duration_sec: float = 300.0,
    force: bool = False,
) -> dict[str, Any]:
    release_path = Path(release).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if max_full_track_duration_sec <= 0:
        raise ValueError("max_full_track_duration_sec must be positive")
    if output_path.exists():
        if not force:
            raise FileExistsError(
                f"Tokenizer view already exists: {output_path}; pass --force to rebuild it"
            )
        shutil.rmtree(output_path)

    table = read_release(release_path)
    containers = ContainerTable.read(release_path / "meta" / "container_table.parquet")
    datasets = _dataset_names(release_path)
    annotation_rows: dict[str, dict[str, Any]] = {}
    if annotations is not None:
        annotation_path = Path(annotations).expanduser().resolve()
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Annotation JSONL does not exist: {annotation_path}")
        for line_number, line in enumerate(
            annotation_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TokenizerViewError(
                    f"Invalid annotation JSONL at line {line_number}: {exc}"
                ) from exc
            sample_id = str(item.get("sample_id") or "")
            if sample_id:
                annotation_rows[sample_id] = item
    accepted = enums.STATUS.code("accepted", strict=True)
    whole_track = enums.GRANULARITY.code("whole_song", strict=True)
    records: list[dict[str, Any]] = []
    for row in table.to_pylist():
        if int(row["status"]) != accepted or int(row["granularity"]) != whole_track:
            continue
        duration = row.get("duration_sec")
        if duration is None or float(duration) <= 0:
            continue
        records.append(
            _training_record(
                row,
                containers=containers,
                datasets=datasets,
                annotations=annotation_rows,
            )
        )
    if not records:
        raise TokenizerViewError(
            f"Release has no accepted whole-track audio records: {release_path}"
        )
    records.sort(key=lambda item: item["sample_id"])
    full_track_records = [
        record
        for record in records
        if float(record["audio"]["duration_sec"]) <= max_full_track_duration_sec
    ]
    if not full_track_records:
        raise TokenizerViewError(
            "No accepted record satisfies the full-track duration limit required by "
            "Tokenizer stages 3 and 4"
        )

    staging = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        stage12 = staging / "stage12.jsonl"
        stage34 = staging / "stage34.jsonl"
        stage4_init = staging / "stage4_init.jsonl"
        vocab = staging / "ctc_vocab.json"
        _write_jsonl(stage12, records)
        _write_jsonl(stage34, full_track_records)
        _write_jsonl(stage4_init, full_track_records)
        _write_json(vocab, _vocabulary(full_track_records))
        artifacts = {
            path.name: {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (stage12, stage34, stage4_init, vocab)
        }
        descriptor = {
            "schema_version": VIEW_SCHEMA,
            "status": "READY",
            "source_release": str(release_path),
            "annotations": str(Path(annotations).expanduser().resolve()) if annotations else None,
            "source_release_version": json.loads(
                (release_path / "VERSION.json").read_text(encoding="utf-8")
            ),
            "records": {
                "stage12": len(records),
                "stage34": len(full_track_records),
                "stage4_init": len(full_track_records),
            },
            "semantic_contract": {
                "sample_rate_hz": 24000,
                "frame_rate_hz": 25.0,
                "codebook_size": 32768,
            },
            "artifacts": artifacts,
        }
        _write_json(staging / "VIEW.json", descriptor)
        _write_json(staging / "READY", descriptor)
        _write_json(
            staging / "MATERIALIZATION_REQUIRED.json",
            {
                "schema_version": "oqm.model-materialization-plan.v1",
                "required_steps": [
                    {
                        "consumer": "tokenizer-stage4",
                        "input": "stage4_init.jsonl",
                        "operation": "data-dependent codebook initialization",
                        "output": "tokenizer Stage4 initialization checkpoint",
                    },
                    {
                        "consumer": "language-model",
                        "input": "stage34.jsonl",
                        "operation": "semantic and melody token encoding with frozen tokenizers",
                        "output": "versioned semantic and melody token shards",
                    },
                ],
            },
        )
        os.replace(staging, output_path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "output_dir": str(output_path),
        "stage12_records": len(records),
        "stage34_records": len(full_track_records),
        "artifacts": artifacts,
    }
