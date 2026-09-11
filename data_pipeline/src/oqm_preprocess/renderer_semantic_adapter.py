from __future__ import annotations

import hashlib
import tarfile
import zipfile
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np


INPUT_READY_SCHEMA = "oqm.semantic-tokens-ready.v1"
INPUT_ROW_SCHEMA = "oqm.semantic-tokens.v1"
OUTPUT_READY_SCHEMA = "oqm.renderer-semantic-release.v1"
OUTPUT_ROW_SCHEMA = "oqm.renderer-semantic-token.v1"
FRAME_HZ = 25.0
CODEBOOK_SIZE = 32_768


class RendererSemanticAdapterError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RendererSemanticAdapterError(f"Cannot parse JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise RendererSemanticAdapterError(f"Expected a JSON object: {path}")
    return value


def _rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RendererSemanticAdapterError(
                    f"Invalid JSONL record at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise RendererSemanticAdapterError(
                    f"Expected an object at {path}:{line_number}"
                )
            yield line_number, value


def _resolve(path: Any, *, base: Path, field: str) -> Path:
    if not isinstance(path, str) or not path:
        raise RendererSemanticAdapterError(f"{field} must be a non-empty local path")
    if path.startswith("file://"):
        path = path[7:]
    elif "://" in path:
        raise RendererSemanticAdapterError(f"{field} must reference a local file")
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_audio_identities(path: Path) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for line_number, row in _rows(path):
        sample_id = row.get("sample_id")
        split = {"validation": "valid", "val": "valid"}.get(row.get("split"), row.get("split"))
        audio = row.get("audio")
        if not isinstance(sample_id, str) or not sample_id or sample_id in result:
            raise RendererSemanticAdapterError(
                f"Missing or duplicate source sample_id at {path}:{line_number}"
            )
        if split not in {"train", "valid", "test"} or not isinstance(audio, Mapping):
            raise RendererSemanticAdapterError(f"Invalid source record for {sample_id}")
        digest = audio.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            value = audio.get("path", audio.get("uri"))
            if isinstance(value, str) and value.startswith(("tar://", "zip://")):
                scheme, body = value.split("://", 1)
                archive_name, separator, member = body.partition("::")
                if not separator or not archive_name or not member:
                    raise RendererSemanticAdapterError(f"Invalid archive URI for {sample_id}")
                archive_path = Path(archive_name).expanduser()
                if not archive_path.is_absolute():
                    archive_path = (path.parent / archive_path).resolve()
                if scheme == "tar":
                    with tarfile.open(archive_path, "r:*") as archive:
                        handle = archive.extractfile(member)
                        if handle is None:
                            raise RendererSemanticAdapterError(f"Missing archive member for {sample_id}")
                        payload = handle.read()
                else:
                    with zipfile.ZipFile(archive_path) as archive:
                        payload = archive.read(member)
                digest = hashlib.sha256(payload).hexdigest()
            else:
                audio_path = _resolve(value, base=path.parent, field=f"{sample_id}.audio")
                digest = _sha256(audio_path)
        result[sample_id] = (str(split), digest)
    return result


def _llm_corpus_rows(
    root: Path, *, source_manifest: Path
) -> tuple[list[dict[str, Any]], str, str]:
    corpus_path = root / "corpus.json"
    corpus = _json(corpus_path)
    if corpus.get("format_version") != "oqm.llm.corpus.v1":
        raise RendererSemanticAdapterError("Unsupported LLM corpus format")
    manifest = root / str(corpus.get("manifest") or "manifest.jsonl")
    if corpus.get("manifest_sha256") != _sha256(manifest):
        raise RendererSemanticAdapterError("LLM corpus manifest SHA-256 mismatch")
    semantic = corpus.get("semantic")
    if (
        not isinstance(semantic, Mapping)
        or float(semantic.get("frame_rate", -1.0)) != FRAME_HZ
        or semantic.get("codebook_size") != CODEBOOK_SIZE
        or semantic.get("dtype") not in {"<u2", "uint16"}
    ):
        raise RendererSemanticAdapterError("LLM corpus semantic contract is incompatible")
    shards = semantic.get("shards")
    if not isinstance(shards, list):
        raise RendererSemanticAdapterError("LLM corpus is missing semantic shard identities")
    shard_ids: dict[str, Mapping[str, Any]] = {}
    for item in shards:
        if not isinstance(item, Mapping) or not isinstance(item.get("file"), str):
            raise RendererSemanticAdapterError("LLM corpus semantic shard identity is invalid")
        shard_ids[str(item["file"])] = item
    for name, item in shard_ids.items():
        shard_path = root / "semantic" / name
        if (
            not shard_path.is_file()
            or shard_path.stat().st_size != item.get("bytes")
            or _sha256(shard_path) != item.get("sha256")
        ):
            raise RendererSemanticAdapterError(f"LLM semantic shard integrity failed: {name}")
    sources = _source_audio_identities(source_manifest)
    records: list[dict[str, Any]] = []
    for line_number, row in _rows(manifest):
        sample_id = row.get("sample_id")
        span = row.get("semantic")
        if not isinstance(sample_id, str) or sample_id not in sources or not isinstance(span, Mapping):
            raise RendererSemanticAdapterError(f"Cannot join LLM record at line {line_number}")
        shard_name = span.get("shard")
        offset = span.get("frame_offset")
        frames = span.get("num_frames")
        if (
            shard_name not in shard_ids
            or not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
            or not isinstance(frames, int)
            or isinstance(frames, bool)
            or frames <= 0
        ):
            raise RendererSemanticAdapterError(f"Invalid LLM semantic span for {sample_id}")
        values = np.memmap(
            root / "semantic" / str(shard_name), dtype="<u2", mode="r", offset=offset * 2, shape=(frames,)
        )
        tokens = np.asarray(values).copy()
        if int(tokens.max()) >= CODEBOOK_SIZE:
            raise RendererSemanticAdapterError(f"Out-of-range LLM semantic token for {sample_id}")
        split, audio_sha = sources[sample_id]
        if row.get("split") != split:
            raise RendererSemanticAdapterError(f"LLM/source split mismatch for {sample_id}")
        records.append(
            {
                "sample_id": sample_id,
                "split": split,
                "source_audio_sha256": audio_sha,
                "tokens": tokens,
            }
        )
    if len(records) != corpus.get("records") or set(sources) != {
        str(row["sample_id"]) for row in records
    }:
        raise RendererSemanticAdapterError("LLM corpus and source sample sets must match exactly")
    revision = corpus.get("tokenizer_revision")
    if not isinstance(revision, str) or not revision:
        raise RendererSemanticAdapterError("LLM corpus tokenizer_revision must be non-empty")
    return records, revision, _sha256(corpus_path)


def adapt_renderer_semantics(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    source_manifest: str | Path | None = None,
) -> dict[str, Any]:
    source_root = Path(input_dir).expanduser().resolve(strict=True)
    target = Path(output_dir).expanduser().resolve()
    source_ready_path = source_root / "READY"
    generic_manifest = source_root / "semantic_tokens.jsonl"
    llm_mode = (source_root / "corpus.json").is_file()
    prepared_llm: list[dict[str, Any]] | None = None
    if llm_mode:
        if source_manifest is None:
            raise RendererSemanticAdapterError("--source-manifest is required for an LLM corpus")
        prepared_llm, tokenizer_revision, source_ready_sha = _llm_corpus_rows(
            source_root,
            source_manifest=Path(source_manifest).expanduser().resolve(strict=True),
        )
        ready: dict[str, Any] = {"records": len(prepared_llm)}
    else:
        ready = _json(source_ready_path)
        if ready.get("schema_version") != INPUT_READY_SCHEMA or ready.get("status") != "READY":
            raise RendererSemanticAdapterError("Semantic token READY schema or status is invalid")
        if ready.get("manifest") != generic_manifest.name:
            raise RendererSemanticAdapterError("Semantic token READY must reference semantic_tokens.jsonl")
        if ready.get("manifest_sha256") != _sha256(generic_manifest):
            raise RendererSemanticAdapterError("Semantic token manifest SHA-256 does not match READY")
        tokenizer_revision = ready.get("tokenizer_revision")
        if not isinstance(tokenizer_revision, str) or not tokenizer_revision:
            raise RendererSemanticAdapterError("READY.tokenizer_revision must be non-empty")
        source_ready_sha = _sha256(source_ready_path)
    if target.exists():
        raise FileExistsError(f"Renderer semantic output already exists: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        artifact_dir = staging / "artifacts"
        artifact_dir.mkdir()
        output_rows: list[dict[str, Any]] = []
        sample_ids: set[str] = set()
        split_counts: dict[str, int] = {}
        incoming = (
            enumerate(prepared_llm or (), start=1)
            if llm_mode
            else _rows(generic_manifest)
        )
        for line_number, row in incoming:
            sample_id = row.get("sample_id")
            split = row.get("split")
            artifact = row.get("artifact")
            if not llm_mode and row.get("schema_version") != INPUT_ROW_SCHEMA:
                raise RendererSemanticAdapterError(
                    f"Unsupported semantic token schema at line {line_number}"
                )
            if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
                raise RendererSemanticAdapterError(
                    f"Missing or duplicate sample_id at line {line_number}"
                )
            if split not in {"train", "valid", "test"}:
                raise RendererSemanticAdapterError(f"Invalid split for {sample_id}")
            if llm_mode:
                tokens = np.asarray(row["tokens"])
                expected_sha = None
            else:
                if not isinstance(artifact, Mapping):
                    raise RendererSemanticAdapterError(f"Missing artifact metadata for {sample_id}")
                source_path = _resolve(
                    artifact.get("path"), base=generic_manifest.parent, field=f"{sample_id}.artifact.path"
                )
                expected_sha = artifact.get("sha256")
                if not source_path.is_file() or expected_sha != _sha256(source_path):
                    raise RendererSemanticAdapterError(f"Semantic artifact SHA-256 failed for {sample_id}")
                try:
                    tokens = np.load(source_path, allow_pickle=False)
                except (OSError, ValueError) as exc:
                    raise RendererSemanticAdapterError(f"Cannot load semantic tokens for {sample_id}") from exc
            if (
                tokens.dtype != np.uint16
                or tokens.ndim != 1
                or tokens.size == 0
                or int(tokens.max()) >= CODEBOOK_SIZE
                or (
                    not llm_mode
                    and (
                        artifact.get("dtype") != "uint16"
                        or artifact.get("shape") != [int(tokens.size)]
                        or float(artifact.get("frame_hz", -1.0)) != FRAME_HZ
                        or artifact.get("codebook_size") != CODEBOOK_SIZE
                    )
                )
            ):
                raise RendererSemanticAdapterError(f"Semantic token contract failed for {sample_id}")
            if expected_sha is None:
                temporary = staging / f".{sample_id}.npy"
                with temporary.open("xb") as handle:
                    np.save(handle, tokens, allow_pickle=False)
                expected_sha = _sha256(temporary)
                source_path = temporary
            destination = artifact_dir / f"{expected_sha}.npy"
            if not destination.exists():
                shutil.copyfile(source_path, destination)
            if llm_mode:
                source_path.unlink()
            source_audio_sha = row.get("source_audio_sha256")
            if not isinstance(source_audio_sha, str) or len(source_audio_sha) != 64:
                raise RendererSemanticAdapterError(f"Invalid source audio SHA-256 for {sample_id}")
            output_rows.append(
                {
                    "schema_version": OUTPUT_ROW_SCHEMA,
                    "sample_id": sample_id,
                    "split": split,
                    "source_audio_sha256": source_audio_sha,
                    "tokenizer_revision": tokenizer_revision,
                    "artifact": {
                        "path": str(Path("artifacts") / destination.name),
                        "sha256": expected_sha,
                        "dtype": "uint16",
                        "shape": [int(tokens.size)],
                        "frame_hz": FRAME_HZ,
                        "codebook_size": CODEBOOK_SIZE,
                    },
                }
            )
            sample_ids.add(sample_id)
            split_counts[split] = split_counts.get(split, 0) + 1
        if len(output_rows) != ready.get("records"):
            raise RendererSemanticAdapterError("Semantic token record count does not match READY")
        manifest = staging / "semantic_tokens.jsonl"
        with manifest.open("x", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        result = {
            "schema_version": OUTPUT_READY_SCHEMA,
            "status": "READY",
            "manifest": manifest.name,
            "manifest_sha256": _sha256(manifest),
            "records": len(output_rows),
            "splits": split_counts,
            "tokenizer_revision": tokenizer_revision,
            "source_ready_sha256": source_ready_sha,
            "source_format": "oqm.llm.corpus.v1" if llm_mode else INPUT_READY_SCHEMA,
            "checks": {"artifact_sha256": True, "semantic_contract": True},
        }
        _write_json(staging / "READY", result)
        os.replace(staging, target)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Adapt generic semantic tokens for Renderer training")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-manifest")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            adapt_renderer_semantics(
                args.input_dir,
                args.output_dir,
                source_manifest=args.source_manifest,
            )
        )
    )
    return 0
