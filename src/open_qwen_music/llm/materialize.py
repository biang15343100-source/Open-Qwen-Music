"""Build an LLM training corpus from a frozen Tokenizer and RMVPE."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from open_qwen_music.common.checkpoint import file_sha256
from open_qwen_music.tokenizer.audio import (
    audio_password_env,
    load_audio,
    password_from_env,
    resample_mono,
)
from open_qwen_music.tokenizer.contracts import SAMPLE_RATE
from open_qwen_music.tokenizer.deployment import DeploymentMusicTokenizer
from open_qwen_music.tokenizer.semantic_materialization import (
    iter_jsonl,
    require_sample_id,
    require_sha256,
    resolve_audio_uri,
)

from .contracts import MELODY_FRAME_RATE, SEMANTIC_FRAME_RATE
from .data.shards import CorpusReader, CorpusWriter, DEFAULT_SHARD_BYTES
from .melody_tokenizer import MelodyTokenizer
from .rmvpe import RMVPEPitchExtractor, RMVPEProfile


_SECTION_LINE = re.compile(r"^\s*\[([a-zA-Z_-]+)\]\s*(.*)$")
_SECTION_ALIASES = {
    "inst": "instrumental",
    "pre_chorus": "verse",
    "prechorus": "verse",
    "post_chorus": "chorus",
    "postchorus": "chorus",
    "interlude": "instrumental",
}


@dataclass(frozen=True, slots=True)
class SemanticTokenizerRuntime:
    model: Any
    artifact_sha256: str
    extractor_sha256: str


def _load_semantic_tokenizer(
    artifact_path: Path, device: torch.device
) -> SemanticTokenizerRuntime:
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Tokenizer deployment artifact does not exist: {artifact_path}"
        )
    artifact_sha256 = file_sha256(artifact_path)
    sidecar_path = artifact_path.with_suffix(artifact_path.suffix + ".json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"Tokenizer deployment sidecar does not exist: {sidecar_path}"
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("format_version") != "oqm.tokenizer.deploy.v1":
        raise ValueError("Tokenizer sidecar format must be oqm.tokenizer.deploy.v1")
    declared_revision = require_sha256(
        sidecar.get("tokenizer_revision"), "sidecar.tokenizer_revision"
    )
    if declared_revision != artifact_sha256:
        raise ValueError(
            "Tokenizer deployment artifact SHA-256 does not match its sidecar"
        )
    source_identity = sidecar.get("source_checkpoint_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("Tokenizer sidecar is missing source_checkpoint_identity")
    extractor_sha256 = require_sha256(
        source_identity.get("sha256"),
        "sidecar.source_checkpoint_identity.sha256",
    )
    artifact = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(artifact, dict):
        raise ValueError("Tokenizer deployment artifact must contain an object")
    if artifact.get("format_version") != "oqm.tokenizer.deploy.v1":
        raise ValueError("Tokenizer artifact format must be oqm.tokenizer.deploy.v1")
    artifact_source = artifact.get("source_checkpoint_identity")
    if not isinstance(artifact_source, Mapping):
        raise ValueError("Tokenizer artifact is missing source_checkpoint_identity")
    if (
        require_sha256(
            artifact_source.get("sha256"),
            "artifact.source_checkpoint_identity.sha256",
        )
        != extractor_sha256
    ):
        raise ValueError(
            "Tokenizer artifact and sidecar reference different Stage 4 checkpoints"
        )
    contract = artifact.get("semantic_contract")
    if not isinstance(contract, Mapping) or (
        int(contract.get("sample_rate", -1)) != SAMPLE_RATE
        or float(contract.get("frame_rate", -1.0)) != SEMANTIC_FRAME_RATE
        or int(contract.get("codebook_size", -1)) != 32_768
    ):
        raise ValueError(
            "Tokenizer artifact violates the 24 kHz, 25 Hz, 32768-token contract"
        )
    model = DeploymentMusicTokenizer(artifact, revision=artifact_sha256)
    model.eval().to(device)
    return SemanticTokenizerRuntime(
        model=model,
        artifact_sha256=artifact_sha256,
        extractor_sha256=extractor_sha256,
    )


def _load_melody_tokenizer(
    checkpoint_path: Path,
    *,
    device: torch.device,
    verify_checkpoint_sha256: str | None,
) -> MelodyTokenizer:
    extractor = RMVPEPitchExtractor(
        checkpoint_path,
        device=device,
        verify_checkpoint_hash=verify_checkpoint_sha256,
    )
    return MelodyTokenizer(extractor)


def _optional_int(value: Any, *, field: str, minimum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _load_record_audio(
    record: Mapping[str, Any], *, source_manifest: Path
) -> tuple[torch.Tensor, int]:
    audio = record.get("audio")
    audio_fields = audio if isinstance(audio, Mapping) else {}
    uri = resolve_audio_uri(record, base_dir=source_manifest.parent)
    start_sec = float(
        record.get("start_sec", audio_fields.get("start_sec", 0.0)) or 0.0
    )
    duration_value = record.get("duration_sec", audio_fields.get("duration_sec"))
    duration_sec = None if duration_value in (None, "") else float(duration_value)
    archive_offset = _optional_int(
        audio_fields.get("archive_offset"), field="audio.archive_offset", minimum=0
    )
    archive_size = _optional_int(
        audio_fields.get("archive_size"), field="audio.archive_size", minimum=1
    )
    password_env = audio_password_env(record)
    password = password_from_env(password_env) if password_env else None
    waveform, sample_rate = load_audio(
        uri,
        start_sec=start_sec,
        duration_sec=duration_sec,
        archive_offset=archive_offset,
        archive_size=archive_size,
        password=password,
        asset_id=audio_fields.get("asset_id") if uri.startswith("tar://") else None,
        payload_sha256=(
            audio_fields.get("payload_sha256") if uri.startswith("tar://") else None
        ),
        asset_revision=(
            audio_fields.get("asset_revision") if uri.startswith("tar://") else None
        ),
        shard_sha256=(
            audio_fields.get("shard_sha256") if uri.startswith("tar://") else None
        ),
    )
    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("Decoded audio must be a non-empty mono waveform")
    return waveform, int(sample_rate)


def _sections_from_lyrics(lyrics: str) -> list[dict[str, Any]]:
    lyrics = lyrics.strip()
    if not lyrics:
        return []
    sections: list[dict[str, Any]] = []
    current_label = "verse"
    current_lines: list[str] = []
    for line in lyrics.splitlines():
        match = _SECTION_LINE.match(line)
        if match:
            if current_lines:
                sections.append(
                    {"label": current_label, "lyrics": "\n".join(current_lines).strip()}
                )
            current_label = _SECTION_ALIASES.get(
                match.group(1).lower().replace("-", "_"),
                match.group(1).lower().replace("-", "_"),
            )
            current_lines = [match.group(2)] if match.group(2).strip() else []
        else:
            current_lines.append(line)
    if current_lines or not sections:
        sections.append(
            {"label": current_label, "lyrics": "\n".join(current_lines).strip()}
        )
    allowed = {"intro", "verse", "chorus", "bridge", "instrumental", "outro", "silence"}
    for section in sections:
        if section["label"] not in allowed:
            raise ValueError(
                f"Unsupported section label in structured lyrics: {section['label']}"
            )
    return sections


def _llm_record(
    source: Mapping[str, Any],
    *,
    line_number: int,
    source_manifest_sha256: str,
    semantic_revision: str,
    extractor_revision: str,
    melody_revision: str,
) -> dict[str, Any]:
    text = source.get("text")
    text_fields = text if isinstance(text, Mapping) else {}
    lyrics = str(text_fields.get("lyrics") or source.get("lyrics") or "")
    sections = source.get("sections")
    if not isinstance(sections, list) or not sections:
        sections = _sections_from_lyrics(lyrics)
    else:
        normalized_sections: list[dict[str, Any]] = []
        for raw in sections:
            if not isinstance(raw, Mapping):
                raise ValueError("Each section must be an object")
            label = str(raw.get("label") or "").lower().replace("-", "_")
            label = _SECTION_ALIASES.get(label, label)
            section: dict[str, Any] = {
                "label": label,
                "lyrics": str(raw.get("lyrics") or ""),
            }
            for field in (
                "start_sec",
                "end_sec",
                "melody_start_frame",
                "melody_end_frame",
            ):
                if raw.get(field) is not None:
                    section[field] = raw[field]
            normalized_sections.append(section)
        allowed = {
            "intro",
            "verse",
            "chorus",
            "bridge",
            "instrumental",
            "outro",
            "silence",
        }
        invalid = sorted(
            {section["label"] for section in normalized_sections} - allowed
        )
        if invalid:
            raise ValueError(f"Unsupported section labels: {invalid}")
        sections = normalized_sections
    language = str(source.get("language") or text_fields.get("language") or "und")
    tags = source.get("tags") if isinstance(source.get("tags"), Mapping) else {}
    quality = (
        source.get("quality") if isinstance(source.get("quality"), Mapping) else {}
    )
    genre = quality.get("genre") or tags.get("genre") or "music"
    if isinstance(genre, (list, tuple)):
        genre = genre[0] if genre else "music"
    split = str(source.get("split") or "train").lower()
    split = {"validation": "valid", "val": "valid"}.get(split, split)
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unsupported split {split!r}; expected train, valid, or test")
    instrumental = source.get("is_instrumental")
    return {
        "split": split,
        "description": str(source.get("description") or "").strip(),
        "tags": dict(tags),
        "sections": sections,
        "language": language,
        "is_instrumental": (
            instrumental if type(instrumental) is bool else not lyrics.strip()
        ),
        "quality": {
            **dict(quality),
            "bucket": str(quality.get("bucket") or "Q3"),
            "genre": str(genre),
        },
        "source": {
            "dataset": str((source.get("source") or {}).get("dataset") or "community")
            if isinstance(source.get("source"), Mapping)
            else "community"
        },
        "materialization": {
            "source_manifest_sha256": source_manifest_sha256,
            "source_line": line_number,
            "semantic_tokenizer_revision": semantic_revision,
            "semantic_extractor_revision": extractor_revision,
            "melody_tokenizer_revision": melody_revision,
        },
    }


def _semantic_tokens(
    runtime: SemanticTokenizerRuntime,
    waveform: torch.Tensor,
    sample_rate: int,
    device: torch.device,
) -> np.ndarray:
    audio = resample_mono(waveform, sample_rate, SAMPLE_RATE).to(device)
    batch = audio.unsqueeze(0)
    with torch.inference_mode():
        result = runtime.model.encode_audio(
            batch,
            SAMPLE_RATE,
            attention_mask=torch.ones_like(batch, dtype=torch.bool),
        )
    if (
        float(result.frame_rate) != SEMANTIC_FRAME_RATE
        or int(result.codebook_size) != 32_768
    ):
        raise ValueError("Semantic Tokenizer output violates the frozen token contract")
    if str(result.tokenizer_revision) != runtime.artifact_sha256:
        raise ValueError("Semantic Tokenizer returned an unexpected artifact revision")
    ids = result.token_ids[0].detach().cpu()
    mask = result.frame_mask[0].detach().cpu().bool()
    if ids.ndim != 1 or mask.shape != ids.shape:
        raise ValueError(
            "Semantic token IDs and mask must have matching one-dimensional shapes"
        )
    values = ids[mask].to(torch.int64).numpy()
    if values.size == 0 or int(values.min()) < 0 or int(values.max()) >= 32_768:
        raise ValueError("Semantic Tokenizer returned empty or out-of-range token IDs")
    input_length = torch.tensor([audio.numel()], dtype=torch.long, device=device)
    feature_length = runtime.model.feature_extractor.lengths(input_length)
    expected = int(runtime.model.subsampling.output_lengths(feature_length).item())
    if values.size != expected:
        raise ValueError(
            f"Semantic Tokenizer returned {values.size} frames; expected {expected} from audio duration"
        )
    return values.astype(np.uint16, copy=False)


def materialize_llm_corpus(
    *,
    input_manifest: str | Path,
    output_dir: str | Path,
    tokenizer_artifact: str | Path,
    rmvpe_checkpoint: str | Path,
    device: str = "cpu",
    verify_rmvpe_sha256: str | None = None,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
) -> Path:
    source_path = Path(input_manifest).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Tokenizer Stage 3/4 manifest does not exist: {source_path}"
        )
    if output_path.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_path}. "
            "This command does not overwrite or resume a corpus; choose a new directory."
        )
    if shard_bytes <= 0:
        raise ValueError("shard_bytes must be positive")
    runtime_device = torch.device(device)
    semantic = _load_semantic_tokenizer(
        Path(tokenizer_artifact).expanduser().resolve(), runtime_device
    )
    melody = _load_melody_tokenizer(
        Path(rmvpe_checkpoint).expanduser().resolve(),
        device=runtime_device,
        verify_checkpoint_sha256=verify_rmvpe_sha256,
    )
    source_sha256 = file_sha256(source_path)
    rmvpe_sha256 = require_sha256(
        melody.pitch_extractor.checkpoint_sha256,
        "RMVPE checkpoint SHA-256",
    )
    melody_revision = str(melody.revision)
    staging = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        with CorpusWriter(
            staging,
            tokenizer_revision=semantic.artifact_sha256,
            shard_bytes=shard_bytes,
            extra_metadata={
                "semantic_extractor_revision": semantic.extractor_sha256,
                "melody_tokenizer_revision": melody_revision,
                "rmvpe_checkpoint_sha256": rmvpe_sha256,
                "rmvpe_profile": RMVPEProfile.released().to_dict(),
                "source_manifest": {
                    "file": source_path.name,
                    "sha256": source_sha256,
                },
            },
        ) as writer:
            for line_number, source_record in iter_jsonl(source_path):
                sample_id = require_sample_id(
                    source_record, location=f"{source_path}:{line_number}"
                )
                waveform, sample_rate = _load_record_audio(
                    source_record, source_manifest=source_path
                )
                semantic_ids = _semantic_tokens(
                    semantic, waveform, sample_rate, runtime_device
                )
                rmvpe_audio = resample_mono(waveform, sample_rate, 16_000)
                melody_result = melody.encode_waveform(rmvpe_audio, 16_000)
                if float(melody_result.frame_rate) != MELODY_FRAME_RATE:
                    raise ValueError(
                        "Melody Tokenizer output must use the 6.25 Hz contract"
                    )
                record = _llm_record(
                    source_record,
                    line_number=line_number,
                    source_manifest_sha256=source_sha256,
                    semantic_revision=semantic.artifact_sha256,
                    extractor_revision=semantic.extractor_sha256,
                    melody_revision=melody_revision,
                )
                writer.add(
                    sample_id=sample_id,
                    semantic=semantic_ids,
                    melody=melody_result.token_ids,
                    record=record,
                )
        corpus = CorpusReader(staging)
        try:
            corpus.validate_integrity(require=True)
        finally:
            corpus.close()
        os.replace(staging, output_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_path / "corpus.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Materialize semantic and melody shards for language-model training"
    )
    parser.add_argument("--input", required=True, help="Tokenizer stage34.jsonl")
    parser.add_argument("--output-dir", required=True, help="New LLM corpus directory")
    parser.add_argument(
        "--tokenizer-artifact", required=True, help="Frozen Tokenizer artifact"
    )
    parser.add_argument(
        "--rmvpe-checkpoint", required=True, help="Frozen RMVPE checkpoint"
    )
    parser.add_argument(
        "--device", default="cpu", help="Torch device, such as cpu or cuda"
    )
    parser.add_argument("--verify-rmvpe-sha256")
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    args = parser.parse_args(argv)
    corpus = materialize_llm_corpus(
        input_manifest=args.input,
        output_dir=args.output_dir,
        tokenizer_artifact=args.tokenizer_artifact,
        rmvpe_checkpoint=args.rmvpe_checkpoint,
        device=args.device,
        verify_rmvpe_sha256=args.verify_rmvpe_sha256,
        shard_bytes=args.shard_bytes,
    )
    print(corpus)


if __name__ == "__main__":
    main()
