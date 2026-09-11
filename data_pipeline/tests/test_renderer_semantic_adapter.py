from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from oqm_preprocess.renderer_semantic_adapter import (
    RendererSemanticAdapterError,
    adapt_renderer_semantics,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_adapts_generic_semantic_release_atomically(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    tokens = source / "tokens.npy"
    np.save(tokens, np.arange(25, dtype=np.uint16), allow_pickle=False)
    manifest = source / "semantic_tokens.jsonl"
    manifest.write_text(json.dumps({
        "schema_version": "oqm.semantic-tokens.v1", "sample_id": "track", "split": "train",
        "source_audio_sha256": "b" * 64,
        "artifact": {"path": tokens.name, "sha256": _sha(tokens), "dtype": "uint16",
                     "shape": [25], "frame_hz": 25.0, "codebook_size": 32768},
    }) + "\n", encoding="utf-8")
    (source / "READY").write_text(json.dumps({
        "schema_version": "oqm.semantic-tokens-ready.v1", "status": "READY",
        "manifest": manifest.name, "manifest_sha256": _sha(manifest), "records": 1,
        "tokenizer_revision": "a" * 64,
    }), encoding="utf-8")
    output = tmp_path / "renderer-semantics"
    result = adapt_renderer_semantics(source, output)
    assert result["records"] == 1
    assert _sha(output / "semantic_tokens.jsonl") == result["manifest_sha256"]
    assert (output / "READY").is_file()
    ready_sha = _sha(output / "READY")
    with pytest.raises(FileExistsError):
        adapt_renderer_semantics(source, output)
    assert _sha(output / "READY") == ready_sha


def test_adapts_llm_corpus_and_rejects_corrupt_shards_atomically(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "track.wav"
    audio.write_bytes(b"public audio identity")
    source_manifest = tmp_path / "stage34.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "sample_id": "track",
                "split": "validation",
                "audio": {"path": audio.name, "sha256": _sha(audio)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "llm-corpus"
    semantic_dir = corpus / "semantic"
    semantic_dir.mkdir(parents=True)
    shard = semantic_dir / "shard-00000.bin"
    shard.write_bytes(np.arange(25, dtype="<u2").tobytes())
    manifest = corpus / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "track",
                "split": "valid",
                "semantic": {
                    "shard": shard.name,
                    "frame_offset": 0,
                    "num_frames": 25,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (corpus / "corpus.json").write_text(
        json.dumps(
            {
                "format_version": "oqm.llm.corpus.v1",
                "manifest": manifest.name,
                "manifest_sha256": _sha(manifest),
                "records": 1,
                "tokenizer_revision": "a" * 64,
                "semantic": {
                    "frame_rate": 25.0,
                    "codebook_size": 32768,
                    "dtype": "<u2",
                    "shards": [
                        {
                            "file": shard.name,
                            "bytes": shard.stat().st_size,
                            "sha256": _sha(shard),
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "renderer-semantics"
    result = adapt_renderer_semantics(
        corpus,
        output,
        source_manifest=source_manifest,
    )
    assert result["source_format"] == "oqm.llm.corpus.v1"
    assert result["splits"] == {"valid": 1}

    broken = tmp_path / "broken-output"
    shard.write_bytes(b"corrupt")
    with pytest.raises(RendererSemanticAdapterError, match="integrity"):
        adapt_renderer_semantics(
            corpus,
            broken,
            source_manifest=source_manifest,
        )
    assert not broken.exists()
