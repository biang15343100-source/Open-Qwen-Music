from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from open_qwen_music.llm import materialize
from open_qwen_music.llm.data.shards import CorpusReader, TokenSpan


SEMANTIC_SHA = "a" * 64
EXTRACTOR_SHA = "b" * 64
RMVPE_SHA = "c" * 64
MELODY_REVISION = "sha256:" + "d" * 64


class _SemanticModel:
    feature_extractor = SimpleNamespace(
        lengths=lambda values: torch.div(values, 240, rounding_mode="floor")
    )
    subsampling = SimpleNamespace(
        output_lengths=lambda values: torch.div(values + 3, 4, rounding_mode="floor")
    )

    def encode_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        assert sample_rate == 24_000
        assert attention_mask.all()
        frames = round(waveform.shape[-1] / sample_rate * 25)
        return SimpleNamespace(
            token_ids=torch.arange(frames, device=waveform.device).unsqueeze(0),
            frame_mask=torch.ones((1, frames), dtype=torch.bool, device=waveform.device),
            frame_rate=25.0,
            codebook_size=32_768,
            tokenizer_revision=SEMANTIC_SHA,
        )


class _MelodyTokenizer:
    revision = MELODY_REVISION
    pitch_extractor = SimpleNamespace(checkpoint_sha256=RMVPE_SHA)

    def encode_waveform(self, waveform: torch.Tensor, sample_rate: int) -> SimpleNamespace:
        assert sample_rate == 16_000
        assert waveform.shape == (16_000,)
        return SimpleNamespace(
            token_ids=np.array([127, 128, 255, 126, 127, 255, 127], dtype=np.uint8),
            frame_rate=6.25,
        )


def _write_wav(path: Path) -> None:
    samples = np.zeros(24_000, dtype="<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(samples.tobytes())


def _patch_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        materialize,
        "_load_semantic_tokenizer",
        lambda artifact, device: materialize.SemanticTokenizerRuntime(
            model=_SemanticModel(),
            artifact_sha256=SEMANTIC_SHA,
            extractor_sha256=EXTRACTOR_SHA,
        ),
    )
    monkeypatch.setattr(
        materialize,
        "_load_melody_tokenizer",
        lambda *args, **kwargs: _MelodyTokenizer(),
    )


def test_materializes_atomic_corpus_with_frozen_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_models(monkeypatch)
    audio = tmp_path / "track.wav"
    _write_wav(audio)
    source = tmp_path / "stage34.jsonl"
    source.write_text(
        json.dumps(
            {
                "sample_id": "track-001",
                "audio": {"path": f"file://{audio}", "duration_sec": 1.0},
                "text": {
                    "language": "en",
                    "lyrics": "[verse]\nA quiet light\n[chorus] Sing it again",
                },
                "description": "Warm acoustic folk with a gentle vocal.",
                "tags": {"genre": ["folk"], "mood": ["calm"]},
                "split": "validation",
                "source": {
                    "dataset": "community",
                    "read_hints": {"private_path": "/private/archive"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "llm-corpus"
    corpus_json = materialize.materialize_llm_corpus(
        input_manifest=source,
        output_dir=output,
        tokenizer_artifact=tmp_path / "tokenizer.pt",
        rmvpe_checkpoint=tmp_path / "rmvpe.pt",
        shard_bytes=32,
    )

    assert corpus_json == output / "corpus.json"
    metadata = json.loads(corpus_json.read_text(encoding="utf-8"))
    assert metadata["tokenizer_revision"] == SEMANTIC_SHA
    assert metadata["semantic_extractor_revision"] == EXTRACTOR_SHA
    assert metadata["melody_tokenizer_revision"] == MELODY_REVISION
    assert metadata["rmvpe_checkpoint_sha256"] == RMVPE_SHA
    assert metadata["source_manifest"]["file"] == "stage34.jsonl"
    assert len(metadata["source_manifest"]["sha256"]) == 64

    reader = CorpusReader(output)
    reader.validate_integrity(require=True)
    row = json.loads(reader.manifest_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["split"] == "valid"
    assert row["language"] == "en"
    assert row["quality"] == {"bucket": "Q3", "genre": "folk"}
    assert row["description"] == "Warm acoustic folk with a gentle vocal."
    assert row["tags"] == {"genre": ["folk"], "mood": ["calm"]}
    assert row["source"] == {"dataset": "community"}
    assert row["sections"] == [
        {"label": "verse", "lyrics": "A quiet light"},
        {"label": "chorus", "lyrics": "Sing it again"},
    ]
    assert "audio" not in row
    semantic = reader.read_semantic(TokenSpan.from_dict(row["semantic"]))
    melody = reader.read_melody(TokenSpan.from_dict(row["melody"]))
    assert semantic.tolist() == list(range(25))
    assert melody.tolist() == [127, 128, 255, 126, 127, 255, 127]
    reader.close()

    with pytest.raises(FileExistsError, match="does not overwrite or resume"):
        materialize.materialize_llm_corpus(
            input_manifest=source,
            output_dir=output,
            tokenizer_artifact=tmp_path / "tokenizer.pt",
            rmvpe_checkpoint=tmp_path / "rmvpe.pt",
        )


def test_failed_materialization_leaves_no_partial_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_models(monkeypatch)
    source = tmp_path / "stage34.jsonl"
    source.write_text(
        json.dumps({"sample_id": "missing", "audio": {"path": "missing.wav"}}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "llm-corpus"

    with pytest.raises(FileNotFoundError):
        materialize.materialize_llm_corpus(
            input_manifest=source,
            output_dir=output,
            tokenizer_artifact=tmp_path / "tokenizer.pt",
            rmvpe_checkpoint=tmp_path / "rmvpe.pt",
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".llm-corpus.tmp-*"))
