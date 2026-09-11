from __future__ import annotations

from pathlib import Path

import pytest
import torch

from open_qwen_music.tokenizer.initialization import (
    INITIALIZATION_FORMAT,
    INITIALIZATION_ROLE,
    spherical_kmeans,
)
from open_qwen_music.tokenizer.trainer import (
    _validate_distributed_contract,
    _validate_stage_lineage,
)


def _save(path: Path, *, stage: int, phase: str, **extra) -> None:
    torch.save(
        {
            "format_version": "oqm.tokenizer.ckpt.v1",
            "stage": stage,
            "config": {"phase": phase},
            "model": {},
            **extra,
        },
        path,
    )


def test_stage4_rejects_plain_stage3_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage3.pt"
    _save(checkpoint, stage=3, phase="stage3")
    config = {
        "stage": 4,
        "phase": "stage4",
        "parent_phase": "stage4_initialization",
        "quantizer": {
            "init_sample_frames": 8,
            "init_kmeans_iterations": 2,
            "code_dim": 2,
        },
        "semantic_contract": {"codebook_size": 4},
    }

    with pytest.raises(ValueError, match="data-dependent initialization artifact"):
        _validate_stage_lineage(config, str(checkpoint), None)


def test_stage4_accepts_matching_initialization_artifact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "initialized.pt"
    manifest = tmp_path / "stage4_init.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    import hashlib

    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    _save(
        checkpoint,
        stage=4,
        phase="stage4_initialization",
        format_version=INITIALIZATION_FORMAT,
        artifact_role=INITIALIZATION_ROLE,
        initialization_contract={
            "source_phase": "stage3",
            "sample_frames": 8,
            "kmeans_iterations": 2,
            "codebook_size": 4,
            "code_dim": 2,
            "seed": 11,
            "source_checkpoint_sha256": "a" * 64,
            "manifest_sha256": manifest_sha,
        },
        model={
            "quantizer.codebook": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
            )
        },
    )
    config = {
        "stage": 4,
        "phase": "stage4",
        "parent_phase": "stage4_initialization",
        "quantizer": {
            "init_sample_frames": 8,
            "init_kmeans_iterations": 2,
            "code_dim": 2,
            "initialization_manifest": str(manifest),
        },
        "semantic_contract": {"codebook_size": 4},
        "train": {"seed": 11},
    }

    _validate_stage_lineage(config, str(checkpoint), None)


def test_phase_parent_is_required(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong.pt"
    _save(wrong, stage=1, phase="stage1a")
    config = {
        "stage": 2,
        "phase": "stage2b",
        "parent_phase": "stage1b",
    }

    with pytest.raises(ValueError, match="expected=stage1b actual=stage1a"):
        _validate_stage_lineage(config, str(wrong), None)
    with pytest.raises(ValueError, match="requires --init-from"):
        _validate_stage_lineage(config, None, None)


def test_global_batch_derives_accumulation() -> None:
    config = {
        "train": {
            "batch_size_per_rank": 2,
            "gradient_accumulation_steps": "auto",
            "target_global_batch_size": 128,
        }
    }

    _validate_distributed_contract(config, world_size=8)

    assert config["train"]["gradient_accumulation_steps"] == 8


def test_spherical_kmeans_is_deterministic() -> None:
    samples = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]]
    )

    first, counts = spherical_kmeans(
        samples, clusters=2, iterations=3, chunk_size=2, seed=7
    )
    second, _ = spherical_kmeans(
        samples, clusters=2, iterations=3, chunk_size=2, seed=7
    )

    assert torch.equal(first, second)
    assert int(counts.sum()) == len(samples)
