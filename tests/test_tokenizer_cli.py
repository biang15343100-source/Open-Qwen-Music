from __future__ import annotations

from pathlib import Path

import pytest

from open_qwen_music.tokenizer import cli


def test_runtime_checkpoint_preserves_source_but_uses_local_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "shared" / "stage4.pt"
    staged = tmp_path / "local" / "stage4.cached.pt"
    calls = []

    def fake_stage(path: str, *, cache_dir: str) -> Path:
        calls.append((path, cache_dir))
        return staged

    monkeypatch.setattr(cli, "stage_checkpoint_locally", fake_stage)

    actual_source, runtime = cli._runtime_checkpoint(
        str(source), "/tmp/oqm-export-cache"
    )

    assert actual_source == str(source.resolve())
    assert runtime == str(staged)
    assert calls == [(str(source.resolve()), "/tmp/oqm-export-cache")]


def test_inherit_checkpoint_contract_requires_stage4_and_stable_lineage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []
    source = str((tmp_path / "shared.pt").resolve())
    runtime = str(tmp_path / "cached.pt")
    monkeypatch.setattr(
        cli,
        "_validate_subsampling_contract_from_checkpoint",
        lambda config, path: calls.append(("subsampling", path)),
    )
    monkeypatch.setattr(
        cli,
        "_inherit_feature_config_from_checkpoint",
        lambda config, path, *, lineage_checkpoint: calls.append(
            ("features", path, lineage_checkpoint)
        ),
    )
    monkeypatch.setattr(
        cli,
        "checkpoint_runtime_identity",
        lambda path: {"stage": 4, "sha256": "a" * 64},
    )

    identity = cli._inherit_checkpoint_contract(
        {}, source_checkpoint=source, runtime_checkpoint=runtime
    )

    assert identity["sha256"] == "a" * 64
    assert calls == [
        ("subsampling", runtime),
        ("features", runtime, source),
    ]

    monkeypatch.setattr(
        cli,
        "checkpoint_runtime_identity",
        lambda path: {"stage": 3, "sha256": "b" * 64},
    )
    with pytest.raises(ValueError, match="require a Stage 4 checkpoint"):
        cli._inherit_checkpoint_contract(
            {}, source_checkpoint=source, runtime_checkpoint=runtime
        )
