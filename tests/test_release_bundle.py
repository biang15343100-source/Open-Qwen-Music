from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from open_qwen_music.common.config import load_config
from open_qwen_music.release import (
    BUNDLE_FORMAT,
    DEFAULT_REVISION,
    OpenQwenMusicWeightBundle,
    _load_safetensors,
)


def _write_bundle(root: Path, component_path: str = "language-model") -> None:
    (root / "language-model").mkdir(parents=True)
    (root / "open_qwen_music.json").write_text(
        json.dumps(
            {
                "format_version": BUNDLE_FORMAT,
                "components": {"language_model": {"path": component_path}},
                "contracts": {
                    "semantic_token": {"revision": "semantic-tokenizer-v1"}
                },
            }
        ),
        encoding="utf-8",
    )


def test_local_bundle_resolves_component(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    bundle = OpenQwenMusicWeightBundle.from_pretrained(tmp_path)
    assert bundle.component_dir("language_model") == tmp_path / "language-model"
    assert bundle.semantic_tokenizer_revision == "semantic-tokenizer-v1"


def test_bundle_rejects_component_outside_snapshot(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "../outside")
    with pytest.raises(ValueError, match="escapes"):
        OpenQwenMusicWeightBundle.from_pretrained(tmp_path).component_dir(
            "language_model"
        )


def test_safetensors_loader_requires_exact_keys(tmp_path: Path) -> None:
    source = torch.nn.Linear(3, 2)
    save_file(source.state_dict(), str(tmp_path / "model.safetensors"))
    target = torch.nn.Linear(3, 2)
    _load_safetensors(target, tmp_path)
    for expected, actual in zip(source.parameters(), target.parameters()):
        assert torch.equal(expected, actual)


def test_default_pipeline_uses_published_bundle_and_temperature() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/inference/pipeline.yaml")
    assert config["format_version"] == "oqm.inference.v2"
    assert config["model"]["repo_id"] == "oqmtest1451/open-qwen-music-weights"
    assert config["model"]["revision"] == DEFAULT_REVISION
    assert config["model"]["revision"] != "main"
    assert config["llm"]["temperature"] == 0.7


def test_language_model_training_recipe_is_bound_to_the_artifact(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    manifest_path = tmp_path / "open_qwen_music.json"
    manifest = json.loads(manifest_path.read_text())
    artifact_sha = "a" * 64
    manifest["artifact_sha256"] = {"language_model": artifact_sha}
    manifest["components"]["language_model"]["training"] = {
        "sequence_mode": "plain",
        "stage1_updates": 5_000,
        "stage2_updates": 5_000,
        "total_updates": 10_000,
        "artifact_sha256": artifact_sha,
    }
    manifest_path.write_text(json.dumps(manifest))

    OpenQwenMusicWeightBundle.from_pretrained(
        tmp_path
    ).validate_language_model_training_recipe()

    manifest["components"]["language_model"]["training"]["total_updates"] = 9_999
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="training recipe mismatch"):
        OpenQwenMusicWeightBundle.from_pretrained(
            tmp_path
        ).validate_language_model_training_recipe()
