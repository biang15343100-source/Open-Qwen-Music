from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
RENDERER_CONFIG = ROOT / "configs/train/renderer.yaml"
PUBLIC_TRAINING_DOCS = (
    ROOT / "README.md",
    ROOT / "TRAINING.md",
    ROOT / "docs/training/data-preparation.md",
    ROOT / "docs/training/renderer.md",
)


def _renderer_config() -> dict[str, Any]:
    value = yaml.safe_load(RENDERER_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _normalized_path(value: Any) -> str:
    assert isinstance(value, str) and value
    return re.sub(r"\$\{[^}]+\}", "/release-root", value).replace("//", "/")


def test_renderer_recipe_has_one_data_root_and_no_pinned_release_hashes() -> None:
    config = _renderer_config()
    assert _normalized_path(config["data"]["root"]).endswith("/renderer")
    assert "upstream_revisions" not in config
    assert "required_upstream_sha_keys" not in config
    assert re.search(
        r"\b[0-9a-f]{64}\b", RENDERER_CONFIG.read_text(encoding="utf-8")
    ) is None


def test_materializer_owns_the_release_layout() -> None:
    config = _renderer_config()
    assert set(config["data"]) == {
        "root",
        "split",
        "latent_posterior_mode",
        "latent_posterior_base_seed",
        "latent_posterior_seed_derivation",
        "latent_posterior_epsilon_draw_layout",
        "required_quality_profile",
        "duration_buckets_seconds",
        "batch_size_by_bucket",
        "num_workers",
        "drop_last",
        "shuffle",
    }
    source = (ROOT / "src/open_qwen_music/render/materialize.py").read_text(
        encoding="utf-8"
    )
    for directory in (
        '"samples"',
        '"latents"',
        '"text"',
        '"crops"',
        '"loudness"',
        '"semantic_embedding"',
        '"semantic_corruption"',
    ):
        assert directory in source


def test_public_renderer_workflow_is_ordered_and_portable() -> None:
    renderer_guide = (ROOT / "docs/training/renderer.md").read_text(encoding="utf-8")
    commands = (
        "oqm-adapt-renderer-semantics",
        "oqm-materialize-renderer",
        "bash scripts/launch_torchrun.sh scripts/train.py renderer",
    )
    positions = [renderer_guide.index(command) for command in commands]
    assert positions == sorted(positions)
    semantic_section = renderer_guide[
        positions[0] : positions[1]
    ]
    assert "--source-manifest" in semantic_section
    materialize_section = renderer_guide[positions[1] : positions[2]]
    for flag in (
        "--config",
        "--source-manifest",
        "--semantic-release",
        "--output-dir",
        "--tokenizer-checkpoint",
        "--vae-checkpoint",
        "--text-encoder",
        "--calibration-report",
        "--semantic-top-k",
    ):
        assert flag in materialize_section
    train_section = renderer_guide[positions[2] :]
    assert '--config "$OQM_DATA_ROOT/renderer/renderer.resolved.yaml"' in train_section
    for obsolete in (
        "--adapter-factory",
        "OQM_RENDERER_ADAPTER_FACTORY",
        "--resume-from",
        "step_025000",
        "9,000 frames",
        "360-second",
    ):
        assert obsolete not in renderer_guide

    for path in PUBLIC_TRAINING_DOCS:
        text = path.read_text(encoding="utf-8")
        assert re.search(r"[\u3400-\u9fff]", text) is None, path


def test_renderer_recipe_starts_fresh_and_materializer_publishes_resolved_config() -> None:
    config = _renderer_config()
    assert config["train"]["max_steps"] == 29_000
    assert "resume_semantic_corruption_bootstrap" not in config["train"]
    assert config["semantic_corruption"]["clean_steps"] == 25_000
    assert config["semantic_corruption"]["ramp_steps"] == 1_000
    assert config["semantic_corruption"]["robust_sample_probability"] == 0.25

    source = (ROOT / "src/open_qwen_music/render/materialize.py").read_text(
        encoding="utf-8"
    )
    assert "renderer.resolved.yaml" in source
    assert '"resolved_renderer_config"' in source
    assert "required_upstream_sha_keys" not in source
    assert "upstream_revisions" not in source
    assert "validate_dit_launch_config" in source
