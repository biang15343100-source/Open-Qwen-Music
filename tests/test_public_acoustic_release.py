from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from open_qwen_music.release import BUNDLE_FORMAT, OpenQwenMusicWeightBundle
from open_qwen_music.render.refiner import refiner_config_from_mapping
from open_qwen_music.render.spec_vae import SpecVAEConfig


ROOT = Path(__file__).resolve().parents[1]


def _bundle_with_artifact(tmp_path: Path, payload: bytes) -> OpenQwenMusicWeightBundle:
    component = tmp_path / "acoustic-vae"
    component.mkdir()
    artifact = component / "model.safetensors"
    artifact.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "format_version": BUNDLE_FORMAT,
        "components": {"acoustic_vae": {"path": "acoustic-vae"}},
        "artifact_sha256": {"acoustic_vae": digest},
    }
    (tmp_path / "open_qwen_music.json").write_text(json.dumps(manifest))
    return OpenQwenMusicWeightBundle.from_pretrained(tmp_path)


def test_bundle_verifies_declared_component_artifact(tmp_path: Path) -> None:
    bundle = _bundle_with_artifact(tmp_path, b"published weights")
    assert bundle.verify_component_artifact("acoustic_vae").name == "model.safetensors"


def test_bundle_rejects_modified_component_artifact(tmp_path: Path) -> None:
    bundle = _bundle_with_artifact(tmp_path, b"published weights")
    (tmp_path / "acoustic-vae/model.safetensors").write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        bundle.verify_component_artifact("acoustic_vae")


def test_sharded_component_uses_per_file_checksums(tmp_path: Path) -> None:
    component = tmp_path / "language-model"
    component.mkdir()
    index = component / "model.safetensors.index.json"
    first = component / "model-00001-of-00002.safetensors"
    second = component / "model-00002-of-00002.safetensors"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.first": first.name,
                    "model.second": second.name,
                }
            }
        )
    )
    first.write_bytes(b"first shard")
    second.write_bytes(b"second shard")
    files = (index, first, second)
    files_sha256 = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }
    manifest = {
        "format_version": BUNDLE_FORMAT,
        "components": {
            "language_model": {
                "path": "language-model",
                "files_sha256": files_sha256,
            }
        },
        "artifact_sha256": {"language_model": "f" * 64},
    }
    (tmp_path / "open_qwen_music.json").write_text(json.dumps(manifest))
    bundle = OpenQwenMusicWeightBundle.from_pretrained(tmp_path)

    assert bundle.verify_component_artifact("language_model") == index
    second.write_bytes(b"modified shard")
    with pytest.raises(RuntimeError, match=second.name):
        bundle.verify_component_artifact("language_model")


def test_acoustic_cli_exposes_stop_at_step_without_overwriting_it() -> None:
    tree = ast.parse((ROOT / "scripts/train_acoustic.py").read_text(encoding="utf-8"))
    options = {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    }
    assert "--stop-at-step" in options
    overwritten = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "args"
            and target.attr == "stop_at_step"
            for target in node.targets
        )
    ]
    assert overwritten == []


def test_inference_cli_forwards_local_text_encoder() -> None:
    tree = ast.parse((ROOT / "scripts/infer.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_pipeline"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    value = keywords["text_encoder_source"]
    assert isinstance(value, ast.Attribute)
    assert isinstance(value.value, ast.Name)
    assert (value.value.id, value.attr) == ("args", "text_encoder")


def test_refiner_recipe_has_materialized_public_gates() -> None:
    import yaml

    config = yaml.safe_load((ROOT / "configs/train/refiner.yaml").read_text())
    assert "__MATERIALIZE_REQUIRED__" not in json.dumps(config)
    assert config["loss"]["recipe"] == "open_qwen_music_refiner_v1"
    assert config["train"]["quality_gate_steps"] == [1250]
    assert refiner_config_from_mapping(config["model"]["refiner"]).__class__.__name__ == (
        "EarVAE2PublicRefinerConfig"
    )


def test_refiner_uses_the_final_vae_architecture() -> None:
    import yaml

    vae_stage1 = yaml.safe_load((ROOT / "configs/train/vae-stage1.yaml").read_text())
    vae = yaml.safe_load((ROOT / "configs/train/vae-stage2.yaml").read_text())
    refiner = yaml.safe_load((ROOT / "configs/train/refiner.yaml").read_text())
    assert refiner["model"]["spec_vae"] == vae["model"]["spec_vae"]
    for config in (vae_stage1, vae, refiner):
        assert "tokenizer_revision" not in config["contract"]
        assert "semantic_extractor_revision" not in config["contract"]


def test_acoustic_vae_recipe_uses_the_public_checkpoint_schema() -> None:
    import yaml

    config = yaml.safe_load((ROOT / "configs/train/vae-stage2.yaml").read_text())
    parsed = SpecVAEConfig.from_dict(config["model"]["spec_vae"])
    assert parsed.revision == "open-qwen-music-acoustic-vae-v1"
