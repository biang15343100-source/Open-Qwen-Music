from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from open_qwen_music.llm.common.config import load_config
from open_qwen_music.llm.trainer import validate_training_topology
from open_qwen_music.render.trainer_dit import validate_runtime_training_topology


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIGS = ROOT / "configs" / "train"


def _yaml(name: str) -> dict:
    value = yaml.safe_load((TRAIN_CONFIGS / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    ("name", "global_batch_key"),
    (
        ("tokenizer-stage1a.yaml", "expected_global_batch_size"),
        ("tokenizer-stage1b.yaml", "expected_global_batch_size"),
        ("tokenizer-stage2b.yaml", "expected_global_batch_size"),
        ("tokenizer-stage3-head-warmup.yaml", "expected_global_batch_size"),
        ("tokenizer-stage3.yaml", "expected_global_batch_size"),
        ("tokenizer-stage4.yaml", "expected_global_batch_size"),
        ("vae-stage1.yaml", "expected_global_batch_size"),
        ("vae-stage2.yaml", "expected_global_batch_size"),
        ("refiner.yaml", "expected_global_batch_size"),
    ),
)
def test_training_recipe_uses_64_processes(name: str, global_batch_key: str) -> None:
    train = _yaml(name)["train"]
    assert train["expected_world_size"] == 64
    accumulation = int(train.get("gradient_accumulation_steps", 1))
    assert (
        64 * int(train["batch_size_per_rank"]) * accumulation
        == int(train[global_batch_key])
    )


def test_llm_preserves_its_global_batch_on_64_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OQM_TOKENIZER_REVISION", "a" * 64)
    monkeypatch.setenv("OQM_TOKENIZER_EXTRACTOR_REVISION", "b" * 64)
    config = load_config(TRAIN_CONFIGS / "llm-stage2.yaml")
    train = config["train"]
    assert train["expected_world_size"] == 64
    assert (
        64
        * int(train["max_batch_size"])
        * int(train["gradient_accumulation_steps"])
        == int(train["expected_global_batch_size"])
        == 128
    )
    validate_training_topology(config, world_size=64)
    with pytest.raises(RuntimeError, match="world size"):
        validate_training_topology(config, world_size=32)


def test_renderer_preserves_its_global_batch_on_64_processes() -> None:
    config = _yaml("renderer.yaml")
    assert config["train"]["expected_world_size"] == 64
    assert config["train"]["gradient_accumulation_steps"] == 4
    assert config["train"]["global_parent_batch_size"] == 256
    assert validate_runtime_training_topology(config, world_size=64) == {
        "expected_world_size": 64,
        "actual_world_size": 64,
        "global_batch_size": 256,
    }
    with pytest.raises(RuntimeError, match="world size"):
        validate_runtime_training_topology(config, world_size=128)


def test_annotation_recipe_uses_64_gpu_workers() -> None:
    config = yaml.safe_load(
        (ROOT / "data_pipeline" / "configs" / "base.yaml").read_text(
            encoding="utf-8"
        )
    )
    for stage in ("separate", "structure", "asr.vocal", "asr.mix", "align", "voice", "tags.llm"):
        assert config[stage]["nnodes"] * config[stage]["nproc_per_node"] == 64
