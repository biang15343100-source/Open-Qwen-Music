from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/open_qwen_music/llm/training_contracts.py"
SPEC = importlib.util.spec_from_file_location("oqm_llm_training_contracts", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CONTRACTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACTS)


def _production_config() -> dict:
    return {
        "data": {"require_production_contract": True},
        "registry": {
            "semantic_tokenizer_revision": "a" * 64,
            "semantic_extractor_revision": "b" * 64,
        },
    }


def test_production_corpus_must_match_both_tokenizer_identities() -> None:
    config = _production_config()
    CONTRACTS.validate_production_tokenizer_identity(
        config,
        tokenizer_revision="a" * 64,
        semantic_extractor_revision="b" * 64,
        source="test corpus",
    )

    with pytest.raises(RuntimeError, match="tokenizer revision"):
        CONTRACTS.validate_production_tokenizer_identity(
            config,
            tokenizer_revision="c" * 64,
            semantic_extractor_revision="b" * 64,
            source="test corpus",
        )

    with pytest.raises(RuntimeError, match="semantic extractor revision"):
        CONTRACTS.validate_production_tokenizer_identity(
            config,
            tokenizer_revision="a" * 64,
            semantic_extractor_revision="c" * 64,
            source="test corpus",
        )


def test_stage2_continues_the_stage1_data_cursor_to_10k() -> None:
    stage1 = yaml.safe_load((ROOT / "configs/train/llm.yaml").read_text())
    stage2 = yaml.safe_load((ROOT / "configs/train/llm-stage2.yaml").read_text())

    assert stage1["train"]["max_steps"] == 5000
    assert stage2["train"]["max_steps"] == 5000
    assert stage2["train"]["inherit_data_state_from_parent"] is True
    assert stage1["sampler"]["modes"] == {"plain": 1.0}
    assert stage1["generation"]["temperature"] == 0.7

    stage2_start = CONTRACTS.initial_data_step(
        optimizer_step=0,
        data_state_active=True,
        parent_step=5000,
        extra={"data_step": 5000},
    )
    assert stage2_start == 5000
    assert stage2_start + stage2["train"]["max_steps"] == 10000


def test_fresh_stage_starts_its_data_cursor_at_the_optimizer_step() -> None:
    assert (
        CONTRACTS.initial_data_step(
            optimizer_step=17,
            data_state_active=False,
            parent_step=5000,
            extra=None,
        )
        == 17
    )
