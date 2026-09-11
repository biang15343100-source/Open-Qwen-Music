from __future__ import annotations

from pathlib import Path

import pytest
import torch

from open_qwen_music.llm.common import checkpoint


def test_only_current_checkpoint_format_is_accepted() -> None:
    current = {"format_version": checkpoint.CHECKPOINT_FORMAT_VERSION}
    checkpoint._check_checkpoint_format(current, resume=False)
    checkpoint._check_checkpoint_format(current, resume=True)

    old = {"format_version": "oqm.llm.ckpt.v1"}
    with pytest.raises(RuntimeError, match="does not support"):
        checkpoint._check_checkpoint_format(old, resume=False)
    with pytest.raises(RuntimeError, match="does not support"):
        checkpoint._check_checkpoint_format(old, resume=True)


def test_atomic_save_moves_an_existing_plain_file_to_previous(tmp_path: Path) -> None:
    destination = tmp_path / "last.pt"
    destination.write_bytes(b"previous checkpoint")

    checkpoint.save_checkpoint(
        destination,
        model=torch.nn.Linear(2, 2),
        optimizer=None,
        scheduler=None,
        config={"stage": "stage1"},
        global_step=1,
        provenance={},
    )

    versions = tmp_path / ".last.pt.versions"
    previous = list(versions.glob("previous-*.pt"))
    assert len(previous) == 1
    assert previous[0].read_bytes() == b"previous checkpoint"
    assert all(
        path.name.startswith(("previous-", "step-"))
        for path in versions.iterdir()
        if not path.name.startswith(".")
    )
    assert destination.is_symlink()
    assert destination.resolve().is_file()
