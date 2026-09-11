from __future__ import annotations

import pytest

from open_qwen_music.llm.cli import main


def test_top_level_help_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--help"])
    assert "Usage: oqm-llm" in capsys.readouterr().out


def test_missing_command_fails(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        main([])
    assert "Usage: oqm-llm" in capsys.readouterr().out
