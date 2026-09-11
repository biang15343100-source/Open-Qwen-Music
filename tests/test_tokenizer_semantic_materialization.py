from __future__ import annotations

import json
from pathlib import Path

from open_qwen_music.tokenizer.semantic_materialization import iter_jsonl


def test_iter_jsonl_yields_each_record_once(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    records = [
        {"sample_id": "sample-1"},
        {"sample_id": "sample-2"},
    ]
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    assert list(iter_jsonl(manifest)) == [(1, records[0]), (2, records[1])]
