from __future__ import annotations

import hashlib
import json
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from oqm_preprocess.acoustic_views import AcousticViewError, build_acoustic_views
from oqm_preprocess.cli import main


def _write_source(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_acoustic_views_with_file_and_indexed_tar(tmp_path: Path) -> None:
    loose = tmp_path / "train.wav"
    loose.write_bytes(b"train-audio")
    archive = tmp_path / "audio.tar"
    payloads = {"valid.wav": b"valid-audio", "test.wav": b"test-audio"}
    with tarfile.open(archive, "w") as handle:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, BytesIO(payload))
    with tarfile.open(archive, "r:") as handle:
        members = {member.name: member for member in handle.getmembers()}

    rows = [
        {
            "sample_id": "train",
            "audio": {
                "path": f"file://{loose}",
                "duration_sec": 1.0,
                "sample_rate_hz": 48_000,
                "channels": 2,
            },
            "source": {"dataset": "test"},
            "split": "train",
        }
    ]
    for split in ("valid", "test"):
        member = members[f"{split}.wav"]
        rows.append(
            {
                "sample_id": split,
                "audio": {
                    "path": f"tar://{archive}::{split}.wav",
                    "duration_sec": 1.0,
                    "sample_rate_hz": 48_000,
                    "channels": 2,
                },
                "source": {
                    "dataset": "test",
                    "read_hints": {"data_offset": member.offset_data, "size": member.size},
                },
                "split": split,
            }
        )
    source = tmp_path / "stage34.jsonl"
    _write_source(source, rows)

    output = tmp_path / "training"
    (output / "tokenizer").mkdir(parents=True)
    (output / "tokenizer" / "stage34.jsonl").write_text("preserve me\n", encoding="utf-8")
    result = build_acoustic_views(source, output)

    assert result["splits"] == {"train": 1, "valid": 1, "test": 1}
    for consumer in ("vae-stage1", "vae-stage2", "refiner"):
        for split in ("train", "valid", "test"):
            record = json.loads((output / consumer / f"{split}.jsonl").read_text())
            assert record["sample_id"] == split
            assert record["split"] == split
            assert record["audio"]["sha256"] == hashlib.sha256(
                loose.read_bytes() if split == "train" else payloads[f"{split}.wav"]
            ).hexdigest()
    assert json.loads((output / "ACOUSTIC_READY").read_text())["status"] == "READY"
    assert (output / "tokenizer" / "stage34.jsonl").read_text() == "preserve me\n"


def test_build_acoustic_views_cli_and_missing_split(tmp_path: Path) -> None:
    audio = tmp_path / "track.wav"
    audio.write_bytes(b"audio")
    source = tmp_path / "stage34.jsonl"
    _write_source(
        source,
        [
            {
                "sample_id": "track",
                "audio": {
                    "path": f"file://{audio}",
                    "duration_sec": 1.0,
                    "sample_rate_hz": 48_000,
                    "channels": 2,
                },
                "split": "train",
            }
        ],
    )
    with pytest.raises(AcousticViewError, match="no records for splits"):
        main(
            [
                "build-acoustic-views",
                "--tokenizer-manifest",
                str(source),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
