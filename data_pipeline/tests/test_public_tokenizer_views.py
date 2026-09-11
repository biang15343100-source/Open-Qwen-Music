from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from oqm_preprocess import enums
from oqm_preprocess.schema import CORPUS_SCHEMA, new_record, records_to_table
from oqm_preprocess.store.containers import ContainerTable
from oqm_preprocess.tokenizer_views import build_tokenizer_views


def test_build_tokenizer_views_from_ready_release(tmp_path: Path) -> None:
    release = tmp_path / "oqm-corpus-v1"
    (release / "corpus").mkdir(parents=True)
    (release / "meta").mkdir()
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"not decoded while building metadata views")
    containers = ContainerTable()
    container_id = containers.intern(audio, "loose")
    containers.write(release / "meta" / "container_table.parquet")
    uid = bytes.fromhex("0011223344556677")
    record = new_record(
        uid=uid,
        dataset_id=1,
        local_id="song",
        storage_class=enums.STORAGE_CLASS.code("loose", strict=True),
        container_id=container_id,
        duration_sec=120.0,
        granularity=enums.GRANULARITY.code("whole_song", strict=True),
        language=enums.LANGUAGE.code("en", strict=True),
        lyrics_text="release lyrics",
        split=enums.SPLIT.code("train", strict=True),
        status=enums.STATUS.code("accepted", strict=True),
    )
    pq.write_table(
        records_to_table([record], CORPUS_SCHEMA), release / "corpus" / "part.parquet"
    )
    (release / "READY").write_text("{}\n", encoding="utf-8")
    (release / "VERSION.json").write_text(
        json.dumps({"schema_version": "oqm.corpus.v1", "release_version": "v1"}),
        encoding="utf-8",
    )
    (release / "meta" / "dataset_registry.json").write_text(
        json.dumps({"datasets": [{"dataset_id": 1, "slug": "community-music"}]}),
        encoding="utf-8",
    )
    annotations = tmp_path / "annotation.jsonl"
    annotations.write_text(
        json.dumps(
            {
                "sample_id": f"oqm:{uid.hex()}",
                "language": "en",
                "description": "Warm acoustic folk with a gentle vocal.",
                "tags": {
                    "genre": ["folk"],
                    "mood": ["calm"],
                    "instrument": ["acoustic guitar"],
                    "vocal_gender": "male",
                },
                "structured_lyrics": "[verse]\nannotated lyrics",
                "sections": [
                    {
                        "label": "inst",
                        "start_sec": 0.0,
                        "end_sec": 10.0,
                        "is_vocal": False,
                        "lyrics": "",
                    },
                    {
                        "label": "verse",
                        "start_sec": 10.0,
                        "end_sec": 20.0,
                        "is_vocal": True,
                        "lyrics": "annotated lyrics",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "training" / "tokenizer"
    result = build_tokenizer_views(release, output, annotations=annotations)

    assert result["stage12_records"] == 1
    assert result["stage34_records"] == 1
    row = json.loads((output / "stage34.jsonl").read_text().splitlines()[0])
    assert row["text"]["lyrics"] == "[verse]\nannotated lyrics"
    assert row["description"] == "Warm acoustic folk with a gentle vocal."
    assert row["tags"]["genre"] == ["folk"]
    assert row["sections"][0] == {
        "label": "instrumental",
        "lyrics": "",
        "start_sec": 0.0,
        "end_sec": 10.0,
    }
    assert row["is_instrumental"] is False
    assert row["quality"] == {"bucket": "Q3", "genre": "folk"}
    assert row["audio"]["path"] == f"file://{audio}"
    assert (output / "stage4_init.jsonl").is_file()
    assert json.loads((output / "READY").read_text())["status"] == "READY"
