#!/usr/bin/env python3
"""Generate a tiny corpus and execute discover -> publish end to end."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import yaml


def write_tone(path: Path, frequency: float, seconds: float = 3.0) -> None:
    sample_rate = 24_000
    time = np.arange(round(seconds * sample_rate), dtype=np.float64) / sample_rate
    signal = 0.25 * np.sin(2.0 * np.pi * frequency * time)
    pcm = np.rint(signal * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / ".demo",
    )
    args = parser.parse_args()
    root = args.output.resolve()
    audio = root / "audio"
    registry = root / "registry"
    work = root / "work"
    release = root / "release"
    audio.mkdir(parents=True, exist_ok=True)
    registry.mkdir(parents=True, exist_ok=True)

    for index, frequency in enumerate((220.0, 330.0, 440.0, 550.0), 1):
        write_tone(audio / f"tone-{index}.wav", frequency)

    registry_doc = {
        "dataset_id": 1,
        "slug": "demo-tones",
        "name": "Bundled demo tones",
        "group": "demo",
        "sources": [
            {
                "storage_class": "loose",
                "root": str(audio),
                "glob": "*.wav",
            }
        ],
        "expected_item_count": 4,
        "content_type": "instrument_solo",
        "granularity": "clip",
        "domain": "music",
        "license": {
            "id": "DEMO-PUBLIC-DOMAIN",
            "family": "public_domain",
            "commercial_ok": True,
        },
    }
    (registry / "01-demo-tones.yaml").write_text(
        yaml.safe_dump(registry_doc, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    project = Path(__file__).resolve().parents[1]
    config_doc = {
        "base": str(project / "preprocess_configs/pipeline.yaml"),
        "version": "demo-v1",
        "work_dir": str(work),
        "release_dir": str(release),
        "registry_dir": str(registry),
        "runtime": {
            "workers": 2,
            "discover_workers": 2,
            "shard_size": 100,
            "row_group_size": 100,
        },
    }
    config = root / "pipeline.demo.yaml"
    config.write_text(
        yaml.safe_dump(config_doc, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    command = [
        sys.executable,
        "-m",
        "oqm_preprocess",
        "run",
        "--config",
        str(config),
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)

    published = release / "oqm-corpus-demo-v1"
    ready = published / "READY"
    if not ready.exists():
        raise RuntimeError(f"demo release has no READY marker: {published}")
    version = json.loads((published / "VERSION.json").read_text(encoding="utf-8"))
    print(
        f"demo complete: rows={version['rows']} release={published}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
