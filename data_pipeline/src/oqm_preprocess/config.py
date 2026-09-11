
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


CONFIG_SECTIONS = ("version", "filters", "dedup", "split")


STAGE_CONFIG_SECTIONS: dict[str, tuple[str, ...]] = {
    "s1_discover": (),
    "s2_probe": (),
    "s3_enrich": ("dedup",),
    "s3_lyrics_timeline": ("dedup",),
    "s3_raw_meta": ("dedup",),
    "s4_filter": ("filters", "dedup"),
    "s5_dedup": ("dedup",),
    "s5_dup_edges": ("dedup",),
    "s6_normalize": ("split",),
    "s7_publish": ("version",),
}


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


class DurationGate(BaseModel):

    model_config = ConfigDict(extra="forbid")

    whole_song: tuple[float, float] = (10.0, 1800.0)
    clip: tuple[float, float] = (2.0, 1800.0)
    phrase: tuple[float, float] = (0.5, 1800.0)
    note: tuple[float, float] = (0.2, 1800.0)

    def bounds(self, granularity: str) -> tuple[float, float]:
        return getattr(self, granularity, self.clip)


class FilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration: DurationGate = Field(default_factory=DurationGate)
    duration_relative_error: float = 0.05
    duration_absolute_error_sec: float = 1.0

    near_silent_ratio: float = 0.95
    clipping_ratio: float = 0.05
    fake_stereo_correlation: float = 0.999


    min_effective_bandwidth_ratio: float = 0.6
    min_bitrate_kbps: float = 64.0
    target_sample_rate_hz: int = 24000
    heavy_leading_silence_ratio: float = 0.30


    lyrics_min_chars_per_sec: float = 0.15
    lyrics_max_chars_per_sec: float = 12.0
    lyrics_min_chars: int = 4


class DedupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable_l0_uri: bool = True
    enable_l1_bytes: bool = True
    enable_l15_external_id: bool = True
    enable_l2_fingerprint: bool = True

    duration_bucket_sec: float = 0.5

    external_id_keys: list[str] = Field(
        default_factory=lambda: [
            "jamendo_track_id", "youtube_id", "spotify_id", "msd_track_id",
            "fma_track_id", "musicbrainz_id", "freesound_id",
        ]
    )

    weight_lossless: float = 10.0
    weight_sample_rate: float = 5.0
    weight_has_lyrics: float = 3.0
    weight_has_tags: float = 3.0
    weight_duration: float = 2.0
    weight_priority: float = 0.1
    penalty_synthetic: float = 5.0


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_pct: int = 96
    valid_pct: int = 2


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workers: int = 8


    discover_workers: int = 12


    probe_prefetch: int = 6

    probe_prefetch_bytes: int = 268435456

    shard_size: int = 20000


    probe_task_bytes: int = 0

    row_group_size: int = 50000
    compression: str = "zstd"
    compression_level: int = 7
    handle_cache: int = 8

    probe_timeout_sec: int = 300


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_dir: Path
    release_dir: Path
    registry_dir: Path
    version: str = "v1"


    stage_read_roots: dict[str, list[Path]] = Field(default_factory=dict)

    filters: FilterConfig = Field(default_factory=FilterConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    _raw: dict[str, Any] = {}

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw = _expand(raw)
        base = raw.pop("base", None)
        if base:
            parent = cls.load((path.parent / base).resolve())
            merged = _deep_merge(parent.raw_dict(), raw)
        else:
            merged = raw

        for key in ("work_dir", "release_dir", "registry_dir"):
            if key in merged and not str(merged[key]).startswith("/"):
                merged[key] = str((path.parent / str(merged[key])).resolve())
        cfg = cls.model_validate(merged)
        cfg._raw = merged
        return cfg

    def raw_dict(self) -> dict[str, Any]:
        return dict(self._raw) if self._raw else self.model_dump(mode="json")

    def _section(self, name: str) -> Any:
        if name == "version":
            return self.version
        return getattr(self, name).model_dump(mode="json")

    def fingerprint(self, sections: Sequence[str] | None = None) -> str:
        names = CONFIG_SECTIONS if sections is None else tuple(sections)
        payload = {name: self._section(name) for name in names}
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def stage_fingerprint(self, stage: str) -> str:
        sections = STAGE_CONFIG_SECTIONS.get(stage)
        return self.fingerprint(sections)

    def stage_dir(self, stage: str) -> Path:
        return self.work_dir / "stages" / stage

    def stage_extra_dirs(self, stage: str) -> list[Path]:
        main = self.stage_dir(stage)
        return [p for p in self.stage_read_roots.get(stage, []) if p != main]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
