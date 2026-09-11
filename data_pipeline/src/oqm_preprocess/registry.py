
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import enums

StorageClass = Literal["loose", "zip", "tar", "targz", "parquet", "sqlite", "external"]

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_RE.sub(lambda match: os.environ.get(match.group(1), match.group(2) or ""), value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


class SourceSpec(BaseModel):

    model_config = ConfigDict(extra="forbid")

    storage_class: StorageClass


    root: Path | None = None
    paths: list[Path] = Field(default_factory=list)
    glob: str | None = None
    audio_only: bool = True

    audio_column: str | None = None
    meta_columns: list[str] = Field(default_factory=list)

    table: str | None = None
    id_column: str | None = None
    blob_column: str | None = None

    include_regex: list[str] = Field(default_factory=list)
    exclude_regex: list[str] = Field(default_factory=list)

    max_containers: int = 0


    password: str | None = None
    password_env: str | None = None

    @model_validator(mode="after")
    def _check(self) -> SourceSpec:
        if not self.paths and not (self.root and self.glob) and not self.root:
            raise ValueError("source must define paths or root with an optional glob")
        if self.storage_class == "sqlite" and not (self.table and self.id_column and self.blob_column):
            raise ValueError("sqlite source must define table, id_column, and blob_column")
        if self.storage_class == "loose" and not self.root:
            raise ValueError("loose source must define a root directory")
        if (self.password or self.password_env) and self.storage_class != "zip":
            raise ValueError(f"password is supported only for zip sources, not {self.storage_class}")
        return self

    def secret(self) -> bytes | None:
        if self.password_env:
            import os

            got = os.environ.get(self.password_env)
            if not got:
                raise ValueError(
                    f"source declares password_env={self.password_env}, but the variable is empty")
            return got.encode()
        return self.password.encode() if self.password else None

    @property
    def member_glob(self) -> str | None:
        return self.glob if self.storage_class == "loose" else None

    def containers(self) -> list[Path]:
        found: list[Path] = []
        for p in self.paths:
            found.append(Path(p))
        if self.storage_class == "loose":

            found.append(Path(str(self.root)))
        elif self.root and self.glob:
            found.extend(sorted(Path(self.root).glob(self.glob)))
        elif self.root and not self.paths:
            found.append(Path(self.root))
        seen: set[str] = set()
        unique: list[Path] = []
        for p in found:
            key = str(p)
            if key not in seen:
                seen.add(key)
                unique.append(p)
        unique.sort(key=str)
        if self.max_containers > 0:
            unique = unique[: self.max_containers]
        return unique


class LicenseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = "UNKNOWN"
    family: str = "unknown"
    commercial_ok: bool = False

    @field_validator("family")
    @classmethod
    def _known_family(cls, v: str) -> str:
        if not enums.LICENSE_FAMILY.has(v):
            raise ValueError(f"Unknown license family {v!r}; expected one of {list(enums.LICENSE_FAMILY)}")
        return v


class LanguageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: str = "unknown"
    evidence: str = "no_evidence"
    confidence: str = "low"

    @model_validator(mode="after")
    def _known(self) -> LanguageSpec:
        if not enums.LANGUAGE.has(self.default):
            raise ValueError(f"Unknown language {self.default!r}")
        if not enums.LANGUAGE_EVIDENCE.has(self.evidence):
            raise ValueError(f"Unknown language evidence {self.evidence!r}")
        if not enums.CONFIDENCE.has(self.confidence):
            raise ValueError(f"Unknown confidence {self.confidence!r}")
        return self


class JoinSpec(BaseModel):

    model_config = ConfigDict(extra="forbid")


    key_field: str

    match: Literal["member", "local_id", "stem", "basename", "basename_stem"] = "member"


    member_pattern: str = ""

    strip_prefix: str = ""
    strip_suffix: str = ""
    lower: bool = False

    @model_validator(mode="after")
    def _check_pattern(self) -> JoinSpec:
        if self.member_pattern:
            try:
                compiled = re.compile(self.member_pattern)
            except re.error as exc:
                raise ValueError(f"member_pattern is not a legal regular expression {self.member_pattern!r}: {exc}") from exc
            if compiled.groups < 1:
                raise ValueError(f"member_pattern must have a capturing group: {self.member_pattern!r}")
        return self


TEXTDIR_CONTENT = "content"


class EnrichSpec(BaseModel):

    model_config = ConfigDict(extra="forbid")

    format: Literal["jsonl", "csv", "tsv", "parquet", "sqlite", "textdir",
                    "inline_parquet", "inline_sqlite"]
    root: Path | None = None
    paths: list[Path] = Field(default_factory=list)
    glob: str | None = None
    join: JoinSpec | None = None

    table: str | None = None

    member_glob: str = "*.txt"
    exclude_regex: list[str] = Field(default_factory=list)


    fields: dict[str, str] = Field(default_factory=dict)

    external_ids: list[str] = Field(default_factory=list)

    group_key_fields: list[str] = Field(default_factory=list)

    split_field: str | None = None
    split_value_map: dict[str, str] = Field(default_factory=dict)


    expected_join_rate: float | None = None
    expected_join_note: str = ""

    lyrics_format: str = "plain"
    lyrics_source: str = "official"
    transcript_source: str = "human_annotation"
    caption_source: str = "human_annotation"

    lyrics_is_prompt: bool = False

    keep_raw: bool = False

    @model_validator(mode="after")
    def _check(self) -> EnrichSpec:
        inline = self.format.startswith("inline_")
        if not inline and not (self.paths or (self.root and self.glob) or self.root):
            raise ValueError(f"{self.format} must be given paths or root(+glob)")
        if not inline and self.join is None:
            raise ValueError(f"{self.format} must be given join")
        if self.format == "sqlite" and not self.table:
            raise ValueError("sqlite bypass must give table")
        if self.format == "textdir":

            allowed = {"lyrics_text", "transcript_text", "caption_text", "prompt_text"}
            bad = set(self.fields) - allowed
            if bad:
                raise ValueError(f"textdir supports only {sorted(allowed)}; got {sorted(bad)}")
            if set(self.fields.values()) - {TEXTDIR_CONTENT}:
                raise ValueError(f"textdir can only be {TEXTDIR_CONTENT!r}")
            for pattern in self.exclude_regex:
                re.compile(pattern)
        for target in self.fields:
            if target not in _ENRICH_TARGETS:
                raise ValueError(f"Unknown enrich target {target!r}; expected one of {sorted(_ENRICH_TARGETS)}")
        if not enums.LYRICS_FORMAT.has(self.lyrics_format):
            raise ValueError(f"Unknown lyrics_format {self.lyrics_format!r}")
        for name, book in (
            (self.lyrics_source, enums.TEXT_SOURCE),
            (self.transcript_source, enums.TEXT_SOURCE),
            (self.caption_source, enums.TEXT_SOURCE),
        ):
            if not book.has(name):
                raise ValueError(f"Unknown text source {name!r}")
        return self

    def sidecar_paths(self) -> list[Path]:
        found = [Path(p) for p in self.paths]
        if self.root and self.glob:
            found.extend(sorted(Path(self.root).glob(self.glob)))
        elif self.root and not self.paths:
            found.append(Path(self.root))
        return sorted({str(p): p for p in found}.values(), key=str)


_ENRICH_TARGETS = frozenset({
    "lyrics_text", "transcript_text", "caption_text", "prompt_text",
    "language", "title", "artist", "album", "license_id",
    "declared_duration_sec", "content_type", "stem_role",
})


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: int
    slug: str
    name: str
    group: str = ""

    domain: str = "music"
    content_type: str = "music_mix"
    granularity: str = "whole_song"
    stem_role: str = "unknown"
    is_synthetic: bool = False
    synthetic_model: str = "unknown"
    is_derived: bool = False

    language: LanguageSpec = Field(default_factory=LanguageSpec)
    license: LicenseSpec = Field(default_factory=LicenseSpec)


    sources: list[SourceSpec] = Field(default_factory=list)
    adapter: str | None = None
    enrich: list[EnrichSpec] = Field(default_factory=list)

    path_fields: dict[str, str] = Field(default_factory=dict)

    group_key_template: str | None = None

    expected_item_count: int | None = None
    expected_tolerance: float = 0.02

    expected_note: str = ""


    enabled: bool = True
    disabled_reason: str = ""


    priority: int = 50

    holdout: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _known_enums(self) -> DatasetSpec:
        checks = (
            (enums.DOMAIN, self.domain, "domain"),
            (enums.CONTENT_TYPE, self.content_type, "content_type"),
            (enums.GRANULARITY, self.granularity, "granularity"),
            (enums.STEM_ROLE, self.stem_role, "stem_role"),
            (enums.SYNTHETIC_MODEL, self.synthetic_model, "synthetic_model"),
        )
        for book, value, label in checks:
            if value != "unknown" and not book.has(value):
                raise ValueError(f"Unknown {label} {value!r}; expected one of {list(book)}")
        if self.dataset_id <= 0:
            raise ValueError("dataset_id must be positive")
        if not self.slug or " " in self.slug:
            raise ValueError(f"Invalid slug {self.slug!r}")
        return self

    @property
    def metadata_only(self) -> bool:
        return not self.sources

    def fingerprint(self, *, scope: str = "source") -> str:
        payload = self.model_dump(mode="json")
        for key in ("notes", "priority", "holdout", "enabled", "disabled_reason",
                    "expected_item_count", "expected_tolerance", "expected_note"):
            payload.pop(key, None)
        if scope == "source":
            payload.pop("enrich", None)
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha1(blob.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]

    def count_ok(self, actual: int) -> bool:
        if self.expected_item_count is None:
            return True
        expected = self.expected_item_count
        if expected <= 0:
            return actual == 0
        return abs(actual - expected) <= max(1, expected * self.expected_tolerance)


def _reject_prefix_collisions(by_slug: dict[str, DatasetSpec]) -> None:
    slugs = sorted(by_slug)
    for slug in slugs:
        head = f"{slug}-"
        clashes = [other for other in slugs if other != slug and other.startswith(head)]
        if clashes:
            raise ValueError(
                f"slug {slug!r} is a prefix of {clashes!r}; rename one slug to avoid "
                "overlapping shard cleanup patterns"
            )


class Registry:

    def __init__(self, specs: list[DatasetSpec]) -> None:
        by_id: dict[int, DatasetSpec] = {}
        by_slug: dict[str, DatasetSpec] = {}
        for spec in specs:
            if spec.dataset_id in by_id:
                raise ValueError(f"dataset_id {spec.dataset_id} Repeat")
            if spec.slug in by_slug:
                raise ValueError(f"slug {spec.slug!r} Repeat")
            by_id[spec.dataset_id] = spec
            by_slug[spec.slug] = spec
        _reject_prefix_collisions(by_slug)
        self._by_id = by_id
        self._by_slug = by_slug

    @classmethod
    def load(cls, directory: Path) -> Registry:
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"registry directory does not exist: {directory}")
        specs: list[DatasetSpec] = []
        for path in sorted(directory.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if raw is None:
                raise ValueError(f"registry The file is empty: {path}")
            try:
                specs.append(DatasetSpec.model_validate(_expand_env(raw)))
            except Exception as exc:
                raise ValueError(f"Invalid registry file {path}: {exc}") from exc
        if not specs:
            raise ValueError(f"registry does not have yaml: {directory}")
        return cls(specs)

    def get(self, key: int | str) -> DatasetSpec:
        spec = self._by_id.get(key) if isinstance(key, int) else self._by_slug.get(key)
        if spec is None:
            raise KeyError(f"Registry has no dataset {key!r}")
        return spec

    def select(self, keys: list[str] | None, *, include_disabled: bool = False) -> list[DatasetSpec]:
        if not keys or keys == ["all"]:
            chosen = [s for s in self._by_id.values() if include_disabled or s.enabled]
        else:
            chosen = []
            for key in keys:
                spec = self.get(int(key) if key.isdigit() else key)
                if not spec.enabled and not include_disabled:
                    reason = spec.disabled_reason or "No reason provided"
                    raise ValueError(
                        f"Dataset {spec.slug} is disabled: {reason}. Set `enabled: true` "
                        "in the registry to include it."
                    )
                chosen.append(spec)
        chosen.sort(key=lambda s: (-s.priority, s.dataset_id))
        return chosen

    @property
    def enabled_ids(self) -> set[int]:
        return {s.dataset_id for s in self._by_id.values() if s.enabled}

    @property
    def disabled(self) -> list[DatasetSpec]:
        return sorted((s for s in self._by_id.values() if not s.enabled),
                      key=lambda s: s.dataset_id)

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[DatasetSpec]:
        return iter(sorted(self._by_id.values(), key=lambda s: s.dataset_id))

    def as_rows(self) -> list[dict[str, Any]]:
        return [s.model_dump(mode="json") for s in self]
