
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from ..config import PipelineConfig
from ..registry import Registry
from ..store import ContainerTable, StageStore


@dataclass(slots=True)
class Context:
    cfg: PipelineConfig
    registry: Registry
    containers: ContainerTable
    licenses: dict[str, int]

    @classmethod
    def create(cls, cfg: PipelineConfig) -> Context:
        registry = Registry.load(cfg.registry_dir)
        containers = ContainerTable.read(cls.container_path(cfg))
        return cls(
            cfg=cfg,
            registry=registry,
            containers=containers,
            licenses=_license_codes(registry),
        )

    def license_code(self, license_id: str) -> int:
        code = self.licenses.get(license_id)
        if code is None:
            raise KeyError(f"license_id {license_id!r} is not declared in the registry")
        return code

    def license_table(self) -> dict[int, str]:
        return {v: k for k, v in self.licenses.items()}

    @staticmethod
    def container_path(cfg: PipelineConfig) -> Path:
        return cfg.work_dir / "container_table.parquet"

    def save_containers(self) -> None:
        self.containers.write(self.container_path(self.cfg))

    def store(self, stage: str, schema: pa.Schema) -> StageStore:
        runtime = self.cfg.runtime
        return StageStore(
            self.cfg.stage_dir(stage),
            schema,
            config_fingerprint=self.cfg.stage_fingerprint(stage),
            compression=runtime.compression,
            compression_level=runtime.compression_level,
            row_group_size=runtime.row_group_size,
            extra_roots=self.cfg.stage_extra_dirs(stage),
        )


def _license_codes(registry: Registry) -> dict[str, int]:
    distinct = {"UNKNOWN"} | {spec.license.id for spec in registry}
    codes = {name: idx for idx, name in enumerate(sorted(distinct))}
    if len(codes) > 32767:
        raise ValueError("license_id values exceed int16 capacity")
    return codes
