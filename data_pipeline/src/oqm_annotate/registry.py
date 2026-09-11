
from __future__ import annotations

from .stages.audio import SeparateStage
from .stages.base import Stage, StageContext
from .stages.emit import EmitStage
from .stages.index import IndexStage
from .stages.lyrics import AlignStage, AsrMixStage, AsrVocalStage, LyricsFusionStage
from .stages.sections import SectionsStage
from .stages.structure import StructureStage
from .stages.tags import TagsFuseStage, TagsLlmStage, VoiceAcousticStage

STAGE_CLASSES: tuple[type[Stage], ...] = (
    IndexStage,
    SeparateStage,
    StructureStage,
    AsrVocalStage,
    AsrMixStage,
    LyricsFusionStage,
    AlignStage,
    SectionsStage,
    VoiceAcousticStage,
    TagsLlmStage,
    TagsFuseStage,
    EmitStage,
)

STAGES: dict[str, type[Stage]] = {cls.name: cls for cls in STAGE_CLASSES}


CPU_STAGES: frozenset[str] = frozenset(
    {"index", "lyrics", "sections", "tags.fuse", "emit"}
)

GPU_STAGES: frozenset[str] = frozenset(STAGES) - CPU_STAGES


def topological_order(names: list[str]) -> list[str]:

    unknown = [name for name in names if name not in STAGES]
    if unknown:
        raise KeyError(f"Unknown stages {unknown}; available stages: {sorted(STAGES)}")

    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()
    selected = set(names)

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise ValueError(f"Stage dependency cycle detected at {name}")
        visiting.add(name)
        for dependency in STAGES[name].depends_on:

            if dependency in selected:
                visit(dependency)
        visiting.discard(name)
        visited.add(name)
        ordered.append(name)


    for name in STAGE_CLASSES:
        if name.name in selected:
            visit(name.name)
    return ordered


def expand_with_dependencies(names: list[str]) -> list[str]:

    result: set[str] = set()
    queue = list(names)
    while queue:
        name = queue.pop()
        if name in result:
            continue
        if name not in STAGES:
            raise KeyError(f"Unknown stage:{name}")
        result.add(name)
        queue.extend(STAGES[name].depends_on)
    return topological_order(sorted(result))


def build(name: str, context: StageContext) -> Stage:
    return STAGES[name](context)
