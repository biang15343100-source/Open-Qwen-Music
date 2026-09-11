
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
import math
from typing import Any

import numpy as np
import torch

from .contracts import (
    SECTION_LABELS,
    SEMANTIC_FRAMES_PER_MELODY_FRAME,
    SequenceMode,
)
from .registry import TokenRegistry

IGNORE_CONSTRAINT_KIND = -1


class GrammarState(Enum):
    AWAIT_PLAN = auto()
    MELODY_LABEL = auto()
    MELODY_CONTENT = auto()
    AWAIT_MUSIC = auto()
    MUSIC = auto()
    AWAIT_EOS = auto()
    DONE = auto()


class ConstraintKind(IntEnum):

    PLAN_OR_MUSIC = 0
    MELODY_BOS_ONLY = 1
    MUSIC_BOS_ONLY = 2
    SECTION_ONLY = 3
    SECTION_OR_MELODY_EOS = 4
    MELODY_ONLY = 5
    MELODY_OR_SEG_END = 6
    SEG_END_ONLY = 7
    MELODY_EOS_ONLY = 8
    SEMANTIC_ONLY = 9
    SEMANTIC_OR_MUSIC_EOS = 10
    MUSIC_EOS_ONLY = 11
    EOS_ONLY = 12


@dataclass(frozen=True)
class GrammarConfig:
    min_semantic_frames: int = 25
    max_semantic_frames: int = 2250
    max_melody_tokens: int = 512
    max_melody_segments: int = 16
    min_melody_segments: int = 1
    strict_mode: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.min_semantic_frames <= self.max_semantic_frames:
            raise ValueError(
                "GrammarConfig requires 0 <= min_semantic_frames <= max_semantic_frames"
            )
        if self.max_semantic_frames < 1:
            raise ValueError("GrammarConfig max_semantic_frames must be >= 1")
        if self.max_melody_tokens < 0 or self.max_melody_segments < 0:
            raise ValueError("GrammarConfig melody limits cannot be negative")
        if not 0 <= self.min_melody_segments <= self.max_melody_segments:
            raise ValueError(
                "GrammarConfig requires 0 <= min_melody_segments <= max_melody_segments"
            )
        if self.min_melody_segments > self.max_melody_tokens:
            raise ValueError("GrammarConfig melody token limit cannot fit the minimum segment count")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GrammarConfig:
        sequence = dict(config.get("sequence", {}) or {})
        generation = dict(config.get("generation", {}) or {})
        max_semantic = int(sequence.get("max_semantic_frames", 2250))
        max_sequence = int(sequence.get("max_sequence_length", 4096))
        return cls(


            min_semantic_frames=0,
            max_semantic_frames=max_semantic,


            max_melody_tokens=max(
                int(generation.get("max_melody_tokens", 0)),
                math.ceil(max_semantic / SEMANTIC_FRAMES_PER_MELODY_FRAME),
                max_sequence,
            ),
            max_melody_segments=max(
                int(generation.get("max_melody_segments", 0)),
                64,
                max_sequence,
            ),
            min_melody_segments=int(generation.get("min_melody_segments", 1)),
            strict_mode=bool(generation.get("strict_mode", True)),
        )


class AllowedTokenSets:

    def __init__(self, registry: TokenRegistry, device: torch.device) -> None:
        self.registry = registry

        def tensor(ids: list[int]) -> torch.Tensor:
            return torch.tensor(sorted(set(ids)), dtype=torch.long, device=device)

        self.melody_bos = registry.control("melody_bos")
        self.melody_eos = registry.control("melody_eos")
        self.music_bos = registry.control("music_bos")
        self.music_eos = registry.control("music_eos")
        self.seg_end = registry.control("seg_end")
        self.eos = registry.eos_id
        self.section_lookup = {
            registry.section_control(label): label for label in SECTION_LABELS
        }
        section = tensor(list(registry.section_control_ids))
        semantic = torch.arange(
            registry.semantic_base,
            registry.semantic_base + registry.semantic_size,
            dtype=torch.long,
            device=device,
        )
        melody = torch.arange(
            registry.melody_base,
            registry.melody_base + registry.melody_size,
            dtype=torch.long,
            device=device,
        )
        self._ids: dict[ConstraintKind, torch.Tensor] = {
            ConstraintKind.PLAN_OR_MUSIC: tensor([self.melody_bos, self.music_bos]),
            ConstraintKind.MELODY_BOS_ONLY: tensor([self.melody_bos]),
            ConstraintKind.MUSIC_BOS_ONLY: tensor([self.music_bos]),
            ConstraintKind.SECTION_ONLY: section,
            ConstraintKind.SECTION_OR_MELODY_EOS: torch.cat(
                [section, tensor([self.melody_eos])]
            ),
            ConstraintKind.MELODY_ONLY: melody,
            ConstraintKind.MELODY_OR_SEG_END: torch.cat(
                [melody, tensor([self.seg_end])]
            ),
            ConstraintKind.SEG_END_ONLY: tensor([self.seg_end]),
            ConstraintKind.MELODY_EOS_ONLY: tensor([self.melody_eos]),
            ConstraintKind.SEMANTIC_ONLY: semantic,
            ConstraintKind.SEMANTIC_OR_MUSIC_EOS: torch.cat(
                [semantic, tensor([self.music_eos])]
            ),
            ConstraintKind.MUSIC_EOS_ONLY: tensor([self.music_eos]),
            ConstraintKind.EOS_ONLY: tensor([self.eos]),
        }

    def ids_for(self, kind: ConstraintKind | int) -> torch.Tensor:
        return self._ids[ConstraintKind(int(kind))]

    def padded_table(self) -> tuple[torch.Tensor, torch.Tensor]:

        rows = len(ConstraintKind)
        width = max(int(ids.numel()) for ids in self._ids.values())
        table = torch.full(
            (rows, width),
            -1,
            dtype=torch.long,
            device=self.ids_for(ConstraintKind.EOS_ONLY).device,
        )
        lengths = torch.empty(rows, dtype=torch.long, device=table.device)
        for kind in ConstraintKind:
            ids = self.ids_for(kind)
            table[int(kind), : ids.numel()] = ids
            lengths[int(kind)] = ids.numel()
        return table, lengths


@dataclass
class GrammarRowState:
    state: GrammarState
    semantic: list[int] = field(default_factory=list)
    melody: list[int] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    segment_lengths: list[int] = field(default_factory=list)
    segments: int = 0
    current_segment_len: int = 0
    finished: bool = False
    tokens: list[int] = field(default_factory=list)


def initial_state(mode: SequenceMode) -> GrammarState:
    return GrammarState.AWAIT_MUSIC if mode.is_cover else GrammarState.AWAIT_PLAN


def allowed_kind_for_state(
    row: GrammarRowState,
    config: GrammarConfig | Any,
    mode: SequenceMode,
) -> ConstraintKind:
    if row.state is GrammarState.AWAIT_PLAN:
        if not config.strict_mode:
            return ConstraintKind.PLAN_OR_MUSIC
        return (
            ConstraintKind.MELODY_BOS_ONLY
            if mode.has_melody
            else ConstraintKind.MUSIC_BOS_ONLY
        )
    if row.state is GrammarState.MELODY_LABEL:
        if (
            row.segments >= config.max_melody_segments
            or len(row.melody) >= config.max_melody_tokens
        ):
            return ConstraintKind.MELODY_EOS_ONLY
        if config.strict_mode and row.segments < config.min_melody_segments:
            return ConstraintKind.SECTION_ONLY
        return ConstraintKind.SECTION_OR_MELODY_EOS
    if row.state is GrammarState.MELODY_CONTENT:
        if row.current_segment_len == 0:
            return ConstraintKind.MELODY_ONLY
        if len(row.melody) >= config.max_melody_tokens:
            return ConstraintKind.SEG_END_ONLY
        return ConstraintKind.MELODY_OR_SEG_END
    if row.state is GrammarState.AWAIT_MUSIC:
        return ConstraintKind.MUSIC_BOS_ONLY
    if row.state is GrammarState.MUSIC:
        if len(row.semantic) < config.min_semantic_frames:
            return ConstraintKind.SEMANTIC_ONLY
        if len(row.semantic) >= config.max_semantic_frames:
            return ConstraintKind.MUSIC_EOS_ONLY
        return ConstraintKind.SEMANTIC_OR_MUSIC_EOS
    return ConstraintKind.EOS_ONLY


def allowed_ids_for_state(
    row: GrammarRowState,
    sets: AllowedTokenSets,
    config: GrammarConfig | Any,
    mode: SequenceMode,
) -> torch.Tensor:

    return sets.ids_for(allowed_kind_for_state(row, config, mode))


def advance_state(
    row: GrammarRowState,
    token: int,
    sets: AllowedTokenSets,
    registry: TokenRegistry,
) -> None:
    row.tokens.append(token)
    if row.state is GrammarState.AWAIT_PLAN:
        row.state = (
            GrammarState.MELODY_LABEL
            if token == sets.melody_bos
            else GrammarState.MUSIC
        )
        return
    if row.state is GrammarState.MELODY_LABEL:
        if token == sets.melody_eos:
            row.state = GrammarState.AWAIT_MUSIC
        else:
            row.sections.append(sets.section_lookup[token])
            row.current_segment_len = 0
            row.state = GrammarState.MELODY_CONTENT
        return
    if row.state is GrammarState.MELODY_CONTENT:
        if token == sets.seg_end:
            row.segment_lengths.append(row.current_segment_len)
            row.segments += 1
            row.state = GrammarState.MELODY_LABEL
        else:
            row.melody.append(token - registry.melody_base)
            row.current_segment_len += 1
        return
    if row.state is GrammarState.AWAIT_MUSIC:
        row.state = GrammarState.MUSIC
        return
    if row.state is GrammarState.MUSIC:
        if token == sets.music_eos:
            row.state = GrammarState.AWAIT_EOS
        else:
            row.semantic.append(token - registry.semantic_base)
        return
    if row.state is GrammarState.AWAIT_EOS:
        row.state = GrammarState.DONE
        row.finished = True


def token_is_allowed(
    kind: ConstraintKind,
    token: int,
    registry: TokenRegistry,
) -> bool:

    if kind is ConstraintKind.PLAN_OR_MUSIC:
        return token in (
            registry.control("melody_bos"),
            registry.control("music_bos"),
        )
    singleton = {
        ConstraintKind.MELODY_BOS_ONLY: registry.control("melody_bos"),
        ConstraintKind.MUSIC_BOS_ONLY: registry.control("music_bos"),
        ConstraintKind.SEG_END_ONLY: registry.control("seg_end"),
        ConstraintKind.MELODY_EOS_ONLY: registry.control("melody_eos"),
        ConstraintKind.MUSIC_EOS_ONLY: registry.control("music_eos"),
        ConstraintKind.EOS_ONLY: registry.eos_id,
    }
    if kind in singleton:
        return token == singleton[kind]
    if kind in (
        ConstraintKind.SECTION_ONLY,
        ConstraintKind.SECTION_OR_MELODY_EOS,
    ):
        return token in registry.section_control_ids or (
            kind is ConstraintKind.SECTION_OR_MELODY_EOS
            and token == registry.control("melody_eos")
        )
    if kind in (ConstraintKind.MELODY_ONLY, ConstraintKind.MELODY_OR_SEG_END):
        return registry.is_melody(token) or (
            kind is ConstraintKind.MELODY_OR_SEG_END
            and token == registry.control("seg_end")
        )
    if kind in (
        ConstraintKind.SEMANTIC_ONLY,
        ConstraintKind.SEMANTIC_OR_MUSIC_EOS,
    ):
        return registry.is_semantic(token) or (
            kind is ConstraintKind.SEMANTIC_OR_MUSIC_EOS
            and token == registry.control("music_eos")
        )
    return False


def constraint_kinds_for_labels(
    labels: np.ndarray,
    *,
    mode: SequenceMode,
    registry: TokenRegistry,
    config: GrammarConfig,
    ignore_label: int,
    sets: AllowedTokenSets | None = None,
) -> np.ndarray:

    values = np.asarray(labels, dtype=np.int64)
    kinds = np.full(values.shape, IGNORE_CONSTRAINT_KIND, dtype=np.int16)
    supervised = np.flatnonzero(values != ignore_label)
    if supervised.size == 0:
        return kinds
    sets = sets or AllowedTokenSets(registry, torch.device("cpu"))
    row = GrammarRowState(initial_state(mode))
    for position in supervised.tolist():
        token = int(values[position])
        kind = allowed_kind_for_state(row, config, mode)
        if not token_is_allowed(kind, token, registry):
            raise ValueError(
                f"gold token {registry.describe(token)} in state {row.state.name} "
                f"does not belong to the constraint set {kind.name}"
            )
        kinds[position] = int(kind)
        advance_state(row, token, sets, registry)
    if not row.finished:
        raise ValueError(f"Gold sequence did not reach DONE; final state={row.state.name}")
    return kinds
