
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from .condition import (
    CONDITION_TEMPLATE_VERSION,
    ConditionRenderConfig,
    MusicCondition,
    Section,
)
from .contracts import (
    MELODY_FRAME_RATE,
    MELODY_UNVOICED_ID,
    NON_VOCAL_SECTION_LABELS,
    SECTION_LABELS,
    SEMANTIC_FRAME_RATE,
    SEMANTIC_FRAMES_PER_MELODY_FRAME,
    SEQUENCE_PROTOCOL_REVISION,
    SequenceMode,
)
from .grammar import (
    AllowedTokenSets,
    GrammarConfig,
    constraint_kinds_for_labels,
)
from .registry import TokenRegistry

IGNORE_LABEL = -100


class TextEncoder(Protocol):

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]:  # pragma: no cover
        ...


@dataclass
class SequenceConfig:
    max_sequence_length: int = 4096
    max_semantic_frames: int = 2250  # 90 s @ 25 Hz


    min_semantic_frames: int = 100
    max_condition_tokens: int = 1024
    min_section_melody_frames: int = 2


    melody_section_policy: str = "lyrics_only"


    resample_unique_sections: bool = True

    #: masked COND_BOS + SECTION_x + local lyrics + COND_EOS.
    semantic_section_reanchor: bool = False
    semantic_anchor_max_text_tokens: int = 256

    semantic_anchor_boundary_policy: str = "oracle_section_start"
    semantic_anchor_require_complete_plan: bool = False

    random_crop: bool = True

    overflow_policy: str = "truncate"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SequenceConfig:
        section = dict(config.get("sequence", {}) or {})
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(section) - known
        if unknown:
            raise KeyError(f"sequence is configured with unknown fields:{sorted(unknown)}")
        config = cls(**section)
        if config.overflow_policy not in {"truncate", "error"}:
            raise ValueError(
                f"sequence.overflow_policy must be truncate or error; got "
                f"{config.overflow_policy!r}"
            )
        if config.melody_section_policy not in {
            "lyrics_only",
            "lyrics_and_vocal_label",
        }:
            raise ValueError(
                "sequence.melody_section_policy only supports "
                "lyrics_only or lyrics_and_vocal_label; got "
                f"{config.melody_section_policy!r}"
            )
        if config.semantic_anchor_max_text_tokens < 0:
            raise ValueError("semantic_anchor_max_text_tokens required >= 0")
        if config.semantic_anchor_boundary_policy not in {
            "oracle_section_start",
            "melody_plan",
        }:
            raise ValueError(
                "semantic_anchor_boundary_policy only supports "
                "oracle_section_start/melody_plan"
            )
        return config


@dataclass
class BuiltSequence:
    input_ids: np.ndarray
    labels: np.ndarray
    constraint_kinds: np.ndarray
    mode: SequenceMode
    sample_id: str
    num_condition_tokens: int
    num_melody_tokens: int
    num_semantic_tokens: int
    condition_truncated: bool = False
    semantic_cropped: bool = False
    crop_start_frame: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.input_ids.size)


class SequenceTooLongError(ValueError):
    pass


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _validate_section_taxonomy(sections: list[Section]) -> None:
    unknown = sorted({section.label for section in sections if section.label not in SECTION_LABELS})
    if unknown:
        raise ValueError(
            f"Unknown section label: {unknown} (expected {SECTION_LABELS}); "
            "inst must be replaced by annotation adapter Explicit mapping"
        )


def sections_to_melody_spans(
    sections: list[Section],
    *,
    melody_length: int,
) -> list[tuple[Section, int, int]]:
    _validate_section_taxonomy(sections)
    spans: list[tuple[Section, int, int]] = []
    for section in sections:
        timing = section.timing
        if timing.melody_start_frame is not None and timing.melody_end_frame is not None:
            start = int(timing.melody_start_frame)
            end = int(timing.melody_end_frame)
        elif timing.start_sec is not None and timing.end_sec is not None:
            start = int(round(float(timing.start_sec) * MELODY_FRAME_RATE))
            end = int(round(float(timing.end_sec) * MELODY_FRAME_RATE))
        else:
            continue
        start = max(0, min(start, melody_length))
        end = max(start, min(end, melody_length))
        if end > start:
            spans.append((section, start, end))
    return spans


def section_carries_melody(
    section: Section,
    *,
    lyrics_annotated: bool,
    policy: str = "lyrics_only",
) -> bool:
    if policy not in {"lyrics_only", "lyrics_and_vocal_label"}:
        raise ValueError(f"Unknown melody section policy: {policy!r}")
    if lyrics_annotated:
        return section.is_vocal and (
            policy == "lyrics_only" or section.label not in NON_VOCAL_SECTION_LABELS
        )
    return section.label not in NON_VOCAL_SECTION_LABELS


def filter_melody_spans(
    spans: list[tuple[Section, int, int]],
    melody: np.ndarray,
    *,
    lyrics_annotated: bool,
    min_frames: int,
    policy: str = "lyrics_only",
    window: tuple[int, int] | None = None,
) -> list[tuple[Section, int, int]]:
    low, high = window if window is not None else (0, int(melody.size))
    kept: list[tuple[Section, int, int]] = []
    for section, start, end in spans:
        if not section_carries_melody(
            section,
            lyrics_annotated=lyrics_annotated,
            policy=policy,
        ):
            continue
        new_start = max(start, low)
        new_end = min(end, high)
        if new_end - new_start < min_frames:
            continue
        if melody_tokens_are_all_unvoiced(melody[new_start:new_end]):
            continue
        kept.append((section, new_start, new_end))
    return kept


def select_unique_sections(
    spans: list[tuple[Section, int, int]],
    *,
    rng: np.random.Generator | None,
) -> list[tuple[Section, int, int]]:
    grouped: dict[str, list[tuple[Section, int, int]]] = {}
    for item in spans:
        grouped.setdefault(item[0].label, []).append(item)
    chosen: list[tuple[Section, int, int]] = []
    for label in sorted(grouped):
        candidates = grouped[label]
        if rng is None or len(candidates) == 1:


            chosen.append(min(candidates, key=lambda item: item[1]))
        else:
            chosen.append(candidates[int(rng.integers(len(candidates)))])
    chosen.sort(key=lambda item: item[1])
    return chosen


def choose_crop_window(
    *,
    total_frames: int,
    max_frames: int,
    section_start_frames: list[int],
    rng: np.random.Generator | None,
) -> tuple[int, int]:
    if total_frames <= max_frames:
        return 0, total_frames
    latest = total_frames - max_frames
    candidates = [start for start in section_start_frames if 0 <= start <= latest]
    if candidates:
        if rng is None:
            start = candidates[0]
        else:
            start = int(candidates[int(rng.integers(len(candidates)))])
    elif rng is None:
        start = 0
    else:
        start = int(rng.integers(total_frames - max_frames + 1))
    return start, start + max_frames


def _mode_control_name(mode: SequenceMode) -> str:
    if mode is SequenceMode.PLAIN:
        return "mode_plain"
    if mode.is_unique_section:
        return "mode_unique_section"
    return "mode_section"


def _condition_prefix(
    registry: TokenRegistry, condition_ids: list[int], mode: SequenceMode
) -> list[int]:
    return [
        registry.bos_id,
        registry.control("task_t2m"),
        registry.control(_mode_control_name(mode)),
        registry.control("cond_bos"),
        *condition_ids,
        registry.control("cond_eos"),
    ]


def _serialize_melody_blocks(
    registry: TokenRegistry, blocks: list[tuple[Section, np.ndarray]]
) -> list[int]:
    if not blocks:
        raise ValueError("melody block requires at least one valid section")
    ids = [registry.control("melody_bos")]
    seg_end = registry.control("seg_end")
    for section, tokens in blocks:
        ids.append(registry.section_control(section.label))
        ids.extend(registry.melody_to_global(int(token)) for token in tokens)
        ids.append(seg_end)
    ids.append(registry.control("melody_eos"))
    return ids


class SequenceBuilder:

    def __init__(
        self,
        registry: TokenRegistry,
        text_encoder: TextEncoder,
        config: SequenceConfig | None = None,
        condition_config: ConditionRenderConfig | None = None,
        grammar_config: GrammarConfig | None = None,
    ) -> None:
        self.registry = registry
        self.text_encoder = text_encoder
        self.config = config or SequenceConfig()
        self.condition_config = condition_config or ConditionRenderConfig()
        self.grammar_config = grammar_config or GrammarConfig(
            min_semantic_frames=0,
            max_semantic_frames=self.config.max_semantic_frames,
            max_melody_tokens=max(
                1,
                math.ceil(
                    self.config.max_semantic_frames
                    / SEMANTIC_FRAMES_PER_MELODY_FRAME
                ),
            ),
            max_melody_segments=64,
        )
        self._grammar_sets = (
            AllowedTokenSets(registry, torch.device("cpu"))
            if registry is not None
            else None
        )
        self.condition_template_version = self.condition_config.revision
        if self.condition_config == ConditionRenderConfig():
            assert self.condition_template_version == CONDITION_TEMPLATE_VERSION
        self.sequence_protocol_revision = SEQUENCE_PROTOCOL_REVISION


    def encode_condition(self, condition: MusicCondition) -> tuple[list[int], bool]:
        _validate_section_taxonomy(condition.sections)
        budget = self.config.max_condition_tokens
        tag_fields = condition.tag_fields(self.condition_config)
        lyric_lines = condition.render_lyric_lines()

        def render(num_tags: int, num_lines: int) -> str:
            blocks: list[str] = []
            if num_tags:
                blocks.append("[tags] " + " | ".join(tag_fields[:num_tags]))
            if num_lines:
                blocks.append("[lyrics]\n" + "\n".join(lyric_lines[:num_lines]))
            return "\n".join(blocks)

        def encode(text: str) -> list[int]:
            return self.text_encoder.encode(text, add_special_tokens=False) if text else []

        full_ids = encode(render(len(tag_fields), len(lyric_lines)))
        if len(full_ids) <= budget:
            return full_ids, False
        if self.config.overflow_policy == "error":
            raise SequenceTooLongError(
                f"condition has {len(full_ids)} tokens, exceeding max_condition_tokens={budget}; "
                "prepare a shorter manifest window instead of dropping lyrics at runtime"
            )

        def largest_fitting(limit: int, build: Callable[[int], str]) -> tuple[int, list[int]]:
            low, high, best, best_ids = 0, limit, 0, encode(build(0))
            while low <= high:
                mid = (low + high) // 2
                ids = encode(build(mid))
                if len(ids) <= budget:
                    best, best_ids, low = mid, ids, mid + 1
                else:
                    high = mid - 1
            while best > 0 and len(best_ids) > budget:
                best -= 1
                best_ids = encode(build(best))
            return best, best_ids

        num_lines, ids = largest_fitting(len(lyric_lines), lambda n: render(len(tag_fields), n))
        if num_lines == 0 and len(ids) > budget:
            _, ids = largest_fitting(len(tag_fields), lambda n: render(n, 0))
        return ids, True

    def encode_section_anchor(self, section: Section) -> tuple[list[int], bool]:

        text_ids = (
            self.text_encoder.encode(section.lyrics, add_special_tokens=False)
            if section.lyrics
            else []
        )
        limit = self.config.semantic_anchor_max_text_tokens
        truncated = len(text_ids) > limit
        text_ids = text_ids[:limit]
        return (
            [
                self.registry.control("cond_bos"),
                self.registry.section_control(section.label),
                *text_ids,
                self.registry.control("cond_eos"),
            ],
            truncated,
        )

    def section_anchor_pools(
        self, condition: MusicCondition | dict[str, Any]
    ) -> dict[str, list[list[int]]]:

        if not isinstance(condition, MusicCondition):
            condition = MusicCondition.from_record(condition)
        pools: dict[str, list[list[int]]] = {}
        for section in condition.vocal_sections:
            anchor, _ = self.encode_section_anchor(section)
            pools.setdefault(section.label, []).append(anchor)
            pools.setdefault("__ordered__", []).append(anchor)
        return pools

    def oracle_section_anchor_schedule(
        self, condition: MusicCondition | dict[str, Any]
    ) -> list[tuple[int, list[int]]]:

        if not isinstance(condition, MusicCondition):
            condition = MusicCondition.from_record(condition)
        schedule: list[tuple[int, list[int]]] = []
        for section in condition.vocal_sections:
            if section.timing.start_sec is None:
                continue
            boundary = int(
                round(float(section.timing.start_sec) * SEMANTIC_FRAME_RATE)
            )
            anchor, _ = self.encode_section_anchor(section)
            schedule.append((boundary, anchor))
        return schedule


    def build(
        self,
        sample: dict[str, Any],
        *,
        mode: SequenceMode,
        semantic: np.ndarray,
        melody: np.ndarray | None = None,
        epoch: int = 0,
        sample_index: int = 0,
        seed: int = 0,
    ) -> BuiltSequence:
        registry = self.registry
        config = self.config
        sample_id = str(sample.get("sample_id", "unknown"))
        condition = MusicCondition.from_record(sample)

        semantic = np.asarray(semantic).astype(np.int64, copy=False)
        if semantic.ndim != 1:
            raise ValueError(f"semantic must be one-dimensional; got {semantic.shape}")
        melody_array = (
            np.asarray(melody).astype(np.int64, copy=False) if melody is not None else None
        )

        rng: np.random.Generator | None = None
        if config.random_crop or config.resample_unique_sections:
            rng = np.random.default_rng(_stable_seed(seed, epoch, sample_index, sample_id))


        effective_mode = mode
        vocal_sections = condition.vocal_sections
        if condition.is_instrumental or melody_array is None or not vocal_sections:
            effective_mode = SequenceMode.PLAIN

        if (
            config.overflow_policy == "error"
            and int(semantic.size) > config.max_semantic_frames
        ):
            raise SequenceTooLongError(
                f"Sample {sample_id} has {semantic.size} semantic frames, exceeding the "
                f"manifest limit {config.max_semantic_frames}; runtime cropping is disabled"
            )


        section_start_frames = [
            int(round(float(s.timing.start_sec) * SEMANTIC_FRAME_RATE))
            for s in condition.sections
            if s.timing.start_sec is not None
        ]
        crop_start, crop_end = choose_crop_window(
            total_frames=int(semantic.size),
            max_frames=config.max_semantic_frames,
            section_start_frames=section_start_frames,
            rng=rng if config.random_crop else None,
        )
        semantic_cropped = (crop_start, crop_end) != (0, int(semantic.size))
        semantic_window = semantic[crop_start:crop_end]


        melody_blocks: list[tuple[Section, np.ndarray]] = []
        if effective_mode.has_melody and melody_array is not None:
            melody_start = crop_start // SEMANTIC_FRAMES_PER_MELODY_FRAME
            melody_end = -(-crop_end // SEMANTIC_FRAMES_PER_MELODY_FRAME)
            spans = sections_to_melody_spans(
                condition.sections, melody_length=int(melody_array.size)
            )
            clipped = filter_melody_spans(
                spans,
                melody_array,
                lyrics_annotated=condition.has_lyric_annotation,
                min_frames=config.min_section_melody_frames,
                policy=config.melody_section_policy,
                window=(melody_start, melody_end),
            )
            if effective_mode.is_unique_section:
                clipped = select_unique_sections(
                    clipped, rng=rng if config.resample_unique_sections else None
                )
            for section, start, end in clipped:
                melody_blocks.append((section, melody_array[start:end]))
            if not melody_blocks:
                effective_mode = SequenceMode.PLAIN


        condition_ids, condition_truncated = self.encode_condition(condition)
        prefix = _condition_prefix(registry, condition_ids, effective_mode)
        melody_ids = _serialize_melody_blocks(registry, melody_blocks) if melody_blocks else []


        # reference song are inserted into the **prefix** together with the target tags and


        if effective_mode.is_cover:
            prefix.extend(melody_ids)
            generated_melody_ids: list[int] = []
        else:
            generated_melody_ids = melody_ids

        semantic_anchors: dict[int, list[int]] = {}
        anchor_text_truncated = 0
        if (
            config.semantic_section_reanchor
            and effective_mode.has_melody
            and not effective_mode.is_unique_section
        ):
            window_end = crop_start + int(semantic_window.size)
            for section in condition.vocal_sections:
                if (
                    section.timing.start_sec is None
                    or section.timing.end_sec is None
                ):
                    continue
                absolute_start = int(
                    round(float(section.timing.start_sec) * SEMANTIC_FRAME_RATE)
                )
                absolute_end = int(
                    round(float(section.timing.end_sec) * SEMANTIC_FRAME_RATE)
                )
                if absolute_end <= crop_start or absolute_start >= window_end:
                    continue
                boundary = max(0, absolute_start - crop_start)
                anchor_ids, truncated = self.encode_section_anchor(section)
                semantic_anchors.setdefault(boundary, []).extend(anchor_ids)
                anchor_text_truncated += int(truncated)

        music_bos = registry.control("music_bos")
        music_eos = registry.control("music_eos")
        overhead = 3  # music_bos + music_eos + eos
        anchor_tokens = sum(len(ids) for ids in semantic_anchors.values())
        available = (
            config.max_sequence_length
            - len(prefix)
            - len(generated_melody_ids)
            - overhead
            - anchor_tokens
        )
        if available < config.min_semantic_frames:
            raise SequenceTooLongError(
                f"Sample {sample_id}:condition {len(prefix)} + melody {len(generated_melody_ids)} tokens "
                f"only remaining {available} locations,is lower than min_semantic_frames="
                f"{config.min_semantic_frames}(max_sequence_length="
                f"{config.max_sequence_length})"
            )
        num_semantic = min(int(semantic_window.size), available)
        if num_semantic < int(semantic_window.size):
            if config.overflow_policy == "error":
                raise SequenceTooLongError(
                    f"Sample {sample_id} sequence budget can only accommodate {num_semantic}/"
                    f"{semantic_window.size} semantic frames; manifest conditions and window differ"
                )
            semantic_cropped = True
        semantic_window = semantic_window[:num_semantic]
        semantic_anchors = {
            position: ids
            for position, ids in semantic_anchors.items()
            if position < num_semantic
        }
        anchor_tokens = sum(len(ids) for ids in semantic_anchors.values())

        body: list[int] = [*generated_melody_ids, music_bos]
        body_labels: list[int] = list(body)
        for position, token in enumerate(semantic_window):
            anchor_ids = semantic_anchors.get(position, ())
            body.extend(anchor_ids)
            body_labels.extend([IGNORE_LABEL] * len(anchor_ids))
            semantic_id = int(registry.semantic_base + int(token))
            body.append(semantic_id)
            body_labels.append(semantic_id)
        body.extend([music_eos, registry.eos_id])
        body_labels.extend([music_eos, registry.eos_id])

        input_ids = np.asarray(prefix + body, dtype=np.int64)
        labels = np.full_like(input_ids, IGNORE_LABEL)
        labels[len(prefix) :] = np.asarray(body_labels, dtype=np.int64)
        constraint_kinds = constraint_kinds_for_labels(
            labels,
            mode=effective_mode,
            registry=registry,
            config=self.grammar_config,
            ignore_label=IGNORE_LABEL,
            sets=self._grammar_sets,
        )

        return BuiltSequence(
            input_ids=input_ids,
            labels=labels,
            constraint_kinds=constraint_kinds,
            mode=effective_mode,
            sample_id=sample_id,
            num_condition_tokens=len(condition_ids),
            num_melody_tokens=len(generated_melody_ids),
            num_semantic_tokens=num_semantic,
            condition_truncated=condition_truncated,
            semantic_cropped=semantic_cropped,
            crop_start_frame=crop_start,
            diagnostics={
                "requested_mode": mode.value,
                "num_melody_blocks": len(melody_blocks),
                "prefix_len": len(prefix),
                "num_reference_melody_tokens": len(melody_ids) if effective_mode.is_cover else 0,
                "semantic_anchor_sections": len(semantic_anchors),
                "semantic_anchor_tokens": anchor_tokens,
                "semantic_anchor_text_truncated": anchor_text_truncated,
            },
        )


    def build_prompt(
        self,
        condition: MusicCondition | dict[str, Any],
        *,
        mode: SequenceMode,
        reference_melody: np.ndarray | None = None,
        reference_sections: list[Section] | None = None,
    ) -> np.ndarray:
        if isinstance(condition, dict):
            condition = MusicCondition.from_record(condition)
        if mode.has_melody and (
            condition.is_instrumental or not condition.vocal_sections
        ):
            raise ValueError(
                f"{mode.value} requires target conditions with vocal lyrics; "
                "request plain mode explicitly for instrumental or lyric-free samples"
            )
        registry = self.registry
        condition_ids, _ = self.encode_condition(condition)
        ids = _condition_prefix(registry, condition_ids, mode)
        if mode.is_cover:
            if reference_melody is None:
                raise ValueError("cover mode must provide reference_melody")
            melody_array = np.asarray(reference_melody).astype(np.int64, copy=False)
            if melody_array.ndim != 1:
                raise ValueError(
                    f"reference_melody must be one-dimensional; got {melody_array.shape}"
                )
            if not reference_sections:
                raise ValueError("cover pattern must provide non-empty reference_sections")
            sections = reference_sections
            spans = sections_to_melody_spans(sections, melody_length=int(melody_array.size))


            spans = filter_melody_spans(
                spans,
                melody_array,
                lyrics_annotated=any(s.is_vocal for s in sections),
                min_frames=self.config.min_section_melody_frames,
                policy=self.config.melody_section_policy,
            )
            if mode.is_unique_section:
                spans = select_unique_sections(spans, rng=None)
            if not spans:
                raise ValueError("cover reference has no lyric-bearing melody section")
            blocks = [(section, melody_array[start:end]) for section, start, end in spans]
            ids.extend(_serialize_melody_blocks(registry, blocks))
        return np.asarray(ids, dtype=np.int64)


def melody_tokens_are_all_unvoiced(melody: np.ndarray) -> bool:
    array = np.asarray(melody)
    return bool(array.size == 0 or np.all(array == MELODY_UNVOICED_ID))
