
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .contracts import MELODY_UNVOICED_ID, SECTION_LABELS, SequenceMode
from .grammar import (
    AllowedTokenSets,
    GrammarRowState,
    GrammarState,
    advance_state,
    allowed_ids_for_state,
)
from .model import MusicLLM
from .registry import TokenRegistry


_State = GrammarState
_AllowedSets = AllowedTokenSets
_RowState = GrammarRowState
_allowed_for = allowed_ids_for_state
_advance = advance_state


@dataclass
class GenerationConfig:
    max_new_tokens: int = 3000
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 0
    min_semantic_frames: int = 25
    max_semantic_frames: int = 2250
    max_melody_tokens: int = 512
    max_melody_segments: int = 16

    semantic_repetition_penalty: float = 0.0
    semantic_repetition_window: int = 64

    plain_semantic_repetition_penalty: float = 0.0

    melody_repetition_penalty: float = 0.0
    melody_repetition_window: int = 32


    melody_unvoiced_run_threshold: int = 0
    melody_unvoiced_run_penalty: float = 0.0


    plain_semantic_eos_bias: float = 0.0
    plain_semantic_eos_bias_start_frames: int = 0
    plain_semantic_eos_bias_interval_frames: int = 625

    #:


    min_melody_segments: int = 1
    seed: int = 0

    strict_mode: bool = True

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens required >= 1")
        if not math.isfinite(float(self.temperature)) or self.temperature < 0:
            raise ValueError("temperature must be finite and >= 0")
        if not 0.0 < float(self.top_p) <= 1.0:
            raise ValueError("top_p must be within (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k required >= 0")
        if not 0 <= self.min_semantic_frames <= self.max_semantic_frames:
            raise ValueError(
                "must satisfy 0 <= min_semantic_frames <= max_semantic_frames"
            )
        if self.max_semantic_frames < 1:
            raise ValueError("max_semantic_frames required >= 1")
        if self.max_melody_tokens < 0 or self.max_melody_segments < 0:
            raise ValueError("Melody token and segment limits cannot be negative")
        if not 0 <= self.min_melody_segments <= self.max_melody_segments:
            raise ValueError(
                "must satisfy 0 <= min_melody_segments <= max_melody_segments"
            )
        if self.min_melody_segments > self.max_melody_tokens:
            raise ValueError("Each melody segment requires at least one token; token limit is too small")
        if self.semantic_repetition_penalty < 0:
            raise ValueError("semantic_repetition_penalty cannot be negative")
        if self.semantic_repetition_window < 1:
            raise ValueError("semantic_repetition_window required >= 1")
        if self.plain_semantic_repetition_penalty < 0:
            raise ValueError("plain_semantic_repetition_penalty cannot be negative")
        if self.melody_repetition_penalty < 0:
            raise ValueError("melody_repetition_penalty cannot be negative")
        if self.melody_repetition_window < 1:
            raise ValueError("melody_repetition_window required >= 1")
        if self.melody_unvoiced_run_threshold < 0:
            raise ValueError("melody_unvoiced_run_threshold cannot be negative")
        if self.melody_unvoiced_run_penalty < 0:
            raise ValueError("melody_unvoiced_run_penalty cannot be negative")
        if self.plain_semantic_eos_bias < 0:
            raise ValueError("plain_semantic_eos_bias cannot be negative")
        if self.plain_semantic_eos_bias_start_frames < 0:
            raise ValueError("plain_semantic_eos_bias_start_frames cannot be negative")
        if self.plain_semantic_eos_bias_interval_frames < 1:
            raise ValueError("plain_semantic_eos_bias_interval_frames required >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_semantic_frames": self.min_semantic_frames,
            "max_semantic_frames": self.max_semantic_frames,
            "max_melody_tokens": self.max_melody_tokens,
            "max_melody_segments": self.max_melody_segments,
            "min_melody_segments": self.min_melody_segments,
            "semantic_repetition_penalty": self.semantic_repetition_penalty,
            "semantic_repetition_window": self.semantic_repetition_window,
            "plain_semantic_repetition_penalty": (
                self.plain_semantic_repetition_penalty
            ),
            "melody_repetition_penalty": self.melody_repetition_penalty,
            "melody_repetition_window": self.melody_repetition_window,
            "melody_unvoiced_run_threshold": self.melody_unvoiced_run_threshold,
            "melody_unvoiced_run_penalty": self.melody_unvoiced_run_penalty,
            "plain_semantic_eos_bias": self.plain_semantic_eos_bias,
            "plain_semantic_eos_bias_start_frames": (
                self.plain_semantic_eos_bias_start_frames
            ),
            "plain_semantic_eos_bias_interval_frames": (
                self.plain_semantic_eos_bias_interval_frames
            ),
            "seed": self.seed,
            "strict_mode": self.strict_mode,
        }


@dataclass
class GenerationOutput:
    semantic_ids: list[np.ndarray]
    generated_melody_ids: list[np.ndarray | None]
    melody_sections: list[list[str]]
    melody_segment_lengths: list[list[int]]
    semantic_anchor_tokens: list[int]
    modes: list[str]
    requested_modes: list[str]
    finished: list[bool]
    stop_reasons: list[str]
    tokenizer_revision: str
    registry_revision: str
    generation_config: dict[str, Any]
    raw_token_ids: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.semantic_ids)


def _sample_rows(
    logits: torch.Tensor,
    allowed_per_row: list[torch.Tensor],
    config: GenerationConfig,
    generators: list[torch.Generator],
) -> list[int]:
    masked = torch.full_like(logits, float("-inf"))
    if len(generators) != logits.shape[0]:
        raise ValueError("Every sample row must have an independent torch.Generator")
    for row_index, allowed in enumerate(allowed_per_row):
        masked[row_index, allowed] = logits[row_index, allowed]
    if config.temperature <= 0:
        return masked.argmax(dim=-1).tolist()

    probabilities = torch.softmax(masked / config.temperature, dim=-1)
    if config.top_k and config.top_k > 0:
        k = min(int(config.top_k), probabilities.shape[-1])
        values, indices = torch.topk(probabilities, k, dim=-1)
        probabilities = torch.zeros_like(probabilities).scatter_(-1, indices, values)
    if 0.0 < config.top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probabilities, dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)


        sorted_probs = sorted_probs.masked_fill(cumulative - sorted_probs > config.top_p, 0.0)
        probabilities = torch.zeros_like(probabilities).scatter_(-1, sorted_indices, sorted_probs)

    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    sampled = [
        torch.multinomial(probabilities[row], 1, generator=generators[row])
        for row in range(probabilities.shape[0])
    ]
    return torch.cat(sampled).tolist()


def _apply_semantic_repetition_penalty(
    logits: torch.Tensor,
    rows: list[_RowState],
    *,
    registry: TokenRegistry,
    penalty: float,
    window: int,
) -> None:

    if penalty <= 0.0:
        return
    if window <= 0:
        raise ValueError("semantic_repetition_window required > 0")
    for row_index, row in enumerate(rows):
        if row.finished or row.state is not _State.MUSIC or not row.semantic:
            continue
        recent = torch.as_tensor(
            row.semantic[-int(window) :],
            dtype=torch.long,
            device=logits.device,
        )
        local_ids, counts = torch.unique(recent, return_counts=True)
        global_ids = local_ids + registry.semantic_base

        logits[row_index, global_ids] -= (
            counts.clamp_max(8).to(logits.dtype) * float(penalty)
        )


def _semantic_repetition_penalty_for_mode(
    config: GenerationConfig,
    mode: SequenceMode,
) -> float:
    return float(config.semantic_repetition_penalty) + (
        float(config.plain_semantic_repetition_penalty)
        if mode is SequenceMode.PLAIN
        else 0.0
    )


def _apply_melody_repetition_penalty(
    logits: torch.Tensor,
    rows: list[_RowState],
    *,
    registry: TokenRegistry,
    repetition_penalty: float,
    repetition_window: int,
    unvoiced_run_threshold: int,
    unvoiced_run_penalty: float,
) -> None:

    if repetition_penalty <= 0.0 and (
        unvoiced_run_threshold <= 0 or unvoiced_run_penalty <= 0.0
    ):
        return
    if repetition_window <= 0:
        raise ValueError("melody_repetition_window required > 0")
    for row_index, row in enumerate(rows):
        if (
            row.finished
            or row.state is not _State.MELODY_CONTENT
            or row.current_segment_len <= 0
            or not row.melody
        ):
            continue
        current_segment = row.melody[-int(row.current_segment_len) :]
        if repetition_penalty > 0.0:
            recent = torch.as_tensor(
                current_segment[-int(repetition_window) :],
                dtype=torch.long,
                device=logits.device,
            )
            local_ids, counts = torch.unique(recent, return_counts=True)
            global_ids = local_ids + registry.melody_base
            logits[row_index, global_ids] -= (
                counts.clamp_max(8).to(logits.dtype) * float(repetition_penalty)
            )
        if unvoiced_run_threshold > 0 and unvoiced_run_penalty > 0.0:
            run = 0
            for token in reversed(current_segment):
                if int(token) != MELODY_UNVOICED_ID:
                    break
                run += 1
            excess = max(run - int(unvoiced_run_threshold) + 1, 0)
            if excess:
                logits[
                    row_index,
                    registry.melody_base + MELODY_UNVOICED_ID,
                ] -= min(excess, 8) * float(unvoiced_run_penalty)


def _apply_plain_semantic_eos_bias(
    logits: torch.Tensor,
    rows: list[_RowState],
    *,
    registry: TokenRegistry,
    mode: SequenceMode,
    bias: float,
    start_frames: int,
    interval_frames: int,
) -> None:

    if mode is not SequenceMode.PLAIN or bias <= 0.0:
        return
    if start_frames < 0 or interval_frames <= 0:
        raise ValueError("Plain semantic EOS bias length is invalid")
    music_eos = registry.control("music_eos")
    for row_index, row in enumerate(rows):
        if row.finished or row.state is not _State.MUSIC:
            continue
        frames = len(row.semantic)
        if frames < start_frames:
            continue
        stages = 1 + (frames - start_frames) // interval_frames
        logits[row_index, music_eos] += min(stages, 8) * float(bias)


@torch.no_grad()
def generate_semantic(
    model: MusicLLM,
    prompts: list[np.ndarray],
    *,
    mode: SequenceMode,
    config: GenerationConfig | None = None,
    tokenizer_revision: str = "unknown",
    device: torch.device | None = None,
    semantic_anchor_schedules: list[
        list[tuple[int, list[int] | np.ndarray]]
    ]
    | None = None,
    semantic_anchor_pools: list[
        dict[str, list[list[int] | np.ndarray]]
    ]
    | None = None,
    row_seeds: list[int] | None = None,
) -> GenerationOutput:
    from transformers import DynamicCache

    config = config or GenerationConfig()
    registry = model.registry
    device = device or next(model.parameters()).device
    model.eval()
    sets = _AllowedSets(registry, device)
    batch_size = len(prompts)
    if batch_size == 0:
        raise ValueError("prompts is empty")
    if semantic_anchor_schedules is not None and len(semantic_anchor_schedules) != batch_size:
        raise ValueError("semantic_anchor_schedules and prompts have different lengths")
    if semantic_anchor_pools is not None and len(semantic_anchor_pools) != batch_size:
        raise ValueError("semantic_anchor_pools and prompts have different lengths")
    if row_seeds is not None and len(row_seeds) != batch_size:
        raise ValueError("row_seeds and prompts have different lengths")
    expected_prompt_end = registry.control("melody_eos" if mode.is_cover else "cond_eos")
    for index, prompt in enumerate(prompts):
        if prompt.size == 0 or int(prompt[-1]) != expected_prompt_end:
            expected_name = "melody_eos" if mode.is_cover else "cond_eos"
            raise ValueError(f"prompt[{index}] must end with {expected_name}")
    lengths = [int(p.size) for p in prompts]
    max_length = max(lengths)
    pad_id = registry.pad_id

    input_ids = torch.full((batch_size, max_length), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
    for row, prompt in enumerate(prompts):
        length = int(prompt.size)
        input_ids[row, max_length - length :] = torch.as_tensor(
            np.asarray(prompt, dtype=np.int64), device=device
        )
        attention_mask[row, max_length - length :] = 1

    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)


    initial_state = _State.AWAIT_MUSIC if mode.is_cover else _State.AWAIT_PLAN
    rows = [_RowState(initial_state) for _ in range(batch_size)]
    anchor_schedules: list[dict[int, list[int]]] = []
    for row_index in range(batch_size):
        schedule: dict[int, list[int]] = {}
        if semantic_anchor_schedules is not None:
            for boundary, tokens in semantic_anchor_schedules[row_index]:
                boundary = int(boundary)
                if boundary < 0:
                    raise ValueError("semantic anchor boundary cannot be negative")
                schedule.setdefault(boundary, []).extend(
                    int(token) for token in np.asarray(tokens).reshape(-1)
                )
        anchor_schedules.append(schedule)
    anchor_pending: list[list[int]] = [[] for _ in range(batch_size)]
    inserted_anchor_tokens = [0] * batch_size

    seeds = (
        [int(config.seed) + index for index in range(batch_size)]
        if row_seeds is None
        else [int(value) for value in row_seeds]
    )
    generators = [
        torch.Generator(device=device).manual_seed(seed)
        for seed in seeds
    ]
    cache = DynamicCache()
    outputs = model.transformer_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    )
    hidden = outputs.last_hidden_state[:, -1, :]
    next_position = position_ids[:, -1] + 1

    generated_steps = [0] * batch_size
    while True:
        logits = model.lm_head(hidden.to(model.output_dtype)).float()
        _apply_semantic_repetition_penalty(
            logits,
            rows,
            registry=registry,
            penalty=_semantic_repetition_penalty_for_mode(config, mode),
            window=int(config.semantic_repetition_window),
        )
        _apply_melody_repetition_penalty(
            logits,
            rows,
            registry=registry,
            repetition_penalty=float(config.melody_repetition_penalty),
            repetition_window=int(config.melody_repetition_window),
            unvoiced_run_threshold=int(config.melody_unvoiced_run_threshold),
            unvoiced_run_penalty=float(config.melody_unvoiced_run_penalty),
        )
        _apply_plain_semantic_eos_bias(
            logits,
            rows,
            registry=registry,
            mode=mode,
            bias=float(config.plain_semantic_eos_bias),
            start_frames=int(config.plain_semantic_eos_bias_start_frames),
            interval_frames=int(config.plain_semantic_eos_bias_interval_frames),
        )
        active = [index for index, row in enumerate(rows) if not row.finished]
        if not active:
            break
        forced: dict[int, int] = {}
        for index in active:
            row = rows[index]
            if (
                row.state is _State.MUSIC
                and len(row.semantic) < config.max_semantic_frames
                and not anchor_pending[index]
            ):
                pending = anchor_schedules[index].pop(
                    len(row.semantic), []
                )
                anchor_pending[index].extend(pending)
            if anchor_pending[index]:
                forced[index] = anchor_pending[index].pop(0)
        sampled_indices = [
            index
            for index in active
            if index not in forced and generated_steps[index] < config.max_new_tokens
        ]
        if not forced and not sampled_indices:
            break
        allowed_per_row = [
            _allowed_for(rows[index], sets, config, mode)
            for index in sampled_indices
        ]
        sampled = (
            _sample_rows(
                logits[sampled_indices],
                allowed_per_row,
                config,
                [generators[index] for index in sampled_indices],
            )
            if sampled_indices
            else []
        )
        tokens = [pad_id] * batch_size
        for index, token in forced.items():
            tokens[index] = token
            inserted_anchor_tokens[index] += 1
        for index, token in zip(sampled_indices, sampled):
            _advance(rows[index], token, sets, registry)
            generated_steps[index] += 1
            tokens[index] = token
            if (
                semantic_anchor_pools is not None
                and rows[index].state is _State.MUSIC
                and not anchor_schedules[index]
                and not rows[index].semantic
            ):
                pools = semantic_anchor_pools[index]
                occurrences: dict[str, int] = {}
                boundary = 0
                ordered = pools.get("__ordered__", [])
                for segment_index, (label, segment_length) in enumerate(zip(
                    rows[index].sections,
                    rows[index].segment_lengths,
                )):
                    occurrence = occurrences.get(label, 0)
                    occurrences[label] = occurrence + 1
                    candidates = pools.get(label, [])
                    selected = (
                        ordered[segment_index]
                        if segment_index < len(ordered)
                        else (
                            candidates[occurrence]
                            if occurrence < len(candidates)
                            else None
                        )
                    )
                    if selected is not None:
                        anchor_schedules[index].setdefault(
                            boundary, []
                        ).extend(
                            int(value)
                            for value in np.asarray(selected).reshape(-1)
                        )
                    boundary += int(segment_length) * 4
        step_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)
        participating = set(forced) | set(sampled_indices)
        step_mask = torch.tensor(
            [[int(index in participating)] for index in range(batch_size)],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.cat([attention_mask, step_mask], dim=1)
        outputs = model.transformer_forward(
            input_ids=step_ids,
            attention_mask=attention_mask,
            position_ids=next_position.unsqueeze(1),
            past_key_values=cache,
            use_cache=True,
        )
        hidden = outputs.last_hidden_state[:, -1, :]
        next_position = next_position + step_mask.squeeze(1)

    return GenerationOutput(
        semantic_ids=[np.asarray(row.semantic, dtype=np.int64) for row in rows],
        generated_melody_ids=[
            np.asarray(row.melody, dtype=np.int64) if row.melody else None for row in rows
        ],
        melody_sections=[list(row.sections) for row in rows],
        melody_segment_lengths=[list(row.segment_lengths) for row in rows],
        semantic_anchor_tokens=inserted_anchor_tokens,
        modes=[
            (
                mode.value
                if mode.is_cover
                else mode.value
                if row.tokens
                and row.tokens[0] == registry.control("melody_bos")
                and mode.has_melody
                else SequenceMode.SECTION.value
                if row.tokens and row.tokens[0] == registry.control("melody_bos")
                else SequenceMode.PLAIN.value
            )
            for row in rows
        ],
        requested_modes=[mode.value] * batch_size,
        finished=[row.finished for row in rows],
        stop_reasons=[
            (
                "frame_cap"
                if row.finished and len(row.semantic) >= config.max_semantic_frames
                else "natural_eos"
                if row.finished
                else "max_new_tokens"
                if len(row.tokens) >= config.max_new_tokens
                else "unfinished"
            )
            for row in rows
        ],
        tokenizer_revision=tokenizer_revision,
        registry_revision=registry.revision,
        generation_config=config.to_dict(),
        raw_token_ids=[np.asarray(row.tokens, dtype=np.int64) for row in rows],
    )


def parse_generated_tokens(token_ids: np.ndarray, registry: TokenRegistry) -> dict[str, Any]:
    music_bos = registry.control("music_bos")
    music_eos = registry.control("music_eos")
    melody_bos = registry.control("melody_bos")
    melody_eos = registry.control("melody_eos")
    seg_end = registry.control("seg_end")
    section_lookup = {registry.section_control(label): label for label in SECTION_LABELS}

    semantic: list[int] = []
    melody: list[int] = []
    sections: list[str] = []
    region = "prefix"
    saw_music_eos = False
    for token in token_ids.tolist():
        token = int(token)
        if token == melody_bos:
            region = "melody"
            continue
        if token == melody_eos:
            region = "prefix"
            continue
        if token == music_bos:
            region = "music"
            continue
        if token == music_eos:
            saw_music_eos = True
            region = "prefix"
            continue
        if token in (seg_end, registry.eos_id, registry.pad_id):
            continue
        if region == "melody":
            if token in section_lookup:
                sections.append(section_lookup[token])
            elif registry.is_melody(token):
                melody.append(token - registry.melody_base)
            else:
                raise ValueError(
                    "Melody region contains a non-melody token: "
                    f"{registry.describe(token)} (id={token})"
                )
        elif region == "music":
            if not registry.is_semantic(token):
                raise ValueError(
                    "Semantic region contains a non-semantic token: "
                    f"{registry.describe(token)} (id={token})"
                )
            semantic.append(token - registry.semantic_base)
    return {
        "semantic_ids": np.asarray(semantic, dtype=np.int64),
        "melody_ids": np.asarray(melody, dtype=np.int64),
        "melody_sections": sections,
        "complete": saw_music_eos,
    }
