
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import (
    config_hash,
    load_checkpoint,
    log_kv,
    read_sidecar,
    resolve_base_model_revision,
    resolve_text_tokenizer_revision,
    source_tree_revision,
)
from .condition import ConditionRenderConfig, MusicCondition, Section
from .contracts import (
    SEMANTIC_CODEBOOK_SIZE,
    SEMANTIC_FRAME_RATE,
    SEQUENCE_PROTOCOL_REVISION,
    SequenceMode,
)
from .generate import GenerationConfig, generate_semantic
from .metrics import legality_report
from .model import build_model
from .render_contract import RenderRequest, prompt_token_sha256, write_render_requests
from .sequence import SequenceBuilder, SequenceConfig


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_ids_sha256(sample_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def _checkpoint_stat_identity(path: Path) -> tuple[tuple[str, int, int], ...]:
    files = (
        [path]
        if path.is_file()
        else sorted(item for item in path.rglob("*") if item.is_file())
    )
    sidecar = Path(str(path) + ".json")
    if sidecar.is_file():
        files.append(sidecar)
    return tuple(
        (
            str(file.resolve()),
            file.stat().st_size,
            file.stat().st_mtime_ns,
        )
        for file in files
    )


def load_prompts(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}  contains invalid JSON: {error}") from error
    if not records:
        raise ValueError(f"{path} ")
    return records


def validate_generation_checkpoint(
    provenance: dict[str, Any],
    *,
    config: dict[str, Any],
    registry_revision: str,
    condition_template_version: str,
    semantic_tokenizer_revision: str | None = None,
) -> str:
    problems: list[str] = []
    if provenance.get("registry_revision") != registry_revision:
        problems.append(
            f"registry {registry_revision!r} != checkpoint "
            f"{provenance.get('registry_revision')!r}"
        )
    if provenance.get("condition_template_version") != condition_template_version:
        problems.append("condition template revision does not match")
    if provenance.get("sequence_protocol_revision") != SEQUENCE_PROTOCOL_REVISION:
        problems.append("sequence protocol revision does not match")
    semantic = provenance.get("semantic_tokenizer") or {}
    if float(semantic.get("frame_rate", -1)) != SEMANTIC_FRAME_RATE:
        problems.append("semantic frame rate does not match")
    if int(semantic.get("codebook_size", -1)) != SEMANTIC_CODEBOOK_SIZE:
        problems.append("semantic codebook size does not match")
    tokenizer_revision = str(semantic.get("revision") or "unknown")
    if tokenizer_revision == "unknown":
        problems.append("semantic tokenizer revision is not pinned")
    if (
        semantic_tokenizer_revision is not None
        and tokenizer_revision != str(semantic_tokenizer_revision)
    ):
        problems.append(
            f"semantic tokenizer revision {semantic_tokenizer_revision!r} != "
            f"checkpoint {tokenizer_revision!r}"
        )
    expected_extractor = str(
        (config.get("registry") or {}).get("semantic_extractor_revision") or ""
    )
    checkpoint_extractor = str(
        semantic.get("semantic_extractor_revision") or "unknown"
    )
    if expected_extractor and checkpoint_extractor != expected_extractor:
        problems.append(
            "semantic extractor revision "
            f"{expected_extractor!r} != checkpoint {checkpoint_extractor!r}"
        )
    expected_text_tokenizer = resolve_text_tokenizer_revision(config)
    checkpoint_text_tokenizer = str(
        (provenance.get("text_tokenizer") or {}).get("revision") or "unknown"
    )
    if (
        checkpoint_text_tokenizer != "unknown"
        and expected_text_tokenizer != "unknown"
        and checkpoint_text_tokenizer != expected_text_tokenizer
    ):
        problems.append(
            "text tokenizer revision "
            f"{expected_text_tokenizer!r} != checkpoint {checkpoint_text_tokenizer!r}"
        )
    checkpoint_source_revision = str(
        (provenance.get("source") or {}).get("revision") or "unknown"
    )
    current_source_revision = source_tree_revision()
    allow_decode_revision_mismatch = bool(
        (config.get("generation") or {}).get(
            "allow_decode_source_revision_mismatch",
            False,
        )
    )
    if (
        checkpoint_source_revision != "unknown"
        and checkpoint_source_revision != current_source_revision
        and not allow_decode_revision_mismatch
    ):
        problems.append(
            "source revision "
            f"{current_source_revision!r} != checkpoint {checkpoint_source_revision!r}"
        )
    configured_base = resolve_base_model_revision(config)
    checkpoint_base = (provenance.get("base_model") or {}).get("revision")
    if configured_base != "unknown" and configured_base != checkpoint_base:
        problems.append(
            f"base model revision {configured_base!r} != checkpoint {checkpoint_base!r}"
        )
    if problems:
        raise RuntimeError("Generation checkpoint contract is incompatible:\n  - " + "\n  - ".join(problems))
    return tokenizer_revision


def required_generation_steps(mode: SequenceMode, config: GenerationConfig) -> int:
    # MUSIC_BOS + semantic + MUSIC_EOS + EOS
    required = int(config.max_semantic_frames) + 3
    if mode.has_melody and not mode.is_cover:

        required += int(config.max_melody_tokens) + 2 * int(config.max_melody_segments) + 2
    return required


def run_generation(
    config: dict[str, Any],
    *,
    checkpoint: str | Path,
    model: torch.nn.Module | None = None,
    semantic_tokenizer_revision: str | None = None,
    prompts_path: str | Path,
    output_path: str | Path,
    mode: str = "section",
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    semantic_repetition_penalty: float | None = None,
    semantic_repetition_window: int | None = None,
    melody_repetition_penalty: float | None = None,
    melody_repetition_window: int | None = None,
    melody_unvoiced_run_threshold: int | None = None,
    melody_unvoiced_run_penalty: float | None = None,
    seed: int = 0,
    batch_size: int = 4,
    strict_mode: bool = True,
    device: str | None = None,
    render_requests_path: str | Path | None = None,
) -> dict[str, Any]:
    from .trainer import build_text_encoder, resolve_registry

    checkpoint = Path(checkpoint).resolve(strict=True)
    checkpoint_identity_before = _checkpoint_stat_identity(checkpoint)
    sequence_mode = SequenceMode(mode)
    registry = resolve_registry(config)
    text_encoder = build_text_encoder(config)
    sequence_config = SequenceConfig.from_config(config)
    sequence_config.random_crop = False
    sequence_config.resample_unique_sections = False
    builder = SequenceBuilder(
        registry,
        text_encoder,
        sequence_config,
        ConditionRenderConfig.from_config(config),
    )

    torch_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if model is None:
        sidecar = read_sidecar(checkpoint)
        provenance = sidecar.get("provenance") or {}
        tokenizer_revision = validate_generation_checkpoint(
            provenance,
            config=config,
            registry_revision=registry.revision,
            condition_template_version=builder.condition_template_version,
        )
    else:
        if not semantic_tokenizer_revision:
            raise ValueError(
                "semantic_tokenizer_revision is required with a pretrained model"
            )
        tokenizer_revision = str(semantic_tokenizer_revision)
        sidecar = {"stage": "published", "global_step": None, "config_hash": None}
        provenance = {"source": {"revision": "published-weight-bundle"}}

    generation_section = dict(config.get("generation", {}) or {})
    generation_config = GenerationConfig(
        max_new_tokens=int(
            max_new_tokens
            if max_new_tokens is not None
            else generation_section.get("max_new_tokens", 3000)
        ),
        temperature=float(
            temperature if temperature is not None else generation_section.get("temperature", 0.9)
        ),
        top_p=float(top_p if top_p is not None else generation_section.get("top_p", 0.95)),
        top_k=int(generation_section.get("top_k", 0)),
        min_semantic_frames=int(generation_section.get("min_semantic_frames", 25)),
        max_semantic_frames=int(
            generation_section.get("max_semantic_frames", sequence_config.max_semantic_frames)
        ),
        max_melody_tokens=int(generation_section.get("max_melody_tokens", 512)),
        max_melody_segments=int(generation_section.get("max_melody_segments", 16)),
        min_melody_segments=int(generation_section.get("min_melody_segments", 1)),
        semantic_repetition_penalty=float(
            semantic_repetition_penalty
            if semantic_repetition_penalty is not None
            else generation_section.get("semantic_repetition_penalty", 0.0)
        ),
        semantic_repetition_window=int(
            semantic_repetition_window
            if semantic_repetition_window is not None
            else generation_section.get("semantic_repetition_window", 64)
        ),
        plain_semantic_repetition_penalty=float(
            generation_section.get("plain_semantic_repetition_penalty", 0.0)
        ),
        melody_repetition_penalty=float(
            melody_repetition_penalty
            if melody_repetition_penalty is not None
            else generation_section.get("melody_repetition_penalty", 0.0)
        ),
        melody_repetition_window=int(
            melody_repetition_window
            if melody_repetition_window is not None
            else generation_section.get("melody_repetition_window", 32)
        ),
        melody_unvoiced_run_threshold=int(
            melody_unvoiced_run_threshold
            if melody_unvoiced_run_threshold is not None
            else generation_section.get("melody_unvoiced_run_threshold", 0)
        ),
        melody_unvoiced_run_penalty=float(
            melody_unvoiced_run_penalty
            if melody_unvoiced_run_penalty is not None
            else generation_section.get("melody_unvoiced_run_penalty", 0.0)
        ),
        plain_semantic_eos_bias=float(
            generation_section.get("plain_semantic_eos_bias", 0.0)
        ),
        plain_semantic_eos_bias_start_frames=int(
            generation_section.get("plain_semantic_eos_bias_start_frames", 0)
        ),
        plain_semantic_eos_bias_interval_frames=int(
            generation_section.get("plain_semantic_eos_bias_interval_frames", 625)
        ),
        seed=int(seed),
        strict_mode=bool(strict_mode),
    )
    required_steps = required_generation_steps(sequence_mode, generation_config)
    if generation_config.max_new_tokens < required_steps:
        raise ValueError(
            f"generation.max_new_tokens={generation_config.max_new_tokens} "
            f"is below the {required_steps} steps required for {sequence_mode.value} "
            "to reach max_semantic_frames. Increase the generation budget or "
            "explicitly lower the semantic-frame limit."
        )

    generation_precision = str(
        generation_section.get(
            "precision",
            (config.get("train") or {}).get("precision", "bf16"),
        )
    )
    if generation_precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("generation.precision must be 'bf16', 'fp16', or 'fp32'")
    generation_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[generation_precision]
    if model is None:
        model = build_model(config, registry)
        if torch_device.type == "cuda":
            model = model.to(device=torch_device, dtype=generation_dtype)
        else:
            model = model.to(torch_device)
        load_checkpoint(checkpoint, model=model, resume=False)
    elif torch_device.type == "cuda":
        model = model.to(device=torch_device, dtype=generation_dtype)
    else:
        model = model.to(torch_device)
    if _checkpoint_stat_identity(checkpoint) != checkpoint_identity_before:
        raise RuntimeError("Checkpoint or sidecar changed while weights were loading")
    model.eval()

    records = load_prompts(prompts_path)
    semantic_results: list[np.ndarray] = []
    melody_results: list[np.ndarray | None] = []
    section_results: list[list[str]] = []
    raw_results: list[np.ndarray] = []
    finished_flags: list[bool] = []
    stop_reason_results: list[str] = []
    actual_mode_results: list[str] = []
    sample_ids: list[str] = []
    condition_results: list[MusicCondition] = []
    prompt_hashes: list[str] = []
    seen_sample_ids: set[str] = set()

    for start in range(0, len(records), max(1, batch_size)):
        chunk = records[start : start + max(1, batch_size)]
        prompts = []
        chunk_seeds: list[int] = []
        for record in chunk:
            condition = MusicCondition.from_record(record)
            reference = record.get("reference_melody")
            raw_reference_sections = record.get("reference_sections")
            reference_sections = (
                None
                if raw_reference_sections is None
                else [
                    Section(**value) if isinstance(value, dict) else value
                    for value in raw_reference_sections
                ]
            )
            prompt = builder.build_prompt(
                condition,
                mode=sequence_mode,
                reference_melody=(
                    np.asarray(reference, dtype=np.int64) if reference is not None else None
                ),
                reference_sections=reference_sections,
            )
            prompts.append(prompt)
            condition_results.append(condition)
            prompt_hash = prompt_token_sha256(prompt)
            prompt_hashes.append(prompt_hash)
            sample_id = str(record.get("sample_id") or "")
            if not sample_id:
                raise ValueError("Every generation prompt must have a stable non-empty sample_id")
            if sample_id in seen_sample_ids:
                raise ValueError(f"Generation prompt sample_id is duplicated: {sample_id}")
            seen_sample_ids.add(sample_id)
            sample_ids.append(sample_id)
            chunk_seeds.append(
                int.from_bytes(
                    hashlib.sha256(
                        f"{seed}\0{sample_id}\0{prompt_hash}".encode()
                    ).digest()[:8],
                    "little",
                )
                & ((1 << 63) - 1)
            )
        output = generate_semantic(
            model,
            prompts,
            mode=sequence_mode,
            config=generation_config,
            tokenizer_revision=tokenizer_revision,
            device=torch_device,
            row_seeds=chunk_seeds,
        )
        semantic_results.extend(output.semantic_ids)
        melody_results.extend(output.generated_melody_ids)
        section_results.extend(output.melody_sections)
        raw_results.extend(output.raw_token_ids)
        finished_flags.extend(output.finished)
        stop_reason_results.extend(output.stop_reasons)
        actual_mode_results.extend(output.modes)
        log_kv(
            "generate",
            {
                "done": len(semantic_results),
                "total": len(records),
                "mean_semantic_frames": float(
                    np.mean([s.size for s in output.semantic_ids]) if output.semantic_ids else 0.0
                ),
            },
        )

    report = legality_report(raw_results, registry, finished=finished_flags)
    output_file = Path(output_path)
    if output_file.suffix != ".npz":
        raise ValueError(f"Generation output must use the .npz suffix: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    raw_lengths = np.asarray([tokens.size for tokens in raw_results], dtype=np.int64)
    stop_reasons = np.asarray(stop_reason_results, dtype=object)
    checkpoint_identity = {
        "path": str(Path(checkpoint).resolve()),
        "stage": sidecar.get("stage"),
        "global_step": sidecar.get("global_step"),
        "config_hash": sidecar.get("config_hash"),
    }
    render_requests = [
        RenderRequest(
            sample_id=sample_id,
            condition=condition,
            semantic_ids=semantic,
            melody_ids=melody,
            melody_sections=tuple(sections),
            mode=actual_mode,
            stop_reason=str(stop_reason),
            finished=bool(finished),
            semantic_tokenizer_revision=tokenizer_revision,
            registry_revision=registry.revision,
            condition_template_version=builder.condition_template_version,
            prompt_sha256=prompt_hash,
            checkpoint=checkpoint_identity,
            generation_config=generation_config.to_dict(),
        )
        for (
            sample_id,
            condition,
            semantic,
            melody,
            sections,
            stop_reason,
            finished,
            prompt_hash,
            actual_mode,
        ) in zip(
            sample_ids,
            condition_results,
            semantic_results,
            melody_results,
            section_results,
            stop_reasons.tolist(),
            finished_flags,
            prompt_hashes,
            actual_mode_results,
        )
    ]
    temporary_output = output_file.with_name(
        f".{output_file.stem}.tmp-{os.getpid()}{output_file.suffix or '.npz'}"
    )
    try:
        np.savez_compressed(
            temporary_output,
            sample_ids=np.asarray(sample_ids, dtype=object),
            semantic_lengths=np.asarray(
                [s.size for s in semantic_results], dtype=np.int64
            ),
            semantic_ids=np.concatenate(semantic_results)
            if semantic_results
            else np.zeros(0, dtype=np.int64),
            melody_lengths=np.asarray(
                [0 if m is None else m.size for m in melody_results], dtype=np.int64
            ),
            melody_ids=np.concatenate([m for m in melody_results if m is not None])
            if any(m is not None for m in melody_results)
            else np.zeros(0, dtype=np.int64),
            finished=np.asarray(finished_flags, dtype=bool),
            raw_lengths=raw_lengths,
            raw_token_ids=np.concatenate(raw_results)
            if raw_results
            else np.zeros(0, dtype=np.int64),
            stop_reasons=stop_reasons,
        )
        os.replace(temporary_output, output_file)
    finally:
        temporary_output.unlink(missing_ok=True)
    render_path = write_render_requests(
        render_requests_path or Path(output_path).with_suffix(".render.jsonl"),
        render_requests,
    )
    summary = {
        "output": str(output_file),
        "output_sha256": _sha256_file(output_file),
        "prompts_sha256": _sha256_file(Path(prompts_path).resolve()),
        "render_requests_sha256": _sha256_file(Path(render_path)),
        "sample_ids_sha256": _sample_ids_sha256(sample_ids),
        "sample_ids": sample_ids,
        "num_samples": len(semantic_results),
        "mode": sequence_mode.value,
        "actual_mode_counts": dict(sorted(Counter(actual_mode_results).items())),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_stage": sidecar.get("stage"),
        "checkpoint_step": sidecar.get("global_step"),
        "checkpoint_config_hash": sidecar.get("config_hash"),
        "checkpoint_source_revision": str(
            (provenance.get("source") or {}).get("revision") or "unknown"
        ),
        "runtime_source_revision": source_tree_revision(),
        "generation_precision": generation_precision,
        "model_dtype": str(next(model.parameters()).dtype),
        "decode_source_revision_mismatch_waived": bool(
            generation_section.get(
                "allow_decode_source_revision_mismatch",
                False,
            )
        ),
        "runtime_config_hash": config_hash(config),
        "registry_revision": registry.revision,
        "semantic_tokenizer_revision": tokenizer_revision,
        "condition_template_version": builder.condition_template_version,
        "sequence_protocol_revision": builder.sequence_protocol_revision,
        "generation_config": generation_config.to_dict(),
        "render_requests": str(render_path),
        "stop_reasons": stop_reasons.tolist(),
        "stop_reason_counts": {
            reason: int((stop_reasons == reason).sum()) for reason in sorted(set(stop_reasons))
        },
        "sections": section_results,
        **report.as_dict(),
    }
    summary["bundle_id"] = "sha256:" + hashlib.sha256(
        "|".join(
            (
                summary["output_sha256"],
                summary["prompts_sha256"],
                summary["render_requests_sha256"],
                summary["sample_ids_sha256"],
            )
        ).encode()
    ).hexdigest()
    summary_path = output_file.with_suffix(".summary.json")
    temporary_summary = summary_path.with_name(
        f".{summary_path.name}.tmp-{os.getpid()}"
    )
    temporary_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_summary, summary_path)
    log_kv("generate_done", {k: v for k, v in summary.items() if not isinstance(v, (dict, list))})
    return summary
