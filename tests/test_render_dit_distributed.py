from __future__ import annotations

import copy
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from open_qwen_music.common.distributed import (
    barrier,
    cleanup_distributed,
    init_distributed,
)
from open_qwen_music.render.conditioning import (
    FrozenQwenEmbeddingAdapter,
    RenderConditioner,
)
from open_qwen_music.render.flow import (
    FlowConfig,
    MaskedLoss,
    make_flow_training_sample,
    sample_source_like,
)
from open_qwen_music.render.trainer_dit import (
    DistributedDurationBucketBatchSampler,
    DurationBucketPolicy,
    RenderDiTTrainer,
    RevisionContract,
    TrainingLossOutput,
    collate_cached_render_batch,
    move_batch_to_device,
    train,
    validate_dit_batch,
)


class FakeTextEncoder(nn.Module):
    hidden_size = 8

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, self.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _revision_mapping() -> dict[str, str]:
    return {
        "tokenizer_revision": "semantic-v1",
        "vae_revision": "vae-v1",
        "text_encoder_revision": "text-model-v1",
        "text_tokenizer_revision": "text-tokenizer-v1",
        "text_cache_revision": "text-cache-v1",
        "rewriter_revision": "rewriter-v1",
        "latent_cache_revision": "latent-cache-v1",
        "latent_stats_sha256": "6" * 64,
    }


def _record(length: int, offset: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(10_000 + offset)
    return {
        "sample_id": f"sample-{offset}",
        "semantic_ids": (torch.arange(length, dtype=torch.long) + offset) % 32_768,
        "latents": torch.randn(length, 128, generator=generator),
        "description_embeddings": torch.randn(2, 8, generator=generator),
        "description_mask": torch.ones(2, dtype=torch.bool),
        "lyrics_embeddings": torch.randn(3, 8, generator=generator),
        "lyrics_mask": torch.ones(3, dtype=torch.bool),
        "revisions": _revision_mapping(),
    }


class TinyRenderModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(128, hidden_size)
        self.output_projection = nn.Linear(hidden_size, 128)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        conditioning: Any,
        frame_mask: torch.Tensor,
        global_loudness_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_weights = conditioning.text_mask.unsqueeze(-1)
        text = (conditioning.text_context * text_weights).sum(dim=1)
        text = text / text_weights.sum(dim=1).clamp_min(1)
        hidden = self.input_projection(noisy_latents)
        hidden = hidden + conditioning.semantic_embeddings + text.unsqueeze(1)
        if global_loudness_embedding is not None:
            hidden = hidden + global_loudness_embedding.unsqueeze(1)
        hidden = hidden + timestep[:, None, None]
        output = self.output_projection(torch.tanh(hidden))
        return output.masked_fill(~frame_mask.unsqueeze(-1), 0.0)


def _make_stack() -> tuple[nn.Module, RenderConditioner]:
    encoder = FakeTextEncoder()
    adapter = FrozenQwenEmbeddingAdapter(
        model_id="fake/qwen",
        revision="text-model-v1",
        tokenizer_revision="text-tokenizer-v1",
        cache_revision="text-cache-v1",
        encoder=encoder,
        hidden_size=8,
    )
    conditioner = RenderConditioner(
        hidden_size=8,
        text_encoder=adapter,
        text_encoder_dim=8,
        lyrics_num_layers=6,
        lyrics_num_heads=2,
        lyrics_head_dim=4,
        lyrics_ffn_expansion=2,
    )
    model = TinyRenderModel(hidden_size=8)
    return model, conditioner


class DeterministicRenderTrainer(RenderDiTTrainer):

    def compute_loss(self, batch: Any) -> TrainingLossOutput:
        values = validate_dit_batch(
            move_batch_to_device(batch, self.device),
            expected_revisions=self.revisions,
        )
        mask = values["semantic_mask"]
        text_drop_mask = torch.zeros(
            values["latents"].shape[0],
            dtype=torch.bool,
            device=self.device,
        )
        timestep = torch.full(
            (values["latents"].shape[0],),
            0.5,
            device=self.device,
            dtype=values["latents"].dtype,
        )
        prediction = self.training_graph(
            values["latents"],
            timestep,
            semantic_ids=values["semantic_ids"],
            frame_mask=mask,
            description_mask=values["description_mask"],
            lyrics_mask=values["lyrics_mask"],
            description_embeddings=values["description_embeddings"],
            lyrics_embeddings=values["lyrics_embeddings"],
            cache_provenance=self._cache_provenance(values),
            text_drop_mask=text_drop_mask,
        )
        frame_error = prediction.float().square().mean(dim=-1)
        numerator = (frame_error * mask).sum()
        denominator = mask.sum(dtype=torch.float32)
        masked = MaskedLoss(
            loss=numerator / denominator,
            numerator=numerator,
            denominator=denominator,
            valid_frames=mask.sum(),
        )
        return TrainingLossOutput(
            masked_loss=masked,
            prediction=prediction,
            target_velocity=torch.zeros_like(prediction),
            text_drop_mask=text_drop_mask,
            timestep=timestep,
        )


def _make_deterministic_trainer(
    *, rank: int = 0, world_size: int = 1
) -> DeterministicRenderTrainer:
    torch.manual_seed(1234)
    model, conditioner = _make_stack()
    return DeterministicRenderTrainer(
        model=model,
        conditioner=conditioner,
        flow_config=FlowConfig(),
        revisions=RevisionContract.from_mapping(_revision_mapping()),
        learning_rate=1.0e-3,
        weight_decay=0.0,
        warmup_steps=0,
        max_steps=1,
        gradient_clip_norm=1.0,
        text_drop_probability=0.0,
        seed=99,
        rank=rank,
        world_size=world_size,
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _set_dist_env(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    torch.set_num_threads(1)


def _global_ot_worker(
    rank: int,
    world_size: int,
    port: int,
    destination: str,
) -> None:
    _set_dist_env(rank, world_size, port)
    init_distributed()
    try:
        config = FlowConfig(
            time_direction="data_to_noise",
            source_coupling="minibatch_ot",
            source_coupling_scope="global",
        )
        reference = torch.empty(1, 3, 2)
        expected_sources = [
            sample_source_like(
                reference,
                config,
                generator=torch.Generator().manual_seed(100 + source_rank),
            )
            for source_rank in range(world_size)
        ]
        target = expected_sources[1 - rank].clone()
        sample = make_flow_training_sample(
            target,
            config,
            source_generator=torch.Generator().manual_seed(100 + rank),
            timestep=torch.tensor([0.5]),
            frame_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        torch.save(
            {
                "source": sample.source,
                "target": target,
                "batch_size": sample.source_coupling_batch_size,
                "scope": sample.source_coupling_scope,
                "cost_before": sample.source_coupling_cost_before,
                "cost_after": sample.source_coupling_cost_after,
            },
            Path(destination) / f"global-ot-rank{rank}.pt",
        )
    finally:
        cleanup_distributed()


def test_world2_global_minibatch_ot_is_effective_with_local_batch_one(
    tmp_path: Path,
) -> None:
    mp.spawn(
        _global_ot_worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        result = torch.load(
            tmp_path / f"global-ot-rank{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert result["scope"] == "global"
        assert result["batch_size"] == 2
        assert torch.equal(result["source"], result["target"])
        assert float(result["cost_after"]) == 0.0
        assert float(result["cost_before"]) > 0.0


def _sampler_worker(
    rank: int,
    world_size: int,
    port: int,
    destination: str,
) -> None:
    _set_dist_env(rank, world_size, port)
    init_distributed()
    try:
        lengths = [100, 200, 400, 600, 800, 1_000, 1_600, 2_000, 2_200]
        sampler = DistributedDurationBucketBatchSampler(
            lengths,
            policy=DurationBucketPolicy(),
            batch_size_by_bucket={30: 2, 90: 2, 180: 2, 360: 2},
            rank=rank,
            world_size=world_size,
            seed=7,
            training_config_hash="test-config",
            shuffle=True,
        )
        batches = list(sampler)
        sampler.advance(1)
        torch.save(
            {"batches": batches, "state": sampler.state_dict()},
            Path(destination) / f"sampler-rank{rank}.pt",
        )
        barrier()
    finally:
        cleanup_distributed()


def test_world2_sampler_is_disjoint_deterministic_and_strict_resume(
    tmp_path: Path,
) -> None:
    mp.spawn(
        _sampler_worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    outputs = [
        torch.load(
            tmp_path / f"sampler-rank{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for rank in range(2)
    ]
    assert outputs[0]["state"] == outputs[1]["state"]
    seen: set[int] = set()
    lengths = [100, 200, 400, 600, 800, 1_000, 1_600, 2_000, 2_200]
    policy = DurationBucketPolicy()
    for left, right in zip(outputs[0]["batches"], outputs[1]["batches"], strict=True):
        assert set(left).isdisjoint(right)
        step = list(left) + list(right)
        assert len({policy.bucket_for_frames(lengths[index]) for index in step}) == 1
        assert seen.isdisjoint(step)
        seen.update(step)

    wrong_world = DistributedDurationBucketBatchSampler(
        lengths,
        policy=policy,
        batch_size_by_bucket={30: 2, 90: 2, 180: 2, 360: 2},
        rank=0,
        world_size=1,
        seed=7,
        training_config_hash="test-config",
    )
    with pytest.raises(RuntimeError, match="world/batch"):
        wrong_world.load_state_dict(outputs[0]["state"])
    wrong_batch = DistributedDurationBucketBatchSampler(
        lengths,
        policy=policy,
        batch_size_by_bucket={30: 3, 90: 3, 180: 3, 360: 3},
        rank=0,
        world_size=2,
        seed=7,
        training_config_hash="test-config",
    )
    with pytest.raises(RuntimeError, match="world/batch"):
        wrong_batch.load_state_dict(outputs[0]["state"])


def _duration_curriculum() -> dict[str, Any]:
    return {
        "format_version": "oqm.render.duration-curriculum.v1",
        "mode": "nested_shortest",
        "ordering": "frame_length_then_sample_id_sha256",
        "stages": [
            {"start_step": 0, "active_records": 64, "batch_size_per_rank": 4},
            {
                "start_step": 1_000,
                "active_records": 128,
                "batch_size_per_rank": 2,
            },
            {
                "start_step": 2_500,
                "active_records": 192,
                "batch_size_per_rank": 1,
            },
            {
                "start_step": 5_000,
                "active_records": 256,
                "batch_size_per_rank": 1,
            },
        ],
    }


def _curriculum_sampler(*, rank: int = 0) -> DistributedDurationBucketBatchSampler:
    return DistributedDurationBucketBatchSampler(
        [25 * (10 + index) for index in range(256)],
        policy=DurationBucketPolicy(),
        batch_size_by_bucket={30: 1, 90: 1, 180: 1, 360: 1},
        rank=rank,
        world_size=16,
        seed=7,
        training_config_hash="duration-curriculum-test",
        sample_ids=[f"sample-{index:03d}" for index in range(256)],
        duration_curriculum=_duration_curriculum(),
        shuffle=True,
    )


def test_duration_curriculum_is_nested_complete_and_length_sorted() -> None:
    samplers = [_curriculum_sampler(rank=rank) for rank in range(16)]
    expected = [
        (0, 64, 4, 63),
        (1_000, 128, 2, 31),
        (2_500, 192, 1, 15),
        (5_000, 256, 1, 15),
    ]
    previous: set[int] = set()
    for step, active_records, local_batch, max_batches in expected:
        for sampler in samplers:
            sampler.set_global_step(step)
        rank_batches = [list(sampler) for sampler in samplers]
        assert len(rank_batches[0]) == active_records // (16 * local_batch)
        active: set[int] = set()
        for batches in zip(*rank_batches, strict=True):
            global_batch = [index for batch in batches for index in batch]
            assert len(global_batch) == 16 * local_batch
            assert len(set(global_batch)) == len(global_batch)
            assert max(global_batch) - min(global_batch) <= max_batches
            active.update(global_batch)
        assert len(active) == active_records
        assert previous <= active
        previous = active
        telemetry = samplers[0].curriculum_telemetry
        assert telemetry is not None
        assert telemetry["active_records"] == active_records
        assert telemetry["batch_size_per_rank"] == local_batch


def test_duration_curriculum_resume_preserves_stage_and_cursor() -> None:
    sampler = _curriculum_sampler()
    assert sampler.set_global_step(2_500)
    sampler.set_epoch(11)
    original_batches = list(sampler)
    sampler.advance(3)
    state = sampler.state_dict()

    resumed = _curriculum_sampler()
    resumed.load_state_dict(state)
    assert resumed.curriculum_telemetry == sampler.curriculum_telemetry
    assert list(resumed) == original_batches[3:]
    assert resumed.set_global_step(5_000)
    assert resumed.cursor == 0
    assert resumed.curriculum_telemetry["active_records"] == 256


def test_duration_assertion_rejects_half_frame_bankers_round_boundary() -> None:
    batch = collate_cached_render_batch([_record(2, 0)])
    batch["duration_seconds"] = torch.tensor([0.1])
    with pytest.raises(ValueError, match="canonical"):
        validate_dit_batch(batch)


def _gradient_worker(
    rank: int,
    world_size: int,
    port: int,
    destination: str,
) -> None:
    _set_dist_env(rank, world_size, port)
    _, local_rank, _, _ = init_distributed()
    try:
        trainer = _make_deterministic_trainer(rank=rank, world_size=world_size)
        trainer.wrap_distributed(local_rank=local_rank, resume=False)
        reducer_count = sum(
            isinstance(module, DistributedDataParallel)
            for module in trainer.training_graph.modules()
        )
        assert reducer_count == 1
        assert isinstance(trainer.training_graph, DistributedDataParallel)
        assert trainer.unwrapped_training_graph.model is trainer.unwrapped_model
        assert (
            trainer.unwrapped_training_graph.conditioner
            is trainer.unwrapped_conditioner
        )
        lengths = ((2, 3), (5, 4))[rank]
        batches = [
            collate_cached_render_batch([_record(length, rank * 10 + index)])
            for index, length in enumerate(lengths)
        ]
        trainer.train_update(batches)
        if rank == 0:
            torch.save(
                {
                    "ddp_reducer_count": reducer_count,
                    "model": {
                        name: parameter.grad.detach().clone()
                        for name, parameter in trainer.unwrapped_model.named_parameters()
                        if parameter.grad is not None
                    },
                    "conditioner": {
                        name: parameter.grad.detach().clone()
                        for name, parameter in trainer.unwrapped_conditioner.named_parameters()
                        if parameter.grad is not None
                    },
                },
                destination,
            )
        barrier()
    finally:
        cleanup_distributed()


def _assert_state_close(
    expected: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
) -> None:
    assert set(expected) == set(actual)
    for name in expected:
        assert torch.allclose(expected[name], actual[name], atol=1.0e-5, rtol=1.0e-5), (
            name
        )


def test_world2_global_weighted_gradient_matches_single_process(
    tmp_path: Path,
) -> None:
    output = tmp_path / "distributed-gradient.pt"
    mp.spawn(
        _gradient_worker,
        args=(2, _free_port(), str(output)),
        nprocs=2,
        join=True,
    )
    distributed = torch.load(output, map_location="cpu", weights_only=False)
    assert distributed["ddp_reducer_count"] == 1

    baseline = _make_deterministic_trainer()
    baseline.train_update(
        [
            collate_cached_render_batch([_record(2, 0), _record(5, 10)]),
            collate_cached_render_batch([_record(3, 1), _record(4, 11)]),
        ]
    )
    _assert_state_close(
        {
            name: parameter.grad
            for name, parameter in baseline.unwrapped_model.named_parameters()
            if parameter.grad is not None
        },
        distributed["model"],
    )
    _assert_state_close(
        {
            name: parameter.grad
            for name, parameter in baseline.unwrapped_conditioner.named_parameters()
            if parameter.grad is not None
        },
        distributed["conditioner"],
    )


def _write_cache_fixture(root: Path) -> Path:
    manifest = root / "cache.jsonl"
    lines = []
    for index, length in enumerate((2, 3, 4, 5, 6, 7)):
        cache = root / f"sample-{index}.pt"
        torch.save(_record(length, index), cache)
        lines.append(
            {
                "sample_id": f"sample-{index}",
                "cache_path": str(cache),
                "latent_frames": length,
                "revisions": _revision_mapping(),
            }
        )
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in lines),
        encoding="utf-8",
    )
    return manifest


def _training_config(manifest: Path, output_dir: Path) -> dict[str, Any]:
    revisions = _revision_mapping()
    return {
        "format_version": "oqm.render.dit.config.test.v1",
        "model": {
            "latent_dim": 128,
            "latent_frame_hz": 25,
            "hidden_size": 8,
            "context_dim": 8,
            "max_frames": 128,
            "adaln_conditioning": "timestep",
            "activation_checkpointing": False,
        },
        "conditioning": {
            "description_max_tokens": 256,
            "lyrics_max_tokens": 1536,
            "lyrics_encoder_layers": 6,
            "lyrics_encoder_heads": 2,
            "lyrics_head_dim": 4,
            "lyrics_ffn_expansion": 2,
            "dropout": 0.0,
            "text_drop_probability": 0.25,
            "text_encoder": {
                "model_id": "fake/qwen",
                "hidden_size": 8,
                "local_files_only": True,
            },
        },
        "flow": FlowConfig().to_dict(),
        "revisions": revisions,
        "data": {
            "cache_manifest": str(manifest),
            "duration_buckets_seconds": [30, 90, 180, 360],
            "require_cached_text": True,
            "num_workers": 0,
            "drop_last": False,
        },
        "optimizer": {
            "lr": 1.0e-3,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "min_lr_ratio": 1.0,
        },
        "train": {
            "seed": 20260810,
            "device": "cpu",
            "batch_size_per_rank": 1,
            "gradient_accumulation_steps": 2,
            "global_audio_seconds_per_update": 0.0,
            "gradient_clip_norm": 1.0,
            "max_steps": 3,
            "save_every_steps": 1,
            "output_dir": str(output_dir),
        },
    }


def _train_worker(
    rank: int,
    world_size: int,
    port: int,
    config: dict[str, Any],
    resume_from: str | None,
    stop_after: int,
    extend_resume_to_step: int,
) -> None:
    _set_dist_env(rank, world_size, port)
    if stop_after:
        os.environ["OQM_STOP_AFTER_CHECKPOINT_STEP"] = str(stop_after)
    else:
        os.environ.pop("OQM_STOP_AFTER_CHECKPOINT_STEP", None)
    if extend_resume_to_step:
        os.environ["OQM_EXTEND_RESUME_TO_STEP"] = str(extend_resume_to_step)
    else:
        os.environ.pop("OQM_EXTEND_RESUME_TO_STEP", None)
    train(
        config,
        fake_text_encoder=FakeTextEncoder(),
        resume_from=resume_from,
    )


def _spawn_train(
    config: dict[str, Any],
    *,
    resume_from: Path | None = None,
    stop_after: int = 0,
    extend_resume_to_step: int = 0,
) -> None:
    mp.spawn(
        _train_worker,
        args=(
            2,
            _free_port(),
            config,
            str(resume_from) if resume_from is not None else None,
            stop_after,
            extend_resume_to_step,
        ),
        nprocs=2,
        join=True,
    )


def _assert_nested_tensor_equal(expected: Any, actual: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert torch.equal(expected, actual)
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert np.array_equal(expected, actual)
    elif isinstance(expected, Mapping):
        assert set(expected) == set(actual)
        for key in expected:
            _assert_nested_tensor_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            _assert_nested_tensor_equal(left, right)
    else:
        assert expected == actual








def _drift_worker(
    rank: int,
    world_size: int,
    port: int,
    config: dict[str, Any],
    destination: str,
) -> None:
    _set_dist_env(rank, world_size, port)
    local = copy.deepcopy(config)
    if rank == 1:
        local["revisions"]["latent_cache_revision"] = "rank1-drift"
    try:
        train(local, fake_text_encoder=FakeTextEncoder())
    except Exception as exc:
        Path(destination, f"drift-rank{rank}.txt").write_text(
            f"{type(exc).__name__}: {exc}", encoding="utf-8"
        )
    else:
        raise AssertionError("revision drift must be within DDP failed before construction")


def test_world2_config_revision_drift_fails_consistently_before_ddp(
    tmp_path: Path,
) -> None:
    manifest = _write_cache_fixture(tmp_path)
    config = _training_config(manifest, tmp_path / "drift-output")
    mp.spawn(
        _drift_worker,
        args=(2, _free_port(), config, str(tmp_path)),
        nprocs=2,
        join=True,
    )
    errors = [
        (tmp_path / f"drift-rank{rank}.txt").read_text(encoding="utf-8")
        for rank in range(2)
    ]
    assert all("Startup state differs across ranks" in error for error in errors)
    assert all("render_startup" in error for error in errors)
