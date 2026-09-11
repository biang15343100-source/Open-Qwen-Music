from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import open_qwen_music.render.conditioning as conditioning_module
from open_qwen_music.render.conditioning import (
    FrozenQwenEmbeddingAdapter,
    LyricsRoPEEncoder,
    LyricsSwiGLU,
    LyricsSelfAttention,
    RenderConditioner,
    TextEncoderProvenance,
    build_render_conditioner_from_config,
    validate_cache_provenance,
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


class PositionAwareFakeTextEncoder(nn.Module):
    hidden_size = 2

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.position_ids: list[torch.Tensor] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        return_dict: bool = True,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        assert use_cache is False
        self.position_ids.append(position_ids.detach().clone())
        hidden = torch.stack(
            (input_ids.float(), position_ids.float()),
            dim=-1,
        )
        return SimpleNamespace(last_hidden_state=hidden + self.anchor * 0)


class NoTruncationTokenizer:
    is_fast = True
    padding_side = "left"
    truncation_side = "right"

    def __call__(
        self,
        texts,
        *,
        padding,
        truncation,
        max_length,
        return_tensors,
        add_special_tokens=True,
    ):
        del padding, truncation, return_tensors, add_special_tokens
        width = max_length + 1
        return {
            "input_ids": torch.ones(len(texts), width, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), width, dtype=torch.long),
        }


def test_shared_conditioner_factory_explicitly_maps_constructor_surfaces() -> None:

    tree = ast.parse(
        Path(conditioning_module.__file__).read_text(encoding="utf-8")
    )
    factory = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_render_conditioner_from_config"
    )

    def call_keywords(name: str) -> set[str]:
        calls = [
            node
            for node in ast.walk(factory)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ]
        assert len(calls) == 1
        return {
            keyword.arg
            for keyword in calls[0].keywords
            if keyword.arg is not None
        }

    adapter_parameters = set(
        inspect.signature(FrozenQwenEmbeddingAdapter.__init__).parameters
    ) - {"self", "tokenizer"}
    conditioner_parameters = set(
        inspect.signature(RenderConditioner.__init__).parameters
    ) - {"self"}
    assert call_keywords("FrozenQwenEmbeddingAdapter") == adapter_parameters
    assert call_keywords("RenderConditioner") == conditioner_parameters


def test_shared_conditioner_factory_maps_complete_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, dict[str, object]] = {}

    class CapturingAdapter:
        def __init__(self, **kwargs: object) -> None:
            observed["adapter"] = kwargs

    sentinel = object()

    def capture_conditioner(**kwargs: object) -> object:
        observed["conditioner"] = kwargs
        return sentinel

    monkeypatch.setattr(
        conditioning_module,
        "FrozenQwenEmbeddingAdapter",
        CapturingAdapter,
    )
    monkeypatch.setattr(
        conditioning_module,
        "RenderConditioner",
        capture_conditioner,
    )
    fake_encoder = nn.Identity()
    condition_config = {
        "semantic_vocab_size": 32_768,
        "description_max_tokens": 256,
        "description_projection_bias": True,
        "lyrics_max_tokens": 1_536,
        "lyrics_encoder_layers": 7,
        "lyrics_encoder_heads": 3,
        "lyrics_head_dim": 12,
        "lyrics_ffn_expansion": 5.0,
        "lyrics_ffn_activation": "swiglu",
        "lyrics_gelu_approximation": "none",
        "lyrics_qkv_bias": False,
        "lyrics_non_qkv_linear_bias": False,
        "lyrics_rope_base": 12_345.0,
        "lyrics_rope_style": "half_split",
        "lyrics_norm_eps": 2.0e-5,
        "lyrics_norm_type": "rmsnorm",
        "lyrics_attention_direction": "bidirectional",
        "lyrics_norm_style": "pre_norm_with_final_norm",
        "lyrics_projection_order": "before_encoder",
        "lyrics_projection_bias": True,
        "null_context_init_std": 0.03,
        "text_drop_granularity": "sample",
        "text_drop_scope": "joint_description_lyrics",
        "null_context_tokens": 1,
        "null_context_layout": "preserve_text_mask",
        "text_context_composition": "concatenate",
        "text_context_layout": "description_then_lyrics",
        "text_context_separator": "none",
        "text_context_segment_embedding": "none",
        "text_compaction_policy": "stable_valid_tokens_right_padded",
        "initialization": "dit_xavier",
        "semantic_embedding_init_std": None,
        "semantic_embedding_asset": {"ready_path": "/asset/READY"},
        "semantic_projection_init": "latent_projection",
        "semantic_source_dim": 16,
        "semantic_projection_bias": True,
        "global_loudness_conditioning": False,
        "global_loudness_mean_lufs": -14.0,
        "global_loudness_std_lu": 5.0,
        "global_loudness_clamp_std": 4.0,
        "global_loudness_initialization_seed": 20260903,
        "dynamics_conditioning": False,
        "dynamics_checkpoint": None,
        "dynamics_checkpoint_sha256": None,
        "dynamics_checkpoint_step": 8250,
        "dynamics_target_kind": "relative_parent_lufs",
        "dynamics_freeze_predictor": True,
        "dynamics_include_activity": True,
        "dynamics_relative_scale_lu": 10.0,
        "dynamics_initialization_seed": 2026090304,
        "dropout": 0.125,
        "text_encoder": {
            "model_id": "fake/model",
            "hidden_size": 48,
            "local_path": "/models/fake",
            "use_fast_tokenizer": False,
            "padding_side": "right",
            "truncation_side": "left",
            "truncation_policy": "reject",
            "add_special_tokens": False,
            "trust_remote_code": True,
            "hidden_state_selection": "last_hidden_state",
            "position_id_policy": "attention_mask_cumsum",
            "encoder_use_cache": True,
            "empty_text_policy": "zero_valid_tokens",
            "frozen_eval_mode": False,
        },
    }
    result = build_render_conditioner_from_config(
        hidden_size=96,
        condition_config=condition_config,
        text_encoder_revision="encoder-revision",
        text_tokenizer_revision="tokenizer-revision",
        text_cache_revision="cache-revision",
        semantic_tokenizer_revision="semantic-revision",
        encoder=fake_encoder,
    )
    assert result is sentinel
    assert observed["adapter"] == {
        "model_id": "fake/model",
        "revision": "encoder-revision",
        "tokenizer_revision": "tokenizer-revision",
        "cache_revision": "cache-revision",
        "encoder": fake_encoder,
        "hidden_size": 48,
        "local_path": "/models/fake",
        "local_files_only": True,
        "use_fast_tokenizer": False,
        "padding_side": "right",
        "truncation_side": "left",
        "truncation_policy": "reject",
        "add_special_tokens": False,
        "trust_remote_code": True,
        "hidden_state_selection": "last_hidden_state",
        "position_id_policy": "attention_mask_cumsum",
        "encoder_use_cache": True,
        "empty_text_policy": "zero_valid_tokens",
        "frozen_eval_mode": False,
    }
    assert observed["conditioner"] == {
        "hidden_size": 96,
        "text_encoder": observed["conditioner"]["text_encoder"],
        "text_encoder_dim": 48,
        "semantic_vocab_size": 32_768,
        "description_max_tokens": 256,
        "description_projection_bias": True,
        "lyrics_max_tokens": 1_536,
        "lyrics_num_layers": 7,
        "lyrics_num_heads": 3,
        "lyrics_head_dim": 12,
        "lyrics_ffn_expansion": 5.0,
        "lyrics_ffn_activation": "swiglu",
        "lyrics_gelu_approximation": "none",
        "lyrics_qkv_bias": False,
        "lyrics_non_qkv_linear_bias": False,
        "lyrics_rope_base": 12_345.0,
        "lyrics_rope_style": "half_split",
        "lyrics_norm_eps": 2.0e-5,
        "lyrics_norm_type": "rmsnorm",
        "lyrics_attention_direction": "bidirectional",
        "lyrics_norm_style": "pre_norm_with_final_norm",
        "lyrics_projection_order": "before_encoder",
        "lyrics_projection_bias": True,
        "null_context_init_std": 0.03,
        "text_drop_granularity": "sample",
        "text_drop_scope": "joint_description_lyrics",
        "null_context_tokens": 1,
        "null_context_layout": "preserve_text_mask",
        "text_context_composition": "concatenate",
        "text_context_layout": "description_then_lyrics",
        "text_context_separator": "none",
        "text_context_segment_embedding": "none",
        "text_compaction_policy": "stable_valid_tokens_right_padded",
        "initialization": "dit_xavier",
        "semantic_embedding_init_std": None,
        "semantic_embedding_asset": {"ready_path": "/asset/READY"},
        "semantic_tokenizer_revision": "semantic-revision",
        "semantic_projection_init": "latent_projection",
        "semantic_source_dim": 16,
        "semantic_projection_bias": True,
        "global_loudness_conditioning": False,
        "global_loudness_mean_lufs": -14.0,
        "global_loudness_std_lu": 5.0,
        "global_loudness_clamp_std": 4.0,
        "global_loudness_initialization_seed": 20260903,
        "dynamics_conditioning": False,
        "dynamics_checkpoint": None,
        "dynamics_checkpoint_sha256": None,
        "dynamics_checkpoint_step": 8250,
        "dynamics_target_kind": "relative_parent_lufs",
        "dynamics_freeze_predictor": True,
        "dynamics_include_activity": True,
        "dynamics_relative_scale_lu": 10.0,
        "dynamics_initialization_seed": 2026090304,
        "dropout": 0.125,
    }
    assert isinstance(observed["conditioner"]["text_encoder"], CapturingAdapter)


def make_conditioner() -> tuple[RenderConditioner, FakeTextEncoder]:
    encoder = FakeTextEncoder()
    adapter = FrozenQwenEmbeddingAdapter(
        model_id="fake/qwen",
        revision="fake-model-v1",
        tokenizer_revision="fake-tokenizer-v1",
        cache_revision="fake-cache-v1",
        encoder=encoder,
        hidden_size=8,
    )
    conditioner = RenderConditioner(
        hidden_size=16,
        text_encoder=adapter,
        text_encoder_dim=8,
        lyrics_num_heads=2,
        lyrics_head_dim=8,
        lyrics_ffn_expansion=2,
    )
    return conditioner, encoder


def test_lyrics_encoder_independent_geometry_controls_are_explicit() -> None:
    encoder = FakeTextEncoder()
    adapter = FrozenQwenEmbeddingAdapter(
        model_id="fake/qwen",
        revision="fake-model-v1",
        tokenizer_revision="fake-tokenizer-v1",
        cache_revision="fake-cache-v1",
        encoder=encoder,
        hidden_size=8,
    )
    conditioner = RenderConditioner(
        hidden_size=16,
        text_encoder=adapter,
        text_encoder_dim=8,
        description_projection_bias=False,
        lyrics_num_heads=2,
        lyrics_head_dim=8,
        lyrics_ffn_expansion=2,
        lyrics_ffn_activation="gelu",
        lyrics_qkv_bias=False,
        lyrics_rope_base=2_000.0,
        lyrics_norm_eps=2.0e-5,
        lyrics_projection_bias=True,
        null_context_init_std=0.03,
    )
    first = conditioner.lyrics_encoder.layers[0]
    assert first.attention.qkv.bias is None
    assert first.attention.rope.base == 2_000.0
    assert first.attention_norm.eps == 2.0e-5
    assert first.ffn_norm.eps == 2.0e-5
    assert conditioner.lyrics_projection.bias is not None
    assert conditioner.description_projection.bias is None


def test_lyrics_swiglu_and_no_linear_bias_are_real_candidates() -> None:
    encoder = LyricsRoPEEncoder(
        hidden_size=16,
        num_layers=6,
        num_heads=2,
        head_dim=8,
        ffn_expansion=2,
        ffn_activation="swiglu",
        gelu_approximation="none",
        qkv_bias=False,
        non_qkv_linear_bias=False,
    )
    first = encoder.layers[0]
    assert first.attention.qkv.bias is None
    assert first.attention.output.bias is None
    assert first.ffn[0].out_features == 64
    assert isinstance(first.ffn[1], LyricsSwiGLU)
    assert first.ffn[0].bias is None
    assert first.ffn[3].bias is None
    hidden = torch.randn(2, 4, 16, requires_grad=True)
    mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]]
    )
    output = encoder(hidden, mask)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_lyrics_rmsnorm_is_a_real_trainable_candidate() -> None:
    encoder = LyricsRoPEEncoder(
        hidden_size=16,
        num_layers=6,
        num_heads=2,
        head_dim=8,
        ffn_expansion=2,
        norm_type="rmsnorm",
    )
    first = encoder.layers[0]
    assert isinstance(first.attention_norm, nn.RMSNorm)
    assert isinstance(first.ffn_norm, nn.RMSNorm)
    assert isinstance(encoder.final_norm, nn.RMSNorm)
    assert first.attention_norm.weight.requires_grad
    hidden = torch.randn(2, 4, 16, requires_grad=True)
    mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]]
    )
    output = encoder(hidden, mask)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_text_drop_semantics_are_explicit_and_reject_drift() -> None:
    conditioner, _ = make_conditioner()
    assert conditioner.text_drop_granularity == "sample"
    assert conditioner.text_drop_scope == "joint_description_lyrics"
    assert conditioner.null_context_tokens == 1
    assert conditioner.null_context_layout == "single_token"
    dropped = conditioner(
        **inputs(),
        text_drop_mask=torch.tensor([True, False]),
    )
    assert dropped.text_mask[0].sum().item() == 1
    assert torch.equal(
        dropped.text_context[0, 0],
        conditioner.null_context,
    )

    encoder = FakeTextEncoder()
    adapter = FrozenQwenEmbeddingAdapter(
        model_id="fake/qwen",
        revision="fake-model-v1",
        tokenizer_revision="fake-tokenizer-v1",
        cache_revision="fake-cache-v1",
        encoder=encoder,
        hidden_size=8,
    )
    with pytest.raises(ValueError, match="text_drop_granularity"):
        RenderConditioner(
            hidden_size=16,
            text_encoder=adapter,
            text_encoder_dim=8,
            lyrics_num_heads=2,
            lyrics_head_dim=8,
            text_drop_granularity="token",
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"semantic_vocab_size": 16_384}, "semantic vocabulary must be 32768"),
        ({"lyrics_gelu_approximation": "fast"}, "gelu_approximation"),
        ({"lyrics_rope_style": "interleaved"}, "rope_style"),
        ({"lyrics_norm_type": "batchnorm"}, "norm_type"),
        ({"lyrics_attention_direction": "causal"}, "attention_direction"),
        ({"lyrics_norm_style": "post_norm"}, "norm_style"),
        ({"lyrics_projection_order": "after_encoder"}, "projection_order"),
        ({"text_context_composition": "sum"}, "composition"),
        ({"text_context_layout": "lyrics_then_description"}, "layout"),
        ({"text_context_separator": "learned"}, "separator"),
        ({"text_context_segment_embedding": "learned"}, "segment_embedding"),
        ({"null_context_layout": "unknown"}, "null_context_layout"),
    ],
)
def test_conditioner_rejects_unimplemented_context_semantics(
    kwargs: dict[str, object],
    match: str,
) -> None:
    encoder = FakeTextEncoder()
    adapter = FrozenQwenEmbeddingAdapter(
        model_id="fake/qwen",
        revision="fake-model-v1",
        tokenizer_revision="fake-tokenizer-v1",
        cache_revision="fake-cache-v1",
        encoder=encoder,
        hidden_size=8,
    )
    with pytest.raises(ValueError, match=match):
        RenderConditioner(
            hidden_size=16,
            text_encoder=adapter,
            text_encoder_dim=8,
            lyrics_num_heads=2,
            lyrics_head_dim=8,
            **kwargs,
        )


def test_dit_xavier_conditioner_initialization_is_explicit_and_rng_isolated() -> None:
    encoder = FakeTextEncoder()
    adapter = FrozenQwenEmbeddingAdapter(
        model_id="fake/qwen",
        revision="fake-model-v1",
        tokenizer_revision="fake-tokenizer-v1",
        cache_revision="fake-cache-v1",
        encoder=encoder,
        hidden_size=8,
    )
    torch.manual_seed(123)
    conditioner = RenderConditioner(
        hidden_size=16,
        text_encoder=adapter,
        text_encoder_dim=8,
        lyrics_num_heads=2,
        lyrics_head_dim=8,
        initialization="dit_xavier",
    )
    assert conditioner.initialization == "dit_xavier"
    assert torch.count_nonzero(conditioner.description_projection.weight) > 0
    assert torch.count_nonzero(conditioner.lyrics_projection.weight) > 0
    first = conditioner.lyrics_encoder.layers[0]
    assert torch.count_nonzero(first.attention.qkv.weight) > 0
    if first.attention.qkv.bias is not None:
        assert torch.count_nonzero(first.attention.qkv.bias) == 0
    assert torch.all(first.attention_norm.weight == 1)
    assert torch.count_nonzero(first.attention_norm.bias) == 0

    with pytest.raises(ValueError, match="initialization"):
        RenderConditioner(
            hidden_size=16,
            text_encoder=adapter,
            text_encoder_dim=8,
            lyrics_num_heads=2,
            lyrics_head_dim=8,
            initialization="implicit_default",
        )


def inputs() -> dict[str, torch.Tensor]:
    return {
        "semantic_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        "semantic_mask": torch.tensor(
            [[True, True, True, False], [True, True, False, False]]
        ),
        "description_input_ids": torch.tensor([[1, 2, 0], [3, 0, 0]]),
        "description_mask": torch.tensor(
            [[True, True, False], [True, False, False]]
        ),
        "lyrics_input_ids": torch.tensor(
            [[4, 5, 6, 0], [7, 8, 0, 0]]
        ),
        "lyrics_mask": torch.tensor(
            [[True, True, True, False], [True, True, False, False]]
        ),
    }


def test_conditioning_concat_mask_and_contract() -> None:
    conditioner, _ = make_conditioner()
    output = conditioner(**inputs())
    assert output.semantic_embeddings.shape == (2, 4, 16)
    assert output.text_context.shape == (2, 7, 16)
    assert output.text_mask.shape == (2, 7)
    assert torch.count_nonzero(output.semantic_embeddings[0, 3]) == 0
    assert torch.count_nonzero(output.text_context[0, 2]) == 0
    assert conditioner.description_projection.bias is None
    assert len(conditioner.lyrics_encoder.layers) == 6


def test_conditioner_compacts_left_padding_before_trainable_text_layers() -> None:
    conditioner, _ = make_conditioner()
    values = inputs()
    description = torch.randn(2, 4, 8)
    lyrics = torch.randn(2, 5, 8)
    description_mask = torch.tensor(
        [[False, False, True, True], [False, True, True, True]]
    )
    lyrics_mask = torch.tensor(
        [[False, False, True, True, True], [False, True, True, True, True]]
    )
    description[~description_mask] = 0.0
    lyrics[~lyrics_mask] = 0.0
    output = conditioner(
        values["semantic_ids"],
        values["semantic_mask"],
        description_embeddings=description,
        description_mask=description_mask,
        lyrics_embeddings=lyrics,
        lyrics_mask=lyrics_mask,
        cache_provenance=conditioner.provenance,
    )
    assert output.text_mask[0].tolist() == [
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
    ]
    assert output.text_mask[1].tolist() == [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
    ]


def test_qwen_position_ids_make_left_padded_batch_match_single_item_encoding() -> None:
    encoder = PositionAwareFakeTextEncoder()
    adapter = FrozenQwenEmbeddingAdapter(
        model_id="fake/qwen",
        revision="fake-model-v1",
        tokenizer_revision="fake-tokenizer-v1",
        cache_revision="fake-cache-v1",
        encoder=encoder,
        hidden_size=2,
    )
    short_ids = torch.tensor([[0, 0, 11, 12]], dtype=torch.long)
    batched_ids = torch.tensor(
        [[0, 0, 11, 12], [21, 22, 23, 24]],
        dtype=torch.long,
    )
    batched_mask = torch.tensor(
        [[False, False, True, True], [True, True, True, True]]
    )
    single = adapter(short_ids[:, 2:], torch.ones(1, 2, dtype=torch.bool))
    batched = adapter(batched_ids, batched_mask)
    assert torch.equal(single[0], batched[0, 2:])
    assert encoder.position_ids[0].tolist() == [[0, 1]]
    assert encoder.position_ids[1].tolist() == [[0, 0, 0, 1], [0, 1, 2, 3]]


def test_tokenizer_reject_policy_reports_oversized_input_explicitly() -> None:
    adapter = FrozenQwenEmbeddingAdapter(
        model_id="fake/qwen",
        revision="fake-model-v1",
        tokenizer_revision="fake-tokenizer-v1",
        cache_revision="fake-cache-v1",
        encoder=FakeTextEncoder(),
        tokenizer=NoTruncationTokenizer(),
        hidden_size=8,
        truncation_policy="reject",
    )
    with pytest.raises(ValueError, match="token count exceeds max_length"):
        adapter.tokenize(["too long"], max_length=4)


def test_lyrics_attention_arbitrary_mask_matches_position_preserving_reference() -> None:
    torch.manual_seed(37)
    attention = LyricsSelfAttention(
        hidden_size=16,
        num_heads=2,
        head_dim=8,
        dropout=0.0,
    ).double()
    hidden = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor(
        [[True, False, True, False, False], [False, True, True, False, True]]
    )
    output = attention(hidden, mask)

    query, key, value = attention.qkv(hidden).chunk(3, dim=-1)
    shape = (2, 5, attention.num_heads, attention.head_dim)
    query = query.view(shape).transpose(1, 2)
    key = key.view(shape).transpose(1, 2)
    value = value.view(shape).transpose(1, 2)
    cosine, sine = attention.rope(
        5,
        device=hidden.device,
        dtype=query.dtype,
    )
    from open_qwen_music.render.conditioning import apply_rope

    query = apply_rope(query, cosine, sine)
    key = apply_rope(key, cosine, sine)
    reference_attention = torch.zeros_like(query)
    for batch_index in range(2):
        valid = mask[batch_index]
        reference_attention[batch_index : batch_index + 1, :, valid] = (
            torch.nn.functional.scaled_dot_product_attention(
                query[batch_index : batch_index + 1, :, valid],
                key[batch_index : batch_index + 1, :, valid],
                value[batch_index : batch_index + 1, :, valid],
                dropout_p=0.0,
            )
        )
    reference = attention.output(
        reference_attention.transpose(1, 2).reshape(2, 5, attention.inner_dim)
    ).masked_fill(~mask.unsqueeze(-1), 0.0)
    assert torch.allclose(output, reference, atol=1.0e-10, rtol=1.0e-10)
    assert torch.count_nonzero(output[~mask]) == 0
    output.square().sum().backward()
    assert torch.isfinite(hidden.grad).all()


def test_text_drop_only_replaces_text_and_all_trainable_paths_get_gradient() -> None:
    conditioner, encoder = make_conditioner()
    values = inputs()
    conditional = conditioner(**values)
    dropped = conditioner(
        **values, text_drop_mask=torch.tensor([False, True])
    )
    assert torch.equal(
        conditional.semantic_embeddings, dropped.semantic_embeddings
    )
    assert dropped.text_mask[1, 0]
    assert dropped.text_mask[1].sum() == 1

    loss = (
        dropped.semantic_embeddings.sum()
        + dropped.text_context[0].sum()
        + dropped.text_context[1, 0].sum()
    )
    loss.backward()
    assert conditioner.semantic_embedding.weight.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in conditioner.lyrics_encoder.parameters()
    )
    assert conditioner.null_context.grad is not None
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_preserve_text_mask_null_repeats_learned_value_on_valid_tokens() -> None:
    conditioner, _ = make_conditioner()
    conditioner.null_context_layout = "preserve_text_mask"
    values = inputs()
    conditional = conditioner(**values)
    dropped = conditioner(
        **values,
        text_drop_mask=torch.tensor([True, True]),
    )
    assert torch.equal(dropped.text_mask, conditional.text_mask)
    expected = conditioner.null_context.view(1, 1, -1).expand_as(
        dropped.text_context
    )
    assert torch.equal(
        dropped.text_context[dropped.text_mask],
        expected[dropped.text_mask],
    )
    assert torch.count_nonzero(dropped.text_context[~dropped.text_mask]) == 0


def test_preserve_text_mask_null_inference_requires_and_uses_text_mask() -> None:
    conditioner, _ = make_conditioner()
    conditioner.null_context_layout = "preserve_text_mask"
    values = inputs()
    conditional = conditioner(**values)
    with pytest.raises(ValueError, match="text_mask"):
        conditioner.null_from_semantic(
            conditional.semantic_embeddings,
            conditional.semantic_mask,
        )
    null = conditioner.null_from_semantic(
        conditional.semantic_embeddings,
        conditional.semantic_mask,
        text_mask=conditional.text_mask,
    )
    assert torch.equal(null.text_mask, conditional.text_mask)
    expected = conditioner.null_context.view(1, 1, -1).expand_as(
        null.text_context
    )
    assert torch.equal(
        null.text_context[null.text_mask],
        expected[null.text_mask],
    )


def test_cache_revision_is_strict() -> None:
    conditioner, _ = make_conditioner()
    expected = conditioner.provenance
    validate_cache_provenance(expected.to_dict(), expected)
    wrong = expected.to_dict()
    wrong["cache_revision"] = "other-cache-v2"
    with pytest.raises(RuntimeError, match="revision"):
        validate_cache_provenance(wrong, expected)

    values = inputs()
    fake_embeddings = torch.randn(2, 3, 8)
    with pytest.raises(RuntimeError, match="missing revision metadata"):
        conditioner(
            values["semantic_ids"],
            values["semantic_mask"],
            description_embeddings=fake_embeddings,
            description_mask=values["description_mask"],
            lyrics_input_ids=values["lyrics_input_ids"],
            lyrics_mask=values["lyrics_mask"],
        )


def test_cached_text_is_exclusive_finite_and_zero_padded() -> None:
    conditioner, _ = make_conditioner()
    values = inputs()
    cached = torch.randn(2, 3, 8)
    cached[:, -1] = 0.0
    mask = torch.tensor([[True, True, False], [True, True, False]])
    with pytest.raises(ValueError, match="provide exactly one"):
        conditioner(
            values["semantic_ids"],
            values["semantic_mask"],
            description_input_ids=values["description_input_ids"],
            description_embeddings=cached,
            description_mask=mask,
            lyrics_input_ids=values["lyrics_input_ids"],
            lyrics_mask=values["lyrics_mask"],
            cache_provenance=conditioner.provenance,
        )
    nonfinite = cached.clone()
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN/Inf"):
        conditioner(
            values["semantic_ids"],
            values["semantic_mask"],
            description_embeddings=nonfinite,
            description_mask=mask,
            lyrics_input_ids=values["lyrics_input_ids"],
            lyrics_mask=values["lyrics_mask"],
            cache_provenance=conditioner.provenance,
        )
    nonzero_padding = cached.clone()
    nonzero_padding[0, -1, 0] = 1.0
    with pytest.raises(ValueError, match="padding"):
        conditioner(
            values["semantic_ids"],
            values["semantic_mask"],
            description_embeddings=nonzero_padding,
            description_mask=mask,
            lyrics_input_ids=values["lyrics_input_ids"],
            lyrics_mask=values["lyrics_mask"],
            cache_provenance=conditioner.provenance,
        )


def test_empty_text_mask_reports_effective_null_drop() -> None:
    conditioner, _ = make_conditioner()
    values = inputs()
    values["description_mask"] = torch.zeros_like(values["description_mask"])
    values["lyrics_mask"] = torch.zeros_like(values["lyrics_mask"])
    output = conditioner(**values)
    assert output.text_drop_mask.tolist() == [True, True]
    assert output.text_mask[:, 0].all()
    assert output.text_mask.sum(dim=1).tolist() == [1, 1]


def test_cfg_null_text_dtype_does_not_follow_semantic_dtype() -> None:
    conditioner, _ = make_conditioner()
    semantic = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    semantic_mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )

    output = conditioner.null_from_semantic(semantic, semantic_mask)

    assert output.semantic_embeddings.dtype == torch.bfloat16
    assert output.text_context.dtype == conditioner.null_context.dtype
    assert output.text_context.dtype == torch.float32


@pytest.mark.parametrize(
    "semantic_mask",
    [
        torch.tensor([[False, False, False, False], [True, True, False, False]]),
        torch.tensor([[True, False, True, False], [True, True, False, False]]),
    ],
)
def test_conditioner_rejects_invalid_semantic_masks(
    semantic_mask: torch.Tensor,
) -> None:
    conditioner, _ = make_conditioner()
    values = inputs()
    values["semantic_mask"] = semantic_mask
    with pytest.raises(ValueError, match="semantic"):
        conditioner(**values)

    semantic_embeddings = conditioner.semantic_embedding(values["semantic_ids"])
    with pytest.raises(ValueError, match="semantic"):
        conditioner.null_from_semantic(semantic_embeddings, semantic_mask)


def test_invalid_ids_lengths_and_floating_revision_fail() -> None:
    conditioner, _ = make_conditioner()
    values = inputs()
    values["semantic_ids"] = values["semantic_ids"].clone()
    values["semantic_ids"][0, -1] = -1
    with pytest.raises(ValueError, match="0..32767"):
        conditioner(**values)

    values = inputs()
    values["description_input_ids"] = torch.zeros(2, 257, dtype=torch.long)
    values["description_mask"] = torch.ones(2, 257, dtype=torch.bool)
    with pytest.raises(ValueError, match="length 257 exceeds the limit 256"):
        conditioner(**values)

    with pytest.raises(ValueError, match="must be a pinned, traceable revision"):
        FrozenQwenEmbeddingAdapter(
            model_id="fake/qwen",
            revision="main",
            tokenizer_revision="fake-tokenizer-v1",
            cache_revision="fake-cache-v1",
            encoder=FakeTextEncoder(),
        )


def test_text_provenance_fingerprint_is_deterministic() -> None:
    value = TextEncoderProvenance(
        "fake/qwen", "model-v1", "tokenizer-v1", "cache-v1"
    )
    assert value.fingerprint == value.fingerprint


def test_text_provenance_mapping_is_strict_and_rejects_alias_conflicts() -> None:
    canonical = {
        "model_id": "fake/qwen",
        "model_revision": "model-v1",
        "tokenizer_revision": "tokenizer-v1",
        "cache_revision": "cache-v1",
    }
    assert TextEncoderProvenance.from_mapping(canonical).to_dict() == canonical
    assert (
        TextEncoderProvenance.from_mapping(
            {
                "model_id": "fake/qwen",
                "text_encoder_revision": "model-v1",
                "text_tokenizer_revision": "tokenizer-v1",
                "text_cache_revision": "cache-v1",
            }
        ).to_dict()
        == canonical
    )

    with pytest.raises(TypeError, match="model_revision must be a string"):
        TextEncoderProvenance.from_mapping(
            {**canonical, "model_revision": 7}
        )
    with pytest.raises(ValueError, match="aliases for model_revision conflict"):
        TextEncoderProvenance.from_mapping(
            {**canonical, "text_encoder_revision": "model-v2"}
        )
    with pytest.raises(ValueError, match="contains unknown fields"):
        TextEncoderProvenance.from_mapping({**canonical, "unexpected": "value"})
