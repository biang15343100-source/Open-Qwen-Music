
from __future__ import annotations

import pytest
import torch

from open_qwen_music.tokenizer.conformer import (
    ConformerEncoder,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
    apply_rotary,
)

DIM = 64
HEADS = 4
HEAD_DIM = DIM // HEADS


def _encoder(position_encoding: str = "rope") -> ConformerEncoder:
    torch.manual_seed(0)
    return ConformerEncoder(
        dim=DIM,
        num_layers=2,
        num_heads=HEADS,
        ffn_dim=128,
        conv_kernel=5,
        dropout=0.0,
        position_encoding=position_encoding,
    ).eval()


def test_rope_score_depends_only_on_relative_distance() -> None:

    torch.manual_seed(3)
    rotary = RotaryPositionalEmbedding(HEAD_DIM)
    cos, sin = rotary(512, torch.device("cpu"), torch.float32)
    query = torch.randn(1, 1, 1, HEAD_DIM)
    key = torch.randn(1, 1, 1, HEAD_DIM)

    def score(query_position: int, key_position: int) -> float:
        rotated_query = apply_rotary(
            query, cos[query_position : query_position + 1], sin[query_position : query_position + 1]
        )
        rotated_key = apply_rotary(
            key, cos[key_position : key_position + 1], sin[key_position : key_position + 1]
        )
        return float((rotated_query * rotated_key).sum())

    baseline = score(10, 4)
    for shift in (1, 37, 200, 400):
        torch.testing.assert_close(
            score(10 + shift, 4 + shift), baseline, rtol=1e-4, atol=1e-4
        )


def test_absolute_encoding_is_not_translation_invariant() -> None:

    encoding = SinusoidalPositionalEncoding(DIM)
    content = torch.randn(1, 8, DIM)
    at_start = encoding(torch.cat([content, torch.zeros(1, 200, DIM)], dim=1))[:, :8]
    at_offset = encoding(torch.cat([torch.zeros(1, 200, DIM), content], dim=1))[:, 200:]
    difference = (at_start - at_offset).abs().mean()
    assert float(difference) > 0.1, "The control group should reflect the translational sensitivity of absolute coding"


def test_rope_rejects_odd_head_dim() -> None:
    with pytest.raises(ValueError, match="even head_dim"):
        RotaryPositionalEmbedding(15)


def test_encoder_rejects_unknown_position_encoding() -> None:
    with pytest.raises(ValueError, match="position_encoding"):
        _encoder(position_encoding="alibi")


@pytest.mark.parametrize("causal", [True, False])
def test_output_is_independent_of_padding_amount(causal: bool) -> None:

    torch.manual_seed(11)
    encoder = _encoder()
    valid = 24
    content = torch.randn(1, valid, DIM)

    def run(total: int) -> torch.Tensor:
        padded = torch.cat([content, torch.randn(1, total - valid, DIM)], dim=1)
        mask = torch.zeros(1, total, dtype=torch.bool)
        mask[:, :valid] = True
        with torch.no_grad():
            return encoder.forward_range(padded, mask, causal=causal)[:, :valid]

    torch.testing.assert_close(run(valid), run(valid + 40), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(run(valid + 7), run(valid + 40), rtol=1e-4, atol=1e-4)


def test_causal_encoder_ignores_future_frames() -> None:

    torch.manual_seed(13)
    encoder = _encoder()
    length = 40
    original = torch.randn(1, length, DIM)
    changed = original.clone()
    changed[:, length // 2 :] = torch.randn(1, length - length // 2, DIM)
    mask = torch.ones(1, length, dtype=torch.bool)
    with torch.no_grad():
        first = encoder.forward_range(original, mask, causal=True)
        second = encoder.forward_range(changed, mask, causal=True)
    torch.testing.assert_close(
        first[:, : length // 2], second[:, : length // 2], rtol=1e-4, atol=1e-4
    )


def test_causal_prefix_is_stable_when_sequence_grows() -> None:

    torch.manual_seed(17)
    encoder = _encoder()
    prefix = torch.randn(1, 32, DIM)

    def run(total: int) -> torch.Tensor:
        sequence = torch.cat([prefix, torch.randn(1, total - 32, DIM)], dim=1)
        mask = torch.ones(1, total, dtype=torch.bool)
        with torch.no_grad():
            return encoder.forward_range(sequence, mask, causal=True)[:, :32]

    torch.testing.assert_close(run(32), run(256), rtol=1e-4, atol=1e-4)


def test_attention_parameter_layout_matches_multihead_attention() -> None:

    encoder = _encoder()
    names = dict(encoder.layers[0].attn.named_parameters())
    assert names["in_proj_weight"].shape == (3 * DIM, DIM)
    assert names["in_proj_bias"].shape == (3 * DIM,)
    assert names["out_proj.weight"].shape == (DIM, DIM)
    reference = torch.nn.MultiheadAttention(DIM, HEADS, batch_first=True)
    assert set(dict(reference.named_parameters())) == set(names)


def test_rope_buffers_do_not_enter_state_dict() -> None:

    encoder = _encoder()
    assert not [key for key in encoder.state_dict() if "inverse_frequency" in key]


def test_sinusoidal_branch_still_available() -> None:

    encoder = _encoder(position_encoding="sinusoidal")
    assert all(not layer.use_rotary for layer in encoder.layers)
    length = 16
    x = torch.randn(1, length, DIM)
    mask = torch.ones(1, length, dtype=torch.bool)
    with torch.no_grad():
        output = encoder.forward_range(x, mask, causal=True)
    assert output.shape == (1, length, DIM)
