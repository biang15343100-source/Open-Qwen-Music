from __future__ import annotations

import torch

from open_qwen_music.tokenizer.melody import MelodyTokenizer


def test_melody_contract_and_unvoiced() -> None:
    tokenizer = MelodyTokenizer()
    pitch = torch.tensor([[440.0] * 8 + [0.0] * 8 + [880.0] * 4])
    result = tokenizer(pitch)
    assert result.token_ids.shape == (1, 3)
    assert result.frame_mask.all()
    assert result.token_ids[0, 1].item() == 255
    assert result.token_ids.min() >= 0
    assert result.token_ids.max() < 256


def test_melody_is_transposition_invariant() -> None:
    tokenizer = MelodyTokenizer()
    pitch = torch.tensor([[220.0] * 8 + [440.0] * 8 + [330.0] * 8])
    first = tokenizer(pitch).token_ids
    second = tokenizer(pitch * 2.0).token_ids
    torch.testing.assert_close(first, second)


def test_melody_padding_mask() -> None:
    tokenizer = MelodyTokenizer()
    pitch = torch.tensor([[440.0] * 10 + [0.0] * 6])
    mask = torch.tensor([[True] * 10 + [False] * 6])
    result = tokenizer(pitch, mask)
    assert result.frame_mask.tolist() == [[True, True]]
    assert result.token_ids.shape == (1, 2)


def test_pooled_frame_needs_majority_voiced() -> None:

    tokenizer = MelodyTokenizer()

    def token_for(voiced_count: int) -> int:
        pitch = torch.zeros(1, 8)
        pitch[0, :voiced_count] = 440.0
        return int(tokenizer(pitch).token_ids[0, 0])

    assert token_for(1) == tokenizer.unvoiced_id
    assert token_for(4) == tokenizer.unvoiced_id
    assert token_for(5) != tokenizer.unvoiced_id
    assert token_for(8) != tokenizer.unvoiced_id


def test_majority_uses_valid_frames_as_denominator() -> None:

    tokenizer = MelodyTokenizer()
    pitch = torch.zeros(1, 11)
    pitch[0, :8] = 440.0
    pitch[0, 8:] = 440.0
    mask = torch.ones(1, 11, dtype=torch.bool)
    tokens = tokenizer(pitch, mask).token_ids
    assert int(tokens[0, 1]) != tokenizer.unvoiced_id
