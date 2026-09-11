from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import open_qwen_music.tokenizer.heads as heads_module
import open_qwen_music.tokenizer.model as model_module
from open_qwen_music.tokenizer.heads import (
    MultiTaskHeads,
    distributed_weighted_mean,
    spectral_loss,
)
from open_qwen_music.tokenizer.bestrq import (
    make_waveform_span_mask,
    waveform_mask_to_feature_mask,
)
from open_qwen_music.tokenizer.metrics import (
    collapse_ctc,
    ctc_prefix_beam_search,
    edit_distance,
    edit_operations,
)
from open_qwen_music.tokenizer.model import distributed_quantizer_frame_mean
from open_qwen_music.tokenizer.trainer import _reduce_training_metrics
from open_qwen_music.tokenizer.text import CharacterTokenizer


def test_ctc_collapse_and_phoneme_units() -> None:
    tokenizer = CharacterTokenizer(["<blank>", "<unk>", "m", "ei", "AY1"])
    assert tokenizer.encode_units(["m", "ei", "missing"]) == [2, 3, 1]
    assert tokenizer.encode_units([2, 3, 4]) == [2, 3, 4]
    with pytest.raises(ValueError, match="blank"):
        tokenizer.encode_units([0])
    with pytest.raises(ValueError, match="out of bounds"):
        tokenizer.encode_units([5])
    with pytest.raises(TypeError, match="bool"):
        tokenizer.encode_units([True])
    assert collapse_ctc([0, 2, 2, 0, 2, 3, 3, 0]) == [2, 2, 3]
    assert tokenizer.decode_units([2, 4, 99]) == ["m", "AY1", "<invalid:99>"]


def test_edit_distance() -> None:
    assert edit_distance(["m", "ei"], ["m", "ei"]) == 0
    assert edit_distance(["m", "ei"], ["m"]) == 1
    assert edit_distance(["m"], ["m", "ei"]) == 1
    assert edit_distance(["m", "ei"], ["n", "ai"]) == 2


def test_edit_operations_decompose_distance() -> None:
    cases = [
        (["a", "b"], ["a", "b"], (0, 0, 0)),
        (["a", "b"], ["a", "c"], (1, 0, 0)),
        (["a", "b"], ["a"], (0, 1, 0)),
        (["a"], ["a", "b"], (0, 0, 1)),
        (["a", "b", "c"], ["a", "x", "c", "d"], (1, 0, 1)),
    ]
    for reference, hypothesis, expected in cases:
        operations = edit_operations(reference, hypothesis)
        observed = (
            operations["substitutions"],
            operations["deletions"],
            operations["insertions"],
        )
        assert observed == expected
        assert operations["edits"] == edit_distance(reference, hypothesis)


def test_ctc_prefix_beam_can_beat_frame_greedy() -> None:

    probabilities = torch.tensor(
        [
            [0.51, 0.49, 1e-6],
            [0.51, 0.49, 1e-6],
        ]
    )
    log_probs = probabilities.log()
    assert collapse_ctc(log_probs.argmax(dim=-1).tolist()) == []
    assert ctc_prefix_beam_search(log_probs, beam_size=4, token_topk=3) == [1]


def test_ctc_loss_uses_standard_target_length_normalization() -> None:
    heads = MultiTaskHeads(model_dim=2, ctc_vocab_size=4)
    logits = torch.tensor(
        [
            [[2.0, 1.0, 0.0, -1.0], [1.0, 2.0, 0.0, -1.0], [2.0, 0.0, 1.0, -1.0]],
            [[2.0, 0.0, 1.0, -1.0], [1.0, 0.0, 2.0, -1.0], [2.0, 1.0, 0.0, -1.0]],
        ]
    )
    targets = torch.tensor([[1, 0], [2, 1]])
    lengths = torch.tensor([1, 2])
    weights = torch.tensor([1.0, 3.0])
    predictions = {
        "ctc_logits": logits,
        "mel": torch.zeros(2, 1, 128),
        "chroma": torch.zeros(2, 1, 12),
    }
    batch = {
        "ctc_enabled": torch.tensor([True, True]),
        "lyrics_token_ids": targets,
        "lyrics_lengths": lengths,

        "sample_weights": torch.tensor([9.0, 9.0]),
        "ctc_sample_weights": weights,
        "ctc_alignment_enabled": torch.tensor([False, False]),
        "ctc_frame_target_mask": torch.zeros(2, 3, dtype=torch.bool),
        "ctc_frame_token_ids": torch.zeros(2, 3, dtype=torch.long),
        "mel_target": torch.zeros(2, 1, 128),
        "mel_target_mask": torch.zeros(2, 1, dtype=torch.bool),
        "mel_enabled": torch.tensor([False, False]),
        "chroma_target": torch.zeros(2, 1, 12),
        "chroma_target_mask": torch.zeros(2, 1, dtype=torch.bool),
        "chroma_enabled": torch.tensor([False, False]),
    }
    losses, _ = heads.compute_losses(
        predictions, torch.tensor([3, 3]), batch
    )
    raw = F.ctc_loss(
        logits.log_softmax(-1).transpose(0, 1),
        torch.tensor([1, 2, 1]),
        torch.tensor([3, 3]),
        lengths,
        blank=0,
        reduction="none",
    )
    expected = ((raw / lengths) * weights).sum() / weights.sum()
    torch.testing.assert_close(losses["ctc"], expected)


def test_mel_and_chroma_use_their_own_sample_weights() -> None:
    heads = MultiTaskHeads(model_dim=2, ctc_vocab_size=4)
    mel_target = torch.ones(2, 1, 2)
    chroma_target = torch.ones(2, 1, 2)
    mel_prediction = mel_target + torch.tensor([1.0, 3.0]).view(2, 1, 1)
    chroma_prediction = chroma_target + torch.tensor([2.0, 4.0]).view(
        2, 1, 1
    )
    predictions = {
        "ctc_logits": torch.zeros(2, 1, 4),
        "mel": mel_prediction,
        "chroma": chroma_prediction,
    }
    batch = {
        "ctc_enabled": torch.tensor([False, False]),
        "lyrics_token_ids": torch.zeros(2, 1, dtype=torch.long),
        "lyrics_lengths": torch.zeros(2, dtype=torch.long),
        "sample_weights": torch.tensor([9.0, 9.0]),
        "ctc_sample_weights": torch.zeros(2),
        "mel_sample_weights": torch.tensor([1.0, 0.0]),
        "chroma_sample_weights": torch.tensor([0.0, 1.0]),
        "ctc_alignment_enabled": torch.tensor([False, False]),
        "ctc_frame_target_mask": torch.zeros(2, 1, dtype=torch.bool),
        "ctc_frame_token_ids": torch.zeros(2, 1, dtype=torch.long),
        "mel_target": mel_target,
        "mel_target_mask": torch.ones(2, 1, dtype=torch.bool),
        "mel_enabled": torch.tensor([True, True]),
        "chroma_target": chroma_target,
        "chroma_target_mask": torch.ones(2, 1, dtype=torch.bool),
        "chroma_enabled": torch.tensor([True, True]),
    }

    losses, _ = heads.compute_losses(
        predictions,
        torch.ones(2, dtype=torch.long),
        batch,
    )
    expected_mel, _, _ = spectral_loss(
        mel_prediction[:1],
        mel_target[:1],
        torch.ones(1, 1, dtype=torch.bool),
    )
    expected_chroma, _, _ = spectral_loss(
        chroma_prediction[1:],
        chroma_target[1:],
        torch.ones(1, 1, dtype=torch.bool),
    )
    torch.testing.assert_close(losses["mel"], expected_mel)
    torch.testing.assert_close(losses["chroma"], expected_chroma)


def test_zero_acoustic_head_weights_keep_zero_gradient_losses() -> None:
    prediction = torch.randn(2, 3, 4, requires_grad=True)
    loss, _, _ = spectral_loss(
        prediction,
        torch.ones_like(prediction),
        torch.ones(2, 3, dtype=torch.bool),
        sample_weights=torch.zeros(2),
    )
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert prediction.grad is not None
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_distributed_weighted_mean_compensates_for_ddp_gradient_average(
    monkeypatch,
) -> None:
    monkeypatch.setattr(heads_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(heads_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(heads_module, "reduce_scalar_sum", lambda _value: 4.0)
    numerator = torch.tensor(2.0, requires_grad=True)
    loss = distributed_weighted_mean(numerator, torch.tensor(1.0))

    assert loss.item() == 1.0
    loss.backward()
    assert numerator.grad.item() == 0.5


def test_bestrq_global_frame_loss_matches_concatenated_gradient(
    monkeypatch,
) -> None:

    monkeypatch.setattr(heads_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(heads_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(heads_module, "reduce_scalar_sum", lambda _value: 13.0)

    logits0 = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0]] * 5,
        requires_grad=True,
    )
    targets0 = torch.tensor([0, 1] * 5)
    logits1 = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]],
        requires_grad=True,
    )
    targets1 = torch.tensor([0, 1, 1])

    numerator0 = F.cross_entropy(logits0, targets0, reduction="sum")
    numerator1 = F.cross_entropy(logits1, targets1, reduction="sum")
    rank0_loss = distributed_weighted_mean(numerator0, torch.tensor(10.0))
    rank1_loss = distributed_weighted_mean(numerator1, torch.tensor(3.0))

    ((rank0_loss + rank1_loss) / 2.0).backward()

    concatenated = torch.cat(
        [logits0.detach(), logits1.detach()]
    ).requires_grad_(True)
    expected = F.cross_entropy(
        concatenated,
        torch.cat([targets0, targets1]),
    )
    expected.backward()

    torch.testing.assert_close(logits0.grad, concatenated.grad[:10])
    torch.testing.assert_close(logits1.grad, concatenated.grad[10:])


def test_bestrq_training_metrics_use_global_frame_denominator(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "open_qwen_music.tokenizer.trainer.reduce_metrics",
        lambda _metrics: {

            "bestrq_loss_sum": 16.25,
            "bestrq_correct_frames": 3.25,
            "bestrq_active_frames": 6.5,
            "bestrq_loss": 2.0,
            "bestrq_accuracy": 0.25,
            "loss": 2.0,
        },
    )
    reduced = _reduce_training_metrics({})
    assert reduced["bestrq_loss"] == pytest.approx(2.5)
    assert reduced["bestrq_accuracy"] == pytest.approx(0.5)
    assert reduced["loss"] == pytest.approx(2.5)


def test_distributed_quantizer_loss_uses_global_effective_weight_mean(
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(model_module.dist, "get_world_size", lambda: 2)

    monkeypatch.setattr(heads_module, "reduce_scalar_sum", lambda _value: 2.0)
    rank0_mean = torch.tensor(2.0, requires_grad=True)
    rank1_mean = torch.tensor(4.0, requires_grad=True)
    rank0 = distributed_quantizer_frame_mean(
        rank0_mean,
        torch.tensor(0.5),
        diversity_beta=0.0,
    )
    rank1 = distributed_quantizer_frame_mean(
        rank1_mean,
        torch.tensor(1.5),
        diversity_beta=0.0,
    )

    global_mean = (rank0 + rank1) / 2.0
    assert float(global_mean.detach()) == pytest.approx(3.5)
    global_mean.backward()
    assert float(rank0_mean.grad) == pytest.approx(0.25)
    assert float(rank1_mean.grad) == pytest.approx(0.75)


def test_distributed_frame_mean_rejects_nonlinear_diversity_scalar(
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(model_module.dist, "get_world_size", lambda: 2)
    with pytest.raises(RuntimeError, match="diversity_beta"):
        distributed_quantizer_frame_mean(
            torch.tensor(1.0),
            torch.tensor(2.0),
            diversity_beta=1.0,
        )


def test_waveform_span_mask_maps_to_100hz_frames() -> None:
    torch.manual_seed(7)
    mask = make_waveform_span_mask(
        torch.tensor([24_000]),
        24_000,
        span_samples=9_600,
        probability=0.0,
    )

    assert int(mask.sum()) in {4_800, 9_600}
    frames = waveform_mask_to_feature_mask(
        mask, hop_length=240, target_length=100
    )
    assert frames.shape == (1, 100)
    assert frames.any()
