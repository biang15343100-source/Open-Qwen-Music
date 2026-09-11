import json
import random
import struct
import wave
from pathlib import Path

import pytest
import torch

from open_qwen_music.tokenizer.data import (
    DistributedBalancedDurationBucketBatchSampler,
    TokenizerDataset,
)
from open_qwen_music.tokenizer.text import (
    CTC_PHONEME_SPEC_VERSION,
    FIXED_ZH_EN_PHONEME_VOCAB,
    lyrics_to_phonemes,
    normalize_existing_phonemes,
    CharacterTokenizer,
)
from open_qwen_music.tokenizer.trainer import (
    _inherit_feature_config_from_checkpoint,
    _validate_subsampling_contract_from_checkpoint,
)


class _FakeDataset:
    duration_buckets_sec = [30.0]

    def __init__(
        self,
        groups: list[str] | None = None,
        localities: list[str] | None = None,
    ):
        self.groups = groups if groups is not None else ["a"] * 8 + ["b"] * 2
        self.localities = localities or ["unknown"] * len(self.groups)
        assert len(self.localities) == len(self.groups)

    def __len__(self):
        return len(self.groups)

    def padding_duration_sec(self, index):
        return 30.0

    def sampling_value(self, index, field, default="unknown"):
        if field in {"training.io_group", "audio.io_group"}:
            return self.localities[index]
        return self.groups[index]


def _sampled_ratios(
    dataset: _FakeDataset,
    weights: dict[str, float],
    *,
    batch_size_per_rank: int,
    world_size: int,
    default_weight: float = 0.0,
    bucket_by_duration: bool = True,
) -> dict[str, float]:

    seen: dict[str, int] = {}
    for rank in range(world_size):
        sampler = DistributedBalancedDurationBucketBatchSampler(
            dataset,
            batch_size_per_rank=batch_size_per_rank,
            rank=rank,
            world_size=world_size,
            seed=11,
            balance_key="training.source_sampling_group",
            weights=weights,
            default_weight=default_weight,
            shuffle=False,
            bucket_by_duration=bucket_by_duration,
        )
        for batch in sampler:
            for index in batch:
                key = dataset.groups[index]
                seen[key] = seen.get(key, 0) + 1
    total = sum(seen.values())
    return {key: value / total for key, value in seen.items()}


def test_fixed_phoneme_vocab_and_code_switch():
    assert len(FIXED_ZH_EN_PHONEME_VOCAB) == 123
    assert len(set(FIXED_ZH_EN_PHONEME_VOCAB)) == 123
    units, metadata = lyrics_to_phonemes("\u4f60\u597d hello")
    assert metadata["spec_version"] == CTC_PHONEME_SPEC_VERSION
    assert units[:4] == ["n", "i", "h", "ao"]
    assert units[4:] == ["HH", "AH0", "L", "OW1"]


def test_existing_phoneme_normalization():
    units, unknown = normalize_existing_phonemes(
        ["AA1", "SP", "B"], "english"
    )
    assert units == ["AA1", "B"]
    assert unknown == 0


def test_balanced_sampler_builds_global_ratio():
    dataset = _FakeDataset()
    rank0 = DistributedBalancedDurationBucketBatchSampler(
        dataset,
        batch_size_per_rank=2,
        rank=0,
        world_size=2,
        seed=7,
        balance_key="training.sampling_group",
        weights={"a": 0.5, "b": 0.5},
    )
    rank1 = DistributedBalancedDurationBucketBatchSampler(
        dataset,
        batch_size_per_rank=2,
        rank=1,
        world_size=2,
        seed=7,
        balance_key="training.sampling_group",
        weights={"a": 0.5, "b": 0.5},
    )
    for left, right in zip(rank0, rank1):
        global_batch = left + right
        groups = [dataset.groups[index] for index in global_batch]
        assert groups.count("a") == 2
        assert groups.count("b") == 2


def test_balanced_sampler_honours_weights_not_divisible_by_global_batch():

    weights = {
        "big": 0.22,
        "mid": 0.08,
        "small": 0.07,
        "tiny": 0.03,
    }
    dataset = _FakeDataset(
        ["big"] * 400 + ["mid"] * 400 + ["small"] * 400 + ["tiny"] * 400
    )
    ratios = _sampled_ratios(
        dataset, weights, batch_size_per_rank=1, world_size=16
    )
    total_weight = sum(weights.values())
    for key, weight in weights.items():
        expected = weight / total_weight
        actual = ratios.get(key, 0.0)
        assert actual == pytest.approx(expected, rel=0.15), (
            f"group={key} Expected proportion {expected:.4f},Actual {actual:.4f};"
            f"Complete distribution {ratios}"
        )


def test_balanced_sampler_never_starves_a_positive_weight_group():

    weights = {"dominant": 0.9, "rare": 0.02}
    dataset = _FakeDataset(["dominant"] * 500 + ["rare"] * 100)
    ratios = _sampled_ratios(
        dataset, weights, batch_size_per_rank=1, world_size=16
    )
    assert ratios.get("rare", 0.0) > 0.0, (
        f"Positive weight group is not sampled at all:{ratios}"
    )


def test_balanced_sampler_rejects_weight_for_absent_group():

    dataset = _FakeDataset(["present"] * 64)
    with pytest.raises(ValueError, match="absent_group"):
        DistributedBalancedDurationBucketBatchSampler(
            dataset,
            batch_size_per_rank=2,
            rank=0,
            world_size=2,
            seed=0,
            balance_key="training.source_sampling_group",
            weights={"present": 0.5, "absent_group": 0.5},
        )


def test_balanced_sampler_never_allocates_negative_counts():


    allocate = DistributedBalancedDurationBucketBatchSampler._allocate_counts
    weights = [
        0.7344, 0.0299, 0.3888, 0.7263, 0.3201, 0.1053,
        0.0011, 0.0819, 0.2281, 0.7612, 0.9032,
    ]
    keys = [f"g{index}" for index in range(len(weights))]
    credit: dict[str, float] = {}
    for _ in range(200):
        counts = allocate(keys, weights, 8, credit)
        assert all(value >= 0 for value in counts.values()), counts
        assert sum(counts.values()) == 8, counts


def test_balanced_sampler_can_apply_weights_globally_without_duration_buckets():
    dataset = _FakeDataset(
        ["trusted_a"] * 40
        + ["trusted_b"] * 10
        + ["disabled_instrumental"] * 100
    )
    weights = {"trusted_a": 0.5, "trusted_b": 0.5}
    ratios = _sampled_ratios(
        dataset,
        weights,
        batch_size_per_rank=2,
        world_size=2,
        bucket_by_duration=False,
    )
    assert ratios == pytest.approx({"trusted_a": 0.5, "trusted_b": 0.5})

    sampler = DistributedBalancedDurationBucketBatchSampler(
        dataset,
        batch_size_per_rank=2,
        rank=0,
        world_size=2,
        seed=11,
        balance_key="training.source_sampling_group",
        weights=weights,
        default_weight=0.0,
        shuffle=False,
        bucket_by_duration=False,
    )

    assert len(sampler) == 13


def test_balanced_sampler_locality_is_seed_rank_and_resume_deterministic():
    dataset = _FakeDataset(
        ["a"] * 64 + ["b"] * 64,
        [f"rg-{index % 8}" for index in range(128)],
    )

    def make(rank=1):
        return DistributedBalancedDurationBucketBatchSampler(
            dataset,
            batch_size_per_rank=2,
            rank=rank,
            world_size=4,
            seed=29,
            balance_key="training.sampling_group",
            weights={"a": 0.6, "b": 0.4},
            locality_key="training.io_group",
        )

    expected = list(make())
    assert list(make()) == expected
    resumed = make()
    resumed.set_skip_batches(5)
    assert list(resumed) == expected[5:]
    other_rank_first = list(make(rank=2))
    other_rank_second = list(make(rank=2))
    assert other_rank_first == other_rank_second


def test_balanced_sampler_locality_preserves_source_allocation_and_slicing():
    dataset = _FakeDataset(
        ["a"] * 32 + ["b"] * 32,
        [f"rg-{index // 4}" for index in range(64)],
    )

    def batches(locality_key, rank):
        sampler = DistributedBalancedDurationBucketBatchSampler(
            dataset,
            batch_size_per_rank=2,
            rank=rank,
            world_size=4,
            seed=17,
            balance_key="training.sampling_group",
            weights={"a": 0.5, "b": 0.5},
            locality_key=locality_key,
        )
        return list(sampler)

    plain = [batches(None, rank) for rank in range(4)]
    local = [
        batches("training.io_group", rank)
        for rank in range(4)
    ]
    plain_counts = {"a": 0, "b": 0}
    local_counts = {"a": 0, "b": 0}
    seen = []
    for batch_index in range(len(local[0])):
        plain_global = [
            index
            for rank_batches in plain
            for index in rank_batches[batch_index]
        ]
        local_global = [
            index
            for rank_batches in local
            for index in rank_batches[batch_index]
        ]
        assert len(local_global) == 8
        assert len(set(local_global)) == 8
        assert [dataset.groups[index] for index in plain_global].count("a") == 4
        assert [dataset.groups[index] for index in local_global].count("a") == 4
        for index in plain_global:
            plain_counts[dataset.groups[index]] += 1
        for index in local_global:
            local_counts[dataset.groups[index]] += 1
        seen.extend(local_global)
    assert local_counts == plain_counts == {"a": 32, "b": 32}
    assert sorted(seen) == list(range(64))


def test_balanced_sampler_locality_reduces_row_group_transitions():
    size = 256
    dataset = _FakeDataset(
        ["only"] * size,
        [f"rg-{index % 16}" for index in range(size)],
    )

    def sequence(locality_key):
        sampler = DistributedBalancedDurationBucketBatchSampler(
            dataset,
            batch_size_per_rank=2,
            rank=0,
            world_size=1,
            seed=41,
            balance_key="training.sampling_group",
            weights={"only": 1.0},
            locality_key=locality_key,
        )
        return [index for batch in sampler for index in batch]

    plain = sequence(None)
    local = sequence("training.io_group")

    def transitions(indices):
        groups = [dataset.localities[index] for index in indices]
        return sum(left != right for left, right in zip(groups, groups[1:]))

    assert transitions(local) < transitions(plain) * 0.7


def test_balanced_sampler_without_locality_keeps_legacy_sequence():
    dataset = _FakeDataset(["a"] * 8 + ["b"] * 8)
    common = {
        "dataset": dataset,
        "batch_size_per_rank": 2,
        "rank": 0,
        "world_size": 2,
        "seed": 7,
        "balance_key": "training.sampling_group",
        "weights": {"a": 0.5, "b": 0.5},
    }
    omitted = DistributedBalancedDurationBucketBatchSampler(**common)
    explicit_none = DistributedBalancedDurationBucketBatchSampler(
        **common, locality_key=None
    )
    assert list(omitted) == list(explicit_none)


def test_lyrics_must_be_a_string():

    from open_qwen_music.tokenizer.data import _get_lyrics

    assert _get_lyrics({"lyrics": "\u6b63\u5e38"}) == "\u6b63\u5e38"
    assert _get_lyrics({"text": {"lyrics": "release\u6b4c\u8bcd"}}) == "release\u6b4c\u8bcd"
    with pytest.raises(TypeError, match="lyrics"):
        _get_lyrics({"sample_id": "x", "lyrics": None})
    with pytest.raises(TypeError, match="lyrics"):
        _get_lyrics({"sample_id": "x", "lyrics": ["a", "b"]})
    with pytest.raises(TypeError, match="text.lyrics"):
        _get_lyrics({"sample_id": "x", "text": {"lyrics": None}})


def test_empty_ctc_units_falls_back_instead_of_zero_length_target():

    from open_qwen_music.tokenizer.data import _get_ctc_units

    assert _get_ctc_units({"ctc_units": []}) is None
    assert _get_ctc_units({"ctc_units": ""}) is None
    assert _get_ctc_units({"text": {"phonemes": []}}) is None
    assert _get_ctc_units({"ctc_units": ["a", "b"]}) == ["a", "b"]


def test_english_phonemes_never_silently_disappear():

    pytest.importorskip("cmudict")
    from open_qwen_music.tokenizer.text import english_to_phonemes

    units, unknown = english_to_phonemes("wanda")
    assert units == ["W", "AA1", "N", "D", "AH0"]
    assert unknown == 0

    from open_qwen_music.tokenizer.text import _cmu_dictionary

    dictionary = _cmu_dictionary()
    for word in ("wanda", "about", "boy", "hello"):
        expected = len(dictionary[word][0])
        assert len(english_to_phonemes(word)[0]) == expected


def test_resample_rejects_coprime_rates_instead_of_exploding():

    from open_qwen_music.tokenizer.audio import resample_mono

    with pytest.raises(ValueError, match="unsupported"):
        resample_mono(torch.randn(1000), 44_101, 24_000)

    assert resample_mono(torch.randn(44_100), 44_100, 24_000).numel() == 24_000


def _write_crop_mismatch_manifest(tmp_path: Path) -> Path:
    audio = tmp_path / "long.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(struct.pack("<h", 1000) * 24_000 * 8)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "crop-mismatch",
                "split": "train",
                "audio_path": str(audio),
                "audio": {"start_sec": 0.0, "duration_sec": 8.0},
                "lyrics": "\u6574\u6bb5\u6b4c\u8bcd",
                "training": {"loss_heads": {"ctc": True}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_ctc_crop_mismatch_raises_by_default(tmp_path: Path):
    dataset = TokenizerDataset(
        _write_crop_mismatch_manifest(tmp_path),
        stage=3,
        max_duration_sec=4.0,
        random_crop=False,
    )
    with pytest.raises(ValueError, match="CTC sample was cropped"):
        dataset[0]


def test_ctc_crop_mismatch_disable_drops_ctc_supervision(tmp_path: Path):
    dataset = TokenizerDataset(
        _write_crop_mismatch_manifest(tmp_path),
        stage=3,
        max_duration_sec=4.0,
        random_crop=False,
        ctc_on_crop_mismatch="disable",
    )
    with pytest.warns(RuntimeWarning):
        sample = dataset[0]
    assert sample["ctc_enabled"] is False
    assert sample["mel_enabled"] is True


def test_ctc_weight_is_separate_from_acoustic_sample_weight(
    tmp_path: Path,
) -> None:
    manifest = _write_crop_mismatch_manifest(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["training"].update(
        {
            "sampling_group": "audited-source",
            "ctc_group": "audited-ctc",
            "sample_weight": 1.0,
            "ctc_weight": 0.5,
        }
    )
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = TokenizerDataset(
        manifest,
        stage=3,
        max_duration_sec=8.0,
        random_crop=False,
        ctc_group_weights={"audited-ctc": 0.25},
        ctc_group_default_weight=0.0,
    )

    sample = dataset[0]
    assert sample["ctc_enabled"] is True
    assert sample["sample_weight"] == 1.0
    assert sample["ctc_sample_weight"] == pytest.approx(0.125)


def _write_overlong_duration_manifest(
    tmp_path: Path, real_sec: int = 4, declared_sec: float = 60.0
) -> Path:
    audio = tmp_path / "short.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(struct.pack("<h", 1000) * 24_000 * real_sec)
    manifest = tmp_path / "overlong.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "overlong-duration",
                "split": "train",
                "audio_path": str(audio),
                "audio": {"start_sec": 0.0, "duration_sec": declared_sec},
                "training": {"loss_heads": {"ctc": False}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_random_crop_beyond_real_end_falls_back_instead_of_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = _write_overlong_duration_manifest(tmp_path)

    monkeypatch.setattr(random, "random", lambda: 0.9)
    dataset = TokenizerDataset(
        manifest, stage=1, max_duration_sec=4.0, random_crop=True
    )
    with pytest.warns(RuntimeWarning, match="Random crop start exceeded"):
        sample = dataset[0]

    assert sample["waveform"].numel() == 4 * 24_000


def test_empty_decode_at_declared_start_degrades_to_silence(tmp_path: Path):
    audio = tmp_path / "empty.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(b"")
    manifest = tmp_path / "empty.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "empty-audio",
                "split": "train",
                "audio_path": str(audio),
                "audio": {"start_sec": 0.0, "duration_sec": 30.0},
                "training": {"loss_heads": {"ctc": False}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = TokenizerDataset(
        manifest, stage=1, max_duration_sec=4.0, random_crop=True
    )
    with pytest.warns(RuntimeWarning, match="Sample decode failed"):
        sample = dataset[0]
    assert sample["waveform"].abs().max().item() == 0.0
    assert sample["sample_weight"] == 0.0


def _write_undecodable_manifest(tmp_path: Path, rows: int = 1) -> Path:
    bogus = tmp_path / "not_audio.wav"
    bogus.write_bytes(b"definitely not a RIFF/WAVE payload")
    manifest = tmp_path / "undecodable.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": f"undecodable-{i}",
                    "split": "train",
                    "audio_path": str(bogus),
                    "audio": {"start_sec": 0.0, "duration_sec": 30.0},
                    "lyrics": "\u6b4c\u8bcd",
                    "training": {"loss_heads": {"ctc": True}},
                }
            )
            + "\n"
            for i in range(rows)
        ),
        encoding="utf-8",
    )
    return manifest


def test_decode_exception_falls_back_to_silence(tmp_path: Path):
    dataset = TokenizerDataset(
        _write_undecodable_manifest(tmp_path),
        stage=3,
        max_duration_sec=4.0,
        random_crop=False,
    )
    with pytest.warns(RuntimeWarning, match="Sample decode failed"):
        sample = dataset[0]
    assert sample["waveform"].abs().max().item() == 0.0

    assert sample["sample_weight"] == 0.0
    assert sample["ctc_enabled"] is False
    assert sample["mel_enabled"] is False
    assert sample["chroma_enabled"] is False


def test_systematic_decode_failure_still_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from open_qwen_music.tokenizer import data as data_module

    monkeypatch.setattr(data_module, "MAX_CONSECUTIVE_DECODE_FAILURES", 2)
    dataset = TokenizerDataset(
        _write_undecodable_manifest(tmp_path, rows=4),
        stage=1,
        max_duration_sec=4.0,
        random_crop=False,
    )
    with pytest.warns(RuntimeWarning):
        dataset[0]
        dataset[1]
    with pytest.raises(RuntimeError, match="storage or archive failure"):
        dataset[2]


def test_successful_decode_resets_failure_counter(tmp_path: Path):
    good = _write_overlong_duration_manifest(tmp_path, declared_sec=4.0)
    bad = _write_undecodable_manifest(tmp_path)
    merged = tmp_path / "merged.jsonl"
    merged.write_text(
        bad.read_text(encoding="utf-8") + good.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    dataset = TokenizerDataset(
        merged, stage=1, max_duration_sec=4.0, random_crop=False
    )
    with pytest.warns(RuntimeWarning, match="Sample decode failed"):
        dataset[0]
    assert dataset._decode_failures == 1
    dataset[1]
    assert dataset._decode_failures == 0


def test_ctc_crop_mismatch_ignored_before_stage3(tmp_path: Path):
    dataset = TokenizerDataset(
        _write_crop_mismatch_manifest(tmp_path),
        stage=1,
        max_duration_sec=4.0,
        random_crop=False,
    )
    assert dataset[0]["ctc_enabled"] is True


def test_ctc_kept_when_crop_covers_full_segment(tmp_path: Path):
    dataset = TokenizerDataset(
        _write_crop_mismatch_manifest(tmp_path),
        stage=3,
        max_duration_sec=30.0,
        random_crop=True,
    )
    assert dataset[0]["ctc_enabled"] is True


def test_rejects_unknown_crop_mismatch_policy(tmp_path: Path):
    with pytest.raises(ValueError, match="ctc_on_crop_mismatch"):
        TokenizerDataset(
            _write_crop_mismatch_manifest(tmp_path),
            stage=3,
            max_duration_sec=4.0,
            random_crop=False,
            ctc_on_crop_mismatch="bogus",
        )


def test_stage3_inherits_feature_stats_from_stage2(tmp_path: Path):
    checkpoint = tmp_path / "stage2.pt"
    source_features = {
        "sample_rate": 24000,
        "n_fft": 1024,
        "hop_length": 240,
        "win_length": 960,
        "n_mels": 128,
        "f_min": 0.0,
        "f_max": 12000.0,
        "log_floor": 1e-5,
        "mel_filter_norm": "legacy",
        "log_mode": "clamp",
        "mean": -1.25,
        "std": 3.5,
    }
    source_chroma = {"mode": "soft", "n_fft": 4096}
    torch.save(
        {
            "config": {"features": source_features, "chroma": source_chroma},
            "model": {
                "feature_extractor.feature_mean": torch.tensor(-1.75),
                "feature_extractor.feature_std": torch.tensor(4.25),
            },
        },
        checkpoint,
    )
    config = {
        "stage": 3,
        "features": {**source_features, "mean": 0.0, "std": 1.0},
        "chroma": dict(source_chroma),
    }
    _inherit_feature_config_from_checkpoint(config, str(checkpoint))
    assert config["features"]["mean"] == -1.75
    assert config["features"]["std"] == 4.25
    assert config["lineage"]["feature_config_inherited_from"] == str(checkpoint)


    assert "feature_stats_source" not in config["lineage"]
    assert config["lineage"]["feature_stats_value_source"] == "model_buffers"


def test_stage3_lineage_uses_shared_checkpoint_instead_of_node_cache(
    tmp_path: Path,
):
    checkpoint = tmp_path / "node-local-cache.pt"
    source_features = {
        "sample_rate": 24000,
        "n_fft": 1024,
        "hop_length": 240,
        "win_length": 960,
        "n_mels": 128,
        "f_min": 0.0,
        "f_max": 12000.0,
        "log_floor": 1e-5,
        "mel_filter_norm": "legacy",
        "log_mode": "clamp",
        "mean": -1.25,
        "std": 3.5,
    }
    source_chroma = {"mode": "soft", "n_fft": 4096}
    torch.save(
        {
            "config": {"features": source_features, "chroma": source_chroma},
            "model": {
                "feature_extractor.feature_mean": torch.tensor(-1.75),
                "feature_extractor.feature_std": torch.tensor(4.25),
            },
        },
        checkpoint,
    )
    config = {
        "stage": 3,
        "features": {**source_features, "mean": 0.0, "std": 1.0},
        "chroma": dict(source_chroma),
    }
    shared_checkpoint = "/shared/checkpoints/stage2.pt"

    _inherit_feature_config_from_checkpoint(
        config,
        str(checkpoint),
        lineage_checkpoint=shared_checkpoint,
    )

    assert config["lineage"]["feature_config_inherited_from"] == shared_checkpoint


def _write_model_config_checkpoint(path: Path, model_config: dict) -> Path:
    torch.save({"config": {"model": model_config}, "model": {}}, path)
    return path


def test_rejects_pre_20260801_causal_padding_checkpoint(tmp_path: Path):

    checkpoint = _write_model_config_checkpoint(
        tmp_path / "old_stage2.pt",
        {"frontend_type": "conv", "subsampling_kernel": 5},
    )
    config = {"stage": 3, "model": {"frontend_type": "conv", "subsampling_kernel": 5}}
    with pytest.raises(RuntimeError, match="Subsampling frontend contract mismatch"):
        _validate_subsampling_contract_from_checkpoint(config, str(checkpoint))


_CURRENT_SEMANTICS = {
    "causal_padding": "within_stride",
    "bestrq_causal_window": True,
    "bestrq_erode_loss_mask": True,
    "causal_heads": True,
    "position_encoding": "rope",
}


def test_accepts_checkpoint_declaring_current_causal_padding(tmp_path: Path):
    checkpoint = _write_model_config_checkpoint(
        tmp_path / "new_stage2.pt",
        {
            "frontend_type": "conv",
            "subsampling_kernel": 5,
            **_CURRENT_SEMANTICS,
        },
    )
    config = {
        "stage": 3,
        "model": {"frontend_type": "conv", "subsampling_kernel": 5},
    }
    _validate_subsampling_contract_from_checkpoint(config, str(checkpoint))


def test_v4_stage1_to_stage2_allows_only_attention_change(tmp_path: Path):
    source_model = {
        "frontend_type": "conv",
        "subsampling_kernel": 5,
        **_CURRENT_SEMANTICS,
        "causal": False,
        "attention_causal": False,
        "frontend_causal": True,
        "conformer_conv_causal": True,
    }
    checkpoint = tmp_path / "v4_stage1.pt"
    torch.save(
        {"config": {"stage": 1, "model": source_model}, "model": {}},
        checkpoint,
    )
    target = {
        "stage": 2,
        "model": {
            **source_model,
            "causal": True,
            "attention_causal": True,
        },
    }
    _validate_subsampling_contract_from_checkpoint(target, str(checkpoint))

    target["model"]["frontend_causal"] = False
    with pytest.raises(RuntimeError, match="frontend_causal"):
        _validate_subsampling_contract_from_checkpoint(target, str(checkpoint))


def test_stage1_zero_causal_stage3_probe_requires_explicit_override(
    tmp_path: Path,
):
    source_model = {
        "frontend_type": "conv",
        "subsampling_kernel": 5,
        **_CURRENT_SEMANTICS,
        "causal": False,
        "attention_causal": False,
        "frontend_causal": True,
        "conformer_conv_causal": True,
    }
    checkpoint = tmp_path / "v4_stage1.pt"
    torch.save(
        {"config": {"stage": 1, "model": source_model}, "model": {}},
        checkpoint,
    )
    target = {
        "stage": 3,
        "model": {
            **source_model,
            "causal": True,
            "attention_causal": True,
        },
        "probe": {},
    }
    with pytest.raises(RuntimeError, match="attention_causal"):
        _validate_subsampling_contract_from_checkpoint(target, str(checkpoint))
    target["probe"]["allow_stage1_attention_causal_override"] = True
    _validate_subsampling_contract_from_checkpoint(target, str(checkpoint))

    target["model"]["frontend_causal"] = False
    with pytest.raises(RuntimeError, match="frontend_causal"):
        _validate_subsampling_contract_from_checkpoint(target, str(checkpoint))


def test_rejects_pre_20260803_causal_semantics(tmp_path: Path):

    checkpoint = _write_model_config_checkpoint(
        tmp_path / "fe4_stage2.pt",
        {
            "frontend_type": "conv",
            "subsampling_kernel": 5,
            "causal_padding": "within_stride",
        },
    )
    config = {
        "stage": 3,
        "model": {"frontend_type": "conv", "subsampling_kernel": 5},
    }
    with pytest.raises(RuntimeError, match="causal_heads"):
        _validate_subsampling_contract_from_checkpoint(config, str(checkpoint))


    legacy = {
        "stage": 3,
        "model": {
            "frontend_type": "conv",
            "subsampling_kernel": 5,
            "bestrq_causal_window": False,
            "bestrq_erode_loss_mask": False,
            "causal_heads": False,
            "position_encoding": "sinusoidal",
        },
    }
    _validate_subsampling_contract_from_checkpoint(legacy, str(checkpoint))


def test_rejects_pre_20260804_position_encoding(tmp_path: Path):

    checkpoint = _write_model_config_checkpoint(
        tmp_path / "sinusoidal_stage2.pt",
        {
            "frontend_type": "conv",
            "subsampling_kernel": 5,
            "causal_padding": "within_stride",
            "bestrq_causal_window": True,
            "bestrq_erode_loss_mask": True,
            "causal_heads": True,
        },
    )
    config = {
        "stage": 3,
        "model": {"frontend_type": "conv", "subsampling_kernel": 5},
    }
    with pytest.raises(RuntimeError, match="position_encoding"):
        _validate_subsampling_contract_from_checkpoint(config, str(checkpoint))

    legacy = {
        "stage": 3,
        "model": {
            "frontend_type": "conv",
            "subsampling_kernel": 5,
            "position_encoding": "sinusoidal",
        },
    }
    _validate_subsampling_contract_from_checkpoint(legacy, str(checkpoint))


@pytest.mark.parametrize(
    "override",
    [
        {"frontend_type": "convnext"},
        {"subsampling_kernel": 7},
        {"convnext_blocks_per_stage": 2},


        {"convnext_layer_scale_init": 1.0},
    ],
)
def test_rejects_frontend_structure_drift(tmp_path: Path, override: dict):
    checkpoint = _write_model_config_checkpoint(
        tmp_path / "stage2.pt",
        {
            "frontend_type": "conv",
            "subsampling_kernel": 5,
            **_CURRENT_SEMANTICS,
        },
    )
    config = {
        "stage": 3,
        "model": {
            "frontend_type": "conv",
            "subsampling_kernel": 5,
            "causal_padding": "within_stride",
            **override,
        },
    }
    with pytest.raises(RuntimeError, match="Subsampling frontend contract mismatch"):
        _validate_subsampling_contract_from_checkpoint(config, str(checkpoint))


def test_multilingual_subword_vocab_roundtrip(tmp_path: Path):
    from tokenizers import SentencePieceBPETokenizer

    backend = SentencePieceBPETokenizer(
        unk_token="<unk>",
        replacement="▁",
        add_prefix_space=True,
        fuse_unk=True,
    )
    backend.train_from_iterator(
        ["\u4f60\u597d hello world", "\u97f3\u4e50 generation \u6d4b\u8bd5"] * 8,
        vocab_size=64,
        min_frequency=1,
        special_tokens=["<unk>"],
    )
    tokenizer_json = tmp_path / "tokenizer.json"
    backend.save(str(tokenizer_json))
    backend_tokens = [
        token
        for token, _ in sorted(
            backend.get_vocab().items(), key=lambda item: item[1]
        )
    ]
    vocab = tmp_path / "vocab.json"
    vocab.write_text(
        json.dumps(
            {
                "format_version": "oqm.ctc-subword-vocab.v1",
                "tokenizer_json": tokenizer_json.name,
                "tokens": ["<blank>", *backend_tokens],
            }
        ),
        encoding="utf-8",
    )
    tokenizer = CharacterTokenizer.from_file(vocab)
    ids = tokenizer.encode("\u4f60\u597d hello world")
    assert tokenizer.kind == "subword"
    assert all(index > 0 for index in ids)
    assert tokenizer.decode_text(ids) == "\u4f60\u597d hello world"
    assert tokenizer.encode_units(tokenizer.decode_units(ids)) == ids


def test_stage3_rejects_chroma_target_mismatch(tmp_path: Path):

    features = {
        "sample_rate": 24000,
        "n_fft": 1024,
        "hop_length": 240,
        "win_length": 960,
        "n_mels": 128,
        "f_min": 0.0,
        "f_max": 12000.0,
        "log_floor": 1e-9,
        "mel_filter_norm": "slaney",
        "log_mode": "additive",
        "mean": -1.25,
        "std": 3.5,
    }
    checkpoint = tmp_path / "stage2_hard_chroma.pt"
    torch.save({"config": {"features": features}, "model": {}}, checkpoint)
    config = {"stage": 3, "features": dict(features), "chroma": {"mode": "soft", "n_fft": 4096}}
    with pytest.raises(RuntimeError, match="chroma"):
        _inherit_feature_config_from_checkpoint(config, str(checkpoint))
