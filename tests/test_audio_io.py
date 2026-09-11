from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import struct
import subprocess
import tarfile
import wave
import zipfile
from io import BytesIO

import pytest
import torch

from open_qwen_music.tokenizer import audio as audio_module
from open_qwen_music.tokenizer import data as data_module
from open_qwen_music.tokenizer.audio import (
    AudioAssetIntegrityError,
    AudioCredentialError,
    EncryptedZipReadError,
    configure_parquet_row_group_cache,
    load_audio,
    prefetch_parquet_audio,
    probe_audio,
    reset_audio_io_state,
)
from open_qwen_music.tokenizer.data import TokenizerDataset


def _wave_bytes(value: int = 1000, frames: int = 2400) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(struct.pack("<h", value) * frames)
    return buffer.getvalue()


def _write_indexed_tar(path, payload: bytes, member: str = "sample.wav"):
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo(member)
        info.size = len(payload)
        archive.addfile(info, BytesIO(payload))
    with tarfile.open(path, "r") as archive:
        return archive.getmember(member)


def _asset_kwargs(path, info, payload: bytes, *, revision: str = "asset-r1"):
    return {
        "archive_offset": info.offset_data,
        "archive_size": info.size,
        "asset_id": "asset-0001",
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "asset_revision": revision,
        "shard_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _fork_audio_cache_state(queue) -> None:
    audio_module._ensure_cache_process()
    queue.put(
        (
            len(audio_module._ARCHIVE_PAYLOAD_CACHE),
            len(audio_module._BINARY_CACHE),
            audio_module._ARCHIVE_PAYLOAD_CACHE_BYTES,
        )
    )


def test_load_wave_segment(tmp_path) -> None:
    path = tmp_path / "segment.wav"
    samples = [int(10_000 * ((index % 20) / 20.0 - 0.5)) for index in range(16_000)]
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"".join(struct.pack("<h", value) for value in samples))
    info = probe_audio(path)
    assert info.duration_sec == 1.0
    waveform, sample_rate = load_audio(path, start_sec=0.25, duration_sec=0.5)
    assert sample_rate == 16_000
    assert waveform.shape == (8_000,)


def test_decode_24bit_pcm(tmp_path) -> None:
    path = tmp_path / "pcm24.wav"
    values = [-8_388_608, -1, 0, 1, 8_388_607]
    payload = bytearray()
    for value in values:
        unsigned = value if value >= 0 else value + (1 << 24)
        payload.extend(
            [unsigned & 0xFF, (unsigned >> 8) & 0xFF, (unsigned >> 16) & 0xFF]
        )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(3)
        output.setframerate(48_000)
        output.writeframes(bytes(payload))
    waveform, _ = load_audio(path)
    torch.testing.assert_close(
        waveform,
        torch.tensor(values, dtype=torch.float32) / float(1 << 23),
        rtol=0,
        atol=0,
    )


def test_ffmpeg_bytes_fallback_uses_pipe_float32_and_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = torch.tensor([0.25, -0.5, 0.75], dtype=torch.float32)
    captured = {}
    monkeypatch.setenv("OQM_TEST_FFMPEG_PASSWORD", "must-not-reach-child")

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=expected.numpy().tobytes(),
            stderr=b"",
        )

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)
    waveform, sample_rate = audio_module._read_audio_bytes(
        b"not-an-audio-container",
        start_sec=1.25,
        duration_sec=0.5,
    )

    assert sample_rate == 24_000
    torch.testing.assert_close(waveform, expected, rtol=0, atol=0)
    command = captured["command"]
    assert command[:5] == [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
    ]
    assert command[command.index("-protocol_whitelist") + 1] == "file,pipe"
    assert command[command.index("-i") + 1] == "pipe:0"
    assert command[command.index("-ss") + 1] == "1.25"
    assert command[command.index("-t") + 1] == "0.5"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "24000"
    assert command[command.index("-c:a") + 1] == "pcm_f32le"
    assert command[command.index("-f") + 1] == "f32le"
    assert command[-1] == "pipe:1"
    assert captured["kwargs"]["input"] == b"not-an-audio-container"
    assert captured["kwargs"]["timeout"] > 0
    assert all(
        marker not in key.upper()
        for key in captured["kwargs"]["env"]
        for marker in ("PASSWORD", "SECRET", "TOKEN", "CREDENTIAL")
    )


def test_ffmpeg_bytes_fallback_rejects_nonzero_without_secret_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "ffmpeg-" + "credential"
    monkeypatch.setenv("OQM_TEST_FFMPEG_PASSWORD", secret)

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            9,
            stdout=b"",
            stderr=(f"decoder failed credential={secret} ".encode() + b"x" * 4096),
        )

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)
    with pytest.raises(
        audio_module.FFmpegAudioDecodeError,
        match="returned non-zero status 9",
    ) as caught:
        audio_module._read_audio_bytes(
            b"not-an-audio-container",
            duration_sec=0.25,
        )
    message = str(caught.value)
    assert secret not in message
    assert "<redacted>" in message
    assert "<truncated:" in message
    assert len(message) < 1400


def test_ffmpeg_bytes_fallback_rejects_timeout_and_empty_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(_command, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd="ffmpeg",
            timeout=kwargs["timeout"],
            stderr=b"must-not-be-copied",
        )

    monkeypatch.setattr(audio_module.subprocess, "run", timeout)
    with pytest.raises(
        audio_module.FFmpegAudioDecodeError,
        match="ffmpeg fallback timed out",
    ) as caught:
        audio_module._read_audio_bytes(
            b"not-an-audio-container",
            duration_sec=0.25,
        )
    assert "must-not-be-copied" not in str(caught.value)

    def empty(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b"",
            stderr=b"no samples",
        )

    monkeypatch.setattr(audio_module.subprocess, "run", empty)
    with pytest.raises(
        audio_module.FFmpegAudioDecodeError,
        match="returned empty audio",
    ):
        audio_module._read_audio_bytes(
            b"not-an-audio-container",
            duration_sec=0.25,
        )


def test_ffmpeg_bytes_fallback_limits_output_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duration_sec = 0.01
    max_bytes = (
        int(duration_sec * audio_module._FFMPEG_SAMPLE_RATE)
        + audio_module._FFMPEG_OUTPUT_SLACK_FRAMES
    ) * audio_module._FFMPEG_BYTES_PER_SAMPLE

    def oversized(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b"\0" * (max_bytes + 4),
            stderr=b"",
        )

    monkeypatch.setattr(audio_module.subprocess, "run", oversized)
    with pytest.raises(
        audio_module.FFmpegAudioDecodeError,
        match="output exceeds the safety limit",
    ):
        audio_module._read_audio_bytes(
            b"not-an-audio-container",
            duration_sec=duration_sec,
        )

    def malformed(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b"\0" * 5,
            stderr=b"",
        )

    monkeypatch.setattr(audio_module.subprocess, "run", malformed)
    with pytest.raises(
        audio_module.FFmpegAudioDecodeError,
        match="byte count that is not a multiple of 4",
    ):
        audio_module._read_audio_bytes(
            b"not-an-audio-container",
            duration_sec=duration_sec,
        )


def test_ffmpeg_bytes_fallback_without_duration_has_safe_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio_module, "_FFMPEG_NO_DURATION_MAX_SEC", 0.001)
    monkeypatch.setattr(audio_module, "_FFMPEG_NO_DURATION_PROBE_SEC", 0.001)
    capped_bytes = (
        int(0.001 * audio_module._FFMPEG_SAMPLE_RATE)
        * audio_module._FFMPEG_BYTES_PER_SAMPLE
    )
    captured_command = []

    def reaches_cap(command, **_kwargs):
        captured_command.extend(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b"\0" * capped_bytes,
            stderr=b"",
        )

    monkeypatch.setattr(audio_module.subprocess, "run", reaches_cap)
    with pytest.raises(
        audio_module.FFmpegAudioDecodeError,
        match="without duration_sec",
    ):
        audio_module._read_audio_bytes(b"not-an-audio-container")
    assert captured_command[captured_command.index("-t") + 1] == "0.002"


def test_real_m4a_bytes_ffmpeg_fallback_start_and_duration(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")
    path = tmp_path / "tone.m4a"
    encoded = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-c:a",
            "aac",
            "-y",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if encoded.returncode != 0:
        pytest.skip("ffmpeg does not support the AAC/M4A encoder required by this test")

    import soundfile as sf

    def force_soundfile_failure(*_args, **_kwargs):
        raise sf.LibsndfileError(1, "forced fallback")

    monkeypatch.setattr(sf, "SoundFile", force_soundfile_failure)
    loose_waveform, loose_sample_rate = load_audio(
        path,
        start_sec=0.2,
        duration_sec=0.25,
    )
    assert loose_sample_rate == 24_000
    assert loose_waveform.shape == (6000,)
    assert loose_waveform.abs().max().item() > 0.01

    payload = path.read_bytes()
    archive_path = tmp_path / "tone.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo("riffusion/tone.m4a")
        info.size = len(payload)
        archive.addfile(info, BytesIO(payload))
    with tarfile.open(archive_path, "r") as archive:
        info = archive.getmember("riffusion/tone.m4a")
    waveform, sample_rate = load_audio(
        f"tar://{archive_path}::riffusion/tone.m4a",
        start_sec=0.2,
        duration_sec=0.25,
        archive_offset=info.offset_data,
        archive_size=info.size,
    )
    assert sample_rate == 24_000
    assert waveform.shape == (6000,)
    assert waveform.abs().max().item() > 0.01


def test_load_indexed_tar_wave(tmp_path) -> None:
    payload = _wave_bytes()
    archive_path = tmp_path / "audio.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo("sample.wav")
        info.size = len(payload)
        archive.addfile(info, BytesIO(payload))
    with tarfile.open(archive_path, "r") as archive:
        info = archive.getmember("sample.wav")
    waveform, sample_rate = load_audio(
        f"tar://{archive_path}::sample.wav",
        archive_offset=info.offset_data,
        archive_size=info.size,
    )
    assert sample_rate == 24_000
    assert waveform.shape == (2400,)


def test_permanent_indexed_tar_cache_hit_and_byte_budget(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_payload = _wave_bytes(1000)
    second_payload = _wave_bytes(2000)
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    first_info = _write_indexed_tar(first, first_payload)
    second_info = _write_indexed_tar(second, second_payload)
    reset_audio_io_state()
    monkeypatch.setattr(
        audio_module,
        "_ARCHIVE_PAYLOAD_CACHE_MAX_BYTES",
        len(first_payload) + 16,
    )
    calls = 0
    original_pread = os.pread

    def counted_pread(descriptor, size, offset):
        nonlocal calls
        calls += 1
        return original_pread(descriptor, size, offset)

    monkeypatch.setattr(audio_module.os, "pread", counted_pread)
    uri = f"tar://{first}::sample.wav"
    kwargs = _asset_kwargs(first, first_info, first_payload)
    load_audio(uri, **kwargs)
    load_audio(uri, **kwargs)
    assert calls == 1
    assert audio_module._ARCHIVE_PAYLOAD_CACHE_BYTES == len(first_payload)

    load_audio(uri, **{**kwargs, "asset_revision": "asset-r1-next"})
    assert calls == 2
    load_audio(uri, **{**kwargs, "shard_sha256": "0" * 64})
    assert calls == 3

    load_audio(
        f"tar://{second}::sample.wav",
        **_asset_kwargs(second, second_info, second_payload, revision="asset-r2"),
    )
    assert audio_module._ARCHIVE_PAYLOAD_CACHE_BYTES <= len(first_payload) + 16
    assert len(audio_module._ARCHIVE_PAYLOAD_CACHE) == 1


def test_permanent_indexed_tar_replacement_invalidates_fd_and_payload(
    tmp_path,
) -> None:
    archive = tmp_path / "audio.tar"
    first_payload = _wave_bytes(1000)
    first_info = _write_indexed_tar(archive, first_payload)
    reset_audio_io_state()
    waveform, _ = load_audio(
        f"tar://{archive}::sample.wav",
        **_asset_kwargs(archive, first_info, first_payload),
    )
    assert waveform.mean().item() == pytest.approx(1000 / 32768)

    replacement = tmp_path / "replacement.tar"
    second_payload = _wave_bytes(3000)
    second_info = _write_indexed_tar(replacement, second_payload)
    os.replace(replacement, archive)
    waveform, _ = load_audio(
        f"tar://{archive}::sample.wav",
        **_asset_kwargs(
            archive,
            second_info,
            second_payload,
            revision="asset-r2",
        ),
    )
    assert waveform.mean().item() == pytest.approx(3000 / 32768)


def test_permanent_indexed_tar_rejects_short_read_truncation_and_bounds(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _wave_bytes()
    archive = tmp_path / "audio.tar"
    info = _write_indexed_tar(archive, payload)
    kwargs = _asset_kwargs(archive, info, payload)
    reset_audio_io_state()
    original_pread = os.pread
    calls = 0

    def terminal_short_read(descriptor, size, offset):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_pread(descriptor, max(1, size // 2), offset)
        return b""

    monkeypatch.setattr(audio_module.os, "pread", terminal_short_read)
    with pytest.raises(AudioAssetIntegrityError, match="short read"):
        load_audio(f"tar://{archive}::sample.wav", **kwargs)

    monkeypatch.setattr(audio_module.os, "pread", original_pread)
    reset_audio_io_state()
    initial_stat = archive.stat()

    def drifting_pread(descriptor, size, offset):
        result = original_pread(descriptor, size, offset)
        os.utime(
            archive,
            ns=(
                initial_stat.st_atime_ns,
                initial_stat.st_mtime_ns + 1_000_000_000,
            ),
        )
        return result

    monkeypatch.setattr(audio_module.os, "pread", drifting_pread)
    with pytest.raises(AudioAssetIntegrityError, match="drift"):
        load_audio(f"tar://{archive}::sample.wav", **kwargs)

    monkeypatch.setattr(audio_module.os, "pread", original_pread)
    reset_audio_io_state()
    with archive.open("r+b") as handle:
        handle.truncate(info.offset_data + info.size - 1)
    with pytest.raises(AudioAssetIntegrityError, match="range is out of bounds"):
        load_audio(f"tar://{archive}::sample.wav", **kwargs)

    reset_audio_io_state()
    with pytest.raises(AudioAssetIntegrityError, match="range is out of bounds"):
        load_audio(
            f"tar://{archive}::sample.wav",
            **{
                **kwargs,
                "archive_offset": archive.stat().st_size + 1,
                "archive_size": 1,
                "shard_sha256": hashlib.sha256(
                    archive.read_bytes()
                ).hexdigest(),
            },
        )


def test_permanent_indexed_tar_rejects_payload_sha_mismatch(tmp_path) -> None:
    payload = _wave_bytes()
    archive = tmp_path / "audio.tar"
    info = _write_indexed_tar(archive, payload)
    kwargs = _asset_kwargs(archive, info, payload)
    kwargs["payload_sha256"] = "0" * 64
    reset_audio_io_state()
    with pytest.raises(AudioAssetIntegrityError, match="payload SHA-256"):
        load_audio(f"tar://{archive}::sample.wav", **kwargs)


def test_permanent_asset_integrity_error_never_degrades_to_silence(
    tmp_path,
) -> None:
    payload = _wave_bytes()
    archive = tmp_path / "audio.tar"
    info = _write_indexed_tar(archive, payload)
    manifest = tmp_path / "permanent.jsonl"
    audio = {
        "duration_sec": 0.1,
        **_asset_kwargs(archive, info, payload),
    }
    audio["payload_sha256"] = "0" * 64
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "permanent",
                "split": "train",
                "audio_path": f"tar://{archive}::sample.wav",
                "audio": audio,
                "training": {"loss_heads": {"ctc": False}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = TokenizerDataset(
        manifest,
        stage=1,
        max_duration_sec=1.0,
        random_crop=False,
    )
    with pytest.raises(AudioAssetIntegrityError, match="payload SHA-256"):
        dataset[0]


def test_partial_permanent_asset_metadata_never_degrades_to_silence(
    tmp_path,
) -> None:
    payload = _wave_bytes()
    archive = tmp_path / "partial.tar"
    info = _write_indexed_tar(archive, payload)
    manifest = tmp_path / "partial.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "partial",
                "split": "train",
                "audio_path": f"tar://{archive}::sample.wav",
                "audio": {
                    "duration_sec": 0.1,
                    "archive_offset": info.offset_data,
                    "archive_size": info.size,
                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                },
                "training": {"loss_heads": {"ctc": False}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = TokenizerDataset(
        manifest,
        stage=1,
        max_duration_sec=1.0,
        random_crop=False,
    )
    with pytest.raises(AudioAssetIntegrityError, match="must declare"):
        dataset[0]


def test_permanent_asset_decode_error_never_degrades_to_silence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "decode-error.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "decode-error",
                "split": "train",
                "audio_path": f"tar://{tmp_path / 'asset.tar'}::sample.wav",
                "audio": {
                    "duration_sec": 0.1,
                    "archive_offset": 0,
                    "archive_size": 1,
                    "asset_id": "asset-1",
                    "payload_sha256": "1" * 64,
                    "asset_revision": "asset-r1",
                    "shard_sha256": "2" * 64,
                },
                "training": {"loss_heads": {"ctc": False}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_decode(*_args, **_kwargs):
        raise ValueError("Analog decoder rejects payload")

    monkeypatch.setattr(data_module, "load_audio", fail_decode)
    dataset = TokenizerDataset(
        manifest,
        stage=1,
        max_duration_sec=1.0,
        random_crop=False,
    )
    with pytest.raises(AudioAssetIntegrityError, match="Permanent audio asset decoding failed"):
        dataset[0]


def test_legacy_indexed_tar_integrity_error_keeps_silence_compatibility(
    tmp_path,
) -> None:
    payload = _wave_bytes()
    archive = tmp_path / "legacy.tar"
    info = _write_indexed_tar(archive, payload)
    manifest = tmp_path / "legacy.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "legacy",
                "split": "train",
                "audio_path": f"tar://{archive}::sample.wav",
                "audio": {
                    "duration_sec": 0.1,
                    "archive_offset": info.offset_data,
                    "archive_size": info.size + archive.stat().st_size,
                },
                "training": {"loss_heads": {"ctc": False}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = TokenizerDataset(
        manifest,
        stage=1,
        max_duration_sec=1.0,
        random_crop=False,
    )
    with pytest.warns(RuntimeWarning, match="decode failed"):
        sample = dataset[0]
    assert sample["decode_failed"] is True


def test_audio_caches_are_cleared_after_fork(tmp_path) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("multiprocessing fork is not supported on this platform")
    payload = _wave_bytes()
    archive = tmp_path / "fork.tar"
    info = _write_indexed_tar(archive, payload)
    reset_audio_io_state()
    load_audio(
        f"tar://{archive}::sample.wav",
        **_asset_kwargs(archive, info, payload),
    )
    assert audio_module._ARCHIVE_PAYLOAD_CACHE
    assert audio_module._BINARY_CACHE

    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    process = context.Process(target=_fork_audio_cache_state, args=(queue,))
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 0
    assert queue.get(timeout=1) == (0, 0, 0)


def test_audio_caches_start_empty_after_spawn() -> None:
    if "spawn" not in multiprocessing.get_all_start_methods():
        pytest.skip("multiprocessing spawn is not supported on this platform")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_fork_audio_cache_state, args=(queue,))
    process.start()
    process.join(timeout=60)
    assert process.exitcode == 0
    assert queue.get(timeout=1) == (0, 0, 0)


def test_file_and_zip_uri_keep_working(tmp_path) -> None:
    payload = _wave_bytes()
    loose = tmp_path / "loose.wav"
    loose.write_bytes(payload)
    waveform, sample_rate = load_audio(f"file://{loose}")
    assert sample_rate == 24_000
    assert waveform.shape == (2400,)

    archive = tmp_path / "audio.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("inside.wav", payload)
    waveform, sample_rate = load_audio(f"zip://{archive}::inside.wav")
    assert sample_rate == 24_000
    assert waveform.shape == (2400,)


def test_zip_password_is_forwarded_and_cache_key_redacted(
    tmp_path, monkeypatch
) -> None:
    payload = _wave_bytes()
    archive = tmp_path / "protected.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("inside.wav", payload)
    secret = b"unit-" + b"credential"
    original = zipfile.ZipFile.read
    received: list[bytes | None] = []

    def observed(self, name, pwd=None):
        received.append(pwd)
        return original(self, name, pwd=pwd)

    reset_audio_io_state()
    monkeypatch.setattr(zipfile.ZipFile, "read", observed)
    waveform, sample_rate = load_audio(
        f"zip://{archive}::inside.wav",
        password=secret,
    )

    assert sample_rate == 24_000
    assert waveform.shape == (2400,)
    assert received == [secret]
    assert all(
        secret.decode() not in key
        for key in audio_module._ARCHIVE_PAYLOAD_CACHE
    )
    rotated = b"rotated-" + b"credential"
    load_audio(
        f"zip://{archive}::inside.wav",
        password=rotated,
    )
    assert received == [secret, rotated]
    assert all(
        value.decode() not in key
        for value in (secret, rotated)
        for key in audio_module._ARCHIVE_PAYLOAD_CACHE
    )


def test_zip_password_error_never_echoes_secret(
    tmp_path, monkeypatch
) -> None:
    archive = tmp_path / "protected.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("inside.wav", _wave_bytes())
    secret = b"do-" + b"not-log-this"

    def rejected(_self, _name, pwd=None):
        raise RuntimeError(f"bad credential {pwd!r}")

    reset_audio_io_state()
    monkeypatch.setattr(zipfile.ZipFile, "read", rejected)
    with pytest.raises(EncryptedZipReadError) as caught:
        load_audio(
            f"zip://{archive}::inside.wav",
            password=secret,
        )
    assert secret.decode() not in str(caught.value)


def test_tokenizer_dataset_password_env_fails_fast_and_injects_bytes(
    tmp_path, monkeypatch
) -> None:
    env_name = "OQM_TEST_TOKENIZER_ZIP_CREDENTIAL"
    secret = "dataset-" + "credential"
    monkeypatch.delenv(env_name, raising=False)
    manifest = tmp_path / "encrypted.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "encrypted",
                "split": "train",
                "audio_path": f"zip://{tmp_path / 'archive.zip'}::inside.wav",
                "audio": {
                    "duration_sec": 0.1,
                    "password_env": env_name,
                },
                "training": {"loss_heads": {"ctc": False}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = TokenizerDataset(
        manifest,
        stage=1,
        max_duration_sec=1.0,
        random_crop=False,
    )
    with pytest.raises(AudioCredentialError, match=env_name):
        dataset[0]
    monkeypatch.setenv(env_name, "")
    with pytest.raises(AudioCredentialError, match=env_name):
        dataset[0]

    def unexpected_prefetch(*_args, **_kwargs):
        raise AssertionError("Should not be done first when credentials are missing batch prefetch")

    monkeypatch.delenv(env_name)
    monkeypatch.setattr(
        data_module,
        "prefetch_parquet_audio",
        unexpected_prefetch,
    )
    with pytest.raises(AudioCredentialError, match=env_name):
        dataset.__getitems__([0])

    received: list[bytes | None] = []

    def fake_load_audio(_path, **kwargs):
        received.append(kwargs.get("password"))
        return torch.zeros(2400), 24_000

    monkeypatch.setenv(env_name, secret)
    monkeypatch.setattr(data_module, "load_audio", fake_load_audio)
    sample = dataset[0]
    assert sample["decode_failed"] is False
    assert received == [secret.encode()]
    assert secret not in manifest.read_text(encoding="utf-8")

    def reject_credential(_path, **_kwargs):
        raise EncryptedZipReadError("Secure encryption ZIP Error")

    monkeypatch.setattr(data_module, "load_audio", reject_credential)
    with pytest.raises(AudioCredentialError, match=env_name):
        dataset[0]


def test_tokenizer_dataset_rejects_invalid_or_plaintext_credential_ref(
    tmp_path, monkeypatch
) -> None:
    base = {
        "sample_id": "invalid-credential",
        "split": "train",
        "audio_path": f"zip://{tmp_path / 'archive.zip'}::inside.wav",
        "audio": {"duration_sec": 0.1, "password_env": "not-valid-name"},
        "training": {"loss_heads": {"ctc": False}},
    }
    manifest = tmp_path / "invalid-credential.jsonl"
    manifest.write_text(json.dumps(base) + "\n", encoding="utf-8")
    dataset = TokenizerDataset(
        manifest,
        stage=1,
        max_duration_sec=1.0,
        random_crop=False,
    )
    with pytest.raises(AudioCredentialError, match="valid environment variable name"):
        dataset[0]

    base["audio"] = {"duration_sec": 0.1, "password": "forbidden"}
    manifest.write_text(json.dumps(base) + "\n", encoding="utf-8")
    dataset = TokenizerDataset(
        manifest,
        stage=1,
        max_duration_sec=1.0,
        random_crop=False,
    )
    with pytest.raises(AudioCredentialError, match="must not contain"):
        dataset[0]


def test_tokenizer_dataset_returns_per_record_vq_enabled(tmp_path) -> None:
    audio = tmp_path / "vq.wav"
    audio.write_bytes(_wave_bytes())
    manifest = tmp_path / "vq.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "vq-off",
                "split": "train",
                "audio_path": str(audio),
                "audio": {"duration_sec": 0.1},
                "training": {
                    "loss_heads": {
                        "ctc": False,
                        "mel": True,
                        "chroma": True,
                        "vq": False,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = TokenizerDataset(
        manifest,
        stage=4,
        max_duration_sec=1.0,
        random_crop=False,
    )
    sample = dataset[0]
    assert sample["decode_failed"] is False
    assert sample["vq_enabled"] is False

    manifest.write_text(
        json.dumps(
            {
                "sample_id": "decode-failed",
                "split": "train",
                "audio_path": str(tmp_path / "missing.wav"),
                "audio": {"duration_sec": 0.1},
                "training": {"loss_heads": {"vq": True}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    failed_dataset = TokenizerDataset(
        manifest,
        stage=4,
        max_duration_sec=1.0,
        random_crop=False,
    )
    with pytest.warns(RuntimeWarning, match="decode failed"):
        failed = failed_dataset[0]
    assert failed["decode_failed"] is True
    assert failed["vq_enabled"] is False
    assert failed["ctc_sample_weight"] == 0.0
    assert failed["mel_sample_weight"] == 0.0
    assert failed["chroma_sample_weight"] == 0.0
    assert failed["vq_sample_weight"] == 0.0


def test_tokenizer_dataset_resolves_independent_head_weights(tmp_path) -> None:
    audio = tmp_path / "head-weights.wav"
    audio.write_bytes(_wave_bytes())
    manifest = tmp_path / "head-weights.jsonl"
    record = {
        "sample_id": "weighted",
        "split": "train",
        "audio_path": str(audio),
        "audio": {"duration_sec": 0.1},
        "lyrics": "a",
        "training": {
            "sample_weight": 0.75,
            "ctc_weight": 0.5,
            "mel_weight": 0.25,
            "chroma_weight": 0.0,
            "vq_weight": 1.5,
            "loss_heads": {
                "ctc": True,
                "mel": True,
                "chroma": True,
                "vq": True,
            },
        },
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = TokenizerDataset(
        manifest,
        stage=4,
        max_duration_sec=1.0,
        random_crop=False,
    )

    sample = dataset[0]
    assert sample["ctc_sample_weight"] == pytest.approx(0.5)
    assert sample["mel_sample_weight"] == pytest.approx(0.25)
    assert sample["chroma_sample_weight"] == 0.0
    assert sample["vq_sample_weight"] == pytest.approx(1.5)


    record["training"] = {
        "sample_weight": 0.75,
        "loss_heads": {
            "ctc": True,
            "mel": True,
            "chroma": True,
            "vq": True,
        },
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    legacy = TokenizerDataset(
        manifest,
        stage=4,
        max_duration_sec=1.0,
        random_crop=False,
    )[0]
    assert legacy["ctc_sample_weight"] == pytest.approx(0.75)
    assert legacy["mel_sample_weight"] == pytest.approx(0.75)
    assert legacy["chroma_sample_weight"] == pytest.approx(0.75)
    assert legacy["vq_sample_weight"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mel_weight", -0.1),
        ("chroma_weight", float("inf")),
        ("vq_weight", float("nan")),
    ],
)
def test_tokenizer_dataset_rejects_invalid_independent_head_weights(
    tmp_path,
    field,
    value,
) -> None:
    audio = tmp_path / f"invalid-{field}.wav"
    audio.write_bytes(_wave_bytes())
    manifest = tmp_path / f"invalid-{field}.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": f"invalid-{field}",
                "split": "train",
                "audio_path": str(audio),
                "audio": {"duration_sec": 0.1},
                "training": {
                    field: value,
                    "loss_heads": {"ctc": False},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = TokenizerDataset(
        manifest,
        stage=4,
        max_duration_sec=1.0,
        random_crop=False,
    )
    with pytest.raises(ValueError, match="non-negative finite number"):
        dataset[0]


def _write_binary_parquet(path, payloads, *, row_group_size=2) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table({"audio": pa.array(payloads, type=pa.binary())})
    pq.write_table(table, path, row_group_size=row_group_size)


def test_load_parquet_binary_and_hf_struct(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    binary = tmp_path / "binary.parquet"
    _write_binary_parquet(binary, [_wave_bytes(1000)])
    waveform, sample_rate = load_audio(
        f"parquet://{binary}::rg=0&row=0&col=audio"
    )
    assert sample_rate == 24_000
    assert waveform.mean().item() == pytest.approx(1000 / 32768)

    external = tmp_path / "external.wav"
    external.write_bytes(_wave_bytes(2000))
    struct = tmp_path / "struct.parquet"
    audio_type = pa.struct(
        [
            pa.field("bytes", pa.binary()),
            pa.field("path", pa.string()),
        ]
    )
    table = pa.table(
        {
            "audio": pa.array(
                [
                    {"bytes": _wave_bytes(3000), "path": None},
                    {"bytes": None, "path": external.name},
                ],
                type=audio_type,
            )
        }
    )
    pq.write_table(table, struct, row_group_size=2)
    embedded, _ = load_audio(
        f"parquet://{struct}::rg=0&row=0&col=audio"
    )
    referenced, _ = load_audio(
        f"parquet://{struct}::rg=0&row=1&col=audio"
    )
    assert embedded.mean().item() == pytest.approx(3000 / 32768)
    assert referenced.mean().item() == pytest.approx(2000 / 32768)


def test_parquet_local_mirror_is_strict_and_preserves_uri(
    tmp_path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    mirror_root = tmp_path / "mirror"
    source = source_root / "nested/audio.parquet"
    mirror = mirror_root / "nested/audio.parquet"
    source.parent.mkdir(parents=True)
    mirror.parent.mkdir(parents=True)
    _write_binary_parquet(source, [_wave_bytes(1000)])
    _write_binary_parquet(mirror, [_wave_bytes(3000)])
    uri = f"parquet://{source}::rg=0&row=0&col=audio"
    monkeypatch.setenv(
        "OQM_PARQUET_LOCAL_MIRROR",
        f"{source_root}::{mirror_root}",
    )
    reset_audio_io_state()
    seen = []
    original = audio_module._read_parquet_row_group

    def recorded(ref):
        seen.append(ref)
        return original(ref)

    monkeypatch.setattr(audio_module, "_read_parquet_row_group", recorded)
    waveform, _ = load_audio(uri)
    assert waveform.mean().item() == pytest.approx(3000 / 32768)
    assert seen[0].uri == uri
    assert seen[0].path == mirror.resolve()

    reset_audio_io_state()
    mirror.unlink()
    with pytest.raises(FileNotFoundError, match="refusing to fall back"):
        load_audio(uri)


def test_parquet_batch_prefetch_reads_row_group_once(
    tmp_path, monkeypatch
) -> None:
    parquet = tmp_path / "batch.parquet"
    _write_binary_parquet(
        parquet, [_wave_bytes(1000), _wave_bytes(2000)]
    )
    paths = [
        f"parquet://{parquet}::rg=0&row={row}&col=audio"
        for row in range(2)
    ]
    reset_audio_io_state()
    configure_parquet_row_group_cache(1)
    original = audio_module._read_parquet_row_group
    calls = 0

    def counted(ref):
        nonlocal calls
        calls += 1
        return original(ref)

    monkeypatch.setattr(audio_module, "_read_parquet_row_group", counted)
    prefetch_parquet_audio(paths)
    assert [load_audio(path)[0].numel() for path in paths] == [2400, 2400]
    assert calls == 1


def test_tokenizer_dataset_getitems_prefetches_same_row_group_once(
    tmp_path, monkeypatch
) -> None:
    parquet = tmp_path / "dataset.parquet"
    _write_binary_parquet(
        parquet, [_wave_bytes(1000), _wave_bytes(2000)]
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": f"sample-{row}",
                    "split": "train",
                    "source": {
                        "uri": (
                            f"parquet://{parquet}::"
                            f"rg=0&row={row}&col=audio"
                        )
                    },
                    "audio": {"duration_sec": 0.1},
                    "training": {"loss_heads": {"ctc": False}},
                }
            )
            + "\n"
            for row in range(2)
        ),
        encoding="utf-8",
    )
    reset_audio_io_state()
    original = audio_module._read_parquet_row_group
    calls = 0

    def counted(ref):
        nonlocal calls
        calls += 1
        return original(ref)

    monkeypatch.setattr(audio_module, "_read_parquet_row_group", counted)
    dataset = TokenizerDataset(
        manifest,
        stage=1,
        max_duration_sec=1.0,
        random_crop=False,
        parquet_row_group_cache_size=1,
    )
    samples = dataset.__getitems__([0, 1])
    assert [sample["waveform"].numel() for sample in samples] == [2400, 2400]
    assert calls == 1


def test_parquet_lru_eviction_and_file_identity(
    tmp_path, monkeypatch
) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _write_binary_parquet(first, [_wave_bytes(1000)])
    _write_binary_parquet(second, [_wave_bytes(2000)])
    first_uri = f"parquet://{first}::rg=0&row=0&col=audio"
    second_uri = f"parquet://{second}::rg=0&row=0&col=audio"
    reset_audio_io_state()
    configure_parquet_row_group_cache(1)
    original = audio_module._read_parquet_row_group
    calls = 0

    def counted(ref):
        nonlocal calls
        calls += 1
        return original(ref)

    monkeypatch.setattr(audio_module, "_read_parquet_row_group", counted)
    load_audio(first_uri)
    load_audio(second_uri)
    load_audio(first_uri)
    assert calls == 3

    previous_mtime = first.stat().st_mtime_ns
    _write_binary_parquet(first, [_wave_bytes(4000)])
    bumped = max(first.stat().st_mtime_ns, previous_mtime + 1_000_000)
    os.utime(first, ns=(bumped, bumped))
    replaced, _ = load_audio(first_uri)
    assert calls == 4
    assert replaced.mean().item() == pytest.approx(4000 / 32768)


@pytest.mark.parametrize(
    "tail",
    [
        "rg=0&row=0",
        "rg=-1&row=0&col=audio",
        "rg=00&row=0&col=audio",
        "rg=0&row=0&col=audio&extra=1",
        "rg=0&row=0&row=1&col=audio",
    ],
)
def test_parquet_uri_rejects_invalid_parameters(tmp_path, tail) -> None:
    parquet = tmp_path / "bad-uri.parquet"
    _write_binary_parquet(parquet, [_wave_bytes()])
    with pytest.raises(ValueError, match="Parquet URI"):
        load_audio(f"parquet://{parquet}::{tail}")


def test_parquet_rejects_out_of_range_and_null(tmp_path) -> None:
    parquet = tmp_path / "range.parquet"
    _write_binary_parquet(parquet, [_wave_bytes(), None])
    with pytest.raises(IndexError, match="row group"):
        load_audio(f"parquet://{parquet}::rg=3&row=0&col=audio")
    with pytest.raises(IndexError, match="row out of bounds"):
        load_audio(f"parquet://{parquet}::rg=0&row=3&col=audio")
    with pytest.raises(ValueError, match="null"):
        load_audio(f"parquet://{parquet}::rg=0&row=1&col=audio")
