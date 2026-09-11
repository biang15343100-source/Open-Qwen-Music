
from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


GRANULARITY_WHOLE = "whole_song"


GRANULARITY_PHRASE = "phrase"


GRANULARITY_NOTE = "note"

GRANULARITIES = (GRANULARITY_WHOLE, GRANULARITY_PHRASE, GRANULARITY_NOTE)


DEFAULT_DURATION_GATE: dict[str, tuple[float, float]] = {
    GRANULARITY_WHOLE: (20.0, 600.0),
    GRANULARITY_PHRASE: (1.0, 600.0),
    GRANULARITY_NOTE: (0.5, 600.0),
}


@dataclass(frozen=True, slots=True)
class AudioRef:

    kind: str  # "plain" | "zip" | "tar"
    container: Path
    member: str
    offset: int | None = None
    size: int | None = None

    @property
    def suffix(self) -> str:
        name = self.member or self.container.name
        return Path(name).suffix.lower()

    def describe(self) -> str:
        if self.kind == "plain":
            return str(self.container)
        return f"{self.kind}://{self.container}::{self.member}"


def parse_audio_ref(
    audio_path: str, dataset_dir: Path, audio: dict[str, Any] | None = None
) -> AudioRef:

    audio = audio or {}
    offset = audio.get("archive_offset")
    size = audio.get("archive_size")

    for prefix, kind in (("zip://", "zip"), ("tar://", "tar")):
        if audio_path.startswith(prefix):
            container, sep, member = audio_path[len(prefix) :].partition("::")
            if not sep:
                raise ValueError(f"{kind} reference is missing `::member name`:{audio_path!r}")
            path = Path(container)
            if not path.is_absolute():
                path = dataset_dir / container
            return AudioRef(
                kind=kind,
                container=path,
                member=member,
                offset=None if offset is None else int(offset),
                size=None if size is None else int(size),
            )

    return AudioRef(kind="plain", container=Path(audio_path), member="")


class ArchiveError(RuntimeError):
    pass


_TAR_BLOCK = 512


def _tar_header_at(handle: Any, header_offset: int) -> dict[str, Any] | None:

    handle.seek(header_offset)
    block = handle.read(_TAR_BLOCK)
    if len(block) < _TAR_BLOCK:
        return None
    magic = block[257:263]
    if magic not in (b"ustar\x00", b"ustar "):
        return None
    name = block[0:100].split(b"\x00", 1)[0].decode("utf-8", "replace")
    prefix = block[345:500].split(b"\x00", 1)[0].decode("utf-8", "replace")
    raw_size = block[124:136].split(b"\x00", 1)[0].strip()
    try:
        size = int(raw_size, 8) if raw_size else -1
    except ValueError:
        size = -1
    return {"name": name, "prefix": prefix, "size": size}


def verify_tar_member(handle: Any, ref: AudioRef) -> str:

    if ref.offset is None or ref.size is None:
        return "archive_offset/archive_size is missing; member identity cannot be verified"
    header = _tar_header_at(handle, ref.offset - _TAR_BLOCK)
    if header is None:
        return f"offset-512={ref.offset - _TAR_BLOCK} is not a valid tar header"
    full = f"{header['prefix']}/{header['name']}" if header["prefix"] else header["name"]
    name_ok = full == ref.member or header["name"] == ref.member
    if not name_ok:

        name_ok = bool(header["name"]) and ref.member.startswith(header["name"])
    size_ok = header["size"] == ref.size
    if name_ok and size_ok:
        return "ok"
    return (
        f"tar header does not match reference: header_name={full!r} vs member={ref.member!r}; "
        f"header_size={header['size']} vs archive_size={ref.size}"
    )


def read_member(ref: AudioRef, *, verify: bool = True) -> tuple[bytes, str]:

    if ref.kind == "plain":
        return ref.container.read_bytes(), "n/a"

    if not ref.container.exists():
        raise ArchiveError(f"The container does not exist:{ref.container}")

    if ref.kind == "zip":


        with zipfile.ZipFile(ref.container) as archive:
            try:
                data = archive.read(ref.member)
            except KeyError as exc:
                raise ArchiveError(f"zip  {ref.member!r}") from exc
        return data, "ok"

    if ref.kind == "tar":
        if ref.offset is None or ref.size is None:
            raise ArchiveError(
                f"tar reference is missing archive_offset/archive_size:{ref.describe()}"
            )
        with ref.container.open("rb") as handle:
            identity = verify_tar_member(handle, ref) if verify else "skipped"
            if verify and identity != "ok":
                raise ArchiveError(f"{ref.describe()}:{identity}")
            handle.seek(ref.offset)
            data = handle.read(ref.size)
        if len(data) != ref.size:
            raise ArchiveError(
                f"{ref.describe()}:Read {len(data)} Byte,Statement {ref.size}"
            )
        return data, identity

    raise ArchiveError(f"Unknown package type {ref.kind!r}")


def verify_zip_offset(ref: AudioRef) -> str:

    if ref.offset is None:
        return "no-offset"
    with zipfile.ZipFile(ref.container) as archive:
        info = archive.getinfo(ref.member)
        with archive.open(info) as member:
            official = member.read()
    with ref.container.open("rb") as handle:
        handle.seek(int(ref.offset))
        raw = handle.read(int(ref.size or 0))
    if raw == official:
        return f"offset=data start (compress_type={info.compress_type})"
    if int(ref.offset) == info.header_offset:
        return f"offset=local header start (compress_type={info.compress_type})"
    return (
        f"offset semantics are unclear: header_offset={info.header_offset} "
        f"data_offset≈{info.header_offset + 30 + len(info.filename)} "
        f"declared={ref.offset} compress_type={info.compress_type}"
    )


@dataclass(frozen=True, slots=True)
class AudioInfo:
    duration_sec: float
    sample_rate: int
    channels: int
    decoder: str


def decode_audio_info(path: Path) -> AudioInfo:

    from .audio_io import DecodeError, probe as _probe

    try:
        result = _probe(path)
    except Exception as exc:  # noqa: BLE001 -
        if isinstance(exc, DecodeError):
            raise ArchiveError(str(exc)) from exc
        raise
    if result.duration_sec <= 0:
        size = path.stat().st_size if path.exists() else -1
        raise ArchiveError(f"is {result.duration_sec}({path.name},{size} Byte)")
    return AudioInfo(
        result.duration_sec, result.sample_rate, result.channels, result.prober
    )


_SAFE = re.compile(r"[^0-9A-Za-z._-]+")


def cache_name(manifest_sample_id: str, ref: AudioRef) -> str:

    stem = _SAFE.sub("_", manifest_sample_id) or hashlib.sha1(
        ref.describe().encode()
    ).hexdigest()[:24]
    return f"{stem}{ref.suffix or '.bin'}"


def materialize(
    ref: AudioRef, manifest_sample_id: str, cache_dir: Path, *, verify: bool = True
) -> tuple[Path, str, int]:

    if ref.kind == "plain":
        if not ref.container.exists():
            raise ArchiveError(f"Audio does not exist:{ref.container}")
        return ref.container, "n/a", ref.container.stat().st_size

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / cache_name(manifest_sample_id, ref)
    if target.exists() and ref.size and target.stat().st_size == ref.size:


        return target, "cached", target.stat().st_size

    data, identity = read_member(ref, verify=verify)
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, target)
    return target, identity, len(data)


def iter_manifest(path: Path) -> Iterator[dict[str, Any]]:

    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not legal JSON:{exc}") from exc


def _selection_key(seed: str, sample_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\x00{sample_id}".encode()).digest()


def sample_manifest(
    path: Path,
    count: int,
    *,
    seed: str = "corpus-v1",
    exclude: frozenset[str] | None = None,
    exclude_key: str = "",
) -> list[dict[str, Any]]:

    import heapq

    if count <= 0:
        return []
    heap: list[tuple[bytes, int, dict[str, Any]]] = []
    tiebreak = 0
    for row in iter_manifest(path):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            continue
        if exclude and exclude_key:
            token = extract_token(row, exclude_key)
            if token and token in exclude:
                continue
        key = _selection_key(seed, sample_id)
        tiebreak += 1

        item = (bytes(255 - b for b in key), tiebreak, row)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)
    return [row for _, _, row in sorted(heap, key=lambda x: x[0], reverse=True)]


_MUCHIN_SONGID = re.compile(r"/(\d+)_src$")


def extract_token(row: dict[str, Any], key: str) -> str:

    if key == "muchin_songid":
        group = str((row.get("source") or {}).get("group_id") or "")
        match = _MUCHIN_SONGID.search(group)
        return match.group(1) if match else ""
    raise KeyError(f"Unknown exclusion key:{key!r}")


DURATION_TOLERANCE_ABS = 1.0
DURATION_TOLERANCE_REL = 0.05


MIN_SAMPLES_FOR_VERDICT = 8


SYSTEMATIC_MISMATCH_RATE = 0.5


#:


MAGNITUDE_CONSISTENT_RANGE = (0.8, 1.25)


WINDOW_PIN_TOLERANCE = 0.15
WINDOW_PIN_RATE = 0.9


#:


MIXED_CONVENTION_RANGE = (0.05, 0.95)


@dataclass
class DurationAudit:

    dataset: str
    declared: list[float] = field(default_factory=list)
    decoded: list[float] = field(default_factory=list)
    crop_windows: set[float] = field(default_factory=set)
    mismatches: int = 0
    decode_failures: int = 0

    declared_is_crop: int = 0

    def observe(
        self, declared_sec: float | None, decoded_sec: float, crop_sec: float | None
    ) -> bool:

        self.decoded.append(decoded_sec)
        if crop_sec:
            self.crop_windows.add(float(crop_sec))
        if declared_sec is None:
            return False
        declared = float(declared_sec)
        self.declared.append(declared)
        if crop_sec and abs(declared - float(crop_sec)) <= WINDOW_PIN_TOLERANCE:
            self.declared_is_crop += 1
        tolerance = max(DURATION_TOLERANCE_ABS, DURATION_TOLERANCE_REL * declared)
        if abs(declared - decoded_sec) > tolerance:
            self.mismatches += 1
            return True
        return False

    def verdict(self) -> dict[str, Any]:
        n = len(self.declared)
        result: dict[str, Any] = {
            "dataset": self.dataset,
            "n_compared": n,
            "n_decoded": len(self.decoded),
            "n_mismatch": self.mismatches,
            "decode_failures": self.decode_failures,
            "declared_median": _median(self.declared),
            "decoded_median": _median(self.decoded),
            "crop_windows": sorted(self.crop_windows),
            "duration_source": "decoded",
        }
        if n < MIN_SAMPLES_FOR_VERDICT:
            result["verdict"] = "undetermined"
            result["detail"] = (
                f"Only {n} comparable samples are available; "
                f"at least {MIN_SAMPLES_FOR_VERDICT} are required."
            )
            return result

        rate = self.mismatches / n
        result["mismatch_rate"] = round(rate, 4)
        declared_med = result["declared_median"] or 0.0
        decoded_med = result["decoded_median"] or 0.0
        ratio = (decoded_med / declared_med) if declared_med else 0.0
        result["decoded_over_declared"] = round(ratio, 3)
        crop_rate = self.declared_is_crop / n
        result["declared_equals_crop_rate"] = round(crop_rate, 4)

        if MIXED_CONVENTION_RANGE[0] <= crop_rate <= MIXED_CONVENTION_RANGE[1]:
            result["verdict"] = "mixed_duration_conventions"
            result["mixed_declaration"] = True
            result["detail"] = (
                f"{self.declared_is_crop}/{n} records declare a crop-window duration; "
                "decoded duration is used for each record."
            )
            return result

        pinned = self._window_pinned()
        if pinned is not None:
            result["verdict"] = "decoded_duration_matches_crop_window"
            result["detail"] = (
                f"{pinned[1]:.0%} of decoded durations are within "
                f"{WINDOW_PIN_TOLERANCE}s of the {pinned[0]}s crop window."
            )
            return result

        low, high = MAGNITUDE_CONSISTENT_RANGE
        magnitude_ok = low <= ratio <= high
        result["magnitude_consistent"] = magnitude_ok
        if rate >= SYSTEMATIC_MISMATCH_RATE and not magnitude_ok:
            result["verdict"] = "systematic_magnitude_mismatch"
            result["detail"] = (
                f"{self.mismatches}/{n} declarations differ from decoded duration; "
                f"the median decoded-to-declared ratio is {ratio:.2f}."
            )
        elif rate >= SYSTEMATIC_MISMATCH_RATE:
            result["verdict"] = "record_level_mismatch"
            result["detail"] = (
                f"{self.mismatches}/{n} records exceed tolerance, while the median "
                f"ratio {ratio:.2f} remains within {MAGNITUDE_CONSISTENT_RANGE}."
            )
        elif self.mismatches:
            result["verdict"] = "partial_mismatch"
            result["detail"] = f"{self.mismatches}/{n} records exceed duration tolerance."
        else:
            result["verdict"] = "consistent"
            result["detail"] = "Declared and decoded durations agree within tolerance."
        return result

    def _window_pinned(self) -> tuple[float, float] | None:
        if not self.crop_windows or len(self.decoded) < MIN_SAMPLES_FOR_VERDICT:
            return None
        for window in sorted(self.crop_windows):
            hits = sum(
                1 for d in self.decoded if abs(d - window) <= WINDOW_PIN_TOLERANCE
            )
            rate = hits / len(self.decoded)
            if rate >= WINDOW_PIN_RATE:
                return window, rate
        return None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[mid], 3)
    return round((ordered[mid - 1] + ordered[mid]) / 2, 3)


WHOLE_SONG_MIN_MEDIAN_SEC = 20.0


PHRASE_MAX_MEDIAN_SEC = 60.0


def check_granularity(
    dataset: str, granularity: str, decoded: list[float]
) -> dict[str, Any]:

    result: dict[str, Any] = {
        "dataset": dataset,
        "granularity": granularity,
        "n": len(decoded),
        "decoded_median": _median(decoded),
    }
    if granularity not in GRANULARITIES:
        result["verdict"] = "not_applicable"
        result["detail"] = f"granularity is not declared ({granularity!r}); validation skipped."
        return result
    if len(decoded) < MIN_SAMPLES_FOR_VERDICT:
        result["verdict"] = "undetermined"
        result["detail"] = (
            f"only decoded {len(decoded)} records (minimum {MIN_SAMPLES_FOR_VERDICT}); "
            "not enough records to validate granularity. No conclusion was reached."
        )
        return result

    median = result["decoded_median"] or 0.0
    if granularity == GRANULARITY_WHOLE and median < WHOLE_SONG_MIN_MEDIAN_SEC:
        result["verdict"] = "declaration_mismatch"
        result["detail"] = (
            f"whole_song is declared, but the decoded median duration is only {median:.1f}s "
            f"(minimum {WHOLE_SONG_MIN_MEDIAN_SEC}s). Structure labels would be invalid."
        )
    elif granularity == GRANULARITY_PHRASE and median > PHRASE_MAX_MEDIAN_SEC:
        result["verdict"] = "declaration_mismatch"
        result["detail"] = (
            f"phrase is declared, but the decoded median duration is {median:.1f}s "
            f"(maximum {PHRASE_MAX_MEDIAN_SEC}s). Phrase routing would skip structure and sections."
        )
    else:
        result["verdict"] = "consistent"
        result["detail"] = f"median duration {median:.1f}s，matches the declared {granularity} granularity."
    return result
