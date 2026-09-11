
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from .base import LocalStage, StageReport, _shard_rank_fn
from ..corpus import (
    DEFAULT_DURATION_GATE,
    GRANULARITIES,
    GRANULARITY_WHOLE,
    ArchiveError,
    DurationAudit,
    check_granularity,
    decode_audio_info,
    materialize,
    parse_audio_ref,
    sample_manifest,
)
from ..store import ShardWriter

DEFAULT_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac")

_HASH_CHUNK = 1 << 20


def content_id(path: Path) -> str:

    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(_HASH_CHUNK))
        if size > 2 * _HASH_CHUNK:
            handle.seek(-_HASH_CHUNK, os.SEEK_END)
            digest.update(handle.read(_HASH_CHUNK))
    return "sha256:" + digest.hexdigest()[:32]


def probe_duration(path: Path) -> tuple[float, int, int]:

    from ..audio_io import probe

    result = probe(path)
    return result.duration_sec, result.sample_rate, result.channels


_SOURCE_KEYS = frozenset(
    {

        "root", "max_files",

        "manifest", "dataset_dir", "granularity", "sample_count", "sample_seed",
        "exclude_key", "exclude_file", "min_duration_sec", "max_duration_sec",

        "release", "preprocess_src", "release_granularity", "release_datasets",
        "release_statuses", "release_splits", "release_exclude_flags",
        "release_exclude_warn", "release_uid_file", "release_expected_lineage",
        "release_sample_id", "release_trust_audio_metadata",
        "release_materialize_workers",

        "dataset", "license_id", "language_hint",

        "wordless", "wordless_basis", "wordless_voicing",

        "duration_gate_ack",
    }
)


def _release_uids(source: dict[str, Any]) -> frozenset[bytes] | None:

    raw = source.get("release_uid_file")
    if raw is None:
        return None
    path = Path(str(raw))
    if not path.is_file():
        raise FileNotFoundError(f"release_uid_file does not exist: {path}")
    values: list[bytes] = []
    seen: set[bytes] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        try:
            uid = bytes.fromhex(token)
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno} is not hexadecimal uid:{token!r}") from exc
        if len(uid) != 8:
            raise ValueError(
                f"{path}:{lineno} UID is {len(uid)} bytes; oqm.corpus.v1 requires 8 bytes"
            )
        if uid in seen:
            raise ValueError(f"{path}:{lineno} has duplicate UID: {token.lower()}")
        seen.add(uid)
        values.append(uid)
    if not values:
        raise ValueError(
            f"release_uid_file {path} contains no UIDs; an empty list cannot mean all records"
        )
    return frozenset(values)


#:


VOICING_HUMAN = "human_voice"
VOICING_INSTRUMENT = "instrument_sample"
_VOICINGS = (VOICING_HUMAN, VOICING_INSTRUMENT)


def _source_wordless(source: dict[str, Any]) -> tuple[bool, str]:

    declared = source.get("wordless")
    voicing = str(source.get("wordless_voicing") or "").strip()
    if declared is None or declared is False:
        if voicing:
            raise ValueError(
                f"data source {source.get('dataset', '?')} wrote wordless_voicing="
                f"{voicing!r} without `wordless: true`"
            )
        return False, ""
    if declared is not True:
        raise ValueError(
            f"data source {source.get('dataset', '?')} of wordless={declared!r} "
            f"must be a Boolean; use true or false"
        )
    basis = str(source.get("wordless_basis") or "").strip()
    if not basis:
        raise ValueError(
            f"data source {source.get('dataset', '?')} declared wordless: true but did not write "
            "wordless_basis. Cite the source-level reason for this declaration."
        )
    if voicing not in _VOICINGS:
        raise ValueError(
            f"data source {source.get('dataset', '?')} declared wordless: true,"
            f"wordless_voicing={voicing or '(missing)'!r}; expected one of {list(_VOICINGS)}"
        )
    return True, voicing


def _assert_source_keys(sources: list[dict[str, Any]]) -> None:

    for index, source in enumerate(sources):
        unknown = sorted(set(source) - _SOURCE_KEYS)
        if unknown:
            raise ValueError(
                f"index.sources[{index}](dataset="
                f"{source.get('dataset', '?')}) contains unknown keys: {unknown}.\n"
                f"  Supported keys: {sorted(_SOURCE_KEYS)}"
            )


def _source_language(source: dict[str, Any]) -> Any:
    return source.get("language_hint")


#:


LANGUAGE_FROM_MANIFEST = "@manifest"


_KIND_KEYS: tuple[tuple[str, str], ...] = (
    ("root", "scan directory"),
    ("manifest", "read corpus manifest"),
    ("release", "read preprocessed release"),
)


def _source_kind(index: int, source: dict[str, Any]) -> str:

    given = [key for key, _ in _KIND_KEYS if source.get(key) is not None]
    if len(given) != 1:
        options = ", ".join(f"`{key}:` ({desc})" for key, desc in _KIND_KEYS)
        raise ValueError(
            f"index.sources[{index}](dataset={source.get('dataset', '?')})"
            f"must define exactly one of {options}; got {given or 'none'}"
        )
    return given[0]


def _release_granularity(source: dict[str, Any]) -> str:

    from ..release_corpus import GRANULARITY_FROM_RELEASE, RELEASE_GRANULARITIES

    declared = source.get("release_granularity")
    annotation_granularity = source.get("granularity")

    if declared is None:

        wanted = str(annotation_granularity or GRANULARITY_WHOLE)
        if wanted not in GRANULARITY_FROM_RELEASE.values():
            raise ValueError(
                f"data source {source.get('dataset', '?')} of granularity="
                f"{wanted!r} has no matching upstream granularity; set `release_granularity` "
                f"to one of {list(RELEASE_GRANULARITIES)}"
            )
        return wanted

    wanted = str(declared)
    if wanted not in RELEASE_GRANULARITIES:
        raise ValueError(
            f"data source {source.get('dataset', '?')} of release_granularity="
            f"{wanted!r} is invalid for oqm.corpus.v1; expected one of "
            f"{list(RELEASE_GRANULARITIES)}"
        )
    if wanted in GRANULARITY_FROM_RELEASE:
        expected = GRANULARITY_FROM_RELEASE[wanted]
        if annotation_granularity is not None and str(annotation_granularity) != expected:
            raise ValueError(
                f"data source {source.get('dataset', '?')} maps upstream {wanted!r} to "
                f"annotation granularity {annotation_granularity!r}; expected {expected!r}"
            )
        return wanted


    if annotation_granularity is None:
        raise ValueError(
            f"data source {source.get('dataset', '?')} uses upstream `clip` records without "
            "declaring annotation `granularity`. Set it to `whole_song` or `phrase` "
            "according to the release contents."
        )
    if str(annotation_granularity) not in GRANULARITIES:
        raise ValueError(
            f"data source {source.get('dataset', '?')} of granularity="
            f"{annotation_granularity!r} is invalid; expected one of {list(GRANULARITIES)}"
        )
    return wanted


def _duration_gate(source: dict[str, Any], granularity: str,
                   stage_default: tuple[float, float]) -> tuple[float, float]:

    low, high = DEFAULT_DURATION_GATE.get(granularity, stage_default)
    if granularity == GRANULARITY_WHOLE:
        low, high = stage_default
    if source.get("min_duration_sec") is not None:
        low = float(source["min_duration_sec"])
    if source.get("max_duration_sec") is not None:
        high = float(source["max_duration_sec"])
    return low, high


def _assert_duration_gate_kept_something(
    source: dict[str, Any], stats: dict[str, Any], gate: tuple[float, float]
) -> None:

    produced = int(stats.get("produced", 0))
    toll = int(stats.get("too_short", 0)) + int(stats.get("too_long", 0))
    if produced and toll <= produced:
        return


    if not int(stats.get("candidates", 0)):
        return


    if produced + toll == 0:
        return

    ack = str(source.get("duration_gate_ack") or "").strip()
    if ack:
        return

    dataset = source.get("dataset", "?")
    verdict = ("produced no records" if not produced
               else f"dropped {toll} records and retained {produced}")
    raise ValueError(
        f"Data source {dataset} with duration gate [{gate[0]}, {gate[1]}] seconds {verdict}. "
        f"Counts: too_short={stats.get('too_short', 0)}, too_long={stats.get('too_long', 0)}, "
        f"candidates={stats.get('candidates', 0)}, resumed={stats.get('resumed', 0)}, "
        f"duplicates={stats.get('duplicate', 0)}, failed={stats.get('failed', 0)}. "
        "Adjust the duration bounds or set `duration_gate_ack` with a reason."
    )


def _load_exclusions(source: dict[str, Any]) -> tuple[frozenset[str], str]:
    key = str(source.get("exclude_key") or "")
    path = source.get("exclude_file")
    if not key and not path:
        return frozenset(), ""
    if not (key and path):
        raise ValueError(
            f"data source {source.get('dataset', '?')} of `exclude_key` and "
            "`exclude_file` must be provided together"
        )
    file = Path(str(path))
    if not file.exists():
        raise FileNotFoundError(
            f"data source {source.get('dataset', '?')} exclude_file does not exist: {file}"
        )
    tokens = {
        line.strip()
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    return frozenset(tokens), key


class IndexStage(LocalStage):

    name = "index"
    depends_on = ()

    def iter_inputs(self) -> Iterable[dict[str, Any]]:


        return []

    def verify(self) -> dict[str, Any]:
        verdict = super().verify()
        sources = self.config.get("sources") or []
        batch_sources = [source for source in sources if source.get("release_uid_file")]
        if not batch_sources:
            return verdict
        if len(batch_sources) != len(sources) or any(
            str(source.get("release_sample_id") or "content") != "uid"
            for source in batch_sources
        ):
            verdict["problems"].append(
                "index mixes batch UID and non-batch sources; the expected set is undefined"
            )
            return verdict

        expected_uid: set[bytes] = set()
        for source in batch_sources:
            expected_uid.update(_release_uids(source) or ())
        expected = {f"oqm:{uid.hex()}" for uid in expected_uid}
        observed = {
            str(record.get("sample_id"))
            for record in self.store.iter_records()
            if record.get("sample_id")
        }
        missing = expected - observed
        extra = observed - expected
        verdict["expected"] = len(expected)
        verdict["missing"] = len(missing)
        verdict["extra"] = len(extra)
        if missing:
            verdict["problems"].append(
                f"batch UID set has {len(missing)} records missing from every index shard"
                f" (examples: {sorted(missing)[:3]}). Archive materialization failures "
                "must be reported here."
            )
        if extra:
            verdict["problems"].append(
                f"index contains {len(extra)} sample IDs outside the current batch"
                f" (examples: {sorted(extra)[:3]})"
            )
        return verdict


    def _iter_files(self, source: dict[str, Any]) -> Iterator[Path]:
        extensions = {
            ext.lower()
            for ext in (self.config.get("extensions") or DEFAULT_EXTENSIONS)
        }
        root = Path(str(source["root"]))
        if not root.exists():
            raise FileNotFoundError(
                f"data source {source.get('dataset')} of root does not exist:{root}"
            )
        limit = source.get("max_files")
        count = 0

        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            yield path
            count += 1
            if limit and count >= int(limit):
                break


    def run(self) -> StageReport:
        started = time.time()
        report = StageReport(stage=self.name)
        nnodes = int(os.environ.get("NNODES", "1"))
        node_rank = int(os.environ.get("NODE_RANK", "0"))
        distributed = bool(self.config.get("distributed_index", False)) and nnodes > 1
        if node_rank < 0 or node_rank >= nnodes:
            raise ValueError(f"index: NODE_RANK={node_rank} is out of bounds for NNODES={nnodes}")
        if distributed:
            mine = (node_rank,)
            own_paths = set(self.store.shard_paths(ranks=mine))
            foreign = [
                path for path in self.store.shard_paths() if path not in own_paths
            ]
            snapshot = self.store.read_resume_snapshot()
            if snapshot is None and foreign:
                raise RuntimeError(
                    f"index Distributed continuation see {len(foreign)} foreign node shards but no"
                    f" {self.store.resume_snapshot_path.name}"
                )
            done = (snapshot or set()) | self.store.completed_ids(ranks=mine)
        else:
            done = self.store.completed_ids()
        self._index_distributed = distributed
        self._index_nnodes = nnodes
        self._index_node_rank = node_rank
        self.store.clear_done()

        sources = self.config.get("sources") or []
        if not sources:
            raise ValueError("index.sources is empty; there are no data sources to scan")
        _assert_source_keys(sources)

        stage_gate = (
            float(self.config.get("min_duration_sec", 20.0)),
            float(self.config.get("max_duration_sec", 600.0)),
        )
        cache_root = Path(
            str(
                self.config.get("cache_dir")
                or (self.context.work_dir / "audio_cache")
            )
        )
        seen: set[str] = set(done)
        audits: list[dict[str, Any]] = []
        per_dataset: dict[str, dict[str, Any]] = {}


        self._release_cache: dict[tuple[str, str], Any] = {}

        with ShardWriter(self.store, rank=node_rank if distributed else 0) as writer:
            for position, source in enumerate(sources):
                kind = _source_kind(position, source)
                stats = self._run_source(
                    source, kind, writer, seen, done, report, stage_gate,
                    cache_root, audits,
                )
                per_dataset[str(source.get("dataset", f"source{position}"))] = stats

        notes = self._summarize(per_dataset, audits)


        if not distributed or node_rank == 0:
            self._write_corpus_audit(notes)
        notes.update(
            {
                "distributed_index": distributed,
                "node_rank": node_rank,
                "nnodes": nnodes,
            }
        )
        report.notes = notes
        report.seconds = time.time() - started
        timing = self.store.dir / f"_TIMING.node{node_rank:04d}.json"
        timing_tmp = timing.with_suffix(".json.tmp")
        timing_tmp.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False), encoding="utf-8"
        )
        os.replace(timing_tmp, timing)
        return self._finish(report)

    def _summarize(
        self,
        per_dataset: dict[str, dict[str, Any]],
        audits: list[dict[str, Any]],
    ) -> dict[str, Any]:

        granularity_counts: dict[str, int] = {}
        filtered_total = {"too_short": 0, "too_long": 0, "duplicate": 0, "excluded": 0}
        resumed_total = 0
        for stats in per_dataset.values():
            key = str(stats.get("granularity"))
            granularity_counts[key] = granularity_counts.get(key, 0) + int(
                stats.get("produced", 0)
            )
            resumed_total += int(stats.get("resumed", 0))
            for reason in filtered_total:
                filtered_total[reason] += int(stats.get(reason, 0))

        verdicts = {a["dataset"]: a.get("verdict") for a in audits}
        undetermined = sorted(k for k, v in verdicts.items() if v == "undetermined")
        magnitude_off = sorted(
            a["dataset"] for a in audits if a.get("magnitude_consistent") is False
        )
        gran_conflicts = sorted(
            name
            for name, stats in per_dataset.items()
            if (stats.get("granularity_check") or {}).get("verdict") == "declaration_mismatch"
        )

        trusted = sum(int(stats.get("metadata_trusted", 0)) for stats in per_dataset.values())
        decoded = sum(int(stats.get("decode_attempts", 0)) for stats in per_dataset.values())
        duration_source = (
            "mixed"
            if trusted and decoded
            else "release_metadata"
            if trusted
            else "decoded"
        )

        summary: dict[str, Any] = {
            "granularity_counts": granularity_counts,
            "filtered": filtered_total,


            "resumed": resumed_total,
            "duration_source": duration_source,
            "duration_source_counts": {
                "release_metadata": trusted,
                "decoded_attempts": decoded,
            },
            "duration_verdicts": verdicts,


            "declared_duration_is_not_real": magnitude_off,
            "granularity_conflicts": gran_conflicts,
            "by_dataset": per_dataset,
        }
        if undetermined:
            summary["undetermined_duration_audit"] = undetermined
            summary["undetermined_note"] = (
                f"{len(undetermined)} data sources lack enough comparable samples "
                "for a duration verdict."
            )
        if gran_conflicts:
            summary["granularity_conflict_note"] = (
                f"{gran_conflicts} have granularity declarations that conflict with "
                "decoded duration; review the configuration."
            )
        return summary

    def _write_corpus_audit(self, notes: dict[str, Any]) -> None:

        path = self.store.dir / "corpus_audit.json"
        per_dataset = notes.get("by_dataset") or {}
        unmeasured = sorted(
            name
            for name, stats in per_dataset.items()
            if not int(stats.get("decode_attempts", 0))
        )
        measured_any = len(unmeasured) < len(per_dataset)

        if not measured_any and path.exists():
            notes["corpus_audit_preserved"] = str(path)
            notes["corpus_audit_preserved_note"] = (
                "No audio was decoded in this run. The previous corpus audit is preserved; "
                "current routing counts remain in this stage report."
            )
            return

        if unmeasured and measured_any:
            notes["not_measured_this_run"] = unmeasured
            notes["partial_refresh_note"] = (
                f"{len(unmeasured)} data sources were not measured in this run; "
                "see not_measured_this_run."
            )

        path.write_text(
            json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _run_source(
        self,
        source: dict[str, Any],
        kind: str,
        writer: ShardWriter,
        seen: set[str],
        done: set[str],
        report: StageReport,
        stage_gate: tuple[float, float],
        cache_root: Path,
        audits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        dataset = str(source.get("dataset", "unknown"))
        granularity = str(source.get("granularity") or GRANULARITY_WHOLE)
        if granularity not in GRANULARITIES:
            raise ValueError(
                f"data source {dataset} of granularity={granularity!r} Don\'t know,"
                f"optional {list(GRANULARITIES)}."
            )
        low, high = _duration_gate(source, granularity, stage_gate)
        declared_language = _source_language(source)
        wordless, wordless_voicing = _source_wordless(source)

        stats: dict[str, Any] = {
            "dataset": dataset,
            "kind": kind,
            "granularity": granularity,
            "duration_gate": [low, high],
            "produced": 0,


            "resumed": 0,
            "duplicate": 0,
            "too_short": 0,
            "too_long": 0,
            "excluded": 0,
            "failed": 0,
            "candidates": 0,


            "decode_attempts": 0,


            "metadata_trusted": 0,
        }
        audit = DurationAudit(dataset=dataset)

        for payload in self._iter_candidates(source, kind, cache_root, stats, done):
            path: Path = payload["path"]
            stats["candidates"] += 1
            sample_id_mode = str(source.get("release_sample_id") or "content")
            if kind == "release" and sample_id_mode == "uid":
                sample_id = str(payload.get("release_sample_id") or "")
                if not sample_id:
                    raise ValueError(
                        f"data source {dataset} requires release_sample_id=uid,"
                        "but the adaptation layer does not give release_sample_id"
                    )
            elif sample_id_mode == "content":
                try:
                    sample_id = content_id(path)
                except OSError as exc:
                    report.failed += 1
                    stats["failed"] += 1
                    writer.write_error(f"path:{path}", exc)
                    continue
            else:
                raise ValueError(
                    f"data source {dataset} of release_sample_id={sample_id_mode!r} "
                    "Don\'t know,optional content / uid"
                )

            if sample_id in seen:
                report.skipped += 1
                if sample_id in done:

                    stats["resumed"] += 1
                else:

                    stats["duplicate"] += 1
                continue
            seen.add(sample_id)

            trust_release = kind == "release" and bool(
                source.get("release_trust_audio_metadata")
            )
            if trust_release:
                try:
                    duration = float(payload["manifest_duration_sec"])
                    sample_rate = int(payload["sample_rate_hint"])
                    channels = int(payload["channels_hint"])
                    if duration <= 0 or sample_rate <= 0 or channels <= 0:
                        raise ValueError(
                            f"Invalid metadata: duration={duration}, sample_rate={sample_rate}, "
                            f"channels={channels}"
                        )
                except (KeyError, TypeError, ValueError) as exc:
                    report.failed += 1
                    stats["failed"] += 1
                    writer.write_error(
                        sample_id,
                        ValueError(
                            "release_trust_audio_metadata is enabled, but valid upstream "
                            f"metadata is unavailable: {exc}"
                        ),
                    )
                    continue
                stats["metadata_trusted"] += 1
            else:
                stats["decode_attempts"] += 1
                try:
                    info = decode_audio_info(path)
                except Exception as exc:  # noqa: BLE001
                    report.failed += 1
                    stats["failed"] += 1
                    audit.decode_failures += 1
                    writer.write_error(sample_id, exc)
                    continue

                duration = info.duration_sec
                sample_rate = info.sample_rate
                channels = info.channels
                audit.observe(
                    payload.get("manifest_duration_sec"), duration, payload.get("crop_sec")
                )


            #


            if duration < low:
                report.skipped += 1
                stats["too_short"] += 1
                continue
            if duration > high:
                report.skipped += 1
                stats["too_long"] += 1
                continue

            record = {
                "sample_id": sample_id,


                # `release-whole-song` / `release-clip`.
                "dataset": (
                    (payload.get("extra") or {}).get("release_dataset_slug")
                    if kind == "release"
                    else None
                )
                or dataset,
                "audio_path": str(path),
                "duration_sec": round(duration, 3),
                "sample_rate": sample_rate,
                "channels": channels,
                "language_hint": payload.get("language_hint", declared_language),
                "license_id": source.get("license_id", "UNKNOWN"),


                "granularity": granularity,


                "wordless": wordless,


                "wordless_voicing": wordless_voicing,
            }
            record.update(payload.get("extra") or {})
            writer.write(record)
            report.produced += 1
            stats["produced"] += 1

        _assert_duration_gate_kept_something(source, stats, (low, high))
        stats["duration_audit"] = audit.verdict()
        stats["granularity_check"] = check_granularity(
            dataset, granularity, audit.decoded
        )
        audits.append(stats["duration_audit"])
        return stats

    def _iter_candidates(
        self,
        source: dict[str, Any],
        kind: str,
        cache_root: Path,
        stats: dict[str, Any],
        done: set[str],
    ) -> Iterator[dict[str, Any]]:
        if kind == "root":
            for path in self._iter_files(source):
                yield {"path": path}
            return
        if kind == "release":
            yield from self._iter_release_rows(source, cache_root, stats, done)
            return
        yield from self._iter_manifest_rows(source, cache_root, stats)

    def _iter_release_rows(
        self,
        source: dict[str, Any],
        cache_root: Path,
        stats: dict[str, Any],
        done: set[str],
    ) -> Iterator[dict[str, Any]]:

        from ..release_corpus import (
            Selection,
            assert_preflight_acknowledged,
            duration_gate_preflight,
            materialize_rows,
            open_release,
            select_rows,
        )

        dataset = str(source.get("dataset", "unknown"))
        cache_key = (
            str(Path(str(source["release"])).resolve()),
            str(source.get("preprocess_src") or ""),
        )
        release = self._release_cache.get(cache_key)
        if release is None:
            release = open_release(
                source["release"], preprocess_src=source.get("preprocess_src")
            )
            self._release_cache[cache_key] = release

        expected_lineage = str(source.get("release_expected_lineage") or "")
        if expected_lineage and release.lineage != expected_lineage:
            raise ValueError(
                f"data source {source.get('dataset', '?')} is based on release lineage "
                f"{expected_lineage!r},current release is {release.lineage!r}."
                "The UID plan belongs to a different corpus and cannot be reused."
            )
        selection = Selection(
            release_granularity=_release_granularity(source),
            datasets=tuple(source.get("release_datasets") or ()),
            statuses=tuple(source.get("release_statuses") or ("accepted",)),
            splits=tuple(source.get("release_splits") or ()),
            exclude_flags=tuple(
                source.get("release_exclude_flags")
                if source.get("release_exclude_flags") is not None
                else ("eval_holdout",)
            ),
            exclude_warn=bool(source.get("release_exclude_warn") or False),
            include_uids=_release_uids(source),
            sample_count=(
                None if source.get("sample_count") is None
                else int(source["sample_count"])
            ),
            sample_seed=str(source.get("sample_seed") or "oqm-release-v1"),
        )
        rows, select_audit = select_rows(release, selection)
        if (
            getattr(self, "_index_distributed", False)
            and str(source.get("release_sample_id") or "content") == "uid"
        ):
            shard_rank = _shard_rank_fn(self.context.repo_root)
            before = len(rows)
            rows = [
                row
                for row in rows
                if (


                    (
                        int(row["container_id"]) % self._index_nnodes
                        if row.get("sequential_only")
                        else shard_rank(
                            f"oqm:{bytes(row['uid']).hex()}", self._index_nnodes
                        )
                    )
                    == self._index_node_rank
                )
            ]
            select_audit["assigned_before_resume"] = len(rows)
            select_audit["not_assigned_to_node"] = before - len(rows)
        if str(source.get("release_sample_id") or "content") == "uid" and done:


            before = len(rows)
            rows = [
                row
                for row in rows
                if f"oqm:{bytes(row['uid']).hex()}" not in done
            ]
            resumed_early = before - len(rows)
            stats["resumed"] = int(stats.get("resumed", 0)) + resumed_early
            select_audit["resumed_before_materialize"] = resumed_early
        stats["release"] = {
            "root": str(release.root),
            "lineage": release.lineage,
            "schema_version": release.version.get("schema_version"),
            "selection": select_audit,
        }

        gate = _duration_gate(
            source,
            str(source.get("granularity") or GRANULARITY_WHOLE),
            (
                float(self.config.get("min_duration_sec", 20.0)),
                float(self.config.get("max_duration_sec", 600.0)),
            ),
        )
        preflight = duration_gate_preflight(
            rows, gate, selection.release_granularity, release.upstream_duration
        )
        stats["duration_gate_preflight"] = preflight
        assert_preflight_acknowledged(
            dataset, preflight, str(source.get("duration_gate_ack") or "")
        )

        materialize_workers = int(source.get("release_materialize_workers") or 1)
        if materialize_workers < 1:
            raise ValueError(
                f"data source {dataset} of release_materialize_workers required >= 1"
            )
        yield from materialize_rows(
            release,
            rows,
            cache_root / dataset,
            stats,
            workers=materialize_workers,
        )

    def _iter_manifest_rows(
        self, source: dict[str, Any], cache_root: Path, stats: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:

        dataset = str(source.get("dataset", "unknown"))
        manifest = Path(str(source["manifest"]))
        if not manifest.exists() or manifest.stat().st_size == 0:
            raise FileNotFoundError(
                f"data source {dataset} of manifest is empty or does not exist:{manifest}"
            )
        dataset_dir = Path(str(source.get("dataset_dir") or manifest.parent))
        exclude, exclude_key = _load_exclusions(source)
        count = source.get("sample_count")
        seed = str(source.get("sample_seed") or "corpus-v1")

        if count is None:
            from ..corpus import iter_manifest

            rows: Iterable[dict[str, Any]] = iter_manifest(manifest)
        else:
            rows = sample_manifest(
                manifest,
                int(count),
                seed=seed,
                exclude=exclude or None,
                exclude_key=exclude_key,
            )
            stats["sampled"] = len(rows)
            stats["sample_seed"] = seed
            stats["sample_requested"] = int(count)

        declared = _source_language(source)
        cache_dir = cache_root / dataset

        for row in rows:
            audio = row.get("audio") or {}
            manifest_id = str(row.get("sample_id") or "")
            try:
                ref = parse_audio_ref(row["audio_path"], dataset_dir, audio)
                path, identity, _ = materialize(ref, manifest_id, cache_dir)
            except (ArchiveError, OSError, KeyError, ValueError) as exc:
                stats["failed"] += 1
                stats.setdefault("archive_errors", []).append(
                    f"{manifest_id}: {type(exc).__name__}: {exc}"
                )
                continue

            language, language_source = _record_language(row, declared)
            yield {
                "path": path,
                "manifest_duration_sec": audio.get("duration_sec"),
                "crop_sec": (row.get("training") or {}).get("max_duration_sec"),
                "language_hint": language,
                "extra": {
                    "manifest_sample_id": manifest_id,


                    "manifest_duration_sec": audio.get("duration_sec"),
                    "encapsulation": ref.kind,
                    "archive_identity": identity,
                    "source_uri": (row.get("source") or {}).get("uri", ""),
                    "language_source": language_source,
                },
            }


def _record_language(row: dict[str, Any], declared: Any) -> tuple[Any, str]:

    if declared == LANGUAGE_FROM_MANIFEST:
        value = (row.get("source") or {}).get("language_decision")
        return value, "manifest"
    if declared is None:
        return None, "none"
    return declared, "config"
