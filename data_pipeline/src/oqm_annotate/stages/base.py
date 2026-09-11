
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..config import config_hash, stage_config
from ..store import ShardWriter, StageStore, write_jsonl
from ..text import edit_distance


_LAUNCH_BANNER = re.compile(r"^=== \d{4}-\d\d-\d\d \d\d:\d\d:\d\d ===$")

_SHARD_RANK_CACHE: dict[str, Any] = {}


def _shard_rank_fn(repo_root: Path) -> Any:

    import importlib.util

    path = repo_root / "workers" / "_worker.py"
    key = str(path)
    if key not in _SHARD_RANK_CACHE:
        spec = (
            importlib.util.spec_from_file_location("_oqm_worker_contract", path)
            if path.exists()
            else None
        )
        if spec is None or spec.loader is None:
            raise FileNotFoundError(
                f"Unable to load sharding rules from {path}. Coverage accounting and "
                "worker assignment must use the same sharding function."
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SHARD_RANK_CACHE[key] = module.shard_rank
    return _SHARD_RANK_CACHE[key]


@dataclass(slots=True)
class StageContext:

    config: dict[str, Any]
    work_dir: Path
    repo_root: Path


    #


    touched: set[str] = field(default_factory=set)

    def store(self, stage: str) -> StageStore:
        self.touched.add(stage)
        return StageStore(self.work_dir, stage)


def _read_ledger(context: Any, *, reset: bool = False) -> set[str] | None:

    ledger = getattr(context, "touched", None)
    if not isinstance(ledger, set):
        return None
    if reset:
        ledger.clear()
    return ledger


@dataclass(slots=True)
class StageReport:
    stage: str
    produced: int = 0
    failed: int = 0
    skipped: int = 0
    seconds: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "produced": self.produced,
            "failed": self.failed,
            "skipped": self.skipped,
            "seconds": round(self.seconds, 2),
            **({"notes": self.notes} if self.notes else {}),
        }


class Stage:
    name: str = ""
    depends_on: tuple[str, ...] = ()

    model_keys: tuple[str, ...] = ()

    def __init__(self, context: StageContext):
        self.context = context
        self.config = stage_config(context.config, self.name)


        _read_ledger(context, reset=True)

    @property
    def store(self) -> StageStore:
        return self.context.store(self.name)

    def upstream(self, stage: str) -> dict[str, dict[str, Any]]:

        return self.context.store(stage).load_by_id()

    def run(self) -> StageReport:  # pragma: no cover -
        raise NotImplementedError

    def verify(self) -> dict[str, Any]:

        return self._verdict(*self._scan_product())

    def _scan_product(self) -> tuple[int, int, int, dict[str, int]]:

        rows = 0
        seen: dict[str, int] = {}
        ok = bad = 0
        for record in self.store.iter_records():
            rows += 1
            if record.get("error"):
                bad += 1
            else:
                ok += 1
            sample_id = record.get("sample_id")
            if sample_id:
                seen[str(sample_id)] = seen.get(str(sample_id), 0) + 1
        return rows, ok, bad, seen

    def _verdict(
        self, rows: int, ok: int, bad: int, seen: dict[str, int]
    ) -> dict[str, Any]:
        duplicated = {key: count for key, count in seen.items() if count > 1}
        problems: list[str] = []

        stale: dict[str, dict[str, int]] = {}
        for name, size_then in (self.store.read_done().get("upstream_bytes") or {}).items():
            size_now = self.context.store(name).product_size()
            if size_now > int(size_then):
                stale[name] = {"then": int(size_then), "now": size_now}
        if stale:
            worst = ", ".join(
                f"{name} (was {info['then']} bytes, now {info['now']})"
                for name, info in sorted(stale.items())
            )
            problems.append(
                f"Upstream output changed after this stage completed: {worst}. "
                f"Rerun {self.name} with `--force`; completed sample IDs are otherwise skipped."
            )

        if duplicated:
            worst = max(duplicated.items(), key=lambda kv: kv[1])
            problems.append(
                f"{len(duplicated)} duplicate sample IDs; {worst[0]} appears "
                f"{worst[1]} times. Check for concurrent drivers sharing one work directory."
            )
        return {
            "stage": self.name,
            "rows": rows,
            "unique": len(seen),
            "ok": ok,
            "failed": bad,
            "duplicated": len(duplicated),
            **({"stale_upstreams": sorted(stale)} if stale else {}),
            "problems": problems,
        }

    def _consumed_upstreams(self) -> dict[str, int]:

        ledger = _read_ledger(self.context)
        if ledger is None:
            return {}
        names = sorted(ledger - {self.name})
        return {name: self.context.store(name).product_size() for name in names}

    def _finish(self, report: StageReport) -> StageReport:
        consumed = self._consumed_upstreams()
        self.store.mark_done(
            {
                "stage": self.name,
                "config_hash": config_hash(self.context.config, self.name),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                **({"upstream_bytes": consumed} if consumed else {}),
                **report.to_dict(),
            }
        )
        return report


class LocalStage(Stage):

    def iter_inputs(self) -> Iterable[dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def process(self, item: dict[str, Any]) -> dict[str, Any] | None:  # pragma: no cover

        raise NotImplementedError

    def verify(self) -> dict[str, Any]:
        verdict = super().verify()
        if not bool(self.config.get("verify_full_coverage", False)):
            return verdict
        expected = {
            str(item.get("sample_id"))
            for item in self.iter_inputs()
            if item.get("sample_id")
        }
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
                f"{len(missing)} inputs have no success or failure record in any shard"
                f" (examples: {sorted(missing)[:3]}). This indicates an incomplete shard "
                "and cannot be relaxed with min_coverage."
            )
        if extra:
            verdict["problems"].append(
                f"{len(extra)} outputs are outside the current input set"
                f" (examples: {sorted(extra)[:3]}); another batch may have been mixed in"
            )
        return verdict

    def run(self) -> StageReport:
        started = time.time()
        report = StageReport(stage=self.name)
        nnodes = int(os.environ.get("NNODES", "1"))
        node_rank = int(os.environ.get("NODE_RANK", "0"))
        distributed = bool(self.config.get("distributed_local", False)) and nnodes > 1
        if node_rank < 0 or node_rank >= nnodes:
            raise ValueError(
                f"{self.name}: NODE_RANK={node_rank} out of bounds,NNODES={nnodes}"
            )

        if distributed:
            mine = (node_rank,)
            foreign = [
                path
                for path in self.store.shard_paths()
                if path not in set(self.store.shard_paths(ranks=mine))
            ]
            snapshot = self.store.read_resume_snapshot()
            if snapshot is None and foreign:
                raise RuntimeError(
                    f"{self.name}:distributed LocalStage see {len(foreign)} foreign node shards,"
                    f"but no {self.store.resume_snapshot_path.name}."
                    "Have the external scheduler capture a resume snapshot while every node is idle."
                )
            done = (snapshot or set()) | self.store.completed_ids(ranks=mine)
        else:
            done = self.store.completed_ids()


        self.store.clear_done()
        assigned = 0


        shard_rank = _shard_rank_fn(self.context.repo_root) if distributed else None
        writer_rank = node_rank if distributed else 0
        with ShardWriter(self.store, rank=writer_rank) as writer:
            for item in self.iter_inputs():
                sample_id = str(item.get("sample_id") or "")
                if not sample_id:
                    continue
                if (
                    distributed
                    and shard_rank is not None
                    and shard_rank(sample_id, nnodes) != node_rank
                ):
                    continue
                assigned += 1
                if sample_id in done:
                    report.skipped += 1
                    continue
                try:
                    result = self.process(item)
                except Exception as exc:  # noqa: BLE001

                    writer.write_error(sample_id, exc)
                    report.failed += 1
                    continue
                if result is None:
                    continue
                result.setdefault("sample_id", sample_id)
                writer.write(result)
                report.produced += 1
        report.seconds = time.time() - started
        report.notes.update(
            {
                "distributed_local": distributed,
                "node_rank": node_rank,
                "nnodes": nnodes,
                "assigned": assigned,
            }
        )


        timing = self.store.dir / f"_TIMING.node{node_rank:04d}.json"
        timing_tmp = timing.with_suffix(".json.tmp")
        timing_tmp.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False), encoding="utf-8"
        )
        os.replace(timing_tmp, timing)
        return self._finish(report)


class WorkerStage(Stage):

    script: str = ""

    def __init__(self, context: StageContext):
        super().__init__(context)
        _assert_no_near_miss_keys(self.name, self.config)

    def select_inputs(self) -> list[dict[str, Any]]:  # pragma: no cover

        raise NotImplementedError

    def extra_args(self) -> list[str]:
        args: list[str] = []
        for key, value in (self.config.get("extra_args") or {}).items():
            if value is None or value is False:
                continue
            args.append(f"--{key}")
            if value is not True:
                args.append(str(value))
        return args

    def verify(self) -> dict[str, Any]:

        rows, ok, bad, seen = self._scan_product()
        result = self._verdict(rows, ok, bad, seen)

        items = self.select_inputs()
        expected = len(items)
        got = ok + bad
        coverage = got / expected if expected else 1.0
        minimum = float(self.config.get("min_coverage", 0.95))
        result["expected"] = expected
        result["coverage"] = round(coverage, 4)
        result["min_coverage"] = minimum


        unprocessed = sorted({str(item["sample_id"]) for item in items} - set(seen))
        result["unprocessed"] = len(unprocessed)
        if unprocessed:
            result["problems"].append(
                f"{len(unprocessed)} samples have no record in any shard. This is independent "
                f"of min_coverage={minimum}. Check for missing rank shards with "
                f"`ls {self.store.dir}/rank_*.jsonl | wc -l`, then inspect node logs. "
                f"Examples: {unprocessed[:3]}"
            )

        if coverage < minimum:
            result["problems"].append(
                f"coverage {coverage:.3f} is below threshold {minimum}: "
                f"expected {expected} records, got {got} records"
            )
        return result

    def _distribution(self) -> tuple[int, int, int]:

        nproc = max(1, int(self.config.get("nproc_per_node", 1) or 1))
        nnodes = max(1, int(self.config.get("nnodes", os.environ.get("NNODES", 1)) or 1))
        node_rank = int(self.config.get("node_rank", os.environ.get("NODE_RANK", 0)) or 0)
        if not 0 <= node_rank < nnodes:


            raise ValueError(
                f"{self.name} of node_rank={node_rank} is not in 0..{nnodes - 1} within"
                f"(nnodes={nnodes}). rank will fall to WORLD_SIZE "
                "; otherwise its assigned input is empty and the stage can report a false success."
            )
        return nproc, nnodes, node_rank

    def _node_ranks(self) -> tuple[int, ...]:
        nproc, _, node_rank = self._distribution()
        return tuple(range(node_rank * nproc, node_rank * nproc + nproc))

    def _completed_ids(self, nnodes: int, my_ranks: tuple[int, ...]) -> set[str]:

        if nnodes == 1:
            return self.store.completed_ids()

        mine = set(self.store.shard_paths(ranks=my_ranks))
        foreign = [path for path in self.store.shard_paths() if path not in mine]
        snapshot = self.store.read_resume_snapshot()

        if snapshot is None:
            if not foreign:


                return self.store.completed_ids(ranks=my_ranks)
            raise RuntimeError(
                f"{self.name}: {self.store.dir} contains {len(foreign)} shards from other "
                f"nodes, but {self.store.resume_snapshot_path.name} is missing. "
                "Create the snapshot while all nodes are idle:\n"
                f"  oqm-annotate resume-snapshot --config <configuration> --stages {self.name}"
            )
        return snapshot | self.store.completed_ids(ranks=my_ranks)

    def run(self) -> StageReport:

        started = time.time()
        report = StageReport(stage=self.name)

        nproc, nnodes, node_rank = self._distribution()
        world_size = nproc * nnodes
        my_ranks = self._node_ranks()

        pending = self.select_inputs()
        done = self._completed_ids(nnodes, my_ranks)
        queued = [item for item in pending if str(item["sample_id"]) not in done]
        report.skipped = len(pending) - len(queued)

        if not queued:
            report.seconds = time.time() - started
            report.notes["reason"] = "no pending items"
            return self._finish(report)


        already = sum(self.store.counts(ranks=my_ranks)) if nnodes > 1 else 0


        suffix = "" if nnodes == 1 else f".node{node_rank:04d}"
        manifest = self.store.dir / f"input{suffix}.jsonl"
        write_jsonl(manifest, queued)

        log_path = self.store.dir / f"launch{suffix}.log"


        self.store.clear_done()

        failures = self._launch(manifest, log_path, len(queued))
        if failures:
            tail = _tail(log_path, 40)
            raise RuntimeError(
                f"{self.name} had {len(failures)} ranks exit nonzero: "
                f"{failures}; log tail:\n{tail}"
            )

        if nnodes == 1:
            ok, bad = self.store.counts()
            expected = len(pending)
            scope = ""
            whose = "expected"
        else:


            ok, bad = self.store.counts(ranks=my_ranks)
            shard_rank = _shard_rank_fn(self.context.repo_root)
            mine = set(my_ranks)
            expected = already + sum(
                1
                for item in queued
                if shard_rank(item["sample_id"], world_size) in mine
            )
            scope = f" node {node_rank}/{nnodes} (ranks {my_ranks[0]}..{my_ranks[-1]})"
            whose = "expected on this node"
            report.notes["coverage_scope"] = "node"
            report.notes["node_rank"] = node_rank
            report.notes["nnodes"] = nnodes

        report.produced = ok
        report.failed = bad
        coverage = (ok + bad) / expected if expected else 1.0
        report.notes["coverage"] = round(coverage, 4)
        report.notes.update(_timing_notes(log_path))

        minimum = float(self.config.get("min_coverage", 0.95))
        if coverage < minimum:
            raise RuntimeError(
                f"{self.name}{scope} coverage {coverage:.3f} is below {minimum}: "
                f"{whose} {expected}, produced {ok + bad}. Log: {log_path}"
            )

        report.seconds = time.time() - started
        return self._finish(report)

    def _launch(self, manifest: Path, log_path: Path, queued: int) -> dict[int, int]:

        python = str(self.config.get("python") or sys.executable)
        if not Path(python).exists():
            raise FileNotFoundError(f"{self.name} configured interpreter does not exist: {python}")

        script_path = self.context.repo_root / "workers" / self.script
        if not script_path.exists():
            raise FileNotFoundError(f"Worker script does not exist: {script_path}")

        nproc, nnodes, node_rank = self._distribution()
        world_size = nproc * nnodes

        base_command = [
            python,
            str(script_path),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(self.store.dir),
            *self.extra_args(),
        ]
        base_env = self._build_env()

        print(
            f"[{self.name}] starting {queued} items across {nproc} processes × {nnodes} nodes:"
            f"{' '.join(base_command)}",
            file=sys.stderr,
        )

        processes: dict[int, subprocess.Popen[bytes]] = {}
        with log_path.open("a", encoding="utf-8") as log:

            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log.write(" ".join(base_command) + "\n")
            log.flush()

            for local_rank in range(nproc):
                rank = node_rank * nproc + local_rank
                env = dict(base_env)
                env["RANK"] = str(rank)
                env["WORLD_SIZE"] = str(world_size)


                env["CUDA_VISIBLE_DEVICES"] = str(local_rank)
                env["LOCAL_RANK"] = "0"
                processes[rank] = subprocess.Popen(
                    base_command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(self.context.repo_root),
                )

            failures: dict[int, int] = {}
            for rank, process in processes.items():
                code = process.wait()
                if code != 0:
                    failures[rank] = code
        return failures

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")


        src = str(self.context.repo_root / "src")
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{src}:{existing}" if existing else src
        for key, value in (self.config.get("env") or {}).items():
            env[str(key)] = str(value)
        return env


_WORKER_STAGE_KEYS = frozenset(
    {
        "python",
        "nproc_per_node",
        "nnodes",
        "node_rank",
        "env",
        "extra_args",
        "min_coverage",
    }
)


def _assert_no_near_miss_keys(stage: str, config: dict[str, Any]) -> None:

    for key in config:
        if key in _WORKER_STAGE_KEYS:
            continue
        for known in _WORKER_STAGE_KEYS:
            if abs(len(key) - len(known)) > 2:
                continue
            if edit_distance(list(str(key)), list(known)) <= 2:
                raise ValueError(
                    f"{stage} contains unknown key `{key}`; did you mean `{known}`? "
                    f"To add `{key}`, declare it in stages/base.py::_WORKER_STAGE_KEYS."
                )


def _tail(path: Path, lines: int) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(log is unreadable)"
    return "\n".join(content[-lines:])


def _timing_notes(log_path: Path) -> dict[str, Any]:

    try:
        content = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return {}

    for position in range(len(content) - 1, -1, -1):
        if _LAUNCH_BANNER.match(content[position]):
            content = content[position + 1 :]
            break

    summaries = []
    decoder = json.JSONDecoder()
    for line in content:
        _, marker, payload = line.partition("WORKER-SUMMARY ")
        if not marker:
            continue


        try:
            obj, _ = decoder.raw_decode(payload.strip())
        except json.JSONDecodeError:
            continue
        summaries.append(obj)
    if not summaries:
        return {}

    loads = [s.get("load_seconds", 0.0) for s in summaries]
    computes = [s.get("compute_seconds", 0.0) for s in summaries]
    notes: dict[str, Any] = {
        "ranks": len(summaries),
        "load_sec_max": round(max(loads), 1),
        "compute_sec_max": round(max(computes), 1),
    }
    total_compute = sum(computes)


    processed = sum(s.get("processed", 0) for s in summaries)
    if processed and total_compute > 0:

        notes["gpu_sec_per_record"] = round(total_compute / processed, 2)
        notes["timing_records"] = processed
    if loads and computes and max(computes) > 0:
        notes["load_share"] = round(max(loads) / (max(loads) + max(computes)), 3)
    slowest = max((s.get("slowest_record_sec", 0.0) for s in summaries), default=0.0)
    if slowest:
        notes["slowest_record_sec"] = slowest
    return notes


def dump_report(reports: Sequence[StageReport]) -> str:
    return json.dumps([r.to_dict() for r in reports], ensure_ascii=False, indent=2)
