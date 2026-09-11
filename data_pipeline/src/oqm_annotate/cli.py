
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import load_config, require
from .registry import (
    CPU_STAGES,
    STAGE_CLASSES,
    build,
    expand_with_dependencies,
    topological_order,
)
from .stages.base import StageContext, WorkerStage


def _context(args: argparse.Namespace) -> tuple[dict[str, Any], StageContext]:
    config = load_config(args.config)
    work_dir = Path(str(require(config, "work_dir")))
    work_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent
    return config, StageContext(config=config, work_dir=work_dir, repo_root=repo_root)


def cmd_stages(_: argparse.Namespace) -> int:
    rows = []
    for cls in STAGE_CLASSES:
        rows.append(
            {
                "stage": cls.name,
                "kind": "cpu" if cls.name in CPU_STAGES else "gpu",
                "depends_on": list(cls.depends_on),
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def _selected_stages(args: argparse.Namespace) -> list[str]:
    if args.with_deps and not args.stages:


        print(
            "Attention:`--with-deps` is only in cooperation with `--stages` ,is ignored this time.",
            file=sys.stderr,
        )
    if args.all:
        names = [cls.name for cls in STAGE_CLASSES]
    elif args.cpu_only:
        names = [cls.name for cls in STAGE_CLASSES if cls.name in CPU_STAGES]
    elif args.stages:
        names = [s.strip() for s in args.stages.split(",") if s.strip()]
        if args.with_deps:
            return expand_with_dependencies(names)
    else:
        names = [cls.name for cls in STAGE_CLASSES if cls.name in CPU_STAGES]
    return topological_order(names)


def cmd_run(args: argparse.Namespace) -> int:
    config, context = _context(args)
    names = _selected_stages(args)

    if args.dry_run:
        print(
            json.dumps(
                {"work_dir": str(context.work_dir), "stages": names},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    reports: list[dict[str, Any]] = []
    for name in names:
        stage = build(name, context)
        if args.force:
            stage.store.reset()
        print(f"=== {name} ===", file=sys.stderr)
        try:
            report = stage.run()
        except Exception as exc:  # noqa: BLE001


            print(json.dumps(reports, ensure_ascii=False, indent=2))
            print(f"stage `{name}` failed:{exc}", file=sys.stderr)
            return 1
        reports.append(report.to_dict())
        print(json.dumps(report.to_dict(), ensure_ascii=False), file=sys.stderr)

    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:

    _, context = _context(args)
    if args.stages:
        names = topological_order(
            [name.strip() for name in args.stages.split(",") if name.strip()]
        )
    else:
        names = [cls.name for cls in STAGE_CLASSES]

    rows: list[dict[str, Any]] = []
    failures = 0
    for name in names:
        stage = build(name, context)
        store = stage.store


        if not store.shard_paths() and not store.is_done():
            rows.append({"stage": name, "verdict": "not_run"})
            continue
        result = stage.verify()
        result["verdict"] = "OK" if not result["problems"] else "failed"
        if result["problems"]:
            failures += 1
        rows.append(result)

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    if failures:
        print(
            f"{failures} stages failed global verification.**Do not continue with this product.**",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_resume_snapshot(args: argparse.Namespace) -> int:

    _, context = _context(args)
    names = topological_order(
        [name.strip() for name in args.stages.split(",") if name.strip()]
    )
    rows = []
    for name in names:
        stage = build(name, context)


        distributed_cpu = bool(
            stage.config.get("distributed_index")
            or stage.config.get("distributed_local")
        )
        if not isinstance(stage, WorkerStage) and not distributed_cpu:
            rows.append({"stage": name, "skipped": "CPU stage runs on one node and does not require a snapshot"})
            continue
        payload = stage.store.write_resume_snapshot()
        rows.append({key: value for key, value in payload.items() if key != "sample_ids"})
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    _, context = _context(args)
    rows = []
    for cls in STAGE_CLASSES:
        store = context.store(cls.name)
        ok, bad = store.counts()
        marker: dict[str, Any] = {}
        if store.done_marker.exists():
            marker = json.loads(store.done_marker.read_text(encoding="utf-8"))
        rows.append(
            {
                "stage": cls.name,
                "ok": ok,
                "failed": bad,
                "finished_at": marker.get("finished_at"),
                "seconds": marker.get("seconds"),
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:

    _, context = _context(args)
    result: dict[str, Any] = {}
    for cls in STAGE_CLASSES:
        record = (
            context.store(cls.name).load_by_id(include_errors=True).get(args.sample_id)
        )
        if record is not None:
            result[cls.name] = record
    if not result:
        print(f"not found sample_id={args.sample_id}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:

    _, context = _context(args)
    from collections import Counter

    sections = context.store("sections").load_by_id()
    lyrics = context.store("lyrics").load_by_id()
    tags = context.store("tags.fuse").load_by_id()

    tier_counts = Counter(r.get("tier", "none") for r in lyrics.values())
    label_counts: Counter[str] = Counter()
    coverage_buckets: Counter[str] = Counter()
    for record in sections.values():
        for section in record.get("sections") or []:
            label_counts[section.get("label", "?")] += 1
        coverage = record.get("lyric_coverage", 0.0)
        bucket = (
            "0.0-0.15"
            if coverage < 0.15
            else "0.15-0.35"
            if coverage < 0.35
            else "0.35-0.60"
            if coverage < 0.60
            else "0.60+"
        )
        coverage_buckets[bucket] += 1

    gender_counts = Counter(
        (r.get("tags") or {}).get("vocal_gender") or "unknown" for r in tags.values()
    )
    conflicts = sum(1 for r in tags.values() if r.get("gender_conflict"))
    unmapped: Counter[str] = Counter()
    for record in tags.values():
        for field, values in (record.get("unmapped_tags") or {}).items():
            for value in values:
                unmapped[f"{field}:{value}"] += 1

    print(
        json.dumps(
            {
                "samples_with_sections": len(sections),
                "lyrics_tier": dict(tier_counts),
                "section_labels": dict(label_counts.most_common()),
                "lyric_coverage_buckets": dict(coverage_buckets),
                "vocal_gender": dict(gender_counts),
                "gender_conflicts": conflicts,


                "top_unmapped_tags": dict(unmapped.most_common(30)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oqm-annotate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("stages").set_defaults(func=cmd_stages)

    def with_config(sub: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sub.add_argument("--config", required=True)
        return sub

    run_parser = with_config(subparsers.add_parser("run"))
    group = run_parser.add_mutually_exclusive_group()
    group.add_argument("--stages", help="comma separated stage first name")
    group.add_argument("--cpu-only", action="store_true")
    group.add_argument("--all", action="store_true")
    run_parser.add_argument("--with-deps", action="store_true", help="automatically completes the upstream")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--force", action="store_true", help="Clear existing products and rerun")
    run_parser.set_defaults(func=cmd_run)

    verify_parser = with_config(subparsers.add_parser("verify"))
    verify_parser.add_argument(
        "--stages", help="comma separated stage first name;If you don\'t give it, check everything.(Report that has not been run\"not running\")"
    )
    verify_parser.set_defaults(func=cmd_verify)

    snapshot_parser = with_config(subparsers.add_parser("resume-snapshot"))
    snapshot_parser.add_argument("--stages", required=True, help="comma separated stage first name")
    snapshot_parser.set_defaults(func=cmd_resume_snapshot)

    with_config(subparsers.add_parser("status")).set_defaults(func=cmd_status)
    with_config(subparsers.add_parser("stats")).set_defaults(func=cmd_stats)

    inspect_parser = with_config(subparsers.add_parser("inspect"))
    inspect_parser.add_argument("--sample-id", required=True)
    inspect_parser.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
