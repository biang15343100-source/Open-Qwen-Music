
from __future__ import annotations

import argparse
import json
import time
from typing import Any

from .config import PipelineConfig
from .runtime import log as logmod
from .runtime.context import Context
from .runtime.lock import WorkDirLock
from .acoustic_views import build_acoustic_views
from .renderer_semantic_adapter import adapt_renderer_semantics
from .tokenizer_views import build_tokenizer_views

STAGE_NAMES = ["discover", "probe", "enrich", "filter", "dedup", "normalize", "publish"]


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="pipeline configuration YAML")
    parser.add_argument("--datasets", default=None, help="Comma-separated slugs or dataset IDs; default: all")
    parser.add_argument("--force", action="store_true", help="Ignore completion markers and rerun")
    parser.add_argument("--log-level", default="INFO")


def _datasets(value: str | None) -> list[str] | None:
    if not value or value == "all":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _shard(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    index, _, total = text.partition("/")
    try:
        i, n = int(index), int(total)
    except ValueError:
        raise SystemExit(f"--shard must use i/n format; got {text!r}") from None
    if n < 1 or not 0 <= i < n:
        raise SystemExit(f"--shard is out of bounds: {text!r}; expected 0 <= i < n")
    return (i, n)


def _run_stage(name: str, ctx: Context, args: argparse.Namespace) -> Any:
    from .stages import dedup_stage, discover, enrich, filter_stage, normalize, probe, publish

    datasets = _datasets(getattr(args, "datasets", None))
    force = bool(getattr(args, "force", False))
    if name == "discover":
        return discover.run(
            ctx, datasets, force=force, strict=not getattr(args, "no_strict", False)
        )
    if name == "probe":
        task_bytes = getattr(args, "task_bytes", None)
        if task_bytes is not None:
            ctx.cfg.runtime.probe_task_bytes = task_bytes
        return probe.run(
            ctx,
            datasets,
            force=force,
            workers=getattr(args, "workers", None),
            shard=_shard(getattr(args, "shard", None)),
        )
    if name == "enrich":
        return enrich.run(ctx, datasets, force=force, workers=getattr(args, "workers", None))
    if name == "filter":
        return filter_stage.run(ctx, datasets, force=force)
    if name == "dedup":
        return dedup_stage.run(ctx, force=force)
    if name == "normalize":
        return normalize.run(ctx, force=force)
    if name == "publish":
        return publish.run(ctx, force=force)
    raise ValueError(f"Unknown stage {name}")


def cmd_run(args: argparse.Namespace) -> int:
    cfg = PipelineConfig.load(args.config)
    logmod.setup(
        args.log_level, cfg.work_dir / "logs" / f"run-{time.strftime('%Y%m%d-%H%M%S')}.log"
    )
    log = logmod.get("cli")

    start = STAGE_NAMES.index(args.from_stage)
    end = STAGE_NAMES.index(args.to_stage)
    if start > end:
        log.error("--from stage %s occurs after --to stage %s", args.from_stage, args.to_stage)
        return 2


    if args.datasets and end >= STAGE_NAMES.index("dedup"):
        log.error(
            "--datasets can be used only through the filter stage; dedup requires all datasets"
        )
        return 2

    with WorkDirLock(cfg.work_dir, purpose=f"run {args.from_stage}..{args.to_stage}"):
        ctx = Context.create(cfg)
        summary: dict[str, Any] = {}
        for name in STAGE_NAMES[start : end + 1]:
            began = time.time()
            log.info("=== stage %s started ===", name)
            summary[name] = _run_stage(name, ctx, args)
            log.info("=== stage %s complete in %.1fs ===", name, time.time() - began)

        out = cfg.work_dir / "run_summary.json"


        merged: dict[str, Any] = {}
        if out.exists():
            try:
                merged = json.loads(out.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                merged = {}
        merged.update(summary)
        merged = {name: merged[name] for name in STAGE_NAMES if name in merged}
        out.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        log.info("Wrote summary to %s", out)
    return 0


def cmd_stage(args: argparse.Namespace) -> int:
    cfg = PipelineConfig.load(args.config)
    logmod.setup(args.log_level, cfg.work_dir / "logs" / f"{args.command}.log")
    with WorkDirLock(cfg.work_dir, purpose=args.command):
        ctx = Context.create(cfg)
        result = _run_stage(args.command, ctx, args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str)[:4000])
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = PipelineConfig.load(args.config)
    logmod.setup(args.log_level)
    print(f"Configure fingerprint {cfg.fingerprint()}   work_dir {cfg.work_dir}")
    stage_dirs = {
        "s1_discover": "discover",
        "s2_probe": "probe",
        "s3_enrich": "enrich",
        "s4_filter": "filter",
        "s5_dedup": "dedup",
        "s6_normalize": "normalize",
    }
    for stage, label in stage_dirs.items():
        directory = cfg.stage_dir(stage)
        success = directory / "_SUCCESS.json"
        if not directory.exists():
            print(f"  {label:10s} Not started")
            continue
        shards = len(list(directory.glob("part-*.parquet")))
        if success.exists():
            payload = json.loads(success.read_text(encoding="utf-8"))

            stale = payload.get("config_fingerprint") != cfg.stage_fingerprint(stage)
            mark = "configuration changed; rerun required" if stale else "complete"
            print(f"  {label:10s} {mark}  Sharding {shards}  line {payload.get('rows', '?')}")
        else:
            print(f"  {label:10s} In progress  Sharding {shards}")

    release = cfg.release_dir / f"oqm-corpus-{cfg.version}"
    ready = release / "READY"
    print(f"  release    {'Ready ' + str(release) if ready.exists() else 'Unpublished'}")
    return 0


def cmd_validate_registry(args: argparse.Namespace) -> int:
    cfg = PipelineConfig.load(args.config)
    logmod.setup(args.log_level)
    log = logmod.get("cli")
    ctx = Context.create(cfg)

    problems: list[str] = []
    pending: list[str] = []
    total_expected = 0
    for spec in ctx.registry:
        if not spec.enabled:
            log.info("%-28s disabled: %s", spec.slug, spec.disabled_reason or "No reason provided")
            continue
        if spec.metadata_only:
            log.info("%-28s Metadata only", spec.slug)
            continue
        total_expected += spec.expected_item_count or 0
        for source in spec.sources:
            containers = source.containers()
            if not containers:
                problems.append(f"{spec.slug}: source({source.storage_class}) did not resolve to any containers")
                continue
            missing = [c for c in containers if not c.exists()]
            if missing:
                problems.append(
                    f"{spec.slug}: {len(missing)}/{len(containers)} containers are missing; for example {missing[0]}"
                )
        if spec.expected_item_count is None:

            if spec.expected_note:
                pending.append(spec.slug)
            else:
                problems.append(
                    f"{spec.slug}: expected_item_count is missing without an explanation, so enumeration cannot be validated"
                )
        for enrich_spec in spec.enrich:
            if enrich_spec.format.startswith("inline_"):
                continue
            if not enrich_spec.sidecar_paths():
                problems.append(f"{spec.slug}: enrich({enrich_spec.format}) sidecar file was not found")

    disabled = ctx.registry.disabled
    print(
        f"data set {len(ctx.registry)} ,which enables {len(ctx.registry) - len(disabled)} ,"
        f"Close {len(disabled)} ,Enable the total number of partial declaration entries {total_expected:,}"
    )
    if pending:
        print(
            f"Number of entries to be backfilled(discover followed by supporting offline verification tool Finalized)"
            f"{len(pending)} : {', '.join(pending)}"
        )
    for problem in problems:
        print(f"  [Question] {problem}")
    print(f"total {len(problems)} questions")
    return 1 if problems else 0


def cmd_check_enrich(args: argparse.Namespace) -> int:
    from .stages import check_enrich

    cfg = PipelineConfig.load(args.config)
    logmod.setup(args.log_level)
    ctx = Context.create(cfg)
    result = check_enrich.run(ctx, _datasets(getattr(args, "datasets", None)))

    for report in result["reports"]:
        rate = report.get("join_rate")
        status = report.get("status")
        head = f"{report['slug']:28s} {report.get('spec', ''):16s}"
        if status and report.get("pending"):
            print(f"  [PENDING] {head} {status} (skipped {report.get('sidecar_records')} items)")
        elif status:
            print(f"  [Question] {head} {status}")
        elif rate is not None and rate < 0.5:
            print(f"  [CHECK] {head} hook rate {rate:.1%} (skipped {report.get('sidecar_records')} items)")
        else:
            print(f"  {head} hook rate {rate:.1%}" if rate is not None else f"  {head} Normal")
        for key in ("hint", "miss_examples", "index_key_examples", "available_columns"):
            if report.get(key):
                print(f"        {key}: {report[key]}")

    print(
        f"Check {result['checked']} item,Suspicious {result['suspicious']} item,"
        f"pending discover Re-examination after {result.get('pending', 0)} item"
    )
    return 1 if result["suspicious"] else 0


def cmd_inspect(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    from . import flags, locator
    from .enums import CONTENT_TYPE, LANGUAGE, STATUS, STORAGE_CLASS

    cfg = PipelineConfig.load(args.config)
    logmod.setup(args.log_level)
    ctx = Context.create(cfg)
    stage = args.stage
    store = ctx.store(
        stage, __import__("oqm_preprocess.schema", fromlist=["WORK_SCHEMA"]).WORK_SCHEMA
    )
    paths = store.shard_paths()
    if not paths:
        print(f"{stage} No product")
        return 1

    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())
        if args.uid is None and len(rows) >= args.sample * 4:
            break

    if args.uid:
        target = bytes.fromhex(args.uid)
        rows = [r for r in rows if bytes(r["uid"]) == target]
        if not rows:
            print(f"not found uid {args.uid}")
            return 1
    else:
        step = max(1, len(rows) // max(1, args.sample))
        rows = rows[::step][: args.sample]

    for record in rows:
        print("-" * 88)
        print(f"uid          {bytes(record['uid']).hex()}")
        print(f"data set       {ctx.registry.get(int(record['dataset_id'])).slug}")
        print(f"local_id     {record['local_id']}")
        try:
            print(f"uri          {locator.to_uri(record, ctx.containers)}")
        except (KeyError, ValueError) as exc:
            print(f"uri          <cannot be parsed: {exc}>")
        print(
            f"Audio         {record['duration_sec']}s  {record['sample_rate_hz']}Hz  "
            f"{record['channels']}ch  Storage={STORAGE_CLASS.name_of(int(record['storage_class']))}"
        )
        print(
            f"Content         {CONTENT_TYPE.name_of(int(record['content_type'] or 0))}  "
            f"Language={LANGUAGE.name_of(int(record['language'] or 0))}"
        )
        print(
            f"status         {STATUS.name_of(int(record['status']))}  "
            f"flags={flags.decode(int(record['flags']))}"
        )
        if record.get("lyrics_text"):
            preview = str(record["lyrics_text"])[:80].replace("\n", " / ")
            print(f"Lyrics         {preview}")
    return 0



def cmd_release_path(args: argparse.Namespace) -> int:
    cfg = PipelineConfig.load(args.config)
    print((cfg.release_dir / f"oqm-corpus-{cfg.version}").resolve())
    return 0


def cmd_build_tokenizer_views(args: argparse.Namespace) -> int:
    report = build_tokenizer_views(
        args.release,
        args.output_dir,
        annotations=args.annotations,
        max_full_track_duration_sec=args.max_full_track_duration_sec,
        force=bool(args.force),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_build_acoustic_views(args: argparse.Namespace) -> int:
    report = build_acoustic_views(
        args.tokenizer_manifest,
        args.output_dir,
        force=bool(args.force),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_adapt_renderer_semantics(args: argparse.Namespace) -> int:
    report = adapt_renderer_semantics(
        args.input_dir,
        args.output_dir,
        source_manifest=args.source_manifest,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oqm-preprocess", description="Unified data preprocessing pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Execute all or part of the phases in sequence")
    _add_common(run_parser)
    run_parser.add_argument("--from", dest="from_stage", default="discover", choices=STAGE_NAMES)
    run_parser.add_argument("--to", dest="to_stage", default="publish", choices=STAGE_NAMES)
    run_parser.add_argument("--workers", type=int, default=None)
    run_parser.add_argument("--no-strict", action="store_true")
    run_parser.set_defaults(func=cmd_run)

    for name in STAGE_NAMES:
        stage_parser = sub.add_parser(name, help=f"only runs {name} stage")
        _add_common(stage_parser)
        if name in {"probe", "enrich"}:
            stage_parser.add_argument("--workers", type=int, default=None)
        if name == "probe":
            stage_parser.add_argument("--task-bytes", type=int, dest="task_bytes")
            stage_parser.add_argument("--shard", metavar="i/n")
        if name == "discover":
            stage_parser.add_argument("--no-strict", action="store_true")
        stage_parser.set_defaults(func=cmd_stage)

    status = sub.add_parser("status", help="View stage progress")
    status.add_argument("--config", required=True)
    status.add_argument("--log-level", default="WARNING")
    status.set_defaults(func=cmd_status)

    validate = sub.add_parser("validate-registry", help="Check the data source before running")
    validate.add_argument("--config", required=True)
    validate.add_argument("--log-level", default="INFO")
    validate.set_defaults(func=cmd_validate_registry)

    enrich = sub.add_parser("check-enrich", help="Check metadata hook rate")
    enrich.add_argument("--config", required=True)
    enrich.add_argument("--datasets")
    enrich.add_argument("--log-level", default="WARNING")
    enrich.set_defaults(func=cmd_check_enrich)

    inspect = sub.add_parser("inspect", help="Sampling view record")
    inspect.add_argument("--config", required=True)
    inspect.add_argument("--stage", default="s6_normalize")
    inspect.add_argument("--uid")
    inspect.add_argument("--sample", type=int, default=10)
    inspect.add_argument("--log-level", default="WARNING")
    inspect.set_defaults(func=cmd_inspect)

    release_path = sub.add_parser(
        "release-path",
        help="Print the immutable release directory produced by a pipeline config",
    )
    release_path.add_argument("--config", required=True)
    release_path.set_defaults(func=cmd_release_path)

    views = sub.add_parser(
        "build-tokenizer-views",
        help="Build the public Tokenizer training manifests from a READY corpus release",
    )
    views.add_argument("--release", required=True)
    views.add_argument("--output-dir", required=True)
    views.add_argument(
        "--annotations",
        help="Optional oqm-annotate annotation.jsonl; joins release samples by stable UID",
    )
    views.add_argument("--max-full-track-duration-sec", type=float, default=300.0)
    views.add_argument("--force", action="store_true")
    views.set_defaults(func=cmd_build_tokenizer_views)

    acoustic_views = sub.add_parser(
        "build-acoustic-views",
        help="Build Acoustic VAE and bandwidth-refiner manifests from a Tokenizer view",
    )
    acoustic_views.add_argument("--tokenizer-manifest", required=True)
    acoustic_views.add_argument("--output-dir", required=True)
    acoustic_views.add_argument("--force", action="store_true")
    acoustic_views.set_defaults(func=cmd_build_acoustic_views)

    renderer_semantics = sub.add_parser(
        "adapt-renderer-semantics",
        help="Validate and adapt generic semantic_tokens.jsonl data for Renderer training",
    )
    renderer_semantics.add_argument("--input-dir", required=True)
    renderer_semantics.add_argument("--output-dir", required=True)
    renderer_semantics.add_argument(
        "--source-manifest",
        help="Source audio manifest required when --input-dir is an oqm.llm.corpus.v1 directory",
    )
    renderer_semantics.set_defaults(func=cmd_adapt_renderer_semantics)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
