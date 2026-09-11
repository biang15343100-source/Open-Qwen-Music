
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .common.config import apply_overrides, load_config


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML configuration path")
    parser.add_argument(
        "--override",
        action="append",
        nargs="+",
        default=[],
        metavar="a.b=value",
        help="Override a configuration value with a.b=value. May be repeated.",
    )
    parser.add_argument(
        "--allow-new-key",
        action="store_true",
        help="Allow --override to create a new key. Disabled by default to catch typos.",
    )


def _load(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    _apply_overrides(
        config, args.override, allow_new_keys=bool(getattr(args, "allow_new_key", False))
    )
    return config


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the Open Qwen Music language model")
    _add_config_argument(parser)
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument(
        "--init-from", default=None, help="Initialize model weights from a checkpoint"
    )
    checkpoint.add_argument(
        "--resume-from",
        default=None,
        help="Resume model, optimizer, scheduler, sampler, and RNG state",
    )
    args = parser.parse_args(argv)

    from .trainer import train

    train(_load(args), init_from=args.init_from, resume_from=args.resume_from)


def eval_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a language-model checkpoint")
    _add_config_argument(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    from .trainer import evaluate

    results = evaluate(_load(args), args.checkpoint, split=args.split, max_batches=args.max_batches)
    if args.output and int(os.environ.get("RANK", "0")) == 0:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, destination)


def generate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate semantic music tokens")
    _add_config_argument(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--prompts",
        required=True,
        help="JSONL file with one prompt object per line",
    )
    parser.add_argument("--output", required=True, help="Output NPZ file")
    parser.add_argument("--mode", default="plain")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--semantic-repetition-penalty", type=float, default=None)
    parser.add_argument("--semantic-repetition-window", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-strict-mode", action="store_true")
    parser.add_argument(
        "--render-requests",
        default=None,
        help="RenderRequest JSONL output (defaults to <output>.render.jsonl)",
    )
    args = parser.parse_args(argv)

    from .inference import run_generation

    run_generation(
        _load(args),
        checkpoint=args.checkpoint,
        prompts_path=args.prompts,
        output_path=args.output,
        mode=args.mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        semantic_repetition_penalty=args.semantic_repetition_penalty,
        semantic_repetition_window=args.semantic_repetition_window,
        seed=args.seed,
        batch_size=args.batch_size,
        strict_mode=not args.no_strict_mode,
        render_requests_path=args.render_requests,
    )


def registry_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Resolve and validate the token registry")
    _add_config_argument(parser)
    parser.add_argument("--output", default=None, help="Optional registry JSON output path")
    args = parser.parse_args(argv)

    from .trainer import resolve_registry

    registry = resolve_registry(_load(args))
    payload = registry.to_dict()
    payload["revision"] = registry.revision
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        registry.save(args.output)


def melody_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract 6.25 Hz melody tokens from an audio file"
    )
    parser.add_argument("--audio", required=True, help="Input audio file")
    parser.add_argument("--checkpoint", required=True, help="RMVPE PyTorch checkpoint")
    parser.add_argument(
        "--output", required=True, help="Output NPZ file and JSON provenance sidecar"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device, for example cpu or cuda",
    )
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--voicing-threshold", type=float, default=None)
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=None,
        help="Optional 50 Hz chunk size; omitted means full-context extraction",
    )
    parser.add_argument("--context-frames", type=int, default=128)
    parser.add_argument(
        "--verify-checkpoint-sha256",
        default=None,
        help="Optional expected checkpoint SHA-256 digest",
    )
    args = parser.parse_args(argv)

    from .melody_tokenizer import MelodyTokenizer
    from .rmvpe import RMVPEPitchExtractor

    extractor = RMVPEPitchExtractor(
        args.checkpoint,
        device=args.device,
        voicing_threshold=args.voicing_threshold,
        chunk_frames=args.chunk_frames,
        context_frames=args.context_frames,
        verify_checkpoint_hash=args.verify_checkpoint_sha256,
    )
    result = MelodyTokenizer(extractor).encode_file(
        args.audio,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
    )
    output, sidecar = result.save(args.output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sidecar": str(sidecar),
                "pitch_frames_50hz": result.pitch.num_frames,
                "melody_frames_6_25hz": result.num_frames,
                "voiced_melody_frames": int((result.token_ids != result.unvoiced_id).sum()),
                "tokenizer_revision": result.tokenizer_revision,
            },
            ensure_ascii=False,
        )
    )


def _apply_overrides(
    config: dict,
    overrides: list[str] | list[list[str]],
    *,
    allow_new_keys: bool = False,
) -> None:
    applied = apply_overrides(config, overrides, allow_new_keys=allow_new_keys)
    if applied:
        print(f"Applied {len(applied)} override(s): {', '.join(applied)}", flush=True)


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "train": train_main,
        "eval": eval_main,
        "generate": generate_main,
        "melody": melody_main,
        "registry": registry_main,
    }
    if argv and argv[0] in {"-h", "--help"}:
        print(f"Usage: oqm-llm {{{'|'.join(commands)}}} [options]")
        return
    if not argv or argv[0] not in commands:
        print(f"Usage: oqm-llm {{{'|'.join(commands)}}} [options]")
        raise SystemExit(2)
    commands[argv[0]](argv[1:])


if __name__ == "__main__":
    main()
