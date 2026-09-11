#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from _worker import (
    RankWriter,
    add_common_arguments,
    distribution,
    resolve_model_dir,
    run,
)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


# dubstep flamenco grunge hardcore industrial musical-theatre noise opera
# piano-solo post-rock progressive rap salsa ska synthwave tango trance`).

#


#


_GENRE_HINT = (
    "pop, rock, indie, folk, country, blues, jazz, soul, funk, disco, rnb, hiphop, "
    "trap, electronic, house, techno, ambient, lofi, edm, classical, orchestral, "
    "metal, punk, alternative, reggae, latin, bossa-nova, afrobeat, world, gospel, "
    "soundtrack, cinematic, ballad, acoustic, singer-songwriter, mandopop, cantopop, "
    "j-pop, k-pop, chinese-folk, chinese-traditional, new-age, experimental"
)
_MOOD_HINT = (
    "happy, sad, melancholic, nostalgic, romantic, dreamy, peaceful, calm, relaxing, "
    "uplifting, hopeful, triumphant, epic, dramatic, dark, tense, angry, aggressive, "
    "energetic, upbeat, playful, sexy, sensual, lonely, bittersweet, longing, warm, "
    "ethereal, mysterious, spiritual, motivational, confident, rebellious, groovy, chill"
)


#


#

_INSTRUMENT_HINT = (
    "acoustic-guitar, electric-guitar, bass-guitar, double-bass, piano, "
    "electric-piano, organ, synthesizer, accordion, harmonica, "
    "drums, drum-machine, percussion, tambourine, "
    "violin, viola, cello, strings, harp, "
    "trumpet, trombone, saxophone, clarinet, oboe, flute, brass, woodwind, "
    "erhu, guzheng, pipa, dizi, suona, guqin, banjo, mandolin, ukulele, sitar, "
    "choir, vocals, male-vocals, female-vocals"
)


_HINT_EXCLUSIONS: dict[str, str] = {
    "keyboard": "Use a specific keyboard-family instrument label when possible.",
}
_TIMBRE_HINT = (
    "warm, bright, dark, husky, raspy, breathy, airy, smooth, silky, clear, crisp, "
    "powerful, soft, gentle, thin, full, rich, nasal, gritty, sweet, deep, mellow, "
    "resonant, ethereal, high-pitched, low-pitched"
)


def plan_windows(
    duration_sec: float, num_windows: int, window_sec: float
) -> list[tuple[float, float]]:

    if num_windows < 1:
        raise ValueError("num_windows required >= 1")
    if window_sec <= 0:
        raise ValueError("window_sec required > 0")

    if duration_sec <= window_sec * num_windows:
        return [(0.0, duration_sec)]
    if num_windows == 1:
        start = max(0.0, (duration_sec - window_sec) / 2)
        return [(start, start + window_sec)]
    span = duration_sec - window_sec
    return [
        (span * i / (num_windows - 1), span * i / (num_windows - 1) + window_sec)
        for i in range(num_windows)
    ]


_INSTRUMENT_RULES = [
    "Rules:",
    "- Prefer tags from the given list. Only invent a tag when nothing in the"
    " list fits.",
    "- Omit a tag rather than guess. An empty array is much better than a wrong tag.",


    "- Do not list an instrument just because it is typical for the style."
    " If you cannot actually hear it in this recording, leave it out.",
    "- Backing vocals or a doubled lead vocal are not a choir. Use 'choir' only"
    " for an actual multi-voice choral ensemble.",
    "- Do not report tempo or musical key.",
]


def build_instrument_prompt(num_windows: int) -> str:

    lines = ["You are annotating a music track for a text-to-music training corpus."]
    if num_windows > 1:
        lines.append(
            f"The audio contains {num_windows} short excerpts from ONE track, in"
            " chronological order, separated by brief silence. They are sampled"
            " evenly from start to end, so together they cover the intro, the"
            " verses, the chorus and the outro. An instrument counts if you can"
            " hear it in ANY excerpt — sparse intros and verses are where a"
            " single instrument is most clearly exposed."
        )
    lines += [
        "Identify the instruments. Return a single JSON object with exactly one key:",
        "",
        f'  "instrument": array of 1-5 tags from [{_INSTRUMENT_HINT}].'
        " List only instruments you can actually pick out in the mix.",
        "",
    ]
    lines += _INSTRUMENT_RULES
    lines += ["", "Output only the JSON object. No prose, no code fence."]
    return "\n".join(lines)


#


#


def build_genre_prompt() -> str:

    return "\n".join(
        [
            "You are annotating a music track for a text-to-music training"
            " corpus. Listen to the audio and return a single JSON object with"
            " exactly one key:",
            "",
            f'  "genre": array of 1-3 tags from [{_GENRE_HINT}]',
            "",
            "Rules:",
            "- Prefer tags from the given list. Only invent a tag when nothing"
            " in the list fits.",
            "- Omit a tag rather than guess. An empty array is much better than"
            " a wrong tag.",
            "- Do not mention artist, title, or release year.",
            "",
            "Output only the JSON object. No prose, no code fence.",
        ]
    )


def build_holistic_prompt(record: dict[str, Any], *, include_genre: bool = True) -> str:

    known_sections = record.get("known_section_count")
    lines = [
        "You are annotating a music track for a text-to-music training corpus.",
        "Listen to the audio and return a single JSON object with exactly"
        " these keys:",
        "",
        *(
            [f'  "genre": array of 1-3 tags from [{_GENRE_HINT}]']
            if include_genre
            else []
        ),


        f'  "mood": array of 1-3 tags from [{_MOOD_HINT}]',
        '  "vocal_gender": exactly one of "female", "male", "mixed", "instrumental".'
        ' Use "instrumental" only when there is no singing at all. Use "mixed" only'
        " when both a male and a female voice sing lead parts.",
        f'  "vocal_timbre": array of 1-3 tags from [{_TIMBRE_HINT}], describing the'
        " singing voice quality. Empty array if there is no singing.",
        '  "section_count": integer, how many distinct structural sections you hear.',
        '  "description": 1-3 sentences describing only what is audible.',
        "",
        "Rules:",
        "- Prefer tags from the given lists. Only invent a tag when nothing in the"
        " list fits.",
        "- Omit a tag rather than guess. An empty array is much better than a wrong tag.",


        "- Do not mention artist, title, release year, or lyrics you cannot clearly hear.",
        "- Do not report tempo or musical key.",
    ]
    grounding = []
    if known_sections:
        grounding.append(
            f"Deterministic analysis found {known_sections} sections in this track."
            " Still report your own section_count above; it is used for consistency"
            " checking, so report what you actually hear."
        )
    if grounding:
        lines += ["", *grounding]
    lines += ["", "Output only the JSON object. No prose, no code fence."]
    return "\n".join(lines)


def build_prompt(
    record: dict[str, Any], num_windows: int = 1, *, include_genre: bool = True
) -> str:

    lines = ["You are annotating a music track for a text-to-music training corpus."]
    if num_windows > 1:
        lines.append(
            f"The audio contains {num_windows} short excerpts from ONE track, in"
            " chronological order, separated by brief silence."
        )
    lines += [
        "Return a single JSON object with exactly these keys:",
        "",
        *(
            [f'  "genre": array of 1-3 tags from [{_GENRE_HINT}]']
            if include_genre
            else []
        ),
        f'  "mood": array of 1-3 tags from [{_MOOD_HINT}]',
        f'  "instrument": array of 1-4 tags from [{_INSTRUMENT_HINT}].'
        " List only instruments you can actually pick out in the mix.",
        '  "vocal_gender": exactly one of "female", "male", "mixed", "instrumental".'
        ' Use "instrumental" only when there is no singing at all. Use "mixed" only'
        " when both a male and a female voice sing lead parts.",
        f'  "vocal_timbre": array of 1-3 tags from [{_TIMBRE_HINT}], describing the'
        " singing voice quality. Empty array if there is no singing.",
        '  "section_count": integer, how many distinct structural sections you hear.',
        '  "description": 1-3 sentences describing only what is audible.',
        "",
    ]
    lines += _INSTRUMENT_RULES
    lines += [
        "- Do not mention artist, title, release year, or lyrics you cannot"
        " clearly hear.",
    ]
    if record.get("known_section_count"):
        lines += [
            "",
            f"Deterministic analysis found {record['known_section_count']} sections"
            " in this track. Still report your own section_count above.",
        ]
    lines += ["", "Output only the JSON object. No prose, no code fence."]
    return "\n".join(lines)


def parse_response(text: str) -> dict[str, Any]:

    match = _JSON_BLOCK.search(text or "")
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:

        try:
            parsed = json.loads(match.group(0) + "}")
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


class _PercentSafeHelp(argparse.HelpFormatter):

    def _get_help_string(self, action: argparse.Action) -> str:
        return re.sub(r"%(?!\()", "%%", action.help or "")


def build_parser() -> argparse.ArgumentParser:

    parser = add_common_arguments(
        argparse.ArgumentParser(formatter_class=_PercentSafeHelp)
    )
    parser.add_argument("--model-dir", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-audio-sec",
        type=float,
        default=120.0,
        help="Total audio budget for split-pass inference. Used only with `--split-passes`.",
    )
    parser.add_argument(
        "--num-windows",
        type=int,
        default=1,
        help="Number of evenly spaced excerpts. Use 1 for the center excerpt; "
        "use multiple excerpts only with split-pass inference.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=25.0,
        help="Duration of each sampled excerpt in seconds.",
    )
    parser.add_argument(
        "--window-gap-sec",
        type=float,
        default=0.4,
        help="Silence inserted between sampled excerpts, in seconds.",
    )
    parser.add_argument(
        "--holistic-audio-sec",
        type=float,
        default=120.0,
        help="Continuous center excerpt duration for holistic tag questions, in seconds.",
    )
    parser.add_argument(
        "--split-passes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run orchestration and holistic tag questions as separate inference passes.",
    )
    parser.add_argument(
        "--split-genre-pass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Infer genre in a separate model pass.",
    )
    parser.add_argument(
        "--genre-max-new-tokens",
        type=int,
        default=64,
        help="Maximum tokens for the genre-only pass.",
    )
    parser.add_argument(
        "--instrument-max-new-tokens",
        type=int,
        default=96,
        help="Maximum tokens for the instrument-only pass.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation used by the model backend.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.split_passes and args.max_audio_sec != parser.get_default(
        "max_audio_sec"
    ):


        parser.error(
            "--max-audio-sec applies only with --split-passes. Use "
            "--holistic-audio-sec for single-pass inference."
        )
    return args


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = distribution()

    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    model_dir = resolve_model_dir(args.model_dir)
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_dir)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation=args.attn_implementation,
    ).eval()

    if args.prewarm:
        print(f'{{"rank": {rank}, "device": "{device}", "prewarm": true}}', flush=True)
        return

    import numpy as np

    from _audio import load_mono

    target_sr = getattr(processor.feature_extractor, "sampling_rate", 16000)


    audio_budget_sec = (
        args.max_audio_sec if args.split_passes else args.holistic_audio_sec
    )

    def montage(wave: Any) -> tuple[Any, list[tuple[float, float]], int]:

        duration_sec = len(wave) / target_sr


        #


        if args.num_windows <= 1:
            window_sec = audio_budget_sec
        else:
            window_sec = min(args.window_sec, audio_budget_sec / args.num_windows)
        windows = plan_windows(duration_sec, args.num_windows, window_sec)
        gap = np.zeros(int(args.window_gap_sec * target_sr), dtype=wave.dtype)
        chunks: list[Any] = []
        for start_sec, end_sec in windows:
            chunk = wave[int(start_sec * target_sr) : int(end_sec * target_sr)]
            if len(chunk):
                chunks.append(chunk)
        if len(chunks) > 1:
            joined = np.concatenate(
                [c for pair in zip(chunks, [gap] * len(chunks)) for c in pair][:-1]
            )
        else:
            joined = chunks[0]
        return joined, windows, len(chunks)

    def ask(wave: Any, prompt_text: str, max_new_tokens: int) -> str:

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": np.asarray(wave)},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=prompt,
            audio=[np.asarray(wave)],
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,


        ).to(model.device, dtype=model.dtype)

        with torch.no_grad():


            generated, _audio = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                thinker_return_dict_in_generate=False,
                return_audio=False,
            )
        return processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def handle(record: dict[str, Any], writer: RankWriter) -> None:
        audio_path = record["audio_path"]
        if not Path(audio_path).exists():
            raise FileNotFoundError(audio_path)

        wave, _ = load_mono(audio_path, target_sr)
        duration_sec = len(wave) / target_sr
        spread, windows, num_windows = montage(wave)

        if args.split_passes:

            inst_raw = ask(
                spread,
                build_instrument_prompt(num_windows),
                args.instrument_max_new_tokens,
            )
            inst_parsed = parse_response(inst_raw)

            (mid_start, mid_end), *_ = plan_windows(
                duration_sec, 1, args.holistic_audio_sec
            )
            contiguous = wave[
                int(mid_start * target_sr) : int(mid_end * target_sr)
            ]
            holistic_raw = ask(
                contiguous,
                build_holistic_prompt(record, include_genre=not args.split_genre_pass),
                args.max_new_tokens,
            )
            parsed = parse_response(holistic_raw)
            instrument = inst_parsed.get("instrument")

            failed = not parsed and not inst_parsed
            raw = "" if (parsed and inst_parsed) else f"{inst_raw}\n---\n{holistic_raw}"
            contiguous_window = [round(mid_start, 1), round(mid_end, 1)]
            genre_audio = contiguous
        else:
            completion = ask(
                spread,
                build_prompt(
                    record, num_windows, include_genre=not args.split_genre_pass
                ),
                args.max_new_tokens,
            )
            parsed = parse_response(completion)
            instrument = parsed.get("instrument")
            failed = not parsed
            raw = completion[:2000] if not parsed else ""
            contiguous_window = None
            genre_audio = spread

        genre = parsed.get("genre")
        if args.split_genre_pass:


            genre_raw = ask(
                genre_audio, build_genre_prompt(), args.genre_max_new_tokens
            )
            genre_parsed = parse_response(genre_raw)
            genre = genre_parsed.get("genre")
            if not genre_parsed:
                raw = f"{raw}\n---\n{genre_raw}" if raw else genre_raw

        writer.write(
            {
                "sample_id": record["sample_id"],
                "tags": {
                    "genre": genre,
                    "mood": parsed.get("mood"),
                    "instrument": instrument,
                    "vocal_gender": parsed.get("vocal_gender"),
                    "vocal_timbre": parsed.get("vocal_timbre"),
                },
                "description": str(parsed.get("description") or "").strip(),

                "section_count": parsed.get("section_count"),
                "known_section_count": record.get("known_section_count"),
                "parse_failed": failed,
                "raw": raw[:2000],
                "model": args.model_dir,


                "windows": [[round(s, 1), round(e, 1)] for s, e in windows],
                "contiguous_window": contiguous_window,
                "split_passes": bool(args.split_passes),
                "split_genre_pass": bool(args.split_genre_pass),
            }
        )

    run(args, rank=rank, world_size=world_size, handle=handle)


if __name__ == "__main__":
    main()
