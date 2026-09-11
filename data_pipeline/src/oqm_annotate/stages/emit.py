
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

from .base import LocalStage, StageReport
from .lyrics import working_point
from ..schema import SCHEMA_VERSION, validate_annotation
from ..store import ShardWriter


def has_no_content(record: dict[str, Any]) -> bool:

    return not (
        record.get("tags")
        or record.get("sections")
        or (record.get("structured_lyrics") or "").strip()
        or (record.get("description") or "").strip()
    )


def _attribute(
    reasons: list[str], record: dict[str, Any], stage_errors: dict[str, str]
) -> list[str]:

    if not stage_errors:
        return reasons

    names = sorted(stage_errors)
    kinds = sorted({str(msg).split(":", 1)[0].strip() or "unknown error"
                    for msg in stage_errors.values()})
    detail = f"{len(names)} failed stages ({', '.join(names)}; {', '.join(kinds)})"
    if has_no_content(record):
        return [
            f"{detail}; this sample produced no content, so all release tiers are false."
        ]
    return [*reasons, f"also{detail}，this sample has incomplete output"]


class EmitStage(LocalStage):
    name = "emit"

    #


    #


    #


    depends_on = ("index", "lyrics", "sections", "tags.fuse")

    def iter_inputs(self) -> Iterable[dict[str, Any]]:
        index = self.upstream("index")
        sections = self.upstream("sections")
        tags = self.upstream("tags.fuse")
        lyrics = self.upstream("lyrics")
        stage_errors = self.stage_errors()

        for sample_id, record in sorted(index.items()):
            yield {
                "sample_id": sample_id,
                "index": record,
                "sections": sections.get(sample_id) or {},
                "tags": tags.get(sample_id) or {},
                "lyrics": lyrics.get(sample_id) or {},
                "stage_errors": stage_errors.get(sample_id) or {},
            }

    def stage_errors(self) -> dict[str, dict[str, str]]:

        result: dict[str, dict[str, str]] = {}
        root = self.context.work_dir / "stages"
        if not root.is_dir():
            return result
        for directory in sorted(p for p in root.iterdir() if p.is_dir()):
            stage = directory.name
            if stage == self.name:
                continue
            records = list(self.context.store(stage).iter_records())
            succeeded = {
                str(r["sample_id"])
                for r in records
                if r.get("sample_id") and not r.get("error")
            }
            for record in records:
                sample_id = record.get("sample_id")
                error = record.get("error")
                if not sample_id or not error or str(sample_id) in succeeded:
                    continue
                result.setdefault(str(sample_id), {})[stage] = str(error)
        return result

    def process(self, item: dict[str, Any]) -> dict[str, Any] | None:
        index = item["index"]
        section_record = item["sections"]
        tag_record = item["tags"]
        stage_errors: dict[str, str] = item.get("stage_errors") or {}

        section_list = section_record.get("sections") or []
        tags = tag_record.get("tags") or {}


        minimum = float(self.config.get("min_tag_confidence", 0.4))
        confidence = tags.get("confidence") or {}
        emitted_tags: dict[str, Any] = {}
        for field in ("genre", "mood", "instrument", "vocal_timbre"):
            if confidence.get(field, 0.0) >= minimum and tags.get(field):
                emitted_tags[field] = tags[field]
        if confidence.get("vocal_gender", 0.0) >= minimum and tags.get("vocal_gender"):
            emitted_tags["vocal_gender"] = tags["vocal_gender"]

        record = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": item["sample_id"],
            "dataset": index.get("dataset", "unknown"),
            "audio_path": index.get("audio_path", ""),
            "audio": {
                "duration_sec": index.get("duration_sec", 0.0),
                "sample_rate": index.get("sample_rate"),
                "channels": index.get("channels"),
            },
            "language": section_record.get("language") or item["lyrics"].get("language"),
            "tags": emitted_tags,
            "description": tag_record.get("description", ""),
            "structured_lyrics": section_record.get("structured_lyrics", ""),
            "sections": section_list,
            "provenance": {
                "lyrics_tier": item["lyrics"].get("tier", "none"),


                "lyrics_status": item["lyrics"].get("lyrics_status", "unknown"),
                "lyrics_reason": item["lyrics"].get("reason", ""),


                "lyrics_confidence": item["lyrics"].get("lyrics_confidence", 0.0),
                "lyrics_pair_error": item["lyrics"].get("pair_error"),
                "lyric_coverage": section_record.get("lyric_coverage", 0.0),
                "align_available": section_record.get("align_available", False),


                #


                "hallucinated_windows": item["lyrics"].get("hallucinated_windows"),
                "hallucinated_chars": item["lyrics"].get("hallucinated_chars"),
                "tag_confidence": confidence,


                "tag_sources": tags.get("sources") or {},
                "gender_conflict": tag_record.get("gender_conflict", False),


                #


                #


                "llm_parse_failed": tag_record.get("llm_parse_failed", False),
                "unmapped_tags": tag_record.get("unmapped_tags") or {},


                "truncated_tags": tag_record.get("truncated_tags") or {},


                "failed_stages": dict(sorted(stage_errors.items())),
            },
            "usability": {},
        }
        record["usability"] = self._usability(record, stage_errors)
        return record

    def _usability(
        self, record: dict[str, Any], stage_errors: dict[str, str] | None = None
    ) -> dict[str, Any]:

        stage_errors = stage_errors or {}
        sections = record["sections"]
        provenance = record["provenance"]
        tags = record["tags"]

        reasons: list[str] = []
        has_structure = len(sections) >= 2
        if not has_structure:
            reasons.append("fewer than two sections")

        lyric_sections = [s for s in sections if s.get("has_lyrics")]


        #


        tier = provenance.get("lyrics_tier")
        no_lyrics_by_nature = tier in {"instrumental", "vocal_no_lyrics"}


        confidence = float(provenance.get("lyrics_confidence") or 0.0)
        checked = provenance.get("lyrics_tier") == "checked"

        #


        #


        #


        hallucinated = provenance.get("hallucinated_windows")
        has_hallucination = isinstance(hallucinated, (int, float)) and hallucinated > 0
        melody_ready = (
            has_structure
            and checked
            and not has_hallucination
            and confidence >= working_point(self.config, "melody_cot")
            and len(lyric_sections) >= int(self.config.get("min_lyric_sections", 2))
            and provenance.get("lyric_coverage", 0.0)
            >= float(self.config.get("min_lyric_coverage", 0.15))
        )
        if tier == "vocal_no_lyrics":


            #


            reasons.append(
                "vocals contain no lexical lyrics; Melody-CoT is not applicable"
            )
        elif no_lyrics_by_nature:
            reasons.append("instrumental audio has no lyrics; Melody-CoT is not applicable")
        elif provenance.get("lyrics_status") == "vocals_untranscribed":


            reasons.append("vocals are present but both transcripts are empty; Melody-CoT is not applicable")
        elif has_hallucination:


            reasons.append(
                f"a transcript window exceeds the language-specific character limit (maximum "
                f"{provenance.get('hallucinated_chars')} characters); Melody-CoT is not applicable"
            )
        elif not melody_ready:
            reasons.append("section lyrics are insufficient for Melody-CoT")


        sft_criteria = {
            "transcript agreement is below the SFT threshold": confidence
            >= working_point(self.config, "sft"),
            "genre tag is missing": bool(tags.get("genre")),
            "vocal_gender tag is missing": bool(tags.get("vocal_gender")),
            "vocal gender estimates disagree": not provenance.get("gender_conflict"),
        }
        sft_ready = melody_ready and all(sft_criteria.values())


        #


        if melody_ready and not sft_ready:
            failed = [name for name, passed in sft_criteria.items() if not passed]
            reasons.append(f"SFT is not applicable: {', '.join(failed)}")


        #


        pretrain_ready = has_structure and (
            no_lyrics_by_nature
            or not checked
            or confidence >= working_point(self.config, "pretrain")
        )

        return {
            "pretrain": pretrain_ready,
            "melody_cot": melody_ready,
            "sft": sft_ready,
            "reasons": _attribute(reasons, record, stage_errors),
        }

    def run(self) -> StageReport:
        started = time.time()
        report = StageReport(stage=self.name)

        output = Path(
            self.config.get("output_path")
            or (self.context.work_dir / "annotation.jsonl")
        )
        output.parent.mkdir(parents=True, exist_ok=True)

        problems: dict[str, list[str]] = {}
        stats = {
            "pretrain": 0,
            "melody_cot": 0,
            "sft": 0,
            "with_tags": 0,
            "with_lyrics": 0,
        }


        self.store.reset()
        with output.open("w", encoding="utf-8") as sink, ShardWriter(
            self.store, rank=0
        ) as writer:
            for item in self.iter_inputs():
                record = self.process(item)
                if record is None:
                    continue
                issues = validate_annotation(record)
                if issues:
                    problems[record["sample_id"]] = issues
                    report.failed += 1
                    writer.write(
                        {"sample_id": record["sample_id"], "error": "; ".join(issues)}
                    )
                    continue
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                writer.write({"sample_id": record["sample_id"], "ok": True})
                report.produced += 1
                for key in ("pretrain", "melody_cot", "sft"):
                    if record["usability"][key]:
                        stats[key] += 1
                if record["tags"]:
                    stats["with_tags"] += 1
                if record["structured_lyrics"]:
                    stats["with_lyrics"] += 1

        summary = {
            "output": str(output),
            "total": report.produced,
            "invalid": report.failed,
            **stats,
        }
        (self.store.dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if problems:
            (self.store.dir / "schema_problems.json").write_text(
                json.dumps(problems, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        report.notes = summary
        report.seconds = time.time() - started
        return self._finish(report)
