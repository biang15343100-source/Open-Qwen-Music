
from __future__ import annotations

from typing import Any, Iterable

from .base import LocalStage, WorkerStage
from ..schema import MusicalTags
from ..taxonomy import normalize_gender, normalize_tag_list_detailed


class TagsLlmStage(WorkerStage):

    name = "tags.llm"
    depends_on = ("index",)
    script = "w_tags_llm.py"

    def select_inputs(self) -> list[dict[str, Any]]:
        index = self.upstream("index")
        structure = self.upstream("structure")
        items: list[dict[str, Any]] = []
        for sample_id, record in sorted(index.items()):
            sections = (structure.get(sample_id) or {}).get("sections") or []
            items.append(
                {
                    "sample_id": sample_id,
                    "audio_path": record["audio_path"],
                    "duration_sec": record.get("duration_sec", 0.0),


                    "known_section_count": len(sections) or None,
                }
            )
        return items


class VoiceAcousticStage(WorkerStage):

    name = "voice"
    depends_on = ("separate",)
    script = "w_voice.py"

    def select_inputs(self) -> list[dict[str, Any]]:
        rows = self.upstream("separate")
        items: list[dict[str, Any]] = []
        for sample_id, record in sorted(rows.items()):
            stem = record.get("vocal_16k_path")
            if not stem:
                continue
            items.append(
                {
                    "sample_id": sample_id,
                    "audio_path": stem,
                    "duration_sec": record.get("duration_sec", 0.0),


                    "vocal_ratio": record.get("vocal_ratio"),
                }
            )
        return items


_LLM_PRIOR = {"genre": 0.85, "mood": 0.75, "instrument": 0.80, "vocal_timbre": 0.60}


#


#

_GENDER_LLM_ONLY = 0.94
_GENDER_BOTH_AGREE = 0.96


_GENDER_ACOUSTIC_ONLY = 0.89


_GENDER_CONFLICT = 0.64


_MIN_VOCAL_RATIO = 0.30


#


#


_TIMBRE_ACOUSTIC_ONLY = 0.35


#


#

_TIMBRE_GENDER_CONTRADICTED = 0.20


_GENDER_FALSIFIED = 0.20


#


#


#


#


_MIN_FALSIFYING_ASR_CHARS = 30


class TagsFuseStage(LocalStage):

    name = "tags.fuse"


    depends_on = (
        "tags.llm",
        "voice",
        "sections",
        "separate",
        "asr.vocal",
        "asr.mix",
    )

    def iter_inputs(self) -> Iterable[dict[str, Any]]:
        llm = self.upstream("tags.llm")
        voice = self.upstream("voice")
        sections = self.upstream("sections")
        separate = self.upstream("separate")


        asr_vocal = self.upstream("asr.vocal")
        asr_mix = self.upstream("asr.mix")
        for sample_id, record in sorted(llm.items()):
            yield {
                "sample_id": sample_id,
                "llm": record,
                "voice": voice.get(sample_id) or {},
                "sections": sections.get(sample_id) or {},
                "separate": separate.get(sample_id) or {},
                "asr_vocal": asr_vocal.get(sample_id) or {},
                "asr_mix": asr_mix.get(sample_id) or {},
            }

    def process(self, item: dict[str, Any]) -> dict[str, Any] | None:
        llm = item["llm"]
        voice = item["voice"]
        raw_tags = llm.get("tags") or {}

        tags = MusicalTags()
        unmapped: dict[str, list[str]] = {}


        truncated: dict[str, list[str]] = {}
        limit = int(self.config.get("max_tags_per_field", 5))

        for field in ("genre", "mood", "instrument", "vocal_timbre"):
            mapped, dropped, cut = normalize_tag_list_detailed(
                raw_tags.get(field), field, limit=limit
            )
            setattr(tags, field, mapped)
            if dropped:
                unmapped[field] = dropped
            if cut:
                truncated[field] = cut
            if mapped:
                tags.confidence[field] = _LLM_PRIOR[field]
                tags.sources[field] = ["llm"]
            else:
                tags.confidence[field] = 0.0
                tags.sources[field] = []

        gender, gender_confidence, gender_sources, conflict = self._fuse_gender(
            raw_tags.get("vocal_gender"),
            voice,
            item["sections"],
            item["separate"],
            item.get("asr_vocal") or {},
            item.get("asr_mix") or {},
        )
        tags.vocal_gender = gender
        tags.confidence["vocal_gender"] = gender_confidence
        tags.sources["vocal_gender"] = gender_sources


        acoustic_timbre = voice.get("timbre_hint")
        if acoustic_timbre and tags.vocal_timbre:
            if acoustic_timbre in tags.vocal_timbre:
                tags.confidence["vocal_timbre"] = min(
                    1.0, tags.confidence["vocal_timbre"] + 0.15
                )
                tags.sources["vocal_timbre"].append("acoustic")
        elif acoustic_timbre and not tags.vocal_timbre:
            tags.vocal_timbre = [acoustic_timbre]
            tags.confidence["vocal_timbre"] = _TIMBRE_ACOUSTIC_ONLY
            tags.sources["vocal_timbre"] = ["acoustic"]


        if conflict and gender == "instrumental" and tags.vocal_timbre:
            tags.confidence["vocal_timbre"] = min(
                tags.confidence.get("vocal_timbre", 0.0), _TIMBRE_GENDER_CONTRADICTED
            )

        return {
            "sample_id": item["sample_id"],
            "tags": tags.to_dict(),
            "description": (llm.get("description") or "").strip(),
            "unmapped_tags": unmapped,
            "truncated_tags": truncated,
            "gender_conflict": conflict,
            "llm_parse_failed": bool(llm.get("parse_failed")),
        }

    @staticmethod
    def _transcribed_chars(record: dict[str, Any]) -> int:
        return len((record.get("text") or "").strip())

    def _singing_is_transcribed(
        self, asr_vocal: dict[str, Any], asr_mix: dict[str, Any]
    ) -> bool:

        floor = _MIN_FALSIFYING_ASR_CHARS
        return (
            self._transcribed_chars(asr_vocal) >= floor
            and self._transcribed_chars(asr_mix) >= floor
        )

    def _fuse_gender(
        self,
        llm_value: Any,
        voice: dict[str, Any],
        sections: dict[str, Any],
        separate: dict[str, Any] | None = None,
        asr_vocal: dict[str, Any] | None = None,
        asr_mix: dict[str, Any] | None = None,
    ) -> tuple[str | None, float, list[str], bool]:

        llm_gender = normalize_gender(llm_value)
        acoustic_gender = voice.get("gender_hint")
        acoustic_confidence = float(voice.get("gender_confidence") or 0.0)
        minimum = float(self.config.get("min_acoustic_gender_confidence", 0.55))


        #


        #


        #


        #


        if llm_gender == "instrumental" and self._singing_is_transcribed(
            asr_vocal or {}, asr_mix or {}
        ):
            return None, _GENDER_FALSIFIED, ["llm", "asr"], True


        #


        #


        #


        section_list = sections.get("sections") or []
        if section_list and not any(s.get("is_vocal") for s in section_list):
            overridden = llm_gender is not None and llm_gender != "instrumental"
            return "instrumental", 0.9, ["sections"], overridden

        if acoustic_confidence < minimum:
            acoustic_gender = None


        vocal_ratio = (separate or {}).get("vocal_ratio")
        quiet = vocal_ratio is not None and float(vocal_ratio) < _MIN_VOCAL_RATIO
        if llm_gender == "instrumental" and quiet:
            return "instrumental", _GENDER_BOTH_AGREE, ["llm", "separate"], False

        if llm_gender and acoustic_gender:
            if llm_gender == acoustic_gender:

                return llm_gender, _GENDER_BOTH_AGREE, ["llm", "acoustic"], False
            if llm_gender == "instrumental":


                return llm_gender, _GENDER_LLM_ONLY, ["llm"], False


            return llm_gender, _GENDER_CONFLICT, ["llm"], True
        if llm_gender:
            return llm_gender, _GENDER_LLM_ONLY, ["llm"], False
        if acoustic_gender:


            return acoustic_gender, min(_GENDER_ACOUSTIC_ONLY, acoustic_confidence), [
                "acoustic"
            ], False
        return None, 0.0, [], False
