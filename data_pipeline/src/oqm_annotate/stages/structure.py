
from __future__ import annotations

import json
from typing import Any

from ..corpus import GRANULARITY_WHOLE
from .base import WorkerStage


class StructureStage(WorkerStage):
    name = "structure"
    depends_on = ("index",)
    script = "w_structure.py"

    def select_inputs(self) -> list[dict[str, Any]]:
        rows = self.upstream("index")
        selected: list[dict[str, Any]] = []
        routed_out: dict[str, int] = {}

        for sample_id, record in sorted(rows.items()):


            granularity = str(record.get("granularity") or GRANULARITY_WHOLE)
            if granularity != GRANULARITY_WHOLE:
                routed_out[granularity] = routed_out.get(granularity, 0) + 1
                continue
            selected.append(
                {
                    "sample_id": sample_id,


                    "audio_path": record["audio_path"],
                    "duration_sec": record.get("duration_sec", 0.0),
                }
            )

        self.store.dir.mkdir(parents=True, exist_ok=True)
        (self.store.dir / "granularity_routing.json").write_text(
            json.dumps(
                {
                    "index_records": len(rows),
                    "sent_to_songformer": len(selected),
                    "routed_out_by_granularity": routed_out,
                    "note": (
                        "routed_out records use phrase-level routing. They still receive "
                        "tags and line-level lyrics, but do not produce sections."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return selected
