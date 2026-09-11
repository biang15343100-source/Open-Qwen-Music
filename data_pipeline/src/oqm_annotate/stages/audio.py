
from __future__ import annotations

from typing import Any

from .base import WorkerStage


class SeparateStage(WorkerStage):
    name = "separate"
    depends_on = ("index",)
    script = "w_separate.py"

    def select_inputs(self) -> list[dict[str, Any]]:
        rows = self.upstream("index")
        return [
            {
                "sample_id": sample_id,
                "audio_path": record["audio_path"],
                "duration_sec": record.get("duration_sec", 0.0),
            }
            for sample_id, record in sorted(rows.items())
        ]
