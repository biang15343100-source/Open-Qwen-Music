
from . import dedup_stage, discover, enrich, filter_stage, normalize, probe, publish

STAGE_ORDER = [
    ("discover", discover),
    ("probe", probe),
    ("enrich", enrich),
    ("filter", filter_stage),
    ("dedup", dedup_stage),
    ("normalize", normalize),
    ("publish", publish),
]

__all__ = [
    "STAGE_ORDER",
    "dedup_stage",
    "discover",
    "enrich",
    "filter_stage",
    "normalize",
    "probe",
    "publish",
]
