
from __future__ import annotations

import numpy as np

from ..config import DedupConfig
from ..enums import CODEC


_LOSSLESS = frozenset(
    CODEC.code(name) for name in
    ("pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "pcm_f64le", "flac", "alac")
)


def compute_scores(
    codec: np.ndarray,
    sample_rate: np.ndarray,
    duration: np.ndarray,
    has_lyrics: np.ndarray,
    has_meta: np.ndarray,
    is_synthetic: np.ndarray,
    priority: np.ndarray,
    cfg: DedupConfig,
) -> np.ndarray:
    lossless = np.isin(codec, list(_LOSSLESS)).astype(np.float32)

    sr = np.log2(np.maximum(sample_rate.astype(np.float32), 1.0) / 8000.0)
    sr = np.clip(sr, 0.0, 4.0) / 4.0
    dur = np.log1p(np.maximum(duration.astype(np.float32), 0.0)) / 10.0

    score = (
        cfg.weight_lossless * lossless
        + cfg.weight_sample_rate * sr
        + cfg.weight_has_lyrics * has_lyrics.astype(np.float32)
        + cfg.weight_has_tags * has_meta.astype(np.float32)
        + cfg.weight_duration * np.clip(dur, 0.0, 1.0)
        + cfg.weight_priority * priority.astype(np.float32)
        - cfg.penalty_synthetic * is_synthetic.astype(np.float32)
    )
    return score.astype(np.float32)


def pick_canonical(roots: np.ndarray, scores: np.ndarray, uids: np.ndarray) -> np.ndarray:
    n = roots.size
    if n == 0:
        return np.empty(0, dtype=np.int64)

    order = np.lexsort((uids, -scores, roots))
    sorted_roots = roots[order]
    first = np.empty(n, dtype=bool)
    first[0] = True
    np.not_equal(sorted_roots[1:], sorted_roots[:-1], out=first[1:])

    representative = np.empty(n, dtype=np.int64)
    current = -1
    for pos in range(n):
        if first[pos]:
            current = int(order[pos])
        representative[order[pos]] = current
    return representative
