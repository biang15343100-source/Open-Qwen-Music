
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np


PROMOTED_FIELD = "quality.promoted"


def load_promoted_ids(
    *, ids: Iterable[str] | None = None, ids_file: str | Path | None = None
) -> set[str]:
    collected: set[str] = set()
    for value in ids or ():
        text = str(value).strip()
        if text:
            collected.add(text)
    if ids_file:
        path = Path(ids_file)
        if not path.exists():
            raise FileNotFoundError(f"Promotion list file does not exist:{path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith("{"):
                record = json.loads(text)
                sample_id = str(record.get("sample_id", "")).strip()
                if not sample_id:
                    raise ValueError(f"Promotion list JSONL line is missing sample_id:{text[:80]}")
                collected.add(sample_id)
            else:
                collected.add(text)
    return collected


def resolve_promoted_indices(
    manifest: str | Path, ids: Iterable[str], *, strict: bool = True
) -> np.ndarray:
    wanted = {str(value) for value in ids}
    if not wanted:
        return np.empty(0, dtype=np.int64)
    found: dict[str, int] = {}
    with Path(manifest).open("rb") as handle:
        row = 0
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            sample_id = str(record.get("sample_id", ""))
            if sample_id in wanted:
                found[sample_id] = row
            row += 1
    missing = sorted(wanted - set(found))
    if missing and strict:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f" {len(missing)}  sample_id in {manifest} does not exist in:{preview}..."
        )
    return np.asarray(sorted(found.values()), dtype=np.int64)


def promoted_indices_from_index(index) -> np.ndarray:  # noqa: ANN001 -  index
    try:
        code = index.code_of(PROMOTED_FIELD, "true")
    except KeyError:

        return np.empty(0, dtype=np.int64)
    if code is None:
        return np.empty(0, dtype=np.int64)
    codes = index.codes(PROMOTED_FIELD)
    return np.nonzero(codes == np.uint16(code))[0].astype(np.int64)


def resolve_promotion_pool(
    index,  # noqa: ANN001
    *,
    from_manifest_flag: bool = True,
    ids: Sequence[str] | None = None,
    ids_file: str | Path | None = None,
    strict: bool = True,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    if from_manifest_flag:
        parts.append(promoted_indices_from_index(index))
    wanted = load_promoted_ids(ids=ids, ids_file=ids_file)
    if wanted:
        parts.append(resolve_promoted_indices(index.manifest_path, wanted, strict=strict))
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(parts)).astype(np.int64)
