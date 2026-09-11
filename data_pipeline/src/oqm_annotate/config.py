
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover -
    yaml = None  # type: ignore[assignment]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:

    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            resolved = os.environ.get(name)
            if resolved is None:
                if default is None:
                    raise KeyError(f"Configuration references an environment variable that is not set ${{{name}}}")
                return default
            return resolved

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:

    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _safe_align_sec() -> float:

    import importlib.util

    path = Path(__file__).resolve().parent / "workers" / "_limits.py"
    spec = importlib.util.spec_from_file_location("_oqm_limits", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(
            f"Unable to load the alignment limit from {path}. "
            "The configured ASR window cannot be validated safely."
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return float(module.SAFE_ALIGN_SEC)


def assert_align_window_bound(config: dict[str, Any]) -> None:

    limit = _safe_align_sec()
    for stage in ("asr.vocal", "asr.mix"):
        section = stage_config(config, stage)
        extra = section.get("extra_args") or {}
        raw = extra.get("window-sec", extra.get("window_sec"))
        if raw is None:
            continue
        try:
            window = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{stage}.window-sec must be numeric; got {raw!r}") from None
        if window > limit:
            raise ValueError(
                f"{stage} of window-sec={window:g}s exceeds the safe upper limit of forced alignment "
                f"{limit:g}s. Longer windows can produce plausible but invalid timestamps. "
                "Recalibrate the alignment limit before increasing this value."
            )


_ALLOWED_TOP_KEYS = frozenset(
    {
        "base",
        "work_dir",
        "index",
        "separate",
        "structure",
        "asr.vocal",
        "asr.mix",
        "lyrics",
        "align",
        "sections",
        "voice",
        "tags.llm",
        "tags.fuse",
        "emit",
    }
)


def assert_known_top_keys(raw: dict[str, Any], target: Path) -> None:

    unknown = sorted(set(raw) - _ALLOWED_TOP_KEYS)
    if not unknown:
        return
    hints = []
    for key in unknown:
        if key in ("extends", "inherit", "include", "parent", "from"):
            hints.append(f"    `{key}` looks like an inheritance declaration; use the `base:` key")
    hint_text = ("\n" + "\n".join(hints)) if hints else ""
    raise ValueError(
        f"{target} contains unknown top-level keys: {unknown}\n"
        f"  Allowed top-level keys: {sorted(_ALLOWED_TOP_KEYS)}"
        f"{hint_text}"
    )


def load_config(path: Path | str) -> dict[str, Any]:

    resolved = _load_merged(Path(path))

    assert_align_window_bound(resolved)
    return resolved


def _load_merged(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required; install it with `pip install pyyaml`")
    target = Path(path).resolve()
    with target.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{target} must contain a mapping at the top level")


    assert_known_top_keys(raw, target)

    base_ref = raw.pop("base", None)
    if base_ref:
        parent = _load_merged((target.parent / str(base_ref)).resolve())
        raw = _deep_merge(parent, raw)

    resolved = _expand_env(raw)
    resolved["_config_path"] = str(target)
    return resolved


def stage_config(config: dict[str, Any], stage: str) -> dict[str, Any]:

    if stage in config and isinstance(config[stage], dict):
        return dict(config[stage])
    node: Any = config
    for part in stage.split("."):
        if not isinstance(node, dict) or part not in node:
            return {}
        node = node[part]
    return dict(node) if isinstance(node, dict) else {}


#


_HASH_EXCLUDED_KEYS = frozenset(
    {


        "python",
    }
)


#


def config_hash(config: dict[str, Any], stage: str) -> str:

    section = {
        key: value
        for key, value in stage_config(config, stage).items()
        if key not in _HASH_EXCLUDED_KEYS
    }
    payload = json.dumps(section, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require(config: dict[str, Any], key: str) -> Any:

    if key not in config or config[key] in (None, ""):
        raise KeyError(f"Configuration {config.get('_config_path', '?')} is missing required field `{key}`")
    return config[key]
