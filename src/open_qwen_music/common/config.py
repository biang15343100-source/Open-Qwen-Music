
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import re
from typing import Any

import yaml


_ENVIRONMENT = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_environment(value: Any) -> Any:

    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise KeyError(f"Configuration references an environment variable that is not set ${{{name}}}")

        return _ENVIRONMENT.sub(replace, value)
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and value.get("__replace__") is True:
            result[key] = deepcopy(
                {
                    nested_key: nested_value
                    for nested_key, nested_value in value.items()
                    if nested_key != "__replace__"
                }
            )
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            previous = result[key]


            if (
                "name" in value
                and "name" in previous
                and value["name"] != previous["name"]
            ):
                result[key] = deepcopy(value)
            else:
                result[key] = deep_merge(previous, value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(
    path: str | Path, *, _stack: tuple[Path, ...] = ()
) -> dict[str, Any]:
    path = Path(path).resolve()
    if path in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, path))
        raise ValueError(f"configuration base Circular inheritance:{chain}")
    current = expand_environment(
        yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    )
    base_spec = current.pop("base", None)
    if base_spec is None:
        return current
    base_paths = [base_spec] if isinstance(base_spec, str) else list(base_spec)
    if not base_paths or not all(isinstance(item, str) for item in base_paths):
        raise ValueError(f"base must be a path string or a non-empty path list,received {base_spec!r}")
    merged: dict[str, Any] = {}
    for base_path in base_paths:
        merged = deep_merge(
            merged,
            load_config(path.parent / base_path, _stack=(*_stack, path)),
        )
    return deep_merge(merged, current)
