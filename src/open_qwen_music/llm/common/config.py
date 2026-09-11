
from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from open_qwen_music.common.config import expand_environment


DECLARE_NEW_KEYS = "declare_new_keys"


def leaf_paths(config: dict[str, Any], prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for key, value in (config or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            paths |= leaf_paths(value, path + ".")
        else:
            paths.add(path)
    return paths


def _declared(prefixes: Iterable[str], path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + ".") for prefix in prefixes)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _coerce_tree_like(
    base: dict[str, Any],
    override: dict[str, Any],
    *,
    prefix: str = "",
) -> None:
    for key, value in list(override.items()):
        if key not in base:
            continue
        dotted = f"{prefix}.{key}" if prefix else key
        current = base[key]
        if isinstance(current, dict) and isinstance(value, dict):
            _coerce_tree_like(current, value, prefix=dotted)
        else:
            override[key] = _coerce_like(value, current, dotted)


def load_config(path: str | Path, *, strict_keys: bool = True) -> dict[str, Any]:
    path = Path(path).resolve()
    current = expand_environment(
        yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    )
    base_path = current.pop("base", None)
    declared = current.pop(DECLARE_NEW_KEYS, None) or []
    if not isinstance(declared, list):
        raise TypeError(f"{path.name} field {DECLARE_NEW_KEYS} must be a list; got {declared!r}")
    if base_path is None:
        return current
    base = load_config(path.parent / base_path, strict_keys=strict_keys)
    if strict_keys:
        unknown = sorted(
            key
            for key in leaf_paths(current) - leaf_paths(base)
            if not _declared(declared, key)
        )
        if unknown:
            raise KeyError(
                f"{path.name} contains keys absent from {base_path}: {unknown}. "
                "A misspelled key would otherwise leave the base value unchanged. "
                f"To add keys intentionally, declare {DECLARE_NEW_KEYS}: [<dotted prefix>]."
            )
        _coerce_tree_like(base, current)
    return deep_merge(base, current)


def apply_overrides(
    config: dict[str, Any],
    overrides: Iterable[str] | Iterable[Iterable[str]],
    *,
    allow_new_keys: bool = False,
) -> list[str]:
    flat: list[str] = []
    for entry in overrides:
        if isinstance(entry, str):
            flat.append(entry)
        else:
            flat.extend(str(item) for item in entry)

    applied: list[str] = []
    for item in flat:
        if "=" not in item:
            raise ValueError(f"--override requires a.b=value form; got {item!r}")
        dotted, raw = item.split("=", 1)
        dotted = dotted.strip()
        if not dotted:
            raise ValueError(f"--override is empty:{item!r}")
        parts = dotted.split(".")
        if not allow_new_keys:
            _require_path(config, parts, dotted)
        node: Any = config
        for part in parts[:-1]:
            existing = node.get(part)
            if not isinstance(existing, dict):
                node[part] = {}
            node = node[part]
        value = yaml.safe_load(raw)
        if parts[-1] in node:
            value = _coerce_like(value, node[parts[-1]], dotted)
        node[parts[-1]] = value
        applied.append(dotted)
    return applied


def _coerce_like(value: Any, current: Any, dotted: str) -> Any:
    if value is None or current is None:
        return value
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ValueError(f"--override {dotted} requires true/false; got {value!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"--override {dotted} requires an integer; got {value!r}") from error
        if number != int(number):
            raise ValueError(f"--override {dotted} requires an integer; got {value!r}")
        return int(number)
    if isinstance(current, float):
        try:
            return float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"--override {dotted} requires a floating-point number; got {value!r} "
                "(write scientific notation in YAML as 1.0e-5; 1e-5 is parsed as a string)"
            ) from error
    if isinstance(current, (list, dict)) and not isinstance(value, type(current)):
        raise ValueError(
            f"--override {dotted} expects {type(current).__name__}; got {value!r}"
        )
    return value


def _require_path(config: dict[str, Any], parts: list[str], dotted: str) -> None:
    node: Any = config
    for depth, part in enumerate(parts):
        if not isinstance(node, dict) or part not in node:
            prefix = ".".join(parts[:depth]) or "<root>"
            candidates = sorted(node.keys()) if isinstance(node, dict) else []
            hint = _closest(part, candidates)
            raise KeyError(
                f"--override {dotted} refers to missing key {part!r} under {prefix}. "
                + (f"The closest existing key is {hint!r}. " if hint else "")
                + "Use --allow-new-key only when the application reads the new key."
            )
        node = node[part]


def _closest(name: str, candidates: list[str]) -> str | None:
    best: tuple[int, str] | None = None
    for candidate in candidates:
        shared = len(set(name) & set(candidate))
        prefix = len([1 for a, b in zip(name, candidate, strict=False) if a == b])
        score = prefix * 2 + shared
        if abs(len(candidate) - len(name)) <= 3 and (best is None or score > best[0]):
            best = (score, candidate)
    return best[1] if best else None


def require(config: dict[str, Any], dotted_key: str) -> Any:
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"configuration is missing {dotted_key}")
        node = node[part]
    return node


def get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
