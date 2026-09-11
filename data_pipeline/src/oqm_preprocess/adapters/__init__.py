
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..registry import DatasetSpec


AdapterFn = Callable[[DatasetSpec, list[dict[str, Any]]], dict[str, Any]]

_REGISTRY: dict[str, AdapterFn] = {}


def register(name: str) -> Callable[[AdapterFn], AdapterFn]:
    def wrap(fn: AdapterFn) -> AdapterFn:
        if name in _REGISTRY:
            raise ValueError(f"Adapter {name!r} is already registered")
        _REGISTRY[name] = fn
        return fn

    return wrap


def get(name: str) -> AdapterFn:
    fn = _REGISTRY.get(name)
    if fn is None:
        raise KeyError(f"Unknown adapter {name!r}; registered adapters: {sorted(_REGISTRY)}")
    return fn


def has(name: str) -> bool:
    return name in _REGISTRY


def names() -> list[str]:
    return sorted(_REGISTRY)
