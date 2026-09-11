
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_NAME = "_oqm_audio"
_MODULE: ModuleType | None = None


def _load() -> ModuleType:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    path = Path(__file__).resolve().parent / "workers" / "_audio.py"
    spec = importlib.util.spec_from_file_location(_NAME, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(
            f"Unable to load the packaged audio decoder from {path}. "
            "A fallback decoder could change duration measurements."
        )
    module = importlib.util.module_from_spec(spec)


    sys.modules[_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_NAME, None)
        raise
    _MODULE = module
    return module


def __getattr__(name: str):
    module = _load()
    try:
        return getattr(module, name)
    except AttributeError:
        raise AttributeError(f"workers/_audio.py does not define {name!r}") from None
