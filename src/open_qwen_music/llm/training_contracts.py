"""Pure validation helpers shared by LLM training and CPU tests."""

from __future__ import annotations

from typing import Any


_MISSING_REVISIONS = frozenset({"", "unknown", "none", "null"})


def _normalized_revision(value: object) -> str:
    return str(value or "").strip().lower()


def validate_production_tokenizer_identity(
    config: dict[str, Any],
    *,
    tokenizer_revision: str | None,
    semantic_extractor_revision: str | None,
    source: str,
) -> None:
    """Require a production corpus to match both frozen tokenizer identities."""

    data = dict(config.get("data", {}) or {})
    if not bool(data.get("require_production_contract", False)):
        return

    registry = dict(config.get("registry", {}) or {})
    pairs = (
        (
            "tokenizer revision",
            tokenizer_revision,
            registry.get("semantic_tokenizer_revision"),
        ),
        (
            "semantic extractor revision",
            semantic_extractor_revision,
            registry.get("semantic_extractor_revision"),
        ),
    )
    for label, found, configured in pairs:
        normalized_found = _normalized_revision(found)
        normalized_configured = _normalized_revision(configured)
        if normalized_found in _MISSING_REVISIONS:
            raise RuntimeError(f"{source} does not declare a frozen {label}: {found!r}")
        if normalized_configured in _MISSING_REVISIONS:
            raise RuntimeError(
                f"The training configuration does not declare the expected {label}: "
                f"{configured!r}"
            )
        if normalized_found != normalized_configured:
            raise RuntimeError(
                f"{source} {label} {found!r} does not match the configured "
                f"revision {configured!r}"
            )


def initial_data_step(
    *,
    optimizer_step: int,
    data_state_active: bool,
    parent_step: int,
    extra: dict[str, Any] | None,
) -> int:
    """Resolve the data cursor independently from the stage-local optimizer step."""

    return int(
        (extra or {}).get(
            "data_step",
            parent_step if data_state_active else optimizer_step,
        )
    )
