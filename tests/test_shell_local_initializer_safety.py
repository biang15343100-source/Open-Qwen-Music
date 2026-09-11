from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHELL_ROOT = ROOT / "scripts"
ASSIGNMENT_PATTERN = re.compile(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=")
REFERENCE_PATTERN = re.compile(r"\$(?:\{)?([A-Za-z_][A-Za-z0-9_]*)")


def test_local_declarations_do_not_reference_same_command_assignments() -> None:

    failures: list[str] = []
    for path in sorted(SHELL_ROOT.rglob("*.sh")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped.startswith("local "):
                continue
            assignments = set(ASSIGNMENT_PATTERN.findall(stripped))
            references = set(REFERENCE_PATTERN.findall(stripped))
            risky = sorted(assignments & references)
            if risky:
                failures.append(
                    f"{path.relative_to(ROOT)}:{line_number}: "
                    f"declares and references {', '.join(risky)} on the same line: {stripped}"
                )

    assert not failures, "\n".join(failures)
