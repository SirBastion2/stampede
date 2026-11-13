"""SVB-27f: one halt length everywhere. The real book reads the SOURCED 5-minute row."""
from __future__ import annotations

from pathlib import Path

from params.loader import get_parameter

_REPO = Path(__file__).resolve().parents[1]
_RETIRED = "M14 trading halt duration"


def test_no_code_reads_the_retired_halt_duration_row() -> None:
    readers = []
    for path in _REPO.rglob("*.py"):
        rel = path.relative_to(_REPO).as_posix()
        if rel.startswith(("tests/", ".venv/", "venv/")) or "/site-packages/" in rel:
            continue
        if _RETIRED in path.read_text(encoding="utf-8", errors="ignore"):
            readers.append(rel)
    assert readers == [], f"still reading {_RETIRED!r}: {readers}"


def test_halt_rows() -> None:
    assert get_parameter("M14 trading halt minutes").value.strip() == "5"
    assert get_parameter(_RETIRED).status.strip().upper() == "DEPRECATED"
