"""SVB-22: registry defaults that match a real run (§8.1)."""

from __future__ import annotations

from pathlib import Path

from params.loader import get_parameter, list_by_status, load_registry

_ROOT = Path(__file__).resolve().parents[2]
_SOFTNESS_NAMES = (
    "Calibration softness initial",
    "Calibration softness final",
)
_SKIP_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
}
_SCAN_SUFFIXES = {
    ".py",
    ".md",
    ".csv",
    ".json",
    ".ts",
    ".tsx",
    ".js",
    ".toml",
    ".yml",
    ".yaml",
}
# Leftover anneal helpers in fit.py still mention the names; live fit uses
# fixed_softness() / Calibration behavioral noise temperature (SVB-01).
_ALLOWED_SOFTNESS_PATHS = {
    Path("params/registry.csv"),
    Path("tests/params/test_svb22_registry_defaults.py"),
    Path("calibration/fit.py"),
}


def test_cma_and_eval_defaults_match_a_real_run() -> None:
    generations = get_parameter("Calibration CMA-ES max generations")
    assert generations.value == "40"
    assert generations.status == "ASSUMED"

    minutes = get_parameter("Calibration eval simulation minutes")
    assert minutes.value == "1440"
    assert minutes.status == "ASSUMED"


def test_softness_anneal_rows_are_deprecated() -> None:
    note = "unused since SVB-01 (fixed behavioral noise)."
    for name in _SOFTNESS_NAMES:
        row = get_parameter(name)
        assert row.status == "DEPRECATED"
        assert row.source == note
        assert row.value  # rows stay; values are not deleted

    deprecated = list_by_status("DEPRECATED")
    names = {row.parameter for row in deprecated}
    assert set(_SOFTNESS_NAMES) <= names
    assert all(row.status == "DEPRECATED" for row in load_registry() if row.parameter in _SOFTNESS_NAMES)


def test_deprecated_softness_rows_unread() -> None:
    hits: list[str] = []
    for path in _ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _SCAN_SUFFIXES:
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        relative = path.relative_to(_ROOT)
        if relative in _ALLOWED_SOFTNESS_PATHS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(name in text for name in _SOFTNESS_NAMES):
            hits.append(relative.as_posix())
    assert hits == []
