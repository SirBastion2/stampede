"""Load and query the parameter registry (§8.1)."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_REGISTRY_PATH = Path(__file__).resolve().parent / "registry.csv"
_REQUIRED_COLUMNS = ("Parameter", "Value", "Status", "Source")


@dataclass(frozen=True, slots=True)
class RegistryRow:
    parameter: str
    value: str
    status: str
    source: str
    module: str
    unit: str


class RegistryError(ValueError):
    """Invalid registry content or lookup failure."""


@dataclass(frozen=True, slots=True)
class _CachedRegistry:
    mtime_ns: int
    size: int
    rows: list[RegistryRow]
    by_name: dict[str, RegistryRow]


_CACHE: dict[Path, _CachedRegistry] = {}


def _normalize_fieldnames(fieldnames: Sequence[str] | None) -> list[str]:
    if not fieldnames:
        raise RegistryError("registry CSV has no header row")
    return [name.strip() for name in fieldnames]


def _resolved_path(path: Path | None) -> Path:
    return (path or _REGISTRY_PATH).resolve()


def _parse_registry(registry_path: Path) -> list[RegistryRow]:
    with registry_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = _normalize_fieldnames(reader.fieldnames)
        for col in _REQUIRED_COLUMNS:
            if col not in fieldnames:
                raise RegistryError(f"missing required column: {col}")
        module_key = "Module" if "Module" in fieldnames else ""
        unit_key = "Unit" if "Unit" in fieldnames else ""
        rows: list[RegistryRow] = []
        for line_no, raw in enumerate(reader, start=2):
            name = (raw.get("Parameter") or "").strip()
            if not name:
                raise RegistryError(f"row {line_no}: empty Parameter")
            rows.append(
                RegistryRow(
                    parameter=name,
                    value=(raw.get("Value") or "").strip(),
                    status=(raw.get("Status") or "").strip(),
                    source=(raw.get("Source") or "").strip(),
                    module=(raw.get(module_key) or "").strip() if module_key else "",
                    unit=(raw.get(unit_key) or "").strip() if unit_key else "",
                )
            )
    return rows


def _cached_registry(path: Path | None = None) -> _CachedRegistry:
    registry_path = _resolved_path(path)
    stat = registry_path.stat()
    cached = _CACHE.get(registry_path)
    if (
        cached is not None
        and cached.mtime_ns == stat.st_mtime_ns
        and cached.size == stat.st_size
    ):
        return cached
    rows = _parse_registry(registry_path)
    by_name: dict[str, RegistryRow] = {}
    for row in rows:
        if row.parameter not in by_name:
            by_name[row.parameter] = row
    cached = _CachedRegistry(
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
        rows=rows,
        by_name=by_name,
    )
    _CACHE[registry_path] = cached
    return cached


def load_registry(path: Path | None = None) -> list[RegistryRow]:
    """Load all rows from the parameter registry CSV."""
    return list(_cached_registry(path).rows)


def get_parameter(name: str, path: Path | None = None) -> RegistryRow:
    """Return the registry row for ``name``; raise RegistryError if missing."""
    row = _cached_registry(path).by_name.get(name)
    if row is None:
        raise RegistryError(f"unknown parameter: {name}")
    return row


def list_by_status(status: str, path: Path | None = None) -> list[RegistryRow]:
    """Return all rows whose Status equals ``status`` (case-sensitive)."""
    return [row for row in load_registry(path) if row.status == status]


def list_by_module(module: str, path: Path | None = None) -> list[RegistryRow]:
    """Return all rows whose Module equals ``module`` (case-sensitive)."""
    return [row for row in load_registry(path) if row.module == module]
