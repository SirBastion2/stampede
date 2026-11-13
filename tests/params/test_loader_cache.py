"""Cache behavior for params/loader.py (§8.1)."""

from __future__ import annotations

import csv
from pathlib import Path

from params.loader import RegistryRow, get_parameter, load_registry

_HEADER = "Parameter,Value,Status,Source,Module,Unit\n"
_REQUIRED_COLUMNS = ("Parameter", "Value", "Status", "Source")


def _write_registry(path: Path, value: str) -> Path:
    path.write_text(
        _HEADER + f"Demo param,{value},ASSUMED,test,M00,\n",
        encoding="utf-8",
    )
    return path


def _uncached_parse(registry_path: Path) -> list[RegistryRow]:
    with registry_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise AssertionError("registry CSV has no header row")
        names = [name.strip() for name in fieldnames]
        for col in _REQUIRED_COLUMNS:
            if col not in names:
                raise AssertionError(f"missing required column: {col}")
        module_key = "Module" if "Module" in names else ""
        unit_key = "Unit" if "Unit" in names else ""
        rows: list[RegistryRow] = []
        for raw in reader:
            name = (raw.get("Parameter") or "").strip()
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


def test_second_get_parameter_does_not_reopen(monkeypatch, tmp_path: Path) -> None:
    path = _write_registry(tmp_path / "registry.csv", "first")
    opens: list[Path] = []
    original_open = Path.open

    def counting_open(self: Path, *args: object, **kwargs: object) -> object:
        opens.append(self)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    assert get_parameter("Demo param", path).value == "first"
    assert len(opens) == 1
    assert get_parameter("Demo param", path).value == "first"
    assert len(opens) == 1


def test_rewritten_csv_is_picked_up(tmp_path: Path) -> None:
    path = _write_registry(tmp_path / "registry.csv", "old")
    assert get_parameter("Demo param", path).value == "old"
    _write_registry(path, "rewritten-value")
    assert get_parameter("Demo param", path).value == "rewritten-value"


def test_mutating_load_registry_list_does_not_affect_later_calls(
    tmp_path: Path,
) -> None:
    path = _write_registry(tmp_path / "registry.csv", "kept")
    rows = load_registry(path)
    original = list(rows)
    rows.clear()
    later = load_registry(path)
    assert later == original
    assert later is not rows


def test_cached_results_match_uncached_parse_of_real_registry() -> None:
    cached = load_registry()
    uncached = _uncached_parse(Path("params/registry.csv"))
    assert cached == uncached
    for row in uncached:
        assert get_parameter(row.parameter) == row
