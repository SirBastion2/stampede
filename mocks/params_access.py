"""Parameter access for fakes (registry via ``params.loader``, §8.1)."""

from __future__ import annotations

import re

from params.loader import RegistryError, get_parameter

# Documented defaults only where registry Value is ``TBD`` (§8.1).
_TBD_DEFAULTS: dict[str, float] = {}


def get_param(name: str) -> float:
    """Return a numeric registry parameter for use in deterministic fakes."""
    try:
        row = get_parameter(name)
    except RegistryError as exc:
        raise KeyError(str(exc)) from exc
    if row.value.strip().upper() == "TBD":
        try:
            return _TBD_DEFAULTS[name]
        except KeyError:
            raise KeyError(f"parameter {name} is TBD with no documented default") from None
    parsed = _parse_human(row.value)
    if parsed is None:
        raise ValueError(f"cannot parse registry value for {name}: {row.value!r}")
    return parsed


def _parse_human(text: str) -> float | None:
    """Parse cells like '$175B', '>93%', '$250,000', '20% of deposits', '4 hours'."""
    lowered = text.strip().lower().replace(",", "").replace("$", "")
    match = re.match(r"[><=\s]*([0-9]*\.?[0-9]+)\s*([a-z%]*)", lowered)
    if match is None:
        return None
    number = float(match.group(1))
    unit = match.group(2)
    if unit.startswith("b"):
        return number * 1_000_000_000.0
    if unit.startswith("m"):
        return number * 1_000_000.0
    if unit.startswith("%") or "percent" in lowered:
        return number / 100.0
    if "hour" in lowered:
        return number
    if "per depositor" in lowered or "depositor" in lowered:
        return number
    return number
