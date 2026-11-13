"""M28 scenario builder: presets, controls, and pydantic Scenario assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contracts.schemas import Bank, Rumor, Scenario, ScenarioShock
from params.loader import get_parameter
from scenarios.presets import bank_for_preset, list_preset_keys, normalize_preset_key

_PARAM_POSTER = "M28 default rumor poster node"
_PARAM_TECH_RATE_PATH = "M28 tech-concentrated default rate path knots"


def _parse_rate_path_cell(value: str) -> list[tuple[float, float]]:
    """Parse ``t:r;t:r`` registry knots into a piecewise-linear path."""
    text = value.strip()
    if not text or text.lower() in {"—", "-", "none", "default"}:
        return []
    knots: list[tuple[float, float]] = []
    for chunk in text.split(";"):
        piece = chunk.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"invalid rate path knot: {piece!r}")
        left, right = piece.split(":", 1)
        knots.append((float(left.strip()), float(right.strip())))
    knots.sort(key=lambda pair: pair[0])
    return knots


@dataclass(frozen=True, slots=True)
class ScenarioControls:
    """User-facing scenario knobs (§5 M28)."""

    preset: str
    seed: int | None = None
    bank_id: str | None = None
    rumors: list[Rumor] | None = None
    rumor_poster: str | None = None
    rate_path: list[tuple[float, float]] | None = None
    shocks: list[ScenarioShock] | None = None
    policies: list[dict[str, Any]] | None = None
    start_minute_of_day: int | None = None
    start_weekday: int | None = None
    fidelity: str = "demo"


def list_presets() -> list[dict[str, str]]:
    """Preset menu entries for UI and tests."""
    rows: list[dict[str, str]] = []
    for key in list_preset_keys():
        bank = bank_for_preset(key)
        rows.append(
            {
                "key": key,
                "bank_id": bank.id,
                "name": bank.name,
                "ticker": bank.ticker,
                "description": get_parameter(
                    {
                        "tech-concentrated": "Preset: tech-concentrated bank (SVB-like)",
                        "crypto-heavy": "Preset: crypto-heavy follower bank (Signature-like)",
                        "insured-community": "Preset: insured community bank",
                    }[key]
                ).value,
            }
        )
    return rows


def default_rate_path(preset_key: str) -> list[tuple[float, float]]:
    canonical = normalize_preset_key(preset_key)
    if canonical == "tech-concentrated":
        row = get_parameter(_PARAM_TECH_RATE_PATH)
        parsed = _parse_rate_path_cell(row.value)
        if parsed:
            return parsed
    return []


def default_rumor_poster() -> str:
    return get_parameter(_PARAM_POSTER).value.strip()


def build_scenario(controls: ScenarioControls, *, bank: Bank | None = None) -> Scenario:
    """Assemble a ``Scenario`` from controls and preset defaults."""
    canonical = normalize_preset_key(controls.preset)
    if canonical not in list_preset_keys():
        raise KeyError(f"unknown preset: {controls.preset!r}")
    resolved_bank = bank or bank_for_preset(canonical)
    if controls.bank_id is not None and controls.bank_id != resolved_bank.id:
        raise ValueError(
            f"bank_id {controls.bank_id!r} does not match preset bank {resolved_bank.id!r}"
        )
    rate_path = controls.rate_path if controls.rate_path is not None else default_rate_path(
        canonical
    )
    shocks = list(controls.shocks or [])
    if controls.start_minute_of_day is not None or controls.start_weekday is not None:
        params: dict[str, Any] = {}
        if controls.start_minute_of_day is not None:
            params["minute_of_day"] = controls.start_minute_of_day
        if controls.start_weekday is not None:
            params["weekday"] = controls.start_weekday
        shocks.append(ScenarioShock(kind="calendar_start", at=0.0, params=params))
    if controls.seed is None:
        raise ValueError("ScenarioControls.seed is required to build a scenario")
    return Scenario(
        preset=canonical,
        seed=controls.seed,
        rate_path=rate_path,
        shocks=shocks,
        rumors=list(controls.rumors or []),
        policies=list(controls.policies or []),
        fidelity=controls.fidelity,
    )


def poster_for_rumor(rumor: Rumor, *, default: str | None = None) -> str:
    """Resolve rumor poster node id from notes or the registry default."""
    for note in rumor.notes:
        if note.startswith("poster:"):
            return note.split(":", 1)[1].strip()
    return default or default_rumor_poster()
