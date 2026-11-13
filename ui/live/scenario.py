"""Deterministic scenarios for the unified live server (no preset rumors)."""

from __future__ import annotations

from contracts.schemas import Scenario


def live_scenario(seed: int = 1) -> Scenario:
    """Tech-concentrated preset without M28 rumors or UI shocks (user drives the rumor box)."""
    return Scenario(
        preset="tech-concentrated",
        seed=seed,
        rate_path=[(0.0, 0.04), (3.0, 0.06)],
        rumors=[],
        policies=[],
        shocks=[],
    )
