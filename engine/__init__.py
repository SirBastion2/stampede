"""STAMPEDE simulation engine (Wave B+ implementation)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from engine.interfaces import SimModule
from engine.modes import ModuleMode

if TYPE_CHECKING:
    from contracts.schemas import Scenario
    from engine.registry import Engine

__all__ = ["ModuleMode", "SimModule", "build_engine"]


def build_engine(
    modes: Mapping[str, ModuleMode] | None = None,
    default_mode: ModuleMode = ModuleMode.FAKE,
    scenario: Scenario | None = None,
) -> Engine:
    """Compose modules on the bus (lazy import avoids registry ↔ mocks cycles)."""
    from engine.registry import build_engine as _build_engine

    return _build_engine(modes=modes, default_mode=default_mode, scenario=scenario)
