"""Wall-clock pacing: sim minutes per real second without changing tick outcomes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from params.loader import RegistryError, get_parameter

_SPEED_STEPS = (1.0, 2.0, 4.0, 8.0)


def _parse_number(name: str, default: float) -> float:
    try:
        row = get_parameter(name)
    except RegistryError:
        return default
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", row.value)
    return float(match.group(1)) if match else default


def sim_tick_minutes() -> float:
    """One engine step is one simulated minute (``M03 sim tick minutes``)."""
    return max(1.0, _parse_number("M03 sim tick minutes", 1.0))


def crisis_sim_minutes() -> float:
    """Simulated minutes the live UI should span at 1× before outcome.

    SVB-61: in real mode the window follows SVB's timeline, the run over the
    trading day plus the overnight queue to the next open
    (``M29 live real crisis sim minutes``); fake mode keeps the 120-minute arc.
    """
    if _real_mode():
        return max(1.0, _parse_number("M29 live real crisis sim minutes", 1440.0))
    return max(1.0, _parse_number("M29 UI live crisis sim minutes", 120.0))


def _real_mode() -> bool:
    from engine.modes import ModuleMode
    from ui.live.config import default_module_mode

    return default_module_mode() is ModuleMode.REAL


def playback_target_seconds() -> float:
    """Target real seconds for a full crisis at 1× (90–120 s band)."""
    return max(1.0, _parse_number("M29 UI live playback target seconds", 105.0))


def playback_head_start_seconds() -> float:
    """P-A2: seconds of 1x playback the buffer holds before playback starts."""
    return max(0.0, _parse_number("M29 live playback head start seconds", 2.0))


def client_stream_fps() -> float:
    return max(1.0, _parse_number("M29 UI client stream fps", 30.0))


def base_sim_minutes_per_real_second() -> float:
    """1× speed: sim minutes advanced per one real second."""
    return crisis_sim_minutes() / playback_target_seconds()


def normalize_speed(speed: float) -> float:
    if speed in _SPEED_STEPS:
        return speed
    for step in _SPEED_STEPS:
        if abs(speed - step) < 1e-6:
            return step
    return max(1.0, min(8.0, speed))


@dataclass
class PacingClock:
    """Accumulates fractional ticks from wall time; speed scales sim rate only."""

    speed: float = 1.0
    _fractional_ticks: float = field(default=0.0, init=False)

    def reset(self) -> None:
        self._fractional_ticks = 0.0

    def set_speed(self, speed: float) -> None:
        self.speed = normalize_speed(speed)

    def advance(self, real_seconds: float) -> int:
        if real_seconds <= 0:
            return 0
        rate = base_sim_minutes_per_real_second() * self.speed
        self._fractional_ticks += real_seconds * rate
        whole = int(self._fractional_ticks)
        self._fractional_ticks -= float(whole)
        return whole

    def ticks_for_wall_second(self, speed: float | None = None) -> float:
        """Expected whole+ fractional ticks over exactly one real second (for tests)."""
        rate = base_sim_minutes_per_real_second() * (speed if speed is not None else self.speed)
        return rate


def speed_steps() -> tuple[float, ...]:
    return _SPEED_STEPS
