"""M00 clock, calendar, and event scheduler (Wave B)."""

from __future__ import annotations

import heapq
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from contracts.events import Tick
from contracts.schemas import Scenario
from params.loader import RegistryError, get_parameter

MINUTES_PER_DAY = 24 * 60
MINUTES_PER_WEEK = 7 * MINUTES_PER_DAY


@dataclass(frozen=True, slots=True)
class CalendarConfig:
    """Business calendar loaded from the parameter registry (§8.1, M00)."""

    business_open_minute: int
    business_close_minute: int
    wire_cutoff_minute: int
    facility_open_minute: int
    facility_close_minute: int
    epoch_open_minute: int


@dataclass(frozen=True, slots=True)
class CalendarFlags:
    business_hours: bool
    wire_window: bool
    facility_open: bool
    weekend: bool
    minute_of_day: int
    day_index: int


def _parse_registry_minutes(value: str) -> int:
    """Parse registry cells like ``9``, ``17``, or ``4 hours`` into minute-of-day."""
    text = value.strip().lower().replace(",", "")
    if "hour" in text:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
        if match is None:
            raise RegistryError(f"cannot parse hours from {value!r}")
        return int(float(match.group(1)) * 60)
    if ":" in text:
        hours_text, minutes_text = text.split(":", 1)
        return int(hours_text.strip()) * 60 + int(minutes_text.strip())
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if match is None:
        raise RegistryError(f"cannot parse minute value from {value!r}")
    number = float(match.group(1))
    if number <= 24 and "." not in match.group(1):
        return int(number * 60)
    return int(number)


def load_calendar_config() -> CalendarConfig:
    """Load M00 calendar parameters via ``params.loader``."""
    open_minute = _parse_registry_minutes(get_parameter("Calendar business open hour").value)
    close_minute = _parse_registry_minutes(get_parameter("Calendar business close hour").value)
    wire_cutoff = _parse_registry_minutes(get_parameter("Calendar wire cutoff hour").value)
    facility_open = _parse_registry_minutes(get_parameter("Calendar facility open hour").value)
    facility_close = _parse_registry_minutes(get_parameter("Calendar facility close hour").value)
    epoch_open = _parse_registry_minutes(get_parameter("Calendar sim epoch minute").value)
    return CalendarConfig(
        business_open_minute=open_minute,
        business_close_minute=close_minute,
        wire_cutoff_minute=wire_cutoff,
        facility_open_minute=facility_open,
        facility_close_minute=facility_close,
        epoch_open_minute=epoch_open,
    )


def _split_absolute(absolute: int) -> tuple[int, int]:
    day = absolute // MINUTES_PER_DAY
    minute = absolute % MINUTES_PER_DAY
    return day, minute


def calendar_flags(sim_t: float, cfg: CalendarConfig) -> CalendarFlags:
    """Return calendar flags at simulated minute ``sim_t`` (minutes from Monday 00:00)."""
    day, minute = _split_absolute(int(sim_t))
    weekend = (day % 7) >= 5
    business = (
        not weekend
        and cfg.business_open_minute <= minute < cfg.business_close_minute
    )
    wire = (
        business
        and minute < cfg.wire_cutoff_minute
    )
    facility = (
        not weekend
        and cfg.facility_open_minute <= minute < cfg.facility_close_minute
    )
    return CalendarFlags(
        business_hours=business,
        wire_window=wire,
        facility_open=facility,
        weekend=weekend,
        minute_of_day=minute,
        day_index=day,
    )


def _start_of_day_open(day: int, cfg: CalendarConfig) -> int:
    return day * MINUTES_PER_DAY + cfg.business_open_minute


def _next_weekday_open(after: int, cfg: CalendarConfig) -> int:
    day, _ = _split_absolute(after)
    candidate_day = day
    while True:
        weekday = candidate_day % 7
        if weekday < 5:
            open_at = _start_of_day_open(candidate_day, cfg)
            if open_at > after:
                return open_at
        candidate_day += 1


def next_business_minute(sim_t: float, cfg: CalendarConfig) -> float:
    """Advance strictly past ``sim_t`` to the next minute counted as a business tick."""
    cursor = int(sim_t) + 1
    while True:
        day, minute = _split_absolute(cursor)
        if (day % 7) >= 5:
            cursor = _next_weekday_open(cursor, cfg)
            continue
        if minute < cfg.business_open_minute:
            cursor = _start_of_day_open(day, cfg)
            if cursor <= sim_t:
                cursor += 1
            continue
        if minute >= cfg.business_close_minute:
            cursor = _next_weekday_open(cursor, cfg)
            continue
        return float(cursor)


@dataclass(order=True)
class _QueueEntry:
    fire_t: float
    phase: int
    seq: int
    event: BaseModel = field(compare=False)


class Clock:
    """M00 real module: calendar, heap scheduler, and ``Tick`` publisher."""

    module_id: str = "M00"

    def __init__(self, config: CalendarConfig | None = None) -> None:
        self.config = config or load_calendar_config()
        self.scenario: Scenario | None = None
        self._sim_t: float | None = None
        self._seq = 0
        self._heap: list[_QueueEntry] = []
        self.flags = CalendarFlags(
            business_hours=False,
            wire_window=False,
            facility_open=False,
            weekend=False,
            minute_of_day=0,
            day_index=0,
        )

    def attach(self, _bus: Any, scenario: Scenario) -> None:
        self.scenario = scenario

    @property
    def seed(self) -> int:
        if self.scenario is None:
            raise RuntimeError("M00 clock is not attached to a scenario")
        return self.scenario.seed

    def schedule(self, event: BaseModel, fire_t: float, phase: int = 0) -> None:
        """Enqueue a scheduled bus event (ordered by time, phase, sequence)."""
        entry = _QueueEntry(fire_t=fire_t, phase=phase, seq=self._seq, event=event)
        self._seq += 1
        heapq.heappush(self._heap, entry)

    def _drain_until(self, until_t: float) -> list[BaseModel]:
        out: list[BaseModel] = []
        while self._heap and self._heap[0].fire_t <= until_t:
            entry = heapq.heappop(self._heap)
            out.append(entry.event)
        return out

    def step(self, t: float) -> list[BaseModel]:
        """One engine step: advance sim time, fire scheduled events, publish ``Tick``."""
        del t  # real clock advances simulated minutes, not the engine index
        if self.scenario is None:
            return []
        if self._sim_t is None:
            self._sim_t = float(self.config.epoch_open_minute)
        else:
            self._sim_t = next_business_minute(self._sim_t, self.config)
        self.flags = calendar_flags(self._sim_t, self.config)
        scheduled = self._drain_until(self._sim_t)
        tick = Tick(
            t=self._sim_t,
            seed=self.seed,
            bank_id=None,
            cause_id=None,
        )
        return scheduled + [tick]

    def run_until(
        self,
        until: float,
        modules: Mapping[str, Any],
        order: Sequence[str],
        publish: Any,
    ) -> list[BaseModel]:
        """Advance simulated time to ``until`` and call each module's ``step`` in order."""
        log: list[BaseModel] = []
        step_index = 0
        while self._sim_t is None or self._sim_t < until:
            for module_id in order:
                for event in modules[module_id].step(float(step_index)):
                    publish(event)
                    log.append(event)
            step_index += 1
            if self._sim_t is not None and self._sim_t >= until:
                break
        return log


def run_until(
    until: float,
    modules: Mapping[str, Any],
    order: Sequence[str],
    publish: Any,
    clock: Clock | None = None,
) -> list[BaseModel]:
    """§19 scheduler entry: ``run(until)`` with each module's ``step(t)``."""
    scheduler = clock or Clock()
    return scheduler.run_until(until, modules, order, publish)
