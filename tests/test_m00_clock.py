"""Unit tests for M00 real clock (Wave B)."""

from __future__ import annotations

import time

from contracts import parse_event
from contracts.events import BankClosed, Tick
from contracts.schemas import Scenario
from engine.core.clock import Clock, calendar_flags, load_calendar_config, next_business_minute
from engine.modes import ModuleMode
from engine.registry import build_engine
from mocks.l1_engine import ClockFake


def _scenario(seed: int = 7) -> Scenario:
    return Scenario(preset="tech-concentrated", seed=seed)


def test_calendar_loads_from_registry() -> None:
    cfg = load_calendar_config()
    assert cfg.business_open_minute == 9 * 60
    assert cfg.business_close_minute == 17 * 60
    assert cfg.epoch_open_minute == 9 * 60


def test_determinism_same_seed_same_ticks() -> None:
    scenario = _scenario(99)
    a = Clock()
    b = Clock()
    a.attach(None, scenario)
    b.attach(None, scenario)
    ticks_a: list[Tick] = []
    ticks_b: list[Tick] = []
    for index in range(12):
        ticks_a.extend(event for event in a.step(float(index)) if isinstance(event, Tick))
        ticks_b.extend(event for event in b.step(float(index)) if isinstance(event, Tick))
    assert [tick.t for tick in ticks_a] == [tick.t for tick in ticks_b]
    assert [tick.seed for tick in ticks_a] == [99] * len(ticks_a)


def test_heap_tie_ordering_stable() -> None:
    clock = Clock()
    clock.attach(None, _scenario(1))
    fire_t = float(clock.config.epoch_open_minute + 1)
    first = BankClosed(
        t=fire_t,
        seed=1,
        bank_id="bank-a",
        cause_id="a",
        reason="first",
    )
    second = BankClosed(
        t=fire_t,
        seed=1,
        bank_id="bank-b",
        cause_id="b",
        reason="second",
    )
    clock.schedule(second, fire_t, phase=1)
    clock.schedule(first, fire_t, phase=0)
    clock.step(0.0)
    events = clock.step(1.0)
    closed = [event for event in events if isinstance(event, BankClosed)]
    assert [event.reason for event in closed] == ["first", "second"]


def test_weekend_scheduled_events_still_fire() -> None:
    clock = Clock()
    clock.attach(None, _scenario(2))
    cfg = clock.config
    saturday_noon = 5 * 24 * 60 + 12 * 60
    weekend_event = BankClosed(
        t=float(saturday_noon),
        seed=2,
        bank_id="bank-weekend",
        cause_id=None,
        reason="weekend-resolution",
    )
    clock.schedule(weekend_event, float(saturday_noon), phase=0)
    friday_close = 4 * 24 * 60 + cfg.business_close_minute - 1
    clock._sim_t = float(friday_close)  # noqa: SLF001 — test positions before the weekend jump
    events = clock.step(0.0)
    reasons = [event.reason for event in events if isinstance(event, BankClosed)]
    assert "weekend-resolution" in reasons
    assert clock.flags.weekend is False


def test_real_vs_fake_tick_contract() -> None:
    scenario = _scenario(11)
    fake_engine = build_engine(scenario=scenario, modes={"M00": ModuleMode.FAKE})
    real_engine = build_engine(scenario=scenario, modes={"M00": ModuleMode.REAL})
    fake_engine.run(6)
    real_engine.run(6)
    fake_clock = fake_engine["M00"]
    real_clock = real_engine["M00"]
    assert isinstance(fake_clock, ClockFake)
    assert isinstance(real_clock, Clock)
    for log in (fake_engine.bus.log, real_engine.bus.log):
        for event in log:
            if event.event_type != "Tick":
                continue
            parsed = parse_event(event.model_dump())
            assert parsed.event_type == "Tick"
    assert fake_clock.business_hours is True
    assert real_clock.flags.business_hours is True


def test_skips_overnight_between_business_days() -> None:
    cfg = load_calendar_config()
    friday_last = 4 * 24 * 60 + cfg.business_close_minute - 1
    monday_next = next_business_minute(float(friday_last), cfg)
    assert calendar_flags(monday_next, cfg).day_index == 7
    assert calendar_flags(monday_next, cfg).business_hours is True


def test_empty_business_week_under_one_second() -> None:
    clock = Clock()
    clock.attach(None, _scenario(0))
    start = time.perf_counter()
    for index in range(5 * 8 * 60):
        clock.step(float(index))
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0


def test_m00_real_mode_wires_in_registry() -> None:
    engine = build_engine(scenario=_scenario(), modes={"M00": ModuleMode.REAL})
    assert isinstance(engine["M00"], Clock)
