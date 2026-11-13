"""M07 payments: capacity, weekend queue, determinism, contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from contracts import parse_event
from contracts.events import PaymentSettled, Tick, WithdrawalRequested
from contracts.schemas import Scenario
from engine.bank.payments import (
    PaymentsModule,
    PaymentState,
    enqueue_withdrawal,
    load_payment_params,
    settle_tick,
)
from engine.core.clock import calendar_flags, load_calendar_config
from engine.modes import ModuleMode
from engine.registry import build_engine, run_scenario
from mocks.l2_banks import PaymentsFake


_MULTIPLIER_ROW = "M07 capacity multiplier (FIT)"


def _pin_capacity_multiplier(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Serve the M07 rows with the capacity multiplier set to ``value`` (SVB-47).

    The row is FITTED (13.414 since svb37): at that capacity nothing queues, so tests of
    queueing mechanics pin the unit multiplier instead of the persisted fit.
    """
    from params.loader import list_by_module

    def _rows(module: str, path=None):
        return [
            replace(row, value=value) if row.parameter == _MULTIPLIER_ROW else row
            for row in list_by_module(module, path)
        ]

    monkeypatch.setattr("engine.bank.payments.list_by_module", _rows)


def _scenario(seed: int = 7) -> Scenario:
    return Scenario(preset="tech-concentrated", seed=seed, rate_path=[(0.0, 0.04)])


def test_load_payment_params_from_registry() -> None:
    limits = load_payment_params()
    assert limits.app_daily_limit > 0.0
    assert limits.wire_capacity_per_tick > 0.0
    assert limits.branch_capacity_per_tick > 0.0


def test_capacity_multiplier_scales_app_wire_ach(monkeypatch) -> None:
    from params.loader import get_parameter

    assert float(get_parameter(_MULTIPLIER_ROW).value) > 0.0
    _pin_capacity_multiplier(monkeypatch, "1.0")
    base = load_payment_params()
    _pin_capacity_multiplier(monkeypatch, "2.0")
    twice = load_payment_params()
    assert twice.app_daily_limit == pytest.approx(2.0 * base.app_daily_limit)
    assert twice.wire_capacity_per_tick == pytest.approx(2.0 * base.wire_capacity_per_tick)
    assert twice.ach_batch_capacity == pytest.approx(2.0 * base.ach_batch_capacity)
    assert twice.branch_capacity_per_tick == pytest.approx(base.branch_capacity_per_tick)


def test_wire_capacity_limits_per_tick() -> None:
    cfg = load_calendar_config()
    limits = load_payment_params(calendar=cfg)
    flags = calendar_flags(float(cfg.epoch_open_minute), cfg)
    state = PaymentState()
    state, _ = enqueue_withdrawal(state, limits.wire_capacity_per_tick * 3, "wire", limits, flags)
    assert len(state.wire_queue) == 1
    settled = settle_tick(state, flags, limits)
    assert settled.get("wire", 0.0) == pytest.approx(limits.wire_capacity_per_tick)
    assert len(state.wire_queue) == 1
    settled2 = settle_tick(state, flags, limits)
    assert settled2.get("wire", 0.0) == pytest.approx(limits.wire_capacity_per_tick)


def test_queued_wires_survive_weekend(monkeypatch) -> None:
    _pin_capacity_multiplier(monkeypatch, "1.0")
    cfg = load_calendar_config()
    limits = load_payment_params(calendar=cfg)
    # Friday 16:30 absolute minutes (day 4, minute 16*60)
    friday = 4 * 24 * 60 + 16 * 60 + 30
    flags = calendar_flags(float(friday), cfg)
    state = PaymentState()
    state, _ = enqueue_withdrawal(state, 50_000_000.0, "wire", limits, flags)
    settle_tick(state, flags, limits)
    assert state.wire_queue
    # Saturday noon
    saturday = 5 * 24 * 60 + 12 * 60
    sat_flags = calendar_flags(float(saturday), cfg)
    assert not sat_flags.wire_window
    weekend_settled = settle_tick(state, sat_flags, limits)
    assert weekend_settled.get("wire", 0.0) == 0.0
    assert state.wire_queue
    # Monday 10:00
    monday = 7 * 24 * 60 + 10 * 60
    mon_flags = calendar_flags(float(monday), cfg)
    mon_settled = settle_tick(state, mon_flags, limits)
    assert mon_settled.get("wire", 0.0) > 0.0


def test_app_daily_limit_splits_to_wire() -> None:
    cfg = load_calendar_config()
    limits = load_payment_params(calendar=cfg)
    flags = calendar_flags(float(cfg.epoch_open_minute), cfg)
    state = PaymentState()
    amount = limits.app_daily_limit + 10_000_000.0
    state, instant = enqueue_withdrawal(state, amount, "app", limits, flags)
    assert instant == [("app", limits.app_daily_limit)]
    assert len(state.wire_queue) == 1
    assert state.wire_queue[0] == pytest.approx(10_000_000.0)


def test_payments_module_determinism() -> None:
    modes = {"M07": ModuleMode.REAL}
    a = run_scenario(_scenario(3), 20, modes=modes)
    b = run_scenario(_scenario(3), 20, modes=modes)
    assert [e.model_dump(mode="json", by_alias=True) for e in a] == [
        e.model_dump(mode="json", by_alias=True) for e in b
    ]


def _published_m07(engine: Any) -> list[dict[str, Any]]:
    module = engine["M07"]
    return [
        event.model_dump(mode="json", by_alias=True) for event in module.published
    ]


def test_real_m07_events_parse() -> None:
    engine = build_engine(modes={"M07": ModuleMode.REAL}, scenario=_scenario(1))
    engine.run(15)
    module = engine["M07"]
    assert isinstance(module, PaymentsModule)
    for event in module.published:
        payload = event.model_dump(mode="json", by_alias=True)
        parsed = parse_event(payload)
        assert parsed.model_dump(mode="json", by_alias=True) == payload


def test_real_vs_fake_m07_contract_shape() -> None:
    steps = 18
    seed = 9
    scenario = _scenario(seed)
    fake_engine = build_engine(scenario=scenario)
    fake_engine.run(steps)
    real_engine = build_engine(modes={"M07": ModuleMode.REAL}, scenario=scenario)
    real_engine.run(steps)
    assert isinstance(fake_engine["M07"], PaymentsFake)
    assert isinstance(real_engine["M07"], PaymentsModule)
    fake_pub = _published_m07(fake_engine)
    real_pub = _published_m07(real_engine)
    for payload in fake_pub + real_pub:
        parse_event(payload)
        assert payload["event_type"] == "PaymentSettled"
        assert payload["amount"] > 0.0
        assert isinstance(payload["queued_remaining"], int)
        assert payload["channel"]


def test_direct_module_wire_queue_count(monkeypatch) -> None:
    _pin_capacity_multiplier(monkeypatch, "1.0")
    bus = __import__("mocks.bus", fromlist=["EventBus"]).EventBus()
    module = PaymentsModule("M07")
    module.attach(bus, _scenario(2))
    bus.publish(
        WithdrawalRequested(
            t=0.0,
            seed=2,
            bank_id="bank-fvb",
            cause_id="w1",
            depositor_id="d1",
            amount=100_000_000.0,
            channel="wire",
        )
    )
    bus.publish(Tick(t=0.0, seed=2, bank_id=None, cause_id=None))
    events = module.step(0.0)
    assert events
    assert all(isinstance(e, PaymentSettled) for e in events)
    assert module.state.wire_queue
