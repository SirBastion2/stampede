"""SVB-53 (D4): optional stop of M07 settlement after BankClosed (flag off by default)."""

from __future__ import annotations

import pytest

import engine.bank.payments as payments
from contracts.events import BankClosed, PaymentSettled, Tick, WithdrawalRequested
from contracts.schemas import Scenario
from engine.bank.payments import PaymentsModule, stop_settlement_after_closure
from mocks.bus import EventBus
from params.loader import RegistryError, get_parameter

ROW = "M07 stop settlement after closure"


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_FULL", raising=False)


def _module(stop: bool | None = None) -> tuple[PaymentsModule, EventBus]:
    bus = EventBus()
    module = PaymentsModule("M07")
    module.attach(bus, Scenario(preset="tech-concentrated", seed=2))
    if stop is not None:
        module.stop_after_closure = stop
    return module, bus


def _request(bus: EventBus, t: float, amount: float = 5_000_000.0) -> None:
    bus.publish(
        WithdrawalRequested(
            t=t, seed=2, bank_id="bank-fvb", cause_id=f"w{t}", depositor_id=f"d{t}",
            amount=amount, channel="app",
        )
    )
    bus.publish(Tick(t=t, seed=2, bank_id=None, cause_id=None))


def _close(bus: EventBus, t: float, bank_id: str = "bank-fvb") -> None:
    bus.publish(BankClosed(t=t, seed=2, bank_id=bank_id, cause_id="m20", reason="regulator"))


def _settled(events) -> float:
    return sum(e.amount for e in events if isinstance(e, PaymentSettled))


def test_row_is_assumed_and_off() -> None:
    row = get_parameter(ROW)
    assert row.status == "ASSUMED" and row.module == "M07"
    assert row.value == "false"
    assert stop_settlement_after_closure() is False
    assert PaymentsModule("M07").stop_after_closure is False


def test_flag_off_keeps_settling_after_closure() -> None:
    module, bus = _module()
    _request(bus, 0.0)
    assert _settled(module.step(0.0)) > 0.0
    _close(bus, 1.0)
    _request(bus, 1.0)
    assert _settled(module.step(1.0)) > 0.0
    assert module.closed is False


def test_flag_on_stops_after_this_banks_closure() -> None:
    module, bus = _module(stop=True)
    _request(bus, 0.0)
    assert _settled(module.step(0.0)) > 0.0
    _close(bus, 1.0)
    _request(bus, 1.0)
    assert module.step(1.0) == []
    assert module.closed is True
    _request(bus, 2.0)
    assert module.step(2.0) == []


def test_flag_on_ignores_another_banks_closure() -> None:
    module, bus = _module(stop=True)
    _close(bus, 0.0, bank_id="bank-other")
    _request(bus, 0.0)
    assert _settled(module.step(0.0)) > 0.0
    assert module.closed is False


def test_flag_off_and_on_agree_before_any_closure() -> None:
    off, bus_off = _module()
    on, bus_on = _module(stop=True)
    for t in range(4):
        _request(bus_off, float(t)); _request(bus_on, float(t))
        a, b = off.step(float(t)), on.step(float(t))
        assert [(e.channel, e.amount) for e in a] == [(e.channel, e.amount) for e in b]


def test_registry_true_turns_it_on(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    import params.loader as loader

    real = loader.get_parameter

    def patched(name, path=None):
        row = real(name, path)
        return dataclasses.replace(row, value="true") if name == ROW else row

    monkeypatch.setattr(loader, "get_parameter", patched)
    assert stop_settlement_after_closure() is True


@pytest.mark.parametrize("env", ["STAMPEDE_CALIB_GPU_SIM", "STAMPEDE_CALIB_GPU_FULL"])
def test_flag_on_refuses_gpu_calibration(monkeypatch: pytest.MonkeyPatch, env: str) -> None:
    monkeypatch.setenv(env, "1")
    monkeypatch.setattr(payments, "stop_settlement_after_closure", lambda: True)
    with pytest.raises(RegistryError, match="gpu"):
        PaymentsModule("M07")
