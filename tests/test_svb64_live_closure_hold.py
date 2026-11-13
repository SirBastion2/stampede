"""SVB-64: the live regulator holds closure until the next morning (SVB timeline)."""

from __future__ import annotations

import pytest

from contracts.events import BankClosed
from mocks.l4_system import RegulatorFake
from params.loader import get_parameter

PANIC = "Copperleaf lost $1.8B on bonds and is raising capital"


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def test_row_and_default_attribute() -> None:
    row = get_parameter("M29 live earliest closure minute")
    assert row.value == "1440" and row.status == "ASSUMED"
    assert RegulatorFake("M20").earliest_closure_minute is None


def test_calibration_engine_does_not_hold_closure() -> None:
    from calibration.fit import _default_svb_scenario, real_calibration_modes
    from engine.modes import ModuleMode
    from engine.registry import build_engine

    engine = build_engine(modes=real_calibration_modes(), default_mode=ModuleMode.FAKE, scenario=_default_svb_scenario(11))
    assert engine["M20"].earliest_closure_minute is None


def _run(monkeypatch: pytest.MonkeyPatch, minutes: int, hold: float | None):
    from ui.live.session import LiveSession

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    session = LiveSession(seed=42)
    assert session.engine["M20"].earliest_closure_minute == 1440.0
    session.engine["M20"].earliest_closure_minute = hold
    session.start()
    session.engine["M29"].queue_live_rumor(claim=PANIC, source_node="vc-partner", bank_id="bank-fvb",
                                           rumor_id="r", credibility=0.9)
    for _ in range(minutes):
        session.step_once()
    closed = [e.t for e in session.engine.bus.log if isinstance(e, BankClosed)]
    return session, closed


def test_live_hold_keeps_the_bank_open_through_day_one(monkeypatch: pytest.MonkeyPatch) -> None:
    session, closed = _run(monkeypatch, 400, 1440.0)
    assert closed == []
    assert session.engine["M20"].held_reason is not None  # the rule would have closed it


def test_closure_happens_once_the_hold_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    session, closed = _run(monkeypatch, 400, 380.0)
    assert closed and min(closed) >= 380.0
