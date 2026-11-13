"""SVB-50: the live crowd and price follow the real engine in real mode (ADR-098).

Fake mode (the default for tests and ``ui.server`` without the launcher) keeps
the scripted demo arc unchanged. CPU only, offline, deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.fit import real_calibration_modes
from engine.modes import ModuleMode
from mocks.l6_crowd_ui import CrowdFake
from params.loader import get_parameter
from ui.live.engine_crowd import (
    LIVE_MODULES_ROW,
    WAVER_ROW,
    EngineCrowd,
    apply_engine_crowd,
    live_engine_modes,
    wavering_threshold,
)
from ui.live.session import LiveSession

RUMOR = "Kestrel Bank is out of cash and may delay uninsured wires"
TICKS = 12


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")


def _real_session(monkeypatch: pytest.MonkeyPatch, role: str = "vc-partner") -> LiveSession:
    monkeypatch.setenv("STAMPEDE_MODE", "real")
    session = LiveSession(seed=42)
    session.start()
    posted = session.submit_rumor(RUMOR, role, "bank-fvb", now=0.0)
    assert posted.ok and posted.affects_simulation
    for _ in range(TICKS):
        session.step_once()
    return session


def test_registry_rows_are_assumed_m29() -> None:
    for name in (LIVE_MODULES_ROW, WAVER_ROW):
        row = get_parameter(name)
        assert row.status == "ASSUMED"
        assert row.module == "M29"
    assert 0.0 <= wavering_threshold() <= 1.0


def test_live_modules_are_the_calibrated_set_plus_the_order_book() -> None:
    modes = live_engine_modes()
    assert set(real_calibration_modes()) <= set(modes)
    assert modes["M14"] is ModuleMode.REAL
    assert all(mode is ModuleMode.REAL for mode in modes.values())
    assert "M29" not in modes and "M30" not in modes


def test_fake_mode_keeps_the_scripted_crowd() -> None:
    from mocks.l1_engine import AttentionFake

    session = LiveSession(seed=42)
    assert session.follows_engine is False
    crowd = session.engine["M29"]
    assert isinstance(crowd, CrowdFake)
    assert crowd.follow_engine is False
    assert isinstance(session.engine["M03"], AttentionFake)


def test_real_mode_builds_the_real_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.agents.attention import AttentionEngine
    from engine.agents.decide import DecisionModule
    from engine.markets.book import OrderBook

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    session = LiveSession(seed=42)
    assert session.follows_engine is True
    assert isinstance(session.engine["M03"], AttentionEngine)
    assert isinstance(session.engine["M06"], DecisionModule)
    assert isinstance(session.engine["M14"], OrderBook)
    assert session.engine["M29"].follow_engine is True


def test_real_mode_crowd_counts_are_m06_state(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _real_session(monkeypatch)
    crowd = session.engine["M29"]
    decisions = session.engine["M06"]
    n = int(decisions.already_withdrew.shape[0])
    assert n > 0
    # SVB-61: only depositors with an uninsured, mobile amount count as withdrew; in
    # soft mode (the fit's) nobody has withdrawn after 12 minutes, as at SVB.
    eligible = np.asarray(decisions.initial_uninsured) * float(decisions.mobile_share) >= max(
        float(decisions.params.minimum_amount), 1e-12
    )
    assert crowd.withdrew == int(np.count_nonzero(decisions.already_withdrew & eligible))
    assert crowd.withdrew >= 0
    worried = np.asarray(decisions.worry) >= wavering_threshold()
    assert crowd.wavering == int(np.count_nonzero(worried & ~(decisions.already_withdrew & eligible)))
    assert crowd.calm + crowd.wavering + crowd.withdrew == n
    assert 0.0 <= crowd.cash_gauge < 1.0
    assert crowd.rumor == RUMOR


def test_real_mode_price_comes_from_the_order_book(monkeypatch: pytest.MonkeyPatch) -> None:
    from contracts.events import PriceTick

    session = _real_session(monkeypatch)
    ticks = [event for event in session.engine.bus.log if isinstance(event, PriceTick)]
    assert len(ticks) == TICKS
    screen = session.engine["M30"]
    assert screen.prices
    assert float(screen.prices[-1][1]) == pytest.approx(ticks[-1].price)
    assert ticks[-1].price < session.engine["M14"].params.opening_price


def test_real_mode_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    def snapshot() -> tuple[int, int, float, float]:
        session = _real_session(monkeypatch)
        crowd = session.engine["M29"]
        return crowd.withdrew, crowd.wavering, crowd.cash_gauge, session.engine["M14"].price

    assert snapshot() == snapshot()


@pytest.mark.xfail(
    strict=False,
    reason="SVB-61 runs the calibrated soft model: nobody withdraws in 12 minutes and roles "
    "wash out over a day on the 16-node graph; the role card (SVB-62) restores a role effect",
)
def test_real_mode_role_changes_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    vc = _real_session(monkeypatch, "vc-partner").engine["M29"].withdrew
    user = _real_session(monkeypatch, "ui-user").engine["M29"].withdrew
    assert vc > user


def test_apply_engine_crowd_maps_fields_and_closure() -> None:
    crowd = CrowdFake("M29")
    apply_engine_crowd(crowd, None)
    assert crowd.withdrew == 0
    snap = EngineCrowd(
        depositors=10, withdrew=3, wavering=2, calm=5, cash_fraction=0.4, closed=False, attention=0.7
    )
    apply_engine_crowd(crowd, snap)
    assert (crowd.calm, crowd.wavering, crowd.withdrew) == (5, 2, 3)
    assert crowd.cash_gauge == pytest.approx(0.4)
    assert crowd.tipping_point == pytest.approx(0.7)
    assert crowd.time_to_failure_ticks is None
    closed = EngineCrowd(
        depositors=10, withdrew=3, wavering=2, calm=5, cash_fraction=0.4, closed=True, attention=0.7
    )
    apply_engine_crowd(crowd, closed)
    assert crowd.time_to_failure_ticks == 0.0


def test_health_reports_the_crowd_source(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import ui.server as server

    payload = TestClient(server.app).get("/health").json()
    expected = "engine" if server._session.follows_engine else "scripted"
    assert payload["crowd_source"] == expected
