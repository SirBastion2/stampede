"""Offline: default fake pitch shows a slow withdrawal arc for a strong rumor."""

from __future__ import annotations

import os

import pytest

from mocks.l6_crowd_ui import CrowdFake
from mocks.l7_market_ui import MarketScreenFake
from ui.live.pacing import crisis_sim_minutes
from ui.live.session import LiveSession

_RUMOR = "Treasury desk hears Fictional Valley Bank may delay uninsured wires this week"


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    os.environ["STAMPEDE_OFFLINE"] = "1"


def _series(poster: str, seed: int = 7) -> list[tuple[float, float, float, float]]:
    """Processed tick, withdrawn fraction, cash gauge, price."""
    session = LiveSession(seed=seed)
    posted = session.submit_rumor(_RUMOR, poster, "bank-fvb")
    assert posted.ok
    horizon = int(crisis_sim_minutes())
    rows: list[tuple[float, float, float, float]] = []
    for _ in range(horizon):
        processed = session.tick
        session.step_once()
        crowd = session.engine["M29"]
        screen = session.engine["M30"]
        assert isinstance(crowd, CrowdFake)
        assert isinstance(screen, MarketScreenFake)
        total = crowd.calm + crowd.wavering + crowd.withdrew
        fraction = crowd.withdrew / total if total else 0.0
        price = float(screen.prices[-1][1]) if screen.prices else 0.0
        rows.append((processed, fraction, crowd.cash_gauge, price))
    return rows


def _at(
    rows: list[tuple[float, float, float, float]], mark: float
) -> tuple[float, float, float, float]:
    horizon = crisis_sim_minutes()
    target = mark * horizon
    return min(rows, key=lambda row: abs(row[0] - target))


def test_strong_rumor_arc_is_slow_s_curve_and_weak_rumor_stays_smaller() -> None:
    horizon = crisis_sim_minutes()
    strong = _series("vc-partner", seed=7)
    again = _series("vc-partner", seed=7)
    weak = _series("ui-user", seed=7)
    assert strong == again

    fractions = [row[1] for row in strong]
    assert fractions == sorted(fractions)
    early = _at(strong, 0.10)
    mid = _at(strong, 0.50)
    late = _at(strong, 0.90)
    drained = _at(strong, 0.70)
    assert early[0] / horizon == pytest.approx(0.10, abs=0.02)
    assert early[1] < 0.05
    assert late[1] > 0.25
    assert mid[2] > 0.0
    assert all(row[2] > 0.0 for row in strong if row[0] <= 0.50 * horizon)
    assert drained[2] == pytest.approx(0.0, abs=1e-9)
    assert late[3] < early[3]

    weak_late = _at(weak, 0.90)
    assert weak_late[1] < late[1] * 0.6
