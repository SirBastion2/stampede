"""SVB-63: live price paced to SVB's day-one drop, labelled as not calibrated (ADR-108)."""

from __future__ import annotations

import pytest

from engine.bank.ledger import registry_deposits_total
from params.loader import get_parameter

ROW = "M29 live price paced to SVB day-one drop"


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def _real(monkeypatch: pytest.MonkeyPatch):
    from ui.live.session import LiveSession

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    session = LiveSession(seed=42)
    session.start()
    return session


def test_row_is_assumed_and_on_for_live() -> None:
    row = get_parameter(ROW)
    assert row.value == "true" and row.status == "ASSUMED" and "not a calibrated market model" in row.source


def test_fake_mode_keeps_the_screen_prices() -> None:
    from ui.live.session import LiveSession

    session = LiveSession(seed=42)
    session.start()
    for _ in range(3):
        session.step_once()
    frame = session.market_frame()
    assert session.price_paced is False
    assert list(frame.prices) == list(session.engine["M30"].prices)


def test_real_mode_shows_the_paced_series(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _real(monkeypatch)
    for _ in range(6):
        session.step_once()
    frame = session.market_frame()
    assert session.price_paced is True
    assert len(frame.prices) == 6
    assert [t for t, _ in frame.prices] == [float(i) for i in range(6)]
    opening = float(session.engine["M14"].params.opening_price)
    assert all(price <= opening + 1e-9 for _, price in frame.prices)


def test_formula_hits_svb_day_one_drop_at_svb_day_one_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _real(monkeypatch)
    session.step_once()
    r1 = float(get_parameter("Calibration E1 requested day-one target").value)
    ratio = float(get_parameter("Calibration E1 day-1 stock target").value)
    opening = float(session.engine["M14"].params.opening_price)
    session._requested_usd = r1 * registry_deposits_total()
    session._requested_cursor = len(session.engine.bus.log)
    session.tick += 1.0
    session._pace_price()
    assert session._paced_prices[-1][1] == pytest.approx(opening * ratio)
    session._requested_usd = 3 * r1 * registry_deposits_total()
    session._pace_price()
    assert session._paced_prices[-1][1] == pytest.approx(opening * ratio)


def test_paced_price_never_rises(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _real(monkeypatch)
    session.engine["M29"].queue_live_rumor(claim="Copperleaf lost $1.8B on bonds and is raising capital",
                                           source_node="vc-partner", bank_id="bank-fvb", rumor_id="r", credibility=0.9)
    for _ in range(30):
        session.step_once()
    prices = [p for _, p in session.market_frame().prices]
    assert all(b <= a + 1e-12 for a, b in zip(prices, prices[1:]))


def test_health_labels_the_price_source() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import ui.server as server

    payload = TestClient(server.app).get("/health").json()
    assert payload["price_source"] in {"scripted", "order book", "paced to SVB's day-one drop (not a calibrated market model)"}
