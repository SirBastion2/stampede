"""Offline E2E: live rumor path through M34 → M29 → L1 panic loop (CPU, fixed seed)."""

from __future__ import annotations

import os

import pytest

from agents.guardrails import moderate_rumor
from mocks.l6_crowd_ui import CrowdFake
from mocks.l7_market_ui import MarketScreenFake
from ui.live.session import LiveSession


def _snapshot(session: LiveSession) -> tuple[int, float, float]:
    crowd = session.engine["M29"]
    screen = session.engine["M30"]
    assert isinstance(crowd, CrowdFake)
    assert isinstance(screen, MarketScreenFake)
    last_price = float(screen.prices[-1][1]) if screen.prices else 1.0
    return crowd.withdrew, crowd.cash_gauge, last_price


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def test_real_bank_name_rejected_by_moderation() -> None:
    result = moderate_rumor("Silicon Valley Bank is insolvent")
    assert not result.allowed
    assert any("blocked bank name" in reason for reason in result.reasons)


def test_live_rumor_raises_withdrawals_cash_drain_and_price_vs_baseline() -> None:
    steps = 24
    rumor_text = (
        "Treasury desk hears Fictional Valley Bank may delay uninsured wires this week"
    )
    baseline = LiveSession(seed=42)
    baseline.start()
    for _ in range(steps):
        baseline.step_once()
    base_withdrew, base_cash, base_price = _snapshot(baseline)

    session = LiveSession(seed=42)
    session.start()
    posted = session.submit_rumor(rumor_text, "treasury-desk", "bank-fvb")
    assert posted.ok
    for _ in range(steps - 1):
        session.step_once()
    rumor_withdrew, rumor_cash, rumor_price = _snapshot(session)

    assert rumor_withdrew > base_withdrew
    assert rumor_cash < base_cash
    assert rumor_price < base_price


def test_health_reports_playback_horizon() -> None:
    os.environ.setdefault("STAMPEDE_OFFLINE", "1")
    from fastapi.testclient import TestClient

    from ui.live.pacing import crisis_sim_minutes
    from ui.server import app

    client = TestClient(app)
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["crisis_sim_minutes"] == crisis_sim_minutes()
    assert payload["offline"] is True
    assert payload["demo_population"] >= 1


def test_live_server_blocks_chase_with_reason() -> None:
    os.environ.setdefault("STAMPEDE_OFFLINE", "1")
    from fastapi.testclient import TestClient

    from ui.server import app

    client = TestClient(app)
    response = client.post(
        "/api/rumor",
        json={"text": "Chase customers should pull their money now", "poster": "ui-user"},
    )
    payload = response.json()
    assert payload["ok"] is False
    assert any("chase" in reason for reason in payload["reasons"])


def test_live_server_rumor_api_rejects_real_bank() -> None:
    os.environ.setdefault("STAMPEDE_OFFLINE", "1")
    from fastapi.testclient import TestClient

    from ui.server import app

    client = TestClient(app)
    response = client.post(
        "/api/rumor",
        json={
            "text": "Signature Bank customers should withdraw now",
            "poster": "ui-user",
            "bank_id": "bank-fvb",
        },
    )
    payload = response.json()
    assert payload["ok"] is False
    assert payload["reasons"]
