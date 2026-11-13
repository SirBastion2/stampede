"""UI-1: the screens' state endpoint and the wired four-screen page."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.guardrails import moderate_rumor

PAGE = Path("ui/live/stampede_ui.html")


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def _real(monkeypatch: pytest.MonkeyPatch):
    from ui.live.session import LiveSession

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    return LiveSession(seed=1)


def test_state_follows_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _real(monkeypatch)
    assert session.ui_state()["run"]["preset"] is None and session.ui_state()["timeline"] == []
    session.load_preset("lost-bonds")
    for _ in range(30):
        session.step_once()
    u = session.ui_state()
    assert u["tick"] == 30.0 and u["run"]["kind"] == "fitted" and u["price_count"] == 30
    assert u["timeline"][0] == {"t": 0.0, "type": "rumor", "poster": "vc-partner",
                                "text": "Copperleaf lost $1.8B on bonds and is raising capital"}
    assert u["money"]["deposits_usd"] > 0 and u["money"]["requested_usd"] >= 0 and u["money"]["cash_usd"] > 0
    assert sum(u["crowd"][k] for k in ("withdrew", "wavering", "calm")) == 100_000
    later = session.ui_state(price_since=25, event_since=1)
    assert len(later["prices"]) == 5 and later["timeline"] == [] and later["timeline_count"] == 1


def test_responses_and_the_halt_reach_the_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.bank.ledger import registry_deposits_total

    session = _real(monkeypatch)
    session.load_preset("lost-bonds")
    for _ in range(5):
        session.step_once()
    session.enact_response("withdrawal-limit")
    session.step_once()
    session._requested_usd = registry_deposits_total()  # the paced price is at its floor
    session.step_once()
    kinds = [(e["type"], e["t"], e["text"]) for e in session.ui_state()["timeline"]]
    assert ("response", 5.0, "Withdrawal limit") in kinds
    assert any(k[0] == "halt" for k in kinds) and session.ui_state()["halt_minute"] == 6.0


def test_buffered_state_is_the_screens_minute(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _real(monkeypatch)
    session.start_producer(thread=False)
    session.load_preset("lost-bonds")
    while session.produce_once():
        pass
    u = session.ui_state()
    assert u["tick"] == 0.0 and u["engine_tick"] > 20 and u["buffered"] is True
    assert u["timeline"] == [] and u["price_count"] == 0  # nothing shows before playback reaches it
    session.advance_wall(1.0)
    v = session.ui_state()
    assert 0 < v["tick"] < v["engine_tick"] and v["price_count"] == int(v["tick"])
    assert [e["type"] for e in v["timeline"]] == ["rumor"]


def test_the_page_is_wired_and_says_nothing_banned() -> None:
    html = PAGE.read_text(encoding="utf-8")
    for needle in ("/api/ui/state", "/api/presets", "/api/response", "/api/limits", "/api/batch", "?mock=1"):
        assert needle in html, needle
    assert "predict" not in html.lower()
    assert "Chase is insolvent" not in html
    for line in ("Customers hear the rumor and decide whether to withdraw.", "Pick a rumor to start the run"):  # the intro (UI-7 words)
        assert line in html and moderate_rumor(line).allowed


def test_server_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import ui.server as server

    client = TestClient(server.app)
    page = client.get("/ui")
    assert page.status_code == 200 and "STAMPEDE" in page.text and "/api/ui/state" in page.text
    state = client.get("/api/ui/state", params={"price_since": 0, "event_since": 0}).json()
    assert {"tick", "run", "crowd", "money", "prices", "timeline", "horizon"} <= set(state)
