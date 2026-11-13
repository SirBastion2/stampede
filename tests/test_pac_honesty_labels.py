"""PA-C: the fitted / scenario labels, the limits screen copy, and typed rumors off by default."""

from __future__ import annotations

import re

import pytest

from agents.guardrails import moderate_rumor
from params.loader import get_parameter
from ui.live.labels import FITTED_BADGE, FREE_TEXT_OFF, LIMITS, SCENARIO_BADGE, limits_screen, run_label

BANNED = re.compile(r"predict", re.IGNORECASE)


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def _all_copy() -> list[str]:
    return [FITTED_BADGE, SCENARIO_BADGE, FREE_TEXT_OFF, limits_screen()["title"]] + [t for pair in LIMITS for t in pair]


def test_copy_has_no_banned_words_or_real_bank_names() -> None:
    for text in _all_copy():
        assert not BANNED.search(text), text
        assert moderate_rumor(text).allowed, (text, moderate_rumor(text).reasons)
        assert "SVB" not in text and "Silicon Valley" not in text, text


def test_limits_screen_has_the_approved_sections() -> None:
    screen = limits_screen()
    assert [s["heading"] for s in screen["sections"]] == [
        "What this is", "What it is not", "What is fitted", "What is a scenario", "Known gaps"]
    assert screen["badges"] == {"fitted": FITTED_BADGE, "scenario": SCENARIO_BADGE}


def test_only_the_2023_replay_is_fitted() -> None:
    assert run_label("lost-bonds", "x", (), False)["kind"] == "fitted"
    assert run_label("lost-bonds", "x", (), False)["name"] == "2023 replay"
    assert run_label("lost-bonds", "x", ("guarantee",), False)["kind"] == "scenario"
    assert run_label("lost-bonds", "x", (), True)["kind"] == "scenario"  # a typed rumor joined it
    for preset in ("frozen-branches", "stadium", "lost-bonds+insured", None):
        assert run_label(preset, "x", (), False)["kind"] == "scenario"


def test_session_label_follows_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from ui.live.session import LiveSession

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    session = LiveSession(seed=1)
    assert session.run_label()["kind"] == "scenario"  # nothing loaded yet
    session.load_preset("lost-bonds")
    assert session.run_label() == {"kind": "fitted", "badge": FITTED_BADGE, "name": "2023 replay",
                                   "preset": "lost-bonds", "responses": []}
    session.enact_guarantee()
    assert session.run_label()["kind"] == "scenario" and session.run_label()["responses"] == ["guarantee"]
    session.reset()
    session.load_preset("lost-bonds+insured")
    assert session.run_label()["kind"] == "scenario"


def test_row_is_assumed_and_off() -> None:
    row = get_parameter("M29 live free-text rumors")
    assert row.value == "false" and row.status == "ASSUMED" and row.module == "M29"


def test_server_refuses_typed_rumors_unless_the_operator_allows(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import ui.server as server

    client = TestClient(server.app)
    off = client.post("/api/rumor", json={"text": "Copperleaf lost $1.8B on bonds", "poster": "ui-user"}).json()
    assert off == {"ok": False, "reasons": [FREE_TEXT_OFF]}
    blocked = client.post("/api/rumor", json={"text": "Chase customers should pull their money now"}).json()
    assert blocked["ok"] is False and any("chase" in r for r in blocked["reasons"])
    assert len(client.get("/api/limits").json()["sections"]) == 5
    assert client.get("/api/labels").json()["run"]["kind"] in {"fitted", "scenario"}
    monkeypatch.setattr(server, "_free_text_allowed", lambda: True)
    assert client.post("/api/rumor", json={"text": "Copperleaf lost $1.8B on bonds"}).json()["ok"] is True
