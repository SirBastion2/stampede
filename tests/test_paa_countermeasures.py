"""PA-A: the three countermeasures (deposit guarantee, emergency credit line, withdrawal limit)."""

from __future__ import annotations

import pytest

from contracts.events import PolicyEnacted, WithdrawalRequested
from mocks.support import levers_in
from params.loader import get_parameter
from ui.live.presets import RESPONSES, response_table


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def _real(monkeypatch: pytest.MonkeyPatch, seed: int = 1):
    from ui.live.session import LiveSession

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    return LiveSession(seed=seed)


def _policies(session) -> list[tuple[float, str]]:
    return [(e.t, e.lever) for e in session.engine.bus.log if isinstance(e, PolicyEnacted)]


def _requested_after(session, minute: float) -> float:
    return sum(e.amount for e in session.engine.bus.log if isinstance(e, WithdrawalRequested) and e.t > minute)


def test_table_order_levers_and_the_limit_note_match_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    assert list(RESPONSES) == ["guarantee", "credit-line", "withdrawal-limit"]
    assert [r.lever for r in RESPONSES.values()] == ["guarantee_deposits", "emergency_lending", "cap_withdrawals"]
    assert all(r.minute == 240 for r in RESPONSES.values())
    assert float(get_parameter("M06 cap withdrawal scale (ASSUMED)").value) == 0.3
    assert "30%" in RESPONSES["withdrawal-limit"].note
    assert _real(monkeypatch).engine["M06"].amount_scale == 1.0  # so the cap is a cut to 30%, not 30% of a fitted scale
    assert [row["name"] for row in response_table()] == list(RESPONSES)


def test_load_preset_schedules_the_response_at_minute_240(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _real(monkeypatch)
    scheduled = session.load_preset("lost-bonds", "credit-line")
    assert {"minute": 240, "guarantee": False, "response": "credit-line"} in scheduled
    assert session.run_label()["kind"] == "scenario" and session.run_label()["responses"] == ["credit-line"]
    with pytest.raises(KeyError):
        session.load_preset("lost-bonds", "bail-in")


@pytest.mark.parametrize("name,lever", [("credit-line", "emergency_lending"), ("withdrawal-limit", "cap_withdrawals"),
                                        ("guarantee", "guarantee_deposits")])
def test_response_mid_run_is_enacted_at_the_engine_minute(monkeypatch: pytest.MonkeyPatch, name: str, lever: str) -> None:
    session = _real(monkeypatch)
    session.load_preset("lost-bonds")
    for _ in range(5):
        session.step_once()
    assert session.enact_response(name) == 5.0
    for _ in range(3):
        session.step_once()
    assert _policies(session) == [(5.0, lever)] and session.response_minutes == {name: 5.0}
    m06 = session.engine["M06"]
    if lever == "cap_withdrawals":
        assert m06.amount_scale == 0.3
    if lever == "guarantee_deposits":
        assert m06.gap_scale == 0.0
    if lever == "emergency_lending":
        assert "emergency_lending" in levers_in(session.engine.bus.log)  # M12 adds the emergency pool from here
        assert any(r.stmt_id == "credit-line" for r in session.market_frame().statements)


def test_the_withdrawal_limit_cuts_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = {}
    for limited in (False, True):
        session = _real(monkeypatch)
        session.load_preset("lost-bonds")
        for minute in range(120):
            if limited and minute == 30:
                session.enact_response("withdrawal-limit")
            session.step_once()
        runs[limited] = _requested_after(session, 30.0)
    assert runs[False] > 0.0 and runs[True] < 0.5 * runs[False]


def test_fake_mode_enacts_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    from ui.live.session import LiveSession

    session = LiveSession(seed=1)
    session.enact_response("credit-line")
    session.step_once()
    assert _policies(session) == []
    with pytest.raises(KeyError):
        session.enact_response("bail-in")


def test_server_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import ui.server as server

    client = TestClient(server.app)
    table = client.get("/api/presets").json()
    assert [r["name"] for r in table["responses"]] == list(RESPONSES)
    ok = client.post("/api/preset", json={"name": "stadium", "response": "withdrawal-limit"}).json()
    assert ok["ok"] is True and ok["run"]["kind"] == "scenario"
    assert client.post("/api/response", json={"name": "credit-line"}).json()["ok"] is True
    assert client.post("/api/response", json={"name": "bail-in"}).json()["ok"] is False
