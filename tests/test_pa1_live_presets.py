"""P-A (part 1): the live preset table, the insured chip's guarantee, the halt at the price floor, 1441 ticks."""

from __future__ import annotations

import pytest

from contracts.events import PolicyEnacted, RumorPosted, WithdrawalRequested
from params.loader import get_parameter
from ui.live.presets import INSURED_CHIP, LIVE_PRESETS, PresetPost, preset_table


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def _real(monkeypatch: pytest.MonkeyPatch):
    from ui.live.session import LiveSession

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    session = LiveSession(seed=42)
    assert session.follows_engine
    return session


def _requested_after(session, minute: float) -> float:
    return sum(e.amount for e in session.engine.bus.log if isinstance(e, WithdrawalRequested) and e.t > minute)


def test_table_matches_the_validated_presets() -> None:
    assert list(LIVE_PRESETS) == ["lost-bonds", "frozen-branches", "lost-bonds+insured", "stadium"]
    lost, = LIVE_PRESETS["lost-bonds"].posts
    frozen, = LIVE_PRESETS["frozen-branches"].posts
    assert (lost.poster, lost.credibility, lost.minute) == ("vc-partner", 0.90, 0)
    assert (frozen.poster, frozen.credibility) == ("ui-user", 0.65)  # Sebastian, Nov 4: stays 0.65
    first, chip = LIVE_PRESETS["lost-bonds+insured"].posts
    assert first == lost and chip == INSURED_CHIP and chip.guarantee and chip.minute == 240
    assert LIVE_PRESETS["stadium"].posts == ()
    assert [row["name"] for row in preset_table()] == list(LIVE_PRESETS)


def test_rows_are_assumed_and_on() -> None:
    for name in ("M29 live insured chip guarantee", "M29 live trading halt at price floor"):
        row = get_parameter(name)
        assert row.value == "true" and row.status == "ASSUMED" and row.module == "M29"
    assert get_parameter("M29 live real crisis sim minutes").value == "1441"
    assert float(get_parameter("M06 guarantee uninsured-gap scale (ASSUMED)").value) == 0.0


def test_guarantee_is_enacted_at_its_minute_not_spread_as_a_rumor(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _real(monkeypatch)
    session.load_preset("lost-bonds")
    session._scheduled.append(PresetPost(5, INSURED_CHIP.claim, "regulator", 0.9, guarantee=True))
    for _ in range(8):
        session.step_once()
    log = session.engine.bus.log
    policies = [e for e in log if isinstance(e, PolicyEnacted)]
    assert [(p.t, p.lever, p.bank_id) for p in policies] == [(5.0, "guarantee_deposits", "bank-fvb")]
    assert session.guarantee_minute == 5.0 and session.engine["M06"].gap_scale == 0.0
    assert not any(isinstance(e, RumorPosted) and e.claim == INSURED_CHIP.claim for e in log)
    assert any(r.stmt_id == "all-deposits-guaranteed" for r in session.market_frame().statements)


def test_guarantee_stops_new_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = {}
    for guarantee in (False, True):
        session = _real(monkeypatch)
        session.load_preset("lost-bonds")
        if guarantee:
            session._scheduled.append(PresetPost(30, INSURED_CHIP.claim, "regulator", 0.9, guarantee=True))
        for _ in range(90):
            session.step_once()
        runs[guarantee] = _requested_after(session, 30.0)
    assert runs[False] > 0.0  # the rumor is still drawing requests without it
    assert runs[True] == 0.0  # from the guarantee's minute on, nobody has an uninsured reason to run


def test_flag_off_posts_the_chip_as_a_rumor(monkeypatch: pytest.MonkeyPatch) -> None:
    import ui.live.session as session_module

    real_flag = session_module.registry_flag
    monkeypatch.setattr(session_module, "registry_flag",
                        lambda name: False if name == "M29 live insured chip guarantee" else real_flag(name))
    session = _real(monkeypatch)
    session.enact_guarantee()
    for _ in range(3):
        session.step_once()
    log = session.engine.bus.log
    assert not any(isinstance(e, PolicyEnacted) for e in log)
    assert any(isinstance(e, RumorPosted) and e.claim == INSURED_CHIP.claim for e in log)
    assert session.guarantee_minute is None


def test_chip_mid_run_takes_effect_at_the_engine_minute(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _real(monkeypatch)
    for _ in range(12):
        session.step_once()
    assert session.enact_guarantee() == 12.0
    session.step_once()
    assert [e.t for e in session.engine.bus.log if isinstance(e, PolicyEnacted)] == [12.0]


def test_fake_mode_never_enacts_the_guarantee() -> None:
    from ui.live.session import LiveSession

    session = LiveSession(seed=1)
    assert not session.follows_engine
    session.enact_guarantee()
    session.step_once()
    assert not any(isinstance(e, PolicyEnacted) for e in session.engine.bus.log)


def test_stadium_posts_nothing_and_unknown_presets_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _real(monkeypatch)
    assert session.load_preset("stadium") == []
    for _ in range(5):
        session.step_once()
    assert not any(isinstance(e, RumorPosted) for e in session.engine.bus.log)
    with pytest.raises(KeyError):
        session.load_preset("bank-is-fine")


def test_trading_halts_at_the_price_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.bank.ledger import registry_deposits_total

    session = _real(monkeypatch)
    assert session.halt_at_floor
    session.step_once()
    assert session.halt_minute is None and session.market_frame().halt is None
    session._requested_usd = registry_deposits_total()  # everything requested: the paced price is at its floor
    session.step_once()
    session.step_once()
    assert session.halt_minute == 1.0
    frame = session.market_frame()
    assert [t for t, _ in frame.prices] == [0.0, 1.0]  # no prints after the halt
    assert round(frame.prices[-1][1], 6) == round(100.0 * float(get_parameter("Calibration E1 day-1 stock target").value), 6)
    assert "price floor" in str(frame.halt["reason"]) and "duration" not in frame.halt
    assert any(m.kind == "halt" and m.t == 1.0 for m in frame.markers)


def test_live_window_shows_the_closure_minute(monkeypatch: pytest.MonkeyPatch) -> None:
    from ui.live.pacing import crisis_sim_minutes

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    assert crisis_sim_minutes() == 1441.0
    assert float(get_parameter("M29 live earliest closure minute").value) == 1440.0


def test_server_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import ui.server as server

    client = TestClient(server.app)
    names = [row["name"] for row in client.get("/api/presets").json()["presets"]]
    assert names == list(LIVE_PRESETS)
    assert client.post("/api/preset", json={"name": "nope"}).json()["ok"] is False
    ok = client.post("/api/preset", json={"name": "stadium"}).json()
    assert ok["ok"] is True and ok["preset"] == "stadium"
    assert client.post("/api/guarantee").json()["ok"] is True


def test_paced_screen_drops_the_order_books_halts(monkeypatch: pytest.MonkeyPatch) -> None:
    from ui.market.snapshot import build_market_frame

    session = _real(monkeypatch)
    session.load_preset("stadium")
    for _ in range(10):
        session.step_once()
    assert build_market_frame(session.engine).halt is not None  # the book's volatility halt (minute 5, seed 42)
    frame = session.market_frame()
    assert frame.halt is None and not any(m.kind == "halt" for m in frame.markers)


def test_crowd_calms_after_the_guarantee(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _real(monkeypatch)
    session.load_preset("lost-bonds")
    session._scheduled.append(PresetPost(20, INSURED_CHIP.claim, "regulator", 0.9, guarantee=True))
    for _ in range(19):
        session.step_once()
    crowd = session.engine["M29"]
    assert crowd.wavering > 0
    for _ in range(11):
        session.step_once()
    assert crowd.wavering == 0 and crowd.calm + crowd.withdrew == crowd.calm + crowd.wavering + crowd.withdrew > 0
