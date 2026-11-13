"""P-A2: buffered playback. The engine steps ahead into a frame buffer; the screens play from it."""

from __future__ import annotations

import time

import pytest

from contracts.events import PolicyEnacted
from params.loader import get_parameter
from ui.live.playback import BufferedFrame, PlaybackBuffer


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def _frame(tick: float) -> BufferedFrame:
    return BufferedFrame(tick, f"crowd-{tick:g}", f"market-{tick:g}")


def test_buffer_waits_for_the_head_start_then_never_passes_the_newest_frame() -> None:
    buffer = PlaybackBuffer(head_ticks=3.0, max_lead_ticks=5.0)
    for t in (0.0, 1.0, 2.0):
        buffer.push(_frame(t))
    assert buffer.advance(10.0) == 0 and buffer.position == 0.0  # 2 minutes buffered < 3: still waiting
    buffer.push(_frame(3.0))
    assert buffer.advance(1.5) == 1 and buffer.playing
    assert buffer.current().tick == 1.0  # newest frame at or before 1.5
    assert buffer.advance(100.0) == 2 and buffer.position == 3.0  # clamped: the screens hold minute 3
    assert buffer.current().crowd == "crowd-3"
    buffer.push(_frame(4.0))
    assert buffer.advance(0.4) == 0 and buffer.current().tick == 3.0
    assert buffer.advance(0.6) == 1 and buffer.current().tick == 4.0


def test_buffer_finish_releases_the_head_start_and_clear_resets() -> None:
    buffer = PlaybackBuffer(head_ticks=10.0, max_lead_ticks=12.0)
    buffer.push(_frame(0.0))
    buffer.push(_frame(1.0))
    assert buffer.advance(1.0) == 0
    assert buffer.advance(1.0, finished=True) == 1 and buffer.position == 1.0
    buffer.clear(5.0)
    assert buffer.current() is None and buffer.position == 5.0 and not buffer.playing
    with pytest.raises(ValueError):
        PlaybackBuffer(head_ticks=3.0, max_lead_ticks=2.0)


def test_rows_are_assumed() -> None:
    from ui.live.pacing import playback_head_start_seconds

    assert get_parameter("M29 live buffered playback").value == "true"
    row = get_parameter("M29 live playback head start seconds")
    assert row.status == "ASSUMED" and playback_head_start_seconds() == 2.0


def _real(monkeypatch: pytest.MonkeyPatch):
    from ui.live.session import LiveSession

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    session = LiveSession(seed=42)
    assert session.buffered and not session.buffering
    return session


def test_without_the_producer_nothing_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _real(monkeypatch)
    session.load_preset("lost-bonds")
    assert session.advance_wall(1.0) >= 13  # inline stepping, as before
    assert session.crowd_frame().t == session.tick and session.playback_tick == session.tick


def test_fake_mode_never_buffers() -> None:
    from ui.live.session import LiveSession

    session = LiveSession(seed=1)
    assert not session.buffered and session.start_producer(thread=False) is False


def test_producer_fills_the_lead_and_playback_follows(monkeypatch: pytest.MonkeyPatch) -> None:
    from ui.live.pacing import base_sim_minutes_per_real_second

    session = _real(monkeypatch)
    assert session.start_producer(thread=False)
    assert session.produce_once() is False  # paused: nothing is computed before the post
    session.load_preset("lost-bonds")
    head = 2.0 * base_sim_minutes_per_real_second()
    lead = head + 2.0
    steps = 0
    while session.produce_once():
        steps += 1
    assert steps == int(lead) + (1 if lead > int(lead) else 0)  # stops at the lead cap
    assert session.tick - session.playback_tick >= lead
    assert session.crowd_frame().t == 0.0 and session.market_frame() is not None  # screens still at minute 0
    moved = session.advance_wall(1.0)
    assert moved == int(base_sim_minutes_per_real_second())
    assert session.crowd_frame().t == float(int(session.playback_tick)) < session.tick
    assert session.produce_once()  # room again


def test_playback_slows_to_the_engine_instead_of_jumping(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _real(monkeypatch)
    session.start_producer(thread=False)
    session.load_preset("stadium")
    while session.produce_once():
        pass
    session.advance_wall(100.0)  # far more than computed: playback stops at the newest minute
    assert session.playback_tick == session.tick and session.crowd_frame().t == session.tick


def test_mid_run_chip_lands_within_the_lead(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _real(monkeypatch)
    session.start_producer(thread=False)
    session.load_preset("lost-bonds")
    while session.produce_once():
        pass
    screen, engine = session.playback_tick, session.tick
    assert session.enact_guarantee() == engine
    session.advance_wall(1.0)
    while session.produce_once():
        pass
    landed = [e.t for e in session.engine.bus.log if isinstance(e, PolicyEnacted)]
    assert landed == [engine] and engine - screen < session._buffer.max_lead_ticks + 1.0  # whole minutes


def test_producer_stops_at_the_live_window(monkeypatch: pytest.MonkeyPatch) -> None:
    import ui.live.session as session_module

    monkeypatch.setattr(session_module, "crisis_sim_minutes", lambda: 5.0)
    session = _real(monkeypatch)
    session.start_producer(thread=False)
    session.load_preset("stadium")
    while session.produce_once():
        pass
    assert session.tick == 5.0
    session.advance_wall(0.01)
    assert session._buffer.playing  # the run is over, so the head start no longer holds playback


def test_skip_jumps_playback_to_the_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    import ui.live.session as session_module

    monkeypatch.setattr(session_module, "crisis_sim_minutes", lambda: 8.0)
    session = _real(monkeypatch)
    session.start_producer(thread=False)
    session.load_preset("stadium")
    session.skip_to_outcome()
    assert session.tick == 8.0 and session.playback_tick == 8.0 and session.crowd_frame().t == 8.0


def test_the_thread_runs_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _real(monkeypatch)
    assert session.start_producer()
    session.load_preset("stadium")
    deadline = time.monotonic() + 60.0
    while session.tick < 5 and time.monotonic() < deadline:
        time.sleep(0.05)
    session.stop_producer()
    assert session.tick >= 5 and not session.buffering
    assert session._producer is None
    stopped_at = session.tick
    time.sleep(0.2)
    assert session.tick == stopped_at


def test_health_reports_playback(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import ui.server as server

    payload = TestClient(server.app).get("/health").json()
    assert {"playback_tick", "buffered"} <= set(payload)
    reply = TestClient(server.app).post("/api/guarantee").json()
    assert {"takes_effect_at_minute", "screen_minute"} <= set(reply)
