"""Offline: wall-clock pacing advances ticks without changing outcomes."""

from __future__ import annotations

import pytest

from mocks.l6_crowd_ui import CrowdFake
from ui.live.pacing import PacingClock, base_sim_minutes_per_real_second
from ui.live.session import LiveSession


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def _crowd_snapshot(session: LiveSession) -> tuple[int, float, float]:
    crowd = session.engine["M29"]
    assert isinstance(crowd, CrowdFake)
    return crowd.withdrew, crowd.cash_gauge, session.tick


def test_pacing_clock_steps_expected_ticks_per_wall_second() -> None:
    clock = PacingClock(speed=1.0)
    expected = base_sim_minutes_per_real_second()
    stepped = clock.advance(1.0)
    assert stepped == int(expected)
    assert abs(clock.ticks_for_wall_second(1.0) - expected) < 1e-6

    clock.set_speed(4.0)
    stepped_4x = clock.advance(1.0)
    assert stepped_4x == int(expected * 4.0)


def test_pacing_speed_does_not_change_tick_outcomes() -> None:
    rumor = "Treasury desk hears Fictional Valley Bank may delay uninsured wires this week"
    target_ticks = 36

    direct = LiveSession(seed=7)
    direct.submit_rumor(rumor, "treasury-desk", "bank-fvb")
    for _ in range(target_ticks):
        direct.step_once()

    paced = LiveSession(seed=7)
    paced.submit_rumor(rumor, "treasury-desk", "bank-fvb")
    clock = PacingClock(speed=8.0)
    stepped = 0
    while stepped < target_ticks:
        batch = clock.advance(1.0)
        if batch <= 0:
            batch = 1
        for _ in range(batch):
            if stepped >= target_ticks:
                break
            paced.step_once()
            stepped += 1

    assert _crowd_snapshot(direct) == _crowd_snapshot(paced)


def test_skip_to_outcome_reaches_failure_state() -> None:
    from ui.live.pacing import crisis_sim_minutes

    session = LiveSession(seed=3)
    session.submit_rumor(
        "Group chat claims Kestrel Bank is hunting for a buyer",
        "group-chat-anon",
        "bank-fvb",
    )
    stepped = session.skip_to_outcome()
    assert stepped > 0
    assert session.tick >= crisis_sim_minutes()
    crowd = session.engine["M29"]
    assert isinstance(crowd, CrowdFake)
    assert crowd.time_to_failure_ticks == 0 or crowd.cash_gauge < 0.15
