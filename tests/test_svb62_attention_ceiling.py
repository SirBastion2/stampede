"""SVB-62: M06 attention ceiling by rumor credibility (role washout fix, ADR-107). CPU, offline."""

from __future__ import annotations

import pytest

from contracts.events import AttentionTick, RumorPosted
from engine.live_overrides import live_flags
from params.loader import get_parameter

ROW = "M06 attention ceiling by credibility enabled"
PANIC = "Copperleaf lost $1.8B on bonds and is raising capital"
CURE = "Copperleaf deposits are fully insured and backstopped, deposits are safe"


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_FULL", raising=False)


def _m06(on: bool):
    from calibration.fit import _default_svb_scenario, real_calibration_modes
    from engine.modes import ModuleMode
    from engine.registry import build_engine

    with live_flags({ROW} if on else set()):
        engine = build_engine(modes=real_calibration_modes(), default_mode=ModuleMode.FAKE,
                              scenario=_default_svb_scenario(11))
    return engine, engine["M06"]


def _post(engine, m06, credibility: float, claim: str = PANIC) -> None:
    engine.bus.publish(RumorPosted(t=0.0, seed=11, bank_id=m06.bank_id, cause_id="t", rumor_id=f"r{credibility}",
                                   claim=claim, source_node="hub-public-1", credibility=credibility))


def _tick(m06, level: float) -> float:
    m06._apply_events([AttentionTick(t=1.0, seed=11, bank_id=m06.bank_id, cause_id="a", level=level, delta_4h=0.0)])
    return m06.attention_level


def test_rows_off_in_registry_on_for_live() -> None:
    assert get_parameter(ROW).value == "false" and get_parameter(ROW).status == "ASSUMED"
    assert float(get_parameter("M06 attention ceiling reference credibility").value) == 0.8
    assert get_parameter("M29 live attention ceiling by credibility").value == "true"


def test_flag_off_keeps_the_fitted_clip_at_one() -> None:
    engine, m06 = _m06(on=False)
    assert m06.credibility_ceiling is False
    _post(engine, m06, 0.3)
    assert _tick(m06, 5.0) == 1.0


@pytest.mark.parametrize(("credibility", "expected"), [(0.4, 0.5), (0.7, 0.875), (0.8, 1.0), (0.9, 1.0)])
def test_ceiling_is_credibility_over_reference(credibility: float, expected: float) -> None:
    engine, m06 = _m06(on=True)
    _post(engine, m06, credibility)
    assert _tick(m06, 5.0) == pytest.approx(expected)
    assert _tick(m06, 0.2) == pytest.approx(min(0.2, expected))


def test_strongest_panic_rumor_sets_the_ceiling_and_cures_do_not() -> None:
    engine, m06 = _m06(on=True)
    _post(engine, m06, 0.4)
    _post(engine, m06, 0.95, claim=CURE)
    assert _tick(m06, 5.0) == pytest.approx(0.5)
    _post(engine, m06, 0.6)
    assert _tick(m06, 5.0) == pytest.approx(0.75)


def test_no_rumor_means_no_cap() -> None:
    _, m06 = _m06(on=True)
    assert _tick(m06, 5.0) == 1.0


def test_live_real_mode_turns_the_ceiling_on(monkeypatch: pytest.MonkeyPatch) -> None:
    from ui.live.session import LiveSession

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    assert LiveSession(seed=42).engine["M06"].credibility_ceiling is True


def test_weaker_rumor_requests_at_most_about_half_by_day_one_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """The preset rule: the weaker preset requests about half of lost-bonds (0.90) by minute 390.

    SVB-65 gives the live engine the fitted news feedback: 0.90 now reaches about 0.26 by
    minute 390 (SVB's day one was 0.249) and 0.70 about 0.18, so "about half" moves to 0.65
    (sandbox seed 42: 0.1285 against 0.2596; seed 7: 0.1361).
    """
    from engine.bank.ledger import registry_deposits_total
    from ui.live.session import LiveSession

    monkeypatch.setenv("STAMPEDE_MODE", "real")

    def requested(credibility: float) -> float:
        session = LiveSession(seed=42)
        session.start()
        session.engine["M29"].queue_live_rumor(claim=PANIC, source_node="hub-public-1", bank_id="bank-fvb",
                                               rumor_id="r", credibility=credibility)
        for _ in range(390):
            session.step_once()
        log = session.engine.bus.log
        return sum(e.amount for e in log if type(e).__name__ == "WithdrawalRequested") / registry_deposits_total()

    strong, weak = requested(0.9), requested(0.65)
    assert strong > 0.12
    assert weak <= 0.6 * strong
