"""SVB-59 (item 9): a calming "cure" rumor, behind default-off flags (ADR-105)."""

from __future__ import annotations

import numpy as np
import pytest

from agents.rumor_input import classify_rumor_market
from contracts.events import RumorPosted
from contracts.schemas import Scenario
from engine.agents.attention import AttentionEngine, claim_tone
from engine.live_overrides import live_flags
from mocks.bus import EventBus
from params.loader import get_parameter

PANIC = "Kestrel Bank is out of cash and may delay uninsured wires"
CURE = "Kestrel Bank is well capitalized and stable, deposits are safe"
M03_ROW, M06_ROW = "M03 cure rumor enabled", "M06 cure rumor worry shift enabled"


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def _post(rumor_id: str, claim: str, t: float, credibility: float = 0.9) -> RumorPosted:
    return RumorPosted(t=t, seed=1, bank_id="bank-fvb", cause_id="t", rumor_id=rumor_id, claim=claim,
                       source_node="hub-public-1", credibility=credibility)


def _m03_run(*, cure: bool, flag: bool) -> float:
    rows = {M03_ROW} if flag else set()
    with live_flags(rows):
        engine = AttentionEngine("M03")
    engine.attach(EventBus(), Scenario(preset="tech-concentrated", seed=1))
    engine.inbox.append(_post("p", PANIC, 0.0))
    for tick in range(4):
        if tick == 3 and cure:
            engine.inbox.append(_post("c", CURE, float(tick)))
        engine.step(float(tick))
    return float(engine.attention.sum())


def test_rows_off_and_live_on() -> None:
    for row in (M03_ROW, M06_ROW):
        assert get_parameter(row).value == "false" and get_parameter(row).status == "ASSUMED"
    assert get_parameter("M29 live cure rumor").value == "true"
    assert float(get_parameter("M03 cure attention cut").value) == 0.5
    assert float(get_parameter("M06 cure rumor worry factor").value) == 0.5


def test_cure_claim_is_positive_and_affects_the_sim() -> None:
    assert claim_tone(CURE) == "positive"
    assert claim_tone(PANIC) == "fear"
    assert classify_rumor_market(CURE, "regulator").affects_simulation is True


def test_flag_off_cure_still_adds_attention_like_before() -> None:
    assert _m03_run(cure=True, flag=False) > _m03_run(cure=False, flag=False)


def test_flag_on_cure_cuts_attention() -> None:
    with_cure = _m03_run(cure=True, flag=True)
    without = _m03_run(cure=False, flag=True)
    assert with_cure < without


def test_flag_on_cure_cut_matches_the_formula() -> None:
    with live_flags({M03_ROW}):
        engine = AttentionEngine("M03")
    engine.attach(EventBus(), Scenario(preset="tech-concentrated", seed=1))
    engine.attention[:] = 1.0
    engine._apply_cures([_post("c", CURE, 0.0, credibility=0.8)])
    np.testing.assert_allclose(engine.attention, 1.0 - 0.5 * 0.8)


def test_live_cure_slows_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from ui.live.session import LiveSession

    monkeypatch.setenv("STAMPEDE_MODE", "real")

    def withdrew(cure: bool) -> float:
        session = LiveSession(seed=42)
        session.start()
        session.submit_rumor(PANIC, "vc-partner", "bank-fvb", now=0.0)
        for minute in range(30):
            if cure and minute == 5:
                session.engine["M29"].queue_live_rumor(
                    claim=CURE, source_node="regulator", bank_id="bank-fvb", rumor_id="cure-1", credibility=0.9
                )
            session.step_once()
        assert session.engine["M06"].cure_worry is True
        # SVB-61: in the calibrated soft model nobody has withdrawn by minute 30, so
        # compare what the cure acts on: M06 worry.
        return float(np.mean(session.engine["M06"].worry))

    assert withdrew(True) < withdrew(False)
