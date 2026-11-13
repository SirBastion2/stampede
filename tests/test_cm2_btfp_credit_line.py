"""CM-2: the live credit line on the 2023 BTFP terms (pool = the opening bonds at par, the whole shortfall,
measured against M08's own cash once the line is on). The live engine's M12 is LiquidityFake."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contracts.events import AssetSold, LiquidityDrawn, PaymentSettled, PolicyEnacted, Tick
from contracts.schemas import Scenario
from engine.bank.ledger import registry_deposits_total, svb_like_opening_book
from engine.liquidity.terms import ASSUMED_TERMS, BTFP_FLAG, emergency_terms
from engine.live_overrides import live_flags
from mocks.bus import EventBus
from mocks.l2_banks import LiquidityFake
from params.loader import get_parameter

LIVE_ROW = "M29 live emergency line on BTFP terms"
SECURITIES_ROW = "M08 SVB-like opening securities fraction of deposits"
CASH_ROW = "M08 SVB-like opening cash fraction of deposits"


@pytest.fixture(autouse=True)
def _offline_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")


def _ladder(flags: list[str]) -> tuple[LiquidityFake, EventBus]:
    bus, ladder = EventBus(), LiquidityFake("M12")
    with svb_like_opening_book(), live_flags(flags):
        ladder.attach(bus, Scenario(preset="tech-concentrated", seed=1))
    return ladder, bus


def _pay(bus: EventBus, t: float, amount: float) -> None:
    bus.publish(PaymentSettled(t=t, seed=1, bank_id="bank-fvb", amount=amount, channel="wire", queued_remaining=0))


def _line(bus: EventBus, t: float) -> None:
    bus.publish(PolicyEnacted(t=t, seed=1, bank_id="bank-fvb", lever="emergency_lending", scope="bank-fvb", cost_model={}))


def _step(ladder: LiquidityFake, bus: EventBus, t: float) -> list:
    bus.publish(Tick(t=t, seed=1))
    out = ladder.step(t)
    for event in out:
        bus.publish(event)
    return out


def test_rows_flag_off_live_on_and_the_spec_file() -> None:
    assert get_parameter(BTFP_FLAG).value == "false"
    assert get_parameter(LIVE_ROW).value == "true"
    share = get_parameter("M12 BTFP line shortfall share")
    assert share.status == "SOURCED" and float(share.value) == 1.0 and "par value" in share.source
    ramp = get_parameter("M12 BTFP line ramp fraction of deposits per minute")
    assert ramp.status == "ASSUMED" and float(ramp.value) == 0.01
    spec = json.loads(Path("calibration/registry_updates/cm2_btfp_credit_line.json").read_text(encoding="utf-8"))
    assert set(spec["append"]) == set(spec["updates"]) == {BTFP_FLAG, share.parameter, ramp.parameter, LIVE_ROW}


def test_terms_are_card_a_unless_the_flag_is_on() -> None:
    assert emergency_terms() == ASSUMED_TERMS and not ASSUMED_TERMS.btfp
    assert (ASSUMED_TERMS.total_fraction, ASSUMED_TERMS.tick_fraction, ASSUMED_TERMS.shortfall_share) == (0.05, 0.01, 0.8)
    with svb_like_opening_book(), live_flags([BTFP_FLAG]):
        terms = emergency_terms()
    assert terms.btfp and terms.shortfall_share == 1.0 and terms.tick_fraction == 0.01
    assert terms.total_fraction == pytest.approx(float(get_parameter(SECURITIES_ROW).value))  # the bonds at par
    assert terms.opening_cash_fraction == pytest.approx(float(get_parameter(CASH_ROW).value))


def test_the_line_pays_only_what_the_ledger_lacks() -> None:
    ladder, bus = _ladder([BTFP_FLAG])
    deposits = registry_deposits_total()
    opening = float(get_parameter(CASH_ROW).value) * deposits
    _line(bus, 0.0)
    _pay(bus, 1.0, opening - 1e6)  # $1M of the bank's own cash left: no draw
    assert _step(ladder, bus, 1.0) == []
    _pay(bus, 2.0, 3e8)  # now $299M short: all of it, from the line
    draw, = _step(ladder, bus, 2.0)
    assert isinstance(draw, LiquidityDrawn) and draw.source == "emergency" and draw.amount == pytest.approx(2.99e8)
    bus.publish(AssetSold.model_validate({"t": 2.0, "seed": 1, "bank_id": "bank-fvb", "class": "afs",
                                          "amount": 1e8, "price_impact": 0.5}))  # $50M of sale proceeds
    _pay(bus, 3.0, 1e8)
    draw, = _step(ladder, bus, 3.0)
    assert draw.amount == pytest.approx(5e7)  # the payment less the sale proceeds
    _pay(bus, 4.0, 0.05 * deposits)
    draw, = _step(ladder, bus, 4.0)
    assert draw.amount == pytest.approx(0.01 * deposits)  # the ramp caps a minute


def test_card_a_terms_are_unchanged() -> None:
    ladder, bus = _ladder([])
    _line(bus, 0.0)
    _pay(bus, 1.0, 1e9)  # the old 0.1% slice: $175M of cash, so $825M short; 80% of it, capped at 1%/min
    draw, = _step(ladder, bus, 1.0)
    assert draw.source == "emergency" and draw.amount == pytest.approx(0.8 * (1e9 - 0.001 * registry_deposits_total()))


def test_before_the_line_fires_nothing_changes() -> None:
    def run(flags: list[str]) -> list[dict]:
        ladder, bus = _ladder(flags)
        out = []
        for t in range(1, 40):
            _pay(bus, float(t), 5e7 * t)
            out += [e.model_dump() for e in _step(ladder, bus, float(t))]
        return out

    assert run([BTFP_FLAG]) == run([]) != []  # the same draws from the old slice: the 2023 replay's path


def test_calibration_keeps_card_a_and_the_live_engine_takes_btfp(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.registry import build_engine
    from ui.live.session import LiveSession

    engine = build_engine(scenario=Scenario(preset="tech-concentrated", seed=1))
    assert engine["M12"].terms == ASSUMED_TERMS
    monkeypatch.setenv("STAMPEDE_MODE", "real")
    terms = LiveSession(seed=1).engine["M12"].terms
    assert terms.btfp and terms.total_fraction == pytest.approx(0.678128)


def _emergency_ten_minutes_in(monkeypatch: pytest.MonkeyPatch) -> tuple[float, float, int]:
    from ui.live.session import LiveSession

    monkeypatch.setenv("STAMPEDE_MODE", "real")
    session = LiveSession(seed=1)
    session.load_preset("lost-bonds", "credit-line")  # at minute 240 (1:30 PM)
    for _ in range(250):
        session.step_once()
    log = session.engine.bus.log
    line = sum(e.amount for e in log if isinstance(e, LiquidityDrawn) and e.source == "emergency")
    return line, float(session.engine["M08"].state.cash), sum(isinstance(e, AssetSold) for e in log)


def test_the_live_line_lends_only_once_the_banks_own_cash_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enacted at 1:30 PM: ten minutes later the bank still has its own cash (seed 1 runs out at minute 296), so
    the BTFP line has lent nothing. PA-A's line, measured on the 0.1% slice, lends from the first minute."""
    from ui.live import session as live

    line, cash, sales = _emergency_ten_minutes_in(monkeypatch)
    assert line == 0.0 and cash > 0.0 and sales == 0
    monkeypatch.delitem(live._LIVE_OPT_INS, LIVE_ROW)
    old_line, _, _ = _emergency_ten_minutes_in(monkeypatch)
    assert old_line > 0.0


def test_the_card_states_the_measured_lending_and_survival() -> None:
    import re

    from agents.guardrails import moderate_rumor
    from ui.live.cards import response_cards
    from ui.live.presets import RESPONSES

    spread = lambda lo, hi: {"min": lo, "mean": (lo + hi) / 2, "max": hi}  # noqa: E731
    batch = {"summary": [  # diag-19 seed 1, as a ten-seed shape
        dict(preset="lost-bonds", response=None, runs=10, closed_runs=10, requested_1440=spread(0.8192, 0.8211),
             cash_out_runs=10, cash_out_minute=spread(960, 991)),
        dict(preset="lost-bonds", response="credit-line", runs=10, closed_runs=0, requested_1440=spread(0.8192, 0.8211),
             cash_out_runs=0, cash_out_minute=None, emergency_usd=spread(1.0e11, 1.05e11))]}
    card = response_cards(batch)["credit-line"]
    text = [card["note"], *card["pros"], *card["cons"], *[s["text"] for s in card["stats"]]]
    assert "The bank is open the next morning in all 10 runs" in card["pros"]
    assert not any("Still closed" in t for t in text) and any("can still fail later" in t for t in card["cons"])
    assert "$102B" in " ".join(card["cons"]) and "the Fed lends about $102B against bonds" in " ".join(text)
    assert not any("43.8" in t for t in text)  # the old assumed 25% exposure is gone
    assert "face value" in RESPONSES["credit-line"].note and "5% of deposits" not in RESPONSES["credit-line"].note
    banned = re.compile(r"predict|\bSVB\b|Silicon Valley|BTFP|First Republic", re.IGNORECASE)
    for t in text:
        assert not banned.search(t) and moderate_rumor(t).allowed, t
