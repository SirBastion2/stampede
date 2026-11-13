"""M12 liquidity ladder: facility hours, discount pledge, determinism, contracts."""

from __future__ import annotations

from typing import Any

from contracts import parse_event
from contracts.events import LiquidityDrawn, PaymentSettled
from contracts.schemas import Scenario
from engine.core.clock import load_calendar_config
from engine.liquidity.ladder import (
    CollateralLedger,
    LiquidityLadderModule,
    cash_shortfall,
    default_funding_sources,
    facility_open_at,
    next_liquidity_draw,
)
from engine.modes import ModuleMode
from engine.registry import build_engine, run_scenario
from mocks.bus import EventBus
from mocks.l2_banks import LiquidityFake
from mocks.support import calendar_at
from tests.test_fakes import panic
from ui.live.pacing import crisis_sim_minutes


def _sources() -> list:
    return default_funding_sources(175_000_000_000.0)


def test_discount_window_requires_pre_pledged_collateral() -> None:
    collateral = CollateralLedger(pools={"securities": 1e9})
    drawn: dict[str, float] = {}
    plan = next_liquidity_draw(
        cash_need=5e9,
        deposits=175e9,
        sources=_sources(),
        drawn_by_source=drawn,
        collateral=collateral,
        facility_open=True,
        emergency_active=False,
        discount_pre_pledged=False,
    )
    assert plan is None or plan[0] != "discount_window"
    plan2 = next_liquidity_draw(
        cash_need=5e9,
        deposits=175e9,
        sources=_sources(),
        drawn_by_source=drawn,
        collateral=collateral,
        facility_open=True,
        emergency_active=False,
        discount_pre_pledged=True,
    )
    assert plan2 is not None
    assert plan2[0] in {"repo", "discount_window", "fhlb"}


def test_fhlb_unavailable_after_facility_hours() -> None:
    cfg = load_calendar_config()
    closed_t = 0.0
    for minute in range(0, 24 * 60):
        if not calendar_at(float(minute)).facility_open:
            closed_t = float(minute)
            break
    assert not facility_open_at(closed_t, tick_sim_minute=None, cfg=cfg)
    collateral = CollateralLedger(pools={"securities": 1e9})
    plan = next_liquidity_draw(
        cash_need=5e9,
        deposits=175e9,
        sources=_sources(),
        drawn_by_source={},
        collateral=collateral,
        facility_open=False,
        emergency_active=False,
        discount_pre_pledged=True,
    )
    assert plan is not None
    assert plan[0] != "fhlb"


def test_collateral_not_double_pledged() -> None:
    collateral = CollateralLedger(pools={"securities": 1_000_000.0})
    drawn: dict[str, float] = {}
    first = next_liquidity_draw(
        cash_need=50_000_000.0,
        deposits=175e9,
        sources=_sources(),
        drawn_by_source=drawn,
        collateral=collateral,
        facility_open=True,
        emergency_active=False,
        discount_pre_pledged=True,
    )
    assert first is not None
    drawn[first[0]] = first[1]
    second = next_liquidity_draw(
        cash_need=50_000_000.0,
        deposits=175e9,
        sources=_sources(),
        drawn_by_source=drawn,
        collateral=collateral,
        facility_open=True,
        emergency_active=False,
        discount_pre_pledged=True,
    )
    if second is not None and second[0] == first[0]:
        assert second[1] <= first[1] + 1.0


def test_cash_shortfall_deterministic() -> None:
    a = cash_shortfall(deposits=175e9, payments=1e9, draws=0.0)
    b = cash_shortfall(deposits=175e9, payments=1e9, draws=0.0)
    assert a == b
    assert a > 0.0


def test_liquidity_module_publishes_on_payment_stress() -> None:
    bus = EventBus()
    scenario = Scenario(preset="tech-concentrated", seed=3)
    module = LiquidityLadderModule("M12")
    module.attach(bus, scenario)
    pay = PaymentSettled(
        t=0.0,
        seed=3,
        bank_id="bank-fvb",
        cause_id="pay-1",
        amount=500_000_000.0,
        channel="wire",
        queued_remaining=0,
    )
    bus.publish(pay)
    events = module.step(0.0)
    assert len(events) == 1
    draw = events[0]
    assert isinstance(draw, LiquidityDrawn)
    parsed = parse_event(draw.model_dump(mode="json", by_alias=True))
    assert isinstance(parsed, LiquidityDrawn)
    assert parsed.amount > 0.0
    assert parsed.source in {"repo", "fhlb", "discount_window", "brokered_deposits"}


def test_real_m12_engine_is_deterministic() -> None:
    scenario = panic(19)
    modes = {"M12": ModuleMode.REAL}
    a = run_scenario(scenario, 20, modes=modes)
    b = run_scenario(scenario, 20, modes=modes)
    assert _dump(a) == _dump(b)


def test_real_vs_fake_liquidity_drawn_share_contract() -> None:
    scenario = panic(11)
    steps = int(crisis_sim_minutes())
    fake_log = run_scenario(scenario, steps, modes={"M12": ModuleMode.FAKE})
    real_log = run_scenario(scenario, steps, modes={"M12": ModuleMode.REAL})
    fake_draws = [e for e in fake_log if isinstance(e, LiquidityDrawn)]
    real_draws = [e for e in real_log if isinstance(e, LiquidityDrawn)]
    for event in fake_draws + real_draws:
        parsed = parse_event(event.model_dump(mode="json", by_alias=True))
        assert isinstance(parsed, LiquidityDrawn)
        assert parsed.amount > 0.0
        assert parsed.source
        assert parsed.eta >= 0.0
    assert fake_draws or real_draws


def test_build_engine_real_m12_not_not_implemented() -> None:
    engine = build_engine(
        modes={"M12": ModuleMode.REAL},
        scenario=Scenario(preset="tech-concentrated", seed=1),
    )
    assert isinstance(engine["M12"], LiquidityLadderModule)
    assert not isinstance(engine["M12"], LiquidityFake)


def test_emergency_lever_enables_emergency_source() -> None:
    collateral = CollateralLedger(pools={"securities": 1e12})
    plan = next_liquidity_draw(
        cash_need=1e10,
        deposits=175e9,
        sources=_sources(),
        drawn_by_source={},
        collateral=collateral,
        facility_open=False,
        emergency_active=True,
        discount_pre_pledged=False,
    )
    assert plan is not None
    assert plan[0] == "emergency"


def _dump(events: list[Any]) -> list[dict[str, Any]]:
    return [event.model_dump(mode="json", by_alias=True) for event in events]
