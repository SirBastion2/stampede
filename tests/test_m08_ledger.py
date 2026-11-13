"""M08 real ledger: invariant, determinism, contracts, real-vs-fake."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from contracts import parse_event
from contracts.events import LiquidityDrawn, LossRealized, PaymentSettled
from contracts.schemas import Rumor, Scenario
from engine.bank.ledger import (
    LedgerModule,
    cet1_ratio,
    initial_ledger_state,
    invariant_holds,
    post_liquidity_drawn,
    post_loss_realized,
    post_payment_settled,
    registry_deposits_total,
)
from engine.modes import ModuleMode
from engine.registry import build_engine, run_scenario
from mocks.l2_banks import LedgerFake
from ui.live.pacing import crisis_sim_minutes


def _panic(seed: int) -> Scenario:
    return Scenario(
        preset="tech-concentrated",
        seed=seed,
        rate_path=[(0.0, 0.04), (3.0, 0.06)],
        rumors=[
            Rumor(
                id="rumor-1",
                claim="Uninsured deposits are leaving Fictional Valley Bank",
                credibility=0.85,
                reach=12000,
            )
        ],
        policies=[{"lever": "guarantee_deposits", "scope": "bank-fvb", "at": 10}],
    )


def _published_m08(engine: Any) -> list[dict[str, Any]]:
    module = engine["M08"]
    return [
        event.model_dump(mode="json", by_alias=True) for event in module.published
    ]


def test_opening_state_balanced_and_uses_registry_deposits() -> None:
    deposits = registry_deposits_total()
    state = initial_ledger_state(deposits=deposits)
    assert invariant_holds(state)
    assert state.deposits == deposits
    assert cet1_ratio(state) == pytest.approx(0.12, rel=1e-6)


def test_initial_state_is_deterministic() -> None:
    d = registry_deposits_total()
    a = initial_ledger_state(deposits=d)
    b = initial_ledger_state(deposits=d)
    assert a == b


@given(
    payments=st.lists(st.floats(min_value=0.0, max_value=1e9), max_size=8),
    draws=st.lists(st.floats(min_value=0.0, max_value=1e9), max_size=4),
    losses=st.lists(st.floats(min_value=0.0, max_value=1e8), max_size=4),
)
def test_random_posting_sequences_keep_invariant(
    payments: list[float],
    draws: list[float],
    losses: list[float],
) -> None:
    state = initial_ledger_state(deposits=registry_deposits_total())
    for amount in payments:
        if amount > 0.0 and state.cash >= amount:
            state = post_payment_settled(state, amount)
            assert invariant_holds(state)
    for amount in draws:
        if amount > 0.0:
            state = post_liquidity_drawn(state, amount)
            assert invariant_holds(state)
    for amount in losses:
        if amount <= 0.0 or amount > state.loans:
            continue
        try:
            state = post_loss_realized(state, amount)
        except ValueError:
            continue
        assert invariant_holds(state)


def test_real_m08_determinism() -> None:
    modes = {"M08": ModuleMode.REAL}
    a = run_scenario(_panic(11), 12, modes=modes)
    b = run_scenario(_panic(11), 12, modes=modes)
    assert [e.model_dump(mode="json", by_alias=True) for e in a] == [
        e.model_dump(mode="json", by_alias=True) for e in b
    ]


def test_real_m08_published_events_parse() -> None:
    engine = build_engine(modes={"M08": ModuleMode.REAL}, scenario=_panic(3))
    engine.run(int(crisis_sim_minutes()))
    ledger = engine["M08"]
    assert isinstance(ledger, LedgerModule)
    for event in ledger.published:
        payload = event.model_dump(mode="json", by_alias=True)
        parsed = parse_event(payload)
        assert parsed.model_dump(mode="json", by_alias=True) == payload
    assert ledger.published


def test_real_vs_fake_m08_contract_shape() -> None:
    steps = int(crisis_sim_minutes())
    seed = 5
    scenario = _panic(seed)
    fake_engine = build_engine(scenario=scenario)
    fake_engine.run(steps)
    real_engine = build_engine(modes={"M08": ModuleMode.REAL}, scenario=scenario)
    real_engine.run(steps)
    assert isinstance(fake_engine["M08"], LedgerFake)
    assert isinstance(real_engine["M08"], LedgerModule)
    fake_pub = _published_m08(fake_engine)
    real_pub = _published_m08(real_engine)
    for payload in fake_pub + real_pub:
        parse_event(payload)
    fake_types = {p["event_type"] for p in fake_pub}
    real_types = {p["event_type"] for p in real_pub}
    assert real_types <= {"AssetSold", "CapitalChanged"}
    assert fake_types <= {"AssetSold", "CapitalChanged"}
    assert "AssetSold" in fake_types or "AssetSold" in real_types
    for payload in real_pub:
        if payload["event_type"] == "CapitalChanged":
            assert 0.0 <= payload["cet1_ratio"] <= 0.12
        if payload["event_type"] == "AssetSold":
            assert payload["class"] in {"afs", "htm"}
            assert payload["amount"] > 0.0


def test_ledger_module_applies_bus_events() -> None:
    module = LedgerModule("M08")
    from mocks.bus import EventBus

    bus = EventBus()
    module.attach(bus, _panic(1))
    start = module.state
    assert start is not None
    pay = PaymentSettled(
        t=0.0,
        seed=1,
        bank_id="bank-fvb",
        cause_id="test-pay",
        amount=1_000_000.0,
        channel="wire",
        queued_remaining=0,
    )
    bus.publish(pay)
    module.step(0.0)
    assert module.state is not None
    assert module.state.cash < start.cash
    assert invariant_holds(module.state)

    draw = LiquidityDrawn(
        t=0.0,
        seed=1,
        bank_id="bank-fvb",
        cause_id="test-draw",
        source="fhlb",
        amount=2_000_000.0,
        eta=1.0,
    )
    bus.publish(draw)
    module.step(1.0)
    assert module.state.cash > start.cash - 1_000_000.0

    loss = LossRealized(
        t=1.0,
        seed=1,
        bank_id="bank-fvb",
        cause_id="test-loss",
        amount=500_000.0,
        kind="securities",
    )
    bus.publish(loss)
    module.step(2.0)
    assert module.state.equity < start.equity
