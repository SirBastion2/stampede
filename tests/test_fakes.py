"""Deterministic fakes: chain, seeds, contract parse, off mode, §6.1 only."""

from __future__ import annotations

from typing import Any

import pytest

from contracts import parse_event
from contracts.events import AttentionTick, CapitalChanged, WithdrawalRequested
from contracts.schemas import Rumor, Scenario, ScenarioShock
from engine.modes import ModuleMode
from engine.registry import build_engine, run_scenario
from mocks.base import FakeModule
from mocks.l1_engine import PopulationFake
from mocks.l6_crowd_ui import CrowdFake
from mocks.l7_market_ui import MarketScreenFake
from mocks.params_access import get_param
from ui.live.pacing import crisis_sim_minutes

SECTION_61 = frozenset(
    {
        "Tick",
        "RatesChanged",
        "RumorPosted",
        "AttentionTick",
        "HeadlinePublished",
        "WithdrawalRequested",
        "PaymentSettled",
        "LiquidityDrawn",
        "AssetSold",
        "LossRealized",
        "CapitalChanged",
        "PriceTick",
        "TradingHalt",
        "StatementIssued",
        "PolicyEnacted",
        "BankClosed",
        "FutureCompleted",
        "DepositRateChanged",
        "DirectiveIssued",
        "VIPDecision",
        "CEOAction",
        "ThesisPublished",
        "RumorMutated",
        "AnalystNote",
        "SupervisoryAction",
    }
)

CHAIN = (
    "RumorPosted",
    "RumorMutated",
    "AttentionTick",
    "WithdrawalRequested",
    "PaymentSettled",
    "LiquidityDrawn",
    "AssetSold",
    "CapitalChanged",
    "PriceTick",
)


def panic(seed: int) -> Scenario:
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
        shocks=[
            ScenarioShock(
                kind="ui_rumor",
                at=3.0,
                params={
                    "rumor_id": "rumor-ui",
                    "claim": "Group chat says Fictional Valley Bank is raising cash",
                    "source_node": "ui-user",
                    "credibility": 0.45,
                },
            )
        ],
    )


def _dump(events: list[Any]) -> list[dict[str, Any]]:
    return [event.model_dump(mode="json", by_alias=True) for event in events]


def _subsequence(types: list[str], expected: tuple[str, ...]) -> bool:
    cursor = 0
    for name in types:
        if cursor < len(expected) and name == expected[cursor]:
            cursor += 1
    return cursor == len(expected)


def test_same_seed_same_log_and_different_seed_diverges() -> None:
    first = _dump(run_scenario(panic(1), 6))
    second = _dump(run_scenario(panic(1), 6))
    other = _dump(run_scenario(panic(2), 6))
    assert first == second
    assert first != other
    amounts = [
        event["amount"]
        for event in first
        if event["event_type"] == "WithdrawalRequested"
    ]
    other_amounts = [
        event["amount"]
        for event in other
        if event["event_type"] == "WithdrawalRequested"
    ]
    assert amounts
    assert amounts != other_amounts


def test_published_events_parse_and_cover_each_module() -> None:
    scenario = panic(7)
    engine = build_engine(scenario=scenario)
    log = engine.run(int(crisis_sim_minutes()))
    assert log
    for module in engine.values():
        assert isinstance(module, FakeModule)
        seen: set[str] = set()
        for event in module.published:
            payload = event.model_dump(mode="json", by_alias=True)
            parsed = parse_event(payload)
            assert parsed.model_dump(mode="json", by_alias=True) == payload
            seen.add(str(payload["event_type"]))
        missing = set(module.publishes) - seen
        assert not missing, (module.module_id, missing)


def test_off_mode_drops_module_and_its_events() -> None:
    present = run_scenario(panic(1), 8)
    absent = run_scenario(panic(1), 8, modes={"M14": ModuleMode.OFF})
    engine = build_engine(modes={"M14": ModuleMode.OFF}, scenario=panic(1))
    assert "M14" not in engine
    assert any(event.event_type == "PriceTick" for event in present)
    assert all(event.event_type not in {"PriceTick", "TradingHalt"} for event in absent)


def test_panic_chain_order_and_attention_rises() -> None:
    log = run_scenario(panic(1), int(crisis_sim_minutes()))
    types = [event.event_type for event in log]
    assert _subsequence(types, CHAIN)
    rumor_at = types.index("RumorPosted")
    attention = next(
        event
        for event in log[rumor_at:]
        if isinstance(event, AttentionTick)
    )
    assert attention.delta_4h > 0.0
    assert attention.level > 0.2
    capital = next(event for event in log if isinstance(event, CapitalChanged))
    assert 0.0 <= capital.cet1_ratio < 0.12
    for event in log:
        if isinstance(event, WithdrawalRequested):
            assert event.amount > 0.0


def test_only_section_61_event_types() -> None:
    log = run_scenario(panic(3), 16)
    found = {event.event_type for event in log}
    assert found <= SECTION_61


def test_real_mode_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="M29"):
        build_engine(modes={"M29": ModuleMode.REAL})
    from agents.vip import VIPModule

    engine = build_engine(modes={"M41": ModuleMode.REAL})
    assert isinstance(engine["M41"], VIPModule)


def test_population_uses_param_accessor_and_community_does_not_run() -> None:
    engine = build_engine(scenario=panic(1))
    engine.run(1)
    population = engine["M05"]
    assert isinstance(population, PopulationFake)
    assert population.insurance_limit == get_param("Insurance limit")
    top_each = get_param("SVB-like: top-10 depositor balance") / 10.0
    assert population.example_top_insured == min(top_each, population.insurance_limit)

    quiet = Scenario(
        preset="insured-community",
        seed=1,
        rumors=[
            Rumor(
                id="rumor-1",
                claim="A rumor about Fictional Community Bank",
                credibility=0.9,
                reach=1000,
            )
        ],
    )
    log = run_scenario(quiet, 4)
    assert not any(event.event_type == "WithdrawalRequested" for event in log)


def test_crowd_and_market_snapshots_follow_the_log() -> None:
    engine = build_engine(scenario=panic(1))
    engine.run(6)
    crowd = engine["M29"]
    screen = engine["M30"]
    assert isinstance(crowd, CrowdFake)
    assert isinstance(screen, MarketScreenFake)
    assert crowd.withdrew > 0
    assert crowd.rumor
    assert screen.prices
    assert screen.futures
