"""M42 bank CEO agents: ledger refusal, offline rule, cache, real mode."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agents.ceo import (
    BankSnapshot,
    CEODecision,
    CEOModule,
    RuleBasedCEOProvider,
    _GrokCEOProvider,
    bank_snapshot,
    choose_ceo_action,
    ledger_allows_action,
    ledger_allows_deposit_rate,
    ledger_state_from_events,
    load_ceo_policy,
    map_tool_call,
    offline_ceo_decision,
    reduce_ceo_decision,
    replay_reduced_rule,
)
from agents.llm_cache import LLMCache
from contracts.events import AttentionTick, CEOAction, DepositRateChanged, PaymentSettled
from contracts.schemas import Scenario
from engine.bank.ledger import initial_ledger_state, registry_deposits_total
from engine.bank.management import (
    MANAGEMENT_ACTIONS,
    capital_raise_success_probability,
    load_management_policy,
)
from engine.modes import ModuleMode
from engine.registry import build_engine
from mocks.bus import EventBus
from mocks.l5_grok import CEOFake


def _scenario(seed: int = 11) -> Scenario:
    return Scenario(preset="tech-concentrated", seed=seed, rate_path=[(0.0, 0.05)])


def _snapshot(**overrides: object) -> BankSnapshot:
    base = BankSnapshot(
        preset="tech-concentrated",
        seed=11,
        bank_id="bank-fvb",
        t=1.0,
        cash=1.0e9,
        cet1=0.12,
        securities=1.0e9,
        deposits=1.0e11,
        attention=0.0,
        stock_return=0.0,
        policy_rate=0.05,
        loss_hit=False,
        hit_names=("AttentionTick",),
    )
    return replace(base, **overrides)


def _attention(level: float, scenario: Scenario) -> AttentionTick:
    return AttentionTick(
        t=1.0,
        seed=scenario.seed,
        bank_id="bank-fvb",
        cause_id="attn-1",
        level=level,
        delta_4h=level,
    )


def test_load_ceo_policy_fleet_and_flag() -> None:
    policy = load_ceo_policy()
    assert policy.ceo_count == 2
    assert policy.grok_enabled is False
    assert policy.deposit_segment == "startup"
    assert policy.attention_trigger > 0.0


def test_impossible_sale_and_borrow_are_refused() -> None:
    state = initial_ledger_state(deposits=registry_deposits_total())
    sale = ledger_allows_action(
        state,
        "sell_assets",
        {"amount": state.securities * 10, "price_impact": 0.1},
    )
    assert sale.allowed is False
    assert "securities" in sale.reason
    exact = ledger_allows_action(
        state,
        "sell_assets",
        {"amount": state.securities, "price_impact": 0.0},
    )
    assert exact.allowed is True
    borrow = ledger_allows_action(state, "borrow_facility", {"amount": state.securities * 10})
    assert borrow.allowed is False
    assert "collateral" in borrow.reason


def test_deposit_rate_outside_cash_or_cap_is_refused() -> None:
    state = initial_ledger_state(deposits=registry_deposits_total())
    policy = load_ceo_policy()
    above_cap = ledger_allows_deposit_rate(state, policy.deposit_rate_cap + 1.0, policy=policy)
    assert above_cap.allowed is False
    at_cap = ledger_allows_deposit_rate(state, policy.deposit_rate_cap, policy=policy)
    assert at_cap.allowed is False
    assert "cash" in at_cap.reason
    affordable = ledger_allows_deposit_rate(state, policy.deposit_rate, policy=policy)
    assert affordable.allowed is True


def test_unknown_tool_is_not_on_the_menu() -> None:
    decision = map_tool_call({"action": "wire_the_treasury", "params": {}, "rationale": "no"})
    assert decision.on_menu is False
    assert decision.action not in MANAGEMENT_ACTIONS
    mapped = map_tool_call(
        {"action": "issue_statement", "params": {"segment": "startup"}, "rationale": "calm"}
    )
    assert mapped.on_menu is True
    assert mapped.action in MANAGEMENT_ACTIONS


def test_offline_rule_follows_management_policy_and_is_deterministic() -> None:
    management = load_management_policy()
    stressed = _snapshot(cet1=management.capital_raise_cet1_threshold - 0.01, attention=0.4)
    first = offline_ceo_decision(stressed)
    second = offline_ceo_decision(stressed)
    assert first == second
    assert first.action == "announce_capital_raise"
    gap = max(0.0, management.capital_raise_cet1_threshold - stressed.cet1)
    expected = capital_raise_success_probability(stressed.stock_return, stressed.attention, gap)
    assert first.params["probability"] == expected

    dividend = _snapshot(cet1=management.dividend_cut_cet1_threshold - 0.001)
    assert offline_ceo_decision(dividend).action == "cut_dividend"

    borrow = _snapshot(cash=management.cash_borrow_threshold - 1.0)
    borrowed = offline_ceo_decision(borrow)
    assert borrowed.action == "borrow_facility"
    assert float(borrowed.params["amount"]) <= borrow.securities

    slide = _snapshot(stock_return=load_ceo_policy().stock_slide_trigger - 0.01)
    sold = offline_ceo_decision(slide)
    assert sold.action == "sell_assets"
    assert float(sold.params["amount"]) <= slide.securities

    calm = _snapshot(attention=load_ceo_policy().attention_trigger)
    assert offline_ceo_decision(calm).action == "issue_statement"


def test_replay_uses_reduced_rule_without_provider() -> None:
    decision = offline_ceo_decision(_snapshot(attention=0.5))
    rule = reduce_ceo_decision(decision)
    policy = load_ceo_policy()
    assert rule.sensitivities["attention"] == policy.sensitivity_attention
    assert rule.sensitivities["cet1"] == policy.sensitivity_cet1
    quiet = replay_reduced_rule(rule, attention=0.0, cet1=0.12, stock_return=0.0, cash=1.0)
    assert quiet is None
    fired = replay_reduced_rule(
        rule,
        attention=policy.attention_trigger,
        cet1=0.12,
        stock_return=0.0,
        cash=1.0,
    )
    assert fired == decision.action


def test_ledger_replay_applies_payment() -> None:
    opening = initial_ledger_state(deposits=registry_deposits_total())
    paid = PaymentSettled(
        t=1.0,
        seed=1,
        bank_id="bank-fvb",
        cause_id="pay-1",
        amount=1_000.0,
        channel="wire",
        queued_remaining=0,
    )
    state = ledger_state_from_events([paid], bank_id="bank-fvb")
    assert state.cash == pytest.approx(opening.cash - 1_000.0)


def test_module_refuses_impossible_sale_and_logs_it() -> None:
    scenario = _scenario()

    class _Greedy:
        def choose(self, snapshot: BankSnapshot) -> CEODecision:
            return CEODecision(
                action="sell_assets",
                params={"amount": snapshot.securities * 10, "price_impact": 0.2},
                rationale="liquidate the book",
                deposit_segment="startup",
                deposit_rate=0.02,
                on_menu=True,
                menu_reason="",
            )

    module = CEOModule("M42", provider=_Greedy())
    module.attach(EventBus(), scenario)
    out = module.on_tick(1.0, [_attention(0.9, scenario)])
    assert out == []
    log = module.log_by_bank["bank-fvb"]
    assert log
    assert log[0].accepted is False
    assert log[0].bank_id == "bank-fvb"
    assert "securities" in log[0].refusal_reason


def test_unaffordable_rate_is_refused_and_menu_action_publishes() -> None:
    scenario = _scenario()

    class _Pricey:
        def choose(self, snapshot: BankSnapshot) -> CEODecision:
            del snapshot
            return CEODecision(
                action="issue_statement",
                params={"segment": "startup"},
                rationale="pay up for deposits",
                deposit_segment="startup",
                deposit_rate=1.0,
                on_menu=True,
                menu_reason="",
            )

    module = CEOModule("M42", provider=_Pricey())
    module.attach(EventBus(), scenario)
    out = module.on_tick(1.0, [_attention(0.9, scenario)])
    assert len(out) == 1
    assert isinstance(out[0], CEOAction)
    assert out[0].action == "issue_statement"
    log = module.log_by_bank["bank-fvb"]
    assert any("deposit rate" in entry.refusal_reason for entry in log)
    assert any(entry.accepted and entry.action == "issue_statement" for entry in log)


def test_real_module_logs_and_publishes_when_attention_hits() -> None:
    scenario = _scenario(7)
    module = CEOModule("M42", provider=RuleBasedCEOProvider())
    bus = EventBus()
    module.attach(bus, scenario)
    bus.publish(_attention(0.8, scenario))
    first = module.step(1.0)
    second = module.step(2.0)
    assert second == []
    kinds = [type(event) for event in first]
    assert kinds == [DepositRateChanged, CEOAction]
    rate = first[0]
    action = first[1]
    assert isinstance(rate, DepositRateChanged)
    assert isinstance(action, CEOAction)
    assert rate.segment == load_ceo_policy().deposit_segment
    assert action.action in MANAGEMENT_ACTIONS
    assert action.action == "issue_statement"
    log = module.log_by_bank["bank-fvb"]
    assert log and log[-1].accepted is True
    assert log[-1].rationale
    rule = module.reduced_rules["bank-fvb"]
    assert rule.action == action.action
    again = bank_snapshot(
        bus.log,
        [_attention(0.8, scenario)],
        preset=scenario.preset,
        seed=scenario.seed,
        bank_id="bank-fvb",
        t=1.0,
        scenario=scenario,
    )
    assert again.attention == 0.8


def test_calm_attention_does_not_call_the_provider() -> None:
    scenario = _scenario()
    calls = 0

    class _Counting(RuleBasedCEOProvider):
        def choose(self, snapshot: BankSnapshot) -> CEODecision:
            nonlocal calls
            calls += 1
            return super().choose(snapshot)

    module = CEOModule("M42", provider=_Counting())
    module.attach(EventBus(), scenario)
    assert module.on_tick(1.0, [_attention(0.01, scenario)]) == []
    assert calls == 0
    assert module.log_by_bank == {}


def test_offline_choose_never_calls_grok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("XAI_API_KEY", "test-key-not-used")

    def _boom(prompt: str, model: str, **kwargs: object) -> dict[str, object]:
        del prompt, model, kwargs
        raise AssertionError("live Grok call")

    monkeypatch.setattr("agents.llm_cache.complete_json_llm", _boom)
    decision = choose_ceo_action(_snapshot(attention=0.6))
    assert decision.action in MANAGEMENT_ACTIONS


def test_grok_provider_caches_and_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    payload = {
        "action": "call_top_clients",
        "params": {"segment": "startup"},
        "rationale": "call the largest depositors",
        "deposit_segment": "startup",
        "deposit_rate": 0.02,
    }

    def _fake(prompt: str, model: str, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        del prompt, model, kwargs
        calls += 1
        return payload

    monkeypatch.setattr("agents.llm_cache.complete_json_llm", _fake)
    cache = LLMCache(tmp_path / "ceo.sqlite3")
    provider = _GrokCEOProvider("grok-4.1-fast", cache)
    snap = _snapshot(attention=0.6)
    first = provider.choose(snap)
    second = provider.choose(snap)
    assert first == second
    assert first.action == "call_top_clients"
    assert calls == 1

    def _bad(prompt: str, model: str, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        del prompt, model, kwargs
        calls += 1
        return {"action": "wire_the_treasury", "rationale": "no"}

    monkeypatch.setattr("agents.llm_cache.complete_json_llm", _bad)
    other = _GrokCEOProvider("grok-4.1-fast", LLMCache(tmp_path / "other.sqlite3"))
    fallen = other.choose(_snapshot())
    assert fallen.on_menu is True
    assert fallen.action in MANAGEMENT_ACTIONS
    cache.close()


def test_build_engine_keeps_fake_default_and_wires_real() -> None:
    scenario = _scenario(3)
    fake_engine = build_engine(scenario=scenario)
    assert isinstance(fake_engine["M42"], CEOFake)
    engine = build_engine(modes={"M42": ModuleMode.REAL}, scenario=scenario)
    module = engine["M42"]
    assert isinstance(module, CEOModule)
    engine.bus.publish(_attention(0.95, scenario))
    published = module.step(4.0)
    assert any(isinstance(event, CEOAction) for event in published)
    assert module.log_by_bank[module.bank_id]
