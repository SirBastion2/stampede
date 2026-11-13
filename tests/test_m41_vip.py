"""M41 VIP agents: schema, reduced rule, offline batch, feed line, real mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agents.guardrails import bank_name_allowed, statement_allowed
from agents.llm_cache import LLMCache
from agents.vip import (
    AdaptiveSemaphore,
    ExposureContext,
    ReducedRule,
    RuleBasedVipProvider,
    VIPModule,
    VipRateLimited,
    allocate_balances,
    batched_cards,
    check_exposure,
    complete_vip_json,
    format_feed_line,
    live_grok_enabled,
    load_vip_config,
    message_group,
    move_funds,
    persona_cards,
    post,
    reduce_decision,
    replay_reduced_rule,
    rules_for_futures,
    validate_vip_batch,
)
from contracts.events import AttentionTick, DirectiveIssued, VIPDecision
from contracts.schemas import Scenario
from engine.modes import ModuleMode
from engine.registry import build_engine
from mocks.l5_grok import VIPFake

_ACCEPTANCE = "Founder of Kestrel Labs moved $40M"


class _CountingProvider:
    def __init__(self) -> None:
        self.calls = 0
        self._rule = RuleBasedVipProvider()

    def complete_json(self, prompt: str) -> dict:
        self.calls += 1
        return self._rule.complete_json(prompt)


class _RateLimitProvider:
    def complete_json(self, prompt: str) -> dict:
        raise VipRateLimited("429")


class _InvalidProvider:
    def complete_json(self, prompt: str) -> dict:
        return {"decisions": []}


def _module(provider: Any = None) -> VIPModule:
    return VIPModule("M41", provider=provider)


def _arm(module: VIPModule, preset: str = "tech-concentrated", seed: int = 11) -> None:
    module.scenario = Scenario(preset=preset, seed=seed)
    module.note_trigger("test")


def test_validate_vip_batch_rejects_unknown_action() -> None:
    with pytest.raises(ValueError, match="invalid VIP batch JSON"):
        validate_vip_batch(
            {
                "decisions": [
                    {
                        "agent_id": "vip-1",
                        "action": "fly",
                        "amount": 1,
                        "rationale": "nope",
                        "tool": "post",
                    }
                ]
            }
        )


def test_validate_vip_batch_rejects_tool_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        validate_vip_batch(
            {
                "decisions": [
                    {
                        "agent_id": "vip-1",
                        "action": "withdraw",
                        "amount": 1,
                        "rationale": "move",
                        "tool": "post",
                    }
                ]
            }
        )


def test_reduced_rule_reproduces_withdraw_and_stay() -> None:
    config = load_vip_config()
    features = {
        "attention": 0.4,
        "uninsured_gap": 0.93,
        "belief": config.baseline_belief,
        "stock_return": 0.0,
    }
    for action, tool, amount in (
        ("withdraw", "move_funds", config.example_amount),
        ("stay", "check_exposure", 0.0),
    ):
        decision = validate_vip_batch(
            {
                "decisions": [
                    {
                        "agent_id": "vip-1",
                        "action": action,
                        "amount": amount,
                        "rationale": "Stays insured." if action == "stay" else "Moving cash.",
                        "tool": tool,
                    }
                ]
            }
        ).decisions[0]
        rule = reduce_decision(
            decision,
            features,
            sensitivities=config.sensitivities,
            margin=config.margin,
        )
        assert replay_reduced_rule(rule, features) == action
        restored = ReducedRule.model_validate(rules_for_futures([rule])[0])
        assert replay_reduced_rule(restored, features) == action


def test_batches_are_ten_then_remainder() -> None:
    config = load_vip_config()
    groups = batched_cards(persona_cards(), config.batch_size)
    assert [len(group) for group in groups] == [10, 5]
    provider = _CountingProvider()
    module = _module(provider)
    _arm(module)
    module.on_tick(1.0, [])
    assert provider.calls == 2
    assert module.gate.limit == config.semaphore_initial + 2


def test_top10_balances_sum_to_registry_total() -> None:
    config = load_vip_config()
    balances = allocate_balances(config)
    assert len(balances) == config.agent_count == 15
    assert abs(sum(balances[:10]) - config.top10_balance) < 1.0


def test_feed_line_matches_acceptance_example() -> None:
    config = load_vip_config()
    line = format_feed_line("Founder of Kestrel Labs", "withdraw", config.example_amount)
    assert line == _ACCEPTANCE


def test_persona_names_stay_fictional() -> None:
    for card in persona_cards():
        assert bank_name_allowed(card.company)
        assert statement_allowed(card.display_name)
        assert statement_allowed(card.company)


def test_four_tools_return_structured_results() -> None:
    card = persona_cards(1)[0]
    context = ExposureContext(balance=80_000_000, insured=250_000, attention=0.2)
    view = check_exposure(card, context)
    assert view.agent_id == "vip-1"
    assert view.uninsured_gap == pytest.approx((80_000_000 - 250_000) / 80_000_000)
    moved = move_funds(card, config_amount())
    assert moved.destination == "external"
    assert moved.amount == config_amount()
    posted = post(card, "Silicon Valley Bank is fine")
    assert statement_allowed(posted.text)
    assert "Silicon" not in posted.text
    vc = next(item for item in persona_cards() if item.role == "vc")
    sent = message_group(vc, "Portfolio companies should move uninsured cash.")
    assert sent.portfolio_ids == vc.portfolio_ids
    assert statement_allowed(sent.text)


def config_amount() -> float:
    return load_vip_config().example_amount


def test_memory_keeps_last_five_events() -> None:
    module = _module()
    module.scenario = Scenario(preset="tech-concentrated", seed=3)
    depth = module.config.memory_depth
    for minute in range(1, depth + 2):
        module.note_trigger(f"tick-{minute}")
        module.on_tick(float(minute), [])
    assert len(module.memory["vip-1"].events) == depth == 5


def test_attention_cross_triggers_before_cadence() -> None:
    module = _module()
    module.scenario = Scenario(preset="tech-concentrated", seed=4)
    assert module.on_tick(1.0, [_attention(1.0, 0.1)]) == []
    events = module.on_tick(2.0, [_attention(2.0, 0.5)])
    assert any(isinstance(event, VIPDecision) for event in events)
    assert _ACCEPTANCE in module.feed


def test_cadence_fires_on_interval_only() -> None:
    module = _module()
    module.scenario = Scenario(preset="tech-concentrated", seed=5)
    assert module.on_tick(29.0, []) == []
    events = module.on_tick(30.0, [])
    assert any(isinstance(event, VIPDecision) and event.agent_id == "vip-1" for event in events)


def test_community_preset_stays() -> None:
    module = _module()
    _arm(module, preset="insured-community")
    events = module.on_tick(1.0, [])
    decisions = [event for event in events if isinstance(event, VIPDecision)]
    assert decisions
    assert all(event.action == "stay" and event.amount == 0.0 for event in decisions)
    assert all("moved" not in line for line in module.feed)


def test_tech_preset_moves_kestrel_and_directs_vcs() -> None:
    module = _module()
    _arm(module)
    events = module.on_tick(1.0, [])
    decisions = [event for event in events if isinstance(event, VIPDecision)]
    directives = [event for event in events if isinstance(event, DirectiveIssued)]
    assert len(decisions) == 15
    kestrel = next(event for event in decisions if event.agent_id == "vip-1")
    assert kestrel.action == "withdraw"
    assert kestrel.amount == load_vip_config().example_amount
    assert kestrel.rationale
    assert directives
    assert all(event.action == "withdraw" and event.portfolio_ids for event in directives)
    assert _ACCEPTANCE in module.feed
    for rule in module.rules.values():
        assert replay_reduced_rule(rule, rule.features) == rule.action


def test_same_seed_same_decisions() -> None:
    first = _dump(9)
    second = _dump(9)
    assert first == second


def test_invalid_reply_falls_back_to_rule() -> None:
    module = _module(_InvalidProvider())
    _arm(module)
    events = module.on_tick(1.0, [])
    assert any(isinstance(event, VIPDecision) for event in events)
    assert _ACCEPTANCE in module.feed
    assert module.gate.limit == module.config.semaphore_initial


def test_rate_limit_halves_semaphore_and_falls_back() -> None:
    module = _module(_RateLimitProvider())
    _arm(module)
    module.on_tick(1.0, [])
    assert module.gate.limit == module.config.semaphore_initial // 4
    assert _ACCEPTANCE in module.feed


def test_adaptive_semaphore_policy() -> None:
    gate = AdaptiveSemaphore(32)
    gate.on_rate_limit()
    assert gate.limit == 16
    gate.on_success()
    assert gate.limit == 17


def test_llm_cache_avoids_second_provider_call(tmp_path: Path) -> None:
    cache = LLMCache(tmp_path / "vip.sqlite3")
    calls = 0

    class _Stub:
        def complete_json(self, prompt: str) -> dict:
            nonlocal calls
            calls += 1
            return {
                "decisions": [
                    {
                        "agent_id": "vip-1",
                        "action": "stay",
                        "amount": 0,
                        "rationale": "Insured balance stays.",
                        "tool": "check_exposure",
                    }
                ]
            }

    prompt = '{"task":"vip_batch"}'
    first = complete_vip_json(
        prompt=prompt,
        model="grok-4.1-fast",
        provider=_Stub(),
        cache=cache,
        expected_ids=("vip-1",),
    )
    second = complete_vip_json(
        prompt=prompt,
        model="grok-4.1-fast",
        provider=_Stub(),
        cache=cache,
        expected_ids=("vip-1",),
    )
    assert first == second
    assert calls == 1
    cache.close()


def test_live_grok_stays_off_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("XAI_API_KEY", "should-not-be-used")
    assert live_grok_enabled() is False
    monkeypatch.setenv("STAMPEDE_OFFLINE", "0")
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    assert live_grok_enabled() is False


def test_scheduler_consumes_tick_only() -> None:
    assert VIPModule.consumes == ("Tick",)


def test_fake_stays_default_and_real_mode_feeds_kestrel() -> None:
    scenario = Scenario(preset="tech-concentrated", seed=11)
    fake_engine = build_engine(scenario=scenario)
    assert isinstance(fake_engine["M41"], VIPFake)
    engine = build_engine(modes={"M41": ModuleMode.REAL}, scenario=scenario)
    module = engine["M41"]
    assert isinstance(module, VIPModule)
    log = engine.run(31)
    assert any(isinstance(event, VIPDecision) for event in log)
    assert any(isinstance(event, DirectiveIssued) for event in log)
    assert _ACCEPTANCE in module.feed


def _attention(t: float, level: float) -> AttentionTick:
    return AttentionTick(
        t=t,
        seed=1,
        bank_id="bank-fvb",
        level=level,
        delta_4h=level,
    )


def _dump(seed: int) -> list[dict]:
    module = _module()
    _arm(module, seed=seed)
    events = module.on_tick(1.0, [])
    return [event.model_dump() for event in events]
