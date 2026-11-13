"""M44 short sellers: score, offline thesis cache, attention, short interest."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import BaseModel

from agents.guardrails import statement_allowed
from agents.llm_cache import LLMCache, offline_enabled
from agents.shorts import (
    AUTHOR_ID,
    ShortSellerModule,
    load_short_params,
    metrics_from_events,
    short_order_for,
    thesis_reach,
    thesis_text,
    vulnerability_score,
)
from contracts import parse_event
from contracts.events import AttentionTick, ThesisPublished
from contracts.schemas import Rumor, Scenario, ScenarioShock
from engine.modes import ModuleMode
from engine.registry import build_engine
from mocks.bus import EventBus
from mocks.l3_markets import MarketFake
from mocks.l5_grok import ShortFake
from params.loader import get_parameter


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


def _attention(level: float, seed: int = 1) -> AttentionTick:
    return AttentionTick(
        t=0.0,
        seed=seed,
        bank_id="bank-fvb",
        level=level,
        delta_4h=level,
    )


def test_threshold_row_is_assumed_and_tbd_row_stays_unset() -> None:
    assumed = get_parameter("M44 assumed vulnerability publish threshold")
    assert assumed.status == "ASSUMED"
    assert assumed.value == "0.2"
    unset = get_parameter("Short-seller vulnerability publish threshold")
    assert unset.status == "UNSET"
    assert unset.value == "TBD"
    assert load_short_params().threshold == pytest.approx(0.2)


def test_capital_stress_bump_is_the_registry_value() -> None:
    params = load_short_params()
    calm = metrics_from_events([], "tech-concentrated", params)
    stressed_cet1 = params.reference_cet1 - 0.01
    stressed = metrics_from_events([], "tech-concentrated", params)
    stressed = replace(stressed, cet1_ratio=stressed_cet1)
    assert vulnerability_score(stressed, params) == pytest.approx(
        vulnerability_score(calm, params) + params.capital_stress_bump
    )


def test_score_uses_registry_threshold_not_a_literal_gate() -> None:
    params = load_short_params()
    quiet = metrics_from_events([], "tech-concentrated", params)
    assert vulnerability_score(quiet, params) < params.threshold
    hot = metrics_from_events([_attention(0.9)], "tech-concentrated", params)
    assert vulnerability_score(hot, params) >= params.threshold
    raised = replace(params, threshold=10.0)
    assert vulnerability_score(hot, raised) < raised.threshold


def test_below_threshold_publishes_nothing() -> None:
    params = replace(load_short_params(), threshold=10.0)
    module = ShortSellerModule("M44", params=params)
    bus = EventBus()
    module.attach(bus, _panic(1))
    bus.publish(_attention(0.9))
    assert module.step(1.0) == []
    assert module.short_orders == []


def test_above_threshold_publishes_thesis_and_records_short_order() -> None:
    module = ShortSellerModule("M44")
    bus = EventBus()
    module.attach(bus, _panic(3))
    bus.publish(_attention(0.9, seed=3))
    out = module.step(1.0)
    assert len(out) == 1
    event = out[0]
    assert isinstance(event, ThesisPublished)
    assert event.author_id == AUTHOR_ID
    assert event.bank_id == "bank-fvb"
    assert event.reach == thesis_reach(3)
    payload = event.model_dump(mode="json")
    assert parse_event(payload).model_dump(mode="json") == payload
    assert module.short_orders[0].quantity == load_short_params().short_quantity
    assert module.theses[0].text
    assert statement_allowed(module.theses[0].text)
    assert "Fictional Valley Bank" in module.theses[0].text
    assert module.step(2.0) == []


def test_offline_default_skips_live_and_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STAMPEDE_OFFLINE", raising=False)
    assert offline_enabled() is True

    def _boom(model: str, prompt: str) -> str:
        del model, prompt
        raise AssertionError("live Grok call")

    monkeypatch.setattr("agents.llm_cache.live_grok_text", _boom)
    params = load_short_params()
    metrics = metrics_from_events([_attention(0.5)], "tech-concentrated", params)
    cache = LLMCache()
    first = thesis_text(
        metrics,
        seed=1,
        bank_name="Fictional Valley Bank",
        score=0.4,
        params=params,
        cache=cache,
    )
    second = thesis_text(
        metrics,
        seed=1,
        bank_name="Fictional Valley Bank",
        score=0.4,
        params=params,
        cache=cache,
    )
    assert first == second
    assert cache.hits == 1
    assert cache.misses == 1
    assert statement_allowed(first)


def test_sqlite_cache_reloads(tmp_path: Path) -> None:
    params = load_short_params()
    metrics = metrics_from_events([_attention(0.4)], "insured-community", params)
    path = tmp_path / "llm.sqlite"
    cache = LLMCache(path)
    text = thesis_text(
        metrics,
        seed=2,
        bank_name="Fictional Community Bank",
        score=0.1,
        params=params,
        cache=cache,
    )
    cache.close()
    again = LLMCache(path)
    loaded = thesis_text(
        metrics,
        seed=2,
        bank_name="Fictional Community Bank",
        score=0.1,
        params=params,
        cache=again,
    )
    assert loaded == text
    assert again.hits == 1
    again.close()


def test_blocked_live_text_is_not_published(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAMPEDE_OFFLINE", "0")

    def _live(model: str, prompt: str) -> str:
        del model, prompt
        return "Silicon Valley Bank is finished"

    monkeypatch.setattr("agents.llm_cache.live_grok_text", _live)
    params = load_short_params()
    metrics = metrics_from_events([_attention(0.8)], "tech-concentrated", params)
    text = thesis_text(
        metrics,
        seed=4,
        bank_name="Fictional Valley Bank",
        score=0.7,
        params=params,
        cache=LLMCache(),
    )
    assert "Silicon Valley" not in text
    assert statement_allowed(text)
    order = short_order_for(metrics, author_id=AUTHOR_ID, bank_id="bank-fvb", params=params)
    assert order.quantity == params.short_quantity


def test_real_mode_thesis_raises_attention_and_short_interest() -> None:
    scenario = _panic(1)
    real = build_engine(modes={"M44": ModuleMode.REAL}, scenario=scenario)
    off = build_engine(modes={"M44": ModuleMode.OFF}, scenario=scenario)
    assert isinstance(real["M44"], ShortSellerModule)
    assert isinstance(build_engine(scenario=scenario)["M44"], ShortFake)
    real_log = real.run(4)
    off_log = off.run(4)
    real_theses = [event for event in real_log if isinstance(event, ThesisPublished)]
    assert len(real_theses) == 1
    assert all(event.event_type != "ThesisPublished" for event in off_log)

    def _level_at(log: list[BaseModel], t: float) -> float:
        for event in log:
            if isinstance(event, AttentionTick) and event.t == t:
                return event.level
        raise AssertionError(f"no attention at {t}")

    assert _level_at(real_log, 1.0) > _level_at(off_log, 1.0)
    real_book = real["M14"]
    off_book = off["M14"]
    assert isinstance(real_book, MarketFake)
    assert isinstance(off_book, MarketFake)
    assert real_book.short_interest > off_book.short_interest
    assert real_book.short_interest > 0.0


def _dump(log: list[BaseModel]) -> list[dict[str, object]]:
    return [event.model_dump(mode="json") for event in log]


def test_real_run_is_deterministic_and_seeds_diverge() -> None:
    first = build_engine(modes={"M44": ModuleMode.REAL}, scenario=_panic(1)).run(4)
    second = build_engine(modes={"M44": ModuleMode.REAL}, scenario=_panic(1)).run(4)
    other = build_engine(modes={"M44": ModuleMode.REAL}, scenario=_panic(2)).run(4)
    assert _dump(first) == _dump(second)
    assert _dump(first) != _dump(other)
    reach_one = next(event.reach for event in first if isinstance(event, ThesisPublished))
    reach_two = next(event.reach for event in other if isinstance(event, ThesisPublished))
    assert reach_one != reach_two
