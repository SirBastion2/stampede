"""M04 media and analyst agents: thresholds, citations, real vs fake."""

from __future__ import annotations

from agents.media import (
    MediaModule,
    MediaThresholds,
    RuleBasedAnalystProvider,
    RuleBasedHeadlineProvider,
    attention_crossed,
    load_media_thresholds,
    loss_surprises,
    price_move_crossed,
)
from contracts.events import AnalystNote, AttentionTick, HeadlinePublished, LossRealized
from contracts.schemas import Scenario
from engine.modes import ModuleMode
from engine.registry import build_engine
from mocks.bus import EventBus
from mocks.l5_grok import MediaFake
from mocks.support import ASSUMED_HEADLINE_ATTENTION, bank_id_for, bank_name_for


def _scenario(seed: int = 3) -> Scenario:
    return Scenario(preset="tech-concentrated", seed=seed, rate_path=[(0.0, 0.05)])


def test_load_media_thresholds_uses_assumed_rows() -> None:
    thresholds = load_media_thresholds()
    assert thresholds.attention == ASSUMED_HEADLINE_ATTENTION
    assert thresholds.price_move_fraction > 0.0
    assert thresholds.loss_surprise_amount == 0.0
    assert thresholds.max_latency_minutes == 2.0


def test_threshold_helpers() -> None:
    assert attention_crossed(0.2, 0.35, 0.30)
    assert not attention_crossed(0.35, 0.40, 0.30)
    assert price_move_crossed(100.0, 94.0, 0.05)
    assert not price_move_crossed(100.0, 97.0, 0.05)
    assert loss_surprises(1.0, 0.0)
    assert not loss_surprises(0.0, 1.0)


def test_no_headline_without_triggering_event() -> None:
    bus = EventBus()
    module = MediaModule("M04", thresholds=load_media_thresholds())
    module.attach(bus, _scenario())
    assert module.step(0.0) == []
    assert module.step(1.0) == []


def test_real_matches_fake_on_attention_and_loss() -> None:
    scenario = _scenario(9)
    bus = EventBus()
    fake = MediaFake("M04")
    fake.attach(bus, scenario)
    attention = AttentionTick(
        t=1.0,
        seed=scenario.seed,
        bank_id=bank_id_for(scenario.preset),
        cause_id="attn-1",
        level=ASSUMED_HEADLINE_ATTENTION + 0.05,
        delta_4h=0.1,
    )
    loss = LossRealized(
        t=2.0,
        seed=scenario.seed,
        bank_id=bank_id_for(scenario.preset),
        cause_id="loss-1",
        amount=1_000_000.0,
        kind="securities",
    )
    fake_out = fake.on_tick(1.0, [attention]) + fake.on_tick(2.0, [loss])

    real = MediaModule(
        "M04",
        thresholds=load_media_thresholds(),
        headline_provider=RuleBasedHeadlineProvider(),
        analyst_provider=RuleBasedAnalystProvider(),
    )
    real.attach(bus, scenario)
    real_out = real._on_tick(1.0, [attention]) + real._on_tick(2.0, [loss])

    fake_headlines = [event for event in fake_out if isinstance(event, HeadlinePublished)]
    real_headlines = [event for event in real_out if isinstance(event, HeadlinePublished)]
    assert len(fake_headlines) == len(real_headlines) == 2
    name = bank_name_for(scenario.preset)
    assert fake_headlines[0].text == real_headlines[0].text == f"Depositors focus on {name}"
    loss_headline = f"{name} books a loss on securities it sold"
    assert fake_headlines[1].text == real_headlines[1].text == loss_headline
    assert fake_headlines[0].reach == real_headlines[0].reach

    fake_notes = [event for event in fake_out if isinstance(event, AnalystNote)]
    real_notes = [event for event in real_out if isinstance(event, AnalystNote)]
    assert len(fake_notes) == len(real_notes) == 1
    assert real_notes[0].cited_event_ids == ["loss-1"]


def test_build_engine_m04_real_does_not_raise() -> None:
    build_engine(modes={"M04": ModuleMode.REAL}, scenario=_scenario())


def test_analyst_note_cites_trigger_cause_id() -> None:
    scenario = _scenario(1)
    module = MediaModule(
        "M04",
        thresholds=MediaThresholds(
            attention=1.0,
            price_move_fraction=1.0,
            loss_surprise_amount=0.0,
            max_latency_minutes=2.0,
            reach_base=8000,
            attention_assumed=True,
            price_assumed=True,
            loss_assumed=True,
        ),
        headline_provider=RuleBasedHeadlineProvider(),
        analyst_provider=RuleBasedAnalystProvider(),
    )
    module.attach(EventBus(), scenario)
    loss = LossRealized(
        t=0.0,
        seed=scenario.seed,
        bank_id=bank_id_for(scenario.preset),
        cause_id="cause-loss-42",
        amount=10.0,
        kind="securities",
    )
    events = module._on_tick(0.0, [loss])
    notes = [event for event in events if isinstance(event, AnalystNote)]
    assert len(notes) == 1
    assert "cause-loss-42" in notes[0].cited_event_ids
