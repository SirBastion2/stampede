"""M25 persona panel: E11 direction, cache, schema, real mode."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.llm_cache import LLMCache, cached_json_completion, prompt_hash
from agents.personas import (
    PersonaPanelModule,
    RuleBasedPersonaProvider,
    personalization_strength,
    score_statement,
    scores_for_statement_event,
    validate_persona_batch,
)
from contracts.events import StatementIssued
from contracts.schemas import Scenario
from engine.modes import ModuleMode
from engine.registry import build_engine
from mocks.l5_grok import PersonaFake


class _RecordingProvider(RuleBasedPersonaProvider):
    calls = 0

    def score(self, **kwargs):  # type: ignore[no-untyped-def]
        _RecordingProvider.calls += 1
        return super().score(**kwargs)


def test_e11_personalized_reassurance_beats_generic() -> None:
    provider = RuleBasedPersonaProvider()
    generic = score_statement(
        preset="tech-concentrated",
        seed=11,
        stmt_id="stmt-generic",
        channel="press_release",
        segments=["startup", "treasurer"],
        text="The bank is well capitalized and operations continue as normal.",
        provider=provider,
    )
    personal = score_statement(
        preset="tech-concentrated",
        seed=11,
        stmt_id="stmt-personal",
        channel="direct_call",
        segments=["startup", "treasurer"],
        text=(
            "Your startup payroll account is fully operational. "
            "We have verified your wires personally with the treasurer team."
        ),
        provider=provider,
    )
    generic_worry = sum(item.worry_delta for item in generic.scores.values()) / len(generic.scores)
    personal_scores = personal.scores.values()
    personal_worry = sum(item.worry_delta for item in personal_scores) / len(personal_scores)
    assert personal_worry < generic_worry < 0.0


def test_personalization_strength_is_deterministic() -> None:
    first = personalization_strength("Your account is safe", "direct_call", ["startup"])
    second = personalization_strength("Your account is safe", "direct_call", ["startup"])
    assert first == second
    assert first > personalization_strength("Operations continue", "press_release", [])


def test_validate_persona_batch_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="invalid persona panel JSON"):
        validate_persona_batch(
            {
                "archetype_version": "1",
                "scores": {
                    "startup-ceo": {
                        "worry_delta": 2.0,
                        "threshold_delta": 0.0,
                        "act_now_share": 0.1,
                    }
                },
            }
        )


def test_llm_cache_avoids_second_provider_call(tmp_path: Path) -> None:
    cache = LLMCache(tmp_path / "cache.sqlite3")
    calls = 0

    class _Stub:
        def complete_json(self, prompt: str) -> dict:
            nonlocal calls
            calls += 1
            return {"echo": prompt, "n": calls}

    provider = _Stub()
    first = cached_json_completion(prompt="p", model="m", provider=provider, cache=cache)
    second = cached_json_completion(prompt="p", model="m", provider=provider, cache=cache)
    assert first == second == {"echo": "p", "n": 1}
    assert calls == 1
    assert prompt_hash("p", "m") == prompt_hash("p", "m")
    cache.close()


def test_real_module_scores_statement_like_fake() -> None:
    scenario = Scenario(preset="tech-concentrated", seed=7)
    event = StatementIssued(
        t=1.0,
        seed=7,
        bank_id="bank-fvb",
        cause_id="cause-1",
        stmt_id="stmt-management-1",
        channel="press_release",
        segments=["startup", "vc", "founder"],
    )
    fake = PersonaFake("M25")
    fake.scenario = scenario
    fake.on_tick(1.0, [event])
    real_scores = scores_for_statement_event(event, scenario.preset)
    for segment in event.segments:
        assert segment in real_scores
        assert real_scores[segment]["worry_delta"] < 0.0
        assert fake.scores[segment]["worry_delta"] < 0.0


def test_build_engine_m25_real_mode() -> None:
    scenario = Scenario(preset="tech-concentrated", seed=3)
    engine = build_engine(modes={"M25": ModuleMode.REAL}, scenario=scenario)
    module = engine["M25"]
    assert isinstance(module, PersonaPanelModule)
    engine.run(120)
    assert module.scores


def test_recording_provider_used_by_module() -> None:
    _RecordingProvider.calls = 0
    scenario = Scenario(preset="tech-concentrated", seed=5)
    module = PersonaPanelModule("M25", provider=_RecordingProvider())
    module.scenario = scenario
    event = StatementIssued(
        t=0.0,
        seed=5,
        bank_id="bank-fvb",
        cause_id="c",
        stmt_id="s1",
        channel="press_release",
        segments=["startup"],
    )
    module.on_tick(0.0, [event])
    assert _RecordingProvider.calls >= 1
    assert "startup" in module.scores
