"""M34 guardrails: blocklist, determinism, real module, fake contract parity."""

from __future__ import annotations

from agents.guardrails import (
    GuardrailConfig,
    GuardrailModule,
    claim_allowed,
    live_attention_enabled,
    load_guardrail_config,
    moderate_rumor,
)
from contracts.schemas import Rumor, Scenario
from engine.modes import ModuleMode
from engine.registry import build_engine
from mocks.l5_grok import GuardrailFake
from mocks.support import claim_allowed as fake_claim_allowed


def test_registry_blocklist_rejects_real_bank_names() -> None:
    assert not claim_allowed("Depositors are leaving Silicon Valley Bank tonight")
    assert not claim_allowed("SBNY stock is moving")
    assert not claim_allowed("Chase customers should pull their money now")
    assert not claim_allowed("Wells Fargo wires are delayed")
    assert claim_allowed("Uninsured deposits are leaving Fictional Valley Bank")
    assert claim_allowed("Kestrel Bank lost billions on bonds")
    assert claim_allowed("Uninsured share is high at Kestrel Bank")


def test_fuzzy_match_catches_typo() -> None:
    cfg = load_guardrail_config()
    typo = "Silicon Vally Bank is in trouble"
    assert not claim_allowed(typo, config=cfg)


def test_moderation_result_is_deterministic() -> None:
    text = "Rumor about Fictional Valley Bank"
    first = moderate_rumor(text)
    second = moderate_rumor(text)
    assert first == second


def test_live_attention_off_by_default() -> None:
    assert live_attention_enabled() is False


def test_real_module_matches_fake_blocked_rumors() -> None:
    scenario = Scenario(
        preset="tech-concentrated",
        seed=7,
        rumors=[
            Rumor(id="ok", claim="Fictional Valley Bank may cut rates", credibility=0.5, reach=10),
            Rumor(id="bad", claim="Silicon Valley Bank is raising cash", credibility=0.5, reach=10),
        ],
    )
    fake = GuardrailFake("M34")
    fake.scenario = scenario
    fake.on_tick(0.0, [])
    real = GuardrailModule("M34")
    real.scenario = scenario
    real.on_tick(0.0, [])
    assert fake.blocked == real.blocked == ["bad"]


def test_fake_claim_allowed_matches_real() -> None:
    samples = (
        "Silicon Valley Bank",
        "Ticker SIVB is down",
        "Fictional Community Bank is calm",
        "Group chat mentions signature bank",
    )
    for sample in samples:
        assert fake_claim_allowed(sample) == claim_allowed(sample)


def test_build_engine_m34_real_mode() -> None:
    scenario = Scenario(
        preset="tech-concentrated",
        seed=1,
        rumors=[
            Rumor(
                id="bad",
                claim="First Republic depositors are nervous",
                credibility=0.4,
                reach=1,
            ),
        ],
    )
    engine = build_engine(modes={"M34": ModuleMode.REAL}, scenario=scenario)
    module = engine["M34"]
    assert isinstance(module, GuardrailModule)
    engine.run(2)
    assert module.blocked == ["bad"]


def test_custom_config_bank_name_allowed() -> None:
    cfg = GuardrailConfig(
        bank_names=("acme bank",),
        tickers=frozenset({"acm"}),
        fuzzy_max_distance=0,
        live_attention_default=False,
        slur_tokens=frozenset(),
        real_person_names=(),
    )
    assert not claim_allowed("Acme Bank fails", config=cfg)
    assert claim_allowed("Fictional Valley Bank", config=cfg)
