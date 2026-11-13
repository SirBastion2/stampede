"""M05 real population: determinism, preset targets, engine contract parse."""

from __future__ import annotations

import pytest

from contracts import parse_event
from engine.agents.population import PopulationModule, parse_registry_number, sample_population
from engine.modes import ModuleMode
from engine.registry import build_engine, run_scenario
from mocks.l1_engine import PopulationFake
from tests.test_fakes import panic


def test_sample_population_matches_preset_targets() -> None:
    scenario = panic(7)
    _arrays, summary = sample_population(scenario)
    target_uninsured = parse_registry_number("SVB-like: uninsured share")
    target_insured = 1.0 - target_uninsured
    insured_tol = parse_registry_number("Population test: insured share tolerance")
    top10_tol = parse_registry_number("Population test: top-10 balance tolerance")
    target_top10 = parse_registry_number("SVB-like: top-10 depositor balance")

    assert abs(summary.insured_share - target_insured) <= insured_tol
    assert abs(summary.top10_balance - target_top10) / target_top10 <= top10_tol
    assert summary.depositor_count == int(parse_registry_number("Depositor count (Tier 0)"))


def test_sample_population_is_deterministic() -> None:
    scenario = panic(11)
    first, _ = sample_population(scenario)
    second, _ = sample_population(scenario)
    assert first.balance == second.balance
    assert first.segment == second.segment
    assert first.insured == second.insured


def test_real_module_step_builds_arrays_once() -> None:
    module = PopulationModule()
    module.attach(panic(3))
    assert module.step(0.0) == []
    assert module.arrays is not None
    assert module.step(1.0) == []


def test_real_vs_fake_summary_contract() -> None:
    scenario = panic(5)
    real_engine = build_engine(modes={"M05": ModuleMode.REAL}, scenario=scenario)
    fake_engine = build_engine(scenario=scenario)
    real_engine.run(1)
    fake_engine.run(1)
    real_pop = real_engine["M05"]
    fake_pop = fake_engine["M05"]
    assert isinstance(real_pop, PopulationModule)
    assert isinstance(fake_pop, PopulationFake)
    assert real_pop.depositor_count == fake_pop.depositor_count
    assert real_pop.insurance_limit == fake_pop.insurance_limit
    target_uninsured = parse_registry_number("SVB-like: uninsured share")
    insured_tol = parse_registry_number("Population test: insured share tolerance")
    top10_tol = parse_registry_number("Population test: top-10 balance tolerance")
    target_top10 = parse_registry_number("SVB-like: top-10 depositor balance")
    assert abs((1.0 - real_pop.insured_share) - target_uninsured) <= insured_tol + 0.005
    assert abs(real_pop.top10_balance - target_top10) / target_top10 <= top10_tol + 0.01


def test_engine_events_parse_with_real_m05() -> None:
    log = run_scenario(panic(2), steps=6, modes={"M05": ModuleMode.REAL})
    for event in log:
        payload = event.model_dump(mode="json", by_alias=True)
        parsed = parse_event(payload)
        assert parsed.model_dump(mode="json", by_alias=True) == payload


def test_build_engine_m05_real_does_not_raise() -> None:
    build_engine(modes={"M05": ModuleMode.REAL}, scenario=panic(1))


def test_other_real_mode_still_raises() -> None:
    with pytest.raises(NotImplementedError, match="M29"):
        build_engine(modes={"M29": ModuleMode.REAL})
