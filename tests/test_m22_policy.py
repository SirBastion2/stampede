"""M22 policy lab: guarantees, moral hazard, real vs fake, offline advisor."""

from __future__ import annotations

from contracts.events import AttentionTick, PolicyEnacted, WithdrawalRequested
from contracts.schemas import DepositorArrays, Scenario
from engine.agents.decide import DecisionModule, load_decision_params
from engine.modes import ModuleMode
from engine.policy.levers import (
    MoralHazardMeter,
    PolicyLabModule,
    build_cost_model,
    explain_policy_options,
    load_policy_params,
)
from engine.registry import run_scenario
from mocks.bus import EventBus
from mocks.l4_system import PolicyFake
from mocks.support import bank_id_for
from tests.test_fakes import panic


def _policy_scenario(**overrides) -> Scenario:
    base = panic(3)
    policies = overrides.pop("policies", [{"lever": "guarantee_deposits", "at": 0.0}])
    return base.model_copy(update={"policies": policies, **overrides})


def test_guarantee_stops_withdrawals_within_scope(monkeypatch) -> None:
    monkeypatch.setattr(DecisionModule, "aware0", 1.0)
    params = load_decision_params()
    k = int(params.k)
    n = k + 1
    scenario = _policy_scenario()
    bank_id = bank_id_for(scenario.preset)
    threshold = params.worry_ceiling
    arrays = DepositorArrays(
        bank_idx=[0] * n,
        segment=["startup"] * n,
        balance=[2_000_000.0] * n,
        insured=[0.0] * n,
        belief=[0.2] * n,
        threshold=[threshold] * n,
        attention=[1.0] * n,
        needs_flag=[False] * n,
        has_second_bank=[False] * n,
        state=["active"] * n,
    )

    def _run_once(with_guarantee: bool) -> list:
        module = DecisionModule("M06")
        bus = EventBus()
        module.attach(bus, scenario)
        module.load_depositor_arrays(arrays, seed=scenario.seed)
        bus.publish(
            AttentionTick(
                t=0.0,
                seed=scenario.seed,
                bank_id=bank_id,
                level=1.0,
                delta_4h=0.0,
            )
        )
        if with_guarantee:
            bus.publish(
                PolicyEnacted(
                    t=0.0,
                    seed=scenario.seed,
                    bank_id=bank_id,
                    cause_id="test-guarantee",
                    lever="guarantee_deposits",
                    scope=bank_id,
                    cost_model={
                        "dif_loss": 0.0,
                        "fed_exposure": 0.0,
                        "moral_hazard": 0.5,
                    },
                )
            )
        return module.step(1.0)

    guaranteed = _run_once(True)
    unguaranteed = _run_once(False)
    assert not any(isinstance(event, WithdrawalRequested) for event in guaranteed)
    assert any(isinstance(event, WithdrawalRequested) for event in unguaranteed)


def test_moral_hazard_meter_tracks_guarantees() -> None:
    params = load_policy_params()
    meter = MoralHazardMeter()
    first = build_cost_model("guarantee_deposits", params, meter)
    meter.record("guarantee_deposits", "bank-a")
    second = build_cost_model("guarantee_deposits", params, meter)
    assert second["moral_hazard"] >= first["moral_hazard"]
    assert first["dif_loss"] > 0.0


def test_real_and_fake_publish_same_levers() -> None:
    scenario = _policy_scenario(
        policies=[
            {"lever": "guarantee_deposits", "at": 2.0},
            {"lever": "emergency_lending", "at": 4.0},
        ]
    )
    fake_log = run_scenario(scenario, steps=6, modes={"M22": ModuleMode.FAKE})
    real_log = run_scenario(scenario, steps=6, modes={"M22": ModuleMode.REAL})
    fake_levers = [event.lever for event in fake_log if isinstance(event, PolicyEnacted)]
    real_levers = [event.lever for event in real_log if isinstance(event, PolicyEnacted)]
    assert fake_levers == real_levers == ["guarantee_deposits", "emergency_lending"]


def test_policy_lab_module_step_matches_fake_schedule() -> None:
    scenario = _policy_scenario(policies=[{"lever": "cap_withdrawals", "at": 1.0}])
    bus = EventBus()
    fake = PolicyFake("M22")
    real = PolicyLabModule("M22")
    fake.attach(bus, scenario)
    real.attach(bus, scenario)
    fake_out = fake.step(1.0)
    real_out = real.step(1.0)
    assert len(fake_out) == len(real_out) == 1
    assert fake_out[0].lever == real_out[0].lever == "cap_withdrawals"


def test_explain_policy_options_offline_has_no_network() -> None:
    batch = explain_policy_options(bank_id="bank-fvb")
    assert batch.explanations
    assert all(0.0 <= item.moral_hazard <= 1.0 for item in batch.explanations)


def test_guarantee_reduces_run_withdrawals_with_real_m06(monkeypatch) -> None:
    monkeypatch.setattr(DecisionModule, "aware0", 1.0)
    scenario = panic(5)
    baseline = run_scenario(
        scenario,
        steps=12,
        modes={"M06": ModuleMode.REAL, "M22": ModuleMode.FAKE},
    )
    guaranteed = run_scenario(
        scenario.model_copy(
            update={"policies": [{"lever": "guarantee_deposits", "at": 0.0}]}
        ),
        steps=12,
        modes={"M06": ModuleMode.REAL, "M22": ModuleMode.REAL},
    )
    base_amt = sum(
        event.amount
        for event in baseline
        if isinstance(event, WithdrawalRequested)
    )
    guar_amt = sum(
        event.amount
        for event in guaranteed
        if isinstance(event, WithdrawalRequested)
    )
    assert guar_amt < base_amt
