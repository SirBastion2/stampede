"""M01 real mode: determinism, Nelson–Siegel, bond-loss path, contract parse."""

from __future__ import annotations

import math
from typing import Any

from contracts import parse_event
from contracts.events import RatesChanged
from contracts.schemas import Rumor, Scenario, ScenarioShock
from engine.macro.rates import MacroRates, nelson_siegel, piecewise_linear
from engine.modes import ModuleMode
from engine.registry import build_engine, run_scenario
from mocks.base import FakeModule
from mocks.bus import EventBus
from params.loader import get_parameter

# 2022-like policy path (scenario input, not a fitted parameter):
# about 0.25% → 1.50% → 3.00% → 4.25% → 5.25%.
HIKE: list[tuple[float, float]] = [
    (0.0, 0.0025),
    (1.0, 0.015),
    (2.0, 0.03),
    (3.0, 0.0425),
    (4.0, 0.0525),
]

# §18.3: +100 bp on a 6-year bond is "about −6%". Tolerance is 1 point.
_ABOUT_ONE_POINT = 0.01
_ONE_HUNDRED_BP = 0.01


def _param(name: str) -> float:
    return float(get_parameter(name).value)


def _scenario(seed: int = 1, **overrides: Any) -> Scenario:
    fields: dict[str, Any] = {
        "preset": "tech-concentrated",
        "seed": seed,
        "rate_path": [(0.0, 0.04), (3.0, 0.06)],
        "rumors": [
            Rumor(
                id="rumor-1",
                claim="Uninsured deposits are leaving Fictional Valley Bank",
                credibility=0.85,
                reach=12000,
            )
        ],
    }
    fields.update(overrides)
    return Scenario(**fields)


def _attach(scenario: Scenario) -> MacroRates:
    module = MacroRates("M01")
    module.attach(EventBus(), scenario)
    return module


def _dump(events: list[Any]) -> list[dict[str, Any]]:
    return [event.model_dump(mode="json", by_alias=True) for event in events]


def _roundtrip(event: Any) -> dict[str, Any]:
    payload = event.model_dump(mode="json", by_alias=True)
    parsed = parse_event(payload)
    assert parsed.model_dump(mode="json", by_alias=True) == payload
    return payload


def _closed_form(
    tenor: float,
    beta0: float,
    beta1: float,
    beta2: float,
    ns_lambda: float,
) -> float:
    """Independent Nelson–Siegel expansion, so a dropped term fails the test."""
    x = ns_lambda * tenor
    exp_neg = math.exp(-x)
    slope_loading = (1.0 - exp_neg) / x
    return beta0 + beta1 * slope_loading + beta2 * (slope_loading - exp_neg)


def _mtm_loss(delta_yield: float, duration: float, convexity: float) -> float:
    """Loss = −(−D·Δy + ½·C·Δy²). A price fall is a positive loss (§19 M09)."""
    price_change = -duration * delta_yield + 0.5 * convexity * delta_yield**2
    return -price_change


def _run(scenario: Scenario, steps: int = 4) -> tuple[MacroRates, list[Any]]:
    module = _attach(scenario)
    log: list[Any] = []
    for step in range(steps):
        log.extend(module.step(float(step)))
    return module, log


def test_determinism_same_seed_and_seed_does_not_move_the_economy() -> None:
    shocks = [
        ScenarioShock(kind="property", at=1.0, params={"sector": "office", "factor": 0.9}),
        ScenarioShock(kind="unemployment", at=2.0, params={"rate": 0.07}),
    ]
    left, log_left = _run(_scenario(seed=1, shocks=shocks))
    right, log_right = _run(_scenario(seed=1, shocks=shocks))
    other, log_other = _run(_scenario(seed=2, shocks=shocks))
    assert _dump(log_left) == _dump(log_right)
    assert left.curve == right.curve
    assert left.cre == right.cre
    assert left.unemployment == right.unemployment
    assert left.policy_rate == right.policy_rate
    for same, changed in zip(_dump(log_left), _dump(log_other), strict=True):
        assert same["seed"] == 1
        assert changed["seed"] == 2
        same["seed"] = 0
        changed["seed"] = 0
        assert same == changed
    assert left.curve == other.curve
    assert left.cre == other.cre
    assert left.policy_rate == other.policy_rate


def test_piecewise_policy_path_knots_midpoint_and_clamp() -> None:
    default = _param("M01 default policy rate")
    floor = _param("M01 policy rate floor")
    cap = _param("M01 policy rate cap")
    path = [(0.0, 0.02), (2.0, 0.04)]
    assert piecewise_linear(path, 0.0, default) == 0.02
    assert piecewise_linear(path, 1.0, default) == 0.03
    assert piecewise_linear(path, -1.0, default) == 0.02
    assert piecewise_linear(path, 5.0, default) == 0.04
    assert piecewise_linear([], 1.0, default) == default

    empty = _attach(_scenario(rate_path=[]))
    empty.step(0.0)
    assert empty.policy_rate == default

    high = _attach(_scenario(rate_path=[(0.0, 0.9)]))
    published = high.step(0.0)[0]
    assert isinstance(published, RatesChanged)
    assert published.policy_rate == cap
    limit = nelson_siegel(
        0.0,
        high.params.beta0,
        cap - high.params.beta0,
        high.params.beta2,
        high.params.ns_lambda,
    )
    assert abs(limit - cap) < 1e-12

    low = _attach(_scenario(rate_path=[(0.0, -0.2)]))
    low.step(0.0)
    assert low.policy_rate == floor
    assert MacroRates("M01").step(0.0) == []


def test_nelson_siegel_matches_published_buckets() -> None:
    module = _attach(_scenario())
    event = module.step(0.0)[0]
    assert isinstance(event, RatesChanged)
    tenors = module.params.tenors
    registry_tenors = [
        float(part) for part in get_parameter("M01 maturity buckets (years)").value.split(",")
    ]
    assert list(tenors) == registry_tenors
    # §5: Treasury curve from 3 months through 30 years.
    assert tenors[0] == 0.25
    assert tenors[-1] == 30.0
    beta1 = event.policy_rate - module.params.beta0
    assert module.params.beta2 != 0.0
    rebuilt = [
        _closed_form(
            tenor,
            module.params.beta0,
            beta1,
            module.params.beta2,
            module.params.ns_lambda,
        )
        for tenor in tenors
    ]
    no_curve = [
        _closed_form(tenor, module.params.beta0, beta1, 0.0, module.params.ns_lambda)
        for tenor in tenors
    ]
    assert event.curve == rebuilt
    assert event.curve != no_curve
    assert event.curve == [
        nelson_siegel(
            tenor,
            module.params.beta0,
            beta1,
            module.params.beta2,
            module.params.ns_lambda,
        )
        for tenor in tenors
    ]
    later = module.step(3.0)[0]
    assert isinstance(later, RatesChanged)
    assert later.policy_rate > event.policy_rate
    assert all(after > before for before, after in zip(event.curve, later.curve, strict=True))


def test_shocks_apply_once_and_prices_stay_non_negative() -> None:
    module = _attach(
        _scenario(
            rate_path=[(0.0, 0.04)],
            shocks=[
                ScenarioShock(kind="unemployment", at=0.0, params={"rate": 0.2}),
                ScenarioShock(kind="unemployment", at=0.0, params={"rate": 0.07}),
                ScenarioShock(kind="property", at=1.0, params={"sector": "office", "factor": 0.5}),
                ScenarioShock(kind="property", at=1.0, params={"sector": "office", "factor": 0.5}),
                ScenarioShock(kind="property", at=1.0, params={"sector": "all", "factor": 0.8}),
                ScenarioShock(kind="property", at=2.0, params={"sector": "retail", "factor": -5}),
                ScenarioShock(kind="unemployment", at=2.0, params={"rate": 0.9}),
                ScenarioShock(kind="not-a-shock", at=0.0, params={"rate": 0.99}),
                ScenarioShock(
                    kind="property",
                    at=0.0,
                    params={"sector": "warehouse", "factor": 0.1},
                ),
                ScenarioShock(kind="property", at=3.0, params={"factor": "nope"}),
            ],
        )
    )
    office0 = module.cre["office"]
    multi0 = module.cre["multifamily"]
    retail0 = module.cre["retail"]
    module.step(0.0)
    # Both unemployment shocks are due; the later one wins. Property at t=1 waits.
    assert module.unemployment == 0.07
    assert module.cre["office"] == office0
    assert module.cre["multifamily"] == multi0
    module.step(1.0)
    assert module.cre["office"] == office0 * 0.5 * 0.5 * 0.8
    assert module.cre["multifamily"] == multi0 * 0.8
    assert module.cre["retail"] == retail0 * 0.8
    module.step(1.0)
    assert module.cre["office"] == office0 * 0.5 * 0.5 * 0.8
    module.step(2.0)
    assert module.cre["retail"] == _param("M01 CRE index floor")
    assert module.unemployment == _param("M01 unemployment cap")
    assert module.cre["office"] == office0 * 0.5 * 0.5 * 0.8
    held = dict(module.cre)
    jobless = module.unemployment
    module.step(3.0)
    assert module.cre == held
    assert module.unemployment == jobless


def test_2022_hike_raises_bond_losses_steadily() -> None:
    duration = _param("M01 representative bond duration")
    convexity = _param("M01 representative bond convexity")
    securities = get_parameter("Securities test: +100 bp on 6-year duration bond")
    assert "100" in securities.parameter
    assert "6" in securities.parameter
    assert "6" in securities.value
    price_change = -duration * _ONE_HUNDRED_BP + 0.5 * convexity * _ONE_HUNDRED_BP**2
    assert price_change < 0.0
    assert abs(price_change - (-duration * _ONE_HUNDRED_BP)) < _ABOUT_ONE_POINT

    module = _attach(_scenario(rate_path=HIKE, shocks=[]))
    losses: list[float] = []
    yields: list[float] = []
    policies: list[float] = []
    for step in range(9):
        event = module.step(step / 2.0)[0]
        assert isinstance(event, RatesChanged)
        level = module.yield_at(duration)
        yields.append(level)
        policies.append(event.policy_rate)
        losses.append(_mtm_loss(level - yields[0], duration, convexity))
    assert losses[0] == 0.0
    assert all(later > earlier for earlier, later in zip(policies, policies[1:], strict=False))
    assert all(later > earlier for earlier, later in zip(yields, yields[1:], strict=False))
    assert all(later > earlier for earlier, later in zip(losses, losses[1:], strict=False))

    flat = _attach(_scenario(rate_path=[(0.0, 0.02), (4.0, 0.02)]))
    flat.step(0.0)
    y0 = flat.yield_at(duration)
    flat.step(4.0)
    assert flat.yield_at(duration) == y0
    assert _mtm_loss(0.0, duration, convexity) == 0.0


def test_real_and_fake_rates_changed_parse_and_policy_path_matches() -> None:
    scenario = _scenario(
        seed=3,
        shocks=[ScenarioShock(kind="ui_rumor", at=3.0, params={"claim": "ignored by M01"})],
    )
    real_engine = build_engine(modes={"M01": ModuleMode.REAL}, scenario=scenario)
    fake_engine = build_engine(scenario=scenario)
    real_m01 = real_engine["M01"]
    assert isinstance(real_m01, MacroRates)
    assert isinstance(fake_engine["M01"], FakeModule)
    assert isinstance(real_engine["M08"], FakeModule)
    assert isinstance(real_engine["M09"], FakeModule)
    assert isinstance(build_engine(scenario=scenario)["M01"], FakeModule)

    real_log = real_engine.run(6)
    fake_log = fake_engine.run(6)
    for event in real_log + fake_log:
        assert _roundtrip(event)["event_type"]
    real_rates = [event for event in real_log if isinstance(event, RatesChanged)]
    fake_rates = [event for event in fake_log if isinstance(event, RatesChanged)]
    assert len(real_rates) == 6
    assert len(fake_rates) == 6
    for real, fake in zip(real_rates, fake_rates, strict=True):
        assert _roundtrip(real)["event_type"] == "RatesChanged"
        assert _roundtrip(fake)["event_type"] == "RatesChanged"
        assert real.policy_rate == fake.policy_rate
        assert real.curve != fake.curve
        assert len(real.curve) == len(real_m01.params.tenors)
        assert real.bank_id is None
    assert _dump(run_scenario(scenario, 6, modes={"M01": ModuleMode.REAL})) == _dump(real_log)

    off = run_scenario(scenario, 3, modes={"M01": ModuleMode.OFF})
    assert all(event.event_type != "RatesChanged" for event in off)
