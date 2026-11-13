"""M26 ensemble and futures engine: CPU path, determinism, registry."""

from __future__ import annotations

import numpy as np
import pytest

from contracts import parse_event
from contracts.events import FutureCompleted
from contracts.schemas import Scenario
from engine.futures.batch import (
    FutureSpec,
    FuturesRunner,
    Intervention,
    MarketSignals,
    _gpu_to_device,
    build_future_specs,
    future_completed_events,
    load_futures_params,
    parse_registry_number,
    resolve_backend,
    run_batch,
    run_serial_reference,
    world_intervention,
)
from engine.modes import ModuleMode
from engine.registry import build_engine, run_scenario
from mocks.l1_engine import FuturesFake
from params.loader import get_parameter, list_by_module

_ASSUMED = (
    "M26 sim-day tick count",
    "M26 complex contagion minimum worried contacts (k)",
    "M26 contact offsets",
    "M26 hub balance fraction",
    "M26 hub contact weight",
    "M26 worry logit bias",
    "M26 worry weights",
    "M26 logit clip",
    "M26 threshold low",
    "M26 threshold high",
    "M26 belief low",
    "M26 belief high",
    "M26 belief evidence scale",
    "M26 attention signal weight",
    "M26 loss signal weight",
    "M26 payroll need rate",
    "M26 balance lognormal sigma",
    "M26 segment names",
    "M26 segment shares",
    "M26 segment withdrawal fractions",
    "M26 ownership categories per depositor",
    "M26 opening cash fraction of deposits",
    "M26 opening price",
    "M26 price outflow beta",
    "M26 price attention beta",
    "M26 price shock scale",
    "M26 weight perturbation amplitude",
    "M26 statement worry shift",
    "M26 statement threshold shift",
    "M26 statement shift spread",
    "M26 cap-withdrawals amount scale",
    "M26 single-future match tolerance",
    "M26 directive withdraw level",
    "M26 probability floor",
    "M26 philox stream ids",
)


def _spec(
    seed: int = 11,
    attention: float = 0.0,
    *,
    guarantee: bool = False,
    future_id: str = "f00",
) -> FutureSpec:
    return FutureSpec(
        future_id=future_id,
        seed=seed,
        attention=attention,
        stock_return=0.0,
        intervention=Intervention(guarantee=guarantee),
    )


def test_registry_numbers_and_assumed_rows() -> None:
    assert parse_registry_number("n", "250k") == 250_000.0
    assert parse_registry_number("n", "5 minutes") == 5.0
    assert parse_registry_number("n", "$175B (year-end 2022)") == 175_000_000_000.0
    assert parse_registry_number("n", "1e-8") == 1e-8
    assert parse_registry_number("n", "$250,000 per depositor") == 250_000.0
    params = load_futures_params()
    assert params.n_futures == 16
    assert params.n_agents == 250_000
    assert params.tick_minutes == 5.0
    assert params.n_ticks == 78
    assert len(params.worry_weights) == 7
    for name in _ASSUMED:
        row = get_parameter(name)
        assert row.status == "ASSUMED"
        assert row.module == "M26"
    assert any(row.parameter == "Futures count (typical batch)" for row in list_by_module("M26"))


def test_cpu_backend_and_gpu_path_is_isolated() -> None:
    xp, name = resolve_backend()
    if name == "cpu":
        assert xp is np
    else:
        assert xp.__name__ == "cupy"
    with pytest.raises(RuntimeError, match="GPU path"):
        _gpu_to_device(np, np.zeros(2), "cpu")
    params = load_futures_params()
    summary = run_batch(
        [_spec()],
        params,
        n_agents=12,
        n_ticks=2,
        fully_insured=False,
        backend=(np, "cpu"),
    )
    assert summary.backend == "cpu"


def test_fixed_seed_is_deterministic_on_cpu() -> None:
    params = load_futures_params()
    specs = [_spec(3, 0.95), _spec(9, 0.0, future_id="f01")]
    kwargs = {
        "n_agents": 36,
        "n_ticks": 4,
        "fully_insured": False,
        "backend": (np, "cpu"),
    }
    first = run_batch(specs, params, **kwargs)
    second = run_batch(specs, params, **kwargs)
    assert first == second


def test_single_future_matches_serial_reference() -> None:
    params = load_futures_params()
    spec = _spec(21, 0.95)
    serial = run_serial_reference(spec, params, n_agents=24, n_ticks=4, fully_insured=False)
    batch = run_batch(
        [spec],
        params,
        n_agents=24,
        n_ticks=4,
        fully_insured=False,
        backend=(np, "cpu"),
    )
    got = batch.futures[0]
    tol = params.match_tolerance
    assert got.failed == serial.failed
    assert got.t_fail == serial.t_fail
    assert abs(got.deposits_kept - serial.deposits_kept) <= tol * params.deposits
    assert abs(got.price_path_band[0] - serial.price_path_band[0]) <= tol * params.opening_price
    assert abs(got.price_path_band[1] - serial.price_path_band[1]) <= tol * params.opening_price
    quiet = _spec(21, 0.0)
    quiet_serial = run_serial_reference(quiet, params, n_agents=24, n_ticks=3, fully_insured=False)
    quiet_batch = run_batch(
        [quiet],
        params,
        n_agents=24,
        n_ticks=3,
        fully_insured=False,
        backend=(np, "cpu"),
    ).futures[0]
    assert quiet_serial.failed is False
    assert quiet_serial.outflow == 0.0
    assert quiet_batch.deposits_kept == quiet_serial.deposits_kept
    assert quiet_batch.t_fail == quiet_serial.t_fail


def test_future_zero_does_not_depend_on_batch_width() -> None:
    params = load_futures_params()
    signals = MarketSignals(attention=0.9, stock_return=-0.02, opening_price=100.0)
    specs = build_future_specs(
        base_seed=5,
        n_futures=4,
        signals=signals,
        statement_ids=(),
        world=world_intervention([], 0.0, params),
        params=params,
    )
    assert specs[0].weight_multipliers is None
    assert specs[1].weight_multipliers is not None
    alone = run_batch([specs[0]], params, n_agents=20, n_ticks=3, fully_insured=False)
    wide = run_batch(specs, params, n_agents=20, n_ticks=3, fully_insured=False)
    assert alone.futures[0] == wide.futures[0]


def test_ranges_failure_and_interventions() -> None:
    params = load_futures_params()
    summary = run_batch(
        [_spec(4, 0.95, future_id="f00"), _spec(4, 0.0, future_id="f01")],
        params,
        n_agents=48,
        n_ticks=4,
        fully_insured=False,
        backend=(np, "cpu"),
    )
    assert summary.futures[0].failed is True
    assert summary.futures[0].t_fail == 10.0
    assert summary.futures[1].failed is False
    assert summary.failure_probability == 0.5
    assert summary.outflow_band[0] == 0.0
    assert summary.outflow_band[1] > params.cash_fraction
    assert summary.price_band[0] < summary.price_band[1]
    assert summary.time_to_failure == (10.0,)
    guaranteed = run_batch(
        [_spec(4, 0.95, guarantee=True)],
        params,
        n_agents=32,
        n_ticks=4,
        fully_insured=False,
    )
    assert guaranteed.futures[0].failed is False
    assert guaranteed.futures[0].outflow == 0.0
    community = run_batch(
        [_spec(4, 0.95)],
        params,
        n_agents=32,
        n_ticks=4,
        fully_insured=True,
    )
    assert community.futures[0].outflow == 0.0
    capped = world_intervention(["cap_withdrawals"], 0.0, params)
    assert capped.withdraw_scale == params.cap_scale
    assert world_intervention(["guarantee_deposits"], 1.0, params).guarantee is True


def test_statement_specs_share_a_seed_and_shift_worry() -> None:
    params = load_futures_params()
    signals = MarketSignals(attention=0.4, stock_return=0.0, opening_price=80.0)
    specs = build_future_specs(
        base_seed=8,
        n_futures=16,
        signals=signals,
        statement_ids=("stmt-a", "stmt-b"),
        world=world_intervention([], 0.0, params),
        params=params,
    )
    assert [item.future_id for item in specs] == ["f00", "f01", "f02"]
    assert {item.seed for item in specs} == {8}
    assert specs[1].statement_id == "stmt-a"
    assert specs[2].intervention.worry_shift < specs[1].intervention.worry_shift < 0.0


def test_future_completed_matches_contract() -> None:
    params = load_futures_params()
    summary = run_batch([_spec(2, 0.2)], params, n_agents=8, n_ticks=2, fully_insured=False)
    events = future_completed_events(summary, t=1.0, seed=2, bank_id="bank-fvb")
    parsed = parse_event(events[0].model_dump())
    assert isinstance(parsed, FutureCompleted)
    assert parsed.future_id == "f00"
    result = summary.futures[0].to_future_result()
    assert result.failed == summary.futures[0].failed
    assert result.price_path_band == summary.futures[0].price_path_band


def test_real_mode_publishes_once_and_fake_stays_default() -> None:
    scenario = Scenario(preset="tech-concentrated", seed=4)
    fake = build_engine(scenario=scenario)
    assert isinstance(fake["M26"], FuturesFake)
    fake_log = run_scenario(scenario, 2)
    assert any(isinstance(event, FutureCompleted) for event in fake_log)

    engine = build_engine(scenario=scenario, modes={"M26": ModuleMode.REAL})
    runner = engine["M26"]
    assert isinstance(runner, FuturesRunner)
    runner.agent_count = 16
    runner.future_count = 3
    runner.tick_count = 2
    first = engine.run(2)
    events = [event for event in first if isinstance(event, FutureCompleted)]
    assert len(events) == 3
    assert events[0].seed == 4
    assert events[0].bank_id == "bank-fvb"
    assert runner.summary is not None
    assert runner.summary.backend in {"cpu", "gpu"}
    again = build_engine(scenario=scenario, modes={"M26": ModuleMode.REAL})
    again_runner = again["M26"]
    assert isinstance(again_runner, FuturesRunner)
    again_runner.agent_count = 16
    again_runner.future_count = 3
    again_runner.tick_count = 2
    second = again.run(2)
    assert [event.model_dump() for event in first if isinstance(event, FutureCompleted)] == [
        event.model_dump() for event in second if isinstance(event, FutureCompleted)
    ]
    bare = FuturesRunner("M26")
    assert bare.step(0.0) == []
