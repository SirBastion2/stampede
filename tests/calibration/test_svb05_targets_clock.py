"""SVB-05: simulation-clock targets, coarse overnight ticks, stage-2 capacity."""

from __future__ import annotations

import math

from calibration.fit import (
    _DAY1_CLOSE,
    _NEXT_OPEN,
    FitResult,
    HoldoutResult,
    _score_outflow_log,
    build_validation_board,
    default_tick_schedule,
    fit_payment_capacity,
    fixed_softness,
    shared_eval_seeds,
)
from contracts.events import PaymentSettled, WithdrawalRequested
from engine.agents.population import clear_sample_population_cache
from engine.gpu_sim import clear_static_pack_cache, relative_error, simulate_logs


def _silence_worker_threads(monkeypatch) -> None:
    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)


def _requested_total(log: list[object]) -> float:
    return sum(event.amount for event in log if isinstance(event, WithdrawalRequested))


def test_numpy_path_requests_are_timestep_invariant(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "24")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    clear_sample_population_cache()
    clear_static_pack_cache()

    import numpy as np

    from calibration.fit import _default_svb_scenario

    scenario = _default_svb_scenario(17)
    vector = np.full(12, 0.5)
    one = simulate_logs(
        scenario,
        60,
        [vector],
        array_module="numpy",
        disable_caps=True,
        softness=fixed_softness(),
        clamp_inputs=False,
        schedule=[(float(t), 1.0) for t in range(60)],
    )
    five = simulate_logs(
        scenario,
        12,
        [vector],
        array_module="numpy",
        disable_caps=True,
        softness=fixed_softness(),
        clamp_inputs=False,
        schedule=[(float(t * 5), 5.0) for t in range(12)],
    )
    left = _requested_total(one[0])
    right = _requested_total(five[0])
    assert relative_error(left, right) <= 0.02

    clear_static_pack_cache()
    clear_sample_population_cache()


def test_scorer_counts_day1_close_and_next_open() -> None:
    day1 = WithdrawalRequested(
        t=389.0,
        seed=0,
        depositor_id="a",
        amount=42.0,
        channel="wire",
    )
    after_close = WithdrawalRequested(
        t=390.0,
        seed=0,
        depositor_id="b",
        amount=100.0,
        channel="wire",
    )
    before_open = WithdrawalRequested(
        t=1439.0,
        seed=0,
        depositor_id="c",
        amount=7.0,
        channel="wire",
    )
    at_open = WithdrawalRequested(
        t=1440.0,
        seed=0,
        depositor_id="d",
        amount=9.0,
        channel="wire",
    )
    settled = PaymentSettled(
        t=100.0,
        seed=0,
        amount=1.0,
        channel="wire",
        queued_remaining=0,
    )
    day1_frac, next_open, closed = _score_outflow_log(
        [day1, after_close, before_open, at_open, settled],
        requested=True,
    )
    deposits = 175_000_000_000.0
    assert not closed
    assert day1.t < _DAY1_CLOSE <= after_close.t
    assert before_open.t < _NEXT_OPEN <= at_open.t
    assert abs(day1_frac - 42.0 / deposits) < 1e-18
    assert abs(next_open - (42.0 + 100.0 + 7.0) / deposits) < 1e-18


def test_stage2_grid_hits_settled_day_one_on_stub(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    calls: list[list[float]] = []

    def stub_engine(multipliers):
        calls.append([float(m) for m in multipliers])
        return [0.24 + 0.05 * math.log(float(m) / 2.0) for m in multipliers]

    capacity = fit_payment_capacity(measure=stub_engine, target=0.24)
    assert 1 <= len(calls) <= 2
    observed = 0.24 + 0.05 * math.log(float(capacity) / 2.0)
    assert abs(observed - 0.24) <= 0.002


def test_validation_board_has_clock_rows(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)

    def _fake_measure(*_args, **kwargs):
        if kwargs.get("requested"):
            return 0.10, 0.81, False
        return 0.24, 0.24, False

    monkeypatch.setattr("calibration.fit.measure_outflow", _fake_measure)
    fit = FitResult(seed=0, vector=(0.5,) * 12, objective=0.0, generations=1)
    holdout = HoldoutResult(signature_day1_fraction=0.20, placebo_survives=True)
    board = build_validation_board(fit, holdout)
    labels = [row.label for row in board]
    assert "requests by next open" in labels
    assert "settled day one" in labels
    by_label = {row.label: row for row in board}
    assert by_label["requests by next open"].expected == 0.81
    assert by_label["settled day one"].expected == 0.24


def test_overnight_schedule_is_about_1_2x_day() -> None:
    schedule = default_tick_schedule()
    assert schedule[:3] == [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]
    assert schedule[389] == (389.0, 1.0)
    assert schedule[390] == (390.0, 15.0)
    assert schedule[-1] == (1425.0, 15.0)
    assert len(schedule) == 390 + 70


def test_shared_seeds_and_softness_unchanged() -> None:
    assert len(shared_eval_seeds(0)) == 8
    assert shared_eval_seeds(0) == shared_eval_seeds(0)
    assert fixed_softness() == 0.06
