"""SVB-13: stage 1 always simulates through next open."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from calibration.fit import (
    _NEXT_OPEN,
    _assert_reaches_next_open,
    _default_svb_scenario,
    _evaluate_population_gpu_sim,
    day1_tick_schedule,
    load_calibration_targets,
)
from calibration.run_fit import _parse_args


def _silence_worker_threads(monkeypatch) -> None:
    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)


def test_evaluate_population_gpu_sim_uses_next_open_schedule(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    recorded: list[list[tuple[float, float]]] = []

    def _record_fractions(_scenario, ticks, samples, seeds, **_kwargs):
        recorded.append([(float(t), float(dt)) for t, dt in ticks])
        return [[0.1, 0.2, 0.3] for _ in range(len(samples) * len(seeds))]

    monkeypatch.setattr("engine.gpu_sim.simulate_requested_fractions", _record_fractions)

    _evaluate_population_gpu_sim(
        [np.full(16, 0.3)],
        fit_seed=13,
        scenario=_default_svb_scenario(13),
        steps=390,
        targets=load_calibration_targets(),
        w_day1=1.0,
        w_day2=1.0,
        softness=0.15,
        on_candidate=None,
        on_candidate_error=None,
    )

    assert recorded
    last_t, last_dt = recorded[0][-1]
    assert last_t + last_dt >= _NEXT_OPEN


def test_assert_reaches_next_open_rejects_day1_schedule() -> None:
    with pytest.raises(ValueError):
        _assert_reaches_next_open(day1_tick_schedule())


def test_run_fit_eval_minutes_default_is_1440(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run_fit"])
    args = _parse_args()
    assert args.eval_minutes == 1440
