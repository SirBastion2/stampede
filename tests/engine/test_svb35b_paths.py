"""SVB-35b: lean / event / float32 / fused once-amount agree at finite h."""

from __future__ import annotations

import numpy as np
import pytest

from calibration.fit import (
    _default_svb_scenario,
    _score_outflow_log,
    fixed_softness,
)
from engine.agents.population import clear_sample_population_cache
from engine.gpu_sim import (
    clear_static_pack_cache,
    relative_error,
    simulate_logs,
    simulate_requested_fractions,
)
from engine.gpu_sim_kernels import _once_amount_fallback, _part2_kernels_enabled
from tests.engine.test_svb28_decide_once import _run, _xp_cases

_ATOL = 1e-6
_F32_RTOL = 1e-4
_N_DEPOSITORS = 48
_MINUTES = 30
_SEEDS = (11,)
_PRIOR = 1.0 / 600.0
_FASTER = 1.0 / 60.0
_H_INSTANT = 1e9


def _silence_worker_threads(monkeypatch) -> None:
    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)


def _parity_like_with_like(monkeypatch) -> None:
    monkeypatch.setattr("mocks.support.ASSUMED_HALT_DRAWDOWN", 1.0)
    monkeypatch.setattr("engine.gpu_full_io.ASSUMED_HALT_DRAWDOWN", 1.0)
    monkeypatch.setattr("mocks.l3_markets.ASSUMED_HALT_DRAWDOWN", 1.0)


def _gpu_full_env(monkeypatch) -> None:
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", str(_N_DEPOSITORS))
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")


def _pin_h(monkeypatch, h: float) -> None:
    monkeypatch.setattr(
        "engine.agents.decide.decision_to_request_rate_default",
        lambda path=None, value=h: float(value),
    )


@pytest.mark.parametrize("h", [_PRIOR, _FASTER])
def test_lean_event_float32_agree_at_finite_h(monkeypatch, h: float) -> None:
    _silence_worker_threads(monkeypatch)
    _parity_like_with_like(monkeypatch)
    _gpu_full_env(monkeypatch)
    _pin_h(monkeypatch, h)
    clear_sample_population_cache()
    clear_static_pack_cache()

    scenario = _default_svb_scenario(_SEEDS[0])
    vector = np.full(15, 0.5, dtype=np.float64)
    vector[12] = 0.70
    vector[14] = 0.65
    ticks = [(float(t), 1.0) for t in range(_MINUTES)]
    logs = simulate_logs(
        scenario,
        _MINUTES,
        [vector],
        array_module="numpy",
        disable_caps=True,
        softness=fixed_softness(),
        clamp_inputs=False,
        seeds=_SEEDS,
        schedule=ticks,
        aware0=1.0,
    )
    lean = simulate_requested_fractions(
        scenario,
        ticks,
        [vector],
        _SEEDS,
        array_module="numpy",
        softness=fixed_softness(),
        clamp_inputs=False,
        aware0=1.0,
    )
    assert lean.shape == (1, 3)
    day1, next_open, _closed = _score_outflow_log(logs[0], requested=True)
    assert relative_error(float(lean[0, 0]), float(day1)) <= _ATOL
    assert relative_error(float(lean[0, 1]), float(next_open)) <= _ATOL
    assert relative_error(float(lean[0, 0]), float(day1)) <= _F32_RTOL
    clear_static_pack_cache()
    clear_sample_population_cache()


def test_fused_once_amount_matches_numpy_fallback_at_finite_h() -> None:
    cases = _xp_cases()
    if len(cases) < 2:
        pytest.skip("needs a CUDA device")
    import os

    cupy = cases[1][1]
    assert _part2_kernels_enabled(cupy)
    for h in (_PRIOR, _FASTER, _H_INSTANT):
        fused, _ = _run(
            cupy,
            40,
            decide_once=True,
            softness=0.06,
            threshold=0.5,
            attention=0.7,
            decision_to_request_rate=h,
        )
        os.environ["STAMPEDE_CALIB_UNFUSED_PART2"] = "1"
        try:
            fallback, _ = _run(
                cupy,
                40,
                decide_once=True,
                softness=0.06,
                threshold=0.5,
                attention=0.7,
                decision_to_request_rate=h,
            )
        finally:
            os.environ.pop("STAMPEDE_CALIB_UNFUSED_PART2", None)
        assert np.allclose(fused, fallback, rtol=1e-4, atol=1e-6), h
    assert _once_amount_fallback is not None
