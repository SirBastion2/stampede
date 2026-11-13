"""Board ``simulate_logs`` requested fractions match the lean fit path."""

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
    calibration_uses_scripted_drawdown,
    clear_static_pack_cache,
    gpu_sim_enabled,
    relative_error,
    simulate_logs,
    simulate_requested_fractions,
)

_ATOL = 1e-4
_N_DEPOSITORS = 48
_MINUTES = 30
_SEED = 3060


def _silence_worker_threads(monkeypatch) -> None:
    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)


def _gpu_full_env(monkeypatch) -> None:
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", str(_N_DEPOSITORS))
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")


def _raise_halt(monkeypatch) -> None:
    monkeypatch.setattr("mocks.support.ASSUMED_HALT_DRAWDOWN", 1.0)
    monkeypatch.setattr("engine.gpu_full_io.ASSUMED_HALT_DRAWDOWN", 1.0)
    monkeypatch.setattr("mocks.l3_markets.ASSUMED_HALT_DRAWDOWN", 1.0)


def test_helper_is_off_when_registry_says_no() -> None:
    assert calibration_uses_scripted_drawdown() is False


def test_simulate_logs_requested_matches_lean_at_seed_3060(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    _gpu_full_env(monkeypatch)
    _raise_halt(monkeypatch)
    assert gpu_sim_enabled() is True
    assert calibration_uses_scripted_drawdown() is False
    clear_sample_population_cache()
    clear_static_pack_cache()

    scenario = _default_svb_scenario(_SEED)
    vector = np.full(15, 0.5, dtype=np.float64)
    vector[12] = 0.70
    vector[14] = 0.65
    ticks = [(float(t), 1.0) for t in range(_MINUTES)]
    try:
        logs = simulate_logs(
            scenario,
            _MINUTES,
            [vector],
            array_module="numpy",
            disable_caps=True,
            softness=fixed_softness(),
            clamp_inputs=False,
            seeds=(_SEED,),
            schedule=ticks,
            aware0=1.0,
        )
        lean = simulate_requested_fractions(
            scenario,
            ticks,
            [vector],
            (_SEED,),
            array_module="numpy",
            softness=fixed_softness(),
            clamp_inputs=False,
            aware0=1.0,
        )
    except ImportError as exc:
        pytest.skip(f"needs a CUDA device or CuPy: {exc}")
    day1, next_open, _closed = _score_outflow_log(logs[0], requested=True)
    assert lean.shape == (1, 3)
    assert relative_error(float(lean[0, 0]), float(day1)) <= _ATOL
    assert relative_error(float(lean[0, 1]), float(next_open)) <= _ATOL
    clear_static_pack_cache()
    clear_sample_population_cache()
