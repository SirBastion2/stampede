"""SVB-07: lean stage-1 fractions match event-scored simulate_logs."""

from __future__ import annotations

import numpy as np

from calibration.fit import (
    _DAY1_CLOSE,
    _NEXT_OPEN,
    _default_svb_scenario,
    _score_day1_stock_ratio,
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

_ATOL = 1e-6
_N_DEPOSITORS = 2000
_MINUTES = 60
_SEEDS = (17, 23)


def _silence_worker_threads(monkeypatch) -> None:
    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)


def _parity_like_with_like(monkeypatch) -> None:
    # Both quote paths honor the M27 scripted-drawdown row (default `no`).
    # Raise the halt threshold so temporary lean-halt vs sticky event-halt
    # cannot mix the comparison.
    monkeypatch.setattr("mocks.support.ASSUMED_HALT_DRAWDOWN", 1.0)
    monkeypatch.setattr("engine.gpu_full_io.ASSUMED_HALT_DRAWDOWN", 1.0)
    monkeypatch.setattr("mocks.l3_markets.ASSUMED_HALT_DRAWDOWN", 1.0)


def _gpu_full_env(monkeypatch, n_depositors: int = _N_DEPOSITORS) -> None:
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", str(n_depositors))
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")


def _vectors() -> list[np.ndarray]:
    """Three 15-vectors: zero feedback, prior feedback, and raised gain/impact."""
    zero = np.full(12, 0.5, dtype=np.float64)
    prior = np.full(15, 0.5, dtype=np.float64)
    prior[1] = 0.55
    hot = np.full(15, 0.5, dtype=np.float64)
    hot[1] = 0.62
    hot[12] = 0.80
    hot[14] = 0.75
    return [zero, prior, hot]


def _schedule(minutes: int) -> list[tuple[float, float]]:
    return [(float(t), 1.0) for t in range(minutes)]


def _scored(log: list[object]) -> np.ndarray:
    day1, next_open, _closed = _score_outflow_log(log, requested=True)
    stock = _score_day1_stock_ratio(log)
    return np.asarray(
        [day1, next_open, 0.0 if stock is None else float(stock)],
        dtype=np.float64,
    )


def test_lean_stage1_matches_event_scores_including_feedback(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    _parity_like_with_like(monkeypatch)
    _gpu_full_env(monkeypatch)
    clear_sample_population_cache()
    clear_static_pack_cache()

    scenario = _default_svb_scenario(_SEEDS[0])
    vectors = _vectors()
    ticks = _schedule(_MINUTES)
    built = {"n": 0}
    original = __import__("engine.gpu_sim", fromlist=["_build_engine"])._build_engine

    def _counting(item):
        built["n"] += 1
        return original(item)

    monkeypatch.setattr("engine.gpu_sim._build_engine", _counting)

    logs = simulate_logs(
        scenario,
        _MINUTES,
        vectors,
        array_module="numpy",
        disable_caps=True,
        softness=fixed_softness(),
        clamp_inputs=False,
        seeds=_SEEDS,
        schedule=ticks,
        aware0=1.0,
    )
    event_n = built["n"]
    built["n"] = 0
    lean = simulate_requested_fractions(
        scenario,
        ticks,
        vectors,
        _SEEDS,
        array_module="numpy",
        softness=fixed_softness(),
        clamp_inputs=False,
        aware0=1.0,
    )
    assert built["n"] == 0
    assert event_n == len(vectors) * len(_SEEDS)
    assert lean.shape == (len(vectors) * len(_SEEDS), 3)
    scored = np.stack([_scored(log) for log in logs])
    for lane in range(lean.shape[0]):
        for col in range(3):
            assert relative_error(float(lean[lane, col]), float(scored[lane, col])) <= _ATOL

    assert scored[2, 0] > scored[0, 0]
    assert _DAY1_CLOSE == 390
    assert _NEXT_OPEN == 1440
    clear_static_pack_cache()
    clear_sample_population_cache()


def test_lean_stage1_dt_schedule_matches_events(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    _parity_like_with_like(monkeypatch)
    _gpu_full_env(monkeypatch, n_depositors=256)
    clear_sample_population_cache()
    clear_static_pack_cache()

    scenario = _default_svb_scenario(11)
    vector = np.full(15, 0.5, dtype=np.float64)
    vector[12] = 0.70
    vector[14] = 0.65
    ticks = [(float(t * 5), 5.0) for t in range(12)]
    logs = simulate_logs(
        scenario,
        12,
        [vector],
        array_module="numpy",
        disable_caps=True,
        softness=fixed_softness(),
        clamp_inputs=False,
        seeds=(11, 19),
        schedule=ticks,
        aware0=1.0,
    )
    lean = simulate_requested_fractions(
        scenario,
        ticks,
        [vector],
        (11, 19),
        array_module="numpy",
        softness=fixed_softness(),
        clamp_inputs=False,
        aware0=1.0,
    )
    scored = np.stack([_scored(log) for log in logs])
    for lane in range(2):
        for col in range(3):
            assert relative_error(float(lean[lane, col]), float(scored[lane, col])) <= _ATOL
    clear_static_pack_cache()
    clear_sample_population_cache()
