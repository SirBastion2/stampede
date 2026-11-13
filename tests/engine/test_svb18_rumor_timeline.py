"""SVB-18a: one host copy of the lead lanes' attention per tick, not one per seed.

The full precompute in the original SVB-18 plan is blocked: M03's _draw_reposts reads the
lead lane's live per-node attention (change request, 2026-11-02). This keeps the exact old
semantics and only batches the copies. The last test is a diagnostic for the coupling that
lead-lane reposts create between candidates of a seed (see SVB-18b).
"""
from __future__ import annotations

import numpy as np
import pytest

import engine.gpu_sim as gpu_sim
from calibration.fit import _default_svb_scenario, fixed_softness
from engine.agents.population import clear_sample_population_cache
from engine.gpu_sim import clear_static_pack_cache, simulate_requested_fractions

_N_DEPOSITORS = 2000
_SEEDS = (17, 23)


def _silence_worker_threads(monkeypatch) -> None:
    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)


def _gpu_full_env(monkeypatch) -> None:
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", str(_N_DEPOSITORS))
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")


def _vectors() -> list[np.ndarray]:
    prior = np.full(15, 0.5, dtype=np.float64)
    prior[1] = 0.55
    hot = np.full(15, 0.5, dtype=np.float64)
    hot[1] = 0.62
    hot[12] = 0.80
    hot[14] = 0.75
    return [prior, hot, np.full(15, 0.45, dtype=np.float64)]


def _run(minutes: int, vectors=None) -> np.ndarray:
    clear_sample_population_cache()
    clear_static_pack_cache()
    ticks = [(float(t), 1.0) for t in range(minutes)]
    return simulate_requested_fractions(
        _default_svb_scenario(_SEEDS[0]), ticks, vectors or _vectors(), list(_SEEDS),
        softness=fixed_softness(),
    )


def test_at_most_one_host_copy_per_tick(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    _gpu_full_env(monkeypatch)
    calls = {"n": 0}
    original = gpu_sim._to_host

    def _counting(array):
        calls["n"] += 1
        return original(array)

    monkeypatch.setattr(gpu_sim, "_to_host", _counting)
    minutes = 40
    _run(minutes)
    assert calls["n"] <= minutes + 1            # one lane-row copy per tick (SVB-66), plus the final result copy


def test_candidates_do_not_affect_each_other(monkeypatch) -> None:
    """Change only candidate 0. Candidate 1's results should not move (SVB-66, was SVB-18b xfail)."""
    _silence_worker_threads(monkeypatch)
    _gpu_full_env(monkeypatch)
    vectors = _vectors()
    base = _run(390, vectors)
    changed = [v.copy() for v in vectors]
    changed[0][12] = 0.95                        # candidate 0 only: much higher feedback gain
    changed[0][1] = 0.70
    moved = _run(390, changed)
    n_vec = len(vectors)
    for seed_i in range(len(_SEEDS)):
        lane1 = seed_i * n_vec + 1
        np.testing.assert_allclose(base[lane1], moved[lane1], rtol=1e-9, atol=0.0)
