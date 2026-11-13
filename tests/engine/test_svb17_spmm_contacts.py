"""SVB-17 end to end: the lean stage-1 path with per-seed SpMM matches the expanded-gather path.

Mirrors test_svb07's environment (NumPy in CI, CuPy on the 3060). The kernel-level
cases live in test_svb17_kernel_parity.py.
"""
from __future__ import annotations

import numpy as np

from calibration.fit import _default_svb_scenario, fixed_softness
from engine.agents.population import clear_sample_population_cache
from engine.gpu_sim import clear_static_pack_cache, simulate_requested_fractions

_RTOL = 1e-5
_N_DEPOSITORS = 2000
_MINUTES = 30
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


def _run(monkeypatch, *, gather: bool) -> np.ndarray:
    clear_sample_population_cache()
    clear_static_pack_cache()
    if gather:
        monkeypatch.setenv("STAMPEDE_CALIB_GATHER_PATH", "1")
    else:
        monkeypatch.delenv("STAMPEDE_CALIB_GATHER_PATH", raising=False)
    scenario = _default_svb_scenario(_SEEDS[0])
    ticks = [(float(t), 1.0) for t in range(_MINUTES)]
    return simulate_requested_fractions(
        scenario, ticks, _vectors(), list(_SEEDS), softness=fixed_softness()
    )


def _max_rel(left: np.ndarray, right: np.ndarray) -> float:
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-30)
    return float(np.max(np.abs(left - right) / scale))


def test_spmm_path_matches_gather_path(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    _gpu_full_env(monkeypatch)
    old = _run(monkeypatch, gather=True)
    new = _run(monkeypatch, gather=False)
    assert old.shape == new.shape == (len(_SEEDS) * len(_vectors()), 3)
    assert _max_rel(old, new) < _RTOL


def test_contacts_are_never_expanded_per_lane(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    _gpu_full_env(monkeypatch)
    import engine.gpu_sim as gpu_sim

    seen: list[int] = []
    original = gpu_sim.step_depositor_batch

    def _spy(xp, **kwargs):
        contacts = kwargs.get("contacts")
        seen.append(int(getattr(contacts, "ndim", 0)))
        assert kwargs.get("adjacency") is not None
        return original(xp, **kwargs)

    monkeypatch.setattr(gpu_sim, "step_depositor_batch", _spy)
    _run(monkeypatch, gather=False)
    assert seen and max(seen) <= 2
