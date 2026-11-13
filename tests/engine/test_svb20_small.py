"""SVB-20: skip the float64 amounts copy; shift segments only when a flag was set."""

from __future__ import annotations

import numpy as np

import engine.gpu_sim as gpu_sim
from calibration.fit import _default_svb_scenario, fixed_softness
from engine.agents.population import clear_sample_population_cache
from engine.gpu_sim import (
    _DAY1_CLOSE,
    _NEXT_OPEN,
    _scoring_deposits,
    clear_static_pack_cache,
    simulate_requested_fractions,
)

_N_DEPOSITORS = 48
_MINUTES = 24
_SEEDS = (20, 21)


def _silence_worker_threads(monkeypatch) -> None:
    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)


def _gpu_full_env(monkeypatch, n_depositors: int = _N_DEPOSITORS) -> None:
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", str(n_depositors))
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")


def _vector() -> np.ndarray:
    vector = np.full(15, 0.5, dtype=np.float64)
    vector[1] = 0.55
    vector[12] = 0.70
    vector[14] = 0.65
    return vector


def _schedule(minutes: int = _MINUTES) -> list[tuple[float, float]]:
    return [(float(t), 1.0) for t in range(minutes)]


def _run_lean(scenario, ticks, seeds=_SEEDS):
    return simulate_requested_fractions(
        scenario,
        ticks,
        [_vector()],
        seeds,
        array_module="numpy",
        softness=fixed_softness(),
        clamp_inputs=False,
    )


def test_float32_sum_matches_asarray_path() -> None:
    rng = np.random.default_rng(20)
    amounts = rng.random((6, 128), dtype=np.float32)
    amounts[0, :4] = 0.0
    amounts[2] *= np.float32(1e6)
    amounts[4, 10:20] = np.float32(1e-3)
    old = np.asarray(amounts, dtype=np.float64).sum(axis=1)
    new = amounts.sum(axis=1, dtype=np.float64)
    assert np.array_equal(old, new)


def test_lean_numpy_outputs_match_previous_requested_sum(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    _gpu_full_env(monkeypatch)
    clear_sample_population_cache()
    clear_static_pack_cache()

    recorded: list[np.ndarray] = []
    real = gpu_sim.step_depositor_batch

    def capturing(*args, **kwargs):
        amounts = real(*args, **kwargs)
        recorded.append(np.asarray(amounts, dtype=np.float32).copy())
        return amounts

    monkeypatch.setattr(gpu_sim, "step_depositor_batch", capturing)

    scenario = _default_svb_scenario(_SEEDS[0])
    ticks = _schedule()
    out = _run_lean(scenario, ticks)
    assert len(recorded) == len(ticks)

    deposits = _scoring_deposits()
    denom = deposits if deposits > 0.0 else 1.0
    req_day1 = np.zeros(out.shape[0], dtype=np.float64)
    req_next = np.zeros(out.shape[0], dtype=np.float64)
    for (t, _dt), amounts in zip(ticks, recorded):
        old = np.asarray(amounts, dtype=np.float64).sum(axis=1)
        new = amounts.sum(axis=1, dtype=np.float64)
        assert np.array_equal(old, new)
        if t < _DAY1_CLOSE:
            req_day1 = req_day1 + old
        if t < _NEXT_OPEN:
            req_next = req_next + old
    assert np.array_equal(out[:, 0], req_day1 / denom)
    assert np.array_equal(out[:, 1], req_next / denom)

    clear_static_pack_cache()
    clear_sample_population_cache()


def test_set_segment_shift_only_on_flag_ticks(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    _gpu_full_env(monkeypatch)
    clear_sample_population_cache()
    clear_static_pack_cache()

    flagged: list[tuple[bool, bool]] = []
    real_note = gpu_sim._pending_host_flags

    def recording(crisis_now, mgmt_now, xp, *, read_crisis, read_mgmt=True):
        pair = real_note(
            crisis_now, mgmt_now, xp, read_crisis=read_crisis, read_mgmt=read_mgmt
        )
        flagged.append(pair)
        return pair

    monkeypatch.setattr(gpu_sim, "_pending_host_flags", recording)

    shift_calls: list[np.ndarray] = []
    real_shift = gpu_sim._set_segment_shift

    def counted(statement_shift, threshold_shift, pending, mask, logit, thresh):
        host_pending = np.asarray(pending, dtype=np.float64)
        assert np.any(host_pending > 0.0)
        shift_calls.append(host_pending.copy())
        return real_shift(statement_shift, threshold_shift, pending, mask, logit, thresh)

    monkeypatch.setattr(gpu_sim, "_set_segment_shift", counted)

    scenario = _default_svb_scenario(_SEEDS[0])
    ticks = _schedule()
    _run_lean(scenario, ticks)

    assert flagged
    expected = sum(int(crisis) + int(mgmt) for crisis, mgmt in flagged[:-1])
    assert len(shift_calls) == expected
    assert len(shift_calls) < 2 * len(ticks)
    assert any(crisis or mgmt for crisis, mgmt in flagged)

    clear_static_pack_cache()
    clear_sample_population_cache()
