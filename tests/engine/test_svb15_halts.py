"""SVB-15: temporary halts freeze the print; unscripted zero-impact stock is flat."""

from __future__ import annotations

import numpy as np

from calibration.fit import _default_svb_scenario, fixed_softness
from engine.agents.population import clear_sample_population_cache
from engine.gpu_sim import (
    _step_halt_left,
    clear_static_pack_cache,
    simulate_requested_fractions,
)
from mocks.support import ASSUMED_HALT_MINUTES

_ATOL = 1e-6
_N_DEPOSITORS = 48
_SEEDS = (7,)
_HALT_AT = 4


def _silence_worker_threads(monkeypatch) -> None:
    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)


def _gpu_full_env(monkeypatch, n_depositors: int = _N_DEPOSITORS) -> None:
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", str(n_depositors))
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")


def test_halt_freezes_price_then_responds_to_withdrawals() -> None:
    """Force a halt at minute m: print holds for halt_minutes, then impact resumes."""
    halt_minutes = float(ASSUMED_HALT_MINUTES)
    assert halt_minutes == 5.0
    dt = 1.0
    quoted = np.array([100.0, 100.0])
    factor = np.ones(2, dtype=np.float64)
    halt_left = np.zeros(2, dtype=np.float64)
    prev = np.array([100.0, 100.0])
    move = 0.02
    prices: list[np.ndarray] = []
    ticks = int(_HALT_AT + halt_minutes + 3)
    for t in range(ticks):
        newly = np.array([t == _HALT_AT, t == _HALT_AT])
        halt_left, live = _step_halt_left(np, halt_left, newly, dt, halt_minutes)
        live_f = np.where(live, 1.0, 0.0)
        factor = factor * np.exp(-move * live_f)
        price = np.where(live, quoted * factor, prev)
        prices.append(np.asarray(price, dtype=np.float64).copy())
        prev = price.copy()

    frozen = prices[_HALT_AT]
    for offset in range(int(halt_minutes)):
        assert np.allclose(prices[_HALT_AT + offset], frozen, atol=_ATOL)
    assert prices[_HALT_AT - 1][0] < prices[0][0]
    released = prices[_HALT_AT + int(halt_minutes)]
    assert released[0] < frozen[0] - _ATOL
    assert released[1] < frozen[1] - _ATOL


def test_unscripted_zero_impact_stock_ratio_is_one(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    _gpu_full_env(monkeypatch)
    # Keep attention at 0 so the unscripted quote stays at the open; a 12-vector
    # already leaves impact and news gain at 0.
    monkeypatch.setattr(
        "engine.gpu_sim._add_hub_shock",
        lambda pending, *args, **kwargs: pending,
    )
    clear_sample_population_cache()
    clear_static_pack_cache()

    scenario = _default_svb_scenario(11).model_copy(update={"rumors": []})
    vector = np.full(12, 0.5, dtype=np.float64)
    ticks = [(float(t), 1.0) for t in range(24)]
    lean = simulate_requested_fractions(
        scenario,
        ticks,
        [vector],
        _SEEDS,
        array_module="numpy",
        softness=fixed_softness(),
        clamp_inputs=False,
    )
    assert lean.shape == (1, 3)
    assert abs(float(lean[0, 2]) - 1.0) <= _ATOL
    clear_static_pack_cache()
    clear_sample_population_cache()
