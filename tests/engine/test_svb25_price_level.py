"""SVB-25: level-targeting price impact and a 5-minute halt."""

from __future__ import annotations

import math

import numpy as np
import pytest

from calibration.fit import _default_svb_scenario, fixed_softness
from engine.agents.population import clear_sample_population_cache
from engine.gpu_full_io import (
    apply_price_impact,
    attention_factor,
    price_impact_bounds,
    price_impact_default,
    price_impact_move,
)
from engine.gpu_sim import _step_halt_left, clear_static_pack_cache, simulate_requested_fractions
from mocks.support import ASSUMED_HALT_MINUTES, OPENING_PRICE, assumed_halt_minutes
from params.loader import get_parameter

_ATOL = 1e-12
_N_DEPOSITORS = 48
_SEEDS = (7,)
_SATURATED_SHARE = 0.813


def _silence_worker_threads(monkeypatch) -> None:
    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)


def _gpu_full_env(monkeypatch, n_depositors: int = _N_DEPOSITORS) -> None:
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", str(n_depositors))
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")


def _registry_float(name: str) -> float:
    return float(get_parameter(name).value)


def test_registry_svb25_rows() -> None:
    impact = get_parameter("M14 price impact (ASSUMED)")
    assert impact.value == "1.18464"
    # SVB-36: carried from the pre-SVB-28 fit and no longer refit, so labeled ASSUMED.
    assert impact.status == "ASSUMED"
    assert "SVB-36" in impact.source
    assert price_impact_default() == pytest.approx(1.18464)
    assert price_impact_bounds() == (0.3, 4.0)

    ref = get_parameter("M14 attention reference level")
    assert ref.value == "1.0"
    assert ref.status == "ASSUMED"
    assert "SVB-25" in ref.source

    halt = get_parameter("M14 trading halt minutes")
    assert halt.value == "5"
    assert halt.status == "SOURCED"
    assert "SVB-25" in halt.source
    assert assumed_halt_minutes() == 5.0
    assert float(ASSUMED_HALT_MINUTES) == 5.0


def test_constant_share_reaches_target_then_stays_flat() -> None:
    impact = _registry_float("M14 price impact (ASSUMED)")
    cap = _registry_float("M14 price impact cap per minute")
    target = impact * _SATURATED_SHARE * 1.0
    price = np.array([OPENING_PRICE], dtype=np.float64)
    steps: list[float] = []
    for _ in range(390):
        nxt = apply_price_impact(
            np,
            price,
            _SATURATED_SHARE,
            impact,
            1.0,
            open_price=OPENING_PRICE,
            attention_level=1.0,
        )
        step = float(price_impact_move(
            np,
            price,
            _SATURATED_SHARE,
            impact,
            1.0,
            open_price=OPENING_PRICE,
            attention_level=1.0,
        )[0])
        steps.append(step)
        price = nxt
    drop = -math.log(float(price[0]) / OPENING_PRICE)
    assert abs(drop - target) <= 0.005 * target
    held = apply_price_impact(
        np,
        price,
        _SATURATED_SHARE,
        impact,
        1.0,
        open_price=OPENING_PRICE,
        attention_level=1.0,
    )
    assert float(held[0]) == pytest.approx(float(price[0]), abs=1e-12)
    assert max(abs(step) for step in steps) <= cap + 1e-15


def test_per_tick_step_never_exceeds_cap() -> None:
    cap = _registry_float("M14 price impact cap per minute")
    price = np.full(4, OPENING_PRICE, dtype=np.float64)
    shares = np.array([0.0, 0.02, 0.5, 4.0], dtype=np.float64)
    for dt in (1.0, 5.0, 15.0):
        move = price_impact_move(np, price, shares, 10.0, dt, noise=0.0, open_price=OPENING_PRICE)
        assert np.all(move <= cap * dt + 1e-15)
        assert np.all(move >= -cap * dt - 1e-15)


def test_zero_target_leaves_price_unchanged() -> None:
    quoted = np.array([OPENING_PRICE, OPENING_PRICE], dtype=np.float64)
    share = np.array([0.04, 0.00], dtype=np.float64)
    zero = apply_price_impact(np, quoted, share, 0.0, dt=1.0, noise=0.0)
    assert float(zero[0]) == OPENING_PRICE
    assert float(zero[1]) == OPENING_PRICE
    flat = apply_price_impact(np, quoted, 0.0, 1.13, dt=1.0, noise=0.0)
    assert np.allclose(flat, quoted)
    assert float(attention_factor(np, 0.4, attention_ref=1.0)) == pytest.approx(0.4)
    assert float(attention_factor(np, 2.0, attention_ref=1.0)) == pytest.approx(1.0)


def test_halt_freezes_price_for_exactly_five_minutes() -> None:
    halt_minutes = float(ASSUMED_HALT_MINUTES)
    assert halt_minutes == 5.0
    dt = 1.0
    impact = _registry_float("M14 price impact (ASSUMED)")
    price = np.array([OPENING_PRICE, OPENING_PRICE], dtype=np.float64)
    halt_left = np.zeros(2, dtype=np.float64)
    halt_at = 4
    frozen = None
    prices: list[np.ndarray] = []
    for t in range(int(halt_at + halt_minutes + 3)):
        newly = np.array([t == halt_at, t == halt_at])
        halt_left, live = _step_halt_left(np, halt_left, newly, dt, halt_minutes)
        nxt = apply_price_impact(
            np,
            price,
            _SATURATED_SHARE,
            impact,
            dt,
            open_price=OPENING_PRICE,
            attention_level=1.0,
        )
        price = np.where(live, nxt, price)
        prices.append(np.asarray(price, dtype=np.float64).copy())
        if t == halt_at:
            frozen = prices[-1].copy()

    assert frozen is not None
    for offset in range(int(halt_minutes)):
        assert np.allclose(prices[halt_at + offset], frozen, atol=_ATOL)
    released = prices[halt_at + int(halt_minutes)]
    assert released[0] < frozen[0] - _ATOL
    assert released[1] < frozen[1] - _ATOL


def test_recovery_step_bounded_by_cap() -> None:
    cap = _registry_float("M14 price impact cap per minute")
    dt = 1.0
    price = np.array([20.0], dtype=np.float64)
    nxt = apply_price_impact(
        np,
        price,
        0.0,
        1.13,
        dt,
        open_price=OPENING_PRICE,
        attention_level=1.0,
    )
    step = float(-np.log(nxt / price)[0])
    assert -cap * dt - 1e-15 <= step <= cap * dt + 1e-15
    assert step == pytest.approx(-cap * dt)
    assert float(nxt[0]) > float(price[0])


def test_lean_stock_falls_below_0_6_after_saturation(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    _gpu_full_env(monkeypatch)
    # Instant-ratchet limit: this case wants a saturated book in one session.
    # At the registry prior h=1/600 the model ramps over about a day (stock ~1.03).
    monkeypatch.setattr(
        "engine.agents.decide.decision_to_request_rate_default",
        lambda path=None: 1e9,
    )
    clear_sample_population_cache()
    clear_static_pack_cache()
    scenario = _default_svb_scenario(11)
    vector = np.full(16, 0.5, dtype=np.float64)
    ticks = [(float(t), 1.0) for t in range(390)]
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
    assert float(lean[0, 2]) < 0.6
    clear_static_pack_cache()
    clear_sample_population_cache()
