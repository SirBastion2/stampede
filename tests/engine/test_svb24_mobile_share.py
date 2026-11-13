"""SVB-24: mobile share of uninsured deposits caps cumulative requests."""

from __future__ import annotations

import os

import numpy as np
import pytest

from calibration.fit import (
    _MOBILE_SHARE_INDEX,
    _VECTOR_PARAM_NAMES,
    _theta_bounds,
    identifiable_indices,
    identifiable_names,
)
from engine.agents.decide import (
    calibration_mobile_share,
    mobile_share_bounds,
    mobile_share_default,
)
from engine.gpu_sim_kernels import make_depositor_scratch, step_depositor_batch
from params.loader import get_parameter

_UNFUSED = "STAMPEDE_CALIB_UNFUSED_PART2"
_RTOL = 1e-6


def _force_world(xp, *, n=24, candidates=2, seed=24):
    """Small book that withdraws every tick (threshold and k cleared)."""
    rng = np.random.default_rng(seed)
    f32 = np.float32
    balance = (rng.uniform(2.0e5, 2.0e6, n)).astype(f32)
    insured = np.full(n, 2.5e5, dtype=f32)
    initial_uninsured = np.maximum(0.0, balance - insured).astype(f32)
    state = dict(
        belief=np.full((candidates, n), 0.4, dtype=f32),
        balance=np.broadcast_to(balance, (candidates, n)).copy(),
        already=np.zeros((candidates, n), dtype=bool),
        insured=insured,
        base_threshold=np.full((candidates, n), -1.0, dtype=f32),
        threshold_shift=np.zeros((candidates, n), dtype=f32),
        statement_shift=np.zeros((candidates, n), dtype=f32),
        deposit_rate=np.zeros((candidates, n), dtype=f32),
        directive=np.zeros((candidates, n), dtype=f32),
        payroll=np.zeros(n, dtype=f32),
        segment_fraction=np.full(n, 0.8, dtype=f32),
        vip_amount=np.zeros((candidates, n), dtype=f32),
        hub=np.ones(n, dtype=f32),
        contacts=np.zeros((n, 0), dtype=np.int64),
        initial_uninsured=initial_uninsured,
    )
    params = dict(
        attention_level=np.ones(candidates, dtype=f32),
        stock_return=np.full(candidates, -0.1, dtype=f32),
        gap_scale=np.ones(candidates, dtype=f32),
        amount_scale=np.ones(candidates, dtype=f32),
        weights=np.zeros((candidates, 7), dtype=f32),
        bias=np.zeros(candidates, dtype=f32),
        k=np.zeros(candidates, dtype=f32),
        belief_strength=np.full(candidates, 0.2, dtype=f32),
        signal_attention=np.full(candidates, 0.4, dtype=f32),
        signal_stock=np.full(candidates, 0.3, dtype=f32),
        signal_base=np.full(candidates, 0.5, dtype=f32),
        rate_relief=np.zeros(candidates, dtype=f32),
    )
    to = lambda d: {key: xp.asarray(value) for key, value in d.items()}
    return to(state), to(params), initial_uninsured, float(balance.sum())


def _run_ticks(xp, state, params, *, mobile_share, ticks, softness=0.0, dt=1.0):
    st = {key: value.copy() for key, value in state.items()}
    n_c, n = st["belief"].shape
    share = np.asarray(mobile_share, dtype=np.float32)
    if share.ndim == 0:
        share = np.full(n_c, float(share), dtype=np.float32)
    requested = np.zeros((n_c, n), dtype=np.float64)
    for _ in range(ticks):
        amounts = step_depositor_batch(
            xp,
            **{key: st[key] for key in st if key != "initial_uninsured"},
            **params,
            minimum_amount=1.0,
            chunk=11,
            softness=softness,
            scratch=make_depositor_scratch(xp, n_c, n, softness),
            dt=dt,
            mobile_share=xp.asarray(share),
            initial_uninsured=st["initial_uninsured"],
        )
        host = np.asarray(getattr(amounts, "get", lambda: amounts)())
        requested += host.astype(np.float64)
    return requested, st


def test_registry_row_and_bounds() -> None:
    row = get_parameter("Depositor mobile share of uninsured deposits (FIT)")
    # SVB-45: the value is whatever the last approved persist wrote, so pin status, bounds and
    # the loader, not the number (it was 0.878938 for svb26 and is 0.963547 since svb37).
    value = float(row.value)
    assert row.status == "FITTED"
    assert row.module == "M06"
    assert mobile_share_bounds() == (0.30, 1.00)
    lo, hi = mobile_share_bounds()
    assert lo <= value <= hi
    assert mobile_share_default() == pytest.approx(value)


def test_fit_vector_includes_mobile_share_with_registry_bounds() -> None:
    assert "mobile_share" in _VECTOR_PARAM_NAMES
    assert _VECTOR_PARAM_NAMES.index("mobile_share") == _MOBILE_SHARE_INDEX
    assert _MOBILE_SHARE_INDEX in identifiable_indices()
    assert identifiable_names()[-1] == "decision_to_request_rate"
    lo, hi = mobile_share_bounds()
    assert (lo, hi) == (0.30, 1.00)
    t_lo, t_hi = _theta_bounds(_MOBILE_SHARE_INDEX)
    default = mobile_share_default()
    assert default * np.exp(t_lo) == pytest.approx(lo)
    assert default * np.exp(t_hi) == pytest.approx(hi)
    dim = int(float(get_parameter("Calibration behavioral parameter dimension").value))
    center = np.full(dim, 0.5, dtype=np.float64)
    assert calibration_mobile_share(center) == pytest.approx(mobile_share_default())
    assert calibration_mobile_share(np.full(15, 0.5)) == pytest.approx(1.0)


def test_cumulative_requested_never_exceeds_mobile_share() -> None:
    xp = np
    state, params, initial_u, _book = _force_world(xp)
    # VIP forced p_act=1 on the first few rows of lane 0; they are capped too.
    vip = np.asarray(state["vip_amount"])
    vip[0, :4] = 1.0e7
    state["vip_amount"] = vip
    for share in (0.5, 0.88):
        requested, _ = _run_ticks(xp, state, params, mobile_share=share, ticks=40)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(initial_u > 0.0, requested / initial_u, 0.0)
        assert np.all(frac <= share + 1e-5)
        uninsured_share = float(initial_u.sum() / _book) if _book else 0.0
        lane_frac = requested.sum(axis=1) / _book
        assert np.all(lane_frac <= share * uninsured_share + 1e-5)


def test_mobile_share_one_matches_old_amounts() -> None:
    xp = np
    state, params, _initial_u, _book = _force_world(xp, n=32, candidates=1)
    vip = np.asarray(state["vip_amount"])
    vip[0, :3] = 5.0e5
    state["vip_amount"] = vip
    old, _ = _run_ticks(xp, state, params, mobile_share=1.0, ticks=12)
    st = {key: value.copy() for key, value in state.items()}
    n_c, n = st["belief"].shape
    requested = np.zeros((n_c, n), dtype=np.float64)
    for _ in range(12):
        amounts = step_depositor_batch(
            xp,
            **{key: st[key] for key in st if key != "initial_uninsured"},
            **params,
            minimum_amount=1.0,
            chunk=11,
            softness=0.0,
            scratch=make_depositor_scratch(xp, n_c, n, 0.0),
            dt=1.0,
        )
        requested += np.asarray(amounts, dtype=np.float64)
    np.testing.assert_allclose(old, requested, rtol=_RTOL, atol=1e-6)


def test_fused_and_unfused_agree(monkeypatch) -> None:
    xp = np
    state, params, _initial_u, _book = _force_world(xp)
    shares = np.array([0.5, 0.88], dtype=np.float32)
    monkeypatch.setenv(_UNFUSED, "1")
    try:
        unfused, _ = _run_ticks(xp, state, params, mobile_share=shares, ticks=8)
    finally:
        monkeypatch.delenv(_UNFUSED, raising=False)
    fused, _ = _run_ticks(xp, state, params, mobile_share=shares, ticks=8)
    np.testing.assert_allclose(fused, unfused, rtol=_RTOL, atol=1e-6)
    assert os.environ.get(_UNFUSED, "") != "1"


def test_per_lane_mobile_share_differs_per_candidate(monkeypatch) -> None:
    from calibration.fit import _default_svb_scenario
    from engine.agents.population import clear_sample_population_cache
    from engine.gpu_sim import _pack_lean_lanes, clear_static_pack_cache

    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "32")
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    clear_sample_population_cache()
    clear_static_pack_cache()
    low = np.full(16, 0.5, dtype=np.float64)
    high = np.full(16, 0.5, dtype=np.float64)
    low[_MOBILE_SHARE_INDEX] = 0.0
    high[_MOBILE_SHARE_INDEX] = 1.0
    assert calibration_mobile_share(low, clamp_inputs=False) != calibration_mobile_share(
        high, clamp_inputs=False
    )
    state = _pack_lean_lanes(
        _default_svb_scenario(24),
        [low, high],
        np,
        clamp_inputs=False,
    )
    shares = np.asarray(state["mobile_share"], dtype=np.float64)
    assert shares.shape == (2,)
    assert shares[0] < shares[1]
    assert shares[0] == pytest.approx(calibration_mobile_share(low, clamp_inputs=False))
    assert shares[1] == pytest.approx(calibration_mobile_share(high, clamp_inputs=False))
    clear_static_pack_cache()
    clear_sample_population_cache()
