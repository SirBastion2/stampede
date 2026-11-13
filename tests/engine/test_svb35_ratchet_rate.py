"""SVB-35: finite-rate decide-once ratchet so requests ramp instead of stepping."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from calibration.fit import (
    _VECTOR_PARAM_NAMES,
    _theta_bounds,
    identifiable_indices,
    identifiable_names,
)
from engine.agents.decide import (
    DecisionInputs,
    _evaluate,
    awareness_media_rate_default,
    calibration_awareness_media_rate,
    calibration_decision_to_request_rate,
    decision_to_request_rate_bounds,
    decision_to_request_rate_default,
    load_decision_params,
)
from engine.gpu_sim_kernels import _once_amount_fallback
from params.loader import get_parameter

_H_INSTANT = 1e9
_PRIOR = 1.0 / 600.0
_LO = 1.0 / 5000.0
_HI = 1.0 / 20.0


def _fallback_tick(
    *,
    decided,
    h,
    dt=1.0,
    worried=1.0,
    contacts=100.0,
    k=0.0,
    softness=0.06,
    init_u=1_000.0,
    mobile=1.0,
):
    xp = np
    f32 = np.float32
    shape = (1, 1)
    decided_sl = xp.array([[decided]], dtype=f32)
    amounts = xp.zeros(shape, dtype=f32)
    balance = xp.array([[init_u + 250_000.0]], dtype=f32)
    already = xp.zeros(shape, dtype=bool)
    _once_amount_fallback(
        xp,
        safe=balance.copy(),
        gap_balance=xp.array([[init_u]], dtype=f32),
        worried_sl=xp.array([[worried]], dtype=f32),
        worried_contacts=xp.array([[contacts]], dtype=f32),
        vip_amount_sl=xp.zeros(shape, dtype=f32),
        segment_fraction_rows=xp.array([[0.6]], dtype=f32),
        amount_scale_col=xp.array([[1.0]], dtype=f32),
        k_col=xp.array([[k]], dtype=f32),
        amounts_sl=amounts,
        balance_sl=balance,
        already_sl=already,
        decided_sl=decided_sl,
        minimum_amount=1.0,
        temp=softness,
        f32=f32,
        mobile_share_col=xp.array([[mobile]], dtype=f32),
        initial_uninsured_sl=xp.array([[init_u]], dtype=f32),
        gap_scale_col=xp.array([[1.0]], dtype=f32),
        dt=dt,
        h_col=xp.array([[h]], dtype=f32),
    )
    return float(amounts[0, 0]), float(decided_sl[0, 0])


def _eval_tick(*, decided, h, dt=1.0, aware=1.0):
    n = 1
    params = replace(load_decision_params(), k=0.0, bias=8.0)
    zeros = np.zeros(n, dtype=np.float64)
    balance = np.array([1.25e6], dtype=np.float64)
    insured = np.array([250_000.0], dtype=np.float64)
    init_u = np.maximum(0.0, balance - insured)
    result = _evaluate(
        np,
        DecisionInputs(
            attention=np.ones(n),
            belief=np.full(n, params.baseline_belief),
            threshold=zeros,
            balance=balance.copy(),
            insured=insured,
            stock_return=-0.1,
            directive=zeros,
            payroll=zeros,
            contacts=np.zeros((n, 0), dtype=np.int64),
            hub_weight=np.ones(n),
            already_withdrew=np.zeros(n, dtype=np.bool_),
            statement_shift=zeros,
            deposit_rate=zeros,
            segment_fraction=np.full(n, params.default_fraction),
            vip_amount=zeros,
            decided=np.array([decided], dtype=np.float64),
            aware=np.full(n, aware),
            news=0.0,
        ),
        params,
        gap_scale=1.0,
        amount_scale=1.0,
        backend="cpu",
        softness=0.06,
        mobile_share=1.0,
        initial_uninsured=init_u,
        decide_once=True,
        decision_to_request_rate=h,
        dt=dt,
    )
    return result


def test_h_instant_matches_old_max_ratchet_bit_for_bit_fallback():
    decided0 = 0.25
    amt_fast, d_fast = _fallback_tick(decided=decided0, h=_H_INSTANT)
    amt_old, d_old = _fallback_tick(decided=decided0, h=_H_INSTANT)
    # Old formula: d_new = max(decided, p_act). With worried=1 and huge contacts, p_act=1.
    assert d_fast == 1.0
    assert d_fast == max(decided0, 1.0)
    assert amt_fast == pytest.approx((d_fast - decided0) * 1_000.0)
    assert d_fast == d_old
    assert amt_fast == amt_old
    # Direct comparison to the pre-SVB-35 one-liner on the same arrays.
    p_act = np.float32(1.0)
    d_expected = np.maximum(np.float32(decided0), p_act)
    assert np.float32(d_fast) == d_expected


def test_h_instant_matches_old_max_ratchet_bit_for_bit_evaluate():
    p_act = _eval_tick(decided=0.0, h=_H_INSTANT).decided.copy()
    decided0 = 0.25
    result = _eval_tick(decided=decided0, h=_H_INSTANT)
    expected = np.maximum(np.array([decided0], dtype=np.float64), p_act)
    assert np.array_equal(result.decided, expected)
    init_u = 1.25e6 - 250_000.0
    expected_amt = (expected - decided0) * init_u
    assert np.array_equal(result.amount, expected_amt)


def test_finite_h_is_monotone_capped_and_slower_than_instant():
    decided = 0.0
    cum = []
    decided_path = []
    p_act_ceiling = 1.0
    h = _PRIOR
    for _ in range(12):
        amt, decided = _fallback_tick(decided=decided, h=h)
        cum.append((cum[-1] if cum else 0.0) + amt)
        decided_path.append(decided)
        assert decided <= p_act_ceiling + 0.0
    assert all(later >= earlier for earlier, later in zip(cum, cum[1:]))
    instant_amt, instant_d = _fallback_tick(decided=0.0, h=_H_INSTANT)
    first_finite = _fallback_tick(decided=0.0, h=h)[0]
    assert first_finite < instant_amt
    assert decided_path[0] < instant_d
    h32 = np.float32(h)
    expected = float((np.float32(1.0) - np.exp(-h32)) * np.float32(1_000.0))
    assert first_finite == expected


def test_registry_row_and_bounds():
    row = get_parameter("M07 decision-to-request rate (FIT)")
    # SVB-45: the SVB-35 prior (1/600) was replaced by the persisted svb37 fit. Pin the status
    # vocabulary, bounds and loader, not the fitted number or the run id in the source.
    value = float(row.value)
    assert row.status in {"ASSUMED", "FITTED"}
    assert row.source.strip()
    assert row.module == "M07"
    assert row.unit == "1/min"
    lo, hi = decision_to_request_rate_bounds()
    assert lo == pytest.approx(_LO)
    assert hi == pytest.approx(_HI)
    assert lo <= value <= hi
    assert decision_to_request_rate_default() == pytest.approx(value)
    lo_row = get_parameter("M07 decision-to-request rate (FIT) min")
    hi_row = get_parameter("M07 decision-to-request rate (FIT) max")
    assert float(lo_row.value) == pytest.approx(_LO)
    assert float(hi_row.value) == pytest.approx(_HI)


def test_fit_set_reuses_idx16_for_h_and_keeps_dim_17():
    assert identifiable_indices() == (7, 15, 16)
    assert identifiable_names() == (
        "worry_bias",
        "mobile_share",
        "decision_to_request_rate",
    )
    assert _VECTOR_PARAM_NAMES[16] == "decision_to_request_rate"
    assert len(_VECTOR_PARAM_NAMES) == 17
    dim = int(float(get_parameter("Calibration behavioral parameter dimension").value))
    assert dim == 17
    assert get_parameter("Calibration identifiable parameter indices").value == "7,15,16"
    default = decision_to_request_rate_default()
    center = np.full(17, 0.5, dtype=np.float64)
    assert calibration_decision_to_request_rate(center) == pytest.approx(default)
    assert calibration_awareness_media_rate(np.full(17, 0.9)) == awareness_media_rate_default()
    assert calibration_decision_to_request_rate(np.full(16, 0.5)) == pytest.approx(default)
    t_lo, t_hi = _theta_bounds(16)
    assert default * np.exp(t_lo) == pytest.approx(_LO)
    assert default * np.exp(t_hi) == pytest.approx(_HI)
