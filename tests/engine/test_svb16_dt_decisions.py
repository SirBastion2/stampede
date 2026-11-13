"""SVB-16: withdrawal take scales with dt; dt=1 matches the old function."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from engine.agents.decide import _sigmoid, _update_belief
from engine.gpu_sim import relative_error
from engine.gpu_sim_kernels import (
    _contact_degree,
    _fuse,
    _gather_neighbors,
    _gap_fraction,
    _signal_from_public,
    _slice_static,
    make_depositor_scratch,
    step_depositor_batch,
)

_RTOL = 0.02
_MINUTES = 15
_SEED = 16


def _legacy_step_depositor_batch(
    xp: Any,
    *,
    belief: Any,
    balance: Any,
    already: Any,
    insured: Any,
    base_threshold: Any,
    threshold_shift: Any,
    statement_shift: Any,
    deposit_rate: Any,
    directive: Any,
    payroll: Any,
    segment_fraction: Any,
    vip_amount: Any,
    hub: Any,
    contacts: Any,
    attention_level: Any,
    stock_return: Any,
    gap_scale: Any,
    amount_scale: Any,
    weights: Any,
    bias: Any,
    k: Any,
    belief_strength: Any,
    signal_attention: Any,
    signal_stock: Any,
    signal_base: Any,
    rate_relief: Any,
    minimum_amount: float,
    chunk: int,
    softness: float = 0.0,
    scratch: dict[str, Any] | None = None,
) -> Any:
    """Frozen pre-SVB-16 ``step_depositor_batch`` (no ``dt``)."""
    n_candidates = int(belief.shape[0])
    n_depositors = int(belief.shape[1])
    if scratch is None:
        scratch = make_depositor_scratch(xp, n_candidates, n_depositors, softness)
    amounts = scratch["amounts"]
    amounts.fill(0)
    if n_depositors == 0 or n_candidates == 0:
        return amounts
    degree = _contact_degree(contacts)
    temp = float(softness)
    worried = scratch["worried"]
    width = max(1, int(chunk))
    attn = attention_level[:, None]
    drawdown = xp.maximum(0.0, -stock_return)[:, None]
    strength = belief_strength[:, None]
    fused_signal = _fuse(xp)(_signal_from_public)
    fused_gap = _fuse(xp)(_gap_fraction)
    f32 = getattr(xp, "float32", np.float32)

    for start in range(0, n_depositors, width):
        stop = min(n_depositors, start + width)
        sl = slice(start, stop)
        columns = stop - start
        if degree == 0:
            share = xp.zeros((n_candidates, columns), dtype=f32)
        else:
            share = _gather_neighbors(already, contacts, sl, xp).astype(f32).mean(axis=2)
        signal = fused_signal(
            signal_base[:, None],
            signal_attention[:, None],
            attn,
            signal_stock[:, None],
            drawdown,
        )
        updated = _update_belief(xp, belief[:, sl], signal, strength)
        belief[:, sl] = updated
        safe = xp.maximum(balance[:, sl], 0.0)
        insured_rows = _slice_static(insured, sl)
        gap_balance = xp.maximum(0.0, safe - insured_rows) * gap_scale[:, None]
        gap_fraction = fused_gap(safe, gap_balance)
        payroll_rows = _slice_static(payroll, sl)
        if getattr(payroll_rows, "ndim", 1) == 1:
            payroll_rows = xp.broadcast_to(payroll_rows, (n_candidates, columns))
        features = xp.stack(
            [
                xp.broadcast_to(attn, (n_candidates, columns)),
                share,
                updated,
                xp.broadcast_to(stock_return[:, None], (n_candidates, columns)),
                directive[:, sl],
                payroll_rows,
                gap_fraction,
            ],
            axis=2,
        )
        logits = xp.einsum("cbf,cf->cb", features, weights)
        logits = (
            logits
            + bias[:, None]
            + statement_shift[:, sl]
            - rate_relief[:, None] * deposit_rate[:, sl]
        )
        worry = _sigmoid(xp, logits)
        threshold = _slice_static(base_threshold, sl) + threshold_shift[:, sl]
        if temp <= 0.0:
            worried[:, sl] = worry > threshold
        else:
            worried[:, sl] = _sigmoid(xp, (worry - threshold) / temp)

    for start in range(0, n_depositors, width):
        stop = min(n_depositors, start + width)
        sl = slice(start, stop)
        safe = xp.maximum(balance[:, sl], 0.0)
        gap_balance = xp.maximum(0.0, safe - _slice_static(insured, sl)) * gap_scale[:, None]
        if degree == 0:
            worried_contacts = xp.zeros((n_candidates, stop - start), dtype=f32)
        else:
            gathered = _gather_neighbors(worried, contacts, sl, xp).astype(f32)
            if getattr(hub, "ndim", 1) == 1:
                if getattr(contacts, "ndim", 1) == 2:
                    hub_rows = hub[contacts[sl]]
                else:
                    hub_rows = hub[contacts[:, sl]]
            else:
                hub_rows = _gather_neighbors(hub, contacts, sl, xp)
            worried_contacts = (gathered * hub_rows).sum(axis=2)
        vip_force = vip_amount[:, sl] >= minimum_amount
        base_amount = _slice_static(segment_fraction, sl) * gap_balance * amount_scale[:, None]
        vip_take = xp.minimum(vip_amount[:, sl], safe) * amount_scale[:, None]
        amount = xp.where(vip_force, vip_take, base_amount)
        if temp <= 0.0:
            rule = worried[:, sl] & (worried_contacts >= k[:, None])
            act = rule | vip_force
            amount = xp.where(act, amount, 0.0)
            amount = xp.where(amount >= minimum_amount, amount, 0.0)
            withdraw = amount >= minimum_amount
        else:
            p_k = _sigmoid(xp, (worried_contacts - k[:, None]) / temp)
            p_act = worried[:, sl].astype(f32) * p_k
            p_act = xp.where(vip_force, xp.ones_like(p_act), p_act)
            amount = amount * p_act
            amount = amount * _sigmoid(xp, (amount - minimum_amount) / temp)
            withdraw = amount > 0.0
        amounts[:, sl] = amount
        balance[:, sl] = xp.maximum(0.0, balance[:, sl] - amount)
        already[:, sl] = already[:, sl] | withdraw
    return amounts


_MUTABLE = ("belief", "balance", "already", "aware")


def _copy_kwargs(frozen: dict[str, Any]) -> dict[str, Any]:
    out = dict(frozen)
    for key in _MUTABLE:
        out[key] = np.array(frozen[key], copy=True)
    return out


def _mixed_inputs(softness: float) -> dict[str, Any]:
    """Frozen mixed book: some VIPs, contacts, and already-withdrew flags."""
    rng = np.random.default_rng(_SEED)
    n_candidates = 2
    n = 17
    balance = rng.uniform(2.0e5, 4.0e6, size=(n_candidates, n))
    insured = np.minimum(balance, 250_000.0)
    belief = np.full((n_candidates, n), 0.12)
    already = rng.random((n_candidates, n)) < 0.2
    contacts = rng.integers(0, n, size=(n, 3), dtype=np.int64)
    hub = np.linspace(0.4, 1.6, n)
    payroll = (rng.random(n) < 0.3).astype(np.float64)
    segment_fraction = np.full(n, 0.02)
    directive = (rng.random((n_candidates, n)) < 0.05).astype(np.float64)
    vip = np.zeros((n_candidates, n))
    vip[0, 2] = 40_000.0
    vip[1, 5] = 80_000.0
    return {
        "belief": belief,
        "balance": balance,
        "already": already,
        "insured": insured,
        "base_threshold": np.full(n, 0.35),
        "threshold_shift": np.zeros((n_candidates, n)),
        "statement_shift": np.zeros((n_candidates, n)),
        "deposit_rate": np.zeros((n_candidates, n)),
        "directive": directive,
        "payroll": payroll,
        "segment_fraction": segment_fraction,
        "vip_amount": vip,
        "hub": hub,
        "contacts": contacts,
        "attention_level": np.array([0.55, 0.70]),
        "stock_return": np.array([-0.06, -0.11]),
        "aware": np.ones((n_candidates, n), dtype=np.float32),
        "gap_scale": np.array([1.0, 0.9]),
        "amount_scale": np.array([1.0, 1.1]),
        "weights": np.array(
            [
                [1.2, 0.8, 0.5, -0.4, 0.3, 0.2, 0.6],
                [1.4, 0.7, 0.4, -0.3, 0.2, 0.3, 0.5],
            ]
        ),
        "bias": np.array([0.1, -0.05]),
        "k": np.array([1.0, 1.0]),
        "belief_strength": np.array([0.08, 0.10]),
        "signal_attention": np.array([0.4, 0.5]),
        "signal_stock": np.array([0.3, 0.25]),
        "signal_base": np.array([0.05, 0.04]),
        "rate_relief": np.array([0.1, 0.1]),
        "minimum_amount": 100.0,
        "chunk": 5,
        "softness": softness,
    }


def _stable_inputs(softness: float) -> dict[str, Any]:
    """Frozen inputs whose act probability stays put across ticks.

    Degree 0, zero belief strength, zero gap weight: worry does not drift, so
    fifteen one-minute takes compound like one 15-minute take.
    """
    n_candidates = 2
    n = 11
    balance = np.full((n_candidates, n), 1.5e6)
    insured = np.full((n_candidates, n), 250_000.0)
    return {
        "belief": np.full((n_candidates, n), 0.2),
        "balance": balance,
        "already": np.zeros((n_candidates, n), dtype=bool),
        "insured": insured,
        "base_threshold": np.full(n, 0.10),
        "threshold_shift": np.zeros((n_candidates, n)),
        "statement_shift": np.zeros((n_candidates, n)),
        "deposit_rate": np.zeros((n_candidates, n)),
        "directive": np.zeros((n_candidates, n)),
        "payroll": np.zeros(n),
        "segment_fraction": np.full(n, 0.02),
        "vip_amount": np.zeros((n_candidates, n)),
        "hub": np.ones(n),
        "contacts": np.zeros((n, 0), dtype=np.int64),
        "attention_level": np.array([1.0, 1.0]),
        "stock_return": np.array([0.0, 0.0]),
        "aware": np.ones((n_candidates, n), dtype=np.float32),
        "gap_scale": np.array([1.0, 1.0]),
        "amount_scale": np.array([1.0, 1.0]),
        "weights": np.array(
            [
                [8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "bias": np.array([2.0, 2.0]),
        "k": np.array([0.0, 0.0]),
        "belief_strength": np.array([0.0, 0.0]),
        "signal_attention": np.array([0.0, 0.0]),
        "signal_stock": np.array([0.0, 0.0]),
        "signal_base": np.array([0.0, 0.0]),
        "rate_relief": np.array([0.0, 0.0]),
        "minimum_amount": 1.0,
        "chunk": 11,
        "softness": softness,
    }


def _total_withdrawn(frozen: dict[str, Any], dts: list[float]) -> float:
    kwargs = _copy_kwargs(frozen)
    total = 0.0
    for dt in dts:
        amounts = step_depositor_batch(np, **kwargs, dt=dt)
        total += float(np.asarray(amounts, dtype=np.float64).sum())
    return total


@pytest.mark.parametrize("softness", [0.0, 0.1])
def test_fifteen_one_minute_matches_one_fifteen_minute(softness: float) -> None:
    frozen = _stable_inputs(softness)
    one_minute = _total_withdrawn(frozen, [1.0] * _MINUTES)
    one_tick = _total_withdrawn(frozen, [float(_MINUTES)])
    assert one_minute > 0.0
    assert one_tick > 0.0
    assert relative_error(one_minute, one_tick) <= _RTOL


@pytest.mark.parametrize("softness", [0.0, 0.1])
def test_dt_one_matches_legacy_exactly(softness: float) -> None:
    frozen = _mixed_inputs(softness)
    new_kwargs = _copy_kwargs(frozen)
    old_kwargs = _copy_kwargs(frozen)
    old_kwargs.pop("aware", None)
    new_amounts = step_depositor_batch(np, **new_kwargs, dt=1.0)
    old_amounts = _legacy_step_depositor_batch(np, **old_kwargs)
    # SVB-19 fuses the logit sum, which reorders float32 adds: allow 1 ULP.
    np.testing.assert_allclose(
        np.asarray(new_amounts), np.asarray(old_amounts), rtol=2e-6, atol=0.0
    )
    for key in _MUTABLE:
        if key not in old_kwargs:
            continue
        np.testing.assert_allclose(
            np.asarray(new_kwargs[key]), np.asarray(old_kwargs[key]), rtol=2e-6, atol=0.0
        )
