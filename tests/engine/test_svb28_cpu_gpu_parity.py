"""SVB-28: once-mode parity between the batch kernel and the CPU M06 reference (one tick)."""
from __future__ import annotations

import numpy as np
import pytest

from engine.agents.decide import (
    DecisionInputs,
    build_contact_index,
    evaluate_decisions,
    hub_weights,
    load_decision_params,
    withdrawal_decision_once,
)
from engine.gpu_sim_kernels import step_depositor_batch


def _case(softness: float):
    rng = np.random.default_rng(0)
    n = 23
    params = load_decision_params()
    balance = rng.uniform(1.0e5, 5.0e6, size=n)
    insured = np.minimum(balance, 250_000.0)
    already = rng.random(n) < 0.25
    decided = np.where(already, 1.0, rng.uniform(0.0, 0.3, n)) if softness > 0 else already.astype(float)
    contacts = build_contact_index(n, params.contact_degree, seed=3)
    hub = hub_weights(balance, params)
    payroll = (rng.random(n) < 0.3).astype(np.float64)
    segment_fraction = np.full(n, params.default_fraction)
    directive = (rng.random(n) < 0.05).astype(np.float64)
    base_threshold = np.full(n, 0.42)
    init_u = np.maximum(0.0, balance - insured) * 1.2      # as if some had been withdrawn earlier
    return dict(rng=rng, n=n, params=params, balance=balance, insured=insured, already=already,
                decided=decided, contacts=contacts, hub=hub, payroll=payroll,
                segment_fraction=segment_fraction, directive=directive, base_threshold=base_threshold,
                init_u=init_u)


@pytest.mark.parametrize("softness", [0.0, 0.06])
def test_once_kernel_matches_m06_reference(softness):
    c = _case(softness)
    p, n = c["params"], c["n"]
    attention, stock, amount_scale = 0.55, -0.08, 1.15
    belief = np.full((1, n), p.baseline_belief)
    bal = c["balance"].reshape(1, -1).copy()
    alr = c["already"].reshape(1, -1).copy()
    dec = c["decided"].astype(np.float32).reshape(1, -1).copy()
    amounts = step_depositor_batch(
        np, belief=belief, balance=bal, already=alr, insured=c["insured"],
        base_threshold=c["base_threshold"], threshold_shift=np.zeros((1, n)),
        statement_shift=np.zeros((1, n)), deposit_rate=np.zeros((1, n)),
        directive=c["directive"].reshape(1, -1), payroll=c["payroll"],
        segment_fraction=c["segment_fraction"], vip_amount=np.zeros((1, n)), hub=c["hub"],
        contacts=c["contacts"], attention_level=np.array([attention]), stock_return=np.array([stock]),
        gap_scale=np.array([1.0]), amount_scale=np.array([amount_scale]),
        weights=np.asarray(p.weights, dtype=np.float64).reshape(1, -1), bias=np.array([p.bias]),
        k=np.array([p.k]), belief_strength=np.array([p.belief_strength]),
        signal_attention=np.array([p.signal_attention]), signal_stock=np.array([p.signal_stock]),
        signal_base=np.array([p.signal_base]), rate_relief=np.array([p.rate_relief_weight]),
        minimum_amount=p.minimum_amount, chunk=7, softness=softness,
        mobile_share=np.array([0.9]), initial_uninsured=c["init_u"].reshape(1, -1),
        decided=dec, decide_once=True,
    )
    result = evaluate_decisions(
        DecisionInputs(
            attention=np.full(n, attention), belief=np.full(n, p.baseline_belief),
            threshold=c["base_threshold"], balance=c["balance"], insured=c["insured"],
            stock_return=stock, directive=c["directive"], payroll=c["payroll"],
            contacts=c["contacts"], hub_weight=c["hub"], already_withdrew=c["already"],
            statement_shift=np.zeros(n), deposit_rate=np.zeros(n),
            segment_fraction=c["segment_fraction"], vip_amount=np.zeros(n), decided=c["decided"],
        ),
        p, gap_scale=1.0, amount_scale=amount_scale, backend="cpu", softness=softness,
        mobile_share=0.9, initial_uninsured=c["init_u"], decide_once=True,
    )
    # float32 kernel vs float64 reference
    assert np.allclose(amounts[0], result.amount, rtol=1e-5, atol=1e-2)
    assert np.array_equal(alr[0], np.logical_or(c["already"], result.withdraw))
    assert result.decided is not None
    assert np.allclose(dec[0], result.decided, rtol=1e-5, atol=1e-6)
    if softness == 0.0:
        # who had already withdrawn requests nothing more
        assert np.all(result.amount[c["already"]] == 0.0)


def test_registry_default_is_once():
    assert withdrawal_decision_once() is True
