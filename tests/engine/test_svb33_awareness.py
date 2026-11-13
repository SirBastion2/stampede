"""SVB-33: per-depositor awareness; once-mode requests respond to contact, news, bias."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from engine.agents.decide import (
    DecisionInputs,
    DecisionModule,
    _evaluate,
    awareness_media_rate_default,
    awareness_news_rate_default,
    awareness_w1_prior_default,
    awareness_wom_rate_default,
    build_contact_index,
    evaluate_decisions,
    hub_weights,
    load_decision_params,
)
from engine.gpu_sim_kernels import make_depositor_scratch, step_depositor_batch

_RTOL = 1e-5
_ATOL = 1e-5
_N = 64
_UNFUSED_LOGITS = "STAMPEDE_CALIB_UNFUSED_LOGITS"
_UNFUSED_PART2 = "STAMPEDE_CALIB_UNFUSED_PART2"


def _host(a):
    get = getattr(a, "get", None)
    return np.asarray(get() if callable(get) and not isinstance(a, np.ndarray) else a)


def _xp_cases():
    cases = [("numpy", np)]
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            cases.append(("cupy", cupy))
    except Exception:
        pass
    return cases


def _fixture(*, seed: int = 33, n: int = _N, attention: float = 0.7, news: float = 0.2):
    rng = np.random.default_rng(seed)
    params = load_decision_params()
    balance = rng.uniform(2.0e5, 1.5e6, size=n)
    insured = np.minimum(balance, 250_000.0)
    contacts = build_contact_index(n, params.contact_degree, seed)
    hub = hub_weights(balance, params)
    payroll = (rng.random(n) < 0.2).astype(np.float64)
    segment_fraction = np.full(n, params.default_fraction)
    already = np.zeros(n, dtype=np.bool_)
    decided = np.zeros(n, dtype=np.float64)
    aware = np.zeros(n, dtype=np.float64)
    init_u = np.maximum(0.0, balance - insured)
    belief = np.full(n, params.baseline_belief)
    threshold = np.full(n, 0.45)
    return dict(
        rng=rng,
        n=n,
        params=params,
        balance=balance,
        insured=insured,
        contacts=contacts,
        hub=hub,
        payroll=payroll,
        segment_fraction=segment_fraction,
        already=already,
        decided=decided,
        aware=aware,
        init_u=init_u,
        belief=belief,
        threshold=threshold,
        attention=attention,
        news=news,
        stock=-0.05,
        mobile_share=0.879,
        softness=0.06,
    )


def _kernel_kwargs(fix, *, C=1):
    n = fix["n"]
    p = fix["params"]
    f32 = np.float32
    return dict(
        belief=np.broadcast_to(fix["belief"].astype(f32), (C, n)).copy(),
        balance=np.broadcast_to(fix["balance"].astype(f32), (C, n)).copy(),
        already=np.broadcast_to(fix["already"], (C, n)).copy(),
        insured=fix["insured"].astype(f32),
        base_threshold=np.broadcast_to(fix["threshold"].astype(f32), (C, n)).copy(),
        threshold_shift=np.zeros((C, n), f32),
        statement_shift=np.zeros((C, n), f32),
        deposit_rate=np.zeros((C, n), f32),
        directive=np.zeros((C, n), f32),
        payroll=fix["payroll"].astype(f32),
        segment_fraction=fix["segment_fraction"].astype(f32),
        vip_amount=np.zeros((C, n), f32),
        hub=fix["hub"].astype(f32),
        contacts=fix["contacts"],
        attention_level=np.full(C, fix["attention"], f32),
        stock_return=np.full(C, fix["stock"], f32),
        gap_scale=np.ones(C, f32),
        amount_scale=np.ones(C, f32),
        weights=np.tile(np.asarray(p.weights, dtype=f32), (C, 1)),
        bias=np.full(C, p.bias, f32),
        k=np.full(C, p.k, f32),
        belief_strength=np.full(C, p.belief_strength, f32),
        signal_attention=np.full(C, p.signal_attention, f32),
        signal_stock=np.full(C, p.signal_stock, f32),
        signal_base=np.full(C, p.signal_base, f32),
        rate_relief=np.full(C, p.rate_relief_weight, f32),
        minimum_amount=p.minimum_amount,
        mobile_share=np.full(C, fix["mobile_share"], f32),
        initial_uninsured=np.broadcast_to(fix["init_u"].astype(f32), (C, n)).copy(),
        decided=np.broadcast_to(fix["decided"].astype(f32), (C, n)).copy(),
        aware=np.broadcast_to(fix["aware"].astype(f32), (C, n)).copy(),
        news=np.full(C, fix["news"], f32),
        awareness_media_rate=np.full(C, awareness_media_rate_default(), f32),
        awareness_wom_rate=np.full(C, awareness_wom_rate_default(), f32),
        awareness_news_rate=np.full(C, awareness_news_rate_default(), f32),
        awareness_w1_prior=awareness_w1_prior_default(),
        decide_once=True,
    )


def _run_kernel(xp, fix, ticks=8, *, env=None, **overrides):
    previous = {}
    if env:
        for key, value in env.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
    try:
        kwargs = _kernel_kwargs(fix)
        kwargs.update(overrides)
        kwargs = {key: xp.asarray(value) if hasattr(value, "shape") else value for key, value in kwargs.items()}
        C, n = kwargs["belief"].shape
        scratch = make_depositor_scratch(xp, C, n, fix["softness"])
        amounts = None
        for _ in range(ticks):
            amounts = step_depositor_batch(
                xp,
                **kwargs,
                chunk=32,
                softness=fix["softness"],
                scratch=scratch,
                dt=1.0,
            )
        return _host(amounts), _host(kwargs["decided"]), _host(kwargs["aware"]), _host(kwargs["belief"])
    finally:
        if env:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _run_evaluate(fix, ticks=8):
    p = fix["params"]
    n = fix["n"]
    belief = fix["belief"].copy()
    balance = fix["balance"].copy()
    already = fix["already"].copy()
    decided = fix["decided"].copy()
    aware = fix["aware"].copy()
    amount = None
    result = None
    for _ in range(ticks):
        result = _evaluate(
            np,
            DecisionInputs(
                attention=np.full(n, fix["attention"]),
                belief=belief,
                threshold=fix["threshold"],
                balance=balance,
                insured=fix["insured"],
                stock_return=fix["stock"],
                directive=np.zeros(n),
                payroll=fix["payroll"],
                contacts=fix["contacts"],
                hub_weight=fix["hub"],
                already_withdrew=already,
                statement_shift=np.zeros(n),
                deposit_rate=np.zeros(n),
                segment_fraction=fix["segment_fraction"],
                vip_amount=np.zeros(n),
                decided=decided,
                aware=aware,
                news=fix["news"],
            ),
            p,
            gap_scale=1.0,
            amount_scale=1.0,
            backend="cpu",
            softness=fix["softness"],
            mobile_share=fix["mobile_share"],
            initial_uninsured=fix["init_u"],
            decide_once=True,
        )
        amount = result.amount
        already = already | result.withdraw
        belief = result.belief
        balance = np.maximum(0.0, balance - result.amount)
        decided = result.decided
        aware = result.aware
    return amount, decided, aware, belief


def _run_module(fix, ticks=8):
    p = fix["params"]
    n = fix["n"]
    module = DecisionModule()
    module.params = p
    module.softness = fix["softness"]
    module.decide_once = True
    module.mobile_share = fix["mobile_share"]
    module.attention_level = fix["attention"]
    module.stock_return = fix["stock"]
    module.news_level = fix["news"]
    module.belief = fix["belief"].copy()
    module.balance = fix["balance"].copy()
    module.insured = fix["insured"].copy()
    module.base_threshold = fix["threshold"].copy()
    module.threshold_shift = np.zeros(n)
    module.statement_shift = np.zeros(n)
    module.deposit_rate = np.zeros(n)
    module.directive = np.zeros(n)
    module.payroll = fix["payroll"].copy()
    module.segment_fraction = fix["segment_fraction"].copy()
    module.vip_amount = np.zeros(n)
    module.already_withdrew = fix["already"].copy()
    module.decided = fix["decided"].copy()
    module.aware = fix["aware"].copy()
    module.contacts = fix["contacts"].copy()
    module.hub_weight = fix["hub"].copy()
    module.initial_uninsured = fix["init_u"].copy()
    module.ids = [f"dep-{i}" for i in range(n)]
    module.segments = ["corp"] * n
    module._loaded = True
    result = None
    for _ in range(ticks):
        result = module._decide()
    return result.amount, module.decided, module.aware, module.belief


def test_aware_starts_at_zero_and_decided_tracks_worry():
    fix = _fixture()
    assert np.all(fix["aware"] == 0.0)
    amounts, decided, aware, _belief = _run_evaluate(fix, ticks=1)
    assert np.all(aware >= 0.0) and np.all(aware <= 1.0)
    assert np.all(aware > 0.0)
    result = evaluate_decisions(
        DecisionInputs(
            attention=np.full(fix["n"], fix["attention"]),
            belief=fix["belief"].copy(),
            threshold=fix["threshold"],
            balance=fix["balance"].copy(),
            insured=fix["insured"],
            stock_return=fix["stock"],
            directive=np.zeros(fix["n"]),
            payroll=fix["payroll"],
            contacts=fix["contacts"],
            hub_weight=fix["hub"],
            already_withdrew=fix["already"].copy(),
            statement_shift=np.zeros(fix["n"]),
            deposit_rate=np.zeros(fix["n"]),
            segment_fraction=fix["segment_fraction"],
            vip_amount=np.zeros(fix["n"]),
            decided=fix["decided"].copy(),
            aware=fix["aware"].copy(),
            news=fix["news"],
        ),
        fix["params"],
        softness=fix["softness"],
        mobile_share=fix["mobile_share"],
        initial_uninsured=fix["init_u"],
        decide_once=True,
        backend="cpu",
    )
    assert result.decided is not None
    assert np.all(result.decided <= result.worry + 1e-6)
    del amounts


def test_parity_four_paths():
    fix = _fixture()
    ticks = 6
    fused_amt, fused_dec, fused_aware, fused_bel = _run_kernel(np, fix, ticks=ticks)
    unfused_amt, unfused_dec, unfused_aware, unfused_bel = _run_kernel(
        np,
        fix,
        ticks=ticks,
        env={_UNFUSED_LOGITS: "1", _UNFUSED_PART2: "1"},
    )
    ev_amt, ev_dec, ev_aware, ev_bel = _run_evaluate(fix, ticks=ticks)
    mod_amt, mod_dec, mod_aware, mod_bel = _run_module(fix, ticks=ticks)

    def _close(a, b, atol=_ATOL):
        assert np.allclose(a, b, rtol=_RTOL, atol=atol)

    _close(fused_amt, unfused_amt, atol=1e-2)
    _close(fused_dec, unfused_dec)
    _close(fused_aware, unfused_aware)
    _close(fused_bel, unfused_bel)
    _close(fused_amt[0], ev_amt, atol=1e-2)
    _close(fused_dec[0], ev_dec)
    _close(fused_aware[0], ev_aware)
    _close(fused_bel[0], ev_bel)
    _close(ev_amt, mod_amt)
    _close(ev_dec, mod_dec)
    _close(ev_aware, mod_aware)
    _close(ev_bel, mod_bel)


@pytest.mark.parametrize("name,xp", _xp_cases())
def test_contact_weight_and_bias_raise_day1_requests(name, xp):
    fix = _fixture(attention=1.0, news=0.4)
    w_prior = awareness_w1_prior_default()
    w_low = w_prior * math.exp(0.8 * (0.1 - 0.5))
    w_high = w_prior * math.exp(0.8 * (0.9 - 0.5))
    weights = np.tile(np.asarray(fix["params"].weights, dtype=np.float32), (2, 1))
    weights[0, 1] = w_low
    weights[1, 1] = w_high
    kwargs = _kernel_kwargs(fix, C=2)
    kwargs["weights"] = weights
    kwargs["awareness_media_rate"] = np.full(2, 0.01, np.float32)
    kwargs = {key: xp.asarray(value) if hasattr(value, "shape") else value for key, value in kwargs.items()}
    C, n = kwargs["belief"].shape
    scratch = make_depositor_scratch(xp, C, n, fix["softness"])
    cum = np.zeros(2)
    for _ in range(120):
        amounts = step_depositor_batch(
            xp, **kwargs, chunk=32, softness=fix["softness"], scratch=scratch, dt=1.0
        )
        cum += _host(amounts).sum(axis=1)
    assert cum[1] > cum[0]

    bias_low = fix["params"].bias + 4.0 * (0.2 - 0.5)
    bias_high = fix["params"].bias + 4.0 * (0.8 - 0.5)
    kwargs = _kernel_kwargs(fix, C=2)
    kwargs["bias"] = np.array([bias_low, bias_high], dtype=np.float32)
    kwargs["awareness_media_rate"] = np.full(2, 0.01, np.float32)
    kwargs = {key: xp.asarray(value) if hasattr(value, "shape") else value for key, value in kwargs.items()}
    scratch = make_depositor_scratch(xp, C, n, fix["softness"])
    cum = np.zeros(2)
    for _ in range(120):
        amounts = step_depositor_batch(
            xp, **kwargs, chunk=32, softness=fix["softness"], scratch=scratch, dt=1.0
        )
        cum += _host(amounts).sum(axis=1)
    assert cum[1] > cum[0]


@pytest.mark.parametrize("name,xp", _xp_cases())
def test_soft_once_ceiling_is_mobile_share_times_uninsured(name, xp):
    fix = _fixture(attention=1.0, news=0.8)
    mobile = 0.5
    kwargs = _kernel_kwargs(fix, C=1)
    kwargs["mobile_share"] = np.array([mobile], dtype=np.float32)
    kwargs["awareness_media_rate"] = np.array([0.01], dtype=np.float32)
    kwargs["bias"] = np.array([2.0], dtype=np.float32)
    kwargs["base_threshold"] = np.full_like(kwargs["base_threshold"], -1.0)
    kwargs = {key: xp.asarray(value) if hasattr(value, "shape") else value for key, value in kwargs.items()}
    C, n = kwargs["belief"].shape
    scratch = make_depositor_scratch(xp, C, n, fix["softness"])
    total = 0.0
    for _ in range(80):
        amounts = step_depositor_batch(
            xp, **kwargs, chunk=32, softness=fix["softness"], scratch=scratch, dt=1.0
        )
        total += float(_host(amounts).sum())
    uninsured = float(fix["init_u"].sum())
    uninsured_frac = uninsured / float(fix["balance"].sum())
    next_open = total / float(fix["balance"].sum())
    assert next_open <= mobile * uninsured_frac + 1e-6
    assert total <= mobile * uninsured + 1e-6 * uninsured


@pytest.mark.parametrize("name,xp", _xp_cases())
def test_hard_gate_bit_identical_to_segment_fraction(name, xp):
    """Hard-gate amounts keep segment_fraction; bit-identical to the d627c35 formula."""
    fix = _fixture(attention=0.9, news=0.0)
    kwargs = _kernel_kwargs(fix, C=1)
    kwargs["base_threshold"] = np.zeros_like(kwargs["base_threshold"])
    kwargs["k"] = np.zeros_like(kwargs["k"])
    kwargs["mobile_share"] = np.ones(1, dtype=np.float32)
    kwargs = {key: xp.asarray(value) if hasattr(value, "shape") else value for key, value in kwargs.items()}
    C, n = kwargs["belief"].shape
    scratch = make_depositor_scratch(xp, C, n, 0.0)
    amounts = step_depositor_batch(
        xp, **kwargs, chunk=32, softness=0.0, scratch=scratch, dt=1.0
    )
    host_amt = _host(amounts)[0].astype(np.float32)
    safe = np.maximum(fix["balance"].astype(np.float32), 0.0)
    gap = np.maximum(0.0, safe - fix["insured"].astype(np.float32))
    expected = (fix["segment_fraction"].astype(np.float32) * gap).astype(np.float32)
    min_amt = np.float32(fix["params"].minimum_amount)
    expected = np.where(expected >= min_amt, expected, np.float32(0.0))
    assert np.array_equal(host_amt, expected)


@pytest.mark.parametrize("name,xp", _xp_cases())
def test_aware0_default_is_zero_and_scratch_resets_when_aware_is_none(name, xp):
    assert DecisionModule.aware0 == 0.0
    assert DecisionModule().aware0 == 0.0
    fix = _fixture(attention=1.0, news=0.0)
    kwargs = _kernel_kwargs(fix, C=1)
    kwargs.pop("aware")
    kwargs["awareness_media_rate"] = np.full(1, 0.05, np.float32)
    kwargs = {
        key: xp.asarray(value) if hasattr(value, "shape") else value
        for key, value in kwargs.items()
    }
    C, n = kwargs["belief"].shape
    scratch = make_depositor_scratch(xp, C, n, fix["softness"])
    step_depositor_batch(
        xp, **kwargs, chunk=32, softness=fix["softness"], scratch=scratch, dt=1.0
    )
    first = _host(scratch["aware"]).copy()
    assert np.all(first > 0.0)
    step_depositor_batch(
        xp, **kwargs, chunk=32, softness=fix["softness"], scratch=scratch, dt=1.0
    )
    second = _host(scratch["aware"])
    np.testing.assert_allclose(second, first, rtol=_RTOL, atol=_ATOL)
