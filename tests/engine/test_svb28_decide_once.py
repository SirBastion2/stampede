"""SVB-28: each depositor decides at most once (no geometric per-tick draining).

Kernel-level, self-contained (no engine fixtures). NumPy always; CuPy cases when a device is visible.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.gpu_sim_kernels import make_depositor_scratch, step_depositor_batch


def _xp_cases():
    cases = [("numpy", np)]
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            cases.append(("cupy", cupy))
    except Exception:
        pass
    return cases


def _host(a):
    get = getattr(a, "get", None)
    return np.asarray(get() if callable(get) and not isinstance(a, np.ndarray) else a)


def _world(xp, *, C=3, n=400, degree=6, threshold=0.7, attention=0.2, seed=0):
    """A calm book by default: worry sits well below the threshold for nearly everyone."""
    rng = np.random.default_rng(seed)
    f32 = np.float32
    balance = (rng.pareto(1.5, n) * 4e5 + 3e5).astype(f32)
    insured = np.full(n, 2.5e5, f32)
    st = dict(
        belief=np.full((C, n), 0.05, f32),
        balance=np.broadcast_to(balance, (C, n)).copy(),
        already=np.zeros((C, n), bool),
        insured=insured,
        base_threshold=np.full((C, n), threshold, f32),
        threshold_shift=np.zeros((C, n), f32),
        statement_shift=np.zeros((C, n), f32),
        deposit_rate=np.zeros((C, n), f32),
        directive=np.zeros((C, n), f32),
        payroll=np.zeros(n, f32),
        segment_fraction=np.full(n, 0.6, f32),
        vip_amount=np.zeros((C, n), f32),
        hub=np.ones(n, f32),
        contacts=rng.integers(0, n, (n, degree)).astype(np.int64),
    )
    params = dict(
        attention_level=np.full(C, attention, f32),
        stock_return=np.zeros(C, f32),
        gap_scale=np.ones(C, f32),
        amount_scale=np.ones(C, f32),
        weights=np.tile(np.array([1.0, 2.0, 0.5, -1.0, 0.0, 0.0, 0.5], f32), (C, 1)),
        bias=np.full(C, -2.0, f32),
        k=np.full(C, 1.0, f32),
        belief_strength=np.full(C, 0.1, f32),
        signal_attention=np.full(C, 0.3, f32),
        signal_stock=np.full(C, 0.5, f32),
        signal_base=np.full(C, 0.05, f32),
        rate_relief=np.zeros(C, f32),
        mobile_share=np.ones(C, f32),
    )
    init_u = np.maximum(0.0, balance - insured).astype(f32)
    to = lambda d: {k: xp.asarray(v) for k, v in d.items()}
    return to(st), to(params), xp.asarray(init_u), float(init_u.sum())


def _run(
    xp, ticks, *, decide_once, softness, aware0=0.0, decision_to_request_rate=None, **world_kw
):
    st, params, init_u, total_u = _world(xp, **world_kw)
    C, n = st["belief"].shape
    scratch = make_depositor_scratch(xp, C, n, softness)
    f32 = getattr(xp, "float32", np.float32)
    aware = xp.full((C, n), float(aware0), dtype=f32)
    extra = {}
    if decision_to_request_rate is not None:
        extra["decision_to_request_rate"] = xp.full(C, float(decision_to_request_rate), dtype=f32)
    cum = np.zeros(C)
    path = []
    for _ in range(ticks):
        amounts = step_depositor_batch(
            xp, **st, **params, minimum_amount=1.0, chunk=10_000, softness=softness,
            scratch=scratch, initial_uninsured=init_u, decide_once=decide_once,
            aware=aware, **extra,
        )
        cum += _host(amounts).sum(axis=1)
        path.append(cum.copy() / total_u)
    return np.asarray(path), st


@pytest.mark.parametrize("name,xp", _xp_cases())
def test_soft_gate_no_longer_drains_a_calm_book(name, xp):
    legacy, _ = _run(xp, 390, decide_once=False, softness=0.06, aware0=1.0)
    once, _ = _run(xp, 390, decide_once=True, softness=0.06, aware0=1.0)
    # The leak: in soft mode every depositor gets amt > 0 on tick 1, so ``already`` is set for
    # everyone, the contagion share jumps to 1, and the per-tick take compounds to the cap.
    assert legacy[-1].max() > 0.5
    assert once[-1].max() < 0.05             # decide once: only the tail mass that ever crosses
    assert legacy[-1].min() > 10 * once[-1].max()


@pytest.mark.parametrize("name,xp", _xp_cases())
def test_once_requests_stop_when_conditions_stop_changing(name, xp):
    once, _ = _run(
        xp, 120, decide_once=True, softness=0.06, threshold=0.5, attention=0.8,
        decision_to_request_rate=1e9,
    )
    # Instant-ratchet limit (h→∞): inputs are constant after the first ticks, so
    # requests plateau instead of growing every tick.
    assert np.allclose(once[60], once[119], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("name,xp", _xp_cases())
def test_hard_gate_each_depositor_requests_once(name, xp):
    once, st = _run(xp, 40, decide_once=True, softness=0.0, threshold=0.0, attention=0.9)
    st0, _, init_u, total_u = _world(xp, threshold=0.0, attention=0.9)
    # everyone acts on tick 1 and never again: total = segment fraction of each uninsured gap
    expected = float((0.6 * _host(init_u)).sum()) / total_u
    assert np.allclose(once[0], expected, rtol=1e-5)
    assert np.allclose(once[-1], once[0], rtol=1e-6)
    legacy, _ = _run(xp, 40, decide_once=False, softness=0.0, threshold=0.0, attention=0.9)
    assert np.allclose(legacy[0], once[0], rtol=1e-6)       # the first decision is the same
    assert legacy[-1].min() > 0.99                          # legacy drains the whole gap


@pytest.mark.parametrize("name,xp", _xp_cases())
def test_once_respects_the_mobile_share_cap(name, xp):
    st, params, init_u, total_u = _world(xp, threshold=0.0, attention=0.9)
    params["mobile_share"] = xp.full(3, 0.4, dtype=np.float32)
    params["mobile_share"][0] = 1.0
    C, n = st["belief"].shape
    scratch = make_depositor_scratch(xp, C, n, 0.0)
    amounts = step_depositor_batch(
        xp, **st, **params, minimum_amount=1.0, chunk=10_000, softness=0.0,
        scratch=scratch, initial_uninsured=init_u, decide_once=True,
    )
    frac = _host(amounts).sum(axis=1) / total_u
    assert frac[0] == pytest.approx(0.6, rel=1e-5)
    assert frac[1] == pytest.approx(0.4, rel=1e-5)          # min(0.6, 0.4) of each gap


def test_fused_once_kernel_matches_fallback_on_cupy():
    cases = _xp_cases()
    if len(cases) < 2:
        pytest.skip("needs a CUDA device")
    import os

    cupy = cases[1][1]
    for softness in (0.06, 0.0):
        fused, _ = _run(cupy, 60, decide_once=True, softness=softness, threshold=0.5, attention=0.7)
        os.environ["STAMPEDE_CALIB_UNFUSED_PART2"] = "1"
        try:
            fallback, _ = _run(cupy, 60, decide_once=True, softness=softness, threshold=0.5, attention=0.7)
        finally:
            os.environ.pop("STAMPEDE_CALIB_UNFUSED_PART2", None)
        assert np.allclose(fused, fallback, rtol=1e-4, atol=1e-6)
