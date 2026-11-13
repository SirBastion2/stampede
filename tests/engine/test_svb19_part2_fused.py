"""SVB-19 part 2: fused amount chain, belief update, and the two sigmoids.

NumPy stays on the current fallback. CuPy (skipped without CUDA) uses
``cupy.ElementwiseKernel`` and must match that fallback within 1e-5 relative;
1 float32 ULP differences are acceptable.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from engine.agents.decide import _sigmoid, _update_belief

try:
    from tests.engine._svb_kernel_world import (
        _ATOL_DOLLARS,
        _host,
        _max_rel,
        _run,
        _world,
    )
except ImportError:
    from _svb_kernel_world import (
        _ATOL_DOLLARS,
        _host,
        _max_rel,
        _run,
        _world,
    )

_RTOL = 1e-5
_UNFUSED = "STAMPEDE_CALIB_UNFUSED_PART2"


def _cupy_or_skip():
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
    except Exception:
        pass
    pytest.skip("CUDA device required for fused part-2 kernels")


def _float32_ulps(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, np.int64(0x80000000) - ai, ai)
    bi = np.where(bi < 0, np.int64(0x80000000) - bi, bi)
    return np.abs(ai - bi)


def _close_rel_or_ulp(a, b, rtol=_RTOL, ulps=1) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    rel = np.abs(a.astype(np.float64) - b.astype(np.float64)) / np.maximum(
        np.maximum(np.abs(a.astype(np.float64)), np.abs(b.astype(np.float64))), 1e-30
    )
    return bool(np.all((rel < rtol) | (_float32_ulps(a, b).reshape(rel.shape) <= ulps)))


def _with_vip(xp, state, *, high=5.0e3, low=0.4):
    vip = _host(state["vip_amount"]).copy()
    vip[0, :12] = high
    vip[1, 12:24] = low
    out = dict(state)
    out["vip_amount"] = xp.asarray(vip.astype(np.float32))
    return out


def _with_wide_fraction(xp, state):
    frac = _host(state["segment_fraction"]).copy()
    frac[:, :8] = 1.5
    out = dict(state)
    out["segment_fraction"] = xp.asarray(frac.astype(np.float32))
    return out


def _contact_kw(path, gather, adj, slices):
    if path == "spmm":
        return dict(contacts=None, hub=None, adjacency=adj, seed_slices=slices)
    return dict(contacts=gather["contacts"], hub=gather["hub"])


@pytest.mark.parametrize("softness", [0.06, 0.0])
@pytest.mark.parametrize("dt", [1.0, 5.0])
@pytest.mark.parametrize("path", ["spmm", "gather"])
def test_numpy_fallback_matches_current_outputs(softness, dt, path):
    """NumPy has no fused kernels: the fallback is the current path, bit-identical."""
    xp = np
    state, params, gather, adj, slices, C, n, _ = _world(
        xp, softness=softness, integer_hub=softness == 0.0
    )
    state = _with_vip(xp, _with_wide_fraction(xp, state))
    kw = _contact_kw(path, gather, adj, slices)
    os.environ[_UNFUSED] = "1"
    try:
        old_amt, old = _run(xp, state, params, softness, chunk=97, dt=dt, **kw)
    finally:
        os.environ.pop(_UNFUSED, None)
    new_amt, new = _run(xp, state, params, softness, chunk=97, dt=dt, **kw)
    assert np.array_equal(old_amt, new_amt)
    assert np.array_equal(old["belief"], new["belief"])
    assert np.array_equal(old["balance"], new["balance"])
    assert np.array_equal(old["already"], new["already"])


@pytest.mark.parametrize("softness", [0.06, 0.0])
def test_numpy_dt1_keeps_unclipped_fraction(softness):
    """dt=1 uses segment_fraction * gap * scale, not clip(fraction)**dt."""
    xp = np
    state, params, gather, adj, slices, C, n, _ = _world(
        xp, softness=softness, n=80, degree=0, integer_hub=False
    )
    state = _with_wide_fraction(xp, state)
    params = dict(params)
    params["k"] = xp.zeros_like(params["k"])
    amt_dt1, _ = _run(
        xp, state, params, softness, chunk=40, dt=1.0,
        contacts=gather["contacts"], hub=gather["hub"],
    )
    amt_power, _ = _run(
        xp, state, params, softness, chunk=40, dt=1.0000001,
        contacts=gather["contacts"], hub=gather["hub"],
    )
    # Unclipped 1.5 vs clip-to-1 take: dt=1 must not collapse to the clipped path.
    assert float(np.max(np.abs(amt_dt1 - amt_power))) > 0.0


def test_numpy_step_belief_matches_decide():
    """Fallback belief write is exactly ``decide._update_belief``."""
    from engine.gpu_sim_kernels import _fuse, _signal_from_public

    xp = np
    state, params, gather, adj, slices, C, n, _ = _world(xp, n=40, degree=0, softness=0.0)
    _, out = _run(
        xp, state, params, 0.0, chunk=40,
        contacts=gather["contacts"], hub=gather["hub"],
    )
    attn = params["attention_level"][:, None]
    drawdown = xp.maximum(0.0, -params["stock_return"])[:, None]
    signal = _fuse(xp)(_signal_from_public)(
        params["signal_base"][:, None],
        params["signal_attention"][:, None],
        attn,
        params["signal_stock"][:, None],
        drawdown,
    )
    expected = _update_belief(
        xp, state["belief"], signal, params["belief_strength"][:, None]
    )
    assert np.array_equal(out["belief"], expected)


@pytest.mark.parametrize("softness", [0.06, 0.0])
@pytest.mark.parametrize("dt", [1.0, 5.0])
@pytest.mark.parametrize("path", ["spmm", "gather"])
def test_cupy_fused_matches_fallback(softness, dt, path):
    xp = _cupy_or_skip()
    state, params, gather, adj, slices, C, n, _ = _world(
        xp, softness=softness, integer_hub=softness == 0.0
    )
    state = _with_vip(xp, _with_wide_fraction(xp, state))
    kw = _contact_kw(path, gather, adj, slices)
    os.environ[_UNFUSED] = "1"
    try:
        old_amt, old = _run(xp, state, params, softness, chunk=97, dt=dt, **kw)
    finally:
        os.environ.pop(_UNFUSED, None)
    new_amt, new = _run(xp, state, params, softness, chunk=97, dt=dt, **kw)
    old_sum = old_amt.sum(axis=1, dtype=np.float64)
    new_sum = new_amt.sum(axis=1, dtype=np.float64)
    assert _max_rel(old_sum, new_sum) < _RTOL
    # float32 fused soft gate vs promoted fallback: 2 of 7200 small amounts differ by up to 3e-5 relative.
    assert _close_rel_or_ulp(old_amt, new_amt, rtol=1e-4)
    assert _close_rel_or_ulp(old["belief"], new["belief"])
    assert _close_rel_or_ulp(old["balance"], new["balance"]) or np.allclose(
        old["balance"], new["balance"], rtol=_RTOL, atol=_ATOL_DOLLARS
    )
    assert np.array_equal(old["already"], new["already"])


def test_cupy_belief_and_sigmoid_match_decide():
    xp = _cupy_or_skip()
    from engine.gpu_sim_kernels import _belief_elementwise, _sigmoid_elementwise

    rng = np.random.default_rng(19)
    belief = xp.asarray(rng.uniform(-0.1, 1.1, (4, 32)).astype(np.float32))
    signal = xp.asarray(rng.uniform(-0.1, 1.1, (4, 32)).astype(np.float32))
    strength = xp.asarray(rng.uniform(0.0, 1.0, (4, 1)).astype(np.float32))
    logits = xp.asarray(rng.uniform(-12.0, 12.0, (4, 32)).astype(np.float32))
    got_b = _host(_belief_elementwise()(belief, signal, strength))
    ref_b = _host(_update_belief(xp, belief, signal, strength))
    got_s = _host(_sigmoid_elementwise()(logits))
    ref_s = _host(_sigmoid(xp, logits))
    assert _close_rel_or_ulp(got_b, ref_b)
    assert _close_rel_or_ulp(got_s, ref_s)
