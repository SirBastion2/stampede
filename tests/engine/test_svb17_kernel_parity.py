"""SVB-17 kernel-level parity: per-seed CSR SpMM contact terms vs the expanded-gather path.

Runs on NumPy/SciPy (CI and cloud agents) and on CuPy when a device is visible.
No engine fixtures needed: random depositor state, seed-major lanes, contacts
expanded per lane exactly the way _concat_lean expanded them before SVB-17.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.gpu_sim_kernels import build_seed_adjacency, seed_slices_from_counts

try:
    from tests.engine._svb_kernel_world import _ATOL_DOLLARS, _RTOL, _max_rel, _run, _world, _xp_cases
except ImportError:  # pytest's default rootdir/prepend import mode without package __init__ files
    from _svb_kernel_world import _ATOL_DOLLARS, _RTOL, _max_rel, _run, _world, _xp_cases


@pytest.mark.parametrize("name,xp", _xp_cases())
@pytest.mark.parametrize("chunk", [10_000, 97])
@pytest.mark.parametrize("variant", ["plain", "duplicates", "negative"])
def test_soft_spmm_matches_expanded_gather(name, xp, chunk, variant):
    state, params, gather, adj, slices, C, n, soft = _world(
        xp, duplicates=variant == "duplicates", negative=variant == "negative")
    old_amt, old = _run(xp, state, params, soft, chunk=chunk, contacts=gather["contacts"], hub=gather["hub"])
    new_amt, new = _run(xp, state, params, soft, chunk=chunk, contacts=None, hub=None,
                        adjacency=adj, seed_slices=slices)
    # What the fit reads: per-lane requested totals.
    assert _max_rel(old_amt.sum(axis=1, dtype=np.float64), new_amt.sum(axis=1, dtype=np.float64)) < _RTOL
    # Per depositor: relative where it means something, plus a cent of absolute slack for amounts the
    # soft gate pushes toward zero (there, relative error of float32 sums is meaningless).
    assert np.allclose(old_amt, new_amt, rtol=_RTOL, atol=_ATOL_DOLLARS)
    assert np.allclose(old["balance"], new["balance"], rtol=_RTOL, atol=_ATOL_DOLLARS)
    assert _max_rel(old["belief"], new["belief"]) < _RTOL
    assert np.array_equal(old["already"], new["already"])


@pytest.mark.parametrize("name,xp", _xp_cases())
def test_hard_spmm_matches_expanded_gather_exactly(name, xp):
    """Hard gates with unit hub weights: contact sums are small integers, so results are exact."""
    state, params, gather, adj, slices, C, n, _ = _world(xp, integer_hub=True)
    old_amt, old = _run(xp, state, params, 0.0, chunk=10_000, contacts=gather["contacts"], hub=gather["hub"])
    new_amt, new = _run(xp, state, params, 0.0, chunk=10_000, contacts=None, hub=None,
                        adjacency=adj, seed_slices=slices)
    assert np.array_equal(old_amt, new_amt)
    assert np.array_equal(old["already"], new["already"])
    assert np.array_equal(old["balance"], new["balance"])


def test_adjacency_rejects_out_of_range_contacts():
    contacts = np.array([[0, 5]], dtype=np.int64)
    with pytest.raises(ValueError):
        build_seed_adjacency(np, contacts, np.ones(1, np.float32))


def test_seed_slices_are_seed_major():
    assert seed_slices_from_counts([2, 3]) == [slice(0, 2), slice(2, 5)]
