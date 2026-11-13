"""SVB-19 (part 1): fused logits and loop-2 reuse match the pre-SVB-19 kernel.

NumPy always; CuPy (cupy.fuse) when a device is visible. Uses the same random world
as test_svb17_kernel_parity so both contact paths are covered.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from tests.engine._svb_kernel_world import _ATOL_DOLLARS, _max_rel, _run, _world, _xp_cases
except ImportError:  # pytest's default rootdir/prepend import mode without package __init__ files
    from _svb_kernel_world import _ATOL_DOLLARS, _max_rel, _run, _world, _xp_cases

_RTOL = 1e-5


@pytest.mark.parametrize("name,xp", _xp_cases())
@pytest.mark.parametrize("softness", [0.06, 0.0])
@pytest.mark.parametrize("path", ["spmm", "gather"])
def test_fused_matches_unfused(name, xp, softness, path, monkeypatch=None):
    import os

    state, params, gather, adj, slices, C, n, _ = _world(xp, softness=softness, integer_hub=softness == 0.0)
    kw = (dict(contacts=None, hub=None, adjacency=adj, seed_slices=slices) if path == "spmm"
          else dict(contacts=gather["contacts"], hub=gather["hub"]))
    os.environ["STAMPEDE_CALIB_UNFUSED_LOGITS"] = "1"
    try:
        old_amt, old = _run(xp, state, params, softness, chunk=97, **kw)
    finally:
        os.environ.pop("STAMPEDE_CALIB_UNFUSED_LOGITS", None)
    new_amt, new = _run(xp, state, params, softness, chunk=97, **kw)
    assert _max_rel(old_amt.sum(axis=1, dtype=np.float64), new_amt.sum(axis=1, dtype=np.float64)) < _RTOL
    assert np.allclose(old_amt, new_amt, rtol=_RTOL, atol=_ATOL_DOLLARS)
    assert np.allclose(old["balance"], new["balance"], rtol=_RTOL, atol=_ATOL_DOLLARS)
    assert np.array_equal(old["already"], new["already"])
