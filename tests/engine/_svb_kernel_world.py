"""Shared random depositor world for the SVB-17 and SVB-19 kernel parity tests (not a test module)."""

from __future__ import annotations

import numpy as np
from engine.gpu_sim_kernels import (
    build_seed_adjacency,
    make_depositor_scratch,
    seed_slices_from_counts,
    step_depositor_batch,
)

_RTOL = 1e-5
_ATOL_DOLLARS = 0.01


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


def _world(xp, *, n_seeds=3, lanes_per_seed=4, n=600, degree=8, softness=0.06, seed=0,
           duplicates=False, negative=False, integer_hub=False):
    rng = np.random.default_rng(seed)
    contacts = [rng.integers(0, n, (n, degree)).astype(np.int64) for _ in range(n_seeds)]
    if duplicates:
        for c in contacts:
            c[:, 1] = c[:, 0]                       # every depositor lists its first contact twice
    if negative:
        for c in contacts:
            c[::7, -1] = -1                         # padding-like entries; NumPy indexing wraps them
    hubs = [(np.ones(n) if integer_hub else rng.uniform(0.2, 2.0, n)).astype(np.float32) for _ in range(n_seeds)]
    counts = [lanes_per_seed] * n_seeds
    C = sum(counts)
    f32 = np.float32
    def lanes(a):        # per-lane copy of a per-seed column
        return np.concatenate([np.broadcast_to(x, (k,) + x.shape) for x, k in zip(a, counts)])
    balance0 = [rng.pareto(1.5, n).astype(f32) * 1e5 + 1e3 for _ in range(n_seeds)]
    state = dict(
        belief=lanes([rng.uniform(0.1, 0.6, n).astype(f32) for _ in range(n_seeds)]),
        balance=lanes(balance0),
        already=lanes([rng.random(n) < 0.15 for _ in range(n_seeds)]),
        insured=lanes([np.full(n, 2.5e5, f32) for _ in range(n_seeds)]),
        base_threshold=rng.uniform(0.3, 0.7, (C, n)).astype(f32),
        threshold_shift=np.zeros((C, n), f32),
        statement_shift=np.zeros((C, n), f32),
        deposit_rate=np.zeros((C, n), f32),
        directive=np.zeros((C, n), f32),
        payroll=lanes([(rng.random(n) < 0.3).astype(f32) for _ in range(n_seeds)]),
        segment_fraction=lanes([rng.uniform(0.2, 1.0, n).astype(f32) for _ in range(n_seeds)]),
        vip_amount=np.zeros((C, n), f32),
    )
    params = dict(
        attention_level=rng.uniform(0.2, 0.9, C).astype(f32),
        stock_return=rng.uniform(-0.3, 0.0, C).astype(f32),
        gap_scale=np.ones(C, f32),
        amount_scale=np.ones(C, f32),
        weights=rng.uniform(-1.0, 3.0, (C, 7)).astype(f32),
        bias=rng.uniform(-2.0, 0.0, C).astype(f32),
        k=rng.uniform(0.5, 2.5, C).astype(f32) if not integer_hub else np.full(C, 1.5, f32),
        belief_strength=np.full(C, 0.2, f32),
        signal_attention=np.full(C, 0.5, f32),
        signal_stock=np.full(C, 0.5, f32),
        signal_base=np.full(C, 0.1, f32),
        rate_relief=np.zeros(C, f32),
    )
    gather = dict(contacts=lanes(contacts), hub=lanes(hubs))          # the old expanded layout
    adjacency = [build_seed_adjacency(xp, c, h) for c, h in zip(contacts, hubs)]
    to = lambda d: {k: xp.asarray(v) for k, v in d.items()}
    return to(state), to(params), to(gather), adjacency, seed_slices_from_counts(counts), C, n, softness


def _run(xp, state, params, softness, *, chunk, **contact_kw):
    st = {k: v.copy() for k, v in state.items()}
    C, n = st["belief"].shape
    amounts = step_depositor_batch(
        xp, **st, **params, minimum_amount=1.0, chunk=chunk, softness=softness,
        scratch=make_depositor_scratch(xp, C, n, softness), **contact_kw,
    )
    return _host(amounts.copy()), {k: _host(st[k]) for k in ("belief", "balance", "already")}


def _max_rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-30)))


