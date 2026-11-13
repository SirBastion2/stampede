"""Fused depositor kernels for the M27 GPU sim (ADR-058).

Used only by ``engine.gpu_sim``. Default engine paths do not import this.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from engine.agents.decide import _BAYES_EPS, _sigmoid, _update_belief

_CHANNELS: tuple[str, ...] = ("app", "wire", "ach", "branch")
_XP: Any = np


def _fuse(xp: Any) -> Any:
    """Bind ``_XP`` so fused bodies call ``xp.clip`` / ``xp.maximum``, not methods."""
    global _XP
    _XP = xp
    fuse = getattr(xp, "fuse", None)
    return fuse() if callable(fuse) else (lambda fn: fn)


def _slice_static(array: Any, sl: slice) -> Any:
    if getattr(array, "ndim", 1) == 1:
        return array[sl]
    return array[:, sl]


def _contact_degree(contacts: Any) -> int:
    ndim = int(getattr(contacts, "ndim", 1))
    if ndim < 2:
        return 0
    return int(contacts.shape[-1])


def _gather_neighbors(source: Any, contacts: Any, sl: slice, xp: Any) -> Any:
    """``source[:, contacts]`` for shared ``(N, D)`` or per-lane ``(C, N, D)``."""
    if getattr(contacts, "ndim", 1) == 2:
        return source[:, contacts[sl]]
    index = contacts[:, sl]
    rows = xp.arange(int(source.shape[0]))[:, None, None]
    return source[rows, index]


def _sparse_module(xp: Any) -> Any:
    """``cupyx.scipy.sparse`` for CuPy, ``scipy.sparse`` for NumPy."""
    if getattr(xp, "__name__", "") == "cupy":
        import cupyx.scipy.sparse as sparse

        return sparse
    import scipy.sparse as sparse

    return sparse


def _host_array(array: Any, dtype: Any) -> np.ndarray:
    getter = getattr(array, "get", None)
    host = getter() if callable(getter) and not isinstance(array, np.ndarray) else array
    return np.asarray(host, dtype=dtype)


def build_seed_adjacency(xp: Any, contacts: Any, hub: Any) -> dict[str, Any] | None:
    """Per-seed CSR adjacency for the two contact terms (SVB-17).

    ``share`` has value ``1/D`` per contact, so ``share @ already`` is the mean
    over contacts. ``worry`` has value ``hub[contact]``, so ``worry @ worried``
    is the hub-weighted sum. Duplicate contacts are kept (summed), and negative
    indices wrap like NumPy indexing, so results match the gather path.
    Built once per seed; static across ticks and generations.
    """
    idx = _host_array(contacts, np.int64)
    if idx.ndim != 2 or idx.shape[0] == 0 or idx.shape[1] == 0:
        return None
    n_rows, degree = int(idx.shape[0]), int(idx.shape[1])
    idx = np.where(idx < 0, idx + n_rows, idx)
    if int(idx.min()) < 0 or int(idx.max()) >= n_rows:
        raise ValueError("contact index out of range for the seed's population")
    hub_host = _host_array(hub, np.float32).reshape(-1)
    if hub_host.shape[0] != n_rows:
        raise ValueError("hub weights must be one per depositor for the seed")
    indptr = np.arange(0, n_rows * degree + 1, degree, dtype=np.int32)
    indices = idx.reshape(-1).astype(np.int32)
    share_vals = np.full(n_rows * degree, 1.0 / degree, dtype=np.float32)
    worry_vals = hub_host[idx].reshape(-1).astype(np.float32)
    sparse = _sparse_module(xp)

    def _csr(values: np.ndarray) -> Any:
        # Fresh copies per matrix: sum_duplicates() sorts indices in place, and on
        # NumPy xp.asarray would share one indices buffer between the two matrices.
        matrix = sparse.csr_matrix(
            (xp.array(values, copy=True), xp.array(indices, copy=True), xp.array(indptr, copy=True)),
            shape=(n_rows, n_rows),
        )
        matrix.sum_duplicates()
        return matrix

    return {"share": _csr(share_vals), "worry": _csr(worry_vals), "degree": degree, "n": n_rows}


def seed_slices_from_counts(lane_counts: Any) -> list[slice]:
    """Lane slices per seed for seed-major lanes (seed0-vec0, seed0-vec1, ...)."""
    slices: list[slice] = []
    start = 0
    for count in lane_counts:
        stop = start + int(count)
        slices.append(slice(start, stop))
        start = stop
    return slices


def contact_terms(xp: Any, matrices: Any, seed_slices: Any, source: Any) -> Any:
    """``out[lanes of seed s] = (A_s @ source[lanes].T).T``: one SpMM per seed, all its lanes as columns."""
    f32 = getattr(xp, "float32", np.float32)
    out = xp.empty(source.shape, dtype=f32)
    for matrix, sl in zip(matrices, seed_slices):
        block = xp.ascontiguousarray(source[sl].T)
        out[sl] = (matrix @ block).T
    return out


def channel_totals_device(xp: Any, amounts: Any, channel_index: Any, out: Any) -> Any:
    """Per-candidate channel sums in float64. Stays on the array device (no ``.get()``)."""
    f64 = getattr(xp, "float64", np.float64)
    money = xp.asarray(amounts, dtype=f64)
    for channel in range(len(_CHANNELS)):
        mask = channel_index == channel
        if getattr(channel_index, "ndim", 1) == 1:
            out[:, channel] = money[:, mask].sum(axis=1)
        else:
            out[:, channel] = (money * mask).sum(axis=1)
    return out


def make_depositor_scratch(xp: Any, n_candidates: int, n_depositors: int, softness: float) -> dict[str, Any]:
    """Resident buffers reused every tick. Amounts/worry are float32; channel totals stay float64."""
    f32 = getattr(xp, "float32", np.float32)
    f64 = getattr(xp, "float64", np.float64)
    return {
        "amounts": xp.zeros((n_candidates, n_depositors), dtype=f32),
        "worried": xp.zeros(
            (n_candidates, n_depositors),
            dtype=f32 if float(softness) > 0.0 else bool,
        ),
        "channel_totals": xp.zeros((n_candidates, len(_CHANNELS)), dtype=f64),
        "gap_keep": xp.zeros((n_candidates, n_depositors), dtype=f32),
        "safe_keep": xp.zeros((n_candidates, n_depositors), dtype=f32),
    }


def step_depositor_batch(
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
    dt: float = 1.0,
    adjacency: Any = None,
    seed_slices: Any = None,
    mobile_share: Any = None,
    initial_uninsured: Any = None,
    decided: Any = None,
    decide_once: bool = False,
    news: Any = None,
    aware: Any = None,
    aware0: float = 0.0,
    awareness_media_rate: Any = None,
    awareness_wom_rate: Any = None,
    awareness_news_rate: Any = None,
    awareness_w1_prior: float | None = None,
    decision_to_request_rate: Any = None,
) -> Any:
    """One decision tick for ``(candidates, depositors)`` arrays.

    Matches ``engine.agents.decide._evaluate``. ``belief``, ``balance``, and
    ``already`` update in place. Returns withdrawal amounts, shape ``(C, N)``.
    Static depositor columns may be ``(N,)`` (shared seed) or ``(C, N)``.
    ``dt`` scales the withdrawal take: ``1 - (1 - per_minute) ** dt``.
    At ``dt = 1`` the take matches the unscaled function exactly.

    SVB-17: when ``adjacency`` (one :func:`build_seed_adjacency` per seed) and
    ``seed_slices`` are given, both contact terms are per-seed CSR products and
    ``contacts``/``hub`` are not read. Without them, the gather path runs.

    SVB-28: ``decide_once=True`` makes each depositor decide at most once. Hard
    gates: a depositor who acts and hasn't withdrawn requests ``segment_fraction``
    of the uninsured gap once, then never again.     Soft gates: ``decided`` (float,
    updated in place; defaults to a scratch buffer seeded from ``already``) is the
    probability the depositor has decided; it ratchets at finite rate
    ``d_new = decided + (1-exp(-dt*h))*max(0, p_act-decided)`` (SVB-35; ``h`` is
    the decision-to-request rate). That closed form composes across sub-steps
    when ``p_act`` is held constant, so a coarse tick of length T matches the
    one-minute expansion of the same ``p_act``. Only the increase is requested,
    against the *initial* uninsured balance.
    Contagion reads ``decided`` instead of the boolean ``already``. No per-tick
    re-roll, so a calm depositor's small ``p_act`` no longer drains the book.
    ``decide_once=False`` keeps the legacy per-tick take (direct-call tests).

    SVB-33: per-depositor ``aware`` (float, 0 to 1, in-place; defaults to a
    scratch buffer filled with ``aware0``, default 0) updates before the logits:
    ``aware += (1-aware)*(1-exp(-dt*(r_media*attention + r_wom*(w1/w1_prior)*share
    + r_news*max(0, news))))``. Feature 0 is ``attention * aware``. Soft once-mode
    sizes the request as ``mobile_share * initial_uninsured * gap_scale *
    amount_scale``; the hard-gate path still uses ``segment_fraction``. ``news``
    is the current (pre-tick) feedback-news level. Rates default from the
    registry when omitted. When ``aware is None``, ``scratch["aware"]`` is filled
    from ``aware0`` on every call so a reused scratch does not persist awareness.
    """
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
    unfused = _unfused_logits_enabled()
    fused_logits = None if unfused else _fuse(xp)(_logits_from_features)
    use_fused_part2 = _part2_kernels_enabled(xp)
    if not unfused:
        w_cols = [weights[:, j][:, None] for j in range(int(weights.shape[1]))]
        stock_col = stock_return[:, None]
        bias_col = bias[:, None]
        relief_col = rate_relief[:, None]
    f32 = getattr(xp, "float32", np.float32)
    if mobile_share is None:
        mobile_share = xp.ones(n_candidates, dtype=f32)
    else:
        mobile_share = xp.asarray(mobile_share, dtype=f32)
        if int(getattr(mobile_share, "ndim", 0)) == 0:
            mobile_share = xp.full(n_candidates, mobile_share, dtype=f32)
    if initial_uninsured is None:
        initial_uninsured = xp.maximum(0.0, xp.maximum(balance, 0.0) - insured)
    else:
        initial_uninsured = xp.asarray(initial_uninsured, dtype=f32)
    # The cap is applied with ``where(mobile_share < 1, ...)``, so no host-side check
    # (and no per-tick ``.get()`` sync) is needed.
    apply_mobile_cap = True
    soft_once = bool(decide_once) and temp > 0.0
    if decide_once:
        if decided is None:
            decided = scratch.get("decided")
            if decided is None or tuple(decided.shape) != (n_candidates, n_depositors):
                decided = already.astype(f32)
                scratch["decided"] = decided
    if aware is None:
        fill = float(aware0)
        aware = scratch.get("aware")
        if aware is None or tuple(aware.shape) != (n_candidates, n_depositors):
            aware = xp.full((n_candidates, n_depositors), fill, dtype=f32)
            scratch["aware"] = aware
        else:
            aware.fill(fill)
    r_media, r_wom, r_news, w1_prior = _awareness_rate_args(
        xp,
        n_candidates,
        f32,
        awareness_media_rate,
        awareness_wom_rate,
        awareness_news_rate,
        awareness_w1_prior,
    )
    h_col = _decision_rate_arg(xp, n_candidates, f32, decision_to_request_rate)
    news_col = _lane_col(xp, news, n_candidates, f32)
    w1_col = weights[:, 1][:, None] if int(weights.shape[1]) > 1 else xp.zeros((n_candidates, 1), dtype=f32)
    # SVB-19: loop 2 reuses loop 1's safe / gap balances.
    gap_keep = scratch.get("gap_keep")
    if gap_keep is None or tuple(gap_keep.shape) != (n_candidates, n_depositors):
        gap_keep = xp.empty((n_candidates, n_depositors), dtype=f32)
    safe_keep = scratch.get("safe_keep")
    if safe_keep is None or tuple(safe_keep.shape) != (n_candidates, n_depositors):
        safe_keep = xp.empty((n_candidates, n_depositors), dtype=f32)
    use_spmm = (
        adjacency is not None
        and seed_slices is not None
        and len(adjacency) > 0
        and all(item is not None for item in adjacency)
    )
    # Contagion source: in soft once-mode the decided probability, else who has withdrawn.
    share_source = decided if soft_once else already
    if use_spmm:
        degree = int(adjacency[0]["degree"])
        share_full = contact_terms(
            xp, [item["share"] for item in adjacency], seed_slices, share_source.astype(f32)
        )

    for start in range(0, n_depositors, width):
        stop = min(n_depositors, start + width)
        sl = slice(start, stop)
        columns = stop - start
        if degree == 0:
            share = xp.zeros((n_candidates, columns), dtype=f32)
        elif use_spmm:
            share = share_full[:, sl]
        else:
            share = _gather_neighbors(share_source, contacts, sl, xp).astype(f32).mean(axis=2)
        signal = fused_signal(
            signal_base[:, None],
            signal_attention[:, None],
            attn,
            signal_stock[:, None],
            drawdown,
        )
        if use_fused_part2:
            bdt = belief.dtype
            updated = _belief_elementwise()(
                belief[:, sl],
                xp.asarray(signal, dtype=bdt),
                xp.asarray(strength, dtype=bdt),
            )
        else:
            updated = _update_belief(xp, belief[:, sl], signal, strength)
        belief[:, sl] = updated
        safe = xp.maximum(balance[:, sl], 0.0)
        insured_rows = _slice_static(insured, sl)
        gap_balance = xp.maximum(0.0, safe - insured_rows) * gap_scale[:, None]
        gap_fraction = fused_gap(safe, gap_balance)
        gap_keep[:, sl] = gap_balance
        safe_keep[:, sl] = safe
        payroll_rows = _slice_static(payroll, sl)
        _update_awareness(
            xp,
            aware[:, sl],
            attn,
            share,
            news_col,
            w1_col,
            w1_prior,
            r_media,
            r_wom,
            r_news,
            float(dt),
            f32,
            use_fused=use_fused_part2,
        )
        feat0 = attn * aware[:, sl]
        if unfused:
            if getattr(payroll_rows, "ndim", 1) == 1:
                payroll_rows = xp.broadcast_to(payroll_rows, (n_candidates, columns))
            features = xp.stack(
                [
                    feat0,
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
        else:
            if getattr(payroll_rows, "ndim", 1) == 1:
                payroll_rows = payroll_rows[None, :]
            logits = fused_logits(
                *w_cols,
                feat0, share, updated, stock_col, directive[:, sl], payroll_rows, gap_fraction,
                bias_col, statement_shift[:, sl], relief_col, deposit_rate[:, sl],
            )
        if use_fused_part2:
            worry = _sigmoid_elementwise()(xp.asarray(logits, dtype=f32))
        else:
            worry = _sigmoid(xp, logits)
        threshold = _slice_static(base_threshold, sl) + threshold_shift[:, sl]
        if temp <= 0.0:
            worried[:, sl] = worry > threshold
        else:
            if use_fused_part2:
                worried[:, sl] = _sigmoid_elementwise()(
                    xp.asarray((worry - threshold) / temp, dtype=f32)
                )
            else:
                worried[:, sl] = _sigmoid(xp, (worry - threshold) / temp)

    if use_spmm and degree:
        worried_source = worried if worried.dtype == f32 else worried.astype(f32)
        contacts_full = contact_terms(
            xp, [item["worry"] for item in adjacency], seed_slices, worried_source
        )

    for start in range(0, n_depositors, width):
        stop = min(n_depositors, start + width)
        sl = slice(start, stop)
        safe = safe_keep[:, sl]
        gap_balance = gap_keep[:, sl]
        if degree == 0:
            worried_contacts = xp.zeros((n_candidates, stop - start), dtype=f32)
        elif use_spmm:
            worried_contacts = contacts_full[:, sl]
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
        segment_fraction_rows = _slice_static(segment_fraction, sl)
        dt_value = float(dt)
        mobile_col = xp.asarray(mobile_share[:, None], dtype=f32)
        init_u = xp.asarray(_slice_static(initial_uninsured, sl), dtype=f32)
        if decide_once:
            gap_col = xp.asarray(gap_scale[:, None], dtype=f32)
            if use_fused_part2:
                _once_amount_elementwise()(
                    safe,
                    gap_balance,
                    worried[:, sl].astype(f32, copy=False),
                    worried_contacts,
                    vip_amount[:, sl],
                    segment_fraction_rows,
                    amount_scale[:, None],
                    k[:, None],
                    balance[:, sl],
                    already[:, sl],
                    decided[:, sl],
                    f32(minimum_amount),
                    f32(temp),
                    mobile_col,
                    init_u,
                    gap_col,
                    f32(dt_value),
                    h_col,
                    amounts[:, sl],
                    balance[:, sl],
                    already[:, sl],
                    decided[:, sl],
                )
            else:
                _once_amount_fallback(
                    xp,
                    safe=safe,
                    gap_balance=gap_balance,
                    worried_sl=worried[:, sl],
                    worried_contacts=worried_contacts,
                    vip_amount_sl=vip_amount[:, sl],
                    segment_fraction_rows=segment_fraction_rows,
                    amount_scale_col=amount_scale[:, None],
                    k_col=k[:, None],
                    amounts_sl=amounts[:, sl],
                    balance_sl=balance[:, sl],
                    already_sl=already[:, sl],
                    decided_sl=decided[:, sl],
                    minimum_amount=minimum_amount,
                    temp=temp,
                    f32=f32,
                    mobile_share_col=mobile_col,
                    initial_uninsured_sl=init_u,
                    gap_scale_col=gap_col,
                    dt=dt_value,
                    h_col=h_col,
                )
        elif use_fused_part2:
            _amount_elementwise()(
                safe,
                gap_balance,
                worried[:, sl].astype(f32, copy=False),
                worried_contacts,
                vip_amount[:, sl],
                segment_fraction_rows,
                amount_scale[:, None],
                k[:, None],
                balance[:, sl],
                already[:, sl],
                f32(minimum_amount),
                f32(dt_value),
                f32(temp),
                np.int32(1 if dt_value == 1.0 else 0),
                mobile_col,
                init_u,
                amounts[:, sl],
                balance[:, sl],
                already[:, sl],
            )
        else:
            _amount_chain_fallback(
                xp,
                safe=safe,
                gap_balance=gap_balance,
                worried_sl=worried[:, sl],
                worried_contacts=worried_contacts,
                vip_amount_sl=vip_amount[:, sl],
                segment_fraction_rows=segment_fraction_rows,
                amount_scale_col=amount_scale[:, None],
                k_col=k[:, None],
                amounts_sl=amounts[:, sl],
                balance_sl=balance[:, sl],
                already_sl=already[:, sl],
                minimum_amount=minimum_amount,
                dt_value=dt_value,
                temp=temp,
                f32=f32,
                mobile_share_col=mobile_col,
                initial_uninsured_sl=init_u,
                apply_mobile_cap=apply_mobile_cap,
            )
    return amounts


_PART2_KERNELS: dict[str, Any] = {}


def _unfused_logits_enabled() -> bool:
    """``STAMPEDE_CALIB_UNFUSED_LOGITS=1`` restores the pre-SVB-19 feature stack + einsum (parity tests only)."""
    import os

    return os.environ.get("STAMPEDE_CALIB_UNFUSED_LOGITS", "").strip() == "1"


def _unfused_part2_enabled() -> bool:
    """``STAMPEDE_CALIB_UNFUSED_PART2=1`` restores the Python amount / belief / sigmoid path."""
    import os

    return os.environ.get("STAMPEDE_CALIB_UNFUSED_PART2", "").strip() == "1"


def _part2_kernels_enabled(xp: Any) -> bool:
    """CuPy elementwise kernels for SVB-19 part 2. NumPy always stays on the fallback."""
    return getattr(xp, "__name__", "") == "cupy" and not _unfused_part2_enabled()


def _elementwise_kernel(name: str, in_params: str, out_params: str, operation: str) -> Any:
    cached = _PART2_KERNELS.get(name)
    if cached is not None:
        return cached
    import cupy

    kernel = cupy.ElementwiseKernel(in_params, out_params, operation, name)
    _PART2_KERNELS[name] = kernel
    return kernel


def _belief_elementwise() -> Any:
    """``decide._update_belief`` as one elementwise kernel (eps = ``_BAYES_EPS``)."""
    return _elementwise_kernel(
        "svb19_belief_update",
        "T belief, T signal, T strength",
        "T updated",
        """
        T eps = (T)""" + repr(float(_BAYES_EPS)) + """;
        T prior = belief;
        if (prior < eps) prior = eps;
        T prior_hi = (T)1 - eps;
        if (prior > prior_hi) prior = prior_hi;
        T likelihood = signal;
        if (likelihood < eps) likelihood = eps;
        if (likelihood > prior_hi) likelihood = prior_hi;
        T posterior = (likelihood * prior) / (
            likelihood * prior + ((T)1 - likelihood) * ((T)1 - prior)
        );
        updated = ((T)1 - strength) * prior + strength * posterior;
        """,
    )


def _sigmoid_elementwise() -> Any:
    """``decide._sigmoid``: exp(-abs) then the two-sided stable form."""
    return _elementwise_kernel(
        "svb19_sigmoid",
        "T logits",
        "T y",
        """
        T exp_neg = exp(-fabs(logits));
        T denom = (T)1 + exp_neg;
        y = (logits >= (T)0) ? ((T)1 / denom) : (exp_neg / denom);
        """,
    )


def _amount_elementwise() -> Any:
    """Loop-2 amount chain: dt take, VIP force, min gate, mobile-share cap, balance, already."""
    return _elementwise_kernel(
        "svb24_amount_chain",
        "T safe, T gap_balance, T worried, T worried_contacts, "
        "T vip_amount, T segment_fraction, T amount_scale, T k, "
        "T balance, bool already, T min_amount, T dt, T softness, int32 use_dt1, "
        "T mobile_share, T initial_uninsured",
        "T amount, T balance_out, bool already_out",
        """
        bool vip_force = vip_amount >= min_amount;
        T vip_take = min(vip_amount, safe) * amount_scale;
        T amt;
        bool withdraw;
        if (softness > (T)0) {
            T z_k = (worried_contacts - k) / softness;
            T exp_k = exp(-fabs(z_k));
            T den_k = (T)1 + exp_k;
            T p_k = (z_k >= (T)0) ? ((T)1 / den_k) : (exp_k / den_k);
            T p_act = worried * p_k;
            if (vip_force) p_act = (T)1;
            if (use_dt1) {
                T base = segment_fraction * gap_balance * amount_scale;
                amt = vip_force ? vip_take : base;
                amt = amt * p_act;
            } else {
                T per_minute = segment_fraction * p_act;
                if (per_minute < (T)0) per_minute = (T)0;
                if (per_minute > (T)1) per_minute = (T)1;
                T take = (T)1 - pow((T)1 - per_minute, dt);
                amt = gap_balance * amount_scale * take;
                if (vip_force) amt = vip_take;
            }
            T z_g = (amt - min_amount) / softness;
            T exp_g = exp(-fabs(z_g));
            T den_g = (T)1 + exp_g;
            T gate = (z_g >= (T)0) ? ((T)1 / den_g) : (exp_g / den_g);
            amt = amt * gate;
            withdraw = amt > (T)0;
        } else {
            bool act = ((worried != (T)0) && (worried_contacts >= k)) || vip_force;
            if (use_dt1) {
                T base = segment_fraction * gap_balance * amount_scale;
                amt = vip_force ? vip_take : base;
            } else {
                T frac = segment_fraction;
                if (frac < (T)0) frac = (T)0;
                if (frac > (T)1) frac = (T)1;
                T take = (T)1 - pow((T)1 - frac, dt);
                amt = gap_balance * amount_scale * take;
                if (vip_force) amt = vip_take;
            }
            if (!act) amt = (T)0;
            if (amt < min_amount) amt = (T)0;
            withdraw = amt >= min_amount;
        }
        if (mobile_share < (T)1) {
            T floor = ((T)1 - mobile_share) * initial_uninsured;
            T room = gap_balance - floor;
            if (room < (T)0) room = (T)0;
            if (amt > room) amt = room;
        }
        amount = amt;
        T next = balance - amt;
        balance_out = next > (T)0 ? next : (T)0;
        already_out = already || withdraw;
        """,
    )


def _amount_chain_fallback(
    xp: Any,
    *,
    safe: Any,
    gap_balance: Any,
    worried_sl: Any,
    worried_contacts: Any,
    vip_amount_sl: Any,
    segment_fraction_rows: Any,
    amount_scale_col: Any,
    k_col: Any,
    amounts_sl: Any,
    balance_sl: Any,
    already_sl: Any,
    minimum_amount: float,
    dt_value: float,
    temp: float,
    f32: Any,
    mobile_share_col: Any,
    initial_uninsured_sl: Any,
    apply_mobile_cap: bool = False,
) -> None:
    """Current loop-2 amount chain. NumPy and ``STAMPEDE_CALIB_UNFUSED_PART2=1``."""
    vip_force = vip_amount_sl >= minimum_amount
    vip_take = xp.minimum(vip_amount_sl, safe) * amount_scale_col
    if temp <= 0.0:
        rule = worried_sl & (worried_contacts >= k_col)
        act = rule | vip_force
        if dt_value == 1.0:
            base_amount = segment_fraction_rows * gap_balance * amount_scale_col
            amount = xp.where(vip_force, vip_take, base_amount)
        else:
            take = 1.0 - xp.power(
                1.0 - xp.clip(segment_fraction_rows, 0.0, 1.0), dt_value
            )
            amount = gap_balance * amount_scale_col * take
            amount = xp.where(vip_force, vip_take, amount)
        amount = xp.where(act, amount, 0.0)
        amount = xp.where(amount >= minimum_amount, amount, 0.0)
        withdraw = amount >= minimum_amount
    else:
        p_k = _sigmoid(xp, (worried_contacts - k_col) / temp)
        p_act = worried_sl.astype(f32) * p_k
        p_act = xp.where(vip_force, xp.ones_like(p_act), p_act)
        if dt_value == 1.0:
            base_amount = segment_fraction_rows * gap_balance * amount_scale_col
            amount = xp.where(vip_force, vip_take, base_amount)
            amount = amount * p_act
        else:
            per_minute = xp.clip(segment_fraction_rows * p_act, 0.0, 1.0)
            take = 1.0 - xp.power(1.0 - per_minute, dt_value)
            amount = gap_balance * amount_scale_col * take
            amount = xp.where(vip_force, vip_take, amount)
        amount = amount * _sigmoid(xp, (amount - minimum_amount) / temp)
        withdraw = amount > 0.0
    if apply_mobile_cap:
        floor = (1.0 - mobile_share_col) * initial_uninsured_sl
        room = xp.maximum(0.0, gap_balance - floor)
        amount = xp.where(mobile_share_col < 1.0, xp.minimum(amount, room), amount)
    amounts_sl[...] = amount
    balance_sl[...] = xp.maximum(0.0, balance_sl - amount)
    already_sl[...] = already_sl | withdraw


def _once_amount_elementwise() -> Any:
    """SVB-28 loop-2 chain: each depositor decides at most once (see ``step_depositor_batch``)."""
    return _elementwise_kernel(
        "svb28_once_amount",
        "T safe, T gap_balance, T worried, T worried_contacts, "
        "T vip_amount, T segment_fraction, T amount_scale, T k, "
        "T balance, bool already, T decided, T min_amount, T softness, "
        "T mobile_share, T initial_uninsured, T gap_scale, T dt, T h",
        "T amount, T balance_out, bool already_out, T decided_out",
        """
        bool vip_force = vip_amount >= min_amount;
        T vip_take = min(vip_amount, safe) * amount_scale;
        T amt = (T)0;
        T d_new = decided;
        bool withdraw = false;
        if (softness > (T)0) {
            T z_k = (worried_contacts - k) / softness;
            T exp_k = exp(-fabs(z_k));
            T den_k = (T)1 + exp_k;
            T p_k = (z_k >= (T)0) ? ((T)1 / den_k) : (exp_k / den_k);
            T p_act = worried * p_k;
            T gap = p_act - d_new;
            if (gap < (T)0) gap = (T)0;
            T alpha = (T)1 - exp(-dt * h);
            d_new = d_new + alpha * gap;
            T base = mobile_share * initial_uninsured * gap_scale * amount_scale;
            if (base >= min_amount) amt = (d_new - decided) * base;
            withdraw = d_new >= (T)0.5;
        } else {
            bool act = (worried != (T)0) && (worried_contacts >= k);
            if (act && !already) {
                amt = segment_fraction * gap_balance * amount_scale;
                if (amt < min_amount) amt = (T)0;
            }
        }
        if (vip_force) amt = vip_take;
        if (mobile_share < (T)1) {
            T floor = ((T)1 - mobile_share) * initial_uninsured;
            T room = gap_balance - floor;
            if (room < (T)0) room = (T)0;
            if (amt > room) amt = room;
        }
        if (softness <= (T)0) {
            withdraw = amt >= min_amount;
            if (withdraw) d_new = (T)1;
        } else if (vip_force && amt > (T)0) {
            withdraw = true;
        }
        amount = amt;
        T next = balance - amt;
        balance_out = next > (T)0 ? next : (T)0;
        already_out = already || withdraw;
        decided_out = d_new;
        """,
    )


def _once_amount_fallback(
    xp: Any,
    *,
    safe: Any,
    gap_balance: Any,
    worried_sl: Any,
    worried_contacts: Any,
    vip_amount_sl: Any,
    segment_fraction_rows: Any,
    amount_scale_col: Any,
    k_col: Any,
    amounts_sl: Any,
    balance_sl: Any,
    already_sl: Any,
    decided_sl: Any,
    minimum_amount: float,
    temp: float,
    f32: Any,
    mobile_share_col: Any,
    initial_uninsured_sl: Any,
    gap_scale_col: Any,
    dt: float = 1.0,
    h_col: Any = None,
) -> None:
    """NumPy (and ``STAMPEDE_CALIB_UNFUSED_PART2=1``) version of ``svb28_once_amount``."""
    vip_force = vip_amount_sl >= minimum_amount
    vip_take = xp.minimum(vip_amount_sl, safe) * amount_scale_col
    if temp > 0.0:
        p_k = _sigmoid(xp, (worried_contacts - k_col) / temp)
        p_act = worried_sl.astype(f32) * p_k
        if h_col is None:
            from engine.agents.decide import decision_to_request_rate_default

            h_col = f32(decision_to_request_rate_default())
        alpha = f32(1.0) - xp.exp(-f32(dt) * h_col)
        d_new = decided_sl + alpha * xp.maximum(f32(0.0), p_act - decided_sl)
        base = mobile_share_col * initial_uninsured_sl * gap_scale_col * amount_scale_col
        amount = xp.where(base >= minimum_amount, (d_new - decided_sl) * base, 0.0)
        withdraw = d_new >= 0.5
    else:
        act = worried_sl.astype(bool) & (worried_contacts >= k_col)
        first = act & ~already_sl
        amount = xp.where(first, segment_fraction_rows * gap_balance * amount_scale_col, 0.0)
        amount = xp.where(amount >= minimum_amount, amount, 0.0)
        d_new = decided_sl
    amount = xp.where(vip_force, vip_take, amount)
    floor = (1.0 - mobile_share_col) * initial_uninsured_sl
    room = xp.maximum(0.0, gap_balance - floor)
    amount = xp.where(mobile_share_col < 1.0, xp.minimum(amount, room), amount)
    if temp > 0.0:
        withdraw = withdraw | (vip_force & (amount > 0.0))
    else:
        withdraw = amount >= minimum_amount
        d_new = xp.where(withdraw, xp.ones_like(d_new), d_new)
    amounts_sl[...] = amount
    balance_sl[...] = xp.maximum(0.0, balance_sl - amount)
    already_sl[...] = already_sl | withdraw
    decided_sl[...] = d_new


def _lane_col(xp: Any, value: Any, n_candidates: int, f32: Any) -> Any:
    """Broadcast a scalar or ``(C,)`` lane vector to ``(C, 1)``."""
    if value is None:
        return xp.zeros((n_candidates, 1), dtype=f32)
    arr = xp.asarray(value, dtype=f32)
    ndim = int(getattr(arr, "ndim", 0))
    if ndim == 0:
        return xp.full((n_candidates, 1), arr, dtype=f32)
    if ndim == 1:
        return arr[:, None]
    return arr


def _decision_rate_arg(xp: Any, n_candidates: int, f32: Any, rate: Any) -> Any:
    """Lane column for the decide-once request hazard. Registry default if omitted."""
    from engine.agents.decide import decision_to_request_rate_default

    h0 = decision_to_request_rate_default()
    return _lane_col(xp, h0 if rate is None else rate, n_candidates, f32)


def _awareness_rate_args(
    xp: Any,
    n_candidates: int,
    f32: Any,
    media: Any,
    wom: Any,
    news_rate: Any,
    w1_prior: float | None,
) -> tuple[Any, Any, Any, float]:
    """Lane columns for the three awareness rates, plus the contact-weight prior."""
    from engine.agents.decide import awareness_rate_defaults

    media0, wom0, news0, w1_0 = awareness_rate_defaults()
    r_media = _lane_col(xp, media0 if media is None else media, n_candidates, f32)
    r_wom = _lane_col(xp, wom0 if wom is None else wom, n_candidates, f32)
    r_news = _lane_col(xp, news0 if news_rate is None else news_rate, n_candidates, f32)
    prior = float(w1_0 if w1_prior is None else w1_prior)
    if prior == 0.0:
        prior = float(w1_0) if float(w1_0) != 0.0 else 1.0
    return r_media, r_wom, r_news, prior


def _update_awareness(
    xp: Any,
    aware_sl: Any,
    attention: Any,
    share: Any,
    news: Any,
    w1: Any,
    w1_prior: float,
    r_media: Any,
    r_wom: Any,
    r_news: Any,
    dt: float,
    f32: Any,
    *,
    use_fused: bool,
) -> None:
    """In-place hazard update. Fused CuPy kernel or the NumPy/unfused xp path."""
    dt_val = f32(dt)
    prior = f32(w1_prior)
    if use_fused:
        _awareness_elementwise()(
            xp.asarray(aware_sl, dtype=f32),
            xp.asarray(attention, dtype=f32),
            xp.asarray(share, dtype=f32),
            xp.asarray(news, dtype=f32),
            xp.asarray(w1, dtype=f32),
            prior,
            xp.asarray(r_media, dtype=f32),
            xp.asarray(r_wom, dtype=f32),
            xp.asarray(r_news, dtype=f32),
            dt_val,
            aware_sl,
        )
        return
    _awareness_fallback(
        xp,
        aware_sl=aware_sl,
        attention=attention,
        share=share,
        news=news,
        w1=w1,
        w1_prior=prior,
        r_media=r_media,
        r_wom=r_wom,
        r_news=r_news,
        dt=float(dt),
        f32=f32,
    )


def _awareness_elementwise() -> Any:
    """SVB-33 per-depositor awareness hazard (see ``step_depositor_batch``)."""
    return _elementwise_kernel(
        "svb33_awareness",
        "T aware, T attention, T share, T news, T w1, T w1_prior, "
        "T r_media, T r_wom, T r_news, T dt",
        "T aware_out",
        """
        T news_pos = news > (T)0 ? news : (T)0;
        T prior = w1_prior;
        if (prior == (T)0) prior = (T)1;
        T hazard = r_media * attention
            + r_wom * (w1 / prior) * share
            + r_news * news_pos;
        T inc = ((T)1 - aware) * ((T)1 - exp(-dt * hazard));
        T next = aware + inc;
        if (next < (T)0) next = (T)0;
        if (next > (T)1) next = (T)1;
        aware_out = next;
        """,
    )


def _awareness_fallback(
    xp: Any,
    *,
    aware_sl: Any,
    attention: Any,
    share: Any,
    news: Any,
    w1: Any,
    w1_prior: Any,
    r_media: Any,
    r_wom: Any,
    r_news: Any,
    dt: float,
    f32: Any,
) -> None:
    """NumPy (and ``STAMPEDE_CALIB_UNFUSED_PART2=1``) version of ``svb33_awareness``."""
    news_pos = xp.maximum(0.0, news)
    prior = float(w1_prior) if float(w1_prior) != 0.0 else 1.0
    hazard = (
        r_media * attention
        + r_wom * (w1 / prior) * share
        + r_news * news_pos
    )
    increment = (1.0 - aware_sl) * (1.0 - xp.exp(-float(dt) * hazard))
    nxt = aware_sl + increment
    aware_sl[...] = xp.clip(nxt, 0.0, 1.0).astype(f32, copy=False)


def _logits_from_features(
    w0: Any, w1: Any, w2: Any, w3: Any, w4: Any, w5: Any, w6: Any,
    attn: Any, share: Any, updated: Any, stock: Any, directive: Any, payroll: Any, gap_fraction: Any,
    bias: Any, statement_shift: Any, rate_relief: Any, deposit_rate: Any,
) -> Any:
    """SVB-19: the 7-feature weighted sum without materializing a ``(C, width, 7)`` stack."""
    return (
        w0 * attn
        + w1 * share
        + w2 * updated
        + w3 * stock
        + w4 * directive
        + w5 * payroll
        + w6 * gap_fraction
        + bias
        + statement_shift
        - rate_relief * deposit_rate
    )


def _signal_from_public(
    signal_base: Any,
    signal_attention: Any,
    attn: Any,
    signal_stock: Any,
    drawdown: Any,
) -> Any:
    raw = signal_base + signal_attention * attn + signal_stock * drawdown
    return _XP.clip(raw, 0.0, 1.0)


def _gap_fraction(safe: Any, gap_balance: Any) -> Any:
    denom = _XP.maximum(safe, _BAYES_EPS)
    return _XP.clip(gap_balance / denom, 0.0, 1.0) * (safe > 0.0)

