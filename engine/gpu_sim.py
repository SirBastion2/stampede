"""Batched M27 candidate evaluation (ADR-051).

The depositor update (worry, belief, contagion, withdrawal amounts) runs as
array math on NumPy or CuPy. With ``--gpu-sim`` only, rumor handling,
attention, the market quote, payments, the ledger, and securities stay on the
same module ``step`` methods as the CPU engine. ``--gpu-full``
(``STAMPEDE_CALIB_GPU_FULL=1``) also moves attention, the market quote, and
payments onto those array kernels. One ``WithdrawalRequested`` per channel
carries the summed amount; M07 settles that sum with the same capacities.

Used only when ``STAMPEDE_BACKEND=gpu`` and ``STAMPEDE_CALIB_GPU_SIM=1``.
The flag-off engine path does not call this module.

The same depositor kernel batches every scenario passed in, including the SVB
preset (E1) and the non-bank equity, crypto, and rates presets (E13, E15, E16).
Price ticks stay on the market ``step``. Agreement with the per-candidate CPU
engine is the E1 relative tolerance for both outflow fractions and price paths.

Stage 1 scores requested fractions through ``simulate_requested_fractions``:
attention, decisions, and the price-feedback loop stay on the device. No
per-lane engines, no events, and no M07/M08/M09/M20. Validation still uses
``simulate_logs``.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from contracts.events import (
    DepositRateChanged,
    DirectiveIssued,
    StatementIssued,
    VIPDecision,
    WithdrawalRequested,
)
from contracts.schemas import Scenario
from engine.agents.decide import (
    DecisionModule,
    apply_calibration_vector,
    awareness_rate_defaults,
    decision_to_request_rate_default,
    withdrawal_decision_once,
)
from engine.gpu_sim_kernels import (
    build_seed_adjacency,
    channel_totals_device,
    make_depositor_scratch,
    seed_slices_from_counts,
    step_depositor_batch,
)
from engine.modes import ModuleMode
from params.loader import get_parameter

_LOG = logging.getLogger("m27.calibration.gpu_sim")

_PARAM_BUDGET = "M27 GPU sim array budget bytes"
_PARAM_TOLERANCE = "M27 GPU sim E1 relative tolerance"

# UI / unused-graph modules. M03 still uses the 16-node fallback (ADR-028),
# so the M02 100k build does not feed the tick. M29 rescans the full log.
_SKIP_MODULES = frozenset({"M02", "M27", "M29", "M30", "M33"})

_CHANNELS: tuple[str, ...] = ("app", "wire", "ach", "branch")

_DYNAMIC_TYPES = (StatementIssued, DepositRateChanged, DirectiveIssued, VIPDecision)

_PARAM_KEYS = (
    "weights",
    "bias",
    "k",
    "belief_strength",
    "signal_attention",
    "signal_stock",
    "signal_base",
    "rate_relief",
    "awareness_media_rate",
    "awareness_wom_rate",
    "awareness_news_rate",
    "decision_to_request_rate",
)
_MUTABLE_KEYS = ("belief", "balance", "already", "decided", "aware")
_STATIC_PACK_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
# SVB-17: per-seed CSR adjacency for the lean stage-1 path. Static across ticks and generations.
_ADJACENCY_CACHE: dict[tuple[Any, ...], dict[str, Any] | None] = {}


def _gather_path_enabled() -> bool:
    """``STAMPEDE_CALIB_GATHER_PATH=1`` restores the pre-SVB-17 expanded gathers (parity tests only)."""
    return os.environ.get("STAMPEDE_CALIB_GATHER_PATH", "").strip() == "1"


def _seed_adjacency(scenario: Any, module: Any, xp: Any, n_dep: int) -> dict[str, Any] | None:
    key = (scenario.preset, int(scenario.seed), getattr(xp, "__name__", "xp"), int(n_dep))
    if key not in _ADJACENCY_CACHE:
        _ADJACENCY_CACHE[key] = build_seed_adjacency(xp, module.contacts, module.hub_weight)
    return _ADJACENCY_CACHE[key]


_PARAM_SCRIPTED_DRAWDOWN = "M27 calibration uses scripted drawdown"


def gpu_sim_enabled() -> bool:
    """True when the fit asked for the batched path and selected the GPU backend.

    CuPy is used when a device is visible. With the flags set and no device,
    the same path runs on NumPy so CI can test it.
    """
    from engine.xp import backend_preference

    flag = os.environ.get("STAMPEDE_CALIB_GPU_SIM", "").strip().lower()
    return backend_preference() == "gpu" and flag in {"1", "true", "yes", "on"}


def calibration_uses_scripted_drawdown() -> bool:
    """True unless the M27 registry row is ``no`` (demo default stays on)."""
    return str(get_parameter(_PARAM_SCRIPTED_DRAWDOWN).value).strip().lower() != "no"


def e1_relative_tolerance() -> float:
    """Relative ceiling on E1 day-1 and two-day fractions (ASSUMED registry row)."""
    return float(get_parameter(_PARAM_TOLERANCE).value)


def array_budget_bytes() -> int:
    """Resident-array ceiling for one candidate group (ASSUMED registry row)."""
    return int(float(get_parameter(_PARAM_BUDGET).value))


def relative_error(left: float, right: float) -> float:
    """Relative gap, scaled by the larger magnitude."""
    scale = max(abs(left), abs(right), 1e-30)
    return abs(left - right) / scale


def resolve_array_module(array_module: str = "auto") -> tuple[Any, str]:
    """Return ``(xp, "gpu"|"cpu")``.

    ``auto`` selects CuPy when it can see a CUDA device, otherwise NumPy.
    ``cpu`` / ``numpy`` always return NumPy. ``gpu`` / ``cupy`` raise when
    no device is visible.
    """
    requested = array_module.strip().lower()
    if requested in {"cpu", "numpy"}:
        return np, "cpu"
    cupy_module = _cupy_if_device()
    if requested in {"gpu", "cupy"}:
        if cupy_module is None:
            raise RuntimeError("CUDA device is not available for the M27 GPU sim")
        return cupy_module, "gpu"
    if requested not in {"auto", ""}:
        raise ValueError(f"unknown array module {array_module!r}")
    if cupy_module is None:
        return np, "cpu"
    return cupy_module, "gpu"


def _cupy_if_device() -> Any | None:
    try:
        import cupy as cp
    except ImportError:
        return None
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            return None
    except Exception:
        return None
    return cp


def _group_size(n_candidates: int, n_depositors: int, budget: int) -> int:
    """How many candidate×seed lanes share one device-resident state."""
    if n_candidates <= 1 or n_depositors <= 0:
        return max(1, n_candidates)
    # belief, balance, already, shifts, vip, worry, amounts, plus per-lane static
    per_candidate = n_depositors * 14 * 8
    fit = max(1, budget // max(per_candidate, 1))
    return min(n_candidates, fit)


def _chunk_width(n_candidates: int, n_depositors: int, degree: int, budget: int) -> int:
    """Depositor rows per scratch chunk. Resident state is budgeted separately."""
    if n_depositors <= 0:
        return 1
    per_row = n_candidates * (max(degree, 1) * 2 + 7) * 8
    room = max(budget // 2, 1)
    width = max(1, room // max(per_row, 1))
    return min(n_depositors, width)


def _to_host(array: Any) -> np.ndarray:
    """Host copy. Uses ``.get()`` when implicit NumPy conversion is forbidden."""
    if isinstance(array, np.ndarray):
        return np.asarray(array, dtype=np.float64)
    getter = getattr(array, "get", None)
    if callable(getter):
        return np.asarray(getter(), dtype=np.float64)
    return np.asarray(array, dtype=np.float64)


def _as_vectors(
    vectors: NDArray[np.float64] | Sequence[NDArray[np.float64]],
) -> list[NDArray[np.float64]]:
    if isinstance(vectors, np.ndarray) and vectors.ndim == 2:
        return [np.asarray(row, dtype=np.float64) for row in vectors]
    return [np.asarray(row, dtype=np.float64) for row in vectors]


def _channel_codes(module: DecisionModule) -> np.ndarray:
    from engine.bank.payments import normalize_channel

    codes = np.empty(len(module.segments), dtype=np.int64)
    for index, segment in enumerate(module.segments):
        raw = module.params.segment_channel.get(segment, module.params.default_channel)
        name = normalize_channel(raw)
        codes[index] = _CHANNELS.index(name) if name in _CHANNELS else _CHANNELS.index("wire")
    return codes


def _channel_index(module: DecisionModule, xp: Any) -> Any:
    return xp.asarray(_channel_codes(module))


def _consume_m06(module: DecisionModule) -> list[BaseModel]:
    """Apply the same inbox events ``DecisionModule.step`` would, without deciding."""
    module._ensure_population()
    new_events = module.inbox[module._cursor :]
    module._cursor = len(module.inbox)
    if module._placebo or not module._loaded:
        return []
    module._apply_events(new_events)
    return new_events


def adjustment_factor(dt: float, tau: float) -> float:
    """Discrete adjustment ``1 - exp(-dt/tau)`` for a continuous rate ``1/tau``."""
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    return float(1.0 - math.exp(-float(dt) / float(tau)))


def decay_factor(dt: float, half_life: float) -> float:
    """Half-life decay ``0.5 ** (dt / half_life)``."""
    if half_life <= 0.0:
        raise ValueError("half_life must be positive")
    return float(0.5 ** (float(dt) / float(half_life)))


def _adjustment_strength(xp: Any, strength: Any, dt: float) -> Any:
    """Map a per-minute adjustment onto ``dt`` via ``1 - exp(-dt/tau)``."""
    if float(dt) == 1.0:
        return strength
    clipped = xp.clip(strength, 0.0, 1.0 - 1e-15)
    return 1.0 - xp.power(1.0 - clipped, float(dt))


def _minute_substeps(dt: float) -> int:
    """Integer-minute coarse ticks expand to one clock minute each."""
    steps = int(round(float(dt)))
    if steps >= 2 and abs(float(dt) - float(steps)) <= 1e-9:
        return steps
    return 1


def _expand_minute_ticks(
    ticks: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Unroll integer ``dt > 1`` into one-minute ticks (ADR-082)."""
    out: list[tuple[float, float]] = []
    for t, dt in ticks:
        n_sub = _minute_substeps(dt)
        if n_sub == 1:
            out.append((float(t), float(dt)))
            continue
        for k in range(n_sub):
            out.append((float(t) + k, 1.0))
    return out


def _apply_tick_rates(engines: Sequence[Any], dt: float) -> None:
    """Scale module rates that expose a 1-minute decay or adjustment constant."""
    decay = None
    for engine in engines:
        attention = engine.get("M03")
        if attention is None or not hasattr(attention, "_decay"):
            continue
        if decay is None:
            half = getattr(getattr(attention, "params", None), "half_life_minutes", None)
            if half is None or float(half) <= 0.0:
                continue
            decay = decay_factor(dt, float(half))
        attention._decay = decay


def _step_module(engine: Any, module_id: str, t: float, dt: float = 1.0) -> None:
    step = engine[module_id].step
    try:
        events = step(float(t), dt=float(dt))
    except TypeError:
        events = step(float(t))
    for event in events:
        engine.bus.publish(event)


def _step_pre_full(
    engines: list[Any],
    modules: list[DecisionModule],
    pre: tuple[str, ...],
    t: float,
    xp: Any,
    shared_network: dict[str, Any],
    dt: float = 1.0,
) -> bool:
    from engine.gpu_full import step_attention_group

    dirty = False
    for module_id in pre:
        if module_id == "M03":
            step_attention_group(engines, t, xp, dt)
            continue
        for engine in engines:
            if module_id == "M02":
                _prepare_network(engine, shared_network)
            _step_module(engine, module_id, t, dt)
            if module_id == "M02":
                _capture_network(engine, shared_network)
    for module in modules:
        new_events = _consume_m06(module)
        if any(isinstance(event, _DYNAMIC_TYPES) for event in new_events):
            dirty = True
    return dirty


def _step_post(
    engines: list[Any],
    post: tuple[str, ...],
    t: float,
    xp: Any,
    full: bool,
    step_market_group: Any,
    step_payments_group: Any,
    dt: float = 1.0,
) -> None:
    for module_id in post:
        if full and module_id == "M07" and step_payments_group is not None:
            step_payments_group(engines, t, xp)
            continue
        if module_id == "M14" and step_market_group is not None:
            step_market_group(engines, t, xp, dt)
            continue
        for engine in engines:
            _step_module(engine, module_id, t, dt)


def _prepare_network(engine: Any, shared: dict[str, Any]) -> None:
    module = engine.get("M02")
    if module is None:
        return
    if getattr(module, "network", None) is None and "network" in shared:
        module.network = shared["network"]


def _capture_network(engine: Any, shared: dict[str, Any]) -> None:
    if "network" in shared:
        return
    module = engine.get("M02")
    if module is None:
        return
    network = getattr(module, "network", None)
    if network is not None:
        shared["network"] = network


def _f32(xp: Any) -> Any:
    return getattr(xp, "float32", np.float32)


def _f64(xp: Any) -> Any:
    return getattr(xp, "float64", np.float64)


def _stack_f32(xp: Any, rows: list[Any]) -> Any:
    return xp.asarray(np.stack(rows), dtype=_f32(xp))


def _shared_or_stack(xp: Any, rows: list[Any], *, dtype: Any) -> Any:
    host = [_to_host(row) if not isinstance(row, np.ndarray) else np.asarray(row) for row in rows]
    first = host[0]
    if all(np.array_equal(first, row) for row in host[1:]):
        return xp.asarray(first, dtype=dtype)
    return xp.asarray(np.stack(host), dtype=dtype)


def clear_static_pack_cache() -> None:
    """Drop cached per-seed static packs and adjacency. Tests call this between cases."""
    _STATIC_PACK_CACHE.clear()
    _ADJACENCY_CACHE.clear()


def _copy_xp(array: Any) -> Any:
    copier = getattr(array, "copy", None)
    if callable(copier):
        return copier()
    return np.array(array, copy=True)


def _pack_lane_params(modules: list[DecisionModule], xp: Any) -> dict[str, Any]:
    """Per-lane calibration rows. Rebuilt every generation."""
    first = modules[0]
    dtype = _f32(xp)
    media0, wom0, news0, w1_0 = awareness_rate_defaults()
    weights = [np.asarray(module.params.weights, dtype=np.float32) for module in modules]
    return {
        "weights": xp.asarray(np.stack(weights), dtype=dtype),
        "bias": xp.asarray([m.params.bias for m in modules], dtype=dtype),
        "k": xp.asarray([m.params.k for m in modules], dtype=dtype),
        "belief_strength": xp.asarray(
            [m.params.belief_strength for m in modules],
            dtype=dtype,
        ),
        "signal_attention": xp.asarray(
            [m.params.signal_attention for m in modules],
            dtype=dtype,
        ),
        "signal_stock": xp.asarray([m.params.signal_stock for m in modules], dtype=dtype),
        "signal_base": xp.asarray([m.params.signal_base for m in modules], dtype=dtype),
        "rate_relief": xp.asarray(
            [m.params.rate_relief_weight for m in modules],
            dtype=dtype,
        ),
        "awareness_media_rate": xp.asarray(
            [getattr(m, "awareness_media_rate", media0) for m in modules],
            dtype=dtype,
        ),
        "awareness_wom_rate": xp.asarray(
            [getattr(m, "awareness_wom_rate", wom0) for m in modules],
            dtype=dtype,
        ),
        "awareness_news_rate": xp.asarray(
            [getattr(m, "awareness_news_rate", news0) for m in modules],
            dtype=dtype,
        ),
        "decision_to_request_rate": xp.asarray(
            [
                getattr(m, "decision_to_request_rate", decision_to_request_rate_default())
                for m in modules
            ],
            dtype=dtype,
        ),
        "minimum_amount": float(first.params.minimum_amount),
        "awareness_w1_prior": float(getattr(first, "awareness_w1_prior", w1_0)),
    }


def _pack_static(modules: list[DecisionModule], xp: Any) -> dict[str, Any]:
    dtype = _f32(xp)
    packed = {
        "insured": _shared_or_stack(xp, [m.insured for m in modules], dtype=dtype),
        "base_threshold": _shared_or_stack(
            xp, [m.base_threshold for m in modules], dtype=dtype
        ),
        "payroll": _shared_or_stack(xp, [m.payroll for m in modules], dtype=dtype),
        "segment_fraction": _shared_or_stack(
            xp, [m.segment_fraction for m in modules], dtype=dtype
        ),
        "hub": _shared_or_stack(xp, [m.hub_weight for m in modules], dtype=dtype),
        "contacts": _shared_or_stack(xp, [m.contacts for m in modules], dtype=xp.int64),
        "belief": _stack_f32(xp, [m.belief for m in modules]),
        "balance": _stack_f32(xp, [m.balance for m in modules]),
        "already": xp.asarray(np.stack([m.already_withdrew for m in modules])),
        # SVB-28: soft once-mode decided probability, seeded from who has already withdrawn.
        "decided": _stack_f32(
            xp,
            [
                np.asarray(getattr(m, "decided", None), dtype=np.float32)
                if getattr(m, "decided", None) is not None
                and int(np.asarray(m.decided).size) == int(np.asarray(m.balance).size)
                else np.asarray(m.already_withdrew, dtype=np.float32)
                for m in modules
            ],
        ),
        "aware": _stack_f32(
            xp,
            [
                np.asarray(getattr(m, "aware", None), dtype=np.float32)
                if getattr(m, "aware", None) is not None
                and int(np.asarray(m.aware).size) == int(np.asarray(m.balance).size)
                else np.zeros(int(np.asarray(m.balance).size), dtype=np.float32)
                for m in modules
            ],
        ),
        "channel_index": _shared_or_stack(
            xp, [_channel_codes(module) for module in modules], dtype=xp.int64
        ),
        "initial_uninsured": _shared_or_stack(
            xp,
            [
                (
                    np.asarray(m.initial_uninsured, dtype=np.float32)
                    if getattr(m, "initial_uninsured", None) is not None
                    and int(np.asarray(m.initial_uninsured).size) == int(np.asarray(m.balance).size)
                    else np.maximum(
                        0.0,
                        np.asarray(m.balance, dtype=np.float32)
                        - np.asarray(m.insured, dtype=np.float32),
                    )
                )
                for m in modules
            ],
            dtype=dtype,
        ),
    }
    packed.update(_pack_lane_params(modules, xp))
    return packed


def _expand_shared(array: Any, n_lanes: int, xp: Any, n_depositors: int | None = None) -> Any:
    """Tile a seed-shared array to ``(n_lanes, ...)`` when concatenating seeds.

    Per-lane depositor columns are already ``(n_lanes, n_depositors[, ...])``
    (thresholds after SVB-06 dispersion) and must not gain another axis.
    """
    shape = tuple(array.shape)
    if (
        n_depositors is not None
        and len(shape) >= 2
        and int(shape[0]) == n_lanes
        and int(shape[1]) == n_depositors
    ):
        return array
    return xp.broadcast_to(array, (n_lanes,) + shape)


def _concat_seed_states(states: list[dict[str, Any]], xp: Any) -> dict[str, Any]:
    if len(states) == 1:
        return states[0]
    out: dict[str, Any] = {
        "minimum_amount": states[0]["minimum_amount"],
        "awareness_w1_prior": states[0].get(
            "awareness_w1_prior", awareness_rate_defaults()[3]
        ),
    }
    lane_counts = [int(item["belief"].shape[0]) for item in states]
    n_depositors = int(states[0]["belief"].shape[1])
    for key in states[0]:
        if key in {"minimum_amount", "awareness_w1_prior"}:
            continue
        pieces = [item[key] for item in states]
        if key in _PARAM_KEYS or key in _MUTABLE_KEYS:
            out[key] = xp.concatenate(pieces, axis=0)
            continue
        out[key] = xp.concatenate(
            [
                _expand_shared(piece, n_lanes, xp, n_depositors)
                for piece, n_lanes in zip(pieces, lane_counts)
            ],
            axis=0,
        )
    return out


def _store_static_cache(packed: dict[str, Any]) -> dict[str, Any]:
    static = {
        key: packed[key]
        for key in packed
        if key not in _PARAM_KEYS and key not in _MUTABLE_KEYS
    }
    initial = {key: _copy_xp(packed[key]) for key in _MUTABLE_KEYS}
    return {"static": static, "initial": initial}


def _working_from_static_cache(
    cached: dict[str, Any],
    modules: list[DecisionModule],
    xp: Any,
) -> dict[str, Any]:
    state = dict(cached["static"])
    for key in _MUTABLE_KEYS:
        state[key] = _copy_xp(cached["initial"][key])
    state.update(_pack_lane_params(modules, xp))
    return state


def _static_state_for_group(
    modules: list[DecisionModule],
    scenarios: list[Scenario],
    xp: Any,
    xp_name: str,
) -> dict[str, Any]:
    """Pack static arrays once per seed; refresh per-lane parameter rows."""
    by_seed: dict[int, list[int]] = {}
    for index, item in enumerate(scenarios):
        by_seed.setdefault(int(item.seed), []).append(index)
    seed_states: list[dict[str, Any]] = []
    for indices in by_seed.values():
        seed_mods = [modules[i] for i in indices]
        n_depositors = int(np.asarray(seed_mods[0].belief).shape[0])
        key = (
            scenarios[indices[0]].preset,
            int(scenarios[indices[0]].seed),
            xp_name,
            len(seed_mods),
            n_depositors,
            float(getattr(seed_mods[0], "aware0", 0.0)),
        )
        cached = _STATIC_PACK_CACHE.get(key)
        if cached is None:
            packed = _pack_static(seed_mods, xp)
            _STATIC_PACK_CACHE[key] = _store_static_cache(packed)
            seed_states.append(packed)
        else:
            seed_states.append(_working_from_static_cache(cached, seed_mods, xp))
    return _concat_seed_states(seed_states, xp)


def _stack_dynamic(modules: list[DecisionModule], xp: Any) -> dict[str, Any]:
    return {
        "threshold_shift": _stack_f32(xp, [m.threshold_shift for m in modules]),
        "statement_shift": _stack_f32(xp, [m.statement_shift for m in modules]),
        "deposit_rate": _stack_f32(xp, [m.deposit_rate for m in modules]),
        "directive": _stack_f32(xp, [m.directive for m in modules]),
        "vip_amount": _stack_f32(xp, [m.vip_amount for m in modules]),
    }


def _scalars(modules: list[DecisionModule], xp: Any) -> dict[str, Any]:
    dtype = _f32(xp)
    return {
        "attention_level": xp.asarray([m.attention_level for m in modules], dtype=dtype),
        "stock_return": xp.asarray([m.stock_return for m in modules], dtype=dtype),
        "gap_scale": xp.asarray([m.gap_scale for m in modules], dtype=dtype),
        "amount_scale": xp.asarray([m.amount_scale for m in modules], dtype=dtype),
        "mobile_share": xp.asarray(
            [getattr(m, "mobile_share", 1.0) for m in modules],
            dtype=dtype,
        ),
        "news": xp.asarray(
            [float(getattr(m, "news_level", 0.0) or 0.0) for m in modules],
            dtype=dtype,
        ),
    }


def _channel_totals(
    xp: Any,
    amounts: Any,
    channel_index: Any,
    scratch: dict[str, Any] | None = None,
) -> np.ndarray:
    if scratch is None:
        out = xp.zeros((int(amounts.shape[0]), len(_CHANNELS)), dtype=xp.float64)
    else:
        out = scratch["channel_totals"]
    channel_totals_device(xp, amounts, channel_index, out)
    return _to_host(out)


def _emit_withdrawals(engine: Any, module: DecisionModule, totals: np.ndarray, t: float) -> None:
    for channel, name in enumerate(_CHANNELS):
        amount = float(totals[channel])
        if amount <= 0.0:
            continue
        module._seq += 1
        engine.bus.publish(
            WithdrawalRequested(
                t=float(t),
                seed=module.seed,
                bank_id=module.bank_id,
                cause_id=f"{module.module_id}-wd-sum-{name}-{module._seq}",
                depositor_id=f"gpu-sim-{name}",
                amount=amount,
                channel=name,
            )
        )


def _release_device(xp: Any, name: str) -> None:
    if name != "gpu":
        return
    pool = getattr(getattr(xp, "get_default_memory_pool", lambda: None)(), "free_all_blocks", None)
    if callable(pool):
        pool()


def _build_engine(scenario: Scenario) -> Any:
    from calibration.fit import real_calibration_modes
    from engine.registry import build_engine

    return build_engine(
        modes=real_calibration_modes(),
        default_mode=ModuleMode.FAKE,
        scenario=scenario,
    )


def _warm_seed_populations(scenarios: Sequence[Scenario]) -> None:
    """Solve M05 once per (preset, seed). Later lanes and generations hit the cache."""
    from engine.agents.population import sample_population

    seen: set[tuple[str, int]] = set()
    for item in scenarios:
        key = (item.preset, int(item.seed))
        if key in seen:
            continue
        seen.add(key)
        sample_population(item)


def _registry_capacity_multiplier() -> float:
    try:
        from engine.depositor_rows import resolve as _resolve_row
        from params.loader import get_parameter

        return float(get_parameter(_resolve_row("M07 capacity multiplier (FIT)")).value)
    except Exception:
        return 1.0


def _apply_lane_capacity_multipliers(
    engines: list[Any],
    multipliers: Sequence[float],
) -> None:
    """Replace each lane's app/wire/ACH scale with the trial FIT multiplier."""
    baseline = _registry_capacity_multiplier()
    denom = baseline if baseline != 0.0 else 1.0
    for engine, mult in zip(engines, multipliers, strict=True):
        module = engine.get("M07")
        if module is None or not hasattr(module, "limits"):
            continue
        factor = float(mult) / denom
        limits = module.limits
        module.limits = replace(
            limits,
            app_daily_limit=limits.app_daily_limit * factor,
            wire_capacity_per_tick=limits.wire_capacity_per_tick * factor,
            ach_batch_capacity=limits.ach_batch_capacity * factor,
        )


def _simulate_group(
    scenarios: list[Scenario],
    steps: int,
    vectors: list[NDArray[np.float64]],
    *,
    xp: Any,
    xp_name: str,
    shared_network: dict[str, Any],
    budget: int,
    disable_caps: bool = False,
    softness: float = 0.0,
    clamp_inputs: bool = True,
    timings: dict[str, float] | None = None,
    schedule: Sequence[tuple[float, float]] | None = None,
    capacity_multipliers: Sequence[float] | None = None,
    aware0: float | None = None,
) -> list[list[BaseModel]]:
    setup_started = time.perf_counter()
    decide_once = withdrawal_decision_once()
    engines = [_build_engine(item) for item in scenarios]
    if capacity_multipliers is not None:
        _apply_lane_capacity_multipliers(engines, capacity_multipliers)
    modules: list[DecisionModule] = []
    for engine, vector in zip(engines, vectors, strict=True):
        module = engine["M06"]
        if not isinstance(module, DecisionModule):
            raise RuntimeError("M27 GPU sim expected a real DecisionModule")
        apply_calibration_vector(module, vector, clamp_inputs=clamp_inputs)
        module.softness = float(softness)
        if aware0 is not None:
            module.aware0 = float(aware0)
        from engine.gpu_full_io import apply_lane_feedback

        apply_lane_feedback(module, vector)
        modules.append(module)

    skip = set(_SKIP_MODULES)
    if disable_caps:
        skip.update({"M07", "M20"})
    order = tuple(mid for mid in engines[0].order if mid not in skip)
    if "M06" not in order:
        raise RuntimeError("M27 GPU sim engine is missing M06")
    split = order.index("M06")
    pre = order[:split]
    post = order[split + 1 :]
    from engine.gpu_full import gpu_full_enabled

    full = gpu_full_enabled()
    # SVB-65: the per-lane path runs the fitted market and attention feedback too.
    from engine.gpu_full_io import apply_composed_attention, step_market_group

    step_payments_group = None
    mixed_caps = (
        capacity_multipliers is not None
        and len({float(x) for x in capacity_multipliers}) > 1
    )
    if full:
        from engine.gpu_full_io import step_payments_group

        if mixed_caps:
            step_payments_group = None

    state: dict[str, Any] | None = None
    dynamic: dict[str, Any] | None = None
    scratch: dict[str, Any] | None = None
    chunk = 1
    if timings is not None:
        timings["setup_s"] = timings.get("setup_s", 0.0) + (time.perf_counter() - setup_started)
        step_started = time.perf_counter()
    else:
        step_started = 0.0
    ticks = _expand_minute_ticks(
        [(float(t), float(dt)) for t, dt in schedule]
        if schedule is not None
        else [(float(t), 1.0) for t in range(steps)]
    )
    n_ticks = len(ticks)
    for index, (t, dt) in enumerate(ticks):
        _apply_tick_rates(engines, dt)
        dirty = False
        if full:
            dirty = _step_pre_full(engines, modules, pre, t, xp, shared_network, dt)
        else:
            for engine, module in zip(engines, modules, strict=True):
                for module_id in pre:
                    if module_id == "M02":
                        _prepare_network(engine, shared_network)
                    _step_module(engine, module_id, t, dt)
                    if module_id == "M02":
                        _capture_network(engine, shared_network)
                    if module_id == "M03":
                        apply_composed_attention([engine], t)
                new_events = _consume_m06(module)
                if any(isinstance(event, _DYNAMIC_TYPES) for event in new_events):
                    dirty = True
        if any(module._placebo or not module._loaded for module in modules):
            if not all(module._placebo or not module._loaded for module in modules):
                raise RuntimeError("M27 GPU sim mixed placebo and live candidates")
            _step_post(engines, post, t, xp, full, step_market_group, step_payments_group, dt)
            continue
        if state is None:
            from engine.gpu_full_io import apply_lane_feedback

            for module in modules:
                stored = getattr(module, "calibration_vector", None)
                if stored is not None:
                    apply_lane_feedback(module, stored)
            state = _static_state_for_group(modules, scenarios, xp, xp_name)
            contacts = state["contacts"]
            degree = int(contacts.shape[-1]) if getattr(contacts, "ndim", 1) >= 2 else 0
            n_dep = int(state["belief"].shape[1])
            chunk = _chunk_width(len(modules), n_dep, degree, budget)
            dynamic = _stack_dynamic(modules, xp)
            scratch = make_depositor_scratch(xp, len(modules), n_dep, softness)
            _LOG.info(
                "gpu-sim xp=%s candidates=%s depositors=%s ticks=%s chunk=%s full=%s",
                xp_name,
                len(modules),
                n_dep,
                n_ticks,
                chunk,
                full,
            )
        elif dirty:
            dynamic = _stack_dynamic(modules, xp)
        assert state is not None and dynamic is not None
        scalars = _scalars(modules, xp)
        amounts = step_depositor_batch(
            xp,
            belief=state["belief"],
            balance=state["balance"],
            already=state["already"],
            insured=state["insured"],
            base_threshold=state["base_threshold"],
            threshold_shift=dynamic["threshold_shift"],
            statement_shift=dynamic["statement_shift"],
            deposit_rate=dynamic["deposit_rate"],
            directive=dynamic["directive"],
            payroll=state["payroll"],
            segment_fraction=state["segment_fraction"],
            vip_amount=dynamic["vip_amount"],
            hub=state["hub"],
            contacts=state["contacts"],
            attention_level=scalars["attention_level"],
            stock_return=scalars["stock_return"],
            gap_scale=scalars["gap_scale"],
            amount_scale=scalars["amount_scale"],
            mobile_share=scalars["mobile_share"],
            initial_uninsured=state["initial_uninsured"],
            decided=state["decided"],
            decide_once=decide_once,
            news=scalars["news"],
            aware=state["aware"],
            awareness_media_rate=state["awareness_media_rate"],
            awareness_wom_rate=state["awareness_wom_rate"],
            awareness_news_rate=state["awareness_news_rate"],
            awareness_w1_prior=float(
                state.get("awareness_w1_prior", awareness_rate_defaults()[3])
            ),
            decision_to_request_rate=state.get("decision_to_request_rate"),
            weights=state["weights"],
            bias=state["bias"],
            k=state["k"],
            belief_strength=_adjustment_strength(xp, state["belief_strength"], dt),
            signal_attention=state["signal_attention"],
            signal_stock=state["signal_stock"],
            signal_base=state["signal_base"],
            rate_relief=state["rate_relief"],
            minimum_amount=state["minimum_amount"],
            chunk=chunk,
            softness=softness,
            scratch=scratch,
            dt=dt,
        )
        dynamic["vip_amount"] = xp.zeros_like(dynamic["vip_amount"])
        for module in modules:
            module.vip_amount.fill(0.0)
        totals = _channel_totals(xp, amounts, state["channel_index"], scratch)
        for engine, module, row in zip(engines, modules, totals, strict=True):
            _emit_withdrawals(engine, module, row, t)
        _step_post(engines, post, t, xp, full, step_market_group, step_payments_group, dt)
        if index == 0 or (index + 1) % 60 == 0 or index + 1 == n_ticks:
            _LOG.info("gpu-sim t=%s dt=%s tick %s/%s", t, dt, index + 1, n_ticks)
    logs = [list(engine.bus.log) for engine in engines]
    _release_device(xp, xp_name)
    if timings is not None:
        timings["step_s"] = timings.get("step_s", 0.0) + (time.perf_counter() - step_started)
    return logs


def simulate_logs(
    scenario: Scenario,
    steps: int,
    vectors: NDArray[np.float64] | Sequence[NDArray[np.float64]],
    *,
    array_module: str = "auto",
    disable_caps: bool = False,
    softness: float = 0.0,
    clamp_inputs: bool = True,
    seeds: Sequence[int] | None = None,
    timings: dict[str, float] | None = None,
    schedule: Sequence[tuple[float, float]] | None = None,
    capacity_multipliers: Sequence[float] | None = None,
    aware0: float | None = None,
) -> list[list[BaseModel]]:
    """Run one real-engine scenario for each behavioral vector.

    When ``seeds`` is omitted, candidates share the scenario seed. When it is
    set, every vector runs on every seed and logs are seed-major: seed0-vec0,
    seed0-vec1, …, seed1-vec0. Packed static state is cached per seed across
    generations; each generation writes only the per-lane parameter rows.
    Lanes are grouped so device-resident depositor
    state stays under the registry byte budget. Returns each lane's event log
    (aggregate withdrawals, not one event per depositor). The scenario may be
    the SVB, Signature, or placebo preset, or a non-bank equity, crypto, or
    rates preset. Price paths are the ``PriceTick`` events on that log. The
    parity ceiling is ``e1_relative_tolerance``. When ``timings`` is set, it
    receives ``setup_s`` (population + engine build) and ``step_s`` (tick loop).
    ``schedule`` is ``(t, dt)`` pairs on the simulation clock; omitted, the
    run is ``steps`` one-minute ticks. Rates scale with ``dt``.
    ``capacity_multipliers`` aligns with ``vectors`` and is repeated per seed;
    each lane's M07 app/wire/ACH limits become base × that multiplier.
    """
    rows = _as_vectors(vectors)
    if steps < 0:
        raise ValueError("steps must be >= 0")
    if not rows:
        return []
    seed_list = [int(scenario.seed)] if seeds is None else [int(item) for item in seeds]
    if not seed_list:
        return []
    mults: list[float] | None = None
    if capacity_multipliers is not None:
        if len(capacity_multipliers) != len(rows):
            raise ValueError("capacity_multipliers must match vectors")
        mults = [float(item) for item in capacity_multipliers]
    lanes_vectors: list[NDArray[np.float64]] = []
    lanes_scenarios: list[Scenario] = []
    lanes_mults: list[float] = []
    for seed in seed_list:
        seeded = (
            scenario
            if seed == int(scenario.seed)
            else scenario.model_copy(update={"seed": seed})
        )
        for index, row in enumerate(rows):
            lanes_vectors.append(row)
            lanes_scenarios.append(seeded)
            if mults is not None:
                lanes_mults.append(mults[index])
    resolved = (
        [(float(t), float(dt)) for t, dt in schedule] if schedule is not None else None
    )
    if (resolved is not None and not resolved) or (resolved is None and steps == 0):
        return [[] for _ in lanes_vectors]
    xp, xp_name = resolve_array_module(array_module)
    budget = array_budget_bytes()
    from engine.agents.population import _depositor_count_for_tier

    n_depositors = _depositor_count_for_tier(0)
    group = _group_size(len(lanes_vectors), n_depositors, budget)
    shared: dict[str, Any] = {}
    logs: list[list[BaseModel]] = []
    if timings is not None:
        timings.setdefault("setup_s", 0.0)
        timings.setdefault("step_s", 0.0)
    warm_started = time.perf_counter()
    _warm_seed_populations(lanes_scenarios)
    if timings is not None:
        timings["setup_s"] += time.perf_counter() - warm_started
    for start in range(0, len(lanes_vectors), group):
        group_mults = lanes_mults[start : start + group] if mults is not None else None
        logs.extend(
            _simulate_group(
                lanes_scenarios[start : start + group],
                steps,
                lanes_vectors[start : start + group],
                xp=xp,
                xp_name=xp_name,
                shared_network=shared,
                budget=budget,
                disable_caps=disable_caps,
                softness=softness,
                clamp_inputs=clamp_inputs,
                timings=timings,
                schedule=resolved,
                capacity_multipliers=group_mults,
                aware0=aware0,
            )
        )
    return logs


_PARAM_MINUTES_PER_DAY = "Non-bank sim minutes per day"
_DAY1_CLOSE = 390.0
_NEXT_OPEN = 1440.0


def _day1_close_minute() -> float:
    """Trading-session close on the SVB-29 clock; same row as calibration.fit.day1_close_minute."""
    return float(get_parameter(_PARAM_MINUTES_PER_DAY).value)


_STATEMENT_DRAWDOWN = 0.95
_STATEMENT_SEGMENTS = frozenset({"startup", "vc", "founder", "treasurer", "retail"})
_CRISIS_SEGMENTS = frozenset({"startup", "founder", "treasurer"})
_CEO_RATE_SEGMENT = "startup"
_THESIS_REACH = 5_000
_HEADLINE_REACH = 8_000
_CEO_RATE = 0.03
_CEO_RATE_CAP = 0.08
_THESIS_VULN = 0.2
_CEO_ATTENTION = 0.25
_VIP_ATTENTION = 0.25
_VIP_INTERVAL = 30


def _inject_seed_rumors(attention: Any, scenario: Scenario) -> None:
    """Put scenario rumors on the seed attention inbox (setup only)."""
    from agents.guardrails import claim_allowed
    from contracts.events import RumorPosted
    from scenarios.builder import poster_for_rumor
    from scenarios.presets import bank_for_preset

    bank_id = bank_for_preset(scenario.preset).id
    for rumor in scenario.rumors:
        if not claim_allowed(rumor.claim):
            continue
        attention.inbox.append(
            RumorPosted(
                t=0.0,
                seed=int(scenario.seed),
                bank_id=bank_id,
                cause_id=f"lean-{rumor.id}",
                rumor_id=rumor.id,
                claim=rumor.claim,
                source_node=poster_for_rumor(rumor),
                credibility=max(0.0, min(1.0, float(rumor.credibility))),
            )
        )


def _rumor_strength(scenario: Scenario) -> float:
    strength = 0.0
    for rumor in scenario.rumors:
        strength = max(strength, max(0.0, min(1.0, float(rumor.credibility))))
    return strength


def _seed_attention_engine(scenario: Scenario) -> Any:
    from engine.agents.attention import AttentionEngine
    from mocks.bus import EventBus

    engine = AttentionEngine()
    engine.attach(EventBus(), scenario)
    _inject_seed_rumors(engine, scenario)
    return engine


def _pack_lean_lanes(
    scenario: Scenario,
    vectors: list[NDArray[np.float64]],
    xp: Any,
    *,
    clamp_inputs: bool,
    aware0: float | None = None,
) -> dict[str, Any]:
    """Device arrays for every vector on one seed. No ``Engine`` is built."""
    from engine.agents.decide import (
        DecisionModule,
        awareness_news_rate_default,
        awareness_w1_prior_default,
        awareness_wom_rate_default,
        calibration_amount_scale,
        calibration_awareness_media_rate,
        calibration_decision_to_request_rate,
        calibration_mobile_share,
        community_placebo,
        decision_params_from_vector,
        load_decision_params,
    )
    from engine.gpu_full_io import feedback_params_from_vector
    from mocks.params_access import get_param

    module = DecisionModule()
    if aware0 is not None:
        module.aware0 = float(aware0)
    module.scenario = scenario
    placebo = community_placebo(scenario.preset)
    if not placebo:
        module._ensure_population()
    n_lanes = len(vectors)
    n_dep = int(np.asarray(module.belief).shape[0]) if module._loaded else 0
    raw = np.asarray(module.base_threshold, dtype=np.float32)
    median = float(get_param("M06 breaking-point median (ASSUMED)"))
    base = load_decision_params()
    dtype = _f32(xp)
    weights: list[np.ndarray] = []
    bias: list[float] = []
    k_vals: list[float] = []
    belief_strength: list[float] = []
    signal_attention: list[float] = []
    signal_stock: list[float] = []
    signal_base: list[float] = []
    rate_relief: list[float] = []
    amount_scale: list[float] = []
    mobile_share: list[float] = []
    media_rate: list[float] = []
    request_rate: list[float] = []
    gains: list[float] = []
    impacts: list[float] = []
    thresholds: list[np.ndarray] = []
    for vector in vectors:
        params = decision_params_from_vector(base, vector, clamp_inputs=clamp_inputs)
        gain, dispersion, impact = feedback_params_from_vector(vector)
        weights.append(np.asarray(params.weights, dtype=np.float32))
        bias.append(float(params.bias))
        k_vals.append(float(params.k))
        belief_strength.append(float(params.belief_strength))
        signal_attention.append(float(params.signal_attention))
        signal_stock.append(float(params.signal_stock))
        signal_base.append(float(params.signal_base))
        rate_relief.append(float(params.rate_relief_weight))
        amount_scale.append(float(calibration_amount_scale(vector, clamp_inputs=clamp_inputs)))
        mobile_share.append(float(calibration_mobile_share(vector, clamp_inputs=clamp_inputs)))
        media_rate.append(
            float(calibration_awareness_media_rate(vector, clamp_inputs=clamp_inputs))
        )
        request_rate.append(
            float(calibration_decision_to_request_rate(vector, clamp_inputs=clamp_inputs))
        )
        gains.append(float(gain))
        impacts.append(float(impact))
        if n_dep:
            thresholds.append(median + float(dispersion) * (raw - median))
        else:
            thresholds.append(np.zeros(0, dtype=np.float32))
    zeros = xp.zeros((n_lanes, max(n_dep, 0)), dtype=dtype)
    belief = (
        xp.broadcast_to(xp.asarray(module.belief, dtype=dtype), (n_lanes, n_dep)).copy()
        if n_dep
        else zeros
    )
    balance = (
        xp.broadcast_to(xp.asarray(module.balance, dtype=dtype), (n_lanes, n_dep)).copy()
        if n_dep
        else zeros
    )
    already = (
        xp.broadcast_to(xp.asarray(module.already_withdrew), (n_lanes, n_dep)).copy()
        if n_dep
        else xp.zeros((n_lanes, 0), dtype=bool)
    )
    decided = already.astype(dtype)  # SVB-28: soft once-mode decided probability
    aware = (
        xp.broadcast_to(xp.asarray(module.aware, dtype=dtype), (n_lanes, n_dep)).copy()
        if n_dep
        else zeros
    )
    contacts = (
        xp.asarray(module.contacts, dtype=xp.int64)
        if n_dep
        else xp.zeros((0, 0), dtype=xp.int64)
    )
    return {
        "placebo": placebo or not module._loaded,
        "n_dep": n_dep,
        "book": float(np.sum(module.balance)) if n_dep else 0.0,
        "strength": _rumor_strength(scenario),
        "seed": int(scenario.seed),
        "segments": list(module.segments),
        "logit_shift": float(module.params.statement_logit_shift),
        "thresh_shift": float(module.params.statement_threshold_shift),
        "minimum_amount": float(base.minimum_amount),
        "awareness_w1_prior": float(awareness_w1_prior_default()),
        "belief": belief,
        "balance": balance,
        "already": already,
        "decided": decided,
        "aware": aware,
        "insured": xp.asarray(module.insured, dtype=dtype) if n_dep else zeros[0],
        "base_threshold": xp.asarray(np.stack(thresholds), dtype=dtype)
        if n_dep
        else zeros,
        "payroll": xp.asarray(module.payroll, dtype=dtype) if n_dep else zeros[0],
        "segment_fraction": (
            xp.asarray(module.segment_fraction, dtype=dtype) if n_dep else zeros[0]
        ),
        "hub": xp.asarray(module.hub_weight, dtype=dtype) if n_dep else zeros[0],
        "contacts": contacts,
        "adjacency": (
            _seed_adjacency(scenario, module, xp, n_dep)
            if n_dep and not _gather_path_enabled()
            else None
        ),
        "weights": xp.asarray(np.stack(weights), dtype=dtype),
        "bias": xp.asarray(bias, dtype=dtype),
        "k": xp.asarray(k_vals, dtype=dtype),
        "belief_strength": xp.asarray(belief_strength, dtype=dtype),
        "signal_attention": xp.asarray(signal_attention, dtype=dtype),
        "signal_stock": xp.asarray(signal_stock, dtype=dtype),
        "signal_base": xp.asarray(signal_base, dtype=dtype),
        "rate_relief": xp.asarray(rate_relief, dtype=dtype),
        "amount_scale": xp.asarray(amount_scale, dtype=dtype),
        "mobile_share": xp.asarray(mobile_share, dtype=dtype),
        "awareness_media_rate": xp.asarray(media_rate, dtype=dtype),
        "awareness_wom_rate": xp.full(n_lanes, float(awareness_wom_rate_default()), dtype=dtype),
        "awareness_news_rate": xp.full(n_lanes, float(awareness_news_rate_default()), dtype=dtype),
        "decision_to_request_rate": xp.asarray(request_rate, dtype=dtype),
        "initial_uninsured": (
            xp.asarray(
                np.maximum(
                    0.0,
                    np.asarray(module.balance, dtype=np.float32)
                    - np.asarray(module.insured, dtype=np.float32),
                ),
                dtype=dtype,
            )
            if n_dep
            else zeros[0]
        ),
        "gain": xp.asarray(gains, dtype=dtype),
        "impact": xp.asarray(impacts, dtype=dtype),
        "gap_scale": xp.ones(n_lanes, dtype=dtype),
        "threshold_shift": zeros.copy(),
        "statement_shift": zeros.copy(),
        "deposit_rate": zeros.copy(),
        "directive": zeros.copy(),
        "vip_amount": zeros.copy(),
        "crisis_mask": (
            xp.asarray(
                [1.0 if name in _CRISIS_SEGMENTS else 0.0 for name in module.segments],
                dtype=dtype,
            )
            if n_dep
            else zeros[0]
        ),
        "mgmt_mask": (
            xp.asarray(
                [1.0 if name in _STATEMENT_SEGMENTS else 0.0 for name in module.segments],
                dtype=dtype,
            )
            if n_dep
            else zeros[0]
        ),
        "startup_mask": (
            xp.asarray(
                [1.0 if name == _CEO_RATE_SEGMENT else 0.0 for name in module.segments],
                dtype=dtype,
            )
            if n_dep
            else zeros[0]
        ),
    }


def _concat_lean(states: list[dict[str, Any]], xp: Any) -> dict[str, Any]:
    use_spmm = all(item.get("adjacency") is not None for item in states)
    if len(states) == 1:
        out = dict(states[0])
        out["adjacency"] = [states[0]["adjacency"]] if use_spmm else None
        return out
    out = dict(states[0])
    stack_keys = (
        "belief",
        "balance",
        "already",
        "decided",
        "aware",
        "base_threshold",
        "weights",
        "bias",
        "k",
        "belief_strength",
        "signal_attention",
        "signal_stock",
        "signal_base",
        "rate_relief",
        "amount_scale",
        "mobile_share",
        "awareness_media_rate",
        "awareness_wom_rate",
        "awareness_news_rate",
        "decision_to_request_rate",
        "gain",
        "impact",
        "gap_scale",
        "threshold_shift",
        "statement_shift",
        "deposit_rate",
        "directive",
        "vip_amount",
    )
    # SVB-17: with per-seed adjacency, contacts and hub stay per seed and are never
    # expanded to (lanes, N, D). The gather path (STAMPEDE_CALIB_GATHER_PATH=1) keeps
    # the old expansion for parity tests.
    contact_keys = () if use_spmm else ("hub", "contacts")
    shared_keys = (
        "insured",
        "initial_uninsured",
        "payroll",
        "segment_fraction",
        *contact_keys,
        "crisis_mask",
        "mgmt_mask",
        "startup_mask",
    )
    lane_counts = [int(item["belief"].shape[0]) for item in states]
    for key in stack_keys:
        out[key] = xp.concatenate([item[key] for item in states], axis=0)
    for key in shared_keys:
        out[key] = xp.concatenate(
            [_expand_shared(item[key], n_lanes, xp) for item, n_lanes in zip(states, lane_counts)],
            axis=0,
        )
    out["adjacency"] = [item["adjacency"] for item in states] if use_spmm else None
    out["placebo"] = all(item["placebo"] for item in states)
    out["books"] = [float(item["book"]) for item in states]
    out["strengths"] = [float(item["strength"]) for item in states]
    out["seeds"] = [int(item["seed"]) for item in states]
    out["lane_counts"] = lane_counts
    return out


def _scoring_deposits() -> float:
    from calibration.fit import _PARAM_DEPOSITS, _parse_money

    deposits = _parse_money(get_parameter(_PARAM_DEPOSITS).value)
    return float(deposits) if deposits else 0.0


def _lane_mask(xp: Any, mask: Any, n_lanes: int) -> Any:
    if mask is None:
        return xp.zeros(0, dtype=_f32(xp))
    array = xp.asarray(mask)
    if getattr(array, "ndim", 1) > 1 or int(getattr(array, "size", 0)) == 0:
        return array
    return xp.broadcast_to(array, (n_lanes,) + tuple(array.shape))


def _per_seed_lanes(xp: Any, lane_counts: Sequence[int], values: Sequence[float]) -> Any:
    parts = [
        xp.full((int(count),), float(value), dtype=xp.float64)
        for value, count in zip(values, lane_counts)
    ]
    return xp.concatenate(parts)


def _add_hub_shock(
    pending: Any,
    seed_index: Any,
    hubs: Sequence[int],
    mask: Any,
    amount: Any,
    xp: Any,
) -> Any:
    if pending.shape[1] == 0:
        return pending
    for seed_i, hub in enumerate(hubs):
        hit = (seed_index == seed_i) & mask
        pending[:, hub] = pending[:, hub] + hit * amount
    return pending


def _set_segment_shift(
    statement_shift: Any,
    threshold_shift: Any,
    pending: Any,
    mask: Any,
    logit: float,
    thresh: float,
) -> tuple[Any, Any]:
    hit = (pending[:, None] * mask[None, :]) > 0.0
    keep = ~hit
    return (
        statement_shift * keep + float(logit) * hit,
        threshold_shift * keep + float(thresh) * hit,
    )


def _host_any(flag: Any) -> bool:
    """Copy one reduced boolean onto the host."""
    if isinstance(flag, (bool, np.bool_)):
        return bool(flag)
    if isinstance(flag, np.ndarray):
        return bool(flag.reshape(())) if flag.shape == () else bool(np.any(flag))
    getter = getattr(flag, "get", None)
    if callable(getter):
        value = np.asarray(getter())
        return bool(value.reshape(())) if value.shape == () else bool(np.any(value))
    return bool(flag)


def _pending_host_flags(
    crisis_now: Any,
    mgmt_now: Any,
    xp: Any,
    *,
    read_crisis: bool,
    read_mgmt: bool = True,
) -> tuple[bool, bool]:
    """Host booleans for this tick. Each flag is set at most once per lane."""
    crisis = _host_any(xp.any(crisis_now)) if read_crisis else False
    mgmt = _host_any(xp.any(mgmt_now)) if read_mgmt else False
    return crisis, mgmt


def _step_halt_left(
    xp: Any, halt_left: Any, newly: Any, dt: float, halt_minutes: float
) -> tuple[Any, Any]:
    """Restart or count down a temporary halt. ``live`` when ``halt_left <= 0``."""
    halt_left = xp.where(newly, float(halt_minutes), xp.maximum(halt_left - float(dt), 0.0))
    return halt_left, halt_left <= 0


def simulate_requested_fractions(
    scenario: Scenario,
    schedule: Sequence[tuple[float, float]],
    vectors: NDArray[np.float64] | Sequence[NDArray[np.float64]],
    seeds: Sequence[int],
    *,
    array_module: str = "auto",
    softness: float = 0.0,
    clamp_inputs: bool = False,
    aware0: float | None = None,
) -> NDArray[np.float64]:
    """Stage-1 requested fractions without per-lane engines or events.

    Returns one row per seed×vector lane (seed-major): requests by day-one
    close, requests by next open, and the day-one stock ratio (last print
    before close over the first). Attention, the depositor kernel, and the
    price-feedback loop stay on the array device. Window totals copy to the
    host once. Does not build an ``Engine``, publish events, or step M07,
    M08, M09, or M20. Validation and holdouts keep ``simulate_logs``.
    """
    from engine.gpu_full import (
        _collect_attention_inbox,
        _spread_posted_rumors,
        update_attention_batch,
    )
    from engine.gpu_full_io import (
        apply_price_impact,
        compose_attention,
        price_quote_batch,
        step_news,
    )
    from mocks.support import (
        ASSUMED_HALT_DRAWDOWN,
        ASSUMED_HALT_MINUTES,
        ASSUMED_HALT_WINDOW,
        ASSUMED_HEADLINE_ATTENTION,
        OPENING_PRICE,
        clamp,
        seed_scale,
        uninsured_share,
    )

    rows = _as_vectors(vectors)
    seed_list = [int(item) for item in seeds]
    ticks = _expand_minute_ticks([(float(t), float(dt)) for t, dt in schedule])
    decide_once = withdrawal_decision_once()
    if not rows or not seed_list:
        return np.zeros((0, 3), dtype=np.float64)
    n_out = len(rows) * len(seed_list)
    if not ticks:
        return np.zeros((n_out, 3), dtype=np.float64)

    xp, xp_name = resolve_array_module(array_module)
    lanes_scenarios = [
        scenario if seed == int(scenario.seed) else scenario.model_copy(update={"seed": seed})
        for seed in seed_list
    ]
    _warm_seed_populations(lanes_scenarios)

    packed_seeds = [
        _pack_lean_lanes(item, rows, xp, clamp_inputs=clamp_inputs, aware0=aware0)
        for item in lanes_scenarios
    ]
    state = _concat_lean(packed_seeds, xp)
    n_lanes = int(state["belief"].shape[0])
    n_dep = int(state["n_dep"])
    attns = [_seed_attention_engine(item) for item in lanes_scenarios]
    n_nodes = int(attns[0].attention.shape[0]) if attns else 0
    layers = attns[0]._layers if attns else []
    hubs = [int(engine._hub) for engine in attns]
    half = float(attns[0].params.half_life_minutes) if attns else 1.0
    headline_div = float(attns[0].params.headline_reach_divisor) if attns else 1.0
    thesis_div = float(attns[0].params.thesis_reach_divisor) if attns else 1.0
    statement_exposure = float(attns[0].params.statement_exposure) if attns else 0.0
    halt_exposure = float(attns[0].params.halt_exposure) if attns else 0.0
    uninsured = float(uninsured_share(scenario.preset))
    lane_counts = state.get("lane_counts", [len(rows)] * len(seed_list))
    books = state.get("books", [state["book"]] * len(seed_list))
    strengths = state.get("strengths", [state["strength"]] * len(seed_list))
    book_lanes = xp.concatenate(
        [
            xp.full((count,), float(book), dtype=xp.float64)
            for book, count in zip(books, lane_counts)
        ]
    )
    strength_lanes = xp.concatenate(
        [
            xp.full((count,), float(strength), dtype=xp.float64)
            for strength, count in zip(strengths, lane_counts)
        ]
    )
    seed_index = xp.concatenate(
        [xp.full((count,), i, dtype=xp.int64) for i, count in enumerate(lane_counts)]
    )

    f32 = _f32(xp)
    f64 = _f64(xp)
    close = _day1_close_minute()
    attention = xp.zeros((n_lanes, n_nodes), dtype=f32)
    news = xp.zeros(n_lanes, dtype=f32)
    factor = xp.ones(n_lanes, dtype=f64)
    halt_left = xp.zeros(n_lanes, dtype=f64)
    newly = xp.zeros(n_lanes, dtype=bool)
    first_px = xp.zeros(n_lanes, dtype=f64)
    last_px = xp.zeros(n_lanes, dtype=f64)
    cum_requested = xp.zeros(n_lanes, dtype=f64)
    have_first = xp.zeros(n_lanes, dtype=f64)
    req_day1 = xp.zeros(n_lanes, dtype=f64)
    req_next = xp.zeros(n_lanes, dtype=f64)
    stock_return = xp.zeros(n_lanes, dtype=f32)
    pending_shock = xp.zeros((n_lanes, n_nodes), dtype=f32)
    prev_attn = xp.zeros(n_lanes, dtype=f32)
    prev_price = xp.zeros(n_lanes, dtype=f64)
    headline_sent = xp.zeros(n_lanes, dtype=f32)
    crisis_sent = xp.zeros(n_lanes, dtype=f32)
    mgmt_sent = xp.zeros(n_lanes, dtype=f32)
    ceo_sent = xp.zeros(n_lanes, dtype=f32)
    thesis_sent = xp.zeros(n_lanes, dtype=f32)
    vip_sent = xp.zeros(n_lanes, dtype=f32)
    pending_crisis = xp.zeros(n_lanes, dtype=f32)
    pending_mgmt = xp.zeros(n_lanes, dtype=f32)
    apply_crisis_shift = False
    apply_mgmt_shift = False
    crisis_watch = True
    hist = xp.zeros((n_lanes, int(ASSUMED_HALT_WINDOW)), dtype=f64)
    hist_len = 0
    scratch = make_depositor_scratch(xp, n_lanes, max(n_dep, 1), softness) if n_dep else None
    contacts = state["contacts"]
    degree = int(contacts.shape[-1]) if n_dep and getattr(contacts, "ndim", 1) >= 2 else 0
    adjacency = state.get("adjacency")
    seed_slices = seed_slices_from_counts(lane_counts) if adjacency else None
    # With SpMM there are no (lanes, width, D) temporaries, so chunks can be wider.
    chunk = _chunk_width(n_lanes, n_dep, 0 if adjacency else degree, array_budget_bytes())
    deposits = _scoring_deposits()
    mgmt_mask = _lane_mask(xp, state.get("mgmt_mask"), n_lanes)
    crisis_mask = _lane_mask(xp, state.get("crisis_mask"), n_lanes)
    startup_mask = _lane_mask(xp, state.get("startup_mask"), n_lanes)
    reach_scale = _per_seed_lanes(
        xp, lane_counts, [seed_scale(seed, "reach") for seed in seed_list]
    )
    thesis_amt = _per_seed_lanes(
        xp,
        lane_counts,
        [
            max(1, int(_THESIS_REACH * seed_scale(seed, "thesis-reach"))) / max(thesis_div, 1e-12)
            for seed in seed_list
        ],
    )
    ceo_rate = _per_seed_lanes(
        xp,
        lane_counts,
        [clamp(_CEO_RATE * seed_scale(seed, "dep-rate"), 0.0, _CEO_RATE_CAP) for seed in seed_list],
    )
    vip_params = getattr(attns[0], "params", None) if attns else None
    vip_exposure = float(getattr(vip_params, "vip_exposure", 0.0) or 0.0)

    # SVB-66: M03's _draw_reposts reads each lane's own attention, so every lane keeps its
    # own rumor state (one AttentionEngine per lane, identical at t=0). The lane rows go to
    # the host as ONE copy per tick.
    lane_attns = [
        _seed_attention_engine(lanes_scenarios[index])
        for index, count in enumerate(lane_counts)
        for _ in range(int(count))
    ]

    for t, dt in ticks:
        if n_dep and crisis_mask.size:
            if apply_crisis_shift:
                state["statement_shift"], state["threshold_shift"] = _set_segment_shift(
                    state["statement_shift"],
                    state["threshold_shift"],
                    pending_crisis,
                    crisis_mask,
                    float(state["logit_shift"]),
                    float(state["thresh_shift"]),
                )
            if apply_mgmt_shift:
                state["statement_shift"], state["threshold_shift"] = _set_segment_shift(
                    state["statement_shift"],
                    state["threshold_shift"],
                    pending_mgmt,
                    mgmt_mask,
                    float(state["logit_shift"]),
                    float(state["thresh_shift"]),
                )
        pending_crisis = xp.zeros(n_lanes, dtype=f32)
        pending_mgmt = xp.zeros(n_lanes, dtype=f32)

        thesis_now = (thesis_sent < 1.0) & (prev_attn * uninsured >= _THESIS_VULN)
        vip_due = float(t) > 0.0 and int(t) % _VIP_INTERVAL == 0
        vip_now = (vip_sent < 1.0) & ((prev_attn >= _VIP_ATTENTION) | vip_due)
        if n_nodes:
            pending_shock = _add_hub_shock(
                pending_shock, seed_index, hubs, thesis_now, thesis_amt, xp
            )
            pending_shock = _add_hub_shock(
                pending_shock, seed_index, hubs, vip_now, vip_exposure, xp
            )
        thesis_sent = xp.where(thesis_now, 1.0, thesis_sent)
        vip_sent = xp.where(vip_now, 1.0, vip_sent)

        decay = decay_factor(dt, half) if half > 0.0 else 1.0
        rumor_rows: list[np.ndarray] = []
        lane_host = _to_host(attention) if lane_attns else None
        for lane, engine in enumerate(lane_attns):
            engine._decay = decay
            engine.attention = np.array(lane_host[lane], dtype=np.float64, copy=True)
            _shock, posted = _collect_attention_inbox(engine)
            rumor_exposure, _mutated = _spread_posted_rumors(engine, t, posted)
            rumor_rows.append(np.asarray(rumor_exposure, dtype=np.float32))
        rumor_dev = xp.asarray(np.stack(rumor_rows), dtype=f32)
        exposure = pending_shock + rumor_dev
        pending_shock = xp.zeros_like(pending_shock)
        if n_nodes:
            attention = update_attention_batch(xp, attention, float(decay), layers, exposure)
            rumor_mean = attention.mean(axis=1)
        else:
            rumor_mean = xp.zeros(n_lanes, dtype=f32)
        composed = compose_attention(xp, rumor_mean, news)
        raw_level = composed
        attention_level = xp.clip(raw_level, 0.0, 1.0)

        ceo_now = (ceo_sent < 1.0) & (prev_attn >= _CEO_ATTENTION)
        if n_dep and startup_mask.size:
            apply_rate = ceo_now[:, None] * startup_mask
            state["deposit_rate"] = xp.where(
                apply_rate > 0.0, ceo_rate[:, None], state["deposit_rate"]
            )
        ceo_sent = xp.where(ceo_now, 1.0, ceo_sent)

        if not state["placebo"] and n_dep > 0:
            amounts = step_depositor_batch(
                xp,
                belief=state["belief"],
                balance=state["balance"],
                already=state["already"],
                insured=state["insured"],
                base_threshold=state["base_threshold"],
                threshold_shift=state["threshold_shift"],
                statement_shift=state["statement_shift"],
                deposit_rate=state["deposit_rate"],
                directive=state["directive"],
                payroll=state["payroll"],
                segment_fraction=state["segment_fraction"],
                vip_amount=state["vip_amount"],
                hub=state["hub"],
                contacts=state["contacts"],
                attention_level=attention_level,
                stock_return=stock_return,
                gap_scale=state["gap_scale"],
                amount_scale=state["amount_scale"],
                mobile_share=state["mobile_share"],
                initial_uninsured=state["initial_uninsured"],
                decided=state["decided"],
                decide_once=decide_once,
                news=news,
                aware=state["aware"],
                awareness_media_rate=state["awareness_media_rate"],
                awareness_wom_rate=state["awareness_wom_rate"],
                awareness_news_rate=state["awareness_news_rate"],
                awareness_w1_prior=float(
                    state.get("awareness_w1_prior", awareness_rate_defaults()[3])
                ),
                decision_to_request_rate=state.get("decision_to_request_rate"),
                weights=state["weights"],
                bias=state["bias"],
                k=state["k"],
                belief_strength=_adjustment_strength(xp, state["belief_strength"], dt),
                signal_attention=state["signal_attention"],
                signal_stock=state["signal_stock"],
                signal_base=state["signal_base"],
                rate_relief=state["rate_relief"],
                minimum_amount=state["minimum_amount"],
                chunk=chunk,
                softness=softness,
                scratch=scratch,
                dt=dt,
                adjacency=adjacency,
                seed_slices=seed_slices,
            )
            requested = amounts.sum(axis=1, dtype=xp.float64)
        else:
            requested = xp.zeros(n_lanes, dtype=f64)
        if t < close:
            req_day1 = req_day1 + requested
        if t < _NEXT_OPEN:
            req_next = req_next + requested

        share = xp.where(book_lanes > 0.0, requested / book_lanes, 0.0)
        cum_requested = cum_requested + requested
        cum_share = xp.where(book_lanes > 0.0, cum_requested / book_lanes, 0.0)
        quoted = price_quote_batch(
            xp,
            t,
            strength_lanes,
            raw_level,
            scripted=calibration_uses_scripted_drawdown(),
        )
        halt_left, live = _step_halt_left(xp, halt_left, newly, dt, ASSUMED_HALT_MINUTES)
        current = quoted * factor
        impacted = apply_price_impact(
            xp,
            current,
            cum_share,
            state["impact"],
            float(dt),
            open_price=float(OPENING_PRICE),
            attention_level=attention_level,
            noise=0.0,
        )
        price = xp.where(live, impacted, xp.where(prev_price > 0.0, prev_price, current))
        factor = xp.where(quoted > 0.0, price / xp.maximum(quoted, 1e-30), factor)
        news = step_news(xp, news, share, state["gain"], float(dt))

        if t < close:
            first_px = xp.where(have_first > 0.0, first_px, price)
            have_first = xp.ones(n_lanes, dtype=f64)
            last_px = price
        stock_return = xp.where(first_px > 0.0, price / first_px - 1.0, 0.0)

        if hist_len < int(ASSUMED_HALT_WINDOW):
            hist[:, hist_len] = price
            hist_len += 1
        else:
            hist = xp.concatenate([hist[:, 1:], price[:, None]], axis=1)
        if hist_len >= 2:
            start = hist[:, 0]
            drawdown = xp.where(start > 0.0, price / start - 1.0, 0.0)
            newly = live & (drawdown <= -float(ASSUMED_HALT_DRAWDOWN))
            if n_nodes:
                pending_shock = _add_hub_shock(
                    pending_shock, seed_index, hubs, newly, halt_exposure, xp
                )
        else:
            newly = xp.zeros(n_lanes, dtype=bool)

        line = float(ASSUMED_HEADLINE_ATTENTION)
        crossed_raw = (headline_sent < 1.0) & (prev_attn < line) & (rumor_mean >= line)
        crossed_composed = (
            (headline_sent < 1.0)
            & (~crossed_raw)
            & (rumor_mean < line)
            & (composed >= line)
        )
        crossed = crossed_raw | crossed_composed
        headline_level = xp.where(crossed_raw, rumor_mean, composed)
        if n_nodes:
            reach = xp.maximum(1.0, xp.floor(_HEADLINE_REACH * headline_level * reach_scale))
            pending_shock = _add_hub_shock(
                pending_shock,
                seed_index,
                hubs,
                crossed,
                reach / max(headline_div, 1e-12),
                xp,
            )
        headline_sent = xp.where(crossed, 1.0, headline_sent)

        crisis_now = (crisis_sent < 1.0) & (have_first > 0.0)
        sliding_now = (
            (mgmt_sent < 1.0)
            & (ceo_now < 1.0)
            & (have_first > 0.0)
            & (prev_price > 0.0)
            & (prev_price <= first_px * _STATEMENT_DRAWDOWN)
        )
        mgmt_now = ceo_now | sliding_now
        if n_nodes:
            pending_shock = _add_hub_shock(
                pending_shock, seed_index, hubs, crisis_now | mgmt_now, statement_exposure, xp
            )
        pending_crisis = xp.where(crisis_now, 1.0, pending_crisis)
        pending_mgmt = xp.where(mgmt_now, 1.0, pending_mgmt)
        if n_dep and crisis_mask.size:
            read_crisis = crisis_watch and t < close
            apply_crisis_shift, apply_mgmt_shift = _pending_host_flags(
                crisis_now,
                mgmt_now,
                xp,
                read_crisis=read_crisis,
            )
            if read_crisis:
                crisis_watch = False
        else:
            apply_crisis_shift = False
            apply_mgmt_shift = False
        crisis_sent = xp.where(crisis_now, 1.0, crisis_sent)
        mgmt_sent = xp.where(mgmt_now, 1.0, mgmt_sent)
        prev_attn = raw_level
        prev_price = price

    denom = deposits if deposits > 0.0 else 1.0
    stock = xp.where((have_first > 0.0) & (first_px > 0.0), last_px / first_px, 0.0)
    out = xp.stack([req_day1 / denom, req_next / denom, stock], axis=1)
    host = _to_host(out)
    _release_device(xp, xp_name)
    return np.asarray(host, dtype=np.float64)
