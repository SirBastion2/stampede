"""GPU-1 (sandbox test): the offline runs (the response search and the batch table), batched, with the depositor
model on the GPU.

Each lane is one run of the LIVE engine: a real ``LiveSession`` with its preset, triggers, responses, opening
book, live flags, fitted feedback, paced price and closure hold, built exactly as ``search.run_plan`` and
``batch.run_one`` build it. What changes is M06, the depositors (about two-thirds of every step on the CPU):
instead of each lane's ``DecisionModule.step``, one ``step_depositor_batch`` call (the calibration GPU kernel,
``engine/gpu_sim_kernels.py``) updates every lane's 100,000 depositors at once, on CuPy when a device is visible,
NumPy otherwise. Every other module still steps per lane on the CPU, in the engine's order, so a run's events,
triggers and outcome come from the same code as before.

M06's events are still applied per lane (``_consume_m06``, then the cure rumors, as ``DecisionModule.step``
does): attention with the credibility ceiling, prices it ignores, the guarantee and the withdrawal limit,
statements. The kernel reads each lane's own scalars. Known differences from the CPU path, measured by
``scripts/gpu_parity.py``:
  - the kernel is float32 (the CPU path is float64);
  - requests are emitted per channel (4 events a minute) instead of per segment and channel (M07's limits and
    M08 are per channel and bank-wide, so totals agree);
  - M06's own arrays (who withdrew, worry) are copied back from the kernel only on the minutes a recorder
    samples, before the crowd reads them; worry is recovered from the kernel's soft worried probability.
Each lane is a whole live engine in host memory, about 60 MB, so lanes per batch are capped by host RAM too
(``STAMPEDE_GPU_LANES``, default 16). Opt-in only (``--gpu``). Not for the fitted replay or any shipped number
until a parity run on the GPU passes.
"""

from __future__ import annotations

import gc
import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np

_DT = 1.0
HOST_LANE_CAP = 16   # one live engine per lane: about 60 MB, up to ~120 MB for a lane on a new seed


def default_group_size() -> int:
    """Lanes per batch: the GPU byte budget (``M27 GPU array budget``) and the host cap, whichever is smaller."""
    from engine.agents.population import _depositor_count_for_tier
    from engine.gpu_sim import _group_size, array_budget_bytes

    override = os.environ.get("STAMPEDE_GPU_LANES", "").strip()
    if override:
        return max(1, int(override))
    return max(1, min(HOST_LANE_CAP, _group_size(10_000, _depositor_count_for_tier(0), array_budget_bytes())))


def step_sessions_batched(sessions: list, ends: list[int], *, xp: Any, xp_name: str,
                          on_minute: Callable[[int, int], None], samples: Callable[[int, int], bool],
                          timings: dict | None = None) -> None:
    """Advance live sessions together, minute by minute, with M06 batched; ``on_minute(k, minute)`` after each.

    ``samples(k, minute)`` says when lane ``k`` reads the crowd that minute: M06's arrays are refreshed from the
    kernel first. Every lane runs to the longest end; ``on_minute`` is called only up to each lane's own end.
    """
    from contracts.events import DepositRateChanged, DirectiveIssued, RumorPosted, StatementIssued, VIPDecision
    from engine.agents.decide import awareness_rate_defaults, withdrawal_decision_once
    from engine.fitted_feedback import step_module
    from engine.gpu_sim import (_channel_totals, _chunk_width, _consume_m06, _emit_withdrawals, _release_device,
                                _scalars, _stack_dynamic, _static_state_for_group, array_budget_bytes,
                                clear_static_pack_cache)
    from engine.gpu_sim_kernels import make_depositor_scratch, step_depositor_batch

    if not all(s.follows_engine for s in sessions):
        raise RuntimeError("GPU-1 needs the live real-mode engine (STAMPEDE_MODE=real)")
    engines = [s.engine for s in sessions]
    modules = [e["M06"] for e in engines]
    order = tuple(engines[0].order)
    if any(tuple(e.order) != order for e in engines):
        raise RuntimeError("GPU-1: every lane must run the same module order")
    split = order.index("M06")
    pre, post = order[:split], order[split + 1:]
    softness = float(getattr(modules[0], "softness", 0.0))
    if any(float(getattr(m, "softness", 0.0)) != softness for m in modules):
        raise RuntimeError("GPU-1: every lane must share M06's softness")
    decide_once = withdrawal_decision_once()
    n, end = len(sessions), max(ends)
    dyn_types = (StatementIssued, DepositRateChanged, DirectiveIssued, VIPDecision, RumorPosted)
    dyn_cursor = [0] * n
    clear_static_pack_cache()
    state = dynamic = scratch = None
    chunk = 1
    t1, kernel_s = time.perf_counter(), 0.0
    for minute in range(1, end + 1):
        dirty = False
        for s in sessions:  # LiveSession.step_once, up to M06
            s._check_triggers()
            s._publish_due_posts()
            for mid in pre:
                step_module(s.engine, mid, s.tick)
        for k, (s, module) in enumerate(zip(sessions, modules)):
            _consume_m06(module)  # this minute's attention, prices, policies and statements, as M06.step
            if module.cure_worry:
                module._apply_cures()
            log = s.engine.bus.log
            if any(isinstance(e, dyn_types) for e in log[dyn_cursor[k]:]):
                dirty = True  # statements, rates, directives, VIPs or a cure moved a per-depositor shift
            dyn_cursor[k] = len(log)
        if state is None:
            state = _static_state_for_group(modules, [m.scenario for m in modules], xp, xp_name)
            contacts = state["contacts"]
            degree = int(contacts.shape[-1]) if getattr(contacts, "ndim", 1) >= 2 else 0
            n_dep = int(state["belief"].shape[1])
            chunk = _chunk_width(n, n_dep, degree, array_budget_bytes())
            dynamic = _stack_dynamic(modules, xp)
            scratch = make_depositor_scratch(xp, n, n_dep, softness)
        elif dirty:
            dynamic = _stack_dynamic(modules, xp)
        scalars = _scalars(modules, xp)
        k0 = time.perf_counter()
        amounts = step_depositor_batch(
            xp, belief=state["belief"], balance=state["balance"], already=state["already"],
            insured=state["insured"], base_threshold=state["base_threshold"],
            threshold_shift=dynamic["threshold_shift"], statement_shift=dynamic["statement_shift"],
            deposit_rate=dynamic["deposit_rate"], directive=dynamic["directive"], payroll=state["payroll"],
            segment_fraction=state["segment_fraction"], vip_amount=dynamic["vip_amount"], hub=state["hub"],
            contacts=state["contacts"], attention_level=scalars["attention_level"],
            stock_return=scalars["stock_return"], gap_scale=scalars["gap_scale"],
            amount_scale=scalars["amount_scale"], mobile_share=scalars["mobile_share"],
            initial_uninsured=state["initial_uninsured"], decided=state["decided"], decide_once=decide_once,
            news=scalars["news"], aware=state["aware"], awareness_media_rate=state["awareness_media_rate"],
            awareness_wom_rate=state["awareness_wom_rate"], awareness_news_rate=state["awareness_news_rate"],
            awareness_w1_prior=float(state.get("awareness_w1_prior", awareness_rate_defaults()[3])),
            decision_to_request_rate=state.get("decision_to_request_rate"), weights=state["weights"],
            bias=state["bias"], k=state["k"], belief_strength=state["belief_strength"],
            signal_attention=state["signal_attention"], signal_stock=state["signal_stock"],
            signal_base=state["signal_base"], rate_relief=state["rate_relief"],
            minimum_amount=state["minimum_amount"], chunk=chunk, softness=softness, scratch=scratch, dt=_DT,
        )
        dynamic["vip_amount"] = xp.zeros_like(dynamic["vip_amount"])
        for module in modules:
            module.vip_amount.fill(0.0)
        totals = _channel_totals(xp, amounts, state["channel_index"], scratch)
        reading = [k for k in range(n) if minute <= ends[k] and samples(k, minute)]
        if reading:
            _write_back(modules, reading, state, dynamic, scratch, softness)
        kernel_s += time.perf_counter() - k0
        for k, (s, module) in enumerate(zip(sessions, modules)):
            if minute <= ends[k]:
                _emit_withdrawals(s.engine, module, totals[k], s.tick)
            for mid in post:
                step_module(s.engine, mid, s.tick)
            s.tick += 1.0
            s._sync_engine_crowd()
            s._ui_scan()
            if minute <= ends[k]:
                on_minute(k, minute)
    if timings is not None:   # CPY-1: the 3060 has 12 GB for every worker; record this batch's peak use
        from engine.xp import vram_mb

        v = vram_mb()
        if v is not None:
            timings["vram_peak_mb"] = max(timings.get("vram_peak_mb", 0.0), v["pool_held"])
            timings["device_free_min_mb"] = min(timings.get("device_free_min_mb", v["device_free"]), v["device_free"])
    _release_device(xp, xp_name)
    if timings is not None:
        timings["step_s"] = timings.get("step_s", 0.0) + (time.perf_counter() - t1)
        timings["kernel_s"] = timings.get("kernel_s", 0.0) + kernel_s
        timings["lane_minutes"] = timings.get("lane_minutes", 0) + sum(ends)
        timings["backend"] = xp_name


def _write_back(modules: list, lanes: list[int], state: dict, dynamic: dict, scratch: dict, softness: float) -> None:
    """Copy the kernel's depositor state onto these lanes' M06 arrays, which the crowd and the vault read."""
    decided = _host(state["decided"])
    worry = _worry_from_kernel(state, dynamic, scratch, softness)
    for k in lanes:
        module = modules[k]
        module.decided = decided[k].astype(np.float64)
        module.already_withdrew = decided[k] >= 0.5   # the soft decide-once rule: withdrew once decided >= 0.5
        module.worry = worry[k]


def _worry_from_kernel(state: dict, dynamic: dict, scratch: dict, softness: float) -> np.ndarray:
    """Each depositor's worry, recovered from the kernel's soft worried probability.

    In soft mode the kernel keeps p = sigmoid((worry - threshold) / softness) with threshold = base + shift,
    so worry = threshold + softness * logit(p): exact up to float32 and the clip at p = 0 or 1. With softness 0
    the kernel keeps a bool, so worry is +/- infinity around the threshold.
    """
    p = _host(scratch["worried"]).astype(np.float64)
    threshold = _host(state["base_threshold"]).astype(np.float64) + _host(dynamic["threshold_shift"]).astype(np.float64)
    if softness <= 0.0:
        return np.where(p > 0.5, np.inf, -np.inf)
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return threshold + softness * np.log(p / (1.0 - p))


def _host(array: Any) -> np.ndarray:
    getter = getattr(array, "get", None)
    return np.asarray(getter() if callable(getter) else array)


def _xp(array_module: str) -> tuple[Any, str]:
    """The kernel's array module: ``auto`` follows the CPY-1 switch (CuPy unless ``STAMPEDE_BACKEND=cpu``)."""
    from engine.gpu_sim import resolve_array_module
    from engine.xp import backend_preference, require_gpu

    if array_module == "auto" and backend_preference() == "cpu":
        array_module = "numpy"
    xp, name = resolve_array_module(array_module)
    if name != "gpu":
        require_gpu("batched runs")
    return xp, name


def _grouped(cells: list, seed_of: Callable[[Any], int], group: int | None):
    """Groups of cell indices, seed-contiguous (the kernel packs static arrays once per seed)."""
    size = group or default_group_size()
    order = sorted(range(len(cells)), key=lambda i: seed_of(cells[i]))
    return [order[i:i + size] for i in range(0, len(order), size)]


# ---- the response search: search.run_plan, batched -------------------------------------------------------------

def _plan_cell(args) -> tuple[list, int, int | None, str, float | None]:
    from ui.live.search import PRESET

    plan, seed, minutes = args[:3]
    return list(plan), int(seed), minutes, (args[3] if len(args) > 3 else PRESET), (args[4] if len(args) > 4 else None)


def run_plans_batched(cells: list, array_module: str = "auto", group: int | None = None,
                      timings: dict | None = None) -> list[dict[str, Any]]:
    """``search.run_plan`` for many cells at once. Rows come back in the order of ``cells``."""
    xp, xp_name = _xp(array_module)
    rows: dict[int, dict[str, Any]] = {}
    for chunk in _grouped(cells, lambda c: _plan_cell(c)[1], group):
        for i, row in zip(chunk, _plans_group([cells[i] for i in chunk], xp, xp_name, timings)):
            rows[i] = row
        gc.collect()   # a live engine is full of reference cycles: free each batch's lanes before the next
    return [rows[i] for i in range(len(cells))]


def _plans_group(cells: list, xp: Any, xp_name: str, timings: dict | None) -> list[dict[str, Any]]:
    os.environ["STAMPEDE_MODE"] = "real"
    os.environ.setdefault("STAMPEDE_OFFLINE", "1")
    from contracts.events import BankClosed, WithdrawalRequested
    from engine.bank.ledger import registry_deposits_total
    from ui.live.pacing import crisis_sim_minutes
    from ui.live.search import PRESET, plan_id
    from ui.live.session import LiveSession

    t0 = time.perf_counter()
    specs = [_plan_cell(c) for c in cells]
    sessions = []
    for plan, seed, _, preset, cred in specs:
        s = LiveSession(seed=seed)
        s.load_preset(preset, plan=list(plan), credibility=cred)
        sessions.append(s)
    setup = time.perf_counter() - t0
    ends = [int(m if m is not None else crisis_sim_minutes()) for _, _, m, _, _ in specs]
    deposits = registry_deposits_total()
    n = len(sessions)
    requested, closure, cash_out = [0.0] * n, [None] * n, [None] * n
    series: list[list] = [[] for _ in range(n)]
    cursors = [0] * n

    def samples(k: int, minute: int) -> bool:   # as search.run_plan: every 10 minutes and the last
        return minute % 10 == 0 or minute == ends[k]

    def on_minute(k: int, minute: int) -> None:
        s = sessions[k]
        log = s.engine.bus.log
        for event in log[cursors[k]:]:
            if isinstance(event, WithdrawalRequested):
                requested[k] += float(event.amount)
            elif isinstance(event, BankClosed) and closure[k] is None:
                closure[k] = float(event.t)
        cursors[k] = len(log)
        cash = float(s.engine["M29"].cash_gauge)
        if cash_out[k] is None and cash <= 0.0:
            cash_out[k] = float(minute)
        if samples(k, minute):
            crowd = s.engine["M29"]
            heads = max(1, int(crowd.withdrew) + int(crowd.wavering) + int(crowd.calm))
            series[k].append([minute, round(requested[k] / deposits, 5), round(cash, 4),
                              round(int(crowd.withdrew) / heads, 4), round(int(crowd.wavering) / heads, 4)])

    step_sessions_batched(sessions, ends, xp=xp, xp_name=xp_name, on_minute=on_minute, samples=samples,
                          timings=timings)
    if timings is not None:
        timings["setup_s"] = timings.get("setup_s", 0.0) + setup
    out = []
    for k, (plan, seed, _, preset, cred) in enumerate(specs):
        tag = plan_id(plan) if preset == PRESET else f"baseline:{preset}"
        if cred is not None:
            tag = f"strength:{cred:g}"
        s = sessions[k]
        out.append({"plan": tag, "seed": seed, "minutes": ends[k], "credibility": cred, "open": closure[k] is None,
                    "closure_minute": closure[k], "fired": dict(s.trigger_minutes),
                    "fired_requested": dict(s.trigger_requested), "cash_out_minute": cash_out[k],
                    "requested": requested[k] / deposits, "requested_usd": requested[k],
                    "emergency_usd": s._ui_emergency, "series": series[k], "seconds": 0.0,
                    "backend": f"gpu-batch:{xp_name}"})
    return out


# ---- the batch table: batch.run_one, batched -------------------------------------------------------------------

def run_batch_cells_batched(cells: list, array_module: str = "auto", group: int | None = None,
                            timings: dict | None = None) -> list[dict[str, Any]]:
    """``batch.run_one`` for cells ``(preset, response, response_minute, seed, minutes)``, in order."""
    xp, xp_name = _xp(array_module)
    rows: dict[int, dict[str, Any]] = {}
    for chunk in _grouped(cells, lambda c: int(c[3]), group):
        for i, row in zip(chunk, _batch_group([cells[i] for i in chunk], xp, xp_name, timings)):
            rows[i] = row
        gc.collect()
    return [rows[i] for i in range(len(cells))]


def _batch_group(cells: list, xp: Any, xp_name: str, timings: dict | None) -> list[dict[str, Any]]:
    os.environ["STAMPEDE_MODE"] = "real"
    os.environ.setdefault("STAMPEDE_OFFLINE", "1")
    from contracts.events import BankClosed, WithdrawalRequested
    from engine.bank.ledger import registry_deposits_total
    from ui.live.batch import CHECKPOINTS, DEFAULT_RESPONSE_MINUTE, SAMPLE_EVERY
    from ui.live.pacing import crisis_sim_minutes
    from ui.live.session import LiveSession

    sessions, specs = [], []
    for preset, response, response_minute, seed, minutes in cells:
        if response is not None and response_minute is None:
            response_minute = DEFAULT_RESPONSE_MINUTE
        s = LiveSession(seed=int(seed))
        s.load_preset(preset, response, response_minute)
        sessions.append(s)
        specs.append((preset, response, response_minute, int(seed),
                      int(minutes if minutes is not None else crisis_sim_minutes())))
    ends = [sp[4] for sp in specs]
    deposits = registry_deposits_total()
    n = len(sessions)
    requested, closure, cash_out = [0.0] * n, [None] * n, [None] * n
    series: list[list] = [[] for _ in range(n)]
    cursors = [0] * n

    def samples(k: int, minute: int) -> bool:   # as batch.run_one
        return minute % SAMPLE_EVERY == 0 or minute in CHECKPOINTS or minute == ends[k]

    def on_minute(k: int, minute: int) -> None:
        s = sessions[k]
        log = s.engine.bus.log
        for event in log[cursors[k]:]:
            if isinstance(event, WithdrawalRequested):
                requested[k] += float(event.amount)
            elif isinstance(event, BankClosed) and closure[k] is None:
                closure[k] = float(event.t)
        cursors[k] = len(log)
        crowd = s.engine["M29"]
        if cash_out[k] is None and crowd.cash_gauge <= 0.0:
            cash_out[k] = float(minute)
        if samples(k, minute):
            prices = s.market_frame().prices
            series[k].append({"minute": minute, "requested": round(requested[k] / deposits, 6),
                              "price": round(float(prices[-1][1]), 4) if prices else None,
                              "cash": round(float(crowd.cash_gauge), 6), "withdrew": int(crowd.withdrew),
                              "wavering": int(crowd.wavering), "calm": int(crowd.calm)})

    step_sessions_batched(sessions, ends, xp=xp, xp_name=xp_name, on_minute=on_minute, samples=samples,
                          timings=timings)
    out = []
    for k, (preset, response, response_minute, seed, end) in enumerate(specs):
        s = sessions[k]
        at = {str(row["minute"]): row["requested"] for row in series[k] if row["minute"] in CHECKPOINTS}
        out.append({"preset": preset, "response": response, "response_minute": response_minute, "seed": seed,
                    "minutes": end, "label": s.run_label(), "borrowed_usd": s._ui_borrowed,
                    "emergency_usd": s._ui_emergency,
                    "response_enacted_minute": s.response_minutes.get(response) if response else None,
                    "halt_minute": s.halt_minute, "closure_minute": closure[k], "cash_out_minute": cash_out[k],
                    "requested_at": at, "series": series[k], "seconds": 0.0, "backend": f"gpu-batch:{xp_name}"})
    return out
