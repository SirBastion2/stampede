"""PA-B: the multi-seed batch table. Every preset x response x seed, run offline, written to one file.

The grid is lost-bonds and frozen-branches, each with no response and each of the three countermeasures,
plus stadium (no rumor, no response), over seeds 1-10 by default: 90 runs of the 1441-minute live window.
Each run keeps a series every 30 minutes (for PA-A's with/without comparison) and its outcome minutes;
the summary gives min, mean and max per preset and response, as proof the results are not hard-coded.
Start districts are not covered: not built.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

PRESETS = ("lost-bonds", "frozen-branches")
RESPONSE_OPTIONS: tuple[str | None, ...] = (None, "guarantee", "credit-line", "withdrawal-limit")
BASELINES = ("stadium",)
DEFAULT_SEEDS = tuple(range(1, 11))
SAMPLE_EVERY = 30
CHECKPOINTS = (240, 390, 720, 1080, 1440, 1441)
DEFAULT_FILE = "data/live_batch.json"
# UI-3: when each response comes, for the bond rumor (the 2023 replay): 11:00 AM, 1:30 PM, the 4:00 PM close, 1:00 AM
TIMING_MINUTES = (90, 240, 390, 930)
DEFAULT_RESPONSE_MINUTE = 240


def batch_file() -> str:
    return os.environ.get("STAMPEDE_BATCH_FILE", DEFAULT_FILE)


def grid(seeds: tuple[int, ...] = DEFAULT_SEEDS) -> list[tuple[str, str | None, int | None, int]]:
    """(preset, response, response minute, seed). The bond rumor gets every response at every time in
    TIMING_MINUTES; frozen-branches gets each at 1:30 PM; stadium gets none."""
    cells: list[tuple[str, str | None, int | None, int]] = []
    for preset in PRESETS:
        for response in RESPONSE_OPTIONS:
            times = (None,) if response is None else (TIMING_MINUTES if preset == "lost-bonds" else (DEFAULT_RESPONSE_MINUTE,))
            cells += [(preset, response, minute, s) for minute in times for s in seeds]
    return cells + [(p, None, None, s) for p in BASELINES for s in seeds]


def run_one(preset: str, response: str | None, seed: int, minutes: int | None = None,
            response_minute: int | None = None) -> dict[str, Any]:
    """One live real-mode run, stepped as the server's producer would (no buffering, no wall clock)."""
    os.environ["STAMPEDE_MODE"] = "real"
    os.environ.setdefault("STAMPEDE_OFFLINE", "1")
    from contracts.events import BankClosed, WithdrawalRequested
    from engine.bank.ledger import registry_deposits_total
    from ui.live.pacing import crisis_sim_minutes
    from ui.live.session import LiveSession

    t0 = time.perf_counter()
    session = LiveSession(seed=seed)
    if response is not None and response_minute is None:
        response_minute = DEFAULT_RESPONSE_MINUTE
    session.load_preset(preset, response, response_minute)
    end = int(minutes if minutes is not None else crisis_sim_minutes())
    deposits = registry_deposits_total()
    log = session.engine.bus.log
    cursor, requested, closure, cash_out = 0, 0.0, None, None
    series: list[dict[str, Any]] = []
    for minute in range(1, end + 1):
        session.step_once()
        for event in log[cursor:]:
            if isinstance(event, WithdrawalRequested):
                requested += float(event.amount)
            elif isinstance(event, BankClosed) and closure is None:
                closure = float(event.t)
        cursor = len(log)
        crowd = session.engine["M29"]
        if cash_out is None and crowd.cash_gauge <= 0.0:
            cash_out = float(minute)
        if minute % SAMPLE_EVERY == 0 or minute in CHECKPOINTS or minute == end:
            prices = session.market_frame().prices
            series.append({"minute": minute, "requested": round(requested / deposits, 6),
                           "price": round(float(prices[-1][1]), 4) if prices else None,
                           "cash": round(float(crowd.cash_gauge), 6), "withdrew": int(crowd.withdrew),
                           "wavering": int(crowd.wavering), "calm": int(crowd.calm)})
    at = {str(row["minute"]): row["requested"] for row in series if row["minute"] in CHECKPOINTS}
    return {"preset": preset, "response": response, "response_minute": response_minute, "seed": seed, "minutes": end,
            "label": session.run_label(),
            "borrowed_usd": session._ui_borrowed, "emergency_usd": session._ui_emergency,  # UI-2
            "response_enacted_minute": session.response_minutes.get(response) if response else None,  # None if the run ended first
            "halt_minute": session.halt_minute, "closure_minute": closure, "cash_out_minute": cash_out,
            "requested_at": at, "series": series, "seconds": round(time.perf_counter() - t0, 1)}


def _spread(values: list[float]) -> dict[str, float] | None:
    return None if not values else {"min": min(values), "mean": round(sum(values) / len(values), 6), "max": max(values)}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str | None, int | None], list[dict[str, Any]]] = {}
    for row in rows:
        minute = row.get("response_minute", DEFAULT_RESPONSE_MINUTE if row["response"] else None)
        groups.setdefault((row["preset"], row["response"], minute), []).append(row)
    out = []
    for (preset, response, minute), part in groups.items():
        halts = [r["halt_minute"] for r in part if r["halt_minute"] is not None]
        out.append({
            "preset": preset, "response": response, "response_minute": minute, "runs": len(part),
            "label": part[0]["label"]["kind"],
            "requested_390": _spread([r["requested_at"]["390"] for r in part if "390" in r["requested_at"]]),
            "requested_1440": _spread([r["requested_at"]["1440"] for r in part if "1440" in r["requested_at"]]),
            "halted_runs": len(halts), "halt_minute": _spread(halts),
            "closed_runs": sum(1 for r in part if r["closure_minute"] is not None),
            "cash_out_runs": sum(1 for r in part if r["cash_out_minute"] is not None),
            "cash_out_minute": _spread([r["cash_out_minute"] for r in part if r["cash_out_minute"] is not None]),
            "emergency_usd": _spread([r["emergency_usd"] for r in part if "emergency_usd" in r]),  # UI-2
            "borrowed_usd": _spread([r["borrowed_usd"] for r in part if "borrowed_usd" in r]),
        })
    return out


def _cell(args: tuple[str, str | None, int | None, int, int | None]) -> dict[str, Any]:
    preset, response, response_minute, seed, minutes = args
    return run_one(preset, response, seed, minutes, response_minute)


def _cells_batched(cells: list) -> list[dict[str, Any]]:
    from ui.live.gpu_batch import run_batch_cells_batched

    return run_batch_cells_batched(cells, array_module=os.environ.get("STAMPEDE_SEARCH_GPU_XP", "auto"))


def run_grid(seeds: tuple[int, ...] = DEFAULT_SEEDS, workers: int = 1, minutes: int | None = None,
             gpu: bool | None = None) -> dict[str, Any]:
    """Every cell, in grid order. ``workers`` > 1 uses separate processes (spawn: fresh, like the server).

    GPU-1: ``gpu`` batches each worker's cells with the depositor model on the GPU. CPY-1: ``None`` (the
    default) means on when the GPU paths run, as the search (``search.gpu_search_enabled``).
    """
    if gpu is None:
        from ui.live.search import gpu_search_enabled

        gpu = gpu_search_enabled()
    cells = [(p, r, m, s, minutes) for p, r, m, s in grid(tuple(seeds))]
    t0 = time.perf_counter()
    if gpu:
        from ui.live.gpu_batch import default_group_size

        size = max(1, min(default_group_size(), -(-len(cells) // max(1, workers))))
        order = sorted(range(len(cells)), key=lambda i: int(cells[i][3]))   # same-seed lanes share static arrays
        chunks = [[cells[i] for i in order[j:j + size]] for j in range(0, len(order), size)]
        if workers <= 1:
            parts = [_cells_batched(c) for c in chunks]
        else:
            import multiprocessing as mp

            with mp.get_context("spawn").Pool(processes=workers) as pool:
                parts = pool.map(_cells_batched, chunks, chunksize=1)
        flat = [row for part in parts for row in part]
        rows = [None] * len(cells)
        for i, row in zip(order, flat):
            rows[i] = row
    elif workers <= 1:
        rows = [_cell(c) for c in cells]
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(processes=workers) as pool:
            rows = pool.map(_cell, cells, chunksize=1)
    return {"kind": "stampede live batch", "seeds": list(seeds), "minutes": rows[0]["minutes"] if rows else minutes,
            "presets": list(PRESETS), "responses": [r for r in RESPONSE_OPTIONS if r], "baselines": list(BASELINES),
            "timing_minutes": list(TIMING_MINUTES),
            "workers": workers, "gpu_batch": bool(gpu), "wall_seconds": round(time.perf_counter() - t0, 1),
            "summary": summarize(rows), "rows": rows}


def write_batch(data: dict[str, Any], path: str | None = None) -> str:
    target = path or batch_file()
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1)
    return target


def load_batch(path: str | None = None) -> dict[str, Any] | None:
    target = path or batch_file()
    if not os.path.exists(target):
        return None
    with open(target, encoding="utf-8") as handle:
        return json.load(handle)
