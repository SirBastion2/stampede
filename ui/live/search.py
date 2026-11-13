"""UI-3: the engine searches response plans for the bond rumor (the 2023 replay) and finds decision deadlines.

Plans are one or more of the three responses, each at its own minute. Every plan is scored on what the engine
measures, over the same seeds (never the luckiest one):
    1. survival: the share of seeds in which the bank is still open at the next morning's open
    2. deposits lost: the mean dollars requested by the next open (lower is better)
    3. assumed public cost: M22's cost model, a tie-break only, labelled as assumed
Successive halving: every plan on 1 seed, the best third on 3 seeds, the best third of those on 10; the top 3
are then confirmed on 10 held-out seeds. The tournament screen replays these rounds from the saved runs.

Decision deadline: for each response alone, the latest minute at which it still keeps the bank open in at least
90% of seeds 1-10 (a binary search over the window, assuming acting earlier is never worse; checked at the edge).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from ui.live.presets import RESPONSES

PRESET = "lost-bonds"
SINGLE_TIMES = (0, 30, 90, 150, 240, 330, 390, 450, 600, 930, 1200)
PAIR_TIMES = (90, 240, 390, 930)
ROUND_SEEDS = ((1,), (1, 2, 3), tuple(range(1, 11)))
HOLDOUT_SEEDS = tuple(range(11, 21))
KEEP = 1 / 3
DEADLINE_SEEDS = tuple(range(1, 11))
DEADLINE_SURVIVAL = 0.9
DEFAULT_FILE = "data/live_search.json"


def search_file() -> str:
    return os.environ.get("STAMPEDE_SEARCH_FILE", DEFAULT_FILE)


def pct(x: float) -> str:
    """One decimal under 10%, so 1.8% and 1.7% never both read as "2%"."""
    return f"{x:.1%}" if abs(x) < 0.1 else f"{x:.0%}"


def clock(minute: float) -> str:
    m = 570 + int(round(minute)); hh = m // 60; h = hh % 24
    return f"{(h + 11) % 12 + 1}:{m % 60:02d} {'AM' if h < 12 else 'PM'}" + (" (day 2)" if hh >= 24 else "")


def _when_key(step) -> tuple:
    name, when = step
    return (0, float(when), name) if isinstance(when, (int, float)) else (1, when[0], float(when[1]), name)


def _when_id(when) -> str:
    return str(when) if isinstance(when, (int, float)) else f"{'req' if when[0] == 'requested' else 'cash'}{float(when[1]):g}"


def _when_text(when) -> str:
    if isinstance(when, (int, float)):
        return f"at {clock(when)}"
    level = float(when[1]) * 100
    pct = f"{level:.2f}".rstrip("0").rstrip(".")
    return f"once withdrawals pass {pct}% of deposits" if when[0] == "requested" else f"once cash falls below {pct}%"


def plan_id(plan) -> str:
    return "+".join(f"{n}@{_when_id(w)}" for n, w in sorted(plan, key=_when_key)) or "none"


def plan_label(plan) -> str:
    if not plan:
        return "The 2023 replay (no response)"
    return " + ".join(f"{RESPONSES[n].label} {_when_text(w)}" for n, w in sorted(plan, key=_when_key))


# UI-3d: trigger rules, the conditions a regulator would act on (withdrawals reported, cash left)
REQ_LEVELS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.16, 0.20)
CASH_LEVELS = (0.9, 0.75, 0.5)
DETECTION_MARGIN = 2.0   # the run "stands out" past twice the most a normal day ever reaches


def trigger_plans() -> list:
    req = lambda x: ("requested", x)
    cash = lambda x: ("cash", x)
    plans: list = [[]]
    plans += [[("guarantee", req(x))] for x in REQ_LEVELS] + [[("guarantee", cash(x))] for x in CASH_LEVELS]
    plans += [[("credit-line", req(x))] for x in (0.005, 0.02, 0.05)] + [[("credit-line", cash(x))] for x in (0.75, 0.5)]
    plans += [[("withdrawal-limit", req(x))] for x in (0.005, 0.02, 0.05)]
    plans += [[("withdrawal-limit", req(a)), ("guarantee", req(b))] for a in (0.005, 0.02) for b in (0.05, 0.08, 0.12)]
    plans += [[("credit-line", cash(0.75)), ("guarantee", req(b))] for b in (0.05, 0.08, 0.12)]
    return plans


def all_plans(earliest: int = 0) -> list[list[tuple[str, int]]]:
    """``earliest``: no response before this minute (a search-space choice: nobody responds before the news spreads)."""
    if earliest:
        return [p for p in all_plans(0) if all(m >= earliest for _, m in p)]
    names = list(RESPONSES)
    plans: list[list[tuple[str, int]]] = [[]]
    plans += [[(n, t)] for n in names for t in SINGLE_TIMES]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            plans += [[(a, ta), (b, tb)] for ta in PAIR_TIMES for tb in PAIR_TIMES]
    plans += [[(n, t) for n in names] for t in PAIR_TIMES]
    plans += [[("credit-line", t1), ("withdrawal-limit", t1), ("guarantee", t2)] for t1 in (90, 240) for t2 in (390, 930)]
    return plans


def run_plan(args) -> dict[str, Any]:
    """One live real-mode run of a plan, recorded every 10 minutes for the tournament replay."""
    plan, seed, minutes = args[:3]
    preset = args[3] if len(args) > 3 else PRESET
    credibility = args[4] if len(args) > 4 else None   # UI-3e: the rumor at another strength
    os.environ["STAMPEDE_MODE"] = "real"
    os.environ.setdefault("STAMPEDE_OFFLINE", "1")
    from contracts.events import BankClosed, WithdrawalRequested
    from engine.bank.ledger import registry_deposits_total
    from ui.live.pacing import crisis_sim_minutes
    from ui.live.session import LiveSession

    t0 = time.perf_counter()
    session = LiveSession(seed=seed)
    session.load_preset(preset, plan=list(plan), credibility=credibility)
    end = int(minutes if minutes is not None else crisis_sim_minutes())
    deposits = registry_deposits_total()
    log, cursor, requested, closure, cash_out, series = session.engine.bus.log, 0, 0.0, None, None, []
    for minute in range(1, end + 1):
        session.step_once()
        for event in log[cursor:]:
            if isinstance(event, WithdrawalRequested):
                requested += float(event.amount)
            elif isinstance(event, BankClosed) and closure is None:
                closure = float(event.t)
        cursor = len(log)
        cash = float(session.engine["M29"].cash_gauge)
        if cash_out is None and cash <= 0.0:
            cash_out = float(minute)
        if minute % 10 == 0 or minute == end:  # UI-3b: the engine's crowd counts colour the tiles' dots
            crowd = session.engine["M29"]
            heads = max(1, int(crowd.withdrew) + int(crowd.wavering) + int(crowd.calm))
            series.append([minute, round(requested / deposits, 5), round(cash, 4),
                           round(int(crowd.withdrew) / heads, 4), round(int(crowd.wavering) / heads, 4)])
    tag = plan_id(plan) if preset == PRESET else f"baseline:{preset}"
    if credibility is not None:
        tag = f"strength:{credibility:g}"
    return {"plan": tag, "seed": seed, "minutes": end, "credibility": credibility,
            "open": closure is None, "closure_minute": closure, "fired": dict(session.trigger_minutes),
            "fired_requested": dict(session.trigger_requested),
            "cash_out_minute": cash_out, "requested": requested / deposits, "requested_usd": requested,
            "emergency_usd": session._ui_emergency, "series": series, "seconds": round(time.perf_counter() - t0, 1)}


def _public_cost(plan: list[tuple[str, int]]) -> float:
    from engine.policy.levers import dollar_costs, load_policy_params

    params = load_policy_params()
    return sum(dollar_costs(RESPONSES[n].lever, params)[0] for n, _ in plan)


def score(runs: list[dict[str, Any]], plan: list[tuple[str, int]]) -> dict[str, Any]:
    n = len(runs)
    return {"runs": n, "survival": sum(r["open"] for r in runs) / n,
            "deposits_lost": sum(r["requested"] for r in runs) / n,
            "deposits_lost_usd": sum(r["requested_usd"] for r in runs) / n,
            "public_cost_assumed_usd": _public_cost(plan), "responses": len(plan)}


def _key(s: dict[str, Any]) -> tuple:
    return (bool(s.get("false_alarm")), -s["survival"], s["deposits_lost"], s["public_cost_assumed_usd"], s["responses"])


class Runner:
    """Runs (plan, seed) cells once each, in worker processes, and remembers them."""

    def __init__(self, workers: int, minutes: int | None, batched: bool | None = None) -> None:
        self.workers, self.minutes, self.done = workers, minutes, {}
        # GPU-1 (sandbox test): batch each worker's runs with the depositor model on the GPU
        self.batched = gpu_search_enabled() if batched is None else bool(batched)
        self.pool = None
        if workers > 1:
            import multiprocessing as mp

            self.pool = mp.get_context("spawn").Pool(processes=workers)

    def _map(self, todo: list) -> list[dict[str, Any]]:
        if not todo:
            return []
        if not self.batched:
            return self.pool.map(run_plan, todo, chunksize=1) if self.pool else [run_plan(c) for c in todo]
        from ui.live.gpu_batch import default_group_size

        size = default_group_size()
        if self.pool:   # each worker batches its own lanes; the per-lane CPU modules run in parallel
            size = max(1, min(size, -(-len(todo) // self.workers)))
        order = sorted(range(len(todo)), key=lambda i: int(todo[i][1]))   # same-seed lanes share static arrays
        chunks = [[todo[i] for i in order[j:j + size]] for j in range(0, len(order), size)]
        parts = self.pool.map(_run_batch, chunks, chunksize=1) if self.pool else [_run_batch(c) for c in chunks]
        rows = [row for part in parts for row in part]
        back = [None] * len(todo)
        for i, row in zip(order, rows):
            back[i] = row
        return back

    def run(self, cells: list[tuple[list[tuple[str, int]], int]]) -> None:
        todo = [(p, s, self.minutes) for p, s in cells if (plan_id(p), s) not in self.done]
        rows = self._map(todo)
        for row in rows:
            self.done[(row["plan"], row["seed"])] = row

    def runs(self, plan: list[tuple[str, int]], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
        return [self.done[(plan_id(plan), s)] for s in seeds]

    def run_baseline(self, preset: str, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
        """UI-3d: the no-rumor day (what normal looks like), with no response."""
        todo = [([], s, self.minutes, preset) for s in seeds if (f"baseline:{preset}", s) not in self.done]
        rows = self._map(todo)
        for row in rows:
            self.done[(row["plan"], row["seed"])] = row
        return [self.done[(f"baseline:{preset}", s)] for s in seeds]

    def run_strengths(self, strengths: tuple[float, ...], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
        """UI-3e: the bond rumor at several strengths, no response: some sink the bank, some fizzle."""
        todo = [([], s, self.minutes, PRESET, c) for c in strengths for s in seeds if (f"strength:{c:g}", s) not in self.done]
        rows = self._map(todo)
        for row in rows:
            self.done[(row["plan"], row["seed"])] = row
        return [self.done[(f"strength:{c:g}", s)] for c in strengths for s in seeds]

    def close(self) -> None:
        if self.pool:
            self.pool.close(); self.pool.join()


def gpu_search_enabled() -> bool:
    """Batch the runs with the depositors on the GPU (GPU-1).

    CPY-1: on by default when the GPU paths run (CuPy with a device, ``STAMPEDE_BACKEND`` not ``cpu``).
    ``STAMPEDE_SEARCH_GPU=1`` forces it (NumPy if there's no device), ``0`` turns it off.
    """
    from engine.xp import gpu_active

    value = os.environ.get("STAMPEDE_SEARCH_GPU", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return gpu_active()


def _run_batch(cells: list) -> list[dict[str, Any]]:
    from ui.live.gpu_batch import run_plans_batched

    return run_plans_batched(cells, array_module=os.environ.get("STAMPEDE_SEARCH_GPU_XP", "auto"))


def tournament(runner: Runner, plans: list[list[tuple[str, int]]], round_seeds=ROUND_SEEDS,
               holdout=HOLDOUT_SEEDS, keep: float = KEEP) -> dict[str, Any]:
    alive, rounds = list(plans), []
    for k, seeds in enumerate(round_seeds):
        runner.run([(p, s) for p in alive for s in seeds])
        scored = sorted(((p, score(runner.runs(p, seeds), p)) for p in alive), key=lambda x: _key(x[1]))
        last = k == len(round_seeds) - 1
        n_keep = len(scored) if last else max(3, int(round(len(scored) * keep)))
        kept = [p for p, _ in scored[:n_keep]]
        rounds.append({"round": k + 1, "seeds": list(seeds),
                       "plans": [{"plan": plan_id(p), "label": plan_label(p), "rank": i + 1, "kept": i < n_keep, **s}
                                 for i, (p, s) in enumerate(scored)]})
        alive = kept
    top = alive[:3]
    runner.run([(p, s) for p in top for s in holdout])
    confirm = [{"plan": plan_id(p), "label": plan_label(p), **score(runner.runs(p, holdout), p)} for p in top]
    return {"rounds": rounds, "holdout": {"seeds": list(holdout), "plans": confirm},
            "winner": confirm[0] if confirm else None}


def deadline(runner: Runner, name: str, seeds=DEADLINE_SEEDS, need: float = DEADLINE_SURVIVAL,
             end: int = 1440) -> dict[str, Any]:
    """Latest minute at which the response alone keeps the bank open in at least ``need`` of ``seeds``."""
    def ok(minute: int) -> float:
        plan = [(name, minute)]
        runner.run([(plan, s) for s in seeds])
        return score(runner.runs(plan, seeds), plan)["survival"]

    first = ok(0)
    if first < need:
        return {"response": name, "label": RESPONSES[name].label, "saves_bank": False, "survival_at_open": first,
                "text": f"{RESPONSES[name].label}: never keeps the bank open on its own, even at 9:30 AM ({first:.0%} of runs)."}
    lo, hi = 0, end
    if ok(end) >= need:
        lo = end
    while hi - lo > 5:
        mid = (lo + hi) // 2
        if ok(mid) >= need:
            lo = mid
        else:
            hi = mid
    edge = ok(lo)
    return {"response": name, "label": RESPONSES[name].label, "saves_bank": True, "deadline_minute": lo,
            "deadline_clock": clock(lo), "survival_at_deadline": edge, "first_failing_minute": hi if lo < end else None,
            "text": f"{RESPONSES[name].label}: keeps the bank open if it comes by {clock(lo)} ({edge:.0%} of runs)."}


FULL_SEEDS = tuple(range(1, 11))


def _mean_series(runs: list[dict[str, Any]]) -> list[list[float]]:
    rows = [r["series"] for r in runs]
    n = min(len(x) for x in rows)
    return [[rows[0][i][0]] + [round(sum(x[i][k] for x in rows) / len(rows), 5) for k in range(1, len(rows[0][i]))]
            for i in range(n)]


def elimination(plans: list[list[tuple[str, int]]], runner: Runner, seeds: tuple[int, ...],
                disqualified: dict[str, str] | None = None) -> dict[str, Any]:
    """UI-3b: every plan on the same seeds; each is knocked out the moment it is certain to lose.

    A plan is out when (a) its mean deposits requested so far already exceed what the best plan loses all
    day (requests only add up, so it cannot catch up), or (b) its bank has closed on enough seeds that its
    survival is below the best plan's. Anything left at the end is ranked out until one plan remains.
    """
    rows = []
    for p in plans:
        runs = runner.runs(p, seeds)
        rows.append({"plan": plan_id(p), "label": plan_label(p), "steps": [[n, list(m) if isinstance(m, tuple) else m] for n, m in p],
                     **score(runs, p), "false_alarm": plan_id(p) in (disqualified or {}),
                     "fired_by_seed": [r.get("fired", {}) for r in runs],
                     "open_by_seed": [bool(r["open"]) for r in runs], "lost_by_seed": [round(r["requested"], 5) for r in runs],
                     "closure_by_seed": [r["closure_minute"] for r in runs], "mean_series": _mean_series(runs)})
    order = sorted(range(len(rows)), key=lambda i: _key(rows[i]))
    for rank, i in enumerate(order, 1):
        rows[i]["rank"] = rank
    best = rows[order[0]]
    best_lost, best_survival, n = best["deposits_lost"], best["survival"], len(seeds)
    for row in rows:
        if row is best:
            row.update(elim_minute=None, reason="the engine's pick")
            continue
        if row["false_alarm"]:  # UI-3d: out at once; it would have guaranteed a healthy bank
            row.update(elim_minute=0, reason=(disqualified or {})[row["plan"]])
            continue
        t_dom = next((m for m, req, *_ in row["mean_series"] if req > best_lost + 1e-9), None)
        closures = sorted(c for c in row["closure_by_seed"] if c is not None)
        need = int((1.0 - best_survival) * n + 1e-9) + 1  # closures that put it below the best's survival
        t_close = closures[need - 1] if len(closures) >= need else None
        if t_dom is not None and (t_close is None or t_dom <= t_close):
            at = row["mean_series"][[m for m, *_ in row["mean_series"]].index(t_dom)][1]
            row.update(elim_minute=t_dom, reason=f"already lost {pct(at)} of deposits by {clock(t_dom)}, more than the best plan loses all day ({pct(best_lost)})")
        elif t_close is not None:
            row.update(elim_minute=t_close, reason=f"bank closed in {len(closures)} of {n} days")
        else:
            why = (f"bank closed in {n - round(row['survival'] * n)} of {n} days" if row["survival"] < best_survival
                   else f"lost {pct(row['deposits_lost'])} of deposits, against {pct(best_lost)} for the pick" if row["deposits_lost"] > best_lost + 1e-9
                   else "same result as the pick, with more responses" if row["responses"] > best["responses"]
                   else "same result as the pick, at a higher assumed public cost*")
            row.update(elim_minute=1441, reason=why)
    seq = sorted((r for r in rows if r is not best), key=lambda r: (r["elim_minute"], -r["rank"]))
    for k, r in enumerate(seq):
        r["elim_order"] = k + 1
    best["elim_order"] = len(rows)
    return {"seeds": list(seeds), "plans": rows, "pick": best["plan"]}


def run_full_search(workers: int = 1, minutes: int | None = None, plans=None, seeds=FULL_SEEDS,
                    holdout=HOLDOUT_SEEDS, deadline_seeds=DEADLINE_SEEDS, earliest: int = 0) -> dict[str, Any]:
    """UI-3b: every plan on the same seeds (no rounds), the pick checked on held-out seeds, and the deadlines."""
    t0 = time.perf_counter()
    runner = Runner(workers, minutes)
    try:
        plans = plans if plans is not None else all_plans(earliest)
        runner.run([(p, s) for p in plans for s in seeds])
        result = elimination(plans, runner, tuple(seeds))
        pick = next(p for p in plans if plan_id(p) == result["pick"])
        runner.run([(pick, s) for s in holdout])
        result["holdout"] = {"seeds": list(holdout), "plan": result["pick"], "label": plan_label(pick),
                             **score(runner.runs(pick, holdout), pick)}
        result["deadlines"] = [deadline(runner, n, deadline_seeds, end=minutes or 1440) for n in RESPONSES]
    finally:
        runner.close()
    result.update({"kind": "stampede response search", "mode": "full", "preset": PRESET, "what_happened": "none",
                   "generated": time.strftime("%Y-%m-%d %H:%M"), "earliest_minute": earliest,
                   "runs": len(runner.done), "minutes": minutes or 1441, "workers": workers,
                   "wall_seconds": round(time.perf_counter() - t0, 1),
                   "series": {f"{p}|{s}": r["series"] for (p, s), r in runner.done.items()}})
    return result


BASELINE_PRESET = "stadium"   # nothing about the bank is posted: what a normal day looks like
# UI-3e: rumor strengths (credibility) for the "failure likely" test, from one that fizzles to the replay's own
STRENGTHS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 0.9)
LIKELY = 0.9          # a guarantee is justified once 9 in 10 runs that got this far failed
LIKELY_MIN_RUNS = 10  # and at least this many runs got that far


def likely_failure(rows: list[dict[str, Any]], need: float = LIKELY, min_runs: int = LIKELY_MIN_RUNS) -> dict[str, Any]:
    """For each withdrawal level: of the runs (every strength, no response) that reached it, how many failed?
    The point of likely failure is the lowest level where at least ``need`` did. It depends on the mix of strengths."""
    table, level = [], None
    for x in REQ_LEVELS:
        reached = [r for r in rows if r["requested"] >= x]
        failed = sum(not r["open"] for r in reached)
        rate = failed / len(reached) if reached else None
        table.append({"level": x, "reached": len(reached), "failed": failed, "rate": rate})
        if level is None and reached and len(reached) >= min_runs and rate >= need:
            level = x
    return {"level": level, "need": need, "table": table, "strengths": sorted({r["credibility"] for r in rows}),
            "runs": len(rows), "note": "measured over rumors of these strengths, weighted equally, with no response"}


def first_cross(series: list, level: float) -> float | None:
    """The minute a run's requested share first reaches ``level``, interpolated between its 10-minute samples."""
    prev_m, prev_q = 0.0, 0.0
    for m, q, *_ in series:
        if q >= level:
            return prev_m + (m - prev_m) * ((level - prev_q) / (q - prev_q) if q > prev_q else 1.0)
        prev_m, prev_q = m, q
    return None


def _requested_at(run: dict[str, Any], minute: float) -> float:
    prev = 0.0
    for m, req, *_ in run["series"]:
        if m > minute:
            break
        prev = req
    return prev


def too_early(plans, runner: Runner, seeds, level: float | None) -> dict[str, str]:
    """Rules whose guarantee fires before failure was likely (withdrawals still under ``level``) on any seed."""
    if level is None:
        return {}
    out = {}
    for p in plans:
        if not any(n == "guarantee" for n, _ in p):
            continue
        for run in runner.runs(p, seeds):
            g = run.get("fired", {}).get("guarantee")
            # the exact share requested when the guarantee fired (recorded by the session); else the 10-minute samples
            at = run.get("fired_requested", {}).get("guarantee")
            if g is not None and (at if at is not None else _requested_at(run, g)) < level - 1e-12:
                out[plan_id(p)] = f"guarantees before failure was likely (withdrawals still under {pct(level)})"
                break
    return out


def false_alarms(plans, baseline: list[dict[str, Any]]) -> dict[str, str]:
    """Rules that would fire on a normal day. A clock-time step always fires, so it counts as one too."""
    top = max(r["requested"] for r in baseline)
    low = min(min(x[2] for x in r["series"]) for r in baseline)
    out = {}
    for p in plans:
        for n, w in p:
            if isinstance(w, (int, float)) or (w[0] == "requested" and top >= float(w[1])) or (w[0] == "cash" and low <= float(w[1])):
                out[plan_id(p)] = "would also fire on a normal day with no rumor: a guarantee for a healthy bank" if n == "guarantee" \
                    else "would also fire on a normal day with no rumor"
                break
    return out


def action_window(runner: Runner, seeds: tuple[int, ...], baseline: list[dict[str, Any]], deadline_row: dict[str, Any],
                  likely: dict[str, Any] | None = None) -> dict[str, Any]:
    """When the run stood out, when failure became likely (UI-3e), and when a guarantee stopped working."""
    noise = max(r["requested"] for r in baseline)
    level = next((x for x in REQ_LEVELS if x >= DETECTION_MARGIN * noise), REQ_LEVELS[-1])
    hits = []
    for r in runner.runs([], seeds):
        m = first_cross(r["series"], level)
        if m is not None:
            hits.append(round(m))
    hits.sort()
    median = hits[len(hits) // 2] if hits else None
    end = deadline_row.get("deadline_minute") if deadline_row.get("saves_bank") else None
    likely_level = (likely or {}).get("level")
    likely_minute = None
    if likely_level is not None:
        hits_l = sorted(round(m) for m in (first_cross(r["series"], likely_level) for r in runner.runs([], seeds)) if m is not None)
        likely_minute = hits_l[len(hits_l) // 2] if hits_l else None
    if likely is not None:   # UI-3e: the window opens when failure is likely, not merely visible
        if likely_level is None:
            text = "In these runs no withdrawal level made failure 9-in-10 likely, so no guarantee was clearly justified."
        elif end is None:
            text = f"Failure became likely by about {clock(likely_minute)} (withdrawals past {pct(likely_level)}), but a guarantee alone never saved the bank."
        elif likely_minute is None or end < likely_minute:
            text = f"By the time failure was likely (withdrawals past {pct(likely_level)}), a guarantee could no longer save the bank."
        else:
            text = (f"Failure became likely by about {clock(likely_minute)}: at least 9 in 10 runs whose withdrawals passed {pct(likely_level)} failed. "
                    f"A guarantee still kept the bank open until {clock(end)}.")
        return {"normal_day_max_requested": noise, "detect_level": level, "detect_minutes": hits, "detect_minute": median,
                "likely_level": likely_level, "likely_minute": likely_minute, "deadline_minute": end,
                "window_minutes": (end - likely_minute) if (end is not None and likely_minute is not None) else None, "text": text}
    if median is None:
        text = "The run never stood out from a normal day."
    elif end is None:
        text = f"The run stood out by about {clock(median)}, but a guarantee alone never saved the bank."
    elif end < median:
        text = f"By the time the run stood out (about {clock(median)}), a guarantee could no longer save the bank."
    else:
        text = (f"The run stood out from normal days by about {clock(median)} (withdrawals past {level:.1%} of deposits); "
                f"a guarantee still kept the bank open until {clock(end)}.")
    return {"normal_day_max_requested": noise, "detect_level": level, "detect_minutes": hits, "detect_minute": median,
            "deadline_minute": end, "window_minutes": (end - median) if (end is not None and median is not None) else None,
            "text": text}


def run_trigger_search(workers: int = 1, minutes: int | None = None, plans=None, seeds=FULL_SEEDS,
                       holdout=HOLDOUT_SEEDS, deadline_seeds=DEADLINE_SEEDS) -> dict[str, Any]:
    """UI-3d: trigger rules on the same seeds; rules that fire on a normal day are out; the action window."""
    t0 = time.perf_counter()
    runner = Runner(workers, minutes)
    try:
        plans = plans if plans is not None else trigger_plans()
        runner.run([(p, s) for p in plans for s in seeds])
        baseline = runner.run_baseline(BASELINE_PRESET, tuple(seeds))
        strengths = runner.run_strengths(STRENGTHS, tuple(seeds))
        likely = likely_failure(strengths)
        disq = {**too_early(plans, runner, tuple(seeds), likely["level"]), **false_alarms(plans, baseline)}
        result = elimination(plans, runner, tuple(seeds), disqualified=disq)
        pick = next(p for p in plans if plan_id(p) == result["pick"])
        runner.run([(pick, s) for s in holdout])
        result["holdout"] = {"seeds": list(holdout), "plan": result["pick"], "label": plan_label(pick),
                             **score(runner.runs(pick, holdout), pick)}
        result["deadlines"] = [deadline(runner, n, deadline_seeds, end=minutes or 1440) for n in RESPONSES]
        result["action_window"] = action_window(runner, tuple(seeds), baseline, result["deadlines"][0], likely)
        result["likely_failure"] = likely
        result["baseline"] = {"preset": BASELINE_PRESET, "max_requested_by_seed": [r["requested"] for r in baseline],
                              "min_cash_by_seed": [min(x[2] for x in r["series"]) for r in baseline]}
    finally:
        runner.close()
    result.update({"kind": "stampede response search", "mode": "triggers", "preset": PRESET, "what_happened": "none",
                   "generated": time.strftime("%Y-%m-%d %H:%M"), "runs": len(runner.done), "minutes": minutes or 1441,
                   "workers": workers, "wall_seconds": round(time.perf_counter() - t0, 1),
                   "series": {f"{p}|{s}": r["series"] for (p, s), r in runner.done.items()}})
    return result


# UI-4: viewers choose how many runs per plan to include, up to a limit, from a pool run before
# the demo. Every option uses the first N seeds of the pool, in a fixed order (never a chosen subset). The held-out
# days sit outside the pool, so "checked on new days" stays true at every option; deadlines stay on seeds 1-10.
RUN_OPTIONS = (10, 25, 50)
POOL_HOLDOUT_SEEDS = tuple(range(1001, 1011))


def run_trigger_options(workers: int = 1, minutes: int | None = None, options: tuple[int, ...] = RUN_OPTIONS,
                        plans=None, holdout=POOL_HOLDOUT_SEEDS, deadline_seeds=DEADLINE_SEEDS) -> dict[int, dict[str, Any]]:
    """UI-4: the trigger search at each run count, from one pool of runs (each (plan, seed) is run once)."""
    t0 = time.perf_counter()
    options = tuple(sorted({int(n) for n in options}))
    if not options or options[0] < LIKELY_MIN_RUNS:
        raise ValueError(f"run options must be at least {LIKELY_MIN_RUNS}: {options}")
    runner = Runner(workers, minutes)
    out: dict[int, dict[str, Any]] = {}
    try:
        plans = plans if plans is not None else trigger_plans()
        pool = tuple(range(1, options[-1] + 1))
        runner.run([(p, s) for p in plans for s in pool])   # the whole pool first, in one batch
        runner.run_baseline(BASELINE_PRESET, pool)
        runner.run_strengths(STRENGTHS, pool)
        for n in reversed(options):
            out[n] = _trigger_result(runner, plans, tuple(range(1, n + 1)), holdout, deadline_seeds, minutes)
    finally:
        runner.close()
    for n, result in out.items():
        result.update({"kind": "stampede response search", "mode": "triggers", "preset": PRESET, "what_happened": "none",
                       "generated": time.strftime("%Y-%m-%d %H:%M"), "runs": len(runner.done), "minutes": minutes or 1441,
                       "workers": workers, "wall_seconds": round(time.perf_counter() - t0, 1),
                       "runs_per_plan": n, "run_options": list(options), "pool_seeds": options[-1]})
    return out


def _trigger_result(runner: Runner, plans, seeds: tuple[int, ...], holdout, deadline_seeds, minutes) -> dict[str, Any]:
    """The trigger search's scoring on ``seeds`` from runs already in ``runner`` (anything missing is run)."""
    runner.run([(p, s) for p in plans for s in seeds])
    baseline = runner.run_baseline(BASELINE_PRESET, seeds)
    strengths = runner.run_strengths(STRENGTHS, seeds)
    likely = likely_failure(strengths)
    disq = {**too_early(plans, runner, seeds, likely["level"]), **false_alarms(plans, baseline)}
    result = elimination(plans, runner, seeds, disqualified=disq)
    pick = next(p for p in plans if plan_id(p) == result["pick"])
    runner.run([(pick, s) for s in holdout])
    result["holdout"] = {"seeds": list(holdout), "plan": result["pick"], "label": plan_label(pick),
                         **score(runner.runs(pick, holdout), pick)}
    result["deadlines"] = [deadline(runner, n, deadline_seeds, end=minutes or 1440) for n in RESPONSES]
    result["deadline_seeds"] = len(deadline_seeds)
    result["action_window"] = action_window(runner, seeds, baseline, result["deadlines"][0], likely)
    result["likely_failure"] = likely
    result["baseline"] = {"preset": BASELINE_PRESET, "max_requested_by_seed": [r["requested"] for r in baseline],
                          "min_cash_by_seed": [min(x[2] for x in r["series"]) for r in baseline]}
    # per-run curves only for this option's seeds (the API never sends them; /api/search/run reads them)
    keep = set(seeds) | set(holdout) | set(deadline_seeds)
    result["series"] = {f"{p}|{s}": r["series"] for (p, s), r in runner.done.items() if s in keep}
    return result


def option_file(n: int, path: str | None = None) -> str:
    """data/live_search.json -> data/live_search_25.json."""
    base, ext = os.path.splitext(path or search_file())
    return f"{base}_{int(n)}{ext or '.json'}"


def write_search_options(results: dict[int, dict[str, Any]], path: str | None = None) -> list[str]:
    """One file per option, and the largest as the default file (what /api/search serves without ?runs=)."""
    written = [write_search(data, option_file(n, path)) for n, data in sorted(results.items())]
    written.append(write_search(results[max(results)], path))
    return written


def available_run_options(path: str | None = None) -> list[int]:
    data = load_search(path)
    opts = (data or {}).get("run_options") or []
    return [n for n in opts if os.path.exists(option_file(n, path))]


def run_search(workers: int = 1, minutes: int | None = None, plans=None, round_seeds=ROUND_SEEDS,
               holdout=HOLDOUT_SEEDS, deadline_seeds=DEADLINE_SEEDS) -> dict[str, Any]:
    t0 = time.perf_counter()
    runner = Runner(workers, minutes)
    try:
        plans = plans if plans is not None else all_plans()
        result = tournament(runner, plans, round_seeds, holdout)
        result["deadlines"] = [deadline(runner, n, deadline_seeds, end=minutes or 1440) for n in RESPONSES]
    finally:
        runner.close()
    result.update({"kind": "stampede response search", "preset": PRESET, "plans": len(plans),
                   "runs": len(runner.done), "minutes": minutes or 1441, "workers": workers,
                   "wall_seconds": round(time.perf_counter() - t0, 1),
                   "series": {f"{p}|{s}": r["series"] for (p, s), r in runner.done.items()},
                   "outcomes": {f"{p}|{s}": {k: r[k] for k in ("open", "closure_minute", "cash_out_minute", "requested")}
                                for (p, s), r in runner.done.items()}})
    return result


def write_search(data: dict[str, Any], path: str | None = None) -> str:
    target = path or search_file()
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    return target


def load_search(path: str | None = None, runs: int | None = None) -> dict[str, Any] | None:
    """The search file; with ``runs`` (UI-4), that option's file."""
    target = option_file(runs, path) if runs else (path or search_file())
    if not os.path.exists(target):
        return None
    with open(target, encoding="utf-8") as handle:
        return json.load(handle)
