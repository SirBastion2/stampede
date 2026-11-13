"""UF-1: rank the search's plans by how useful they are to the bank (Sebastian, Nov 5).

No new simulations. Before its FIRST action a plan's run is the no-response run on the
same seed. A later cash condition needs its own funded cash path; this shortcut cannot
establish that path and the plan is withheld from ranking rather than guessed.
Eligible plans use the search's no-response strength runs:
  guarantee    -> no new requests from where it fires (+0.08 points, the measured overshoot)
  withdrawal limit -> later requests are cut to 30%
  Fed program  -> requests unchanged; the bank can pay more if it is on in time (``engine/system/can_pay.py``)
A day is lost when depositors have asked for more than the bank can pay by the next open.

  threat recognized : the lowest withdrawal level after which at least 9 in 10 days go on to sink the bank
  point of no return: what the bank can pay with nothing done

Knocked out, in order (timed on the 2023 day, the wall's replay):
  1. acts too early: steps in on more than 1 in 10 days whose rumor would have fizzled
  2. not open to a US bank: the withdrawal limit (deposits are payable on demand)
  3. doesn't save it: keeps the bank open on fewer than 9 in 10 days a rumor would sink it
The rest are the useful plans, ranked: most dangerous days saved; fewest harmless days acted on; escalation (a
milder step before a guarantee); acting between the threat and the point of no return; acting later.
"""

from __future__ import annotations

import copy
import statistics as st
from typing import Any

from engine.system.can_pay import CanPay, can_pay_fractions

STRENGTHS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 0.9)
REPLAY_STRENGTH = 0.9
LEVELS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.16, 0.2)
HARMLESS_MAX, SAVE_MIN, LIKELY = 0.10, 0.90, 0.90
OVERSHOOT = 0.0008
LIMIT_SCALE = 0.3
MILD = {"credit-line", "withdrawal-limit"}


def cross_req(series: list, level: float, scale=None) -> float | None:
    """First minute the (possibly scaled) demand reaches ``level``, interpolated between samples."""
    prev = None
    for row in series:
        r = row[1] if scale is None else scale(row[1])
        if r >= level:
            if prev is None:
                return float(row[0])
            m0, r0 = prev
            return m0 + (row[0] - m0) * (level - r0) / max(1e-12, r - r0)
        prev = (row[0], r)
    return None


def cross_cash(series: list, level: float) -> float | None:
    """First minute cash falls to ``level`` (interpolated), plus the minute a trigger takes to fire."""
    prev = None
    for row in series:
        if row[2] <= level:
            if prev is None:
                return float(row[0])
            m0, c0 = prev
            return m0 + (row[0] - m0) * (c0 - level) / max(1e-12, c0 - row[2]) + 1.0
        prev = (row[0], row[2])
    return None


def at_minute(series: list, minute: float) -> float:
    prev = series[0]
    for row in series:
        if row[0] >= minute:
            return row[1] if row[0] == prev[0] else prev[1] + (row[1] - prev[1]) * (minute - prev[0]) / (row[0] - prev[0])
        prev = row
    return series[-1][1]


class CashPathRequired(ValueError):
    """A later cash condition needs its own funded path, not the no-response cash series."""


def predict(steps: list, series: list, fr: CanPay) -> tuple[dict, float, float, float | None]:
    """(minute each step fired, demand by the next open, what the bank can pay, first fire) for one day."""
    fired: dict[str, float] = {}
    prepared = any(n == "prepared" for n, _ in steps)   # UF-2: set up before the day; not an action on the rumor
    remaining = [(n, w) for n, w in steps if n != "prepared"]
    if prepared and any(isinstance(w, (list, tuple)) and w[0] == "cash" for _, w in remaining):
        raise CashPathRequired("A cash condition after bonds are pledged needs a funded cash-path simulation.")
    scale = None
    frozen = None
    line_at = None
    big_at = None
    while remaining:
        best = None
        for n, w in remaining:
            if isinstance(w, (list, tuple)) and w[0] == "requested":
                m = cross_req(series, float(w[1]), scale)
            elif isinstance(w, (list, tuple)) and w[0] == "cash":
                if fired:
                    raise CashPathRequired("A cash condition after another action needs its own cash-path simulation.")
                m = cross_cash(series, float(w[1]))
            else:
                m = float(w)
            if m is not None and frozen is not None and m > frozen[0] and n != "credit-line":
                m = None
            if m is not None and (best is None or m < best[0]):
                best = (m, n, w)
        if best is None:
            break
        m, n, w = best
        remaining.remove((n, w))
        fired[n] = m
        now = at_minute(series, m) if scale is None else scale(at_minute(series, m))
        if n == "guarantee":
            frozen = (m, now + OVERSHOOT)
        elif n == "withdrawal-limit":
            scale = (lambda b: (lambda r: b + LIMIT_SCALE * max(0.0, r - b)))(now)
        elif n == "credit-line":
            line_at = m
        elif n == "big-bank-deposits":
            big_at = m
    demand = series[-1][1] if scale is None else scale(series[-1][1])
    if frozen is not None:
        demand = min(demand, frozen[1])
    cap = fr.prepared if prepared else fr.none
    if line_at is not None:   # the Fed's line helps only if it is on before the bank runs short
        past = cross_req(series, cap, scale)
        if past is None or line_at <= past:
            cap = max(cap, fr.fed)
    if big_at is not None:    # UF-2: so do deposits from big banks; they add to whatever else it has
        past = cross_req(series, cap, scale)
        if past is None or big_at <= past:
            cap += fr.big_banks
    return fired, demand, cap, (min(fired.values()) if fired else None)


def _family(steps: list) -> str:
    names = [n for n, _ in steps]
    if not names:
        return "baseline"
    if "withdrawal-limit" in names:
        return "not-us"
    if "guarantee" in names or "credit-line" in names:
        return "regulator"
    return "other-banks" if "big-bank-deposits" in names else "bank"


FAMRANK = {"bank": 0, "other-banks": 1, "regulator": 2, "not-us": 3, "baseline": 4}   # prefer what the bank controls
BIG_TRIGGERS = (("requested", 0.02), ("requested", 0.05), ("requested", 0.08), ("requested", 0.12), ("cash", 0.75), ("cash", 0.5))
NAMES = {"prepared": "Bonds pledged in advance", "big-bank-deposits": "Deposits from big banks"}


def _when(w) -> str:
    if isinstance(w, (list, tuple)) and w[0] == "requested":
        return f"once withdrawals pass {float(w[1]) * 100:g}% of deposits"
    if isinstance(w, (list, tuple)) and w[0] == "cash":
        return f"once cash falls below {float(w[1]) * 100:g}%"
    return ""


def computed_plans() -> list[dict]:
    """UF-2: plans made of tools that change what the bank can pay, not what depositors do. Computed, not run."""
    out = [{"plan": "prepared@0", "steps": [["prepared", 0]], "label": NAMES["prepared"]}]
    for w in BIG_TRIGGERS:
        wid = f"{'req' if w[0] == 'requested' else 'cash'}{w[1]:g}"
        out.append({"plan": f"big-bank-deposits@{wid}", "steps": [["big-bank-deposits", list(w)]],
                    "label": f"{NAMES['big-bank-deposits']} {_when(w)}"})
        out.append({"plan": f"prepared@0+big-bank-deposits@{wid}", "steps": [["prepared", 0], ["big-bank-deposits", list(w)]],
                    "label": f"{NAMES['prepared']} + {NAMES['big-bank-deposits'].lower()} {_when(w)}"})
    return out


def _synth_series(steps: list, series: list, fr: CanPay) -> list:
    """A computed plan's day for the wall: the no-response day's requests (depositors don't see these tools), and the
    cash gauge held at a working level (30%) by the new borrowing once the tool is in place, while the bank can
    still pay; empty once depositors have asked for more than it can pay. A display stand-in, not a simulation."""
    fired, demand, cap, _ = predict(steps, series, fr)
    start = min(fired.values()) if fired else 0.0
    out = []
    for row in series:
        cash = row[2]
        if row[0] >= start:
            cash = max(cash, 0.3) if row[1] <= cap else 0.0
        out.append([row[0], row[1], round(cash, 4), *row[3:]])
    return out


def day_path(steps: list, series: list, fr: CanPay) -> list:
    """AU-2: a worked-out plan's day for the wall: the no-response day with the plan's own effect. A guarantee stops
    new requests where it fires; a limit cuts later ones to 30%; depositors who would have asked but haven't are
    scaled with the requests; cash drains with what is paid and, once a borrowing tool is in place, is held at a
    working level (30%) while the bank can still pay. A display stand-in for the wall, not a simulation."""
    fired, demand, cap, _ = predict(steps, series, fr)
    names = {n for n, _ in steps}
    lim = fired.get("withdrawal-limit")
    base_at = at_minute(series, lim) if lim is not None else None
    g = fired.get("guarantee")
    out = []
    borrow_from = min([fired[n] for n in fired if n in ("credit-line", "big-bank-deposits")] + ([0.0] if "prepared" in names else []), default=None)
    for row in series:
        m, r = row[0], row[1]
        req = r if (lim is None or m < lim) else base_at + LIMIT_SCALE * max(0.0, r - base_at)
        if g is not None and m >= g:
            req = min(req, demand)   # frozen where the guarantee fired (predict's value)
        k = req / r if r > 1e-9 else 1.0
        cash = 1.0 - (1.0 - row[2]) * k
        if borrow_from is not None and m >= borrow_from:
            cash = max(cash, 0.3) if req <= cap else 0.0
        out.append([m, round(req, 5), round(max(0.0, min(1.0, cash)), 4), round(row[3] * k, 4), round(row[4], 4)])
    return out


def _summary(p: dict, threat: float | None) -> str:
    """AU-6: one plain sentence per plan, made only from its computed numbers (nothing for a model to invent)."""
    if not p["steps"]:
        return "Nothing done: the bank fails at the next open on every day a strong rumor hits."
    when = f", typically around {_clock(p['fire_median'])}" if p.get("fire_median") is not None else ""
    if p.get("knocked") == "early":
        return f"{p['label']} would also step in on {p['acts_harmless']:.0%} of the rumors that died out on their own{when}: too early."
    if p.get("knocked") == "not-us":
        return f"{p['label']} isn't open to a US bank: deposits are payable on demand."
    if p.get("knocked") == "fails":
        return f"{p['label']} keeps the bank open on only {p['saves_dangerous']:.0%} of the days a rumor would sink it."
    close = f" (at worst by {p['closest'] * 100:.1f} points)" if p.get("closest") is not None else ""
    return (f"{p['label']} keeps the bank open on {p['saves_dangerous']:.0%} of the days a rumor would sink it{close}, "
            f"and steps in on {p['acts_harmless'] or 0:.0%} of the rumors that died out{when}.")


def _clock(m: float) -> str:
    t = int(round(9 * 60 + 30 + m)) % 1440
    return f"{(t // 60 + 11) % 12 + 1}:{t % 60:02d} {'AM' if t < 720 else 'PM'}"


def rescore(data: dict[str, Any], fr: CanPay | None = None, generated: list | None = None,
            wall_size: int = 44, wall_useful: int = 16, checks: bool = True) -> dict[str, Any]:
    """A search result (with its ``series``) rescored for usefulness; the same schema the wall reads, plus
    ``useful``, ``threat`` and per-plan ``knocked``, ``useful_rank``, ``saves_dangerous``, ``acts_harmless``."""
    fr = fr or can_pay_fractions()
    S = dict(data["series"])   # a copy: the computed plans' days are added to it
    seeds = list(data["seeds"])
    found = sorted({float(k.split("|")[0].split(":")[1]) for k in S if k.startswith("strength:")})
    strengths = [c for c in found if all(f"strength:{c:g}|{s_}" in S for s_ in seeds)] or list(STRENGTHS)
    days = [(c, s) for c in strengths for s in seeds]
    base = {(c, s): S[f"strength:{c:g}|{s}"] for c, s in days}
    sinks = {k: v[-1][1] > fr.none for k, v in base.items()}
    table = []
    for level in LEVELS:
        reached = [k for k, v in base.items() if v[-1][1] >= level]
        bad = sum(sinks[k] for k in reached)
        table.append({"level": level, "reached": len(reached), "failed": bad, "rate": bad / len(reached) if reached else None})
    threat = next((r["level"] for r in table if r["rate"] is not None and r["rate"] >= LIKELY), None)
    replay = [S[f"strength:{REPLAY_STRENGTH:g}|{s}"] for s in seeds]
    threat_t = [cross_req(v, threat) for v in replay] if threat is not None else []
    noreturn_t = [cross_req(v, fr.none) for v in replay]
    t_threat = st.median(x for x in threat_t if x is not None) if any(x is not None for x in threat_t) else None
    t_noreturn = st.median(x for x in noreturn_t if x is not None) if any(x is not None for x in noreturn_t) else None
    out = copy.deepcopy({k: v for k, v in data.items() if k != "series"})
    # Reconsider previously withheld plans on every rescore; never silently lose their provenance.
    previous = out.pop("unverified_plans", [])
    existing = {p["plan"] for p in out["plans"]}
    out["plans"].extend(p for p in previous if p["plan"] not in existing)
    known = {p["plan"] for p in out["plans"]}
    simulated_ids = {p["plan"] for p in out["plans"] if p["steps"] and not p.get("computed")}
    if generated is None:   # AU-2: the search writes its own plans, worked out from the no-response days
        from ui.live.plangen import generate

        generated = generate()
    for extra in generated:
        if extra["plan"] in known:
            continue
        known.add(extra["plan"])
        steps = [(n, tuple(w) if isinstance(w, list) else w) for n, w in extra["steps"]]
        out["plans"].append({**extra, "computed": True, "responses": len(steps),
                             "public_cost_assumed_usd": 0.0})
    unverified = []
    eligible = []
    for p in out["plans"]:
        steps = [(n, tuple(w) if isinstance(w, list) else w) for n, w in p["steps"]]
        try:
            for day in base.values():
                predict(steps, day, fr)
        except CashPathRequired as exc:
            # No survival, trigger-time or margin claim is available for this policy rule.
            unverified.append({"plan": p["plan"], "label": p["label"], "steps": p["steps"],
                               "computed": bool(p.get("computed")), "status": "needs-simulation",
                               "reason": str(exc)})
            continue
        if p.get("computed"):
            own = [day_path(steps, v, fr) for v in replay]
            p["mean_series"] = [[own[0][i][0], *[sum(d[i][j] for d in own) / len(own)
                                              for j in range(1, len(own[0][i]))]]
                                for i in range(len(own[0]))]
            p["fired_by_seed"] = [predict(steps, v, fr)[0] for v in replay]
            for s_, d in zip(seeds, own):
                S[f"{p['plan']}|{s_}"] = d
        eligible.append(p)
    out["plans"] = eligible
    out["unverified_plans"] = unverified
    rows = []
    for p in out["plans"]:
        steps = [(n, tuple(w) if isinstance(w, list) else w) for n, w in p["steps"]]
        saved = dang = acts = harm = inwin = 0
        fires = []
        closest = None   # the smallest margin, in deposits, between what it can pay and what was asked
        for k in days:
            fired, demand, cap, first = predict(steps, base[k], fr)
            if not sinks[k]:
                harm += 1
                acts += 1 if fired else 0
                continue
            dang += 1
            saved += demand <= cap
            closest = cap - demand if closest is None else min(closest, cap - demand)
            if first is not None:
                fires.append(first)
                thr = cross_req(base[k], threat) if threat is not None else None
                nr = cross_req(base[k], fr.none)
                if (thr is None or first >= thr) and (nr is None or first <= nr):
                    inwin += 1
        day23 = [predict(steps, v, fr) for v in replay]
        open23 = [demand <= cap for _, demand, cap, _ in day23]
        names = [n for n, _ in steps]
        f23 = [first for _, _, _, first in day23 if first is not None]
        p.update({
            "family": _family(steps),
            "saves_dangerous": saved / dang if dang else None, "acts_harmless": acts / harm if harm else None,
            "acts_in_window": inwin / dang if dang else None, "fire_median": st.median(fires) if fires else None,
            "escalates": len(names) > 1 and names[-1] in ("guarantee", "big-bank-deposits") and names[0] in MILD | {"prepared"},
            "survival": sum(open23) / len(open23), "open_by_seed": open23,
            "closure_by_seed": [None if ok else 1440.0 for ok in open23],
            "deposits_lost": sum(d for _, d, _, _ in day23) / len(day23), "lost_by_seed": [round(d, 5) for _, d, _, _ in day23],
            "useful_rank": None, "knocked": None, "closest": closest,
        })
        when = round(st.median(f23)) if f23 else 0
        if not steps:
            p.update(elim_minute=1441, reason="the 2023 replay: nothing done")
        elif p["family"] == "baseline":
            p.update(elim_minute=1441, reason="the 2023 replay: nothing done")
        elif p["acts_harmless"] is not None and p["acts_harmless"] > HARMLESS_MAX:
            p.update(elim_minute=when, knocked="early",
                     reason=f"acts before the run is a threat: it would also step in on {p['acts_harmless']:.0%} of the rumors that fizzle")
        elif "withdrawal-limit" in names:
            p.update(elim_minute=when, knocked="not-us", reason="not open to a US bank: deposits are payable on demand")
        elif (p["saves_dangerous"] or 0) < SAVE_MIN:
            past = [cross_req(v, cap) for (_, _, cap, _), v, ok in zip(day23, replay, open23) if not ok]
            past = [x for x in past if x is not None]
            p.update(elim_minute=round(st.median(past)) if past else 1441, knocked="fails",
                     reason=f"doesn't save the bank: open on only {p['saves_dangerous']:.0%} of the days a rumor would sink it")
        else:
            p.update(elim_minute=None, reason="")
        rows.append(p)
    kept = sorted((p for p in rows if p["elim_minute"] is None),
                  key=lambda p: (-round(p["saves_dangerous"], 3), round(p["acts_harmless"] or 0.0, 3), FAMRANK[p["family"]], not p["escalates"],
                                 -round(p["acts_in_window"] or 0.0, 3), -(p["fire_median"] or 0.0)))
    for k, p in enumerate(kept, 1):
        p["useful_rank"] = k
        bits = [f"saves {p['saves_dangerous']:.0%} of dangerous days", f"acts on {p['acts_harmless'] or 0:.0%} of harmless ones"]
        if p["escalates"]:
            bits.append("escalates: a smaller step first")
        if p["closest"] is not None and p["closest"] < 0.01:
            bits.append(f"close call: within {p['closest'] * 100:.1f} points of what the bank can pay")
        if p.get("computed"):
            bits.append("computed from the runs (these tools don't change what depositors do)")
        p["reason"] = f"#{k} most useful: " + "; ".join(bits)
    knocked = sorted((p for p in rows if p["elim_minute"] is not None), key=lambda p: p["elim_minute"])
    for r, p in enumerate(kept + knocked, 1):
        p["rank"] = r
    for k, p in enumerate(sorted(knocked, key=lambda p: (p["elim_minute"], -p["rank"]))):
        p["elim_order"] = k + 1
    for k, p in enumerate(reversed(kept)):
        p["elim_order"] = len(knocked) + 1 + k
    out["plans"] = kept + knocked
    out["pick"] = kept[0]["plan"] if kept else None
    out["useful"] = [p["plan"] for p in kept]
    out["strengths"] = list(STRENGTHS)
    out["threat"] = {"level": threat, "minute_2023": t_threat, "table": table, "no_return_level": fr.none,
                     "no_return_minute_2023": t_noreturn, "can_pay": {"none": fr.none, "fed": fr.fed}}
    text = []
    if threat is not None and t_threat is not None:
        text.append(f"On the 2023 day the run became a threat at {threat:.0%} asked (about {_clock(t_threat)}): from there 9 in 10 days sink the bank.")
    if t_noreturn is not None:
        text.append(f"Past {fr.none:.1%} asked (about {_clock(t_noreturn)}) it can't pay without selling underwater bonds.")
    text.append(f"{len(kept)} plans act in time without acting on rumors that fizzle.")
    out["action_window"] = {**(out.get("action_window") or {}), "likely_minute": t_threat, "deadline_minute": t_noreturn,
                            "detect_minute": None, "text": " ".join(text)}
    # AU-2: the wall shows the most useful plans and the simulated ones that were knocked out (plus a few
    # worked-out ones), so every reason is seen; the rest are counted and stay on the Why page's list
    everything = out["plans"]
    wall = kept[:wall_useful]
    wall += [p for p in knocked if p["plan"] in simulated_ids][: max(0, wall_size - len(wall))]
    by_reason: dict = {}
    for p in knocked:
        if p["plan"] not in simulated_ids and p["steps"]:
            by_reason.setdefault(p.get("knocked"), []).append(p)
    extra_room = max(0, wall_size - len(wall))
    for reason in ("early", "fails", "not-us"):
        take = by_reason.get(reason, [])[: max(1, extra_room // 3)]
        wall += take[: max(0, wall_size - len(wall))]
    wall += [p for p in knocked if not p["steps"]]
    for r, p in enumerate(wall, 1):
        p["wall_rank"] = r
    out["considered"] = {"plans": sum(1 for p in everything if p["steps"]) + len(unverified),
                         "simulated": sum(1 for p in everything if p["plan"] in simulated_ids),
                         "simulated_total": len(simulated_ids),
                         "unverified": len(unverified),
                         "worked_out": sum(1 for p in everything if p.get("computed")), "useful": len(kept),
                         # UI-5: why the rest are out, across every plan considered (the What helps page counts them)
                         "knocked": {r: sum(1 for p in knocked if p["steps"] and p.get("knocked") == r)
                                     for r in ("early", "not-us", "fails")}}
    out["useful_all"] = [{"plan": p["plan"], "label": p["label"], "family": p["family"], "rank": p["useful_rank"],
                          "saves": p["saves_dangerous"], "acts": p["acts_harmless"], "computed": bool(p.get("computed")),
                          "summary": _summary(p, threat)} for p in kept]
    for p in everything:
        p["summary"] = _summary(p, threat)
    wall_ids = {p["plan"] for p in wall}
    out["plans"] = [p for p in kept if p["plan"] in wall_ids] + [p for p in knocked if p["plan"] in wall_ids]
    # VF-1: ``useful`` is the useful plans ON THE WALL (at most ``wall_useful``, the ones ``plans`` describes);
    # every useful plan is ``useful_all``, and the count to show is ``considered.useful``.
    out["useful"] = [p["plan"] for p in kept if p["plan"] in wall_ids]
    out["pick"] = out["useful"][0] if out["useful"] else out["pick"]
    out["strengths"] = list(strengths)
    if checks:
        out["checks"] = _checks(everything, simulated_ids, S, seeds, fr)
        out["sensitivity"] = _sensitivity(kept[:25] + knocked[:10], base, sinks, days, threat, replay, fr)
    out["tipping"] = _tipping(strengths, seeds, sinks)
    out["warning"] = _warning(base, sinks)
    out["groups"], out["groupPlans"] = _groups(out["plans"], S, seeds, base, sinks, fr, threat, strengths)
    holdout = out.get("holdout")
    if holdout and holdout.get("plan") != out["pick"]:
        out["previous_holdout"] = holdout
        out["holdout"] = None   # evidence for the former winner is not validation of the reranked winner
    out["prediction_scope"] = "cash paths after earlier actions require independent simulation"
    out["mode"] = "useful"
    out["rescored"] = "usefulness to the bank, from the search's own runs (UF-1, no new simulations)"
    return out


def _groups(plans: list, S: dict, seeds: list, base: dict, sinks: dict, fr: CanPay, threat: float | None,
            strengths=STRENGTHS):
    """GR-1: each group is one version of the 2023 day (one seed) with every plan tried on it. Inside a group the same
    knock-outs apply (acts on a rumor that died out in that group; lets the bank fail on one that sank it; not open to
    a US bank); the rest are ranked as above. Each cell carries that plan's own day (every 30 minutes)."""
    order = sorted((p for p in plans if p["steps"]), key=lambda p: (FAMRANK[_family(p["steps"])], p["label"]))
    groups = []
    for s in seeds:
        day23 = S[f"strength:{REPLAY_STRENGTH:g}|{s}"]
        cells = []
        for p in order:
            steps = [(n, tuple(w) if isinstance(w, list) else w) for n, w in p["steps"]]
            names = [n for n, _ in steps]
            acts_h, saved, dang, fires, inwin = False, 0, 0, [], 0
            for c in strengths:
                fired, demand, cap, first = predict(steps, base[(c, s)], fr)
                if not sinks[(c, s)]:
                    acts_h |= bool(fired)
                else:
                    dang += 1
                    saved += demand <= cap
                    if first is not None:
                        fires.append(first)
                        thr = cross_req(base[(c, s)], threat) if threat is not None else None
                        nr = cross_req(base[(c, s)], fr.none)
                        inwin += (thr is None or first >= thr) and (nr is None or first <= nr)
            fired23, demand23, cap23, first23 = predict(steps, day23, fr)
            if acts_h:
                kind, minute = "early", first23 or 0
            elif "withdrawal-limit" in names:
                kind, minute = "not-us", first23 or 0
            elif saved < dang:
                kind = "fails"
                minute = (cross_req(day23, cap23) or 1441) if demand23 > cap23 else 1441
            else:
                kind, minute = "useful", None
            own = S.get(f"{p['plan']}|{s}") or day23
            cells.append({"k": kind, "m": None if minute is None else round(minute), "ok": bool(demand23 <= cap23),
                          "esc": bool(p.get("escalates")), "fam": FAMRANK[_family(steps)], "w": inwin,
                          "f": round(st.median(fires)) if fires else None, "c": bool(p.get("computed")),
                          "s": [[round(r[1], 4), round(r[2], 3), round(r[3], 3), round(r[4], 3)] for r in own[2::3]]})
        ranked = sorted((k for k, c in enumerate(cells) if c["k"] == "useful"),
                        key=lambda k: (cells[k]["fam"], not cells[k]["esc"], -cells[k]["w"], -(cells[k]["f"] or 0)))
        for r, k in enumerate(ranked, 1):
            cells[k]["r"] = r
        t_threat = cross_req(day23, threat) if threat is not None else None
        t_nr = cross_req(day23, fr.none)
        groups.append({"seed": s, "threat_m": round(t_threat) if t_threat else None, "noreturn_m": round(t_nr) if t_nr else None,
                       "asked": round(day23[-1][1], 4), "curve": [[r[0], round(r[1], 4)] for r in day23[2::3]], "cells": cells})
    return groups, [{"plan": p["plan"], "label": p["label"], "steps": p["steps"], "computed": bool(p.get("computed"))} for p in order]


def _checks(plans: list, simulated_ids: set, S: dict, seeds: list, fr: CanPay) -> dict:
    """AU-5: the checks that make the shortcut trustworthy, on every rescoring. For each simulated plan: its run
    matches the replay until it acts, and the shortcut's demand matches its own run by the next open."""
    prefix, err, n, worst = 0.0, 0.0, 0, None
    for p in plans:
        if p["plan"] not in simulated_ids:
            continue
        steps = [(n_, tuple(w) if isinstance(w, list) else w) for n_, w in p["steps"]]
        if any(n_ in ("statement", "rate-rise") for n_, _ in steps):
            continue   # behaviour changes: no shortcut to check
        for s in seeds:
            own, rep = S.get(f"{p['plan']}|{s}"), S.get(f"strength:{REPLAY_STRENGTH:g}|{s}")
            if not own or not rep:
                continue
            fired, demand, _, first = predict(steps, rep, fr)
            for a, b in zip(rep, own):
                if first is not None and a[0] < first - 10:
                    prefix = max(prefix, abs(a[1] - b[1]))
            e = abs(demand - own[-1][1])
            if e > err:
                err, worst = e, p["label"]
            n += 1
    ok = prefix < 1e-4 and err < 0.01   # the stored days are rounded to 5 decimals
    return {"runs_checked": n, "prefix_max_difference": prefix, "shortcut_max_error": err, "worst_plan": worst,
            "label": (f"Checked predictions against {n} independently simulated runs: requests match before acting, "
                      f"and next-open request differences are at most {err * 100:.1f} points. "
                      "This does not validate the cash path of plans not simulated.") if ok else
                     (f"Check failed on {n} simulated runs: largest difference before acting {prefix:.4f}, "
                      f"worked-out error {err * 100:.1f} points ({worst})."),
            "ok": ok}


def _sensitivity(plans: list, base: dict, sinks: dict, days: list, threat, replay: list, fr: CanPay) -> dict:
    """AU-3: does the ranking survive the assumptions? Rerank the leading plans with the Fed's margin at 94% and
    96%, what the bank can pay without help one point lower and higher, and the two cutoffs moved."""
    from dataclasses import replace

    pledged = (fr.prepared - fr.none) / 0.95   # the margin's share of the pledged capacity, approximately
    variants = [("as assumed", fr, HARMLESS_MAX, SAVE_MIN),
                ("Fed margin 94%", replace(fr, prepared=fr.prepared - 0.01 * pledged), HARMLESS_MAX, SAVE_MIN),
                ("Fed margin 96%", replace(fr, prepared=fr.prepared + 0.01 * pledged), HARMLESS_MAX, SAVE_MIN),
                ("bank can pay 1 point less", replace(fr, none=fr.none - 0.01), HARMLESS_MAX, SAVE_MIN),
                ("bank can pay 1 point more", replace(fr, none=fr.none + 0.01), HARMLESS_MAX, SAVE_MIN),
                ("too early = 1 in 20", fr, 0.05, SAVE_MIN), ("saves it = 8 in 10", fr, HARMLESS_MAX, 0.80)]
    results = []
    for name, f, hmax, smin in variants:
        sinks_v = {k: v[-1][1] > f.none for k, v in base.items()}
        scored = []
        for p in plans:
            steps = [(n, tuple(w) if isinstance(w, list) else w) for n, w in p["steps"]]
            if not steps:
                continue
            saved = dang = acts = harm = 0
            for k in days:
                fired, demand, cap, first = predict(steps, base[k], f)
                if not sinks_v[k]:
                    harm += 1
                    acts += 1 if fired else 0
                else:
                    dang += 1
                    saved += demand <= cap
            sv, ah = (saved / dang if dang else 0.0), (acts / harm if harm else 0.0)
            names = [n for n, _ in steps]
            if ah > hmax or "withdrawal-limit" in names or sv < smin:
                continue
            scored.append((-round(sv, 3), round(ah, 3), FAMRANK[_family(steps)], p["useful_rank"] or 99, p["plan"], p["label"]))
        scored.sort()
        results.append({"variant": name, "top": [x[5] for x in scored[:3]], "top_plans": [x[4] for x in scored[:3]]})
    top1 = results[0]["top_plans"][:1]
    holds = sum(1 for r in results if r["top_plans"][:1] == top1)
    return {"variants": results, "top1_holds": holds, "of": len(results),
            "text": f"The #1 plan stays #1 in {holds} of {len(results)} versions of the assumptions."}


def _tipping(strengths: list, seeds: list, sinks: dict) -> dict:
    """AU-3: where the rumor goes from dying out to sinking the bank: the first pair of neighbouring strengths whose
    share of days that sink the bank jumps by half or more, and strengths in between worth simulating."""
    share = {c: sum(sinks[(c, s)] for s in seeds) / len(seeds) for c in strengths}
    for a, b in zip(strengths, strengths[1:]):
        if share[b] - share[a] >= 0.5:
            step = (b - a) / 5
            return {"between": [a, b], "share": [share[a], share[b]], "suggest": [round(a + step * k, 3) for k in range(1, 5)],
                    "text": f"The rumor tips: a slightly weaker one (believability {a:g}) sinks the bank on {share[a]:.0%} of the "
                            f"versions of the day, a slightly stronger one ({b:g}) on {share[b]:.0%}."}
    return {"between": None, "share": None, "suggest": [], "text": "No small change in how believable the rumor is flips the result."}


def _warning(base: dict, sinks: dict) -> dict:
    """AU-6: the early-warning meter's data: every no-response day's requests every 30 minutes, and whether it sank."""
    minutes = [row[0] for row in next(iter(base.values()))[2::3]]
    days = [[int(sinks[k])] + [round(row[1], 5) for row in v[2::3]] for k, v in base.items()]
    return {"minutes": minutes, "days": days, "no_return": None}
