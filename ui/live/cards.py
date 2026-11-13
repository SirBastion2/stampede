"""UI-2: the response cards (a game-style pros-and-cons box shown when a viewer hovers over a response).

Measured parts come from the ten-seed batch (PA-B: lost-bonds with each response at minute 240 against
no response, same seeds). Public cost and moral hazard come from the engine's existing M22 cost model,
whose rates are ASSUMED registry rows; the card marks them with an asterisk. Nothing is hard-coded in the page.
"""

from __future__ import annotations

from typing import Any

from ui.live.presets import RESPONSES

FOOT = "Measured over {runs} seeds of the bond rumor, with the response at 1:30 PM. * The engine's assumed costs."


def _usd(v: float) -> str:
    return f"${v / 1e9:.1f}B" if abs(v) < 1e11 else f"${v / 1e9:.0f}B"


def _clock(minute: float) -> str:
    m = 570 + int(round(minute)); hh = m // 60; h = hh % 24
    return f"{(h + 11) % 12 + 1}:{m % 60:02d} {'AM' if h < 12 else 'PM'}"


def _mean(d: dict | None) -> float | None:
    return None if not d else float(d["mean"])


def response_cards(batch: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    from engine.policy.levers import dollar_costs, load_policy_params

    params = load_policy_params()
    # UI-3: the batch now holds each response at several minutes; the cards describe 1:30 PM (minute 240)
    summary = {(r["preset"], r["response"]): r for r in (batch or {}).get("summary", [])
               if r.get("response_minute") in (None, 240)}
    base = summary.get(("lost-bonds", None))
    # CM-1: the public stat is money at risk, with the expected loss in words. The FDIC charges a loss only when
    # the bank fails (2023: the loss on protecting SVB's uninsured depositors was 10.4% of them, row 306), so the
    # guarantee's expected loss is (share of runs closed) x that rate x the uninsured deposits it covers. A Fed
    # loan against bonds is exposure (the measured lending), not a loss.
    lent = {n: _mean(r.get("emergency_usd")) for (p, n), r in summary.items() if p == "lost-bonds" and n}  # CM-2
    uninsured = params.deposits_total * params.uninsured_share
    at_risk: dict[str, float] = {}
    for n, r in RESPONSES.items():
        row = summary.get(("lost-bonds", n))
        if r.lever == "guarantee_deposits":  # requests freeze when it lands, so the next open's share is the share out
            asked = (_mean(row.get("requested_1440")) or 0.0) * params.deposits_total if row else 0.0
            at_risk[n] = max(0.0, uninsured - asked)
        else:
            at_risk[n] = (lent.get(n) or 0.0) if r.lever == "emergency_lending" else 0.0
    hazard = {n: float(params.moral_bases.get(r.lever, 0.25)) for n, r in RESPONSES.items()}
    top_risk, top_hazard = max(at_risk.values()) or 1.0, max(hazard.values()) or 1.0
    cards: dict[str, dict[str, Any]] = {}
    for name, response in RESPONSES.items():
        dif, fed = dollar_costs(response.lever, params)
        row = summary.get(("lost-bonds", name))
        card: dict[str, Any] = {"name": name, "label": response.label, "note": response.note, "kind": "scenario",
                                "measured": bool(row and base), "stats": [], "pros": [], "cons": []}
        if row and base:
            runs = int(row["runs"])
            none_1440, resp_1440 = _mean(base["requested_1440"]) or 0.0, _mean(row["requested_1440"]) or 0.0
            stop = max(0.0, min(1.0, 1.0 - resp_1440 / none_1440)) if none_1440 else 0.0
            open_runs = runs - int(row["closed_runs"])
            out_none = _mean(base.get("cash_out_minute"))
            never_out = int(row.get("cash_out_runs", 0)) == 0
            out_resp = None if never_out else _mean(row.get("cash_out_minute"))
            if out_none is None:
                time_bar, time_text = 0.0, "the bank kept its cash anyway"
            elif never_out:
                time_bar, time_text = 1.0, f"cash lasts all night (no response: out around {_clock(out_none)})"
            else:
                time_bar = max(0.0, min(1.0, (out_resp - out_none) / max(1.0, 1440.0 - out_none)))
                time_text = f"cash out around {_clock(out_resp)} (no response: {_clock(out_none)})"
            card["stats"] = [
                {"key": "stops", "label": "Stops the run", "bar": round(5 * stop), "good": True,
                 "text": f"{resp_1440:.0%} of deposits asked by the next open (no response: {none_1440:.0%})"},
                {"key": "time", "label": "Buys time", "bar": round(5 * time_bar), "good": True, "text": time_text},
                {"key": "survives", "label": "Bank survives", "bar": round(5 * open_runs / runs), "good": True,
                 "text": f"open at the next morning in {open_runs} of {runs} runs"},
            ]
            card["footer"] = FOOT.format(runs=runs)
            debt = _mean(row.get("emergency_usd"))
            if name == "guarantee":
                card["pros"] = [f"Stops the run: withdrawals end at {resp_1440:.0%} of deposits", f"The bank stays open in {open_runs} of {runs} runs"]
                card["cons"] = [f"Most public money at risk: about {_usd(at_risk[name])} of deposits guaranteed; about "
                                f"{_usd(params.dif_uninsured_fraction * at_risk[name])} lost if the bank still fails",
                                f"Highest moral hazard ({hazard[name]:.2f}*): banks and depositors expect the next rescue",
                                "Money already requested stays out"]
            elif name == "credit-line":
                # CM-2: the line is on the 2023 BTFP terms; the Fed's exposure is the measured debt, not a fraction
                card["pros"] = ["The bank keeps paying all night: its cash never runs out" if never_out else time_text,
                                "No loss to the deposit insurance fund"]
                if open_runs == runs:
                    card["pros"].append(f"The bank is open the next morning in all {runs} runs")
                card["cons"] = [f"The bank takes on about {_usd(debt)} of emergency debt from the Fed, against its bonds"
                                if debt is not None else "Takes on emergency debt (rerun the batch table to measure it)",
                                f"Depositors don't see it, so the run doesn't slow: {resp_1440:.0%} asked by the next open"]
                card["cons"].append(f"Still closed at the next open in {runs - open_runs} of {runs} runs" if open_runs < runs
                                    else "It survives on borrowed money: if its loans have lost value, it can still fail later")
            else:
                card["pros"] = [f"Withdrawals fall to about {resp_1440 / none_1440:.0%} of the no-response run" if none_1440 else "Withdrawals fall",
                                "No public money", f"Lowest moral hazard ({hazard[name]:.2f}*)"]
                card["cons"] = ["Depositors can't reach 70% of what they ask for until the next open",
                                "If people expect a limit, they rush before it lands (money funds, 2020)"]
                if open_runs < runs:
                    card["cons"].append(f"Still closed at the next open in {runs - open_runs} of {runs} runs")
        else:
            card["footer"] = "Measured effects appear once the batch table is on the server (scripts/batch_table.py). * The engine's assumed costs."
        if dif:   # the guarantee: expected loss from the runs, else the loss if it fails
            closed = (int(row["closed_runs"]) / int(row["runs"])) if row and base else None
            if_fails = params.dif_uninsured_fraction * at_risk[name]
            opened = f"{int(row['runs']) - int(row['closed_runs'])} of {int(row['runs'])} runs open" if closed is not None else ""
            expect = "no loss expected" if not closed else f"expected loss about {_usd(closed * if_fails)}"
            text = (f"about {_usd(at_risk[name])} guaranteed; {expect} ({opened}), about {_usd(if_fails)} if it fails"
                    if closed is not None else f"up to about {_usd(at_risk[name])} guaranteed; about {_usd(if_fails)} lost if it fails")
        elif fed:
            text = (f"no loss expected; the Fed lends about {_usd(lent[name])} against bonds" if lent.get(name)
                    else "no loss expected; the Fed lends against bonds")
        else:
            text = "none"
        card["stats"] += [
            {"key": "public", "label": "Public money at risk", "bar": round(5 * at_risk[name] / top_risk), "good": False,
             "text": text},
            {"key": "hazard", "label": "Moral hazard", "bar": round(5 * hazard[name] / top_hazard), "good": False,
             "text": f"{hazard[name]:.2f}*"},
        ]
        cards[name] = card
    return cards
