"""diag-19 (CM-2): the credit line on 2023 BTFP terms against the 2023 replay, full live window.

Per run: closure (minute, reason), cash-out minute, the line's draws and what is left of its pool,
bond sales (and any after the pool ran out: those bonds were pledged), realised losses, M08's lowest
and final cash, requests at the 4 PM close and the next open, and an event-log digest.

Run it on the tree before CM-2 with --out data/diag19_before.json, then after with
--compare data/diag19_before.json: every replay digest must match (the replay must not move).
Usage: PYTHONPATH=. python scripts/diag19_credit_line.py --seeds 1-10 --workers 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

CASES = (("lost-bonds", None), ("lost-bonds", "credit-line"))


def _seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def run(case: tuple[str, str | None, int, int | None]) -> dict:
    preset, response, seed, minutes = case
    os.environ["STAMPEDE_MODE"] = "real"
    os.environ.setdefault("STAMPEDE_OFFLINE", "1")
    from contracts.events import AssetSold, BankClosed, LiquidityDrawn, LossRealized, WithdrawalRequested
    from engine.bank.ledger import registry_deposits_total
    from ui.live.pacing import crisis_sim_minutes
    from ui.live.session import LiveSession

    t0 = time.perf_counter()
    session = LiveSession(seed=seed)
    session.load_preset(preset, response, 240 if response else None)
    end = int(minutes or crisis_sim_minutes())
    deposits = registry_deposits_total()
    log, cursor = session.engine.bus.log, 0
    digest = hashlib.sha256()
    out = {"preset": preset, "response": response, "seed": seed, "requested": 0.0, "closure": None,
           "closure_reason": None, "cash_out": None, "line_usd": 0.0, "other_draws_usd": 0.0, "first_line_minute": None,
           "sold_usd": 0.0, "first_sale_minute": None, "loss_usd": 0.0, "min_m08_cash_usd": None,
           "sold_after_pool_usd": 0.0, "pool_usd": None, "pool_left_usd": None, "m08_end_cash_usd": None,
           "regulator_status": None}
    terms = getattr(session.engine.get("M12"), "terms", None)  # None on the tree before CM-2
    pool = None if terms is None else terms.total_fraction * deposits
    out["pool_usd"] = pool
    at = {}
    for minute in range(1, end + 1):
        session.step_once()
        for event in log[cursor:]:
            digest.update(type(event).__name__.encode())
            digest.update(json.dumps(event.model_dump(), sort_keys=True, default=str).encode())
            if isinstance(event, WithdrawalRequested):
                out["requested"] += float(event.amount)
            elif isinstance(event, LiquidityDrawn):
                if event.source == "emergency":
                    out["line_usd"] += float(event.amount)
                    out["first_line_minute"] = out["first_line_minute"] or minute
                else:
                    out["other_draws_usd"] += float(event.amount)
            elif isinstance(event, AssetSold):
                out["sold_usd"] += float(event.amount)
                if response and pool is not None and out["line_usd"] >= pool - 1.0:
                    out["sold_after_pool_usd"] += float(event.amount)
                out["first_sale_minute"] = out["first_sale_minute"] or minute
            elif isinstance(event, LossRealized):
                out["loss_usd"] += float(event.amount)
            elif isinstance(event, BankClosed) and out["closure"] is None:
                out["closure"], out["closure_reason"] = float(event.t), event.reason
        cursor = len(log)
        state = getattr(session.engine.get("M08"), "state", None)
        if state is not None:
            cash = float(state.cash)
            low = out["min_m08_cash_usd"]
            out["min_m08_cash_usd"] = cash if low is None else min(low, cash)
        if out["cash_out"] is None and session.engine["M29"].cash_gauge <= 0.0:
            out["cash_out"] = minute
        if minute in (390, 1440):
            at[str(minute)] = round(out["requested"] / deposits, 4)
    out["requested_at"] = at
    state = getattr(session.engine.get("M08"), "state", None)
    out["m08_end_cash_usd"] = None if state is None else float(state.cash)
    out["pool_left_usd"] = None if pool is None or not response else pool - out["line_usd"]
    out["regulator_status"] = getattr(session.engine.get("M20"), "status", None)
    out["log_digest"] = digest.hexdigest()[:16]
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--minutes", type=int, default=None)
    ap.add_argument("--out", default="data/diag19_credit_line.json")
    ap.add_argument("--compare", default=None, help="a run on the tree before CM-2: replay digests must match")
    args = ap.parse_args(argv)
    cases = [(p, r, s, args.minutes) for s in _seeds(args.seeds) for p, r in CASES]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(run, cases))
    else:
        rows = [run(c) for c in cases]
    for r in rows:
        b = 1e9
        left = "-" if r["pool_left_usd"] is None else f"${r['pool_left_usd'] / b:.1f}B"
        print(f"seed {r['seed']:2} {str(r['response'] or 'replay'):12} req390 {r['requested_at'].get('390')} "
              f"req1440 {r['requested_at'].get('1440')} close {r['closure']} ({r['closure_reason']}) cash_out {r['cash_out']} "
              f"line ${r['line_usd']/b:.1f}B from {r['first_line_minute']} sold ${r['sold_usd']/b:.1f}B loss ${r['loss_usd']/b:.2f}B "
              f"pool left {left} "
              f"m08_low ${(r['min_m08_cash_usd'] or 0)/b:.1f}B digest {r['log_digest']} ({r['seconds']}s)", flush=True)
    failed = []
    for r in rows:
        if r["response"] is None and "1440" in r["requested_at"]:  # the full window only
            if not 0.25 <= (r["requested_at"].get("390") or 0) <= 0.275:
                failed.append(f"seed {r['seed']}: replay asked {r['requested_at'].get('390')} by the close (25-27.5%)")
            if not 0.80 <= (r["requested_at"].get("1440") or 0) <= 0.84:
                failed.append(f"seed {r['seed']}: replay asked {r['requested_at'].get('1440')} by the open (80-84%)")
        elif r["sold_after_pool_usd"] > 0.0:
            failed.append(f"seed {r['seed']}: the pool ran out and ${r['sold_after_pool_usd'] / 1e9:.1f}B of pledged bonds were sold")
    if args.compare:
        with open(args.compare, encoding="utf-8") as fh:
            before = {(r["seed"], r["response"]): r["log_digest"] for r in json.load(fh)}
        for r in rows:
            if r["response"] is None and (r["seed"], None) in before and before[(r["seed"], None)] != r["log_digest"]:
                failed.append(f"seed {r['seed']}: the replay moved (digest {before[(r['seed'], None)]} -> {r['log_digest']})")
    print("checks: " + ("all passed" if not failed else "; ".join(failed)))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
