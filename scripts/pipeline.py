"""AU-1: the whole search as one job: simulate, fill in the tipping point, check messaging, rescore, done.

    STAMPEDE_MODE=real PYTHONPATH=. python scripts/pipeline.py                 # what the UI's "Run the search" button runs
    STAMPEDE_MODE=real PYTHONPATH=. python scripts/pipeline.py --force         # resimulate even if nothing changed
    STAMPEDE_MODE=real PYTHONPATH=. python scripts/pipeline.py --reuse         # keep the runs on disk (files made before
                                                                               # the input key existed); stages 2-4 only

Stages (progress in ``data/pipeline_status.json``, which ``/api/pipeline/status`` serves):
  1. simulate: the trigger search's pool (``search.run_trigger_options``), skipped when the inputs that decide it
     (the registry, the plans, the options) haven't changed since the files on disk were made;
  2. tipping point: no-response runs at the strengths the last rescoring suggested, on the first ``--tip-seeds``
     seeds, added to the files (the rescoring uses a strength once every seed of a file has it);
  3. messaging: the behaviour-changing tools (statement, rate rise) added to the best ladders, simulated at two
     strengths on a few seeds, and compared with the ladder alone (a report, not ranked: they need full runs);
  4. rescore: ``scripts/rescore_useful.py`` on every file (the generated plans, the checks, the sensitivity).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())


def _status_path() -> str:
    from ui.live.search import search_file

    return os.path.join(os.path.dirname(search_file()) or ".", "pipeline_status.json")


def status(stage: str, done: int = 0, total: int = 0, message: str = "", **extra) -> None:
    path = _status_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {"stage": stage, "done": done, "total": total, "message": message, "time": time.strftime("%H:%M:%S"),
            "pid": os.getpid(), **extra}
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    os.replace(path + ".tmp", path)


def input_key(options) -> str:
    """What decides the simulations: the registry, the plan list and the option counts."""
    from ui.live.search import trigger_plans

    h = hashlib.sha256()
    with open("params/registry.csv", "rb") as fh:
        h.update(fh.read())
    h.update(json.dumps([[list(map(str, s)) for s in p] for p in trigger_plans()]).encode())
    h.update(json.dumps(sorted(options)).encode())
    return h.hexdigest()[:16]


def _raw(path: str) -> str:
    return path[:-5] + ".raw.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="10,25,50")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--minutes", type=int, default=None)
    ap.add_argument("--tip-seeds", type=int, default=10)
    ap.add_argument("--tip-strengths", default=None, metavar="LIST",
                    help="VF-5: simulate these rumor strengths (e.g. 0.26,0.27,0.28,0.29) instead of the suggested ones")
    ap.add_argument("--messaging-seeds", type=int, default=3)
    ap.add_argument("--messaging-plans", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="keep the simulations on disk even without a matching input key (stage 1 skipped)")
    args = ap.parse_args(argv)
    os.environ.setdefault("STAMPEDE_MODE", "real")
    from engine.xp import default_workers
    from ui.live.search import option_file, search_file

    workers = args.workers or default_workers()
    options = tuple(int(x) for x in args.runs.split(","))
    key = input_key(options)
    base = search_file()
    files = [option_file(n, base) for n in options] + [base]
    try:
        # 1. simulate (or reuse)
        if args.force and args.reuse:
            raise SystemExit("--force and --reuse contradict each other")
        cached = all(os.path.exists(_raw(p)) or os.path.exists(p) for p in files)
        if args.reuse and not cached:
            raise SystemExit("--reuse: not every search file is on disk; run without it to simulate")
        if cached and not args.force and not args.reuse:
            with open(_raw(base) if os.path.exists(_raw(base)) else base, encoding="utf-8") as fh:
                cached = json.load(fh).get("input_key") == key
        if not cached or args.force:
            from ui.live.search import run_trigger_options, write_search_options

            status("simulate", 0, 1, f"simulating the plans on {max(options)} versions of the day ({workers} workers)")
            results = run_trigger_options(workers=workers, minutes=args.minutes, options=options)
            for data in results.values():
                data["input_key"] = key
            for p in files:   # the new runs replace any old rescoring's originals
                if os.path.exists(_raw(p)):
                    os.remove(_raw(p))
            write_search_options(results, base)
        elif args.reuse:
            status("simulate", 1, 1, "reusing the simulations on disk (--reuse)")
        else:
            status("simulate", 1, 1, "nothing changed since the last run: reusing its simulations")
        # 2. the tipping point (strengths between the two where the rumor flips)
        from ui.live.useful import rescore

        raw10 = _raw(files[0]) if os.path.exists(_raw(files[0])) else files[0]
        with open(raw10, encoding="utf-8") as fh:
            first = json.load(fh)
        tip = rescore(first, checks=False, generated=[]).get("tipping", {})
        if args.tip_strengths:   # VF-5: named strengths, e.g. to fill in versions of the day a rerun dropped
            suggest = [float(x) for x in args.tip_strengths.split(",") if x.strip()]
        else:
            suggest = [c for c in tip.get("suggest", []) if f"strength:{c:g}|1" not in first.get("series", {})]
        if suggest:
            _tipping_runs(suggest, args.tip_seeds, workers, args.minutes, files)
        # 3. messaging, on top of the best ladders
        _messaging(first, args.messaging_plans, args.messaging_seeds, workers, args.minutes, base)
        # 4. rescore every file
        import importlib.util

        spec = importlib.util.spec_from_file_location("rescore_useful", "scripts/rescore_useful.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for k, p in enumerate(files):
            status("rescore", k, len(files), f"ranking {os.path.basename(p)}")
            if os.path.exists(p) or os.path.exists(_raw(p)):
                print(mod.summary_line(p, mod.rescore_file(p)), flush=True)   # VF-1: the log has each file's result
        status("done", len(files), len(files), "the wall and the Why page have the new results", finished=True)
        return 0
    except Exception as exc:   # the UI shows the failure instead of hanging
        status("failed", 0, 0, f"{type(exc).__name__}: {exc}", finished=True)
        raise


def _tipping_runs(strengths: list, seeds: int, workers: int, minutes, files: list) -> None:
    from ui.live.search import PRESET, Runner

    have: set = set()   # VF-5: runs already on disk (the largest file has every seed) are not simulated again
    for p in files:
        src = _raw(p) if os.path.exists(_raw(p)) else p
        if os.path.exists(src):
            with open(src, encoding="utf-8") as fh:
                have |= set(json.load(fh).get("series", {}))
    todo = [([], s, minutes, PRESET, c) for c in strengths for s in range(1, seeds + 1)
            if f"strength:{c:g}|{s}" not in have]
    status("tipping point", 0, len(todo), f"simulating strengths {', '.join(f'{c:g}' for c in strengths)} "
           f"on {seeds} versions of the day ({len(todo)} new runs)")
    if not todo:
        return
    runner = Runner(workers, minutes)
    try:
        rows = runner._map(todo)
    finally:
        runner.close()
    new = {f"strength:{c:g}|{r['seed']}": r["series"] for (_, _, _, _, c), r in zip(todo, rows)}
    for p in files:   # into each file's originals, so the next rescoring sees them
        src = _raw(p) if os.path.exists(_raw(p)) else p
        if not os.path.exists(src):
            continue
        with open(src, encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("series", {}).update({k: v for k, v in new.items() if int(k.split("|")[1]) <= len(data["seeds"])})
        with open(src, "w", encoding="utf-8") as fh:
            json.dump(data, fh)


def _messaging(first: dict, plans: int, seeds: int, workers: int, minutes, base: str) -> None:
    """Statements and rate rises change what depositors do, so they need their own runs: add them to the best
    worked-out ladders and simulate at a middling and the 2023 strength, against the ladder alone."""
    if plans <= 0 or seeds <= 0:
        return
    from ui.live.plangen import messaging_plans
    from ui.live.search import PRESET, Runner
    from ui.live.useful import rescore

    ranked = rescore(first, checks=False)
    best = [[(n, tuple(w) if isinstance(w, list) else w) for n, w in p["steps"]] for p in ranked["plans"][:plans]
            if p["steps"]]
    variants = messaging_plans(best, plans)
    todo = [(v, s, minutes, PRESET, c) for v in variants for c in (0.3, 0.9) for s in range(1, seeds + 1)]
    from engine.system.can_pay import can_pay_fractions
    from ui.live.useful import predict

    fr = can_pay_fractions()
    status("messaging", 0, len(todo), f"simulating a statement and a rate rise on top of the best {plans} plans")
    runner = Runner(workers, minutes)
    try:
        rows = runner._map(todo)
    finally:
        runner.close()
    S = first["series"]
    report = []
    for v in variants:
        for c in (0.3, 0.9):
            got = [r["requested"] for (pv, _, _, _, cc), r in zip(todo, rows) if pv is v and cc == c]
            ladder = [st for st in v if st[0] not in ("statement", "rate-rise")]   # the same plan without the message
            ref = [predict(ladder, S[f"strength:{c:g}|{s}"], fr)[1] for s in range(1, seeds + 1)]
            report.append({"plan": [[n, list(w) if isinstance(w, tuple) else w] for n, w in v], "strength": c,
                           "asked_change_points": round((sum(got) / len(got) - sum(ref) / len(ref)) * 100, 2), "runs": len(got)})
    with open(os.path.join(os.path.dirname(base) or ".", "messaging_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"note": "Statements and rate rises change what depositors do; their effects are assumed in the engine. "
                           "Change in what is asked by the next open, against the same plan without the message on the same days.",
                   "rows": report}, fh, indent=1)


if __name__ == "__main__":
    raise SystemExit(main())
