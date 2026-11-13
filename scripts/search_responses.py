"""UI-3: search response plans for the 2023 replay and write data/live_search.json for the tournament screen.

    STAMPEDE_MODE=real PYTHONPATH=. $PY scripts/search_responses.py --workers 6     # about 1 hour on the 3060 desktop
    STAMPEDE_MODE=real PYTHONPATH=. $PY scripts/search_responses.py --workers 6 --runs 10,25,50   # UI-4: about 4 hours

Trigger rules (UI-3d) by default: every rule on the same 10 seeds plus the no-rumor day; --clock searches clock times
instead (with --earliest), --rounds the older successive halving.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())


def _backend_flags(parser) -> None:
    """CPY-1: GPU by default. --cpu is the NumPy reference; --require-gpu refuses to fall back."""
    parser.add_argument("--cpu", action="store_true", help="the NumPy reference: STAMPEDE_BACKEND=cpu, no batching")
    parser.add_argument("--require-gpu", action="store_true", help="fail instead of falling back to NumPy")


def _apply_backend_flags(args) -> None:
    if args.cpu:
        os.environ["STAMPEDE_BACKEND"] = "cpu"
        os.environ["STAMPEDE_SEARCH_GPU"] = "0"
    if args.require_gpu:
        os.environ["STAMPEDE_REQUIRE_GPU"] = "1"
    if getattr(args, "gpu", False):
        os.environ["STAMPEDE_SEARCH_GPU"] = "1"
    from engine.xp import format_report, require_gpu

    require_gpu("this run")
    print(format_report(), flush=True)


def main() -> int:
    from ui.live.search import all_plans, clock, run_full_search, run_search, run_trigger_search, write_search

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=None, help="default: 6 (STAMPEDE_WORKERS overrides); with the GPU, "
                        "every worker shares the one 12 GB RTX 3060")
    parser.add_argument("--minutes", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--rounds", action="store_true", help="successive halving instead of every plan on 10 seeds")
    parser.add_argument("--earliest", type=int, default=0, help="--clock only: no response before this minute (90 = 11:00 AM)")
    parser.add_argument("--clock", action="store_true", help="plans at clock times instead of trigger rules")
    parser.add_argument("--runs", default=None, help="UI-4: runs per plan the viewer can choose, e.g. 10,25,50 "
                        "(one pool of the largest; a file per option). Default: the older single 10-seed search")
    parser.add_argument("--gpu", action="store_true", help="force batching (GPU-1); CPY-1: already the default "
                        "when CuPy sees a device. Check scripts/gpu_parity.py first")
    _backend_flags(parser)
    args = parser.parse_args()
    _apply_backend_flags(args)
    if args.workers is None:
        from engine.xp import default_workers

        args.workers = default_workers()
    if args.runs and not args.rounds and not args.clock:
        from ui.live.search import run_trigger_options, write_search_options

        options = tuple(int(x) for x in args.runs.split(","))
        results = run_trigger_options(workers=args.workers, minutes=args.minutes, options=options)
        for path in write_search_options(results, args.out):
            print("wrote", path)
        for n, data in sorted(results.items()):
            pick = next(r for r in data["plans"] if r["plan"] == data["pick"])
            h = data["holdout"]
            print(f"{n:>3} runs per plan: pick {pick['label']} (open {pick['survival']:.0%}, {pick['deposits_lost']:.1%} lost); "
                  f"held-out {h['survival']:.0%} open; {data['action_window']['text']}")
        print(f"{results[max(results)]['runs']} runs in {results[max(results)]['wall_seconds']} s")
        return 0
    if not args.rounds:
        data = (run_full_search(workers=args.workers, minutes=args.minutes, earliest=args.earliest) if args.clock
                else run_trigger_search(workers=args.workers, minutes=args.minutes))
        path = write_search(data, args.out)
        print(f"wrote {path}: {len(data['plans'])} plans x {len(data['seeds'])} seeds, {data['runs']} runs in {data['wall_seconds']} s")
        rows = sorted(data["plans"], key=lambda r: r["elim_order"])
        for r in rows[-6:]:
            when = "the pick" if r["elim_minute"] is None else ("at the next open" if r["elim_minute"] >= 1441 else clock(r["elim_minute"]))
            print(f"  {r['elim_order']:>3}. {r['label']}: open {r['survival']:.0%}, {r['deposits_lost']:.1%} lost; out {when}: {r['reason']}")
        h = data["holdout"]
        print(f"held-out check of the pick: {h['label']}: open {h['survival']:.0%}, {h['deposits_lost']:.1%} lost, over {h['runs']} new seeds")
        for d in data["deadlines"]:
            print(d["text"])
        if data.get("action_window"):
            print("action window:", data["action_window"]["text"])
            print(f"rules knocked out for firing on a normal day: {sum(r.get('false_alarm', False) for r in data['plans'])}")
        return 0
    data = run_search(workers=args.workers, minutes=args.minutes)
    path = write_search(data, args.out)
    print(f"wrote {path}: {data['plans']} plans, {data['runs']} runs in {data['wall_seconds']} s")
    for r in data["rounds"]:
        kept = sum(p["kept"] for p in r["plans"])
        print(f"round {r['round']}: {len(r['plans'])} plans on {len(r['seeds'])} seed(s), {kept} kept; best: {r['plans'][0]['label']}"
              f" (open {r['plans'][0]['survival']:.0%}, {r['plans'][0]['deposits_lost']:.1%} of deposits lost)")
    w = data["winner"]
    if w:
        print(f"held-out check of the winner: {w['label']}: open {w['survival']:.0%}, {w['deposits_lost']:.1%} lost, over {w['runs']} new seeds")
    for d in data["deadlines"]:
        print(d["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
