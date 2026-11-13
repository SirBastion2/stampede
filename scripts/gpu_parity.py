"""GPU-1 / CPY-1 parity and speed check: the same (plan, seed) runs on the NumPy reference (``search.run_plan``
with ``STAMPEDE_BACKEND=cpu``, in worker processes) and on CuPy (batched, ``ui/live/gpu_batch.py``, with every
array path on the GPU: population solve, network, attention, depositors), compared run by run.

    STAMPEDE_MODE=real PYTHONPATH=. python scripts/gpu_parity.py --seeds 1-3 --workers 6            # full window
    STAMPEDE_MODE=real PYTHONPATH=. python scripts/gpu_parity.py --seeds 1 --minutes 240 --xp numpy  # no GPU

Plans: the replay, and one plan of each kind the search tries (guarantee, credit line, limit, two-step).

Pass (VF-1, ADR-132), in every run:
  - the bank's outcome (open or closed at the next open) is the same;
  - every response fires on both paths, at minutes within 1 of each other;
  - deposits requested agree within ``TOL`` = 1e-4 OF DEPOSITS (0.01 points, about $17.5M on $175B), both at
    the end and at every 10-minute sample of the day.
The relative gap is still printed, for information only.

Why an absolute tolerance, not the old "1e-4 relative or 1e-6 of deposits" (GPU-1):
  - The gap is float32 rounding. With the kernel forced to float64 (the same batched code, NumPy), a guarantee
    run matches the CPU path to 2e-10 relative, so the batching itself adds nothing (independent NumPy check, Nov 6).
  - Its sign is random by seed, not by plan: on the 3060 (Nov 6) seeds 1 and 2 came out higher on the GPU for
    every plan and seed 3 lower, by 0.0001 to 0.0036 points. A missing or wrong update would push one way.
  - It is largest where the bank stays open with most depositors still deciding (a guarantee keeps about 92%
    of deposits in the bank for 20 hours, so the same rounding is tested every minute). Those runs ask little
    (8% of deposits), so a relative test divides the same dollar error by 10x less and fails them: 8 of 21
    failed at 1e-4 relative while every outcome and trigger minute agreed; the worst was $6.4M (0.0036 points).
  - What the search ranks on is outcomes (days the bank is saved) and trigger minutes, compared exactly here,
    and margins of 0.3 points (the close call) and 0.9 points (the checks). 0.01 points is 30x below the
    smallest of them and 2.5x above the worst gap seen anywhere in the day (0.004 points at a sample, which
    is rounded to 0.001 points), so a real regression (a lost guarantee moves whole points) still fails.
``--unfused`` runs the GPU side without CuPy's fused float32 kernels (``STAMPEDE_CALIB_UNFUSED_LOGITS`` and
``_PART2``). On the 3060 (Nov 6, VF-2) it changed no gap by more than 1e-9 of deposits: the fused kernels are not
why the 3060's gap is about 45x NumPy float32's (0.0032 against 0.00007 points for seed 2's guarantee run).
``--kernel-only`` keeps the network (M02), attention (M03) and population solve (M05) on NumPy float64
(``STAMPEDE_BACKEND=cpu`` on the batched side) and only the depositor kernel on CuPy: if the gap falls to about
NumPy float32's, those modules' float32 GPU paths make the rest; if it stays, the kernel's own CuPy arithmetic does. ``--reuse-cpu`` reads the NumPy reference from an earlier
output (the CPU side is deterministic and takes about 18 minutes), so the GPU side alone can be rerun.
It also prints the time per lane-minute for each path and the speed-up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

PLANS = (
    [],
    [("guarantee", ("requested", 0.08))],
    [("guarantee", ("cash", 0.75))],
    [("credit-line", ("cash", 0.75))],
    [("withdrawal-limit", ("requested", 0.02))],
    [("withdrawal-limit", ("requested", 0.02)), ("guarantee", ("requested", 0.12))],
    [("credit-line", ("cash", 0.75)), ("guarantee", ("requested", 0.08))],
)


def _seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


TOL = 1e-4   # share of deposits (ADR-132)


def batched_backend(xp: str, kernel_only: bool) -> str:
    """``STAMPEDE_BACKEND`` for the batched side: ``gpu`` (every array path on CuPy), or ``cpu`` with
    ``--kernel-only`` or ``--xp numpy`` (M02, M03 and M05 on NumPy; the kernel follows ``--xp``)."""
    return "cpu" if xp == "numpy" or kernel_only else "gpu"


def compare(a: dict, b: dict, tol: float = TOL) -> tuple[bool, dict]:
    """One run on the reference path (``a``) against the batched path (``b``): the pass rule above."""
    end_gap = abs(a["requested"] - b["requested"])
    sa, sb = a.get("series") or [], b.get("series") or []
    series_gap = max((abs(x[1] - y[1]) for x, y in zip(sa, sb)), default=0.0)
    fired_ok = set(a["fired"]) == set(b["fired"]) and all(abs(a["fired"][k] - b["fired"][k]) <= 1 for k in a["fired"])
    same_len = len(sa) == len(sb)
    ok = a["open"] == b["open"] and fired_ok and same_len and end_gap <= tol and series_gap <= tol
    return ok, {"end_gap": end_gap, "series_gap": series_gap, "rel": end_gap / max(1e-12, a["requested"]),
                "open_ok": a["open"] == b["open"], "fired_ok": fired_ok, "samples_ok": same_len}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-3")
    ap.add_argument("--minutes", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2, help="NumPy reference workers (the batched path is one process)")
    ap.add_argument("--xp", default="cupy", help="cupy (the default: fails without a device), numpy, or auto")
    ap.add_argument("--out", default="data/gpu_parity.json")
    ap.add_argument("--reuse-cpu", default=None, metavar="JSON",
                    help="take the NumPy reference rows from an earlier output with the same seeds and minutes")
    ap.add_argument("--unfused", action="store_true",
                    help="diagnostic: the GPU side without CuPy's fused float32 kernels")
    ap.add_argument("--kernel-only", action="store_true",
                    help="diagnostic: only the depositor kernel on CuPy; M02, M03 and M05 on NumPy float64")
    args = ap.parse_args(argv)
    os.environ["STAMPEDE_MODE"] = "real"
    os.environ.setdefault("STAMPEDE_OFFLINE", "1")
    from ui.live import search
    from ui.live.gpu_batch import run_plans_batched
    from ui.live.pacing import crisis_sim_minutes

    cells = [(p, s, args.minutes) for s in _seeds(args.seeds) for p in PLANS]
    lane_minutes = len(cells) * (args.minutes or crisis_sim_minutes())
    import multiprocessing as mp

    from engine.xp import format_report

    # the reference: NumPy everywhere, in fresh processes (spawn reads the environment at start)
    ids = [[search.plan_id(p), s] for p, s, _ in cells]
    if args.reuse_cpu:
        with open(args.reuse_cpu, encoding="utf-8") as fh:
            old = json.load(fh)
        if old.get("cells") != ids or any(r.get("minutes") != (args.minutes or crisis_sim_minutes()) for r in old["cpu"]):
            raise SystemExit(f"--reuse-cpu: {args.reuse_cpu} has other runs (seeds, plans or minutes); run without it")
        cpu, cpu_s = old["cpu"], float(old.get("cpu_s", 0.0))
        args.workers = int(old.get("workers", args.workers))   # the speed-up uses the reference's own workers
        print(f"reference: reusing {len(cpu)} NumPy runs from {args.reuse_cpu} ({cpu_s:.0f} s when made)", flush=True)
    else:
        before = os.environ.get("STAMPEDE_BACKEND")
        os.environ["STAMPEDE_BACKEND"] = "cpu"
        t0 = time.perf_counter()
        with mp.get_context("spawn").Pool(max(1, args.workers)) as pool:
            cpu = pool.map(search.run_plan, cells, chunksize=1)
        cpu_s = time.perf_counter() - t0
        if before is None:
            os.environ.pop("STAMPEDE_BACKEND", None)
        else:
            os.environ["STAMPEDE_BACKEND"] = before
    if args.unfused:
        os.environ["STAMPEDE_CALIB_UNFUSED_LOGITS"] = "1"
        os.environ["STAMPEDE_CALIB_UNFUSED_PART2"] = "1"
    if args.kernel_only and args.xp == "numpy":
        raise SystemExit("--kernel-only compares the CuPy kernel; it needs --xp cupy")
    os.environ["STAMPEDE_BACKEND"] = batched_backend(args.xp, args.kernel_only)
    print("batched side:", format_report(), flush=True)
    timings: dict = {}
    t1 = time.perf_counter()
    gpu = run_plans_batched(cells, array_module=args.xp, timings=timings)
    gpu_s = time.perf_counter() - t1
    from engine.bank.ledger import registry_deposits_total

    deposits = registry_deposits_total()
    failed, gaps = [], []
    print(f"pass: same outcome, triggers within 1 minute, asked within {TOL:g} of deposits ({TOL * 100:g} points, "
          f"${TOL * deposits / 1e6:,.1f}M) at the end and every sample" + ("; GPU side unfused" if args.unfused else "")
          + ("; kernel only on the GPU (M02, M03, M05 on NumPy)" if args.kernel_only else ""))
    for (plan, seed, _), a, b in zip(cells, cpu, gpu):
        ok, g = compare(a, b)
        gaps.append({"plan": search.plan_id(plan), "seed": seed, "signed_end_gap": b["requested"] - a["requested"], **g})
        print(f"{'ok  ' if ok else 'FAIL'} seed {seed:2} {search.plan_label(plan)[:60]:60} asked {a['requested']:.4f}/{b['requested']:.4f}"
              f" gap {(b['requested'] - a['requested']) * 100:+.4f} pts (${(b['requested'] - a['requested']) * deposits / 1e6:+.1f}M,"
              f" rel {g['rel']:.1e}), worst sample {g['series_gap'] * 100:.4f} pts; open {a['open']}/{b['open']} fired {a['fired']}/{b['fired']}",
              flush=True)
        if not ok:
            failed.append((seed, search.plan_id(plan)))
    worst = max(gaps, key=lambda g: max(g["end_gap"], g["series_gap"]))
    print(f"worst gap: {max(worst['end_gap'], worst['series_gap']) * 100:.4f} points ({worst['plan']}, seed {worst['seed']}); "
          f"GPU higher on {sum(g['signed_end_gap'] > 0 for g in gaps)} runs, lower on {sum(g['signed_end_gap'] < 0 for g in gaps)}")
    cpu_lm = cpu_s * max(1, args.workers) / lane_minutes * 1000   # per worker of the reference run
    gpu_lm = gpu_s / lane_minutes * 1000
    print(f"\nCPU path: {cpu_s:.0f} s with {args.workers} worker(s) = {cpu_lm:.1f} ms per lane-minute per worker")
    print(f"batched ({timings.get('backend')}): {gpu_s:.0f} s in one process = {gpu_lm:.1f} ms per lane-minute "
          f"(depositor kernel {timings.get('kernel_s', 0) / lane_minutes * 1000:.1f} ms of it)")
    if "vram_peak_mb" in timings:
        print(f"GPU memory: this process held up to {timings['vram_peak_mb']:.0f} MB; the device had as little as "
              f"{timings['device_free_min_mb']:.0f} MB free. Multiply the first by --workers for the search on the 3060.")
    print(f"speed-up per process: {cpu_lm / max(1e-9, gpu_lm):.1f}x; checks: "
          + ("all passed" if not failed else f"{len(failed)} failed: {failed}"))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"cells": ids, "cpu": cpu, "gpu": gpu, "cpu_s": cpu_s, "workers": args.workers, "gpu_s": gpu_s, "timings": timings,
                   "tolerance": TOL, "tolerance_kind": "share of deposits, end and every sample (ADR-132)",
                   "unfused": bool(args.unfused), "kernel_only": bool(args.kernel_only), "gaps": gaps, "failed": failed}, fh)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
