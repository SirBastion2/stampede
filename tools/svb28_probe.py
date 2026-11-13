"""SVB-28 saturation probe: does requested outflow respond to the behavioral vector?

Sets vector coordinates directly (no fit projection), so a "low contact weight" run really has a
low contact weight. Prints requests by day-one close and by next open, and the day-one stock ratio,
averaged over the shared seeds, for each probe vector. Run on the 3060 from the repo root:

    STAMPEDE_CALIBRATION_ENGINE=real STAMPEDE_BACKEND=gpu STAMPEDE_CALIB_GPU_SIM=1 \
    STAMPEDE_CALIB_GPU_FULL=1 STAMPEDE_OFFLINE=1 PYTHONPATH=. python tools/svb28_probe.py

Pass --legacy to compare with the per-tick take (sets the decision mode to 'repeat' in-process).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

_PROBES = {
    "prior (all 0.5)": {},
    "contact weight low (idx1=0.1)": {1: 0.1},
    "contact weight high (idx1=0.9)": {1: 0.9},
    "feedback gain low (idx12=0.1)": {12: 0.1},
    "worry bias low (idx7=0.2)": {7: 0.2},
    "mobile share low (idx15=0.2)": {15: 0.2},
    "mobile share high (idx15=0.8)": {15: 0.8},
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=3060)
    parser.add_argument("--legacy", action="store_true", help="per-tick take ('repeat') for comparison")
    args = parser.parse_args(argv)

    import engine.agents.decide as decide
    import engine.gpu_sim as gpu_sim
    from calibration.fit import (
        _default_svb_scenario,
        default_tick_schedule,
        fixed_softness,
        shared_eval_seeds,
    )

    if args.legacy:
        gpu_sim.withdrawal_decision_once = lambda path=None: False
        decide.withdrawal_decision_once = lambda path=None: False
    from params.loader import get_parameter

    dim = int(float(get_parameter("Calibration behavioral parameter dimension").value))
    seeds = list(shared_eval_seeds(args.seed))
    ticks = default_tick_schedule()
    vectors = []
    for changes in _PROBES.values():
        vector = np.full(dim, 0.5)
        for index, value in changes.items():
            vector[index] = value
        vectors.append(vector)
    started = time.perf_counter()
    rows = gpu_sim.simulate_requested_fractions(
        _default_svb_scenario(args.seed), ticks, vectors, seeds, softness=fixed_softness()
    )
    n = len(vectors)
    per_vector = rows.reshape(len(seeds), n, 3).mean(axis=0)
    mode = "repeat (legacy)" if args.legacy else "once"
    print(f"decision mode {mode}; {len(seeds)} seeds; {len(ticks)} ticks; {time.perf_counter() - started:.1f} s")
    print(f"{'probe':34s} {'day1 req':>9s} {'next req':>9s} {'stock':>7s}")
    names = list(_PROBES)
    by_name = {name: row for name, row in zip(names, per_vector)}
    for name, (day1, nxt, stock) in zip(names, per_vector):
        print(f"{name:34s} {day1:9.4f} {nxt:9.4f} {stock:7.3f}")
    prior_day1 = float(by_name["prior (all 0.5)"][0])
    checks = (
        ("contact weight", abs(float(by_name["contact weight high (idx1=0.9)"][0]) - float(by_name["contact weight low (idx1=0.1)"][0]))),
        ("feedback gain", float(by_name["prior (all 0.5)"][0]) - float(by_name["feedback gain low (idx12=0.1)"][0])),
        ("worry bias", abs(prior_day1 - float(by_name["worry bias low (idx7=0.2)"][0]))),
    )
    failed: list[str] = []
    for label, delta in checks:
        print(f"day-1 Δ {label}: {delta:.4f}")
        if delta <= 0.002:
            print(f"FLAT: {label}")
            failed.append(label)
    feedback_monotone = float(by_name["prior (all 0.5)"][0]) >= float(by_name["feedback gain low (idx12=0.1)"][0])
    if not feedback_monotone:
        print("FLAT: feedback gain")
        if "feedback gain" not in failed:
            failed.append("feedback gain")
    if failed:
        print("FAIL: " + ", ".join(failed))
        return 1
    print("pass: contact weight, feedback gain and worry bias each move day-1 requests (>0.002; feedback monotone)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
