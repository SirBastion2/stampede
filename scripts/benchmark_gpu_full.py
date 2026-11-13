"""Sims/second for the M27 ``--gpu-full`` generation batch.

Times one candidate×seed lane versus a 12×8 generation on the same array
module. This VM has no GPU; run it on the RTX 3060.

Example:
  python scripts/benchmark_gpu_full.py --depositors 100000 --minutes 390
  python scripts/benchmark_gpu_full.py --minutes 20 --depositors 2000 --lanes 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from calibration.fit import _default_svb_scenario  # noqa: E402
from engine.agents.population import clear_sample_population_cache  # noqa: E402
from engine.gpu_sim import resolve_array_module, simulate_logs  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark batched --gpu-full sims/s.")
    parser.add_argument("--seed", type=int, default=3060)
    parser.add_argument(
        "--minutes",
        type=int,
        default=20,
        help="Simulated minutes. Use 390 for the 3060 full-scale eval.",
    )
    parser.add_argument(
        "--depositors",
        type=int,
        default=2000,
        help="Depositor count. Use 100000 for the 3060 full-scale eval.",
    )
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--lanes", type=int, default=0, help="Override candidates×seeds.")
    return parser.parse_args()


def _prepare(args: argparse.Namespace) -> None:
    os.environ.setdefault("STAMPEDE_OFFLINE", "1")
    os.environ["STAMPEDE_CALIBRATION_ENGINE"] = "real"
    os.environ["STAMPEDE_BACKEND"] = "gpu"
    os.environ["STAMPEDE_CALIB_GPU_SIM"] = "1"
    os.environ["STAMPEDE_CALIB_GPU_FULL"] = "1"
    if args.depositors > 0:
        os.environ["STAMPEDE_CALIB_DEPOSITORS"] = str(args.depositors)
    clear_sample_population_cache()


def _vectors(count: int) -> list[np.ndarray]:
    return [np.full(12, 0.35 + 0.03 * index, dtype=np.float64) for index in range(count)]


def _rate(
    label: str,
    n_sims: int,
    elapsed: float,
    *,
    setup_s: float | None = None,
    step_s: float | None = None,
) -> None:
    sims = n_sims / elapsed if elapsed > 0.0 else 0.0
    print(f"{label}_s={elapsed:.6f}")
    if setup_s is not None:
        print(f"{label}_setup_s={setup_s:.6f}")
    if step_s is not None:
        print(f"{label}_step_s={step_s:.6f}")
    print(f"{label}_sims_per_s={sims:.4f}")


def main() -> int:
    args = _parse_args()
    _prepare(args)
    scenario = _default_svb_scenario(args.seed)
    n_cand = max(1, args.candidates)
    n_seeds = max(1, args.seeds)
    if args.lanes > 0:
        n_cand = max(1, args.lanes)
        n_seeds = 1
    seed_list = tuple(int(args.seed + offset) for offset in range(n_seeds))
    try:
        _xp, xp_name = resolve_array_module("auto")
    except Exception:
        xp_name = "cpu"
        _xp = None
    del _xp
    print(f"array_module={xp_name} depositors={args.depositors} minutes={args.minutes}")

    started = time.perf_counter()
    single = {}
    simulate_logs(scenario, args.minutes, _vectors(1), seeds=seed_list[:1], timings=single)
    _rate(
        "single_lane",
        1,
        time.perf_counter() - started,
        setup_s=single.get("setup_s"),
        step_s=single.get("step_s"),
    )

    started = time.perf_counter()
    generation = {}
    simulate_logs(scenario, args.minutes, _vectors(n_cand), seeds=seed_list, timings=generation)
    _rate(
        "generation",
        n_cand * n_seeds,
        time.perf_counter() - started,
        setup_s=generation.get("setup_s"),
        step_s=generation.get("step_s"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
