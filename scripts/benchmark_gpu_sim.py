"""Seconds per full-scale M27 candidate: legacy engine, NumPy sim, GPU sim.

Default is the registry depositor tier and 390 simulated minutes. That legacy
leg is the ~18 minute run. This VM has no GPU; run the script on the RTX 3060.
``gpu_sim_s`` prints ``unavailable`` when CUDA is missing.

Example:
  python scripts/benchmark_gpu_sim.py
  python scripts/benchmark_gpu_sim.py --minutes 20 --depositors 2000 --skip-legacy
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

from calibration.fit import (  # noqa: E402
    _default_svb_scenario,
    load_calibration_targets,
    objective_e1,
)
from engine.agents.population import clear_sample_population_cache  # noqa: E402
from engine.gpu_sim import resolve_array_module, simulate_logs  # noqa: E402
from params.loader import get_parameter  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one M27 candidate.")
    parser.add_argument("--seed", type=int, default=3060)
    parser.add_argument("--minutes", type=int, default=390)
    parser.add_argument(
        "--depositors",
        type=int,
        default=0,
        help="Override depositor count. 0 keeps the registry tier (100k).",
    )
    parser.add_argument(
        "--skip-legacy",
        action="store_true",
        help="Do not time the per-depositor engine.",
    )
    return parser.parse_args()


def _prepare(args: argparse.Namespace) -> None:
    os.environ.setdefault("STAMPEDE_OFFLINE", "1")
    os.environ["STAMPEDE_CALIBRATION_ENGINE"] = "real"
    os.environ["STAMPEDE_BACKEND"] = "cpu"
    os.environ.pop("STAMPEDE_CALIB_GPU_SIM", None)
    if args.depositors > 0:
        os.environ["STAMPEDE_CALIB_DEPOSITORS"] = str(args.depositors)
    else:
        os.environ.pop("STAMPEDE_CALIB_DEPOSITORS", None)
    clear_sample_population_cache()


def main() -> int:
    args = _parse_args()
    _prepare(args)
    vector = np.full(12, 0.5, dtype=np.float64)
    scenario = _default_svb_scenario(args.seed)
    targets = load_calibration_targets()
    w_day1 = float(get_parameter("Calibration E1 day-1 objective weight").value)
    w_day2 = float(get_parameter("Calibration E1 two-day objective weight").value)

    if args.skip_legacy:
        print("legacy_engine_s=skipped")
    else:
        started = time.perf_counter()
        objective_e1(
            vector,
            scenario=scenario,
            steps=args.minutes,
            targets=targets,
            w_day1=w_day1,
            w_day2=w_day2,
        )
        print(f"legacy_engine_s={time.perf_counter() - started:.6f}")

    started = time.perf_counter()
    simulate_logs(scenario, args.minutes, [vector], array_module="numpy")
    print(f"cpu_numpy_sim_s={time.perf_counter() - started:.6f}")

    try:
        resolve_array_module("gpu")
    except RuntimeError:
        print("gpu_sim_s=unavailable")
        print("no CUDA device; gpu_sim_s was not timed")
        return 0
    started = time.perf_counter()
    simulate_logs(scenario, args.minutes, [vector], array_module="gpu")
    print(f"gpu_sim_s={time.perf_counter() - started:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
