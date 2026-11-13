"""Compare one M27 candidate: legacy engine vs batched GPU sim.

Prints the max relative error on the E1 day-1 fraction, the two-day fraction,
and the E1 objective. Exits 0 when that error is within the registry tolerance.

On this VM there is no GPU. Run it on the RTX 3060. CuPy is used for the
depositor kernel when a CUDA device is visible; otherwise the script compares
the NumPy array module and says so.

Full scale (registry tier, 390 minutes) is the default. The legacy engine is
the slow leg. Override with --minutes and --depositors for a smoke check.

Example:
  STAMPEDE_CALIBRATION_ENGINE=real python scripts/check_gpu_sim_agreement.py
  python scripts/check_gpu_sim_agreement.py --minutes 30 --depositors 5000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from calibration.fit import (  # noqa: E402
    _default_svb_scenario,
    _objective_from_fractions,
    _score_outflow_log,
    load_calibration_targets,
    measure_outflow,
)
from engine.agents.population import clear_sample_population_cache  # noqa: E402
from engine.gpu_sim import (  # noqa: E402
    e1_relative_tolerance,
    relative_error,
    resolve_array_module,
    simulate_logs,
)
from params.loader import get_parameter  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU engine vs M27 GPU-sim agreement.")
    parser.add_argument("--seed", type=int, default=3060)
    parser.add_argument(
        "--minutes",
        type=int,
        default=390,
        help="Simulated minutes (default 390, the full-scale fit horizon).",
    )
    parser.add_argument(
        "--depositors",
        type=int,
        default=0,
        help="Override depositor count. 0 keeps the registry tier (100k).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("STAMPEDE_OFFLINE", "1")
    os.environ["STAMPEDE_CALIBRATION_ENGINE"] = "real"
    # NumPy population for both legs so the balance solve is not a second source
    # of error. M03 still uses its own backend (Philox on CPU, CuPy RNG if CUDA
    # is visible) inside AttentionEngine, which both legs call.
    os.environ["STAMPEDE_BACKEND"] = "cpu"
    os.environ.pop("STAMPEDE_CALIB_GPU_SIM", None)
    if args.depositors > 0:
        os.environ["STAMPEDE_CALIB_DEPOSITORS"] = str(args.depositors)
    else:
        os.environ.pop("STAMPEDE_CALIB_DEPOSITORS", None)
    clear_sample_population_cache()

    vector = np.full(12, 0.5, dtype=np.float64)
    scenario = _default_svb_scenario(args.seed)
    targets = load_calibration_targets()
    w_day1 = float(get_parameter("Calibration E1 day-1 objective weight").value)
    w_day2 = float(get_parameter("Calibration E1 two-day objective weight").value)
    legacy = measure_outflow(scenario, steps=args.minutes, param_vector=vector)
    legacy_obj = _objective_from_fractions(
        legacy[0],
        legacy[1],
        targets=targets,
        w_day1=w_day1,
        w_day2=w_day2,
    )

    try:
        xp, xp_name = resolve_array_module("gpu")
        del xp
        array_module = "gpu"
    except RuntimeError:
        xp_name = "cpu"
        array_module = "numpy"
    logs = simulate_logs(scenario, args.minutes, [vector], array_module=array_module)
    got = _score_outflow_log(logs[0])
    got_obj = _objective_from_fractions(
        got[0],
        got[1],
        targets=targets,
        w_day1=w_day1,
        w_day2=w_day2,
    )
    errors = {
        "day1_fraction": relative_error(got[0], legacy[0]),
        "two_day_fraction": relative_error(got[1], legacy[1]),
        "objective_e1": relative_error(got_obj, legacy_obj),
    }
    worst = max(errors.values())
    tolerance = e1_relative_tolerance()
    print(f"array_module={xp_name}")
    print(f"legacy_day1={legacy[0]:.12g} gpu_sim_day1={got[0]:.12g}")
    print(f"legacy_two_day={legacy[1]:.12g} gpu_sim_two_day={got[1]:.12g}")
    print(f"legacy_objective={legacy_obj:.12g} gpu_sim_objective={got_obj:.12g}")
    print(f"legacy_closed={legacy[2]} gpu_sim_closed={got[2]}")
    for name, err in errors.items():
        print(f"rel_error_{name}={err:.6g}")
    print(f"max_rel_error={worst:.6g}")
    print(f"tolerance={tolerance:.6g}")
    if xp_name != "gpu":
        print("no CUDA device; compared the NumPy array module to the CPU engine")
    if got[2] is not legacy[2]:
        print("closed flag differs")
        return 1
    if worst > tolerance:
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
