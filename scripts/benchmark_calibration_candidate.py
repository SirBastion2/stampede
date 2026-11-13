#!/usr/bin/env python3
"""Time one M27 candidate eval (reduced steps/depositors) on CPU NumPy vs optional GPU."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from calibration.fit import (
    _default_svb_scenario,
    eval_simulation_minutes,
    load_calibration_targets,
    objective_e1,
)
from engine.agents.decide import cuda_device_available
from engine.agents.population import clear_sample_population_cache, population_balance_backend


def _one_candidate() -> float:
    targets = load_calibration_targets()
    steps = eval_simulation_minutes()
    scenario = _default_svb_scenario(3060)
    vector = np.full(12, 0.5, dtype=np.float64)
    t0 = time.perf_counter()
    objective_e1(
        vector,
        scenario=scenario,
        steps=steps,
        targets=targets,
        w_day1=1.0,
        w_day2=1.0,
    )
    return time.perf_counter() - t0


def main() -> None:
    os.environ.setdefault("STAMPEDE_OFFLINE", "1")
    os.environ.setdefault("STAMPEDE_CALIBRATION_ENGINE", "real")
    os.environ.setdefault("STAMPEDE_CALIB_EVAL_MINUTES", "10")
    os.environ.setdefault("STAMPEDE_CALIB_DEPOSITORS", "10000")

    os.environ["STAMPEDE_BACKEND"] = "cpu"
    clear_sample_population_cache()
    cpu_seconds = _one_candidate()
    print(f"cpu_numpy_seconds={cpu_seconds:.3f} m05={population_balance_backend()}")

    if cuda_device_available():
        os.environ["STAMPEDE_BACKEND"] = "gpu"
        clear_sample_population_cache()
        gpu_seconds = _one_candidate()
        print(f"gpu_cupy_seconds={gpu_seconds:.3f} m05={population_balance_backend()}")
    else:
        print("gpu_cupy_seconds=unmeasured (no CUDA in this environment)")


if __name__ == "__main__":
    main()
