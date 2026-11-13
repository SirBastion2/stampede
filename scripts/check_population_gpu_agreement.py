#!/usr/bin/env python3
"""Compare M05 CPU vs GPU balance solve on the local CUDA device (RTX 3060 check)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine.agents.decide import cuda_device_available
from engine.agents.population import (
    clear_sample_population_cache,
    population_device_description,
    sample_population,
)
from tests.test_fakes import panic


def main() -> int:
    if not cuda_device_available():
        print("CUDA device not available; install CuPy and run on a GPU host.", file=sys.stderr)
        return 1
    os.environ.setdefault("STAMPEDE_OFFLINE", "1")
    os.environ.setdefault("STAMPEDE_CALIB_DEPOSITORS", "5000")
    scenario = panic(3060)
    os.environ["STAMPEDE_BACKEND"] = "cpu"
    clear_sample_population_cache()
    cpu_arrays, cpu_summary = sample_population(scenario)
    os.environ["STAMPEDE_BACKEND"] = "gpu"
    clear_sample_population_cache()
    gpu_arrays, gpu_summary = sample_population(scenario)
    a = np.asarray(cpu_arrays.balance, dtype=np.float64)
    b = np.asarray(gpu_arrays.balance, dtype=np.float64)
    rel = np.max(np.abs(a - b) / np.maximum(np.abs(a), 1.0))
    print(f"device={population_device_description()}")
    print(f"max_relative_error={rel:.3e}")
    print(f"cpu_top10={cpu_summary.top10_balance:.6g} gpu_top10={gpu_summary.top10_balance:.6g}")
    if rel > 1e-9:
        print("FAIL: GPU balances diverge from CPU reference beyond 1e-9 relative.", file=sys.stderr)
        return 2
    print("OK: GPU matches CPU within 1e-9 relative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
