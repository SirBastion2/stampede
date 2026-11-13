"""M05 CuPy balance solve matches NumPy reference when CUDA is available."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from engine.agents.population import (
    _refine_balances_cupy,
    _refine_balances_numpy,
    clear_sample_population_cache,
    parse_registry_number,
    population_balance_backend,
    sample_population,
)
from engine.agents.decide import cuda_device_available
from mocks.support import deposits_base, uninsured_share
from tests.m05_population_balance_reference import solve_balances as reference_solve_balances
from tests.test_fakes import panic

_GPU_TOLERANCE = 1e-9


def _relative_close(observed: list[float], expected: list[float]) -> bool:
    a = np.asarray(observed, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    denom = np.maximum(np.abs(b), 1.0)
    return bool(np.all(np.abs(a - b) / denom <= _GPU_TOLERANCE))


@pytest.mark.parametrize("seed", [3, 19])
@pytest.mark.parametrize("count", [800, 3_000])
def test_cupy_refine_matches_numpy(seed: int, count: int, monkeypatch) -> None:
    if not cuda_device_available():
        pytest.skip("CUDA not available")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", str(count))
    clear_sample_population_cache()
    scenario = panic(seed)
    insurance_limit = parse_registry_number("Insurance limit")
    target_top10 = parse_registry_number("SVB-like: top-10 depositor balance")
    target_insured_share = 1.0 - uninsured_share(scenario.preset)
    total_deposits = deposits_base()

    rng = random.Random(scenario.seed)
    from engine.agents.population import segment_weights_for_preset

    weights = segment_weights_for_preset(scenario.preset)
    names = list(weights)
    probs = [weights[name] for name in names]
    cumulative = 0.0
    edges = []
    for prob in probs:
        cumulative += prob
        edges.append(cumulative)
    segments: list[str] = []
    for _ in range(count):
        draw = rng.random()
        for name, edge in zip(names, edges, strict=True):
            if draw <= edge:
                segments.append(name)
                break
        else:
            segments.append(names[-1])
    z_scores = [rng.gauss(0.0, 1.0) for _ in range(count)]
    seed_balances = reference_solve_balances(
        z_scores,
        segments,
        total_deposits,
        insurance_limit,
        target_insured_share,
        target_top10,
    )
    numpy_out = _refine_balances_numpy(
        seed_balances,
        segments,
        total_deposits,
        insurance_limit,
        target_insured_share,
        target_top10,
    )
    cupy_out = _refine_balances_cupy(
        seed_balances,
        segments,
        total_deposits,
        insurance_limit,
        target_insured_share,
        target_top10,
    )
    assert _relative_close(cupy_out, numpy_out)


def test_gpu_backend_falls_back_without_cuda(monkeypatch) -> None:
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    if cuda_device_available():
        assert population_balance_backend() == "gpu"
    else:
        assert population_balance_backend() == "cpu"


def test_sample_population_gpu_matches_cpu(monkeypatch) -> None:
    if not cuda_device_available():
        pytest.skip("CUDA not available")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "1200")
    scenario = panic(27)
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    clear_sample_population_cache()
    cpu_arrays, _ = sample_population(scenario)
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    clear_sample_population_cache()
    gpu_arrays, _ = sample_population(scenario)
    assert _relative_close(gpu_arrays.balance, cpu_arrays.balance)
