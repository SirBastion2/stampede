"""M05 balance solve: vectorized implementation matches pure-Python reference."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from engine.agents.population import (
    _refine_balances_cupy,
    _refine_balances_numpy,
    _transfer_mass_array,
    _transfer_mass_cupy,
    clear_sample_population_cache,
    parse_registry_number,
    sample_population,
)
from mocks.support import deposits_base, uninsured_share
from tests.m05_population_balance_reference import solve_balances as reference_solve_balances
from tests.test_fakes import panic


def _relative_close(observed: list[float], expected: list[float]) -> bool:
    a = np.asarray(observed, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    denom = np.maximum(np.abs(b), 1.0)
    return bool(np.all(np.abs(a - b) / denom <= 1e-9))


@pytest.mark.parametrize("seed", [1, 7, 42])
@pytest.mark.parametrize("count", [500, 2_000])
def test_refine_balances_matches_reference(seed: int, count: int, monkeypatch) -> None:
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

    reference = reference_solve_balances(
        z_scores,
        segments,
        total_deposits,
        insurance_limit,
        target_insured_share,
        target_top10,
    )
    clear_sample_population_cache()
    arrays, _ = sample_population(scenario)
    assert _relative_close(arrays.balance, reference)


def test_sample_population_cache_reuses_one_solve(monkeypatch) -> None:
    calls = 0
    import engine.agents.population as pop

    original = pop._refine_balances_numpy

    def counting_refine(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pop, "_refine_balances_numpy", counting_refine)
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    clear_sample_population_cache()
    scenario = panic(99)
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "800")
    sample_population(scenario)
    sample_population(scenario)
    assert calls == 1


class _FakeDeviceArray:
    """CuPy-like wrapper: ``np.asarray`` must not succeed (use ``.get()``)."""

    __array_ufunc__ = None

    def __init__(self, data: np.ndarray) -> None:
        self._data = np.array(data, copy=True)

    def __array__(self, *_args, **_kwargs):
        raise TypeError("Implicit conversion to a NumPy array is not allowed. Please use .get()")

    def get(self) -> np.ndarray:
        return np.array(self._data, copy=True)

    @property
    def size(self) -> int:
        return int(self._data.size)

    def __setitem__(self, key, value) -> None:
        payload = value.get() if hasattr(value, "get") else value
        self._data[key] = np.asarray(payload)


def test_transfer_mass_cupy_fake_device_bit_identical() -> None:
    rng = np.random.default_rng(7)
    balances = rng.uniform(2.0, 80.0, size=240)
    donors = np.arange(80, 240, dtype=np.intp)
    receivers = np.arange(0, 40, dtype=np.intp)
    amount = 256.125
    host = np.array(balances, copy=True)
    _transfer_mass_array(host, donors, receivers, amount)
    device = _FakeDeviceArray(balances)
    _transfer_mass_cupy(device, _FakeDeviceArray(donors), _FakeDeviceArray(receivers), amount)
    np.testing.assert_array_equal(device.get(), host)


def test_refine_host_path_bit_identical_to_numpy() -> None:
    count = 180
    rng = random.Random(13)
    from engine.agents.population import segment_weights_for_preset

    scenario = panic(13)
    weights = segment_weights_for_preset(scenario.preset)
    names = list(weights)
    segments = [names[int(rng.random() * len(names)) % len(names)] for _ in range(count)]
    balances = [max(1.0, rng.gauss(20_000.0, 8_000.0)) for _ in range(count)]
    insurance_limit = parse_registry_number("Insurance limit")
    target_top10 = parse_registry_number("SVB-like: top-10 depositor balance")
    target_insured_share = 1.0 - uninsured_share(scenario.preset)
    total_deposits = deposits_base()
    host = _refine_balances_numpy(
        balances,
        segments,
        total_deposits,
        insurance_limit,
        target_insured_share,
        target_top10,
    )
    patched = _refine_balances_cupy(
        balances,
        segments,
        total_deposits,
        insurance_limit,
        target_insured_share,
        target_top10,
    )
    assert host == patched


def test_sample_population_host_and_gpu_backend_bit_identical(monkeypatch) -> None:
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", "160")
    scenario = panic(21)
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    clear_sample_population_cache()
    cpu_arrays, _ = sample_population(scenario)
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    clear_sample_population_cache()
    gpu_arrays, _ = sample_population(scenario)
    assert cpu_arrays.balance == gpu_arrays.balance
    assert cpu_arrays.segment == gpu_arrays.segment
    assert cpu_arrays.insured == gpu_arrays.insured
