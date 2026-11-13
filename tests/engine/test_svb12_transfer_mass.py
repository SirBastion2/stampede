"""SVB-12: transfer-mass stays on the active array module."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from engine.agents.population import _cupy_runtime, _transfer_mass_array


def _transfer_mass_array_reference(
    balances: np.ndarray,
    donors: np.ndarray,
    receivers: np.ndarray,
    amount: float,
) -> None:
    """Pre-SVB-12 sequential donor loop (host NumPy)."""
    if amount <= 0.0 or donors.size == 0 or receivers.size == 0:
        return
    remaining = amount
    donor_count = int(donors.size)
    donor_indices = donors if isinstance(donors, np.ndarray) else np.asarray(donors)
    for index in donor_indices:
        if remaining <= 0.0:
            break
        room = float(balances[index] - 1.0)
        take = min(room, remaining / max(donor_count, 1))
        if take <= 0.0:
            continue
        balances[index] -= take
        remaining -= take
    if remaining <= 0.0:
        return
    balances[receivers] += remaining / receivers.size


def _max_rel(left: np.ndarray, right: np.ndarray) -> float:
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-30)
    return float(np.max(np.abs(left - right) / scale))


def _unique_indices(rng: np.random.Generator, size: int, high: int) -> np.ndarray:
    return rng.choice(high, size=size, replace=False).astype(np.intp)


def _cases(seed: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    rng = np.random.default_rng(seed)
    cases: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

    balances = rng.uniform(2.0, 80.0, size=240)
    donors = _unique_indices(rng, 80, 240)
    receivers = _unique_indices(rng, 40, 240)
    cases.append((balances, donors, receivers, 256.125))

    tied = np.full(60, 25.0, dtype=np.float64)
    tied[::3] = 25.0
    tied[1::3] = 25.0
    tied[2::3] = 4.0
    cases.append((tied, np.arange(20, 60, dtype=np.intp), np.arange(0, 15, dtype=np.intp), 90.0))

    zeros = rng.uniform(0.0, 3.0, size=80)
    zeros[:10] = 0.0
    zeros[10:20] = 1.0
    cases.append(
        (
            zeros,
            np.arange(0, 50, dtype=np.intp),
            np.arange(50, 80, dtype=np.intp),
            12.5,
        )
    )
    cases.append(
        (
            rng.uniform(2.0, 10.0, size=40),
            np.arange(10, dtype=np.intp),
            np.arange(10, 20, dtype=np.intp),
            0.0,
        )
    )
    cases.append(
        (
            rng.uniform(2.0, 10.0, size=16),
            np.zeros(0, dtype=np.intp),
            np.arange(4, dtype=np.intp),
            8.0,
        )
    )
    cases.append(
        (
            rng.uniform(2.0, 10.0, size=16),
            np.arange(6, dtype=np.intp),
            np.zeros(0, dtype=np.intp),
            8.0,
        )
    )

    small = rng.uniform(-2.0, 5.0, size=32)
    cases.append((small, np.arange(32, dtype=np.intp), np.array([0, 1, 2], dtype=np.intp), 20.0))

    one = rng.uniform(5.0, 40.0, size=12)
    cases.append((one, np.array([3], dtype=np.intp), np.array([0, 1], dtype=np.intp), 7.25))
    return cases


@pytest.mark.parametrize("seed", [1, 7, 42, 99])
def test_transfer_mass_matches_reference_cpu(seed: int) -> None:
    for balances, donors, receivers, amount in _cases(seed):
        expected = np.array(balances, dtype=np.float64, copy=True)
        observed = np.array(balances, dtype=np.float64, copy=True)
        _transfer_mass_array_reference(expected, donors, receivers, amount)
        _transfer_mass_array(observed, donors, receivers, amount)
        assert _max_rel(observed, expected) <= 1e-12


def test_transfer_mass_source_stays_on_device() -> None:
    source = inspect.getsource(_transfer_mass_array)
    assert ".get(" not in source
    assert "add.at" not in source
    assert "float(balances" not in source


def test_transfer_mass_matches_reference_gpu() -> None:
    cp = _cupy_runtime()
    if cp is None:
        pytest.skip("CuPy device not available")
    for balances, donors, receivers, amount in _cases(11):
        expected = np.array(balances, dtype=np.float64, copy=True)
        _transfer_mass_array_reference(expected, donors, receivers, amount)
        device = cp.asarray(balances, dtype=cp.float64)
        _transfer_mass_array(device, cp.asarray(donors), cp.asarray(receivers), amount)
        assert _max_rel(cp.asnumpy(device), expected) <= 1e-12
