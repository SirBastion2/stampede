"""SVB-68: M02's advisor sample repeats for a fixed seed on every backend.

CuPy's Generator has no ``permutation`` (ADR-027), so on a CUDA machine ``_permutation`` fell back to
``cupy.random.permutation``: CuPy's global, unseeded RNG. Every build then sampled different advisors,
so the professional layer, the vc-partner role node and M03's attention changed between two seed-3
engines (diag-12: 45 of 45 builds, the SVB-57 test failing about half the time).

``_CupyLikeGenerator`` reproduces that on CPU: a seeded NumPy Generator without ``permutation``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

import engine.agents.network as network_module
from engine.agents.network import PROFESSIONAL, as_numpy, build_information_network, load_network_params
from tests.test_m03_uses_m02_network import _engine, _post


class _CupyLikeGenerator:
    """The part of ``cupy.random.Generator`` M02 uses, backed by NumPy. No ``permutation``."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def integers(self, *args: Any, **kwargs: Any) -> Any:
        return self._rng.integers(*args, **kwargs)

    def random(self, *args: Any, **kwargs: Any) -> Any:
        return self._rng.random(*args, **kwargs)


def _segments(n: int) -> list[str]:
    """Same mix as test_m02_network's fixture; large enough that the advisor pool is sampled."""
    n_founder, n_vc, n_treasurer = int(0.18 * n), int(0.15 * n), int(0.16 * n)
    return (["founder"] * n_founder + ["vc"] * n_vc + ["treasurer"] * n_treasurer
            + ["startup"] * (n - n_founder - n_vc - n_treasurer))


def _cupy_like(monkeypatch: pytest.MonkeyPatch) -> None:
    """NumPy arrays with a CuPy-style generator. The backend is pinned to NumPy as well: on a CUDA
    machine select_backend() returns CuPy, and CuPy kernels reject the NumPy draws this generator makes."""
    monkeypatch.setattr(network_module, "select_backend", lambda: (np, "cpu"))
    monkeypatch.setattr(network_module, "_generator", lambda xp, backend, seed: _CupyLikeGenerator(seed))


def _pool_is_sampled(segments: list[str]) -> bool:
    params = load_network_params()
    pool = sum(1 for s in segments if s in ("founder", "vc")) + params.media_count
    return pool > params.advisor_count


def _advisors_and_professional(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    net = build_information_network(_segments(12000), seed=seed)
    layer = net.layers[PROFESSIONAL]
    return as_numpy(net.is_advisor), as_numpy(layer.indptr), as_numpy(layer.indices)


def test_fallback_permutation_is_seeded_and_leaves_global_rngs_alone() -> None:
    before = np.random.get_state(legacy=False)["state"]["key"].copy()
    a = network_module._permutation(np, _CupyLikeGenerator(3), 5000)
    b = network_module._permutation(np, _CupyLikeGenerator(3), 5000)
    c = network_module._permutation(np, _CupyLikeGenerator(4), 5000)
    np.testing.assert_array_equal(as_numpy(a), as_numpy(b))
    np.testing.assert_array_equal(np.sort(as_numpy(a)), np.arange(5000))
    assert not np.array_equal(as_numpy(a), as_numpy(c))
    np.testing.assert_array_equal(np.random.get_state(legacy=False)["state"]["key"], before)


def test_cpu_permutation_is_unchanged() -> None:
    """The CPU stream (and so every CPU log and seeded hash) keeps NumPy's own permutation."""
    np.testing.assert_array_equal(
        network_module._permutation(np, np.random.default_rng(7), 50), np.random.default_rng(7).permutation(50)
    )


def test_advisor_sample_repeats_with_a_cupy_like_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    _cupy_like(monkeypatch)
    assert _pool_is_sampled(_segments(12000))
    first, second, other = _advisors_and_professional(3), _advisors_and_professional(3), _advisors_and_professional(4)
    for x, y in zip(first, second):
        np.testing.assert_array_equal(x, y)
    assert int(first[0].sum()) == load_network_params().advisor_count
    assert not np.array_equal(first[0], other[0])


def test_advisor_sample_repeats_on_this_machines_backend() -> None:
    """On a CUDA machine this is the real CuPy path, which the CPU-only CI never runs."""
    first, second = _advisors_and_professional(3), _advisors_and_professional(3)
    for x, y in zip(first, second):
        np.testing.assert_array_equal(x, y)


def test_flag_on_is_deterministic_with_a_cupy_like_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    """SVB-57's test, forced onto the GPU-style generator, where it used to flake."""
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")
    monkeypatch.setenv("STAMPEDE_MODE", "fake")
    monkeypatch.setenv("STAMPEDE_BACKEND", "cpu")
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_SIM", raising=False)
    monkeypatch.delenv("STAMPEDE_CALIB_GPU_FULL", raising=False)
    _cupy_like(monkeypatch)
    runs = [_post(_engine(on=True, role_nodes=True, seed=3), "vc-partner") for _ in range(3)]
    for run in runs[1:]:
        np.testing.assert_array_equal(runs[0], run)
