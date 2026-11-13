"""SVB-08: float32 per-depositor arrays and one batched SpMM per layer."""

from __future__ import annotations

import inspect

import numpy as np
import scipy.sparse as sparse

from calibration.fit import _default_svb_scenario, _score_outflow_log, fixed_softness
from engine.agents.attention import fallback_layers, load_attention_params
from engine.agents.population import clear_sample_population_cache
from engine.gpu_full import csr_matvec_batch, update_attention_batch
from engine.gpu_sim import (
    clear_static_pack_cache,
    relative_error,
    simulate_logs,
    simulate_requested_fractions,
)

_SPREAD_RTOL = 1e-5
_REQUEST_RTOL = 1e-4
_N_DEPOSITORS = 256
_MINUTES = 20
_SEEDS = (17, 23)


def _silence_worker_threads(monkeypatch) -> None:
    monkeypatch.setattr("calibration.fit._limit_worker_threads", lambda: None)


def _gpu_full_env(monkeypatch, n_depositors: int = _N_DEPOSITORS) -> None:
    monkeypatch.setenv("STAMPEDE_CALIBRATION_ENGINE", "real")
    monkeypatch.setenv("STAMPEDE_BACKEND", "gpu")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_SIM", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_GPU_FULL", "1")
    monkeypatch.setenv("STAMPEDE_CALIB_DEPOSITORS", str(n_depositors))
    monkeypatch.setenv("STAMPEDE_OFFLINE", "1")


def _max_rel(left: np.ndarray, right: np.ndarray) -> float:
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-30)
    return float(np.max(np.abs(left - right) / scale))


def _float64_spmm(layer: object, vectors: np.ndarray) -> np.ndarray:
    """Old float64 path: one CSR product per layer, no per-lane scatter."""
    indptr = np.asarray(layer.indptr, dtype=np.int32)
    indices = np.asarray(layer.indices, dtype=np.int32)
    data = np.asarray(layer.data, dtype=np.float64)
    n_rows = int(indptr.shape[0]) - 1
    n_cols = int(vectors.shape[1])
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))
    return np.asarray((matrix @ np.asarray(vectors, dtype=np.float64).T).T, dtype=np.float64)


def test_csr_matvec_batch_is_one_spmm() -> None:
    source = inspect.getsource(csr_matvec_batch)
    assert "add.at" not in source
    assert "@" in source


def test_float32_spmm_matches_float64_spread() -> None:
    params = load_attention_params()
    layers, _ids, _banks = fallback_layers(params, "bank-svb")
    rng = np.random.default_rng(8)
    exposure = rng.random((6, params.fallback_nodes), dtype=np.float64)
    for layer in layers:
        got32 = csr_matvec_batch(np, layer, exposure)
        assert np.asarray(got32).dtype == np.float32
        got = np.asarray(got32, dtype=np.float64)
        ref = _float64_spmm(layer, exposure)
        assert _max_rel(got, ref) < _SPREAD_RTOL
    decay = 0.5 ** (1.0 / 4.0)
    attention = rng.random((6, params.fallback_nodes), dtype=np.float64)
    batched = np.asarray(
        update_attention_batch(np, attention.astype(np.float32), decay, layers, exposure.astype(np.float32)),
        dtype=np.float64,
    )
    spread = np.zeros_like(exposure)
    for layer in layers:
        spread = spread + float(layer.channel_weight) * _float64_spmm(layer, exposure)
    ref = attention * float(decay) + spread
    assert _max_rel(batched, ref) < _SPREAD_RTOL


def test_float32_requests_match_within_1e4(monkeypatch) -> None:
    _silence_worker_threads(monkeypatch)
    _gpu_full_env(monkeypatch)
    clear_sample_population_cache()
    clear_static_pack_cache()

    scenario = _default_svb_scenario(_SEEDS[0])
    vectors = [
        np.full(15, 0.5, dtype=np.float64),
        np.clip(np.full(15, 0.5) + 0.08, 0.0, 1.0),
    ]
    vectors[1][12] = 0.7
    ticks = [(float(t), 1.0) for t in range(_MINUTES)]
    logs = simulate_logs(
        scenario,
        _MINUTES,
        vectors,
        array_module="numpy",
        disable_caps=True,
        softness=fixed_softness(),
        clamp_inputs=False,
        seeds=_SEEDS,
        schedule=ticks,
        aware0=1.0,
    )
    lean = simulate_requested_fractions(
        scenario,
        ticks,
        vectors,
        _SEEDS,
        array_module="numpy",
        softness=fixed_softness(),
        clamp_inputs=False,
        aware0=1.0,
    )
    assert lean.shape == (len(vectors) * len(_SEEDS), 3)
    for lane, log in enumerate(logs):
        day1, next_open, _closed = _score_outflow_log(log, requested=True)
        assert relative_error(float(lean[lane, 0]), float(day1)) <= _REQUEST_RTOL
        assert relative_error(float(lean[lane, 1]), float(next_open)) <= _REQUEST_RTOL

    clear_static_pack_cache()
    clear_sample_population_cache()
