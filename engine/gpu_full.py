"""Opt-in full-tick GPU kernels for M27 (attention, market, payments).

Used only when ``STAMPEDE_BACKEND=gpu``, ``STAMPEDE_CALIB_GPU_SIM=1``, and
``STAMPEDE_CALIB_GPU_FULL=1`` (``--gpu-full``). CuPy runs when a device is
visible; the same kernels use NumPy otherwise. Default engine paths do not
import this module.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from pydantic import BaseModel

from contracts.events import AnalystNote, RumorPosted, Tick
from engine.agents.attention import propagate

_ON = frozenset({"1", "true", "yes", "on"})
_CSR_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def gpu_full_enabled() -> bool:
    """True when the fit asked for the whole-tick array path."""
    from engine.xp import backend_preference

    sim = os.environ.get("STAMPEDE_CALIB_GPU_SIM", "").strip().lower()
    flag = os.environ.get("STAMPEDE_CALIB_GPU_FULL", "").strip().lower()
    return backend_preference() == "gpu" and sim in _ON and flag in _ON


def configure_calibration_gpu_full(enable: bool) -> None:
    """Set the full-tick env flag. ``--gpu-full`` also turns on ``--gpu-sim``."""
    if not enable:
        return
    os.environ["STAMPEDE_CALIB_GPU_FULL"] = "1"
    os.environ["STAMPEDE_CALIB_GPU_SIM"] = "1"


def _as_host(array: Any, dtype: Any = np.float64) -> np.ndarray:
    """Host copy. Uses ``.get()`` when the array forbids implicit NumPy conversion."""
    if isinstance(array, np.ndarray):
        return np.asarray(array, dtype=dtype)
    getter = getattr(array, "get", None)
    if callable(getter):
        return np.asarray(getter(), dtype=dtype)
    return np.asarray(array, dtype=dtype)


def _to_host(array: Any) -> np.ndarray:
    return _as_host(array, np.float64)


def _f32(xp: Any) -> Any:
    return getattr(xp, "float32", np.float32)


def _is_real_cupy(xp: Any) -> bool:
    return getattr(xp, "__name__", "") == "cupy" and getattr(xp, "cuda", None) is not None


def _sparse_backend(xp: Any) -> tuple[Any, bool]:
    """``cupyx.scipy.sparse`` on CuPy, ``scipy.sparse`` on NumPy."""
    if _is_real_cupy(xp):
        try:
            import cupyx.scipy.sparse as sparse

            return sparse, True
        except Exception:
            pass
    import scipy.sparse as sparse

    return sparse, False


def _csr_device_parts(xp: Any, layer: Any, n_cols: int) -> dict[str, Any]:
    """One float32 CSR matrix per layer, cached per array module."""
    indptr = _as_host(layer.indptr, np.int32)
    indices_h = _as_host(layer.indices, np.int32)
    data_h = _as_host(layer.data, np.float32)
    key = (
        id(xp),
        int(n_cols),
        str(getattr(layer, "name", "")),
        "f32",
        indptr.tobytes(),
        indices_h.tobytes(),
    )
    cached = _CSR_CACHE.get(key)
    if cached is not None:
        return cached
    n_rows = int(indptr.shape[0]) - 1
    sparse, on_device = _sparse_backend(xp)
    if on_device:
        matrix = sparse.csr_matrix(
            (
                xp.asarray(data_h, dtype=xp.float32),
                xp.asarray(indices_h, dtype=xp.int32),
                xp.asarray(indptr, dtype=xp.int32),
            ),
            shape=(n_rows, n_cols),
        )
    else:
        matrix = sparse.csr_matrix((data_h, indices_h, indptr), shape=(n_rows, n_cols))
    parts: dict[str, Any] = {
        "xp": xp,
        "n_cols": n_cols,
        "n_rows": n_rows,
        "on_device": on_device,
        "matrix": matrix,
    }
    _CSR_CACHE[key] = parts
    return parts


def csr_matvec_batch(xp: Any, layer: Any, vectors: Any) -> Any:
    """One float32 SpMM: ``(A @ X.T).T`` for ``X`` shaped ``(candidates, nodes)``."""
    n_rows = int(layer.indptr.shape[0]) - 1
    n_candidates = int(vectors.shape[0])
    n_cols = int(vectors.shape[1]) if getattr(vectors, "ndim", 1) > 1 else 0
    dtype = _f32(xp)
    if n_rows <= 0 or int(layer.indices.size) == 0 or n_cols <= 0:
        return xp.zeros((n_candidates, max(n_rows, 0)), dtype=dtype)
    parts = _csr_device_parts(xp, layer, n_cols)
    if parts["on_device"]:
        x = xp.asarray(vectors, dtype=dtype)
        return (parts["matrix"] @ x.T).T
    host_x = _as_host(vectors, np.float32)
    return xp.asarray((parts["matrix"] @ host_x.T).T, dtype=dtype)


def update_attention_batch(
    xp: Any,
    attention: Any,
    decay: float,
    layers: Any,
    exposure: Any,
) -> Any:
    """``a ← decay(a) + Σ_layer w · A_layer · exposure`` for a candidate batch."""
    spread = xp.zeros_like(exposure)
    for layer in layers:
        spread = spread + float(layer.channel_weight) * csr_matvec_batch(xp, layer, exposure)
    return attention * float(decay) + spread


def step_attention_group(engines: list[Any], t: float, xp: Any, dt: float = 1.0) -> None:
    """Advance every M03 with one device-resident attention update.

    ``dt`` is accepted so the caller can thread the simulation clock. Rumor
    decay already uses ``module._decay`` (set from ``dt`` in gpu_sim). News
    decay lives in ``gpu_full_io.step_news`` and is applied after withdrawals.
    """
    del dt
    modules = [engine["M03"] for engine in engines]
    layers = getattr(modules[0], "_layers", None)
    if layers is None or modules[0].scenario is None:
        return
    # SVB-66: every lane spreads its own rumors. Reposts read the lane's own attention
    # (``AttentionEngine._draw_reposts``), so a seed-shared spread coupled every candidate
    # to the seed's first lane.
    exposures, mutated_rows = _exposures_per_lane(modules, t)
    dtype = _f32(xp)
    device = getattr(modules[0], "_gpu_full_attention", None)
    if device is None or int(device.shape[0]) != len(modules):
        device = xp.asarray(np.stack([m.attention for m in modules]), dtype=dtype)
        modules[0]._gpu_full_attention = device
    stacked = update_attention_batch(
        xp,
        device,
        float(modules[0]._decay),
        layers,
        xp.asarray(np.stack(exposures), dtype=dtype),
    )
    modules[0]._gpu_full_attention = stacked
    host = _to_host(stacked)
    for engine, module, row, mutated in zip(engines, modules, host, mutated_rows, strict=True):
        module.attention = np.asarray(row, dtype=np.float64)
        seed = int(module.scenario.seed)
        for event in [*mutated, *module._attention_ticks(t, seed)]:
            engine.bus.publish(event)
    from engine.gpu_full_io import apply_composed_attention

    apply_composed_attention(engines, t)


def _exposures_per_lane(
    modules: list[Any], t: float
) -> tuple[list[np.ndarray], list[list[BaseModel]]]:
    """Inbox, shock, and rumor spread for each lane with its own attention (SVB-66)."""
    exposures: list[np.ndarray] = []
    mutated_rows: list[list[BaseModel]] = []
    for module in modules:
        exposure, mutated = _attention_exposure(module, t)
        exposures.append(exposure)
        mutated_rows.append(mutated)
    return exposures, mutated_rows


def _attention_exposure(module: Any, t: float) -> tuple[np.ndarray, list[BaseModel]]:
    """Per-lane inbox + shock + rumor spread."""
    shock, posted = _collect_attention_inbox(module)
    rumor_exposure, mutated = _spread_posted_rumors(module, t, posted)
    return shock + rumor_exposure, mutated


def _collect_attention_inbox(module: Any) -> tuple[np.ndarray, list[tuple[Any, int]]]:
    """Advance ``_cursor``, apply notes, and return this lane's shock and posted rumors."""
    new_events = module.inbox[module._cursor :]
    module._cursor = len(module.inbox)
    shock = np.zeros(module.attention.shape[0], dtype=np.float32)
    posted: list[tuple[Any, int]] = []
    for event in new_events:
        if isinstance(event, RumorPosted):
            posted.append((module._rumor_for(event), module._node_index(event.source_node)))
        elif isinstance(event, AnalystNote):
            module._apply_note(event)
        elif isinstance(event, Tick):
            continue
        else:
            amount = module._shock_amount(event)
            if amount is None:
                continue
            bank_value = getattr(event, "bank_id", None)
            bank = bank_value if isinstance(bank_value, str) else None
            shock[module._hub_for(bank)] += amount
    return shock, posted


def _spread_posted_rumors(
    module: Any, t: float, posted: list[tuple[Any, int]]
) -> tuple[np.ndarray, list[BaseModel]]:
    """Posted impulses, reposts, propagate, holders, and reach for one lane."""
    scenario = module.scenario
    layers = module._layers
    impulse = module.params.rumor_exposure
    for rumor, src in posted:
        rumor.exposure[src] += rumor.credibility * impulse
        rumor.holders[src] = True
    mutated = module._draw_reposts(t, scenario.seed)
    combined = np.zeros(module.attention.shape[0], dtype=np.float32)
    for rumor in module._rumors:
        combined = combined + rumor.exposure
        holders = propagate(layers, rumor.exposure, backend=module.backend)
        rumor.holders = np.logical_or(rumor.holders, holders > 0.0)
        rumor.reach = int(np.count_nonzero(rumor.holders))
        rumor.exposure.fill(0.0)
    return combined, mutated
