"""CPY-1: one switch for every array path. CuPy is the default; NumPy is the fallback and the reference.

``STAMPEDE_BACKEND`` picks the preference: ``gpu`` (the default since CPY-1) or ``cpu`` (the reference path, and
the only way to force NumPy on a machine with a GPU). A GPU path runs only when the preference is ``gpu`` and CuPy
imports with a CUDA device visible; otherwise it falls back to NumPy, as before. With ``STAMPEDE_REQUIRE_GPU=1``
the fallback is an error instead (``require_gpu``), so a run that claims the GPU can't quietly use NumPy.

Who reads it: M05 (the population's balance solve), M02 (the contact network), M03 (attention), M06 (the
depositors, per session), the batched offline runs (``ui/live/gpu_batch.py``), the calibration's batched paths
(with their own ``STAMPEDE_CALIB_GPU_*`` flags), and the live server's health report. The rest of the engine
(payments, the ledger, bonds, the liquidity ladder, the regulator, the order book) is scalar or per-order
bookkeeping: it stays on the CPU, where CuPy would only add transfer and launch cost.
"""

from __future__ import annotations

import importlib
import os
from functools import lru_cache
from typing import Any

_TRUE = {"1", "true", "yes", "on"}


def backend_preference() -> str:
    """``gpu`` unless ``STAMPEDE_BACKEND=cpu`` (CPY-1: the default was ``cpu`` before)."""
    value = os.environ.get("STAMPEDE_BACKEND", "gpu").strip().lower()
    return "cpu" if value == "cpu" else "gpu"


@lru_cache(maxsize=1)
def _cupy_with_device() -> Any | None:
    try:
        cupy = importlib.import_module("cupy")
    except Exception:
        return None
    try:
        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            return None
    except Exception:
        return None
    return cupy


def cupy_available() -> bool:
    """CuPy imports and sees a CUDA device (cached for the process)."""
    return _cupy_with_device() is not None


def gpu_active() -> bool:
    """The GPU paths run: the preference is ``gpu`` and CuPy has a device."""
    return backend_preference() == "gpu" and cupy_available()


def array_module() -> tuple[Any, str]:
    """``(cupy, "cupy")`` when the GPU paths run, else ``(numpy, "numpy")``."""
    import numpy as np

    return (_cupy_with_device(), "cupy") if gpu_active() else (np, "numpy")


GPU_WORKERS = 6   # Sebastian (Nov 5): 6 workers are fine on the one 12 GB RTX 3060; gpu_parity.py prints the memory


def default_workers() -> int:
    """Workers for the offline runs: ``STAMPEDE_WORKERS``, else 6 (up to the CPU cores less 2), GPU or not.

    The workers are what drive the GPU: each steps its lanes' per-session modules on its CPU core and sends their
    depositors to the card together, so they're needed with the GPU too.
    """
    override = os.environ.get("STAMPEDE_WORKERS", "").strip()
    if override:
        return max(1, int(override))
    return max(1, min(GPU_WORKERS, (os.cpu_count() or 2) - 2))


def vram_mb() -> dict[str, float] | None:
    """This process's CuPy pool in use and held, and the device's free and total memory, in MB (None on NumPy)."""
    cupy = _cupy_with_device()
    if cupy is None or not gpu_active():
        return None
    try:
        pool = cupy.get_default_memory_pool()
        free, total = cupy.cuda.runtime.memGetInfo()
        mb = 1024 * 1024
        return {"pool_used": pool.used_bytes() / mb, "pool_held": pool.total_bytes() / mb,
                "device_free": free / mb, "device_total": total / mb}
    except Exception:
        return None


def require_gpu_enabled() -> bool:
    return os.environ.get("STAMPEDE_REQUIRE_GPU", "").strip().lower() in _TRUE


def require_gpu(where: str = "this run") -> None:
    """Raise if ``STAMPEDE_REQUIRE_GPU=1`` and the GPU paths would fall back to NumPy."""
    if not require_gpu_enabled() or gpu_active():
        return
    why = ("STAMPEDE_BACKEND=cpu" if backend_preference() == "cpu"
           else "CuPy did not import or sees no CUDA device")
    raise RuntimeError(f"{where} requires the GPU (STAMPEDE_REQUIRE_GPU=1), but {why}")


def device_name() -> str | None:
    cupy = _cupy_with_device()
    if cupy is None:
        return None
    try:
        name = cupy.cuda.runtime.getDeviceProperties(0)["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return "cuda:0"


def backend_report() -> dict[str, Any]:
    """What each array path will use in this process, for logs, scripts and ``/api/health``."""
    from engine.agents.attention import resolve_backend
    from engine.agents.decide import active_backend_name
    from engine.agents.network import select_backend
    from engine.agents.population import population_balance_backend

    cupy = _cupy_with_device()
    return {
        "preference": backend_preference(),
        "cupy": getattr(cupy, "__version__", None) if cupy is not None else None,
        "device": device_name(),
        "M05 population solve": population_balance_backend(),
        "M02 network": select_backend()[1],
        "M03 attention": resolve_backend().name,
        "M06 depositors": active_backend_name(),
        "batched runs": array_module()[1],
        "require_gpu": require_gpu_enabled(),
    }


def format_report(report: dict[str, Any] | None = None) -> str:
    r = report or backend_report()
    parts = [f"{k} {v}" for k, v in r.items() if k not in {"preference", "cupy", "device", "require_gpu"}]
    head = f"backend: {r['preference']} preferred, " + (f"CuPy {r['cupy']} on {r['device']}" if r["cupy"] else "no CuPy device")
    return head + "; " + ", ".join(parts)
