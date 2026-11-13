"""Environment flags for the live FastAPI server."""

from __future__ import annotations

import os

from engine.modes import ModuleMode


def server_port() -> int:
    raw = os.environ.get("STAMPEDE_PORT", "8000").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 8000


def default_module_mode() -> ModuleMode:
    """``STAMPEDE_MODE=real`` turns on REAL modules; default is FAKE (CPU offline)."""
    mode = os.environ.get("STAMPEDE_MODE", "fake").strip().lower()
    if mode == "real":
        return ModuleMode.REAL
    return ModuleMode.FAKE


def prefer_gpu() -> bool:
    """CuPy when importable unless ``STAMPEDE_BACKEND=cpu`` (CPY-1: GPU by default); otherwise CPU fallback."""
    from engine.xp import backend_preference

    return backend_preference() == "gpu"


def gpu_available() -> bool:
    if not prefer_gpu():
        return False
    try:
        import cupy  # noqa: F401
    except ImportError:
        return False
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False
