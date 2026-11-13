"""Engine flags the live server may switch on for its own engine only (SVB-58, SVB-59).

A flag is on when its registry row is true, or when its row name is in the
``live_flags()`` context that ``ui/live/session.py`` opens around its real-mode
build. Calibration and seeded runs never open the context, so with the rows
false they read exactly what they read before.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from params.loader import RegistryError, get_parameter

_ACTIVE: ContextVar[frozenset[str]] = ContextVar("stampede_live_flags", default=frozenset())
_TRUE = {"true", "yes", "on", "1"}


def registry_flag(row: str) -> bool:
    try:
        return get_parameter(row).value.strip().lower() in _TRUE
    except RegistryError:
        return False


def flag_enabled(row: str) -> bool:
    """True if the registry row is true or the live context enables it."""
    return row in _ACTIVE.get() or registry_flag(row)


@contextmanager
def live_flags(rows: Iterable[str]) -> Iterator[None]:
    token = _ACTIVE.set(frozenset(rows))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def refuse_under_gpu_calibration(row: str) -> None:
    """A registry-on flag the batched GPU kernels do not mirror must not meet GPU calibration."""
    if not registry_flag(row):
        return
    if any(os.environ.get(name, "").strip() == "1" for name in ("STAMPEDE_CALIB_GPU_SIM", "STAMPEDE_CALIB_GPU_FULL")):
        raise RegistryError(f"{row} is on, but the GPU calibration kernels do not mirror it; turn it off")
