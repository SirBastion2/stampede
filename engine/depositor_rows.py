"""Which registry rows hold the depositor defaults (SVB-52 / D3, ADR-099).

The SVB fit (4ed6ad9) persisted four depositor rows. Non-bank presets must not
inherit them, so inside ``nonbank_depositor_rows()`` the loaders read the
dedicated non-bank rows instead. The map is held in a ``ContextVar``: it is
inert unless a caller activates it, so every SVB and seeded path reads the same
rows as before.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

SVB_TO_NONBANK: dict[str, str] = {
    "M06 worry bias (ASSUMED)": "Non-bank M06 worry bias (ASSUMED)",
    "Depositor mobile share of uninsured deposits (FIT)": (
        "Non-bank depositor mobile share of uninsured deposits (ASSUMED)"
    ),
    "M07 decision-to-request rate (FIT)": "Non-bank M07 decision-to-request rate (ASSUMED)",
    "M07 capacity multiplier (FIT)": "Non-bank M07 capacity multiplier (ASSUMED)",
}

_ACTIVE: ContextVar[dict[str, str] | None] = ContextVar("stampede_depositor_rows", default=None)


def resolve(name: str) -> str:
    """The row to read for ``name`` in the current context (``name`` when inactive)."""
    mapping = _ACTIVE.get()
    if not mapping:
        return name
    return mapping.get(name, name)


def nonbank_rows_active() -> bool:
    return bool(_ACTIVE.get())


@contextmanager
def nonbank_depositor_rows() -> Iterator[None]:
    """Read the non-bank depositor rows for the duration of the block."""
    token = _ACTIVE.set(dict(SVB_TO_NONBANK))
    try:
        yield
    finally:
        _ACTIVE.reset(token)
