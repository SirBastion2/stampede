"""Live crowd from the real engine (SVB-50, ADR-098).

With ``STAMPEDE_MODE=real`` the live server builds the calibrated module set
(registry row ``M29 live engine modules``) and the crowd panel reads engine
state instead of the scripted ``mocks/demo_arc`` curve:

* withdrew = depositors M06 has marked ``already_withdrew`` who had an uninsured,
  mobile amount to request (SVB-61);
* wavering = the rest whose M06 worry is at least ``M29 live crowd wavering
  worry threshold``;
* calm = everyone else;
* cash gauge = M08 liquid assets (cash plus securities) over their opening value. The
  real ledger keeps only a thin cash buffer and pays withdrawals by selling
  securities, so cash alone hits zero within minutes of the first settlements;
* time to failure = 0 once the bank is closed or out of cash.

The price on the market screen is the real M14 order book's ``PriceTick`` (no
extra wiring: M30 reads whatever M14 publishes). Fake mode is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from contracts.events import AttentionTick, BankClosed
from engine.modes import ModuleMode
from params.loader import get_parameter

LIVE_MODULES_ROW = "M29 live engine modules"
WAVER_ROW = "M29 live crowd wavering worry threshold"


def live_engine_modes() -> dict[str, ModuleMode]:
    """Modules that run REAL in the live server's real mode; every other module is fake."""
    raw = get_parameter(LIVE_MODULES_ROW).value
    ids = [part.strip() for part in raw.split(",") if part.strip()]
    if not ids:
        raise ValueError(f"{LIVE_MODULES_ROW} lists no modules")
    return {module_id: ModuleMode.REAL for module_id in ids}


def wavering_threshold() -> float:
    value = float(get_parameter(WAVER_ROW).value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{WAVER_ROW} must be in [0, 1]")
    return value


@dataclass(frozen=True, slots=True)
class EngineCrowd:
    """Crowd counts read from engine state at one tick."""

    depositors: int
    withdrew: int
    wavering: int
    calm: int
    cash_fraction: float
    closed: bool
    attention: float


def _module(engine: Any, module_id: str) -> Any:
    getter = getattr(engine, "get", None)
    if callable(getter):
        return getter(module_id)
    try:
        return engine[module_id]
    except (KeyError, TypeError):
        return None


def read_engine_crowd(engine: Any, *, opening_cash: float | None) -> EngineCrowd | None:
    """Counts from M06 arrays and M08 liquid assets; None until M06 has a population.

    ``opening_cash`` is the opening ``ledger_cash(engine)`` (liquid assets).
    """
    decisions = _module(engine, "M06")
    withdrew_mask = getattr(decisions, "already_withdrew", None)
    if withdrew_mask is None:
        return None
    withdrew_mask = np.asarray(withdrew_mask, dtype=np.bool_)
    n = int(withdrew_mask.shape[0])
    if n == 0:
        return None
    worry = getattr(decisions, "worry", None)
    if worry is None:
        worried = np.zeros(n, dtype=np.bool_)
    else:
        worried = np.asarray(worry, dtype=np.float64) >= wavering_threshold()
    # SVB-61: count only depositors who had something to request. In soft mode every
    # depositor's ``decided`` drifts towards 1, including fully insured ones whose
    # request is $0; they show as wavering if worried, not as withdrew.
    uninsured = getattr(decisions, "initial_uninsured", None)
    if uninsured is not None and np.asarray(uninsured).shape[0] == n:
        mobile = float(getattr(decisions, "mobile_share", 1.0))
        minimum = float(getattr(getattr(decisions, "params", None), "minimum_amount", 0.0))
        eligible = np.asarray(uninsured, dtype=np.float64) * mobile >= max(minimum, 1e-12)
        withdrew_mask = withdrew_mask & eligible
    withdrew = int(np.count_nonzero(withdrew_mask))
    wavering = int(np.count_nonzero(worried & ~withdrew_mask))
    cash = ledger_cash(engine)
    if cash is None or not opening_cash or opening_cash <= 0.0:
        cash_fraction = 1.0
    else:
        cash_fraction = max(0.0, cash / opening_cash)
    log = list(getattr(getattr(engine, "bus", None), "log", []) or [])
    closed = any(isinstance(event, BankClosed) for event in log)
    attention = 0.0
    for event in reversed(log):
        if isinstance(event, AttentionTick):
            attention = float(event.level)
            break
    return EngineCrowd(
        depositors=n,
        withdrew=withdrew,
        wavering=wavering,
        calm=n - withdrew - wavering,
        cash_fraction=cash_fraction,
        closed=closed,
        attention=attention,
    )


def ledger_cash(engine: Any) -> float | None:
    """M08 liquid assets: cash plus securities (what the bank can pay withdrawals from)."""
    state = getattr(_module(engine, "M08"), "state", None)
    if state is None:
        return None
    return float(getattr(state, "cash", 0.0)) + float(getattr(state, "securities", 0.0))


def apply_engine_crowd(crowd: Any, snapshot: EngineCrowd | None) -> None:
    """Write the engine counts onto the M29 crowd module that the frames read."""
    if snapshot is None:
        return
    crowd.calm = snapshot.calm
    crowd.wavering = snapshot.wavering
    crowd.withdrew = snapshot.withdrew
    crowd.cash_gauge = min(1.5, snapshot.cash_fraction)
    crowd.tipping_point = snapshot.attention
    if snapshot.closed or snapshot.cash_fraction <= 0.0:
        crowd.time_to_failure_ticks = 0.0
    else:
        crowd.time_to_failure_ticks = None
