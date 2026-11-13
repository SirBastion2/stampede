"""Event-log metrics for calibration diagnostics (NumPy-free)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from contracts.events import BankClosed, PaymentSettled, Tick, WithdrawalRequested


@dataclass(frozen=True, slots=True)
class BindRates:
    """How often payment caps and closure bind on one log."""

    payment_cap_bind_rate: float
    closure_bind_rate: float
    payment_settled_count: int
    cap_bound_count: int
    closed: bool


def withdrawn_share_from_log(
    log: Iterable[object],
    *,
    depositor_count: float,
) -> float:
    """Share of depositors who requested a withdrawal. 0 when the count is empty."""
    if depositor_count <= 0.0:
        return 0.0
    ids: set[str] = set()
    for event in log:
        if isinstance(event, WithdrawalRequested):
            ids.add(str(event.depositor_id))
    return min(1.0, len(ids) / depositor_count)


def bind_rates_from_log(log: Iterable[object]) -> BindRates:
    """Payment-cap bind: settled prints with queued_remaining > 0. Closure: BankClosed."""
    events = list(log)
    tick_times = {float(event.t) for event in events if isinstance(event, Tick)}
    settled = [event for event in events if isinstance(event, PaymentSettled)]
    closed = any(isinstance(event, BankClosed) for event in events)
    cap_bound = sum(1 for event in settled if int(event.queued_remaining) > 0)
    settled_n = len(settled)
    if settled_n > 0:
        cap_rate = cap_bound / settled_n
    elif tick_times:
        cap_rate = 0.0
    else:
        cap_rate = 0.0
    closure_rate = 1.0 if closed else 0.0
    return BindRates(
        payment_cap_bind_rate=cap_rate,
        closure_bind_rate=closure_rate,
        payment_settled_count=settled_n,
        cap_bound_count=cap_bound,
        closed=closed,
    )
