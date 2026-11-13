"""ASSUMED pacing for the default fake pitch (not the real M06 kernel).

Headcount follows a logistic over the crisis horizon. The cash gauge drains
across the first part of that window. Dollar withdrawals track the gauge, then
a late hole lets the book close. Registry rows are ASSUMED; FIT and VERIFIED
rows are not read.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel

from contracts.events import (
    DepositRateChanged,
    DirectiveIssued,
    PolicyEnacted,
    RumorPosted,
    StatementIssued,
)
from mocks.params_access import get_param
from mocks.support import (
    CAP_WITHDRAW_SCALE,
    DEPOSIT_RATE_WITHDRAW_SCALE,
    GUARANTEE_WITHDRAW_SCALE,
    OPENING_CASH_FRACTION,
    OPENING_PRICE,
    STATEMENT_WITHDRAW_SCALE,
    deposits_base,
)
from params.loader import get_parameter

_PARAM_END = "M06 fake arc strong end withdraw share (ASSUMED)"
_PARAM_MID = "M06 fake arc logistic midpoint (ASSUMED)"
_PARAM_STEEP = "M06 fake arc logistic steepness (ASSUMED)"
_PARAM_STRONG = "M06 fake arc strong credibility (ASSUMED)"
_PARAM_CASH_AT = "M06 fake arc cash empty at horizon fraction (ASSUMED)"
_PARAM_HOLE = "M06 fake arc closure hole deposit fraction (ASSUMED)"
_PARAM_PARTIAL = "M06 fake arc partial hole deposit fraction (ASSUMED)"
_PARAM_HOLE_AT = "M06 fake arc closure hole start (ASSUMED)"
_PARAM_PARTIAL_AT = "M06 fake arc partial hole start (ASSUMED)"
_PARAM_LEAD = "M06 fake arc wavering lead minutes (ASSUMED)"
_PARAM_DRAWDOWN = "M14 fake arc price drawdown (ASSUMED)"
_PARAM_ATTN_PX = "M14 fake arc attention price weight (ASSUMED)"
_PARAM_POSTERS = "M29 fake poster credibility (ASSUMED)"
_PARAM_DEFAULT_CRED = "M29 fake default rumor credibility (ASSUMED)"
_PARAM_HORIZON = "M29 UI live crisis sim minutes"
_PARAM_DIRECTED = "M06 fake arc directed withdraw credibility (ASSUMED)"


def horizon_minutes() -> float:
    """Simulated minutes of the default fake pitch (the live crisis window)."""
    return max(1.0, get_param(_PARAM_HORIZON))


def progress(t: float) -> float:
    return max(0.0, min(1.0, t / horizon_minutes()))


def _logistic(u: float) -> float:
    midpoint = get_param(_PARAM_MID)
    steepness = get_param(_PARAM_STEEP)
    return 1.0 / (1.0 + math.exp(-steepness * (u - midpoint)))


def shaped(u: float) -> float:
    """Logistic remapped so the curve is 0 at the start and 1 at the horizon."""
    start = _logistic(0.0)
    end = _logistic(1.0)
    span = end - start
    if span <= 1e-12:
        return max(0.0, min(1.0, u))
    return max(0.0, min(1.0, (_logistic(u) - start) / span))


def rumor_strength(events: Sequence[BaseModel]) -> float:
    """Highest rumor credibility on the log, or 0 when nobody has posted."""
    strength = 0.0
    directed = False
    for event in events:
        if isinstance(event, RumorPosted):
            strength = max(strength, max(0.0, min(1.0, event.credibility)))
        elif isinstance(event, DirectiveIssued) and event.action == "withdraw":
            directed = True
    # A posted rumor keeps its own credibility. The directive floor is only a
    # stand-in when nobody has posted, so a weak rumor is not promoted.
    if directed and strength <= 0.0:
        strength = get_param(_PARAM_DIRECTED)
    return strength


def _dampener(events: Sequence[BaseModel]) -> tuple[float, float | None]:
    """One scale per lever, starting at the first dampening event. Never stacks twice."""
    scale = 1.0
    started: float | None = None
    saw_statement = False
    saw_rate = False
    saw_guarantee = False
    saw_cap = False
    for event in events:
        apply = False
        if isinstance(event, StatementIssued) and not saw_statement:
            saw_statement = True
            scale *= STATEMENT_WITHDRAW_SCALE
            apply = True
        elif isinstance(event, DepositRateChanged) and not saw_rate:
            saw_rate = True
            scale *= DEPOSIT_RATE_WITHDRAW_SCALE
            apply = True
        elif (
            isinstance(event, PolicyEnacted)
            and event.lever == "guarantee_deposits"
            and not saw_guarantee
        ):
            saw_guarantee = True
            scale *= GUARANTEE_WITHDRAW_SCALE
            apply = True
        elif isinstance(event, PolicyEnacted) and event.lever == "cap_withdrawals" and not saw_cap:
            saw_cap = True
            scale *= CAP_WITHDRAW_SCALE
            apply = True
        if apply:
            started = event.t if started is None else min(started, event.t)
    return scale, started


def withdrawn_fraction(t: float, strength: float, events: Sequence[BaseModel]) -> float:
    """Share of depositors who have left. Monotone in t for a fixed log."""
    end = get_param(_PARAM_END) * max(0.0, min(1.0, strength))
    base = end * shaped(progress(t))
    scale, started = _dampener(events)
    if started is None or t <= started:
        return base
    anchored = end * shaped(progress(started))
    return anchored + scale * (base - anchored)


def wavering_fraction(t: float, strength: float, events: Sequence[BaseModel]) -> float:
    """Depositors the curve reaches a few minutes ahead, still short of withdrawal."""
    lead = get_param(_PARAM_LEAD)
    ahead = withdrawn_fraction(t + lead, strength, events)
    now = withdrawn_fraction(t, strength, events)
    return max(0.0, ahead - now)


def _depth(strength: float) -> float:
    if strength <= 0.0:
        return 0.0
    strong = get_param(_PARAM_STRONG)
    if strong <= 0.0:
        return 1.0
    return max(0.0, min(1.0, strength / strong))


def cash_gauge_value(t: float, strength: float) -> float:
    """Operating-cash gauge in [0, 1]. A full-strength rumor is empty at the ASSUMED mark."""
    depth = _depth(strength)
    if depth <= 0.0:
        return 1.0
    empty_at = get_param(_PARAM_CASH_AT)
    if empty_at <= 0.0:
        return 0.0
    u = progress(t)
    if u >= empty_at:
        return max(0.0, 1.0 - depth)
    ratio = u / empty_at
    return max(0.0, min(1.0, 1.0 - depth * ratio * ratio))


def time_to_failure_ticks(t: float, strength: float, gauge: float) -> float | None:
    """Minutes until the gauge is empty, or 0 in the outcome window."""
    if strength <= 0.0:
        return None
    if gauge <= 0.0 or progress(t) >= 0.9:
        return 0.0
    if _depth(strength) >= 1.0:
        empty_at = get_param(_PARAM_CASH_AT) * horizon_minutes()
        return max(0.0, empty_at - t)
    return max(0.0, horizon_minutes() * 0.9 - t)


def buffer_dollars(t: float, strength: float, *, jitter: float = 1.0) -> float:
    """Dollars that drain opening cash in step with the gauge. ``jitter`` is near 1."""
    opening = deposits_base() * OPENING_CASH_FRACTION
    drained = 1.0 - cash_gauge_value(t, strength)
    return max(0.0, opening * drained * jitter)


def stress_dollars(t: float, strength: float, *, seed_factor: float) -> float:
    """Late book hole for a strong rumor, so closure still happens after the gauge empties.

    The first stage is small enough to leave securities on the book (a watch).
    The second stage exhausts them. Headcount does not use these dollars.
    """
    if _depth(strength) < 1.0:
        return 0.0
    u = progress(t)
    partial_at = get_param(_PARAM_PARTIAL_AT)
    if u < partial_at:
        return 0.0
    deposits = deposits_base()
    partial = get_param(_PARAM_PARTIAL) * deposits * seed_factor
    if u < get_param(_PARAM_HOLE_AT):
        return partial
    return partial + get_param(_PARAM_HOLE) * deposits * seed_factor


def price_quote(
    t: float, strength: float, attention: float, *, scripted: bool = True
) -> float:
    """Level price, not a per-tick compound, so the line slopes with the crowd.

    ``scripted=True`` (the demo default) applies the logistic drawdown.
    Calibration callers pass the M27 registry flag; ``False`` leaves only the
    attention discount.
    """
    gap = (
        get_param(_PARAM_DRAWDOWN) * shaped(progress(t)) * _depth(strength)
        if scripted
        else 0.0
    )
    gap += get_param(_PARAM_ATTN_PX) * max(0.0, min(1.0, attention))
    gap = max(0.0, min(0.95, gap))
    return max(0.5, OPENING_PRICE * (1.0 - gap))


def poster_credibility(poster: str) -> float:
    """ASSUMED credibility for a live rumor poster. Unknown posters use the default row."""
    table: dict[str, float] = {}
    raw = get_parameter(_PARAM_POSTERS).value
    for part in raw.split(","):
        piece = part.strip()
        if ":" not in piece:
            continue
        key, value = piece.split(":", 1)
        table[key.strip()] = float(value.strip())
    return table.get(poster.strip(), get_param(_PARAM_DEFAULT_CRED))
