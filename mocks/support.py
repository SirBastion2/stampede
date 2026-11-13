"""Shared deterministic helpers. Registry numbers come from params_access only."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from contracts.events import (
    AssetSold,
    LiquidityDrawn,
    LossRealized,
    PaymentSettled,
    PolicyEnacted,
    RatesChanged,
)
from contracts.schemas import Scenario
from mocks.params_access import get_param

# Fractions and thresholds the architecture names but does not quantify.
# They are assumptions so the fakes can move; they are not registry rows.
OPENING_CET1 = 0.12
OPENING_CASH_FRACTION = 0.001
SECURITIES_FRACTION = 0.004
REPO_TOTAL_FRACTION = 0.008
REPO_TICK_FRACTION = 0.001
FHLB_TOTAL_FRACTION = 0.01
FHLB_TICK_FRACTION = 0.001
DISCOUNT_TOTAL_FRACTION = 0.008
DISCOUNT_TICK_FRACTION = 0.001
SAME_TICK_DRAW_SHARE = 0.35
SAME_TICK_SETTLE_FRACTION = 0.85
PANIC_BURST_TICKS = 30  # M41 decision cadence; spreads the first-day outflow
OPENING_PRICE = 100.0
ASSUMED_POLICY_RATE = 0.045
ASSUMED_CET1_FLOOR = 0.045
ASSUMED_HALT_DRAWDOWN = 0.08
ASSUMED_HALT_WINDOW = 5
ASSUMED_HEADLINE_ATTENTION = 0.30
ASSUMED_WITHDRAW_ATTENTION = 0.25
ASSUMED_COMMUNITY_UNINSURED = 0.15
ASSUMED_UNEMPLOYMENT = 0.04
GUARANTEE_WITHDRAW_SCALE = 0.35
CAP_WITHDRAW_SCALE = 0.30
STATEMENT_WITHDRAW_SCALE = 0.80
DEPOSIT_RATE_WITHDRAW_SCALE = 0.90
FUTURE_COUNT = 16
DEPOSITORS_TIER0 = 100_000
VIP_COUNT = 15
CURVE_SPREADS = (0.0, 0.002, 0.004, 0.007, 0.009, 0.011)
SEGMENT_WEIGHTS = (
    ("startup", 0.34),
    ("vc", 0.15),
    ("founder", 0.18),
    ("treasurer", 0.16),
    ("retail", 0.07),
    ("municipal", 0.06),
    ("crypto", 0.04),
)
UNINSURED_WITHDRAW_SPLIT = (
    ("startup", 0.38),
    ("vc", 0.22),
    ("founder", 0.25),
    ("treasurer", 0.15),
)
_LEVER_ALIASES = {
    "guarantee": "guarantee_deposits",
    "guarantee_all_deposits": "guarantee_deposits",
    "emergency": "emergency_lending",
    "btfp": "emergency_lending",
    "suspend": "suspend_trading",
}


def assumed_halt_minutes() -> float:
    """Registry ``M14 trading halt minutes`` (SVB-25; SEC LULD 5-minute pause)."""
    return float(get_param("M14 trading halt minutes"))


def __getattr__(name: str):
    """Keep ``ASSUMED_HALT_MINUTES`` as a live registry alias so imports still work."""
    if name == "ASSUMED_HALT_MINUTES":
        return assumed_halt_minutes()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def unit_interval(seed: int, salt: str) -> float:
    """FNV-1a in [0, 1). Stable across processes; not Python's salted hash()."""
    acc = 2166136261
    for ch in f"{seed}:{salt}":
        acc ^= ord(ch)
        acc = (acc * 16777619) & 0xFFFFFFFF
    return (acc % 1_000_000) / 1_000_000.0


def seed_scale(seed: int, salt: str) -> float:
    """Deterministic multiplier in [0.8, 1.2], used so two seeds diverge."""
    return 0.8 + 0.4 * unit_interval(seed, salt)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def bank_id_for(preset: str) -> str:
    lowered = preset.lower()
    if "community" in lowered:
        return "bank-community"
    if "crypto" in lowered:
        return "bank-crypto-fictional"
    return "bank-fvb"


def bank_name_for(preset: str) -> str:
    lowered = preset.lower()
    if "community" in lowered:
        return "Fictional Community Bank"
    if "crypto" in lowered:
        return "Fictional Harbor Bank"
    return "Fictional Valley Bank"


def is_community(preset: str) -> bool:
    return "community" in preset.lower()


def uninsured_share(preset: str) -> float:
    if is_community(preset):
        return ASSUMED_COMMUNITY_UNINSURED
    return get_param("SVB-like: uninsured share")


def deposits_base() -> float:
    # The registry has one deposit total. Other presets are not given their own.
    return get_param("SVB-like preset: deposits")


def securities_loss_ratio(policy_rate: float) -> float:
    """Monotone mark-down of long bonds as the policy rate rises.

    Duration and convexity are not in the parameter registry.
    """
    return clamp((policy_rate - 0.01) * 4.0, 0.0, 0.45)


def curve_for(policy_rate: float) -> list[float]:
    return [policy_rate + spread for spread in CURVE_SPREADS]


def policy_rate_at(path: Sequence[tuple[float, float]], t: float) -> float:
    if not path:
        return ASSUMED_POLICY_RATE
    if t <= path[0][0]:
        return path[0][1]
    for (t0, r0), (t1, r1) in zip(path, path[1:], strict=False):
        if t0 <= t <= t1 and t1 > t0:
            weight = (t - t0) / (t1 - t0)
            return r0 + (r1 - r0) * weight
    return path[-1][1]


def claim_allowed(text: str) -> bool:
    """Delegate to M34 real guardrails so fakes share the same blocklist (§10.5)."""
    from agents.guardrails import claim_allowed as guardrail_claim_allowed

    return guardrail_claim_allowed(text)


def canonical_lever(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    return _LEVER_ALIASES.get(key, key)


@dataclass(frozen=True)
class Calendar:
    business_hours: bool
    weekend: bool
    facility_open: bool
    minute_of_day: int


def calendar_at(t: float) -> Calendar:
    """t=0 is Monday 09:00. Each step is one simulated minute; ticks are not skipped."""
    absolute = 9 * 60 + int(t)
    day = absolute // (24 * 60)
    minute = absolute % (24 * 60)
    weekend = (day % 7) >= 5
    business = (9 * 60) <= minute < (17 * 60) and not weekend
    return Calendar(
        business_hours=business,
        weekend=weekend,
        facility_open=business,
        minute_of_day=minute,
    )


@dataclass(frozen=True)
class Book:
    cash: float
    securities: float
    deposits_remaining: float
    borrowings: float
    equity: float
    policy_rate: float
    cet1: float
    unpaid: float


class _Run:
    __slots__ = ("log", "n", "last", "state")


_RUNS: OrderedDict[tuple[Any, ...], _Run] = OrderedDict()
_RUNS_MAX = 16


def resume_fold(events: Sequence[BaseModel], key: tuple[Any, ...], init: Callable[[], Any],
                step: Callable[[Any, BaseModel], None]) -> Any:
    """Fold ``step`` over ``events`` (state mutated in place), resuming where the last call stopped.

    SVB-69: the bus log is append-only and the fakes re-read all of it every tick, so a run's
    step time grew from about 40 ms to 108 ms by minute 1440 (diag-16 v2). A fold resumes only
    on the very same list, not shorter than before, whose last folded event is still in place;
    anything else starts over. The same events go through the same arithmetic in the same order,
    so the result is identical to folding from scratch. A tuple or other sequence always starts over.
    """
    if not isinstance(events, list):
        state = init()
        for event in events:
            step(state, event)
        return state
    full_key = (id(events),) + key
    run = _RUNS.get(full_key)
    if run is None or run.log is not events or run.n > len(events) or (run.n and events[run.n - 1] is not run.last):
        run = _Run()
        run.log, run.n, run.last, run.state = events, 0, None, init()
        _RUNS[full_key] = run
        while len(_RUNS) > _RUNS_MAX:
            _RUNS.popitem(last=False)
    else:
        _RUNS.move_to_end(full_key)
    count = len(events)
    for index in range(run.n, count):
        step(run.state, events[index])
    if count:
        run.n, run.last = count, events[count - 1]
    return run.state


class _BookState:
    __slots__ = ("cash", "securities", "remaining", "borrowings", "equity", "rate")

    def __init__(self, deposits: float, opening_cet1: float) -> None:
        self.cash = deposits * OPENING_CASH_FRACTION
        self.securities = deposits * SECURITIES_FRACTION
        self.remaining = deposits
        self.borrowings = 0.0
        self.equity = deposits * opening_cet1
        self.rate = ASSUMED_POLICY_RATE


def _book_step(book: _BookState, event: BaseModel) -> None:
    if isinstance(event, RatesChanged):
        book.rate = event.policy_rate
    elif isinstance(event, PaymentSettled):
        book.cash -= event.amount
        book.remaining = max(0.0, book.remaining - event.amount)
    elif isinstance(event, LiquidityDrawn):
        book.cash += event.amount
        book.borrowings += event.amount
    elif isinstance(event, AssetSold):
        book.securities = max(0.0, book.securities - event.amount)
        book.cash += event.amount * (1.0 - event.price_impact)
    elif isinstance(event, LossRealized):
        book.equity -= event.amount


def fold_book(events: Sequence[BaseModel], *, deposits: float, opening_cet1: float) -> Book:
    """Replay published events into one bank book. Pure function of the log (resumed, SVB-69)."""
    state = resume_fold(events, ("book", float(deposits), float(opening_cet1)),
                        lambda: _BookState(deposits, opening_cet1), _book_step)
    cash = state.cash
    unpaid = -cash if cash < -1.0 else 0.0
    rwa = deposits if deposits > 0.0 else 1.0
    return Book(
        cash=cash,
        securities=state.securities,
        deposits_remaining=state.remaining,
        borrowings=state.borrowings,
        equity=state.equity,
        policy_rate=state.rate,
        cet1=state.equity / rwa,
        unpaid=unpaid,
    )


def _lever_step(found: list[str], event: BaseModel) -> None:
    if isinstance(event, PolicyEnacted) and event.lever not in found:
        found.append(event.lever)


def levers_in(events: Sequence[BaseModel]) -> tuple[str, ...]:
    return tuple(resume_fold(events, ("levers",), list, _lever_step))


def attention_window_ticks() -> int:
    hours = get_param("Attention memory window")
    return max(1, int(round(hours * 60.0)))


def scenario_deposits(scenario: Scenario) -> float:
    del scenario
    return deposits_base()
