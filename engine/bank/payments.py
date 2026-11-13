"""M07 payments and withdrawal channels (§5, §18, §19)."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar

from pydantic import BaseModel

from contracts.events import BankClosed, PaymentSettled, Tick, VIPDecision, WithdrawalRequested
from contracts.schemas import Scenario
from engine.core.clock import (
    CalendarConfig,
    CalendarFlags,
    _parse_registry_minutes,
    calendar_flags,
    load_calendar_config,
)
from mocks.bus import EventBus
from mocks.support import bank_id_for
from params.loader import RegistryError, list_by_module

_APP = "app"
_WIRE = "wire"
_ACH = "ach"
_BRANCH = "branch"

_CHANNEL_ALIASES = {
    "app": _APP,
    "online": _APP,
    "mobile": _APP,
    "wire": _WIRE,
    "fedwire": _WIRE,
    "ach": _ACH,
    "transfer": _ACH,
    "branch": _BRANCH,
    "teller": _BRANCH,
}


@dataclass(frozen=True, slots=True)
class ChannelLimits:
    """Per-channel capacity and limits loaded from the registry (§8.1)."""

    app_daily_limit: float
    wire_capacity_per_tick: float
    ach_batch_capacity: float
    ach_batch_close_minute: int
    branch_capacity_per_tick: float
    vip_default_channel: str


@dataclass
class PaymentState:
    """Queues and daily counters for the payment processor."""

    wire_queue: deque[float] = field(default_factory=deque)
    ach_pending: deque[float] = field(default_factory=deque)
    branch_queue: deque[float] = field(default_factory=deque)
    app_settled_today: float = 0.0
    calendar_day: int = -1


def normalize_channel(channel: str) -> str:
    """Map event channel strings onto the four processor channels."""
    key = channel.strip().lower().replace(" ", "_")
    return _CHANNEL_ALIASES.get(key, _WIRE)


def _parse_money(value: str) -> float:
    text = value.strip().lower().replace(",", "").replace("$", "")
    match = re.match(r"[><=\s]*([0-9]*\.?[0-9]+)\s*([a-z%]*)", text)
    if match is None:
        raise RegistryError(f"cannot parse money from {value!r}")
    number = float(match.group(1))
    unit = match.group(2)
    if unit.startswith("b"):
        return number * 1_000_000_000.0
    if unit.startswith("m"):
        return number * 1_000_000.0
    if unit.startswith("k"):
        return number * 1_000.0
    return number


_STOP_AFTER_CLOSURE = "M07 stop settlement after closure"


def _gpu_calibration_requested() -> bool:
    import os

    return any(
        os.environ.get(name, "").strip() == "1"
        for name in ("STAMPEDE_CALIB_GPU_SIM", "STAMPEDE_CALIB_GPU_FULL")
    )


def stop_settlement_after_closure() -> bool:
    """SVB-53 (D4) flag, default off: no PaymentSettled after this bank's BankClosed."""
    from params.loader import get_parameter

    try:
        value = get_parameter(_STOP_AFTER_CLOSURE).value
    except RegistryError:
        return False
    return str(value).strip().lower() in {"true", "yes", "on", "1"}


def load_payment_params(
    *,
    calendar: CalendarConfig | None = None,
) -> ChannelLimits:
    """Load M07 ASSUMED rows from ``params/registry.csv``."""
    rows = {row.parameter: row.value for row in list_by_module("M07")}
    required = (
        "M07 app transfer daily limit",
        "M07 wire capacity per tick",
        "M07 ACH batch capacity",
        "M07 branch capacity per tick",
        "M07 VIP withdrawal channel",
    )
    missing = [name for name in required if name not in rows]
    if missing:
        raise RegistryError(f"M07 registry rows missing: {', '.join(missing)}")
    cfg = calendar or load_calendar_config()
    close_key = "M07 ACH batch at business close minute"
    close_minute = (
        _parse_registry_minutes(rows[close_key])
        if close_key in rows
        else cfg.business_close_minute
    )
    vip = normalize_channel(rows["M07 VIP withdrawal channel"])
    from engine.depositor_rows import resolve as _resolve_row

    raw_mult = rows.get(_resolve_row("M07 capacity multiplier (FIT)"), "1.0")
    try:
        multiplier = float(str(raw_mult).strip())
    except ValueError:
        multiplier = 1.0
    return ChannelLimits(
        app_daily_limit=_parse_money(rows["M07 app transfer daily limit"]) * multiplier,
        wire_capacity_per_tick=_parse_money(rows["M07 wire capacity per tick"]) * multiplier,
        ach_batch_capacity=_parse_money(rows["M07 ACH batch capacity"]) * multiplier,
        ach_batch_close_minute=close_minute,
        branch_capacity_per_tick=_parse_money(rows["M07 branch capacity per tick"]),
        vip_default_channel=vip,
    )


def _roll_app_daily(state: PaymentState, flags: CalendarFlags) -> None:
    if flags.day_index != state.calendar_day:
        state.calendar_day = flags.day_index
        state.app_settled_today = 0.0


def enqueue_withdrawal(
    state: PaymentState,
    amount: float,
    channel: str,
    limits: ChannelLimits,
    flags: CalendarFlags,
) -> tuple[PaymentState, list[tuple[str, float]]]:
    """Accept a withdrawal; return instant settlements and updated queues."""
    if amount <= 0.0:
        return state, []
    _roll_app_daily(state, flags)
    instant: list[tuple[str, float]] = []
    normalized = normalize_channel(channel)
    if normalized == _APP:
        remaining_limit = max(0.0, limits.app_daily_limit - state.app_settled_today)
        now = min(amount, remaining_limit)
        if now > 0.0:
            instant.append((_APP, now))
            state.app_settled_today += now
        leftover = amount - now
        if leftover > 0.0:
            state.wire_queue.append(leftover)
        return state, instant
    if normalized == _WIRE:
        state.wire_queue.append(amount)
    elif normalized == _ACH:
        state.ach_pending.append(amount)
    else:
        state.branch_queue.append(amount)
    return state, instant


def _settle_wire(
    state: PaymentState,
    limits: ChannelLimits,
    flags: CalendarFlags,
) -> float:
    if not flags.wire_window:
        return 0.0
    capacity = limits.wire_capacity_per_tick
    settled = 0.0
    while capacity > 0.0 and state.wire_queue:
        head = state.wire_queue[0]
        take = min(head, capacity)
        if take <= 0.0:
            state.wire_queue.popleft()
            continue
        settled += take
        capacity -= take
        if take >= head - 1e-9:
            state.wire_queue.popleft()
        else:
            state.wire_queue[0] = head - take
    return settled


def _settle_branch(
    state: PaymentState,
    limits: ChannelLimits,
    flags: CalendarFlags,
) -> float:
    if not flags.business_hours:
        return 0.0
    capacity = limits.branch_capacity_per_tick
    settled = 0.0
    while capacity > 0.0 and state.branch_queue:
        head = state.branch_queue[0]
        take = min(head, capacity)
        if take <= 0.0:
            state.branch_queue.popleft()
            continue
        settled += take
        capacity -= take
        if take >= head - 1e-9:
            state.branch_queue.popleft()
        else:
            state.branch_queue[0] = head - take
    return settled


def _settle_ach(
    state: PaymentState,
    limits: ChannelLimits,
    flags: CalendarFlags,
) -> float:
    if flags.weekend or not state.ach_pending:
        return 0.0
    if flags.minute_of_day < limits.ach_batch_close_minute:
        return 0.0
    total = sum(state.ach_pending)
    pay = min(total, limits.ach_batch_capacity)
    if pay <= 0.0:
        return 0.0
    remaining = pay
    new_pending: deque[float] = deque()
    for item in state.ach_pending:
        if remaining <= 0.0:
            new_pending.append(item)
            continue
        take = min(item, remaining)
        remaining -= take
        left = item - take
        if left > 1e-9:
            new_pending.append(left)
    state.ach_pending = new_pending
    return pay


def settle_tick(
    state: PaymentState,
    flags: CalendarFlags,
    limits: ChannelLimits,
) -> dict[str, float]:
    """Drain queues for this tick; amounts keyed by channel."""
    _roll_app_daily(state, flags)
    out: dict[str, float] = {}
    wire = _settle_wire(state, limits, flags)
    if wire > 0.0:
        out[_WIRE] = wire
    branch = _settle_branch(state, limits, flags)
    if branch > 0.0:
        out[_BRANCH] = branch
    ach = _settle_ach(state, limits, flags)
    if ach > 0.0:
        out[_ACH] = ach
    return out


class PaymentsModule:
    """Payment channel processor (§19 M07).

    ``step(t)`` consumes Tick, WithdrawalRequested, and VIPDecision; publishes
    ``PaymentSettled`` for each channel that settles this tick.
    """

    publishes: ClassVar[tuple[str, ...]] = ("PaymentSettled",)
    consumes: ClassVar[tuple[str, ...]] = ("Tick", "WithdrawalRequested", "VIPDecision")

    def __init__(self, module_id: str = "M07") -> None:
        self.module_id = module_id
        self.inbox: list[BaseModel] = []
        self.published: list[BaseModel] = []
        self.scenario: Scenario | None = None
        self.bus: EventBus | None = None
        self._cursor = 0
        self._seq = 0
        self.limits = load_payment_params()
        self.calendar = load_calendar_config()
        self.state = PaymentState()
        # SVB-53 (D4): off by default; on, nothing settles after this bank's BankClosed.
        self.stop_after_closure = stop_settlement_after_closure()
        if self.stop_after_closure and _gpu_calibration_requested():
            raise RegistryError(
                f"{_STOP_AFTER_CLOSURE} is on, but engine/gpu_full.py's payments kernel does not "
                "read it; turn it off for --gpu-sim / --gpu-full runs (ADR-100)"
            )
        self.closed = False
        self._log_cursor = 0

    def attach(self, bus: EventBus, scenario: Scenario) -> None:
        self.bus = bus
        self.scenario = scenario
        self.state = PaymentState()
        self.closed = False
        self._log_cursor = 0
        for event_type in self.consumes:
            bus.subscribe(event_type, self)

    def step(self, t: float) -> list[BaseModel]:
        if self.scenario is None or self.bus is None:
            return []
        new_events = self.inbox[self._cursor :]
        self._cursor = len(self.inbox)
        if self.stop_after_closure:
            self._note_closure()
            if self.closed:
                return []
        tick_event: Tick | None = None
        instant_total: dict[str, float] = {}
        saw_tick = False
        for event in new_events:
            if isinstance(event, WithdrawalRequested) and event.amount > 0.0:
                pass  # handled after sim_minute is known
            elif isinstance(event, VIPDecision):
                pass
            elif isinstance(event, Tick):
                saw_tick = True
                tick_event = event
        sim_minute = self._sim_minute(t, tick_event)
        flags = calendar_flags(sim_minute, self.calendar)
        for event in new_events:
            if isinstance(event, WithdrawalRequested) and event.amount > 0.0:
                _, instant = enqueue_withdrawal(
                    self.state,
                    event.amount,
                    event.channel,
                    self.limits,
                    flags,
                )
                for channel, amount in instant:
                    instant_total[channel] = instant_total.get(channel, 0.0) + amount
            elif (
                isinstance(event, VIPDecision)
                and event.action == "withdraw"
                and event.amount > 0.0
            ):
                _, instant = enqueue_withdrawal(
                    self.state,
                    event.amount,
                    self.limits.vip_default_channel,
                    self.limits,
                    flags,
                )
                for channel, amount in instant:
                    instant_total[channel] = instant_total.get(channel, 0.0) + amount
        settlements: dict[str, float] = {}
        if saw_tick:
            settlements = settle_tick(self.state, flags, self.limits)
        for channel, amount in instant_total.items():
            settlements[channel] = settlements.get(channel, 0.0) + amount
        out: list[BaseModel] = []
        queued = len(self.state.wire_queue)
        for channel, amount in sorted(settlements.items()):
            if amount <= 0.0:
                continue
            out.append(
                PaymentSettled(
                    t=t,
                    seed=self.seed,
                    bank_id=self.bank_id,
                    cause_id=self._cause(channel),
                    amount=amount,
                    channel=channel,
                    queued_remaining=queued,
                )
            )
        self.published.extend(out)
        return out

    def _note_closure(self) -> None:
        """Mark this bank closed once its BankClosed is on the bus (read via the log, so
        ``consumes`` and the §6 consumer table stay as they are)."""
        if self.bus is None or self.closed:
            return
        log = self.bus.log
        for event in log[self._log_cursor :]:
            if isinstance(event, BankClosed) and event.bank_id == self.bank_id:
                self.closed = True
                break
        self._log_cursor = len(log)

    @property
    def seed(self) -> int:
        if self.scenario is None:
            raise RuntimeError(f"{self.module_id} is not attached")
        return self.scenario.seed

    @property
    def bank_id(self) -> str:
        if self.scenario is None:
            raise RuntimeError(f"{self.module_id} is not attached")
        return bank_id_for(self.scenario.preset)

    def _cause(self, salt: str) -> str:
        self._seq += 1
        return f"{self.module_id}-{salt}-{self._seq}"

    def _sim_minute(self, step_t: float, tick: Tick | None) -> float:
        """Map engine step index (fake M00) or absolute minutes (real M00)."""
        if tick is not None and tick.t >= self.calendar.epoch_open_minute:
            return tick.t
        return float(self.calendar.epoch_open_minute + int(step_t))
