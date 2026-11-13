"""M14 per-bank order book and traders — Wave C real mode.

Price-time priority matching is compiled with Numba. Each trader type submits
orders every tick from public signals. A halt fires when the print moves more
than X% within Y minutes. X, Y, and the halt duration are ASSUMED registry rows
because §18.4 leaves them unspecified (ADR-023). The TBD rows are not read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numba import njit
from numpy.typing import NDArray
from pydantic import BaseModel

from contracts.events import (
    AnalystNote,
    AttentionTick,
    CapitalChanged,
    HeadlinePublished,
    PriceTick,
    RatesChanged,
    ThesisPublished,
    TradingHalt,
)
from contracts.schemas import Scenario
from mocks.bus import EventBus
from mocks.support import bank_id_for
from params.loader import RegistryError, RegistryRow, load_registry

# §18.4 lists the halt rule as "X% within Y minutes" and the registry still has
# TBD rows for those names. Real mode reads the ASSUMED rows, not the TBD ones.
_HALT_X = "M14 halt price move threshold (X%)"
_HALT_Y = "M14 halt window (Y minutes)"
_HALT_DURATION = "M14 trading halt minutes"
_OPENING_PRICE = "M14 opening price"
_OPENING_CET1 = "M14 opening CET1"
_OPENING_RATE = "M14 opening policy rate"
_BOND_DURATION = "M14 public bond duration"
_MM_BASE = "M14 market-maker base half-spread"
_MM_MULT = "M14 market-maker volatility multiplier"
_BORROW_LIMIT = "M14 short borrow limit per trader"
_BORROW_COST = "M14 short borrow cost"
_MM_SIZE = "M14 quote size per market maker"
_AGGRESSIVE_SIZE = "M14 aggressive order size"
_ETA = "M14 reservation adjustment per minute"
_HEADLINE_SCALE = "M14 headline reach scale"
_NOTE_WEIGHT = "M14 analyst note attention weight"
_MOMENTUM_SCALE = "M14 momentum return scale"
_STARTING_CASH = "M14 trader starting cash"
_TRADER_COUNT = "Stock traders per bank"
_ATTENTION_PREMIUM = "High pre-run Twitter exposure stock loss premium"

_ASSUMED_ROWS = (
    _HALT_X,
    _HALT_Y,
    _HALT_DURATION,
    _OPENING_PRICE,
    _OPENING_CET1,
    _OPENING_RATE,
    _BOND_DURATION,
    _MM_BASE,
    _MM_MULT,
    _BORROW_LIMIT,
    _BORROW_COST,
    _MM_SIZE,
    _AGGRESSIVE_SIZE,
    _ETA,
    _HEADLINE_SCALE,
    _NOTE_WEIGHT,
    _MOMENTUM_SCALE,
    _STARTING_CASH,
)

_NUMBER = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")

KIND_MM = 0
KIND_MOMENTUM = 1
KIND_VALUE = 2
KIND_ATTENTION = 3
KIND_SHORT = 4
_KIND_COUNT = 5

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class BookParams:
    """Numbers the order book reads from ``params/registry.csv``."""

    halt_move: float
    halt_window_min: float
    halt_duration_min: float
    opening_price: float
    opening_cet1: float
    opening_policy_rate: float
    bond_duration: float
    mm_base_half_spread: float
    mm_vol_multiplier: float
    short_borrow_limit: float
    short_borrow_cost: float
    mm_quote_size: float
    aggressive_size: float
    reservation_eta: float
    headline_reach_scale: float
    analyst_note_weight: float
    momentum_return_scale: float
    starting_cash: float
    trader_count: int
    attention_premium: float


def _parse_float(row: RegistryRow) -> float:
    match = _NUMBER.search(row.value.replace(",", ""))
    if match is None:
        raise RegistryError(f"{row.parameter} has no number: {row.value!r}")
    value = float(match.group(0))
    if "percent" in row.unit.lower():
        return value / 100.0
    return value


def _required(rows: dict[str, RegistryRow], name: str) -> RegistryRow:
    try:
        return rows[name]
    except KeyError as exc:
        raise RegistryError(f"unknown parameter: {name}") from exc


def load_book_params() -> BookParams:
    """Load M14 inputs. Halt X%/Y minutes must be marked ASSUMED."""
    rows = {row.parameter: row for row in load_registry()}
    assumed: dict[str, float] = {}
    for name in _ASSUMED_ROWS:
        row = _required(rows, name)
        # The halt length row is SOURCED (SVB-27f); the others are ASSUMED.
        if row.status not in ({"ASSUMED", "SOURCED"} if name == _HALT_DURATION else {"ASSUMED"}):
            raise RegistryError(
                f"{name} must be marked ASSUMED (status is {row.status!r})"
            )
        assumed[name] = _parse_float(row)
    traders = _parse_float(_required(rows, _TRADER_COUNT))
    trader_count = int(round(traders))
    if trader_count < _KIND_COUNT:
        raise RegistryError(f"{_TRADER_COUNT} must cover the five trader types")
    return BookParams(
        halt_move=assumed[_HALT_X],
        halt_window_min=assumed[_HALT_Y],
        halt_duration_min=assumed[_HALT_DURATION],
        opening_price=assumed[_OPENING_PRICE],
        opening_cet1=assumed[_OPENING_CET1],
        opening_policy_rate=assumed[_OPENING_RATE],
        bond_duration=assumed[_BOND_DURATION],
        mm_base_half_spread=assumed[_MM_BASE],
        mm_vol_multiplier=assumed[_MM_MULT],
        short_borrow_limit=assumed[_BORROW_LIMIT],
        short_borrow_cost=assumed[_BORROW_COST],
        mm_quote_size=assumed[_MM_SIZE],
        aggressive_size=assumed[_AGGRESSIVE_SIZE],
        reservation_eta=assumed[_ETA],
        headline_reach_scale=assumed[_HEADLINE_SCALE],
        analyst_note_weight=assumed[_NOTE_WEIGHT],
        momentum_return_scale=assumed[_MOMENTUM_SCALE],
        starting_cash=assumed[_STARTING_CASH],
        trader_count=trader_count,
        attention_premium=_parse_float(_required(rows, _ATTENTION_PREMIUM)),
    )


def trader_kinds(trader_count: int) -> IntArray:
    """Split traders evenly across the five §5 types. Remainder goes to the front."""
    kinds = np.empty(trader_count, dtype=np.int64)
    base, rem = divmod(trader_count, _KIND_COUNT)
    start = 0
    for kind in range(_KIND_COUNT):
        count = base + (1 if kind < rem else 0)
        kinds[start : start + count] = kind
        start += count
    return kinds


def market_maker_half_spread(volatility: float, base: float, multiplier: float) -> float:
    """Half-spread in price points. Rises linearly with trailing absolute return."""
    if volatility < 0.0:
        volatility = 0.0
    return base * (1.0 + multiplier * volatility)


def public_fair(
    opening_price: float,
    opening_cet1: float,
    cet1: float,
    opening_rate: float,
    policy_rate: float,
    duration: float,
    attention: float,
    delta_4h: float,
    news: float,
    premium: float,
) -> float:
    """Value from public capital, rate-driven unrealized losses, and attention.

    Unrealized loss fraction is duration × max(Δpolicy rate, 0). Attention uses
    the level, a positive 4-hour change, and headline/note pressure. The E10
    premium scales that pressure: attention predicts losses, not gains.
    """
    if opening_cet1 <= 0.0:
        capital_ratio = 1.0
    else:
        capital_ratio = cet1 / opening_cet1
    capital_ratio = min(2.0, max(0.05, capital_ratio))
    unrealized = duration * max(0.0, policy_rate - opening_rate)
    unrealized = min(0.9, max(0.0, unrealized))
    solvency = max(0.05, capital_ratio * (1.0 - unrealized))
    pressure = max(0.0, attention) + max(0.0, delta_4h) + max(0.0, news)
    drag = min(0.9, max(0.0, premium * pressure))
    return max(0.01, opening_price * solvency * (1.0 - drag))


def halt_triggered(
    times: FloatArray,
    prices: FloatArray,
    t: float,
    move_frac: float,
    window_min: float,
) -> bool:
    """True when the last print has moved more than ``move_frac`` within ``window_min``.

    The reference is the last price at or before ``t - window_min``. The move is
    two-sided: §18.4 says the price "moves", up or down.
    """
    n = int(times.shape[0])
    if n == 0 or move_frac < 0.0:
        return False
    cutoff = t - window_min
    ref = float(prices[0])
    for i in range(n):
        if float(times[i]) <= cutoff:
            ref = float(prices[i])
        else:
            break
    current = float(prices[n - 1])
    if ref <= 0.0:
        return False
    return abs(current / ref - 1.0) > move_frac


def trailing_volatility(prices: FloatArray, window: int) -> float:
    """Mean absolute simple return over the last ``window`` steps."""
    n = int(prices.shape[0])
    if n < 2 or window < 1:
        return 0.0
    start = max(1, n - window)
    total = 0.0
    count = 0
    for i in range(start, n):
        prev = float(prices[i - 1])
        if prev <= 0.0:
            continue
        total += abs(float(prices[i]) / prev - 1.0)
        count += 1
    if count == 0:
        return 0.0
    return total / count


def _match_price_time_py(
    side: IntArray,
    price: FloatArray,
    quantity: FloatArray,
    time_key: FloatArray,
) -> tuple[FloatArray, FloatArray, float, float]:
    """Price-time priority match. Earlier order sets the trade price.

    Bids rank by higher price, then earlier time. Asks rank by lower price, then
    earlier time. Equal price and time keep the lower index.
    """
    n = int(side.shape[0])
    filled = np.zeros(n, dtype=np.float64)
    notional = np.zeros(n, dtype=np.float64)
    if n == 0:
        return filled, notional, 0.0, 0.0
    qty = quantity.copy()
    vwap_notional = 0.0
    volume = 0.0
    guard = 0
    limit = n * n + 1
    while guard < limit:
        guard += 1
        best_b = -1
        best_a = -1
        for i in range(n):
            if qty[i] <= 0.0:
                continue
            if side[i] > 0:
                if best_b < 0 or price[i] > price[best_b] or (
                    price[i] == price[best_b] and time_key[i] < time_key[best_b]
                ):
                    best_b = i
            elif side[i] < 0 and (
                best_a < 0
                or price[i] < price[best_a]
                or (price[i] == price[best_a] and time_key[i] < time_key[best_a])
            ):
                best_a = i
        if best_b < 0 or best_a < 0 or price[best_b] < price[best_a]:
            break
        trade_qty = qty[best_b] if qty[best_b] < qty[best_a] else qty[best_a]
        if trade_qty <= 0.0:
            break
        trade_px = price[best_b] if time_key[best_b] <= time_key[best_a] else price[best_a]
        qty[best_b] -= trade_qty
        qty[best_a] -= trade_qty
        filled[best_b] += trade_qty
        filled[best_a] += trade_qty
        notional[best_b] += trade_qty * trade_px
        notional[best_a] += trade_qty * trade_px
        volume += trade_qty
        vwap_notional += trade_qty * trade_px
    vwap = vwap_notional / volume if volume > 0.0 else 0.0
    return filled, notional, vwap, volume


_match_compiled = njit(cache=True)(_match_price_time_py)


def match_price_time(
    side: IntArray,
    price: FloatArray,
    quantity: FloatArray,
    time_key: FloatArray,
) -> tuple[FloatArray, FloatArray, float, float]:
    """Numba-compiled price-time match. Returns fills, notionals, vwap, and volume."""
    filled, notional, vwap, volume = _match_compiled(side, price, quantity, time_key)
    filled_arr = np.asarray(filled, dtype=np.float64)
    notional_arr = np.asarray(notional, dtype=np.float64)
    return filled_arr, notional_arr, float(vwap), float(volume)


class OrderBook:
    """One limit order book and its trader agents (§19 M14)."""

    consumes: ClassVar[tuple[str, ...]] = (
        "Tick",
        "RatesChanged",
        "AttentionTick",
        "HeadlinePublished",
        "AnalystNote",
        "CapitalChanged",
        "ThesisPublished",
    )
    publishes: ClassVar[tuple[str, ...]] = ("PriceTick", "TradingHalt")

    def __init__(self, module_id: str) -> None:
        self.module_id = module_id
        self.inbox: list[BaseModel] = []
        self.scenario: Scenario | None = None
        self.bus: EventBus | None = None
        self._cursor = 0
        self._seq = 0
        self.params = load_book_params()
        self.kinds = trader_kinds(self.params.trader_count)
        n = self.params.trader_count
        self.inventory = np.zeros(n, dtype=np.float64)
        self.cash = np.full(n, self.params.starting_cash, dtype=np.float64)
        self.price = self.params.opening_price
        self.half_spread = self.params.mm_base_half_spread
        self.attention = 0.0
        self.delta_4h = 0.0
        self.news = 0.0
        self.thesis = 0.0
        self.cet1 = self.params.opening_cet1
        self.policy_rate = self.params.opening_policy_rate
        self.curve: tuple[float, ...] = ()
        self.history: list[tuple[float, float]] = []
        self.halt_until: float | None = None
        self._seeded = False

    def attach(self, bus: EventBus, scenario: Scenario) -> None:
        self.bus = bus
        self.scenario = scenario
        for event_type in self.consumes:
            bus.subscribe(event_type, self)

    def step(self, t: float) -> list[BaseModel]:
        if self.scenario is None or self.bus is None:
            return []
        new_events = self.inbox[self._cursor :]
        self._cursor = len(self.inbox)
        for event in new_events:
            self._absorb(event)
        if not self._seeded:
            # Opening print sits one halt-window before the first tick so a
            # first-tick jump counts as a move within Y minutes.
            self.history.append(
                (t - self.params.halt_window_min, self.params.opening_price)
            )
            self._seeded = True
        if self.halt_until is not None and t < self.halt_until:
            self.history.append((t, self.price))
            held: list[BaseModel] = []
            held.append(self._price_tick(t, self.price, 0.0))
            return held

        fair = self._fair()
        prices = np.asarray([px for _, px in self.history], dtype=np.float64)
        window = max(1, int(round(self.params.halt_window_min)))
        volatility = trailing_volatility(prices, window)
        half = market_maker_half_spread(
            volatility,
            self.params.mm_base_half_spread,
            self.params.mm_vol_multiplier,
        )
        self.half_spread = half
        center = self.price + self.params.reservation_eta * (fair - self.price)
        side, limit, size, time_key, owners = self._orders(t, fair, center, half)
        volume = 0.0
        new_price = self.price
        if side.shape[0] > 0:
            filled, notional, vwap, volume = match_price_time(side, limit, size, time_key)
            self._apply_fills(owners, side, filled, notional)
            if volume > 0.0:
                new_price = vwap
        new_price = max(0.01, new_price)
        times = np.asarray([ts for ts, _ in self.history] + [t], dtype=np.float64)
        path = np.asarray([px for _, px in self.history] + [new_price], dtype=np.float64)
        tripped = halt_triggered(
            times,
            path,
            t,
            self.params.halt_move,
            self.params.halt_window_min,
        )
        self.price = new_price
        self.history.append((t, new_price))
        out: list[BaseModel] = [self._price_tick(t, new_price, volume)]
        if tripped:
            self.halt_until = t + self.params.halt_duration_min
            out.append(
                TradingHalt(
                    t=t,
                    seed=self.seed,
                    bank_id=self.bank_id,
                    cause_id=self._cause("halt"),
                    reason="volatility",
                    duration=self.params.halt_duration_min,
                )
            )
        return out

    def _fair(self) -> float:
        return public_fair(
            self.params.opening_price,
            self.params.opening_cet1,
            self.cet1,
            self.params.opening_policy_rate,
            self.policy_rate,
            self.params.bond_duration,
            self.attention,
            self.delta_4h,
            self.news,
            self.params.attention_premium,
        )

    def _orders(
        self,
        t: float,
        fair: float,
        center: float,
        half: float,
    ) -> tuple[IntArray, FloatArray, FloatArray, FloatArray, IntArray]:
        mm_bid = max(0.01, center - half)
        mm_ask = max(mm_bid + 0.01, center + half)
        prev = self.history[-1][1] if self.history else self.price
        ret = (self.price / prev - 1.0) if prev > 0.0 else 0.0
        sides: list[int] = []
        limits: list[float] = []
        sizes: list[float] = []
        times: list[float] = []
        owners: list[int] = []

        def add(tid: int, order_side: int, limit_px: float, qty: float, when: float) -> None:
            if qty <= 0.0 or limit_px <= 0.0:
                return
            if order_side > 0:
                affordable = self.cash[tid] / limit_px
                qty = min(qty, affordable)
                if qty <= 1e-9:
                    return
            sides.append(order_side)
            limits.append(limit_px)
            sizes.append(qty)
            times.append(when)
            owners.append(tid)

        pressure = max(0.0, self.attention) + max(0.0, self.delta_4h) + max(0.0, self.news)
        scale = self.params.momentum_return_scale
        for tid in range(int(self.kinds.shape[0])):
            kind = int(self.kinds[tid])
            if kind == KIND_MM:
                when = t + tid * 1e-9
                add(tid, 1, mm_bid, self.params.mm_quote_size, when)
                add(tid, -1, mm_ask, self.params.mm_quote_size, when)
                continue
            when = t + 0.001 + tid * 1e-9
            if kind == KIND_MOMENTUM:
                if scale <= 0.0 or ret == 0.0:
                    continue
                qty = self.params.aggressive_size * min(1.0, abs(ret) / scale)
                if ret > 0.0:
                    add(tid, 1, mm_ask, qty, when)
                else:
                    add(tid, -1, mm_bid, qty, when)
            elif kind == KIND_VALUE:
                if fair < self.price - half:
                    add(tid, -1, min(fair, mm_bid), self.params.aggressive_size, when)
                elif fair > self.price + half:
                    add(tid, 1, max(fair, mm_ask), self.params.aggressive_size, when)
            elif kind == KIND_ATTENTION:
                if pressure <= 0.0:
                    continue
                qty = self.params.aggressive_size * min(1.0, pressure)
                add(tid, -1, mm_bid, qty, when)
            elif kind == KIND_SHORT:
                room = self.params.short_borrow_limit + float(self.inventory[tid])
                if room <= 1e-9 or center <= 0.0:
                    continue
                downside = max(0.0, (center - fair) / center)
                if self.thesis <= 0.0 and downside <= self.params.short_borrow_cost:
                    continue
                intensity = max(self.thesis, downside)
                haircut = max(0.0, 1.0 - self.params.short_borrow_cost)
                qty = min(room, self.params.aggressive_size * min(1.0, intensity) * haircut)
                add(tid, -1, mm_bid, qty, when)
        return (
            np.asarray(sides, dtype=np.int64),
            np.asarray(limits, dtype=np.float64),
            np.asarray(sizes, dtype=np.float64),
            np.asarray(times, dtype=np.float64),
            np.asarray(owners, dtype=np.int64),
        )

    def _apply_fills(
        self,
        owners: IntArray,
        side: IntArray,
        filled: FloatArray,
        notional: FloatArray,
    ) -> None:
        for i in range(int(filled.shape[0])):
            qty = float(filled[i])
            if qty <= 0.0:
                continue
            tid = int(owners[i])
            sign = float(side[i])
            self.inventory[tid] += sign * qty
            self.cash[tid] -= sign * float(notional[i])

    def _absorb(self, event: object) -> None:
        bank = getattr(event, "bank_id", None)
        if bank is not None and bank != self.bank_id:
            return
        if isinstance(event, AttentionTick):
            self.attention = event.level
            self.delta_4h = event.delta_4h
        elif isinstance(event, CapitalChanged):
            self.cet1 = event.cet1_ratio
        elif isinstance(event, RatesChanged):
            self.policy_rate = event.policy_rate
            self.curve = tuple(event.curve)
        elif isinstance(event, HeadlinePublished):
            scale = self.params.headline_reach_scale
            if scale > 0.0:
                self.news = min(1.0, max(self.news, event.reach / scale))
        elif isinstance(event, AnalystNote):
            cites = len(event.cited_event_ids)
            bump = self.params.analyst_note_weight * float(max(1, cites))
            self.news = min(1.0, self.news + bump)
        elif isinstance(event, ThesisPublished):
            scale = self.params.headline_reach_scale
            if scale > 0.0:
                self.thesis = min(1.0, max(self.thesis, event.reach / scale))

    def _price_tick(self, t: float, price: float, volume: float) -> PriceTick:
        return PriceTick(
            t=t,
            seed=self.seed,
            bank_id=self.bank_id,
            cause_id=self._cause("px"),
            price=price,
            volume=volume,
        )

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
