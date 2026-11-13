"""M12 liquidity and funding facilities — Wave C."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from pydantic import BaseModel

from contracts.events import LiquidityDrawn, PaymentSettled, PolicyEnacted, Tick
from contracts.schemas import Scenario
from engine.bank.ledger import registry_deposits_total
from engine.core.clock import CalendarConfig, calendar_flags, load_calendar_config
from mocks.bus import EventBus
from mocks.support import (
    DISCOUNT_TICK_FRACTION,
    DISCOUNT_TOTAL_FRACTION,
    FHLB_TICK_FRACTION,
    FHLB_TOTAL_FRACTION,
    OPENING_CASH_FRACTION,
    REPO_TICK_FRACTION,
    REPO_TOTAL_FRACTION,
    SAME_TICK_DRAW_SHARE,
    bank_id_for,
    canonical_lever,
)

_BROKERED_TOTAL_FRACTION = 0.006
_BROKERED_TICK_FRACTION = 0.0005
_EMERGENCY_TOTAL_FRACTION = 0.05
_EMERGENCY_TICK_FRACTION = 0.01
_REPO_HAIRCUT = 0.02
_FHLB_HAIRCUT = 0.05
_DISCOUNT_HAIRCUT = 0.0
_BROKERED_HAIRCUT = 0.0
_SECURITIES_FRACTION = 0.004


@dataclass(frozen=True, slots=True)
class FundingSource:
    """One rung on the liquidity ladder (§5 M12)."""

    name: str
    rank: int
    total_capacity_fraction: float
    tick_capacity_fraction: float
    eta_hours: float
    hours_only: bool
    haircut: float
    collateral_pool: str | None
    stigma: float
    cost_spread: float


@dataclass
class CollateralLedger:
    """Tracks pledged collateral so nothing is pledged twice (§5 M12)."""

    pools: dict[str, float] = field(default_factory=dict)
    pledged: dict[str, float] = field(default_factory=dict)

    def available(self, pool: str) -> float:
        return max(0.0, self.pools.get(pool, 0.0) - self.pledged.get(pool, 0.0))

    def pledge(self, pool: str, face_amount: float) -> None:
        if face_amount <= 0.0:
            return
        self.pledged[pool] = self.pledged.get(pool, 0.0) + face_amount

    def max_draw_for_pool(self, pool: str, haircut: float) -> float:
        face = self.available(pool)
        if face <= 0.0:
            return 0.0
        return face * (1.0 - haircut)


def default_funding_sources(deposits: float) -> list[FundingSource]:
    """Ordered ladder rungs excluding cash on hand and M09 asset sales."""
    del deposits
    return [
        FundingSource(
            name="repo",
            rank=10,
            total_capacity_fraction=REPO_TOTAL_FRACTION,
            tick_capacity_fraction=REPO_TICK_FRACTION,
            eta_hours=0.0,
            hours_only=False,
            haircut=_REPO_HAIRCUT,
            collateral_pool="securities",
            stigma=0.05,
            cost_spread=0.0025,
        ),
        FundingSource(
            name="fhlb",
            rank=20,
            total_capacity_fraction=FHLB_TOTAL_FRACTION,
            tick_capacity_fraction=FHLB_TICK_FRACTION,
            eta_hours=1.0,
            hours_only=True,
            haircut=_FHLB_HAIRCUT,
            collateral_pool="securities",
            stigma=0.08,
            cost_spread=0.004,
        ),
        FundingSource(
            name="discount_window",
            rank=30,
            total_capacity_fraction=DISCOUNT_TOTAL_FRACTION,
            tick_capacity_fraction=DISCOUNT_TICK_FRACTION,
            eta_hours=2.0,
            hours_only=True,
            haircut=_DISCOUNT_HAIRCUT,
            collateral_pool="securities",
            stigma=0.15,
            cost_spread=0.01,
        ),
        FundingSource(
            name="emergency",
            rank=15,
            total_capacity_fraction=_EMERGENCY_TOTAL_FRACTION,
            tick_capacity_fraction=_EMERGENCY_TICK_FRACTION,
            eta_hours=0.25,
            hours_only=False,
            haircut=0.0,
            collateral_pool=None,
            stigma=0.02,
            cost_spread=0.0,
        ),
        FundingSource(
            name="brokered_deposits",
            rank=40,
            total_capacity_fraction=_BROKERED_TOTAL_FRACTION,
            tick_capacity_fraction=_BROKERED_TICK_FRACTION,
            eta_hours=4.0,
            hours_only=False,
            haircut=_BROKERED_HAIRCUT,
            collateral_pool=None,
            stigma=0.12,
            cost_spread=0.02,
        ),
    ]


def _opening_cash(deposits: float) -> float:
    return deposits * OPENING_CASH_FRACTION


def cash_shortfall(
    *,
    deposits: float,
    payments: float,
    draws: float,
) -> float:
    """Cash hole after on-hand cash, payments, and prior draws (§5 rung 1)."""
    cash = _opening_cash(deposits) - payments + draws
    return max(0.0, -cash)


def _drawn_totals(events: list[BaseModel]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for event in events:
        if isinstance(event, LiquidityDrawn):
            totals[event.source] = totals.get(event.source, 0.0) + event.amount
    return totals


def _payments_total(events: list[BaseModel]) -> float:
    return sum(
        event.amount
        for event in events
        if isinstance(event, PaymentSettled) and event.amount > 0.0
    )


def active_emergency_levers(events: list[BaseModel]) -> bool:
    for event in events:
        if isinstance(event, PolicyEnacted):
            if canonical_lever(event.lever) == "emergency_lending":
                return True
    return False


def facility_open_at(
    engine_t: float,
    *,
    tick_sim_minute: float | None,
    cfg: CalendarConfig,
) -> bool:
    """Facility hours from M00 calendar params; fake ticks use engine index."""
    if tick_sim_minute is not None and tick_sim_minute >= cfg.epoch_open_minute - 1.0:
        return calendar_flags(tick_sim_minute, cfg).facility_open
    absolute = cfg.epoch_open_minute + int(engine_t)
    day = absolute // (24 * 60)
    minute = absolute % (24 * 60)
    weekend = (day % 7) >= 5
    if weekend:
        return False
    return cfg.facility_open_minute <= minute < cfg.facility_close_minute


def next_liquidity_draw(
    *,
    cash_need: float,
    deposits: float,
    sources: list[FundingSource],
    drawn_by_source: dict[str, float],
    collateral: CollateralLedger,
    facility_open: bool,
    emergency_active: bool,
    discount_pre_pledged: bool,
    same_tick_share: float = SAME_TICK_DRAW_SHARE,
) -> tuple[str, float, float] | None:
    """Pick the fastest feasible source and return (source, amount, eta)."""
    if cash_need <= 1.0:
        return None
    share = 0.8 if emergency_active else same_tick_share

    def _rank(source: FundingSource) -> int:
        if source.name == "emergency" and emergency_active:
            return 5
        return source.rank

    ordered = sorted(sources, key=_rank)
    for source in ordered:
        if source.name == "emergency" and not emergency_active:
            continue
        if source.hours_only and not facility_open:
            continue
        if source.name == "discount_window" and not discount_pre_pledged:
            continue
        capacity_left = deposits * source.total_capacity_fraction - drawn_by_source.get(
            source.name, 0.0
        )
        tick_cap = deposits * source.tick_capacity_fraction
        need_slice = cash_need * share
        amount = min(capacity_left, tick_cap, need_slice)
        if source.collateral_pool is not None:
            pool_cap = collateral.max_draw_for_pool(
                source.collateral_pool, source.haircut
            )
            amount = min(amount, pool_cap)
        if amount <= 1.0:
            continue
        if source.collateral_pool is not None and source.haircut < 1.0:
            face = amount / max(1e-9, 1.0 - source.haircut)
            collateral.pledge(source.collateral_pool, face)
        return (source.name, amount, source.eta_hours)
    return None


class LiquidityLadderModule:
    """M12 liquidity-ladder allocator (§19)."""

    publishes: ClassVar[tuple[str, ...]] = ("LiquidityDrawn",)
    consumes: ClassVar[tuple[str, ...]] = ("Tick", "PolicyEnacted", "PaymentSettled")

    def __init__(self, module_id: str) -> None:
        self.module_id = module_id
        self.inbox: list[BaseModel] = []
        self.scenario: Scenario | None = None
        self.bus: EventBus | None = None
        self._cursor = 0
        self._seq = 0
        self.discount_pre_pledged = False
        self._calendar = load_calendar_config()
        self._collateral = CollateralLedger()
        self._last_tick_minute: float | None = None

    def attach(self, bus: EventBus, scenario: Scenario) -> None:
        self.bus = bus
        self.scenario = scenario
        deposits = registry_deposits_total()
        self._collateral.pools["securities"] = deposits * _SECURITIES_FRACTION
        for event_type in self.consumes:
            bus.subscribe(event_type, self)

    def step(self, t: float) -> list[BaseModel]:
        if self.scenario is None or self.bus is None:
            return []
        new_events = self.inbox[self._cursor :]
        self._cursor = len(self.inbox)
        for event in new_events:
            if isinstance(event, Tick):
                self._last_tick_minute = event.t
        log = self.bus.log
        deposits = registry_deposits_total()
        payments = _payments_total(log)
        drawn = _drawn_totals(log)
        draws_sum = sum(drawn.values())
        hole = cash_shortfall(deposits=deposits, payments=payments, draws=draws_sum)
        if hole <= 1.0:
            return []
        facility = facility_open_at(
            t,
            tick_sim_minute=self._last_tick_minute,
            cfg=self._calendar,
        )
        emergency = active_emergency_levers(log)
        sources = default_funding_sources(deposits)
        planned = next_liquidity_draw(
            cash_need=hole,
            deposits=deposits,
            sources=sources,
            drawn_by_source=drawn,
            collateral=self._collateral,
            facility_open=facility,
            emergency_active=emergency,
            discount_pre_pledged=self.discount_pre_pledged,
        )
        if planned is None:
            return []
        source, amount, eta = planned
        return [
            LiquidityDrawn(
                t=t,
                seed=self.seed,
                bank_id=self.bank_id,
                cause_id=self._cause(source),
                source=source,
                amount=amount,
                eta=eta,
            )
        ]

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
