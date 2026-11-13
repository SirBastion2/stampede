"""M09 securities and interest-rate risk — Wave B."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import ClassVar

from pydantic import BaseModel

from contracts.events import AssetSold, LossRealized, RatesChanged, Tick
from contracts.schemas import Scenario
from mocks.bus import EventBus
from mocks.support import SECURITIES_FRACTION, bank_id_for, deposits_base
from params.loader import get_parameter

# Portfolio split is not in the registry; §5 names AFS and HTM buckets only.
_HTM_BOOK_SHARE = 0.6
_AFS_BOOK_SHARE = 0.4
_MATURITY_BUCKETS: tuple[tuple[str, float], ...] = (
    ("2y", 2.0),
    ("5y", 5.0),
    ("10y", 10.0),
)


@dataclass(frozen=True, slots=True)
class BondBucket:
    """One maturity bucket: book and market values plus rate sensitivity."""

    asset_class: str
    maturity_bucket: str
    book_value: float
    market_value: float
    coupon: float
    duration: float
    convexity: float
    htm: bool


def price_change_fraction(duration: float, convexity: float, delta_y: float) -> float:
    """Fractional bond price change from a parallel yield shift (§18.3)."""
    return (-duration * delta_y) + (0.5 * convexity * delta_y * delta_y)


def mark_buckets(buckets: list[BondBucket], delta_y: float) -> list[BondBucket]:
    """Revalue every bucket after a yield change."""
    if abs(delta_y) < 1e-15:
        return buckets
    out: list[BondBucket] = []
    for bucket in buckets:
        shock = price_change_fraction(bucket.duration, bucket.convexity, delta_y)
        factor = max(0.05, 1.0 + shock)
        out.append(
            replace(bucket, market_value=max(0.0, bucket.market_value * factor))
        )
    return out


def realize_sale(
    buckets: list[BondBucket],
    asset_class: str,
    book_amount: float,
    *,
    htm_reclassify: bool,
) -> tuple[list[BondBucket], float]:
    """Sell ``book_amount`` from ``asset_class``; return updated buckets and loss."""
    if book_amount <= 0.0:
        return buckets, 0.0
    remaining = book_amount
    loss = 0.0
    updated: list[BondBucket] = []
    sold_htm = False
    for bucket in buckets:
        if bucket.asset_class != asset_class or remaining <= 0.0:
            updated.append(bucket)
            continue
        take = min(remaining, bucket.book_value)
        if take <= 0.0:
            updated.append(bucket)
            continue
        if bucket.book_value <= 0.0:
            updated.append(bucket)
            continue
        fraction = take / bucket.book_value
        market_slice = bucket.market_value * fraction
        book_slice = take
        if book_slice > market_slice:
            loss += book_slice - market_slice
        new_book = bucket.book_value - take
        new_market = bucket.market_value - market_slice
        if bucket.htm:
            sold_htm = True
        if new_book <= 1e-9:
            continue
        updated.append(
            replace(
                bucket,
                book_value=new_book,
                market_value=max(0.0, new_market),
            )
        )
        remaining -= take
    if sold_htm and htm_reclassify:
        reclassified: list[BondBucket] = []
        for bucket in updated:
            if bucket.htm:
                reclassified.append(
                    replace(
                        bucket,
                        htm=False,
                        asset_class="afs",
                    )
                )
            else:
                reclassified.append(bucket)
        updated = reclassified
    return updated, loss


def _parse_percent_cell(value: str) -> float:
    text = value.replace("−", "-").replace("about", "").strip().lower()
    match = re.search(r"(-?[0-9]+(?:\.[0-9]+)?)\s*%", text)
    if match is None:
        raise ValueError(f"cannot parse percent from registry value: {value!r}")
    return float(match.group(1)) / 100.0


def expected_six_year_hundred_bp_move() -> float:
    """Registry acceptance row for §18.3 M09 tests."""
    row = get_parameter("Securities test: +100 bp on 6-year duration bond")
    return _parse_percent_cell(row.value)


def initial_buckets(preset: str) -> list[BondBucket]:
    """Build starting bond buckets for a preset (book = market at t=0)."""
    del preset
    deposits = deposits_base()
    total = deposits * SECURITIES_FRACTION
    htm_total = total * _HTM_BOOK_SHARE
    afs_total = total * _AFS_BOOK_SHARE
    buckets: list[BondBucket] = []
    n = len(_MATURITY_BUCKETS)

    def add_pool(pool: float, asset_class: str, htm: bool) -> None:
        per = pool / n
        for label, duration in _MATURITY_BUCKETS:
            convexity = duration * duration * 0.01
            buckets.append(
                BondBucket(
                    asset_class=asset_class,
                    maturity_bucket=label,
                    book_value=per,
                    market_value=per,
                    coupon=0.03,
                    duration=duration,
                    convexity=convexity,
                    htm=htm,
                )
            )

    add_pool(htm_total, "htm", True)
    add_pool(afs_total, "afs", False)
    return buckets


class SecuritiesValuer:
    """Values bond holdings; publishes LossRealized on sales (§19 M09)."""

    consumes: ClassVar[tuple[str, ...]] = ("Tick", "RatesChanged", "AssetSold")
    publishes: ClassVar[tuple[str, ...]] = ("LossRealized",)

    def __init__(self, module_id: str) -> None:
        self.module_id = module_id
        self.inbox: list[BaseModel] = []
        self.scenario: Scenario | None = None
        self.bus: EventBus | None = None
        self._cursor = 0
        self._seq = 0
        self.buckets: list[BondBucket] = []
        self._policy_rate: float | None = None
        self.htm_reclassify_on_sale = True

    def attach(self, bus: EventBus, scenario: Scenario) -> None:
        self.bus = bus
        self.scenario = scenario
        self.buckets = initial_buckets(scenario.preset)
        for event_type in self.consumes:
            bus.subscribe(event_type, self)

    def step(self, t: float) -> list[BaseModel]:
        if self.scenario is None or self.bus is None:
            return []
        new_events = self.inbox[self._cursor :]
        self._cursor = len(self.inbox)
        out: list[BaseModel] = []
        for event in new_events:
            if isinstance(event, RatesChanged):
                prev = self._policy_rate if self._policy_rate is not None else event.policy_rate
                delta_y = event.policy_rate - prev
                self._policy_rate = event.policy_rate
                self.buckets = mark_buckets(self.buckets, delta_y)
            elif isinstance(event, AssetSold):
                self.buckets, loss = realize_sale(
                    self.buckets,
                    event.asset_class,
                    event.amount,
                    htm_reclassify=self.htm_reclassify_on_sale,
                )
                if loss > 1.0:
                    out.append(
                        LossRealized(
                            t=t,
                            seed=self.seed,
                            bank_id=self.bank_id,
                            cause_id=self._cause("loss"),
                            amount=loss,
                            kind="securities",
                        )
                    )
            elif isinstance(event, Tick):
                del event
        return out

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

    def total_unrealized_loss(self) -> float:
        return sum(
            max(0.0, bucket.book_value - bucket.market_value) for bucket in self.buckets
        )
