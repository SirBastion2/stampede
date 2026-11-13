"""L2 fakes: rates, payments, ledger, securities, liquidity ladder."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from contracts.events import (
    AssetSold,
    CapitalChanged,
    DepositRateChanged,
    LiquidityDrawn,
    LossRealized,
    PaymentSettled,
    RatesChanged,
    VIPDecision,
    WithdrawalRequested,
)
from contracts.schemas import Scenario
from engine.liquidity.terms import ASSUMED_TERMS, emergency_terms
from mocks.base import FakeModule
from mocks.bus import EventBus
from mocks.support import (
    ASSUMED_POLICY_RATE,
    DISCOUNT_TICK_FRACTION,
    DISCOUNT_TOTAL_FRACTION,
    FHLB_TICK_FRACTION,
    FHLB_TOTAL_FRACTION,
    OPENING_CASH_FRACTION,
    OPENING_CET1,
    REPO_TICK_FRACTION,
    REPO_TOTAL_FRACTION,
    SAME_TICK_DRAW_SHARE,
    SAME_TICK_SETTLE_FRACTION,
    calendar_at,
    clamp,
    curve_for,
    deposits_base,
    fold_book,
    levers_in,
    policy_rate_at,
    securities_loss_ratio,
)


class RatesFake(FakeModule):
    """M01. Scenario rate path, plus CRE and unemployment carried as state."""

    publishes: ClassVar[tuple[str, ...]] = ("RatesChanged",)

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.policy_rate = ASSUMED_POLICY_RATE
        self.unemployment = 0.04
        self.cre = {"office": 1.0, "multifamily": 1.0, "retail": 1.0}

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        del new_events
        if self.scenario is None:
            return []
        self.policy_rate = policy_rate_at(self.scenario.rate_path, t)
        for shock in self.scenario.shocks:
            if t < shock.at:
                continue
            if shock.kind == "unemployment":
                raw = shock.params.get("rate", self.unemployment)
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    self.unemployment = clamp(float(raw), 0.0, 0.5)
            elif shock.kind == "property":
                factor = shock.params.get("factor", 0.9)
                sector = shock.params.get("sector", "office")
                if (
                    isinstance(factor, (int, float))
                    and not isinstance(factor, bool)
                    and isinstance(sector, str)
                    and sector in self.cre
                ):
                    self.cre[sector] = max(0.0, self.cre[sector] * float(factor))
        rate = clamp(self.policy_rate, 0.0, 0.25)
        self.policy_rate = rate
        return [
            RatesChanged(
                t=t,
                seed=self.seed,
                bank_id=self.bank_id,
                cause_id=self.cause("rates"),
                curve=curve_for(rate),
                policy_rate=rate,
            )
        ]


class PaymentsFake(FakeModule):
    """M07. Settles most of each request now and carries the rest as a wire queue."""

    publishes: ClassVar[tuple[str, ...]] = ("PaymentSettled",)
    consumes: ClassVar[tuple[str, ...]] = ("Tick", "WithdrawalRequested", "VIPDecision")

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.wires: list[float] = []
        self.queued_remaining = 0

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        arrived: list[float] = []
        for event in new_events:
            if isinstance(event, WithdrawalRequested) and event.amount > 0.0:
                arrived.append(event.amount)
            elif (
                isinstance(event, VIPDecision)
                and event.action == "withdraw"
                and event.amount > 0.0
            ):
                arrived.append(event.amount)
        settled = 0.0
        still: list[float] = []
        for remaining in self.wires:
            part = remaining * 0.5
            settled += part
            left = remaining - part
            if left > 1.0:
                still.append(left)
        for amount in arrived:
            part = amount * SAME_TICK_SETTLE_FRACTION
            settled += part
            left = amount - part
            if left > 1.0:
                still.append(left)
        self.wires = still
        self.queued_remaining = len(self.wires)
        if settled <= 1.0:
            return []
        return [
            PaymentSettled(
                t=t,
                seed=self.seed,
                bank_id=self.bank_id,
                cause_id=self.cause("pay"),
                amount=settled,
                channel="wire",
                queued_remaining=self.queued_remaining,
            )
        ]


class LiquidityFake(FakeModule):
    """M12. Same-minute slice of the cash hole; slower rungs and sales cover the rest.

    §19 lists Tick and PolicyEnacted as consumed events. The ladder's other input
    is cash need, which is not its own event; settled payments are that need.
    """

    publishes: ClassVar[tuple[str, ...]] = ("LiquidityDrawn",)
    consumes: ClassVar[tuple[str, ...]] = ("Tick", "PolicyEnacted", "PaymentSettled")

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.terms = ASSUMED_TERMS

    def attach(self, bus: EventBus, scenario: Scenario) -> None:
        super().attach(bus, scenario)
        self.terms = emergency_terms()  # CM-2: fixed at build, inside the live context (like M08's book)
        from engine.bank.ledger import opening_book_fractions
        from engine.system.can_pay import can_pay_fractions

        fr = can_pay_fractions()   # AU-1: the pledged bonds' lendable value and the big banks' deposits
        self._opening_cash = opening_book_fractions()[0]
        self._pledged_fraction = fr.prepared - self._opening_cash - (REPO_TOTAL_FRACTION + FHLB_TOTAL_FRACTION)
        self._big_fraction = fr.big_banks
        self._big_paid = False

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        del new_events
        deposits = deposits_base()
        book = fold_book(self.log(), deposits=deposits, opening_cet1=OPENING_CET1)
        levers = levers_in(self.log())
        line_on = "emergency_lending" in levers
        pledged_on = "pledge_collateral" in levers
        if "big_bank_deposits" in levers and not self._big_paid:   # AU-1: the big banks' deposits arrive at once
            self._big_paid = True
            return [LiquidityDrawn(t=t, seed=self.seed, bank_id=self.bank_id, source="big-bank-deposits",
                                   amount=self._big_fraction * deposits, eta=0.0)]
        terms = self.terms
        cash = book.cash
        if pledged_on and not (line_on and terms.opening_cash_fraction is not None):
            cash += deposits * (self._opening_cash - OPENING_CASH_FRACTION)   # AU-1: what M08 really has
        if line_on and terms.opening_cash_fraction is not None:
            # CM-2: on BTFP terms the line pays what the bank really lacks: the folded book's postings
            # (payments out, draws and sale proceeds in) from M08's own opening cash, not the 0.1% slice.
            cash += deposits * (terms.opening_cash_fraction - OPENING_CASH_FRACTION)
        hole = max(0.0, -cash)
        if hole <= 1.0:
            return []
        pools: list[tuple[str, float, float, float, bool]] = [
            ("repo", REPO_TOTAL_FRACTION, REPO_TICK_FRACTION, 0.0, False),
            ("fhlb", FHLB_TOTAL_FRACTION, FHLB_TICK_FRACTION, 1.0, True),
            ("discount_window", DISCOUNT_TOTAL_FRACTION, DISCOUNT_TICK_FRACTION, 2.0, True),
        ]
        if pledged_on:   # AU-1: the discount window lends against the pledged bonds at any hour, first in line
            pools = [p for p in pools if p[0] != "discount_window"]
            pools.insert(0, ("discount_window", self._pledged_fraction, terms.tick_fraction if terms.btfp else 0.01, 0.25, False))
        if line_on:
            pools.insert(0, ("emergency", terms.total_fraction, terms.tick_fraction, 0.25, False))
        drawn = _drawn_by_source(self.log())
        flags = calendar_at(t)
        share = terms.shortfall_share if line_on else (1.0 if pledged_on else SAME_TICK_DRAW_SHARE)
        for source, total_frac, tick_frac, eta, hours_only in pools:
            if hours_only and not flags.facility_open:
                continue
            left = deposits * total_frac - drawn.get(source, 0.0)
            take = min(left, deposits * tick_frac, hole * share)
            if take <= 1.0:
                continue
            return [
                LiquidityDrawn(
                    t=t,
                    seed=self.seed,
                    bank_id=self.bank_id,
                    cause_id=self.cause(source),
                    source=source,
                    amount=take,
                    eta=eta,
                )
            ]
        return []


class LedgerFake(FakeModule):
    """M08. Posts payments, draws, and sales; capital moves when a loss is in the log."""

    publishes: ClassVar[tuple[str, ...]] = ("AssetSold", "CapitalChanged")
    consumes: ClassVar[tuple[str, ...]] = (
        "Tick",
        "PaymentSettled",
        "LiquidityDrawn",
        "LossRealized",
        "DepositRateChanged",
        "RatesChanged",
    )

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.last_cet1 = OPENING_CET1
        self.deposit_rate: dict[str, float] = {}
        self.cash = 0.0
        self.securities = 0.0
        self.unpaid = 0.0

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        for event in new_events:
            if isinstance(event, DepositRateChanged):
                self.deposit_rate[event.segment] = event.rate
        book = fold_book(self.log(), deposits=deposits_base(), opening_cet1=OPENING_CET1)
        self.cash = book.cash
        self.securities = book.securities
        self.unpaid = book.unpaid
        out: list[BaseModel] = []
        cet1 = clamp(book.cet1, 0.0, 1.0)
        if cet1 + 1e-12 < self.last_cet1:
            out.append(
                CapitalChanged(
                    t=t,
                    seed=self.seed,
                    bank_id=self.bank_id,
                    cause_id=self.cause("cet1"),
                    cet1_ratio=cet1,
                )
            )
            self.last_cet1 = cet1
        if book.cash < -1.0 and book.securities > 1.0:
            ratio = securities_loss_ratio(book.policy_rate)
            proceed = max(0.05, 1.0 - ratio)
            book_sold = min(book.securities, (-book.cash) / proceed)
            if book_sold > 1.0:
                out.append(
                    AssetSold.model_validate(
                        {
                            "t": t,
                            "seed": self.seed,
                            "bank_id": self.bank_id,
                            "cause_id": self.cause("sale"),
                            "class": "afs",
                            "amount": book_sold,
                            "price_impact": ratio,
                        }
                    )
                )
        return out


class SecuritiesFake(FakeModule):
    """M09. Sales turn the rate mark into a realized securities loss."""

    publishes: ClassVar[tuple[str, ...]] = ("LossRealized",)
    consumes: ClassVar[tuple[str, ...]] = ("Tick", "RatesChanged", "AssetSold")

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.policy_rate = ASSUMED_POLICY_RATE
        self.unrealized = 0.0

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        for event in new_events:
            if isinstance(event, RatesChanged):
                self.policy_rate = event.policy_rate
        book = fold_book(self.log(), deposits=deposits_base(), opening_cet1=OPENING_CET1)
        self.unrealized = book.securities * securities_loss_ratio(self.policy_rate)
        out: list[BaseModel] = []
        for event in new_events:
            if not isinstance(event, AssetSold):
                continue
            loss = event.amount * clamp(event.price_impact, 0.0, 1.0)
            if loss <= 1.0:
                continue
            out.append(
                LossRealized(
                    t=t,
                    seed=self.seed,
                    bank_id=self.bank_id,
                    cause_id=self.cause("loss"),
                    amount=loss,
                    kind="securities",
                )
            )
        return out


def _drawn_step(totals: dict[str, float], event: BaseModel) -> None:
    if isinstance(event, LiquidityDrawn):
        totals[event.source] = totals.get(event.source, 0.0) + event.amount


def _drawn_by_source(events: list[BaseModel]) -> dict[str, float]:
    from mocks.support import resume_fold

    return dict(resume_fold(events, ("drawn",), dict, _drawn_step))


def l2_modules() -> dict[str, FakeModule]:
    return {
        "M01": RatesFake("M01"),
        "M07": PaymentsFake("M07"),
        "M08": LedgerFake("M08"),
        "M09": SecuritiesFake("M09"),
        "M12": LiquidityFake("M12"),
    }
